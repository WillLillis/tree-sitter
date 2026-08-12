# The IR: a query VM

**Status: design sketch for discussion. Nothing measured, nothing built.**

The IR is a **deliverable in its own right**, not only scaffolding for the matcher rewrite.
Tooling stays the last priority, but designing for it improves the internal APIs too — the
discipline that makes a representation consumable from outside is the same one that makes it
reasonable to maintain from inside.

## The actual problem being solved

Not speed. **Locality of behaviour.**

> Fix a bug in one place, and somewhere else it breaks.

That is a property of the representation, not of the people maintaining it. `query.c` has
behaviour spread across: a field whose meaning depends on three other flags; runtime state set
on one step and read by another; analysis results interleaved with instructions; and a dedup
pass that reaches across every live thread. Nothing can be reasoned about in isolation because
nothing *is* isolated.

A program can be reasoned about. So the goal is to turn implicit, distributed, mutable state
into an explicit program plus a small, stated machine.

## Three principles, each aimed at a real bug class

**P1 — one opcode, one meaning.** No behaviour encoded in flag combinations.

Today `alternative_index` means four different things depending on `is_dead_end`,
`is_pass_through`, and `alternative_is_skip` ([`01-current-state.md`](01-current-state.md)).
That table is why the alternative-following logic cannot be read in isolation, and it is what
made the merge spike double-seed every control state.

**P2 — anchors are operands on position advancement, never runtime state.**

This is the big one. Today `seeking_immediate_match` and `skipped_quantifier` are *set on one
step and read at another*, because the same step can be reached by threads with different
adjacency obligations. Four separate commits in this tree
(`a6bc72474`, `139b801cc`, `dfcf73921`, `1ffd612be`) exist to get those transfers right, and
their logic lives in a 50-line comment block at `query.c:4453-4501`.

The fix is the classic move: **encode the state in the program counter.** If a step can be
reached under two different adjacency requirements, the compiler emits *two instructions* with
different operands. The flag disappears into the program, and the semantics become local — you
read one instruction and know what it requires.

Cost is program size. Step arrays measure 2–5 KiB for real queries, so there is enormous
headroom.

**P3 — thread state is exactly `(pc, position, captures)`.**

Every other field on `QueryState` either moves into the program (P2) or is eliminated. No
`seeking_immediate_match`, no `skipped_quantifier`, no `needs_parent`, no
`has_in_progress_alternatives`. If a thread's future behaviour depends on something, that
something is in the program, not in a boolean someone set three steps ago.

## Shape of the machine

The tree walk is the input tape, and it is **two-dimensional**: siblings and depth. Two possible
shapes:

- **event-driven** — the engine walks the tree and delivers "entered node N at depth d";
  threads waiting for that event run.
- **program-driven** — instructions actively navigate, and the program drives the cursor.

Event-driven is right for the real workload: `rust/highlights.scm` is 94 patterns over one
traversal, and program-driven would mean 94 walks. Program-driven becomes interesting later for
*selective* single patterns — start at the rarest node and verify upward
([`05-database-angle.md`](05-database-angle.md) §3) — so the IR should not foreclose it, but the
first VM is event-driven.

Consequence: **instructions test "the current node"**, and position advancement is what blocks a
thread until the traversal delivers the next one.

## Instruction set

Operands are small integers; the encoding is a later question.

```
; ---- node tests: fail the thread unless they hold ----
MATCH_SYM      sym            current node's symbol is sym
MATCH_ANY      named          wildcard; named=1 restricts to named nodes
MATCH_SUPER    sym            sym is among the current node's supertypes
MATCH_MISSING                 current node is a MISSING node
CHECK_FIELD    field          current node occupies field
CHECK_NO_FIELD list           current node has no child in any of these fields

; ---- captures ----
CAPTURE        cap            bind the current node to cap
CAPTURE_PARENT cap            bind the parent (today's wildcard-root shortcut, made explicit)

; ---- position advancement: these BLOCK until the traversal supplies a node ----
CHILD          anchor         descend; anchor ∈ {ANY, FIRST}
SIBLING        anchor         advance among siblings; anchor ∈ {ANY, IMMEDIATE}
LAST                          assert the node just matched was the last named child
UP                            return to the parent level

; ---- control flow ----
SPLIT          a, b           fork: one thread to a, one to b
JMP            t
ACCEPT         pattern
```

Two things to notice.

**Anchors live on `CHILD`/`SIBLING`, not on the node test.** `SIBLING IMMEDIATE` and
`SIBLING ANY` are different instructions, so "must this be adjacent?" is answered by reading one
instruction rather than by reconstructing which flags a thread is carrying.

**`LAST` is a separate instruction**, not a bit on the node test, because it is an assertion
about the *position just consumed*. Today it is `is_last_child` on the step, which is why it has
to be propagated onto alternation branches by hand (`query.c:2763-2779`).

## Thread state and the dispatch loop

```c
typedef struct {
  uint32_t pc;
  Position pos;        // depth + sibling index, relative to the pattern's start
  CaptureSet caps;     // sorted positions -- see below
} Thread;
```

The loop has two levels. The outer is the traversal; the inner runs each thread until it
blocks:

```c
for (each node event: entered N at depth d) {
  for (each thread T with T->pos waiting at depth d) {
    for (;;) {
      Instr *in = &program[T->pc];
      switch (in->op) {
        case MATCH_SYM:   if (sym(N) != in->a) goto kill;  T->pc++; continue;
        case CHECK_FIELD: if (field(N) != in->a) goto kill; T->pc++; continue;
        case CAPTURE:     capture_set_add(&T->caps, T->pos, in->a); T->pc++; continue;
        case SPLIT:       fork(T, in->b); T->pc = in->a; continue;
        case JMP:         T->pc = in->a; continue;
        case ACCEPT:      emit(T, in->a); goto done;
        case SIBLING:     /* blocks */ T->pos = advance(T->pos, in->a); T->pc++; goto block;
        case CHILD:       /* blocks */ T->pos = descend(T->pos, in->a); T->pc++; goto block;
        ...
      }
    }
  }
}
```

Everything that is not a position advance executes immediately, so a thread consumes a *run* of
instructions per node. That is what lets tests, captures and control flow compose without any of
them needing to know about the traversal.

Dispatch is a plain switch to start with. Computed goto is a later micro-optimisation and
should not shape the design.

## Captures

Settled empirically by the merge spike
([`02-execution-model.md`](02-execution-model.md)): a capture set is a **sorted vector of
positions**, not node references. `CAPTURE cap` records *(position, cap)*; the node is recovered
from the traversal at emit time.

That is what made subset testing structural — and therefore what made longest-match enforceable
at merge time rather than by an O(n²) pass. It also satisfies the public API's interior-pointer
requirement, since a contiguous `TSQueryCapture[]` gets materialised once per emitted match
anyway.

**Do not let node references into thread state.** It is the single decision most likely to be
made by accident and most expensive to undo.

## Disambiguation

Stated once, in the program header, rather than emerging from a runtime pass:

```
program {
  disambiguation: longest-match      // the current behaviour, written down
  patterns: 94
  ...
}
```

The VM implements the named policy at thread-merge time. Changing it becomes a deliberate,
reviewable act rather than a side effect of editing a comparison function
([`03-correctness.md`](03-correctness.md) §C1).

## What this structurally eliminates

The maintainability argument, made concrete against bugs actually found:

| bug found this session | why it becomes impossible |
|---|---|
| A1 — 4th capture silently dropped | captures are an unbounded list in the IR; no `capture_ids[3]` |
| A3 — 9th negated field dropped | same |
| A2 — `(MISS)` parsed as `(MISSING _)` | keyword recognition lives in a lexer stage, not a `strncmp` mid-parse |
| A5 — comparator not antisymmetric | ordering is a stated property of one data structure, not an invariant spread across a walk |
| the four anchor commits | anchors are operands (P2); no runtime transfer between steps |
| spike double-seeded every control state | one authoritative enumeration of entry points; `pattern_map` and `alternative_index` do not both encode it |
| `pass_through` / `dead_end` confusion | distinct opcodes (P1) |
| analysis entangled with matching | analysis is a side table keyed by pc, swappable and cacheable ([`13-node-types-analysis.md`](13-node-types-analysis.md)) |

## Worked example

`((line_comment)* @doc . (function_item name: (identifier) @name) @fn)`

```
  0: CHILD    ANY            ; enter the sibling sequence
  1: SPLIT    2, 6           ; zero or more line_comments
  2: MATCH_SYM line_comment
  3: CAPTURE  @doc
  4: SIBLING  IMMEDIATE      ; the run is contiguous
  5: JMP      1
  6: MATCH_SYM function_item ; '.' anchor: adjacency required here
  7: CAPTURE  @fn
  8: CHILD    ANY
  9: CHECK_FIELD name
 10: MATCH_SYM identifier
 11: CAPTURE  @name
 12: UP
 13: ACCEPT   0
```

Compare with reading the same pattern out of today's step array, where the quantifier is a
`pass_through` step with a backward `alternative_index`, the anchor is an `is_immediate` bit
whose meaning depends on whether a zero-match skip was taken, and the `name:` child is a depth
change inferred from a `uint16`.

Note what the anchor became: `SIBLING IMMEDIATE` at 4 (the run is contiguous) versus the entry
at 6 being reached from either the loop or the zero-match skip. If those two paths need
different adjacency, the compiler emits two instructions rather than setting a flag — P2 in
practice.

## Open questions

- **Encoding.** Fixed-width vs variable-length; operand widths; whether spans and provenance
  live in side tables keyed by pc (they should, so bytecode stays stable when only diagnostics
  change).
- **Where the plan layer goes.** Selectivity-driven start points and program-driven traversal
  are *plan* choices. Do they get their own level between HIR and bytecode, or does the
  bytecode simply grow instructions to express them?
- **Predicates.** Represented in the IR? Evaluated by the VM (needs text access, an API change,
  cross-binding coordination)? Or left where they are? The largest optional scope fork.
- **Incrementality.** If per-node VM state is ever cached against shared subtrees, it must be
  explicit and addressable. Cheap to design for now, expensive to retrofit.
- **Stability policy.** Since the IR is a deliverable: what is versioned, what may change, and
  what happens when a grammar regenerates and symbol ids move. The schema work chose
  names-plus-load-time-resolve for exactly this reason; the same answer probably applies.
- **How much program duplication P2 causes in practice.** Bounded by alternation × quantifier
  nesting, but unmeasured. Worth checking against real query files before committing.
