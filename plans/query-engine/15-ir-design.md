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

**Position-keying is also what makes quantified captures work.** Each iteration of a loop
executes the *same* `CAPTURE` instruction at a *different* position, setting a different entry —
three `line_comment`s produce three `@doc` entries, in position order. That matches the existing
contract, where `TSQueryMatch.captures` may repeat a capture index and `capture_quantifiers`
already records `@doc` as `ZeroOrMore`.

The failure mode worth naming: if the capture set were keyed by *capture id* rather than by
*position*, the second iteration would overwrite the first and every quantified capture would
silently collapse to one binding. So position-keying is load-bearing for quantifier semantics,
not only for cheap subset testing — which makes it exactly the kind of thing an implementer
might "simplify" away.

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
  0: CHILD    ANY             ; first candidate in the sibling sequence
  1: SPLIT    2, 20           ; enter the run, or skip it entirely
  ; --- run body ---
  2: MATCH_SYM line_comment
  3: CAPTURE  @doc
  4: SPLIT    5, 8            ; another comment, or leave the run
  5: SIBLING  ANY             ; comments need NOT be adjacent to each other
  6: JMP      2
  ; --- leaving the run: the '.' applies to this edge ---
  8: SIBLING  IMMEDIATE
  9: JMP      30
  ; --- zero comments: the anchor is vacuous, position unchanged ---
 20: JMP      30
  ; --- shared tail ---
 30: MATCH_SYM function_item
 31: CAPTURE  @fn
 32: CHILD    ANY
 33: CHECK_FIELD name
 34: MATCH_SYM identifier
 35: CAPTURE  @name
 36: UP
 37: ACCEPT   0
```

Two things this gets right that an earlier draft of this document got wrong, both worth
dwelling on because they are the whole point of P2.

**The anchor belongs to the edge leaving the loop, not to the loop body.** `(line_comment)*`
with no internal anchor does **not** require the comments to be adjacent to one another — that
is the unanchored semantics measured in [`02-execution-model.md`](02-execution-model.md), and it
is why the unanchored form yields 22,100 matches where the anchored one yields 128. The `.`
constrains only the transition out of the run. Hence `SIBLING ANY` at 5 inside the loop and
`SIBLING IMMEDIATE` at 8 on the way out.

**`function_item` is reachable two ways, with different requirements.** Via the loop it must be
immediately after the last comment; via the zero-skip the anchor is vacuous and it may appear
anywhere. That is exactly what `skipped_quantifier` encodes at runtime today, and what the four
anchor commits are about. Here it is two code paths, and the flag does not exist.

### P2's duplication cost is bounded, and small

The two paths differ by **exactly one instruction** — `SIBLING IMMEDIATE` at 8 versus falling
through at 20 — and then rejoin at 30 via `JMP`.

That generalises: **P2 duplicates position-advance instructions, not the matching that follows
them.** The divergence is in *how a thread arrived*, not in *what it does next*, so a jump to a
shared tail collapses it. The open question in an earlier draft — "how much does P2 inflate the
program?" — is largely answered by construction: on the order of one extra instruction per
distinct arrival requirement, not a duplicated subtree.

## Resolved in review

**Two levels, not three — no plan layer.** There is one matching strategy today and no concrete
second one in view. *Document* the strategy; do not *abstract* it. An abstraction with a single
implementation is a cost with no payer, and it is precisely the speculative generality that
makes code hard to read later. If selectivity-driven start points ever arrive, they can grow
instructions in the bytecode or a level can be introduced then, with a real second case to
design against.

**Predicates stay out of the VM.** The current engine does not evaluate them at all: it parses
them into `predicate_steps` and hands them to the binding via
`ts_query_predicates_for_pattern`, and every binding implements `#eq?`, `#match?` and friends
independently. Parity is therefore trivial — keep the side table, keep the accessor.

Measured across 1,672 real query files: 42% contain a predicate, ~3,420 uses over ~16,437
patterns. But the most common by far is `#set!` (1,310), which is a **directive** attaching
metadata rather than a filter, and `#lua-match?` (371), `#offset!`, `#make-range!` and
`#select-adjacent!` are **nvim-treesitter's own**, not tree-sitter core. Actual filtering
predicates are ~1,814 uses, about 0.11 per pattern.

That settles it: **the predicate namespace is a consumer-owned extension point the ecosystem has
already extended.** A VM that evaluated predicates would either cover only the core set — leaving
the rest in bindings, so two mechanisms — or need an extension mechanism of its own. The
predicate-pushdown enthusiasm in [`05-database-angle.md`](05-database-angle.md) §3 is retracted
on that basis. If it returns it is an *optimisation with a fallback* for the three core
predicates (`#eq?`, `#any-of?`, `#match?`, 1,364 uses), not a semantic change, and it still
needs text access in the cursor.

## Implementation language: the bytecode is a boundary

Much of what makes `query.c` hard is the absence of a standard library — every map, pool and
dynamic array is hand-rolled or built on `array.h`.

The bytecode turns that into an architectural choice rather than a constraint. The half needing
rich data structures — parsing, resolution, analysis, optimisation — is **ahead-of-time** work
that can live in Rust, exactly as `generate` does and exactly as the node-type schema will
([`13-node-types-analysis.md`](13-node-types-analysis.md)). The half that must stay C is the VM,
and the VM needs almost nothing: a thread list, a program array, a capture set. The dispatch
loop above is a `switch` over two arrays.

**The constraint that keeps this honest:** `lib/` cannot depend on Rust — it must build
standalone for embedding. So `ts_query_new(source)` still needs a C front end unless precompiled
queries become the primary path and source compilation stays a slower convenience. That is a
real fork and should be decided deliberately.

### The data structures actually needed, by measurement

| structure | evidence |
|---|---|
| open-addressed map with epoch clearing | the merge spike's `MIndex`, for control-state lookup; epoch clearing avoided a 256 KB memset per sibling |
| intrusive free-list pool | P2 — capture-list acquisition scans 191 entries on average today |
| small sorted vector with subset ops | capture sets — what made merge-time disambiguation cheap |
| interned string → id table | `symbol_table_id_for_name` is a linear `strncmp` scan |

Four structures, each with a measured need, which is a tractable in-tree library rather than a
stdlib reimplementation. Prototyping them in Rust first would establish the shapes before
committing to C implementations.

## Open questions

- **Encoding.** Fixed-width vs variable-length; operand widths; whether spans and provenance
  live in side tables keyed by pc (they should, so bytecode stays stable when only diagnostics
  change).
- **Incrementality.** If per-node VM state is ever cached against shared subtrees, it must be
  explicit and addressable. Cheap to design for now, expensive to retrofit.
- **Stability policy.** Since the IR is a deliverable: what is versioned, what may change, and
  what happens when a grammar regenerates and symbol ids move. The schema work chose
  names-plus-load-time-resolve for exactly this reason; the same answer probably applies.
- **Whether P2's duplication stays bounded on real queries.** The worked example shows it costs
  one instruction per distinct arrival requirement, with tails shared via `JMP`. Deeply nested
  alternation-inside-quantifier may behave worse; worth checking against real query files, but
  it is no longer an open-ended risk.
