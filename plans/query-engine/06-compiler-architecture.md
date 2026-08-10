# Compiler architecture: stages, IR, and a stable bytecode

The brief asked to "treat the query builder as a real compiler, rather than an ad hoc
implementation", with "better separated stages, consumable as a library". This is the design
for that.

## Why the current shape resists change

Concretely, from [`01-current-state.md`](01-current-state.md):

- `ts_query__parse_pattern` emits `QueryStep`s as a side effect of parsing, and back-patches
  `alternative_index` during the parse. There is no point at which a complete, inspectable
  representation of the user's query exists.
- Because of that, alternation linking and quantifier linking contend for the same field, and
  the fix is a **post-hoc code-rewriting pass** (`query.c:3144-3189`) that clones steps and
  splices redirect nodes into the already-emitted program.
- Because there is no IR, the public introspection API is keyed by **source byte offset**
  (`ts_query_is_pattern_guaranteed_at_step`), and diagnostics are a single offset plus a
  7-variant enum.
- Because analysis results live in the same struct as the instructions, "what the user wrote",
  "what we derived", and "what the machine runs" cannot be separated, versioned, cached, or
  serialized independently.

Every one of those is a consequence of the missing stage boundary, not of any individual
decision.

## Proposed stages

```
  .scm source
      │
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 1. LEX + PARSE                                               │
  │    → CST/AST with full spans, error recovery, no language    │
  │      knowledge required                                      │
  │    Library surface: parse a .scm without a TSLanguage        │
  └───┬──────────────────────────────────────────────────────────┘
      │  AST
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 2. RESOLVE                                                   │
  │    → HIR: symbols/fields/supertypes resolved against the     │
  │      TSLanguage; captures interned; predicates parsed into   │
  │      structured form; quantifiers computed                   │
  │    All "unknown node type / field / bad supertype" errors    │
  │      surface here, with spans, and ALL of them at once       │
  └───┬──────────────────────────────────────────────────────────┘
      │  HIR  (pattern tree, still structural — no control flow)
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 3. ANALYZE                        [cacheable, parallel]      │
  │    → per-step guarantees, feasibility, selectivity estimates │
  │    Split: language-only facts (cache on TSLanguage)          │
  │           vs pattern-specific facts (cache by pattern hash)  │
  └───┬──────────────────────────────────────────────────────────┘
      │  HIR + facts
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 4. PLAN / OPTIMIZE                                           │
  │    → choose start point (root vs most-selective node),       │
  │      order sibling constraints, merge shared prefixes across │
  │      patterns, decide predicate pushdown                     │
  │    This stage does not exist today at all.                   │
  └───┬──────────────────────────────────────────────────────────┘
      │  LIR / plan
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 5. LOWER → BYTECODE                                          │
  │    → explicit opcodes, no overloaded flags                   │
  │    Serializable, versioned, verifiable                       │
  └───┬──────────────────────────────────────────────────────────┘
      │  bytecode
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 6. EXECUTE (VM)                                              │
  └──────────────────────────────────────────────────────────────┘
```

The stage boundaries are the product. Each of 1→2→3→4→5 should be callable independently,
which is what makes a query LSP, a formatter, a linter, and an optimizer possible without any
of them re-implementing the parser.

## The IR: two levels, not one

**HIR — structural, close to what the user wrote.** A tree, not a flat array:

```
Pattern {
  root: Node
  predicates: [Predicate]
  span: Span
}
Node =
  | Symbol   { symbol, supertype?, field?, negated_fields: [FieldId],
               captures: [CaptureId], children: [Child], anchored_start, anchored_end,
               is_missing, span }
  | Wildcard { named: bool, field?, captures, children, ..., span }
  | Alt      { branches: [Node], captures, span }
  | Quant    { inner: Node, kind: ?|*|+, captures, span }
  | Group    { items: [Node], span }
Child = { node: Node, immediate: bool }
```

Properties that matter:

- **Adjacency is an explicit axis, not an implicit depth delta.** Today a step's relationship
  to its parent is encoded solely as `depth`, and "child of" is inferred from `depth + 1`.
  The HIR should name the relation — `Child`, `Descendant`, and whatever else the language
  grows — as an edge kind on `Child { node, immediate, axis }`. This is a requirement, not a
  nicety: see the `#880` discussion below.
- **Captures live on nodes, as a list with no fixed bound** — A1 in
  [`03-correctness.md`](03-correctness.md) disappears structurally.
- **Negated fields are a list, unbounded** — A3 disappears structurally.
- **Anchors are properties of positions**, not a flag that gets transferred between steps by
  runtime logic. The four anchor-semantics commits in this tree become HIR-level rules that
  can be stated and tested in isolation.
- **Spans everywhere** — diagnostics, LSP, formatting, and `--explain` all become possible.
- This is the level at which a query **formatter** and **linter** operate, and the level at
  which query composition (`03` §C4) would be defined.

**LIR/bytecode — explicit control flow.** A flat instruction array, one opcode per operation:

```
  ; matching
  MATCH_SYMBOL   sym                 ; fail thread if current node symbol != sym
  MATCH_WILDCARD named
  MATCH_SUPERTYPE super
  CHECK_FIELD    field
  CHECK_NEG_FIELDS  list_id
  CHECK_MISSING
  CHECK_FIRST_CHILD                  ; leading anchor
  CHECK_LAST_CHILD                   ; trailing anchor
  ; captures
  CAPTURE        capture_id
  CAPTURE_PARENT capture_id          ; the wildcard-root optimization, made explicit
  ; control flow
  SPLIT          a, b                ; fork: one thread to a, one to b
  JMP            target
  DESCEND                            ; enter child level
  ASCEND
  ACCEPT         pattern_id
```

The point is not the exact opcode list. The point is that `alternative_index` +
`is_pass_through` + `is_dead_end` + `alternative_is_skip` — four fields whose combinations
encode four different control-flow constructs — become four distinct opcodes. The table in
[`01-current-state.md`](01-current-state.md) §"Control flow is encoded in flag combinations"
becomes unnecessary because the code says what it means.

This is a Thompson NFA program, deliberately. It makes the regex literature
([`05-database-angle.md`](05-database-angle.md) §2) directly applicable, and it is the form
that a later TDFA construction consumes.

## The capture representation: a derived requirement

Everything else in this document is design-by-argument. This section is the one part derived
from experiment ([`02-execution-model.md`](02-execution-model.md), spike results), and it is a
hard constraint on the matcher the IR has to serve.

**The requirement:** capture histories must support *subset testing structurally*, not by set
comparison. The spike showed that control-state merging is settled — 56× collapse, already
faster than the stock engine — and that what blocks the remaining win is enforcing longest-match
at merge time. A cons-list makes suffix-extension cheap but general subset O(n·m), which is the
stock engine's pairwise cost relocated rather than removed.

### Why not "just change the disambiguation policy"

The tagged-automata literature resolves this by specifying a policy (POSIX, greedy) applied at
merge time, and that is the principled answer. It is also the **highest-risk change available**,
for a reason this investigation surfaced by accident:

> The golden corpus measures **1.00× state collapse on every real query file**. Real queries
> barely exercise the disambiguation machinery at all.

So a policy change would pass the entire test corpus and surface downstream, in whatever
unusual query shapes do exercise it, long after landing. The safety net is thinnest exactly
where the change is riskiest. Defer it; it is not needed for the measured 34×.

### Bitsets over sibling positions, with node identity kept out of the continuation

Continuations that reach the same control state have matched the *same steps*; they differ only
in **which nodes they bound to quantified steps**. And those nodes are always drawn from the
sibling sequence currently being scanned. That gives a dense, bounded index for free:

```
continuation = one bitset per quantified step in the pattern,
               indexed by sibling position within the current parent's child list
```

`A ⊇ B` becomes a per-step bitset AND — O(quantified_steps × words), and for realistic patterns
(1–3 quantifiers, sequences of tens of siblings) that is one or two machine words per test.

**This answers where node ids live: nowhere in the continuation.** The bitset records *which
sibling positions* were bound; the nodes themselves are recovered from the sibling array when a
match is actually emitted. Node identity stays in the traversal, and continuations carry only
positional information. Consequences:

- continuations become small and fixed-size, so merging is cheap and the arena shrinks;
- the `TSQueryCapture` array is materialized once per *emitted* match, which is also what the
  public API's interior pointer requires ([`11-data-oriented-design.md`](11-data-oriented-design.md));
- the O(n·m) subset test disappears without touching semantics.

Open questions before committing:

- **Sizing.** A sequence with thousands of siblings makes the bitset large. Realistic child
  counts are tens; the pathological ones are hundreds. Needs a measured distribution over the
  corpus, and probably a fallback for the tail.
- **Non-quantified captures.** Captures on non-quantified steps are identical across all
  continuations at a control state, so they need not be in the bitset at all — they can be
  recovered from the path. Worth confirming.
- **Depth > 1 patterns.** The sibling-position index is per-sequence; a pattern spanning
  several depths needs one index space per level, or a different scheme.

### What is still not ready to draft

The rest of the IR — HIR node kinds, opcode set, bytecode encoding — should wait. The capture
representation is derivable now because an experiment produced a constraint; the others would be
speculative until the matcher is faithful enough to say what it needs. The productive order is:
finish the matcher against this representation, collect the constraints it produces, then draft.

## Stable serialized bytecode

Real benefits, in order of value:

1. **Precompilation eliminates the compile cost entirely.** 16.9 ms per Rust highlights
   compile ([`04-performance.md`](04-performance.md)) becomes a `mmap` + a symbol-resolution
   pass. For an editor loading 60 query files this is the difference between ~500 ms and
   ~nothing.
2. **Cross-implementation conformance.** The Rust, WASM/JS, Go, Python, and C implementations
   currently agree only by construction. A specified bytecode plus a conformance corpus makes
   divergence detectable.
3. **Third-party tooling.** Optimizers, visualizers, coverage tools, and differential fuzzers
   all become possible against a documented artifact.
4. **Golden testing.** Bytecode snapshots catch unintended semantic changes in the compiler —
   exactly the class of regression the four anchor commits in this tree represent.

### The hard part: symbol ids are not stable

`TSSymbol` and `TSFieldId` values are assigned by `crates/generate` and change whenever a
grammar is regenerated. Serialized bytecode that embeds raw symbol ids is invalidated by any
grammar change, silently and dangerously (a stale id is still a *valid* id, just the wrong
node type).

Recommended design:

- **Serialize names, not ids.** Store a string table of node-type names, field names, and
  supertype names; the bytecode references string-table indices. Loading performs a resolution
  pass — one hash lookup per distinct name, so tens of lookups, microseconds.
- **Stamp the artifact** with the grammar name, its semantic version (already available via
  the `ts_language_*` version API added in `8bb1448a6`), the language ABI version, and a
  content hash of the parse table. Refuse to load, or fall back to source compilation, on
  mismatch — never silently accept.
- **Version the bytecode format independently** of both the grammar and the library, with an
  explicit `(major, minor)` and a documented compatibility policy.
- **Validate on load.** A verifier pass — jump targets in range, stack discipline sound,
  capture ids within bounds — so that a corrupt or hostile artifact cannot produce memory
  unsafety. This is the WebAssembly lesson: a bytecode format without a specified validation
  pass is an attack surface.

### Where the analysis results live

Deliberately **not** in the bytecode. Analysis output (`root_pattern_guaranteed`, selectivity
estimates) is a function of `(pattern, language)` and is an *optimization input*, not
semantics. Keeping it in a separate side table means:

- bytecode stays a pure description of matching semantics, so golden tests over it are stable
  when only the analyzer changes;
- analysis can be cached, recomputed, skipped, or improved independently
  ([`04-performance.md`](04-performance.md) P1/P4);
- an implementation that skips analysis entirely is still *correct*, just slower — a much
  better property than today, where analysis failure conservatively degrades but is entangled
  with the step array.

## Library surface

The stages should be a Rust crate (`tree-sitter-query`?) that the C library can also use, or
at minimum a documented C API mirroring the stage boundaries:

```
parse(source)                     -> Ast | [Diagnostic]
resolve(ast, language)            -> Hir | [Diagnostic]
analyze(hir, language, cache)     -> Facts
plan(hir, facts, options)         -> Plan
lower(plan)                       -> Bytecode
serialize/deserialize(bytecode)   -> bytes
verify(bytecode)                  -> Result
```

### The feature that forces the axis question: `#880`, the descendant axis

[tree-sitter#880](https://github.com/tree-sitter/tree-sitter/issues/880) — "Specify descendant
or ancestor in query" — is the longest-running query-language request, and it is the same gap
identified independently in [`03-correctness.md`](03-correctness.md) §C3 and
[`05-database-angle.md`](05-database-angle.md). Users currently enumerate nesting levels by
hand:

```scheme
(declaration declarator: [
  (identifier) @name
  (_ declarator: (identifier) @name)
  (_ declarator: (_ declarator: (identifier) @name))
  ...                                    ; "manual recursion hell", per the thread
])
```

There is a draft PR, [#5403](https://github.com/tree-sitter/tree-sitter/pull/5403), adding
`(^)` back-references inside alternations — De Bruijn-indexed recursive descent (`(^^)` for the
next outer alternation), implemented by reusing the dead-end step mechanism with a *backward*
`alternative_index`. +1,844/-50, three files.

Two structural reasons to take a different route, both of which are about this document's
thesis rather than about that PR's quality:

1. **It adds a fifth meaning to `alternative_index`.** That field already encodes four distinct
   control-flow constructs through flag combinations
   ([`01-current-state.md`](01-current-state.md) §"Control flow is encoded in flag
   combinations"), and that overloading is the direct cause of the back-patching bugs and of
   the peephole repair pass at `query.c:3144-3189`. The PR notes "no new fields on QueryStep or
   QueryState" as a virtue; in this codebase, *not* adding a field means overloading one that
   is already carrying four jobs.

2. **Recursive descent as control flow multiplies depth-scoped threads.** Query states are
   depth-scoped — a state matches only when `start_depth + step->depth == self->depth`
   (`query.c:4263`). Expressing descent as a backward jump means a thread re-enters the same
   steps at successively greater depths, so a candidate ancestor spawns work per descendant
   level. That is the same mechanism that produces the M² live-state growth measured in
   [`02-execution-model.md`](02-execution-model.md), applied to a construct users would reach
   for constantly. **This is a hypothesis, not a measurement** — the profiler could be pointed
   at that branch over a deeply-nested corpus to settle it, and doing so would be a genuinely
   useful contribution to the PR discussion either way.

The route this architecture implies instead: **a descendant axis is a relation between HIR
nodes, not a jump in the instruction stream.** With region encoding — which tree-sitter already
has, since every node carries a byte range — ancestor/descendant is an O(1) containment test,
and matching a descendant edge becomes a structural join rather than a per-depth thread
([`05-database-angle.md`](05-database-angle.md) §1). It composes with everything else and it
does not need new control flow.

That has three consequences worth recording now:

- **the HIR needs the explicit axis field** described above, from the start;
- **the semantics need deciding alongside the disambiguation policy**: what does `.` mean under
  a descendant axis (probably an error), how does `is_rooted` interact with a pattern that
  spans arbitrary depth, and what happens to `max_start_depth` and range restriction;
- **it should land after the matcher can merge threads**, not before. Implementing `#880` on
  the current engine means shipping the pathological shape as a first-class language feature.
  Another reason the sequencing in [`09-roadmap.md`](09-roadmap.md) puts language growth in
  Phase 5.

### A real consumer to design against: `ts_query_ls`

[`ts_query_ls`](https://github.com/ribru17/ts_query_ls) (ribru17) is a language server for
`.scm` files: diagnostics for impossible patterns and invalid node names, completion for node
names / fields / captures / predicates, nvim-treesitter-compatible formatting, and
go-to-definition, references, and rename for captures. It is the closest thing to a real-world
test of the API this document proposes.

Worth treating as a **design data point, not a requirement** — nothing here should gate on it.
But three things about how it is built are direct evidence for the diagnosis in
[`01-current-state.md`](01-current-state.md):

1. **It parses `.scm` with a tree-sitter grammar for the query language**, not with
   tree-sitter's own query parser. That is the only option available: `ts_query_new` either
   returns an opaque `TSQuery` or one byte offset, and neither is something a language server
   can build on. The ecosystem's answer to "there is no AST" was to write a second parser for
   the same language. Stage 1 of the pipeline above is exactly the thing that would not have
   needed writing.

2. **Its documented limitation is our §C5.** Impossible-pattern detection "requires expensive
   full query file execution". The library *already computes this* — that is what the
   `analysis.finished_parent_symbols.size == 0` path in `ts_query__analyze_patterns`
   (2013-2049 region) decides — but the only way to consume it is to compile the whole file
   and get back a single `TSQueryErrorStructure` at a single byte offset. Finding the *second*
   bad pattern means editing it out and compiling again. The information exists and is thrown
   away at the API boundary.

3. **It implements `; inherits:` module imports itself**, because the query language has no
   composition ([`03-correctness.md`](03-correctness.md) §C4). Another capability the ecosystem
   built around the library rather than in it.

This gives a concrete acceptance test for the staged API, worth checking before committing to
a shape: **could `ts_query_ls` replace its expensive impossible-pattern path with a single
`resolve()` + `analyze()` call returning every diagnostic with a span?** If yes, the stage
boundaries are drawn in the right places. If it still needs to compile repeatedly, they are
not. Validating a design against an existing consumer is much cheaper than discovering the
boundaries are wrong after shipping them.

Consumers this immediately unlocks, none of which are possible today:

- **query LSP** — completion of node types and fields from the grammar, go-to-definition on
  captures, diagnostics with spans, hover showing whether a step is guaranteed.
- **formatter** — `.scm` files have no canonical format; every downstream project formats
  differently.
- **linter** — "this pattern can never match" (already computed, currently only surfaced as a
  compile error), "this capture is never used by any predicate", "this pattern is subsumed by
  pattern 4", "this quantifier shape is known-quadratic" (which would have flagged the 4.2 s
  query at authoring time).
- **`tree-sitter query --explain` / `--profile`** — the DB EXPLAIN analogue.
- **coverage** — which patterns in highlights.scm never fire on a corpus.

## Migration strategy

The strategy that makes this survivable is **not** a rewrite behind a flag day.

1. Build stages 1–2 (parse → HIR) as a **new, independent** path that produces the *existing*
   `QueryStep` array as its output. Same engine, same semantics, new front end.
2. Differential-test it: for every `.scm` in the fixture corpus plus generated queries,
   assert the new front end emits a byte-identical step array to the old one. Where it
   differs, either the old one has a bug (fix and record) or the new one does.
3. Only once the front end is proven, introduce the bytecode as a second output, and the new
   VM behind a runtime switch, differential-tested on *match results* over a corpus rather
   than on internal structures.
4. The disambiguation change (P7) is the one step that intentionally changes results. It must
   be gated on the written spec ([`03-correctness.md`](03-correctness.md) §C1) and a
   conformance suite, and it should land last.

This ordering means every intermediate state is shippable, and the risky change happens once,
late, with the strongest possible test coverage behind it.
