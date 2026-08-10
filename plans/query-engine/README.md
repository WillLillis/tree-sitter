# Query engine: exploration and rebuild plan

Status: exploration complete, nothing agreed. This is a survey + argument, not a commitment.
All numbers in this doc set were measured on this tree at `003b10c28` unless stated otherwise;
the harness that produced them lives in `tools/query-profiler/` and is described in
[`08-measurement-harness.md`](08-measurement-harness.md).

**Scope: these are working notes on the `query-engine/*` branches. They are not upstream
material** — tree-sitter's published docs are user-facing, and internal implementation notes
would rot there. The public artifacts are (a) code, (b) the conformance suite, (c) decision
records next to the code, and (d) a post per phase, written when that phase lands. See
[`09-roadmap.md`](09-roadmap.md) §"Writing it up".

## The thesis

`lib/src/query.c` is 4,876 lines that fuse five distinct jobs into two functions:

- `ts_query__parse_pattern` (620 lines) is simultaneously a lexer, a parser, a semantic
  analyzer, and a code generator. There is no AST. Steps are emitted and back-patched in place.
- `ts_query_cursor__advance` (615 lines) is simultaneously a tree walker, an NFA simulator,
  a capture-set manager, and a match disambiguator.

Everything that is hard about the current engine follows from those two sentences. The
correctness bugs are back-patching bugs. The performance cliffs come from a disambiguation
policy that is *emergent* from an O(n²) pairwise comparison pass rather than *specified*
in the automaton. The absence of tooling follows from there being no intermediate
representation to build tooling on.

The proposal is not "optimize query.c". It is: **build a real compiler with real stages,
give it a stable IR, and replace the emergent disambiguation with a specified one.** The
literature for all three of those is mature and largely from databases, regex engines, and
tree automata — see [`05-database-angle.md`](05-database-angle.md) and
[`07-references.md`](07-references.md).

## Five headline measurements

1. **Query compilation is ~99% static analysis, and most of it does not depend on the query.**
   `(identifier) @v` — sixteen bytes — takes **2.90 ms** to compile against the Rust grammar.
   2.77 ms of that (96%) is a scan of the entire parse table plus a `state_count × 257`
   predecessor map (**1.9 MiB `calloc`** for Rust). Every `ts_query_new` call pays it again.

2. **Compiling a query costs more than parsing the file it runs on.** Rust `highlights.scm`
   (3.5 KB) compiles in **16.3 ms**. Parsing a 195 KB Rust file takes 21 ms; running the
   query over it takes 13 ms. An editor loading 20 languages × 3 query files pays this
   60 times at startup.

3. **A quantified sibling pattern without anchors has a combinatorial answer set — and that,
   not the engine, is most of the cost.** `((attribute_item)* @attr (line_comment)* @doc
   (function_item name: (identifier) @name) @fn)` over M attributed functions:

   | M | unanchored | matches | **anchored** (`.` between runs) | matches |
   |---|---|---|---|---|
   | 25 | 137.8 ms | 2,925 | **0.145 ms** | 25 |
   | 50 | 3,904 ms | 22,100 | **0.270 ms** | 50 |
   | 100 | > 60 s | — | **0.566 ms** | 100 |

   Adding two `.` characters is a **104× speedup on a real 195 KB file** (738 ms → 7.07 ms)
   and yields the answer the author almost certainly wanted. Without anchors, sibling steps do
   not require adjacency, so every in-order subset of preceding attributes and comments is a
   legitimate match. The engine is *slower per match* on the large answer set — 177 µs vs
   5.4 µs anchored, roughly 33× — which is the real O(n²) dedup cost and is worth fixing. But
   the dominant factor is that the query asks for a combinatorially large answer.
   **This is a semantics and ergonomics trap first, an engine defect second.**
   See [`02-execution-model.md`](02-execution-model.md).

4. **The engine has no index over the tree and no index over its own state.** At every node
   it linearly scans every live state; on the pathological case **99.1% of those visits are
   rejected by the depth test alone**. The capture-list pool finds a free slot by linear
   scan — average scan length **191.6** on that same case.

5. **Silent data loss is reachable from valid queries.** `(identifier) @a @b @c @d` compiles
   without diagnostic, reports `capture_count == 4`, and returns matches with 3 captures
   forever. Verified, see [`03-correctness.md`](03-correctness.md).

## Validated: 7.9-17.2x on the target case, with identical results

The one design question the whole investigation converged on has an experimental answer.
Representing capture sets as **bitsets over (sibling position × capture id)**, with node
identity kept out of the continuation, makes subset testing structural — which makes
longest-match enforceable at merge time rather than by the O(n²) pairwise pass:

| workload | stock | merge + bitset | matches | |
|---|---|---|---|---|
| unanchored, M=50 | 4,034 ms | **514 ms** | 22,100 = 22,100 | **7.9×** |
| unanchored, real 195 KB file | 727 ms | **25.9 ms** | 8,003 = 8,003 | **28.1×** |
| unanchored, M=100 | 125,377 ms | **24,393 ms** | 171,700 = 171,700 | **5.1×** |
| anchored (control) | 0.37 ms | 0.98 ms | 120 = 120 | 0.4×, nothing to merge |

**Identical match sets, no semantics change, no disambiguation-policy decision.** Capture sets
are best stored as *sorted position vectors* rather than dense bitsets — a dense bitset's width
scales with the sibling count, a sparse one with the capture count.

The speedup still shrinks with input size, so this is a large constant-factor win rather than a
change of complexity class. The remaining bottleneck is located: the prune step scans every
head in a control state, so a state holding *h* continuations costs O(h²) — the stock engine's
dedup problem scoped down from "all live states" to "one control state", which is where the 28×
comes from and why the shape persists. Removing it needs a subset-query index. See
[`02-execution-model.md`](02-execution-model.md); the spike is `tools/query-profiler/merge_spike.c`.

## Reading order

| Doc | What it is |
|---|---|
| [`01-current-state.md`](01-current-state.md) | Accurate map of the code as it exists: stages, structures, encoding, invariants, limits |
| [`02-execution-model.md`](02-execution-model.md) | How the VM actually runs, and why the blowups happen |
| [`03-correctness.md`](03-correctness.md) | Verified defects with repros, plus unverified suspicions and semantic gaps |
| [`04-performance.md`](04-performance.md) | Cost model, measurements, ranked opportunities |
| [`05-database-angle.md`](05-database-angle.md) | What DB/automata/regex research actually transfers, and what doesn't |
| [`06-compiler-architecture.md`](06-compiler-architecture.md) | Staged compiler, IR, bytecode, stability story |
| [`07-references.md`](07-references.md) | Annotated bibliography — what to read and why |
| [`08-measurement-harness.md`](08-measurement-harness.md) | The profiler used here; recommend landing it in-tree |
| [`09-roadmap.md`](09-roadmap.md) | Sequencing, risk, and what to do first |
| [`10-differential-rig.md`](10-differential-rig.md) | What the testing oracle actually is at each phase, and what to build when |
| [`11-data-oriented-design.md`](11-data-oriented-design.md) | Where DoD pays here, where the algorithmic fix dominates it, and what that means for P2 |

## If you only do three things

1. **Land the measurement harness, then pin today's behaviour with goldens** (`08`, `10`).
   The harness is done (`tools/query-profiler/`). The goldens are ~100 lines of dumper plus
   one `#[test]` in the existing suite, and taking them *before* P1/P2 land is what turns
   "these changes are semantics-preserving" from a claim into a check. Note `10` walks back
   the earlier "build a differential rig in Phase 0" advice: the oracle differs per phase and
   most of it is not needed yet.
2. **Cache the language-only half of query analysis** (`04`, opportunity P1). It is a
   contained change, it does not touch semantics, and it removes 0.7–2.8 ms from every
   `ts_query_new` call.
3. **Decide the disambiguation question before writing any new engine code** (`02`, `05`).
   "What does this query mean when several matches overlap?" currently has no written answer
   — the answer is whatever `ts_query_cursor__compare_captures` does. Every downstream design
   depends on picking one deliberately.
