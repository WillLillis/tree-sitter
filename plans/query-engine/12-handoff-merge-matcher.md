# Handoff: the merge-based matcher spike

**Standalone.** You should be able to pick this up without reading the rest of the doc set,
though [`02-execution-model.md`](02-execution-model.md) has the fuller derivation.

Artifact: `tools/query-profiler/merge_spike.c` (743 lines) on branch `query-engine/profiler`.
Build: `make -C tools/query-profiler spike`.

---

## 1. What this is, in one paragraph

`ts_query_cursor__advance` keeps one thread per *(control state × capture history)* and
reconciles them afterwards by pairwise capture-subset comparison — the "longest match" pass at
`query.c:4511-4625`. That pass is O(n²) in the number of live states. The spike replaces it
with one entry per **control state** `(step_index, start_depth, pattern_index, thread-flags)`
holding a *set* of capture continuations, and enforces longest-match **at merge time** instead
of afterwards. Capture sets are sorted vectors of *positions*, not node pointers.

## 2. What it proved

Measured against the stock engine in the same process, over the same tree:

| workload | stock | spike | matches | |
|---|---|---|---|---|
| unanchored two-quant, real 195 KB file | 727 ms | **25.9 ms** | 8,003 = 8,003 | **28.1×** |
| unanchored two-quant, M=50 synthetic | 4,034 ms | 443 ms | 22,100 = 22,100 | 9.1× |
| unanchored two-quant, M=100 synthetic | 125,377 ms | 12,298 ms | 171,700 = 171,700 | 10.2× |
| anchored two-quant (control) | 0.37 ms | 0.98 ms | 120 = 120 | 0.4× |

**Identical match sets in every case**, capture-count histograms agreeing bucket for bucket.
No semantics change, no disambiguation-policy change — the current longest-match rule is
preserved exactly and made cheap.

The 0.4× on the anchored control is expected and correct: live-state collapse there is 1.03×,
so there is nothing to merge and the spike only pays overhead. On real query files collapse is
**1.00×**, so a production version must not regress that path.

## 2b. Effect on anchored queries — a regression, but not the design's

The 28× is on unanchored patterns. Anchored ones (and everything else with 1.00× collapse,
which is every real query file) are **slower** in the spike as it stands. On the real 195 KB
file:

| query | stock | spike | |
|---|---|---|---|
| anchored two-quant | 8.68 ms | 14.11 ms | 0.6× |
| single quantifier, anchored | 10.83 ms | 14.55 ms | 0.7× |
| plain `(function_item name: (identifier) @n) @f` | 8.50 ms | 12.82 ms | 0.7× |

`merged = 0` in all three, so nothing was merged and the merge machinery was pure overhead.
A production version that regressed the common case by 1.3-1.6× would be unshippable.

**But that cost is the prototype's traversal, not the design.** Isolating it — same query, the
walk with matching disabled:

| | time |
|---|---|
| `walk_parents` alone, no matching | **12.27 ms** |
| full spike | 13.65 ms |
| stock engine, entire job | **8.50 ms** |

**90% of the spike's runtime is traversal.** The matching itself costs ~1.4 ms, against the
stock engine's 8.50 ms for everything. `walk_parents` materialises every node's child list into
a stack array and iterates children twice (once to collect, once to recurse), where the stock
engine walks incrementally with the tree cursor and never materialises anything.

Corroborating: `node_tests = 128` on the plain query, meaning the merged representation performs
*far less* per-node matching work than the stock engine's per-state scan — the collapse benefit
is present even at 1.00×, it is just buried under the traversal.

So the expected shape of a production version is **neutral-to-better on anchored queries** and
much better on unanchored ones, provided it reuses the stock traversal
(`ts_tree_cursor_goto_first_child_internal` and the visible/hidden handling) instead of
`walk_parents`. That is a hypothesis with strong evidence, not a measurement — **validating it
is step 1 of any productionisation**, because if it does not hold, nothing else matters.

## 3. Read this before deciding to productionise it

**The target shape is rare.** Scanning 1,672 real query files from `~/grammar_dump`
(314 grammars, nvim-treesitter-sourced), with string literals and comments stripped:

- 76 files contain a sibling-level quantifier
- 60 have two adjacent quantifiers at one nesting level
- **18 have no `.` anchor between them** — the shape the spike optimises

and several of those 18 look like nested-quantifier artifacts rather than true adjacent sibling
pairs, so the real count is lower. They cluster in a few grammars (foam, dart).

So this is a large win for a small number of queries, **not** a broad win. Anyone picking this
up should decide whether that justifies the work before starting. The counter-argument is that
the O(n²) pass is a latent cliff any user can hit by writing an unanchored quantifier pair, and
that 28× on those is worth having — but it is a judgement call, not a foregone conclusion.

Two related corpus findings, same scan, that argue *against* adjacent work:

- **Defect A1** (4th capture on a node silently dropped, limit 3): max observed in a real query
  is **3**, in 2 files out of ~1,760. Not reachable in practice.
- **Defect A3** (9th negated field dropped, limit 8): max observed is **1**. Not reachable.

(One file appeared to use 7 captures on a node — `tree-sitter-typst/test/corpus/typst.scm` —
but it is a 2.8 MB tree-sitter *test corpus* file that happens to end in `.scm`, not a query.)

## 4. Current state: what works, what does not

**Works**
- Dual-engine differ: runs both engines, compares match streams as multisets with captures
  sorted within each match, prints a capture-count histogram and a speedup.
- Control-state merging with an open-addressed index (`MIndex`, epoch-stamped, O(1) clear).
- Sparse capture sets with merge-time longest-match pruning.
- Scope guard: reports `UNSUPPORTED` rather than diverging silently on out-of-scope patterns.

**Does not work / not implemented**
- **Step depth > 1.** `match_child_tail` handles a depth-1 tail on a matched sibling by direct
  field lookup. Anything deeper is rejected by the guard in `main`.
- **`has_in_progress_alternatives`** is not modelled. In the stock engine this defers a state's
  completion while a longer alternative is still live (`query.c:4614`). Its absence has not
  caused a divergence on the tested queries, but it is part of the same longest-match machinery
  and almost certainly matters somewhere.
- **Anchors are only partly modelled.** `expand` mirrors `query.c:4431-4504` including the
  skip/immediate/last-child interactions, but this has only been exercised by two queries.
- **Traversal is simplified.** `walk_parents` materialises each node's child list and recurses,
  rather than mirroring the cursor's visible/hidden-node logic
  (`ts_tree_cursor_goto_first_child_internal`). Hidden nodes may make query-level "siblings"
  span a boundary this ignores. Unvalidated.
- **No range restriction, no `max_start_depth`, no predicates, no supertypes, no
  negated fields, no `MISSING`.**

**Dead code to delete first.** Three generations of capture representation are present and two
are obsolete: the cons-list (`CapCell`, `cap_push`, `cap_is_suffix`, `cap_flatten`, ~lines
280-320) and the dense bitset (`bs_*_dense`, ~244-277). Also unused: `match_cmp`, `F_NEEDS_PAR`.
Removing them takes the file from 743 to roughly 500 lines. Do this before using it as a
reference.

## 5. The open problem, if you continue

**Per-match cost still grows with input size**: 6.4 → 20.0 → 71.6 µs/match across M = 25/50/100.
So this is a large constant-factor win, not a change of complexity class.

The bottleneck is located: `merged_add_a` scans **every existing head** in a control state on
each insertion, so a state accumulating *h* continuations costs O(h²). That is the stock
engine's O(n²) dedup again, scoped down from "all live states" to "heads within one control
state" — which is exactly where the 28× comes from, and why the shape persists.

Removing it needs an index supporting **subset queries** over capture sets. Directions:

- order heads by cardinality, so a superset search can skip everything smaller than the query;
- a cheap signature (e.g. a 64-bit OR-fold of positions) per head for O(1) rejection before the
  merge-walk;
- bound heads per control state and fall back to the stock path beyond it.

Whether flat-delay enumeration is reachable at all is open. Bagan (CSL 2006) says MSO queries
over trees are enumerable with linear delay in principle — see
[`07-references.md`](07-references.md) §2b — so it is an engineering question, not a research
one, but nobody has shown it for this setting.

## 6. Design decisions worth preserving

**Capture sets are sorted vectors of positions, not node pointers.** A set bit means "capture
*c* was bound at sibling *i*". Node identity is recovered at emit time from the sibling array
(`kids[i]`, or its field child for a depth-1 step, via the `cap_depth`/`cap_field` map built in
`run_merge`). Consequences: continuations are small and fixed-cost to copy; the `TSQueryCapture`
array is materialised once per *emitted* match, which is also what the public API's interior
pointer (`query.c:4679`) requires anyway.

**Sparse beats dense.** A dense bitset over `(sibling × capture_id)` has width proportional to
the sibling count — 600 children → 38 words — so every clone and every subset test costs
O(sequence length). A sorted position vector has length bounded by captures-per-match (single
digits). Measured: 46.3 ms dense vs 25.9 ms sparse on the same input. Both give identical
results; keep the sparse one.

**Do not change the disambiguation policy.** It is what the tagged-automata literature would
do, and it is the highest-risk change available: the golden corpus measures **1.00× state
collapse on every real query file**, meaning real queries barely exercise the disambiguation
machinery, so a policy change would pass the whole test suite and surface downstream much
later. The bitset/vector approach avoids needing it.

**Merge by pointer, not by value.** `MIndex` is 256 KB; an early version swapped it by value
each sibling position and the speedup read as 0.0×.

## 7. The bug that will bite you again

`pattern_map` **already enumerates a pattern's alternative entry points** — the target query has
1 pattern and 3 map entries, built by the loop in `ts_query_new` that walks
`step->alternative_index` (`query.c:3092-3142`). Seeding from the map *and* expanding
`alternative_index` seeds each control state twice under different flags and emits **every match
exactly twice**. The stock engine seeds at `pattern->step_index` only, with
`seeking_immediate_match = true` (`ts_query_cursor__add_state`), and expands only after a match
advances.

Nothing in the representation says which enumeration is authoritative. This is the concrete
argument for the IR ([`06-compiler-architecture.md`](06-compiler-architecture.md)): the
semantics live in the interaction between fields, so a faithful reimplementation is archaeology
rather than translation.

## 8. How to run it

```sh
make -C tools/query-profiler spike

# correctness + speedup against the stock engine
SPIKE_SPARSE=1 SPIKE_PRUNE=1 ./target/query-profiler/merge_spike \
    tools/query-profiler/queries/two-quant.scm crates/cli/src/tests/query_test.rs

# scale the matcher without waiting on the stock engine (125 s at M=100)
SPIKE_NOSTOCK=1 SPIKE_SPARSE=1 SPIKE_PRUNE=1 ./target/query-profiler/merge_spike <query> <src>
```

| env var | effect |
|---|---|
| `SPIKE_SPARSE=1` | sorted position vectors (recommended); unset = dense bitset |
| `SPIKE_PRUNE=1` | enforce longest-match at merge time; unset = keep all continuations |
| `SPIKE_NOSTOCK=1` | skip the stock engine, and therefore the diff |

Output reports match counts for both engines, a capture-count histogram, `IDENTICAL match sets`
or the first divergence, and a speedup. Rust only (`tree_sitter_rust` is linked in `main`);
adding grammars means extending the `LANGDEF`/`SRCS` lists in the Makefile and the language
table in `main`.

## 9. Suggested order of work

1. **Replace `walk_parents` with the stock traversal** and re-measure the anchored cases (§2b).
   If the regression does not go away, stop — the design is not viable regardless of the 28×.
2. Delete the dead representations (§4) — halves the file.
3. Decide the §3 question: is 28× on ~18 real queries worth productionising?
4. If yes: model `has_in_progress_alternatives`, then extend past depth 1, re-diffing at each
   step against the stock engine on the corpus.
5. Attack the O(h²) prune scan (§5); re-measure the M-series to see whether the growth was it.
6. Only then consider merging into the real engine — and note that at that point the work is
   better done as part of the IR than as a patch to `query.c`.
