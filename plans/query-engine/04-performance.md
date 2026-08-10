# Performance: cost model, measurements, ranked opportunities

Measured at `003b10c28`, `gcc -O2`, WSL2 / Linux 6.6, single run each (variance on repeated
runs was under 5% on the numbers that matter; the 675 ms case reproduced at 690 ms and
663 ms).

## Baseline: real queries on real files

| language | query | compile | parse file | execute | file |
|---|---|---|---|---|---|
| rust | highlights.scm (3.5 KB) | **16.95 ms** | 21.33 ms | 12.89 ms | query_test.rs, 195 KB, 42,904 nodes |
| javascript | highlights.scm (2.7 KB) | **8.04 ms** | 2.30 ms | 1.48 ms | playground.js, 20 KB, 6,006 nodes |
| c | highlights.scm (1.4 KB) | **3.25 ms** | 18.19 ms | 7.51 ms | query.c, 171 KB, 34,376 nodes |
| python | highlights.scm (2.0 KB) | **5.02 ms** | — | — | grammar.js, 27 KB |
| go | highlights.scm (1.4 KB) | **1.63 ms** | — | — | grammar.js, 24 KB |

Execution itself is respectable: rust/highlights processes 42,904 nodes in 12.89 ms
(~300 ns/node including capture allocation), and that is with `next_capture`, the expensive
mode. **The problem is not the common path.** The problems are (a) compilation, and (b) a
cliff that specific query shapes fall off.

## Problem 1: compilation is dominated by query-independent work

`ts_query_new` is **96–99% `ts_query__analyze_patterns`** across every query measured. Within
that, the breakdown for Rust:

| query | total compile | of which: full parse-table scan | of which: `perform_analysis` |
|---|---|---|---|
| `(identifier) @v` (16 B) | 2.90 ms | **2.77 ms (96%)** | 0.00 ms (0 calls) |
| `(call_expression function: (identifier) @f)` (44 B) | 3.26 ms | 2.70 ms | 0.44 ms (1 call) |
| 4 simple patterns (118 B) | 3.44 ms | 2.59 ms | 0.64 ms (2 calls) |
| highlights.scm (3,527 B) | 16.26 ms | 2.72 ms | 12.96 ms (30 calls, 282 iters) |

**The parse-table scan is a constant ~2.7 ms for the Rust grammar regardless of the query.**
It is a floor on `ts_query_new`. Per language:

| language | parse states | `predecessor_map` `calloc` | scan floor |
|---|---|---|---|
| rust | 3,823 | 1,918 KiB | 2.72 ms |
| python | 2,809 | 1,409 KiB | 1.79 ms |
| c | 2,015 | 1,011 KiB | 1.52 ms |
| javascript | 1,870 | 938 KiB | 1.39 ms |
| go | 1,442 | 723 KiB | 0.70 ms |

The allocation is `ts_calloc(state_count * (MAX_STATE_PREDECESSOR_COUNT + 1), sizeof(TSStateId))`
(`query.c:1016-1019`) — `state_count × 257 × 2` bytes, zeroed, per `ts_query_new`.

### What is actually language-only

Of the analysis work in `ts_query__analyze_patterns`:

| step | depends on | cacheable per language? |
|---|---|---|
| predecessor map construction (1768-1840, shift branch) | language only | **yes, fully** |
| subgraphs for hidden symbols (1754-1758) | language only | **yes, fully** |
| subgraph node collection (1768-1840, reduce branch) | language + which symbols are in the query | mostly — the scan is over the whole table either way |
| backward closure over predecessors (1844-1882) | same | mostly |
| `perform_analysis` (1307-1613) | query | no |
| fallibility propagation (2052-2087) | query | no |

So the floor is removable. Two options, in increasing order of ambition:

- **P1 (cheap): cache it on the `TSLanguage`.** Compute the predecessor map + hidden-symbol
  subgraphs once per language, refcounted alongside `ts_language_copy`. Cost: one lazily
  built side table per language; benefit: the entire floor disappears for the 2nd..Nth query
  compiled against that language. For an editor with 3–4 query files per language this is a
  3–4× reduction in total query-compile time immediately, more if injections cause repeated
  compiles.
- **P2 (better): precompute at grammar-generation time.** The predecessor map and the
  hidden-symbol subgraph structure are pure functions of the parse table, which
  `crates/generate` already has in memory when it emits `parser.c`. Emitting them as static
  data removes the cost *and* the allocation entirely, at the price of parser.c size and an
  ABI addition. Worth measuring the size delta before committing — 1.9 MiB of predecessor map
  for Rust is not something you want in the binary uncompressed, so this needs a compact
  encoding (the map is sparse; most states have few predecessors).

P1 is the right first move: contained, no ABI change, no semantic risk.

### `perform_analysis` is the other 80% on real queries

12.96 ms of Rust highlights' 16.26 ms, across 30 calls and 282 iterations. This is per-pattern
and genuinely query-dependent, but:

- it is **per-pattern independent** — 30 calls that share nothing but the subgraphs. Trivially
  parallelizable, and more usefully, **cacheable per (language ABI hash, pattern text)**.
- for a highlights.scm that changes once a year and a grammar that changes rarely, ~all of
  this is recomputed on every editor start.
- it could be **lazy**: the only consumers of the result are `next_capture` streaming and
  fallible-step splitting. A cursor using `next_match` never reads `root_pattern_guaranteed`.
  Deferring analysis until the first `next_capture` call would make `next_match`-only
  workloads (most non-editor uses: linters, codemods, `tree-sitter query`) skip it entirely.

## Problem 1b: the compile cliff is parent-subgraph breadth, not query size

Compilation and execution are **separate budgets with separate pathological inputs**, and they
should be optimized separately. The one place they couple is the analysis abort — see below.

Compile cost is well-behaved along the axes you would expect:

| axis | behaviour |
|---|---|
| pattern count | **linear**, ~0.44 ms per *analyzed* pattern (1 → 256 patterns: 0.49 → 113.73 ms) |
| nesting depth | **linear**, ~0.67 ms per level (139 analysis iterations per level) |
| sibling width | saturates — for `(source_file (function_item) × W)`, flat from W=7 through W=16 |

The cost is driven by something else entirely: **how much of the parse table the pattern's
parent symbol can span.** Same shape, six identical children, one grammar:

| pattern | perform_analysis | set inserts | peak set |
|---|---|---|---|
| `(declaration_list (function_item) × 6)` | 0.35 ms | 3,408 | 10 |
| `(source_file (function_item) × 6)` | 0.70 ms | 2,563 | 11 |
| `(field_declaration_list … × 6)` | 2.20 ms | 10,933 | 136 |
| `(block (let_declaration) × 6)` | 10.66 ms | 65,862 | 52 |
| `(parameters (parameter) × 6)` | 11.38 ms | 95,497 | 155 |
| **`(arguments (identifier) × 6)`** | **68.22 ms** | **342,573** | **444** |

**A 195× spread for the same query shape.** `arguments` can contain any expression, so its
analysis subgraph spans a large fraction of the parse table and every hypothetical child
multiplies the reachable state set. `declaration_list` admits only items, so it barely moves.

Growth for the expensive parent, with the iteration cap lifted to 20,000:

| W | perform_analysis | inserts | peak set |
|---|---|---|---|
| 1 | 0.31 ms | 1,562 | 11 |
| 2 | 1.12 ms | 6,877 | 36 |
| 3 | 1.34 ms | 8,847 | 40 |
| 4 | 6.30 ms | 36,476 | 107 |
| 5 | 25.26 ms | 137,579 | 254 |
| 6 | 68.87 ms | 342,573 | 444 |
| 7 | 127.67 ms | 617,034 | 575 |
| 8 | 126.20 ms | 614,850 | 594 |

Roughly 3–4× per added child through the middle, then saturating near W=7 as the analysis
state space is exhausted. Not unbounded — but 127 ms for **one pattern** is already a cliff,
and a query file with a handful of such patterns is seconds.

### Re-verified against wall clock: the cliff is capped, and the spread is 4.7x

The table above was measured with the iteration cap **raised**, and an earlier draft omitted
that caveat. Re-measured uninstrumented, at the shipped `MAX_ANALYSIS_ITERATION_COUNT = 256`,
compiling the same six-child patterns (min of 10 reps):

| pattern | compile |
|---|---|
| `(field_declaration_list … x 6)` | 1.58 ms |
| `(declaration_list (function_item) x 6)` | 1.72 ms |
| `(source_file (function_item) x 6)` | 2.22 ms |
| `(block (let_declaration) x 6)` | 3.72 ms |
| `(arguments (identifier) x 6)` | 7.17 ms |
| `(parameters (parameter) x 6)` | 7.48 ms |
| **`rust/highlights.scm` (94 patterns)** | **16.07 ms** |

**The spread at shipped settings is 4.7x, not 195x** — the cap truncates the expensive cases.
And a real query file costs more than any single pathological pattern. So the user-visible
compile story is *a large constant per pattern* (~0.17 ms), not a cliff. The cliff is real but
only reachable by removing the containment.

### The abort-degrades-execution claim: weak, and previously overstated

An earlier draft asserted that a pattern hitting the abort is "penalised twice" and "runs
slower forever after", because `did_abort` marks every step fallible. **Measured, that is
barely true.** Same query, same input, 400 matches on both sides
(`(block (let_declaration) x 6)` over 400 functions with 8 lets each):

| mode | cap = 256 (aborts) | cap = 20000 (completes) |
|---|---|---|
| `next_match` | min 4.43, mean 4.66 ms | min 3.52, mean 3.93 ms |
| `next_capture` | min 4.52, mean 4.75 ms | min 4.30, mean **5.16** ms |

Roughly 15% in `next_match`; in `next_capture` the means move in the *opposite* direction and
the result is within noise. Also note **no real query file in the fixture corpus aborts at
all** — all five `highlights.scm` measured zero aborts. So this coupling between the compile
and execution budgets, which an earlier draft leaned on, is weak-to-inconclusive and should
not be used to justify work.

### Why: the cost is inherent to what the analysis computes

Three hypotheses for the blow-up, each tested against `(arguments (identifier) × 6)` and each
**rejected by measurement**. Recorded because the negative results are what rule out the cheap
fixes and force the architectural one.

Measured shape of the hot loop (uncapped, one pattern): **4,649,167 lookahead-symbol
iterations**, producing 366,798 candidate states, of which 342,573 reach the sorted set and
86% of those are already present. Roughly 12,700 state-processings × ~370 lookahead symbols
each.

| hypothesis | prediction | measured | verdict |
|---|---|---|---|
| The `array_search_sorted_with` over `subgraph->nodes`, run 4.6 M times, dominates | removing it should be a large win | a one-element memo eliminated **45%** of those searches (4.65 M → 2.57 M) and moved wall time **63.71 → 63.34 ms, 0.6%** | **rejected** — the search is not the cost |
| Many lookahead symbols are redundant; grouping by their *effect* `(successor, visible_symbol)` collapses the loop | large redundancy factor | 93 raw symbols per state-processing → **80 distinct effects**, a redundancy of only **1.2×** | **rejected** — each symbol genuinely leads to a different outcome |
| Collapsing merely by `successor` (ignoring which symbol) helps | large factor | 93 raw → **44.3 distinct successors** per processing | **~2× at best**, and only for the non-matching symbols |

So the loop is not doing redundant work per symbol. The redundancy is further downstream —
86% of *constructed* states are duplicates — because many distinct `(successor, symbol)` pairs
collapse to the same `AnalysisState` once `does_match` comes out false and only
`(parse_state, child_index, field_id, done, step_index)` is retained.

The conclusion that matters: **`perform_analysis` explores the cross product of hypothetical
tree positions × possible next tokens, and for a broad parent symbol that product is genuinely
large.** There is no cheap trick inside the loop. A large win requires changing *what* the
analysis computes or *when* it runs — not optimizing how it runs.

That reorders the compile-side work:

1. **Laziness — skip analysis when nothing consumes it.** Now clearly the biggest lever. A
   `next_match`-only consumer never reads `root_pattern_guaranteed`. Avoiding an inherently
   expensive computation beats speeding it up. Caveat: analysis also rejects impossible
   patterns, so this changes when that error surfaces.
2. **Caching** — by `(language)` for the parse-table half (P1), and by
   `(language ABI, pattern text)` for `perform_analysis`, which is per-pattern and independent.
3. **`insert_sorted`** — the 15–20% constant factor, worth having, once the above are in.
4. **Micro-optimizing the lookahead loop** — measured ceiling of ~2×, and only on the
   pathological shapes. Lowest priority.

### Correction to an earlier draft

An earlier version of this document reported "width 9 does not finish in 45 seconds". That was
an artifact: `w9.scm` and `w11.scm` were never generated, and a `|| echo TIMEOUT` fallback in
the measurement script turned *file not found* into an apparent timeout. There is no width
cliff on `source_file`; the real driver is parent-subgraph breadth, measured above. Recorded
here because the failure mode — a harness reporting a missing input as a result — is one to
watch for in the benchmark suite.

## Problem 2: the execution cliff

Full analysis in [`02-execution-model.md`](02-execution-model.md). Summary of the cost model:

```
per node entered:
    O(wildcard_root_patterns)              linear scan, 4205-4224
  + O(log patterns) + O(matching roots)    binary search, 4227-4252
  + O(live_states)                         advance loop, 4255-4505      <-- 99% wasted when hot
  + O(live_states log live_states)         insertion sort, 4509
  + O(group_size² × capture_list_len)      dedup, 4511-4600             <-- the cliff
  + O(pool_size) per capture acquire       free-slot scan, 483-505      <-- secondary
```

Measured on `((attribute_item)* @attr (line_comment)* @doc (function_item name: (identifier) @name) @fn)`
over query_test.rs (42,904 nodes):

```
exec                 675.57 ms          (vs 21 ms to parse the same file)
state_visits         5,742,580          133.8 per node, 99.1% rejected by the depth test
dedup inner steps    321,927,770        7,503 per node
sort compares        5,711,796          133.1 per node
pool scan steps      2,413,286          avg 191.6 per acquire
peak live states     369
```

And the synthetic scaling (M attributed functions):

| M | peak live | wall | ratio |
|---|---|---|---|
| 25 | 624 | 139.60 ms | — |
| 50 | 2,499 | 4,161.81 ms | 29.8× |
| 100 | — | > 120 s | > 29× |

`peak live ≈ M²`. Wall ≈ M⁴·(constant), tempered by the early-break optimizations.

## Ranked opportunities

Ordered by (value × confidence) / risk. The first four are all *semantics-preserving* and can
land independently of any rewrite.

| # | Change | Expected effect | Risk | Where |
|---|---|---|---|---|
| **P1** | Cache the language-only analysis on `TSLanguage` | −0.7 to −2.8 ms on every `ts_query_new` after the first for that language | low — pure memoization, no semantic surface | `query.c:1012-1051`, `1767-1882` |
| **P2** | Free-list in `CaptureListPool` | O(1) acquire; kills a 191-step average scan on hot queries | very low — internal, ~20 lines | `query.c:483-511` |
| **P3** | Bucket live states by `start_depth + step->depth` | removes 99% of the advance-loop scan on hot queries | low-medium — must preserve the array ordering the dedup group-break relies on | `query.c:4255-4263` |
| **P4** | Make analysis lazy (skip unless `next_capture` is used) | `next_match`-only workloads skip 96–99% of compile | low — but changes when errors surface, since analysis also rejects impossible patterns; needs care | `query.c:3196` |
| **P5** | Intern capture/predicate names via a hash map | `symbol_table_id_for_name` is O(n) `strncmp`; O(n²) to build a 21-capture table is nothing, but it is on the parse path and trivially fixed | very low | `query.c:920-933` |
| **P6** | Precompute predecessor map at grammar-generation time | removes the floor entirely, and the multi-MB allocation | medium — ABI addition, binary size | `crates/generate/src/render.rs` |
| **P7** | Replace post-hoc dedup with in-automaton disambiguation | removes the cliff; makes the engine output-sensitive | **high — this is the semantics change** | the rewrite |
| **P8** | Predicate pushdown (requires text access in the cursor) | order-of-magnitude on `tags.scm` / `locals.scm` shapes | high — API change, binding coordination | the rewrite |

P1–P5 are, collectively, a few days of work with a test suite in front of them, and they
address every measured cost except the cliff. **P7 is the only one that needs the rewrite**,
and it is the one that matters most — but it should be attempted only after the disambiguation
spec exists ([`03-correctness.md`](03-correctness.md) §C1) and a differential-testing rig is
running ([`09-roadmap.md`](09-roadmap.md)).

## What not to bother with

- **Micro-optimizing `QueryStep` size.** It is already 20 bytes and the steps array is 2–5 KiB
  for real queries. It is not a memory problem and it is not a cache problem at that size.
- **Parallelism inside a single query execution.** The tree walk is inherently sequential and
  the state list is shared. Parallelism belongs at the *compile* stage (independent
  `perform_analysis` calls) and at the *file* level, not inside `advance`.
- **SIMD anything.** There is no data-parallel inner loop here; the hot loop is pointer-chasing
  over capture lists. Fix the algorithm first; there may be a vectorizable comparison
  afterwards, but not before.
- **The `dead_end` step overhead (41% of steps on alternation-heavy queries).** It costs step
  array space, not time — dead-end steps are jumped through in one iteration at 4439-4443.
  A real IR removes them for clarity, not for speed.
