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
