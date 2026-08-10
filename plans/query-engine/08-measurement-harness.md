# The measurement harness

Every number in this doc set came from a throwaway harness built during the exploration.
**Recommendation: land a cleaned-up version in-tree before starting any engine work.** Months
of performance work without a committed benchmark is not survivable, and the specific counters
below are the ones that made the diagnosis possible.

Current location (throwaway, will be garbage-collected):
`/tmp/claude-1000/-home-lillis-projects-tree-shitter/8e3ed4ec-1743-43db-bf72-ad82ee56e7d3/scratchpad/qprobe/`

## How it works

The `TSQuery` and `TSQueryCursor` structs are private to `query.c`, so the harness
**textually includes an instrumented copy of `query.c`** rather than linking against the
library. That gives full visibility into internals with no changes to the repo:

```c
#include "lib/src/alloc.c"
#include "lib/src/get_changed_ranges.c"
#include "lib/src/language.c"
#include "lib/src/lexer.c"
#include "lib/src/node.c"
#include "lib/src/parser.c"
#include "lib/src/point.c"
#include "query_instr.c"          // <- instrumented copy of lib/src/query.c
#include "lib/src/stack.c"
#include "lib/src/subtree.c"
#include "lib/src/tree_cursor.c"
#include "lib/src/tree.c"
```

Notes for reproducing:

- Use `#define _DEFAULT_SOURCE 1`, **not** `_POSIX_C_SOURCE` — the latter hides `le16toh` /
  `be16toh`, which `unicode.h` needs.
- `lib.c` cannot be used directly because it pulls in `wasm_store.c`. Include the individual
  files and stub the five wasm entry points:
  `ts_wasm_store_delete`, `ts_wasm_store_reset`, `ts_language_is_wasm`,
  `ts_wasm_language_retain`, `ts_wasm_language_release`.
- Link the grammar's `parser.c` (+ `scanner.c` where present) from
  `test/fixtures/grammars/<lang>/src/`.
- Compile with `-I lib/include -I lib/src` so the in-tree headers win. (On this machine
  `/usr/local/include` holds stale tree-sitter headers that shadow `lib/include`; explicit
  `-I` ordering avoids it. clangd still reports phantom errors — ignore them, gcc is fine.)

The instrumentation is applied by a small Python script that does ~12 uniquely-anchored string
replacements against a copy of `query.c`, each asserting exactly one match so that the patch
fails loudly if the source moves. That approach is worth keeping: it means the harness cannot
silently drift out of sync with the engine.

## Counters that earned their keep

These are the ones that produced the findings; a landed version should keep all of them.

| Counter | Where instrumented | What it revealed |
|---|---|---|
| `nodes_entered` | node-entry path (~4199) | denominator for everything |
| `states_started` | `ts_query_cursor__add_state` | how many threads a query spawns |
| `states_copied` | `ts_query_cursor__copy_state` | **fan-out factor — up to 34.6×** |
| `max_states` | node entry | **peak live states ≈ M², the core finding** |
| `state_visits` + `state_visits_depth_skip` | advance loop (~4263) | **99.1% of the scan is wasted** |
| `dedup_compares` + `dedup_capture_steps` | `compare_captures` | **322 M inner steps — the cliff** |
| `sort_compares` | `state_precedes` | insertion-sort cost per node |
| `pool_acquires` + `pool_scan_steps` | `capture_list_pool_acquire` | **avg free-slot scan 191.6** |
| `descend_scan_steps` | `should_descend` | (0 on tested workloads; matters for range queries) |
| `capture_drops` | `query_step__add_capture` | **found defect A1** |
| `matches` | `push_finished_state` | output size, for output-sensitivity ratios |

Compile-phase timers, equally important:

| Timer | What it revealed |
|---|---|
| `qp_t_analyze_ms` | **analysis is 96–99% of `ts_query_new`** |
| `qp_t_subgraph_scan_ms` | the full parse-table scan — **the query-independent floor** |
| `qp_t_perform_ms` + call/iteration counts | per-pattern analysis cost |
| `qp_predecessor_map_bytes` | **1.9 MiB `calloc` per compile on Rust** |
| `qp_analysis_aborts` | 0 on all tested queries — relevant to suspicion B1 |

Static structure dump (no instrumentation needed, just struct access): pattern/step counts,
`pass_through`/`dead_end` fractions, guarantee fractions, captures-per-step histogram,
`sizeof(QueryStep)`.

## What a landed version should add

1. **A committed corpus.** Fixture grammars + real query files + representative source files,
   with sizes recorded. The `crates/cli/benches/benchmark.rs` harness already walks
   `test/fixtures/grammars/*/queries/`; extend rather than duplicate it.
2. **The pathological query set — as measured, not as assumed.** Rechecked against wall
   clock and match counts; only one of the original four survives:

   | query | matches | mean delay | verdict |
   |---|---|---|---|
   | `((attribute_item)* @attr (line_comment)* @doc (function_item …) @fn)` unanchored | 22,100 | 0.190 ms, grows 3× | **genuinely pathological** — 34× the anchored per-match cost |
   | `((line_comment)* @doc (function_item) @fn)` | 128 | 0.067 ms, flat | fine |
   | `(block (_)* @stmt)` | 289 | 0.037 ms, flat | fine — only 1.3× the trivial `(block) @b` |
   | `(_) @any` | 21,316 | 0.0005 ms, flat | excellent |

   Keep all four as regression tests, but only the first is an acceptance test for the
   matcher rewrite. The others are guards against regressing something that currently works.

3. **A methodological warning, learned the hard way.** Three of those four were called
   pathological on the strength of *internal counter magnitudes* — 23× state fan-out, 98.4%
   of state visits failing the depth test, 322 M dedup inner steps — without checking wall
   clock or counting the matches. Counters localise where time *could* go; they do not
   establish that it does. `(block (_)* @stmt)` has 23× fan-out and costs 1.3×.

   **Every counter-based claim in this doc set should be paired with a wall-clock or
   match-count check before it is acted on.** The `dedup_capture_steps / matches` ratio
   proposed below is subject to the same caveat: validate it against wall clock before
   treating it as an acceptance metric.
4. **An output-sensitivity assertion.** The most useful single metric is
   `dedup_capture_steps / matches`. Today it is ~40,000:1 on the pathological case. A healthy
   engine keeps it bounded by a small constant. Track it explicitly; it is the number that
   says whether the rewrite achieved its goal.
5. **Differential mode.** Run old and new engines and diff the match streams (pattern index,
   capture ids, node ranges). This is the single most important piece of infrastructure for
   the whole project — see [`09-roadmap.md`](09-roadmap.md).

   Build it as **a second backend for `crates/cli/src/tests/query_test.rs`**, not as a
   standalone corpus runner. That suite (125 tests, ~6,500 lines) is already the authoritative
   safety net; the rig's job is to run it twice and compare, plus sweep the 39 real query
   files already in `test/fixtures/grammars/*/queries/`. Adding a competing test mechanism
   with its own corpus would split authority over "what is correct" — the one thing a
   single-maintainer project cannot afford.

## Reproducing the headline numbers

```
# compile-cost floor: a 16-byte query on the Rust grammar
qprobe rust tiny.scm  <any.rs> match       # => 2.90 ms, 96% full parse-table scan

# the cliff: M attributed functions
for M in 25 50 100; do gen_rust_file $M > run.rs; qprobe rust path4.scm run.rs match; done
# => 139 ms / 4,162 ms / >120 s ; peak live states 624 / 2,499 / -
```

where `path4.scm` is
`((attribute_item)* @attr (line_comment)* @doc (function_item name: (identifier) @name) @fn)`
and the generator emits, per function, two `#[...]` attributes, three `//` comments, and a
one-line `fn`.
