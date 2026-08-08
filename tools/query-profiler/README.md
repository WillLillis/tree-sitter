# query-profiler

Static and dynamic profiling of the query engine. Built to answer "where does the time
actually go in `ts_query_new` and `ts_query_cursor__advance`", which is not answerable from
the outside because `TSQuery` and `TSQueryCursor` are private to `lib/src/query.c`.

## How it works

`patch.py` reads `lib/src/query.c` and writes `query_instr.c` — the same file plus counters
and phase timers. The drivers `#include` that generated copy along with the rest of the
library sources, so they see the internals directly. Nothing in `lib/` is modified.

Every anchor in `patch.py` asserts it matched **exactly once**, so the patch fails loudly if
`query.c` moves rather than silently mis-instrumenting. If you change `query.c` and this stops
building, that is the tool working.

`query_instr.c` is generated and gitignored. Do not edit it.

## Build and run

```sh
make -C tools/query-profiler          # builds target/query-profiler/{qprobe,bugs}
make -C tools/query-profiler check    # builds, then runs the correctness repros
```

Needs the fixture grammars in `test/fixtures/grammars/<lang>/src/parser.c` (gitignored;
fetched by the normal test setup). Only the grammars present are linked — currently
javascript, rust, python, go, c.

```sh
# qprobe <lang> <query.scm> <source-file> <match|capture>
./target/query-profiler/qprobe rust \
    test/fixtures/grammars/rust/queries/highlights.scm \
    crates/cli/src/tests/query_test.rs capture

# the known-pathological set
./target/query-profiler/qprobe rust tools/query-profiler/queries/doc-run.scm  <file.rs> match
./target/query-profiler/qprobe rust tools/query-profiler/queries/two-quant.scm <file.rs> match
```

`bugs` takes no arguments; it compiles a fixed set of queries against the Rust grammar and
prints four reproductions of silent-data-loss defects.

## What it reports

**Static** (per compiled query): pattern/step counts, how many steps are control-flow-only
(`pass_through` / `dead_end`), the captures-per-step histogram, what fraction of steps the
analyzer proved guaranteed, `sizeof(QueryStep)`.

**Compile phases**: total `ts_query_new`, the share spent in `ts_query__analyze_patterns`,
and within that the full parse-table scan vs `ts_query__perform_analysis` (with call and
iteration counts), plus the size of the `predecessor_map` allocation.

**Execution counters**: nodes entered, states started/copied (fan-out), peak live states,
state visits and how many are rejected by the depth test alone, dedup comparisons and inner
steps, sort comparisons, capture-pool acquires and free-slot scan length.

The single most diagnostic ratio is **`dedup_capture_steps / matches`** — work per unit of
output. On the pathological query below it exceeds 40,000:1. A healthy engine keeps it near a
small constant; it is the acceptance metric for any rewrite of the matching algorithm.

## Reproducing the findings

```sh
# Compile cost is ~99% analysis, and most of it does not depend on the query.
printf '(identifier) @v\n' > /tmp/tiny.scm
./target/query-profiler/qprobe rust /tmp/tiny.scm crates/cli/src/tests/query_test.rs match
# => ~2.9 ms, of which ~96% is the full parse-table scan; 1918 KiB predecessor map
```

For the quadratic-blowup case, generate M attributed functions and run `queries/two-quant.scm`
over them:

```sh
python3 -c 'M=50
for i in range(M):
    print("#[allow(dead_code)]"); print("#[inline]")
    for j in range(3): print("// doc %d %d" % (i, j))
    print("fn f%d() { let x = 1; }" % i)' > /tmp/run.rs
./target/query-profiler/qprobe rust tools/query-profiler/queries/two-quant.scm /tmp/run.rs match
```

Measured at `003b10c28`: peak live states ≈ M² (624 at M=25, 2,499 at M=50); wall 140 ms at
M=25, 4.2 s at M=50, >120 s at M=100.

## Caveats

- Timings are single-run wall clock, no warmup. Fine for the order-of-magnitude effects this
  was built to find; not a benchmark harness. Treat sub-2x differences as noise.
- The counters add branches to hot loops, so absolute wall times here run slightly above an
  uninstrumented build. Ratios and counts are unaffected.
- On machines with stale tree-sitter headers in `/usr/local/include`, clangd may report
  phantom errors in these files. The Makefile's `-I` ordering means the actual build is fine.
