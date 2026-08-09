# Data-oriented design in the query engine

The `generate` crate's DoD pass (`plans/ACTION_POOL_DESIGN.md`, and the `dod-ify` line of work
— -61.2% wall / -73.1% RSS vs master) established both the method and the appetite. The same
method applies here, but the conclusions are different, and the difference is worth being
precise about: **in the query engine, one structure is a genuine DoD target and the others are
better fixed algorithmically.**

The discipline that makes DoD work is the one from the action-pool doc: measure the actual
access pattern first, then choose granularity deliberately. Applying it here gives a clear
ranking rather than a blanket "make everything SoA".

## The real target: capture lists in the dedup loop

Verified layout:

```c
typedef struct TSNode {          // 32 bytes
  uint32_t context[4];           //   16
  const void *id;                //    8
  const TSTree *tree;            //    8
} TSNode;

typedef struct TSQueryCapture {  // 40 bytes with padding
  TSNode node;
  uint32_t index;
} TSQueryCapture;
```

`ts_query_cursor__compare_captures` (`query.c:3682-3737`) is the hottest loop in the engine —
**321,927,770 inner iterations** on `queries/two-quant.scm` over a 195 KB file
([`04-performance.md`](04-performance.md)). Each iteration reads two captures and uses:

- `left->node.id == right->node.id` (8 bytes)
- `left->index == right->index` (4 bytes)
- `ts_node_start_byte` / `ts_node_end_byte` on both (from `context[]`)

That is on the order of **12 useful bytes out of 80 loaded**. At 40 bytes per element, a
64-byte cache line holds 1.6 captures. The comparison is pure pointer-chasing over a structure
laid out for the *public API's* convenience, in the one place where density matters most.

**The DoD move:** store capture lists internally as parallel arrays — a `{start_byte,
end_byte}` array (8 bytes/element, 8 per cache line) for the comparison keys, a separate
`uint32_t index[]`, and the full `TSNode[]` only for materialization. The dedup loop then
streams 8 bytes per element instead of 40, a **5× density improvement on a 322M-iteration
loop**.

### The constraint that shapes it

```c
match->captures = captures->contents;   // query.c:4679, and again at :4834
```

The public API hands the caller a **raw interior pointer into the pool's capture array**. So
whatever the internal representation, a contiguous `TSQueryCapture[]` must exist at the moment
a match is returned.

That is not fatal — keep SoA internally and materialize AoS when a match is actually returned.
The cost is O(captures in that match), paid once per *returned* match; the saving is on a
quadratic loop over *candidate* matches. On the pathological case that is ~7,700 returned
matches against 322M comparison steps, so the trade is heavily favourable. But it must be
measured, not assumed, and it means capture storage cannot be redesigned without touching the
return path.

This is also the reason a naive "one big arena of captures" does not work: a reallocating
arena would invalidate pointers already handed out. A chunked arena that never moves existing
data would.

## Where DoD loses to fixing the algorithm

Two structures look like obvious DoD candidates and are not, and it is worth saying why.

**`QueryStep` (20 bytes, AoS).** The advance loop's first action is a depth test
(`query.c:4263`) needing only `step->depth`. Splitting depth into its own `uint16_t[]` would
give 32 depths per cache line instead of 3.2 whole steps — a 10× density win on the hottest
filter in the per-node loop.

Except: **P3 (bucket live states by `start_depth + step->depth`) removes that scan entirely.**
99.1% of those visits fail the depth test on the pathological case; the answer is not to make
the failing test cheaper, it is to not perform it. Doing the SoA split as well would be work
spent making a loop dense that should not exist.

**`QueryState` (20 bytes), scanned linearly per node.** Same argument, same fix.

The general rule this suggests, and it is the same one the action-pool work followed: *DoD is
for the work you have established you must do.* Where the measurement says the work itself is
unnecessary, remove it first — then decide whether what remains needs a layout change. Getting
that order wrong produces beautifully packed structures for loops that should have been
deleted.

## Consequences for P2, the immediate next change

`capture_list_pool_acquire` (`query.c:483-505`) finds a free list by linear scan — average
**191.6 steps per acquire** on the pathological case. Two ways to scope the fix:

**(a) Minimal — intrusive free list.** Thread a singly-linked free list through the unused
entries (a list is free iff `size == UINT32_MAX`, so the `contents` pointer slot is available
to hold the next free index). O(1) acquire and release, ~20 lines, no API surface, no semantic
surface.

**(b) DoD — restructure capture storage.** `Array(CaptureList)` is an array of *independently
heap-allocated* arrays: reaching capture *j* of state *i* is two pointer hops, and each state's
list grows on its own. Replacing that with a chunked arena plus SoA keys is the change
described above.

**Recommendation: (a) now, (b) later, and deliberately.** (a) is the change that proves the
whole loop — profiler, goldens, review, land — works end to end, and it is correct on its own
terms regardless of what (b) eventually does. (b) wants to be designed *together with* the
SoA question and the new matcher's needs, because all three touch the same structure; bolting
it on now would mean designing capture storage twice. (a) does not preclude (b) in any way.

## Where DoD should be a standing input

For the phases that write new code rather than patch old:

- **The bytecode** ([`06-compiler-architecture.md`](06-compiler-architecture.md)) is a fresh
  layout decision with no legacy constraint. Instruction stream density directly determines
  dispatch cost. Design it packed from the start — variable-length operands, hot fields first,
  cold metadata (spans, provenance) in side tables keyed by instruction index rather than
  inline. The side-table split is also what keeps the bytecode stable when only diagnostics
  change.
- **The analysis cache** (P1) is a pure side table keyed by language — the natural DoD shape
  already.
- **The new matcher's state representation** is the highest-leverage layout decision in the
  project, because it is what the innermost loop touches. It should be designed with the
  measured access pattern in hand, which is what the Phase 1.5 spike is for.

## What to measure

The profiler already reports the counters that matter. For any DoD change here, the numbers to
move are:

| Metric | Today (two-quant.scm / 195 KB) | What a win looks like |
|---|---|---|
| `dedup_capture_steps` | 321,927,770 | unchanged by layout — this is P7's job |
| wall time per dedup step | — | **this** is what SoA moves; needs `perf stat` for cache misses |
| `pool_scan_steps` per acquire | 191.6 | 0 (free list) |
| `state_visits` rejected by depth test | 99.1% | ~0 (bucketing, not layout) |

Note the first row: **SoA does not reduce the number of comparisons, only their cost.** The
count is an algorithmic problem and belongs to the disambiguation rewrite. Conflating the two
would let a 2× constant-factor win disguise the fact that the loop is still quadratic.
