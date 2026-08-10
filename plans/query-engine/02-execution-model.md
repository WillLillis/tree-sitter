# The execution model, and why it falls over

## What the VM is

`ts_query_cursor__advance` (4030-4645) is a **Thompson/Pike-style NFA simulation driven by a
depth-first tree walk**. It is worth naming that explicitly, because once you do, the entire
regex-engine literature becomes applicable (see [`07-references.md`](07-references.md)).

- The "input string" is the pre-order traversal of the syntax tree, with depth bookkeeping.
- A `QueryState` is a **thread**: `{step_index, start_depth, capture_list_id, flags}`.
- `self->states` is the thread list. `self->finished_states` is the accept queue.
- Threads split at `alternative_index` (4431-4504) — `SPLIT` in Pike VM terms.
- Threads die when they cannot make progress (4095-4109, 4322-4337).

The crucial difference from a Pike VM, and the source of every problem below:

> **A Pike VM keeps at most one thread per NFA state. This VM keeps one thread per
> *(NFA state × capture history)* pair, and deduplicates them afterwards by comparing
> capture sets pairwise.**

A Pike VM is O(program × input) because merging threads at the same PC is sound — the
disambiguation policy (leftmost-first, or leftmost-longest) decides which thread's captures
survive at merge time, in O(1). This engine has no such policy. Instead it lets duplicate
threads accumulate and then runs `ts_query_cursor__compare_captures` (3682-3737) over pairs,
keeping whichever has a superset of captures. That is where the blowups live, and it is also
where the *semantics* live — which is the deeper problem.

## The per-node loop

For each visible node entered (4170-4626), in order:

1. **Start wildcard-root patterns** (4205-4224) — linear over `wildcard_root_pattern_count`.
2. **Start symbol-root patterns** (4227-4252) — binary search into `pattern_map`, then walk
   the run of equal symbols.
3. **Advance every live state** (4255-4505) — **linear over all live states**, with the depth
   test as the first filter.
4. **Sort states by capture position** (4509) — insertion sort, `ts_query_cursor__sort_states_by_capture`.
5. **Longest-match dedup** (4511-4625) — pairwise `compare_captures` within
   `(start_depth, pattern_index)` groups, with two early-break optimizations.

Steps 3 and 5 are both unindexed linear/quadratic scans over the thread list.

### Step 3 is almost entirely wasted work

The first thing the loop does after fetching a state is:

```c
if ((uint32_t)state->start_depth + (uint32_t)step->depth != self->depth) continue;   // 4263
```

A state can only match at one specific depth, and the loop discovers this by checking every
state at every node. Measured fraction of state visits rejected by that test alone:

| workload | state visits / node | rejected by depth test |
|---|---|---|
| c/highlights over query.c | 0.7 | 0.0% |
| rust/highlights over query_test.rs | 1.3 | 11.3% |
| `(block (_)* @stmt)` over query_test.rs | 5.2 | **98.4%** |
| `(attr* doc* fn)` over query_test.rs | 133.8 | **99.1%** |

On well-behaved queries this is fine. On the queries that are already slow, it is 99% of the
scan. The fix is structural and cheap: **bucket the thread list by
`start_depth + step->depth`**, so entering a node at depth *d* touches only the threads that
could possibly match at depth *d*. This is an index over the engine's own state, and it does
not change semantics at all.

### Step 5 is the real cost, and it is a semantics problem wearing a performance costume

`ts_query_cursor__compare_captures` walks two capture lists in lockstep to decide whether
either is a subset of the other (3682-3737). It is called from a doubly-nested loop over
live states (4511-4600).

Two optimizations already exist in this tree and both help:

- the group break on `(start_depth, pattern_index)` (4533-4536);
- the disjointness break — once `other`'s first capture starts at or after `state`'s last
  capture ends, nothing later in the group can overlap (4544-4555, commit `91fd04f96`).

They are not enough, because the group itself grows quadratically. See below.

## The blowup, precisely

### Reproduction

```scheme
((attribute_item)* @attr (line_comment)* @doc (function_item name: (identifier) @name) @fn)
```

Input: M top-level Rust functions, each preceded by 2 attributes and 3 line comments
(~6M lines total). Measured:

| M | peak live states | states copied | dedup compares | dedup inner steps | wall |
|---|---|---|---|---|---|
| 25 | 624 | 7,075 | — | 78,191,675 | 139.60 ms |
| 50 | 2,499 | 37,900 | — | 2,351,292,100 | 4,161.81 ms |
| 100 | — | — | — | — | **> 120 s** |

`peak live states ≈ M²` — 624 ≈ 25², 2,499 ≈ 50². Wall time grows ~30× per doubling of M.
The query yields roughly M matches.

### Correction: most of this is the answer set, not the algorithm

An earlier draft presented this as the flagship example of the engine being non-output-
sensitive — "~50 matches after 2.35 billion units of work". **That was wrong**, and the error
was mine: I never counted the matches. The unanchored pattern returns **22,100** matches at
M=50, not ~50.

Anchoring it (`.` between the quantified runs) changes everything:

| M | unanchored | matches | anchored | matches |
|---|---|---|---|---|
| 25 | 137.8 ms | 2,925 | 0.145 ms | 25 |
| 50 | 3,904 ms | 22,100 | 0.270 ms | 50 |
| 100 | > 60 s | — | 0.566 ms | 100 |

The anchored form is **linear** and returns exactly M matches. On the real 195 KB corpus file
the same two characters take 738 ms → **7.07 ms**, a 104× speedup, and 8,003 → 128 matches.

A hand-written structural-join prototype over the same tree
(`tools/query-profiler/join_proto.c`) computes the anchored answer in **9.05 ms** — *slower*
than the engine's 7.07 ms. So for anchored patterns there is **no order-of-magnitude prize
available in execution**; the engine is already close to a hand-rolled single-pass walk.

What remains genuinely wrong: on the large answer set the engine costs **177 µs per match**
versus **5.4 µs** anchored, ~33×. That gap is the O(n²) dedup pass and is worth removing. But
it is a 33× constant on a pathological input, not the 1000× the earlier framing implied.

**The practical consequence is a reordering of what matters.** The highest-value fix for this
entire class is not an engine rewrite — it is a *diagnostic*: "quantified sibling steps with
no anchor between them; did you mean `.`?" That would prevent the pain at authoring time, for
free. It also strengthens the case for the compiler/IR work on completely different grounds
than raw throughput: the justification is tooling and diagnostics
([`06-compiler-architecture.md`](06-compiler-architecture.md)), not speed.

### The right metric for a large answer set: enumeration delay

Unanchored patterns are frequently *correct* — non-adjacent sibling matching is often exactly
what the author wants. So "add anchors" is a workaround, not a fix, and the engine has to be
good at the unanchored case on its own terms.

When the answer set is inherently large, total runtime measures the answer, not the algorithm.
The meaningful metric is **delay**: the time between consecutive emitted matches, and the
latency to the first one. A consumer that takes the first 100 of 22,100 matches should pay for
100. Measured at M=50 (`bench ... delay`):

| | unanchored (22,100 matches) | anchored (50 matches) |
|---|---|---|
| first-match latency | **0.018 ms** | 0.016 ms |
| mean delay | **0.190 ms** | **0.0058 ms** |
| delay by decile | 0.046 → 0.090 → 0.157 → 0.150 → 0.209 → 0.287 → 0.231 → 0.278 → 0.320 → 0.137 | flat at ~0.0056 |
| last decile / first | 3.0× | 0.7× |

Three findings, all actionable:

1. **First-match latency is already excellent** (18 µs), so streaming consumers are not
   penalised by the answer-set size. That is a genuine strength worth preserving.
2. **Per-match cost is ~34× higher** on the unanchored pattern — 0.190 ms vs 0.0058 ms. That
   gap is the engine, not the semantics, and it is the real prize.
3. **Delay grows ~3× across the enumeration** (0.046 → 0.32 ms), so this is not
   constant-delay. The growth tracks the accumulating live-state set that the O(n²) dedup
   pass scans.

**This gives the rewrite a crisp, falsifiable target: constant-delay enumeration at
~5 µs/match for unanchored patterns — i.e. match what the engine already achieves for anchored
ones.** That is a 34× improvement with a flat profile, on inputs where the user's query is
correct as written. It is a much better goal statement than "make it faster", and it has a
mature theory behind it (see [`07-references.md`](07-references.md) §Enumeration).

### Why the unanchored form is expensive

Three properties compose:

1. **The pattern is non-rooted.** It has no single root node; its top-level steps are
   siblings at the same depth. So states are not scoped to a subtree — they are created
   while scanning the children of `source_file` and stay alive until `source_file` is exited.
   With one enormous root node, "alive until the parent is exited" means "alive for the whole
   file".

2. **`*` can match zero occurrences.** So at every sibling position the engine must consider
   "the run starts here" — M start positions.

3. **Two adjacent `*` quantifiers create a split point.** For a given start and a given
   `function_item`, the boundary between the `attribute_item` run and the `line_comment` run
   can be placed in several ways, each a distinct capture set, each a distinct thread.

Cross those and you get Θ(M²) simultaneously live threads, each carrying a capture list of
size Θ(M) in the worst case. The dedup pass compares pairs within a group → Θ(M⁴) capture
comparisons before the early-break optimizations, which is what 2.35 billion inner steps at
M=50 reflects.

The single-quantifier case is cleanly quadratic and much more benign — `((line_comment)* @doc
(function_item) @fn)` with K consecutive comments before one function:

| K | dedup inner steps | wall |
|---|---|---|
| 10 | 261 | 0.01 ms |
| 20 | 921 | 0.02 ms |
| 40 | 3,441 | 0.03 ms |
| 80 | 13,281 | 0.06 ms |
| 160 | 52,161 | 0.15 ms |
| 320 | 206,721 | 0.45 ms |

Exactly 4× per doubling. O(K²), small constants. Survivable, but it is the same mechanism
one order down.

### The secondary victim: the capture list pool

`capture_list_pool_acquire` (483-505) finds a free list by scanning `self->list` linearly for
a slot with `size == UINT32_MAX`. With thousands of live threads the pool is large and mostly
in use:

| workload | acquires | avg scan length |
|---|---|---|
| c/highlights over query.c | 22,564 | 1.3 |
| rust/highlights over query_test.rs | 33,290 | 1.3 |
| `(block (_)* @stmt)` | 6,128 | 5.1 |
| `(attr* doc* fn)` | 12,595 | **191.6** |

This is a free-list waiting to be written — a singly-linked list threaded through the unused
entries makes acquire and release both O(1). It is a ~20-line change with no semantic effect,
and it is one of the highest ratio-of-value-to-risk items in this whole document.

## Measured: how much merging is actually available

Before writing a merge-based matcher, the question of whether merging is worth anything is
answerable by instrumenting the *existing* engine — bucket the live states at each node by a
candidate merge key and see how much collapses. If the answer were ~1 state per key, the whole
approach would be dead.

Live states bucketed by `(step_index, start_depth)`, and by that plus the thread-local flags
(`seeking_immediate_match`, `skipped_quantifier`, `needs_parent`,
`has_in_progress_alternatives`):

| query | live states sampled | collapse by (step, depth) | collapse by (step, depth, **flags**) | largest group |
|---|---|---|---|---|
| **unanchored two-quant** (the 34× target) | 5,897,144 | **56.67×** | **54.11×** | 545 |
| anchored two-quant (already fast) | — | 1.17× | 1.03× | 2 |
| `rust/highlights.scm` | — | 1.00× | 1.00× | 1 |

Three conclusions, all favourable:

1. **The available collapse is 56× exactly where the cost is.** On the pathological case, 545
   threads share a single `(step_index, start_depth)` at peak. That is the O(n²) dedup pass's
   entire input.
2. **It is a no-op everywhere else.** 1.00× on a real query file, 1.17× on the anchored form.
   So a merge-based matcher targets the bad case precisely and cannot regress the good ones —
   the best possible profile for a risky change.
3. **The thread flags are nearly free to include in the merge key**: 54.11× versus 56.67×, so
   keeping them costs **4.7%** of the available collapse. That settles the first design
   question in favour of the safe option — put the flags in the key, keep the thread-local
   semantics exactly, and give up almost nothing.

The 54× collapse and the measured 34× per-match cost gap are consistent with each other, which
is the first independent corroboration that the merge approach can deliver the target.

**What remains open is captures.** Merging 54 threads that agree on `(step, depth, flags)` and
differ only in capture history is precisely the tagged-automaton problem
([`05-database-angle.md`](05-database-angle.md) §2). That is the spike's real work, and it is
where the disambiguation policy stops being abstract.

## Spike result: merging works, but the dedup pass is load-bearing semantics

`tools/query-profiler/merge_spike.c` runs the stock engine and a merge-based matcher over the
same tree in one process and diffs the match streams. The matcher keeps one entry per control
state `(step_index, start_depth, pattern, flags)` holding a **set** of capture continuations,
with histories as persistent cons-lists in an arena.

**Anchored query — identical.** 120 matches from both engines, same capture shapes. The
matcher is a faithful implementation for the case with nothing to disambiguate. (It is 0.4×
the speed, as expected: collapse there is 1.03×, so there is nothing to win and the spike pays
its overhead.)

**Unanchored query, M=50 — the target case:**

| | stock | merge matcher |
|---|---|---|
| matches | 22,100 | **428,700** |
| captures per match | all 7 | spread: 50 @2, 6,425 @3, 14,030 @4, 16,240 @5, 94,825 @6, 24,700 @7 |
| time | 3,995 ms | 4,696 ms |
| continuations absorbed | — | **2,896,905** |
| control-state match tests | — | **1,645** |

Two conclusions, and the second is the important one.

**1. Merging the control state works, and works hard.** 2.9 million continuations were
absorbed into shared control states instead of forked into threads, and the whole run
performed only 1,645 control-state match tests. The mechanism does what the 56× collapse
measurement predicted.

**2. It does not reduce the output, because the O(n²) dedup pass is not deduplication — it is
the longest-match rule.** The stock engine emits only 7-capture matches; the merge matcher
emits a spread from 2 to 7 captures. The extra 406,600 are matches whose capture set is a
strict subset of another, which `ts_query_cursor__compare_captures` discards. Delete the pass
and they come back.

This is the concrete form of the claim made throughout this document: **the dedup pass *is* the
disambiguation semantics.** It was an argument before; it is now a measurement.

### What that means for the rewrite

The 34× target needs **both** halves:

- merge the control state — demonstrated, and it is where the 56× collapse lives;
- **replace longest-match with a rule applied at merge time**, in bounded work, rather than by
  pairwise capture-set comparison afterwards.

The second is the real problem, and it is exactly the tagged-automaton question
([`05-database-angle.md`](05-database-angle.md) §2). Note what it is *not*: "keep the longest
continuation per control state" does not work, because the 22,100 stock matches are genuinely
distinct maximal matches, not one per control state. The rule has to prune strict subsets while
retaining incomparable alternatives — cheaply, on a shared-prefix representation.

### Lessons for the IR

Recorded because they are the reason to build one, learned from writing a matcher against the
current encoding:

- **`pattern_map` already enumerates a pattern's alternative entry points.** This query has 1
  pattern and 3 map entries. Seeding from the map *and* expanding `alternative_index` seeds the
  same control state twice under different flags, and emits every match twice. Nothing in the
  representation says which of the two enumerations is authoritative; it has to be recovered
  by reading `ts_query__pattern_map_insert`.
- **New states must seek an immediate match** (`ts_query_cursor__add_state` sets
  `seeking_immediate_match = true`), which is what stops a pattern seeded at every position
  from matching everywhere. That is load-bearing semantics expressed as a struct-field default.
- **The alternative-following logic cannot be understood in isolation** — dead-end,
  pass-through, and skip interact with `is_immediate`, `is_last_child`, and the depth of the
  *previous* step. A faithful reimplementation is archaeology, not translation.

Each of these is a case where the semantics live in the interaction between fields rather than
in a stated rule, which is the argument for the IR restated in concrete terms.

## The disambiguation question

This is the part that matters most, and it is not primarily about performance.

When several matches of the same pattern overlap, which ones does the engine return? Today
the answer is: **whatever survives `compare_captures`**. Specifically (4557-4599):

- if one state's capture set is a strict superset of another's, and they are at the same
  `step_index`, and the `seeking_immediate_match` flags line up, the subset is dropped;
- otherwise the subset state is marked `has_in_progress_alternatives`, which defers its
  completion (4614-4615) so a longer match can win later.

That is a *policy*, and a defensible one ("prefer longer matches"), but:

- It is not written down anywhere — not in `docs/`, not in `api.h`, not in a spec.
- It is expressed as an O(n²) post-hoc filter rather than as a rule in the automaton.
- It is entangled with performance: the early-break optimizations at 4544-4555 are sound only
  because of an argument about capture ordering that is stated in a comment. Any change to
  capture ordering silently changes semantics.
- It interacts with `match_limit`: when the capture pool is exhausted,
  `ts_query_cursor__prepare_to_capture` (3845-3890) **steals a capture list from another live
  state and kills it** (3873-3876). Which match you lose depends on pool pressure, i.e. on
  the file, i.e. results are not a pure function of (query, tree) once the limit binds.

The regex world settled this decades ago. Leftmost-longest (POSIX) and leftmost-first (Perl)
are both *specified* disambiguation policies, and both admit implementations where the
decision is made **at thread-merge time in O(1)**, not by comparing whole capture sets
afterwards. The machinery is tagged automata — see [`05-database-angle.md`](05-database-angle.md)
§"The captures problem is the tagged-automata problem".

**Recommendation: before any new engine code is written, write down the disambiguation
policy as a spec with a conformance test suite, derived from current behaviour where current
behaviour is sane and deliberately chosen where it is not.** Everything else in the rebuild
depends on it, and it is the one decision that cannot be revisited cheaply later.

## Other execution-path observations

- **`ts_query_cursor__should_descend`** (3947-3999) scans all live states to decide whether to
  descend. It measured 0 scan steps on the workloads above because the early return at 3952
  fires first — but on range-restricted queries (the case it exists for) it is another linear
  scan per node.
- **`array_erase` on the states array** (4333, 4394, 4514, 4576, 4593) is a `memmove` per
  removal. With thousands of live states and removals in the inner loop, this is O(n) per
  erase inside an O(n²) loop. Swap-remove is not possible because the array's sorted order is
  load-bearing for the group-break optimization — another instance of performance and
  semantics being coupled through a data-structure invariant.
- **`next_match` does a linear scan for the lowest `heap_insert_order`** (4662-4668) even
  though `finished_states` is a heap, because the heap is ordered by capture position and
  `next_match` wants insertion order. Two different orderings over one array; the heap is
  maintained lazily and `next_match` pays a linear scan to undo it.
- **Progress callback accounting is subtly off**: `operation_count` is incremented and wrapped
  at 4046-4048, but the callback check at 4053-4062 reads `self->operation_count == 0` — so
  the callback fires once per 100 operations, which is intended, but `current_byte_offset` is
  updated on *every* iteration (4050-4052) including a `ts_node_start_byte` call. Minor, but
  it is per-operation work for a per-100-operation feature.
