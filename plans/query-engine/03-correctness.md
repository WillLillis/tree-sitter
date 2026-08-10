# Correctness: verified defects, suspicions, and semantic gaps

Three tiers, kept strictly separate:

- **A — verified.** Reproduced against this tree with a harness. Repro included.
- **B — code-reading, unverified.** A specific line and a specific argument, but no test yet.
- **C — semantic gaps.** Not bugs; behaviour that is undefined, unspecified, or absent.

---

## Tier A — verified defects

Repro harness: `08-measurement-harness.md`. Language: Rust grammar from `test/fixtures`.

### A1. The 4th capture on a node is silently discarded

```
(identifier) @a @b @c @d     ->  compiles clean
ts_query_capture_count()     ->  4        (a, b, c, d)
capture_ids on the step      ->  a, b, c
runtime match capture_count  ->  3        forever
```

`MAX_STEP_CAPTURE_COUNT` is 3 (`query.c:27`). `query_step__add_capture` (984-991) loops over
the three slots and, finding none free, **falls out of the loop and returns with no
indication**. The capture name is still interned into the `captures` symbol table, so the
public capture count disagrees with what any match can ever contain.

Severity: silent, permanent, wrong results. A binding that indexes captures by name will
look up `@d`, get a valid id, and never see it in a match.

Note this limits captures **per node**, not per pattern. `[(a) (b) (c) (d)] @x` is fine
because `add_capture` is called once per branch head (2954-2967).

Minimum fix: return `TSQueryErrorCapture` (or a new `TSQueryErrorTooManyCaptures`) instead of
dropping. Real fix: make the capture list a `Slice` into a side array like every other
variable-length thing in the file, removing the limit entirely.

### A2. `MISSING` is matched by prefix, so `(MISS)` and `(M)` compile as `(MISSING _)`

`query.c:2554`:

```c
} else if (!strncmp(node_name, "MISSING", length)) {
```

`length` is the length of the scanned identifier, so this returns 0 for **any prefix of
`MISSING`**. Verified:

```
(MISS)     -> compiles          (should be TSQueryErrorNodeType)
(M)        -> compiles          (should be TSQueryErrorNodeType)
(NOPE)     -> TSQueryErrorNodeType @1     (control: correct)
(MISSING)  -> compiles                    (control: correct)
```

`(MISS)` silently becomes "match any missing node". A grammar with a node type named `M`,
`MI`, `MIS`, `MISS`, `MISSI`, or `MISSIN` would have that node type become unqueryable and
silently reinterpreted.

Fix is one line: `length == 7 && !strncmp(node_name, "MISSING", 7)`.

### A3. The 9th negated field is silently discarded

`query.c:2729-2732`:

```c
if (negated_field_count < MAX_NEGATED_FIELD_COUNT) {
  negated_field_ids[negated_field_count] = field_id;
  negated_field_count++;
}
```

No `else`. Verified: a pattern with 9 distinct `!field` assertions compiles and records 8.
The 9th constraint is dropped, so the query matches nodes it should reject — a **false
positive**, which is worse than a false negative for most consumers.

`MAX_NEGATED_FIELD_COUNT` is 8 (`query.c:28`).

### A4. Results depend on `match_limit`, and the default is unbounded

`ts_query_cursor__prepare_to_capture` (3845-3890), when the pool is exhausted, calls
`ts_query_cursor__first_in_progress_capture` and **steals the capture list from the state
holding the earliest capture, marking that state dead** (3873-3876).

Consequences:

- The set of returned matches is a function of `(query, tree, match_limit, allocation
  history)`. Not a pure function of `(query, tree)`.
- The default `max_capture_list_count` is `UINT32_MAX` (447), so by default there is **no
  bound on memory**; the `(attr* doc* fn)` case above holds thousands of capture lists live.
- `did_exceed_match_limit` reports that *something* was dropped but not what.

Verified that behaviour varies with the limit; the interesting property is not the exact
counts but that the limit silently changes the result set rather than erroring.

This is arguably by design ("degrade rather than fail"), but it should be a documented,
deterministic degradation — e.g. drop *whole patterns* in a defined priority order, or
surface a recoverable error — not "whichever state happened to hold the earliest capture".

### A5. `analysis_state__compare` is not antisymmetric, and the sorted analysis set relies on it

`query.c:1071-1092`. The depth test fires *before* the stack entries are compared, and only in
one direction:

```c
if ((*self)->depth < (*other)->depth) return 1;   // shallower sorts later
for (unsigned i = 0; i < (*self)->depth; i++) {
  if (i >= (*other)->depth) return -1;
  ... compare stack[i] ...
}
```

If `A` is shallower **and** `A.stack[0]` sorts before `B.stack[0]`, then `compare(A,B)` returns
1 via the depth test, and `compare(B,A)` returns 1 via the stack comparison. Both report the
other is smaller.

Verified by running every real comparison both ways:

| query | comparisons | inconsistent | |
|---|---|---|---|
| `rust/highlights.scm` | 756,626 | 3,509 | 0.46% |
| `javascript/highlights.scm` | 272,246 | 7,777 | 2.86% |
| `(arguments (identifier) × 6)` | 202,531 | 12,979 | 6.41% |
| `(block (let_declaration) × 6)` | 46,545 | 12,372 | **26.58%** |

All are the `both > 0` case. `analysis_state_set__insert_sorted` calls
`array_search_sorted_with`, which requires a total order, so the set is not reliably sorted:
binary search can miss an entry that is present (inserting a duplicate) and can compute a
wrong insertion position. The same comparator also drives the iteration-ordering shortcut in
`perform_analysis` (1380-1402) that decides which states to defer.

**The awkward part: fixing it makes compilation slower.** Two correct repairs, both
antisymmetric, both passing all 127 query tests and the goldens unchanged:

| pattern | current (buggy) | compare stacks first | depth primary both ways |
|---|---|---|---|
| `rust/highlights.scm` | 14.39 ms | 13.96 ms | 13.85 ms |
| `(block (let_declaration) × 6)` | 3.11 ms | 4.48 ms | 7.60 ms |
| `(parameters (parameter) × 6)` | 6.09 ms | 12.08 ms | 10.52 ms |
| `(arguments (identifier) × 6)` | **6.31 ms** | **16.61 ms** | **47.67 ms** |

The broken ordering is acting as accidental pruning: an unsorted region causes the binary
search and the deferral shortcut to truncate exploration. So **the analysis as designed costs
2.6–7.6× more than the analysis as implemented**, on broad-parent patterns.

Recommendation: **do not land either repair standalone.** No observable behaviour changes on
the corpus, so there is no user-facing bug to fix today, and paying 2.6–7.6× compile for
internal correctness is a bad trade in isolation. Instead treat it as a constraint on the
analysis rewrite: a correct total order must be part of the new design, and the design has to
absorb the cost the current implementation is avoiding by accident. It also means every
measurement of analysis cost in these docs understates the true cost of the algorithm.

---

## Tier B — code-reading, not yet verified

Each of these deserves a targeted test before being treated as real.

### B1. Aborted analysis may under-populate `repeat_symbols_with_rootless_patterns`, causing missed matches

`ts_query__analyze_patterns:2096-2145`. The second analysis pass sets
`analysis.did_abort = false` at 2096, runs `ts_query__perform_analysis` per non-rooted
pattern (2131), and then **never checks `did_abort` again** before using
`analysis.finished_parent_symbols` (2137-2144) to populate
`self->repeat_symbols_with_rootless_patterns`.

If analysis aborts (hitting `MAX_ANALYSIS_ITERATION_COUNT` = 256 or
`MAX_ANALYSIS_STATE_DEPTH` = 8), `finished_parent_symbols` is incomplete. That array is then
consulted by `ts_query_cursor__should_descend` (3982-3992) to decide whether to descend into a
repetition node:

```c
if (ts_subtree_is_repetition(subtree)) {
  ... array_search_sorted_by(&self->query->repeat_symbols_with_rootless_patterns, ...);
  return exists;     // <-- false => do not descend
}
```

A missing symbol here means the cursor **does not descend into a repetition node it should
have**, silently losing matches.

Contrast with the first analysis pass, which handles abort correctly by conservatively
clearing all guarantees (1964-1977). The difference matters: the first pass's fallback is
conservative (assume fallible), the second pass's implicit fallback is *optimistic* (assume
no rootless pattern can match there). Optimistic fallbacks on aborted analysis are unsound.

To verify: find or construct a grammar + query where `did_abort` fires on the rootless pass.
Instrumenting `qp_analysis_aborts` (already in the harness) across the full grammar fixture
set with rootless patterns would find candidates. Measured aborts on the five
`highlights.scm` files tested: **0**, so this is not hit by common queries — but a
deeply-recursive grammar is exactly where it would bite.

### B2. `analysis_state__compare` reads `stack[i]` up to `self->depth` without bounding by `MAX_ANALYSIS_STATE_DEPTH`

`query.c:1071-1092`. The loop is bounded by `(*self)->depth`, and depth is incremented at
1521 after a guard at 1512 (`next_state.depth + 1 >= MAX_ANALYSIS_STATE_DEPTH`). The guard
looks correct, so this is probably fine — noting it only because the invariant "depth ≤ 8" is
maintained at a distance from every place that indexes `stack`.

### B3. `ts_query__step_is_fallible` asserts rather than bounds-checks

`query.c:3376`: `ts_assert((uint32_t)step_index + i < self->steps.size);` inside a loop that
skips pass-through steps. In release builds `ts_assert` compiles out. If a pattern ever ends
with a run of pass-through steps and no terminator, this walks off the array. The
`PATTERN_DONE_MARKER` push at 3070 should prevent it. Worth a fuzz target rather than an
argument.

### B4. `is_rooted` computation stops at the first dead-end step

`query.c:3114-3121`:

```c
for (uint32_t step_index = start_step_index + 1; step_index < self->steps.size; step_index++) {
  QueryStep *child_step = array_get(&self->steps, step_index);
  if (child_step->is_dead_end) break;
  if (child_step->depth == start_depth) { is_rooted = false; break; }
}
```

The scan is over *all* remaining steps in the query, not just this pattern's steps, and it
terminates at the first `is_dead_end`. For a pattern whose first alternation branch is short,
the `is_dead_end` at the end of that branch stops the scan before later branches are examined
— so a pattern that is non-rooted via a later branch could be classified rooted. `is_rooted`
controls whether range restrictions apply (4214-4216, 4237-4239), so a misclassification
changes which matches are produced under `set_byte_range`. Needs a test with an alternation
where branch 1 is a single node and branch 2 is a sibling sequence.

---

## Tier C — semantic gaps and missing features

Not bugs. Places where the language or the contract is absent.

### C1. There is no specification of match semantics

There is no document that answers:

- Which of several overlapping matches of the same pattern are returned?
- Are matches returned in a defined order? (`next_match` uses insertion order; `next_capture`
  uses byte order — these disagree, by design, but it is nowhere stated.)
- What does `.` mean when the preceding quantifier matched zero nodes? (Four commits in this
  tree answer this — `a6bc72474`, `139b801cc`, `dfcf73921`, `1ffd612be` — and the answer
  exists only in code comments at `query.c:4453-4501`.)
- What is the interaction of `!field` with alternations and quantifiers?
- Are captures within a match ordered? (They are, by traversal order, and
  `compare_captures` depends on it — but that is an implementation detail leaking into
  contract.)

**This is the highest-leverage document nobody has written.** Everything in
[`06-compiler-architecture.md`](06-compiler-architecture.md) presupposes it.

### C2. Predicates are outside the engine entirely

`ts_query_predicates_for_pattern` hands raw steps to the binding; the C engine never
evaluates a predicate. Consequences:

- `#eq?`, `#match?`, `#any-of?` semantics are **defined by each binding independently** —
  Rust, JS/WASM, Python, Go, and the CLI can and do differ in edge cases.
- No predicate can influence matching. `(identifier) @x (#eq? @x "foo")` matches *every*
  identifier in the file and then discards nearly all of them in the binding. For a
  `tags.scm` or `locals.scm` workload this is the dominant cost and it is entirely avoidable
  (see [`05-database-angle.md`](05-database-angle.md) §"Predicate pushdown").
- The engine partially knows about predicates anyway — it clears `root_pattern_guaranteed`
  for predicate-referenced captures (2013-2049) — so the abstraction is already leaking, just
  in the direction that costs performance rather than the direction that buys it.

### C3. No descendant axis

The pattern language has a child axis and a sibling axis. It has no `//`. Expressing "an
`identifier` anywhere inside a `function_item`" requires either nested wildcards to a fixed
depth or a non-rooted pattern plus post-filtering. XPath solved this in 1999; the
implementation technique (region encoding + structural join) is directly available because
tree-sitter nodes already carry byte ranges. See [`05-database-angle.md`](05-database-angle.md).

### C4. No query composition

`highlights.scm` files are combined by **text concatenation**. There is no import, no
namespacing of captures, no way to say "these patterns, but with `@function` renamed", no
way to compose a base grammar's queries with a dialect's. Every downstream project
(nvim-treesitter, Helix, Zed) has reinvented some form of query file inheritance in its own
configuration layer.

### C5. Diagnostics are one byte offset and one enum

`TSQueryError` has 7 variants and `ts_query_new` returns a single `error_offset`. There is no
span, no note, no suggestion, no multiple-error reporting. Compare to what a query LSP would
need. This is a direct consequence of the fused parser: there is no AST to attach diagnostics
to, and errors are reported by resetting the `Stream` to a saved pointer (2271, 2339, 2569,
etc.).

### C6. `ts_query_disable_capture` / `ts_query_disable_pattern` mutate a supposedly immutable object

`TSQuery` is documented as immutable (`query.c:306-308`) and `TSQueryCursor` holds a
`const TSQuery *`. But `ts_query_disable_capture` (3388-3402) rewrites steps in place and
`ts_query_disable_pattern` (3404-3417) erases `pattern_map` entries. Neither can be undone.
Calling either while a cursor is mid-execution over that query is undefined and unchecked.

The step-rewriting version is also asymmetric with the analysis: disabling a capture does not
re-run the fallibility analysis that was influenced by that capture (2013-2049), so
`root_pattern_guaranteed` is left stale — conservative, so not wrong, but a latent
inconsistency.
