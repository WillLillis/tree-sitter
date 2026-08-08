# Current state: an accurate map of `lib/src/query.c`

4,876 lines. Everything below cites line numbers at `003b10c28`.

## The pipeline as it exists

There are notionally five stages. Only two of them are separate functions.

```
source .scm
   │
   │  ts_query_new                                            (3020-3208)
   │  ┌──────────────────────────────────────────────────┐
   ├─▶│ 1. PARSE + 2. RESOLVE + 3. EMIT   — all fused    │  ts_query__parse_pattern (2396-3018)
   │  │    Stream (37-43) reads UTF-8 directly.          │
   │  │    Symbols/fields resolved against TSLanguage    │
   │  │      inline, mid-parse.                          │
   │  │    QueryStep structs appended to self->steps     │
   │  │      as a side effect of parsing.                │
   │  │    alternative_index back-patched in place.      │
   │  └──────────────────────────────────────────────────┘
   │  ┌──────────────────────────────────────────────────┐
   ├─▶│ 3b. PEEPHOLE PATCH  (3144-3189)                  │  inline block in ts_query_new
   │  │    "Fix up quantifier loop-backs within          │
   │  │     alternations" — clones steps and splices     │
   │  │     dead-end redirects into the emitted program  │
   │  │     because the fused emitter could not express  │
   │  │     the control flow correctly the first time.   │
   │  └──────────────────────────────────────────────────┘
   │  ┌──────────────────────────────────────────────────┐
   └─▶│ 4. ANALYZE   ts_query__analyze_patterns          │  (1653-2175)
      │    Walks the LANGUAGE PARSE TABLE to decide,     │
      │    per step, whether matching can fail.          │
      │    Also rejects impossible patterns.             │
      └──────────────────────────────────────────────────┘
                            │
                            ▼
                     TSQuery (immutable)
                            │
   TSQueryCursor ───────────┴──▶ 5. EXECUTE  ts_query_cursor__advance  (4030-4645)
```

Stage 3b is the tell. It is a code-rewriting pass bolted onto the end of `ts_query_new`,
operating on already-emitted steps, undoing damage caused by the fact that alternation
linking and quantifier linking both want to own the same `alternative_index` field. In a
staged compiler this is a structural property of the IR, not a repair.

## The `QueryStep` — the entire instruction set

`query.c:104-123`. 20 bytes measured.

```c
typedef struct {
  TSSymbol symbol;                            // 0 == wildcard '_'
  TSSymbol supertype_symbol;
  TSFieldId field;
  uint16_t capture_ids[MAX_STEP_CAPTURE_COUNT];  // 3 slots, NONE-terminated
  uint16_t depth;                             // PATTERN_DONE_MARKER (UINT16_MAX) == accept
  uint16_t alternative_index;                 // NONE == none
  uint16_t negated_field_list_id;
  bool is_named: 1;              bool is_immediate: 1;
  bool is_last_child: 1;         bool is_pass_through: 1;
  bool is_dead_end: 1;           bool is_inside_alternation: 1;
  bool contains_captures: 1;     bool root_pattern_guaranteed: 1;
  bool parent_pattern_guaranteed: 1;  bool is_missing: 1;
  bool alternative_is_skip: 1;
} QueryStep;
```

This single struct is the opcode, the operands, the control flow, *and* the results of
static analysis. There is no separation between "what the user wrote", "what we derived",
and "what the machine executes".

### Control flow is encoded in flag combinations, not opcodes

`alternative_index` means four different things depending on three flags:

| `is_dead_end` | `is_pass_through` | `alternative_is_skip` | meaning |
|---|---|---|---|
| — | — | — | this step *or* the alternative (alternation branch) |
| ✓ | — | — | unconditional jump; step has no matching logic |
| — | ✓ | — | split: one thread continues, one jumps (quantifier loop) |
| — | — | ✓ | the alternative is a `?`/`*` zero-occurrence skip |

Reading `ts_query_cursor__advance:4431-4504` requires holding all of that in your head at
once, plus the interaction with `seeking_immediate_match` and `skipped_quantifier`. The
comments there (4453-4501) are genuinely good, and they are load-bearing — the code is not
readable without them. That is a signal about the encoding, not about the comments.

**Measured cost of this encoding.** Steps that exist only for control flow, with no matching
logic of their own:

| query | real steps | `dead_end` | `pass_through` | control-flow-only |
|---|---|---|---|---|
| javascript/highlights.scm | 246 | 101 | 0 | **41.1%** |
| python/highlights.scm | 179 | 75 | 0 | **41.9%** |
| go/highlights.scm | 155 | 67 | 0 | **43.2%** |
| rust/highlights.scm | 128 | 0 | 0 | 0% |
| c/highlights.scm | 70 | 0 | 0 | 0% |

The split is entirely about whether the query author used `[...]` alternations. Rust and C
highlights are written as many small separate patterns (94 and 65 patterns respectively);
JS/Python/Go use root-level alternations, and pay ~42% of their step budget on branch
plumbing. Note also that root-level alternations multiply `pattern_map` entries: JS has 27
patterns but **124** `pattern_map` entries.

## The other data structures

| Structure | Location | Purpose | Notes |
|---|---|---|---|
| `Slice` | 131-134 | `{offset, length}` into a shared array | used for captures, strings, predicates |
| `SymbolTable` | 139-142 | two-way string↔id | **`symbol_table_id_for_name` (920-933) is a linear scan with `strncmp`** → O(n²) to build |
| `CaptureQuantifiers` | 147 | per-pattern `Array(uint8_t)` of `TSQuantifier` | the quantifier algebra (651-902) is a clean semilattice; the only genuinely well-factored part of the file |
| `PatternEntry` / `pattern_map` | 160-164 | sorted (root symbol → pattern start) | the engine's *only* index; binary searched (1238-1274) |
| `QueryPattern` | 166-172 | per-pattern slices + byte range + `is_non_local` | |
| `StepOffset` | 174-177 | step index → source byte offset | used for diagnostics **and** for a public API (see below) |
| `QueryState` | 206-219 | one live VM thread | 20 bytes; `consumed_capture_count` is a **12-bit field** (max 4095) |
| `CaptureListPool` | 230-242 | reusable capture vectors | **`acquire` (483-505) finds a free slot by linear scan** |
| `AnalysisState` etc. | 248-303 | parse-table walk state | `MAX_ANALYSIS_STATE_DEPTH` = 8 |

## Hard limits, and what happens when you hit them

This is the most important table in this document. Every one of these is silent.

| Constant | Value | Line | Behaviour on overflow | Verified |
|---|---|---|---|---|
| `MAX_STEP_CAPTURE_COUNT` | 3 | 27 | **4th capture on a node silently discarded**; `ts_query_capture_count()` still counts it | ✅ yes |
| `MAX_NEGATED_FIELD_COUNT` | 8 | 28 | **9th `!field` silently discarded** (2729-2732) | ✅ yes |
| `MAX_STATE_PREDECESSOR_COUNT` | 256 | 29 | predecessors past 256 dropped → analysis under-approximates | not tested |
| `MAX_ANALYSIS_STATE_DEPTH` | 8 | 30 | `did_abort = true` → all steps marked fallible (conservative) | not tested |
| `MAX_ANALYSIS_ITERATION_COUNT` | 256 | 31 | `did_abort = true` (conservative for guarantees; **see `03` for a possible unsound path**) | not tested |
| `supertypes[8]` | 8 | 4178 | fixed stack buffer passed to `ts_tree_cursor_current_status` | not tested |
| `consumed_capture_count` | 12 bits | 213 | wraps past 4095 captures in one match | not tested |
| `step_index`, `alternative_index` | `uint16_t` | 110, 211 | queries over 65534 steps | not tested |

The `MAX_STEP_CAPTURE_COUNT = 3` limit is worth dwelling on: it is not a limit on captures
per *pattern*, it is a limit on captures attached to a *single node*, and it interacts with
alternations — `query_step__add_capture` is called once per alternation branch head
(2954-2967), so `[(a) (b) (c) (d)] @x` distributes fine, but `(x) @a @b @c @d` does not.

## The analysis stage

`ts_query__analyze_patterns` (1653-2175) is the single most expensive thing in the library
per byte of input, and it is the least understood. What it does:

1. **Marks `contains_captures`** by scanning forward from each step over its descendants
   (1670-1695). O(steps²) worst case.
2. **Validates supertype/subtype pairings** (1700-1735).
3. **Builds `AnalysisSubgraph`s** — one per query parent symbol, *plus one per hidden symbol
   in the grammar* (1747-1759).
4. **Scans the entire parse table** (1768-1840) to find, for every state: reduce actions
   (subgraph end states), shift actions (predecessor map), and start states.
5. **Walks backward** through the predecessor map to complete each subgraph (1844-1882).
6. **Simulates hypothetical trees** (`ts_query__perform_analysis`, 1307-1613) to find steps
   where a match can terminate → `parent_pattern_guaranteed` / `root_pattern_guaranteed`.
7. **Clears `root_pattern_guaranteed` for predicate-referenced captures** (2013-2050).
8. **Propagates fallibility backwards** to a fixpoint (2052-2087) — an O(steps²) loop.
9. **Re-runs the whole walk** for non-rooted patterns to populate
   `repeat_symbols_with_rootless_patterns` (2096-2145).

Steps 3-5 are dominated by the language, not the query. Measured:

| language | states | `predecessor_map` alloc | full-table scan | `perform_analysis` (highlights.scm) |
|---|---|---|---|---|
| rust | 3,823 | **1,918 KiB** | 2.72 ms | 12.96 ms |
| python | 2,809 | 1,409 KiB | 1.79 ms | 2.96 ms |
| c | 2,015 | 1,011 KiB | 1.52 ms | 1.53 ms |
| javascript | 1,870 | 938 KiB | 1.39 ms | 6.25 ms |
| go | 1,442 | 723 KiB | 0.70 ms | 0.79 ms |

The `predecessor_map` is `ts_calloc(state_count * 257, sizeof(TSStateId))` (1016-1019) —
a fresh multi-megabyte zeroed allocation on **every** `ts_query_new`, including for
`(identifier) @v`.

### What the analysis buys, and whether it is worth it

The output is two bits per step. Their only consumers:

- `root_pattern_guaranteed` → `ts_query_cursor__first_in_progress_capture` (3652) decides
  whether an *unfinished* match's capture can be streamed early by `next_capture`.
- `parent_pattern_guaranteed` → `ts_query__step_is_fallible` (3368-3386) decides whether to
  split a state when a later sibling could also match (4344-4356).

Measured yield on real queries — fraction of steps where the analysis proved a guarantee:

| query | `root_pattern_guaranteed` |
|---|---|
| rust/highlights.scm | 5.5% |
| javascript/highlights.scm | 2.8% |
| c/highlights.scm | 2.9% |

So: 13–17 ms of analysis to prove a property about ~3–6% of steps. This is not an argument
that the analysis is wrong — the guarantee it computes is genuinely needed for `next_capture`
streaming, and it *also* rejects impossible patterns, which is a real correctness feature.
It is an argument that its cost is grossly mismatched to its yield and that it is being
recomputed constantly for facts that change only when the grammar changes.

## Public API surface for introspection

Three functions, exported to C, Rust, and WASM. **Nothing in this repository calls any of
them.**

```c
bool ts_query_is_pattern_rooted(const TSQuery *, uint32_t pattern_index);       // O(pattern_map) linear scan
bool ts_query_is_pattern_non_local(const TSQuery *, uint32_t pattern_index);
bool ts_query_is_pattern_guaranteed_at_step(const TSQuery *, uint32_t byte_offset);
```

The third is keyed by **source byte offset**, resolved by a linear scan of `step_offsets`
(3351-3366). That signature is the clearest evidence that there is no IR: the only stable
name the library can offer for "a point in the compiled query" is a byte offset into the
original text. Everything a tool would want to ask — what does this compile to, why is this
step fallible, what is the plan, how selective is this pattern — is unaskable.

`crates/cli/src/query.rs` (174 lines) runs a query and prints matches. There is no
`explain`, no `--profile`, no bytecode dump, no plan output.

## What is genuinely good here

Worth preserving through any rewrite:

- **The quantifier algebra** (651-902). `quantifier_mul` / `_join` / `_add` form a correct,
  total, well-documented semilattice over `TSQuantifier`. This is real semantics, written
  down properly. Keep it verbatim.
- **The `pattern_map` design** (1224-1274). Grouping patterns by root symbol with a binary
  search, and segregating wildcard roots into a prefix, is the right idea. It is just the
  *only* index that exists.
- **The anchor/quantifier interaction comments** (4453-4501, and the commit series
  `a6bc72474`, `139b801cc`, `dfcf73921`, `1ffd612be`). These encode hard-won semantics that
  exist nowhere else — not in docs, not in a spec. They are the closest thing to a
  specification of what anchors mean, and any rewrite must treat them as the requirements
  document.
- **`ts_query__perform_analysis`'s core idea** — using the parse table to reason about
  hypothetical trees — is genuinely clever and is a capability most query engines do not
  have. The problem is when and how often it runs, not that it exists.
