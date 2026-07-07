# Flat-pool backend redesign - investigation handoff

Status as of 2026-07-02. Branch `rule_ir` (main repo) + `rule_ir` (dsl-tests).
This doc is a one-time planning artifact at the repo root; delete it before any PR.

## TL;DR

We want to replace the recursive `Rule` box-tree in the **shared** grammar backend
(`crates/generate/src/prepare_grammar/*`, used by BOTH the `grammar.js` frontend and
the native `.tsg` frontend) with a **flat, pool-indexed IR**, for two goals:
- **Perf** (fewer/cheaper allocations, better cache).
- **Depth-safety** (iterative pool walks -> drop `MAX_RULE_DEPTH`, kill the
  stack-overflow class).

**Attempt #1 failed and was reverted.** It used a hash-consed pool and regressed the
shared backend ~15-20% vs master (see "Why #1 failed"). This is **attempt #2**:
design the representation *around how the passes actually access rules*, with **no
hash-consing**. We spiked `intern_symbols` on the new rep and it is **2-3x faster
than master**.

**2026-07-06: the `flatten_grammar` spike (the honest go/no-go) is done and it is a
GO: 8-12x faster than master with output verified byte-equal on all six fixture
grammars** (see measurements). The win is the fork-at-choice walk deleting
`extract_choices`' materialized cross-product; no in-place rewrites involved, so
the arena wins at both ends of the pass access-pattern table.

**Verification correction (same day):** the first measurement session claimed the
equality assert passed, but the run commands used `grep -m1`, whose SIGPIPE
killed `generate` before the assert executed. Running to completion exposed a
real fork bug: restore between choice branches used `truncate` on the metadata
stacks, but a branch pops context pushed BEFORE the fork (an enclosing
`Metadata`/`Reserved` exit is part of the branch's continuation), which truncate
cannot re-grow; alias/prec/field context leaked or vanished across branches.
Fix: fork snapshots ALL consumable context stacks (frames, prec, assoc, alias,
field, reserved); only `steps` is truncate-safe. Post-fix, full runs (exit 0,
assert executed) verify all six grammars; timing was not materially affected.
Protocol rule going forward: never pipe-truncate a run whose verification comes
after the grepped line; capture to a file and check the exit code.

**Scope decision (maintainer, 2026-07-06): all or nothing.** Either the backend
moves fully to the pool IR or the spike is abandoned; no with-adapters
incremental landing (that was attempt #1's valley). The bridge timings below
remain useful only as development-time conversion bounds.

**Sequencing decision (maintainer, 2026-07-06): master first.** The IR rewrite
is worked out here, then lands on master BEFORE the native DSL changes. The
backend between (synced) master and `native_dsl` is essentially identical
(local `master` was merely stale; the apparent divergence is upstream commits),
so the implementation branch forks from upstream master and the spikes port
trivially. Consequences:
- The rewrite becomes an upstream-able series independent of native DSL:
  grammar.json frontend (`parse_grammar`) builds pools directly, all prepare
  passes go flat, `SyntaxGrammar` is pooled, byte-identical `parser.c` is the
  oracle. The standalone quick wins (intern_name O(1) map, `Symbol` u32) can
  lead the series.
- `native_dsl` then rebases and retargets its lowering to emit pools directly,
  which deletes `build_rule`/`str_to_string` (the alloc bomb measured in
  `plans/nativedsl-string-interning-design.md`) as part of the native PR: the
  native frontend ships already built for the new backend.
- The `build_tables` item-key layer lands with the series or immediately
  after; its prerequisite (pooled deduped productions) is in the series.

Next: corpus validation when available; the real design doc is DONE
(`FLAT_POOL_DESIGN.md`, 2026-07-07, includes the full pass survey,
frontend contracts incl. the metadata-merge parity seam, and the landing
plan); then pass-by-pass implementation on `rule_ir` per that doc.

## Why attempt #1 failed (do NOT repeat)

- The pool hash-consed **every node on every pass rebuild** (an FxHashMap probe +
  insert per node, plus a throwaway `Vec` key for each `Seq`/`Choice`). Master only
  dedups where a pass needs it (`expand_repeats`).
- Dedup was real (~56% of nodes on cpp) but only saved **storage**, not the
  per-reference **walk** work the passes do - so we paid the hash tax and got a net
  loss for cache-resident grammars.
- Measured regression on `prepare_grammar` (the shared backend, grammar.js pays it
  too): cpp 3.72ms(master) -> 4.41ms; +14-25% across cpp/c/python/go/js.
- Reverted to `d43966acc` ("nativedsl: convert lowering to iterative walks") via 10
  revert commits on `native_dsl` (history preserved). dsl-tests reverted `00e5b0f`.
- Full writeup: memory `nativedsl_pool_reverted_lesson.md`.

Key realization: **depth-safety does NOT need a pool** (iterative-over-`Rule` gets
it with zero perf change). The pool must justify itself on **perf**, or it is not
worth the `Rule`-deletion churn.

## The design (attempt #2)

Principles, derived from the backend's access patterns:

| pass | reads | writes | dedup? |
|---|---|---|---|
| intern_symbols | walk tree | `NamedSymbol`->`Symbol` (leaf rewrite, in place) | none |
| extract_tokens | walk tree | hoist token subtrees to lexical, replace w/ `Symbol` | token dedup (local) |
| expand_repeats | walk tree | `Repeat`->aux `Symbol` + new aux rules | aux dedup (local) |
| flatten_grammar | walk tree | `Vec<Production>` (flat step lists) - tree ends here | production dedup (local) |
| expand_tokens | walk token trees | NFA | none |

1. **Flat arena, no hash-consing.** `nodes: Vec<FNode>`; a child holds a `NodeId`
   into it. Build = plain `push` (O(1), no hashing). Dedup ONLY locally in the ~3
   passes that need it, structurally, exactly like master.
2. **Children as ranges.** `Seq`/`Choice` hold `(offset,len)` into ONE backing
   `children: Vec<NodeId>`. Never a per-node `Vec`. Aspiration: "never add, only
   subtract" - build the backing store once, rewrite in place / shrink.
3. **In-place mutation is the win.** No hash-consing => no node sharing => it's a
   tree => leaf rewrites (`NamedSymbol`->`Symbol`, `Repeat`->`Symbol`,
   token->`Symbol`) are `nodes[id] = ...`, O(1), zero alloc. Master rebuilds the
   whole enclosing tree for these. THIS is how the pool beats master.
4. **String interning.** Payloads (`String`/`Pattern`/`NamedSymbol` text) intern to
   a `StrId` (u32) in a per-pool interner. Hash once at build; every later use is a
   cheap `u32`. Amortizes across passes (this is why interning pays off across the
   backend, NOT within a single pass).
   - `str -> StrId` is a content lookup => must be a hashmap (`FxHashMap<Box<str>,
     StrId>`). `StrId`s being dense does NOT help this direction.
   - `StrId -> str` (reverse, for serialize/errors) is dense => a `Vec` when needed.
     The spike drops it (intern_symbols never resolves text). Real design: use
     `Rc<str>` or a blob+ranges to avoid double-allocating each string (do NOT keep
     both a `Vec<Box<str>>` and a `Box<str>`-keyed map - that double-stores).
5. **Name resolution via a dense `Vec`.** Names are interned first, so name `StrId`s
   are `0..k`; `name_of_symbol: Vec<Symbol>` indexed by `StrId` (NOT a map, NOT
   `Option` - every `0..k` slot is a name; an undefined symbol's `StrId` is `>= k`,
   caught by the `.get()` bounds check).
6. **Two id-spaces** (confirmed with the user): the dense `Symbol` index (`0..num_rules`,
   what parse tables need) and the arena `NodeId` (structure). A named rule maps
   `name -> Symbol` and lives at some arena node.

## Measurements so far (`intern_symbols` only)

Release, driven by `target/release/tree-sitter generate <grammar>/grammar.js`, which
prints `[SPIKE intern_symbols] master <A>  flat <B>` (300 iters, pass only; the
`Rule->pool` conversion is excluded setup, flat pools pre-cloned so only the in-place
rewrite is timed). us/iter:

| grammar | master (scan) | master + name map | flat (in-place, StrId) |
|---|---|---|---|
| cpp | 253.8 | ~152 | **53** |
| c | 86.9 | ~72 | 27 |
| python | 55.0 | ~43 | 19 |
| go | 54.6 | ~48 | 26 |
| javascript | 57.5 | ~51 | 21 |

Decomposition (cpp), each isolated:
- **scan -> map: 254 -> 152us.** Master's `intern_name` is an O(V) linear scan; an
  O(1) map fixes it. This is a `Rule`-only win, NO pool needed - **landable on master
  today** (see next steps).
- **Rule -> flat pool + in-place: 152 -> 75us (~2x).** The real backing-type payoff
  (no new-tree allocation).
- **String -> StrId: 75 -> 53us (~1.4x).** Real, but amortizes across passes; a
  single pass under-credits it (interning cost sits in setup).

`name_of_symbol` structure (hashmap vs dense `Vec<Symbol>`): **perf-neutral at the
current 16-byte `Symbol`** (~53us either way for cpp). The dense `Vec` is the right,
clean choice and will pull ahead once `Symbol` shrinks, but it's not the lever here.

## Measurements: `flatten_grammar` (2026-07-06, WSL2, repo fixture grammars)

`[SPIKE flatten_grammar]` block in `prepare_grammar.rs`; 300 iters; master's
flatten consumes its input so its loop clones per iteration (clone timed
separately and netted out below); the flat side mirrors master's post-checks
(empty-string, recursive-inline) and skips only the pass-through SyntaxGrammar
assembly. Output materialized (untimed) and asserted equal to master's on every
grammar. us/iter:

Post-fix numbers (fork snapshot fix in, assert executed and passed, exit 0):

| grammar | master (net of clone) | flat | speedup |
|---|---|---|---|
| c | 1335 | 126 | 10.6x |
| python | 646 | 63 | 10.2x |
| go | 599 | 49 | 12.3x |
| javascript | 915 | 76 | 12.0x |
| rust | 2575 | 281 | 9.2x |
| cpp | 2750 | 340 | 8.1x |

(cpp runs from the repo fixture after `mkdir -p
test/fixtures/grammars/cpp/node_modules && ln -sfn ../../c
test/fixtures/grammars/cpp/node_modules/tree-sitter-c`; its grammar.js
requires `tree-sitter-c/grammar` and the c fixture is already a sibling.)

Side effect of the `FNode` 104->12 B shrink: cpp `intern_symbols` flat improved
from the handoff's 53us to 22.9us (master 258.7us, 11.3x) - the in-place walk
is pure node traversal, so it is footprint-bound.

### Combined view (both converted passes in syntax-path context)

`[SPIKE combined]` sums the two converted passes and places them against the
unconverted middle (`extract_tokens` + `expand_repeats`, timed net of input
clones). "Hybrid" = flat intern + unconverted middle + flat flatten, i.e. the
end-state projection with no bridges; one-shot pool-build (bridge) costs are
printed for the with-adapters scenario. us:

Post-fix numbers, all six grammars, asserts verified:

| grammar | two passes master | two passes flat | x | middle | syntax path master -> hybrid | x |
|---|---|---|---|---|---|---|
| c | 1471 | 138 | 10.7 | 400 | 1871 -> 538 | 3.5 |
| cpp | 3012 | 360 | 8.4 | 777 | 3789 -> 1137 | 3.3 |
| python | 725 | 71 | 10.2 | 245 | 970 -> 316 | 3.1 |
| go | 692 | 58 | 11.9 | 267 | 959 -> 326 | 2.9 |
| javascript | 1014 | 85 | 12.0 | 299 | 1313 -> 384 | 3.4 |
| rust | 2692 | 293 | 9.2 | 448 | 3140 -> 742 | 4.2 |

Readings: the two converted passes are 71-86% of the syntax path, so
converting just them is a 2.9-4.2x whole-syntax-path win; in the projected
full conversion the current middle (`extract_tokens`/`expand_repeats`) is the
dominant remaining term, and it is the same tree-rebuild disease, so similar
per-pass wins are plausible.

### Where this sits in the whole `generate` pipeline (2026-07-06)

Real single-shot stage times (`[SPIKE prepare-stages]` / `[SPIKE
generate-stages]`, spike loops excluded), ms:

| grammar | prepare total | of which syntax path | expand_tokens | aliases+inlines | build_tables | render |
|---|---|---|---|---|---|---|
| c | 3.3 | 1.8 | 1.0 | 0.3 | 552 | 26 |
| cpp | 5.5 | 3.4 | 1.2 | 0.5 | 6366 | 150 |
| python | 2.5 | 0.9 | 1.0 | 0.4 | 284 | 27 |
| go | 2.0 | 1.0 | 0.6 | 0.2 | 221 | 15 |
| javascript | 2.8 | 1.3 | 0.8 | 0.6 | 575 | 16 |
| rust | 14.0 | 3.0 | 0.6 | 10.2 | 1358 | 45 |

So `prepare_grammar` is ~0.1-1% of a full `tree-sitter generate`;
`build_tables` is 85-97%. The honest framing for the all-or-nothing decision:
- The full pool replacement does NOT measurably speed up end-to-end `generate`
  today. Its immediate wins are prepare-bounded workloads (grammar.json
  emission, node-types-only, LSP-grade validation of `.tsg` grammars, the
  roundtrip corpus) and depth-safety (dropping `MAX_RULE_DEPTH`).
- Its strategic value is as the foundation for attacking `build_tables`, where
  the real time lives: the flat `SyntaxGrammar` (20 B `Copy` steps, dense
  symbols) is exactly the input representation `build_tables`' item-set loops
  want, and the handoff's `Symbol` u32 shrink already points there.
- rust's `aliases+inlines` at 10.2 ms (`process_inlines`, another
  tree-rebuild pass) is worth a row in the real design doc's pass table.

Design shape that produced this (all in `flat_spike.rs`):
- Fork-at-choice path enumeration replaces `extract_choices` + `RuleFlattener`:
  no alternative trees are ever materialized; a fork snapshots the frame stack
  into a reusable buffer and truncate-restores between branches.
- Zero steady-state allocations: all flatten state is persistent scratch;
  output is pooled (`FStep` 20 B `Copy` steps + production ranges), so the only
  allocation is amortized output-pool growth.
- Packed layouts, const-asserted: `Frame` 8 B (2-bit tag in the high bits),
  `FNode` 12 B (`MetadataParams` moved to a side pool - it was ~104 B inline in
  the first spike revision), `FStep` 20 B, `StrId` = `NonZeroU32`.
- Master's `at_end` (top-down) is recomputed bottom-up at metadata-exit as
  "every remaining Seq frame is exhausted"; the last-step precedence/assoc
  fix-ups apply per completed path.

Caveat: the 300-iter loop retains output-pool capacity across iterations
(steady-state framing, same convention as the intern spike's pre-cloned
pools); a one-shot run additionally pays first pool growth, which is small
next to master's per-alternative tree cloning.

### CAVEATS on these numbers (read before trusting them)

- **`intern_symbols` is the BEST case for in-place** - it's a pure leaf rewrite, so
  master needlessly rebuilds a whole tree. `flatten_grammar` (tree -> productions)
  CANNOT be done in place. **Do not extrapolate 2-3x to the whole backend.** The
  flatten spike is the honest test.
- The flat timing excludes `Rule->pool` conversion (legit: a real frontend would
  build the pool directly) and pre-clones pools (legit: measures per-invocation cost;
  in the real pipeline the pass runs once, no clone).

## `build_tables` under the flat IR (quick investigation, 2026-07-06)

`build_tables` is 85-97% of `generate`, so this is where a generate-wide win
must come from. Temp instrumentation (`[SPIKE tables-stages]`,
`[SPIKE item-ops]`; counters add a few percent to the counted phases):

| grammar | item_set_builder | build_parse_table | token_conflicts | keywords+error+used | minimize | lex |
|---|---|---|---|---|---|---|
| c | 9ms | 278ms | 23ms | 139ms | 94ms | 6ms |
| cpp | 44ms | 2639ms | 50ms | 1238ms | 1746ms | 41ms |
| rust | 29ms | 919ms | 10ms | 283ms | 226ms | 10ms |

| grammar | item hashes | steps walked by hashes | item eqs | item cmps |
|---|---|---|---|---|
| c | 0.47M | 1.5M | 1.5M | 5.2M |
| cpp | 3.4M | 10.3M | 9.3M | **50.7M** |
| rust | 1.3M | 4.0M | 4.2M | 24.1M |

**The finding: `ParseItem` identity is structural over the entire production.**
`Hash`/`PartialEq`/`Ord` in `build_tables/item.rs` walk every step, hashing and
comparing `Precedence` (string for named), `Option<Alias>` and field-name
`String`s per step. `ParseItemSet::insert` binary-searches with that `Ord`
(the 50.7M cmps); state dedup (`state_ids_by_item_set`, `core_ids_by_core`)
hashes and compares entire item sets through the same impls. On cpp that is
~60M string-touching, production-length identity operations inside the 2.6s
`build_parse_table`.

Why it is this way today: items hold `&Production`, and dynamic inlining
(`substitute_production`) creates distinct-but-equal productions, so identity
cannot be pointer-based; and item equivalence is deliberately COARSER than
production equality (pre-dot steps are ignored except their aliases/fields,
see `has_preceding_inherited_fields`), which merges states.

**The flat-IR design that collapses this:**
1. Pooled, deduped productions - exactly what the flatten rework's output
   already is - make production identity an id; `substitute_production`
   becomes an id swap (inlined productions join the pool).
2. Precompute, once per `(production, dot)` pair, an interned
   **item-equivalence key**: hash-cons the exact projection the current
   `Hash`/`Eq` use (dyn prec, steps len, prec/assoc at dot, completed steps'
   alias+field (+symbol under the inherited-fields flag), remaining steps).
   Pairs are bounded (total steps across the grammar), so this is the one
   legitimate hash-consing site: bounded, precomputed, off the hot path -
   unlike attempt #1's per-rebuild consing.
3. `ParseItem` becomes `{ key: u32, prod: u32, dot: u32 }`, `Copy`;
   hash/eq/cmp are integer ops; item-set hashing is ids + `TokenSet` words;
   the 50.7M cmps become u32 compares.
4. Step strings as `StrId`s also turn conflict-resolution precedence lookups
   (scans of `precedence_orderings` with string equality) into id lookups.

Expected impact, stated conservatively: if the ~60M identity walks are half of
cpp's `build_parse_table`, collapsing them saves ~1.3s of cpp's 6.4s generate;
c/rust proportional. Explicitly NOT addressed by this lever: `minimize`
(1.75s) and the `keywords+error+used` bundle (1.24s) - separate
investigations (state-refinement and O(states x tokens^2) shapes, not
item-identity-bound).

Oracle for any of this: byte-identical `parser.c` across fixture/corpus
grammars - the equivalence-key projection must reproduce the current item
equivalence exactly, or state counts (and tables) change.

Landing: with the rework or immediately after it - the key layer REQUIRES
pooled deduped productions, so it slots in as the first post-flatten consumer
of the new `SyntaxGrammar`.

## `Symbol` shrink (investigated, not done)

`Symbol { kind: SymbolType, index: usize }` today: 16 B; `Option<Symbol>` 16 B
(rides `SymbolType`'s niche).
- Runtime `TSSymbol` is `uint16_t` -> a valid parser has <= 65535 symbols, so
  `index: u32` is **free** (4-billion headroom, `as u32` never truncates, no overflow
  checks). Shrinks `Symbol` to 8 B, `Option<Symbol>` to 8 B.
- `u16` would be 4 B but needs `try_from` overflow guards at every construction (the
  65535 ceiling is reachable for huge grammars; `as u16` truncates silently). Not
  worth it. **Recommendation: u32.**
- Blast radius: ~119 `.index` array-index sites need `as usize`; constructors
  (`Symbol::{non_terminal,terminal,external}`) take `usize` -> take/cast `u32`.
- **Do this as its OWN general optimization**, measured against `build_tables` /
  `render` (the symbol-heavy phases that actually benefit), NOT judged by this spike.

## Next steps (priority order)

1. ~~**`flatten_grammar` spike** = the go/no-go.~~ DONE 2026-07-06: GO (7-13x,
   output byte-equal; table above).
2. **Validate both spikes on the corpus machine** (cpp especially; it is the
   choice-heaviest grammar and the fixture checkout cannot run it).
3. **Land the name-map win on master independently** (`intern_name` O(V) scan -> O(1)
   map). It cut cpp `intern_symbols` 254 -> 152us with zero pool. Cheap, standalone.
4. **`Symbol` usize -> u32** shrink, standalone, measured vs `build_tables`.
5. Write the real design doc (all pass access patterns; converge the string
   interner with the nativedsl lex-time interning design in
   `plans/nativedsl-string-interning-design.md`; include the
   `build_tables` item-equivalence-key layer, whose prerequisite is the
   pooled deduped productions), THEN implement pass-by-pass keeping green +
   measuring each. Do NOT implement-before-measuring (the attempt-#1
   mistake).

## How to run the spike

```
cargo build --release -p tree-sitter-cli
for g in cpp c python go javascript; do
  ( cd ~/projects/grammars/tree-sitter-$g \
    && /path/to/tree-sitter/target/release/tree-sitter generate grammar.js 2>&1 ) \
    | grep '\[SPIKE'
done
```
`generate` runs `prepare_grammar` once; the temp bench block there prints the A/B and
then generation continues to `build_tables` (slow for cpp - the number is already
printed, Ctrl-C or wait). Grammars live at `~/projects/grammars/tree-sitter-<lang>/`.

## Temp code inventory (all on `rule_ir`, remove/relocate before landing)

- `crates/generate/src/prepare_grammar/flat_spike.rs` - the whole spike module
  (flat `Pool` + `Rule->pool` converter + flat `intern_symbols`). Plus
  `mod flat_spike;` in `prepare_grammar.rs`.
- The `[SPIKE intern_symbols] ...` bench block at the top of `prepare_grammar`
  (`prepare_grammar.rs`).
- `intern_symbols.rs`: the temp O(1) `name_map` (Interner field + `FxHashMap` import +
  `intern_name` rewrite). **This one is a genuine independent win** - consider KEEPING
  it / landing it on master rather than reverting (see next-step 2).

## Git state

- Main repo: branch `rule_ir` (this doc + spike committed here). Base = `native_dsl`
  @ `63e862ec1` (the last of 10 revert commits undoing pool attempt #1; tree ==
  `d43966acc`).
- `native_dsl`: the reverted feature branch (pool #1 undone, iterative-passes work
  kept). Do not confuse with `rule_ir`.
- dsl-tests repo: branch `rule_ir`; reverted `00e5b0f` (its pool adaptation).
- master baseline: `aec288f83`.

## Relevant memory files (`~/.config/chud/.../memory/`)

- `nativedsl_pool_reverted_lesson.md` - why attempt #1 failed (READ FIRST re: pool).
- `nativedsl_iterative_passes_followup.md` - the KEPT depth-safety work + recursive
  `bench_lower_only` baselines (C++ ~272-290us).
- `nativedsl_rule_elimination_feasibility.md` - attempt #1's approach (what NOT to do).
