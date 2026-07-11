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

## Promoted passes (running table, updated 2026-07-07)

Spike retired (`flat_spike.rs` deleted). These are the real pool passes,
chained in `prepare_grammar` behind `[POOL <pass>]` TEMP A/B blocks: 300-iter
loops, master netted of input clones where it consumes input, full-output
equality asserted on every `generate` run (exit 0 on all six fixture
grammars). One coherent run, us/iter as master -> pool; WSL2 run-to-run
variance is ~2x on sub-ms master numbers, so treat ratios as indicative and
re-establish on the corpus machine.

| pass | c | cpp | python | go | javascript | rust |
|---|---|---|---|---|---|---|
| intern_symbols | 15.8x | 6.3x | 8.5x | 7.4x | 7.5x | 7.2x |
| extract_tokens | 5.4x | 5.8x | 5.4x | 6.1x | 3.8x | 4.8x |
| expand_repeats | 3.8x | 7.6x | 1.9x | 4.4x | 5.8x | 3.2x |
| flatten_grammar | 4.9x | 4.5x | 8.1x | 5.5x | 7.5x | 6.4x |
| chain (sum, us) | 3055 -> 575 | 5993 -> 1213 | 1619 -> 257 | 1314 -> 233 | 2037 -> 328 | 5069 -> 838 |

`expand_tokens` (2026-07-07) is converted for IR uniformity, not speed: the
pass is regex-parse/NFA-bound and that half is shared code, so it measures at
parity (0.95-1.29x across the six grammars; us master -> pool: c 1393 -> 1459,
cpp 1830 -> 1865, python 1197 -> 1090, go 942 -> 731, javascript 1062 -> 1094,
rust 871 -> 815). It removes the last `Rule`-typed consumer on the lexical
side (`ExtractedLexicalGrammar` dies at the flip) and produces the final
`LexicalGrammar` directly, so it has no materialize step. Parity details worth
remembering: the combined separator rule mirrors
`Rule::repeat(Rule::choice(seps + Blank))` including choice flatten + dedup
with NO singleton unwrap, and `expand_rule`'s Repeat leaves its placeholder
`Accept` state behind when the inner rule fails to expand (quirk preserved).

`extract_default_aliases` + `process_inlines` (2026-07-07, converted together
because inlines consumes the aliases-mutated grammar; n=50 loops, us master ->
pool):

| pass | c | cpp | python | go | javascript | rust |
|---|---|---|---|---|---|---|
| extract_default_aliases | 162 -> 71 | 60 -> 134 (*) | 106 -> 29 | 42 -> 14 | 88 -> 33 | 141 -> 114 |
| process_inlines | 158 -> 63 | 559 -> 125 | 733 -> 174 | 167 -> 74 | 951 -> 375 | **17533 -> 1063 (16.5x)** |

The rust `process_inlines` hotspot (the branch's biggest remaining prepare
term) is the payoff: master's O(created^2) dedup scan with per-step String
compares becomes a content-hash memo over `Copy` step slices, and the
address-keyed result map becomes `(production id, dot)` keys. Verified: alias
map, alias-mutated whole `SyntaxGrammar`, created inlined productions, and
the inline map (through a canonical `(variable, production, dot)` form, since
master's map is address-keyed) all asserted equal on the six grammars.
(*) master cpp aliases cannot genuinely be 3x faster than master c on a
superset grammar; netted sub-100us numbers at n=50 are noise.

The two pre-pipeline `validate_*` walks (2026-07-07) complete the set: parity
timing (69-198us both sides across the six grammars; the cost is the shared
name-graph DFS + ordering-conflict check, not the tree walk), converted so no
pass consumes `Rule` trees. The ordering-conflict half and `get_cycle` DFS are
shared with master, not duplicated; parity quirk preserved: neither walk
descends into `Reserved` subtrees.

With that, every `prepare_grammar` pass runs pool-side. Still Rule-based: only
the containers themselves (the flip).

## Master A/B baseline + the perf-followup levers (2026-07-08)

First true master-vs-branch measurement (interleaved min-of-3, item-op
counters stubbed out of the branch binary, grammar.json inputs so the JS
runtime stays out of RSS):

| grammar | wall master -> branch | peak RSS master -> branch |
|---|---|---|
| c | 0.62 -> 0.58s (-6%) | 160.7 -> 162.2MB (+1%) |
| cpp | 6.08 -> 5.68s (-7%) | 996 -> 1004MB (+0.8%) |
| rust | 1.54 -> 1.25s (-19%) | 302 -> 303MB (+0.4%) |

Faithful re-measurement 2026-07-10 (after stage 3 portable half; item-op
counters fully deleted from the branch; interleaved min-of-5, all six
fixtures, vs a `git archive`-built local master @ 9fc2f486a, whose
crates/generate is unchanged since aec288f83). parser.c and node-types.json
are byte-identical master vs branch on all six. Run distributions do not
overlap (cpp worst branch 5.58s < best master 5.93s):

| grammar | wall master -> branch | peak RSS master -> branch |
|---|---|---|
| c | 0.56 -> 0.54s (-4%) | 160.5 -> 161.6MB (+0.7%) |
| cpp | 5.93 -> 5.41s (-9%) | 996.3 -> 1004.1MB (+0.8%) |
| python | 0.29 -> 0.25s (-14%) | 77.8 -> 78.5MB (+0.8%) |
| go | 0.23 -> 0.20s (-13%) | 65.1 -> 66.3MB (+1.8%) |
| javascript | 0.58 -> 0.49s (-16%) | 160.4 -> 162.1MB (+1.1%) |
| rust | 1.42 -> 1.18s (-17%) | 301.6 -> 302.8MB (+0.4%) |

Perf campaign on `rule_ir` (2026-07-10, user decision: campaign runs here
BEFORE the port). Per-stage RSS attribution (/proc/self/status marks between
build_tables stages) found the peak was never where the levers assumed:
on cpp, `keywords+error+used` added +458MB, all of it
`CoincidentTokenIndex.entries` (a `Vec<ParseStateId>` per token pair, one
push per coincident pair per state, sole consumer: one all-states-have-word
boolean in `identify_keywords`). Two levers landed, both byte-identical on
all six fixtures vs master:

1. Coincident state lists -> a second n*n "coincident without word token"
   bitset (commit b6a2bbc24). cpp keywords stage 1.18s -> 0.18s, peak RSS
   1005 -> 541MB.
2. parse_state_info kernel dedup (commit 87e15dbc3): every state's kernel
   item set was cloned into `parse_state_info_by_id` while the identical
   set is the state-dedup map key at the same index. `ParseStateInfo` now
   carries preceding symbols + the moved-out dedup map. cpp -33MB retained.

Standing after both (interleaved min-of-5 vs master, all runs
non-overlapping):

| grammar | wall master -> branch | peak RSS master -> branch |
|---|---|---|
| c | 0.56 -> 0.39s (-30%) | 160.1 -> 92.5MB (-42%) |
| cpp | 5.65 -> 4.11s (-27%) | 996.2 -> 546.0MB (-45%) |
| python | 0.28 -> 0.23s (-18%) | 77.7 -> 60.6MB (-22%) |
| go | 0.22 -> 0.18s (-18%) | 64.8 -> 54.2MB (-16%) |
| javascript | 0.57 -> 0.41s (-28%) | 160.4 -> 100.3MB (-37%) |
| rust | 1.40 -> 0.91s (-35%) | 301.0 -> 175.7MB (-42%) |

Note the coincident-index lever is IR-independent (would apply to master
as-is); user decision: extract and land it upstream BEFORE the IR port
(extraction deferred; re-baseline the A/B when it lands).

3. minimize token_conflicts bitsets (2026-07-10, also IR-independent):
   attribution showed minimize's 1.75s was 1.06s in the first
   split_state_id_groups pass, and op counters showed token_conflicts did
   278M entry probes over 9.3M calls. Precomputed ConflictBits (per-state
   terminal bits, per-token conflict rows, keyword bits, internal/external
   mask) turn each call into a few word ANDs. cpp minimize 1.75 -> 1.40s
   (split1 1.06 -> 0.78s). Remaining split1 floor is the states_conflict
   merge-join itself (2.1M pairs, 101M steps); cutting it means cutting
   pair count - algorithmic risk, parked.

Standing after lever 3 (min-of-3 interleaved): cpp 5.73 -> 4.01s / 996 ->
547MB; rust 1.41 -> 0.92s / 302 -> 176MB; c 0.54 -> 0.42s / 160 -> 93MB;
js 0.58 -> 0.42s / 160 -> 100MB.

Remaining cpp buckets: build_parse_table 2.08s / 498MB plateau (retained:
table 173MB + kernels 33MB; rest is closure churn + malloc slack - the
TokenSet-interning and ParseItem-shrink levers), minimize 1.40s (merge-join
floor), render 154ms.

4. Lookahead TokenSet interning (2026-07-11, the pool-dependent lever):
   `ParseItemSetEntry.lookaheads` is a `LookaheadSetId` into a
   `LookaheadSetPool` (canonical ids via TokenSet's own Hash/Eq, so dedup
   semantics are untouched); unions and single-token inserts memoized by
   id; closure additions carry interned ids + precomputed word membership.
   cpp: 3566 distinct sets serve 49992 states (0.2MB of set payload),
   kernels map 33.4 -> 18.7MB, build_parse_table 2.08 -> 1.72s.

Standing after lever 4 (min-of-5 interleaved vs master): cpp 5.95 -> 3.54s
(-41%) / 996 -> 529MB (-47%); rust 1.43 -> 0.80s (-44%) / 302 -> 171MB
(-43%); c 0.58 -> 0.37s (-36%) / 160 -> 90MB (-44%); js 0.58 -> 0.36s
(-38%) / 160 -> 96MB (-40%). Remaining cpp: build_parse_table 1.72s / 479MB
plateau (table 173MB accounted; ~300MB is IndexMap/action-vec malloc slack),
minimize ~1.4-1.5s, ParseItem shrink still open.

5. ParseItem shrink (2026-07-11): items store a u32 production id instead
   of ProdRef (identity only reads the keys slice; `is_done` is
   `step_index + 1 == keys.len()`); step content derefs through the grammar
   at ~15 sites; dead `ProdRef.id` dropped. ParseItem 56 -> 32B, entries
   72 -> 48B (halves closure-insert memmove). cpp build_parse_table
   1.72 -> 1.53s, kernels map 18.7 -> 12.4MB, plateau 479 -> 471MB.

CAMPAIGN STANDING after all five levers (min-of-5 interleaved vs master):

| grammar | wall | peak RSS |
|---|---|---|
| c | 0.60 -> 0.38s (-37%) | 160.5 -> 88.3MB (-45%) |
| cpp | 5.99 -> 3.53s (-41%) | 996.4 -> 522.1MB (-48%) |
| javascript | 0.62 -> 0.37s (-40%) | 160.4 -> 93.9MB (-41%) |
| rust | 1.49 -> 0.82s (-45%) | 301.3 -> 166.9MB (-45%) |

Levers 6 and 7 (2026-07-11): get_production_id memoized by prod_id (the
info depends only on the production; was rebuilt with String allocs plus a
linear deep-equality scan per reduce item per state; rust build_parse_table
579 -> 417ms) and split1 signature bucketing (states identical under the
frozen group ids are interchangeable in states_conflict; greedy loop runs
over representatives; cpp split1 783 -> 615ms). Standing: cpp 5.78 ->
3.18s (-45%) / 996 -> 521MB (-48%); rust 1.43 -> 0.77s (-46%) / 301 ->
166MB (-45%).

Byte-identical parser.c + node-types.json vs master on all six fixtures
after every lever. Remaining known levers, unranked: minimize merge-join
pair verification (equal-signature bucketing + union-row fast negatives -
needs a counter run to size; order-preservation is the risk), the parse
table's 173MB representation (per-entry Vec<ParseAction> + IndexMap
overhead), the ~300MB malloc slack (closure entry vecs + table maps), and
the "better partitions" idea (upstream semantics change, post-port
discussion only).

Earlier honest reading (pre-campaign): end-to-end time was a modest real
win; peak memory was flat because the peak was never the grammar
representation. On cpp the ~1GB
lives in build_tables' own structures: per-entry lookahead `TokenSet`s
(cloned per transitive-closure addition), `parse_state_info` retaining a
full `ParseItemSet` per state for conflict reporting, and the parse table.
The +1% is the fatter `ParseItem` (~56B vs ~24B: `ProdRef` + keys slice).

The maintainer bar for the master port is demonstrated, meaningful time AND
memory wins. Levers for the perf campaign, in expected-impact order (the
IR makes all of these cheap to implement now; none are speculative
representation changes):

1. **Lookahead `TokenSet` interning** - the item-key trick applied to the
   other half of `ParseItemSetEntry`: rank-intern lookahead sets, store ids,
   memoize unions. Kills the per-addition set clones, turns item-set
   hash/eq's set traversals into int ops, and dedups massively-repeated
   sets (time AND memory; likely the biggest single lever).
2. **`parse_state_info` retention** - a `(SymbolSequence, ParseItemSet)` per
   state held for the whole build, used only for conflict reporting. Store
   interned/compact forms, or materialize lazily on the conflict path.
   Probably the biggest pure-memory lever.
3. **`ParseItem` shrink** - `prod_id: u32` + `DotKeys` base index instead of
   `ProdRef` + keys slice (identity already goes through keys; step access
   can deref through the grammar at the few use sites). Item back to ~16B,
   halving item-set memory traffic.
4. **minimize / keywords+error+used / token_conflicts** - untouched by the
   IR (cpp: ~4s combined); need profiling, likely state-signature grouping
   and bitset-operation work.

Also in the master-PR value story beyond generate wall time: the
prepare-bound paths (tsg frontend, LSP-grade validation, grammar.json
emission: 5-7x), the native frontend's alloc-bomb deletion (stage 3),
depth-safety (`MAX_RULE_DEPTH` removal), and rust `process_inlines` 16x.

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

1. ~~**Remaining pass conversions**~~ DONE 2026-07-07: every prepare pass has
   a verified pool twin.
2. **Container flip**, staged (stage 1 DONE 2026-07-07): the pipeline runs
   pool passes only; master passes, `Interner`/`TokenExtractor`/
   `RuleFlattener`/`InlinedProductionMapBuilder`, and every A/B block are
   deleted; the pool passes own the plain names. Legacy-shaped
   `SyntaxGrammar`/`InlinedProductionMap`/`AliasMap` materialize once at the
   boundary. Oracle: regenerated parser.c byte-identical on all six fixture
   grammars; 604/604 tests (pass tests now run the pool path against the old
   literal expectations; parity-only tests died with master). Real-pipeline
   rust aliases+inlines: 10.2ms -> 0.77ms.
   Stage 2 DONE 2026-07-08 in three verified commits: build_tables reads
   pooled storage (`ProdRef`/`FStep`, id-keyed inlining, `Prec` conflict
   resolution), then node_types/render/minimize walks, then the legacy drop
   (`SyntaxVariable::productions` is cfg(test) fixture vocabulary,
   `Production`/`ProductionStep` cfg(test), `InlinedProductionMap` is just
   the `(id, dot)` map, `assemble_syntax_grammar` moves the pools in with no
   clones). Byte-identical parser.c after every commit; interleaved min-of-3
   perf: parity with stage 2a (cpp -1.5%, rust prepare -1ms). Perf protocol
   note: single-shot WSL numbers swing +/-10-20%; regressions are checked
   with interleaved min-of-N runs of both binaries.
   Remaining: **stage 3** (frontends emit pools; `InputGrammar` pool-owning;
   delete `Rule` + the `add_rule`/`rule`/`from_input_grammar` bridges + the
   `pool_from_*` test bridges; dsl-tests `pub mod rules` exposure resolved).
   Stage 3 portable half DONE 2026-07-09: `RulePool` grew the `Rule`-mirroring
   constructors (`metadata_with` merge = `add_metadata`, `choice` =
   flatten+dedup), `parse_grammar` builds a `PoolGrammar` directly (`parse_rule`
   returns `NodeId`), `PoolGrammar::normalize` ports the used-set DFS +
   empty-string diagnostics + config cleanup, and `prepare_grammar`/
   `generate_*` take `PoolGrammar` by value, so the JSON path is pool
   end-to-end with zero grammar copies. Enabler: `intern_symbols`' dense-name
   contract relaxed to a `StrId`-indexed `Vec<Option<Symbol>>` (normalize
   drops variables from a parse-order pool, so names are no longer ids
   `1..=n`; nothing else consumed the ordering - the one derived-`Ord` use,
   the `shift_precedence` dedup, is order-insensitive). nativedsl still emits
   `Rule` and bridges via `from_input_grammar` at its `generate.rs` call site
   (TEMP; keeps the series separable for the master port). `to_input_grammar`
   test bridge added for the nativedsl `json_roundtrip` test. Oracle:
   parser.c + node-types.json byte-identical on all six (fresh HEAD baseline
   via `git archive` build - the user vetoed worktrees); 604/604 generate
   tests + `test_feature_corpus_files` (covers `unused_rules`, i.e. the
   normalize drop path); interleaved min-of-3: cpp -0.9%, rust -2.4%, RSS
   parity. Remaining stage-3 work (nativedsl half): lowering emits pool nodes
   directly (deletes `build_rule`/`str_to_string`), then delete `Rule`, the
   Rule-based `InputGrammar::normalize` (cfg-gated under `nativedsl` in
   `parse_grammar.rs`), and all remaining bridges.
3. ~~**Harden or drop the A/B blocks before corpus runs**~~ moot: the A/B
   blocks are gone with stage 1; corpus verification is now generate-and-diff
   against a master-built parser.c.
4. **Lock in the `FStep`/`Frame` bit packing** (bitflags! or a layout doc +
   roundtrip tests) before the master port (user: style/readability, can
   follow the flip).
5. **Corpus-machine validation**: generate the translated grammars, diff
   parser.c vs master output, roundtrip suite, `ident_histogram_corpus`.
6. ~~**`build_tables` item-equivalence-key layer**~~ DONE 2026-07-07 (stage
   2a): `ItemKeyMap` precomputes per-`(production, dot)` dense ids by ranking
   the whole key universe with the item-content comparator (strings compared
   once, at table-build start), plus a second id subdividing by preceding
   symbols for `has_preceding_inherited_fields` items. `ParseItem` carries the
   per-production key slice; Hash/Eq/Ord are now integer ops with identical op
   counts. Order-preserving rank assignment keeps item-set ordering, so
   parser.c stays byte-identical (verified on all six). build_parse_table:
   cpp -15%, rust -32%, c -30%, js -23%; build_tables total: rust 1.53 ->
   1.18s, cpp 6.19 -> 5.46s. Remaining stage-2b work: swap `SyntaxGrammar`
   storage to `FProd`/`FStep`, convert node_types/render/build_tables reads,
   delete `Production`/`ProductionStep` and the boundary materialization.

## How to run the A/B

```
cargo build --release -p tree-sitter-cli
for g in c cpp python go javascript rust; do
  target/release/tree-sitter generate -o /tmp/gen_$g \
    test/fixtures/grammars/$g/grammar.js > /tmp/ab_$g.txt 2>&1; echo "$g exit=$?"
done
grep '\[POOL' /tmp/ab_*.txt
```
Full runs to files + exit codes, never pipe-truncated (see the verification
correction above): the equality asserts must actually execute. Exit 0 means
every `[POOL]` assert passed; the `[SPIKE *-stages]` lines give real one-shot
stage times.

## Temp code inventory (all on `rule_ir`, remove/relocate before landing)

- The `[SPIKE prepare-stages]`/`[SPIKE generate-stages]`/`[SPIKE
  tables-stages]` prints (`prepare_grammar.rs`, `generate.rs`,
  `build_tables.rs`). The `[SPIKE item-ops]` counters (hot-path atomics in
  `ParseItem`'s Hash/Eq/Ord) were fully removed 2026-07-10 so A/B runs are
  faithful; resurrect via `git log -S ITEM_HASHES` if the perf campaign
  wants op counts again.
- `materialize_interned`/`materialize_extracted`/`materialize_flattened` and
  the `add_rule`/`rule` pool<->Rule bridges (die at the flip).
- TEMP derives: `Clone/Debug/PartialEq/Eq` on `IntermediateGrammar`,
  `PartialEq/Eq` on `SyntaxGrammar`, `Clone` on `ExtractedLexicalGrammar`,
  `PartialEq/Eq` on `Diagnostic`; TEMP `#[allow(dead_code)]` on
  `PoolGrammar::name` and `RulePool::str_count`.

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
