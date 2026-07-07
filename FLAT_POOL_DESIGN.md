# Flat-pool backend redesign - design

Status: draft for review, 2026-07-07. Companion to `FLAT_POOL_INVESTIGATION.md`
(measurements, spike history, decisions); this doc is the implementation
contract. Like the investigation doc, it lives at the repo root on `rule_ir`
and is deleted before any PR (its content becomes the PR description and
module docs).

Binding decisions carried in from the investigation:
- All or nothing: the backend moves fully to the pool IR; no adapter landing.
- Master-first landing, but developed and validated on `rule_ir` against the
  native frontend before the series is cherry-picked to a master-based branch.
- Byte-identical output is the oracle at every step.
- The standalone quick wins (intern_name O(1) map, `Symbol` u32) are NOT
  landed separately; they are absorbed into the pool design (dense
  `name_of_symbol`, u32 symbol indices in pooled types).

## 1. Goals and non-goals

Goals, in priority order:
1. Prepare-bounded workloads get the measured wins (tsg frontend and LSP-grade
   validation are the primary motivation; grammar.json/node-types emission and
   the roundtrip corpus are the same shape). Measured so far: intern 2.9-11.3x,
   flatten 8-12x, whole syntax path 2.9-4.2x with the middle unconverted.
2. Depth-safety: every pass is an iterative pool walk; `MAX_RULE_DEPTH` and the
   recursive `Rule` box-tree stack hazard are deleted.
3. Foundation for `build_tables`: pooled deduped productions enable the
   item-equivalence-key layer (investigation doc), where 85-97% of `generate`
   lives.
4. Both frontends (grammar.json and nativedsl) emit pools directly; the
   `Rule -> pool -> Rule` bridges exist only during development.

Non-goals for this series: `minimize_parse_table` and the
keywords/error/coincident bundle (separate levers, already partially optimized
upstream); NFA/lex redesign; render output changes; any change to generated
`parser.c` bytes.

## 2. The IR

Validated by the spikes; sizes const-asserted. Working names as in
`flat_spike.rs`; final module is `crates/generate/src/rule_pool.rs` (name TBD).

- `NodeId(u32)`, `StrId(NonZeroU32)` (so `Option<StrId>` is 4 B),
  `ParamsId(u32)`, `NodeRange { start: u32, len: u32 }`.
- `FNode`, 12 B, `Copy`:
  `Blank | String(StrId) | Pattern(StrId, StrId) | NamedSymbol(StrId) |
   Sym { kind: SymbolType, index: u32 } | Seq(NodeRange) | Choice(NodeRange) |
   Repeat(NodeId) | Metadata { params: ParamsId, rule: NodeId } |
   Reserved { rule: NodeId, ctx: StrId }`.
  `String`/`Pattern`/`NamedSymbol`/`Repeat` exist only before
  extract/expand; passes that run later treat them as unreachable.
- `FParams` (side pool, `Copy`): `prec: FPrec { None | Integer(i32) |
  Name(StrId) }`, `assoc: Option<Associativity>`, `dynamic_precedence: i32`,
  `alias: Option<FAlias { value: StrId, is_named: bool }>`,
  `field: Option<StrId>`, plus `is_token`/`is_main_token` (needed pre-extract;
  the spike dropped them because its passes ran post-extract).
- Interner: adopt the proven in-tree shape (nativedsl `StringPool`):
  `FxHashMap<Rc<str>, StrId>` + `Vec<Rc<str>>` - one allocation per unique
  string, map key shares it, dense reverse lookup for errors/serialize/render.
  This is the vocabulary nativedsl interning phases 1-2 adopt
  (`plans/nativedsl-string-interning-design.md`).
- Flatten output pools (these BECOME `SyntaxGrammar`): `FStep` (20 B, `Copy`:
  packed symbol kind+index, prec tag+value, alias/field as raw `StrId`s,
  reserved set id, flags byte) and `FProd { steps: StepRange,
  dynamic_precedence: i32 }`.
- Flatten scratch: `Frame` (8 B, 2-bit tag: `Seq{cur,end}` / `MetaExit{flags,
  steps_before}` / `ReservedExit`), fork protocol snapshots ALL consumable
  context stacks (frames, prec, assoc, alias, field, reserved) - a branch pops
  context pushed before the fork via enclosing exits, so truncate-only restore
  is wrong (the bug the equality oracle caught). Only `steps` is
  truncate-safe.

One pool per generate call, append-only, spanning the pipeline; passes orphan
nodes rather than freeing (arena semantics, measured fine). Single pool with
multiple root lists - the lexical split is by root ownership, not by moving
nodes (see extract_tokens below).

## 3. Grammar containers

- `InputGrammar`: pool-owning. `{ pool, name: String, variables: Vec<{name:
  StrId, kind?, root: NodeId}>, extra_roots, external_roots, reserved sets,
  config lists (conflicts/supertypes/inline as StrIds until interned,
  precedence orderings) }`. The reverted `95582080f` did this conversion once;
  its shape is the reference, minus the strings-still-owned mistake.
- `InternedGrammar` / `ExtractedSyntaxGrammar`: same pool, symbols resolved in
  place; extracted adds the lexical root list and `Symbol`-typed config lists.
- `SyntaxGrammar`: `FProd`/`FStep` pools + per-variable production ranges +
  passthrough fields. `Production`/`ProductionStep` as owned structs die.
- `LexicalGrammar`: lexical variable metadata + roots into the shared pool;
  `expand_tokens` walks the pool read-only.
- `InlinedProductionMap`: inlined `FProd`s appended to the production pool;
  `production_map` keyed by `(ProdId, u32 dot)` instead of
  `(*const Production, u32)` - the raw-pointer identity dies with the clones.

## 4. Pass-by-pass

Format: what it does today / cost shape / pool design.

- `validate_precedences`, `validate_indirect_recursion` (pre-pipeline walks):
  read-only tree walks; become read-only pool walks. Named-precedence checks
  compare `StrId`s against interned ordering entries.
- `intern_symbols`: today rebuilds every tree (clone-heavy). Pool: in-place
  `NamedSymbol -> Sym` leaf rewrite + dense `name_of_symbol: Vec<Symbol>`
  (names interned first occupy the low `StrId` range). Measured 2.9-11.3x.
  `check_single` diagnostics read child counts from ranges.
- `extract_tokens`: today TWO full rebuilds (extraction, then symbol
  renumbering), O(tokens^2) structural `Rule` equality for token dedup,
  O(replacements) index adjustment per symbol, string compares throughout.
  Pool: one in-place walk; a token subtree is "hoisted" by pushing its root
  onto the lexical root list and overwriting the syntax-side node with
  `Sym{Terminal}` in place - no nodes move, no second pool. Token dedup via a
  local structural memo (hash by walking pool nodes - integer content since
  strings are `StrId`s; equality likewise). Symbol renumbering via a
  prefix-sum `Vec<u32>` built once (O(1) per symbol). The
  variable-absorption step (single-use whole-rule tokens) is bookkeeping over
  metadata lists, unchanged in spirit.
- `expand_repeats`: today rebuilds every tree and hashes ENTIRE `Rule` trees
  as dedup-map keys. Pool: in-place `Repeat -> Sym` rewrite; aux variables are
  new roots wrapping `choice(seq(sym sym), inner)` built by appending a few
  nodes; dedup uses the same local structural memo keyed on the inner subtree.
- `flatten_grammar`: the validated spike. Fork-at-choice enumeration, zero
  steady-state allocation, output into the `FStep`/`FProd` pools, per-variable
  order-preserving dedup by `Copy`-slice equality. 8-12x measured, byte-equal
  verified on six grammars.
- `expand_tokens`: read-only walk over lexical roots; `Pattern`/`String`
  payloads resolve `StrId -> &str` at the regex-parse boundary. Shape
  otherwise unchanged (NFA construction is out of scope).
- `extract_default_aliases`: walks `FStep`s; alias identity becomes
  `(StrId, bool)` compares instead of `Alias` string compares; the output
  `AliasMap` keeps owned `Alias` (render-facing), resolved once.
- `process_inlines`: today clones whole `Production`s per inline expansion,
  dedups by linear structural scans (rust: 10.2 ms), keys the result map by
  production ADDRESS. Pool: expansion builds new step ranges by copying
  `FStep` slices (with `StrId` alias/field propagation - integer writes);
  dedup via the structural memo over step slices; map keyed `(ProdId, dot)`.
- `node_types`: consumer; walks `FStep`s, field/alias maps keyed by `StrId`,
  strings resolved at JSON output.
- `render`: consumer; resolves `StrId`s once at emission. No format changes.
- `serialize`/`grammar_to_json` (nativedsl) and generate.rs JSON output:
  pool walk -> `RuleJSON`. Requires the interner reverse lookup (have it).
- `build_tables`: unchanged in this series EXCEPT enabled by it - the
  item-equivalence-key layer (investigation doc: 50.7M item cmps on cpp
  collapse to u32 ops) lands with the series or immediately after.

## 5. Frontend contracts

- grammar.json (`parse_grammar`): `parse_rule` builds pool nodes bottom-up
  from `RuleJSON`, interning strings on the way; validation walks
  (`rule_is_referenced`, empty-string checks, `collect_referenced_names`)
  become pool walks.
- nativedsl: lowering emits the pool DIRECTLY; `build_rule` and the
  `ARule -> Rule` materialization die (this deletes the >99.9%-of-allocations
  hotspot measured in the interning doc). `ARule` and `FNode` are
  near-isomorphic; prefer deleting `ARule` in favor of `FNode` + `FParams`.
- **Metadata-merge parity (correctness-critical):** `Rule`'s builders merge
  successive wrappers into ONE `Metadata` node via `add_metadata`, EXCEPT
  across a token boundary (`!params.is_token` guard). Both frontends must
  reproduce exactly this merge when emitting `FNode::Metadata`, or flatten
  fix-up semantics and output diverge. Implement as one shared
  `params_builder` used by both frontends; oracle: byte-identical output on
  the roundtrip corpus (which exists to catch exactly this).
- The interner is shared end-to-end: nativedsl lex-time interning (phases 1-2
  of the interning doc) feeds the same pool the backend consumes - one
  interner per generate call.

## 6. Verification and measurement protocol

- Development: keep the spike-style A/B per converted pass (master impl still
  present, assert equal output) until the pass lands; delete the master path
  at series end.
- Oracles: byte-identical `parser.c` on all fixture grammars (and corpus);
  byte-identical `grammar.json` for the JSON emission path; nativedsl suite +
  127-grammar roundtrip corpus for the native frontend retarget.
- Full-run rule: never claim a check passed unless the run demonstrably
  executed it - capture output to a file and check exit codes (the
  pipe-truncation lesson is recorded in the investigation doc).
- Per-pass measurement gates, attempt-#1 lesson: a pass that regresses the
  A/B blocks the series until understood. Bench surface: `[SPIKE
  prepare-stages]`-style timing plus the dsl-tests counting-allocator harness.

## 7. Landing plan

1. Implement pass-by-pass on `rule_ir` (validates against the native frontend
   throughout, per maintainer). One commit per pass conversion, each green
   and measured. Frontends converted after the passes (parse_grammar first,
   then nativedsl lowering).
2. When the backend + parse_grammar frontend are complete and byte-identical:
   cherry-pick the series (minus nativedsl-specific commits and all temp
   instrumentation) onto a master-based branch; upstream PR with the
   benchmark evidence.
3. `native_dsl` rebases onto the landed backend; the lowering retarget (emit
   pools, delete `build_rule`/`ARule`) lands within the native PR.
4. Item-key `build_tables` layer follows as its own measured change.
5. Delete: `flat_spike.rs`, every `[SPIKE ...]` block, the item-op counters,
   both root-level investigation/design docs.

## 8. Risks and open questions

- Metadata-merge parity (section 5) is the highest-risk correctness seam;
  gated by the roundtrip corpus.
- `Rule` currently derives `Serialize`/`Hash` and is `pub` via the temporary
  `pub mod rules` (kept for dsl-tests; see second-review disposition #9).
  Deleting `Rule` entirely is the end state, but the dsl-tests roundtrip
  comparison must move to `RuleJSON`/`grammar.json` first. Sequence that
  swap early.
- `u16`/`u32` range guards: child counts and step counts keep the spike's
  `checked_len` pattern; frame payloads assert `< 2^30`.
- Error paths that need names mid-pipeline (e.g. `NoReservedWordSet`,
  `EmptyString`) resolve via the interner reverse table; every such site is
  an error path, so resolution cost is irrelevant.
- Pool memory: append-only orphans accumulate across passes; measured fine on
  the spikes (nodes are 12 B). If a corpus grammar shows pathology, a single
  compaction between extract and flatten is the escape hatch - do not build
  it speculatively.
- `START_PRODUCTION` in `build_tables/item.rs` (a static `Production`) needs
  a pooled equivalent when `SyntaxGrammar` goes flat; fold into the item-key
  change.
