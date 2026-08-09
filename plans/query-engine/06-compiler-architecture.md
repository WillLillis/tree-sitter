# Compiler architecture: stages, IR, and a stable bytecode

The brief asked to "treat the query builder as a real compiler, rather than an ad hoc
implementation", with "better separated stages, consumable as a library". This is the design
for that.

## Why the current shape resists change

Concretely, from [`01-current-state.md`](01-current-state.md):

- `ts_query__parse_pattern` emits `QueryStep`s as a side effect of parsing, and back-patches
  `alternative_index` during the parse. There is no point at which a complete, inspectable
  representation of the user's query exists.
- Because of that, alternation linking and quantifier linking contend for the same field, and
  the fix is a **post-hoc code-rewriting pass** (`query.c:3144-3189`) that clones steps and
  splices redirect nodes into the already-emitted program.
- Because there is no IR, the public introspection API is keyed by **source byte offset**
  (`ts_query_is_pattern_guaranteed_at_step`), and diagnostics are a single offset plus a
  7-variant enum.
- Because analysis results live in the same struct as the instructions, "what the user wrote",
  "what we derived", and "what the machine runs" cannot be separated, versioned, cached, or
  serialized independently.

Every one of those is a consequence of the missing stage boundary, not of any individual
decision.

## Proposed stages

```
  .scm source
      │
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 1. LEX + PARSE                                               │
  │    → CST/AST with full spans, error recovery, no language    │
  │      knowledge required                                      │
  │    Library surface: parse a .scm without a TSLanguage        │
  └───┬──────────────────────────────────────────────────────────┘
      │  AST
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 2. RESOLVE                                                   │
  │    → HIR: symbols/fields/supertypes resolved against the     │
  │      TSLanguage; captures interned; predicates parsed into   │
  │      structured form; quantifiers computed                   │
  │    All "unknown node type / field / bad supertype" errors    │
  │      surface here, with spans, and ALL of them at once       │
  └───┬──────────────────────────────────────────────────────────┘
      │  HIR  (pattern tree, still structural — no control flow)
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 3. ANALYZE                        [cacheable, parallel]      │
  │    → per-step guarantees, feasibility, selectivity estimates │
  │    Split: language-only facts (cache on TSLanguage)          │
  │           vs pattern-specific facts (cache by pattern hash)  │
  └───┬──────────────────────────────────────────────────────────┘
      │  HIR + facts
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 4. PLAN / OPTIMIZE                                           │
  │    → choose start point (root vs most-selective node),       │
  │      order sibling constraints, merge shared prefixes across │
  │      patterns, decide predicate pushdown                     │
  │    This stage does not exist today at all.                   │
  └───┬──────────────────────────────────────────────────────────┘
      │  LIR / plan
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 5. LOWER → BYTECODE                                          │
  │    → explicit opcodes, no overloaded flags                   │
  │    Serializable, versioned, verifiable                       │
  └───┬──────────────────────────────────────────────────────────┘
      │  bytecode
  ┌───▼──────────────────────────────────────────────────────────┐
  │ 6. EXECUTE (VM)                                              │
  └──────────────────────────────────────────────────────────────┘
```

The stage boundaries are the product. Each of 1→2→3→4→5 should be callable independently,
which is what makes a query LSP, a formatter, a linter, and an optimizer possible without any
of them re-implementing the parser.

## The IR: two levels, not one

**HIR — structural, close to what the user wrote.** A tree, not a flat array:

```
Pattern {
  root: Node
  predicates: [Predicate]
  span: Span
}
Node =
  | Symbol   { symbol, supertype?, field?, negated_fields: [FieldId],
               captures: [CaptureId], children: [Child], anchored_start, anchored_end,
               is_missing, span }
  | Wildcard { named: bool, field?, captures, children, ..., span }
  | Alt      { branches: [Node], captures, span }
  | Quant    { inner: Node, kind: ?|*|+, captures, span }
  | Group    { items: [Node], span }
Child = { node: Node, immediate: bool }
```

Properties that matter:

- **Captures live on nodes, as a list with no fixed bound** — A1 in
  [`03-correctness.md`](03-correctness.md) disappears structurally.
- **Negated fields are a list, unbounded** — A3 disappears structurally.
- **Anchors are properties of positions**, not a flag that gets transferred between steps by
  runtime logic. The four anchor-semantics commits in this tree become HIR-level rules that
  can be stated and tested in isolation.
- **Spans everywhere** — diagnostics, LSP, formatting, and `--explain` all become possible.
- This is the level at which a query **formatter** and **linter** operate, and the level at
  which query composition (`03` §C4) would be defined.

**LIR/bytecode — explicit control flow.** A flat instruction array, one opcode per operation:

```
  ; matching
  MATCH_SYMBOL   sym                 ; fail thread if current node symbol != sym
  MATCH_WILDCARD named
  MATCH_SUPERTYPE super
  CHECK_FIELD    field
  CHECK_NEG_FIELDS  list_id
  CHECK_MISSING
  CHECK_FIRST_CHILD                  ; leading anchor
  CHECK_LAST_CHILD                   ; trailing anchor
  ; captures
  CAPTURE        capture_id
  CAPTURE_PARENT capture_id          ; the wildcard-root optimization, made explicit
  ; control flow
  SPLIT          a, b                ; fork: one thread to a, one to b
  JMP            target
  DESCEND                            ; enter child level
  ASCEND
  ACCEPT         pattern_id
```

The point is not the exact opcode list. The point is that `alternative_index` +
`is_pass_through` + `is_dead_end` + `alternative_is_skip` — four fields whose combinations
encode four different control-flow constructs — become four distinct opcodes. The table in
[`01-current-state.md`](01-current-state.md) §"Control flow is encoded in flag combinations"
becomes unnecessary because the code says what it means.

This is a Thompson NFA program, deliberately. It makes the regex literature
([`05-database-angle.md`](05-database-angle.md) §2) directly applicable, and it is the form
that a later TDFA construction consumes.

## Stable serialized bytecode

Real benefits, in order of value:

1. **Precompilation eliminates the compile cost entirely.** 16.9 ms per Rust highlights
   compile ([`04-performance.md`](04-performance.md)) becomes a `mmap` + a symbol-resolution
   pass. For an editor loading 60 query files this is the difference between ~500 ms and
   ~nothing.
2. **Cross-implementation conformance.** The Rust, WASM/JS, Go, Python, and C implementations
   currently agree only by construction. A specified bytecode plus a conformance corpus makes
   divergence detectable.
3. **Third-party tooling.** Optimizers, visualizers, coverage tools, and differential fuzzers
   all become possible against a documented artifact.
4. **Golden testing.** Bytecode snapshots catch unintended semantic changes in the compiler —
   exactly the class of regression the four anchor commits in this tree represent.

### The hard part: symbol ids are not stable

`TSSymbol` and `TSFieldId` values are assigned by `crates/generate` and change whenever a
grammar is regenerated. Serialized bytecode that embeds raw symbol ids is invalidated by any
grammar change, silently and dangerously (a stale id is still a *valid* id, just the wrong
node type).

Recommended design:

- **Serialize names, not ids.** Store a string table of node-type names, field names, and
  supertype names; the bytecode references string-table indices. Loading performs a resolution
  pass — one hash lookup per distinct name, so tens of lookups, microseconds.
- **Stamp the artifact** with the grammar name, its semantic version (already available via
  the `ts_language_*` version API added in `8bb1448a6`), the language ABI version, and a
  content hash of the parse table. Refuse to load, or fall back to source compilation, on
  mismatch — never silently accept.
- **Version the bytecode format independently** of both the grammar and the library, with an
  explicit `(major, minor)` and a documented compatibility policy.
- **Validate on load.** A verifier pass — jump targets in range, stack discipline sound,
  capture ids within bounds — so that a corrupt or hostile artifact cannot produce memory
  unsafety. This is the WebAssembly lesson: a bytecode format without a specified validation
  pass is an attack surface.

### Where the analysis results live

Deliberately **not** in the bytecode. Analysis output (`root_pattern_guaranteed`, selectivity
estimates) is a function of `(pattern, language)` and is an *optimization input*, not
semantics. Keeping it in a separate side table means:

- bytecode stays a pure description of matching semantics, so golden tests over it are stable
  when only the analyzer changes;
- analysis can be cached, recomputed, skipped, or improved independently
  ([`04-performance.md`](04-performance.md) P1/P4);
- an implementation that skips analysis entirely is still *correct*, just slower — a much
  better property than today, where analysis failure conservatively degrades but is entangled
  with the step array.

## Library surface

The stages should be a Rust crate (`tree-sitter-query`?) that the C library can also use, or
at minimum a documented C API mirroring the stage boundaries:

```
parse(source)                     -> Ast | [Diagnostic]
resolve(ast, language)            -> Hir | [Diagnostic]
analyze(hir, language, cache)     -> Facts
plan(hir, facts, options)         -> Plan
lower(plan)                       -> Bytecode
serialize/deserialize(bytecode)   -> bytes
verify(bytecode)                  -> Result
```

### A real consumer to design against: `ts_query_ls`

[`ts_query_ls`](https://github.com/ribru17/ts_query_ls) (ribru17) is a language server for
`.scm` files: diagnostics for impossible patterns and invalid node names, completion for node
names / fields / captures / predicates, nvim-treesitter-compatible formatting, and
go-to-definition, references, and rename for captures. It is the closest thing to a real-world
test of the API this document proposes.

Worth treating as a **design data point, not a requirement** — nothing here should gate on it.
But three things about how it is built are direct evidence for the diagnosis in
[`01-current-state.md`](01-current-state.md):

1. **It parses `.scm` with a tree-sitter grammar for the query language**, not with
   tree-sitter's own query parser. That is the only option available: `ts_query_new` either
   returns an opaque `TSQuery` or one byte offset, and neither is something a language server
   can build on. The ecosystem's answer to "there is no AST" was to write a second parser for
   the same language. Stage 1 of the pipeline above is exactly the thing that would not have
   needed writing.

2. **Its documented limitation is our §C5.** Impossible-pattern detection "requires expensive
   full query file execution". The library *already computes this* — that is what the
   `analysis.finished_parent_symbols.size == 0` path in `ts_query__analyze_patterns`
   (2013-2049 region) decides — but the only way to consume it is to compile the whole file
   and get back a single `TSQueryErrorStructure` at a single byte offset. Finding the *second*
   bad pattern means editing it out and compiling again. The information exists and is thrown
   away at the API boundary.

3. **It implements `; inherits:` module imports itself**, because the query language has no
   composition ([`03-correctness.md`](03-correctness.md) §C4). Another capability the ecosystem
   built around the library rather than in it.

This gives a concrete acceptance test for the staged API, worth checking before committing to
a shape: **could `ts_query_ls` replace its expensive impossible-pattern path with a single
`resolve()` + `analyze()` call returning every diagnostic with a span?** If yes, the stage
boundaries are drawn in the right places. If it still needs to compile repeatedly, they are
not. Validating a design against an existing consumer is much cheaper than discovering the
boundaries are wrong after shipping them.

Consumers this immediately unlocks, none of which are possible today:

- **query LSP** — completion of node types and fields from the grammar, go-to-definition on
  captures, diagnostics with spans, hover showing whether a step is guaranteed.
- **formatter** — `.scm` files have no canonical format; every downstream project formats
  differently.
- **linter** — "this pattern can never match" (already computed, currently only surfaced as a
  compile error), "this capture is never used by any predicate", "this pattern is subsumed by
  pattern 4", "this quantifier shape is known-quadratic" (which would have flagged the 4.2 s
  query at authoring time).
- **`tree-sitter query --explain` / `--profile`** — the DB EXPLAIN analogue.
- **coverage** — which patterns in highlights.scm never fire on a corpus.

## Migration strategy

The strategy that makes this survivable is **not** a rewrite behind a flag day.

1. Build stages 1–2 (parse → HIR) as a **new, independent** path that produces the *existing*
   `QueryStep` array as its output. Same engine, same semantics, new front end.
2. Differential-test it: for every `.scm` in the fixture corpus plus generated queries,
   assert the new front end emits a byte-identical step array to the old one. Where it
   differs, either the old one has a bug (fix and record) or the new one does.
3. Only once the front end is proven, introduce the bytecode as a second output, and the new
   VM behind a runtime switch, differential-tested on *match results* over a corpus rather
   than on internal structures.
4. The disambiguation change (P7) is the one step that intentionally changes results. It must
   be gated on the written spec ([`03-correctness.md`](03-correctness.md) §C1) and a
   conformance suite, and it should land last.

This ordering means every intermediate state is shippable, and the risky change happens once,
late, with the strongest possible test coverage behind it.
