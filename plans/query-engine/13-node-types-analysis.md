# Replacing query analysis with a generated node-type schema

**Status: feasibility established end to end. Two schema additions identified, both derivable
from `grammar.json`. Persistence mechanism undecided. Not yet prototyped or measured.**

## Why this is the broadest lever available

`ts_query_new` is **16–20 ms per real query file and 96–99% of it is
`ts_query__analyze_patterns`** ([`04-performance.md`](04-performance.md)), paid on every call,
by every consumer, in every process. It is the widest confirmed cost in the whole
investigation — wider than the execution work, which by measurement helps ~18 of 1,672 real
query files ([`12-handoff-merge-matcher.md`](12-handoff-merge-matcher.md) §3).

And the analysis is re-deriving, by simulation over the parse table, facts that `generate`
already computes and then discards into `node-types.json` — which the C library cannot see.

## What the analysis actually produces

Two things, from `ts_query__perform_analysis`:

1. **Per-step guarantees** (`parent_pattern_guaranteed` / `root_pattern_guaranteed`) — "if this
   step is reached, it and its remaining siblings must match". Consumed by `next_capture`
   streaming and by fallible-step splitting.
2. **Impossible-pattern rejection** → `TSQueryErrorStructure`.

Both are *optimisation and diagnostics*, not matching semantics. An implementation that
computed weaker guarantees would be **slower, not wrong** — which is what makes a conservative
approximation viable at all.

## Experiment: how much can the schema recover?

Derivation rule used, over 5 grammars' `highlights.scm`, comparing per-step against the C
analyzer's own verdict (dumped via `QP_DUMP_STEPS=1`):

> A field-constrained step `field: (T)` under parent `P` is guaranteed iff
> `P.fields[field].required` and not `multiple`, and every type the field admits (after
> expanding supertypes) is matched by `T`.

| | count |
|---|---|
| child steps compared | 76 |
| ...of which field-constrained | 55 |
| C guaranteed, schema agrees | **10** |
| C guaranteed, schema says no — *guarantees lost* | **11** |
| **C fallible, schema says guaranteed — *unsound*** | **0** |
| both fallible | 55 |

**Sound on this sample, and recovers roughly half the guarantees.** Soundness is the property
that matters: a guarantee the analyzer would not give could make `next_capture` stream a
capture that gets retracted. Zero such cases.

## The gap, and it is closable in `generate`

**All 11 lost guarantees are positional (non-field) children, and nearly all are anonymous
tokens:**

```
rust        (scoped_identifier (::))      (macro_invocation (!))
rust        (type_arguments (<))  ((>))   (type_parameters (<))  ((>))
rust        (lifetime (identifier))
javascript  (template_substitution (${))  ((}))
python      (interpolation ({))  ((}))
```

The cause is precise: `node-types.json` lists 111 anonymous types as *top-level entries*, but
**every type listed inside a `children` set is named**. Anonymous children are omitted by
design — the schema describes the named tree shape. So there is no way to learn that
`type_arguments` always contains `<` and `>`, or that `lifetime` always contains `'`.

`type_arguments` illustrates it: `children.types` lists six named types and neither delimiter.
`lifetime` has no `children` entry at all.

### What `generate` would need to emit — and it recovers all 11

Not ordering — just **the set of children that appear in every production of a rule, including
anonymous ones**. A "mandatory children" set, computed over the rule expression: union across
`SEQ`, **intersection across `CHOICE`**, empty for `REPEAT`/`BLANK`, recurse through
`PREC`/`TOKEN`/`FIELD`/`ALIAS`.

**Verified: 11/11.** Running that evaluator over `grammar.json` for each lost case:

| parent | needs | recovered |
|---|---|---|
| rust `scoped_identifier` | `::` | ✅ |
| rust `macro_invocation` | `!` | ✅ |
| rust `type_arguments` | `<`, `>` | ✅ ✅ |
| rust `type_parameters` | `<`, `>` | ✅ ✅ |
| rust `lifetime` | `identifier` | ✅ |
| javascript `template_substitution` | `${`, `}` | ✅ ✅ |
| python `interpolation` | `{`, `}` | ✅ ✅ |

Sample sets: `rust/type_arguments → ['<','>']`, `rust/lifetime → ["'", 'identifier']`,
`python/interpolation → ['_f_expression','{','}']`.

Combined with the 10 the field rule already recovers, that is **21/21 — full parity with the C
analyzer on this sample**, from a ~30-line evaluator over data `generate` already holds in
memory.

For **impossible-pattern rejection** a second set is needed: the *possible* children including
anonymous ones (union rather than intersection). Same evaluator, same pass.

## Impossible-pattern rejection: nothing actually blocks it

The rejection path was the open risk, since a schema that is *more permissive* than the parse
table would let genuinely-impossible queries compile and silently match nothing — a diagnostic
regression. Tested against the shipped analyzer to see which kinds of impossibility it catches:

| kind | example | caught today | schema-derivable |
|---|---|---|---|
| child-set | `(function_item (string_literal))`, `(lifetime (block))` | ✅ `STRUCTURE` | **yes** — possible-children incl. anonymous |
| field type | `(function_declaration name: (statement_block))` | ✅ `STRUCTURE` | **yes** — `fields[f].types` |
| field cardinality | `(binary_expression left: … left: …)` | ✅ `STRUCTURE` | **yes** — `fields[f].multiple == false` |
| ordering | `(type_arguments (">") ("<"))` | ✅ `STRUCTURE` | **no** — needs sequence info |
| anchor position | `(type_arguments . (">"))`, `(type_arguments ("<") .)` | ❌ **compiles** | n/a — no parity to preserve |

Two useful facts fall out. The analyzer is **already incomplete** — it does not reject a pattern
anchoring a token to a position it can never occupy. And only the *ordering* case resists a
set-based schema.

**Auditing every `QueryErrorKind::Structure` assertion in `query_test.rs` (12 of them):**

| asserted-impossible pattern | kind |
|---|---|
| `(binary_expression left: (expression (identifier)) left: (expression (identifier)))` | field cardinality |
| `(function_declaration name: (statement_block))` | field type |
| `(call receiver: (binary))` | field type |
| `(identifier (identifier))`, `(true (true))` | leaf has no children |
| `(if_statement condition: (expression))` | field type via supertype |
| `(identifier/identifier)`, `(statement/identifier)`, `(statement/pattern)` | supertype/subtype — **already a separate parse-time path** (`query.c:2671-2692`), not `perform_analysis` |

**Every one is schema-derivable. None requires ordering.** The ordering case is something that
had to be constructed artificially; real impossible patterns are type and set violations —
typos, wrong node type, a field that cannot hold that child.

### Recommendation: keep rejection complete — it costs 1–2 KiB

An earlier draft of this section recommended **accepting** the ordering gap, on the grounds
that no in-tree test covers it and that the failing example had to be constructed artificially.
**That reasoning was wrong, twice over.**

*The evidence was unfalsifiable.* Impossible patterns are transient by construction: an author
writes one, gets `TSQueryErrorStructure`, fixes it, and commits the fixed version. They can
never appear in a committed corpus. So their absence from `~/grammar_dump` and from the test
suite is evidence the diagnostic **is working**, not that it is unnecessary — and the value of
the check is precisely at the authoring moment, which is the one place with no observable data.

*And the failure mode is the worst one.* `(type_arguments (">") ("<"))` would go from an
immediate compile error to compiling and silently matching nothing. "Compiles but never matches"
is the hardest query bug to diagnose, and eliminating it is the whole point of the tooling
direction.

*The alternative was dismissed without checking, and it is cheap.* A per-node-type **precedence
relation** — the ordered pairs `(x, y)` such that `x` can appear before `y` among that node's
children — is derivable from `grammar.json` by the same evaluator (cross-product across `SEQ`
members in order, union across `CHOICE`, all-pairs for `REPEAT` since repetition admits any
order). Verified on the failing case:

```
type_arguments:   '<' before '>' = True     '>' before '<' = False   -> correctly rejected
type_parameters:  '<' before '>' = True     '>' before '<' = False   -> correctly rejected
```

Size, as a dense bitmatrix over each rule's own child set:

| grammar | rules | precedence pairs | bitmatrix | widest child set |
|---|---|---|---|---|
| rust | 179 | 3,690 | **2 KiB** | 96 |
| javascript | 132 | 1,013 | 1 KiB | 27 |
| python | 148 | 1,004 | 1 KiB | 26 |
| go | 115 | 915 | 1 KiB | 24 |
| c | 179 | 1,356 | 1 KiB | 26 |

**1–2 KiB per grammar**, against `parser.c` files measured in megabytes. There is no trade to
make here.

### So the schema needs three sets, from one pass

All derivable from `grammar.json`, which `generate` already holds:

1. **mandatory children** (incl. anonymous) — intersection across `CHOICE` → per-step guarantees
2. **possible children** (incl. anonymous) — union → rejection by child-set
3. **precedence pairs** — ordered-pair relation → rejection by ordering

plus the field information already emitted (`types`, `required`, `multiple`), which covers field
type and cardinality rejection.

Together these give **full parity on both guarantees and rejection**, with no permissiveness
regression and no diagnostic lost. Anchor-position impossibility remains uncaught, but it is
uncaught today too, so nothing regresses.

### A hazard to design around

If the schema is used for **impossible-pattern rejection** as well as guarantees, the anonymous
gap becomes a correctness bug rather than a lost optimisation: a query matching an anonymous
child would find that child absent from the parent's `children` set and be wrongly rejected as
impossible. **Do not use the schema for rejection until anonymous children are represented** —
or keep rejection on the existing path.

## Expected saving: the ceiling is ~30×

From the existing breakdown of `rust/highlights.scm` (16.26 ms total):

| component | cost | fate under a schema-driven analyzer |
|---|---|---|
| `perform_analysis` | 12.96 ms | **gone** — replaced by table lookups |
| full parse-table scan | 2.72 ms | **gone** — it exists only to build the subgraphs `perform_analysis` walks |
| S-expression parse + rest | ~0.5 ms | unchanged |

So the ceiling is **16.26 ms → ~0.5 ms, roughly 30×** on query compilation. Note both halves go:
the scan is not independently useful, so this subsumes the P1 caching idea rather than stacking
with it.

**Conditional on rejection also moving.** If impossible-pattern detection stays on the existing
path, `perform_analysis` still runs and the saving is zero. That is why the "possible children
including anonymous" set matters as much as the mandatory one.

This is a ceiling derived from existing measurements, not a measured result — a prototype
analyzer would be needed to confirm the lookup path is as cheap as assumed. It should be: a few
hash lookups per step against a table, versus simulating hypothetical trees.

## Persistence: read-only program data, not necessarily an ABI change

An earlier draft framed this as an ABI addition. That was too narrow. The schema is
**read-only program data** — the same category as precompiled query bytecode
([`06-compiler-architecture.md`](06-compiler-architecture.md) §"Stable serialized bytecode") —
and the two should very likely share one mechanism, one versioning story, and one loading path
rather than inventing separate ones.

That reframing opens options that do not touch `TSLanguage`'s layout:

- **An additional exported symbol** in `parser.c` (`tree_sitter_<lang>_schema()`), resolved
  optionally. The loader already `dlsym`s `tree_sitter_<lang>`; a missing schema symbol simply
  means falling back to today's analysis. Purely additive, no version bump, old grammars keep
  working.
- **An additive C API** — `ts_query_new_with_schema(...)` — so the caller supplies the data from
  wherever it has it. Additive functions are not an ABI break, and this is what makes the Rust
  `Query` constructor idea work end to end rather than only for Rust: the Rust binding passes
  through to the same C entry point every other binding can use.
- **A sidecar artifact** carrying schema *and* precompiled queries together, which is where this
  most plausibly wants to end up.

The thing worth avoiding is a Rust-only path that leaves the C library, WASM, and C-API editors
on the slow route while creating a second analysis implementation to keep in agreement. Routing
the Rust constructor through a new C entry point avoids that at no extra cost.

## What is not yet known

- **How much time this actually saves.** The schema path should be table lookups rather than
  simulation, but that has not been measured. The floor is the parse-table scan (~2.7 ms on
  Rust), which is separately cacheable ([`04-performance.md`](04-performance.md) P1). Worth
  measuring before committing to an ABI change.
- **Whether losing ~half the guarantees costs anything measurable.** Guarantees only drive
  `next_capture` streaming and state splitting. The corpus already shows guarantees on just
  2.8–5.5% of steps, so the absolute effect may be negligible — but it should be measured on
  the `next_capture` path, not assumed.
- **Whether the mandatory-children set recovers all 11**, and whether it generalises beyond
  these five grammars. The scan should be repeated across `~/grammar_dump`.
- **Impossible-pattern rejection**: whether the schema can do it soundly once anonymous
  children are present, or whether it stays on the existing path.

## Reproducing

```sh
# per-step verdicts from the C analyzer
QP_DUMP_STEPS=1 ./target/query-profiler/qprobe rust \
    test/fixtures/grammars/rust/queries/highlights.scm <any-source> match | grep '^STEP'
```
Columns: `STEP  label  index  depth  symbol  field  root_guar  parent_guar  flags(P/D/S)`.
The comparison script lives in this document's history rather than in-tree; it is ~40 lines of
Python over that dump plus `node-types.json`.
