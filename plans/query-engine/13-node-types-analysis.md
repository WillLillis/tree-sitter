# Replacing query analysis with a generated node-type schema

**Status: feasibility established, one schema gap identified, persistence undecided.**

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
