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

### What `generate` would need to emit

Not ordering — just **the set of children that appear in every production of a rule, including
anonymous ones**. A "mandatory children" set. That is derivable at generation time and is far
simpler than positional/sequence information, and it plausibly recovers all 11 cases above,
since every one is mandatory punctuation.

Worth checking whether that also subsumes the field case, which would let one mechanism cover
both.

### A hazard to design around

If the schema is used for **impossible-pattern rejection** as well as guarantees, the anonymous
gap becomes a correctness bug rather than a lost optimisation: a query matching an anonymous
child would find that child absent from the parent's `children` set and be wrongly rejected as
impossible. **Do not use the schema for rejection until anonymous children are represented** —
or keep rejection on the existing path.

## Persistence and consumption: the open decision

Three options, and they are not equivalent in who benefits.

**A. In the language ABI**, as a static table in `parser.c` beside the parse table.
*For:* every consumer benefits automatically — C, Rust, WASM, editors — with no user action,
which is the only option that actually achieves the stated goal. *Against:* ABI addition,
binary size, version coordination, and it is the largest commitment.

**B. A new Rust `Query` constructor taking the schema** (the idea raised in discussion).
*For:* no ABI change, opt-in, shippable immediately, and an excellent vehicle for validating
the design against real queries. *Against:* only Rust consumers benefit; the C library, WASM,
and every editor on the C API keep paying the 16–20 ms; and it creates two analysis paths to
keep in agreement, which is exactly the class of divergence that is hard to test.

**C. Load `node-types.json` at runtime.** *Against:* file-path dependency, version skew between
schema and parser, and a new failure mode at query-compile time. Not recommended except as a
prototype.

**Suggested sequence:** use **B or an offline harness to validate**, then commit to **A** for
the actual win. B as a permanent answer would leave the broadest cost unaddressed for the
majority of consumers, which is the thing this workstream exists to fix.

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
