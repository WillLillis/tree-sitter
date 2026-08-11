# Persisting the query schema

**Status: sized across the full grammar corpus. A recommendation, not a decision.**

Prerequisite reading: [`13-node-types-analysis.md`](13-node-types-analysis.md) — what the schema
contains and why it replaces `ts_query__perform_analysis`.

## The sizing that settles most of the argument

`gen_schema.py` needs only `grammar.json` and `node-types.json`, both source files, so the whole
corpus can be measured without building anything. Across **295 grammars** from `~/grammar_dump`
(2 failed to parse):

| | KiB |
|---|---|
| min | 0.0 |
| **median** | **0.6** |
| **p90** | **8.1** |
| max (`tree-sitter-hoon`) | **1,234.7** |
| total, all 295 | 2.1 MiB |

**Exactly one grammar exceeds 100 KiB.** The next largest is `abl` at 69 KiB, then `julia` at 64.
For the median grammar the schema is 0.6 KiB against a `parser.c` measured in hundreds of
kilobytes to megabytes — three to four orders of magnitude smaller.

An earlier concern that size might force a separate artifact turns out to apply to **one grammar
out of 295**, and even that one is fixable — see below.

## Precedence is 81% of the weight, and it is separable

Splitting the schema into its components:

| grammar | child-sets | precedence | precedence share |
|---|---|---|---|
| hoon | 89.8 KiB | 1,234.7 KiB | **93.2%** |
| abl | 15.3 KiB | 69.4 KiB | 82.0% |
| julia | 6.0 KiB | 64.2 KiB | 91.4% |
| rust | 7.2 KiB | 21.3 KiB | 74.8% |
| c | 2.2 KiB | 1.8 KiB | 44.5% |
| **corpus total** | **0.51 MiB** | **2.13 MiB** | **81%** |

The child-sets — mandatory and possible children, which drive **guarantees** and **child-set
rejection** — total half a megabyte across 295 grammars. The precedence relation, which drives
**ordering rejection** alone, is everything else, and it is what explodes on outliers (hoon's
widest child set is 280 symbols, giving 6.6 M pairs as a dense bitmatrix).

That is a clean lever: **make each component independently optional**, with defined degradation.

| present | behaviour |
|---|---|
| no schema | today's `perform_analysis` — full parity, slow |
| child-sets only | guarantees + child-set/field/cardinality rejection; **no ordering rejection** |
| child-sets + precedence | full parity with today, at ~1 µs |

This also resolves the earlier disagreement about ordering rejection properly. An earlier draft
proposed dropping it as a blanket policy; that was rightly pushed back on. The data says keep it
for **294 of 295 grammars at negligible cost**, and drop it only where the relation is
pathological — a per-grammar decision made on measurement rather than a global one made on
assumption. A sparse encoding, or a cap on the widest child set, would likely recover hoon too.

## Options

**A. An additional exported symbol in `parser.c`** (`tree_sitter_<lang>_schema()`), resolved
optionally — the loader already `dlsym`s `tree_sitter_<lang>`, so a missing symbol just means
falling back to today's analysis.

- Sizing says this is comfortable for essentially every grammar.
- No `TSLanguage` layout change, no ABI version bump, old grammars keep working untouched.
- **Version skew is structurally impossible**: schema and parse table are generated together
  from the same grammar and shipped in the same object. Nothing can load a schema that does not
  match its parser. That is a strong argument, and it is the one thing a sidecar cannot offer.

**B. An additive C entry point**, `ts_query_new_with_schema(...)`, so a caller can supply the
data from wherever it has it. Additive functions are not an ABI break. This is what makes a
Rust `Query` constructor work end to end rather than only for Rust — the binding passes through
to a C entry point every other binding can also use. Complements A rather than competing:
A is the default path, B is the escape hatch for callers with their own artifact.

**C. A sidecar artifact carrying schema and precompiled queries together.** The natural home
once compiled queries exist, since both are read-only program data wanting one format, one
version stamp and one loading path. But it introduces the skew problem A avoids, so it needs a
grammar content hash to validate against — which is machinery worth building once, for both,
rather than twice.

## Recommendation

1. **A as the default**, with the schema split into independently-optional components. The
   sizing makes the "embed it" objection largely evaporate.
2. **B alongside**, because it costs one function, unblocks the Rust ergonomics you wanted, and
   keeps every other binding on the same path rather than stranding them.
3. **C deferred** to whenever compiled queries land, at which point the schema should move into
   the same artifact and share its versioning — designed then, with the compiled-query format,
   not now in isolation.

## What is still undecided, and should be

- **Encoding.** Dense bitmatrix was used for sizing; sparse would help the tail. Worth deciding
  with the compiled-query format rather than ahead of it.
- **Whether `generate` emits it by default** or behind a flag, and what happens to the ~2 % of
  grammars where the precedence relation is expensive.
- **Whether the schema is validated at load.** With option A it cannot mismatch; with B or C it
  can, and needs a grammar hash.
- **Whether guarantees and rejection ship together.** They are independent: guarantees are an
  optimisation, rejection is a diagnostic. Shipping guarantees first would be lower-risk, since
  a wrong guarantee degrades performance while a wrong rejection breaks a working query.
