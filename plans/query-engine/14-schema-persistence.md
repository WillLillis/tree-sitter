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

### All-or-nothing, and why that is the right call

Per-component optionality was considered and **rejected**: a grammar either ships a schema or it
does not. The consumer then has one branch — schema present, use the fast path; absent, fall
back to today's analysis — rather than a matrix of partial states, each needing its own parity
testing and its own version negotiation. The simplification is worth more than the bytes.

That puts the whole weight on making the representation small, since hoon must then ship
1.2 MiB or nothing.

## Row pooling: 11.5× on the worst case, at zero runtime cost

The precedence relation is **65% dense**, so a sparse pair list is *worse* than a bitmatrix, not
better. But the rows are highly repetitive — symbols that behave identically with respect to
ordering share a row — which is the same observation the parse-table action pooling exploited
(`plans/ACTION_POOL_DESIGN.md`: 98–99.4% duplication).

Measured, pooling distinct rows globally and storing one row id per symbol:

| grammar | widest | rows | distinct | dedup | dense | **pooled** |
|---|---|---|---|---|---|---|
| hoon | 280 | 45,990 | 870 | **52.9×** | 1,234.7 KiB | **107.6 KiB** |
| abl | 221 | 7,815 | 1,311 | 6.0× | 69.4 KiB | 21.3 KiB |
| julia | 627 | 3,084 | 186 | 16.6× | 64.2 KiB | **6.9 KiB** |
| swift | 121 | 6,119 | 497 | 12.3× | 62.3 KiB | 14.5 KiB |
| rust | 94 | 3,684 | 486 | 7.6× | 21.3 KiB | 8.9 KiB |

**hoon drops from 1.23 MiB to 108 KiB**, which puts the worst grammar in the corpus roughly
where `abl` and `julia` sat *before* pooling. Extrapolating the observed ~10× average, corpus
precedence falls from 2.13 MiB to somewhere near 0.2 MiB, and the whole schema for 295 grammars
lands under a megabyte.

**The property that matters is that this is a representation, not a compression.** Looking up
whether *x* can precede *y* is: read the row id for *x*, index the pool, test bit *y*. O(1), no
unpacking, no decompression step to claw back the runtime gain. Generic compression would trade
size for load-time work; pooling does not.

A per-symbol **rank** encoding was also tested — storing an ordinal and reducing precedence to
`rank(x) < rank(y)`, which would be O(|children|) instead of O(|children|²). It reproduces the
relation exactly for only **3–11%** of node types, so it is dead as a general scheme, though it
may be worth keeping as a special case for the node types where it does hold.

### On ordering rejection

The earlier disagreement resolves differently now. A previous draft proposed dropping ordering
rejection as a blanket policy; that was rightly pushed back on. With pooling the question mostly
disappears — precedence is affordable everywhere — so ordering rejection stays, and the schema
stays all-or-nothing.

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

1. **A as the default**, shipping the schema whole or not at all. The sizing, with pooling,
   makes the "embed it" objection largely evaporate.
2. **B alongside**, because it costs one function, unblocks the Rust ergonomics you wanted, and
   keeps every other binding on the same path rather than stranding them.
3. **C deferred** to whenever compiled queries land, at which point the schema should move into
   the same artifact and share its versioning — designed then, with the compiled-query format,
   not now in isolation.

## What is still undecided, and should be

- **How much further the encoding goes.** Row pooling is measured; further wins are plausible
  (pooling the child-sets too, narrower row ids, exploiting the ~3–11% of node types where a
  rank encoding is exact) but unmeasured. The pooled figures above are arithmetic on distinct
  row counts, not a built artifact.
- **Whether `generate` emits it by default** or behind a flag, and what happens to the ~2 % of
  grammars where the precedence relation is expensive.
- **Whether the schema is validated at load.** With option A it cannot mismatch; with B or C it
  can, and needs a grammar hash.
- **Whether guarantees and rejection ship together.** They are independent: guarantees are an
  optimisation, rejection is a diagnostic. Shipping guarantees first would be lower-risk, since
  a wrong guarantee degrades performance while a wrong rejection breaks a working query.
