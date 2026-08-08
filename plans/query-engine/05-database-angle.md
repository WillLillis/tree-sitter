# The database angle: what actually transfers

The instinct in the brief — "tree-sitter queries look like database queries" — is right, but
the most useful mapping is not to *relational* databases. It is to three adjacent bodies of
work, in descending order of directness:

1. **XML/XPath tree pattern matching** (1999–2008). Nearly a perfect match: same problem,
   same data shape, twenty years of solutions. Most under-exploited.
2. **Regex submatch extraction / tagged automata** (2000–present). The correct framing for
   the captures-and-disambiguation problem, which is the actual source of the blowups.
3. **Relational query optimization** (1979–present). Transfers as *architecture and
   discipline* — cost models, plan/execute separation, EXPLAIN — more than as specific
   algorithms.

Plus one that transfers as a *goal* rather than a technique: **incremental view maintenance**,
which is what editors actually need and what nothing in tree-sitter currently provides.

---

## 1. A tree-sitter query is an XPath twig query

Set the vocabulary straight:

| tree-sitter | XPath / XML literature |
|---|---|
| pattern | twig pattern / tree pattern |
| `(a (b))` | `a/b` — child axis |
| `(a (b) (c))` | twig with two branches |
| `field: (x)` | attribute/axis predicate |
| `.` anchor | positional predicate (`following-sibling::*[1]`) |
| `@capture` | the output/projection list |
| `#eq?` predicate | value predicate |
| (missing) | `//` — descendant axis |

Two facts make the XML literature immediately applicable:

**Fact 1: tree-sitter already has region encoding.** The single most important enabling
technique in XML query processing is *interval labelling* — annotate each node with
`(start, end, depth)` so that ancestor/descendant is an O(1) containment test rather than a
pointer walk. Every `TSNode` already carries `start_byte`, `end_byte`, and the cursor tracks
depth. `ts_query_cursor__compare_nodes` (`query.c:3667-3679`) is already doing interval
comparison, it just isn't being used as an index.

**Fact 2: the current engine has no index over the tree at all.** It walks every node and
tests it against live states. XML systems learned early that this is the wrong shape: you want
an **inverted index from tag name → node list in document order**, and then pattern matching
becomes a *structural join* between posting lists.

### Structural joins and holistic twig joins

For `(call_expression function: (identifier) @f)`:

- *Today*: walk 42,904 nodes, test each against the pattern.
- *Structural join*: take the posting list for `call_expression` (say 900 nodes) and the one
  for `identifier` (say 8,000), and merge-join them on the containment predicate. Both lists
  are in document order, so this is a linear merge with a stack — the Stack-Tree-Desc
  algorithm (Al-Khalifa et al., ICDE 2002). Cost is O(|A| + |B| + |output|), not O(|tree|).

- *Holistic twig join* (TwigStack, Bruno et al., SIGMOD 2002): for multi-branch patterns,
  binary structural joins produce large useless intermediate results. TwigStack matches the
  whole twig at once and is **worst-case optimal for path patterns** — it never produces an
  intermediate match that isn't part of a final answer. That property is precisely what the
  current engine violates: the 4.2-second case does 2.35 billion units of work to produce
  ~50 matches.

The catch, and it is a real one: **building the inverted index costs a tree walk.** For a
one-shot query over one file, index build ≈ the cost of the current approach, so you gain
nothing. It pays off when:

- one tree is queried by **many** patterns (exactly the highlights.scm case: 94 patterns);
- one tree is queried **repeatedly** across edits (exactly the editor case);
- the query is **selective** (a rare root symbol — you skip most of the tree entirely).

That third case is available today with no index at all, via the parse table: see
§3 below.

### The missing descendant axis

`//` is absent from the pattern language, and this is the feature most obviously enabled by
the above. With region encoding, "an `identifier` anywhere inside a `function_item`" is a
containment join, not a nested-wildcard hack. Users currently write non-rooted patterns or
deep wildcard nests to approximate it — and non-rooted patterns are exactly the shape that
triggers the state explosion. **Adding a descendant axis would likely reduce pathological
query load, not increase it**, because it gives users the construct they are currently
open-coding badly.

---

## 2. The captures problem is the tagged-automata problem

This is the most directly actionable idea in this document.

The engine is a Pike VM (see [`02-execution-model.md`](02-execution-model.md)) that does not
merge threads. A Pike VM merges threads at the same PC and is therefore O(program × input);
this one keeps one thread per *capture history* and reconciles them afterwards by pairwise
set comparison, which is where Θ(M²) live states and Θ(M⁴) comparisons come from.

The reason the engine does not merge is real: **merging threads loses capture information**,
and captures are the whole point. But this is a solved problem, and it has a name.

**Tagged NFAs / TDFAs** (Laurikari 2000; Trofimovich 2017) extend an automaton with *tags* —
positions in the input recorded on transitions. Submatch extraction becomes: run a
deterministic automaton, and on each transition, execute a small fixed set of register
operations that update tag values. Disambiguation (which of several possible tag assignments
wins) is resolved **at determinization time**, according to a stated policy — leftmost-first,
leftmost-longest/POSIX, or greedy-per-operator. At runtime, threads at the same state merge
in O(1) because the policy already decided whose registers survive.

Mapping to tree-sitter:

| regex | tree-sitter query |
|---|---|
| input string | pre-order node stream with depth |
| character class | symbol / supertype / field / `!field` test |
| `(...)` group with submatch | `@capture` |
| `*`, `+`, `?` | `*`, `+`, `?` (same) |
| alternation | `[...]` |
| anchors `^`/`$` | `.` anchor (first/last child) |
| POSIX longest-match | the current "longest match" dedup pass |

The current dedup pass **is** an attempt at POSIX longest-match disambiguation, implemented
as a runtime O(n²) filter instead of a compile-time determinization rule. Okui & Suzuki
(CIAA 2013) and Borsotti & Trofimovich (2021) give correct, efficient POSIX disambiguation
algorithms for exactly this.

**What this buys, concretely:**

- The Θ(M²) live-state blowup disappears — threads at the same automaton state merge.
- The Θ(M⁴) dedup pass disappears entirely; there is nothing to dedup.
- The semantics become *specified* instead of emergent, which is the point of
  [`03-correctness.md`](03-correctness.md) §C1.

**What is genuinely harder than the regex case:**

- The "input" is a tree, not a string. Threads are scoped by depth, and a tree walk visits
  siblings and then backtracks up. Tag registers must be scoped per (thread, subtree), which
  is more like a *visibly pushdown* automaton than a finite one.
- Full determinization can blow up in state count. The practical answer used by regex engines
  is **lazy DFA construction with a bounded cache** (RE2, and Go's regexp): build DFA states
  on demand, evict under memory pressure, fall back to NFA simulation. Same technique applies.
- Tree-sitter's node tests are over a large alphabet (symbol ids up to ~360 for these
  grammars), so transition tables want to be sparse/compressed — again exactly what RE2 and
  lexer generators do.

Recommended reading order for this thread: Cox's "Regular Expression Matching Can Be Simple
And Fast" series first (it *is* the current architecture, described clearly), then Laurikari,
then Trofimovich's TDFA-with-lookahead paper. See [`07-references.md`](07-references.md).

---

## 3. Relational query optimization: take the architecture, not the algorithms

### What transfers directly

**Selectivity-driven start-point selection.** This is the biggest available win that requires
no index at all.

Today, matching always begins at the pattern's *root* (`pattern_map` is keyed by root symbol)
and proceeds top-down. But consider:

```scheme
(macro_invocation macro: (identifier) @m)
```

Measured node counts on `crates/cli/src/tests/query_test.rs` (42,904 nodes):

| symbol | occurrences |
|---|---|
| `identifier` | 4,704 |
| `string_literal` | 1,791 |
| `call_expression` | 1,390 |
| `macro_invocation` | 379 |
| `function_item` | 126 |

So for the pattern above, starting at the root (`macro_invocation`, 379) rather than the child
(`identifier`, 4,704) is 12× less work — and the engine gets that right by accident, because
`pattern_map` is keyed by root symbol.

Now invert it: `(call_expression function: (identifier) @f)` starts at 1,390 nodes to find
identifiers, which is fine; but a pattern rooted at a common symbol with a rare required
descendant has no way to say "start at the rare one and verify upward via `ts_node_parent`".
`function_item` is 37× rarer than `identifier`; any pattern that constrains both should start
from `function_item` regardless of which one the user happened to write as the root. The
engine cannot express this, because the plan is implicit in the step order, which is the
source order.

Where selectivity estimates come from, in increasing order of effort:

1. **Statically, from the grammar.** The parse table already tells you which symbols can
   appear as children of which — `ts_query__analyze_patterns` walks exactly this structure
   already. A crude "how many productions can produce this symbol" is a usable prior and is
   free given the analysis is already running.
2. **From `node-types.json`.** Already generated, already shipped.
3. **From a corpus.** Sample N files per language at grammar-build time, record per-symbol
   frequency, ship a histogram. A few hundred bytes per grammar.
4. **Adaptively, at runtime.** Count what you actually see and re-plan. This is where Leis et
   al. ("How Good Are Query Optimizers, Really?", VLDB 2015) is the important corrective:
   cardinality estimation errors compound catastrophically through join trees, and the
   practical lesson is *prefer robust plans and adaptive execution over precise estimates*.
   For a single-tree, few-node-pattern setting the risk is much lower than in a relational
   engine, but the lesson stands: do not build an elaborate cost model.

**Plan/execute separation and EXPLAIN.** A database's most user-visible feature after speed is
the ability to ask *why*. `tree-sitter query --explain` showing:

```
pattern 3: (call_expression function: (identifier) @f)
  plan: scan pattern_map[call_expression]  (est. 900 nodes, actual 874)
        └ verify child[function] is identifier   (est. 0.9 sel, actual 0.94)
  guaranteed steps: 1/2      capture list peak: 2      matches: 821
```

...is achievable the moment an IR exists, and it addresses the "introspection and analysis
tools for users" thread directly. Today it is unimplementable: there is nothing to print.

**Multi-query optimization.** 94 Rust highlight patterns are compiled and executed as 94
independent programs, sharing only the `pattern_map` bucket. They have enormous common
structure. Two well-studied answers:

- **Rete** (Forgy 1982; Doorenbos 1995) — the canonical "many patterns, many facts"
  algorithm, from production rule systems. Builds a dataflow network with shared prefix nodes
  (alpha memories = per-symbol filters, beta memories = partial joins). Notably, Rete is
  *inherently incremental*, which is the other thing we want.
- **Automaton union.** Merge all patterns into one automaton with per-pattern accept states.
  This is what lexer generators do and what a TDFA construction would give for free.

**Predicate pushdown.** Covered in [`03-correctness.md`](03-correctness.md) §C2. `#eq?` and
`#match?` are evaluated in the binding *after* a full match is produced. For
`tags.scm`/`locals.scm`, which are dense with `#match?`, this means matching everything and
throwing away most of it. Pushing predicates into the engine turns a scan into a lookup. The
blocker is architectural: the C query engine deliberately has no text access. Fixing it means
`TSQueryCursor` gaining an optional text provider — a real API change requiring coordination
across bindings, which is why it is P8 and not P1.

### What does not transfer

Be disciplined about this; a lot of DB machinery is irrelevant here:

- **Join ordering over many relations.** Patterns are small connected tree patterns (2–5
  nodes typically, max depth 2 in every `highlights.scm` measured). The exponential
  join-ordering problem does not arise. Selinger-style DP is overkill; a greedy
  most-selective-first heuristic is sufficient.
- **Worst-case optimal joins (Leapfrog Triejoin, Generic Join).** The theory is beautiful and
  the *goal* — bound work by output size, not intermediate size — is exactly right for this
  problem. But WCOJ's advantage appears on cyclic queries over large relations; tree patterns
  are acyclic and tiny. Take the framing ("our algorithm is not output-sensitive; that is the
  bug"), not the algorithm.
- **Cost-based physical operator selection, buffer pools, spilling, statistics maintenance.**
  All assume a persistent store. Here the "database" is rebuilt on every keystroke.
- **Transactions, concurrency control.** Not applicable.

The constraint that rules out most heavyweight DB machinery is worth stating plainly:
**any index must be either free or incremental**, because the tree changes constantly. This
is why the answer is not "build a proper index" but "build an incremental one" — which is the
next section.

---

## 4. What editors actually need: incremental matching

Tree-sitter's defining feature is incremental *parsing*. Its query engine is not incremental
at all. After an edit, every consumer re-runs queries over either the whole file or a
heuristically-chosen damaged range. This is the largest gap between what the library provides
and what its biggest users need.

The database framing is exact: **query results are a materialized view over the tree; an edit
is a delta; keeping the view current is incremental view maintenance.**

The pieces already exist:

- `ts_tree_get_changed_ranges` gives the delta.
- Incremental parsing means most subtrees are **physically shared** between old and new trees
  — pointer equality is a valid "unchanged" test, which is a much stronger primitive than
  most IVM settings get.
- Matches are naturally scoped: a match rooted at node *n* depends only on *n*'s subtree
  (for rooted patterns) or on *n*'s parent's child list (for non-rooted ones).

Two credible designs:

- **Rete-style.** Alpha memories keyed by symbol, beta memories holding partial matches.
  Edits retract and assert facts. Well-understood, inherently incremental, and handles the
  many-patterns case in the same mechanism. Cost: memory proportional to partial-match count,
  which is exactly the thing that blows up today — so this must come *after* the
  disambiguation fix, not before.
- **Memoized bottom-up automaton.** If matching is a bottom-up tree automaton, the automaton
  state at each node is a function of its children's states. Cache it on the subtree. Shared
  subtrees keep their cached state for free; only the spine from the edit to the root
  recomputes. This composes beautifully with tree-sitter's existing subtree sharing and is,
  I think, the more natural fit for this codebase. The relevant general theory is
  self-adjusting computation (Acar; Adapton) and DBSP (Budiu et al., VLDB 2023) for the
  clean formal treatment.

Either way this is a phase-5 item. Listing it here because it should shape the IR design now —
specifically, **an IR that makes per-node automaton state explicit is incrementalizable; one
that threads mutable capture lists through a global state list is not.**

---

## 5. Prior art worth studying directly

Systems solving adjacent problems, with lessons:

- **CodeQL / Semmle `.QL`** — Datalog over code, with a real optimizer and a cost model.
  Demonstrates how far the declarative-query-over-code idea scales, and how much machinery it
  takes.
- **Soufflé** — Datalog compiled to C++, with semi-naive evaluation and specialized index
  selection. The *index-selection-from-the-query* work is directly relevant to "which
  inverted indexes should I build for this query set".
- **Glean** (Meta) — incremental code indexing at scale; the storage/incrementality design is
  the interesting part.
- **ast-grep** — tree-sitter-based, built its own pattern engine and rule language rather than
  using `.scm`. Studying *why* is worthwhile — it is the clearest existing statement of what
  the query language lacks.
- **Semgrep** — different matching model (pattern-as-code with metavariables) over multiple
  languages. Their published notes on matching semantics and on "why naive AST matching is
  insufficient" are a good source of requirements.
- **RE2 / Go `regexp`** — the reference implementation of "NFA simulation with a lazy DFA
  cache and bounded memory", which is the architecture this engine should probably converge
  on.

---

## Summary: the five ideas worth stealing

1. **Region-encoded structural joins** (XML) — turns tree matching into merge joins over
   posting lists; enables a descendant axis. Pays off for many-pattern and repeated-query
   workloads.
2. **Tagged automata / TDFA** (regex) — the correct, specified way to do captures with
   disambiguation, and the direct fix for the state explosion. **Highest value.**
3. **Selectivity-driven start-point selection** (relational) — start matching at the rarest
   node in the pattern, not the root. Cheap to prototype, no index needed.
4. **Rete or automaton union** (rule systems) — share work across the ~90 patterns in a real
   highlights file instead of running them independently.
5. **Incremental view maintenance** (DB) — the thing editors actually want; shapes the IR
   design now even if implemented last.
