# Annotated bibliography

Cited from memory; venues and years are believed correct but worth confirming before quoting
any of them in a public design doc. Grouped by which problem they solve for us, and ordered
within each group so you can read top-down and stop when it stops paying.

Tiers: **[core]** = read this, it changes the design. **[useful]** = read if you go down that
path. **[context]** = know it exists.

---

## 1. The captures + disambiguation problem — start here

This is the highest-value thread. Read in this order.

**[core] Russ Cox, "Regular Expression Matching Can Be Simple And Fast" (swtch.com, 2007),
plus the follow-ups "Regular Expression Matching: the Virtual Machine Approach" (2009) and
"Regular Expression Matching in the Wild" (2010).**
Free, short, and describes the exact architecture `query.c` has independently reinvented —
Thompson NFA simulation, the Pike VM, submatch tracking, and lazy DFA construction with a
bounded cache. The second article's Pike VM section is essentially a specification of what
`ts_query_cursor__advance` is trying to be. Read this before anything else; it will reframe
the entire engine for you.

**[core] Ville Laurikari, "NFAs with Tagged Transitions, their Conversion to Deterministic
Automata and Application to Regular Expressions" (SPIRE 2000).**
The original tagged-NFA/TDFA paper. This is the mechanism that lets you merge threads at the
same automaton state *without* losing capture information — i.e. the direct fix for the Θ(M²)
live-state blowup in [`02-execution-model.md`](02-execution-model.md).

**[core] Ulya Trofimovich, "Tagged Deterministic Finite Automata with Lookahead" (2017), and
the related re2c submatch-extraction papers.**
The modern, practical, implementable treatment of the above, by the re2c author. More
detailed and more directly codeable than Laurikari. If you implement one paper from this
document, it is probably this one.

**[core] Satoshi Okui and Taro Suzuki, "Disambiguation in Regular Expression Matching via
Position Automata with Augmented Transitions" (CIAA 2013).**
Correct POSIX leftmost-longest disambiguation. Directly relevant because the current dedup
pass *is* an attempt at longest-match disambiguation, done at runtime in O(n²) instead of at
determinization time in O(1).

**[useful] Angelo Borsotti and Ulya Trofimovich, "Efficient POSIX submatch extraction on NFA"
(2021).**
Refines the above with better constants; useful once you have committed to a policy.

**[context] Ken Thompson, "Programming Techniques: Regular expression search algorithm"
(CACM, 1968).** The origin. Two pages.

---

## 2. Tree pattern matching (XML/XPath) — the closest structural analogue

**[core] Nicolas Bruno, Nick Koudas, Divesh Srivastava, "Holistic Twig Joins: Optimal XML
Pattern Matching" (SIGMOD 2002).**
TwigStack. The key property is **no useless intermediate results** — every partial match
produced participates in a final answer. That is precisely the property the current engine
violates (2.35 billion units of work for ~50 matches). Even if you never implement TwigStack,
this paper gives you the right correctness criterion for a matching algorithm.

**[core] Shurug Al-Khalifa et al., "Structural Joins: A Primitive for Efficient XML Query
Pattern Matching" (ICDE 2002).**
Stack-Tree-Desc/Anc. Establishes structural join as *the* primitive, using interval labelling
— which tree-sitter already has for free in every node's byte range.

**[useful] Torsten Grust, "Accelerating XPath Location Steps" (SIGMOD 2002).**
The pre/post plane encoding. A clean, geometric way to think about ancestor/descendant/
preceding/following as region queries. Good intuition-builder for adding a descendant axis.

**[useful] Georg Gottlob, Christoph Koch, Reinhard Pichler, "Efficient Algorithms for
Processing XPath Queries" (VLDB 2002 / TODS 2005).**
Complexity results for XPath fragments, and polynomial algorithms for the full language.
Useful for knowing which language extensions are cheap and which are not — worth consulting
*before* adding a descendant or ancestor axis.

**[useful] Christoph Koch, "Efficient processing of expressive node-selecting queries on XML
data in secondary storage: A tree automata-based approach" (VLDB 2003).**
Bridges the XPath and tree-automata views, which is exactly the bridge this project needs.

**[context] Hubert Comon et al., "Tree Automata Techniques and Applications" (TATA).**
Free online book, continuously updated. The standard reference for tree automata. Reference
material rather than a read-through; consult chapters 1 and 6 when you need the formal
grounding.

**[core, if you add types or composition] Haruo Hosoya and Benjamin Pierce, "Regular
Expression Pattern Matching for XML" (POPL 2001; JFP 2003).**
This is startlingly close to our problem: regular expression patterns over trees **with
variable binding**, i.e. captures, with a well-specified semantics for which binding wins.
The XDuce work. If you want a principled answer to "what does this pattern *mean*", this is
where the answer is written down properly.

---

## 3. Many patterns at once, and incrementality

**[core] Charles Forgy, "Rete: A Fast Algorithm for the Many Pattern/Many Object Pattern Match
Problem" (Artificial Intelligence, 1982).**
The canonical answer to "94 patterns against one fact base". Shares prefix work across
patterns and is inherently incremental. Note the memory cost — partial matches are
materialized, which is the thing that currently blows up, so Rete comes *after* the
disambiguation fix.

**[useful] Robert Doorenbos, "Production Matching for Large Learning Systems" (CMU PhD thesis,
1995).**
Rete/UL. The practical engineering treatment — how Rete actually behaves at scale and what to
do about its memory profile. More useful than the original paper if you intend to build it.

**[core] Mihai Budiu et al., "DBSP: Automatic Incremental View Maintenance for Rich Query
Languages" (VLDB 2023).**
The cleanest modern theory of incremental computation over changing collections. Gives you a
principled way to derive an incremental version of a query from its non-incremental
definition. Directly applicable to "keep highlight results current across edits".

**[useful] Frank McSherry, Derek Murray et al., "Differential Dataflow" (CIDR 2013).**
The predecessor and the practical system. Good for intuition about how deltas propagate.

**[useful] Ashish Gupta, Inderpal Mumick, V.S. Subrahmanian, "Maintaining Views Incrementally"
(SIGMOD 1993).**
The classical IVM statement, including counting and DRed. Short and foundational.

**[useful] Matthew Hammer et al., "Adapton: Composable, Demand-Driven Incremental Computation"
(PLDI 2014); and Umut Acar's "Self-Adjusting Computation" (CMU thesis, 2005).**
The other tradition — incrementality via memoization and change propagation over a dependency
graph. Arguably a *better* fit for tree-sitter than the dataflow tradition, because
incremental parsing already gives you physically shared subtrees, which is exactly the
memoization key you want.

---

## 4. Query optimization — architecture and discipline

**[core] Viktor Leis et al., "How Good Are Query Optimizers, Really?" (VLDB 2015).**
Read this *before* building any cost model. The finding — that cardinality estimation errors
dominate and compound, and that plan robustness matters more than estimate precision — is the
main guard against over-engineering the optimizer described in
[`05-database-angle.md`](05-database-angle.md) §3.

**[core] Goetz Graefe, "The Cascades Framework for Query Optimization" (IEEE Data Eng. Bull.,
1995); and Graefe & McKenna, "The Volcano Optimizer Generator" (ICDE 1993).**
How to structure an extensible, rule-based optimizer with a memo. The right architectural
model for stage 4 in [`06-compiler-architecture.md`](06-compiler-architecture.md), even at
our much smaller scale — mostly for the discipline of separating logical from physical plans
and expressing transformations as rules.

**[context] Patricia Selinger et al., "Access Path Selection in a Relational Database
Management System" (SIGMOD 1979).**
System R. The origin of cost-based optimization. Read for the ideas (selectivity, access
paths, interesting orders), not for the algorithms — our join problem is far too small for
the DP.

**[context] Hung Ngo, Christopher Ré, Atri Rudra, "Skew Strikes Back: New Developments in the
Theory of Join Algorithms" (SIGMOD Record, 2013).**
The readable survey of worst-case optimal joins. Take the *framing* — bound work by output
size — as the design goal. The algorithms themselves (Leapfrog Triejoin, Generic Join) are
aimed at cyclic queries over large relations and do not fit tree patterns. Primary sources if
you want them: Ngo/Porat/Ré/Rudra (PODS 2012) and Veldhuizen (ICDT 2014).

---

## 5. Compiler architecture, IR, and bytecode design

**[core] Chris Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific
Computation" (CGO 2021).**
The direct model for "better separated stages that can be consumed as a library". The
multi-level IR idea — progressively lowering through dialects, each with its own verifier — is
exactly the HIR→plan→bytecode structure proposed here.

**[core] Andreas Haas et al., "Bringing the Web up to Speed with WebAssembly" (PLDI 2017),
plus the WebAssembly Core Specification.**
The reference example of a *stable, formally specified, verifiable* bytecode with an explicit
compatibility policy. If you serialize bytecode, steal Wasm's discipline: a validation pass is
part of the format, not an optional extra. Directly relevant to
[`06-compiler-architecture.md`](06-compiler-architecture.md) §"Stable serialized bytecode".

**[useful] Robert Nystrom, "Crafting Interpreters" (2021), part II.**
Free online. The most practical writing available on designing a small bytecode VM: opcode
design, operand encoding, dispatch. Right level of abstraction for the instruction set we
need.

**[useful] Chris Lattner and Vikram Adve, "LLVM: A Compilation Framework for Lifelong Program
Analysis & Transformation" (CGO 2004).**
For the "IR as a product other people build on" argument, which is the real justification for
stable bytecode.

**[context] Cooper & Torczon, "Engineering a Compiler"; Appel, "Modern Compiler
Implementation".**
Standard texts. Relevant chapters: IR design, and — for the pattern-matching connection —
instruction selection via tree pattern matching (BURS/iburg), which is literally the same
algorithmic problem in a different domain.

---

## 6. Prior art: query languages over code

**[useful] Oege de Moor et al., work on `.QL` / Semmle / CodeQL** (various; "Keynote: .QL —
Object-Oriented Queries Made Easy" and the QL optimizer papers).
The most mature declarative-query-over-code system with a real optimizer. Shows both how far
the idea scales and how much machinery it costs.

**[useful] Herbert Jordan, Bernhard Scholz, Pavle Subotić, "Soufflé: On Synthesis of Program
Analyzers" (CAV 2016).**
Datalog compiled to C++. The **automatic index selection** work is the directly transferable
piece: given a set of queries, decide which indexes to build. That is our
"which inverted indexes does this highlights.scm need" problem exactly.

**[context] Glean (Meta), Stack Graphs (GitHub), ast-grep, Semgrep, Comby.**
Systems, not papers. `ast-grep` is the most instructive — it is tree-sitter-based and chose to
build its own pattern language rather than use `.scm`, so its documentation is an implicit
list of what our query language lacks. Semgrep's notes on matching semantics are a good
source of requirements.

---

## Suggested first week of reading

If the goal is to start designing rather than to survey:

1. Cox's three articles (an afternoon) — reframes the engine.
2. Trofimovich's TDFA paper — the concrete mechanism for the main fix.
3. Bruno et al., TwigStack — the correctness criterion for matching.
4. Leis et al., "How Good Are Query Optimizers, Really?" — the guard against over-building.
5. Skim MLIR and the Wasm spec's validation chapter — the model for stages and for a stable
   artifact.

That is roughly two days of reading and it covers the four decisions in
[`09-roadmap.md`](09-roadmap.md) that are hard to reverse.
