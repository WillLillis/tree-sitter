# Roadmap: sequencing, risk, and what to do first

Assumes several months of focused work. Phases are ordered so that **every phase boundary is
a shippable state** and the one irreversible change happens last, with maximum test coverage
behind it.

## The shape of the problem: one knot, everything else separable

"Big rewrite or section-by-section?" is the wrong axis. The accurate statement is:

> **The engine has exactly one knot. Everything else is independently replaceable behind a
> mechanical oracle.**

The knot is that **the O(n²) dedup pass *is* the disambiguation semantics**
([`02-execution-model.md`](02-execution-model.md)). Nothing else in the codebase defines which
of several overlapping matches wins. You cannot delete it incrementally, because there is
nothing to fall back to. That one replacement has to be coordinated, and it has to be preceded
by writing down what the answer should be.

Everything else — the parser, the analysis, the capture pool, the state indexing, the step
encoding — can be swapped one piece at a time, because each has a **mechanical oracle**:

| Component | Oracle for "did I break it?" |
|---|---|
| front end (parse → HIR → steps) | emitted `QueryStep` array is **bit-identical** on the corpus |
| analysis caching | cached result **equals** freshly computed result (assert in debug builds) |
| capture pool, state indexing | match stream **identical** on the corpus |
| the disambiguation change | *no oracle* — only a written spec and adjudicated diffs |

That asymmetry is the whole plan. A front-end replacement with a bit-exact oracle is a
**refactor**, not a rewrite, in every way that matters for risk — you can do it in pieces, and
CI tells you immediately if you were wrong. The back-end replacement has no oracle, so it gets
done once, late, with the strongest possible net under it.

Hence: **strangle the front end, one-shot the back end.**

### The IR is not the goal

Worth stating plainly, because it is easy to lose: the new IR, the staged compiler, and the
bytecode are **infrastructure for the disambiguation fix and for tooling**. They are not the
deliverable. The deliverable is that idiomatic queries stop being quadratic and that match
semantics become specified.

This matters because the front-end work is both the *safest* part (bit-exact oracle) and the
most enjoyable part (greenfield, clean design, no legacy semantics to preserve), while the
disambiguation change is the riskiest and least pleasant. The natural failure mode for a solo
effort is to spend a year building a beautiful compiler while `((attr)* (doc)* (fn))` still
takes 4.2 seconds. Phase 1.5 exists partly as a guard against exactly that.

The mitigating fact is that Phase 2's payoff is real and user-visible on its own — spans,
multiple diagnostics per compile, and the API that
[`ts_query_ls`](https://github.com/ribru17/ts_query_ls) currently has to work around
([`06-compiler-architecture.md`](06-compiler-architecture.md)). It is not pure scaffolding. But
it is not the point either.

## The constraint that decides this: one maintainer, no safety net but the tests

Practical reality as of this writing: **maintenance is effectively one person**, with the
original author reviewing occasionally. That is the single most important input to sequencing,
and it points the same direction as everything else — but for different reasons than a
multi-maintainer project would.

An earlier draft of this doc argued from *review capacity*: "a 5,000-line PR will not land
because nobody can absorb it." That argument is weak here — there is no gate to fail. Two
stronger ones replace it:

**1. Continuity risk dominates.** A six-month rewrite branch owned by the only person who
could finish it is the classic way these efforts die — not from technical failure, from life.
A branch that is 80% done is worth zero. Incremental landing means that if work stops at any
point, everything done so far is already on `master` and already delivering value. This is a
much stronger argument for incrementality than review capacity ever was.

**2. There is no reviewer to catch you, so the oracles have to.** Review capacity was doing
double duty in the old argument: bounding diff size *and* detecting errors. Remove the
reviewer and the error-detection need does not go away — it just has to be mechanized.
The differential rig therefore moves from "important" to **the only thing standing between a
mistake and the entire downstream ecosystem.** Self-review does not catch semantic regressions
in a 5,000-line diff; a bit-exact corpus comparison does, and it does not get tired or
over-familiar with the code.

Three consequences worth designing around:

- **Extend `crates/cli/src/tests/query_test.rs`; do not build a second test mechanism.** It is
  already 125 tests and ~6,500 lines of intentional, minimal, diagnostic cases — when one
  fails you know exactly what broke. That is the safety net, and the differential rig should
  be *a way of running it* (execute the existing suite against both engines and diff), not a
  parallel corpus runner with its own notion of authority.

  An earlier draft of this doc proposed vendoring downstream `.scm` files from
  nvim-treesitter/Helix/Zed. **Withdrawn.** It imports a dependency to solve a coverage
  problem: those files target grammar versions we do not ship, they churn for reasons
  unrelated to us, a failure gives low signal (our bug or their odd query?), and the project
  should stand alone. It was also partly redundant — `test/fixtures/grammars/*/queries/`
  already holds **39 real query files across 15 grammars**, in-tree and maintained here, which
  `crates/cli/benches/benchmark.rs` already walks.

  The legitimate worry underneath it — hand-written tests only cover shapes someone thought of
  — is better answered by **generative testing**: synthesize queries from a grammar's own node
  types and fields, run them against synthesized or corpus trees, and assert invariants
  (compiles ⇒ terminates; every returned capture is within its match; old engine ≡ new
  engine). That is adversarial like real-world files, needs no external dependency, and there
  is precedent in `test/fuzz`.
- **The occasional upstream review is a scarce resource — spend it on the irreversible
  things.** Batch the four decisions below into a small number of written proposals with
  measurements attached, rather than dribbling questions across months. Do not spend it on
  perf PRs; those justify themselves with numbers.
- **Order by risk, not by persuasion.** The old advice was "order by contentiousness" to make
  PRs easy to approve. With no approver, the same order still holds for a better reason: the
  uncontroversial work (P1, P2) is what builds and *validates* the safety net you will be
  betting the semantics change on later. Do it first because it proves the harness, not
  because it is easy to sell.
- **Assume a three-week gap between sessions.** Solo and part-time means future-you has
  forgotten the context. This is what the decision records in
  [`06-compiler-architecture.md`](06-compiler-architecture.md) and this doc set are actually
  for; treat writing them as part of the work, not overhead on it.

## The four decisions that are hard to reverse

Make these deliberately and early, before writing engine code. Everything else is
implementation.

1. **What is the disambiguation policy?** When several matches of a pattern overlap, which
   are returned? Today: whatever `compare_captures` does. This must become a written spec.
   Options: preserve current behaviour exactly (safest, but you inherit its oddities);
   POSIX leftmost-longest (well-studied, implementable in-automaton); greedy-per-operator
   (Perl-like, simplest to explain). See [`02-execution-model.md`](02-execution-model.md)
   §"The disambiguation question".

2. **Does the engine get text access?** Predicate pushdown is worth an order of magnitude on
   `tags.scm`/`locals.scm` shapes, and it requires `TSQueryCursor` to be able to read source
   text. That is a real API change with cross-binding coordination cost. Deciding "no" is
   fine, but decide it, because it changes the IR.

3. **Is bytecode a stable public artifact or an internal detail?** Public means a versioning
   policy, a verifier, and a conformance suite forever. Internal means you can move fast but
   get none of the precompilation or cross-implementation benefits.

4. **Does the query language grow?** A descendant axis (`//`) is the most-requested missing
   construct and is cheap given region encoding — and would likely *reduce* pathological load
   by giving users the thing they currently open-code as non-rooted patterns. But it is a
   language change and cannot be walked back.

## Phase 0 — instrumentation and safety net (2–3 weeks)

Nothing else starts until this exists.

- Land the profiler ([`08-measurement-harness.md`](08-measurement-harness.md)) with the
  counters, the compile-phase timers, and a committed corpus.
- Land the pathological query set as timed regression tests.
- **Build the differential rig.** Run two engines over the corpus and diff match streams.
  This is what makes every later phase safe; without it, "did I change semantics?" is
  unanswerable and the project stalls the first time a downstream user reports a subtle
  highlighting change.
- Write down the current semantics as a **conformance suite** — especially the anchor and
  quantifier behaviour encoded in commits `a6bc72474`, `139b801cc`, `dfcf73921`, `1ffd612be`
  and the comments at `query.c:4453-4501`. These tests are the requirements document for
  everything after.

Exit criterion: you can change `query.c` and know within one command whether match results
moved.

## Phase 1 — semantics-preserving wins (2–4 weeks)

All independently landable, none touching the matching algorithm. From
[`04-performance.md`](04-performance.md):

- **P1** cache language-only analysis on `TSLanguage` → −0.7 to −2.8 ms per `ts_query_new`
- **P2** free-list in `CaptureListPool` → O(1) acquire, kills the 191-step scan
- **P3** bucket live states by depth → removes 99% of the advance-loop scan on hot queries
- **P5** hash the symbol tables → removes an O(n²) from the parse path

Plus the Tier-A correctness fixes from [`03-correctness.md`](03-correctness.md):

- **A2** the `MISSING` prefix bug — one line, unambiguous, no interaction with anything else
- **A1 / A3** capture and negated-field overflow — **test now, fix after Phase 4.** See below.
- Write the B1–B4 tests; confirm or dismiss each suspicion

### The analysis blow-up outranks `insert_sorted`

`analysis_state_set__insert_sorted` is 10% of total instructions on a realistic run, and a
fix there is worth an estimated 15-20% of compile time on well-behaved queries — real, and
worth doing. But it is a constant factor sitting on top of a much larger effect: insert counts
grow from 1,562 to 617,034 across `(arguments (identifier) x 1..7)`
([`04-performance.md`](04-performance.md) §1b). Reduce the state churn and the constant factor
matters proportionally less; optimize the constant factor first and the cliff is untouched.

Sequence: characterize why `perform_analysis` generates ~96% duplicate states over broad
parent symbols, fix that, then revisit `insert_sorted` against the new profile.

### Why the limit bumps are deferred, not "one character"

An earlier draft proposed raising `MAX_STEP_CAPTURE_COUNT` 3 → 8 and
`MAX_NEGATED_FIELD_COUNT` 8 → 16 immediately, on the grounds that the struct growth is
negligible. That reasoning was wrong on two counts, and the limits are load-bearing today:

1. **They are multipliers on the quadratic pass.** Capture-list *length* is the inner
   dimension of `ts_query_cursor__compare_captures` — the dedup pass is
   O(group² × capture_list_len). Raising the per-node capture ceiling from 3 to 8 scales the
   inner loop of the exact thing that already costs 322 M steps on
   `queries/two-quant.scm`. Likewise the negated-field loop (`query.c:4305-4319`) calls
   `ts_node_child_by_field_id` once per negated field, per state, per node. These caps are
   currently acting as a blast radius limiter on explosive queries, whether or not that was
   their original intent.
2. **`QueryStep` cache density is not free.** 20 bytes gives 3.2 steps per 64-byte cache
   line; 30 bytes gives 2.1 — a ~35% drop in step density in a struct that is read once per
   live state per node. "The array is only 5 KiB" was the wrong measure.

So the sequencing is inverted from the earlier draft: **fix the explosion first, then raise
the limits once they are no longer the thing holding it back.** In the interim, land a test
that pins the current behaviour explicitly — so the truncation is a known, intentional,
documented limitation rather than an accident — and revisit in Phase 5, with the pathological
suite as the acceptance gate.

The structural fix (unbounded lists in the HIR) still arrives with Phase 2, but it should not
be *exposed* through the step encoding until the matching algorithm can absorb it.

Exit criterion: measurable compile-time reduction, no match-stream diffs, three real bugs
closed.

### Which of this survives the rewrite

The honest per-item answer, because "will we just rewrite this anyway?" has a different
answer for each:

| Item | Lives in | Replaced by | Durable? |
|---|---|---|---|
| **P2** capture-pool free list | `CaptureListPool` | nothing — capture storage is still needed | **fully durable** |
| **P1** language-only analysis cache | new side table on `TSLanguage` | nothing — the analysis stage survives as stage 3 | **fully durable** (the walk gets ported; the cache keying/invalidation carries verbatim) |
| **P3** depth-bucketed state list | VM state management | Phase 4 VM | code thrown away, *insight* carries — but it is what keeps the engine usable in the interim |
| **A1/A3** limit bumps | `QueryStep` + parser | deferred to Phase 5 — the caps currently limit blast radius on explosive queries | the *test* pinning current behaviour is durable; the bump waits |
| **A2** `MISSING` prefix | parser | Phase 2 front end | one line thrown away; **the test survives** |
| **P5** hash the symbol tables | parser | Phase 2 | **skip it** — on measured data (21 captures per query) this is noise. Listing it in `04` was over-eager |
| harness, corpus, conformance suite, every test | — | nothing | **the most durable artifact in the project** |

Roughly half of Phase 1 is permanent infrastructure; the disposable half is five lines per fix
and closes live bugs. Carrying a known silent-data-loss defect for six months to avoid writing
a line that later gets deleted is the wrong trade.

## Phase 1.5 — spec the semantics, then spike the matcher (3–4 weeks)

Inserted after review. Phases 2 and 3 build an IR and a bytecode whose most important
consumer is the *new matching algorithm* from Phase 4. Designing them without knowing what
that algorithm needs is how you get an IR that has to be reworked once its real consumer
arrives.

Two deliverables, neither of which is production code.

**1. The disambiguation policy, written down** — decision 1 above. This is a writing task, not
a coding one: derive it from current behaviour where current behaviour is sane, choose
deliberately where it is not, and land it as a conformance suite in `query_test.rs`. It blocks
Phase 4 and it costs nothing but thought, so there is no reason to defer it.

**2. A time-boxed, deliberately throwaway spike** of merge-based matching. The unknown worth
buying down is *not* "can a tagged automaton be implemented" — the regex literature answers
that ([`07-references.md`](07-references.md) §1). It is:

> Does thread merging survive **tree-shaped input**? The literature is about strings. Our input
> is a depth-scoped tree walk with sibling anchors, `!field` assertions, non-rooted patterns,
> and quantifiers whose zero-match case transfers anchor obligations across steps
> (`query.c:4453-4501`). No paper answers whether tag registers can be merged soundly under
> those rules.

That is the single largest technical risk in the whole plan, and it is answerable in a few
weeks by a prototype that is allowed to be ugly — over the *existing* `QueryStep` encoding,
possibly in Rust, possibly not even complete. The deliverable is a written answer to:

- Does merging preserve the anchor and quantifier semantics pinned by the conformance suite?
- What state identity makes two threads mergeable?
- What must the IR expose for this — and what is it therefore wrong to hide?
- Does the `((attr)* (doc)* (fn))` case actually become linear?

If the answer is "merging does not work here", that is worth knowing **before** spending three
months on an IR designed to serve it. If it works, Phase 2's design lands with its hardest
requirement already known.

## Phase 2 — the front end (4–8 weeks)

Build stages 1–2 from [`06-compiler-architecture.md`](06-compiler-architecture.md) — parse →
AST → HIR — as a **new path that emits the existing `QueryStep` array**.

- Differential-test at the *step array* level: byte-identical output on the whole corpus.
- Where output differs, adjudicate: old bug, or new bug? Record every adjudication.
- Retire the peephole patch at `query.c:3144-3189` — the HIR should make it unnecessary. If it
  doesn't, the HIR is wrong.
- A1 and A3 disappear structurally here (unbounded capture and negated-field lists).

This phase pays for itself immediately in diagnostics: spans, multiple errors, and the
foundation for a query LSP and formatter.

Exit criterion: new front end is the only front end; step arrays are bit-identical to
pre-phase output on the corpus.

## Phase 3 — bytecode and a new VM, same semantics (6–10 weeks)

- Define the instruction set. Lower HIR → bytecode.
- Write the verifier.
- New VM executes bytecode with **exactly today's disambiguation**, including the O(n²) dedup
  pass, deliberately unchanged.
- Differential-test at the *match stream* level, behind a runtime switch.
- Decide the serialization question (decision 3 above). If public: version, stamp with
  grammar identity + ABI + parse-table hash, resolve symbols by name at load.

Exit criterion: new VM is default, old engine removable, match streams identical.

Note the ordering rationale: bytecode *before* the semantics change means the risky change
happens in one well-tested component rather than being tangled with a rewrite.

## Phase 4 — the semantics change (6–12 weeks)

The disambiguation rewrite. This is where the cliff dies.

- Implement the policy chosen in decision 1, in the automaton — tagged transitions with
  disambiguation resolved at construction time
  ([`05-database-angle.md`](05-database-angle.md) §2).
- Delete the O(n²) dedup pass.
- Expect deliberate diffs in the match stream. Every one must be adjudicated against the
  conformance suite from Phase 0 and either accepted with a changelog entry or fixed.
- Acceptance metric: `dedup_capture_steps / matches` bounded by a small constant, and the
  pathological set within its time budget. The M=100 case must complete.

This is the phase most likely to slip. Budget accordingly, and keep the old VM behind a flag
until downstream users have shipped a release on the new one.

## Phase 5 — the optimizer, tooling, and language growth (ongoing)

Now that a plan stage exists, these become incremental work rather than rewrites:

- selectivity estimation + start-point selection (biggest win, no index required)
- shared plans across patterns (Rete or automaton union) for the 94-pattern highlights case
- `tree-sitter query --explain` / `--profile`
- query LSP, formatter, linter (the "known-quadratic quantifier shape" lint would have caught
  the 4.2 s query at authoring time)
- predicate pushdown, if decision 2 said yes
- descendant axis ([#880](https://github.com/tree-sitter/tree-sitter/issues/880)), if
  decision 4 said yes — deliberately *after* the matcher can merge threads, since implementing
  it on the current engine would ship the pathological shape as a language feature. See
  [`06-compiler-architecture.md`](06-compiler-architecture.md) for why the draft PR's
  control-flow approach is the wrong layer, and [`11-data-oriented-design.md`](11-data-oriented-design.md)
  for where DoD does and does not pay.

## Phase 6 — incrementality (speculative)

The thing editors actually want ([`05-database-angle.md`](05-database-angle.md) §4). Deferred
because it depends on bounded partial-match counts, which only Phase 4 delivers. Keep it in
mind while designing the IR: **per-node automaton state must be explicit and cacheable** for
this to be possible later.

## Risks

| Risk | Mitigation |
|---|---|
| **Downstream breakage.** nvim-treesitter, Helix, Zed, and hundreds of grammar repos depend on current behaviour, including its bugs. | The conformance suite is built from *current* behaviour, not from an ideal. Phase 4 diffs are adjudicated one at a time, with a changelog. Keep the old VM behind a flag through one release cycle. |
| **The rewrite never lands** — a years-long branch that diverges. | Every phase ships. Phases 1–3 are semantics-preserving, so they can land continuously on `master`. |
| **Semantics change is discovered late**, by users rather than tests. | The differential rig in Phase 0 is the single most important deliverable in this document. |
| **Scope creep into the language** (descendant axis, composition, typed captures). | Decisions 2 and 4 are explicit gates. Defaulting to "no" for the first three phases is the right call. |
| **Analysis caching introduces staleness bugs** (P1). | Key the cache on the language pointer + ABI + version, invalidate on `ts_language_delete`, and assert equality against a freshly computed result in debug builds. |
| **Estimating effort from this document.** Every phase estimate here is a guess by someone who has read the code but not maintained it. | Treat the *ordering* as the contribution and the durations as placeholders. |

## Writing it up

These docs are working notes on a branch, not upstream material. The public artifact is a
series of posts, **each written when the corresponding work lands, never ahead of it.** A post
that ships with a merged PR has a concrete result attached; a post that describes a plan
becomes wrong the moment the plan changes, and creates an obligation to a design you may want
to abandon.

That ordering also improves the posts. "Your tree-sitter query is quadratic" is a complaint
when written today; written alongside the Phase 4 fix, it is a war story with a resolution.

Sketch, one per landing:

| Lands with | Post |
|---|---|
| Phase 0 | How the query engine was profiled, and what fell out — the harness *is* the deliverable, the findings are its justification |
| Phase 1 | Where `ts_query_new` spends 17 ms (and why a 16-byte query costs 2.9 ms) |
| Phase 2 | Giving the query language a real front end |
| Phase 3 | A bytecode for tree-sitter queries |
| Phase 4 | Your tree-sitter query is quadratic — and what a specified disambiguation policy fixes |

Each should be assembled from artifacts that already exist by then — PR descriptions, harness
output, the decision records — rather than written as a separate research effort. If a post
requires new investigation, it is a sign it is being written too early.

## Smallest useful first step

If the appetite is one week rather than one quarter: build the differential rig and the
profiler (Phase 0, first two bullets), then land **P2** (capture-pool free list) as a
proof that the loop works end to end. It is ~20 lines, it is measurable, and it exercises
every piece of the safety net you will depend on for the next several months.
