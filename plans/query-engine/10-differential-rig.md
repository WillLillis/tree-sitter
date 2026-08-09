# The differential rig: what it actually is

Earlier drafts of [`09-roadmap.md`](09-roadmap.md) called this "the single most important
piece of infrastructure" and put it in Phase 0 as one thing. Having worked out what it would
concretely do at each phase, that was over-specified. It is not one thing, most of it is not
needed in Phase 0, and the parts that are needed are small.

## The problem

The existing suite (`crates/cli/src/tests/query_test.rs`, 125 tests) asserts specific expected
output for cases someone thought to write. That is the right safety net for targeted changes.
It is not sufficient for a change whose claim is *"nothing observable changed at all"* —
swapping the front end, or replacing the VM — because the interesting failures are in shapes
nobody wrote a test for.

For those, you want a stronger statement than "the tests still pass": you want *"the engine
produces byte-identical output on everything we can point it at."*

## The key point: the oracle is different at every phase

This is what makes it not-one-thing.

| Phase | What changes | What must be identical | Right mechanism |
|---|---|---|---|
| 1 — P1 analysis cache | internals only | everything | **in-process assertion**: cached result == freshly computed, in debug builds |
| 1 — P2 pool free list | internals only | everything | existing tests + ASan/valgrind (failure mode is use-after-free / wrong list reused) |
| 2 — front end | parse → emit path | the emitted `QueryStep` array, **byte for byte** | **Dump A** + goldens |
| 3 — new VM | execution | the match stream | **Dump B** + goldens |
| 4 — disambiguation | semantics, **deliberately** | nothing — diffs *are* the deliverable | same dumps, now a review artifact rather than a gate |

Phase 1 does not need a rig. The right check for a cache is an assertion that the cache is
correct, which is exact, in-process, and localized — strictly better than noticing downstream
that a match went missing. Building corpus-diffing infrastructure to validate a memoization is
using the wrong tool.

## So what gets built: two dumpers and a directory of goldens

Not a framework. A canonical text serializer for each of the two things worth pinning, plus
checked-in expected output.

**Dump A — the compiled form.** One line per step, deterministic ordering:

```
# javascript/highlights.scm
pattern 0  steps 0..3  rooted  bytes 0..46
   0  sym=call_expression   depth=0  field=-         caps=[]
   1  sym=identifier        depth=1  field=function  caps=[@function]
   2  DONE
pattern 1  steps 3..9  non-rooted  bytes 47..131
   ...
```

Any change to parsing, resolution, or codegen shows up as a diff here. This is the Phase 2
oracle, and it is a *stronger* check than match-stream equality — identical step arrays imply
identical behaviour, and it localizes a failure to the exact pattern and step.

**Dump B — the match stream.** What the API actually returns, in the order it returns it, for
both modes:

```
# javascript/highlights.scm over playground.js  [next_match]
match p=3  caps=[@keyword 12..15, @function 16..20]
match p=7  caps=[@string 41..55]
# ... [next_capture]
capture p=3 idx=0  @keyword 12..15
```

This is the Phase 3/4 oracle, and the only one that covers execution.

## Why goldens rather than running two engines side by side

Linking two copies of `query.c` with mangled symbols so they can be compared in one process is
heavy, fragile, and only works while both exist. Checked-in goldens instead:

- **compare across time**, not just across a build — today's output vs. what `master` produced
  last month;
- **are the review artifact**. With effectively one maintainer, a PR whose diff shows
  `0 files changed` under `test/fixtures/query-goldens/` is a much stronger statement than
  "tests pass", and one that can be eyeballed;
- need no build gymnastics;
- become the Phase 4 deliverable for free — "here is exactly what my semantics change did,
  line by line" is precisely what you want to adjudicate against the conformance suite, and
  precisely what a blog post is written from.

The cost is that goldens are only as good as the moment they were generated. Which argues for
generating them early — see below.

## Revision to the Phase 0 plan

Generate **Dump B goldens from current `master` now**, before P1 and P2 land, even though
Phase 1 does not need the rig. Rationale: P1 and P2 are *claimed* to be semantics-preserving.
Goldens taken beforehand let that claim be checked rather than asserted, and they cost a
dumper plus a `#[test]`. Wait until Phase 2 and you are generating goldens from a tree that
already contains the changes you wanted to verify.

Dump A can wait for Phase 2 — there is no front-end change before then, so there is nothing
for it to protect.

So Phase 0 becomes:

1. Profiler — **done**, `tools/query-profiler/`
2. Pathological queries as timed regression tests (`queries/*.scm` in the profiler dir)
3. **Dump B + goldens from current master** — ~100 lines of dumper, one test, one fixture dir
4. Conformance tests for the anchor/quantifier semantics currently living only in commit
   messages and `query.c:4453-4501`

Not: a general two-engine comparison framework.

## Where it lives

Per the constraint in [`09-roadmap.md`](09-roadmap.md) — no second test mechanism competing
for authority over "what is correct":

- the dumpers go next to the existing tests, and the golden check is **a `#[test]` in the same
  suite**, walking the 39 real query files already in `test/fixtures/grammars/*/queries/`;
- goldens live in `test/fixtures/` alongside everything else;
- the 125 existing tests are untouched and remain the authoritative statement of intended
  behaviour. The goldens pin *observed* behaviour, which is a different and weaker claim —
  they record what the engine does, including its bugs. That distinction matters at Phase 4,
  when some golden diffs will be *fixes*.

## Scope note

A golden that changes is not automatically a regression, and a golden that does not change is
not automatically proof of correctness — the corpus only covers what is in it. The rig
narrows the search for "what did I break", it does not answer "is this right". The 125
hand-written tests answer that.
