# Native DSL / `dod-ify` Integration Audit

## Purpose

This document records the exploratory port of the native DSL frontend onto the
pool-based generate pipeline from `dod-ify`. The work was performed on the
scratch branch `native_dsl-dod-ify-scratch`; the `dod-ify` branch itself was not
modified.

The prototype was intended to answer these questions:

- Can `native_dsl` be rebased onto the new generate backend without preserving
  its heap-based intermediate rule representation?
- Can lowering emit production `Rule` nodes directly?
- Can every imported helper, inherited grammar, and root grammar share one
  `RulePool` and one string interner?
- Can the direct representation meet or improve on the old lowering path's
  allocation behavior?
- What complications should be addressed in a careful, squashed implementation?

The short answer is yes. The final scratch implementation uses one loader-wide
pool, emits production rules directly, contains no `ARule`, contains no native
DSL string pool, and passes all 602 generate-library tests. Two performance
questions remain: owned object keys in the lowering IR and retention of
unreachable intermediate nodes in the final append-only pool.

## Final State

The final architecture has these properties:

- `parse_native_dsl` creates exactly one `RulePool`.
- The loader lends that pool to parsing expansion/lowering for every module.
- Helper modules store `(StrId, RuleId)` pairs.
- Inherited grammar modules store pool-less grammar components whose IDs refer
  to the loader-wide pool.
- The root grammar takes ownership of the pool only after module loading and
  lowering are complete.
- Module-qualified rule references return an existing `RuleId`; they do not
  copy, rebuild, or remap a rule.
- The frontend emits `crate::rules::Rule` directly. `ARule`, `APrec`, and the
  native lowering rule arena have been removed.
- The native DSL's separate string pool has been removed. Native AST-derived
  strings and production rule strings use the `StrPool` embedded in `RulePool`.
- Metadata composition is copy-on-write so shared rule DAG nodes are never
  mutated accidentally.
- Choice flattening, structural deduplication, and depth validation are
  iterative and use retained scratch buffers.
- Tests inspect production `RuleId` values in the real pool. The temporary
  owned shadow grammar and its recursive normalizer have been deleted.

## Production Changes

### Direct production-rule emission

The old native lowering representation stored `ARule` nodes and later
materialized them into heap-based rules. The prototype instead stores
`crate::rules::RuleId` directly in lowering values:

```rust
pub enum Value {
    Int(i32),
    Str(StrId),
    Rule(RuleId),
    // ...
}
```

`IrPools` no longer contains a rule-node arena or rule-child arena. Rule nodes,
children, metadata, and strings are appended directly to `RulePool`.

This removes:

- `ARule`
- `APrec`
- the native-only `RuleId`
- the native rule-child pool
- the final `build_rule` materialization walk
- conversions from native strings to owned production strings

### One loader-wide pool

The pool is created once in `parse_native_dsl` and passed through `Loader` and
the lowering evaluator. No module owns a separate pool.

The root result is assembled in two stages:

1. Each grammar module produces `LoweredGrammar`, which contains grammar fields
   but not a pool.
2. The root `LoweredGrammar` is converted into `InputGrammar` by moving the
   loader-wide pool into it.

`LoweredGrammar` currently contains:

- grammar name as `StrId`
- variables as `Variable { name: StrId, root: RuleId }`
- external and extra roots as `RuleId`
- reserved sets as `StrId` plus `RuleId` roots
- inline, supertype, conflict, word, and precedence configuration as IDs

This avoids multiple owners for the pool while allowing every stored ID in
every loaded module to refer to the same domain.

### Imports and inheritance

Helpers store lowered rules as:

```rust
Vec<(StrId, RuleId)>
```

Grammar modules store a `LoweredGrammar`. `RuleTarget` identifies a helper
rule, grammar rule, or grammar external by module-local index. Resolving a
module rule is now a direct lookup that returns the existing `RuleId`.

Consequences:

- Qualified helper rules can be inlined without reconstruction.
- Transitive helper promotion reuses existing roots.
- Inherited variables and configuration roots are copied as small IDs.
- Overrides replace roots without cloning rule trees.
- Reserved sets can merge by `StrId` and reuse their existing roots.
- Spans and module IDs no longer need to be converted back into strings to move
  lowered rules between modules.

This was the primary correctness win of the prototype. The previous mapping
between spans, module-local strings, and final owned strings was a recurring
source of native frontend bugs.

### Metadata composition

The old heap representation could mutate or rebuild owned rule trees without
sharing concerns. Direct pooled rules form a DAG, so a rule ID may be used by
multiple call sites or modules.

Metadata combinators therefore use copy-on-write behavior:

- If the child is compatible metadata, its parameter record is copied and
  merged into a new parameter record.
- A new `Rule::Metadata` node is emitted.
- The original node and parameters remain unchanged.
- Token boundaries retain the prior behavior: metadata outside a token does
  not incorrectly merge through the token wrapper.

This behavior is required for correctness and should be preserved explicitly
in the clean port.

### Choice semantics

The production `RulePool` choice constructor and native DSL choice lowering
must preserve the established semantics:

- nested choices are flattened
- first occurrence order is retained
- structurally duplicate alternatives are removed
- a choice node remains a choice even when flattening produces one member

Native lowering uses retained buffers for this work:

- a rule-result buffer
- an iterative traversal stack
- a structural-equality stack

No `to_vec` copies are needed in the hot flatten/dedup path. Structural
equality is iterative through `RulePool::subtree_eq_with_scratch`.

The deduplication algorithm remains quadratic in the number of alternatives,
as before: every candidate may be compared with every already-retained
alternative. The rewrite removes repeated allocation and recursive comparison,
but does not change that asymptotic behavior.

### Iterative depth validation

The frontend can construct rules iteratively at depths beyond what downstream
recursive backend passes can consume safely. The prototype validates final
lowered rule depth with a retained iterative stack and returns
`RuleNestingTooDeep` instead of allowing a later stack overflow.

This validation should remain until the downstream recursive consumers become
iterative.

### Configuration representation

The following fields now flow through native lowering without owned-string
conversion:

- grammar name
- variable names
- inline names
- supertype names
- conflict names
- word name
- named precedences
- named-symbol rule nodes
- reserved-set names
- alias names
- field names
- reserved contexts

All use `StrId` from the shared production pool.

Inherited configuration still clones vectors when a child inherits a field
unchanged. These are copies of compact IDs, not owned rule trees or strings.
They are final-output vectors, so these allocations are generally necessary
unless the grammar representation itself changes to shared slices.

### Serialization

Native grammar serialization was updated to traverse `InputGrammar.pool` and
serialize `RuleId` roots directly.

`RuleJSON` and `PrecedenceValueJSON` derive `Serialize` and are visible within
the crate so the native serializer can reuse the grammar JSON schema rather
than maintaining a parallel schema.

The JSON round-trip test exposed an ordering requirement: the native grammar
must be normalized before comparison with a grammar reparsed through
`parse_grammar`, because that path removes unreachable variables and associated
configuration.

## Backend Changes

### Structural equality with retained scratch

`RulePool::subtree_eq_with_scratch` was added so hot callers can provide a
retained `Vec<(RuleId, RuleId)>`. The existing convenience `subtree_eq` remains
available and allocates its own short-lived stack.

The retained form is used by native choice deduplication and should be kept.

### Cross-pool subtree import experiment

`RulePool::import_subtree` and `RuleImportScratch` were introduced while the
prototype still had multiple pools. The function iteratively copies a rule DAG
between pools and remaps rule, child, metadata, and string IDs.

After switching to one loader-wide pool, native lowering stopped using this
API. Its only remaining caller is its own test. It is therefore experimental
debris for purposes of the clean native port and should not be included unless
another backend consumer independently needs cross-pool imports.

### Debug implementations

`Debug` was derived for production grammar containers and `RulePool`. These
implementations are useful for diagnostics and test failures and have no
runtime cost unless formatting is requested. There is no reason to avoid these
derives in the clean implementation.

## Test Changes

The initial prototype introduced an owned test-only grammar mirroring the old
heap-based representation. It normalized the production pool into owned
strings and recursive rules so the existing tests could compile quickly.

That bridge was useful only as a migration tool and was subsequently removed.
The final tests use:

- the real production `InputGrammar`
- direct access to production configuration fields
- `RuleId` lookup by rule name
- `RulePool::subtree_eq` when comparing two actual rules
- an `ExpectedRule` pattern type when concise structural expectations are
  clearer than manual node-by-node assertions

`ExpectedRule` is deliberately named differently from the real `Rule` type.
`Rule` always means `crate::rules::Rule`. Expected patterns may own strings and
vectors because they are test data, not production lowering state.

The pool matcher traverses the actual rule in place and resolves strings only
for comparison or failure output. It never materializes the actual grammar
into a shadow structure.

All native DSL test groups, including imports and inheritance, now use this
pool-native infrastructure.

## Performance Audit

### Improvements confirmed by inspection

- Final heap-rule materialization is gone.
- Rule strings are not copied into owned strings at the lowering boundary.
- Imported and inherited rule trees are not copied.
- Choice flattening does not use `to_vec` or allocate a fresh traversal stack.
- Structural equality can reuse a retained stack.
- Rule-depth validation reuses a retained stack.
- Metadata changes allocate only new pooled parameter and wrapper nodes.
- Inherited vectors contain compact IDs rather than recursive rules or strings.
- Module rule lookup is an index lookup returning a `RuleId`.

### Remaining owned strings in lowering objects

The most important unresolved allocation issue is:

```rust
pub object_pool: Vec<FxHashMap<String, ValueId>>
```

Object literal lowering allocates a `String` for every key. Reading inherited
reserved configuration resolves every reserved-set `StrId` back into a
`String` to construct another object map.

This should be changed during the clean port to:

```rust
pub object_pool: Vec<FxHashMap<StrId, ValueId>>
```

Object-field spans can be interned into the shared pool. Field-access errors can
resolve IDs into owned diagnostic strings only on the error path. This would
eliminate the remaining owned-string conversion in a normal lowering path and
extend the shared-pool correctness benefit to object configuration.

### Other owned-name tables

Some earlier phases still use owned string keys:

- module export maps use `FxHashMap<Box<str>, Export>`
- macro collection uses `FxHashMap<String, MacroId>` and `FxHashSet<String>`
- configuration flags use owned string maps
- typechecking records some object field names as owned strings

These mostly predate final lowering, but converting them to `StrId` is a
possible follow-up if the objective expands from pool-native rule lowering to a
fully interned native frontend. Export maps are especially relevant to the
correctness motivation because they connect module names to lowered IDs.

Doing this cleanly requires deciding when all parsed declaration names become
interned. It should not be mixed casually into the final-lowering port.

### Temporary concatenation buffers

Runtime `concat` and computed rule-name expansion build temporary `String`
buffers before interning their results. Some allocation is unavoidable when a
new concatenated string must be constructed, but the buffers could be retained
in scratch storage if profiling shows these operations are common.

### Unreachable pooled nodes

The direct approach appends every emitted rule node to the same pool that is
eventually returned in `InputGrammar`. This includes nodes that may no longer
be reachable from final roots:

- intermediate combinator results
- helper rules never materialized into the root grammar
- overridden roots
- metadata predecessors
- temporary results of macro and loop evaluation

The previous materialization phase naturally copied only reachable final
trees. The new approach avoids that copy but may retain more memory.

This is the main performance question requiring measurement. Before adding a
compaction/remapping phase, instrument realistic grammars with:

- total rule nodes allocated
- reachable rule nodes from all final roots
- total and reachable child entries
- total metadata entries
- total interned strings
- lowering wall time and peak memory

If unreachable storage is modest, retaining it is preferable to reintroducing
a final graph copy. If it is large, consider an optional final compaction pass
or an evaluator design that emits fewer transient wrapper nodes.

### Small avoidable allocations

- Undefined-symbol validation creates a fresh referenced-ID vector once per
  grammar. This can use retained lowering scratch if desired.
- Inheritance validation collects all inherit nodes into a vector even though
  it primarily needs the first two plus error notes.
- Start-rule selection resolves IDs back to strings for equality. With one
  pool, it should compare `StrId` directly.
- Missing-override diagnostics correctly allocate owned strings, but only on an
  error path.

## Correctness Audit

### Invariants confirmed

- There is one `RulePool::default()` in native parse entry.
- Every native `StrId` stored in lowered output belongs to that pool.
- Every native `RuleId` stored in a module or final grammar belongs to that
  pool.
- Module records cannot outlive the parse call's pool.
- The pool is moved into the root grammar only after lowering has completed.
- Imports and inheritance do not remap strings or rule IDs.
- Metadata operations do not mutate shared nodes.
- Choice flattening preserves source order and removes structural duplicates.
- Reserved inheritance preserves base order, replaces same-name sets in place,
  and appends new sets.
- Start-rule rotation operates after inherited, local, and helper variables are
  assembled.
- Anonymous external rules remain rules; named externals participate in name
  resolution through their pooled `NamedSymbol` IDs.
- Rule-depth validation remains iterative.

### Areas to tighten

- Start-rule lookup should compare `StrId`, not resolved text.
- `LoweredRef` currently stores the same pool reference in both variants. Pass
  the shared pool once to `build_exports` instead.
- A comment in helper materialization says a rule is cloned even though the
  implementation now reuses its ID.
- Consider renaming `LoweredGrammar` to `GrammarParts` or
  `LoweredGrammarParts` if the distinction from the final pool-owning
  `InputGrammar` is not immediately clear during the careful refactor.

## Validation Performed

- Full `tree-sitter-generate` library suite with `nativedsl`: 602 passed.
- Import test group: 53 passed.
- Inheritance test group: 38 passed.
- Deep iterative rule import test: 50,000 nested repeats.
- Native lowering iterative-depth tests passed.
- The worktree was clean before this report was added.
- `dod-ify` was never checked out for editing or modified.

Clippy was also run with warnings denied. It reported nine production warnings
and one test warning. They are cleanup-level findings rather than correctness
issues:

- one manual `debug_assert_eq`
- one needless return with `?`
- one manual `let ... else`
- one missing semicolon style warning
- one identity `map`
- one explicit-default style warning
- one missing `#[must_use]`
- one `Self` style warning in the unused cross-pool importer
- one unnecessarily mutable evaluator receiver
- one test-only unnecessary `vec!`

These should be cleared during the clean implementation rather than preserved
in a squashed commit.

## Scratch History Assessment

The following commits contain useful architectural discoveries, but should not
be cherry-picked mechanically into the final branch:

- `bd2d57d6f`: direct pooled rule lowering and native string-pool removal
- `cd4ca2cdb`: metadata composition, choice semantics, depth validation, and
  reserved inheritance fixes discovered by testing
- `11af9d46d`: retained scratch and iterative traversal work
- `965c358d8`: final one-pool architecture
- `8d0021258`: terminology cleanup
- `28b978bb5` through `fc169bac0`: final pool-native test design and shadow
  grammar removal

The following are explicitly disposable experiments:

- `3ff2ed1c0`: cross-pool subtree-import bridge
- `ec294f731`: root-only pool transfer experiment
- `866c077f4`: revert of that experiment
- the temporary shadow-grammar test adapter
- migration-only `pooled_*` and `as_same_pool` naming

The final implementation should be rebuilt coherently and squashed, using the
scratch branch as a behavioral reference rather than as the desired history.

## Recommended Clean-Port Plan

1. Rebase or reconstruct the native frontend directly on `dod-ify`.
2. Introduce one loader-owned `RulePool` at native parse entry.
3. Replace native `Str` with production `StrId` throughout expansion and
   lowering.
4. Convert lowering object keys to `StrId` at the same time.
5. Delete the native string pool.
6. Delete `ARule`, `APrec`, native `RuleId`, native rule nodes, and native rule
   children.
7. Emit production `Rule` nodes from the evaluator.
8. Introduce the pool-less intermediate grammar-parts structure.
9. Store helper and inherited module output as shared-pool IDs.
10. Make module-qualified imports return existing `RuleId` values directly.
11. Implement metadata copy-on-write.
12. Implement iterative choice flattening/deduplication with retained scratch.
13. Keep iterative final-depth validation.
14. Merge inherited configuration and reserved sets by ID.
15. Update serialization to traverse the production pool.
16. Port tests directly to production `InputGrammar` plus `ExpectedRule`; never
    introduce the shadow grammar.
17. Remove the unused cross-pool import API unless another consumer needs it.
18. Remove redundant pool parameters, stale comments, and Clippy warnings.
19. Instrument reachable versus allocated pool storage.
20. Benchmark representative native grammars before deciding whether final
    pool compaction is justified.

## Open Questions

1. **How much unreachable rule storage remains in realistic grammars?**
   This determines whether direct append-only emission is unequivocally better
   than reachable-only final materialization for memory as well as time.

2. **Should object keys become `StrId` as part of the clean port?**
   The audit recommends yes because they are in lowering's normal path and are
   the clearest remaining owned-string bridge.

3. **How far should interning extend into earlier frontend phases?**
   Export maps, macro maps, flags, and typechecker object fields still own
   strings. Converting all of them may be valuable, but broadens the refactor.

4. **Should `LoweredGrammar` be renamed?**
   A `GrammarParts` name may communicate more clearly that it contains IDs but
   deliberately does not own their pool.

5. **Should `RulePool::subtree_eq` itself avoid allocation?**
   Hot callers already use `subtree_eq_with_scratch`. The convenience method's
   allocation is acceptable for cold callers, but the API could make the cost
   more obvious.

6. **Is final compaction ever worthwhile?**
   It would reduce retained intermediates but reintroduce traversal, remapping,
   and allocation. This should be decided from measurements, not aesthetics.

7. **Should native serialization continue reusing `RuleJSON`?**
   Reuse avoids schema drift, but requires crate-visible serialization types.
   That trade appears reasonable.

8. **Should the test-only call to `InputGrammar::normalize` widen production
   visibility?**
   Prefer avoiding API visibility changes made solely for one test if an equally
   clear round-trip test can be structured through an existing public pipeline.

9. **What benchmark corpus should gate the final refactor?**
   At minimum it should include a grammar with many macros, large choices,
   imports, multiple inheritance levels, reserved sets, and computed names.

## Conclusion

The prototype demonstrates that the native DSL can sit naturally on the
`dod-ify` backend. A single shared pool removes the most error-prone mapping
boundary, makes module rule transfer an ID lookup, eliminates final heap-rule
materialization, and supports allocation-conscious iterative lowering.

The clean implementation should preserve that architecture while tightening
the last normal-path owned strings in lowering objects and measuring the memory
cost of retaining unreachable intermediate pool nodes. No evidence from the
prototype suggests that a bridge back to the old rule representation is needed.
