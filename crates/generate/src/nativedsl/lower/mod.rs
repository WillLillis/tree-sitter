//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].
//!
//! Pipeline:
//! 1. [`evaluate`] walks the root grammar's items, populating an
//!    [`evaluator::Evaluator`] with intermediate IR (values + `ARule` pools).
//! 2. [`build_grammar`] consumes the [`EvalResult`] and the optional
//!    inherited base grammar, materializing a final [`InputGrammar`] with
//!    overrides applied and unset fields inherited.

mod error;
mod evaluator;
mod repr;

use rustc_hash::{FxHashMap, FxHashSet};

use crate::{
    grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable},
    nativedsl::{ImportedRule, Module, ast::ModuleContext},
    rules::Rule,
};

use super::{
    LowerError, ModuleId, Note, NoteMessage,
    ast::{ForId, Node, NodeId, SharedAst, Span, Spanned},
};
use crate::{
    rules::{RuleImportScratch, RulePool},
    strpool::StrId as Str,
};

pub use error::{DisallowedItemKind, LowerErrorKind, LowerResult};

/// Rule-decl name source. Resolution is deferred to the final build step
/// so the body-lowering loop doesn't hold a `StringPool` borrow across
/// `lower_to_rule` calls (which may intern further entries).
#[derive(Clone, Copy)]
enum NameSource {
    Span(Span), // Node::Rule
    Str(Str),   // Node::ExpandedRule
}

use evaluator::{Evaluator, Task};
use repr::{IrPools, RuleId, ValueId};

const MAX_CALL_DEPTH: u16 = 128;
/// Maximum nesting depth of a materialized [`Rule`] tree. The evaluator builds
/// rules iteratively (no native-stack bound), but the resulting tree is consumed
/// by the *recursive* shared grammar backend (`crate::prepare_grammar`, also used
/// by the grammar.js front-end). This bounds the tree depth so a pathologically
/// deep rule - reachable via macro expansion, whose product of `MAX_CALL_DEPTH`
/// and per-body nesting far exceeds this - is rejected with a clean error here
/// rather than overflowing the backend. Lift it once that backend is iterative.
const MAX_RULE_DEPTH: u16 = 256;

/// One stack frame for the macro-call trace.
#[derive(Clone, Copy)]
pub(super) struct CallFrame {
    /// Span of the macro's name in its defining module.
    pub name_span: Span,
    pub name_mod: ModuleId,
    /// Span of the call site in the caller's module.
    pub call_span: Span,
    pub caller_mod: ModuleId,
}

/// Long-lived state shared across grammar lowerings in one `parse_native_dsl`
/// call.
///
/// Each module's let bindings are evaluated eagerly when it is lowered; the
/// `let_values` cache (keyed by `Let` node id) then serves cross-module reads
/// without re-evaluating. `scratch` persists across grammars to reuse allocated
/// capacity but its contents are cleared between them.
#[derive(Default)]
pub struct LoweringState {
    pub(super) ir: IrPools,
    // Cross-grammar cache: let values keyed by their Let node id, shared across
    // every module's evaluation in one run.
    let_values: FxHashMap<NodeId, ValueId>,
    scratch: Scratch,
}

/// Per-grammar scratch buffers. Cleared (capacity retained) at the start of
/// each grammar lowering via [`Scratch::clear`].
#[allow(clippy::struct_field_names)]
#[derive(Default)]
struct Scratch {
    call_stack: Vec<CallFrame>,
    /// Work stack driving the iterative expression/rule evaluation in
    /// [`evaluator`]; shared across walks (re-entrant macro/for/let evaluation
    /// nests on it via base offsets) so its capacity is retained.
    work: Vec<Task>,
    /// Let bindings whose value is mid-evaluation; reentry signals a cycle.
    lets_in_progress: FxHashSet<NodeId>,
    macro_args: Vec<ValueId>,
    macro_arg_bases: Vec<usize>,
    for_binding_values: Vec<ValueId>,
    for_binding_frames: Vec<(ForId, usize)>,
    val_scratch: Vec<ValueId>,
    rule_scratch: Vec<RuleId>,
    /// Retained traversal stack used while flattening choices and validating
    /// pooled rule depth.
    rule_walk: Vec<(RuleId, u16)>,
    rule_eq: Vec<(RuleId, RuleId)>,
    import_map: Vec<Option<RuleId>>,
    import_scratch: RuleImportScratch,
    /// Result-stack bases for variable-arity [`Task::Combine`]s, kept out of the
    /// `Task` itself so the work stack stays 8 bytes/entry. Combines nest LIFO, so
    /// this is a plain stack.
    combine_bases: Vec<u32>,
}

impl Scratch {
    fn clear(&mut self) {
        self.call_stack.clear();
        self.work.clear();
        self.lets_in_progress.clear();
        self.macro_args.clear();
        self.macro_arg_bases.clear();
        self.for_binding_values.clear();
        self.for_binding_frames.clear();
        self.val_scratch.clear();
        self.rule_scratch.clear();
        self.rule_walk.clear();
        self.rule_eq.clear();
        self.import_map.clear();
        self.combine_bases.clear();
    }
}

impl LoweringState {
    fn reset_per_grammar(&mut self) {
        self.scratch.clear();
    }
}

struct EvalResult {
    language: Str,
    rules: Vec<(Str, RuleId)>,
    overrides: Vec<(Str, RuleId, Span)>,
    extras: Option<Vec<RuleId>>,
    externals: Option<Vec<RuleId>>,
    inline: Option<Vec<Str>>,
    supertypes: Option<Vec<Str>>,
    word: Option<Str>,
    start: Option<(Str, Span)>,
    conflicts: Option<Vec<Vec<Str>>>,
    precedences: Option<Vec<Vec<PrecedenceEntry>>>,
    reserved: Option<Vec<ReservedWordContext>>,
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
/// - `previous` contains the modules already loaded
/// - `current` is the root module being lowered (not yet pushed into `previous`)
/// - `state` persists across the whole `parse_native_dsl` pipeline.
pub fn lower_with_base(
    state: &mut LoweringState,
    strings: &mut RulePool,
    shared: &SharedAst,
    previous: &[Module],
    current: &ModuleContext,
    imported_rules: &[ImportedRule],
) -> LowerResult<InputGrammar> {
    let base_grammar = current
        .inherit_module(&shared.arena)
        .and_then(|(idx, _)| previous[usize::from(idx)].lowered());
    let result = evaluate(state, strings, shared, previous, current)?;
    state.scratch.rule_walk.clear();
    for &(_, root) in &result.rules {
        state.scratch.rule_walk.push((root, 1));
        while let Some((id, depth)) = state.scratch.rule_walk.pop() {
            if depth > MAX_RULE_DEPTH {
                return Err(LowerError::new(
                    LowerErrorKind::RuleNestingTooDeep,
                    Span::new(0, 0),
                ));
            }
            match strings.node(id) {
                Rule::Seq(range) | Rule::Choice(range) => state.scratch.rule_walk.extend(
                    strings
                        .child_slice(range)
                        .iter()
                        .map(|&child| (child, depth + 1)),
                ),
                Rule::Repeat(child)
                | Rule::Metadata { rule: child, .. }
                | Rule::Reserved { rule: child, .. } => {
                    state.scratch.rule_walk.push((child, depth + 1))
                }
                _ => {}
            }
        }
    }
    let grammar = build_grammar(
        strings,
        current,
        result,
        base_grammar,
        previous,
        imported_rules,
    )?;
    check_symbol_completeness(shared, current, previous, &grammar)?;
    Ok(grammar)
}

/// Benchmark hook: run only the eval walk (AST -> `ARule`/value IR pool) over the
/// root items, skipping the `build_rule` (`ARule` -> [`Rule`]) materialization.
/// Isolates the permanent lowering pass from the transitional `Rule`-building
/// bridge, so the iterative eval engine can be timed against the recursive one
/// without the `Rule`-box allocation noise. Not part of the real pipeline.
#[doc(hidden)]
pub fn eval_only_for_bench(
    state: &mut LoweringState,
    strings: &mut RulePool,
    shared: &SharedAst,
    previous: &[Module],
    current: &ModuleContext,
) -> LowerResult<()> {
    let mut eval = Evaluator::new(state, strings, shared, previous, current);
    let items = lower_items(&mut eval, shared, current)?;
    std::hint::black_box(&items);
    Ok(())
}

/// Every `NamedSymbol` a grammar references must resolve to a definition.
/// An `expect` forward-decl makes a name *referenceable* without defining it,
/// so an unfulfilled `expect` must be caught.
fn check_symbol_completeness(
    shared: &SharedAst,
    current: &ModuleContext,
    previous: &[Module],
    grammar: &InputGrammar,
) -> LowerResult<()> {
    if !current.has_forward_decls && !previous.iter().any(|m| m.ctx().has_forward_decls) {
        return Ok(());
    }
    let mut defined: FxHashSet<Str> = grammar.variables.iter().map(|v| v.name).collect();
    for &root in &grammar.external_roots {
        if let Rule::NamedSymbol(name) = grammar.pool.node(root) {
            defined.insert(name);
        }
    }
    let mut referenced = Vec::new();
    for v in &grammar.variables {
        grammar
            .pool
            .collect_referenced_ids(v.root, false, &mut referenced);
    }
    let undefined: Vec<&str> = referenced
        .into_iter()
        .filter(|name| !defined.contains(name))
        .map(|name| grammar.pool.resolve(name))
        .collect();
    if undefined.is_empty() {
        return Ok(());
    }
    Err(undefined_symbols_error(
        shared, current, previous, undefined,
    ))
}

/// Build the error for one or more dangling `names`.
fn undefined_symbols_error(
    shared: &SharedAst,
    current: &ModuleContext,
    previous: &[Module],
    mut names: Vec<&str>,
) -> LowerError {
    names.sort_unstable();
    names.dedup();
    let kind = LowerErrorKind::UndefinedSymbols(names.iter().map(|n| (*n).to_string()).collect());
    let mut notes = names
        .iter()
        .filter_map(|name| forward_decl_note(shared, current, previous, name));
    let Some(primary) = notes.next() else {
        // No `expect` behind any dangling symbol: anchor at the grammar block,
        // which `validate_grammar` guarantees is present in a grammar module.
        let block = current
            .root_items
            .iter()
            .find(|&&id| matches!(shared.arena.get(id), Node::Grammar));
        return match block {
            Some(&id) => LowerError::new(kind, shared.arena.span(id)),
            None => LowerError::without_span(kind),
        };
    };
    let mut err = LowerError::new(kind, primary.span).with_source(&primary.src, &primary.path);
    for note in notes {
        err.add_note(note);
    }
    err
}

/// A `forward-declared here` note anchored at the `expect <name>` decl, searched
/// in this grammar then any imported helper so the note renders against the file
/// that made the promise. `None` if no `expect` declares `name`.
fn forward_decl_note(
    shared: &SharedAst,
    current: &ModuleContext,
    previous: &[Module],
    name: &str,
) -> Option<Note> {
    if let Some(span) = forward_decl_span(shared, current, name) {
        return Some(current.note(NoteMessage::ForwardDeclaredHere, span));
    }
    previous.iter().find_map(|module| match module {
        Module::Helper { ctx, .. } => forward_decl_span(shared, ctx, name)
            .map(|span| ctx.note(NoteMessage::ForwardDeclaredHere, span)),
        Module::Grammar { .. } => None,
    })
}

/// Span of an `expect <name>` forward-decl in `ctx`, if one is present.
fn forward_decl_span(shared: &SharedAst, ctx: &ModuleContext, name: &str) -> Option<Span> {
    ctx.root_items
        .iter()
        .find_map(|&id| match *shared.arena.get(id) {
            Node::Forward { name: decl } if ctx.text(decl) == name => Some(decl),
            _ => None,
        })
}

/// One lowered top-level item: a rule (plain or `override`) as a `RuleId`, tagged
/// so grammar lowering can split overrides out while helper lowering treats them
/// all as plain rules. Built in source order.
struct LoweredItem {
    src: NameSource,
    rule_id: RuleId,
    is_override: bool,
    /// Attribution span for an override (the rule name, or the macro call site).
    span: Span,
}

/// Walk `root_items`, evaluating lets and lowering each rule / expanded rule to
/// a `RuleId` in source order. Shared by grammar and helper lowering; the caller
/// decides what to do with the `override`-tagged items.
fn lower_items(
    eval: &mut Evaluator,
    shared: &SharedAst,
    ctx: &ModuleContext,
) -> LowerResult<Vec<LoweredItem>> {
    let mut items = Vec::with_capacity(ctx.root_items.len());
    for &item_id in &ctx.root_items {
        match shared.arena.get(item_id) {
            // Grammar block (config is read separately), macros, and externals
            // register names but don't materialize a rule here.
            Node::Grammar | Node::Macro(_) | Node::Forward { .. } => {}
            Node::Let { .. } => {
                eval.eval_let(item_id)?;
            }
            &Node::Rule {
                is_override,
                name,
                body,
            } => {
                let rule_id = eval.lower_to_rule(body)?;
                items.push(LoweredItem {
                    src: NameSource::Span(name),
                    rule_id,
                    is_override,
                    span: name,
                });
            }
            &Node::ExpandedRule(expand_id) => {
                let exp = *shared.pools.get_expansion(expand_id);
                // expand_macro_calls sets the item span to the macro call site,
                // used for override attribution diagnostics.
                let span = shared.arena.span(item_id);
                let rule_id = eval.lower_expansion(expand_id, span)?;
                items.push(LoweredItem {
                    src: NameSource::Str(exp.name),
                    rule_id,
                    is_override: exp.is_override,
                    span,
                });
            }
            _ => unreachable!(),
        }
    }
    Ok(items)
}

/// Lower a helper module's rules into a name-keyed list.
///   - Lets/macros are evaluated through the same Evaluator as grammar lowering;
///   - `external` decls and macros register names but don't materialize. Grammar
///     blocks and direct override rules are rejected by validation; an override
///     reaching the top level via a called macro is rejected below.
pub fn lower_helper(
    state: &mut LoweringState,
    strings: &mut RulePool,
    shared: &SharedAst,
    previous: &[super::Module],
    current: &super::ModuleContext,
) -> LowerResult<Vec<(Str, RuleId)>> {
    let mut eval = Evaluator::new(state, strings, shared, previous, current);
    let mut rules = Vec::new();
    for it in lower_items(&mut eval, shared, current)? {
        if it.is_override {
            // A helper can't inherit, so an `override` reaching its top level via
            // a called rules-macro (a direct `override rule` is rejected earlier
            // by validate_import_items) has nothing to override. Reject it rather
            // than silently demoting it to a plain rule.
            return Err(LowerError::new(
                LowerErrorKind::ModuleDisallowedItem(DisallowedItemKind::OverrideRule),
                it.span,
            ));
        }
        rules.push((resolve_name(it.src, &mut eval, current), it.rule_id));
    }
    Ok(rules)
}

fn resolve_name(src: NameSource, eval: &mut Evaluator, ctx: &super::ModuleContext) -> Str {
    match src {
        NameSource::Span(s) => eval.strings.intern(ctx.text(s)),
        NameSource::Str(s) => s,
    }
}

fn evaluate(
    state: &mut LoweringState,
    strings: &mut RulePool,
    shared: &SharedAst,
    previous: &[super::Module],
    ctx: &super::ModuleContext,
) -> LowerResult<EvalResult> {
    let mut eval = Evaluator::new(state, strings, shared, previous, ctx);
    let mut rule_entries: Vec<(NameSource, RuleId)> = Vec::new();
    let mut override_entries: Vec<(NameSource, RuleId, Span)> = Vec::new();
    for it in lower_items(&mut eval, shared, ctx)? {
        if it.is_override {
            override_entries.push((it.src, it.rule_id, it.span));
        } else {
            rule_entries.push((it.src, it.rule_id));
        }
    }
    // grammar_config is guaranteed present by `validate_grammar`, language
    // is guaranteed present by the parser (`MissingLanguageField` error).
    let config = ctx.grammar_config.as_ref().unwrap();
    let language = eval.strings.intern(config.language.as_ref().unwrap());

    let rules: Vec<(Str, RuleId)> = rule_entries
        .into_iter()
        .map(|(src, rid)| (src, rid))
        .map(|(src, rid)| (resolve_name(src, &mut eval, ctx), rid))
        .collect();
    let overrides: Vec<(Str, RuleId, Span)> = override_entries
        .into_iter()
        .map(|(src, rid, span)| (resolve_name(src, &mut eval, ctx), rid, span))
        .collect();

    Ok(EvalResult {
        language,
        rules,
        overrides,
        extras: config
            .extras
            .map(|id| eval.eval_rule_list(id))
            .transpose()?,
        externals: config
            .externals
            .map(|id| eval.eval_rule_list(id))
            .transpose()?,
        inline: config
            .inline
            .map(|id| eval.eval_name_list(id))
            .transpose()?,
        supertypes: config
            .supertypes
            .map(|id| eval.eval_name_list(id))
            .transpose()?,
        word: config.word.map(|id| eval.eval_rule_name(id)).transpose()?,
        start: config
            .start
            .map(|id| Ok((eval.eval_rule_name(id)?, shared.arena.span(id))))
            .transpose()?,
        conflicts: config
            .conflicts
            .map(|id| eval.eval_conflicts(id))
            .transpose()?,
        precedences: config
            .precedences
            .map(|id| eval.eval_precedences(id))
            .transpose()?,
        reserved: config
            .reserved
            .map(|id| eval.eval_reserved(id))
            .transpose()?,
    })
}

fn build_grammar(
    staging: &mut RulePool,
    ctx: &ModuleContext,
    result: EvalResult,
    base: Option<&InputGrammar>,
    previous: &[Module],
    imported_rules: &[ImportedRule],
) -> LowerResult<InputGrammar> {
    let mut overrides: FxHashMap<Str, Spanned<RuleId>> = FxHashMap::default();
    for (name, rule, span) in result.overrides {
        overrides.insert(name, Spanned::new(rule, span));
    }

    let mut pool = RulePool::default();
    let mut staging_map = Vec::new();
    let mut base_map = Vec::new();
    let mut import_scratch = RuleImportScratch::default();
    let mut variables = Vec::new();

    // Base rules first - preserves the inherited grammar's start rule.
    if let Some(base) = base {
        variables.reserve(base.variables.len());
        for v in &base.variables {
            let base_name = base.pool.resolve(v.name);
            let override_name = staging.intern(base_name);
            let root = if let Some(Spanned { value: root, .. }) = overrides.remove(&override_name) {
                pool.import_subtree(staging, root, &mut staging_map, &mut import_scratch)
            } else {
                pool.import_subtree(&base.pool, v.root, &mut base_map, &mut import_scratch)
            };
            let name = pool.intern(base_name);
            variables.push(Variable { name, root });
        }
    }

    for (name, root) in result.rules {
        let name = pool.intern(staging.resolve(name));
        let root = pool.import_subtree(staging, root, &mut staging_map, &mut import_scratch);
        variables.push(Variable { name, root });
    }

    // Helper rules (materialized from the shared imported-rule list) can also be
    // override targets. The clone happens here, once, at final assembly.
    for ir in imported_rules {
        expect_pat!(
            Module::Helper { lowered_rules, .. },
            &previous[usize::from(ir.module)]
        );
        let &(name, helper_root) = &lowered_rules[ir.index as usize];
        let root = overrides.remove(&name).map_or(helper_root, |s| s.value);
        let root = pool.import_subtree(staging, root, &mut staging_map, &mut import_scratch);
        let name = pool.intern(staging.resolve(name));
        variables.push(Variable { name, root });
    }

    if !overrides.is_empty() {
        // Names (sorted) go in the message; each override's location becomes a
        // located snippet - the first (in source order) as the primary span,
        // the rest as `override declared here` notes. This gets all locations
        // through the generic note machinery, no renderer special-casing.
        let mut entries: Vec<(String, Span)> = overrides
            .into_iter()
            .map(|(name, s)| (staging.resolve(name).to_string(), s.span))
            .collect();
        entries.sort_unstable_by_key(|(_, span)| span.start);
        let mut names: Vec<String> = entries.iter().map(|(name, _)| name.clone()).collect();
        names.sort_unstable();
        let (_, primary) = entries[0];
        let mut err = LowerError::new(LowerErrorKind::OverrideRuleNotFound(names), primary);
        for (_, span) in &entries[1..] {
            err.add_note(ctx.note(NoteMessage::OverrideDeclaredHere, *span));
        }
        return Err(err);
    }

    // Tree-sitter's start symbol is `variables[0]` (a non-terminal), so honor
    // `start: <rule>` by rotating the named rule into position 0. Inherited,
    // local, and helper rules are all in `variables`. An external token is a
    // valid rule reference but lives in `external_tokens`, not `variables`, so
    // it can't be the start symbol.
    if let Some((name, span)) = result.start {
        let name_text = staging.resolve(name);
        let pos = variables
            .iter()
            .position(|v| pool.resolve(v.name) == name_text)
            .ok_or_else(|| {
                LowerError::new(
                    LowerErrorKind::ExternalCannotBeStart(name_text.to_string()),
                    span,
                )
            })?;
        if pos != 0 {
            let v = variables.remove(pos);
            variables.insert(0, v);
        }
    }

    let import_staging_roots = |roots: Vec<RuleId>,
                                pool: &mut RulePool,
                                map: &mut Vec<_>,
                                scratch: &mut RuleImportScratch| {
        roots
            .into_iter()
            .map(|root| pool.import_subtree(staging, root, map, scratch))
            .collect::<Vec<_>>()
    };
    let external_roots = if let Some(roots) = result.externals {
        import_staging_roots(roots, &mut pool, &mut staging_map, &mut import_scratch)
    } else if let Some(base) = base {
        base.external_roots
            .iter()
            .map(|&root| pool.import_subtree(&base.pool, root, &mut base_map, &mut import_scratch))
            .collect()
    } else {
        Vec::new()
    };
    let extra_roots = if let Some(roots) = result.extras {
        import_staging_roots(roots, &mut pool, &mut staging_map, &mut import_scratch)
    } else if let Some(base) = base {
        base.extra_roots
            .iter()
            .map(|&root| pool.import_subtree(&base.pool, root, &mut base_map, &mut import_scratch))
            .collect()
    } else {
        let pattern = pool.intern("\\s");
        vec![pool.pattern(pattern, Default::default())]
    };
    let map_staging_strs = |ids: Vec<Str>, pool: &mut RulePool| {
        ids.into_iter()
            .map(|id| pool.intern(staging.resolve(id)))
            .collect::<Vec<_>>()
    };
    let inline_names = match result.inline {
        Some(ids) => map_staging_strs(ids, &mut pool),
        None => base.map_or_else(Vec::new, |b| {
            b.inline_names
                .iter()
                .map(|&s| pool.intern(b.pool.resolve(s)))
                .collect()
        }),
    };
    let supertype_names = match result.supertypes {
        Some(ids) => map_staging_strs(ids, &mut pool),
        None => base.map_or_else(Vec::new, |b| {
            b.supertype_names
                .iter()
                .map(|&s| pool.intern(b.pool.resolve(s)))
                .collect()
        }),
    };
    let conflict_names = match result.conflicts {
        Some(groups) => groups
            .into_iter()
            .map(|g| map_staging_strs(g, &mut pool))
            .collect(),
        None => base.map_or_else(Vec::new, |b| {
            b.conflict_names
                .iter()
                .map(|g| g.iter().map(|&s| pool.intern(b.pool.resolve(s))).collect())
                .collect()
        }),
    };
    let precedence_orderings = match result.precedences {
        None => base.map_or_else(Vec::new, |b| {
            b.precedence_orderings
                .iter()
                .map(|g| {
                    g.iter()
                        .map(|entry| match *entry {
                            PrecedenceEntry::Name(s) => {
                                PrecedenceEntry::Name(pool.intern(b.pool.resolve(s)))
                            }
                            PrecedenceEntry::Symbol(s) => {
                                PrecedenceEntry::Symbol(pool.intern(b.pool.resolve(s)))
                            }
                        })
                        .collect()
                })
                .collect()
        }),
        Some(groups) => groups
            .into_iter()
            .map(|g| {
                g.into_iter()
                    .map(|entry| match entry {
                        PrecedenceEntry::Name(s) => {
                            PrecedenceEntry::Name(pool.intern(staging.resolve(s)))
                        }
                        PrecedenceEntry::Symbol(s) => {
                            PrecedenceEntry::Symbol(pool.intern(staging.resolve(s)))
                        }
                    })
                    .collect()
            })
            .collect(),
    };
    let word_name = match result.word {
        Some(s) => Some(pool.intern(staging.resolve(s))),
        None => base.and_then(|b| b.word_name.map(|s| pool.intern(b.pool.resolve(s)))),
    };
    let name = pool.intern(staging.resolve(result.language));

    // Reserved-set inheritance is completed below once all roots share `pool`.
    let reserved_sets = if let Some(sets) = result.reserved {
        let mut merged = base.map_or_else(Vec::new, |base| {
            base.reserved_sets
                .iter()
                .map(|set| ReservedWordContext {
                    name: pool.intern(base.pool.resolve(set.name)),
                    roots: set
                        .roots
                        .iter()
                        .map(|&root| {
                            pool.import_subtree(
                                &base.pool,
                                root,
                                &mut base_map,
                                &mut import_scratch,
                            )
                        })
                        .collect(),
                })
                .collect()
        });
        for set in sets {
            let name = pool.intern(staging.resolve(set.name));
            let replacement = ReservedWordContext {
                name,
                roots: import_staging_roots(
                    set.roots,
                    &mut pool,
                    &mut staging_map,
                    &mut import_scratch,
                ),
            };
            if let Some(index) = merged.iter().position(|old| old.name == name) {
                merged[index] = replacement;
            } else {
                merged.push(replacement);
            }
        }
        merged
    } else if let Some(base) = base {
        base.reserved_sets
            .iter()
            .map(|set| ReservedWordContext {
                name: pool.intern(base.pool.resolve(set.name)),
                roots: set
                    .roots
                    .iter()
                    .map(|&root| {
                        pool.import_subtree(&base.pool, root, &mut base_map, &mut import_scratch)
                    })
                    .collect(),
            })
            .collect()
    } else {
        Vec::new()
    };

    Ok(InputGrammar {
        pool,
        name,
        variables,
        external_roots,
        extra_roots,
        reserved_sets,
        supertype_names,
        conflict_names,
        inline_names,
        word_name,
        precedence_orderings,
    })
}
