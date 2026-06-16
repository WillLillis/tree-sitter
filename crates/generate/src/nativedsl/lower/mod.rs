//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].
//!
//! Pipeline:
//! 1. [`evaluate`] walks the root grammar's items, populating an
//!    [`evaluator::Evaluator`] with intermediate IR (values + `ARule` pools).
//! 2. [`build_grammar`] consumes the [`EvalResult`] and the optional
//!    inherited base grammar, materializing a final [`InputGrammar`] with
//!    overrides applied and unset fields inherited.

mod evaluator;
mod repr;

use std::path::PathBuf;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType},
    nativedsl::{Module, ast::ModuleContext},
    rules::Rule,
};

use super::{
    LowerError, ModuleId,
    ast::{ForId, Node, NodeId, SharedAst, Span, Spanned},
    string_pool::{Str, StringPool},
};

/// Rule-decl name source. Resolution is deferred to the final build step
/// so the body-lowering loop doesn't hold a `StringPool` borrow across
/// `lower_to_rule` calls (which may intern further entries).
#[derive(Clone, Copy)]
enum NameSource {
    Span(Span), // Node::Rule
    Str(Str),   // Node::ExpandedRule
}

use evaluator::Evaluator;
use repr::{IrPools, LoadedModules, RuleId, ValueId};

const MAX_CALL_DEPTH: u16 = 128;

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
/// `ir` and the caches make imported/inherited let bindings evaluate exactly
/// once; `scratch` persists across grammars to reuse allocated capacity but
/// its contents are cleared between them.
#[derive(Default)]
pub struct LoweringState {
    pub(super) ir: IrPools,
    // Cross-grammar caches:
    loaded: LoadedModules,
    let_values: FxHashMap<NodeId, ValueId>,
    scratch: Scratch,
}

/// Per-grammar scratch buffers. Cleared (capacity retained) at the start of
/// each grammar lowering via [`Scratch::clear`].
#[allow(clippy::struct_field_names)]
#[derive(Default)]
struct Scratch {
    call_stack: Vec<CallFrame>,
    macro_args: Vec<ValueId>,
    macro_arg_bases: Vec<usize>,
    for_binding_values: Vec<ValueId>,
    for_binding_frames: Vec<(ForId, usize)>,
    val_scratch: Vec<ValueId>,
    rule_scratch: Vec<RuleId>,
}

impl Scratch {
    fn clear(&mut self) {
        self.call_stack.clear();
        self.macro_args.clear();
        self.macro_arg_bases.clear();
        self.for_binding_values.clear();
        self.for_binding_frames.clear();
        self.val_scratch.clear();
        self.rule_scratch.clear();
    }
}

impl LoweringState {
    fn reset_per_grammar(&mut self) {
        self.scratch.clear();
    }
}

struct EvalResult {
    language: String,
    rules: Vec<(String, Rule)>,
    overrides: Vec<(String, Rule, Span)>,
    extras: Option<Vec<Rule>>,
    externals: Option<Vec<Rule>>,
    inline: Option<Vec<String>>,
    supertypes: Option<Vec<String>>,
    word: Option<String>,
    start: Option<String>,
    conflicts: Option<Vec<Vec<String>>>,
    precedences: Option<Vec<Vec<PrecedenceEntry>>>,
    reserved: Option<Vec<ReservedWordContext<Rule>>>,
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
/// - `previous` contains the modules already loaded
/// - `current` is the root module being lowered (not yet pushed into `previous`)
/// - `state` persists across the whole `parse_native_dsl` pipeline.
pub fn lower_with_base(
    state: &mut LoweringState,
    strings: &mut StringPool,
    shared: &SharedAst,
    previous: &[Module],
    current: &ModuleContext,
) -> LowerResult<InputGrammar> {
    let base_grammar = current
        .inherit_module(&shared.arena)
        .and_then(|(idx, _)| previous[idx as usize].lowered());
    let helper_rules = collect_helper_rules(shared, current, previous);
    let result = evaluate(state, strings, shared, previous, current)?;
    build_grammar(result, base_grammar, helper_rules)
}

/// Collect rules from all transitively-reachable imported helpers, in
/// depth-first order. Cloned because helpers may be referenced from
/// multiple parents (diamond imports).
fn collect_helper_rules(
    shared: &SharedAst,
    current: &ModuleContext,
    previous: &[Module],
) -> Vec<(String, Rule)> {
    let mut collected = Vec::new();
    super::for_each_imported_helper::<std::convert::Infallible>(
        &shared.arena,
        &current.module_refs,
        previous,
        |_idx, module, _ref_span| {
            let Module::Helper { lowered_rules, .. } = module else {
                unreachable!()
            };
            collected.extend(lowered_rules.iter().cloned());
            Ok(())
        },
    )
    .unwrap();
    collected
}

/// Lower a helper module's rules into a name-keyed list.
///   - Lets/macros are evaluated through the same Evaluator as grammar lowering;
///   - `external` decls and macros register names but don't materialize. Helper
///     validation rejects grammar blocks and override rules.
pub fn lower_helper(
    state: &mut LoweringState,
    strings: &mut StringPool,
    shared: &SharedAst,
    previous: &[super::Module],
    current: &super::ModuleContext,
) -> LowerResult<Vec<(String, Rule)>> {
    let mut eval = Evaluator::new(state, strings, shared, previous, current);
    let mut rule_entries: Vec<(NameSource, RuleId)> = Vec::new();

    for &item_id in &current.root_items {
        match shared.arena.get(item_id) {
            Node::Macro(_) | Node::External { .. } => {}
            Node::Let { value, .. } => {
                eval.eval_let(item_id, *value)?;
            }
            &Node::Rule {
                name,
                body,
                is_override: false,
            } => {
                let rule_id = eval.lower_to_rule(body)?;
                rule_entries.push((NameSource::Span(name), rule_id));
            }
            &Node::ExpandedRule(expand_id) => {
                let name = shared.pools.get_expansion(expand_id).name;
                let rule_id = eval.lower_expansion(expand_id, shared.arena.span(item_id))?;
                rule_entries.push((NameSource::Str(name), rule_id));
            }
            // Validation rejects grammar blocks and override rules in helpers.
            _ => unreachable!(),
        }
    }

    Ok(rule_entries
        .into_iter()
        .map(|(src, rid)| (resolve_name(src, &eval, current), eval.build_rule(rid)))
        .collect())
}

fn resolve_name(src: NameSource, eval: &Evaluator, ctx: &super::ModuleContext) -> String {
    match src {
        NameSource::Span(s) => ctx.text(s).to_string(),
        NameSource::Str(s) => eval.strings.resolve_local(s, &ctx.source).to_string(),
    }
}

fn evaluate(
    state: &mut LoweringState,
    strings: &mut StringPool,
    shared: &SharedAst,
    previous: &[super::Module],
    ctx: &super::ModuleContext,
) -> LowerResult<EvalResult> {
    let mut eval = Evaluator::new(state, strings, shared, previous, ctx);
    let mut rule_entries: Vec<(NameSource, RuleId)> = Vec::with_capacity(ctx.root_items.len());
    let mut override_entries: Vec<(NameSource, RuleId, Span)> = Vec::new();

    for &item_id in &ctx.root_items {
        match shared.arena.get(item_id) {
            // External decls register a symbol name; nothing to lower.
            Node::Grammar | Node::Macro(_) | Node::External { .. } => {}
            Node::Let { value, .. } => {
                eval.eval_let(item_id, *value)?;
            }
            Node::Rule {
                is_override,
                name,
                body,
            } => {
                let rule_id = eval.lower_to_rule(*body)?;
                if *is_override {
                    override_entries.push((NameSource::Span(*name), rule_id, *name));
                } else {
                    rule_entries.push((NameSource::Span(*name), rule_id));
                }
            }
            &Node::ExpandedRule(expand_id) => {
                let exp = *shared.pools.get_expansion(expand_id);
                // Set by expand_macro_calls to the macro call's span; used for
                // override attribution diagnostics.
                let span = shared.arena.span(item_id);
                let rule_id = eval.lower_expansion(expand_id, span)?;
                if exp.is_override {
                    override_entries.push((NameSource::Str(exp.name), rule_id, span));
                } else {
                    rule_entries.push((NameSource::Str(exp.name), rule_id));
                }
            }
            _ => unreachable!(),
        }
    }
    // grammar_config is guaranteed present by `validate_grammar`, language
    // is guaranteed present by the parser (`MissingLanguageField` error).
    let config = ctx.grammar_config.as_ref().unwrap();
    let language = config.language.as_ref().unwrap();

    let rules: Vec<(String, Rule)> = rule_entries
        .into_iter()
        .map(|(src, rid)| (resolve_name(src, &eval, ctx), eval.build_rule(rid)))
        .collect();
    let overrides: Vec<(String, Rule, Span)> = override_entries
        .into_iter()
        .map(|(src, rid, span)| (resolve_name(src, &eval, ctx), eval.build_rule(rid), span))
        .collect();

    Ok(EvalResult {
        language: language.clone(),
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
        start: config.start.map(|id| eval.eval_rule_name(id)).transpose()?,
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
    result: EvalResult,
    base: Option<&InputGrammar>,
    helper_rules: Vec<(String, Rule)>,
) -> LowerResult<InputGrammar> {
    let mut overrides: FxHashMap<String, Spanned<Rule>> = FxHashMap::default();
    for (name, rule, span) in result.overrides {
        overrides.insert(name, Spanned::new(rule, span));
    }

    let mut variables = Vec::new();

    // Base rules first - preserves the inherited grammar's start rule.
    if let Some(base) = base {
        variables.reserve(base.variables.len());
        for v in &base.variables {
            if let Some(Spanned { value: rule, .. }) = overrides.remove(v.name.as_str()) {
                variables.push(Variable {
                    name: v.name.clone(),
                    kind: v.kind,
                    rule,
                });
            } else {
                variables.push(v.clone());
            }
        }
    }

    for (name, rule) in result.rules {
        variables.push(Variable {
            name,
            kind: VariableType::Named,
            rule,
        });
    }

    // Helper rules can also be override targets.
    for (name, rule) in helper_rules {
        let final_rule = overrides.remove(&name).map_or(rule, |s| s.value);
        variables.push(Variable {
            name,
            kind: VariableType::Named,
            rule: final_rule,
        });
    }

    if !overrides.is_empty() {
        let mut entries: Vec<Spanned<String>> = overrides
            .into_iter()
            .map(|(name, s)| Spanned::new(name, s.span))
            .collect();
        entries.sort_unstable_by(|a, b| a.value.cmp(&b.value));
        return Err(LowerError::without_span(
            LowerErrorKind::OverrideRuleNotFound(entries),
        ));
    }

    // Tree-sitter's start symbol is `variables[0]`, so honor `start: <rule>`
    // by rotating the named rule into position 0. The resolver + typecheck
    // guarantee the name exists somewhere in `variables`.
    if let Some(name) = result.start {
        let pos = variables.iter().position(|v| v.name == name).unwrap();
        if pos != 0 {
            let v = variables.remove(pos);
            variables.insert(0, v);
        }
    }

    fn inherit<T: Clone>(
        overridden: Option<Vec<T>>,
        base: Option<&InputGrammar>,
        field: fn(&InputGrammar) -> &[T],
    ) -> Vec<T> {
        overridden.unwrap_or_else(|| base.map_or_else(Vec::new, |b| field(b).to_vec()))
    }
    Ok(InputGrammar {
        name: result.language,
        variables,
        // Default extras matches grammar.js (dsl.js:254): `[/\s/]` applied
        // when neither the grammar nor its base specifies extras.
        extra_symbols: result.extras.unwrap_or_else(|| {
            base.map_or_else(
                || vec![Rule::Pattern("\\s".into(), String::new())],
                |b| b.extra_symbols.clone(),
            )
        }),
        expected_conflicts: inherit(result.conflicts, base, |b| &b.expected_conflicts),
        precedence_orderings: inherit(result.precedences, base, |b| &b.precedence_orderings),
        external_tokens: inherit(result.externals, base, |b| &b.external_tokens),
        variables_to_inline: inherit(result.inline, base, |b| &b.variables_to_inline),
        supertype_symbols: inherit(result.supertypes, base, |b| &b.supertype_symbols),
        word_token: result
            .word
            .or_else(|| base.and_then(|b| b.word_token.clone())),
        reserved_words: inherit(result.reserved, base, |b| &b.reserved_words),
    })
}

pub type LowerResult<T> = Result<T, super::LowerError>;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisallowedItemKind {
    OverrideRule,
    GrammarBlock,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum LowerErrorKind {
    #[error("missing grammar block")]
    MissingGrammarBlock,
    #[error("override rule(s) not found in base grammar or imported helpers: {}", format_override_names(.0))]
    OverrideRuleNotFound(Vec<Spanned<String>>),
    #[error("failed to resolve '{}': {error}", path.display())]
    ModuleResolveFailed { path: PathBuf, error: String },
    #[error("failed to read '{}': {error}", path.display())]
    ModuleReadFailed { path: PathBuf, error: String },
    #[error("too many modules (max 256)")]
    ModuleTooMany,
    #[error("module import chain too deep (max 256)")]
    ModuleDepthExceeded,
    #[error("imported files cannot contain a {}", format_disallowed(.0))]
    ModuleDisallowedItem(DisallowedItemKind),
    #[error("cycle while loading module")]
    ModuleCycle,
    #[error("expected a rule name reference")]
    ExpectedRuleName,
    #[error("config field is not set in the base grammar")]
    ConfigFieldUnset,
    #[error("inherit() requires 'inherits' to be set in the grammar config")]
    InheritWithoutConfig,
    #[error("'inherits' must reference a variable bound to inherit()")]
    InheritsWithoutInherit,
    #[error("too many elements ({0}, maximum {max})", max = u16::MAX)]
    TooManyChildren(usize),
    #[error("maximum macro call depth ({MAX_CALL_DEPTH}) exceeded")]
    CallDepthExceeded(Vec<(String, PathBuf, usize, usize)>), // name, path, line, col
    #[error("integer overflow: {0} does not fit in i32")]
    IntegerOverflow(i64),
}

fn format_override_names(entries: &[Spanned<String>]) -> String {
    entries
        .iter()
        .map(|e| e.value.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

const fn format_disallowed(kind: &DisallowedItemKind) -> &'static str {
    match kind {
        DisallowedItemKind::OverrideRule => "override rule",
        DisallowedItemKind::GrammarBlock => "grammar block",
    }
}
