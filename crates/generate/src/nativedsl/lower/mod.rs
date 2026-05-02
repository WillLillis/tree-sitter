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
use serde::Serialize;
use thiserror::Error;

use crate::{
    grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType},
    rules::Rule,
};

use super::LowerError;
use super::ast::{Node, SharedAst, Span};

pub use evaluator::LoweringState;
use evaluator::Evaluator;
use repr::RuleId;

const MAX_CALL_DEPTH: u16 = 128;

/// Intermediate results from the evaluation phase, ready to be assembled
/// into an `InputGrammar` once the base grammar can be moved out.
struct EvalResult {
    language: String,
    rules: Vec<(String, Rule)>,
    overrides: Vec<(String, Rule, Span)>,
    extras: Option<Vec<Rule>>,
    externals: Option<Vec<Rule>>,
    inline: Option<Vec<String>>,
    supertypes: Option<Vec<String>>,
    word: Option<String>,
    conflicts: Option<Vec<Vec<String>>>,
    precedences: Option<Vec<Vec<PrecedenceEntry>>>,
    reserved: Option<Vec<ReservedWordContext<Rule>>>,
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
/// `previous` are the modules already loaded; `current` is the root module
/// being lowered (not yet pushed into `previous`). `state` persists across
/// the whole `parse_native_dsl` pipeline so that imported/inherited modules'
/// let bindings evaluate exactly once.
pub fn lower_with_base(
    state: &mut LoweringState,
    shared: &SharedAst,
    previous: &[super::Module],
    current: &super::ModuleContext,
) -> LowerResult<InputGrammar> {
    let base_grammar = current
        .inherit_module(&shared.arena)
        .and_then(|(idx, _)| previous[idx as usize].lowered());
    let result = evaluate(state, shared, previous, current)?;
    build_grammar(result, base_grammar)
}

fn evaluate(
    state: &mut LoweringState,
    shared: &SharedAst,
    previous: &[super::Module],
    ctx: &super::ModuleContext,
) -> LowerResult<EvalResult> {
    let mut eval = Evaluator::new(state, shared, previous, ctx);
    let mut rule_entries: Vec<(&str, RuleId)> = Vec::with_capacity(ctx.root_items.len());
    let mut override_entries: Vec<(&str, RuleId, Span)> = Vec::new();

    for &item_id in &ctx.root_items {
        match shared.arena.get(item_id) {
            Node::Grammar | Node::Macro(_) => {}
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
                    override_entries.push((ctx.text(*name), rule_id, *name));
                } else {
                    rule_entries.push((ctx.text(*name), rule_id));
                }
            }
            _ => unreachable!(),
        }
    }
    // grammar_config is guaranteed present by validate_grammar in mod.rs,
    // language is guaranteed present by the parser (MissingLanguageField error).
    let config = ctx.grammar_config.as_ref().unwrap();
    let language = config.language.as_ref().unwrap();

    let rules: Vec<(String, Rule)> = rule_entries
        .iter()
        .map(|(name, rid)| (name.to_string(), eval.build_rule(*rid)))
        .collect();
    let overrides: Vec<(String, Rule, Span)> = override_entries
        .iter()
        .map(|(name, rid, span)| (name.to_string(), eval.build_rule(*rid), *span))
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

/// Assemble evaluated results into an `InputGrammar`, optionally merging
/// with an inherited base grammar.
fn build_grammar(result: EvalResult, base: Option<&InputGrammar>) -> LowerResult<InputGrammar> {
    // Start with inherited rules (cloned), apply overrides, then append new rules.
    let mut variables = if let Some(base) = base {
        // Build override map so we can apply them in a single pass over the
        // base variables, avoiding a full clone of every rule tree.
        let mut overrides: FxHashMap<String, (Rule, Span)> = FxHashMap::default();
        for (name, rule, span) in result.overrides {
            overrides.insert(name, (rule, span));
        }
        let mut vars = Vec::with_capacity(base.variables.len());
        for v in &base.variables {
            if let Some((rule, _)) = overrides.remove(v.name.as_str()) {
                vars.push(Variable {
                    name: v.name.clone(),
                    kind: v.kind,
                    rule,
                });
            } else {
                vars.push(v.clone());
            }
        }
        // Any remaining overrides reference rules that don't exist in the base.
        if !overrides.is_empty() {
            let entries = overrides
                .into_iter()
                .map(|(name, (_, span))| (name, span))
                .collect();
            return Err(LowerError::without_span(
                LowerErrorKind::OverrideRuleNotFound(entries),
            ));
        }
        vars
    } else if !result.overrides.is_empty() {
        return Err(LowerError::new(
            LowerErrorKind::OverrideWithoutInherit,
            result.overrides[0].2,
        ));
    } else {
        Vec::new()
    };

    for (name, rule) in result.rules {
        variables.push(Variable {
            name,
            kind: VariableType::Named,
            rule,
        });
    }

    // Derived values override base; only clone base fields that aren't overridden.
    fn inherit<T: Clone>(
        overridden: Option<Vec<T>>,
        base: Option<&InputGrammar>,
        field: fn(&InputGrammar) -> &Vec<T>,
    ) -> Vec<T> {
        overridden.unwrap_or_else(|| base.map_or_else(Vec::new, |b| field(b).clone()))
    }
    Ok(InputGrammar {
        name: result.language,
        variables,
        extra_symbols: inherit(result.extras, base, |b| &b.extra_symbols),
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

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum DisallowedItemKind {
    Rule,
    OverrideRule,
    GrammarBlock,
}

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum LowerErrorKind {
    #[error("missing grammar block")]
    MissingGrammarBlock,
    #[error("override rule(s) not found in base grammar: {}", format_override_names(.0))]
    OverrideRuleNotFound(Vec<(String, Span)>),
    #[error("'override rule' requires a grammar with 'inherits'")]
    OverrideWithoutInherit,
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
    #[error("module cycle detected")]
    ModuleCycle,
    #[error("member '{0}' not found in module")]
    ModuleMemberNotFound(String),
    #[error("expected a rule name reference")]
    ExpectedRuleName,
    #[error("config field is not set in the base grammar")]
    ConfigFieldUnset,
    #[error("inherit() requires 'inherits' to be set in the grammar config")]
    InheritWithoutConfig,
    #[error("'inherits' must reference a variable bound to inherit()")]
    InheritsWithoutInherit,
    #[error("too many elements (maximum 65535)")]
    TooManyChildren,
    #[error("maximum macro call depth ({MAX_CALL_DEPTH}) exceeded")]
    CallDepthExceeded(Vec<(String, PathBuf, usize, usize)>), // name, path, line, col
    #[error("integer overflow: {0} does not fit in i32")]
    IntegerOverflow(i64),
}

fn format_override_names(entries: &[(String, Span)]) -> String {
    entries
        .iter()
        .map(|(n, _)| n.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

const fn format_disallowed(kind: &DisallowedItemKind) -> &'static str {
    match kind {
        DisallowedItemKind::Rule => "rule",
        DisallowedItemKind::OverrideRule => "override rule",
        DisallowedItemKind::GrammarBlock => "grammar block",
    }
}
