use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{FlatInternedGrammar, FlatVariable};
use super::flat_rule::{FlatRules, RuleId};
use crate::{
    Diagnostic,
    grammars::{InputGrammar, ReservedWordContext, VariableType},
    rules::{Rule, Symbol},
};

pub type InternSymbolsResult<T> = Result<T, InternSymbolsError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum InternSymbolsError {
    #[error("A grammar's start rule must be visible.")]
    HiddenStartRule,
    #[error("Undefined symbol `{0}`")]
    Undefined(String),
    #[error("Undefined symbol `{0}` in grammar's supertypes array")]
    UndefinedSupertype(String),
    #[error("Undefined symbol `{0}` in grammar's conflicts array")]
    UndefinedConflict(String),
    #[error("Undefined symbol `{0}` as grammar's word token")]
    UndefinedWordToken(String),
}

pub(super) fn intern_symbols(
    grammar: &InputGrammar,
    pool: &mut FlatRules,
    diagnostics: &mut Vec<Diagnostic>,
) -> InternSymbolsResult<FlatInternedGrammar> {
    let interner = Interner { grammar };

    if variable_type_for_name(&grammar.variables[0].name) == VariableType::Hidden {
        Err(InternSymbolsError::HiddenStartRule)?;
    }

    let mut variables = Vec::with_capacity(grammar.variables.len());
    for variable in &grammar.variables {
        variables.push(FlatVariable {
            name: variable.name.clone(),
            kind: variable_type_for_name(&variable.name),
            rule: interner.intern_rule(pool, &variable.rule, Some(&variable.name), diagnostics)?,
        });
    }

    let mut external_tokens = Vec::with_capacity(grammar.external_tokens.len());
    for external_token in &grammar.external_tokens {
        let rule = interner.intern_rule(pool, external_token, None, diagnostics)?;
        let (name, kind) = if let Rule::NamedSymbol(name) = external_token {
            (name.clone(), variable_type_for_name(name))
        } else {
            (String::new(), VariableType::Anonymous)
        };
        external_tokens.push(FlatVariable { name, kind, rule });
    }

    let mut extra_symbols = Vec::with_capacity(grammar.extra_symbols.len());
    for extra_token in &grammar.extra_symbols {
        extra_symbols.push(interner.intern_rule(pool, extra_token, None, diagnostics)?);
    }

    let mut supertype_symbols = Vec::with_capacity(grammar.supertype_symbols.len());
    for supertype_symbol_name in &grammar.supertype_symbols {
        supertype_symbols.push(interner.intern_name(supertype_symbol_name).ok_or_else(|| {
            InternSymbolsError::UndefinedSupertype(supertype_symbol_name.clone())
        })?);
    }

    let mut reserved_words = Vec::with_capacity(grammar.reserved_words.len());
    for reserved_word_set in &grammar.reserved_words {
        let mut interned_set = Vec::with_capacity(reserved_word_set.reserved_words.len());
        for rule in &reserved_word_set.reserved_words {
            interned_set.push(interner.intern_rule(pool, rule, None, diagnostics)?);
        }
        reserved_words.push(ReservedWordContext {
            name: reserved_word_set.name.clone(),
            reserved_words: interned_set,
        });
    }

    let mut expected_conflicts = Vec::with_capacity(grammar.expected_conflicts.len());
    for conflict in &grammar.expected_conflicts {
        let mut interned_conflict = Vec::with_capacity(conflict.len());
        for name in conflict {
            interned_conflict.push(
                interner
                    .intern_name(name)
                    .ok_or_else(|| InternSymbolsError::UndefinedConflict(name.clone()))?,
            );
        }
        expected_conflicts.push(interned_conflict);
    }

    let mut variables_to_inline = Vec::new();
    for name in &grammar.variables_to_inline {
        if let Some(symbol) = interner.intern_name(name) {
            variables_to_inline.push(symbol);
        }
    }

    let word_token = if let Some(name) = grammar.word_token.as_ref() {
        Some(
            interner
                .intern_name(name)
                .ok_or_else(|| InternSymbolsError::UndefinedWordToken(name.clone()))?,
        )
    } else {
        None
    };

    for (i, variable) in variables.iter_mut().enumerate() {
        if supertype_symbols.contains(&Symbol::non_terminal(i)) {
            variable.kind = VariableType::Hidden;
        }
    }

    Ok(FlatInternedGrammar {
        variables,
        external_tokens,
        extra_symbols,
        expected_conflicts,
        variables_to_inline,
        supertype_symbols,
        word_token,
        precedence_orderings: grammar.precedence_orderings.clone(),
        reserved_word_sets: reserved_words,
    })
}

struct Interner<'a> {
    grammar: &'a InputGrammar,
}

impl Interner<'_> {
    /// Resolve every `NamedSymbol` in `rule` to a `Symbol`, returning the rule's
    /// `RuleId` in the shared, hash-consed `pool`. An explicit-stack post-order
    /// walk interns each node as it is built (so deep nesting cannot overflow the
    /// native stack); `Seq`/`Choice` emit unary diagnostics on entry, and an
    /// undefined `NamedSymbol` errors in source order - matching the recursive
    /// interner. Materialization back to a `Rule` is deferred to the caller.
    fn intern_rule(
        &self,
        pool: &mut FlatRules,
        rule: &Rule,
        name: Option<&str>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> InternSymbolsResult<RuleId> {
        enum Step<'r> {
            Visit(&'r Rule),
            Build(&'r Rule),
        }
        let mut work = vec![Step::Visit(rule)];
        let mut out: Vec<RuleId> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(rule) => match rule {
                    Rule::Blank => out.push(pool.intern_blank()),
                    Rule::String(s) => out.push(pool.intern_string(s)),
                    Rule::Pattern(p, f) => out.push(pool.intern_pattern(p, f)),
                    Rule::NamedSymbol(symbol_name) => {
                        let symbol = self.intern_name(symbol_name).ok_or_else(|| {
                            InternSymbolsError::Undefined(symbol_name.clone())
                        })?;
                        out.push(pool.intern_symbol(symbol));
                    }
                    Rule::Symbol(symbol) => out.push(pool.intern_symbol(*symbol)),
                    Rule::Choice(elements) | Rule::Seq(elements) => {
                        Self::check_single(
                            elements,
                            name,
                            matches!(rule, Rule::Choice(_)),
                            diagnostics,
                        );
                        work.push(Step::Build(rule));
                        for element in elements.iter().rev() {
                            work.push(Step::Visit(element));
                        }
                    }
                    Rule::Repeat(inner)
                    | Rule::Metadata { rule: inner, .. }
                    | Rule::Reserved { rule: inner, .. } => {
                        work.push(Step::Build(rule));
                        work.push(Step::Visit(inner));
                    }
                },
                Step::Build(rule) => match rule {
                    Rule::Choice(elements) | Rule::Seq(elements) => {
                        let base = out.len() - elements.len();
                        let children = out.split_off(base);
                        out.push(pool.intern_seq_or_choice(matches!(rule, Rule::Seq(_)), &children));
                    }
                    Rule::Repeat(_) => {
                        let inner = out.pop().unwrap();
                        out.push(pool.intern_repeat(inner));
                    }
                    Rule::Metadata { params, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(pool.intern_metadata(params.clone(), inner));
                    }
                    Rule::Reserved { context_name, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(pool.intern_reserved(context_name, inner));
                    }
                    // Leaves never enqueue a Build step.
                    _ => unreachable!(),
                },
            }
        }
        Ok(out.pop().unwrap())
    }

    fn intern_name(&self, symbol: &str) -> Option<Symbol> {
        for (i, variable) in self.grammar.variables.iter().enumerate() {
            if variable.name == symbol {
                return Some(Symbol::non_terminal(i));
            }
        }

        for (i, external_token) in self.grammar.external_tokens.iter().enumerate() {
            if let Rule::NamedSymbol(name) = external_token
                && name == symbol
            {
                return Some(Symbol::external(i));
            }
        }

        None
    }

    // In the case of a seq or choice rule of 1 element in a hidden rule, weird
    // inconsistent behavior with queries can occur. So we should warn the user about it.
    fn check_single(
        elements: &[Rule],
        name: Option<&str>,
        is_choice: bool,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        if elements.len() == 1 && matches!(elements[0], Rule::String(_) | Rule::Pattern(..)) {
            let name = name.map(str::to_string);
            diagnostics.push(if is_choice {
                Diagnostic::UnaryChoice { name }
            } else {
                Diagnostic::UnarySeq { name }
            });
        }
    }
}

fn variable_type_for_name(name: &str) -> VariableType {
    if name.starts_with('_') {
        VariableType::Hidden
    } else {
        VariableType::Named
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{InternedGrammar, materialize_interned};
    use crate::grammars::Variable;

    /// Run `intern_symbols` on a throwaway pool and materialize the result, so
    /// the assertions below can keep comparing against `Rule` trees.
    fn intern(grammar: &InputGrammar) -> InternSymbolsResult<InternedGrammar> {
        let mut pool = FlatRules::default();
        Ok(materialize_interned(
            intern_symbols(grammar, &mut pool, &mut Vec::new())?,
            &pool,
        ))
    }

    #[test]
    fn test_basic_repeat_expansion() {
        let grammar = intern(&build_grammar(vec![
            Variable::named("x", Rule::choice(vec![Rule::named("y"), Rule::named("_z")])),
            Variable::named("y", Rule::named("_z")),
            Variable::named("_z", Rule::string("a")),
        ]))
        .unwrap();

        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "x",
                    Rule::choice(vec![Rule::non_terminal(1), Rule::non_terminal(2),])
                ),
                Variable::named("y", Rule::non_terminal(2)),
                Variable::hidden("_z", Rule::string("a")),
            ]
        );
    }

    #[test]
    fn test_interning_external_token_names() {
        // Variable `y` is both an internal and an external token.
        // Variable `z` is just an external token.
        let mut input_grammar = build_grammar(vec![
            Variable::named(
                "w",
                Rule::choice(vec![Rule::named("x"), Rule::named("y"), Rule::named("z")]),
            ),
            Variable::named("x", Rule::string("a")),
            Variable::named("y", Rule::string("b")),
        ]);
        input_grammar
            .external_tokens
            .extend(vec![Rule::named("y"), Rule::named("z")]);

        let grammar = intern(&input_grammar).unwrap();

        // Variable `y` is referred to by its internal index.
        // Variable `z` is referred to by its external index.
        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "w",
                    Rule::choice(vec![
                        Rule::non_terminal(1),
                        Rule::non_terminal(2),
                        Rule::external(1),
                    ])
                ),
                Variable::named("x", Rule::string("a")),
                Variable::named("y", Rule::string("b")),
            ]
        );

        // The external token for `y` refers back to its internal index.
        assert_eq!(
            grammar.external_tokens,
            vec![
                Variable::named("y", Rule::non_terminal(2)),
                Variable::named("z", Rule::external(1)),
            ]
        );
    }

    #[test]
    fn test_grammar_with_undefined_symbols() {
        let result = intern(&build_grammar(vec![Variable::named("x", Rule::named("y"))]));

        assert!(result.is_err(), "Expected an error but got none");
        let e = result.err().unwrap();
        assert_eq!(e.to_string(), "Undefined symbol `y`");
    }

    fn build_grammar(variables: Vec<Variable>) -> InputGrammar {
        InputGrammar {
            variables,
            name: "the_language".to_string(),
            ..Default::default()
        }
    }
}
