use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::flat_rule::{FlatRule, FlatRules, RuleId, Step};
use super::{
    FlatExtractedLexicalGrammar, FlatExtractedSyntaxGrammar, FlatInternedGrammar, FlatVariable,
};
use crate::{
    grammars::{ExternalToken, ReservedWordContext, VariableType},
    rules::{MetadataParams, Symbol, SymbolType},
};

pub type ExtractTokensResult<T> = Result<T, ExtractTokensError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExtractTokensError {
    #[error(
        "The rule `{0}` contains an empty string.

Tree-sitter does not support syntactic rules that contain an empty string
unless they are used only as the grammar's start rule.
"
    )]
    EmptyString(String),
    #[error("Terminal rule '{0}' cannot be used as a supertype")]
    SupertypeTerminal(String),
    #[error("Rule '{0}' cannot be used as both an external token and a non-terminal rule")]
    ExternalTokenNonTerminal(String),
    #[error("Non-symbol rules cannot be used as external tokens")]
    NonSymbolExternalToken,
    #[error(transparent)]
    WordToken(NonTerminalWordTokenError),
    #[error("Reserved word '{0}' must be a token")]
    NonTokenReservedWord(String),
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub struct NonTerminalWordTokenError {
    pub symbol_name: String,
    pub conflicting_symbol_name: Option<String>,
}

impl std::fmt::Display for NonTerminalWordTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Non-terminal symbol '{}' cannot be used as the word token",
            self.symbol_name
        )?;
        if let Some(conflicting_name) = &self.conflicting_symbol_name {
            writeln!(
                f,
                ", because its rule is duplicated in '{conflicting_name}'",
            )
        } else {
            writeln!(f)
        }
    }
}

pub(super) fn extract_tokens(
    mut grammar: FlatInternedGrammar,
    pool: &mut FlatRules,
) -> ExtractTokensResult<(FlatExtractedSyntaxGrammar, FlatExtractedLexicalGrammar)> {
    let mut extractor = TokenExtractor {
        current_variable_name: String::new(),
        current_variable_token_count: 0,
        is_first_rule: false,
        extracted_variables: Vec::new(),
        extracted_usage_counts: Vec::new(),
    };

    for (i, variable) in &mut grammar.variables.iter_mut().enumerate() {
        extractor.extract_tokens_in_variable(pool, i == 0, variable)?;
    }

    for variable in &mut grammar.external_tokens {
        extractor.extract_tokens_in_variable(pool, false, variable)?;
    }

    let mut lexical_variables = extractor.extracted_variables;

    // If a variable's entire rule was extracted as a token and that token didn't
    // appear within any other rule, then remove that variable from the syntax
    // grammar, giving its name to the token in the lexical grammar. Any symbols
    // that pointed to that variable will need to be updated to point to the
    // variable in the lexical grammar. Symbols that pointed to later variables
    // will need to have their indices decremented.
    let mut variables = Vec::with_capacity(grammar.variables.len());
    let mut symbol_replacer = SymbolReplacer {
        replacements: FxHashMap::default(),
    };
    for (i, variable) in grammar.variables.into_iter().enumerate() {
        if let FlatRule::Symbol(Symbol {
            kind: SymbolType::Terminal,
            index,
        }) = *pool.get(variable.rule)
            && i > 0
            && extractor.extracted_usage_counts[index] == 1
        {
            let lexical_variable = &mut lexical_variables[index];
            if lexical_variable.kind == VariableType::Auxiliary
                || variable.kind != VariableType::Hidden
            {
                lexical_variable.kind = variable.kind;
                lexical_variable.name = variable.name;
                symbol_replacer.replacements.insert(i, index);
                continue;
            }
        }
        variables.push(variable);
    }

    for variable in &mut variables {
        variable.rule = symbol_replacer.replace_symbols_in_rule(pool, variable.rule);
    }

    let expected_conflicts = grammar
        .expected_conflicts
        .into_iter()
        .map(|conflict| {
            let mut result = conflict
                .iter()
                .map(|symbol| symbol_replacer.replace_symbol(*symbol))
                .collect::<Vec<_>>();
            result.sort_unstable();
            result.dedup();
            result
        })
        .collect();

    let supertype_symbols: Vec<Symbol> = grammar
        .supertype_symbols
        .into_iter()
        .map(|symbol| symbol_replacer.replace_symbol(symbol))
        .collect();
    for supertype_symbol in &supertype_symbols {
        if supertype_symbol.is_terminal() {
            Err(ExtractTokensError::SupertypeTerminal(
                lexical_variables[supertype_symbol.index].name.clone(),
            ))?;
        }
    }

    let variables_to_inline = grammar
        .variables_to_inline
        .into_iter()
        .map(|symbol| symbol_replacer.replace_symbol(symbol))
        .collect();

    let mut separators = Vec::new();
    let mut extra_symbols = Vec::new();
    for rule in grammar.extra_symbols {
        if let FlatRule::Symbol(symbol) = *pool.get(rule) {
            extra_symbols.push(symbol_replacer.replace_symbol(symbol));
        } else if let Some(index) = lexical_variables.iter().position(|v| v.rule == rule) {
            extra_symbols.push(Symbol::terminal(index));
        } else {
            separators.push(rule);
        }
    }

    let mut external_tokens = Vec::with_capacity(grammar.external_tokens.len());
    for external_token in grammar.external_tokens {
        let rule = symbol_replacer.replace_symbols_in_rule(pool, external_token.rule);
        if let FlatRule::Symbol(symbol) = *pool.get(rule) {
            if symbol.is_non_terminal() {
                Err(ExtractTokensError::ExternalTokenNonTerminal(
                    variables[symbol.index].name.clone(),
                ))?;
            }

            if symbol.is_external() {
                external_tokens.push(ExternalToken {
                    name: external_token.name,
                    kind: external_token.kind,
                    corresponding_internal_token: None,
                });
            } else {
                external_tokens.push(ExternalToken {
                    name: lexical_variables[symbol.index].name.clone(),
                    kind: external_token.kind,
                    corresponding_internal_token: Some(symbol),
                });
            }
        } else {
            Err(ExtractTokensError::NonSymbolExternalToken)?;
        }
    }

    let word_token = if let Some(token) = grammar.word_token {
        let token = symbol_replacer.replace_symbol(token);
        if token.is_non_terminal() {
            let word_token_variable = &variables[token.index];
            let conflicting_symbol_name = variables
                .iter()
                .enumerate()
                .find(|(i, v)| *i != token.index && v.rule == word_token_variable.rule)
                .map(|(_, v)| v.name.clone());

            Err(ExtractTokensError::WordToken(NonTerminalWordTokenError {
                symbol_name: word_token_variable.name.clone(),
                conflicting_symbol_name,
            }))?;
        }
        Some(token)
    } else {
        None
    };

    let mut reserved_word_contexts = Vec::with_capacity(grammar.reserved_word_sets.len());
    for reserved_word_context in grammar.reserved_word_sets {
        let mut reserved_words = Vec::with_capacity(reserved_word_contexts.len());
        for reserved_rule in reserved_word_context.reserved_words {
            if let FlatRule::Symbol(symbol) = *pool.get(reserved_rule) {
                reserved_words.push(symbol_replacer.replace_symbol(symbol));
            } else if let Some(index) = lexical_variables.iter().position(|v| v.rule == reserved_rule)
            {
                reserved_words.push(Symbol::terminal(index));
            } else {
                Err(ExtractTokensError::NonTokenReservedWord(
                    reserved_token_name(pool, reserved_rule),
                ))?;
            }
        }
        reserved_word_contexts.push(ReservedWordContext {
            name: reserved_word_context.name,
            reserved_words,
        });
    }

    Ok((
        FlatExtractedSyntaxGrammar {
            variables,
            expected_conflicts,
            extra_symbols,
            variables_to_inline,
            supertype_symbols,
            external_tokens,
            word_token,
            precedence_orderings: grammar.precedence_orderings,
            reserved_word_sets: reserved_word_contexts,
        },
        FlatExtractedLexicalGrammar {
            variables: lexical_variables,
            separators,
        },
    ))
}

/// The token name reported in a [`NonTokenReservedWord`](ExtractTokensError) error:
/// the underlying string/pattern (unwrapping a metadata wrapper), else `unknown`.
fn reserved_token_name(pool: &FlatRules, id: RuleId) -> String {
    let inner = if let FlatRule::Metadata { rule, .. } = pool.get(id) {
        *rule
    } else {
        id
    };
    match pool.get(inner) {
        FlatRule::String(s) => s.clone(),
        FlatRule::Pattern(p, _) => p.clone(),
        _ => "unknown".to_string(),
    }
}

struct TokenExtractor {
    current_variable_name: String,
    current_variable_token_count: usize,
    is_first_rule: bool,
    extracted_variables: Vec<FlatVariable>,
    extracted_usage_counts: Vec<usize>,
}

struct SymbolReplacer {
    replacements: FxHashMap<usize, usize>,
}

impl TokenExtractor {
    fn extract_tokens_in_variable(
        &mut self,
        pool: &mut FlatRules,
        is_first: bool,
        variable: &mut FlatVariable,
    ) -> ExtractTokensResult<()> {
        self.current_variable_name.clear();
        self.current_variable_name.push_str(&variable.name);
        self.current_variable_token_count = 0;
        self.is_first_rule = is_first;
        variable.rule = self.extract_tokens_in_rule(pool, variable.rule)?;
        Ok(())
    }

    /// Replace each extracted token (a string, pattern, or `token(...)`) with a
    /// `Symbol(terminal)` and descend through the rest. Iterative post-order, so a
    /// deeply nested rule cannot overflow the native stack.
    fn extract_tokens_in_rule(
        &mut self,
        pool: &mut FlatRules,
        root: RuleId,
    ) -> ExtractTokensResult<RuleId> {
        // What to do with a visited node: extract it as a token (with the rule to
        // store and the token's string value, if any), descend, or keep as-is.
        enum Visit {
            Token(RuleId, Option<String>),
            Descend,
            Reuse,
        }
        let mut work = vec![Step::Visit(root)];
        let mut out: Vec<RuleId> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(id) => {
                    let action = match pool.get(id) {
                        FlatRule::String(value) => Visit::Token(id, Some(value.clone())),
                        FlatRule::Pattern(..) => Visit::Token(id, None),
                        FlatRule::Metadata { params, rule } if params.is_token => {
                            // `token(...)`: extract the whole subtree as one token.
                            // If the only metadata was `is_token`, store the inner
                            // rule; otherwise store the metadata-wrapped rule.
                            let mut bare = params.clone();
                            bare.is_token = false;
                            let inner = *rule;
                            let string_value = match pool.get(inner) {
                                FlatRule::String(value) => Some(value.clone()),
                                _ => None,
                            };
                            let to_store = if bare == MetadataParams::default() {
                                inner
                            } else {
                                id
                            };
                            Visit::Token(to_store, string_value)
                        }
                        FlatRule::Metadata { .. }
                        | FlatRule::Repeat(_)
                        | FlatRule::Seq(_)
                        | FlatRule::Choice(_)
                        | FlatRule::Reserved { .. } => Visit::Descend,
                        FlatRule::Blank | FlatRule::Symbol(_) | FlatRule::NamedSymbol(_) => {
                            Visit::Reuse
                        }
                    };
                    match action {
                        Visit::Token(to_store, string_value) => {
                            let symbol = self.extract_token(to_store, string_value.as_deref())?;
                            out.push(pool.intern_symbol(symbol));
                        }
                        Visit::Reuse => out.push(id),
                        Visit::Descend => pool.descend(id, &mut work),
                    }
                }
                Step::Build(id) => pool.rebuild(id, &mut out),
            }
        }
        Ok(out.pop().unwrap())
    }

    fn extract_token(
        &mut self,
        rule: RuleId,
        string_value: Option<&str>,
    ) -> ExtractTokensResult<Symbol> {
        // Dedup by id: the pool is hash-consed, so structurally identical token
        // rules share one RuleId.
        for (i, variable) in self.extracted_variables.iter_mut().enumerate() {
            if variable.rule == rule {
                self.extracted_usage_counts[i] += 1;
                return Ok(Symbol::terminal(i));
            }
        }

        let index = self.extracted_variables.len();
        let variable = if let Some(string_value) = string_value {
            if string_value.is_empty() && !self.is_first_rule {
                Err(ExtractTokensError::EmptyString(
                    self.current_variable_name.clone(),
                ))?;
            }
            FlatVariable {
                name: string_value.to_owned(),
                kind: VariableType::Anonymous,
                rule,
            }
        } else {
            self.current_variable_token_count += 1;
            FlatVariable {
                name: format!(
                    "{}_token{}",
                    self.current_variable_name, self.current_variable_token_count
                ),
                kind: VariableType::Auxiliary,
                rule,
            }
        };

        self.extracted_variables.push(variable);
        self.extracted_usage_counts.push(1);
        Ok(Symbol::terminal(index))
    }
}

impl SymbolReplacer {
    /// Rewrite every `Symbol` in the rule via [`replace_symbol`](Self::replace_symbol),
    /// re-interning the structure. Iterative post-order.
    fn replace_symbols_in_rule(&self, pool: &mut FlatRules, root: RuleId) -> RuleId {
        let mut work = vec![Step::Visit(root)];
        let mut out: Vec<RuleId> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(id) => match *pool.get(id) {
                    FlatRule::Symbol(symbol) => {
                        let replaced = self.replace_symbol(symbol);
                        out.push(pool.intern_symbol(replaced));
                    }
                    FlatRule::Metadata { .. }
                    | FlatRule::Repeat(_)
                    | FlatRule::Seq(_)
                    | FlatRule::Choice(_)
                    | FlatRule::Reserved { .. } => pool.descend(id, &mut work),
                    FlatRule::Blank
                    | FlatRule::String(_)
                    | FlatRule::Pattern(..)
                    | FlatRule::NamedSymbol(_) => out.push(id),
                },
                Step::Build(id) => pool.rebuild(id, &mut out),
            }
        }
        out.pop().unwrap()
    }

    fn replace_symbol(&self, symbol: Symbol) -> Symbol {
        if !symbol.is_non_terminal() {
            return symbol;
        }

        if let Some(replacement) = self.replacements.get(&symbol.index) {
            return Symbol::terminal(*replacement);
        }

        let mut adjusted_index = symbol.index;
        for replaced_index in self.replacements.keys() {
            if *replaced_index < symbol.index {
                adjusted_index -= 1;
            }
        }

        Symbol::non_terminal(adjusted_index)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use super::super::{
        InternedGrammar, materialize_extracted_lexical, materialize_extracted_syntax,
    };
    use crate::grammars::Variable;
    use crate::rules::Rule;

    /// Intern an `InternedGrammar` into a fresh pool, run `extract_tokens`, and
    /// materialize the outputs, so the assertions can compare against `Rule` trees.
    fn extract(
        grammar: InternedGrammar,
    ) -> ExtractTokensResult<(
        crate::prepare_grammar::ExtractedSyntaxGrammar,
        crate::prepare_grammar::ExtractedLexicalGrammar,
    )> {
        let mut pool = FlatRules::default();
        let flat = FlatInternedGrammar {
            variables: import_variables(grammar.variables, &mut pool),
            extra_symbols: grammar
                .extra_symbols
                .iter()
                .map(|r| pool.intern_import(r))
                .collect(),
            expected_conflicts: grammar.expected_conflicts,
            precedence_orderings: grammar.precedence_orderings,
            external_tokens: import_variables(grammar.external_tokens, &mut pool),
            variables_to_inline: grammar.variables_to_inline,
            supertype_symbols: grammar.supertype_symbols,
            word_token: grammar.word_token,
            reserved_word_sets: grammar
                .reserved_word_sets
                .into_iter()
                .map(|s| ReservedWordContext {
                    name: s.name,
                    reserved_words: s.reserved_words.iter().map(|r| pool.intern_import(r)).collect(),
                })
                .collect(),
        };
        let (syntax, lexical) = extract_tokens(flat, &mut pool)?;
        Ok((
            materialize_extracted_syntax(syntax, &pool),
            materialize_extracted_lexical(lexical, &pool),
        ))
    }

    fn import_variables(variables: Vec<Variable>, pool: &mut FlatRules) -> Vec<FlatVariable> {
        variables
            .into_iter()
            .map(|v| FlatVariable {
                name: v.name,
                kind: v.kind,
                rule: pool.intern_import(&v.rule),
            })
            .collect()
    }

    #[test]
    fn test_extraction() {
        let (syntax_grammar, lexical_grammar) = extract(build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::repeat(Rule::seq(vec![
                    Rule::string("a"),
                    Rule::pattern("b", ""),
                    Rule::choice(vec![
                        Rule::non_terminal(1),
                        Rule::non_terminal(2),
                        Rule::token(Rule::repeat(Rule::choice(vec![
                            Rule::string("c"),
                            Rule::string("d"),
                        ]))),
                    ]),
                ])),
            ),
            Variable::named("rule_1", Rule::pattern("e", "")),
            Variable::named("rule_2", Rule::pattern("b", "")),
            Variable::named(
                "rule_3",
                Rule::seq(vec![Rule::non_terminal(2), Rule::Blank]),
            ),
        ]))
        .unwrap();

        assert_eq!(
            syntax_grammar.variables,
            vec![
                Variable::named(
                    "rule_0",
                    Rule::repeat(Rule::seq(vec![
                        // The string "a" was replaced by a symbol referencing the lexical grammar
                        Rule::terminal(0),
                        // The pattern "b" was replaced by a symbol referencing the lexical grammar
                        Rule::terminal(1),
                        Rule::choice(vec![
                            // The symbol referencing `rule_1` was replaced by a symbol referencing
                            // the lexical grammar.
                            Rule::terminal(3),
                            // The symbol referencing `rule_2` had its index decremented because
                            // `rule_1` was moved to the lexical grammar.
                            Rule::non_terminal(1),
                            // The rule wrapped in `token` was replaced by a symbol referencing
                            // the lexical grammar.
                            Rule::terminal(2),
                        ])
                    ]))
                ),
                // The pattern "e" was only used in one place: as the definition of `rule_1`,
                // so that rule was moved to the lexical grammar. The pattern "b" appeared in
                // two places, so it was not moved into the lexical grammar.
                Variable::named("rule_2", Rule::terminal(1)),
                Variable::named(
                    "rule_3",
                    Rule::seq(vec![Rule::non_terminal(1), Rule::Blank,])
                ),
            ]
        );

        assert_eq!(
            lexical_grammar.variables,
            vec![
                Variable::anonymous("a", Rule::string("a")),
                Variable::auxiliary("rule_0_token1", Rule::pattern("b", "")),
                Variable::auxiliary(
                    "rule_0_token2",
                    Rule::repeat(Rule::choice(vec![Rule::string("c"), Rule::string("d"),]))
                ),
                Variable::named("rule_1", Rule::pattern("e", "")),
            ]
        );
    }

    #[test]
    fn test_start_rule_is_token() {
        let (syntax_grammar, lexical_grammar) =
            extract(build_grammar(vec![Variable::named(
                "rule_0",
                Rule::string("hello"),
            )]))
            .unwrap();

        assert_eq!(
            syntax_grammar.variables,
            vec![Variable::named("rule_0", Rule::terminal(0)),]
        );
        assert_eq!(
            lexical_grammar.variables,
            vec![Variable::anonymous("hello", Rule::string("hello")),]
        );
    }

    #[test]
    fn test_extracting_extra_symbols() {
        let mut grammar = build_grammar(vec![
            Variable::named("rule_0", Rule::string("x")),
            Variable::named("comment", Rule::pattern("//.*", "")),
        ]);
        grammar.extra_symbols = vec![Rule::string(" "), Rule::non_terminal(1)];

        let (syntax_grammar, lexical_grammar) = extract(grammar).unwrap();
        assert_eq!(syntax_grammar.extra_symbols, vec![Symbol::terminal(1),]);
        assert_eq!(lexical_grammar.separators, vec![Rule::string(" "),]);
    }

    #[test]
    fn test_extract_externals() {
        let mut grammar = build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::seq(vec![
                    Rule::external(0),
                    Rule::string("a"),
                    Rule::non_terminal(1),
                    Rule::non_terminal(2),
                ]),
            ),
            Variable::named("rule_1", Rule::string("b")),
            Variable::named("rule_2", Rule::string("c")),
        ]);
        grammar.external_tokens = vec![
            Variable::named("external_0", Rule::external(0)),
            Variable::anonymous("a", Rule::string("a")),
            Variable::named("rule_2", Rule::non_terminal(2)),
        ];

        let (syntax_grammar, _) = extract(grammar).unwrap();

        assert_eq!(
            syntax_grammar.external_tokens,
            vec![
                ExternalToken {
                    name: "external_0".to_string(),
                    kind: VariableType::Named,
                    corresponding_internal_token: None,
                },
                ExternalToken {
                    name: "a".to_string(),
                    kind: VariableType::Anonymous,
                    corresponding_internal_token: Some(Symbol::terminal(0)),
                },
                ExternalToken {
                    name: "rule_2".to_string(),
                    kind: VariableType::Named,
                    corresponding_internal_token: Some(Symbol::terminal(2)),
                },
            ]
        );
    }

    #[test]
    fn test_error_on_external_with_same_name_as_non_terminal() {
        let mut grammar = build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
            ),
            Variable::named(
                "rule_1",
                Rule::seq(vec![Rule::non_terminal(2), Rule::non_terminal(2)]),
            ),
            Variable::named("rule_2", Rule::string("a")),
        ]);
        grammar.external_tokens = vec![Variable::named("rule_1", Rule::non_terminal(1))];

        let result = extract(grammar);
        assert!(result.is_err(), "Expected an error but got no error");
        let err = result.err().unwrap();
        assert_eq!(
            err.to_string(),
            "Rule 'rule_1' cannot be used as both an external token and a non-terminal rule"
        );
    }

    #[test]
    fn test_extraction_on_hidden_terminal() {
        let (syntax_grammar, lexical_grammar) = extract(build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::hidden("_rule_1", Rule::string("a")),
        ]))
        .unwrap();

        // The rule `_rule_1` should not "absorb" the
        // terminal "a", since it is hidden,
        // so we expect two variables still
        assert_eq!(
            syntax_grammar.variables,
            vec![
                Variable::named("rule_0", Rule::non_terminal(1)),
                Variable::hidden("_rule_1", Rule::terminal(0)),
            ]
        );

        // We should not have a hidden rule in our lexical grammar, only the terminal "a"
        assert_eq!(
            lexical_grammar.variables,
            vec![Variable::anonymous("a", Rule::string("a"))]
        );
    }

    #[test]
    fn test_extraction_with_empty_string() {
        assert!(
            extract(build_grammar(vec![
                Variable::named("rule_0", Rule::non_terminal(1)),
                Variable::hidden("_rule_1", Rule::string("")),
            ]))
            .is_err()
        );
    }

    fn build_grammar(variables: Vec<Variable>) -> InternedGrammar {
        InternedGrammar {
            variables,
            ..Default::default()
        }
    }
}
