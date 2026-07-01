use rustc_hash::FxHashMap;

use super::{FlatExtractedSyntaxGrammar, FlatVariable};
use super::flat_rule::{FlatRule, FlatRules, RuleId, Step};
use crate::{grammars::VariableType, rules::Symbol};

struct Expander<'a> {
    variable_name: String,
    repeat_count_in_variable: usize,
    preceding_symbol_count: usize,
    auxiliary_variables: Vec<FlatVariable>,
    pool: &'a mut FlatRules,
    /// An expanded repeat's content (its canonical `RuleId` in the hash-consed
    /// pool) keys the auxiliary symbol, so structurally identical repeated content
    /// anywhere in the grammar reuses one auxiliary rule.
    existing_repeats: FxHashMap<RuleId, Symbol>,
}

impl Expander<'_> {
    fn expand_variable(&mut self, index: usize, variable: &mut FlatVariable) -> bool {
        self.variable_name.clear();
        self.variable_name.push_str(&variable.name);
        self.repeat_count_in_variable = 0;

        // Special case: a hidden variable that is itself a top-level repetition is
        // turned into a binary tree in place, rather than introducing a separate
        // auxiliary rule.
        let top_repeat = match self.pool.get(variable.rule) {
            FlatRule::Repeat(content) => Some(*content),
            _ => None,
        };
        if variable.kind == VariableType::Hidden
            && let Some(content) = top_repeat
        {
            let inner = self.expand_rule(content);
            variable.rule = self.wrap_rule_in_binary_tree(Symbol::non_terminal(index), inner);
            variable.kind = VariableType::Auxiliary;
            return true;
        }

        variable.rule = self.expand_rule(variable.rule);
        false
    }

    /// Replace each `Repeat` with a reference to an auxiliary variable holding a
    /// recursive binary tree of the repeated content. Iterative post-order: a
    /// `Repeat`'s `Build` step runs the expansion on its already-expanded content.
    fn expand_rule(&mut self, root: RuleId) -> RuleId {
        let mut work = vec![Step::Visit(root)];
        let mut out: Vec<RuleId> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(id) => match self.pool.get(id) {
                    FlatRule::Choice(_)
                    | FlatRule::Seq(_)
                    | FlatRule::Metadata { .. }
                    | FlatRule::Repeat(_) => self.pool.descend(id, &mut work),
                    // Primitives are unchanged. `Reserved` is copied wholesale (its
                    // repeats are not expanded), matching the previous interner.
                    FlatRule::Blank
                    | FlatRule::String(_)
                    | FlatRule::Pattern(..)
                    | FlatRule::Symbol(_)
                    | FlatRule::NamedSymbol(_)
                    | FlatRule::Reserved { .. } => out.push(id),
                },
                Step::Build(id) => {
                    if matches!(self.pool.get(id), FlatRule::Repeat(_)) {
                        let inner = out.pop().unwrap();
                        let symbol = self.expand_repeat(inner);
                        out.push(symbol);
                    } else {
                        self.pool.rebuild(id, &mut out);
                    }
                }
            }
        }
        out.pop().unwrap()
    }

    /// Map an expanded repeat's `content` to a reference to its auxiliary variable,
    /// creating that variable (and its binary tree) on first sight.
    fn expand_repeat(&mut self, content: RuleId) -> RuleId {
        if let Some(existing_symbol) = self.existing_repeats.get(&content) {
            return self.pool.intern_symbol(*existing_symbol);
        }

        self.repeat_count_in_variable += 1;
        let rule_name = format!(
            "{}_repeat{}",
            self.variable_name, self.repeat_count_in_variable
        );
        let repeat_symbol =
            Symbol::non_terminal(self.preceding_symbol_count + self.auxiliary_variables.len());
        self.existing_repeats.insert(content, repeat_symbol);
        let rule = self.wrap_rule_in_binary_tree(repeat_symbol, content);
        self.auxiliary_variables.push(FlatVariable {
            name: rule_name,
            kind: VariableType::Auxiliary,
            rule,
        });

        self.pool.intern_symbol(repeat_symbol)
    }

    fn wrap_rule_in_binary_tree(&mut self, symbol: Symbol, content: RuleId) -> RuleId {
        let symbol = self.pool.intern_symbol(symbol);
        let pair = self.pool.intern_seq_or_choice(true, &[symbol, symbol]);
        self.pool.intern_choice_flattened(&[pair, content])
    }
}

pub(super) fn expand_repeats(
    mut grammar: FlatExtractedSyntaxGrammar,
    pool: &mut FlatRules,
) -> FlatExtractedSyntaxGrammar {
    let mut expander = Expander {
        variable_name: String::new(),
        repeat_count_in_variable: 0,
        preceding_symbol_count: grammar.variables.len(),
        auxiliary_variables: Vec::new(),
        pool,
        existing_repeats: FxHashMap::default(),
    };

    for (i, variable) in grammar.variables.iter_mut().enumerate() {
        let expanded_top_level_repetition = expander.expand_variable(i, variable);

        // If a hidden variable had a top-level repetition and it was converted to
        // a recursive rule, then it can't be inlined.
        if expanded_top_level_repetition {
            grammar
                .variables_to_inline
                .retain(|symbol| *symbol != Symbol::non_terminal(i));
        }
    }

    grammar.variables.extend(expander.auxiliary_variables);
    grammar
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::{ExtractedSyntaxGrammar, materialize_extracted_syntax};
    use crate::grammars::Variable;
    use crate::rules::Rule;

    /// Intern an `ExtractedSyntaxGrammar` into a fresh pool, run `expand_repeats`,
    /// and materialize the result, so the assertions can compare `Rule` trees.
    fn expand(grammar: ExtractedSyntaxGrammar) -> ExtractedSyntaxGrammar {
        let mut pool = FlatRules::default();
        let flat = FlatExtractedSyntaxGrammar {
            variables: grammar
                .variables
                .into_iter()
                .map(|v| FlatVariable {
                    name: v.name,
                    kind: v.kind,
                    rule: pool.intern_import(&v.rule),
                })
                .collect(),
            extra_symbols: grammar.extra_symbols,
            expected_conflicts: grammar.expected_conflicts,
            precedence_orderings: grammar.precedence_orderings,
            external_tokens: grammar.external_tokens,
            variables_to_inline: grammar.variables_to_inline,
            supertype_symbols: grammar.supertype_symbols,
            word_token: grammar.word_token,
            reserved_word_sets: grammar.reserved_word_sets,
        };
        let result = expand_repeats(flat, &mut pool);
        materialize_extracted_syntax(result, &pool)
    }

    #[test]
    fn test_basic_repeat_expansion() {
        // Repeats nested inside of sequences and choices are expanded.
        let grammar = expand(build_grammar(vec![Variable::named(
            "rule0",
            Rule::seq(vec![
                Rule::terminal(10),
                Rule::choice(vec![
                    Rule::repeat(Rule::terminal(11)),
                    Rule::repeat(Rule::terminal(12)),
                ]),
                Rule::terminal(13),
            ]),
        )]));

        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "rule0",
                    Rule::seq(vec![
                        Rule::terminal(10),
                        Rule::choice(vec![Rule::non_terminal(1), Rule::non_terminal(2),]),
                        Rule::terminal(13),
                    ])
                ),
                Variable::auxiliary(
                    "rule0_repeat1",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(1),]),
                        Rule::terminal(11),
                    ])
                ),
                Variable::auxiliary(
                    "rule0_repeat2",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::non_terminal(2), Rule::non_terminal(2),]),
                        Rule::terminal(12),
                    ])
                ),
            ]
        );
    }

    #[test]
    fn test_repeat_deduplication() {
        // Terminal 4 appears inside of a repeat in three different places.
        let grammar = expand(build_grammar(vec![
            Variable::named(
                "rule0",
                Rule::choice(vec![
                    Rule::seq(vec![Rule::terminal(1), Rule::repeat(Rule::terminal(4))]),
                    Rule::seq(vec![Rule::terminal(2), Rule::repeat(Rule::terminal(4))]),
                ]),
            ),
            Variable::named(
                "rule1",
                Rule::seq(vec![Rule::terminal(3), Rule::repeat(Rule::terminal(4))]),
            ),
        ]));

        // Only one auxiliary rule is created for repeating terminal 4.
        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "rule0",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::terminal(1), Rule::non_terminal(2)]),
                        Rule::seq(vec![Rule::terminal(2), Rule::non_terminal(2)]),
                    ])
                ),
                Variable::named(
                    "rule1",
                    Rule::seq(vec![Rule::terminal(3), Rule::non_terminal(2),])
                ),
                Variable::auxiliary(
                    "rule0_repeat1",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::non_terminal(2), Rule::non_terminal(2),]),
                        Rule::terminal(4),
                    ])
                )
            ]
        );
    }

    #[test]
    fn test_expansion_of_nested_repeats() {
        let grammar = expand(build_grammar(vec![Variable::named(
            "rule0",
            Rule::seq(vec![
                Rule::terminal(10),
                Rule::repeat(Rule::seq(vec![
                    Rule::terminal(11),
                    Rule::repeat(Rule::terminal(12)),
                ])),
            ]),
        )]));

        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "rule0",
                    Rule::seq(vec![Rule::terminal(10), Rule::non_terminal(2),])
                ),
                Variable::auxiliary(
                    "rule0_repeat1",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(1),]),
                        Rule::terminal(12),
                    ])
                ),
                Variable::auxiliary(
                    "rule0_repeat2",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::non_terminal(2), Rule::non_terminal(2),]),
                        Rule::seq(vec![Rule::terminal(11), Rule::non_terminal(1),]),
                    ])
                ),
            ]
        );
    }

    #[test]
    fn test_expansion_of_repeats_at_top_of_hidden_rules() {
        let grammar = expand(build_grammar(vec![
            Variable::named("rule0", Rule::non_terminal(1)),
            Variable::hidden(
                "_rule1",
                Rule::repeat(Rule::choice(vec![Rule::terminal(11), Rule::terminal(12)])),
            ),
        ]));

        assert_eq!(
            grammar.variables,
            vec![
                Variable::named("rule0", Rule::non_terminal(1),),
                Variable::auxiliary(
                    "_rule1",
                    Rule::choice(vec![
                        Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(1)]),
                        Rule::terminal(11),
                        Rule::terminal(12),
                    ]),
                ),
            ]
        );
    }

    fn build_grammar(variables: Vec<Variable>) -> ExtractedSyntaxGrammar {
        ExtractedSyntaxGrammar {
            variables,
            ..Default::default()
        }
    }
}
