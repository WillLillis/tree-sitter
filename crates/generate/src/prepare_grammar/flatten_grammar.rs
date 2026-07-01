use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::flat_rule::{FlatRule, FlatRules, RuleId, Step};
use super::{FlatExtractedSyntaxGrammar, FlatVariable};
use crate::{
    grammars::{Production, ProductionStep, ReservedWordSetId, SyntaxGrammar, SyntaxVariable},
    rules::{Alias, Associativity, Precedence, Symbol, TokenSet},
};

pub type FlattenGrammarResult<T> = Result<T, FlattenGrammarError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum FlattenGrammarError {
    #[error("No such reserved word set: {0}")]
    NoReservedWordSet(String),
    #[error(
        "The rule `{0}` matches the empty string.

Tree-sitter does not support syntactic rules that match the empty string
unless they are used only as the grammar's start rule.
"
    )]
    EmptyString(String),
    #[error("Rule `{0}` cannot be inlined because it contains a reference to itself")]
    RecursiveInline(String),
}

struct RuleFlattener {
    production: Production,
    reserved_word_set_ids: FxHashMap<String, ReservedWordSetId>,
    precedence_stack: Vec<Precedence>,
    associativity_stack: Vec<Associativity>,
    reserved_word_stack: Vec<ReservedWordSetId>,
    alias_stack: Vec<Alias>,
    field_name_stack: Vec<String>,
}

/// One step of [`RuleFlattener::apply`]'s iterative walk over a choice-free rule.
/// `Enter` does a node's pre-order work; the `Exit*` variants do its post-order
/// work once its children are done. `ExitSeq` folds the children's "did-push"
/// flags; `ExitMetadata`/`ExitReserved` pop the context a matching `Enter` pushed.
/// The `bool` carried by `Enter`/`ExitMetadata` is `at_end`: whether the node is
/// the final position of the production, where a wrapping precedence is not
/// overridden by the enclosing one.
enum Task {
    Enter(RuleId, bool),
    ExitSeq(usize),
    ExitMetadata(RuleId, bool),
    ExitReserved,
}

impl RuleFlattener {
    const fn new(reserved_word_set_ids: FxHashMap<String, ReservedWordSetId>) -> Self {
        Self {
            production: Production {
                steps: Vec::new(),
                dynamic_precedence: 0,
            },
            reserved_word_set_ids,
            precedence_stack: Vec::new(),
            associativity_stack: Vec::new(),
            reserved_word_stack: Vec::new(),
            alias_stack: Vec::new(),
            field_name_stack: Vec::new(),
        }
    }

    fn flatten_variable(
        &mut self,
        pool: &mut FlatRules,
        variable: FlatVariable,
    ) -> FlattenGrammarResult<SyntaxVariable> {
        let choices = extract_choices(pool, variable.rule);
        let mut productions = Vec::with_capacity(choices.len());
        for rule in choices {
            let production = self.flatten_rule(pool, rule)?;
            if !productions.contains(&production) {
                productions.push(production);
            }
        }
        Ok(SyntaxVariable {
            name: variable.name,
            kind: variable.kind,
            productions,
        })
    }

    fn flatten_rule(&mut self, pool: &FlatRules, rule: RuleId) -> FlattenGrammarResult<Production> {
        self.production = Production::default();
        self.alias_stack.clear();
        self.reserved_word_stack.clear();
        self.precedence_stack.clear();
        self.associativity_stack.clear();
        self.field_name_stack.clear();
        self.apply(pool, rule)?;
        Ok(self.production.clone())
    }

    /// Walk a choice-free rule, appending its steps to `self.production`. Iterative
    /// post-order: `Enter` pushes metadata context (precedence/associativity/alias/
    /// field/reserved) and appends a step for each `Symbol`; the `Exit*` tasks pop
    /// that context and apply the trailing-precedence fixup. The `did_push` stack
    /// stands in for the recursive version's `bool` return - the fixup only fires
    /// when a metadata subtree actually pushed a step and is not at the end.
    fn apply(&mut self, pool: &FlatRules, root: RuleId) -> FlattenGrammarResult<()> {
        let mut work = vec![Task::Enter(root, true)];
        let mut did_push: Vec<bool> = Vec::new();
        while let Some(task) = work.pop() {
            match task {
                Task::Enter(id, at_end) => match pool.get(id) {
                    FlatRule::Seq(range) => {
                        let range = *range;
                        let children = pool.children(range);
                        let last = children.len().wrapping_sub(1);
                        work.push(Task::ExitSeq(children.len()));
                        // Reversed so child 0 is processed first; only the last
                        // child inherits `at_end`.
                        for (i, &child) in children.iter().enumerate().rev() {
                            work.push(Task::Enter(child, i == last && at_end));
                        }
                    }
                    FlatRule::Metadata { params, rule } => {
                        let inner = *rule;
                        if !params.precedence.is_none() {
                            self.precedence_stack.push(params.precedence.clone());
                        }
                        if let Some(associativity) = params.associativity {
                            self.associativity_stack.push(associativity);
                        }
                        if let Some(alias) = &params.alias {
                            self.alias_stack.push(alias.clone());
                        }
                        if let Some(field_name) = &params.field_name {
                            self.field_name_stack.push(field_name.clone());
                        }
                        if params.dynamic_precedence.abs()
                            > self.production.dynamic_precedence.abs()
                        {
                            self.production.dynamic_precedence = params.dynamic_precedence;
                        }
                        work.push(Task::ExitMetadata(id, at_end));
                        work.push(Task::Enter(inner, at_end));
                    }
                    FlatRule::Reserved { rule, context_name } => {
                        let inner = *rule;
                        let set_id = self
                            .reserved_word_set_ids
                            .get(context_name)
                            .copied()
                            .ok_or_else(|| {
                                FlattenGrammarError::NoReservedWordSet(context_name.clone())
                            })?;
                        self.reserved_word_stack.push(set_id);
                        work.push(Task::ExitReserved);
                        work.push(Task::Enter(inner, at_end));
                    }
                    FlatRule::Symbol(symbol) => {
                        let symbol = *symbol;
                        self.production.steps.push(ProductionStep {
                            symbol,
                            precedence: self
                                .precedence_stack
                                .last()
                                .cloned()
                                .unwrap_or(Precedence::None),
                            associativity: self.associativity_stack.last().copied(),
                            reserved_word_set_id: self
                                .reserved_word_stack
                                .last()
                                .copied()
                                .unwrap_or_default(),
                            alias: self.alias_stack.last().cloned(),
                            field_name: self.field_name_stack.last().cloned(),
                        });
                        did_push.push(true);
                    }
                    // Blank (and, defensively, any other leaf) contributes no step.
                    _ => did_push.push(false),
                },
                Task::ExitSeq(n) => {
                    let mut any = false;
                    for _ in 0..n {
                        any |= did_push.pop().unwrap();
                    }
                    did_push.push(any);
                }
                Task::ExitMetadata(id, at_end) => {
                    let FlatRule::Metadata { params, .. } = pool.get(id) else {
                        unreachable!("ExitMetadata on a non-metadata node")
                    };
                    // A metadata subtree passes its inner's `did_push` straight
                    // through, so peek (don't pop) it for the fixup.
                    let pushed = *did_push.last().unwrap();
                    if !params.precedence.is_none() {
                        self.precedence_stack.pop();
                        if pushed && !at_end {
                            self.production.steps.last_mut().unwrap().precedence = self
                                .precedence_stack
                                .last()
                                .cloned()
                                .unwrap_or(Precedence::None);
                        }
                    }
                    if params.associativity.is_some() {
                        self.associativity_stack.pop();
                        if pushed && !at_end {
                            self.production.steps.last_mut().unwrap().associativity =
                                self.associativity_stack.last().copied();
                        }
                    }
                    if params.alias.is_some() {
                        self.alias_stack.pop();
                    }
                    if params.field_name.is_some() {
                        self.field_name_stack.pop();
                    }
                }
                Task::ExitReserved => {
                    self.reserved_word_stack.pop();
                }
            }
        }
        Ok(())
    }
}

/// Expand every `Choice` in `root` into the flat list of choice-free rules it
/// stands for (the cartesian product of the choices), interning each into `pool`
/// and returning their `RuleId`s. Iterative post-order over the pool via
/// [`Step`]: a `Seq` takes the product of its members' expansions, a `Choice`
/// their concatenation, and `Metadata`/`Reserved` distribute over each
/// alternative. The `out` stack holds one expansion list per visited node.
fn extract_choices(pool: &mut FlatRules, root: RuleId) -> Vec<RuleId> {
    let mut work = vec![Step::Visit(root)];
    let mut out: Vec<Vec<RuleId>> = Vec::new();
    while let Some(step) = work.pop() {
        match step {
            Step::Visit(id) => match pool.get(id) {
                FlatRule::Seq(_)
                | FlatRule::Choice(_)
                | FlatRule::Metadata { .. }
                | FlatRule::Reserved { .. } => pool.descend(id, &mut work),
                // A leaf stands for a single choice-free alternative: itself.
                _ => out.push(vec![id]),
            },
            Step::Build(id) => match pool.get(id) {
                FlatRule::Seq(range) => {
                    let n = range.len as usize;
                    let base = out.len() - n;
                    let child_lists = out.split_off(base);
                    // Cartesian product of the members, in source order, with the
                    // last member varying fastest (matching the recursive fold).
                    // Seed with the first member's alternatives, so a single-member
                    // seq needs no wrapping.
                    let mut lists = child_lists.into_iter();
                    let mut result = lists.next().unwrap_or_else(|| vec![pool.intern_blank()]);
                    for extraction in lists {
                        let mut next = Vec::with_capacity(result.len() * extraction.len());
                        for &entry in &result {
                            for &ext in &extraction {
                                next.push(pool.intern_seq_or_choice(true, &[entry, ext]));
                            }
                        }
                        result = next;
                    }
                    out.push(result);
                }
                FlatRule::Choice(range) => {
                    let n = range.len as usize;
                    let base = out.len() - n;
                    let child_lists = out.split_off(base);
                    out.push(child_lists.into_iter().flatten().collect());
                }
                FlatRule::Metadata { params, .. } => {
                    let params = params.clone();
                    let inner = out.pop().unwrap();
                    out.push(
                        inner
                            .into_iter()
                            .map(|rule| pool.intern_metadata(params.clone(), rule))
                            .collect(),
                    );
                }
                FlatRule::Reserved { context_name, .. } => {
                    let context_name = context_name.clone();
                    let inner = out.pop().unwrap();
                    out.push(
                        inner
                            .into_iter()
                            .map(|rule| pool.intern_reserved(&context_name, rule))
                            .collect(),
                    );
                }
                _ => unreachable!("Build on a leaf"),
            },
        }
    }
    out.pop().unwrap()
}

fn symbol_is_used(variables: &[SyntaxVariable], symbol: Symbol) -> bool {
    for variable in variables {
        for production in &variable.productions {
            for step in &production.steps {
                if step.symbol == symbol {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn flatten_grammar(
    grammar: FlatExtractedSyntaxGrammar,
    pool: &mut FlatRules,
) -> FlattenGrammarResult<SyntaxGrammar> {
    let mut reserved_word_set_ids_by_name = FxHashMap::default();
    for (ix, set) in grammar.reserved_word_sets.iter().enumerate() {
        reserved_word_set_ids_by_name.insert(set.name.clone(), ReservedWordSetId(ix));
    }

    let mut flattener = RuleFlattener::new(reserved_word_set_ids_by_name);
    let mut variables = Vec::with_capacity(grammar.variables.len());
    for variable in grammar.variables {
        variables.push(flattener.flatten_variable(pool, variable)?);
    }

    for (i, variable) in variables.iter().enumerate() {
        let symbol = Symbol::non_terminal(i);
        let used = symbol_is_used(&variables, symbol);

        for production in &variable.productions {
            if used && production.steps.is_empty() {
                Err(FlattenGrammarError::EmptyString(variable.name.clone()))?;
            }

            if grammar.variables_to_inline.contains(&symbol)
                && production.steps.iter().any(|step| step.symbol == symbol)
            {
                Err(FlattenGrammarError::RecursiveInline(variable.name.clone()))?;
            }
        }
    }
    let mut reserved_word_sets = grammar
        .reserved_word_sets
        .into_iter()
        .map(|set| set.reserved_words.into_iter().collect())
        .collect::<Vec<_>>();

    // If no default reserved word set is specified, there are no reserved words.
    if reserved_word_sets.is_empty() {
        reserved_word_sets.push(TokenSet::default());
    }

    Ok(SyntaxGrammar {
        extra_symbols: grammar.extra_symbols,
        expected_conflicts: grammar.expected_conflicts,
        variables_to_inline: grammar.variables_to_inline,
        precedence_orderings: grammar.precedence_orderings,
        external_tokens: grammar.external_tokens,
        supertype_symbols: grammar.supertype_symbols,
        word_token: grammar.word_token,
        reserved_word_sets,
        variables,
    })
}

#[cfg(test)]
mod tests {
    use super::super::ExtractedSyntaxGrammar;
    use super::*;
    use crate::grammars::{Variable, VariableType};
    use crate::rules::Rule;

    /// Flatten a single rule (interned into a fresh pool) into its productions, so
    /// the assertions below can stay written against `Rule`.
    fn productions_of(rule: &Rule) -> Vec<Production> {
        let mut pool = FlatRules::default();
        let rule = pool.intern_import(rule);
        let mut flattener = RuleFlattener::new(FxHashMap::default());
        flattener
            .flatten_variable(
                &mut pool,
                FlatVariable {
                    name: "test".to_string(),
                    kind: VariableType::Named,
                    rule,
                },
            )
            .unwrap()
            .productions
    }

    /// Run `flatten_grammar` on an `ExtractedSyntaxGrammar` by interning its rules
    /// into a fresh pool first.
    fn flatten(grammar: ExtractedSyntaxGrammar) -> FlattenGrammarResult<SyntaxGrammar> {
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
        flatten_grammar(flat, &mut pool)
    }

    #[test]
    fn test_flatten_grammar() {
        let productions = productions_of(&Rule::seq(vec![
            Rule::non_terminal(1),
            Rule::prec_left(
                Precedence::Integer(101),
                Rule::seq(vec![
                    Rule::non_terminal(2),
                    Rule::choice(vec![
                        Rule::prec_right(
                            Precedence::Integer(102),
                            Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                        ),
                        Rule::non_terminal(5),
                    ]),
                    Rule::non_terminal(6),
                ]),
            ),
            Rule::non_terminal(7),
        ]));

        assert_eq!(
            productions,
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(3))
                            .with_prec(Precedence::Integer(102), Some(Associativity::Right)),
                        ProductionStep::new(Symbol::non_terminal(4))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ]
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(5))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ]
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_maximum_dynamic_precedence() {
        let productions = productions_of(&Rule::seq(vec![
            Rule::non_terminal(1),
            Rule::prec_dynamic(
                101,
                Rule::seq(vec![
                    Rule::non_terminal(2),
                    Rule::choice(vec![
                        Rule::prec_dynamic(
                            102,
                            Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                        ),
                        Rule::non_terminal(5),
                    ]),
                    Rule::non_terminal(6),
                ]),
            ),
            Rule::non_terminal(7),
        ]));

        assert_eq!(
            productions,
            vec![
                Production {
                    dynamic_precedence: 102,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::non_terminal(3)),
                        ProductionStep::new(Symbol::non_terminal(4)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ],
                },
                Production {
                    dynamic_precedence: 101,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::non_terminal(5)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ],
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_final_precedence() {
        let productions = productions_of(&Rule::prec_left(
            Precedence::Integer(101),
            Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
        ));

        assert_eq!(
            productions,
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::non_terminal(1))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                    ProductionStep::new(Symbol::non_terminal(2))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                ]
            }]
        );

        let productions = productions_of(&Rule::prec_left(
            Precedence::Integer(101),
            Rule::seq(vec![Rule::non_terminal(1)]),
        ));

        assert_eq!(
            productions,
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::non_terminal(1))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                ]
            }]
        );
    }

    #[test]
    fn test_flatten_grammar_with_field_names() {
        let productions = productions_of(&Rule::seq(vec![
            Rule::field("first-thing".to_string(), Rule::terminal(1)),
            Rule::terminal(2),
            Rule::choice(vec![
                Rule::Blank,
                Rule::field("second-thing".to_string(), Rule::terminal(3)),
            ]),
        ]));

        assert_eq!(
            productions,
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(1)).with_field_name("first-thing"),
                        ProductionStep::new(Symbol::terminal(2))
                    ]
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(1)).with_field_name("first-thing"),
                        ProductionStep::new(Symbol::terminal(2)),
                        ProductionStep::new(Symbol::terminal(3)).with_field_name("second-thing"),
                    ]
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_recursive_inline_variable() {
        let result = flatten(ExtractedSyntaxGrammar {
            extra_symbols: Vec::new(),
            expected_conflicts: Vec::new(),
            variables_to_inline: vec![Symbol::non_terminal(0)],
            precedence_orderings: Vec::new(),
            external_tokens: Vec::new(),
            supertype_symbols: Vec::new(),
            word_token: None,
            reserved_word_sets: Vec::new(),
            variables: vec![Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::seq(vec![
                    Rule::non_terminal(0),
                    Rule::non_terminal(1),
                    Rule::non_terminal(2),
                ]),
            }],
        });

        assert_eq!(
            result.unwrap_err().to_string(),
            "Rule `test` cannot be inlined because it contains a reference to itself",
        );
    }
}
