use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use super::ExtractedSyntaxGrammar;
#[cfg(test)]
use crate::{
    grammars::{Production, ProductionStep, Variable},
    rules::{Precedence, Rule},
};
use crate::{
    grammars::{SyntaxGrammar, SyntaxVariable},
    rules::{Associativity, Symbol, TokenSet},
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

use super::extract_tokens::PoolExtractedMeta;
use crate::rule_pool::{
    FProd, FSTEP_ASSOC_LEFT, FSTEP_ASSOC_MASK, FSTEP_ASSOC_RIGHT, FSTEP_PREC_INTEGER,
    FSTEP_PREC_MASK, FSTEP_PREC_NAME, FStep, FlatOut, Node as PoolNode, NodeId, PoolGrammar, Prec,
    RulePool, StrId,
};
use crate::rules::SymbolType;

#[derive(Clone, Copy, Default)]
struct Context {
    prec: Prec,
    assoc: Option<Associativity>,
    alias: Option<crate::rule_pool::Alias>,
    field: Option<StrId>,
    reserved: u16,
}

/// Selects one branch at each choice encountered during a root-to-leaf walk.
/// After a production is emitted, `advance` increments this path like a
/// mixed-radix counter. Later decisions are discarded because choosing an
/// earlier branch can expose a different set of nested choices.
#[derive(Default)]
struct ChoiceCursor {
    decisions: Vec<u32>,
    arities: Vec<u32>,
    depth: usize,
}

impl ChoiceCursor {
    fn reset(&mut self) {
        self.decisions.clear();
        self.arities.clear();
        self.depth = 0;
    }

    fn begin_path(&mut self) {
        self.arities.clear();
        self.depth = 0;
    }

    fn select(&mut self, len: u32) -> u32 {
        debug_assert!(len > 0);
        let selected = self.decisions.get(self.depth).copied().unwrap_or(0);
        debug_assert!(selected < len);
        self.arities.push(len);
        self.depth += 1;
        selected
    }

    fn advance(&mut self) -> bool {
        for depth in (0..self.arities.len()).rev() {
            let selected = self.decisions.get(depth).copied().unwrap_or(0);
            if selected + 1 < self.arities[depth] {
                self.decisions.resize(depth + 1, 0);
                self.decisions[depth] = selected + 1;
                return true;
            }
        }
        false
    }
}

/// Reusable scratch for enumerating and flattening one production path at a
/// time. Context is passed by value during the walk, so no traversal state has
/// to be snapshotted or restored at choices.
#[derive(Default)]
pub(super) struct FlattenState {
    steps: Vec<FStep>,
    choices: ChoiceCursor,
    dyn_prec: i32,
}

impl FlattenState {
    fn reset_variable(&mut self) {
        self.choices.reset();
        self.reset_path();
    }

    fn reset_path(&mut self) {
        self.steps.clear();
        self.dyn_prec = 0;
        self.choices.begin_path();
    }

    fn push_step(&mut self, kind: SymbolType, index: u32, context: Context) {
        self.steps.push(FStep::pack(
            Symbol {
                kind,
                index: index as usize,
            },
            context.prec,
            context.assoc,
            context.alias,
            context.field,
            context.reserved,
        ));
    }

    fn restore_outer_prec(&mut self, outer: Prec) {
        let (bits, val) = match outer {
            Prec::None => (0, 0),
            Prec::Integer(n) => (FSTEP_PREC_INTEGER, n),
            Prec::Name(sid) => (FSTEP_PREC_NAME, sid.raw() as i32),
        };
        let step = self.steps.last_mut().unwrap();
        step.prec_val = val;
        step.flags = (step.flags & !FSTEP_PREC_MASK) | bits;
    }

    fn restore_outer_assoc(&mut self, outer: Option<Associativity>) {
        let bits = match outer {
            None => 0,
            Some(Associativity::Left) => FSTEP_ASSOC_LEFT,
            Some(Associativity::Right) => FSTEP_ASSOC_RIGHT,
        };
        let step = self.steps.last_mut().unwrap();
        step.flags = (step.flags & !FSTEP_ASSOC_MASK) | bits;
    }
}

pub(super) fn flatten_grammar(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    st: &mut FlattenState,
    out: &mut FlatOut,
) -> FlattenGrammarResult<()> {
    // Last wins on duplicate names.
    let reserved_ids: FxHashMap<StrId, u16> = meta
        .reserved_sets
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, i as u16))
        .collect();
    out.clear();
    for v in &g.variables {
        let prod_start = out.productions.len() as u32;
        st.reset_variable();
        loop {
            apply(&g.pool, &reserved_ids, v.root, Context::default(), true, st)?;
            emit(st, out, prod_start);
            if !st.choices.advance() {
                break;
            }
            st.reset_path();
        }
        out.var_prods
            .push((prod_start, out.productions.len() as u32));
    }
    check(g, meta, out)
}

/// Post-flatten checks: empty productions in used variables, recursive
/// inlines. Same iteration order as the historical Rule-based pass.
fn check(g: &PoolGrammar, meta: &PoolExtractedMeta, out: &FlatOut) -> FlattenGrammarResult<()> {
    for (i, &(ps, pe)) in out.var_prods.iter().enumerate() {
        let symbol = Symbol::non_terminal(i);
        let used = out.steps.iter().any(|s| s.symbol() == symbol);
        let inlined = meta.inline.contains(&symbol);
        for p in &out.productions[ps as usize..pe as usize] {
            if used && p.steps_len == 0 {
                Err(FlattenGrammarError::EmptyString(
                    g.pool.resolve(g.variables[i].name).to_string(),
                ))?;
            }
            if inlined
                && out.steps[p.step_range()]
                    .iter()
                    .any(|s| s.symbol() == symbol)
            {
                Err(FlattenGrammarError::RecursiveInline(
                    g.pool.resolve(g.variables[i].name).to_string(),
                ))?;
            }
        }
    }
    Ok(())
}

/// Flatten one deterministic path. Choices only select a child here; the
/// outer loop enumerates subsequent paths from the root.
fn apply(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    node: NodeId,
    context: Context,
    at_end: bool,
    st: &mut FlattenState,
) -> FlattenGrammarResult<bool> {
    match pool.node(node) {
        PoolNode::Sym { kind, index } => {
            st.push_step(kind, index, context);
            Ok(true)
        }
        PoolNode::Seq(range) => {
            let children = pool.child_slice(range);
            let mut did_push = false;
            for (i, &child) in children.iter().enumerate() {
                did_push |= apply(
                    pool,
                    reserved_ids,
                    child,
                    context,
                    at_end && i + 1 == children.len(),
                    st,
                )?;
            }
            Ok(did_push)
        }
        PoolNode::Choice(range) => {
            let selected = st.choices.select(range.len);
            let child = pool.child_slice(range)[selected as usize];
            apply(pool, reserved_ids, child, context, at_end, st)
        }
        PoolNode::Metadata { params, rule } => {
            let params = pool.params(params);
            let mut inner = context;
            if params.prec != Prec::None {
                inner.prec = params.prec;
            }
            if let Some(assoc) = params.assoc {
                inner.assoc = Some(assoc);
            }
            if let Some(alias) = params.alias {
                inner.alias = Some(alias);
            }
            if let Some(field) = params.field {
                inner.field = Some(field);
            }
            if params.dynamic_precedence.abs() > st.dyn_prec.abs() {
                st.dyn_prec = params.dynamic_precedence;
            }

            let did_push = apply(pool, reserved_ids, rule, inner, at_end, st)?;
            if did_push && !at_end {
                if params.prec != Prec::None {
                    st.restore_outer_prec(context.prec);
                }
                if params.assoc.is_some() {
                    st.restore_outer_assoc(context.assoc);
                }
            }
            Ok(did_push)
        }
        PoolNode::Reserved { rule, ctx } => {
            let Some(&reserved) = reserved_ids.get(&ctx) else {
                return Err(FlattenGrammarError::NoReservedWordSet(
                    pool.resolve(ctx).to_string(),
                ));
            };
            let inner = Context {
                reserved,
                ..context
            };
            apply(pool, reserved_ids, rule, inner, at_end, st)
        }
        // Blank pushes nothing; String/Pattern/NamedSymbol/Repeat cannot
        // appear post-extract/expand.
        _ => Ok(false),
    }
}

/// Append the completed path as a production unless this variable already
/// has an identical one (order-preserving dedup).
fn emit(st: &FlattenState, out: &mut FlatOut, prod_start: u32) {
    for p in &out.productions[prod_start as usize..] {
        if p.dynamic_precedence == st.dyn_prec && out.steps[p.step_range()] == st.steps[..] {
            return;
        }
    }
    let steps_start = out.steps.len() as u32;
    out.steps.extend_from_slice(&st.steps);
    out.productions.push(FProd {
        steps_start,
        steps_len: st.steps.len() as u32,
        dynamic_precedence: st.dyn_prec,
    });
}

/// Assemble the [`SyntaxGrammar`]: the flat pools move in, names resolve
/// once, and the pass metadata becomes the passthrough fields.
pub(super) fn assemble_syntax_grammar(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    out: FlatOut,
) -> SyntaxGrammar {
    let mut reserved_word_sets: Vec<TokenSet> = meta
        .reserved_sets
        .iter()
        .map(|(_, symbols)| symbols.iter().copied().collect())
        .collect();
    if reserved_word_sets.is_empty() {
        reserved_word_sets.push(TokenSet::default());
    }
    SyntaxGrammar {
        variables: g
            .variables
            .iter()
            .zip(&meta.kinds)
            .map(|(v, &kind)| SyntaxVariable {
                name: g.pool.resolve(v.name).to_string(),
                kind,
                #[cfg(test)]
                productions: Vec::new(),
            })
            .collect(),
        steps: out.steps,
        productions: out.productions,
        var_prods: out.var_prods,
        strs: g.pool.strs().to_vec(),
        extra_symbols: meta.extra_symbols.clone(),
        expected_conflicts: meta.conflicts.clone(),
        variables_to_inline: meta.inline.clone(),
        precedence_orderings: g.precedence_orderings.clone(),
        external_tokens: meta.external_tokens.clone(),
        supertype_symbols: meta.supertypes.clone(),
        word_token: meta.word,
        reserved_word_sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::VariableType;

    #[test]
    fn test_flatten_grammar() {
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::seq(vec![
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
            ]),
        });

        assert_eq!(
            result,
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
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::seq(vec![
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
            ]),
        });

        assert_eq!(
            result,
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
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::prec_left(
                Precedence::Integer(101),
                Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
            ),
        });

        assert_eq!(
            result,
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

        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::prec_left(
                Precedence::Integer(101),
                Rule::seq(vec![Rule::non_terminal(1)]),
            ),
        });

        assert_eq!(
            result,
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
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::seq(vec![
                Rule::field("first-thing".to_string(), Rule::terminal(1)),
                Rule::terminal(2),
                Rule::choice(vec![
                    Rule::Blank,
                    Rule::field("second-thing".to_string(), Rule::terminal(3)),
                ]),
            ]),
        });

        assert_eq!(
            result,
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

    fn pool_from_extracted(g: &ExtractedSyntaxGrammar) -> (PoolGrammar, PoolExtractedMeta) {
        use crate::rule_pool::PoolVariable;
        let mut pool = RulePool::default();
        let variables = g
            .variables
            .iter()
            .map(|v| PoolVariable {
                name: pool.intern(&v.name),
                root: pool.add_rule(&v.rule),
            })
            .collect();
        let reserved_sets = g
            .reserved_word_sets
            .iter()
            .map(|s| (pool.intern(&s.name), s.reserved_words.clone()))
            .collect();
        let meta = PoolExtractedMeta {
            kinds: g.variables.iter().map(|v| v.kind).collect(),
            lexical_variables: Vec::new(),
            separator_roots: Vec::new(),
            extra_symbols: g.extra_symbols.clone(),
            external_tokens: g.external_tokens.clone(),
            reserved_sets,
            supertypes: g.supertype_symbols.clone(),
            conflicts: g.expected_conflicts.clone(),
            inline: g.variables_to_inline.clone(),
            word: g.word_token,
        };
        let pg = PoolGrammar {
            pool,
            name: String::new(),
            variables,
            external_roots: Vec::new(),
            extra_roots: Vec::new(),
            reserved_sets: Vec::new(),
            supertype_names: Vec::new(),
            conflict_names: Vec::new(),
            inline_names: Vec::new(),
            word_name: None,
            precedence_orderings: g.precedence_orderings.clone(),
        };
        (pg, meta)
    }

    fn pool_pass(g: &ExtractedSyntaxGrammar) -> FlattenGrammarResult<SyntaxGrammar> {
        let (pg, meta) = pool_from_extracted(g);
        let mut st = FlattenState::default();
        let mut out = crate::rule_pool::FlatOut::default();
        flatten_grammar(&pg, &meta, &mut st, &mut out)?;
        Ok(assemble_syntax_grammar(&pg, &meta, out))
    }

    fn flatten_variable(variable: Variable) -> Vec<Production> {
        let g = ExtractedSyntaxGrammar {
            variables: vec![variable],
            ..Default::default()
        };
        pool_pass(&g).unwrap().legacy_variable_productions(0)
    }

    #[test]
    fn test_errors() {
        // Unknown reserved context.
        let g = ExtractedSyntaxGrammar {
            variables: vec![Variable::named(
                "test",
                Rule::Reserved {
                    rule: Box::new(Rule::terminal(1)),
                    context_name: "nope".to_string(),
                },
            )],
            ..Default::default()
        };
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "No such reserved word set: nope"
        );

        // Empty production in a used variable.
        let g = ExtractedSyntaxGrammar {
            variables: vec![
                Variable::named("a", Rule::non_terminal(1)),
                Variable::named("b", Rule::Blank),
            ],
            ..Default::default()
        };
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "The rule `b` matches the empty string.

Tree-sitter does not support syntactic rules that match the empty string
unless they are used only as the grammar's start rule.
"
        );
    }

    #[test]
    fn test_flatten_grammar_with_recursive_inline_variable() {
        let result = pool_pass(&ExtractedSyntaxGrammar {
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
