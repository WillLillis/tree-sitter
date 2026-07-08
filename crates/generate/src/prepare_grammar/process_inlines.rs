use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use crate::grammars::{LexicalGrammar, ProductionStep};
use crate::{
    grammars::{InlinedProductionMap, Production, SyntaxGrammar},
    rules::SymbolType,
};

pub type ProcessInlinesResult<T> = Result<T, ProcessInlinesError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ProcessInlinesError {
    #[error("External token `{0}` cannot be inlined")]
    ExternalToken(String),
    #[error("Token `{0}` cannot be inlined")]
    Token(String),
    #[error("Rule `{0}` cannot be inlined because it is the first rule")]
    FirstRule(String),
}

use std::hash::{Hash, Hasher};

use rustc_hash::FxHasher;

use super::extract_tokens::PoolExtractedMeta;
use crate::rule_pool::{
    FProd, FSTEP_ALIAS_NAMED, FSTEP_ASSOC_MASK, FSTEP_PREC_MASK, FStep, FlatOut, PoolGrammar,
    RulePool,
};
use crate::rules::Symbol;

/// Pool inline map: created productions are appended to the `FlatOut`
/// production pool, so ids are unified (ids at or above `first_inlined` are
/// created). `map` is keyed by `(production id, step index)`.
#[derive(Debug)]
pub(super) struct PoolInlines {
    pub first_inlined: u32,
    pub map: FxHashMap<(u32, u32), Vec<u32>>,
}

/// Expand the grammar's inlined variables: walk every production step,
/// splicing inlined variables' productions in place of referencing steps
/// (cartesian over their alternatives, repeated to a fixed point).
pub(super) fn process_inlines(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    out: &mut FlatOut,
) -> ProcessInlinesResult<PoolInlines> {
    for symbol in &meta.inline {
        match symbol.kind {
            SymbolType::External => {
                Err(ProcessInlinesError::ExternalToken(
                    meta.external_tokens[symbol.index].name.clone(),
                ))?;
            }
            SymbolType::Terminal => {
                Err(ProcessInlinesError::Token(
                    g.pool
                        .resolve(meta.lexical_variables[symbol.index].name)
                        .to_string(),
                ))?;
            }
            SymbolType::NonTerminal if symbol.index == 0 => {
                Err(ProcessInlinesError::FirstRule(
                    g.pool.resolve(g.variables[0].name).to_string(),
                ))?;
            }
            _ => {}
        }
    }

    Ok(PoolInlineBuilder {
        first_inlined: out.productions.len() as u32,
        out,
        inline: &meta.inline,
        map: FxHashMap::default(),
        memo: FxHashMap::default(),
    }
    .build())
}

struct PoolInlineBuilder<'a> {
    out: &'a mut FlatOut,
    first_inlined: u32,
    inline: &'a [Symbol],
    map: FxHashMap<(u32, u32), Vec<u32>>,
    /// Created-production dedup: content hash -> unified ids, candidates in
    /// creation order so the first structural match wins.
    memo: FxHashMap<u64, Vec<u32>>,
}

#[derive(Clone, Default)]
struct ScratchProd {
    steps: Vec<FStep>,
    dynamic_precedence: i32,
}

impl PoolInlineBuilder<'_> {
    fn build(mut self) -> PoolInlines {
        let mut worklist: Vec<(u32, u32)> = Vec::new();
        for prod_id in 0..self.first_inlined {
            worklist.push((prod_id, 0));
            while !worklist.is_empty() {
                let mut i = 0;
                while i < worklist.len() {
                    let (prod_id, step_index) = worklist[i];
                    if let Some(step) = self.step_at(prod_id, step_index) {
                        if self.inline.contains(&step.symbol()) {
                            let ids = self.inline_at(prod_id, step_index);
                            worklist.splice(i..=i, ids.into_iter().map(|id| (id, step_index)));
                        } else {
                            worklist[i].1 += 1;
                            i += 1;
                        }
                    } else {
                        worklist.remove(i);
                    }
                }
            }
        }
        PoolInlines {
            first_inlined: self.first_inlined,
            map: self.map,
        }
    }

    fn step_at(&self, prod_id: u32, step: u32) -> Option<FStep> {
        let p = self.out.productions[prod_id as usize];
        (step < p.steps_len).then(|| self.out.steps[(p.steps_start + step) as usize])
    }

    /// Expand inlining at one step of one production, including nested
    /// inlining at the same step index, then dedup and store the results.
    fn inline_at(&mut self, prod_id: u32, step_index: u32) -> Vec<u32> {
        if let Some(ids) = self.map.get(&(prod_id, step_index)) {
            return ids.clone();
        }

        let si = step_index as usize;
        let src = self.out.productions[prod_id as usize];
        let mut scratch = vec![ScratchProd {
            steps: self.out.steps[src.step_range()].to_vec(),
            dynamic_precedence: src.dynamic_precedence,
        }];
        let mut i = 0;
        while i < scratch.len() {
            let symbol = match scratch[i].steps.get(si) {
                Some(s) if self.inline.contains(&s.symbol()) => s.symbol(),
                _ => {
                    i += 1;
                    continue;
                }
            };

            let removed_prod = std::mem::take(&mut scratch[i]);
            let removed_step = removed_prod.steps[si];
            let (vs, ve) = self.out.var_prods[symbol.index];
            let replacements: Vec<ScratchProd> = (vs..ve)
                .map(|p_idx| {
                    let p = self.out.productions[p_idx as usize];
                    let mut production = removed_prod.clone();
                    production
                        .steps
                        .splice(si..=si, self.out.steps[p.step_range()].iter().copied());
                    let inserted = &mut production.steps[si..si + p.steps_len as usize];
                    if removed_step.alias != 0 {
                        for step in inserted.iter_mut() {
                            step.alias = removed_step.alias;
                            step.flags = (step.flags & !FSTEP_ALIAS_NAMED)
                                | (removed_step.flags & FSTEP_ALIAS_NAMED);
                        }
                    }
                    if removed_step.field != 0 {
                        for step in inserted.iter_mut() {
                            step.field = removed_step.field;
                        }
                    }
                    if let Some(last) = inserted.last_mut() {
                        if last.flags & FSTEP_PREC_MASK == 0 {
                            last.prec_val = removed_step.prec_val;
                            last.flags |= removed_step.flags & FSTEP_PREC_MASK;
                        }
                        if last.flags & FSTEP_ASSOC_MASK == 0 {
                            last.flags |= removed_step.flags & FSTEP_ASSOC_MASK;
                        }
                    }
                    if p.dynamic_precedence.abs() > production.dynamic_precedence.abs() {
                        production.dynamic_precedence = p.dynamic_precedence;
                    }
                    production
                })
                .collect();
            scratch.splice(i..=i, replacements);
        }

        let mut result = Vec::with_capacity(scratch.len());
        for sp in scratch {
            let mut hasher = FxHasher::default();
            sp.dynamic_precedence.hash(&mut hasher);
            sp.steps.hash(&mut hasher);
            let candidates = self.memo.entry(hasher.finish()).or_default();
            let existing = candidates.iter().copied().find(|&id| {
                let p = self.out.productions[id as usize];
                p.dynamic_precedence == sp.dynamic_precedence
                    && self.out.steps[p.step_range()] == sp.steps[..]
            });
            result.push(existing.unwrap_or_else(|| {
                let steps_start = self.out.steps.len() as u32;
                self.out.steps.extend_from_slice(&sp.steps);
                self.out.productions.push(FProd {
                    steps_start,
                    steps_len: sp.steps.len() as u32,
                    dynamic_precedence: sp.dynamic_precedence,
                });
                let id = (self.out.productions.len() - 1) as u32;
                candidates.push(id);
                id
            }));
        }

        self.map.insert((prod_id, step_index), result.clone());
        result
    }
}

/// Boundary materialization to the address-keyed
/// [`InlinedProductionMap`] (created productions materialized once, map keys
/// resolved to production addresses). Deleted at the `SyntaxGrammar` flip.
pub(super) fn materialize_inlines(
    pool: &RulePool,
    out: &FlatOut,
    inl: &PoolInlines,
    syntax_grammar: &SyntaxGrammar,
) -> InlinedProductionMap {
    let first = inl.first_inlined as usize;
    let productions: Vec<Production> = out.productions[first..]
        .iter()
        .map(|p| Production {
            dynamic_precedence: p.dynamic_precedence,
            steps: out.steps[p.step_range()]
                .iter()
                .map(|s| super::flatten_grammar::step_to_master(pool, s))
                .collect(),
        })
        .collect();
    let mut flat_to_var = vec![(0usize, 0usize); first];
    for (v, &(ps, pe)) in out.var_prods.iter().enumerate() {
        for (p, id) in (ps..pe).enumerate() {
            flat_to_var[id as usize] = (v, p);
        }
    }
    let production_map = inl
        .map
        .iter()
        .map(|(&(id, step), ids)| {
            let production = if (id as usize) < first {
                let (v, p) = flat_to_var[id as usize];
                std::ptr::from_ref::<Production>(&syntax_grammar.variables[v].productions[p])
            } else {
                std::ptr::from_ref::<Production>(&productions[id as usize - first])
            };
            (
                (production, step),
                ids.iter().map(|&id| id as usize - first).collect(),
            )
        })
        .collect();
    InlinedProductionMap {
        productions,
        production_map,
        first_inlined: inl.first_inlined,
        map: inl.map.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammars::{ExternalToken, LexicalVariable, SyntaxVariable, VariableType},
        rules::{Associativity, Precedence, Symbol},
    };

    /// Returns the materialized grammar too: the map is address-keyed, so
    /// lookups must go through the same materialization.
    fn pool_pass(
        grammar: &SyntaxGrammar,
        lexical_grammar: &LexicalGrammar,
    ) -> ProcessInlinesResult<(SyntaxGrammar, InlinedProductionMap)> {
        let (g, meta, mut out) = super::super::pool_from_syntax(grammar, lexical_grammar);
        let inl = process_inlines(&g, &meta, &mut out)?;
        let sg = super::super::flatten_grammar::materialize_flattened(&g, &meta, &out);
        let map = materialize_inlines(&g.pool, &out, &inl, &sg);
        Ok((sg, map))
    }

    #[test]
    fn test_basic_inlining() {
        let grammar = SyntaxGrammar {
            variables_to_inline: vec![Symbol::non_terminal(1)],
            variables: vec![
                SyntaxVariable {
                    name: "non-terminal-0".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            ProductionStep::new(Symbol::terminal(10)),
                            ProductionStep::new(Symbol::non_terminal(1)), // inlined
                            ProductionStep::new(Symbol::terminal(11)),
                        ],
                    }],
                },
                SyntaxVariable {
                    name: "non-terminal-1".to_string(),
                    kind: VariableType::Named,
                    productions: vec![
                        Production {
                            dynamic_precedence: 0,
                            steps: vec![
                                ProductionStep::new(Symbol::terminal(12)),
                                ProductionStep::new(Symbol::terminal(13)),
                            ],
                        },
                        Production {
                            dynamic_precedence: -2,
                            steps: vec![ProductionStep::new(Symbol::terminal(14))],
                        },
                    ],
                },
            ],
            ..Default::default()
        };

        let (grammar, inline_map) = pool_pass(&grammar, &LexicalGrammar::default()).unwrap();

        // Nothing to inline at step 0.
        assert!(
            inline_map
                .inlined_productions(&grammar.variables[0].productions[0], 0)
                .is_none()
        );

        // Inlining variable 1 yields two productions.
        assert_eq!(
            inline_map
                .inlined_productions(&grammar.variables[0].productions[0], 1)
                .unwrap()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(10)),
                        ProductionStep::new(Symbol::terminal(12)),
                        ProductionStep::new(Symbol::terminal(13)),
                        ProductionStep::new(Symbol::terminal(11)),
                    ],
                },
                Production {
                    dynamic_precedence: -2,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(10)),
                        ProductionStep::new(Symbol::terminal(14)),
                        ProductionStep::new(Symbol::terminal(11)),
                    ],
                },
            ]
        );
    }

    #[test]
    fn test_nested_inlining() {
        let grammar = SyntaxGrammar {
            variables: vec![
                SyntaxVariable {
                    name: "non-terminal-0".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            ProductionStep::new(Symbol::terminal(10)),
                            ProductionStep::new(Symbol::non_terminal(1)), // inlined
                            ProductionStep::new(Symbol::terminal(11)),
                            ProductionStep::new(Symbol::non_terminal(2)), // inlined
                            ProductionStep::new(Symbol::terminal(12)),
                        ],
                    }],
                },
                SyntaxVariable {
                    name: "non-terminal-1".to_string(),
                    kind: VariableType::Named,
                    productions: vec![
                        Production {
                            dynamic_precedence: 0,
                            steps: vec![ProductionStep::new(Symbol::terminal(13))],
                        },
                        Production {
                            dynamic_precedence: 0,
                            steps: vec![
                                ProductionStep::new(Symbol::non_terminal(3)), // inlined
                                ProductionStep::new(Symbol::terminal(14)),
                            ],
                        },
                    ],
                },
                SyntaxVariable {
                    name: "non-terminal-2".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![ProductionStep::new(Symbol::terminal(15))],
                    }],
                },
                SyntaxVariable {
                    name: "non-terminal-3".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![ProductionStep::new(Symbol::terminal(16))],
                    }],
                },
            ],
            variables_to_inline: vec![
                Symbol::non_terminal(1),
                Symbol::non_terminal(2),
                Symbol::non_terminal(3),
            ],
            ..Default::default()
        };

        let (grammar, inline_map) = pool_pass(&grammar, &LexicalGrammar::default()).unwrap();

        let productions = inline_map
            .inlined_productions(&grammar.variables[0].productions[0], 1)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(
            productions.iter().copied().cloned().collect::<Vec<_>>(),
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(10)),
                        ProductionStep::new(Symbol::terminal(13)),
                        ProductionStep::new(Symbol::terminal(11)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::terminal(12)),
                    ],
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(10)),
                        ProductionStep::new(Symbol::terminal(16)),
                        ProductionStep::new(Symbol::terminal(14)),
                        ProductionStep::new(Symbol::terminal(11)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::terminal(12)),
                    ],
                },
            ]
        );

        assert_eq!(
            inline_map
                .inlined_productions(productions[0], 3)
                .unwrap()
                .cloned()
                .collect::<Vec<_>>(),
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::terminal(10)),
                    ProductionStep::new(Symbol::terminal(13)),
                    ProductionStep::new(Symbol::terminal(11)),
                    ProductionStep::new(Symbol::terminal(15)),
                    ProductionStep::new(Symbol::terminal(12)),
                ],
            },]
        );
    }

    #[test]
    fn test_inlining_with_precedence_and_alias() {
        let grammar = SyntaxGrammar {
            variables_to_inline: vec![Symbol::non_terminal(1), Symbol::non_terminal(2)],
            variables: vec![
                SyntaxVariable {
                    name: "non-terminal-0".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            // inlined
                            ProductionStep::new(Symbol::non_terminal(1))
                                .with_prec(Precedence::Integer(1), Some(Associativity::Left)),
                            ProductionStep::new(Symbol::terminal(10)),
                            // inlined
                            ProductionStep::new(Symbol::non_terminal(2))
                                .with_alias("outer_alias", true),
                        ],
                    }],
                },
                SyntaxVariable {
                    name: "non-terminal-1".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            ProductionStep::new(Symbol::terminal(11))
                                .with_prec(Precedence::Integer(2), None)
                                .with_alias("inner_alias", true),
                            ProductionStep::new(Symbol::terminal(12)),
                        ],
                    }],
                },
                SyntaxVariable {
                    name: "non-terminal-2".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![ProductionStep::new(Symbol::terminal(13))],
                    }],
                },
            ],
            ..Default::default()
        };

        let (grammar, inline_map) = pool_pass(&grammar, &LexicalGrammar::default()).unwrap();

        let productions = inline_map
            .inlined_productions(&grammar.variables[0].productions[0], 0)
            .unwrap()
            .collect::<Vec<_>>();

        assert_eq!(
            productions.iter().copied().cloned().collect::<Vec<_>>(),
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    // The first step in the inlined production retains its precedence
                    // and alias.
                    ProductionStep::new(Symbol::terminal(11))
                        .with_prec(Precedence::Integer(2), None)
                        .with_alias("inner_alias", true),
                    // The final step of the inlined production inherits the precedence of
                    // the inlined step.
                    ProductionStep::new(Symbol::terminal(12))
                        .with_prec(Precedence::Integer(1), Some(Associativity::Left)),
                    ProductionStep::new(Symbol::terminal(10)),
                    ProductionStep::new(Symbol::non_terminal(2)).with_alias("outer_alias", true),
                ]
            }],
        );

        assert_eq!(
            inline_map
                .inlined_productions(productions[0], 3)
                .unwrap()
                .cloned()
                .collect::<Vec<_>>(),
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::terminal(11))
                        .with_prec(Precedence::Integer(2), None)
                        .with_alias("inner_alias", true),
                    ProductionStep::new(Symbol::terminal(12))
                        .with_prec(Precedence::Integer(1), Some(Associativity::Left)),
                    ProductionStep::new(Symbol::terminal(10)),
                    // All steps of the inlined production inherit their alias from the
                    // inlined step.
                    ProductionStep::new(Symbol::terminal(13)).with_alias("outer_alias", true),
                ]
            }],
        );
    }

    #[test]
    fn test_error_when_inlining_tokens() {
        let lexical_grammar = LexicalGrammar {
            variables: vec![LexicalVariable {
                name: "something".to_string(),
                kind: VariableType::Named,
                implicit_precedence: 0,
                start_state: 0,
            }],
            ..Default::default()
        };

        let grammar = SyntaxGrammar {
            variables_to_inline: vec![Symbol::terminal(0)],
            variables: vec![SyntaxVariable {
                name: "non-terminal-0".to_string(),
                kind: VariableType::Named,
                productions: vec![Production {
                    dynamic_precedence: 0,
                    steps: vec![ProductionStep::new(Symbol::terminal(0))],
                }],
            }],
            ..Default::default()
        };

        let result = pool_pass(&grammar, &lexical_grammar);
        assert!(result.is_err(), "expected an error, but got none");
        let err = result.err().unwrap();
        assert_eq!(err.to_string(), "Token `something` cannot be inlined",);
    }

    #[test]
    fn test_errors() {
        let lexical_grammar = LexicalGrammar {
            variables: vec![LexicalVariable {
                name: "tok".to_string(),
                kind: VariableType::Named,
                implicit_precedence: 0,
                start_state: 0,
            }],
            ..Default::default()
        };
        for (inline, expected) in [
            (Symbol::terminal(0), "Token `tok` cannot be inlined"),
            (
                Symbol::external(0),
                "External token `ext` cannot be inlined",
            ),
            (
                Symbol::non_terminal(0),
                "Rule `rule0` cannot be inlined because it is the first rule",
            ),
        ] {
            let grammar = SyntaxGrammar {
                variables_to_inline: vec![inline],
                variables: vec![SyntaxVariable {
                    name: "rule0".to_string(),
                    kind: VariableType::Named,
                    productions: vec![Production::default()],
                }],
                external_tokens: vec![ExternalToken {
                    name: "ext".to_string(),
                    kind: VariableType::Named,
                    corresponding_internal_token: None,
                }],
                ..Default::default()
            };
            let err = pool_pass(&grammar, &lexical_grammar).unwrap_err();
            assert_eq!(err.to_string(), *expected);
        }
    }
}
