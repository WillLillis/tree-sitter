use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    grammars::{InlinedProductionMap, LexicalGrammar, Production, ProductionStep, SyntaxGrammar},
    rules::SymbolType,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct ProductionStepId {
    // A `None` value here means that the production itself was produced via inlining,
    // and is stored in the builder's `productions` vector, as opposed to being
    // stored in one of the grammar's variables.
    variable: Option<usize>,
    production: usize,
    step: usize,
}

struct InlinedProductionMapBuilder {
    production_indices_by_step_id: FxHashMap<ProductionStepId, Vec<usize>>,
    productions: Vec<Production>,
}

impl InlinedProductionMapBuilder {
    fn build(mut self, grammar: &SyntaxGrammar) -> InlinedProductionMap {
        let mut step_ids_to_process = Vec::new();
        for (variable_index, variable) in grammar.variables.iter().enumerate() {
            for production_index in 0..variable.productions.len() {
                step_ids_to_process.push(ProductionStepId {
                    variable: Some(variable_index),
                    production: production_index,
                    step: 0,
                });
                while !step_ids_to_process.is_empty() {
                    let mut i = 0;
                    while i < step_ids_to_process.len() {
                        let step_id = step_ids_to_process[i];
                        if let Some(step) = self.production_step_for_id(step_id, grammar) {
                            if grammar.variables_to_inline.contains(&step.symbol) {
                                let inlined_step_ids = self
                                    .inline_production_at_step(step_id, grammar)
                                    .iter()
                                    .copied()
                                    .map(|production_index| ProductionStepId {
                                        variable: None,
                                        production: production_index,
                                        step: step_id.step,
                                    });
                                step_ids_to_process.splice(i..=i, inlined_step_ids);
                            } else {
                                step_ids_to_process[i] = ProductionStepId {
                                    variable: step_id.variable,
                                    production: step_id.production,
                                    step: step_id.step + 1,
                                };
                                i += 1;
                            }
                        } else {
                            step_ids_to_process.remove(i);
                        }
                    }
                }
            }
        }

        let productions = self.productions;
        let production_indices_by_step_id = self.production_indices_by_step_id;
        let production_map = production_indices_by_step_id
            .into_iter()
            .map(|(step_id, production_indices)| {
                let production = core::ptr::from_ref::<Production>(step_id.variable.map_or_else(
                    || &productions[step_id.production],
                    |variable_index| {
                        &grammar.variables[variable_index].productions[step_id.production]
                    },
                ));
                ((production, step_id.step as u32), production_indices)
            })
            .collect();

        InlinedProductionMap {
            productions,
            production_map,
        }
    }

    fn inline_production_at_step<'a>(
        &'a mut self,
        step_id: ProductionStepId,
        grammar: &'a SyntaxGrammar,
    ) -> &'a [usize] {
        // Build a list of productions produced by inlining rules.
        let mut i = 0;
        let step_index = step_id.step;
        let mut productions_to_add = vec![self.production_for_id(step_id, grammar).clone()];
        while i < productions_to_add.len() {
            if let Some(step) = productions_to_add[i].steps.get(step_index) {
                let symbol = step.symbol;
                if grammar.variables_to_inline.contains(&symbol) {
                    // Remove the production from the vector, replacing it with a placeholder.
                    let production = productions_to_add
                        .splice(i..=i, std::iter::once(&Production::default()).cloned())
                        .next()
                        .unwrap();

                    // Replace the placeholder with the inlined productions.
                    productions_to_add.splice(
                        i..=i,
                        grammar.variables[symbol.index].productions.iter().map(|p| {
                            let mut production = production.clone();
                            let removed_step = production
                                .steps
                                .splice(step_index..=step_index, p.steps.iter().cloned())
                                .next()
                                .unwrap();
                            let inserted_steps =
                                &mut production.steps[step_index..(step_index + p.steps.len())];
                            if let Some(alias) = removed_step.alias {
                                for inserted_step in inserted_steps.iter_mut() {
                                    inserted_step.alias = Some(alias.clone());
                                }
                            }
                            if let Some(field_name) = removed_step.field_name {
                                for inserted_step in inserted_steps.iter_mut() {
                                    inserted_step.field_name = Some(field_name.clone());
                                }
                            }
                            if let Some(last_inserted_step) = inserted_steps.last_mut() {
                                if last_inserted_step.precedence.is_none() {
                                    last_inserted_step.precedence = removed_step.precedence;
                                }
                                if last_inserted_step.associativity.is_none() {
                                    last_inserted_step.associativity = removed_step.associativity;
                                }
                            }
                            if p.dynamic_precedence.abs() > production.dynamic_precedence.abs() {
                                production.dynamic_precedence = p.dynamic_precedence;
                            }
                            production
                        }),
                    );

                    continue;
                }
            }
            i += 1;
        }

        // Store all the computed productions.
        let result = productions_to_add
            .into_iter()
            .map(|production| {
                self.productions
                    .iter()
                    .position(|p| *p == production)
                    .unwrap_or_else(|| {
                        self.productions.push(production);
                        self.productions.len() - 1
                    })
            })
            .collect();

        // Cache these productions based on the original production step.
        self.production_indices_by_step_id
            .entry(step_id)
            .or_insert(result)
    }

    fn production_for_id<'a>(
        &'a self,
        id: ProductionStepId,
        grammar: &'a SyntaxGrammar,
    ) -> &'a Production {
        id.variable.map_or_else(
            || &self.productions[id.production],
            |variable_index| &grammar.variables[variable_index].productions[id.production],
        )
    }

    fn production_step_for_id<'a>(
        &'a self,
        id: ProductionStepId,
        grammar: &'a SyntaxGrammar,
    ) -> Option<&'a ProductionStep> {
        self.production_for_id(id, grammar).steps.get(id.step)
    }
}

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

pub(super) fn process_inlines(
    grammar: &SyntaxGrammar,
    lexical_grammar: &LexicalGrammar,
) -> ProcessInlinesResult<InlinedProductionMap> {
    for symbol in &grammar.variables_to_inline {
        match symbol.kind {
            SymbolType::External => {
                Err(ProcessInlinesError::ExternalToken(
                    grammar.external_tokens[symbol.index].name.clone(),
                ))?;
            }
            SymbolType::Terminal => {
                Err(ProcessInlinesError::Token(
                    lexical_grammar.variables[symbol.index].name.clone(),
                ))?;
            }
            SymbolType::NonTerminal if symbol.index == 0 => {
                Err(ProcessInlinesError::FirstRule(
                    grammar.variables[symbol.index].name.clone(),
                ))?;
            }
            _ => {}
        }
    }

    Ok(InlinedProductionMapBuilder {
        productions: Vec::new(),
        production_indices_by_step_id: FxHashMap::default(),
    }
    .build(grammar))
}

// --- pool-based pass (replaces the Rule-based one at the container flip) ---

use std::collections::BTreeMap;
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

/// Pool twin of `process_inlines`: the same worklist over `Copy` step
/// buffers, with the O(created^2) structural dedup scan replaced by a
/// content-hash memo and the address-keyed map replaced by production ids.
pub(super) fn process_inlines_pool(
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
    /// creation order so the first structural match mirrors master's
    /// first-position scan.
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

    /// Mirror of `inline_production_at_step` on `Copy` step buffers; also
    /// expands nested inlining at the same step index before storing.
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

/// Canonical production identity for A/B comparison (the master map is
/// address-keyed): grammar productions as `(0, variable, production)`,
/// created ones as `(1, created index, 0)`. Deleted at the container flip.
type CanonInlines = BTreeMap<((u8, u32, u32), u32), Vec<u32>>;

pub(super) fn canon_pool_inlines(inl: &PoolInlines, out: &FlatOut) -> CanonInlines {
    let first = inl.first_inlined;
    let mut flat_to_var = vec![(0u32, 0u32); first as usize];
    for (v, &(ps, pe)) in out.var_prods.iter().enumerate() {
        for (p, id) in (ps..pe).enumerate() {
            flat_to_var[id as usize] = (v as u32, p as u32);
        }
    }
    let ident = |id: u32| {
        if id < first {
            let (v, p) = flat_to_var[id as usize];
            (0u8, v, p)
        } else {
            (1u8, id - first, 0)
        }
    };
    inl.map
        .iter()
        .map(|(&(id, step), ids)| {
            (
                (ident(id), step),
                ids.iter().map(|&id| id - first).collect(),
            )
        })
        .collect()
}

pub(super) fn canon_master_inlines(
    map: &InlinedProductionMap,
    grammar: &SyntaxGrammar,
) -> CanonInlines {
    let mut by_addr: FxHashMap<*const Production, (u8, u32, u32)> = FxHashMap::default();
    for (v, variable) in grammar.variables.iter().enumerate() {
        for (p, production) in variable.productions.iter().enumerate() {
            by_addr.insert(std::ptr::from_ref(production), (0, v as u32, p as u32));
        }
    }
    for (i, production) in map.productions.iter().enumerate() {
        by_addr.insert(std::ptr::from_ref(production), (1, i as u32, 0));
    }
    map.production_map
        .iter()
        .map(|(&(addr, step), ids)| {
            (
                (by_addr[&addr], step),
                ids.iter().map(|&id| id as u32).collect(),
            )
        })
        .collect()
}

/// A/B materialization of the created productions. Deleted at the flip.
pub(super) fn materialize_inlined_productions(
    pool: &RulePool,
    out: &FlatOut,
    first_inlined: u32,
) -> Vec<Production> {
    out.productions[first_inlined as usize..]
        .iter()
        .map(|p| Production {
            dynamic_precedence: p.dynamic_precedence,
            steps: out.steps[p.step_range()]
                .iter()
                .map(|s| super::flatten_grammar::step_to_master(pool, s))
                .collect(),
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammars::{ExternalToken, LexicalVariable, SyntaxVariable, VariableType},
        rules::{Associativity, Precedence, Symbol},
    };

    fn assert_pool_matches_master(grammar: &SyntaxGrammar, lexical_grammar: &LexicalGrammar) {
        let master = process_inlines(grammar, lexical_grammar).unwrap();
        let (g, meta, mut out) = super::super::pool_from_syntax(grammar, lexical_grammar);
        let inl = process_inlines_pool(&g, &meta, &mut out).unwrap();
        assert_eq!(
            materialize_inlined_productions(&g.pool, &out, inl.first_inlined),
            master.productions
        );
        assert_eq!(
            canon_pool_inlines(&inl, &out),
            canon_master_inlines(&master, grammar)
        );
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

        assert_pool_matches_master(&grammar, &LexicalGrammar::default());
        let inline_map = process_inlines(&grammar, &LexicalGrammar::default()).unwrap();

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

        assert_pool_matches_master(&grammar, &LexicalGrammar::default());
        let inline_map = process_inlines(&grammar, &LexicalGrammar::default()).unwrap();

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

        assert_pool_matches_master(&grammar, &LexicalGrammar::default());
        let inline_map = process_inlines(&grammar, &LexicalGrammar::default()).unwrap();

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

        let result = process_inlines(&grammar, &lexical_grammar);
        assert!(result.is_err(), "expected an error, but got none");
        let err = result.err().unwrap();
        assert_eq!(err.to_string(), "Token `something` cannot be inlined",);
    }

    #[test]
    fn pool_errors_match_master() {
        let lexical_grammar = LexicalGrammar {
            variables: vec![LexicalVariable {
                name: "tok".to_string(),
                kind: VariableType::Named,
                implicit_precedence: 0,
                start_state: 0,
            }],
            ..Default::default()
        };
        for inline in [
            Symbol::terminal(0),
            Symbol::external(0),
            Symbol::non_terminal(0),
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
            let master_err = process_inlines(&grammar, &lexical_grammar).unwrap_err();
            let (g, meta, mut out) = super::super::pool_from_syntax(&grammar, &lexical_grammar);
            let pool_err = process_inlines_pool(&g, &meta, &mut out).unwrap_err();
            assert_eq!(pool_err.to_string(), master_err.to_string());
        }
    }
}
