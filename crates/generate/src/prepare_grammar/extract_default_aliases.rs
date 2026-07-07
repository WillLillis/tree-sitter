#[cfg(test)]
use crate::grammars::{LexicalGrammar, SyntaxGrammar};
use crate::rules::{Alias, AliasMap, Symbol, SymbolType};

use std::collections::BTreeMap;

use super::extract_tokens::PoolExtractedMeta;
use crate::rule_pool::{Alias as PoolAlias, FSTEP_ALIAS_NAMED, FlatOut, PoolGrammar, RulePool};

#[derive(Clone, Default)]
struct PoolSymbolStatus {
    aliases: Vec<(PoolAlias, usize)>,
    appears_unaliased: bool,
}

/// Find symbols that always appear aliased and promote their most frequent
/// alias to a global default, clearing the now-redundant step aliases in
/// place. Alias identity is the interned `(StrId, is_named)` pair; clearing
/// zeroes both the id and the named bit to keep steps canonical for
/// `==`/`Hash` dedup.
pub(super) fn extract_default_aliases(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    out: &mut FlatOut,
) -> BTreeMap<Symbol, PoolAlias> {
    let mut terminal_status_list = vec![PoolSymbolStatus::default(); meta.lexical_variables.len()];
    let mut non_terminal_status_list = vec![PoolSymbolStatus::default(); g.variables.len()];
    let mut external_status_list = vec![PoolSymbolStatus::default(); meta.external_tokens.len()];

    for prod in &out.productions[..] {
        for step in &out.steps[prod.step_range()] {
            let symbol = step.symbol();
            let status = match symbol.kind {
                SymbolType::External => &mut external_status_list[symbol.index],
                SymbolType::NonTerminal => &mut non_terminal_status_list[symbol.index],
                SymbolType::Terminal => &mut terminal_status_list[symbol.index],
                SymbolType::End | SymbolType::EndOfNonTerminalExtra => {
                    panic!("Unexpected end token")
                }
            };

            // Default aliases don't work for inlined variables.
            if meta.inline.contains(&symbol) {
                continue;
            }

            if let Some(alias) = step.alias() {
                if let Some(count_for_alias) = status
                    .aliases
                    .iter_mut()
                    .find_map(|(a, count)| (*a == alias).then_some(count))
                {
                    *count_for_alias += 1;
                } else {
                    status.aliases.push((alias, 1));
                }
            } else {
                status.appears_unaliased = true;
            }
        }
    }

    for symbol in &meta.extra_symbols {
        let status = match symbol.kind {
            SymbolType::External => &mut external_status_list[symbol.index],
            SymbolType::NonTerminal => &mut non_terminal_status_list[symbol.index],
            SymbolType::Terminal => &mut terminal_status_list[symbol.index],
            SymbolType::End | SymbolType::EndOfNonTerminalExtra => panic!("Unexpected end token"),
        };
        status.appears_unaliased = true;
    }

    let symbols_with_statuses = (terminal_status_list
        .iter_mut()
        .enumerate()
        .map(|(i, status)| (Symbol::terminal(i), status)))
    .chain(
        non_terminal_status_list
            .iter_mut()
            .enumerate()
            .map(|(i, status)| (Symbol::non_terminal(i), status)),
    )
    .chain(
        external_status_list
            .iter_mut()
            .enumerate()
            .map(|(i, status)| (Symbol::external(i), status)),
    );

    let mut result = BTreeMap::new();
    for (symbol, status) in symbols_with_statuses {
        if status.appears_unaliased {
            status.aliases.clear();
        } else if let Some(default_entry) = status
            .aliases
            .iter()
            .enumerate()
            .max_by_key(|(i, (_, count))| (count, -(*i as i64)))
            .map(|(_, entry)| *entry)
        {
            status.aliases.clear();
            status.aliases.push(default_entry);
            result.insert(symbol, default_entry.0);
        }
    }

    let mut alias_positions_to_clear = Vec::new();
    for &(ps, pe) in &out.var_prods {
        alias_positions_to_clear.clear();

        let productions = &out.productions[ps as usize..pe as usize];
        for (i, prod) in productions.iter().enumerate() {
            for (j, step) in out.steps[prod.step_range()].iter().enumerate() {
                let symbol = step.symbol();
                let status = match symbol.kind {
                    SymbolType::External => &external_status_list[symbol.index],
                    SymbolType::NonTerminal => &non_terminal_status_list[symbol.index],
                    SymbolType::Terminal => &terminal_status_list[symbol.index],
                    SymbolType::End | SymbolType::EndOfNonTerminalExtra => {
                        panic!("Unexpected end token")
                    }
                };

                // If this step is aliased as the symbol's default alias, then remove that alias.
                if step.alias().is_some() && step.alias() == status.aliases.first().map(|t| t.0) {
                    let mut other_productions_must_use_this_alias_at_this_index = false;
                    for (other_i, other_prod) in productions.iter().enumerate() {
                        let other_steps = &out.steps[other_prod.step_range()];
                        if other_i != i
                            && other_steps.len() > j
                            && other_steps[j].alias() == step.alias()
                            && result.get(&other_steps[j].symbol()) != step.alias().as_ref()
                        {
                            other_productions_must_use_this_alias_at_this_index = true;
                            break;
                        }
                    }

                    if !other_productions_must_use_this_alias_at_this_index {
                        alias_positions_to_clear.push(prod.steps_start as usize + j);
                    }
                }
            }
        }

        for &step_index in &alias_positions_to_clear {
            let step = &mut out.steps[step_index];
            step.alias = 0;
            step.flags &= !FSTEP_ALIAS_NAMED;
        }
    }

    result
}

/// Boundary materialization to the render-facing [`AliasMap`]. Deleted at
/// the `SyntaxGrammar` flip.
pub(super) fn materialize_default_aliases(
    pool: &RulePool,
    map: &BTreeMap<Symbol, PoolAlias>,
) -> AliasMap {
    map.iter()
        .map(|(&symbol, alias)| {
            (
                symbol,
                Alias {
                    value: pool.resolve(alias.value).to_string(),
                    is_named: alias.is_named,
                },
            )
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammars::{LexicalVariable, Production, ProductionStep, SyntaxVariable, VariableType},
        nfa::Nfa,
    };

    #[test]
    fn test_extract_simple_aliases() {
        let mut syntax_grammar = SyntaxGrammar {
            variables: vec![
                SyntaxVariable {
                    name: "v1".to_owned(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            ProductionStep::new(Symbol::terminal(0)).with_alias("a1", true),
                            ProductionStep::new(Symbol::terminal(1)).with_alias("a2", true),
                            ProductionStep::new(Symbol::terminal(2)).with_alias("a3", true),
                            ProductionStep::new(Symbol::terminal(3)).with_alias("a4", true),
                        ],
                    }],
                },
                SyntaxVariable {
                    name: "v2".to_owned(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            // Token 0 is always aliased as "a1".
                            ProductionStep::new(Symbol::terminal(0)).with_alias("a1", true),
                            // Token 1 is aliased within rule `v1` above, but not here.
                            ProductionStep::new(Symbol::terminal(1)),
                            // Token 2 is aliased differently here than in `v1`. The alias from
                            // `v1` should be promoted to the default alias, because `v1` appears
                            // first in the grammar.
                            ProductionStep::new(Symbol::terminal(2)).with_alias("a5", true),
                            // Token 3 is also aliased differently here than in `v1`. In this case,
                            // this alias should be promoted to the default alias, because it is
                            // used a greater number of times (twice).
                            ProductionStep::new(Symbol::terminal(3)).with_alias("a6", true),
                            ProductionStep::new(Symbol::terminal(3)).with_alias("a6", true),
                        ],
                    }],
                },
            ],
            ..Default::default()
        };

        let lexical_grammar = LexicalGrammar {
            nfa: Nfa::new(),
            variables: vec![
                LexicalVariable {
                    name: "t0".to_string(),
                    kind: VariableType::Anonymous,
                    implicit_precedence: 0,
                    start_state: 0,
                },
                LexicalVariable {
                    name: "t1".to_string(),
                    kind: VariableType::Anonymous,
                    implicit_precedence: 0,
                    start_state: 0,
                },
                LexicalVariable {
                    name: "t2".to_string(),
                    kind: VariableType::Anonymous,
                    implicit_precedence: 0,
                    start_state: 0,
                },
                LexicalVariable {
                    name: "t3".to_string(),
                    kind: VariableType::Anonymous,
                    implicit_precedence: 0,
                    start_state: 0,
                },
            ],
        };

        let (g, meta, mut out) = super::super::pool_from_syntax(&syntax_grammar, &lexical_grammar);
        let pool_map = extract_default_aliases(&g, &meta, &mut out);
        let default_aliases = materialize_default_aliases(&g.pool, &pool_map);
        // The mutated grammar is compared through the same materialization
        // the pipeline uses.
        syntax_grammar = super::super::flatten_grammar::materialize_flattened(&g, &meta, &out);
        assert_eq!(default_aliases.len(), 3);

        assert_eq!(
            default_aliases.get(&Symbol::terminal(0)),
            Some(&Alias {
                value: "a1".to_string(),
                is_named: true,
            })
        );
        assert_eq!(
            default_aliases.get(&Symbol::terminal(2)),
            Some(&Alias {
                value: "a3".to_string(),
                is_named: true,
            })
        );
        assert_eq!(
            default_aliases.get(&Symbol::terminal(3)),
            Some(&Alias {
                value: "a6".to_string(),
                is_named: true,
            })
        );
        assert_eq!(default_aliases.get(&Symbol::terminal(1)), None);

        assert_eq!(
            syntax_grammar.variables,
            vec![
                SyntaxVariable {
                    name: "v1".to_owned(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            ProductionStep::new(Symbol::terminal(0)),
                            ProductionStep::new(Symbol::terminal(1)).with_alias("a2", true),
                            ProductionStep::new(Symbol::terminal(2)),
                            ProductionStep::new(Symbol::terminal(3)).with_alias("a4", true),
                        ],
                    },],
                },
                SyntaxVariable {
                    name: "v2".to_owned(),
                    kind: VariableType::Named,
                    productions: vec![Production {
                        dynamic_precedence: 0,
                        steps: vec![
                            ProductionStep::new(Symbol::terminal(0)),
                            ProductionStep::new(Symbol::terminal(1)),
                            ProductionStep::new(Symbol::terminal(2)).with_alias("a5", true),
                            ProductionStep::new(Symbol::terminal(3)),
                            ProductionStep::new(Symbol::terminal(3)),
                        ],
                    },],
                },
            ]
        );
    }
}
