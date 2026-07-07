use rustc_hash::FxHashMap;

#[cfg(test)]
use super::ExtractedSyntaxGrammar;
#[cfg(test)]
use crate::{grammars::Variable, rules::Rule};
use crate::{grammars::VariableType, rules::Symbol};

use super::extract_tokens::PoolExtractedMeta;
use crate::rule_pool::{Node as PoolNode, NodeId, PoolGrammar, PoolVariable, RulePool, StrId};

const fn pool_sym(symbol: Symbol) -> PoolNode {
    PoolNode::Sym {
        kind: symbol.kind,
        index: symbol.index as u32,
    }
}

/// Mirror of `Rule::choice(vec![seq(sym, sym), inner])`: recursive flatten of
/// nested choices, structural `contains` dedup, singleton unwrap.
fn wrap_in_binary_tree(pool: &mut RulePool, symbol: Symbol, inner: NodeId) -> NodeId {
    let s1 = pool.push_node(pool_sym(symbol));
    let s2 = pool.push_node(pool_sym(symbol));
    let range = pool.push_children(&[s1, s2]);
    let seq = pool.push_node(PoolNode::Seq(range));
    let mut elements = vec![seq];
    let mut stack = vec![inner];
    while let Some(id) = stack.pop() {
        if let PoolNode::Choice(range) = pool.node(id) {
            let base = stack.len();
            stack.extend_from_slice(pool.child_slice(range));
            stack[base..].reverse();
        } else if !elements.iter().any(|&e| pool.subtree_eq(e, id)) {
            elements.push(id);
        }
    }
    if elements.len() == 1 {
        elements[0]
    } else {
        let range = pool.push_children(&elements);
        pool.push_node(PoolNode::Choice(range))
    }
}

enum Task {
    Visit(NodeId),
    Expand(NodeId),
}

/// Post-order repeat expansion over one root: children expand first, fixing
/// the memo keys and aux naming order. `Reserved` nodes are not descended.
#[expect(clippy::too_many_arguments, reason = "transitional free function")]
fn expand_root(
    pool: &mut RulePool,
    root: NodeId,
    var_name: StrId,
    counter: &mut u32,
    preceding: usize,
    aux: &mut Vec<PoolVariable>,
    memo: &mut FxHashMap<u64, Vec<(NodeId, Symbol)>>,
    stack: &mut Vec<Task>,
) {
    stack.clear();
    stack.push(Task::Visit(root));
    'walk: while let Some(task) = stack.pop() {
        match task {
            Task::Visit(id) => match pool.node(id) {
                PoolNode::Repeat(content) => {
                    stack.push(Task::Expand(id));
                    stack.push(Task::Visit(content));
                }
                PoolNode::Seq(range) | PoolNode::Choice(range) => {
                    let base = stack.len();
                    for &c in pool.child_slice(range) {
                        stack.push(Task::Visit(c));
                    }
                    stack[base..].reverse();
                }
                PoolNode::Metadata { rule, .. } => stack.push(Task::Visit(rule)),
                _ => {}
            },
            Task::Expand(id) => {
                let PoolNode::Repeat(content) = pool.node(id) else {
                    unreachable!()
                };
                let hash = pool.subtree_hash(content);
                if let Some(candidates) = memo.get(&hash) {
                    for &(node, symbol) in candidates {
                        if pool.subtree_eq(node, content) {
                            pool.set_node(id, pool_sym(symbol));
                            continue 'walk;
                        }
                    }
                }
                *counter += 1;
                let name = format!("{}_repeat{}", pool.resolve(var_name), counter);
                let name = pool.intern(&name);
                let symbol = Symbol::non_terminal(preceding + aux.len());
                memo.entry(hash).or_default().push((content, symbol));
                let root = wrap_in_binary_tree(pool, symbol, content);
                aux.push(PoolVariable { name, root });
                pool.set_node(id, pool_sym(symbol));
            }
        }
    }
}

pub(super) fn expand_repeats(g: &mut PoolGrammar, meta: &mut PoolExtractedMeta) {
    let preceding = g.variables.len();
    let mut aux: Vec<PoolVariable> = Vec::new();
    let mut memo: FxHashMap<u64, Vec<(NodeId, Symbol)>> = FxHashMap::default();
    let mut stack = Vec::new();
    for i in 0..g.variables.len() {
        let PoolVariable { name, root } = g.variables[i];
        let mut counter = 0u32;

        // A hidden variable with a top-level repetition becomes its own
        // recursive binary tree instead of gaining an auxiliary rule, and
        // can no longer be inlined.
        if meta.kinds[i] == VariableType::Hidden
            && let PoolNode::Repeat(content) = g.pool.node(root)
        {
            expand_root(
                &mut g.pool,
                content,
                name,
                &mut counter,
                preceding,
                &mut aux,
                &mut memo,
                &mut stack,
            );
            g.variables[i].root =
                wrap_in_binary_tree(&mut g.pool, Symbol::non_terminal(i), content);
            meta.kinds[i] = VariableType::Auxiliary;
            meta.inline.retain(|s| *s != Symbol::non_terminal(i));
            continue;
        }

        expand_root(
            &mut g.pool,
            root,
            name,
            &mut counter,
            preceding,
            &mut aux,
            &mut memo,
            &mut stack,
        );
    }
    for var in aux {
        g.variables.push(var);
        meta.kinds.push(VariableType::Auxiliary);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_repeat_expansion() {
        // Repeats nested inside of sequences and choices are expanded.
        let grammar = pool_pass(&build_grammar(vec![Variable::named(
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
        let grammar = pool_pass(&build_grammar(vec![
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
        let grammar = pool_pass(&build_grammar(vec![Variable::named(
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
        let grammar = pool_pass(&build_grammar(vec![
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

    fn pool_from_extracted(g: &ExtractedSyntaxGrammar) -> (PoolGrammar, PoolExtractedMeta) {
        use crate::rule_pool::RulePool;
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

    fn pool_pass(g: &ExtractedSyntaxGrammar) -> ExtractedSyntaxGrammar {
        let (mut pg, mut meta) = pool_from_extracted(g);
        expand_repeats(&mut pg, &mut meta);
        super::super::extract_tokens::materialize_extracted(&pg, &meta, &pg.precedence_orderings).0
    }
}
