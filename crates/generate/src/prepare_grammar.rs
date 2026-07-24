mod expand_repeats;
mod expand_tokens;
mod extract_default_aliases;
mod extract_tokens;
mod flatten_grammar;
mod intern_symbols;
mod process_inlines;

use std::{
    cmp::Ordering,
    collections::{BTreeSet, hash_map},
    mem,
};

pub use expand_tokens::ExpandTokensError;
pub use extract_tokens::ExtractTokensError;
pub use flatten_grammar::FlattenGrammarError;
use indexmap::IndexMap;
pub use intern_symbols::InternSymbolsError;
pub use process_inlines::ProcessInlinesError;
use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use thiserror::Error;

pub use self::expand_tokens::expand_tokens;
use self::{
    expand_repeats::expand_repeats, extract_default_aliases::extract_default_aliases,
    extract_tokens::extract_tokens, flatten_grammar::flatten_grammar,
    intern_symbols::intern_symbols, process_inlines::process_inlines,
};
use super::{
    Diagnostic,
    grammars::{InlinedProductionMap, LexicalGrammar, SyntaxGrammar},
    prepare_grammar::flatten_grammar::{FlattenState, assemble_syntax_grammar},
    rule_pool::{FlatOut, Node, PoolGrammar, PoolPrecedenceEntry, Prec, StrId},
    rules::AliasMap,
};

pub type PrepareGrammarResult<T> = Result<T, PrepareGrammarError>;

#[derive(Debug, Error, Serialize, Deserialize)]
#[error(transparent)]
pub enum PrepareGrammarError {
    ValidatePrecedences(#[from] ValidatePrecedenceError),
    ValidateIndirectRecursion(#[from] IndirectRecursionError),
    InternSymbols(#[from] InternSymbolsError),
    ExtractTokens(#[from] ExtractTokensError),
    FlattenGrammar(#[from] FlattenGrammarError),
    ExpandTokens(#[from] ExpandTokensError),
    ProcessInlines(#[from] ProcessInlinesError),
}

pub type ValidatePrecedenceResult<T> = Result<T, ValidatePrecedenceError>;

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
#[error(transparent)]
pub enum ValidatePrecedenceError {
    Undeclared(#[from] UndeclaredPrecedenceError),
    Ordering(#[from] ConflictingPrecedenceOrderingError),
}

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndirectRecursionError(pub Vec<String>);

impl std::fmt::Display for IndirectRecursionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Grammar contains an indirectly recursive rule: ")?;
        for (i, symbol) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, " -> ")?;
            }
            write!(f, "{symbol}")?;
        }
        Ok(())
    }
}

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
#[error("Undeclared precedence '{}' in rule '{}'", self.precedence, self.rule)]
pub struct UndeclaredPrecedenceError {
    pub precedence: String,
    pub rule: String,
}

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
#[error("Conflicting orderings for precedences {} and {}", self.precedence_1, self.precedence_2)]
pub struct ConflictingPrecedenceOrderingError {
    pub precedence_1: String,
    pub precedence_2: String,
}

/// Transform an input grammar into separate components that are ready
/// for parse table construction.
pub fn prepare_grammar(
    mut g: PoolGrammar,
    diagnostics: &mut Vec<Diagnostic>,
) -> PrepareGrammarResult<(
    SyntaxGrammar,
    LexicalGrammar,
    InlinedProductionMap,
    AliasMap,
)> {
    validate_precedences(&g)?;
    validate_indirect_recursion(&g)?;

    let interned_meta = intern_symbols(&mut g, diagnostics)?;
    let mut ext_meta = extract_tokens(&mut g, &interned_meta)?;
    expand_repeats(&mut g, &mut ext_meta);

    let mut state = FlattenState::default();
    let mut out = FlatOut::default();
    flatten_grammar(&g, &ext_meta, &mut state, &mut out)?;

    let lexical_grammar = expand_tokens(
        &mut g.pool,
        &ext_meta.lexical_variables,
        &ext_meta.separator_roots,
    )?;

    let default_aliases = extract_default_aliases(&g, &ext_meta, &mut out);
    let inlines = process_inlines(&g, &ext_meta, &mut out)?;

    // Resolve the pooled default aliases into the string keyed map so that build tables
    // can stay untouched in this commit.
    let default_aliases: AliasMap = default_aliases
        .into_iter()
        .map(|(symbol, alias)| {
            (
                symbol,
                crate::rules::Alias {
                    value: g.pool.resolve(alias.value).to_string(),
                    is_named: alias.is_named,
                },
            )
        })
        .collect();

    let syntax_grammar = assemble_syntax_grammar(g, ext_meta, out);
    Ok((syntax_grammar, lexical_grammar, inlines, default_aliases))
}

/// Check for indirect recursion cycles in the grammar that can cause infinite loops while
/// parsing. An indirect recursion cycle occurs when a non-terminal can derive itself through
/// a chain of single-symbol productions (e.g., A -> B, B -> A).
fn validate_indirect_recursion(grammar: &PoolGrammar) -> Result<(), IndirectRecursionError> {
    let mut epsilon_transitions = IndexMap::new();
    let mut stack = Vec::new();
    for variable in &grammar.variables {
        let mut productions = BTreeSet::new();
        stack.clear();
        stack.push(variable.root);
        while let Some(id) = stack.pop() {
            match grammar.pool.node(id) {
                Node::NamedSymbol(sid) => {
                    // Rules that *directly* reference themselves don't cause a parsing loop.
                    if sid != variable.name {
                        productions.insert(sid);
                    }
                }
                Node::Choice(range) => stack.extend_from_slice(grammar.pool.child_slice(range)),
                Node::Metadata { rule, .. } => stack.push(rule),
                _ => {}
            }
        }
        epsilon_transitions.insert(variable.name, productions);
    }

    for &start_symbol in epsilon_transitions.keys() {
        let mut visited = BTreeSet::new();
        let mut path = Vec::new();
        if let Some((start_idx, end_idx)) =
            get_cycle(start_symbol, &epsilon_transitions, &mut visited, &mut path)
        {
            let cycle_symbols = path[start_idx..=end_idx]
                .iter()
                .map(|&s| grammar.pool.resolve(s).to_string())
                .collect();
            return Err(IndirectRecursionError(cycle_symbols));
        }
    }

    Ok(())
}

/// Perform a depth-first search to detect cycles in single state transitions.
fn get_cycle(
    current: StrId,
    transitions: &IndexMap<StrId, BTreeSet<StrId>>,
    visited: &mut BTreeSet<StrId>,
    path: &mut Vec<StrId>,
) -> Option<(usize, usize)> {
    if let Some(first_idx) = path.iter().position(|s| *s == current) {
        path.push(current);
        return Some((first_idx, path.len() - 1));
    }

    if visited.contains(&current) {
        return None;
    }

    path.push(current);
    visited.insert(current);

    if let Some(next_symbols) = transitions.get(&current) {
        for next in next_symbols {
            if let Some(cycle) = get_cycle(*next, transitions, visited, path) {
                return Some(cycle);
            }
        }
    }

    path.pop();
    None
}

/// Check that all of the named precedences used in the grammar are declared
/// within the `precedences` lists, and also that there are no conflicting
/// precedence orderings declared in those lists.
fn validate_precedences(grammar: &PoolGrammar) -> ValidatePrecedenceResult<()> {
    let display = |e: &PoolPrecedenceEntry| match *e {
        PoolPrecedenceEntry::Name(sid) => format!("'{}'", grammar.pool.resolve(sid)),
        PoolPrecedenceEntry::Symbol(sid) => format!("$.{}", grammar.pool.resolve(sid)),
    };

    // For any two precedence names `a` and `b`, if `a` comes before `b`
    // in some list, then it cannot come *after* `b` in any list.
    let mut pairs = FxHashMap::default();
    for list in &grammar.precedence_orderings {
        for (i, mut entry1) in list.iter().enumerate() {
            for mut entry2 in list.iter().skip(i + 1) {
                if entry1 == entry2 {
                    continue;
                }
                let mut ordering = Ordering::Greater;
                if entry1 > entry2 {
                    ordering = Ordering::Less;
                    mem::swap(&mut entry1, &mut entry2);
                }
                match pairs.entry((entry1, entry2)) {
                    hash_map::Entry::Vacant(e) => {
                        e.insert(ordering);
                    }
                    hash_map::Entry::Occupied(e) => {
                        if e.get() != &ordering {
                            Err(ConflictingPrecedenceOrderingError {
                                precedence_1: display(entry1),
                                precedence_2: display(entry2),
                            })?;
                        }
                    }
                };
            }
        }
    }

    let precedence_names = grammar
        .precedence_orderings
        .iter()
        .flatten()
        .filter_map(|p| match *p {
            PoolPrecedenceEntry::Name(sid) => Some(sid),
            PoolPrecedenceEntry::Symbol(_) => None,
        })
        .collect::<FxHashSet<_>>();

    let mut stack = Vec::new();
    for variable in &grammar.variables {
        stack.clear();
        stack.push(variable.root);
        while let Some(id) = stack.pop() {
            match grammar.pool.node(id) {
                Node::Repeat(inner) => stack.push(inner),
                Node::Seq(range) | Node::Choice(range) => {
                    stack.extend_from_slice(grammar.pool.child_slice(range));
                }
                Node::Metadata { params, rule } => {
                    if let Prec::Name(sid) = grammar.pool.params(params).prec
                        && !precedence_names.contains(&sid)
                    {
                        Err(UndeclaredPrecedenceError {
                            precedence: grammar.pool.resolve(sid).to_string(),
                            rule: grammar.pool.resolve(variable.name).to_string(),
                        })?;
                    }
                    stack.push(rule);
                }
                _ => {}
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rule_pool::{NodeId, PoolVariable, RulePool};

    #[test]
    fn test_validate_precedences_with_undeclared_precedences() {
        let mut pool = RulePool::default();

        // v1: seq(prec_left('b', "w"), prec('c', "x"))
        let v1 = {
            let w = leaf(&mut pool, "w");
            let left = prec_left(&mut pool, "b", w);
            let x = leaf(&mut pool, "x");
            let right = prec(&mut pool, "c", x);
            pool.seq(&[left, right])
        };
        // v2: repeat(choice(prec_left('omg', "y"), prec('c', "z")))
        let v2 = {
            let y = leaf(&mut pool, "y");
            let left = prec_left(&mut pool, "omg", y);
            let z = leaf(&mut pool, "z");
            let right = prec(&mut pool, "c", z);
            let choice = pool.choice(&[left, right]);
            pool.repeat(choice)
        };
        let v1_name = pool.intern("v1");
        let v2_name = pool.intern("v2");
        let precedence_orderings = vec![
            vec![name_entry(&mut pool, "a"), name_entry(&mut pool, "b")],
            vec![
                name_entry(&mut pool, "b"),
                name_entry(&mut pool, "c"),
                name_entry(&mut pool, "d"),
            ],
        ];

        let grammar = PoolGrammar {
            variables: vec![
                PoolVariable {
                    name: v1_name,
                    root: v1,
                },
                PoolVariable {
                    name: v2_name,
                    root: v2,
                },
            ],
            precedence_orderings,
            pool,
            ..Default::default()
        };

        assert_eq!(
            validate_precedences(&grammar).unwrap_err(),
            ValidatePrecedenceError::Undeclared(UndeclaredPrecedenceError {
                precedence: "omg".to_string(),
                rule: "v2".to_string()
            })
        );
    }

    #[test]
    fn test_validate_precedences_with_conflicting_order() {
        let mut pool = RulePool::default();
        let precedence_orderings = vec![
            vec![name_entry(&mut pool, "a"), name_entry(&mut pool, "b")],
            vec![
                name_entry(&mut pool, "b"),
                name_entry(&mut pool, "c"),
                name_entry(&mut pool, "a"),
            ],
        ];
        let grammar = PoolGrammar {
            precedence_orderings,
            pool,
            ..Default::default()
        };

        assert_eq!(
            validate_precedences(&grammar).unwrap_err(),
            ValidatePrecedenceError::Ordering(ConflictingPrecedenceOrderingError {
                precedence_1: "a".to_string(),
                precedence_2: "b".to_string()
            })
        );
    }

    #[test]
    fn test_validate_indirect_recursion() {
        // a -> b -> a
        let case1 = build_grammar(|p| {
            let b_ref = named(p, "b");
            let x = leaf(p, "x");
            let a = p.choice(&[b_ref, x]);
            let a_ref = named(p, "a");
            let b = p.prec(Prec::Integer(1), a_ref);
            vec![
                PoolVariable {
                    name: p.intern("a"),
                    root: a,
                },
                PoolVariable {
                    name: p.intern("b"),
                    root: b,
                },
            ]
        });
        // A direct self-reference is allowed.
        let case2 = build_grammar(|p| {
            let a = named(p, "a");
            vec![PoolVariable {
                name: p.intern("a"),
                root: a,
            }]
        });
        // b -> c -> d -> b, entered from a non-cycle start rule.
        let case3 = build_grammar(|p| {
            let x = leaf(p, "x");
            let c_ref = named(p, "c");
            let d_ref = named(p, "d");
            let b_ref = named(p, "b");
            vec![
                PoolVariable {
                    name: p.intern("a"),
                    root: x,
                },
                PoolVariable {
                    name: p.intern("b"),
                    root: c_ref,
                },
                PoolVariable {
                    name: p.intern("c"),
                    root: d_ref,
                },
                PoolVariable {
                    name: p.intern("d"),
                    root: b_ref,
                },
            ]
        });

        let err1 = IndirectRecursionError(vec!["a".to_string(), "b".to_string(), "a".to_string()]);
        let err3 = IndirectRecursionError(vec![
            "b".to_string(),
            "c".to_string(),
            "d".to_string(),
            "b".to_string(),
        ]);

        for (g, expected) in &[(case1, Err(err1)), (case2, Ok(())), (case3, Err(err3))] {
            assert_eq!(*expected, validate_indirect_recursion(g));
        }
    }

    fn named(pool: &mut RulePool, name: &str) -> NodeId {
        let id = pool.intern(name);
        pool.named_symbol(id)
    }
    fn leaf(pool: &mut RulePool, s: &str) -> NodeId {
        let id = pool.intern(s);
        pool.string(id)
    }
    fn prec(pool: &mut RulePool, name: &str, content: NodeId) -> NodeId {
        let p = Prec::Name(pool.intern(name));
        pool.prec(p, content)
    }
    fn prec_left(pool: &mut RulePool, name: &str, content: NodeId) -> NodeId {
        let p = Prec::Name(pool.intern(name));
        pool.prec_left(p, content)
    }
    fn name_entry(pool: &mut RulePool, name: &str) -> PoolPrecedenceEntry {
        PoolPrecedenceEntry::Name(pool.intern(name))
    }

    fn build_grammar(build: impl FnOnce(&mut RulePool) -> Vec<PoolVariable>) -> PoolGrammar {
        let mut pool = RulePool::default();
        let variables = build(&mut pool);
        PoolGrammar {
            pool,
            variables,
            ..Default::default()
        }
    }
}
