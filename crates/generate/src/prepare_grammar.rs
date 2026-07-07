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

#[cfg(test)]
pub use self::expand_tokens::expand_token_rules;
#[cfg(test)]
use super::{
    grammars::{ExternalToken, ReservedWordContext, Variable},
    rules::{Precedence, Rule, Symbol},
};
use super::{
    grammars::{
        InlinedProductionMap, InputGrammar, LexicalGrammar, PrecedenceEntry, SyntaxGrammar,
    },
    rules::AliasMap,
};
use crate::Diagnostic;

/// Test-only since the pipeline flip: the pool passes carry this state as
/// `PoolGrammar` + pass metadata; tests still assert against this shape via
/// the `materialize_*` helpers. Deleted at the frontend flip.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IntermediateGrammar<T, U> {
    variables: Vec<Variable>,
    extra_symbols: Vec<T>,
    expected_conflicts: Vec<Vec<Symbol>>,
    precedence_orderings: Vec<Vec<PrecedenceEntry>>,
    external_tokens: Vec<U>,
    variables_to_inline: Vec<Symbol>,
    supertype_symbols: Vec<Symbol>,
    word_token: Option<Symbol>,
    reserved_word_sets: Vec<ReservedWordContext<T>>,
}

#[cfg(test)]
pub type InternedGrammar = IntermediateGrammar<Rule, Variable>;

#[cfg(test)]
pub type ExtractedSyntaxGrammar = IntermediateGrammar<Symbol, ExternalToken>;

#[cfg(test)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExtractedLexicalGrammar {
    pub variables: Vec<Variable>,
    pub separators: Vec<Rule>,
}

#[cfg(test)]
impl<T, U> Default for IntermediateGrammar<T, U> {
    fn default() -> Self {
        Self {
            variables: Vec::default(),
            extra_symbols: Vec::default(),
            expected_conflicts: Vec::default(),
            precedence_orderings: Vec::default(),
            external_tokens: Vec::default(),
            variables_to_inline: Vec::default(),
            supertype_symbols: Vec::default(),
            word_token: Option::default(),
            reserved_word_sets: Vec::default(),
        }
    }
}

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

#[derive(Debug, Error, Serialize, Deserialize)]
#[error(transparent)]
pub enum ValidatePrecedenceError {
    Undeclared(#[from] UndeclaredPrecedenceError),
    Ordering(#[from] ConflictingPrecedenceOrderingError),
}

#[derive(Debug, Error, Serialize, Deserialize)]
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

#[derive(Debug, Error, Serialize, Deserialize)]
pub struct UndeclaredPrecedenceError {
    pub precedence: String,
    pub rule: String,
}

impl std::fmt::Display for UndeclaredPrecedenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Undeclared precedence '{}' in rule '{}'",
            self.precedence, self.rule
        )?;
        Ok(())
    }
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub struct ConflictingPrecedenceOrderingError {
    pub precedence_1: String,
    pub precedence_2: String,
}

impl std::fmt::Display for ConflictingPrecedenceOrderingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Conflicting orderings for precedences {} and {}",
            self.precedence_1, self.precedence_2
        )?;
        Ok(())
    }
}

/// Transform an input grammar into separate components that are ready
/// for parse table construction.
pub fn prepare_grammar(
    input_grammar: &InputGrammar,
    diagnostics: &mut Vec<Diagnostic>,
) -> PrepareGrammarResult<(
    SyntaxGrammar,
    LexicalGrammar,
    InlinedProductionMap,
    AliasMap,
)> {
    // TEMP (until the frontend flip): the frontends still produce Rule-based
    // grammars; convert once at the boundary.
    let mut g = crate::rule_pool::PoolGrammar::from_input_grammar(input_grammar);

    let t0 = std::time::Instant::now();
    validate_precedences(&g)?;
    validate_indirect_recursion(&g)?;
    let t_validate = t0.elapsed();

    let t0 = std::time::Instant::now();
    let interned_meta = intern_symbols::intern_symbols(&mut g, diagnostics)?;
    let t_intern = t0.elapsed();

    let t0 = std::time::Instant::now();
    let mut ext_meta = extract_tokens::extract_tokens(&mut g, &interned_meta)?;
    let t_extract = t0.elapsed();

    let t0 = std::time::Instant::now();
    expand_repeats::expand_repeats(&mut g, &mut ext_meta);
    let t_expand = t0.elapsed();

    let t0 = std::time::Instant::now();
    let mut state = flatten_grammar::FlattenState::default();
    let mut out = crate::rule_pool::FlatOut::default();
    flatten_grammar::flatten_grammar(&g, &ext_meta, &mut state, &mut out)?;
    let t_flatten = t0.elapsed();

    let t0 = std::time::Instant::now();
    let lexical_grammar = expand_tokens::expand_tokens(
        &mut g.pool,
        &ext_meta.lexical_variables,
        &ext_meta.separator_roots,
    )?;
    let t_tokens = t0.elapsed();

    let t0 = std::time::Instant::now();
    let default_aliases = extract_default_aliases::extract_default_aliases(&g, &ext_meta, &mut out);
    let inlines = process_inlines::process_inlines(&g, &ext_meta, &mut out)?;
    let t_rest = t0.elapsed();

    // TEMP (until the `SyntaxGrammar` flip): materialize the legacy-shaped
    // outputs once at the boundary for build_tables/node_types/render.
    let default_aliases =
        extract_default_aliases::materialize_default_aliases(&g.pool, &default_aliases);
    let syntax_grammar = flatten_grammar::materialize_flattened(&g, &ext_meta, &out);
    let inlines = process_inlines::materialize_inlines(&g.pool, &out, &inlines, &syntax_grammar);

    // TEMP SPIKE: real single-shot stage times.
    eprintln!(
        "[SPIKE prepare-stages] validate {t_validate:?} intern {t_intern:?} extract {t_extract:?} expand {t_expand:?} flatten {t_flatten:?} expand_tokens {t_tokens:?} aliases+inlines {t_rest:?}"
    );
    Ok((syntax_grammar, lexical_grammar, inlines, default_aliases))
}
/// Perform a depth-first search to detect cycles in single state transitions.
fn get_cycle<'a>(
    current: &'a str,
    transitions: &'a IndexMap<&'a str, BTreeSet<String>>,
    visited: &mut BTreeSet<&'a str>,
    path: &mut Vec<&'a str>,
) -> Option<(usize, usize)> {
    if let Some(first_idx) = path.iter().position(|s| *s == current) {
        path.push(current);
        return Some((first_idx, path.len() - 1));
    }

    if visited.contains(current) {
        return None;
    }

    path.push(current);
    visited.insert(current);

    if let Some(next_symbols) = transitions.get(current) {
        for next in next_symbols {
            if let Some(cycle) = get_cycle(next, transitions, visited, path) {
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
/// For any two precedence names `a` and `b`, if `a` comes before `b` in some
/// list, then it cannot come *after* `b` in any list.
fn check_ordering_conflicts(
    precedence_orderings: &[Vec<PrecedenceEntry>],
) -> ValidatePrecedenceResult<()> {
    let mut pairs = FxHashMap::default();
    for list in precedence_orderings {
        for (i, mut entry1) in list.iter().enumerate() {
            for mut entry2 in list.iter().skip(i + 1) {
                if entry2 == entry1 {
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
                                precedence_1: entry1.to_string(),
                                precedence_2: entry2.to_string(),
                            })?;
                        }
                    }
                }
            }
        }
    }
    Ok(())
}

/// Check that all named precedences used in the grammar are declared in the
/// `precedences` lists, and that those lists don't conflict. `Reserved`
/// subtrees are not descended.
fn validate_precedences(g: &crate::rule_pool::PoolGrammar) -> ValidatePrecedenceResult<()> {
    use crate::rule_pool::{Node, Prec};

    check_ordering_conflicts(&g.precedence_orderings)?;

    let precedence_names = g
        .precedence_orderings
        .iter()
        .flat_map(|l| l.iter())
        .filter_map(|p| {
            if let PrecedenceEntry::Name(n) = p {
                Some(n.as_str())
            } else {
                None
            }
        })
        .collect::<FxHashSet<&str>>();
    let mut stack = Vec::new();
    for variable in &g.variables {
        stack.clear();
        stack.push(variable.root);
        while let Some(id) = stack.pop() {
            match g.pool.node(id) {
                Node::Repeat(inner) => stack.push(inner),
                Node::Seq(range) | Node::Choice(range) => {
                    let base = stack.len();
                    stack.extend_from_slice(g.pool.child_slice(range));
                    stack[base..].reverse();
                }
                Node::Metadata { params, rule } => {
                    if let Prec::Name(sid) = g.pool.params(params).prec
                        && !precedence_names.contains(g.pool.resolve(sid))
                    {
                        Err(UndeclaredPrecedenceError {
                            precedence: g.pool.resolve(sid).to_string(),
                            rule: g.pool.resolve(variable.name).to_string(),
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

/// Check for indirect recursion cycles that would loop while parsing: a
/// non-terminal deriving itself through a chain of single-symbol
/// productions. `Seq`/`Repeat`/`Reserved` are not single-symbol productions.
fn validate_indirect_recursion(
    g: &crate::rule_pool::PoolGrammar,
) -> Result<(), IndirectRecursionError> {
    use crate::rule_pool::Node;

    let mut epsilon_transitions: IndexMap<&str, BTreeSet<String>> = IndexMap::new();
    let mut stack = Vec::new();
    for variable in &g.variables {
        let name = g.pool.resolve(variable.name);
        let mut productions = BTreeSet::new();
        stack.clear();
        stack.push(variable.root);
        while let Some(id) = stack.pop() {
            match g.pool.node(id) {
                Node::NamedSymbol(sid) => {
                    // Rules that *directly* reference themselves don't cause
                    // a parsing loop.
                    let symbol = g.pool.resolve(sid);
                    if symbol != name {
                        productions.insert(symbol.to_string());
                    }
                }
                Node::Choice(range) => {
                    stack.extend_from_slice(g.pool.child_slice(range));
                }
                Node::Metadata { rule, .. } => stack.push(rule),
                _ => {}
            }
        }
        epsilon_transitions.insert(name, productions);
    }

    for start_symbol in epsilon_transitions.keys() {
        let mut visited = BTreeSet::new();
        let mut path = Vec::new();
        if let Some((start_idx, end_idx)) =
            get_cycle(start_symbol, &epsilon_transitions, &mut visited, &mut path)
        {
            let cycle_symbols = path[start_idx..=end_idx]
                .iter()
                .map(|s| (*s).to_string())
                .collect();
            return Err(IndirectRecursionError(cycle_symbols));
        }
    }

    Ok(())
}

/// Test fixture bridge: convert a hand-built `SyntaxGrammar` (+
/// `LexicalGrammar` for token names) into the pass inputs. Steps pack
/// through `FStep::pack`, so the result is canonical. Deleted at the
/// `SyntaxGrammar` flip.
#[cfg(test)]
fn pool_from_syntax(
    sg: &SyntaxGrammar,
    lex: &LexicalGrammar,
) -> (
    crate::rule_pool::PoolGrammar,
    extract_tokens::PoolExtractedMeta,
    crate::rule_pool::FlatOut,
) {
    use crate::rule_pool::{
        Alias, FProd, FStep, FlatOut, Node, PoolGrammar, PoolVariable, Prec, RulePool,
    };

    let mut pool = RulePool::default();
    let blank = pool.push_node(Node::Blank);
    let variables = sg
        .variables
        .iter()
        .map(|v| PoolVariable {
            name: pool.intern(&v.name),
            root: blank,
        })
        .collect();

    let mut out = FlatOut::default();
    for v in &sg.variables {
        let ps = out.productions.len() as u32;
        for p in &v.productions {
            let steps_start = out.steps.len() as u32;
            for s in &p.steps {
                out.steps.push(FStep::pack(
                    s.symbol,
                    match &s.precedence {
                        Precedence::None => Prec::None,
                        Precedence::Integer(n) => Prec::Integer(*n),
                        Precedence::Name(n) => Prec::Name(pool.intern(n)),
                    },
                    s.associativity,
                    s.alias.as_ref().map(|a| Alias {
                        value: pool.intern(&a.value),
                        is_named: a.is_named,
                    }),
                    s.field_name.as_ref().map(|f| pool.intern(f)),
                    s.reserved_word_set_id.0 as u16,
                ));
            }
            out.productions.push(FProd {
                steps_start,
                steps_len: out.steps.len() as u32 - steps_start,
                dynamic_precedence: p.dynamic_precedence,
            });
        }
        out.var_prods.push((ps, out.productions.len() as u32));
    }

    let meta = extract_tokens::PoolExtractedMeta {
        kinds: sg.variables.iter().map(|v| v.kind).collect(),
        lexical_variables: lex
            .variables
            .iter()
            .map(|v| extract_tokens::PoolLexicalVariable {
                name: pool.intern(&v.name),
                kind: v.kind,
                root: blank,
            })
            .collect(),
        separator_roots: Vec::new(),
        extra_symbols: sg.extra_symbols.clone(),
        external_tokens: sg.external_tokens.clone(),
        reserved_sets: Vec::new(),
        supertypes: sg.supertype_symbols.clone(),
        conflicts: sg.expected_conflicts.clone(),
        inline: sg.variables_to_inline.clone(),
        word: sg.word_token,
    };

    let g = PoolGrammar {
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
        precedence_orderings: sg.precedence_orderings.clone(),
    };
    (g, meta, out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::VariableType;

    #[test]
    fn test_validate_precedences_with_undeclared_precedence() {
        let grammar = InputGrammar {
            precedence_orderings: vec![
                vec![
                    PrecedenceEntry::Name("a".to_string()),
                    PrecedenceEntry::Name("b".to_string()),
                ],
                vec![
                    PrecedenceEntry::Name("b".to_string()),
                    PrecedenceEntry::Name("c".to_string()),
                    PrecedenceEntry::Name("d".to_string()),
                ],
            ],
            variables: vec![
                Variable {
                    name: "v1".to_string(),
                    kind: VariableType::Named,
                    rule: Rule::Seq(vec![
                        Rule::prec_left(Precedence::Name("b".to_string()), Rule::string("w")),
                        Rule::prec(Precedence::Name("c".to_string()), Rule::string("x")),
                    ]),
                },
                Variable {
                    name: "v2".to_string(),
                    kind: VariableType::Named,
                    rule: Rule::repeat(Rule::Choice(vec![
                        Rule::prec_left(Precedence::Name("omg".to_string()), Rule::string("y")),
                        Rule::prec(Precedence::Name("c".to_string()), Rule::string("z")),
                    ])),
                },
            ],
            ..Default::default()
        };

        let pool = crate::rule_pool::PoolGrammar::from_input_grammar(&grammar);
        assert_eq!(
            validate_precedences(&pool).unwrap_err().to_string(),
            "Undeclared precedence 'omg' in rule 'v2'",
        );
    }

    #[test]
    fn test_validate_precedences_with_conflicting_order() {
        let grammar = InputGrammar {
            precedence_orderings: vec![
                vec![
                    PrecedenceEntry::Name("a".to_string()),
                    PrecedenceEntry::Name("b".to_string()),
                ],
                vec![
                    PrecedenceEntry::Name("b".to_string()),
                    PrecedenceEntry::Name("c".to_string()),
                    PrecedenceEntry::Name("a".to_string()),
                ],
            ],
            variables: vec![
                Variable {
                    name: "v1".to_string(),
                    kind: VariableType::Named,
                    rule: Rule::Seq(vec![
                        Rule::prec_left(Precedence::Name("b".to_string()), Rule::string("w")),
                        Rule::prec(Precedence::Name("c".to_string()), Rule::string("x")),
                    ]),
                },
                Variable {
                    name: "v2".to_string(),
                    kind: VariableType::Named,
                    rule: Rule::repeat(Rule::Choice(vec![
                        Rule::prec_left(Precedence::Name("a".to_string()), Rule::string("y")),
                        Rule::prec(Precedence::Name("c".to_string()), Rule::string("z")),
                    ])),
                },
            ],
            ..Default::default()
        };

        let pool = crate::rule_pool::PoolGrammar::from_input_grammar(&grammar);
        assert_eq!(
            validate_precedences(&pool).unwrap_err().to_string(),
            "Conflicting orderings for precedences 'a' and 'b'",
        );
    }

    #[test]
    fn test_validate_indirect_recursion() {
        let cycle = |variables: Vec<Variable>| InputGrammar {
            variables,
            ..Default::default()
        };
        let cases = [
            // indirect cycle through a choice and metadata wrapper
            (
                cycle(vec![
                    Variable::named("a", Rule::choice(vec![Rule::named("b"), Rule::string("x")])),
                    Variable::named("b", Rule::prec(Precedence::Integer(1), Rule::named("a"))),
                ]),
                Err("Grammar contains an indirectly recursive rule: a -> b -> a".to_string()),
            ),
            // direct self-reference only: allowed
            (cycle(vec![Variable::named("a", Rule::named("a"))]), Ok(())),
            // three-step cycle, entered from the second variable's subgraph
            (
                cycle(vec![
                    Variable::named("a", Rule::string("x")),
                    Variable::named("b", Rule::named("c")),
                    Variable::named("c", Rule::named("d")),
                    Variable::named("d", Rule::named("b")),
                ]),
                Err("Grammar contains an indirectly recursive rule: b -> c -> d -> b".to_string()),
            ),
        ];
        for (grammar, expected) in &cases {
            let pool = crate::rule_pool::PoolGrammar::from_input_grammar(grammar);
            let result = validate_indirect_recursion(&pool).map_err(|e| e.to_string());
            assert_eq!(result, *expected);
        }
    }
}
