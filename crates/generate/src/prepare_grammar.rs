mod expand_repeats;
mod expand_tokens;
mod extract_default_aliases;
mod extract_tokens;
mod flat_spike; // TEMP: perf spike, remove after the pool decision
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
    grammars::{
        ExternalToken, InlinedProductionMap, InputGrammar, LexicalGrammar, PrecedenceEntry,
        SyntaxGrammar, Variable,
    },
    rules::{AliasMap, Precedence, Rule, Symbol},
};
use crate::{Diagnostic, grammars::ReservedWordContext};

#[derive(Clone, Debug, PartialEq, Eq)] // TEMP: for the pass A/B loops and equality asserts
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

pub type InternedGrammar = IntermediateGrammar<Rule, Variable>;

pub type ExtractedSyntaxGrammar = IntermediateGrammar<Symbol, ExternalToken>;

#[derive(Debug, PartialEq, Eq)]
pub struct ExtractedLexicalGrammar {
    pub variables: Vec<Variable>,
    pub separators: Vec<Rule>,
}

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
    // TEMP SPIKE A/B (remove after the pool decision): master intern_symbols
    // (Rule, produces a new tree incl. alloc) vs flat-pool intern_symbols
    // (in-place, no hash-cons, O(1) name map). The Rule->pool conversion is done
    // up front (setup) and excluded from timing; the flat pool is pre-cloned per
    // iteration so we time only the in-place rewrite.
    // TEMP A/B (until the container flip): the real pool-based intern pass vs
    // master's, timing plus full-output and diagnostics equality.
    let (intern_master, intern_flat, pool_build) = {
        use std::hint::black_box;
        let n = 300u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = black_box(intern_symbols(input_grammar, &mut Vec::new()));
        }
        let master = t.elapsed() / n;

        let t = std::time::Instant::now();
        let base = crate::rule_pool::PoolGrammar::from_input_grammar(input_grammar);
        let pool_build = t.elapsed();
        let mut pools: Vec<_> = (0..n).map(|_| base.clone()).collect();
        let t = std::time::Instant::now();
        for p in &mut pools {
            let _ = black_box(intern_symbols::intern_symbols_pool(p, &mut Vec::new()));
        }
        let pool_time = t.elapsed() / n;
        eprintln!("[POOL intern_symbols] master {master:?}  pool {pool_time:?}");

        let mut master_diags = Vec::new();
        let master_out = intern_symbols(input_grammar, &mut master_diags).unwrap();
        let mut g = base;
        let mut pool_diags = Vec::new();
        let meta = intern_symbols::intern_symbols_pool(&mut g, &mut pool_diags).unwrap();
        assert_eq!(
            intern_symbols::materialize_interned(&g, &meta),
            master_out,
            "pool intern_symbols diverged from master"
        );
        assert_eq!(pool_diags, master_diags, "pool intern diagnostics diverged");
        (master, pool_time, pool_build)
    };

    let t0 = std::time::Instant::now();
    validate_precedences(input_grammar)?;
    validate_indirect_recursion(input_grammar)?;
    let t_validate = t0.elapsed();

    let t0 = std::time::Instant::now();
    let interned_grammar = intern_symbols(input_grammar, diagnostics)?;
    let t_intern = t0.elapsed();

    // TEMP SPIKE: time the unconverted middle passes (net of the input clone
    // each iteration needs) so the combined summary below can place the two
    // converted passes in syntax-path context.
    let extract_net = {
        use std::hint::black_box;
        let n = 300u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            black_box(interned_grammar.clone());
        }
        let clone_only = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = black_box(extract_tokens(interned_grammar.clone()));
        }
        (t.elapsed() / n).checked_sub(clone_only).unwrap_or_default()
    };

    // TEMP A/B (until the container flip): pool extract_tokens vs master,
    // chained after the pool intern pass so the converted prefix verifies as
    // a real pipeline. Asserts both outputs (syntax and lexical grammars).
    {
        use std::hint::black_box;
        let n = 300u32;
        let mut base = crate::rule_pool::PoolGrammar::from_input_grammar(input_grammar);
        let interned_meta =
            intern_symbols::intern_symbols_pool(&mut base, &mut Vec::new()).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n {
            black_box(base.clone());
        }
        let clone_only = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let mut g = base.clone();
            let _ = black_box(extract_tokens::extract_tokens_pool(&mut g, &interned_meta));
        }
        let pool_time = (t.elapsed() / n).checked_sub(clone_only).unwrap_or_default();
        eprintln!("[POOL extract_tokens] master {extract_net:?}  pool {pool_time:?}");

        let master_out = extract_tokens(interned_grammar.clone()).unwrap();
        let mut g = base.clone();
        let meta = extract_tokens::extract_tokens_pool(&mut g, &interned_meta).unwrap();
        assert_eq!(
            extract_tokens::materialize_extracted(&g, &meta, &g.precedence_orderings),
            master_out,
            "pool extract_tokens diverged from master"
        );
    }

    let t0 = std::time::Instant::now();
    let (syntax_grammar, lexical_grammar) = extract_tokens(interned_grammar)?;
    let t_extract = t0.elapsed();

    let expand_net = {
        use std::hint::black_box;
        let n = 300u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            black_box(syntax_grammar.clone());
        }
        let clone_only = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            black_box(expand_repeats(syntax_grammar.clone()));
        }
        (t.elapsed() / n).checked_sub(clone_only).unwrap_or_default()
    };

    // TEMP A/B (until the container flip): pool expand_repeats vs master,
    // chained after the pool intern + extract passes.
    {
        use std::hint::black_box;
        let n = 300u32;
        let mut base = crate::rule_pool::PoolGrammar::from_input_grammar(input_grammar);
        let interned_meta =
            intern_symbols::intern_symbols_pool(&mut base, &mut Vec::new()).unwrap();
        let ext_meta = extract_tokens::extract_tokens_pool(&mut base, &interned_meta).unwrap();
        let t = std::time::Instant::now();
        for _ in 0..n {
            black_box((base.clone(), ext_meta.clone()));
        }
        let clone_only = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let (mut g, mut m) = (base.clone(), ext_meta.clone());
            expand_repeats::expand_repeats_pool(&mut g, &mut m);
            black_box((&g, &m));
        }
        let pool_time = (t.elapsed() / n).checked_sub(clone_only).unwrap_or_default();
        eprintln!("[POOL expand_repeats] master {expand_net:?}  pool {pool_time:?}");

        let master_out = expand_repeats(syntax_grammar.clone());
        let (mut g, mut m) = (base, ext_meta);
        expand_repeats::expand_repeats_pool(&mut g, &mut m);
        assert_eq!(
            extract_tokens::materialize_extracted(&g, &m, &g.precedence_orderings).0,
            master_out,
            "pool expand_repeats diverged from master"
        );
    }

    let t0 = std::time::Instant::now();
    let syntax_grammar = expand_repeats(syntax_grammar);
    let t_expand = t0.elapsed();

    // TEMP SPIKE A/B (remove after the pool decision): master flatten_grammar
    // vs flat-pool flatten (fork-at-choice, pooled output, truncate-restore
    // scratch). Master's flatten consumes its input, so its timing includes a
    // per-iteration clone, reported separately. The flat side mirrors master's
    // post-checks; it skips only the pass-through SyntaxGrammar assembly
    // (reserved TokenSets, field moves), which is noise on real grammars.
    // Materialized output is asserted equal to master's.
    {
        use std::hint::black_box;
        let n = 300u32;
        let t = std::time::Instant::now();
        for _ in 0..n {
            black_box(syntax_grammar.clone());
        }
        let clone_only = t.elapsed() / n;
        let t = std::time::Instant::now();
        for _ in 0..n {
            let _ = black_box(flatten_grammar(syntax_grammar.clone()));
        }
        let master = t.elapsed() / n;

        let t = std::time::Instant::now();
        let pool = flat_spike::Pool::from_extracted(&syntax_grammar);
        let pool_build2 = t.elapsed();
        let mut st = flat_spike::FlattenState::default();
        let mut out = flat_spike::FlatOut::default();
        let t = std::time::Instant::now();
        for _ in 0..n {
            pool.flatten(&syntax_grammar, &mut st, &mut out).unwrap();
            black_box(&out);
        }
        let flat = t.elapsed() / n;
        eprintln!(
            "[SPIKE flatten_grammar] master {master:?} (incl clone {clone_only:?})  flat {flat:?}"
        );

        // The two converted passes together, then in syntax-path context
        // (intern -> extract -> expand -> flatten; the middle is unconverted
        // and shared). Bridge costs (one-shot pool builds) shown for the
        // with-adapters scenario; the end state has no bridges.
        let master_net = master.checked_sub(clone_only).unwrap_or_default();
        let two_m = intern_master + master_net;
        let two_f = intern_flat + flat;
        let mid = extract_net + expand_net;
        let path_m = two_m + mid;
        let path_f = two_f + mid;
        eprintln!(
            "[SPIKE combined] converted passes: master {two_m:?}  flat {two_f:?}  ({:.1}x) | middle (unconverted extract+expand): {mid:?} | syntax path: master {path_m:?} -> hybrid {path_f:?} ({:.1}x) | bridges: {pool_build:?} + {pool_build2:?}",
            two_m.as_secs_f64() / two_f.as_secs_f64(),
            path_m.as_secs_f64() / path_f.as_secs_f64(),
        );

        let master_out = flatten_grammar(syntax_grammar.clone()).unwrap();
        pool.flatten(&syntax_grammar, &mut st, &mut out).unwrap();
        assert_eq!(
            pool.materialize(&out, &syntax_grammar.variables),
            master_out.variables,
            "flat flatten diverged from master"
        );
    }

    let t0 = std::time::Instant::now();
    let mut syntax_grammar = flatten_grammar(syntax_grammar)?;
    let t_flatten = t0.elapsed();
    let t0 = std::time::Instant::now();
    let lexical_grammar = expand_tokens(lexical_grammar)?;
    let t_tokens = t0.elapsed();
    let t0 = std::time::Instant::now();
    let default_aliases = extract_default_aliases(&mut syntax_grammar, &lexical_grammar);
    let inlines = process_inlines(&syntax_grammar, &lexical_grammar)?;
    let t_rest = t0.elapsed();
    // TEMP SPIKE: real single-shot stage times (spike loops excluded).
    eprintln!(
        "[SPIKE prepare-stages] validate {t_validate:?} intern {t_intern:?} extract {t_extract:?} expand {t_expand:?} flatten {t_flatten:?} expand_tokens {t_tokens:?} aliases+inlines {t_rest:?}"
    );
    Ok((syntax_grammar, lexical_grammar, inlines, default_aliases))
}

/// Check for indirect recursion cycles in the grammar that can cause infinite loops while
/// parsing. An indirect recursion cycle occurs when a non-terminal can derive itself through
/// a chain of single-symbol productions (e.g., A -> B, B -> A).
fn validate_indirect_recursion(grammar: &InputGrammar) -> Result<(), IndirectRecursionError> {
    let mut epsilon_transitions: IndexMap<&str, BTreeSet<String>> = IndexMap::new();

    for variable in &grammar.variables {
        let productions = get_single_symbol_productions(&variable.rule);
        // Filter out rules that *directly* reference themselves, as this doesn't
        // cause a parsing loop.
        let filtered: BTreeSet<String> = productions
            .into_iter()
            .filter(|s| s != &variable.name)
            .collect();
        epsilon_transitions.insert(variable.name.as_str(), filtered);
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

fn get_single_symbol_productions(rule: &Rule) -> BTreeSet<String> {
    match rule {
        Rule::NamedSymbol(name) => BTreeSet::from([name.clone()]),
        Rule::Choice(choices) => choices
            .iter()
            .flat_map(get_single_symbol_productions)
            .collect(),
        Rule::Metadata { rule, .. } => get_single_symbol_productions(rule),
        _ => BTreeSet::new(),
    }
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
fn validate_precedences(grammar: &InputGrammar) -> ValidatePrecedenceResult<()> {
    // Check that no rule contains a named precedence that is not present in
    // any of the `precedences` lists.
    fn validate(
        rule_name: &str,
        rule: &Rule,
        names: &FxHashSet<&String>,
    ) -> ValidatePrecedenceResult<()> {
        match rule {
            Rule::Repeat(rule) => validate(rule_name, rule, names),
            Rule::Seq(elements) | Rule::Choice(elements) => elements
                .iter()
                .try_for_each(|e| validate(rule_name, e, names)),
            Rule::Metadata { rule, params } => {
                if let Precedence::Name(n) = &params.precedence
                    && !names.contains(n)
                {
                    Err(UndeclaredPrecedenceError {
                        precedence: n.clone(),
                        rule: rule_name.to_string(),
                    })?;
                }
                validate(rule_name, rule, names)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    // For any two precedence names `a` and `b`, if `a` comes before `b`
    // in some list, then it cannot come *after* `b` in any list.
    let mut pairs = FxHashMap::default();
    for list in &grammar.precedence_orderings {
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

    let precedence_names = grammar
        .precedence_orderings
        .iter()
        .flat_map(|l| l.iter())
        .filter_map(|p| {
            if let PrecedenceEntry::Name(n) = p {
                Some(n)
            } else {
                None
            }
        })
        .collect::<FxHashSet<&String>>();
    for variable in &grammar.variables {
        validate(&variable.name, &variable.rule, &precedence_names)?;
    }

    Ok(())
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

        let result = validate_precedences(&grammar);
        assert_eq!(
            result.unwrap_err().to_string(),
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

        let result = validate_precedences(&grammar);
        assert_eq!(
            result.unwrap_err().to_string(),
            "Conflicting orderings for precedences 'a' and 'b'",
        );
    }
}
