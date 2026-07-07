use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::InternedGrammar;
use crate::{
    Diagnostic,
    grammars::{InputGrammar, ReservedWordContext, Variable, VariableType},
    rules::{Rule, Symbol},
};

pub type InternSymbolsResult<T> = Result<T, InternSymbolsError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum InternSymbolsError {
    #[error("A grammar's start rule must be visible.")]
    HiddenStartRule,
    #[error("Undefined symbol `{0}`")]
    Undefined(String),
    #[error("Undefined symbol `{0}` in grammar's supertypes array")]
    UndefinedSupertype(String),
    #[error("Undefined symbol `{0}` in grammar's conflicts array")]
    UndefinedConflict(String),
    #[error("Undefined symbol `{0}` as grammar's word token")]
    UndefinedWordToken(String),
}

pub(super) fn intern_symbols(
    grammar: &InputGrammar,
    diagnostics: &mut Vec<Diagnostic>,
) -> InternSymbolsResult<InternedGrammar> {
    // TEMP A/B: give master's name resolution the same O(1) map the flat spike
    // uses, to isolate the map win from the backing-type/in-place win.
    let mut name_map: FxHashMap<&str, Symbol> = FxHashMap::default();
    for (i, v) in grammar.variables.iter().enumerate() {
        name_map.entry(v.name.as_str()).or_insert_with(|| Symbol::non_terminal(i));
    }
    for (i, e) in grammar.external_tokens.iter().enumerate() {
        if let Rule::NamedSymbol(name) = e {
            name_map.entry(name.as_str()).or_insert_with(|| Symbol::external(i));
        }
    }
    let interner = Interner { grammar, name_map };

    if variable_type_for_name(&grammar.variables[0].name) == VariableType::Hidden {
        Err(InternSymbolsError::HiddenStartRule)?;
    }

    let mut variables = Vec::with_capacity(grammar.variables.len());
    for variable in &grammar.variables {
        variables.push(Variable {
            name: variable.name.clone(),
            kind: variable_type_for_name(&variable.name),
            rule: interner.intern_rule(&variable.rule, Some(&variable.name), diagnostics)?,
        });
    }

    let mut external_tokens = Vec::with_capacity(grammar.external_tokens.len());
    for external_token in &grammar.external_tokens {
        let rule = interner.intern_rule(external_token, None, diagnostics)?;
        let (name, kind) = if let Rule::NamedSymbol(name) = external_token {
            (name.clone(), variable_type_for_name(name))
        } else {
            (String::new(), VariableType::Anonymous)
        };
        external_tokens.push(Variable { name, kind, rule });
    }

    let mut extra_symbols = Vec::with_capacity(grammar.extra_symbols.len());
    for extra_token in &grammar.extra_symbols {
        extra_symbols.push(interner.intern_rule(extra_token, None, diagnostics)?);
    }

    let mut supertype_symbols = Vec::with_capacity(grammar.supertype_symbols.len());
    for supertype_symbol_name in &grammar.supertype_symbols {
        supertype_symbols.push(interner.intern_name(supertype_symbol_name).ok_or_else(|| {
            InternSymbolsError::UndefinedSupertype(supertype_symbol_name.clone())
        })?);
    }

    let mut reserved_words = Vec::with_capacity(grammar.reserved_words.len());
    for reserved_word_set in &grammar.reserved_words {
        let mut interned_set = Vec::with_capacity(reserved_word_set.reserved_words.len());
        for rule in &reserved_word_set.reserved_words {
            interned_set.push(interner.intern_rule(rule, None, diagnostics)?);
        }
        reserved_words.push(ReservedWordContext {
            name: reserved_word_set.name.clone(),
            reserved_words: interned_set,
        });
    }

    let mut expected_conflicts = Vec::with_capacity(grammar.expected_conflicts.len());
    for conflict in &grammar.expected_conflicts {
        let mut interned_conflict = Vec::with_capacity(conflict.len());
        for name in conflict {
            interned_conflict.push(
                interner
                    .intern_name(name)
                    .ok_or_else(|| InternSymbolsError::UndefinedConflict(name.clone()))?,
            );
        }
        expected_conflicts.push(interned_conflict);
    }

    let mut variables_to_inline = Vec::new();
    for name in &grammar.variables_to_inline {
        if let Some(symbol) = interner.intern_name(name) {
            variables_to_inline.push(symbol);
        }
    }

    let word_token = if let Some(name) = grammar.word_token.as_ref() {
        Some(
            interner
                .intern_name(name)
                .ok_or_else(|| InternSymbolsError::UndefinedWordToken(name.clone()))?,
        )
    } else {
        None
    };

    for (i, variable) in variables.iter_mut().enumerate() {
        if supertype_symbols.contains(&Symbol::non_terminal(i)) {
            variable.kind = VariableType::Hidden;
        }
    }

    Ok(InternedGrammar {
        variables,
        external_tokens,
        extra_symbols,
        expected_conflicts,
        variables_to_inline,
        supertype_symbols,
        word_token,
        precedence_orderings: grammar.precedence_orderings.clone(),
        reserved_word_sets: reserved_words,
    })
}

struct Interner<'a> {
    grammar: &'a InputGrammar,
    name_map: FxHashMap<&'a str, Symbol>,
}

impl Interner<'_> {
    fn intern_rule(
        &self,
        rule: &Rule,
        name: Option<&str>,
        diagnostics: &mut Vec<Diagnostic>,
    ) -> InternSymbolsResult<Rule> {
        match rule {
            Rule::Choice(elements) => {
                Self::check_single(elements, name, true, diagnostics);
                let mut result = Vec::with_capacity(elements.len());
                for element in elements {
                    result.push(self.intern_rule(element, name, diagnostics)?);
                }
                Ok(Rule::Choice(result))
            }
            Rule::Seq(elements) => {
                Self::check_single(elements, name, false, diagnostics);
                let mut result = Vec::with_capacity(elements.len());
                for element in elements {
                    result.push(self.intern_rule(element, name, diagnostics)?);
                }
                Ok(Rule::Seq(result))
            }
            Rule::Repeat(content) => Ok(Rule::Repeat(Box::new(self.intern_rule(
                content,
                name,
                diagnostics,
            )?))),
            Rule::Metadata { rule, params } => Ok(Rule::Metadata {
                rule: Box::new(self.intern_rule(rule, name, diagnostics)?),
                params: params.clone(),
            }),
            Rule::Reserved { rule, context_name } => Ok(Rule::Reserved {
                rule: Box::new(self.intern_rule(rule, name, diagnostics)?),
                context_name: context_name.clone(),
            }),
            Rule::NamedSymbol(name) => self.intern_name(name).map_or_else(
                || Err(InternSymbolsError::Undefined(name.clone())),
                |symbol| Ok(Rule::Symbol(symbol)),
            ),
            _ => Ok(rule.clone()),
        }
    }

    fn intern_name(&self, symbol: &str) -> Option<Symbol> {
        // TEMP A/B: was an O(V) linear scan over variables + externals; now an O(1)
        // map lookup, to isolate the map win in the spike comparison.
        self.name_map.get(symbol).copied()
    }

    // In the case of a seq or choice rule of 1 element in a hidden rule, weird
    // inconsistent behavior with queries can occur. So we should warn the user about it.
    fn check_single(
        elements: &[Rule],
        name: Option<&str>,
        is_choice: bool,
        diagnostics: &mut Vec<Diagnostic>,
    ) {
        if elements.len() == 1 && matches!(elements[0], Rule::String(_) | Rule::Pattern(_, _)) {
            let name = name.map(str::to_string);
            diagnostics.push(if is_choice {
                Diagnostic::UnaryChoice { name }
            } else {
                Diagnostic::UnarySeq { name }
            });
        }
    }
}

fn variable_type_for_name(name: &str) -> VariableType {
    if name.starts_with('_') {
        VariableType::Hidden
    } else {
        VariableType::Named
    }
}

// --- pool-based pass (replaces the Rule-based one at the container flip) ---

use crate::rule_pool::{Node as PoolNode, NodeId, PoolGrammar, RulePool, StrId};

/// The non-pool outputs of the pool pass. The rules themselves are rewritten
/// in place (`NamedSymbol -> Sym`); this carries the derived metadata that
/// `InternedGrammar` holds alongside them.
pub(super) struct PoolInternedMeta {
    pub kinds: Vec<VariableType>,
    /// Per external token: its name (if it was a named symbol) and kind.
    pub external_tokens: Vec<(Option<StrId>, VariableType)>,
    pub supertypes: Vec<Symbol>,
    pub conflicts: Vec<Vec<Symbol>>,
    pub inline: Vec<Symbol>,
    pub word: Option<Symbol>,
}

pub(super) fn intern_symbols_pool(
    g: &mut PoolGrammar,
    diagnostics: &mut Vec<Diagnostic>,
) -> InternSymbolsResult<PoolInternedMeta> {
    let PoolGrammar {
        pool,
        variables,
        external_roots,
        extra_roots,
        reserved_sets,
        ..
    } = g;

    // Dense name table: variable i holds StrId i+1 (PoolGrammar invariant);
    // an external name not shadowing a variable got the next id in order. A
    // name's StrId indexes this table directly; undefined names have larger
    // ids and fall out via the bounds check.
    let mut name_of_symbol: Vec<Symbol> =
        (0..variables.len()).map(Symbol::non_terminal).collect();
    for (i, &root) in external_roots.iter().enumerate() {
        if let PoolNode::NamedSymbol(sid) = pool.node(root) {
            if sid.index() == name_of_symbol.len() {
                name_of_symbol.push(Symbol::external(i));
            }
        }
    }

    if variable_type_for_name(pool.resolve(variables[0].name)) == VariableType::Hidden {
        Err(InternSymbolsError::HiddenStartRule)?;
    }

    // External names/kinds read before the in-place rewrite consumes them.
    let external_tokens: Vec<(Option<StrId>, VariableType)> = external_roots
        .iter()
        .map(|&root| match pool.node(root) {
            PoolNode::NamedSymbol(sid) => {
                (Some(sid), variable_type_for_name(pool.resolve(sid)))
            }
            _ => (None, VariableType::Anonymous),
        })
        .collect();

    let mut kinds: Vec<VariableType> = variables
        .iter()
        .map(|v| variable_type_for_name(pool.resolve(v.name)))
        .collect();

    let mut stack = Vec::new();
    for v in variables.iter() {
        intern_root(pool, v.root, Some(v.name), &name_of_symbol, diagnostics, &mut stack)?;
    }
    for &root in external_roots.iter() {
        intern_root(pool, root, None, &name_of_symbol, diagnostics, &mut stack)?;
    }
    for &root in extra_roots.iter() {
        intern_root(pool, root, None, &name_of_symbol, diagnostics, &mut stack)?;
    }
    for set in reserved_sets.iter() {
        for &root in &set.roots {
            intern_root(pool, root, None, &name_of_symbol, diagnostics, &mut stack)?;
        }
    }

    let lookup = |sid: StrId| name_of_symbol.get(sid.index()).copied();
    let supertypes = g
        .supertype_names
        .iter()
        .map(|&s| {
            lookup(s).ok_or_else(|| {
                InternSymbolsError::UndefinedSupertype(g.pool.resolve(s).to_string())
            })
        })
        .collect::<InternSymbolsResult<Vec<_>>>()?;
    let conflicts = g
        .conflict_names
        .iter()
        .map(|c| {
            c.iter()
                .map(|&s| {
                    lookup(s).ok_or_else(|| {
                        InternSymbolsError::UndefinedConflict(g.pool.resolve(s).to_string())
                    })
                })
                .collect::<InternSymbolsResult<Vec<_>>>()
        })
        .collect::<InternSymbolsResult<Vec<_>>>()?;
    // Unknown inline names are silently skipped, matching master.
    let inline = g.inline_names.iter().filter_map(|&s| lookup(s)).collect();
    let word = g
        .word_name
        .map(|s| {
            lookup(s).ok_or_else(|| {
                InternSymbolsError::UndefinedWordToken(g.pool.resolve(s).to_string())
            })
        })
        .transpose()?;

    for s in &supertypes {
        if s.is_non_terminal() {
            kinds[s.index] = VariableType::Hidden;
        }
    }

    Ok(PoolInternedMeta {
        kinds,
        external_tokens,
        supertypes,
        conflicts,
        inline,
        word,
    })
}

/// Iterative pre-order walk rewriting `NamedSymbol -> Sym` in place, emitting
/// the same `check_single` diagnostics as the recursive pass, in the same
/// order.
fn intern_root(
    pool: &mut RulePool,
    root: NodeId,
    var_name: Option<StrId>,
    name_of_symbol: &[Symbol],
    diagnostics: &mut Vec<Diagnostic>,
    stack: &mut Vec<NodeId>,
) -> InternSymbolsResult<()> {
    stack.clear();
    stack.push(root);
    while let Some(id) = stack.pop() {
        match pool.node(id) {
            PoolNode::NamedSymbol(sid) => match name_of_symbol.get(sid.index()) {
                Some(&s) => pool.set_node(
                    id,
                    PoolNode::Sym {
                        kind: s.kind,
                        index: s.index as u32,
                    },
                ),
                None => Err(InternSymbolsError::Undefined(pool.resolve(sid).to_string()))?,
            },
            PoolNode::Seq(range) | PoolNode::Choice(range) => {
                let children = pool.child_slice(range);
                if children.len() == 1
                    && matches!(
                        pool.node(children[0]),
                        PoolNode::String(_) | PoolNode::Pattern(..)
                    )
                {
                    let name = var_name.map(|s| pool.resolve(s).to_string());
                    diagnostics.push(if matches!(pool.node(id), PoolNode::Choice(_)) {
                        Diagnostic::UnaryChoice { name }
                    } else {
                        Diagnostic::UnarySeq { name }
                    });
                }
                let base = stack.len();
                stack.extend_from_slice(pool.child_slice(range));
                stack[base..].reverse();
            }
            PoolNode::Repeat(inner)
            | PoolNode::Metadata { rule: inner, .. }
            | PoolNode::Reserved { rule: inner, .. } => stack.push(inner),
            _ => {}
        }
    }
    Ok(())
}

/// Materialize the pool pass's output as a master [`InternedGrammar`] for A/B
/// verification. Deleted at the container flip.
pub(super) fn materialize_interned(g: &PoolGrammar, meta: &PoolInternedMeta) -> InternedGrammar {
    InternedGrammar {
        variables: g
            .variables
            .iter()
            .zip(&meta.kinds)
            .map(|(v, &kind)| Variable {
                name: g.pool.resolve(v.name).to_string(),
                kind,
                rule: g.pool.rule(v.root),
            })
            .collect(),
        external_tokens: g
            .external_roots
            .iter()
            .zip(&meta.external_tokens)
            .map(|(&root, &(name, kind))| Variable {
                name: name.map_or_else(String::new, |s| g.pool.resolve(s).to_string()),
                kind,
                rule: g.pool.rule(root),
            })
            .collect(),
        extra_symbols: g.extra_roots.iter().map(|&r| g.pool.rule(r)).collect(),
        expected_conflicts: meta.conflicts.clone(),
        variables_to_inline: meta.inline.clone(),
        supertype_symbols: meta.supertypes.clone(),
        word_token: meta.word,
        precedence_orderings: g.precedence_orderings.clone(),
        reserved_word_sets: g
            .reserved_sets
            .iter()
            .map(|set| ReservedWordContext {
                name: g.pool.resolve(set.name).to_string(),
                reserved_words: set.roots.iter().map(|&r| g.pool.rule(r)).collect(),
            })
            .collect(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_repeat_expansion() {
        let grammar = intern_symbols(
            &build_grammar(vec![
                Variable::named("x", Rule::choice(vec![Rule::named("y"), Rule::named("_z")])),
                Variable::named("y", Rule::named("_z")),
                Variable::named("_z", Rule::string("a")),
            ]),
            &mut Vec::new(),
        )
        .unwrap();

        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "x",
                    Rule::choice(vec![Rule::non_terminal(1), Rule::non_terminal(2),])
                ),
                Variable::named("y", Rule::non_terminal(2)),
                Variable::hidden("_z", Rule::string("a")),
            ]
        );
    }

    #[test]
    fn test_interning_external_token_names() {
        // Variable `y` is both an internal and an external token.
        // Variable `z` is just an external token.
        let mut input_grammar = build_grammar(vec![
            Variable::named(
                "w",
                Rule::choice(vec![Rule::named("x"), Rule::named("y"), Rule::named("z")]),
            ),
            Variable::named("x", Rule::string("a")),
            Variable::named("y", Rule::string("b")),
        ]);
        input_grammar
            .external_tokens
            .extend(vec![Rule::named("y"), Rule::named("z")]);

        let grammar = intern_symbols(&input_grammar, &mut Vec::new()).unwrap();

        // Variable `y` is referred to by its internal index.
        // Variable `z` is referred to by its external index.
        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "w",
                    Rule::choice(vec![
                        Rule::non_terminal(1),
                        Rule::non_terminal(2),
                        Rule::external(1),
                    ])
                ),
                Variable::named("x", Rule::string("a")),
                Variable::named("y", Rule::string("b")),
            ]
        );

        // The external token for `y` refers back to its internal index.
        assert_eq!(
            grammar.external_tokens,
            vec![
                Variable::named("y", Rule::non_terminal(2)),
                Variable::named("z", Rule::external(1)),
            ]
        );
    }

    #[test]
    fn test_grammar_with_undefined_symbols() {
        let result = intern_symbols(
            &build_grammar(vec![Variable::named("x", Rule::named("y"))]),
            &mut Vec::new(),
        );

        assert!(result.is_err(), "Expected an error but got none");
        let e = result.err().unwrap();
        assert_eq!(e.to_string(), "Undefined symbol `y`");
    }

    fn build_grammar(variables: Vec<Variable>) -> InputGrammar {
        InputGrammar {
            variables,
            name: "the_language".to_string(),
            ..Default::default()
        }
    }

    fn pool_pass(
        g: &InputGrammar,
    ) -> InternSymbolsResult<(InternedGrammar, Vec<Diagnostic>)> {
        let mut pg = crate::rule_pool::PoolGrammar::from_input_grammar(g);
        let mut diags = Vec::new();
        let meta = intern_symbols_pool(&mut pg, &mut diags)?;
        Ok((materialize_interned(&pg, &meta), diags))
    }

    fn assert_pool_matches_master(g: &InputGrammar) {
        let mut master_diags = Vec::new();
        let master = intern_symbols(g, &mut master_diags).unwrap();
        let (pool_out, pool_diags) = pool_pass(g).unwrap();
        assert_eq!(pool_out, master);
        assert_eq!(pool_diags, master_diags);
    }

    #[test]
    fn pool_pass_matches_master() {
        // Bodies, hidden rules, external interplay.
        assert_pool_matches_master(&build_grammar(vec![
            Variable::named("x", Rule::choice(vec![Rule::named("y"), Rule::named("_z")])),
            Variable::named("y", Rule::named("_z")),
            Variable::named("_z", Rule::string("a")),
        ]));
        let mut g = build_grammar(vec![
            Variable::named(
                "w",
                Rule::choice(vec![Rule::named("x"), Rule::named("y"), Rule::named("z")]),
            ),
            Variable::named("x", Rule::string("a")),
            Variable::named("y", Rule::string("b")),
        ]);
        g.external_tokens
            .extend(vec![Rule::named("y"), Rule::named("z")]);
        assert_pool_matches_master(&g);

        // Every meta path: supertype hiding, conflicts, silent inline skip,
        // word token, reserved sets, extras, and a unary-choice diagnostic.
        let mut g = build_grammar(vec![
            Variable::named("x", Rule::seq(vec![Rule::named("y"), Rule::named("y")])),
            Variable::named("y", Rule::Choice(vec![Rule::string("a")])),
        ]);
        g.supertype_symbols = vec!["y".to_string()];
        g.word_token = Some("y".to_string());
        g.expected_conflicts = vec![vec!["x".to_string(), "y".to_string()]];
        g.variables_to_inline = vec!["x".to_string(), "nonexistent".to_string()];
        g.extra_symbols = vec![Rule::named("y")];
        g.reserved_words = vec![crate::grammars::ReservedWordContext {
            name: "default".to_string(),
            reserved_words: vec![Rule::string("if")],
        }];
        assert_pool_matches_master(&g);
    }

    #[test]
    fn pool_pass_matches_master_errors() {
        let cases = [
            build_grammar(vec![Variable::named("x", Rule::named("y"))]),
            build_grammar(vec![Variable::named("_x", Rule::string("a"))]),
            {
                let mut g = build_grammar(vec![Variable::named("x", Rule::string("a"))]);
                g.word_token = Some("nope".to_string());
                g
            },
        ];
        for g in &cases {
            let master_err = intern_symbols(g, &mut Vec::new()).unwrap_err();
            let pool_err = pool_pass(g).unwrap_err();
            assert_eq!(pool_err.to_string(), master_err.to_string());
        }
    }
}
