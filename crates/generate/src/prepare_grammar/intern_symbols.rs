use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use super::InternedGrammar;
use crate::{Diagnostic, grammars::VariableType, rules::Symbol};
#[cfg(test)]
use crate::{
    grammars::{InputGrammar, ReservedWordContext, Variable},
    rules::Rule,
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
fn variable_type_for_name(name: &str) -> VariableType {
    if name.starts_with('_') {
        VariableType::Hidden
    } else {
        VariableType::Named
    }
}

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

pub(super) fn intern_symbols(
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
    let mut name_of_symbol: Vec<Symbol> = (0..variables.len()).map(Symbol::non_terminal).collect();
    for (i, &root) in external_roots.iter().enumerate() {
        if let PoolNode::NamedSymbol(sid) = pool.node(root)
            && sid.index() == name_of_symbol.len()
        {
            name_of_symbol.push(Symbol::external(i));
        }
    }

    if variable_type_for_name(pool.resolve(variables[0].name)) == VariableType::Hidden {
        Err(InternSymbolsError::HiddenStartRule)?;
    }

    // External names/kinds read before the in-place rewrite consumes them.
    let external_tokens: Vec<(Option<StrId>, VariableType)> = external_roots
        .iter()
        .map(|&root| match pool.node(root) {
            PoolNode::NamedSymbol(sid) => (Some(sid), variable_type_for_name(pool.resolve(sid))),
            _ => (None, VariableType::Anonymous),
        })
        .collect();

    let mut kinds: Vec<VariableType> = variables
        .iter()
        .map(|v| variable_type_for_name(pool.resolve(v.name)))
        .collect();

    let mut stack = Vec::new();
    for v in variables.iter() {
        intern_root(
            pool,
            v.root,
            Some(v.name),
            &name_of_symbol,
            diagnostics,
            &mut stack,
        )?;
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
    // Unknown inline names are silently skipped.
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

/// Iterative pre-order walk rewriting `NamedSymbol -> Sym` in place,
/// emitting unary-choice/seq diagnostics in pre-order.
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

/// Materialize the pass's output as a test-shape [`InternedGrammar`].
/// Deleted at the frontend flip.
#[cfg(test)]
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
    fn test_basic_symbol_interning() {
        let (grammar, _) = pool_pass(&build_grammar(vec![
            Variable::named("x", Rule::choice(vec![Rule::named("y"), Rule::named("_z")])),
            Variable::named("y", Rule::named("_z")),
            Variable::named("_z", Rule::string("a")),
        ]))
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

        let (grammar, _) = pool_pass(&input_grammar).unwrap();

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
    fn test_meta_paths_and_diagnostics() {
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

        let (grammar, diagnostics) = pool_pass(&g).unwrap();
        assert_eq!(
            grammar.variables,
            vec![
                Variable::named(
                    "x",
                    Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(1)])
                ),
                Variable::hidden("y", Rule::Choice(vec![Rule::string("a")])),
            ]
        );
        assert_eq!(grammar.supertype_symbols, vec![Symbol::non_terminal(1)]);
        assert_eq!(grammar.word_token, Some(Symbol::non_terminal(1)));
        assert_eq!(
            grammar.expected_conflicts,
            vec![vec![Symbol::non_terminal(0), Symbol::non_terminal(1)]]
        );
        assert_eq!(grammar.variables_to_inline, vec![Symbol::non_terminal(0)]);
        assert_eq!(grammar.extra_symbols, vec![Rule::non_terminal(1)]);
        assert_eq!(
            grammar.reserved_word_sets,
            vec![crate::grammars::ReservedWordContext {
                name: "default".to_string(),
                reserved_words: vec![Rule::string("if")],
            }]
        );
        assert_eq!(
            diagnostics,
            vec![Diagnostic::UnaryChoice {
                name: Some("y".to_string())
            }]
        );
    }

    #[test]
    fn test_errors() {
        let cases = [
            (
                build_grammar(vec![Variable::named("x", Rule::named("y"))]),
                "Undefined symbol `y`",
            ),
            (
                build_grammar(vec![Variable::named("_x", Rule::string("a"))]),
                "A grammar's start rule must be visible.",
            ),
            (
                {
                    let mut g = build_grammar(vec![Variable::named("x", Rule::string("a"))]);
                    g.word_token = Some("nope".to_string());
                    g
                },
                "Undefined symbol `nope` as grammar's word token",
            ),
        ];
        for (g, expected) in &cases {
            assert_eq!(pool_pass(g).unwrap_err().to_string(), *expected);
        }
    }

    fn build_grammar(variables: Vec<Variable>) -> InputGrammar {
        InputGrammar {
            variables,
            name: "the_language".to_string(),
            ..Default::default()
        }
    }

    fn pool_pass(g: &InputGrammar) -> InternSymbolsResult<(InternedGrammar, Vec<Diagnostic>)> {
        let mut pg = crate::rule_pool::PoolGrammar::from_input_grammar(g);
        let mut diags = Vec::new();
        let meta = intern_symbols(&mut pg, &mut diags)?;
        Ok((materialize_interned(&pg, &meta), diags))
    }
}
