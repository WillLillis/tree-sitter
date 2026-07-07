use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use super::{ExtractedLexicalGrammar, ExtractedSyntaxGrammar, InternedGrammar};
use crate::{
    grammars::{ExternalToken, VariableType},
    rules::Symbol,
};
#[cfg(test)]
use crate::{
    grammars::{ReservedWordContext, Variable},
    rules::Rule,
};

pub type ExtractTokensResult<T> = Result<T, ExtractTokensError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExtractTokensError {
    #[error(
        "The rule `{0}` contains an empty string.

Tree-sitter does not support syntactic rules that contain an empty string
unless they are used only as the grammar's start rule.
"
    )]
    EmptyString(String),
    #[error("Terminal rule '{0}' cannot be used as a supertype")]
    SupertypeTerminal(String),
    #[error("Rule '{0}' cannot be used as both an external token and a non-terminal rule")]
    ExternalTokenNonTerminal(String),
    #[error("Non-symbol rules cannot be used as external tokens")]
    NonSymbolExternalToken,
    #[error(transparent)]
    WordToken(NonTerminalWordTokenError),
    #[error("Reserved word '{0}' must be a token")]
    NonTokenReservedWord(String),
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub struct NonTerminalWordTokenError {
    pub symbol_name: String,
    pub conflicting_symbol_name: Option<String>,
}

impl std::fmt::Display for NonTerminalWordTokenError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Non-terminal symbol '{}' cannot be used as the word token",
            self.symbol_name
        )?;
        if let Some(conflicting_name) = &self.conflicting_symbol_name {
            writeln!(
                f,
                ", because its rule is duplicated in '{conflicting_name}'",
            )
        } else {
            writeln!(f)
        }
    }
}
use super::intern_symbols::PoolInternedMeta;
#[cfg(test)]
use crate::grammars::PrecedenceEntry;
use crate::rule_pool::{Node as PoolNode, NodeId, Params, PoolGrammar, RulePool, StrId};

#[derive(Clone, Debug)]
pub(super) struct PoolLexicalVariable {
    pub name: StrId,
    pub kind: VariableType,
    pub root: NodeId,
}

/// The extract pass's outputs besides the in-place rewrites: the lexical
/// side, post-absorption variable metadata, and symbol-typed config lists.
#[derive(Clone, Debug)]
pub(super) struct PoolExtractedMeta {
    pub kinds: Vec<VariableType>,
    pub lexical_variables: Vec<PoolLexicalVariable>,
    pub separator_roots: Vec<NodeId>,
    pub extra_symbols: Vec<Symbol>,
    pub external_tokens: Vec<ExternalToken>,
    pub reserved_sets: Vec<(StrId, Vec<Symbol>)>,
    pub supertypes: Vec<Symbol>,
    pub conflicts: Vec<Vec<Symbol>>,
    pub inline: Vec<Symbol>,
    pub word: Option<Symbol>,
}

/// Token dedup: structural memo over pool subtrees (hash of interned
/// content, verified with `subtree_eq`).
#[derive(Default)]
struct PoolTokenExtractor {
    lexical: Vec<PoolLexicalVariable>,
    usage_counts: Vec<u32>,
    memo: FxHashMap<u64, Vec<u32>>,
}

impl PoolTokenExtractor {
    /// Find or create the lexical token for `boundary`, hoisting by shallow
    /// root copy (the subtree below is shared with the syntax side until the
    /// caller overwrites the original node).
    fn extract(
        &mut self,
        pool: &mut RulePool,
        boundary: NodeId,
        string_name: Option<StrId>,
        var_name: Option<StrId>,
        counter: &mut u32,
        is_first: bool,
    ) -> ExtractTokensResult<u32> {
        let hash = pool.subtree_hash(boundary);
        if let Some(candidates) = self.memo.get(&hash) {
            for &i in candidates {
                if pool.subtree_eq(self.lexical[i as usize].root, boundary) {
                    self.usage_counts[i as usize] += 1;
                    return Ok(i);
                }
            }
        }
        let (name, kind) = if let Some(sid) = string_name {
            if pool.resolve(sid).is_empty() && !is_first {
                Err(ExtractTokensError::EmptyString(
                    var_name.map_or_else(String::new, |v| pool.resolve(v).to_string()),
                ))?;
            }
            (sid, VariableType::Anonymous)
        } else {
            *counter += 1;
            let name = format!(
                "{}_token{}",
                var_name.map_or("", |v| pool.resolve(v)),
                counter
            );
            (pool.intern(&name), VariableType::Auxiliary)
        };
        let index = self.lexical.len() as u32;
        let root = pool.push_node(pool.node(boundary));
        self.lexical.push(PoolLexicalVariable { name, kind, root });
        self.usage_counts.push(1);
        self.memo.entry(hash).or_default().push(index);
        Ok(index)
    }

    /// Structural lookup without insertion (extras/reserved routing).
    fn find(&self, pool: &RulePool, root: NodeId) -> Option<u32> {
        self.memo.get(&pool.subtree_hash(root)).and_then(|cands| {
            cands
                .iter()
                .copied()
                .find(|&i| pool.subtree_eq(self.lexical[i as usize].root, root))
        })
    }
}

const fn sym(symbol: Symbol) -> PoolNode {
    PoolNode::Sym {
        kind: symbol.kind,
        index: symbol.index as u32,
    }
}

const fn node_symbol(node: PoolNode) -> Option<Symbol> {
    match node {
        PoolNode::Sym { kind, index } => Some(Symbol {
            kind,
            index: index as usize,
        }),
        _ => None,
    }
}

/// In-place token extraction over one root. Token boundaries:
/// `String`/`Pattern` always; `token(...)` metadata extracts the inner
/// child when no other params are set, else the whole metadata node (which
/// keeps `is_token` on the lexical side).
fn extract_in_root(
    pool: &mut RulePool,
    root: NodeId,
    var_name: Option<StrId>,
    is_first: bool,
    ex: &mut PoolTokenExtractor,
    stack: &mut Vec<NodeId>,
) -> ExtractTokensResult<()> {
    let mut counter = 0u32;
    stack.clear();
    stack.push(root);
    while let Some(id) = stack.pop() {
        match pool.node(id) {
            PoolNode::String(sid) => {
                let i = ex.extract(pool, id, Some(sid), var_name, &mut counter, is_first)?;
                pool.set_node(id, sym(Symbol::terminal(i as usize)));
            }
            PoolNode::Pattern(..) => {
                let i = ex.extract(pool, id, None, var_name, &mut counter, is_first)?;
                pool.set_node(id, sym(Symbol::terminal(i as usize)));
            }
            PoolNode::Metadata { params, rule } => {
                let p = pool.params(params);
                if p.is_token {
                    let cleaned = Params {
                        is_token: false,
                        ..p
                    };
                    let string_name = match pool.node(rule) {
                        PoolNode::String(s) => Some(s),
                        _ => None,
                    };
                    let boundary = if cleaned == Params::default() {
                        rule
                    } else {
                        id
                    };
                    let i = ex.extract(
                        pool,
                        boundary,
                        string_name,
                        var_name,
                        &mut counter,
                        is_first,
                    )?;
                    pool.set_node(id, sym(Symbol::terminal(i as usize)));
                } else {
                    stack.push(rule);
                }
            }
            PoolNode::Seq(range) | PoolNode::Choice(range) => {
                let base = stack.len();
                stack.extend_from_slice(pool.child_slice(range));
                stack[base..].reverse();
            }
            PoolNode::Repeat(inner) | PoolNode::Reserved { rule: inner, .. } => stack.push(inner),
            _ => {}
        }
    }
    Ok(())
}

pub(super) fn extract_tokens(
    g: &mut PoolGrammar,
    interned: &PoolInternedMeta,
) -> ExtractTokensResult<PoolExtractedMeta> {
    let mut ex = PoolTokenExtractor::default();
    let mut stack = Vec::new();
    for (i, v) in g.variables.iter().enumerate() {
        extract_in_root(
            &mut g.pool,
            v.root,
            Some(v.name),
            i == 0,
            &mut ex,
            &mut stack,
        )?;
    }
    for (&root, &(name, _)) in g.external_roots.iter().zip(&interned.external_tokens) {
        extract_in_root(&mut g.pool, root, name, false, &mut ex, &mut stack)?;
    }

    // Absorption: a variable (other than the start rule) whose entire body
    // became one single-use terminal donates its name/kind to that token and
    // is removed.
    let old_len = g.variables.len();
    let mut replacements: FxHashMap<u32, u32> = FxHashMap::default();
    let mut retained = Vec::with_capacity(old_len);
    let mut kinds = Vec::with_capacity(old_len);
    for (i, v) in g.variables.iter().enumerate() {
        if i > 0
            && let Some(s) = node_symbol(g.pool.node(v.root))
            && s.is_terminal()
            && ex.usage_counts[s.index] == 1
        {
            let lexical = &mut ex.lexical[s.index];
            if lexical.kind == VariableType::Auxiliary || interned.kinds[i] != VariableType::Hidden
            {
                lexical.kind = interned.kinds[i];
                lexical.name = v.name;
                replacements.insert(i as u32, s.index as u32);
                continue;
            }
        }
        retained.push(*v);
        kinds.push(interned.kinds[i]);
    }
    g.variables = retained;

    // Prefix-sum renumbering: O(1) per symbol.
    let mut shift = vec![0u32; old_len];
    let mut removed = 0u32;
    for (i, slot) in shift.iter_mut().enumerate() {
        *slot = removed;
        if replacements.contains_key(&(i as u32)) {
            removed += 1;
        }
    }
    let replace_symbol = |s: Symbol| {
        if !s.is_non_terminal() {
            return s;
        }
        replacements.get(&(s.index as u32)).map_or_else(
            || Symbol::non_terminal(s.index - shift[s.index] as usize),
            |&r| Symbol::terminal(r as usize),
        )
    };

    for v in &g.variables {
        renumber_root(&mut g.pool, v.root, &replace_symbol, &mut stack);
    }
    for &root in &g.external_roots {
        renumber_root(&mut g.pool, root, &replace_symbol, &mut stack);
    }

    let conflicts = interned
        .conflicts
        .iter()
        .map(|c| {
            let mut result: Vec<Symbol> = c.iter().map(|&s| replace_symbol(s)).collect();
            result.sort_unstable();
            result.dedup();
            result
        })
        .collect();

    let supertypes: Vec<Symbol> = interned
        .supertypes
        .iter()
        .map(|&s| replace_symbol(s))
        .collect();
    for s in &supertypes {
        if s.is_terminal() {
            Err(ExtractTokensError::SupertypeTerminal(
                g.pool.resolve(ex.lexical[s.index].name).to_string(),
            ))?;
        }
    }

    let inline = interned.inline.iter().map(|&s| replace_symbol(s)).collect();

    let mut separator_roots = Vec::new();
    let mut extra_symbols = Vec::new();
    for &root in &g.extra_roots {
        if let Some(s) = node_symbol(g.pool.node(root)) {
            extra_symbols.push(replace_symbol(s));
        } else if let Some(i) = ex.find(&g.pool, root) {
            extra_symbols.push(Symbol::terminal(i as usize));
        } else {
            separator_roots.push(root);
        }
    }

    let mut external_tokens = Vec::with_capacity(g.external_roots.len());
    for (&root, &(name, kind)) in g.external_roots.iter().zip(&interned.external_tokens) {
        let Some(s) = node_symbol(g.pool.node(root)) else {
            Err(ExtractTokensError::NonSymbolExternalToken)?
        };
        if s.is_non_terminal() {
            Err(ExtractTokensError::ExternalTokenNonTerminal(
                g.pool.resolve(g.variables[s.index].name).to_string(),
            ))?;
        }
        external_tokens.push(if s.is_external() {
            ExternalToken {
                name: name.map_or_else(String::new, |n| g.pool.resolve(n).to_string()),
                kind,
                corresponding_internal_token: None,
            }
        } else {
            ExternalToken {
                name: g.pool.resolve(ex.lexical[s.index].name).to_string(),
                kind,
                corresponding_internal_token: Some(s),
            }
        });
    }

    let word = match interned.word.map(replace_symbol) {
        Some(token) if token.is_non_terminal() => {
            let word_root = g.variables[token.index].root;
            let conflicting_symbol_name = g
                .variables
                .iter()
                .enumerate()
                .find(|(i, v)| *i != token.index && g.pool.subtree_eq(v.root, word_root))
                .map(|(_, v)| g.pool.resolve(v.name).to_string());
            Err(ExtractTokensError::WordToken(NonTerminalWordTokenError {
                symbol_name: g.pool.resolve(g.variables[token.index].name).to_string(),
                conflicting_symbol_name,
            }))?
        }
        word => word,
    };

    let mut reserved_sets = Vec::with_capacity(g.reserved_sets.len());
    for set in &g.reserved_sets {
        let mut symbols = Vec::with_capacity(set.roots.len());
        for &root in &set.roots {
            if let Some(s) = node_symbol(g.pool.node(root)) {
                symbols.push(replace_symbol(s));
            } else if let Some(i) = ex.find(&g.pool, root) {
                symbols.push(Symbol::terminal(i as usize));
            } else {
                let inner = match g.pool.node(root) {
                    PoolNode::Metadata { rule, .. } => g.pool.node(rule),
                    node => node,
                };
                let token_name = match inner {
                    PoolNode::String(s) | PoolNode::Pattern(s, _) => g.pool.resolve(s).to_string(),
                    _ => "unknown".to_string(),
                };
                Err(ExtractTokensError::NonTokenReservedWord(token_name))?;
            }
        }
        reserved_sets.push((set.name, symbols));
    }

    Ok(PoolExtractedMeta {
        kinds,
        lexical_variables: ex.lexical,
        separator_roots,
        extra_symbols,
        external_tokens,
        reserved_sets,
        supertypes,
        conflicts,
        inline,
        word,
    })
}

/// In-place symbol renumbering after absorption.
fn renumber_root(
    pool: &mut RulePool,
    root: NodeId,
    replace: &impl Fn(Symbol) -> Symbol,
    stack: &mut Vec<NodeId>,
) {
    stack.clear();
    stack.push(root);
    while let Some(id) = stack.pop() {
        match pool.node(id) {
            node @ PoolNode::Sym { .. } => {
                let s = node_symbol(node).unwrap();
                let replaced = replace(s);
                if replaced != s {
                    pool.set_node(id, sym(replaced));
                }
            }
            PoolNode::Seq(range) | PoolNode::Choice(range) => {
                stack.extend_from_slice(pool.child_slice(range));
            }
            PoolNode::Repeat(inner)
            | PoolNode::Metadata { rule: inner, .. }
            | PoolNode::Reserved { rule: inner, .. } => stack.push(inner),
            _ => {}
        }
    }
}

/// Test-shape materialization of the pass's outputs. Deleted at the
/// frontend flip.
#[cfg(test)]
pub(super) fn materialize_extracted(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    precedence_orderings: &[Vec<PrecedenceEntry>],
) -> (ExtractedSyntaxGrammar, ExtractedLexicalGrammar) {
    (
        ExtractedSyntaxGrammar {
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
            expected_conflicts: meta.conflicts.clone(),
            extra_symbols: meta.extra_symbols.clone(),
            variables_to_inline: meta.inline.clone(),
            supertype_symbols: meta.supertypes.clone(),
            external_tokens: meta.external_tokens.clone(),
            word_token: meta.word,
            precedence_orderings: precedence_orderings.to_vec(),
            reserved_word_sets: meta
                .reserved_sets
                .iter()
                .map(|(name, symbols)| ReservedWordContext {
                    name: g.pool.resolve(*name).to_string(),
                    reserved_words: symbols.clone(),
                })
                .collect(),
        },
        ExtractedLexicalGrammar {
            variables: meta
                .lexical_variables
                .iter()
                .map(|v| Variable {
                    name: g.pool.resolve(v.name).to_string(),
                    kind: v.kind,
                    rule: g.pool.rule(v.root),
                })
                .collect(),
            separators: meta
                .separator_roots
                .iter()
                .map(|&r| g.pool.rule(r))
                .collect(),
        },
    )
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_extraction() {
        let (syntax_grammar, lexical_grammar) = pool_pass(&build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::repeat(Rule::seq(vec![
                    Rule::string("a"),
                    Rule::pattern("b", ""),
                    Rule::choice(vec![
                        Rule::non_terminal(1),
                        Rule::non_terminal(2),
                        Rule::token(Rule::repeat(Rule::choice(vec![
                            Rule::string("c"),
                            Rule::string("d"),
                        ]))),
                    ]),
                ])),
            ),
            Variable::named("rule_1", Rule::pattern("e", "")),
            Variable::named("rule_2", Rule::pattern("b", "")),
            Variable::named(
                "rule_3",
                Rule::seq(vec![Rule::non_terminal(2), Rule::Blank]),
            ),
        ]))
        .unwrap();

        assert_eq!(
            syntax_grammar.variables,
            vec![
                Variable::named(
                    "rule_0",
                    Rule::repeat(Rule::seq(vec![
                        // The string "a" was replaced by a symbol referencing the lexical grammar
                        Rule::terminal(0),
                        // The pattern "b" was replaced by a symbol referencing the lexical grammar
                        Rule::terminal(1),
                        Rule::choice(vec![
                            // The symbol referencing `rule_1` was replaced by a symbol referencing
                            // the lexical grammar.
                            Rule::terminal(3),
                            // The symbol referencing `rule_2` had its index decremented because
                            // `rule_1` was moved to the lexical grammar.
                            Rule::non_terminal(1),
                            // The rule wrapped in `token` was replaced by a symbol referencing
                            // the lexical grammar.
                            Rule::terminal(2),
                        ])
                    ]))
                ),
                // The pattern "e" was only used in one place: as the definition of `rule_1`,
                // so that rule was moved to the lexical grammar. The pattern "b" appeared in
                // two places, so it was not moved into the lexical grammar.
                Variable::named("rule_2", Rule::terminal(1)),
                Variable::named(
                    "rule_3",
                    Rule::seq(vec![Rule::non_terminal(1), Rule::Blank,])
                ),
            ]
        );

        assert_eq!(
            lexical_grammar.variables,
            vec![
                Variable::anonymous("a", Rule::string("a")),
                Variable::auxiliary("rule_0_token1", Rule::pattern("b", "")),
                Variable::auxiliary(
                    "rule_0_token2",
                    Rule::repeat(Rule::choice(vec![Rule::string("c"), Rule::string("d"),]))
                ),
                Variable::named("rule_1", Rule::pattern("e", "")),
            ]
        );
    }

    #[test]
    fn test_start_rule_is_token() {
        let (syntax_grammar, lexical_grammar) = pool_pass(&build_grammar(vec![Variable::named(
            "rule_0",
            Rule::string("hello"),
        )]))
        .unwrap();

        assert_eq!(
            syntax_grammar.variables,
            vec![Variable::named("rule_0", Rule::terminal(0)),]
        );
        assert_eq!(
            lexical_grammar.variables,
            vec![Variable::anonymous("hello", Rule::string("hello")),]
        );
    }

    #[test]
    fn test_extracting_extra_symbols() {
        let mut grammar = build_grammar(vec![
            Variable::named("rule_0", Rule::string("x")),
            Variable::named("comment", Rule::pattern("//.*", "")),
        ]);
        grammar.extra_symbols = vec![Rule::string(" "), Rule::non_terminal(1)];

        let (syntax_grammar, lexical_grammar) = pool_pass(&grammar).unwrap();
        assert_eq!(syntax_grammar.extra_symbols, vec![Symbol::terminal(1),]);
        assert_eq!(lexical_grammar.separators, vec![Rule::string(" "),]);
    }

    #[test]
    fn test_extract_externals() {
        let mut grammar = build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::seq(vec![
                    Rule::external(0),
                    Rule::string("a"),
                    Rule::non_terminal(1),
                    Rule::non_terminal(2),
                ]),
            ),
            Variable::named("rule_1", Rule::string("b")),
            Variable::named("rule_2", Rule::string("c")),
        ]);
        grammar.external_tokens = vec![
            Variable::named("external_0", Rule::external(0)),
            Variable::anonymous("a", Rule::string("a")),
            Variable::named("rule_2", Rule::non_terminal(2)),
        ];

        let (syntax_grammar, _) = pool_pass(&grammar).unwrap();

        assert_eq!(
            syntax_grammar.external_tokens,
            vec![
                ExternalToken {
                    name: "external_0".to_string(),
                    kind: VariableType::Named,
                    corresponding_internal_token: None,
                },
                ExternalToken {
                    name: "a".to_string(),
                    kind: VariableType::Anonymous,
                    corresponding_internal_token: Some(Symbol::terminal(0)),
                },
                ExternalToken {
                    name: "rule_2".to_string(),
                    kind: VariableType::Named,
                    corresponding_internal_token: Some(Symbol::terminal(2)),
                },
            ]
        );
    }

    #[test]
    fn test_error_on_external_with_same_name_as_non_terminal() {
        let mut grammar = build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
            ),
            Variable::named(
                "rule_1",
                Rule::seq(vec![Rule::non_terminal(2), Rule::non_terminal(2)]),
            ),
            Variable::named("rule_2", Rule::string("a")),
        ]);
        grammar.external_tokens = vec![Variable::named("rule_1", Rule::non_terminal(1))];

        let result = pool_pass(&grammar);
        assert!(result.is_err(), "Expected an error but got no error");
        let err = result.err().unwrap();
        assert_eq!(
            err.to_string(),
            "Rule 'rule_1' cannot be used as both an external token and a non-terminal rule"
        );
    }

    #[test]
    fn test_extraction_on_hidden_terminal() {
        let (syntax_grammar, lexical_grammar) = pool_pass(&build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::hidden("_rule_1", Rule::string("a")),
        ]))
        .unwrap();

        // The rule `_rule_1` should not "absorb" the
        // terminal "a", since it is hidden,
        // so we expect two variables still
        assert_eq!(
            syntax_grammar.variables,
            vec![
                Variable::named("rule_0", Rule::non_terminal(1)),
                Variable::hidden("_rule_1", Rule::terminal(0)),
            ]
        );

        // We should not have a hidden rule in our lexical grammar, only the terminal "a"
        assert_eq!(
            lexical_grammar.variables,
            vec![Variable::anonymous("a", Rule::string("a"))]
        );
    }

    #[test]
    fn test_extraction_with_empty_string() {
        assert!(
            pool_pass(&build_grammar(vec![
                Variable::named("rule_0", Rule::non_terminal(1)),
                Variable::hidden("_rule_1", Rule::string("")),
            ]))
            .is_err()
        );
    }

    fn build_grammar(variables: Vec<Variable>) -> InternedGrammar {
        InternedGrammar {
            variables,
            ..Default::default()
        }
    }

    fn pool_from_interned(
        g: &InternedGrammar,
    ) -> (PoolGrammar, super::super::intern_symbols::PoolInternedMeta) {
        use crate::rule_pool::{PoolReservedSet, PoolVariable, RulePool};
        let mut pool = RulePool::default();
        let variables = g
            .variables
            .iter()
            .map(|v| PoolVariable {
                name: pool.intern(&v.name),
                root: pool.add_rule(&v.rule),
            })
            .collect();
        let external_roots = g
            .external_tokens
            .iter()
            .map(|v| pool.add_rule(&v.rule))
            .collect();
        let extra_roots = g.extra_symbols.iter().map(|r| pool.add_rule(r)).collect();
        let reserved_sets = g
            .reserved_word_sets
            .iter()
            .map(|set| PoolReservedSet {
                name: pool.intern(&set.name),
                roots: set
                    .reserved_words
                    .iter()
                    .map(|r| pool.add_rule(r))
                    .collect(),
            })
            .collect();
        let external_tokens = g
            .external_tokens
            .iter()
            .map(|v| ((!v.name.is_empty()).then(|| pool.intern(&v.name)), v.kind))
            .collect();
        let meta = super::super::intern_symbols::PoolInternedMeta {
            kinds: g.variables.iter().map(|v| v.kind).collect(),
            external_tokens,
            supertypes: g.supertype_symbols.clone(),
            conflicts: g.expected_conflicts.clone(),
            inline: g.variables_to_inline.clone(),
            word: g.word_token,
        };
        let pg = PoolGrammar {
            pool,
            name: String::new(),
            variables,
            external_roots,
            extra_roots,
            reserved_sets,
            supertype_names: Vec::new(),
            conflict_names: Vec::new(),
            inline_names: Vec::new(),
            word_name: None,
            precedence_orderings: g.precedence_orderings.clone(),
        };
        (pg, meta)
    }

    fn pool_pass(
        g: &InternedGrammar,
    ) -> ExtractTokensResult<(ExtractedSyntaxGrammar, ExtractedLexicalGrammar)> {
        let (mut pg, meta) = pool_from_interned(g);
        let pool_meta = extract_tokens(&mut pg, &meta)?;
        Ok(materialize_extracted(
            &pg,
            &pool_meta,
            &pg.precedence_orderings,
        ))
    }

    #[test]
    fn test_errors() {
        // Empty string outside the first rule.
        let g = build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::hidden("_rule_1", Rule::string("")),
        ]);
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "The rule `_rule_1` contains an empty string.

Tree-sitter does not support syntactic rules that contain an empty string
unless they are used only as the grammar's start rule.
"
        );

        // External token that is also a non-terminal.
        let mut g = build_grammar(vec![
            Variable::named(
                "rule_0",
                Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
            ),
            Variable::named(
                "rule_1",
                Rule::seq(vec![Rule::non_terminal(2), Rule::non_terminal(2)]),
            ),
            Variable::named("rule_2", Rule::string("a")),
        ]);
        g.external_tokens = vec![Variable::named("rule_1", Rule::non_terminal(1))];
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "Rule 'rule_1' cannot be used as both an external token and a non-terminal rule"
        );

        // Supertype resolving to a terminal.
        let mut g = build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::named("rule_1", Rule::string("a")),
        ]);
        g.supertype_symbols = vec![Symbol::non_terminal(1)];
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "Terminal rule 'rule_1' cannot be used as a supertype"
        );
    }
}
