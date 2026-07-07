use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{ExtractedLexicalGrammar, ExtractedSyntaxGrammar, InternedGrammar};
use crate::{
    grammars::{ExternalToken, ReservedWordContext, Variable, VariableType},
    rules::{MetadataParams, Rule, Symbol, SymbolType},
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

pub(super) fn extract_tokens(
    mut grammar: InternedGrammar,
) -> ExtractTokensResult<(ExtractedSyntaxGrammar, ExtractedLexicalGrammar)> {
    let mut extractor = TokenExtractor {
        current_variable_name: String::new(),
        current_variable_token_count: 0,
        is_first_rule: false,
        extracted_variables: Vec::new(),
        extracted_usage_counts: Vec::new(),
    };

    for (i, variable) in &mut grammar.variables.iter_mut().enumerate() {
        extractor.extract_tokens_in_variable(i == 0, variable)?;
    }

    for variable in &mut grammar.external_tokens {
        extractor.extract_tokens_in_variable(false, variable)?;
    }

    let mut lexical_variables = Vec::with_capacity(extractor.extracted_variables.len());
    for variable in extractor.extracted_variables {
        lexical_variables.push(variable);
    }

    // If a variable's entire rule was extracted as a token and that token didn't
    // appear within any other rule, then remove that variable from the syntax
    // grammar, giving its name to the token in the lexical grammar. Any symbols
    // that pointed to that variable will need to be updated to point to the
    // variable in the lexical grammar. Symbols that pointed to later variables
    // will need to have their indices decremented.
    let mut variables = Vec::with_capacity(grammar.variables.len());
    let mut symbol_replacer = SymbolReplacer {
        replacements: FxHashMap::default(),
    };
    for (i, variable) in grammar.variables.into_iter().enumerate() {
        if let Rule::Symbol(Symbol {
            kind: SymbolType::Terminal,
            index,
        }) = variable.rule
            && i > 0
            && extractor.extracted_usage_counts[index] == 1
        {
            let lexical_variable = &mut lexical_variables[index];
            if lexical_variable.kind == VariableType::Auxiliary
                || variable.kind != VariableType::Hidden
            {
                lexical_variable.kind = variable.kind;
                lexical_variable.name = variable.name;
                symbol_replacer.replacements.insert(i, index);
                continue;
            }
        }
        variables.push(variable);
    }

    for variable in &mut variables {
        variable.rule = symbol_replacer.replace_symbols_in_rule(&variable.rule);
    }

    let expected_conflicts = grammar
        .expected_conflicts
        .into_iter()
        .map(|conflict| {
            let mut result = conflict
                .iter()
                .map(|symbol| symbol_replacer.replace_symbol(*symbol))
                .collect::<Vec<_>>();
            result.sort_unstable();
            result.dedup();
            result
        })
        .collect();

    let supertype_symbols: Vec<Symbol> = grammar
        .supertype_symbols
        .into_iter()
        .map(|symbol| symbol_replacer.replace_symbol(symbol))
        .collect();
    for supertype_symbol in &supertype_symbols {
        if supertype_symbol.is_terminal() {
            Err(ExtractTokensError::SupertypeTerminal(
                lexical_variables[supertype_symbol.index].name.clone(),
            ))?;
        }
    }

    let variables_to_inline = grammar
        .variables_to_inline
        .into_iter()
        .map(|symbol| symbol_replacer.replace_symbol(symbol))
        .collect();

    let mut separators = Vec::new();
    let mut extra_symbols = Vec::new();
    for rule in grammar.extra_symbols {
        if let Rule::Symbol(symbol) = rule {
            extra_symbols.push(symbol_replacer.replace_symbol(symbol));
        } else if let Some(index) = lexical_variables.iter().position(|v| v.rule == rule) {
            extra_symbols.push(Symbol::terminal(index));
        } else {
            separators.push(rule);
        }
    }

    let mut external_tokens = Vec::with_capacity(grammar.external_tokens.len());
    for external_token in grammar.external_tokens {
        let rule = symbol_replacer.replace_symbols_in_rule(&external_token.rule);
        if let Rule::Symbol(symbol) = rule {
            if symbol.is_non_terminal() {
                Err(ExtractTokensError::ExternalTokenNonTerminal(
                    variables[symbol.index].name.clone(),
                ))?;
            }

            if symbol.is_external() {
                external_tokens.push(ExternalToken {
                    name: external_token.name,
                    kind: external_token.kind,
                    corresponding_internal_token: None,
                });
            } else {
                external_tokens.push(ExternalToken {
                    name: lexical_variables[symbol.index].name.clone(),
                    kind: external_token.kind,
                    corresponding_internal_token: Some(symbol),
                });
            }
        } else {
            Err(ExtractTokensError::NonSymbolExternalToken)?;
        }
    }

    let word_token = if let Some(token) = grammar.word_token {
        let token = symbol_replacer.replace_symbol(token);
        if token.is_non_terminal() {
            let word_token_variable = &variables[token.index];
            let conflicting_symbol_name = variables
                .iter()
                .enumerate()
                .find(|(i, v)| *i != token.index && v.rule == word_token_variable.rule)
                .map(|(_, v)| v.name.clone());

            Err(ExtractTokensError::WordToken(NonTerminalWordTokenError {
                symbol_name: word_token_variable.name.clone(),
                conflicting_symbol_name,
            }))?;
        }
        Some(token)
    } else {
        None
    };

    let mut reserved_word_contexts = Vec::with_capacity(grammar.reserved_word_sets.len());
    for reserved_word_context in grammar.reserved_word_sets {
        let mut reserved_words = Vec::with_capacity(reserved_word_contexts.len());
        for reserved_rule in reserved_word_context.reserved_words {
            if let Rule::Symbol(symbol) = reserved_rule {
                reserved_words.push(symbol_replacer.replace_symbol(symbol));
            } else if let Some(index) = lexical_variables
                .iter()
                .position(|v| v.rule == reserved_rule)
            {
                reserved_words.push(Symbol::terminal(index));
            } else {
                let rule = if let Rule::Metadata { rule, .. } = &reserved_rule {
                    rule.as_ref()
                } else {
                    &reserved_rule
                };
                let token_name = match rule {
                    Rule::String(s) => s.clone(),
                    Rule::Pattern(p, _) => p.clone(),
                    _ => "unknown".to_string(),
                };
                Err(ExtractTokensError::NonTokenReservedWord(token_name))?;
            }
        }
        reserved_word_contexts.push(ReservedWordContext {
            name: reserved_word_context.name,
            reserved_words,
        });
    }

    Ok((
        ExtractedSyntaxGrammar {
            variables,
            expected_conflicts,
            extra_symbols,
            variables_to_inline,
            supertype_symbols,
            external_tokens,
            word_token,
            precedence_orderings: grammar.precedence_orderings,
            reserved_word_sets: reserved_word_contexts,
        },
        ExtractedLexicalGrammar {
            variables: lexical_variables,
            separators,
        },
    ))
}

struct TokenExtractor {
    current_variable_name: String,
    current_variable_token_count: usize,
    is_first_rule: bool,
    extracted_variables: Vec<Variable>,
    extracted_usage_counts: Vec<usize>,
}

struct SymbolReplacer {
    replacements: FxHashMap<usize, usize>,
}

impl TokenExtractor {
    fn extract_tokens_in_variable(
        &mut self,
        is_first: bool,
        variable: &mut Variable,
    ) -> ExtractTokensResult<()> {
        self.current_variable_name.clear();
        self.current_variable_name.push_str(&variable.name);
        self.current_variable_token_count = 0;
        self.is_first_rule = is_first;
        variable.rule = self.extract_tokens_in_rule(&variable.rule)?;
        Ok(())
    }

    fn extract_tokens_in_rule(&mut self, input: &Rule) -> ExtractTokensResult<Rule> {
        match input {
            Rule::String(name) => Ok(self.extract_token(input, Some(name))?.into()),
            Rule::Pattern(..) => Ok(self.extract_token(input, None)?.into()),
            Rule::Metadata { params, rule } => {
                if params.is_token {
                    let mut params = params.clone();
                    params.is_token = false;

                    let string_value = if let Rule::String(value) = rule.as_ref() {
                        Some(value)
                    } else {
                        None
                    };

                    let rule_to_extract = if params == MetadataParams::default() {
                        rule.as_ref()
                    } else {
                        input
                    };

                    Ok(self.extract_token(rule_to_extract, string_value)?.into())
                } else {
                    Ok(Rule::Metadata {
                        params: params.clone(),
                        rule: Box::new(self.extract_tokens_in_rule(rule)?),
                    })
                }
            }
            Rule::Repeat(content) => Ok(Rule::Repeat(Box::new(
                self.extract_tokens_in_rule(content)?,
            ))),
            Rule::Seq(elements) => Ok(Rule::Seq(
                elements
                    .iter()
                    .map(|e| self.extract_tokens_in_rule(e))
                    .collect::<ExtractTokensResult<Vec<_>>>()?,
            )),
            Rule::Choice(elements) => Ok(Rule::Choice(
                elements
                    .iter()
                    .map(|e| self.extract_tokens_in_rule(e))
                    .collect::<ExtractTokensResult<Vec<_>>>()?,
            )),
            Rule::Reserved { rule, context_name } => Ok(Rule::Reserved {
                rule: Box::new(self.extract_tokens_in_rule(rule)?),
                context_name: context_name.clone(),
            }),
            _ => Ok(input.clone()),
        }
    }

    fn extract_token(
        &mut self,
        rule: &Rule,
        string_value: Option<&String>,
    ) -> ExtractTokensResult<Symbol> {
        for (i, variable) in self.extracted_variables.iter_mut().enumerate() {
            if variable.rule == *rule {
                self.extracted_usage_counts[i] += 1;
                return Ok(Symbol::terminal(i));
            }
        }

        let index = self.extracted_variables.len();
        let variable = if let Some(string_value) = string_value {
            if string_value.is_empty() && !self.is_first_rule {
                Err(ExtractTokensError::EmptyString(
                    self.current_variable_name.clone(),
                ))?;
            }
            Variable {
                name: string_value.clone(),
                kind: VariableType::Anonymous,
                rule: rule.clone(),
            }
        } else {
            self.current_variable_token_count += 1;
            Variable {
                name: format!(
                    "{}_token{}",
                    self.current_variable_name, self.current_variable_token_count
                ),
                kind: VariableType::Auxiliary,
                rule: rule.clone(),
            }
        };

        self.extracted_variables.push(variable);
        self.extracted_usage_counts.push(1);
        Ok(Symbol::terminal(index))
    }
}

impl SymbolReplacer {
    fn replace_symbols_in_rule(&mut self, rule: &Rule) -> Rule {
        match rule {
            Rule::Symbol(symbol) => self.replace_symbol(*symbol).into(),
            Rule::Choice(elements) => Rule::Choice(
                elements
                    .iter()
                    .map(|e| self.replace_symbols_in_rule(e))
                    .collect(),
            ),
            Rule::Seq(elements) => Rule::Seq(
                elements
                    .iter()
                    .map(|e| self.replace_symbols_in_rule(e))
                    .collect(),
            ),
            Rule::Repeat(content) => Rule::Repeat(Box::new(self.replace_symbols_in_rule(content))),
            Rule::Metadata { rule, params } => Rule::Metadata {
                params: params.clone(),
                rule: Box::new(self.replace_symbols_in_rule(rule)),
            },
            Rule::Reserved { rule, context_name } => Rule::Reserved {
                rule: Box::new(self.replace_symbols_in_rule(rule)),
                context_name: context_name.clone(),
            },
            _ => rule.clone(),
        }
    }

    fn replace_symbol(&self, symbol: Symbol) -> Symbol {
        if !symbol.is_non_terminal() {
            return symbol;
        }

        if let Some(replacement) = self.replacements.get(&symbol.index) {
            return Symbol::terminal(*replacement);
        }

        let mut adjusted_index = symbol.index;
        for replaced_index in self.replacements.keys() {
            if *replaced_index < symbol.index {
                adjusted_index -= 1;
            }
        }

        Symbol::non_terminal(adjusted_index)
    }
}

// --- pool-based pass (replaces the Rule-based one at the container flip) ---

use super::intern_symbols::PoolInternedMeta;
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

/// In-place token extraction over one root, mirroring master's boundary
/// rules: `String`/`Pattern` always; `token(...)` metadata extracts the inner
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
                    let boundary = if cleaned == Params::default() { rule } else { id };
                    let i =
                        ex.extract(pool, boundary, string_name, var_name, &mut counter, is_first)?;
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

pub(super) fn extract_tokens_pool(
    g: &mut PoolGrammar,
    interned: &PoolInternedMeta,
) -> ExtractTokensResult<PoolExtractedMeta> {
    let mut ex = PoolTokenExtractor::default();
    let mut stack = Vec::new();
    for (i, v) in g.variables.iter().enumerate() {
        extract_in_root(&mut g.pool, v.root, Some(v.name), i == 0, &mut ex, &mut stack)?;
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
                    PoolNode::String(s) | PoolNode::Pattern(s, _) => {
                        g.pool.resolve(s).to_string()
                    }
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

/// Materialize the pool pass's output for A/B verification. Deleted at the
/// container flip.
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
        let (syntax_grammar, lexical_grammar) = extract_tokens(build_grammar(vec![
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
        let (syntax_grammar, lexical_grammar) =
            extract_tokens(build_grammar(vec![Variable::named(
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

        let (syntax_grammar, lexical_grammar) = extract_tokens(grammar).unwrap();
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

        let (syntax_grammar, _) = extract_tokens(grammar).unwrap();

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

        let result = extract_tokens(grammar);
        assert!(result.is_err(), "Expected an error but got no error");
        let err = result.err().unwrap();
        assert_eq!(
            err.to_string(),
            "Rule 'rule_1' cannot be used as both an external token and a non-terminal rule"
        );
    }

    #[test]
    fn test_extraction_on_hidden_terminal() {
        let (syntax_grammar, lexical_grammar) = extract_tokens(build_grammar(vec![
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
            extract_tokens(build_grammar(vec![
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
                roots: set.reserved_words.iter().map(|r| pool.add_rule(r)).collect(),
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

    fn assert_pool_matches_master(g: InternedGrammar) {
        let (mut pg, meta) = pool_from_interned(&g);
        let master = extract_tokens(g);
        let pool_result = extract_tokens_pool(&mut pg, &meta);
        match (master, pool_result) {
            (Ok((master_syntax, master_lexical)), Ok(pool_meta)) => {
                let (pool_syntax, pool_lexical) =
                    materialize_extracted(&pg, &pool_meta, &pg.precedence_orderings);
                assert_eq!(pool_syntax, master_syntax);
                assert_eq!(pool_lexical, master_lexical);
            }
            (Err(master_err), Err(pool_err)) => {
                assert_eq!(pool_err.to_string(), master_err.to_string());
            }
            (master, pool) => panic!("master {master:?} but pool {pool:?}"),
        }
    }

    #[test]
    fn pool_extract_matches_master() {
        // The dedup/absorption shape from test_extraction.
        assert_pool_matches_master(build_grammar(vec![
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
        ]));

        // Start rule fully tokenized; hidden terminal not absorbed.
        assert_pool_matches_master(build_grammar(vec![Variable::named(
            "rule_0",
            Rule::string("hello"),
        )]));
        assert_pool_matches_master(build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::hidden("_rule_1", Rule::string("a")),
        ]));

        // Extras routing (symbol, lexical match, separator).
        let mut g = build_grammar(vec![
            Variable::named("rule_0", Rule::string("x")),
            Variable::named("comment", Rule::pattern("//.*", "")),
        ]);
        g.extra_symbols = vec![Rule::string(" "), Rule::non_terminal(1)];
        assert_pool_matches_master(g);

        // Externals: pure external, anonymous internal, named internal.
        let mut g = build_grammar(vec![
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
        g.external_tokens = vec![
            Variable::named("external_0", Rule::external(0)),
            Variable::anonymous("a", Rule::string("a")),
            Variable::named("rule_2", Rule::non_terminal(2)),
        ];
        assert_pool_matches_master(g);
    }

    #[test]
    fn pool_extract_matches_master_errors() {
        // Empty string outside the first rule.
        assert_pool_matches_master(build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::hidden("_rule_1", Rule::string("")),
        ]));

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
        assert_pool_matches_master(g);

        // Supertype resolving to a terminal.
        let mut g = build_grammar(vec![
            Variable::named("rule_0", Rule::non_terminal(1)),
            Variable::named("rule_1", Rule::string("a")),
        ]);
        g.supertype_symbols = vec![Symbol::non_terminal(1)];
        assert_pool_matches_master(g);
    }
}
