use regex_syntax::{
    ParserBuilder,
    hir::{Class, Hir, HirKind},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ExtractedLexicalGrammar;
use crate::{
    grammars::{LexicalGrammar, LexicalVariable},
    nfa::{CharacterSet, Nfa, NfaState},
    rules::{Precedence, Rule},
};

struct NfaBuilder {
    nfa: Nfa,
    is_sep: bool,
    precedence_stack: Vec<i32>,
}

pub type ExpandTokensResult<T> = Result<T, ExpandTokensError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExpandTokensError {
    #[error(
        "The rule `{0}` matches the empty string.
Tree-sitter does not support syntactic rules that match the empty string
unless they are used only as the grammar's start rule.
"
    )]
    EmptyString(String),
    #[error(transparent)]
    Processing(ExpandTokensProcessingError),
    #[error(transparent)]
    ExpandRule(ExpandRuleError),
}

#[derive(Debug, Error, Serialize, Deserialize)]
pub struct ExpandTokensProcessingError {
    rule: String,
    error: ExpandRuleError,
}

impl std::fmt::Display for ExpandTokensProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "Error processing rule {}: Grammar error: Unexpected rule {:?}",
            self.rule, self.error
        )?;
        Ok(())
    }
}

fn get_implicit_precedence(rule: &Rule) -> i32 {
    match rule {
        Rule::String(_) => 2,
        Rule::Metadata { rule, params } => {
            if params.is_main_token {
                get_implicit_precedence(rule) + 1
            } else {
                get_implicit_precedence(rule)
            }
        }
        _ => 0,
    }
}

const fn get_completion_precedence(rule: &Rule) -> i32 {
    if let Rule::Metadata { params, .. } = rule
        && let Precedence::Integer(p) = params.precedence
    {
        return p;
    }
    0
}

pub fn expand_tokens(mut grammar: ExtractedLexicalGrammar) -> ExpandTokensResult<LexicalGrammar> {
    let mut builder = NfaBuilder {
        nfa: Nfa::new(),
        is_sep: true,
        precedence_stack: vec![0],
    };

    let separator_rule = if grammar.separators.is_empty() {
        Rule::Blank
    } else {
        grammar.separators.push(Rule::Blank);
        Rule::repeat(Rule::choice(grammar.separators))
    };

    let mut variables = Vec::with_capacity(grammar.variables.len());
    for (i, variable) in grammar.variables.into_iter().enumerate() {
        if variable.rule.is_empty() {
            Err(ExpandTokensError::EmptyString(variable.name.clone()))?;
        }

        let is_immediate_token = match &variable.rule {
            Rule::Metadata { params, .. } => params.is_main_token,
            _ => false,
        };

        builder.is_sep = false;
        builder.nfa.states.push(NfaState::Accept {
            variable_index: i,
            precedence: get_completion_precedence(&variable.rule),
        });
        let last_state_id = builder.nfa.last_state_id();
        builder
            .expand_rule(&variable.rule, last_state_id)
            .map_err(|e| {
                ExpandTokensError::Processing(ExpandTokensProcessingError {
                    rule: variable.name.clone(),
                    error: e,
                })
            })?;

        if !is_immediate_token {
            builder.is_sep = true;
            let last_state_id = builder.nfa.last_state_id();
            builder
                .expand_rule(&separator_rule, last_state_id)
                .map_err(ExpandTokensError::ExpandRule)?;
        }

        variables.push(LexicalVariable {
            name: variable.name,
            kind: variable.kind,
            implicit_precedence: get_implicit_precedence(&variable.rule),
            start_state: builder.nfa.last_state_id(),
        });
    }

    Ok(LexicalGrammar {
        nfa: builder.nfa,
        variables,
    })
}

pub type ExpandRuleResult<T> = Result<T, ExpandRuleError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExpandRuleError {
    #[error("Grammar error: Unexpected rule {0:?}")]
    UnexpectedRule(Rule),
    #[error("{0}")]
    Parse(String),
    #[error(transparent)]
    ExpandRegex(ExpandRegexError),
}

pub type ExpandRegexResult<T> = Result<T, ExpandRegexError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum ExpandRegexError {
    #[error("{0}")]
    Utf8(String),
    #[error("Regex error: Assertions are not supported")]
    Assertion,
}

impl NfaBuilder {
    fn expand_rule(&mut self, rule: &Rule, mut next_state_id: u32) -> ExpandRuleResult<bool> {
        match rule {
            Rule::Pattern(s, f) => {
                // With unicode enabled, `\w`, `\s` and `\d` expand to character sets that are much
                // larger than intended, so we replace them with the actual
                // character sets they should represent. If the full unicode range
                // of `\w`, `\s` or `\d` are needed then `\p{L}`, `\p{Z}` and `\p{N}` should be
                // used.
                let s = s
                    .replace(r"\w", r"[0-9A-Za-z_]")
                    .replace(r"\s", r"[\t-\r ]")
                    .replace(r"\d", r"[0-9]")
                    .replace(r"\W", r"[^0-9A-Za-z_]")
                    .replace(r"\S", r"[^\t-\r ]")
                    .replace(r"\D", r"[^0-9]");
                let mut parser = ParserBuilder::new()
                    .case_insensitive(f.contains('i'))
                    .unicode(true)
                    .utf8(false)
                    .build();
                let hir = parser
                    .parse(&s)
                    .map_err(|e| ExpandRuleError::Parse(e.to_string()))?;
                self.expand_regex(&hir, next_state_id)
                    .map_err(ExpandRuleError::ExpandRegex)
            }
            Rule::String(s) => {
                for c in s.chars().rev() {
                    self.push_advance(CharacterSet::from_char(c), next_state_id);
                    next_state_id = self.nfa.last_state_id();
                }
                Ok(!s.is_empty())
            }
            Rule::Choice(elements) => {
                let mut alternative_state_ids = Vec::with_capacity(elements.len());
                for element in elements {
                    if self.expand_rule(element, next_state_id)? {
                        alternative_state_ids.push(self.nfa.last_state_id());
                    } else {
                        alternative_state_ids.push(next_state_id);
                    }
                }
                alternative_state_ids.sort_unstable();
                alternative_state_ids.dedup();
                alternative_state_ids.retain(|i| *i != self.nfa.last_state_id());
                for alternative_state_id in alternative_state_ids {
                    self.push_split(alternative_state_id);
                }
                Ok(true)
            }
            Rule::Seq(elements) => {
                let mut result = false;
                for element in elements.iter().rev() {
                    if self.expand_rule(element, next_state_id)? {
                        result = true;
                    }
                    next_state_id = self.nfa.last_state_id();
                }
                Ok(result)
            }
            Rule::Repeat(rule) => {
                self.nfa.states.push(NfaState::Accept {
                    variable_index: 0,
                    precedence: 0,
                }); // Placeholder for split
                let split_state_id = self.nfa.last_state_id();
                if self.expand_rule(rule, split_state_id)? {
                    self.nfa.states[split_state_id as usize] =
                        NfaState::Split(self.nfa.last_state_id(), next_state_id);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Rule::Metadata { rule, params } => {
                let has_precedence = if let Precedence::Integer(precedence) = &params.precedence {
                    self.precedence_stack.push(*precedence);
                    true
                } else {
                    false
                };
                let result = self.expand_rule(rule, next_state_id);
                if has_precedence {
                    self.precedence_stack.pop();
                }
                result
            }
            Rule::Blank => Ok(false),
            _ => Err(ExpandRuleError::UnexpectedRule(rule.clone()))?,
        }
    }

    fn expand_regex(&mut self, hir: &Hir, mut next_state_id: u32) -> ExpandRegexResult<bool> {
        match hir.kind() {
            HirKind::Empty => Ok(false),
            HirKind::Literal(literal) => {
                for character in std::str::from_utf8(&literal.0)
                    .map_err(|e| ExpandRegexError::Utf8(e.to_string()))?
                    .chars()
                    .rev()
                {
                    let char_set = CharacterSet::from_char(character);
                    self.push_advance(char_set, next_state_id);
                    next_state_id = self.nfa.last_state_id();
                }

                Ok(true)
            }
            HirKind::Class(class) => match class {
                Class::Unicode(class) => {
                    let mut chars = CharacterSet::default();
                    for c in class.ranges() {
                        chars = chars.add_range(c.start(), c.end());
                    }

                    // For some reason, the long s `ſ` is included if the letter `s` is in a
                    // pattern, so we remove it.
                    if chars.range_count() == 3
                        && chars
                            .ranges()
                            // exact check to ensure that `ſ` wasn't intentionally added.
                            .all(|r| ['s'..='s', 'S'..='S', 'ſ'..='ſ'].contains(&r))
                    {
                        chars = chars.difference(CharacterSet::from_char('ſ'));
                    }
                    self.push_advance(chars, next_state_id);
                    Ok(true)
                }
                Class::Bytes(bytes_class) => {
                    let mut chars = CharacterSet::default();
                    for c in bytes_class.ranges() {
                        chars = chars.add_range(c.start().into(), c.end().into());
                    }
                    self.push_advance(chars, next_state_id);
                    Ok(true)
                }
            },
            HirKind::Look(_) => Err(ExpandRegexError::Assertion)?,
            HirKind::Repetition(repetition) => match (repetition.min, repetition.max) {
                (0, Some(1)) => self.expand_zero_or_one(&repetition.sub, next_state_id),
                (1, None) => self.expand_one_or_more(&repetition.sub, next_state_id),
                (0, None) => self.expand_zero_or_more(&repetition.sub, next_state_id),
                (min, Some(max)) if min == max => {
                    self.expand_count(&repetition.sub, min, next_state_id)
                }
                (min, None) => {
                    if self.expand_zero_or_more(&repetition.sub, next_state_id)? {
                        self.expand_count(&repetition.sub, min, next_state_id)
                    } else {
                        Ok(false)
                    }
                }
                (min, Some(max)) => {
                    let mut result = self.expand_count(&repetition.sub, min, next_state_id)?;
                    for _ in min..max {
                        if result {
                            next_state_id = self.nfa.last_state_id();
                        }
                        if self.expand_zero_or_one(&repetition.sub, next_state_id)? {
                            result = true;
                        }
                    }
                    Ok(result)
                }
            },
            HirKind::Capture(capture) => self.expand_regex(&capture.sub, next_state_id),
            HirKind::Concat(concat) => {
                let mut result = false;
                for hir in concat.iter().rev() {
                    if self.expand_regex(hir, next_state_id)? {
                        result = true;
                        next_state_id = self.nfa.last_state_id();
                    }
                }
                Ok(result)
            }
            HirKind::Alternation(alternations) => {
                let mut alternative_state_ids = Vec::with_capacity(alternations.len());
                for hir in alternations {
                    if self.expand_regex(hir, next_state_id)? {
                        alternative_state_ids.push(self.nfa.last_state_id());
                    } else {
                        alternative_state_ids.push(next_state_id);
                    }
                }
                alternative_state_ids.sort_unstable();
                alternative_state_ids.dedup();
                alternative_state_ids.retain(|i| *i != self.nfa.last_state_id());
                for alternative_state_id in alternative_state_ids {
                    self.push_split(alternative_state_id);
                }
                Ok(true)
            }
        }
    }

    fn expand_one_or_more(&mut self, hir: &Hir, next_state_id: u32) -> ExpandRegexResult<bool> {
        self.nfa.states.push(NfaState::Accept {
            variable_index: 0,
            precedence: 0,
        }); // Placeholder for split
        let split_state_id = self.nfa.last_state_id();
        if self.expand_regex(hir, split_state_id)? {
            self.nfa.states[split_state_id as usize] =
                NfaState::Split(self.nfa.last_state_id(), next_state_id);
            Ok(true)
        } else {
            self.nfa.states.pop();
            Ok(false)
        }
    }

    fn expand_zero_or_one(&mut self, hir: &Hir, next_state_id: u32) -> ExpandRegexResult<bool> {
        if self.expand_regex(hir, next_state_id)? {
            self.push_split(next_state_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expand_zero_or_more(&mut self, hir: &Hir, next_state_id: u32) -> ExpandRegexResult<bool> {
        if self.expand_one_or_more(hir, next_state_id)? {
            self.push_split(next_state_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    fn expand_count(
        &mut self,
        hir: &Hir,
        count: u32,
        mut next_state_id: u32,
    ) -> ExpandRegexResult<bool> {
        let mut result = false;
        for _ in 0..count {
            if self.expand_regex(hir, next_state_id)? {
                result = true;
                next_state_id = self.nfa.last_state_id();
            }
        }
        Ok(result)
    }

    fn push_advance(&mut self, chars: CharacterSet, state_id: u32) {
        let precedence = *self.precedence_stack.last().unwrap();
        self.nfa.states.push(NfaState::Advance {
            chars,
            state_id,
            precedence,
            is_sep: self.is_sep,
        });
    }

    fn push_split(&mut self, state_id: u32) {
        let last_state_id = self.nfa.last_state_id();
        self.nfa
            .states
            .push(NfaState::Split(state_id, last_state_id));
    }
}

// --- pool-based pass (replaces the Rule-based one at the container flip) ---

use super::extract_tokens::PoolLexicalVariable;
use crate::rule_pool::{Node, NodeId, Prec, RulePool};

/// Pool twin of `expand_tokens`: a read-only walk over the lexical roots
/// (only the combined separator rule appends nodes). Produces the final
/// `LexicalGrammar` directly; strings resolve at the NFA boundary.
pub(super) fn expand_tokens_pool(
    pool: &mut RulePool,
    lexical_variables: &[PoolLexicalVariable],
    separator_roots: &[NodeId],
) -> ExpandTokensResult<LexicalGrammar> {
    let mut builder = NfaBuilder {
        nfa: Nfa::new(),
        is_sep: true,
        precedence_stack: vec![0],
    };

    let separator_root = build_separator(pool, separator_roots);

    let mut variables = Vec::with_capacity(lexical_variables.len());
    for (i, variable) in lexical_variables.iter().enumerate() {
        if subtree_is_empty(pool, variable.root) {
            Err(ExpandTokensError::EmptyString(
                pool.resolve(variable.name).to_string(),
            ))?;
        }

        let is_immediate_token = match pool.node(variable.root) {
            Node::Metadata { params, .. } => pool.params(params).is_main_token,
            _ => false,
        };

        builder.is_sep = false;
        builder.nfa.states.push(NfaState::Accept {
            variable_index: i,
            precedence: completion_precedence(pool, variable.root),
        });
        let last_state_id = builder.nfa.last_state_id();
        builder
            .expand_node(pool, variable.root, last_state_id)
            .map_err(|e| {
                ExpandTokensError::Processing(ExpandTokensProcessingError {
                    rule: pool.resolve(variable.name).to_string(),
                    error: e,
                })
            })?;

        if !is_immediate_token {
            builder.is_sep = true;
            let last_state_id = builder.nfa.last_state_id();
            builder
                .expand_node(pool, separator_root, last_state_id)
                .map_err(ExpandTokensError::ExpandRule)?;
        }

        variables.push(LexicalVariable {
            name: pool.resolve(variable.name).to_string(),
            kind: variable.kind,
            implicit_precedence: implicit_precedence(pool, variable.root),
            start_state: builder.nfa.last_state_id(),
        });
    }

    Ok(LexicalGrammar {
        nfa: builder.nfa,
        variables,
    })
}

/// Mirror of the master separator construction: `Blank` when there are no
/// separators, else `repeat(choice(separators + Blank))` where choice
/// flattens nested choices and dedups (but keeps a single-element `Choice`).
fn build_separator(pool: &mut RulePool, separator_roots: &[NodeId]) -> NodeId {
    if separator_roots.is_empty() {
        return pool.push_node(Node::Blank);
    }
    let blank = pool.push_node(Node::Blank);
    let mut elements: Vec<NodeId> = Vec::with_capacity(separator_roots.len() + 1);
    let mut stack = Vec::with_capacity(separator_roots.len() + 1);
    stack.push(blank);
    stack.extend(separator_roots.iter().rev().copied());
    while let Some(id) = stack.pop() {
        if let Node::Choice(range) = pool.node(id) {
            let base = stack.len();
            stack.extend_from_slice(pool.child_slice(range));
            stack[base..].reverse();
        } else if !elements.iter().any(|&e| pool.subtree_eq(e, id)) {
            elements.push(id);
        }
    }
    let range = pool.push_children(&elements);
    let choice = pool.push_node(Node::Choice(range));
    pool.push_node(Node::Repeat(choice))
}

/// Mirror of `Rule::is_empty`.
fn subtree_is_empty(pool: &RulePool, id: NodeId) -> bool {
    match pool.node(id) {
        Node::String(s) => pool.resolve(s).is_empty(),
        Node::Metadata { rule, .. } | Node::Repeat(rule) | Node::Reserved { rule, .. } => {
            subtree_is_empty(pool, rule)
        }
        Node::Choice(range) => pool
            .child_slice(range)
            .iter()
            .any(|&c| subtree_is_empty(pool, c)),
        Node::Seq(range) => pool
            .child_slice(range)
            .iter()
            .all(|&c| subtree_is_empty(pool, c)),
        _ => false,
    }
}

/// Mirror of `get_implicit_precedence`.
fn implicit_precedence(pool: &RulePool, root: NodeId) -> i32 {
    let mut id = root;
    let mut boost = 0;
    loop {
        match pool.node(id) {
            Node::String(_) => return 2 + boost,
            Node::Metadata { params, rule } => {
                if pool.params(params).is_main_token {
                    boost += 1;
                }
                id = rule;
            }
            _ => return boost,
        }
    }
}

/// Mirror of `get_completion_precedence`.
fn completion_precedence(pool: &RulePool, id: NodeId) -> i32 {
    if let Node::Metadata { params, .. } = pool.node(id)
        && let Prec::Integer(p) = pool.params(params).prec
    {
        return p;
    }
    0
}

impl NfaBuilder {
    /// Pool twin of `expand_rule`; shares `expand_regex` and the NFA helpers.
    fn expand_node(
        &mut self,
        pool: &RulePool,
        id: NodeId,
        mut next_state_id: u32,
    ) -> ExpandRuleResult<bool> {
        match pool.node(id) {
            Node::Pattern(s, f) => {
                // See `expand_rule` for why `\w`, `\s` and `\d` are replaced.
                let s = pool
                    .resolve(s)
                    .replace(r"\w", r"[0-9A-Za-z_]")
                    .replace(r"\s", r"[\t-\r ]")
                    .replace(r"\d", r"[0-9]")
                    .replace(r"\W", r"[^0-9A-Za-z_]")
                    .replace(r"\S", r"[^\t-\r ]")
                    .replace(r"\D", r"[^0-9]");
                let mut parser = ParserBuilder::new()
                    .case_insensitive(pool.resolve(f).contains('i'))
                    .unicode(true)
                    .utf8(false)
                    .build();
                let hir = parser
                    .parse(&s)
                    .map_err(|e| ExpandRuleError::Parse(e.to_string()))?;
                self.expand_regex(&hir, next_state_id)
                    .map_err(ExpandRuleError::ExpandRegex)
            }
            Node::String(s) => {
                let s = pool.resolve(s);
                for c in s.chars().rev() {
                    self.push_advance(CharacterSet::from_char(c), next_state_id);
                    next_state_id = self.nfa.last_state_id();
                }
                Ok(!s.is_empty())
            }
            Node::Choice(range) => {
                let elements = pool.child_slice(range);
                let mut alternative_state_ids = Vec::with_capacity(elements.len());
                for &element in elements {
                    if self.expand_node(pool, element, next_state_id)? {
                        alternative_state_ids.push(self.nfa.last_state_id());
                    } else {
                        alternative_state_ids.push(next_state_id);
                    }
                }
                alternative_state_ids.sort_unstable();
                alternative_state_ids.dedup();
                alternative_state_ids.retain(|i| *i != self.nfa.last_state_id());
                for alternative_state_id in alternative_state_ids {
                    self.push_split(alternative_state_id);
                }
                Ok(true)
            }
            Node::Seq(range) => {
                let mut result = false;
                for &element in pool.child_slice(range).iter().rev() {
                    if self.expand_node(pool, element, next_state_id)? {
                        result = true;
                    }
                    next_state_id = self.nfa.last_state_id();
                }
                Ok(result)
            }
            Node::Repeat(rule) => {
                self.nfa.states.push(NfaState::Accept {
                    variable_index: 0,
                    precedence: 0,
                }); // Placeholder for split
                let split_state_id = self.nfa.last_state_id();
                if self.expand_node(pool, rule, split_state_id)? {
                    self.nfa.states[split_state_id as usize] =
                        NfaState::Split(self.nfa.last_state_id(), next_state_id);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
            Node::Metadata { params, rule } => {
                let has_precedence = if let Prec::Integer(precedence) = pool.params(params).prec {
                    self.precedence_stack.push(precedence);
                    true
                } else {
                    false
                };
                let result = self.expand_node(pool, rule, next_state_id);
                if has_precedence {
                    self.precedence_stack.pop();
                }
                result
            }
            Node::Blank => Ok(false),
            Node::NamedSymbol(_) | Node::Sym { .. } | Node::Reserved { .. } => {
                Err(ExpandRuleError::UnexpectedRule(pool.rule(id)))?
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammars::Variable,
        nfa::{NfaCursor, NfaTransition},
    };

    fn simulate_nfa<'a>(grammar: &'a LexicalGrammar, s: &'a str) -> Option<(usize, &'a str)> {
        let start_states = grammar.variables.iter().map(|v| v.start_state).collect();
        let mut cursor = NfaCursor::new(&grammar.nfa, start_states);

        let mut result = None;
        let mut result_precedence = i32::MIN;
        let mut start_char = 0;
        let mut end_char = 0;
        for c in s.chars() {
            for (id, precedence) in cursor.completions() {
                if result.is_none() || result_precedence <= precedence {
                    result = Some((id, &s[start_char..end_char]));
                    result_precedence = precedence;
                }
            }
            if let Some(NfaTransition {
                states,
                is_separator,
                ..
            }) = cursor
                .transitions()
                .into_iter()
                .find(|t| t.characters.contains(c) && t.precedence >= result_precedence)
            {
                cursor.reset(states);
                end_char += c.len_utf8();
                if is_separator {
                    start_char = end_char;
                }
            } else {
                break;
            }
        }

        for (id, precedence) in cursor.completions() {
            if result.is_none() || result_precedence <= precedence {
                result = Some((id, &s[start_char..end_char]));
                result_precedence = precedence;
            }
        }

        result
    }

    #[test]
    fn test_rule_expansion() {
        struct Row {
            rules: Vec<Rule>,
            separators: Vec<Rule>,
            examples: Vec<(&'static str, Option<(usize, &'static str)>)>,
        }

        let table = [
            // regex with sequences and alternatives
            Row {
                rules: vec![Rule::pattern("(a|b|c)d(e|f|g)h?", "")],
                separators: vec![],
                examples: vec![
                    ("ade1", Some((0, "ade"))),
                    ("bdf1", Some((0, "bdf"))),
                    ("bdfh1", Some((0, "bdfh"))),
                    ("ad1", None),
                ],
            },
            // regex with repeats
            Row {
                rules: vec![Rule::pattern("a*", "")],
                separators: vec![],
                examples: vec![("aaa1", Some((0, "aaa"))), ("b", Some((0, "")))],
            },
            // regex with repeats in sequences
            Row {
                rules: vec![Rule::pattern("a((bc)+|(de)*)f", "")],
                separators: vec![],
                examples: vec![
                    ("af1", Some((0, "af"))),
                    ("adedef1", Some((0, "adedef"))),
                    ("abcbcbcf1", Some((0, "abcbcbcf"))),
                    ("a", None),
                ],
            },
            // regex with character ranges
            Row {
                rules: vec![Rule::pattern("[a-fA-F0-9]+", "")],
                separators: vec![],
                examples: vec![("A1ff0.", Some((0, "A1ff0")))],
            },
            // regex with perl character classes
            Row {
                rules: vec![Rule::pattern("\\w\\d\\s", "")],
                separators: vec![],
                examples: vec![("_0  ", Some((0, "_0 ")))],
            },
            // string
            Row {
                rules: vec![Rule::string("abc")],
                separators: vec![],
                examples: vec![("abcd", Some((0, "abc"))), ("ab", None)],
            },
            // complex rule containing strings and regexes
            Row {
                rules: vec![Rule::repeat(Rule::seq(vec![
                    Rule::string("{"),
                    Rule::pattern("[a-f]+", ""),
                    Rule::string("}"),
                ]))],
                separators: vec![],
                examples: vec![
                    ("{a}{", Some((0, "{a}"))),
                    ("{a}{d", Some((0, "{a}"))),
                    ("ab", None),
                ],
            },
            // longest match rule
            Row {
                rules: vec![
                    Rule::pattern("a|bc", ""),
                    Rule::pattern("aa", ""),
                    Rule::pattern("bcd", ""),
                ],
                separators: vec![],
                examples: vec![
                    ("a.", Some((0, "a"))),
                    ("bc.", Some((0, "bc"))),
                    ("aa.", Some((1, "aa"))),
                    ("bcd?", Some((2, "bcd"))),
                    ("b.", None),
                    ("c.", None),
                ],
            },
            // regex with an alternative including the empty string
            Row {
                rules: vec![Rule::pattern("a(b|)+c", "")],
                separators: vec![],
                examples: vec![
                    ("ac.", Some((0, "ac"))),
                    ("abc.", Some((0, "abc"))),
                    ("abbc.", Some((0, "abbc"))),
                ],
            },
            // separators
            Row {
                rules: vec![Rule::pattern("[a-f]+", "")],
                separators: vec![Rule::string("\\\n"), Rule::pattern("\\s", "")],
                examples: vec![
                    ("  a", Some((0, "a"))),
                    ("  \nb", Some((0, "b"))),
                    ("  \\a", None),
                    ("  \\\na", Some((0, "a"))),
                ],
            },
            // shorter tokens with higher precedence
            Row {
                rules: vec![
                    Rule::prec(Precedence::Integer(2), Rule::pattern("abc", "")),
                    Rule::prec(Precedence::Integer(1), Rule::pattern("ab[cd]e", "")),
                    Rule::pattern("[a-e]+", ""),
                ],
                separators: vec![Rule::string("\\\n"), Rule::pattern("\\s", "")],
                examples: vec![
                    ("abceef", Some((0, "abc"))),
                    ("abdeef", Some((1, "abde"))),
                    ("aeeeef", Some((2, "aeeee"))),
                ],
            },
            // immediate tokens with higher precedence
            Row {
                rules: vec![
                    Rule::prec(Precedence::Integer(1), Rule::pattern("[^a]+", "")),
                    Rule::immediate_token(Rule::prec(
                        Precedence::Integer(2),
                        Rule::pattern("[^ab]+", ""),
                    )),
                ],
                separators: vec![Rule::pattern("\\s", "")],
                examples: vec![("cccb", Some((1, "ccc")))],
            },
            Row {
                rules: vec![Rule::seq(vec![
                    Rule::string("a"),
                    Rule::choice(vec![Rule::string("b"), Rule::string("c")]),
                    Rule::string("d"),
                ])],
                separators: vec![],
                examples: vec![
                    ("abd", Some((0, "abd"))),
                    ("acd", Some((0, "acd"))),
                    ("abc", None),
                    ("ad", None),
                    ("d", None),
                    ("a", None),
                ],
            },
            // nested choices within sequences
            Row {
                rules: vec![Rule::seq(vec![
                    Rule::pattern("[0-9]+", ""),
                    Rule::choice(vec![
                        Rule::Blank,
                        Rule::choice(vec![Rule::seq(vec![
                            Rule::choice(vec![Rule::string("e"), Rule::string("E")]),
                            Rule::choice(vec![
                                Rule::Blank,
                                Rule::choice(vec![Rule::string("+"), Rule::string("-")]),
                            ]),
                            Rule::pattern("[0-9]+", ""),
                        ])]),
                    ]),
                ])],
                separators: vec![],
                examples: vec![
                    ("12", Some((0, "12"))),
                    ("12e", Some((0, "12"))),
                    ("12g", Some((0, "12"))),
                    ("12e3", Some((0, "12e3"))),
                    ("12e+", Some((0, "12"))),
                    ("12E+34 +", Some((0, "12E+34"))),
                    ("12e34", Some((0, "12e34"))),
                ],
            },
            // nested groups
            Row {
                rules: vec![Rule::seq(vec![Rule::pattern(r"([^x\\]|\\(.|\n))+", "")])],
                separators: vec![],
                examples: vec![("abcx", Some((0, "abc"))), ("abc\\0x", Some((0, "abc\\0")))],
            },
            // allowing unrecognized escape sequences
            Row {
                rules: vec![
                    // Escaped forward slash (used in JS because '/' is the regex delimiter)
                    Rule::pattern(r"\/", ""),
                    // Escaped quotes
                    Rule::pattern(r#"\"\'"#, ""),
                    // Quote preceded by a literal backslash
                    Rule::pattern(r"[\\']+", ""),
                ],
                separators: vec![],
                examples: vec![
                    ("/", Some((0, "/"))),
                    ("\"\'", Some((1, "\"\'"))),
                    (r"'\'a", Some((2, r"'\'"))),
                ],
            },
            // unicode property escapes
            Row {
                rules: vec![
                    Rule::pattern(r"\p{L}+\P{L}+", ""),
                    Rule::pattern(r"\p{White_Space}+\P{White_Space}+[\p{White_Space}]*", ""),
                ],
                separators: vec![],
                examples: vec![
                    ("  123   abc", Some((1, "  123   "))),
                    ("ბΨƁ___ƀƔ", Some((0, "ბΨƁ___"))),
                ],
            },
            // unicode property escapes in bracketed sets
            Row {
                rules: vec![Rule::pattern(r"[\p{L}\p{Nd}]+", "")],
                separators: vec![],
                examples: vec![("abΨ12٣٣, ok", Some((0, "abΨ12٣٣")))],
            },
            // unicode character escapes
            Row {
                rules: vec![
                    Rule::pattern(r"\u{00dc}", ""),
                    Rule::pattern(r"\U{000000dd}", ""),
                    Rule::pattern(r"\u00de", ""),
                    Rule::pattern(r"\U000000df", ""),
                ],
                separators: vec![],
                examples: vec![
                    ("\u{00dc}", Some((0, "\u{00dc}"))),
                    ("\u{00dd}", Some((1, "\u{00dd}"))),
                    ("\u{00de}", Some((2, "\u{00de}"))),
                    ("\u{00df}", Some((3, "\u{00df}"))),
                ],
            },
            Row {
                rules: vec![
                    Rule::pattern(r"u\{[0-9a-fA-F]+\}", ""),
                    // Already-escaped curly braces
                    Rule::pattern(r"\{[ab]{3}\}", ""),
                    // Unicode codepoints
                    Rule::pattern(r"\u{1000A}", ""),
                    // Unicode codepoints (lowercase)
                    Rule::pattern(r"\u{1000b}", ""),
                ],
                separators: vec![],
                examples: vec![
                    ("u{1234} ok", Some((0, "u{1234}"))),
                    ("{aba}}", Some((1, "{aba}"))),
                    ("\u{1000A}", Some((2, "\u{1000A}"))),
                    ("\u{1000b}", Some((3, "\u{1000b}"))),
                ],
            },
            // Emojis
            Row {
                rules: vec![Rule::pattern(r"\p{Emoji}+", "")],
                separators: vec![],
                examples: vec![
                    ("🐎", Some((0, "🐎"))),
                    ("🐴🐴", Some((0, "🐴🐴"))),
                    ("#0", Some((0, "#0"))), // These chars are technically emojis!
                    ("⻢", None),
                    ("♞", None),
                    ("horse", None),
                ],
            },
            // Intersection
            Row {
                rules: vec![Rule::pattern(r"[[0-7]&&[4-9]]+", "")],
                separators: vec![],
                examples: vec![
                    ("456", Some((0, "456"))),
                    ("64", Some((0, "64"))),
                    ("452", Some((0, "45"))),
                    ("91", None),
                    ("8", None),
                    ("3", None),
                ],
            },
            // Difference
            Row {
                rules: vec![Rule::pattern(r"[[0-9]--[4-7]]+", "")],
                separators: vec![],
                examples: vec![
                    ("123", Some((0, "123"))),
                    ("83", Some((0, "83"))),
                    ("9", Some((0, "9"))),
                    ("124", Some((0, "12"))),
                    ("67", None),
                    ("4", None),
                ],
            },
            // Symmetric difference
            Row {
                rules: vec![Rule::pattern(r"[[0-7]~~[4-9]]+", "")],
                separators: vec![],
                examples: vec![
                    ("123", Some((0, "123"))),
                    ("83", Some((0, "83"))),
                    ("9", Some((0, "9"))),
                    ("124", Some((0, "12"))),
                    ("67", None),
                    ("4", None),
                ],
            },
            // Nested set operations
            Row {
                //               0 1 2 3 4 5 6 7 8 9
                // [0-5]:        y y y y y y
                // [2-4]:            y y y
                // [0-5]--[2-4]: y y       y
                // [3-9]:              y y y y y y y
                // [6-7]:                    y y
                // [3-9]--[5-7]:       y y y     y y
                // final regex:  y y   y y       y y
                rules: vec![Rule::pattern(r"[[[0-5]--[2-4]]~~[[3-9]--[6-7]]]+", "")],
                separators: vec![],
                examples: vec![
                    ("01", Some((0, "01"))),
                    ("432", Some((0, "43"))),
                    ("8", Some((0, "8"))),
                    ("9", Some((0, "9"))),
                    ("2", None),
                    ("567", None),
                ],
            },
        ];

        for Row {
            rules,
            separators,
            examples,
        } in &table
        {
            let variables: Vec<Variable> = rules
                .iter()
                .map(|rule| Variable::named("", rule.clone()))
                .collect();
            let grammar = expand_tokens(ExtractedLexicalGrammar {
                separators: separators.clone(),
                variables: variables.clone(),
            })
            .unwrap();

            let (mut pool, vars, seps) = pool_from_lexical(&variables, separators);
            let pool_grammar = expand_tokens_pool(&mut pool, &vars, &seps).unwrap();
            assert_eq!(pool_grammar, grammar, "pool NFA diverged from master");

            for (haystack, needle) in examples {
                assert_eq!(simulate_nfa(&grammar, haystack), *needle);
            }
        }
    }

    fn pool_from_lexical(
        variables: &[Variable],
        separators: &[Rule],
    ) -> (RulePool, Vec<PoolLexicalVariable>, Vec<NodeId>) {
        let mut pool = RulePool::default();
        let vars = variables
            .iter()
            .map(|v| PoolLexicalVariable {
                name: pool.intern(&v.name),
                kind: v.kind,
                root: pool.add_rule(&v.rule),
            })
            .collect();
        let seps = separators.iter().map(|r| pool.add_rule(r)).collect();
        (pool, vars, seps)
    }

    #[test]
    fn pool_expand_matches_master_errors() {
        let cases: Vec<(Vec<Variable>, Vec<Rule>)> = vec![
            // empty-string token
            (vec![Variable::named("tok", Rule::string(""))], vec![]),
            // unexpected rule: symbol inside a token
            (
                vec![Variable::named(
                    "tok",
                    Rule::Seq(vec![Rule::string("a"), Rule::non_terminal(3)]),
                )],
                vec![],
            ),
            // invalid regex
            (vec![Variable::named("tok", Rule::pattern("[", ""))], vec![]),
            // invalid regex in a separator
            (
                vec![Variable::named("tok", Rule::string("a"))],
                vec![Rule::pattern("(", "")],
            ),
        ];
        for (variables, separators) in cases {
            let master = expand_tokens(ExtractedLexicalGrammar {
                separators: separators.clone(),
                variables: variables.clone(),
            })
            .unwrap_err();
            let (mut pool, vars, seps) = pool_from_lexical(&variables, &separators);
            let pool_err = expand_tokens_pool(&mut pool, &vars, &seps).unwrap_err();
            assert_eq!(pool_err.to_string(), master.to_string());
        }
    }
}
