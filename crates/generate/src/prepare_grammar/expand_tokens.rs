use regex_syntax::{
    ParserBuilder,
    hir::{Class, Hir, HirKind},
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    grammars::{LexicalGrammar, LexicalVariable},
    nfa::{CharacterSet, Nfa, NfaState},
    prepare_grammar::extract_tokens::PoolLexicalVariable,
    rule_pool::{Node, NodeId, Prec, RulePool},
    rules::Symbol,
};

struct NfaBuilder {
    nfa: Nfa,
    is_sep: bool,
    precedence_stack: Vec<i32>,
}

pub type ExpandTokensResult<T> = Result<T, ExpandTokensError>;

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
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

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExpandTokensProcessingError {
    rule: String,
    error: ExpandRuleError,
}

impl std::fmt::Display for ExpandTokensProcessingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Error processing rule {}: {}", self.rule, self.error)
    }
}

fn get_implicit_precedence(pool: &RulePool, root: NodeId) -> i32 {
    let mut id = root;
    let mut boost = 0;
    loop {
        match pool.node(id) {
            Node::String(_) => return 2 + boost,
            Node::Metadata { params, rule } => {
                if pool.params(params).is_main_token {
                    boost += 1;
                }
                id = rule
            }
            _ => return boost,
        }
    }
}

fn get_completion_precedence(pool: &RulePool, id: NodeId) -> i32 {
    if let Node::Metadata { params, .. } = pool.node(id)
        && let Prec::Integer(p) = pool.params(params).prec
    {
        return p;
    }
    0
}

pub fn expand_tokens(
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
        if pool.subtree_matches_empty_str(variable.root) {
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
            precedence: get_completion_precedence(pool, variable.root),
        });
        let last_state_id = builder.nfa.last_state_id();
        builder
            .expand_rule(pool, variable.root, last_state_id)
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
                .expand_rule(pool, separator_root, last_state_id)
                .map_err(ExpandTokensError::ExpandRule)?;
        }

        variables.push(LexicalVariable {
            name: pool.resolve(variable.name).to_string(),
            kind: variable.kind,
            implicit_precedence: get_implicit_precedence(pool, variable.root),
            start_state: builder.nfa.last_state_id(),
        });
    }

    Ok(LexicalGrammar {
        nfa: builder.nfa,
        variables,
    })
}

fn build_separator(pool: &mut RulePool, separator_roots: &[NodeId]) -> NodeId {
    let blank = pool.push_node(Node::Blank);
    if separator_roots.is_empty() {
        return blank;
    }
    let mut elements = Vec::with_capacity(separator_roots.len() + 1);
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

pub type ExpandRuleResult<T> = Result<T, ExpandRuleError>;

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExpandRuleError {
    #[error("unexpected symbol {0:?}")]
    UnexpectedSymbol(Symbol),
    #[error("unexpected reserved-word context {0}")]
    UnexpectedReserved(String),
    #[error("{0}")]
    Parse(String),
    #[error(transparent)]
    ExpandRegex(ExpandRegexError),
}

pub type ExpandRegexResult<T> = Result<T, ExpandRegexError>;

#[derive(Debug, Error, Serialize, Deserialize, PartialEq, Eq)]
pub enum ExpandRegexError {
    #[error("{0}")]
    Utf8(String),
    #[error("Regex error: Assertions are not supported")]
    Assertion,
}

impl NfaBuilder {
    fn expand_rule(
        &mut self,
        pool: &RulePool,
        id: NodeId,
        mut next_state_id: u32,
    ) -> ExpandRuleResult<bool> {
        match pool.node(id) {
            Node::Pattern(s, f) => {
                // With unicode enabled, `\w`, `\s` and `\d` expand to character sets that are much
                // larger than intended, so we replace them with the actual
                // character sets they should represent. If the full unicode range
                // of `\w`, `\s` or `\d` are needed then `\p{L}`, `\p{Z}` and `\p{N}` should be
                // used.
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
                    if self.expand_rule(pool, element, next_state_id)? {
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
                    if self.expand_rule(pool, element, next_state_id)? {
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
                if self.expand_rule(pool, rule, next_state_id)? {
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
                let result = self.expand_rule(pool, rule, next_state_id);
                if has_precedence {
                    self.precedence_stack.pop();
                }
                result
            }
            Node::Blank => Ok(false),
            Node::Sym { kind, index } => Err(ExpandRuleError::UnexpectedSymbol(Symbol {
                kind,
                index: index as usize,
            }))?,
            Node::Reserved { ctx, .. } => Err(ExpandRuleError::UnexpectedReserved(
                pool.resolve(ctx).to_string(),
            ))?,
            // `NamedSymbol` is interned to `Sym` by intern_symbols
            Node::NamedSymbol(_) => unreachable!(),
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

                    // Unicode simple case folding (used when a regex contains the case-insensitive
                    // 'i' flag) maps exactly two non-ASCII code points onto ASCII letters:
                    //      - the long s `ſ` (U+017F) folds with `s`
                    //      - the Kelvin sign `K` (U+212A) folds with `k`
                    //
                    // Because of this folding, a case-insensitive pattern that mentions
                    // `s` or `k` silently pulls in these code points. That is virtually
                    // never intended, and it prevents such tokens from being extracted
                    // as keywords (because they are no longer a subset of an ASCII-only
                    // `word` token). In this case, drop the unicode code point.
                    for (folded, lower, upper) in [('ſ', 's', 'S'), ('\u{212a}', 'k', 'K')] {
                        if chars.contains(folded) && chars.contains(lower) && chars.contains(upper)
                        {
                            chars = chars.difference(CharacterSet::from_char(folded));
                        }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        grammars::VariableType,
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

    fn check(
        //                                   (token roots, separator roots)
        build: impl FnOnce(&mut RulePool) -> (Vec<NodeId>, Vec<NodeId>),
        examples: &[(&str, Option<(usize, &str)>)],
    ) {
        let mut pool = RulePool::default();
        let (roots, separators) = build(&mut pool);
        let vars = roots
            .into_iter()
            .enumerate()
            .map(|(i, root)| PoolLexicalVariable {
                name: pool.intern(&format!("tok{i}")),
                kind: VariableType::Anonymous,
                root,
            })
            .collect::<Vec<_>>();
        let grammar = expand_tokens(&mut pool, &vars, &separators).unwrap();
        for &(input, expected) in examples {
            assert_eq!(simulate_nfa(&grammar, input), expected, "input {input}");
        }
    }

    #[test]
    fn test_rule_expansion() {
        // regex with sequences and alternatives
        check(
            |p| {
                let (v, f) = (p.intern("(a|b|c)d(e|f|g)h?"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("ade1", Some((0, "ade"))),
                ("bdf1", Some((0, "bdf"))),
                ("bdfh1", Some((0, "bdfh"))),
                ("ad1", None),
            ],
        );
        // regex with repeats
        check(
            |p| {
                let (v, f) = (p.intern("a*"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[("aaa1", Some((0, "aaa"))), ("b", Some((0, "")))],
        );
        // regex with repeats in sequences
        check(
            |p| {
                let (v, f) = (p.intern("a((bc)+|(de)*)f"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("af1", Some((0, "af"))),
                ("adedef1", Some((0, "adedef"))),
                ("abcbcbcf1", Some((0, "abcbcbcf"))),
                ("a", None),
            ],
        );
        // regex with character ranges
        check(
            |p| {
                let (v, f) = (p.intern("[a-fA-F0-9]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[("A1ff0.", Some((0, "A1ff0")))],
        );
        // regex with perl character classes
        check(
            |p| {
                let (v, f) = (p.intern("\\w\\d\\s"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[("_0  ", Some((0, "_0 ")))],
        );
        // string
        check(
            |p| {
                let s = p.intern("abc");
                (vec![p.string(s)], vec![])
            },
            &[("abcd", Some((0, "abc"))), ("ab", None)],
        );
        // complex rule containing strings and regexes
        check(
            |p| {
                let (reg, empty) = (p.intern("[a-f]+"), p.intern(""));
                let (lb, pat, rb) = (p.intern("{"), p.pattern(reg, empty), p.intern("}"));
                let (left, right) = (p.string(lb), p.string(rb));
                let seq = p.seq(&[left, pat, right]);
                (vec![p.repeat(seq)], vec![])
            },
            &[
                ("{a}{", Some((0, "{a}"))),
                ("{a}{d", Some((0, "{a}"))),
                ("ab", None),
            ],
        );
        // longest match rule
        check(
            |p| {
                let empty = p.intern("");
                let (x, y, z) = (p.intern("a|bc"), p.intern("aa"), p.intern("bcd"));
                (
                    vec![
                        p.pattern(x, empty),
                        p.pattern(y, empty),
                        p.pattern(z, empty),
                    ],
                    vec![],
                )
            },
            &[
                ("a.", Some((0, "a"))),
                ("bc.", Some((0, "bc"))),
                ("aa.", Some((1, "aa"))),
                ("bcd?", Some((2, "bcd"))),
                ("b.", None),
                ("c.", None),
            ],
        );
        // regex with an alternative including the empty string
        check(
            |p| {
                let (v, f) = (p.intern("a(b|)+c"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("ac.", Some((0, "ac"))),
                ("abc.", Some((0, "abc"))),
                ("abbc.", Some((0, "abbc"))),
            ],
        );
        // separators
        check(
            |p| {
                let (v, f) = (p.intern("[a-f]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("  a", Some((0, "a"))),
                ("  \nb", Some((0, "b"))),
                ("  \\a", None),
                ("  \\\na", Some((0, "a"))),
            ],
        );
        // shorter tokens with higher precedence
        check(
            |p| {
                let f = p.intern("");
                let (v1, v2, v3) = (p.intern("abc"), p.intern("ab[cd]e"), p.intern("[a-e]+"));
                let (pat1, pat2, pat3) = (p.pattern(v1, f), p.pattern(v2, f), p.pattern(v3, f));
                (
                    vec![
                        p.prec(Prec::Integer(2), pat1),
                        p.prec(Prec::Integer(1), pat2),
                        pat3,
                    ],
                    vec![],
                )
            },
            &[
                ("abceef", Some((0, "abc"))),
                ("abdeef", Some((1, "abde"))),
                ("aeeeef", Some((2, "aeeee"))),
            ],
        );
        // immediate tokens with higher precedence
        check(
            |p| {
                let f = p.intern("");
                let (v1, v2) = (p.intern("[^a]+"), p.intern("[^ab]+"));
                let (pat1, pat2) = (p.pattern(v1, f), p.pattern(v2, f));
                let r1 = p.prec(Prec::Integer(1), pat1);
                let r2 = {
                    let imm = p.prec(Prec::Integer(2), pat2);
                    p.immediate_token(imm)
                };
                let sep = {
                    let s = p.intern("\\s");
                    p.pattern(s, f)
                };
                (vec![r1, r2], vec![sep])
            },
            &vec![("cccb", Some((1, "ccc")))],
        );
        check(
            |p| {
                let (a, b, c, d) = (p.intern("a"), p.intern("b"), p.intern("c"), p.intern("d"));
                let r1 = p.string(a);
                let r2 = {
                    let (x, y) = (p.string(b), p.string(c));
                    p.choice(&[x, y])
                };
                let r3 = p.string(d);
                (vec![p.seq(&[r1, r2, r3])], vec![])
            },
            &[
                ("abd", Some((0, "abd"))),
                ("acd", Some((0, "acd"))),
                ("abc", None),
                ("ad", None),
                ("d", None),
                ("a", None),
            ],
        );
        // nested choices within sequences
        check(
            |p| {
                let r1 = {
                    let (v, f) = (p.intern("[0-9]+"), p.intern(""));
                    p.pattern(v, f)
                };
                let r2 = {
                    let blank = p.blank();
                    let inner_ch = {
                        let ch1 = {
                            let (e1, e2) = (p.intern("e"), p.intern("E"));
                            let (e1, e2) = (p.string(e1), p.string(e2));
                            p.choice(&[e1, e2])
                        };
                        let ch2 = {
                            let blank = p.blank();
                            let (plus, minus) = (p.intern("+"), p.intern("-"));
                            let (plus, minus) = (p.string(plus), p.string(minus));
                            let inner_ch = p.choice(&[plus, minus]);
                            p.choice(&[blank, inner_ch])
                        };
                        let pat = {
                            let (v, f) = (p.intern("[0-9]+"), p.intern(""));
                            p.pattern(v, f)
                        };
                        let sq = p.seq(&[ch1, ch2, pat]);
                        p.choice(&[sq])
                    };
                    p.choice(&[blank, inner_ch])
                };
                let seq = p.seq(&[r1, r2]);
                (vec![seq], vec![])
            },
            &[
                ("12", Some((0, "12"))),
                ("12e", Some((0, "12"))),
                ("12g", Some((0, "12"))),
                ("12e3", Some((0, "12e3"))),
                ("12e+", Some((0, "12"))),
                ("12E+34 +", Some((0, "12E+34"))),
                ("12e34", Some((0, "12e34"))),
            ],
        );
        // nested groups
        check(
            |p| {
                let (v, f) = (p.intern(r"([^x\\]|\\(.|\n))+"), p.intern(""));
                let pat = p.pattern(v, f);
                let sq = p.seq(&[pat]);
                (vec![sq], vec![])
            },
            &[("abcx", Some((0, "abc"))), ("abc\\0x", Some((0, "abc\\0")))],
        );
        // allowing unrecognized escape sequences
        check(
            |p| {
                let f = p.intern("");
                // Escaped forward slash (used in JS because '/' is the regex delimiter)
                let v1 = p.intern(r"\/");
                // Escaped quotes
                let v2 = p.intern(r#"\"\'"#);
                // Quote preceded by a literal backslash
                let v3 = p.intern(r"[\\']+");
                (
                    vec![p.pattern(v1, f), p.pattern(v2, f), p.pattern(v3, f)],
                    vec![],
                )
            },
            &[
                ("/", Some((0, "/"))),
                ("\"\'", Some((1, "\"\'"))),
                (r"'\'a", Some((2, r"'\'"))),
            ],
        );
        // unicode property escapes
        check(
            |p| {
                let f = p.intern("");
                let v1 = p.intern(r"\p{L}+\P{L}+");
                let v2 = p.intern(r"\p{White_Space}+\P{White_Space}+[\p{White_Space}]*");
                (vec![p.pattern(v1, f), p.pattern(v2, f)], vec![])
            },
            &[
                ("  123   abc", Some((1, "  123   "))),
                ("ბΨƁ___ƀƔ", Some((0, "ბΨƁ___"))),
            ],
        );
        // unicode property escapes in bracketed sets
        check(
            |p| {
                let (v, f) = (p.intern(r"[\p{L}\p{Nd}]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[("abΨ12٣٣, ok", Some((0, "abΨ12٣٣")))],
        );
        // unicode character escapes
        check(
            |p| {
                let f = p.intern("");
                let v1 = p.intern(r"\u{00dc}");
                let v2 = p.intern(r"\U{000000dd}");
                let v3 = p.intern(r"\u00de");
                let v4 = p.intern(r"\U000000df");
                (
                    vec![
                        p.pattern(v1, f),
                        p.pattern(v2, f),
                        p.pattern(v3, f),
                        p.pattern(v4, f),
                    ],
                    vec![],
                )
            },
            &[
                ("\u{00dc}", Some((0, "\u{00dc}"))),
                ("\u{00dd}", Some((1, "\u{00dd}"))),
                ("\u{00de}", Some((2, "\u{00de}"))),
                ("\u{00df}", Some((3, "\u{00df}"))),
            ],
        );
        check(
            |p| {
                let f = p.intern("");
                let v1 = p.intern(r"u\{[0-9a-fA-F]+\}");
                // Already-escaped curly braces
                let v2 = p.intern(r"\{[ab]{3}\}");
                // Unicode codepoints
                let v3 = p.intern(r"\u{1000A}");
                // Unicode codepoints (lowercase)
                let v4 = p.intern(r"\u{1000b}");
                (
                    vec![
                        p.pattern(v1, f),
                        p.pattern(v2, f),
                        p.pattern(v3, f),
                        p.pattern(v4, f),
                    ],
                    vec![],
                )
            },
            &[
                ("u{1234} ok", Some((0, "u{1234}"))),
                ("{aba}}", Some((1, "{aba}"))),
                ("\u{1000A}", Some((2, "\u{1000A}"))),
                ("\u{1000b}", Some((3, "\u{1000b}"))),
            ],
        );
        // Case-insensitive patterns must not fold in the two non-ASCII code
        // points that Unicode simple case folding maps onto ASCII letters:
        // `ſ` (U+017F) onto `s`, and the Kelvin sign `K` (U+212A) onto `k`.
        check(
            |p| {
                let (v, f) = (p.intern("[sk]+"), p.intern("i"));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("sSkK.", Some((0, "sSkK"))),
                ("\u{017f}", None),              // long s, not matched by `s`
                ("\u{212a}", None),              // Kelvin sign, not matched by `k`
                ("sk\u{212a}", Some((0, "sk"))), // folded code point ends the token
            ],
        );
        // An intentionally-written `ſ`/`K` is preserved: the stripping above
        // only fires when the ASCII pair it folds with is also present.
        check(
            |p| {
                let (v, f) = (p.intern("[\u{017f}\u{212a}]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[("\u{017f}\u{212a}.", Some((0, "\u{017f}\u{212a}")))],
        );
        // Emojis
        check(
            |p| {
                let (v, f) = (p.intern(r"\p{Emoji}+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("🐎", Some((0, "🐎"))),
                ("🐴🐴", Some((0, "🐴🐴"))),
                ("#0", Some((0, "#0"))), // These chars are technically emojis!
                ("⻢", None),
                ("♞", None),
                ("horse", None),
            ],
        );
        // Intersection
        check(
            |p| {
                let (v, f) = (p.intern(r"[[0-7]&&[4-9]]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("456", Some((0, "456"))),
                ("64", Some((0, "64"))),
                ("452", Some((0, "45"))),
                ("91", None),
                ("8", None),
                ("3", None),
            ],
        );
        // Difference
        check(
            |p| {
                let (v, f) = (p.intern(r"[[0-9]--[4-7]]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("123", Some((0, "123"))),
                ("83", Some((0, "83"))),
                ("9", Some((0, "9"))),
                ("124", Some((0, "12"))),
                ("67", None),
                ("4", None),
            ],
        );
        // Symmetric difference
        check(
            |p| {
                let (v, f) = (p.intern(r"[[0-7]~~[4-9]]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("123", Some((0, "123"))),
                ("83", Some((0, "83"))),
                ("9", Some((0, "9"))),
                ("124", Some((0, "12"))),
                ("67", None),
                ("4", None),
            ],
        );
        // Nested set operations
        check(
            //               0 1 2 3 4 5 6 7 8 9
            // [0-5]:        y y y y y y
            // [2-4]:            y y y
            // [0-5]--[2-4]: y y       y
            // [3-9]:              y y y y y y y
            // [6-7]:                    y y
            // [3-9]--[5-7]:       y y y     y y
            // final regex:  y y   y y       y y
            |p| {
                let (v, f) = (p.intern(r"[[[0-5]--[2-4]]~~[[3-9]--[6-7]]]+"), p.intern(""));
                (vec![p.pattern(v, f)], vec![])
            },
            &[
                ("01", Some((0, "01"))),
                ("432", Some((0, "43"))),
                ("8", Some((0, "8"))),
                ("9", Some((0, "9"))),
                ("2", None),
                ("567", None),
            ],
        );
    }
}
