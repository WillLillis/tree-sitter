use rustc_hash::{FxHashMap, FxHashSet};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::{
    Diagnostic,
    flat_rule::{FlatRule, FlatRules, RuleId},
    grammars::{
        FlatVariable, InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType,
    },
    rules::{Precedence, Rule},
};

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
#[expect(
    non_camel_case_types,
    reason = "variant names match JSON grammar format"
)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "variant names match JSON grammar format"
)]
pub(crate) enum RuleJSON {
    ALIAS {
        content: Box<Self>,
        named: bool,
        value: String,
    },
    BLANK,
    STRING {
        value: String,
    },
    PATTERN {
        value: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        flags: Option<String>,
    },
    SYMBOL {
        name: String,
    },
    CHOICE {
        members: Vec<Self>,
    },
    FIELD {
        name: String,
        content: Box<Self>,
    },
    SEQ {
        members: Vec<Self>,
    },
    REPEAT {
        content: Box<Self>,
    },
    REPEAT1 {
        content: Box<Self>,
    },
    PREC_DYNAMIC {
        value: i32,
        content: Box<Self>,
    },
    PREC_LEFT {
        value: PrecedenceValueJSON,
        content: Box<Self>,
    },
    PREC_RIGHT {
        value: PrecedenceValueJSON,
        content: Box<Self>,
    },
    PREC {
        value: PrecedenceValueJSON,
        content: Box<Self>,
    },
    TOKEN {
        content: Box<Self>,
    },
    IMMEDIATE_TOKEN {
        content: Box<Self>,
    },
    RESERVED {
        context_name: String,
        content: Box<Self>,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum PrecedenceValueJSON {
    Integer(i32),
    Name(String),
}

#[derive(Deserialize)]
pub struct GrammarJSON {
    pub name: String,
    rules: Map<String, Value>,
    #[serde(default)]
    precedences: Vec<Vec<RuleJSON>>,
    #[serde(default)]
    conflicts: Vec<Vec<String>>,
    #[serde(default)]
    externals: Vec<RuleJSON>,
    #[serde(default)]
    extras: Vec<RuleJSON>,
    #[serde(default)]
    inline: Vec<String>,
    #[serde(default)]
    supertypes: Vec<String>,
    #[serde(default)]
    word: Option<String>,
    #[serde(default)]
    reserved: Map<String, Value>,
}

pub type ParseGrammarResult<T> = Result<T, ParseGrammarError>;

#[derive(Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum ParseGrammarError {
    #[error("{0}")]
    Serialization(String),
    #[error("Rules in the `extras` array must not contain empty strings")]
    InvalidExtra,
    #[error("Invalid rule in precedences array. Only strings and symbols are allowed")]
    Unexpected,
    #[error("Reserved word sets must be arrays")]
    InvalidReservedWordSet,
    #[error("Grammar Error: Unexpected rule `{0}` in `token()` call")]
    UnexpectedRule(String),
}

impl From<serde_json::Error> for ParseGrammarError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

/// Check if a rule is referenced by another rule.
///
/// This function is used to determine if a variable is used in a given rule,
/// and `is_other` indicates if the rule is an external, and if it is,
/// to not assume that a named symbol that is equal to itself means it's being referenced.
///
/// For example, if we have an external rule **and** a normal rule both called `foo`,
/// `foo` should not be thought of as directly used unless it's used within another rule.
fn rule_is_referenced(pool: &FlatRules, root: RuleId, target: &str, is_external: bool) -> bool {
    // `is_external` is only meaningful at a top-level `NamedSymbol` (an externals
    // entry names itself); it passes through Metadata/Reserved but resets under a
    // Seq/Choice/Repeat.
    let mut stack = vec![(root, is_external)];
    while let Some((id, ext)) = stack.pop() {
        match pool.get(id) {
            FlatRule::NamedSymbol(name) => {
                if name == target && !ext {
                    return true;
                }
            }
            FlatRule::Choice(range) | FlatRule::Seq(range) => {
                for &child in pool.children(*range) {
                    stack.push((child, false));
                }
            }
            FlatRule::Metadata { rule, .. } | FlatRule::Reserved { rule, .. } => {
                stack.push((*rule, ext));
            }
            FlatRule::Repeat(inner) => stack.push((*inner, false)),
            _ => {}
        }
    }
    false
}

impl InputGrammar {
    /// Strip unused rules from the grammar and clean up references to them
    /// in the surrounding config. (conflicts, supertypes, inline, extras,
    /// externals, precedences).
    ///
    /// A variable is "used" if it is the start rule, the word token, named in
    /// `extras`/`externals`, or transitively reachable via rule references from
    /// any of the above.
    #[must_use]
    pub fn normalize(mut self, diagnostics: &mut Vec<Diagnostic>) -> Self {
        // Compute the used set via forward DFS from the implicit roots
        // (start rule, word_token, refs in extras and externals).
        //
        // Extras count their top-level `NamedSymbol` as a use (so naming
        // a rule directly in `extras` keeps it), but externals do not (the
        // external entry is the rule itself, not a reference to one).
        let used: FxHashSet<String> = {
            let by_name: FxHashMap<&str, RuleId> = self
                .variables
                .iter()
                .map(|v| (v.name.as_str(), v.rule))
                .collect();
            let mut visited: FxHashSet<&str> = FxHashSet::default();
            let mut stack: Vec<&str> = Vec::new();
            if let Some(first) = self.variables.first() {
                stack.push(first.name.as_str());
            }
            if let Some(word) = self.word_token.as_deref() {
                stack.push(word);
            }
            for &rule in &self.extra_symbols {
                collect_referenced_names(&self.pool, rule, false, &mut stack);
            }
            for &rule in &self.external_tokens {
                collect_referenced_names(&self.pool, rule, true, &mut stack);
            }
            // Reserved-word entries are uses of the named rule (the entry
            // names a token to reserve in some context). Top-level
            // `NamedSymbol` counts, same as for extras.
            for ctx in &self.reserved_words {
                for &rule in &ctx.reserved_words {
                    collect_referenced_names(&self.pool, rule, false, &mut stack);
                }
            }
            while let Some(name) = stack.pop() {
                if !visited.insert(name) {
                    continue;
                }
                if let Some(&rule) = by_name.get(name) {
                    collect_referenced_names(&self.pool, rule, false, &mut stack);
                }
            }
            visited.into_iter().map(String::from).collect()
        };

        for v in &self.variables {
            if !used.contains(v.name.as_str()) {
                continue;
            }
            if !self
                .extra_symbols
                .iter()
                .any(|&r| rule_is_referenced(&self.pool, r, &v.name, false))
            {
                continue;
            }
            let inner = if let FlatRule::Metadata { rule, .. } = self.pool.get(v.rule) {
                *rule
            } else {
                v.rule
            };
            let matches_empty = match self.pool.get(inner) {
                FlatRule::String(s) => s.is_empty(),
                FlatRule::Pattern(value, _) => {
                    Regex::new(value).is_ok_and(|reg| reg.is_match(""))
                }
                _ => false,
            };
            if matches_empty {
                diagnostics.push(Diagnostic::EmptyStringMatch(v.name.clone()));
            }
        }

        // Drop unused variables and clean up any references to them in the
        // surrounding grammar config.
        let dropped: Vec<String> = self
            .variables
            .iter()
            .filter(|v| !used.contains(v.name.as_str()))
            .map(|v| v.name.clone())
            .collect();
        self.variables.retain(|v| used.contains(v.name.as_str()));
        for name in &dropped {
            self.expected_conflicts.retain(|r| !r.contains(name));
            self.supertype_symbols.retain(|r| r != name);
            self.variables_to_inline.retain(|r| r != name);
            self.extra_symbols
                .retain(|&r| !rule_is_referenced(&self.pool, r, name, true));
            self.external_tokens
                .retain(|&r| !rule_is_referenced(&self.pool, r, name, true));
            self.precedence_orderings.retain(|r| {
                !r.iter()
                    .any(|e| matches!(e, PrecedenceEntry::Symbol(s) if s == name))
            });
            // Prune entries but keep the context: an intentionally-empty
            // reserved-word context is a meaningful marker that rule bodies
            // may reference by name.
            for ctx in &mut self.reserved_words {
                ctx.reserved_words
                    .retain(|&r| !rule_is_referenced(&self.pool, r, name, false));
            }
        }

        self
    }
}

/// Append every `NamedSymbol` name reachable in `rule` to `out`. If
/// `skip_top_level` is true, a `NamedSymbol` at the root of `rule` is
/// ignored (used for externals entries, which name themselves).
fn collect_referenced_names<'a>(
    pool: &'a FlatRules,
    root: RuleId,
    skip_top_level: bool,
    out: &mut Vec<&'a str>,
) {
    let mut stack = vec![(root, skip_top_level)];
    while let Some((id, skip)) = stack.pop() {
        match pool.get(id) {
            FlatRule::NamedSymbol(name) => {
                if !skip {
                    out.push(name.as_str());
                }
            }
            FlatRule::Choice(range) | FlatRule::Seq(range) => {
                for &child in pool.children(*range) {
                    stack.push((child, false));
                }
            }
            FlatRule::Metadata { rule, .. } | FlatRule::Reserved { rule, .. } => {
                stack.push((*rule, skip));
            }
            FlatRule::Repeat(inner) => stack.push((*inner, false)),
            _ => {}
        }
    }
}

pub fn parse_grammar(
    input: &str,
    diagnostics: &mut Vec<Diagnostic>,
) -> ParseGrammarResult<InputGrammar> {
    let grammar_json = serde_json::from_str::<GrammarJSON>(input)?;

    let extra_symbols =
        grammar_json
            .extras
            .into_iter()
            .try_fold(Vec::<Rule>::new(), |mut acc, item| {
                let rule = parse_rule(item, false, diagnostics)?;
                if let Rule::String(ref value) = rule
                    && value.is_empty()
                {
                    Err(ParseGrammarError::InvalidExtra)?;
                }
                acc.push(rule);
                ParseGrammarResult::Ok(acc)
            })?;

    let external_tokens = grammar_json
        .externals
        .into_iter()
        .map(|e| parse_rule(e, false, diagnostics))
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    let mut precedence_orderings = Vec::with_capacity(grammar_json.precedences.len());
    for list in grammar_json.precedences {
        let mut ordering = Vec::with_capacity(list.len());
        for entry in list {
            ordering.push(match entry {
                RuleJSON::STRING { value } => PrecedenceEntry::Name(value),
                RuleJSON::SYMBOL { name } => PrecedenceEntry::Symbol(name),
                _ => Err(ParseGrammarError::Unexpected)?,
            });
        }
        precedence_orderings.push(ordering);
    }

    let variables = grammar_json
        .rules
        .into_iter()
        .map(|(name, r)| {
            Ok(Variable {
                name,
                kind: VariableType::Named,
                rule: parse_rule(serde_json::from_value(r)?, false, diagnostics)?,
            })
        })
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    let reserved_words = grammar_json
        .reserved
        .into_iter()
        .map(|(name, rule_values)| {
            let Value::Array(rule_values) = rule_values else {
                Err(ParseGrammarError::InvalidReservedWordSet)?
            };

            let mut reserved_words = Vec::with_capacity(rule_values.len());
            for value in rule_values {
                reserved_words.push(parse_rule(
                    serde_json::from_value(value)?,
                    false,
                    diagnostics,
                )?);
            }
            Ok(ReservedWordContext {
                name,
                reserved_words,
            })
        })
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    // Intern all rules into a shared pool so the grammar is flat (RuleId-based).
    let mut pool = FlatRules::default();
    let variables = variables
        .into_iter()
        .map(|v| FlatVariable {
            name: v.name,
            kind: v.kind,
            rule: pool.intern_import(&v.rule),
        })
        .collect();
    let extra_symbols = extra_symbols.iter().map(|r| pool.intern_import(r)).collect();
    let external_tokens = external_tokens.iter().map(|r| pool.intern_import(r)).collect();
    let reserved_words = reserved_words
        .into_iter()
        .map(|ctx| ReservedWordContext {
            name: ctx.name,
            reserved_words: ctx.reserved_words.iter().map(|r| pool.intern_import(r)).collect(),
        })
        .collect();

    let grammar = InputGrammar {
        pool,
        name: grammar_json.name,
        word_token: grammar_json.word,
        expected_conflicts: grammar_json.conflicts,
        supertype_symbols: grammar_json.supertypes,
        variables_to_inline: grammar_json.inline,
        precedence_orderings,
        variables,
        extra_symbols,
        external_tokens,
        reserved_words,
    }
    .normalize(diagnostics);
    Ok(grammar)
}

fn parse_rule(
    json: RuleJSON,
    is_token: bool,
    diagnostics: &mut Vec<Diagnostic>,
) -> ParseGrammarResult<Rule> {
    match json {
        RuleJSON::ALIAS {
            content,
            value,
            named,
        } => parse_rule(*content, is_token, diagnostics).map(|r| Rule::alias(r, value, named)),
        RuleJSON::BLANK => Ok(Rule::Blank),
        RuleJSON::STRING { value } => Ok(Rule::String(value)),
        RuleJSON::PATTERN { value, flags } => {
            let processed_flags = flags.map_or(String::new(), |f| {
                f.matches(|c| {
                    if c == 'i' {
                        true
                    } else {
                        // silently ignore unicode flags
                        if c != 'u' && c != 'v' {
                            diagnostics.push(Diagnostic::UnsupportedRegexFlag {
                                flag: c,
                                pattern: value.clone(),
                            });
                        }
                        false
                    }
                })
                .collect()
            });
            Ok(Rule::Pattern(value, processed_flags))
        }
        RuleJSON::SYMBOL { name } => {
            if is_token {
                Err(ParseGrammarError::UnexpectedRule(name))?
            } else {
                Ok(Rule::NamedSymbol(name))
            }
        }
        RuleJSON::CHOICE { members } => members
            .into_iter()
            .map(|m| parse_rule(m, is_token, diagnostics))
            .collect::<ParseGrammarResult<Vec<_>>>()
            .map(Rule::choice),
        RuleJSON::FIELD { content, name } => {
            parse_rule(*content, is_token, diagnostics).map(|r| Rule::field(name, r))
        }
        RuleJSON::SEQ { members } => members
            .into_iter()
            .map(|m| parse_rule(m, is_token, diagnostics))
            .collect::<ParseGrammarResult<Vec<_>>>()
            .map(Rule::seq),
        RuleJSON::REPEAT1 { content } => {
            parse_rule(*content, is_token, diagnostics).map(Rule::repeat)
        }
        RuleJSON::REPEAT { content } => parse_rule(*content, is_token, diagnostics)
            .map(|m| Rule::choice(vec![Rule::repeat(m), Rule::Blank])),
        RuleJSON::PREC { value, content } => {
            parse_rule(*content, is_token, diagnostics).map(|r| Rule::prec(value.into(), r))
        }
        RuleJSON::PREC_LEFT { value, content } => {
            parse_rule(*content, is_token, diagnostics).map(|r| Rule::prec_left(value.into(), r))
        }
        RuleJSON::PREC_RIGHT { value, content } => {
            parse_rule(*content, is_token, diagnostics).map(|r| Rule::prec_right(value.into(), r))
        }
        RuleJSON::PREC_DYNAMIC { value, content } => {
            parse_rule(*content, is_token, diagnostics).map(|r| Rule::prec_dynamic(value, r))
        }
        RuleJSON::RESERVED {
            content,
            context_name,
        } => parse_rule(*content, is_token, diagnostics).map(|r| Rule::Reserved {
            rule: Box::new(r),
            context_name,
        }),
        RuleJSON::TOKEN { content } => parse_rule(*content, true, diagnostics).map(Rule::token),
        RuleJSON::IMMEDIATE_TOKEN { content } => {
            parse_rule(*content, true, diagnostics).map(Rule::immediate_token)
        }
    }
}

impl From<PrecedenceValueJSON> for Precedence {
    fn from(val: PrecedenceValueJSON) -> Self {
        match val {
            PrecedenceValueJSON::Integer(i) => Self::Integer(i),
            PrecedenceValueJSON::Name(i) => Self::Name(i),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_grammar() {
        let grammar = parse_grammar(
            r#"{
            "name": "my_lang",
            "rules": {
                "file": {
                    "type": "REPEAT1",
                    "content": {
                        "type": "SYMBOL",
                        "name": "statement"
                    }
                },
                "statement": {
                    "type": "STRING",
                    "value": "foo"
                }
            }
        }"#,
            &mut Vec::new(),
        )
        .unwrap();

        assert_eq!(grammar.name, "my_lang");
        let variables = grammar
            .variables
            .iter()
            .map(|v| Variable {
                name: v.name.clone(),
                kind: v.kind,
                rule: grammar.pool.materialize(v.rule),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            variables,
            vec![
                Variable {
                    name: "file".to_string(),
                    kind: VariableType::Named,
                    rule: Rule::repeat(Rule::NamedSymbol("statement".to_string()))
                },
                Variable {
                    name: "statement".to_string(),
                    kind: VariableType::Named,
                    rule: Rule::String("foo".to_string())
                },
            ]
        );
    }
}
