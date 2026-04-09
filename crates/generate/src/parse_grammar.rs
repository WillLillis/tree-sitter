use rustc_hash::FxHashSet;

use log::warn;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::{
    grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType},
    rules::{Precedence, Rule},
};

#[derive(Deserialize)]
#[serde(tag = "type")]
#[expect(
    non_camel_case_types,
    reason = "variant names match JSON grammar format"
)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "variant names match JSON grammar format"
)]
enum RuleJSON {
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

#[derive(Deserialize)]
#[serde(untagged)]
enum PrecedenceValueJSON {
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

#[derive(Debug, Error, Serialize)]
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
/// Strip unused rules and clean up references to them from the grammar config.
///
/// `parse_grammar` calls this internally. It must also be called on any
/// `InputGrammar` constructed outside of `parse_grammar` (e.g. from the
/// native DSL) before passing to `prepare_grammar`.
pub fn normalize_grammar(grammar: &mut InputGrammar) {
    let rules: Vec<(String, Rule)> = grammar
        .variables
        .iter()
        .map(|v| (v.name.clone(), v.rule.clone()))
        .collect();
    let mut in_progress = FxHashSet::default();
    let mut keep = Vec::with_capacity(rules.len());

    for (name, rule) in &rules {
        if grammar.word_token.as_ref().is_none_or(|w| w != name)
            && !variable_is_used(
                &rules,
                &grammar.extra_symbols,
                &grammar.external_tokens,
                name,
                &mut in_progress,
            )
        {
            grammar.expected_conflicts.retain(|r| !r.contains(name));
            grammar.supertype_symbols.retain(|r| r != name);
            grammar.variables_to_inline.retain(|r| r != name);
            grammar
                .extra_symbols
                .retain(|r| !rule_is_referenced(r, name, true));
            grammar
                .external_tokens
                .retain(|r| !rule_is_referenced(r, name, true));
            grammar.precedence_orderings.retain(|r| {
                !r.iter()
                    .any(|e| matches!(e, PrecedenceEntry::Symbol(s) if s == name))
            });
            continue;
        }

        if grammar
            .extra_symbols
            .iter()
            .any(|r| rule_is_referenced(r, name, false))
        {
            let inner_rule = match rule {
                Rule::Metadata { rule, .. } => rule,
                _ => rule,
            };
            let matches_empty = match inner_rule {
                Rule::String(s) => s.is_empty(),
                Rule::Pattern(value, _) => Regex::new(value).is_ok_and(|reg| reg.is_match("")),
                _ => false,
            };
            if matches_empty {
                warn!(
                    "Named extra rule `{name}` matches the empty string. \
                     Inline this to avoid infinite loops while parsing."
                );
            }
        }
        keep.push(name.clone());
    }

    grammar.variables.retain(|v| keep.contains(&v.name));
}

fn rule_is_referenced(rule: &Rule, target: &str, is_external: bool) -> bool {
    match rule {
        Rule::NamedSymbol(name) => name == target && !is_external,
        Rule::Choice(rules) | Rule::Seq(rules) => {
            rules.iter().any(|r| rule_is_referenced(r, target, false))
        }
        Rule::Metadata { rule, .. } | Rule::Reserved { rule, .. } => {
            rule_is_referenced(rule, target, is_external)
        }
        Rule::Repeat(inner) => rule_is_referenced(inner, target, false),
        Rule::Blank | Rule::String(_) | Rule::Pattern(_, _) | Rule::Symbol(_) => false,
    }
}

fn variable_is_used(
    grammar_rules: &[(String, Rule)],
    extras: &[Rule],
    externals: &[Rule],
    target_name: &str,
    in_progress: &mut FxHashSet<String>,
) -> bool {
    let root = &grammar_rules.first().unwrap().0;
    if target_name == root {
        return true;
    }

    if extras
        .iter()
        .any(|rule| rule_is_referenced(rule, target_name, false))
    {
        return true;
    }

    if externals
        .iter()
        .any(|rule| rule_is_referenced(rule, target_name, true))
    {
        return true;
    }

    in_progress.insert(target_name.to_string());
    let result = grammar_rules
        .iter()
        .filter(|(key, _)| *key != target_name)
        .any(|(name, rule)| {
            if !rule_is_referenced(rule, target_name, false) || in_progress.contains(name) {
                return false;
            }
            variable_is_used(grammar_rules, extras, externals, name, in_progress)
        });
    in_progress.remove(target_name);

    result
}

pub(crate) fn parse_grammar(input: &str) -> ParseGrammarResult<InputGrammar> {
    let grammar_json = serde_json::from_str::<GrammarJSON>(input)?;

    let extra_symbols =
        grammar_json
            .extras
            .into_iter()
            .try_fold(Vec::<Rule>::new(), |mut acc, item| {
                let rule = parse_rule(item, false)?;
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
        .map(|e| parse_rule(e, false))
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
        .map(|(n, r)| {
            Ok(Variable {
                name: n,
                kind: VariableType::Named,
                rule: parse_rule(serde_json::from_value(r)?, false)?,
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
                reserved_words.push(parse_rule(serde_json::from_value(value)?, false)?);
            }
            Ok(ReservedWordContext {
                name,
                reserved_words,
            })
        })
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    let mut grammar = InputGrammar {
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
    };
    normalize_grammar(&mut grammar);
    Ok(grammar)
}

fn parse_rule(json: RuleJSON, is_token: bool) -> ParseGrammarResult<Rule> {
    match json {
        RuleJSON::ALIAS {
            content,
            value,
            named,
        } => parse_rule(*content, is_token).map(|r| Rule::alias(r, value, named)),
        RuleJSON::BLANK => Ok(Rule::Blank),
        RuleJSON::STRING { value } => Ok(Rule::String(value)),
        RuleJSON::PATTERN { value, flags } => Ok(Rule::Pattern(
            value,
            flags.map_or(String::new(), |f| {
                f.matches(|c| {
                    if c == 'i' {
                        true
                    } else {
                        // silently ignore unicode flags
                        if c != 'u' && c != 'v' {
                            warn!("unsupported flag {c}");
                        }
                        false
                    }
                })
                .collect()
            }),
        )),
        RuleJSON::SYMBOL { name } => {
            if is_token {
                Err(ParseGrammarError::UnexpectedRule(name))?
            } else {
                Ok(Rule::NamedSymbol(name))
            }
        }
        RuleJSON::CHOICE { members } => members
            .into_iter()
            .map(|m| parse_rule(m, is_token))
            .collect::<ParseGrammarResult<Vec<_>>>()
            .map(Rule::choice),
        RuleJSON::FIELD { content, name } => {
            parse_rule(*content, is_token).map(|r| Rule::field(name, r))
        }
        RuleJSON::SEQ { members } => members
            .into_iter()
            .map(|m| parse_rule(m, is_token))
            .collect::<ParseGrammarResult<Vec<_>>>()
            .map(Rule::seq),
        RuleJSON::REPEAT1 { content } => parse_rule(*content, is_token).map(Rule::repeat),
        RuleJSON::REPEAT { content } => {
            parse_rule(*content, is_token).map(|m| Rule::choice(vec![Rule::repeat(m), Rule::Blank]))
        }
        RuleJSON::PREC { value, content } => {
            parse_rule(*content, is_token).map(|r| Rule::prec(value.into(), r))
        }
        RuleJSON::PREC_LEFT { value, content } => {
            parse_rule(*content, is_token).map(|r| Rule::prec_left(value.into(), r))
        }
        RuleJSON::PREC_RIGHT { value, content } => {
            parse_rule(*content, is_token).map(|r| Rule::prec_right(value.into(), r))
        }
        RuleJSON::PREC_DYNAMIC { value, content } => {
            parse_rule(*content, is_token).map(|r| Rule::prec_dynamic(value, r))
        }
        RuleJSON::RESERVED {
            content,
            context_name,
        } => parse_rule(*content, is_token).map(|r| Rule::Reserved {
            rule: Box::new(r),
            context_name,
        }),
        RuleJSON::TOKEN { content } => parse_rule(*content, true).map(Rule::token),
        RuleJSON::IMMEDIATE_TOKEN { content } => {
            parse_rule(*content, is_token).map(Rule::immediate_token)
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
        )
        .unwrap();

        assert_eq!(grammar.name, "my_lang");
        assert_eq!(
            grammar.variables,
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
