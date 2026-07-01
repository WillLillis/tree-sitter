//! JSON serialization for [`InputGrammar`] - produces `grammar.json` format.

use crate::flat_rule::{FlatRules, RuleId};
use crate::grammars::{InputGrammar, PrecedenceEntry};
use crate::parse_grammar::{PrecedenceValueJSON, RuleJSON};
use crate::rules::{Alias, Associativity, Precedence, Rule};

use serde_json::{Map, Value};

/// Convert an [`InputGrammar`] to `grammar.json` format.
pub fn grammar_to_json(grammar: &InputGrammar) -> serde_json::Value {
    let mut obj = Map::new();
    obj.insert("name".into(), Value::String(grammar.name.clone()));

    let mut rules = Map::new();
    for var in &grammar.variables {
        rules.insert(var.name.clone(), rule_to_json(&grammar.pool, var.rule));
    }
    obj.insert("rules".into(), Value::Object(rules));

    let str_array = |items: &[String]| -> Value {
        Value::Array(items.iter().map(|s| Value::String(s.clone())).collect())
    };

    obj.insert(
        "extras".into(),
        Value::Array(
            grammar
                .extra_symbols
                .iter()
                .map(|&id| rule_to_json(&grammar.pool, id))
                .collect(),
        ),
    );
    obj.insert(
        "conflicts".into(),
        Value::Array(
            grammar
                .expected_conflicts
                .iter()
                .map(|g| str_array(g))
                .collect(),
        ),
    );
    obj.insert(
        "precedences".into(),
        Value::Array(
            grammar
                .precedence_orderings
                .iter()
                .map(|g| {
                    Value::Array(
                        g.iter()
                            .map(|e| {
                                let node = match e {
                                    PrecedenceEntry::Name(s) => {
                                        RuleJSON::STRING { value: s.clone() }
                                    }
                                    PrecedenceEntry::Symbol(s) => {
                                        RuleJSON::SYMBOL { name: s.clone() }
                                    }
                                };
                                node_to_value(node)
                            })
                            .collect(),
                    )
                })
                .collect(),
        ),
    );
    obj.insert(
        "externals".into(),
        Value::Array(
            grammar
                .external_tokens
                .iter()
                .map(|&id| rule_to_json(&grammar.pool, id))
                .collect(),
        ),
    );
    obj.insert("inline".into(), str_array(&grammar.variables_to_inline));
    obj.insert("supertypes".into(), str_array(&grammar.supertype_symbols));
    if let Some(word) = &grammar.word_token {
        obj.insert("word".into(), Value::String(word.clone()));
    }
    if !grammar.reserved_words.is_empty() {
        let mut reserved = Map::new();
        for ctx in &grammar.reserved_words {
            reserved.insert(
                ctx.name.clone(),
                Value::Array(
                    ctx.reserved_words
                        .iter()
                        .map(|&id| rule_to_json(&grammar.pool, id))
                        .collect(),
                ),
            );
        }
        obj.insert("reserved".into(), Value::Object(reserved));
    }
    Value::Object(obj)
}

/// Serialize a typed `grammar.json` node to a JSON value.
fn node_to_value(node: RuleJSON) -> Value {
    serde_json::to_value(node).expect("RuleJSON serialization cannot fail")
}

fn rule_to_json(pool: &FlatRules, id: RuleId) -> Value {
    node_to_value(build_rule(&pool.materialize(id)))
}

/// Lower the internal [`Rule`] representation into the typed `grammar.json`
/// schema ([`RuleJSON`]).
fn build_rule(rule: &Rule) -> RuleJSON {
    match rule {
        Rule::Blank => RuleJSON::BLANK,
        Rule::String(s) => RuleJSON::STRING { value: s.clone() },
        Rule::Pattern(p, f) => RuleJSON::PATTERN {
            value: p.clone(),
            flags: (!f.is_empty()).then(|| f.clone()),
        },
        Rule::NamedSymbol(n) => RuleJSON::SYMBOL { name: n.clone() },
        Rule::Symbol(s) => RuleJSON::SYMBOL {
            name: format!("__symbol_{}", s.index),
        },
        Rule::Choice(ms) => RuleJSON::CHOICE {
            members: ms.iter().map(build_rule).collect(),
        },
        Rule::Seq(ms) => RuleJSON::SEQ {
            members: ms.iter().map(build_rule).collect(),
        },
        Rule::Repeat(inner) => RuleJSON::REPEAT1 {
            content: Box::new(build_rule(inner)),
        },
        Rule::Metadata { params, rule } => {
            let mut c = build_rule(rule);
            if params.dynamic_precedence != 0 {
                c = RuleJSON::PREC_DYNAMIC {
                    value: params.dynamic_precedence,
                    content: Box::new(c),
                };
            }
            let pv = match &params.precedence {
                Precedence::None => None,
                Precedence::Integer(n) => Some(PrecedenceValueJSON::Integer(*n)),
                Precedence::Name(s) => Some(PrecedenceValueJSON::Name(s.clone())),
            };
            if let Some(pv) = pv {
                c = match params.associativity {
                    Some(Associativity::Left) => RuleJSON::PREC_LEFT {
                        value: pv,
                        content: Box::new(c),
                    },
                    Some(Associativity::Right) => RuleJSON::PREC_RIGHT {
                        value: pv,
                        content: Box::new(c),
                    },
                    None => RuleJSON::PREC {
                        value: pv,
                        content: Box::new(c),
                    },
                };
            }
            if let Some(Alias { value, is_named }) = &params.alias {
                c = RuleJSON::ALIAS {
                    content: Box::new(c),
                    named: *is_named,
                    value: value.clone(),
                };
            }
            if let Some(field_name) = &params.field_name {
                c = RuleJSON::FIELD {
                    name: field_name.clone(),
                    content: Box::new(c),
                };
            }
            if params.is_token && params.is_main_token {
                c = RuleJSON::IMMEDIATE_TOKEN {
                    content: Box::new(c),
                };
            } else if params.is_token {
                c = RuleJSON::TOKEN {
                    content: Box::new(c),
                };
            }
            c
        }
        Rule::Reserved { rule, context_name } => RuleJSON::RESERVED {
            context_name: context_name.clone(),
            content: Box::new(build_rule(rule)),
        },
    }
}
