//! JSON serialization for [`InputGrammar`] - produces `grammar.json` format.

use crate::grammars::{InputGrammar, PrecedenceEntry};
use crate::parse_grammar::{PrecedenceValueJSON, RuleJSON};
use crate::rules::{Alias, Associativity, Precedence, Rule, RuleId, RulePool};

use serde_json::{Map, Value};

/// Convert an [`InputGrammar`] to `grammar.json` format.
pub fn grammar_to_json(grammar: &InputGrammar) -> serde_json::Value {
    let mut obj = Map::new();
    obj.insert(
        "name".into(),
        Value::String(grammar.pool.resolve(grammar.name).into()),
    );

    let mut rules = Map::new();
    for var in &grammar.variables {
        rules.insert(
            grammar.pool.resolve(var.name).into(),
            rule_to_json(&grammar.pool, var.root),
        );
    }
    obj.insert("rules".into(), Value::Object(rules));

    let str_array = |items: &[crate::strpool::StrId]| -> Value {
        Value::Array(
            items
                .iter()
                .map(|&s| Value::String(grammar.pool.resolve(s).into()))
                .collect(),
        )
    };

    obj.insert(
        "extras".into(),
        Value::Array(
            grammar
                .extra_roots
                .iter()
                .map(|&r| rule_to_json(&grammar.pool, r))
                .collect(),
        ),
    );
    obj.insert(
        "conflicts".into(),
        Value::Array(
            grammar
                .conflict_names
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
                                    PrecedenceEntry::Name(s) => RuleJSON::STRING {
                                        value: grammar.pool.resolve(*s).into(),
                                    },
                                    PrecedenceEntry::Symbol(s) => RuleJSON::SYMBOL {
                                        name: grammar.pool.resolve(*s).into(),
                                    },
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
                .external_roots
                .iter()
                .map(|&r| rule_to_json(&grammar.pool, r))
                .collect(),
        ),
    );
    obj.insert("inline".into(), str_array(&grammar.inline_names));
    obj.insert("supertypes".into(), str_array(&grammar.supertype_names));
    if let Some(word) = grammar.word_name {
        obj.insert(
            "word".into(),
            Value::String(grammar.pool.resolve(word).into()),
        );
    }
    if !grammar.reserved_sets.is_empty() {
        let mut reserved = Map::new();
        for ctx in &grammar.reserved_sets {
            reserved.insert(
                grammar.pool.resolve(ctx.name).into(),
                Value::Array(
                    ctx.roots
                        .iter()
                        .map(|&r| rule_to_json(&grammar.pool, r))
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

fn rule_to_json(pool: &RulePool, root: RuleId) -> Value {
    node_to_value(build_rule(pool, root))
}

/// Lower the internal [`Rule`] representation into the typed `grammar.json`
/// schema ([`RuleJSON`]).
fn build_rule(pool: &RulePool, root: RuleId) -> RuleJSON {
    match pool.node(root) {
        Rule::Blank => RuleJSON::BLANK,
        Rule::String(s) => RuleJSON::STRING {
            value: pool.resolve(s).into(),
        },
        Rule::Pattern(p, f) => RuleJSON::PATTERN {
            value: pool.resolve(p).into(),
            flags: (!pool.resolve(f).is_empty()).then(|| pool.resolve(f).into()),
        },
        Rule::NamedSymbol(n) => RuleJSON::SYMBOL {
            name: pool.resolve(n).into(),
        },
        Rule::Sym { index, .. } => RuleJSON::SYMBOL {
            name: format!("__symbol_{index}"),
        },
        Rule::Choice(ms) => RuleJSON::CHOICE {
            members: pool
                .child_slice(ms)
                .iter()
                .map(|&r| build_rule(pool, r))
                .collect(),
        },
        Rule::Seq(ms) => RuleJSON::SEQ {
            members: pool
                .child_slice(ms)
                .iter()
                .map(|&r| build_rule(pool, r))
                .collect(),
        },
        Rule::Repeat(inner) => RuleJSON::REPEAT1 {
            content: Box::new(build_rule(pool, inner)),
        },
        Rule::Metadata { params, rule } => {
            let params = pool.params(params);
            let mut c = build_rule(pool, rule);
            if params.dynamic_precedence != 0 {
                c = RuleJSON::PREC_DYNAMIC {
                    value: params.dynamic_precedence,
                    content: Box::new(c),
                };
            }
            let pv = match params.precedence {
                Precedence::None => None,
                Precedence::Integer(n) => Some(PrecedenceValueJSON::Integer(n)),
                Precedence::Name(s) => Some(PrecedenceValueJSON::Name(pool.resolve(s).into())),
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
            if let Some(Alias { value, is_named }) = params.alias {
                c = RuleJSON::ALIAS {
                    content: Box::new(c),
                    named: is_named,
                    value: pool.resolve(value).into(),
                };
            }
            if let Some(field_name) = params.field {
                c = RuleJSON::FIELD {
                    name: pool.resolve(field_name).into(),
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
        Rule::Reserved { rule, ctx } => RuleJSON::RESERVED {
            context_name: pool.resolve(ctx).into(),
            content: Box::new(build_rule(pool, rule)),
        },
    }
}
