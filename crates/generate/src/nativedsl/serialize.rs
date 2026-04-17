//! JSON serialization for [`InputGrammar`] - produces `grammar.json` format.

use crate::grammars::{InputGrammar, PrecedenceEntry};
use crate::rules::{Precedence, Rule};

/// Convert an [`InputGrammar`] to `grammar.json` format.
pub fn grammar_to_json(grammar: &InputGrammar) -> serde_json::Value {
    use serde_json::{Map, Value, json};

    let mut obj = Map::new();
    obj.insert("name".into(), Value::String(grammar.name.clone()));

    let mut rules = Map::new();
    for var in &grammar.variables {
        rules.insert(var.name.clone(), rule_to_json(&var.rule));
    }
    obj.insert("rules".into(), Value::Object(rules));

    let str_array = |items: &[String]| -> Value {
        Value::Array(items.iter().map(|s| Value::String(s.clone())).collect())
    };

    obj.insert(
        "extras".into(),
        Value::Array(grammar.extra_symbols.iter().map(rule_to_json).collect()),
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
                            .map(|e| match e {
                                PrecedenceEntry::Name(s) => json!({"type": "STRING", "value": s}),
                                PrecedenceEntry::Symbol(s) => json!({"type": "SYMBOL", "name": s}),
                            })
                            .collect(),
                    )
                })
                .collect(),
        ),
    );
    obj.insert(
        "externals".into(),
        Value::Array(grammar.external_tokens.iter().map(rule_to_json).collect()),
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
                Value::Array(ctx.reserved_words.iter().map(rule_to_json).collect()),
            );
        }
        obj.insert("reserved".into(), Value::Object(reserved));
    }
    Value::Object(obj)
}

fn rule_to_json(rule: &Rule) -> serde_json::Value {
    use crate::rules::Alias;
    use serde_json::json;
    match rule {
        Rule::Blank => json!({"type": "BLANK"}),
        Rule::String(s) => json!({"type": "STRING", "value": s}),
        Rule::Pattern(p, f) if f.is_empty() => json!({"type": "PATTERN", "value": p}),
        Rule::Pattern(p, f) => json!({"type": "PATTERN", "value": p, "flags": f}),
        Rule::NamedSymbol(n) => json!({"type": "SYMBOL", "name": n}),
        Rule::Symbol(s) => json!({"type": "SYMBOL", "name": format!("__symbol_{}", s.index)}),
        Rule::Choice(ms) => {
            json!({"type": "CHOICE", "members": ms.iter().map(rule_to_json).collect::<Vec<_>>()})
        }
        Rule::Seq(ms) => {
            json!({"type": "SEQ", "members": ms.iter().map(rule_to_json).collect::<Vec<_>>()})
        }
        Rule::Repeat(inner) => json!({"type": "REPEAT1", "content": rule_to_json(inner)}),
        Rule::Metadata { params, rule } => {
            let mut c = rule_to_json(rule);
            if params.dynamic_precedence != 0 {
                c = json!({"type": "PREC_DYNAMIC", "value": params.dynamic_precedence, "content": c});
            }
            if let Some(pv) = match &params.precedence {
                Precedence::None => None,
                Precedence::Integer(n) => Some(serde_json::Value::Number((*n).into())),
                Precedence::Name(s) => Some(serde_json::Value::String(s.clone())),
            } {
                c = match params.associativity {
                    Some(crate::rules::Associativity::Left) => {
                        json!({"type": "PREC_LEFT", "value": pv, "content": c})
                    }
                    Some(crate::rules::Associativity::Right) => {
                        json!({"type": "PREC_RIGHT", "value": pv, "content": c})
                    }
                    None => json!({"type": "PREC", "value": pv, "content": c}),
                };
            }
            if let Some(Alias { value, is_named }) = &params.alias {
                c = json!({"type": "ALIAS", "content": c, "named": is_named, "value": value});
            }
            if let Some(field_name) = &params.field_name {
                c = json!({"type": "FIELD", "name": field_name, "content": c});
            }
            if params.is_token && params.is_main_token {
                c = json!({"type": "IMMEDIATE_TOKEN", "content": c});
            } else if params.is_token {
                c = json!({"type": "TOKEN", "content": c});
            }
            c
        }
        Rule::Reserved { rule, context_name } => {
            json!({"type": "RESERVED", "context_name": context_name, "content": rule_to_json(rule)})
        }
    }
}
