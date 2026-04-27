use super::*;

rule_tests! {
    seq_and_choice {
        r#"grammar { language: "test" } rule program { seq(choice("a", "b"), "c") }"#,
        Rule::seq(vec![
            Rule::choice(vec![Rule::String("a".into()), Rule::String("b".into())]),
            Rule::String("c".into()),
        ])
    }
    repeat1 {
        r#"grammar { language: "test" } rule program { repeat1("x") }"#,
        Rule::repeat(Rule::String("x".into()))
    }
    optional_combinator {
        r#"grammar { language: "test" } rule program { optional("x") }"#,
        Rule::choice(vec![Rule::String("x".into()), Rule::Blank])
    }
    blank_combinator {
        r#"grammar { language: "test" } rule program { blank() }"#,
        Rule::Blank
    }
    token_combinator {
        r#"grammar { language: "test" } rule program { token(seq("a", "b")) }"#,
        Rule::token(Rule::seq(vec![Rule::String("a".into()), Rule::String("b".into())]))
    }
    token_immediate_combinator {
        r#"grammar { language: "test" } rule program { token_immediate("x") }"#,
        Rule::immediate_token(Rule::String("x".into()))
    }
    field_combinator {
        r#"grammar { language: "test" }
        rule program { field(name, "x") }
        rule name { "n" }"#,
        Rule::field("name".to_string(), Rule::String("x".into()))
    }
    alias_with_string {
        r#"grammar { language: "test" }
        rule program { alias(identifier, "id") }
        rule identifier { regexp("[a-z]+") }"#,
        Rule::alias(Rule::NamedSymbol("identifier".into()), "id".to_string(), false)
    }
    alias_with_named_rule {
        r#"grammar { language: "test" }
        rule program { alias(_impl, block) }
        rule _impl { "impl" }
        rule block { "{}" }"#,
        Rule::alias(Rule::NamedSymbol("_impl".into()), "block".to_string(), true)
    }
    alias_with_variable_target {
        r#"grammar { language: "test" }
        macro make_alias(target: rule_t) rule_t = alias("x", target)
        rule program { make_alias(some_rule) }
        rule some_rule { "y" }"#,
        Rule::alias(Rule::String("x".into()), "some_rule".to_string(), true)
    }
    prec_default {
        r#"grammar { language: "test" } rule program { prec(1, "x") }"#,
        Rule::prec(Precedence::Integer(1), Rule::String("x".into()))
    }
    prec_left {
        r#"grammar { language: "test" } rule program { prec_left(2, "x") }"#,
        Rule::prec_left(Precedence::Integer(2), Rule::String("x".into()))
    }
    prec_right {
        r#"grammar { language: "test" } rule program { prec_right(3, "x") }"#,
        Rule::prec_right(Precedence::Integer(3), Rule::String("x".into()))
    }
    prec_dynamic {
        r#"grammar { language: "test" } rule program { prec_dynamic(4, "x") }"#,
        Rule::prec_dynamic(4, Rule::String("x".into()))
    }
    prec_with_string_name {
        r#"grammar { language: "test" } rule program { prec_left("assign", "x") }"#,
        Rule::prec_left(Precedence::Name("assign".into()), Rule::String("x".into()))
    }
    rule_reference_in_body {
        r#"grammar { language: "test" }
        rule program { other }
        rule other { "x" }"#,
        Rule::NamedSymbol("other".into())
    }
    for_inline_in_seq {
        r#"grammar { language: "test" }
        rule program {
            seq("start", for (kw: str_t) in ["a", "b"] { kw }, "end")
        }"#,
        Rule::seq(vec![
            Rule::String("start".into()),
            Rule::String("a".into()),
            Rule::String("b".into()),
            Rule::String("end".into()),
        ])
    }
}

#[test]
fn regexp_combinator() {
    let g = dsl(r#"grammar { language: "test" } rule program { regexp("[a-z]+") }"#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == "[a-z]+"));
}

#[test]
fn regexp_with_flags() {
    let g = dsl(r#"grammar { language: "test" } rule program { regexp("[a-z]+", "i") }"#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, f) if p == "[a-z]+" && f == "i"));
}

#[test]
fn concat_combinator() {
    let g =
        dsl(r#"grammar { language: "test" } rule program { regexp(concat("[", "a-z", "]+")) }"#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == "[a-z]+"));
}

#[test]
fn prec_inside_token_immediate() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { token_immediate(prec(1, regexp("[a-z]+"))) }
    "#);
    if let Rule::Metadata { params, .. } = &g.variables[0].rule {
        assert_eq!(params.precedence, Precedence::Integer(1));
        assert!(params.is_token && params.is_main_token);
    } else {
        panic!("expected Metadata, got {:?}", g.variables[0].rule);
    }
}

#[test]
fn reserved_combinator() {
    let g = dsl(r#"grammar {
        language: "test", reserved: { default: [identifier] },
    }
    rule program { reserved("default", identifier) }
    rule identifier { regexp("[a-z]+") }"#);
    assert!(
        matches!(&g.variables[0].rule, Rule::Reserved { context_name, .. } if context_name == "default")
    );
}

#[test]
fn reserved_multiple_sets() {
    let g = dsl(r#"grammar {
        language: "test",
        reserved: { global: ["if", "else", "for"], properties: ["get", "set"] },
    }
    rule program { reserved("global", regexp("[a-z]+")) }"#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(g.reserved_words[0].reserved_words.len(), 3);
    assert_eq!(g.reserved_words[1].name, "properties");
    assert_eq!(g.reserved_words[1].reserved_words.len(), 2);
}

#[test]
fn reserved_inherited() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_with_reserved.tsg")
        grammar { language: "derived", inherits: base, reserved: grammar_config(base, reserved) }
    "#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("for".into())
        ]
    );
    assert_eq!(g.reserved_words[1].name, "properties");
    assert_eq!(
        g.reserved_words[1].reserved_words,
        vec![Rule::String("get".into()), Rule::String("set".into())]
    );
}

#[test]
fn reserved_empty_inherited() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, reserved: grammar_config(base, reserved) }
    "#);
    assert!(g.reserved_words.is_empty());
}

#[test]
fn trailing_comma_in_builtins() {
    macro_rules! g {
        ($expr:expr) => {
            concat!(r#"grammar { language: "test" } rule foo { "#, $expr, " }")
        };
    }
    for src in [
        g!(r#"repeat("x",)"#),
        g!(r#"repeat1("x",)"#),
        g!(r#"optional("x",)"#),
        g!(r#"token("x",)"#),
        g!(r#"token_immediate("x",)"#),
        g!(r#"prec(1, "x",)"#),
        g!(r#"prec_left(1, "x",)"#),
        g!(r#"prec_right(1, "x",)"#),
        g!(r#"prec_dynamic(1, "x",)"#),
        g!(r#"field(name, "x",)"#),
        g!(r#"alias("x", foo,)"#),
        g!(r#"reserved("ctx", "x",)"#),
        g!(r#"regexp("pat",)"#),
        g!(r#"regexp("pat", "flags",)"#),
    ] {
        dsl(src);
    }
    dsl(r#"grammar { language: "test", extras: append(["x"], ["y"],) } rule foo { "x" }"#);
}
