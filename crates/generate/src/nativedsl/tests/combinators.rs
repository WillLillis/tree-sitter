use super::*;

#[test]
fn seq_and_choice() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { seq(choice("a", "b"), "c") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::choice(vec![Rule::String("a".into()), Rule::String("b".into())]),
            Rule::String("c".into()),
        ])
    );
}

#[test]
fn repeat1() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { repeat1("x") }
    "#);
    assert_eq!(g.variables[0].rule, Rule::repeat(Rule::String("x".into())));
}

#[test]
fn optional_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { optional("x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![Rule::String("x".into()), Rule::Blank])
    );
}

#[test]
fn blank_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { blank() }
    "#);
    assert_eq!(g.variables[0].rule, Rule::Blank);
}

#[test]
fn token_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { token(seq("a", "b")) }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::token(Rule::seq(vec![
            Rule::String("a".into()),
            Rule::String("b".into()),
        ]))
    );
}

#[test]
fn token_immediate_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { token_immediate("x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::immediate_token(Rule::String("x".into()))
    );
}

#[test]
fn field_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { field(name, "x") }
        rule name { "n" }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::field("name".to_string(), Rule::String("x".into()))
    );
}

#[test]
fn alias_with_string() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { alias(identifier, "id") }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::alias(
            Rule::NamedSymbol("identifier".into()),
            "id".to_string(),
            false
        )
    );
}

#[test]
fn alias_with_named_rule() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { alias(_impl, block) }
        rule _impl { "impl" }
        rule block { "{}" }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::alias(Rule::NamedSymbol("_impl".into()), "block".to_string(), true)
    );
}

#[test]
fn regexp_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { regexp("[a-z]+") }
    "#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == "[a-z]+"));
}

#[test]
fn regexp_with_flags() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { regexp("[a-z]+", "i") }
    "#);
    assert!(matches!(
        &g.variables[0].rule,
        Rule::Pattern(p, f) if p == "[a-z]+" && f == "i"
    ));
}

#[test]
fn concat_combinator() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { regexp(concat("[", "a-z", "]+")) }
    "#);
    assert!(matches!(&g.variables[0].rule, Rule::Pattern(p, _) if p == "[a-z]+"));
}

// ===== Prec variants =====

#[test]
fn prec_default() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec(1, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(1), Rule::String("x".into()))
    );
}

#[test]
fn prec_left() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_left(2, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_left(Precedence::Integer(2), Rule::String("x".into()))
    );
}

#[test]
fn prec_right() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_right(3, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_right(Precedence::Integer(3), Rule::String("x".into()))
    );
}

#[test]
fn prec_dynamic() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_dynamic(4, "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_dynamic(4, Rule::String("x".into()))
    );
}

#[test]
fn prec_with_string_name() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { prec_left("assign", "x") }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec_left(Precedence::Name("assign".into()), Rule::String("x".into()))
    );
}

#[test]
fn prec_inside_token_immediate() {
    let source = read_native_grammar("json_native");
    let g = parse_native_dsl(&source, &native_grammar_path("json_native")).unwrap();
    let string_content = g
        .variables
        .iter()
        .find(|v| v.name == "string_content")
        .unwrap();
    if let Rule::Metadata { params, .. } = &string_content.rule {
        assert_eq!(params.precedence, Precedence::Integer(1));
        assert!(params.is_token);
        assert!(params.is_main_token);
    } else {
        panic!("expected Metadata, got {:?}", string_content.rule);
    }
}

#[test]
fn reserved_combinator() {
    let g = dsl(r#"
        grammar {
            language: "test",
            reserved: { default: [identifier] },
        }
        rule program { reserved("default", identifier) }
        rule identifier { regexp("[a-z]+") }
    "#);
    assert!(matches!(
        &g.variables[0].rule,
        Rule::Reserved { context_name, .. } if context_name == "default"
    ));
}

#[test]
fn reserved_multiple_sets() {
    let g = dsl(r#"
        grammar {
            language: "test",
            reserved: {
                global: ["if", "else", "for"],
                properties: ["get", "set"],
            },
        }
        rule program { reserved("global", regexp("[a-z]+")) }
    "#);
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
        grammar { language: "derived", inherits: base, reserved: grammar_config(base).reserved }
    "#);
    assert_eq!(g.reserved_words.len(), 2);
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![
            Rule::String("if".into()),
            Rule::String("else".into()),
            Rule::String("for".into()),
        ]
    );
    assert_eq!(g.reserved_words[1].name, "properties");
    assert_eq!(
        g.reserved_words[1].reserved_words,
        vec![Rule::String("get".into()), Rule::String("set".into()),]
    );
}

#[test]
fn reserved_empty_inherited() {
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, reserved: grammar_config(base).reserved }
    "#);
    assert!(g.reserved_words.is_empty());
}

#[test]
fn for_inline_in_seq() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program {
            seq(
                "start",
                for (kw: str_t) in ["a", "b"] { kw },
                "end",
            )
        }
    "#);
    assert_eq!(
        g.variables[0].rule,
        Rule::seq(vec![
            Rule::String("start".into()),
            Rule::String("a".into()),
            Rule::String("b".into()),
            Rule::String("end".into()),
        ])
    );
}

#[test]
fn rule_reference_in_body() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { other }
        rule other { "x" }
    "#);
    assert_eq!(g.variables[0].rule, Rule::NamedSymbol("other".into()));
}

#[test]
fn trailing_comma_in_builtins() {
    macro_rules! g {
        ($expr:expr) => {
            concat!(r#"grammar { language: "test" } rule foo { "#, $expr, " }")
        };
    }
    // All of these have a trailing comma after the correct number of args.
    // They should parse successfully, not produce WrongArgumentCount.
    for src in [
        // 1-arg
        g!(r#"repeat("x",)"#),
        g!(r#"repeat1("x",)"#),
        g!(r#"optional("x",)"#),
        g!(r#"token("x",)"#),
        g!(r#"token_immediate("x",)"#),
        // 2-arg
        g!(r#"prec(1, "x",)"#),
        g!(r#"prec_left(1, "x",)"#),
        g!(r#"prec_right(1, "x",)"#),
        g!(r#"prec_dynamic(1, "x",)"#),
        g!(r#"field(name, "x",)"#),
        g!(r#"alias("x", foo,)"#),
        g!(r#"reserved("ctx", "x",)"#),
        // regexp (1 and 2 arg)
        g!(r#"regexp("pat",)"#),
        g!(r#"regexp("pat", "flags",)"#),
    ] {
        dsl(src);
    }
    // append needs a list context
    dsl(r#"grammar { language: "test", extras: append(["x"], ["y"],) } rule foo { "x" }"#);
}
