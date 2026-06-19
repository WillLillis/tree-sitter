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
        macro make_alias(target: rule_t) rule_t { alias("x", target) }
        rule program { make_alias(some_rule) }
        rule some_rule { "y" }"#,
        Rule::alias(Rule::String("x".into()), "some_rule".to_string(), true)
    }
    alias_with_let_string_target {
        // A `let`-bound string as the alias target resolves to the let's value
        // (unnamed alias "renamed"), not a force-resolved named rule reference
        // "FOO". Matches grammar.js evaluating a const used as the alias name.
        r#"grammar { language: "test" }
        let FOO = "renamed"
        rule program { alias(identifier, FOO) }
        rule identifier { regexp("[a-z]+") }"#,
        Rule::alias(Rule::NamedSymbol("identifier".into()), "renamed".to_string(), false)
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
    prec_dynamic_negative {
        r#"grammar { language: "test" } rule program { prec_dynamic(-1, "x") }"#,
        Rule::prec_dynamic(-1, Rule::String("x".into()))
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
    raw_ident_as_rule_reference {
        // r#let escapes the `let` keyword so it can name a rule.
        r#"grammar { language: "test" }
        rule program { r#let }
        rule r#let { "in" }"#,
        Rule::NamedSymbol("let".into())
    }
    string_hex_escape {
        // \xHH (ASCII range) decodes to the literal byte. Matches perl's __DATA__ ctrl-D usage.
        "grammar { language: \"test\" } rule program { \"a\\x04b\" }",
        Rule::String("a\x04b".into())
    }
    string_unicode_escape_4digit {
        // \uHHHH (4 hex) decodes to UTF-8.
        r#"grammar { language: "test" } rule program { "\u00A0" }"#,
        Rule::String("\u{00A0}".into())
    }
    string_unicode_escape_braced {
        // \u{H..H} (1-6 hex in braces) decodes to UTF-8, allows non-BMP.
        r#"grammar { language: "test" } rule program { "\u{1F389}" }"#,
        Rule::String("\u{1F389}".into())
    }
    int_arith_add_literals {
        r#"grammar { language: "test" } rule program { prec(1 + 2, "x") }"#,
        Rule::prec(Precedence::Integer(3), Rule::String("x".into()))
    }
    int_arith_sub_literals {
        r#"grammar { language: "test" } rule program { prec(10 - 3, "x") }"#,
        Rule::prec(Precedence::Integer(7), Rule::String("x".into()))
    }
    int_arith_chained {
        // Left-associative: 1 + 2 - 5 + 10 = 8.
        r#"grammar { language: "test" } rule program { prec(1 + 2 - 5 + 10, "x") }"#,
        Rule::prec(Precedence::Integer(8), Rule::String("x".into()))
    }
    int_arith_named_plus_literal {
        // Named-int + literal: PREC.foo = 5, +1 = 6.
        r#"let PREC = { foo: 5 } grammar { language: "test" } rule program { prec(PREC.foo + 1, "x") }"#,
        Rule::prec(Precedence::Integer(6), Rule::String("x".into()))
    }
    int_arith_two_named {
        // Both sides named: PREC.a + PREC.b = 8.
        r#"let PREC = { a: 5, b: 3 } grammar { language: "test" } rule program { prec(PREC.a + PREC.b, "x") }"#,
        Rule::prec(Precedence::Integer(8), Rule::String("x".into()))
    }
    int_arith_in_let_binding {
        // Arithmetic in a let RHS is folded at bind time.
        r#"let PREC = { base: 10 } let X = PREC.base + 1 grammar { language: "test" } rule program { prec(X, "x") }"#,
        Rule::prec(Precedence::Integer(11), Rule::String("x".into()))
    }
    int_arith_with_unary_neg {
        // -PREC.foo combined with new + arithmetic: -5 + 10 = 5.
        r#"let PREC = { foo: 5 } grammar { language: "test" } rule program { prec(-PREC.foo + 10, "x") }"#,
        Rule::prec(Precedence::Integer(5), Rule::String("x".into()))
    }
    raw_ident_keyword_collisions {
        // Shadowing builtin combinator/keyword names: r#field, r#import, r#alias.
        // Each is a normal rule whose grammar.json key is the bare keyword.
        r#"grammar { language: "test" }
        rule program { seq(r#field, r#import, r#alias) }
        rule r#field { "f" }
        rule r#import { "i" }
        rule r#alias { "a" }"#,
        Rule::seq(vec![
            Rule::NamedSymbol("field".into()),
            Rule::NamedSymbol("import".into()),
            Rule::NamedSymbol("alias".into()),
        ])
    }
}

#[test]
fn raw_ident_emits_bare_name_in_grammar_json() {
    let g = dsl(r#"grammar { language: "test" }
        rule program { r#let }
        rule r#let { "in" }"#);
    // grammar.json rule names must match the bare identifier, not `r#let`.
    let names: Vec<&str> = g.variables.iter().map(|v| v.name.as_str()).collect();
    assert_eq!(names, vec!["program", "let"]);
}

#[test]
fn raw_ident_as_let_binding_name() {
    // `r#name` works in let-binding position when the bare name collides
    // with a DSL keyword.
    let g = dsl(r#"grammar { language: "test" }
        let r#for: str_t = "x"
        rule program { r#for }"#);
    assert_eq!(g.variables[0].rule, Rule::String("x".into()));
}

#[test]
fn raw_ident_as_object_key() {
    // `r#name` works as an object literal key when the bare name collides
    // with a DSL keyword.
    let g = dsl(r#"grammar { language: "test" }
        let cfg = { r#for: 1, r#in: 2 }
        rule program { prec(cfg.r#for, "x") }"#);
    assert_eq!(
        g.variables[0].rule,
        Rule::prec(Precedence::Integer(1), Rule::String("x".into())),
    );
}

#[test]
fn regexp_combinator() {
    let g = dsl(r#"grammar { language: "test" } rule program { regexp("[a-z]+") }"#);
    assert_eq!(
        g.variables[0].rule,
        Rule::Pattern("[a-z]+".into(), String::new())
    );
}

#[test]
fn regexp_with_flags() {
    let g = dsl(r#"grammar { language: "test" } rule program { regexp("[a-z]+", "i") }"#);
    assert_eq!(
        g.variables[0].rule,
        Rule::Pattern("[a-z]+".into(), "i".into())
    );
}

#[test]
fn concat_combinator() {
    let g =
        dsl(r#"grammar { language: "test" } rule program { regexp(concat("[", "a-z", "]+")) }"#);
    assert_eq!(
        g.variables[0].rule,
        Rule::Pattern("[a-z]+".into(), String::new())
    );
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
fn object_key_accepts_contextual_keyword() {
    // An object-literal key is a name, so it accepts contextual keywords like
    // `field`, the same as the field-access member position already does.
    let g = dsl(r#"
        let x = { field: "y" }
        grammar { language: "test" }
        rule program { x.field }
    "#);
    assert_eq!(g.variables[0].rule, Rule::String("y".into()));
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
    // A child with no `reserved` of its own inherits the base's reserved sets
    // (in base order, so the first/default set is preserved). No explicit
    // `grammar_config(base, reserved)` re-import needed - it merges by default.
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_with_reserved.tsg")
        grammar { language: "derived", inherits: base }
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
fn reserved_child_merges_with_base() {
    // The child adds a new set and overrides an existing one. Base order is
    // preserved (global stays first = default), the overridden set keeps its
    // base position with new words, and the new set is appended. Matches dsl.js.
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar_with_reserved.tsg")
        grammar {
            language: "derived",
            inherits: base,
            reserved: { global: ["if"], extra: ["new"] },
        }
    "#);
    assert_eq!(g.reserved_words.len(), 3);
    // Base "global" stays first (default), with the child's overriding words.
    assert_eq!(g.reserved_words[0].name, "global");
    assert_eq!(
        g.reserved_words[0].reserved_words,
        vec![Rule::String("if".into())]
    );
    // Base "properties" kept (not redefined by the child).
    assert_eq!(g.reserved_words[1].name, "properties");
    // Child's new set appended.
    assert_eq!(g.reserved_words[2].name, "extra");
}

#[test]
fn reserved_empty_inherited() {
    // A base with no reserved contributes none; the child inherits an empty set.
    let g = dsl(r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
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
