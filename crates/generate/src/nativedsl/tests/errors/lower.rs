use super::super::*;

error_tests! { Lower {
    error_override_without_inherit {
        r#"grammar { language: "test" } override rule foo { "bar" }"#,
        LowerErrorKind::OverrideWithoutInherit
    }
    error_override_rule_not_found {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule nonexistent { "oops" }"#,
        LowerErrorKind::OverrideRuleNotFound(vec![("nonexistent".into(), Span::new(125, 136))])
    }
    error_missing_grammar_block {
        r#"rule program { "x" }"#,
        LowerErrorKind::MissingGrammarBlock
    }
    error_inherit_without_config {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "test" }
        rule program { "x" }"#,
        LowerErrorKind::InheritWithoutConfig
    }
    error_inherits_without_inherit {
        r#"grammar { language: "test", inherits: "not_inherit" }
        rule program { "x" }"#,
        LowerErrorKind::InheritsWithoutInherit
    }
    error_config_field_unset {
        r#"let base = inherit("inherit_base_no_word/grammar.tsg")
        grammar { language: "test", inherits: base, word: grammar_config(base, word) }
        override rule program { "x" }"#,
        LowerErrorKind::ConfigFieldUnset
    }
    error_word_not_named_symbol {
        r#"let my_word = seq("x", "y")
        grammar { language: "test", word: my_word }
        rule program { "z" }"#,
        LowerErrorKind::ExpectedRuleName
    }
    error_int_literal_too_large_positive {
        r#"grammar { language: "test" }
        rule program { prec(3000000000, "x") }"#,
        LowerErrorKind::IntegerOverflow(3_000_000_000)
    }
    error_int_literal_too_large_negative {
        r#"grammar { language: "test" }
        rule program { prec(-3000000000, "x") }"#,
        LowerErrorKind::IntegerOverflow(-3_000_000_000)
    }
    error_double_negation_of_i32_min_overflows {
        r#"grammar { language: "test" }
        let x: int_t = -2147483648
        rule program { prec(-x, "y") }"#,
        LowerErrorKind::IntegerOverflow(2_147_483_648)
    }
}}

#[test]
fn error_inherit_bad_path() {
    let e = assert_err!(
        dsl_err(
            r#"let base = inherit("nonexistent/grammar.tsg")
        grammar { language: "derived", inherits: base }"#
        ),
        Lower
    );
    assert!(matches!(e.kind, LowerErrorKind::ModuleResolveFailed { .. }));
}

inherit_error_tests! { Type {
    error_inherit_child_error {
        "grammar { language: \"base\" }\nrule program { bogus_ref }\n",
        TypeErrorKind::UnknownIdentifier("bogus_ref".into())
    }
}}

#[test]
fn error_inherit_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let a_path = dir.path().join("a.tsg");
    let b_path = dir.path().join("b.tsg");
    std::fs::write(
        &a_path,
        format!(
            "let base = inherit(\"{}\")\ngrammar {{ language: \"a\", inherits: base }}",
            dsl_path(&b_path)
        ),
    )
    .unwrap();
    std::fs::write(
        &b_path,
        format!(
            "let base = inherit(\"{}\")\ngrammar {{ language: \"b\", inherits: base }}",
            dsl_path(&a_path)
        ),
    )
    .unwrap();
    let source = std::fs::read_to_string(&a_path).unwrap();
    let err = parse_native_dsl(&source, &a_path).unwrap_err();
    let mut chain = Vec::new();
    let mut current: &DslError = &err;
    loop {
        match current {
            DslError::Module(m) => {
                chain.push(m.path.file_name().unwrap().to_str().unwrap().to_string());
                current = &m.inner;
            }
            DslError::Lower(e) => {
                assert!(matches!(e.kind, LowerErrorKind::ModuleCycle));
                break;
            }
            other => panic!("unexpected error variant: {other:?}"),
        }
    }
    assert_eq!(chain, ["b.tsg", "a.tsg"]);
}

#[test]
fn error_self_inherit_cycle() {
    // A grammar that inherits itself should be caught as a cycle,
    // not spin until MAX_MODULE_DEPTH.
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("grammar.tsg");
    std::fs::write(
        &path,
        r#"let base = inherit("grammar.tsg")
        grammar { language: "test", inherits: base }
        rule program { "x" }"#,
    )
    .unwrap();
    let source = std::fs::read_to_string(&path).unwrap();
    let err = parse_native_dsl(&source, &path).unwrap_err();
    match &err {
        DslError::Module(m) => {
            assert!(
                matches!(*m.inner, DslError::Lower(ref e) if matches!(e.kind, LowerErrorKind::ModuleCycle))
            );
        }
        other => panic!("expected Module(ModuleCycle), got {other:?}"),
    }
}

#[test]
fn error_inherit_rule_not_found() {
    let e = assert_err!(
        dsl_err(
            r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        rule extra { base::nonexistent_rule }"#
        ),
        Type
    );
    assert_eq!(
        e.kind,
        TypeErrorKind::ImportMemberNotFound("nonexistent_rule".into())
    );
}

#[test]
fn error_alias_non_name_rule_via_var() {
    // A variable holding a non-name rule (e.g. seq) passed as alias target
    // should produce a proper error, not panic.
    let e = assert_err!(
        dsl_err(
            r#"grammar { language: "test" }
            macro make_alias(target: rule_t) rule_t { alias("x", target) }
            rule a { "a" }
            rule b { "b" }
            rule program { make_alias(seq(a, b)) }"#
        ),
        Lower
    );
    assert_eq!(e.kind, LowerErrorKind::ExpectedRuleName);
}

#[test]
fn error_inline_non_name_rule_via_var() {
    // A variable holding a non-name rule in config name-lists (inline,
    // supertypes, conflicts) should produce a proper error, not panic.
    let e = assert_err!(
        dsl_err(
            r#"let x = seq(a, b)
            grammar { language: "test", inline: [x] }
            rule a { "a" }
            rule b { "b" }
            rule program { a }"#
        ),
        Lower
    );
    assert_eq!(e.kind, LowerErrorKind::ExpectedRuleName);
}
