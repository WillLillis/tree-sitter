use super::super::*;

error_tests! { Lower {
    // Field access on a computed object (here a macro result) can't be validated
    // at typecheck - obj_t<X> carries the value type, not the key set - so a
    // missing field is reported at lower with the available keys.
    error_field_not_found_on_computed_object {
        r#"grammar { language: "test" }
        macro mk() obj_t<rule_t> { { a: program } }
        rule program { seq(mk().b) }"#,
        LowerErrorKind::FieldNotFound { field: "b".into(), available: vec!["a".into()] }
    }
    error_missing_language_field {
        r#"grammar { extras: [] } rule program { "x" }"#,
        LowerErrorKind::MissingLanguageField
    }
    // A grammar that emits grammar.json must have at least one rule (matches
    // grammar.js). Config-only *bases* are allowed (checked only on the root).
    error_grammar_has_no_rules {
        r#"grammar { language: "test" }"#,
        LowerErrorKind::GrammarHasNoRules
    }
    // An external resolves rule-like but lives in external_tokens, not
    // variables, so it can't be variables[0] (the start symbol).
    error_external_as_start_rule {
        r#"grammar { language: "test", externals: [tok], start: tok }
        rule program { "x" }"#,
        LowerErrorKind::ExternalCannotBeStart("tok".into())
    }
    error_multiple_inherits {
        r#"let a = inherit("inherit_base/grammar.tsg")
        let b = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: a }
        rule extra { "x" }"#,
        LowerErrorKind::MultipleInherits
    }
    error_multiple_inherits_let_and_config {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: inherit("inherit_base/grammar.tsg") }
        rule extra { "x" }"#,
        LowerErrorKind::MultipleInherits
    }
    error_override_without_target {
        r#"grammar { language: "test" } override rule foo { "bar" }"#,
        LowerErrorKind::OverrideRuleNotFound(vec![Spanned::new("foo".into(), Span::new(43, 46))])
    }
    error_override_rule_not_found {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule nonexistent { "oops" }"#,
        LowerErrorKind::OverrideRuleNotFound(vec![Spanned::new(
            "nonexistent".into(),
            Span::new(125, 136),
        )])
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
    error_int_arith_add_overflow {
        r#"grammar { language: "test" }
        let x: int_t = 2000000000
        rule program { prec(x + x, "y") }"#,
        LowerErrorKind::IntegerOverflow(4_000_000_000)
    }
    error_int_arith_sub_underflow {
        r#"grammar { language: "test" }
        let x: int_t = -2000000000
        rule program { prec(x - 2000000000, "y") }"#,
        LowerErrorKind::IntegerOverflow(-4_000_000_000)
    }
}}

#[test]
fn error_multiple_override_rules_not_found() {
    // Multiple unmatched overrides should produce a list in deterministic
    // (sorted) order so users see a stable error and tests don't flake.
    let e = assert_err!(
        dsl_err(
            r#"let base = inherit("inherit_base/grammar.tsg")
            grammar { language: "derived", inherits: base }
            override rule zzz { "z" }
            override rule aaa { "a" }
            override rule mmm { "m" }"#
        ),
        Lower
    );
    let LowerErrorKind::OverrideRuleNotFound(entries) = e.kind else {
        panic!("expected OverrideRuleNotFound, got {:?}", e.kind);
    };
    let names: Vec<&str> = entries.iter().map(|e| e.value.as_str()).collect();
    assert_eq!(names, vec!["aaa", "mmm", "zzz"]);
}

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

#[test]
fn error_module_read_failure_renders_without_panic() {
    // Path canonicalizes but read_to_string fails (a directory).
    // The inner ModuleError carries an empty source_text plus the import-path
    // span from the parent; NativeDslError rendering must not slice out of range.
    let dir = tempfile::tempdir().unwrap();
    let subdir = dir.path().join("not_a_file");
    std::fs::create_dir(&subdir).unwrap();
    let parent_path = dir.path().join("parent.tsg");
    let parent_src = format!(
        "let h = import(\"{}\")\ngrammar {{ language: \"test\" }}\nrule program {{ \"x\" }}",
        dsl_path(&subdir)
    );
    std::fs::write(&parent_path, &parent_src).unwrap();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();
    let wrapped = NativeDslError {
        error: err,
        src: parent_src,
        path: parent_path,
    };
    let _rendered = format!("{wrapped}");
}

inherit_error_tests! { Resolve {
    error_inherit_child_error {
        "grammar { language: \"base\" }\nrule program { bogus_ref }\n",
        ResolveErrorKind::UnknownIdentifier("bogus_ref".into())
    }
}}

inherit_error_tests! { Lower {
    error_inherited_missing_language {
        "grammar { extras: [] }\nrule program { \"x\" }\n",
        LowerErrorKind::MissingLanguageField
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
        Resolve
    );
    assert_eq!(
        e.kind,
        ResolveErrorKind::ImportMemberNotFound("nonexistent_rule".into())
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
