use std::path::Path;

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
        LowerErrorKind::OverrideRuleNotFound(vec!["nonexistent".into()])
    }
    error_missing_grammar_block {
        r#"rule program { "x" }"#,
        LowerErrorKind::MissingGrammarBlock
    }
    multiple_inherit_bindings_rejected {
        r#"let a = inherit("inherit_base/grammar.tsg")
        let b = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: a }
        rule extra { "x" }"#,
        LowerErrorKind::MultipleInherits
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
        grammar { language: "test", inherits: base, word: grammar_config(base).word }
        override rule program { "x" }"#,
        LowerErrorKind::ConfigFieldUnset
    }
    error_word_not_named_symbol {
        r#"let my_word = seq("x", "y")
        grammar { language: "test", word: my_word }
        rule program { "z" }"#,
        LowerErrorKind::ExpectedRuleName
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

#[test]
fn error_inherit_bad_extension() {
    let dir = tempfile::tempdir().unwrap();
    let bad_file = dir.path().join("grammar.txt");
    std::fs::write(&bad_file, "not a grammar").unwrap();
    let input = format!(
        r#"let base = inherit("{}") grammar {{ language: "derived", inherits: base }}"#,
        dsl_path(&bad_file)
    );
    let e = assert_err!(parse_native_dsl(&input, Path::new(".")).unwrap_err(), Lower);
    assert_eq!(e.kind, LowerErrorKind::ModuleUnsupportedExtension);
}

#[test]
fn error_inherit_child_error() {
    let (err, base_path) =
        inherit_err("grammar { language: \"base\" }\nrule program { bogus_ref }\n");
    let DslError::Module(m) = &err else {
        panic!("expected Module, got {err:?}")
    };
    let DslError::Type(e) = m.inner.as_ref() else {
        panic!("expected Type, got {:?}", m.inner)
    };
    assert_eq!(e.kind, TypeErrorKind::UnknownIdentifier("bogus_ref".into()));
    assert_eq!(m.path, base_path);
}

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
    let err = parse_native_dsl(&source, dir.path()).unwrap_err();
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
