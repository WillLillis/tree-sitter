use std::path::Path;

use super::super::*;

macro_rules! lower_error_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let e = assert_err!(dsl_err($input), Lower);
            assert_eq!(e.kind, $expected);
        })*
    };
}

lower_error_tests! {
    error_override_without_inherit {
        r#"grammar { language: "test" } override rule foo { "bar" }"#,
        LowerErrorKind::OverrideWithoutInherit
    }
    error_override_rule_not_found {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule nonexistent { "oops" }"#,
        LowerErrorKind::OverrideRuleNotFound("nonexistent".into())
    }
    error_missing_grammar_block {
        r#"rule program { "x" }"#,
        LowerErrorKind::MissingGrammarBlock
    }
    error_missing_language_field {
        r#"grammar { extras: [] } rule program { "x" }"#,
        LowerErrorKind::MissingLanguageField
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
fn error_inherit_bad_extension() {
    let dir = tempfile::tempdir().unwrap();
    let bad_file = dir.path().join("grammar.txt");
    std::fs::write(&bad_file, "not a grammar").unwrap();
    let input = format!(
        r#"let base = inherit("{}") grammar {{ language: "derived", inherits: base }}"#,
        bad_file.display()
    );
    let e = assert_err!(parse_native_dsl(&input, Path::new(".")).unwrap_err(), Lower);
    assert_eq!(e.kind, LowerErrorKind::ModuleUnsupportedExtension);
}

#[test]
fn error_inherit_child_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        "grammar { language: \"base\" }\nrule program { bogus_ref }\n",
    )
    .unwrap();
    let parent_path = dir.path().join("parent.tsg");
    let parent_src = "let base = inherit(\"base.tsg\")\ngrammar { language: \"derived\", inherits: base }\nrule extra { \"hello\" }\n".to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();
    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    let DslError::Resolve(resolve_err) = inherited.inner.as_ref() else {
        panic!("expected Resolve error, got {:?}", inherited.inner)
    };
    assert_eq!(
        resolve_err.kind,
        ResolveErrorKind::UnknownIdentifier("bogus_ref".to_string())
    );
    assert_eq!(inherited.path, base_path);
    assert!(inherited.source_text.contains("bogus_ref"));
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
            b_path.display()
        ),
    )
    .unwrap();
    std::fs::write(
        &b_path,
        format!(
            "let base = inherit(\"{}\")\ngrammar {{ language: \"b\", inherits: base }}",
            a_path.display()
        ),
    )
    .unwrap();
    let source = std::fs::read_to_string(&a_path).unwrap();
    let err = parse_native_dsl(&source, dir.path()).unwrap_err();
    let outer = assert_err!(err, Module);
    let DslError::Module(inner) = outer.inner.as_ref() else {
        panic!("expected nested Module error, got {:?}", outer.inner)
    };
    let DslError::Lower(e) = inner.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", inner.inner)
    };
    let LowerErrorKind::ModuleCycle(chain) = &e.kind else {
        panic!("expected ModuleCycle, got {:?}", e.kind)
    };
    let filenames: Vec<_> = chain
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap())
        .collect();
    assert_eq!(filenames, ["b.tsg", "a.tsg", "b.tsg"]);
}

#[test]
fn error_inherited_lower_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        "grammar { extras: [] }\nrule program { \"x\" }\n",
    )
    .unwrap();
    let parent_path = dir.path().join("parent.tsg");
    let parent_src = "let base = inherit(\"base.tsg\")\ngrammar { language: \"derived\", inherits: base }\nrule extra { \"hello\" }\n".to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();
    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    let DslError::Lower(lower_err) = inherited.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", inherited.inner)
    };
    assert_eq!(lower_err.kind, LowerErrorKind::MissingLanguageField);
    assert_eq!(inherited.path, base_path);
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
