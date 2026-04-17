use std::path::Path;

use super::super::*;

#[test]
fn error_override_without_inherit() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        override rule foo { "bar" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::OverrideWithoutInherit);
}

#[test]
fn error_override_rule_not_found() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        override rule nonexistent { "oops" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(
        e.kind,
        LowerErrorKind::OverrideRuleNotFound("nonexistent".into())
    );
}

#[test]
fn error_inherit_child_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        r#"grammar { language: "base" }
rule program { bogus_ref }
"#,
    )
    .unwrap();

    let parent_path = dir.path().join("parent.tsg");
    let parent_src = r#"let base = inherit("base.tsg")
grammar { language: "derived", inherits: base }
rule extra { "hello" }
"#
    .to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();

    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    // Inner error is a resolve error from the base grammar
    let DslError::Resolve(resolve_err) = inherited.inner.as_ref() else {
        panic!("expected Resolve error, got {:?}", inherited.inner)
    };
    assert_eq!(
        resolve_err.kind,
        ResolveErrorKind::UnknownIdentifier("bogus_ref".to_string())
    );
    // ModuleError carries the base file's context
    assert_eq!(inherited.path, base_path);
    assert!(inherited.source_text.contains("bogus_ref"));
    // reference_span points to the base.tsg path in the parent (quotes stripped)
    assert_eq!(
        &parent_src[inherited.reference_span.start as usize..inherited.reference_span.end as usize],
        "base.tsg"
    );
}

#[test]
fn error_inherit_bad_path() {
    let err = dsl_err(
        r#"
        let base = inherit("nonexistent/grammar.tsg")
        grammar { language: "derived", inherits: base }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::ModuleResolveFailed { .. }));
}

#[test]
fn error_inherit_bad_extension() {
    let dir = tempfile::tempdir().unwrap();
    let bad_file = dir.path().join("grammar.txt");
    std::fs::write(&bad_file, "not a grammar").unwrap();

    let input = format!(
        r#"
        let base = inherit("{}")
        grammar {{ language: "derived", inherits: base }}
    "#,
        bad_file.display()
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::ModuleUnsupportedExtension);
}

#[test]
fn error_inherit_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let a_path = dir.path().join("a.tsg");
    let b_path = dir.path().join("b.tsg");

    std::fs::write(
        &a_path,
        format!(
            r#"
        let base = inherit("{}")
        grammar {{ language: "a", inherits: base }}
    "#,
            b_path.display()
        ),
    )
    .unwrap();

    std::fs::write(
        &b_path,
        format!(
            r#"
        let base = inherit("{}")
        grammar {{ language: "b", inherits: base }}
    "#,
            a_path.display()
        ),
    )
    .unwrap();

    let source = std::fs::read_to_string(&a_path).unwrap();
    let err = parse_native_dsl(&source, dir.path()).unwrap_err();
    // a inherits b, b inherits a -> cycle detected when a is loaded recursively from b
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
fn error_missing_grammar_block() {
    let err = dsl_err(
        r#"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::MissingGrammarBlock);
}

#[test]
fn error_missing_language_field() {
    let err = dsl_err(
        r#"
        grammar { extras: [] }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::MissingLanguageField);
}

#[test]
fn multiple_inherit_bindings_rejected() {
    let err = dsl_err(
        r#"
        let a = inherit("inherit_base/grammar.tsg")
        let b = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: a }
        rule extra { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::MultipleInherits);
}

#[test]
fn error_inherit_without_config() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::InheritWithoutConfig);
}

#[test]
fn error_inherits_without_inherit() {
    let err = dsl_err(
        r#"
        grammar { language: "test", inherits: "not_inherit" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::InheritsWithoutInherit);
}

#[test]
fn error_config_field_unset() {
    // inherit_base_no_word has no word set, so grammar_config(base).word produces ConfigFieldUnset
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base_no_word/grammar.tsg")
        grammar { language: "test", inherits: base, word: grammar_config(base).word }
        override rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::ConfigFieldUnset);
}

#[test]
fn error_inherit_rule_not_found() {
    // base::nonexistent_rule is caught at typecheck since :: access
    // validates members against the module's TypeEnv.
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        rule extra { base::nonexistent_rule }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ImportMemberNotFound("nonexistent_rule".into())
    );
}

#[test]
fn error_inherited_lower_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        r#"grammar { extras: [] }
rule program { "x" }
"#,
    )
    .unwrap();

    let parent_path = dir.path().join("parent.tsg");
    let parent_src = r#"let base = inherit("base.tsg")
grammar { language: "derived", inherits: base }
rule extra { "hello" }
"#
    .to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();

    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    let DslError::Lower(lower_err) = inherited.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", inherited.inner)
    };
    assert_eq!(lower_err.kind, LowerErrorKind::MissingLanguageField);
    assert_eq!(inherited.path, base_path);
    assert_eq!(
        &parent_src[inherited.reference_span.start as usize..inherited.reference_span.end as usize],
        "base.tsg"
    );
}

#[test]
fn error_word_not_named_symbol() {
    // word config accepts any VarRef with Ty::Rule at typecheck, but lower
    // requires a NamedSymbol specifically (a bare rule reference, not a
    // computed rule expression).
    let err = dsl_err(
        r#"
        let my_word = seq("x", "y")
        grammar { language: "test", word: my_word }
        rule program { "z" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert_eq!(e.kind, LowerErrorKind::ExpectedRuleName);
}
