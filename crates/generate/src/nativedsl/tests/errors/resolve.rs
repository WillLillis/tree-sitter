use super::super::*;

#[test]
fn error_unknown_identifier() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { nonexistent }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier("nonexistent".into())
    );
}

#[test]
fn error_duplicate_rule() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { "a" }
        rule program { "b" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::DuplicateDeclaration("program".into())
    );
}

#[test]
fn error_forward_reference_let() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { MY_VAR }
        let MY_VAR: str_t = "x"
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(e.kind, ResolveErrorKind::UnknownIdentifier("MY_VAR".into()));
}

#[test]
fn error_undefined_function() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { nonexistent_fn(identifier) }
        rule identifier { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier("nonexistent_fn".into())
    );
}

#[test]
fn error_duplicate_let() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let X: int_t = 1
        let X: int_t = 2
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(e.kind, ResolveErrorKind::DuplicateDeclaration("X".into()));
}

#[test]
fn error_duplicate_fn() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn f(x: rule_t) -> rule_t { x }
        fn f(x: rule_t) -> rule_t { x }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(e.kind, ResolveErrorKind::DuplicateDeclaration("f".into()));
}

#[test]
fn error_inherited_resolve_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        r#"grammar { language: "base" }
rule program { undefined_name }
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

    let DslError::Inherited(inherited) = &err else {
        panic!("expected Inherited error, got {err:?}")
    };
    let DslError::Resolve(resolve_err) = inherited.inner.as_ref() else {
        panic!("expected Resolve error, got {:?}", inherited.inner)
    };
    assert_eq!(
        resolve_err.kind,
        ResolveErrorKind::UnknownIdentifier("undefined_name".to_string())
    );
    assert_eq!(inherited.path, base_path);
    assert!(inherited.source_text.contains("undefined_name"));
    assert_eq!(
        &parent_src[inherited.inherit_span.start as usize..inherited.inherit_span.end as usize],
        "base.tsg"
    );
}
