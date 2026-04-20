use super::super::*;

macro_rules! resolve_error_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let e = assert_err!(dsl_err($input), Type);
            assert_eq!(e.kind, $expected);
        })*
    };
}

resolve_error_tests! {
    error_unknown_identifier {
        r#"grammar { language: "test" } rule program { nonexistent }"#,
        TypeErrorKind::UnknownIdentifier("nonexistent".into())
    }
    error_duplicate_rule {
        r#"grammar { language: "test" } rule program { "a" } rule program { "b" }"#,
        TypeErrorKind::DuplicateDeclaration("program".into())
    }
    error_forward_reference_let {
        r#"grammar { language: "test" } rule program { MY_VAR } let MY_VAR: str_t = "x""#,
        TypeErrorKind::UnknownIdentifier("MY_VAR".into())
    }
    error_undefined_function {
        r#"grammar { language: "test" }
        rule program { nonexistent_fn(identifier) }
        rule identifier { "x" }"#,
        TypeErrorKind::UnknownIdentifier("nonexistent_fn".into())
    }
    error_duplicate_let {
        r#"grammar { language: "test" } let X: int_t = 1 let X: int_t = 2 rule program { "x" }"#,
        TypeErrorKind::DuplicateDeclaration("X".into())
    }
    error_duplicate_fn {
        r#"grammar { language: "test" }
        fn f(x: rule_t) -> rule_t { x }
        fn f(x: rule_t) -> rule_t { x }
        rule program { "x" }"#,
        TypeErrorKind::DuplicateDeclaration("f".into())
    }
}

#[test]
fn error_inherited_resolve_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        "grammar { language: \"base\" }\nrule program { undefined_name }\n",
    )
    .unwrap();
    let parent_path = dir.path().join("parent.tsg");
    let parent_src = "let base = inherit(\"base.tsg\")\ngrammar { language: \"derived\", inherits: base }\nrule extra { \"hello\" }\n".to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();
    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    let DslError::Type(type_err) = inherited.inner.as_ref() else {
        panic!("expected Type error, got {:?}", inherited.inner)
    };
    assert_eq!(
        type_err.kind,
        TypeErrorKind::UnknownIdentifier("undefined_name".to_string())
    );
    assert_eq!(inherited.path, base_path);
    assert!(inherited.source_text.contains("undefined_name"));
}
