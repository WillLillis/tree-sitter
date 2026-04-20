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
    // Cross-kind duplicate: let binding with same name as a rule
    error_duplicate_let_and_rule {
        r#"grammar { language: "test" }
        let program: str_t = "x"
        rule program { "y" }"#,
        TypeErrorKind::DuplicateDeclaration("program".into())
    }
    // Unknown identifier inside a function body
    error_unknown_in_fn_body {
        r#"grammar { language: "test" }
        fn f(x: rule_t) -> rule_t { bogus }
        rule program { f(program) }"#,
        TypeErrorKind::UnknownIdentifier("bogus".into())
    }
    // Unknown identifier inside a for-loop body
    error_unknown_in_for_body {
        r#"grammar { language: "test" }
        rule program { for (v: rule_t) in [program] { bogus } }"#,
        TypeErrorKind::UnknownIdentifier("bogus".into())
    }
    // Externals via for-loop expression
    error_externals_via_for_loop {
        r#"grammar { language: "test", externals: for (x: rule_t) in [a] { x } }
        rule program { "x" }"#,
        TypeErrorKind::InvalidExternalsExpression
    }
}

#[test]
fn error_duplicate_declaration_has_note() {
    let e = assert_err!(
        dsl_err(r#"grammar { language: "test" } rule foo { "a" } rule foo { "b" }"#),
        Type
    );
    assert_eq!(e.kind, TypeErrorKind::DuplicateDeclaration("foo".into()));
    let note = e
        .note
        .as_ref()
        .expect("DuplicateDeclaration should have a note");
    assert_eq!(note.message, NoteMessage::FirstDefinedHere);
    // The note span should point to the first declaration, not the duplicate.
    // "rule foo { "a" }" starts at byte 29.
    assert!(
        note.span.start < e.span.start,
        "note span should precede error span"
    );
}

#[test]
fn external_and_rule_same_name_is_valid() {
    // A rule can also be an external token - the rule is registered first,
    // then the externals walk sees it's already declared and skips re-registration.
    let g = dsl(r#"
        grammar { language: "test", externals: [foo] }
        rule program { "x" }
        rule foo { "y" }
    "#);
    assert_eq!(g.external_tokens, vec![Rule::NamedSymbol("foo".into())]);
    assert_eq!(g.variables.len(), 2);
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
