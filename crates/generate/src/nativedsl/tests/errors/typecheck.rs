use super::super::*;

#[test]
fn error_type_mismatch_fn_args() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn needs_int(x: int_t) -> rule_t { prec(x, "a") }
        rule program { needs_int("not_an_int") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Str
        }
    );
}

#[test]
fn error_for_binding_count_mismatch() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule bad { choice(for (a: str_t, b: int_t) in [("x")] { a }) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ForBindingCountMismatch {
            bindings: 2,
            tuple_elements: 1
        }
    );
}

#[test]
fn error_append_non_list_arg() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let c = append("x", ["y"])
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::AppendRequiresList(Ty::Str));
}

#[test]
fn error_append_non_list() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let a: str_t = "x"
        let b: list_str_t = ["y"]
        let c = append(a, b)
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::AppendRequiresList(Ty::Str));
}

#[test]
fn error_let_type_annotation_mismatch() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let X: int_t = "not_an_int"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Str
        }
    );
}

#[test]
fn error_rule_called_as_function() {
    let err = dsl_err(
        r#"
        grammar { language: "test", word: ident }
        rule program { ident("default", regexp("[a-z]+")) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::UndefinedFunction("ident".into()));
}

#[test]
fn error_self_referential_let() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let P = { a: P.a }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::UnresolvedVariable("P".into()));
}

#[test]
fn error_wrong_arg_count() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn one_arg(x: rule_t) -> rule_t { x }
        rule program { one_arg("a", "b") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ArgCountMismatch {
            fn_name: "one_arg".into(),
            expected: 1,
            got: 2,
        }
    );
}

#[test]
fn error_rule_inline_on_non_grammar() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let obj = { x: 1 }
        rule program { obj::x }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ConfigAccessOnNonGrammar(Ty::Object(InnerTy::Int))
    );
}

#[test]
fn error_field_access_on_non_object() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: int_t = 5
        rule program { prec(x.foo, "a") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::FieldAccessOnNonObject(Ty::Int));
}

#[test]
fn error_field_not_found() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { prec(x.b, "a") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::FieldNotFound {
            field: "b".into(),
            on_type: Ty::Object(InnerTy::Int),
        }
    );
}

#[test]
fn error_word_non_rule_var() {
    let err = dsl_err(
        r#"
        let my_word = 42
        grammar { language: "test", word: my_word }
        rule program { "y" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Rule,
            got: Ty::Int,
        }
    );
}

#[test]
fn error_config_access_unknown_field() {
    let err = dsl_err(
        r#"
        let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        let x = base.bogus
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::UnknownConfigField("bogus".into()));
}

#[test]
fn error_for_requires_list() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: int_t = 5
        rule program { choice(for (v: int_t) in x { prec(v, "a") }) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ForRequiresList(Ty::Int));
}

#[test]
fn error_rule_body_not_rule_typed() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { 42 }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Rule,
            got: Ty::Int
        }
    );
}

#[test]
fn error_fn_return_type_mismatch() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        fn bad(x: rule_t) -> int_t { x }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Rule
        }
    );
}

#[test]
fn error_prec_value_not_int_or_str() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { prec(identifier, "x") }
        rule identifier { "id" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Int,
            got: Ty::Rule
        }
    );
}

#[test]
fn error_list_inconsistent_types() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = ["a", 1]
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ListElementTypeMismatch {
            first: Ty::Str,
            got: Ty::Int
        }
    );
}

#[test]
fn error_expected_rule_name_in_word() {
    let err = dsl_err(
        r#"
        grammar { language: "test", word: "not_a_name" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ExpectedRuleName);
}

#[test]
fn error_print_rejects_for_loop_arg() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        print(for (k: str_t) in ["a", "b"] { k })
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::CannotPrintType(Ty::Spread));
}

#[test]
fn error_let_rhs_for_loop_rejected() {
    // Previously panicked at lowering with `unreachable!` because lower has
    // no `Node::For` case in `eval_expr`. Typecheck now rejects at the let.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let foo = for (k: str_t) in ["a", "b"] { k }
        rule program { seq(foo) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::CannotBindSpread);
}

#[test]
fn error_void_t_type_annotation_rejected() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: void_t = "hello"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::VoidTypeNotAllowed);
}

#[test]
fn error_spread_t_type_annotation_rejected() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: spread_t = "hello"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::SpreadTypeNotAllowed);
}

#[test]
fn error_conflicts_rejects_non_name() {
    // conflicts inner elements must be name refs.
    let err = dsl_err(
        r#"
        grammar {
            language: "test",
            conflicts: [[repeat("x")]],
        }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ExpectedRuleName);
}

#[test]
fn error_int_literal_assigned_to_list_rule() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: list_rule_t = [1, 2]
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::ListRule,
            got: Ty::ListInt,
        }
    );
}

#[test]
fn error_flat_list_assigned_to_list_list_rule() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: list_list_rule_t = [a, b]
        rule program { "x" }
        rule a { "a" }
        rule b { "b" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::ListListRule,
            got: Ty::ListRule,
        }
    );
}

#[test]
fn error_for_bindings_not_tuple() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let items: list_rule_t = [identifier]
        rule identifier { regexp("[a-z]+") }
        rule program { seq(for (a: rule_t, b: rule_t) in items { a }) }
    "#,
    );
    let e = assert_err!(err, Type);
    matches!(
        e.kind,
        TypeErrorKind::ForBindingsNotTuple { bindings: 2, .. }
    );
}

#[test]
fn error_inherited_type_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        r#"grammar { language: "base" }
rule program { 42 }
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
    let DslError::Type(type_err) = inherited.inner.as_ref() else {
        panic!("expected Type error, got {:?}", inherited.inner)
    };
    assert_eq!(
        type_err.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Rule,
            got: Ty::Int
        }
    );
    assert_eq!(inherited.path, base_path);
    assert!(inherited.source_text.contains("42"));
    assert_eq!(
        &parent_src[inherited.inherit_span.start as usize..inherited.inherit_span.end as usize],
        "base.tsg"
    );
}

#[test]
fn error_invalid_object_value() {
    // Object field values must be rule_t, str_t, int_t, list_rule_t, or
    // list_str_t. list_int_t is not a valid object value type.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: [1, 2] }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::InvalidObjectValue(Ty::ListInt));
}

#[test]
fn error_invalid_list_element() {
    // Lists can contain rule_t, str_t, int_t, or list types (up to 2 levels).
    // A 3-level nested list (list of list_list_rule_t) is not allowed.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let inner: list_list_rule_t = [[a]]
        let x = [inner]
        rule program { "x" }
        rule a { "a" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::InvalidListElement);
}

#[test]
fn error_empty_list_needs_annotation() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = []
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::EmptyListNeedsAnnotation);
}

#[test]
fn error_for_requires_tuples() {
    // For-loop with multiple bindings and a non-literal iterable requires tuples.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let items: list_str_t = ["a", "b"]
        rule program { choice(for (a: str_t, b: str_t) in items { a }) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ForRequiresTuples);
}

#[test]
fn error_invalid_alias_target() {
    // Alias target must be a rule reference or string, not an integer.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { alias("x", 42) }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::InvalidAliasTarget(Ty::Int));
}

#[test]
fn error_expected_reserved_config() {
    // Reserved config must be an object literal or inherited value.
    let err = dsl_err(
        r#"
        grammar { language: "test", reserved: "not_an_object" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::ExpectedReservedConfig);
}

#[test]
fn error_cannot_infer_type() {
    // Empty object has no fields to infer the value type from.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = {}
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::CannotInferType);
}
