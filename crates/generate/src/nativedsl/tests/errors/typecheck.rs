use super::super::*;

error_tests! { Type {
    error_type_mismatch_fn_args {
        r#"grammar { language: "test" }
        macro needs_int(x: int_t) rule_t = prec(x, "a")
        rule program { needs_int("not_an_int") }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Int, got: Ty::Str }
    }
    error_for_binding_count_mismatch {
        r#"grammar { language: "test" }
        rule bad { choice(for (a: str_t, b: int_t) in [("x")] { a }) }"#,
        TypeErrorKind::ForBindingCountMismatch { bindings: 2, tuple_elements: 1 }
    }
    error_append_non_list_arg {
        r#"grammar { language: "test" }
        let c = append("x", ["y"])
        rule program { "x" }"#,
        TypeErrorKind::AppendRequiresList(Ty::Str)
    }
    error_append_non_list {
        r#"grammar { language: "test" }
        let a: str_t = "x"
        let b: list_t<str_t> = ["y"]
        let c = append(a, b)
        rule program { "x" }"#,
        TypeErrorKind::AppendRequiresList(Ty::Str)
    }
    error_let_type_annotation_mismatch {
        r#"grammar { language: "test" }
        let X: int_t = "not_an_int"
        rule program { "x" }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Int, got: Ty::Str }
    }
    error_module_t_annotation_mismatch {
        r#"grammar { language: "test" }
        let X: module_t = "not_a_module"
        rule program { "x" }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::AnyModule, got: Ty::Str }
    }
    error_rule_called_as_function {
        r#"grammar { language: "test", word: ident }
        rule ident { regexp("[a-z]+") }
        rule program { ident("default", regexp("[a-z]+")) }"#,
        TypeErrorKind::UndefinedMacro("ident".into())
    }
    error_self_referential_let {
        r#"grammar { language: "test" }
        let P = { a: P.a }
        rule program { "x" }"#,
        TypeErrorKind::UnknownIdentifier("P".into())
    }
    error_wrong_arg_count {
        r#"grammar { language: "test" }
        macro one_arg(x: rule_t) rule_t = x
        rule program { one_arg("a", "b") }"#,
        TypeErrorKind::ArgCountMismatch { macro_name: "one_arg".into(), expected: 1, got: 2 }
    }
    error_rule_inline_on_non_grammar {
        r#"grammar { language: "test" }
        let obj = { x: 1 }
        rule program { obj::x }"#,
        TypeErrorKind::QualifiedAccessOnInvalidType(Ty::Object(InnerTy::Int))
    }
    error_field_access_on_non_object {
        r#"grammar { language: "test" }
        let x: int_t = 5
        rule program { prec(x.foo, "a") }"#,
        TypeErrorKind::FieldAccessOnNonObject(Ty::Int)
    }
    error_field_not_found {
        r#"grammar { language: "test" }
        let x = { a: 1 }
        rule program { prec(x.b, "a") }"#,
        TypeErrorKind::FieldNotFound { field: "b".into(), on_type: Ty::Object(InnerTy::Int) }
    }
    error_word_non_rule_var {
        r#"let my_word = 42
        grammar { language: "test", word: my_word }
        rule program { "y" }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Rule, got: Ty::Int }
    }
    error_config_access_unknown_field {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        let x = base::bogus"#,
        TypeErrorKind::ImportMemberNotFound("bogus".into())
    }
    error_for_requires_list {
        r#"grammar { language: "test" }
        let x: int_t = 5
        rule program { choice(for (v: int_t) in x { prec(v, "a") }) }"#,
        TypeErrorKind::ForRequiresList(Ty::Int)
    }
    error_rule_body_not_rule_typed {
        r#"grammar { language: "test" }
        rule program { 42 }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Rule, got: Ty::Int }
    }
    error_fn_return_type_mismatch {
        r#"grammar { language: "test" }
        macro bad(x: rule_t) int_t = x
        rule program { "x" }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Int, got: Ty::Rule }
    }
    error_prec_value_not_int_or_str {
        r#"grammar { language: "test" }
        rule program { prec(identifier, "x") }
        rule identifier { "id" }"#,
        TypeErrorKind::PrecValueTypeMismatch(Ty::Rule)
    }
    error_list_inconsistent_types {
        r#"grammar { language: "test" }
        let x = ["a", 1]
        rule program { "x" }"#,
        TypeErrorKind::ListElementTypeMismatch { first: Ty::Str, got: Ty::Int }
    }
    error_expected_rule_name_in_word {
        r#"grammar { language: "test", word: "not_a_name" }
        rule program { "x" }"#,
        TypeErrorKind::ExpectedRuleName
    }
    error_let_rhs_for_loop_rejected {
        r#"grammar { language: "test" }
        let foo = for (k: str_t) in ["a", "b"] { k }
        rule program { seq(foo) }"#,
        TypeErrorKind::NonBindableType(Ty::Spread)
    }
    error_conflicts_rejects_non_name {
        r#"grammar {
            language: "test",
            conflicts: [[repeat("x")]],
        }
        rule program { "x" }"#,
        TypeErrorKind::ExpectedRuleName
    }
    error_qualified_access_in_name_ref_config {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, inline: [base::_inline_rule] }"#,
        TypeErrorKind::ExpectedRuleName
    }
    error_int_literal_assigned_to_list_rule {
        r#"grammar { language: "test" }
        let x: list_t<rule_t> = [1, 2]
        rule program { "x" }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::ListRule, got: Ty::ListInt }
    }
    error_flat_list_assigned_to_list_list_rule {
        r#"grammar { language: "test" }
        let x: list_t<list_t<rule_t>> = [a, b]
        rule program { "x" }
        rule a { "a" }
        rule b { "b" }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::ListListRule, got: Ty::ListRule }
    }
    error_for_bindings_not_tuple {
        r#"grammar { language: "test" }
        let items: list_t<rule_t> = [identifier]
        rule identifier { regexp("[a-z]+") }
        rule program { seq(for (a: rule_t, b: rule_t) in items { a }) }"#,
        TypeErrorKind::ForRequiresTuples
    }
    error_invalid_object_value {
        r#"grammar { language: "test" }
        let inner = { a: 1 }
        let x = { nested: inner }
        rule program { "x" }"#,
        TypeErrorKind::InvalidObjectValue(Ty::Object(InnerTy::Int))
    }
    error_invalid_list_element {
        r#"grammar { language: "test" }
        let inner: list_t<list_t<rule_t>> = [[a]]
        let x = [inner]
        rule program { "x" }
        rule a { "a" }"#,
        TypeErrorKind::InvalidListElement
    }
    error_empty_list_needs_annotation {
        r#"grammar { language: "test" }
        let x = []
        rule program { "x" }"#,
        TypeErrorKind::EmptyContainerNeedsAnnotation
    }
    error_for_requires_tuples {
        r#"grammar { language: "test" }
        let items: list_t<str_t> = ["a", "b"]
        rule program { choice(for (a: str_t, b: str_t) in items { a }) }"#,
        TypeErrorKind::ForRequiresTuples
    }
    error_invalid_alias_target {
        r#"grammar { language: "test" }
        rule program { alias("x", 42) }"#,
        TypeErrorKind::InvalidAliasTarget(Ty::Int)
    }
    error_expected_reserved_config {
        r#"grammar { language: "test", reserved: "not_an_object" }
        rule program { "x" }"#,
        TypeErrorKind::ExpectedReservedConfig
    }
    error_cannot_infer_type {
        r#"grammar { language: "test" }
        let x = {}
        rule program { "x" }"#,
        TypeErrorKind::EmptyContainerNeedsAnnotation
    }
    error_bare_function_reference {
        r#"grammar { language: "test" }
        macro make_rule(x: str_t) rule_t = x
        rule program { make_rule }"#,
        TypeErrorKind::MacroUsedAsValue("make_rule".into())
    }
    error_concat_with_non_string_arg {
        r#"grammar { language: "test" }
        rule program { regexp(concat("a", 1)) }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Str, got: Ty::Int }
    }
    error_regexp_with_non_string_pattern {
        r#"grammar { language: "test" }
        let x: int_t = 1
        rule program { regexp(x) }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Str, got: Ty::Int }
    }
    error_regexp_with_non_string_flags {
        r#"grammar { language: "test" }
        let f: int_t = 1
        rule program { regexp("[a-z]+", f) }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Str, got: Ty::Int }
    }
    error_prec_dynamic_with_string_value {
        r#"grammar { language: "test" }
        rule program { prec_dynamic("high", "x") }"#,
        TypeErrorKind::TypeMismatch { expected: Ty::Int, got: Ty::Str }
    }
}}

inherit_error_tests! { Type {
    error_inherited_type_error {
        "grammar { language: \"base\" }\nrule program { 42 }\n",
        TypeErrorKind::TypeMismatch { expected: Ty::Rule, got: Ty::Int }
    }
}}

#[test]
fn error_grammar_config_on_import_module() {
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("helper.tsg");
    std::fs::write(&helper, "let X: int_t = 1").unwrap();
    let grammar = dir.path().join("grammar.tsg");
    let src = format!(
        "let h = import(\"{}\")\n\
         grammar {{ language: \"t\", extras: grammar_config(h, extras) }}\n\
         rule program {{ \"x\" }}",
        dsl_path(&helper)
    );
    std::fs::write(&grammar, &src).unwrap();
    let err = parse_native_dsl(&src, &grammar).unwrap_err();
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::GrammarConfigRequiresInherit);
}
