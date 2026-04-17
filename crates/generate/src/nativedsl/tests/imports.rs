use super::*;
use std::path::Path;

// -- Basic import functionality --

#[test]
fn import_function_call() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(g.variables[0].name, "program");
}

#[test]
fn import_value_access() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program {
            prec_left(h::PREC.ADD, seq(program, "+", program))
        }
    "#);
    assert_eq!(g.variables[0].name, "program");
}

#[test]
fn import_string_value() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::GREETING }
    "#);
    assert_eq!(
        g.variables[0].rule,
        crate::rules::Rule::String("hello".into())
    );
}

#[test]
fn import_multiple_modules() {
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program {
            choice(
                h::comma_sep1(identifier),
                n::sep_by1(";", identifier),
            )
        }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(g.variables[0].name, "program");
}

#[test]
fn import_transitive() {
    let g = dsl(r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { n::sep_by1(",", identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    assert_eq!(g.variables[0].name, "program");
}

// -- Error cases --

#[test]
fn error_import_member_not_found() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ImportMemberNotFound("nonexistent".into())
    );
}

#[test]
fn error_import_function_not_found() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent("x") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::ImportFunctionNotFound("nonexistent".into())
    );
}

#[test]
fn error_qualified_call_on_non_module() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { x::something("y") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::QualifiedCallOnNonModule(Ty::Object(InnerTy::Int))
    );
}

#[test]
fn error_import_disallowed_rule() {
    let dir = tempfile::tempdir().unwrap();
    let bad_helper = dir.path().join("bad.tsg");
    std::fs::write(&bad_helper, "rule foo { \"x\" }").unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        bad_helper.display()
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let outer = assert_err!(err, Module);
    let DslError::Lower(e) = outer.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", outer.inner)
    };
    assert_eq!(e.kind, LowerErrorKind::ModuleDisallowedItem);
}

#[test]
fn error_import_disallowed_grammar_block() {
    let dir = tempfile::tempdir().unwrap();
    let bad_helper = dir.path().join("bad.tsg");
    std::fs::write(
        &bad_helper,
        "grammar { language: \"bad\" }\nfn f(x: rule_t) -> rule_t { x }",
    )
    .unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        bad_helper.display()
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    let outer = assert_err!(err, Module);
    let DslError::Lower(e) = outer.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", outer.inner)
    };
    assert_eq!(e.kind, LowerErrorKind::ModuleDisallowedItem);
}

#[test]
fn error_import_cycle() {
    let dir = tempfile::tempdir().unwrap();
    let a_path = dir.path().join("a.tsg");
    let b_path = dir.path().join("b.tsg");

    std::fs::write(&a_path, format!("let b = import(\"{}\")", b_path.display())).unwrap();
    std::fs::write(&b_path, format!("let a = import(\"{}\")", a_path.display())).unwrap();

    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "test" }}
        rule program {{ "x" }}
    "#,
        a_path.display()
    );
    let err = parse_native_dsl(&input, Path::new(".")).unwrap_err();
    // Should detect cycle somewhere in the chain
    let is_cycle = matches!(
        &err,
        DslError::Module(m) if {
            fn has_cycle(e: &DslError) -> bool {
                match e {
                    DslError::Lower(l) => matches!(l.kind, LowerErrorKind::ModuleCycle(_)),
                    DslError::Module(m) => has_cycle(&m.inner),
                    _ => false,
                }
            }
            has_cycle(&m.inner)
        }
    );
    assert!(is_cycle, "expected cycle error, got {err:?}");
}

#[test]
fn error_import_bad_path() {
    let err = dsl_err(
        r#"
        let h = import("nonexistent/helpers.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::ModuleResolveFailed { .. }));
}

// -- Keyword as identifier in imported context --

#[test]
fn import_keyword_as_rule_name() {
    // The keyword `import` should work as a rule name
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { import }
        rule import { "import_stmt" }
    "#);
    assert_eq!(g.variables[1].name, "import");
}
