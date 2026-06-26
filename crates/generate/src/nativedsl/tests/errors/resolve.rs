use super::super::*;

error_tests! { Resolve {
    error_unknown_identifier {
        r#"grammar { language: "test" } rule program { nonexistent }"#,
        ResolveErrorKind::UnknownIdentifier("nonexistent".into())
    }
    error_symref_inner_resolved_in_rule_set_template {
        // `@<ident>` in a rule-set template has its inner name resolved like any
        // ident; an undefined one is a normal unknown-identifier error (it once
        // stayed Unresolved and crashed definition-time typecheck).
        r#"grammar { language: "test" }
        rules r() { rule x { @foo } }
        rule program { "x" }"#,
        ResolveErrorKind::UnknownIdentifier("foo".into())
    }
    error_computed_rule_name_resolved_in_rule_set_template {
        // The computed name in `rule @<expr> { ... }` is resolved too; an undefined
        // bare-ident name is a normal unknown-identifier error, not a typecheck crash.
        r#"grammar { language: "test" }
        rules r() { rule @foo { "x" } }
        rule program { "x" }"#,
        ResolveErrorKind::UnknownIdentifier("foo".into())
    }
    error_duplicate_rule {
        r#"grammar { language: "test" } rule program { "a" } rule program { "b" }"#,
        ResolveErrorKind::DuplicateDeclaration("program".into())
    }
    error_undefined_function {
        r#"grammar { language: "test" }
        rule program { nonexistent_fn(identifier) }
        rule identifier { "x" }"#,
        ResolveErrorKind::UnknownIdentifier("nonexistent_fn".into())
    }
    error_duplicate_let {
        r#"grammar { language: "test" } let X: int_t = 1 let X: int_t = 2 rule program { "x" }"#,
        ResolveErrorKind::DuplicateDeclaration("X".into())
    }
    error_duplicate_fn {
        r#"grammar { language: "test" }
        macro f(x: rule_t) rule_t { x }
        macro f(x: rule_t) rule_t { x }
        rule program { "x" }"#,
        ResolveErrorKind::DuplicateDeclaration("f".into())
    }
    error_duplicate_let_and_rule {
        r#"grammar { language: "test" }
        let program: str_t = "x"
        rule program { "y" }"#,
        ResolveErrorKind::DuplicateDeclaration("program".into())
    }
    error_unknown_in_fn_body {
        r#"grammar { language: "test" }
        macro f(x: rule_t) rule_t { bogus }
        rule program { f(program) }"#,
        ResolveErrorKind::UnknownIdentifier("bogus".into())
    }
    error_unknown_in_for_body {
        r#"grammar { language: "test" }
        rule program { for (v: rule_t) in [program] { bogus } }"#,
        ResolveErrorKind::UnknownIdentifier("bogus".into())
    }
    error_externals_via_for_loop {
        r#"grammar { language: "test", externals: for (x: rule_t) in [a] { x } }
        rule program { "x" }"#,
        ResolveErrorKind::InvalidExternalsExpression
    }
    error_self_referential_let_in_externals {
        r#"let C = C
        grammar { language: "test", externals: [C] }
        rule program { "x" }"#,
        ResolveErrorKind::InvalidExternalsExpression
    }
    error_config_access_unknown_field {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base }
        let x = base::bogus"#,
        ResolveErrorKind::ImportMemberNotFound("bogus".into())
    }
    // Mutual let references should not cause infinite recursion in
    // collect_external_names.
    error_mutual_let_ref_in_externals {
        r#"let A: list_t<rule_t> = B
        let B: list_t<rule_t> = A
        grammar { language: "test", externals: A }
        rule program { "x" }"#,
        ResolveErrorKind::InvalidExternalsExpression
    }
}}

#[test]
fn error_self_ref_in_container_reports_cycle() {
    // `let X = [X]` references itself. Lets resolve regardless of order, so the
    // self-reference is caught as a cycle at typecheck, with a self-reference note.
    let e = assert_err!(
        dsl_err(
            r#"grammar { language: "test" }
            let X: list_t<rule_t> = [X]
            rule program { "x" }"#
        ),
        Type
    );
    assert_eq!(e.kind, TypeErrorKind::CircularLet("X".into()));
    assert!(
        matches!(
            e.notes.first().map(|n| &n.message),
            Some(NoteMessage::SelfReferenceHere)
        ),
        "expected a self-reference note: {:?}",
        e.notes
    );
}

#[test]
fn error_duplicate_declaration_has_note() {
    let e = assert_err!(
        dsl_err(r#"grammar { language: "test" } rule foo { "a" } rule foo { "b" }"#),
        Resolve
    );
    assert_eq!(e.kind, ResolveErrorKind::DuplicateDeclaration("foo".into()));
    let note = e
        .notes
        .first()
        .expect("DuplicateDeclaration should have a note");
    assert_eq!(note.message, NoteMessage::FirstDefinedHere);
    // The note span should point to the first declaration, not the duplicate.
    // "rule foo { "a" }" starts at byte 29.
    assert!(
        note.span.start < e.span.unwrap().start,
        "note span should precede error span"
    );
}

inherit_error_tests! { Resolve {
    error_inherited_resolve_error {
        "grammar { language: \"base\" }\nrule program { undefined_name }\n",
        ResolveErrorKind::UnknownIdentifier("undefined_name".into())
    }
}}
