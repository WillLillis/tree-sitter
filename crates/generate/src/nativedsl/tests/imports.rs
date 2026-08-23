use super::*;
use crate::rules::{Precedence, Rule};
use std::fmt::Write as _;
use std::path::Path;

#[test]
fn import_function_uses_own_let_binding() {
    let mut g = parse_with_modules(
        &[(
            "self_ref.tsg",
            r#"
let DELIM = ","
macro delimited(item: rule_t) rule_t { seq(item, repeat(seq(DELIM, item))) }
"#,
        )],
        r#"
        let h = import("self_ref.tsg")
        grammar { language: "test" }
        rule program { h::delimited(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
    )
    .unwrap();
    let actual = find_rule(&g, "program");
    let expected = sep_by1_rule(&mut g.pool, ",", "identifier");
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn import_member_not_found_suggests_close_name() {
    // A misspelled member access gets a did-you-mean note, like in-module errors.
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::GREETIN }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::ImportMemberNotFound("GREETIN".into())
    );
    assert!(
        e.notes
            .iter()
            .any(|n| matches!(&n.message, NoteMessage::DidYouMean(s) if s == "GREETING")),
        "expected did-you-mean GREETING, got {:?}",
        e.notes
    );
}

#[test]
fn import_call_member_not_found_suggests_close_name() {
    // The call path resolves the missing member at resolve time too, so a
    // misspelled `h::sep_byy(...)` suggests the macro `sep_by`.
    let err = dsl_err(
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::sep_byy("x") }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::ImportMemberNotFound("sep_byy".into())
    );
    assert!(
        e.notes
            .iter()
            .any(|n| matches!(&n.message, NoteMessage::DidYouMean(s) if s == "sep_by")),
        "expected did-you-mean sep_by, got {:?}",
        e.notes
    );
}

#[test]
fn error_import_disallowed_items() {
    for (content, expected) in [
        (
            "override rule foo { \"x\" }",
            DisallowedItemKind::OverrideRule,
        ),
        (
            "grammar { language: \"bad\" }\nmacro f(x: rule_t) rule_t { x }",
            DisallowedItemKind::GrammarBlock,
        ),
        ("let b = inherit(\"base.tsg\")", DisallowedItemKind::Inherit),
    ] {
        let err = expect_err(parse_with_modules(
            &[("bad.tsg", content)],
            "let h = import(\"bad.tsg\")\ngrammar { language: \"test\" }\nrule program { \"x\" }",
        ));
        let outer = assert_err!(err, Module);
        let DslError::Lower(e) = outer.inner.as_ref() else {
            panic!("expected Lower error, got {:?}", outer.inner)
        };
        assert!(
            matches!(&e.kind, LowerErrorKind::ModuleDisallowedItem(k) if *k == expected),
            "got {:?}",
            e.kind
        );
    }
}

#[test]
fn error_import_cycle() {
    let err = expect_err(parse_with_modules(
        &[
            ("a.tsg", "let b = import(\"b.tsg\")"),
            ("b.tsg", "let a = import(\"a.tsg\")"),
        ],
        r#"
        let h = import("a.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    ));
    // Should detect cycle somewhere in the chain
    let is_cycle = matches!(
        &err,
        DslError::Module(m) if {
            fn has_cycle(e: &DslError) -> bool {
                match e {
                    DslError::Lower(l) => matches!(l.kind, LowerErrorKind::ModuleCycle),
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
fn error_import_cycle_three_levels() {
    let err = expect_err(parse_with_modules(
        &[
            ("a.tsg", "let b = import(\"b.tsg\")"),
            ("b.tsg", "let c = import(\"c.tsg\")"),
            ("c.tsg", "let a = import(\"a.tsg\")"),
        ],
        r#"let h = import("a.tsg")
        grammar { language: "test" }
        rule program { "x" }"#,
    ));
    fn has_cycle(e: &DslError) -> bool {
        match e {
            DslError::Lower(l) => matches!(l.kind, LowerErrorKind::ModuleCycle),
            DslError::Module(m) => has_cycle(&m.inner),
            _ => false,
        }
    }
    assert!(has_cycle(&err), "expected cycle error, got {err:?}");
}

/// Parse a root importing 256 helpers (root would be the 257th module), with
/// `tail` appended after the imports. Returns the error.
fn too_many_modules_err(tail: &str) -> DslError {
    let dir = tempfile::tempdir().unwrap();
    let mut root = String::new();
    for i in 0..256 {
        let path = dir.path().join(format!("h{i}.tsg"));
        std::fs::write(&path, format!("let v{i}: str_t = \"x\"")).unwrap();
        let _ = writeln!(root, "let h{i} = import(\"{}\")", dsl_path(&path));
    }
    root.push_str(tail);
    let root_path = dir.path().join("root.tsg");
    std::fs::write(&root_path, &root).unwrap();
    expect_err(parse_native_dsl(&root, &root_path))
}

#[test]
fn error_too_many_modules() {
    // Loading 256 helpers + 1 root tips the loader's module-id counter past
    // u8::MAX, which should surface as `ModuleTooMany`.
    let err = too_many_modules_err("grammar { language: \"test\" }\nrule program { \"x\" }\n");
    let e = assert_err!(err, Lower);
    assert!(matches!(e.kind, LowerErrorKind::ModuleTooMany));
}

#[test]
fn error_too_many_modules_fires_before_lowering() {
    // The 257th module must be rejected before its lowering runs: the
    // evaluator derives its root module id as `previous.len() as u8`, which
    // wraps to 0 for module 257 and misattributes module-0 spans. If lowering
    // ran, the overflowing prec below would error first.
    let err = too_many_modules_err(
        "grammar { language: \"test\" }\nrule program { prec(2147483647 + 1, \"x\") }\n",
    );
    let e = assert_err!(err, Lower);
    assert!(
        matches!(e.kind, LowerErrorKind::ModuleTooMany),
        "expected ModuleTooMany, got {:?}",
        e.kind
    );
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
    assert!(matches!(e.kind, LowerErrorKind::ModuleResolveFailed(_)));
}

#[test]
fn error_import_nested_error() {
    let err = expect_err(parse_with_modules(
        &[("bad_syntax.tsg", "let x = ~~~")],
        r#"
        let h = import("bad_syntax.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    ));
    let outer = assert_err!(err, Module);
    assert!(matches!(outer.inner.as_ref(), DslError::Lex(_)));
}

#[test]
fn error_import_transitive_nested_error() {
    // root -> middle.tsg -> bad.tsg (lex error)
    // Error should be double-wrapped: Module(Module(Lex(...)))
    let err = expect_err(parse_with_modules(
        &[
            ("bad.tsg", "let x = ~~~"),
            ("middle.tsg", "let h = import(\"bad.tsg\")"),
        ],
        r#"
        let m = import("middle.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    ));
    let outer = assert_err!(err, Module);
    let inner = assert_err!(*outer.inner, Module);
    assert!(matches!(inner.inner.as_ref(), DslError::Lex(_)));
}

#[test]
fn import_function_receives_complex_expr() {
    let mut g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1(seq(identifier, ":", identifier)) }
        rule identifier { regexp(r"[a-z]+") }
    "#);
    let actual = find_rule(&g, "program");
    let expected = {
        let p = &mut g.pool;
        // `pair` appears twice; the pool is a DAG so the same id is reused.
        let pair = {
            let a = r_sym!(p, "identifier");
            let colon = r_str!(p, ":");
            let b = r_sym!(p, "identifier");
            r_seq!(p, [a, colon, b])
        };
        let comma = r_str!(p, ",");
        let tail_item = r_seq!(p, [comma, pair]);
        let rep = r_repeat!(p, tail_item);
        let blank = r_blank!(p);
        let tail = r_choice!(p, [rep, blank]);
        r_seq!(p, [pair, tail])
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn import_diamond() {
    // A imports B and C, which both import helpers.tsg; each gets its own copy.
    let mut g = parse_with_modules(
        &[
            ("helpers.tsg", "let VAL = 10"),
            (
                "b.tsg",
                "let h = import(\"helpers.tsg\")\nmacro b_fn(x: rule_t) rule_t { prec(h::VAL, x) }",
            ),
            (
                "c.tsg",
                "let h = import(\"helpers.tsg\")\nmacro c_fn(x: rule_t) rule_t { prec(h::VAL, x) }",
            ),
        ],
        r#"
        let b = import("b.tsg")
        let c = import("c.tsg")
        grammar { language: "test" }
        rule program { choice(b::b_fn("a"), c::c_fn("b")) }
    "#,
    )
    .unwrap();
    let actual = find_rule(&g, "program");
    let expected = {
        let p = &mut g.pool;
        r_choice!(
            p,
            [
                r_prec!(p, Precedence::Integer(10), r_str!(p, "a")),
                r_prec!(p, Precedence::Integer(10), r_str!(p, "b"))
            ]
        )
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn import_diamond_dedups_shared_leaf() {
    // Loader-wide dedup loads the shared leaf once (130 modules), not once per
    // helper (257, which would trip the u8 module limit).
    let dir = tempfile::tempdir().unwrap();
    let leaf = dir.path().join("leaf.tsg");
    std::fs::write(&leaf, "let X: int_t = 10").unwrap();

    let mut imports = String::new();
    for i in 0..128 {
        let h = dir.path().join(format!("h{i}.tsg"));
        std::fs::write(
            &h,
            format!(
                "let l = import(\"{}\")\nmacro f{i}(r: rule_t) rule_t {{ prec(l::X, r) }}",
                dsl_path(&leaf)
            ),
        )
        .unwrap();
        let _ = writeln!(imports, "let h{i} = import(\"{}\")", dsl_path(&h));
    }
    let root =
        format!("{imports}grammar {{ language: \"test\" }}\nrule program {{ h0::f0(\"x\") }}\n");
    parse_native_dsl(&root, Path::new(".")).unwrap();
}

#[test]
fn helper_rule_collision_errors() {
    // Two helpers each defining `expression` -> resolver collision error.
    let err = expect_err(parse_with_modules(
        &[
            ("a.tsg", "rule expression { \"a\" }"),
            ("b.tsg", "rule expression { \"b\" }"),
        ],
        r#"
        let a = import("a.tsg")
        let b = import("b.tsg")
        grammar { language: "test" }
        rule program { expression }
    "#,
    ));
    let outer = assert_err!(err, Resolve);
    assert_eq!(
        outer.kind,
        ResolveErrorKind::DuplicateDeclaration("expression".into())
    );
}

#[test]
fn override_helper_rule() {
    // An `override` of a helper rule wins in the final grammar; the original is
    // still reachable via h::digit (which inlines).
    let mut g = parse_with_modules(
        &[("h.tsg", r#"rule digit { regexp(r"[0-9]") }"#)],
        r#"
        let h = import("h.tsg")
        grammar { language: "test" }
        rule program { digit }
        override rule digit { choice(h::digit, "x") }
    "#,
    )
    .unwrap();
    // Final `digit` is the override body, with h::digit inlined to the original.
    let actual = find_rule(&g, "digit");
    let expected = {
        let p = &mut g.pool;
        r_choice!(p, [r_pattern!(p, r"[0-9]"), r_str!(p, "x")])
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn helper_rule_transitive_promotion() {
    // root -> A -> B: root references B's `inner` by bare name, which transitive
    // promotion must make resolvable.
    let mut g = parse_with_modules(
        &[
            ("b.tsg", r#"rule inner { "leaf" }"#),
            ("a.tsg", "let b = import(\"b.tsg\")\nrule middle { inner }"),
        ],
        r#"
        let a = import("a.tsg")
        grammar { language: "test" }
        rule program { seq(middle, inner) }
    "#,
    )
    .unwrap();
    assert!(rule_names(&g).contains(&"inner"));
    assert!(rule_names(&g).contains(&"middle"));
    let actual = find_rule(&g, "inner");
    let expected = {
        let p = &mut g.pool;
        r_str!(p, "leaf")
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn helper_rule_qualified_inlines() {
    // `h::rule_name` inlines the rule body, mirroring `base::rule_name` for
    // inherited grammars.
    let mut g = parse_with_modules(
        &[("h.tsg", r#"rule greeting { "hello" }"#)],
        r#"
        let h = import("h.tsg")
        grammar { language: "test" }
        rule program { seq(h::greeting, "!") }
    "#,
    )
    .unwrap();
    // h::greeting inlined - program body has the literal "hello", not a
    // NamedSymbol reference.
    let actual = find_rule(&g, "program");
    let expected = {
        let p = &mut g.pool;
        r_seq!(p, [r_str!(p, "hello"), r_str!(p, "!")])
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn helper_rule_materialized_into_grammar() {
    // A helper rule referenced by bare name materializes as a top-level Variable,
    // reachable as a NamedSymbol.
    let mut g = parse_with_modules(
        &[("h.tsg", r#"rule digit { regexp(r"[0-9]") }"#)],
        r#"
        let h = import("h.tsg")
        grammar { language: "test" }
        rule program { repeat1(digit) }
    "#,
    )
    .unwrap();
    assert_eq!(rule_names(&g), vec!["program", "digit"]);
    let actual = find_rule(&g, "program");
    let expected = {
        let p = &mut g.pool;
        r_repeat!(p, r_sym!(p, "digit"))
    };
    assert_rule_eq(&g.pool, actual, expected);
    let actual = find_rule(&g, "digit");
    let expected = {
        let p = &mut g.pool;
        r_pattern!(p, r"[0-9]")
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn error_helper_rule_collides_with_root_rule() {
    // A helper rule and a root rule sharing a name is a duplicate declaration,
    // not silent shadowing (grammar JSON can't hold two same-named variables).
    let err = parse_with_modules(
        &[("h.tsg", r#"rule expression { "from_helper" }"#)],
        r#"
        let h = import("h.tsg")
        grammar { language: "test" }
        rule program { expression }
        rule expression { "from_root" }
    "#,
    );
    let e = assert_err!(expect_err(err), Resolve);
    assert_eq!(
        e.kind,
        ResolveErrorKind::DuplicateDeclaration("expression".into())
    );
}

#[test]
fn helper_can_define_rules() {
    // Mutually-recursive helper rules (plus a forward-declared external)
    // materialize into the root's variables, connecting to externals by bare name.
    let g = parse_with_modules(
        &[(
            "exp.tsg",
            r"
        expect _paren_open
        expect _paren_close
        rule expression { choice(application, seq(_paren_open, expression, _paren_close)) }
        rule application { seq(expression, expression) }
    ",
        )],
        r#"
        let exp = import("exp.tsg")
        grammar { language: "test", externals: [_paren_open, _paren_close] }
        rule program { expression }
    "#,
    )
    .unwrap();
    assert_eq!(rule_names(&g), vec!["program", "expression", "application"]);
}

#[test]
fn helper_rule_uses_grammar_registered_external() {
    // A helper forward-declares an external and uses it; the grammar registers it
    // by bare name, and they connect by name with no qualified reference.
    let mut g = parse_with_modules(
        &[("ext.tsg", "expect _tok\nrule wrapped { seq(_tok, _tok) }\n")],
        r#"
        let e = import("ext.tsg")
        grammar { language: "test", externals: [_tok] }
        rule program { wrapped }
    "#,
    )
    .unwrap();
    assert_eq!(g.external_roots.len(), 1);
    let actual = find_rule(&g, "wrapped");
    let expected = {
        let p = &mut g.pool;
        r_seq!(p, [r_sym!(p, "_tok"), r_sym!(p, "_tok")])
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn helper_hole_unregistered_external_is_rejected() {
    // With the grammar's `externals: [_tok]` dropped, the materialized helper rule
    // carries a dangling `_tok` that the completeness check rejects, anchored at
    // the helper's `expect`.
    let err = expect_err(parse_with_modules(
        &[("ext.tsg", "expect _tok\nrule wrapped { seq(_tok, _tok) }\n")],
        r#"
        let e = import("ext.tsg")
        grammar { language: "test" }
        rule program { wrapped }
    "#,
    ));
    let e = assert_err!(err, Lower);
    assert!(
        matches!(&e.kind, LowerErrorKind::UndefinedSymbols(names) if *names == ["_tok"]),
        "got {:?}",
        e.kind
    );
}

#[test]
fn import_empty_module() {
    let mut g = parse_with_modules(
        &[("empty.tsg", "// nothing here\n")],
        r#"
        let e = import("empty.tsg")
        grammar { language: "test" }
        rule program { "x" }
    "#,
    )
    .unwrap();
    let actual = find_rule(&g, "program");
    let expected = {
        let p = &mut g.pool;
        r_str!(p, "x")
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn import_value_in_config_extras() {
    let mut g = parse_with_modules(
        &[("ws.tsg", r#"let WS = [regexp(r"\s"), regexp(r"//[^\n]*")]"#)],
        r#"
        let ws = import("ws.tsg")
        grammar { language: "test", extras: ws::WS }
        rule program { "x" }
    "#,
    )
    .unwrap();
    let actual = g.extra_roots.clone();
    let expected = {
        let p = &mut g.pool;
        vec![r_pattern!(p, r"\s"), r_pattern!(p, r"//[^\n]*")]
    };
    assert_rules_eq(&g.pool, &actual, &expected);
}

#[test]
fn import_keyword_as_rule_name() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { import }
        rule import { "import_stmt" }
    "#);
    assert_eq!(g.pool.resolve(g.variables[1].name), "import");
}

#[test]
fn import_helper_rule_set_macro_expands_locally() {
    // A helper's `@pair(...)` rule-set call expands into ExpandedRule items that
    // flow into the importing grammar's variables.
    let mut g = dsl(r#"
        let h = import("import_helpers/rule_set_self.tsg")
        grammar { language: "test", start: program }
        rule program { seq(a_helper, b_helper) }
    "#);
    let names: Vec<&str> = g.variables.iter().map(|v| g.pool.resolve(v.name)).collect();
    assert!(names.contains(&"a_helper"), "missing a_helper in {names:?}");
    assert!(names.contains(&"b_helper"), "missing b_helper in {names:?}");
    let actual = find_rule(&g, "a_helper");
    let expected = {
        let p = &mut g.pool;
        r_str!(p, "x")
    };
    assert_rule_eq(&g.pool, actual, expected);
    let actual = find_rule(&g, "b_helper");
    let expected = {
        let p = &mut g.pool;
        r_str!(p, "y")
    };
    assert_rule_eq(&g.pool, actual, expected);
}

#[test]
fn import_call_depth_shared_across_modules() {
    let err = dsl_err(
        r#"
        let h = import("import_helpers/recursive.tsg")
        grammar { language: "test" }
        rule program { h::recurse("x") }
    "#,
    );

    let e = assert_err!(err, Lower);
    let LowerErrorKind::CallDepthExceeded(trace) = &e.kind else {
        panic!("expected CallDepthExceeded");
    };
    let fixtures = test_fixtures_dir();
    let grammar_path = fixtures.join("grammar.tsg");
    let recursive_path = fixtures.join("import_helpers/recursive.tsg");
    let (src, path) = e
        .src
        .as_deref()
        .expect("call-depth error must carry its source");
    assert_eq!(path, &grammar_path);
    assert!(
        src.contains(r#"h::recurse("x")"#),
        "expected root source, got {src:?}"
    );
    // First frame: call site in root grammar
    assert_eq!(trace[0], ("recurse".into(), grammar_path, 4, 24));
    // Remaining frames: self-recursive calls within the imported helper
    for frame in &trace[1..] {
        assert_eq!(*frame, ("recurse".into(), recursive_path.clone(), 4, 5));
    }
}

#[test]
fn import_rule_preserves_metadata_and_reserved() {
    // h::decorated inlines via import_rule; it must reconstruct the same Rule as
    // lowering directly, covering the Metadata (prec/field/alias/token) and
    // Reserved arms.
    let g = dsl(r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule direct { prec_left(1, field(f, alias(token("kw"), Kw))) }
        rule via_import { h::decorated }
        rule res_direct { reserved("ctx", "rw") }
        rule res_via_import { h::reserved_rule }
    "#);
    assert_rule_eq(
        &g.pool,
        find_rule(&g, "via_import"),
        find_rule(&g, "direct"),
    );
    assert_rule_eq(
        &g.pool,
        find_rule(&g, "res_via_import"),
        find_rule(&g, "res_direct"),
    );
}

#[test]
fn helper_rules_materialize_in_import_source_order() {
    let g = parse_with_modules(
        &[
            ("a.tsg", "rule a_rule { \"a\" }\n"),
            ("b.tsg", "rule b_rule { \"b\" }\n"),
        ],
        r#"
        let ha = import("a.tsg")
        let hb = import("b.tsg")
        grammar { language: "test" }
        rule program { "p" }
    "#,
    )
    .unwrap();
    let names: Vec<&str> = g.variables.iter().map(|v| g.pool.resolve(v.name)).collect();
    // ha imports before hb, so its rules materialize first.
    assert_eq!(names, vec!["program", "a_rule", "b_rule"]);
}

#[test]
fn override_reaching_helper_top_level_via_macro_is_rejected() {
    // A helper can't inherit, so an `override` reaching its top level via a
    // rules-macro call has nothing to override and is rejected (the macro
    // definition stays legal; only the call is rejected).
    let err = expect_err(parse_with_modules(
        &[(
            "helper.tsg",
            r#"
            rules wrap(rhs: rule_t) { override rule expr { choice("a", rhs) } }
            @wrap("b")
            "#,
        )],
        r#"
        let h = import("helper.tsg")
        grammar { language: "t" }
        rule program { h::expr }
        "#,
    ));
    let outer = assert_err!(err, Module);
    let DslError::Lower(e) = outer.inner.as_ref() else {
        panic!("expected Lower error, got {:?}", outer.inner)
    };
    assert!(
        matches!(
            e.kind,
            LowerErrorKind::ModuleDisallowedItem(DisallowedItemKind::OverrideRule)
        ),
        "got {:?}",
        e.kind
    );
}

find_rule_tests! {
    import_function_expands_comma_sep1 {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| comma_sep1_rule(p, "identifier")
    }
    import_function_expands_comma_sep {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| comma_sep_rule(p, "identifier")
    }
    import_function_with_string_param {
        r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { n::sep_by1(";", identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| sep_by1_rule(p, ";", "identifier")
    }
    import_function_intra_module_call {
        // double_wrap(x) = wrap(seq(x, ",", x))
        // wrap(y) = seq("(", y, ")")
        r#"
        let c = import("import_helpers/chained.tsg")
        grammar { language: "test" }
        rule program { c::double_wrap(identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| r_seq!(p, [r_str!(p, "("), r_seq!(p, [r_sym!(p, "identifier"), r_str!(p, ","), r_sym!(p, "identifier")]), r_str!(p, ")")])
    }
    import_string_value {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::GREETING }
    "#,
        "program",
        |p| r_str!(p, "hello")
    }
    import_let_name_does_not_collide_with_local {
        r#"
        let h = import("import_helpers/helpers.tsg")
        let GREETING: str_t = "goodbye"
        grammar { language: "test" }
        rule program { seq(h::GREETING, GREETING) }
    "#,
        "program",
        |p| r_seq!(p, [r_str!(p, "hello"), r_str!(p, "goodbye")])
    }
    import_int_value_in_prec {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program {
            prec_left(h::PREC.ADD, seq(program, "+", program))
        }
    "#,
        "program",
        |p| r_prec_left!(p, Precedence::Integer(1), r_seq!(p, [r_sym!(p, "program"), r_str!(p, "+"), r_sym!(p, "program")]))
    }
    import_value_reassigned_to_local {
        r#"
        let h = import("import_helpers/helpers.tsg")
        let p = h::PREC
        grammar { language: "test" }
        rule program {
            prec_left(p.MUL, seq(program, "*", program))
        }
    "#,
        "program",
        |p| r_prec_left!(p, Precedence::Integer(2), r_seq!(p, [r_sym!(p, "program"), r_str!(p, "*"), r_sym!(p, "program")]))
    }
    import_transitive_function_body {
        r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { n::sep_by1(",", identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| sep_by1_rule(p, ",", "identifier")
    }
    import_transitive_value {
        r#"
        let n = import("import_helpers/nested.tsg")
        grammar { language: "test" }
        rule program { prec(n::NESTED_VAL, "x") }
    "#,
        "program",
        |p| r_prec!(p, Precedence::Integer(42), r_str!(p, "x"))
    }
    import_multiple_modules_body {
        r#"
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
    "#,
        "program",
        |p| {
            let a = sep_by1_rule(p, ",", "identifier");
            let b = sep_by1_rule(p, ";", "identifier");
            r_choice!(p, [a, b])
        }
    }
    import_function_result_in_seq {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { seq("{", h::comma_sep1(identifier), "}") }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| {
            let open = r_str!(p, "{");
            let inner = comma_sep1_rule(p, "identifier");
            let close = r_str!(p, "}");
            r_seq!(p, [open, inner, close])
        }
    }
    import_module_values_only {
        r#"
        let m = import("import_helpers/minimal.tsg")
        grammar { language: "test" }
        rule program { prec(m::X, "x") }
    "#,
        "program",
        |p| r_prec!(p, Precedence::Integer(1), r_str!(p, "x"))
    }
    import_function_receives_caller_let_binding {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        let SEP: str_t = ";"
        rule program { h::sep_by(SEP, identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| sep_by1_rule(p, ";", "identifier")
    }
    import_function_receives_object_field {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        let SEPS = { list: ",", stmt: ";" }
        rule program { h::sep_by(SEPS.stmt, identifier) }
        rule identifier { regexp(r"[a-z]+") }
    "#,
        "program",
        |p| sep_by1_rule(p, ";", "identifier")
    }
    // Both arms resolve to the same "hello", and `choice` dedups, so the result
    // has a single member. The expectation is written with one member because
    // `r_choice!` bypasses dedup - it describes the literal shape.
    import_same_module_twice {
        r#"
        let h1 = import("import_helpers/helpers.tsg")
        let h2 = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { choice(h1::GREETING, h2::GREETING) }
    "#,
        "program",
        |p| r_choice!(p, [r_str!(p, "hello")])
    }
    chained_let_alias_to_import_module {
        r#"
        let h = import("import_helpers/helpers.tsg")
        let h2 = h
        grammar { language: "test" }
        rule program { h2::GREETING }
    "#,
        "program",
        |p| r_str!(p, "hello")
    }
}

error_tests! { Resolve {
    error_import_member_not_found {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent }
    "#,
        ResolveErrorKind::ImportMemberNotFound("nonexistent".into())
    }
    error_import_function_not_found {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::nonexistent("x") }
    "#,
        ResolveErrorKind::ImportMemberNotFound("nonexistent".into())
    }
}}

error_tests! { Type {
    error_import_macro_used_as_value {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1 }
    "#,
        TypeErrorKind::MacroUsedAsValue("h::comma_sep1".into())
    }
    error_qualified_call_on_non_module {
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { x::something("y") }
    "#,
        TypeErrorKind::TypeMismatch {
            expected: Ty::ANY_MODULE,
            got: Ty::Data(DataTy::Object(InnerTy::Scalar(ScalarTy::Int))),
        }
    }
    error_import_wrong_arg_count {
        r#"
        let h = import("import_helpers/helpers.tsg")
        grammar { language: "test" }
        rule program { h::comma_sep1("a", "b") }
    "#,
        TypeErrorKind::ArgCountMismatch {
            macro_name: "comma_sep1".into(),
            expected: 1,
            got: 2,
        }
    }
}}
