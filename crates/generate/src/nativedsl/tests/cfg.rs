//! Tests for `flags: { enabled: [...], disabled: [...] }` config + `#[cfg(X)]`.

use super::*;

#[test]
fn cfg_rule_def_enabled() {
    let g = dsl(
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
    );
    assert_eq!(rule_names(&g), vec!["program", "strikethrough"]);
}

#[test]
fn cfg_rule_def_disabled() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
    );
    // The cfg-dropped rule is gone, so the reference is undefined.
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier(ref n) if n == "strikethrough"
    ));
}

#[test]
fn cfg_choice_member_enabled() {
    let g = dsl(
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { choice("a", #[cfg(GFM)] "b", "c") }
    "#,
    );
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::choice(vec![
            Rule::String("a".into()),
            Rule::String("b".into()),
            Rule::String("c".into()),
        ])
    );
}

#[test]
fn cfg_choice_member_disabled() {
    let g = dsl(
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { choice("a", #[cfg(GFM)] "b", "c") }
    "#,
    );
    // GFM disabled -> "b" arm dropped from the choice.
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::choice(vec![
            Rule::String("a".into()),
            Rule::String("c".into()),
        ])
    );
}

#[test]
fn cfg_seq_member_disabled() {
    let g = dsl(
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { seq("a", #[cfg(GFM)] "b", "c") }
    "#,
    );
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::seq(vec![Rule::String("a".into()), Rule::String("c".into())])
    );
}

#[test]
fn cfg_nested_both_active() {
    let g = dsl(
        r#"
        grammar { language: "t", flags: { enabled: ["A", "B"] } }
        rule program { choice("x", #[cfg(A)] #[cfg(B)] "y", "z") }
    "#,
    );
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::choice(vec![
            Rule::String("x".into()),
            Rule::String("y".into()),
            Rule::String("z".into()),
        ])
    );
}

#[test]
fn cfg_nested_inner_off() {
    let g = dsl(
        r#"
        grammar { language: "t", flags: { enabled: ["A"], disabled: ["B"] } }
        rule program { choice("x", #[cfg(A)] #[cfg(B)] "y", "z") }
    "#,
    );
    // Outer A is on but inner B is off -> drop the whole "y" branch.
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::choice(vec![Rule::String("x".into()), Rule::String("z".into())])
    );
}

#[test]
fn cfg_unknown_flag_errors() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { choice("a", #[cfg(TYPO)] "b") }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::CfgFlagUnknown(ref n) if n == "TYPO"
    ));
}

#[test]
fn cfg_in_non_list_position_parse_error() {
    // cfg attributes are only allowed in items + list members. Anywhere else
    // (e.g. as the bare value in a rule body) is a parse error.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { #[cfg(GFM)] "x" }
    "#,
    );
    assert_err!(err, Parse);
}

#[test]
fn cfg_flags_unknown_key_errors() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { active: ["GFM"] } }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(e.kind, ResolveErrorKind::CfgFlagsUnknownKey(ref s) if s == "active"));
}

#[test]
fn cfg_in_grammar_config_field() {
    // cfg attrs on conflicts/precedences/externals members must be processed
    // by apply_cfg too, not just on rule bodies. Regression test for an
    // earlier bug where Node::Grammar was treated as a leaf.
    let g = dsl(
        r#"
        grammar {
            language: "t",
            flags: { enabled: ["A"], disabled: ["B"] },
            conflicts: [
                [foo, bar],
                #[cfg(A)] [foo, baz],
                #[cfg(B)] [bar, baz],
            ],
        }
        rule foo { "a" }
        rule bar { "b" }
        rule baz { "c" }
    "#,
    );
    // A active -> [foo, baz] kept; B disabled -> [bar, baz] dropped.
    assert_eq!(g.expected_conflicts.len(), 2);
}

#[test]
fn cfg_dropped_decl_enriches_undefined_symbol_error() {
    // Reference to a cfg-dropped rule should fail as UnknownIdentifier with a
    // note pointing at the gated decl and naming the flag.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier(ref n) if n == "strikethrough"
    ));
    let note = e.note.as_ref().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "GFM"));
}

#[test]
fn helper_module_inherits_cfg_from_importer() {
    // Helper modules can't declare their own flags (no grammar block). They
    // see the importing grammar's full declared set transparently.
    let dir = tempfile::tempdir().unwrap();
    let helper = dir.path().join("h.tsg");
    std::fs::write(
        &helper,
        r#"
            macro gated() rule_t { choice("a", #[cfg(GFM)] "b", "c") }
        "#,
    )
    .unwrap();
    let input = format!(
        r#"
        let h = import("{}")
        grammar {{ language: "t", flags: {{ enabled: ["GFM"] }} }}
        rule program {{ h::gated() }}
        "#,
        dsl_path(&helper)
    );
    let g = parse_native_dsl(&input, std::path::Path::new(".")).unwrap();
    // GFM is enabled in the importer; the helper's cfg branch survives.
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::choice(vec![
            Rule::String("a".into()),
            Rule::String("b".into()),
            Rule::String("c".into()),
        ])
    );
}

#[test]
fn cfg_inside_flags_errors() {
    // cfg attrs inside the `flags` field itself are nonsensical: declarations
    // are read before any cfg gating runs. Reject with a clear error.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: ["X", #[cfg(X)] "FOO"] } }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(e.kind, ResolveErrorKind::CfgHasCfg));
}

#[test]
fn cfg_disabled_import_does_not_load_file() {
    // The path doesn't exist on disk. If apply_cfg ran *after* child loading
    // (the old order) we'd get a ModuleReadFailed error before cfg got a
    // chance to drop the let. With early gating, the import is gone before
    // load_import_children runs and the parse succeeds.
    let input = r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        #[cfg(GFM)]
        let h = import("does-not-exist.tsg")
        rule program { "x" }
    "#;
    let g = parse_native_dsl(input, std::path::Path::new(".")).unwrap();
    assert_eq!(rule_names(&g), vec!["program"]);
}

#[test]
fn cfg_disabled_inherit_does_not_merge_parent() {
    // Parent grammar defines `parent_only`. If cfg gating doesn't actually
    // skip the inherit load, we'd see `parent_only` in the child's rule set.
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("parent.tsg");
    std::fs::write(
        &parent,
        r#"
            grammar { language: "p" }
            rule parent_only { "p" }
        "#,
    )
    .unwrap();
    let input = format!(
        r#"
        grammar {{ language: "t", flags: {{ disabled: ["EXT"] }} }}
        #[cfg(EXT)]
        let base = inherit("{}")
        rule program {{ "x" }}
        "#,
        dsl_path(&parent)
    );
    let g = parse_native_dsl(&input, std::path::Path::new(".")).unwrap();
    assert_eq!(rule_names(&g), vec!["program"]);
}

#[test]
fn cfg_enabled_inherit_still_loads_parent() {
    // Positive control: with the flag on, parent rules merge as usual.
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("parent.tsg");
    std::fs::write(
        &parent,
        r#"
            grammar { language: "p" }
            rule parent_only { "p" }
        "#,
    )
    .unwrap();
    let input = format!(
        r#"
        #[cfg(EXT)]
        let base = inherit("{}")
        grammar {{
            language: "t",
            flags: {{ enabled: ["EXT"] }},
            inherits: base,
        }}
        rule program {{ "x" }}
        "#,
        dsl_path(&parent)
    );
    let g = parse_native_dsl(&input, std::path::Path::new(".")).unwrap();
    assert_eq!(rule_names(&g), vec!["parent_only", "program"]);
}

#[test]
fn cfg_inheriting_grammar_overrides_parent_flag_value() {
    // Both modules declare flag X. The inheriting (root) grammar loads first,
    // so `merge_module_flags` puts its value into the global active map first;
    // the inherited parent's later write loses to first-write-wins.
    //
    // Setup: parent declares X disabled and gates `parent_only`. Child
    // declares X enabled and gates `child_only`. With child overriding,
    // X=enabled wins and BOTH gated rules survive.
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("parent.tsg");
    std::fs::write(
        &parent,
        r#"
            grammar { language: "p", flags: { disabled: ["X"] } }
            rule parent_base { "p" }
            #[cfg(X)]
            rule parent_only { "po" }
        "#,
    )
    .unwrap();
    let input = format!(
        r#"
        let base = inherit("{}")
        grammar {{
            language: "t",
            flags: {{ enabled: ["X"] }},
            inherits: base,
        }}
        rule program {{ "x" }}
        #[cfg(X)]
        rule child_only {{ "co" }}
        "#,
        dsl_path(&parent)
    );
    let g = parse_native_dsl(&input, std::path::Path::new(".")).unwrap();
    // parent_base appears unconditionally; both X-gated rules survive because
    // child's enabled overrode parent's disabled in the global active map.
    assert_eq!(
        rule_names(&g),
        vec!["parent_base", "parent_only", "program", "child_only"]
    );
}

#[test]
fn cfg_flags_non_string_errors() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: [GFM] } }
        rule program { "x" }
    "#,
    );
    // GFM here is an identifier, not a string literal.
    let e = assert_err!(err, Resolve);
    assert!(matches!(e.kind, ResolveErrorKind::CfgFlagsNonLiteral));
}
