//! Tests for `flags: { enabled: [...], disabled: [...] }` config + `#[cfg(X)]`.

use super::*;

#[test]
fn cfg_rule_def_enabled() {
    let g = dsl(r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#);
    assert_eq!(rule_names(&g), vec!["program", "strikethrough"]);
}

#[test]
fn cfg_active_let_keeps_type_annotation() {
    // Unwrapping an ACTIVE #[cfg] on an annotated let must carry the let's type
    // annotation (kept in let_types keyed by the Let's node id) to the unwrapped
    // slot. Otherwise typecheck reads None there and an empty-container let fails
    // with EmptyContainerNeedsAnnotation, even though the same grammar without
    // the (enabled) cfg compiles.
    let g = dsl(r#"
        grammar { language: "test", flags: { enabled: ["X"] } }
        #[cfg(X)] let x: list_t<str_t> = []
        rule foo { "x" }
    "#);
    assert_eq!(rule_names(&g), vec!["foo"]);
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
    let g = dsl(r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { choice("a", #[cfg(GFM)] "b", "c") }
    "#);
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
    let g = dsl(r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { choice("a", #[cfg(GFM)] "b", "c") }
    "#);
    // GFM disabled -> "b" arm dropped from the choice.
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::choice(vec![Rule::String("a".into()), Rule::String("c".into()),])
    );
}

#[test]
fn cfg_concat_member_disabled() {
    // Regression: walk_children's Concat arm previously only recursed,
    // never filtering. A disabled cfg in concat() left a stale Node::Cfg
    // for typecheck to panic on.
    let g = dsl(r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { concat("a", #[cfg(GFM)] "b", "c") }
    "#);
    assert_eq!(find_rule(&g, "program"), &Rule::String("ac".into()));
}

#[test]
fn cfg_seq_member_disabled() {
    let g = dsl(r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { seq("a", #[cfg(GFM)] "b", "c") }
    "#);
    assert_eq!(
        find_rule(&g, "program"),
        &Rule::seq(vec![Rule::String("a".into()), Rule::String("c".into())])
    );
}

#[test]
fn cfg_nested_both_active() {
    let g = dsl(r#"
        grammar { language: "t", flags: { enabled: ["A", "B"] } }
        rule program { choice("x", #[cfg(A)] #[cfg(B)] "y", "z") }
    "#);
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
    let g = dsl(r#"
        grammar { language: "t", flags: { enabled: ["A"], disabled: ["B"] } }
        rule program { choice("x", #[cfg(A)] #[cfg(B)] "y", "z") }
    "#);
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
fn cfg_flags_not_object_errors() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: ["X"] }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(e.kind, ResolveErrorKind::CfgFlagsNotObject));
}

#[test]
fn cfg_flags_not_list_errors() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: "X" } }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(e.kind, ResolveErrorKind::CfgFlagsNotList));
}

#[test]
fn cfg_flag_declared_twice_errors() {
    // A flag appearing more than once in this module's `flags` (whether
    // both in `enabled`/`disabled` or twice in the same list) is a hard
    // error with a `FirstDefinedHere` note pointing at the first occurrence.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: ["X"], disabled: ["X"] } }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::CfgFlagDeclaredTwice(ref n) if n == "X"
    ));
    let note = e.notes.first().expect("expected FirstDefinedHere note");
    assert!(matches!(note.message, NoteMessage::FirstDefinedHere));
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
    let g = dsl(r#"
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
    "#);
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
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "GFM"));
}

#[test]
fn cfg_enrichment_preserves_existing_note() {
    // An UnknownIdentifier can already carry a `did you mean` note. When the
    // same name was also a cfg-dropped decl, enrich appends the cfg note rather
    // than clobbering the existing one.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { disabled: ["X"] } }
        #[cfg(X)] rule widget { "x" }
        rule widgets { "y" }
        rule program { widget }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier(ref n) if n == "widget"
    ));
    let kinds: Vec<_> = e.notes.iter().map(|n| &n.message).collect();
    assert!(
        kinds
            .iter()
            .any(|m| matches!(m, NoteMessage::DidYouMean(s) if s == "widgets")),
        "expected DidYouMean note, got {kinds:?}"
    );
    assert!(
        kinds
            .iter()
            .any(|m| matches!(m, NoteMessage::GatedByDisabledCfg(f) if f == "X")),
        "expected GatedByDisabledCfg note, got {kinds:?}"
    );
}

#[test]
fn cfg_dropped_macro_enriches_error() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { disabled: ["X"] } }
        rule program { gated() }
        #[cfg(X)] macro gated() rule_t { "a" }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier(ref n) if n == "gated"
    ));
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "X"));
}

#[test]
fn cfg_dropped_ruleset_macro_enriches_error() {
    // A cfg-dropped rule-set macro invoked via `@name()` errors at expand with
    // UnknownMacro, enriched with the same GatedByDisabledCfg note the
    // expression-macro path gets at resolve.
    let err = dsl_err(
        r#"
        #[cfg(X)] rules gated() { rule a { "x" } }
        grammar { language: "t", flags: { disabled: ["X"] } }
        rule program { "p" }
        @gated()
    "#,
    );
    let e = assert_err!(err, Expand);
    assert!(matches!(e.kind, ExpandErrorKind::UnknownMacro(ref n) if n == "gated"));
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "X"));
}

#[test]
fn cfg_dropped_macro_keeps_same_named_survivor() {
    // Two cfg branches define the same rule-set macro; only the enabled one
    // survives. macro_index is rebuilt from the post-cfg root_items, so `@dup()`
    // stays bound to the survivor regardless of which branch is parsed last.
    let g = dsl(r#"
        #[cfg(Y)] rules dup() { rule program { "stable" } }
        #[cfg(X)] rules dup() { rule program { "gated" } }
        grammar { language: "test", flags: { enabled: ["Y"], disabled: ["X"] } }
        @dup()
    "#);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].rule, Rule::String("stable".into()));
}

#[test]
fn cfg_dropped_external_enriches_error() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { disabled: ["X"] } }
        rule program { gated_token }
        #[cfg(X)] external gated_token
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier(ref n) if n == "gated_token"
    ));
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "X"));
}

#[test]
fn cfg_dropped_let_enriches_error() {
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { disabled: ["X"] } }
        #[cfg(X)] let gated_let = "x"
        rule program { gated_let }
    "#,
    );
    let e = assert_err!(err, Resolve);
    assert!(matches!(
        e.kind,
        ResolveErrorKind::UnknownIdentifier(ref n) if n == "gated_let"
    ));
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "X"));
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
fn cfg_attribute_nesting_is_bounded() {
    // Pathological nesting must error rather than blow the parse / apply_cfg
    // recursion stack. 300 layers exceeds MAX_PARSE_DEPTH.
    let nest = "#[cfg(X)] ".repeat(300);
    let src = format!(
        r#"grammar {{ language: "t", flags: {{ enabled: ["X"] }} }} {nest} rule r {{ "x" }}"#
    );
    let err = dsl_err(&src);
    let e = assert_err!(err, Parse);
    assert!(matches!(e.kind, ParseErrorKind::NestingTooDeep));
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
    assert!(matches!(e.kind, ResolveErrorKind::CfgInsideFlags));
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
fn cfg_disabled_first_inherit_promotes_second() {
    // The first inherit() is cfg-disabled, so the second must become the active
    // inherit. apply_cfg used to clear inherit_ref but leave a stale
    // duplicate_inherit, so validate_grammar's `inherit_ref.expect(...)` panicked.
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.tsg");
    std::fs::write(
        &base,
        "grammar { language: \"base\" }\nrule base_rule { \"b\" }\n",
    )
    .unwrap();
    let input = format!(
        r#"
        #[cfg(X)] let skipped = inherit("missing.tsg")
        let chosen = inherit("{}")
        grammar {{ language: "t", inherits: chosen, flags: {{ disabled: ["X"] }} }}
        rule program {{ "x" }}
        "#,
        dsl_path(&base)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert!(
        rule_names(&g).contains(&"base_rule"),
        "got {:?}",
        rule_names(&g)
    );
}

#[test]
fn cfg_disabled_second_inherit_is_not_multiple_inherits() {
    // The second inherit() is cfg-disabled, leaving a single active inherit.
    // apply_cfg left duplicate_inherit set, so this valid grammar was wrongly
    // rejected with MultipleInherits pointing at the disabled line.
    let dir = tempfile::tempdir().unwrap();
    let base = dir.path().join("base.tsg");
    std::fs::write(
        &base,
        "grammar { language: \"base\" }\nrule base_rule { \"b\" }\n",
    )
    .unwrap();
    let input = format!(
        r#"
        let chosen = inherit("{}")
        #[cfg(X)] let skipped = inherit("missing.tsg")
        grammar {{ language: "t", inherits: chosen, flags: {{ disabled: ["X"] }} }}
        rule program {{ "x" }}
        "#,
        dsl_path(&base)
    );
    let g = parse_native_dsl(&input, Path::new(".")).unwrap();
    assert!(
        rule_names(&g).contains(&"base_rule"),
        "got {:?}",
        rule_names(&g)
    );
}

#[test]
fn cfg_dropped_attribution_uses_owning_module() {
    // Both grammars in the inherit chain drop a rule named `strikethrough`
    // under different cfg flags. Parent's resolve fires UnknownIdentifier
    // for its own reference to `strikethrough`; the note should point at
    // parent's own cfg flag (P), not at child's (C) just because child
    // happened to apply_cfg first and win or_insert in a global map.
    let dir = tempfile::tempdir().unwrap();
    let parent = dir.path().join("parent.tsg");
    std::fs::write(
        &parent,
        r#"
            grammar { language: "p", flags: { disabled: ["P"] } }
            rule program_p { strikethrough }
            #[cfg(P)] rule strikethrough { "p" }
        "#,
    )
    .unwrap();
    let input = format!(
        r#"
        let base = inherit("{}")
        grammar {{
            language: "t",
            flags: {{ disabled: ["C"] }},
            inherits: base,
        }}
        rule program {{ "x" }}
        #[cfg(C)] rule strikethrough {{ "c" }}
        "#,
        dsl_path(&parent),
    );
    let err = parse_native_dsl(&input, std::path::Path::new(".")).unwrap_err();
    // Error originates from the inherited (parent) grammar load, so it's
    // wrapped in `Module(...)` once.
    let inner = *assert_err!(err, Module).inner;
    let e = assert_err!(inner, Resolve);
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "P"));
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
fn cfg_three_level_inheritance_root_flag_wins() {
    // Three-level chain: root -> parent -> grandparent. Each declares X.
    // Grandparent and parent want X enabled; root forces it disabled.
    // First-write-wins on the global active map (root parses first), so
    // X=disabled across all three levels and every gated rule drops.
    let dir = tempfile::tempdir().unwrap();
    let grandparent = dir.path().join("grandparent.tsg");
    std::fs::write(
        &grandparent,
        r#"
            grammar { language: "g", flags: { enabled: ["X"] } }
            rule grandparent_base { "g" }
            #[cfg(X)]
            rule grandparent_only { "go" }
        "#,
    )
    .unwrap();
    let parent = dir.path().join("parent.tsg");
    std::fs::write(
        &parent,
        format!(
            r#"
                let gbase = inherit("{}")
                grammar {{
                    language: "p",
                    flags: {{ enabled: ["X"] }},
                    inherits: gbase,
                }}
                rule parent_base {{ "p" }}
                #[cfg(X)]
                rule parent_only {{ "po" }}
            "#,
            dsl_path(&grandparent),
        ),
    )
    .unwrap();
    let input = format!(
        r#"
        let pbase = inherit("{}")
        grammar {{
            language: "t",
            flags: {{ disabled: ["X"] }},
            inherits: pbase,
        }}
        rule program {{ "x" }}
        #[cfg(X)]
        rule root_only {{ "ro" }}
        "#,
        dsl_path(&parent),
    );
    let g = parse_native_dsl(&input, std::path::Path::new(".")).unwrap();
    // X=disabled (root won), so all three `#[cfg(X)]` rules dropped.
    assert_eq!(
        rule_names(&g),
        vec!["grandparent_base", "parent_base", "program"]
    );
}

#[test]
fn cfg_in_macro_body_bare_expression_parse_error() {
    // Macro body is a single expression, not a list of items. cfg in
    // expression position is a parse error - same rule as rule bodies.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: ["X"] } }
        macro m() rule_t { #[cfg(X)] "a" }
        rule program { m() }
    "#,
    );
    let e = assert_err!(err, Parse);
    // parse_expr hits `#` and bails before any cfg-specific check.
    assert!(matches!(e.kind, ParseErrorKind::ExpectedExpression));
}

#[test]
fn cfg_in_for_loop_body_parse_error() {
    // for-loop body is expression-shaped (single expression that gets
    // spread per iteration). cfg in that position is rejected the same
    // way as cfg in any other expression slot.
    let err = dsl_err(
        r#"
        grammar { language: "t", flags: { enabled: ["X"] } }
        rule program { for (x: str_t) in ["a", "b"] { #[cfg(X)] x } }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert!(matches!(e.kind, ParseErrorKind::ExpectedExpression));
}

#[test]
fn cfg_on_grammar_block_is_rejected() {
    // Gating the grammar block itself doesn't model anything coherent: the
    // block's own `flags` would have to declare the gating flag, which is
    // either self-defeating or inconsistent. Parser rejects.
    let err = dsl_err(
        r#"
        #[cfg(X)] grammar { language: "t", flags: { enabled: ["X"] } }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert!(matches!(e.kind, ParseErrorKind::CfgOnGrammarBlock));
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
