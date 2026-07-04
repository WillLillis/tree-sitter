//! Tests for `flags: { enabled: [...], disabled: [...] }` config + `#[cfg(X)]`.

use super::*;

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
fn cfg_flag_declared_twice_errors() {
    // A flag declared twice (across or within enabled/disabled) errors, with a
    // FirstDefinedHere note at the first occurrence.
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
fn cfg_in_grammar_config_field() {
    // apply_cfg must process cfg on grammar-config members (conflicts/etc.),
    // not just rule bodies (Node::Grammar was once treated as a leaf).
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
    // When a name is both a typo (did-you-mean) and cfg-dropped, enrich appends
    // the cfg note instead of clobbering the existing one.
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
    // A cfg-dropped `@name()` rule-set macro errors at expand as UnknownMacro,
    // enriched with the same GatedByDisabledCfg note as the resolve path.
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
    // macro_index is rebuilt from post-cfg root_items, so when two cfg branches
    // define the same rule-set macro `@dup()` binds to the enabled survivor.
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
        #[cfg(X)] expect gated_token
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
    let g = parse_with_modules(
        &[(
            "h.tsg",
            r#"
            macro gated() rule_t { choice("a", #[cfg(GFM)] "b", "c") }
        "#,
        )],
        r#"
        let h = import("h.tsg")
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { h::gated() }
        "#,
    )
    .unwrap();
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
fn cfg_disabled_import_does_not_load_file() {
    // Early cfg gating drops the disabled import before load_import_children
    // runs, so the nonexistent path is never read (old order: ModuleReadFailed).
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
    let g = parse_with_modules(
        &[(
            "parent.tsg",
            r#"
            grammar { language: "p" }
            rule parent_only { "p" }
        "#,
        )],
        r#"
        grammar { language: "t", flags: { disabled: ["EXT"] } }
        #[cfg(EXT)]
        let base = inherit("parent.tsg")
        rule program { "x" }
        "#,
    )
    .unwrap();
    assert_eq!(rule_names(&g), vec!["program"]);
}

#[test]
fn cfg_enabled_inherit_still_loads_parent() {
    // Positive control: with the flag on, parent rules merge as usual.
    let g = parse_with_modules(
        &[(
            "parent.tsg",
            r#"
            grammar { language: "p" }
            rule parent_only { "p" }
        "#,
        )],
        r#"
        #[cfg(EXT)]
        let base = inherit("parent.tsg")
        grammar {
            language: "t",
            flags: { enabled: ["EXT"] },
            inherits: base,
        }
        rule program { "x" }
        "#,
    )
    .unwrap();
    assert_eq!(rule_names(&g), vec!["parent_only", "program"]);
}

#[test]
fn cfg_disabled_first_inherit_promotes_second() {
    // A cfg-disabled first inherit promotes the second (apply_cfg once cleared
    // inherit_ref but left a stale duplicate_inherit, panicking validate_grammar).
    let g = parse_with_modules(
        &[(
            "base.tsg",
            "grammar { language: \"base\" }\nrule base_rule { \"b\" }\n",
        )],
        r#"
        #[cfg(X)] let skipped = inherit("missing.tsg")
        let chosen = inherit("base.tsg")
        grammar { language: "t", inherits: chosen, flags: { disabled: ["X"] } }
        rule program { "x" }
        "#,
    )
    .unwrap();
    assert!(
        rule_names(&g).contains(&"base_rule"),
        "got {:?}",
        rule_names(&g)
    );
}

#[test]
fn cfg_disabled_second_inherit_is_not_multiple_inherits() {
    // A cfg-disabled second inherit leaves one active inherit (apply_cfg once
    // left duplicate_inherit set, wrongly raising MultipleInherits).
    let g = parse_with_modules(
        &[(
            "base.tsg",
            "grammar { language: \"base\" }\nrule base_rule { \"b\" }\n",
        )],
        r#"
        let chosen = inherit("base.tsg")
        #[cfg(X)] let skipped = inherit("missing.tsg")
        grammar { language: "t", inherits: chosen, flags: { disabled: ["X"] } }
        rule program { "x" }
        "#,
    )
    .unwrap();
    assert!(
        rule_names(&g).contains(&"base_rule"),
        "got {:?}",
        rule_names(&g)
    );
}

#[test]
fn cfg_dropped_attribution_uses_owning_module() {
    // When parent and child both cfg-drop a `strikethrough` under different flags,
    // parent's UnknownIdentifier note must name parent's flag (P), not child's (C)
    // winning a global or_insert.
    let err = parse_with_modules(
        &[(
            "parent.tsg",
            r#"
            grammar { language: "p", flags: { disabled: ["P"] } }
            rule program_p { strikethrough }
            #[cfg(P)] rule strikethrough { "p" }
        "#,
        )],
        r#"
        let base = inherit("parent.tsg")
        grammar {
            language: "t",
            flags: { disabled: ["C"] },
            inherits: base,
        }
        rule program { "x" }
        #[cfg(C)] rule strikethrough { "c" }
        "#,
    )
    .unwrap_err();
    // Error originates from the inherited (parent) grammar load, so it's
    // wrapped in `Module(...)` once.
    let inner = *assert_err!(err, Module).inner;
    let e = assert_err!(inner, Resolve);
    let note = e.notes.first().expect("expected cfg note on error");
    assert!(matches!(note.message, NoteMessage::GatedByDisabledCfg(ref f) if f == "P"));
}

#[test]
fn cfg_inheriting_grammar_overrides_parent_flag_value() {
    // First-write-wins on the global flag map: the root grammar loads first, so
    // its X=enabled overrides the parent's X=disabled and both gated rules survive.
    let g = parse_with_modules(
        &[(
            "parent.tsg",
            r#"
            grammar { language: "p", flags: { disabled: ["X"] } }
            rule parent_base { "p" }
            #[cfg(X)]
            rule parent_only { "po" }
        "#,
        )],
        r#"
        let base = inherit("parent.tsg")
        grammar {
            language: "t",
            flags: { enabled: ["X"] },
            inherits: base,
        }
        rule program { "x" }
        #[cfg(X)]
        rule child_only { "co" }
        "#,
    )
    .unwrap();
    // parent_base appears unconditionally; both X-gated rules survive because
    // child's enabled overrode parent's disabled in the global active map.
    assert_eq!(
        rule_names(&g),
        vec!["parent_base", "parent_only", "program", "child_only"]
    );
}

#[test]
fn cfg_three_level_inheritance_root_flag_wins() {
    // Across a 3-level chain, first-write-wins means root's X=disabled beats the
    // ancestors' X=enabled, dropping every gated rule.
    let g = parse_with_modules(
        &[
            (
                "grandparent.tsg",
                r#"
            grammar { language: "g", flags: { enabled: ["X"] } }
            rule grandparent_base { "g" }
            #[cfg(X)]
            rule grandparent_only { "go" }
        "#,
            ),
            (
                "parent.tsg",
                r#"
                let gbase = inherit("grandparent.tsg")
                grammar {
                    language: "p",
                    flags: { enabled: ["X"] },
                    inherits: gbase,
                }
                rule parent_base { "p" }
                #[cfg(X)]
                rule parent_only { "po" }
            "#,
            ),
        ],
        r#"
        let pbase = inherit("parent.tsg")
        grammar {
            language: "t",
            flags: { disabled: ["X"] },
            inherits: pbase,
        }
        rule program { "x" }
        #[cfg(X)]
        rule root_only { "ro" }
        "#,
    )
    .unwrap();
    // X=disabled (root won), so all three `#[cfg(X)]` rules dropped.
    assert_eq!(
        rule_names(&g),
        vec!["grandparent_base", "parent_base", "program"]
    );
}

#[test]
fn wrapper_overrides_base_extension_flags() {
    // The markdown wrapper pattern: a base grammar bakes a default extension
    // config (TABLE on, TAGS off) gating rules; a thin wrapper inherits it and
    // flips both flags while declaring no rules of its own. The inherited rules
    // must re-gate against the wrapper's flags (now-enabled rule appears,
    // now-disabled drops) and a rules-less flag-only wrapper must be valid.
    let g = parse_with_modules(
        &[(
            "base.tsg",
            r#"
            grammar { language: "md", flags: { enabled: ["TABLE"], disabled: ["TAGS"] } }
            rule document { "x" }
            #[cfg(TABLE)] rule pipe_table { "|" }
            #[cfg(TAGS)] rule tag { "<" }
        "#,
        )],
        r#"
        let base = inherit("base.tsg")
        grammar {
            language: "md",
            inherits: base,
            flags: { enabled: ["TAGS"], disabled: ["TABLE"] },
        }
        "#,
    )
    .unwrap();
    assert_eq!(rule_names(&g), vec!["document", "tag"]);
}

#[test]
fn cfg_disabled_nested_import_does_not_load() {
    // A cfg-disabled import nested in a list (not a top-level let) must not load
    // (module_refs was once rebuilt only when a top-level item dropped).
    let g = dsl(r#"
        grammar {
            language: "t",
            flags: { disabled: ["EXT"] },
            externals: [ #[cfg(EXT)] import("does-not-exist.tsg") ],
        }
        rule program { "x" }
    "#);
    assert!(g.variables.iter().any(|v| v.name == "program"));
}

#[test]
fn cfg_disabled_nested_import_does_not_merge_rules() {
    // The silent-miscompilation form: a cfg-disabled nested import of a real
    // helper must not merge that helper's rules into the grammar.
    let g = parse_with_modules(
        &[("helper.tsg", "rule helper_only { \"h\" }\n")],
        r#"
        grammar {
            language: "t",
            flags: { disabled: ["EXT"] },
            extras: [ #[cfg(EXT)] import("helper.tsg") ],
        }
        rule program { "x" }
    "#,
    )
    .unwrap();
    assert!(
        !g.variables.iter().any(|v| v.name == "helper_only"),
        "cfg-disabled import's rule leaked into the grammar"
    );
}

#[test]
fn cfg_disabled_symref_not_validated() {
    // A cfg-disabled `@<expr>` ref in a rule-set body must not be evaluated
    // (MacroConfig.sym_refs once kept it, so resolve flagged a nonexistent rule).
    let g = dsl(r#"
        rules m(s: str_t) {
            rule program { choice(#[cfg(X)] @concat("missing_", s), "ok") }
        }
        grammar { language: "t", flags: { disabled: ["X"] } }
        @m("rule")
    "#);
    assert!(g.variables.iter().any(|v| v.name == "program"));
}

find_rule_tests! {
    cfg_choice_member_enabled {
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { choice("a", #[cfg(GFM)] "b", "c") }
    "#,
        "program",
        Rule::choice(vec![
            Rule::String("a".into()),
            Rule::String("b".into()),
            Rule::String("c".into()),
        ])
    }
    cfg_choice_member_disabled {
        // GFM disabled -> "b" arm dropped from the choice.
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { choice("a", #[cfg(GFM)] "b", "c") }
    "#,
        "program",
        Rule::choice(vec![Rule::String("a".into()), Rule::String("c".into())])
    }
    cfg_concat_member_disabled {
        // walk_children's Concat arm once only recursed without filtering, leaving
        // a stale Node::Cfg for typecheck to panic on.
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { concat("a", #[cfg(GFM)] "b", "c") }
    "#,
        "program",
        Rule::String("ac".into())
    }
    cfg_seq_member_disabled {
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { seq("a", #[cfg(GFM)] "b", "c") }
    "#,
        "program",
        Rule::seq(vec![Rule::String("a".into()), Rule::String("c".into())])
    }
    cfg_nested_both_active {
        r#"
        grammar { language: "t", flags: { enabled: ["A", "B"] } }
        rule program { choice("x", #[cfg(A)] #[cfg(B)] "y", "z") }
    "#,
        "program",
        Rule::choice(vec![
            Rule::String("x".into()),
            Rule::String("y".into()),
            Rule::String("z".into()),
        ])
    }
    cfg_nested_inner_off {
        // Outer A is on but inner B is off -> drop the whole "y" branch.
        r#"
        grammar { language: "t", flags: { enabled: ["A"], disabled: ["B"] } }
        rule program { choice("x", #[cfg(A)] #[cfg(B)] "y", "z") }
    "#,
        "program",
        Rule::choice(vec![Rule::String("x".into()), Rule::String("z".into())])
    }
}

rule_names_tests! {
    cfg_rule_def_enabled {
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
        vec!["program", "strikethrough"]
    }
    cfg_active_let_keeps_type_annotation {
        // Unwrapping an active #[cfg] on an annotated let must carry the type
        // annotation to the unwrapped slot, else an empty-container let wrongly
        // fails with EmptyContainerNeedsAnnotation.
        r#"
        grammar { language: "test", flags: { enabled: ["X"] } }
        #[cfg(X)] let x: list_t<str_t> = []
        rule foo { "x" }
    "#,
        vec!["foo"]
    }
    cfg_enabled_definition_with_expect_compiles {
        // The same grammar with the flag enabled keeps the definition, so the
        // forward-decl is fulfilled and the symbol-completeness check passes.
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        expect strikethrough
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
        vec!["program", "strikethrough"]
    }
    cfg_disabled_expect_and_user_compiles {
        // cfg drops the forward-decl and its only reference together, so the
        // completeness check must not false-positive on the gated-out `expect`.
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        #[cfg(GFM)] expect _ext
        #[cfg(GFM)] rule uses_ext { _ext }
        rule program { "x" }
    "#,
        vec!["program"]
    }
    cfg_rule_in_rule_set_enabled {
        // A #[cfg]-gated rule decl inside a `rules` body is kept when its flag is on.
        r#"
        grammar { language: "test", flags: { enabled: ["X"] } }
        rule program { a }
        rules pair() {
            rule a { "x" }
            #[cfg(X)]
            rule b { "y" }
        }
        @pair()
    "#,
        vec!["program", "a", "b"]
    }
    cfg_rule_in_rule_set_disabled {
        // The gated decl is dropped from the set when its flag is off.
        r#"
        grammar { language: "test", flags: { disabled: ["X"] } }
        rule program { a }
        rules pair() {
            rule a { "x" }
            #[cfg(X)]
            rule b { "y" }
        }
        @pair()
    "#,
        vec!["program", "a"]
    }
    cfg_rule_set_fully_gated_call_is_noop {
        // Every decl gated out: the @call expands to nothing and is dropped rather
        // than erroring (an empty expansion contributes no rules).
        r#"
        grammar { language: "test", flags: { disabled: ["X"] } }
        rule program { "p" }
        rules extras() {
            #[cfg(X)]
            rule a { "x" }
        }
        @extras()
    "#,
        vec!["program"]
    }
}

error_tests! { Resolve {
    cfg_rule_def_disabled {
        // The cfg-dropped rule is gone, so the reference is undefined.
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
        ResolveErrorKind::UnknownIdentifier("strikethrough".into())
    }
    cfg_unknown_flag_errors {
        r#"
        grammar { language: "t", flags: { enabled: ["GFM"] } }
        rule program { choice("a", #[cfg(TYPO)] "b") }
    "#,
        ResolveErrorKind::CfgFlagUnknown("TYPO".into())
    }
    cfg_flags_not_object_errors {
        r#"
        grammar { language: "t", flags: ["X"] }
        rule program { "x" }
    "#,
        ResolveErrorKind::CfgFlagsNotObject
    }
    cfg_flags_not_list_errors {
        r#"
        grammar { language: "t", flags: { enabled: "X" } }
        rule program { "x" }
    "#,
        ResolveErrorKind::CfgFlagsNotList
    }
    cfg_flags_unknown_key_errors {
        r#"
        grammar { language: "t", flags: { active: ["GFM"] } }
        rule program { "x" }
    "#,
        ResolveErrorKind::CfgFlagsUnknownKey("active".into())
    }
    cfg_inside_flags_errors {
        // cfg inside `flags` is nonsensical: flags are read before cfg gating runs.
        r#"
        grammar { language: "t", flags: { enabled: ["X", #[cfg(X)] "FOO"] } }
        rule program { "x" }
    "#,
        ResolveErrorKind::CfgInsideFlags
    }
    cfg_flags_non_string_errors {
        // GFM here is an identifier, not a string literal.
        r#"
        grammar { language: "t", flags: { enabled: [GFM] } }
        rule program { "x" }
    "#,
        ResolveErrorKind::CfgFlagsNonLiteral
    }
    cfg_rule_in_rule_set_dropped_reference_is_undefined {
        // Referencing a rule gated out of the set is undefined, exactly as for a
        // gated-out top-level rule.
        r#"
        grammar { language: "test", flags: { disabled: ["X"] } }
        rules pair() {
            rule a { b }
            #[cfg(X)]
            rule b { "y" }
        }
        @pair()
    "#,
        ResolveErrorKind::UnknownIdentifier("b".into())
    }
}}

error_tests! { Type {
    cfg_flags_duplicate_key_errors {
        // Duplicate keys in the flags object error like any other object
        // literal (flags bypasses the generic object typecheck).
        r#"
        grammar { language: "t", flags: { enabled: ["A"], enabled: ["B"] } }
        rule program { "x" }
    "#,
        TypeErrorKind::DuplicateObjectKey("enabled".into())
    }
}}

error_tests! { match Lower {
    cfg_disabled_definition_with_expect_is_undefined_symbol {
        // An `expect` keeps the symbol past resolve; cfg then drops its only
        // definition, so the completeness check catches the dangling symbol at lower.
        r#"
        grammar { language: "t", flags: { disabled: ["GFM"] } }
        expect strikethrough
        rule program { strikethrough }
        #[cfg(GFM)]
        rule strikethrough { "~~" }
    "#,
        LowerErrorKind::UndefinedSymbols(names) if *names == ["strikethrough"]
    }
}}

error_tests! { Parse {
    cfg_in_macro_body_bare_expression_parse_error {
        // A macro body is a single expression; cfg in expression position is a
        // parse error (parse_expr bails on `#`).
        r#"
        grammar { language: "t", flags: { enabled: ["X"] } }
        macro m() rule_t { #[cfg(X)] "a" }
        rule program { m() }
    "#,
        ParseErrorKind::ExpectedExpression
    }
    cfg_in_for_loop_body_parse_error {
        // A for-loop body is expression-shaped, so cfg there is rejected like any
        // other expression-position cfg.
        r#"
        grammar { language: "t", flags: { enabled: ["X"] } }
        rule program { for (x: str_t) in ["a", "b"] { #[cfg(X)] x } }
    "#,
        ParseErrorKind::ExpectedExpression
    }
    cfg_on_grammar_block_is_rejected {
        // Gating the grammar block is incoherent (its own `flags` would declare the
        // gating flag), so the parser rejects it.
        r#"
        #[cfg(X)] grammar { language: "t", flags: { enabled: ["X"] } }
        rule program { "x" }
    "#,
        ParseErrorKind::CfgOnGrammarBlock
    }
}}
