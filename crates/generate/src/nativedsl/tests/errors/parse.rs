use super::super::*;

error_tests! { Parse {
    error_unknown_grammar_field {
        r#"grammar { language: "test", bogus: "x" } rule program { "x" }"#,
        ParseErrorKind::UnknownGrammarField("bogus".into())
    }
    error_expected_token {
        r#"grammar { language: "test" } rule program { seq("a" "b") }"#,
        ParseErrorKind::ExpectedToken { expected: TokenKind::RParen, got: TokenKind::StringLit }
    }
    error_expected_expression {
        r#"grammar { language: "test" } rule program { seq(,) }"#,
        ParseErrorKind::ExpectedExpression
    }
    error_expected_item {
        r#"grammar { language: "test" } "stray_string""#,
        ParseErrorKind::ExpectedItem
    }
    error_unknown_type {
        r#"grammar { language: "test" } let x: foo = "y" rule program { "x" }"#,
        ParseErrorKind::UnknownType("foo".into())
    }
    error_spread_t_not_allowed {
        r#"grammar { language: "test" } let x: spread_t = "y" rule program { "x" }"#,
        ParseErrorKind::InternalTypeNotAllowed(Ty::Spread)
    }
    error_list_of_module_rejected {
        r#"grammar { language: "test" } let x: list_t<module_t> = [] rule program { "x" }"#,
        ParseErrorKind::ListInnerType(Ty::AnyModule)
    }
    error_obj_of_module_rejected {
        r#"grammar { language: "test" } let x: obj_t<module_t> = { a: 1 } rule program { "x" }"#,
        ParseErrorKind::ObjectInnerType(Ty::AnyModule)
    }
    error_list_triple_nesting_rejected {
        r#"grammar { language: "test" } let x: list_t<list_t<list_t<rule_t>>> = [] rule program { "x" }"#,
        ParseErrorKind::ListInnerType(Ty::ListListRule)
    }
    error_missing_return_type {
        r#"grammar { language: "test" } macro f(x: rule_t) = x rule program { "x" }"#,
        ParseErrorKind::ExpectedType
    }
    error_missing_language_field {
        r#"grammar { extras: [] } rule program { "x" }"#,
        ParseErrorKind::MissingLanguageField
    }
    error_multiple_grammar_blocks {
        r#"grammar { language: "first" }
        grammar { language: "second" }
        rule program { "x" }"#,
        ParseErrorKind::DuplicateGrammarBlock
    }
    error_expected_function_name {
        r#"grammar { language: "test" }
        let x = { a: 1 }
        rule program { x.a("y") }"#,
        ParseErrorKind::ExpectedFunctionName
    }
    error_expected_ident {
        r#"grammar { language: "test" } rule { "x" }"#,
        ParseErrorKind::ExpectedIdent
    }
    error_expected_string {
        r#"grammar { language: 123 } rule program { "x" }"#,
        ParseErrorKind::ExpectedString
    }
    error_expected_name {
        r#"grammar { : "test" } rule program { "x" }"#,
        ParseErrorKind::ExpectedName
    }
    error_expected_type {
        r#"grammar { language: "test" }
        let x: = "y"
        rule program { "x" }"#,
        ParseErrorKind::ExpectedType
    }
    error_truncated_rule_body {
        r#"grammar { language: "test" } rule program {"#,
        ParseErrorKind::ExpectedExpression
    }
    error_nesting_too_deep {
        // Constructed inline since it needs dynamic depth
        &{
            let deep = "(-".repeat(300) + "1" + &")".repeat(300);
            format!("grammar {{ language: \"test\" }} rule foo {{ {deep} }}")
        },
        ParseErrorKind::NestingTooDeep
    }
    error_too_many_children {
        &{
            let args = vec!["\"x\""; 65_536].join(",");
            format!("grammar {{ language: \"test\" }} rule foo {{ seq({args}) }}")
        },
        ParseErrorKind::TooManyChildren
    }
    error_grammar_config_unknown_field {
        r#"let base = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: base, extras: grammar_config(base, bogus) }"#,
        ParseErrorKind::UnknownGrammarField("bogus".into())
    }
}}

#[test]
fn error_duplicates_have_notes() {
    for (src, expected) in [
        (
            r#"grammar { language: "test", word: foo, word: bar } rule foo { "x" } rule bar { "y" }"#,
            ParseErrorKind::DuplicateGrammarField("word".into()),
        ),
        (
            r#"grammar { language: "test" } let x = { a: 1, b: 2, a: 3 } rule foo { "x" }"#,
            ParseErrorKind::DuplicateObjectKey("a".into()),
        ),
    ] {
        let e = assert_err!(dsl_err(src), Parse);
        assert_eq!(e.kind, expected);
        assert!(e.note.is_some());
    }
}

#[test]
fn error_builtin_arg_count_messages() {
    macro_rules! g {
        ($expr:expr) => {
            concat!(r#"grammar { language: "test" } rule foo { "#, $expr, " }")
        };
    }
    #[rustfmt::skip]
    let cases: &[(&str, TokenKind, u8, usize)] = &[
        (g!("blank(\"a\")"), TokenKind::KwBlank, 0, 1),
        (g!("blank(\"a\",)"), TokenKind::KwBlank, 0, 1),
        (g!("blank(\"a\", \"b\")"), TokenKind::KwBlank, 0, 2),
        (g!("repeat()"), TokenKind::KwRepeat, 1, 0),
        (g!("repeat1()"), TokenKind::KwRepeat1, 1, 0),
        (g!("optional()"), TokenKind::KwOptional, 1, 0),
        (g!("token()"), TokenKind::KwToken, 1, 0),
        (g!("token_immediate()"), TokenKind::KwTokenImmediate, 1, 0),
        (g!("inherit()"), TokenKind::KwInherit, 1, 0),
        (g!(r#"repeat("a", "b")"#), TokenKind::KwRepeat, 1, 2),
        (g!(r#"repeat1("a", "b")"#), TokenKind::KwRepeat1, 1, 2),
        (g!(r#"optional("a", "b")"#), TokenKind::KwOptional, 1, 2),
        (g!(r#"token("a", "b")"#), TokenKind::KwToken, 1, 2),
        (g!(r#"token_immediate("a", "b")"#), TokenKind::KwTokenImmediate, 1, 2),
        (g!(r#"inherit("a", "b")"#), TokenKind::KwInherit, 1, 2),
        (g!("prec()"), TokenKind::KwPrec, 2, 0),
        (g!("prec(1)"), TokenKind::KwPrec, 2, 1),
        (g!("prec_left()"), TokenKind::KwPrecLeft, 2, 0),
        (g!("prec_left(1)"), TokenKind::KwPrecLeft, 2, 1),
        (g!("prec_right()"), TokenKind::KwPrecRight, 2, 0),
        (g!("prec_right(1)"), TokenKind::KwPrecRight, 2, 1),
        (g!("prec_dynamic()"), TokenKind::KwPrecDynamic, 2, 0),
        (g!("prec_dynamic(1)"), TokenKind::KwPrecDynamic, 2, 1),
        (g!("field()"), TokenKind::KwField, 2, 0),
        (g!("field(name)"), TokenKind::KwField, 2, 1),
        (g!("alias()"), TokenKind::KwAlias, 2, 0),
        (g!(r#"alias("a")"#), TokenKind::KwAlias, 2, 1),
        (g!("append()"), TokenKind::KwAppend, 2, 0),
        (g!(r#"append("a")"#), TokenKind::KwAppend, 2, 1),
        (g!("reserved()"), TokenKind::KwReserved, 2, 0),
        (g!(r#"reserved("ctx")"#), TokenKind::KwReserved, 2, 1),
        (g!(r#"prec(1, "a", "b")"#), TokenKind::KwPrec, 2, 3),
        (g!(r#"prec_left(1, "a", "b")"#), TokenKind::KwPrecLeft, 2, 3),
        (g!(r#"prec_right(1, "a", "b")"#), TokenKind::KwPrecRight, 2, 3),
        (g!(r#"prec_dynamic(1, "a", "b")"#), TokenKind::KwPrecDynamic, 2, 3),
        (g!(r#"field(name, "a", "b")"#), TokenKind::KwField, 2, 3),
        (g!(r#"alias("a", "b", "c")"#), TokenKind::KwAlias, 2, 3),
        (g!(r#"append([1], [2], [3])"#), TokenKind::KwAppend, 2, 3),
        (g!(r#"reserved("ctx", "a", "b")"#), TokenKind::KwReserved, 2, 3),
        (g!("regexp()"), TokenKind::KwRegexp, 1, 0),
        (g!(r#"regexp("a", "b", "c")"#), TokenKind::KwRegexp, 2, 3),
    ];
    for &(src, name, expected, got) in cases {
        let e = assert_err!(dsl_err(src), Parse);
        assert_eq!(
            e.kind,
            ParseErrorKind::WrongArgumentCount {
                name,
                expected,
                got
            },
            "wrong error for: {src}"
        );
    }
}

#[test]
fn error_builtin_arg_count_display() {
    for (src, expected_msg) in [
        (
            r#"grammar { language: "test" } rule foo { prec(1) }"#,
            "'prec' takes 2 arguments, got 1",
        ),
        (
            r#"grammar { language: "test" } rule foo { repeat() }"#,
            "'repeat' takes 1 argument, got 0",
        ),
        (
            r#"grammar { language: "test" } rule foo { blank("x") }"#,
            "'blank' takes no arguments, got 1",
        ),
    ] {
        let e = assert_err!(dsl_err(src), Parse);
        assert_eq!(e.to_string(), expected_msg, "wrong message for: {src}");
    }
}

#[test]
fn keywords_as_identifiers() {
    let g = dsl(r#"grammar { language: "test" } rule seq { "x" }"#);
    assert_eq!(g.variables[0].name, "seq");
    let g = dsl(r#"grammar { language: "test" } let token = "x" rule foo { token }"#);
    assert_eq!(g.variables[0].name, "foo");
    let g = dsl(
        r#"grammar { language: "test" } macro repeat(x: rule_t) rule_t = x rule foo { repeat("a") }"#,
    );
    assert_eq!(g.variables[0].name, "foo");
}

#[test]
fn error_duplicate_fn_param() {
    let e = assert_err!(
        dsl_err(
            r#"grammar { language: "test" }
            macro f(x: rule_t, x: str_t) rule_t = x
            rule program { f("a") }"#,
        ),
        Parse
    );
    assert_eq!(e.kind, ParseErrorKind::DuplicateParameter("x".into()));
}

#[test]
fn error_duplicate_for_binding() {
    let e = assert_err!(
        dsl_err(
            r#"grammar { language: "test" }
            rule program { choice(for (x: str_t, x: int_t) in [("a", 1)] { x }) }"#,
        ),
        Parse
    );
    assert_eq!(e.kind, ParseErrorKind::DuplicateParameter("x".into()));
}

inherit_error_tests! { Parse {
    error_inherited_parse_error {
        "grammar { language: \"base\" }\nrule program { seq(\"a\" \"b\") }\n",
        ParseErrorKind::ExpectedToken { expected: TokenKind::RParen, got: TokenKind::StringLit }
    }
    error_inherited_missing_language {
        "grammar { extras: [] }\nrule program { \"x\" }\n",
        ParseErrorKind::MissingLanguageField
    }
    multiple_inherit_bindings_rejected {
        r#"let a = inherit("inherit_base/grammar.tsg")
        let b = inherit("inherit_base/grammar.tsg")
        grammar { language: "derived", inherits: a }
        rule extra { "x" }"#,
        ParseErrorKind::MultipleInherits
    }
}}
