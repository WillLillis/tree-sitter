use super::super::*;

#[test]
fn error_unknown_grammar_field() {
    let err = dsl_err(
        r#"
        grammar { language: "test", bogus: "x" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::UnknownGrammarField("bogus".into()));
}

#[test]
fn error_expected_token() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { seq("a" "b") }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(
        e.kind,
        ParseErrorKind::ExpectedToken {
            expected: TokenKind::RParen,
            got: TokenKind::StringLit
        }
    );
}

#[test]
fn error_expected_expression() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { seq(,) }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedExpression);
}

#[test]
fn error_expected_item() {
    let err = dsl_err(r#"grammar { language: "test" } "stray_string""#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedItem);
}

#[test]
fn error_unknown_type() {
    let err = dsl_err(r#"grammar { language: "test" } let x: foo = "y" rule program { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::UnknownType("foo".into()));
}

#[test]
fn error_missing_return_type() {
    let err = dsl_err(r#"grammar { language: "test" } fn f(x: rule_t) { x } rule program { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::MissingReturnType);
}

#[test]
fn error_builtin_arg_count_messages() {
    macro_rules! g {
        ($expr:expr) => {
            concat!(r#"grammar { language: "test" } rule foo { "#, $expr, " }")
        };
    }
    let cases: &[(&str, &str, u8, usize)] = &[
        // 0-arg: blank
        (g!("blank(\"a\")"), "blank", 0, 1),
        (g!("blank(\"a\", \"b\")"), "blank", 0, 2),
        // 1-arg: too few
        (g!("repeat()"), "repeat", 1, 0),
        (g!("repeat1()"), "repeat1", 1, 0),
        (g!("optional()"), "optional", 1, 0),
        (g!("token()"), "token", 1, 0),
        (g!("token_immediate()"), "token_immediate", 1, 0),
        (g!("inherit()"), "inherit", 1, 0),
        // 1-arg: too many
        (g!(r#"repeat("a", "b")"#), "repeat", 1, 2),
        (g!(r#"repeat1("a", "b")"#), "repeat1", 1, 2),
        (g!(r#"optional("a", "b")"#), "optional", 1, 2),
        (g!(r#"token("a", "b")"#), "token", 1, 2),
        (g!(r#"token_immediate("a", "b")"#), "token_immediate", 1, 2),
        (g!(r#"inherit("a", "b")"#), "inherit", 1, 2),
        // 2-arg: too few
        (g!("prec()"), "prec", 2, 0),
        (g!("prec(1)"), "prec", 2, 1),
        (g!("prec_left()"), "prec_left", 2, 0),
        (g!("prec_left(1)"), "prec_left", 2, 1),
        (g!("prec_right()"), "prec_right", 2, 0),
        (g!("prec_right(1)"), "prec_right", 2, 1),
        (g!("prec_dynamic()"), "prec_dynamic", 2, 0),
        (g!("prec_dynamic(1)"), "prec_dynamic", 2, 1),
        (g!("field()"), "field", 2, 0),
        (g!("field(name)"), "field", 2, 1),
        (g!("alias()"), "alias", 2, 0),
        (g!(r#"alias("a")"#), "alias", 2, 1),
        (g!("append()"), "append", 2, 0),
        (g!(r#"append("a")"#), "append", 2, 1),
        (g!("reserved()"), "reserved", 2, 0),
        (g!(r#"reserved("ctx")"#), "reserved", 2, 1),
        // 2-arg: too many
        (g!(r#"prec(1, "a", "b")"#), "prec", 2, 3),
        (g!(r#"prec_left(1, "a", "b")"#), "prec_left", 2, 3),
        (g!(r#"prec_right(1, "a", "b")"#), "prec_right", 2, 3),
        (g!(r#"prec_dynamic(1, "a", "b")"#), "prec_dynamic", 2, 3),
        (g!(r#"field(name, "a", "b")"#), "field", 2, 3),
        (g!(r#"alias("a", "b", "c")"#), "alias", 2, 3),
        (g!(r#"append([1], [2], [3])"#), "append", 2, 3),
        (g!(r#"reserved("ctx", "a", "b")"#), "reserved", 2, 3),
        // regexp (1-2 args)
        (g!("regexp()"), "regexp", 1, 0),
        (g!(r#"regexp("a", "b", "c")"#), "regexp", 2, 3),
    ];
    for &(src, name, expected, got) in cases {
        let err = dsl_err(src);
        let e = assert_err!(err, Parse);
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
    let cases = [
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
    ];
    for (src, expected_msg) in cases {
        let err = dsl_err(src);
        let e = assert_err!(err, Parse);
        assert_eq!(e.to_string(), expected_msg, "wrong message for: {src}");
    }
}

#[test]
fn keywords_as_identifiers() {
    // Keywords are contextual - they are valid as rule/let/fn names.
    let g = dsl(r#"grammar { language: "test" } rule seq { "x" }"#);
    assert_eq!(g.variables[0].name, "seq");
    let g = dsl(r#"grammar { language: "test" } let token = "x" rule foo { token }"#);
    assert_eq!(g.variables[0].name, "foo");
    let g = dsl(
        r#"grammar { language: "test" } fn repeat(x: rule_t) -> rule_t { x } rule foo { repeat("a") }"#,
    );
    assert_eq!(g.variables[0].name, "foo");
}

#[test]
fn error_multiple_grammar_blocks() {
    let err = dsl_err(
        r#"
        grammar { language: "first" }
        grammar { language: "second" }
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::DuplicateGrammarBlock);
}

#[test]
fn error_duplicate_grammar_field() {
    let err = dsl_err(
        r#"
        grammar { language: "test", word: foo, word: bar }
        rule foo { "x" }
        rule bar { "y" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::DuplicateGrammarField("word".into()));
    assert!(e.note.is_some());
}

#[test]
fn error_duplicate_object_key() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1, b: 2, a: 3 }
        rule foo { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::DuplicateObjectKey("a".into()));
    assert!(e.note.is_some());
}

#[test]
fn error_nesting_too_deep() {
    let deep = "(-".repeat(300) + "1" + &")".repeat(300);
    let src = format!("grammar {{ language: \"test\" }} rule foo {{ {deep} }}");
    let err = dsl_err(&src);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::NestingTooDeep);
}

#[test]
fn error_print_as_expression_rejected() {
    // print() in expression position parses successfully (contextual keywords)
    // but is rejected by the typechecker since it returns void_t.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        rule program { seq(print("hi"), "x") }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(
        e.kind,
        TypeErrorKind::TypeMismatch {
            expected: Ty::Rule,
            got: Ty::Void
        }
    );
}

#[test]
fn error_let_rhs_print_rejected() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = print("hi")
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Type);
    assert_eq!(e.kind, TypeErrorKind::CannotBindVoid);
}

#[test]
fn error_expected_function_name() {
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x = { a: 1 }
        rule program { x.a("y") }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedFunctionName);
}

#[test]
fn error_expected_ident() {
    // Missing rule name after `rule` keyword.
    let err = dsl_err(r#"grammar { language: "test" } rule { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedIdent);
}

#[test]
fn error_expected_string() {
    // `language` expects a string literal, not an integer.
    let err = dsl_err(r#"grammar { language: 123 } rule program { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedString);
}

#[test]
fn error_expected_name() {
    // Grammar field key must be a name, not a colon.
    let err = dsl_err(r#"grammar { : "test" } rule program { "x" }"#);
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedName);
}

#[test]
fn error_expected_type() {
    // Type annotation position has an `=` instead of a type name.
    let err = dsl_err(
        r#"
        grammar { language: "test" }
        let x: = "y"
        rule program { "x" }
    "#,
    );
    let e = assert_err!(err, Parse);
    assert_eq!(e.kind, ParseErrorKind::ExpectedType);
}

#[test]
fn error_inherited_parse_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(
        &base_path,
        r#"grammar { language: "base" }
rule program { seq("a" "b") }
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

    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    let DslError::Parse(parse_err) = inherited.inner.as_ref() else {
        panic!("expected Parse error, got {:?}", inherited.inner)
    };
    assert_eq!(
        parse_err.kind,
        ParseErrorKind::ExpectedToken {
            expected: TokenKind::RParen,
            got: TokenKind::StringLit
        }
    );
    assert_eq!(inherited.path, base_path);
    assert_eq!(
        &parent_src[inherited.reference_span.start as usize..inherited.reference_span.end as usize],
        "base.tsg"
    );
}
