use super::super::*;

#[test]
fn error_unterminated_string() {
    let err = dsl_err(r#"grammar { language: "test }rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnterminatedString);
}

#[test]
fn error_newline_in_string() {
    let err = dsl_err("grammar { language: \"test\n\" }");
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::NewlineInString);
}

#[test]
fn error_invalid_escape() {
    let err = dsl_err(r#"grammar { language: "te\qst" } rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::InvalidEscape('q'));
}

#[test]
fn error_unterminated_escape() {
    let err = dsl_err(r#"grammar { language: "test\"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnterminatedEscape);
}

#[test]
fn error_unterminated_raw_string() {
    let err = dsl_err(r#"grammar { language: r#"test } rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnterminatedRawString);
}

#[test]
fn error_expected_raw_string_quote() {
    let err = dsl_err(r#"grammar { language: r## } rule program { "x" }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::ExpectedRawStringQuote);
}

#[test]
fn error_integer_overflow() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { prec(99999999999, "x") }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::IntegerOverflow);
}

#[test]
fn error_too_many_raw_string_hashes() {
    let src = format!("let x = r{}\"test\"", "#".repeat(256));
    let err = dsl_err(&src);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::TooManyHashes);
}

#[test]
fn error_inherited_lex_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    // File ends mid-string: no closing quote and no newline, so the lexer
    // hits EOF and produces UnterminatedString.
    std::fs::write(&base_path, r#"grammar { language: "base"#).unwrap();

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
    let DslError::Lex(lex_err) = inherited.inner.as_ref() else {
        panic!("expected Lex error, got {:?}", inherited.inner)
    };
    assert_eq!(lex_err.kind, LexErrorKind::UnterminatedString);
    assert_eq!(inherited.path, base_path);
    assert_eq!(
        &parent_src[inherited.inherit_span.start as usize..inherited.inherit_span.end as usize],
        "base.tsg"
    );
}

#[test]
fn error_unexpected_char() {
    let err = dsl_err(r#"grammar { language: "test" } rule program { $ }"#);
    let e = assert_err!(err, Lex);
    assert_eq!(e.kind, LexErrorKind::UnexpectedChar('$'));
}
