use super::super::*;

error_tests! { Lex {
    error_unterminated_string {
        r#"grammar { language: "test }rule program { "x" }"#,
        LexErrorKind::UnterminatedString
    }
    error_newline_in_string {
        "grammar { language: \"test\n\" }",
        LexErrorKind::NewlineInString
    }
    error_invalid_escape {
        r#"grammar { language: "te\qst" } rule program { "x" }"#,
        LexErrorKind::InvalidEscape('q')
    }
    error_unterminated_escape {
        r#"grammar { language: "test\"#,
        LexErrorKind::UnterminatedEscape
    }
    error_unterminated_raw_string {
        r#"grammar { language: r#"test } rule program { "x" }"#,
        LexErrorKind::UnterminatedRawString
    }
    error_expected_raw_string_quote {
        r#"grammar { language: r## } rule program { "x" }"#,
        LexErrorKind::ExpectedRawStringQuote
    }
    error_integer_overflow {
        r#"grammar { language: "test" } rule program { prec(99999999999, "x") }"#,
        LexErrorKind::IntegerOverflow
    }
    error_too_many_raw_string_hashes {
        &format!("let x = r{}\"test\"", "#".repeat(256)),
        LexErrorKind::TooManyHashes
    }
    error_unexpected_char {
        r#"grammar { language: "test" } rule program { $ }"#,
        LexErrorKind::UnexpectedChar('$')
    }
    error_non_ascii_latin {
        r#"grammar { language: "test" } rule café { "x" }"#,
        LexErrorKind::UnexpectedChar('é')
    }
    error_non_ascii_emoji {
        r#"grammar { language: "test" } rule 🎉 { "x" }"#,
        LexErrorKind::UnexpectedChar('🎉')
    }
    error_non_ascii_cjk {
        r#"grammar { language: "test" } rule 名前 { "x" }"#,
        LexErrorKind::UnexpectedChar('名')
    }
    error_non_ascii_after_ident {
        r#"grammar { language: "test" } rule foo™ { "x" }"#,
        LexErrorKind::UnexpectedChar('™')
    }
    error_non_ascii_escape {
        "grammar { language: \"te\\ést\" } rule program { \"x\" }",
        LexErrorKind::InvalidEscape('é')
    }
}}

#[test]
fn error_inherited_lex_error() {
    let (err, base_path) = inherit_err(r#"grammar { language: "base"#);
    let DslError::Module(m) = &err else {
        panic!("expected Module, got {err:?}")
    };
    let DslError::Lex(e) = m.inner.as_ref() else {
        panic!("expected Lex, got {:?}", m.inner)
    };
    assert_eq!(e.kind, LexErrorKind::UnterminatedString);
    assert_eq!(m.path, base_path);
}
