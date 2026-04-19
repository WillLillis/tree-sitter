use super::super::*;

macro_rules! lex_error_tests {
    ($($name:ident { $input:expr, $expected:expr })*) => {
        $(#[test] fn $name() {
            let e = assert_err!(dsl_err($input), Lex);
            assert_eq!(e.kind, $expected);
        })*
    };
}

lex_error_tests! {
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
}

#[test]
fn error_inherited_lex_error() {
    let dir = tempfile::tempdir().unwrap();
    let base_path = dir.path().join("base.tsg");
    std::fs::write(&base_path, r#"grammar { language: "base"#).unwrap();
    let parent_path = dir.path().join("parent.tsg");
    let parent_src = "let base = inherit(\"base.tsg\")\ngrammar { language: \"derived\", inherits: base }\nrule extra { \"hello\" }\n".to_string();
    let err = parse_native_dsl(&parent_src, &parent_path).unwrap_err();
    let DslError::Module(inherited) = &err else {
        panic!("expected Module error, got {err:?}")
    };
    let DslError::Lex(lex_err) = inherited.inner.as_ref() else {
        panic!("expected Lex error, got {:?}", inherited.inner)
    };
    assert_eq!(lex_err.kind, LexErrorKind::UnterminatedString);
    assert_eq!(inherited.path, base_path);
}
