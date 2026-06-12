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
    error_carriage_return_in_string {
        "grammar { language: \"test\r\" }",
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
    error_hex_escape_too_short {
        r#"grammar { language: "x\x0" } rule program { "x" }"#,
        LexErrorKind::InvalidHexEscape
    }
    error_hex_escape_non_hex {
        r#"grammar { language: "x\xZZ" } rule program { "x" }"#,
        LexErrorKind::InvalidHexEscape
    }
    // A multibyte char where a hex digit is expected: must error cleanly, not
    // slice mid-char (which was UB via from_utf8_unchecked + a panicking span).
    error_hex_escape_multibyte {
        "grammar { language: \"x\\x)\u{3a3}\" } rule program { \"x\" }",
        LexErrorKind::InvalidHexEscape
    }
    error_hex_escape_out_of_ascii_range {
        r#"grammar { language: "x\x80" } rule program { "x" }"#,
        LexErrorKind::InvalidHexEscape
    }
    error_unicode_escape_too_short {
        r#"grammar { language: "x\u123" } rule program { "x" }"#,
        LexErrorKind::InvalidUnicodeEscape
    }
    error_unicode_escape_multibyte {
        "grammar { language: \"x\\u00\u{3a3}\" } rule program { \"x\" }",
        LexErrorKind::InvalidUnicodeEscape
    }
    error_unicode_escape_braced_empty {
        r#"grammar { language: "x\u{}" } rule program { "x" }"#,
        LexErrorKind::InvalidUnicodeEscape
    }
    error_unicode_escape_braced_unclosed {
        r#"grammar { language: "x\u{1234" } rule program { "x" }"#,
        LexErrorKind::InvalidUnicodeEscape
    }
    error_unicode_escape_surrogate {
        r#"grammar { language: "x\uD800" } rule program { "x" }"#,
        LexErrorKind::InvalidUnicodeEscape
    }
    error_unicode_escape_out_of_range {
        r#"grammar { language: "x\u{110000}" } rule program { "x" }"#,
        LexErrorKind::InvalidUnicodeEscape
    }
}}

inherit_error_tests! { Lex {
    error_inherited_lex_error {
        r#"grammar { language: "base"#,
        LexErrorKind::UnterminatedString
    }
}}
