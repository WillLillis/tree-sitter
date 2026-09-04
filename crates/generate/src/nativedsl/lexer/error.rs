//! Lexer error taxonomy.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::nativedsl::LexError;

pub type LexResult<T> = Result<T, LexError>;

/// The specific kind of lexer error.
#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum LexErrorKind {
    #[error("unterminated string literal")]
    UnterminatedString,
    #[error("unterminated raw string literal")]
    UnterminatedRawString,
    #[error("unterminated escape sequence")]
    UnterminatedEscape,
    #[error("invalid escape sequence: \\{}", DiagnosticChar(*.0))]
    InvalidEscape(char),
    #[error("invalid hex escape: expected \\xHH with HH in 00..7F (ASCII range)")]
    InvalidHexEscape,
    #[error(
        "invalid unicode escape: expected \\uHHHH or \\u{{H..H}} (max 0x10FFFF, no surrogates)"
    )]
    InvalidUnicodeEscape,
    #[error("unexpected character: {}", DiagnosticChar(*.0))]
    UnexpectedChar(char),
    #[error("unterminated string literal (newline before closing quote)")]
    NewlineInString,
    #[error("expected '\"' after 'r' and '#' delimiters")]
    ExpectedRawStringQuote,
    #[error("raw string has {0} '#' delimiters (maximum {max})", max = u8::MAX)]
    TooManyHashes(u32),
    #[error("input exceeds the maximum size ({} bytes)", u32::MAX)]
    InputTooLarge,
}

struct DiagnosticChar(char);

impl std::fmt::Display for DiagnosticChar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let displayed = self.0.escape_debug();
        let escaped = self.0.escape_default();

        if displayed.clone().eq(escaped.clone()) {
            write!(f, "{displayed}")
        } else {
            write!(f, "{displayed} ({escaped})")
        }
    }
}
