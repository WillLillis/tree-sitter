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
    #[error("invalid escape sequence: \\{0}")]
    InvalidEscape(char),
    #[error("invalid hex escape: expected \\xHH with HH in 00..7F (ASCII range)")]
    InvalidHexEscape,
    #[error(
        "invalid unicode escape: expected \\uHHHH or \\u{{H..H}} (max 0x10FFFF, no surrogates)"
    )]
    InvalidUnicodeEscape,
    #[error("unexpected character: {0}")]
    UnexpectedChar(char),
    #[error("unterminated string literal (newline before closing quote)")]
    NewlineInString,
    #[error("expected '\"' after 'r' and '#' delimiters")]
    ExpectedRawStringQuote,
    #[error("integer literal out of range (maximum {})", u32::MAX)]
    IntegerOverflow,
    #[error("raw string has {0} '#' delimiters (maximum 255)")]
    TooManyHashes(u32),
    #[error("input exceeds the maximum size ({} bytes)", u32::MAX)]
    InputTooLarge,
}
