//! Zero-allocation lexer for the native grammar DSL.
//!
//! Operates directly on the source bytes (`&[u8]`) and produces a flat
//! `Vec<Token>` with spans pointing back into the source. Uses `memchr` /
//! `memchr2` for fast scanning of string literals and comments.
//!
//! Escape sequences inside string literals (`"..."`) are validated inline
//! during lexing so that invalid escapes are caught early. Raw string
//! literals (`r"..."`, `r#"..."#`) are scanned without escape processing.

use memchr::{memchr, memchr2};
use serde::Serialize;
use thiserror::Error;

use super::ast::{FileId, FileSpan, Span};

/// The kind of a lexer token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum TokenKind {
    // Identifiers and literals
    Ident,
    /// String literal. Token span covers `"..."` including quotes.
    StringLit,
    /// Raw string literal. Token span covers `r"..."`, `r#"..."#`, etc.
    /// `hash_count` is the number of `#` delimiters.
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i32),

    // Structural keywords
    KwGrammar,
    KwRule,
    KwLet,
    KwFn,
    KwFor,
    KwIn,

    // Combinator keywords
    KwSeq,
    KwChoice,
    KwRepeat,
    KwRepeat1,
    KwOptional,
    KwBlank,
    KwField,
    KwAlias,
    KwToken,
    KwPrec,
    KwReserved,
    KwTokenImmediate,
    KwConcat,
    KwRegexp,
    KwInherit,
    KwOverride,
    KwAppend,

    // Punctuation
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    ColonColon,
    Dot,
    Arrow, // ->
    Minus,
    Eq,
    Lt,
    Gt,

    Eof,
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Ident => "identifier",
            Self::StringLit => "string literal",
            Self::RawStringLit { .. } => "raw string literal",
            Self::IntLit(_) => "integer literal",
            Self::KwGrammar => "'grammar'",
            Self::KwRule => "'rule'",
            Self::KwLet => "'let'",
            Self::KwFn => "'fn'",
            Self::KwFor => "'for'",
            Self::KwIn => "'in'",
            Self::KwSeq => "'seq'",
            Self::KwChoice => "'choice'",
            Self::KwRepeat => "'repeat'",
            Self::KwRepeat1 => "'repeat1'",
            Self::KwOptional => "'optional'",
            Self::KwBlank => "'blank'",
            Self::KwField => "'field'",
            Self::KwAlias => "'alias'",
            Self::KwToken => "'token'",
            Self::KwPrec => "'prec'",
            Self::KwReserved => "'reserved'",
            Self::KwTokenImmediate => "'token_immediate'",
            Self::KwConcat => "'concat'",
            Self::KwRegexp => "'regexp'",
            Self::KwInherit => "'inherit'",
            Self::KwOverride => "'override'",
            Self::KwAppend => "'append'",
            Self::LBrace => "'{'",
            Self::RBrace => "'}'",
            Self::LParen => "'('",
            Self::RParen => "')'",
            Self::LBracket => "'['",
            Self::RBracket => "']'",
            Self::Comma => "','",
            Self::Colon => "':'",
            Self::ColonColon => "'::'",
            Self::Dot => "'.'",
            Self::Arrow => "'->'",
            Self::Minus => "'-'",
            Self::Eq => "'='",
            Self::Lt => "'<'",
            Self::Gt => "'>'",
            Self::Eof => "end of file",
        })
    }
}

/// A token produced by the lexer, pairing a kind with its source span.
#[derive(Clone, Debug)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Lexer state: a byte-level cursor over the source text.
pub struct Lexer<'src> {
    source: &'src [u8],
    pos: usize,
    file: FileId,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source text.
    pub const fn new(source: &'src str, file: FileId) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
            file,
        }
    }

    fn file_span(&self, span: Span) -> FileSpan {
        FileSpan {
            file: self.file,
            span,
        }
    }

    /// Consume the entire source and return all tokens.
    ///
    /// The returned vector always ends with a `TokenKind::Eof` token.
    /// Pre-allocates capacity based on source length (roughly 1 token per 4
    /// bytes).
    ///
    /// # Errors
    ///
    /// Returns [`LexError`] on unterminated strings, invalid escapes, or
    /// unexpected characters.
    pub fn tokenize(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens = Vec::with_capacity(self.source.len() / 4);
        loop {
            self.skip_whitespace_and_comments();
            if self.pos >= self.source.len() {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span: Span::from_usize(self.pos, self.pos),
                });
                break;
            }
            tokens.push(self.next_token()?);
        }
        Ok(tokens)
    }

    fn peek(&self) -> Option<u8> {
        self.source.get(self.pos).copied()
    }

    fn advance(&mut self) -> u8 {
        let b = self.source[self.pos];
        self.pos += 1;
        b
    }

    fn skip_whitespace_and_comments(&mut self) {
        loop {
            while self.pos < self.source.len() && self.source[self.pos].is_ascii_whitespace() {
                self.pos += 1;
            }
            if self.pos + 1 < self.source.len()
                && self.source[self.pos] == b'/'
                && self.source[self.pos + 1] == b'/'
            {
                self.pos += 2;
                // Use memchr to find end of line
                match memchr(b'\n', &self.source[self.pos..]) {
                    Some(offset) => self.pos += offset,
                    None => self.pos = self.source.len(),
                }
                continue;
            }
            break;
        }
    }

    fn next_token(&mut self) -> Result<Token, LexError> {
        let start = self.pos;
        let b = self.advance();
        let kind = match b {
            b'{' => TokenKind::LBrace,
            b'}' => TokenKind::RBrace,
            b'(' => TokenKind::LParen,
            b')' => TokenKind::RParen,
            b'[' => TokenKind::LBracket,
            b']' => TokenKind::RBracket,
            b',' => TokenKind::Comma,
            b':' => {
                if self.peek() == Some(b':') {
                    self.advance();
                    TokenKind::ColonColon
                } else {
                    TokenKind::Colon
                }
            }
            b'.' => TokenKind::Dot,
            b'=' => TokenKind::Eq,
            b'<' => TokenKind::Lt,
            b'>' => TokenKind::Gt,
            b'-' => {
                if self.peek() == Some(b'>') {
                    self.advance();
                    TokenKind::Arrow
                } else {
                    TokenKind::Minus
                }
            }
            b'"' => self.lex_string(start)?,
            b'0'..=b'9' => self.lex_int(start)?,
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_ident(start)?,
            _ => {
                return Err(LexError {
                    kind: LexErrorKind::UnexpectedChar(b as char),
                    span: self.file_span(Span::from_usize(start, self.pos)),
                });
            }
        };
        Ok(Token {
            kind,
            span: Span::from_usize(start, self.pos),
        })
    }

    /// Scan a string literal. Token span will cover `"..."` including quotes.
    /// Uses memchr to skip normal characters, validates escapes inline.
    fn lex_string(&mut self, start: usize) -> Result<TokenKind, LexError> {
        loop {
            // Fast-skip to the next interesting byte: " or \ or \n
            match memchr2(b'"', b'\\', &self.source[self.pos..]) {
                None => {
                    return Err(LexError {
                        kind: LexErrorKind::UnterminatedString,
                        span: self.file_span(Span::from_usize(start, self.source.len())),
                    });
                }
                Some(offset) => {
                    // Check for newline in the skipped region
                    if let Some(nl) = memchr(b'\n', &self.source[self.pos..self.pos + offset]) {
                        return Err(LexError {
                            kind: LexErrorKind::NewlineInString,
                            span: self.file_span(Span::from_usize(start, self.pos + nl)),
                        });
                    }
                    self.pos += offset;
                    match self.source[self.pos] {
                        b'"' => {
                            self.pos += 1;
                            break;
                        }
                        b'\\' => {
                            let esc_pos = self.pos;
                            self.pos += 1;
                            if self.pos >= self.source.len() {
                                return Err(LexError {
                                    kind: LexErrorKind::UnterminatedEscape,
                                    span: self
                                        .file_span(Span::from_usize(esc_pos, self.source.len())),
                                });
                            }
                            match self.source[self.pos] {
                                b'"' | b'\\' | b'n' | b't' | b'r' | b'0' => self.pos += 1,
                                other => {
                                    return Err(LexError {
                                        kind: LexErrorKind::InvalidEscape(other as char),
                                        span: self
                                            .file_span(Span::from_usize(esc_pos, self.pos + 1)),
                                    });
                                }
                            }
                        }
                        _ => unreachable!(),
                    }
                }
            }
        }
        Ok(TokenKind::StringLit)
    }

    /// Scan a raw string literal: r"...", r#"..."#, r##"..."##, etc.
    /// Called when we've already consumed `r` and peeked `"` or `#`.
    fn lex_raw_string(&mut self, start: usize) -> Result<TokenKind, LexError> {
        let mut hash_count: u8 = 0;
        while self.peek() == Some(b'#') {
            self.advance();
            hash_count += 1;
        }
        if self.peek() != Some(b'"') {
            return Err(LexError {
                kind: LexErrorKind::ExpectedRawStringQuote,
                span: self.file_span(Span::from_usize(start, self.pos)),
            });
        }
        self.advance(); // skip opening "

        // Scan for closing " followed by hash_count #'s
        loop {
            match memchr(b'"', &self.source[self.pos..]) {
                None => {
                    return Err(LexError {
                        kind: LexErrorKind::UnterminatedRawString,
                        span: self.file_span(Span::from_usize(start, self.source.len())),
                    });
                }
                Some(offset) => {
                    self.pos += offset + 1; // advance past the "
                    // Check if followed by the right number of #'s
                    let mut found = 0u8;
                    while found < hash_count
                        && self.pos < self.source.len()
                        && self.source[self.pos] == b'#'
                    {
                        self.pos += 1;
                        found += 1;
                    }
                    if found == hash_count {
                        break; // found the closing delimiter
                    }
                    // Not enough #'s, keep scanning
                }
            }
        }
        Ok(TokenKind::RawStringLit { hash_count })
    }

    fn lex_int(&mut self, start: usize) -> Result<TokenKind, LexError> {
        let int_start = self.pos - 1;
        while let Some(b'0'..=b'9') = self.peek() {
            self.advance();
        }
        // Safe: digits are ASCII
        let text = unsafe { std::str::from_utf8_unchecked(&self.source[int_start..self.pos]) };
        let value: i32 = text.parse().map_err(|_| LexError {
            kind: LexErrorKind::IntegerOverflow,
            span: self.file_span(Span::from_usize(start, self.pos)),
        })?;
        Ok(TokenKind::IntLit(value))
    }

    fn lex_ident(&mut self, start: usize) -> Result<TokenKind, LexError> {
        let ident_start = self.pos - 1;
        while let Some(b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'_') = self.peek() {
            self.advance();
        }
        // Check for raw string: r"..." or r#"..."#
        let text = unsafe { std::str::from_utf8_unchecked(&self.source[ident_start..self.pos]) };
        if text == "r" && matches!(self.peek(), Some(b'"' | b'#')) {
            return self.lex_raw_string(start);
        }
        Ok(match text {
            "grammar" => TokenKind::KwGrammar,
            "rule" => TokenKind::KwRule,
            "let" => TokenKind::KwLet,
            "fn" => TokenKind::KwFn,
            "for" => TokenKind::KwFor,
            "in" => TokenKind::KwIn,
            "seq" => TokenKind::KwSeq,
            "choice" => TokenKind::KwChoice,
            "repeat" => TokenKind::KwRepeat,
            "repeat1" => TokenKind::KwRepeat1,
            "optional" => TokenKind::KwOptional,
            "blank" => TokenKind::KwBlank,
            "field" => TokenKind::KwField,
            "alias" => TokenKind::KwAlias,
            "token" => TokenKind::KwToken,
            "prec" => TokenKind::KwPrec,
            "reserved" => TokenKind::KwReserved,
            "token_immediate" => TokenKind::KwTokenImmediate,
            "concat" => TokenKind::KwConcat,
            "regexp" => TokenKind::KwRegexp,
            "inherit" => TokenKind::KwInherit,
            "override" => TokenKind::KwOverride,
            "append" => TokenKind::KwAppend,
            _ => TokenKind::Ident,
        })
    }
}

/// An error encountered during lexing, with the source span of the problem.
#[derive(Debug, Serialize, Error)]
pub struct LexError {
    pub kind: LexErrorKind,
    pub span: FileSpan,
}

/// The specific kind of lexer error.
#[derive(Debug, Serialize)]
pub enum LexErrorKind {
    UnterminatedString,
    UnterminatedRawString,
    UnterminatedEscape,
    InvalidEscape(char),
    UnexpectedChar(char),
    NewlineInString,
    ExpectedRawStringQuote,
    IntegerOverflow,
}

impl std::fmt::Display for LexError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LexErrorKind::UnterminatedString => write!(f, "unterminated string literal"),
            LexErrorKind::UnterminatedRawString => write!(f, "unterminated raw string literal"),
            LexErrorKind::UnterminatedEscape => write!(f, "unterminated escape sequence"),
            LexErrorKind::InvalidEscape(ch) => write!(f, "invalid escape sequence: \\{ch}"),
            LexErrorKind::UnexpectedChar(ch) => write!(f, "unexpected character: {ch:?}"),
            LexErrorKind::NewlineInString => {
                write!(
                    f,
                    "unterminated string literal (newline before closing quote)"
                )
            }
            LexErrorKind::ExpectedRawStringQuote => {
                write!(f, "expected '\"' after 'r' and '#' delimiters")
            }
            LexErrorKind::IntegerOverflow => {
                write!(
                    f,
                    "integer literal out of range [{}, {}]",
                    i32::MIN,
                    i32::MAX,
                )
            }
        }
    }
}
