//! The byte-level tokenizer.
//!
//! Splits into [`token`] (the [`TokenKind`]/[`Token`] vocabulary, shared with
//! the parser), [`error`] ([`LexErrorKind`]), and [`escape`] (escape-sequence
//! decoding/validation); this file is the machine.

mod error;
mod escape;
mod token;

pub use error::{LexErrorKind, LexResult};
pub(crate) use escape::unescape_string;
pub use token::{Token, TokenKind};

use memchr::{memchr, memchr2};

use crate::nativedsl::{LexError, ast::Span};

use escape::{validate_hex_escape, validate_unicode_escape};

/// Byte classification flags for the lexer's hot loops.
const CLASS_WHITESPACE: u8 = 0b0000_0001;
const CLASS_IDENT_CONTINUE: u8 = 0b0000_0010;
const CLASS_IDENT_START: u8 = 0b0000_0100;

/// Lookup table mapping byte value to classification flags.
const BYTE_CLASS: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b' ' as usize] = CLASS_WHITESPACE;
    t[b'\t' as usize] = CLASS_WHITESPACE;
    t[b'\n' as usize] = CLASS_WHITESPACE;
    t[b'\r' as usize] = CLASS_WHITESPACE;
    t[0x0c] = CLASS_WHITESPACE; // form feed
    let mut i = b'a';
    while i <= b'z' {
        t[i as usize] |= CLASS_IDENT_CONTINUE | CLASS_IDENT_START;
        i += 1;
    }
    i = b'A';
    while i <= b'Z' {
        t[i as usize] |= CLASS_IDENT_CONTINUE | CLASS_IDENT_START;
        i += 1;
    }
    i = b'0';
    while i <= b'9' {
        t[i as usize] |= CLASS_IDENT_CONTINUE;
        i += 1;
    }
    t[b'_' as usize] |= CLASS_IDENT_CONTINUE | CLASS_IDENT_START;
    t
};

#[inline]
const fn byte_is(b: u8, class: u8) -> bool {
    BYTE_CLASS[b as usize] & class != 0
}

/// True iff `s` is shaped like an `Ident` token.
#[must_use]
pub fn is_ident_str(s: &str) -> bool {
    let bytes = s.as_bytes();
    let Some((&first, rest)) = bytes.split_first() else {
        return false;
    };
    byte_is(first, CLASS_IDENT_START) && rest.iter().all(|&b| byte_is(b, CLASS_IDENT_CONTINUE))
}

pub struct Lexer<'src> {
    source: &'src [u8],
    pos: usize,
}

impl<'src> Lexer<'src> {
    #[must_use]
    pub const fn new(source: &'src str) -> Self {
        Self {
            source: source.as_bytes(),
            pos: 0,
        }
    }

    /// Consume the entire source and return all tokens. The returned vector
    /// always ends with `TokenKind::Eof`.
    ///
    /// # Errors
    ///
    /// Returns [`LexError`] on unterminated strings, invalid escapes, or
    /// unexpected characters.
    pub fn tokenize(&mut self) -> LexResult<Vec<Token>> {
        let mut tokens = Vec::with_capacity(self.source.len() / 4);
        let source = self.source;
        let len = source.len();
        loop {
            self.skip_whitespace();
            if self.pos >= len {
                tokens.push(Token {
                    kind: TokenKind::Eof,
                    span: Span::from_usize(self.pos, self.pos),
                });
                break;
            }
            // SAFETY: the `pos + 1 < len` guard makes both indexes in-bounds.
            if self.pos + 1 < len
                && unsafe { *source.get_unchecked(self.pos) == b'/' }
                && unsafe { *source.get_unchecked(self.pos + 1) == b'/' }
            {
                let start = self.pos;
                self.pos += 2;
                match memchr(b'\n', &source[self.pos..]) {
                    Some(offset) => self.pos += offset,
                    None => self.pos = len,
                }
                tokens.push(Token {
                    kind: TokenKind::Comment,
                    span: Span::from_usize(start, self.pos),
                });
                continue;
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

    fn skip_whitespace(&mut self) {
        let source = self.source;
        while self.pos < source.len() // SAFETY: pos < source.len() checked by loop condition.
            && byte_is(unsafe { *source.get_unchecked(self.pos) }, CLASS_WHITESPACE)
        {
            self.pos += 1;
        }
    }

    fn next_token(&mut self) -> LexResult<Token> {
        let mut start = self.pos;
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
            b'-' => TokenKind::Minus,
            b'+' => TokenKind::Plus,
            b'#' => TokenKind::Pound,
            b'@' => TokenKind::At,
            b'"' => self.lex_string(start)?,
            b'0'..=b'9' => self.lex_int(start)?,
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_ident(&mut start)?,
            _ => {
                // SAFETY: source originates from &str (Lexer::new takes &str).
                let rest = unsafe { std::str::from_utf8_unchecked(&self.source[start..]) };
                let ch = rest.chars().next().unwrap();
                self.pos = start + ch.len_utf8();
                return Err(LexError::new(
                    LexErrorKind::UnexpectedChar(ch),
                    Span::from_usize(start, self.pos),
                ));
            }
        };
        Ok(Token {
            kind,
            span: Span::from_usize(start, self.pos),
        })
    }

    /// Scan a string literal. Token span will cover `"..."` including quotes.
    /// `self.pos` is not advanced on error.
    fn lex_string(&mut self, start: usize) -> LexResult<TokenKind> {
        let source = self.source;
        let mut pos = self.pos;
        loop {
            // Fast-skip to the next interesting byte: " or \
            match memchr2(b'"', b'\\', &source[pos..]) {
                None => {
                    Err(LexError::new(
                        LexErrorKind::UnterminatedString,
                        Span::from_usize(start, source.len()),
                    ))?;
                }
                Some(offset) => {
                    if let Some(nl) = memchr2(b'\n', b'\r', &source[pos..pos + offset]) {
                        Err(LexError::new(
                            LexErrorKind::NewlineInString,
                            Span::from_usize(start, pos + nl),
                        ))?;
                    }
                    pos += offset;
                    // SAFETY: memchr2 found a byte at this position, so pos < source.len().
                    match unsafe { *source.get_unchecked(pos) } {
                        b'"' => {
                            self.pos = pos + 1;
                            break;
                        }
                        b'\\' => {
                            let esc_pos = pos;
                            pos += 1;
                            if pos >= source.len() {
                                Err(LexError::new(
                                    LexErrorKind::UnterminatedEscape,
                                    Span::from_usize(esc_pos, source.len()),
                                ))?;
                            }
                            // SAFETY: pos < source.len() checked above.
                            match unsafe { *source.get_unchecked(pos) } {
                                b'"' | b'\\' | b'n' | b't' | b'r' | b'0' => pos += 1,
                                b'x' => pos = validate_hex_escape(source, esc_pos, pos)?,
                                b'u' => pos = validate_unicode_escape(source, esc_pos, pos)?,
                                _ => {
                                    // SAFETY: source is valid UTF-8 (from &str).
                                    let rest =
                                        unsafe { std::str::from_utf8_unchecked(&source[pos..]) };
                                    let ch = rest.chars().next().unwrap();
                                    Err(LexError::new(
                                        LexErrorKind::InvalidEscape(ch),
                                        Span::from_usize(esc_pos, pos + ch.len_utf8()),
                                    ))?;
                                }
                            }
                        }
                        // SAFETY: memchr2 only returns positions of b'"' or b'\\'.
                        _ => unreachable!(),
                    }
                }
            }
        }
        Ok(TokenKind::StringLit)
    }

    /// Scan a raw string literal: r"...", r#"..."#, r##"..."##, etc.
    /// Called when we've already consumed `r` and peeked `"` or `#`.
    fn lex_raw_string(&mut self, start: usize) -> LexResult<TokenKind> {
        let source = self.source;
        let hash_start = self.pos;
        while self.peek() == Some(b'#') {
            self.advance();
        }
        let hash_count = self.pos - hash_start;
        let Ok(hash_count) = u8::try_from(hash_count) else {
            return Err(LexError::new(
                LexErrorKind::TooManyHashes(hash_count as u32),
                Span::from_usize(start, self.pos),
            ));
        };
        if self.peek() != Some(b'"') {
            Err(LexError::new(
                LexErrorKind::ExpectedRawStringQuote,
                Span::from_usize(start, self.pos),
            ))?;
        }
        self.advance(); // skip opening "

        // Scan for closing " followed by hash_count #'s
        let mut pos = self.pos;
        loop {
            match memchr(b'"', &source[pos..]) {
                None => {
                    return Err(LexError::new(
                        LexErrorKind::UnterminatedRawString,
                        Span::from_usize(start, source.len()),
                    ));
                }
                Some(offset) => {
                    pos += offset + 1; // advance past the "
                    let mut found = 0u8;
                    // SAFETY: pos < source.len() is checked by loop condition.
                    while found < hash_count
                        && pos < source.len()
                        && unsafe { *source.get_unchecked(pos) } == b'#'
                    {
                        pos += 1;
                        found += 1;
                    }
                    if found == hash_count {
                        break;
                    }
                }
            }
        }
        self.pos = pos;
        Ok(TokenKind::RawStringLit { hash_count })
    }

    fn lex_int(&mut self, start: usize) -> LexResult<TokenKind> {
        let source = self.source;
        let mut pos = self.pos;
        while pos < source.len() && source[pos].is_ascii_digit() {
            pos += 1;
        }
        self.pos = pos;
        // SAFETY: the slice contains only ASCII digits (b'0'..=b'9'), which are valid UTF-8.
        let text = unsafe { std::str::from_utf8_unchecked(&source[start..pos]) };
        let value: u32 = text.parse().map_err(|_| {
            LexError::new(
                LexErrorKind::IntegerOverflow,
                Span::from_usize(start, self.pos),
            )
        })?;
        Ok(TokenKind::IntLit(value))
    }

    fn lex_ident(&mut self, start: &mut usize) -> LexResult<TokenKind> {
        self.scan_ident_continue();
        // SAFETY: the slice contains only ASCII ident chars (a-z, A-Z, 0-9, _), which are valid UTF-8.
        let text = unsafe { std::str::from_utf8_unchecked(&self.source[*start..self.pos]) };
        if text == "r" {
            match (self.peek(), self.source.get(self.pos + 1).copied()) {
                // r#<ident> - raw identifier. Skip `r#` so the span and
                // returned text cover just the bare name (e.g. `let`).
                (Some(b'#'), Some(c)) if byte_is(c, CLASS_IDENT_START) => {
                    self.pos += 1; // skip '#'
                    *start = self.pos;
                    self.pos += 1; // skip the ident-start char (already validated)
                    self.scan_ident_continue();
                    return Ok(TokenKind::Ident);
                }
                // r"..." or r#"..."# - raw string literal.
                (Some(b'"' | b'#'), _) => return self.lex_raw_string(*start),
                _ => {}
            }
        }
        Ok(TokenKind::from_keyword(text))
    }

    fn scan_ident_continue(&mut self) {
        let source = self.source;
        while self.pos < source.len()
            // SAFETY: pos < source.len() checked by loop condition.
            && byte_is(unsafe { *source.get_unchecked(self.pos) }, CLASS_IDENT_CONTINUE)
        {
            self.pos += 1;
        }
    }
}
