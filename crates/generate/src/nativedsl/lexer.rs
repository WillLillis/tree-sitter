//! Lexer for the native grammar DSL. Produces a flat `Vec<Token>` from
//! source bytes.

use memchr::{memchr, memchr2};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::LexError;
use super::ast::Span;

/// Byte classification flags for the lexer's hot loops.
const CLASS_WHITESPACE: u8 = 1;
const CLASS_IDENT_CONTINUE: u8 = 2;
const CLASS_IDENT_START: u8 = 4;

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

/// True iff `s` is shaped like an `Ident` token. Used by expand to
/// validate computed names against the same predicate the lexer uses
/// for source-text idents.
#[must_use]
pub fn is_ident_str(s: &str) -> bool {
    let bytes = s.as_bytes();
    let Some((&first, rest)) = bytes.split_first() else {
        return false;
    };
    byte_is(first, CLASS_IDENT_START) && rest.iter().all(|&b| byte_is(b, CLASS_IDENT_CONTINUE))
}

/// The kind of a lexer token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenKind {
    // Identifiers and literals
    Ident,
    /// String literal. Token span covers `"..."` including quotes.
    StringLit,
    /// Raw string literal. Token span covers `r"..."`, `r#"..."#`, etc.
    RawStringLit {
        hash_count: u8,
    },
    IntLit(u32),
    // Structural keywords
    KwGrammar,
    KwRule,
    KwRules,
    KwLet,
    KwMacro,
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
    KwPrecLeft,
    KwPrecRight,
    KwPrecDynamic,
    KwReserved,
    KwTokenImmediate,
    KwConcat,
    KwRegexp,
    KwInherit,
    KwImport,
    KwOverride,
    KwAppend,
    KwGrammarConfig,
    KwExternal,
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
    Minus,
    Plus,
    Eq,
    Lt,
    Gt,
    Pound,
    At,
    // Other
    Comment,
    Eof,
}

/// Maps keyword text to `TokenKind` variant. Used by the lexer, Display, and `is_keyword`.
macro_rules! keywords {
    (
        decls { $($d_variant:ident => $d_str:literal),* $(,)? }
        combinators { $($c_variant:ident => $c_str:literal),* $(,)? }
    ) => {
        impl TokenKind {
            /// Keywords callable in expression position. Used to seed
            /// "did you mean?" suggestions when an identifier in a rule body
            /// fails to resolve.
            pub(super) const COMBINATOR_KEYWORD_NAMES: &'static [&'static str] = &[$($c_str),*];

            #[must_use]
            pub const fn is_keyword(self) -> bool {
                matches!(self, $(Self::$d_variant)|* $(| Self::$c_variant)*)
            }

            /// Return the keyword string if this is a keyword token, or `None`.
            const fn keyword_str(self) -> Option<&'static str> {
                match self {
                    $(Self::$d_variant => Some($d_str),)*
                    $(Self::$c_variant => Some($c_str),)*
                    _ => None,
                }
            }

            /// Map keyword text to a `TokenKind`, or return `Ident`.
            pub(super) fn from_keyword(text: &str) -> Self {
                match text {
                    $($d_str => Self::$d_variant,)*
                    $($c_str => Self::$c_variant,)*
                    _ => Self::Ident,
                }
            }
        }
    };
}

keywords! {
    decls {
        KwGrammar => "grammar", KwRule => "rule", KwRules => "rules", KwLet => "let",
        KwMacro => "macro", KwOverride => "override", KwExternal => "external",
        KwIn => "in",
    }
    combinators {
        KwFor => "for", KwSeq => "seq", KwChoice => "choice",
        KwRepeat => "repeat", KwRepeat1 => "repeat1", KwOptional => "optional",
        KwBlank => "blank", KwField => "field", KwAlias => "alias", KwToken => "token",
        KwPrec => "prec", KwPrecLeft => "prec_left", KwPrecRight => "prec_right",
        KwPrecDynamic => "prec_dynamic", KwReserved => "reserved",
        KwTokenImmediate => "token_immediate", KwConcat => "concat", KwRegexp => "regexp",
        KwInherit => "inherit", KwImport => "import",
        KwAppend => "append", KwGrammarConfig => "grammar_config",
    }
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(kw) = self.keyword_str() {
            return write!(f, "'{kw}'");
        }
        f.write_str(match self {
            Self::Ident => "identifier",
            Self::StringLit => "string literal",
            Self::RawStringLit { .. } => "raw string literal",
            Self::IntLit(_) => "integer literal",
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
            Self::Minus => "'-'",
            Self::Plus => "'+'",
            Self::Eq => "'='",
            Self::Lt => "'<'",
            Self::Gt => "'>'",
            Self::Pound => "'#'",
            Self::At => "'@'",
            Self::Comment => "comment",
            Self::Eof => "end of file",
            _ => unreachable!(),
        })
    }
}

#[derive(Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
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
        let mut hash_count: u8 = 0;
        while self.peek() == Some(b'#') {
            self.advance();
            hash_count = hash_count.checked_add(1).ok_or_else(|| {
                LexError::new(
                    LexErrorKind::TooManyHashes,
                    Span::from_usize(start, self.pos),
                )
            })?;
        }
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

pub type LexResult<T> = Result<T, super::LexError>;

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
    #[error("raw string has too many '#' delimiters (maximum 255)")]
    TooManyHashes,
    #[error("input exceeds maximum size")]
    InputTooLarge,
}

/// First char boundary strictly after byte `i` (or `source.len()` if `i` is at
/// or past the end).
fn past_char(source: &[u8], i: usize) -> usize {
    if i >= source.len() {
        return source.len();
    }
    // SAFETY: the lexer's source is the bytes of a `&str` (`Lexer::new` takes
    // `&str`), so it is valid UTF-8.
    let s = unsafe { std::str::from_utf8_unchecked(source) };
    let mut j = i + 1;
    while !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

/// Read exactly `n` ASCII hex digits starting at `start`, returning the parsed
/// value. `Err` carries the index *past* the first non-hex-digit char (or EOF)
/// so the error span includes the offending character.
///
/// Both the inspected positions (after ASCII hex digits) and the returned index
/// (via [`past_char`]) land on char boundaries, so spans built from them never
/// split a multibyte char - which would mislead the diagnostic and, via the old
/// `from_utf8_unchecked`, was UB.
fn read_hex_digits(source: &[u8], start: usize, n: usize) -> Result<u32, usize> {
    let mut value = 0u32;
    for i in 0..n {
        match source.get(start + i).and_then(|&b| char::from(b).to_digit(16)) {
            Some(d) => value = value * 16 + d,
            None => return Err(past_char(source, start + i)),
        }
    }
    Ok(value)
}

/// Validate `\xHH` (exactly 2 hex digits, value 0x00-0x7F). Returns the new
/// position past the escape on success. `esc_pos` points at the backslash;
/// `pos` points at the `x`.
fn validate_hex_escape(source: &[u8], esc_pos: usize, pos: usize) -> LexResult<usize> {
    let bad = |e| LexError::new(LexErrorKind::InvalidHexEscape, Span::from_usize(esc_pos, e));
    let value = read_hex_digits(source, pos + 1, 2).map_err(bad)?;
    let end = pos + 3; // both digits are ASCII, so this is a char boundary
    if value <= 0x7F {
        Ok(end)
    } else {
        Err(bad(end))
    }
}

/// Validate `\uHHHH` (4 hex digits) or `\u{H..H}` (1-6 hex digits in braces).
/// Codepoint must be <= 0x10FFFF and not a surrogate (0xD800-0xDFFF).
fn validate_unicode_escape(source: &[u8], esc_pos: usize, pos: usize) -> LexResult<usize> {
    let after_u = pos + 1;
    let bad = |e: usize| {
        LexError::new(
            LexErrorKind::InvalidUnicodeEscape,
            Span::from_usize(esc_pos, e.min(source.len())),
        )
    };
    let (codepoint, end) = if source.get(after_u) == Some(&b'{') {
        let digits_start = after_u + 1;
        let Some(close) = memchr(b'}', &source[digits_start..]).map(|o| digits_start + o) else {
            return Err(bad(source.len()));
        };
        if !(1..=6).contains(&(close - digits_start)) {
            return Err(bad(close + 1));
        }
        // `digits_start..close` lies between the ASCII `{` and `}`, so it is
        // boundary-aligned; non-hex (incl. multibyte) content just fails to parse.
        let Ok(codepoint) = std::str::from_utf8(&source[digits_start..close])
            .ok()
            .and_then(|hex| u32::from_str_radix(hex, 16).ok())
            .ok_or(())
        else {
            return Err(bad(close + 1));
        };
        (codepoint, close + 1)
    } else {
        // `\uHHHH`: four ASCII hex digits.
        let codepoint = read_hex_digits(source, after_u, 4).map_err(bad)?;
        (codepoint, after_u + 4)
    };
    if codepoint <= 0x0010_FFFF && !(0xD800..=0xDFFF).contains(&codepoint) {
        Ok(end)
    } else {
        Err(bad(end))
    }
}
