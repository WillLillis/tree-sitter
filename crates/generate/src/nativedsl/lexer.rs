//! Lexer for the native grammar DSL. Produces a flat `Vec<Token>` from
//! source bytes.

use memchr::{memchr, memchr2};
use serde::Serialize;

use super::LexError;
use super::ast::Span;

/// Byte classification flags for the lexer's hot loops.
const CLASS_WHITESPACE: u8 = 1;
const CLASS_IDENT_CONTINUE: u8 = 2;

/// Lookup table mapping byte value to classification flags.
/// Replaces per-byte range checks with a single indexed load.
const BYTE_CLASS: [u8; 256] = {
    let mut t = [0u8; 256];
    t[b' ' as usize] = CLASS_WHITESPACE;
    t[b'\t' as usize] = CLASS_WHITESPACE;
    t[b'\n' as usize] = CLASS_WHITESPACE;
    t[b'\r' as usize] = CLASS_WHITESPACE;
    t[0x0c] = CLASS_WHITESPACE; // form feed
    let mut i = b'a';
    while i <= b'z' {
        t[i as usize] |= CLASS_IDENT_CONTINUE;
        i += 1;
    }
    i = b'A';
    while i <= b'Z' {
        t[i as usize] |= CLASS_IDENT_CONTINUE;
        i += 1;
    }
    i = b'0';
    while i <= b'9' {
        t[i as usize] |= CLASS_IDENT_CONTINUE;
        i += 1;
    }
    t[b'_' as usize] |= CLASS_IDENT_CONTINUE;
    t
};

/// Classify a byte using the lookup table.
#[inline]
fn byte_is(b: u8, class: u8) -> bool {
    // SAFETY: b is u8, BYTE_CLASS has 256 entries, so every u8 is a valid index.
    unsafe { *BYTE_CLASS.get_unchecked(b as usize) & class != 0 }
}

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
    Eq,
    Lt,
    Gt,
    // Other
    Comment,
    Eof,
}

/// Maps keyword text to `TokenKind` variant. Used by the lexer, Display, and `is_keyword`.
macro_rules! keywords {
    ($($variant:ident => $str:literal),* $(,)?) => {
        impl TokenKind {
            #[must_use]
            pub const fn is_keyword(self) -> bool {
                matches!(self, $(Self::$variant)|*)
            }

            /// Return the keyword string if this is a keyword token, or `None`.
            const fn keyword_str(self) -> Option<&'static str> {
                match self {
                    $(Self::$variant => Some($str),)*
                    _ => None,
                }
            }

            /// Map keyword text to a `TokenKind`, or return `Ident`.
            pub(super) fn from_keyword(text: &str) -> Self {
                match text {
                    $($str => Self::$variant,)*
                    _ => Self::Ident,
                }
            }
        }
    };
}

keywords! {
    KwGrammar => "grammar", KwRule => "rule", KwLet => "let", KwFn => "fn",
    KwFor => "for", KwIn => "in", KwSeq => "seq", KwChoice => "choice",
    KwRepeat => "repeat", KwRepeat1 => "repeat1", KwOptional => "optional",
    KwBlank => "blank", KwField => "field", KwAlias => "alias", KwToken => "token",
    KwPrec => "prec", KwPrecLeft => "prec_left", KwPrecRight => "prec_right",
    KwPrecDynamic => "prec_dynamic", KwReserved => "reserved",
    KwTokenImmediate => "token_immediate", KwConcat => "concat", KwRegexp => "regexp",
    KwInherit => "inherit", KwImport => "import", KwOverride => "override",
    KwAppend => "append", KwGrammarConfig => "grammar_config",
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
            Self::Eq => "'='",
            Self::Lt => "'<'",
            Self::Gt => "'>'",
            Self::Comment => "comment",
            Self::Eof => "end of file",
            _ => unreachable!(),
        })
    }
}

/// A token produced by the lexer, pairing a kind with its source span.
#[derive(Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

/// Lexer state: a byte-level cursor over the source text.
pub struct Lexer<'src> {
    source: &'src [u8],
    pos: usize,
}

impl<'src> Lexer<'src> {
    /// Create a new lexer for the given source text.
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
            b'-' => TokenKind::Minus,
            b'"' => self.lex_string(start)?,
            b'0'..=b'9' => self.lex_int(start)?,
            b'a'..=b'z' | b'A'..=b'Z' | b'_' => self.lex_ident(start)?,
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
                    // Check for newline in the skipped region
                    if let Some(nl) = memchr(b'\n', &source[pos..pos + offset]) {
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
        let value: i32 = text.parse().map_err(|_| {
            LexError::new(
                LexErrorKind::IntegerOverflow,
                Span::from_usize(start, self.pos),
            )
        })?;
        Ok(TokenKind::IntLit(value))
    }

    fn lex_ident(&mut self, start: usize) -> LexResult<TokenKind> {
        let source = self.source;
        let mut pos = self.pos;
        while pos < source.len() // SAFETY: pos < source.len() checked by loop condition.
            && byte_is(unsafe { *source.get_unchecked(pos) }, CLASS_IDENT_CONTINUE)
        {
            pos += 1;
        }
        self.pos = pos;
        // SAFETY: the slice contains only ASCII ident chars (a-z, A-Z, 0-9, _), which are valid UTF-8.
        let text = unsafe { std::str::from_utf8_unchecked(&self.source[start..self.pos]) };
        if text == "r" && matches!(self.peek(), Some(b'"' | b'#')) {
            return self.lex_raw_string(start);
        }
        Ok(TokenKind::from_keyword(text))
    }
}

pub type LexResult<T> = Result<T, super::LexError>;

/// The specific kind of lexer error.
#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum LexErrorKind {
    UnterminatedString,
    UnterminatedRawString,
    UnterminatedEscape,
    InvalidEscape(char),
    UnexpectedChar(char),
    NewlineInString,
    ExpectedRawStringQuote,
    IntegerOverflow,
    TooManyHashes,
    InputTooLarge,
}

impl std::fmt::Display for LexErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnterminatedString => write!(f, "unterminated string literal"),
            Self::UnterminatedRawString => write!(f, "unterminated raw string literal"),
            Self::UnterminatedEscape => write!(f, "unterminated escape sequence"),
            Self::InvalidEscape(ch) => write!(f, "invalid escape sequence: \\{ch}"),
            Self::UnexpectedChar(ch) => write!(f, "unexpected character: {ch:?}"),
            Self::NewlineInString => {
                write!(
                    f,
                    "unterminated string literal (newline before closing quote)"
                )
            }
            Self::ExpectedRawStringQuote => {
                write!(f, "expected '\"' after 'r' and '#' delimiters")
            }
            Self::TooManyHashes => {
                write!(
                    f,
                    "raw string has too many '#' delimiters (maximum {})",
                    u8::MAX
                )
            }
            Self::IntegerOverflow => {
                write!(
                    f,
                    "integer literal out of range [{}, {}]",
                    i32::MIN,
                    i32::MAX,
                )
            }
            Self::InputTooLarge => {
                write!(f, "input exceeds maximum size of {} bytes", u32::MAX)
            }
        }
    }
}
