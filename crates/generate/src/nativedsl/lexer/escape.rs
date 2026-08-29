//! String escape-sequence decoding ([`unescape_string_into`]) and the validators
//! the lexer uses while scanning a string literal.

use memchr::memchr;

use crate::nativedsl::{LexError, LexErrorKind, LexResult, ast::Span};

/// Decode the escape sequences in a quote-stripped string literal's text into
/// `out`, replacing its previous contents.
///
/// Assumes [`lex_string`](super::Lexer) already validated every escape is
/// well-formed.
pub fn unescape_string_into(raw: &str, out: &mut String) {
    let bytes = raw.as_bytes();
    out.clear();
    out.reserve(raw.len());
    let mut i = 0;
    while let Some(off) = memchr(b'\\', &bytes[i..]) {
        // SAFETY: `i` is a UTF-8 boundary and `i+off` points at an ASCII backslash,
        // so both ends of the slice are valid UTF-8 boundaries.
        out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[i..i + off]) });
        let after = i + off + 1;
        match bytes[after] {
            c @ (b'"' | b'\\') => {
                out.push(char::from(c));
                i = after + 1;
            }
            b'n' => {
                out.push('\n');
                i = after + 1;
            }
            b't' => {
                out.push('\t');
                i = after + 1;
            }
            b'r' => {
                out.push('\r');
                i = after + 1;
            }
            b'0' => {
                out.push('\0');
                i = after + 1;
            }
            b'x' => {
                // \xHH - 2 hex digits, ASCII range, push as single byte.
                let hex = unsafe { std::str::from_utf8_unchecked(&bytes[after + 1..after + 3]) };
                out.push(char::from(u8::from_str_radix(hex, 16).unwrap()));
                i = after + 3;
            }
            b'u' => {
                // \uHHHH (4 hex) or \u{H..H} (1-6 hex in braces), UTF-8 encoded.
                let (hex, end) = if bytes[after + 1] == b'{' {
                    let h = after + 2;
                    let p = h + memchr(b'}', &bytes[h..]).unwrap();
                    (&bytes[h..p], p + 1)
                } else {
                    (&bytes[after + 1..after + 5], after + 5)
                };
                let hex = unsafe { std::str::from_utf8_unchecked(hex) };
                let cp = u32::from_str_radix(hex, 16).unwrap();
                // SAFETY: lexer rejected surrogates and values > 0x10FFFF.
                let ch = unsafe { char::from_u32_unchecked(cp) };
                out.push(ch);
                i = end;
            }
            _ => unreachable!(),
        }
    }

    // SAFETY: `i` follows a validated escape and is therefore a UTF-8 boundary.
    out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[i..]) });
}

/// First char boundary strictly after byte `i` (or `source.len()` if `i` is at
/// or past the end).
const fn past_char(source: &[u8], i: usize) -> usize {
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
        match source
            .get(start + i)
            .and_then(|&b| char::from(b).to_digit(16))
        {
            Some(d) => value = value * 16 + d,
            None => return Err(past_char(source, start + i)),
        }
    }
    Ok(value)
}

/// Validate `\xHH` (exactly 2 hex digits, value 0x00-0x7F). Returns the new
/// position past the escape on success. `esc_pos` points at the backslash.
pub(super) fn validate_hex_escape(source: &[u8], esc_pos: usize) -> LexResult<usize> {
    let bad = |e| LexError::new(LexErrorKind::InvalidHexEscape, Span::from_usize(esc_pos, e));
    let digits_start = esc_pos + 2;
    let value = read_hex_digits(source, digits_start, 2).map_err(bad)?;
    let end = digits_start + 2; // both digits are ASCII, so this is a char boundary
    if value <= 0x7F {
        Ok(end)
    } else {
        Err(bad(end))
    }
}

/// Validate `\uHHHH` (4 hex digits) or `\u{H..H}` (1-6 hex digits in braces).
/// Codepoint must be <= 0x10FFFF and not a surrogate (0xD800-0xDFFF).
pub(super) fn validate_unicode_escape(source: &[u8], esc_pos: usize) -> LexResult<usize> {
    let after_u = esc_pos + 2;
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
