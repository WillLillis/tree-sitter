//! String escape-sequence decoding ([`unescape_string`]) and the validators the
//! lexer uses while scanning a string literal.

use memchr::memchr;

use crate::nativedsl::{LexError, LexErrorKind, LexResult, ast::Span};

/// Decode the escape sequences in a quote-stripped string literal's text.
/// Assumes [`lex_string`](super::Lexer) already validated every escape is
/// well-formed, so the unwraps below cannot fail.
pub fn unescape_string(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while let Some(off) = memchr(b'\\', &bytes[i..]) {
        out.extend_from_slice(&bytes[i..i + off]);
        let after = i + off + 1;
        match bytes[after] {
            c @ (b'"' | b'\\') => {
                out.push(c);
                i = after + 1;
            }
            b'n' => {
                out.push(b'\n');
                i = after + 1;
            }
            b't' => {
                out.push(b'\t');
                i = after + 1;
            }
            b'r' => {
                out.push(b'\r');
                i = after + 1;
            }
            b'0' => {
                out.push(0);
                i = after + 1;
            }
            b'x' => {
                // \xHH - 2 hex digits, ASCII range, push as single byte.
                let hex = unsafe { std::str::from_utf8_unchecked(&bytes[after + 1..after + 3]) };
                out.push(u8::from_str_radix(hex, 16).unwrap());
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
                let mut buf = [0u8; 4];
                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                i = end;
            }
            _ => unreachable!(),
        }
    }
    out.extend_from_slice(&bytes[i..]);
    // SAFETY: input was UTF-8 and every escape resolves to valid UTF-8
    // (ASCII for simple/\x, UTF-8-encoded codepoint for \u).
    unsafe { String::from_utf8_unchecked(out) }
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
/// position past the escape on success. `esc_pos` points at the backslash;
/// `pos` points at the `x`.
pub(super) fn validate_hex_escape(source: &[u8], esc_pos: usize, pos: usize) -> LexResult<usize> {
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
pub(super) fn validate_unicode_escape(
    source: &[u8],
    esc_pos: usize,
    pos: usize,
) -> LexResult<usize> {
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
