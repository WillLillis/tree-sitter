//! String escape-sequence decoding ([`unescape_string_into`]) and the validators
//! the lexer uses while scanning a string literal.

use memchr::memchr;

use crate::nativedsl::{DocumentId, LexError, LexErrorKind, LexResult, ast::Span};

/// Decode the escape sequences in a quote-stripped string literal's text into
/// `out`, replacing its previous contents.
///
/// Assumes [`lex_string`](super::Lexer) already validated every escape is
/// well-formed.
///
/// # Safety
///
/// `raw` must be the contents of an ordinary string literal accepted by the lexer.
pub unsafe fn unescape_string_into(raw: &str, out: &mut String) {
    let bytes = raw.as_bytes();
    out.clear();
    out.reserve(raw.len());
    let mut i = 0;
    while let Some(backslash_off) = memchr(b'\\', &bytes[i..]) {
        // SAFETY: `i` is a UTF-8 boundary and `i+backslash_off` points at an ASCII
        // backslash, so both ends of the slice are valid UTF-8 boundaries.
        out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[i..i + backslash_off]) });
        let escape_code_pos = i + backslash_off + 1;
        match bytes[escape_code_pos] {
            c @ (b'"' | b'\\') => {
                out.push(char::from(c));
                i = escape_code_pos + 1;
            }
            b'n' => {
                out.push('\n');
                i = escape_code_pos + 1;
            }
            b't' => {
                out.push('\t');
                i = escape_code_pos + 1;
            }
            b'r' => {
                out.push('\r');
                i = escape_code_pos + 1;
            }
            b'0' => {
                out.push('\0');
                i = escape_code_pos + 1;
            }
            b'x' => {
                // SAFETY: the function contract guarantees these bytes exist and are
                // ASCII hexadecimal digits.
                let hex = unsafe {
                    std::str::from_utf8_unchecked(&bytes[escape_code_pos + 1..escape_code_pos + 3])
                };
                out.push(char::from(u8::from_str_radix(hex, 16).unwrap()));
                i = escape_code_pos + 3;
            }
            b'u' => {
                // \uHHHH (4 hex) or \u{H..H} (1-6 hex in braces), UTF-8 encoded.
                let (hex, end) = if bytes[escape_code_pos + 1] == b'{' {
                    let digits_start = escape_code_pos + 2;
                    let close = digits_start
                        + bytes[digits_start..]
                            .iter()
                            .position(|&byte| byte == b'}')
                            .unwrap();
                    (&bytes[digits_start..close], close + 1)
                } else {
                    (
                        &bytes[escape_code_pos + 1..escape_code_pos + 5],
                        escape_code_pos + 5,
                    )
                };
                // SAFETY: the function contract guarantees `hex` contains only ASCII
                // hexadecimal digits.
                let hex = unsafe { std::str::from_utf8_unchecked(hex) };
                let codepoint = u32::from_str_radix(hex, 16).unwrap();
                // SAFETY: lexer validation guarantees `codepoint` is a Unicode scalar value.
                let ch = unsafe { char::from_u32_unchecked(codepoint) };
                out.push(ch);
                i = end;
            }
            _ => unreachable!(),
        }
    }

    // SAFETY: `i` is zero or immediately follows a validated escape, so it is a
    // UTF-8 boundary.
    out.push_str(unsafe { std::str::from_utf8_unchecked(&bytes[i..]) });
}

/// First char boundary strictly after byte `i` (or `source.len()` if `i` is at
/// or past the end).
const fn past_char(source: &[u8], i: usize) -> usize {
    if i >= source.len() {
        return source.len();
    }
    // SAFETY: the lexer's source comes from `DocumentRef::text`, so it is valid UTF-8.
    let s = unsafe { std::str::from_utf8_unchecked(source) };
    let mut j = i + 1;
    while !s.is_char_boundary(j) {
        j += 1;
    }
    j
}

/// Read exactly `digit_count` ASCII hexadecimal digits starting at `start`.
///
/// On failure, returns the first UTF-8 boundary after the invalid character, or
/// `source.len()` at EOF. This allows the caller to include the entire invalid
/// range in its error span.
fn read_hex_digits(source: &[u8], start: usize, digit_count: usize) -> Result<u32, usize> {
    let mut value = 0u32;
    for i in 0..digit_count {
        let Some(digit) = source
            .get(start + i)
            .and_then(|&byte| char::from(byte).to_digit(16))
        else {
            return Err(past_char(source, start + i));
        };
        value = value * 16 + digit;
    }
    Ok(value)
}

/// Validate `a \xHH` escape, where `HH` must be in the ASCII range `0x00..=0x7F`.
/// `esc_pos` points at the backslash.
///
/// Returns the position past the escape on success
pub(super) fn validate_hex_escape(
    source: &[u8],
    document: DocumentId,
    esc_pos: usize,
) -> LexResult<usize> {
    let err = |end| {
        LexError::new(
            LexErrorKind::InvalidHexEscape,
            document,
            Span::from_usize(esc_pos, end),
        )
    };
    let digits_start = esc_pos + 2;
    let value = read_hex_digits(source, digits_start, 2).map_err(err)?;
    let end = digits_start + 2; // Both digits are ASCII, so this is a char boundary.
    if value <= 0x7F {
        Ok(end)
    } else {
        Err(err(end))
    }
}

/// Validate a `\uHHHH` or `\u{H..H}` escape.
///
/// The braced form accepts one to six hexadecimal digits. Both forms must encode
/// a Unicode scalar value.
pub(super) fn validate_unicode_escape(
    source: &[u8],
    document: DocumentId,
    esc_pos: usize,
) -> LexResult<usize> {
    let after_u = esc_pos + 2;
    let err = |end: usize| {
        LexError::new(
            LexErrorKind::InvalidUnicodeEscape,
            document,
            Span::from_usize(esc_pos, end),
        )
    };
    let (codepoint, end) = if source.get(after_u) == Some(&b'{') {
        let digits_start = after_u + 1;
        let mut codepoint = 0;
        let mut pos = digits_start;

        for _ in 0..6 {
            let Some(digit) = source
                .get(pos)
                .and_then(|&byte| char::from(byte).to_digit(16))
            else {
                break;
            };
            codepoint = codepoint * 16 + digit;
            pos += 1;
        }

        if pos == digits_start || source.get(pos) != Some(&b'}') {
            return Err(err(past_char(source, pos)));
        }

        (codepoint, pos + 1)
    } else {
        // `\uHHHH`: four ASCII hex digits.
        let codepoint = read_hex_digits(source, after_u, 4).map_err(err)?;
        (codepoint, after_u + 4)
    };
    if char::from_u32(codepoint).is_some() {
        Ok(end)
    } else {
        Err(err(end))
    }
}
