//! Diagnostic rendering for DSL errors.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use anstyle::{AnsiColor, Color, Style};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{DslError, NoteMessage, ast::Span};

struct Paint<T>(Style, T);

impl<T: std::fmt::Display> std::fmt::Display for Paint<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if color_enabled() {
            write!(f, "{}{}{:#}", self.0, self.1, self.0)
        } else {
            self.1.fmt(f)
        }
    }
}

fn color_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    // Per https://no-color.org: any non-empty value of NO_COLOR disables color.
    *ENABLED.get_or_init(|| std::env::var_os("NO_COLOR").is_none_or(|v| v.is_empty()))
}

const PIPE: Paint<&str> = Paint(CYAN_STYLE, "|");
const ARROW: Paint<&str> = Paint(CYAN_STYLE, "-->");
const ELLIP: Paint<&str> = Paint(CYAN_STYLE, "...");
const EQUALS: Paint<&str> = Paint(CYAN_STYLE, "=");
const ERROR: Paint<&str> = Paint(RED_STYLE, "error");
const NOTE: Paint<&str> = Paint(CYAN_STYLE, "note");

const RED_STYLE: Style = Style::new()
    .fg_color(Some(Color::Ansi(AnsiColor::Red)))
    .bold();
const CYAN_STYLE: Style = Style::new()
    .fg_color(Some(Color::Ansi(AnsiColor::Cyan)))
    .bold();

struct SnippetMarker {
    style: Style,
    glyph: char,
}

const ERROR_MARKER: SnippetMarker = SnippetMarker {
    style: RED_STYLE,
    glyph: '^',
};
const NOTE_MARKER: SnippetMarker = SnippetMarker {
    style: CYAN_STYLE,
    glyph: '-',
};

/// A located region of source text.
#[derive(Clone, Copy)]
struct Source<'a> {
    span: Span,
    text: &'a str,
    path: &'a Path,
}

#[derive(Clone)]
enum SnippetKind {
    Error,
    Note(NoteMessage),
}

impl SnippetKind {
    const fn marker(&self) -> SnippetMarker {
        match self {
            Self::Error => ERROR_MARKER,
            Self::Note(_) => NOTE_MARKER,
        }
    }
}

/// A [`DslError`] bundled with source text and file path for diagnostic rendering.
#[derive(Debug, Error, Serialize, Deserialize)]
pub struct NativeDslError {
    pub error: DslError,
    #[serde(skip)]
    pub src: String,
    #[serde(skip)]
    pub path: PathBuf,
}

impl std::fmt::Display for NativeDslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if matches!(self.error, DslError::Module(_)) {
            // Walk the Module chain to find the leaf error and collect
            // intermediate ModuleErrors for the reference note chain.
            let mut modules = Vec::new();
            let mut current = &self.error;
            while let DslError::Module(m) = current {
                modules.push(m);
                current = &m.inner;
            }
            let innermost = modules.last().unwrap();

            // Render the leaf error with the innermost module's source context
            render_error(f, current, &innermost.source_text, &innermost.path)?;

            // Each module is referenced *from* its parent's source: modules[0]
            // from the root, modules[i] from modules[i-1]. Pair each module
            // with its parent's (text, path) and render innermost-out.
            let parents: Vec<(&str, &Path)> =
                std::iter::once((self.src.as_str(), self.path.as_path()))
                    .chain(
                        modules[..modules.len() - 1]
                            .iter()
                            .map(|m| (m.source_text.as_str(), m.path.as_path())),
                    )
                    .collect();

            for (m, (text, path)) in modules.iter().zip(parents).rev() {
                writeln!(f)?;
                render_snippet(
                    f,
                    Source {
                        span: m.reference_span,
                        text,
                        path,
                    },
                    SnippetKind::Note(NoteMessage::ReferencedFromHere),
                    false,
                )?;
            }

            return Ok(());
        }

        render_error(f, &self.error, &self.src, &self.path)
    }
}

fn render_error(
    f: &mut std::fmt::Formatter<'_>,
    error: &DslError,
    text: &str,
    path: &Path,
) -> std::fmt::Result {
    writeln!(f, "{ERROR}: {error}")?;

    // A file-level error (e.g. a missing grammar block) has no location within
    // the source, but still name the file so the user knows where; located
    // errors fall through to the snippet below.
    let Some(span) = error.span() else {
        writeln!(f, " {ARROW} {}", path.display())?;
        return Ok(());
    };

    render_snippet(f, Source { span, text, path }, SnippetKind::Error, true)?;

    for note in error.notes() {
        writeln!(f)?;
        render_snippet(
            f,
            Source {
                span: note.span,
                text: &note.source,
                path: &note.path,
            },
            SnippetKind::Note(note.message.clone()),
            false,
        )?;
    }

    if let Some(trace) = error.call_trace() {
        let show = 3;
        let elided = trace.len().saturating_sub(show * 2);
        writeln!(f)?;
        writeln!(f, " {EQUALS} call trace:")?;
        for (i, (name, path, line, col)) in trace.iter().enumerate() {
            if elided > 0 && i == show {
                writeln!(f, "   {ELLIP} ... {elided} more frames ...")?;
            }
            if i >= show && i < trace.len().saturating_sub(show) {
                continue;
            }
            writeln!(f, "   {ARROW} {}:{line}:{col} in {name}()", path.display())?;
        }
    }

    Ok(())
}

/// Render a source snippet with a location header, source line, and underline.
fn render_snippet(
    f: &mut std::fmt::Formatter<'_>,
    src: Source<'_>,
    kind: SnippetKind,
    show_prev_line: bool,
) -> std::fmt::Result {
    let marker = kind.marker();
    let ctx = SpanContext::new(src.span, src.text);
    let underline = Paint(
        marker.style,
        marker.glyph.to_string().repeat(ctx.underline_len),
    );
    let gutter = ctx.gutter_width;

    writeln!(
        f,
        " {ARROW} {}:{}:{}",
        src.path.display(),
        ctx.line_num,
        ctx.col
    )?;
    writeln!(f, " {:>gutter$} {PIPE}", "")?;
    if show_prev_line && let Some(prev_text) = ctx.prev_line_text {
        writeln!(
            f,
            " {} {PIPE} {prev_text}",
            Paint(CYAN_STYLE, ctx.prev_line_num)
        )?;
    }
    writeln!(
        f,
        " {} {PIPE} {}",
        Paint(CYAN_STYLE, ctx.line_num),
        ctx.line_text,
    )?;
    write!(
        f,
        " {pad:>gutter$} {PIPE} {pad:>width$}{underline}",
        pad = "",
        width = ctx.span_start_in_line
    )?;
    if let SnippetKind::Note(msg) = kind {
        write!(f, " {NOTE}: {}", Paint(marker.style, msg))?;
    }

    Ok(())
}

/// Largest char boundary `<= i` (clamped to the string length). Spans are byte
/// offsets and an error span can land inside a multibyte char, so clamp before
/// slicing `source_text` to avoid a panic while rendering the diagnostic.
fn floor_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// Smallest char boundary `>= i` (clamped to the string length).
fn ceil_char_boundary(s: &str, i: usize) -> usize {
    let mut i = i.min(s.len());
    while !s.is_char_boundary(i) {
        i += 1;
    }
    i
}

struct SpanContext<'a> {
    line_num: usize,
    col: usize,
    line_text: &'a str,
    prev_line_text: Option<&'a str>,
    prev_line_num: usize,
    span_start_in_line: usize,
    underline_len: usize,
    gutter_width: usize,
}

impl SpanContext<'_> {
    fn new(span: Span, source_text: &str) -> SpanContext<'_> {
        let offset = floor_char_boundary(source_text, span.start as usize);
        let bytes = source_text.as_bytes();
        let prefix = &bytes[..offset];
        let line_num = memchr::memchr_iter(b'\n', prefix).count() + 1;
        let line_start = memchr::memrchr(b'\n', prefix).map_or(0, |i| i + 1);
        let prev_line_start = if line_start > 0 {
            Some(memchr::memrchr(b'\n', &prefix[..line_start - 1]).map_or(0, |i| i + 1))
        } else {
            None
        };
        let line_end =
            memchr::memchr(b'\n', &bytes[line_start..]).map_or(bytes.len(), |pos| line_start + pos);
        let line_text = &source_text[line_start..line_end];
        // Char-counted so the printed column and caret padding align with the
        // displayed glyphs on lines containing multibyte UTF-8.
        let span_start_in_line = source_text[line_start..offset].chars().count();
        let col = span_start_in_line + 1;
        let underline_end =
            ceil_char_boundary(source_text, (span.end as usize).min(line_end)).max(offset);
        let underline_len = source_text[offset..underline_end].chars().count().max(1);

        let prev_line_text =
            prev_line_start.map(|ps| &source_text[ps..line_start.saturating_sub(1)]);

        SpanContext {
            line_num,
            col,
            line_text,
            prev_line_text,
            prev_line_num: line_num.saturating_sub(1),
            span_start_in_line,
            underline_len,
            gutter_width: digit_count(line_num),
        }
    }
}

fn digit_count(n: usize) -> usize {
    n.checked_ilog10().map_or(1, |d| d as usize + 1)
}
