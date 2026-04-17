//! Diagnostic rendering for DSL errors - rustc-style error output with
//! source snippets, underlines, and cross-file notes.

use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use super::DslError;
use super::ast::{self, Span};

/// A [`DslError`] bundled with source text and file path for diagnostic rendering.
#[derive(Debug, Error, Serialize)]
#[error("{}", render_diagnostic(error, source_text, path))]
pub struct NativeDslError {
    pub error: DslError,
    #[serde(skip)]
    pub source_text: String,
    #[serde(skip)]
    pub path: std::path::PathBuf,
}

fn paint(color: anstyle::AnsiColor, text: &str) -> String {
    let style = anstyle::Style::new().fg_color(Some(color.into())).bold();
    format!("{style}{text}{style:#}")
}

struct SpanContext<'a> {
    line_num: usize,
    col: usize,
    line_text: &'a str,
    prev_line_text: Option<&'a str>,
    prev_line_num: usize,
    span_start_in_line: usize,
    underline_len: usize,
}

fn span_context(span: Span, source_text: &str) -> SpanContext<'_> {
    let offset = span.start as usize;
    let bytes = source_text.as_bytes();

    let mut line_start = 0;
    let mut prev_line_start = None;
    let mut line_num = 1;
    for (i, &b) in bytes.iter().enumerate().take(offset) {
        if b == b'\n' {
            prev_line_start = Some(line_start);
            line_start = i + 1;
            line_num += 1;
        }
    }
    let line_end =
        memchr::memchr(b'\n', &bytes[line_start..]).map_or(bytes.len(), |pos| line_start + pos);
    let col = offset - line_start + 1;
    let line_text = &source_text[line_start..line_end];
    let span_start_in_line = col.saturating_sub(1);
    let span_len = (span.end - span.start) as usize;
    let underline_len = span_len
        .max(1)
        .min(line_text.len().saturating_sub(span_start_in_line));

    let prev_line_text = prev_line_start.map(|ps| &source_text[ps..line_start.saturating_sub(1)]);

    SpanContext {
        line_num,
        col,
        line_text,
        prev_line_text,
        prev_line_num: line_num.saturating_sub(1),
        span_start_in_line,
        underline_len,
    }
}

fn render_diagnostic(error: &DslError, source_text: &str, path: &Path) -> String {
    use anstyle::AnsiColor;
    use std::fmt::Write;

    // Module errors: render the inner error with the child file's context,
    // then append a note pointing back to the import/inherit in the parent.
    if let DslError::Module(e) = error {
        let mut out = render_diagnostic(&e.inner, &e.source_text, &e.path);
        render_note_snippet(
            &mut out,
            e.reference_span,
            source_text,
            path,
            &ast::NoteMessage::ReferencedFromHere,
        );
        return out;
    }

    let ctx = span_context(error.span(), source_text);
    let path_display = path.display();
    let gutter_width = digit_count(ctx.line_num);
    let pipe = paint(AnsiColor::Cyan, "|");
    let underline = paint(AnsiColor::Red, &"^".repeat(ctx.underline_len));

    let mut out = String::new();
    writeln!(out, "{}: {error}", paint(AnsiColor::Red, "error")).unwrap();
    writeln!(
        out,
        " {} {path_display}:{line_num}:{col}",
        paint(AnsiColor::Cyan, "-->"),
        line_num = ctx.line_num,
        col = ctx.col,
    )
    .unwrap();
    writeln!(out, " {:>gutter_width$} {pipe}", "").unwrap();
    if let Some(prev_text) = ctx.prev_line_text {
        writeln!(
            out,
            " {} {pipe} {prev_text}",
            paint(AnsiColor::Cyan, &ctx.prev_line_num.to_string()),
        )
        .unwrap();
    }
    writeln!(
        out,
        " {} {pipe} {}",
        paint(AnsiColor::Cyan, &ctx.line_num.to_string()),
        ctx.line_text,
    )
    .unwrap();
    write!(
        out,
        " {:>gutter_width$} {pipe} {:>width$}{underline}",
        "",
        "",
        width = ctx.span_start_in_line,
    )
    .unwrap();

    if let Some(note) = error.note() {
        render_note_snippet(&mut out, note.span, &note.source, &note.path, &note.message);
    }

    if let Some(trace) = error.call_trace() {
        let show = 3;
        let elided = trace.len().saturating_sub(show * 2);
        writeln!(out).unwrap();
        writeln!(out, " {} call trace:", paint(AnsiColor::Cyan, "=")).unwrap();
        for (i, (name, span)) in trace.iter().enumerate() {
            if elided > 0 && i == show {
                writeln!(
                    out,
                    "   {} ... {elided} more frames ...",
                    paint(AnsiColor::Cyan, "...")
                )
                .unwrap();
            }
            if i >= show && i < trace.len() - show {
                continue;
            }
            let fctx = span_context(*span, source_text);
            writeln!(
                out,
                "   {} {path_display}:{}:{} in {name}()",
                paint(AnsiColor::Cyan, "-->"),
                fctx.line_num,
                fctx.col,
            )
            .unwrap();
        }
    }

    out
}

fn render_note_snippet(
    out: &mut String,
    span: Span,
    source_text: &str,
    path: &Path,
    message: &ast::NoteMessage,
) {
    use anstyle::AnsiColor;
    use std::fmt::Write;

    let nctx = span_context(span, source_text);
    let note_gutter = digit_count(nctx.line_num);
    let note_pipe = paint(AnsiColor::Cyan, "|");
    let note_underline = paint(AnsiColor::Cyan, &"-".repeat(nctx.underline_len));
    let note_path = path.display();

    writeln!(out).unwrap();
    writeln!(
        out,
        " {} {note_path}:{line_num}:{col}",
        paint(AnsiColor::Cyan, "-->"),
        line_num = nctx.line_num,
        col = nctx.col,
    )
    .unwrap();
    writeln!(out, " {:>note_gutter$} {note_pipe}", "").unwrap();
    writeln!(
        out,
        " {} {note_pipe} {}",
        paint(AnsiColor::Cyan, &nctx.line_num.to_string()),
        nctx.line_text,
    )
    .unwrap();
    write!(
        out,
        " {:>note_gutter$} {note_pipe} {:>width$}{note_underline} {}",
        "",
        "",
        paint(AnsiColor::Cyan, &format!("note: {message}")),
        width = nctx.span_start_in_line,
    )
    .unwrap();
}

fn digit_count(n: usize) -> usize {
    n.checked_ilog10().map_or(1, |d| d as usize + 1)
}
