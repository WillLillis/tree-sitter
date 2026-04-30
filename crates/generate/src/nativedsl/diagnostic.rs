//! Diagnostic rendering for DSL errors.

use std::path::{Path, PathBuf};

use anstyle::{AnsiColor, Style};
use serde::Serialize;
use thiserror::Error;

use super::{Diagnostic, DslError, NoteMessage, ast::Span, lower::LowerErrorKind};

/// A [`DslError`] bundled with source text and file path for diagnostic rendering.
#[derive(Debug, Error, Serialize)]
pub struct NativeDslError {
    pub error: DslError,
    #[serde(skip)]
    pub src: String,
    #[serde(skip)]
    pub path: PathBuf,
}

impl std::fmt::Display for NativeDslError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let DslError::Module(_) = &self.error {
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

            // Render "referenced from" notes from innermost to outermost
            for i in (0..modules.len()).rev() {
                let (parent_src, parent_path): (&str, &Path) = if i > 0 {
                    (&modules[i - 1].source_text, &modules[i - 1].path)
                } else {
                    (&self.src, &self.path)
                };
                writeln!(f)?;
                let label = format!("note: {}", NoteMessage::ReferencedFromHere);
                render_snippet(
                    f,
                    modules[i].reference_span,
                    parent_src,
                    parent_path,
                    NOTE_STYLE,
                    false,
                    Some(&label),
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
    src: &str,
    path: &Path,
) -> std::fmt::Result {
    let Some(span) = error.span() else {
        writeln!(f, "{}: {}", paint(AnsiColor::Red, "error"), error)?;
        // Render per-entry spans for errors that carry their own locations
        if let DslError::Lower(Diagnostic {
            kind: LowerErrorKind::OverrideRuleNotFound(entries),
            ..
        }) = error
        {
            for (_, entry_span) in entries {
                render_snippet(f, *entry_span, src, path, ERROR_STYLE, false, None)?;
            }
        }
        return Ok(());
    };

    writeln!(f, "{}: {}", paint(AnsiColor::Red, "error"), error)?;
    render_snippet(f, span, src, path, ERROR_STYLE, true, None)?;

    if let Some(note) = error.note() {
        writeln!(f)?;
        let label = format!("note: {}", note.message);
        render_snippet(
            f,
            note.span,
            &note.source,
            &note.path,
            NOTE_STYLE,
            false,
            Some(&label),
        )?;
    }

    if let Some(trace) = error.call_trace() {
        let show = 3;
        let elided = trace.len().saturating_sub(show * 2);
        writeln!(f)?;
        writeln!(f, " {} call trace:", paint(AnsiColor::Cyan, "="))?;
        for (i, (name, path, line, col)) in trace.iter().enumerate() {
            if elided > 0 && i == show {
                writeln!(
                    f,
                    "   {} ... {elided} more frames ...",
                    paint(AnsiColor::Cyan, "...")
                )?;
            }
            if i >= show && i < trace.len().saturating_sub(show) {
                continue;
            }
            writeln!(
                f,
                "   {} {}:{line}:{col} in {name}()",
                paint(AnsiColor::Cyan, "-->"),
                path.display(),
            )?;
        }
    }

    Ok(())
}

/// Render a source snippet with a location header, source line, and underline.
fn render_snippet(
    f: &mut std::fmt::Formatter<'_>,
    span: Span,
    src: &str,
    path: &Path,
    style: (AnsiColor, char),
    show_prev_line: bool,
    label: Option<&str>,
) -> std::fmt::Result {
    let ctx = SpanContext::new(span, src);
    let gutter_width = digit_count(ctx.line_num);
    let pipe = paint(AnsiColor::Cyan, "|");
    let underline = paint(style.0, &style.1.to_string().repeat(ctx.underline_len));

    writeln!(
        f,
        " {} {}:{}:{}",
        paint(AnsiColor::Cyan, "-->"),
        path.display(),
        ctx.line_num,
        ctx.col,
    )?;
    writeln!(f, " {:>gutter_width$} {pipe}", "")?;
    if show_prev_line && let Some(prev_text) = ctx.prev_line_text {
        writeln!(
            f,
            " {} {pipe} {prev_text}",
            paint(AnsiColor::Cyan, &ctx.prev_line_num.to_string()),
        )?;
    }
    writeln!(
        f,
        " {} {pipe} {}",
        paint(AnsiColor::Cyan, &ctx.line_num.to_string()),
        ctx.line_text,
    )?;
    match label {
        Some(label) => write!(
            f,
            " {:>gutter_width$} {pipe} {:>width$}{underline} {}",
            "",
            "",
            paint(style.0, label),
            width = ctx.span_start_in_line,
        ),
        None => write!(
            f,
            " {:>gutter_width$} {pipe} {:>width$}{underline}",
            "",
            "",
            width = ctx.span_start_in_line,
        ),
    }
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

impl SpanContext<'_> {
    fn new(span: Span, source_text: &str) -> SpanContext<'_> {
        let offset = span.start as usize;
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
        let col = offset - line_start + 1;
        let line_text = &source_text[line_start..line_end];
        let span_start_in_line = col.saturating_sub(1);
        let span_len = (span.end - span.start) as usize;
        let underline_len = span_len
            .max(1)
            .min(line_text.len().saturating_sub(span_start_in_line));

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
        }
    }
}

const ERROR_STYLE: (AnsiColor, char) = (AnsiColor::Red, '^');
const NOTE_STYLE: (AnsiColor, char) = (AnsiColor::Cyan, '-');

fn digit_count(n: usize) -> usize {
    n.checked_ilog10().map_or(1, |d| d as usize + 1)
}

fn paint(color: AnsiColor, text: &str) -> String {
    let style = Style::new().fg_color(Some(color.into())).bold();
    format!("{style}{text}{style:#}")
}
