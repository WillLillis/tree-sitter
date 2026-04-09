//! Native DSL front-end for tree-sitter grammar definitions.
//!
//! This module implements a five-stage pipeline that compiles `.tsg` source
//! files into the same [`InputGrammar`] representation produced by parsing
//! `grammar.json`:
//!
//! 1. **Lexing** ([`lexer`]) - tokenizes source text into a flat `Vec<Token>`.
//! 2. **Parsing** ([`parser`]) - builds an arena-based AST from the token stream.
//! 3. **Name resolution** ([`resolve`]) - resolves identifiers to rule/variable references.
//! 4. **Type checking** ([`typecheck`]) - validates types across expressions and function calls.
//! 5. **Lowering** ([`lower`]) - evaluates the AST into an [`InputGrammar`].
//!
//! Each stage produces structured, serializable errors that carry source spans.

pub mod ast;
mod lexer;
pub mod lower;
mod parser;
mod resolve;
#[cfg(test)]
mod tests;
mod typecheck;

use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;

use ast::Span;

pub use lexer::{LexError, LexErrorKind};
pub use lower::{LowerError, LowerErrorKind};
pub use parser::{ParseError, ParseErrorKind};
pub use resolve::{ResolveError, ResolveErrorKind};
pub use typecheck::{Ty, TypeError, TypeErrorKind};

use serde::Serialize;
use thiserror::Error;

/// Parse a native DSL source file into an [`InputGrammar`].
///
/// Runs the full pipeline: lex, parse, resolve, typecheck, lower.
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails. The error carries the
/// source [`Span`] of the offending construct.
pub fn parse_native_dsl(input: &str, grammar_path: &Path) -> Result<InputGrammar, DslError> {
    parse_native_dsl_inner(input, grammar_path, &[])
}

fn parse_native_dsl_inner(
    input: &str,
    grammar_path: &Path,
    ancestor_paths: &[PathBuf],
) -> Result<InputGrammar, DslError> {
    if input.len() >= u32::MAX as usize {
        return Err(LexError {
            kind: LexErrorKind::InputTooLarge,
            span: Span::new(0, 0),
        }
        .into());
    }

    let grammar_dir = grammar_path.parent().unwrap();

    let tokens = lexer::Lexer::new(input).tokenize()?;
    let ast = parser::Parser::new(tokens, input, grammar_path.to_path_buf()).parse()?;

    // Load base grammar before resolve so we can register its rule names
    let base = load_base_grammar(&ast, grammar_dir, ancestor_paths)?;
    let base_rule_names: Vec<String> = base
        .as_ref()
        .map(|g| g.variables.iter().map(|v| v.name.clone()).collect())
        .unwrap_or_default();

    // Find the inherit statement span for "first defined here" notes
    let inherit_span = ast.root_items.iter().find_map(|&id| {
        if let ast::Node::Let { value, .. } = ast.node(id)
            && matches!(ast.node(*value), ast::Node::Inherit { .. })
        {
            return Some(ast.span(id));
        }
        None
    });

    let mut ast = ast;
    resolve::resolve(&mut ast, &base_rule_names, inherit_span, grammar_path)?;
    typecheck::check(&ast)?;
    Ok(lower::lower_with_base(&ast, base)?)
}

/// Scan root items for `let _ = inherit("path")`, load the base grammar if found.
fn load_base_grammar(
    ast: &ast::Ast<'_>,
    grammar_dir: &Path,
    ancestor_paths: &[PathBuf],
) -> Result<Option<InputGrammar>, DslError> {
    use lower::{LowerError, LowerErrorKind};

    for &item_id in &ast.root_items {
        if let ast::Node::Let { value, .. } = ast.node(item_id)
            && let ast::Node::Inherit { path } = ast.node(*value)
        {
            let path_str = ast.string_lit_content(*path);
            let span = ast.span(*path);
            let full_path = grammar_dir.join(path_str);
            let canonical = dunce::canonicalize(&full_path).map_err(|e| {
                LowerError::new(
                    LowerErrorKind::InheritLoadError(format!(
                        "failed to resolve '{}': {e}",
                        full_path.display()
                    )),
                    span,
                )
            })?;

            // Cycle detection
            if ancestor_paths.iter().any(|p| p == &canonical) {
                let mut chain: Vec<PathBuf> = ancestor_paths.to_vec();
                chain.push(canonical);
                return Err(LowerError::new(LowerErrorKind::InheritCycle(chain), span).into());
            }

            let content = std::fs::read_to_string(&full_path).map_err(|e| {
                LowerError::new(
                    LowerErrorKind::InheritLoadError(format!(
                        "failed to read '{}': {e}",
                        full_path.display()
                    )),
                    span,
                )
            })?;

            let mut child_ancestors = ancestor_paths.to_vec();
            child_ancestors.push(canonical.clone());

            let ext = full_path.extension().and_then(|e| e.to_str());
            let grammar = match ext {
                Some("tsg") => parse_native_dsl_inner(&content, &canonical, &child_ancestors)
                    .map_err(|inner| InheritedError {
                        inner: Box::new(inner),
                        source_text: content.clone(),
                        path: canonical.clone(),
                        inherit_span: span,
                    })?,
                Some("json") => crate::parse_grammar::parse_grammar(&content).map_err(|e| {
                    LowerError::new(
                        LowerErrorKind::InheritLoadError(format!(
                            "in '{}': {e}",
                            full_path.display()
                        )),
                        span,
                    )
                })?,
                _ => {
                    return Err(LowerError::new(
                        LowerErrorKind::InheritLoadError(
                            "unsupported file extension, expected .tsg or .json".to_string(),
                        ),
                        span,
                    )
                    .into());
                }
            };

            return Ok(Some(grammar));
        }
    }
    Ok(None)
}

#[derive(Debug, Error, Serialize)]
pub enum DslError {
    #[error(transparent)]
    Lex(#[from] LexError),
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Resolve(#[from] ResolveError),
    #[error(transparent)]
    Type(#[from] TypeError),
    #[error(transparent)]
    Lower(#[from] LowerError),
    #[error(transparent)]
    Inherited(#[from] InheritedError),
}

#[derive(Debug, Serialize, Error)]
#[error("{inner}")]
pub struct InheritedError {
    pub inner: Box<DslError>,
    #[serde(skip)]
    pub source_text: String,
    #[serde(skip)]
    pub path: PathBuf,
    pub inherit_span: Span,
}

use ast::Note;

impl DslError {
    #[must_use]
    pub const fn span(&self) -> Span {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Resolve(e) => e.span,
            Self::Type(e) => e.span,
            Self::Lower(e) => e.span,
            Self::Inherited(e) => e.inherit_span,
        }
    }

    #[must_use]
    pub fn note(&self) -> Option<&Note> {
        match self {
            Self::Parse(e) => e.note.as_ref(),
            Self::Resolve(e) => e.note.as_ref(),
            Self::Lower(e) => e.note.as_ref(),
            Self::Inherited(e) => e.inner.note(),
            _ => None,
        }
    }
}

/// A [`DslError`] bundled with source text and file path for diagnostic rendering.
#[derive(Debug, Error, Serialize)]
#[error("{}", render_diagnostic(error, source_text, path))]
pub struct NativeDslError {
    pub error: DslError,
    #[serde(skip)]
    pub source_text: String,
    #[serde(skip)]
    pub path: PathBuf,
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

    let prev_line_text = prev_line_start.map(|ps| {
        let pe = memchr::memchr(b'\n', &bytes[ps..]).map_or(bytes.len(), |pos| ps + pos);
        &source_text[ps..pe]
    });

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
    use std::fmt::Write;

    use anstyle::AnsiColor;

    // Inherited errors: render the inner error with the inherited file's context,
    // then append a "inherited from here" note pointing to the parent.
    if let DslError::Inherited(e) = error {
        let mut out = render_diagnostic(&e.inner, &e.source_text, &e.path);
        render_note_snippet(
            &mut out,
            e.inherit_span,
            source_text,
            path,
            &ast::NoteMessage::InheritedFromHere,
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

    // Render note if present (uses note's own source/path for cross-file notes)
    if let Some(note) = error.note() {
        render_note_snippet(&mut out, note.span, &note.source, &note.path, &note.message);
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
    use std::fmt::Write;

    use anstyle::AnsiColor;

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
    if n == 0 {
        return 1;
    }
    ((n as f64).log10().floor() as usize) + 1
}
