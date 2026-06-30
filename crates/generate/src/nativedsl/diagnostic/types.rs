//! Error types shared by every native-DSL pipeline stage.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::nativedsl::{
    ast::Span, expand_macro_calls::ExpandErrorKind, lexer::LexErrorKind, lower::LowerErrorKind,
    parser::ParseErrorKind, resolve::ResolveErrorKind, typecheck::TypeErrorKind,
};

pub type DslResult<T> = Result<T, DslError>;

pub type LexError = Diagnostic<LexErrorKind>;
pub type ParseError = Diagnostic<ParseErrorKind>;
pub type ResolveError = Diagnostic<ResolveErrorKind>;
pub type TypeError = Diagnostic<TypeErrorKind>;
pub type LowerError = Diagnostic<LowerErrorKind>;
pub type ExpandError = Diagnostic<ExpandErrorKind>;

/// Diagnostic error shared by all pipeline stages.
#[derive(Debug, Serialize, Deserialize, Error)]
pub struct Diagnostic<K> {
    pub kind: K,
    pub span: Option<Span>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<Note>,
    /// Source + path the primary `span` indexes into, when that differs from the
    /// module the error surfaces in (a lower error born evaluating an imported
    /// module's macro body). `None` uses the caller-supplied source.
    #[serde(skip)]
    pub src: Option<Box<(String, PathBuf)>>,
}

impl<K> Diagnostic<K> {
    pub const fn new(kind: K, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
            notes: Vec::new(),
            src: None,
        }
    }

    pub const fn without_span(kind: K) -> Self {
        Self {
            kind,
            span: None,
            notes: Vec::new(),
            src: None,
        }
    }

    pub fn with_note(kind: K, span: Span, note: Note) -> Self {
        Self {
            kind,
            span: Some(span),
            notes: vec![note],
            src: None,
        }
    }

    /// Stamp the source + path the primary span belongs to, so a cross-module
    /// error renders against the right file. See [`Diagnostic::src`].
    #[must_use]
    pub fn with_source(mut self, source: &str, path: &Path) -> Self {
        self.src = Some(Box::new((source.to_owned(), path.to_owned())));
        self
    }

    pub fn add_note(&mut self, note: Note) {
        self.notes.push(note);
    }
}

impl<K: std::fmt::Display> std::fmt::Display for Diagnostic<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}

#[derive(Debug, Error, Serialize, Deserialize)]
#[error(transparent)]
pub enum DslError {
    Lex(#[from] LexError),
    Parse(#[from] ParseError),
    Expand(#[from] ExpandError),
    Resolve(#[from] ResolveError),
    Type(#[from] TypeError),
    Lower(#[from] LowerError),
    Module(#[from] ModuleError),
}

/// Error from loading a child module (inherited or imported).
#[derive(Debug, Serialize, Deserialize, Error)]
#[error("{inner}")]
pub struct ModuleError {
    pub inner: Box<DslError>,
    #[serde(skip)]
    pub source_text: String,
    #[serde(skip)]
    pub path: PathBuf,
    pub reference_span: Span,
}

impl ModuleError {
    #[must_use]
    pub fn new(inner: DslError, source_text: String, path: &Path, reference_span: Span) -> Self {
        Self {
            inner: Box::new(inner),
            source_text,
            path: path.to_path_buf(),
            reference_span,
        }
    }
}

/// Secondary annotation on an error, pointing to a related source location.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    pub message: NoteMessage,
    pub span: Span,
    #[serde(skip)]
    pub path: PathBuf,
    #[serde(skip)]
    pub src: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoteMessage {
    FirstDefinedHere,
    ReferencedFromHere,
    OverrideDeclaredHere,
    /// One per redundant `inherit()` beyond the first in a `MultipleInherits` error
    AlsoInheritedHere,
    /// Points at an `expect` forward-decl whose promised symbol was never defined.
    ForwardDeclaredHere,
    /// Carries the cfg flag name, the note's span points at the gated decl.
    GatedByDisabledCfg(String),
    DidYouMean(String),
    /// Emitted alongside `GrammarConfigRequiresInherit`. Carries the imported path string.
    SwitchImportToInherit(String),
    SelfReferenceHere,
    PrecNeedsExplicitPrecedence {
        is_left: bool,
        arg: String,
    },
}

impl std::fmt::Display for NoteMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstDefinedHere => write!(f, "first defined here"),
            Self::ReferencedFromHere => write!(f, "referenced from here"),
            Self::OverrideDeclaredHere => write!(f, "override declared here"),
            Self::AlsoInheritedHere => write!(f, "also inherited here"),
            Self::ForwardDeclaredHere => write!(f, "forward-declared with `expect` here"),
            Self::GatedByDisabledCfg(flag) => {
                write!(f, "this declaration is disabled by `#[cfg({flag})]`")
            }
            Self::DidYouMean(name) => write!(f, "did you mean `{name}`?"),
            Self::SwitchImportToInherit(path) => {
                write!(f, "switch `import(\"{path}\")` to `inherit(\"{path}\")`")
            }
            Self::SelfReferenceHere => write!(f, "self-reference here"),
            Self::PrecNeedsExplicitPrecedence { is_left, arg } => {
                let name = if *is_left { "prec_left" } else { "prec_right" };
                write!(
                    f,
                    "{name} requires a precedence, did you mean `{name}(0, {arg})`?"
                )
            }
        }
    }
}

impl DslError {
    #[must_use]
    pub const fn span(&self) -> Option<Span> {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Expand(e) => e.span,
            Self::Resolve(e) => e.span,
            Self::Type(e) => e.span,
            Self::Lower(e) => e.span,
            Self::Module(e) => Some(e.reference_span),
        }
    }

    #[must_use]
    pub fn call_trace(&self) -> Option<&[(String, PathBuf, usize, usize)]> {
        if let Self::Lower(e) = self
            && let LowerErrorKind::CallDepthExceeded(trace) = &e.kind
        {
            Some(trace)
        } else {
            None
        }
    }

    #[must_use]
    pub fn notes(&self) -> &[Note] {
        match self {
            Self::Lex(e) => &e.notes,
            Self::Parse(e) => &e.notes,
            Self::Expand(e) => &e.notes,
            Self::Resolve(e) => &e.notes,
            Self::Type(e) => &e.notes,
            Self::Lower(e) => &e.notes,
            Self::Module(e) => e.inner.notes(),
        }
    }

    /// Source + path the primary span belongs to, when the error carries one (a
    /// lower error from an imported module's macro body). Lets the renderer put
    /// the caret in that module's file instead of the root's.
    #[must_use]
    pub fn primary_source(&self) -> Option<(&str, &Path)> {
        let src = match self {
            Self::Lex(e) => &e.src,
            Self::Parse(e) => &e.src,
            Self::Expand(e) => &e.src,
            Self::Resolve(e) => &e.src,
            Self::Type(e) => &e.src,
            Self::Lower(e) => &e.src,
            Self::Module(e) => return e.inner.primary_source(),
        };
        src.as_deref().map(|(s, p)| (s.as_str(), p.as_path()))
    }
}
