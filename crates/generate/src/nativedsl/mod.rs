//! Native DSL front-end for tree-sitter grammar definitions.
//!
//! Pipeline: lex -> parse -> resolve+typecheck -> lower, producing an
//! [`InputGrammar`] equivalent to what `grammar.json` parsing produces.

/// Save the length of one or more `Vec`s used as stacks, run a body, then
/// truncate each back. The body is wrapped in a closure so `?` inside
/// short-circuits the closure rather than the outer function, letting the
/// truncates always run.
macro_rules! stack_scope {
    ($buf:expr, |$base:ident| $body:expr) => {{
        let $base = $buf.len();
        let result = {
            #[allow(clippy::redundant_closure_call, reason = "IIFE scopes `?` to the closure so truncate runs")]
            (|| $body)()
        };
        $buf.truncate($base);
        result
    }};
    ($($buf:expr => $base:ident),+; $body:expr) => {{
        $(let $base = $buf.len();)+
        let result = {
            #[allow(clippy::redundant_closure_call, reason = "IIFE scopes `?` to the closure so truncates run")]
            (|| $body)()
        };
        $($buf.truncate($base);)+
        result
    }};
}

/// Extract a node matching the given pattern from the expression
///
/// # Panics
///
/// Panics if the expression does not match the given pattern.
macro_rules! expect_pat {
    ($pat:pat, $expr:expr $(,)?) => {
        let $pat = $expr else {
            panic!("Expected {}, got {:?}", stringify!($pat), $expr);
        };
    };
}

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod loader;
pub mod lower;
pub mod parser;
pub mod resolve;
pub mod serialize;
#[cfg(test)]
mod tests;
pub mod typecheck;

pub use crate::grammars::InputGrammar;
pub use diagnostic::NativeDslError;
pub use lexer::{LexErrorKind, LexResult};
pub use lower::{DisallowedItemKind, LowerErrorKind, LowerResult, LoweringState};
pub use parser::{ParseErrorKind, ParseResult};
pub use resolve::{ResolveErrorKind, ResolveResult};
pub use typecheck::{
    Constraint, ContainerKind, DataTy, InnerTy, ModuleTy, ScalarTy, Ty, TypeErrorKind, TypeResult,
};

use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::rules::Rule;

use ast::{ModuleContext, SharedAst, Span};
use loader::Loader;
use typecheck::TypeEnv;

/// Global module index. Every loaded module gets a unique `ModuleId`.
pub type ModuleId = u8;

/// A loaded and resolved module.
///
/// `Helper` modules come from `import(...)` and expose let/macro/rule/external
/// bindings. Their rules are lowered eagerly into `lowered_rules` so importers
/// can materialize them by bare name or inline via `helper::rule_name`.
/// `Grammar` modules come from `inherit(...)` (or the root grammar) and carry
/// a fully lowered grammar for rule merging and `grammar_config` access. The
/// `lowered` field is boxed because `InputGrammar` is far larger than
/// `ModuleContext`, and most modules are `Helper`s.
#[derive(Debug)]
pub enum Module {
    Helper {
        ctx: ModuleContext,
        lowered_rules: Vec<(String, Rule)>,
    },
    Grammar {
        ctx: ModuleContext,
        lowered: Box<InputGrammar>,
    },
}

impl Module {
    #[must_use]
    pub const fn ctx(&self) -> &ModuleContext {
        match self {
            Self::Helper { ctx, .. } | Self::Grammar { ctx, .. } => ctx,
        }
    }

    #[must_use]
    pub fn lowered(&self) -> Option<&InputGrammar> {
        match self {
            Self::Grammar { lowered, .. } => Some(lowered),
            Self::Helper { .. } => None,
        }
    }
}

/// Walk the import graph BFS from `initial_refs`, calling `f` for each
/// transitively-reachable helper module exactly once. The `Span` argument
/// is the span of the original import statement that brought the helper
/// into scope (used for "first defined here" notes when registering
/// helper rule names).
pub(super) fn for_each_imported_helper<'a, E>(
    arena: &ast::NodeArena,
    initial_refs: &[ast::NodeId],
    modules: &'a [Module],
    mut f: impl FnMut(ModuleId, &'a Module, Span) -> Result<(), E>,
) -> Result<(), E> {
    let mut visited: rustc_hash::FxHashSet<u8> = rustc_hash::FxHashSet::default();
    let mut stack: Vec<(u8, Span)> = Vec::new();

    let seed = |stack: &mut Vec<(u8, Span)>, refs: &[ast::NodeId]| {
        for &mref_id in refs {
            if let &ast::Node::ModuleRef {
                import: true,
                module: Some(idx),
                ..
            } = arena.get(mref_id)
            {
                stack.push((idx, arena.span(mref_id)));
            }
        }
    };
    seed(&mut stack, initial_refs);

    while let Some((idx, ref_span)) = stack.pop() {
        if !visited.insert(idx) {
            continue;
        }
        let module = &modules[usize::from(idx)];
        if matches!(module, Module::Helper { .. }) {
            f(idx, module, ref_span)?;
            seed(&mut stack, &module.ctx().module_refs);
        }
    }
    Ok(())
}

/// Entry point. Parse a native DSL source file into an [`InputGrammar`].
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails, or if `grammar_path`
/// cannot be canonicalized.
pub fn parse_native_dsl(input: &str, grammar_path: &Path) -> DslResult<InputGrammar> {
    let canonical = dunce::canonicalize(grammar_path).map_err(|e| {
        LowerError::without_span(LowerErrorKind::ModuleResolveFailed {
            path: grammar_path.to_path_buf(),
            error: e.to_string(),
        })
    })?;
    let cap = input.len() / 30;
    let mut shared = SharedAst::new(cap);
    let mut modules: Vec<Module> = Vec::new();
    let mut env = TypeEnv::default();
    let mut state = LoweringState::default();
    let mut dsl_loader = Loader {
        shared: &mut shared,
        modules: &mut modules,
        env: &mut env,
        state: &mut state,
        ancestor_paths: vec![canonical.clone()],
        loaded: Vec::new(),
    };
    dsl_loader.load_module(input, &canonical, loader::ModuleKind::Grammar)?;
    // Root is the last-pushed module by construction.
    expect_pat!(Some(Module::Grammar { lowered, .. }), modules.pop());
    Ok(*lowered)
}

pub type DslResult<T> = Result<T, DslError>;

pub type LexError = Diagnostic<LexErrorKind>;
pub type ParseError = Diagnostic<ParseErrorKind>;
pub type ResolveError = Diagnostic<ResolveErrorKind>;
pub type TypeError = Diagnostic<TypeErrorKind>;
pub type LowerError = Diagnostic<LowerErrorKind>;

/// Diagnostic error shared by all pipeline stages.
#[derive(Debug, Serialize, Error)]
pub struct Diagnostic<K> {
    pub kind: K,
    pub span: Option<Span>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<Box<Note>>,
}

impl<K> Diagnostic<K> {
    pub const fn new(kind: K, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
            note: None,
        }
    }

    pub const fn without_span(kind: K) -> Self {
        Self {
            kind,
            span: None,
            note: None,
        }
    }

    pub fn with_note(kind: K, span: Span, note: Note) -> Self {
        Self {
            kind,
            span: Some(span),
            note: Some(Box::new(note)),
        }
    }
}

impl<K: std::fmt::Display> std::fmt::Display for Diagnostic<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}

#[derive(Debug, Error, Serialize)]
#[error(transparent)]
pub enum DslError {
    Lex(#[from] LexError),
    Parse(#[from] ParseError),
    Resolve(#[from] ResolveError),
    Type(#[from] TypeError),
    Lower(#[from] LowerError),
    Module(#[from] ModuleError),
}

/// Error from loading a child module (inherited or imported).
#[derive(Debug, Serialize, Error)]
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
#[derive(Clone, Debug, Serialize)]
pub struct Note {
    pub message: NoteMessage,
    pub span: Span,
    #[serde(skip)]
    pub path: PathBuf,
    #[serde(skip)]
    pub source: String,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize)]
pub enum NoteMessage {
    FirstDefinedHere,
    ReferencedFromHere,
    DefinedLater,
}

impl std::fmt::Display for NoteMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstDefinedHere => write!(f, "first defined here"),
            Self::ReferencedFromHere => write!(f, "referenced from here"),
            Self::DefinedLater => write!(f, "let binding defined here (move it before the usage)"),
        }
    }
}

impl DslError {
    #[must_use]
    pub const fn span(&self) -> Option<Span> {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Resolve(e) => e.span,
            Self::Type(e) => e.span,
            Self::Lower(e) => e.span,
            Self::Module(e) => Some(e.reference_span),
        }
    }

    #[must_use]
    pub fn call_trace(&self) -> Option<&[(String, PathBuf, usize, usize)]> {
        if let Self::Lower(e) = self
            && let lower::LowerErrorKind::CallDepthExceeded(trace) = &e.kind
        {
            Some(trace)
        } else {
            None
        }
    }

    #[must_use]
    pub fn note(&self) -> Option<&Note> {
        match self {
            Self::Lex(e) => e.note.as_deref(),
            Self::Parse(e) => e.note.as_deref(),
            Self::Resolve(e) => e.note.as_deref(),
            Self::Type(e) => e.note.as_deref(),
            Self::Lower(e) => e.note.as_deref(),
            Self::Module(e) => e.inner.note(),
        }
    }
}
