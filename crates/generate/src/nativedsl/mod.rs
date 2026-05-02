//! Native DSL front-end for tree-sitter grammar definitions.
//!
//! Pipeline: lex -> parse -> resolve+typecheck -> lower, producing an
//! [`InputGrammar`] equivalent to what `grammar.json` parsing produces.

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod lower;
pub mod parser;
pub mod serialize;
#[cfg(test)]
mod tests;
pub mod typecheck;

// Re-export errors and inner types for downstream consumers
pub use diagnostic::NativeDslError;
pub use lexer::LexErrorKind;
pub use lower::{DisallowedItemKind, LowerErrorKind, LowerResult};
pub use parser::ParseErrorKind;
pub use typecheck::{DataTy, InnerTy, ModuleTy, ScalarTy, Ty, TypeErrorKind};

pub type LexError = Diagnostic<LexErrorKind>;
pub type ParseError = Diagnostic<ParseErrorKind>;
pub type TypeError = Diagnostic<TypeErrorKind>;
pub type LowerError = Diagnostic<LowerErrorKind>;

/// Global module index. Every loaded module (root, inherited, imported)
/// gets a unique `ModuleId`. The cap is `u8::MAX`; exceeding it raises
/// `LowerErrorKind::ModuleTooMany`.
pub type ModuleId = u8;

use std::path::{Path, PathBuf};

use serde::Serialize;
use thiserror::Error;

use crate::grammars::InputGrammar;

use ast::{ModuleContext, Node, SharedAst, Span};
use lower::LoweringState;
use typecheck::TypeEnv;

fn resolve_path(path: &Path, span: Span) -> DslResult<PathBuf> {
    dunce::canonicalize(path).map_err(|e| {
        LowerError::new(
            LowerErrorKind::ModuleResolveFailed {
                path: path.to_path_buf(),
                error: e.to_string(),
            },
            span,
        )
        .into()
    })
}

/// Mutable pipeline state passed through the load/lower recursion. Bundled
/// to keep `load_module` / `load_child_module` argument lists tractable.
struct Loader<'a> {
    shared: &'a mut SharedAst,
    modules: &'a mut Vec<Module>,
    env: &'a mut TypeEnv,
    state: &'a mut LoweringState,
}

/// Parse a native DSL source file into an [`InputGrammar`].
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
    let mut loader = Loader {
        shared: &mut shared,
        modules: &mut modules,
        env: &mut env,
        state: &mut state,
    };
    loader.load_module(input, grammar_path, ModuleKind::Grammar, &[canonical])?;
    // Root is the last-pushed module by construction.
    let Module::Grammar { lowered, .. } = modules.pop().unwrap() else {
        unreachable!("root module is always loaded as Grammar");
    };
    Ok(*lowered)
}

#[derive(Clone, Copy)]
pub enum ModuleKind {
    /// Grammar file (root or inherited). Must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported). Only let/macro/import allowed.
    Helper,
}

impl Loader<'_> {
    /// Load a tsg module, imported either via `import` or `inherit`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the imported module isn't valid tsg source.
    ///
    /// # Panics
    ///
    /// Panics if `path` has no parent directory.
    fn load_module(
        &mut self,
        source: &str,
        path: &Path,
        kind: ModuleKind,
        ancestor_paths: &[PathBuf],
    ) -> DslResult<ModuleId> {
        if source.len() >= u32::MAX as usize {
            Err(LexError::without_span(LexErrorKind::InputTooLarge))?;
        }

        let module_dir = path.parent().unwrap();

        let tokens = lexer::Lexer::new(source).tokenize()?;
        let ctx = parser::Parser::new(&tokens, source.to_string(), path, self.shared).parse()?;

        match kind {
            ModuleKind::Grammar => validate_grammar(self.shared, &ctx)?,
            ModuleKind::Helper => validate_import_items(self.shared, &ctx)?,
        }

        // Load inherited grammar (Grammar kind only, must happen before typecheck).
        if let Some(inherit_id) = ctx.inherit_ref
            && let Node::ModuleRef {
                import: false,
                path: ref_path,
                ..
            } = *self.shared.arena.get(inherit_id)
        {
            let child_path = resolve_path(&module_dir.join(ctx.text(ref_path)), ref_path)?;
            let child_id =
                self.load_child_module(&child_path, ref_path, ancestor_paths, ModuleKind::Grammar)?;
            self.shared.arena.set(
                inherit_id,
                Node::ModuleRef {
                    import: false,
                    path: ref_path,
                    module: Some(child_id),
                },
            );
        }

        self.load_import_children(&ctx, module_dir, ancestor_paths)?;

        // Resolve identifiers + typecheck. Child modules already populated env
        // during their own load_module calls.
        let base = ctx
            .inherit_module(&self.shared.arena)
            .and_then(|(idx, span)| Some((self.modules[idx as usize].lowered()?, span)));
        typecheck::resolve_and_check(self.shared, &ctx, self.env, self.modules, base, path)?;

        let global_id = u8::try_from(self.modules.len())
            .map_err(|_| LowerError::without_span(LowerErrorKind::ModuleTooMany))?;
        let module = match kind {
            ModuleKind::Grammar => {
                let lowered = Box::new(lower::lower_with_base(
                    self.state,
                    self.shared,
                    self.modules,
                    &ctx,
                )?);
                Module::Grammar { ctx, lowered }
            }
            ModuleKind::Helper => Module::Helper { ctx },
        };
        self.modules.push(module);

        Ok(global_id)
    }
}

/// Validate: grammar block exists, inherits field consistency.
/// Multiple `inherit()` calls are caught by the parser.
fn validate_grammar(shared: &SharedAst, ctx: &ModuleContext) -> DslResult<()> {
    if ctx.grammar_config.is_none() {
        return Err(LowerError::without_span(LowerErrorKind::MissingGrammarBlock).into());
    }
    let config_inherits = ctx.grammar_config.as_ref().and_then(|c| c.inherits);

    // inherit() exists but no `inherits` in grammar config
    if let Some(inherit_ref) = ctx.inherit_ref
        && config_inherits.is_none()
    {
        Err(LowerError::new(
            LowerErrorKind::InheritWithoutConfig,
            shared.arena.span(inherit_ref),
        ))?;
    }

    // `inherits` in config but doesn't resolve to an inherit() call
    if let Some(id) = config_inherits
        && ctx.inherit_ref.is_none()
    {
        Err(LowerError::new(
            LowerErrorKind::InheritsWithoutInherit,
            shared.arena.span(id),
        ))?;
    }

    Ok(())
}

const MAX_MODULE_DEPTH: usize = 256;

impl Loader<'_> {
    fn load_child_module(
        &mut self,
        module_path: &Path,
        span: Span,
        ancestor_paths: &[PathBuf],
        kind: ModuleKind,
    ) -> DslResult<ModuleId> {
        if ancestor_paths.len() >= MAX_MODULE_DEPTH {
            return Err(LowerError::new(LowerErrorKind::ModuleDepthExceeded, span).into());
        }

        let content = std::fs::read_to_string(module_path).map_err(|e| {
            DslError::from(ModuleError {
                inner: Box::new(
                    LowerError::new(
                        LowerErrorKind::ModuleReadFailed {
                            path: module_path.to_path_buf(),
                            error: e.to_string(),
                        },
                        span,
                    )
                    .into(),
                ),
                source_text: String::new(),
                path: module_path.to_path_buf(),
                reference_span: span,
            })
        })?;

        if ancestor_paths.iter().any(|p| p == module_path) {
            return Err(ModuleError {
                inner: Box::new(LowerError::without_span(LowerErrorKind::ModuleCycle).into()),
                source_text: content,
                path: module_path.to_path_buf(),
                reference_span: span,
            })?;
        }

        let mut child_ancestors = ancestor_paths.to_vec();
        child_ancestors.push(module_path.to_path_buf());

        let module_id = self
            .load_module(&content, module_path, kind, &child_ancestors)
            .map_err(|inner| ModuleError {
                inner: Box::new(inner),
                source_text: content,
                path: module_path.to_path_buf(),
                reference_span: span,
            })?;

        Ok(module_id)
    }
}

/// A loaded and resolved module.
///
/// `Helper` modules come from `import(...)` and expose only let/macro bindings.
/// `Grammar` modules come from `inherit(...)` (or the root grammar) and carry
/// a fully lowered grammar for rule merging and `grammar_config` access. The
/// `lowered` field is boxed because `InputGrammar` is far larger than
/// `ModuleContext`, and most modules are `Helper`s.
pub enum Module {
    Helper {
        ctx: ModuleContext,
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
            Self::Helper { ctx } | Self::Grammar { ctx, .. } => ctx,
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

impl Loader<'_> {
    /// Resolve unresolved `ModuleRef` nodes (those still `module: None`),
    /// loading each child file. Deduplicates by canonical path.
    fn load_import_children(
        &mut self,
        ctx: &ModuleContext,
        module_dir: &Path,
        ancestor_paths: &[PathBuf],
    ) -> Result<(), DslError> {
        // Dedup cache: canonical path -> module index. Typically 0-3 entries,
        // so linear search beats hashing.
        let mut loaded: Vec<(PathBuf, ModuleId)> = Vec::new();

        for &node_id in &ctx.module_refs {
            let Node::ModuleRef {
                import: is_import,
                path: path_span,
                module: None,
            } = *self.shared.arena.get(node_id)
            else {
                // Already resolved (e.g. the inherit ref handled by the caller).
                continue;
            };
            let kind = if is_import {
                ModuleKind::Helper
            } else {
                ModuleKind::Grammar
            };

            let path_str = ctx.text(path_span);
            let canonical = resolve_path(&module_dir.join(path_str), path_span)?;

            let gid = if let Some(&(_, gid)) = loaded.iter().find(|(p, _)| p == &canonical) {
                gid
            } else {
                let gid = self.load_child_module(&canonical, path_span, ancestor_paths, kind)?;
                loaded.push((canonical, gid));
                gid
            };

            self.shared.arena.set(
                node_id,
                Node::ModuleRef {
                    import: is_import,
                    path: path_span,
                    module: Some(gid),
                },
            );
        }

        Ok(())
    }
}

/// Validate that an imported file only contains allowed items.
fn validate_import_items(shared: &SharedAst, ctx: &ModuleContext) -> DslResult<()> {
    for &item_id in &ctx.root_items {
        let kind = match shared.arena.get(item_id) {
            Node::Grammar => DisallowedItemKind::GrammarBlock,
            Node::Rule { is_override, .. } => {
                if *is_override {
                    DisallowedItemKind::OverrideRule
                } else {
                    DisallowedItemKind::Rule
                }
            }
            _ => continue,
        };
        Err(LowerError::new(
            LowerErrorKind::ModuleDisallowedItem(kind),
            shared.arena.span(item_id),
        ))?;
    }
    Ok(())
}

pub type DslResult<T> = Result<T, DslError>;

/// Diagnostic error shared by all pipeline stages.
#[derive(Debug, Serialize)]
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

impl<K: std::fmt::Debug + std::fmt::Display> std::error::Error for Diagnostic<K> {}

#[derive(Debug, Error, Serialize)]
pub enum DslError {
    #[error(transparent)]
    Lex(#[from] LexError),
    #[error(transparent)]
    Parse(#[from] ParseError),
    #[error(transparent)]
    Type(#[from] TypeError),
    #[error(transparent)]
    Lower(#[from] LowerError),
    #[error(transparent)]
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
            Self::Type(e) => e.note.as_deref(),
            Self::Lower(e) => e.note.as_deref(),
            Self::Module(e) => e.inner.note(),
        }
    }
}
