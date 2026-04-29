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
pub use typecheck::{InnerTy, Ty, TypeErrorKind};

pub type LexError = Diagnostic<LexErrorKind>;
pub type ParseError = Diagnostic<ParseErrorKind>;
pub type TypeError = Diagnostic<TypeErrorKind>;
pub type LowerError = Diagnostic<LowerErrorKind>;

use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use crate::grammars::InputGrammar;

use ast::{IdentKind, ModuleContext, Node, NodeId, SharedAst, Span};
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

/// Parse a native DSL source file into an [`InputGrammar`].
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails.
///
/// # Panics
///
/// Panics if `grammar_path` cannot be canonicalized.
pub fn parse_native_dsl(input: &str, grammar_path: &Path) -> DslResult<InputGrammar> {
    let canonical = dunce::canonicalize(grammar_path)
        .expect("grammar path should be canonicalizable (file was already read)");
    let cap = input.len() / 30;
    let mut shared = SharedAst::new(cap);
    let mut modules: Vec<Module> = Vec::new();
    let root_id = load_module(
        &mut shared,
        &mut modules,
        input,
        grammar_path,
        ModuleKind::Grammar,
        &[canonical],
    )?;
    Ok(modules[root_id as usize]
        .lowered
        .take()
        .expect("Grammar module always has lowered grammar"))
}

#[derive(Clone, Copy)]
pub enum ModuleKind {
    /// Grammar file (root or inherited). Must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported). Only let/fn/import allowed.
    Helper,
}

/// Loads a tsg module, imported either via `import` or `inherit`
///
/// # Errors
///
/// Returns `Err` if the imported module isn't valid tsg source.
///
/// # Panics
///
/// Panics if `path` has no parent directory.
pub fn load_module(
    shared: &mut SharedAst,
    modules: &mut Vec<Module>,
    source: &str,
    path: &Path,
    kind: ModuleKind,
    ancestor_paths: &[PathBuf],
) -> DslResult<u8> {
    if source.len() >= u32::MAX as usize {
        Err(LexError::without_span(LexErrorKind::InputTooLarge))?;
    }

    let module_dir = path.parent().unwrap();

    let tokens = lexer::Lexer::new(source).tokenize()?;
    // Record node range for this module so load_import_children only scans
    // nodes belonging to this module (not parent or child modules).
    let node_start = shared.arena.len() + 1;
    let ctx = parser::Parser::new(&tokens, source.to_string(), path, shared).parse()?;
    let node_end = shared.arena.len();

    let inherit_node = match kind {
        ModuleKind::Grammar => {
            let node = find_inherit_node(shared, &ctx);
            validate_grammar(shared, &ctx, node)?;
            node
        }
        ModuleKind::Helper => {
            validate_import_items(shared, &ctx)?;
            None
        }
    };

    // Load inherited grammar (Grammar kind only, must happen before typecheck).
    if let Some(inherit_id) = inherit_node {
        let Node::ModuleRef {
            import: false,
            path: ref_path,
            ..
        } = *shared.arena.get(inherit_id)
        else {
            unreachable!()
        };
        let child_id = load_child_module(
            shared,
            modules,
            ctx.text(ref_path),
            module_dir,
            ref_path,
            ancestor_paths,
            ModuleKind::Grammar,
        )?;
        shared.arena.set(
            inherit_id,
            Node::ModuleRef {
                import: false,
                path: ref_path,
                module: Some(child_id),
            },
        );
    }

    load_import_children(
        shared,
        &ctx,
        module_dir,
        ancestor_paths,
        modules,
        node_start..=node_end,
    )?;

    // Resolve identifiers + typecheck. For Grammar modules, also lower.
    let mut env = TypeEnv::default();
    typecheck_modules(shared, modules, &mut env)?;
    let base = inherit_node.and_then(|id| {
        let module = modules.iter().find(|m| m.is_grammar())?;
        Some((module.lowered.as_ref()?, shared.arena.span(id)))
    });
    typecheck::resolve_and_check(shared, &ctx, &mut env, modules, base, path)?;

    let global_id = u8::try_from(modules.len())
        .map_err(|_| LowerError::without_span(LowerErrorKind::ModuleTooMany))?;
    let lowered = if matches!(kind, ModuleKind::Grammar) {
        Some(lower::lower_with_base(shared, &ctx, modules, global_id)?)
    } else {
        None
    };
    modules.push(Module {
        ctx,
        lowered,
        global_id,
    });

    Ok(global_id)
}

/// Validate: grammar block exists, at most one `inherit()`, inherits field consistency.
pub fn validate_grammar(
    shared: &SharedAst,
    ctx: &ModuleContext,
    inherit_node: Option<NodeId>,
) -> DslResult<()> {
    if ctx.grammar_config.is_none() {
        return Err(LowerError::without_span(LowerErrorKind::MissingGrammarBlock).into());
    }
    let config_inherits = ctx.grammar_config.as_ref().and_then(|c| c.inherits);

    let direct_in_config = config_inherits
        .is_some_and(|id| matches!(shared.arena.get(id), Node::ModuleRef { import: false, .. }));

    // Find all inherit() calls in let bindings, error if more than one
    let mut inherit_let: Option<Span> = None;
    for &item_id in &ctx.root_items {
        if let Node::Let { value, .. } = shared.arena.get(item_id)
            && matches!(
                shared.arena.get(*value),
                Node::ModuleRef { import: false, .. }
            )
        {
            if inherit_let.is_some() || direct_in_config {
                Err(LowerError::new(
                    LowerErrorKind::MultipleInherits,
                    shared.arena.span(item_id),
                ))?;
            }
            inherit_let = Some(shared.arena.span(item_id));
        }
    }

    // inherit() in a let binding but no `inherits` in grammar config
    if let Some(span) = inherit_let
        && config_inherits.is_none()
    {
        Err(LowerError::new(LowerErrorKind::InheritWithoutConfig, span))?;
    }

    // `inherits` in config but doesn't resolve to an inherit() call
    if let Some(id) = config_inherits
        && inherit_node.is_none()
    {
        Err(LowerError::new(
            LowerErrorKind::InheritsWithoutInherit,
            shared.arena.span(id),
        ))?;
    }

    Ok(())
}

/// Resolve the inherit node from the `inherits` config field, following
/// let-binding indirection if needed.
fn find_inherit_node(shared: &SharedAst, ctx: &ModuleContext) -> Option<NodeId> {
    let inherits_id = ctx.grammar_config.as_ref()?.inherits?;
    match shared.arena.get(inherits_id) {
        Node::ModuleRef { import: false, .. } => Some(inherits_id),
        Node::Ident(IdentKind::Unresolved) => {
            let name = ctx.text(shared.arena.span(inherits_id));
            ctx.root_items.iter().find_map(|&item_id| {
                if let Node::Let { name: n, value, .. } = shared.arena.get(item_id)
                    && ctx.text(*n) == name
                    && matches!(
                        shared.arena.get(*value),
                        Node::ModuleRef { import: false, .. }
                    )
                {
                    Some(*value)
                } else {
                    None
                }
            })
        }
        _ => None,
    }
}

/// Load a child module from a path string. Handles .tsg and .json extensions,
/// cycle detection, and error wrapping.
const MAX_MODULE_DEPTH: usize = 256;

fn load_child_module(
    shared: &mut SharedAst,
    modules: &mut Vec<Module>,
    path_str: &str,
    module_dir: &Path,
    span: Span,
    ancestor_paths: &[PathBuf],
    kind: ModuleKind,
) -> DslResult<u8> {
    if ancestor_paths.len() >= MAX_MODULE_DEPTH {
        return Err(LowerError::new(LowerErrorKind::ModuleDepthExceeded, span).into());
    }

    let full_path = module_dir.join(path_str);
    let canonical = resolve_path(&full_path, span)?;

    let content = std::fs::read_to_string(&full_path).map_err(|e| {
        LowerError::new(
            LowerErrorKind::ModuleReadFailed {
                path: full_path.clone(),
                error: e.to_string(),
            },
            span,
        )
    })?;

    if ancestor_paths.iter().any(|p| p == &canonical) {
        #[expect(
            clippy::needless_return_with_question_mark,
            reason = "bad move semantics"
        )]
        return Err(ModuleError {
            inner: Box::new(LowerError::without_span(LowerErrorKind::ModuleCycle).into()),
            source_text: content,
            path: canonical,
            reference_span: span,
        })?;
    }

    let mut child_ancestors = ancestor_paths.to_vec();
    child_ancestors.push(canonical.clone());

    let ext = canonical.extension().and_then(|e| e.to_str());
    let module_id = match ext {
        Some("tsg") => load_module(
            shared,
            modules,
            &content,
            &canonical,
            kind,
            &child_ancestors,
        )
        .map_err(|inner| ModuleError {
            inner: Box::new(inner),
            source_text: content,
            path: canonical,
            reference_span: span,
        })?,
        _ => return Err(LowerError::new(LowerErrorKind::ModuleUnsupportedExtension, span).into()),
    };

    Ok(module_id)
}

/// A loaded and resolved module (imported or inherited), ready for
/// typechecking and lowering.
pub struct Module {
    pub ctx: ModuleContext,
    /// For inherited grammar modules: the fully lowered grammar, needed
    /// for rule merging and config access. `None` for import-only modules.
    pub lowered: Option<InputGrammar>,
    /// Unique ID across the entire module tree, assigned during loading.
    pub global_id: u8,
}

impl Module {
    /// Whether this module is a grammar (inherited) rather than a helper (imported).
    #[must_use]
    pub const fn is_grammar(&self) -> bool {
        self.lowered.is_some()
    }
}

/// Scan AST for unresolved module refs, load files, and set indices.
/// Deduplicates: same file imported multiple times is loaded once.
fn load_import_children(
    shared: &mut SharedAst,
    ctx: &ModuleContext,
    module_dir: &Path,
    ancestor_paths: &[PathBuf],
    modules: &mut Vec<Module>,
    node_range: std::ops::RangeInclusive<usize>,
) -> Result<(), DslError> {
    // Cache: canonical path -> global module ID, so duplicate imports share one load.
    let mut loaded: FxHashMap<PathBuf, u8> = FxHashMap::default();

    // Only scan nodes belonging to this module (not parent or child modules).
    let mut node_id = NodeId::from_index(*node_range.start());
    let end = *node_range.end();
    while node_id.index() <= end {
        let (path, is_import, kind) = if let Node::ModuleRef {
            import: is_import,
            path,
            module: None,
        } = *shared.arena.get(node_id)
        {
            let kind = if is_import {
                ModuleKind::Helper
            } else {
                ModuleKind::Grammar
            };
            (path, is_import, kind)
        } else {
            node_id = node_id.next();
            continue;
        };

        let path_str = ctx.text(path);
        let canonical = resolve_path(&module_dir.join(path_str), path)?;

        let gid = if let Some(&gid) = loaded.get(&canonical) {
            gid
        } else {
            let gid = load_child_module(
                shared,
                modules,
                path_str,
                module_dir,
                path,
                ancestor_paths,
                kind,
            )?;
            loaded.insert(canonical, gid);
            gid
        };

        shared.arena.set(
            node_id,
            Node::ModuleRef {
                import: is_import,
                path,
                module: Some(gid),
            },
        );

        node_id = node_id.next();
    }

    Ok(())
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

/// Typecheck child modules in loading order (children before parents).
/// The flat `modules` vec is already in this order since children are
/// pushed before their parents during loading.
// TODO: eliminate this in favor of a single recursive typecheck pass
// through the root module once we have a global TypeEnv.
pub fn typecheck_modules<'m>(
    shared: &SharedAst,
    modules: &'m [Module],
    env: &mut TypeEnv<'m>,
) -> DslResult<()> {
    for m in modules {
        typecheck::check(shared, &m.ctx, env, modules)?;
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum NoteMessage {
    FirstDefinedHere,
    ReferencedFromHere,
}

impl std::fmt::Display for NoteMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstDefinedHere => write!(f, "first defined here"),
            Self::ReferencedFromHere => write!(f, "referenced from here"),
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
