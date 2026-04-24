//! Native DSL front-end for tree-sitter grammar definitions.
//!
//! Pipeline: lex -> parse -> resolve+typecheck -> lower, producing an
//! [`InputGrammar`] equivalent to what `grammar.json` parsing produces.

pub mod ast;
pub mod diagnostic;
pub mod lexer;
pub mod lower;
pub mod parser;
mod scope_stack;
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

use ast::{Ast, IdentKind, Node, NodeId, Span};
use typecheck::TypeEnv;

/// Parse a native DSL source file into an [`InputGrammar`].
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails.
pub fn parse_native_dsl(input: &str, grammar_path: &Path) -> DslResult<InputGrammar> {
    let module = load_module(input, grammar_path, ModuleKind::Grammar, &[], &mut 0)?;
    Ok(module
        .lowered
        .expect("Grammar module always has lowered grammar"))
}

#[derive(Clone, Copy)]
pub enum ModuleKind {
    /// Grammar file (root or inherited). Must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported). Only let/fn/import/print allowed.
    Helper,
}

/// Loads a tsg module, imported either via `import` or `inherit`
///
/// # Errors
///
/// Returns `Err` if the imported module isn't valid tsg source
pub fn load_module(
    source: &str,
    path: &Path,
    kind: ModuleKind,
    ancestor_paths: &[PathBuf],
    next_global_id: &mut u8,
) -> DslResult<Module> {
    let global_id = *next_global_id;
    *next_global_id = next_global_id
        .checked_add(1)
        .ok_or_else(|| LowerError::new(LowerErrorKind::ModuleTooMany, Span::new(0, 0)))?;

    if source.len() >= u32::MAX as usize {
        Err(LexError::new(LexErrorKind::InputTooLarge, Span::new(0, 0)))?;
    }

    let module_dir = path.parent().unwrap();

    let tokens = lexer::Lexer::new(source).tokenize()?;
    let mut ast = parser::Parser::new(&tokens, source.to_string(), path).parse()?;

    // Validate items based on module kind
    let inherit_node = match kind {
        ModuleKind::Grammar => {
            let node = find_inherit_node(&ast);
            validate_grammar(&ast, node)?;
            node
        }
        ModuleKind::Helper => {
            validate_import_items(&ast)?;
            None
        }
    };

    // All child modules go into one list.
    let mut sub_modules: Vec<Module> = Vec::new();

    // Load inherited grammar (Grammar kind only, must happen before typecheck).
    if let Some(inherit_id) = inherit_node {
        let Node::ModuleRef {
            import: false,
            path,
            ..
        } = ast.node(inherit_id)
        else {
            unreachable!()
        };
        let path = *path;
        let child = load_child_module(
            ast.ctx.text(path),
            module_dir,
            path,
            ancestor_paths,
            ModuleKind::Grammar,
            next_global_id,
        )?;
        let child_gid = child.global_id;
        sub_modules.push(child);
        ast.arena.set(
            inherit_id,
            Node::ModuleRef {
                import: false,
                path,
                module: Some(child_gid),
            },
        );
    }

    // Load imports. Unresolved module refs are identified by
    // Node::ModuleRef { module: None } which is set by the parser.
    load_import_children(
        &mut ast,
        module_dir,
        ancestor_paths,
        &mut sub_modules,
        next_global_id,
    )?;

    // Resolve identifiers + typecheck. For Grammar modules, also lower.
    let mut type_envs: Vec<Option<TypeEnv<'_>>> = (0..*next_global_id).map(|_| None).collect();
    typecheck_modules(&sub_modules, &mut type_envs)?;
    let base = inherit_node.and_then(|id| {
        let module = sub_modules.iter().find(|m| m.is_grammar())?;
        Some((module.lowered.as_ref()?, ast.span(id)))
    });
    let _ = typecheck::resolve_and_check(&mut ast, &type_envs, base, path)?;

    let lowered = if matches!(kind, ModuleKind::Grammar) {
        Some(lower::lower_with_base(&ast, &sub_modules, global_id, path)?)
    } else {
        None
    };

    Ok(Module {
        ast,
        path: path.to_path_buf(),
        sub_modules,
        lowered,
        global_id,
    })
}

/// Validate grammar module structure:
/// - Grammar block must exist
/// - At most one `inherit()` call
/// - If `inherit()` exists, `inherits` must be set in grammar config
/// - If `inherits` is set, it must trace to an `inherit()` call
pub fn validate_grammar(ast: &Ast, inherit_node: Option<NodeId>) -> DslResult<()> {
    if ast.ctx.grammar_config.is_none() {
        return Err(LowerError::new(LowerErrorKind::MissingGrammarBlock, Span::new(0, 0)).into());
    }
    let config_inherits = ast.ctx.grammar_config.as_ref().and_then(|c| c.inherits);

    let direct_in_config = config_inherits
        .is_some_and(|id| matches!(ast.node(id), Node::ModuleRef { import: false, .. }));

    // Find all inherit() calls in let bindings, error if more than one
    let mut inherit_let: Option<Span> = None;
    for &item_id in &ast.root_items {
        if let Node::Let { value, .. } = ast.node(item_id)
            && matches!(ast.node(*value), Node::ModuleRef { import: false, .. })
        {
            if inherit_let.is_some() || direct_in_config {
                Err(LowerError::new(
                    LowerErrorKind::MultipleInherits,
                    ast.span(item_id),
                ))?;
            }
            inherit_let = Some(ast.span(item_id));
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
            ast.span(id),
        ))?;
    }

    Ok(())
}

/// Resolve the `Inherit` node from the grammar config's `inherits` field,
/// following variable references to let bindings if needed.
#[must_use]
pub fn find_inherit_node(ast: &Ast) -> Option<NodeId> {
    let inherits_id = ast.ctx.grammar_config.as_ref()?.inherits?;
    match ast.node(inherits_id) {
        Node::ModuleRef { import: false, .. } => Some(inherits_id),
        Node::Ident(IdentKind::Unresolved) => {
            let name = ast.ctx.text(ast.span(inherits_id));
            ast.root_items.iter().find_map(|&item_id| {
                if let Node::Let { name: n, value, .. } = ast.node(item_id)
                    && ast.ctx.text(ast.span(*n)) == name
                    && matches!(ast.node(*value), Node::ModuleRef { import: false, .. })
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
fn load_child_module(
    path_str: &str,
    module_dir: &Path,
    span: Span,
    ancestor_paths: &[PathBuf],
    kind: ModuleKind,
    next_global_id: &mut u8,
) -> DslResult<Module> {
    let (content, canonical) = read_child_module(path_str, module_dir, span, ancestor_paths)?;

    let mut child_ancestors = ancestor_paths.to_vec();
    child_ancestors.push(canonical.clone());

    let ext = canonical.extension().and_then(|e| e.to_str());
    match ext {
        Some("tsg") => load_module(&content, &canonical, kind, &child_ancestors, next_global_id)
            .map_err(|inner| {
                ModuleError {
                    inner: Box::new(inner),
                    source_text: content,
                    path: canonical,
                    reference_span: span,
                }
                .into()
            }),
        Some("json") if matches!(kind, ModuleKind::Grammar) => {
            let gid = *next_global_id;
            *next_global_id = next_global_id
                .checked_add(1)
                .ok_or_else(|| LowerError::new(LowerErrorKind::ModuleTooMany, span))?;
            let grammar = crate::parse_grammar::parse_grammar(&content).map_err(|e| {
                LowerError::new(
                    LowerErrorKind::ModuleJsonError {
                        path: canonical.display().to_string(),
                        error: e,
                    },
                    span,
                )
            })?;
            Ok(Module {
                ast: Ast::new(String::new()),
                path: canonical,
                sub_modules: Vec::new(),
                lowered: Some(grammar),
                global_id: gid,
            })
        }
        Some("json") => Err(LowerError::new(LowerErrorKind::JsonImportNotAllowed, span).into()),
        _ => Err(LowerError::new(LowerErrorKind::ModuleUnsupportedExtension, span).into()),
    }
}

const MAX_MODULE_DEPTH: usize = 256;

/// Resolve the path for a child module, read its source, and handle
/// cycle/depth detection. Returns `(content, canonical_path)`.
fn read_child_module(
    path_str: &str,
    module_dir: &Path,
    span: Span,
    ancestor_paths: &[PathBuf],
) -> DslResult<(String, PathBuf)> {
    if ancestor_paths.len() >= MAX_MODULE_DEPTH {
        return Err(LowerError::new(LowerErrorKind::ModuleDepthExceeded, span).into());
    }

    let full_path = module_dir.join(path_str);
    let canonical = dunce::canonicalize(&full_path).map_err(|e| {
        LowerError::new(
            LowerErrorKind::ModuleResolveFailed {
                path: full_path.display().to_string(),
                error: e.to_string(),
            },
            span,
        )
    })?;

    // Cycle detection
    if ancestor_paths.iter().any(|p| p == &canonical) {
        Err(LowerError::new(LowerErrorKind::ModuleCycle, span))?;
    }

    let content = std::fs::read_to_string(&full_path).map_err(|e| {
        LowerError::new(
            LowerErrorKind::ModuleReadFailed {
                path: full_path.display().to_string(),
                error: e.to_string(),
            },
            span,
        )
    })?;

    Ok((content, canonical))
}

/// A loaded and resolved module (imported or inherited), ready for
/// typechecking and lowering.
pub struct Module {
    pub ast: Ast,
    pub path: PathBuf,
    /// Child modules (imports within this module).
    pub sub_modules: Vec<Self>,
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

/// Walk ALL nodes in the AST for unresolved module ref nodes, load each
/// file via `load_module`, and set `Node::ModuleRef { module }` indices.
/// Deduplicates: if the same file is imported multiple times, it is loaded
/// once and all references share the same module index.
fn load_import_children(
    ast: &mut Ast,
    module_dir: &Path,
    ancestor_paths: &[PathBuf],
    modules: &mut Vec<Module>,
    next_global_id: &mut u8,
) -> Result<(), DslError> {
    // Cache: canonical path -> global module ID, so duplicate imports share one load.
    let mut loaded: FxHashMap<PathBuf, u8> = FxHashMap::default();

    // Scan all nodes (not just top-level lets) so imports/inherits in any
    // expression position are resolved (e.g. `extras: import("h.tsg")::EXTRAS`).
    let mut node_id = NodeId::FIRST;
    while node_id.index() <= ast.arena.len() {
        let (path, is_import, kind) = if let Node::ModuleRef {
            import: is_import,
            path,
            module: None,
        } = ast.node(node_id)
        {
            let kind = if *is_import {
                ModuleKind::Helper
            } else {
                ModuleKind::Grammar
            };
            (*path, *is_import, kind)
        } else {
            node_id = node_id.next();
            continue;
        };

        let path_str = ast.ctx.text(path);
        let canonical = dunce::canonicalize(module_dir.join(path_str)).map_err(|e| {
            LowerError::new(
                LowerErrorKind::ModuleResolveFailed {
                    path: module_dir.join(path_str).display().to_string(),
                    error: e.to_string(),
                },
                path,
            )
        })?;

        let idx = if let Some(&idx) = loaded.get(&canonical) {
            idx
        } else {
            let child = load_child_module(
                path_str,
                module_dir,
                path,
                ancestor_paths,
                kind,
                next_global_id,
            )?;
            let idx = child.global_id;
            loaded.insert(canonical, idx);
            modules.push(child);
            idx
        };

        ast.arena.set(
            node_id,
            Node::ModuleRef {
                import: is_import,
                path,
                module: Some(idx),
            },
        );

        node_id = node_id.next();
    }

    Ok(())
}

/// Validate that an imported file only contains allowed items.
fn validate_import_items(ast: &Ast) -> DslResult<()> {
    for &item_id in &ast.root_items {
        let kind = match ast.node(item_id) {
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
            ast.span(item_id),
        ))?;
    }
    Ok(())
}

/// Recursively typecheck imported modules, writing each module's type env
/// into `table` at the slot corresponding to its `global_id`.
pub fn typecheck_modules<'m>(
    modules: &'m [Module],
    table: &mut Vec<Option<TypeEnv<'m>>>,
) -> DslResult<()> {
    for m in modules {
        typecheck_modules(&m.sub_modules, table)?;
        let env = typecheck::check(&m.ast, table)?;
        table[m.global_id as usize] = Some(env);
    }
    Ok(())
}

pub type DslResult<T> = Result<T, DslError>;

/// Diagnostic error shared by all pipeline stages.
#[derive(Debug, Serialize)]
pub struct Diagnostic<K> {
    pub kind: K,
    pub span: Span,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<Box<Note>>,
}

impl<K> Diagnostic<K> {
    pub const fn new(kind: K, span: Span) -> Self {
        Self {
            kind,
            span,
            note: None,
        }
    }

    pub fn with_note(kind: K, span: Span, note: Note) -> Self {
        Self {
            kind,
            span,
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
    pub const fn span(&self) -> Span {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Type(e) => e.span,
            Self::Lower(e) => e.span,
            Self::Module(e) => e.reference_span,
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
