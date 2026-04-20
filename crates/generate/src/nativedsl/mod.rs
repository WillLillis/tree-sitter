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
pub use lexer::{LexError, LexErrorKind};
pub use lower::{DisallowedItemKind, LowerError, LowerErrorKind};
pub use parser::{ParseError, ParseErrorKind};
pub use typecheck::{InnerTy, Ty, TypeError, TypeErrorKind};

use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;

use ast::{Ast, Node, NodeId, Span};
use typecheck::TypeEnv;

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

/// Parse a native DSL source file into an [`InputGrammar`].
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails.
pub fn parse_native_dsl(input: &str, grammar_path: &Path) -> DslResult<InputGrammar> {
    parse_native_dsl_inner(input, grammar_path, &[])
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
) -> DslResult<Module> {
    if source.len() >= u32::MAX as usize {
        Err(LexError {
            kind: LexErrorKind::InputTooLarge,
            span: Span::new(0, 0),
        })?;
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

    // Load inherited grammar (Grammar kind only, must happen before resolve).
    if let Some(inherit_id) = inherit_node {
        load_inherit_child(
            &ast,
            inherit_id,
            module_dir,
            ancestor_paths,
            &mut sub_modules,
        )?;
        let Node::Inherit { path: p, .. } = ast.node(inherit_id) else {
            unreachable!()
        };
        // Inherit is always the first child loaded into an empty sub_modules vec,
        // so it always occupies index 0.
        ast.arena.set(
            inherit_id,
            Node::Inherit {
                path: *p,
                module: Some(0),
            },
        );
    }

    // Load imports. Import nodes are identified by Node::Import { module: None }
    // which is set by the parser - no resolve pass needed beforehand.
    load_import_children(&mut ast, module_dir, ancestor_paths, &mut sub_modules)?;

    // Resolve identifiers + typecheck. For Grammar modules, also lower.
    let type_envs = typecheck_modules(&sub_modules)?;
    let base = inherit_node.and_then(|id| {
        let module = sub_modules.iter().find(|m| m.is_grammar())?;
        Some((module.lowered.as_ref()?, ast.span(id)))
    });
    let _ = typecheck::resolve_and_check(&mut ast, type_envs, base, path)?;

    let lowered = if matches!(kind, ModuleKind::Grammar) {
        Some(lower::lower_with_base(&ast, path, &sub_modules)?)
    } else {
        None
    };

    Ok(Module {
        ast,
        path: path.to_path_buf(),
        sub_modules,
        lowered,
    })
}

fn parse_native_dsl_inner(
    input: &str,
    grammar_path: &Path,
    ancestor_paths: &[PathBuf],
) -> DslResult<InputGrammar> {
    let module = load_module(input, grammar_path, ModuleKind::Grammar, ancestor_paths)?;
    Ok(module
        .lowered
        .expect("Grammar module always has lowered grammar"))
}

/// Validate grammar module structure:
/// - Grammar block must exist
/// - At most one `inherit()` call
/// - If `inherit()` exists, `inherits` must be set in grammar config
/// - If `inherits` is set, it must trace to an `inherit()` call
pub fn validate_grammar(ast: &Ast, inherit_node: Option<NodeId>) -> DslResult<()> {
    if ast.context.grammar_config.is_none() {
        return Err(LowerError::new(LowerErrorKind::MissingGrammarBlock, Span::new(0, 0)).into());
    }
    let config_inherits = ast.context.grammar_config.as_ref().and_then(|c| c.inherits);

    let direct_in_config =
        config_inherits.is_some_and(|id| matches!(ast.node(id), Node::Inherit { .. }));

    // Find all inherit() calls in let bindings, error if more than one
    let mut inherit_let: Option<Span> = None;
    for &item_id in &ast.root_items {
        if let Node::Let { value, .. } = ast.node(item_id)
            && matches!(ast.node(*value), Node::Inherit { .. })
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
    let inherits_id = ast.context.grammar_config.as_ref()?.inherits?;
    match ast.node(inherits_id) {
        Node::Inherit { .. } => Some(inherits_id),
        Node::Ident => {
            let name = ast.text(ast.span(inherits_id));
            ast.root_items.iter().find_map(|&item_id| {
                if let Node::Let { name: n, value, .. } = ast.node(item_id)
                    && ast.text(ast.span(*n)) == name
                    && matches!(ast.node(*value), Node::Inherit { .. })
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
) -> DslResult<Module> {
    let (content, canonical) = read_child_module(path_str, module_dir, span, ancestor_paths)?;

    let mut child_ancestors = ancestor_paths.to_vec();
    child_ancestors.push(canonical.clone());

    let ext = canonical.extension().and_then(|e| e.to_str());
    match ext {
        Some("tsg") => load_module(&content, &canonical, kind, &child_ancestors).map_err(|inner| {
            ModuleError {
                inner: Box::new(inner),
                source_text: content,
                path: canonical,
                reference_span: span,
            }
            .into()
        }),
        Some("json") if matches!(kind, ModuleKind::Grammar) => {
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

/// Load the inherited grammar as a child module and push it to the modules list.
fn load_inherit_child(
    ast: &Ast,
    inherit_id: NodeId,
    module_dir: &Path,
    ancestor_paths: &[PathBuf],
    modules: &mut Vec<Module>,
) -> DslResult<()> {
    let Node::Inherit { path, .. } = ast.node(inherit_id) else {
        unreachable!()
    };

    let path_str = ast.node_text(*path);
    let span = ast.span(*path);
    let child_module = load_child_module(
        path_str,
        module_dir,
        span,
        ancestor_paths,
        ModuleKind::Grammar,
    )?;

    modules.push(child_module);
    Ok(())
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
}

impl Module {
    /// Whether this module is a grammar (inherited) rather than a helper (imported).
    #[must_use]
    pub const fn is_grammar(&self) -> bool {
        self.lowered.is_some()
    }
}

/// Walk ALL nodes in the AST for unresolved `import()` nodes, load each
/// file via `load_module`, and set `Node::Import { module }` indices.
/// Deduplicates: if the same file is imported multiple times, it is loaded
/// once and all references share the same module index.
fn load_import_children(
    ast: &mut Ast,
    module_dir: &Path,
    ancestor_paths: &[PathBuf],
    modules: &mut Vec<Module>,
) -> Result<(), DslError> {
    // Cache: canonical path -> module index, so duplicate imports share one load.
    let mut loaded: FxHashMap<PathBuf, u8> = FxHashMap::default();

    // Scan all nodes (not just top-level lets) so imports/inherits in any
    // expression position are resolved (e.g. `extras: import("h.tsg")::EXTRAS`).
    let mut node_id = NodeId::FIRST;
    while node_id.index() <= ast.arena.len() {
        let (import_path_id, span, kind) = match ast.node(node_id) {
            Node::Import { path, module: None } => (*path, ast.span(*path), ModuleKind::Helper),
            Node::Inherit { path, module: None } => (*path, ast.span(*path), ModuleKind::Grammar),
            _ => {
                node_id = node_id.next();
                continue;
            }
        };

        let path_str = ast.node_text(import_path_id);
        let canonical = dunce::canonicalize(module_dir.join(path_str)).map_err(|e| {
            LowerError::new(
                LowerErrorKind::ModuleResolveFailed {
                    path: module_dir.join(path_str).display().to_string(),
                    error: e.to_string(),
                },
                span,
            )
        })?;

        let idx = if let Some(&idx) = loaded.get(&canonical) {
            idx
        } else {
            let child = load_child_module(path_str, module_dir, span, ancestor_paths, kind)?;
            let idx = u8::try_from(modules.len())
                .map_err(|_| LowerError::new(LowerErrorKind::ModuleTooMany, span))?;
            loaded.insert(canonical, idx);
            modules.push(child);
            idx
        };

        let new_node = match ast.node(node_id) {
            Node::Import { .. } => Node::Import {
                path: import_path_id,
                module: Some(idx),
            },
            Node::Inherit { .. } => Node::Inherit {
                path: import_path_id,
                module: Some(idx),
            },
            _ => unreachable!(),
        };
        ast.arena.set(node_id, new_node);

        node_id = node_id.next();
    }

    Ok(())
}

/// Validate that an imported file only contains allowed items.
fn validate_import_items(ast: &Ast) -> DslResult<()> {
    for &item_id in &ast.root_items {
        let kind = match ast.node(item_id) {
            Node::Grammar => DisallowedItemKind::GrammarBlock,
            Node::Rule { .. } => DisallowedItemKind::Rule,
            Node::OverrideRule { .. } => DisallowedItemKind::OverrideRule,
            _ => continue,
        };
        Err(LowerError::new(
            LowerErrorKind::ModuleDisallowedItem(kind),
            ast.span(item_id),
        ))?;
    }
    Ok(())
}

/// Recursively typecheck imported modules, producing type envs indexed
/// to match the module indices in `Ty::Module(idx)`.
pub fn typecheck_modules(modules: &[Module]) -> DslResult<Vec<TypeEnv<'_>>> {
    modules
        .iter()
        .map(|m| {
            let sub_envs = typecheck_modules(&m.sub_modules)?;
            Ok(typecheck::check(&m.ast, sub_envs)?)
        })
        .collect()
}

pub type DslResult<T> = Result<T, DslError>;

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

#[derive(Clone, Debug, Serialize)]
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
    pub fn call_trace(&self) -> Option<&[(String, Span)]> {
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
            Self::Parse(e) => e.note.as_deref(),
            Self::Type(e) => e.note.as_deref(),
            Self::Lower(e) => e.note.as_deref(),
            Self::Module(e) => e.inner.note(),
            _ => None,
        }
    }
}
