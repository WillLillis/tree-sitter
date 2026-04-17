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
pub mod resolve;
mod scope_stack;
pub mod serialize;
#[cfg(test)]
mod tests;
pub mod typecheck;

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use crate::grammars::InputGrammar;

use ast::Span;

pub use lexer::{LexError, LexErrorKind};
pub use lower::{LowerError, LowerErrorKind};
pub use parser::{ParseError, ParseErrorKind};
pub use resolve::{ResolveError, ResolveErrorKind};
pub use typecheck::{InnerTy, Ty, TypeError, TypeErrorKind};

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

enum ModuleKind {
    /// Grammar file (root or inherited) - must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported) - only let/fn/import allowed.
    Helper,
}

fn load_module(
    source: &str,
    path: &Path,
    kind: ModuleKind,
    ancestor_paths: &[PathBuf],
) -> Result<Module, DslError> {
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
    match kind {
        ModuleKind::Grammar => validate_grammar(&ast)?,
        ModuleKind::Helper => validate_import_items(&ast)?,
    }

    // All child modules go into one list.
    let mut sub_modules: Vec<Module> = Vec::new();

    // Load inherited grammar (Grammar kind only, must happen before resolve).
    let base_rule_names = if matches!(kind, ModuleKind::Grammar) {
        let names = load_inherit_child(&ast, module_dir, ancestor_paths, &mut sub_modules)?;
        // Tag the Inherit node with its module index (always 0 - first child)
        if let Some(inherit_id) = find_inherit_node(&ast) {
            let ast::Node::Inherit { path: p, .. } = ast.node(inherit_id) else {
                unreachable!()
            };
            ast.nodes[inherit_id.index()] = ast::Node::Inherit {
                path: *p,
                module: Some(0),
            };
        }
        names
    } else {
        Vec::new()
    };

    let inherit_span = find_inherit_node(&ast).map(|id| ast.span(id));

    resolve::resolve(&mut ast, &base_rule_names, inherit_span, path)?;

    // Load imports (after resolve so import nodes are identified).
    load_import_children(&mut ast, module_dir, ancestor_paths, &mut sub_modules)?;

    // For Grammar modules: typecheck and lower to produce InputGrammar.
    let lowered = if matches!(kind, ModuleKind::Grammar) {
        let type_envs = typecheck_modules(&sub_modules)?;
        let _ = typecheck::check(&ast, type_envs)?;
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
) -> Result<InputGrammar, DslError> {
    let module = load_module(input, grammar_path, ModuleKind::Grammar, ancestor_paths)?;
    Ok(module
        .lowered
        .expect("Grammar module always has lowered grammar"))
}

/// Validate that `inherit()` usage is consistent:
/// - At most one `inherit()` call
/// - If `inherit()` exists, `inherits` must be set in grammar config
/// - If `inherits` is set, it must trace to an `inherit()` call
pub fn validate_grammar(ast: &ast::Ast) -> Result<(), DslError> {
    use lower::{LowerError, LowerErrorKind};

    let config_inherits = ast.context.grammar_config.as_ref().and_then(|c| c.inherits);

    let direct_in_config =
        config_inherits.is_some_and(|id| matches!(ast.node(id), ast::Node::Inherit { .. }));

    // Find all inherit() calls in let bindings, error if more than one
    let mut inherit_let: Option<Span> = None;
    for &item_id in &ast.root_items {
        if let ast::Node::Let { value, .. } = ast.node(item_id)
            && matches!(ast.node(*value), ast::Node::Inherit { .. })
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
        && find_inherit_node(ast).is_none()
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
pub fn find_inherit_node(ast: &ast::Ast) -> Option<ast::NodeId> {
    let inherits_id = ast.context.grammar_config.as_ref()?.inherits?;
    match ast.node(inherits_id) {
        ast::Node::Inherit { .. } => Some(inherits_id),
        ast::Node::Ident => {
            let name = ast.text(ast.span(inherits_id));
            ast.root_items.iter().find_map(|&item_id| {
                if let ast::Node::Let { name: n, value, .. } = ast.node(item_id)
                    && ast.text(ast.span(*n)) == name
                    && matches!(ast.node(*value), ast::Node::Inherit { .. })
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

// -- Child module loading --

/// Load a child module from a path string. Handles .tsg and .json extensions,
/// cycle detection, and error wrapping.
fn load_child_module(
    path_str: &str,
    module_dir: &Path,
    span: ast::Span,
    ancestor_paths: &[PathBuf],
    kind: ModuleKind,
) -> Result<Module, DslError> {
    use lower::{LowerError, LowerErrorKind};

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
                ast: ast::Ast::new(String::new()),
                path: canonical,
                sub_modules: Vec::new(),
                lowered: Some(grammar),
            })
        }
        _ => Err(LowerError::new(LowerErrorKind::ModuleUnsupportedExtension, span).into()),
    }
}

/// Resolve the path for a child module, read its source, and handle
/// cycle detection. Returns `(content, canonical_path)`.
fn read_child_module(
    path_str: &str,
    module_dir: &Path,
    span: Span,
    ancestor_paths: &[PathBuf],
) -> Result<(String, PathBuf), DslError> {
    use lower::{LowerError, LowerErrorKind};

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
        let mut chain: Vec<PathBuf> = ancestor_paths.to_vec();
        chain.push(canonical);
        return Err(LowerError::new(LowerErrorKind::ModuleCycle(chain), span).into());
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

/// Load the inherited grammar as a child module. Tags the Inherit node with
/// its module index and returns the base rule names for resolve.
fn load_inherit_child(
    ast: &ast::Ast,
    module_dir: &Path,
    ancestor_paths: &[PathBuf],
    modules: &mut Vec<Module>,
) -> Result<Vec<String>, DslError> {
    let Some(inherit_id) = find_inherit_node(ast) else {
        return Ok(Vec::new());
    };
    let ast::Node::Inherit { path, .. } = ast.node(inherit_id) else {
        unreachable!()
    };

    let path_str = ast.node_text(*path);
    let span = ast.span(*path);
    let child_module = load_child_module(path_str, module_dir, span, ancestor_paths, ModuleKind::Grammar)?;

    let rule_names: Vec<String> = child_module
        .lowered
        .as_ref()
        .map(|g| g.variables.iter().map(|v| v.name.clone()).collect())
        .unwrap_or_default();

    modules.push(child_module);
    Ok(rule_names)
}

/// Load import children. Tags each Import node with its module index.

/// A loaded and resolved module (imported or inherited), ready for
/// typechecking and lowering.
pub struct Module {
    pub ast: ast::Ast,
    pub path: PathBuf,
    /// Child modules (imports within this module).
    pub sub_modules: Vec<Module>,
    /// For inherited grammar modules: the fully lowered grammar, needed
    /// for rule merging and config access. `None` for import-only modules.
    pub lowered: Option<InputGrammar>,
}

/// Walk ALL nodes in the AST for unresolved `import()` nodes, load each
/// file via `load_module`, and set `Node::Import { module }` indices.
fn load_import_children(
    ast: &mut ast::Ast,
    module_dir: &Path,
    ancestor_paths: &[PathBuf],
    modules: &mut Vec<Module>,
) -> Result<(), DslError> {
    use lower::{LowerError, LowerErrorKind};

    // Scan all nodes (not just top-level lets) so imports/inherits in any
    // expression position are resolved (e.g. `extras: import("h.tsg")::EXTRAS`).
    let mut i = 1; // skip index 0 (Unreachable sentinel)
    while i < ast.nodes.len() {
        let node_id = ast::NodeId(NonZeroU32::new(i as u32).unwrap());
        let (import_path_id, span, kind) = match ast.node(node_id) {
            ast::Node::Import { path, module: None } => (*path, ast.span(*path), ModuleKind::Helper),
            ast::Node::Inherit { path, module: None } => (*path, ast.span(*path), ModuleKind::Grammar),
            _ => {
                i += 1;
                continue;
            }
        };

        let path_str = ast.node_text(import_path_id);
        let child = load_child_module(path_str, module_dir, span, ancestor_paths, kind)?;

        let idx = u8::try_from(modules.len())
            .map_err(|_| LowerError::new(LowerErrorKind::ModuleTooMany, span))?;

        let new_node = match ast.node(node_id) {
            ast::Node::Import { .. } => ast::Node::Import { path: import_path_id, module: Some(idx) },
            ast::Node::Inherit { .. } => ast::Node::Inherit { path: import_path_id, module: Some(idx) },
            _ => unreachable!(),
        };
        ast.nodes[node_id.index()] = new_node;

        modules.push(child);
        i += 1;
    }

    Ok(())
}

/// Validate that an imported file only contains allowed items.
fn validate_import_items(ast: &ast::Ast) -> Result<(), DslError> {
    use lower::{LowerError, LowerErrorKind};
    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            ast::Node::Let { .. } | ast::Node::Fn(_) | ast::Node::Print(_) => {}
            _ => {
                return Err(LowerError::new(
                    LowerErrorKind::ModuleDisallowedItem,
                    ast.span(item_id),
                ))?;
            }
        }
    }
    if ast.context.grammar_config.is_some() {
        // Find the Grammar node to get its span
        let span = ast
            .root_items
            .iter()
            .find(|&&id| matches!(ast.node(id), ast::Node::Grammar))
            .map(|&id| ast.span(id))
            .unwrap_or(Span::new(0, 0));
        return Err(LowerError::new(
            LowerErrorKind::ModuleContainsGrammarBlock,
            span,
        ))?;
    }
    Ok(())
}

/// Recursively typecheck imported modules, producing type envs indexed
/// to match the module indices in `Ty::Module(idx)`.
fn typecheck_modules(modules: &[Module]) -> Result<Vec<typecheck::TypeEnv<'_>>, DslError> {
    modules
        .iter()
        .map(|m| {
            let sub_envs = typecheck_modules(&m.sub_modules)?;
            Ok(typecheck::check(&m.ast, sub_envs)?)
        })
        .collect()
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
            Self::Resolve(e) => e.note.as_deref(),
            Self::Lower(e) => e.note.as_deref(),
            Self::Module(e) => e.inner.note(),
            _ => None,
        }
    }
}

pub use diagnostic::NativeDslError;
