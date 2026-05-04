//! Pre-typecheck name resolution for the native grammar DSL.
//!
//! Two passes over the parsed AST:
//!
//! - [`collect_decls`] gathers top-level names (rules, macros, lets, externals, inherited rules)
//! - [`resolve`] rewrites `Ident(Unresolved)` nodes to `Ident(Rule | Var(_) | Macro(_))` and
//!   resolves `mod::name` accesses against imported modules.
//!
//! Let bindings register into the decl table after their RHS is resolved.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use thiserror::Error;

use crate::{
    grammars::InputGrammar,
    nativedsl::{
        Module, Note, NoteMessage, ResolveError,
        ast::{
            AstPools, IdentKind, MacroId, ModuleContext, Node, NodeArena, NodeId, SharedAst, Span,
        },
    },
};

/// Intermediate resolve environment used during phase 1. Maps declaration
/// names to their kind (for Ident resolution) and span
/// (for duplicate checking / "first defined here" notes).
pub type Decls<'src> = FxHashMap<&'src str, (Decl, Span)>;

#[derive(Clone, Copy)]
pub enum Decl {
    Rule,
    Var(NodeId),
    Macro(MacroId),
}

/// Context for [`collect_external_names`].
struct ExternalNameCtx<'src, 'a, 'b> {
    decls: &'a mut Decls<'src>,
    root_items: &'b [NodeId],
    grammar_path: &'b Path,
    source: &'src str,
    /// `let` names currently being expanded; reentry indicates a cycle.
    expanding_lets: FxHashSet<&'src str>,
}

/// Collects declarations, registers inherited names, and resolves `Node::Ident`
/// nodes from `Unresolved` to `Rule` / `Var`.
///
/// # Errors
///
/// Returns `ResolveError` if name resolution fails.
pub fn resolve<'ast>(
    shared: &'ast mut SharedAst,
    ctx: &'ast ModuleContext,
    modules: &'ast [Module],
    base: Option<(&'ast InputGrammar, Span)>,
    grammar_path: &Path,
) -> Result<(), ResolveError> {
    let mut decls = collect_decls(shared, ctx, &ctx.root_items, grammar_path, base)?;

    for &item_id in &ctx.root_items {
        resolve_item(
            &mut shared.arena,
            &shared.pools,
            ctx,
            &decls,
            modules,
            item_id,
        )?;
        // Register let bindings in order so forward references are caught
        if let Node::Let { name, .. } = shared.arena.get(item_id) {
            let name_text = ctx.text(*name);
            let span = shared.arena.span(item_id);
            insert_decl(
                &mut decls,
                name_text,
                Decl::Var(item_id),
                span,
                grammar_path,
                &ctx.source,
            )?;
        }
    }

    Ok(())
}

pub fn insert_decl<'src>(
    decls: &mut Decls<'src>,
    name: &'src str,
    kind: Decl,
    span: Span,
    grammar_path: &Path,
    source: &str,
) -> Result<(), ResolveError> {
    if let Some(&(_, first_span)) = decls.get(name) {
        return Err(ResolveError::with_note(
            ResolveErrorKind::DuplicateDeclaration(name.to_string()),
            span,
            Note {
                message: NoteMessage::FirstDefinedHere,
                span: first_span,
                path: grammar_path.to_path_buf(),
                source: source.to_string(),
            },
        ));
    }
    decls.insert(name, (kind, span));
    Ok(())
}

/// Pass 1: scan top-level items and register all declared names.
///
/// Rules, let-bindings, macros, and external tokens in the grammar block
/// all occupy the same namespace. Duplicate names are rejected.
///
/// `base` is the inherited grammar (if any). Its rule names are registered
/// before the externals walk so that an `externals` field referencing an
/// inherited rule is correctly identified as a known name rather than being
/// re-registered as a fresh external token.
pub fn collect_decls<'a>(
    shared: &SharedAst,
    ctx: &'a ModuleContext,
    root_items: &[NodeId],
    grammar_path: &Path,
    base: Option<(&'a InputGrammar, Span)>,
) -> Result<Decls<'a>, ResolveError> {
    let mut decls = Decls::default();
    let source = &ctx.source;

    // Register rules and macros upfront (forward references allowed).
    // Let bindings are registered during pass 2 in item order (no forward references).
    for &item_id in root_items {
        let span = shared.arena.span(item_id);
        match shared.arena.get(item_id) {
            Node::Rule { name, .. } => {
                insert_decl(
                    &mut decls,
                    ctx.text(*name),
                    Decl::Rule,
                    span,
                    grammar_path,
                    source,
                )?;
            }
            Node::Macro(macro_id) => {
                let config = shared.pools.get_macro(*macro_id);
                insert_decl(
                    &mut decls,
                    ctx.text(config.name),
                    Decl::Macro(*macro_id),
                    span,
                    grammar_path,
                    source,
                )?;
            }
            Node::Let { name, value, .. } => {
                // Self-referential let (e.g. `let C = C`). This must be caught before
                // the call below to `collect_external_names` to avoid an infinite loop
                let lhs = ctx.text(*name);
                if matches!(shared.arena.get(*value), Node::Ident(IdentKind::Unresolved))
                    && ctx.text(shared.arena.span(*value)) == lhs
                    && !decls.contains_key(lhs)
                {
                    return Err(ResolveError::new(
                        ResolveErrorKind::UnknownIdentifier(lhs.to_string()),
                        shared.arena.span(*value),
                    ));
                }
            }
            _ => {}
        }
    }

    // Register inherited rule names before walking externals, so that an
    // externals field referencing an inherited rule sees it as already declared
    // rather than re-registering it as a fresh external token. The inherit
    // statement's span gives "first defined here" notes a sensible target.
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            if !decls.contains_key(var.name.as_str()) {
                decls.insert(&var.name, (Decl::Rule, inherit_span));
            }
        }
    }

    // Pre-register external token names. Externals are the only config field
    // that introduces names not declared as rules in the grammar file.
    if let Some(config) = &ctx.grammar_config
        && let Some(ext_id) = config.externals
    {
        let mut ec = ExternalNameCtx {
            decls: &mut decls,
            root_items,
            grammar_path,
            source,
            expanding_lets: FxHashSet::default(),
        };
        collect_external_names(shared, ctx, ext_id, &mut ec)?;
    }

    Ok(decls)
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    modules: &[Module],
    item_id: NodeId,
) -> Result<(), ResolveError> {
    match arena.get(item_id) {
        Node::Grammar => {
            // INVARIANT: set during grammar block parsing, always present here
            let grammar_config = ctx.grammar_config.as_ref().unwrap();
            for (_, id) in grammar_config.node_fields() {
                resolve_expr(arena, pools, ctx, decls, modules, id)?;
            }
            Ok(())
        }
        &Node::Rule { body: inner, .. } | &Node::Let { value: inner, .. } => {
            resolve_expr(arena, pools, ctx, decls, modules, inner)
        }
        Node::Macro(macro_id) => {
            let macro_cfg = pools.get_macro(*macro_id);
            // Macro params must not shadow any top-level declaration
            for param in &macro_cfg.params {
                let name = ctx.text(param.name);
                if let Some(&(_, first_span)) = decls.get(name) {
                    return Err(ResolveError::with_note(
                        ResolveErrorKind::ShadowedBinding(name.to_string()),
                        param.name,
                        Note {
                            message: NoteMessage::FirstDefinedHere,
                            span: first_span,
                            path: ctx.path.clone(),
                            source: ctx.source.clone(),
                        },
                    ));
                }
            }
            let body = macro_cfg.body;
            resolve_expr(arena, pools, ctx, decls, modules, body)
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
pub fn resolve_expr(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    modules: &[Module],
    id: NodeId,
) -> Result<(), ResolveError> {
    // Handle Ident mutation first
    if matches!(arena.get(id), Node::Ident(IdentKind::Unresolved)) {
        let span = arena.span(id);
        let name = ctx.text(span);
        // locals check removed: parser emits MacroParam/ForBinding directly,
        // so no Ident(Unresolved) can be a local binding at this point.
        if let Some(&(decl, _)) = decls.get(name) {
            match decl {
                Decl::Var(let_id) => arena.resolve_as(id, IdentKind::Var(let_id)),
                Decl::Rule => arena.resolve_as(id, IdentKind::Rule),
                Decl::Macro(macro_id) => arena.resolve_as(id, IdentKind::Macro(macro_id)),
            }
        } else {
            return Err(unknown_ident_error(arena, ctx, name, span));
        }
        return Ok(());
    }

    // Alias target: bare identifiers resolve as Rule (named alias).
    // Targets don't require the name to be declared - they're just node names,
    // same as grammar.js `alias($.x, $.undeclared_name)`.
    // Parser already emits MacroParam/ForBinding for local bindings,
    // so any remaining Ident(Unresolved) target is a rule name.
    if let &Node::Alias { content, target } = arena.get(id) {
        resolve_expr(arena, pools, ctx, decls, modules, content)?;
        if matches!(*arena.get(target), Node::Ident(IdentKind::Unresolved)) {
            arena.resolve_as(target, IdentKind::Rule);
        } else {
            resolve_expr(arena, pools, ctx, decls, modules, target)?;
        }
        return Ok(());
    }

    // For-loop bindings are resolved by the parser (ForBinding nodes).
    // Just recurse into iterable and body.
    if let &Node::For { for_id, body } = arena.get(id) {
        let iterable = pools.get_for(for_id).iterable;
        resolve_expr(arena, pools, ctx, decls, modules, iterable)?;
        return resolve_expr(arena, pools, ctx, decls, modules, body);
    }

    // All remaining cases
    resolve_children(arena, pools, ctx, decls, modules, id)
}

/// Resolve identifiers in children of a node (mechanical traversal).
fn resolve_children(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    modules: &[Module],
    id: NodeId,
) -> Result<(), ResolveError> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = arena.get(id).child_range() {
        for child in pools.child_slice(range) {
            resolve_expr(arena, pools, ctx, decls, modules, *child)?;
        }
        return Ok(());
    }

    match arena.get(id) {
        &Node::Call { name, args } => {
            resolve_expr(arena, pools, ctx, decls, modules, name)?;
            for arg in pools.child_slice(args) {
                resolve_expr(arena, pools, ctx, decls, modules, *arg)?;
            }
            Ok(())
        }
        Node::Object(range) => {
            for field in pools.get_object(*range) {
                resolve_expr(arena, pools, ctx, decls, modules, field.1)?;
            }
            Ok(())
        }
        Node::Repeat { inner: c, .. }
        | Node::Token { inner: c, .. }
        | Node::Neg(c)
        | Node::GrammarConfig { module: c, .. }
        | Node::Field { content: c, .. }
        | Node::Reserved { content: c, .. } => resolve_expr(arena, pools, ctx, decls, modules, *c),
        &Node::Append { left: a, right: b }
        | &Node::Prec {
            value: a,
            content: b,
            ..
        } => {
            resolve_expr(arena, pools, ctx, decls, modules, a)?;
            resolve_expr(arena, pools, ctx, decls, modules, b)
        }
        Node::DynRegex(range) => {
            let (pattern, flags) = pools.get_regex(*range);
            resolve_expr(arena, pools, ctx, decls, modules, pattern)?;
            if let Some(flags) = flags {
                resolve_expr(arena, pools, ctx, decls, modules, flags)?;
            }
            Ok(())
        }
        &Node::FieldAccess { obj, .. } => resolve_expr(arena, pools, ctx, decls, modules, obj),
        &Node::QualifiedAccess { obj, member } => {
            resolve_expr(arena, pools, ctx, decls, modules, obj)?;
            // None when obj isn't a module ref - type checker reports the error
            if let Some(idx) = resolve_module_index(arena, obj) {
                let member_name = ctx.text(member);
                resolve_qualified_member(arena, pools, &modules[idx], id, member_name, member)?;
            }
            Ok(())
        }
        Node::QualifiedCall(range) => {
            let (obj, name, args) = pools.get_qualified_call(*range);
            resolve_expr(arena, pools, ctx, decls, modules, obj)?;
            // None when obj isn't a module ref - type checker reports the error
            if let Some(idx) = resolve_module_index(arena, obj) {
                let macro_name = ctx.text(arena.span(name));
                resolve_qualified_call_name(arena, pools, &modules[idx], name, macro_name);
            }
            for &arg in args {
                resolve_expr(arena, pools, ctx, decls, modules, arg)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Resolve the name node in a `QualifiedCall` to `Ident(Macro(macro_id))`.
fn resolve_qualified_call_name(
    arena: &mut NodeArena,
    pools: &AstPools,
    target: &Module,
    name_id: NodeId,
    macro_name: &str,
) {
    let target_ctx = target.ctx();
    for &item_id in &target_ctx.root_items {
        if let Node::Macro(macro_id) = arena.get(item_id) {
            let config = pools.get_macro(*macro_id);
            if target_ctx.text(config.name) == macro_name {
                arena.set(name_id, Node::Ident(IdentKind::Macro(*macro_id)));
                return;
            }
        }
    }
    // Not found - leave unresolved, type checker will report the error
}

/// Resolve a `QualifiedAccess { obj, member }` node by looking up `member_name`
/// in the target module's root items.
/// - Let -> replace node with `Ident(Var(item_id))`
/// - Rule -> replace with `Ident(Rule)`
/// - Macro -> error (macro used as value)
/// - Not found -> leave as `QualifiedAccess` (inherited grammar rule for lowerer)
fn resolve_qualified_member(
    arena: &mut NodeArena,
    pools: &AstPools,
    target: &Module,
    node_id: NodeId,
    member_name: &str,
    member_span: Span,
) -> Result<(), ResolveError> {
    let target_ctx = target.ctx();
    for &item_id in &target_ctx.root_items {
        match arena.get(item_id) {
            Node::Let { name, .. } if target_ctx.text(*name) == member_name => {
                arena.set(node_id, Node::Ident(IdentKind::Var(item_id)));
                return Ok(());
            }
            Node::Macro(macro_id) => {
                let config = pools.get_macro(*macro_id);
                if target_ctx.text(config.name) == member_name {
                    arena.set(node_id, Node::Ident(IdentKind::Macro(*macro_id)));
                    return Ok(());
                }
            }
            _ => {}
        }
    }
    // Check inherited grammar rules (only for inherit modules with a lowered grammar)
    if target
        .lowered()
        .is_some_and(|g| g.variables.iter().any(|v| v.name == member_name))
    {
        // Leave as QualifiedAccess - the lowerer handles inherited rule
        // lookups via base_grammar.variables (name-based).
        return Ok(());
    }
    Err(ResolveError::new(
        ResolveErrorKind::ImportMemberNotFound(member_name.to_string()),
        member_span,
    ))
}

/// Follow a chain of `Ident(Var(_))` -> `Let { value }` bindings until we hit
/// a `ModuleRef`, returning its module index. Handles `let h = import(...)`
/// directly as well as chained aliases like `let h2 = h`.
fn resolve_module_index(arena: &NodeArena, mut obj: NodeId) -> Option<usize> {
    loop {
        match arena.get(obj) {
            Node::Ident(IdentKind::Var(let_id)) => {
                let Node::Let { value, .. } = arena.get(*let_id) else {
                    return None;
                };
                obj = *value;
            }
            Node::ModuleRef {
                module: Some(idx), ..
            } => return Some(*idx as usize),
            _ => return None,
        }
    }
}

/// Recursively collect external token names from an expression.
fn collect_external_names<'src>(
    shared: &SharedAst,
    ctx: &'src ModuleContext,
    id: NodeId,
    ec: &mut ExternalNameCtx<'src, '_, '_>,
) -> Result<(), ResolveError> {
    let arena = &shared.arena;
    match arena.get(id) {
        // Bare identifier: either a let binding (follow it) or an external token name.
        Node::Ident(IdentKind::Unresolved) => {
            let name = ctx.text(arena.span(id));
            if let Some(value_id) = find_let_value(arena, ctx, ec.root_items, name) {
                if !ec.expanding_lets.insert(name) {
                    return Err(ResolveError::new(
                        ResolveErrorKind::InvalidExternalsExpression,
                        arena.span(id),
                    ));
                }
                collect_external_names(shared, ctx, value_id, ec)?;
                ec.expanding_lets.remove(name);
            } else if !ec.decls.contains_key(name) {
                // Unknown bare identifier - register as external token.
                insert_decl(
                    ec.decls,
                    name,
                    Decl::Rule,
                    arena.span(id),
                    ec.grammar_path,
                    ec.source,
                )?;
            }
        }
        // List: scan children for identifiers.
        Node::List(range) | Node::SeqOrChoice { range, .. } | Node::Tuple(range) => {
            for &child in shared.pools.child_slice(*range) {
                collect_external_names(shared, ctx, child, ec)?;
            }
        }
        // Append: recurse into both arms.
        Node::Append { left, right } => {
            collect_external_names(shared, ctx, *left, ec)?;
            collect_external_names(shared, ctx, *right, ec)?;
        }
        // Literals don't introduce names.
        // grammar_config(base).externals - inherited values already registered.
        Node::StringLit
        | Node::RawStringLit { .. }
        | Node::IntLit(_)
        | Node::Blank
        | Node::GrammarConfig { .. }
        | Node::FieldAccess { .. }
        | Node::QualifiedAccess { .. } => {}
        // Anything else (macro calls, for-loops, etc.) is not a valid
        // externals expression. Externals must be a list literal, append,
        // or variable reference.
        _ => {
            return Err(ResolveError::new(
                ResolveErrorKind::InvalidExternalsExpression,
                arena.span(id),
            ));
        }
    }
    Ok(())
}

/// Find the value node of a let binding by name.
fn find_let_value(
    arena: &NodeArena,
    ctx: &ModuleContext,
    root_items: &[NodeId],
    name: &str,
) -> Option<NodeId> {
    root_items.iter().find_map(|&item_id| {
        if let Node::Let { name: n, value, .. } = arena.get(item_id)
            && ctx.text(*n) == name
        {
            return Some(*value);
        }
        None
    })
}

pub type ResolveResult<T> = Result<T, ResolveError>;

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum ResolveErrorKind {
    #[error("imported module has no member '{0}'")]
    ImportMemberNotFound(String),
    #[error("duplicate declaration '{0}'")]
    DuplicateDeclaration(String),
    #[error("unknown identifier '{0}'")]
    UnknownIdentifier(String),
    #[error("'{0}' shadows an existing declaration")]
    ShadowedBinding(String),
    #[error("externals must be a list literal, append(), or variable reference")]
    InvalidExternalsExpression,
}

/// Build an `UnknownIdentifier` error, attaching a "defined later" note if the
/// name matches a let binding that appears later in the same module.
fn unknown_ident_error(
    arena: &NodeArena,
    ctx: &ModuleContext,
    name: &str,
    span: Span,
) -> ResolveError {
    let forward_span = ctx.root_items.iter().find_map(|&rid| {
        if let Node::Let { name: let_name, .. } = arena.get(rid)
            && ctx.text(*let_name) == name
        {
            return Some(arena.span(rid));
        }
        None
    });
    let kind = ResolveErrorKind::UnknownIdentifier(name.to_string());
    if let Some(let_span) = forward_span {
        ResolveError::with_note(
            kind,
            span,
            Note {
                message: NoteMessage::DefinedLater,
                span: let_span,
                path: ctx.path.clone(),
                source: ctx.source.clone(),
            },
        )
    } else {
        ResolveError::new(kind, span)
    }
}
