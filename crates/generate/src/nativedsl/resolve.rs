//! Pre-typecheck name resolution for the native grammar DSL.
//!
//! Two passes over the parsed AST:
//!
//! - [`collect_decls`] gathers top-level names (rules, macros, lets, externals, inherited rules)
//! - [`resolve`] rewrites `Ident(Unresolved)` nodes to `Ident(Rule | Var(_) | Macro(_))` and
//!   resolves `mod::name` accesses against imported modules.
//!
//! Let bindings register into the decl table after their RHS is resolved.

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use thiserror::Error;

use crate::{
    grammars::InputGrammar,
    nativedsl::{
        Module, NoteMessage, ResolveError,
        ast::{AstPools, IdentKind, ModuleContext, Node, NodeArena, NodeId, SharedAst, Span},
    },
};

/// Intermediate resolve environment used during phase 1. Maps declaration
/// names to their resolved kind (for Ident rewriting) and span
/// (for duplicate checking / "first defined here" notes). The kind never
/// stores [`IdentKind::Unresolved`] - by the time a name reaches the table
/// we know what it resolves to.
type Decls<'src> = FxHashMap<&'src str, (IdentKind, Span)>;

/// Context for [`collect_external_names`].
struct ExternalNameCtx<'src, 'a, 'b> {
    decls: &'a mut Decls<'src>,
    root_items: &'b [NodeId],
    /// `let` names currently being expanded; reentry indicates a cycle.
    expanding_lets: FxHashSet<&'src str>,
}

/// Collects declarations, registers inherited names, and resolves `Node::Ident`
/// nodes from `Unresolved` to `Rule` / `Var`.
///
/// # Errors
///
/// Returns `ResolveError` if name resolution fails.
pub fn resolve(
    shared: &mut SharedAst,
    ctx: &ModuleContext,
    modules: &[Module],
    base: Option<(&InputGrammar, Span)>,
) -> ResolveResult<()> {
    let mut decls = collect_decls(shared, ctx, &ctx.root_items, base, modules)?;

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
            insert_decl(&mut decls, name_text, IdentKind::Var(item_id), span, ctx)?;
        }
    }

    Ok(())
}

fn insert_decl<'src>(
    decls: &mut Decls<'src>,
    name: &'src str,
    kind: IdentKind,
    span: Span,
    ctx: &ModuleContext,
) -> ResolveResult<()> {
    if let Some(&(_, first_span)) = decls.get(name) {
        return Err(ResolveError::with_note(
            ResolveErrorKind::DuplicateDeclaration(name.to_string()),
            span,
            ctx.note(NoteMessage::FirstDefinedHere, first_span),
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
fn collect_decls<'a>(
    shared: &SharedAst,
    ctx: &'a ModuleContext,
    root_items: &[NodeId],
    base: Option<(&'a InputGrammar, Span)>,
    modules: &'a [Module],
) -> ResolveResult<Decls<'a>> {
    let mut decls = Decls::default();
    let mut override_names: FxHashSet<&str> = FxHashSet::default();

    // Register rules and macros upfront (forward references allowed).
    // Let bindings are registered during pass 2 in item order (no forward references).
    for &item_id in root_items {
        let span = shared.arena.span(item_id);
        match shared.arena.get(item_id) {
            Node::Rule {
                name, is_override, ..
            } => {
                let name_text = ctx.text(*name);
                insert_decl(&mut decls, name_text, IdentKind::Rule, span, ctx)?;
                if *is_override {
                    override_names.insert(name_text);
                }
            }
            Node::Macro(macro_id) => {
                let config = shared.pools.get_macro(*macro_id);
                insert_decl(
                    &mut decls,
                    ctx.text(config.name),
                    IdentKind::Macro(*macro_id),
                    span,
                    ctx,
                )?;
            }
            Node::Let { name, value, .. } => {
                // Catch `let X = X` here so externals referencing X surface
                // UnknownIdentifier rather than the less-useful
                // InvalidExternalsExpression from collect_external_names below.
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
            Node::External { name } => {
                // External decls register the symbol name as a rule-shaped
                // identifier so it can be referenced from rule bodies.
                insert_decl(&mut decls, ctx.text(*name), IdentKind::Rule, span, ctx)?;
            }
            _ => {}
        }
    }

    // Register inherited rule names. Strict: collisions with local rules
    // error, except for names that have a corresponding `override rule`
    // (which is the explicit way to redefine an inherited rule). The
    // inherit statement's span gives "first defined here" notes a sensible
    // target.
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            if override_names.contains(var.name.as_str()) {
                continue;
            }
            insert_decl(&mut decls, &var.name, IdentKind::Rule, inherit_span, ctx)?;
        }
    }

    // Register helper rule names from imported helpers (transitively). Same
    // strictness as inherited: collisions error unless overridden locally.
    register_helper_rules(shared, ctx, modules, &mut decls, &override_names)?;

    // Pre-register external token names. Externals are the only config field
    // that introduces names not declared as rules in the grammar file.
    if let Some(config) = &ctx.grammar_config
        && let Some(ext_id) = config.externals
    {
        let mut ec = ExternalNameCtx {
            decls: &mut decls,
            root_items,
            expanding_lets: FxHashSet::default(),
        };
        collect_external_names(shared, ctx, ext_id, &mut ec)?;
    }

    Ok(decls)
}

/// Register each transitively-reachable helper's rule names. Names already
/// in `override_names` are skipped (the local override will replace them).
fn register_helper_rules<'a>(
    shared: &SharedAst,
    ctx: &'a ModuleContext,
    modules: &'a [Module],
    decls: &mut Decls<'a>,
    override_names: &FxHashSet<&str>,
) -> ResolveResult<()> {
    super::for_each_imported_helper(
        &shared.arena,
        &ctx.module_refs,
        modules,
        |_idx, module, ref_span| {
            let Module::Helper { lowered_rules, .. } = module else {
                unreachable!()
            };
            for (name, _) in lowered_rules {
                if !override_names.contains(name.as_str()) {
                    insert_decl(decls, name.as_str(), IdentKind::Rule, ref_span, ctx)?;
                }
            }
            Ok(())
        },
    )
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    modules: &[Module],
    item_id: NodeId,
) -> ResolveResult<()> {
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
                        ctx.note(NoteMessage::FirstDefinedHere, first_span),
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
fn resolve_expr(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    modules: &[Module],
    id: NodeId,
) -> ResolveResult<()> {
    // Local bindings are emitted as MacroParam/ForBinding by the parser,
    // so any remaining Ident(Unresolved) is a top-level reference.
    if matches!(arena.get(id), Node::Ident(IdentKind::Unresolved)) {
        let span = arena.span(id);
        let name = ctx.text(span);
        if let Some(&(kind, _)) = decls.get(name) {
            arena.resolve_as(id, kind);
        } else {
            return Err(unknown_ident_error(arena, ctx, name, span));
        }
        return Ok(());
    }

    // Alias target: a bare identifier becomes a named-alias rule reference,
    // even if undeclared (mirrors grammar.js `alias($.x, $.undeclared)`).
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
) -> ResolveResult<()> {
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
        &Node::DynRegex { pattern, flags } => {
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
) -> ResolveResult<()> {
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
    // Cached external decls (helper or grammar with top-level `external`).
    // Leave as QualifiedAccess - the lowerer interns the name from the
    // target's source.
    if target_ctx
        .external_names
        .iter()
        .any(|&s| target_ctx.text(s) == member_name)
    {
        return Ok(());
    }
    // Inherited grammar's lowered output: rules first, then external_tokens.
    // Both leave the QualifiedAccess intact for the lowerer.
    if let Some(g) = target.lowered()
        && (g.variables.iter().any(|v| v.name == member_name)
            || g.external_tokens
                .iter()
                .any(|r| matches!(r, crate::rules::Rule::NamedSymbol(n) if n == member_name)))
    {
        return Ok(());
    }
    // Helper rule lookup: helper modules carry their own lowered rules.
    // Leave as QualifiedAccess; the lowerer inlines via `import_rule`.
    if let Module::Helper { lowered_rules, .. } = target
        && lowered_rules.iter().any(|(name, _)| name == member_name)
    {
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
) -> ResolveResult<()> {
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
                insert_decl(ec.decls, name, IdentKind::Rule, arena.span(id), ctx)?;
            }
        }
        Node::List(range) | Node::SeqOrChoice { range, .. } | Node::Tuple(range) => {
            for &child in shared.pools.child_slice(*range) {
                collect_external_names(shared, ctx, child, ec)?;
            }
        }
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
        | Node::DynRegex { .. }
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
    #[error("`flags` must be an object literal `{{ enabled: [...], disabled: [...] }}`")]
    CfgFlagsNotObject,
    #[error("`flags` only accepts `enabled` and `disabled` keys, got '{0}'")]
    CfgFlagsUnknownKey(String),
    #[error("`flags.{{enabled,disabled}}` must be a list literal of string flag names")]
    CfgFlagsNotList,
    #[error("`flags.{{enabled,disabled}}` entries must be string literals")]
    CfgFlagsNonLiteral,
    #[error(
        "`#[cfg(...)]` is not allowed inside the `flags` field; flag declarations are read before cfg gating runs"
    )]
    CfgHasCfg,
    #[error("`#[cfg({0})]` references an unknown flag; declare it in this grammar's `flags`")]
    CfgFlagUnknown(String),
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
        let let_span = arena.span(rid);
        if let Node::Let { name: let_name, .. } = arena.get(rid)
            && let_span.start > span.start
            && ctx.text(*let_name) == name
        {
            return Some(let_span);
        }
        None
    });
    let kind = ResolveErrorKind::UnknownIdentifier(name.to_string());
    if let Some(let_span) = forward_span {
        ResolveError::with_note(kind, span, ctx.note(NoteMessage::DefinedLater, let_span))
    } else {
        ResolveError::new(kind, span)
    }
}
