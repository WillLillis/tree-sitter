//! Pre-typecheck name resolution for the native grammar DSL.
//!
//! Two passes over the parsed AST:
//!
//! - [`collect_decls`] gathers top-level names (rules, macros, lets, externals, inherited rules)
//! - [`resolve`] rewrites `Ident(Unresolved)` nodes to `Ident(Rule | Var(_) | Macro(_))` and
//!   resolves `mod::name` accesses against imported modules.

use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    grammars::InputGrammar,
    nativedsl::{
        Export, ImportedRule, Module, ModuleId, NoteMessage, ResolveError,
        ast::{
            AstPools, IdentKind, ModuleContext, Node, NodeArena, NodeId, SharedAst,
            Span, Spanned,
        },
        string_pool::StringPool,
        suggest::suggest_name,
    },
    rules::Rule,
};

/// Intermediate resolve environment used during phase 1. Maps declaration
/// names to their resolved kind (for Ident rewriting) and span
/// (for duplicate checking / "first defined here" notes). The kind never
/// stores [`IdentKind::Unresolved`] - by the time a name reaches the table
/// we know what it resolves to.
type Decls<'src> = FxHashMap<&'src str, Spanned<IdentKind>>;

/// Context for [`collect_external_names`].
struct ExternalNameCtx<'src, 'a> {
    decls: &'a mut Decls<'src>,
    /// `let` names currently being expanded; reentry indicates a cycle.
    expanding_lets: FxHashSet<&'src str>,
}

/// Read-only context threaded through the pass-2 resolve walk
/// ([`resolve_item`] / [`resolve_expr`] / [`resolve_children`]).
struct ResolveCtx<'a> {
    pools: &'a AstPools,
    ctx: &'a ModuleContext,
    decls: &'a Decls<'a>,
    modules: &'a [Module],
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
    strings: &StringPool,
    modules: &[Module],
    base: Option<(&InputGrammar, Span)>,
    imported_rules: &[ImportedRule],
) -> ResolveResult<()> {
    let decls = collect_decls(
        shared,
        ctx,
        strings,
        &ctx.root_items,
        base,
        modules,
        imported_rules,
    )?;

    // Validate computed-name references (`@<expr>`), evaluated under each call's
    // args at expand, against the complete name table.
    for &Spanned { value: name, span } in &ctx.computed_refs {
        let text = strings.resolve_local(name, &ctx.source);
        // Must resolve to a rule: decls already holds rules, macros, and lets
        // here, and lower emits a NamedSymbol for whatever name this is.
        if !matches!(
            decls.get(text),
            Some(Spanned {
                value: IdentKind::Rule,
                ..
            })
        ) {
            return Err(ResolveError::new(
                ResolveErrorKind::UnknownIdentifier(text.to_string()),
                span,
            ));
        }
    }

    for &item_id in &ctx.root_items {
        let rcx = ResolveCtx {
            pools: &shared.pools,
            ctx,
            decls: &decls,
            modules,
        };
        resolve_item(&mut shared.arena, &rcx, item_id)?;
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
    if let Some(&Spanned {
        span: first_span, ..
    }) = decls.get(name)
    {
        return Err(ResolveError::with_note(
            ResolveErrorKind::DuplicateDeclaration(name.to_string()),
            span,
            ctx.note(NoteMessage::FirstDefinedHere, first_span),
        ));
    }
    decls.insert(name, Spanned::new(kind, span));
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
    strings: &'a StringPool,
    root_items: &[NodeId],
    base: Option<(&'a InputGrammar, Span)>,
    modules: &'a [Module],
    imported_rules: &[ImportedRule],
) -> ResolveResult<Decls<'a>> {
    let mut decls = FxHashMap::with_capacity_and_hasher(root_items.len(), FxBuildHasher);
    let mut override_names = FxHashSet::default();

    // Register rules, macros, and lets upfront (forward references allowed).
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
            Node::ExpandedRule(expand_id) => {
                let exp = shared.pools.get_expansion(*expand_id);
                let (name, is_override) = (exp.name, exp.is_override);
                // Source entries from expand always reference ctx.source.
                let name_text = strings.resolve_local(name, &ctx.source);
                insert_decl(&mut decls, name_text, IdentKind::Rule, span, ctx)?;
                if is_override {
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
            Node::Let { name, .. } => {
                insert_decl(
                    &mut decls,
                    ctx.text(*name),
                    IdentKind::Var(item_id),
                    span,
                    ctx,
                )?;
            }
            Node::Forward { name } => {
                // External decls register the symbol name as a rule-shaped
                // identifier so it can be referenced from rule bodies.
                insert_decl(&mut decls, ctx.text(*name), IdentKind::Rule, span, ctx)?;
            }
            _ => {}
        }
    }

    // Register inherited rule names. Collisions with local rules error, except
    // for names that have a corresponding `override rule`. The inherit statement's
    // span gives "first defined here" notes a sensible target.
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            // The first source of an overridden name coexists with the override
            // (claim it by removing). A later source is no longer skipped and so
            // collides as a duplicate, exactly as it would without the override.
            if override_names.remove(var.name.as_str()) {
                continue;
            }
            insert_decl(&mut decls, &var.name, IdentKind::Rule, inherit_span, ctx)?;
        }
        // TODO: Check here as well for externals nonsense
        // Inherited external tokens are referenceable by bare name too, just
        // like inherited rules). Anonymous externals (string/pattern) have no
        // name to bring into scope. A base may list one of its own rules in
        // `externals` (an external-scanner token with a grammar-rule fallback),
        // putting the name in both lists; it's one symbol, already registered by
        // the rule loop, so skip it rather than colliding with ourselves.
        for ext in &base_grammar.external_tokens {
            if let Rule::NamedSymbol(name) = ext
                && !override_names.contains(name.as_str())
                && !base_grammar.variables.iter().any(|v| v.name == *name)
            {
                insert_decl(&mut decls, name, IdentKind::Rule, inherit_span, ctx)?;
            }
        }
    }

    // Register bare names for every transitively-imported helper rule, from the
    // shared imported-rule list.
    for ir in imported_rules {
        expect_pat!(
            Module::Helper { lowered_rules, .. },
            &modules[ir.module as usize]
        );
        let name = &lowered_rules[ir.index as usize].0;
        // First source of an overridden name claims the override. A later
        // source is no longer skipped and collides (see the base loop above).
        if !override_names.remove(name.as_str()) {
            insert_decl(&mut decls, name, IdentKind::Rule, ir.ref_span, ctx)?;
        }
    }

    // Pre-register external token names. Externals are the only config field
    // that introduces names not declared as rules in the grammar file.
    if let Some(config) = &ctx.grammar_config
        && let Some(ext_id) = config.externals
    {
        let mut ec = ExternalNameCtx {
            decls: &mut decls,
            expanding_lets: FxHashSet::default(),
        };
        collect_external_names(shared, ctx, ext_id, &mut ec)?;
    }

    Ok(decls)
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item(arena: &mut NodeArena, rcx: &ResolveCtx, item_id: NodeId) -> ResolveResult<()> {
    match arena.get(item_id) {
        Node::Grammar => {
            // INVARIANT: set during grammar block parsing, always present here
            let grammar_config = rcx.ctx.grammar_config.as_ref().unwrap();
            for (_, id) in grammar_config.node_fields() {
                resolve_expr(arena, rcx, id)?;
            }
            Ok(())
        }
        &Node::Rule { body: inner, .. } | &Node::Let { value: inner, .. } => {
            resolve_expr(arena, rcx, inner)
        }
        // The template body is shared and resolved once via the Macro item,
        // here we only resolve the call's args (module-scope expressions).
        &Node::ExpandedRule(expand_id) => {
            let args = rcx.pools.get_expansion(expand_id).args;
            for &arg in rcx.pools.child_slice(args) {
                resolve_expr(arena, rcx, arg)?;
            }
            Ok(())
        }
        Node::Macro(macro_id) => {
            let macro_cfg = rcx.pools.get_macro(*macro_id);
            let params = macro_cfg.params;
            let body = macro_cfg.body;
            // Macro params must not shadow any top-level declaration
            for param in rcx.pools.param_slice(params) {
                let name = rcx.ctx.text(param.name);
                if let Some(&Spanned {
                    span: first_span, ..
                }) = rcx.decls.get(name)
                {
                    return Err(ResolveError::with_note(
                        ResolveErrorKind::ShadowedBinding(name.to_string()),
                        param.name,
                        rcx.ctx.note(NoteMessage::FirstDefinedHere, first_span),
                    ));
                }
            }
            // Resolve the body once: an expression body directly, a rule-set body
            // through resolve_children -> each Rule/ComputedRule decl.
            resolve_expr(arena, rcx, body)
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
fn resolve_expr(arena: &mut NodeArena, rcx: &ResolveCtx, id: NodeId) -> ResolveResult<()> {
    // Local bindings are emitted as MacroParam/ForBinding by the parser,
    // so any remaining `Ident(Unresolved)` is a top-level reference.
    if matches!(arena.get(id), Node::Ident(IdentKind::Unresolved)) {
        let span = arena.span(id);
        let name = rcx.ctx.text(span);
        if let Some(&Spanned { value: kind, .. }) = rcx.decls.get(name) {
            arena.resolve_as(id, kind);
        } else {
            return Err(unknown_ident_error(rcx.ctx, rcx.decls, name, span));
        }
        return Ok(());
    }

    // Alias target: resolve a bare identifier against `decls` like any other
    // (a `let` resolves to its value, a rule to a named-symbol reference). Only
    // an UNDECLARED bare identifier becomes a named-alias rule reference,
    // mirroring grammar.js `alias($.x, $.undeclared)`.
    if let &Node::Alias { content, target } = arena.get(id) {
        resolve_expr(arena, rcx, content)?;
        if matches!(arena.get(target), Node::Ident(IdentKind::Unresolved)) {
            let span = arena.span(target);
            let kind = rcx
                .decls
                .get(rcx.ctx.text(span))
                .map_or(IdentKind::Rule, |s| s.value);
            arena.resolve_as(target, kind);
        } else {
            resolve_expr(arena, rcx, target)?;
        }
        return Ok(());
    }

    if let &Node::For { for_id, body } = arena.get(id) {
        let iterable = rcx.pools.get_for(for_id).iterable;
        resolve_expr(arena, rcx, iterable)?;
        return resolve_expr(arena, rcx, body);
    }

    resolve_children(arena, rcx, id)
}

/// Resolve identifiers in children of a node.
fn resolve_children(arena: &mut NodeArena, rcx: &ResolveCtx, id: NodeId) -> ResolveResult<()> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = arena.get(id).child_range() {
        for child in rcx.pools.child_slice(range) {
            resolve_expr(arena, rcx, *child)?;
        }
        return Ok(());
    }

    match arena.get(id) {
        &Node::Call { name, args } => {
            resolve_expr(arena, rcx, name)?;
            for arg in rcx.pools.child_slice(args) {
                resolve_expr(arena, rcx, *arg)?;
            }
            Ok(())
        }
        Node::Object(range) => {
            for field in rcx.pools.get_object(*range) {
                resolve_expr(arena, rcx, field.value)?;
            }
            Ok(())
        }
        #[rustfmt::skip]
        Node::Repeat { inner: c, .. } | Node::Token { inner: c, .. } | Node::Neg(c)
        | Node::GrammarConfig { module: c, .. } | Node::Field { content: c, .. }
        | Node::Reserved { content: c, .. }
        | Node::Rule { body: c, .. } => resolve_expr(arena, rcx, *c),
        &Node::Append { left: a, right: b }
        | &Node::BinOp { lhs: a, rhs: b, .. }
        | &Node::Prec {
            value: a,
            content: b,
            ..
        }
        | &Node::ComputedRule {
            name_expr: a,
            body: b,
            ..
        } => {
            resolve_expr(arena, rcx, a)?;
            resolve_expr(arena, rcx, b)
        }
        &Node::DynRegex { pattern, flags } => {
            resolve_expr(arena, rcx, pattern)?;
            if let Some(flags) = flags {
                resolve_expr(arena, rcx, flags)?;
            }
            Ok(())
        }
        &Node::SymRef { expr } => resolve_expr(arena, rcx, expr),
        &Node::FieldAccess { obj, .. } => resolve_expr(arena, rcx, obj),
        &Node::QualifiedAccess { obj, member } => {
            resolve_expr(arena, rcx, obj)?;
            // None when obj isn't a module ref, let type checker reports the error
            if let Some(idx) = resolve_module_id(arena, obj) {
                let member_name = rcx.ctx.text(member);
                resolve_qualified_member(
                    rcx.ctx,
                    arena,
                    &rcx.modules[usize::from(idx)],
                    idx,
                    id,
                    member_name,
                    member,
                )?;
            }
            Ok(())
        }
        Node::QualifiedCall(range) => {
            let (obj, name, args) = rcx.pools.get_qualified_call(*range);
            resolve_expr(arena, rcx, obj)?;
            // None when obj isn't a module ref, let type checker reports the error
            if let Some(idx) = resolve_module_id(arena, obj) {
                let macro_name = rcx.ctx.text(arena.span(name));
                resolve_qualified_call_name(
                    rcx.ctx,
                    arena,
                    &rcx.modules[usize::from(idx)],
                    name,
                    macro_name,
                )?;
            }
            for &arg in args {
                resolve_expr(arena, rcx, arg)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Resolve the name node in a `QualifiedCall` to `Ident(Macro(macro_id))`.
fn resolve_qualified_call_name(
    ctx: &ModuleContext,
    arena: &mut NodeArena,
    target: &Module,
    name_id: NodeId,
    macro_name: &str,
) -> ResolveResult<()> {
    match target.export(macro_name) {
        Some(Export::Local(kind @ IdentKind::Macro(_))) => arena.set(name_id, Node::Ident(kind)),
        None => Err(import_member_not_found(
            ctx,
            target,
            macro_name,
            arena.span(name_id),
        ))?,
        // member exists but isn't a macro, leave for typechecker to reject
        Some(_) => {}
    }
    Ok(())
}

/// Resolve a `QualifiedAccess { obj, member }` against the target module's
/// export table:
///   - `let` / `macro` -> `Ident(Var | Macro)`
///   - lowered rule / external -> `ImportRef` (the lowerer indexes directly)
///   - not found -> error
fn resolve_qualified_member(
    ctx: &ModuleContext,
    arena: &mut NodeArena,
    target: &Module,
    module: ModuleId,
    node_id: NodeId,
    member_name: &str,
    member_span: Span,
) -> ResolveResult<()> {
    match target.export(member_name) {
        Some(Export::Local(kind)) => arena.set(node_id, Node::Ident(kind)),
        Some(Export::Rule(target)) => arena.set(node_id, Node::ModuleRule { module, target }),
        None => Err(import_member_not_found(
            ctx,
            target,
            member_name,
            member_span,
        ))?,
    }
    Ok(())
}

/// Build an `ImportMemberNotFound` error, attaching a "did you mean" note when appropriate
fn import_member_not_found(
    ctx: &ModuleContext,
    target: &Module,
    name: &str,
    span: Span,
) -> ResolveError {
    let kind = ResolveErrorKind::ImportMemberNotFound(name.to_string());
    if let Some(suggestion) = suggest_name(name, target.export_keys()) {
        return ResolveError::with_note(
            kind,
            span,
            ctx.note(NoteMessage::DidYouMean(suggestion.to_string()), span),
        );
    }
    ResolveError::new(kind, span)
}

/// Follow a chain of `Ident(Var(_))` -> `Let { value }` bindings until we hit
/// a `ModuleRef`, returning its module id.
fn resolve_module_id(arena: &NodeArena, obj: NodeId) -> Option<ModuleId> {
    let ref_id = resolve_module_ref(arena, obj)?;
    match arena.get(ref_id) {
        Node::ModuleRef {
            module: Some(idx), ..
        } => Some(*idx),
        _ => None,
    }
}

/// Walk `Ident(Var) -> Let.value` chains to find the underlying `ModuleRef` node
pub(super) fn resolve_module_ref(arena: &NodeArena, mut obj: NodeId) -> Option<NodeId> {
    loop {
        match arena.get(obj) {
            Node::Ident(IdentKind::Var(let_id)) => {
                let Node::Let { value, .. } = arena.get(*let_id) else {
                    return None;
                };
                obj = *value;
            }
            Node::ModuleRef { .. } => return Some(obj),
            _ => return None,
        }
    }
}

/// Recursively collect external token names from an expression.
fn collect_external_names<'src>(
    shared: &SharedAst,
    ctx: &'src ModuleContext,
    id: NodeId,
    ec: &mut ExternalNameCtx<'src, '_>,
) -> ResolveResult<()> {
    let arena = &shared.arena;
    match arena.get(id) {
        // Bare identifier: follow a let, ignore an already-declared name, or
        // register an unknown name as a new external token.
        Node::Ident(IdentKind::Unresolved) => {
            let name = ctx.text(arena.span(id));
            match ec.decls.get(name).map(|d| d.value) {
                Some(IdentKind::Var(let_id)) => {
                    if !ec.expanding_lets.insert(name) {
                        return Err(ResolveError::new(
                            ResolveErrorKind::InvalidExternalsExpression,
                            arena.span(id),
                        ));
                    }
                    expect_pat!(Node::Let { value, .. }, *arena.get(let_id));
                    collect_external_names(shared, ctx, value, ec)?;
                    ec.expanding_lets.remove(name);
                }
                None => insert_decl(ec.decls, name, IdentKind::Rule, arena.span(id), ctx)?,
                Some(_) => {}
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
        // Literals don't introduce names, inherited values already registered.
        #[rustfmt::skip]
        Node::StringLit | Node::RawStringLit { .. } | Node::IntLit(_) | Node::Blank
        | Node::DynRegex { .. } | Node::GrammarConfig { .. } | Node::FieldAccess { .. }
        | Node::QualifiedAccess { .. } => {}
        // Anything else is not a valid externals expression. Externals must be a
        // list literal, append, or variable reference.
        _ => Err(ResolveError::new(
            ResolveErrorKind::InvalidExternalsExpression,
            arena.span(id),
        ))?,
    }
    Ok(())
}

pub type ResolveResult<T> = Result<T, ResolveError>;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
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
    #[error(
        "`flags.{{enabled,disabled}}` entries must be plain string literals (not raw strings or expressions)"
    )]
    CfgFlagsNonLiteral,
    #[error(
        "`#[cfg(...)]` is not allowed inside the `flags` field; flag declarations are read before cfg gating runs"
    )]
    CfgInsideFlags,
    #[error("`#[cfg({0})]` references an unknown flag")]
    CfgFlagUnknown(String),
    #[error("flag '{0}' is declared more than once in this grammar's `flags`")]
    CfgFlagDeclaredTwice(String),
}

/// Build an `UnknownIdentifier` error, attaching a "did you mean" note if appropriate
fn unknown_ident_error(ctx: &ModuleContext, decls: &Decls, name: &str, span: Span) -> ResolveError {
    let kind = ResolveErrorKind::UnknownIdentifier(name.to_string());
    let candidates = decls.keys().copied().chain(
        super::lexer::TokenKind::COMBINATOR_KEYWORD_NAMES
            .iter()
            .copied(),
    );
    if let Some(suggestion) = suggest_name(name, candidates) {
        return ResolveError::with_note(
            kind,
            span,
            ctx.note(NoteMessage::DidYouMean(suggestion.to_string()), span),
        );
    }
    ResolveError::new(kind, span)
}
