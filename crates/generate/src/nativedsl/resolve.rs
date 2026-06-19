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
        Export, Module, ModuleId, NoteMessage, ResolveError,
        ast::{
            AstPools, IdentKind, MacroKind, ModuleContext, Node, NodeArena, NodeId, SharedAst,
            Span, Spanned,
        },
        string_pool::StringPool,
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
struct ExternalNameCtx<'src, 'a, 'b> {
    decls: &'a mut Decls<'src>,
    root_items: &'b [NodeId],
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
) -> ResolveResult<()> {
    let decls = collect_decls(shared, ctx, strings, &ctx.root_items, base, modules)?;

    // Validate computed-name references (`@<expr>`), evaluated under each call's
    // args at expand, against the complete name table (which includes generated
    // and inherited rules).
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
) -> ResolveResult<Decls<'a>> {
    // Size to the item count up front: nearly every top-level item declares a
    // name, so this avoids rehashing the table as decls are collected.
    let mut decls = FxHashMap::with_capacity_and_hasher(root_items.len(), FxBuildHasher);
    let mut override_names: FxHashSet<&str> = FxHashSet::default();

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
                // Registered up front (like rules/macros) so forward and
                // mutually-referential lets resolve; a self-reference resolves
                // to its own Var and is caught downstream as a cycle.
                insert_decl(
                    &mut decls,
                    ctx.text(*name),
                    IdentKind::Var(item_id),
                    span,
                    ctx,
                )?;
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
        // Inherited external tokens are referenceable by bare name too, just
        // like inherited rules (and like grammar.js's `$.name`). Anonymous
        // externals (string/pattern) have no name to bring into scope. A base
        // may list one of its own rules in `externals` (an external-scanner
        // token with a grammar-rule fallback), putting the name in both lists;
        // it's one symbol, already registered by the rule loop, so skip it
        // rather than colliding with ourselves.
        for ext in &base_grammar.external_tokens {
            if let Rule::NamedSymbol(name) = ext
                && !override_names.contains(name.as_str())
                && !base_grammar.variables.iter().any(|v| v.name == *name)
            {
                insert_decl(&mut decls, name, IdentKind::Rule, inherit_span, ctx)?;
            }
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
        // The template body is shared and resolved once via the Macro item;
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
            // Macro params must not shadow any top-level declaration
            for param in &macro_cfg.params {
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
            match macro_cfg.kind {
                MacroKind::Expression(_) => resolve_expr(arena, rcx, macro_cfg.body),
                // Walk the RuleSet body so the original (template) decls'
                // identifiers get resolved too. Typecheck reuses these to
                // catch errors at definition time.
                MacroKind::RuleSet => {
                    let Node::RuleSet(range) = arena.get(macro_cfg.body) else {
                        unreachable!()
                    };
                    let range = *range;
                    let start = range.start as usize;
                    let end = start + range.len as usize;
                    for i in start..end {
                        let decl_id = rcx.pools.children[i];
                        match *arena.get(decl_id) {
                            Node::Rule { body, .. } => resolve_expr(arena, rcx, body)?,
                            // Resolve the computed name too, not just the body -
                            // otherwise a bare-ident name leaves Ident(Unresolved)
                            // for definition-time typecheck to hit unreachable.
                            Node::ComputedRule {
                                name_expr, body, ..
                            } => {
                                resolve_expr(arena, rcx, name_expr)?;
                                resolve_expr(arena, rcx, body)?;
                            }
                            _ => unreachable!(),
                        }
                    }
                    Ok(())
                }
            }
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
fn resolve_expr(arena: &mut NodeArena, rcx: &ResolveCtx, id: NodeId) -> ResolveResult<()> {
    // Local bindings are emitted as MacroParam/ForBinding by the parser,
    // so any remaining Ident(Unresolved) is a top-level reference.
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
    // mirroring grammar.js `alias($.x, $.undeclared)`. Forcing every bare ident
    // to a rule would shadow a same-named `let` (wrong value + named flag).
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

    // For-loop bindings are resolved by the parser (ForBinding nodes).
    // Just recurse into iterable and body.
    if let &Node::For { for_id, body } = arena.get(id) {
        let iterable = rcx.pools.get_for(for_id).iterable;
        resolve_expr(arena, rcx, iterable)?;
        return resolve_expr(arena, rcx, body);
    }

    resolve_children(arena, rcx, id)
}

/// Resolve identifiers in children of a node (mechanical traversal).
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
        Node::Repeat { inner: c, .. }
        | Node::Token { inner: c, .. }
        | Node::Neg(c)
        | Node::GrammarConfig { module: c, .. }
        | Node::Field { content: c, .. }
        | Node::Reserved { content: c, .. } => resolve_expr(arena, rcx, *c),
        &Node::Append { left: a, right: b }
        | &Node::BinOp { lhs: a, rhs: b, .. }
        | &Node::Prec {
            value: a,
            content: b,
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
        // `@<expr>` computed-name ref: resolve the inner name expression.
        // Only reached for rule-set macro templates (expand rewrites invoked
        // instances to SynthRef before resolve runs); without this the inner
        // ident would stay Unresolved and crash definition-time typecheck.
        &Node::SymRef { expr } => resolve_expr(arena, rcx, expr),
        &Node::FieldAccess { obj, .. } => resolve_expr(arena, rcx, obj),
        &Node::QualifiedAccess { obj, member } => {
            resolve_expr(arena, rcx, obj)?;
            // None when obj isn't a module ref - type checker reports the error
            if let Some(idx) = resolve_module_index(arena, obj) {
                let member_name = rcx.ctx.text(member);
                resolve_qualified_member(
                    arena,
                    &rcx.modules[idx],
                    idx as ModuleId,
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
            // None when obj isn't a module ref - type checker reports the error
            if let Some(idx) = resolve_module_index(arena, obj) {
                let macro_name = rcx.ctx.text(arena.span(name));
                resolve_qualified_call_name(arena, &rcx.modules[idx], name, macro_name);
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
/// Only a macro is valid here; anything else is left unresolved for the type
/// checker to report.
fn resolve_qualified_call_name(
    arena: &mut NodeArena,
    target: &Module,
    name_id: NodeId,
    macro_name: &str,
) {
    if let Some(Export::Local(kind @ IdentKind::Macro(_))) = target.export(macro_name) {
        arena.set(name_id, Node::Ident(kind));
    }
}

/// Resolve a `QualifiedAccess { obj, member }` against the target module's
/// export table:
/// - `let` / `macro` -> `Ident(Var | Macro)`
/// - lowered rule / external -> `ImportRef` (the lowerer indexes directly)
/// - not found -> error
fn resolve_qualified_member(
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
        None => {
            return Err(ResolveError::new(
                ResolveErrorKind::ImportMemberNotFound(member_name.to_string()),
                member_span,
            ));
        }
    }
    Ok(())
}

/// Follow a chain of `Ident(Var(_))` -> `Let { value }` bindings until we hit
/// a `ModuleRef`, returning its module index. Handles `let h = import(...)`
/// directly as well as chained aliases like `let h2 = h`.
fn resolve_module_index(arena: &NodeArena, obj: NodeId) -> Option<usize> {
    let ref_id = resolve_module_ref(arena, obj)?;
    match arena.get(ref_id) {
        Node::ModuleRef {
            module: Some(idx), ..
        } => Some(*idx as usize),
        _ => None,
    }
}

/// Walk `Ident(Var) -> Let.value` chains to find the underlying `ModuleRef`
/// node, returning its `NodeId` so callers can read its span or fields.
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
    #[error("`flags.{{enabled,disabled}}` entries must be string literals")]
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

/// Build an `UnknownIdentifier` error, attaching a "did you mean" note if the
/// name is close to a known decl or keyword.
fn unknown_ident_error(ctx: &ModuleContext, decls: &Decls, name: &str, span: Span) -> ResolveError {
    let kind = ResolveErrorKind::UnknownIdentifier(name.to_string());
    let candidates = decls.keys().copied().chain(
        super::lexer::TokenKind::COMBINATOR_KEYWORD_NAMES
            .iter()
            .copied(),
    );
    if let Some(suggestion) = super::suggest::suggest_name(name, candidates) {
        return ResolveError::with_note(
            kind,
            span,
            ctx.note(NoteMessage::DidYouMean(suggestion), span),
        );
    }
    ResolveError::new(kind, span)
}
