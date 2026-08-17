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
    nativedsl::{
        Export, ImportedRule, LoweredGrammar, Module, ModuleId, NoteMessage, ResolveError,
        ast::{
            AstPools, IdentKind, MacroId, ModuleContext, Node, NodeArena, NodeId, Param, SharedAst,
            Span, Spanned,
        },
        diagnostic::suggest_name,
    },
    rules::{Rule, RulePool},
    strpool::{StrId, StrPool},
};

/// Intermediate resolve environment used during phase 1. Maps declaration
/// names to their resolved kind (for Ident rewriting) and span
/// (for duplicate checking / "first defined here" notes). The kind never
/// stores [`IdentKind::Unresolved`] - by the time a name reaches the table
/// we know what it resolves to.
type Decls = FxHashMap<StrId, Spanned<DeclKind>>;

/// What a declared name denotes
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DeclKind {
    Rule,
    Var(NodeId),
    Macro(MacroId),
}

impl DeclKind {
    /// The [`IdentKind`] a reference to `name` takes on once it resolves.
    const fn to_ident(self, name: StrId) -> IdentKind {
        match self {
            Self::Rule => IdentKind::Rule(name),
            Self::Var(id) => IdentKind::Var(id),
            Self::Macro(id) => IdentKind::Macro(id),
        }
    }
}

/// Context for [`collect_external_names`].
struct ExternalNameCtx<'a> {
    decls: &'a mut Decls,
    /// `let` names currently being expanded; reentry indicates a cycle.
    expanding_lets: FxHashSet<StrId>,
}

/// Read-only context threaded through the pass-2 resolve walk
/// ([`resolve_item`] / [`resolve_expr`] / [`resolve_children`]).
struct ResolveCtx<'a> {
    pools: &'a AstPools,
    ctx: &'a ModuleContext,
    strs: &'a StrPool,
    decls: &'a Decls,
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
    pool: &mut RulePool,
    modules: &[Module],
    base: Option<(&LoweredGrammar, Span)>,
    imported_rules: &[ImportedRule],
) -> ResolveResult<()> {
    let decls = collect_decls(
        shared,
        ctx,
        pool,
        &ctx.root_items,
        base,
        modules,
        imported_rules,
    )?;

    let rcx = ResolveCtx {
        pools: &shared.pools,
        ctx,
        strs: pool.strs(),
        decls: &decls,
        modules,
    };

    // Validate computed-name references (`@<expr>`), evaluated under each call's
    // args at expand, against the complete name table.
    for &Spanned { value: name, span } in &ctx.computed_refs {
        // Must resolve to a rule: lower emits a NamedSymbol for whatever name
        // this is, so a let/macro match would lower a dangling symbol.
        match rcx.decls.get(&name).map(|d| d.value) {
            Some(DeclKind::Rule) => {}
            Some(_) => Err(ResolveError::new(
                ResolveErrorKind::ComputedNameNotARule(rcx.strs.resolve(name).to_string()),
                span,
            ))?,
            None => Err(unknown_ident_error(&rcx, rcx.strs.resolve(name), span))?,
        }
    }

    // The resolve work-stack holds at most one item's branch frontier (measured
    // ~20-80% of root_items across the corpus); root_items.len() is a cheap seed
    // that covers the realistic worst case and avoids the growth reallocs.
    let mut stack = Vec::with_capacity(ctx.root_items.len());
    for &item_id in &ctx.root_items {
        resolve_item(&mut shared.arena, &rcx, item_id, &mut stack)?;
    }

    Ok(())
}

fn insert_decl(
    decls: &mut Decls,
    strs: &StrPool,
    name: StrId,
    kind: DeclKind,
    span: Span,
    ctx: &ModuleContext,
) -> ResolveResult<()> {
    if let Some(&Spanned {
        span: first_span, ..
    }) = decls.get(&name)
    {
        return Err(ResolveError::with_note(
            ResolveErrorKind::DuplicateDeclaration(strs.resolve(name).to_string()),
            span,
            ctx.note(NoteMessage::FirstDefinedHere, first_span),
        ));
    }
    decls.insert(name, Spanned::new(kind, span));
    Ok(())
}

/// Reject a macro-param / for-binding name that shadows a top-level declaration.
fn check_shadowing(rcx: &ResolveCtx, bindings: &[Param]) -> ResolveResult<()> {
    for binding in bindings {
        if let Some(&Spanned {
            span: first_span, ..
        }) = rcx.decls.get(&binding.name.value)
        {
            return Err(ResolveError::with_note(
                ResolveErrorKind::ShadowedBinding(rcx.strs.resolve(binding.name.value).to_string()),
                binding.name.span,
                rcx.ctx.note(NoteMessage::FirstDefinedHere, first_span),
            ));
        }
    }
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
fn collect_decls(
    shared: &SharedAst,
    ctx: &ModuleContext,
    pool: &mut RulePool,
    root_items: &[NodeId],
    base: Option<(&LoweredGrammar, Span)>,
    modules: &[Module],
    imported_rules: &[ImportedRule],
) -> ResolveResult<Decls> {
    let mut decls = FxHashMap::with_capacity_and_hasher(root_items.len(), FxBuildHasher);
    let mut override_names = FxHashSet::default();

    // Register rules, macros, and lets upfront (forward references allowed).
    for &item_id in root_items {
        let span = shared.arena.span(item_id);
        match shared.arena.get(item_id) {
            Node::Rule {
                name, is_override, ..
            } => {
                insert_decl(&mut decls, pool.strs(), *name, DeclKind::Rule, span, ctx)?;
                if *is_override {
                    override_names.insert(*name);
                }
            }
            Node::ExpandedRule(expand_id) => {
                let exp = shared.pools.get_expansion(*expand_id);
                insert_decl(&mut decls, pool.strs(), exp.name, DeclKind::Rule, span, ctx)?;
                if exp.is_override {
                    override_names.insert(exp.name);
                }
            }
            Node::Macro(macro_id) => {
                let config = shared.pools.get_macro(*macro_id);
                insert_decl(
                    &mut decls,
                    pool.strs(),
                    config.name.value,
                    DeclKind::Macro(*macro_id),
                    span,
                    ctx,
                )?;
            }
            Node::Let { name, .. } => {
                insert_decl(
                    &mut decls,
                    pool.strs(),
                    *name,
                    DeclKind::Var(item_id),
                    span,
                    ctx,
                )?;
            }
            // Forward-decls (`expect X`) are registered last (below)
            _ => {}
        }
    }

    // Register inherited rule names. Collisions with local non-`override` rules error
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            // The first source of an overridden name coexists with the override
            // (claim it by removing). A later source is no longer skipped and so
            // collides as a duplicate, exactly as it would without the override.
            if override_names.remove(&var.name) {
                continue;
            }
            insert_decl(
                &mut decls,
                pool.strs(),
                var.name,
                DeclKind::Rule,
                inherit_span,
                ctx,
            )?;
        }
        // Inherited external tokens are referenceable by bare name too, just
        // like inherited rules). Anonymous externals (string/pattern) have no
        // name to bring into scope. A base may list one of its own rules in
        // `externals` (an external-scanner token with a grammar-rule fallback),
        // putting the name in both lists; it's one symbol, already registered by
        // the rule loop, so skip it rather than colliding with ourselves.
        for ext in &base_grammar.external_roots {
            if let Rule::NamedSymbol(name) = pool.node(*ext)
                && !override_names.contains(&name)
                && !base_grammar.variables.iter().any(|v| v.name == name)
            {
                insert_decl(
                    &mut decls,
                    pool.strs(),
                    name,
                    DeclKind::Rule,
                    inherit_span,
                    ctx,
                )?;
            }
        }
    }

    // Register bare names for every transitively-imported helper rule, from the
    // shared imported-rule list.
    for ir in imported_rules {
        expect_pat!(
            Module::Helper { lowered_rules, .. },
            &modules[usize::from(ir.module)]
        );
        let name = lowered_rules[ir.index as usize].0;
        // First source of an overridden name claims the override. A later
        // source is no longer skipped and collides (see the base loop above).
        if !override_names.remove(&name) {
            insert_decl(
                &mut decls,
                pool.strs(),
                name,
                DeclKind::Rule,
                ir.ref_span,
                ctx,
            )?;
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
        collect_external_names(shared, ctx, pool.strs_mut(), ext_id, &mut ec)?;
    }

    // Forward-decls (`expect X`) name a symbol provided elsewhere: a rule (here or
    // in an imported helper), an external token, or an inherited rule. Registered
    // last, so a real definition from any pass above fulfills the declaration
    // rather than colliding with it; the name is added only if still undefined (an
    // unfulfilled `expect` is then caught by the lower symbol-completeness check).
    for &item_id in root_items {
        let Node::Forward { name } = shared.arena.get(item_id) else {
            continue;
        };
        let span = shared.arena.span(item_id);
        decls
            .entry(*name)
            .or_insert_with(|| Spanned::new(DeclKind::Rule, span));
    }

    Ok(decls)
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item(
    arena: &mut NodeArena,
    rcx: &ResolveCtx,
    item_id: NodeId,
    stack: &mut Vec<Resolve>,
) -> ResolveResult<()> {
    match arena.get(item_id) {
        Node::Grammar => {
            // INVARIANT: set during grammar block parsing, always present here
            let grammar_config = rcx.ctx.grammar_config.as_ref().unwrap();
            for (_, id) in grammar_config.node_fields() {
                resolve_expr(arena, rcx, id, stack)?;
            }
            Ok(())
        }
        &Node::Rule { body: inner, .. } | &Node::Let { value: inner, .. } => {
            resolve_expr(arena, rcx, inner, stack)
        }
        // The template body is shared and resolved once via the Macro item,
        // here we only resolve the call's args (module-scope expressions).
        &Node::ExpandedRule(expand_id) => {
            let args = rcx.pools.get_expansion(expand_id).args;
            for &arg in rcx.pools.child_slice(args) {
                resolve_expr(arena, rcx, arg, stack)?;
            }
            Ok(())
        }
        Node::Macro(macro_id) => {
            let macro_cfg = rcx.pools.get_macro(*macro_id);
            let params = macro_cfg.params;
            let body = macro_cfg.body;
            // Macro params must not shadow any top-level declaration
            check_shadowing(rcx, rcx.pools.param_slice(params))?;
            // Resolve the body once: an expression body directly, a rule-set body
            // through resolve_children -> each Rule/ComputedRule decl.
            resolve_expr(arena, rcx, body, stack)
        }
        _ => Ok(()),
    }
}

/// Work item for the iterative [`resolve_expr`] traversal.
#[derive(Clone, Copy)]
enum Resolve {
    /// Resolve this node: rewrite an ident, or schedule its children.
    Node(NodeId),
    /// Resolve a `::` access's member/macro name, once its object subtree is
    /// resolved (so the object's module is known).
    Member(NodeId),
}

/// Resolve identifiers within an expression. Uses an explicit stack rather than
/// recursion so a deeply nested expression cannot overflow the native stack. The
/// `stack` scratch is reused across calls; single-child / first-child links are
/// followed directly, so only branch children take a stack round-trip.
fn resolve_expr(
    arena: &mut NodeArena,
    rcx: &ResolveCtx,
    root: NodeId,
    stack: &mut Vec<Resolve>,
) -> ResolveResult<()> {
    stack.clear();
    stack.push(Resolve::Node(root));
    while let Some(work) = stack.pop() {
        match work {
            Resolve::Node(mut id) => {
                while let Some(next) = resolve_node(arena, rcx, id, stack)? {
                    id = next;
                }
            }
            Resolve::Member(id) => resolve_member(arena, rcx, id)?,
        }
    }
    Ok(())
}

/// Resolve one node in place (rewriting an ident), push its branch children onto
/// `stack` (reversed, to pop in source order), and return its first child to
/// descend into directly (no stack round-trip). The node is read once; an alias
/// target and a `::` member/name are resolved specially.
#[inline]
fn resolve_node(
    arena: &mut NodeArena,
    rcx: &ResolveCtx,
    id: NodeId,
    stack: &mut Vec<Resolve>,
) -> ResolveResult<Option<NodeId>> {
    // Push children after the first (reversed) and return the first to descend into.
    fn descend(stack: &mut Vec<Resolve>, children: &[NodeId]) -> Option<NodeId> {
        let (&first, rest) = children.split_first()?;
        stack.extend(rest.iter().rev().map(|&c| Resolve::Node(c)));
        Some(first)
    }
    // SAFETY: every id comes from a child slice or the root, all valid arena ids.
    Ok(match unsafe { *arena.get_unchecked(id) } {
        // Local bindings are emitted as MacroParam/ForBinding by the parser, so
        // any remaining `Ident(Unresolved)` is a top-level reference.
        Node::Ident(IdentKind::Unresolved(name)) => {
            let Some(&Spanned { value: kind, .. }) = rcx.decls.get(&name) else {
                let span = arena.span(id);
                return Err(unknown_ident_error(rcx, rcx.strs.resolve(name), span));
            };
            arena.resolve_as(id, kind.to_ident(name));
            None
        }
        // Alias target: a bare identifier resolves against `decls` like any other
        // (a `let` to its value, a rule to a named-symbol reference). Only an
        // UNDECLARED bare identifier becomes a named-alias rule reference,
        // mirroring grammar.js `alias($.x, $.undeclared)`.
        Node::Alias { content, target } => {
            if let Node::Ident(IdentKind::Unresolved(name)) = *arena.get(target) {
                let kind = rcx
                    .decls
                    .get(&name)
                    .map_or(IdentKind::Rule(name), |s| s.value.to_ident(name));
                arena.resolve_as(target, kind);
            } else {
                stack.push(Resolve::Node(target));
            }
            Some(content)
        }
        // `::` access: the member/macro name resolves after the object subtree,
        // so push `Member` below the object and descend into the object.
        Node::QualifiedAccess { obj, .. } => {
            stack.push(Resolve::Member(id));
            Some(obj)
        }
        Node::QualifiedCall(range) => {
            let (obj, _name, args) = rcx.pools.get_qualified_call(range);
            stack.push(Resolve::Member(id));
            stack.extend(args.iter().rev().map(|&c| Resolve::Node(c)));
            Some(obj)
        }
        // Variadic: Seq, Choice, List, Tuple, Concat, RuleSet.
        #[rustfmt::skip]
        Node::SeqOrChoice { range, .. } | Node::List(range) | Node::Tuple(range)
        | Node::Concat(range) | Node::RuleSet(range) => descend(stack, rcx.pools.child_slice(range)),
        Node::Call { name, args } => {
            stack.extend(
                rcx.pools
                    .child_slice(args)
                    .iter()
                    .rev()
                    .map(|&c| Resolve::Node(c)),
            );
            Some(name)
        }
        Node::Object(range) => match rcx.pools.get_object(range).split_first() {
            Some((first, rest)) => {
                stack.extend(rest.iter().rev().map(|f| Resolve::Node(f.value)));
                Some(first.value)
            }
            None => None,
        },
        Node::For { for_id, body } => {
            let config = rcx.pools.get_for(for_id);
            check_shadowing(rcx, rcx.pools.param_slice(config.bindings))?;
            stack.push(Resolve::Node(body));
            Some(config.iterable)
        }
        #[rustfmt::skip]
        Node::ComputedRule { name_expr: a, body: b, .. } | Node::Append { left: a, right: b }
        | Node::BinOp { lhs: a, rhs: b, .. } | Node::Prec { value: a, content: b, .. } => {
            stack.push(Resolve::Node(b));
            Some(a)
        }
        Node::DynRegex { pattern, flags } => {
            if let Some(f) = flags {
                stack.push(Resolve::Node(f));
            }
            Some(pattern)
        }
        #[rustfmt::skip]
        Node::Repeat { inner: c, .. } | Node::Token { inner: c, .. } | Node::Neg(c)
        | Node::GrammarConfig { module: c, .. } | Node::Field { content: c, .. }
        | Node::Reserved { content: c, .. } | Node::Rule { body: c, .. }
        | Node::SymRef { expr: c } | Node::FieldAccess { obj: c, .. } => Some(c),
        _ => None,
    })
}

/// Resolve the member/macro name of a `::` access whose object is now resolved.
fn resolve_member(arena: &mut NodeArena, rcx: &ResolveCtx, id: NodeId) -> ResolveResult<()> {
    // None when obj isn't a module ref; the type checker reports the error.
    match *arena.get(id) {
        Node::QualifiedAccess { obj, member } => {
            if let Some(idx) = resolve_module_id(arena, obj) {
                let member_name = rcx.ctx.text(member);
                resolve_qualified_member(rcx, arena, idx, id, member_name, member)?;
            }
            Ok(())
        }
        Node::QualifiedCall(range) => {
            let (obj, name, _args) = rcx.pools.get_qualified_call(range);
            if let Some(idx) = resolve_module_id(arena, obj) {
                let macro_name = rcx.ctx.text(arena.span(name));
                resolve_qualified_call_name(rcx, arena, idx, name, macro_name)?;
            }
            Ok(())
        }
        _ => unreachable!(),
    }
}

/// Resolve the name node in a `QualifiedCall` to `Ident(Macro(macro_id))`.
fn resolve_qualified_call_name(
    rcx: &ResolveCtx,
    arena: &mut NodeArena,
    module: ModuleId,
    name_id: NodeId,
    macro_name: &str,
) -> ResolveResult<()> {
    let target = &rcx.modules[usize::from(module)];
    match rcx.strs.get(macro_name).and_then(|id| target.export(id)) {
        Some(Export::Local(kind @ IdentKind::Macro(_))) => arena.set(name_id, Node::Ident(kind)),
        None => Err(import_member_not_found(
            rcx,
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
    rcx: &ResolveCtx,
    arena: &mut NodeArena,
    module: ModuleId,
    node_id: NodeId,
    member_name: &str,
    member_span: Span,
) -> ResolveResult<()> {
    let target = &rcx.modules[usize::from(module)];
    match rcx.strs.get(member_name).and_then(|id| target.export(id)) {
        Some(Export::Local(kind)) => arena.set(node_id, Node::Ident(kind)),
        Some(Export::Rule(target)) => arena.set(node_id, Node::ModuleRule { module, target }),
        None => Err(import_member_not_found(
            rcx,
            target,
            member_name,
            member_span,
        ))?,
    }
    Ok(())
}

/// Build an `ImportMemberNotFound` error, attaching a "did you mean" note when appropriate
fn import_member_not_found(
    rcx: &ResolveCtx,
    target: &Module,
    name: &str,
    span: Span,
) -> ResolveError {
    let kind = ResolveErrorKind::ImportMemberNotFound(name.to_string());
    let candidates = target.export_keys().map(|id| rcx.strs.resolve(id));
    if let Some(suggestion) = suggest_name(name, candidates) {
        return ResolveError::with_note(
            kind,
            span,
            rcx.ctx
                .note(NoteMessage::DidYouMean(suggestion.to_string()), span),
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

/// Walk `Ident(Var) -> Let.value` chains to find the underlying `ModuleRef` node.
/// A self-referential or mutually-recursive `let` (e.g. `let a = a`) bails with
/// `None`; the cycle is then reported gracefully by typecheck (`CircularLet`).
pub(super) fn resolve_module_ref(arena: &NodeArena, mut obj: NodeId) -> Option<NodeId> {
    // A chain longer than the arena has necessarily revisited a let, i.e. cycled.
    for _ in 0..arena.len() {
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
    None
}

/// Recursively collect external token names from an expression.
fn collect_external_names(
    shared: &SharedAst,
    ctx: &ModuleContext,
    strs: &mut StrPool,
    id: NodeId,
    ec: &mut ExternalNameCtx<'_>,
) -> ResolveResult<()> {
    let arena = &shared.arena;
    match arena.get(id) {
        // Bare identifier: follow a let, ignore an already-declared name, or
        // register an unknown name as a new external token.
        Node::Ident(IdentKind::Unresolved(name)) => match ec.decls.get(name).map(|d| d.value) {
            Some(DeclKind::Var(let_id)) => {
                if !ec.expanding_lets.insert(*name) {
                    return Err(ResolveError::new(
                        ResolveErrorKind::InvalidExternalsExpression,
                        arena.span(id),
                    ));
                }
                expect_pat!(Node::Let { value, .. }, *arena.get(let_id));
                collect_external_names(shared, ctx, strs, value, ec)?;
                ec.expanding_lets.remove(name);
            }
            None => insert_decl(ec.decls, strs, *name, DeclKind::Rule, arena.span(id), ctx)?,
            Some(_) => {}
        },
        Node::List(range) | Node::SeqOrChoice { range, .. } | Node::Tuple(range) => {
            for &child in shared.pools.child_slice(*range) {
                collect_external_names(shared, ctx, strs, child, ec)?;
            }
        }
        Node::Append { left, right } => {
            collect_external_names(shared, ctx, strs, *left, ec)?;
            collect_external_names(shared, ctx, strs, *right, ec)?;
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
    #[error("computed rule name '{0}' does not name a rule")]
    ComputedNameNotARule(String),
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
fn unknown_ident_error(rcx: &ResolveCtx, name: &str, span: Span) -> ResolveError {
    let kind = ResolveErrorKind::UnknownIdentifier(name.to_string());
    let candidates = rcx.decls.keys().map(|&id| rcx.strs.resolve(id)).chain(
        super::lexer::TokenKind::COMBINATOR_KEYWORD_NAMES
            .iter()
            .copied(),
    );
    if let Some(suggestion) = suggest_name(name, candidates) {
        return ResolveError::with_note(
            kind,
            span,
            rcx.ctx
                .note(NoteMessage::DidYouMean(suggestion.to_string()), span),
        );
    }
    ResolveError::new(kind, span)
}
