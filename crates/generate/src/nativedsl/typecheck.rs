//! Name resolution and type checking for the native grammar DSL.
//!
//! Entry point is [`resolve_and_check`]: collects declarations, resolves
//! identifiers, then type-checks the resolved AST.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use thiserror::Error;

use crate::grammars::InputGrammar;

use super::{
    Note, NoteMessage,
    ast::{
        AstPools, ChildRange, ForId, IdentKind, ModuleContext, Node, NodeArena, NodeId, PrecKind,
        SharedAst, Span,
    },
};

/// Function pointer type for element/inner checkers passed to `expect_list` and
/// `expect_list_list`. Checks a single node against the type environment.
type CheckFn<'ast> = fn(
    &SharedAst,
    &'ast ModuleContext,
    NodeId,
    &mut TypeEnv<'ast>,
    &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError>;

/// Types that can appear as homogeneous object field values.
/// Each variant maps 1:1 to a `Ty` variant of the same name.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum InnerTy {
    Rule,
    Str,
    Int,
    ListRule,
    ListStr,
    ListInt,
    ListListRule,
    ListListStr,
    ListListInt,
}

impl std::fmt::Display for InnerTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ty::from(*self).fmt(f)
    }
}

macro_rules! inner_ty_conversions {
    ($($variant:ident),+ $(,)?) => {
        impl From<InnerTy> for Ty {
            fn from(inner: InnerTy) -> Self {
                match inner { $(InnerTy::$variant => Self::$variant),+ }
            }
        }
        impl TryFrom<Ty> for InnerTy {
            type Error = ();
            fn try_from(ty: Ty) -> Result<Self, ()> {
                match ty { $(Ty::$variant => Ok(Self::$variant),)+ _ => Err(()) }
            }
        }
    };
}
inner_ty_conversions!(
    Rule,
    Str,
    Int,
    ListRule,
    ListStr,
    ListInt,
    ListListRule,
    ListListStr,
    ListListInt
);

/// The type of a DSL expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Ty {
    Rule,
    Str,
    Int,
    ListRule,
    ListStr,
    ListInt,
    ListListRule,
    ListListStr,
    ListListInt,
    /// Result of `import(...)`: a helper module namespace.
    /// Members accessed via `::`. The `u8` indexes into the module list.
    ImportModule(u8),
    /// Result of `inherit(...)`: a grammar module with rules and config.
    /// Members accessed via `::`, config accessed via `grammar_config()`.
    GrammarModule(u8),
    /// User-facing module type annotation (`module_t`). Matches any
    /// concrete module via `is_compatible`.
    AnyModule,
    Spread,
    /// Homogeneous object `{ field: T, ... }`. The inner type is what field
    /// access returns. Field name validation is done via `ObjectInfo` in the env.
    Object(InnerTy),
}

impl Ty {
    fn is_rule_like(self) -> bool {
        self == Self::Rule || self == Self::Str
    }

    const fn is_list(self) -> bool {
        self.elem_type().is_some()
    }

    /// Check if `self` is assignable to `expected`.
    fn is_compatible(self, expected: Self) -> bool {
        self == expected
            || (self == Self::Str && expected == Self::Rule)
            || (self == Self::ListStr && expected == Self::ListRule)
            || (self == Self::ListListStr && expected == Self::ListListRule)
            || (matches!(self, Self::ImportModule(_) | Self::GrammarModule(_))
                && expected == Self::AnyModule)
            || matches!(
                (self, expected),
                (Self::Object(InnerTy::Str), Self::Object(InnerTy::Rule))
                    | (
                        Self::Object(InnerTy::ListStr),
                        Self::Object(InnerTy::ListRule)
                    )
                    | (
                        Self::Object(InnerTy::ListListStr),
                        Self::Object(InnerTy::ListListRule)
                    )
            )
    }

    fn is_bindable(self) -> bool {
        self != Self::Spread
    }

    /// Promote a type to its list wrapper (e.g. `Rule` -> `ListRule`).
    /// Returns `None` for types that can't be list elements.
    pub(super) const fn to_list(self) -> Option<Self> {
        Some(match self {
            Self::Rule => Self::ListRule,
            Self::Str => Self::ListStr,
            Self::Int => Self::ListInt,
            Self::ListRule => Self::ListListRule,
            Self::ListStr => Self::ListListStr,
            Self::ListInt => Self::ListListInt,
            _ => return None,
        })
    }

    /// Unwrap a list type to its element type (e.g. `ListRule` -> `Rule`).
    /// Returns `None` for non-list types.
    const fn elem_type(self) -> Option<Self> {
        Some(match self {
            Self::ListRule => Self::Rule,
            Self::ListStr => Self::Str,
            Self::ListInt => Self::Int,
            Self::ListListRule => Self::ListRule,
            Self::ListListStr => Self::ListStr,
            Self::ListListInt => Self::ListInt,
            _ => return None,
        })
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rule => f.write_str("rule_t"),
            Self::Str => f.write_str("str_t"),
            Self::Int => f.write_str("int_t"),
            Self::ListRule => f.write_str("list_t<rule_t>"),
            Self::ListStr => f.write_str("list_t<str_t>"),
            Self::ListInt => f.write_str("list_t<int_t>"),
            Self::ListListRule => f.write_str("list_t<list_t<rule_t>>"),
            Self::ListListStr => f.write_str("list_t<list_t<str_t>>"),
            Self::ListListInt => f.write_str("list_t<list_t<int_t>>"),
            Self::ImportModule(_) | Self::GrammarModule(_) | Self::AnyModule => {
                f.write_str("module_t")
            }
            Self::Spread => f.write_str("spread_t"),
            Self::Object(inner) => write!(f, "obj_t<{inner}>"),
        }
    }
}

#[derive(Clone)]
pub struct MacroSig {
    pub params: Vec<Ty>,
    pub return_ty: Ty,
}

#[derive(Clone)]
pub struct TypeEnv<'src> {
    pub vars: FxHashMap<&'src str, Ty>,
    pub fns: FxHashMap<&'src str, MacroSig>,
    /// Field names for Object variables, keyed by variable name.
    pub object_fields: FxHashMap<&'src str, Vec<&'src str>>,
}

impl<'src> TypeEnv<'src> {
    fn new() -> Self {
        Self {
            vars: FxHashMap::default(),
            fns: FxHashMap::default(),
            object_fields: FxHashMap::default(),
        }
    }
    fn insert_var(&mut self, name: &'src str, ty: Ty) {
        self.vars.insert(name, ty);
    }
    fn insert_object(&mut self, name: &'src str, ty: Ty, fields: Vec<&'src str>) {
        self.insert_var(name, ty);
        self.object_fields.insert(name, fields);
    }
    #[inline]
    fn get_object_fields(&self, name: &str) -> Option<&[&'src str]> {
        self.object_fields.get(name).map(Vec::as_slice)
    }
    fn get_var(&self, name: &str) -> Option<Ty> {
        self.vars.get(name).copied()
    }
}

type LocalNames<'a> = FxHashSet<&'a str>;

#[derive(Clone, Copy)]
enum Decl {
    Rule,
    Var,
    Macro,
}

/// Run name resolution and type checking in a single function.
///
/// **Phase 1** (resolve): collects declarations, registers inherited names,
/// and resolves `Node::Ident` nodes from `Unresolved` to `Rule` / `Var`.
///
/// **Phase 2** (typecheck): calls the existing `check()` function on the
/// now-resolved AST.
///
/// # Errors
///
/// Returns [`TypeError`] for duplicate declarations, unknown identifiers,
/// or any type error.
pub fn resolve_and_check<'ast>(
    shared: &'ast mut SharedAst,
    ctx: &'ast ModuleContext,
    modules: &[Option<TypeEnv<'ast>>],
    base: Option<(&'ast InputGrammar, Span)>,
    grammar_path: &Path,
) -> Result<TypeEnv<'ast>, TypeError> {
    // Phase 1: Name resolution
    let mut decls = collect_decls(shared, ctx, &ctx.root_items, grammar_path)?;

    // Register inherited rule names so they resolve as rules.
    // Use the inherit statement's span so "first defined here" points there.
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            if !decls.contains_key(var.name.as_str()) {
                decls.insert(&var.name, (Decl::Rule, inherit_span));
            }
        }
    }

    let n_items = ctx.root_items.len();
    for i in 0..n_items {
        let item_id = ctx.root_items[i];
        resolve_item_tc(&mut shared.arena, &shared.pools, ctx, &decls, item_id)?;
        // Register let bindings in order so forward references are caught
        if let Node::Let { name, .. } = shared.arena.get(item_id) {
            let name_text = ctx.text(*name);
            let span = shared.arena.span(item_id);
            insert_decl(
                &mut decls,
                name_text,
                Decl::Var,
                span,
                grammar_path,
                &ctx.source,
            )?;
        }
    }

    // Phase 2: Type checking
    check(shared, ctx, modules)
}

/// Intermediate resolve environment used during phase 1. Maps declaration
/// names to their kind (for Ident resolution) and span
/// (for duplicate checking / "first defined here" notes).
type Decls<'src> = FxHashMap<&'src str, (Decl, Span)>;

fn insert_decl<'src>(
    decls: &mut Decls<'src>,
    name: &'src str,
    kind: Decl,
    span: Span,
    grammar_path: &Path,
    source: &str,
) -> Result<(), TypeError> {
    if let Some(&(_, first_span)) = decls.get(name) {
        return Err(TypeError::with_note(
            TypeErrorKind::DuplicateDeclaration(name.to_string()),
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
/// Rules, let-bindings, functions, and external tokens in the grammar block
/// all occupy the same namespace. Duplicate names are rejected.
fn collect_decls<'a>(
    shared: &SharedAst,
    ctx: &'a ModuleContext,
    root_items: &[NodeId],
    grammar_path: &Path,
) -> Result<Decls<'a>, TypeError> {
    let mut decls = Decls::default();
    let source = &ctx.source;

    // Register rules and fns upfront (forward references allowed).
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
            Node::Macro(fn_idx) => {
                let config = shared.pools.get_macro(*fn_idx);
                insert_decl(
                    &mut decls,
                    ctx.text(config.name),
                    Decl::Macro,
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
                    return Err(TypeError::new(
                        TypeErrorKind::UnknownIdentifier(lhs.to_string()),
                        shared.arena.span(*value),
                    ));
                }
            }
            _ => {}
        }
    }

    // Pre-register external token names. Externals are the only config field
    // that introduces names not declared as rules - they're provided by an
    // external scanner.
    if let Some(config) = &ctx.grammar_config
        && let Some(ext_id) = config.externals
    {
        let mut ec = ExternalNameCtx {
            decls: &mut decls,
            root_items,
            grammar_path,
            source,
            visited: FxHashSet::default(),
        };
        collect_external_names(shared, ctx, ext_id, &mut ec)?;
    }

    Ok(decls)
}

/// Context for [`collect_external_names`].
struct ExternalNameCtx<'src, 'a, 'b> {
    decls: &'a mut Decls<'src>,
    root_items: &'b [NodeId],
    grammar_path: &'b Path,
    source: &'src str,
    visited: FxHashSet<NodeId>,
}

/// Recursively collect external token names from an expression.
fn collect_external_names<'src>(
    shared: &SharedAst,
    ctx: &'src ModuleContext,
    id: NodeId,
    ec: &mut ExternalNameCtx<'src, '_, '_>,
) -> Result<(), TypeError> {
    let arena = &shared.arena;
    if !ec.visited.insert(id) {
        return Err(TypeError::new(
            TypeErrorKind::InvalidExternalsExpression,
            arena.span(id),
        ));
    }
    match arena.get(id) {
        // Bare identifier: either a let binding (follow it) or an external token name.
        Node::Ident(IdentKind::Unresolved) => {
            let name = ctx.text(arena.span(id));
            if let Some(value_id) = find_let_value(arena, ctx, ec.root_items, name) {
                // It's a let binding - recurse into its value.
                collect_external_names(shared, ctx, value_id, ec)?;
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
        // Anything else (function calls, for-loops, etc.) is not a valid
        // externals expression. Externals must be a list literal, append,
        // or variable reference.
        _ => {
            return Err(TypeError::new(
                TypeErrorKind::InvalidExternalsExpression,
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

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item_tc(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    item_id: NodeId,
) -> Result<(), TypeError> {
    match arena.get(item_id) {
        Node::Grammar => {
            // INVARIANT: set during grammar block parsing, always present here
            let grammar_config = ctx.grammar_config.as_ref().unwrap();
            for id in [
                grammar_config.inherits,
                grammar_config.extras,
                grammar_config.externals,
                grammar_config.inline,
                grammar_config.supertypes,
                grammar_config.word,
                grammar_config.conflicts,
                grammar_config.precedences,
                grammar_config.reserved,
            ]
            .into_iter()
            .flatten()
            {
                resolve_expr_tc(arena, pools, ctx, decls, id, &LocalNames::default())?;
            }
            Ok(())
        }
        Node::Rule { body: inner, .. } | Node::Let { value: inner, .. } => {
            let inner = *inner;
            resolve_expr_tc(arena, pools, ctx, decls, inner, &LocalNames::default())
        }
        Node::Macro(fn_idx) => {
            let fn_config = pools.get_macro(*fn_idx);
            // Macro params must not shadow any top-level declaration
            for param in &fn_config.params {
                let name = ctx.text(param.name);
                if let Some(&(_, first_span)) = decls.get(name) {
                    return Err(TypeError::with_note(
                        TypeErrorKind::ShadowedBinding(name.to_string()),
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
            let body = fn_config.body;
            let locals: LocalNames<'_> =
                fn_config.params.iter().map(|p| ctx.text(p.name)).collect();
            resolve_expr_tc(arena, pools, ctx, decls, body, &locals)
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
fn resolve_expr_tc(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    id: NodeId,
    locals: &LocalNames<'_>,
) -> Result<(), TypeError> {
    // Handle Ident mutation first
    if matches!(arena.get(id), Node::Ident(IdentKind::Unresolved)) {
        let span = arena.span(id);
        let name = ctx.text(span);
        if locals.contains(name) {
            arena.resolve_as(id, IdentKind::Var);
        } else if let Some(&(decl, _)) = decls.get(name) {
            match decl {
                Decl::Var => arena.resolve_as(id, IdentKind::Var),
                Decl::Rule => arena.resolve_as(id, IdentKind::Rule),
                Decl::Macro => arena.resolve_as(id, IdentKind::Macro),
            }
        } else {
            return Err(TypeError::new(
                TypeErrorKind::UnknownIdentifier(name.to_string()),
                span,
            ));
        }
        return Ok(());
    }

    // Alias target: bare identifiers resolve as Rule (named alias) unless
    // they refer to a local variable (Var for fn params / for bindings).
    // Targets don't require the name to be declared - they're just node names,
    // same as grammar.js `alias($.x, $.undeclared_name)`.
    if let Node::Alias { content, target } = arena.get(id) {
        let (content, target) = (*content, *target);
        resolve_expr_tc(arena, pools, ctx, decls, content, locals)?;
        if matches!(*arena.get(target), Node::Ident(IdentKind::Unresolved)) {
            let name = ctx.text(arena.span(target));
            if locals.contains(name) {
                arena.resolve_as(target, IdentKind::Var);
            } else {
                arena.resolve_as(target, IdentKind::Rule);
            }
        } else {
            resolve_expr_tc(arena, pools, ctx, decls, target, locals)?;
        }
        return Ok(());
    }

    // For expressions need extended locals
    if let Node::For(for_idx) = arena.get(id) {
        let config = pools.get_for(*for_idx);
        // For-loop bindings must not shadow any outer name
        for &(name_span, _) in &config.bindings {
            let name = ctx.text(name_span);
            if let Some(&(_, first_span)) = decls.get(name) {
                return Err(TypeError::with_note(
                    TypeErrorKind::ShadowedBinding(name.to_string()),
                    name_span,
                    Note {
                        message: NoteMessage::FirstDefinedHere,
                        span: first_span,
                        path: ctx.path.clone(),
                        source: ctx.source.clone(),
                    },
                ));
            }
            if locals.contains(name) {
                return Err(TypeError::new(
                    TypeErrorKind::ShadowedBinding(name.to_string()),
                    name_span,
                ));
            }
        }
        let iterable = config.iterable;
        let body = config.body;
        let bindings = &config.bindings;
        resolve_expr_tc(arena, pools, ctx, decls, iterable, locals)?;
        let mut inner = locals.clone();
        for &(name_span, _) in bindings {
            inner.insert(ctx.text(name_span));
        }
        return resolve_expr_tc(arena, pools, ctx, decls, body, &inner);
    }

    // All remaining cases
    resolve_children_tc(arena, pools, ctx, decls, id, locals)
}

/// Resolve identifiers in children of a node (mechanical traversal).
fn resolve_children_tc(
    arena: &mut NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    decls: &Decls,
    id: NodeId,
    locals: &LocalNames<'_>,
) -> Result<(), TypeError> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = arena.get(id).child_range() {
        for child in pools.child_slice(range) {
            resolve_expr_tc(arena, pools, ctx, decls, *child, locals)?;
        }
        return Ok(());
    }

    match arena.get(id) {
        Node::Call { name, args } => {
            let (name, args) = (*name, *args);
            resolve_expr_tc(arena, pools, ctx, decls, name, locals)?;
            for arg in pools.child_slice(args) {
                resolve_expr_tc(arena, pools, ctx, decls, *arg, locals)?;
            }
            Ok(())
        }
        Node::Object(range) => {
            for field in pools.get_object(*range) {
                resolve_expr_tc(arena, pools, ctx, decls, field.1, locals)?;
            }
            Ok(())
        }
        Node::Repeat { inner: c, .. }
        | Node::Token { inner: c, .. }
        | Node::Neg(c)
        | Node::GrammarConfig { module: c, .. }
        | Node::Field { content: c, .. }
        | Node::Reserved { content: c, .. } => {
            resolve_expr_tc(arena, pools, ctx, decls, *c, locals)
        }
        Node::Append { left: a, right: b }
        | Node::Prec {
            value: a,
            content: b,
            ..
        } => {
            let (a, b) = (*a, *b);
            resolve_expr_tc(arena, pools, ctx, decls, a, locals)?;
            resolve_expr_tc(arena, pools, ctx, decls, b, locals)
        }
        Node::DynRegex(range) => {
            let (pattern, flags) = pools.get_regex(*range);
            resolve_expr_tc(arena, pools, ctx, decls, pattern, locals)?;
            if let Some(flags) = flags {
                resolve_expr_tc(arena, pools, ctx, decls, flags, locals)?;
            }
            Ok(())
        }
        Node::FieldAccess { obj, .. } | Node::QualifiedAccess { obj, .. } => {
            let obj = *obj;
            resolve_expr_tc(arena, pools, ctx, decls, obj, locals)
        }
        Node::QualifiedCall(range) => {
            let (obj, _name, args) = pools.get_qualified_call(*range);
            resolve_expr_tc(arena, pools, ctx, decls, obj, locals)?;
            // name stays as Ident - it's a member of the import namespace
            for &arg in args {
                resolve_expr_tc(arena, pools, ctx, decls, arg, locals)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Run typechecking and return the type environment.
///
/// `modules` holds imported modules' type environments (indexed by
/// `Ty::ImportModule(idx)` / `Ty::GrammarModule(idx)`), populated by the import pre-pass.
pub fn check<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<TypeEnv<'ast>, TypeError> {
    let mut env = TypeEnv::new();
    for &item_id in &ctx.root_items {
        match shared.arena.get(item_id) {
            Node::Rule { name, .. } => {
                env.insert_var(ctx.text(*name), Ty::Rule);
            }
            Node::Macro(fn_idx) => {
                let config = shared.pools.get_macro(*fn_idx);
                let params: Vec<Ty> = config.params.iter().map(|p| p.ty).collect();
                let return_ty = config.return_ty;
                env.fns
                    .insert(ctx.text(config.name), MacroSig { params, return_ty });
            }
            _ => {}
        }
    }
    for &item_id in &ctx.root_items {
        check_item(shared, ctx, item_id, &mut env, modules)?;
    }
    Ok(env)
}

fn check_item<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    match shared.arena.get(id) {
        Node::Grammar => {
            let config = ctx.grammar_config.as_ref().unwrap();
            for id in [config.extras, config.externals].into_iter().flatten() {
                expect_rule_list(shared, ctx, id, env, modules)?;
            }
            for id in [config.inline, config.supertypes].into_iter().flatten() {
                expect_name_list(shared, ctx, id, env, modules)?;
            }
            if let Some(id) = config.conflicts {
                expect_list_of_groups(shared, ctx, id, env, modules, expect_name_list)?;
            }
            if let Some(id) = config.precedences {
                expect_list_of_groups(shared, ctx, id, env, modules, expect_precedence_group)?;
            }
            if let Some(id) = config.word {
                expect_name_ref(shared, ctx, id, env, modules)?;
            }
            if let Some(id) = config.reserved {
                expect_reserved(shared, ctx, id, env, modules)?;
            }
            Ok(())
        }
        Node::Let { name, ty, value } => {
            let var_name = ctx.text(*name);
            let declared_ty = *ty;
            // For empty containers, the inner type comes from annotation only
            let is_empty_list = matches!(shared.arena.get(*value), Node::List(r) if shared.pools.child_slice(*r).is_empty());
            let is_empty_obj = matches!(shared.arena.get(*value), Node::Object(r) if shared.pools.get_object(*r).is_empty());
            let resolved_ty = match (is_empty_list, is_empty_obj, declared_ty) {
                (true, _, Some(ty)) if ty.is_list() => ty,
                (_, true, Some(ty)) if matches!(ty, Ty::Object(_)) => ty,
                (true, _, None) | (_, true, None) => {
                    return Err(TypeError::new(
                        TypeErrorKind::EmptyContainerNeedsAnnotation,
                        shared.arena.span(*value),
                    ));
                }
                (_, _, Some(ty)) => {
                    let inferred = type_of(shared, ctx, *value, env, modules)?;
                    if !inferred.is_compatible(ty) {
                        return Err(mismatch(ty, inferred, shared.arena.span(*value)));
                    }
                    inferred
                }
                (_, _, None) => type_of(shared, ctx, *value, env, modules)?,
            };
            if !resolved_ty.is_bindable() {
                return Err(TypeError::new(
                    TypeErrorKind::NonBindableType(resolved_ty),
                    shared.arena.span(*value),
                ));
            }
            // If the value is an object literal, register its field names
            if let Node::Object(range) = shared.arena.get(*value) {
                let fields: Vec<&str> = shared
                    .pools
                    .get_object(*range)
                    .iter()
                    .map(|(span, _)| ctx.text(*span))
                    .collect();
                env.insert_object(var_name, resolved_ty, fields);
            } else {
                env.insert_var(var_name, resolved_ty);
            }
            Ok(())
        }
        Node::Macro(fn_idx) => {
            let config = shared.pools.get_macro(*fn_idx);
            let fn_name = ctx.text(config.name);
            let sig = &env.fns[fn_name];
            let return_ty = sig.return_ty;
            let n_params = sig.params.len();
            for i in 0..n_params {
                let ty = env.fns[fn_name].params[i];
                env.insert_var(ctx.text(config.params[i].name), ty);
            }
            let body_ty = type_of(shared, ctx, config.body, env, modules)?;
            for param in &config.params {
                env.vars.remove(ctx.text(param.name));
            }
            if !body_ty.is_compatible(return_ty) {
                return Err(mismatch(return_ty, body_ty, shared.arena.span(config.body)));
            }
            Ok(())
        }
        Node::Rule { body, .. } => expect_rule(shared, ctx, *body, env, modules),
        _ => unreachable!(),
    }
}

/// Widen `widest` to accommodate `ty`. Returns the wider type, or `None` if incompatible.
fn widen_type(widest: Ty, ty: Ty) -> Option<Ty> {
    if ty == widest || ty.is_compatible(widest) {
        Some(widest)
    } else if widest.is_compatible(ty) {
        Some(ty)
    } else {
        None
    }
}

/// Check a list config field. For literal lists, checks each element with `check_elem`.
/// For non-literal expressions, checks the overall type is a list type.
fn expect_list<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
    check_elem: CheckFn<'ast>,
    accept_list_str: bool,
) -> Result<(), TypeError> {
    if let Node::List(range) = shared.arena.get(id) {
        for &child in shared.pools.child_slice(*range) {
            check_elem(shared, ctx, child, env, modules)?;
        }
        return Ok(());
    }
    let ty = type_of(shared, ctx, id, env, modules)?;
    if ty == Ty::ListRule || (accept_list_str && ty == Ty::ListStr) {
        return Ok(());
    }
    Err(mismatch(Ty::ListRule, ty, shared.arena.span(id)))
}

fn expect_rule_list<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    expect_list(shared, ctx, id, env, modules, expect_rule, true)
}

fn expect_name_list<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    expect_list(shared, ctx, id, env, modules, expect_name_ref, false)
}

fn expect_name_ref<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    match shared.arena.get(id) {
        Node::Ident(IdentKind::Rule) => Ok(()),
        Node::Ident(IdentKind::Var) | Node::FieldAccess { .. } | Node::GrammarConfig { .. } => {
            let ty = type_of(shared, ctx, id, env, modules)?;
            if ty != Ty::Rule {
                return Err(mismatch(Ty::Rule, ty, shared.arena.span(id)));
            }
            Ok(())
        }
        _ => Err(TypeError::new(
            TypeErrorKind::ExpectedRuleName,
            shared.arena.span(id),
        )),
    }
}

/// Check a `list_list_rule_t` config field, validating inner elements.
fn expect_list_of_groups<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
    check_inner: CheckFn<'ast>,
) -> Result<(), TypeError> {
    if let Node::List(range) = shared.arena.get(id) {
        for &child in shared.pools.child_slice(*range) {
            check_inner(shared, ctx, child, env, modules)?;
        }
        return Ok(());
    }
    let ty = type_of(shared, ctx, id, env, modules)?;
    if ty.is_compatible(Ty::ListListRule) {
        return Ok(());
    }
    Err(mismatch(Ty::ListListRule, ty, shared.arena.span(id)))
}

/// Inner element for `precedences`: a name ref or a string literal.
fn expect_name_or_str<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    if matches!(
        shared.arena.get(id),
        Node::StringLit | Node::RawStringLit { .. }
    ) {
        return Ok(());
    }
    expect_name_ref(shared, ctx, id, env, modules)
}

/// Check a `precedences` inner list: each element must be a name-ref or string literal.
fn expect_precedence_group<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    expect_list(shared, ctx, id, env, modules, expect_name_or_str, true)
}

fn expect_reserved<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    // Object literal: each field value must be a rule list
    if let Node::Object(range) = shared.arena.get(id) {
        for &(_, val_id) in shared.pools.get_object(*range) {
            expect_rule_list(shared, ctx, val_id, env, modules)?;
        }
        return Ok(());
    }
    // Non-literal: must be inherited (base.reserved -> Object(ListRule))
    let ty = type_of(shared, ctx, id, env, modules)?;
    if ty.is_compatible(Ty::Object(InnerTy::ListRule)) {
        return Ok(());
    }
    Err(TypeError::new(
        TypeErrorKind::ExpectedReservedConfig,
        shared.arena.span(id),
    ))
}

fn expect_rule<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    let ty = type_of(shared, ctx, id, env, modules)?;
    if !ty.is_rule_like() {
        return Err(mismatch(Ty::Rule, ty, shared.arena.span(id)));
    }
    Ok(())
}

fn type_of<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let span = shared.arena.span(id);
    match shared.arena.get(id) {
        Node::IntLit(_) => Ok(Ty::Int),
        Node::StringLit | Node::RawStringLit { .. } => Ok(Ty::Str),
        Node::Ident(IdentKind::Rule) | Node::Blank => Ok(Ty::Rule),
        Node::ModuleRef { import, module, .. } => {
            let idx = module.expect("module index not set by loading pre-pass");
            Ok(if *import {
                Ty::ImportModule(idx)
            } else {
                Ty::GrammarModule(idx)
            })
        }
        Node::GrammarConfig { module, field } => {
            let module_ty = type_of(shared, ctx, *module, env, modules)?;
            if !matches!(module_ty, Ty::GrammarModule(_)) {
                return Err(TypeError::new(
                    TypeErrorKind::GrammarConfigRequiresInherit,
                    shared.arena.span(*module),
                ));
            }
            use super::ast::QueryableField as Q;
            Ok(match field {
                Q::Extras | Q::Externals | Q::Inline | Q::Supertypes => Ty::ListRule,
                Q::Conflicts | Q::Precedences => Ty::ListListRule,
                Q::Word => Ty::Rule,
                Q::Reserved => Ty::Object(InnerTy::ListRule),
            })
        }
        Node::Ident(IdentKind::Unresolved) => unreachable!(),
        Node::Ident(IdentKind::Macro) => Err(TypeError::new(
            TypeErrorKind::FunctionUsedAsValue(ctx.text(span).to_string()),
            span,
        )),
        Node::Ident(IdentKind::Var) => {
            let name = ctx.text(span);
            env.get_var(name).ok_or_else(|| {
                TypeError::new(TypeErrorKind::UnresolvedVariable(name.to_string()), span)
            })
        }
        Node::Neg(inner) => {
            let ty = type_of(shared, ctx, *inner, env, modules)?;
            if ty != Ty::Int {
                return Err(mismatch(Ty::Int, ty, span));
            }
            Ok(Ty::Int)
        }
        Node::Append { left, right } => {
            type_of_append(shared, ctx, *left, *right, span, env, modules)
        }
        Node::FieldAccess { obj, field } => {
            type_of_field_access(shared, ctx, *obj, *field, env, modules)
        }
        Node::QualifiedAccess { obj, member } => {
            let obj_ty = type_of(shared, ctx, *obj, env, modules)?;
            match obj_ty {
                Ty::ImportModule(idx) | Ty::GrammarModule(idx) => {
                    type_of_import_access(ctx, idx, *member, modules)
                }
                _ => Err(TypeError::new(
                    TypeErrorKind::QualifiedAccessOnInvalidType(obj_ty),
                    shared.arena.span(*obj),
                )),
            }
        }
        Node::Object(range) => type_of_object(shared, ctx, *range, span, env, modules),
        Node::List(range) => type_of_list(shared, ctx, *range, span, env, modules),
        Node::Tuple(_) => {
            // Tuples are only valid as elements of for-loop lists.
            // They're checked structurally in check_for_expr.
            // If we reach here, it's a tuple in an invalid context.
            Err(TypeError::new(TypeErrorKind::CannotInferType, span))
        }
        Node::Concat(range) => {
            for &part in shared.pools.child_slice(*range) {
                let ty = type_of(shared, ctx, part, env, modules)?;
                if ty != Ty::Str {
                    return Err(mismatch(Ty::Str, ty, shared.arena.span(part)));
                }
            }
            Ok(Ty::Str)
        }
        Node::DynRegex(range) => {
            let (pattern, flags) = shared.pools.get_regex(*range);
            let pt = type_of(shared, ctx, pattern, env, modules)?;
            if pt != Ty::Str {
                return Err(mismatch(Ty::Str, pt, shared.arena.span(pattern)));
            }
            if let Some(fid) = flags {
                let ft = type_of(shared, ctx, fid, env, modules)?;
                if ft != Ty::Str {
                    return Err(mismatch(Ty::Str, ft, shared.arena.span(fid)));
                }
            }
            Ok(Ty::Rule)
        }
        Node::SeqOrChoice { range, .. } => {
            for &member in shared.pools.child_slice(*range) {
                let ty = type_of(shared, ctx, member, env, modules)?;
                if ty != Ty::Spread && !ty.is_rule_like() {
                    return Err(mismatch(Ty::Rule, ty, shared.arena.span(member)));
                }
            }
            Ok(Ty::Rule)
        }
        Node::Repeat { inner, .. }
        | Node::Token { inner, .. }
        | Node::Field { content: inner, .. }
        | Node::Reserved { content: inner, .. } => {
            expect_rule(shared, ctx, *inner, env, modules)?;
            Ok(Ty::Rule)
        }
        Node::Alias { content, target } => {
            expect_rule(shared, ctx, *content, env, modules)?;
            let target_ty = type_of(shared, ctx, *target, env, modules)?;
            let is_valid = matches!(shared.arena.get(*target), Node::Ident(IdentKind::Rule))
                || (matches!(shared.arena.get(*target), Node::Ident(IdentKind::Var))
                    && target_ty.is_rule_like())
                || target_ty == Ty::Str;
            if !is_valid {
                return Err(TypeError::new(
                    TypeErrorKind::InvalidAliasTarget(target_ty),
                    shared.arena.span(*target),
                ));
            }
            Ok(Ty::Rule)
        }
        Node::Prec {
            kind,
            value,
            content,
        } => {
            let vt = type_of(shared, ctx, *value, env, modules)?;
            if *kind == PrecKind::Dynamic {
                if vt != Ty::Int {
                    return Err(mismatch(Ty::Int, vt, shared.arena.span(*value)));
                }
            } else if vt != Ty::Int && vt != Ty::Str {
                return Err(TypeError::new(
                    TypeErrorKind::PrecValueTypeMismatch(vt),
                    shared.arena.span(*value),
                ));
            }
            expect_rule(shared, ctx, *content, env, modules)?;
            Ok(Ty::Rule)
        }
        Node::For(for_idx) => {
            check_for_expr(shared, ctx, *for_idx, env, modules)?;
            Ok(Ty::Spread)
        }
        Node::Call { name, args } => type_of_call(shared, ctx, *name, *args, span, env, modules),
        Node::QualifiedCall(range) => {
            type_of_qualified_call(shared, ctx, *range, span, env, modules)
        }
        _ => Err(TypeError::new(TypeErrorKind::CannotInferType, span)),
    }
}

fn type_of_append<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    left: NodeId,
    right: NodeId,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let is_empty = |id| matches!(shared.arena.get(id), Node::List(r) if shared.pools.child_slice(*r).is_empty());
    let lt = (!is_empty(left))
        .then(|| type_of(shared, ctx, left, env, modules))
        .transpose()?;
    let rt = (!is_empty(right))
        .then(|| type_of(shared, ctx, right, env, modules))
        .transpose()?;
    match (lt, rt) {
        (Some(l), Some(r)) => {
            if !l.is_list() {
                return Err(TypeError::new(
                    TypeErrorKind::AppendRequiresList(l),
                    shared.arena.span(left),
                ));
            }
            if !r.is_list() {
                return Err(mismatch(l, r, shared.arena.span(right)));
            }
            widen_type(l, r).ok_or_else(|| mismatch(l, r, shared.arena.span(right)))
        }
        (Some(t), None) | (None, Some(t)) => {
            if !t.is_list() {
                return Err(TypeError::new(TypeErrorKind::AppendRequiresList(t), span));
            }
            Ok(t)
        }
        (None, None) => Err(TypeError::new(
            TypeErrorKind::EmptyContainerNeedsAnnotation,
            span,
        )),
    }
}

fn type_of_field_access<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    obj: NodeId,
    field: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let obj_ty = type_of(shared, ctx, obj, env, modules)?;
    let field_name = ctx.text(field);
    match obj_ty {
        Ty::Object(inner) => {
            let field_known = match shared.arena.get(obj) {
                Node::Ident(IdentKind::Var) => {
                    let var_name = ctx.text(shared.arena.span(obj));
                    env.get_object_fields(var_name)
                        .is_none_or(|fields| fields.contains(&field_name))
                }
                Node::Object(range) => {
                    let fields = shared.pools.get_object(*range);
                    fields
                        .iter()
                        .any(|(key_span, _)| ctx.text(*key_span) == field_name)
                }
                _ => true, // can't validate dynamically-produced objects
            };
            if !field_known {
                return Err(TypeError::new(
                    TypeErrorKind::FieldNotFound {
                        field: field_name.to_string(),
                        on_type: obj_ty,
                    },
                    field,
                ));
            }
            Ok(Ty::from(inner))
        }
        _ => Err(TypeError::new(
            TypeErrorKind::FieldAccessOnNonObject(obj_ty),
            shared.arena.span(obj),
        )),
    }
}

fn type_of_import_access(
    ctx: &ModuleContext,
    idx: u8,
    member: Span,
    modules: &[Option<TypeEnv<'_>>],
) -> Result<Ty, TypeError> {
    let member_name = ctx.text(member);
    // Invariant: idx was assigned during loading and `typecheck_module` populates
    // the table before any module that references it is checked.
    let import_env = modules[idx as usize].as_ref().unwrap();
    if let Some(&ty) = import_env.vars.get(member_name) {
        return Ok(ty);
    }
    if import_env.fns.contains_key(member_name) {
        return Err(TypeError::new(
            TypeErrorKind::FunctionUsedAsValue(member_name.to_string()),
            member,
        ));
    }
    Err(TypeError::new(
        TypeErrorKind::ImportMemberNotFound(member_name.to_string()),
        member,
    ))
}

fn type_of_qualified_call<'a>(
    shared: &SharedAst,
    ctx: &'a ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv<'a>,
    modules: &[Option<TypeEnv<'a>>],
) -> Result<Ty, TypeError> {
    let (obj, name, args) = shared.pools.get_qualified_call(range);
    let obj_ty = type_of(shared, ctx, obj, env, modules)?;
    let (Ty::ImportModule(idx) | Ty::GrammarModule(idx)) = obj_ty else {
        return Err(TypeError::new(
            TypeErrorKind::QualifiedCallOnNonModule(obj_ty),
            shared.arena.span(obj),
        ));
    };
    let fn_name = ctx.text(shared.arena.span(name));
    let import_env = modules[idx as usize].as_ref().unwrap();
    let Some(sig) = import_env.fns.get(fn_name) else {
        return Err(TypeError::new(
            TypeErrorKind::ImportFunctionNotFound(fn_name.to_string()),
            shared.arena.span(name),
        ));
    };
    if args.len() != sig.params.len() {
        return Err(TypeError::new(
            TypeErrorKind::ArgCountMismatch {
                fn_name: fn_name.to_string(),
                expected: sig.params.len(),
                got: args.len(),
            },
            span,
        ));
    }
    let return_ty = sig.return_ty;
    let params = &sig.params;
    for (i, &arg_id) in args.iter().enumerate() {
        let expected = params[i];
        let arg_ty = type_of(shared, ctx, arg_id, env, modules)?;
        if !arg_ty.is_compatible(expected) {
            return Err(mismatch(expected, arg_ty, shared.arena.span(arg_id)));
        }
    }
    Ok(return_ty)
}

fn type_of_object<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let fields = shared.pools.get_object(range);
    if fields.is_empty() {
        return Err(TypeError::new(TypeErrorKind::CannotInferType, span));
    }
    let mut widest = type_of(shared, ctx, fields[0].1, env, modules)?;
    for &(_, val_id) in &fields[1..] {
        let ty = type_of(shared, ctx, val_id, env, modules)?;
        widest = widen_type(widest, ty)
            .ok_or_else(|| mismatch(widest, ty, shared.arena.span(val_id)))?;
    }
    let inner = InnerTy::try_from(widest).map_err(|()| {
        TypeError::new(
            TypeErrorKind::InvalidObjectValue(widest),
            shared.arena.span(fields[0].1),
        )
    })?;
    Ok(Ty::Object(inner))
}

fn type_of_list<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let items = shared.pools.child_slice(range);
    if items.is_empty() {
        return Err(TypeError::new(
            TypeErrorKind::EmptyContainerNeedsAnnotation,
            span,
        ));
    }
    let mut widest = type_of(shared, ctx, items[0], env, modules)?;
    for &item_id in &items[1..] {
        let ty = type_of(shared, ctx, item_id, env, modules)?;
        widest = widen_type(widest, ty).ok_or_else(|| {
            TypeError::new(
                TypeErrorKind::ListElementTypeMismatch {
                    first: widest,
                    got: ty,
                },
                shared.arena.span(item_id),
            )
        })?;
    }
    widest.to_list().ok_or_else(|| {
        TypeError::new(
            TypeErrorKind::InvalidListElement,
            shared.arena.span(items[0]),
        )
    })
}

fn type_of_call<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    name: NodeId,
    args: ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let fn_name = ctx.text(shared.arena.span(name));
    let sig = env.fns.get(fn_name).ok_or_else(|| {
        TypeError::new(TypeErrorKind::UndefinedFunction(fn_name.to_string()), span)
    })?;
    let return_ty = sig.return_ty;
    let n_params = sig.params.len();
    let args = shared.pools.child_slice(args);
    if args.len() != n_params {
        return Err(TypeError::new(
            TypeErrorKind::ArgCountMismatch {
                fn_name: fn_name.to_string(),
                expected: n_params,
                got: args.len(),
            },
            span,
        ));
    }
    for (i, &arg_id) in args.iter().enumerate() {
        let arg_ty = type_of(shared, ctx, arg_id, env, modules)?;
        let param_ty = env.fns[fn_name].params[i];
        if !arg_ty.is_compatible(param_ty) {
            return Err(mismatch(param_ty, arg_ty, shared.arena.span(arg_id)));
        }
    }
    Ok(return_ty)
}

fn check_for_expr<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    for_idx: ForId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    let config = shared.pools.get_for(for_idx);

    // Check if iterable is a literal list of tuples (for destructuring)
    if let Node::List(range) = shared.arena.get(config.iterable) {
        let items = shared.pools.child_slice(*range);
        let has_tuples = items
            .first()
            .is_some_and(|&id| matches!(shared.arena.get(id), Node::Tuple(_)));
        if has_tuples {
            for &item_id in items {
                let Node::Tuple(range) = shared.arena.get(item_id) else {
                    return Err(TypeError::new(
                        TypeErrorKind::ForRequiresTuples,
                        shared.arena.span(item_id),
                    ));
                };
                let elems = shared.pools.child_slice(*range);
                if elems.len() != config.bindings.len() {
                    return Err(TypeError::new(
                        TypeErrorKind::ForBindingCountMismatch {
                            bindings: config.bindings.len(),
                            tuple_elements: elems.len(),
                        },
                        shared.arena.span(item_id),
                    ));
                }
                for (i, &(name_span, declared_ty)) in config.bindings.iter().enumerate() {
                    let actual = type_of(shared, ctx, elems[i], env, modules)?;
                    if !actual.is_compatible(declared_ty) {
                        return Err(mismatch(declared_ty, actual, name_span));
                    }
                }
            }
            for &(name_span, declared_ty) in &config.bindings {
                env.insert_var(ctx.text(name_span), declared_ty);
            }
            let result = expect_rule(shared, ctx, config.body, env, modules);
            for &(name_span, _) in &config.bindings {
                env.vars.remove(ctx.text(name_span));
            }
            return result;
        }
    }

    // Single binding over a flat list
    if config.bindings.len() != 1 {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresTuples,
            shared.arena.span(config.iterable),
        ));
    }
    let iter_ty = type_of(shared, ctx, config.iterable, env, modules)?;
    if !iter_ty.is_list() {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresList(iter_ty),
            shared.arena.span(config.iterable),
        ));
    }
    let (name_span, declared_ty) = config.bindings[0];
    // is_list() check above guarantees elem_type() returns Some
    let elem_ty = iter_ty.elem_type().unwrap();
    if !elem_ty.is_compatible(declared_ty) {
        return Err(mismatch(declared_ty, elem_ty, name_span));
    }
    env.insert_var(ctx.text(name_span), declared_ty);
    let result = expect_rule(shared, ctx, config.body, env, modules);
    env.vars.remove(ctx.text(name_span));
    result
}

const fn mismatch(expected: Ty, got: Ty, span: Span) -> TypeError {
    TypeError::new(TypeErrorKind::TypeMismatch { expected, got }, span)
}

use super::TypeError;

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum TypeErrorKind {
    #[error("expected {expected}, got {got}")]
    TypeMismatch { expected: Ty, got: Ty },
    #[error("undefined function '{0}'")]
    UndefinedFunction(String),
    #[error("fn '{fn_name}': expected {expected} arguments, got {got}")]
    ArgCountMismatch {
        fn_name: String,
        expected: usize,
        got: usize,
    },
    #[error("no field '{field}' on {on_type}")]
    FieldNotFound { field: String, on_type: Ty },
    #[error("field access requires obj_t, got {0}")]
    FieldAccessOnNonObject(Ty),
    #[error("list elements have inconsistent types: {first} vs {got}")]
    ListElementTypeMismatch { first: Ty, got: Ty },
    #[error("object values must be rule_t, str_t, or int_t, got {0}")]
    InvalidObjectValue(Ty),
    #[error("list elements must be rule_t, str_t, int_t, or a list type")]
    InvalidListElement,
    #[error("empty list requires a type annotation")]
    EmptyContainerNeedsAnnotation,
    #[error("for-expression requires a list, got {0}")]
    ForRequiresList(Ty),
    #[error("for with multiple bindings requires a list of tuples")]
    ForRequiresTuples,
    #[error("for has {bindings} bindings but tuples have {tuple_elements} elements")]
    ForBindingCountMismatch {
        bindings: usize,
        tuple_elements: usize,
    },
    #[error("unresolved variable '{0}'")]
    UnresolvedVariable(String),
    #[error("append requires list arguments, got {0}")]
    AppendRequiresList(Ty),
    #[error("'::' requires module_t, got {0}")]
    QualifiedAccessOnInvalidType(Ty),
    #[error("unknown grammar config field '{0}'")]
    UnknownConfigField(String),
    #[error("alias target must be a name or string, got {0}")]
    InvalidAliasTarget(Ty),
    #[error("prec value must be int_t or str_t, got {0}")]
    PrecValueTypeMismatch(Ty),
    #[error("expected a rule name")]
    ExpectedRuleName,
    #[error("reserved config must be an object literal or inherited")]
    ExpectedReservedConfig,
    #[error("grammar_config() requires an inherited grammar, not an imported module")]
    GrammarConfigRequiresInherit,
    #[error("cannot infer type of this expression")]
    CannotInferType,
    #[error("cannot bind a for-loop expansion to a variable: for-loops are expanded inline")]
    NonBindableType(Ty),
    #[error("imported module has no member '{0}'")]
    ImportMemberNotFound(String),
    #[error("imported module has no function '{0}'")]
    ImportFunctionNotFound(String),
    #[error("'::' call requires module_t, got {0}")]
    QualifiedCallOnNonModule(Ty),
    #[error("'{0}' is a function, not a value; call it with {0}(...)")]
    FunctionUsedAsValue(String),
    #[error("duplicate declaration '{0}'")]
    DuplicateDeclaration(String),
    #[error("unknown identifier '{0}'")]
    UnknownIdentifier(String),
    #[error("'{0}' shadows an existing declaration")]
    ShadowedBinding(String),
    #[error("externals must be a list literal, append(), or variable reference")]
    InvalidExternalsExpression,
}
