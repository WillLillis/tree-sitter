//! Name resolution and type checking for the native grammar DSL.
//!
//! Entry point is [`resolve_and_check`]: collects declarations, resolves
//! identifiers, then type-checks the resolved AST.

use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use super::ast::{
    Ast, AstContext, ChildRange, ForId, Node, NodeArena, NodeId, Param, PrecKind, Span,
};
use super::scope_stack::ScopeStack;

/// Function pointer type for element/inner checkers passed to `expect_list` and
/// `expect_list_list`. Checks a single node against the type environment.
type CheckFn<'ast> =
    fn(&'ast Ast, NodeId, &mut TypeEnv<'ast>, &[Option<TypeEnv<'ast>>]) -> Result<(), TypeError>;
use super::{Note, NoteMessage};
use crate::grammars::InputGrammar;

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
    /// Result of `inherit(...)` or `import(...)`: a loaded module namespace.
    /// Members accessed via `::`. The `u8` indexes into the module list.
    Module(u8),
    /// User-facing module type annotation (`module_t`). Matches any
    /// concrete `Module(idx)` via `is_compatible`.
    AnyModule,
    /// Result of `grammar_config(module)`: the grammar config of a module.
    /// Field access (`.extras`, `.word`, etc.) returns per-field types.
    GrammarConfig,
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
        matches!(
            self,
            Self::ListRule
                | Self::ListStr
                | Self::ListInt
                | Self::ListListRule
                | Self::ListListStr
                | Self::ListListInt
        )
    }

    /// Check if `self` is assignable to `expected`.
    fn is_compatible(self, expected: Self) -> bool {
        self == expected
            || (self == Self::Str && expected == Self::Rule)
            || (self == Self::ListStr && expected == Self::ListRule)
            || (self == Self::ListListStr && expected == Self::ListListRule)
            || (matches!(self, Self::Module(_)) && expected == Self::AnyModule)
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
            Self::Module(_) | Self::AnyModule => f.write_str("module_t"),
            Self::GrammarConfig => f.write_str("grammar_config_t"),
            Self::Spread => f.write_str("spread_t"),
            Self::Object(inner) => write!(f, "obj_t<{inner}>"),
        }
    }
}

pub struct FnSig {
    pub params: Vec<Ty>,
    pub return_ty: Ty,
}

pub struct TypeEnv<'src> {
    pub vars: ScopeStack<'src, Ty>,
    pub fns: FxHashMap<&'src str, FnSig>,
    /// Field names for Object variables, keyed by variable name.
    pub object_fields: FxHashMap<&'src str, Vec<&'src str>>,
}

impl<'src> TypeEnv<'src> {
    fn new() -> Self {
        Self {
            vars: ScopeStack::new(),
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
        self.vars.get(name)
    }
    #[inline]
    fn scoped(&mut self) -> ScopeGuard<'_, 'src> {
        self.vars.push();
        ScopeGuard(self)
    }
}

struct ScopeGuard<'a, 'src>(&'a mut TypeEnv<'src>);

impl Drop for ScopeGuard<'_, '_> {
    fn drop(&mut self) {
        self.0.vars.pop();
    }
}

impl<'src> std::ops::Deref for ScopeGuard<'_, 'src> {
    type Target = TypeEnv<'src>;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl std::ops::DerefMut for ScopeGuard<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

// -- Name resolution --

/// Zero-allocation scope chain (linked list on stack, borrows from configs).
struct Locals<'a> {
    scope: LocalScope<'a>,
    #[expect(clippy::use_self, reason = "no Self with generics")]
    parent: Option<&'a Locals<'a>>,
}

enum LocalScope<'a> {
    Empty,
    FnParams(&'a [Param]),
    ForBindings(&'a [(Span, Ty)]),
}

impl Locals<'_> {
    const EMPTY: Locals<'static> = Locals {
        scope: LocalScope::Empty,
        parent: None,
    };

    /// Iterative walk + manual loops (instead of recursion + `.any()` closures)
    /// to reduce register pressure in the inner loops.
    fn contains(&self, ctx: &AstContext, name: &str) -> bool {
        let source = &ctx.source;
        let mut current = self;
        loop {
            match &current.scope {
                LocalScope::Empty => {}
                LocalScope::FnParams(params) => {
                    for p in *params {
                        if p.name.resolve(source) == name {
                            return true;
                        }
                    }
                }
                LocalScope::ForBindings(bindings) => {
                    for (span, _) in *bindings {
                        if span.resolve(source) == name {
                            return true;
                        }
                    }
                }
            }
            match current.parent {
                Some(p) => current = p,
                None => return false,
            }
        }
    }
}

#[derive(Clone, Copy)]
enum Decl {
    Rule,
    Var,
    Fn,
}

/// Run name resolution and type checking in a single function.
///
/// **Phase 1** (resolve): collects declarations, registers inherited names,
/// and mutates `Node::Ident` nodes into `Node::RuleRef` / `Node::VarRef`.
///
/// **Phase 2** (typecheck): calls the existing `check()` function on the
/// now-resolved AST.
///
/// # Errors
///
/// Returns [`TypeError`] for duplicate declarations, unknown identifiers,
/// or any type error.
pub fn resolve_and_check<'ast>(
    ast: &'ast mut Ast,
    modules: &[Option<TypeEnv<'ast>>],
    base: Option<(&'ast InputGrammar, Span)>,
    grammar_path: &Path,
) -> Result<TypeEnv<'ast>, TypeError> {
    // -- Phase 1: Name resolution (mutates arena) --
    // Split borrow: collect_decls borrows ast.context (for &str slices in decls)
    // but not ast.arena, so we can later mutably borrow ast.arena for resolve_item.
    let mut decls = collect_decls(&ast.arena, &ast.ctx, &ast.root_items, grammar_path)?;

    // Register inherited rule names so they resolve to RuleRef.
    // Use the inherit statement's span so "first defined here" points there.
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            if !decls.contains_key(var.name.as_str()) {
                decls.insert(&var.name, (Decl::Rule, inherit_span));
            }
        }
    }

    // Split borrow: decls holds &str from ast.context.source (immutable),
    // while resolve_item_tc mutates ast.arena. Access context and arena separately.
    let n_items = ast.root_items.len();
    for i in 0..n_items {
        let item_id = ast.root_items[i];
        // Register let bindings in order so forward references are caught
        if let Node::Let { name, .. } = ast.arena.get(item_id) {
            let name_text = ast.ctx.text(ast.arena.span(*name));
            let span = ast.arena.span(item_id);
            insert_decl(
                &mut decls,
                name_text,
                Decl::Var,
                span,
                grammar_path,
                &ast.ctx.source,
            )?;
        }
        resolve_item_tc(&mut ast.arena, &ast.ctx, &decls, item_id)?;
    }

    // -- Phase 2: Type checking (reads ast immutably) --
    check(ast, modules)
}

/// Intermediate resolve environment used during phase 1. Maps declaration
/// names to their kind (for Ident -> RuleRef/VarRef mutation) and span
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
    arena: &NodeArena,
    ctx: &'a AstContext,
    root_items: &[NodeId],
    grammar_path: &Path,
) -> Result<Decls<'a>, TypeError> {
    let mut decls = Decls::default();
    let source = &ctx.source;

    // Register rules and fns upfront (forward references allowed).
    // Let bindings are registered during pass 2 in item order (no forward references).
    for &item_id in root_items {
        let span = arena.span(item_id);
        match arena.get(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                insert_decl(
                    &mut decls,
                    ctx.text(*name),
                    Decl::Rule,
                    span,
                    grammar_path,
                    source,
                )?;
            }
            Node::Fn(fn_idx) => {
                let config = ctx.get_fn(*fn_idx);
                insert_decl(
                    &mut decls,
                    ctx.text(config.name),
                    Decl::Fn,
                    span,
                    grammar_path,
                    source,
                )?;
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
        collect_external_names_tc(
            arena,
            ctx,
            ext_id,
            &mut decls,
            root_items,
            grammar_path,
            source,
        )?;
    }

    Ok(decls)
}

/// Recursively walk an externals expression to find bare identifiers that
/// need pre-registration as external token names. Follows all expression
/// forms: lists, appends, variable references (via let binding values),
/// and any other compound expression.
///
/// Cycles are impossible: let bindings cannot forward-reference each other,
/// so circular references are caught as `UnknownIdentifier` during resolution.
fn collect_external_names_tc<'a>(
    arena: &NodeArena,
    ctx: &'a AstContext,
    id: NodeId,
    decls: &mut Decls<'a>,
    root_items: &[NodeId],
    grammar_path: &Path,
    source: &'a str,
) -> Result<(), TypeError> {
    match arena.get(id) {
        // Bare identifier: either a let binding (follow it) or an external token name.
        Node::Ident => {
            let name = ctx.text(arena.span(id));
            if let Some(value_id) = find_let_value(arena, ctx, root_items, name) {
                // Self-referential let (e.g. `let C = C`) - skip without registering.
                // Phase 2 will catch this as UnknownIdentifier.
                if matches!(arena.get(value_id), Node::Ident)
                    && ctx.text(arena.span(value_id)) == name
                {
                    return Ok(());
                }
                // It's a let binding - recurse into its value.
                collect_external_names_tc(
                    arena,
                    ctx,
                    value_id,
                    decls,
                    root_items,
                    grammar_path,
                    source,
                )?;
            } else if !decls.contains_key(name) {
                // Unknown bare identifier - register as external token.
                insert_decl(
                    decls,
                    name,
                    Decl::Rule,
                    arena.span(id),
                    grammar_path,
                    source,
                )?;
            }
        }
        // List: scan children for identifiers.
        Node::List(range) | Node::Seq(range) | Node::Choice(range) | Node::Tuple(range) => {
            for &child in ctx.child_slice(*range) {
                collect_external_names_tc(
                    arena,
                    ctx,
                    child,
                    decls,
                    root_items,
                    grammar_path,
                    source,
                )?;
            }
        }
        // Append: recurse into both arms.
        Node::Append { left, right } => {
            collect_external_names_tc(arena, ctx, *left, decls, root_items, grammar_path, source)?;
            collect_external_names_tc(arena, ctx, *right, decls, root_items, grammar_path, source)?;
        }
        // Literals don't introduce names.
        // grammar_config(base).externals - inherited values already registered.
        Node::StringLit
        | Node::RawStringLit { .. }
        | Node::IntLit(_)
        | Node::Blank
        | Node::GrammarConfig(_)
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
    ctx: &AstContext,
    root_items: &[NodeId],
    name: &str,
) -> Option<NodeId> {
    root_items.iter().find_map(|&item_id| {
        if let Node::Let { name: n, value, .. } = arena.get(item_id)
            && ctx.text(arena.span(*n)) == name
        {
            return Some(*value);
        }
        None
    })
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item_tc(
    arena: &mut NodeArena,
    ctx: &AstContext,
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
                resolve_expr_tc(arena, ctx, decls, id, &Locals::EMPTY)?;
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::OverrideRule { body, .. } => {
            let body = *body;
            resolve_expr_tc(arena, ctx, decls, body, &Locals::EMPTY)
        }
        Node::Let { value, .. } => {
            let value = *value;
            resolve_expr_tc(arena, ctx, decls, value, &Locals::EMPTY)
        }
        Node::Fn(fn_idx) => {
            let fn_config = ctx.get_fn(*fn_idx);
            let body = fn_config.body;
            let locals = Locals {
                scope: LocalScope::FnParams(&fn_config.params),
                parent: None,
            };
            resolve_expr_tc(arena, ctx, decls, body, &locals)
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
fn resolve_expr_tc(
    arena: &mut NodeArena,
    ctx: &AstContext,
    decls: &Decls,
    id: NodeId,
    locals: &Locals<'_>,
) -> Result<(), TypeError> {
    // Handle Ident mutation first
    if matches!(arena.get(id), Node::Ident) {
        let span = arena.span(id);
        let name = ctx.text(span);
        if locals.contains(ctx, name) {
            arena.set(id, Node::VarRef);
        } else if let Some(&(decl, _)) = decls.get(name) {
            match decl {
                Decl::Var => arena.set(id, Node::VarRef),
                Decl::Rule | Decl::Fn => arena.set(id, Node::RuleRef),
            }
        } else {
            return Err(TypeError::new(
                TypeErrorKind::UnknownIdentifier(name.to_string()),
                span,
            ));
        }
        return Ok(());
    }

    // Alias target: bare identifiers become RuleRef (named alias) unless
    // they refer to a local variable (VarRef for fn params / for bindings).
    // Targets don't require the name to be declared - they're just node names,
    // same as grammar.js `alias($.x, $.undeclared_name)`.
    if let Node::Alias { content, target } = arena.get(id) {
        let (content, target) = (*content, *target);
        resolve_expr_tc(arena, ctx, decls, content, locals)?;
        if matches!(*arena.get(target), Node::Ident) {
            let name = ctx.text(arena.span(target));
            if locals.contains(ctx, name) {
                arena.set(target, Node::VarRef);
            } else {
                arena.set(target, Node::RuleRef);
            }
        } else {
            resolve_expr_tc(arena, ctx, decls, target, locals)?;
        }
        return Ok(());
    }

    // For expressions need extended locals
    if let Node::For(for_idx) = arena.get(id) {
        let config = ctx.get_for(*for_idx);
        let iterable = config.iterable;
        let body = config.body;
        let bindings = &config.bindings;
        resolve_expr_tc(arena, ctx, decls, iterable, locals)?;
        let inner = Locals {
            scope: LocalScope::ForBindings(bindings),
            parent: Some(locals),
        };
        return resolve_expr_tc(arena, ctx, decls, body, &inner);
    }

    // All remaining cases
    resolve_children_tc(arena, ctx, decls, id, locals)
}

/// Resolve identifiers in children of a node (mechanical traversal).
fn resolve_children_tc(
    arena: &mut NodeArena,
    ctx: &AstContext,
    decls: &Decls,
    id: NodeId,
    locals: &Locals<'_>,
) -> Result<(), TypeError> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = arena.get(id).child_range() {
        for child in ctx.child_slice(range) {
            resolve_expr_tc(arena, ctx, decls, *child, locals)?;
        }
        return Ok(());
    }

    match arena.get(id) {
        Node::Call { name, args } => {
            let (name, args) = (*name, *args);
            resolve_expr_tc(arena, ctx, decls, name, locals)?;
            for arg in ctx.child_slice(args) {
                resolve_expr_tc(arena, ctx, decls, *arg, locals)?;
            }
            Ok(())
        }
        Node::Object(range) => {
            for field in ctx.get_object(*range) {
                resolve_expr_tc(arena, ctx, decls, field.1, locals)?;
            }
            Ok(())
        }
        Node::Repeat { inner, .. }
        | Node::Token { inner, .. }
        | Node::Neg(inner)
        | Node::GrammarConfig(inner) => {
            let inner = *inner;
            resolve_expr_tc(arena, ctx, decls, inner, locals)
        }
        Node::Field { content, .. } | Node::Reserved { content, .. } => {
            let content = *content;
            resolve_expr_tc(arena, ctx, decls, content, locals)
        }
        Node::Append { left, right } => {
            let (left, right) = (*left, *right);
            resolve_expr_tc(arena, ctx, decls, left, locals)?;
            resolve_expr_tc(arena, ctx, decls, right, locals)
        }
        Node::Prec { value, content, .. } => {
            let (value, content) = (*value, *content);
            resolve_expr_tc(arena, ctx, decls, value, locals)?;
            resolve_expr_tc(arena, ctx, decls, content, locals)
        }
        Node::DynRegex { pattern, flags } => {
            let (pattern, flags) = (*pattern, *flags);
            resolve_expr_tc(arena, ctx, decls, pattern, locals)?;
            if let Some(flags) = flags {
                resolve_expr_tc(arena, ctx, decls, flags, locals)?;
            }
            Ok(())
        }
        Node::FieldAccess { obj, .. } | Node::QualifiedAccess { obj, .. } => {
            let obj = *obj;
            resolve_expr_tc(arena, ctx, decls, obj, locals)
        }
        Node::QualifiedCall(range) => {
            let (obj, _name, args) = ctx.get_qualified_call(*range);
            resolve_expr_tc(arena, ctx, decls, obj, locals)?;
            // name stays as Ident - it's a member of the import namespace
            for &arg in args {
                resolve_expr_tc(arena, ctx, decls, arg, locals)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

/// Run typechecking and return the type environment.
///
/// `modules` holds imported modules' type environments (indexed by
/// `Ty::Module(idx)`), populated by the import pre-pass in `mod.rs`.
pub fn check<'ast>(
    ast: &'ast Ast,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<TypeEnv<'ast>, TypeError> {
    let mut env = TypeEnv::new();
    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                env.insert_var(ast.ctx.text(*name), Ty::Rule);
            }
            Node::Fn(fn_idx) => {
                let config = ast.ctx.get_fn(*fn_idx);
                let params: Vec<Ty> = config.params.iter().map(|p| p.ty).collect();
                let return_ty = config.return_ty;
                env.fns
                    .insert(ast.ctx.text(config.name), FnSig { params, return_ty });
            }
            _ => {}
        }
    }
    for &item_id in &ast.root_items {
        check_item(ast, item_id, &mut env, modules)?;
    }
    Ok(env)
}

fn check_item<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    match ast.node(id) {
        Node::Grammar => {
            let config = ast.ctx.grammar_config.as_ref().unwrap();
            for id in [config.extras, config.externals].into_iter().flatten() {
                expect_rule_list(ast, id, env, modules)?;
            }
            for id in [config.inline, config.supertypes].into_iter().flatten() {
                expect_name_list(ast, id, env, modules)?;
            }
            if let Some(id) = config.conflicts {
                expect_list_list(ast, id, env, modules, expect_name_list)?;
            }
            if let Some(id) = config.precedences {
                expect_list_list(ast, id, env, modules, expect_precedence_group)?;
            }
            if let Some(id) = config.word {
                expect_name_ref(ast, id, env, modules)?;
            }
            if let Some(id) = config.reserved {
                expect_reserved(ast, id, env, modules)?;
            }
            Ok(())
        }
        Node::Let { name, ty, value } => {
            let var_name = ast.node_text(*name);
            let declared = *ty;
            // Empty list: type comes from annotation, not inference
            let is_empty_list =
                matches!(ast.node(*value), Node::List(r) if ast.ctx.child_slice(*r).is_empty());
            let resolved_ty = if is_empty_list {
                match declared {
                    Some(ty) if ty.is_list() => ty,
                    _ => {
                        return Err(TypeError::new(
                            TypeErrorKind::EmptyListNeedsAnnotation,
                            ast.span(*value),
                        ));
                    }
                }
            } else {
                let inferred = type_of(ast, *value, env, modules)?;
                // `for (...) { ... }` yields non-storable types. Reject it
                // at the let binding with a specific message; without this the
                // for-expansion case would panic during lowering.
                if inferred == Ty::Spread {
                    return Err(TypeError::new(
                        TypeErrorKind::CannotBindSpread,
                        ast.span(*value),
                    ));
                }
                if let Some(d) = declared
                    && !inferred.is_compatible(d)
                {
                    return Err(mismatch(d, inferred, ast.span(*value)));
                }
                inferred
            };
            // If the value is an object literal, register its field names
            if let Node::Object(range) = ast.node(*value) {
                let fields: Vec<&str> = ast
                    .ctx
                    .get_object(*range)
                    .iter()
                    .map(|(span, _)| ast.ctx.text(*span))
                    .collect();
                env.insert_object(var_name, resolved_ty, fields);
            } else {
                env.insert_var(var_name, resolved_ty);
            }
            Ok(())
        }
        Node::Fn(fn_idx) => {
            let config = ast.ctx.get_fn(*fn_idx);
            let fn_name = ast.ctx.text(config.name);
            let sig = &env.fns[fn_name];
            let return_ty = sig.return_ty;
            let n_params = sig.params.len();
            let mut scope = env.scoped();
            for i in 0..n_params {
                let ty = scope.fns[fn_name].params[i];
                scope.insert_var(ast.ctx.text(config.params[i].name), ty);
            }
            let body_ty = type_of(ast, config.body, &mut scope, modules)?;
            if !body_ty.is_compatible(return_ty) {
                return Err(mismatch(return_ty, body_ty, ast.span(config.body)));
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::OverrideRule { body, .. } => {
            expect_rule(ast, *body, env, modules)
        }
        // Unreachable: parse_item() only produces Grammar, Let, Fn, Rule,
        // OverrideRule, and Print nodes at root level.
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

// -- Config field validators --

/// Check a list config field. For literal lists, checks each element with `check_elem`.
/// For non-literal expressions, checks the overall type is a list type.
fn expect_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
    check_elem: CheckFn<'ast>,
    accept_list_str: bool,
) -> Result<(), TypeError> {
    if let Node::List(range) = ast.node(id) {
        for &child in ast.ctx.child_slice(*range) {
            check_elem(ast, child, env, modules)?;
        }
        return Ok(());
    }
    let ty = type_of(ast, id, env, modules)?;
    if ty == Ty::ListRule || (accept_list_str && ty == Ty::ListStr) {
        return Ok(());
    }
    Err(mismatch(Ty::ListRule, ty, ast.span(id)))
}

fn expect_rule_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    expect_list(ast, id, env, modules, expect_rule, true)
}

fn expect_name_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    expect_list(ast, id, env, modules, expect_name_ref, false)
}

fn expect_name_ref<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    match ast.node(id) {
        Node::RuleRef => Ok(()),
        Node::VarRef | Node::FieldAccess { .. } => {
            let ty = type_of(ast, id, env, modules)?;
            if ty != Ty::Rule {
                return Err(mismatch(Ty::Rule, ty, ast.span(id)));
            }
            Ok(())
        }
        _ => Err(TypeError::new(
            TypeErrorKind::ExpectedRuleName,
            ast.span(id),
        )),
    }
}

/// Check a `list_list_rule_t` config field. For literal lists-of-lists, validates
/// each inner list element-wise with `check_inner`. For non-literal expressions,
/// checks the overall type is `list_list_rule_t` (or a subtype thereof).
fn expect_list_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
    check_inner: CheckFn<'ast>,
) -> Result<(), TypeError> {
    if let Node::List(range) = ast.node(id) {
        for &child in ast.ctx.child_slice(*range) {
            check_inner(ast, child, env, modules)?;
        }
        return Ok(());
    }
    let ty = type_of(ast, id, env, modules)?;
    if ty == Ty::ListListRule || ty == Ty::ListListStr {
        return Ok(());
    }
    Err(mismatch(Ty::ListListRule, ty, ast.span(id)))
}

/// Inner element for `precedences`: a name ref or a string literal.
fn expect_name_or_str<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    if matches!(ast.node(id), Node::StringLit | Node::RawStringLit { .. }) {
        return Ok(());
    }
    expect_name_ref(ast, id, env, modules)
}

/// Check a `precedences` inner list: each element must be a name-ref or string literal.
fn expect_precedence_group<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    expect_list(ast, id, env, modules, expect_name_or_str, true)
}

fn expect_reserved<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    // Object literal: each field value must be a rule list
    if let Node::Object(range) = ast.node(id) {
        for &(_, val_id) in ast.ctx.get_object(*range) {
            expect_rule_list(ast, val_id, env, modules)?;
        }
        return Ok(());
    }
    // Non-literal: must be inherited (base.reserved -> Object(ListRule))
    let ty = type_of(ast, id, env, modules)?;
    if ty == Ty::Object(InnerTy::ListRule) {
        return Ok(());
    }
    Err(TypeError::new(
        TypeErrorKind::ExpectedReservedConfig,
        ast.span(id),
    ))
}

fn expect_rule<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    let ty = type_of(ast, id, env, modules)?;
    if !ty.is_rule_like() {
        return Err(mismatch(Ty::Rule, ty, ast.span(id)));
    }
    Ok(())
}

// -- Type inference --

fn type_of<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let span = ast.span(id);
    match ast.node(id) {
        Node::IntLit(_) => Ok(Ty::Int),
        Node::StringLit | Node::RawStringLit { .. } => Ok(Ty::Str),
        Node::RuleRef | Node::Blank => Ok(Ty::Rule),
        Node::Inherit { module, .. } | Node::Import { module, .. } => {
            let idx = module.expect("module index not set by loading pre-pass");
            Ok(Ty::Module(idx))
        }
        Node::GrammarConfig(inner) => {
            let inner_ty = type_of(ast, *inner, env, modules)?;
            if !matches!(inner_ty, Ty::Module(_)) {
                return Err(TypeError::new(
                    TypeErrorKind::QualifiedAccessOnInvalidType(inner_ty),
                    ast.span(*inner),
                ));
            }
            Ok(Ty::GrammarConfig)
        }
        Node::Ident => unreachable!(),
        Node::VarRef => {
            let name = ast.ctx.text(span);
            env.get_var(name).ok_or_else(|| {
                TypeError::new(TypeErrorKind::UnresolvedVariable(name.to_string()), span)
            })
        }
        Node::Neg(inner) => {
            let ty = type_of(ast, *inner, env, modules)?;
            if ty != Ty::Int {
                return Err(mismatch(Ty::Int, ty, span));
            }
            Ok(Ty::Int)
        }
        Node::Append { left, right } => type_of_append(ast, *left, *right, span, env, modules),
        Node::FieldAccess { obj, field } => type_of_field_access(ast, *obj, *field, env, modules),
        Node::QualifiedAccess { obj, member } => {
            let obj_ty = type_of(ast, *obj, env, modules)?;
            match obj_ty {
                Ty::Module(idx) => type_of_import_access(ast, idx, *member, modules),
                _ => Err(TypeError::new(
                    TypeErrorKind::QualifiedAccessOnInvalidType(obj_ty),
                    ast.span(*obj),
                )),
            }
        }
        Node::Object(range) => type_of_object(ast, *range, span, env, modules),
        Node::List(range) => type_of_list(ast, *range, span, env, modules),
        Node::Tuple(_) => {
            // Tuples are only valid as elements of for-loop lists.
            // They're checked structurally in check_for_expr.
            // If we reach here, it's a tuple in an invalid context.
            Err(TypeError::new(TypeErrorKind::CannotInferType, span))
        }
        Node::Concat(range) => {
            for &part in ast.ctx.child_slice(*range) {
                let ty = type_of(ast, part, env, modules)?;
                if ty != Ty::Str {
                    return Err(mismatch(Ty::Str, ty, ast.span(part)));
                }
            }
            Ok(Ty::Str)
        }
        Node::DynRegex { pattern, flags } => {
            let pt = type_of(ast, *pattern, env, modules)?;
            if pt != Ty::Str {
                return Err(mismatch(Ty::Str, pt, ast.span(*pattern)));
            }
            if let Some(fid) = flags {
                let ft = type_of(ast, *fid, env, modules)?;
                if ft != Ty::Str {
                    return Err(mismatch(Ty::Str, ft, ast.span(*fid)));
                }
            }
            Ok(Ty::Rule)
        }
        Node::Seq(range) | Node::Choice(range) => {
            for &member in ast.ctx.child_slice(*range) {
                let ty = type_of(ast, member, env, modules)?;
                if ty != Ty::Spread && !ty.is_rule_like() {
                    return Err(mismatch(Ty::Rule, ty, ast.span(member)));
                }
            }
            Ok(Ty::Rule)
        }
        Node::Repeat { inner, .. }
        | Node::Token { inner, .. }
        | Node::Field { content: inner, .. }
        | Node::Reserved { content: inner, .. } => {
            expect_rule(ast, *inner, env, modules)?;
            Ok(Ty::Rule)
        }
        Node::Alias { content, target } => {
            expect_rule(ast, *content, env, modules)?;
            let target_ty = type_of(ast, *target, env, modules)?;
            let is_valid =
                matches!(ast.node(*target), Node::RuleRef | Node::VarRef) || target_ty == Ty::Str;
            if !is_valid {
                return Err(TypeError::new(
                    TypeErrorKind::InvalidAliasTarget(target_ty),
                    ast.span(*target),
                ));
            }
            Ok(Ty::Rule)
        }
        Node::Prec {
            kind,
            value,
            content,
        } => {
            let vt = type_of(ast, *value, env, modules)?;
            if *kind == PrecKind::Dynamic {
                if vt != Ty::Int {
                    return Err(mismatch(Ty::Int, vt, ast.span(*value)));
                }
            } else if vt != Ty::Int && vt != Ty::Str {
                return Err(mismatch(Ty::Int, vt, ast.span(*value)));
            }
            expect_rule(ast, *content, env, modules)?;
            Ok(Ty::Rule)
        }
        Node::For(for_idx) => {
            check_for_expr(ast, *for_idx, env, modules)?;
            Ok(Ty::Spread)
        }
        Node::Call { name, args } => type_of_call(ast, *name, *args, span, env, modules),
        Node::QualifiedCall(range) => type_of_qualified_call(ast, *range, span, env, modules),
        _ => Err(TypeError::new(TypeErrorKind::CannotInferType, span)),
    }
}

// -- type_of helpers (extracted from the main match) --

fn type_of_append<'ast>(
    ast: &'ast Ast,
    left: NodeId,
    right: NodeId,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let l_empty = matches!(ast.node(left), Node::List(r) if ast.ctx.child_slice(*r).is_empty());
    let r_empty = matches!(ast.node(right), Node::List(r) if ast.ctx.child_slice(*r).is_empty());
    let lt = if l_empty {
        None
    } else {
        Some(type_of(ast, left, env, modules)?)
    };
    let rt = if r_empty {
        None
    } else {
        Some(type_of(ast, right, env, modules)?)
    };
    match (lt, rt) {
        (Some(l), Some(r)) => {
            if !l.is_list() {
                return Err(TypeError::new(
                    TypeErrorKind::AppendRequiresList(l),
                    ast.span(left),
                ));
            }
            if !r.is_list() {
                return Err(mismatch(l, r, ast.span(right)));
            }
            widen_type(l, r).ok_or_else(|| mismatch(l, r, ast.span(right)))
        }
        (Some(t), None) | (None, Some(t)) => {
            if !t.is_list() {
                return Err(TypeError::new(TypeErrorKind::AppendRequiresList(t), span));
            }
            Ok(t)
        }
        (None, None) => Err(TypeError::new(
            TypeErrorKind::EmptyListNeedsAnnotation,
            span,
        )),
    }
}

fn type_of_field_access<'ast>(
    ast: &'ast Ast,
    obj: NodeId,
    field: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let obj_ty = type_of(ast, obj, env, modules)?;
    let field_name = ast.ctx.text(field);
    match obj_ty {
        Ty::Object(inner) => {
            let field_known = match ast.node(obj) {
                Node::VarRef => {
                    let var_name = ast.ctx.text(ast.span(obj));
                    env.get_object_fields(var_name)
                        .is_none_or(|fields| fields.contains(&field_name))
                }
                Node::Object(range) => {
                    let fields = ast.ctx.get_object(*range);
                    fields
                        .iter()
                        .any(|(key_span, _)| ast.ctx.text(*key_span) == field_name)
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
        Ty::GrammarConfig => match field_name {
            "extras" | "externals" | "inline" | "supertypes" => Ok(Ty::ListRule),
            "conflicts" | "precedences" => Ok(Ty::ListListRule),
            "word" => Ok(Ty::Rule),
            "reserved" => Ok(Ty::Object(InnerTy::ListRule)),
            _ => Err(TypeError::new(
                TypeErrorKind::UnknownConfigField(field_name.to_string()),
                field,
            )),
        },
        _ => Err(TypeError::new(
            TypeErrorKind::FieldAccessOnNonObject(obj_ty),
            ast.span(obj),
        )),
    }
}

fn type_of_import_access(
    ast: &Ast,
    idx: u8,
    member: Span,
    modules: &[Option<TypeEnv<'_>>],
) -> Result<Ty, TypeError> {
    let member_name = ast.ctx.text(member);
    // Safe: idx was assigned during loading and typecheck_modules populates
    // the table before any module that references it is checked.
    let import_env = modules[idx as usize].as_ref().unwrap();
    if let Some(ty) = import_env.vars.get(member_name) {
        return Ok(ty);
    }
    if import_env.fns.contains_key(member_name) {
        return Err(TypeError::new(
            TypeErrorKind::ImportFunctionUsedAsValue(member_name.to_string()),
            member,
        ));
    }
    Err(TypeError::new(
        TypeErrorKind::ImportMemberNotFound(member_name.to_string()),
        member,
    ))
}

fn type_of_qualified_call<'a>(
    ast: &'a Ast,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv<'a>,
    modules: &[Option<TypeEnv<'a>>],
) -> Result<Ty, TypeError> {
    let (obj, name, args) = ast.ctx.get_qualified_call(range);
    let obj_ty = type_of(ast, obj, env, modules)?;
    let Ty::Module(idx) = obj_ty else {
        return Err(TypeError::new(
            TypeErrorKind::QualifiedCallOnNonModule(obj_ty),
            ast.span(obj),
        ));
    };
    let fn_name = ast.node_text(name);
    let import_env = modules[idx as usize].as_ref().unwrap();
    let Some(sig) = import_env.fns.get(fn_name) else {
        return Err(TypeError::new(
            TypeErrorKind::ImportFunctionNotFound(fn_name.to_string()),
            ast.span(name),
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
        let arg_ty = type_of(ast, arg_id, env, modules)?;
        if !arg_ty.is_compatible(expected) {
            return Err(mismatch(expected, arg_ty, ast.span(arg_id)));
        }
    }
    Ok(return_ty)
}

fn type_of_object<'ast>(
    ast: &'ast Ast,
    range: super::ast::ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let fields = ast.ctx.get_object(range);
    if fields.is_empty() {
        return Err(TypeError::new(TypeErrorKind::CannotInferType, span));
    }
    let mut widest = type_of(ast, fields[0].1, env, modules)?;
    for &(_, val_id) in &fields[1..] {
        let ty = type_of(ast, val_id, env, modules)?;
        widest = widen_type(widest, ty).ok_or_else(|| mismatch(widest, ty, ast.span(val_id)))?;
    }
    let inner = InnerTy::try_from(widest).map_err(|()| {
        TypeError::new(
            TypeErrorKind::InvalidObjectValue(widest),
            ast.span(fields[0].1),
        )
    })?;
    Ok(Ty::Object(inner))
}

fn type_of_list<'ast>(
    ast: &'ast Ast,
    range: super::ast::ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let items = ast.ctx.child_slice(range);
    if items.is_empty() {
        return Err(TypeError::new(
            TypeErrorKind::EmptyListNeedsAnnotation,
            span,
        ));
    }
    let mut widest = type_of(ast, items[0], env, modules)?;
    for &item_id in &items[1..] {
        let ty = type_of(ast, item_id, env, modules)?;
        widest = widen_type(widest, ty).ok_or_else(|| {
            TypeError::new(
                TypeErrorKind::ListElementTypeMismatch {
                    first: widest,
                    got: ty,
                },
                ast.span(item_id),
            )
        })?;
    }
    match widest {
        Ty::Str => Ok(Ty::ListStr),
        _ if widest.is_rule_like() => Ok(Ty::ListRule),
        Ty::Int => Ok(Ty::ListInt),
        Ty::ListRule => Ok(Ty::ListListRule),
        Ty::ListStr => Ok(Ty::ListListStr),
        Ty::ListInt => Ok(Ty::ListListInt),
        _ => Err(TypeError::new(
            TypeErrorKind::InvalidListElement,
            ast.span(items[0]),
        )),
    }
}

fn type_of_call<'ast>(
    ast: &'ast Ast,
    name: NodeId,
    args: super::ast::ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<Ty, TypeError> {
    let fn_name = ast.node_text(name);
    let sig = env.fns.get(fn_name).ok_or_else(|| {
        TypeError::new(TypeErrorKind::UndefinedFunction(fn_name.to_string()), span)
    })?;
    let return_ty = sig.return_ty;
    let n_params = sig.params.len();
    let args = ast.ctx.child_slice(args);
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
        let arg_ty = type_of(ast, arg_id, env, modules)?;
        let param_ty = env.fns[fn_name].params[i];
        if !arg_ty.is_compatible(param_ty) {
            return Err(mismatch(param_ty, arg_ty, ast.span(arg_id)));
        }
    }
    Ok(return_ty)
}

// -- For-loop checking --

fn check_for_expr<'ast>(
    ast: &'ast Ast,
    for_idx: ForId,
    env: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    let config = ast.ctx.get_for(for_idx);

    // Check if iterable is a literal list of tuples (for destructuring)
    if let Node::List(range) = ast.node(config.iterable) {
        let items = ast.ctx.child_slice(*range);
        let has_tuples = items
            .first()
            .is_some_and(|&id| matches!(ast.node(id), Node::Tuple(_)));
        if has_tuples {
            let mut scope = env.scoped();
            for &item_id in items {
                check_for_tuple_element(ast, item_id, config, &mut scope, modules)?;
            }
            return expect_rule(ast, config.body, &mut scope, modules);
        }
    }

    // Single binding over a flat list
    if config.bindings.len() != 1 {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresTuples,
            ast.span(config.iterable),
        ));
    }
    let iter_ty = type_of(ast, config.iterable, env, modules)?;
    if !iter_ty.is_list() {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresList(iter_ty),
            ast.span(config.iterable),
        ));
    }
    let (name_span, declared_ty) = config.bindings[0];
    let elem_ty = match iter_ty {
        Ty::ListRule => Ty::Rule,
        Ty::ListStr => Ty::Str,
        Ty::ListInt => Ty::Int,
        Ty::ListListRule => Ty::ListRule,
        Ty::ListListStr => Ty::ListStr,
        Ty::ListListInt => Ty::ListInt,
        _ => unreachable!(),
    };
    if !elem_ty.is_compatible(declared_ty) {
        return Err(mismatch(declared_ty, elem_ty, name_span));
    }
    let mut scope = env.scoped();
    scope.insert_var(ast.ctx.text(name_span), declared_ty);
    expect_rule(ast, config.body, &mut scope, modules)
}

fn check_for_tuple_element<'ast>(
    ast: &'ast Ast,
    item_id: NodeId,
    config: &super::ast::ForConfig,
    scope: &mut TypeEnv<'ast>,
    modules: &[Option<TypeEnv<'ast>>],
) -> Result<(), TypeError> {
    let Node::Tuple(range) = ast.node(item_id) else {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresTuples,
            ast.span(item_id),
        ));
    };
    let elems = ast.ctx.child_slice(*range);
    if elems.len() != config.bindings.len() {
        return Err(TypeError::new(
            TypeErrorKind::ForBindingCountMismatch {
                bindings: config.bindings.len(),
                tuple_elements: elems.len(),
            },
            ast.span(item_id),
        ));
    }
    // Pop and re-push scope vars for each tuple (last iteration's bindings win)
    for (i, &(name_span, declared_ty)) in config.bindings.iter().enumerate() {
        let actual = type_of(ast, elems[i], scope, modules)?;
        if !actual.is_compatible(declared_ty) {
            return Err(mismatch(declared_ty, actual, name_span));
        }
        scope.insert_var(ast.ctx.text(name_span), declared_ty);
    }
    Ok(())
}

// -- Helpers --

const fn mismatch(expected: Ty, got: Ty, span: Span) -> TypeError {
    TypeError::new(TypeErrorKind::TypeMismatch { expected, got }, span)
}

// -- Error types --

#[derive(Debug, Serialize, Error)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
    pub note: Option<Box<Note>>,
}

impl TypeError {
    const fn new(kind: TypeErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            note: None,
        }
    }

    fn with_note(kind: TypeErrorKind, span: Span, note: Note) -> Self {
        Self {
            kind,
            span,
            note: Some(Box::new(note)),
        }
    }
}

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
    #[error("field access requires obj_t or grammar_config_t, got {0}")]
    FieldAccessOnNonObject(Ty),
    #[error("list elements have inconsistent types: {first} vs {got}")]
    ListElementTypeMismatch { first: Ty, got: Ty },
    #[error("object values must be rule_t, str_t, or int_t, got {0}")]
    InvalidObjectValue(Ty),
    #[error("list elements must be rule_t, str_t, int_t, or a list type")]
    InvalidListElement,
    #[error("empty list requires a type annotation")]
    EmptyListNeedsAnnotation,
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
    #[error("expected a rule name")]
    ExpectedRuleName,
    #[error("reserved config must be an object literal or inherited")]
    ExpectedReservedConfig,
    #[error("cannot infer type of this expression")]
    CannotInferType,
    #[error("cannot bind 'print' to a variable: 'print' does not produce a value")]
    CannotBindVoid,
    #[error("cannot bind a for-loop expansion to a variable: for-loops are expanded inline")]
    CannotBindSpread,
    #[error("cannot print a value of type {0}")]
    CannotPrintType(Ty),
    #[error("imported module has no member '{0}'")]
    ImportMemberNotFound(String),
    #[error("'{0}' is a function, not a value; call it with {0}()")]
    ImportFunctionUsedAsValue(String),
    #[error("imported module has no function '{0}'")]
    ImportFunctionNotFound(String),
    #[error("'::' call requires module_t, got {0}")]
    QualifiedCallOnNonModule(Ty),
    #[error("duplicate declaration '{0}'")]
    DuplicateDeclaration(String),
    #[error("unknown identifier '{0}'")]
    UnknownIdentifier(String),
    #[error("externals must be a list literal, append(), or variable reference")]
    InvalidExternalsExpression,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}
