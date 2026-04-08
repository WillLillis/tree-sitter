//! Name resolution pass for the native grammar DSL.
//!
//! This pass runs after parsing and before type checking. It operates in two
//! phases:
//!
//! ## Pass 1: Declaration collection ([`collect_names`])
//!
//! Scans all top-level items to build a map of declared names and their kinds
//! (rule, variable, or function). Also registers external tokens declared in
//! the `grammar { externals: [...] }` block. This pass reads the AST
//! immutably.
//!
//! ## Pass 2: Identifier resolution ([`resolve_item`] / [`resolve_expr`])
//!
//! Walks every expression in the AST and replaces `Node::Ident` nodes with
//! either:
//! - `Node::RuleRef` - the identifier refers to a grammar rule or function
//! - `Node::VarRef` - the identifier refers to a `let` binding, function
//!   parameter, or `for` loop variable
//!
//! Local variables (function parameters and for-loop bindings) shadow
//! top-level names. The scope chain is a zero-allocation linked list
//! ([`Locals`]) that borrows directly from the function/for config data.
//!
//! ## Borrow splitting
//!
//! The resolve pass needs to mutate `Ast::nodes` (rewriting `Ident` to
//! `RuleRef`/`VarRef`) while reading `AstContext` (for source text,
//! function configs, etc.). This is why `resolve_expr` takes
//! `(&mut [Node], &AstContext)` rather than `&mut Ast`.

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use super::ast::{Ast, AstContext, Node, NodeId, Param, Span};

/// Zero-allocation scope chain for local variable names.
///
/// Each level borrows directly from `FnConfig.params` or
/// `ForConfig.bindings` without copying. The `parent` pointer forms a linked
/// list on the stack, so no heap allocation is needed for nested scopes.
struct Locals<'a> {
    scope: LocalScope<'a>,
    #[expect(clippy::use_self, reason = "no Self with generics")]
    parent: Option<&'a Locals<'a>>,
}

/// The payload of a single scope level.
///
/// Using an enum avoids a `dyn` trait object or copying data into a uniform
/// format. `FnParams` borrows the parameter list from `FnConfig`, while
/// `ForBindings` borrows the binding list from `ForConfig`.
enum LocalScope<'a> {
    /// Sentinel for the root scope (no local variables).
    Empty,
    /// Function parameters - borrows `&FnConfig.params`.
    FnParams(&'a [Param]),
    /// For-loop bindings - borrows `&ForConfig.bindings`.
    ForBindings(&'a [(Span, NodeId)]),
}

impl Locals<'_> {
    /// A static empty scope for use as the initial scope at top level.
    const EMPTY: Locals<'static> = Locals {
        scope: LocalScope::Empty,
        parent: None,
    };

    /// Check whether `name` is bound in this scope or any parent.
    fn contains(&self, ctx: &AstContext<'_>, name: &str) -> bool {
        let found = match &self.scope {
            LocalScope::Empty => false,
            LocalScope::FnParams(params) => params.iter().any(|p| ctx.node_text(p.name) == name),
            LocalScope::ForBindings(bindings) => {
                bindings.iter().any(|(span, _)| ctx.text(*span) == name)
            }
        };
        found || self.parent.is_some_and(|p| p.contains(ctx, name))
    }
}

/// Run name resolution on the AST.
///
/// Collects all top-level declarations, then resolves every identifier in
/// every item body. After this pass, no `Node::Ident` nodes remain in the
/// AST - they have all been replaced with `Node::RuleRef` or `Node::VarRef`.
///
/// # Errors
///
/// Returns [`ResolveError`] for duplicate declarations or unknown identifiers.
pub fn resolve(ast: &mut Ast<'_>, base_rule_names: &[String]) -> Result<(), ResolveError> {
    let mut names = collect_names(ast)?;
    // Register inherited rule names so they resolve to RuleRef
    for name in base_rule_names {
        // Don't error on duplicates - derived rules intentionally shadow base rules
        if names.get(name).is_none() {
            names.decls.insert(name.clone(), Decl::Rule);
        }
    }
    let n_items = ast.root_items.len();
    for i in 0..n_items {
        let item_id = ast.root_items[i];
        // Register let bindings in order so forward references are caught
        if let Node::Let { name, .. } = ast.node(item_id) {
            let name_text = ast.node_text(*name);
            let span = ast.span(item_id);
            names.insert(name_text, Decl::Var, span)?;
        }
        resolve_item(ast, &names, item_id)?;
    }
    Ok(())
}

/// The kind of a top-level declaration, used to decide whether an identifier
/// resolves to `RuleRef` or `VarRef`.
#[derive(Clone, Copy)]
enum Decl {
    Rule,
    Var,
    Fn,
}

/// Map of top-level declaration names to their kinds.
struct Names {
    decls: FxHashMap<String, Decl>,
}

impl Names {
    fn new() -> Self {
        Self {
            decls: FxHashMap::default(),
        }
    }

    fn insert(&mut self, name: &str, kind: Decl, span: Span) -> Result<(), ResolveError> {
        if self.decls.contains_key(name) {
            return Err(ResolveError {
                kind: ResolveErrorKind::DuplicateDeclaration(name.to_string()),
                span,
            });
        }
        self.decls.insert(name.to_string(), kind);
        Ok(())
    }

    fn get(&self, name: &str) -> Option<Decl> {
        self.decls.get(name).copied()
    }
}

/// Pass 1: scan top-level items and register all declared names.
///
/// Rules, let-bindings, functions, and external tokens in the grammar block
/// all occupy the same namespace. Duplicate names are rejected.
fn collect_names(ast: &Ast<'_>) -> Result<Names, ResolveError> {
    let mut names = Names::new();

    // Register rules and fns upfront (forward references allowed).
    // Let bindings are registered during pass 2 in item order (no forward references).
    for i in 0..ast.root_items.len() {
        let item_id = ast.root_items[i];
        let span = ast.span(item_id);
        match ast.node(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                names.insert(ast.node_text(*name), Decl::Rule, span)?;
            }
            Node::Fn(fn_idx) => {
                let config = ast.get_fn(*fn_idx);
                names.insert(ast.node_text(config.name), Decl::Fn, span)?;
            }
            _ => {}
        }
    }

    // Register externals that aren't already declared as rules
    if let Some(config) = &ast.context.grammar_config {
        for j in 0..config.externals.len() {
            let ext_id = config.externals[j];
            if let Node::Ident = ast.node(ext_id) {
                let name = ast.text(ast.span(ext_id));
                if names.get(name).is_none() {
                    names.insert(name, Decl::Rule, ast.span(ext_id))?;
                }
            }
        }
    }

    Ok(names)
}

/// Pass 2: resolve all identifiers within a single top-level item.
///
/// Splits the borrow on `ast` into `&mut ast.nodes` and `&ast.context` so
/// that nodes can be mutated while reading source text and configs.
fn resolve_item(ast: &mut Ast<'_>, names: &Names, item_id: NodeId) -> Result<(), ResolveError> {
    match ast.node(item_id) {
        Node::Grammar => {
            let ctx = &ast.context;
            // SAFETY: Earlier pass has already verified this is `Some`
            let grammar_config = ctx.grammar_config.as_ref().unwrap();
            let nodes = &mut ast.nodes;
            if let Some(inherits) = grammar_config.inherits {
                resolve_expr(nodes, ctx, names, inherits, &Locals::EMPTY)?;
            }
            for &id in &grammar_config.extras {
                resolve_expr(nodes, ctx, names, id, &Locals::EMPTY)?;
            }
            for &id in &grammar_config.externals {
                resolve_expr(nodes, ctx, names, id, &Locals::EMPTY)?;
            }
            for rws in &grammar_config.reserved {
                for &id in &rws.words {
                    resolve_expr(nodes, ctx, names, id, &Locals::EMPTY)?;
                }
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::OverrideRule { body, .. } => {
            let body = *body;
            resolve_expr(&mut ast.nodes, &ast.context, names, body, &Locals::EMPTY)
        }
        Node::Let { value, .. } => {
            let value = *value;
            resolve_expr(&mut ast.nodes, &ast.context, names, value, &Locals::EMPTY)
        }
        Node::Fn(fn_idx) => {
            let fn_config = ast.context.get_fn(*fn_idx);
            let body = fn_config.body;
            let locals = Locals {
                scope: LocalScope::FnParams(&fn_config.params),
                parent: None,
            };
            resolve_expr(&mut ast.nodes, &ast.context, names, body, &locals)
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
///
/// Handles three special cases before falling through to child traversal:
/// 1. `Node::Ident` - looks up in locals first, then top-level names, and
///    rewrites to `RuleRef` or `VarRef`.
/// 2. `Node::Alias` - the target identifier always becomes `RuleRef`.
/// 3. `Node::For` - extends the scope chain with the for-loop bindings.
fn resolve_expr(
    nodes: &mut [Node],
    ctx: &AstContext<'_>,
    names: &Names,
    id: NodeId,
    locals: &Locals<'_>,
) -> Result<(), ResolveError> {
    // Handle Ident mutation first
    if let Node::Ident = &nodes[id.index()] {
        let span = ctx.span(id);
        let name = ctx.text(span);
        if locals.contains(ctx, name) {
            nodes[id.index()] = Node::VarRef;
        } else if let Some(decl) = names.get(name) {
            match decl {
                Decl::Var => nodes[id.index()] = Node::VarRef,
                Decl::Rule | Decl::Fn => nodes[id.index()] = Node::RuleRef,
            }
        } else {
            return Err(ResolveError {
                kind: ResolveErrorKind::UnknownIdentifier(name.to_string()),
                span,
            });
        }
        return Ok(());
    }

    // Alias target gets special handling - Ident becomes RuleRef
    if let Node::Alias { content, target } = &nodes[id.index()] {
        let (content, target) = (*content, *target);
        resolve_expr(nodes, ctx, names, content, locals)?;
        if let Node::Ident = &nodes[target.index()] {
            nodes[target.index()] = Node::RuleRef;
        } else {
            resolve_expr(nodes, ctx, names, target, locals)?;
        }
        return Ok(());
    }

    // For expressions need extended locals
    if let Node::For(for_idx) = &nodes[id.index()] {
        let config = ctx.get_for(*for_idx);
        let iterable = config.iterable;
        let body = config.body;
        let bindings = &config.bindings;
        resolve_expr(nodes, ctx, names, iterable, locals)?;
        let inner = Locals {
            scope: LocalScope::ForBindings(bindings),
            parent: Some(locals),
        };
        return resolve_expr(nodes, ctx, names, body, &inner);
    }

    // All remaining cases
    resolve_children(nodes, ctx, names, id, locals)
}

/// Recursively resolve identifiers in the children of a node.
///
/// This handles the mechanical traversal for all remaining node variants.
/// Each child is re-read from the `nodes` slice inside the loop because
/// earlier children may have been mutated by `resolve_expr`.
fn resolve_children(
    nodes: &mut [Node],
    ctx: &AstContext<'_>,
    names: &Names,
    id: NodeId,
    locals: &Locals<'_>,
) -> Result<(), ResolveError> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = nodes[id.index()].child_range() {
        let children = ctx.child_slice(range);
        for i in 0..children.len() {
            resolve_expr(nodes, ctx, names, children[i], locals)?;
        }
        return Ok(());
    }

    match &nodes[id.index()] {
        Node::Call { args, .. } => {
            let args = ctx.child_slice(*args);
            for i in 0..args.len() {
                resolve_expr(nodes, ctx, names, args[i], locals)?;
            }
            Ok(())
        }
        &Node::Object(range) => {
            let fields = ctx.get_object(range);
            for i in 0..fields.len() {
                resolve_expr(nodes, ctx, names, fields[i].1, locals)?;
            }
            Ok(())
        }
        Node::Repeat(inner)
        | Node::Repeat1(inner)
        | Node::Optional(inner)
        | Node::Token(inner)
        | Node::TokenImmediate(inner)
        | Node::Neg(inner) => {
            let inner = *inner;
            resolve_expr(nodes, ctx, names, inner, locals)
        }
        Node::Field { content, .. } | Node::Reserved { content, .. } => {
            let content = *content;
            resolve_expr(nodes, ctx, names, content, locals)
        }
        Node::Append { left, right } => {
            let (left, right) = (*left, *right);
            resolve_expr(nodes, ctx, names, left, locals)?;
            resolve_expr(nodes, ctx, names, right, locals)
        }
        Node::Prec { value, content }
        | Node::PrecLeft { value, content }
        | Node::PrecRight { value, content }
        | Node::PrecDynamic { value, content } => {
            let (value, content) = (*value, *content);
            resolve_expr(nodes, ctx, names, value, locals)?;
            resolve_expr(nodes, ctx, names, content, locals)
        }
        Node::DynRegex { pattern, flags } => {
            let (pattern, flags) = (*pattern, *flags);
            resolve_expr(nodes, ctx, names, pattern, locals)?;
            if let Some(flags) = flags {
                resolve_expr(nodes, ctx, names, flags, locals)?;
            }
            Ok(())
        }
        Node::FieldAccess { obj, .. } | Node::RuleInline { obj, .. } => {
            let obj = *obj;
            resolve_expr(nodes, ctx, names, obj, locals)
        }
        _ => Ok(()),
    }
}

/// An error encountered during name resolution.
#[derive(Debug, Serialize, Error)]
pub struct ResolveError {
    pub kind: ResolveErrorKind,
    pub span: Span,
}

/// The specific kind of resolve error.
#[derive(Debug, PartialEq, Serialize)]
pub enum ResolveErrorKind {
    DuplicateDeclaration(String),
    UnknownIdentifier(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ResolveErrorKind::DuplicateDeclaration(name) => {
                write!(f, "duplicate declaration '{name}'")
            }
            ResolveErrorKind::UnknownIdentifier(name) => write!(f, "unknown identifier '{name}'"),
        }
    }
}
