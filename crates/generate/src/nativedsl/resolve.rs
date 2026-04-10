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

use super::ast::{Ast, AstContext, Node, NodeId, Note, NoteMessage, Param, Span};

/// Zero-allocation scope chain (linked list on stack, borrows from configs).
struct Locals<'a> {
    scope: LocalScope<'a>,
    #[expect(clippy::use_self, reason = "no Self with generics")]
    parent: Option<&'a Locals<'a>>,
}

enum LocalScope<'a> {
    Empty,
    FnParams(&'a [Param]),
    ForBindings(&'a [(Span, NodeId)]),
}

impl Locals<'_> {
    const EMPTY: Locals<'static> = Locals {
        scope: LocalScope::Empty,
        parent: None,
    };

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
pub fn resolve(
    ast: &mut Ast<'_>,
    base_rule_names: &[String],
    inherit_span: Option<Span>,
    grammar_path: &std::path::Path,
) -> Result<(), ResolveError> {
    let mut names = collect_names(ast, grammar_path)?;
    // Register inherited rule names so they resolve to RuleRef.
    // Use the inherit statement's span so "first defined here" points there.
    let inherit_span = inherit_span.unwrap_or(Span::new(0, 0));
    for name in base_rule_names {
        // Don't error on duplicates - derived rules intentionally shadow base rules
        if names.get(name).is_none() {
            names.decls.insert(name.clone(), (Decl::Rule, inherit_span));
        }
    }
    let n_items = ast.root_items.len();
    for i in 0..n_items {
        let item_id = ast.root_items[i];
        // Register let bindings in order so forward references are caught
        if let Node::Let { name, .. } = ast.node(item_id) {
            let name_text = ast.node_text(*name);
            let span = ast.span(item_id);
            names.insert(name_text, Decl::Var, span, grammar_path, ast.source())?;
        }
        resolve_item(ast, &names, item_id)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Decl {
    Rule,
    Var,
    Fn,
}

struct Names {
    decls: FxHashMap<String, (Decl, Span)>,
}

impl Names {
    fn insert(
        &mut self,
        name: &str,
        kind: Decl,
        span: Span,
        grammar_path: &std::path::Path,
        source: &str,
    ) -> Result<(), ResolveError> {
        if let Some((_, first_span)) = self.decls.get(name) {
            return Err(ResolveError {
                kind: ResolveErrorKind::DuplicateDeclaration(name.to_string()),
                span,
                note: Some(Note {
                    message: NoteMessage::FirstDefinedHere,
                    span: *first_span,
                    path: grammar_path.to_path_buf(),
                    source: source.to_string(),
                }),
            });
        }
        self.decls.insert(name.to_string(), (kind, span));
        Ok(())
    }

    fn get(&self, name: &str) -> Option<Decl> {
        self.decls.get(name).map(|(d, _)| *d)
    }
}

/// Pass 1: scan top-level items and register all declared names.
///
/// Rules, let-bindings, functions, and external tokens in the grammar block
/// all occupy the same namespace. Duplicate names are rejected.
fn collect_names(ast: &Ast<'_>, grammar_path: &std::path::Path) -> Result<Names, ResolveError> {
    let mut names = Names {
        decls: FxHashMap::default(),
    };
    let source = ast.source();

    // Register rules and fns upfront (forward references allowed).
    // Let bindings are registered during pass 2 in item order (no forward references).
    for i in 0..ast.root_items.len() {
        let item_id = ast.root_items[i];
        let span = ast.span(item_id);
        match ast.node(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                names.insert(ast.node_text(*name), Decl::Rule, span, grammar_path, source)?;
            }
            Node::Fn(fn_idx) => {
                let config = ast.get_fn(*fn_idx);
                names.insert(
                    ast.node_text(config.name),
                    Decl::Fn,
                    span,
                    grammar_path,
                    source,
                )?;
            }
            _ => {}
        }
    }

    // Register identifiers from config fields that aren't already declared.
    // In grammar.js, externals/inline/supertypes/word can reference names
    // not declared as rules. We mirror that by pre-registering them.
    // Only works for literal list expressions; arbitrary expressions
    // (e.g. append(...)) are resolved normally.
    if let Some(config) = &ast.context.grammar_config {
        let config_lists = [config.externals, config.inline, config.supertypes];
        for list_id in config_lists.into_iter().flatten() {
            if let Node::List(range) = ast.node(list_id) {
                for &child in ast.child_slice(*range) {
                    if matches!(ast.node(child), Node::Ident) {
                        let name = ast.text(ast.span(child));
                        if names.get(name).is_none() {
                            names.insert(
                                name,
                                Decl::Rule,
                                ast.span(child),
                                grammar_path,
                                source,
                            )?;
                        }
                    }
                }
            }
        }
        if let Some(word_id) = config.word
            && matches!(ast.node(word_id), Node::Ident)
        {
            let name = ast.text(ast.span(word_id));
            if names.get(name).is_none() {
                names.insert(name, Decl::Rule, ast.span(word_id), grammar_path, source)?;
            }
        }
    }

    Ok(names)
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item(ast: &mut Ast<'_>, names: &Names, item_id: NodeId) -> Result<(), ResolveError> {
    match ast.node(item_id) {
        Node::Grammar => {
            let ctx = &ast.context;
            // INVARIANT: set during grammar block parsing, always present here
            let grammar_config = ctx.grammar_config.as_ref().unwrap();
            let nodes = &mut ast.nodes;
            for id in [
                grammar_config.inherits,
                grammar_config.extras,
                grammar_config.externals,
                grammar_config.inline,
                grammar_config.supertypes,
                grammar_config.word,
            ]
            .into_iter()
            .flatten()
            {
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
fn resolve_expr(
    nodes: &mut [Node],
    ctx: &AstContext<'_>,
    names: &Names,
    id: NodeId,
    locals: &Locals<'_>,
) -> Result<(), ResolveError> {
    // Handle Ident mutation first
    if matches!(nodes[id.index()], Node::Ident) {
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
                note: None,
            });
        }
        return Ok(());
    }

    // Alias target gets special handling - Ident becomes RuleRef
    if let Node::Alias { content, target } = &nodes[id.index()] {
        let (content, target) = (*content, *target);
        resolve_expr(nodes, ctx, names, content, locals)?;
        if matches!(nodes[target.index()], Node::Ident) {
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

/// Resolve identifiers in children of a node (mechanical traversal).
fn resolve_children(
    nodes: &mut [Node],
    ctx: &AstContext<'_>,
    names: &Names,
    id: NodeId,
    locals: &Locals<'_>,
) -> Result<(), ResolveError> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = nodes[id.index()].child_range() {
        for child in ctx.child_slice(range) {
            resolve_expr(nodes, ctx, names, *child, locals)?;
        }
        return Ok(());
    }

    match &nodes[id.index()] {
        Node::Call { args, .. } => {
            for arg in ctx.child_slice(*args) {
                resolve_expr(nodes, ctx, names, *arg, locals)?;
            }
            Ok(())
        }
        &Node::Object(range) => {
            for field in ctx.get_object(range) {
                resolve_expr(nodes, ctx, names, field.1, locals)?;
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

#[derive(Debug, Serialize, Error)]
pub struct ResolveError {
    pub kind: ResolveErrorKind,
    pub span: Span,
    pub note: Option<Note>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
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
