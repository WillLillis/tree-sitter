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
//! Walks every expression in the AST and replaces `Node::Ident` nodes in
//! expression positions with either:
//! - `Node::RuleRef` - the identifier refers to a grammar rule or function
//! - `Node::VarRef` - the identifier refers to a `let` binding, function
//!   parameter, or `for` loop variable
//!
//! Identifiers in defining positions (rule names, let binding names, function
//! names, parameter names, object keys, field names) remain as `Node::Ident`.
//!
//! Local variables (function parameters and for-loop bindings) shadow
//! top-level names. The scope chain is a zero-allocation linked list
//! ([`Locals`]) that borrows directly from the function/for config data.

use std::path::Path;

use rustc_hash::{FxHashMap, FxHashSet};
use serde::Serialize;
use thiserror::Error;

use super::{
    Note, NoteMessage,
    ast::{Ast, AstContext, Node, NodeArena, NodeId, Param, Span},
};
use crate::grammars::InputGrammar;

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

    /// Iterative walk + manual loops (instead of recursion + `.any()` closures)
    /// to reduce register pressure in the inner loops.
    fn contains(&self, arena: &NodeArena, ctx: &AstContext, name: &str) -> bool {
        let source = ctx.source();
        let mut current = self;
        loop {
            match &current.scope {
                LocalScope::Empty => {}
                LocalScope::FnParams(params) => {
                    for p in *params {
                        if arena.span(p.name).resolve(source) == name {
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

/// Run name resolution on the AST.
///
/// Collects all top-level declarations, then resolves every identifier in
/// every item body. After this pass, no `Node::Ident` nodes remain in the
/// AST - they have all been replaced with `Node::RuleRef` or `Node::VarRef`.
///
/// # Errors
///
/// Returns [`ResolveError`] for duplicate declarations or unknown identifiers.
pub fn resolve<'a>(
    ast: &'a mut Ast,
    base: Option<(&'a InputGrammar, Span)>,
    grammar_path: &Path,
) -> ResolveResult<()> {
    // Split borrow: collect_names borrows ast.context (for &str slices in Names)
    // but not ast.arena, so we can later mutably borrow ast.arena for resolve_item.
    // We pass the arena by-ref but its lifetime is not captured in Names.
    let mut names = collect_names(&ast.arena, &ast.context, &ast.root_items, grammar_path)?;
    // Register inherited rule names so they resolve to RuleRef.
    // Use the inherit statement's span so "first defined here" points there.
    if let Some((base_grammar, inherit_span)) = base {
        for var in &base_grammar.variables {
            if names.get(&var.name).is_none() {
                names.decls.insert(&var.name, (Decl::Rule, inherit_span));
            }
        }
    }
    // Split borrow: names holds &str from ast.context.source (immutable),
    // while resolve_item mutates ast.arena. Access context and arena separately.
    let n_items = ast.root_items.len();
    for i in 0..n_items {
        let item_id = ast.root_items[i];
        // Register let bindings in order so forward references are caught
        if let Node::Let { name, .. } = ast.arena.get(item_id) {
            let name_text = ast.context.text(ast.arena.span(*name));
            let span = ast.arena.span(item_id);
            names.insert(
                name_text,
                Decl::Var,
                span,
                grammar_path,
                ast.context.source(),
            )?;
        }
        resolve_item(&mut ast.arena, &ast.context, &names, item_id)?;
    }

    Ok(())
}

#[derive(Clone, Copy)]
enum Decl {
    Rule,
    Var,
    Fn,
}

struct Names<'src> {
    decls: FxHashMap<&'src str, (Decl, Span)>,
}

impl<'src> Names<'src> {
    #[inline]
    fn insert(
        &mut self,
        name: &'src str,
        kind: Decl,
        span: Span,
        grammar_path: &Path,
        source: &str,
    ) -> ResolveResult<()> {
        if let Some((_, first_span)) = self.decls.get(name) {
            Err(Self::duplicate_error(
                name,
                span,
                *first_span,
                grammar_path,
                source,
            ))?;
        }
        self.decls.insert(name, (kind, span));
        Ok(())
    }

    #[cold]
    fn duplicate_error(
        name: &str,
        span: Span,
        first_span: Span,
        grammar_path: &Path,
        source: &str,
    ) -> ResolveError {
        ResolveError {
            kind: ResolveErrorKind::DuplicateDeclaration(name.to_string()),
            span,
            note: Some(Box::new(Note {
                message: NoteMessage::FirstDefinedHere,
                span: first_span,
                path: grammar_path.to_path_buf(),
                source: source.to_string(),
            })),
        }
    }

    fn get(&self, name: &str) -> Option<Decl> {
        self.decls.get(name).map(|(d, _)| *d)
    }
}

/// Pass 1: scan top-level items and register all declared names.
///
/// Rules, let-bindings, functions, and external tokens in the grammar block
/// all occupy the same namespace. Duplicate names are rejected.
fn collect_names<'a>(
    arena: &NodeArena,
    ctx: &'a AstContext,
    root_items: &[NodeId],
    grammar_path: &Path,
) -> ResolveResult<Names<'a>> {
    let mut names = Names {
        decls: FxHashMap::default(),
    };
    let source = ctx.source();

    // Register rules and fns upfront (forward references allowed).
    // Let bindings are registered during pass 2 in item order (no forward references).
    for &item_id in root_items {
        let span = arena.span(item_id);
        match arena.get(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                names.insert(
                    ctx.text(arena.span(*name)),
                    Decl::Rule,
                    span,
                    grammar_path,
                    source,
                )?;
            }
            Node::Fn(fn_idx) => {
                let config = ctx.get_fn(*fn_idx);
                names.insert(
                    ctx.text(arena.span(config.name)),
                    Decl::Fn,
                    span,
                    grammar_path,
                    source,
                )?;
            }
            _ => {}
        }
    }

    // Collect let binding names (not yet registered - they're added in pass 2
    // to prevent forward references). We need them here to avoid pre-registering
    // config identifiers that shadow let bindings.
    let let_names: FxHashSet<&str> = root_items
        .iter()
        .filter_map(|&id| match arena.get(id) {
            Node::Let { name, .. } => Some(ctx.text(arena.span(*name))),
            _ => None,
        })
        .collect();

    // Pre-register external token names. Externals are the only config field
    // that introduces names not declared as rules - they're provided by an
    // external scanner. Other config fields (inline, supertypes, word, extras,
    // conflicts) must reference rules declared in this file or inherited.
    // We walk List and Append nodes to find bare identifiers.
    if let Some(config) = &ctx.grammar_config
        && let Some(ext_id) = config.externals
    {
        collect_external_names(
            arena,
            ctx,
            ext_id,
            &mut names,
            &let_names,
            grammar_path,
            source,
        )?;
    }

    Ok(names)
}

/// Recursively walk an externals expression to find bare identifiers that
/// need pre-registration. Handles inline lists and `append()` calls.
fn collect_external_names<'a>(
    arena: &NodeArena,
    ctx: &'a AstContext,
    id: NodeId,
    names: &mut Names<'a>,
    let_names: &FxHashSet<&str>,
    grammar_path: &Path,
    source: &'a str,
) -> ResolveResult<()> {
    match arena.get(id) {
        Node::List(range) => {
            for &child in ctx.child_slice(*range) {
                if matches!(arena.get(child), Node::Ident) {
                    let name = ctx.text(arena.span(child));
                    if names.get(name).is_none() && !let_names.contains(name) {
                        names.insert(name, Decl::Rule, arena.span(child), grammar_path, source)?;
                    }
                }
            }
        }
        Node::Append { left, right } => {
            collect_external_names(arena, ctx, *left, names, let_names, grammar_path, source)?;
            collect_external_names(arena, ctx, *right, names, let_names, grammar_path, source)?;
        }
        // String literals in externals (e.g. externals: ["||"]) don't
        // introduce names that need registration.
        Node::StringLit | Node::RawStringLit { .. } => {}
        // grammar_config(base).externals and field access chains resolve to
        // inherited values whose names are already registered via base_grammar.
        Node::GrammarConfig(_) | Node::FieldAccess { .. } => {}
        // Variable references, function calls, etc. can't be followed
        // statically to discover external token names.
        _ => {
            return Err(ResolveError {
                kind: ResolveErrorKind::ExternalsNotInline,
                span: arena.span(id),
                note: None,
            });
        }
    }
    Ok(())
}

/// Pass 2: resolve identifiers within a single top-level item.
fn resolve_item(
    arena: &mut NodeArena,
    ctx: &AstContext,
    names: &Names,
    item_id: NodeId,
) -> ResolveResult<()> {
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
                resolve_expr(arena, ctx, names, id, &Locals::EMPTY)?;
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::OverrideRule { body, .. } => {
            let body = *body;
            resolve_expr(arena, ctx, names, body, &Locals::EMPTY)
        }
        Node::Let { value, .. } => {
            let value = *value;
            resolve_expr(arena, ctx, names, value, &Locals::EMPTY)
        }
        Node::Print(arg) => {
            let arg = *arg;
            resolve_expr(arena, ctx, names, arg, &Locals::EMPTY)
        }
        Node::Fn(fn_idx) => {
            let fn_config = ctx.get_fn(*fn_idx);
            let body = fn_config.body;
            let locals = Locals {
                scope: LocalScope::FnParams(&fn_config.params),
                parent: None,
            };
            resolve_expr(arena, ctx, names, body, &locals)
        }
        _ => Ok(()),
    }
}

/// Resolve identifiers within a single expression.
fn resolve_expr(
    arena: &mut NodeArena,
    ctx: &AstContext,
    names: &Names,
    id: NodeId,
    locals: &Locals<'_>,
) -> ResolveResult<()> {
    // Handle Ident mutation first
    if matches!(arena.get(id), Node::Ident) {
        let span = arena.span(id);
        let name = ctx.text(span);
        if locals.contains(arena, ctx, name) {
            arena.set(id, Node::VarRef);
        } else if let Some(decl) = names.get(name) {
            match decl {
                Decl::Var => arena.set(id, Node::VarRef),
                Decl::Rule | Decl::Fn => arena.set(id, Node::RuleRef),
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

    // Alias target: bare identifiers become RuleRef (named alias) unless
    // they refer to a local variable (VarRef for fn params / for bindings).
    if let Node::Alias { content, target } = arena.get(id) {
        let (content, target) = (*content, *target);
        resolve_expr(arena, ctx, names, content, locals)?;
        if matches!(*arena.get(target), Node::Ident) {
            let name = ctx.text(arena.span(target));
            if locals.contains(arena, ctx, name) {
                arena.set(target, Node::VarRef);
            } else {
                arena.set(target, Node::RuleRef);
            }
        } else {
            resolve_expr(arena, ctx, names, target, locals)?;
        }
        return Ok(());
    }

    // For expressions need extended locals
    if let Node::For(for_idx) = arena.get(id) {
        let config = ctx.get_for(*for_idx);
        let iterable = config.iterable;
        let body = config.body;
        let bindings = &config.bindings;
        resolve_expr(arena, ctx, names, iterable, locals)?;
        let inner = Locals {
            scope: LocalScope::ForBindings(bindings),
            parent: Some(locals),
        };
        return resolve_expr(arena, ctx, names, body, &inner);
    }

    // All remaining cases
    resolve_children(arena, ctx, names, id, locals)
}

/// Resolve identifiers in children of a node (mechanical traversal).
fn resolve_children(
    arena: &mut NodeArena,
    ctx: &AstContext,
    names: &Names,
    id: NodeId,
    locals: &Locals<'_>,
) -> ResolveResult<()> {
    // Variadic nodes: Seq, Choice, List, Tuple, Concat
    if let Some(range) = arena.get(id).child_range() {
        for child in ctx.child_slice(range) {
            resolve_expr(arena, ctx, names, *child, locals)?;
        }
        return Ok(());
    }

    match arena.get(id) {
        Node::Call { name, args } => {
            let (name, args) = (*name, *args);
            resolve_expr(arena, ctx, names, name, locals)?;
            for arg in ctx.child_slice(args) {
                resolve_expr(arena, ctx, names, *arg, locals)?;
            }
            Ok(())
        }
        Node::Object(range) => {
            for field in ctx.get_object(*range) {
                resolve_expr(arena, ctx, names, field.1, locals)?;
            }
            Ok(())
        }
        Node::Repeat(inner)
        | Node::Repeat1(inner)
        | Node::Optional(inner)
        | Node::Token(inner)
        | Node::TokenImmediate(inner)
        | Node::Neg(inner)
        | Node::GrammarConfig(inner)
        | Node::Print(inner) => {
            let inner = *inner;
            resolve_expr(arena, ctx, names, inner, locals)
        }
        Node::Field { content, .. } | Node::Reserved { content, .. } => {
            let content = *content;
            resolve_expr(arena, ctx, names, content, locals)
        }
        Node::Append { left, right } => {
            let (left, right) = (*left, *right);
            resolve_expr(arena, ctx, names, left, locals)?;
            resolve_expr(arena, ctx, names, right, locals)
        }
        Node::Prec { value, content }
        | Node::PrecLeft { value, content }
        | Node::PrecRight { value, content }
        | Node::PrecDynamic { value, content } => {
            let (value, content) = (*value, *content);
            resolve_expr(arena, ctx, names, value, locals)?;
            resolve_expr(arena, ctx, names, content, locals)
        }
        Node::DynRegex { pattern, flags } => {
            let (pattern, flags) = (*pattern, *flags);
            resolve_expr(arena, ctx, names, pattern, locals)?;
            if let Some(flags) = flags {
                resolve_expr(arena, ctx, names, flags, locals)?;
            }
            Ok(())
        }
        Node::FieldAccess { obj, .. } | Node::QualifiedAccess { obj, .. } => {
            let obj = *obj;
            resolve_expr(arena, ctx, names, obj, locals)
        }
        Node::QualifiedCall(range) => {
            let (obj, _name, args) = ctx.get_qualified_call(*range);
            resolve_expr(arena, ctx, names, obj, locals)?;
            // name stays as Ident - it's a member of the import namespace
            for &arg in args {
                resolve_expr(arena, ctx, names, arg, locals)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

pub type ResolveResult<T> = Result<T, ResolveError>;

#[derive(Debug, Serialize, Error)]
pub struct ResolveError {
    pub kind: ResolveErrorKind,
    pub span: Span,
    pub note: Option<Box<Note>>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum ResolveErrorKind {
    DuplicateDeclaration(String),
    UnknownIdentifier(String),
    ExternalsNotInline,
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ResolveErrorKind::DuplicateDeclaration(name) => {
                write!(f, "duplicate declaration '{name}'")
            }
            ResolveErrorKind::UnknownIdentifier(name) => write!(f, "unknown identifier '{name}'"),
            ResolveErrorKind::ExternalsNotInline => {
                write!(
                    f,
                    "external tokens must be listed directly (e.g. `externals: [a, b]`), \
                     not via variable references"
                )
            }
        }
    }
}
