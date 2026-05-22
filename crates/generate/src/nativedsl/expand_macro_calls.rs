//! Inlines top-level rule-set macro invocations. Runs between apply_cfg and
//! resolve so the rest of the pipeline only sees concrete rule decls.
//!
//! For each `Node::Call` in `root_items`:
//!   - Look up the named macro in the **current module** (imported and
//!     qualified macro calls are intentionally not supported at top level;
//!     the parser rejects `mod::name(...)` shape at item position).
//!   - Require `MacroKind::RuleSet` (else error).
//!   - For each rule decl in the macro's `Node::RuleSet` body, deep-clone
//!     the subtree substituting `MacroParam` refs with copies of the
//!     corresponding arg subtree, then emit a `Node::ExpandedRule`:
//!       - `Node::Rule` -> name interned via `intern_span` (source span)
//!       - `Node::ComputedRule` -> evaluate the substituted `name_expr` via
//!         the tiny 3-arm string evaluator (StringLit / RawStringLit /
//!         Concat), intern via `intern_owned`
//!
//! `SymRef` in expression position currently clones literally; the
//! synthesized-name reference path lands in a followup (needs IdentKind
//! extension + StringPool plumbing in resolve).

use serde::Serialize;
use thiserror::Error;

use super::{
    ModuleId,
    ast::{ChildRange, MacroId, MacroKind, ModuleContext, Node, NodeId, SharedAst, Span},
    lexer::is_ident_str,
    string_pool::{Str, StringPool},
};

/// Expand all top-level rule-set macro invocations in `ctx.root_items`,
/// replacing each `Node::Call` with the sequence of `Node::ExpandedRule`
/// nodes its expansion produced.
pub fn expand_macro_calls(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &mut ModuleContext,
    mod_id: ModuleId,
) -> Result<(), ExpandError> {
    // Walk only the original indices. Each Call replaces itself in place
    // (first expanded rule into its slot; remaining rules pushed to the end
    // of root_items). Parser enforces >=1 rule per rule-set macro body so
    // the slot is always filled.
    let original_len = ctx.root_items.len();
    for i in 0..original_len {
        let id = ctx.root_items[i];
        if matches!(shared.arena.get(id), Node::Call { .. }) {
            expand_one_call(shared, strings, ctx, mod_id, id, i)?;
        }
    }
    Ok(())
}

/// Linear scan of `ctx.root_items` for a local macro by name. Helper /
/// imported macros are intentionally not visible at item position.
fn lookup_local_macro(shared: &SharedAst, ctx: &ModuleContext, name: &str) -> Option<MacroId> {
    ctx.root_items.iter().find_map(|&item_id| {
        if let Node::Macro(macro_id) = shared.arena.get(item_id) {
            let cfg = shared.pools.get_macro(*macro_id);
            (ctx.text(cfg.name) == name).then_some(*macro_id)
        } else {
            None
        }
    })
}

fn expand_one_call(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &mut ModuleContext,
    mod_id: ModuleId,
    call_id: NodeId,
    slot: usize,
) -> Result<(), ExpandError> {
    let Node::Call { name, args } = *shared.arena.get(call_id) else {
        unreachable!()
    };
    let name_span = shared.arena.span(name);
    let name_text = ctx.text(name_span);
    // Linear scan over root_items for the matching local macro. Helper /
    // imported macros are intentionally not visible at item position (the
    // parser also rejects `mod::name(...)` here). For typical grammars with
    // <50 macros, this is fast enough that a cache is not worth its borrow
    // complexity.
    let macro_id = lookup_local_macro(shared, ctx, name_text).ok_or_else(|| {
        ExpandError::new(
            ExpandErrorKind::UnknownMacro(name_text.to_string()),
            name_span,
        )
    })?;
    // Snapshot the bits of the macro config we need before recursive calls
    // start borrowing `shared` mutably.
    let config = shared.pools.get_macro(macro_id);
    let kind = config.kind;
    let param_count = config.params.len();
    let body_id = config.body;
    let MacroKind::RuleSet = kind else {
        return Err(ExpandError::new(
            ExpandErrorKind::ExpressionMacroAsItem(name_text.to_string()),
            name_span,
        ));
    };
    let Node::RuleSet(rule_range) = *shared.arena.get(body_id) else {
        // Parser guarantees RuleSet body for RuleSet-kind macros.
        unreachable!()
    };
    if args.len as usize != param_count {
        return Err(ExpandError::new(
            ExpandErrorKind::ArgCountMismatch {
                macro_name: name_text.to_string(),
                expected: param_count,
                got: args.len as usize,
            },
            shared.arena.span(call_id),
        ));
    }
    // Reading by absolute index keeps the iteration safe across recursive
    // calls that only ever *append* to `pools.children` (per the established
    // walker pattern in apply_cfg).
    let call_span = shared.arena.span(call_id);
    let args_start = args.start as usize;
    let rule_start = rule_range.start as usize;
    let rule_end = rule_start + rule_range.len as usize;
    for (offset, i) in (rule_start..rule_end).enumerate() {
        let rule_id = shared.pools.children[i];
        let expanded = expand_rule(
            shared, strings, ctx, args_start, param_count, mod_id, rule_id, call_span,
        )?;
        if offset == 0 {
            ctx.root_items[slot] = expanded;
        } else {
            ctx.root_items.push(expanded);
        }
    }
    Ok(())
}

/// Expand one `Node::Rule` or `Node::ComputedRule` from a macro body into a
/// `Node::ExpandedRule`. Args live at `pools.children[args_start..args_start+arg_count]`.
fn expand_rule(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &ModuleContext,
    args_start: usize,
    arg_count: usize,
    mod_id: ModuleId,
    rule_id: NodeId,
    call_span: Span,
) -> Result<NodeId, ExpandError> {
    let node = *shared.arena.get(rule_id);
    let (is_override, name, body) = match node {
        Node::Rule {
            is_override,
            name,
            body,
        } => {
            let name_str = strings.intern_span(name, mod_id);
            let cloned_body = clone_with_subst(shared, args_start, body)?;
            (is_override, name_str, cloned_body)
        }
        Node::ComputedRule {
            is_override,
            name_expr,
            body,
        } => {
            let name_span = shared.arena.span(name_expr);
            let evaluated =
                eval_name(shared, strings, ctx, args_start, arg_count, name_expr, name_span)?;
            let cloned_body = clone_with_subst(shared, args_start, body)?;
            (is_override, evaluated, cloned_body)
        }
        _ => {
            // Parser only places Rule/ComputedRule in a RuleSet body.
            unreachable!()
        }
    };
    Ok(shared.arena.push(
        Node::ExpandedRule {
            is_override,
            name,
            body,
        },
        call_span,
    ))
}

/// Tiny 3-arm name evaluator. Accepts string literals, macro params
/// (whose arg substring resolves recursively), and `concat(...)` of
/// string-shaped children. Anything else is an error here; typecheck
/// would catch most of these earlier in a later iteration.
fn eval_name(
    shared: &SharedAst,
    strings: &mut StringPool,
    ctx: &ModuleContext,
    args_start: usize,
    arg_count: usize,
    node_id: NodeId,
    name_expr_span: Span,
) -> Result<Str, ExpandError> {
    let mut buf = String::new();
    eval_name_into(shared, ctx, args_start, arg_count, node_id, &mut buf)?;
    if !is_ident_str(&buf) {
        return Err(ExpandError::new(
            ExpandErrorKind::InvalidRuleName(buf),
            name_expr_span,
        ));
    }
    Ok(strings.intern_owned(&buf))
}

fn eval_name_into(
    shared: &SharedAst,
    ctx: &ModuleContext,
    args_start: usize,
    arg_count: usize,
    node_id: NodeId,
    out: &mut String,
) -> Result<(), ExpandError> {
    let node = *shared.arena.get(node_id);
    let span = shared.arena.span(node_id);
    match node {
        Node::StringLit => {
            // Span is quote-stripped by parse_primary. Push the raw bytes;
            // escapes that aren't valid identifier chars (`\n`, etc.) fall
            // out at the post-eval `is_ident_str` check.
            out.push_str(ctx.text(span));
            Ok(())
        }
        Node::RawStringLit { hash_count } => {
            // `r#"text"#`: strip `r` + (hash_count+1) quote-side chars.
            let prefix = 2 + u32::from(hash_count);
            let suffix = 1 + u32::from(hash_count);
            let inner = Span::new(span.start + prefix, span.end - suffix);
            out.push_str(ctx.text(inner));
            Ok(())
        }
        Node::MacroParam { index, .. } => {
            let idx = index as usize;
            assert!(idx < arg_count, "param index out of range; checked at call site");
            let arg_id = shared.pools.children[args_start + idx];
            eval_name_into(shared, ctx, args_start, arg_count, arg_id, out)
        }
        Node::Concat(range) => {
            let start = range.start as usize;
            let end = start + range.len as usize;
            for i in start..end {
                let child = shared.pools.children[i];
                eval_name_into(shared, ctx, args_start, arg_count, child, out)?;
            }
            Ok(())
        }
        _ => Err(ExpandError::new(ExpandErrorKind::NonStringInName, span)),
    }
}

/// Deep-copy `src_id` into the arena, substituting each `MacroParam` ref
/// with a fresh copy of the corresponding arg subtree.
fn clone_with_subst(
    shared: &mut SharedAst,
    args_start: usize,
    src_id: NodeId,
) -> Result<NodeId, ExpandError> {
    let span = shared.arena.span(src_id);
    let src = *shared.arena.get(src_id);
    // MacroParam: substitute with a fresh clone of args[index] (so multiple
    // uses don't share NodeIds; keeps per-use span attribution distinct).
    if let Node::MacroParam { index, .. } = src {
        let arg_id = shared.pools.children[args_start + index as usize];
        return clone_with_subst(shared, args_start, arg_id);
    }
    let new_node = match src {
        // Leaves: identical copy.
        Node::Grammar
        | Node::StringLit
        | Node::RawStringLit { .. }
        | Node::IntLit(_)
        | Node::Blank
        | Node::Ident(_)
        | Node::ModuleRef { .. }
        | Node::ForBinding { .. }
        | Node::Unreachable
        | Node::External { .. } => src,
        // Single-child wrappers.
        Node::Repeat { kind, inner } => Node::Repeat {
            kind,
            inner: clone_with_subst(shared, args_start, inner)?,
        },
        Node::Token { immediate, inner } => Node::Token {
            immediate,
            inner: clone_with_subst(shared, args_start, inner)?,
        },
        Node::Neg(c) => Node::Neg(clone_with_subst(shared, args_start, c)?),
        Node::Field { name, content } => Node::Field {
            name,
            content: clone_with_subst(shared, args_start, content)?,
        },
        Node::Reserved { context, content } => Node::Reserved {
            context,
            content: clone_with_subst(shared, args_start, content)?,
        },
        Node::FieldAccess { obj, field } => Node::FieldAccess {
            obj: clone_with_subst(shared, args_start, obj)?,
            field,
        },
        Node::QualifiedAccess { obj, member } => Node::QualifiedAccess {
            obj: clone_with_subst(shared, args_start, obj)?,
            member,
        },
        Node::GrammarConfig { module, field } => Node::GrammarConfig {
            module: clone_with_subst(shared, args_start, module)?,
            field,
        },
        Node::SymRef { expr } => Node::SymRef {
            expr: clone_with_subst(shared, args_start, expr)?,
        },
        // Two-child wrappers.
        Node::Alias { content, target } => Node::Alias {
            content: clone_with_subst(shared, args_start, content)?,
            target: clone_with_subst(shared, args_start, target)?,
        },
        Node::Append { left, right } => Node::Append {
            left: clone_with_subst(shared, args_start, left)?,
            right: clone_with_subst(shared, args_start, right)?,
        },
        Node::BinOp { op, lhs, rhs } => Node::BinOp {
            op,
            lhs: clone_with_subst(shared, args_start, lhs)?,
            rhs: clone_with_subst(shared, args_start, rhs)?,
        },
        Node::Prec {
            kind,
            value,
            content,
        } => Node::Prec {
            kind,
            value: clone_with_subst(shared, args_start, value)?,
            content: clone_with_subst(shared, args_start, content)?,
        },
        Node::DynRegex { pattern, flags } => {
            let p = clone_with_subst(shared, args_start, pattern)?;
            let f = flags
                .map(|id| clone_with_subst(shared, args_start, id))
                .transpose()?;
            Node::DynRegex {
                pattern: p,
                flags: f,
            }
        }
        // Variadic via ChildRange.
        Node::SeqOrChoice { seq, range } => Node::SeqOrChoice {
            seq,
            range: clone_range(shared, args_start, range)?,
        },
        Node::List(range) => Node::List(clone_range(shared, args_start, range)?),
        Node::Tuple(range) => Node::Tuple(clone_range(shared, args_start, range)?),
        Node::Concat(range) => Node::Concat(clone_range(shared, args_start, range)?),
        Node::Call { name, args: cr } => Node::Call {
            name: clone_with_subst(shared, args_start, name)?,
            args: clone_range(shared, args_start, cr)?,
        },
        Node::QualifiedCall(range) => {
            Node::QualifiedCall(clone_range(shared, args_start, range)?)
        }
        Node::For { for_id, body } => Node::For {
            for_id,
            body: clone_range(shared, args_start, body)?,
        },
        Node::Object(range) => {
            // Object children live in object_fields as (Span, NodeId). Read by
            // index to avoid borrowing the pool across the recursive append.
            let start = range.start as usize;
            let end = start + range.len as usize;
            let mut new_pairs: Vec<(Span, NodeId)> = Vec::with_capacity(end - start);
            for i in start..end {
                let (k, v) = shared.pools.object_fields[i];
                let v2 = clone_with_subst(shared, args_start, v)?;
                new_pairs.push((k, v2));
            }
            let new_range = shared
                .pools
                .push_object(new_pairs)
                .ok_or_else(|| ExpandError::without_span(ExpandErrorKind::TooManyChildren))?;
            Node::Object(new_range)
        }
        // Not expected inside a rule body — parser/validation prevents these
        // shapes here, but match exhaustively.
        Node::Macro(_)
        | Node::Rule { .. }
        | Node::ComputedRule { .. }
        | Node::RuleSet(_)
        | Node::Let { .. }
        | Node::ExpandedRule { .. }
        | Node::Cfg { .. } => unreachable!(),
        // Handled by the early MacroParam guard above.
        Node::MacroParam { .. } => unreachable!(),
    };
    Ok(shared.arena.push(new_node, span))
}

fn clone_range(
    shared: &mut SharedAst,
    args_start: usize,
    range: ChildRange,
) -> Result<ChildRange, ExpandError> {
    let start = range.start as usize;
    let end = start + range.len as usize;
    let mut new_ids: Vec<NodeId> = Vec::with_capacity(end - start);
    for i in start..end {
        let id = shared.pools.children[i];
        new_ids.push(clone_with_subst(shared, args_start, id)?);
    }
    shared
        .pools
        .push_children(&new_ids)
        .ok_or_else(|| ExpandError::without_span(ExpandErrorKind::TooManyChildren))
}

use super::ExpandError;

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum ExpandErrorKind {
    #[error("unknown macro '{0}'")]
    UnknownMacro(String),
    #[error(
        "'{0}' is an expression macro and cannot be invoked at top level; \
         use it inside a rule body instead"
    )]
    ExpressionMacroAsItem(String),
    #[error("macro '{macro_name}': expected {expected} arguments, got {got}")]
    ArgCountMismatch {
        macro_name: String,
        expected: usize,
        got: usize,
    },
    #[error("computed rule name expression must be a string literal, parameter, or concat(...)")]
    NonStringInName,
    #[error("computed rule name '{0}' is not a valid identifier")]
    InvalidRuleName(String),
    #[error("too many elements (maximum 65535)")]
    TooManyChildren,
}
