//! Inlines top-level rule-set macro invocations between apply_cfg and
//! resolve. Each `Node::Call` at item position is replaced by the
//! `Node::ExpandedRule`s produced from the macro's body, with `MacroParam`
//! refs substituted by call-site args. `Node::SymRef` becomes
//! `Node::SynthRef` carrying the evaluated name.
//!
//! Only local macros are visible at item position; qualified calls aren't
//! supported (the parser rejects `mod::foo(...)` there).

use serde::Serialize;
use thiserror::Error;

use super::{
    ModuleId,
    ast::{ChildRange, MacroId, MacroKind, ModuleContext, Node, NodeId, SharedAst, Span},
    lexer::is_ident_str,
    string_pool::{Str, StringPool},
};

pub fn expand_macro_calls(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &mut ModuleContext,
    mod_id: ModuleId,
) -> Result<(), ExpandError> {
    // Walk only the original indices. Each Call writes its first expanded
    // rule into its own slot; the rest are pushed to the end. Parser's
    // EmptyRuleSetMacroBody guarantees the slot is always filled.
    //
    // `scratch` is a stack reused across every variadic clone in the macro
    // bodies (SeqOrChoice, List, Tuple, Concat, Call args, QualifiedCall).
    // stack_scope! saves the current len at entry and truncates on exit, so
    // nested recursion just stacks above the parent's range.
    let mut scratch: Vec<NodeId> = Vec::new();
    let original_len = ctx.root_items.len();
    for i in 0..original_len {
        let id = ctx.root_items[i];
        if matches!(shared.arena.get(id), Node::Call { .. }) {
            expand_one_call(shared, strings, ctx, mod_id, id, i, &mut scratch)?;
        }
    }
    Ok(())
}

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
    scratch: &mut Vec<NodeId>,
) -> Result<(), ExpandError> {
    let Node::Call { name, args } = *shared.arena.get(call_id) else {
        unreachable!()
    };
    let name_span = shared.arena.span(name);
    let name_text = ctx.text(name_span);
    let macro_id = lookup_local_macro(shared, ctx, name_text).ok_or_else(|| {
        ExpandError::new(
            ExpandErrorKind::UnknownMacro(name_text.to_string()),
            name_span,
        )
    })?;
    // Snapshot before recursive calls reborrow shared mutably.
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
    // Read by absolute index: recursive expansion only appends to
    // pools.children (same walker pattern apply_cfg uses).
    let call_span = shared.arena.span(call_id);
    let args_start = args.start as usize;
    let rule_start = rule_range.start as usize;
    let rule_end = rule_start + rule_range.len as usize;
    for (offset, i) in (rule_start..rule_end).enumerate() {
        let rule_id = shared.pools.children[i];
        let expanded = expand_rule(
            shared, strings, ctx, args_start, mod_id, rule_id, call_span, scratch,
        )?;
        if offset == 0 {
            ctx.root_items[slot] = expanded;
        } else {
            ctx.root_items.push(expanded);
        }
    }
    Ok(())
}

fn expand_rule(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &ModuleContext,
    args_start: usize,
    mod_id: ModuleId,
    rule_id: NodeId,
    call_span: Span,
    scratch: &mut Vec<NodeId>,
) -> Result<NodeId, ExpandError> {
    let node = *shared.arena.get(rule_id);
    let (is_override, name, body) = match node {
        Node::Rule {
            is_override,
            name,
            body,
        } => {
            let name_str = strings.intern_span(name, mod_id);
            let cloned_body = clone_with_subst(shared, strings, ctx, args_start, body, scratch)?;
            (is_override, name_str, cloned_body)
        }
        Node::ComputedRule {
            is_override,
            name_expr,
            body,
        } => {
            let name_span = shared.arena.span(name_expr);
            let evaluated = eval_name(shared, strings, ctx, args_start, name_expr, name_span)?;
            let cloned_body = clone_with_subst(shared, strings, ctx, args_start, body, scratch)?;
            (is_override, evaluated, cloned_body)
        }
        // Parser only places Rule/ComputedRule in a RuleSet body.
        _ => unreachable!(),
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

/// Accepts string literals, macro params (resolved into args), and
/// `concat(...)` of those. Typecheck catches non-str_t at definition
/// time; this errors at expand time for shapes outside the limited
/// compile-time evaluator subset (e.g. references to lets).
fn eval_name(
    shared: &SharedAst,
    strings: &mut StringPool,
    ctx: &ModuleContext,
    args_start: usize,
    node_id: NodeId,
    name_expr_span: Span,
) -> Result<Str, ExpandError> {
    let mut buf = String::new();
    eval_name_into(shared, ctx, args_start, node_id, &mut buf)?;
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
    node_id: NodeId,
    out: &mut String,
) -> Result<(), ExpandError> {
    let node = *shared.arena.get(node_id);
    let span = shared.arena.span(node_id);
    match node {
        Node::StringLit => {
            // Span is quote-stripped by parse_primary.
            out.push_str(ctx.text(span));
            Ok(())
        }
        Node::RawStringLit { hash_count } => {
            let prefix = 2 + u32::from(hash_count);
            let suffix = 1 + u32::from(hash_count);
            let inner = Span::new(span.start + prefix, span.end - suffix);
            out.push_str(ctx.text(inner));
            Ok(())
        }
        Node::MacroParam { index, .. } => {
            let arg_id = shared.pools.children[args_start + index as usize];
            eval_name_into(shared, ctx, args_start, arg_id, out)
        }
        Node::Concat(range) => {
            let start = range.start as usize;
            let end = start + range.len as usize;
            for i in start..end {
                let child = shared.pools.children[i];
                eval_name_into(shared, ctx, args_start, child, out)?;
            }
            Ok(())
        }
        _ => Err(ExpandError::new(ExpandErrorKind::NonStringInName, span)),
    }
}

/// Deep-copy `src_id`, substituting each `MacroParam` with a fresh clone
/// of the corresponding arg subtree (per-use clones keep span attribution
/// distinct). `SymRef` collapses to a `SynthRef` carrying the evaluated
/// name.
fn clone_with_subst(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &ModuleContext,
    args_start: usize,
    src_id: NodeId,
    scratch: &mut Vec<NodeId>,
) -> Result<NodeId, ExpandError> {
    let span = shared.arena.span(src_id);
    let src = *shared.arena.get(src_id);
    if let Node::MacroParam { index, .. } = src {
        // Top-level rule-set macro args come from grammar-level expressions
        // (no enclosing macro), so they can't contain MacroParam themselves.
        // Return the arg id directly instead of deep-cloning - downstream
        // passes are read-only on resolved nodes and tolerate the resulting
        // DAG. For a body that references the same param N times, this
        // collapses N deep clones into N references to one subtree.
        return Ok(shared.pools.children[args_start + index as usize]);
    }
    if let Node::SymRef { expr } = src {
        // eval_name handles MacroParam substitution inline, so no separate
        // clone of `expr` is needed.
        let name = eval_name(shared, strings, ctx, args_start, expr, span)?;
        return Ok(shared.arena.push(Node::SynthRef { name }, span));
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
            inner: clone_with_subst(shared, strings, ctx, args_start, inner, scratch)?,
        },
        Node::Token { immediate, inner } => Node::Token {
            immediate,
            inner: clone_with_subst(shared, strings, ctx, args_start, inner, scratch)?,
        },
        Node::Neg(c) => Node::Neg(clone_with_subst(
            shared, strings, ctx, args_start, c, scratch,
        )?),
        Node::Field { name, content } => Node::Field {
            name,
            content: clone_with_subst(shared, strings, ctx, args_start, content, scratch)?,
        },
        Node::Reserved { context, content } => Node::Reserved {
            context,
            content: clone_with_subst(shared, strings, ctx, args_start, content, scratch)?,
        },
        Node::FieldAccess { obj, field } => Node::FieldAccess {
            obj: clone_with_subst(shared, strings, ctx, args_start, obj, scratch)?,
            field,
        },
        Node::QualifiedAccess { obj, member } => Node::QualifiedAccess {
            obj: clone_with_subst(shared, strings, ctx, args_start, obj, scratch)?,
            member,
        },
        Node::GrammarConfig { module, field } => Node::GrammarConfig {
            module: clone_with_subst(shared, strings, ctx, args_start, module, scratch)?,
            field,
        },
        // Two-child wrappers.
        Node::Alias { content, target } => Node::Alias {
            content: clone_with_subst(shared, strings, ctx, args_start, content, scratch)?,
            target: clone_with_subst(shared, strings, ctx, args_start, target, scratch)?,
        },
        Node::Append { left, right } => Node::Append {
            left: clone_with_subst(shared, strings, ctx, args_start, left, scratch)?,
            right: clone_with_subst(shared, strings, ctx, args_start, right, scratch)?,
        },
        Node::BinOp { op, lhs, rhs } => Node::BinOp {
            op,
            lhs: clone_with_subst(shared, strings, ctx, args_start, lhs, scratch)?,
            rhs: clone_with_subst(shared, strings, ctx, args_start, rhs, scratch)?,
        },
        Node::Prec {
            kind,
            value,
            content,
        } => Node::Prec {
            kind,
            value: clone_with_subst(shared, strings, ctx, args_start, value, scratch)?,
            content: clone_with_subst(shared, strings, ctx, args_start, content, scratch)?,
        },
        Node::DynRegex { pattern, flags } => {
            let p = clone_with_subst(shared, strings, ctx, args_start, pattern, scratch)?;
            let f = flags
                .map(|id| clone_with_subst(shared, strings, ctx, args_start, id, scratch))
                .transpose()?;
            Node::DynRegex {
                pattern: p,
                flags: f,
            }
        }
        // Variadic via ChildRange.
        Node::SeqOrChoice { seq, range } => Node::SeqOrChoice {
            seq,
            range: clone_range(shared, strings, ctx, args_start, range, scratch)?,
        },
        Node::List(range) => Node::List(clone_range(
            shared, strings, ctx, args_start, range, scratch,
        )?),
        Node::Tuple(range) => Node::Tuple(clone_range(
            shared, strings, ctx, args_start, range, scratch,
        )?),
        Node::Concat(range) => Node::Concat(clone_range(
            shared, strings, ctx, args_start, range, scratch,
        )?),
        Node::Call { name, args: cr } => Node::Call {
            name: clone_with_subst(shared, strings, ctx, args_start, name, scratch)?,
            args: clone_range(shared, strings, ctx, args_start, cr, scratch)?,
        },
        Node::QualifiedCall(range) => Node::QualifiedCall(clone_range(
            shared, strings, ctx, args_start, range, scratch,
        )?),
        Node::For { for_id, body } => Node::For {
            for_id,
            body: clone_with_subst(shared, strings, ctx, args_start, body, scratch)?,
        },
        Node::Object(range) => {
            // Object children live in object_fields as (Span, NodeId). Read by
            // index to avoid borrowing the pool across the recursive append.
            let start = range.start as usize;
            let end = start + range.len as usize;
            let mut new_pairs: Vec<(Span, NodeId)> = Vec::with_capacity(end - start);
            for i in start..end {
                let (k, v) = shared.pools.object_fields[i];
                let v2 = clone_with_subst(shared, strings, ctx, args_start, v, scratch)?;
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
        | Node::SynthRef { .. }
        | Node::Cfg { .. } => unreachable!(),
        // Handled by early guards above.
        Node::MacroParam { .. } | Node::SymRef { .. } => unreachable!(),
    };
    Ok(shared.arena.push(new_node, span))
}

fn clone_range(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &ModuleContext,
    args_start: usize,
    range: ChildRange,
    scratch: &mut Vec<NodeId>,
) -> Result<ChildRange, ExpandError> {
    let start = range.start as usize;
    let end = start + range.len as usize;
    stack_scope!(scratch, |saved| {
        for i in start..end {
            let id = shared.pools.children[i];
            let cloned = clone_with_subst(shared, strings, ctx, args_start, id, scratch)?;
            scratch.push(cloned);
        }
        shared
            .pools
            .push_children(&scratch[saved..])
            .ok_or_else(|| ExpandError::without_span(ExpandErrorKind::TooManyChildren))
    })
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
