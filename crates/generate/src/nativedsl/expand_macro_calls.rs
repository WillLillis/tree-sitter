//! Inlines top-level rule-set macro invocations between `apply_cfg` and
//! `resolve`.
//!
//! Each `Node::Call` at item position becomes one `Node::ExpandedRule` per decl
//! in the macro's body. Each `ExpandedRule` records the shared body and the call's
//! args in an [`Expansion`], and lower binds the args before lowering the body
//! (the expression-macro mechanism). Only the rule *name* is materialized here,
//! since resolve needs it.
//!
//! Only local macros are visible at item position; qualified calls aren't
//! supported (the parser rejects `@mod::foo(...)`).

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    ExpandError, NoteMessage,
    ast::{Expansion, MacroKind, ModuleContext, Node, NodeId, SharedAst, Span, Spanned},
    lexer::is_ident_str,
    string_pool::{Str, StringPool},
};

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
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
}

pub fn expand_macro_calls(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &mut ModuleContext,
) -> Result<(), ExpandError> {
    // Walk only the original indices. Each Call writes its first expanded
    // rule into its own slot; the rest are pushed to the end. Parser's
    // EmptyRuleSetMacroBody guarantees the slot is always filled.
    let original_len = ctx.root_items.len();
    for i in 0..original_len {
        let id = ctx.root_items[i];
        if matches!(shared.arena.get(id), Node::Call { .. }) {
            expand_one_call(shared, strings, ctx, id, i)?;
        }
    }
    Ok(())
}

fn expand_one_call(
    shared: &mut SharedAst,
    strings: &mut StringPool,
    ctx: &mut ModuleContext,
    call_id: NodeId,
    slot: usize,
) -> Result<(), ExpandError> {
    let Node::Call { name, args } = *shared.arena.get(call_id) else {
        unreachable!()
    };
    let name_span = shared.arena.span(name);
    let name_text = ctx.text(name_span);
    let Some(macro_id) = ctx.macro_index.get(name_text).copied() else {
        let mut err = ExpandError::new(
            ExpandErrorKind::UnknownMacro(name_text.to_string()),
            name_span,
        );
        // If the name was a cfg-dropped macro, say so - parity with the
        // GatedByDisabledCfg enrichment a cfg-dropped expression-macro
        // reference gets at resolve.
        if let Some(&cfg_node) = ctx.cfg_dropped.get(name_text)
            && let Node::Cfg {
                name: flag_span, ..
            } = *shared.arena.get(cfg_node)
        {
            err.add_note(ctx.note(
                NoteMessage::GatedByDisabledCfg(ctx.text(flag_span).to_owned()),
                shared.arena.span(cfg_node),
            ));
        }
        return Err(err);
    };
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
    // Read template decls by absolute index. Each instance records the shared
    // template body + the call's args, and lower binds the args on the macro-arg
    // stack before lowering the body (the same mechanism expression macros use).
    // Only the rule *name* is materialized now, since resolve needs it to
    // register the generated rule.
    let call_span = shared.arena.span(call_id);
    let args_start = args.start as usize;
    let rule_start = rule_range.start as usize;
    let rule_end = rule_start + rule_range.len as usize;
    for (offset, i) in (rule_start..rule_end).enumerate() {
        let rule_id = shared.pools.children[i];
        let (is_override, name, body) = match *shared.arena.get(rule_id) {
            Node::Rule {
                is_override,
                name,
                body,
            } => (is_override, strings.intern_owned(ctx.text(name)), body),
            Node::ComputedRule {
                is_override,
                name_expr,
                body,
            } => {
                let name_span = shared.arena.span(name_expr);
                let name = eval_name(shared, strings, ctx, args_start, name_expr, name_span)?;
                (is_override, name, body)
            }
            // Parser only places Rule/ComputedRule in a RuleSet body.
            _ => unreachable!(),
        };
        let expand_id = shared.pools.push_expansion(Expansion {
            is_override,
            name,
            macro_id,
            body,
            args,
        });
        let expanded = shared.arena.push(Node::ExpandedRule(expand_id), call_span);
        if offset == 0 {
            ctx.root_items[slot] = expanded;
        } else {
            ctx.root_items.push(expanded);
        }
    }
    // Evaluate this macro's computed-name references under the call's args and
    // record them for resolve to validate.
    for &sym_ref in &shared.pools.get_macro(macro_id).sym_refs {
        expect_pat!(Node::SymRef { expr }, *shared.arena.get(sym_ref));
        let span = shared.arena.span(sym_ref);
        let name = eval_name(shared, strings, ctx, args_start, expr, span)?;
        ctx.computed_refs.push(Spanned::new(name, span));
    }
    Ok(())
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

/// Build a computed rule name (`@<expr>`) at expand time, before resolve - so
/// only the early-evaluable subset works: string literals, the macro's params,
/// and `concat` of those. A `let` can't be read here (lets run at lower); pass
/// it as a macro param instead.
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
            out.push_str(ctx.text(span.strip_raw(hash_count)));
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
