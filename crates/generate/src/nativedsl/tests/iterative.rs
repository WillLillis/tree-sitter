//! Regression tests for the iterative (non-recursive) passes: each pass must
//! handle an arbitrarily deep AST without overflowing the native stack. The
//! parser caps source nesting at `MAX_PARSE_DEPTH`, so the deep AST is built
//! directly via the arena to exercise a single pass past that bound.

use crate::nativedsl::ast::{GrammarConfig, ModuleContext, Node, SharedAst, Span};
use crate::nativedsl::resolve::resolve;
use crate::nativedsl::typecheck::{self, TypeEnv};
use crate::nativedsl::{DocumentMap, ModuleId};
use crate::rules::RulePool;

/// A `program` rule whose body is an `n`-deep `token(token(...blank...))` chain,
/// plus a minimal grammar `ModuleContext` (config carries only `language`), built
/// directly (no parser). `token(rule_t)` is `rule_t` and `blank` is `rule_t`, so
/// the chain resolves and type-checks. The builder serves both deep tests.
fn deep_token_chain(n: usize) -> (SharedAst, ModuleContext, RulePool, DocumentMap) {
    let mut pool = RulePool::default();
    let source = String::from("program");
    let span = Span::from_usize(0, source.len());
    let name = pool.intern(&source);
    let mut shared = SharedAst::new(n + 4);
    let node_start = shared.arena.next_id();
    let mut inner = shared.arena.push(Node::Blank, span);
    for _ in 0..n {
        inner = shared.arena.push(
            Node::Token {
                immediate: false,
                inner,
            },
            span,
        );
    }
    let rule = shared.arena.push(
        Node::Rule {
            is_override: false,
            name,
            body: inner,
        },
        span,
    );
    let mut documents = DocumentMap::default();
    let document = documents.insert(std::path::PathBuf::from("deep.tsg"), source);
    let mut ctx = ModuleContext::new(document, 1, node_start);
    ctx.grammar_config = Some(GrammarConfig {
        language: Some(pool.intern("deep")),
        ..GrammarConfig::default()
    });
    ctx.root_items.push(rule);
    ctx.set_node_end(shared.arena.next_id());
    (shared, ctx, pool, documents)
}

#[test]
fn resolve_deep_nesting_does_not_overflow() {
    // 50k-deep nesting once overflowed resolve_expr's recursion; the iterative
    // walk handles it.
    let (mut shared, ctx, pool, _) = deep_token_chain(50_000);
    resolve(&mut shared, &ctx, &pool, &[], ModuleId::from(0), None, &[]).unwrap();
}

#[test]
fn typecheck_deep_nesting_does_not_overflow() {
    // The same `token(token(...blank))` chain type-checks because `token(rule_t)`
    // is `rule_t`. A 50k-deep chain would overflow a recursive `type_of`, but the
    // iterative walk handles it.
    let (shared, ctx, pool, documents) = deep_token_chain(50_000);
    let mut env = TypeEnv::default();
    let source = documents.document(ctx.document).text();
    typecheck::check(&shared, &ctx, source, &mut env, pool.strs()).unwrap();
}
