//! Regression tests for the iterative (non-recursive) passes: each pass must
//! handle an arbitrarily deep AST without overflowing the native stack. The
//! parser caps source nesting at `MAX_PARSE_DEPTH`, so the deep AST is built
//! directly via the arena to exercise a single pass past that bound.

use rustc_hash::FxHashMap;

use crate::nativedsl::LoweringState;
use crate::nativedsl::ast::{GrammarConfig, ModuleContext, Node, SharedAst, Span};
use crate::nativedsl::lower::{LowerErrorKind, lower_with_base};
use crate::nativedsl::resolve::resolve;
use crate::nativedsl::string_pool::StringPool;
use crate::nativedsl::typecheck::{self, TypeEnv};

/// A `program` rule whose body is an `n`-deep `token(token(...blank...))` chain,
/// plus a minimal grammar `ModuleContext` (config carries only `language`), built
/// directly (no parser). `token(rule_t)` is `rule_t` and `blank` is `rule_t`, so
/// the chain resolves, type-checks, and lowers - the one builder serves all three
/// deep tests.
fn deep_token_chain(n: usize) -> (SharedAst, ModuleContext) {
    let source = String::from("program");
    let name = Span::from_usize(0, source.len());
    let mut shared = SharedAst::new(n + 4);
    let mut inner = shared.arena.push(Node::Blank, name);
    for _ in 0..n {
        inner = shared.arena.push(
            Node::Token {
                immediate: false,
                inner,
            },
            name,
        );
    }
    let rule = shared.arena.push(
        Node::Rule {
            is_override: false,
            name,
            body: inner,
        },
        name,
    );
    let ctx = ModuleContext {
        source,
        path: std::path::PathBuf::from("deep.tsg"),
        grammar_config: Some(GrammarConfig {
            language: Some(String::from("deep")),
            ..GrammarConfig::default()
        }),
        root_items: vec![rule],
        module_refs: Vec::new(),
        has_cfg: false,
        has_forward_decls: false,
        cfg_declared: FxHashMap::default(),
        cfg_dropped: FxHashMap::default(),
        computed_refs: Vec::new(),
        let_types: FxHashMap::default(),
        node_range: 0..0,
    };
    (shared, ctx)
}

#[test]
fn resolve_deep_nesting_does_not_overflow() {
    // 50k-deep nesting once overflowed resolve_expr's recursion; the iterative
    // walk handles it.
    let (mut shared, ctx) = deep_token_chain(50_000);
    let strings = StringPool::default();
    resolve(&mut shared, &ctx, &strings, &[], None, &[]).unwrap();
}

#[test]
fn typecheck_deep_nesting_does_not_overflow() {
    // The same `token(token(...blank))` chain type-checks (token(rule_t) is
    // rule_t); 50k deep would overflow a recursive `type_of`, but the iterative
    // walk handles it.
    let (mut shared, ctx) = deep_token_chain(50_000);
    let mut env = TypeEnv::default();
    typecheck::check(&mut shared, &ctx, &mut env).unwrap();
}

#[test]
fn lower_deep_nesting_errors_cleanly() {
    // The 50k-deep `token(token(...blank))` chain once overflowed the native stack
    // in eval (per-level recursion). The iterative engine builds the IR on the heap
    // work stack without overflow; `build_rule` then walks that IR iteratively too,
    // and the MAX_RULE_DEPTH bound (protecting the recursive shared backend) turns
    // this into a clean error rather than a crash.
    let (shared, ctx) = deep_token_chain(50_000);
    let mut state = LoweringState::default();
    let mut strings = StringPool::default();
    let err = lower_with_base(&mut state, &mut strings, &shared, &[], &ctx, &[]).unwrap_err();
    assert!(
        matches!(err.kind, LowerErrorKind::RuleNestingTooDeep),
        "expected RuleNestingTooDeep, got {:?}",
        err.kind
    );
}
