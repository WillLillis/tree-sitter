use crate::nativedsl::typecheck::types::{
    ScalarTy, TUPLE_MAX_ARITY, TUPLE_MIN_ARITY, TupleSig, TupleSigError,
};
use crate::nativedsl::{
    ContainerKind, DataTy, InnerTy, ModuleTy, NoteMessage, Ty, TypeError, TypeErrorKind,
    ast::{
        ForId, IdentKind, MacroId, MacroKind, ModuleContext, Node, NodeId, ObjectField, Param,
        PrecKind, SharedAst, Span,
    },
    diagnostic::suggest_name,
    resolve::resolve_module_ref,
    typecheck::{Constraint, TypeEnv, TypeResult},
};

/// Function pointer type for element checkers passed to `expect_list`.
type CheckFn = fn(&SharedAst, &ModuleContext, NodeId, &mut TypeEnv) -> TypeResult<()>;

pub(super) fn check_item(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    match shared.arena.get(id) {
        Node::Grammar => {
            // INVARIANT: validate_grammar enforces grammar_config.is_some()
            let config = ctx.grammar_config.as_ref().unwrap();
            for id in [config.extras, config.externals].into_iter().flatten() {
                expect_list(shared, ctx, id, env, expect_rule, Ty::LIST_RULE)?;
            }
            for id in [config.inline, config.supertypes].into_iter().flatten() {
                expect_name_list(shared, ctx, id, env)?;
            }
            if let Some(id) = config.conflicts {
                expect_list(shared, ctx, id, env, expect_name_list, Ty::LIST_LIST_RULE)?;
            }
            if let Some(id) = config.precedences {
                expect_list(
                    shared,
                    ctx,
                    id,
                    env,
                    |shared, ctx, id, env| {
                        expect_list(shared, ctx, id, env, expect_name_or_str, Ty::LIST_RULE)
                    },
                    Ty::LIST_LIST_RULE,
                )?;
            }
            if let Some(id) = config.word {
                expect_name_ref(shared, ctx, id, env)?;
            }
            if let Some(id) = config.start {
                expect_name_ref(shared, ctx, id, env)?;
            }
            if let Some(id) = config.reserved {
                expect_reserved(shared, ctx, id, env)?;
            }
            Ok(())
        }
        Node::Let { .. } => {
            type_of_let(shared, ctx, id, env)?;
            Ok(())
        }
        Node::Macro(macro_id) => {
            let config = shared.pools.get_macro(*macro_id);
            let params = config.params;
            let kind = config.kind;
            let name = config.name;
            let body = config.body;
            check_duplicate_names(
                ctx,
                shared.pools.param_slice(params),
                |p| p.name,
                TypeErrorKind::DuplicateParameter,
            )?;
            for p in shared.pools.param_slice(params) {
                reject_module_type(p.ty, p.name)?;
            }
            match kind {
                MacroKind::Expression(return_ty) => {
                    // No return-type span is kept on the AST, so point at the
                    // macro name for a module_t return type.
                    reject_module_type(return_ty, name)?;
                    type_of(shared, ctx, body, env, Constraint::Exact(return_ty))?;
                }
                MacroKind::RuleSet => check_rule_set_body(shared, ctx, body, env)?,
            }
            Ok(())
        }
        Node::Rule { body, .. } => expect_rule(shared, ctx, *body, env),
        // The template body is checked once via the Macro item; per instance we
        // only check the call args against the macro's params (the arg count was
        // validated at expand, so params and args line up).
        &Node::ExpandedRule(expand_id) => {
            let exp = *shared.pools.get_expansion(expand_id);
            let params = shared.pools.get_macro(exp.macro_id).params;
            for (i, &arg) in shared.pools.child_slice(exp.args).iter().enumerate() {
                let ty = shared.pools.param_slice(params)[i].ty;
                type_of(shared, ctx, arg, env, Constraint::Exact(ty))?;
            }
            Ok(())
        }
        // External decls have no body to typecheck.
        Node::Forward { .. } => Ok(()),
        _ => unreachable!(),
    }
}

/// Type a `let` and any `let`s it transitively references, memoizing each.
///
/// Lets can reference one another in any order, so types are computed lazily
/// rather than strictly in source order. Dependencies are resolved with an
/// explicit stack instead of native recursion, so an arbitrarily long `let`
/// chain cannot overflow the stack; a reference back into a `let` still being
/// resolved is a self-reference cycle. Once a `let`'s dependencies are all
/// memoized, typing its value recurses only through expression nesting, which
/// `type_of` walks iteratively.
fn type_of_let(
    shared: &SharedAst,
    ctx: &ModuleContext,
    let_id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<Ty> {
    if let Some(&ty) = env.vars.get(&let_id) {
        return Ok(ty);
    }
    env.lets_in_progress.insert(let_id);
    let mut stack = vec![let_id];
    while let Some(&cur) = stack.last() {
        let Node::Let { value, .. } = *shared.arena.get(cur) else {
            unreachable!()
        };
        // Resolve one not-yet-typed dependency at a time; revisiting `cur` once
        // it is on the stack would re-find an earlier dependency, so we only
        // type `cur` once the scan reports none remain.
        if let Some((dep, reference)) =
            shared.first_unresolved_let_dep(value, |id| env.vars.contains_key(&id))
        {
            if !env.lets_in_progress.insert(dep) {
                let Node::Let { name, .. } = *shared.arena.get(dep) else {
                    unreachable!()
                };
                return Err(TypeError::with_note(
                    TypeErrorKind::CircularLet(ctx.text(name).to_string()),
                    shared.arena.span(dep),
                    ctx.note(NoteMessage::SelfReferenceHere, shared.arena.span(reference)),
                ));
            }
            stack.push(dep);
            continue;
        }
        let constraint = ctx
            .let_types
            .get(&cur)
            .copied()
            .map_or(Constraint::None, Constraint::Exact);
        let inferred = type_of(shared, ctx, value, env, constraint)?;
        if let Node::Object(range) = *shared.arena.get(value) {
            let fields: Vec<String> = shared
                .pools
                .get_object(range)
                .iter()
                .map(|f| ctx.text(f.name).to_string())
                .collect();
            env.object_fields.insert(cur, fields);
        }
        env.vars.insert(cur, inferred);
        env.lets_in_progress.remove(&cur);
        stack.pop();
    }
    Ok(env.vars[&let_id])
}

/// Typecheck rule decls inside a rule-set macro body at definition time.
/// `MacroParam` refs resolve via their stamped `Ty` so type errors point
/// at the macro definition, not each call site.
fn check_rule_set_body(
    shared: &SharedAst,
    ctx: &ModuleContext,
    body_id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    let Node::RuleSet(range) = shared.arena.get(body_id) else {
        unreachable!()
    };
    for &decl_id in shared.pools.child_slice(*range) {
        match shared.arena.get(decl_id) {
            Node::Rule { body, .. } => expect_rule(shared, ctx, *body, env)?,
            Node::ComputedRule {
                name_expr, body, ..
            } => {
                type_of(shared, ctx, *name_expr, env, Constraint::Exact(Ty::STR))?;
                expect_rule(shared, ctx, *body, env)?;
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

/// Check a list config field. For literal lists, checks each element with
/// `check_elem`. For non-literal expressions, defers to `type_of` with an
/// `Exact(expected)` constraint.
fn expect_list(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    check_elem: CheckFn,
    expected: Ty,
) -> TypeResult<()> {
    if let Node::List(range) = shared.arena.get(id) {
        // Wrap `check_elem` to fit `check_spread_item`'s Ty-returning leaf signature.
        let leaf = |shared: &SharedAst, ctx: &ModuleContext, id: NodeId, env: &mut TypeEnv| {
            check_elem(shared, ctx, id, env).map(|()| Ty::RULE)
        };
        for &child in shared.pools.child_slice(*range) {
            check_spread_item(shared, ctx, child, env, leaf)?;
        }
        return Ok(());
    }
    type_of(shared, ctx, id, env, Constraint::Exact(expected))?;
    Ok(())
}

fn expect_name_list(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    expect_list(shared, ctx, id, env, expect_name_ref, Ty::LIST_RULE)
}

fn expect_name_ref(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    match shared.arena.get(id) {
        Node::Ident(IdentKind::Rule) => Ok(()),
        Node::Ident(IdentKind::Var(_))
        | Node::FieldAccess { .. }
        | Node::GrammarConfig { .. }
        | Node::MacroParam { .. }
        | Node::ForBinding { .. } => {
            type_of(shared, ctx, id, env, Constraint::Strict(Ty::RULE))?;
            Ok(())
        }
        _ => Err(TypeError::new(
            TypeErrorKind::ExpectedRuleName,
            shared.arena.span(id),
        )),
    }
}

/// Inner element for `precedences`: a name ref or a string literal.
fn expect_name_or_str(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    if matches!(
        shared.arena.get(id),
        Node::StringLit | Node::RawStringLit { .. }
    ) {
        return Ok(());
    }
    expect_name_ref(shared, ctx, id, env)
}

/// Check the `reserved` config field. It must be an object literal
/// `{name: [rules]}`: the first set is the default applied to every token, so
/// the set order is significant and can't come from a computed/unordered value.
/// Inheritance is handled by merging the base's reserved at lower, not by
/// re-importing it as a computed value. The word lists themselves may compute.
fn expect_reserved(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    let Node::Object(range) = shared.arena.get(id) else {
        return Err(TypeError::new(
            TypeErrorKind::ReservedMustBeLiteral,
            shared.arena.span(id),
        ));
    };
    for &ObjectField { value: val_id, .. } in shared.pools.get_object(*range) {
        expect_list(shared, ctx, val_id, env, expect_rule, Ty::LIST_RULE)?;
    }
    Ok(())
}

fn expect_rule(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    type_of(shared, ctx, id, env, Constraint::Exact(Ty::RULE))?;
    Ok(())
}

/// One step of the iterative `type_of` traversal. Pushed onto an explicit work
/// stack (held on [`TypeEnv`]) so expression nesting is bounded by the heap,
/// not the native stack.
///
/// `emit` marks whether a step contributes its type to the results stack. Only
/// a *combining* parent (list/object/tuple/append/`::`-receiver, etc.) consumes
/// a child's type; under any other parent a node's type is fixed and the child
/// is type-checked purely for its own constraint, so it emits nothing. Keeping
/// the bulk of nodes (rule/seq/token/field wrappers) off the results stack is
/// what makes the iterative walk competitive with native recursion.
#[derive(Clone, Copy)]
pub(super) enum Work {
    /// Type `id` under `expected`; push its `Ty` iff `emit`.
    Eval {
        id: NodeId,
        expected: Constraint,
        emit: bool,
    },
    /// Type a spread-position item (list element / seq-choice member). A
    /// `Node::For` is flattened into the surrounding sequence; otherwise the
    /// item is typed under `leaf`. Pushes its `Ty` iff `emit`.
    Spread {
        id: NodeId,
        leaf: Constraint,
        emit: bool,
    },
    /// A `for`'s iterable type is on the results stack: match it against the
    /// loop bindings, then type the body as a spread item under `leaf`. Stores
    /// the `Node::For` id (which yields both the for-config and the body) rather
    /// than both separately, so this variant does not widen `Work`.
    ForBindings {
        node: NodeId,
        leaf: Constraint,
        emit: bool,
    },
    /// A combining node's children are typed (their `Ty`s are on the results
    /// stack); re-read the node to fold them into its type, enforce `expected`,
    /// and push the result iff `emit`. Only the node id is stored - how to fold
    /// is recovered from the node in [`combine`] - so `Work` stays small (the
    /// dominant cost of the walk is popping these items off the stack).
    Combine {
        id: NodeId,
        expected: Constraint,
        emit: bool,
    },
}

/// Type a node. `expected` propagates through structural nodes (lists, objects,
/// append) so empty containers can infer from context, and the resulting type
/// is verified to satisfy `expected` before returning. `Constraint::None`
/// always satisfies, effectively skipping enforcement.
///
/// The walk is iterative: an explicit `work` stack drives the traversal and a
/// `results` stack carries the (few) child types a combining parent consumes
/// up to its [`Work::Combine`] step. Native recursion would bound expression
/// nesting by the call stack; the work stack bounds it only by the heap. The
/// one re-entrant call, `type_of_let` for a not-yet-typed `let` reference, is
/// depth-bounded: it pre-resolves a let's dependencies before typing its value,
/// so the nested `type_of` only ever sees already-memoized lets.
fn type_of(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    expected: Constraint,
) -> TypeResult<Ty> {
    // The work and results stacks live on `env` and are shared across all walks,
    // including re-entrant ones (a not-yet-typed `let`, or a `::` call's
    // receiver). Record the stack lengths at entry, push this walk's root, and
    // drive until the work stack returns to that base; on success the walk
    // leaves exactly one new result - the root's type - so no buffer is moved or
    // reallocated per call. On error, unwind both stacks back to the base so an
    // enclosing walk sees them intact.
    let work_base = env.work.len();
    let results_base = env.results.len();
    env.work.push(Work::Eval { id, expected, emit: true });
    match drive(shared, ctx, env, work_base) {
        Ok(()) => Ok(pop_result(&mut env.results)),
        Err(e) => {
            env.work.truncate(work_base);
            env.results.truncate(results_base);
            Err(e)
        }
    }
}

/// Run the iterative traversal until the work stack returns to `work_base`,
/// leaving exactly one new result: the root node's type.
fn drive(
    shared: &SharedAst,
    ctx: &ModuleContext,
    env: &mut TypeEnv,
    work_base: usize,
) -> TypeResult<()> {
    while env.work.len() > work_base {
        match env.work.pop().unwrap() {
            Work::Eval { mut id, mut expected, mut emit } => {
                // Fast-path single-child descent: `eval` returns the next node to
                // descend into directly, so a wrapper chain (token/field/repeat/
                // prec/...) is walked without a work-stack round-trip per level.
                while let Some((next_id, next_expected, next_emit)) =
                    eval(shared, ctx, env, id, expected, emit)?
                {
                    id = next_id;
                    expected = next_expected;
                    emit = next_emit;
                }
            }
            // `push_spread_item` only defers a `Node::For`, so `id` is one here.
            Work::Spread { id, leaf, emit } => enqueue_for(shared, ctx, id, leaf, emit, &mut env.work)?,
            Work::ForBindings { node, leaf, emit } => {
                let iter_ty = pop_result(&mut env.results);
                expect_pat!(Node::For { for_id, body }, *shared.arena.get(node));
                match_for_elem(shared, for_id, iter_ty)?;
                push_spread_item(shared, body, leaf, emit, &mut env.work);
            }
            Work::Combine { id, expected, emit } => combine(shared, ctx, env, id, expected, emit)?,
        }
    }
    Ok(())
}

/// The next node for the fast-path descent loop: `(id, expected, emit)`.
type Descent = Option<(NodeId, Constraint, bool)>;

/// Handle a [`Work::Eval`]. A leaf or fixed-result wrapper enforces `expected`
/// and emits its type inline; a single-child wrapper then returns its child for
/// the caller to descend into without a work-stack round-trip, and a multi-child
/// wrapper queues the rest and returns its first child. A combining node queues
/// a [`Work::Combine`] plus its children (reversed, so they execute left to
/// right and their results land in source order) and returns `None`.
fn eval(
    shared: &SharedAst,
    ctx: &ModuleContext,
    env: &mut TypeEnv,
    id: NodeId,
    expected: Constraint,
    emit: bool,
) -> TypeResult<Descent> {
    let span = shared.arena.span(id);
    Ok(match *shared.arena.get(id) {
        Node::IntLit(_) => {
            leaf(&mut env.results, expected, emit, Ty::INT, span)?;
            None
        }
        Node::StringLit | Node::RawStringLit { .. } => {
            leaf(&mut env.results, expected, emit, Ty::STR, span)?;
            None
        }
        // `ModuleRule`: a cross-module reference resolve already validated as a
        // rule / external / inherited rule, so it is rule-typed.
        Node::Ident(IdentKind::Rule) | Node::Blank | Node::ModuleRule { .. } => {
            leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            None
        }
        Node::ModuleRef { import, module, .. } => {
            let idx = module.expect("module index not set by loading pre-pass");
            let ty = Ty::Module(if import {
                ModuleTy::Import(idx)
            } else {
                ModuleTy::Grammar(idx)
            });
            leaf(&mut env.results, expected, emit, ty, span)?;
            None
        }
        Node::MacroParam { ty, .. } | Node::ForBinding { ty, .. } => {
            leaf(&mut env.results, expected, emit, ty, span)?;
            None
        }
        Node::Ident(IdentKind::Macro(_)) => {
            return Err(TypeError::new(
                TypeErrorKind::MacroUsedAsValue(ctx.text(span).to_string()),
                span,
            ));
        }
        // Type the referenced let on demand (it may be defined later or chain
        // through other lets). type_of_let resolves chains iteratively and
        // reports any self-reference cycle.
        Node::Ident(IdentKind::Var(let_id)) => {
            let ty = type_of_let(shared, ctx, let_id, env)?;
            leaf(&mut env.results, expected, emit, ty, span)?;
            None
        }
        // ---- fixed-result wrappers: enforce + emit now, then descend ----
        // `@<str_expr>` computed-name rule reference: the name must be str_t;
        // it resolves to a rule at lower time (validated there).
        Node::SymRef { expr } => {
            leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            Some((expr, Constraint::Exact(Ty::STR), false))
        }
        Node::Neg(inner) => {
            leaf(&mut env.results, expected, emit, Ty::INT, span)?;
            Some((inner, Constraint::Exact(Ty::INT), false))
        }
        Node::Repeat { inner, .. }
        | Node::Token { inner, .. }
        | Node::Field { content: inner, .. }
        | Node::Reserved { content: inner, .. } => {
            leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            Some((inner, Constraint::Exact(Ty::RULE), false))
        }
        Node::BinOp { lhs, rhs, .. } => {
            leaf(&mut env.results, expected, emit, Ty::INT, span)?;
            env.work.push(Work::Eval { id: rhs, expected: Constraint::Exact(Ty::INT), emit: false });
            Some((lhs, Constraint::Exact(Ty::INT), false))
        }
        Node::Prec { kind, value, content } => {
            let value_expected = if kind == PrecKind::Dynamic {
                Constraint::Exact(Ty::INT)
            } else {
                Constraint::IntOrStr
            };
            leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            env.work.push(Work::Eval { id: content, expected: Constraint::Exact(Ty::RULE), emit: false });
            Some((value, value_expected, false))
        }
        Node::DynRegex { pattern, flags } => {
            leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            if let Some(fid) = flags {
                env.work.push(Work::Eval { id: fid, expected: Constraint::Exact(Ty::STR), emit: false });
            }
            Some((pattern, Constraint::Exact(Ty::STR), false))
        }
        Node::Concat(range) => {
            leaf(&mut env.results, expected, emit, Ty::STR, span)?;
            descend_eval(shared.pools.child_slice(range), Constraint::Exact(Ty::STR), &mut env.work)
        }
        Node::SeqOrChoice { range, .. } => {
            leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            for &member in shared.pools.child_slice(range).iter().rev() {
                push_spread_item(shared, member, Constraint::RuleLike, false, &mut env.work);
            }
            None
        }
        // ---- combining nodes: queue a Combine + children, no descent ----
        // `combine` re-reads the node, so only the children's constraints are set
        // here; the fold and the final enforce happen there.
        Node::Append { left, right } => {
            let (left_evaled, right_evaled) = append_operands(shared, left, right, expected);
            push_combine(&mut env.work, id, expected, emit);
            if right_evaled {
                env.work.push(Work::Eval { id: right, expected, emit: true });
            }
            if left_evaled {
                env.work.push(Work::Eval { id: left, expected, emit: true });
            }
            None
        }
        Node::FieldAccess { obj, .. } => {
            push_combine(&mut env.work, id, expected, emit);
            env.work.push(Work::Eval { id: obj, expected: Constraint::AnyObject, emit: true });
            None
        }
        // `::` member access is valid only on a module bound by import()/inherit(),
        // which resolve rewrites to a direct reference before typecheck; and
        // `module_t` can't reach here through a macro (rejected at parse). So any
        // `QualifiedAccess` surviving to typecheck applies `::` to a non-module
        // value and is always an error. Type the object first to surface any
        // error inside it and to name its type, then report the misuse.
        Node::QualifiedAccess { obj, .. } => {
            push_combine(&mut env.work, id, expected, emit);
            env.work.push(Work::Eval { id: obj, expected: Constraint::None, emit: true });
            None
        }
        Node::GrammarConfig { module, .. } => {
            push_combine(&mut env.work, id, expected, emit);
            env.work.push(Work::Eval { id: module, expected: Constraint::Exact(Ty::ANY_MODULE), emit: true });
            None
        }
        Node::Alias { content, target } => {
            push_combine(&mut env.work, id, expected, emit);
            env.work.push(Work::Eval { id: target, expected: Constraint::None, emit: true });
            env.work.push(Work::Eval { id: content, expected: Constraint::Exact(Ty::RULE), emit: false });
            None
        }
        Node::Object(range) => {
            let fields = shared.pools.get_object(range);
            if fields.is_empty() {
                let ty = empty_container_result(expected, ContainerKind::Object, span)?;
                if emit {
                    env.results.push(ty);
                }
                return Ok(None);
            }
            check_duplicate_names(ctx, fields, |f| f.name, TypeErrorKind::DuplicateObjectKey)?;
            let value_expected = expected.object_value();
            push_combine(&mut env.work, id, expected, emit);
            for f in fields.iter().rev() {
                env.work.push(Work::Eval { id: f.value, expected: value_expected, emit: true });
            }
            None
        }
        Node::List(range) => {
            let items = shared.pools.child_slice(range);
            if items.is_empty() {
                let ty = empty_container_result(expected, ContainerKind::List, span)?;
                if emit {
                    env.results.push(ty);
                }
                return Ok(None);
            }
            let elem_expected = expected.elem();
            push_combine(&mut env.work, id, expected, emit);
            for &item in items.iter().rev() {
                push_spread_item(shared, item, elem_expected, true, &mut env.work);
            }
            None
        }
        Node::Tuple(range) => {
            // Arity is checked first so the common grouping mistake `(expr)` (a
            // would-be 1-tuple) gets the teaching error rather than an element
            // error.
            let items = shared.pools.child_slice(range);
            let n = items.len();
            if !(TUPLE_MIN_ARITY..=TUPLE_MAX_ARITY).contains(&n) {
                return Err(TypeError::new(TypeErrorKind::TupleArityInvalid(n), span));
            }
            push_combine(&mut env.work, id, expected, emit);
            for &item in items.iter().rev() {
                env.work.push(Work::Eval { id: item, expected: Constraint::None, emit: true });
            }
            None
        }
        Node::Call { name, args } => {
            let macro_id = resolve_macro_name(shared, ctx, name, span)?;
            let args = shared.pools.child_slice(args);
            enqueue_macro_call(shared, ctx, macro_id, id, name, args, expected, emit, &mut env.work)?;
            None
        }
        Node::QualifiedCall(range) => {
            let (obj, name, args) = shared.pools.get_qualified_call(range);
            // Type the receiver first (matches the recursive order: a non-module
            // receiver is reported before an unknown macro name). The receiver
            // is a module value - either a resolved reference or an immediately
            // erroring `::` - so a nested type_of stays shallow.
            type_of(shared, ctx, obj, env, Constraint::Exact(Ty::ANY_MODULE))?;
            let Node::Ident(IdentKind::Macro(macro_id)) = *shared.arena.get(name) else {
                let macro_name = ctx.text(shared.arena.span(name));
                return Err(TypeError::new(
                    TypeErrorKind::ImportMacroNotFound(macro_name.to_string()),
                    shared.arena.span(name),
                ));
            };
            enqueue_macro_call(shared, ctx, macro_id, id, name, args, expected, emit, &mut env.work)?;
            None
        }
        // A for-loop in value position is invalid. Validate its internals first
        // (so loop errors surface), then report the position error: for-loops
        // are spliced inline and only appear in seq/choice/list, where
        // `Work::Spread` intercepts them before this arm.
        Node::For { .. } => {
            push_combine(&mut env.work, id, expected, emit);
            enqueue_for(shared, ctx, id, Constraint::Exact(Ty::RULE), false, &mut env.work)?;
            None
        }
        // Top-level items (Grammar/Rule/Let/Macro/...) and Unreachable never
        // appear in expression position - parser dispatch keeps them apart.
        _ => unreachable!(),
    })
}

/// Queue all but the first of `children` (under `c`, emitting nothing) and
/// return the first child for the fast-path descent, or `None` if there are no
/// children.
fn descend_eval(children: &[NodeId], c: Constraint, work: &mut Vec<Work>) -> Descent {
    let (&first, rest) = children.split_first()?;
    for &child in rest.iter().rev() {
        work.push(Work::Eval { id: child, expected: c, emit: false });
    }
    Some((first, c, false))
}

/// Whether each operand of an `a ++ b` append is typed. An empty `[]` operand is
/// skipped (it can't be typed without context) unless a list annotation supplies
/// the element type. Computed identically at queue and combine time.
fn append_operands(
    shared: &SharedAst,
    left: NodeId,
    right: NodeId,
    expected: Constraint,
) -> (bool, bool) {
    let is_empty = |id| {
        matches!(shared.arena.get(id), Node::List(r) if shared.pools.child_slice(*r).is_empty())
    };
    let is_list_anno = matches!(expected, Constraint::Exact(t) if t.is_list());
    (!is_empty(left) || is_list_anno, !is_empty(right) || is_list_anno)
}

/// Handle a [`Work::Combine`]: re-read the node, pop its emitted child results,
/// fold them into the node's type, enforce `expected`, and push it iff `emit`.
fn combine(
    shared: &SharedAst,
    ctx: &ModuleContext,
    env: &mut TypeEnv,
    id: NodeId,
    expected: Constraint,
    emit: bool,
) -> TypeResult<()> {
    let span = shared.arena.span(id);
    let ty = match *shared.arena.get(id) {
        Node::List(range) => {
            let items = shared.pools.child_slice(range);
            let base = env.results.len() - items.len();
            let mut widest = env.results[base];
            for (k, &item) in items.iter().enumerate().skip(1) {
                let got = env.results[base + k];
                widest = widest.widen(got).ok_or_else(|| {
                    TypeError::new(
                        TypeErrorKind::ListElementTypeMismatch { first: widest, got },
                        shared.arena.span(item),
                    )
                })?;
            }
            env.results.truncate(base);
            widest
                .to_list()
                .ok_or_else(|| TypeError::new(TypeErrorKind::InvalidListElement(widest), span))?
        }
        Node::Object(range) => {
            let fields = shared.pools.get_object(range);
            let base = env.results.len() - fields.len();
            let mut widest = env.results[base];
            for (k, field) in fields.iter().enumerate().skip(1) {
                let got = env.results[base + k];
                widest = widest.widen(got).ok_or_else(|| {
                    TypeError::new(
                        TypeErrorKind::ObjectFieldTypeMismatch { first: widest, got },
                        shared.arena.span(field.value),
                    )
                })?;
            }
            env.results.truncate(base);
            let invalid = || {
                TypeError::new(
                    TypeErrorKind::InvalidObjectValue(widest),
                    shared.arena.span(fields[0].value),
                )
            };
            let Ty::Data(d) = widest else {
                return Err(invalid());
            };
            let inner = InnerTy::try_from(d).map_err(|()| invalid())?;
            Ty::Data(DataTy::Object(inner))
        }
        Node::Tuple(range) => {
            let items = shared.pools.child_slice(range);
            let n = items.len();
            let base = env.results.len() - n;
            // `n` is in range (checked at eval), so this fixed buffer holds
            // every element and the signature build cannot fail on arity.
            let mut scalars = [ScalarTy::Rule; TUPLE_MAX_ARITY];
            for (k, &item) in items.iter().enumerate() {
                let got = env.results[base + k];
                let Ty::Data(DataTy::Scalar(s)) = got else {
                    return Err(TypeError::new(
                        TypeErrorKind::TupleElementNotScalar(got),
                        shared.arena.span(item),
                    ));
                };
                scalars[k] = s;
            }
            env.results.truncate(base);
            let sig = TupleSig::new(&scalars[..n]).map_err(|TupleSigError(bad)| {
                TypeError::new(TypeErrorKind::TupleArityInvalid(bad), span)
            })?;
            Ty::Data(DataTy::Tuple(sig))
        }
        Node::Append { left, right } => {
            let (left_evaled, right_evaled) = append_operands(shared, left, right, expected);
            // Operands were pushed left-then-right, so the right result (if any)
            // is on top.
            let r_ty = right_evaled.then(|| pop_result(&mut env.results));
            let l_ty = left_evaled.then(|| pop_result(&mut env.results));
            match (l_ty, r_ty) {
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
                    l.widen(r)
                        .ok_or_else(|| mismatch(l, r, shared.arena.span(right)))?
                }
                // One operand is `[]` (skipped above); the other typed to `t`.
                // The result is `t` if it is list-shaped.
                (Some(t), None) | (None, Some(t)) => {
                    if !t.is_list() {
                        return Err(TypeError::new(TypeErrorKind::AppendRequiresList(t), span));
                    }
                    t
                }
                (None, None) => {
                    return Err(TypeError::new(
                        TypeErrorKind::EmptyContainerNeedsAnnotation(ContainerKind::List),
                        span,
                    ));
                }
            }
        }
        // The call's type is the macro's declared return type; arguments emitted
        // nothing, so there is nothing to pop. The macro name was validated at
        // eval, so the lookup is infallible here.
        Node::Call { name, .. } => macro_call_result(shared, ctx, name, span)?,
        Node::QualifiedCall(range) => {
            let (_, name, _) = shared.pools.get_qualified_call(range);
            macro_call_result(shared, ctx, name, span)?
        }
        Node::GrammarConfig { module, field } => {
            let module_ty = pop_result(&mut env.results);
            if !matches!(module_ty, Ty::Module(ModuleTy::Grammar(_))) {
                let err_kind = TypeErrorKind::GrammarConfigRequiresInherit;
                let arg_span = shared.arena.span(module);
                if let Some(ref_id) = resolve_module_ref(&shared.arena, module)
                    && let Node::ModuleRef { path, .. } = shared.arena.get(ref_id)
                {
                    let path_text = ctx.text(*path).to_string();
                    return Err(TypeError::with_note(
                        err_kind,
                        arg_span,
                        ctx.note(
                            NoteMessage::SwitchImportToInherit(path_text),
                            shared.arena.span(ref_id),
                        ),
                    ));
                }
                return Err(TypeError::new(err_kind, arg_span));
            }
            use crate::nativedsl::ast::ConfigField as C;
            match field {
                C::Language => Ty::STR,
                C::Extras | C::Externals | C::Inline | C::Supertypes => Ty::LIST_RULE,
                C::Conflicts | C::Precedences => Ty::LIST_LIST_RULE,
                C::Word | C::Start => Ty::RULE,
                C::Reserved => Ty::OBJ_LIST_RULE,
                C::Inherits | C::Flags => unreachable!(),
            }
        }
        Node::FieldAccess { obj, field } => {
            let obj_ty = pop_result(&mut env.results);
            let field_name = ctx.text(field);
            // AnyObject auto-enforce at the object's eval guarantees the variant.
            let inner = obj_ty.object_inner().unwrap();
            let field_known = match shared.arena.get(obj) {
                Node::Ident(IdentKind::Var(let_id)) => {
                    // None = not an object literal, can't validate field names
                    env.object_fields
                        .get(let_id)
                        .is_none_or(|fields| fields.iter().any(|f| f == field_name))
                }
                Node::Object(range) => {
                    let fields = shared.pools.get_object(*range);
                    fields.iter().any(|f| ctx.text(f.name) == field_name)
                }
                _ => true, // can't validate dynamically-produced objects
            };
            if !field_known {
                // Collect the valid field names only now that the access fails.
                let available = match shared.arena.get(obj) {
                    Node::Ident(IdentKind::Var(let_id)) => {
                        env.object_fields.get(let_id).cloned().unwrap_or_default()
                    }
                    Node::Object(range) => shared
                        .pools
                        .get_object(*range)
                        .iter()
                        .map(|f| ctx.text(f.name).to_string())
                        .collect(),
                    _ => Vec::new(),
                };
                return Err(TypeError::new(
                    TypeErrorKind::FieldNotFound {
                        field: field_name.to_string(),
                        available,
                    },
                    field,
                ));
            }
            Ty::from(inner)
        }
        Node::Alias { target, .. } => {
            let target_ty = pop_result(&mut env.results);
            let is_valid = matches!(
                shared.arena.get(target),
                Node::Ident(IdentKind::Rule | IdentKind::Var(_))
                    | Node::MacroParam { .. }
                    | Node::ForBinding { .. }
                    | Node::QualifiedAccess { .. }
                    | Node::FieldAccess { .. }
                    | Node::GrammarConfig { .. }
            ) || target_ty == Ty::STR;
            if !target_ty.is_rule_like() || !is_valid {
                return Err(TypeError::new(
                    TypeErrorKind::InvalidAliasTarget(target_ty),
                    shared.arena.span(target),
                ));
            }
            Ty::RULE
        }
        Node::QualifiedAccess { .. } => {
            let obj_ty = pop_result(&mut env.results);
            return Err(TypeError::new(
                TypeErrorKind::MemberAccessRequiresModule(obj_ty),
                span,
            ));
        }
        Node::For { .. } => return Err(TypeError::new(TypeErrorKind::BoundForLoop, span)),
        _ => unreachable!(),
    };
    enforce(expected, ty, span)?;
    if emit {
        env.results.push(ty);
    }
    Ok(())
}

/// The result type of a macro call: the macro's declared return type, or an
/// error for a rule-set macro used in expression position. `name` is the call's
/// macro identifier, already validated to resolve to a macro.
fn macro_call_result(
    shared: &SharedAst,
    ctx: &ModuleContext,
    name: NodeId,
    span: Span,
) -> TypeResult<Ty> {
    let Node::Ident(IdentKind::Macro(macro_id)) = *shared.arena.get(name) else {
        unreachable!()
    };
    match shared.pools.get_macro(macro_id).kind {
        MacroKind::Expression(return_ty) => Ok(return_ty),
        MacroKind::RuleSet => Err(TypeError::new(
            TypeErrorKind::RuleSetMacroInExpressionContext(
                ctx.text(shared.arena.span(name)).to_string(),
            ),
            span,
        )),
    }
}

/// Queue a spread-position item (list element / seq-choice member / loop body).
/// Only a `Node::For` needs the deferred [`Work::Spread`] step - it expands into
/// the surrounding sequence once its iterable is typed - so a plain item is
/// queued as a direct [`Work::Eval`], saving a work-stack round-trip on the
/// common case. The For-ness check reads the node, which `eval` would read
/// anyway, so it adds no extra arena access overall.
fn push_spread_item(
    shared: &SharedAst,
    id: NodeId,
    leaf: Constraint,
    emit: bool,
    work: &mut Vec<Work>,
) {
    if matches!(shared.arena.get(id), Node::For { .. }) {
        work.push(Work::Spread { id, leaf, emit });
    } else {
        work.push(Work::Eval { id, expected: leaf, emit });
    }
}

/// Validate a for-loop's bindings, then schedule typing its iterable and body.
/// An empty literal iterable has no elements to match, so the bindings are only
/// validated and the body is typed directly.
fn enqueue_for(
    shared: &SharedAst,
    ctx: &ModuleContext,
    node: NodeId,
    leaf: Constraint,
    emit: bool,
    work: &mut Vec<Work>,
) -> TypeResult<()> {
    expect_pat!(Node::For { for_id, body }, *shared.arena.get(node));
    let config = shared.pools.get_for(for_id);
    let bindings = shared.pools.param_slice(config.bindings);
    let iterable = config.iterable;
    validate_for_bindings(ctx, bindings, shared.arena.span(body))?;
    if let Node::List(range) = shared.arena.get(iterable)
        && shared.pools.child_slice(*range).is_empty()
    {
        push_spread_item(shared, body, leaf, emit, work);
        return Ok(());
    }
    work.push(Work::ForBindings { node, leaf, emit });
    work.push(Work::Eval { id: iterable, expected: Constraint::None, emit: true });
    Ok(())
}

/// Resolve a macro call's name to its id, or report an undefined macro (with a
/// "did you mean" suggestion when a close name exists).
fn resolve_macro_name(
    shared: &SharedAst,
    ctx: &ModuleContext,
    name: NodeId,
    span: Span,
) -> TypeResult<MacroId> {
    if let Node::Ident(IdentKind::Macro(macro_id)) = *shared.arena.get(name) {
        return Ok(macro_id);
    }
    let macro_name = ctx.text(shared.arena.span(name));
    let kind = TypeErrorKind::UndefinedMacro(macro_name.to_string());
    let macros = ctx.root_items.iter().filter_map(|&id| {
        if let Node::Macro(mid) = shared.arena.get(id) {
            Some(ctx.text(shared.pools.get_macro(*mid).name))
        } else {
            None
        }
    });
    if let Some(suggestion) = suggest_name(macro_name, macros) {
        return Err(TypeError::with_note(
            kind,
            span,
            ctx.note(NoteMessage::DidYouMean(suggestion.to_string()), span),
        ));
    }
    Err(TypeError::new(kind, span))
}

/// Validate a macro call's arg count, then schedule a [`Work::Combine`] for the
/// call node (which yields the macro's return type) and the arguments (each
/// checked against the matching parameter type; they emit nothing). A `::`
/// call's receiver is typed by the caller, before this, so a non-module receiver
/// is reported first.
#[allow(clippy::too_many_arguments)]
fn enqueue_macro_call(
    shared: &SharedAst,
    ctx: &ModuleContext,
    macro_id: MacroId,
    node: NodeId,
    name: NodeId,
    args: &[NodeId],
    expected: Constraint,
    emit: bool,
    work: &mut Vec<Work>,
) -> TypeResult<()> {
    let params = shared.pools.get_macro(macro_id).params;
    if args.len() != params.len as usize {
        return Err(TypeError::new(
            TypeErrorKind::ArgCountMismatch {
                macro_name: ctx.text(shared.arena.span(name)).to_string(),
                expected: params.len as usize,
                got: args.len(),
            },
            shared.arena.span(node),
        ));
    }
    push_combine(work, node, expected, emit);
    // Push args reversed so they execute left-to-right (matching the recursive
    // order: the first ill-typed argument is reported first).
    for (i, &arg) in args.iter().enumerate().rev() {
        let ty = shared.pools.param_slice(params)[i].ty;
        work.push(Work::Eval { id: arg, expected: Constraint::Exact(ty), emit: false });
    }
    Ok(())
}

/// Enforce a constraint on a computed type, mirroring the tail check every
/// `type_of` node runs. `Constraint::None` always satisfies.
fn enforce(expected: Constraint, ty: Ty, span: Span) -> TypeResult<()> {
    if matches!(expected, Constraint::None) {
        return Ok(());
    }
    if !expected.satisfies(ty) {
        let kind = match expected {
            Constraint::Exact(t) | Constraint::Strict(t) => TypeErrorKind::TypeMismatch {
                expected: t,
                got: ty,
            },
            _ => TypeErrorKind::ConstraintMismatch { expected, got: ty },
        };
        return Err(TypeError::new(kind, span));
    }
    Ok(())
}

/// Enforce `expected` on a leaf/fixed-result node's type and emit it iff `emit`.
fn leaf(results: &mut Vec<Ty>, expected: Constraint, emit: bool, ty: Ty, span: Span) -> TypeResult<()> {
    enforce(expected, ty, span)?;
    if emit {
        results.push(ty);
    }
    Ok(())
}

/// Pop the type a child emitted onto the results stack. A combine step is only
/// queued behind its emitting children, so the walk's emit discipline
/// guarantees the result is present - its absence is a bug in that discipline,
/// not a runtime condition. Centralized so the invariant is asserted in one
/// place rather than at every consuming site.
#[inline]
fn pop_result(results: &mut Vec<Ty>) -> Ty {
    results.pop().unwrap()
}

/// Schedule a combining node's `Combine` step after its children.
fn push_combine(work: &mut Vec<Work>, id: NodeId, expected: Constraint, emit: bool) {
    work.push(Work::Combine { id, expected, emit });
}

/// Type-check an item in a spread-accepting position (list element, seq/choice
/// member, etc) for a config list. If the item is a `Node::For`, recursively
/// flatMap into the surrounding sequence by re-applying `leaf` to the inner
/// body. Otherwise dispatch to `leaf` directly. Used by `expect_list`, whose
/// element checkers are bespoke (e.g. `expect_name_ref`); expression-position
/// spreads go through [`Work::Spread`] instead.
///
/// Returns the type of the deepest non-For body so the caller can widen against
/// sibling items.
fn check_spread_item<Leaf>(
    shared: &SharedAst,
    ctx: &ModuleContext,
    item: NodeId,
    env: &mut TypeEnv,
    leaf: Leaf,
) -> TypeResult<Ty>
where
    Leaf: Fn(&SharedAst, &ModuleContext, NodeId, &mut TypeEnv) -> TypeResult<Ty> + Copy,
{
    if let &Node::For { for_id, body } = shared.arena.get(item) {
        check_for_expr(shared, ctx, for_id, body, env, |body, env| {
            check_spread_item(shared, ctx, body, env, leaf)
        })
    } else {
        leaf(shared, ctx, item, env)
    }
}

/// Validate a for-loop's bindings independently of its iterable: they must be
/// present, uniquely named, and (when destructuring) scalar-typed. Shared by
/// the iterative expression walk and the recursive config-list walk.
fn validate_for_bindings(
    ctx: &ModuleContext,
    bindings: &[Param],
    body_span: Span,
) -> TypeResult<()> {
    if bindings.is_empty() {
        return Err(TypeError::new(TypeErrorKind::EmptyForBindings, body_span));
    }
    check_duplicate_names(ctx, bindings, |p| p.name, TypeErrorKind::DuplicateBinding)?;
    // Two-or-more bindings destructure a tuple, whose components are scalars.
    // Validate the binding annotations up front so a non-scalar binding points
    // at the binding itself, independent of the iterable (including empty ones).
    if bindings.len() >= 2 {
        for param in bindings {
            if !matches!(param.ty, Ty::Data(DataTy::Scalar(_))) {
                return Err(TypeError::new(
                    TypeErrorKind::TupleElementNotScalar(param.ty),
                    param.name,
                ));
            }
        }
    }
    Ok(())
}

/// Match a for-loop's iterable type against its bindings. Every iterable source
/// types to a concrete list whose element is matched against the bindings
/// uniformly: a single binding takes the element directly, and two or more
/// destructure a tuple element-wise.
fn match_for_elem(shared: &SharedAst, for_id: ForId, iter_ty: Ty) -> TypeResult<()> {
    let config = shared.pools.get_for(for_id);
    let bindings = shared.pools.param_slice(config.bindings);
    let iterable = config.iterable;
    let Some(elem_ty) = iter_ty.list_elem() else {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresList(iter_ty),
            shared.arena.span(iterable),
        ));
    };
    if bindings.len() == 1 {
        let Param { name, ty: declared } = bindings[0];
        if !elem_ty.is_compatible(declared) {
            return Err(mismatch(declared, elem_ty, name));
        }
    } else {
        let Ty::Data(DataTy::Tuple(sig)) = elem_ty else {
            return Err(TypeError::new(
                TypeErrorKind::ForRequiresTuples,
                shared.arena.span(iterable),
            ));
        };
        if sig.arity() != bindings.len() {
            return Err(TypeError::new(
                TypeErrorKind::ForBindingCountMismatch {
                    bindings: bindings.len(),
                    tuple_elements: sig.arity(),
                },
                shared.arena.span(iterable),
            ));
        }
        for (i, param) in bindings.iter().enumerate() {
            let elem_scalar = Ty::Data(DataTy::Scalar(sig.elem(i)));
            if !elem_scalar.is_compatible(param.ty) {
                return Err(mismatch(param.ty, elem_scalar, param.name));
            }
        }
    }
    Ok(())
}

/// Type-check a config-list for-loop (recursive; config lists are fixed-depth).
/// The body is typed via `check_body`, which re-enters `check_spread_item` so
/// nested loops flatten. Expression-position loops use [`enqueue_for`] /
/// [`match_for_elem`] on the work stack instead; the binding/element checks are
/// shared.
fn check_for_expr<CheckBody>(
    shared: &SharedAst,
    ctx: &ModuleContext,
    for_idx: ForId,
    body: NodeId,
    env: &mut TypeEnv,
    check_body: CheckBody,
) -> TypeResult<Ty>
where
    CheckBody: FnOnce(NodeId, &mut TypeEnv) -> TypeResult<Ty>,
{
    let config = shared.pools.get_for(for_idx);
    let bindings = shared.pools.param_slice(config.bindings);
    let iterable = config.iterable;
    validate_for_bindings(ctx, bindings, shared.arena.span(body))?;

    // Empty literal iterable: no elements to match against the bindings. The
    // binding annotations define the element type and lowering iterates zero
    // times, so nothing more is checked here.
    if let Node::List(range) = shared.arena.get(iterable)
        && shared.pools.child_slice(*range).is_empty()
    {
        return check_body(body, env);
    }

    // Constraint::None (not an auto-enforced list constraint) so a non-list
    // iterable surfaces as ForRequiresList rather than a generic mismatch.
    let iter_ty = type_of(shared, ctx, iterable, env, Constraint::None)?;
    match_for_elem(shared, for_idx, iter_ty)?;
    check_body(body, env)
}

/// Resolve an empty list/object literal against an outer constraint. Returns
/// the constraint's exact type if shape-compatible, else an error.
fn empty_container_result(expected: Constraint, kind: ContainerKind, span: Span) -> TypeResult<Ty> {
    let matches_kind = |ty: Ty| match kind {
        ContainerKind::List => ty.is_list(),
        ContainerKind::Object => ty.is_object(),
    };
    match expected {
        Constraint::Exact(ty) if matches_kind(ty) => Ok(ty),
        Constraint::Exact(declared) => Err(TypeError::new(
            TypeErrorKind::EmptyContainerAnnotationMismatch { declared, kind },
            span,
        )),
        _ => Err(TypeError::new(
            TypeErrorKind::EmptyContainerNeedsAnnotation(kind),
            span,
        )),
    }
}

const fn mismatch(expected: Ty, got: Ty, span: Span) -> TypeError {
    TypeError::new(TypeErrorKind::TypeMismatch { expected, got }, span)
}

/// Reject `module_t` in a position where a module value can't be used (macro
/// parameter / return type). A module is only usable where it is bound by
/// `import()`/`inherit()`, which resolve rewrites; see [`MemberAccessRequiresModule`].
const fn reject_module_type(ty: Ty, span: Span) -> TypeResult<()> {
    if matches!(ty, Ty::Module(_)) {
        return Err(TypeError::new(TypeErrorKind::ModuleTypeNotAllowed, span));
    }
    Ok(())
}

/// Reject duplicate names among `items`, reporting the later one with a
/// "first defined here" note. Used for macro parameters, for-loop bindings, and
/// object keys, whose name spans are preserved on the AST.
fn check_duplicate_names<T>(
    ctx: &ModuleContext,
    items: &[T],
    name_of: impl Fn(&T) -> Span,
    make_kind: impl Fn(String) -> TypeErrorKind,
) -> TypeResult<()> {
    for i in 1..items.len() {
        let span = name_of(&items[i]);
        let name = ctx.text(span);
        for prev in &items[..i] {
            if ctx.text(name_of(prev)) == name {
                return Err(TypeError::with_note(
                    make_kind(name.to_string()),
                    span,
                    ctx.note(NoteMessage::FirstDefinedHere, name_of(prev)),
                ));
            }
        }
    }
    Ok(())
}
