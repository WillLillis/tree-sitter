use crate::nativedsl::{
    ContainerKind, DataTy, InnerTy, ModuleTy, NoteMessage, Ty, TypeError, TypeErrorKind,
    ast::{
        ForId, IdentKind, MacroId, MacroKind, ModuleContext, Node, NodeId, ObjectField, Param,
        PrecKind, SharedAst, Span,
    },
    diagnostic::suggest_name,
    resolve::resolve_module_ref,
    typecheck::{
        Constraint, TypeEnv, TypeResult,
        types::{ScalarTy, TUPLE_MAX_ARITY, TUPLE_MIN_ARITY, TupleSig, TupleSigError},
    },
};

type CheckFn = fn(&SharedAst, &ModuleContext, NodeId, &mut TypeEnv) -> TypeResult<()>;

pub(super) fn check_item(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    match shared.arena.get(id) {
        Node::Grammar => {
            // INVARIANT: validate_grammar enforces `grammar_config.is_some()`
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
            if let Some(id) = config.flags {
                expect_pat!(Node::Object(range), shared.arena.get(id));
                check_duplicate_names(
                    ctx,
                    shared.pools.get_object(*range),
                    |f| f.name,
                    TypeErrorKind::DuplicateObjectKey,
                )?;
            }
            Ok(())
        }
        Node::Let { .. } => {
            _ = type_of_let(shared, ctx, id, env)?;
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
                    reject_module_type(return_ty, name)?;
                    type_of(shared, ctx, body, env, Constraint::Exact(return_ty))?;
                }
                MacroKind::RuleSet => check_rule_set_body(shared, ctx, body, env)?,
            }
            Ok(())
        }
        Node::Rule { body, .. } => expect_rule(shared, ctx, *body, env),
        &Node::ExpandedRule(expand_id) => {
            let exp = *shared.pools.get_expansion(expand_id);
            let params = shared.pools.get_macro(exp.macro_id).params;
            for (i, &arg) in shared.pools.child_slice(exp.args).iter().enumerate() {
                let ty = shared.pools.param_slice(params)[i].ty;
                type_of(shared, ctx, arg, env, Constraint::Exact(ty))?;
            }
            Ok(())
        }
        Node::Forward { .. } => Ok(()),
        // Dispatcher only calls this for top-level item nodes.
        _ => unreachable!(),
    }
}

/// Type a `let` and any transitively referenced `let`s, memoizing each.
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
            // Only let ids are pushed onto this stack.
            unreachable!()
        };
        if let Some((dep, reference)) =
            shared.first_unresolved_let_dep(value, |id| env.vars.contains_key(&id))
        {
            if !env.lets_in_progress.insert(dep) {
                let Node::Let { name, .. } = *shared.arena.get(dep) else {
                    // `first_unresolved_let_dep` only reports let dependencies.
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

/// Check a list config field.
fn expect_list(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    check_elem: CheckFn,
    expected: Ty,
) -> TypeResult<()> {
    if let Node::List(range) = shared.arena.get(id) {
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
        // A qualified ref names a rule but is not a name here; suggest the
        // bare name, which is in scope.
        Node::ModuleRule { .. } => {
            let span = shared.arena.span(id);
            Err(TypeError::with_note(
                TypeErrorKind::ExpectedRuleName,
                span,
                ctx.note(NoteMessage::UseBareName(bare_name(ctx, span)), span),
            ))
        }
        _ => Err(TypeError::new(
            TypeErrorKind::ExpectedRuleName,
            shared.arena.span(id),
        )),
    }
}

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

/// Check the `reserved` config field.
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
    let fields = shared.pools.get_object(*range);
    check_duplicate_names(ctx, fields, |f| f.name, TypeErrorKind::DuplicateObjectKey)?;
    for &ObjectField { value: val_id, .. } in fields {
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

#[derive(Clone, Copy)]
pub(super) enum Work {
    /// Type `id` under `expected`.
    Eval {
        id: NodeId,
        expected: Constraint,
        emit: bool,
    },
    /// Type a spread-position item.
    Spread {
        id: NodeId,
        leaf: Constraint,
        emit: bool,
    },
    /// Match a `for`'s iterable type, then type its body under `leaf`.
    ForBindings {
        node: NodeId,
        leaf: Constraint,
        emit: bool,
    },
    /// Fold a combining node's emitted child types.
    Combine {
        id: NodeId,
        expected: Constraint,
        emit: bool,
    },
}

const _: () = assert!(std::mem::size_of::<Work>() == 16);

/// Type a node.
fn type_of(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    expected: Constraint,
) -> TypeResult<Ty> {
    let work_base = env.work.len();
    let results_base = env.results.len();
    push_eval(&mut env.work, id, expected, true);
    match drive(shared, ctx, env, work_base) {
        Ok(()) => Ok(pop_result(&mut env.results)),
        Err(e) => {
            env.work.truncate(work_base);
            env.results.truncate(results_base);
            Err(e)
        }
    }
}

/// Run the iterative traversal until the work stack returns to `work_base`.
fn drive(
    shared: &SharedAst,
    ctx: &ModuleContext,
    env: &mut TypeEnv,
    work_base: usize,
) -> TypeResult<()> {
    while env.work.len() > work_base {
        match env.work.pop().unwrap() {
            Work::Eval {
                mut id,
                mut expected,
                mut emit,
            } => {
                while let Some((next_id, next_expected, next_emit)) =
                    eval(shared, ctx, env, id, expected, emit)?
                {
                    id = next_id;
                    expected = next_expected;
                    emit = next_emit;
                }
            }
            Work::Spread { id, leaf, emit } => {
                enqueue_for(shared, ctx, id, leaf, emit, &mut env.work)?;
            }
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

type Descent = Option<(NodeId, Constraint, bool)>;

/// Handle a [`Work::Eval`].
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
            enforce_leaf(&mut env.results, expected, emit, Ty::INT, span)?;
            None
        }
        Node::StringLit | Node::RawStringLit { .. } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::STR, span)?;
            None
        }
        Node::Ident(IdentKind::Rule) | Node::Blank | Node::ModuleRule { .. } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            None
        }
        Node::ModuleRef { import, module, .. } => {
            let idx = module.expect("module index not set by loading pre-pass");
            let ty = Ty::Module(if import {
                ModuleTy::Import(idx)
            } else {
                ModuleTy::Grammar(idx)
            });
            enforce_leaf(&mut env.results, expected, emit, ty, span)?;
            None
        }
        Node::MacroParam { ty, .. } | Node::ForBinding { ty, .. } => {
            enforce_leaf(&mut env.results, expected, emit, ty, span)?;
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
            enforce_leaf(&mut env.results, expected, emit, ty, span)?;
            None
        }
        Node::SymRef { expr } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            Some((expr, Constraint::Exact(Ty::STR), false))
        }
        Node::Neg(inner) => {
            enforce_leaf(&mut env.results, expected, emit, Ty::INT, span)?;
            Some((inner, Constraint::Exact(Ty::INT), false))
        }
        Node::Repeat { inner, .. }
        | Node::Token { inner, .. }
        | Node::Field { content: inner, .. }
        | Node::Reserved { content: inner, .. } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            Some((inner, Constraint::Exact(Ty::RULE), false))
        }
        Node::BinOp { lhs, rhs, .. } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::INT, span)?;
            push_eval(&mut env.work, rhs, Constraint::Exact(Ty::INT), false);
            Some((lhs, Constraint::Exact(Ty::INT), false))
        }
        Node::Prec {
            kind,
            value,
            content,
        } => {
            let value_expected = if kind == PrecKind::Dynamic {
                Constraint::Exact(Ty::INT)
            } else {
                Constraint::IntOrStr
            };
            enforce_leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            push_eval(&mut env.work, content, Constraint::Exact(Ty::RULE), false);
            Some((value, value_expected, false))
        }
        Node::DynRegex { pattern, flags } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            if let Some(fid) = flags {
                push_eval(&mut env.work, fid, Constraint::Exact(Ty::STR), false);
            }
            Some((pattern, Constraint::Exact(Ty::STR), false))
        }
        Node::Concat(range) => {
            enforce_leaf(&mut env.results, expected, emit, Ty::STR, span)?;
            descend_eval(
                shared.pools.child_slice(range),
                Constraint::Exact(Ty::STR),
                &mut env.work,
            )
        }
        Node::SeqOrChoice { range, .. } => {
            enforce_leaf(&mut env.results, expected, emit, Ty::RULE, span)?;
            for &member in shared.pools.child_slice(range).iter().rev() {
                push_spread_item(shared, member, Constraint::RuleLike, false, &mut env.work);
            }
            None
        }
        Node::Append { left, right } => {
            let (left_evaled, right_evaled) = append_operands(shared, left, right, expected);
            push_combine(&mut env.work, id, expected, emit);
            if right_evaled {
                push_eval(&mut env.work, right, expected, true);
            }
            if left_evaled {
                push_eval(&mut env.work, left, expected, true);
            }
            None
        }
        Node::FieldAccess { obj, .. } => {
            push_combine(&mut env.work, id, expected, emit);
            push_eval(&mut env.work, obj, Constraint::AnyObject, true);
            None
        }
        Node::QualifiedAccess { obj, .. } => {
            push_combine(&mut env.work, id, expected, emit);
            push_eval(&mut env.work, obj, Constraint::None, true);
            None
        }
        Node::GrammarConfig { module, .. } => {
            push_combine(&mut env.work, id, expected, emit);
            push_eval(
                &mut env.work,
                module,
                Constraint::Exact(Ty::ANY_MODULE),
                true,
            );
            None
        }
        Node::Alias { content, target } => {
            push_combine(&mut env.work, id, expected, emit);
            push_eval(&mut env.work, target, Constraint::None, true);
            push_eval(&mut env.work, content, Constraint::Exact(Ty::RULE), false);
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
                push_eval(&mut env.work, f.value, value_expected, true);
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
            let items = shared.pools.child_slice(range);
            let n = items.len();
            if !(TUPLE_MIN_ARITY..=TUPLE_MAX_ARITY).contains(&n) {
                return Err(TypeError::new(TypeErrorKind::TupleArityInvalid(n), span));
            }
            push_combine(&mut env.work, id, expected, emit);
            for &item in items.iter().rev() {
                push_eval(&mut env.work, item, Constraint::None, true);
            }
            None
        }
        Node::Call { name, args } => {
            let macro_id = resolve_macro_name(shared, ctx, name, span)?;
            let args = shared.pools.child_slice(args);
            enqueue_macro_call(
                shared,
                ctx,
                macro_id,
                id,
                name,
                args,
                expected,
                emit,
                &mut env.work,
            )?;
            None
        }
        Node::QualifiedCall(range) => {
            let (obj, name, args) = shared.pools.get_qualified_call(range);
            // Preserve recursive error order: receiver before macro name/args.
            type_of(shared, ctx, obj, env, Constraint::Exact(Ty::ANY_MODULE))?;
            let Node::Ident(IdentKind::Macro(macro_id)) = *shared.arena.get(name) else {
                let macro_name = ctx.text(shared.arena.span(name));
                return Err(TypeError::new(
                    TypeErrorKind::ImportMacroNotFound(macro_name.to_string()),
                    shared.arena.span(name),
                ));
            };
            enqueue_macro_call(
                shared,
                ctx,
                macro_id,
                id,
                name,
                args,
                expected,
                emit,
                &mut env.work,
            )?;
            None
        }
        Node::For { .. } => {
            push_combine(&mut env.work, id, expected, emit);
            enqueue_for(
                shared,
                ctx,
                id,
                Constraint::Exact(Ty::RULE),
                false,
                &mut env.work,
            )?;
            None
        }
        // Non-expression nodes never reach `type_of`.
        _ => unreachable!(),
    })
}

fn descend_eval(children: &[NodeId], c: Constraint, work: &mut Vec<Work>) -> Descent {
    let (&first, rest) = children.split_first()?;
    for &child in rest.iter().rev() {
        push_eval(work, child, c, false);
    }
    Some((first, c, false))
}

fn append_operands(
    shared: &SharedAst,
    left: NodeId,
    right: NodeId,
    expected: Constraint,
) -> (bool, bool) {
    let is_empty = |id| matches!(shared.arena.get(id), Node::List(r) if shared.pools.child_slice(*r).is_empty());
    let is_list_anno = matches!(expected, Constraint::Exact(t) if t.is_list());
    (
        !is_empty(left) || is_list_anno,
        !is_empty(right) || is_list_anno,
    )
}

/// Handle a [`Work::Combine`].
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
            let inner = obj_ty.object_inner().unwrap();
            let field_known = match shared.arena.get(obj) {
                Node::Ident(IdentKind::Var(let_id)) => env
                    .object_fields
                    .get(let_id)
                    .is_none_or(|fields| fields.iter().any(|f| f == field_name)),
                Node::Object(range) => {
                    let fields = shared.pools.get_object(*range);
                    fields.iter().any(|f| ctx.text(f.name) == field_name)
                }
                _ => true,
            };
            if !field_known {
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
            if matches!(shared.arena.get(target), Node::ModuleRule { .. }) {
                let target_span = shared.arena.span(target);
                return Err(TypeError::with_note(
                    TypeErrorKind::InvalidAliasTarget(target_ty),
                    target_span,
                    ctx.note(
                        NoteMessage::UseBareName(bare_name(ctx, target_span)),
                        target_span,
                    ),
                ));
            }
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

fn macro_call_result(
    shared: &SharedAst,
    ctx: &ModuleContext,
    name: NodeId,
    span: Span,
) -> TypeResult<Ty> {
    let Node::Ident(IdentKind::Macro(macro_id)) = *shared.arena.get(name) else {
        // Call names are resolved before the combine step is queued.
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
        push_eval(work, id, leaf, emit);
    }
}

/// Schedule a for-loop's iterable and body.
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
    push_eval(work, iterable, Constraint::None, true);
    Ok(())
}

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
    let mut err = TypeError::new(kind, span);
    if let Some(suggestion) = suggest_name(macro_name, macros) {
        err.add_note(ctx.note(NoteMessage::DidYouMean(suggestion.to_string()), span));
    }
    Err(err)
}

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
        push_eval(work, arg, Constraint::Exact(ty), false);
    }
    Ok(())
}

fn enforce(expected: Constraint, ty: Ty, span: Span) -> TypeResult<()> {
    if expected == Constraint::None {
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

fn enforce_leaf(
    results: &mut Vec<Ty>,
    expected: Constraint,
    emit: bool,
    ty: Ty,
    span: Span,
) -> TypeResult<()> {
    enforce(expected, ty, span)?;
    if emit {
        results.push(ty);
    }
    Ok(())
}

#[inline]
fn pop_result(results: &mut Vec<Ty>) -> Ty {
    // SAFETY: combine steps are queued behind their emitting children, so the
    // result is present.
    results.pop().unwrap()
}

fn push_eval(work: &mut Vec<Work>, id: NodeId, expected: Constraint, emit: bool) {
    work.push(Work::Eval { id, expected, emit });
}

fn push_combine(work: &mut Vec<Work>, id: NodeId, expected: Constraint, emit: bool) {
    work.push(Work::Combine { id, expected, emit });
}

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

fn validate_for_bindings(
    ctx: &ModuleContext,
    bindings: &[Param],
    body_span: Span,
) -> TypeResult<()> {
    if bindings.is_empty() {
        return Err(TypeError::new(TypeErrorKind::EmptyForBindings, body_span));
    }
    check_duplicate_names(ctx, bindings, |p| p.name, TypeErrorKind::DuplicateBinding)?;
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

    if let Node::List(range) = shared.arena.get(iterable)
        && shared.pools.child_slice(*range).is_empty()
    {
        return check_body(body, env);
    }

    let iter_ty = type_of(shared, ctx, iterable, env, Constraint::None)?;
    match_for_elem(shared, for_idx, iter_ty)?;
    check_body(body, env)
}

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

/// Bare member name of a module-qualified reference's source text.
fn bare_name(ctx: &ModuleContext, span: Span) -> String {
    ctx.text(span).rsplit("::").next().unwrap().trim().to_string()
}

const fn reject_module_type(ty: Ty, span: Span) -> TypeResult<()> {
    if matches!(ty, Ty::Module(_)) {
        return Err(TypeError::new(TypeErrorKind::ModuleTypeNotAllowed, span));
    }
    Ok(())
}

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
