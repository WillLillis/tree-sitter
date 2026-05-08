use crate::nativedsl::{
    ContainerKind, DataTy, InnerTy, ModuleTy, Ty, TypeError, TypeErrorKind,
    ast::{
        ChildRange, ForId, IdentKind, MacroId, ModuleContext, Node, NodeId, Param, PrecKind,
        SharedAst, Span,
    },
    typecheck::{Constraint, TypeEnv, TypeResult},
};

/// Function pointer type for element checkers passed to `expect_list`.
type CheckFn<'ast> = fn(&SharedAst, &'ast ModuleContext, NodeId, &mut TypeEnv) -> TypeResult<()>;

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
            if let Some(id) = config.reserved {
                expect_reserved(shared, ctx, id, env)?;
            }
            Ok(())
        }
        Node::Let { ty, value, .. } => {
            let constraint = ty.map_or(Constraint::None, Constraint::Exact);
            let inferred = type_of(shared, ctx, *value, env, constraint)?;
            if !inferred.is_bindable() {
                return Err(TypeError::new(
                    TypeErrorKind::NonBindableType(inferred),
                    shared.arena.span(*value),
                ));
            }
            // If the value is an object literal, register its field names
            if let Node::Object(range) = shared.arena.get(*value) {
                let fields: Vec<String> = shared
                    .pools
                    .get_object(*range)
                    .iter()
                    .map(|(span, _)| ctx.text(*span).to_string())
                    .collect();
                env.vars.insert(id, inferred);
                env.object_fields.insert(id, fields);
            } else {
                env.vars.insert(id, inferred);
            }
            Ok(())
        }
        Node::Macro(macro_id) => {
            let config = shared.pools.get_macro(*macro_id);
            type_of(
                shared,
                ctx,
                config.body,
                env,
                Constraint::Exact(config.return_ty),
            )?;
            Ok(())
        }
        Node::Rule { body, .. } => expect_rule(shared, ctx, *body, env),
        // External decls have no body to typecheck.
        Node::External { .. } => Ok(()),
        _ => unreachable!(),
    }
}

/// Check a list config field. For literal lists, checks each element with
/// `check_elem`. For non-literal expressions, defers to `type_of` with an
/// `Exact(expected)` constraint.
fn expect_list<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    check_elem: CheckFn<'ast>,
    expected: Ty,
) -> TypeResult<()> {
    if let Node::List(range) = shared.arena.get(id) {
        for &child in shared.pools.child_slice(*range) {
            check_elem(shared, ctx, child, env)?;
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

/// Check the `reserved` config field: an object literal `{name: [rules]}` or
/// an inherited expression typing to `obj_t<list_t<rule_t>>`.
fn expect_reserved(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    // Object literal: each field value must be a rule list
    if let Node::Object(range) = shared.arena.get(id) {
        for &(_, val_id) in shared.pools.get_object(*range) {
            expect_list(shared, ctx, val_id, env, expect_rule, Ty::LIST_RULE)?;
        }
        return Ok(());
    }
    // Non-literal: must type to obj_t<list_t<rule_t>>. Common case is
    // inherited but any expression of that type works.
    type_of(shared, ctx, id, env, Constraint::Exact(Ty::OBJ_LIST_RULE))?;
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

/// Type a node. `expected` propagates through structural nodes (lists,
/// objects, append) so empty containers can infer from context, and the
/// resulting type is verified to satisfy `expected` before returning.
/// `Constraint::None` always satisfies, effectively skipping enforcement.
fn type_of(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    expected: Constraint,
) -> TypeResult<Ty> {
    let span = shared.arena.span(id);
    let ty = match shared.arena.get(id) {
        Node::IntLit(_) => Ok(Ty::INT),
        Node::StringLit | Node::RawStringLit { .. } => Ok(Ty::STR),
        Node::Ident(IdentKind::Rule) | Node::Blank => Ok(Ty::RULE),
        Node::ModuleRef { import, module, .. } => {
            let idx = module.expect("module index not set by loading pre-pass");
            Ok(Ty::Module(if *import {
                ModuleTy::Import(idx)
            } else {
                ModuleTy::Grammar(idx)
            }))
        }
        Node::GrammarConfig { module, field } => {
            let module_ty = type_of(shared, ctx, *module, env, Constraint::Exact(Ty::ANY_MODULE))?;
            if !matches!(module_ty, Ty::Module(ModuleTy::Grammar(_))) {
                return Err(TypeError::new(
                    TypeErrorKind::GrammarConfigRequiresInherit,
                    shared.arena.span(*module),
                ));
            }
            use crate::nativedsl::ast::ConfigField as C;
            Ok(match field {
                C::Extras | C::Externals | C::Inline | C::Supertypes => Ty::LIST_RULE,
                C::Conflicts | C::Precedences => Ty::LIST_LIST_RULE,
                C::Word => Ty::RULE,
                C::Reserved => Ty::OBJ_LIST_RULE,
                C::Language | C::Inherits => unreachable!(),
            })
        }
        &Node::MacroParam { ty, .. } | &Node::ForBinding { ty, .. } => Ok(ty),
        Node::Ident(IdentKind::Macro(_)) => Err(TypeError::new(
            TypeErrorKind::MacroUsedAsValue(ctx.text(span).to_string()),
            span,
        )),
        // Resolver guarantees Var(let_id) points at a Let typechecked earlier
        // in source order (cross-module: helpers checked before parent).
        Node::Ident(IdentKind::Var(let_id)) => Ok(*env.vars.get(let_id).unwrap()),
        Node::Neg(inner) => {
            type_of(shared, ctx, *inner, env, Constraint::Exact(Ty::INT))?;
            Ok(Ty::INT)
        }
        Node::Append { left, right } => {
            type_of_append(shared, ctx, *left, *right, span, env, expected)
        }
        Node::FieldAccess { obj, field } => type_of_field_access(shared, ctx, *obj, *field, env),
        // Only unresolved QualifiedAccess reaches type_of: inherited grammar
        // rules that exist in base_grammar.variables but not in root_items.
        // The resolver already resolved lets and caught macro-as-value errors.
        Node::QualifiedAccess { obj, .. } => {
            type_of(shared, ctx, *obj, env, Constraint::Exact(Ty::ANY_MODULE))?;
            Ok(Ty::RULE)
        }
        Node::Object(range) => type_of_object(shared, ctx, *range, span, env, expected),
        Node::List(range) => type_of_list(shared, ctx, *range, span, env, expected),
        // Tuples are only first-class as direct elements of a for-loop
        // iterable's literal list (handled in check_for_expr without going
        // through type_of). Anywhere else, returning Ty::Tuple lets the
        // surrounding context surface the appropriate error: a let binding
        // produces NonBindableType, a containing list produces InvalidListElement,
        // a rule body produces TypeMismatch.
        Node::Tuple(_) => Ok(Ty::Tuple),
        Node::Concat(range) => {
            for &part in shared.pools.child_slice(*range) {
                type_of(shared, ctx, part, env, Constraint::Exact(Ty::STR))?;
            }
            Ok(Ty::STR)
        }
        &Node::DynRegex { pattern, flags } => {
            type_of(shared, ctx, pattern, env, Constraint::Exact(Ty::STR))?;
            if let Some(fid) = flags {
                type_of(shared, ctx, fid, env, Constraint::Exact(Ty::STR))?;
            }
            Ok(Ty::RULE)
        }
        Node::SeqOrChoice { range, .. } => {
            for &member in shared.pools.child_slice(*range) {
                // Spread (from for-loop expansion) is also valid here, but
                // can't be expressed in user-facing constraint Display, so
                // we type without enforcement and validate by hand.
                let ty = type_of(shared, ctx, member, env, Constraint::None)?;
                if ty != Ty::Spread && !ty.is_rule_like() {
                    return Err(mismatch(Ty::RULE, ty, shared.arena.span(member)));
                }
            }
            Ok(Ty::RULE)
        }
        Node::Repeat { inner, .. }
        | Node::Token { inner, .. }
        | Node::Field { content: inner, .. }
        | Node::Reserved { content: inner, .. } => {
            expect_rule(shared, ctx, *inner, env)?;
            Ok(Ty::RULE)
        }
        Node::Alias { content, target } => {
            expect_rule(shared, ctx, *content, env)?;
            let target_ty = type_of(shared, ctx, *target, env, Constraint::RULE_LIKE)
                .map_err(|e| translate_kind_mismatch(e, TypeErrorKind::InvalidAliasTarget))?;
            let is_valid = matches!(
                shared.arena.get(*target),
                Node::Ident(IdentKind::Rule | IdentKind::Var(_))
                    | Node::MacroParam { .. }
                    | Node::ForBinding { .. }
                    | Node::QualifiedAccess { .. }
                    | Node::FieldAccess { .. }
                    | Node::GrammarConfig { .. }
            ) && target_ty.is_rule_like()
                || target_ty == Ty::STR;
            if !is_valid {
                return Err(TypeError::new(
                    TypeErrorKind::InvalidAliasTarget(target_ty),
                    shared.arena.span(*target),
                ));
            }
            Ok(Ty::RULE)
        }
        Node::Prec {
            kind,
            value,
            content,
        } => {
            let value_expected = if *kind == PrecKind::Dynamic {
                Constraint::Exact(Ty::INT)
            } else {
                Constraint::INT_OR_STR
            };
            type_of(shared, ctx, *value, env, value_expected)?;
            expect_rule(shared, ctx, *content, env)?;
            Ok(Ty::RULE)
        }
        Node::For { for_id, body } => {
            check_for_expr(shared, ctx, *for_id, *body, env)?;
            Ok(Ty::Spread)
        }
        Node::Call { name, args } => type_of_call(shared, ctx, *name, *args, span, env),
        Node::QualifiedCall(range) => type_of_qualified_call(shared, ctx, *range, span, env),
        // Top-level items (Grammar/Rule/Let/Macro) and Unreachable never
        // appear in expression position - parser dispatch keeps them apart.
        _ => unreachable!(),
    }?;
    // Short-circuit the most common no-op case before the satisfies dispatch.
    if matches!(expected, Constraint::None) {
        return Ok(ty);
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
    Ok(ty)
}

fn type_of_append(
    shared: &SharedAst,
    ctx: &ModuleContext,
    left: NodeId,
    right: NodeId,
    span: Span,
    env: &mut TypeEnv,
    expected: Constraint,
) -> TypeResult<Ty> {
    let is_empty = |id| matches!(shared.arena.get(id), Node::List(r) if shared.pools.child_slice(*r).is_empty());
    let is_list = |constr| matches!(constr, Constraint::Exact(t) if t.is_list());
    // Don't typecheck empty list arguments unless we have an explicit annotation
    let l_ty = (!is_empty(left) || is_list(expected))
        .then(|| type_of(shared, ctx, left, env, expected))
        .transpose()?;
    let r_ty = (!is_empty(right) || is_list(expected))
        .then(|| type_of(shared, ctx, right, env, expected))
        .transpose()?;
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
                .ok_or_else(|| mismatch(l, r, shared.arena.span(right)))
        }
        // One arm is `[]` (skipped `type_of` above); the other typed to `t`. Result
        // is `t` if it's list-shaped.
        (Some(t), None) | (None, Some(t)) => {
            if !t.is_list() {
                return Err(TypeError::new(TypeErrorKind::AppendRequiresList(t), span));
            }
            Ok(t)
        }
        (None, None) => Err(TypeError::new(
            TypeErrorKind::EmptyContainerNeedsAnnotation,
            span,
        )),
    }
}

fn type_of_field_access(
    shared: &SharedAst,
    ctx: &ModuleContext,
    obj: NodeId,
    field: Span,
    env: &mut TypeEnv,
) -> TypeResult<Ty> {
    let obj_ty = type_of(shared, ctx, obj, env, Constraint::AnyObject)?;
    let field_name = ctx.text(field);
    // AnyObject auto-enforce above guarantees the variant.
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
            fields
                .iter()
                .any(|(key_span, _)| ctx.text(*key_span) == field_name)
        }
        _ => true, // can't validate dynamically-produced objects
    };
    if !field_known {
        return Err(TypeError::new(
            TypeErrorKind::FieldNotFound {
                field: field_name.to_string(),
                on_type: obj_ty,
            },
            field,
        ));
    }
    Ok(Ty::from(inner))
}

fn type_of_qualified_call(
    shared: &SharedAst,
    ctx: &ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv,
) -> TypeResult<Ty> {
    let (obj, name, args) = shared.pools.get_qualified_call(range);
    type_of(shared, ctx, obj, env, Constraint::Exact(Ty::ANY_MODULE))?;
    let macro_name = ctx.text(shared.arena.span(name));
    let Node::Ident(IdentKind::Macro(macro_id)) = shared.arena.get(name) else {
        return Err(TypeError::new(
            TypeErrorKind::ImportMacroNotFound(macro_name.to_string()),
            shared.arena.span(name),
        ));
    };
    check_macro_args(shared, ctx, *macro_id, macro_name, args, span, env)
}

fn check_macro_args(
    shared: &SharedAst,
    ctx: &ModuleContext,
    macro_id: MacroId,
    macro_name: &str,
    args: &[NodeId],
    span: Span,
    env: &mut TypeEnv,
) -> TypeResult<Ty> {
    let config = shared.pools.get_macro(macro_id);
    if args.len() != config.params.len() {
        return Err(TypeError::new(
            TypeErrorKind::ArgCountMismatch {
                macro_name: macro_name.to_string(),
                expected: config.params.len(),
                got: args.len(),
            },
            span,
        ));
    }
    for (i, &arg_id) in args.iter().enumerate() {
        type_of(
            shared,
            ctx,
            arg_id,
            env,
            Constraint::Exact(config.params[i].ty),
        )?;
    }
    Ok(config.return_ty)
}

fn type_of_object(
    shared: &SharedAst,
    ctx: &ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv,
    expected: Constraint,
) -> TypeResult<Ty> {
    let fields = shared.pools.get_object(range);
    if fields.is_empty() {
        return empty_container_result(expected, ContainerKind::Object, span);
    }

    let expected_inner = expected.object_value();
    let mut widest = type_of(shared, ctx, fields[0].1, env, expected_inner)?;
    for &(_, val_id) in &fields[1..] {
        let ty = type_of(shared, ctx, val_id, env, expected_inner)?;
        widest = widest
            .widen(ty)
            .ok_or_else(|| mismatch(widest, ty, shared.arena.span(val_id)))?;
    }
    let invalid = || {
        TypeError::new(
            TypeErrorKind::InvalidObjectValue(widest),
            shared.arena.span(fields[0].1),
        )
    };
    let Ty::Data(d) = widest else {
        return Err(invalid());
    };
    let inner = InnerTy::try_from(d).map_err(|()| invalid())?;
    Ok(Ty::Data(DataTy::Object(inner)))
}

fn type_of_list(
    shared: &SharedAst,
    ctx: &ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv,
    expected: Constraint,
) -> TypeResult<Ty> {
    let items = shared.pools.child_slice(range);
    if items.is_empty() {
        return empty_container_result(expected, ContainerKind::List, span);
    }
    let elem_expected = expected.elem();
    let mut widest = type_of(shared, ctx, items[0], env, elem_expected)?;
    for &item_id in &items[1..] {
        let ty = type_of(shared, ctx, item_id, env, elem_expected)?;
        widest = widest.widen(ty).ok_or_else(|| {
            TypeError::new(
                TypeErrorKind::ListElementTypeMismatch {
                    first: widest,
                    got: ty,
                },
                shared.arena.span(item_id),
            )
        })?;
    }
    widest
        .to_list()
        .ok_or_else(|| TypeError::new(TypeErrorKind::InvalidListElement(widest), span))
}

fn type_of_call(
    shared: &SharedAst,
    ctx: &ModuleContext,
    name: NodeId,
    args: ChildRange,
    span: Span,
    env: &mut TypeEnv,
) -> TypeResult<Ty> {
    let macro_name = ctx.text(shared.arena.span(name));
    let Node::Ident(IdentKind::Macro(macro_id)) = shared.arena.get(name) else {
        return Err(TypeError::new(
            TypeErrorKind::UndefinedMacro(macro_name.to_string()),
            span,
        ));
    };
    let args = shared.pools.child_slice(args);
    check_macro_args(shared, ctx, *macro_id, macro_name, args, span, env)
}

fn check_for_expr(
    shared: &SharedAst,
    ctx: &ModuleContext,
    for_idx: ForId,
    body: NodeId,
    env: &mut TypeEnv,
) -> TypeResult<()> {
    let config = shared.pools.get_for(for_idx);

    // Check if iterable is a literal list of tuples (for destructuring)
    if let Node::List(range) = shared.arena.get(config.iterable) {
        let items = shared.pools.child_slice(*range);
        let has_tuples = items
            .first()
            .is_some_and(|&id| matches!(shared.arena.get(id), Node::Tuple(_)));
        if has_tuples {
            for &item_id in items {
                let Node::Tuple(range) = shared.arena.get(item_id) else {
                    return Err(TypeError::new(
                        TypeErrorKind::ForRequiresTuples,
                        shared.arena.span(item_id),
                    ));
                };
                let elems = shared.pools.child_slice(*range);
                if elems.len() != config.bindings.len() {
                    return Err(TypeError::new(
                        TypeErrorKind::ForBindingCountMismatch {
                            bindings: config.bindings.len(),
                            tuple_elements: elems.len(),
                        },
                        shared.arena.span(item_id),
                    ));
                }
                for (i, p) in config.bindings.iter().enumerate() {
                    type_of(shared, ctx, elems[i], env, Constraint::Exact(p.ty))?;
                }
            }
            return expect_rule(shared, ctx, body, env);
        }
    }

    // Single binding over a flat list
    if config.bindings.len() != 1 {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresTuples,
            shared.arena.span(config.iterable),
        ));
    }
    let Param {
        name,
        ty: declared_ty,
    } = config.bindings[0];
    // Bespoke checks below distinguish "not a list" (ForRequiresList) from
    // "wrong element type" (TypeMismatch); auto-enforcing a list constraint
    // here would conflate them.
    let iter_ty = type_of(shared, ctx, config.iterable, env, Constraint::None)?;
    let Some(elem_ty) = iter_ty.list_elem() else {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresList(iter_ty),
            shared.arena.span(config.iterable),
        ));
    };
    if !elem_ty.is_compatible(declared_ty) {
        return Err(mismatch(declared_ty, elem_ty, name));
    }
    expect_rule(shared, ctx, body, env)
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
            TypeErrorKind::EmptyContainerNeedsAnnotation,
            span,
        )),
    }
}

/// Translate an auto-enforce constraint failure to a bespoke error variant
/// that carries the actual `got` type. Non-mismatch errors pass through.
fn translate_kind_mismatch(err: TypeError, make: impl FnOnce(Ty) -> TypeErrorKind) -> TypeError {
    match err.kind {
        TypeErrorKind::TypeMismatch { got, .. } | TypeErrorKind::ConstraintMismatch { got, .. } => {
            TypeError::new(make(got), err.span.unwrap())
        }
        _ => err,
    }
}

const fn mismatch(expected: Ty, got: Ty, span: Span) -> TypeError {
    TypeError::new(TypeErrorKind::TypeMismatch { expected, got }, span)
}
