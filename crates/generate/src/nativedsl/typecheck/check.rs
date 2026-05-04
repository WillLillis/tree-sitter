use crate::nativedsl::{
    DataTy, InnerTy, ModuleTy, ScalarTy, Ty, TypeError, TypeErrorKind,
    ast::{
        ChildRange, ForId, IdentKind, MacroId, ModuleContext, Node, NodeId, PrecKind, SharedAst,
        Span,
    },
    typecheck::TypeEnv,
};

/// Function pointer type for element checkers passed to `expect_list`.
type CheckFn<'ast> =
    fn(&SharedAst, &'ast ModuleContext, NodeId, &mut TypeEnv) -> Result<(), TypeError>;

pub(super) fn check_item(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    match shared.arena.get(id) {
        Node::Grammar => {
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
            let declared_ty = *ty;
            // For empty containers, the inner type comes from annotation only
            let is_empty_list = matches!(shared.arena.get(*value), Node::List(r) if shared.pools.child_slice(*r).is_empty());
            let is_empty_obj = matches!(shared.arena.get(*value), Node::Object(r) if shared.pools.get_object(*r).is_empty());
            let resolved_ty = match (is_empty_list, is_empty_obj, declared_ty) {
                (true, _, Some(ty)) if ty.is_list() => ty,
                (_, true, Some(ty)) if matches!(ty, Ty::Data(DataTy::Object(_))) => ty,
                (true, _, None) | (_, true, None) => {
                    return Err(TypeError::new(
                        TypeErrorKind::EmptyContainerNeedsAnnotation,
                        shared.arena.span(*value),
                    ));
                }
                (_, _, Some(ty)) => {
                    let inferred = type_of(shared, ctx, *value, env)?;
                    if !inferred.is_compatible(ty) {
                        return Err(mismatch(ty, inferred, shared.arena.span(*value)));
                    }
                    inferred
                }
                (_, _, None) => type_of(shared, ctx, *value, env)?,
            };
            if !resolved_ty.is_bindable() {
                return Err(TypeError::new(
                    TypeErrorKind::NonBindableType(resolved_ty),
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
                env.vars.insert(id, resolved_ty);
                env.object_fields.insert(id, fields);
            } else {
                env.vars.insert(id, resolved_ty);
            }
            Ok(())
        }
        Node::Macro(macro_id) => {
            let config = shared.pools.get_macro(*macro_id);
            let return_ty = config.return_ty;
            let prev = env.current_macro.replace(*macro_id);
            let body_ty = type_of(shared, ctx, config.body, env)?;
            env.current_macro = prev;
            if !body_ty.is_compatible(return_ty) {
                return Err(mismatch(return_ty, body_ty, shared.arena.span(config.body)));
            }
            Ok(())
        }
        Node::Rule { body, .. } => {
            env.vars.insert(id, Ty::RULE);
            expect_rule(shared, ctx, *body, env)
        }
        _ => unreachable!(),
    }
}

/// Check a list config field. For literal lists, checks each element with `check_elem`.
/// For non-literal expressions, checks the overall type is compatible with `expected`.
fn expect_list<'ast>(
    shared: &SharedAst,
    ctx: &'ast ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
    check_elem: CheckFn<'ast>,
    expected: Ty,
) -> Result<(), TypeError> {
    if let Node::List(range) = shared.arena.get(id) {
        for &child in shared.pools.child_slice(*range) {
            check_elem(shared, ctx, child, env)?;
        }
        return Ok(());
    }
    let ty = type_of(shared, ctx, id, env)?;
    if ty.is_compatible(expected) {
        return Ok(());
    }
    Err(mismatch(expected, ty, shared.arena.span(id)))
}

fn expect_name_list(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    expect_list(shared, ctx, id, env, expect_name_ref, Ty::LIST_RULE)
}

fn expect_name_ref(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    match shared.arena.get(id) {
        Node::Ident(IdentKind::Rule) => Ok(()),
        Node::Ident(IdentKind::Var(_))
        | Node::FieldAccess { .. }
        | Node::GrammarConfig { .. }
        | Node::MacroParam(_)
        | Node::ForBinding { .. } => {
            let ty = type_of(shared, ctx, id, env)?;
            if ty != Ty::RULE {
                return Err(mismatch(Ty::RULE, ty, shared.arena.span(id)));
            }
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
) -> Result<(), TypeError> {
    if matches!(
        shared.arena.get(id),
        Node::StringLit | Node::RawStringLit { .. }
    ) {
        return Ok(());
    }
    expect_name_ref(shared, ctx, id, env)
}

/// Check a `precedences` inner list: each element must be a name-ref or string literal.
fn expect_reserved(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    // Object literal: each field value must be a rule list
    if let Node::Object(range) = shared.arena.get(id) {
        for &(_, val_id) in shared.pools.get_object(*range) {
            expect_list(shared, ctx, val_id, env, expect_rule, Ty::LIST_RULE)?;
        }
        return Ok(());
    }
    // Non-literal: must be inherited (base.reserved -> Object(ListRule))
    let ty = type_of(shared, ctx, id, env)?;
    if ty.is_compatible(Ty::Data(DataTy::Object(InnerTy::List(ScalarTy::Rule)))) {
        return Ok(());
    }
    Err(TypeError::new(
        TypeErrorKind::ExpectedReservedConfig,
        shared.arena.span(id),
    ))
}

fn expect_rule(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    let ty = type_of(shared, ctx, id, env)?;
    if !ty.is_rule_like() {
        return Err(mismatch(Ty::RULE, ty, shared.arena.span(id)));
    }
    Ok(())
}

fn type_of(
    shared: &SharedAst,
    ctx: &ModuleContext,
    id: NodeId,
    env: &mut TypeEnv,
) -> Result<Ty, TypeError> {
    let span = shared.arena.span(id);
    match shared.arena.get(id) {
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
            let module_ty = type_of(shared, ctx, *module, env)?;
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
                C::Reserved => Ty::Data(DataTy::Object(InnerTy::List(ScalarTy::Rule))),
                C::Language | C::Inherits => unreachable!(),
            })
        }
        Node::Ident(IdentKind::Unresolved) => unreachable!(),
        Node::MacroParam(i) => {
            let macro_id = env.current_macro.unwrap();
            Ok(shared.pools.get_macro(macro_id).params[*i as usize].ty)
        }
        Node::ForBinding { for_id, index } => {
            Ok(shared.pools.get_for(*for_id).bindings[*index as usize].1)
        }
        Node::Ident(IdentKind::Macro(_)) => Err(TypeError::new(
            TypeErrorKind::MacroUsedAsValue(ctx.text(span).to_string()),
            span,
        )),
        Node::Ident(IdentKind::Var(let_id)) => env.vars.get(let_id).copied().ok_or_else(|| {
            let name = ctx.text(span);
            TypeError::new(TypeErrorKind::UnresolvedVariable(name.to_string()), span)
        }),
        Node::Neg(inner) => {
            let ty = type_of(shared, ctx, *inner, env)?;
            if ty != Ty::INT {
                return Err(mismatch(Ty::INT, ty, span));
            }
            Ok(Ty::INT)
        }
        Node::Append { left, right } => type_of_append(shared, ctx, *left, *right, span, env),
        Node::FieldAccess { obj, field } => type_of_field_access(shared, ctx, *obj, *field, env),
        // Only unresolved QualifiedAccess reaches type_of: inherited grammar
        // rules that exist in base_grammar.variables but not in root_items.
        // The resolver already resolved lets and caught macro-as-value errors.
        Node::QualifiedAccess { obj, .. } => {
            let obj_ty = type_of(shared, ctx, *obj, env)?;
            if !matches!(
                obj_ty,
                Ty::Module(ModuleTy::Import(_) | ModuleTy::Grammar(_))
            ) {
                return Err(TypeError::new(
                    TypeErrorKind::QualifiedAccessOnInvalidType(obj_ty),
                    shared.arena.span(*obj),
                ));
            }
            Ok(Ty::RULE)
        }
        Node::Object(range) => type_of_object(shared, ctx, *range, span, env),
        Node::List(range) => type_of_list(shared, ctx, *range, span, env),
        // Tuples are only first-class as direct elements of a for-loop
        // iterable's literal list (handled in check_for_expr without going
        // through type_of). Anywhere else, returning Ty::Tuple lets the
        // surrounding context surface the appropriate error: a let binding
        // produces NonBindableType, a containing list produces InvalidListElement,
        // a rule body produces TypeMismatch.
        Node::Tuple(_) => Ok(Ty::Tuple),
        Node::Concat(range) => {
            for &part in shared.pools.child_slice(*range) {
                let ty = type_of(shared, ctx, part, env)?;
                if ty != Ty::STR {
                    return Err(mismatch(Ty::STR, ty, shared.arena.span(part)));
                }
            }
            Ok(Ty::STR)
        }
        Node::DynRegex(range) => {
            let (pattern, flags) = shared.pools.get_regex(*range);
            let pt = type_of(shared, ctx, pattern, env)?;
            if pt != Ty::STR {
                return Err(mismatch(Ty::STR, pt, shared.arena.span(pattern)));
            }
            if let Some(fid) = flags {
                let ft = type_of(shared, ctx, fid, env)?;
                if ft != Ty::STR {
                    return Err(mismatch(Ty::STR, ft, shared.arena.span(fid)));
                }
            }
            Ok(Ty::RULE)
        }
        Node::SeqOrChoice { range, .. } => {
            for &member in shared.pools.child_slice(*range) {
                let ty = type_of(shared, ctx, member, env)?;
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
            let target_ty = type_of(shared, ctx, *target, env)?;
            let is_valid = matches!(
                shared.arena.get(*target),
                Node::Ident(IdentKind::Rule | IdentKind::Var(_))
                    | Node::MacroParam(_)
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
            let vt = type_of(shared, ctx, *value, env)?;
            if *kind == PrecKind::Dynamic {
                if vt != Ty::INT {
                    return Err(mismatch(Ty::INT, vt, shared.arena.span(*value)));
                }
            } else if vt != Ty::INT && vt != Ty::STR {
                return Err(TypeError::new(
                    TypeErrorKind::PrecValueTypeMismatch(vt),
                    shared.arena.span(*value),
                ));
            }
            expect_rule(shared, ctx, *content, env)?;
            Ok(Ty::RULE)
        }
        Node::For { for_id, body } => {
            check_for_expr(shared, ctx, *for_id, *body, env)?;
            Ok(Ty::Spread)
        }
        Node::Call { name, args } => type_of_call(shared, ctx, *name, *args, span, env),
        Node::QualifiedCall(range) => type_of_qualified_call(shared, ctx, *range, span, env),
        _ => Err(TypeError::new(TypeErrorKind::CannotInferType, span)),
    }
}

fn type_of_append(
    shared: &SharedAst,
    ctx: &ModuleContext,
    left: NodeId,
    right: NodeId,
    span: Span,
    env: &mut TypeEnv,
) -> Result<Ty, TypeError> {
    let is_empty = |id| matches!(shared.arena.get(id), Node::List(r) if shared.pools.child_slice(*r).is_empty());
    let lt = (!is_empty(left))
        .then(|| type_of(shared, ctx, left, env))
        .transpose()?;
    let rt = (!is_empty(right))
        .then(|| type_of(shared, ctx, right, env))
        .transpose()?;
    match (lt, rt) {
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
) -> Result<Ty, TypeError> {
    let obj_ty = type_of(shared, ctx, obj, env)?;
    let field_name = ctx.text(field);
    match obj_ty {
        Ty::Data(DataTy::Object(inner)) => {
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
        _ => Err(TypeError::new(
            TypeErrorKind::FieldAccessOnNonObject(obj_ty),
            shared.arena.span(obj),
        )),
    }
}

fn type_of_qualified_call(
    shared: &SharedAst,
    ctx: &ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv,
) -> Result<Ty, TypeError> {
    let (obj, name, args) = shared.pools.get_qualified_call(range);
    let obj_ty = type_of(shared, ctx, obj, env)?;
    if !matches!(
        obj_ty,
        Ty::Module(ModuleTy::Import(_) | ModuleTy::Grammar(_))
    ) {
        return Err(TypeError::new(
            TypeErrorKind::QualifiedCallOnNonModule(obj_ty),
            shared.arena.span(obj),
        ));
    }
    // Name was resolved to Ident(Macro(macro_id)) by the resolver
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
) -> Result<Ty, TypeError> {
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
        let arg_ty = type_of(shared, ctx, arg_id, env)?;
        if !arg_ty.is_compatible(config.params[i].ty) {
            return Err(mismatch(
                config.params[i].ty,
                arg_ty,
                shared.arena.span(arg_id),
            ));
        }
    }
    Ok(config.return_ty)
}

fn type_of_object(
    shared: &SharedAst,
    ctx: &ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv,
) -> Result<Ty, TypeError> {
    let fields = shared.pools.get_object(range);
    if fields.is_empty() {
        return Err(TypeError::new(TypeErrorKind::CannotInferType, span));
    }
    let mut widest = type_of(shared, ctx, fields[0].1, env)?;
    for &(_, val_id) in &fields[1..] {
        let ty = type_of(shared, ctx, val_id, env)?;
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
) -> Result<Ty, TypeError> {
    let items = shared.pools.child_slice(range);
    if items.is_empty() {
        return Err(TypeError::new(
            TypeErrorKind::EmptyContainerNeedsAnnotation,
            span,
        ));
    }
    let mut widest = type_of(shared, ctx, items[0], env)?;
    for &item_id in &items[1..] {
        let ty = type_of(shared, ctx, item_id, env)?;
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
) -> Result<Ty, TypeError> {
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
) -> Result<(), TypeError> {
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
                for (i, &(name_span, declared_ty)) in config.bindings.iter().enumerate() {
                    let actual = type_of(shared, ctx, elems[i], env)?;
                    if !actual.is_compatible(declared_ty) {
                        return Err(mismatch(declared_ty, actual, name_span));
                    }
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
    let iter_ty = type_of(shared, ctx, config.iterable, env)?;
    if !iter_ty.is_list() {
        return Err(TypeError::new(
            TypeErrorKind::ForRequiresList(iter_ty),
            shared.arena.span(config.iterable),
        ));
    }
    let (name_span, declared_ty) = config.bindings[0];
    // is_list() check above guarantees elem_type() returns Some
    let elem_ty = iter_ty.elem_type().unwrap();
    if !elem_ty.is_compatible(declared_ty) {
        return Err(mismatch(declared_ty, elem_ty, name_span));
    }
    expect_rule(shared, ctx, body, env)
}

const fn mismatch(expected: Ty, got: Ty, span: Span) -> TypeError {
    TypeError::new(TypeErrorKind::TypeMismatch { expected, got }, span)
}
