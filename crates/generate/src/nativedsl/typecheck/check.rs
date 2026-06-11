use crate::nativedsl::{
    ContainerKind, DataTy, InnerTy, ModuleTy, NoteMessage, Ty, TypeError, TypeErrorKind,
    ast::{
        ChildRange, ForId, IdentKind, MacroId, MacroKind, ModuleContext, Node, NodeId, Param,
        PrecKind, SharedAst, Span,
    },
    resolve::resolve_module_ref,
    suggest::suggest_name,
    typecheck::{Constraint, TypeEnv, TypeResult},
};
use crate::nativedsl::typecheck::types::{
    ScalarTy, TUPLE_MAX_ARITY, TUPLE_MIN_ARITY, TupleSig, TupleSigError,
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
        Node::Let { ty, value, .. } => {
            let constraint = ty.map_or(Constraint::None, Constraint::Exact);
            let inferred = type_of(shared, ctx, *value, env, constraint)?;
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
            match config.kind {
                MacroKind::Expression(return_ty) => {
                    type_of(shared, ctx, config.body, env, Constraint::Exact(return_ty))?;
                }
                MacroKind::RuleSet => check_rule_set_body(shared, ctx, config.body, env)?,
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::ExpandedRule { body, .. } => {
            expect_rule(shared, ctx, *body, env)
        }
        // External decls have no body to typecheck.
        Node::External { .. } => Ok(()),
        _ => unreachable!(),
    }
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
        Node::Ident(IdentKind::Rule) | Node::SynthRef { .. } => Ok(()),
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
    if let Node::Object(range) = shared.arena.get(id) {
        for &(_, val_id) in shared.pools.get_object(*range) {
            expect_list(shared, ctx, val_id, env, expect_rule, Ty::LIST_RULE)?;
        }
        return Ok(());
    }
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
        // `ModuleRule`: a cross-module reference resolve already validated as a
        // rule / external / inherited rule, so it is rule-typed.
        Node::Ident(IdentKind::Rule)
        | Node::Blank
        | Node::SynthRef { .. }
        | Node::ModuleRule { .. } => Ok(Ty::RULE),
        &Node::SymRef { expr } => {
            type_of(shared, ctx, expr, env, Constraint::Exact(Ty::STR))?;
            Ok(Ty::RULE)
        }
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
                let kind = TypeErrorKind::GrammarConfigRequiresInherit;
                let arg_span = shared.arena.span(*module);
                if let Some(ref_id) = resolve_module_ref(&shared.arena, *module)
                    && let Node::ModuleRef { path, .. } = shared.arena.get(ref_id)
                {
                    let path_text = ctx.text(*path).to_string();
                    return Err(TypeError::with_note(
                        kind,
                        arg_span,
                        ctx.note(
                            NoteMessage::SwitchImportToInherit(path_text),
                            shared.arena.span(ref_id),
                        ),
                    ));
                }
                return Err(TypeError::new(kind, arg_span));
            }
            use crate::nativedsl::ast::ConfigField as C;
            Ok(match field {
                C::Extras | C::Externals | C::Inline | C::Supertypes => Ty::LIST_RULE,
                C::Conflicts | C::Precedences => Ty::LIST_LIST_RULE,
                C::Word | C::Start => Ty::RULE,
                C::Reserved => Ty::OBJ_LIST_RULE,
                C::Language | C::Inherits | C::Flags => unreachable!(),
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
        Node::BinOp { lhs, rhs, .. } => {
            type_of(shared, ctx, *lhs, env, Constraint::Exact(Ty::INT))?;
            type_of(shared, ctx, *rhs, env, Constraint::Exact(Ty::INT))?;
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
        Node::Tuple(range) => type_of_tuple(shared, ctx, *range, span, env),
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
            let leaf = |shared: &SharedAst, ctx: &ModuleContext, id: NodeId, env: &mut TypeEnv| {
                type_of(shared, ctx, id, env, Constraint::RuleLike)
            };
            for &member in shared.pools.child_slice(*range) {
                check_spread_item(shared, ctx, member, env, leaf)?;
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
            let target_ty = type_of(shared, ctx, *target, env, Constraint::RuleLike)
                .map_err(|e| translate_kind_mismatch(e, TypeErrorKind::InvalidAliasTarget))?;
            let is_valid = matches!(
                shared.arena.get(*target),
                Node::Ident(IdentKind::Rule | IdentKind::Var(_))
                    | Node::MacroParam { .. }
                    | Node::ForBinding { .. }
                    | Node::QualifiedAccess { .. }
                    | Node::FieldAccess { .. }
                    | Node::GrammarConfig { .. }
            ) || target_ty == Ty::STR;
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
                Constraint::IntOrStr
            };
            type_of(shared, ctx, *value, env, value_expected)?;
            expect_rule(shared, ctx, *content, env)?;
            Ok(Ty::RULE)
        }
        Node::For { for_id, body } => {
            // A for-loop in value position is invalid. Validate its internals
            // first (so loop errors surface), then report the position error:
            // for-loops are spliced inline and only appear in seq/choice/list,
            // where `check_spread_item` intercepts them before this arm.
            check_for_expr(shared, ctx, *for_id, *body, env, |body, env| {
                expect_rule(shared, ctx, body, env).map(|()| Ty::RULE)
            })?;
            Err(TypeError::new(TypeErrorKind::BoundForLoop, span))
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
            TypeErrorKind::EmptyContainerNeedsAnnotation(ContainerKind::List),
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
    match config.kind {
        MacroKind::Expression(return_ty) => Ok(return_ty),
        MacroKind::RuleSet => Err(TypeError::new(
            TypeErrorKind::RuleSetMacroInExpressionContext(macro_name.to_string()),
            span,
        )),
    }
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
    let leaf = |shared: &SharedAst, ctx: &ModuleContext, id: NodeId, env: &mut TypeEnv| {
        type_of(shared, ctx, id, env, elem_expected)
    };
    let mut widest = check_spread_item(shared, ctx, items[0], env, leaf)?;
    for &item_id in &items[1..] {
        let ty = check_spread_item(shared, ctx, item_id, env, leaf)?;
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
                ctx.note(NoteMessage::DidYouMean(suggestion), span),
            ));
        }
        return Err(TypeError::new(kind, span));
    };
    let args = shared.pools.child_slice(args);
    check_macro_args(shared, ctx, *macro_id, macro_name, args, span, env)
}

/// Type-check an item in a spread-accepting position (list element, seq/choice
/// member, etc). If the item is a `Node::For`, recursively flatMap into the
/// surrounding sequence by re-applying `leaf` to the inner body. Otherwise
/// dispatch to `leaf` directly.
///
/// Returns the type of the deepest non-For body so the caller can widen
/// against sibling items.
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

/// Type a tuple literal `(a, b, ...)`. Elements must be scalars and arity must
/// fall in `TUPLE_MIN_ARITY..=TUPLE_MAX_ARITY`. Arity is checked first so the
/// common grouping mistake `(expr)` (a would-be 1-tuple) gets the teaching
/// error rather than an element error.
fn type_of_tuple(
    shared: &SharedAst,
    ctx: &ModuleContext,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv,
) -> TypeResult<Ty> {
    let items = shared.pools.child_slice(range);
    let n = items.len();
    if !(TUPLE_MIN_ARITY..=TUPLE_MAX_ARITY).contains(&n) {
        return Err(TypeError::new(TypeErrorKind::TupleArityInvalid(n), span));
    }
    // `n` is in range, so this fixed buffer holds every element and the build
    // below cannot fail on arity.
    let mut scalars = [ScalarTy::Rule; TUPLE_MAX_ARITY];
    for (slot, &item_id) in scalars.iter_mut().zip(items) {
        let ty = type_of(shared, ctx, item_id, env, Constraint::None)?;
        let Ty::Data(DataTy::Scalar(s)) = ty else {
            return Err(TypeError::new(
                TypeErrorKind::TupleElementNotScalar(ty),
                shared.arena.span(item_id),
            ));
        };
        *slot = s;
    }
    let sig = TupleSig::new(&scalars[..n]).map_err(|e| match e {
        TupleSigError::Arity(bad) => TypeError::new(TypeErrorKind::TupleArityInvalid(bad), span),
    })?;
    Ok(Ty::Data(DataTy::Tuple(sig)))
}

/// Type-check a for-loop's iterable against its bindings. Every iterable source
/// - named list, `append`, macro arg, or non-empty literal - types to a
/// concrete list whose element is matched against the bindings uniformly: a
/// single binding takes the element directly; two or more destructure a tuple
/// element-wise.
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
    let bindings = &config.bindings;
    let iterable = config.iterable;

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
