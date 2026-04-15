//! Type checker for the native grammar DSL.
//!
//! Runs after name resolution and before lowering. Uses a simple flat type
//! system with no generics - just concrete types for the values that grammar
//! definitions actually use.
//!
//! Subtyping: `str_t` is a subtype of `rule_t`, `list_str_t` is a subtype of
//! `list_rule_t`, and `list_list_str_t` is a subtype of `list_list_rule_t`.

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use super::ast::{Ast, ForId, Node, NodeId, Span};

/// Types that can appear as homogeneous object field values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum InnerTy {
    Rule,
    Str,
    Int,
    ListRule,
    ListStr,
}

impl std::fmt::Display for InnerTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rule => "rule_t",
            Self::Str => "str_t",
            Self::Int => "int_t",
            Self::ListRule => "list_rule_t",
            Self::ListStr => "list_str_t",
        })
    }
}

/// The type of a DSL expression.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Ty {
    Rule,
    Str,
    Int,
    ListRule,
    ListStr,
    ListInt,
    ListListRule,
    ListListStr,
    ListListInt,
    Grammar,
    Spread,
    /// Homogeneous object `{ field: T, ... }`. The inner type is what field
    /// access returns. Field name validation is done via `ObjectInfo` in the env.
    Object(InnerTy),
}

impl Ty {
    fn is_rule_like(self) -> bool {
        self == Self::Rule || self == Self::Str
    }

    const fn is_list(self) -> bool {
        matches!(
            self,
            Self::ListRule
                | Self::ListStr
                | Self::ListInt
                | Self::ListListRule
                | Self::ListListStr
                | Self::ListListInt
        )
    }

    /// Check if `self` is assignable to `expected`.
    fn is_compatible(self, expected: Self) -> bool {
        self == expected
            || (self == Self::Str && expected == Self::Rule)
            || (self == Self::ListStr && expected == Self::ListRule)
            || (self == Self::ListListStr && expected == Self::ListListRule)
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Rule => f.write_str("rule_t"),
            Self::Str => f.write_str("str_t"),
            Self::Int => f.write_str("int_t"),
            Self::ListRule => f.write_str("list_rule_t"),
            Self::ListStr => f.write_str("list_str_t"),
            Self::ListInt => f.write_str("list_int_t"),
            Self::ListListRule => f.write_str("list_list_rule_t"),
            Self::ListListStr => f.write_str("list_list_str_t"),
            Self::ListListInt => f.write_str("list_list_int_t"),
            Self::Grammar => f.write_str("grammar_t"),
            Self::Spread => f.write_str("for-loop expansion"),
            Self::Object(inner) => write!(f, "object<{inner}>"),
        }
    }
}

// -- Type environment --

pub struct FnSig {
    pub params: Vec<Ty>,
    pub return_ty: Ty,
}

pub struct TypeEnv<'src> {
    pub var_scopes: Vec<FxHashMap<&'src str, Ty>>,
    pub fns: FxHashMap<&'src str, FnSig>,
    /// Field names for Object variables, keyed by variable name.
    pub object_fields: FxHashMap<&'src str, Vec<&'src str>>,
}

impl<'src> TypeEnv<'src> {
    fn new() -> Self {
        Self {
            var_scopes: vec![FxHashMap::default()],
            fns: FxHashMap::default(),
            object_fields: FxHashMap::default(),
        }
    }
    fn insert_var(&mut self, name: &'src str, ty: Ty) {
        self.var_scopes.last_mut().unwrap().insert(name, ty);
    }
    fn insert_object(&mut self, name: &'src str, ty: Ty, fields: Vec<&'src str>) {
        self.insert_var(name, ty);
        self.object_fields.insert(name, fields);
    }
    fn get_object_fields(&self, name: &str) -> Option<&[&'src str]> {
        self.object_fields.get(name).map(Vec::as_slice)
    }
    fn get_var(&self, name: &str) -> Option<Ty> {
        self.var_scopes
            .iter()
            .rev()
            .find_map(|s| s.get(name).copied())
    }
    fn scoped(&mut self) -> ScopeGuard<'_, 'src> {
        self.var_scopes.push(FxHashMap::default());
        ScopeGuard(self)
    }
}

struct ScopeGuard<'a, 'src>(&'a mut TypeEnv<'src>);

impl Drop for ScopeGuard<'_, '_> {
    fn drop(&mut self) {
        self.0.var_scopes.pop();
    }
}

impl<'src> std::ops::Deref for ScopeGuard<'_, 'src> {
    type Target = TypeEnv<'src>;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl std::ops::DerefMut for ScopeGuard<'_, '_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

// -- Type annotation resolution --

fn resolve_type_annotation(ast: &Ast<'_>, id: NodeId) -> Result<Ty, TypeError> {
    match ast.node(id) {
        Node::TypeRule => Ok(Ty::Rule),
        Node::TypeStr => Ok(Ty::Str),
        Node::TypeInt => Ok(Ty::Int),
        Node::TypeListRule => Ok(Ty::ListRule),
        Node::TypeListStr => Ok(Ty::ListStr),
        Node::TypeListInt => Ok(Ty::ListInt),
        Node::TypeListListRule => Ok(Ty::ListListRule),
        Node::TypeListListStr => Ok(Ty::ListListStr),
        Node::TypeListListInt => Ok(Ty::ListListInt),
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span: ast.span(id),
        }),
    }
}

// -- Entry point --

/// Run typechecking and return the type environment.
pub fn check<'src>(ast: &'src Ast<'src>) -> Result<TypeEnv<'src>, TypeError> {
    let mut env = TypeEnv::new();
    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                env.insert_var(ast.node_text(*name), Ty::Rule);
            }
            Node::Fn(fn_idx) => {
                let config = ast.get_fn(*fn_idx);
                let params: Vec<Ty> = config
                    .params
                    .iter()
                    .map(|p| resolve_type_annotation(ast, p.ty))
                    .collect::<Result<_, _>>()?;
                let return_ty = resolve_type_annotation(ast, config.return_ty)?;
                env.fns
                    .insert(ast.node_text(config.name), FnSig { params, return_ty });
            }
            _ => {}
        }
    }
    for &item_id in &ast.root_items {
        check_item(ast, item_id, &mut env)?;
    }
    Ok(env)
}

fn check_item<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    match ast.node(id) {
        Node::Grammar => {
            let config = ast.context.grammar_config.as_ref().unwrap();
            for id in [config.extras, config.externals].into_iter().flatten() {
                expect_rule_list(ast, id, env)?;
            }
            for id in [config.inline, config.supertypes].into_iter().flatten() {
                expect_name_list(ast, id, env)?;
            }
            if let Some(id) = config.conflicts {
                expect_list_list(ast, id, env, expect_name_list)?;
            }
            if let Some(id) = config.precedences {
                expect_list_list(ast, id, env, expect_precedence_group)?;
            }
            if let Some(id) = config.word {
                expect_name_ref(ast, id, env)?;
            }
            if let Some(id) = config.reserved {
                expect_reserved(ast, id, env)?;
            }
            Ok(())
        }
        Node::Let { name, ty, value } => {
            let var_name = ast.node_text(*name);
            let declared = ty.map(|t| resolve_type_annotation(ast, t)).transpose()?;
            // Empty list: type comes from annotation, not inference
            let is_empty_list =
                matches!(ast.node(*value), Node::List(r) if ast.child_slice(*r).is_empty());
            let resolved_ty = if is_empty_list {
                match declared {
                    Some(ty) if ty.is_list() => ty,
                    _ => {
                        return Err(TypeError {
                            kind: TypeErrorKind::EmptyListNeedsAnnotation,
                            span: ast.span(*value),
                        });
                    }
                }
            } else {
                let inferred = type_of(ast, *value, env)?;
                if let Some(d) = declared
                    && !inferred.is_compatible(d)
                {
                    return Err(mismatch(d, inferred, ast.span(*value)));
                }
                inferred
            };
            // If the value is an object literal, register its field names
            if let Node::Object(range) = ast.node(*value) {
                let fields: Vec<&str> = ast
                    .get_object(*range)
                    .iter()
                    .map(|(span, _)| ast.text(*span))
                    .collect();
                env.insert_object(var_name, resolved_ty, fields);
            } else {
                env.insert_var(var_name, resolved_ty);
            }
            Ok(())
        }
        Node::Fn(fn_idx) => {
            let config = ast.get_fn(*fn_idx);
            let fn_name = ast.node_text(config.name);
            let sig = &env.fns[fn_name];
            let return_ty = sig.return_ty;
            let n_params = sig.params.len();
            let mut scope = env.scoped();
            for i in 0..n_params {
                let ty = scope.fns[fn_name].params[i];
                scope.insert_var(ast.node_text(config.params[i].name), ty);
            }
            let body_ty = type_of(ast, config.body, &mut scope)?;
            if !body_ty.is_compatible(return_ty) {
                return Err(mismatch(return_ty, body_ty, ast.span(config.body)));
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::OverrideRule { body, .. } => expect_rule(ast, *body, env),
        _ => Err(TypeError {
            kind: TypeErrorKind::UnexpectedTopLevel,
            span: ast.span(id),
        }),
    }
}

// -- Config field validators --

/// Check a list config field. For literal lists, checks each element with `check_elem`.
/// For non-literal expressions, checks the overall type is a list type.
fn expect_list<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    check_elem: fn(&'src Ast<'src>, NodeId, &mut TypeEnv<'src>) -> Result<(), TypeError>,
    accept_list_str: bool,
) -> Result<(), TypeError> {
    if let Node::List(range) = ast.node(id) {
        for &child in ast.child_slice(*range) {
            check_elem(ast, child, env)?;
        }
        return Ok(());
    }
    let ty = type_of(ast, id, env)?;
    if ty == Ty::ListRule || (accept_list_str && ty == Ty::ListStr) {
        return Ok(());
    }
    Err(mismatch(Ty::ListRule, ty, ast.span(id)))
}

fn expect_rule_list<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    expect_list(ast, id, env, expect_rule, true)
}

fn expect_name_list<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    expect_list(ast, id, env, expect_name_ref, false)
}

fn expect_name_ref<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    match ast.node(id) {
        Node::RuleRef => Ok(()),
        Node::VarRef | Node::FieldAccess { .. } => {
            let ty = type_of(ast, id, env)?;
            if ty != Ty::Rule {
                return Err(mismatch(Ty::Rule, ty, ast.span(id)));
            }
            Ok(())
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::ExpectedRuleName,
            span: ast.span(id),
        }),
    }
}

/// Check a `list_list_rule_t` config field. For literal lists-of-lists, validates
/// each inner list element-wise with `check_inner`. For non-literal expressions,
/// checks the overall type is `list_list_rule_t` (or a subtype thereof).
fn expect_list_list<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    check_inner: fn(&'src Ast<'src>, NodeId, &mut TypeEnv<'src>) -> Result<(), TypeError>,
) -> Result<(), TypeError> {
    if let Node::List(range) = ast.node(id) {
        for &child in ast.child_slice(*range) {
            check_inner(ast, child, env)?;
        }
        return Ok(());
    }
    let ty = type_of(ast, id, env)?;
    if ty == Ty::ListListRule || ty == Ty::ListListStr {
        return Ok(());
    }
    Err(mismatch(Ty::ListListRule, ty, ast.span(id)))
}

/// Inner element for `precedences`: a name ref or a string literal.
fn expect_name_or_str<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    if matches!(ast.node(id), Node::StringLit | Node::RawStringLit { .. }) {
        return Ok(());
    }
    expect_name_ref(ast, id, env)
}

/// Check a `precedences` inner list: each element must be a name-ref or string literal.
fn expect_precedence_group<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    expect_list(ast, id, env, expect_name_or_str, true)
}

fn expect_reserved<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    // Object literal: each field value must be a rule list
    if let Node::Object(range) = ast.node(id) {
        for &(_, val_id) in ast.get_object(*range) {
            expect_rule_list(ast, val_id, env)?;
        }
        return Ok(());
    }
    // Non-literal: must be inherited (base.reserved -> Object(ListRule))
    let ty = type_of(ast, id, env)?;
    if ty == Ty::Object(InnerTy::ListRule) {
        return Ok(());
    }
    Err(TypeError {
        kind: TypeErrorKind::ExpectedReservedConfig,
        span: ast.span(id),
    })
}

fn expect_rule<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    let ty = type_of(ast, id, env)?;
    if !ty.is_rule_like() {
        return Err(mismatch(Ty::Rule, ty, ast.span(id)));
    }
    Ok(())
}

// -- Type inference --

fn type_of<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<Ty, TypeError> {
    let span = ast.span(id);
    match ast.node(id) {
        Node::IntLit(_) => Ok(Ty::Int),
        Node::StringLit | Node::RawStringLit { .. } => Ok(Ty::Str),
        Node::RuleRef | Node::Blank => Ok(Ty::Rule),
        Node::Inherit { .. } => Ok(Ty::Grammar),
        Node::Ident => unreachable!(),
        Node::VarRef => {
            let name = ast.text(span);
            env.get_var(name).ok_or_else(|| TypeError {
                kind: TypeErrorKind::UnresolvedVariable(name.to_string()),
                span,
            })
        }
        Node::Neg(inner) => {
            let ty = type_of(ast, *inner, env)?;
            if ty != Ty::Int {
                return Err(mismatch(Ty::Int, ty, span));
            }
            Ok(Ty::Int)
        }
        Node::Append { left, right } => {
            let l_empty =
                matches!(ast.node(*left), Node::List(r) if ast.child_slice(*r).is_empty());
            let r_empty =
                matches!(ast.node(*right), Node::List(r) if ast.child_slice(*r).is_empty());
            let lt = if l_empty {
                None
            } else {
                Some(type_of(ast, *left, env)?)
            };
            let rt = if r_empty {
                None
            } else {
                Some(type_of(ast, *right, env)?)
            };
            match (lt, rt) {
                (Some(l), Some(r)) => {
                    if !l.is_list() {
                        return Err(TypeError {
                            kind: TypeErrorKind::AppendRequiresList(l),
                            span: ast.span(*left),
                        });
                    }
                    if !r.is_list() {
                        return Err(mismatch(l, r, ast.span(*right)));
                    }
                    if l == r {
                        Ok(l)
                    } else if l.is_compatible(r) {
                        Ok(r)
                    } else if r.is_compatible(l) {
                        Ok(l)
                    } else {
                        Err(mismatch(l, r, ast.span(*right)))
                    }
                }
                (Some(t), None) | (None, Some(t)) => {
                    if !t.is_list() {
                        return Err(TypeError {
                            kind: TypeErrorKind::AppendRequiresList(t),
                            span,
                        });
                    }
                    Ok(t)
                }
                (None, None) => Err(TypeError {
                    kind: TypeErrorKind::EmptyListNeedsAnnotation,
                    span,
                }),
            }
        }
        Node::FieldAccess { obj, field } => {
            let obj_ty = type_of(ast, *obj, env)?;
            let field_name = ast.node_text(*field);
            match obj_ty {
                Ty::Object(inner) => {
                    // Validate field exists if the object came from a known variable
                    if matches!(ast.node(*obj), Node::VarRef) {
                        let var_name = ast.text(ast.span(*obj));
                        if let Some(fields) = env.get_object_fields(var_name)
                            && !fields.contains(&field_name)
                        {
                            return Err(TypeError {
                                kind: TypeErrorKind::FieldNotFound {
                                    field: field_name.to_string(),
                                    on_type: obj_ty,
                                },
                                span: ast.span(*field),
                            });
                        }
                    }
                    Ok(match inner {
                        InnerTy::Rule => Ty::Rule,
                        InnerTy::Str => Ty::Str,
                        InnerTy::Int => Ty::Int,
                        InnerTy::ListRule => Ty::ListRule,
                        InnerTy::ListStr => Ty::ListStr,
                    })
                }
                Ty::Grammar => match field_name {
                    "extras" | "externals" | "inline" | "supertypes" => Ok(Ty::ListRule),
                    "conflicts" | "precedences" => Ok(Ty::ListListRule),
                    "reserved" => Ok(Ty::Object(InnerTy::ListRule)),
                    "word" => Ok(Ty::Rule),
                    _ => Err(TypeError {
                        kind: TypeErrorKind::UnknownConfigField(field_name.to_string()),
                        span: ast.span(*field),
                    }),
                },
                _ => Err(TypeError {
                    kind: TypeErrorKind::FieldAccessOnNonObject(obj_ty),
                    span: ast.span(*obj),
                }),
            }
        }
        Node::RuleInline { obj, .. } => {
            let obj_ty = type_of(ast, *obj, env)?;
            if obj_ty != Ty::Grammar {
                return Err(TypeError {
                    kind: TypeErrorKind::ConfigAccessOnNonGrammar(obj_ty),
                    span: ast.span(*obj),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Object(range) => {
            let fields = ast.get_object(*range);
            if fields.is_empty() {
                return Err(TypeError {
                    kind: TypeErrorKind::CannotInferType,
                    span,
                });
            }
            let first_ty = type_of(ast, fields[0].1, env)?;
            let inner = match first_ty {
                Ty::Rule => InnerTy::Rule,
                Ty::Str => InnerTy::Str,
                Ty::Int => InnerTy::Int,
                Ty::ListRule => InnerTy::ListRule,
                Ty::ListStr => InnerTy::ListStr,
                _ => {
                    return Err(TypeError {
                        kind: TypeErrorKind::InvalidObjectValue(first_ty),
                        span: ast.span(fields[0].1),
                    });
                }
            };
            for &(_, val_id) in &fields[1..] {
                let ty = type_of(ast, val_id, env)?;
                if ty != first_ty {
                    return Err(mismatch(first_ty, ty, ast.span(val_id)));
                }
            }
            Ok(Ty::Object(inner))
        }
        Node::List(range) => {
            let items = ast.child_slice(*range);
            if items.is_empty() {
                return Err(TypeError {
                    kind: TypeErrorKind::EmptyListNeedsAnnotation,
                    span,
                });
            }
            let first = type_of(ast, items[0], env)?;
            for &item_id in &items[1..] {
                let ty = type_of(ast, item_id, env)?;
                if ty != first {
                    return Err(TypeError {
                        kind: TypeErrorKind::ListElementTypeMismatch { first, got: ty },
                        span: ast.span(item_id),
                    });
                }
            }
            match first {
                Ty::Str => Ok(Ty::ListStr),
                _ if first.is_rule_like() => Ok(Ty::ListRule),
                Ty::Int => Ok(Ty::ListInt),
                Ty::ListRule => Ok(Ty::ListListRule),
                Ty::ListStr => Ok(Ty::ListListStr),
                Ty::ListInt => Ok(Ty::ListListInt),
                _ => Err(TypeError {
                    kind: TypeErrorKind::InvalidListElement,
                    span: ast.span(items[0]),
                }),
            }
        }
        Node::Tuple(_) => {
            // Tuples are only valid as elements of for-loop lists.
            // They're checked structurally in check_for_expr.
            // If we reach here, it's a tuple in an invalid context.
            Err(TypeError {
                kind: TypeErrorKind::CannotInferType,
                span,
            })
        }
        Node::Concat(range) => {
            for &part in ast.child_slice(*range) {
                let ty = type_of(ast, part, env)?;
                if ty != Ty::Str {
                    return Err(mismatch(Ty::Str, ty, ast.span(part)));
                }
            }
            Ok(Ty::Str)
        }
        Node::DynRegex { pattern, flags } => {
            let pt = type_of(ast, *pattern, env)?;
            if pt != Ty::Str {
                return Err(mismatch(Ty::Str, pt, ast.span(*pattern)));
            }
            if let Some(fid) = flags {
                let ft = type_of(ast, *fid, env)?;
                if ft != Ty::Str {
                    return Err(mismatch(Ty::Str, ft, ast.span(*fid)));
                }
            }
            Ok(Ty::Rule)
        }
        Node::Seq(range) | Node::Choice(range) => {
            for &member in ast.child_slice(*range) {
                let ty = type_of(ast, member, env)?;
                if ty != Ty::Spread && !ty.is_rule_like() {
                    return Err(mismatch(Ty::Rule, ty, ast.span(member)));
                }
            }
            Ok(Ty::Rule)
        }
        Node::Repeat(inner)
        | Node::Repeat1(inner)
        | Node::Optional(inner)
        | Node::Token(inner)
        | Node::TokenImmediate(inner) => {
            expect_rule(ast, *inner, env)?;
            Ok(Ty::Rule)
        }
        Node::Field { content, .. } | Node::Reserved { content, .. } => {
            expect_rule(ast, *content, env)?;
            Ok(Ty::Rule)
        }
        Node::Alias { content, target } => {
            expect_rule(ast, *content, env)?;
            let target_ty = type_of(ast, *target, env)?;
            let is_valid =
                matches!(ast.node(*target), Node::RuleRef | Node::VarRef) || target_ty == Ty::Str;
            if !is_valid {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidAliasTarget(target_ty),
                    span: ast.span(*target),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Prec { value, content }
        | Node::PrecLeft { value, content }
        | Node::PrecRight { value, content } => {
            let vt = type_of(ast, *value, env)?;
            if vt != Ty::Int && vt != Ty::Str {
                return Err(mismatch(Ty::Int, vt, ast.span(*value)));
            }
            expect_rule(ast, *content, env)?;
            Ok(Ty::Rule)
        }
        Node::PrecDynamic { value, content } => {
            let vt = type_of(ast, *value, env)?;
            if vt != Ty::Int {
                return Err(mismatch(Ty::Int, vt, ast.span(*value)));
            }
            expect_rule(ast, *content, env)?;
            Ok(Ty::Rule)
        }
        Node::For(for_idx) => {
            check_for_expr(ast, *for_idx, env)?;
            Ok(Ty::Spread)
        }
        Node::Call { name, args } => {
            let fn_name = ast.node_text(*name);
            let sig = env.fns.get(fn_name).ok_or_else(|| TypeError {
                kind: TypeErrorKind::UndefinedFunction(fn_name.to_string()),
                span,
            })?;
            let return_ty = sig.return_ty;
            let n_params = sig.params.len();
            let args = ast.child_slice(*args);
            if args.len() != n_params {
                return Err(TypeError {
                    kind: TypeErrorKind::ArgCountMismatch {
                        fn_name: fn_name.to_string(),
                        expected: n_params,
                        got: args.len(),
                    },
                    span,
                });
            }
            for (i, &arg_id) in args.iter().enumerate() {
                let arg_ty = type_of(ast, arg_id, env)?;
                let param_ty = env.fns[fn_name].params[i];
                if !arg_ty.is_compatible(param_ty) {
                    return Err(mismatch(param_ty, arg_ty, ast.span(arg_id)));
                }
            }
            Ok(return_ty)
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span,
        }),
    }
}

// -- For-loop checking --

fn check_for_expr<'src>(
    ast: &'src Ast<'src>,
    for_idx: ForId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    let config = ast.get_for(for_idx);

    // Check if iterable is a literal list of tuples (for destructuring)
    if let Node::List(range) = ast.node(config.iterable) {
        let items = ast.child_slice(*range);
        let has_tuples = items
            .first()
            .is_some_and(|&id| matches!(ast.node(id), Node::Tuple(_)));
        if has_tuples {
            let mut scope = env.scoped();
            for &item_id in items {
                check_for_tuple_element(ast, item_id, config, &mut scope)?;
            }
            return expect_rule(ast, config.body, &mut scope);
        }
    }

    // Single binding over a flat list
    if config.bindings.len() != 1 {
        return Err(TypeError {
            kind: TypeErrorKind::ForRequiresTuples,
            span: ast.span(config.iterable),
        });
    }
    let iter_ty = type_of(ast, config.iterable, env)?;
    if !iter_ty.is_list() {
        return Err(TypeError {
            kind: TypeErrorKind::ForRequiresList(iter_ty),
            span: ast.span(config.iterable),
        });
    }
    let (name_span, ty_id) = &config.bindings[0];
    let declared = resolve_type_annotation(ast, *ty_id)?;
    let elem_ty = match iter_ty {
        Ty::ListRule => Ty::Rule,
        Ty::ListStr => Ty::Str,
        Ty::ListInt => Ty::Int,
        Ty::ListListRule => Ty::ListRule,
        Ty::ListListStr => Ty::ListStr,
        Ty::ListListInt => Ty::ListInt,
        _ => unreachable!(),
    };
    if !elem_ty.is_compatible(declared) {
        return Err(mismatch(declared, elem_ty, *name_span));
    }
    let mut scope = env.scoped();
    scope.insert_var(ast.text(*name_span), declared);
    expect_rule(ast, config.body, &mut scope)
}

fn check_for_tuple_element<'src>(
    ast: &'src Ast<'src>,
    item_id: NodeId,
    config: &super::ast::ForConfig,
    scope: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    let Node::Tuple(range) = ast.node(item_id) else {
        return Err(TypeError {
            kind: TypeErrorKind::ForRequiresTuples,
            span: ast.span(item_id),
        });
    };
    let elems = ast.child_slice(*range);
    if elems.len() != config.bindings.len() {
        return Err(TypeError {
            kind: TypeErrorKind::ForBindingCountMismatch {
                bindings: config.bindings.len(),
                tuple_elements: elems.len(),
            },
            span: ast.span(item_id),
        });
    }
    // Pop and re-push scope vars for each tuple (last iteration's bindings win)
    for (i, &(name_span, ty_id)) in config.bindings.iter().enumerate() {
        let declared = resolve_type_annotation(ast, ty_id)?;
        let actual = type_of(ast, elems[i], scope)?;
        if !actual.is_compatible(declared) {
            return Err(mismatch(declared, actual, name_span));
        }
        scope.insert_var(ast.text(name_span), declared);
    }
    Ok(())
}

// -- Helpers --

const fn mismatch(expected: Ty, got: Ty, span: Span) -> TypeError {
    TypeError {
        kind: TypeErrorKind::TypeMismatch { expected, got },
        span,
    }
}

// -- Error types --

#[derive(Debug, Serialize, Error)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum TypeErrorKind {
    TypeMismatch {
        expected: Ty,
        got: Ty,
    },
    UndefinedFunction(String),
    ArgCountMismatch {
        fn_name: String,
        expected: usize,
        got: usize,
    },
    FieldNotFound {
        field: String,
        on_type: Ty,
    },
    FieldAccessOnNonObject(Ty),
    ListElementTypeMismatch {
        first: Ty,
        got: Ty,
    },
    InvalidObjectValue(Ty),
    InvalidListElement,
    EmptyListNeedsAnnotation,
    ForRequiresList(Ty),
    ForRequiresTuples,
    ForBindingCountMismatch {
        bindings: usize,
        tuple_elements: usize,
    },
    ForBindingsNotTuple {
        bindings: usize,
        got: Ty,
    },
    UnresolvedVariable(String),
    AppendRequiresList(Ty),
    ConfigAccessOnNonGrammar(Ty),
    UnknownConfigField(String),
    InvalidAliasTarget(Ty),
    ExpectedRuleName,
    ExpectedReservedConfig,
    UnexpectedTopLevel,
    CannotInferType,
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            TypeErrorKind::TypeMismatch { expected, got } => {
                write!(f, "expected {expected}, got {got}")
            }
            TypeErrorKind::UndefinedFunction(n) => write!(f, "undefined function '{n}'"),
            TypeErrorKind::ArgCountMismatch {
                fn_name,
                expected,
                got,
            } => write!(
                f,
                "fn '{fn_name}': expected {expected} arguments, got {got}"
            ),
            TypeErrorKind::FieldNotFound { field, on_type } => {
                write!(f, "no field '{field}' on {on_type}")
            }
            TypeErrorKind::FieldAccessOnNonObject(ty) => {
                write!(f, "field access requires object or grammar_t, got {ty}")
            }
            TypeErrorKind::ListElementTypeMismatch { first, got } => {
                write!(f, "list elements have inconsistent types: {first} vs {got}")
            }
            TypeErrorKind::InvalidObjectValue(ty) => {
                write!(f, "object values must be rule_t, str_t, or int_t, got {ty}")
            }
            TypeErrorKind::InvalidListElement => {
                write!(
                    f,
                    "list elements must be rule_t, str_t, int_t, or a list type"
                )
            }
            TypeErrorKind::EmptyListNeedsAnnotation => {
                write!(f, "empty list requires a type annotation")
            }
            TypeErrorKind::ForRequiresList(got) => {
                write!(f, "for-expression requires a list, got {got}")
            }
            TypeErrorKind::ForRequiresTuples => {
                write!(f, "for with multiple bindings requires a list of tuples")
            }
            TypeErrorKind::ForBindingCountMismatch {
                bindings,
                tuple_elements,
            } => write!(
                f,
                "for has {bindings} bindings but tuples have {tuple_elements} elements"
            ),
            TypeErrorKind::ForBindingsNotTuple { bindings, got } => write!(
                f,
                "for has {bindings} bindings but list elements are {got}, not tuples"
            ),
            TypeErrorKind::UnresolvedVariable(n) => write!(f, "unresolved variable '{n}'"),
            TypeErrorKind::AppendRequiresList(got) => {
                write!(f, "append requires list arguments, got {got}")
            }
            TypeErrorKind::ConfigAccessOnNonGrammar(ty) => {
                write!(f, "'::' requires grammar_t, got {ty}")
            }
            TypeErrorKind::UnknownConfigField(n) => write!(f, "unknown grammar config field '{n}'"),
            TypeErrorKind::InvalidAliasTarget(got) => {
                write!(f, "alias target must be a name or string, got {got}")
            }
            TypeErrorKind::ExpectedRuleName => write!(f, "expected a rule name"),
            TypeErrorKind::ExpectedReservedConfig => {
                write!(f, "reserved config must be an object literal or inherited")
            }
            TypeErrorKind::UnexpectedTopLevel => write!(f, "unexpected node at top level"),
            TypeErrorKind::CannotInferType => write!(f, "cannot infer type of this expression"),
        }
    }
}
