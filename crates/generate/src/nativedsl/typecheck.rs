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

use super::ast::{Ast, ChildRange, ForId, Node, NodeId, Span};
use super::scope_stack::ScopeStack;

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
    /// Result of `inherit(...)` or `import(...)`: a loaded module namespace.
    /// Members accessed via `::`. The `u8` indexes into the module list.
    Module(u8),
    Spread,
    /// Result of `print(...)`: no value. Purely internal - users cannot
    /// annotate this type. Rejected anywhere a real value is expected.
    Void,
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
            Self::Module(_) => f.write_str("module_t"),
            Self::Spread => f.write_str("spread_t"),
            Self::Void => f.write_str("void_t"),
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
    pub vars: ScopeStack<'src, Ty>,
    pub fns: FxHashMap<&'src str, FnSig>,
    /// Field names for Object variables, keyed by variable name.
    pub object_fields: FxHashMap<&'src str, Vec<&'src str>>,
    /// Type environments of imported modules, indexed by `Ty::Module(idx)`.
    pub modules: Vec<TypeEnv<'src>>,
}

impl<'src> TypeEnv<'src> {
    fn new() -> Self {
        Self {
            modules: Vec::new(),
            vars: ScopeStack::new(),
            fns: FxHashMap::default(),
            object_fields: FxHashMap::default(),
        }
    }
    #[inline]
    fn insert_var(&mut self, name: &'src str, ty: Ty) {
        self.vars.insert(name, ty);
    }
    #[inline]
    fn insert_object(&mut self, name: &'src str, ty: Ty, fields: Vec<&'src str>) {
        self.insert_var(name, ty);
        self.object_fields.insert(name, fields);
    }
    #[inline]
    fn get_object_fields(&self, name: &str) -> Option<&[&'src str]> {
        self.object_fields.get(name).map(Vec::as_slice)
    }
    #[inline]
    fn get_var(&self, name: &str) -> Option<Ty> {
        self.vars.get(name)
    }
    fn scoped(&mut self) -> ScopeGuard<'_, 'src> {
        self.vars.push();
        ScopeGuard(self)
    }
}

struct ScopeGuard<'a, 'src>(&'a mut TypeEnv<'src>);

impl Drop for ScopeGuard<'_, '_> {
    fn drop(&mut self) {
        self.0.vars.pop();
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

fn resolve_type_annotation(ast: &Ast, id: NodeId) -> Result<Ty, TypeError> {
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
        Node::TypeVoid => Err(TypeError {
            kind: TypeErrorKind::VoidTypeNotAllowed,
            span: ast.span(id),
        }),
        Node::TypeSpread => Err(TypeError {
            kind: TypeErrorKind::SpreadTypeNotAllowed,
            span: ast.span(id),
        }),
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span: ast.span(id),
        }),
    }
}

// -- Entry point --

/// Run typechecking and return the type environment.
///
/// `modules` holds imported modules' type environments (indexed by
/// `Ty::Module(idx)`), populated by the import pre-pass in `mod.rs`.
pub fn check<'ast>(
    ast: &'ast Ast,
    modules: Vec<TypeEnv<'ast>>,
) -> Result<TypeEnv<'ast>, TypeError> {
    let mut env = TypeEnv::new();
    env.modules = modules;
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

fn check_item<'ast>(ast: &'ast Ast, id: NodeId, env: &mut TypeEnv<'ast>) -> Result<(), TypeError> {
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
                // `print(...)` and `for (...) { ... }` yield non-storable types
                // (`void_t` and for-expansion respectively). Reject them at the
                // let binding with a specific message; without this the
                // for-expansion case would panic during lowering.
                if inferred == Ty::Void {
                    return Err(TypeError {
                        kind: TypeErrorKind::CannotBindVoid,
                        span: ast.span(*value),
                    });
                }
                if inferred == Ty::Spread {
                    return Err(TypeError {
                        kind: TypeErrorKind::CannotBindSpread,
                        span: ast.span(*value),
                    });
                }
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
        Node::Print(arg) => {
            // The argument can have any concrete type, but not a for-expansion
            // (which doesn't produce a standalone value) or another `print`
            // call (void is nothing to print).
            let ty = type_of(ast, *arg, env)?;
            if ty == Ty::Void || ty == Ty::Spread {
                return Err(TypeError {
                    kind: TypeErrorKind::CannotPrintType(ty),
                    span: ast.span(*arg),
                });
            }
            Ok(())
        }
        // Unreachable: parse_item() only produces Grammar, Let, Fn, Rule,
        // OverrideRule, and Print nodes at root level.
        _ => unreachable!(),
    }
}

// -- Config field validators --

/// Check a list config field. For literal lists, checks each element with `check_elem`.
/// For non-literal expressions, checks the overall type is a list type.
fn expect_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    check_elem: fn(&'ast Ast, NodeId, &mut TypeEnv<'ast>) -> Result<(), TypeError>,
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

fn expect_rule_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
) -> Result<(), TypeError> {
    expect_list(ast, id, env, expect_rule, true)
}

fn expect_name_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
) -> Result<(), TypeError> {
    expect_list(ast, id, env, expect_name_ref, false)
}

fn expect_name_ref<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
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
fn expect_list_list<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
    check_inner: fn(&'ast Ast, NodeId, &mut TypeEnv<'ast>) -> Result<(), TypeError>,
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
fn expect_name_or_str<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
) -> Result<(), TypeError> {
    if matches!(ast.node(id), Node::StringLit | Node::RawStringLit { .. }) {
        return Ok(());
    }
    expect_name_ref(ast, id, env)
}

/// Check a `precedences` inner list: each element must be a name-ref or string literal.
fn expect_precedence_group<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
) -> Result<(), TypeError> {
    expect_list(ast, id, env, expect_name_or_str, true)
}

fn expect_reserved<'ast>(
    ast: &'ast Ast,
    id: NodeId,
    env: &mut TypeEnv<'ast>,
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

fn expect_rule<'ast>(ast: &'ast Ast, id: NodeId, env: &mut TypeEnv<'ast>) -> Result<(), TypeError> {
    let ty = type_of(ast, id, env)?;
    if !ty.is_rule_like() {
        return Err(mismatch(Ty::Rule, ty, ast.span(id)));
    }
    Ok(())
}

// -- Type inference --

fn type_of<'ast>(ast: &'ast Ast, id: NodeId, env: &mut TypeEnv<'ast>) -> Result<Ty, TypeError> {
    let span = ast.span(id);
    match ast.node(id) {
        Node::IntLit(_) => Ok(Ty::Int),
        Node::StringLit | Node::RawStringLit { .. } => Ok(Ty::Str),
        Node::RuleRef | Node::Blank => Ok(Ty::Rule),
        Node::Inherit { module, .. } | Node::Import { module, .. } => {
            let idx = module.expect("module index not set by loading pre-pass");
            Ok(Ty::Module(idx))
        }
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
        Node::Append { left, right } => type_of_append(ast, *left, *right, span, env),
        Node::FieldAccess { obj, field } => type_of_field_access(ast, *obj, *field, env),
        Node::QualifiedAccess { obj, member } => {
            let obj_ty = type_of(ast, *obj, env)?;
            match obj_ty {
                Ty::Module(idx) => type_of_import_access(ast, idx, *member, env),
                _ => Err(TypeError {
                    kind: TypeErrorKind::QualifiedAccessOnInvalidType(obj_ty),
                    span: ast.span(*obj),
                }),
            }
        }
        Node::Object(range) => type_of_object(ast, *range, span, env),
        Node::List(range) => type_of_list(ast, *range, span, env),
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
        Node::Call { name, args } => type_of_call(ast, *name, *args, span, env),
        Node::QualifiedCall(range) => type_of_qualified_call(ast, *range, span, env),
        // print() in expression position - returns Void which will be rejected
        // by the caller (e.g. CannotBindVoid in let, type mismatch in seq/choice)
        Node::Print(arg) => {
            type_of(ast, *arg, env)?;
            Ok(Ty::Void)
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span,
        }),
    }
}

// -- For-loop checking --

// -- type_of helpers (extracted from the main match) --

fn type_of_append<'ast>(
    ast: &'ast Ast,
    left: NodeId,
    right: NodeId,
    span: Span,
    env: &mut TypeEnv<'ast>,
) -> Result<Ty, TypeError> {
    let l_empty = matches!(ast.node(left), Node::List(r) if ast.child_slice(*r).is_empty());
    let r_empty = matches!(ast.node(right), Node::List(r) if ast.child_slice(*r).is_empty());
    let lt = if l_empty {
        None
    } else {
        Some(type_of(ast, left, env)?)
    };
    let rt = if r_empty {
        None
    } else {
        Some(type_of(ast, right, env)?)
    };
    match (lt, rt) {
        (Some(l), Some(r)) => {
            if !l.is_list() {
                return Err(TypeError {
                    kind: TypeErrorKind::AppendRequiresList(l),
                    span: ast.span(left),
                });
            }
            if !r.is_list() {
                return Err(mismatch(l, r, ast.span(right)));
            }
            if l == r {
                Ok(l)
            } else if l.is_compatible(r) {
                Ok(r)
            } else if r.is_compatible(l) {
                Ok(l)
            } else {
                Err(mismatch(l, r, ast.span(right)))
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

fn type_of_field_access<'ast>(
    ast: &'ast Ast,
    obj: NodeId,
    field: NodeId,
    env: &mut TypeEnv<'ast>,
) -> Result<Ty, TypeError> {
    let obj_ty = type_of(ast, obj, env)?;
    let field_name = ast.node_text(field);
    match obj_ty {
        Ty::Object(inner) => {
            if matches!(ast.node(obj), Node::VarRef) {
                let var_name = ast.text(ast.span(obj));
                if let Some(fields) = env.get_object_fields(var_name)
                    && !fields.contains(&field_name)
                {
                    return Err(TypeError {
                        kind: TypeErrorKind::FieldNotFound {
                            field: field_name.to_string(),
                            on_type: obj_ty,
                        },
                        span: ast.span(field),
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
        // Module values use :: for access, not . - this is caught at typecheck time
        _ => Err(TypeError {
            kind: TypeErrorKind::FieldAccessOnNonObject(obj_ty),
            span: ast.span(obj),
        }),
    }
}

fn type_of_import_access(
    ast: &Ast,
    idx: u8,
    member: NodeId,
    env: &mut TypeEnv<'_>,
) -> Result<Ty, TypeError> {
    let member_name = ast.node_text(member);
    let import_env = &env.modules[idx as usize];
    if let Some(ty) = import_env.vars.get(member_name) {
        return Ok(ty);
    }
    if import_env.fns.contains_key(member_name) {
        // Accessing a function without calling it - not valid as a value
        return Err(TypeError {
            kind: TypeErrorKind::ImportMemberNotFound(member_name.to_string()),
            span: ast.span(member),
        });
    }
    Err(TypeError {
        kind: TypeErrorKind::ImportMemberNotFound(member_name.to_string()),
        span: ast.span(member),
    })
}

fn type_of_qualified_call<'a>(
    ast: &'a Ast,
    range: ChildRange,
    span: Span,
    env: &mut TypeEnv<'a>,
) -> Result<Ty, TypeError> {
    let (obj, name, args) = ast.get_qualified_call(range);
    let obj_ty = type_of(ast, obj, env)?;
    let Ty::Module(idx) = obj_ty else {
        return Err(TypeError {
            kind: TypeErrorKind::QualifiedCallOnNonModule(obj_ty),
            span: ast.span(obj),
        });
    };
    let fn_name = ast.node_text(name);
    let import_env = &env.modules[idx as usize];
    let Some(sig) = import_env.fns.get(fn_name) else {
        return Err(TypeError {
            kind: TypeErrorKind::ImportFunctionNotFound(fn_name.to_string()),
            span: ast.span(name),
        });
    };
    // Clone sig data out before calling type_of (which borrows env mutably)
    let param_types: Vec<Ty> = sig.params.clone();
    let return_ty = sig.return_ty;
    let fn_name = fn_name.to_string();
    if args.len() != param_types.len() {
        return Err(TypeError {
            kind: TypeErrorKind::ArgCountMismatch {
                fn_name,
                expected: param_types.len(),
                got: args.len(),
            },
            span,
        });
    }
    for (i, &arg_id) in args.iter().enumerate() {
        let arg_ty = type_of(ast, arg_id, env)?;
        if !arg_ty.is_compatible(param_types[i]) {
            return Err(mismatch(param_types[i], arg_ty, ast.span(arg_id)));
        }
    }
    Ok(return_ty)
}

fn type_of_object<'ast>(
    ast: &'ast Ast,
    range: super::ast::ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
) -> Result<Ty, TypeError> {
    let fields = ast.get_object(range);
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

fn type_of_list<'ast>(
    ast: &'ast Ast,
    range: super::ast::ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
) -> Result<Ty, TypeError> {
    let items = ast.child_slice(range);
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

fn type_of_call<'ast>(
    ast: &'ast Ast,
    name: NodeId,
    args: super::ast::ChildRange,
    span: Span,
    env: &mut TypeEnv<'ast>,
) -> Result<Ty, TypeError> {
    let fn_name = ast.node_text(name);
    let sig = env.fns.get(fn_name).ok_or_else(|| TypeError {
        kind: TypeErrorKind::UndefinedFunction(fn_name.to_string()),
        span,
    })?;
    let return_ty = sig.return_ty;
    let n_params = sig.params.len();
    let args = ast.child_slice(args);
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

// -- For-loop checking --

fn check_for_expr<'ast>(
    ast: &'ast Ast,
    for_idx: ForId,
    env: &mut TypeEnv<'ast>,
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

fn check_for_tuple_element<'ast>(
    ast: &'ast Ast,
    item_id: NodeId,
    config: &super::ast::ForConfig,
    scope: &mut TypeEnv<'ast>,
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
    QualifiedAccessOnInvalidType(Ty),
    UnknownConfigField(String),
    InvalidAliasTarget(Ty),
    ExpectedRuleName,
    ExpectedReservedConfig,
    CannotInferType,
    VoidTypeNotAllowed,
    SpreadTypeNotAllowed,
    CannotBindVoid,
    CannotBindSpread,
    CannotPrintType(Ty),
    ImportMemberNotFound(String),
    ImportFunctionNotFound(String),
    QualifiedCallOnNonModule(Ty),
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
            TypeErrorKind::QualifiedAccessOnInvalidType(ty) => {
                write!(f, "'::' requires module_t, got {ty}")
            }
            TypeErrorKind::UnknownConfigField(n) => write!(f, "unknown grammar config field '{n}'"),
            TypeErrorKind::InvalidAliasTarget(got) => {
                write!(f, "alias target must be a name or string, got {got}")
            }
            TypeErrorKind::ExpectedRuleName => write!(f, "expected a rule name"),
            TypeErrorKind::ExpectedReservedConfig => {
                write!(f, "reserved config must be an object literal or inherited")
            }
            TypeErrorKind::CannotInferType => write!(f, "cannot infer type of this expression"),
            TypeErrorKind::VoidTypeNotAllowed => write!(
                f,
                "void_t is an internal type and cannot be used as a type annotation"
            ),
            TypeErrorKind::SpreadTypeNotAllowed => write!(
                f,
                "spread_t is an internal type and cannot be used as a type annotation"
            ),
            TypeErrorKind::CannotBindVoid => write!(
                f,
                "cannot bind 'print' to a variable: 'print' does not produce a value"
            ),
            TypeErrorKind::CannotBindSpread => write!(
                f,
                "cannot bind a for-loop expansion to a variable: for-loops are expanded inline into their enclosing list"
            ),
            TypeErrorKind::CannotPrintType(ty) => write!(f, "cannot print a value of type {ty}"),
            TypeErrorKind::ImportMemberNotFound(name) => {
                write!(f, "imported module has no member '{name}'")
            }
            TypeErrorKind::ImportFunctionNotFound(name) => {
                write!(f, "imported module has no function '{name}'")
            }
            TypeErrorKind::QualifiedCallOnNonModule(ty) => {
                write!(f, "'::' call requires module_t, got {ty}")
            }
        }
    }
}
