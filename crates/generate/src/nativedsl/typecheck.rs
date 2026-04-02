//! Structural type checker for the native grammar DSL.
//!
//! Runs after name resolution and before lowering. Validates that:
//!
//! - Rule bodies produce rule-typed expressions
//! - Function arguments match parameter types
//! - Function bodies match declared return types
//! - `let` binding values match declared types
//! - `for` loop iterables are lists with element types matching bindings
//! - Combinator arguments have the correct types (`seq`/`choice` need rules,
//!   `concat` needs strings, `prec` needs int/string values, etc.)
//!
//! The type system is simple and structural. `str` is a subtype of `rule`
//! (string literals can appear anywhere a rule expression is expected).
//! `Object({field: int, ...})` is compatible with `int` if all fields are
//! `int`.

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use super::ast::{Ast, Node, NodeId, Span};

/// The type of a DSL expression.
///
/// The type system is intentionally simple: five base types plus structural
/// containers. `Str` is a subtype of `Rule` (see [`is_compatible`]).
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Ty {
    Int,
    Str,
    Rule,
    List(Box<Self>),
    Tuple(Vec<Self>),
    Object(Vec<(String, Self)>),
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Int => write!(f, "int"),
            Self::Str => write!(f, "str"),
            Self::Rule => write!(f, "rule"),
            Self::List(inner) => write!(f, "list<{inner}>"),
            Self::Tuple(elems) => {
                write!(f, "(")?;
                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{elem}")?;
                }
                write!(f, ")")
            }
            Self::Object(fields) => {
                write!(f, "{{")?;
                for (i, (key, val)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{key}: {val}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

/// Check whether `actual` is assignable to `expected`.
///
/// This is the subtyping relation: `Str` is compatible with `Rule`, and an
/// `Object` with all-int fields is compatible with `Int`.
fn is_compatible(actual: &Ty, expected: &Ty) -> bool {
    if actual == expected {
        return true;
    }
    match (actual, expected) {
        (Ty::Str, Ty::Rule) => true,
        (Ty::Object(fields), Ty::Int) => fields.iter().all(|(_, ty)| ty == &Ty::Int),
        _ => false,
    }
}

/// Returns `true` for types that can appear where a grammar rule is expected.
const fn is_rule_like(ty: &Ty) -> bool {
    matches!(ty, Ty::Rule | Ty::Str)
}

/// Returns `true` for types that can appear as precedence values.
const fn is_prec_like(ty: &Ty) -> bool {
    matches!(ty, Ty::Int | Ty::Str)
}

/// A function's type signature: parameter names/types and return type.
#[derive(Clone, Debug)]
struct FnSig {
    params: Vec<(String, Ty)>,
    return_ty: Ty,
}

/// The type environment: a scope stack for variables and a map of function
/// signatures.
///
/// Variable scopes form a stack - `push_scope`/`pop_scope` manage nesting
/// for function bodies and for-loop bodies. Lookups walk the stack from top
/// to bottom, implementing lexical scoping.
struct TypeEnv<'src> {
    var_scopes: Vec<FxHashMap<&'src str, Ty>>,
    fns: FxHashMap<&'src str, FnSig>,
}

impl<'src> TypeEnv<'src> {
    fn new() -> Self {
        Self {
            var_scopes: vec![FxHashMap::default()],
            fns: FxHashMap::default(),
        }
    }

    fn insert_var(&mut self, name: &'src str, ty: Ty) {
        self.var_scopes.last_mut().unwrap().insert(name, ty);
    }

    fn get_var(&self, name: &str) -> Option<&Ty> {
        for scope in self.var_scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn push_scope(&mut self) {
        self.var_scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.var_scopes.pop();
    }
}

/// Convert an AST type node (`TypeRule`, `TypeStr`, etc.) to a [`Ty`].
fn ast_type_to_ty(ast: &Ast<'_>, ty_id: NodeId) -> Result<Ty, TypeError> {
    match ast.node(ty_id) {
        Node::TypeInt => Ok(Ty::Int),
        Node::TypeStr => Ok(Ty::Str),
        Node::TypeRule => Ok(Ty::Rule),
        Node::TypeList(inner) => Ok(Ty::List(Box::new(ast_type_to_ty(ast, *inner)?))),
        Node::TypeTuple(elems) => {
            let tys = elems
                .iter()
                .map(|&id| ast_type_to_ty(ast, id))
                .collect::<Result<_, _>>()?;
            Ok(Ty::Tuple(tys))
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span: ast.span(ty_id),
        }),
    }
}

/// Run type checking on the entire AST.
///
/// First pre-scans all top-level items to register rule names as `Ty::Rule`
/// and function signatures. Then checks each item body.
///
/// # Errors
///
/// Returns [`TypeError`] on type mismatches, undefined functions, wrong
/// argument counts, etc.
pub fn check<'src>(ast: &'src Ast<'src>) -> Result<(), TypeError> {
    let mut env = TypeEnv::new();

    // Pre-scan: register all rule names and fn signatures.
    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Rule { name, .. } => {
                env.insert_var(ast.text(*name), Ty::Rule);
            }
            Node::Fn(fn_idx) => {
                let config = ast.get_fn(*fn_idx);
                let params: Vec<(String, Ty)> = config
                    .params
                    .iter()
                    .map(|param| {
                        Ok((
                            ast.text(param.name).to_string(),
                            ast_type_to_ty(ast, param.ty)?,
                        ))
                    })
                    .collect::<Result<_, TypeError>>()?;
                let return_ty = ast_type_to_ty(ast, config.return_ty)?;
                env.fns
                    .insert(ast.text(config.name), FnSig { params, return_ty });
            }
            _ => {}
        }
    }

    for &item_id in &ast.root_items {
        check_item(ast, item_id, &mut env)?;
    }
    Ok(())
}

/// Type-check a single top-level item (rule, let, fn).
fn check_item<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    let span = ast.span(id);
    match ast.node(id) {
        Node::Grammar => Ok(()),
        Node::Let { name, ty, value } => {
            let inferred = type_of(ast, *value, env)?;
            if let Some(ty_id) = ty {
                let declared = ast_type_to_ty(ast, *ty_id)?;
                if !is_compatible(&inferred, &declared) {
                    let name_str = ast.text(*name);
                    return Err(TypeError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: declared,
                            got: inferred,
                            context: format!("let '{name_str}'"),
                        },
                        span: ast.span(*value),
                    });
                }
            }
            env.insert_var(ast.text(*name), inferred);
            Ok(())
        }
        Node::Fn(fn_idx) => {
            let config = ast.get_fn(*fn_idx);
            let params: Vec<(String, Ty)> = config
                .params
                .iter()
                .map(|param| {
                    Ok((
                        ast.text(param.name).to_string(),
                        ast_type_to_ty(ast, param.ty)?,
                    ))
                })
                .collect::<Result<_, TypeError>>()?;

            let return_ty = ast_type_to_ty(ast, config.return_ty)?;
            let body = config.body;
            let fn_name_span = config.name;

            env.fns.insert(
                ast.text(fn_name_span),
                FnSig {
                    params,
                    return_ty: return_ty.clone(),
                },
            );

            env.push_scope();
            for param in &config.params {
                let ty = ast_type_to_ty(ast, param.ty)?;
                env.insert_var(ast.text(param.name), ty);
            }

            let body_ty = type_of(ast, body, env)?;
            env.pop_scope();

            if !is_compatible(&body_ty, &return_ty) {
                let fn_name = ast.text(fn_name_span);
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: return_ty,
                        got: body_ty,
                        context: format!("fn '{fn_name}'"),
                    },
                    span: ast.span(body),
                });
            }
            Ok(())
        }
        Node::Rule { body, .. } => {
            let ty = type_of(ast, *body, env)?;
            if !is_rule_like(&ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: ty,
                        context: "rule body".to_string(),
                    },
                    span: ast.span(*body),
                });
            }
            Ok(())
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::UnexpectedTopLevel,
            span,
        }),
    }
}

/// Validate function call arguments against the function's signature.
///
/// Checks argument count and that each argument type is compatible with the
/// corresponding parameter type.
fn check_call_args<'src>(
    ast: &'src Ast<'src>,
    fn_name: &str,
    args: &[NodeId],
    sig: &FnSig,
    call_span: Span,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    if args.len() != sig.params.len() {
        return Err(TypeError {
            kind: TypeErrorKind::ArgCountMismatch {
                fn_name: fn_name.to_string(),
                expected: sig.params.len(),
                got: args.len(),
            },
            span: call_span,
        });
    }
    for (&arg_id, (param_name, param_ty)) in args.iter().zip(sig.params.iter()) {
        let arg_ty = type_of(ast, arg_id, env)?;
        if !is_compatible(&arg_ty, param_ty) {
            return Err(TypeError {
                kind: TypeErrorKind::TypeMismatch {
                    expected: param_ty.clone(),
                    got: arg_ty,
                    context: format!("fn '{fn_name}': argument '{param_name}'"),
                },
                span: ast.span(arg_id),
            });
        }
    }
    Ok(())
}

/// Infer the type of an expression node.
///
/// Recursively types sub-expressions and validates that combinator arguments
/// have the correct types. Returns the resulting [`Ty`].
fn type_of<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<Ty, TypeError> {
    let span = ast.span(id);
    match ast.node(id) {
        Node::IntLit(_) => Ok(Ty::Int),
        Node::StringLit | Node::RawStringLit { .. } => Ok(Ty::Str),
        Node::RuleRef(_) | Node::Blank => Ok(Ty::Rule),
        Node::Ident(_) => {
            unreachable!("resolve pass should have replaced all Ident nodes")
        }
        Node::VarRef(span) => {
            let name = ast.text(*span);
            env.get_var(name).cloned().ok_or_else(|| TypeError {
                kind: TypeErrorKind::UnresolvedVariable(name.to_string()),
                span: *span,
            })
        }
        Node::FieldAccess { obj, field } => {
            let obj_ty = type_of(ast, *obj, env)?;
            let field_name = ast.text(*field);
            match &obj_ty {
                Ty::Object(fields) => fields
                    .iter()
                    .find(|(k, _)| k == field_name)
                    .map(|(_, ty)| ty.clone())
                    .ok_or_else(|| TypeError {
                        kind: TypeErrorKind::FieldNotFound {
                            field: field_name.to_string(),
                            on_type: obj_ty.clone(),
                        },
                        span,
                    }),
                _ => Err(TypeError {
                    kind: TypeErrorKind::FieldAccessOnNonObject(obj_ty.clone()),
                    span: ast.span(*obj),
                }),
            }
        }
        Node::Neg(inner) => {
            let ty = type_of(ast, *inner, env)?;
            if ty != Ty::Int {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Int,
                        got: ty,
                        context: "unary minus".to_string(),
                    },
                    span,
                });
            }
            Ok(Ty::Int)
        }
        Node::Object(fields) => {
            let tys = fields
                .iter()
                .map(|(key_span, val_id)| {
                    let ty = type_of(ast, *val_id, env)?;
                    Ok((ast.text(*key_span).to_string(), ty))
                })
                .collect::<Result<Vec<_>, TypeError>>()?;
            Ok(Ty::Object(tys))
        }
        Node::List(items) => {
            if items.is_empty() {
                return Ok(Ty::List(Box::new(Ty::Rule)));
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
            Ok(Ty::List(Box::new(first)))
        }
        Node::Tuple(items) => {
            let tys = items
                .iter()
                .map(|&id| type_of(ast, id, env))
                .collect::<Result<_, _>>()?;
            Ok(Ty::Tuple(tys))
        }
        Node::Concat(parts) => {
            for &part in parts {
                let ty = type_of(ast, part, env)?;
                if ty != Ty::Str {
                    return Err(TypeError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: Ty::Str,
                            got: ty,
                            context: "concat() argument".to_string(),
                        },
                        span: ast.span(part),
                    });
                }
            }
            Ok(Ty::Str)
        }
        Node::DynRegex { pattern, flags } => {
            let ty = type_of(ast, *pattern, env)?;
            if ty != Ty::Str {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Str,
                        got: ty,
                        context: "regex() pattern".to_string(),
                    },
                    span,
                });
            }
            if let Some(flags_id) = flags {
                let flags_ty = type_of(ast, *flags_id, env)?;
                if flags_ty != Ty::Str {
                    return Err(TypeError {
                        kind: TypeErrorKind::TypeMismatch {
                            expected: Ty::Str,
                            got: flags_ty,
                            context: "regex() flags".to_string(),
                        },
                        span,
                    });
                }
            }
            Ok(Ty::Rule)
        }
        Node::Seq(members) | Node::Choice(members) => {
            for &member in members {
                if matches!(ast.node(member), Node::For(_)) {
                    check_for_expr(ast, member, env)?;
                } else {
                    let ty = type_of(ast, member, env)?;
                    if !is_rule_like(&ty) {
                        return Err(TypeError {
                            kind: TypeErrorKind::TypeMismatch {
                                expected: Ty::Rule,
                                got: ty,
                                context: "seq/choice member".to_string(),
                            },
                            span: ast.span(member),
                        });
                    }
                }
            }
            Ok(Ty::Rule)
        }
        Node::Repeat(inner) | Node::Repeat1(inner) | Node::Optional(inner) => {
            let ty = type_of(ast, *inner, env)?;
            if !is_rule_like(&ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: ty,
                        context: "repeat/optional argument".to_string(),
                    },
                    span: ast.span(*inner),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Field { content, .. } => {
            let ty = type_of(ast, *content, env)?;
            if !is_rule_like(&ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: ty,
                        context: "field() content".to_string(),
                    },
                    span: ast.span(*content),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Alias { content, target } => {
            let ty = type_of(ast, *content, env)?;
            if !is_rule_like(&ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: ty,
                        context: "alias() content".to_string(),
                    },
                    span: ast.span(*content),
                });
            }
            let target_ty = type_of(ast, *target, env)?;
            if !matches!(ast.node(*target), Node::RuleRef(_) | Node::VarRef(_))
                && target_ty != Ty::Str
                && target_ty != Ty::Rule
            {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: target_ty,
                        context: "alias target".to_string(),
                    },
                    span: ast.span(*target),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Token(inner) | Node::TokenImmediate(inner) => {
            let ty = type_of(ast, *inner, env)?;
            if !is_rule_like(&ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: ty,
                        context: "token() argument".to_string(),
                    },
                    span: ast.span(*inner),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Prec { value, content }
        | Node::PrecLeft { value, content }
        | Node::PrecRight { value, content } => {
            let value_ty = type_of(ast, *value, env)?;
            if !is_prec_like(&value_ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Int,
                        got: value_ty,
                        context: "precedence value".to_string(),
                    },
                    span: ast.span(*value),
                });
            }
            let content_ty = type_of(ast, *content, env)?;
            if !is_rule_like(&content_ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: content_ty,
                        context: "prec() content".to_string(),
                    },
                    span: ast.span(*content),
                });
            }
            Ok(Ty::Rule)
        }
        Node::PrecDynamic { value, content } => {
            let value_ty = type_of(ast, *value, env)?;
            if value_ty != Ty::Int {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Int,
                        got: value_ty,
                        context: "prec.dynamic() value".to_string(),
                    },
                    span: ast.span(*value),
                });
            }
            let content_ty = type_of(ast, *content, env)?;
            if !is_rule_like(&content_ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: content_ty,
                        context: "prec.dynamic() content".to_string(),
                    },
                    span: ast.span(*content),
                });
            }
            Ok(Ty::Rule)
        }
        Node::Reserved { content, .. } => {
            let ty = type_of(ast, *content, env)?;
            if !is_rule_like(&ty) {
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: Ty::Rule,
                        got: ty,
                        context: "reserved() content".to_string(),
                    },
                    span: ast.span(*content),
                });
            }
            Ok(Ty::Rule)
        }
        Node::For(_) => {
            check_for_expr(ast, id, env)?;
            Ok(Ty::Rule)
        }
        Node::Call { name, args } => {
            let fn_name = ast.text(*name);
            let sig = env
                .fns
                .get(fn_name)
                .ok_or_else(|| TypeError {
                    kind: TypeErrorKind::UndefinedFunction(fn_name.to_string()),
                    span,
                })?
                .clone();
            check_call_args(ast, fn_name, args, &sig, span, env)?;
            Ok(sig.return_ty)
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span,
        }),
    }
}

/// Type-check a `for` expression.
///
/// Validates that the iterable is a list, that bindings match element types
/// (with tuple destructuring for multi-binding), and that the body produces
/// a rule.
fn check_for_expr<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
) -> Result<(), TypeError> {
    let span = ast.span(id);
    let Node::For(for_idx) = ast.node(id) else {
        return Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span,
        });
    };

    let config = ast.get_for(*for_idx);
    let iterable = config.iterable;
    let body = config.body;
    let bindings: Vec<_> = config
        .bindings
        .iter()
        .map(|(span, ty)| (*span, *ty))
        .collect();

    let iter_ty = type_of(ast, iterable, env)?;
    let elem_ty = match &iter_ty {
        Ty::List(inner) => inner.as_ref().clone(),
        _ => {
            return Err(TypeError {
                kind: TypeErrorKind::ForRequiresList(iter_ty.clone()),
                span: ast.span(iterable),
            });
        }
    };

    env.push_scope();
    if let [(name_span, ty_id)] = &bindings[..] {
        let declared = ast_type_to_ty(ast, *ty_id)?;
        if !is_compatible(&elem_ty, &declared) {
            let binding_name = ast.text(*name_span);
            env.pop_scope();
            return Err(TypeError {
                kind: TypeErrorKind::TypeMismatch {
                    expected: declared,
                    got: elem_ty,
                    context: format!("for binding '{binding_name}'"),
                },
                span,
            });
        }
        env.insert_var(ast.text(*name_span), declared);
    } else {
        let Ty::Tuple(elem_tys) = &elem_ty else {
            env.pop_scope();
            return Err(TypeError {
                kind: TypeErrorKind::ForBindingsNotTuple {
                    bindings: bindings.len(),
                    got: elem_ty,
                },
                span,
            });
        };
        if bindings.len() != elem_tys.len() {
            env.pop_scope();
            return Err(TypeError {
                kind: TypeErrorKind::ForBindingCountMismatch {
                    bindings: bindings.len(),
                    tuple_elements: elem_tys.len(),
                },
                span,
            });
        }
        for ((name_span, ty_id), actual_ty) in bindings.iter().zip(elem_tys.iter()) {
            let declared = ast_type_to_ty(ast, *ty_id)?;
            if !is_compatible(actual_ty, &declared) {
                let binding_name = ast.text(*name_span);
                env.pop_scope();
                return Err(TypeError {
                    kind: TypeErrorKind::TypeMismatch {
                        expected: declared,
                        got: actual_ty.clone(),
                        context: format!("for binding '{binding_name}'"),
                    },
                    span,
                });
            }
            env.insert_var(ast.text(*name_span), declared);
        }
    }

    let body_ty = type_of(ast, body, env)?;
    env.pop_scope();
    if !is_rule_like(&body_ty) {
        return Err(TypeError {
            kind: TypeErrorKind::TypeMismatch {
                expected: Ty::Rule,
                got: body_ty,
                context: "for-expression body".to_string(),
            },
            span: ast.span(body),
        });
    }
    Ok(())
}

/// An error encountered during type checking.
#[derive(Debug, Serialize, Error)]
pub struct TypeError {
    pub kind: TypeErrorKind,
    pub span: Span,
}

/// The specific kind of type error.
#[derive(Debug, Serialize)]
pub enum TypeErrorKind {
    TypeMismatch {
        expected: Ty,
        got: Ty,
        context: String,
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
    ForRequiresList(Ty),
    ForBindingCountMismatch {
        bindings: usize,
        tuple_elements: usize,
    },
    ForBindingsNotTuple {
        bindings: usize,
        got: Ty,
    },
    UnresolvedVariable(String),
    UnexpectedTopLevel,
    CannotInferType,
    MissingFnReturnType(String),
}

impl std::fmt::Display for TypeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            TypeErrorKind::TypeMismatch {
                expected,
                got,
                context,
            } => {
                write!(f, "{context}: expected {expected}, got {got}")
            }
            TypeErrorKind::UndefinedFunction(name) => write!(f, "undefined function '{name}'"),
            TypeErrorKind::ArgCountMismatch {
                fn_name,
                expected,
                got,
            } => {
                write!(
                    f,
                    "fn '{fn_name}': expected {expected} arguments, got {got}"
                )
            }
            TypeErrorKind::FieldNotFound { field, on_type } => {
                write!(f, "no field '{field}' on {on_type}")
            }
            TypeErrorKind::FieldAccessOnNonObject(ty) => {
                write!(f, "field access on non-object type {ty}")
            }
            TypeErrorKind::ListElementTypeMismatch { first, got } => {
                write!(f, "list elements have inconsistent types: {first} vs {got}")
            }
            TypeErrorKind::ForRequiresList(got) => {
                write!(f, "for-expression requires a list, got {got}")
            }
            TypeErrorKind::ForBindingCountMismatch {
                bindings,
                tuple_elements,
            } => {
                write!(
                    f,
                    "for has {bindings} bindings but tuples have {tuple_elements} elements"
                )
            }
            TypeErrorKind::ForBindingsNotTuple { bindings, got } => {
                write!(
                    f,
                    "for has {bindings} bindings but list elements are {got}, not tuples"
                )
            }
            TypeErrorKind::UnresolvedVariable(name) => write!(f, "unresolved variable '{name}'"),
            TypeErrorKind::UnexpectedTopLevel => write!(f, "unexpected node at top level"),
            TypeErrorKind::CannotInferType => write!(f, "cannot infer type of this expression"),
            TypeErrorKind::MissingFnReturnType(name) => {
                write!(f, "fn '{name}' has no return type annotation")
            }
        }
    }
}
