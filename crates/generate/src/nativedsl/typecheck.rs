//! Structural type checker for the native grammar DSL.
//!
//! Runs after name resolution and before lowering. `Str` is a subtype of
//! `Rule` (string literals can appear where rules are expected).
//! Internally uses interned type IDs ([`TyId`]) to avoid heap allocations;
//! the public [`Ty`] enum is only constructed for error messages.

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use super::ast::{Ast, Node, NodeId, Span};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub enum Ty {
    Int,
    Str,
    Rule,
    Grammar,
    Spread,
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
            Self::Spread => write!(f, "for-loop expansion"),
            Self::Grammar => write!(f, "grammar"),
            Self::List(inner) => write!(f, "list<{inner}>"),
            Self::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{e}")?;
                }
                write!(f, ")")
            }
            Self::Object(fields) => {
                write!(f, "{{")?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{k}: {v}")?;
                }
                write!(f, "}}")
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TyId(u32);

impl TyId {
    const INT: Self = Self(0);
    const STR: Self = Self(1);
    const RULE: Self = Self(2);
    const GRAMMAR: Self = Self(3);
    const SPREAD: Self = Self(4);
}

enum TyEntry {
    Int,
    Str,
    Rule,
    Grammar,
    Spread,
    List(TyId),
    Tuple { start: u32, len: u16 },
    Object { start: u32, len: u16 },
}

struct TyPool<'src> {
    entries: Vec<TyEntry>,
    tuple_elems: Vec<TyId>,
    object_fields: Vec<(&'src str, TyId)>,
}

impl<'src> TyPool<'src> {
    fn new() -> Self {
        Self {
            entries: vec![
                TyEntry::Int,
                TyEntry::Str,
                TyEntry::Rule,
                TyEntry::Grammar,
                TyEntry::Spread,
            ],
            tuple_elems: Vec::new(),
            object_fields: Vec::new(),
        }
    }

    fn list(&mut self, inner: TyId) -> TyId {
        let id = TyId(self.entries.len() as u32);
        self.entries.push(TyEntry::List(inner));
        id
    }

    fn tuple(&mut self, elems: &[TyId]) -> TyId {
        let start = self.tuple_elems.len() as u32;
        let len = elems.len() as u16;
        self.tuple_elems.extend_from_slice(elems);
        let id = TyId(self.entries.len() as u32);
        self.entries.push(TyEntry::Tuple { start, len });
        id
    }

    fn object(&mut self, fields: Vec<(&'src str, TyId)>) -> TyId {
        let start = self.object_fields.len() as u32;
        let len = fields.len() as u16;
        self.object_fields.extend(fields);
        let id = TyId(self.entries.len() as u32);
        self.entries.push(TyEntry::Object { start, len });
        id
    }

    fn get(&self, id: TyId) -> &TyEntry {
        &self.entries[id.0 as usize]
    }

    fn eq(&self, a: TyId, b: TyId) -> bool {
        if a == b {
            return true;
        }
        match (self.get(a), self.get(b)) {
            (TyEntry::List(a), TyEntry::List(b)) => self.eq(*a, *b),
            (TyEntry::Tuple { start: sa, len: la }, TyEntry::Tuple { start: sb, len: lb }) => {
                la == lb && {
                    let ae = &self.tuple_elems[*sa as usize..*sa as usize + *la as usize];
                    let be = &self.tuple_elems[*sb as usize..*sb as usize + *lb as usize];
                    ae.iter().zip(be).all(|(a, b)| self.eq(*a, *b))
                }
            }
            (TyEntry::Object { start: sa, len: la }, TyEntry::Object { start: sb, len: lb }) => {
                la == lb && {
                    let af = &self.object_fields[*sa as usize..*sa as usize + *la as usize];
                    let bf = &self.object_fields[*sb as usize..*sb as usize + *lb as usize];
                    af.iter()
                        .zip(bf)
                        .all(|((ka, ta), (kb, tb))| ka == kb && self.eq(*ta, *tb))
                }
            }
            _ => false,
        }
    }

    fn is_compatible(&self, actual: TyId, expected: TyId) -> bool {
        self.eq(actual, expected)
            || (actual == TyId::STR && expected == TyId::RULE)
            || (expected == TyId::INT
                && matches!(self.get(actual), TyEntry::Object { start: s, len: l } if {
                    self.object_fields[*s as usize..*s as usize + *l as usize]
                        .iter().all(|(_, ty)| *ty == TyId::INT)
                }))
    }

    fn is_list(&self, id: TyId) -> Option<TyId> {
        match self.get(id) {
            TyEntry::List(inner) => Some(*inner),
            _ => None,
        }
    }

    fn tuple_elems(&self, id: TyId) -> Option<&[TyId]> {
        match self.get(id) {
            TyEntry::Tuple { start: s, len: l } => {
                Some(&self.tuple_elems[*s as usize..*s as usize + *l as usize])
            }
            _ => None,
        }
    }

    fn object_field(&self, id: TyId, name: &str) -> Option<TyId> {
        match self.get(id) {
            TyEntry::Object { start: s, len: l } => self.object_fields
                [*s as usize..*s as usize + *l as usize]
                .iter()
                .find(|(k, _)| *k == name)
                .map(|(_, ty)| *ty),
            _ => None,
        }
    }

    /// Convert interned type to public `Ty` (only for error messages).
    fn to_ty(&self, id: TyId) -> Ty {
        match self.get(id) {
            TyEntry::Int => Ty::Int,
            TyEntry::Str => Ty::Str,
            TyEntry::Rule => Ty::Rule,
            TyEntry::Grammar => Ty::Grammar,
            TyEntry::Spread => Ty::Spread,
            TyEntry::List(inner) => Ty::List(Box::new(self.to_ty(*inner))),
            TyEntry::Tuple { start: s, len: l } => Ty::Tuple(
                self.tuple_elems[*s as usize..*s as usize + *l as usize]
                    .iter()
                    .map(|&id| self.to_ty(id))
                    .collect(),
            ),
            TyEntry::Object { start: s, len: l } => Ty::Object(
                self.object_fields[*s as usize..*s as usize + *l as usize]
                    .iter()
                    .map(|(k, v)| (k.to_string(), self.to_ty(*v)))
                    .collect(),
            ),
        }
    }

    fn mismatch(&self, expected: TyId, got: TyId, span: Span) -> TypeError {
        TypeError {
            kind: TypeErrorKind::TypeMismatch {
                expected: self.to_ty(expected),
                got: self.to_ty(got),
            },
            span,
        }
    }
}

fn is_rule_like(ty: TyId) -> bool {
    ty == TyId::RULE || ty == TyId::STR
}
fn is_prec_like(ty: TyId) -> bool {
    ty == TyId::INT || ty == TyId::STR
}

// -- Type environment --

#[derive(Clone, Debug)]
struct FnSig {
    params: Vec<(String, TyId)>,
    return_ty: TyId,
}

struct TypeEnv<'src> {
    var_scopes: Vec<FxHashMap<&'src str, TyId>>,
    fns: FxHashMap<&'src str, FnSig>,
}

impl<'src> TypeEnv<'src> {
    fn new() -> Self {
        Self {
            var_scopes: vec![FxHashMap::default()],
            fns: FxHashMap::default(),
        }
    }
    fn insert_var(&mut self, name: &'src str, ty: TyId) {
        self.var_scopes.last_mut().unwrap().insert(name, ty);
    }
    fn get_var(&self, name: &str) -> Option<TyId> {
        self.var_scopes
            .iter()
            .rev()
            .find_map(|s| s.get(name).copied())
    }
    fn push_scope(&mut self) {
        self.var_scopes.push(FxHashMap::default());
    }
    fn pop_scope(&mut self) {
        self.var_scopes.pop();
    }
}

fn ast_type_to_ty(ast: &Ast<'_>, ty_id: NodeId, pool: &mut TyPool<'_>) -> Result<TyId, TypeError> {
    match ast.node(ty_id) {
        Node::TypeInt => Ok(TyId::INT),
        Node::TypeStr => Ok(TyId::STR),
        Node::TypeRule => Ok(TyId::RULE),
        Node::TypeList(inner) => {
            let inner = ast_type_to_ty(ast, *inner, pool)?;
            Ok(pool.list(inner))
        }
        Node::TypeTuple(range) => {
            let elems: Vec<TyId> = ast
                .child_slice(*range)
                .iter()
                .map(|&id| ast_type_to_ty(ast, id, pool))
                .collect::<Result<_, _>>()?;
            Ok(pool.tuple(&elems))
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span: ast.span(ty_id),
        }),
    }
}

fn parse_fn_params(
    ast: &Ast<'_>,
    config: &super::ast::FnConfig,
    pool: &mut TyPool<'_>,
) -> Result<Vec<(String, TyId)>, TypeError> {
    config
        .params
        .iter()
        .map(|p| {
            Ok((
                ast.node_text(p.name).to_string(),
                ast_type_to_ty(ast, p.ty, pool)?,
            ))
        })
        .collect()
}

// -- Entry point --

pub fn check<'src>(ast: &'src Ast<'src>) -> Result<(), TypeError> {
    let mut env = TypeEnv::new();
    let mut pool = TyPool::new();
    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Rule { name, .. } | Node::OverrideRule { name, .. } => {
                env.insert_var(ast.node_text(*name), TyId::RULE);
            }
            Node::Fn(fn_idx) => {
                let config = ast.get_fn(*fn_idx);
                let params = parse_fn_params(ast, config, &mut pool)?;
                let return_ty = ast_type_to_ty(ast, config.return_ty, &mut pool)?;
                env.fns
                    .insert(ast.node_text(config.name), FnSig { params, return_ty });
            }
            _ => {}
        }
    }
    for &item_id in &ast.root_items {
        check_item(ast, item_id, &mut env, &mut pool)?;
    }
    Ok(())
}

fn check_item<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    match ast.node(id) {
        Node::Grammar => {
            let config = ast.context.grammar_config.as_ref().unwrap();
            // extras/externals accept any rule expression (regexp, strings, etc.)
            for id in [config.extras, config.externals].into_iter().flatten() {
                expect_rule_list(ast, id, env, pool)?;
            }
            // inline/supertypes/word only accept rule name references
            for id in [config.inline, config.supertypes].into_iter().flatten() {
                expect_name_list(ast, id, env, pool)?;
            }
            if let Some(id) = config.word {
                expect_name_ref(ast, id, env, pool)?;
            }
            if let Some(id) = config.reserved {
                expect_reserved(ast, id, env, pool)?;
            }
            Ok(())
        }
        Node::Let { name, ty, value } => {
            let inferred = type_of(ast, *value, env, pool)?;
            if let Some(ty_id) = ty {
                let declared = ast_type_to_ty(ast, *ty_id, pool)?;
                if !pool.is_compatible(inferred, declared) {
                    return Err(pool.mismatch(declared, inferred, ast.span(*value)));
                }
            }
            env.insert_var(ast.node_text(*name), inferred);
            Ok(())
        }
        Node::Fn(fn_idx) => {
            let config = ast.get_fn(*fn_idx);
            let fn_name = ast.node_text(config.name);
            // Registered in the first pass of `check()`
            let sig = &env.fns[fn_name];
            let return_ty = sig.return_ty;
            let n_params = sig.params.len();
            env.push_scope();
            for i in 0..n_params {
                // Re-index each iteration so the borrow is temporary
                let ty = env.fns[fn_name].params[i].1;
                env.insert_var(ast.node_text(config.params[i].name), ty);
            }
            let body_ty = type_of(ast, config.body, env, pool)?;
            env.pop_scope();
            if !pool.is_compatible(body_ty, return_ty) {
                return Err(pool.mismatch(return_ty, body_ty, ast.span(config.body)));
            }
            Ok(())
        }
        Node::Rule { body, .. } | Node::OverrideRule { body, .. } => {
            expect_rule(ast, *body, env, pool)
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::UnexpectedTopLevel,
            span: ast.span(id),
        }),
    }
}

/// Check that an expression evaluates to a list where each element is rule-like.
/// For literal lists, checks each element individually to permit mixed rule/str
/// (e.g. `[regexp("\\s"), " "]` in extras).
fn expect_rule_list<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    // Literal lists: check each element (allows mixed rule/str)
    if let Node::List(range) = ast.node(id) {
        for &child in ast.child_slice(*range) {
            expect_rule(ast, child, env, pool)?;
        }
        return Ok(());
    }
    // Non-literal: check overall type is list<rule> or list<str>
    let ty = type_of(ast, id, env, pool)?;
    let rule_list = pool.list(TyId::RULE);
    if pool.is_compatible(ty, rule_list) {
        return Ok(());
    }
    let str_list = pool.list(TyId::STR);
    if pool.is_compatible(ty, str_list) {
        return Ok(());
    }
    Err(pool.mismatch(rule_list, ty, ast.span(id)))
}

/// Check that an expression is a rule name reference (`RuleRef`, `VarRef`,
/// or field access like `base.word`). Rejects complex rule expressions
/// like `regexp(...)` or `prec(...)` that have type `rule` but aren't names.
fn expect_name_ref<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    match ast.node(id) {
        Node::RuleRef => Ok(()),
        Node::VarRef | Node::FieldAccess { .. } => {
            let ty = type_of(ast, id, env, pool)?;
            if ty != TyId::RULE {
                return Err(pool.mismatch(TyId::RULE, ty, ast.span(id)));
            }
            Ok(())
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::ExpectedRuleName,
            span: ast.span(id),
        }),
    }
}

/// Check that an expression is a list of rule name references.
fn expect_name_list<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    if let Node::List(range) = ast.node(id) {
        for &child in ast.child_slice(*range) {
            expect_name_ref(ast, child, env, pool)?;
        }
        return Ok(());
    }
    // Non-literal: allow list<rule> variables (e.g. c.inline)
    let ty = type_of(ast, id, env, pool)?;
    let rule_list = pool.list(TyId::RULE);
    if pool.is_compatible(ty, rule_list) {
        return Ok(());
    }
    Err(pool.mismatch(rule_list, ty, ast.span(id)))
}

/// Check that the `reserved` config expression is valid.
/// Accepts an object literal `{ name: [words] }` where each value is a rule list,
/// or an expression like `base.reserved` that evaluates to the reserved type.
fn expect_reserved<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    // Object literal: each field value must be a rule list
    if let Node::Object(range) = ast.node(id) {
        for &(_, val_id) in ast.get_object(*range) {
            expect_rule_list(ast, val_id, env, pool)?;
        }
        return Ok(());
    }
    // Otherwise, check the type matches list<{name: str, words: list<rule>}>
    let ty = type_of(ast, id, env, pool)?;
    let words = pool.list(TyId::RULE);
    let set = pool.object(vec![("name", TyId::STR), ("words", words)]);
    let expected = pool.list(set);
    if !pool.is_compatible(ty, expected) {
        return Err(pool.mismatch(expected, ty, ast.span(id)));
    }
    Ok(())
}

fn expect_rule<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    let ty = type_of(ast, id, env, pool)?;
    if !is_rule_like(ty) {
        return Err(pool.mismatch(TyId::RULE, ty, ast.span(id)));
    }
    Ok(())
}

fn check_call_args<'src>(
    ast: &'src Ast<'src>,
    fn_name: &str,
    args: &[NodeId],
    sig: &FnSig,
    call_span: Span,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
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
    for (&arg_id, (_, param_ty)) in args.iter().zip(sig.params.iter()) {
        let arg_ty = type_of(ast, arg_id, env, pool)?;
        if !pool.is_compatible(arg_ty, *param_ty) {
            return Err(pool.mismatch(*param_ty, arg_ty, ast.span(arg_id)));
        }
    }
    Ok(())
}

fn type_of<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<TyId, TypeError> {
    let span = ast.span(id);
    match ast.node(id) {
        Node::IntLit(_) => Ok(TyId::INT),
        Node::StringLit | Node::RawStringLit { .. } => Ok(TyId::STR),
        Node::RuleRef | Node::Blank => Ok(TyId::RULE),
        Node::Inherit { .. } => Ok(TyId::GRAMMAR),
        Node::Append { left, right } => {
            let (lt, rt) = (
                type_of(ast, *left, env, pool)?,
                type_of(ast, *right, env, pool)?,
            );
            match (pool.is_list(lt), pool.is_list(rt)) {
                (Some(_), Some(_)) if pool.eq(lt, rt) => Ok(lt),
                (Some(_), _) => Err(pool.mismatch(lt, rt, ast.span(*right))),
                _ => Err(TypeError {
                    kind: TypeErrorKind::AppendRequiresList(pool.to_ty(lt)),
                    span: ast.span(*left),
                }),
            }
        }
        Node::Ident => unreachable!(),
        Node::VarRef => {
            let name = ast.text(span);
            env.get_var(name).ok_or_else(|| TypeError {
                kind: TypeErrorKind::UnresolvedVariable(name.to_string()),
                span,
            })
        }
        Node::FieldAccess { obj, field } => {
            let obj_ty = type_of(ast, *obj, env, pool)?;
            let field_name = ast.node_text(*field);
            if let Some(ty) = pool.object_field(obj_ty, field_name) {
                return Ok(ty);
            }
            if obj_ty == TyId::GRAMMAR {
                return match field_name {
                    "extras" | "externals" | "inline" | "supertypes" => Ok(pool.list(TyId::RULE)),
                    "conflicts" | "precedences" => {
                        let lr = pool.list(TyId::RULE);
                        Ok(pool.list(lr))
                    }
                    "word" => Ok(TyId::RULE),
                    "reserved" => {
                        let words = pool.list(TyId::RULE);
                        let set = pool.object(vec![("name", TyId::STR), ("words", words)]);
                        Ok(pool.list(set))
                    }
                    _ => Err(TypeError {
                        kind: TypeErrorKind::UnknownConfigField(field_name.to_string()),
                        span: ast.span(*field),
                    }),
                };
            }
            if matches!(pool.get(obj_ty), TyEntry::Object { .. }) {
                return Err(TypeError {
                    kind: TypeErrorKind::FieldNotFound {
                        field: field_name.to_string(),
                        on_type: pool.to_ty(obj_ty),
                    },
                    span,
                });
            }
            Err(TypeError {
                kind: TypeErrorKind::FieldAccessOnNonObject(pool.to_ty(obj_ty)),
                span: ast.span(*obj),
            })
        }
        Node::RuleInline { obj, .. } => {
            let obj_ty = type_of(ast, *obj, env, pool)?;
            if obj_ty != TyId::GRAMMAR {
                return Err(TypeError {
                    kind: TypeErrorKind::ConfigAccessOnNonGrammar(pool.to_ty(obj_ty)),
                    span: ast.span(*obj),
                });
            }
            Ok(TyId::RULE)
        }
        Node::Neg(inner) => {
            let ty = type_of(ast, *inner, env, pool)?;
            if ty != TyId::INT {
                return Err(pool.mismatch(TyId::INT, ty, span));
            }
            Ok(TyId::INT)
        }
        Node::Object(range) => {
            let fields: Vec<(&str, TyId)> = ast
                .get_object(*range)
                .iter()
                .map(|(ks, vid)| Ok((ast.text(*ks), type_of(ast, *vid, env, pool)?)))
                .collect::<Result<_, TypeError>>()?;
            Ok(pool.object(fields))
        }
        Node::List(range) => {
            let items = ast.child_slice(*range);
            if items.is_empty() {
                return Ok(pool.list(TyId::RULE));
            }
            let first = type_of(ast, items[0], env, pool)?;
            for &item_id in &items[1..] {
                let ty = type_of(ast, item_id, env, pool)?;
                if !pool.eq(ty, first) {
                    return Err(TypeError {
                        kind: TypeErrorKind::ListElementTypeMismatch {
                            first: pool.to_ty(first),
                            got: pool.to_ty(ty),
                        },
                        span: ast.span(item_id),
                    });
                }
            }
            Ok(pool.list(first))
        }
        Node::Tuple(range) => {
            let elems: Vec<TyId> = ast
                .child_slice(*range)
                .iter()
                .map(|&id| type_of(ast, id, env, pool))
                .collect::<Result<_, _>>()?;
            Ok(pool.tuple(&elems))
        }
        Node::Concat(range) => {
            for &part in ast.child_slice(*range) {
                let ty = type_of(ast, part, env, pool)?;
                if ty != TyId::STR {
                    return Err(pool.mismatch(TyId::STR, ty, ast.span(part)));
                }
            }
            Ok(TyId::STR)
        }
        Node::DynRegex { pattern, flags } => {
            let pt = type_of(ast, *pattern, env, pool)?;
            if pt != TyId::STR {
                return Err(pool.mismatch(TyId::STR, pt, ast.span(*pattern)));
            }
            if let Some(fid) = flags {
                let ft = type_of(ast, *fid, env, pool)?;
                if ft != TyId::STR {
                    return Err(pool.mismatch(TyId::STR, ft, ast.span(*fid)));
                }
            }
            Ok(TyId::RULE)
        }
        Node::Seq(range) | Node::Choice(range) => {
            for &member in ast.child_slice(*range) {
                let ty = type_of(ast, member, env, pool)?;
                if ty != TyId::SPREAD && !is_rule_like(ty) {
                    return Err(pool.mismatch(TyId::RULE, ty, ast.span(member)));
                }
            }
            Ok(TyId::RULE)
        }
        Node::Repeat(inner)
        | Node::Repeat1(inner)
        | Node::Optional(inner)
        | Node::Token(inner)
        | Node::TokenImmediate(inner) => {
            expect_rule(ast, *inner, env, pool)?;
            Ok(TyId::RULE)
        }
        Node::Field { content, .. } | Node::Reserved { content, .. } => {
            expect_rule(ast, *content, env, pool)?;
            Ok(TyId::RULE)
        }
        Node::Alias { content, target } => {
            expect_rule(ast, *content, env, pool)?;
            let target_ty = type_of(ast, *target, env, pool)?;
            // Alias target must be a named symbol or a string literal
            let is_valid_target =
                matches!(ast.node(*target), Node::RuleRef | Node::VarRef) || target_ty == TyId::STR;
            if !is_valid_target {
                return Err(TypeError {
                    kind: TypeErrorKind::InvalidAliasTarget(pool.to_ty(target_ty)),
                    span: ast.span(*target),
                });
            }
            Ok(TyId::RULE)
        }
        Node::Prec { value, content }
        | Node::PrecLeft { value, content }
        | Node::PrecRight { value, content } => {
            let vt = type_of(ast, *value, env, pool)?;
            if !is_prec_like(vt) {
                return Err(pool.mismatch(TyId::INT, vt, ast.span(*value)));
            }
            expect_rule(ast, *content, env, pool)?;
            Ok(TyId::RULE)
        }
        Node::PrecDynamic { value, content } => {
            let vt = type_of(ast, *value, env, pool)?;
            if vt != TyId::INT {
                return Err(pool.mismatch(TyId::INT, vt, ast.span(*value)));
            }
            expect_rule(ast, *content, env, pool)?;
            Ok(TyId::RULE)
        }
        Node::For(_) => {
            check_for_expr(ast, id, env, pool)?;
            Ok(TyId::SPREAD)
        }
        Node::Call { name, args } => {
            let fn_name = ast.node_text(*name);
            let sig = env
                .fns
                .get(fn_name)
                .ok_or_else(|| TypeError {
                    kind: TypeErrorKind::UndefinedFunction(fn_name.to_string()),
                    span,
                })?
                .clone();
            check_call_args(ast, fn_name, ast.child_slice(*args), &sig, span, env, pool)?;
            Ok(sig.return_ty)
        }
        _ => Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span,
        }),
    }
}

fn check_for_expr<'src>(
    ast: &'src Ast<'src>,
    id: NodeId,
    env: &mut TypeEnv<'src>,
    pool: &mut TyPool<'src>,
) -> Result<(), TypeError> {
    let span = ast.span(id);
    let Node::For(for_idx) = ast.node(id) else {
        return Err(TypeError {
            kind: TypeErrorKind::CannotInferType,
            span,
        });
    };
    let config = ast.get_for(*for_idx);
    let iter_ty = type_of(ast, config.iterable, env, pool)?;
    let Some(elem_ty) = pool.is_list(iter_ty) else {
        return Err(TypeError {
            kind: TypeErrorKind::ForRequiresList(pool.to_ty(iter_ty)),
            span: ast.span(config.iterable),
        });
    };

    env.push_scope();
    if let Some(elem_tys) = pool.tuple_elems(elem_ty) {
        let elem_tys = elem_tys.to_vec();
        if config.bindings.len() != elem_tys.len() {
            env.pop_scope();
            return Err(TypeError {
                kind: TypeErrorKind::ForBindingCountMismatch {
                    bindings: config.bindings.len(),
                    tuple_elements: elem_tys.len(),
                },
                span,
            });
        }
        for ((name_span, ty_id), &actual_ty) in config.bindings.iter().zip(elem_tys.iter()) {
            let declared = ast_type_to_ty(ast, *ty_id, pool)?;
            if !pool.is_compatible(actual_ty, declared) {
                env.pop_scope();
                return Err(pool.mismatch(declared, actual_ty, span));
            }
            env.insert_var(ast.text(*name_span), declared);
        }
    } else if let [(name_span, ty_id)] = config.bindings.as_slice() {
        let declared = ast_type_to_ty(ast, *ty_id, pool)?;
        if !pool.is_compatible(elem_ty, declared) {
            env.pop_scope();
            return Err(pool.mismatch(declared, elem_ty, span));
        }
        env.insert_var(ast.text(*name_span), declared);
    } else {
        env.pop_scope();
        return Err(TypeError {
            kind: TypeErrorKind::ForBindingsNotTuple {
                bindings: config.bindings.len(),
                got: pool.to_ty(elem_ty),
            },
            span,
        });
    }

    let body_ty = type_of(ast, config.body, env, pool)?;
    env.pop_scope();
    if !is_rule_like(body_ty) {
        return Err(pool.mismatch(TyId::RULE, body_ty, ast.span(config.body)));
    }
    Ok(())
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
    AppendRequiresList(Ty),
    ConfigAccessOnNonGrammar(Ty),
    UnknownConfigField(String),
    InvalidAliasTarget(Ty),
    ExpectedRuleName,
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
                write!(f, "'::' requires a grammar value, got {ty}")
            }
            TypeErrorKind::UnknownConfigField(n) => write!(f, "unknown grammar config field '{n}'"),
            TypeErrorKind::InvalidAliasTarget(got) => {
                write!(f, "alias target must be a name or string, got {got}")
            }
            TypeErrorKind::ExpectedRuleName => {
                write!(f, "expected a rule name")
            }
            TypeErrorKind::UnexpectedTopLevel => write!(f, "unexpected node at top level"),
            TypeErrorKind::CannotInferType => write!(f, "cannot infer type of this expression"),
        }
    }
}
