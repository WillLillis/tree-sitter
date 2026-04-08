//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].
//!
//! Walks the AST, evaluating expressions into an intermediate arena, then
//! converts to output [`InputGrammar`] / [`Rule`] types. Key types:
//! [`StringPool`] (interned strings), rule arena (`Vec<ARule>`), value arena
//! (`Vec<Value>`), and [`Evaluator`] (scope chain + function registry).

use std::num::NonZeroU32;

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use crate::grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType};
use crate::rules::{Precedence, Rule};

use super::ast::{Ast, FnConfig, ForConfig, Node, NodeId, Note, PrecEntry, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Str(NonZeroU32);

enum StrEntry {
    Unreachable,
    Source(Span),
    Owned(String),
}

struct StringPool<'src> {
    source: &'src str,
    entries: Vec<StrEntry>,
}

impl<'src> StringPool<'src> {
    fn new(source: &'src str) -> Self {
        Self {
            source,
            entries: vec![StrEntry::Unreachable],
        }
    }

    fn intern_span(&mut self, span: Span) -> Str {
        let id = Str(NonZeroU32::new(self.entries.len() as u32).unwrap());
        self.entries.push(StrEntry::Source(span));
        id
    }

    fn intern_owned(&mut self, s: String) -> Str {
        let id = Str(NonZeroU32::new(self.entries.len() as u32).unwrap());
        self.entries.push(StrEntry::Owned(s));
        id
    }

    fn resolve(&self, id: Str) -> &str {
        match &self.entries[id.0.get() as usize] {
            StrEntry::Source(span) => span.resolve(self.source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }

    fn to_string(&self, id: Str) -> String {
        self.resolve(id).to_string()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuleId(u32);

#[derive(Clone, Copy, Debug)]
enum APrec {
    Integer(i32),
    Name(Str),
}

enum ARule {
    Blank,
    String(Str),
    Pattern(Str, Option<Str>),
    NamedSymbol(Str),
    Seq(u32, u16),
    Choice(u32, u16),
    Repeat(RuleId),
    Prec(APrec, RuleId),
    PrecLeft(APrec, RuleId),
    PrecRight(APrec, RuleId),
    PrecDynamic(i32, RuleId),
    Field(Str, RuleId),
    Alias(Str, bool, RuleId),
    Token(RuleId),
    ImmediateToken(RuleId),
    Reserved(Str, RuleId),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueId(u32);

enum Value<'src> {
    Int(i32),
    Str(Str),
    Rule(RuleId),
    Object(FxHashMap<&'src str, ValueId>),
    List(Vec<ValueId>),
    Tuple(Vec<ValueId>),
    Grammar,
}

struct Evaluator<'src> {
    ast: &'src Ast<'src>,
    values: Vec<Value<'src>>,
    scopes: Vec<FxHashMap<&'src str, ValueId>>,
    fns: FxHashMap<&'src str, NodeId>,
    rules: Vec<ARule>,
    rule_children: Vec<RuleId>,
    strings: StringPool<'src>,
    base_grammar: Option<&'src InputGrammar>,
}

impl<'src> Evaluator<'src> {
    fn new(ast: &'src Ast<'src>, base_grammar: Option<&'src InputGrammar>) -> Self {
        Self {
            ast,
            values: Vec::new(),
            scopes: vec![FxHashMap::default()],
            fns: FxHashMap::default(),
            rules: Vec::new(),
            rule_children: Vec::new(),
            strings: StringPool::new(ast.source()),
            base_grammar,
        }
    }

    fn alloc_val(&mut self, val: Value<'src>) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(val);
        id
    }

    fn get_val(&self, id: ValueId) -> &Value<'src> {
        &self.values[id.0 as usize]
    }

    fn push_scope(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: &'src str, val: ValueId) {
        self.scopes.last_mut().unwrap().insert(name, val);
    }

    fn lookup(&self, name: &str) -> Option<ValueId> {
        self.scopes
            .iter()
            .rev()
            .find_map(|scope| scope.get(name).copied())
    }

    fn alloc_rule(&mut self, rule: ARule) -> RuleId {
        let id = RuleId(self.rules.len() as u32);
        self.rules.push(rule);
        id
    }

    const fn children_start(&self) -> u32 {
        self.rule_children.len() as u32
    }

    fn children_range(&self, start: u32, span: Span) -> Result<(u32, u16), LowerError> {
        let count = self.rule_children.len() as u32 - start;
        u16::try_from(count)
            .map(|len| (start, len))
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))
    }

    fn get_fn_config(&self, fn_id: NodeId) -> &'src FnConfig {
        match self.ast.node(fn_id) {
            Node::Fn(fn_idx) => self.ast.get_fn(*fn_idx),
            _ => unreachable!(),
        }
    }

    fn get_str_val(&self, id: ValueId) -> Str {
        match self.get_val(id) {
            Value::Str(s) => *s,
            _ => unreachable!(),
        }
    }

    fn build_rule(&self, id: RuleId) -> Rule {
        let s = &self.strings;
        match &self.rules[id.0 as usize] {
            ARule::Blank => Rule::Blank,
            ARule::String(sid) => Rule::String(s.to_string(*sid)),
            ARule::Pattern(p, f) => Rule::Pattern(
                s.to_string(*p),
                f.map_or_else(String::new, |f| s.to_string(f)),
            ),
            ARule::NamedSymbol(sid) => Rule::NamedSymbol(s.to_string(*sid)),
            ARule::Seq(start, len) | ARule::Choice(start, len) => {
                let children: Vec<Rule> = self.rule_children
                    [*start as usize..*start as usize + *len as usize]
                    .iter()
                    .map(|&id| self.build_rule(id))
                    .collect();
                if matches!(&self.rules[id.0 as usize], ARule::Seq(..)) {
                    Rule::seq(children)
                } else {
                    Rule::choice(children)
                }
            }
            ARule::Repeat(inner) => Rule::repeat(self.build_rule(*inner)),
            ARule::Prec(p, r) => Rule::prec(self.build_prec(*p), self.build_rule(*r)),
            ARule::PrecLeft(p, r) => Rule::prec_left(self.build_prec(*p), self.build_rule(*r)),
            ARule::PrecRight(p, r) => Rule::prec_right(self.build_prec(*p), self.build_rule(*r)),
            ARule::PrecDynamic(v, r) => Rule::prec_dynamic(*v, self.build_rule(*r)),
            ARule::Field(n, r) => Rule::field(s.to_string(*n), self.build_rule(*r)),
            ARule::Alias(v, named, r) => Rule::alias(self.build_rule(*r), s.to_string(*v), *named),
            ARule::Token(r) => Rule::token(self.build_rule(*r)),
            ARule::ImmediateToken(r) => Rule::immediate_token(self.build_rule(*r)),
            ARule::Reserved(ctx, r) => Rule::Reserved {
                rule: Box::new(self.build_rule(*r)),
                context_name: s.to_string(*ctx),
            },
        }
    }

    fn build_prec(&self, prec: APrec) -> Precedence {
        match prec {
            APrec::Integer(n) => Precedence::Integer(n),
            APrec::Name(sid) => Precedence::Name(self.strings.to_string(sid)),
        }
    }

    fn eval_rule_list(&mut self, id: NodeId) -> Result<Vec<Rule>, LowerError> {
        let vid = self.eval_expr(id)?;
        let Value::List(items) = self.get_val(vid) else {
            unreachable!()
        };
        items
            .iter()
            .map(|&v| match self.get_val(v) {
                Value::Rule(rid) => Ok(self.build_rule(*rid)),
                Value::Str(sid) => Ok(Rule::String(self.strings.to_string(*sid))),
                _ => unreachable!(),
            })
            .collect()
    }

    fn eval_name_list(&mut self, id: NodeId) -> Result<Vec<String>, LowerError> {
        let vid = self.eval_expr(id)?;
        let Value::List(items) = self.get_val(vid) else {
            unreachable!()
        };
        items
            .iter()
            .map(|&v| {
                let Value::Rule(rid) = *self.get_val(v) else {
                    unreachable!()
                };
                match &self.rules[rid.0 as usize] {
                    ARule::NamedSymbol(sid) => Ok(self.strings.to_string(*sid)),
                    _ => unreachable!(),
                }
            })
            .collect()
    }

    fn eval_rule_name(&mut self, id: NodeId) -> Result<String, LowerError> {
        let rid = self.lower_to_rule(id)?;
        match &self.rules[rid.0 as usize] {
            ARule::NamedSymbol(sid) => Ok(self.strings.to_string(*sid)),
            _ => unreachable!(),
        }
    }

    /// Import a config field from the base grammar as a DSL value.
    fn import_config_field(&mut self, field: &str) -> ValueId {
        let grammar = self.base_grammar.unwrap();
        match field {
            "extras" => self.import_rules_as_list(&grammar.extra_symbols),
            "externals" => self.import_rules_as_list(&grammar.external_tokens),
            "inline" => self.import_names_as_list(&grammar.variables_to_inline),
            "supertypes" => self.import_names_as_list(&grammar.supertype_symbols),
            "conflicts" => {
                let vals: Vec<ValueId> = grammar
                    .expected_conflicts
                    .iter()
                    .map(|g| self.import_names_as_list(g))
                    .collect();
                self.alloc_val(Value::List(vals))
            }
            "word" => {
                if let Some(name) = &grammar.word_token {
                    let sid = self.strings.intern_owned(name.clone());
                    let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                    self.alloc_val(Value::Rule(rid))
                } else {
                    let rid = self.alloc_rule(ARule::Blank);
                    self.alloc_val(Value::Rule(rid))
                }
            }
            _ => unreachable!(),
        }
    }

    fn import_rules_as_list(&mut self, rules_data: &[Rule]) -> ValueId {
        let vals: Vec<ValueId> = rules_data
            .iter()
            .map(|r| {
                let rid = self.import_rule(r);
                self.alloc_val(Value::Rule(rid))
            })
            .collect();
        self.alloc_val(Value::List(vals))
    }

    fn import_names_as_list(&mut self, names: &[String]) -> ValueId {
        let vals: Vec<ValueId> = names
            .iter()
            .map(|name| {
                let sid = self.strings.intern_owned(name.clone());
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                self.alloc_val(Value::Rule(rid))
            })
            .collect();
        self.alloc_val(Value::List(vals))
    }

    fn import_rule(&mut self, rule: &Rule) -> RuleId {
        match rule {
            Rule::Blank => self.alloc_rule(ARule::Blank),
            Rule::String(s) => {
                let sid = self.strings.intern_owned(s.clone());
                self.alloc_rule(ARule::String(sid))
            }
            Rule::Pattern(p, f) => {
                let pid = self.strings.intern_owned(p.clone());
                let fid = (!f.is_empty()).then(|| self.strings.intern_owned(f.clone()));
                self.alloc_rule(ARule::Pattern(pid, fid))
            }
            Rule::NamedSymbol(name) => {
                let sid = self.strings.intern_owned(name.clone());
                self.alloc_rule(ARule::NamedSymbol(sid))
            }
            Rule::Choice(members) | Rule::Seq(members) => {
                let imported: Vec<RuleId> = members.iter().map(|m| self.import_rule(m)).collect();
                let start = self.rule_children.len() as u32;
                self.rule_children.extend_from_slice(&imported);
                let len = imported.len() as u16;
                if matches!(rule, Rule::Choice(_)) {
                    self.alloc_rule(ARule::Choice(start, len))
                } else {
                    self.alloc_rule(ARule::Seq(start, len))
                }
            }
            Rule::Repeat(inner) => {
                let inner = self.import_rule(inner);
                self.alloc_rule(ARule::Repeat(inner))
            }
            Rule::Metadata {
                params,
                rule: inner,
            } => {
                let mut rid = self.import_rule(inner);
                if params.dynamic_precedence != 0 {
                    rid = self.alloc_rule(ARule::PrecDynamic(params.dynamic_precedence, rid));
                }
                if let Precedence::Integer(n) = &params.precedence {
                    let aprec = APrec::Integer(*n);
                    rid = self.import_prec_rule(aprec, rid, params.associativity);
                } else if let Precedence::Name(s) = &params.precedence {
                    let aprec = APrec::Name(self.strings.intern_owned(s.clone()));
                    rid = self.import_prec_rule(aprec, rid, params.associativity);
                }
                if let Some(crate::rules::Alias { value, is_named }) = &params.alias {
                    let sid = self.strings.intern_owned(value.clone());
                    rid = self.alloc_rule(ARule::Alias(sid, *is_named, rid));
                }
                if let Some(field_name) = &params.field_name {
                    let sid = self.strings.intern_owned(field_name.clone());
                    rid = self.alloc_rule(ARule::Field(sid, rid));
                }
                if params.is_token && params.is_main_token {
                    rid = self.alloc_rule(ARule::ImmediateToken(rid));
                } else if params.is_token {
                    rid = self.alloc_rule(ARule::Token(rid));
                }
                rid
            }
            Rule::Reserved {
                rule: inner,
                context_name,
            } => {
                let inner = self.import_rule(inner);
                let sid = self.strings.intern_owned(context_name.clone());
                self.alloc_rule(ARule::Reserved(sid, inner))
            }
            Rule::Symbol(_) => unreachable!(),
        }
    }

    fn import_prec_rule(
        &mut self,
        prec: APrec,
        inner: RuleId,
        assoc: Option<crate::rules::Associativity>,
    ) -> RuleId {
        match assoc {
            Some(crate::rules::Associativity::Left) => {
                self.alloc_rule(ARule::PrecLeft(prec, inner))
            }
            Some(crate::rules::Associativity::Right) => {
                self.alloc_rule(ARule::PrecRight(prec, inner))
            }
            None => self.alloc_rule(ARule::Prec(prec, inner)),
        }
    }
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
pub fn lower_with_base(
    ast: &Ast<'_>,
    base_grammar: Option<&InputGrammar>,
) -> Result<InputGrammar, LowerError> {
    let mut eval = Evaluator::new(ast, base_grammar);
    let mut rule_entries: Vec<(&str, RuleId)> = Vec::with_capacity(ast.root_items.len());
    let mut override_entries: Vec<(&str, RuleId, Span)> = Vec::new();

    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Grammar => {}
            Node::Let { name, value, .. } => {
                let val = eval.eval_expr(*value)?;
                eval.bind(ast.node_text(*name), val);
            }
            Node::Fn(fn_idx) => {
                let config = ast.get_fn(*fn_idx);
                eval.fns.insert(ast.node_text(config.name), item_id);
            }
            Node::Rule { name, body } => {
                let rule_id = eval.lower_to_rule(*body)?;
                rule_entries.push((ast.node_text(*name), rule_id));
            }
            Node::OverrideRule { name, body } => {
                let rule_id = eval.lower_to_rule(*body)?;
                override_entries.push((ast.node_text(*name), rule_id, ast.span(*name)));
            }
            _ => unreachable!(),
        }
    }

    let config = ast
        .context
        .grammar_config
        .as_ref()
        .ok_or_else(|| LowerError::new(LowerErrorKind::MissingGrammarBlock, Span::new(0, 0)))?;
    let language = config
        .language
        .as_ref()
        .ok_or_else(|| LowerError::new(LowerErrorKind::MissingLanguageField, Span::new(0, 0)))?;

    let variables = if let Some(base) = base_grammar {
        if !override_entries.is_empty() && base.variables.is_empty() {
            return Err(LowerError::new(
                LowerErrorKind::OverrideRuleNotFound(override_entries[0].0.to_string()),
                override_entries[0].2,
            ));
        }
        let mut vars: Vec<Variable> = base.variables.clone();
        for (name, rid, span) in &override_entries {
            if let Some(var) = vars.iter_mut().find(|v| v.name == *name) {
                var.rule = eval.build_rule(*rid);
            } else {
                return Err(LowerError::new(
                    LowerErrorKind::OverrideRuleNotFound(name.to_string()),
                    *span,
                ));
            }
        }
        for (name, rid) in &rule_entries {
            vars.push(Variable {
                name: name.to_string(),
                kind: VariableType::Named,
                rule: eval.build_rule(*rid),
            });
        }
        vars
    } else {
        if !override_entries.is_empty() {
            return Err(LowerError::new(
                LowerErrorKind::OverrideWithoutInherit,
                override_entries[0].2,
            ));
        }
        rule_entries
            .into_iter()
            .map(|(name, rid)| Variable {
                name: name.to_string(),
                kind: VariableType::Named,
                rule: eval.build_rule(rid),
            })
            .collect()
    };

    let extra_symbols = match config.extras {
        Some(id) => eval.eval_rule_list(id)?,
        None => base_grammar.map_or_else(Vec::new, |b| b.extra_symbols.clone()),
    };
    let external_tokens = match config.externals {
        Some(id) => eval.eval_rule_list(id)?,
        None => base_grammar.map_or_else(Vec::new, |b| b.external_tokens.clone()),
    };
    let variables_to_inline = match config.inline {
        Some(id) => eval.eval_name_list(id)?,
        None => base_grammar.map_or_else(Vec::new, |b| b.variables_to_inline.clone()),
    };
    let supertype_symbols = match config.supertypes {
        Some(id) => eval.eval_name_list(id)?,
        None => base_grammar.map_or_else(Vec::new, |b| b.supertype_symbols.clone()),
    };
    let word_token = match config.word {
        Some(id) => Some(eval.eval_rule_name(id)?),
        None => base_grammar.and_then(|b| b.word_token.clone()),
    };
    let expected_conflicts = if config.conflicts.is_empty() {
        base_grammar.map_or_else(Vec::new, |b| b.expected_conflicts.clone())
    } else {
        config
            .conflicts
            .iter()
            .map(|g| g.iter().map(|s| ast.text(*s).to_string()).collect())
            .collect()
    };
    let precedence_orderings = if config.precedences.is_empty() {
        base_grammar.map_or_else(Vec::new, |b| b.precedence_orderings.clone())
    } else {
        config
            .precedences
            .iter()
            .map(|g| {
                g.iter()
                    .map(|e| match e {
                        PrecEntry::Name(s) => PrecedenceEntry::Name(s.clone()),
                        PrecEntry::Symbol(s) => PrecedenceEntry::Symbol(ast.text(*s).to_string()),
                    })
                    .collect()
            })
            .collect()
    };
    let reserved_words = if config.reserved.is_empty() {
        base_grammar.map_or_else(Vec::new, |b| b.reserved_words.clone())
    } else {
        config
            .reserved
            .iter()
            .map(|rws| {
                let words = rws
                    .words
                    .iter()
                    .map(|&id| {
                        let rid = eval.lower_to_rule(id)?;
                        Ok(eval.build_rule(rid))
                    })
                    .collect::<Result<_, LowerError>>()?;
                Ok(ReservedWordContext {
                    name: ast.text(rws.name).to_string(),
                    reserved_words: words,
                })
            })
            .collect::<Result<_, LowerError>>()?
    };

    Ok(InputGrammar {
        name: language.clone(),
        variables,
        extra_symbols,
        expected_conflicts,
        precedence_orderings,
        external_tokens,
        variables_to_inline,
        supertype_symbols,
        word_token,
        reserved_words,
    })
}

impl Evaluator<'_> {
    fn eval_expr(&mut self, id: NodeId) -> Result<ValueId, LowerError> {
        let ast = self.ast;
        let span = ast.span(id);
        match ast.node(id) {
            Node::IntLit(n) => Ok(self.alloc_val(Value::Int(*n))),
            Node::StringLit => {
                let content = Span::new(span.start + 1, span.end - 1);
                let raw = ast.text(content);
                let sid = if memchr::memchr(b'\\', raw.as_bytes()).is_some() {
                    self.strings.intern_owned(unescape_string(raw))
                } else {
                    self.strings.intern_span(content)
                };
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let prefix = 2 + u32::from(*hash_count);
                let suffix = 1 + u32::from(*hash_count);
                let sid = self
                    .strings
                    .intern_span(Span::new(span.start + prefix, span.end - suffix));
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::Neg(inner) => {
                let vid = self.eval_expr(*inner)?;
                let Value::Int(n) = self.get_val(vid) else {
                    unreachable!()
                };
                Ok(self.alloc_val(Value::Int(-*n)))
            }
            Node::Ident => unreachable!(),
            Node::RuleRef => {
                let sid = self.strings.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::VarRef => Ok(self.lookup(ast.text(span)).unwrap()),
            Node::Inherit { .. } => Ok(self.alloc_val(Value::Grammar)),
            Node::Append { left, right } => {
                let (left, right) = (*left, *right);
                let lv = self.eval_expr(left)?;
                let rv = self.eval_expr(right)?;
                let Value::List(li) = self.get_val(lv) else {
                    unreachable!()
                };
                let Value::List(ri) = self.get_val(rv) else {
                    unreachable!()
                };
                let mut combined = li.clone();
                combined.extend_from_slice(ri);
                Ok(self.alloc_val(Value::List(combined)))
            }
            Node::FieldAccess { obj, field } => {
                let (obj, field) = (*obj, *field);
                let obj_val = self.eval_expr(obj)?;
                let field_name = ast.node_text(field);
                match self.get_val(obj_val) {
                    // Typecheck validates all field accesses: Object fields are
                    // checked against the known field list, Grammar fields against
                    // the known config names. No unknown field reaches lowering.
                    Value::Object(map) => Ok(*map.get(field_name).unwrap()),
                    Value::Grammar => Ok(self.import_config_field(field_name)),
                    _ => unreachable!(),
                }
            }
            Node::RuleInline { obj, rule } => {
                let (obj, rule) = (*obj, *rule);
                self.eval_expr(obj)?;
                let rule_name = ast.node_text(rule);
                let grammar = self.base_grammar.unwrap();
                match grammar.variables.iter().find(|v| v.name == rule_name) {
                    Some(var) => {
                        let rid = self.import_rule(&var.rule);
                        Ok(self.alloc_val(Value::Rule(rid)))
                    }
                    None => Err(LowerError::new(
                        LowerErrorKind::InheritRuleNotFound(rule_name.to_string()),
                        ast.span(rule),
                    )),
                }
            }
            Node::Object(range) => {
                let fields = ast.get_object(*range);
                let mut map =
                    FxHashMap::with_capacity_and_hasher(fields.len(), rustc_hash::FxBuildHasher);
                for &(key_span, value_id) in fields {
                    map.insert(ast.text(key_span), self.eval_expr(value_id)?);
                }
                Ok(self.alloc_val(Value::Object(map)))
            }
            Node::List(range) | Node::Tuple(range) => {
                let items = ast.child_slice(*range);
                let mut vals = Vec::with_capacity(items.len());
                for &item_id in items {
                    vals.push(self.eval_expr(item_id)?);
                }
                let val = if matches!(ast.node(id), Node::List(_)) {
                    Value::List(vals)
                } else {
                    Value::Tuple(vals)
                };
                Ok(self.alloc_val(val))
            }
            Node::Concat(range) => {
                let mut result = String::new();
                for &part_id in ast.child_slice(*range) {
                    let vid = self.eval_expr(part_id)?;
                    result.push_str(self.strings.resolve(self.get_str_val(vid)));
                }
                let sid = self.strings.intern_owned(result);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::DynRegex { pattern, flags } => {
                let pv = self.eval_expr(*pattern)?;
                let ps = self.get_str_val(pv);
                let fs = match flags {
                    Some(fid) => {
                        let fv = self.eval_expr(*fid)?;
                        Some(self.get_str_val(fv))
                    }
                    None => None,
                };
                let rid = self.alloc_rule(ARule::Pattern(ps, fs));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::Call { name, args } => {
                let (name, args) = (*name, *args);
                let mut arg_vals = Vec::with_capacity(ast.child_slice(args).len());
                for &arg_id in ast.child_slice(args) {
                    arg_vals.push(self.eval_expr(arg_id)?);
                }
                self.eval_fn_call(ast.node_text(name), &arg_vals)
            }
            _ => {
                let rid = self.eval_combinator(id)?;
                Ok(self.alloc_val(Value::Rule(rid)))
            }
        }
    }

    fn lower_to_rule(&mut self, id: NodeId) -> Result<RuleId, LowerError> {
        let ast = self.ast;
        match ast.node(id) {
            Node::RuleRef => {
                let sid = self.strings.intern_span(ast.span(id));
                Ok(self.alloc_rule(ARule::NamedSymbol(sid)))
            }
            Node::StringLit => {
                let span = ast.span(id);
                let content = Span::new(span.start + 1, span.end - 1);
                let raw = ast.text(content);
                let sid = if memchr::memchr(b'\\', raw.as_bytes()).is_some() {
                    self.strings.intern_owned(unescape_string(raw))
                } else {
                    self.strings.intern_span(content)
                };
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let span = ast.span(id);
                let prefix = 2 + u32::from(*hash_count);
                let suffix = 1 + u32::from(*hash_count);
                let sid = self
                    .strings
                    .intern_span(Span::new(span.start + prefix, span.end - suffix));
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            #[rustfmt::skip]
            Node::Seq(_) | Node::Choice(_) | Node::Repeat(_) | Node::Repeat1(_)
            | Node::Optional(_) | Node::Blank | Node::Field { .. } | Node::Alias { .. }
            | Node::Token(_) | Node::TokenImmediate(_) | Node::Prec { .. }
            | Node::PrecLeft { .. } | Node::PrecRight { .. } | Node::PrecDynamic { .. }
            | Node::Reserved { .. } => self.eval_combinator(id),
            _ => {
                let vid = self.eval_expr(id)?;
                Ok(self.value_to_rule(vid))
            }
        }
    }

    fn value_to_rule(&mut self, id: ValueId) -> RuleId {
        match self.get_val(id) {
            Value::Rule(rid) => *rid,
            Value::Str(s) => {
                let s = *s;
                self.alloc_rule(ARule::String(s))
            }
            _ => unreachable!(),
        }
    }

    fn value_to_prec(&self, id: ValueId) -> APrec {
        match self.get_val(id) {
            Value::Int(n) => APrec::Integer(*n),
            Value::Str(s) => APrec::Name(*s),
            _ => unreachable!(),
        }
    }

    fn eval_combinator(&mut self, id: NodeId) -> Result<RuleId, LowerError> {
        let ast = self.ast;
        match ast.node(id) {
            Node::Seq(range) | Node::Choice(range) => {
                let is_seq = matches!(ast.node(id), Node::Seq(_));
                let mut child_ids = Vec::with_capacity(ast.child_slice(*range).len());
                for &member in ast.child_slice(*range) {
                    if let Node::For(idx) = ast.node(member) {
                        child_ids.extend(self.eval_for_to_rules(ast.get_for(*idx))?);
                    } else {
                        child_ids.push(self.lower_to_rule(member)?);
                    }
                }
                let start = self.children_start();
                self.rule_children.extend_from_slice(&child_ids);
                let (offset, len) = self.children_range(start, ast.span(id))?;
                Ok(self.alloc_rule(if is_seq {
                    ARule::Seq(offset, len)
                } else {
                    ARule::Choice(offset, len)
                }))
            }
            Node::Repeat(inner) => {
                let inner = self.lower_to_rule(*inner)?;
                let rep = self.alloc_rule(ARule::Repeat(inner));
                let blank = self.alloc_rule(ARule::Blank);
                let start = self.children_start();
                self.rule_children.extend_from_slice(&[rep, blank]);
                let (s, l) = self.children_range(start, ast.span(id)).unwrap();
                Ok(self.alloc_rule(ARule::Choice(s, l)))
            }
            Node::Repeat1(inner) => {
                let inner = self.lower_to_rule(*inner)?;
                Ok(self.alloc_rule(ARule::Repeat(inner)))
            }
            Node::Optional(inner) => {
                let inner = self.lower_to_rule(*inner)?;
                let blank = self.alloc_rule(ARule::Blank);
                let start = self.children_start();
                self.rule_children.extend_from_slice(&[inner, blank]);
                let (s, l) = self.children_range(start, ast.span(id)).unwrap();
                Ok(self.alloc_rule(ARule::Choice(s, l)))
            }
            Node::Blank => Ok(self.alloc_rule(ARule::Blank)),
            Node::Field { name, content } => {
                let inner = self.lower_to_rule(*content)?;
                let sid = self.strings.intern_span(ast.span(*name));
                Ok(self.alloc_rule(ARule::Field(sid, inner)))
            }
            Node::Alias { content, target } => {
                let (content, target) = (*content, *target);
                let inner = self.lower_to_rule(content)?;
                match ast.node(target) {
                    Node::RuleRef | Node::VarRef => {
                        let sid = self.strings.intern_span(ast.span(target));
                        Ok(self.alloc_rule(ARule::Alias(sid, true, inner)))
                    }
                    _ => {
                        let vid = self.eval_expr(target)?;
                        match self.get_val(vid) {
                            Value::Str(s) => Ok(self.alloc_rule(ARule::Alias(*s, false, inner))),
                            Value::Rule(rid) => {
                                let ARule::NamedSymbol(s) = &self.rules[rid.0 as usize] else {
                                    unreachable!()
                                };
                                Ok(self.alloc_rule(ARule::Alias(*s, true, inner)))
                            }
                            _ => unreachable!(),
                        }
                    }
                }
            }
            Node::Token(inner) => {
                let inner = self.lower_to_rule(*inner)?;
                Ok(self.alloc_rule(ARule::Token(inner)))
            }
            Node::TokenImmediate(inner) => {
                let inner = self.lower_to_rule(*inner)?;
                Ok(self.alloc_rule(ARule::ImmediateToken(inner)))
            }
            Node::Prec { value, content }
            | Node::PrecLeft { value, content }
            | Node::PrecRight { value, content } => {
                let vid = self.eval_expr(*value)?;
                let prec = self.value_to_prec(vid);
                let inner = self.lower_to_rule(*content)?;
                let arule = match ast.node(id) {
                    Node::Prec { .. } => ARule::Prec(prec, inner),
                    Node::PrecLeft { .. } => ARule::PrecLeft(prec, inner),
                    Node::PrecRight { .. } => ARule::PrecRight(prec, inner),
                    _ => unreachable!(),
                };
                Ok(self.alloc_rule(arule))
            }
            Node::PrecDynamic { value, content } => {
                let vid = self.eval_expr(*value)?;
                let Value::Int(n) = self.get_val(vid) else {
                    unreachable!()
                };
                let n = *n;
                let inner = self.lower_to_rule(*content)?;
                Ok(self.alloc_rule(ARule::PrecDynamic(n, inner)))
            }
            Node::Reserved { context, content } => {
                let inner = self.lower_to_rule(*content)?;
                let sid = self.strings.intern_span(ast.span(*context));
                Ok(self.alloc_rule(ARule::Reserved(sid, inner)))
            }
            _ => unreachable!(),
        }
    }

    fn eval_fn_call(&mut self, name: &str, arg_vals: &[ValueId]) -> Result<ValueId, LowerError> {
        let &fn_id = self.fns.get(name).unwrap();
        let config = self.get_fn_config(fn_id);
        let body = config.body;
        self.push_scope();
        for (i, &arg_val) in arg_vals.iter().enumerate() {
            let param_name = self.get_fn_config(fn_id).params[i].name;
            self.bind(self.ast.node_text(param_name), arg_val);
        }
        let result = self.eval_expr(body);
        self.pop_scope();
        result
    }

    fn eval_for_to_rules(&mut self, config: &ForConfig) -> Result<Vec<RuleId>, LowerError> {
        let iter_vid = self.eval_expr(config.iterable)?;
        let n_items = match self.get_val(iter_vid) {
            Value::List(items) => items.len(),
            _ => unreachable!(),
        };
        let mut results = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let item_id = match self.get_val(iter_vid) {
                Value::List(items) => items[i],
                _ => unreachable!(),
            };
            self.push_scope();
            for (j, &(name_span, _)) in config.bindings.iter().enumerate() {
                let value_id = match self.get_val(item_id) {
                    Value::Tuple(ids) => ids[j],
                    _ => item_id,
                };
                self.bind(self.ast.text(name_span), value_id);
            }
            results.push(self.lower_to_rule(config.body)?);
            self.pop_scope();
        }
        Ok(results)
    }
}

/// Convert an [`InputGrammar`] to `grammar.json` format.
pub fn grammar_to_json(grammar: &InputGrammar) -> serde_json::Value {
    use serde_json::{Map, Value, json};

    let mut obj = Map::new();
    obj.insert("name".into(), Value::String(grammar.name.clone()));

    let mut rules = Map::new();
    for var in &grammar.variables {
        rules.insert(var.name.clone(), rule_to_json(&var.rule));
    }
    obj.insert("rules".into(), Value::Object(rules));

    let str_array = |items: &[String]| -> Value {
        Value::Array(items.iter().map(|s| Value::String(s.clone())).collect())
    };

    obj.insert(
        "extras".into(),
        Value::Array(grammar.extra_symbols.iter().map(rule_to_json).collect()),
    );
    obj.insert(
        "conflicts".into(),
        Value::Array(
            grammar
                .expected_conflicts
                .iter()
                .map(|g| str_array(g))
                .collect(),
        ),
    );
    obj.insert(
        "precedences".into(),
        Value::Array(
            grammar
                .precedence_orderings
                .iter()
                .map(|g| {
                    Value::Array(
                        g.iter()
                            .map(|e| match e {
                                PrecedenceEntry::Name(s) => json!({"type": "STRING", "value": s}),
                                PrecedenceEntry::Symbol(s) => json!({"type": "SYMBOL", "name": s}),
                            })
                            .collect(),
                    )
                })
                .collect(),
        ),
    );
    obj.insert(
        "externals".into(),
        Value::Array(grammar.external_tokens.iter().map(rule_to_json).collect()),
    );
    obj.insert("inline".into(), str_array(&grammar.variables_to_inline));
    obj.insert("supertypes".into(), str_array(&grammar.supertype_symbols));
    if let Some(word) = &grammar.word_token {
        obj.insert("word".into(), Value::String(word.clone()));
    }
    if !grammar.reserved_words.is_empty() {
        let mut reserved = Map::new();
        for ctx in &grammar.reserved_words {
            reserved.insert(
                ctx.name.clone(),
                Value::Array(ctx.reserved_words.iter().map(rule_to_json).collect()),
            );
        }
        obj.insert("reserved".into(), Value::Object(reserved));
    }
    Value::Object(obj)
}

fn rule_to_json(rule: &Rule) -> serde_json::Value {
    use crate::rules::Alias;
    use serde_json::json;
    match rule {
        Rule::Blank => json!({"type": "BLANK"}),
        Rule::String(s) => json!({"type": "STRING", "value": s}),
        Rule::Pattern(p, f) if f.is_empty() => json!({"type": "PATTERN", "value": p}),
        Rule::Pattern(p, f) => json!({"type": "PATTERN", "value": p, "flags": f}),
        Rule::NamedSymbol(n) => json!({"type": "SYMBOL", "name": n}),
        Rule::Symbol(s) => json!({"type": "SYMBOL", "name": format!("__symbol_{}", s.index)}),
        Rule::Choice(ms) => {
            json!({"type": "CHOICE", "members": ms.iter().map(rule_to_json).collect::<Vec<_>>()})
        }
        Rule::Seq(ms) => {
            json!({"type": "SEQ", "members": ms.iter().map(rule_to_json).collect::<Vec<_>>()})
        }
        Rule::Repeat(inner) => json!({"type": "REPEAT1", "content": rule_to_json(inner)}),
        Rule::Metadata { params, rule } => {
            let mut c = rule_to_json(rule);
            if params.dynamic_precedence != 0 {
                c = json!({"type": "PREC_DYNAMIC", "value": params.dynamic_precedence, "content": c});
            }
            if let Some(pv) = match &params.precedence {
                Precedence::None => None,
                Precedence::Integer(n) => Some(serde_json::Value::Number((*n).into())),
                Precedence::Name(s) => Some(serde_json::Value::String(s.clone())),
            } {
                c = match params.associativity {
                    Some(crate::rules::Associativity::Left) => {
                        json!({"type": "PREC_LEFT", "value": pv, "content": c})
                    }
                    Some(crate::rules::Associativity::Right) => {
                        json!({"type": "PREC_RIGHT", "value": pv, "content": c})
                    }
                    None => json!({"type": "PREC", "value": pv, "content": c}),
                };
            }
            if let Some(Alias { value, is_named }) = &params.alias {
                c = json!({"type": "ALIAS", "content": c, "named": is_named, "value": value});
            }
            if let Some(field_name) = &params.field_name {
                c = json!({"type": "FIELD", "name": field_name, "content": c});
            }
            if params.is_token && params.is_main_token {
                c = json!({"type": "IMMEDIATE_TOKEN", "content": c});
            } else if params.is_token {
                c = json!({"type": "TOKEN", "content": c});
            }
            c
        }
        Rule::Reserved { rule, context_name } => {
            json!({"type": "RESERVED", "context_name": context_name, "content": rule_to_json(rule)})
        }
    }
}

fn unescape_string(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.bytes();
    while let Some(b) = chars.next() {
        if b == b'\\' {
            match chars.next() {
                Some(b'"') => result.push('"'),
                Some(b'\\') => result.push('\\'),
                Some(b'n') => result.push('\n'),
                Some(b't') => result.push('\t'),
                Some(b'r') => result.push('\r'),
                Some(b'0') => result.push('\0'),
                _ => unreachable!(),
            }
        } else {
            result.push(b as char);
        }
    }
    result
}

#[derive(Debug, Serialize, Error)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    pub span: Span,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<Note>,
}

impl LowerError {
    #[must_use]
    pub const fn new(kind: LowerErrorKind, span: Span) -> Self {
        Self {
            kind,
            span,
            note: None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum LowerErrorKind {
    MissingGrammarBlock,
    MissingLanguageField,
    OverrideRuleNotFound(String),
    OverrideWithoutInherit,
    InheritLoadError(String),
    InheritCycle(Vec<std::path::PathBuf>),
    InheritRuleNotFound(String),
    TooManyChildren,
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LowerErrorKind::MissingGrammarBlock => write!(f, "missing grammar block"),
            LowerErrorKind::MissingLanguageField => {
                write!(f, "grammar block missing 'language' field")
            }
            LowerErrorKind::OverrideRuleNotFound(n) => {
                write!(f, "override rule '{n}' not found in base grammar")
            }
            LowerErrorKind::OverrideWithoutInherit => {
                write!(f, "'override rule' requires a grammar with 'inherits'")
            }
            LowerErrorKind::InheritLoadError(msg) => {
                write!(f, "failed to load inherited grammar: {msg}")
            }
            LowerErrorKind::InheritCycle(chain) => {
                write!(f, "inheritance cycle detected: ")?;
                for (i, p) in chain.iter().enumerate() {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    write!(f, "{}", p.display())?;
                }
                Ok(())
            }
            LowerErrorKind::InheritRuleNotFound(n) => write!(
                f,
                "rule '{n}' not found in base grammar (use '.' for config fields)"
            ),
            LowerErrorKind::TooManyChildren => {
                write!(f, "too many elements (maximum {})", u16::MAX)
            }
        }
    }
}
