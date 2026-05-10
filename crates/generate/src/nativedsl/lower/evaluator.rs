//!  Walks the typed AST into intermediate IR; [`super`] materializes it into
//!  [`crate::grammars::InputGrammar`].

use std::borrow::Cow;

use rustc_hash::FxHashMap;

use super::super::LowerError;
use super::super::ast::{
    ChildRange, ConfigField, ForId, IdentKind, MacroId, ModuleContext, Node, NodeId, PrecKind,
    RepeatKind, SharedAst, Span,
};
use super::super::{Module, ModuleId};
use super::repr::{APrec, ARule, RuleId, Str, StrEntry, Value, ValueId};
use super::{CallFrame, LowerErrorKind, LowerResult, LoweringState, MAX_CALL_DEPTH};

use crate::grammars::{PrecedenceEntry, ReservedWordContext};
use crate::rules::{Associativity, Precedence, Rule};

/// Per-grammar evaluation wrapper around long-lived [`LoweringState`].
pub(super) struct Evaluator<'a, 'ast> {
    pub(super) state: &'a mut LoweringState,
    pub(super) shared: &'ast SharedAst,
    /// Modules already loaded before this evaluation. The module being lowered
    /// ("root") is not in this slice; it lives in `root_ctx`. Its module id is
    /// `previous.len()` (== `root_id`).
    previous: &'a [Module],
    root_ctx: &'a ModuleContext,
    root_id: ModuleId,
    /// May equal `root_id` (current module) or index into `previous`.
    current_module: ModuleId,
}

impl<'a, 'ast> Evaluator<'a, 'ast> {
    pub(super) fn new(
        state: &'a mut LoweringState,
        shared: &'ast SharedAst,
        previous: &'a [Module],
        root_ctx: &'a ModuleContext,
    ) -> Self {
        let root_id = previous.len() as ModuleId;
        state.reset_per_grammar();
        // Mark the in-progress grammar as "loaded" so that subsequent imports
        // of it from other grammars short-circuit in `eval_import_module`.
        state.loaded.set_loaded(root_id);
        Self {
            state,
            shared,
            previous,
            root_ctx,
            root_id,
            current_module: root_id,
        }
    }

    pub(super) fn eval_let(&mut self, let_id: NodeId, value: NodeId) -> LowerResult<()> {
        let val = self.eval_expr(value)?;
        self.state.let_values.insert(let_id, val);
        Ok(())
    }

    fn ctx(&self) -> &'a ModuleContext {
        self.module_ctx(self.current_module)
    }

    /// `idx` is a global module id; `idx == self.root_id` is the in-progress
    /// module (whose ctx isn't in `previous`).
    fn module_ctx(&self, idx: ModuleId) -> &'a ModuleContext {
        if idx == self.root_id {
            self.root_ctx
        } else {
            self.previous[usize::from(idx)].ctx()
        }
    }

    fn resolve_str(&self, id: Str) -> &str {
        match self.state.ir.strings.entry(id) {
            &StrEntry::Source(span, mod_id) => span.resolve(&self.module_ctx(mod_id).source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }

    fn str_to_string(&self, id: Str) -> String {
        self.resolve_str(id).to_string()
    }

    fn alloc_val(&mut self, val: Value) -> ValueId {
        let id = ValueId(self.state.ir.values.len() as u32);
        self.state.ir.values.push(val);
        id
    }

    fn get_val(&self, id: ValueId) -> &Value {
        // Safety: id was produced by alloc_val which returns sequential indices
        // into self.state.ir.values. No ValueId can outlive the Evaluator.
        unsafe { self.state.ir.values.get_unchecked(id.0 as usize) }
    }

    fn alloc_list(&mut self, items: &[ValueId], span: Span) -> LowerResult<ValueId> {
        let start = self.state.ir.value_children.len() as u32;
        let len = u16::try_from(items.len())
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
        self.state.ir.value_children.extend_from_slice(items);
        Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
    }

    fn finish_list(&mut self, start: u32, len: usize, span: Span) -> LowerResult<ValueId> {
        let len = u16::try_from(len)
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
        Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
    }

    fn alloc_object(&mut self, map: FxHashMap<String, ValueId>) -> ValueId {
        let idx = self.state.ir.object_pool.len() as u32;
        self.state.ir.object_pool.push(map);
        self.alloc_val(Value::Object(idx))
    }

    fn list_items(&self, v: ValueId) -> &[ValueId] {
        &self.state.ir.value_children[self.list_range(v).as_range()]
    }

    fn list_range(&self, v: ValueId) -> ChildRange {
        expect_pat!(Value::List(range), *self.get_val(v));
        range
    }

    fn rule_id(&self, v: ValueId) -> RuleId {
        expect_pat!(Value::Rule(rid), *self.get_val(v));
        rid
    }

    fn str_id(&self, v: ValueId) -> Str {
        expect_pat!(Value::Str(s), *self.get_val(v));
        s
    }

    fn int_val(&self, v: ValueId) -> i32 {
        expect_pat!(Value::Int(n), *self.get_val(v));
        n
    }

    fn object_fields(&self, v: ValueId) -> &FxHashMap<String, ValueId> {
        expect_pat!(Value::Object(idx), *self.get_val(v));
        &self.state.ir.object_pool[idx as usize]
    }

    fn alloc_rule(&mut self, rule: ARule) -> RuleId {
        let id = RuleId(self.state.ir.rules.len() as u32);
        self.state.ir.rules.push(rule);
        id
    }

    fn get_rule(&self, id: RuleId) -> &ARule {
        // Safety: id was produced by alloc_rule which returns sequential indices.
        unsafe { self.state.ir.rules.get_unchecked(id.0 as usize) }
    }

    const fn children_start(&self) -> u32 {
        self.state.ir.rule_children.len() as u32
    }

    fn children_range(&self, start: u32, span: Span) -> LowerResult<ChildRange> {
        let count = self.state.ir.rule_children.len() as u32 - start;
        u16::try_from(count)
            .map(|len| ChildRange::new(start, len))
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))
    }

    /// Intern a span using the current module's source text.
    fn intern_span(&mut self, span: Span) -> Str {
        self.state.ir.strings.intern_span(span, self.current_module)
    }

    fn intern_string_lit(&mut self, span: Span) -> Str {
        let raw = self.ctx().text(span);
        if memchr::memchr(b'\\', raw.as_bytes()).is_some() {
            self.state
                .ir
                .strings
                .intern_owned(Cow::Owned(unescape_string(raw)))
        } else {
            self.intern_span(span)
        }
    }

    fn intern_raw_string_lit(&mut self, span: Span, hash_count: u8) -> Str {
        let prefix = 2 + u32::from(hash_count);
        let suffix = 1 + u32::from(hash_count);
        self.intern_span(Span::new(span.start + prefix, span.end - suffix))
    }

    /// Intern a string, create a `NamedSymbol` rule, and wrap as a `Value::Rule`.
    fn owned_symbol_val(&mut self, name: Cow<'_, str>) -> ValueId {
        let sid = self.state.ir.strings.intern_owned(name);
        let rid = self.alloc_rule(ARule::NamedSymbol(sid));
        self.alloc_val(Value::Rule(rid))
    }

    pub(super) fn build_rule(&self, id: RuleId) -> Rule {
        match self.get_rule(id) {
            ARule::Blank => Rule::Blank,
            ARule::String(sid) => Rule::String(self.str_to_string(*sid)),
            ARule::Pattern(p, f) => Rule::Pattern(
                self.str_to_string(*p),
                f.map_or_else(String::new, |f| self.str_to_string(f)),
            ),
            ARule::NamedSymbol(sid) => Rule::NamedSymbol(self.str_to_string(*sid)),
            ARule::SeqOrChoice(is_seq, range) => {
                // Safety: range is produced by children_start/children_range which
                // track rule_children indices.
                let children: Vec<Rule> =
                    unsafe { self.state.ir.rule_children.get_unchecked(range.as_range()) }
                        .iter()
                        .map(|&id| self.build_rule(id))
                        .collect();
                if *is_seq {
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
            ARule::Field(n, r) => Rule::field(self.str_to_string(*n), self.build_rule(*r)),
            ARule::Alias(v, named, r) => {
                Rule::alias(self.build_rule(*r), self.str_to_string(*v), *named)
            }
            ARule::Token(imm, r) => {
                let r = self.build_rule(*r);
                if *imm {
                    Rule::immediate_token(r)
                } else {
                    Rule::token(r)
                }
            }
            ARule::Reserved(ctx, r) => Rule::Reserved {
                rule: Box::new(self.build_rule(*r)),
                context_name: self.str_to_string(*ctx),
            },
        }
    }

    fn build_prec(&self, prec: APrec) -> Precedence {
        match prec {
            APrec::Integer(n) => Precedence::Integer(n),
            APrec::Name(sid) => Precedence::Name(self.str_to_string(sid)),
        }
    }

    fn value_to_owned_rule(&self, id: ValueId) -> Rule {
        match self.get_val(id) {
            Value::Rule(rid) => self.build_rule(*rid),
            Value::Str(sid) => Rule::String(self.str_to_string(*sid)),
            // Guarded by super::typecheck - only rule-like values reach here
            _ => unreachable!(),
        }
    }

    pub(super) fn eval_rule_list(&mut self, id: NodeId) -> LowerResult<Vec<Rule>> {
        let vid = self.eval_expr(id)?;
        let items = self.list_items(vid);
        Ok(items.iter().map(|&v| self.value_to_owned_rule(v)).collect())
    }

    fn rule_name_string(&self, v: ValueId, span: Span) -> LowerResult<String> {
        let rid = self.rule_id(v);
        let ARule::NamedSymbol(sid) = self.get_rule(rid) else {
            return Err(LowerError::new(LowerErrorKind::ExpectedRuleName, span));
        };
        Ok(self.str_to_string(*sid))
    }

    pub(super) fn eval_name_list(&mut self, id: NodeId) -> LowerResult<Vec<String>> {
        let vid = self.eval_expr(id)?;
        let span = self.shared.arena.span(id);
        let items = self.list_items(vid);
        items
            .iter()
            .map(|&v| self.rule_name_string(v, span))
            .collect()
    }

    /// Evaluate a `conflicts` expression into `Vec<Vec<String>>`. Each inner
    /// element must evaluate to a named rule reference.
    pub(super) fn eval_conflicts(&mut self, id: NodeId) -> LowerResult<Vec<Vec<String>>> {
        let vid = self.eval_expr(id)?;
        let span = self.shared.arena.span(id);
        let outer = self.list_items(vid);
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(group_vid);
                inner
                    .iter()
                    .map(|&v| self.rule_name_string(v, span))
                    .collect()
            })
            .collect()
    }

    /// Evaluate a `precedences` expression into `Vec<Vec<PrecedenceEntry>>`.
    /// Each inner element is either a string literal (`PrecedenceEntry::Name`)
    /// or a named rule reference (`PrecedenceEntry::Symbol`).
    pub(super) fn eval_precedences(
        &mut self,
        id: NodeId,
    ) -> LowerResult<Vec<Vec<PrecedenceEntry>>> {
        let vid = self.eval_expr(id)?;
        let span = self.shared.arena.span(id);
        let outer = self.list_items(vid);
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(group_vid);
                inner
                    .iter()
                    .map(|&v| match *self.get_val(v) {
                        Value::Str(sid) => Ok(PrecedenceEntry::Name(self.str_to_string(sid))),
                        Value::Rule(_) => {
                            Ok(PrecedenceEntry::Symbol(self.rule_name_string(v, span)?))
                        }
                        _ => unreachable!(),
                    })
                    .collect()
            })
            .collect()
    }

    pub(super) fn eval_rule_name(&mut self, id: NodeId) -> LowerResult<String> {
        let rid = self.lower_to_rule(id)?;
        match self.get_rule(rid) {
            ARule::NamedSymbol(sid) => Ok(self.str_to_string(*sid)),
            ARule::Blank => Err(LowerError::new(
                LowerErrorKind::ConfigFieldUnset,
                self.shared.arena.span(id),
            )),
            _ => Err(LowerError::new(
                LowerErrorKind::ExpectedRuleName,
                self.shared.arena.span(id),
            )),
        }
    }

    /// Evaluate the `reserved` config expression into `Vec<ReservedWordContext<Rule>>`.
    /// Accepts either an object literal `{ name: [words] }` or an expression
    /// evaluating to a list of `{name: str, words: list<rule>}` objects.
    pub(super) fn eval_reserved(
        &mut self,
        id: NodeId,
    ) -> LowerResult<Vec<ReservedWordContext<Rule>>> {
        let ctx = self.ctx();
        if let Node::Object(range) = self.shared.arena.get(id) {
            return self
                .shared
                .pools
                .get_object(*range)
                .iter()
                .map(|&(name_span, val_id)| {
                    let words = self.eval_rule_list(val_id)?;
                    Ok(ReservedWordContext {
                        name: ctx.text(name_span).to_string(),
                        reserved_words: words,
                    })
                })
                .collect();
        }
        // Sort by key for deterministic output (FxHashMap iteration is unordered).
        let vid = self.eval_expr(id)?;
        let fields = self.object_fields(vid);
        let mut keys: Vec<&str> = fields.keys().map(String::as_str).collect();
        keys.sort_unstable();
        keys.iter()
            .map(|&name| {
                let words_vid = fields[name];
                let word_vals = self.list_items(words_vid);
                let words = word_vals
                    .iter()
                    .map(|&wv| self.value_to_owned_rule(wv))
                    .collect();
                Ok(ReservedWordContext {
                    name: name.to_string(),
                    reserved_words: words,
                })
            })
            .collect()
    }

    fn eval_grammar_config(
        &mut self,
        mod_idx: ModuleId,
        field: ConfigField,
        span: Span,
    ) -> LowerResult<ValueId> {
        use ConfigField as C;
        // grammar_config target is always an inherit'd module (typecheck-enforced),
        // i.e. in `previous` with a lowered grammar set.
        let grammar = self.previous[usize::from(mod_idx)].lowered().unwrap();
        match field {
            C::Extras => self.import_rules_as_list(&grammar.extra_symbols, span),
            C::Externals => self.import_rules_as_list(&grammar.external_tokens, span),
            C::Inline => self.import_names_as_list(&grammar.variables_to_inline, span),
            C::Supertypes => self.import_names_as_list(&grammar.supertype_symbols, span),
            C::Conflicts => {
                let vals: Vec<ValueId> = grammar
                    .expected_conflicts
                    .iter()
                    .map(|g| self.import_names_as_list(g, span))
                    .collect::<LowerResult<_>>()?;
                self.alloc_list(&vals, span)
            }
            C::Precedences => {
                let vals: Vec<ValueId> = grammar
                    .precedence_orderings
                    .iter()
                    .map(|group| {
                        let inner_start = self.state.ir.value_children.len() as u32;
                        for entry in group {
                            let vid = match entry {
                                PrecedenceEntry::Name(s) => {
                                    let sid = self.state.ir.strings.intern_owned(Cow::Borrowed(s));
                                    self.alloc_val(Value::Str(sid))
                                }
                                PrecedenceEntry::Symbol(s) => {
                                    self.owned_symbol_val(Cow::Borrowed(s))
                                }
                            };
                            self.state.ir.value_children.push(vid);
                        }
                        self.finish_list(inner_start, group.len(), span)
                    })
                    .collect::<LowerResult<_>>()?;
                self.alloc_list(&vals, span)
            }
            C::Word => {
                if let Some(name) = &grammar.word_token {
                    Ok(self.owned_symbol_val(Cow::Borrowed(name)))
                } else {
                    Err(LowerError::new(LowerErrorKind::ConfigFieldUnset, span))
                }
            }
            C::Start => {
                // The start rule is always the first variable; this isn't
                // ConfigFieldUnset because every grammar has one.
                let name = &grammar.variables[0].name;
                Ok(self.owned_symbol_val(Cow::Borrowed(name)))
            }
            C::Reserved => {
                let n = grammar.reserved_words.len();
                let mut map = FxHashMap::with_capacity_and_hasher(n, rustc_hash::FxBuildHasher);
                for rwc in &grammar.reserved_words {
                    let words_vid = self.import_rules_as_list(&rwc.reserved_words, span)?;
                    map.insert(rwc.name.clone(), words_vid);
                }
                Ok(self.alloc_object(map))
            }
            C::Language | C::Inherits => unreachable!(),
        }
    }

    fn import_rules_as_list(&mut self, rules_data: &[Rule], span: Span) -> LowerResult<ValueId> {
        let start = self.state.ir.value_children.len() as u32;
        self.state.ir.value_children.reserve(rules_data.len());
        for r in rules_data {
            let rid = self.import_rule(r);
            let vid = self.alloc_val(Value::Rule(rid));
            self.state.ir.value_children.push(vid);
        }
        self.finish_list(start, rules_data.len(), span)
    }

    fn import_names_as_list(&mut self, names: &[String], span: Span) -> LowerResult<ValueId> {
        let start = self.state.ir.value_children.len() as u32;
        self.state.ir.value_children.reserve(names.len());
        for name in names {
            let vid = self.owned_symbol_val(Cow::Borrowed(name));
            self.state.ir.value_children.push(vid);
        }
        self.finish_list(start, names.len(), span)
    }

    fn import_rule(&mut self, rule: &Rule) -> RuleId {
        match rule {
            Rule::Blank => self.alloc_rule(ARule::Blank),
            Rule::String(s) => {
                let sid = self.state.ir.strings.intern_owned(Cow::Borrowed(s));
                self.alloc_rule(ARule::String(sid))
            }
            Rule::Pattern(p, f) => {
                let pid = self.state.ir.strings.intern_owned(Cow::Borrowed(p));
                let fid =
                    (!f.is_empty()).then(|| self.state.ir.strings.intern_owned(Cow::Borrowed(f)));
                self.alloc_rule(ARule::Pattern(pid, fid))
            }
            Rule::NamedSymbol(name) => {
                let sid = self.state.ir.strings.intern_owned(Cow::Borrowed(name));
                self.alloc_rule(ARule::NamedSymbol(sid))
            }
            Rule::Choice(members) | Rule::Seq(members) => {
                let is_seq = matches!(rule, Rule::Seq(_));
                self.state.rule_scratch.reserve(members.len());
                self.state.ir.rule_children.reserve(members.len());
                let (start, count) = stack_scope!(self.state.rule_scratch, |base| {
                    for m in members {
                        let rid = self.import_rule(m);
                        self.state.rule_scratch.push(rid);
                    }
                    let start = self.children_start();
                    self.state
                        .ir
                        .rule_children
                        .extend_from_slice(&self.state.rule_scratch[base..]);
                    (start, self.state.ir.rule_children.len() as u32 - start)
                });
                debug_assert!(u16::try_from(count).is_ok());
                self.alloc_rule(ARule::SeqOrChoice(
                    is_seq,
                    ChildRange::new(start, count as u16),
                ))
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
                    let aprec = APrec::Name(self.state.ir.strings.intern_owned(Cow::Borrowed(s)));
                    rid = self.import_prec_rule(aprec, rid, params.associativity);
                }
                if let Some(crate::rules::Alias { value, is_named }) = &params.alias {
                    let sid = self.state.ir.strings.intern_owned(Cow::Borrowed(value));
                    rid = self.alloc_rule(ARule::Alias(sid, *is_named, rid));
                }
                if let Some(field_name) = &params.field_name {
                    let sid = self.state.ir.strings.intern_owned(Cow::Borrowed(field_name));
                    rid = self.alloc_rule(ARule::Field(sid, rid));
                }
                if params.is_token {
                    rid = self.alloc_rule(ARule::Token(params.is_main_token, rid));
                }
                rid
            }
            Rule::Reserved {
                rule: inner,
                context_name,
            } => {
                let inner = self.import_rule(inner);
                let sid = self.state.ir.strings.intern_owned(Cow::Borrowed(context_name));
                self.alloc_rule(ARule::Reserved(sid, inner))
            }
            Rule::Symbol(_) => unreachable!(),
        }
    }

    fn import_prec_rule(
        &mut self,
        prec: APrec,
        inner: RuleId,
        assoc: Option<Associativity>,
    ) -> RuleId {
        match assoc {
            Some(Associativity::Left) => self.alloc_rule(ARule::PrecLeft(prec, inner)),
            Some(Associativity::Right) => self.alloc_rule(ARule::PrecRight(prec, inner)),
            None => self.alloc_rule(ARule::Prec(prec, inner)),
        }
    }

    pub(super) fn eval_expr(&mut self, id: NodeId) -> LowerResult<ValueId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::IntLit(n) => {
                let v = i32::try_from(*n)
                    .map_err(|_| LowerError::new(LowerErrorKind::IntegerOverflow(*n), span))?;
                Ok(self.alloc_val(Value::Int(v)))
            }
            Node::StringLit => {
                let sid = self.intern_string_lit(span);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(span, *hash_count);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::Neg(inner) => {
                // Special-case Neg over a literal so the most negative i32
                // (-i32::MIN as positive overflows i32, but fits in i64) is
                // reachable as `-2147483648`.
                if let Node::IntLit(m) = *self.shared.arena.get(*inner) {
                    let neg = -m;
                    let v = i32::try_from(neg)
                        .map_err(|_| LowerError::new(LowerErrorKind::IntegerOverflow(neg), span))?;
                    return Ok(self.alloc_val(Value::Int(v)));
                }
                let vid = self.eval_expr(*inner)?;
                // Guarded by super::typecheck::type_of (Node::Neg) - ensures inner is int
                let n = self.int_val(vid);
                let neg = n.checked_neg().ok_or_else(|| {
                    LowerError::new(LowerErrorKind::IntegerOverflow(-i64::from(n)), span)
                })?;
                Ok(self.alloc_val(Value::Int(neg)))
            }
            // Guarded by super::resolve + typecheck
            Node::Ident(IdentKind::Unresolved | IdentKind::Macro(_)) => unreachable!(),
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::MacroParam { index, .. } => {
                let base = *self.state.macro_arg_bases.last().unwrap();
                Ok(self.state.macro_args[base + *index as usize])
            }
            Node::ForBinding { for_id, index, .. } => {
                let base = self
                    .state
                    .for_binding_frames
                    .iter()
                    .rev()
                    .find(|(id, _)| *id == *for_id)
                    .unwrap()
                    .1;
                Ok(self.state.for_binding_values[base + *index as usize])
            }
            // Guarded by super::resolve - all variable names validated
            Node::Ident(IdentKind::Var(let_id)) => Ok(*self.state.let_values.get(let_id).unwrap()),
            &Node::GrammarConfig { module, field } => {
                let mod_val = self.eval_expr(module)?;
                expect_pat!(Value::Module(mod_idx), *self.get_val(mod_val));
                self.eval_grammar_config(mod_idx, field, span)
            }
            Node::ModuleRef { module, .. } => {
                let global_id = module.expect("module index not set by loading pre-pass");
                self.eval_import_module(global_id)?;
                Ok(self.alloc_val(Value::Module(global_id)))
            }
            &Node::Append { left, right } => {
                let lv = self.eval_expr(left)?;
                let rv = self.eval_expr(right)?;
                // Guarded by super::typecheck::type_of (Node::Append) - ensures both are lists
                let lr = self.list_range(lv);
                let rr = self.list_range(rv);
                let start = self.state.ir.value_children.len() as u32;
                let total = lr.len as usize + rr.len as usize;
                let len = u16::try_from(total)
                    .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
                self.state.ir.value_children.extend_from_within(lr.as_range());
                self.state.ir.value_children.extend_from_within(rr.as_range());
                Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
            }
            &Node::FieldAccess { obj, field } => {
                let obj_val = self.eval_expr(obj)?;
                let field_name = self.ctx().text(field);
                match *self.get_val(obj_val) {
                    Value::Object(idx) => Ok(*self.state.ir.object_pool[idx as usize]
                        .get(field_name)
                        .unwrap()),
                    _ => unreachable!(), // guarded by typecheck
                }
            }
            // QualifiedAccess targets that survive resolve are one of:
            //   - cached external decl (helper, or grammar with top-level externals)
            //   - helper rule (inlined via import_rule)
            //   - inherited grammar rule (via lowered.variables)
            //   - inherited grammar external (via lowered.external_tokens)
            // Resolver already verified the target exists.
            &Node::QualifiedAccess { obj, member } => {
                let obj_val = self.eval_expr(obj)?;
                let member_name = self.ctx().text(member);
                expect_pat!(Value::Module(mod_idx), *self.get_val(obj_val));

                let target_ctx = self.previous[usize::from(mod_idx)].ctx();
                if let Some(name_span) = target_ctx
                    .external_names
                    .iter()
                    .copied()
                    .find(|&s| target_ctx.text(s) == member_name)
                {
                    let sid = self.state.ir.strings.intern_span(name_span, mod_idx);
                    let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                    return Ok(self.alloc_val(Value::Rule(rid)));
                }

                let target = &self.previous[usize::from(mod_idx)];
                let rule = match target {
                    Module::Helper { lowered_rules, .. } => lowered_rules
                        .iter()
                        .find(|(name, _)| name == member_name)
                        .map(|(_, r)| r)
                        .unwrap(),
                    Module::Grammar { lowered, .. } => lowered
                        .variables
                        .iter()
                        .find(|v| v.name == member_name)
                        .map(|v| &v.rule)
                        .or_else(|| {
                            lowered
                                .external_tokens
                                .iter()
                                .find(|r| matches!(r, Rule::NamedSymbol(n) if n == member_name))
                        })
                        .unwrap(),
                };
                let rid = self.import_rule(rule);
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            &Node::Object(range) => {
                let fields = self.shared.pools.get_object(range);
                let mut map =
                    FxHashMap::with_capacity_and_hasher(fields.len(), rustc_hash::FxBuildHasher);
                for &(key_span, value_id) in fields {
                    map.insert(
                        self.ctx().text(key_span).to_string(),
                        self.eval_expr(value_id)?,
                    );
                }
                Ok(self.alloc_object(map))
            }
            node @ (&Node::List(range) | &Node::Tuple(range)) => {
                let is_list = matches!(node, Node::List(_));
                let items = self.shared.pools.child_slice(range);
                let range = stack_scope!(self.state.val_scratch, |base| {
                    for &item_id in items {
                        if let Node::For { for_id, body } = *self.shared.arena.get(item_id) {
                            self.eval_for_to_values(for_id, body)?;
                        } else {
                            let val = self.eval_expr(item_id)?;
                            self.state.val_scratch.push(val);
                        }
                    }
                    let start = self.state.ir.value_children.len() as u32;
                    let len = u16::try_from(self.state.val_scratch.len() - base)
                        .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
                    self.state
                        .ir
                        .value_children
                        .extend_from_slice(&self.state.val_scratch[base..]);
                    Ok(ChildRange::new(start, len))
                })?;
                if is_list {
                    Ok(self.alloc_val(Value::List(range)))
                } else {
                    Ok(self.alloc_val(Value::Tuple(range)))
                }
            }
            &Node::Concat(range) => {
                let mut result = String::new();
                for &part_id in self.shared.pools.child_slice(range) {
                    let vid = self.eval_expr(part_id)?;
                    result.push_str(self.resolve_str(self.str_id(vid)));
                }
                let sid = self.state.ir.strings.intern_owned(Cow::Owned(result));
                Ok(self.alloc_val(Value::Str(sid)))
            }
            &Node::DynRegex { pattern, flags } => {
                let pv = self.eval_expr(pattern)?;
                let ps = self.str_id(pv);
                let fs = flags
                    .map(|fid| self.eval_expr(fid))
                    .transpose()?
                    .map(|fv| self.str_id(fv));
                let rid = self.alloc_rule(ARule::Pattern(ps, fs));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            &Node::Call { name, args } => {
                expect_pat!(
                    Node::Ident(IdentKind::Macro(macro_id)),
                    self.shared.arena.get(name)
                );
                self.invoke_macro(*macro_id, args, span, self.current_module)
            }
            &Node::QualifiedCall(range) => {
                let children = self.shared.pools.child_slice(range);
                let (obj, name) = (children[0], children[1]);
                let obj_val = self.eval_expr(obj)?;
                expect_pat!(Value::Module(mod_idx), *self.get_val(obj_val)); // guarded by typecheck
                self.eval_import_module(mod_idx)?;
                // guarded by resolver
                expect_pat!(
                    Node::Ident(IdentKind::Macro(macro_id)),
                    self.shared.arena.get(name)
                );
                let args = ChildRange::new(range.start + 2, range.len - 2);
                self.invoke_macro(*macro_id, args, span, mod_idx)
            }
            _ => {
                let rid = self.eval_combinator(id)?;
                Ok(self.alloc_val(Value::Rule(rid)))
            }
        }
    }

    pub(super) fn lower_to_rule(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                Ok(self.alloc_rule(ARule::NamedSymbol(sid)))
            }
            Node::StringLit => {
                let sid = self.intern_string_lit(span);
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(span, *hash_count);
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            #[rustfmt::skip]
            Node::SeqOrChoice { .. } | Node::Repeat { .. }
            | Node::Blank | Node::Field { .. } | Node::Alias { .. }
            | Node::Token { .. } | Node::Prec { .. }
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
            Value::Str(s) => self.alloc_rule(ARule::String(*s)),
            // Guarded by super::typecheck::expect_rule - only rule-like values reach here
            _ => unreachable!(),
        }
    }

    fn eval_combinator(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            &Node::SeqOrChoice { seq, range } => {
                let children = self.shared.pools.child_slice(range);
                let child_range = stack_scope!(self.state.rule_scratch, |base| {
                    for &member in children {
                        if let Node::For { for_id, body } = *self.shared.arena.get(member) {
                            self.eval_for_to_rules(for_id, body)?;
                        } else {
                            let rid = self.lower_to_rule(member)?;
                            self.state.rule_scratch.push(rid);
                        }
                    }
                    let start = self.children_start();
                    self.state
                        .ir
                        .rule_children
                        .extend_from_slice(&self.state.rule_scratch[base..]);
                    self.children_range(start, span)
                })?;
                Ok(self.alloc_rule(ARule::SeqOrChoice(seq, child_range)))
            }
            &Node::Repeat { kind, inner } => {
                let inner = self.lower_to_rule(inner)?;
                if kind == RepeatKind::OneOrMore {
                    return Ok(self.alloc_rule(ARule::Repeat(inner)));
                }
                let first = if kind == RepeatKind::ZeroOrMore {
                    self.alloc_rule(ARule::Repeat(inner))
                } else {
                    inner
                };
                let blank = self.alloc_rule(ARule::Blank);
                let start = self.children_start();
                self.state.ir.rule_children.extend_from_slice(&[first, blank]);
                let child_range = self.children_range(start, span).unwrap();
                Ok(self.alloc_rule(ARule::SeqOrChoice(false, child_range)))
            }
            Node::Blank => Ok(self.alloc_rule(ARule::Blank)),
            &Node::Field { name, content } => {
                let inner = self.lower_to_rule(content)?;
                let sid = self.intern_span(name);
                Ok(self.alloc_rule(ARule::Field(sid, inner)))
            }
            &Node::Alias { content, target } => {
                let inner = self.lower_to_rule(content)?;
                if matches!(self.shared.arena.get(target), Node::Ident(IdentKind::Rule)) {
                    let sid = self.intern_span(self.shared.arena.span(target));
                    Ok(self.alloc_rule(ARule::Alias(sid, true, inner)))
                } else {
                    let vid = self.eval_expr(target)?;
                    match self.get_val(vid) {
                        Value::Str(s) => Ok(self.alloc_rule(ARule::Alias(*s, false, inner))),
                        Value::Rule(rid) => {
                            let ARule::NamedSymbol(s) = self.get_rule(*rid) else {
                                return Err(LowerError::new(
                                    LowerErrorKind::ExpectedRuleName,
                                    self.shared.arena.span(target),
                                ));
                            };
                            Ok(self.alloc_rule(ARule::Alias(*s, true, inner)))
                        }
                        // Guarded by super::typecheck::type_of (Node::Alias) -
                        // only str or rule-name targets pass
                        _ => unreachable!(),
                    }
                }
            }
            &Node::Token { immediate, inner } => {
                let inner = self.lower_to_rule(inner)?;
                Ok(self.alloc_rule(ARule::Token(immediate, inner)))
            }
            &Node::Prec {
                kind,
                value,
                content,
            } => {
                let vid = self.eval_expr(value)?;
                let inner = self.lower_to_rule(content)?;
                if kind == PrecKind::Dynamic {
                    let n = self.int_val(vid);
                    Ok(self.alloc_rule(ARule::PrecDynamic(n, inner)))
                } else {
                    // Guarded by super::typecheck::type_of (Node::Prec)
                    let prec = match self.get_val(vid) {
                        Value::Int(n) => APrec::Integer(*n),
                        Value::Str(s) => APrec::Name(*s),
                        _ => unreachable!(),
                    };
                    let ctor = match kind {
                        PrecKind::Left => ARule::PrecLeft,
                        PrecKind::Right => ARule::PrecRight,
                        _ => ARule::Prec,
                    };
                    Ok(self.alloc_rule(ctor(prec, inner)))
                }
            }
            &Node::Reserved { context, content } => {
                let inner = self.lower_to_rule(content)?;
                let sid = self.intern_span(context);
                Ok(self.alloc_rule(ARule::Reserved(sid, inner)))
            }
            // Guarded by lower_to_rule - only combinator nodes are dispatched here
            _ => unreachable!(),
        }
    }

    fn push_call(
        &mut self,
        name_span: Span,
        name_mod: ModuleId,
        call_span: Span,
    ) -> LowerResult<()> {
        // Push first, then check, so the reported trace includes the call that
        // tripped the limit.
        self.state.call_stack.push(CallFrame {
            name_span,
            name_mod,
            call_span,
            caller_mod: self.current_module,
        });
        if self.state.call_stack.len() > MAX_CALL_DEPTH as usize {
            let trace = self
                .state
                .call_stack
                .iter()
                .map(|frame| {
                    let name_ctx = self.module_ctx(frame.name_mod);
                    let name = name_ctx.text(frame.name_span).to_string();
                    let call_ctx = self.module_ctx(frame.caller_mod);
                    let offset = frame.call_span.start as usize;
                    let bytes = &call_ctx.source.as_bytes()[..offset];
                    let line = memchr::memchr_iter(b'\n', bytes).count() + 1;
                    let col = offset - memchr::memrchr(b'\n', bytes).map_or(0, |i| i + 1) + 1;
                    (name, call_ctx.path.clone(), line, col)
                })
                .collect();
            let root_span = self.state.call_stack[0].call_span;
            return Err(LowerError::new(
                LowerErrorKind::CallDepthExceeded(trace),
                root_span,
            ));
        }
        Ok(())
    }

    /// Evaluate a call to a macro defined in `def_module`.
    fn invoke_macro(
        &mut self,
        macro_id: MacroId,
        args: ChildRange,
        span: Span,
        def_module: ModuleId,
    ) -> LowerResult<ValueId> {
        let config = self.shared.pools.get_macro(macro_id);
        // Args evaluate in the caller's macro context, not the callee's, so
        // arg eval happens before macro_arg_bases is pushed.
        stack_scope!(
            self.state.macro_args => args_base,
            self.state.call_stack => _call_base,
            self.state.macro_arg_bases => _bases_base;
            {
                self.push_call(config.name, def_module, span)?;
                for &arg_id in self.shared.pools.child_slice(args) {
                    let v = self.eval_expr(arg_id)?;
                    self.state.macro_args.push(v);
                }
                self.state.macro_arg_bases.push(args_base);
                let saved = self.current_module;
                self.current_module = def_module;
                let result = self.eval_expr(config.body);
                self.current_module = saved;
                result
            }
        )
    }

    /// Lazily evaluate an imported module's top-level let bindings.
    fn eval_import_module(&mut self, mod_idx: ModuleId) -> LowerResult<()> {
        if self.state.loaded.is_loaded(mod_idx) {
            return Ok(());
        }
        let saved_module = self.current_module;
        self.current_module = mod_idx;
        let result = self.eval_let_bindings();
        self.current_module = saved_module;
        result?;
        self.state.loaded.set_loaded(mod_idx);
        Ok(())
    }

    fn eval_let_bindings(&mut self) -> LowerResult<()> {
        let ctx = self.ctx();
        let n = ctx.root_items.len();
        for i in 0..n {
            let item_id = ctx.root_items[i];
            if let &Node::Let { value, .. } = self.shared.arena.get(item_id) {
                let val = self.eval_expr(value)?;
                self.state.let_values.insert(item_id, val);
            }
        }
        Ok(())
    }

    /// Evaluate a for-loop, pushing each iteration's rule into `rule_scratch`.
    /// If the body is itself a for-loop, recurses (flatMap semantics).
    fn eval_for_to_rules(&mut self, for_id: ForId, body: NodeId) -> LowerResult<()> {
        self.eval_for_each(for_id, |evaluator| {
            if let &Node::For {
                for_id: inner,
                body: inner_body,
            } = evaluator.shared.arena.get(body)
            {
                evaluator.eval_for_to_rules(inner, inner_body)
            } else {
                let rule_id = evaluator.lower_to_rule(body)?;
                evaluator.state.rule_scratch.push(rule_id);
                Ok(())
            }
        })
    }

    /// Evaluate a for-loop, pushing each iteration's value into `val_scratch`.
    /// If the body is itself a for-loop, recurses (flatMap semantics).
    fn eval_for_to_values(&mut self, for_id: ForId, body: NodeId) -> LowerResult<()> {
        self.eval_for_each(for_id, |evaluator| {
            if let &Node::For {
                for_id: inner,
                body: inner_body,
            } = evaluator.shared.arena.get(body)
            {
                evaluator.eval_for_to_values(inner, inner_body)
            } else {
                let value_id = evaluator.eval_expr(body)?;
                evaluator.state.val_scratch.push(value_id);
                Ok(())
            }
        })
    }

    fn eval_for_each<EachIter>(&mut self, for_id: ForId, mut each_iter: EachIter) -> LowerResult<()>
    where
        EachIter: FnMut(&mut Self) -> LowerResult<()>,
    {
        let for_config = self.shared.pools.get_for(for_id);
        let iterable = for_config.iterable;
        let n_bindings = for_config.bindings.len();
        let iter_vid = self.eval_expr(iterable)?;
        let iter_range = self.list_range(iter_vid);
        let n_items = iter_range.len as usize;
        stack_scope!(
            self.state.for_binding_values => base,
            self.state.for_binding_frames => _frames_base;
            {
                self.state.for_binding_frames.push((for_id, base));
                for i in 0..n_items {
                    let item_id = self.state.ir.value_children[iter_range.start as usize + i];
                    for j in 0..n_bindings {
                        let val = match *self.get_val(item_id) {
                            Value::Tuple(range) => self.state.ir.value_children[range.start as usize + j],
                            _ => item_id,
                        };
                        if i == 0 {
                            self.state.for_binding_values.push(val);
                        } else {
                            self.state.for_binding_values[base + j] = val;
                        }
                    }
                    each_iter(self)?;
                }
                Ok(())
            }
        )
    }
}

fn unescape_string(raw: &str) -> String {
    let bytes = raw.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(raw.len());
    let mut i = 0;
    while let Some(off) = memchr::memchr(b'\\', &bytes[i..]) {
        out.extend_from_slice(&bytes[i..i + off]);
        // Guarded by lexer escape validation
        out.push(match bytes[i + off + 1] {
            b'"' => b'"',
            b'\\' => b'\\',
            b'n' => b'\n',
            b't' => b'\t',
            b'r' => b'\r',
            b'0' => 0,
            _ => unreachable!(),
        });
        i += off + 2;
    }
    out.extend_from_slice(&bytes[i..]);
    // SAFETY: input was UTF-8 and every escape resolves to an ASCII byte.
    unsafe { String::from_utf8_unchecked(out) }
}
