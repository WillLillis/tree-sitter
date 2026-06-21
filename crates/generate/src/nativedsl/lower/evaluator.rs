//!  Walks the typed AST into intermediate IR; [`super`] materializes it into
//!  [`crate::grammars::InputGrammar`].

use std::path::PathBuf;

use rustc_hash::FxHashMap;

use super::{
    super::{
        LowerError, Module, ModuleId, NoteMessage,
        ast::{
            BinOp, ChildRange, ConfigField, ExpandId, ForId, IdentKind, MacroId, ModuleContext,
            Node, NodeId, ObjectField, PrecKind, RepeatKind, RuleTarget, SharedAst, Span,
        },
        lexer::unescape_string,
        string_pool::{Str, StrEntry, StringPool},
    },
    CallFrame, LowerErrorKind, LowerResult, LoweringState, MAX_CALL_DEPTH, MAX_EVAL_DEPTH,
    repr::{APrec, ARule, RuleId, Value, ValueId},
};

use crate::{
    grammars::{PrecedenceEntry, ReservedWordContext},
    rules::{Associativity, Precedence, Rule},
};

/// Per-grammar evaluation wrapper around long-lived [`LoweringState`].
pub(super) struct Evaluator<'a, 'ast> {
    pub state: &'a mut LoweringState,
    pub strings: &'a mut StringPool,
    pub shared: &'ast SharedAst,
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
    pub fn new(
        state: &'a mut LoweringState,
        strings: &'a mut StringPool,
        shared: &'ast SharedAst,
        previous: &'a [Module],
        root_ctx: &'a ModuleContext,
    ) -> Self {
        let root_id = previous.len() as ModuleId;
        state.reset_per_grammar();
        Self {
            state,
            strings,
            shared,
            previous,
            root_ctx,
            root_id,
            current_module: root_id,
        }
    }

    /// Evaluate a let binding's value and memoize it under `let_id`.
    pub fn eval_let(&mut self, let_id: NodeId) -> LowerResult<ValueId> {
        if let Some(&val) = self.state.let_values.get(&let_id) {
            return Ok(val);
        }
        self.state.scratch.lets_in_progress.insert(let_id);
        expect_pat!(Node::Let { value, .. }, *self.shared.arena.get(let_id));
        let val = self.eval_expr(value)?;
        self.state.scratch.lets_in_progress.remove(&let_id);
        self.state.let_values.insert(let_id, val);
        Ok(val)
    }

    fn module_ctx(&self, idx: ModuleId) -> &'a ModuleContext {
        if idx == self.root_id {
            self.root_ctx
        } else {
            self.previous[usize::from(idx)].ctx()
        }
    }

    /// Build a lower error located in the current module's source, so a span in
    /// an imported module's macro body renders against that file, not the root.
    /// Born located, mirroring how [`ModuleContext::note`] stamps each note.
    fn err(&self, kind: LowerErrorKind, span: Span) -> LowerError {
        let ctx = self.module_ctx(self.current_module);
        LowerError::new(kind, span).with_source(&ctx.source, &ctx.path)
    }

    /// Convert a child count to the `u16` a [`ChildRange`] holds, mapping
    /// overflow to the standard "too many children" error.
    fn checked_len(&self, n: usize, span: Span) -> LowerResult<u16> {
        u16::try_from(n).map_err(|_| self.err(LowerErrorKind::TooManyChildren(n), span))
    }

    fn resolve_str(&self, id: Str) -> &str {
        match self.strings.entry(id) {
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
        // Safety: id came from alloc_val, which hands out sequential indices into
        // self.state.ir.values. That pool is append-only and never reset for the
        // life of the `LoweringState`.
        unsafe { self.state.ir.values.get_unchecked(id.0 as usize) }
    }

    fn alloc_list(&mut self, items: &[ValueId], span: Span) -> LowerResult<ValueId> {
        let start = self.state.ir.value_children.len() as u32;
        let len = self.checked_len(items.len(), span)?;
        self.state.ir.value_children.extend_from_slice(items);
        Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
    }

    fn finish_list(&mut self, start: u32, len: usize, span: Span) -> LowerResult<ValueId> {
        let len = self.checked_len(len, span)?;
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

    fn str_id(&self, v: ValueId) -> Str {
        expect_pat!(Value::Str(s), *self.get_val(v));
        s
    }

    fn int_val(&self, v: ValueId) -> i32 {
        expect_pat!(Value::Int(n), *self.get_val(v));
        n
    }

    fn alloc_rule(&mut self, rule: ARule) -> RuleId {
        let id = RuleId(self.state.ir.rules.len() as u32);
        self.state.ir.rules.push(rule);
        id
    }

    fn get_rule(&self, id: RuleId) -> &ARule {
        // Safety: id came from alloc_rule, which hands out sequential indices into
        // the append-only, never-reset `self.state.ir.rules pool`.
        unsafe { self.state.ir.rules.get_unchecked(id.0 as usize) }
    }

    const fn children_start(&self) -> u32 {
        self.state.ir.rule_children.len() as u32
    }

    fn children_range(&self, start: u32, span: Span) -> LowerResult<ChildRange> {
        let count = self.state.ir.rule_children.len() - start as usize;
        Ok(ChildRange::new(start, self.checked_len(count, span)?))
    }

    /// Intern a span using the current module's source text.
    fn intern_span(&mut self, span: Span) -> Str {
        self.strings.intern_span(span, self.current_module)
    }

    fn intern_string_lit(&mut self, span: Span) -> Str {
        let raw = self.module_ctx(self.current_module).text(span);
        if memchr::memchr(b'\\', raw.as_bytes()).is_some() {
            self.strings.intern_owned(&unescape_string(raw))
        } else {
            self.intern_span(span)
        }
    }

    fn intern_raw_string_lit(&mut self, span: Span, hash_count: u8) -> Str {
        self.intern_span(span.strip_raw(hash_count))
    }

    /// Intern a string, create a `NamedSymbol` rule, and wrap as a `Value::Rule`.
    fn owned_symbol_val(&mut self, name: &str) -> ValueId {
        let sid = self.strings.intern_owned(name);
        let rid = self.alloc_rule(ARule::NamedSymbol(sid));
        self.alloc_val(Value::Rule(rid))
    }

    pub fn build_rule(&self, id: RuleId) -> Rule {
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

    pub fn eval_rule_list(&mut self, id: NodeId) -> LowerResult<Vec<Rule>> {
        let vid = self.eval_expr(id)?;
        let items = self.list_items(vid);
        Ok(items.iter().map(|&v| self.value_to_owned_rule(v)).collect())
    }

    fn rule_name_string(&self, v: ValueId, span: Span) -> LowerResult<String> {
        // A name position (inline/supertypes/conflicts, precedences symbol) needs a
        // named-symbol reference. typecheck's expect_name_ref rejects direct
        // non-names; a string laundered through a computed rule_t list, or a
        // non-name rule (e.g. seq(a, b)), reaches here - a user error, not a panic.
        let Value::Rule(rid) = *self.get_val(v) else {
            return Err(self.err(LowerErrorKind::ExpectedRuleName, span));
        };
        let ARule::NamedSymbol(sid) = self.get_rule(rid) else {
            return Err(self.err(LowerErrorKind::ExpectedRuleName, span));
        };
        Ok(self.str_to_string(*sid))
    }

    pub fn eval_name_list(&mut self, id: NodeId) -> LowerResult<Vec<String>> {
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
    pub fn eval_conflicts(&mut self, id: NodeId) -> LowerResult<Vec<Vec<String>>> {
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
    pub fn eval_precedences(&mut self, id: NodeId) -> LowerResult<Vec<Vec<PrecedenceEntry>>> {
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

    pub fn eval_rule_name(&mut self, id: NodeId) -> LowerResult<String> {
        let rid = self.lower_to_rule(id)?;
        match self.get_rule(rid) {
            ARule::NamedSymbol(sid) => Ok(self.str_to_string(*sid)),
            _ => Err(self.err(LowerErrorKind::ExpectedRuleName, self.shared.arena.span(id))),
        }
    }

    /// Evaluate the `reserved` config expression into `Vec<ReservedWordContext<Rule>>`.
    /// Accepts either an object literal `{ name: [words] }` or an expression
    /// evaluating to a list of `{name: str, words: list<rule>}` objects.
    pub fn eval_reserved(&mut self, id: NodeId) -> LowerResult<Vec<ReservedWordContext<Rule>>> {
        let ctx = self.module_ctx(self.current_module);
        // typecheck guarantees `reserved` is an object literal (its first set is
        // the default, so order must be explicit), so the source-order AST node
        // is here. Inheritance merges the base's reserved at build time.
        expect_pat!(Node::Object(range), self.shared.arena.get(id));
        self.shared
            .pools
            .get_object(*range)
            .iter()
            .map(
                |&ObjectField {
                     name: name_span,
                     value: val_id,
                 }| {
                    let words = self.eval_rule_list(val_id)?;
                    Ok(ReservedWordContext {
                        name: ctx.text(name_span).to_string(),
                        reserved_words: words,
                    })
                },
            )
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
            C::Language => {
                let sid = self.strings.intern_owned(&grammar.name);
                Ok(self.alloc_val(Value::Str(sid)))
            }
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
                                    let sid = self.strings.intern_owned(s);
                                    self.alloc_val(Value::Str(sid))
                                }
                                PrecedenceEntry::Symbol(s) => self.owned_symbol_val(s),
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
                    Ok(self.owned_symbol_val(name))
                } else {
                    Err(self.err(LowerErrorKind::ConfigFieldUnset, span))
                }
            }
            C::Start => match grammar.variables.first() {
                // The start rule is the first variable.
                Some(first) => Ok(self.owned_symbol_val(&first.name)),
                // A config-only base (no rules) has no start rule; treat it like
                // any other unset config field.
                None => Err(self.err(LowerErrorKind::ConfigFieldUnset, span)),
            },
            C::Reserved => {
                let n = grammar.reserved_words.len();
                let mut map = FxHashMap::with_capacity_and_hasher(n, rustc_hash::FxBuildHasher);
                for rwc in &grammar.reserved_words {
                    let words_vid = self.import_rules_as_list(&rwc.reserved_words, span)?;
                    map.insert(rwc.name.clone(), words_vid);
                }
                Ok(self.alloc_object(map))
            }
            C::Inherits | C::Flags => unreachable!(),
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
            let vid = self.owned_symbol_val(name);
            self.state.ir.value_children.push(vid);
        }
        self.finish_list(start, names.len(), span)
    }

    fn import_rule(&mut self, rule: &Rule) -> RuleId {
        match rule {
            Rule::Blank => self.alloc_rule(ARule::Blank),
            Rule::String(s) => {
                let sid = self.strings.intern_owned(s);
                self.alloc_rule(ARule::String(sid))
            }
            Rule::Pattern(p, f) => {
                let pid = self.strings.intern_owned(p);
                let fid = (!f.is_empty()).then(|| self.strings.intern_owned(f));
                self.alloc_rule(ARule::Pattern(pid, fid))
            }
            Rule::NamedSymbol(name) => {
                let sid = self.strings.intern_owned(name);
                self.alloc_rule(ARule::NamedSymbol(sid))
            }
            Rule::Choice(members) | Rule::Seq(members) => {
                let is_seq = matches!(rule, Rule::Seq(_));
                self.state.scratch.rule_scratch.reserve(members.len());
                self.state.ir.rule_children.reserve(members.len());
                let (start, count) = stack_scope!(self.state.scratch.rule_scratch, |base| {
                    for m in members {
                        let rid = self.import_rule(m);
                        self.state.scratch.rule_scratch.push(rid);
                    }
                    let start = self.children_start();
                    self.state
                        .ir
                        .rule_children
                        .extend_from_slice(&self.state.scratch.rule_scratch[base..]);
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
                    let aprec = APrec::Name(self.strings.intern_owned(s));
                    rid = self.import_prec_rule(aprec, rid, params.associativity);
                }
                if let Some(crate::rules::Alias { value, is_named }) = &params.alias {
                    let sid = self.strings.intern_owned(value);
                    rid = self.alloc_rule(ARule::Alias(sid, *is_named, rid));
                }
                if let Some(field_name) = &params.field_name {
                    let sid = self.strings.intern_owned(field_name);
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
                let sid = self.strings.intern_owned(context_name);
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

    pub fn eval_expr(&mut self, id: NodeId) -> LowerResult<ValueId> {
        self.enter_eval(id)?;
        let result = self.eval_expr_inner(id);
        self.state.scratch.eval_depth -= 1;
        result
    }

    fn eval_expr_inner(&mut self, id: NodeId) -> LowerResult<ValueId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::IntLit(n) => {
                let v = i32::try_from(*n)
                    .map_err(|_| self.err(LowerErrorKind::IntegerOverflow(*n), span))?;
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
                        .map_err(|_| self.err(LowerErrorKind::IntegerOverflow(neg), span))?;
                    return Ok(self.alloc_val(Value::Int(v)));
                }
                let vid = self.eval_expr(*inner)?;
                // Guarded by super::typecheck::type_of (Node::Neg) - ensures inner is int
                let n = self.int_val(vid);
                let neg = n.checked_neg().ok_or_else(|| {
                    self.err(LowerErrorKind::IntegerOverflow(-i64::from(n)), span)
                })?;
                Ok(self.alloc_val(Value::Int(neg)))
            }
            Node::BinOp { op, lhs, rhs } => {
                let lhs_val = self.eval_expr(*lhs)?;
                let rhs_val = self.eval_expr(*rhs)?;
                // Guarded by typecheck (Node::BinOp), both sides are int.
                let l = self.int_val(lhs_val);
                let r = self.int_val(rhs_val);
                let (narrow, wide) = match op {
                    BinOp::Add => (l.checked_add(r), i64::from(l) + i64::from(r)),
                    BinOp::Sub => (l.checked_sub(r), i64::from(l) - i64::from(r)),
                };
                let v =
                    narrow.ok_or_else(|| self.err(LowerErrorKind::IntegerOverflow(wide), span))?;
                Ok(self.alloc_val(Value::Int(v)))
            }
            // Guarded by super::resolve + typecheck
            Node::Ident(IdentKind::Unresolved | IdentKind::Macro(_)) => unreachable!(),
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            // `@<expr>` computed-name ref: evaluate the name under the current
            // args (existence validated by resolve), then emit the symbol.
            &Node::SymRef { expr } => {
                let rid = self.lower_sym_ref(expr)?;
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::MacroParam { index, .. } => {
                let base = *self.state.scratch.macro_arg_bases.last().unwrap();
                Ok(self.state.scratch.macro_args[base + usize::from(*index)])
            }
            Node::ForBinding { for_id, index, .. } => {
                let base = self
                    .state
                    .scratch
                    .for_binding_frames
                    .iter()
                    .rev()
                    .find(|(id, _)| *id == *for_id)
                    .unwrap()
                    .1;
                Ok(self.state.scratch.for_binding_values[base + usize::from(*index)])
            }
            // On demand: a macro expanded at an earlier call site can reach a let
            // defined later in source order (resolve validated the name exists). A
            // read while the let is still mid-evaluation is a self-reference cycle,
            // reachable only through a macro since resolve rejects direct ones.
            &Node::Ident(IdentKind::Var(let_id)) => {
                if self.state.scratch.lets_in_progress.contains(&let_id) {
                    expect_pat!(Node::Let { name, .. }, *self.shared.arena.get(let_id));
                    let mut err = self.err(
                        LowerErrorKind::CircularLet(
                            self.module_ctx(self.current_module).text(name).to_owned(),
                        ),
                        self.shared.arena.span(let_id),
                    );
                    err.add_note(
                        self.module_ctx(self.current_module)
                            .note(NoteMessage::SelfReferenceHere, span),
                    );
                    Err(err)
                } else {
                    self.eval_let(let_id)
                }
            }
            &Node::GrammarConfig { module, field } => {
                let mod_val = self.eval_expr(module)?;
                expect_pat!(Value::Module(mod_idx), *self.get_val(mod_val));
                self.eval_grammar_config(mod_idx, field, span)
            }
            Node::ModuleRef { module, .. } => {
                let global_id = module.expect("module index not set by loading pre-pass");
                Ok(self.alloc_val(Value::Module(global_id)))
            }
            &Node::Append { left, right } => {
                let lv = self.eval_expr(left)?;
                let rv = self.eval_expr(right)?;
                // Guarded by super::typecheck::type_of (Node::Append) - ensures both are lists
                let lr = self.list_range(lv);
                let rr = self.list_range(rv);
                let start = self.state.ir.value_children.len() as u32;
                let total = usize::from(lr.len) + usize::from(rr.len);
                let len = self.checked_len(total, span)?;
                self.state
                    .ir
                    .value_children
                    .extend_from_within(lr.as_range());
                self.state
                    .ir
                    .value_children
                    .extend_from_within(rr.as_range());
                Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
            }
            &Node::FieldAccess { obj, field } => {
                let obj_val = self.eval_expr(obj)?;
                let field_name = self.module_ctx(self.current_module).text(field);
                // typecheck guarantees an object here, and validates the field
                // name for objects whose keys are statically known. For a
                // computed object (e.g. a macro result) the keys aren't known
                // until now, so a missing field is a lower-time error.
                expect_pat!(Value::Object(idx), *self.get_val(obj_val));
                let map = &self.state.ir.object_pool[idx as usize];
                map.get(field_name).copied().ok_or_else(|| {
                    let mut available: Vec<String> = map.keys().cloned().collect();
                    available.sort_unstable();
                    self.err(
                        LowerErrorKind::FieldNotFound {
                            field: field_name.to_string(),
                            available,
                        },
                        field,
                    )
                })
            }
            // A `mod::name` reference that resolve recorded as a concrete
            // target in another module's lowered output. Index directly; no
            // name scan. `&self.previous[..]` is `&'a Module`, decoupled from
            // the `&mut self` that `import_rule` needs.
            &Node::ModuleRule { module, target } => {
                let target_module = &self.previous[usize::from(module)];
                let rid = match target {
                    RuleTarget::ExternalName(span) => {
                        let sid = self.strings.intern_span(span, module);
                        self.alloc_rule(ARule::NamedSymbol(sid))
                    }
                    RuleTarget::HelperRule(i) => {
                        expect_pat!(Module::Helper { lowered_rules, .. }, target_module);
                        self.import_rule(&lowered_rules[i as usize].1)
                    }
                    RuleTarget::GrammarRule(i) => {
                        expect_pat!(Module::Grammar { lowered, .. }, target_module);
                        self.import_rule(&lowered.variables[i as usize].rule)
                    }
                    RuleTarget::GrammarExternal(i) => {
                        expect_pat!(Module::Grammar { lowered, .. }, target_module);
                        self.import_rule(&lowered.external_tokens[i as usize])
                    }
                };
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            &Node::Object(range) => {
                let fields = self.shared.pools.get_object(range);
                let mut map =
                    FxHashMap::with_capacity_and_hasher(fields.len(), rustc_hash::FxBuildHasher);
                for &ObjectField {
                    name: key_span,
                    value: value_id,
                } in fields
                {
                    map.insert(
                        self.module_ctx(self.current_module)
                            .text(key_span)
                            .to_string(),
                        self.eval_expr(value_id)?,
                    );
                }
                Ok(self.alloc_object(map))
            }
            node @ (&Node::List(range) | &Node::Tuple(range)) => {
                let is_list = matches!(node, Node::List(_));
                let items = self.shared.pools.child_slice(range);
                let range = stack_scope!(self.state.scratch.val_scratch, |base| {
                    for &item_id in items {
                        if let Node::For { for_id, body } = *self.shared.arena.get(item_id) {
                            self.eval_for_to_values(for_id, body)?;
                        } else {
                            let val = self.eval_expr(item_id)?;
                            self.state.scratch.val_scratch.push(val);
                        }
                    }
                    let start = self.state.ir.value_children.len() as u32;
                    let len =
                        self.checked_len(self.state.scratch.val_scratch.len() - base, span)?;
                    self.state
                        .ir
                        .value_children
                        .extend_from_slice(&self.state.scratch.val_scratch[base..]);
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
                let sid = self.strings.intern_owned(&result);
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

    pub fn lower_to_rule(&mut self, id: NodeId) -> LowerResult<RuleId> {
        self.enter_eval(id)?;
        let result = self.lower_to_rule_inner(id);
        self.state.scratch.eval_depth -= 1;
        result
    }

    fn lower_to_rule_inner(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                Ok(self.alloc_rule(ARule::NamedSymbol(sid)))
            }
            &Node::SymRef { expr } => self.lower_sym_ref(expr),
            Node::StringLit => {
                let sid = self.intern_string_lit(span);
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(span, *hash_count);
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            #[rustfmt::skip]
            Node::SeqOrChoice { .. } | Node::Repeat { .. } | Node::Blank
            | Node::Field { .. } | Node::Alias { .. } | Node::Token { .. }
            | Node::Prec { .. } | Node::Reserved { .. } => self.eval_combinator(id),
            _ => {
                let vid = self.eval_expr(id)?;
                Ok(match self.get_val(vid) {
                    Value::Rule(rid) => *rid,
                    Value::Str(s) => self.alloc_rule(ARule::String(*s)),
                    // Guarded by super::typecheck::expect_rule - only rule-like values reach here
                    _ => unreachable!(),
                })
            }
        }
    }

    fn eval_combinator(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            &Node::SeqOrChoice { seq, range } => {
                let children = self.shared.pools.child_slice(range);
                let child_range = stack_scope!(self.state.scratch.rule_scratch, |base| {
                    for &member in children {
                        if let Node::For { for_id, body } = *self.shared.arena.get(member) {
                            self.eval_for_to_rules(for_id, body)?;
                        } else {
                            let rid = self.lower_to_rule(member)?;
                            self.state.scratch.rule_scratch.push(rid);
                        }
                    }
                    let start = self.children_start();
                    self.state
                        .ir
                        .rule_children
                        .extend_from_slice(&self.state.scratch.rule_scratch[base..]);
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
                self.state
                    .ir
                    .rule_children
                    .extend_from_slice(&[first, blank]);
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
                                return Err(self.err(
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
        self.state.scratch.call_stack.push(CallFrame {
            name_span,
            name_mod,
            call_span,
            caller_mod: self.current_module,
        });
        if self.state.scratch.call_stack.len() > usize::from(MAX_CALL_DEPTH) {
            let root_span = self.state.scratch.call_stack[0].call_span;
            return Err(self.err(
                LowerErrorKind::CallDepthExceeded(self.build_call_trace()),
                root_span,
            ));
        }
        Ok(())
    }

    /// Build the macro-call trace (name + call-site `file:line:col` per frame)
    /// from the current `call_stack`, shared by the `CallDepthExceeded` and
    /// `RecursionTooDeep` errors so both render the same chain.
    fn build_call_trace(&self) -> Vec<(String, PathBuf, usize, usize)> {
        self.state
            .scratch
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
            .collect()
    }

    /// Enter one evaluator recursion frame, erroring if the depth cap is hit.
    /// The caller must decrement `eval_depth` once the guarded body returns.
    fn enter_eval(&mut self, id: NodeId) -> LowerResult<()> {
        if self.state.scratch.eval_depth >= MAX_EVAL_DEPTH {
            return Err(self.err(
                LowerErrorKind::RecursionTooDeep(self.build_call_trace()),
                self.shared.arena.span(id),
            ));
        }
        self.state.scratch.eval_depth += 1;
        Ok(())
    }

    /// Evaluate a call to a macro defined in `def_module`.
    /// Bind `args` (evaluated in the caller's context) on the macro-arg stack,
    /// then run `body_eval` with `current_module` set to `def_module`. Shared by
    /// expression-macro invocation (body -> value) and rule-set instantiation
    /// (template body -> rule).
    fn bind_args_and<R>(
        &mut self,
        name: Span,
        def_module: ModuleId,
        args: ChildRange,
        span: Span,
        body_eval: impl FnOnce(&mut Self) -> LowerResult<R>,
    ) -> LowerResult<R> {
        // Args evaluate in the caller's macro context, not the callee's, so
        // arg eval happens before macro_arg_bases is pushed.
        stack_scope!(
            self.state.scratch.macro_args => args_base,
            self.state.scratch.call_stack => _call_base,
            self.state.scratch.macro_arg_bases => _bases_base;
            {
                self.push_call(name, def_module, span)?;
                for &arg_id in self.shared.pools.child_slice(args) {
                    let v = self.eval_expr(arg_id)?;
                    self.state.scratch.macro_args.push(v);
                }
                self.state.scratch.macro_arg_bases.push(args_base);
                let saved = self.current_module;
                self.current_module = def_module;
                let result = body_eval(self);
                self.current_module = saved;
                result
            }
        )
    }

    fn invoke_macro(
        &mut self,
        macro_id: MacroId,
        args: ChildRange,
        span: Span,
        def_module: ModuleId,
    ) -> LowerResult<ValueId> {
        let config = self.shared.pools.get_macro(macro_id);
        let (name, body) = (config.name, config.body);
        self.bind_args_and(name, def_module, args, span, |e| e.eval_expr(body))
    }

    /// Lower one rule-set macro instance: bind the call's args, then lower the
    /// *shared* template body as a rule (no clone). `current_module` stays put -
    /// rule-set macros are local (qualified calls are rejected at parse time).
    pub(super) fn lower_expansion(
        &mut self,
        expand_id: ExpandId,
        span: Span,
    ) -> LowerResult<RuleId> {
        let exp = *self.shared.pools.get_expansion(expand_id);
        let name = self.shared.pools.get_macro(exp.macro_id).name;
        let def_module = self.current_module;
        self.bind_args_and(name, def_module, exp.args, span, |e| {
            e.lower_to_rule(exp.body)
        })
    }

    /// Lower a `@<expr>` computed-name reference: evaluate its name under the
    /// current args (resolve already validated it exists) and emit the symbol.
    fn lower_sym_ref(&mut self, expr: NodeId) -> LowerResult<RuleId> {
        let vid = self.eval_expr(expr)?;
        let name = self.str_id(vid);
        Ok(self.alloc_rule(ARule::NamedSymbol(name)))
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
                evaluator.state.scratch.rule_scratch.push(rule_id);
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
                evaluator.state.scratch.val_scratch.push(value_id);
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
        let n_items = usize::from(iter_range.len);
        stack_scope!(
            self.state.scratch.for_binding_values => base,
            self.state.scratch.for_binding_frames => _frames_base;
            {
                self.state.scratch.for_binding_frames.push((for_id, base));
                for i in 0..n_items {
                    let item_id = self.state.ir.value_children[iter_range.start as usize + i];
                    for j in 0..n_bindings {
                        // A single binding takes the whole item (scalar or
                        // tuple). Multiple bindings destructure a tuple item -
                        // typecheck guarantees the item is a tuple of matching
                        // arity, so the non-tuple case is unreachable.
                        let val = if n_bindings == 1 {
                            item_id
                        } else {
                            match *self.get_val(item_id) {
                                Value::Tuple(range) => {
                                    self.state.ir.value_children[range.start as usize + j]
                                }
                                _ => unreachable!(),
                            }
                        };
                        if i == 0 {
                            self.state.scratch.for_binding_values.push(val);
                        } else {
                            self.state.scratch.for_binding_values[base + j] = val;
                        }
                    }
                    each_iter(self)?;
                }
                Ok(())
            }
        )
    }
}
