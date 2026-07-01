//!  Walks the typed AST into intermediate IR; [`super`] materializes it into
//!  [`crate::grammars::InputGrammar`].

use std::path::PathBuf;

use rustc_hash::FxHashMap;

use super::{
    super::{
        LowerError, Module, ModuleId, NoteMessage,
        ast::{
            BinOp, ChildRange, ConfigField, ExpandId, ForId, IdentKind, MacroId, ModuleContext,
            Node, NodeId, PrecKind, RepeatKind, RuleTarget, SharedAst, Span,
        },
        lexer::unescape_string,
        string_pool::{Str, StrEntry, StringPool},
    },
    CallFrame, LowerErrorKind, LowerResult, LoweringState, MAX_CALL_DEPTH, MAX_RULE_DEPTH,
    repr::{APrec, ARule, RuleId, Value, ValueId},
};

use crate::{
    // `RuleId` here is the IR id (from `repr`); the flat-pool id is aliased.
    flat_rule::{FlatRules, RuleId as FlatRuleId},
    grammars::{PrecedenceEntry, ReservedWordContext},
    rules::{Alias, Associativity, Precedence, Rule},
};

/// One item on the lowering work stack. Expression and combinator nesting is
/// driven through this stack instead of native recursion, so arbitrarily deep
/// input cannot overflow the call stack. Macro / for-loop / let evaluation stays
/// recursive (bounded by `MAX_CALL_DEPTH` and for-nesting) and re-enters the
/// engine, nesting on the shared stacks via base offsets.
// Kept to 8 bytes (one `NodeId` payload): the work stack is the hottest memory in
// lowering, so every variant carries at most a node id. A `For` member holds the
// `For` node (its `for_id`/`body` are re-read on pop, both rare); a `Combine` of a
// variable-arity node pushes its result-stack base to `Scratch::combine_bases`
// rather than inline (popped when the combine runs - combines nest LIFO).
#[derive(Clone, Copy)]
pub(super) enum Task {
    /// Evaluate a node to a value; result lands on `val_scratch`.
    Expr(NodeId),
    /// Lower a node to a rule; result lands on `rule_scratch`.
    Rule(NodeId),
    /// A `For` node member of a list/tuple: expand to zero or more values.
    ForVal(NodeId),
    /// A `For` node member of a seq/choice: expand to zero or more rules.
    ForRule(NodeId),
    /// Fold a compound node's evaluated children.
    Combine(NodeId),
    /// Pop a rule off `rule_scratch` and wrap it as a `Value::Rule`.
    WrapRule,
    /// Pop a value off `val_scratch` and coerce it to a rule on `rule_scratch`.
    ExtractRule,
}

/// One step of [`Evaluator::build_rule`]'s explicit-stack walk over the IR. `Visit`
/// descends (carrying nesting depth for the [`MAX_RULE_DEPTH`] bound); `Build`
/// assembles a node from its already-materialized children.
#[derive(Clone, Copy)]
pub(super) enum BuildStep {
    Visit(RuleId, u16),
    Build(RuleId),
}

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
        let root_id = ModuleId::from(previous.len() as u8);
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

    /// Evaluate a let binding's value and memoize it under `let_id`, resolving the
    /// lets it transitively references first.
    ///
    /// Dependencies are walked with an explicit stack instead of native recursion,
    /// so an arbitrarily long `let` chain cannot overflow; a reference back into a
    /// let still being evaluated is a cycle. A cycle routed through a macro body is
    /// caught by [`eval_expr_inner`](Self::eval_expr_inner)'s `Var` arm instead,
    /// since the dependency scan does not descend into macro bodies.
    pub fn eval_let(&mut self, let_id: NodeId) -> LowerResult<ValueId> {
        if let Some(&val) = self.state.let_values.get(&let_id) {
            return Ok(val);
        }
        self.state.scratch.lets_in_progress.insert(let_id);
        let mut stack = vec![let_id];
        while let Some(&cur) = stack.last() {
            expect_pat!(Node::Let { value, .. }, *self.shared.arena.get(cur));
            let dep = {
                let resolved = &self.state.let_values;
                self.shared
                    .first_unresolved_let_dep(value, |id| resolved.contains_key(&id))
            };
            if let Some((dep, reference)) = dep {
                if !self.state.scratch.lets_in_progress.insert(dep) {
                    let span = self.shared.arena.span(reference);
                    return Err(self.circular_let_error(dep, span));
                }
                stack.push(dep);
                continue;
            }
            let val = self.eval_expr(value)?;
            self.state.let_values.insert(cur, val);
            self.state.scratch.lets_in_progress.remove(&cur);
            stack.pop();
        }
        Ok(self.state.let_values[&let_id])
    }

    /// Build a `CircularLet` error for `let_id`, with a self-reference note at
    /// `reference`.
    fn circular_let_error(&self, let_id: NodeId, reference: Span) -> LowerError {
        expect_pat!(Node::Let { name, .. }, *self.shared.arena.get(let_id));
        let mut err = self.err(
            LowerErrorKind::CircularLet(self.module_ctx(self.current_module).text(name).to_owned()),
            self.shared.arena.span(let_id),
        );
        err.add_note(
            self.module_ctx(self.current_module)
                .note(NoteMessage::SelfReferenceHere, reference),
        );
        err
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

    /// Materialize IR rule `root` into a [`Rule`] tree.
    ///
    /// Walks the IR with an explicit stack rather than recursing, so a deeply
    /// nested rule does not overflow the native stack here. The depth bound is
    /// for the *consumer*: the resulting tree feeds the recursive shared backend
    /// ([`MAX_RULE_DEPTH`]).
    ///
    /// # Errors
    /// [`LowerErrorKind::RuleNestingTooDeep`] if nesting exceeds [`MAX_RULE_DEPTH`].
    pub fn build_rule(&mut self, root: RuleId) -> LowerResult<Rule> {
        // Two-phase post-order walk: `Visit` descends (pushing a matching `Build`
        // plus its children), `Build` pops the children's finished `Rule`s off
        // `out` and assembles the parent. Children are pushed reversed so they
        // land on `out` in source order. The two stacks are taken from scratch so
        // their capacity is reused across the per-rule calls rather than reallocated;
        // both are empty here (drained each call) and restored before returning.
        let mut work = std::mem::take(&mut self.state.scratch.build_work);
        let mut out = std::mem::take(&mut self.state.scratch.build_out);
        work.push(BuildStep::Visit(root, 0));
        while let Some(step) = work.pop() {
            match step {
                BuildStep::Visit(id, depth) => {
                    if depth > MAX_RULE_DEPTH {
                        return Err(LowerError::without_span(LowerErrorKind::RuleNestingTooDeep));
                    }
                    match self.get_rule(id) {
                        ARule::Blank => out.push(Rule::Blank),
                        ARule::String(sid) => out.push(Rule::String(self.str_to_string(*sid))),
                        ARule::Pattern(p, f) => out.push(Rule::Pattern(
                            self.str_to_string(*p),
                            f.map_or_else(String::new, |f| self.str_to_string(f)),
                        )),
                        ARule::NamedSymbol(sid) => {
                            out.push(Rule::NamedSymbol(self.str_to_string(*sid)));
                        }
                        ARule::SeqOrChoice(_, range) => {
                            work.push(BuildStep::Build(id));
                            // Safety: range is produced by children_start/children_range
                            // which track rule_children indices.
                            let children = unsafe {
                                self.state.ir.rule_children.get_unchecked(range.as_range())
                            };
                            for &child in children.iter().rev() {
                                work.push(BuildStep::Visit(child, depth + 1));
                            }
                        }
                        ARule::Repeat(r)
                        | ARule::Prec(_, r)
                        | ARule::PrecLeft(_, r)
                        | ARule::PrecRight(_, r)
                        | ARule::PrecDynamic(_, r)
                        | ARule::Field(_, r)
                        | ARule::Alias(_, _, r)
                        | ARule::Token(_, r)
                        | ARule::Reserved(_, r) => {
                            work.push(BuildStep::Build(id));
                            work.push(BuildStep::Visit(*r, depth + 1));
                        }
                    }
                }
                // A Build step's children are already on `out` (one Visit each).
                // SeqOrChoice consumes `range.len` of them; every other compound
                // wraps exactly one, so they share a single pop site.
                BuildStep::Build(id) => {
                    let rule = self.get_rule(id);
                    if let ARule::SeqOrChoice(is_seq, range) = rule {
                        let children = out.split_off(out.len() - range.len as usize);
                        out.push(if *is_seq {
                            Rule::seq(children)
                        } else {
                            Rule::choice(children)
                        });
                        continue;
                    }
                    let inner = out.pop().unwrap();
                    out.push(match rule {
                        ARule::Repeat(_) => Rule::repeat(inner),
                        ARule::Prec(p, _) => Rule::prec(self.build_prec(*p), inner),
                        ARule::PrecLeft(p, _) => Rule::prec_left(self.build_prec(*p), inner),
                        ARule::PrecRight(p, _) => Rule::prec_right(self.build_prec(*p), inner),
                        ARule::PrecDynamic(v, _) => Rule::prec_dynamic(*v, inner),
                        ARule::Field(n, _) => Rule::field(self.str_to_string(*n), inner),
                        ARule::Alias(v, named, _) => {
                            Rule::alias(inner, self.str_to_string(*v), *named)
                        }
                        ARule::Token(imm, _) => {
                            if *imm {
                                Rule::immediate_token(inner)
                            } else {
                                Rule::token(inner)
                            }
                        }
                        ARule::Reserved(ctx, _) => Rule::Reserved {
                            rule: Box::new(inner),
                            context_name: self.str_to_string(*ctx),
                        },
                        // Leaves never enqueue a Build step.
                        _ => unreachable!(),
                    });
                }
            }
        }
        // Exactly the root's finished rule remains; return the now-empty buffers
        // to scratch so the next call reuses their capacity.
        let result = out.pop().unwrap();
        self.state.scratch.build_work = work;
        self.state.scratch.build_out = out;
        Ok(result)
    }

    /// Direct-emission counterpart of [`build_rule`](Self::build_rule): the same
    /// post-order walk over the eval IR, but each finished node is interned into
    /// `pool` (returning a [`FlatRuleId`]) instead of boxed into a `Rule`. This
    /// skips the intermediate `Rule` box tree that `build_rule` + `intern_import`
    /// would allocate (the malloc win). Metadata wrappers, which the IR keeps
    /// unbundled, are merged into a single `MetadataParams` exactly as `Rule`'s
    /// constructors do (via [`FlatRules::intern_metadata_merged`]), and choices are
    /// flattened/deduped ([`FlatRules::intern_choice_flattened`]), so the interned
    /// result is structurally identical to interning the built `Rule`.
    pub fn intern_arule(&mut self, pool: &mut FlatRules, root: RuleId) -> LowerResult<FlatRuleId> {
        let mut work = std::mem::take(&mut self.state.scratch.build_work);
        let mut out: Vec<FlatRuleId> = Vec::new();
        work.push(BuildStep::Visit(root, 0));
        while let Some(step) = work.pop() {
            match step {
                BuildStep::Visit(id, depth) => {
                    if depth > MAX_RULE_DEPTH {
                        return Err(LowerError::without_span(LowerErrorKind::RuleNestingTooDeep));
                    }
                    match self.get_rule(id) {
                        ARule::Blank => out.push(pool.intern_blank()),
                        ARule::String(sid) => {
                            let s = self.str_to_string(*sid);
                            out.push(pool.intern_string(s));
                        }
                        ARule::Pattern(p, f) => {
                            let pat = self.str_to_string(*p);
                            let flags = f.map_or_else(String::new, |f| self.str_to_string(f));
                            out.push(pool.intern_pattern(pat, flags));
                        }
                        ARule::NamedSymbol(sid) => {
                            let n = self.str_to_string(*sid);
                            out.push(pool.intern_named_symbol(n));
                        }
                        ARule::SeqOrChoice(_, range) => {
                            work.push(BuildStep::Build(id));
                            // Safety: as in `build_rule`, `range` comes from
                            // children_start/children_range over `rule_children`.
                            let children = unsafe {
                                self.state.ir.rule_children.get_unchecked(range.as_range())
                            };
                            for &child in children.iter().rev() {
                                work.push(BuildStep::Visit(child, depth + 1));
                            }
                        }
                        ARule::Repeat(r)
                        | ARule::Prec(_, r)
                        | ARule::PrecLeft(_, r)
                        | ARule::PrecRight(_, r)
                        | ARule::PrecDynamic(_, r)
                        | ARule::Field(_, r)
                        | ARule::Alias(_, _, r)
                        | ARule::Token(_, r)
                        | ARule::Reserved(_, r) => {
                            work.push(BuildStep::Build(id));
                            work.push(BuildStep::Visit(*r, depth + 1));
                        }
                    }
                }
                BuildStep::Build(id) => {
                    let rule = self.get_rule(id);
                    if let ARule::SeqOrChoice(is_seq, range) = rule {
                        let is_seq = *is_seq;
                        let children = out.split_off(out.len() - range.len as usize);
                        out.push(if is_seq {
                            pool.intern_seq_or_choice(true, &children)
                        } else {
                            pool.intern_choice_flattened(&children)
                        });
                        continue;
                    }
                    let inner = out.pop().unwrap();
                    let interned = match rule {
                        ARule::Repeat(_) => pool.intern_repeat(inner),
                        ARule::Prec(p, _) => {
                            let prec = self.build_prec(*p);
                            pool.intern_metadata_merged(inner, |m| m.precedence = prec)
                        }
                        ARule::PrecLeft(p, _) => {
                            let prec = self.build_prec(*p);
                            pool.intern_metadata_merged(inner, |m| {
                                m.associativity = Some(Associativity::Left);
                                m.precedence = prec;
                            })
                        }
                        ARule::PrecRight(p, _) => {
                            let prec = self.build_prec(*p);
                            pool.intern_metadata_merged(inner, |m| {
                                m.associativity = Some(Associativity::Right);
                                m.precedence = prec;
                            })
                        }
                        ARule::PrecDynamic(v, _) => {
                            let v = *v;
                            pool.intern_metadata_merged(inner, |m| m.dynamic_precedence = v)
                        }
                        ARule::Field(n, _) => {
                            let name = self.str_to_string(*n);
                            pool.intern_metadata_merged(inner, |m| m.field_name = Some(name))
                        }
                        ARule::Alias(v, named, _) => {
                            let value = self.str_to_string(*v);
                            let is_named = *named;
                            pool.intern_metadata_merged(inner, |m| {
                                m.alias = Some(Alias { value, is_named });
                            })
                        }
                        ARule::Token(imm, _) => {
                            let imm = *imm;
                            pool.intern_metadata_merged(inner, |m| {
                                m.is_token = true;
                                if imm {
                                    m.is_main_token = true;
                                }
                            })
                        }
                        ARule::Reserved(ctx, _) => {
                            let ctx = self.str_to_string(*ctx);
                            pool.intern_reserved(&ctx, inner)
                        }
                        // Leaves never enqueue a Build step.
                        _ => unreachable!(),
                    };
                    out.push(interned);
                }
            }
        }
        let result = out.pop().unwrap();
        self.state.scratch.build_work = work;
        Ok(result)
    }

    fn build_prec(&self, prec: APrec) -> Precedence {
        match prec {
            APrec::Integer(n) => Precedence::Integer(n),
            APrec::Name(sid) => Precedence::Name(self.str_to_string(sid)),
        }
    }

    fn value_to_owned_rule(&mut self, pool: &mut FlatRules, id: ValueId) -> LowerResult<FlatRuleId> {
        match *self.get_val(id) {
            Value::Rule(rid) => self.intern_arule(pool, rid),
            Value::Str(sid) => {
                let s = self.str_to_string(sid);
                Ok(pool.intern_string(s))
            }
            // Guarded by super::typecheck - only rule-like values reach here
            _ => unreachable!(),
        }
    }

    pub fn eval_rule_list(
        &mut self,
        pool: &mut FlatRules,
        id: NodeId,
    ) -> LowerResult<Vec<FlatRuleId>> {
        let vid = self.eval_expr(id)?;
        // Read each item by index: `value_to_owned_rule` needs `&mut self`, so we
        // can't hold a borrow of the value_children slice.
        let range = self.list_range(vid);
        let mut rules = Vec::with_capacity(range.len as usize);
        for i in range.as_range() {
            let v = self.state.ir.value_children[i];
            rules.push(self.value_to_owned_rule(pool, v)?);
        }
        Ok(rules)
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
    pub fn eval_reserved(
        &mut self,
        pool: &mut FlatRules,
        id: NodeId,
    ) -> LowerResult<Vec<ReservedWordContext<FlatRuleId>>> {
        // typecheck guarantees `reserved` is an object literal (its first set is
        // the default, so order must be explicit), so the source-order AST node
        // is here. Inheritance merges the base's reserved at build time.
        expect_pat!(Node::Object(range), self.shared.arena.get(id));
        // Copy out (name, value) pairs so the `shared.pools` borrow drops before
        // the `&mut self` calls below (`eval_rule_list`, `module_ctx`).
        let fields: Vec<(Span, NodeId)> = self
            .shared
            .pools
            .get_object(*range)
            .iter()
            .map(|f| (f.name, f.value))
            .collect();
        let mut out = Vec::with_capacity(fields.len());
        for (name_span, val_id) in fields {
            let reserved_words = self.eval_rule_list(pool, val_id)?;
            let name = self.module_ctx(self.current_module).text(name_span).to_string();
            out.push(ReservedWordContext { name, reserved_words });
        }
        Ok(out)
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
            C::Extras => self.import_rules_as_list(&grammar.extra_symbols, &grammar.pool, span),
            C::Externals => {
                self.import_rules_as_list(&grammar.external_tokens, &grammar.pool, span)
            }
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
                    let words_vid = self.import_rules_as_list(&rwc.reserved_words, &grammar.pool, span)?;
                    map.insert(rwc.name.clone(), words_vid);
                }
                Ok(self.alloc_object(map))
            }
            C::Inherits | C::Flags => unreachable!(),
        }
    }

    fn import_rules_as_list(
        &mut self,
        rules_data: &[FlatRuleId],
        pool: &FlatRules,
        span: Span,
    ) -> LowerResult<ValueId> {
        let start = self.state.ir.value_children.len() as u32;
        self.state.ir.value_children.reserve(rules_data.len());
        for &id in rules_data {
            // Interim: materialize the base's pooled rule, then import into the IR.
            // At SP4 this imports from the pool directly (materialize deleted).
            let rid = self.import_rule(&pool.materialize(id));
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

    /// Import an external [`Rule`] tree (from an inherited base grammar) into the
    /// IR, returning its [`RuleId`].
    ///
    /// Post-order walk over an explicit stack rather than native recursion, so a
    /// deep base rule cannot overflow here. Children's `RuleId`s accumulate on
    /// `rule_scratch` (base-relative, so this nests safely inside an in-progress
    /// rule build); each `Build` consumes its children and pushes its own result.
    fn import_rule(&mut self, root: &Rule) -> RuleId {
        enum Step<'r> {
            Visit(&'r Rule),
            Build(&'r Rule),
        }
        let mut work = vec![Step::Visit(root)];
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(rule) => match rule {
                    Rule::Blank => self.push_rule(ARule::Blank),
                    Rule::String(s) => {
                        let sid = self.strings.intern_owned(s);
                        self.push_rule(ARule::String(sid));
                    }
                    Rule::Pattern(p, f) => {
                        let pid = self.strings.intern_owned(p);
                        let fid = (!f.is_empty()).then(|| self.strings.intern_owned(f));
                        self.push_rule(ARule::Pattern(pid, fid));
                    }
                    Rule::NamedSymbol(name) => {
                        let sid = self.strings.intern_owned(name);
                        self.push_rule(ARule::NamedSymbol(sid));
                    }
                    Rule::Seq(members) | Rule::Choice(members) => {
                        self.state.scratch.rule_scratch.reserve(members.len());
                        self.state.ir.rule_children.reserve(members.len());
                        work.push(Step::Build(rule));
                        for m in members.iter().rev() {
                            work.push(Step::Visit(m));
                        }
                    }
                    Rule::Repeat(inner)
                    | Rule::Metadata { rule: inner, .. }
                    | Rule::Reserved { rule: inner, .. } => {
                        work.push(Step::Build(rule));
                        work.push(Step::Visit(inner));
                    }
                    Rule::Symbol(_) => unreachable!(),
                },
                Step::Build(rule) => match rule {
                    Rule::Seq(members) | Rule::Choice(members) => {
                        let is_seq = matches!(rule, Rule::Seq(_));
                        let scratch_base = self.state.scratch.rule_scratch.len() - members.len();
                        let start = self.children_start();
                        self.state
                            .ir
                            .rule_children
                            .extend_from_slice(&self.state.scratch.rule_scratch[scratch_base..]);
                        self.state.scratch.rule_scratch.truncate(scratch_base);
                        let count = self.state.ir.rule_children.len() as u32 - start;
                        debug_assert!(u16::try_from(count).is_ok());
                        self.push_rule(ARule::SeqOrChoice(is_seq, ChildRange::new(start, count as u16)));
                    }
                    Rule::Repeat(_) => {
                        let inner = self.pop_rule();
                        self.push_rule(ARule::Repeat(inner));
                    }
                    Rule::Metadata { params, .. } => {
                        let mut rid = self.pop_rule();
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
                        self.push_rule_id(rid);
                    }
                    Rule::Reserved { context_name, .. } => {
                        let inner = self.pop_rule();
                        let sid = self.strings.intern_owned(context_name);
                        self.push_rule(ARule::Reserved(sid, inner));
                    }
                    // Leaves never enqueue a Build step.
                    _ => unreachable!(),
                },
            }
        }
        self.pop_rule()
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

    /// Evaluate an expression node to a [`ValueId`].
    ///
    /// Drives the shared work stack (see [`Task`]) rather than recursing, so an
    /// arbitrarily deep expression cannot overflow the native stack. Re-entrant
    /// callers (macro / for-loop / let evaluation) nest naturally: each records
    /// the current work length, drives only the work it pushed, and leaves exactly
    /// one result on `val_scratch`. On error the shared stacks are left as-is - the
    /// error aborts the whole lowering and [`reset_per_grammar`] clears the scratch
    /// before the next grammar, so unwinding them here would be dead work.
    ///
    /// [`reset_per_grammar`]: super::LoweringState::reset_per_grammar
    pub fn eval_expr(&mut self, id: NodeId) -> LowerResult<ValueId> {
        let work_base = self.state.scratch.work.len();
        self.push_task(Task::Expr(id));
        self.drive(work_base)?;
        Ok(self.pop_val())
    }

    /// Lower an expression node to a [`RuleId`], driving the same work stack as
    /// [`eval_expr`](Self::eval_expr) and leaving one result on `rule_scratch`.
    pub fn lower_to_rule(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let work_base = self.state.scratch.work.len();
        self.push_task(Task::Rule(id));
        self.drive(work_base)?;
        Ok(self.pop_rule())
    }

    /// Run the work stack down to `work_base` (this walk's starting length, so
    /// re-entrant macro/for/let walks nest).
    fn drive(&mut self, work_base: usize) -> LowerResult<()> {
        while self.state.scratch.work.len() > work_base {
            let task = self.state.scratch.work.pop().unwrap();
            self.step(task)?;
        }
        Ok(())
    }

    /// Execute one work item: dispatch a node, expand a for-loop member, fold
    /// evaluated children, or convert between the value and rule stacks.
    fn step(&mut self, task: Task) -> LowerResult<()> {
        match task {
            Task::Expr(id) => self.dispatch_expr(id),
            Task::Rule(id) => {
                self.dispatch_rule(id);
                Ok(())
            }
            Task::ForVal(node) => {
                expect_pat!(Node::For { for_id, body }, *self.shared.arena.get(node));
                self.eval_for_to_values(for_id, body)
            }
            Task::ForRule(node) => {
                expect_pat!(Node::For { for_id, body }, *self.shared.arena.get(node));
                self.eval_for_to_rules(for_id, body)
            }
            Task::Combine(id) => self.combine(id),
            Task::WrapRule => {
                let rid = self.pop_rule();
                self.push_val(Value::Rule(rid));
                Ok(())
            }
            Task::ExtractRule => {
                let vid = self.pop_val();
                // Guarded by super::typecheck::expect_rule - only rule-like values reach here
                let rid = match *self.get_val(vid) {
                    Value::Rule(rid) => rid,
                    Value::Str(s) => self.alloc_rule(ARule::String(s)),
                    _ => unreachable!(),
                };
                self.push_rule_id(rid);
                Ok(())
            }
        }
    }

    /// Pop the single result a finished sub-walk left on `val_scratch`. The
    /// engine maintains the invariant that every value-producing task pushes
    /// exactly one result, so a missing one is an internal bug, not user input.
    fn pop_val(&mut self) -> ValueId {
        self.state.scratch.val_scratch.pop().unwrap()
    }

    /// Pop the single result a finished sub-walk left on `rule_scratch`; relies
    /// on the same one-result-per-task discipline as [`pop_val`](Self::pop_val).
    fn pop_rule(&mut self) -> RuleId {
        self.state.scratch.rule_scratch.pop().unwrap()
    }

    fn push_val(&mut self, val: Value) {
        let id = self.alloc_val(val);
        self.state.scratch.val_scratch.push(id);
    }

    fn push_val_id(&mut self, id: ValueId) {
        self.state.scratch.val_scratch.push(id);
    }

    fn push_rule(&mut self, rule: ARule) {
        let id = self.alloc_rule(rule);
        self.state.scratch.rule_scratch.push(id);
    }

    fn push_rule_id(&mut self, id: RuleId) {
        self.state.scratch.rule_scratch.push(id);
    }

    /// Push a fixed-arity combine (pops a known child count; no base needed).
    fn push_combine(&mut self, id: NodeId) {
        self.push_task(Task::Combine(id));
    }

    /// Push a variable-arity combine, recording where its children begin on the
    /// result stack (read back in [`combine`](Self::combine) via `combine_bases`).
    fn push_combine_var(&mut self, id: NodeId, base: usize) {
        self.state.scratch.combine_bases.push(base as u32);
        self.push_task(Task::Combine(id));
    }

    fn push_task(&mut self, task: Task) {
        self.state.scratch.work.push(task);
    }

    /// If `id` is a leaf rule node (no children), allocate it and return its
    /// [`RuleId`]; otherwise return `None`. Lets the hot paths - leaf members of a
    /// wide node, and leaf children of a single-child wrapper (`token("kw")`,
    /// `field("n", x)`) - build without a work-stack round-trip. Mirrors
    /// [`dispatch_rule`](Self::dispatch_rule)'s leaf arms; kept separate so
    /// `dispatch_rule` avoids a redundant leaf check on its (common) non-leaf input.
    fn try_leaf_rule(&mut self, id: NodeId) -> Option<RuleId> {
        let span = self.shared.arena.span(id);
        let rule = match self.shared.arena.get(id) {
            Node::Ident(IdentKind::Rule) => ARule::NamedSymbol(self.intern_span(span)),
            Node::StringLit => ARule::String(self.intern_string_lit(span)),
            Node::RawStringLit { hash_count } => {
                ARule::String(self.intern_raw_string_lit(span, *hash_count))
            }
            Node::Blank => ARule::Blank,
            _ => return None,
        };
        Some(self.alloc_rule(rule))
    }

    /// Inline a leaf member onto `rule_scratch` (see [`push_spread_items`]).
    fn try_push_leaf_rule(&mut self, id: NodeId) -> bool {
        match self.try_leaf_rule(id) {
            Some(rid) => {
                self.push_rule_id(rid);
                true
            }
            None => false,
        }
    }

    /// Enqueue a list/tuple `range`'s members reversed, so they run in source
    /// order; a `For` member becomes a deferred for-task. (Seq/choice members are
    /// handled inline by [`dispatch_rule`](Self::dispatch_rule), which can skip the
    /// combine for an all-leaf node.)
    fn push_spread_items(&mut self, range: ChildRange, as_rule: bool) {
        for &item in self.shared.pools.child_slice(range).iter().rev() {
            let task = match self.shared.arena.get(item) {
                Node::For { .. } if as_rule => Task::ForRule(item),
                Node::For { .. } => Task::ForVal(item),
                _ if as_rule => Task::Rule(item),
                _ => Task::Expr(item),
            };
            self.push_task(task);
        }
    }

    /// Import a `mod::name` reference resolve recorded as a concrete target in
    /// another module's lowered output.
    fn import_module_rule(&mut self, module: ModuleId, target: RuleTarget) -> RuleId {
        let target_module = &self.previous[usize::from(module)];
        match target {
            RuleTarget::HelperRule(i) => {
                expect_pat!(Module::Helper { lowered_rules, .. }, target_module);
                self.import_rule(&lowered_rules[i as usize].1)
            }
            RuleTarget::GrammarRule(i) => {
                expect_pat!(Module::Grammar { lowered, .. }, target_module);
                self.import_rule(&lowered.pool.materialize(lowered.variables[i as usize].rule))
            }
            RuleTarget::GrammarExternal(i) => {
                expect_pat!(Module::Grammar { lowered, .. }, target_module);
                self.import_rule(&lowered.pool.materialize(lowered.external_tokens[i as usize]))
            }
        }
    }

    /// Dispatch a node in value position: push its result on `val_scratch`
    /// (leaves) or enqueue its children plus a [`Task::Combine`] to fold them
    /// (compounds). Mirrors the arms of the former recursive `eval_expr_inner`.
    fn dispatch_expr(&mut self, id: NodeId) -> LowerResult<()> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::IntLit(n) => {
                let v = i32::try_from(*n)
                    .map_err(|_| self.err(LowerErrorKind::IntegerOverflow(*n), span))?;
                self.push_val(Value::Int(v));
            }
            Node::StringLit => {
                let sid = self.intern_string_lit(span);
                self.push_val(Value::Str(sid));
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(span, *hash_count);
                self.push_val(Value::Str(sid));
            }
            Node::Neg(inner) => {
                // Special-case Neg over a literal so the most negative i32
                // (-i32::MIN as positive overflows i32, but fits in i64) is
                // reachable as `-2147483648`.
                if let Node::IntLit(m) = *self.shared.arena.get(*inner) {
                    let neg = -m;
                    let v = i32::try_from(neg)
                        .map_err(|_| self.err(LowerErrorKind::IntegerOverflow(neg), span))?;
                    self.push_val(Value::Int(v));
                } else {
                    let inner = *inner;
                    self.push_combine(id);
                    self.push_task(Task::Expr(inner));
                }
            }
            &Node::BinOp { lhs, rhs, .. } => {
                self.push_combine(id);
                self.push_task(Task::Expr(rhs));
                self.push_task(Task::Expr(lhs));
            }
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                self.push_val(Value::Rule(rid));
            }
            Node::MacroParam { index, .. } => {
                let base = *self.state.scratch.macro_arg_bases.last().unwrap();
                let v = self.state.scratch.macro_args[base + usize::from(*index)];
                self.push_val_id(v);
            }
            Node::ForBinding { for_id, index, .. } => {
                let (for_id, index) = (*for_id, usize::from(*index));
                let base = self
                    .state
                    .scratch
                    .for_binding_frames
                    .iter()
                    .rev()
                    .find(|(fid, _)| *fid == for_id)
                    .unwrap()
                    .1;
                let v = self.state.scratch.for_binding_values[base + index];
                self.push_val_id(v);
            }
            // On demand: a macro expanded at an earlier call site can reach a let
            // defined later in source order (resolve validated the name exists). A
            // read while the let is still mid-evaluation is a self-reference cycle,
            // reachable only through a macro since resolve rejects direct ones.
            &Node::Ident(IdentKind::Var(let_id)) => {
                if self.state.scratch.lets_in_progress.contains(&let_id) {
                    return Err(self.circular_let_error(let_id, span));
                }
                let v = self.eval_let(let_id)?;
                self.push_val_id(v);
            }
            &Node::GrammarConfig { module, .. } => {
                self.push_combine(id);
                self.push_task(Task::Expr(module));
            }
            Node::ModuleRef { module, .. } => {
                let global_id = module.expect("module index not set by loading pre-pass");
                self.push_val(Value::Module(global_id));
            }
            &Node::Append { left, right } => {
                self.push_combine(id);
                self.push_task(Task::Expr(right));
                self.push_task(Task::Expr(left));
            }
            &Node::FieldAccess { obj, .. } => {
                self.push_combine(id);
                self.push_task(Task::Expr(obj));
            }
            &Node::ModuleRule { module, target } => {
                let rid = self.import_module_rule(module, target);
                self.push_val(Value::Rule(rid));
            }
            &Node::Object(range) => {
                let base = self.state.scratch.val_scratch.len();
                self.push_combine_var(id, base);
                for field in self.shared.pools.get_object(range).iter().rev() {
                    self.push_task(Task::Expr(field.value));
                }
            }
            &Node::List(range) | &Node::Tuple(range) => {
                let base = self.state.scratch.val_scratch.len();
                self.push_combine_var(id, base);
                self.push_spread_items(range, false);
            }
            &Node::Concat(range) => {
                let base = self.state.scratch.val_scratch.len();
                self.push_combine_var(id, base);
                for &part in self.shared.pools.child_slice(range).iter().rev() {
                    self.push_task(Task::Expr(part));
                }
            }
            &Node::DynRegex { pattern, flags } => {
                let base = self.state.scratch.val_scratch.len();
                self.push_combine_var(id, base);
                // pattern lands at [base], the optional flags (if any) at [base + 1].
                if let Some(flags) = flags {
                    self.push_task(Task::Expr(flags));
                }
                self.push_task(Task::Expr(pattern));
            }
            &Node::Call { name, args } => {
                expect_pat!(
                    Node::Ident(IdentKind::Macro(macro_id)),
                    self.shared.arena.get(name)
                );
                // Macro expansion stays recursive, bounded by MAX_CALL_DEPTH.
                let v = self.invoke_macro(*macro_id, args, span, self.current_module)?;
                self.push_val_id(v);
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
                let v = self.invoke_macro(*macro_id, args, span, mod_idx)?;
                self.push_val_id(v);
            }
            // SymRef and the combinator nodes produce a rule; lower as a rule,
            // then wrap it as a `Value::Rule`. Listed explicitly (rather than a
            // blanket arm) so an expression node missing above fails fast here
            // instead of ping-ponging with `dispatch_rule`'s value fallthrough.
            #[rustfmt::skip]
            Node::SymRef { .. } | Node::SeqOrChoice { .. } | Node::Repeat { .. }
            | Node::Blank | Node::Field { .. } | Node::Alias { .. }
            | Node::Token { .. } | Node::Prec { .. } | Node::Reserved { .. } => {
                self.push_task(Task::WrapRule);
                self.push_task(Task::Rule(id));
            }
            // Unresolved/Macro idents are replaced before lower, and non-expression
            // nodes never reach here; anything else is an internal invariant break.
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Dispatch a node in rule position: push its `RuleId` on `rule_scratch`
    /// (leaves) or enqueue its children plus a [`Task::Combine`] (compounds).
    /// Merges the former recursive `lower_to_rule_inner` and `eval_combinator`.
    fn dispatch_rule(&mut self, id: NodeId) {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                self.push_rule(ARule::NamedSymbol(sid));
            }
            Node::StringLit => {
                let sid = self.intern_string_lit(span);
                self.push_rule(ARule::String(sid));
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(span, *hash_count);
                self.push_rule(ARule::String(sid));
            }
            Node::Blank => self.push_rule(ARule::Blank),
            // `@<expr>` computed-name ref: evaluate the name under the current
            // args (existence validated by resolve), then emit the symbol.
            &Node::SymRef { expr } => {
                self.push_combine(id);
                self.push_task(Task::Expr(expr));
            }
            &Node::SeqOrChoice { seq, range } => {
                let items = self.shared.pools.child_slice(range);
                let base = self.state.scratch.rule_scratch.len();
                // Inline the leading run of leaf members in one pass.
                let mut i = 0;
                while i < items.len() && self.try_push_leaf_rule(items[i]) {
                    i += 1;
                }
                if i == items.len() {
                    // Every member was a leaf: build inline, no combine round-trip
                    // (common for a `choice` of symbol refs). Child count equals
                    // `range.len` (<= u16::MAX), so the conversion can't overflow.
                    let start = self.children_start();
                    self.state
                        .ir
                        .rule_children
                        .extend_from_slice(&self.state.scratch.rule_scratch[base..]);
                    self.state.scratch.rule_scratch.truncate(base);
                    let len = (self.state.ir.rule_children.len() as u32 - start) as u16;
                    self.push_rule(ARule::SeqOrChoice(seq, ChildRange::new(start, len)));
                    return;
                }
                // Mixed: fold the rest in the combine, pushing remaining members
                // (the first non-leaf onward) reversed so they run in source order.
                self.push_combine_var(id, base);
                for &item in items[i..].iter().rev() {
                    let task = match self.shared.arena.get(item) {
                        Node::For { .. } => Task::ForRule(item),
                        _ => Task::Rule(item),
                    };
                    self.push_task(task);
                }
            }
            // Single-rule-child wrappers. When the child is a leaf, build inline -
            // skipping a combine round-trip for the ubiquitous `token("kw")` /
            // `field("n", x)` shapes; otherwise defer the child and fold in the
            // combine. Allocation + intern order matches the deferred path, so the
            // IR stays identical.
            &Node::Token { immediate, inner } => {
                if let Some(c) = self.try_leaf_rule(inner) {
                    self.push_rule(ARule::Token(immediate, c));
                } else {
                    self.push_combine(id);
                    self.push_task(Task::Rule(inner));
                }
            }
            &Node::Field { name, content } => {
                if let Some(c) = self.try_leaf_rule(content) {
                    let sid = self.intern_span(name);
                    self.push_rule(ARule::Field(sid, c));
                } else {
                    self.push_combine(id);
                    self.push_task(Task::Rule(content));
                }
            }
            &Node::Reserved { context, content } => {
                if let Some(c) = self.try_leaf_rule(content) {
                    let sid = self.intern_span(context);
                    self.push_rule(ARule::Reserved(sid, c));
                } else {
                    self.push_combine(id);
                    self.push_task(Task::Rule(content));
                }
            }
            &Node::Repeat { inner, .. } => {
                self.push_combine(id);
                self.push_task(Task::Rule(inner));
            }
            // Prec's content lands on rule_scratch and its value on val_scratch;
            // the value is pushed last so it pops first in the combine.
            &Node::Prec { value, content, .. } => {
                self.push_combine(id);
                self.push_task(Task::Rule(content));
                self.push_task(Task::Expr(value));
            }
            // A named-target alias (`alias(x, name)`) needs only its content
            // lowered; any other target is an expression evaluated to a value.
            &Node::Alias { content, target } => {
                self.push_combine(id);
                self.push_task(Task::Rule(content));
                if !matches!(self.shared.arena.get(target), Node::Ident(IdentKind::Rule)) {
                    self.push_task(Task::Expr(target));
                }
            }
            // Any other node produces a value; evaluate it, then coerce to a rule.
            _ => {
                self.push_task(Task::ExtractRule);
                self.push_task(Task::Expr(id));
            }
        }
    }

    /// Pop the result-stack base a variable-arity [`Task::Combine`] recorded (via
    /// [`push_combine_var`](Self::push_combine_var)); combines nest LIFO so the top
    /// of `combine_bases` is always this combine's.
    fn pop_combine_base(&mut self) -> usize {
        self.state.scratch.combine_bases.pop().unwrap() as usize
    }

    /// Fold a compound node's already-evaluated children - left on `val_scratch`
    /// / `rule_scratch` by earlier work items - into its single result. Variable-
    /// arity arms recover their children's start via [`pop_combine_base`]; fixed-
    /// arity arms pop a known count. Each arm mirrors one recursive arm of the
    /// former `eval_expr_inner` / `eval_combinator`.
    fn combine(&mut self, id: NodeId) -> LowerResult<()> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::Neg(_) => {
                let v = self.pop_val();
                // Guarded by super::typecheck::type_of (Node::Neg) - inner is int
                let n = self.int_val(v);
                let neg = n.checked_neg().ok_or_else(|| {
                    self.err(LowerErrorKind::IntegerOverflow(-i64::from(n)), span)
                })?;
                self.push_val(Value::Int(neg));
            }
            &Node::BinOp { op, .. } => {
                // rhs was pushed last, so it pops first.
                let rv = self.pop_val();
                let lv = self.pop_val();
                // Guarded by typecheck (Node::BinOp), both sides are int.
                let l = self.int_val(lv);
                let r = self.int_val(rv);
                let (narrow, wide) = match op {
                    BinOp::Add => (l.checked_add(r), i64::from(l) + i64::from(r)),
                    BinOp::Sub => (l.checked_sub(r), i64::from(l) - i64::from(r)),
                };
                let v =
                    narrow.ok_or_else(|| self.err(LowerErrorKind::IntegerOverflow(wide), span))?;
                self.push_val(Value::Int(v));
            }
            Node::Append { .. } => {
                let rv = self.pop_val();
                let lv = self.pop_val();
                // Guarded by super::typecheck::type_of (Node::Append) - both are lists
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
                self.push_val(Value::List(ChildRange::new(start, len)));
            }
            &Node::GrammarConfig { field, .. } => {
                let mod_val = self.pop_val();
                expect_pat!(Value::Module(mod_idx), *self.get_val(mod_val));
                let v = self.eval_grammar_config(mod_idx, field, span)?;
                self.push_val_id(v);
            }
            &Node::FieldAccess { field, .. } => {
                let obj_val = self.pop_val();
                let field_name = self.module_ctx(self.current_module).text(field);
                // typecheck guarantees an object here, and validates the field
                // name for objects whose keys are statically known. For a
                // computed object (e.g. a macro result) the keys aren't known
                // until now, so a missing field is a lower-time error.
                expect_pat!(Value::Object(idx), *self.get_val(obj_val));
                let map = &self.state.ir.object_pool[idx as usize];
                let v = map.get(field_name).copied().ok_or_else(|| {
                    let mut available: Vec<String> = map.keys().cloned().collect();
                    available.sort_unstable();
                    self.err(
                        LowerErrorKind::FieldNotFound {
                            field: field_name.to_string(),
                            available,
                        },
                        field,
                    )
                })?;
                self.push_val_id(v);
            }
            &Node::Object(range) => {
                let base = self.pop_combine_base();
                let fields = self.shared.pools.get_object(range);
                let mut map =
                    FxHashMap::with_capacity_and_hasher(fields.len(), rustc_hash::FxBuildHasher);
                for (i, field) in fields.iter().enumerate() {
                    let key = self
                        .module_ctx(self.current_module)
                        .text(field.name)
                        .to_string();
                    map.insert(key, self.state.scratch.val_scratch[base + i]);
                }
                self.state.scratch.val_scratch.truncate(base);
                let obj = self.alloc_object(map);
                self.push_val_id(obj);
            }
            node @ (&Node::List(_) | &Node::Tuple(_)) => {
                let base = self.pop_combine_base();
                let is_list = matches!(node, Node::List(_));
                let start = self.state.ir.value_children.len() as u32;
                let len = self.checked_len(self.state.scratch.val_scratch.len() - base, span)?;
                self.state
                    .ir
                    .value_children
                    .extend_from_slice(&self.state.scratch.val_scratch[base..]);
                self.state.scratch.val_scratch.truncate(base);
                let range = ChildRange::new(start, len);
                self.push_val(if is_list {
                    Value::List(range)
                } else {
                    Value::Tuple(range)
                });
            }
            Node::Concat(_) => {
                let base = self.pop_combine_base();
                let mut result = String::new();
                for &vid in &self.state.scratch.val_scratch[base..] {
                    result.push_str(self.resolve_str(self.str_id(vid)));
                }
                self.state.scratch.val_scratch.truncate(base);
                let sid = self.strings.intern_owned(&result);
                self.push_val(Value::Str(sid));
            }
            Node::DynRegex { .. } => {
                let base = self.pop_combine_base();
                let pattern_vid = self.state.scratch.val_scratch[base];
                let flags_vid = self.state.scratch.val_scratch.get(base + 1).copied();
                self.state.scratch.val_scratch.truncate(base);
                let ps = self.str_id(pattern_vid);
                let fs = flags_vid.map(|fv| self.str_id(fv));
                let rid = self.alloc_rule(ARule::Pattern(ps, fs));
                self.push_val(Value::Rule(rid));
            }
            // `@<expr>` symbol reference: its name string is on val_scratch.
            &Node::SymRef { .. } => {
                let vid = self.pop_val();
                let name = self.str_id(vid);
                self.push_rule(ARule::NamedSymbol(name));
            }
            &Node::SeqOrChoice { seq, .. } => {
                let base = self.pop_combine_base();
                let start = self.children_start();
                self.state
                    .ir
                    .rule_children
                    .extend_from_slice(&self.state.scratch.rule_scratch[base..]);
                self.state.scratch.rule_scratch.truncate(base);
                let child_range = self.children_range(start, span)?;
                self.push_rule(ARule::SeqOrChoice(seq, child_range));
            }
            &Node::Repeat { kind, .. } => {
                let inner = self.pop_rule();
                if kind == RepeatKind::OneOrMore {
                    self.push_rule(ARule::Repeat(inner));
                } else {
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
                    self.push_rule(ARule::SeqOrChoice(false, child_range));
                }
            }
            &Node::Field { name, .. } => {
                let inner = self.pop_rule();
                let sid = self.intern_span(name);
                self.push_rule(ARule::Field(sid, inner));
            }
            &Node::Token { immediate, .. } => {
                let inner = self.pop_rule();
                self.push_rule(ARule::Token(immediate, inner));
            }
            &Node::Reserved { context, .. } => {
                let inner = self.pop_rule();
                let sid = self.intern_span(context);
                self.push_rule(ARule::Reserved(sid, inner));
            }
            &Node::Prec { kind, .. } => {
                // value is on top of val_scratch, content on top of rule_scratch.
                let vid = self.pop_val();
                let inner = self.pop_rule();
                if kind == PrecKind::Dynamic {
                    let n = self.int_val(vid);
                    self.push_rule(ARule::PrecDynamic(n, inner));
                } else {
                    // Guarded by super::typecheck::type_of (Node::Prec)
                    let prec = match *self.get_val(vid) {
                        Value::Int(n) => APrec::Integer(n),
                        Value::Str(s) => APrec::Name(s),
                        _ => unreachable!(),
                    };
                    let ctor = match kind {
                        PrecKind::Left => ARule::PrecLeft,
                        PrecKind::Right => ARule::PrecRight,
                        _ => ARule::Prec,
                    };
                    self.push_rule(ctor(prec, inner));
                }
            }
            &Node::Alias { target, .. } => {
                if matches!(self.shared.arena.get(target), Node::Ident(IdentKind::Rule)) {
                    let inner = self.pop_rule();
                    let sid = self.intern_span(self.shared.arena.span(target));
                    self.push_rule(ARule::Alias(sid, true, inner));
                } else {
                    let vid = self.pop_val();
                    let inner = self.pop_rule();
                    // Guarded by super::typecheck::type_of (Node::Alias) - only
                    // str or rule-name targets pass
                    match *self.get_val(vid) {
                        Value::Str(s) => self.push_rule(ARule::Alias(s, false, inner)),
                        Value::Rule(rid) => {
                            let s = match self.get_rule(rid) {
                                ARule::NamedSymbol(s) => *s,
                                _ => {
                                    return Err(self.err(
                                        LowerErrorKind::ExpectedRuleName,
                                        self.shared.arena.span(target),
                                    ));
                                }
                            };
                            self.push_rule(ARule::Alias(s, true, inner));
                        }
                        _ => unreachable!(),
                    }
                }
            }
            // combine is only enqueued for the compound nodes handled above.
            _ => unreachable!(),
        }
        Ok(())
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
    /// from the current `call_stack` for the `CallDepthExceeded` error.
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
        let n_bindings = usize::from(for_config.bindings.len);
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
