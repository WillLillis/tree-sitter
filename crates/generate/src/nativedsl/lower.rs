//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].

use std::num::NonZeroU32;
use std::path::PathBuf;

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use crate::{
    grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType},
    rules::{Associativity, Precedence, Rule},
};

use super::ast::{
    ChildRange, ForConfig, IdentKind, ModuleContext, Node, NodeId, PrecKind, QueryableField,
    RepeatKind, SharedAst, Span,
};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Str(NonZeroU32);

enum StrEntry<'src> {
    Unreachable,
    Source(Span, &'src str),
    Owned(String),
}

struct StringPool<'src> {
    entries: Vec<StrEntry<'src>>,
}

impl<'src> StringPool<'src> {
    fn new() -> Self {
        Self {
            entries: vec![StrEntry::Unreachable],
        }
    }

    fn intern_span(&mut self, span: Span, source: &'src str) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Source(span, source));
        id
    }

    fn intern_owned(&mut self, s: String) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Owned(s));
        id
    }

    fn resolve(&self, id: Str) -> &str {
        // Safety: id was produced by intern_span/intern_owned which return
        // sequential indices into self.entries.
        match unsafe { self.entries.get_unchecked(id.0.get() as usize) } {
            StrEntry::Source(span, source) => span.resolve(source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }

    fn to_string(&self, id: Str) -> String {
        self.resolve(id).to_string()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct RuleId(u32);

#[derive(Clone, Copy)]
enum APrec {
    Integer(i32),
    Name(Str),
}

enum ARule {
    Blank,
    String(Str),
    Pattern(Str, Option<Str>),
    NamedSymbol(Str),
    SeqOrChoice(bool, u32, u16),
    Repeat(RuleId),
    Prec(APrec, RuleId),
    PrecLeft(APrec, RuleId),
    PrecRight(APrec, RuleId),
    PrecDynamic(i32, RuleId),
    Field(Str, RuleId),
    Alias(Str, bool, RuleId),
    Token(bool, RuleId),
    Reserved(Str, RuleId),
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct ValueId(u32);

/// Typed wrappers around [`ValueId`] that guarantee the underlying value has
/// been validated. Constructed via `Evaluator::expect_*` methods; zero-cost
/// at runtime (same size as `ValueId`).
macro_rules! typed_val { ($($name:ident),+) => { $(#[derive(Clone, Copy)] struct $name(ValueId);)+ } }
typed_val!(ListVal, RuleVal, StrVal, IntVal, ObjectVal);

/// Compact value representation. Compound data (List/Tuple/Object) stores
/// indices into external pools on the Evaluator rather than inline heap data.
#[derive(Clone, Copy)]
enum Value {
    Int(i32),
    Str(Str),
    Rule(RuleId),
    /// Index into `Evaluator::object_pool`.
    Object(u32),
    /// Range into `Evaluator::value_children`.
    List(ChildRange),
    /// Range into `Evaluator::value_children`.
    Tuple(ChildRange),
    Module(u8),
}

const MAX_CALL_DEPTH: u16 = 64;

/// Flat entry for one module in the evaluator's module table.
struct ModuleInfo<'ast> {
    ctx: &'ast ModuleContext,
    base_grammar: Option<&'ast InputGrammar>,
}

struct Evaluator<'ast> {
    // Shared AST data (arena + pools), immutable during evaluation
    shared: &'ast SharedAst,
    // Flat module table (immutable after construction)
    modules: Vec<ModuleInfo<'ast>>,
    module_macros: Vec<FxHashMap<&'ast str, NodeId>>,
    // Per-module let binding values (lazily populated on first access)
    module_values: Vec<Option<FxHashMap<&'ast str, ValueId>>>,
    // Offset: table index = global_id - base_id
    base_id: usize,
    // Current execution state
    current_module: usize,
    scopes: FxHashMap<&'ast str, ValueId>,
    // Shared across module boundaries
    call_stack: Vec<(&'ast str, Span, usize)>, // name, span, module index
    arg_scratch: Vec<ValueId>,
    val_scratch: Vec<ValueId>,
    rule_scratch: Vec<RuleId>,
    values: Vec<Value>,
    rules: Vec<ARule>,
    rule_children: Vec<RuleId>,
    value_children: Vec<ValueId>,
    object_pool: Vec<FxHashMap<&'ast str, ValueId>>,
    strings: StringPool<'ast>,
}

/// Collect all `fn` definitions from a module into a name -> `NodeId` map.
fn collect_macros<'a>(shared: &'a SharedAst, ctx: &'a ModuleContext) -> FxHashMap<&'a str, NodeId> {
    let mut fns = FxHashMap::default();
    for &item_id in &ctx.root_items {
        if let Node::Macro(fn_idx) = shared.arena.get(item_id) {
            let config = shared.pools.get_macro(*fn_idx);
            fns.insert(ctx.text(config.name), item_id);
        }
    }
    fns
}

/// Intermediate results from the evaluation phase, ready to be assembled
/// into an `InputGrammar` once the base grammar can be moved out.
struct EvalResult {
    language: String,
    rules: Vec<(String, Rule)>,
    overrides: Vec<(String, Rule, Span)>,
    extras: Option<Vec<Rule>>,
    externals: Option<Vec<Rule>>,
    inline: Option<Vec<String>>,
    supertypes: Option<Vec<String>>,
    word: Option<String>,
    conflicts: Option<Vec<Vec<String>>>,
    precedences: Option<Vec<Vec<PrecedenceEntry>>>,
    reserved: Option<Vec<ReservedWordContext<Rule>>>,
}

/// Build a flat module table from a module and its descendants. The root
/// module is placed at index 0; children follow in depth-first pre-order.
fn build_module_table<'ast>(
    shared: &'ast SharedAst,
    root_ctx: &'ast ModuleContext,
    root_base_grammar: Option<&'ast InputGrammar>,
    sub_modules: &'ast [super::Module],
) -> (Vec<ModuleInfo<'ast>>, Vec<FxHashMap<&'ast str, NodeId>>) {
    let mut infos = vec![ModuleInfo {
        ctx: root_ctx,
        base_grammar: root_base_grammar,
    }];
    let mut fns = vec![collect_macros(shared, root_ctx)];

    fn add_children<'a>(
        shared: &'a SharedAst,
        modules: &'a [super::Module],
        infos: &mut Vec<ModuleInfo<'a>>,
        fns: &mut Vec<FxHashMap<&'a str, NodeId>>,
    ) {
        for m in modules {
            infos.push(ModuleInfo {
                ctx: &m.ctx,
                base_grammar: m.lowered.as_ref(),
            });
            fns.push(collect_macros(shared, &m.ctx));
            add_children(shared, &m.sub_modules, infos, fns);
        }
    }

    add_children(shared, sub_modules, &mut infos, &mut fns);
    (infos, fns)
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
pub fn lower_with_base(
    shared: &SharedAst,
    ctx: &ModuleContext,
    modules: &[super::Module],
    root_global_id: u8,
) -> LowerResult<InputGrammar> {
    let base_grammar = modules.iter().find_map(|m| m.lowered.as_ref());
    let result = evaluate(shared, ctx, modules, base_grammar, root_global_id)?;
    build_grammar(result, base_grammar)
}

/// Evaluate the AST, producing intermediate results.
fn evaluate(
    shared: &SharedAst,
    ctx: &ModuleContext,
    modules: &[super::Module],
    base_grammar: Option<&InputGrammar>,
    root_global_id: u8,
) -> LowerResult<EvalResult> {
    let (module_infos, module_macros) =
        build_module_table(shared, ctx, base_grammar, modules);
    let n = module_infos.len();
    let mut eval = Evaluator {
        shared,
        modules: module_infos,
        module_macros,
        module_values: vec![None; n],
        base_id: root_global_id as usize,
        current_module: 0,
        scopes: FxHashMap::default(),
        call_stack: Vec::new(),
        arg_scratch: Vec::new(),
        val_scratch: Vec::new(),
        rule_scratch: Vec::new(),
        values: Vec::with_capacity(shared.arena.len()),
        rules: Vec::with_capacity(shared.arena.len()),
        rule_children: Vec::with_capacity(shared.arena.len()),
        value_children: Vec::with_capacity(shared.arena.len()),
        object_pool: Vec::new(),
        strings: StringPool::new(),
    };
    let mut rule_entries: Vec<(&str, RuleId)> = Vec::with_capacity(ctx.root_items.len());
    let mut override_entries: Vec<(&str, RuleId, Span)> = Vec::new();

    for &item_id in &ctx.root_items {
        match shared.arena.get(item_id) {
            Node::Grammar | Node::Macro(_) => {}
            Node::Let { name, value, .. } => {
                let val = eval.eval_expr(*value)?;
                eval.scopes.insert(ctx.text(*name), val);
            }
            Node::Rule {
                is_override,
                name,
                body,
            } => {
                let rule_id = eval.lower_to_rule(*body)?;
                if *is_override {
                    override_entries.push((ctx.text(*name), rule_id, *name));
                } else {
                    rule_entries.push((ctx.text(*name), rule_id));
                }
            }
            _ => unreachable!(),
        }
    }

    // grammar_config is guaranteed present by validate_grammar in mod.rs,
    // language is guaranteed present by the parser (MissingLanguageField error).
    let config = ctx.grammar_config.as_ref().unwrap();
    let language = config.language.as_ref().unwrap();

    let rules: Vec<(String, Rule)> = rule_entries
        .iter()
        .map(|(name, rid)| (name.to_string(), eval.build_rule(*rid)))
        .collect();
    let overrides: Vec<(String, Rule, Span)> = override_entries
        .iter()
        .map(|(name, rid, span)| (name.to_string(), eval.build_rule(*rid), *span))
        .collect();

    Ok(EvalResult {
        language: language.clone(),
        rules,
        overrides,
        extras: config
            .extras
            .map(|id| eval.eval_rule_list(id))
            .transpose()?,
        externals: config
            .externals
            .map(|id| eval.eval_rule_list(id))
            .transpose()?,
        inline: config
            .inline
            .map(|id| eval.eval_name_list(id))
            .transpose()?,
        supertypes: config
            .supertypes
            .map(|id| eval.eval_name_list(id))
            .transpose()?,
        word: config.word.map(|id| eval.eval_rule_name(id)).transpose()?,
        conflicts: config
            .conflicts
            .map(|id| eval.eval_conflicts(id))
            .transpose()?,
        precedences: config
            .precedences
            .map(|id| eval.eval_precedences(id))
            .transpose()?,
        reserved: config
            .reserved
            .map(|id| eval.eval_reserved(id))
            .transpose()?,
    })
}

/// Assemble evaluated results into an `InputGrammar`, optionally merging
/// with an inherited base grammar.
fn build_grammar(result: EvalResult, base: Option<&InputGrammar>) -> LowerResult<InputGrammar> {
    // Start with inherited rules (cloned), apply overrides, then append new rules.
    let mut variables = if let Some(base) = base {
        // Build override map so we can apply them in a single pass over the
        // base variables, avoiding a full clone of every rule tree.
        let mut overrides: FxHashMap<String, (Rule, Span)> = FxHashMap::default();
        for (name, rule, span) in result.overrides {
            overrides.insert(name, (rule, span));
        }
        let mut vars = Vec::with_capacity(base.variables.len());
        for v in &base.variables {
            if let Some((rule, _)) = overrides.remove(v.name.as_str()) {
                vars.push(Variable {
                    name: v.name.clone(),
                    kind: v.kind,
                    rule,
                });
            } else {
                vars.push(v.clone());
            }
        }
        // Any remaining overrides reference rules that don't exist in the base.
        if !overrides.is_empty() {
            let entries: Vec<_> = overrides.into_iter().collect();
            let span = entries[0].1.1;
            let names = entries.into_iter().map(|(n, _)| n).collect();
            return Err(LowerError::new(
                LowerErrorKind::OverrideRuleNotFound(names),
                span,
            ));
        }
        vars
    } else if !result.overrides.is_empty() {
        return Err(LowerError::new(
            LowerErrorKind::OverrideWithoutInherit,
            result.overrides[0].2,
        ));
    } else {
        Vec::new()
    };

    for (name, rule) in result.rules {
        variables.push(Variable {
            name,
            kind: VariableType::Named,
            rule,
        });
    }

    // Derived values override base; only clone base fields that aren't overridden.
    fn inherit<T: Clone>(
        overridden: Option<Vec<T>>,
        base: Option<&InputGrammar>,
        field: fn(&InputGrammar) -> &Vec<T>,
    ) -> Vec<T> {
        overridden.unwrap_or_else(|| base.map_or_else(Vec::new, |b| field(b).clone()))
    }
    Ok(InputGrammar {
        name: result.language,
        variables,
        extra_symbols: inherit(result.extras, base, |b| &b.extra_symbols),
        expected_conflicts: inherit(result.conflicts, base, |b| &b.expected_conflicts),
        precedence_orderings: inherit(result.precedences, base, |b| &b.precedence_orderings),
        external_tokens: inherit(result.externals, base, |b| &b.external_tokens),
        variables_to_inline: inherit(result.inline, base, |b| &b.variables_to_inline),
        supertype_symbols: inherit(result.supertypes, base, |b| &b.supertype_symbols),
        word_token: result
            .word
            .or_else(|| base.and_then(|b| b.word_token.clone())),
        reserved_words: inherit(result.reserved, base, |b| &b.reserved_words),
    })
}

impl<'ast> Evaluator<'ast> {
    fn ctx(&self) -> &'ast ModuleContext {
        self.modules[self.current_module].ctx
    }

    fn module_idx(&self, global_id: u8) -> usize {
        debug_assert!(
            global_id as usize >= self.base_id,
            "global_id {global_id} is below base_id {}",
            self.base_id
        );
        global_id as usize - self.base_id
    }

    fn alloc_val(&mut self, val: Value) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(val);
        id
    }

    fn get_val(&self, id: ValueId) -> &Value {
        // Safety: id was produced by alloc_val which returns sequential indices
        // into self.values. No ValueId can outlive the Evaluator.
        unsafe { self.values.get_unchecked(id.0 as usize) }
    }

    fn alloc_list(&mut self, items: &[ValueId], span: Span) -> LowerResult<ValueId> {
        let start = self.value_children.len() as u32;
        let len = u16::try_from(items.len())
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
        self.value_children.extend_from_slice(items);
        Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
    }

    fn alloc_object(&mut self, map: FxHashMap<&'ast str, ValueId>) -> ValueId {
        let idx = self.object_pool.len() as u32;
        self.object_pool.push(map);
        self.alloc_val(Value::Object(idx))
    }

    const fn expect_list(vid: ValueId) -> ListVal {
        ListVal(vid)
    }
    fn list_items(&self, v: ListVal) -> &[ValueId] {
        &self.value_children[self.list_range(v).as_range()]
    }

    fn list_range(&self, v: ListVal) -> ChildRange {
        let Value::List(range) = *self.get_val(v.0) else {
            unreachable!()
        };
        range
    }

    const fn expect_rule(vid: ValueId) -> RuleVal {
        RuleVal(vid)
    }
    fn rule_id(&self, v: RuleVal) -> RuleId {
        let Value::Rule(rid) = *self.get_val(v.0) else {
            unreachable!()
        };
        rid
    }

    const fn expect_str(vid: ValueId) -> StrVal {
        StrVal(vid)
    }
    fn str_id(&self, v: StrVal) -> Str {
        let Value::Str(s) = *self.get_val(v.0) else {
            unreachable!()
        };
        s
    }

    const fn expect_int(vid: ValueId) -> IntVal {
        IntVal(vid)
    }
    fn int_val(&self, v: IntVal) -> i32 {
        let Value::Int(n) = *self.get_val(v.0) else {
            unreachable!()
        };
        n
    }

    const fn expect_object(vid: ValueId) -> ObjectVal {
        ObjectVal(vid)
    }
    fn object_fields(&self, v: ObjectVal) -> &FxHashMap<&'ast str, ValueId> {
        let Value::Object(idx) = *self.get_val(v.0) else {
            unreachable!()
        };
        &self.object_pool[idx as usize]
    }

    fn alloc_rule(&mut self, rule: ARule) -> RuleId {
        let id = RuleId(self.rules.len() as u32);
        self.rules.push(rule);
        id
    }

    /// Get a rule by arena ID. All `RuleId`s are produced by `alloc_rule`.
    fn get_rule(&self, id: RuleId) -> &ARule {
        // Safety: id was produced by alloc_rule which returns sequential indices.
        unsafe { self.rules.get_unchecked(id.0 as usize) }
    }

    const fn children_start(&self) -> u32 {
        self.rule_children.len() as u32
    }

    fn children_range(&self, start: u32, span: Span) -> LowerResult<(u32, u16)> {
        let count = self.rule_children.len() as u32 - start;
        u16::try_from(count)
            .map(|len| (start, len))
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))
    }

    /// Intern a span using the current module's source text.
    fn intern_span(&mut self, span: Span) -> Str {
        self.strings.intern_span(span, &self.ctx().source)
    }

    fn intern_string_lit(&mut self, span: Span) -> Str {
        let raw = self.ctx().text(span);
        if memchr::memchr(b'\\', raw.as_bytes()).is_some() {
            self.strings.intern_owned(unescape_string(raw))
        } else {
            self.intern_span(span)
        }
    }

    fn intern_raw_string_lit(&mut self, span: Span, hash_count: u8) -> Str {
        let prefix = 2 + u32::from(hash_count);
        let suffix = 1 + u32::from(hash_count);
        self.intern_span(Span::new(span.start + prefix, span.end - suffix))
    }

    fn get_str_val(&self, id: ValueId) -> Str {
        self.str_id(Self::expect_str(id))
    }

    /// Intern a string, create a `NamedSymbol` rule, and wrap as a `Value::Rule`.
    fn owned_symbol_val(&mut self, name: String) -> ValueId {
        let sid = self.strings.intern_owned(name);
        let rid = self.alloc_rule(ARule::NamedSymbol(sid));
        self.alloc_val(Value::Rule(rid))
    }

    fn build_rule(&self, id: RuleId) -> Rule {
        let s = &self.strings;
        match self.get_rule(id) {
            ARule::Blank => Rule::Blank,
            ARule::String(sid) => Rule::String(s.to_string(*sid)),
            ARule::Pattern(p, f) => Rule::Pattern(
                s.to_string(*p),
                f.map_or_else(String::new, |f| s.to_string(f)),
            ),
            ARule::NamedSymbol(sid) => Rule::NamedSymbol(s.to_string(*sid)),
            ARule::SeqOrChoice(is_seq, start, len) => {
                let range = *start as usize..*start as usize + *len as usize;
                // Safety: start and len are produced by children_start/children_range
                // which track rule_children indices.
                let children: Vec<Rule> = unsafe { self.rule_children.get_unchecked(range) }
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
            ARule::Field(n, r) => Rule::field(s.to_string(*n), self.build_rule(*r)),
            ARule::Alias(v, named, r) => Rule::alias(self.build_rule(*r), s.to_string(*v), *named),
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

    fn value_to_owned_rule(&self, id: ValueId) -> Rule {
        match self.get_val(id) {
            Value::Rule(rid) => self.build_rule(*rid),
            Value::Str(sid) => Rule::String(self.strings.to_string(*sid)),
            // Guarded by super::typecheck - only rule-like values reach here
            _ => unreachable!(),
        }
    }

    fn eval_rule_list(&mut self, id: NodeId) -> LowerResult<Vec<Rule>> {
        let vid = self.eval_expr(id)?;
        let items = self.list_items(Self::expect_list(vid));
        Ok(items.iter().map(|&v| self.value_to_owned_rule(v)).collect())
    }

    fn rule_name_string(&self, v: ValueId, span: Span) -> LowerResult<String> {
        let rid = self.rule_id(Self::expect_rule(v));
        let ARule::NamedSymbol(sid) = self.get_rule(rid) else {
            return Err(LowerError::new(LowerErrorKind::ExpectedRuleName, span));
        };
        Ok(self.strings.to_string(*sid))
    }

    fn eval_name_list(&mut self, id: NodeId) -> LowerResult<Vec<String>> {
        let vid = self.eval_expr(id)?;
        let span = self.shared.arena.span(id);
        let items = self.list_items(Self::expect_list(vid));
        items
            .iter()
            .map(|&v| self.rule_name_string(v, span))
            .collect()
    }

    /// Evaluate a `conflicts` expression into `Vec<Vec<String>>`. Each inner
    /// element must evaluate to a named rule reference.
    fn eval_conflicts(&mut self, id: NodeId) -> LowerResult<Vec<Vec<String>>> {
        let vid = self.eval_expr(id)?;
        let span = self.shared.arena.span(id);
        let outer = self.list_items(Self::expect_list(vid));
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(Self::expect_list(group_vid));
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
    fn eval_precedences(&mut self, id: NodeId) -> LowerResult<Vec<Vec<PrecedenceEntry>>> {
        let vid = self.eval_expr(id)?;
        let span = self.shared.arena.span(id);
        let outer = self.list_items(Self::expect_list(vid));
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(Self::expect_list(group_vid));
                inner
                    .iter()
                    .map(|&v| match *self.get_val(v) {
                        Value::Str(sid) => Ok(PrecedenceEntry::Name(self.strings.to_string(sid))),
                        Value::Rule(_) => {
                            Ok(PrecedenceEntry::Symbol(self.rule_name_string(v, span)?))
                        }
                        _ => unreachable!(),
                    })
                    .collect()
            })
            .collect()
    }

    fn eval_rule_name(&mut self, id: NodeId) -> LowerResult<String> {
        let rid = self.lower_to_rule(id)?;
        match self.get_rule(rid) {
            ARule::NamedSymbol(sid) => Ok(self.strings.to_string(*sid)),
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
    fn eval_reserved(&mut self, id: NodeId) -> LowerResult<Vec<ReservedWordContext<Rule>>> {
        let ctx = self.ctx();
        // Object literal: each field is a named word set
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
        // Expression: evaluate to Object(ListRule), keyed by context name
        let vid = self.eval_expr(id)?;
        let fields = self.object_fields(Self::expect_object(vid));
        fields
            .iter()
            .map(|(&name, &words_vid)| {
                let word_vals = self.list_items(Self::expect_list(words_vid));
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

    /// Evaluate a `grammar_config(module, field)` expression.
    fn eval_grammar_config(
        &mut self,
        mod_idx: usize,
        field: QueryableField,
        span: Span,
    ) -> LowerResult<ValueId> {
        use QueryableField as Q;
        let grammar = self.modules[mod_idx].base_grammar.unwrap();
        match field {
            Q::Extras => self.import_rules_as_list(&grammar.extra_symbols, span),
            Q::Externals => self.import_rules_as_list(&grammar.external_tokens, span),
            Q::Inline => self.import_names_as_list(&grammar.variables_to_inline, span),
            Q::Supertypes => self.import_names_as_list(&grammar.supertype_symbols, span),
            Q::Conflicts => {
                let vals: Vec<ValueId> = grammar
                    .expected_conflicts
                    .iter()
                    .map(|g| self.import_names_as_list(g, span))
                    .collect::<LowerResult<_>>()?;
                self.alloc_list(&vals, span)
            }
            Q::Precedences => {
                let vals: Vec<ValueId> = grammar
                    .precedence_orderings
                    .iter()
                    .map(|group| {
                        let inner: Vec<ValueId> = group
                            .iter()
                            .map(|entry| match entry {
                                PrecedenceEntry::Name(s) => {
                                    let sid = self.strings.intern_owned(s.clone());
                                    self.alloc_val(Value::Str(sid))
                                }
                                PrecedenceEntry::Symbol(s) => self.owned_symbol_val(s.clone()),
                            })
                            .collect();
                        self.alloc_list(&inner, span)
                    })
                    .collect::<LowerResult<_>>()?;
                self.alloc_list(&vals, span)
            }
            Q::Word => {
                if let Some(name) = &grammar.word_token {
                    Ok(self.owned_symbol_val(name.clone()))
                } else {
                    Err(LowerError::new(LowerErrorKind::ConfigFieldUnset, span))
                }
            }
            Q::Reserved => {
                let contexts: Vec<_> = grammar
                    .reserved_words
                    .iter()
                    .map(|ctx| (ctx.name.as_str(), ctx.reserved_words.as_slice()))
                    .collect();
                let mut map =
                    FxHashMap::with_capacity_and_hasher(contexts.len(), rustc_hash::FxBuildHasher);
                for (name, words) in contexts {
                    let words_vid = self.import_rules_as_list(words, span)?;
                    map.insert(name, words_vid);
                }
                Ok(self.alloc_object(map))
            }
        }
    }

    fn import_rules_as_list(&mut self, rules_data: &[Rule], span: Span) -> LowerResult<ValueId> {
        let vals: Vec<ValueId> = rules_data
            .iter()
            .map(|r| {
                let rid = self.import_rule(r);
                self.alloc_val(Value::Rule(rid))
            })
            .collect();
        self.alloc_list(&vals, span)
    }

    fn import_names_as_list(&mut self, names: &[String], span: Span) -> LowerResult<ValueId> {
        let vals: Vec<ValueId> = names
            .iter()
            .map(|name| self.owned_symbol_val(name.clone()))
            .collect();
        self.alloc_list(&vals, span)
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
                let is_seq = matches!(rule, Rule::Seq(_));
                let imported: Vec<RuleId> = members.iter().map(|m| self.import_rule(m)).collect();
                let start = self.children_start();
                self.rule_children.extend_from_slice(&imported);
                let count = self.rule_children.len() as u32 - start;
                debug_assert!(
                    u16::try_from(count).is_ok(),
                    "inherited rule has too many children"
                );
                self.alloc_rule(ARule::SeqOrChoice(is_seq, start, count as u16))
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
        assoc: Option<Associativity>,
    ) -> RuleId {
        match assoc {
            Some(Associativity::Left) => self.alloc_rule(ARule::PrecLeft(prec, inner)),
            Some(Associativity::Right) => self.alloc_rule(ARule::PrecRight(prec, inner)),
            None => self.alloc_rule(ARule::Prec(prec, inner)),
        }
    }
}

impl<'ast> Evaluator<'ast> {
    fn eval_expr(&mut self, id: NodeId) -> LowerResult<ValueId> {
        let span = self.shared.arena.span(id);
        match self.shared.arena.get(id) {
            Node::IntLit(n) => Ok(self.alloc_val(Value::Int(*n))),
            Node::StringLit => {
                let sid = self.intern_string_lit(span);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(span, *hash_count);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::Neg(inner) => {
                let vid = self.eval_expr(*inner)?;
                // Guarded by super::typecheck::type_of (Node::Neg) - ensures inner is int
                let n = self.int_val(Self::expect_int(vid));
                Ok(self.alloc_val(Value::Int(-n)))
            }
            // Guarded by super::resolve + typecheck
            Node::Ident(IdentKind::Unresolved | IdentKind::Macro) => unreachable!(),
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            // Guarded by super::resolve - all variable names validated
            Node::Ident(IdentKind::Var) => Ok(*self.scopes.get(self.ctx().text(span)).unwrap()),
            Node::GrammarConfig { module, field } => {
                let (module, field) = (*module, *field);
                let mod_val = self.eval_expr(module)?;
                let Value::Module(idx) = *self.get_val(mod_val) else {
                    unreachable!() // guarded by typecheck
                };
                self.eval_grammar_config(idx as usize, field, span)
            }
            Node::ModuleRef { module, .. } => {
                let global_id = module.expect("module index not set by loading pre-pass");
                let table_id = self.module_idx(global_id);
                self.eval_import_module(table_id)?;
                Ok(self.alloc_val(Value::Module(table_id as u8)))
            }
            Node::Append { left, right } => {
                let (left, right) = (*left, *right);
                let lv = self.eval_expr(left)?;
                let rv = self.eval_expr(right)?;
                // Guarded by super::typecheck::type_of (Node::Append) - ensures both are lists
                let lr = self.list_range(Self::expect_list(lv));
                let rr = self.list_range(Self::expect_list(rv));
                let start = self.value_children.len() as u32;
                let total = lr.len as usize + rr.len as usize;
                let len = u16::try_from(total)
                    .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
                self.value_children.extend_from_within(lr.as_range());
                self.value_children.extend_from_within(rr.as_range());
                Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
            }
            Node::FieldAccess { obj, field } => {
                let (obj, field) = (*obj, *field);
                let obj_val = self.eval_expr(obj)?;
                let field_name = self.ctx().text(field);
                match *self.get_val(obj_val) {
                    Value::Object(idx) => {
                        Ok(*self.object_pool[idx as usize].get(field_name).unwrap())
                    }
                    _ => unreachable!(), // guarded by typecheck
                }
            }
            Node::QualifiedAccess { obj, member } => {
                let (obj, member) = (*obj, *member);
                let obj_val = self.eval_expr(obj)?;
                let member_name = self.ctx().text(member);
                let Value::Module(idx) = *self.get_val(obj_val) else {
                    unreachable!() // guarded by typecheck
                };
                let table_id = idx as usize;
                // Check pre-evaluated let values first
                let values = self.module_values[table_id].as_ref().unwrap();
                if let Some(&val) = values.get(member_name) {
                    return Ok(val);
                }
                // For inherited grammar modules, fall back to rule bodies.
                if let Some(var) = self.modules[table_id]
                    .base_grammar
                    .and_then(|g| g.variables.iter().find(|v| v.name == member_name))
                {
                    let rid = self.import_rule(&var.rule);
                    return Ok(self.alloc_val(Value::Rule(rid)));
                }
                Err(LowerError::new(
                    LowerErrorKind::ModuleMemberNotFound(member_name.to_string()),
                    member,
                ))
            }
            Node::Object(range) => {
                let range = *range;
                let fields = self.shared.pools.get_object(range);
                let mut map =
                    FxHashMap::with_capacity_and_hasher(fields.len(), rustc_hash::FxBuildHasher);
                for &(key_span, value_id) in fields {
                    map.insert(self.ctx().text(key_span), self.eval_expr(value_id)?);
                }
                Ok(self.alloc_object(map))
            }
            Node::List(range) | Node::Tuple(range) => {
                let range = *range;
                let is_list = matches!(self.shared.arena.get(id), Node::List(_));
                let items = self.shared.pools.child_slice(range);
                let base = self.val_scratch.len();
                for &item_id in items {
                    let v = self.eval_expr(item_id)?;
                    self.val_scratch.push(v);
                }
                let start = self.value_children.len() as u32;
                let len = u16::try_from(self.val_scratch.len() - base)
                    .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
                self.value_children
                    .extend_from_slice(&self.val_scratch[base..]);
                self.val_scratch.truncate(base);
                let range = ChildRange::new(start, len);
                if is_list {
                    Ok(self.alloc_val(Value::List(range)))
                } else {
                    Ok(self.alloc_val(Value::Tuple(range)))
                }
            }
            Node::Concat(range) => {
                let range = *range;
                let mut result = String::new();
                for &part_id in self.shared.pools.child_slice(range) {
                    let vid = self.eval_expr(part_id)?;
                    result.push_str(self.strings.resolve(self.get_str_val(vid)));
                }
                let sid = self.strings.intern_owned(result);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::DynRegex(range) => {
                let (pattern, flags) = self.shared.pools.get_regex(*range);
                let pv = self.eval_expr(pattern)?;
                let ps = self.get_str_val(pv);
                let fs = flags
                    .map(|fid| self.eval_expr(fid))
                    .transpose()?
                    .map(|fv| self.get_str_val(fv));
                let rid = self.alloc_rule(ARule::Pattern(ps, fs));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::Call { name, args } => {
                let (name, args) = (*name, *args);
                let fn_name = self.ctx().text(self.shared.arena.span(name));
                let base = self.arg_scratch.len();
                for &arg_id in self.shared.pools.child_slice(args) {
                    let v = self.eval_expr(arg_id)?;
                    self.arg_scratch.push(v);
                }
                let module = self.current_module;
                let result = self.call_fn(module, fn_name, base, span);
                self.arg_scratch.truncate(base);
                result
            }
            Node::QualifiedCall(range) => {
                let range = *range;
                let (obj, name, args) = self.shared.pools.get_qualified_call(range);
                let fn_name = self.ctx().text(self.shared.arena.span(name));
                let obj_val = self.eval_expr(obj)?;
                let Value::Module(idx) = *self.get_val(obj_val) else {
                    unreachable!() // guarded by typecheck
                };
                let base = self.arg_scratch.len();
                for &arg_id in args {
                    let v = self.eval_expr(arg_id)?;
                    self.arg_scratch.push(v);
                }
                let result = self.call_fn(idx as usize, fn_name, base, span);
                self.arg_scratch.truncate(base);
                result
            }
            _ => {
                let rid = self.eval_combinator(id)?;
                Ok(self.alloc_val(Value::Rule(rid)))
            }
        }
    }

    fn lower_to_rule(&mut self, id: NodeId) -> LowerResult<RuleId> {
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
            Node::SeqOrChoice { seq, range } => {
                let (seq, range) = (*seq, *range);
                let children = self.shared.pools.child_slice(range);
                let base = self.rule_scratch.len();
                for &member in children {
                    if let Node::For(idx) = self.shared.arena.get(member) {
                        let for_rules =
                            self.eval_for_to_rules(self.shared.pools.get_for(*idx))?;
                        self.rule_scratch.extend(for_rules);
                    } else {
                        let rid = self.lower_to_rule(member)?;
                        self.rule_scratch.push(rid);
                    }
                }
                let start = self.children_start();
                self.rule_children
                    .extend_from_slice(&self.rule_scratch[base..]);
                self.rule_scratch.truncate(base);
                let (offset, len) = self.children_range(start, span)?;
                Ok(self.alloc_rule(ARule::SeqOrChoice(seq, offset, len)))
            }
            Node::Repeat { kind, inner } => {
                let (kind, inner) = (*kind, *inner);
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
                self.rule_children.extend_from_slice(&[first, blank]);
                let (s, l) = self.children_range(start, span).unwrap();
                Ok(self.alloc_rule(ARule::SeqOrChoice(false, s, l)))
            }
            Node::Blank => Ok(self.alloc_rule(ARule::Blank)),
            Node::Field { name, content } => {
                let (name, content) = (*name, *content);
                let inner = self.lower_to_rule(content)?;
                let sid = self.intern_span(name);
                Ok(self.alloc_rule(ARule::Field(sid, inner)))
            }
            Node::Alias { content, target } => {
                let (content, target) = (*content, *target);
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
            Node::Token { immediate, inner } => {
                let (immediate, inner) = (*immediate, *inner);
                let inner = self.lower_to_rule(inner)?;
                Ok(self.alloc_rule(ARule::Token(immediate, inner)))
            }
            Node::Prec {
                kind,
                value,
                content,
            } => {
                let (kind, value, content) = (*kind, *value, *content);
                let vid = self.eval_expr(value)?;
                let inner = self.lower_to_rule(content)?;
                if kind == PrecKind::Dynamic {
                    let n = self.int_val(Self::expect_int(vid));
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
            Node::Reserved { context, content } => {
                let (context, content) = (*context, *content);
                let inner = self.lower_to_rule(content)?;
                let sid = self.intern_span(context);
                Ok(self.alloc_rule(ARule::Reserved(sid, inner)))
            }
            // Guarded by lower_to_rule - only combinator nodes are dispatched here
            _ => unreachable!(),
        }
    }

    fn push_call(&mut self, name: &'ast str, span: Span) -> LowerResult<()> {
        self.call_stack.push((name, span, self.current_module));
        if self.call_stack.len() > MAX_CALL_DEPTH as usize {
            let trace = self
                .call_stack
                .iter()
                .map(|(name, span, mod_idx)| {
                    let ctx = self.modules[*mod_idx].ctx;
                    let offset = span.start as usize;
                    let bytes = &ctx.source.as_bytes()[..offset];
                    let line = memchr::memchr_iter(b'\n', bytes).count() + 1;
                    let col = offset - memchr::memrchr(b'\n', bytes).map_or(0, |i| i + 1) + 1;
                    (name.to_string(), ctx.path.clone(), line, col)
                })
                .collect();
            let root_span = self.call_stack[0].1;
            return Err(LowerError::new(
                LowerErrorKind::CallDepthExceeded(trace),
                root_span,
            ));
        }
        Ok(())
    }

    /// Call a function, either in the current module or an imported one.
    /// For cross-module calls, saves/restores module context and binds
    /// the target module's let values into scope.
    fn call_fn(
        &mut self,
        table_id: usize,
        fn_name: &'ast str,
        arg_base: usize,
        call_span: Span,
    ) -> LowerResult<ValueId> {
        self.push_call(fn_name, call_span)?;

        let fn_ctx = self.modules[table_id].ctx;
        let &fn_node_id = self.module_macros[table_id].get(fn_name).unwrap();
        let fn_config = match self.shared.arena.get(fn_node_id) {
            Node::Macro(fi) => self.shared.pools.get_macro(*fi),
            _ => unreachable!(),
        };
        let body = fn_config.body;

        let cross_module = table_id != self.current_module;
        let saved_module = self.current_module;
        let saved_scopes = if cross_module {
            self.current_module = table_id;
            Some(std::mem::replace(&mut self.scopes, FxHashMap::default()))
        } else {
            None
        };

        if cross_module {
            for (&name, &val) in self.module_values[table_id].as_ref().unwrap() {
                self.scopes.insert(name, val);
            }
        }
        for (i, param) in fn_config.params.iter().enumerate() {
            self.scopes
                .insert(fn_ctx.text(param.name), self.arg_scratch[arg_base + i]);
        }

        let result = self.eval_expr(body);
        self.call_stack.pop();

        // Remove param bindings (no shadowing guarantees safe removal)
        for param in &fn_config.params {
            self.scopes.remove(fn_ctx.text(param.name));
        }

        if let Some(saved) = saved_scopes {
            self.current_module = saved_module;
            self.scopes = saved;
        }

        result
    }

    /// Lazily evaluate an imported module's top-level let bindings.
    fn eval_import_module(&mut self, table_id: usize) -> LowerResult<()> {
        if self.module_values[table_id].is_some() {
            return Ok(());
        }

        let saved_module = self.current_module;
        let saved_scopes = std::mem::replace(&mut self.scopes, FxHashMap::default());
        self.current_module = table_id;

        let result = self.eval_let_bindings();

        self.current_module = saved_module;
        self.scopes = saved_scopes;

        self.module_values[table_id] = Some(result?);
        Ok(())
    }

    /// Evaluate top-level let bindings in the current module context.
    fn eval_let_bindings(&mut self) -> LowerResult<FxHashMap<&'ast str, ValueId>> {
        let ctx = self.modules[self.current_module].ctx;
        let n = ctx.root_items.len();
        let mut values = FxHashMap::default();
        for i in 0..n {
            let item_id = ctx.root_items[i];
            if let Node::Let { name, value, .. } = self.shared.arena.get(item_id) {
                let (name, value) = (*name, *value);
                let val = self.eval_expr(value)?;
                let var_name = ctx.text(name);
                self.scopes.insert(var_name, val);
                values.insert(var_name, val);
            }
        }
        Ok(values)
    }

    fn eval_for_to_rules(&mut self, config: &ForConfig) -> LowerResult<Vec<RuleId>> {
        let iter_vid = self.eval_expr(config.iterable)?;
        let n_items = self.list_items(Self::expect_list(iter_vid)).len();
        let mut results = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let item_id = self.list_items(Self::expect_list(iter_vid))[i];
            // Insert bindings (overwrite per iteration, no shadowing)
            for (j, &(name_span, _)) in config.bindings.iter().enumerate() {
                let value_id = match *self.get_val(item_id) {
                    Value::Tuple(range) => self.value_children[range.start as usize + j],
                    _ => item_id,
                };
                self.scopes.insert(self.ctx().text(name_span), value_id);
            }
            results.push(self.lower_to_rule(config.body)?);
        }
        // Remove bindings after loop
        for &(name_span, _) in &config.bindings {
            self.scopes.remove(self.ctx().text(name_span));
        }
        Ok(results)
    }
}

fn unescape_string(raw: &str) -> String {
    let mut result = String::with_capacity(raw.len());
    let mut chars = raw.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('"') => result.push('"'),
                Some('\\') => result.push('\\'),
                Some('n') => result.push('\n'),
                Some('t') => result.push('\t'),
                Some('r') => result.push('\r'),
                Some('0') => result.push('\0'),
                // Guarded by lexer escape validation
                _ => unreachable!(),
            }
        } else {
            result.push(c);
        }
    }
    result
}

pub type LowerResult<T> = Result<T, super::LowerError>;

use super::LowerError;

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum DisallowedItemKind {
    Rule,
    OverrideRule,
    GrammarBlock,
}

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum LowerErrorKind {
    #[error("missing grammar block")]
    MissingGrammarBlock,
    #[error("override rule(s) not found in base grammar: {}", .0.join(", "))]
    OverrideRuleNotFound(Vec<String>),
    #[error("'override rule' requires a grammar with 'inherits'")]
    OverrideWithoutInherit,
    #[error("failed to resolve '{}': {error}", path.display())]
    ModuleResolveFailed { path: PathBuf, error: String },
    #[error("failed to read '{}': {error}", path.display())]
    ModuleReadFailed { path: PathBuf, error: String },
    #[error("failed to parse '{}': {error}", path.display())]
    ModuleJsonError {
        path: PathBuf,
        error: crate::parse_grammar::ParseGrammarError,
    },
    #[error("unsupported file extension, expected .tsg or .json")]
    ModuleUnsupportedExtension,
    #[error("JSON files can only be used with inherit(), not import()")]
    JsonImportNotAllowed,
    #[error("too many modules (max 256)")]
    ModuleTooMany,
    #[error("module import chain too deep (max 256)")]
    ModuleDepthExceeded,
    #[error("imported files cannot contain a {}", format_disallowed(.0))]
    ModuleDisallowedItem(DisallowedItemKind),
    #[error("module cycle detected")]
    ModuleCycle,
    #[error("member '{0}' not found in module")]
    ModuleMemberNotFound(String),
    #[error("expected a rule name reference")]
    ExpectedRuleName,
    #[error("config field is not set in the base grammar")]
    ConfigFieldUnset,
    #[error("only one inherit() call is allowed per grammar")]
    MultipleInherits,
    #[error("inherit() requires 'inherits' to be set in the grammar config")]
    InheritWithoutConfig,
    #[error("'inherits' must reference a variable bound to inherit()")]
    InheritsWithoutInherit,
    #[error("too many elements (maximum 65535)")]
    TooManyChildren,
    #[error("maximum function call depth ({MAX_CALL_DEPTH}) exceeded")]
    CallDepthExceeded(Vec<(String, PathBuf, usize, usize)>), // name, path, line, col
}

const fn format_disallowed(kind: &DisallowedItemKind) -> &'static str {
    match kind {
        DisallowedItemKind::Rule => "rule",
        DisallowedItemKind::OverrideRule => "override rule",
        DisallowedItemKind::GrammarBlock => "grammar block",
    }
}
