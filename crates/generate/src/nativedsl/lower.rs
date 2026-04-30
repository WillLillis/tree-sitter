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
    ChildRange, ForId, IdentKind, MacroId, ModuleContext, Node, NodeId, PrecKind, QueryableField,
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

const MAX_CALL_DEPTH: u16 = 128;

/// Save the current length of a scratch buffer, run a block, then truncate
/// back. The block receives `base` (the saved length) for slicing.
macro_rules! scratch_scope {
    ($buf:expr, |$base:ident| $body:expr) => {{
        let $base = $buf.len();
        let result = $body;
        $buf.truncate($base);
        result
    }};
}

/// Tracks which modules have had their let bindings evaluated.
#[derive(Default)]
struct LoadedModules([u64; 4]);

impl LoadedModules {
    const fn is_loaded(&self, idx: usize) -> bool {
        self.0[idx / 64] & (1 << (idx % 64)) != 0
    }
    const fn set_loaded(&mut self, idx: usize) {
        self.0[idx / 64] |= 1 << (idx % 64);
    }
}

struct Evaluator<'ast> {
    // Shared AST data (arena + pools), immutable during evaluation
    shared: &'ast SharedAst,
    // All modules including root (root is last, at index root_global_id)
    modules: &'ast [super::Module],
    loaded: LoadedModules,
    // Current execution state
    current_module: usize,
    let_values: FxHashMap<NodeId, ValueId>,
    call_stack: Vec<(&'ast str, Span, usize)>, // name, span, module index
    macro_args: Vec<ValueId>,
    macro_arg_bases: Vec<usize>,
    for_binding_values: Vec<ValueId>,
    for_binding_frames: Vec<(ForId, usize)>,
    val_scratch: Vec<ValueId>,
    rule_scratch: Vec<RuleId>,
    values: Vec<Value>,
    rules: Vec<ARule>,
    rule_children: Vec<RuleId>,
    value_children: Vec<ValueId>,
    object_pool: Vec<FxHashMap<&'ast str, ValueId>>,
    strings: StringPool<'ast>,
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

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
pub fn lower_with_base(shared: &SharedAst, modules: &[super::Module]) -> LowerResult<InputGrammar> {
    let root_ctx = &modules.last().unwrap().ctx;
    let base_grammar = root_ctx.inherit_ref.and_then(|id| {
        if let Node::ModuleRef {
            module: Some(child_id),
            ..
        } = shared.arena.get(id)
        {
            modules[*child_id as usize].lowered.as_ref()
        } else {
            None
        }
    });
    let result = evaluate(shared, modules)?;
    build_grammar(result, base_grammar)
}

/// Evaluate the AST, producing intermediate results.
fn evaluate(shared: &SharedAst, modules: &[super::Module]) -> LowerResult<EvalResult> {
    let root = modules.len() - 1;
    let mut eval = Evaluator {
        shared,
        modules,
        loaded: LoadedModules::default(),
        current_module: root,
        let_values: FxHashMap::default(),
        call_stack: Vec::new(),
        macro_args: Vec::new(),
        macro_arg_bases: Vec::new(),
        for_binding_values: Vec::new(),
        for_binding_frames: Vec::new(),
        val_scratch: Vec::new(),
        rule_scratch: Vec::new(),
        values: Vec::with_capacity(shared.arena.len()),
        rules: Vec::with_capacity(shared.arena.len()),
        rule_children: Vec::with_capacity(shared.arena.len()),
        value_children: Vec::with_capacity(shared.arena.len()),
        object_pool: Vec::new(),
        strings: StringPool::new(),
    };
    let ctx = &modules[root].ctx;
    let mut rule_entries: Vec<(&str, RuleId)> = Vec::with_capacity(ctx.root_items.len());
    let mut override_entries: Vec<(&str, RuleId, Span)> = Vec::new();

    for &item_id in &ctx.root_items {
        match shared.arena.get(item_id) {
            Node::Grammar | Node::Macro(_) => {}
            Node::Let { value, .. } => {
                let val = eval.eval_expr(*value)?;
                eval.let_values.insert(item_id, val);
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
            let entries = overrides
                .into_iter()
                .map(|(name, (_, span))| (name, span))
                .collect();
            return Err(LowerError::without_span(
                LowerErrorKind::OverrideRuleNotFound(entries),
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
        &self.modules[self.current_module].ctx
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
        let grammar = self.modules[mod_idx].lowered.as_ref().unwrap();
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
                        let inner_start = self.value_children.len() as u32;
                        for entry in group {
                            let vid = match entry {
                                PrecedenceEntry::Name(s) => {
                                    let sid = self.strings.intern_owned(s.clone());
                                    self.alloc_val(Value::Str(sid))
                                }
                                PrecedenceEntry::Symbol(s) => self.owned_symbol_val(s.clone()),
                            };
                            self.value_children.push(vid);
                        }
                        let inner_len = u16::try_from(group.len())
                            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
                        Ok(self.alloc_val(Value::List(ChildRange::new(inner_start, inner_len))))
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
                let n = grammar.reserved_words.len();
                let mut map = FxHashMap::with_capacity_and_hasher(n, rustc_hash::FxBuildHasher);
                for rwc in &grammar.reserved_words {
                    let words_vid = self.import_rules_as_list(&rwc.reserved_words, span)?;
                    map.insert(rwc.name.as_str(), words_vid);
                }
                Ok(self.alloc_object(map))
            }
        }
    }

    fn import_rules_as_list(&mut self, rules_data: &[Rule], span: Span) -> LowerResult<ValueId> {
        let start = self.value_children.len() as u32;
        for r in rules_data {
            let rid = self.import_rule(r);
            let vid = self.alloc_val(Value::Rule(rid));
            self.value_children.push(vid);
        }
        let len = u16::try_from(rules_data.len())
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
        Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
    }

    fn import_names_as_list(&mut self, names: &[String], span: Span) -> LowerResult<ValueId> {
        let start = self.value_children.len() as u32;
        for name in names {
            let vid = self.owned_symbol_val(name.clone());
            self.value_children.push(vid);
        }
        let len = u16::try_from(names.len())
            .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
        Ok(self.alloc_val(Value::List(ChildRange::new(start, len))))
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
                let (start, count) = scratch_scope!(self.rule_scratch, |base| {
                    for m in members {
                        let rid = self.import_rule(m);
                        self.rule_scratch.push(rid);
                    }
                    let start = self.children_start();
                    self.rule_children
                        .extend_from_slice(&self.rule_scratch[base..]);
                    (start, self.rule_children.len() as u32 - start)
                });
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
            Node::Ident(IdentKind::Unresolved | IdentKind::Macro(_)) => unreachable!(),
            Node::Ident(IdentKind::Rule) => {
                let sid = self.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::MacroParam(i) => {
                let base = *self.macro_arg_bases.last().unwrap();
                Ok(self.macro_args[base + *i as usize])
            }
            Node::ForBinding { for_id, index } => {
                let base = self
                    .for_binding_frames
                    .iter()
                    .rev()
                    .find(|(id, _)| *id == *for_id)
                    .unwrap()
                    .1;
                Ok(self.for_binding_values[base + *index as usize])
            }
            // Guarded by super::resolve - all variable names validated
            Node::Ident(IdentKind::Var(let_id)) => Ok(*self.let_values.get(let_id).unwrap()),
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
                self.eval_import_module(usize::from(global_id))?;
                Ok(self.alloc_val(Value::Module(global_id)))
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
            // Only unresolved QualifiedAccess reaches here: inherited grammar
            // rules looked up by name in base_grammar.variables.
            Node::QualifiedAccess { obj, member } => {
                let (obj, member) = (*obj, *member);
                let obj_val = self.eval_expr(obj)?;
                let member_name = self.ctx().text(member);
                let Value::Module(idx) = *self.get_val(obj_val) else {
                    unreachable!() // guarded by typecheck
                };
                let table_id = idx as usize;
                if let Some(var) = self.modules[table_id]
                    .lowered
                    .as_ref()
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
                let range = scratch_scope!(self.val_scratch, |base| {
                    for &item_id in items {
                        let v = self.eval_expr(item_id)?;
                        self.val_scratch.push(v);
                    }
                    let start = self.value_children.len() as u32;
                    let len = u16::try_from(self.val_scratch.len() - base)
                        .map_err(|_| LowerError::new(LowerErrorKind::TooManyChildren, span))?;
                    self.value_children
                        .extend_from_slice(&self.val_scratch[base..]);
                    Ok(ChildRange::new(start, len))
                })?;
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
                let Node::Ident(IdentKind::Macro(macro_id)) = self.shared.arena.get(name) else {
                    unreachable!() // guarded by resolver
                };
                self.invoke_macro(*macro_id, args, span, None)
            }
            Node::QualifiedCall(range) => {
                let range = *range;
                let children = self.shared.pools.child_slice(range);
                let (obj, name) = (children[0], children[1]);
                let obj_val = self.eval_expr(obj)?;
                let Value::Module(idx) = *self.get_val(obj_val) else {
                    unreachable!() // guarded by typecheck
                };
                let table_id = idx as usize;
                self.eval_import_module(table_id)?;
                let Node::Ident(IdentKind::Macro(macro_id)) = self.shared.arena.get(name) else {
                    unreachable!() // guarded by resolver
                };
                let args = ChildRange::new(range.start + 2, range.len - 2);
                self.invoke_macro(*macro_id, args, span, Some(table_id))
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
                let (offset, len) = scratch_scope!(self.rule_scratch, |base| {
                    for &member in children {
                        if let Node::For { for_id, body } = *self.shared.arena.get(member) {
                            self.eval_for_to_rules(for_id, body)?;
                        } else {
                            let rid = self.lower_to_rule(member)?;
                            self.rule_scratch.push(rid);
                        }
                    }
                    let start = self.children_start();
                    self.rule_children
                        .extend_from_slice(&self.rule_scratch[base..]);
                    self.children_range(start, span)
                })?;
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
                    let ctx = &self.modules[*mod_idx].ctx;
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

    /// Evaluate a macro call. If `module` is `Some`, switches to that module
    /// for the body evaluation (cross-module call).
    fn invoke_macro(
        &mut self,
        macro_id: MacroId,
        args: ChildRange,
        span: Span,
        module: Option<usize>,
    ) -> LowerResult<ValueId> {
        let config = self.shared.pools.get_macro(macro_id);
        let macro_name = match module {
            Some(id) => self.modules[id].ctx.text(config.name),
            None => self.ctx().text(config.name),
        };
        self.push_call(macro_name, span)?;
        // Evaluate args before pushing base - args are evaluated in
        // the caller's macro context, not the callee's.
        scratch_scope!(self.macro_args, |base| {
            for &arg_id in self.shared.pools.child_slice(args) {
                let v = self.eval_expr(arg_id)?;
                self.macro_args.push(v);
            }
            self.macro_arg_bases.push(base);
            let saved = self.current_module;
            if let Some(id) = module {
                self.current_module = id;
            }
            let result = self.eval_expr(config.body);
            self.current_module = saved;
            self.call_stack.pop();
            self.macro_arg_bases.pop();
            result
        })
    }

    /// Lazily evaluate an imported module's top-level let bindings.
    fn eval_import_module(&mut self, table_id: usize) -> LowerResult<()> {
        if self.loaded.is_loaded(table_id) {
            return Ok(());
        }
        self.loaded.set_loaded(table_id);

        let saved_module = self.current_module;
        self.current_module = table_id;
        let result = self.eval_let_bindings();
        self.current_module = saved_module;
        result
    }

    /// Evaluate top-level let bindings in the current module context.
    fn eval_let_bindings(&mut self) -> LowerResult<()> {
        let ctx = &self.modules[self.current_module].ctx;
        let n = ctx.root_items.len();
        for i in 0..n {
            let item_id = ctx.root_items[i];
            if let Node::Let { value, .. } = self.shared.arena.get(item_id) {
                let value = *value;
                let val = self.eval_expr(value)?;
                self.let_values.insert(item_id, val);
            }
        }
        Ok(())
    }

    /// Evaluate a for-loop, pushing each iteration's rule into `rule_scratch`.
    fn eval_for_to_rules(&mut self, for_id: ForId, body: NodeId) -> LowerResult<()> {
        let for_config = self.shared.pools.get_for(for_id);
        let iterable = for_config.iterable;
        let n_bindings = for_config.bindings.len();
        let iter_vid = self.eval_expr(iterable)?;
        let iter_range = self.list_range(Self::expect_list(iter_vid));
        let n_items = iter_range.len as usize;
        scratch_scope!(self.for_binding_values, |base| {
            self.for_binding_frames.push((for_id, base));
            for i in 0..n_items {
                let item_id = self.value_children[iter_range.start as usize + i];
                for j in 0..n_bindings {
                    let val = match *self.get_val(item_id) {
                        Value::Tuple(range) => self.value_children[range.start as usize + j],
                        _ => item_id,
                    };
                    if i == 0 {
                        self.for_binding_values.push(val);
                    } else {
                        self.for_binding_values[base + j] = val;
                    }
                }
                let rid = self.lower_to_rule(body)?;
                self.rule_scratch.push(rid);
            }
            self.for_binding_frames.pop();
            Ok(())
        })
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
    #[error("override rule(s) not found in base grammar: {}", format_override_names(.0))]
    OverrideRuleNotFound(Vec<(String, Span)>),
    #[error("'override rule' requires a grammar with 'inherits'")]
    OverrideWithoutInherit,
    #[error("failed to resolve '{}': {error}", path.display())]
    ModuleResolveFailed { path: PathBuf, error: String },
    #[error("failed to read '{}': {error}", path.display())]
    ModuleReadFailed { path: PathBuf, error: String },
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
    #[error("inherit() requires 'inherits' to be set in the grammar config")]
    InheritWithoutConfig,
    #[error("'inherits' must reference a variable bound to inherit()")]
    InheritsWithoutInherit,
    #[error("too many elements (maximum 65535)")]
    TooManyChildren,
    #[error("maximum macro call depth ({MAX_CALL_DEPTH}) exceeded")]
    CallDepthExceeded(Vec<(String, PathBuf, usize, usize)>), // name, path, line, col
}

fn format_override_names(entries: &[(String, Span)]) -> String {
    entries
        .iter()
        .map(|(n, _)| n.as_str())
        .collect::<Vec<_>>()
        .join(", ")
}

const fn format_disallowed(kind: &DisallowedItemKind) -> &'static str {
    match kind {
        DisallowedItemKind::Rule => "rule",
        DisallowedItemKind::OverrideRule => "override rule",
        DisallowedItemKind::GrammarBlock => "grammar block",
    }
}
