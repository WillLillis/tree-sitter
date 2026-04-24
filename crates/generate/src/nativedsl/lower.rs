//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].

use std::num::NonZeroU32;
use std::path::{Path, PathBuf};

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use crate::{
    grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType},
    rules::{Associativity, Precedence, Rule},
};

use super::{
    ast::{Ast, FnConfig, ForConfig, Node, NodeId, PrecKind, RepeatKind, Span},
    scope_stack::ScopeStack,
};

#[derive(Clone, Copy, PartialEq, Eq)]
struct Str(NonZeroU32);

enum StrEntry<'src> {
    Unreachable,
    /// Span into a source string, captured at intern time so that entries
    /// from different modules (import/inherit) resolve correctly.
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
    Seq(u32, u16),
    Choice(u32, u16),
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

enum Value<'src> {
    Int(i32),
    Str(Str),
    Rule(RuleId),
    Object(FxHashMap<&'src str, ValueId>),
    List(Vec<ValueId>),
    Tuple(Vec<ValueId>),
    Module(u8),
    /// Grammar config accessor - lazily resolves fields on `.` access.
    GrammarConfig,
}

const MAX_CALL_DEPTH: u16 = 64;

/// Flat entry for one module in the evaluator's module table.
struct ModuleInfo<'ast> {
    ast: &'ast Ast,
    path: &'ast Path,
    base_grammar: Option<&'ast InputGrammar>,
}

struct Evaluator<'ast> {
    // Flat module table (immutable after construction)
    modules: Vec<ModuleInfo<'ast>>,
    module_fns: Vec<FxHashMap<&'ast str, NodeId>>,
    // Per-module let binding values (lazily populated on first access)
    module_values: Vec<Option<FxHashMap<&'ast str, ValueId>>>,
    // Offset: table index = global_id - base_id
    base_id: usize,
    // Current execution state
    current_module: usize,
    scopes: ScopeStack<'ast, ValueId>,
    // Shared across module boundaries
    call_stack: Vec<(&'ast str, Span, usize)>, // name, span, module index
    values: Vec<Value<'ast>>,
    rules: Vec<ARule>,
    rule_children: Vec<RuleId>,
    strings: StringPool<'ast>,
}

/// Collect all `fn` definitions from an AST into a name -> `NodeId` map.
fn collect_fns(ast: &Ast) -> FxHashMap<&str, NodeId> {
    let mut fns = FxHashMap::default();
    for &item_id in &ast.root_items {
        if let Node::Fn(fn_idx) = ast.node(item_id) {
            let config = ast.ctx.get_fn(*fn_idx);
            fns.insert(ast.ctx.text(config.name), item_id);
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
    root_ast: &'ast Ast,
    root_path: &'ast Path,
    root_base_grammar: Option<&'ast InputGrammar>,
    sub_modules: &'ast [super::Module],
) -> (Vec<ModuleInfo<'ast>>, Vec<FxHashMap<&'ast str, NodeId>>) {
    let mut infos = vec![ModuleInfo {
        ast: root_ast,
        path: root_path,
        base_grammar: root_base_grammar,
    }];
    let mut fns = vec![collect_fns(root_ast)];

    fn add_children<'a>(
        modules: &'a [super::Module],
        infos: &mut Vec<ModuleInfo<'a>>,
        fns: &mut Vec<FxHashMap<&'a str, NodeId>>,
    ) {
        for m in modules {
            infos.push(ModuleInfo {
                ast: &m.ast,
                path: &m.path,
                base_grammar: m.lowered.as_ref(),
            });
            fns.push(collect_fns(&m.ast));
            add_children(&m.sub_modules, infos, fns);
        }
    }

    add_children(sub_modules, &mut infos, &mut fns);
    (infos, fns)
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
pub fn lower_with_base(
    ast: &Ast,
    modules: &[super::Module],
    root_global_id: u8,
    root_path: &Path,
) -> LowerResult<InputGrammar> {
    let base_grammar = modules.iter().find_map(|m| m.lowered.as_ref());
    let result = evaluate(ast, modules, base_grammar, root_global_id, root_path)?;
    build_grammar(result, base_grammar)
}

/// Evaluate the AST, producing intermediate results.
fn evaluate(
    ast: &Ast,
    modules: &[super::Module],
    base_grammar: Option<&InputGrammar>,
    root_global_id: u8,
    root_path: &Path,
) -> LowerResult<EvalResult> {
    let (module_infos, module_fns) = build_module_table(ast, root_path, base_grammar, modules);
    let n = module_infos.len();
    let mut eval = Evaluator {
        modules: module_infos,
        module_fns,
        module_values: vec![None; n],
        base_id: root_global_id as usize,
        current_module: 0,
        scopes: ScopeStack::new(),
        call_stack: Vec::new(),
        values: Vec::with_capacity(ast.arena.len()),
        rules: Vec::with_capacity(ast.arena.len()),
        rule_children: Vec::with_capacity(ast.arena.len()),
        strings: StringPool::new(),
    };
    let mut rule_entries: Vec<(&str, RuleId)> = Vec::with_capacity(ast.root_items.len());
    let mut override_entries: Vec<(&str, RuleId, Span)> = Vec::new();

    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Grammar | Node::Fn(_) => {}
            Node::Let { name, value, .. } => {
                let val = eval.eval_expr(*value)?;
                eval.scopes.insert(ast.node_text(*name), val);
            }
            Node::Rule {
                is_override,
                name,
                body,
            } => {
                let rule_id = eval.lower_to_rule(*body)?;
                if *is_override {
                    override_entries.push((ast.ctx.text(*name), rule_id, *name));
                } else {
                    rule_entries.push((ast.ctx.text(*name), rule_id));
                }
            }
            _ => unreachable!(),
        }
    }

    // grammar_config is guaranteed present by validate_grammar in mod.rs,
    // language is guaranteed present by the parser (MissingLanguageField error).
    let config = ast.ctx.grammar_config.as_ref().unwrap();
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
    fn ast(&self) -> &'ast Ast {
        self.modules[self.current_module].ast
    }

    fn module_idx(&self, global_id: u8) -> usize {
        debug_assert!(
            global_id as usize >= self.base_id,
            "global_id {global_id} is below base_id {}",
            self.base_id
        );
        global_id as usize - self.base_id
    }

    fn alloc_val(&mut self, val: Value<'ast>) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(val);
        id
    }

    fn get_val(&self, id: ValueId) -> &Value<'ast> {
        // Safety: id was produced by alloc_val which returns sequential indices
        // into self.values. No ValueId can outlive the Evaluator.
        unsafe { self.values.get_unchecked(id.0 as usize) }
    }

    // -- Typed value constructors and accessors --
    //
    // `expect_*` wraps a `ValueId` in a typed newtype. The compile-time
    // safety comes from the wrapper: you can't call `list_items` without
    // going through `expect_list`. Each accessor has a single `unreachable!`
    // as the runtime safety net.

    const fn expect_list(vid: ValueId) -> ListVal {
        ListVal(vid)
    }
    fn list_items(&self, v: ListVal) -> &[ValueId] {
        let Value::List(items) = self.get_val(v.0) else {
            unreachable!()
        };
        items
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
        let Value::Object(map) = self.get_val(v.0) else {
            unreachable!()
        };
        map
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

    fn get_fn_config(&self, fn_id: NodeId) -> &'ast FnConfig {
        match self.ast().node(fn_id) {
            Node::Fn(fn_idx) => self.ast().ctx.get_fn(*fn_idx),
            _ => unreachable!(),
        }
    }

    /// Intern a span using the current module's source text.
    fn intern_span(&mut self, span: Span) -> Str {
        self.strings.intern_span(span, &self.ast().ctx.source)
    }

    fn intern_string_lit(&mut self, span: Span) -> Str {
        let raw = self.ast().ctx.text(span);
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
            ARule::Seq(start, len) | ARule::Choice(start, len) => {
                let is_seq = matches!(self.get_rule(id), ARule::Seq(..));
                let range = *start as usize..*start as usize + *len as usize;
                // Safety: start and len are produced by children_start/children_range
                // which track rule_children indices.
                let children: Vec<Rule> = unsafe { self.rule_children.get_unchecked(range) }
                    .iter()
                    .map(|&id| self.build_rule(id))
                    .collect();
                if is_seq {
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

    fn rule_name_string(&self, v: ValueId) -> String {
        let rid = self.rule_id(Self::expect_rule(v));
        let ARule::NamedSymbol(sid) = self.get_rule(rid) else {
            unreachable!("expect_name_ref guarantees NamedSymbol")
        };
        self.strings.to_string(*sid)
    }

    fn eval_name_list(&mut self, id: NodeId) -> LowerResult<Vec<String>> {
        let vid = self.eval_expr(id)?;
        let items = self.list_items(Self::expect_list(vid));
        Ok(items.iter().map(|&v| self.rule_name_string(v)).collect())
    }

    /// Evaluate a `conflicts` expression into `Vec<Vec<String>>`. Each inner
    /// element must evaluate to a named rule reference.
    fn eval_conflicts(&mut self, id: NodeId) -> LowerResult<Vec<Vec<String>>> {
        let vid = self.eval_expr(id)?;
        let outer = self.list_items(Self::expect_list(vid));
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(Self::expect_list(group_vid));
                Ok(inner.iter().map(|&v| self.rule_name_string(v)).collect())
            })
            .collect()
    }

    /// Evaluate a `precedences` expression into `Vec<Vec<PrecedenceEntry>>`.
    /// Each inner element is either a string literal (`PrecedenceEntry::Name`)
    /// or a named rule reference (`PrecedenceEntry::Symbol`).
    fn eval_precedences(&mut self, id: NodeId) -> LowerResult<Vec<Vec<PrecedenceEntry>>> {
        let vid = self.eval_expr(id)?;
        let outer = self.list_items(Self::expect_list(vid));
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(Self::expect_list(group_vid));
                inner
                    .iter()
                    .map(|&v| match *self.get_val(v) {
                        Value::Str(sid) => Ok(PrecedenceEntry::Name(self.strings.to_string(sid))),
                        Value::Rule(_) => Ok(PrecedenceEntry::Symbol(self.rule_name_string(v))),
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
                self.ast().span(id),
            )),
            _ => Err(LowerError::new(
                LowerErrorKind::ExpectedRuleName,
                self.ast().span(id),
            )),
        }
    }

    /// Evaluate the `reserved` config expression into `Vec<ReservedWordContext<Rule>>`.
    /// Accepts either an object literal `{ name: [words] }` or an expression
    /// evaluating to a list of `{name: str, words: list<rule>}` objects.
    fn eval_reserved(&mut self, id: NodeId) -> LowerResult<Vec<ReservedWordContext<Rule>>> {
        let ast = self.ast();
        // Object literal: each field is a named word set
        if let Node::Object(range) = ast.node(id) {
            return ast
                .ctx
                .get_object(*range)
                .iter()
                .map(|&(name_span, val_id)| {
                    let words = self.eval_rule_list(val_id)?;
                    Ok(ReservedWordContext {
                        name: ast.ctx.text(name_span).to_string(),
                        reserved_words: words,
                    })
                })
                .collect();
        }
        // Expression: evaluate to list of {name, words} objects
        let vid = self.eval_expr(id)?;
        let items = self.list_items(Self::expect_list(vid));
        items
            .iter()
            .map(|&v| {
                let map = self.object_fields(Self::expect_object(v));
                let name_vid = *map.get("name").unwrap();
                let words_vid = *map.get("words").unwrap();
                let name = self
                    .strings
                    .to_string(self.str_id(Self::expect_str(name_vid)));
                let word_vals = self.list_items(Self::expect_list(words_vid));
                let words = word_vals
                    .iter()
                    .map(|&wv| self.value_to_owned_rule(wv))
                    .collect();
                Ok(ReservedWordContext {
                    name,
                    reserved_words: words,
                })
            })
            .collect()
    }

    /// Import a config field from the base grammar as a DSL value.
    /// Returns an error if the field name is not a known config field.
    fn import_config_field_or_err(&mut self, field: &str, span: Span) -> LowerResult<ValueId> {
        let grammar = self.modules[self.current_module].base_grammar.unwrap();
        match field {
            "extras" => Ok(self.import_rules_as_list(&grammar.extra_symbols)),
            "externals" => Ok(self.import_rules_as_list(&grammar.external_tokens)),
            "inline" => Ok(self.import_names_as_list(&grammar.variables_to_inline)),
            "supertypes" => Ok(self.import_names_as_list(&grammar.supertype_symbols)),
            "conflicts" => {
                let vals: Vec<ValueId> = grammar
                    .expected_conflicts
                    .iter()
                    .map(|g| self.import_names_as_list(g))
                    .collect();
                Ok(self.alloc_val(Value::List(vals)))
            }
            "precedences" => {
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
                                PrecedenceEntry::Symbol(s) => {
                                    let sid = self.strings.intern_owned(s.clone());
                                    let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                                    self.alloc_val(Value::Rule(rid))
                                }
                            })
                            .collect();
                        self.alloc_val(Value::List(inner))
                    })
                    .collect();
                Ok(self.alloc_val(Value::List(vals)))
            }
            "word" => {
                if let Some(name) = &grammar.word_token {
                    let sid = self.strings.intern_owned(name.clone());
                    let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                    Ok(self.alloc_val(Value::Rule(rid)))
                } else {
                    Err(LowerError::new(LowerErrorKind::ConfigFieldUnset, span))
                }
            }
            "reserved" => {
                let sets: Vec<ValueId> = grammar
                    .reserved_words
                    .iter()
                    .map(|ctx| {
                        let name_sid = self.strings.intern_owned(ctx.name.clone());
                        let name_val = self.alloc_val(Value::Str(name_sid));
                        let words = self.import_rules_as_list(&ctx.reserved_words);
                        let map = FxHashMap::from_iter([("name", name_val), ("words", words)]);
                        self.alloc_val(Value::Object(map))
                    })
                    .collect();
                Ok(self.alloc_val(Value::List(sets)))
            }
            _ => Err(LowerError::new(
                LowerErrorKind::ModuleMemberNotFound(field.to_string()),
                span,
            )),
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
                let is_seq = matches!(rule, Rule::Seq(_));
                let imported: Vec<RuleId> = members.iter().map(|m| self.import_rule(m)).collect();
                let start = self.children_start();
                self.rule_children.extend_from_slice(&imported);
                let count = self.rule_children.len() as u32 - start;
                debug_assert!(
                    u16::try_from(count).is_ok(),
                    "inherited rule has too many children"
                );
                let len = count as u16;
                if is_seq {
                    self.alloc_rule(ARule::Seq(start, len))
                } else {
                    self.alloc_rule(ARule::Choice(start, len))
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
                    rid = self.alloc_rule(ARule::Token(true, rid));
                } else if params.is_token {
                    rid = self.alloc_rule(ARule::Token(false, rid));
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
        let ast = self.ast();
        let span = ast.span(id);
        match ast.node(id) {
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
            // Guarded by super::resolve - all Idents replaced with RuleRef/VarRef
            Node::Ident => unreachable!(),
            Node::RuleRef => {
                let sid = self.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            // Guarded by super::resolve::resolve - all variable names validated
            Node::VarRef => Ok(self.scopes.get(ast.ctx.text(span)).unwrap()),
            Node::GrammarConfig(inner) => {
                let inner_val = self.eval_expr(*inner)?;
                let Value::Module(_) = *self.get_val(inner_val) else {
                    unreachable!() // guarded by typecheck
                };
                Ok(self.alloc_val(Value::GrammarConfig))
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
                let li = self.list_items(Self::expect_list(lv));
                let ri = self.list_items(Self::expect_list(rv));
                let mut combined = Vec::with_capacity(li.len() + ri.len());
                combined.extend_from_slice(li);
                combined.extend_from_slice(ri);
                Ok(self.alloc_val(Value::List(combined)))
            }
            Node::FieldAccess { obj, field } => {
                let (obj, field) = (*obj, *field);
                let obj_val = self.eval_expr(obj)?;
                let field_name = ast.ctx.text(field);
                match self.get_val(obj_val) {
                    Value::Object(map) => Ok(*map.get(field_name).unwrap()),
                    Value::GrammarConfig => self.import_config_field_or_err(field_name, field),
                    _ => unreachable!(), // guarded by typecheck
                }
            }
            Node::QualifiedAccess { obj, member } => {
                let (obj, member) = (*obj, *member);
                let obj_val = self.eval_expr(obj)?;
                let member_name = ast.ctx.text(member);
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
                let fields = ast.ctx.get_object(*range);
                let mut map =
                    FxHashMap::with_capacity_and_hasher(fields.len(), rustc_hash::FxBuildHasher);
                for &(key_span, value_id) in fields {
                    map.insert(ast.ctx.text(key_span), self.eval_expr(value_id)?);
                }
                Ok(self.alloc_val(Value::Object(map)))
            }
            Node::List(range) | Node::Tuple(range) => {
                let items = ast.ctx.child_slice(*range);
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
                for &part_id in ast.ctx.child_slice(*range) {
                    let vid = self.eval_expr(part_id)?;
                    result.push_str(self.strings.resolve(self.get_str_val(vid)));
                }
                let sid = self.strings.intern_owned(result);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::DynRegex { pattern, flags } => {
                let pv = self.eval_expr(*pattern)?;
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
                let mut arg_vals = Vec::with_capacity(ast.ctx.child_slice(args).len());
                for &arg_id in ast.ctx.child_slice(args) {
                    arg_vals.push(self.eval_expr(arg_id)?);
                }
                self.eval_fn_call(ast.node_text(name), &arg_vals, span)
            }
            Node::QualifiedCall(range) => {
                let (obj, name, args) = ast.ctx.get_qualified_call(*range);
                let obj_val = self.eval_expr(obj)?;
                let Value::Module(idx) = *self.get_val(obj_val) else {
                    unreachable!() // guarded by typecheck
                };
                // Evaluate args in current AST context
                let mut arg_vals = Vec::with_capacity(args.len());
                for &arg_id in args {
                    arg_vals.push(self.eval_expr(arg_id)?);
                }
                let fn_name = ast.node_text(name);
                self.eval_import_fn_call(idx as usize, fn_name, &arg_vals, span)
            }
            _ => {
                let rid = self.eval_combinator(id)?;
                Ok(self.alloc_val(Value::Rule(rid)))
            }
        }
    }

    fn lower_to_rule(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let ast = self.ast();
        match ast.node(id) {
            Node::RuleRef => {
                let sid = self.intern_span(ast.span(id));
                Ok(self.alloc_rule(ARule::NamedSymbol(sid)))
            }
            Node::StringLit => {
                let sid = self.intern_string_lit(ast.span(id));
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(ast.span(id), *hash_count);
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            #[rustfmt::skip]
            Node::Seq(_) | Node::Choice(_) | Node::Repeat { .. }
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

    fn value_to_prec(&self, id: ValueId) -> APrec {
        match self.get_val(id) {
            Value::Int(n) => APrec::Integer(*n),
            Value::Str(s) => APrec::Name(*s),
            // Guarded by super::typecheck::type_of (Node::Prec) - ensures value is int or str
            _ => unreachable!(),
        }
    }

    fn eval_combinator(&mut self, id: NodeId) -> LowerResult<RuleId> {
        let ast = self.ast();
        match ast.node(id) {
            Node::Seq(range) | Node::Choice(range) => {
                let is_seq = matches!(ast.node(id), Node::Seq(_));
                let children = ast.ctx.child_slice(*range);
                // Collect into a temporary Vec before pushing to rule_children.
                // We can't push directly because nested combinators (e.g.
                // seq inside choice) also push their children to rule_children
                // during lower_to_rule. The parent's children must go after.
                let mut child_ids = Vec::with_capacity(children.len());
                for &member in children {
                    if let Node::For(idx) = ast.node(member) {
                        child_ids.extend(self.eval_for_to_rules(ast.ctx.get_for(*idx))?);
                    } else {
                        child_ids.push(self.lower_to_rule(member)?);
                    }
                }
                let start = self.children_start();
                self.rule_children.extend_from_slice(&child_ids);
                let (offset, len) = self.children_range(start, ast.span(id))?;
                let arule = if is_seq {
                    ARule::Seq(offset, len)
                } else {
                    ARule::Choice(offset, len)
                };
                Ok(self.alloc_rule(arule))
            }
            Node::Repeat { kind, inner } => {
                let inner = self.lower_to_rule(*inner)?;
                match kind {
                    RepeatKind::ZeroOrMore => {
                        let rep = self.alloc_rule(ARule::Repeat(inner));
                        let blank = self.alloc_rule(ARule::Blank);
                        let start = self.children_start();
                        self.rule_children.extend_from_slice(&[rep, blank]);
                        let (s, l) = self.children_range(start, ast.span(id)).unwrap();
                        Ok(self.alloc_rule(ARule::Choice(s, l)))
                    }
                    RepeatKind::OneOrMore => Ok(self.alloc_rule(ARule::Repeat(inner))),
                    RepeatKind::Optional => {
                        let blank = self.alloc_rule(ARule::Blank);
                        let start = self.children_start();
                        self.rule_children.extend_from_slice(&[inner, blank]);
                        let (s, l) = self.children_range(start, ast.span(id)).unwrap();
                        Ok(self.alloc_rule(ARule::Choice(s, l)))
                    }
                }
            }
            Node::Blank => Ok(self.alloc_rule(ARule::Blank)),
            Node::Field { name, content } => {
                let inner = self.lower_to_rule(*content)?;
                let sid = self.intern_span(ast.span(*name));
                Ok(self.alloc_rule(ARule::Field(sid, inner)))
            }
            Node::Alias { content, target } => {
                let (content, target) = (*content, *target);
                let inner = self.lower_to_rule(content)?;
                match ast.node(target) {
                    Node::RuleRef | Node::VarRef => {
                        let sid = self.intern_span(ast.span(target));
                        Ok(self.alloc_rule(ARule::Alias(sid, true, inner)))
                    }
                    _ => {
                        let vid = self.eval_expr(target)?;
                        match self.get_val(vid) {
                            Value::Str(s) => Ok(self.alloc_rule(ARule::Alias(*s, false, inner))),
                            Value::Rule(rid) => {
                                // Guarded by super::typecheck::type_of (Node::Alias) -
                                // InvalidAliasTarget rejects non-name/non-str targets
                                let ARule::NamedSymbol(s) = self.get_rule(*rid) else {
                                    unreachable!()
                                };
                                Ok(self.alloc_rule(ARule::Alias(*s, true, inner)))
                            }
                            // Guarded by super::typecheck::type_of (Node::Alias) -
                            // only str or rule-name targets pass
                            _ => unreachable!(),
                        }
                    }
                }
            }
            Node::Token { immediate, inner } => {
                let inner = self.lower_to_rule(*inner)?;
                let arule = ARule::Token(*immediate, inner);
                Ok(self.alloc_rule(arule))
            }
            Node::Prec {
                kind,
                value,
                content,
            } => {
                let vid = self.eval_expr(*value)?;
                let inner = self.lower_to_rule(*content)?;
                if *kind == PrecKind::Dynamic {
                    let n = self.int_val(Self::expect_int(vid));
                    Ok(self.alloc_rule(ARule::PrecDynamic(n, inner)))
                } else {
                    let prec = self.value_to_prec(vid);
                    let ctor = match kind {
                        PrecKind::Left => ARule::PrecLeft,
                        PrecKind::Right => ARule::PrecRight,
                        _ => ARule::Prec,
                    };
                    Ok(self.alloc_rule(ctor(prec, inner)))
                }
            }
            Node::Reserved { context, content } => {
                let inner = self.lower_to_rule(*content)?;
                let sid = self.intern_span(ast.span(*context));
                Ok(self.alloc_rule(ARule::Reserved(sid, inner)))
            }
            // Guarded by lower_to_rule - only combinator nodes are dispatched here
            _ => unreachable!(),
        }
    }

    fn push_call(&mut self, name: &'ast str, span: Span) -> LowerResult<()> {
        self.call_stack.push((name, span, self.current_module));
        if self.call_stack.len() > MAX_CALL_DEPTH as usize {
            return Err(self.build_call_depth_error());
        }
        Ok(())
    }

    fn build_call_depth_error(&self) -> LowerError {
        let trace = self
            .call_stack
            .iter()
            .map(|(name, span, mod_idx)| {
                let info = &self.modules[*mod_idx];
                let offset = span.start as usize;
                let bytes = info.ast.ctx.source.as_bytes();
                let mut line_start = 0;
                let mut line = 1usize;
                for (i, &b) in bytes.iter().enumerate().take(offset) {
                    if b == b'\n' {
                        line_start = i + 1;
                        line += 1;
                    }
                }
                let col = offset - line_start + 1;
                (name.to_string(), info.path.to_path_buf(), line, col)
            })
            .collect();
        let root_span = self.call_stack[0].1;
        LowerError::new(LowerErrorKind::CallDepthExceeded(trace), root_span)
    }

    fn eval_fn_call(
        &mut self,
        name: &'ast str,
        arg_vals: &[ValueId],
        call_span: Span,
    ) -> LowerResult<ValueId> {
        self.push_call(name, call_span)?;
        // Guarded by super::typecheck::check_call_args - validates function exists
        let &fn_id = self.module_fns[self.current_module].get(name).unwrap();
        let config = self.get_fn_config(fn_id);
        let body = config.body;
        self.scopes.push();
        for (i, &arg_val) in arg_vals.iter().enumerate() {
            self.scopes
                .insert(self.ast().ctx.text(config.params[i].name), arg_val);
        }
        let result = self.eval_expr(body);
        self.scopes.pop();
        self.call_stack.pop();
        result
    }

    /// Lazily evaluate an imported module's top-level let bindings.
    fn eval_import_module(&mut self, table_id: usize) -> LowerResult<()> {
        if self.module_values[table_id].is_some() {
            return Ok(());
        }

        let saved_module = self.current_module;
        let saved_scopes = std::mem::replace(&mut self.scopes, ScopeStack::new());
        self.current_module = table_id;

        let result = self.eval_let_bindings();

        self.current_module = saved_module;
        self.scopes = saved_scopes;

        self.module_values[table_id] = Some(result?);
        Ok(())
    }

    /// Evaluate top-level let bindings in the current module context.
    fn eval_let_bindings(&mut self) -> LowerResult<FxHashMap<&'ast str, ValueId>> {
        let n = self.ast().root_items.len();
        self.scopes.push();
        let mut values = FxHashMap::default();
        for i in 0..n {
            let item_id = self.ast().root_items[i];
            if let Node::Let { name, value, .. } = self.ast().node(item_id) {
                let (name, value) = (*name, *value);
                let val = self.eval_expr(value)?;
                let var_name = self.ast().node_text(name);
                self.scopes.insert(var_name, val);
                values.insert(var_name, val);
            }
        }
        self.scopes.pop();
        Ok(values)
    }

    /// Call a function defined in an imported module.
    fn eval_import_fn_call(
        &mut self,
        table_id: usize,
        fn_name: &'ast str,
        arg_vals: &[ValueId],
        call_span: Span,
    ) -> LowerResult<ValueId> {
        let import_ast = self.modules[table_id].ast;
        let fn_node_id = self.module_fns[table_id][fn_name];
        let fn_idx = match import_ast.node(fn_node_id) {
            Node::Fn(fi) => *fi,
            _ => unreachable!(),
        };
        let fn_config = import_ast.ctx.get_fn(fn_idx);
        let body = fn_config.body;

        // Record the call with the caller's span but the callee's module,
        // so the trace shows the correct file path for the function.
        self.call_stack
            .push((fn_name, call_span, self.current_module));
        if self.call_stack.len() > MAX_CALL_DEPTH as usize {
            return Err(self.build_call_depth_error());
        }

        let saved_module = self.current_module;
        let saved_scopes = std::mem::replace(&mut self.scopes, ScopeStack::new());
        self.current_module = table_id;

        self.scopes.push();
        // Bind import's let values. Direct field access allows split borrowing
        // of self.module_values (immutable) and self.scopes (mutable).
        for (&name, &val) in self.module_values[table_id].as_ref().unwrap() {
            self.scopes.insert(name, val);
        }
        for (i, param) in fn_config.params.iter().enumerate() {
            self.scopes
                .insert(import_ast.ctx.text(param.name), arg_vals[i]);
        }

        let result = self.eval_expr(body);
        self.call_stack.pop();
        self.scopes.pop();

        self.current_module = saved_module;
        self.scopes = saved_scopes;
        result
    }

    fn eval_for_to_rules(&mut self, config: &ForConfig) -> LowerResult<Vec<RuleId>> {
        let iter_vid = self.eval_expr(config.iterable)?;
        let n_items = self.list_items(Self::expect_list(iter_vid)).len();
        let mut results = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let item_id = self.list_items(Self::expect_list(iter_vid))[i];
            self.scopes.push();
            for (j, &(name_span, _)) in config.bindings.iter().enumerate() {
                let value_id = match self.get_val(item_id) {
                    Value::Tuple(ids) => ids[j],
                    _ => item_id,
                };
                self.scopes.insert(self.ast().ctx.text(name_span), value_id);
            }
            results.push(self.lower_to_rule(config.body)?);
            self.scopes.pop();
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
    #[error("failed to resolve '{path}': {error}")]
    ModuleResolveFailed { path: String, error: String },
    #[error("failed to read '{path}': {error}")]
    ModuleReadFailed { path: String, error: String },
    #[error("failed to parse '{path}': {error}")]
    ModuleJsonError {
        path: String,
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
