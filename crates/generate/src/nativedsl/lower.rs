//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].
//!
//! Walks the AST, evaluating expressions into an intermediate arena, then
//! converts to output [`InputGrammar`] / [`Rule`] types. Key types:
//! [`StringPool`] (interned strings), rule arena (`Vec<ARule>`), value arena
//! (`Vec<Value>`), and [`Evaluator`] (scope chain + function registry).

use std::num::NonZeroU32;
use std::path::Path;

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use crate::grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType};
use crate::nativedsl::scope_stack::ScopeStack;
use crate::rules::{Precedence, Rule};

use super::ast::{Ast, FnConfig, ForConfig, Node, NodeId, Note, Span};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Str(NonZeroU32);

enum StrEntry<'src> {
    Unreachable,
    /// Span into a source string, captured at intern time so that entries
    /// from different modules (import/inherit) resolve correctly.
    Source(Span, &'src str),
    Owned(String),
}

struct StringPool<'src> {
    /// Current source string, used by `intern_span` to capture the
    /// source reference into the entry. Swapped when evaluating imports.
    source: &'src str,
    entries: Vec<StrEntry<'src>>,
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
        self.entries.push(StrEntry::Source(span, self.source));
        id
    }

    fn intern_owned(&mut self, s: String) -> Str {
        let id = Str(NonZeroU32::new(self.entries.len() as u32).unwrap());
        self.entries.push(StrEntry::Owned(s));
        id
    }

    #[inline]
    fn resolve(&self, id: Str) -> &str {
        match &self.entries[id.0.get() as usize] {
            StrEntry::Source(span, source) => span.resolve(source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }

    #[inline]
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

/// Typed wrappers around [`ValueId`] that guarantee the underlying value has
/// been validated. Constructed via `Evaluator::expect_*` methods; zero-cost
/// at runtime (same size as `ValueId`).
#[derive(Clone, Copy)]
struct ListVal(ValueId);
#[derive(Clone, Copy)]
struct RuleVal(ValueId);
#[derive(Clone, Copy)]
struct StrVal(ValueId);
#[derive(Clone, Copy)]
struct IntVal(ValueId);
#[derive(Clone, Copy)]
struct ObjectVal(ValueId);
#[derive(Clone, Copy)]
struct TupleVal(ValueId);

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

struct Evaluator<'ast> {
    ast: &'ast Ast,
    values: Vec<Value<'ast>>,
    scopes: ScopeStack<'ast, ValueId>,
    fns: FxHashMap<&'ast str, NodeId>,
    rules: Vec<ARule>,
    rule_children: Vec<RuleId>,
    strings: StringPool<'ast>,
    base_grammar: Option<&'ast InputGrammar>,
    call_stack: Vec<(&'ast str, Span)>,
    /// Loaded import modules, indexed by `Value::Import(idx)`.
    import_modules: &'ast [super::Module],
    /// Evaluated state for each import module, populated when
    /// `let h = import(...)` is first evaluated during lowering.
    import_evals: Vec<ImportEval<'ast>>,
}

/// Evaluated state for a single imported module. Populated when the import
/// is first encountered during lowering.
struct ImportEval<'ast> {
    /// Top-level let binding values from the imported file.
    values: FxHashMap<&'ast str, ValueId>,
    /// Function definitions from the imported file (name -> fn node in import AST).
    fns: FxHashMap<&'ast str, NodeId>,
    /// Evaluated state for this module's own imports, needed when calling
    /// the module's functions (they may reference the module's imports).
    sub_evals: Vec<ImportEval<'ast>>,
}

/// A `print(...)` statement captured during the item loop but rendered after
/// all rules are lowered, so name references can be expanded into their bodies.
struct PendingPrint {
    value: ValueId,
    span: Span,
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
pub fn lower_with_base(
    ast: &Ast,
    grammar_path: &Path,
    modules: &[super::Module],
) -> Result<InputGrammar, LowerError> {
    let base_grammar = modules.iter().find_map(|m| m.lowered.as_ref());
    let result = evaluate(ast, grammar_path, modules, base_grammar)?;
    build_grammar(result, base_grammar)
}

/// Evaluate the AST, producing intermediate results.
fn evaluate(
    ast: &Ast,
    grammar_path: &Path,
    modules: &[super::Module],
    base_grammar: Option<&InputGrammar>,
) -> Result<EvalResult, LowerError> {
    let mut eval = Evaluator::new(ast, base_grammar, modules);
    let mut rule_entries: Vec<(&str, RuleId)> = Vec::with_capacity(ast.root_items.len());
    let mut override_entries: Vec<(&str, RuleId, Span)> = Vec::new();
    let mut pending_prints: Vec<PendingPrint> = Vec::new();

    // Register all functions first so forward references work
    for &item_id in &ast.root_items {
        if let Node::Fn(fn_idx) = ast.node(item_id) {
            let config = ast.get_fn(*fn_idx);
            eval.fns.insert(ast.node_text(config.name), item_id);
        }
    }
    for &item_id in &ast.root_items {
        match ast.node(item_id) {
            Node::Grammar | Node::Fn(_) => {}
            Node::Let { name, value, .. } => {
                let val = eval.eval_expr(*value)?;
                eval.bind(ast.node_text(*name), val);
            }
            Node::Rule { name, body } => {
                let rule_id = eval.lower_to_rule(*body)?;
                rule_entries.push((ast.node_text(*name), rule_id));
            }
            Node::OverrideRule { name, body } => {
                let rule_id = eval.lower_to_rule(*body)?;
                override_entries.push((ast.node_text(*name), rule_id, ast.span(*name)));
            }
            Node::Print(arg) => {
                let value = eval.eval_expr(*arg)?;
                pending_prints.push(PendingPrint {
                    value,
                    span: ast.span(item_id),
                });
            }
            _ => unreachable!(),
        }
    }

    for p in &pending_prints {
        eval.emit_print(p.value, p.span, grammar_path, &rule_entries);
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
fn build_grammar(
    result: EvalResult,
    base: Option<&InputGrammar>,
) -> Result<InputGrammar, LowerError> {
    // Start with inherited rules (cloned), apply overrides, then append new rules.
    let mut variables = if let Some(base) = base {
        let mut vars = base.variables.clone();
        for (name, rule, span) in result.overrides {
            if let Some(var) = vars.iter_mut().find(|v| v.name == name) {
                var.rule = rule;
            } else {
                return Err(LowerError::new(
                    LowerErrorKind::OverrideRuleNotFound(name),
                    span,
                ));
            }
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
    Ok(InputGrammar {
        name: result.language,
        variables,
        extra_symbols: result
            .extras
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.extra_symbols.clone())),
        expected_conflicts: result
            .conflicts
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.expected_conflicts.clone())),
        precedence_orderings: result
            .precedences
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.precedence_orderings.clone())),
        external_tokens: result
            .externals
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.external_tokens.clone())),
        variables_to_inline: result
            .inline
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.variables_to_inline.clone())),
        supertype_symbols: result
            .supertypes
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.supertype_symbols.clone())),
        word_token: result
            .word
            .or_else(|| base.and_then(|b| b.word_token.clone())),
        reserved_words: result
            .reserved
            .unwrap_or_else(|| base.map_or_else(Vec::new, |b| b.reserved_words.clone())),
    })
}

impl<'ast> Evaluator<'ast> {
    fn new(
        ast: &'ast Ast,
        base_grammar: Option<&'ast InputGrammar>,
        modules: &'ast [super::Module],
    ) -> Self {
        let cap = ast.nodes.len();
        Self {
            ast,
            values: Vec::with_capacity(cap),
            scopes: ScopeStack::new(),
            fns: FxHashMap::default(),
            rules: Vec::with_capacity(cap),
            rule_children: Vec::with_capacity(cap),
            strings: StringPool::new(ast.source()),
            base_grammar,
            call_stack: Vec::new(),
            import_modules: modules,
            import_evals: Vec::new(),
        }
    }

    fn alloc_val(&mut self, val: Value<'ast>) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(val);
        id
    }

    fn get_val(&self, id: ValueId) -> &Value<'ast> {
        &self.values[id.0 as usize]
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

    #[expect(dead_code, reason = "API symmetry with other typed wrappers")]
    const fn expect_tuple(vid: ValueId) -> TupleVal {
        TupleVal(vid)
    }
    #[expect(dead_code, reason = "API symmetry with other typed wrappers")]
    fn tuple_items(&self, v: TupleVal) -> &[ValueId] {
        let Value::Tuple(items) = self.get_val(v.0) else {
            unreachable!()
        };
        items
    }

    #[inline]
    fn push_scope(&mut self) {
        self.scopes.push();
    }

    #[inline]
    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    #[inline]
    fn bind(&mut self, name: &'ast str, val: ValueId) {
        self.scopes.insert(name, val);
    }

    #[inline]
    fn lookup(&self, name: &str) -> Option<ValueId> {
        self.scopes.get(name)
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

    fn get_fn_config(&self, fn_id: NodeId) -> &'ast FnConfig {
        match self.ast.node(fn_id) {
            Node::Fn(fn_idx) => self.ast.get_fn(*fn_idx),
            _ => unreachable!(),
        }
    }

    fn intern_string_lit(&mut self, span: Span) -> Str {
        let raw = self.ast.text(span);
        if memchr::memchr(b'\\', raw.as_bytes()).is_some() {
            self.strings.intern_owned(unescape_string(raw))
        } else {
            self.strings.intern_span(span)
        }
    }

    fn intern_raw_string_lit(&mut self, span: Span, hash_count: u8) -> Str {
        let prefix = 2 + u32::from(hash_count);
        let suffix = 1 + u32::from(hash_count);
        self.strings
            .intern_span(Span::new(span.start + prefix, span.end - suffix))
    }

    fn get_str_val(&self, id: ValueId) -> Str {
        self.str_id(Self::expect_str(id))
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
                match &self.rules[id.0 as usize] {
                    ARule::Seq(..) => Rule::seq(children),
                    ARule::Choice(..) => Rule::choice(children),
                    _ => unreachable!(),
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

    fn value_to_owned_rule(&self, id: ValueId) -> Rule {
        match self.get_val(id) {
            Value::Rule(rid) => self.build_rule(*rid),
            Value::Str(sid) => Rule::String(self.strings.to_string(*sid)),
            // Guarded by super::typecheck - only rule-like values reach here
            _ => unreachable!(),
        }
    }

    fn eval_rule_list(&mut self, id: NodeId) -> Result<Vec<Rule>, LowerError> {
        let vid = self.eval_expr(id)?;
        let items = self.list_items(Self::expect_list(vid));
        Ok(items.iter().map(|&v| self.value_to_owned_rule(v)).collect())
    }

    fn eval_name_list(&mut self, id: NodeId) -> Result<Vec<String>, LowerError> {
        let vid = self.eval_expr(id)?;
        let items = self.list_items(Self::expect_list(vid));
        items
            .iter()
            .map(|&v| {
                let rid = self.rule_id(Self::expect_rule(v));
                match &self.rules[rid.0 as usize] {
                    ARule::NamedSymbol(sid) => Ok(self.strings.to_string(*sid)),
                    _ => unreachable!("expect_name_ref guarantees NamedSymbol"),
                }
            })
            .collect()
    }

    /// Evaluate a `conflicts` expression into `Vec<Vec<String>>`. Each inner
    /// element must evaluate to a named rule reference.
    fn eval_conflicts(&mut self, id: NodeId) -> Result<Vec<Vec<String>>, LowerError> {
        let vid = self.eval_expr(id)?;
        let outer = self.list_items(Self::expect_list(vid));
        outer
            .iter()
            .map(|&group_vid| {
                let inner = self.list_items(Self::expect_list(group_vid));
                inner
                    .iter()
                    .map(|&v| {
                        let rid = self.rule_id(Self::expect_rule(v));
                        match &self.rules[rid.0 as usize] {
                            ARule::NamedSymbol(sid) => Ok(self.strings.to_string(*sid)),
                            _ => unreachable!("expect_name_ref guarantees NamedSymbol"),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Evaluate a `precedences` expression into `Vec<Vec<PrecedenceEntry>>`.
    /// Each inner element is either a string literal (`PrecedenceEntry::Name`)
    /// or a named rule reference (`PrecedenceEntry::Symbol`).
    fn eval_precedences(&mut self, id: NodeId) -> Result<Vec<Vec<PrecedenceEntry>>, LowerError> {
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
                        Value::Rule(rid) => match &self.rules[rid.0 as usize] {
                            ARule::NamedSymbol(sid) => {
                                Ok(PrecedenceEntry::Symbol(self.strings.to_string(*sid)))
                            }
                            _ => unreachable!(),
                        },
                        _ => unreachable!(),
                    })
                    .collect()
            })
            .collect()
    }

    fn eval_rule_name(&mut self, id: NodeId) -> Result<String, LowerError> {
        let rid = self.lower_to_rule(id)?;
        match &self.rules[rid.0 as usize] {
            ARule::NamedSymbol(sid) => Ok(self.strings.to_string(*sid)),
            ARule::Blank => Err(LowerError::new(
                LowerErrorKind::ConfigFieldUnset,
                self.ast.span(id),
            )),
            _ => Err(LowerError::new(
                LowerErrorKind::ExpectedRuleName,
                self.ast.span(id),
            )),
        }
    }

    /// Evaluate the `reserved` config expression into `Vec<ReservedWordContext<Rule>>`.
    /// Accepts either an object literal `{ name: [words] }` or an expression
    /// evaluating to a list of `{name: str, words: list<rule>}` objects.
    fn eval_reserved(&mut self, id: NodeId) -> Result<Vec<ReservedWordContext<Rule>>, LowerError> {
        let ast = self.ast;
        // Object literal: each field is a named word set
        if let Node::Object(range) = ast.node(id) {
            return ast
                .get_object(*range)
                .iter()
                .map(|&(name_span, val_id)| {
                    let words = self.eval_rule_list(val_id)?;
                    Ok(ReservedWordContext {
                        name: ast.text(name_span).to_string(),
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
    fn import_config_field_or_err(
        &mut self,
        field: &str,
        span: Span,
    ) -> Result<ValueId, LowerError> {
        let grammar = self.base_grammar.unwrap();
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
                        let mut map = FxHashMap::default();
                        map.insert("name", name_val);
                        map.insert("words", words);
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
                let imported: Vec<RuleId> = members.iter().map(|m| self.import_rule(m)).collect();
                let start = self.rule_children.len() as u32;
                self.rule_children.extend_from_slice(&imported);
                let len = imported.len() as u16;
                match rule {
                    Rule::Seq(_) => self.alloc_rule(ARule::Seq(start, len)),
                    Rule::Choice(_) => self.alloc_rule(ARule::Choice(start, len)),
                    _ => unreachable!(),
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

impl<'ast> Evaluator<'ast> {
    fn eval_expr(&mut self, id: NodeId) -> Result<ValueId, LowerError> {
        let ast = self.ast;
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
                let sid = self.strings.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            // Guarded by super::resolve::resolve - all variable names validated
            Node::VarRef => Ok(self.lookup(ast.text(span)).unwrap()),
            Node::GrammarConfig(inner) => {
                let inner_val = self.eval_expr(*inner)?;
                let Value::Module(_) = *self.get_val(inner_val) else {
                    unreachable!() // guarded by typecheck
                };
                Ok(self.alloc_val(Value::GrammarConfig))
            }
            Node::Inherit { module, .. } | Node::Import { module, .. } => {
                let idx = module.expect("module index not set by loading pre-pass");
                self.eval_import_module(idx)?;
                Ok(self.alloc_val(Value::Module(idx)))
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
                let field_name = ast.node_text(field);
                match self.get_val(obj_val) {
                    Value::Object(map) => Ok(*map.get(field_name).unwrap()),
                    Value::GrammarConfig => {
                        self.import_config_field_or_err(field_name, ast.span(field))
                    }
                    _ => unreachable!(), // guarded by typecheck
                }
            }
            Node::QualifiedAccess { obj, member } => {
                let (obj, member) = (*obj, *member);
                let obj_val = self.eval_expr(obj)?;
                let member_name = ast.node_text(member);
                let Value::Module(idx) = *self.get_val(obj_val) else {
                    unreachable!() // guarded by typecheck
                };
                let idx = idx as usize;
                // Check pre-evaluated let values first
                if let Some(&val) = self.import_evals[idx].values.get(member_name) {
                    Ok(val)
                } else if let Some(grammar) = &self.base_grammar {
                    // For inherited grammar modules, fall back to rule bodies.
                    // Config fields are accessed via grammar_config() builtin.
                    match grammar.variables.iter().find(|v| v.name == member_name) {
                        Some(var) => {
                            let rid = self.import_rule(&var.rule);
                            Ok(self.alloc_val(Value::Rule(rid)))
                        }
                        None => Err(LowerError::new(
                            LowerErrorKind::ModuleMemberNotFound(member_name.to_string()),
                            ast.span(member),
                        )),
                    }
                } else {
                    Err(LowerError::new(
                        LowerErrorKind::ModuleMemberNotFound(member_name.to_string()),
                        ast.span(member),
                    ))
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
                self.eval_fn_call(ast.node_text(name), &arg_vals, span)
            }
            Node::QualifiedCall(range) => {
                let (obj, name, args) = ast.get_qualified_call(*range);
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
                self.eval_import_fn_call(idx, fn_name, &arg_vals, span)
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
                let sid = self.intern_string_lit(ast.span(id));
                Ok(self.alloc_rule(ARule::String(sid)))
            }
            Node::RawStringLit { hash_count } => {
                let sid = self.intern_raw_string_lit(ast.span(id), *hash_count);
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

    fn eval_combinator(&mut self, id: NodeId) -> Result<RuleId, LowerError> {
        let ast = self.ast;
        match ast.node(id) {
            Node::Seq(range) | Node::Choice(range) => {
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
                Ok(self.alloc_rule(match ast.node(id) {
                    Node::Seq(_) => ARule::Seq(offset, len),
                    Node::Choice(_) => ARule::Choice(offset, len),
                    _ => unreachable!(),
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
                                // Guarded by super::typecheck::type_of (Node::Alias) -
                                // InvalidAliasTarget rejects non-name/non-str targets
                                let ARule::NamedSymbol(s) = &self.rules[rid.0 as usize] else {
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
                let n = self.int_val(Self::expect_int(vid));
                let inner = self.lower_to_rule(*content)?;
                Ok(self.alloc_rule(ARule::PrecDynamic(n, inner)))
            }
            Node::Reserved { context, content } => {
                let inner = self.lower_to_rule(*content)?;
                let sid = self.strings.intern_span(ast.span(*context));
                Ok(self.alloc_rule(ARule::Reserved(sid, inner)))
            }
            // Guarded by lower_to_rule - only combinator nodes are dispatched here
            _ => unreachable!(),
        }
    }

    fn eval_fn_call(
        &mut self,
        name: &'ast str,
        arg_vals: &[ValueId],
        call_span: Span,
    ) -> Result<ValueId, LowerError> {
        self.call_stack.push((name, call_span));
        if self.call_stack.len() > MAX_CALL_DEPTH as usize {
            let trace = self
                .call_stack
                .iter()
                .map(|(name, span)| (name.to_string(), *span))
                .collect();
            return Err(LowerError::new(
                LowerErrorKind::CallDepthExceeded(trace),
                call_span,
            ));
        }
        // Guarded by super::typecheck::check_call_args - validates function exists
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
        self.call_stack.pop();
        result
    }

    /// Evaluate an imported module's top-level let bindings and register its fns.
    /// Called when `let h = import(...)` is first evaluated.
    fn eval_import_module(&mut self, idx: u8) -> Result<(), LowerError> {
        let idx = idx as usize;
        if idx < self.import_evals.len() {
            return Ok(()); // already evaluated
        }

        let module = &self.import_modules[idx];
        let import_ast = &module.ast;

        // Collect fns from the imported file
        let mut fns = FxHashMap::default();
        for &item_id in &import_ast.root_items {
            if let Node::Fn(fn_idx) = import_ast.node(item_id) {
                let config = import_ast.get_fn(*fn_idx);
                fns.insert(import_ast.node_text(config.name), item_id);
            }
        }

        // Swap to the import's context
        let saved_ast = self.ast;
        let saved_fns = std::mem::take(&mut self.fns);
        let saved_source = self.strings.source;
        let saved_modules = self.import_modules;
        let saved_evals = std::mem::take(&mut self.import_evals);
        let saved_base = self.base_grammar;
        self.ast = import_ast;
        self.fns = fns.clone();
        self.strings.source = import_ast.source();
        self.import_modules = &module.sub_modules;
        self.base_grammar = module.lowered.as_ref();

        self.push_scope();
        let mut values = FxHashMap::default();
        for &item_id in &import_ast.root_items {
            if let Node::Let { name, value, .. } = import_ast.node(item_id) {
                let val = self.eval_expr(*value)?;
                let var_name = import_ast.node_text(*name);
                self.bind(var_name, val);
                values.insert(var_name, val);
            }
        }
        self.pop_scope();

        // Save child's import evals before restoring parent state
        let sub_evals = std::mem::replace(&mut self.import_evals, saved_evals);

        // Restore parent context
        self.fns = saved_fns;
        self.ast = saved_ast;
        self.strings.source = saved_source;
        self.import_modules = saved_modules;
        self.base_grammar = saved_base;

        // Pad import_evals if needed (for ordered evaluation)
        while self.import_evals.len() < idx {
            self.import_evals.push(ImportEval {
                values: FxHashMap::default(),
                fns: FxHashMap::default(),
                sub_evals: Vec::new(),
            });
        }
        self.import_evals.push(ImportEval {
            values,
            fns,
            sub_evals,
        });
        Ok(())
    }

    /// Call a function defined in an imported module.
    fn eval_import_fn_call(
        &mut self,
        idx: u8,
        fn_name: &'ast str,
        arg_vals: &[ValueId],
        call_span: Span,
    ) -> Result<ValueId, LowerError> {
        let idx = idx as usize;
        let module = &self.import_modules[idx];
        let import_ast = &module.ast;

        // Extract everything we need from import_evals before mutating self
        let fn_node_id = self.import_evals[idx].fns[fn_name];
        let fn_idx = match import_ast.node(fn_node_id) {
            Node::Fn(fi) => *fi,
            _ => unreachable!(),
        };
        let config = import_ast.get_fn(fn_idx);
        let body = config.body;
        let params: Vec<_> = config.params.iter().map(|p| p.name).collect();
        let import_fns = self.import_evals[idx].fns.clone();
        let import_values: Vec<_> = self.import_evals[idx]
            .values
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect();

        // Swap to the import's context
        let child_evals = std::mem::take(&mut self.import_evals[idx].sub_evals);
        let saved_ast = self.ast;
        let saved_fns = std::mem::take(&mut self.fns);
        let saved_source = self.strings.source;
        let saved_modules = self.import_modules;
        let saved_evals = std::mem::replace(&mut self.import_evals, child_evals);
        let saved_base = self.base_grammar;
        self.ast = import_ast;
        self.fns = import_fns;
        self.strings.source = import_ast.source();
        self.import_modules = &module.sub_modules;
        self.base_grammar = module.lowered.as_ref();

        self.push_scope();
        // Bind the import's let values so the function body can reference them
        for (var_name, val) in &import_values {
            self.bind(var_name, *val);
        }
        // Bind arguments
        for (i, &arg_val) in arg_vals.iter().enumerate() {
            self.bind(import_ast.node_text(params[i]), arg_val);
        }

        self.call_stack.push((fn_name, call_span));
        let result = self.eval_expr(body);
        self.call_stack.pop();

        self.pop_scope();

        // Restore parent context, preserving any child eval changes
        let child_evals = std::mem::replace(&mut self.import_evals, saved_evals);
        self.import_evals[idx].sub_evals = child_evals;
        self.fns = saved_fns;
        self.ast = saved_ast;
        self.strings.source = saved_source;
        self.import_modules = saved_modules;
        self.base_grammar = saved_base;

        result
    }

    fn eval_for_to_rules(&mut self, config: &ForConfig) -> Result<Vec<RuleId>, LowerError> {
        let iter_vid = self.eval_expr(config.iterable)?;
        let n_items = self.list_items(Self::expect_list(iter_vid)).len();
        let mut results = Vec::with_capacity(n_items);
        for i in 0..n_items {
            let item_id = self.list_items(Self::expect_list(iter_vid))[i];
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

// -- `print(...)` rendering ----------------------------------------------
//
// Debug output for the top-level `print` item. Write to stderr with a line
// prefix, swallow any IO errors. Compound values (lists, objects, and rule
// combinators with children) break their direct children onto indented lines;
// each child renders on a single line. Rules are fully traversed.

impl Evaluator<'_> {
    /// Format `val_id` and write it to stderr, prefixed with `path:line:`. IO
    /// errors are ignored. If the value is a bare `NamedSymbol` (e.g. from
    /// `print(my_rule)`), the rule body is expanded once so the user sees
    /// what the rule lowered to; nested rule references remain as names.
    fn emit_print(
        &self,
        val_id: ValueId,
        item_span: Span,
        grammar_path: &Path,
        rule_entries: &[(&str, RuleId)],
    ) {
        use std::io::Write as _;
        let line = offset_to_line(self.ast.source(), item_span.start as usize);
        let mut buf = String::new();
        // Top-level NamedSymbol expansion: look up the rule body and render it.
        if let Value::Rule(rid) = self.get_val(val_id)
            && let ARule::NamedSymbol(sid) = &self.rules[rid.0 as usize]
            && let Some((_, body_rid)) = {
                let name = self.strings.resolve(*sid);
                rule_entries.iter().find(|(n, _)| *n == name).copied()
            }
        {
            self.write_rule(&mut buf, body_rid, 0);
        } else {
            self.write_value(&mut buf, val_id, 0);
        }
        let stderr = std::io::stderr();
        let mut stderr = stderr.lock();
        let _ = writeln!(stderr, "{}:{line}: {buf}", grammar_path.display());
    }

    /// Write a value in compact form. Compound values at depth 0 break their
    /// direct children onto their own lines (one-level destructuring); nested
    /// compounds render flat.
    fn write_value(&self, w: &mut String, val_id: ValueId, depth: u32) {
        match self.get_val(val_id) {
            Value::Int(n) => {
                use std::fmt::Write as _;
                let _ = write!(w, "{n}");
            }
            Value::Str(sid) => {
                w.push('"');
                w.push_str(self.strings.resolve(*sid));
                w.push('"');
            }
            Value::Rule(rid) => self.write_rule(w, *rid, depth),
            Value::List(items) => self.write_compound(w, '[', ']', items, depth, |this, w, vid| {
                this.write_value(w, vid, depth + 1);
            }),
            Value::Tuple(items) => {
                self.write_compound(w, '(', ')', items, depth, |this, w, vid| {
                    this.write_value(w, vid, depth + 1);
                });
            }
            Value::Object(fields) => {
                // Render in source-insertion order; FxHashMap iteration is
                // undefined, so sort for deterministic output.
                let mut pairs: Vec<_> = fields.iter().collect();
                pairs.sort_by_key(|(k, _)| *k);
                self.write_compound(w, '{', '}', &pairs, depth, |this, w, (key, vid)| {
                    w.push_str(key);
                    w.push_str(": ");
                    this.write_value(w, *vid, depth + 1);
                });
            }
            Value::Module(idx) => {
                w.push_str("<module:");
                w.push_str(
                    &self.import_modules[*idx as usize]
                        .path
                        .display()
                        .to_string(),
                );
                w.push('>');
            }
            Value::GrammarConfig => w.push_str("<grammar_config>"),
        }
    }

    /// Write a rule fully traversed. At `depth == 0`, combinators with
    /// multiple children (seq/choice) break their children onto indented
    /// lines; otherwise everything is flat. `NamedSymbol`s render as just the
    /// name (no expansion; top-level expansion is handled by `emit_print`).
    fn write_rule(&self, w: &mut String, rid: RuleId, depth: u32) {
        match &self.rules[rid.0 as usize] {
            ARule::Blank => w.push_str("blank()"),
            ARule::String(s) => {
                w.push('"');
                w.push_str(self.strings.resolve(*s));
                w.push('"');
            }
            ARule::Pattern(p, flags) => {
                w.push_str("regexp(\"");
                w.push_str(self.strings.resolve(*p));
                w.push('"');
                if let Some(f) = flags {
                    w.push_str(", \"");
                    w.push_str(self.strings.resolve(*f));
                    w.push('"');
                }
                w.push(')');
            }
            ARule::NamedSymbol(s) => w.push_str(self.strings.resolve(*s)),
            ARule::Seq(offset, len) | ARule::Choice(offset, len) => {
                let name = match &self.rules[rid.0 as usize] {
                    ARule::Seq(..) => "seq",
                    ARule::Choice(..) => "choice",
                    _ => unreachable!(),
                };
                let start = *offset as usize;
                let end = start + *len as usize;
                let children: Vec<RuleId> = self.rule_children[start..end].to_vec();
                w.push_str(name);
                self.write_compound(w, '(', ')', &children, depth, |this, w, cid| {
                    this.write_rule(w, cid, depth + 1);
                });
            }
            ARule::Repeat(inner) => {
                w.push_str("repeat1(");
                self.write_rule(w, *inner, depth + 1);
                w.push(')');
            }
            ARule::Token(inner) => {
                w.push_str("token(");
                self.write_rule(w, *inner, depth + 1);
                w.push(')');
            }
            ARule::ImmediateToken(inner) => {
                w.push_str("token_immediate(");
                self.write_rule(w, *inner, depth + 1);
                w.push(')');
            }
            ARule::Prec(v, inner) => self.write_prec(w, "prec", *v, *inner, depth),
            ARule::PrecLeft(v, inner) => self.write_prec(w, "prec_left", *v, *inner, depth),
            ARule::PrecRight(v, inner) => self.write_prec(w, "prec_right", *v, *inner, depth),
            ARule::PrecDynamic(n, inner) => {
                use std::fmt::Write as _;
                let _ = write!(w, "prec_dynamic({n}, ");
                self.write_rule(w, *inner, depth + 1);
                w.push(')');
            }
            ARule::Field(name, inner) => {
                w.push_str("field(");
                w.push_str(self.strings.resolve(*name));
                w.push_str(", ");
                self.write_rule(w, *inner, depth + 1);
                w.push(')');
            }
            ARule::Alias(name, named, inner) => {
                w.push_str("alias(");
                self.write_rule(w, *inner, depth + 1);
                w.push_str(", ");
                if *named {
                    w.push_str(self.strings.resolve(*name));
                } else {
                    w.push('"');
                    w.push_str(self.strings.resolve(*name));
                    w.push('"');
                }
                w.push(')');
            }
            ARule::Reserved(ctx, inner) => {
                w.push_str("reserved(");
                w.push_str(self.strings.resolve(*ctx));
                w.push_str(", ");
                self.write_rule(w, *inner, depth + 1);
                w.push(')');
            }
        }
    }

    fn write_prec(&self, w: &mut String, name: &str, v: APrec, inner: RuleId, depth: u32) {
        w.push_str(name);
        w.push('(');
        match v {
            APrec::Integer(n) => {
                use std::fmt::Write as _;
                let _ = write!(w, "{n}");
            }
            APrec::Name(s) => {
                w.push('"');
                w.push_str(self.strings.resolve(s));
                w.push('"');
            }
        }
        w.push_str(", ");
        self.write_rule(w, inner, depth + 1);
        w.push(')');
    }

    /// Write a compound `(open ... close)` form. At `depth == 0` with 2+
    /// children, breaks each child onto its own indented line; otherwise
    /// flat comma-space separated.
    fn write_compound<T, F: Fn(&Self, &mut String, T)>(
        &self,
        w: &mut String,
        open: char,
        close: char,
        children: &[T],
        depth: u32,
        write_child: F,
    ) where
        T: Copy,
    {
        w.push(open);
        if depth == 0 && children.len() > 1 {
            w.push('\n');
            for (i, &child) in children.iter().enumerate() {
                w.push_str("  ");
                write_child(self, w, child);
                if i + 1 < children.len() {
                    w.push(',');
                }
                w.push('\n');
            }
        } else {
            for (i, &child) in children.iter().enumerate() {
                if i > 0 {
                    w.push_str(", ");
                }
                write_child(self, w, child);
            }
        }
        w.push(close);
    }
}

/// Count newlines in `source[..offset]` and return a 1-based line number.
fn offset_to_line(source: &str, offset: usize) -> usize {
    1 + memchr::memchr_iter(b'\n', &source.as_bytes()[..offset.min(source.len())]).count()
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

#[derive(Debug, Serialize, Error)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    pub span: Span,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub note: Option<Box<Note>>,
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
    ModuleResolveFailed {
        path: String,
        error: String,
    },
    ModuleReadFailed {
        path: String,
        error: String,
    },
    ModuleJsonError {
        path: String,
        error: crate::parse_grammar::ParseGrammarError,
    },
    ModuleUnsupportedExtension,
    ModuleTooMany,
    ModuleDisallowedItem,
    ModuleContainsGrammarBlock,
    ModuleCycle(Vec<std::path::PathBuf>),
    ModuleMemberNotFound(String),
    ExpectedRuleName,
    ConfigFieldUnset,
    MultipleInherits,
    InheritWithoutConfig,
    InheritsWithoutInherit,
    TooManyChildren,
    CallDepthExceeded(Vec<(String, Span)>),
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
            LowerErrorKind::ModuleResolveFailed { path, error } => {
                write!(f, "failed to resolve '{path}': {error}")
            }
            LowerErrorKind::ModuleReadFailed { path, error } => {
                write!(f, "failed to read '{path}': {error}")
            }
            LowerErrorKind::ModuleJsonError { path, error } => {
                write!(f, "failed to parse '{path}': {error}")
            }
            LowerErrorKind::ModuleUnsupportedExtension => {
                write!(f, "unsupported file extension, expected .tsg or .json")
            }
            LowerErrorKind::ModuleTooMany => {
                write!(f, "too many modules (max 256)")
            }
            LowerErrorKind::ModuleDisallowedItem => {
                write!(
                    f,
                    "imported files can only contain let, fn, and print items"
                )
            }
            LowerErrorKind::ModuleContainsGrammarBlock => {
                write!(f, "imported files cannot contain a grammar block")
            }
            LowerErrorKind::ModuleCycle(chain) => {
                write!(f, "module cycle detected: ")?;
                for (i, p) in chain.iter().enumerate() {
                    if i > 0 {
                        write!(f, " -> ")?;
                    }
                    write!(f, "{}", p.display())?;
                }
                Ok(())
            }
            LowerErrorKind::ModuleMemberNotFound(n) => {
                write!(f, "member '{n}' not found in module")
            }
            LowerErrorKind::ExpectedRuleName => {
                write!(f, "expected a rule name reference")
            }
            LowerErrorKind::ConfigFieldUnset => {
                write!(f, "config field is not set in the base grammar")
            }
            LowerErrorKind::MultipleInherits => {
                write!(f, "only one inherit() call is allowed per grammar")
            }
            LowerErrorKind::InheritWithoutConfig => {
                write!(
                    f,
                    "inherit() requires 'inherits' to be set in the grammar config"
                )
            }
            LowerErrorKind::InheritsWithoutInherit => {
                write!(f, "'inherits' must reference a variable bound to inherit()")
            }
            LowerErrorKind::TooManyChildren => {
                write!(f, "too many elements (maximum {})", u16::MAX)
            }
            LowerErrorKind::CallDepthExceeded(_) => {
                write!(f, "maximum function call depth ({MAX_CALL_DEPTH}) exceeded")
            }
        }
    }
}
