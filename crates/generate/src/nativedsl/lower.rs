//! Lowering pass: evaluates the typed AST into an [`InputGrammar`].
//!
//! This is the final pipeline stage. It walks the AST, evaluating expressions
//! into an intermediate arena representation, then converts the arena into
//! the output [`InputGrammar`] / [`Rule`] types.
//!
//! ## Key data structures
//!
//! - **[`StringPool`]** - interned strings using `NonZeroU32` handles. Strings
//!   that come directly from source are stored as spans (zero-copy), while
//!   computed strings (escapes, concat) are owned.
//! - **Rule arena** (`Vec<ARule>` + `Vec<RuleId>` for children) - intermediate
//!   rule representation using compact arena IDs instead of heap-allocated
//!   `Box<Rule>`. Variadic nodes (`Seq`/`Choice`) use `(offset, len)` into a
//!   shared children vector.
//! - **Value arena** (`Vec<Value>`) - evaluated expression results. All values
//!   are allocated in a flat vector and referenced by [`ValueId`].
//! - **[`Evaluator`]** - ties everything together with scope-chain-based
//!   variable lookup and function call evaluation.
//!
//! ## JSON serialization
//!
//! [`grammar_to_json`] and [`rule_to_json`] convert the output `InputGrammar`
//! into the same JSON format as `grammar.json`, for compatibility with the
//! existing parser generator pipeline.

use std::num::NonZeroU32;

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use crate::grammars::{InputGrammar, PrecedenceEntry, ReservedWordContext, Variable, VariableType};
use crate::rules::{Precedence, Rule};

use super::ast::{Ast, FnConfig, ForConfig, Node, NodeId, PrecEntry, Span};

/// A handle to an interned string in the [`StringPool`].
///
/// Uses `NonZeroU32` so that `Option<Str>` is 4 bytes (niche optimization).
/// Index 0 is a poison entry that is never handed out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Str(NonZeroU32);

/// Storage for a single interned string.
enum StrEntry {
    /// Poison entry at index 0 so `Str` can use `NonZeroU32`.
    Unreachable,
    /// Zero-copy reference into source text.
    Source(Span),
    /// Owned string for computed values (escape processing, concat).
    Owned(String),
}

/// Pool of interned strings, supporting both zero-copy source references and
/// owned computed strings.
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

    /// Intern a span from the source.
    fn intern_span(&mut self, span: Span) -> Str {
        let id = Str(NonZeroU32::new(self.entries.len() as u32).unwrap());
        self.entries.push(StrEntry::Source(span));
        id
    }

    /// Intern an owned string.
    fn intern_owned(&mut self, s: String) -> Str {
        let id = Str(NonZeroU32::new(self.entries.len() as u32).unwrap());
        self.entries.push(StrEntry::Owned(s));
        id
    }

    /// Resolve a Str handle to a &str.
    fn resolve(&self, id: Str) -> &str {
        match &self.entries[id.0.get() as usize] {
            StrEntry::Source(span) => span.resolve(self.source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!("poison string entry"),
        }
    }

    /// Resolve to an owned String (for output).
    fn to_string(&self, id: Str) -> String {
        self.resolve(id).to_string()
    }
}

/// Typed index into the rule arena (`Evaluator::rules`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RuleId(u32);

/// A precedence value in the arena representation.
#[derive(Clone, Copy, Debug)]
enum APrec {
    Integer(i32),
    Name(Str),
}

/// Arena-based rule representation.
///
/// Each variant mirrors a tree-sitter grammar rule type but uses arena IDs
/// (`RuleId`, `Str`) instead of heap pointers. Variadic nodes (`Seq`,
/// `Choice`) store `(start_offset, len)` into `Evaluator::rule_children`.
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

/// Typed index into the value arena (`Evaluator::values`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ValueId(u32);

/// An evaluated expression result.
///
/// All values are allocated in the evaluator's flat value arena. Rule values
/// wrap a [`RuleId`] pointing into the separate rule arena.
enum Value<'src> {
    Int(i32),
    Str(Str),
    Rule(RuleId),
    Object(FxHashMap<&'src str, ValueId>),
    List(Vec<ValueId>),
    Tuple(Vec<ValueId>),
}

/// The evaluator: walks the AST and builds values and rules in arenas.
///
/// Combines a value arena, rule arena, string pool, scope chain, and
/// function registry. After evaluation, arena rules are converted to output
/// `Rule` trees via [`Evaluator::build_rule`].
struct Evaluator<'src> {
    ast: &'src Ast<'src>,
    // Value arena
    values: Vec<Value<'src>>,
    /// Variable scope stack - each level maps names to value IDs.
    scopes: Vec<FxHashMap<&'src str, ValueId>>,
    /// Function name -> AST node ID of the `Node::Fn` item.
    fns: FxHashMap<&'src str, NodeId>,
    // Rule arena
    rules: Vec<ARule>,
    /// Shared child list for variadic rules (`Seq`, `Choice`).
    rule_children: Vec<RuleId>,
    // Unified string pool
    strings: StringPool<'src>,
}

impl<'src> Evaluator<'src> {
    fn new(ast: &'src Ast<'src>) -> Self {
        Self {
            ast,
            values: Vec::new(),
            scopes: vec![FxHashMap::default()],
            fns: FxHashMap::default(),
            rules: Vec::new(),
            rule_children: Vec::new(),
            strings: StringPool::new(ast.source()),
        }
    }

    // -- Value arena --
    fn alloc_val(&mut self, val: Value<'src>) -> ValueId {
        let id = ValueId(self.values.len() as u32);
        self.values.push(val);
        id
    }
    fn get_val(&self, id: ValueId) -> &Value<'src> {
        &self.values[id.0 as usize]
    }

    // -- Scopes --
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
        for scope in self.scopes.iter().rev() {
            if let Some(&id) = scope.get(name) {
                return Some(id);
            }
        }
        None
    }

    // -- Rule arena --
    fn alloc_rule(&mut self, rule: ARule) -> RuleId {
        let id = RuleId(self.rules.len() as u32);
        self.rules.push(rule);
        id
    }
    const fn children_start(&self) -> u32 {
        self.rule_children.len() as u32
    }
    fn push_child(&mut self, id: RuleId) {
        self.rule_children.push(id);
    }
    const fn children_range(&self, start: u32) -> (u32, u16) {
        (start, (self.rule_children.len() as u32 - start) as u16)
    }

    fn get_fn_config(&self, fn_id: NodeId) -> &'src FnConfig {
        match self.ast.node(fn_id) {
            Node::Fn(fn_idx) => self.ast.get_fn(*fn_idx),
            _ => unreachable!(),
        }
    }

    /// Extract the Str from a `Value::Str`, copying the Str handle.
    fn get_str_val(&self, id: ValueId) -> Str {
        match self.get_val(id) {
            Value::Str(s) => *s,
            _ => unreachable!("type checker ensures string value"),
        }
    }

    /// Convert an arena rule to the output `Rule` tree.
    ///
    /// Recursively expands arena IDs into heap-allocated `Rule` variants,
    /// resolving all `Str` handles through the string pool.
    fn build_rule(&self, id: RuleId) -> Rule {
        let str_pool = &self.strings;
        match &self.rules[id.0 as usize] {
            ARule::Blank => Rule::Blank,
            ARule::String(sid) => Rule::String(str_pool.to_string(*sid)),
            ARule::Pattern(pattern, flags) => Rule::Pattern(
                str_pool.to_string(*pattern),
                flags.map_or_else(String::new, |flags| str_pool.to_string(flags)),
            ),
            ARule::NamedSymbol(sid) => Rule::NamedSymbol(str_pool.to_string(*sid)),
            ARule::Seq(start, len) => {
                let range = *start as usize..(*start as usize + *len as usize);
                Rule::seq(
                    self.rule_children[range]
                        .iter()
                        .map(|&id| self.build_rule(id))
                        .collect(),
                )
            }
            ARule::Choice(start, len) => {
                let range = *start as usize..(*start as usize + *len as usize);
                Rule::choice(
                    self.rule_children[range]
                        .iter()
                        .map(|&id| self.build_rule(id))
                        .collect(),
                )
            }
            ARule::Repeat(inner) => Rule::repeat(self.build_rule(*inner)),
            ARule::Prec(prec, inner) => Rule::prec(self.build_prec(*prec), self.build_rule(*inner)),
            ARule::PrecLeft(prec, inner) => {
                Rule::prec_left(self.build_prec(*prec), self.build_rule(*inner))
            }
            ARule::PrecRight(prec, inner) => {
                Rule::prec_right(self.build_prec(*prec), self.build_rule(*inner))
            }
            ARule::PrecDynamic(prec_value, inner) => {
                Rule::prec_dynamic(*prec_value, self.build_rule(*inner))
            }
            ARule::Field(name, inner) => {
                Rule::field(str_pool.to_string(*name), self.build_rule(*inner))
            }
            ARule::Alias(value, is_named, inner) => Rule::alias(
                self.build_rule(*inner),
                str_pool.to_string(*value),
                *is_named,
            ),
            ARule::Token(inner) => Rule::token(self.build_rule(*inner)),
            ARule::ImmediateToken(inner) => Rule::immediate_token(self.build_rule(*inner)),
            ARule::Reserved(ctx, inner) => Rule::Reserved {
                rule: Box::new(self.build_rule(*inner)),
                context_name: str_pool.to_string(*ctx),
            },
        }
    }

    /// Convert an arena precedence to the output `Precedence`.
    fn build_prec(&self, prec: APrec) -> Precedence {
        match prec {
            APrec::Integer(n) => Precedence::Integer(n),
            APrec::Name(sid) => Precedence::Name(self.strings.to_string(sid)),
        }
    }
}

/// Lower a fully resolved and type-checked AST into an [`InputGrammar`].
///
/// Evaluates all expressions, expands function calls, and assembles the
/// grammar configuration, rule definitions, and metadata into the output
/// format consumed by the tree-sitter parser generator.
///
/// # Errors
///
/// Returns [`LowerError`] if the grammar block is missing, a required field
/// is absent, or an expression cannot be used as a grammar rule.
pub fn lower(ast: &Ast<'_>) -> Result<InputGrammar, LowerError> {
    let mut eval = Evaluator::new(ast);
    let mut rule_entries: Vec<(String, RuleId)> = Vec::new();

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
                let rule_name = ast.node_text(*name).to_string();
                let rule_id = eval.lower_to_rule(*body)?;
                rule_entries.push((rule_name, rule_id));
            }
            _ => unreachable!(),
        }
    }

    let config = ast
        .context
        .grammar_config
        .as_ref()
        .ok_or_else(|| LowerError {
            kind: LowerErrorKind::MissingGrammarBlock,
            span: Span::new(0, 0),
        })?;
    let language = config.language.as_ref().ok_or_else(|| LowerError {
        kind: LowerErrorKind::MissingLanguageField,
        span: Span::new(0, 0),
    })?;

    let extra_symbols = config
        .extras
        .iter()
        .map(|&id| eval.lower_to_rule(id).map(|rid| eval.build_rule(rid)))
        .collect::<Result<_, _>>()?;
    let external_tokens = config
        .externals
        .iter()
        .map(|&id| eval.lower_to_rule(id).map(|rid| eval.build_rule(rid)))
        .collect::<Result<_, _>>()?;
    let reserved_words = config
        .reserved
        .iter()
        .map(|rws| {
            let words = rws
                .words
                .iter()
                .map(|&id| eval.lower_to_rule(id).map(|rid| eval.build_rule(rid)))
                .collect::<Result<_, _>>()?;
            Ok(ReservedWordContext {
                name: ast.text(rws.name).to_string(),
                reserved_words: words,
            })
        })
        .collect::<Result<_, LowerError>>()?;

    let variables = rule_entries
        .into_iter()
        .map(|(name, rid)| Variable {
            name,
            kind: VariableType::Named,
            rule: eval.build_rule(rid),
        })
        .collect();

    Ok(InputGrammar {
        name: language.clone(),
        variables,
        extra_symbols,
        expected_conflicts: config
            .conflicts
            .iter()
            .map(|g| g.iter().map(|s| ast.text(*s).to_string()).collect())
            .collect(),
        precedence_orderings: config
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
            .collect(),
        external_tokens,
        variables_to_inline: config
            .inline
            .iter()
            .map(|s| ast.text(*s).to_string())
            .collect(),
        supertype_symbols: config
            .supertypes
            .iter()
            .map(|s| ast.text(*s).to_string())
            .collect(),
        word_token: config.word.map(|s| ast.text(s).to_string()),
        reserved_words,
    })
}

impl Evaluator<'_> {
    /// Evaluate an AST expression into a value.
    ///
    /// Literals become `Value::Int`/`Value::Str`, identifiers are looked up
    /// in the scope chain, combinators are delegated to `eval_combinator`,
    /// and function calls are expanded inline.
    fn eval_expr(&mut self, id: NodeId) -> Result<ValueId, LowerError> {
        let ast = self.ast;
        let span = ast.span(id);
        match ast.node(id) {
            Node::IntLit(n) => Ok(self.alloc_val(Value::Int(*n))),
            Node::StringLit => {
                let content_span = Span::new(span.start + 1, span.end - 1);
                let raw = ast.text(content_span);
                if raw.contains('\\') {
                    let sid = self.strings.intern_owned(unescape_string(raw));
                    Ok(self.alloc_val(Value::Str(sid)))
                } else {
                    let sid = self.strings.intern_span(content_span);
                    Ok(self.alloc_val(Value::Str(sid)))
                }
            }
            Node::RawStringLit { hash_count } => {
                let hash_count = *hash_count;
                let prefix_len = 2 + u32::from(hash_count); // r + hashes + "
                let suffix_len = 1 + u32::from(hash_count); // " + hashes
                let content_span = Span::new(span.start + prefix_len, span.end - suffix_len);
                let sid = self.strings.intern_span(content_span);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::Neg(inner) => {
                let inner = *inner;
                let vid = self.eval_expr(inner)?;
                // Typecheck ensures this is an int
                let Value::Int(n) = self.get_val(vid) else {
                    unreachable!();
                };
                Ok(self.alloc_val(Value::Int(-*n)))
            }
            Node::Ident => {
                unreachable!("resolve pass should have replaced all Ident nodes")
            }
            Node::RuleRef => {
                let sid = self.strings.intern_span(span);
                let rid = self.alloc_rule(ARule::NamedSymbol(sid));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::VarRef => {
                let name = ast.text(span);
                Ok(self.lookup(name).expect("resolve pass validated this"))
            }
            Node::FieldAccess { obj, field } => {
                let (obj, field) = (*obj, *field);
                let obj_id = self.eval_expr(obj)?;
                let field_name = ast.node_text(field);
                // Typecheck ensures obj is an object with the right fields
                let Value::Object(map) = self.get_val(obj_id) else {
                    unreachable!();
                };
                Ok(*map.get(field_name).unwrap())
            }
            Node::Object(range) => {
                let fields = ast.get_object(*range);
                let mut map = FxHashMap::default();
                for i in 0..fields.len() {
                    let (key_span, value_id) = ast.get_object(*range)[i];
                    let val = self.eval_expr(value_id)?;
                    map.insert(ast.text(key_span), val);
                }
                Ok(self.alloc_val(Value::Object(map)))
            }
            Node::List(range) | Node::Tuple(range) => {
                let make = match ast.node(id) {
                    Node::List(_) => Value::List,
                    Node::Tuple(_) => Value::Tuple,
                    _ => unreachable!(),
                };
                let items = ast.child_slice(*range);
                let mut vals = Vec::with_capacity(items.len());
                for &item_id in items {
                    vals.push(self.eval_expr(item_id)?);
                }
                Ok(self.alloc_val(make(vals)))
            }
            Node::Concat(range) => {
                let parts = ast.child_slice(*range);
                let mut result = String::new();
                for &part_id in parts {
                    let vid = self.eval_expr(part_id)?;
                    let s = self.get_str_val(vid);
                    result.push_str(self.strings.resolve(s));
                }
                let sid = self.strings.intern_owned(result);
                Ok(self.alloc_val(Value::Str(sid)))
            }
            Node::DynRegex { pattern, flags } => {
                let (pattern, flags) = (*pattern, *flags);
                let pattern_val = self.eval_expr(pattern)?;
                let pattern_str = self.get_str_val(pattern_val);
                let flags_str = match flags {
                    Some(flags_id) => {
                        let flags_val = self.eval_expr(flags_id)?;
                        Some(self.get_str_val(flags_val))
                    }
                    None => None,
                };
                let rid = self.alloc_rule(ARule::Pattern(pattern_str, flags_str));
                Ok(self.alloc_val(Value::Rule(rid)))
            }
            Node::Call { name, args } => {
                let (name, args) = (*name, *args);
                let arg_ids = ast.child_slice(args);
                let mut arg_vals = Vec::with_capacity(arg_ids.len());
                for &arg_id in arg_ids {
                    arg_vals.push(self.eval_expr(arg_id)?);
                }
                self.eval_fn_call_with_vals(ast.node_text(name), &arg_vals)
            }
            _ => {
                let rule_id = self.eval_combinator(id)?;
                Ok(self.alloc_val(Value::Rule(rule_id)))
            }
        }
    }

    /// Evaluate an expression and convert the result to an arena rule.
    ///
    /// This is the main entry point for lowering a rule body or combinator
    /// argument. Evaluates the expression, then converts the resulting value
    /// to a `RuleId` (strings become `ARule::String`).
    fn lower_to_rule(&mut self, id: NodeId) -> Result<RuleId, LowerError> {
        let vid = self.eval_expr(id)?;
        Ok(self.value_to_rule(vid))
    }

    /// Convert an evaluated value to an arena rule ID.
    ///
    /// `Value::Rule` passes through directly, `Value::Str` becomes
    /// `ARule::String`. Other value types are not valid rule expressions.
    fn value_to_rule(&mut self, id: ValueId) -> RuleId {
        match self.get_val(id) {
            Value::Rule(rid) => *rid,
            Value::Str(s) => {
                let s = *s;
                self.alloc_rule(ARule::String(s))
            }
            // Typecheck ensures only rule-like values reach here
            _ => unreachable!(),
        }
    }

    /// Convert an evaluated value to an arena precedence.
    ///
    /// Integers become `APrec::Integer`, strings become `APrec::Name`.
    fn value_to_prec(&self, id: ValueId) -> APrec {
        match self.get_val(id) {
            Value::Int(n) => APrec::Integer(*n),
            Value::Str(s) => APrec::Name(*s),
            // Typecheck ensures only prec-like values reach here
            _ => unreachable!(),
        }
    }

    /// Evaluate a grammar combinator node directly to an arena rule.
    ///
    /// Handles `seq`, `choice`, `repeat`, `repeat1`, `optional`, `blank`,
    /// `field`, `alias`, `token`, `prec` variants, and `reserved`. For
    /// `seq`/`choice`, inline `for` expressions are expanded in-place.
    fn eval_combinator(&mut self, id: NodeId) -> Result<RuleId, LowerError> {
        let ast = self.ast;
        match ast.node(id) {
            Node::Seq(range) | Node::Choice(range) => {
                let is_seq = matches!(ast.node(id), Node::Seq(_));
                let members = ast.child_slice(*range);
                let mut child_ids = Vec::new();
                for &member in members {
                    if matches!(ast.node(member), Node::For(_)) {
                        let for_idx = match ast.node(member) {
                            Node::For(idx) => *idx,
                            _ => unreachable!(),
                        };
                        let config = ast.get_for(for_idx);
                        child_ids.extend(self.eval_for_to_rules(config)?);
                    } else {
                        child_ids.push(self.lower_to_rule(member)?);
                    }
                }
                let start = self.children_start();
                for id in child_ids {
                    self.push_child(id);
                }
                let (offset, len) = self.children_range(start);
                if is_seq {
                    Ok(self.alloc_rule(ARule::Seq(offset, len)))
                } else {
                    Ok(self.alloc_rule(ARule::Choice(offset, len)))
                }
            }
            Node::Repeat(inner) => {
                let inner = self.lower_to_rule(*inner)?;
                let rep = self.alloc_rule(ARule::Repeat(inner));
                let blank = self.alloc_rule(ARule::Blank);
                let start = self.children_start();
                self.push_child(rep);
                self.push_child(blank);
                let (s, l) = self.children_range(start);
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
                self.push_child(inner);
                self.push_child(blank);
                let (s, l) = self.children_range(start);
                Ok(self.alloc_rule(ARule::Choice(s, l)))
            }
            Node::Blank => Ok(self.alloc_rule(ARule::Blank)),
            Node::Field { name, content } => {
                let (name, content) = (*name, *content);
                let inner = self.lower_to_rule(content)?;
                let sid = self.strings.intern_span(ast.span(name));
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
                            Value::Str(str_handle) => {
                                let str_handle = *str_handle;
                                Ok(self.alloc_rule(ARule::Alias(str_handle, false, inner)))
                            }
                            Value::Rule(rid) => {
                                let str_handle = match &self.rules[rid.0 as usize] {
                                    ARule::NamedSymbol(str_handle) => *str_handle,
                                    _ => unreachable!(),
                                };
                                Ok(self.alloc_rule(ARule::Alias(str_handle, true, inner)))
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
                let (value, content) = (*value, *content);
                let vid = self.eval_expr(value)?;
                let prec = self.value_to_prec(vid);
                let inner = self.lower_to_rule(content)?;
                let make_rule: fn(APrec, RuleId) -> ARule = match ast.node(id) {
                    Node::Prec { .. } => ARule::Prec,
                    Node::PrecLeft { .. } => ARule::PrecLeft,
                    Node::PrecRight { .. } => ARule::PrecRight,
                    _ => unreachable!(),
                };
                Ok(self.alloc_rule(make_rule(prec, inner)))
            }
            Node::PrecDynamic { value, content } => {
                let (value, content) = (*value, *content);
                let vid = self.eval_expr(value)?;
                let prec_value = match self.get_val(vid) {
                    Value::Int(n) => *n,
                    _ => unreachable!(),
                };
                let inner = self.lower_to_rule(content)?;
                Ok(self.alloc_rule(ARule::PrecDynamic(prec_value, inner)))
            }
            Node::Reserved { context, content } => {
                let (context, content) = (*context, *content);
                let inner = self.lower_to_rule(content)?;
                let sid = self.strings.intern_span(ast.span(context));
                Ok(self.alloc_rule(ARule::Reserved(sid, inner)))
            }
            _ => unreachable!(),
        }
    }

    /// Evaluate a user-defined function call with pre-evaluated arguments.
    ///
    /// Looks up the function by name, pushes a new scope with parameters
    /// bound to the argument values, evaluates the body, then pops the scope.
    fn eval_fn_call_with_vals(
        &mut self,
        name: &str,
        arg_vals: &[ValueId],
    ) -> Result<ValueId, LowerError> {
        // Resolve pass ensures all called functions are registered
        let &fn_id = self.fns.get(name).unwrap();

        let config = self.get_fn_config(fn_id);
        let param_names: Vec<NodeId> = config.params.iter().map(|param| param.name).collect();
        let body = config.body;

        self.push_scope();
        for (name_id, val_id) in param_names.iter().zip(arg_vals.iter()) {
            self.bind(self.ast.node_text(*name_id), *val_id);
        }

        let result = self.eval_expr(body);
        self.pop_scope();
        result
    }

    /// Evaluate a `for` expression, producing a list of arena rule IDs.
    ///
    /// Evaluates the iterable, then for each element binds the loop
    /// variables and evaluates the body to a rule. Used by `seq`/`choice`
    /// to splice for-expression results inline.
    fn eval_for_to_rules(&mut self, config: &ForConfig) -> Result<Vec<RuleId>, LowerError> {
        let iterable = config.iterable;
        let body = config.body;
        let n_bindings = config.bindings.len();

        let iter_vid = self.eval_expr(iterable)?;
        // Typecheck ensures iterable is a list
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
            for j in 0..n_bindings {
                let value_id = match self.get_val(item_id) {
                    Value::Tuple(ids) => ids[j],
                    _ => item_id,
                };
                let (name_span, _) = config.bindings[j];
                self.bind(self.ast.text(name_span), value_id);
            }
            results.push(self.lower_to_rule(body)?);
            self.pop_scope();
        }
        Ok(results)
    }
}

/// Convert an [`InputGrammar`] to a `serde_json::Value` in `grammar.json` format.
///
/// The output matches the JSON schema expected by the existing tree-sitter
/// parser generator, enabling round-trip compatibility between the native
/// DSL and the JavaScript-based grammar definition.
pub fn grammar_to_json(grammar: &InputGrammar) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert(
        "name".into(),
        serde_json::Value::String(grammar.name.clone()),
    );
    let mut rules = serde_json::Map::new();
    for var in &grammar.variables {
        rules.insert(var.name.clone(), rule_to_json(&var.rule));
    }
    obj.insert("rules".into(), serde_json::Value::Object(rules));
    obj.insert(
        "extras".into(),
        serde_json::Value::Array(grammar.extra_symbols.iter().map(rule_to_json).collect()),
    );
    obj.insert(
        "conflicts".into(),
        serde_json::Value::Array(
            grammar
                .expected_conflicts
                .iter()
                .map(|g| {
                    serde_json::Value::Array(
                        g.iter()
                            .map(|s| serde_json::Value::String(s.clone()))
                            .collect(),
                    )
                })
                .collect(),
        ),
    );
    obj.insert(
        "precedences".into(),
        serde_json::Value::Array(
            grammar
                .precedence_orderings
                .iter()
                .map(|g| {
                    serde_json::Value::Array(
                        g.iter()
                            .map(|e| match e {
                                PrecedenceEntry::Name(s) => {
                                    serde_json::json!({"type": "STRING", "value": s})
                                }
                                PrecedenceEntry::Symbol(s) => {
                                    serde_json::json!({"type": "SYMBOL", "name": s})
                                }
                            })
                            .collect(),
                    )
                })
                .collect(),
        ),
    );
    obj.insert(
        "externals".into(),
        serde_json::Value::Array(grammar.external_tokens.iter().map(rule_to_json).collect()),
    );
    obj.insert(
        "inline".into(),
        serde_json::Value::Array(
            grammar
                .variables_to_inline
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
    );
    obj.insert(
        "supertypes".into(),
        serde_json::Value::Array(
            grammar
                .supertype_symbols
                .iter()
                .map(|s| serde_json::Value::String(s.clone()))
                .collect(),
        ),
    );
    if let Some(word) = &grammar.word_token {
        obj.insert("word".into(), serde_json::Value::String(word.clone()));
    }
    if !grammar.reserved_words.is_empty() {
        let mut reserved = serde_json::Map::new();
        for ctx in &grammar.reserved_words {
            reserved.insert(
                ctx.name.clone(),
                serde_json::Value::Array(ctx.reserved_words.iter().map(rule_to_json).collect()),
            );
        }
        obj.insert("reserved".into(), serde_json::Value::Object(reserved));
    }
    serde_json::Value::Object(obj)
}

/// Convert a single `Rule` to its JSON representation.
fn rule_to_json(rule: &Rule) -> serde_json::Value {
    use crate::rules::Alias;
    match rule {
        Rule::Blank => serde_json::json!({"type": "BLANK"}),
        Rule::String(s) => serde_json::json!({"type": "STRING", "value": s}),
        Rule::Pattern(p, f) if f.is_empty() => serde_json::json!({"type": "PATTERN", "value": p}),
        Rule::Pattern(p, f) => serde_json::json!({"type": "PATTERN", "value": p, "flags": f}),
        Rule::NamedSymbol(name) => serde_json::json!({"type": "SYMBOL", "name": name}),
        Rule::Symbol(sym) => {
            serde_json::json!({"type": "SYMBOL", "name": format!("__symbol_{}", sym.index)})
        }
        Rule::Choice(ms) => {
            serde_json::json!({"type": "CHOICE", "members": ms.iter().map(rule_to_json).collect::<Vec<_>>()})
        }
        Rule::Seq(ms) => {
            serde_json::json!({"type": "SEQ", "members": ms.iter().map(rule_to_json).collect::<Vec<_>>()})
        }
        Rule::Repeat(inner) => {
            serde_json::json!({"type": "REPEAT1", "content": rule_to_json(inner)})
        }
        Rule::Metadata { params, rule } => {
            let mut content = rule_to_json(rule);
            if params.dynamic_precedence != 0 {
                content = serde_json::json!({"type": "PREC_DYNAMIC", "value": params.dynamic_precedence, "content": content});
            }
            if let Some(pv) = match &params.precedence {
                Precedence::None => None,
                Precedence::Integer(n) => Some(serde_json::Value::Number((*n).into())),
                Precedence::Name(s) => Some(serde_json::Value::String(s.clone())),
            } {
                content = match params.associativity {
                    Some(crate::rules::Associativity::Left) => {
                        serde_json::json!({"type": "PREC_LEFT", "value": pv, "content": content})
                    }
                    Some(crate::rules::Associativity::Right) => {
                        serde_json::json!({"type": "PREC_RIGHT", "value": pv, "content": content})
                    }
                    None => serde_json::json!({"type": "PREC", "value": pv, "content": content}),
                };
            }
            if let Some(Alias { value, is_named }) = &params.alias {
                content = serde_json::json!({"type": "ALIAS", "content": content, "named": is_named, "value": value});
            }
            if let Some(ref field_name) = params.field_name {
                content =
                    serde_json::json!({"type": "FIELD", "name": field_name, "content": content});
            }
            if params.is_token && params.is_main_token {
                content = serde_json::json!({"type": "IMMEDIATE_TOKEN", "content": content});
            } else if params.is_token {
                content = serde_json::json!({"type": "TOKEN", "content": content});
            }
            content
        }
        Rule::Reserved { rule, context_name } => {
            serde_json::json!({"type": "RESERVED", "context_name": context_name, "content": rule_to_json(rule)})
        }
    }
}

/// Process escape sequences in a string literal's content.
///
/// Only called for strings that contain backslashes (detected by a fast
/// `contains('\\')` check in `eval_expr`). The lexer has already validated
/// that all escape sequences are valid.
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
                _ => unreachable!("lexer validated escape sequences"),
            }
        } else {
            result.push(b as char);
        }
    }
    result
}

/// An error encountered during lowering/evaluation.
#[derive(Debug, Serialize, Error)]
pub struct LowerError {
    pub kind: LowerErrorKind,
    pub span: Span,
}

/// The specific kind of lowering error.
#[derive(Debug, Serialize)]
pub enum LowerErrorKind {
    MissingGrammarBlock,
    MissingLanguageField,
}

impl std::fmt::Display for LowerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            LowerErrorKind::MissingGrammarBlock => write!(f, "missing grammar block"),
            LowerErrorKind::MissingLanguageField => {
                write!(f, "grammar block missing 'language' field")
            }
        }
    }
}
