use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::ExtractedSyntaxGrammar;
use crate::{
    grammars::{
        Production, ProductionStep, ReservedWordSetId, SyntaxGrammar, SyntaxVariable, Variable,
    },
    rules::{Alias, Associativity, Precedence, Rule, Symbol, TokenSet},
};

pub type FlattenGrammarResult<T> = Result<T, FlattenGrammarError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum FlattenGrammarError {
    #[error("No such reserved word set: {0}")]
    NoReservedWordSet(String),
    #[error(
        "The rule `{0}` matches the empty string.

Tree-sitter does not support syntactic rules that match the empty string
unless they are used only as the grammar's start rule.
"
    )]
    EmptyString(String),
    #[error("Rule `{0}` cannot be inlined because it contains a reference to itself")]
    RecursiveInline(String),
}

struct RuleFlattener {
    production: Production,
    reserved_word_set_ids: FxHashMap<String, ReservedWordSetId>,
    precedence_stack: Vec<Precedence>,
    associativity_stack: Vec<Associativity>,
    reserved_word_stack: Vec<ReservedWordSetId>,
    alias_stack: Vec<Alias>,
    field_name_stack: Vec<String>,
}

impl RuleFlattener {
    const fn new(reserved_word_set_ids: FxHashMap<String, ReservedWordSetId>) -> Self {
        Self {
            production: Production {
                steps: Vec::new(),
                dynamic_precedence: 0,
            },
            reserved_word_set_ids,
            precedence_stack: Vec::new(),
            associativity_stack: Vec::new(),
            reserved_word_stack: Vec::new(),
            alias_stack: Vec::new(),
            field_name_stack: Vec::new(),
        }
    }

    fn flatten_variable(&mut self, variable: Variable) -> FlattenGrammarResult<SyntaxVariable> {
        let choices = extract_choices(variable.rule);
        let mut productions = Vec::with_capacity(choices.len());
        for rule in choices {
            let production = self.flatten_rule(rule)?;
            if !productions.contains(&production) {
                productions.push(production);
            }
        }
        Ok(SyntaxVariable {
            name: variable.name,
            kind: variable.kind,
            productions,
        })
    }

    fn flatten_rule(&mut self, rule: Rule) -> FlattenGrammarResult<Production> {
        self.production = Production::default();
        self.alias_stack.clear();
        self.reserved_word_stack.clear();
        self.precedence_stack.clear();
        self.associativity_stack.clear();
        self.field_name_stack.clear();
        self.apply(rule, true)?;
        Ok(self.production.clone())
    }

    fn apply(&mut self, rule: Rule, at_end: bool) -> FlattenGrammarResult<bool> {
        match rule {
            Rule::Seq(members) => {
                let mut result = false;
                let last_index = members.len() - 1;
                for (i, member) in members.into_iter().enumerate() {
                    result |= self.apply(member, i == last_index && at_end)?;
                }
                Ok(result)
            }
            Rule::Metadata { rule, params } => {
                let mut has_precedence = false;
                if !params.precedence.is_none() {
                    has_precedence = true;
                    self.precedence_stack.push(params.precedence);
                }

                let mut has_associativity = false;
                if let Some(associativity) = params.associativity {
                    has_associativity = true;
                    self.associativity_stack.push(associativity);
                }

                let mut has_alias = false;
                if let Some(alias) = params.alias {
                    has_alias = true;
                    self.alias_stack.push(alias);
                }

                let mut has_field_name = false;
                if let Some(field_name) = params.field_name {
                    has_field_name = true;
                    self.field_name_stack.push(field_name);
                }

                if params.dynamic_precedence.abs() > self.production.dynamic_precedence.abs() {
                    self.production.dynamic_precedence = params.dynamic_precedence;
                }

                let did_push = self.apply(*rule, at_end)?;

                if has_precedence {
                    self.precedence_stack.pop();
                    if did_push && !at_end {
                        self.production.steps.last_mut().unwrap().precedence = self
                            .precedence_stack
                            .last()
                            .cloned()
                            .unwrap_or(Precedence::None);
                    }
                }

                if has_associativity {
                    self.associativity_stack.pop();
                    if did_push && !at_end {
                        self.production.steps.last_mut().unwrap().associativity =
                            self.associativity_stack.last().copied();
                    }
                }

                if has_alias {
                    self.alias_stack.pop();
                }

                if has_field_name {
                    self.field_name_stack.pop();
                }

                Ok(did_push)
            }
            Rule::Reserved { rule, context_name } => {
                self.reserved_word_stack.push(
                    self.reserved_word_set_ids
                        .get(&context_name)
                        .copied()
                        .ok_or_else(|| {
                            FlattenGrammarError::NoReservedWordSet(context_name.clone())
                        })?,
                );
                let did_push = self.apply(*rule, at_end)?;
                self.reserved_word_stack.pop();
                Ok(did_push)
            }
            Rule::Symbol(symbol) => {
                self.production.steps.push(ProductionStep {
                    symbol,
                    precedence: self
                        .precedence_stack
                        .last()
                        .cloned()
                        .unwrap_or(Precedence::None),
                    associativity: self.associativity_stack.last().copied(),
                    reserved_word_set_id: self
                        .reserved_word_stack
                        .last()
                        .copied()
                        .unwrap_or_default(),
                    alias: self.alias_stack.last().cloned(),
                    field_name: self.field_name_stack.last().cloned(),
                });
                Ok(true)
            }
            _ => Ok(false),
        }
    }
}

fn extract_choices(rule: Rule) -> Vec<Rule> {
    match rule {
        Rule::Seq(elements) => {
            let mut result = vec![Rule::Blank];
            for element in elements {
                let extraction = extract_choices(element);
                let mut next_result = Vec::with_capacity(result.len());
                for entry in result {
                    for extraction_entry in &extraction {
                        next_result.push(Rule::Seq(vec![entry.clone(), extraction_entry.clone()]));
                    }
                }
                result = next_result;
            }
            result
        }
        Rule::Choice(elements) => {
            let mut result = Vec::with_capacity(elements.len());
            for element in elements {
                for rule in extract_choices(element) {
                    result.push(rule);
                }
            }
            result
        }
        Rule::Metadata { rule, params } => extract_choices(*rule)
            .into_iter()
            .map(|rule| Rule::Metadata {
                rule: Box::new(rule),
                params: params.clone(),
            })
            .collect(),
        Rule::Reserved { rule, context_name } => extract_choices(*rule)
            .into_iter()
            .map(|rule| Rule::Reserved {
                rule: Box::new(rule),
                context_name: context_name.clone(),
            })
            .collect(),
        _ => vec![rule],
    }
}

fn symbol_is_used(variables: &[SyntaxVariable], symbol: Symbol) -> bool {
    for variable in variables {
        for production in &variable.productions {
            for step in &production.steps {
                if step.symbol == symbol {
                    return true;
                }
            }
        }
    }
    false
}

pub(super) fn flatten_grammar(
    grammar: ExtractedSyntaxGrammar,
) -> FlattenGrammarResult<SyntaxGrammar> {
    let mut reserved_word_set_ids_by_name = FxHashMap::default();
    for (ix, set) in grammar.reserved_word_sets.iter().enumerate() {
        reserved_word_set_ids_by_name.insert(set.name.clone(), ReservedWordSetId(ix));
    }

    let mut flattener = RuleFlattener::new(reserved_word_set_ids_by_name);
    let variables = grammar
        .variables
        .into_iter()
        .map(|variable| flattener.flatten_variable(variable))
        .collect::<FlattenGrammarResult<Vec<_>>>()?;

    for (i, variable) in variables.iter().enumerate() {
        let symbol = Symbol::non_terminal(i);
        let used = symbol_is_used(&variables, symbol);

        for production in &variable.productions {
            if used && production.steps.is_empty() {
                Err(FlattenGrammarError::EmptyString(variable.name.clone()))?;
            }

            if grammar.variables_to_inline.contains(&symbol)
                && production.steps.iter().any(|step| step.symbol == symbol)
            {
                Err(FlattenGrammarError::RecursiveInline(variable.name.clone()))?;
            }
        }
    }
    let mut reserved_word_sets = grammar
        .reserved_word_sets
        .into_iter()
        .map(|set| set.reserved_words.into_iter().collect())
        .collect::<Vec<_>>();

    // If no default reserved word set is specified, there are no reserved words.
    if reserved_word_sets.is_empty() {
        reserved_word_sets.push(TokenSet::default());
    }

    Ok(SyntaxGrammar {
        extra_symbols: grammar.extra_symbols,
        expected_conflicts: grammar.expected_conflicts,
        variables_to_inline: grammar.variables_to_inline,
        precedence_orderings: grammar.precedence_orderings,
        external_tokens: grammar.external_tokens,
        supertype_symbols: grammar.supertype_symbols,
        word_token: grammar.word_token,
        reserved_word_sets,
        variables,
    })
}

// --- pool-based pass (replaces the Rule-based one at the container flip) ---

use super::extract_tokens::PoolExtractedMeta;
use crate::rule_pool::{
    FProd, FStep, FlatOut, FSTEP_ALIAS_NAMED, FSTEP_ASSOC_LEFT, FSTEP_ASSOC_RIGHT,
    FSTEP_PREC_INTEGER, FSTEP_PREC_NAME, Node as PoolNode, NodeId, PoolGrammar, Prec,
    RulePool, StrId,
};
use crate::rules::SymbolType;

/// One continuation frame, packed to 8 bytes (tag in the 2 high bits of `.0`).
///   Seq:          tag 0, `.0` = next child (abs index into `children`), `.1` = end
///   MetaExit:     tag 1, `.0` low bits = pushed-stack flags, `.1` = steps len at entry
///   ReservedExit: tag 2
#[derive(Clone, Copy)]
struct Frame(u32, u32);

const _: () = assert!(std::mem::size_of::<Frame>() == 8);

const FR_SEQ: u32 = 0;
const FR_META: u32 = 1;
const FR_RES: u32 = 2;
const FR_PAYLOAD: u32 = (1 << 30) - 1;

const MP_PREC: u32 = 1;
const MP_ASSOC: u32 = 2;
const MP_ALIAS: u32 = 4;
const MP_FIELD: u32 = 8;

impl Frame {
    fn seq(start: u32, end: u32) -> Self {
        debug_assert!(end < (1 << 30));
        Self(start, end)
    }
    const fn meta(flags: u32, steps_before: u32) -> Self {
        Self((FR_META << 30) | flags, steps_before)
    }
    const fn reserved() -> Self {
        Self(FR_RES << 30, 0)
    }
    const fn tag(self) -> u32 {
        self.0 >> 30
    }
    const fn seq_exhausted(self) -> bool {
        self.0 == self.1
    }
}

/// Reusable flatten scratch. Everything is truncate-restore or snapshot
/// scratch; steady state allocates only output-pool growth.
#[derive(Default)]
pub(super) struct FlattenState {
    frames: Vec<Frame>,
    steps: Vec<FStep>,
    prec: Vec<Prec>,
    assoc: Vec<Associativity>,
    alias: Vec<crate::rule_pool::Alias>,
    field: Vec<StrId>,
    reserved: Vec<u16>,
    /// Fork snapshots, stacked as segments. A branch can consume context
    /// pushed BEFORE the fork (an enclosing Metadata/Reserved exit pops it),
    /// so every stack that exits pop through needs a full snapshot; only
    /// `steps` grows monotonically within a branch and truncates back.
    snap_frames: Vec<Frame>,
    snap_prec: Vec<Prec>,
    snap_assoc: Vec<Associativity>,
    snap_alias: Vec<crate::rule_pool::Alias>,
    snap_field: Vec<StrId>,
    snap_reserved: Vec<u16>,
    dyn_prec: i32,
}

/// Snapshot offsets captured at a fork; lives on the Rust call stack.
#[derive(Clone, Copy)]
struct Mark {
    steps: u32,
    frames: u32,
    prec: u32,
    assoc: u32,
    alias: u32,
    field: u32,
    reserved: u32,
    dyn_prec: i32,
}

impl FlattenState {
    fn reset(&mut self) {
        self.frames.clear();
        self.steps.clear();
        self.prec.clear();
        self.assoc.clear();
        self.alias.clear();
        self.field.clear();
        self.reserved.clear();
        self.snap_frames.clear();
        self.snap_prec.clear();
        self.snap_assoc.clear();
        self.snap_alias.clear();
        self.snap_field.clear();
        self.snap_reserved.clear();
        self.dyn_prec = 0;
    }

    fn fork_save(&mut self) -> Mark {
        let mark = Mark {
            steps: self.steps.len() as u32,
            frames: self.snap_frames.len() as u32,
            prec: self.snap_prec.len() as u32,
            assoc: self.snap_assoc.len() as u32,
            alias: self.snap_alias.len() as u32,
            field: self.snap_field.len() as u32,
            reserved: self.snap_reserved.len() as u32,
            dyn_prec: self.dyn_prec,
        };
        self.snap_frames.extend_from_slice(&self.frames);
        self.snap_prec.extend_from_slice(&self.prec);
        self.snap_assoc.extend_from_slice(&self.assoc);
        self.snap_alias.extend_from_slice(&self.alias);
        self.snap_field.extend_from_slice(&self.field);
        self.snap_reserved.extend_from_slice(&self.reserved);
        mark
    }

    fn restore(&mut self, mark: Mark) {
        self.steps.truncate(mark.steps as usize);
        self.dyn_prec = mark.dyn_prec;
        self.frames.clear();
        self.frames
            .extend_from_slice(&self.snap_frames[mark.frames as usize..]);
        self.prec.clear();
        self.prec
            .extend_from_slice(&self.snap_prec[mark.prec as usize..]);
        self.assoc.clear();
        self.assoc
            .extend_from_slice(&self.snap_assoc[mark.assoc as usize..]);
        self.alias.clear();
        self.alias
            .extend_from_slice(&self.snap_alias[mark.alias as usize..]);
        self.field.clear();
        self.field
            .extend_from_slice(&self.snap_field[mark.field as usize..]);
        self.reserved.clear();
        self.reserved
            .extend_from_slice(&self.snap_reserved[mark.reserved as usize..]);
    }

    fn fork_done(&mut self, mark: Mark) {
        self.snap_frames.truncate(mark.frames as usize);
        self.snap_prec.truncate(mark.prec as usize);
        self.snap_assoc.truncate(mark.assoc as usize);
        self.snap_alias.truncate(mark.alias as usize);
        self.snap_field.truncate(mark.field as usize);
        self.snap_reserved.truncate(mark.reserved as usize);
    }

    fn push_step(&mut self, kind: SymbolType, index: u32) {
        let (prec_bits, prec_val) = match self.prec.last() {
            None | Some(Prec::None) => (0u8, 0i32),
            Some(Prec::Integer(n)) => (FSTEP_PREC_INTEGER, *n),
            Some(Prec::Name(sid)) => (FSTEP_PREC_NAME, sid.raw() as i32),
        };
        let assoc_bits = match self.assoc.last() {
            None => 0u8,
            Some(Associativity::Left) => FSTEP_ASSOC_LEFT,
            Some(Associativity::Right) => FSTEP_ASSOC_RIGHT,
        };
        let (alias, named_bit) = self.alias.last().map_or((0, 0u8), |a| {
            (a.value.raw(), if a.is_named { FSTEP_ALIAS_NAMED } else { 0 })
        });
        self.steps.push(FStep {
            sym_index: index,
            prec_val,
            alias,
            field: self.field.last().map_or(0, |f| f.raw()),
            reserved: self.reserved.last().copied().unwrap_or(0),
            flags: (kind as u8) | prec_bits | assoc_bits | named_bit,
        });
    }

    /// `true` if nothing after the current point can push a step. Equivalent
    /// to master's top-down `at_end`.
    fn at_end(&self) -> bool {
        self.frames
            .iter()
            .all(|f| f.tag() != FR_SEQ || f.seq_exhausted())
    }

    /// Master's `Metadata` pop branch: pop what was pushed; if a step was
    /// pushed in the subtree and more steps can follow, rewrite the last
    /// step's precedence/associativity to the outer context.
    fn meta_exit(&mut self, flags: u32, steps_before: u32) {
        let did_push = self.steps.len() as u32 > steps_before;
        let fixup = did_push && !self.at_end();
        if flags & MP_PREC != 0 {
            self.prec.pop();
            if fixup {
                let (bits, val) = match self.prec.last() {
                    None | Some(Prec::None) => (0u8, 0i32),
                    Some(Prec::Integer(n)) => (FSTEP_PREC_INTEGER, *n),
                    Some(Prec::Name(sid)) => (FSTEP_PREC_NAME, sid.raw() as i32),
                };
                let step = self.steps.last_mut().unwrap();
                step.prec_val = val;
                step.flags = (step.flags & !(0b11 << 3)) | bits;
            }
        }
        if flags & MP_ASSOC != 0 {
            self.assoc.pop();
            if fixup {
                let bits = match self.assoc.last() {
                    None => 0u8,
                    Some(Associativity::Left) => FSTEP_ASSOC_LEFT,
                    Some(Associativity::Right) => FSTEP_ASSOC_RIGHT,
                };
                let step = self.steps.last_mut().unwrap();
                step.flags = (step.flags & !(0b11 << 5)) | bits;
            }
        }
        if flags & MP_ALIAS != 0 {
            self.alias.pop();
        }
        if flags & MP_FIELD != 0 {
            self.field.pop();
        }
    }
}

pub(super) fn flatten_grammar_pool(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    st: &mut FlattenState,
    out: &mut FlatOut,
) -> FlattenGrammarResult<()> {
    // Last wins on duplicate names, matching master's map build.
    let reserved_ids: FxHashMap<StrId, u16> = meta
        .reserved_sets
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, i as u16))
        .collect();
    out.clear();
    for v in &g.variables {
        let prod_start = out.productions.len() as u32;
        st.reset();
        run_path(&g.pool, &reserved_ids, v.root, st, out, prod_start)?;
        out.var_prods.push((prod_start, out.productions.len() as u32));
    }
    check(g, meta, out)
}

/// Mirror of master's post-flatten checks, same iteration order.
fn check(g: &PoolGrammar, meta: &PoolExtractedMeta, out: &FlatOut) -> FlattenGrammarResult<()> {
    for (i, &(ps, pe)) in out.var_prods.iter().enumerate() {
        let symbol = Symbol::non_terminal(i);
        let used = out.steps.iter().any(|s| s.symbol() == symbol);
        let inlined = meta.inline.contains(&symbol);
        for p in &out.productions[ps as usize..pe as usize] {
            if used && p.steps_len == 0 {
                Err(FlattenGrammarError::EmptyString(
                    g.pool.resolve(g.variables[i].name).to_string(),
                ))?;
            }
            if inlined
                && out.steps[p.step_range()].iter().any(|s| s.symbol() == symbol)
            {
                Err(FlattenGrammarError::RecursiveInline(
                    g.pool.resolve(g.variables[i].name).to_string(),
                ))?;
            }
        }
    }
    Ok(())
}

/// Enter `node` and, unless it forked, drive the continuation to completion.
fn run_path(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    node: NodeId,
    st: &mut FlattenState,
    out: &mut FlatOut,
    prod_start: u32,
) -> FlattenGrammarResult<()> {
    if enter(pool, reserved_ids, node, st, out, prod_start)? {
        drive(pool, reserved_ids, st, out, prod_start)?;
    }
    Ok(())
}

/// Process one node. Returns `false` if it forked (a `Choice` drove every
/// branch to completion; the caller's continuation is consumed). Unary
/// chains iterate in place.
fn enter(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    mut node: NodeId,
    st: &mut FlattenState,
    out: &mut FlatOut,
    prod_start: u32,
) -> FlattenGrammarResult<bool> {
    loop {
        match pool.node(node) {
            PoolNode::Sym { kind, index } => {
                st.push_step(kind, index);
                return Ok(true);
            }
            PoolNode::Seq(range) => {
                st.frames
                    .push(Frame::seq(range.start, range.start + range.len));
                return Ok(true);
            }
            PoolNode::Metadata { params, rule } => {
                let p = pool.params(params);
                let mut flags = 0;
                if p.prec != Prec::None {
                    st.prec.push(p.prec);
                    flags |= MP_PREC;
                }
                if let Some(a) = p.assoc {
                    st.assoc.push(a);
                    flags |= MP_ASSOC;
                }
                if let Some(a) = p.alias {
                    st.alias.push(a);
                    flags |= MP_ALIAS;
                }
                if let Some(f) = p.field {
                    st.field.push(f);
                    flags |= MP_FIELD;
                }
                if p.dynamic_precedence.abs() > st.dyn_prec.abs() {
                    st.dyn_prec = p.dynamic_precedence;
                }
                st.frames.push(Frame::meta(flags, st.steps.len() as u32));
                node = rule;
            }
            PoolNode::Reserved { rule, ctx } => {
                let Some(&set) = reserved_ids.get(&ctx) else {
                    return Err(FlattenGrammarError::NoReservedWordSet(
                        pool.resolve(ctx).to_string(),
                    ));
                };
                st.reserved.push(set);
                st.frames.push(Frame::reserved());
                node = rule;
            }
            PoolNode::Choice(range) => {
                let mark = st.fork_save();
                for i in range.as_range() {
                    let child = pool.child_slice(range)[i - range.start as usize];
                    run_path(pool, reserved_ids, child, st, out, prod_start)?;
                    st.restore(mark);
                }
                st.fork_done(mark);
                return Ok(false);
            }
            // Blank pushes nothing; String/Pattern/NamedSymbol/Repeat cannot
            // appear post-extract/expand and match master's no-op arm.
            _ => return Ok(true),
        }
    }
}

/// Consume frames until the continuation is exhausted, then emit. Stops
/// without emitting if a deeper fork took over.
fn drive(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    st: &mut FlattenState,
    out: &mut FlatOut,
    prod_start: u32,
) -> FlattenGrammarResult<()> {
    loop {
        let Some(&top) = st.frames.last() else {
            emit(st, out, prod_start);
            return Ok(());
        };
        match top.tag() {
            FR_SEQ => {
                if top.seq_exhausted() {
                    st.frames.pop();
                } else {
                    let cur = top.0 & FR_PAYLOAD;
                    st.frames.last_mut().unwrap().0 += 1;
                    let child = pool.child_slice(crate::rule_pool::NodeRange {
                        start: cur,
                        len: 1,
                    })[0];
                    if !enter(pool, reserved_ids, child, st, out, prod_start)? {
                        return Ok(());
                    }
                }
            }
            FR_META => {
                let (flags, steps_before) = (top.0 & FR_PAYLOAD, top.1);
                st.frames.pop();
                st.meta_exit(flags, steps_before);
            }
            _ => {
                st.frames.pop();
                st.reserved.pop();
            }
        }
    }
}

/// Append the completed path as a production unless this variable already
/// has an identical one (master's order-preserving dedup).
fn emit(st: &mut FlattenState, out: &mut FlatOut, prod_start: u32) {
    for p in &out.productions[prod_start as usize..] {
        if p.dynamic_precedence == st.dyn_prec && out.steps[p.step_range()] == st.steps[..] {
            return;
        }
    }
    let steps_start = out.steps.len() as u32;
    out.steps.extend_from_slice(&st.steps);
    out.productions.push(FProd {
        steps_start,
        steps_len: st.steps.len() as u32,
        dynamic_precedence: st.dyn_prec,
    });
}

/// Materialize the pool pass's output as a master [`SyntaxGrammar`] for A/B
/// verification. Deleted at the container flip.
pub(super) fn materialize_flattened(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    out: &FlatOut,
) -> SyntaxGrammar {
    let step_to_master = |s: &FStep| ProductionStep {
        symbol: s.symbol(),
        precedence: match s.precedence() {
            Prec::None => Precedence::None,
            Prec::Integer(n) => Precedence::Integer(n),
            Prec::Name(sid) => Precedence::Name(g.pool.resolve(sid).to_string()),
        },
        associativity: s.associativity(),
        alias: s.alias().map(|a| crate::rules::Alias {
            value: g.pool.resolve(a.value).to_string(),
            is_named: a.is_named,
        }),
        field_name: s.field().map(|f| g.pool.resolve(f).to_string()),
        reserved_word_set_id: ReservedWordSetId(s.reserved as usize),
    };
    let mut reserved_word_sets: Vec<TokenSet> = meta
        .reserved_sets
        .iter()
        .map(|(_, symbols)| symbols.iter().copied().collect())
        .collect();
    if reserved_word_sets.is_empty() {
        reserved_word_sets.push(TokenSet::default());
    }
    SyntaxGrammar {
        variables: g
            .variables
            .iter()
            .zip(&meta.kinds)
            .zip(&out.var_prods)
            .map(|((v, &kind), &(ps, pe))| SyntaxVariable {
                name: g.pool.resolve(v.name).to_string(),
                kind,
                productions: out.productions[ps as usize..pe as usize]
                    .iter()
                    .map(|p| Production {
                        dynamic_precedence: p.dynamic_precedence,
                        steps: out.steps[p.step_range()].iter().map(step_to_master).collect(),
                    })
                    .collect(),
            })
            .collect(),
        extra_symbols: meta.extra_symbols.clone(),
        expected_conflicts: meta.conflicts.clone(),
        variables_to_inline: meta.inline.clone(),
        precedence_orderings: g.precedence_orderings.clone(),
        external_tokens: meta.external_tokens.clone(),
        supertype_symbols: meta.supertypes.clone(),
        word_token: meta.word,
        reserved_word_sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::VariableType;

    #[test]
    fn test_flatten_grammar() {
        let mut flattener = RuleFlattener::new(FxHashMap::default());
        let result = flattener
            .flatten_variable(Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::seq(vec![
                    Rule::non_terminal(1),
                    Rule::prec_left(
                        Precedence::Integer(101),
                        Rule::seq(vec![
                            Rule::non_terminal(2),
                            Rule::choice(vec![
                                Rule::prec_right(
                                    Precedence::Integer(102),
                                    Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                                ),
                                Rule::non_terminal(5),
                            ]),
                            Rule::non_terminal(6),
                        ]),
                    ),
                    Rule::non_terminal(7),
                ]),
            })
            .unwrap();

        assert_eq!(
            result.productions,
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(3))
                            .with_prec(Precedence::Integer(102), Some(Associativity::Right)),
                        ProductionStep::new(Symbol::non_terminal(4))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ]
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(5))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ]
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_maximum_dynamic_precedence() {
        let mut flattener = RuleFlattener::new(FxHashMap::default());
        let result = flattener
            .flatten_variable(Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::seq(vec![
                    Rule::non_terminal(1),
                    Rule::prec_dynamic(
                        101,
                        Rule::seq(vec![
                            Rule::non_terminal(2),
                            Rule::choice(vec![
                                Rule::prec_dynamic(
                                    102,
                                    Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                                ),
                                Rule::non_terminal(5),
                            ]),
                            Rule::non_terminal(6),
                        ]),
                    ),
                    Rule::non_terminal(7),
                ]),
            })
            .unwrap();

        assert_eq!(
            result.productions,
            vec![
                Production {
                    dynamic_precedence: 102,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::non_terminal(3)),
                        ProductionStep::new(Symbol::non_terminal(4)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ],
                },
                Production {
                    dynamic_precedence: 101,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::non_terminal(5)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ],
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_final_precedence() {
        let mut flattener = RuleFlattener::new(FxHashMap::default());
        let result = flattener
            .flatten_variable(Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::prec_left(
                    Precedence::Integer(101),
                    Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
                ),
            })
            .unwrap();

        assert_eq!(
            result.productions,
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::non_terminal(1))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                    ProductionStep::new(Symbol::non_terminal(2))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                ]
            }]
        );

        let result = flattener
            .flatten_variable(Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::prec_left(
                    Precedence::Integer(101),
                    Rule::seq(vec![Rule::non_terminal(1)]),
                ),
            })
            .unwrap();

        assert_eq!(
            result.productions,
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::non_terminal(1))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                ]
            }]
        );
    }

    #[test]
    fn test_flatten_grammar_with_field_names() {
        let mut flattener = RuleFlattener::new(FxHashMap::default());
        let result = flattener
            .flatten_variable(Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::seq(vec![
                    Rule::field("first-thing".to_string(), Rule::terminal(1)),
                    Rule::terminal(2),
                    Rule::choice(vec![
                        Rule::Blank,
                        Rule::field("second-thing".to_string(), Rule::terminal(3)),
                    ]),
                ]),
            })
            .unwrap();

        assert_eq!(
            result.productions,
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(1)).with_field_name("first-thing"),
                        ProductionStep::new(Symbol::terminal(2))
                    ]
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(1)).with_field_name("first-thing"),
                        ProductionStep::new(Symbol::terminal(2)),
                        ProductionStep::new(Symbol::terminal(3)).with_field_name("second-thing"),
                    ]
                },
            ]
        );
    }

    fn pool_from_extracted(
        g: &ExtractedSyntaxGrammar,
    ) -> (PoolGrammar, PoolExtractedMeta) {
        use crate::rule_pool::PoolVariable;
        let mut pool = RulePool::default();
        let variables = g
            .variables
            .iter()
            .map(|v| PoolVariable {
                name: pool.intern(&v.name),
                root: pool.add_rule(&v.rule),
            })
            .collect();
        let reserved_sets = g
            .reserved_word_sets
            .iter()
            .map(|s| (pool.intern(&s.name), s.reserved_words.clone()))
            .collect();
        let meta = PoolExtractedMeta {
            kinds: g.variables.iter().map(|v| v.kind).collect(),
            lexical_variables: Vec::new(),
            separator_roots: Vec::new(),
            extra_symbols: g.extra_symbols.clone(),
            external_tokens: g.external_tokens.clone(),
            reserved_sets,
            supertypes: g.supertype_symbols.clone(),
            conflicts: g.expected_conflicts.clone(),
            inline: g.variables_to_inline.clone(),
            word: g.word_token,
        };
        let pg = PoolGrammar {
            pool,
            name: String::new(),
            variables,
            external_roots: Vec::new(),
            extra_roots: Vec::new(),
            reserved_sets: Vec::new(),
            supertype_names: Vec::new(),
            conflict_names: Vec::new(),
            inline_names: Vec::new(),
            word_name: None,
            precedence_orderings: g.precedence_orderings.clone(),
        };
        (pg, meta)
    }

    fn assert_pool_matches_master(g: ExtractedSyntaxGrammar) {
        let (pg, meta) = pool_from_extracted(&g);
        let master = flatten_grammar(g);
        let mut st = FlattenState::default();
        let mut out = crate::rule_pool::FlatOut::default();
        let pool_result = flatten_grammar_pool(&pg, &meta, &mut st, &mut out);
        match (master, pool_result) {
            (Ok(m), Ok(())) => assert_eq!(materialize_flattened(&pg, &meta, &out), m),
            (Err(master_err), Err(pool_err)) => {
                assert_eq!(pool_err.to_string(), master_err.to_string());
            }
            (m, p) => panic!("master {m:?} but pool {p:?}"),
        }
    }

    fn extracted(variables: Vec<Variable>) -> ExtractedSyntaxGrammar {
        ExtractedSyntaxGrammar {
            variables,
            ..Default::default()
        }
    }

    #[test]
    fn pool_flatten_matches_master() {
        // Precedence fix-ups across a fork under enclosing metadata.
        assert_pool_matches_master(extracted(vec![Variable::named(
            "test",
            Rule::seq(vec![
                Rule::non_terminal(1),
                Rule::prec_left(
                    Precedence::Integer(101),
                    Rule::seq(vec![
                        Rule::non_terminal(2),
                        Rule::choice(vec![
                            Rule::prec_right(
                                Precedence::Integer(102),
                                Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                            ),
                            Rule::non_terminal(5),
                        ]),
                        Rule::non_terminal(6),
                    ]),
                ),
                Rule::non_terminal(7),
            ]),
        )]));

        // Fields, aliases, blanks, named precedence.
        assert_pool_matches_master(extracted(vec![Variable::named(
            "test",
            Rule::seq(vec![
                Rule::field("first-thing".to_string(), Rule::terminal(1)),
                Rule::alias(Rule::terminal(2), "two".to_string(), true),
                Rule::prec(Precedence::Name("p".to_string()), Rule::terminal(4)),
                Rule::choice(vec![
                    Rule::Blank,
                    Rule::field("second-thing".to_string(), Rule::terminal(3)),
                ]),
            ]),
        )]));

        // Reserved contexts resolve to set ids.
        let mut g = extracted(vec![Variable::named(
            "test",
            Rule::Reserved {
                rule: Box::new(Rule::terminal(1)),
                context_name: "default".to_string(),
            },
        )]);
        g.reserved_word_sets = vec![crate::grammars::ReservedWordContext {
            name: "default".to_string(),
            reserved_words: vec![Symbol::terminal(9)],
        }];
        assert_pool_matches_master(g);
    }

    #[test]
    fn pool_flatten_matches_master_errors() {
        // Unknown reserved context.
        assert_pool_matches_master(extracted(vec![Variable::named(
            "test",
            Rule::Reserved {
                rule: Box::new(Rule::terminal(1)),
                context_name: "nope".to_string(),
            },
        )]));

        // Empty production in a used variable.
        assert_pool_matches_master(extracted(vec![
            Variable::named("a", Rule::non_terminal(1)),
            Variable::named("b", Rule::Blank),
        ]));

        // Recursive inline.
        let mut g = extracted(vec![Variable::named(
            "test",
            Rule::seq(vec![
                Rule::non_terminal(0),
                Rule::non_terminal(1),
                Rule::non_terminal(2),
            ]),
        )]);
        g.variables_to_inline = vec![Symbol::non_terminal(0)];
        assert_pool_matches_master(g);
    }

    #[test]
    fn test_flatten_grammar_with_recursive_inline_variable() {
        let result = flatten_grammar(ExtractedSyntaxGrammar {
            extra_symbols: Vec::new(),
            expected_conflicts: Vec::new(),
            variables_to_inline: vec![Symbol::non_terminal(0)],
            precedence_orderings: Vec::new(),
            external_tokens: Vec::new(),
            supertype_symbols: Vec::new(),
            word_token: None,
            reserved_word_sets: Vec::new(),
            variables: vec![Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::seq(vec![
                    Rule::non_terminal(0),
                    Rule::non_terminal(1),
                    Rule::non_terminal(2),
                ]),
            }],
        });

        assert_eq!(
            result.unwrap_err().to_string(),
            "Rule `test` cannot be inlined because it contains a reference to itself",
        );
    }
}
