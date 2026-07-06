//! TEMP SPIKE (remove after the perf decision): flat-pool `intern_symbols` and
//! `flatten_grammar` to A/B against master's Rule-based passes. No hash-consing;
//! string payloads intern to a `StrId`; children of `Seq`/`Choice` live as
//! `(offset, len)` ranges into one backing `Vec`; `MetadataParams` live out of
//! line in a params pool so `FNode` stays 12 bytes.
//!
//! `flatten` enumerates choice alternatives by forking the walk at `Choice`
//! nodes (master materializes the cross-product as cloned `Rule` trees). All
//! flatten state is truncate-restore scratch: steady state does zero
//! allocations beyond output-pool growth. Output is pooled (`FStep`/`FProd`
//! ranges); materialization to master `Production`s happens only for the
//! untimed equality assert, mirroring the `Rule -> pool` setup exclusion (a
//! real backend consumes the flat form directly).
//!
//! Kept recursive at choice forks on purpose (depth = choice nesting); the
//! iterative conversion is a separate later step.

use std::num::NonZeroU32;

use rustc_hash::FxHashMap;

use super::{ExtractedSyntaxGrammar, flatten_grammar::FlattenGrammarError};
use crate::grammars::{
    InputGrammar, Production, ProductionStep, ReservedWordSetId, SyntaxVariable, Variable,
};
use crate::rules::{Alias, Associativity, Precedence, Rule, Symbol, SymbolType};

#[derive(Clone, Copy)]
pub struct NodeId(u32);

/// Interned string id: 1-based index into [`Pool::strs`] (`NonZeroU32` so
/// `Option<StrId>` stays 4 bytes).
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StrId(NonZeroU32);

impl StrId {
    fn idx(self) -> usize {
        self.0.get() as usize - 1
    }
    const fn raw(self) -> u32 {
        self.0.get()
    }
}

/// Index into [`Pool::params`].
#[derive(Clone, Copy)]
struct ParamsId(u32);

#[derive(Clone, Copy)]
struct NodeRange {
    start: u32,
    len: u32,
}

#[derive(Clone, Copy)]
enum FNode {
    Blank,
    String(StrId),
    Pattern(StrId, StrId),
    NamedSymbol(StrId),
    Sym { kind: SymbolType, index: u32 },
    Seq(NodeRange),
    Choice(NodeRange),
    Repeat(NodeId),
    Metadata { params: ParamsId, rule: NodeId },
    Reserved { rule: NodeId, ctx: StrId },
}

const _: () = assert!(std::mem::size_of::<FNode>() == 12);

/// `Precedence` with the name interned. `Copy`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FPrec {
    None,
    Integer(i32),
    Name(StrId),
}

/// `Alias` with the value interned. `Copy`.
#[derive(Clone, Copy)]
struct FAlias {
    value: StrId,
    is_named: bool,
}

/// The `MetadataParams` fields flatten reads, interned and `Copy`. The token
/// flags are dropped: neither spike pass reads them.
#[derive(Clone, Copy)]
struct FParams {
    prec: FPrec,
    assoc: Option<Associativity>,
    dynamic_precedence: i32,
    alias: Option<FAlias>,
    field: Option<StrId>,
}

/// Flat pool: `nodes` is the arena (a child holds a `NodeId` into it); `children`
/// backs `Seq`/`Choice` ranges; `params` backs `Metadata` nodes; `strs` +
/// `str_ids` double-store the interner (spike shortcut; the real design uses a
/// blob or `Rc<str>`); `name_of_symbol` resolves an interned rule name (a dense
/// low `StrId`) to its `Symbol` by direct index.
#[derive(Clone)]
pub struct Pool {
    nodes: Vec<FNode>,
    children: Vec<NodeId>,
    params: Vec<FParams>,
    str_ids: FxHashMap<Box<str>, StrId>,
    strs: Vec<Box<str>>,
    roots: Vec<NodeId>,
    ext_roots: Vec<NodeId>,
    extra_roots: Vec<NodeId>,
    reserved_roots: Vec<NodeId>,
    /// `StrId -> Symbol` for the dense low name `StrId`s (see `from_grammar`).
    name_of_symbol: Vec<Symbol>,
    /// Reserved-set name -> set index, resolved at import (from_extracted).
    reserved_sets: FxHashMap<StrId, u16>,
}

impl Pool {
    fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            children: Vec::new(),
            params: Vec::new(),
            str_ids: FxHashMap::default(),
            strs: Vec::new(),
            roots: Vec::new(),
            ext_roots: Vec::new(),
            extra_roots: Vec::new(),
            reserved_roots: Vec::new(),
            name_of_symbol: Vec::new(),
            reserved_sets: FxHashMap::default(),
        }
    }

    /// Convert a master `InputGrammar` into the flat pool. SETUP ONLY - excluded
    /// from the timed pass (mirrors how, in a real pipeline, the frontend would
    /// build the pool directly, with strings already interned).
    pub fn from_grammar(g: &InputGrammar) -> Self {
        let mut p = Self::empty();
        // Intern names FIRST so name StrIds occupy the dense low range: variable
        // `i` gets StrId `i+1`, so `name_of_symbol[sid.idx()]` resolves directly.
        for v in &g.variables {
            p.intern_str(&v.name);
        }
        let mut name_of_symbol: Vec<Symbol> =
            (0..g.variables.len()).map(Symbol::non_terminal).collect();
        // An external name not already a variable gets the next StrId and is
        // appended in order; a dup resolves to an existing StrId and is skipped,
        // matching master's variables-before-externals scan.
        for (i, e) in g.external_tokens.iter().enumerate() {
            if let Rule::NamedSymbol(name) = e {
                let sid = p.intern_str(name);
                if sid.idx() == name_of_symbol.len() {
                    name_of_symbol.push(Symbol::external(i));
                }
            }
        }
        p.name_of_symbol = name_of_symbol;
        for v in &g.variables {
            let r = p.add(&v.rule);
            p.roots.push(r);
        }
        for e in &g.external_tokens {
            let r = p.add(e);
            p.ext_roots.push(r);
        }
        for r in &g.extra_symbols {
            let id = p.add(r);
            p.extra_roots.push(id);
        }
        for ctx in &g.reserved_words {
            for r in &ctx.reserved_words {
                let id = p.add(r);
                p.reserved_roots.push(id);
            }
        }
        p
    }

    /// Convert flatten's input into the flat pool. SETUP ONLY, like
    /// [`Self::from_grammar`]. Reserved-set names resolve to set indices here.
    pub fn from_extracted(g: &ExtractedSyntaxGrammar) -> Self {
        let mut p = Self::empty();
        // Last wins on duplicate names, matching master's map build.
        for (i, set) in g.reserved_word_sets.iter().enumerate() {
            let sid = p.intern_str(&set.name);
            p.reserved_sets.insert(sid, i as u16);
        }
        for v in &g.variables {
            let r = p.add(&v.rule);
            p.roots.push(r);
        }
        p
    }

    fn intern_str(&mut self, s: &str) -> StrId {
        if let Some(&id) = self.str_ids.get(s) {
            return id;
        }
        let id = StrId(NonZeroU32::new(self.strs.len() as u32 + 1).unwrap());
        self.strs.push(s.into());
        self.str_ids.insert(s.into(), id);
        id
    }

    fn str_of(&self, id: StrId) -> &str {
        &self.strs[id.idx()]
    }

    fn add_params(&mut self, m: &crate::rules::MetadataParams) -> ParamsId {
        let prec = match &m.precedence {
            Precedence::None => FPrec::None,
            Precedence::Integer(n) => FPrec::Integer(*n),
            Precedence::Name(s) => FPrec::Name(self.intern_str(s)),
        };
        let alias = m.alias.as_ref().map(|a| FAlias {
            value: self.intern_str(&a.value),
            is_named: a.is_named,
        });
        let field = m.field_name.as_ref().map(|f| self.intern_str(f));
        let id = ParamsId(self.params.len() as u32);
        self.params.push(FParams {
            prec,
            assoc: m.associativity,
            dynamic_precedence: m.dynamic_precedence,
            alias,
            field,
        });
        id
    }

    fn add(&mut self, rule: &Rule) -> NodeId {
        let node = match rule {
            Rule::Blank => FNode::Blank,
            Rule::String(s) => FNode::String(self.intern_str(s)),
            Rule::Pattern(a, b) => FNode::Pattern(self.intern_str(a), self.intern_str(b)),
            Rule::NamedSymbol(n) => FNode::NamedSymbol(self.intern_str(n)),
            Rule::Symbol(s) => FNode::Sym {
                kind: s.kind,
                index: s.index as u32,
            },
            Rule::Seq(elems) => FNode::Seq(self.add_children(elems)),
            Rule::Choice(elems) => FNode::Choice(self.add_children(elems)),
            Rule::Repeat(c) => FNode::Repeat(self.add(c)),
            Rule::Metadata { params, rule } => FNode::Metadata {
                params: self.add_params(params),
                rule: self.add(rule),
            },
            Rule::Reserved { rule, context_name } => {
                let context_name = self.intern_str(context_name);
                FNode::Reserved {
                    rule: self.add(rule),
                    ctx: context_name,
                }
            }
        };
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    fn add_children(&mut self, elems: &[Rule]) -> NodeRange {
        // Convert each child subtree first (each returns its root id), then lay
        // the ids down contiguously. The temp Vec is setup-only.
        let ids: Vec<NodeId> = elems.iter().map(|e| self.add(e)).collect();
        let start = self.children.len() as u32;
        debug_assert!(start < (1 << 30));
        self.children.extend_from_slice(&ids);
        NodeRange {
            start,
            len: ids.len() as u32,
        }
    }

    /// The timed intern pass: resolve every `NamedSymbol` to a symbol in place,
    /// over all root sets. Recursive, in-place, zero allocation.
    pub fn intern_symbols(&mut self) {
        for i in 0..self.roots.len() {
            self.resolve(self.roots[i]);
        }
        for i in 0..self.ext_roots.len() {
            self.resolve(self.ext_roots[i]);
        }
        for i in 0..self.extra_roots.len() {
            self.resolve(self.extra_roots[i]);
        }
        for i in 0..self.reserved_roots.len() {
            self.resolve(self.reserved_roots[i]);
        }
    }

    fn resolve(&mut self, id: NodeId) {
        let idx = id.0 as usize;
        match self.nodes[idx] {
            FNode::NamedSymbol(sid) => {
                if let Some(&s) = self.name_of_symbol.get(sid.idx()) {
                    self.nodes[idx] = FNode::Sym {
                        kind: s.kind,
                        index: s.index as u32,
                    };
                }
            }
            FNode::Seq(r) | FNode::Choice(r) => {
                for i in 0..r.len {
                    let c = self.children[(r.start + i) as usize];
                    self.resolve(c);
                }
            }
            FNode::Repeat(c)
            | FNode::Metadata { rule: c, .. }
            | FNode::Reserved { rule: c, .. } => self.resolve(c),
            _ => {}
        }
    }
}

// --- flatten ---

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
    fn seq(r: NodeRange) -> Self {
        debug_assert!(r.start + r.len < (1 << 30));
        Self(r.start, r.start + r.len)
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

/// One pooled production step, 20 bytes and `Copy`. `flags` packs the symbol
/// kind (bits 0..3), precedence tag (3..5: none/int/name), associativity
/// (5..7: none/left/right), and alias-is-named (7). `prec_val` holds the
/// integer value or the raw name `StrId` per the tag; `alias`/`field` are raw
/// `StrId`s with 0 = none.
#[derive(Clone, Copy, PartialEq, Eq)]
struct FStep {
    sym_index: u32,
    prec_val: i32,
    alias: u32,
    field: u32,
    reserved: u16,
    flags: u8,
}

const _: () = assert!(std::mem::size_of::<FStep>() == 20);

const fn sym_type_from_bits(bits: u8) -> SymbolType {
    match bits {
        0 => SymbolType::External,
        1 => SymbolType::End,
        2 => SymbolType::EndOfNonTerminalExtra,
        3 => SymbolType::Terminal,
        _ => SymbolType::NonTerminal,
    }
}

/// Pooled flatten output: one backing store for steps, productions as ranges
/// into it, per-variable production ranges.
#[derive(Clone, Copy)]
struct FProd {
    steps_start: u32,
    steps_len: u32,
    dynamic_precedence: i32,
}

#[derive(Default)]
pub struct FlatOut {
    steps: Vec<FStep>,
    productions: Vec<FProd>,
    var_prods: Vec<(u32, u32)>,
}

impl FlatOut {
    fn clear(&mut self) {
        self.steps.clear();
        self.productions.clear();
        self.var_prods.clear();
    }
}

/// Reusable flatten scratch. Everything is truncate-restore; capacity persists
/// across variables and runs.
#[derive(Default)]
pub struct FlattenState {
    frames: Vec<Frame>,
    steps: Vec<FStep>,
    prec: Vec<FPrec>,
    assoc: Vec<Associativity>,
    alias: Vec<FAlias>,
    field: Vec<StrId>,
    reserved: Vec<u16>,
    /// Fork snapshots, stacked as segments. A branch can consume context
    /// pushed BEFORE the fork (an enclosing Metadata/Reserved exit pops it),
    /// so every stack that exits pop through needs a full snapshot; only
    /// `steps` grows monotonically within a branch and truncates back.
    snap_frames: Vec<Frame>,
    snap_prec: Vec<FPrec>,
    snap_assoc: Vec<Associativity>,
    snap_alias: Vec<FAlias>,
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

    /// Snapshot everything a branch can consume; returns the segment offsets.
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

    /// Restore to the fork point from the snapshot segments.
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

    /// Pop the fork's snapshot segments.
    fn fork_done(&mut self, mark: Mark) {
        self.snap_frames.truncate(mark.frames as usize);
        self.snap_prec.truncate(mark.prec as usize);
        self.snap_assoc.truncate(mark.assoc as usize);
        self.snap_alias.truncate(mark.alias as usize);
        self.snap_field.truncate(mark.field as usize);
        self.snap_reserved.truncate(mark.reserved as usize);
    }

    fn push_step(&mut self, kind: SymbolType, index: u32) {
        let (prec_tag, prec_val) = match self.prec.last() {
            None | Some(FPrec::None) => (0u8, 0i32),
            Some(FPrec::Integer(n)) => (1, *n),
            Some(FPrec::Name(sid)) => (2, sid.raw() as i32),
        };
        let assoc_bits = match self.assoc.last() {
            None => 0u8,
            Some(Associativity::Left) => 1,
            Some(Associativity::Right) => 2,
        };
        let (alias, named) = self
            .alias
            .last()
            .map_or((0, 0u8), |a| (a.value.raw(), u8::from(a.is_named)));
        self.steps.push(FStep {
            sym_index: index,
            prec_val,
            alias,
            field: self.field.last().map_or(0, |f| f.raw()),
            reserved: self.reserved.last().copied().unwrap_or(0),
            flags: (kind as u8) | (prec_tag << 3) | (assoc_bits << 5) | (named << 7),
        });
    }

    /// `true` if nothing after the current point can push a step: every `Seq`
    /// frame in the remaining continuation is exhausted. Equivalent to master's
    /// top-down `at_end`.
    fn at_end(&self) -> bool {
        self.frames
            .iter()
            .all(|f| f.tag() != FR_SEQ || f.seq_exhausted())
    }

    /// Master's `Metadata` pop branch: pop what was pushed; if a step was
    /// pushed in the subtree and more steps can follow, rewrite the last step's
    /// precedence/associativity to the outer context.
    fn meta_exit(&mut self, flags: u32, steps_before: u32) {
        let did_push = self.steps.len() as u32 > steps_before;
        let fixup = did_push && !self.at_end();
        if flags & MP_PREC != 0 {
            self.prec.pop();
            if fixup {
                let (tag, val) = match self.prec.last() {
                    None | Some(FPrec::None) => (0u8, 0i32),
                    Some(FPrec::Integer(n)) => (1, *n),
                    Some(FPrec::Name(sid)) => (2, sid.raw() as i32),
                };
                let step = self.steps.last_mut().unwrap();
                step.prec_val = val;
                step.flags = (step.flags & !(0b11 << 3)) | (tag << 3);
            }
        }
        if flags & MP_ASSOC != 0 {
            self.assoc.pop();
            if fixup {
                let bits = match self.assoc.last() {
                    None => 0u8,
                    Some(Associativity::Left) => 1,
                    Some(Associativity::Right) => 2,
                };
                let step = self.steps.last_mut().unwrap();
                step.flags = (step.flags & !(0b11 << 5)) | (bits << 5);
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

impl Pool {
    /// The timed flatten pass: variables -> pooled productions, mirroring
    /// master `flatten_grammar` (including its post-checks) minus the final
    /// `SyntaxGrammar` assembly of pass-through fields.
    pub fn flatten(
        &self,
        g: &ExtractedSyntaxGrammar,
        st: &mut FlattenState,
        out: &mut FlatOut,
    ) -> Result<(), FlattenGrammarError> {
        out.clear();
        for &root in &self.roots {
            let prod_start = out.productions.len() as u32;
            st.reset();
            self.run_path(root, st, out, prod_start)?;
            out.var_prods.push((prod_start, out.productions.len() as u32));
        }
        self.check(g, out)?;
        Ok(())
    }

    /// Mirror of master's post-flatten checks (empty-string and
    /// recursive-inline), same iteration order and same O(V * steps) scan.
    fn check(
        &self,
        g: &ExtractedSyntaxGrammar,
        out: &FlatOut,
    ) -> Result<(), FlattenGrammarError> {
        let nt_bits = SymbolType::NonTerminal as u8;
        for (i, &(ps, pe)) in out.var_prods.iter().enumerate() {
            let used = out.steps.iter().any(|s| {
                s.sym_index as usize == i && (s.flags & 0b111) == nt_bits
            });
            let inlined = g.variables_to_inline.contains(&Symbol::non_terminal(i));
            for p in &out.productions[ps as usize..pe as usize] {
                if used && p.steps_len == 0 {
                    Err(FlattenGrammarError::EmptyString(g.variables[i].name.clone()))?;
                }
                if inlined {
                    let steps = &out.steps[p.steps_start as usize
                        ..(p.steps_start + p.steps_len) as usize];
                    if steps.iter().any(|s| {
                        s.sym_index as usize == i && (s.flags & 0b111) == nt_bits
                    }) {
                        Err(FlattenGrammarError::RecursiveInline(
                            g.variables[i].name.clone(),
                        ))?;
                    }
                }
            }
        }
        Ok(())
    }

    /// Enter `node` and, unless it forked, drive the continuation to
    /// completion (emitting one production per completed path).
    fn run_path(
        &self,
        node: NodeId,
        st: &mut FlattenState,
        out: &mut FlatOut,
        prod_start: u32,
    ) -> Result<(), FlattenGrammarError> {
        if self.enter(node, st, out, prod_start)? {
            self.drive(st, out, prod_start)?;
        }
        Ok(())
    }

    /// Process one node. Returns `false` if the node forked (a `Choice` drove
    /// every branch to completion, emitting; the caller's continuation is
    /// consumed). Unary chains (`Metadata`/`Reserved`) iterate in place.
    fn enter(
        &self,
        mut node: NodeId,
        st: &mut FlattenState,
        out: &mut FlatOut,
        prod_start: u32,
    ) -> Result<bool, FlattenGrammarError> {
        loop {
            match self.nodes[node.0 as usize] {
                FNode::Sym { kind, index } => {
                    st.push_step(kind, index);
                    return Ok(true);
                }
                FNode::Seq(r) => {
                    st.frames.push(Frame::seq(r));
                    return Ok(true);
                }
                FNode::Metadata { params, rule } => {
                    let p = self.params[params.0 as usize];
                    let mut flags = 0;
                    if p.prec != FPrec::None {
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
                FNode::Reserved { rule, ctx } => {
                    let Some(&set) = self.reserved_sets.get(&ctx) else {
                        return Err(FlattenGrammarError::NoReservedWordSet(
                            self.str_of(ctx).to_string(),
                        ));
                    };
                    st.reserved.push(set);
                    st.frames.push(Frame::reserved());
                    node = rule;
                }
                FNode::Choice(r) => {
                    // Fork: snapshot the full continuation and context, drive
                    // each branch to completion, restore between branches.
                    let mark = st.fork_save();
                    for i in r.start..r.start + r.len {
                        let c = self.children[i as usize];
                        self.run_path(c, st, out, prod_start)?;
                        st.restore(mark);
                    }
                    st.fork_done(mark);
                    return Ok(false);
                }
                // Blank pushes nothing; String/Pattern/NamedSymbol/Repeat
                // cannot appear post-extract/expand and match master's
                // silent no-op arm (`_ => Ok(false)`).
                _ => return Ok(true),
            }
        }
    }

    /// Consume frames until the continuation is exhausted, then emit. Stops
    /// without emitting if a deeper fork took over (it emitted for us).
    fn drive(
        &self,
        st: &mut FlattenState,
        out: &mut FlatOut,
        prod_start: u32,
    ) -> Result<(), FlattenGrammarError> {
        loop {
            let Some(&top) = st.frames.last() else {
                self.emit(st, out, prod_start);
                return Ok(());
            };
            match top.tag() {
                FR_SEQ => {
                    if top.seq_exhausted() {
                        st.frames.pop();
                    } else {
                        let cur = top.0 & FR_PAYLOAD;
                        st.frames.last_mut().unwrap().0 += 1;
                        let c = self.children[cur as usize];
                        if !self.enter(c, st, out, prod_start)? {
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
    /// has an identical one (master's order-preserving `contains` dedup).
    fn emit(&self, st: &mut FlattenState, out: &mut FlatOut, prod_start: u32) {
        for p in &out.productions[prod_start as usize..] {
            if p.dynamic_precedence == st.dyn_prec
                && out.steps[p.steps_start as usize..(p.steps_start + p.steps_len) as usize]
                    == st.steps[..]
            {
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

    /// Materialize pooled output into master `SyntaxVariable`s. UNTIMED - only
    /// for the equality assert against master's flatten.
    pub fn materialize(&self, out: &FlatOut, vars: &[Variable]) -> Vec<SyntaxVariable> {
        vars.iter()
            .zip(&out.var_prods)
            .map(|(v, &(ps, pe))| SyntaxVariable {
                name: v.name.clone(),
                kind: v.kind,
                productions: out.productions[ps as usize..pe as usize]
                    .iter()
                    .map(|p| Production {
                        dynamic_precedence: p.dynamic_precedence,
                        steps: out.steps
                            [p.steps_start as usize..(p.steps_start + p.steps_len) as usize]
                            .iter()
                            .map(|s| self.step_to_master(s))
                            .collect(),
                    })
                    .collect(),
            })
            .collect()
    }

    fn step_to_master(&self, s: &FStep) -> ProductionStep {
        let str_at = |raw: u32| self.strs[raw as usize - 1].to_string();
        ProductionStep {
            symbol: Symbol {
                kind: sym_type_from_bits(s.flags & 0b111),
                index: s.sym_index as usize,
            },
            precedence: match (s.flags >> 3) & 0b11 {
                0 => Precedence::None,
                1 => Precedence::Integer(s.prec_val),
                _ => Precedence::Name(str_at(s.prec_val as u32)),
            },
            associativity: match (s.flags >> 5) & 0b11 {
                0 => None,
                1 => Some(Associativity::Left),
                _ => Some(Associativity::Right),
            },
            alias: (s.alias != 0).then(|| Alias {
                value: str_at(s.alias),
                is_named: s.flags & (1 << 7) != 0,
            }),
            field_name: (s.field != 0).then(|| str_at(s.field)),
            reserved_word_set_id: ReservedWordSetId(s.reserved as usize),
        }
    }
}
