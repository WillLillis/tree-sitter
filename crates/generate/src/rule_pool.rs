//! Flat, pool-indexed rule IR for the grammar backend.
//!
//! Nodes live in one append-only arena and reference children by index, so
//! passes are iterative pool walks with in-place rewrites instead of
//! recursive `Rule` tree rebuilds. String payloads intern to a [`StrId`];
//! `Metadata` params live out of line so [`Node`] stays 12 bytes.
//!
//! `add_rule`/`rule` are transitional bridges to [`Rule`] for per-pass A/B
//! verification while the backend converts; they are deleted once every pass
//! and both frontends produce and consume the pool directly.

use std::num::NonZeroU32;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::grammars::{InputGrammar, PrecedenceEntry};
use crate::rules::{Associativity, MetadataParams, Precedence, Rule, Symbol, SymbolType};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(u32);

impl NodeId {
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Interned string id: 1-based index into the pool's string table
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct StrId(NonZeroU32);

impl StrId {
    /// Dense 0-based index (ids are 1-based).
    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize - 1
    }

    /// The raw 1-based id, for packed encodings where 0 means "none".
    #[must_use]
    pub const fn raw(self) -> u32 {
        self.0.get()
    }

    /// Inverse of [`Self::raw`]. Caller must pass a value produced by `raw`.
    #[must_use]
    pub const fn from_raw(raw: u32) -> Self {
        Self(NonZeroU32::new(raw).unwrap())
    }
}

/// Index into [`RulePool::params`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ParamsId(u32);

/// A `[start, start + len)` slice of [`RulePool::children`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeRange {
    pub start: u32,
    pub len: u32,
}

impl NodeRange {
    #[must_use]
    pub const fn as_range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

/// A pooled rule node. `String`/`Pattern`/`NamedSymbol`/`Repeat` exist only
/// before the extract/expand passes run; later passes treat them as
/// unreachable.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Node {
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

const _: () = assert!(std::mem::size_of::<Node>() == 12);

/// [`Precedence`] with the name interned. `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum Prec {
    #[default]
    None,
    Integer(i32),
    Name(StrId),
}

/// [`crate::rules::Alias`] with the value interned. `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Alias {
    pub value: StrId,
    pub is_named: bool,
}

/// [`MetadataParams`] with interned payloads. `Copy`; side pool keeps
/// [`Node`] small.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Params {
    pub prec: Prec,
    pub assoc: Option<Associativity>,
    pub dynamic_precedence: i32,
    pub alias: Option<Alias>,
    pub field: Option<StrId>,
    pub is_token: bool,
    pub is_main_token: bool,
}

/// The arena: nodes, their pooled children/params, and the string interner.
/// Append-only; passes rewrite nodes in place and may orphan subtrees.
#[derive(Clone, Default)]
pub struct RulePool {
    nodes: Vec<Node>,
    children: Vec<NodeId>,
    params: Vec<Params>,
    strs: Vec<Rc<str>>,
    str_ids: FxHashMap<Rc<str>, StrId>,
}

impl RulePool {
    #[must_use]
    pub fn node(&self, id: NodeId) -> Node {
        self.nodes[id.index()]
    }

    pub fn set_node(&mut self, id: NodeId, node: Node) {
        self.nodes[id.index()] = node;
    }

    pub fn push_node(&mut self, node: Node) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    #[must_use]
    pub fn child_slice(&self, range: NodeRange) -> &[NodeId] {
        &self.children[range.as_range()]
    }

    pub fn push_children(&mut self, ids: &[NodeId]) -> NodeRange {
        let start = self.children.len() as u32;
        debug_assert!(start + ids.len() as u32 <= (1 << 30));
        self.children.extend_from_slice(ids);
        NodeRange {
            start,
            len: ids.len() as u32,
        }
    }

    #[must_use]
    pub fn params(&self, id: ParamsId) -> Params {
        self.params[id.0 as usize]
    }

    pub fn push_params(&mut self, params: Params) -> ParamsId {
        let id = ParamsId(self.params.len() as u32);
        self.params.push(params);
        id
    }

    pub fn set_params(&mut self, id: ParamsId, params: Params) {
        self.params[id.0 as usize] = params;
    }

    // Rule-construction methods mirroring the `Rule` smart constructors, for
    // frontends that build pool nodes directly. Metadata-bearing wrappers
    // merge their params into an existing non-token `Metadata` node instead
    // of nesting a second one.

    fn metadata_with(&mut self, content: NodeId, f: impl FnOnce(&mut Params)) -> NodeId {
        if let Node::Metadata { params, .. } = self.node(content) {
            let mut p = self.params(params);
            if !p.is_token {
                f(&mut p);
                self.set_params(params, p);
                return content;
            }
        }
        let mut p = Params::default();
        f(&mut p);
        let params = self.push_params(p);
        self.push_node(Node::Metadata {
            params,
            rule: content,
        })
    }

    pub fn blank(&mut self) -> NodeId {
        self.push_node(Node::Blank)
    }

    pub fn string(&mut self, value: &str) -> NodeId {
        let id = self.intern(value);
        self.push_node(Node::String(id))
    }

    pub fn pattern(&mut self, value: &str, flags: &str) -> NodeId {
        let value = self.intern(value);
        let flags = self.intern(flags);
        self.push_node(Node::Pattern(value, flags))
    }

    pub fn named_symbol(&mut self, name: &str) -> NodeId {
        let id = self.intern(name);
        self.push_node(Node::NamedSymbol(id))
    }

    pub fn field(&mut self, name: StrId, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| p.field = Some(name))
    }

    pub fn alias(&mut self, content: NodeId, value: StrId, is_named: bool) -> NodeId {
        self.metadata_with(content, |p| p.alias = Some(Alias { value, is_named }))
    }

    pub fn token(&mut self, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| p.is_token = true)
    }

    pub fn immediate_token(&mut self, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| {
            p.is_token = true;
            p.is_main_token = true;
        })
    }

    pub fn prec(&mut self, value: Prec, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| p.prec = value)
    }

    pub fn prec_left(&mut self, value: Prec, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| {
            p.assoc = Some(Associativity::Left);
            p.prec = value;
        })
    }

    pub fn prec_right(&mut self, value: Prec, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| {
            p.assoc = Some(Associativity::Right);
            p.prec = value;
        })
    }

    pub fn prec_dynamic(&mut self, value: i32, content: NodeId) -> NodeId {
        self.metadata_with(content, |p| p.dynamic_precedence = value)
    }

    pub fn repeat(&mut self, content: NodeId) -> NodeId {
        self.push_node(Node::Repeat(content))
    }

    pub fn seq(&mut self, ids: &[NodeId]) -> NodeId {
        let range = self.push_children(ids);
        self.push_node(Node::Seq(range))
    }

    /// Mirror of `Rule::choice`: flatten nested choices and dedup
    /// structurally, keeping a `Choice` node even for a single element.
    pub fn choice(&mut self, ids: &[NodeId]) -> NodeId {
        let mut elements: Vec<NodeId> = Vec::with_capacity(ids.len());
        let mut stack: Vec<NodeId> = Vec::with_capacity(ids.len());
        stack.extend(ids.iter().rev());
        while let Some(id) = stack.pop() {
            if let Node::Choice(range) = self.node(id) {
                let base = stack.len();
                stack.extend_from_slice(self.child_slice(range));
                stack[base..].reverse();
            } else if !elements.iter().any(|&e| self.subtree_eq(e, id)) {
                elements.push(id);
            }
        }
        let range = self.push_children(&elements);
        self.push_node(Node::Choice(range))
    }

    pub fn reserved(&mut self, content: NodeId, ctx: StrId) -> NodeId {
        self.push_node(Node::Reserved { rule: content, ctx })
    }

    pub fn intern(&mut self, s: &str) -> StrId {
        if let Some(&id) = self.str_ids.get(s) {
            return id;
        }
        let owned: Rc<str> = Rc::from(s);
        let id = StrId(NonZeroU32::new(self.strs.len() as u32 + 1).unwrap());
        self.strs.push(Rc::clone(&owned));
        self.str_ids.insert(owned, id);
        id
    }

    #[must_use]
    pub fn resolve(&self, id: StrId) -> &str {
        &self.strs[id.index()]
    }

    /// The interned strings, indexed by `StrId::index`. Cloning is cheap
    /// (`Rc` bumps); the pooled `SyntaxGrammar` carries this table so its
    /// `FStep`s stay resolvable after the pool is gone.
    #[must_use]
    pub fn strs(&self) -> &[Rc<str>] {
        &self.strs
    }

    /// Number of unique interned strings.
    #[allow(dead_code)] // TEMP: test-only until the container flip
    #[must_use]
    pub const fn str_count(&self) -> usize {
        self.strs.len()
    }

    /// Structural hash of a subtree: node tags plus interned/`Copy` payloads,
    /// so content-equal subtrees hash equal regardless of their node ids.
    #[must_use]
    pub fn subtree_hash(&self, root: NodeId) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = rustc_hash::FxHasher::default();
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            let node = self.node(id);
            std::mem::discriminant(&node).hash(&mut hasher);
            match node {
                Node::Blank => {}
                Node::String(s) | Node::NamedSymbol(s) => s.hash(&mut hasher),
                Node::Pattern(p, f) => {
                    p.hash(&mut hasher);
                    f.hash(&mut hasher);
                }
                Node::Sym { kind, index } => {
                    (kind as u8).hash(&mut hasher);
                    index.hash(&mut hasher);
                }
                Node::Seq(range) | Node::Choice(range) => {
                    hasher.write_u32(range.len);
                    let base = stack.len();
                    stack.extend_from_slice(self.child_slice(range));
                    stack[base..].reverse();
                }
                Node::Repeat(inner) => stack.push(inner),
                Node::Metadata { params, rule } => {
                    let p = self.params(params);
                    (p.prec, p.assoc, p.dynamic_precedence).hash(&mut hasher);
                    p.alias.map(|a| (a.value, a.is_named)).hash(&mut hasher);
                    (p.field, p.is_token, p.is_main_token).hash(&mut hasher);
                    stack.push(rule);
                }
                Node::Reserved { rule, ctx } => {
                    ctx.hash(&mut hasher);
                    stack.push(rule);
                }
            }
        }
        hasher.finish()
    }

    /// Structural equality of two subtrees (content, not node identity).
    #[must_use]
    pub fn subtree_eq(&self, a: NodeId, b: NodeId) -> bool {
        let mut stack = vec![(a, b)];
        while let Some((a, b)) = stack.pop() {
            match (self.node(a), self.node(b)) {
                (Node::Blank, Node::Blank) => {}
                (Node::String(x), Node::String(y))
                | (Node::NamedSymbol(x), Node::NamedSymbol(y)) => {
                    if x != y {
                        return false;
                    }
                }
                (Node::Pattern(p1, f1), Node::Pattern(p2, f2)) => {
                    if p1 != p2 || f1 != f2 {
                        return false;
                    }
                }
                (n1 @ Node::Sym { .. }, n2 @ Node::Sym { .. }) => {
                    if n1 != n2 {
                        return false;
                    }
                }
                (Node::Seq(r1), Node::Seq(r2)) | (Node::Choice(r1), Node::Choice(r2)) => {
                    if r1.len != r2.len {
                        return false;
                    }
                    stack.extend(
                        self.child_slice(r1)
                            .iter()
                            .copied()
                            .zip(self.child_slice(r2).iter().copied()),
                    );
                }
                (Node::Repeat(i1), Node::Repeat(i2)) => stack.push((i1, i2)),
                (
                    Node::Metadata {
                        params: p1,
                        rule: r1,
                    },
                    Node::Metadata {
                        params: p2,
                        rule: r2,
                    },
                ) => {
                    if self.params(p1) != self.params(p2) {
                        return false;
                    }
                    stack.push((r1, r2));
                }
                (Node::Reserved { rule: r1, ctx: c1 }, Node::Reserved { rule: r2, ctx: c2 }) => {
                    if c1 != c2 {
                        return false;
                    }
                    stack.push((r1, r2));
                }
                _ => return false,
            }
        }
        true
    }

    // --- transitional Rule bridges (deleted at series end) ---

    /// Import a [`Rule`] tree. Bottom-up, so a parent's children precede it.
    pub fn add_rule(&mut self, rule: &Rule) -> NodeId {
        let node = match rule {
            Rule::Blank => Node::Blank,
            Rule::String(s) => Node::String(self.intern(s)),
            Rule::Pattern(p, f) => Node::Pattern(self.intern(p), self.intern(f)),
            Rule::NamedSymbol(n) => Node::NamedSymbol(self.intern(n)),
            Rule::Symbol(s) => Node::Sym {
                kind: s.kind,
                index: s.index as u32,
            },
            Rule::Seq(elems) => Node::Seq(self.add_rule_children(elems)),
            Rule::Choice(elems) => Node::Choice(self.add_rule_children(elems)),
            Rule::Repeat(inner) => Node::Repeat(self.add_rule(inner)),
            Rule::Metadata { params, rule } => Node::Metadata {
                params: self.add_metadata_params(params),
                rule: self.add_rule(rule),
            },
            Rule::Reserved { rule, context_name } => {
                let ctx = self.intern(context_name);
                Node::Reserved {
                    rule: self.add_rule(rule),
                    ctx,
                }
            }
        };
        self.push_node(node)
    }

    fn add_rule_children(&mut self, elems: &[Rule]) -> NodeRange {
        let ids: Vec<NodeId> = elems.iter().map(|e| self.add_rule(e)).collect();
        self.push_children(&ids)
    }

    fn add_metadata_params(&mut self, m: &MetadataParams) -> ParamsId {
        let prec = match &m.precedence {
            Precedence::None => Prec::None,
            Precedence::Integer(n) => Prec::Integer(*n),
            Precedence::Name(s) => Prec::Name(self.intern(s)),
        };
        let alias = m.alias.as_ref().map(|a| Alias {
            value: self.intern(&a.value),
            is_named: a.is_named,
        });
        let field = m.field_name.as_ref().map(|f| self.intern(f));
        self.push_params(Params {
            prec,
            assoc: m.associativity,
            dynamic_precedence: m.dynamic_precedence,
            alias,
            field,
            is_token: m.is_token,
            is_main_token: m.is_main_token,
        })
    }

    /// Materialize a pooled subtree back into a [`Rule`] (A/B verification).
    #[must_use]
    pub fn rule(&self, id: NodeId) -> Rule {
        match self.node(id) {
            Node::Blank => Rule::Blank,
            Node::String(s) => Rule::String(self.resolve(s).to_string()),
            Node::Pattern(p, f) => {
                Rule::Pattern(self.resolve(p).to_string(), self.resolve(f).to_string())
            }
            Node::NamedSymbol(n) => Rule::NamedSymbol(self.resolve(n).to_string()),
            Node::Sym { kind, index } => Rule::Symbol(Symbol {
                kind,
                index: index as usize,
            }),
            Node::Seq(range) => Rule::Seq(self.rule_children(range)),
            Node::Choice(range) => Rule::Choice(self.rule_children(range)),
            Node::Repeat(inner) => Rule::Repeat(Box::new(self.rule(inner))),
            Node::Metadata { params, rule } => Rule::Metadata {
                params: self.metadata_params(params),
                rule: Box::new(self.rule(rule)),
            },
            Node::Reserved { rule, ctx } => Rule::Reserved {
                rule: Box::new(self.rule(rule)),
                context_name: self.resolve(ctx).to_string(),
            },
        }
    }

    fn rule_children(&self, range: NodeRange) -> Vec<Rule> {
        self.child_slice(range)
            .iter()
            .map(|&c| self.rule(c))
            .collect()
    }

    fn metadata_params(&self, id: ParamsId) -> MetadataParams {
        let p = self.params(id);
        MetadataParams {
            precedence: match p.prec {
                Prec::None => Precedence::None,
                Prec::Integer(n) => Precedence::Integer(n),
                Prec::Name(s) => Precedence::Name(self.resolve(s).to_string()),
            },
            dynamic_precedence: p.dynamic_precedence,
            associativity: p.assoc,
            is_token: p.is_token,
            is_main_token: p.is_main_token,
            alias: p.alias.map(|a| crate::rules::Alias {
                value: self.resolve(a.value).to_string(),
                is_named: a.is_named,
            }),
            field_name: p.field.map(|f| self.resolve(f).to_string()),
        }
    }
}

/// One flattened production step, 20 bytes and `Copy`. `flags` packs the
/// symbol kind (bits 0..3), precedence tag (3..5: none/integer/name),
/// associativity (5..7: none/left/right), and alias-is-named (7). `prec_val`
/// holds the integer value or the raw name `StrId` per the tag;
/// `alias`/`field` are raw `StrId`s with 0 meaning none.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FStep {
    pub sym_index: u32,
    pub prec_val: i32,
    pub alias: u32,
    pub field: u32,
    pub reserved: u16,
    pub flags: u8,
}

const _: () = assert!(std::mem::size_of::<FStep>() == 20);

pub const FSTEP_PREC_INTEGER: u8 = 1 << 3;
pub const FSTEP_PREC_NAME: u8 = 2 << 3;
pub const FSTEP_ASSOC_LEFT: u8 = 1 << 5;
pub const FSTEP_ASSOC_RIGHT: u8 = 2 << 5;
pub const FSTEP_ALIAS_NAMED: u8 = 1 << 7;
pub const FSTEP_PREC_MASK: u8 = 0b11 << 3;
pub const FSTEP_ASSOC_MASK: u8 = 0b11 << 5;
/// `FStep::reserved` sentinel: no reserved word set at all (as opposed to id
/// 0, the default set). Only the augmented start production carries it; it
/// must never index the reserved-sets table.
pub const NO_RESERVED_WORDS: u16 = u16::MAX;

impl FStep {
    /// The one packing site. Canonical: absent parts zero their value bits
    /// too (`prec_val`/`alias`/`field`), so packed steps compare with `==`.
    #[must_use]
    pub fn pack(
        symbol: Symbol,
        prec: Prec,
        assoc: Option<Associativity>,
        alias: Option<Alias>,
        field: Option<StrId>,
        reserved: u16,
    ) -> Self {
        let (prec_bits, prec_val) = match prec {
            Prec::None => (0u8, 0i32),
            Prec::Integer(n) => (FSTEP_PREC_INTEGER, n),
            Prec::Name(sid) => (FSTEP_PREC_NAME, sid.raw() as i32),
        };
        let assoc_bits = match assoc {
            None => 0u8,
            Some(Associativity::Left) => FSTEP_ASSOC_LEFT,
            Some(Associativity::Right) => FSTEP_ASSOC_RIGHT,
        };
        let (alias_raw, named_bit) = alias.map_or((0, 0u8), |a| {
            (
                a.value.raw(),
                if a.is_named { FSTEP_ALIAS_NAMED } else { 0 },
            )
        });
        Self {
            sym_index: symbol.index as u32,
            prec_val,
            alias: alias_raw,
            field: field.map_or(0, StrId::raw),
            reserved,
            flags: (symbol.kind as u8) | prec_bits | assoc_bits | named_bit,
        }
    }

    #[must_use]
    pub const fn symbol(self) -> Symbol {
        Symbol {
            kind: match self.flags & 0b111 {
                0 => SymbolType::External,
                1 => SymbolType::End,
                2 => SymbolType::EndOfNonTerminalExtra,
                3 => SymbolType::Terminal,
                _ => SymbolType::NonTerminal,
            },
            index: self.sym_index as usize,
        }
    }

    #[must_use]
    pub const fn precedence(self) -> Prec {
        match self.flags & (0b11 << 3) {
            FSTEP_PREC_INTEGER => Prec::Integer(self.prec_val),
            FSTEP_PREC_NAME => Prec::Name(StrId::from_raw(self.prec_val as u32)),
            _ => Prec::None,
        }
    }

    #[must_use]
    pub const fn associativity(self) -> Option<Associativity> {
        match self.flags & (0b11 << 5) {
            FSTEP_ASSOC_LEFT => Some(Associativity::Left),
            FSTEP_ASSOC_RIGHT => Some(Associativity::Right),
            _ => None,
        }
    }

    #[must_use]
    pub fn alias(self) -> Option<Alias> {
        (self.alias != 0).then(|| Alias {
            value: StrId::from_raw(self.alias),
            is_named: self.flags & FSTEP_ALIAS_NAMED != 0,
        })
    }

    #[must_use]
    pub fn field(self) -> Option<StrId> {
        (self.field != 0).then(|| StrId::from_raw(self.field))
    }
}

/// A flattened production: a step range plus its dynamic precedence.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FProd {
    pub steps_start: u32,
    pub steps_len: u32,
    pub dynamic_precedence: i32,
}

impl FProd {
    #[must_use]
    pub const fn step_range(self) -> std::ops::Range<usize> {
        self.steps_start as usize..(self.steps_start + self.steps_len) as usize
    }
}

/// Pooled flatten output: one backing store for steps, productions as ranges
/// into it, per-variable production ranges. Becomes the `SyntaxGrammar`
/// rule representation at the container flip.
#[derive(Clone, Default)]
pub struct FlatOut {
    pub steps: Vec<FStep>,
    pub productions: Vec<FProd>,
    pub var_prods: Vec<(u32, u32)>,
}

impl FlatOut {
    pub fn clear(&mut self) {
        self.steps.clear();
        self.productions.clear();
        self.var_prods.clear();
    }
}

#[derive(Clone, Copy)]
pub struct PoolVariable {
    pub name: StrId,
    pub root: NodeId,
}

#[derive(Clone)]
pub struct PoolReservedSet {
    pub name: StrId,
    pub roots: Vec<NodeId>,
}

/// Pool-owning input grammar. Both frontends guarantee unique variable
/// names.
#[derive(Clone)]
pub struct PoolGrammar {
    pub pool: RulePool,
    pub name: String,
    pub variables: Vec<PoolVariable>,
    pub external_roots: Vec<NodeId>,
    pub extra_roots: Vec<NodeId>,
    pub reserved_sets: Vec<PoolReservedSet>,
    pub supertype_names: Vec<StrId>,
    pub conflict_names: Vec<Vec<StrId>>,
    pub inline_names: Vec<StrId>,
    pub word_name: Option<StrId>,
    pub precedence_orderings: Vec<Vec<PrecedenceEntry>>,
}

impl PoolGrammar {
    /// Transitional: convert a master [`InputGrammar`]. Deleted when the
    /// frontends build pools directly.
    #[must_use]
    pub fn from_input_grammar(g: &InputGrammar) -> Self {
        let mut pool = RulePool::default();
        let var_names: Vec<StrId> = g.variables.iter().map(|v| pool.intern(&v.name)).collect();
        let variables = g
            .variables
            .iter()
            .zip(var_names)
            .map(|(v, name)| PoolVariable {
                name,
                root: pool.add_rule(&v.rule),
            })
            .collect();
        let external_roots = g.external_tokens.iter().map(|e| pool.add_rule(e)).collect();
        let extra_roots = g.extra_symbols.iter().map(|e| pool.add_rule(e)).collect();
        let reserved_sets = g
            .reserved_words
            .iter()
            .map(|set| PoolReservedSet {
                name: pool.intern(&set.name),
                roots: set
                    .reserved_words
                    .iter()
                    .map(|r| pool.add_rule(r))
                    .collect(),
            })
            .collect();
        let supertype_names = g.supertype_symbols.iter().map(|s| pool.intern(s)).collect();
        let conflict_names = g
            .expected_conflicts
            .iter()
            .map(|c| c.iter().map(|n| pool.intern(n)).collect())
            .collect();
        let inline_names = g
            .variables_to_inline
            .iter()
            .map(|n| pool.intern(n))
            .collect();
        let word_name = g.word_token.as_ref().map(|w| pool.intern(w));
        Self {
            pool,
            name: g.name.clone(),
            variables,
            external_roots,
            extra_roots,
            reserved_sets,
            supertype_names,
            conflict_names,
            inline_names,
            word_name,
            precedence_orderings: g.precedence_orderings.clone(),
        }
    }

    /// Test bridge: materialize back into a master [`InputGrammar`].
    #[cfg(test)]
    #[must_use]
    pub fn to_input_grammar(&self) -> InputGrammar {
        use crate::grammars::{ReservedWordContext, Variable, VariableType};
        let resolve = |&s: &StrId| self.pool.resolve(s).to_string();
        InputGrammar {
            name: self.name.clone(),
            variables: self
                .variables
                .iter()
                .map(|v| Variable {
                    name: resolve(&v.name),
                    kind: VariableType::Named,
                    rule: self.pool.rule(v.root),
                })
                .collect(),
            extra_symbols: self
                .extra_roots
                .iter()
                .map(|&r| self.pool.rule(r))
                .collect(),
            expected_conflicts: self
                .conflict_names
                .iter()
                .map(|c| c.iter().map(resolve).collect())
                .collect(),
            precedence_orderings: self.precedence_orderings.clone(),
            external_tokens: self
                .external_roots
                .iter()
                .map(|&r| self.pool.rule(r))
                .collect(),
            variables_to_inline: self.inline_names.iter().map(resolve).collect(),
            supertype_symbols: self.supertype_names.iter().map(resolve).collect(),
            word_token: self.word_name.as_ref().map(resolve),
            reserved_words: self
                .reserved_sets
                .iter()
                .map(|set| ReservedWordContext {
                    name: resolve(&set.name),
                    reserved_words: set.roots.iter().map(|&r| self.pool.rule(r)).collect(),
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Precedence;

    /// A rule exercising every variant, metadata merging (token boundary
    /// kept separate per `add_metadata`), and nesting.
    fn gnarly() -> Rule {
        Rule::seq(vec![
            Rule::field(
                "f".to_string(),
                Rule::prec_left(
                    Precedence::Name("p".to_string()),
                    Rule::choice(vec![
                        Rule::named("x"),
                        Rule::token(Rule::repeat(Rule::string("t"))),
                        Rule::alias(Rule::pattern("[a-z]", "i"), "a".to_string(), true),
                        Rule::Blank,
                    ]),
                ),
            ),
            Rule::Reserved {
                rule: Box::new(Rule::prec_dynamic(-3, Rule::non_terminal(7))),
                context_name: "ctx".to_string(),
            },
            Rule::terminal(2),
            Rule::external(1),
        ])
    }

    #[test]
    fn rule_roundtrips_through_pool() {
        let mut pool = RulePool::default();
        let original = gnarly();
        let id = pool.add_rule(&original);
        assert_eq!(pool.rule(id), original);
    }

    #[test]
    fn interner_dedups_and_resolves() {
        let mut pool = RulePool::default();
        let a1 = pool.intern("alpha");
        let b = pool.intern("beta");
        let a2 = pool.intern("alpha");
        assert_eq!(a1, a2);
        assert_ne!(a1, b);
        assert_eq!(pool.resolve(a1), "alpha");
        assert_eq!(pool.resolve(b), "beta");
        assert_eq!(pool.str_count(), 2);
        assert_eq!(std::mem::size_of::<Option<StrId>>(), 4);
    }

    #[test]
    fn shared_strings_intern_once() {
        let mut pool = RulePool::default();
        let r = Rule::seq(vec![Rule::string("dup"), Rule::string("dup")]);
        pool.add_rule(&r);
        assert_eq!(pool.str_count(), 1);
    }
}
