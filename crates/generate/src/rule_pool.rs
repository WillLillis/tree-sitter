//! Flat, pool-indexed rule IR for the grammar backend.
//!
//! Nodes live in one append-only arena and reference children by index, so passes
//! are iterative pool walks with in-place rewrites.

use std::num::NonZeroU32;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use crate::rules::{Associativity, Symbol, SymbolType};

/// The parsed grammar.
#[derive(Default)]
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
    pub precedence_orderings: Vec<Vec<PoolPrecedenceEntry>>,
}

/// The arena: nodes, their pooled children/params, and the string interner.
/// Storage is append-only, passes rewrite nodes in place and may orphan subtrees.
#[derive(Clone, Default)]
pub struct RulePool {
    nodes: Vec<Node>,
    children: Vec<NodeId>,
    params: Vec<Params>,
    strs: Vec<Rc<str>>,
    str_ids: FxHashMap<Rc<str>, StrId>,
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

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub enum PoolPrecedenceEntry {
    Name(StrId),
    Symbol(StrId),
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

    pub fn string(&mut self, value: StrId) -> NodeId {
        self.push_node(Node::String(value))
    }

    pub fn pattern(&mut self, value: StrId, flags: StrId) -> NodeId {
        self.push_node(Node::Pattern(value, flags))
    }

    pub fn named_symbol(&mut self, name: StrId) -> NodeId {
        self.push_node(Node::NamedSymbol(name))
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
        self.metadata_with(content, |p| p.dynamic_prec = value)
    }

    pub fn repeat(&mut self, content: NodeId) -> NodeId {
        self.push_node(Node::Repeat(content))
    }

    pub fn seq(&mut self, ids: &[NodeId]) -> NodeId {
        let range = self.push_children(ids);
        self.push_node(Node::Seq(range))
    }

    /// Flatten nested choices and de-dup structurally, keeping a `Choice` node
    /// event for a single element
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

    #[must_use]
    pub fn strs(&self) -> &[Rc<str>] {
        &self.strs
    }

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
                    (p.prec, p.assoc, p.dynamic_prec).hash(&mut hasher);
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
                #[rustfmt::skip]
                (Node::Metadata { params: p1, rule: r1 }, Node::Metadata { params: p2, rule: r2 }) => {
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
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct NodeId(u32);

impl NodeId {
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Interned string id, a 1-based index into the pool's string table
#[derive(Clone, Copy, PartialEq, PartialOrd, Ord, Hash, Debug, Eq)]
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

/// A pooled rule node.
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

const _: () = assert!(std::mem::size_of::<Node>() <= 12);

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub enum Prec {
    #[default]
    None,
    Integer(i32),
    Name(StrId),
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Alias {
    pub value: StrId,
    pub is_named: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct Params {
    pub prec: Prec,
    pub assoc: Option<Associativity>,
    pub dynamic_prec: i32,
    pub alias: Option<Alias>,
    pub field: Option<StrId>,
    pub is_token: bool,
    pub is_main_token: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub struct FStep {
    pub sym_index: u32,
    pub prec_val: i32,
    pub alias: u32,
    pub field: u32,
    pub reserved: u16,
    pub flags: u8,
}

const _: () = assert!(std::mem::size_of::<FStep>() == 20);

#[rustfmt::skip]
impl FStep {
    // `FStep::flags` bit layout
    //     - 0-2 kind (`SymbolType`)
    //     - 3-4 precedence tag (None 00, Integer 01, Name 10)
    //     - 5-6 associativity  (None 00, Left 01, Right 10)
    //     - 7   alias `is_named`
    const KIND_MASK: u8    = 0b0000_0111;
    pub const PREC_INTEGER: u8 = 0b0000_1000;
    pub const PREC_NAME: u8    = 0b0001_0000;
    pub const PREC_MASK: u8    = 0b0001_1000;
    pub const ASSOC_LEFT: u8   = 0b0010_0000;
    pub const ASSOC_RIGHT: u8  = 0b0100_0000;
    pub const ASSOC_MASK: u8   = 0b0110_0000;
    const ALIAS_NAMED: u8  = 0b1000_0000;

    /// `FStep::reserved` sentinel, meaning no reserved word set at all. Only the augmented
    /// start production carries it, and it must never index the reserved-sets table.
    const NO_RESERVED_WORDS: u16 = u16::MAX;
}

impl FStep {
    /// Pack a production step's components into its flat `FStep` representation.
    #[must_use]
    pub fn pack(
        symbol: Symbol,
        prec: Prec,
        assoc: Option<Associativity>,
        alias: Option<Alias>,
        field: Option<StrId>,
        reserved: u16,
    ) -> Self {
        let mut step = Self {
            sym_index: symbol.index as u32,
            reserved,
            flags: symbol.kind as u8,
            ..Default::default()
        };
        step.set_precedence(prec);
        step.set_associativity(assoc);
        step.set_alias(alias);
        step.set_field(field);
        step
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
            Self::PREC_INTEGER => Prec::Integer(self.prec_val),
            Self::PREC_NAME => Prec::Name(StrId::from_raw(self.prec_val as u32)),
            _ => Prec::None,
        }
    }

    pub const fn set_precedence(&mut self, prec: Prec) {
        let (bits, val) = match prec {
            Prec::None => (0, 0),
            Prec::Integer(n) => (Self::PREC_INTEGER, n),
            Prec::Name(sid) => (Self::PREC_NAME, sid.raw() as i32),
        };
        self.prec_val = val;
        self.flags = (self.flags & !Self::PREC_MASK) | bits;
    }

    #[must_use]
    pub const fn associativity(self) -> Option<Associativity> {
        match self.flags & (0b11 << 5) {
            Self::ASSOC_LEFT => Some(Associativity::Left),
            Self::ASSOC_RIGHT => Some(Associativity::Right),
            _ => None,
        }
    }

    pub const fn set_associativity(&mut self, assoc: Option<Associativity>) {
        let bits = match assoc {
            None => 0,
            Some(Associativity::Left) => Self::ASSOC_LEFT,
            Some(Associativity::Right) => Self::ASSOC_RIGHT,
        };
        self.flags = (self.flags & !Self::ASSOC_MASK) | bits;
    }

    #[must_use]
    pub fn alias(self) -> Option<Alias> {
        (self.alias != 0).then(|| Alias {
            value: StrId::from_raw(self.alias),
            is_named: self.flags & Self::ALIAS_NAMED != 0,
        })
    }

    pub fn set_alias(&mut self, alias: Option<Alias>) {
        self.alias = alias.map_or(0, |a| a.value.raw());
        self.set_alias_named(alias.is_some_and(|a| a.is_named));
    }

    pub const fn set_alias_named(&mut self, named: bool) {
        self.flags = (self.flags & !Self::ALIAS_NAMED) | if named { Self::ALIAS_NAMED } else { 0 };
    }

    #[must_use]
    pub fn field(self) -> Option<StrId> {
        (self.field != 0).then(|| StrId::from_raw(self.field))
    }

    pub fn set_field(&mut self, field: Option<StrId>) {
        self.field = field.map_or(0, StrId::raw);
    }
}

/// A flattened production consisting of a step range and its dynamic precedence
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct FProd {
    pub steps_start: u32,
    pub steps_len: u32,
    pub dynamic_prec: i32,
}

impl FProd {
    #[must_use]
    pub const fn step_range(self) -> std::ops::Range<usize> {
        self.steps_start as usize..(self.steps_start + self.steps_len) as usize
    }
}

/// Flattened output: one backing store of steps, productions as `[start, len)`
/// ranges into it, and per-variable production ranges.
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
