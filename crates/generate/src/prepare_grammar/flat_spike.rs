//! TEMP SPIKE (remove after the perf decision): a flat-pool `intern_symbols` that
//! mutates `NamedSymbol -> Symbol` in place, to A/B against master's Rule-based
//! `intern_symbols`. No hash-consing; string payloads are interned to a `StrId`
//! (so nodes are small/`Copy` and name resolution is a `u32`-keyed lookup);
//! children of `Seq`/`Choice` live as `(offset, len)` ranges into one backing
//! `Vec`.
//!
//! Kept recursive on purpose (the iterative conversion is a separate later step) so
//! this measures the flat backing type + in-place rewrite + string interning vs
//! `Rule`.

use rustc_hash::FxHashMap;

use crate::grammars::InputGrammar;
use crate::rules::{MetadataParams, Rule, Symbol};

#[derive(Clone, Copy)]
pub struct NodeId(u32);

/// Interned string id into [`Pool::strings`].
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct StrId(u32);

#[derive(Clone, Copy)]
struct NodeRange {
    start: u32,
    len: u32,
}

#[derive(Clone)]
enum FNode {
    Blank,
    String(StrId),
    Pattern(StrId, StrId),
    NamedSymbol(StrId),
    Symbol(Symbol),
    Seq(NodeRange),
    Choice(NodeRange),
    Repeat(NodeId),
    Metadata { params: MetadataParams, rule: NodeId },
    Reserved { rule: NodeId, context_name: StrId },
}

/// Flat pool: `nodes` is the arena (a child holds a `NodeId` into it); `children`
/// is the backing store `Seq`/`Choice` index by range; `str_ids` interns string
/// payloads to a `StrId`; `name_of_symbol` resolves an interned rule name (a dense
/// `StrId`) to its `Symbol` by direct index.
///
/// Note: no `StrId -> str` reverse map. `intern_symbols` never needs the text (it
/// resolves `StrId -> Symbol`), so each string is stored exactly once, in
/// `str_ids`. The real design will need the reverse for serialize/errors; do that
/// with `Rc<str>` or a blob + ranges to avoid a second allocation.
#[derive(Clone)]
pub struct Pool {
    nodes: Vec<FNode>,
    children: Vec<NodeId>,
    str_ids: FxHashMap<Box<str>, StrId>,
    roots: Vec<NodeId>,
    ext_roots: Vec<NodeId>,
    extra_roots: Vec<NodeId>,
    reserved_roots: Vec<NodeId>,
    /// `StrId -> Symbol` for the `0..k` name `StrId`s. Dense (every entry is a name,
    /// no `Option`): an *undefined* symbol's `StrId` is `>= k`, caught by the bounds
    /// check in `resolve`, not by an in-range `None`. A `Vec` since `StrId`s are dense.
    name_of_symbol: Vec<Symbol>,
}

impl Pool {
    /// Convert a master `InputGrammar` into the flat pool. SETUP ONLY - excluded
    /// from the timed pass (mirrors how, in a real pipeline, the frontend would
    /// build the pool directly, with strings already interned).
    pub fn from_grammar(g: &InputGrammar) -> Self {
        let mut p = Self {
            nodes: Vec::new(),
            children: Vec::new(),
            str_ids: FxHashMap::default(),
            roots: Vec::with_capacity(g.variables.len()),
            ext_roots: Vec::with_capacity(g.external_tokens.len()),
            extra_roots: Vec::with_capacity(g.extra_symbols.len()),
            reserved_roots: Vec::new(),
            name_of_symbol: Vec::new(),
        };
        // Intern names FIRST so name StrIds occupy the dense low range `0..k`.
        // Variables are interned in order, so variable `i` gets StrId `i` and
        // `name_of_symbol[i] = non_terminal(i)` directly.
        for v in &g.variables {
            p.intern_str(&v.name);
        }
        let mut name_of_symbol: Vec<Symbol> =
            (0..g.variables.len()).map(Symbol::non_terminal).collect();
        // An external name not already a variable gets the next StrId and is appended
        // in order; a name that dups a variable (or an earlier external) resolves to
        // an existing StrId, so we skip it - first writer wins, matching master's
        // variables-before-externals scan.
        for (i, e) in g.external_tokens.iter().enumerate() {
            if let Rule::NamedSymbol(name) = e {
                let sid = p.intern_str(name);
                if sid.0 as usize == name_of_symbol.len() {
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

    fn intern_str(&mut self, s: &str) -> StrId {
        if let Some(&id) = self.str_ids.get(s) {
            return id;
        }
        let id = StrId(self.str_ids.len() as u32);
        self.str_ids.insert(s.into(), id);
        id
    }

    fn add(&mut self, rule: &Rule) -> NodeId {
        let node = match rule {
            Rule::Blank => FNode::Blank,
            Rule::String(s) => FNode::String(self.intern_str(s)),
            Rule::Pattern(a, b) => FNode::Pattern(self.intern_str(a), self.intern_str(b)),
            Rule::NamedSymbol(n) => FNode::NamedSymbol(self.intern_str(n)),
            Rule::Symbol(s) => FNode::Symbol(*s),
            Rule::Seq(elems) => FNode::Seq(self.add_children(elems)),
            Rule::Choice(elems) => FNode::Choice(self.add_children(elems)),
            Rule::Repeat(c) => FNode::Repeat(self.add(c)),
            Rule::Metadata { params, rule } => FNode::Metadata {
                params: params.clone(),
                rule: self.add(rule),
            },
            Rule::Reserved { rule, context_name } => {
                let context_name = self.intern_str(context_name);
                FNode::Reserved {
                    rule: self.add(rule),
                    context_name,
                }
            }
        };
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    fn add_children(&mut self, elems: &[Rule]) -> NodeRange {
        // Convert each child subtree first (each returns its root id), then lay the
        // ids down contiguously. The temp Vec is setup-only.
        let ids: Vec<NodeId> = elems.iter().map(|e| self.add(e)).collect();
        let start = self.children.len() as u32;
        self.children.extend_from_slice(&ids);
        NodeRange {
            start,
            len: ids.len() as u32,
        }
    }

    /// The timed pass: resolve every `NamedSymbol` to a `Symbol` in place, over all
    /// root sets. Recursive, in-place, zero allocation.
    pub fn intern_symbols(&mut self) {
        // All root sets, indexed by position so we hold no borrow across `resolve`.
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
        // Read phase: decide the action without holding a mutable borrow.
        enum Act {
            Sym(Symbol),
            One(NodeId),
            Range(NodeRange),
            Nop,
        }
        let act = match &self.nodes[idx] {
            FNode::NamedSymbol(sid) => self
                .name_of_symbol
                .get(sid.0 as usize)
                .copied()
                .map_or(Act::Nop, Act::Sym),
            FNode::Seq(r) | FNode::Choice(r) => Act::Range(*r),
            FNode::Repeat(c)
            | FNode::Metadata { rule: c, .. }
            | FNode::Reserved { rule: c, .. } => Act::One(*c),
            _ => Act::Nop,
        };
        // Apply phase.
        match act {
            Act::Sym(s) => self.nodes[idx] = FNode::Symbol(s),
            Act::One(c) => self.resolve(c),
            Act::Range(r) => {
                for i in 0..r.len {
                    let c = self.children[(r.start + i) as usize];
                    self.resolve(c);
                }
            }
            Act::Nop => {}
        }
    }
}
