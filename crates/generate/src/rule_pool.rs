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
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StrId(NonZeroU32);

impl StrId {
    /// Dense 0-based index (ids are 1-based).
    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize - 1
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
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Prec {
    #[default]
    None,
    Integer(i32),
    Name(StrId),
}

/// [`crate::rules::Alias`] with the value interned. `Copy`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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

    /// Number of unique interned strings.
    #[must_use]
    pub fn str_count(&self) -> usize {
        self.strs.len()
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
        self.child_slice(range).iter().map(|&c| self.rule(c)).collect()
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

/// Pool-owning input grammar. Built from [`InputGrammar`] transitionally;
/// frontends build this directly at series end.
///
/// Invariant: variable names are interned first (variable `i` holds
/// `StrId(i + 1)`), then external token names, so name resolution is a dense
/// index (see `intern_symbols_pool`). Both frontends guarantee unique
/// variable names.
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
        // Dense-name contract: variable names first, external names second,
        // before any body or config string.
        let var_names: Vec<StrId> = g.variables.iter().map(|v| pool.intern(&v.name)).collect();
        debug_assert!(var_names.iter().enumerate().all(|(i, s)| s.index() == i));
        for e in &g.external_tokens {
            if let Rule::NamedSymbol(n) = e {
                pool.intern(n);
            }
        }
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
                roots: set.reserved_words.iter().map(|r| pool.add_rule(r)).collect(),
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
