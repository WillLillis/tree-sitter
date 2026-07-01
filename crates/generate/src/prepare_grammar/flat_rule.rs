//! Spike: a flat, pool-indexed mirror of [`crate::rules::Rule`] for the grammar
//! backend.
//!
//! Each node lives in a `Vec` and references its children by index, so
//! transforming a rule becomes an iterative pool walk instead of a recursive walk
//! over a `Box`/`Vec` tree - removing the native-stack depth bound that the
//! recursive `Rule` passes impose. The variant set mirrors [`Rule`] one-to-one
//! (it keeps the resolved [`Symbol`] form and the bundled [`MetadataParams`]); the
//! only difference is that children are pool indices rather than owned boxes/vecs:
//! `Box<Rule>` becomes a [`RuleId`] and `Vec<Rule>` a [`ChildRange`].
//!
//! [`import`](FlatRules::import) and [`materialize`](FlatRules::materialize) are
//! lossless inverse adapters between `Rule` and the pool, both iterative.

use rustc_hash::FxHashMap;

use crate::rules::{MetadataParams, Rule, Symbol};

/// Index into [`FlatRules::nodes`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RuleId(pub u32);

/// A `[start, start + len)` slice of [`FlatRules::children`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ChildRange {
    pub start: u32,
    pub len: u32,
}

impl ChildRange {
    const fn as_range(self) -> std::ops::Range<usize> {
        self.start as usize..(self.start + self.len) as usize
    }
}

/// A [`Rule`] with pooled children. One variant per `Rule` variant; the payload
/// is identical except that child rules are referenced by [`RuleId`] /
/// [`ChildRange`] into the owning [`FlatRules`] instead of being owned inline.
#[derive(Debug)]
pub enum FlatRule {
    Blank,
    String(String),
    Pattern(String, String),
    NamedSymbol(String),
    Symbol(Symbol),
    Choice(ChildRange),
    Seq(ChildRange),
    Repeat(RuleId),
    Metadata { params: MetadataParams, rule: RuleId },
    Reserved { rule: RuleId, context_name: String },
}

/// Structural hash-consing key: rules with equal keys collapse to one pooled node,
/// so within a pool `RuleId` equality becomes structural equality. Variadic nodes
/// key on their child *ids*, not their [`ChildRange`] (which varies with pool
/// offset).
#[derive(PartialEq, Eq, Hash)]
enum NodeKey {
    Blank,
    String(String),
    Pattern(String, String),
    NamedSymbol(String),
    Symbol(Symbol),
    SeqOrChoice(bool, Vec<RuleId>),
    Repeat(RuleId),
    Metadata(MetadataParams, RuleId),
    Reserved(RuleId, String),
}

/// The node + child pools backing a set of [`FlatRule`]s. `table` hash-conses nodes
/// built through the `intern_*` path ([`intern_import`](FlatRules::intern_import));
/// the plain [`import`](FlatRules::import) path leaves it empty.
#[derive(Default)]
pub struct FlatRules {
    nodes: Vec<FlatRule>,
    children: Vec<RuleId>,
    table: FxHashMap<NodeKey, RuleId>,
}

impl FlatRules {
    fn alloc(&mut self, node: FlatRule) -> RuleId {
        let id = RuleId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    pub fn get(&self, id: RuleId) -> &FlatRule {
        &self.nodes[id.0 as usize]
    }

    pub fn get_mut(&mut self, id: RuleId) -> &mut FlatRule {
        &mut self.nodes[id.0 as usize]
    }

    pub fn children(&self, range: ChildRange) -> &[RuleId] {
        &self.children[range.as_range()]
    }

    /// Import a recursive [`Rule`] into the pool, returning the root [`RuleId`].
    ///
    /// Two-phase post-order walk over an explicit stack (`Visit` descends, `Build`
    /// assembles from already-imported children) so a deeply nested rule cannot
    /// overflow the native stack. Children are pushed reversed so they land on the
    /// output stack in source order.
    pub fn import(&mut self, root: &Rule) -> RuleId {
        enum Step<'r> {
            Visit(&'r Rule),
            Build(&'r Rule),
        }
        let mut work = vec![Step::Visit(root)];
        let mut out: Vec<RuleId> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(rule) => match rule {
                    Rule::Blank => out.push(self.alloc(FlatRule::Blank)),
                    Rule::String(s) => out.push(self.alloc(FlatRule::String(s.clone()))),
                    Rule::Pattern(p, f) => {
                        out.push(self.alloc(FlatRule::Pattern(p.clone(), f.clone())));
                    }
                    Rule::NamedSymbol(n) => out.push(self.alloc(FlatRule::NamedSymbol(n.clone()))),
                    Rule::Symbol(sym) => out.push(self.alloc(FlatRule::Symbol(*sym))),
                    Rule::Choice(elems) | Rule::Seq(elems) => {
                        work.push(Step::Build(rule));
                        for e in elems.iter().rev() {
                            work.push(Step::Visit(e));
                        }
                    }
                    Rule::Repeat(inner)
                    | Rule::Metadata { rule: inner, .. }
                    | Rule::Reserved { rule: inner, .. } => {
                        work.push(Step::Build(rule));
                        work.push(Step::Visit(inner));
                    }
                },
                Step::Build(rule) => match rule {
                    Rule::Choice(elems) | Rule::Seq(elems) => {
                        let n = elems.len();
                        let start = self.children.len() as u32;
                        let base = out.len() - n;
                        self.children.extend_from_slice(&out[base..]);
                        out.truncate(base);
                        let range = ChildRange {
                            start,
                            len: n as u32,
                        };
                        let node = if matches!(rule, Rule::Seq(_)) {
                            FlatRule::Seq(range)
                        } else {
                            FlatRule::Choice(range)
                        };
                        out.push(self.alloc(node));
                    }
                    Rule::Repeat(_) => {
                        let inner = out.pop().unwrap();
                        out.push(self.alloc(FlatRule::Repeat(inner)));
                    }
                    Rule::Metadata { params, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(self.alloc(FlatRule::Metadata {
                            params: params.clone(),
                            rule: inner,
                        }));
                    }
                    Rule::Reserved { context_name, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(self.alloc(FlatRule::Reserved {
                            rule: inner,
                            context_name: context_name.clone(),
                        }));
                    }
                    // Leaves never enqueue a Build step.
                    _ => unreachable!(),
                },
            }
        }
        out.pop().unwrap()
    }

    /// Materialize a pooled rule back into a recursive [`Rule`]. Inverse of
    /// [`import`](Self::import); iterative post-order, so it cannot overflow on
    /// the pool side. Uses the raw `Seq`/`Choice` variants (not the flattening
    /// constructors) so the result is byte-identical to the imported rule.
    pub fn materialize(&self, root: RuleId) -> Rule {
        enum Step {
            Visit(RuleId),
            Build(RuleId),
        }
        let mut work = vec![Step::Visit(root)];
        let mut out: Vec<Rule> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(id) => match self.get(id) {
                    FlatRule::Blank => out.push(Rule::Blank),
                    FlatRule::String(s) => out.push(Rule::String(s.clone())),
                    FlatRule::Pattern(p, f) => out.push(Rule::Pattern(p.clone(), f.clone())),
                    FlatRule::NamedSymbol(n) => out.push(Rule::NamedSymbol(n.clone())),
                    FlatRule::Symbol(sym) => out.push(Rule::Symbol(*sym)),
                    FlatRule::Choice(range) | FlatRule::Seq(range) => {
                        work.push(Step::Build(id));
                        for &child in self.children(*range).iter().rev() {
                            work.push(Step::Visit(child));
                        }
                    }
                    FlatRule::Repeat(inner)
                    | FlatRule::Metadata { rule: inner, .. }
                    | FlatRule::Reserved { rule: inner, .. } => {
                        work.push(Step::Build(id));
                        work.push(Step::Visit(*inner));
                    }
                },
                Step::Build(id) => match self.get(id) {
                    FlatRule::Choice(range) | FlatRule::Seq(range) => {
                        let base = out.len() - range.len as usize;
                        let children = out.split_off(base);
                        out.push(if matches!(self.get(id), FlatRule::Seq(_)) {
                            Rule::Seq(children)
                        } else {
                            Rule::Choice(children)
                        });
                    }
                    FlatRule::Repeat(_) => {
                        let inner = out.pop().unwrap();
                        out.push(Rule::Repeat(Box::new(inner)));
                    }
                    FlatRule::Metadata { params, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(Rule::Metadata {
                            params: params.clone(),
                            rule: Box::new(inner),
                        });
                    }
                    FlatRule::Reserved { context_name, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(Rule::Reserved {
                            rule: Box::new(inner),
                            context_name: context_name.clone(),
                        });
                    }
                    // Leaves never enqueue a Build step.
                    _ => unreachable!(),
                },
            }
        }
        out.pop().unwrap()
    }

    /// Intern a leaf / single-child node: return the existing id for a
    /// structurally identical node, or allocate and record a fresh one.
    fn intern(&mut self, key: NodeKey, node: FlatRule) -> RuleId {
        if let Some(&id) = self.table.get(&key) {
            return id;
        }
        let id = self.alloc(node);
        self.table.insert(key, id);
        id
    }

    /// Intern a `Seq`/`Choice` over already-interned `children`; on a miss the
    /// children are appended to the child pool and a fresh node allocated.
    fn intern_seq_or_choice(&mut self, is_seq: bool, children: &[RuleId]) -> RuleId {
        let key = NodeKey::SeqOrChoice(is_seq, children.to_vec());
        if let Some(&id) = self.table.get(&key) {
            return id;
        }
        let start = self.children.len() as u32;
        self.children.extend_from_slice(children);
        let range = ChildRange {
            start,
            len: children.len() as u32,
        };
        let id = self.alloc(if is_seq {
            FlatRule::Seq(range)
        } else {
            FlatRule::Choice(range)
        });
        self.table.insert(key, id);
        id
    }

    /// Import a [`Rule`] into the pool with hash-consing: structurally identical
    /// subtrees collapse to one [`RuleId`], so within this pool `RuleId` equality
    /// *is* structural equality. Same iterative post-order shape as
    /// [`import`](Self::import), allocating via the interning helpers. (The two
    /// share structure; they consolidate once the migration uses interning
    /// everywhere and the plain `import` is retired.)
    pub fn intern_import(&mut self, root: &Rule) -> RuleId {
        enum Step<'r> {
            Visit(&'r Rule),
            Build(&'r Rule),
        }
        let mut work = vec![Step::Visit(root)];
        let mut out: Vec<RuleId> = Vec::new();
        while let Some(step) = work.pop() {
            match step {
                Step::Visit(rule) => match rule {
                    Rule::Blank => out.push(self.intern(NodeKey::Blank, FlatRule::Blank)),
                    Rule::String(s) => {
                        out.push(self.intern(NodeKey::String(s.clone()), FlatRule::String(s.clone())));
                    }
                    Rule::Pattern(p, f) => out.push(self.intern(
                        NodeKey::Pattern(p.clone(), f.clone()),
                        FlatRule::Pattern(p.clone(), f.clone()),
                    )),
                    Rule::NamedSymbol(n) => out.push(self.intern(
                        NodeKey::NamedSymbol(n.clone()),
                        FlatRule::NamedSymbol(n.clone()),
                    )),
                    Rule::Symbol(sym) => {
                        out.push(self.intern(NodeKey::Symbol(*sym), FlatRule::Symbol(*sym)));
                    }
                    Rule::Choice(elems) | Rule::Seq(elems) => {
                        work.push(Step::Build(rule));
                        for e in elems.iter().rev() {
                            work.push(Step::Visit(e));
                        }
                    }
                    Rule::Repeat(inner)
                    | Rule::Metadata { rule: inner, .. }
                    | Rule::Reserved { rule: inner, .. } => {
                        work.push(Step::Build(rule));
                        work.push(Step::Visit(inner));
                    }
                },
                Step::Build(rule) => match rule {
                    Rule::Choice(elems) | Rule::Seq(elems) => {
                        let base = out.len() - elems.len();
                        let children = out.split_off(base);
                        out.push(self.intern_seq_or_choice(matches!(rule, Rule::Seq(_)), &children));
                    }
                    Rule::Repeat(_) => {
                        let inner = out.pop().unwrap();
                        out.push(self.intern(NodeKey::Repeat(inner), FlatRule::Repeat(inner)));
                    }
                    Rule::Metadata { params, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(self.intern(
                            NodeKey::Metadata(params.clone(), inner),
                            FlatRule::Metadata {
                                params: params.clone(),
                                rule: inner,
                            },
                        ));
                    }
                    Rule::Reserved { context_name, .. } => {
                        let inner = out.pop().unwrap();
                        out.push(self.intern(
                            NodeKey::Reserved(inner, context_name.clone()),
                            FlatRule::Reserved {
                                rule: inner,
                                context_name: context_name.clone(),
                            },
                        ));
                    }
                    _ => unreachable!(),
                },
            }
        }
        out.pop().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::{Precedence, Symbol};

    #[test]
    fn round_trips_every_variant() {
        // One rule exercising all ten variants, with nesting, so import followed
        // by materialize must reproduce it exactly.
        let rule = Rule::Seq(vec![
            Rule::Blank,
            Rule::string("lit"),
            Rule::pattern("[a-z]+", "i"),
            Rule::named("ref"),
            Rule::Symbol(Symbol::non_terminal(3)),
            Rule::Choice(vec![Rule::string("a"), Rule::repeat(Rule::named("b"))]),
            Rule::field("name".to_string(), Rule::named("id")),
            Rule::prec(Precedence::Integer(2), Rule::Blank),
            Rule::Reserved {
                rule: Box::new(Rule::string("kw")),
                context_name: "ctx".to_string(),
            },
        ]);
        let mut pool = FlatRules::default();
        let id = pool.import(&rule);
        assert_eq!(pool.materialize(id), rule);
    }

    #[test]
    fn intern_import_canonicalizes() {
        let a = Rule::seq(vec![Rule::string("x"), Rule::named("y")]);
        let b = Rule::seq(vec![Rule::string("x"), Rule::named("y")]); // structurally equal to a
        let c = Rule::seq(vec![Rule::string("x"), Rule::named("z")]); // differs from a
        let mut pool = FlatRules::default();
        let ia = pool.intern_import(&a);
        let ib = pool.intern_import(&b);
        let ic = pool.intern_import(&c);
        assert_eq!(ia, ib, "structurally identical rules must intern to one id");
        assert_ne!(ia, ic, "different rules must intern to different ids");
        assert_eq!(pool.materialize(ia), a, "interning preserves structure");
    }
}
