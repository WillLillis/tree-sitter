//! Side-table pools backing the out-of-line `Node` data: indexed handles
//! (`MacroId`/`ForId`/`ExpandId`, `ChildRange`), the [`SharedAst`]/[`AstPools`]
//! containers, and the element types they store.

use super::{IdentKind, Node, NodeArena, NodeId, Span};
use crate::nativedsl::string_pool::Str;
use crate::nativedsl::typecheck::Ty;

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
        pub struct $name(u32);
        impl $name {
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(MacroId);
id_type!(ForId);
id_type!(ExpandId);

#[derive(Clone, Copy, Debug)]
pub struct ChildRange {
    pub start: u32,
    pub len: u16,
}

impl ChildRange {
    #[must_use]
    pub const fn new(start: u32, len: u16) -> Self {
        Self { start, len }
    }

    #[must_use]
    pub const fn as_range(self) -> std::ops::Range<usize> {
        self.start as usize..self.start as usize + self.len as usize
    }
}

/// One `name: value` entry of an object literal, stored in
/// [`AstPools::object_fields`]. `name` is the span of the field key.
#[derive(Clone, Copy, Debug)]
pub struct ObjectField {
    pub name: Span,
    pub value: NodeId,
}

#[derive(Clone)]
pub struct MacroConfig {
    pub name: Span,
    pub params: ChildRange,
    pub body: NodeId,
    pub kind: MacroKind,
    /// Computed-name references (`@<expr>` -> `SymRef`) in a rule-set macro's
    /// body, recorded by the parser as it builds them. Expand evaluates each
    /// under a call's args; resolve validates the result exists.
    pub sym_refs: ChildRange,
}

#[derive(Clone, Copy, Debug)]
pub enum MacroKind {
    /// Body is an expression typechecking to the carried return type.
    Expression(Ty),
    /// Body is `Node::RuleSet`; expanded inline at each top-level call site.
    RuleSet,
}

#[derive(Copy, Clone)]
pub struct Param {
    pub name: Span,
    pub ty: Ty,
}

#[derive(Clone, Copy)]
pub struct ForConfig {
    pub bindings: ChildRange,
    pub iterable: NodeId,
}

/// One instantiation of a rule-set macro decl at a `@name(args)` call site.
#[derive(Clone, Copy)]
pub struct Expansion {
    pub is_override: bool,
    pub name: Str,
    pub macro_id: MacroId,
    pub body: NodeId,
    pub args: ChildRange,
}

/// Shared AST data across all modules in a grammar. All `NodeId`, `MacroId`,
/// `ForId`, and `ChildRange` values are globally valid within this structure.
pub struct SharedAst {
    pub arena: NodeArena,
    pub pools: AstPools,
}

/// Indexed pool data backing `MacroId`, `ForId`, and `ChildRange`.
pub struct AstPools {
    pub children: Vec<NodeId>,
    pub macro_configs: Vec<MacroConfig>,
    pub for_configs: Vec<ForConfig>,
    pub object_fields: Vec<ObjectField>,
    pub params: Vec<Param>,
    pub expansions: Vec<Expansion>,
}

impl SharedAst {
    #[must_use]
    pub fn new(estimated_cap: usize) -> Self {
        Self {
            arena: NodeArena::new(estimated_cap),
            pools: AstPools {
                children: Vec::with_capacity(estimated_cap),
                macro_configs: Vec::new(),
                for_configs: Vec::new(),
                object_fields: Vec::new(),
                params: Vec::new(),
                expansions: Vec::new(),
            },
        }
    }

    /// Find the first `let` referenced in `root`'s expression subtree that is not
    /// yet resolved (per `is_resolved`, keyed by the let's `NodeId`), returning
    /// that let and the referencing node (for a self-reference note). Walks with
    /// an explicit stack so a deep expression tree cannot overflow; returns `None`
    /// once every referenced `let` is resolved. Used by typecheck and lower to
    /// resolve `let` chains iteratively rather than recursively.
    #[must_use]
    pub fn first_unresolved_let_dep(
        &self,
        root: NodeId,
        is_resolved: impl Fn(NodeId) -> bool,
    ) -> Option<(NodeId, NodeId)> {
        let mut stack = vec![root];
        while let Some(id) = stack.pop() {
            match *self.arena.get(id) {
                Node::Ident(IdentKind::Var(let_id)) if !is_resolved(let_id) => {
                    return Some((let_id, id));
                }
                node => self.push_expr_children(node, &mut stack),
            }
        }
        None
    }

    /// Push the child node ids of an expression node onto `stack`. Mirrors the
    /// expression children walked by [`resolve_expr`](crate::nativedsl::resolve);
    /// keep the two in sync.
    fn push_expr_children(&self, node: Node, stack: &mut Vec<NodeId>) {
        if let Some(range) = node.child_range() {
            stack.extend(self.pools.child_slice(range).iter().copied());
            return;
        }
        match node {
            Node::Call { name, args } => {
                stack.push(name);
                stack.extend(self.pools.child_slice(args).iter().copied());
            }
            Node::QualifiedCall(range) => {
                stack.extend(self.pools.child_slice(range).iter().copied());
            }
            Node::Object(range) => {
                stack.extend(self.pools.get_object(range).iter().map(|f| f.value));
            }
            Node::For { for_id, body } => {
                stack.push(self.pools.get_for(for_id).iterable);
                stack.push(body);
            }
            Node::Alias { content, target } => {
                stack.push(content);
                stack.push(target);
            }
            Node::Append { left: a, right: b }
            | Node::BinOp { lhs: a, rhs: b, .. }
            | Node::Prec {
                value: a,
                content: b,
                ..
            }
            | Node::ComputedRule {
                name_expr: a,
                body: b,
                ..
            } => {
                stack.push(a);
                stack.push(b);
            }
            Node::DynRegex { pattern, flags } => {
                stack.push(pattern);
                stack.extend(flags);
            }
            Node::Repeat { inner: c, .. }
            | Node::Token { inner: c, .. }
            | Node::Neg(c)
            | Node::GrammarConfig { module: c, .. }
            | Node::Field { content: c, .. }
            | Node::Reserved { content: c, .. }
            | Node::Rule { body: c, .. }
            | Node::SymRef { expr: c }
            | Node::FieldAccess { obj: c, .. }
            | Node::QualifiedAccess { obj: c, .. }
            | Node::Let { value: c, .. }
            | Node::Cfg { child: c, .. } => stack.push(c),
            _ => {}
        }
    }
}

impl AstPools {
    pub fn push_macro(&mut self, config: MacroConfig) -> MacroId {
        let id = MacroId(self.macro_configs.len() as u32);
        self.macro_configs.push(config);
        id
    }

    pub fn push_for(&mut self, config: ForConfig) -> ForId {
        let id = ForId(self.for_configs.len() as u32);
        self.for_configs.push(config);
        id
    }

    pub fn push_expansion(&mut self, expansion: Expansion) -> ExpandId {
        let id = ExpandId(self.expansions.len() as u32);
        self.expansions.push(expansion);
        id
    }

    pub fn push_object(&mut self, fields: Vec<ObjectField>) -> Option<ChildRange> {
        let start = self.object_fields.len() as u32;
        let len = u16::try_from(fields.len()).ok()?;
        self.object_fields.extend(fields);
        Some(ChildRange::new(start, len))
    }

    pub fn push_children(&mut self, items: &[NodeId]) -> Option<ChildRange> {
        let start = self.children.len() as u32;
        let len = u16::try_from(items.len()).ok()?;
        self.children.extend_from_slice(items);
        Some(ChildRange::new(start, len))
    }

    /// Pool a macro's params or a for-loop's bindings. Both are bounded to
    /// `u8::MAX` at parse time (their indices are `u8`), so the count always
    /// fits `ChildRange::len` and cannot overflow.
    pub fn push_params(&mut self, params: &[Param]) -> ChildRange {
        let start = self.params.len() as u32;
        debug_assert!(u8::try_from(params.len()).is_ok());
        let len = params.len() as u16;
        self.params.extend_from_slice(params);
        ChildRange::new(start, len)
    }

    #[must_use]
    pub fn get_macro(&self, id: MacroId) -> &MacroConfig {
        &self.macro_configs[id.index()]
    }

    pub fn get_macro_mut(&mut self, id: MacroId) -> &mut MacroConfig {
        &mut self.macro_configs[id.index()]
    }

    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        &self.for_configs[id.index()]
    }

    #[must_use]
    pub fn get_expansion(&self, id: ExpandId) -> &Expansion {
        &self.expansions[id.index()]
    }

    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[ObjectField] {
        &self.object_fields[range.as_range()]
    }

    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        &self.children[range.as_range()]
    }

    #[must_use]
    pub fn param_slice(&self, range: ChildRange) -> &[Param] {
        &self.params[range.as_range()]
    }

    /// Unpack a `QualifiedCall(range)` into `(obj, name, &[args])`.
    ///
    /// SAFETY:
    ///
    /// Caller must pass a range from an actual `Node::QualifiedCall` node.
    #[must_use]
    pub fn get_qualified_call(&self, range: ChildRange) -> (NodeId, NodeId, &[NodeId]) {
        let children = self.child_slice(range);
        // SAFETY: QualifiedCall nodes are always constructed with [obj, name, ...args]
        // by the parser, so len >= 2 is structurally guaranteed.
        let len = children.len();
        debug_assert!(len >= 2);
        unsafe {
            (
                *children.get_unchecked(0),
                *children.get_unchecked(1),
                if len > 2 {
                    children.get_unchecked(2..)
                } else {
                    &[]
                },
            )
        }
    }
}
