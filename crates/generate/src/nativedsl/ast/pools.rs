//! Side-table pools backing the out-of-line `Node` data: indexed handles
//! (`MacroId`/`ForId`/`ExpandId`, `ChildRange`), the [`SharedAst`]/[`AstPools`]
//! containers, and the element types they store.

use super::{IdentKind, Node, NodeArena, NodeId};
use crate::nativedsl::ModuleId;
use crate::nativedsl::ast::Spanned;
use crate::nativedsl::typecheck::Ty;
use crate::strpool::StrId;

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

/// One `name: value` entry of an object literal, stored in [`AstPools::object_fields`].
#[derive(Clone, Copy, Debug)]
pub struct ObjectField {
    pub name: Spanned<StrId>,
    pub value: NodeId,
}

#[derive(Clone)]
pub struct MacroConfig {
    pub name: Spanned<StrId>,
    pub params: ChildRange,
    pub body: NodeId,
    pub kind: MacroKind,
    /// Computed-name references (`@<expr>` -> `SymRef`) in a rule-set macro's
    /// body, recorded by the parser as it builds them. Expand evaluates each
    /// under a call's args; resolve validates the result exists.
    pub sym_refs: ChildRange,
    /// Set when this macro enters the resolved declaration table, after its module's
    /// final is known. Lowering never observes `None`.
    def_module: Option<ModuleId>,
}

impl MacroConfig {
    #[must_use]
    pub(crate) const fn new(
        name: Spanned<StrId>,
        params: ChildRange,
        body: NodeId,
        kind: MacroKind,
        sym_refs: ChildRange,
    ) -> Self {
        Self {
            name,
            params,
            body,
            kind,
            sym_refs,
            def_module: None,
        }
    }

    pub(crate) fn set_def_module(&mut self, module: ModuleId) {
        debug_assert!(self.def_module.is_none());
        self.def_module = Some(module);
    }

    #[must_use]
    pub(crate) const fn def_module(&self) -> ModuleId {
        self.def_module.unwrap()
    }
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
    pub name: Spanned<StrId>,
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
    pub name: StrId,
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
    /// that let and the referencing node (for a self-reference note). Returns `None`
    /// once every referenced `let` is resolved.
    #[must_use]
    pub fn first_unresolved_let_dep(
        &self,
        root: NodeId,
        is_resolved: impl Fn(NodeId) -> bool,
        stack: &mut Vec<NodeId>,
    ) -> Option<(NodeId, NodeId)> {
        stack.clear();
        stack.push(root);
        while let Some(id) = stack.pop() {
            match *self.arena.get(id) {
                Node::Ident(IdentKind::Var(let_id)) if !is_resolved(let_id) => {
                    return Some((let_id, id));
                }
                node => self.push_expr_children(node, stack),
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

    #[inline]
    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[ObjectField] {
        &self.object_fields[range.as_range()]
    }

    #[inline]
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        &self.children[range.as_range()]
    }

    #[must_use]
    pub fn param_slice(&self, range: ChildRange) -> &[Param] {
        &self.params[range.as_range()]
    }
}
