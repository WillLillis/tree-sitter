//! Arena-based AST for the native grammar DSL.

use std::num::NonZeroU32;

use serde::Serialize;

use super::typecheck::Ty;

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(NonZeroU32);

impl NodeId {
    /// The first valid `NodeId` (index 1, after the sentinel).
    pub const FIRST: Self = Self(NonZeroU32::new(1).unwrap());

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize
    }

    /// Return the next `NodeId` (index + 1).
    #[must_use]
    pub const fn next(self) -> Self {
        // SAFETY: self.0 >= 1, so self.0 + 1 >= 2, always non-zero.
        Self(unsafe { NonZeroU32::new_unchecked(self.0.get() + 1) })
    }
}

/// Arena for AST nodes and their spans.
pub struct NodeArena {
    nodes: Vec<Node>,
    spans: Vec<Span>,
}

impl NodeArena {
    #[must_use]
    pub fn new(estimated_cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(estimated_cap);
        let mut spans = Vec::with_capacity(estimated_cap);
        nodes.push(Node::Unreachable);
        spans.push(Span::new(0, 0));
        Self { nodes, spans }
    }

    #[inline]
    pub fn push(&mut self, node: Node, span: Span) -> NodeId {
        let index = self.nodes.len() as u32;
        // SAFETY: nodes[0] is always the Unreachable sentinel, so len() >= 1.
        let id = NodeId(unsafe { NonZeroU32::new_unchecked(index) });
        self.nodes.push(node);
        self.spans.push(span);
        id
    }

    #[must_use]
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.spans[id.index()]
    }

    pub fn set(&mut self, id: NodeId, node: Node) {
        self.nodes[id.index()] = node;
    }

    #[inline]
    pub fn resolve_as(&mut self, id: NodeId, kind: IdentKind) {
        debug_assert!(matches!(self.get(id), Node::Ident(IdentKind::Unresolved)));
        self.set(id, Node::Ident(kind));
    }

    /// Number of nodes (excluding sentinel).
    #[must_use]
    pub const fn len(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Returns `true` if the arena contains no nodes (excluding sentinel).
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.nodes.len() == 1
    }

    /// Iterate all valid [`NodeId`]s.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        (1..self.nodes.len()).map(|i| {
            // SAFETY: i starts at 1, always non-zero.
            NodeId(unsafe { NonZeroU32::new_unchecked(i as u32) })
        })
    }

    /// Iterate all `(NodeId, &Node)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.node_ids().map(|id| (id, &self.nodes[id.index()]))
    }
}

macro_rules! id_type {
    ($name:ident) => {
        #[derive(Clone, Copy, PartialEq, Eq, Hash)]
        pub struct $name(u32);
        impl $name {
            #[must_use]
            pub const fn index(self) -> usize {
                self.0 as usize
            }
        }
    };
}

id_type!(FnId);
id_type!(ForId);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PrecKind {
    Default,
    Left,
    Right,
    Dynamic,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    Optional,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum IdentKind {
    Unresolved,
    Rule,
    Var,
    Fn,
}

/// Byte offset range `[start, end)` in the source text.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Span {
    pub start: u32,
    pub end: u32,
}

impl Span {
    #[must_use]
    pub const fn new(start: u32, end: u32) -> Self {
        Self { start, end }
    }

    #[must_use]
    pub const fn from_usize(start: usize, end: usize) -> Self {
        Self {
            start: start as u32,
            end: end as u32,
        }
    }

    /// Strip the first and last byte (e.g. surrounding quotes).
    /// Assumes the span covers at least 2 bytes.
    #[must_use]
    pub const fn strip_quotes(self) -> Self {
        Self {
            start: self.start + 1,
            end: self.end - 1,
        }
    }

    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Resolve to a `&str` slice.
    #[must_use]
    pub fn resolve<'src>(&self, source: &'src str) -> &'src str {
        // SAFETY: span boundaries are at ASCII byte positions. The
        // input length is checked against u32::MAX before lexing.
        unsafe { source.get_unchecked(self.start as usize..self.end as usize) }
    }
}

pub struct Ast {
    pub arena: NodeArena,
    pub root_items: Vec<NodeId>,
    pub ctx: AstContext,
}

/// Immutable context data split from nodes
pub struct AstContext {
    pub grammar_config: Option<GrammarConfig>,
    pub fn_configs: Vec<FnConfig>,
    pub for_configs: Vec<ForConfig>,
    pub object_fields: Vec<(Span, NodeId)>,
    pub children: Vec<NodeId>,
    pub source: String,
}

impl AstContext {
    /// Resolve a span to the source text it covers.
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(&self.source)
    }
    #[must_use]
    pub fn get_fn(&self, id: FnId) -> &FnConfig {
        &self.fn_configs[id.index()]
    }
    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        &self.for_configs[id.index()]
    }
    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[(Span, NodeId)] {
        &self.object_fields[range.start as usize..range.start as usize + range.len as usize]
    }
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        &self.children[range.start as usize..range.start as usize + range.len as usize]
    }
    /// Unpack a `QualifiedCall(range)` into `(obj, name, &[args])`.
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
    /// Unpack a `DynRegex(range)` into `(pattern, Option<flags>)`.
    #[must_use]
    pub fn get_regex(&self, range: ChildRange) -> (NodeId, Option<NodeId>) {
        let c = self.child_slice(range);
        (c[0], c.get(1).copied())
    }
}

impl Ast {
    #[must_use]
    pub fn new(source: String) -> Self {
        let cap = source.len() / 30;
        Self {
            arena: NodeArena::new(cap),
            root_items: Vec::new(),
            ctx: AstContext {
                grammar_config: None,
                fn_configs: Vec::new(),
                for_configs: Vec::new(),
                object_fields: Vec::new(),
                children: Vec::with_capacity(cap),
                source,
            },
        }
    }

    #[inline]
    pub fn push(&mut self, node: Node, span: Span) -> NodeId {
        self.arena.push(node, span)
    }

    #[must_use]
    pub fn node(&self, id: NodeId) -> &Node {
        self.arena.get(id)
    }
    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.arena.span(id)
    }

    pub fn push_fn(&mut self, config: FnConfig) -> FnId {
        let id = FnId(self.ctx.fn_configs.len() as u32);
        self.ctx.fn_configs.push(config);
        id
    }

    pub fn push_for(&mut self, config: ForConfig) -> ForId {
        let id = ForId(self.ctx.for_configs.len() as u32);
        self.ctx.for_configs.push(config);
        id
    }

    pub fn push_object(&mut self, fields: Vec<(Span, NodeId)>) -> Option<ChildRange> {
        let start = self.ctx.object_fields.len() as u32;
        let len = u16::try_from(fields.len()).ok()?;
        self.ctx.object_fields.extend(fields);
        Some(ChildRange::new(start, len))
    }

    pub fn push_children(&mut self, items: &[NodeId]) -> Option<ChildRange> {
        let start = self.ctx.children.len() as u32;
        let len = u16::try_from(items.len()).ok()?;
        self.ctx.children.extend_from_slice(items);
        Some(ChildRange::new(start, len))
    }
}

#[derive(Clone, Copy)]
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

#[derive(Clone)]
pub enum Node {
    Grammar,
    Rule {
        is_override: bool,
        name: Span,
        body: NodeId,
    },
    Let {
        name: Span,
        ty: Option<Ty>,
        value: NodeId,
    },
    Fn(FnId),
    StringLit,
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i32),
    Ident(IdentKind),
    FieldAccess {
        obj: NodeId,
        field: Span,
    },
    QualifiedAccess {
        obj: NodeId,
        member: Span,
    },
    /// `seq(a, b, ...)` or `choice(a, b, ...)`.
    /// Children: `[member0, member1, ...]` - each is a rule expression.
    SeqOrChoice {
        seq: bool,
        range: ChildRange,
    },
    Repeat {
        kind: RepeatKind,
        inner: NodeId,
    },
    Blank,
    Field {
        name: Span,
        content: NodeId,
    },
    Alias {
        content: NodeId,
        target: NodeId,
    },
    Token {
        immediate: bool,
        inner: NodeId,
    },
    Prec {
        kind: PrecKind,
        value: NodeId,
        content: NodeId,
    },
    Reserved {
        context: Span,
        content: NodeId,
    },
    /// `concat(a, b, ...)`. Children: `[part0, part1, ...]` - each must be `str_t`.
    Concat(ChildRange),
    /// `regexp(pattern)` or `regexp(pattern, flags)`. Children layout:
    /// `[pattern]` or `[pattern, flags]` in the children vec.
    DynRegex(ChildRange),
    /// `inherit("path.tsg")` or `import("path.tsg")`. `module` is `None`
    /// after parsing, set to a global module index by the loading pre-pass.
    ModuleRef {
        import: bool,
        path: Span,
        module: Option<u8>,
    },
    /// `grammar_config(module)` - returns the grammar config of a module as an object.
    GrammarConfig(NodeId),
    /// `expr::name(args)` - function call through `::` access.
    /// Children layout: `[obj, name, arg0, arg1, ...]` where obj is the
    /// namespace value and name is the function identifier.
    QualifiedCall(ChildRange),
    Append {
        left: NodeId,
        right: NodeId,
    },
    For(ForId),
    /// `name(arg0, arg1, ...)`. `args` children: `[arg0, arg1, ...]`.
    Call {
        name: NodeId,
        args: ChildRange,
    },
    /// `[elem0, elem1, ...]`. Children: `[elem0, elem1, ...]`.
    List(ChildRange),
    /// `(elem0, elem1, ...)`. Children: `[elem0, elem1, ...]`.
    Tuple(ChildRange),
    /// `{ key0: val0, key1: val1, ... }`. Fields stored in `AstContext::object_fields`.
    Object(ChildRange),
    Neg(NodeId),
    /// Sentinel value occupying index 0 in the arena. Not part of the public API.
    #[doc(hidden)]
    Unreachable,
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

impl Node {
    /// Returns the child range for nodes that store plain `NodeId` children.
    /// `Object` and `Call` use separate arenas and are not included.
    #[must_use]
    pub const fn child_range(&self) -> Option<ChildRange> {
        match self {
            Self::SeqOrChoice { range: r, .. }
            | Self::List(r)
            | Self::Tuple(r)
            | Self::Concat(r) => Some(*r),
            _ => None,
        }
    }
}

#[derive(Clone, Default)]
pub struct GrammarConfig {
    pub language: Option<String>,
    pub inherits: Option<NodeId>,
    pub extras: Option<NodeId>,
    pub externals: Option<NodeId>,
    pub inline: Option<NodeId>,
    pub supertypes: Option<NodeId>,
    pub word: Option<NodeId>,
    pub conflicts: Option<NodeId>,
    pub precedences: Option<NodeId>,
    pub reserved: Option<NodeId>,
}

#[derive(Clone)]
pub struct FnConfig {
    pub name: Span,
    pub params: Vec<Param>,
    pub return_ty: Ty,
    pub body: NodeId,
}

#[derive(Clone)]
pub struct Param {
    pub name: Span,
    pub ty: Ty,
}

#[derive(Clone)]
pub struct ForConfig {
    pub bindings: Vec<(Span, Ty)>,
    pub iterable: NodeId,
    pub body: NodeId,
}
