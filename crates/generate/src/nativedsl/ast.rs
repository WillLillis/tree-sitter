//! Arena-based AST for the native grammar DSL.
//!
//! Nodes and spans live in parallel vectors indexed by [`NodeId`]. Compound
//! config (grammar settings, fn/for configs) is in [`AstContext`], separate
//! from nodes so later passes can mutate nodes while reading context.

use std::{fmt, num::NonZeroU32, path::PathBuf};

use serde::Serialize;

#[derive(Debug)]
pub struct CapacityError;

impl fmt::Display for CapacityError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "too many elements (maximum {})", u16::MAX)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(NonZeroU32);

impl NodeId {
    /// The first valid NodeId (index 1, after the sentinel).
    pub const FIRST: Self = Self(unsafe { NonZeroU32::new_unchecked(1) });

    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize
    }

    /// Return the next NodeId (index + 1).
    #[inline]
    #[must_use]
    pub const fn next(self) -> Self {
        // SAFETY: self.0 >= 1, so self.0 + 1 >= 2, always non-zero.
        Self(unsafe { NonZeroU32::new_unchecked(self.0.get() + 1) })
    }
}

/// Arena for AST nodes. Hides the sentinel at index 0 and enforces
/// [`NodeId`]-based access so callers can't accidentally index at 0
/// or iterate from the wrong starting point.
pub struct NodeArena {
    nodes: Vec<Node>,
}

impl NodeArena {
    #[must_use]
    pub fn new(estimated_cap: usize) -> Self {
        let mut nodes = Vec::with_capacity(estimated_cap);
        nodes.push(Node::Unreachable);
        Self { nodes }
    }

    pub fn push(&mut self, node: Node) -> NodeId {
        let index = self.nodes.len() as u32;
        // SAFETY: nodes[0] is always the Unreachable sentinel, so len() >= 1.
        let id = NodeId(unsafe { NonZeroU32::new_unchecked(index) });
        self.nodes.push(node);
        id
    }

    #[inline]
    #[must_use]
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    #[inline]
    pub fn set(&mut self, id: NodeId, node: Node) {
        self.nodes[id.index()] = node;
    }

    /// Number of nodes (excluding sentinel).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len() - 1
    }

    /// Iterate all valid [`NodeId`]s.
    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        (1..self.nodes.len()).map(|i| {
            // SAFETY: i starts at 1, always non-zero.
            NodeId(unsafe { NonZeroU32::new_unchecked(i as u32) })
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FnId(u32);

impl FnId {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForId(u32);

impl ForId {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Secondary annotation on an error, pointing to a related source location.
#[derive(Clone, Debug, Serialize)]
pub struct Note {
    pub message: NoteMessage,
    pub span: Span,
    #[serde(skip)]
    pub path: PathBuf,
    #[serde(skip)]
    pub source: String,
}

#[derive(Clone, Debug, Serialize)]
pub enum NoteMessage {
    FirstDefinedHere,
    ReferencedFromHere,
}

impl fmt::Display for NoteMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstDefinedHere => write!(f, "first defined here"),
            Self::ReferencedFromHere => write!(f, "referenced from here"),
        }
    }
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
    #[inline]
    #[must_use]
    pub fn resolve<'src>(&self, source: &'src str) -> &'src str {
        // SAFETY: span boundaries are at ASCII byte positions (all token
        // delimiters are ASCII), which are always valid UTF-8 boundaries.
        // The input length is checked against u32::MAX before lexing.
        unsafe { source.get_unchecked(self.start as usize..self.end as usize) }
    }
}

pub struct Ast {
    pub arena: NodeArena,
    pub root_items: Vec<NodeId>,
    pub context: AstContext,
}

/// Immutable context data split from nodes so resolve can borrow
/// `&mut arena` while reading `&context`.
pub struct AstContext {
    pub grammar_config: Option<GrammarConfig>,
    pub fn_configs: Vec<FnConfig>,
    pub for_configs: Vec<ForConfig>,
    pub object_fields: Vec<(Span, NodeId)>,
    pub spans: Vec<Span>,
    pub children: Vec<NodeId>,
    pub source: String,
}

impl AstContext {
    #[inline]
    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.spans[id.index()]
    }
    #[inline]
    #[must_use]
    pub fn get_fn(&self, id: FnId) -> &FnConfig {
        &self.fn_configs[id.index()]
    }
    #[inline]
    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        &self.for_configs[id.index()]
    }
    #[inline]
    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[(Span, NodeId)] {
        &self.object_fields[range.start as usize..range.start as usize + range.len as usize]
    }
    #[inline]
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        &self.children[range.start as usize..range.start as usize + range.len as usize]
    }
    /// Unpack a `QualifiedCall(range)` into `(obj, name, &[args])`.
    #[inline]
    #[must_use]
    pub fn get_qualified_call(&self, range: ChildRange) -> (NodeId, NodeId, &[NodeId]) {
        let children = self.child_slice(range);
        // SAFETY: QualifiedCall nodes are always constructed with [obj, name, ...args]
        // by the parser, so len >= 2 is structurally guaranteed.
        unsafe {
            (
                *children.get_unchecked(0),
                *children.get_unchecked(1),
                children.get_unchecked(2..),
            )
        }
    }
    #[inline]
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }
    #[inline]
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(&self.source)
    }
    #[inline]
    #[must_use]
    pub fn node_text(&self, id: NodeId) -> &str {
        self.text(self.span(id))
    }
}

impl Ast {
    #[must_use]
    pub fn new(source: String) -> Self {
        let cap = source.len() / 30;
        let mut spans = Vec::with_capacity(cap);
        // Sentinel span parallel to NodeArena's sentinel node at index 0.
        // TODO: consider moving spans into NodeArena to co-locate these.
        spans.push(Span::new(0, 0));
        Self {
            arena: NodeArena::new(cap),
            root_items: Vec::new(),
            context: AstContext {
                grammar_config: None,
                fn_configs: Vec::new(),
                for_configs: Vec::new(),
                object_fields: Vec::new(),
                spans,
                children: Vec::with_capacity(cap),
                source,
            },
        }
    }

    #[inline]
    pub fn push(&mut self, node: Node, span: Span) -> NodeId {
        let id = self.arena.push(node);
        self.context.spans.push(span);
        id
    }

    #[inline]
    #[must_use]
    pub fn node(&self, id: NodeId) -> &Node {
        self.arena.get(id)
    }
    #[inline]
    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.context.spans[id.index()]
    }

    pub fn push_fn(&mut self, config: FnConfig) -> FnId {
        let id = FnId(self.context.fn_configs.len() as u32);
        self.context.fn_configs.push(config);
        id
    }

    pub fn push_for(&mut self, config: ForConfig) -> ForId {
        let id = ForId(self.context.for_configs.len() as u32);
        self.context.for_configs.push(config);
        id
    }

    pub fn push_object(
        &mut self,
        fields: Vec<(Span, NodeId)>,
    ) -> Result<ChildRange, CapacityError> {
        let start = self.context.object_fields.len() as u32;
        let len = u16::try_from(fields.len()).map_err(|_| CapacityError)?;
        self.context.object_fields.extend(fields);
        Ok(ChildRange::new(start, len))
    }

    pub fn push_children(&mut self, items: &[NodeId]) -> Result<ChildRange, CapacityError> {
        let start = self.context.children.len() as u32;
        let len = u16::try_from(items.len()).map_err(|_| CapacityError)?;
        self.context.children.extend_from_slice(items);
        Ok(ChildRange::new(start, len))
    }

    // Delegating accessors used by all pipeline stages
    #[inline]
    #[must_use]
    pub fn get_fn(&self, id: FnId) -> &FnConfig {
        self.context.get_fn(id)
    }
    #[inline]
    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        self.context.get_for(id)
    }
    #[inline]
    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[(Span, NodeId)] {
        self.context.get_object(range)
    }
    #[inline]
    #[must_use]
    pub fn get_qualified_call(&self, range: ChildRange) -> (NodeId, NodeId, &[NodeId]) {
        self.context.get_qualified_call(range)
    }
    #[inline]
    #[must_use]
    pub fn source(&self) -> &str {
        &self.context.source
    }
    #[inline]
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        self.context.text(span)
    }
    #[inline]
    #[must_use]
    pub fn node_text(&self, id: NodeId) -> &str {
        self.context.node_text(id)
    }
    #[inline]
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        self.context.child_slice(range)
    }
}

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
}

#[derive(Clone, Debug)]
pub enum Node {
    Grammar,
    Rule {
        name: NodeId,
        body: NodeId,
    },
    OverrideRule {
        name: NodeId,
        body: NodeId,
    },
    Let {
        name: NodeId,
        ty: Option<NodeId>,
        value: NodeId,
    },
    Fn(FnId),
    StringLit,
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i32),
    Ident,
    RuleRef,
    VarRef,
    FieldAccess {
        obj: NodeId,
        field: NodeId,
    },
    QualifiedAccess {
        obj: NodeId,
        member: NodeId,
    },
    Seq(ChildRange),
    Choice(ChildRange),
    Repeat(NodeId),
    Repeat1(NodeId),
    Optional(NodeId),
    Blank,
    Field {
        name: NodeId,
        content: NodeId,
    },
    Alias {
        content: NodeId,
        target: NodeId,
    },
    Token(NodeId),
    TokenImmediate(NodeId),
    Prec {
        value: NodeId,
        content: NodeId,
    },
    PrecLeft {
        value: NodeId,
        content: NodeId,
    },
    PrecRight {
        value: NodeId,
        content: NodeId,
    },
    PrecDynamic {
        value: NodeId,
        content: NodeId,
    },
    Reserved {
        context: NodeId,
        content: NodeId,
    },
    Concat(ChildRange),
    DynRegex {
        pattern: NodeId,
        flags: Option<NodeId>,
    },
    Inherit {
        path: NodeId,
        /// Module index, set by the loading pre-pass.
        module: Option<u8>,
    },
    /// `grammar_config(module)` - returns the grammar config of a module as an object.
    GrammarConfig(NodeId),
    /// `import("path.tsg")` expression - loads a helper file, returns module_t.
    /// `module` is `None` after parsing, set to an index into `Vec<Module>`
    /// by the loading pre-pass.
    Import {
        path: NodeId,
        module: Option<u8>,
    },
    /// `expr::name(args)` - function call through `::` access.
    /// Children layout: `[obj, name, arg0, arg1, ...]` where obj is the
    /// namespace value and name is the function identifier.
    QualifiedCall(ChildRange),
    Append {
        left: NodeId,
        right: NodeId,
    },
    For(ForId),
    Call {
        name: NodeId,
        args: ChildRange,
    },
    List(ChildRange),
    Tuple(ChildRange),
    Object(ChildRange),
    Neg(NodeId),
    /// Top-level `print(expr)` debugging item. Evaluates `expr` during
    /// lowering and writes a representation of the value to stderr. Returns
    /// `Ty::Void` from typecheck - only valid as a top-level item, never
    /// inside an expression that expects a value.
    Print(NodeId),
    TypeRule,
    TypeStr,
    TypeInt,
    TypeListRule,
    TypeListStr,
    TypeListInt,
    TypeListListRule,
    TypeListListStr,
    TypeListListInt,
    /// `void_t` annotation. Parseable so users get a clear error message
    /// instead of "unknown type"; always rejected by `resolve_type_annotation`.
    TypeVoid,
    /// `spread_t` annotation. Like `TypeVoid`, parseable but always rejected -
    /// `Ty::Spread` is an internal type produced only by for-loop expansions.
    TypeSpread,
    Unreachable,
}

impl Node {
    /// Returns the child range for nodes that store plain `NodeId` children.
    /// `Object` and `Call` use separate arenas and are not included.
    #[must_use]
    pub const fn child_range(&self) -> Option<ChildRange> {
        match self {
            Self::Seq(r) | Self::Choice(r) | Self::List(r) | Self::Tuple(r) | Self::Concat(r) => {
                Some(*r)
            }
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default)]
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

#[derive(Clone, Debug)]
pub struct FnConfig {
    pub name: NodeId,
    pub params: Vec<Param>,
    pub return_ty: NodeId,
    pub body: NodeId,
}

#[derive(Clone, Debug)]
pub struct Param {
    pub name: NodeId,
    pub ty: NodeId,
}

#[derive(Clone, Debug)]
pub struct ForConfig {
    pub bindings: Vec<(Span, NodeId)>,
    pub iterable: NodeId,
    pub body: NodeId,
}
