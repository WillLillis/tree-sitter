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
pub struct NodeId(pub NonZeroU32);

impl NodeId {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize
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
    InheritedFromHere,
}

impl fmt::Display for NoteMessage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FirstDefinedHere => write!(f, "first defined here"),
            Self::InheritedFromHere => write!(f, "inherited from here"),
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
        // SAFETY: span boundaries are at ASCII byte positions (all token
        // delimiters are ASCII), which are always valid UTF-8 boundaries.
        // The input length is checked against u32::MAX before lexing.
        unsafe { source.get_unchecked(self.start as usize..self.end as usize) }
    }
}

pub struct Ast<'src> {
    pub nodes: Vec<Node>,
    pub root_items: Vec<NodeId>,
    pub context: AstContext<'src>,
}

/// Immutable context data split from nodes so resolve can borrow `&mut nodes`
/// while reading `&context`.
pub struct AstContext<'src> {
    pub grammar_config: Option<GrammarConfig>,
    pub fn_configs: Vec<FnConfig>,
    pub for_configs: Vec<ForConfig>,
    pub object_fields: Vec<(Span, NodeId)>,
    pub spans: Vec<Span>,
    pub children: Vec<NodeId>,
    pub source: &'src str,
}

impl AstContext<'_> {
    #[inline]
    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.spans[id.index()]
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
    #[inline]
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        &self.children[range.start as usize..range.start as usize + range.len as usize]
    }
    #[must_use]
    pub const fn source(&self) -> &str {
        self.source
    }
    #[inline]
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(self.source)
    }
    #[inline]
    #[must_use]
    pub fn node_text(&self, id: NodeId) -> &str {
        self.text(self.span(id))
    }
}

impl<'src> Ast<'src> {
    #[must_use]
    pub fn new(source: &'src str) -> Self {
        let cap = source.len() / 30;
        let mut nodes = Vec::with_capacity(cap);
        let mut spans = Vec::with_capacity(cap);
        nodes.push(Node::Unreachable);
        spans.push(Span::new(0, 0));
        Self {
            nodes,
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
        let index = self.nodes.len() as u32;
        let id = NodeId(NonZeroU32::new(index).expect("AST overflow"));
        self.nodes.push(node);
        self.context.spans.push(span);
        id
    }

    #[inline]
    #[must_use]
    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
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
    #[must_use]
    pub fn get_fn(&self, id: FnId) -> &FnConfig {
        self.context.get_fn(id)
    }
    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        self.context.get_for(id)
    }
    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[(Span, NodeId)] {
        self.context.get_object(range)
    }
    #[must_use]
    pub const fn source(&self) -> &str {
        self.context.source
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
    RuleInline {
        obj: NodeId,
        rule: NodeId,
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
    },
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
    TypeRule,
    TypeStr,
    TypeInt,
    TypeListRule,
    TypeListStr,
    TypeListInt,
    TypeListListRule,
    TypeListListStr,
    TypeListListInt,
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
