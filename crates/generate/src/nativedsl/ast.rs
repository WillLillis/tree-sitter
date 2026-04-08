//! Arena-based AST for the native grammar DSL.
//!
//! All AST nodes live in a single `Vec<Node>` with a parallel `Vec<Span>`
//! for source locations. Nodes reference each other through [`NodeId`], a
//! typed wrapper around `NonZeroU32` (index 0 is reserved for a sentinel).
//!
//! Compound configuration (grammar settings, function signatures, for-loop
//! bindings) is stored in [`AstContext`], separate from the node arena, so
//! that later passes can mutate nodes while borrowing context immutably.

use std::num::NonZeroU32;

use serde::Serialize;

/// Typed index into the AST node arena.
///
/// Wraps `NonZeroU32` so that `Option<NodeId>` is the same size as `NodeId`.
/// Index 0 is occupied by the `Node::Unreachable` sentinel and is never
/// handed out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(NonZeroU32);

impl NodeId {
    /// Returns the index as a `usize` for indexing into the arena vectors.
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize
    }
}

/// Typed index into `AstContext::fn_configs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FnId(u32);

impl FnId {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// Typed index into `AstContext::for_configs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ForId(u32);

impl ForId {
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.0 as usize
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

    /// Returns a span covering both `self` and `other`.
    #[must_use]
    pub fn merge(self, other: Self) -> Self {
        Self {
            start: self.start.min(other.start),
            end: self.end.max(other.end),
        }
    }

    /// Resolve this span to a `&str` slice of the source.
    ///
    /// # Safety (internal)
    /// Span boundaries are produced by the lexer at ASCII byte positions
    /// (all token delimiters are ASCII), which are always valid UTF-8
    /// boundaries.
    #[must_use]
    pub fn resolve<'src>(&self, source: &'src str) -> &'src str {
        unsafe { source.get_unchecked(self.start as usize..self.end as usize) }
    }
}

/// The complete AST for a grammar source file.
///
/// Nodes and spans are stored in parallel vectors indexed by [`NodeId`].
/// Mutable and immutable data are split: `nodes` and `spans` can be mutated
/// by later passes (e.g. resolve) while `context` remains shared/immutable.
pub struct Ast<'src> {
    pub nodes: Vec<Node>,
    pub root_items: Vec<NodeId>,
    pub context: AstContext<'src>,
}

/// Immutable context data that accompanies the AST nodes.
///
/// Separated from `nodes`/`spans` so that passes like resolve can borrow
/// `&mut ast.nodes` while simultaneously reading `&ast.context`. Holds the
/// source text, grammar configuration, and all function/for-loop configs.
pub struct AstContext<'src> {
    pub grammar_config: Option<GrammarConfig>,
    pub fn_configs: Vec<FnConfig>,
    pub for_configs: Vec<ForConfig>,
    /// Flat storage for object field (key, value) pairs, indexed by `ChildRange`.
    pub object_fields: Vec<(Span, NodeId)>,
    pub spans: Vec<Span>,
    /// Shared storage for variadic node children (Seq, Choice, List, etc.).
    pub children: Vec<NodeId>,
    source: &'src str,
}

impl AstContext<'_> {
    /// Look up the span for a node.
    #[inline]
    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.spans[id.index()]
    }

    /// Look up a function config by its index (stored in `Node::Fn(id)`).
    #[must_use]
    pub fn get_fn(&self, id: FnId) -> &FnConfig {
        &self.fn_configs[id.index()]
    }

    /// Look up a for-loop config by its index (stored in `Node::For(id)`).
    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        &self.for_configs[id.index()]
    }

    /// Look up object fields by range (stored in `Node::Object`).
    #[must_use]
    pub fn get_object(&self, range: ChildRange) -> &[(Span, NodeId)] {
        &self.object_fields[range.start as usize..range.start as usize + range.len as usize]
    }

    /// Resolve a `ChildRange` to a slice of child NodeIds.
    #[inline]
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        &self.children[range.start as usize..range.start as usize + range.len as usize]
    }

    /// Returns the full source text.
    #[must_use]
    pub const fn source(&self) -> &str {
        self.source
    }

    /// Resolve a span to the corresponding `&str` slice of source.
    #[inline]
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(self.source)
    }

    /// Look up the source text for a node by its ID.
    #[inline]
    #[must_use]
    pub fn node_text(&self, id: NodeId) -> &str {
        self.text(self.span(id))
    }
}

impl<'src> Ast<'src> {
    /// Create a new AST arena pre-allocated based on source length.
    ///
    /// Pushes a `Node::Unreachable` sentinel at index 0 so that all valid
    /// `NodeId`s use `NonZeroU32`.
    #[must_use]
    pub fn new(source: &'src str) -> Self {
        let estimated_nodes = source.len() / 30;
        let mut nodes = Vec::with_capacity(estimated_nodes);
        let mut spans = Vec::with_capacity(estimated_nodes);
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
                children: Vec::with_capacity(estimated_nodes),
                source,
            },
        }
    }

    /// Append a node and its span to the arena, returning its [`NodeId`].
    ///
    /// # Panics
    ///
    /// Panics if the arena exceeds `u32::MAX` nodes.
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
    pub fn node_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.index()]
    }

    #[inline]
    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.context.spans[id.index()]
    }

    /// Store a function config and return its [`FnId`] for `Node::Fn`.
    pub fn push_fn(&mut self, config: FnConfig) -> FnId {
        let id = FnId(self.context.fn_configs.len() as u32);
        self.context.fn_configs.push(config);
        id
    }

    #[must_use]
    pub fn get_fn(&self, id: FnId) -> &FnConfig {
        self.context.get_fn(id)
    }

    /// Store a for-loop config and return its [`ForId`] for `Node::For`.
    pub fn push_for(&mut self, config: ForConfig) -> ForId {
        let id = ForId(self.context.for_configs.len() as u32);
        self.context.for_configs.push(config);
        id
    }

    #[must_use]
    pub fn get_for(&self, id: ForId) -> &ForConfig {
        self.context.get_for(id)
    }

    /// Store object fields and return a `ChildRange` for `Node::Object`.
    pub fn push_object(&mut self, fields: Vec<(Span, NodeId)>) -> ChildRange {
        let start = self.context.object_fields.len() as u32;
        let len = fields.len() as u16;
        self.context.object_fields.extend(fields);
        ChildRange::new(start, len)
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

    /// Look up the source text for a node by its ID.
    #[inline]
    #[must_use]
    pub fn node_text(&self, id: NodeId) -> &str {
        self.context.node_text(id)
    }

    /// Push a list of child NodeIds into the shared children vector,
    /// returning a `ChildRange` referencing them.
    pub fn push_children(&mut self, items: &[NodeId]) -> ChildRange {
        let start = self.context.children.len() as u32;
        let len = items.len() as u16;
        self.context.children.extend_from_slice(items);
        ChildRange::new(start, len)
    }

    /// Resolve a `ChildRange` to a slice of child NodeIds.
    #[inline]
    #[must_use]
    pub fn child_slice(&self, range: ChildRange) -> &[NodeId] {
        self.context.child_slice(range)
    }

    /// Get the raw content of a string literal (between quotes, not unescaped).
    #[must_use]
    pub fn string_lit_content(&self, id: NodeId) -> &str {
        let span = self.span(id);
        unsafe {
            self.context
                .source
                .get_unchecked((span.start + 1) as usize..(span.end - 1) as usize)
        }
    }

    /// Check if a string literal contains escape sequences.
    #[must_use]
    pub fn string_lit_has_escapes(&self, id: NodeId) -> bool {
        self.string_lit_content(id).contains('\\')
    }

    /// Get the raw content of a raw string literal (between delimiters).
    #[must_use]
    pub fn raw_string_lit_content(&self, id: NodeId, hash_count: u8) -> &str {
        let span = self.span(id);
        let prefix_len = 2 + u32::from(hash_count);
        let suffix_len = 1 + u32::from(hash_count);
        unsafe {
            self.context
                .source
                .get_unchecked((span.start + prefix_len) as usize..(span.end - suffix_len) as usize)
        }
    }
}

/// A list of child node references, used by variadic constructs like `Seq`
/// and `Choice`.
pub type NodeList = Vec<NodeId>;

/// A compact reference to a range of children in `Ast::children`.
///
/// Replaces inline `Vec<NodeId>` in Node variants to keep the enum small.
/// 6 bytes instead of 24.
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

/// Every AST construct lives in this enum.
///
/// The enum is kept at 16 bytes (4 nodes per 64-byte cache line) by storing
/// large payloads out-of-line in [`AstContext`] - function/for configs by
/// index, variadic children via [`ChildRange`], and identifier spans via the
/// parallel spans vector.
///
/// After the resolve pass, all `Ident` nodes are replaced with either
/// `RuleRef` or `VarRef`.
#[derive(Clone, Debug)]
pub enum Node {
    // ---- Items ----
    /// Marker node - actual config stored in `Ast::grammar_config`.
    Grammar,
    Rule {
        name: NodeId,
        body: NodeId,
    },
    /// Override rule definition - replaces a rule from the base grammar.
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

    // ---- Expressions ----
    /// String literal. Node span covers `"..."` including quotes.
    StringLit,
    /// Raw string literal. Node span covers `r"..."` or `r#"..."#` etc.
    /// Content is taken verbatim (no escape processing).
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i32),
    /// Unresolved identifier - replaced by name resolution pass.
    /// Span comes from the parallel spans vec.
    Ident,
    /// Resolved: reference to a grammar rule.
    RuleRef,
    /// Resolved: reference to a variable (let binding, fn param, for binding).
    VarRef,
    FieldAccess {
        obj: NodeId,
        field: NodeId,
    },
    /// `c::expression` - inline the body of a rule from a base grammar.
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
    /// `inherit("path/to/grammar")` - loads a base grammar for inheritance.
    /// The `path` NodeId points to the string literal containing the path.
    Inherit {
        path: NodeId,
    },
    /// `append(list1, list2)` - concatenates two lists.
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
    /// Range into `AstContext::object_fields`.
    Object(ChildRange),
    Neg(NodeId),

    // ---- Types ----
    TypeRule,
    TypeStr,
    TypeInt,
    TypeList(NodeId),
    TypeTuple(ChildRange),

    // Sentinel (occupies index 0)
    Unreachable,
}

impl Node {
    /// Returns the child range for variadic nodes (`Seq`, `Choice`,
    /// `List`, `Tuple`, `Concat`), or `None` for other variants.
    /// Use `Ast::child_slice()` to resolve the range to a `&[NodeId]`.
    #[must_use]
    pub fn child_range(&self) -> Option<ChildRange> {
        match self {
            Self::Seq(r) | Self::Choice(r) | Self::List(r) | Self::Tuple(r) | Self::Concat(r) => {
                Some(*r)
            }
            _ => None,
        }
    }
}

/// Configuration from the `grammar { ... }` block at the top of a source file.
///
/// All identifier-like fields (extras, externals, word, conflicts, etc.) store
/// [`Span`]s or [`NodeId`]s pointing back into the source/AST rather than owned
/// strings, to avoid allocations during parsing.
#[derive(Clone, Debug, Default)]
pub struct GrammarConfig {
    /// Language name - processed string (from string literal).
    pub language: Option<String>,
    /// Base grammar to inherit from (the `inherits:` field).
    pub inherits: Option<NodeId>,
    pub extras: NodeList,
    pub externals: NodeList,
    /// Conflict groups - identifiers stored as spans.
    pub conflicts: Vec<Vec<Span>>,
    pub precedences: Vec<Vec<PrecEntry>>,
    /// Inline rule names - spans.
    pub inline: Vec<Span>,
    /// Supertype rule names - spans.
    pub supertypes: Vec<Span>,
    /// Word token name - span.
    pub word: Option<Span>,
    pub reserved: Vec<ReservedWordSet>,
}

/// An entry in a precedence ordering: either a quoted string name or a bare
/// rule identifier.
#[derive(Clone, Debug)]
pub enum PrecEntry {
    /// Quoted string - processed value.
    Name(String),
    /// Bare identifier - span into source.
    Symbol(Span),
}

/// A named set of reserved words (e.g. `reserved { default: [...] }`).
#[derive(Clone, Debug)]
pub struct ReservedWordSet {
    /// Context name - span (identifier).
    pub name: Span,
    pub words: NodeList,
}

/// Configuration for a user-defined function (`fn name(params) -> ty { body }`).
///
/// Stored out-of-line in `AstContext::fn_configs` and referenced by index
/// from `Node::Fn(idx)`.
#[derive(Clone, Debug)]
pub struct FnConfig {
    pub name: NodeId,
    pub params: Vec<Param>,
    pub return_ty: NodeId,
    pub body: NodeId,
}

/// A function parameter: `name: type`.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: NodeId,
    pub ty: NodeId,
}

/// Configuration for a `for` expression (`for (bindings) in iterable { body }`).
///
/// Stored out-of-line in `AstContext::for_configs` and referenced by index
/// from `Node::For(idx)`.
#[derive(Clone, Debug)]
pub struct ForConfig {
    /// (`binding_name` span, `type_node_id`)
    pub bindings: Vec<(Span, NodeId)>,
    pub iterable: NodeId,
    pub body: NodeId,
}
