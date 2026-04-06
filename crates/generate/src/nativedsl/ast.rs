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
    /// Span boundaries are always produced by the lexer at valid UTF-8
    /// positions (ASCII token boundaries), so unchecked indexing is safe.
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
    pub spans: Vec<Span>,
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
    source: &'src str,
}

impl AstContext<'_> {
    /// Look up a function config by its index (stored in `Node::Fn(idx)`).
    #[must_use]
    pub fn get_fn(&self, idx: u32) -> &FnConfig {
        &self.fn_configs[idx as usize]
    }

    /// Look up a for-loop config by its index (stored in `Node::For(idx)`).
    #[must_use]
    pub fn get_for(&self, idx: u32) -> &ForConfig {
        &self.for_configs[idx as usize]
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
            spans,
            root_items: Vec::new(),
            context: AstContext {
                grammar_config: None,
                fn_configs: Vec::new(),
                for_configs: Vec::new(),
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
        self.spans.push(span);
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
        self.spans[id.index()]
    }

    /// Store a function config and return its index for `Node::Fn(idx)`.
    pub fn push_fn(&mut self, config: FnConfig) -> u32 {
        let idx = self.context.fn_configs.len() as u32;
        self.context.fn_configs.push(config);
        idx
    }

    #[must_use]
    pub fn get_fn(&self, idx: u32) -> &FnConfig {
        self.context.get_fn(idx)
    }

    /// Store a for-loop config and return its index for `Node::For(idx)`.
    pub fn push_for(&mut self, config: ForConfig) -> u32 {
        let idx = self.context.for_configs.len() as u32;
        self.context.for_configs.push(config);
        idx
    }

    #[must_use]
    pub fn get_for(&self, idx: u32) -> &ForConfig {
        self.context.get_for(idx)
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

/// Every AST construct lives in this enum.
///
/// The enum is kept at 32 bytes by storing large payloads (function configs,
/// for-loop configs, grammar config) out-of-line in [`AstContext`] and
/// referencing them by index.
///
/// After the resolve pass, all `Ident` nodes are replaced with either
/// `RuleRef` or `VarRef`.
#[derive(Clone, Debug)]
pub enum Node {
    // ---- Items ----
    /// Marker node - actual config stored in `Ast::grammar_config`.
    Grammar,
    /// `name` is a Span into source (static identifier).
    Rule {
        name: Span,
        body: NodeId,
    },
    Let {
        name: Span,
        ty: Option<NodeId>,
        value: NodeId,
    },
    /// Index into `Ast::fn_configs`.
    Fn(u32),

    // ---- Expressions ----
    /// String literal. Node span covers `"..."` including quotes.
    StringLit,
    /// Raw string literal. Node span covers `r"..."` or `r#"..."#` etc.
    /// Content is taken verbatim (no escape processing).
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i64),
    /// Unresolved identifier - replaced by name resolution pass.
    Ident(Span),
    /// Resolved: reference to a grammar rule.
    RuleRef(Span),
    /// Resolved: reference to a variable (let binding, fn param, for binding).
    VarRef(Span),
    FieldAccess {
        obj: NodeId,
        field: Span,
    },
    Seq(NodeList),
    Choice(NodeList),
    Repeat(NodeId),
    Repeat1(NodeId),
    Optional(NodeId),
    Blank,
    /// field(name, content) - name is a span for the field name identifier.
    Field {
        name: Span,
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
    /// reserved("context", content) - context is a span of the string literal (between quotes).
    Reserved {
        context: Span,
        content: NodeId,
    },
    Concat(NodeList),
    DynRegex {
        pattern: NodeId,
        flags: Option<NodeId>,
    },
    /// Index into `Ast::for_configs`.
    For(u32),
    Call {
        name: Span,
        args: NodeList,
    },
    List(NodeList),
    Tuple(NodeList),
    Object(Vec<(Span, NodeId)>),
    Neg(NodeId),

    // ---- Types ----
    TypeRule,
    TypeStr,
    TypeInt,
    TypeList(NodeId),
    TypeTuple(NodeList),

    // Sentinel (occupies index 0)
    Unreachable,
}

impl Node {
    /// Returns the child node list for variadic nodes (`Seq`, `Choice`,
    /// `List`, `Tuple`, `Concat`), or `None` for other variants.
    #[must_use]
    pub fn children(&self) -> Option<&[NodeId]> {
        match self {
            Self::Seq(c) | Self::Choice(c) | Self::List(c) | Self::Tuple(c) | Self::Concat(c) => {
                Some(c)
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
    pub name: Span,
    pub params: Vec<Param>,
    pub return_ty: NodeId,
    pub body: NodeId,
}

/// A function parameter: `name: type`.
#[derive(Clone, Debug)]
pub struct Param {
    pub name: Span,
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
