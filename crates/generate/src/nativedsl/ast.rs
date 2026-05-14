//! Arena-based AST for the native grammar DSL.

use std::num::NonZeroU32;
use std::path::PathBuf;

use serde::Serialize;

use super::typecheck::Ty;
use super::{ModuleId, Note, NoteMessage};

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NodeId(NonZeroU32);

impl NodeId {
    pub const FIRST: Self = Self(NonZeroU32::new(1).unwrap());

    #[must_use]
    pub const fn index(self) -> usize {
        self.0.get() as usize
    }

    /// Create a `NodeId` from a 1-based index.
    ///
    /// # Panics
    ///
    /// Panics if `index` is 0.
    #[must_use]
    pub const fn from_index(index: usize) -> Self {
        Self(NonZeroU32::new(index as u32).unwrap())
    }

    #[must_use]
    pub const fn next(self) -> Self {
        // SAFETY: self.0 >= 1 and < u32::MAX (arena is bounded well below that),
        // so self.0 + 1 >= 2 and fits in u32. Always non-zero.
        debug_assert!(self.0.get() < u32::MAX);
        Self(unsafe { NonZeroU32::new_unchecked(self.0.get() + 1) })
    }
}

impl From<NodeId> for u32 {
    fn from(id: NodeId) -> Self {
        id.0.get()
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
        debug_assert!(index >= 1);
        let id = NodeId(unsafe { NonZeroU32::new_unchecked(index) });
        self.nodes.push(node);
        self.spans.push(span);
        id
    }

    #[must_use]
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.index()]
    }

    pub fn get_mut(&mut self, id: NodeId) -> &mut Node {
        &mut self.nodes[id.index()]
    }

    #[must_use]
    pub fn span(&self, id: NodeId) -> Span {
        self.spans[id.index()]
    }

    pub fn set(&mut self, id: NodeId, node: Node) {
        self.nodes[id.index()] = node;
    }

    pub fn set_span(&mut self, id: NodeId, span: Span) {
        self.spans[id.index()] = span;
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

    pub fn node_ids(&self) -> impl Iterator<Item = NodeId> + '_ {
        (1..self.nodes.len()).map(|i| {
            // SAFETY: i starts at 1, always non-zero.
            NodeId(unsafe { NonZeroU32::new_unchecked(i as u32) })
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &Node)> {
        self.node_ids().map(|id| (id, &self.nodes[id.index()]))
    }

    /// Iterate a half-open `[start, end)` slice of `NodeId`s and their nodes.
    /// Caller must ensure both bounds are valid arena indices.
    pub(super) fn iter_range(
        &self,
        range: std::ops::Range<u32>,
    ) -> impl Iterator<Item = (NodeId, &Node)> {
        range.map(|i| {
            // SAFETY: i in [start, end), both produced by ModuleContext after
            // a successful Parser::parse, so within self.nodes bounds.
            let id = NodeId(unsafe { NonZeroU32::new_unchecked(i) });
            (id, unsafe { self.nodes.get_unchecked(i as usize) })
        })
    }
}

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrecKind {
    Default,
    Left,
    Right,
    Dynamic,
}

/// All grammar config fields. Used for parsing the grammar block, `grammar_config()`
/// access (Language/Inherits excluded by the parser), and iterating config node fields.
#[derive(Clone, Copy, Debug)]
pub enum ConfigField {
    Language,
    Inherits,
    Extras,
    Externals,
    Supertypes,
    Inline,
    Word,
    Conflicts,
    Precedences,
    Reserved,
    Start,
    Flags,
}

impl ConfigField {
    pub const COUNT: usize = 12;
}

impl TryFrom<&str> for ConfigField {
    type Error = ();
    fn try_from(value: &str) -> Result<Self, Self::Error> {
        Ok(match value {
            "language" => Self::Language,
            "inherits" => Self::Inherits,
            "extras" => Self::Extras,
            "externals" => Self::Externals,
            "supertypes" => Self::Supertypes,
            "inline" => Self::Inline,
            "word" => Self::Word,
            "conflicts" => Self::Conflicts,
            "precedences" => Self::Precedences,
            "reserved" => Self::Reserved,
            "start" => Self::Start,
            "flags" => Self::Flags,
            _ => return Err(()),
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    Optional,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdentKind {
    Unresolved,
    Rule,
    Var(NodeId),
    Macro(MacroId),
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
    #[must_use]
    pub const fn strip_quotes(self) -> Self {
        debug_assert!(self.end >= self.start + 2);
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

    /// Caller must pair the span with the source it was lexed from.
    #[must_use]
    pub(super) fn resolve(self, source: &str) -> &str {
        // SAFETY: lexer emits spans at UTF-8 char boundaries within `source`.
        unsafe { source.get_unchecked(self.start as usize..self.end as usize) }
    }
}

/// Shared AST data across all modules in a grammar. All `NodeId`, `MacroId`,
/// `ForId`, and `ChildRange` values are globally valid within this structure.
///
/// The `arena` and `pools` fields are separate to allow split borrowing:
/// resolve functions can mutate `arena` while reading `pools`.
pub struct SharedAst {
    pub arena: NodeArena,
    pub pools: AstPools,
}

/// Indexed pool data backing `MacroId`, `ForId`, and `ChildRange`.
/// Separated from `NodeArena` so that name resolution can hold
/// `&mut arena` and `&pools` simultaneously.
pub struct AstPools {
    pub children: Vec<NodeId>,
    pub macro_configs: Vec<MacroConfig>,
    pub for_configs: Vec<ForConfig>,
    pub object_fields: Vec<(Span, NodeId)>,
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
            },
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

    pub fn push_object(&mut self, fields: Vec<(Span, NodeId)>) -> Option<ChildRange> {
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

    #[must_use]
    pub fn get_macro(&self, id: MacroId) -> &MacroConfig {
        &self.macro_configs[id.index()]
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

#[derive(Clone)]
pub struct MacroConfig {
    pub name: Span,
    pub params: Vec<Param>,
    pub return_ty: Ty,
    pub body: NodeId,
}

#[derive(Copy, Clone)]
pub struct Param {
    pub name: Span,
    pub ty: Ty,
}

#[derive(Clone)]
pub struct ForConfig {
    pub bindings: Vec<Param>,
    pub iterable: NodeId,
}

/// Per-module data produced by the parser.
///
/// Each loaded module (root, inherited, imported) has its own `ModuleContext`
/// containing source text and module-specific configuration, while sharing a
/// single [`SharedAst`] for all node data.
#[derive(Debug)]
pub struct ModuleContext {
    pub source: String,
    pub path: PathBuf,
    pub grammar_config: Option<GrammarConfig>,
    pub root_items: Vec<NodeId>,
    /// The `inherit()` `ModuleRef` node, if this module has one.
    /// Set by the parser when it encounters `inherit(...)`.
    pub inherit_ref: Option<NodeId>,
    /// All `ModuleRef` nodes (`import(...)` and `inherit(...)`) in source order,
    /// collected by the parser so the loader can iterate without scanning the arena.
    pub module_refs: Vec<NodeId>,
    /// Spans of all top-level `external <name>` decls' names, in source order.
    /// Populated by the parser. Used by resolve and lower to look up
    /// `helper::external_name` references without scanning `root_items`.
    pub external_names: Vec<Span>,
    /// Half-open `[start, end)` range of `NodeId`s this module owns in the
    /// shared arena. The parser pushes all of a module's nodes contiguously
    /// before any child loads, so this slice is well-defined and stable.
    /// Populated by `Parser::parse`; default `0..0` means "uninitialized".
    pub(super) node_range: std::ops::Range<u32>,
}

impl ModuleContext {
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(&self.source)
    }

    /// The resolved inherited-module index and its `inherit(...)` call span,
    /// once the loader has populated it. `None` for non-inheriting modules
    /// or before child loading completes.
    #[must_use]
    pub fn inherit_module(&self, arena: &NodeArena) -> Option<(ModuleId, Span)> {
        let id = self.inherit_ref?;
        let &Node::ModuleRef {
            module: Some(idx), ..
        } = arena.get(id)
        else {
            return None;
        };
        Some((idx, arena.span(id)))
    }

    /// Iterate just this module's own nodes, paired with their `NodeId`s.
    /// Avoids scanning unrelated modules' entries in the shared arena.
    pub fn iter_own_nodes<'a>(
        &self,
        arena: &'a NodeArena,
    ) -> impl Iterator<Item = (NodeId, &'a Node)> {
        arena.iter_range(self.node_range.clone())
    }

    /// `true` if `id` was produced while parsing this module.
    #[must_use]
    pub fn owns_node(&self, id: NodeId) -> bool {
        self.node_range.contains(&(id.index() as u32))
    }

    /// Build a [`Note`] anchored to this module's source.
    #[must_use]
    pub fn note(&self, message: NoteMessage, span: Span) -> Note {
        Note {
            message,
            span,
            path: self.path.clone(),
            source: self.source.clone(),
        }
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

    #[must_use]
    pub const fn as_range(self) -> std::ops::Range<usize> {
        self.start as usize..self.start as usize + self.len as usize
    }
}

#[derive(Clone, Copy, Debug)]
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
    Macro(MacroId),
    /// `external <name>` top-level declaration. Forward-declares an
    /// externally-provided symbol name. In a grammar file, redundant with
    /// the grammar block's `externals: [...]` list. In a helper file, the
    /// only way to expose a scanner symbol via qualified access.
    External {
        name: Span,
    },
    StringLit,
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i64),
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
    /// `regexp(pattern)` or `regexp(pattern, flags)`.
    DynRegex {
        pattern: NodeId,
        flags: Option<NodeId>,
    },
    /// `inherit("path.tsg")` or `import("path.tsg")`. `module` is `None`
    /// after parsing, set to a global module index by the loading pre-pass.
    ModuleRef {
        import: bool,
        path: Span,
        module: Option<ModuleId>,
    },
    /// `grammar_config(module, field)` - access a specific config field from
    /// an inherited grammar module.
    GrammarConfig {
        module: NodeId,
        field: ConfigField,
    },
    /// `expr::name(args)` - macro call through `::` access.
    /// Children layout: `[obj, name, arg0, arg1, ...]` where obj is the
    /// namespace value and name is the macro identifier.
    QualifiedCall(ChildRange),
    Append {
        left: NodeId,
        right: NodeId,
    },
    For {
        for_id: ForId,
        body: NodeId,
    },
    /// `name(arg0, arg1, ...)`. `args` children: `[arg0, arg1, ...]`.
    Call {
        name: NodeId,
        args: ChildRange,
    },
    /// `[elem0, elem1, ...]`. Children: `[elem0, elem1, ...]`.
    List(ChildRange),
    /// `(elem0, elem1, ...)`. Children: `[elem0, elem1, ...]`.
    Tuple(ChildRange),
    /// `{ key0: val0, key1: val1, ... }`. Fields stored in `SharedAst::object_fields`.
    Object(ChildRange),
    Neg(NodeId),
    /// Parameter reference in a macro body. `index` is the position into the
    /// call's arg list; `ty` lets the typechecker avoid a macro-context lookup.
    MacroParam {
        ty: Ty,
        index: u8,
    },
    /// Binding reference in a for-loop body. `for_id` disambiguates the
    /// enclosing frame (for-loops nest); `index` selects within it; `ty`
    /// lets the typechecker avoid a pool lookup.
    ForBinding {
        for_id: ForId,
        ty: Ty,
        index: u8,
    },
    /// `#[cfg(NAME)] ITEM` — gates `child` on the named flag. Survives parse;
    /// dropped (or unwrapped) by the `apply_cfg` pass before resolve.
    Cfg {
        name: Span,
        child: NodeId,
    },
    /// Sentinel value occupying index 0 in the arena. Not part of the public API.
    #[doc(hidden)]
    Unreachable,
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

impl Node {
    /// Range of children for variadic nodes whose children are `NodeId`s
    /// directly: `SeqOrChoice`, `List`, `Tuple`, and `Concat`.
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

    /// Mutable counterpart to [`Self::child_range`]. Lets callers shrink a
    /// variadic node's range in place (e.g. after dropping cfg-gated members)
    /// without rebuilding the whole `Node` value.
    pub fn child_range_mut(&mut self) -> Option<&mut ChildRange> {
        match self {
            Self::SeqOrChoice { range: r, .. }
            | Self::List(r)
            | Self::Tuple(r)
            | Self::Concat(r) => Some(r),
            _ => None,
        }
    }
}

#[derive(Clone, Default, Debug)]
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
    pub start: Option<NodeId>,
    pub flags: Option<NodeId>,
}

impl GrammarConfig {
    /// Iterate all node-valued config fields with their kind (excludes `language`).
    pub fn node_fields(&self) -> impl Iterator<Item = (ConfigField, NodeId)> + '_ {
        use ConfigField as C;
        #[rustfmt::skip]
        let fields = [
            (C::Inherits, self.inherits),     (C::Extras, self.extras),
            (C::Externals, self.externals),   (C::Inline, self.inline),
            (C::Supertypes, self.supertypes), (C::Word, self.word),
            (C::Conflicts, self.conflicts),   (C::Precedences, self.precedences),
            (C::Reserved, self.reserved),     (C::Start, self.start),
            (C::Flags, self.flags),
        ];
        fields
            .into_iter()
            .filter_map(|(f, opt)| opt.map(|id| (f, id)))
    }
}
