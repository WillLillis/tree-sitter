//! Per-module parser output: [`ModuleContext`], the grammar block
//! ([`GrammarConfig`]/[`ConfigField`]), and their accessors.

use std::path::PathBuf;

use rustc_hash::FxHashMap;

use super::{Node, NodeArena, NodeId, Span, Spanned};
use crate::nativedsl::string_pool::Str;
use crate::nativedsl::typecheck::Ty;
use crate::nativedsl::{ModuleId, Note, NoteMessage};

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

/// Per-module data produced by the parser.
///
/// Each loaded module (root, inherited, imported) has its own `ModuleContext`
/// containing source text and module-specific configuration, while sharing a
/// single [`SharedAst`](super::SharedAst) for all node data.
#[derive(Debug)]
pub struct ModuleContext {
    pub source: String,
    pub path: PathBuf,
    pub grammar_config: Option<GrammarConfig>,
    pub root_items: Vec<NodeId>,
    /// All `ModuleRef` nodes (`import(...)` and `inherit(...)`) in source order,
    /// collected by the parser so the loader can iterate without scanning the
    /// arena. The inherit set (see [`ModuleContext::inherits`]) is derived from
    /// this, so cfg gating only has to prune this one structure.
    pub module_refs: Vec<NodeId>,
    /// `true` if the parser pushed at least one `Node::Cfg` for this module.
    /// Lets the loader skip `apply_cfg` entirely when no `#[cfg(...)]`
    /// attributes appear in source.
    pub has_cfg: bool,
    /// `true` if the parser pushed at least one `Node::Forward` for this module.
    pub has_forward_decls: bool,
    /// Flag names declared in this module's `flags`, mapped to the span of
    /// their first occurrence (used for `FirstDefinedHere` notes).
    pub cfg_declared: FxHashMap<String, Span>,
    /// Top-level declarations dropped by cfg in this module: name -> Cfg
    /// node id. Drives `GatedByDisabledCfg` enrichment.
    pub cfg_dropped: FxHashMap<String, NodeId>,
    /// Computed-name references (`@<expr>`) from rule-set macro instances,
    /// evaluated under each call's args at expand time, paired with their span.
    /// Resolve validates each against the rule-name table.
    pub computed_refs: Vec<Spanned<Str>>,
    /// Optional `let name: ty` annotations, keyed by the `Node::Let` id. Stored
    /// out-of-line (most lets have none) so the annotation doesn't widen `Node`.
    /// Written by the parser, read by typecheck.
    pub let_types: FxHashMap<NodeId, Ty>,
    /// Half-open `[start, end)` range of `NodeId`s this module owns in the
    /// shared arena. The parser pushes all of a module's nodes contiguously
    /// before any child loads, so this slice is well-defined and stable.
    /// Populated by `Parser::parse`; default `0..0` means "uninitialized".
    pub(crate) node_range: std::ops::Range<u32>,
}

impl ModuleContext {
    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(&self.source)
    }

    /// The `inherit()` `ModuleRef`s in source order, derived from `module_refs`. The first
    /// is the active base, a second means `MultipleInherits` (reported by `validate_grammar`).
    pub fn inherits<'a>(&'a self, arena: &'a NodeArena) -> impl Iterator<Item = NodeId> + 'a {
        self.module_refs
            .iter()
            .copied()
            .filter(move |&r| matches!(arena.get(r), Node::ModuleRef { import: false, .. }))
    }

    /// The resolved inherited-module index and its `inherit(...)` call span,
    /// once the loader has populated it. `None` for non-inheriting modules
    /// or before child loading completes.
    #[must_use]
    pub fn inherit_module(&self, arena: &NodeArena) -> Option<(ModuleId, Span)> {
        let id = self.inherits(arena).next()?;
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

    /// Build a [`Note`] anchored to this module's source.
    #[must_use]
    pub fn note(&self, message: NoteMessage, span: Span) -> Note {
        Note {
            message,
            span,
            path: self.path.clone(),
            src: self.source.clone(),
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
    // TODO: Don't skip language here, and instead filter out caller side?
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
