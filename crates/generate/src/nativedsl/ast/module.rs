//! Per-module parser output: [`ModuleContext`], the grammar block
//! ([`GrammarConfig`]/[`ConfigField`]), and their accessors.

use std::path::PathBuf;

use rustc_hash::FxHashMap;

use super::{Node, NodeArena, NodeId, Span, Spanned};
use crate::{
    nativedsl::{ModuleId, Note, NoteMessage, typecheck::Ty},
    strpool::StrId,
};

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
/// Owns the module's source text and module-specific state. Its nodes live in
/// the [`SharedAst`](super::SharedAst) shared by the entire grammar.
#[derive(Debug)]
pub struct ModuleContext {
    pub source: String,
    pub path: PathBuf,
    pub grammar_config: Option<GrammarConfig>,
    pub root_items: Vec<NodeId>,
    /// All `Import` and `Inherit` nodes in source order, collected by the parser.
    pub module_refs: Vec<NodeId>,
    /// `true` if the parser pushed at least one `Node::Cfg` for this module.
    pub has_cfg: bool,
    /// `true` if the parser pushed at least one `Node::Forward` for this module.
    pub has_forward_decls: bool,
    /// Flag names declared in this module's `flags`, mapped to the span of
    /// their first occurrence (used for `FirstDefinedHere` notes).
    pub cfg_declared: FxHashMap<StrId, Span>,
    /// Top-level declarations dropped by cfg in this module: name -> Cfg
    /// node id. Drives `GatedByDisabledCfg` enrichment.
    pub cfg_dropped: FxHashMap<StrId, NodeId>,
    /// Computed-name references (`@<expr>`) from rule-set macro instances,
    /// evaluated under each call's args at expand time, paired with their span.
    pub computed_refs: Vec<Spanned<StrId>>,
    /// Optional `let name: ty` annotations, keyed by the `Node::Let` id. Stored
    pub let_types: FxHashMap<NodeId, Ty>,
    /// Half-open range of nodes this module owns in the shared arena.
    node_range: std::ops::Range<NodeId>,
}

impl ModuleContext {
    pub(crate) fn new(
        source: String,
        path: PathBuf,
        root_capacity: usize,
        node_start: NodeId,
    ) -> Self {
        Self {
            source,
            path,
            grammar_config: None,
            root_items: Vec::with_capacity(root_capacity),
            module_refs: Vec::new(),
            has_cfg: false,
            has_forward_decls: false,
            cfg_declared: FxHashMap::default(),
            cfg_dropped: FxHashMap::default(),
            computed_refs: Vec::new(),
            let_types: FxHashMap::default(),
            node_range: node_start..node_start,
        }
    }

    #[must_use]
    pub fn text(&self, span: Span) -> &str {
        span.resolve(&self.source)
    }

    /// The [`Node::Inherit`] nodes in source order, derived from `module_refs`. The first
    /// is the active base, a second means `MultipleInherits` (reported by `validate_grammar`).
    pub fn inherits<'a>(&'a self, arena: &'a NodeArena) -> impl Iterator<Item = NodeId> + 'a {
        self.module_refs
            .iter()
            .copied()
            .filter(move |&r| matches!(arena.get(r), Node::Inherit { .. }))
    }

    /// The resolved inherited-module index and its `inherit(...)` call span,
    /// once the loader has populated it. `None` for non-inheriting modules
    /// or before child loading completes.
    #[must_use]
    pub fn inherit_module(&self, arena: &NodeArena) -> Option<(ModuleId, Span)> {
        let id = self.inherits(arena).next()?;
        let &Node::Inherit {
            module: Some(idx), ..
        } = arena.get(id)
        else {
            return None;
        };
        Some((idx, arena.span(id)))
    }

    /// Iterate just this module's own nodes in allocation order.
    ///
    /// `arena` must be the shared arena backing this context.
    ///
    /// # Panics
    ///
    /// The returned iterator panics if the recorded range exceeds `arena`.
    pub fn iter_own_nodes<'a>(
        &self,
        arena: &'a NodeArena,
    ) -> impl Iterator<Item = (NodeId, &'a Node)> {
        arena.iter_range(self.node_range.clone())
    }

    pub(crate) fn set_node_end(&mut self, end: NodeId) {
        debug_assert!(end.index() >= self.node_range.end.index());
        self.node_range.end = end;
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
    pub language: Option<StrId>,
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
    /// All node-valued config fields with their `ConfigField` kind.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_fields_covers_every_config_field() {
        // A ConfigField missing from node_fields silently skips cfg gating
        // and resolution; the exhaustive literal forces new fields through
        // this test.
        let id = NodeId::from_index(1);
        let config = GrammarConfig {
            language: Some(StrId::default()),
            inherits: Some(id),
            extras: Some(id),
            externals: Some(id),
            inline: Some(id),
            supertypes: Some(id),
            word: Some(id),
            conflicts: Some(id),
            precedences: Some(id),
            reserved: Some(id),
            start: Some(id),
            flags: Some(id),
        };
        // COUNT minus `language`, the one non-node (string-valued) field.
        assert_eq!(config.node_fields().count(), ConfigField::COUNT - 1);
    }
}
