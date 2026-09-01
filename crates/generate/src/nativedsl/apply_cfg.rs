//! Cfg attribute evaluation. Runs between parse and resolve. Drops disabled
//! `#[cfg(X)]` subtrees in place; unwraps active ones.
//!
//! Active flags are scoped to the current module path. A module's declarations
//! apply throughout its subtree, while flags from its parent take precedence.

use std::collections::hash_map::Entry;

use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::strpool::{StrId, StrPool};

use super::{
    CfgError, NoteMessage,
    ast::{
        ChildRange, ConfigField, GrammarConfig, ModuleContext, Node, NodeId, ObjectField,
        SharedAst, Span,
    },
    loader::ModuleKind,
};

pub type CfgResult<T> = Result<T, CfgError>;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum CfgErrorKind {
    #[error("`flags` must be an object literal `{{ enabled: [...], disabled: [...] }}`")]
    FlagsNotObject,
    #[error("`flags` only accepts `enabled` and `disabled` keys, got '{0}'")]
    FlagsUnknownKey(String),
    #[error("`flags.{{enabled,disabled}}` must be a list literal of string flag names")]
    FlagsNotList,
    #[error(
        "`flags.{{enabled,disabled}}` entries must be plain string literals (no raw strings or expressions)"
    )]
    FlagsNonLiteral,
    #[error(
        "`#[cfg(...)]` is not allowed inside the `flags` field. `flags` declarations are read before cfg gating"
    )]
    InsideFlags,
    #[error("`#[cfg({0})]` references an unknown flag")]
    FlagUnknown(String),
    #[error("flag '{0}' is declared more than once in this grammar's `flags`")]
    FlagDeclaredTwice(String),
}

/// Identity of an effective cfg environment.
#[derive(Clone, Copy, Default, PartialEq, Eq)]
pub(super) struct CfgEnvId(u32);

/// State captured before loading a child module.
#[derive(Clone, Copy)]
pub(super) struct CfgCheckpoint {
    inserted_len: usize,
    env_id: CfgEnvId,
}

/// Active flag values along the current module path.
#[derive(Default)]
pub struct CfgState {
    active: FxHashMap<StrId, bool>,
    inserted: Vec<StrId>,
    env_id: CfgEnvId,
    last_env_id: u32,
}

impl CfgState {
    /// Read this module's `flags: { enabled: [...], disabled: [...] }`,
    /// populate `ctx.cfg_declared`, and merge into the active environment.
    pub fn merge_module_flags(
        &mut self,
        shared: &SharedAst,
        ctx: &mut ModuleContext,
        strs: &StrPool,
    ) -> CfgResult<()> {
        let Some(flags_id) = ctx.grammar_config.as_ref().and_then(|c| c.flags) else {
            return Ok(());
        };
        let Node::Object(range) = *shared.arena.get(flags_id) else {
            return Err(err(
                CfgErrorKind::FlagsNotObject,
                shared.arena.span(flags_id),
            ));
        };
        let mut assigned_env = false;
        for &ObjectField {
            name: key,
            value: value_id,
        } in shared.pools.get_object(range)
        {
            let enable = match ctx.text(key.span) {
                "enabled" => true,
                "disabled" => false,
                other => {
                    return Err(err(CfgErrorKind::FlagsUnknownKey(other.into()), key.span));
                }
            };
            let Node::List(items) = shared.arena.get(value_id) else {
                return Err(err(CfgErrorKind::FlagsNotList, shared.arena.span(value_id)));
            };
            for &elem in shared.pools.child_slice(*items) {
                let span = shared.arena.span(elem);
                let name = match shared.arena.get(elem) {
                    Node::StringLit(sid) => *sid,
                    Node::Cfg { .. } => return Err(err(CfgErrorKind::InsideFlags, span)),
                    _ => return Err(err(CfgErrorKind::FlagsNonLiteral, span)),
                };
                let duplicate = match ctx.cfg_declared.entry(name) {
                    Entry::Occupied(entry) => Some(*entry.get()),
                    Entry::Vacant(entry) => {
                        entry.insert(span);
                        None
                    }
                };
                if let Some(first_span) = duplicate {
                    return Err(CfgError::with_note(
                        CfgErrorKind::FlagDeclaredTwice(strs.resolve(name).to_string()),
                        span,
                        ctx.note(NoteMessage::FirstDefinedHere, first_span),
                    ));
                }
                if let Entry::Vacant(entry) = self.active.entry(name) {
                    entry.insert(enable);
                    self.inserted.push(name);

                    if !std::mem::replace(&mut assigned_env, true) {
                        self.last_env_id += 1;
                        self.env_id = CfgEnvId(self.last_env_id);
                    }
                }
            }
        }
        Ok(())
    }

    /// Iterate over the effective flag names and their enabled state.
    pub fn flags(&self) -> impl Iterator<Item = (StrId, bool)> + '_ {
        self.active.iter().map(|(&name, &enabled)| (name, enabled))
    }

    pub(super) const fn env_id(&self) -> CfgEnvId {
        self.env_id
    }

    pub(super) const fn checkpoint(&self) -> CfgCheckpoint {
        CfgCheckpoint {
            inserted_len: self.inserted.len(),
            env_id: self.env_id,
        }
    }

    pub(super) fn restore(&mut self, checkpoint: CfgCheckpoint) {
        for name in self.inserted.drain(checkpoint.inserted_len..) {
            let removed = self.active.remove(&name);
            debug_assert!(removed.is_some());
        }
        self.env_id = checkpoint.env_id;
    }
}

/// Walk this module's AST: drop disabled cfg subtrees, unwrap active ones.
pub(super) fn apply_cfg(
    shared: &mut SharedAst,
    ctx: &mut ModuleContext,
    strs: &StrPool,
    state: &CfgState,
    kind: ModuleKind,
) -> CfgResult<()> {
    ctx.module_refs.clear(); // rebuilt during cfg walk
    let mut w = Walker {
        shared: &mut *shared,
        state,
        kind,
        strs,
        cfg_declared: &ctx.cfg_declared,
        cfg_dropped: &mut ctx.cfg_dropped,
        module_refs: &mut ctx.module_refs,
        sym_ref_cursor: None,
    };
    for (_, id) in ctx
        .grammar_config
        .as_ref()
        .into_iter()
        .flat_map(GrammarConfig::node_fields)
        // `flags` already consumed by `merge_module_flags`
        .filter(|(field, _)| !matches!(field, ConfigField::Flags))
    {
        w.walk(id)?;
    }
    // Compact root_items, repointing each surviving entry at the id the walk
    // surfaced.
    let mut has_forward_decls = false;
    let original_len = ctx.root_items.len();
    let mut write = 0;
    for read in 0..original_len {
        let id = ctx.root_items[read];
        if let Some(kept) = w.walk(id)? {
            has_forward_decls |= matches!(w.shared.arena.get(kept), Node::Forward { .. });
            ctx.root_items[write] = kept;
            write += 1;
        }
    }
    ctx.root_items.truncate(write);
    ctx.has_forward_decls = has_forward_decls;
    Ok(())
}

struct Walker<'a> {
    shared: &'a mut SharedAst,
    state: &'a CfgState,
    kind: ModuleKind,
    /// Resolves cfg names in [`Self::walk_cfg`].
    strs: &'a StrPool,
    /// Local declared set for the grammar visibility check
    cfg_declared: &'a FxHashMap<StrId, Span>,
    /// cfg-dropped top-level decls for `enrich_resolve_error` to use later.
    cfg_dropped: &'a mut FxHashMap<StrId, NodeId>,
    /// The module's import/inherit refs, rebuilt from the surviving AST as the
    /// walk visits each one - a nested cfg-dropped ref is never collected.
    module_refs: &'a mut Vec<NodeId>,
    /// Write head into the active rule-set macro's `sym_refs` range while its
    /// body is walked. `None` outside a macro body. Surviving `@<expr>` refs are
    /// compacted to the front of the range in place.
    sym_ref_cursor: Option<u32>,
}

impl Walker<'_> {
    /// `Ok(Some(id))` = keep `id` (an active cfg resolves to its unwrapped child's id)
    /// `Ok(None)` = drop. Shrinks filtered list ranges in place.
    fn walk(&mut self, id: NodeId) -> CfgResult<Option<NodeId>> {
        if let &Node::Cfg {
            name,
            name_offset,
            child,
        } = self.shared.arena.get(id)
        {
            return self.walk_cfg(id, name, name_offset, child);
        }
        self.walk_children(id)?;
        Ok(Some(id))
    }

    fn walk_cfg(
        &mut self,
        id: NodeId,
        name: StrId,
        name_offset: u32,
        child: NodeId,
    ) -> CfgResult<Option<NodeId>> {
        // Grammar modules require a local declaration. Helper modules transparently
        // see the importing grammar's declared set.
        let active = match self.kind {
            ModuleKind::Grammar => self
                .cfg_declared
                .contains_key(&name)
                .then(|| self.state.active.get(&name).copied().unwrap_or(false)),
            ModuleKind::Helper => self.state.active.get(&name).copied(),
        };

        let Some(active) = active else {
            let name_len = self.strs.resolve(name).len() as u32;
            let name_span = Span::new(name_offset, name_offset + name_len);
            return Err(err(
                CfgErrorKind::FlagUnknown(self.strs.resolve(name).into()),
                name_span,
            ));
        };

        if !active {
            // Peel any nested cfg layers to find the underlying item.
            let mut item = child;
            while let &Node::Cfg { child: inner, .. } = self.shared.arena.get(item) {
                item = inner;
            }
            // Record a dropped named top-level decl for `enrich_resolve_error`.
            let dropped_name = match self.shared.arena.get(item) {
                Node::Rule { name, .. } | Node::Let { name, .. } | Node::Forward { name } => {
                    Some(*name)
                }
                Node::Macro(macro_id) => Some(self.shared.pools.get_macro(*macro_id).name.value),
                _ => None,
            };
            if let Some(decl_name) = dropped_name {
                self.cfg_dropped.entry(decl_name).or_insert(id);
            }
            Ok(None)
        } else {
            // Active: surface the unwrapped child's id so the parent references it directly.
            self.walk(child)
        }
    }

    fn walk_children(&mut self, id: NodeId) -> CfgResult<()> {
        let node = *self.shared.arena.get(id);
        match node {
            // List-shaped variants whose members can be cfg-gated. Filter dropped
            // members in place, shrink the node's range on change.
            #[rustfmt::skip]
            Node::SeqOrChoice { range, .. } | Node::List(range)
            | Node::Concat(range) | Node::RuleSet(range) => {
                self.filter(id, range)?;
            }
            // Fixed-arity / positional members can't be cfg-gated, but recurse to
            // reach any cfg nested deeper inside each.
            Node::Tuple(r) => {
                for i in r.as_range() {
                    let c = self.shared.pools.children[i];
                    self.walk(c)?;
                }
            }
            Node::Object(range) => {
                for i in range.as_range() {
                    let v = self.shared.pools.object_fields[i].value;
                    self.walk(v)?;
                }
            }
            Node::Call { name, args } => {
                self.walk(name)?;
                for i in args.as_range() {
                    let c = self.shared.pools.children[i];
                    self.walk(c)?;
                }
            }
            #[rustfmt::skip]
            Node::Rule { body: c, .. } | Node::Let { value: c, .. } | Node::Repeat { inner: c, .. }
            | Node::Token { inner: c, .. } | Node::Field { content: c, .. }
            | Node::Reserved { content: c, .. } | Node::FieldAccess { obj: c, .. }  | Node::Neg(c)
            | Node::QualifiedAccess { obj: c, .. } | Node::GrammarConfig { module: c, .. }
            | Node::For { body: c, .. } => {
                self.walk(c)?;
            }
            // SymRef (`@<expr>` in a rule-set body): compact it to the front of
            // the enclosing macro's sym_refs range in place, then walk the expr.
            Node::SymRef { expr } => {
                if let Some(w) = self.sym_ref_cursor.as_mut() {
                    self.shared.pools.children[*w as usize] = id;
                    *w += 1;
                }
                self.walk(expr)?;
            }
            #[rustfmt::skip]
            Node::Alias { content: a, target: b } | Node::Append { left: a, right: b }
            | Node::BinOp { lhs: a, rhs: b, .. } | Node::Prec { value: a, content: b, .. }
            | Node::ComputedRule { name_expr: a, body: b, .. } => {
                self.walk(a)?;
                self.walk(b)?;
            }
            Node::DynRegex { pattern, flags } => {
                for c in std::iter::once(pattern).chain(flags) {
                    self.walk(c)?;
                }
            }
            // Macro: prune its sym_refs to the cfg-surviving subset by compacting
            // the range in place as the body is walked, then shrink its length to
            // the survivors.
            Node::Macro(macro_id) => {
                let sym_refs = self.shared.pools.get_macro(macro_id).sym_refs;
                let body = self.shared.pools.get_macro(macro_id).body;
                self.sym_ref_cursor = Some(sym_refs.start);
                self.walk(body)?;
                let kept = (self.sym_ref_cursor.take().unwrap() - sym_refs.start) as u16;
                self.shared.pools.get_macro_mut(macro_id).sym_refs.len = kept;
            }
            // Import/inherit ref: collected so resolve and lower see the
            // surviving set.
            Node::Import { .. } | Node::Inherit { .. } => self.module_refs.push(id),
            // Leaves: nothing to descend into.
            #[rustfmt::skip]
            Node::Grammar | Node::Forward { .. } | Node::StringLit(_) | Node::IntLit(_)
            | Node::Ident(_) | Node::Blank | Node::Eof | Node::MacroParam { .. }
            | Node::ForBinding { .. } => {}
            // `Cfg` is intercepted by walk(), `ExpandedRule`/`ModuleRule` are emitted by
            // expand/resolve (after apply_cfg), `Unreachable` is the arena sentinel.
            // None can legitimately reach here.
            #[rustfmt::skip]
            Node::Cfg { .. } | Node::ExpandedRule(_) | Node::ModuleRule { .. }
            | Node::Unreachable => unreachable!(),
        }
        Ok(())
    }

    /// Filter a child range in place and shrink `id`'s range to match. Walk each
    /// `NodeId` slot, then overwrite earlier slots with survivors. Slots past the
    /// new end are orphaned but no `ChildRange` references them.
    fn filter(&mut self, id: NodeId, range: ChildRange) -> CfgResult<()> {
        let start = range.start as usize;
        let mut write = start;
        for read in start..start + range.len as usize {
            let c = self.shared.pools.children[read];
            if let Some(kept) = self.walk(c)? {
                self.shared.pools.children[write] = kept;
                write += 1;
            }
        }
        let new_len = (write - start) as u16;
        if new_len < range.len {
            // Safe: caller passes a node that has a child range.
            self.shared.arena.get_mut(id).child_range_mut().unwrap().len = new_len;
        }
        Ok(())
    }
}

const fn err(kind: CfgErrorKind, span: Span) -> CfgError {
    CfgError::new(kind, span)
}
