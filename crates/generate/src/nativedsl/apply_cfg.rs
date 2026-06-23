//! Cfg attribute evaluation. Runs between parse and resolve. Drops disabled
//! `#[cfg(X)]` subtrees in place; unwraps active ones.
//!
//! Active flag set is built in load order with first-write-wins. Each module
//! merges its own flags before loading the modules it imports or inherits, so
//! an importing module overrides the flag values of the modules it pulls in.

use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    Diagnostic, NoteMessage, ResolveError, ResolveErrorKind,
    ast::{
        ChildRange, ConfigField, GrammarConfig, ModuleContext, Node, NodeId, ObjectField, SharedAst,
        Span,
    },
    loader::ModuleKind,
};

#[derive(Default)]
pub struct CfgState {
    /// Union of every module's `cfg_declared`: every flag name declared
    /// anywhere in the load chain. Cached for the helper-module visibility
    /// check (which sees the global declared set, not per-module).
    pub declared_any: FxHashSet<String>,
    /// `name -> enabled?`. First write wins via `or_insert`; a module merges
    /// before loading its imports, so importers override imported modules.
    pub active: FxHashMap<String, bool>,
}

impl CfgState {
    /// Read this module's `flags: { enabled: [...], disabled: [...] }`,
    /// populate `ctx.cfg_declared`, and merge into global state.
    pub fn merge_module_flags(
        &mut self,
        shared: &SharedAst,
        ctx: &mut ModuleContext,
    ) -> Result<(), ResolveError> {
        let Some(flags_id) = ctx.grammar_config.as_ref().and_then(|c| c.flags) else {
            return Ok(());
        };
        let Node::Object(range) = *shared.arena.get(flags_id) else {
            return Err(err(
                ResolveErrorKind::CfgFlagsNotObject,
                shared.arena.span(flags_id),
            ));
        };
        for &ObjectField {
            name: key_span,
            value: value_id,
        } in shared.pools.get_object(range)
        {
            let enable = match ctx.text(key_span) {
                "enabled" => true,
                "disabled" => false,
                other => {
                    return Err(err(
                        ResolveErrorKind::CfgFlagsUnknownKey(other.into()),
                        key_span,
                    ));
                }
            };
            let Node::List(items) = shared.arena.get(value_id) else {
                return Err(err(
                    ResolveErrorKind::CfgFlagsNotList,
                    shared.arena.span(value_id),
                ));
            };
            for &elem in shared.pools.child_slice(*items) {
                let span = shared.arena.span(elem);
                match shared.arena.get(elem) {
                    Node::StringLit => {}
                    Node::Cfg { .. } => {
                        return Err(err(ResolveErrorKind::CfgInsideFlags, span));
                    }
                    _ => return Err(err(ResolveErrorKind::CfgFlagsNonLiteral, span)),
                }
                let name_text = ctx.text(span);
                if let Some(&first_span) = ctx.cfg_declared.get(name_text) {
                    return Err(ResolveError::with_note(
                        ResolveErrorKind::CfgFlagDeclaredTwice(name_text.to_owned()),
                        span,
                        ctx.note(NoteMessage::FirstDefinedHere, first_span),
                    ));
                }
                let name = name_text.to_owned();
                self.declared_any.insert(name.clone());
                ctx.cfg_declared.insert(name.clone(), span);
                self.active.entry(name).or_insert(enable);
            }
        }
        Ok(())
    }
}

/// Walk this module's AST: drop disabled cfg subtrees, unwrap active ones.
pub fn apply_cfg(
    shared: &mut SharedAst,
    ctx: &mut ModuleContext,
    state: &CfgState,
    kind: ModuleKind,
) -> Result<(), ResolveError> {
    // The cfg walk rebuilds module_refs from the surviving AST: it visits every
    // surviving node, so a nested cfg-dropped import / inherit is never
    // collected. Cleared here, repopulated by the walk; the no-cfg path keeps
    // the parser's population since apply_cfg only runs when has_cfg. (sym_refs
    // are pruned per-macro below; top-level-only tables are derived where used.)
    ctx.module_refs.clear();
    let mut w = Walker {
        shared: &mut *shared,
        state,
        kind,
        source: &ctx.source,
        cfg_declared: &ctx.cfg_declared,
        cfg_dropped: &mut ctx.cfg_dropped,
        module_refs: &mut ctx.module_refs,
        sym_ref_cursor: None,
    };
    // Walk grammar config fields first - cfg attrs may appear on members of
    // `conflicts`, `precedences`, `externals`, etc. Skip `flags:` already
    // consumed by `merge_module_flags`.
    for (_, id) in ctx
        .grammar_config
        .as_ref()
        .into_iter()
        .flat_map(GrammarConfig::node_fields)
        .filter(|(field, _)| !matches!(field, ConfigField::Flags))
    {
        w.walk(id)?;
    }
    // Compact root_items, repointing each surviving entry at the id the walk
    // surfaced (an active cfg resolves to its unwrapped child's id).
    let original_len = ctx.root_items.len();
    let mut write = 0;
    for read in 0..original_len {
        let id = ctx.root_items[read];
        if let Some(kept) = w.walk(id)? {
            ctx.root_items[write] = kept;
            write += 1;
        }
    }
    ctx.root_items.truncate(write);
    Ok(())
}

struct Walker<'a> {
    shared: &'a mut SharedAst,
    state: &'a CfgState,
    kind: ModuleKind,
    source: &'a str,
    /// Disjoint borrows of `ctx`. `cfg_declared` is the local declared set for
    /// the grammar visibility check; `cfg_dropped` records cfg-dropped top-level
    /// decls for `enrich_resolve_error` to use later.
    cfg_declared: &'a FxHashMap<String, Span>,
    cfg_dropped: &'a mut FxHashMap<String, NodeId>,
    /// The module's import/inherit refs, rebuilt from the surviving AST as the
    /// walk visits each one - a nested cfg-dropped ref is never collected.
    module_refs: &'a mut Vec<NodeId>,
    /// Write head into the active rule-set macro's `sym_refs` range while its
    /// body is walked; `None` outside a macro body. Surviving `@<expr>` refs are
    /// compacted to the front of the range in place (cfg only ever drops refs),
    /// mirroring [`Self::filter`].
    sym_ref_cursor: Option<usize>,
}

impl Walker<'_> {
    /// `Ok(Some(id))` = keep and reference `id` (an active cfg resolves to its
    /// unwrapped child's id); `Ok(None)` = drop. Shrinks filtered list ranges in
    /// place but never relocates a node.
    fn walk(&mut self, id: NodeId) -> Result<Option<NodeId>, ResolveError> {
        if let &Node::Cfg { name, child } = self.shared.arena.get(id) {
            return self.walk_cfg(id, name, child);
        }
        self.walk_children(id)?;
        Ok(Some(id))
    }

    fn walk_cfg(
        &mut self,
        id: NodeId,
        name: Span,
        child: NodeId,
    ) -> Result<Option<NodeId>, ResolveError> {
        let name_text: &str = name.resolve(self.source);
        // Grammar modules require local declaration. Helper modules transparently
        // see the importing grammar's declared set.
        let visible = match self.kind {
            ModuleKind::Grammar => self.cfg_declared.contains_key(name_text),
            ModuleKind::Helper => self.state.declared_any.contains(name_text),
        };
        if !visible {
            return Err(err(
                ResolveErrorKind::CfgFlagUnknown(name_text.into()),
                name,
            ));
        }
        let active = self.state.active.get(name_text).copied().unwrap_or(false);
        if !active {
            // Peel any nested cfg layers to find the underlying item.
            let mut item = child;
            while let &Node::Cfg { child: inner, .. } = self.shared.arena.get(item) {
                item = inner;
            }
            // Record a dropped named top-level decl for `enrich_resolve_error`.
            // The macro/module working sets are rebuilt from the surviving
            // root_items after the walk, so a dropped decl needs no fixup here.
            let dropped_name = match self.shared.arena.get(item) {
                Node::Rule { name, .. } | Node::Let { name, .. } | Node::External { name } => {
                    Some(*name)
                }
                Node::Macro(macro_id) => Some(self.shared.pools.get_macro(*macro_id).name),
                _ => None,
            };
            if let Some(decl_name) = dropped_name {
                let key = decl_name.resolve(self.source).to_owned();
                self.cfg_dropped.entry(key).or_insert(id);
            }
            return Ok(None);
        }
        // Active: surface the unwrapped child's id (it already carries the
        // child's span) so the parent references it directly. No relocation
        // means no node-id-keyed side table needs re-keying.
        self.walk(child)
    }

    fn walk_children(&mut self, id: NodeId) -> Result<(), ResolveError> {
        let node = *self.shared.arena.get(id);
        match node {
            // List-shaped variants where cfg members are allowed (per the
            // parser's `parse_expr_with_cfg` routing): filter dropped members
            // in place, shrink the node's range if any change.
            Node::SeqOrChoice { range, .. } | Node::List(range) | Node::Concat(range) => {
                self.filter(id, range)?;
            }
            // Variadic but cfg not allowed in members: just recurse. Index
            // into the backing pool directly so we don't hold a borrow across
            // the recursive `walk` call (which needs `&mut self.shared`).
            // Recursive walks only ever append to the pools, never touching
            // existing slots, so indices in our range stay valid.
            Node::Tuple(r) | Node::QualifiedCall(r) | Node::RuleSet(r) => {
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
            // Single-child wrappers.
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
                    self.shared.pools.children[*w] = id;
                    *w += 1;
                }
                self.walk(expr)?;
            }
            // Two-child wrappers.
            Node::Alias {
                content: a,
                target: b,
            }
            | Node::Append { left: a, right: b }
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
            // the survivors. The range sits past every body pool entry, so the
            // body walk never touches its slots.
            Node::Macro(macro_id) => {
                let sym_refs = self.shared.pools.get_macro(macro_id).sym_refs;
                let body = self.shared.pools.get_macro(macro_id).body;
                self.sym_ref_cursor = Some(sym_refs.start as usize);
                self.walk(body)?;
                let kept = (self.sym_ref_cursor.take().unwrap() - sym_refs.start as usize) as u16;
                self.shared.pools.get_macro_mut(macro_id).sym_refs.len = kept;
            }
            // Import/inherit ref: collected so resolve and lower see the
            // surviving set; no children to descend into.
            Node::ModuleRef { .. } => self.module_refs.push(id),
            // Leaves / nothing to descend into.
            #[rustfmt::skip]
            Node::Grammar | Node::External { .. } | Node::StringLit | Node::RawStringLit { .. }
            | Node::IntLit(_) | Node::Ident(_) | Node::Blank
            | Node::MacroParam { .. } | Node::ForBinding { .. } | Node::Unreachable
            // handled by walk(), unreachable here
            | Node::Cfg { .. } => {}
            // Emitted by expand_macro_calls / resolve, both of which run after
            // apply_cfg.
            Node::ExpandedRule(_) | Node::ModuleRule { .. } => {
                unreachable!()
            }
        }
        Ok(())
    }

    /// Filter a child range in place and shrink `id`'s range to match.
    /// `NodeIds` are `Copy` so we can read each slot, walk it (recursive
    /// walks only ever *append* to `pools.children`, never touch our slots),
    /// then overwrite earlier slots with survivors. Slots past the new end
    /// are orphaned but no `ChildRange` references them.
    fn filter(&mut self, id: NodeId, range: ChildRange) -> Result<(), ResolveError> {
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

const fn err(kind: ResolveErrorKind, span: Span) -> ResolveError {
    Diagnostic::new(kind, span)
}
