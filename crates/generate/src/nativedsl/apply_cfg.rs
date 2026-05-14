//! Cfg attribute evaluation. Runs between parse and resolve. Drops disabled
//! `#[cfg(X)]` subtrees in place; unwraps active ones.
//!
//! Active flag set is built in load order with first-write-wins, so the
//! deepest descendant (parses first) overrides ancestors.

use rustc_hash::{FxHashMap, FxHashSet};

use super::{
    Diagnostic, ResolveError, ResolveErrorKind,
    ast::{ChildRange, ConfigField, GrammarConfig, ModuleContext, Node, NodeId, SharedAst, Span},
    loader::ModuleKind,
};

#[derive(Default)]
pub struct CfgState {
    /// All flag names declared anywhere in the chain.
    pub declared: FxHashSet<String>,
    /// `name -> enabled?`. Descendants (loaded earlier) win via `or_insert`.
    pub active: FxHashMap<String, bool>,
    /// Per-module local declarations. Cfg attrs in M may only name flags in
    /// `declared_per_module[M]`; cross-module refs must re-declare.
    pub declared_per_module: Vec<FxHashSet<String>>,
    /// `decl_name` -> node id for a top-level declaration (rule, macro, let,
    /// external) dropped by cfg. Used by resolve to turn "undefined symbol"
    /// into "definition exists but gated by ...".
    pub dropped_decls: FxHashMap<String, NodeId>,
}

impl CfgState {
    /// Read this module's `flags: { enabled: [...], disabled: [...] }` and
    /// merge into the global state. Returns the assigned per-module index.
    pub fn merge_module_flags(
        &mut self,
        shared: &SharedAst,
        ctx: &ModuleContext,
    ) -> Result<usize, ResolveError> {
        let mut local = FxHashSet::default();
        if let Some(flags_id) = ctx.grammar_config.as_ref().and_then(|c| c.flags) {
            let Node::Object(range) = *shared.arena.get(flags_id) else {
                return Err(err(
                    ResolveErrorKind::CfgFlagsNotObject,
                    shared.arena.span(flags_id),
                ));
            };
            for &(key_span, value_id) in shared.pools.get_object(range) {
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
                            return Err(err(ResolveErrorKind::CfgHasCfg, span));
                        }
                        _ => return Err(err(ResolveErrorKind::CfgFlagsNonLiteral, span)),
                    }
                    let name = ctx.text(span).to_owned();
                    self.declared.insert(name.clone());
                    local.insert(name.clone());
                    self.active.entry(name).or_insert(enable);
                }
            }
        }
        self.declared_per_module.push(local);
        Ok(self.declared_per_module.len() - 1)
    }
}

/// Walk this module's AST: drop disabled cfg subtrees, unwrap active ones.
pub fn apply_cfg(
    shared: &mut SharedAst,
    ctx: &mut ModuleContext,
    state: &mut CfgState,
    module_idx: usize,
    kind: ModuleKind,
) -> Result<(), ResolveError> {
    let mut w = Walker {
        shared,
        state,
        kind,
        module_idx,
        source: &ctx.source,
        module_refs: &mut ctx.module_refs,
        inherit_ref: &mut ctx.inherit_ref,
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
    // In-place compact root_items
    let mut write = 0;
    for read in 0..ctx.root_items.len() {
        let id = ctx.root_items[read];
        if w.walk(id)? {
            ctx.root_items[write] = id;
            write += 1;
        }
    }
    ctx.root_items.truncate(write);
    Ok(())
}

struct Walker<'a> {
    shared: &'a mut SharedAst,
    state: &'a mut CfgState,
    kind: ModuleKind,
    /// Index into `state.declared_per_module`.
    module_idx: usize,
    source: &'a str,
    /// Disjoint borrows of `ctx`. When we drop a cfg-gated `let h = import(...)`
    /// we also pull its `ModuleRef` id out of these so the loader doesn't
    /// load the file.
    module_refs: &'a mut Vec<NodeId>,
    inherit_ref: &'a mut Option<NodeId>,
}

impl Walker<'_> {
    /// `Ok(true)` = keep, `Ok(false)` = drop from `root_items`. Mutates the arena to unwrap
    /// active Cfg nodes and to rebuild filtered list ranges.
    fn walk(&mut self, id: NodeId) -> Result<bool, ResolveError> {
        if let &Node::Cfg { name, child } = self.shared.arena.get(id) {
            return self.walk_cfg(id, name, child);
        }
        self.walk_children(id)?;
        Ok(true)
    }

    fn walk_cfg(&mut self, id: NodeId, name: Span, child: NodeId) -> Result<bool, ResolveError> {
        let name_text: &str = name.resolve(self.source);
        // Grammar modules require local declaration. Helper modules transparently
        // see the importing grammar's declared set.
        let visible = match self.kind {
            ModuleKind::Grammar => {
                self.state.declared_per_module[self.module_idx].contains(name_text)
            }
            ModuleKind::Helper => self.state.declared.contains(name_text),
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
            // If we're dropping a named top-level decl, record for diagnostic.
            let dropped_name = match self.shared.arena.get(item) {
                Node::Rule { name, .. } | Node::Let { name, .. } | Node::External { name } => {
                    Some(*name)
                }
                Node::Macro(macro_id) => Some(self.shared.pools.get_macro(*macro_id).name),
                _ => None,
            };
            if let Some(decl_name) = dropped_name {
                let key = decl_name.resolve(self.source).to_owned();
                self.state.dropped_decls.entry(key).or_insert(id);
            }
            // `let h = import/inherit(...)` is the only shape that puts a
            // `ModuleRef` at item level. Pull it out of the loader's worklist.
            if let &Node::Let { value, .. } = self.shared.arena.get(item)
                && matches!(self.shared.arena.get(value), Node::ModuleRef { .. })
            {
                self.module_refs.retain(|x| *x != value);
                if *self.inherit_ref == Some(value) {
                    *self.inherit_ref = None;
                }
            }
            return Ok(false);
        }
        let kept = self.walk(child)?;
        if !kept {
            return Ok(false);
        }
        // Active and kept: unwrap by overwriting our slot with the child's
        // data AND span (so downstream phases see the child's source range,
        // not the cfg-attribute syntax around it).
        let child_node = *self.shared.arena.get(child);
        let child_span = self.shared.arena.span(child);
        self.shared.arena.set(id, child_node);
        self.shared.arena.set_span(id, child_span);
        Ok(true)
    }

    fn walk_children(&mut self, id: NodeId) -> Result<(), ResolveError> {
        let node = *self.shared.arena.get(id);
        match node {
            // List-shaped variants where cfg members are allowed: filter
            // dropped members in place, shrink the node's range if any change.
            Node::SeqOrChoice { range, .. } | Node::List(range) => {
                self.filter(id, range)?;
            }
            // Variadic but cfg not allowed in members: just recurse. Index
            // into the backing pool directly so we don't hold a borrow across
            // the recursive `walk` call (which needs `&mut self.shared`).
            // Recursive walks only ever append to the pools, never touching
            // existing slots, so indices in our range stay valid.
            Node::Tuple(r) | Node::Concat(r) | Node::QualifiedCall(r) => {
                for i in r.start as usize..r.start as usize + r.len as usize {
                    let c = self.shared.pools.children[i];
                    self.walk(c)?;
                }
            }
            Node::Object(range) => {
                for i in range.start as usize..range.start as usize + range.len as usize {
                    let v = self.shared.pools.object_fields[i].1;
                    self.walk(v)?;
                }
            }
            Node::Call { name, args } => {
                self.walk(name)?;
                for i in args.start as usize..args.start as usize + args.len as usize {
                    let c = self.shared.pools.children[i];
                    self.walk(c)?;
                }
            }
            // Single-child wrappers.
            #[rustfmt::skip]
            Node::Rule { body: c, .. } | Node::Let { value: c, .. } | Node::Repeat { inner: c, .. }
            | Node::Token { inner: c, .. } | Node::For { body: c, .. }  | Node::Field { content: c, .. }
            | Node::Reserved { content: c, .. } | Node::FieldAccess { obj: c, .. }  | Node::Neg(c)
            | Node::QualifiedAccess { obj: c, .. } | Node::GrammarConfig { module: c, .. } => {
                self.walk(c)?;
            }
            // Two-child wrappers.
            Node::Alias {
                content: a,
                target: b,
            }
            | Node::Append { left: a, right: b }
            | Node::Prec {
                value: a,
                content: b,
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
            // Macro: body lives in pools.get_macro(id).body, not in arena
            // children. Walk it explicitly.
            Node::Macro(macro_id) => _ = self.walk(self.shared.pools.get_macro(macro_id).body)?,
            // Leaves / nothing to descend into.
            #[rustfmt::skip]
            Node::Grammar | Node::External { .. } | Node::StringLit | Node::RawStringLit { .. }
            | Node::IntLit(_) | Node::Ident(_) | Node::Blank | Node::ModuleRef { .. }
            | Node::MacroParam { .. } | Node::ForBinding { .. } | Node::Unreachable
            // handled by walk(), unreachable here
            | Node::Cfg { .. } => {}
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
            if self.walk(c)? {
                self.shared.pools.children[write] = c;
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
