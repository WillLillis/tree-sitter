//! Module system and glue code for the native DSL.
//!
//! Dispatches the lexer, parser, and lowerer throughout the module system.

use std::path::{Path, PathBuf};

use crate::nativedsl::{
    DisallowedItemKind, DslError, DslResult, LexError, LexErrorKind, LowerError, LowerErrorKind,
    LoweringState, Module, ModuleError, ModuleId, NoteMessage, ResolveError,
    apply_cfg::{CfgState, apply_cfg},
    ast::{ModuleContext, Node, SharedAst, Span},
    expand_macro_calls, lexer, lower, parser, resolve,
    resolve::ResolveErrorKind,
    string_pool::StringPool,
    typecheck::{self, TypeEnv},
};

const MAX_MODULE_DEPTH: usize = 256;

/// Mutable pipeline state passed through the load/lower recursion.
pub struct Loader<'a> {
    pub shared: &'a mut SharedAst,
    pub modules: &'a mut Vec<Module>,
    pub env: &'a mut TypeEnv,
    pub state: &'a mut LoweringState,
    pub strings: &'a mut StringPool,
    pub cfg: &'a mut CfgState,
    pub ancestor_paths: Vec<PathBuf>,
    /// Loader-wide dedup cache: each (canonical path, kind) loads at most once
    /// across the whole import graph for a single `parse_native_dsl` call. Keyed
    /// by kind so that the unusual case of the same file being both inherited
    /// and imported still loads twice (once per kind), letting validation fail
    /// the inappropriate one.
    pub loaded: Vec<(PathBuf, ModuleKind, ModuleId)>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ModuleKind {
    /// Grammar file (root or inherited). Must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported). Anything except a grammar block or override rule.
    Helper,
}

impl Loader<'_> {
    /// Load a tsg module, imported either via `import` or `inherit`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the module isn't valid tsg source.
    ///
    /// # Panics
    ///
    /// Panics if `path` has no parent directory.
    pub fn load_module(
        &mut self,
        source: &str,
        path: &Path,
        kind: ModuleKind,
    ) -> DslResult<ModuleId> {
        if source.len() >= u32::MAX as usize {
            Err(LexError::without_span(LexErrorKind::InputTooLarge))?;
        }

        let module_dir = path.parent().unwrap();

        let tokens = lexer::Lexer::new(source).tokenize()?;
        let mut ctx =
            parser::Parser::new(&tokens, source.to_string(), path.to_path_buf(), self.shared)
                .parse()?;

        // Register this module's flag declarations into the global state.
        // First-write-wins, so descendants (loaded earlier) override ancestors.
        self.cfg.merge_module_flags(self.shared, &mut ctx)?;

        // Apply cfg gating *before* loading children so cfg-disabled
        // `inherit(...)` / `import(...)` calls don't trigger file loads (and
        // don't merge their rules into this module). Also runs before module
        // validation so the inherit/inherits consistency check sees the
        // post-gating state. Skip the whole walk when no `#[cfg(...)]`
        // appears anywhere in this module (the common case for grammars that
        // don't use the feature).
        if ctx.has_cfg {
            apply_cfg(self.shared, &mut ctx, self.cfg, kind)?;
        }

        match kind {
            ModuleKind::Grammar => self.validate_grammar(&ctx)?,
            ModuleKind::Helper => self.validate_import_items(&ctx)?,
        }

        // Inline top-level rule-set macro invocations into ExpandedRule
        // decls. Runs before child loads so all nodes this module owns
        // sit in one contiguous arena range; runs before resolve so the
        // name table sees the post-expansion shape of root_items.
        expand_macro_calls::expand_macro_calls(self.shared, self.strings, &mut ctx)?;
        ctx.node_range.end = self.shared.arena.next_id().into();

        // Load inherited grammar (Grammar kind only, must happen before typecheck).
        // The active base is the first inherit() in source order (a second is
        // rejected by validate_grammar above).
        let inherit_id = ctx.inherits(&self.shared.arena).next();
        if let Some(inherit_id) = inherit_id
            && let Node::ModuleRef {
                import: false,
                path: ref_path,
                ..
            } = *self.shared.arena.get(inherit_id)
        {
            let child_path = resolve_path(&module_dir.join(ctx.text(ref_path)), ref_path)?;
            let child_id = self.load_child_module(&child_path, ref_path, ModuleKind::Grammar)?;
            self.shared.arena.set(
                inherit_id,
                Node::ModuleRef {
                    import: false,
                    path: ref_path,
                    module: Some(child_id),
                },
            );
        }

        self.load_import_children(&ctx, module_dir)?;

        // Resolve identifiers + typecheck. Child modules already populated env
        // during their own load_module calls.
        let base = ctx
            .inherit_module(&self.shared.arena)
            .and_then(|(idx, span)| Some((self.modules[idx as usize].lowered()?, span)));
        resolve::resolve(self.shared, &ctx, self.strings, self.modules, base)
            .map_err(|e| self.enrich_resolve_error(&ctx, e))?;

        typecheck::check(self.shared, &ctx, self.env)?;
        let module = match kind {
            ModuleKind::Grammar => {
                let lowered = Box::new(lower::lower_with_base(
                    self.state,
                    self.strings,
                    self.shared,
                    self.modules,
                    &ctx,
                )?);
                let exports = super::build_exports(
                    &self.shared.arena,
                    &self.shared.pools,
                    &ctx,
                    super::LoweredRef::Grammar(&lowered),
                );
                Module::Grammar {
                    ctx,
                    lowered,
                    exports,
                }
            }
            ModuleKind::Helper => {
                let lowered_rules =
                    lower::lower_helper(self.state, self.strings, self.shared, self.modules, &ctx)?;
                let exports = super::build_exports(
                    &self.shared.arena,
                    &self.shared.pools,
                    &ctx,
                    super::LoweredRef::Helper(&lowered_rules),
                );
                Module::Helper {
                    ctx,
                    lowered_rules,
                    exports,
                }
            }
        };
        let global_id = u8::try_from(self.modules.len())
            .map_err(|_| LowerError::without_span(LowerErrorKind::ModuleTooMany))?;
        self.modules.push(module);

        Ok(global_id)
    }

    /// If `e` is `UnknownIdentifier(name)` and `name` matches a cfg-dropped
    /// declaration, attach a note pointing at the gated decl with the cfg
    /// flag named in the message. Otherwise pass through unchanged.
    fn enrich_resolve_error(&self, current: &ModuleContext, mut e: ResolveError) -> ResolveError {
        let ResolveErrorKind::UnknownIdentifier(name) = &e.kind else {
            return e;
        };
        // Prefer the failing module's own drops (resolves the collision case
        // where two modules both gate a decl with the same name); fall back
        // to scanning other modules so inherited / imported drops are still
        // attributed correctly.
        let cfg_node = current.cfg_dropped.get(name).copied().or_else(|| {
            self.modules
                .iter()
                .map(Module::ctx)
                .find_map(|m| m.cfg_dropped.get(name).copied())
        });
        let Some(cfg_node) = cfg_node else {
            return e;
        };
        let Node::Cfg {
            name: flag_span, ..
        } = *self.shared.arena.get(cfg_node)
        else {
            return e; // shouldn't happen - apply_cfg only stores Cfg nodes here
        };
        // Find which module's source the cfg flag span resolves against.
        // `current` covers the module-being-processed case (not yet pushed
        // to `self.modules`).
        let module_ctx = self
            .modules
            .iter()
            .map(Module::ctx)
            .chain(std::iter::once(current))
            .find(|m| m.owns_node(cfg_node))
            .unwrap_or(current);
        let flag_name = module_ctx.text(flag_span).to_owned();
        let decl_span = self.shared.arena.span(cfg_node);
        e.add_note(module_ctx.note(NoteMessage::GatedByDisabledCfg(flag_name), decl_span));
        e
    }

    fn load_child_module(
        &mut self,
        module_path: &Path,
        span: Span,
        kind: ModuleKind,
    ) -> DslResult<ModuleId> {
        if let Some(&(_, _, gid)) = self
            .loaded
            .iter()
            .find(|(p, k, _)| *k == kind && p == module_path)
        {
            return Ok(gid);
        }

        if self.ancestor_paths.len() >= MAX_MODULE_DEPTH {
            return Err(LowerError::new(LowerErrorKind::ModuleDepthExceeded, span).into());
        }

        let content = std::fs::read_to_string(module_path).map_err(|e| {
            LowerError::new(
                LowerErrorKind::ModuleReadFailed {
                    path: module_path.to_path_buf(),
                    error: e.to_string(),
                },
                span,
            )
        })?;

        if self.ancestor_paths.iter().any(|p| p == module_path) {
            return Err(ModuleError::new(
                LowerError::without_span(LowerErrorKind::ModuleCycle).into(),
                content,
                module_path,
                span,
            ))?;
        }

        self.ancestor_paths.push(module_path.to_path_buf());
        let result = self
            .load_module(&content, module_path, kind)
            .map_err(|inner| ModuleError::new(inner, content, module_path, span));
        self.ancestor_paths.pop();
        let gid = result?;
        self.loaded.push((module_path.to_path_buf(), kind, gid));
        Ok(gid)
    }

    /// Resolve unresolved `ModuleRef` nodes (those still `module: None`),
    /// loading each child file.
    fn load_import_children(
        &mut self,
        ctx: &ModuleContext,
        module_dir: &Path,
    ) -> Result<(), DslError> {
        for &node_id in &ctx.module_refs {
            let &Node::ModuleRef {
                import: is_import,
                path: path_span,
                module: None,
            } = self.shared.arena.get(node_id)
            else {
                // Already resolved (e.g. the inherit ref handled by the caller).
                continue;
            };
            let kind = if is_import {
                ModuleKind::Helper
            } else {
                ModuleKind::Grammar
            };

            let path_str = ctx.text(path_span);
            let canonical = resolve_path(&module_dir.join(path_str), path_span)?;
            let gid = self.load_child_module(&canonical, path_span, kind)?;

            self.shared.arena.set(
                node_id,
                Node::ModuleRef {
                    import: is_import,
                    path: path_span,
                    module: Some(gid),
                },
            );
        }

        Ok(())
    }

    /// Validate that grammar block exists, inherits field consistency.
    /// Multiple `inherit()` calls are caught by the parser.
    fn validate_grammar(&self, ctx: &ModuleContext) -> DslResult<()> {
        let Some(config) = ctx.grammar_config.as_ref() else {
            return Err(LowerError::without_span(LowerErrorKind::MissingGrammarBlock).into());
        };
        // A grammar block without `language` and more than one inherit() are
        // both reported here rather than at parse, so a well-formed AST is still
        // produced for tooling. The inherit set is derived from `module_refs`.
        if config.language.is_none() {
            // The grammar block always exists here (config is Some); find it for
            // the span. Only runs on this error path, so the scan is fine.
            let block = ctx
                .root_items
                .iter()
                .find(|&&id| matches!(self.shared.arena.get(id), Node::Grammar))
                .expect("grammar config implies a grammar block node");
            return Err(LowerError::new(
                LowerErrorKind::MissingLanguageField,
                self.shared.arena.span(*block),
            )
            .into());
        }
        // Derived from module_refs (post-cfg-gating), so no cached field can go
        // stale: the first inherit is the active base, any others are errors.
        let inherits = ctx.inherits(&self.shared.arena).collect::<Vec<_>>();
        if let [first, second, rest @ ..] = inherits.as_slice() {
            // Point at the first redundant inherit, note the base, then note
            // every further redundant inherit so the user sees all of them.
            let mut err = LowerError::with_note(
                LowerErrorKind::MultipleInherits,
                self.shared.arena.span(*second),
                ctx.note(
                    NoteMessage::FirstDefinedHere,
                    self.shared.arena.span(*first),
                ),
            );
            for &extra in rest {
                err.add_note(ctx.note(
                    NoteMessage::AlsoInheritedHere,
                    self.shared.arena.span(extra),
                ));
            }
            return Err(err.into());
        }
        let config_inherits = config.inherits;

        // inherit() exists but no `inherits` in grammar config
        if let Some(&inherit_ref) = inherits.first()
            && config_inherits.is_none()
        {
            Err(LowerError::new(
                LowerErrorKind::InheritWithoutConfig,
                self.shared.arena.span(inherit_ref),
            ))?;
        }

        // `inherits` in config but doesn't resolve to an inherit() call
        if let Some(id) = config_inherits
            && inherits.is_empty()
        {
            Err(LowerError::new(
                LowerErrorKind::InheritsWithoutInherit,
                self.shared.arena.span(id),
            ))?;
        }

        Ok(())
    }

    /// Validate that an imported file only contains allowed items.
    /// Helpers may host rules, lets, macros, externals, and imports - but
    /// not a grammar block (that's a root concept) or `override rule`
    /// (overrides are a root-grammar privilege).
    fn validate_import_items(&self, ctx: &ModuleContext) -> DslResult<()> {
        for &item_id in &ctx.root_items {
            let kind = match self.shared.arena.get(item_id) {
                Node::Grammar => DisallowedItemKind::GrammarBlock,
                Node::Rule {
                    is_override: true, ..
                } => DisallowedItemKind::OverrideRule,
                _ => continue,
            };
            Err(LowerError::new(
                LowerErrorKind::ModuleDisallowedItem(kind),
                self.shared.arena.span(item_id),
            ))?;
        }
        Ok(())
    }
}

fn resolve_path(path: &Path, span: Span) -> DslResult<PathBuf> {
    dunce::canonicalize(path).map_err(|e| {
        LowerError::new(
            LowerErrorKind::ModuleResolveFailed {
                path: path.to_path_buf(),
                error: e.to_string(),
            },
            span,
        )
        .into()
    })
}
