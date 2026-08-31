//! Module system and glue code for the native DSL pipeline.

use std::path::{Path, PathBuf};

use crate::{
    IoError,
    nativedsl::{
        DisallowedItemKind, DslError, DslResult, Export, LexError, LexErrorKind, LowerError,
        LowerErrorKind, LoweringState, MAX_MODULE_DEPTH, Module, ModuleError, ModuleId,
        ModuleIdSet, NoteMessage, ResolveError, TypeError, TypeErrorKind,
        apply_cfg::{CfgEnvId, CfgState, apply_cfg},
        ast::{IdentKind, ModuleContext, Node, NodeId, SharedAst, Span},
        expand_macro_calls, lexer, lower, parser,
        resolve::{self, ResolveErrorKind},
        typecheck::{self, TypeEnv},
    },
    rules::RulePool,
    strpool::StrId,
};

/// Mutable pipeline state passed through the load/lower recursion.
pub struct Loader<'a> {
    shared: &'a mut SharedAst,
    modules: &'a mut Vec<Module>,
    env: &'a mut TypeEnv,
    state: &'a mut LoweringState,
    pool: &'a mut RulePool,
    cfg: &'a mut CfgState,
    ancestor_paths: Vec<PathBuf>,
    /// Module dedup cache keyed by canonical path, kind, and cfg environment.
    loaded: Vec<LoadedModuleRef>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ModuleKind {
    /// Grammar file (root or inherited). Must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported). Anything except a grammar block or override rule.
    Helper,
}

/// A reference to a module loaded into [`Loader::modules`].
struct LoadedModuleRef {
    /// Canonicalized path to the module on disk.
    path: PathBuf,
    kind: ModuleKind,
    cfg_env: CfgEnvId,
    /// Index into the global [`Loader::modules`] table.
    gid: ModuleId,
}

impl<'a> Loader<'a> {
    pub const fn new(
        shared: &'a mut SharedAst,
        modules: &'a mut Vec<Module>,
        env: &'a mut TypeEnv,
        state: &'a mut LoweringState,
        pool: &'a mut RulePool,
        cfg: &'a mut CfgState,
    ) -> Self {
        Self {
            shared,
            modules,
            env,
            state,
            pool,
            cfg,
            ancestor_paths: Vec::new(),
            loaded: Vec::new(),
        }
    }

    /// Load a root tsg grammar module.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the module isn't valid tsg source.
    pub fn load_root(mut self, source: &str, path: &Path) -> DslResult<()> {
        let canonical = dunce::canonicalize(path).map_err(|error| {
            LowerError::without_span(LowerErrorKind::ModuleResolveFailed(IoError {
                error,
                path: Some(path.to_path_buf()),
            }))
        })?;
        self.ancestor_paths.push(canonical.clone());
        _ = self.load_module(source, &canonical, ModuleKind::Grammar)?;
        Ok(())
    }

    fn load_module(&mut self, source: &str, path: &Path, kind: ModuleKind) -> DslResult<ModuleId> {
        if source.len() >= u32::MAX as usize {
            Err(LexError::without_span(LexErrorKind::InputTooLarge))?;
        }

        // The root's capacity is seeded by `parse_native_dsl`. Reserve here for children.
        if self.ancestor_paths.len() > 1 {
            self.shared.reserve_for_module(source.len());
        }

        let tokens = lexer::Lexer::new(source).tokenize()?;
        let mut ctx = parser::Parser::new(
            &tokens,
            source.to_string(),
            path.to_path_buf(),
            self.shared,
            self.pool.strs_mut(),
        )
        .parse()?;

        // Merge this module's flags into the current environment. Existing values
        // win, so parent declarations override child declarations.
        self.cfg
            .merge_module_flags(self.shared, &mut ctx, self.pool.strs())?;

        // Apply cfg gating *before* loading children so cfg-disabled imports aren't evaluated.
        if ctx.has_cfg {
            apply_cfg(self.shared, &mut ctx, self.pool.strs(), self.cfg, kind)?;
        }

        match kind {
            ModuleKind::Grammar => self.validate_grammar(&ctx)?,
            ModuleKind::Helper => self.validate_import_items(&ctx)?,
        }

        // Inline top-level rule-set macro invocations into `ExpandedRule` decls.
        // Runs before child loads so this module's nodes all sit in one contiguous
        // arena range.
        expand_macro_calls::expand_macro_calls(self.shared, self.pool.strs_mut(), &mut ctx)?;
        ctx.node_range.end = self.shared.arena.next_id().into();

        self.load_children(&ctx)?;

        // Child loading is complete, so this module's final table index is fixed.
        let global_id = ModuleId::from_index(self.modules.len())
            .ok_or_else(|| LowerError::without_span(LowerErrorKind::ModuleTooMany))?;

        // Flatten the transitive helper imports once
        let imported_rules =
            super::collect_imported_rules(&self.shared.arena, &ctx.module_refs, self.modules);

        // Resolve identifiers
        let base = ctx
            .inherit_module(&self.shared.arena)
            .and_then(|(idx, span)| self.modules[usize::from(idx)].lowered().map(|g| (g, span)));
        resolve::resolve(
            self.shared,
            &ctx,
            self.pool,
            self.modules,
            global_id,
            base,
            &imported_rules,
        )
        .map_err(|e| self.enrich_resolve_error(&ctx, e))?;

        // Child modules already populated `self.env` during their own `load_module` calls.
        typecheck::check(self.shared, &ctx, self.env, self.pool.strs())
            .map_err(|e| self.enrich_type_error(&ctx, e))?;
        let module = match kind {
            ModuleKind::Grammar => {
                let lowered = Box::new(lower::lower_grammar(
                    self.state,
                    self.pool,
                    self.shared,
                    self.modules,
                    &ctx,
                    &imported_rules,
                )?);
                let exports = super::build_exports(
                    self.shared,
                    &ctx,
                    self.pool,
                    &lowered.variables,
                    &lowered.external_roots,
                );
                Module::Grammar {
                    ctx,
                    lowered,
                    exports,
                }
            }
            ModuleKind::Helper => {
                let lowered_rules =
                    lower::lower_helper(self.state, self.pool, self.shared, self.modules, &ctx)?;
                let exports =
                    super::build_exports(self.shared, &ctx, self.pool, &lowered_rules, &[]);
                Module::Helper {
                    ctx,
                    lowered_rules,
                    exports,
                }
            }
        };
        debug_assert_eq!(usize::from(global_id), self.modules.len());
        self.modules.push(module);

        Ok(global_id)
    }

    /// If `e` is `UnknownIdentifier(name)` and `name` matches a cfg-dropped declaration,
    /// attach a note pointing at the gated decl with the cfg flag name.
    fn enrich_resolve_error(&self, current: &ModuleContext, mut e: ResolveError) -> ResolveError {
        let ResolveErrorKind::UnknownIdentifier(name) = &e.kind else {
            return e;
        };
        let Some(name_id) = self.pool.strs().get(name) else {
            return e;
        };
        let Some((cfg_id, owner)) = self.find_cfg_drop(current, name_id) else {
            return e;
        };
        expect_pat!(Node::Cfg { name: flag, .. }, *self.shared.arena.get(cfg_id));
        let flag_name = self.pool.strs().resolve(flag).to_string();
        let decl_span = self.shared.arena.span(cfg_id);
        e.add_note(owner.note(NoteMessage::GatedByDisabledCfg(flag_name), decl_span));
        e
    }

    /// Attach a cross-module definition note when a qualified call targets a non-macro export.
    fn enrich_type_error(&self, current: &ModuleContext, mut e: TypeError) -> TypeError {
        let TypeErrorKind::UndefinedMacro(_) = e.kind else {
            return e;
        };
        let Some(call_span) = e.span else {
            return e;
        };
        let arena = &self.shared.arena;
        let Some(callee) = current.iter_own_nodes(arena).find_map(|(id, node)| {
            if let Node::Call { name, .. } = *node
                && arena.span(id) == call_span
            {
                Some(name)
            } else {
                None
            }
        }) else {
            return e;
        };

        let (module, member) = match *arena.get(callee) {
            Node::Ident(IdentKind::Var(let_id)) => {
                expect_pat!(Node::Let { name, .. }, *arena.get(let_id));
                let Some(module) = self.modules.iter().find(|module| {
                    matches!(
                        module.export(name),
                        Some(Export::Local(IdentKind::Var(id))) if id == let_id
                    )
                }) else {
                    return e;
                };
                (module, name)
            }
            Node::ModuleRule { module, member, .. } => {
                let module = &self.modules[usize::from(module)];
                (module, member)
            }
            _ => return e,
        };

        let Some(decl) = module.ctx().root_items.iter().copied().find(|&id| {
            let decl_name = match *arena.get(id) {
                Node::Let { name, .. } | Node::Rule { name, .. } => name,
                Node::ExpandedRule(expand_id) => self.shared.pools.get_expansion(expand_id).name,
                _ => return false,
            };
            decl_name == member
        }) else {
            return e;
        };

        e.add_note(
            module
                .ctx()
                .note(NoteMessage::DefinedHere, arena.span(decl)),
        );
        e
    }

    fn load_child_module(
        &mut self,
        module_path: &Path,
        span: Span,
        kind: ModuleKind,
    ) -> DslResult<ModuleId> {
        let cfg_env = self.cfg.env_id();
        if let Some(module_ref) = self.loaded.iter().find(|module| {
            module.kind == kind && module.cfg_env == cfg_env && module.path == module_path
        }) {
            return Ok(module_ref.gid);
        }

        if self.ancestor_paths.len() >= MAX_MODULE_DEPTH {
            return Err(LowerError::new(LowerErrorKind::ModuleDepthExceeded, span).into());
        }

        let content = std::fs::read_to_string(module_path).map_err(|error| {
            LowerError::new(
                LowerErrorKind::ModuleReadFailed(IoError {
                    error,
                    path: Some(module_path.to_path_buf()),
                }),
                span,
            )
        })?;

        if self.ancestor_paths.iter().any(|p| p == module_path) {
            return Err(ModuleError::new(
                LowerError::without_span(LowerErrorKind::ModuleCycle).into(),
                content,
                module_path,
                span,
            )
            .into());
        }

        self.ancestor_paths.push(module_path.to_path_buf());
        let checkpoint = self.cfg.checkpoint();
        let result = self
            .load_module(&content, module_path, kind)
            .map_err(|inner| ModuleError::new(inner, content, module_path, span));
        self.cfg.restore(checkpoint);
        self.ancestor_paths.pop();
        let gid = result?;
        self.loaded.push(LoadedModuleRef {
            path: module_path.to_path_buf(),
            kind,
            cfg_env,
            gid,
        });
        Ok(gid)
    }

    /// Resolve `Import` and `Inherit` nodes, loading each child file.
    fn load_children(&mut self, ctx: &ModuleContext) -> Result<(), DslError> {
        let module_dir = ctx.path.parent().unwrap();
        for &node_id in &ctx.module_refs {
            let (kind, &path) = match self.shared.arena.get(node_id) {
                Node::Inherit { path, module: None } => (ModuleKind::Grammar, path),
                Node::Import { path, module: None } => (ModuleKind::Helper, path),
                Node::Inherit {
                    module: Some(_), ..
                }
                | Node::Import {
                    module: Some(_), ..
                } => continue, // Already resolved
                _ => unreachable!(),
            };

            let path_str = ctx.text(path);
            let canonical = resolve_path(&module_dir.join(path_str), path)?;
            let gid = self.load_child_module(&canonical, path, kind)?;

            match self.shared.arena.get_mut(node_id) {
                Node::Inherit { module, .. } | Node::Import { module, .. } => {
                    *module = Some(gid);
                }
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    /// Validate that grammar block exists and structural constraints on `inherit()` calls
    fn validate_grammar(&self, ctx: &ModuleContext) -> DslResult<()> {
        let Some(config) = ctx.grammar_config.as_ref() else {
            return Err(LowerError::without_span(LowerErrorKind::MissingGrammarBlock).into());
        };
        if config.language.is_none() {
            let block = ctx
                .root_items
                .iter()
                .find(|&&id| matches!(self.shared.arena.get(id), Node::Grammar))
                .unwrap();
            Err(LowerError::new(
                LowerErrorKind::MissingLanguageField,
                self.shared.arena.span(*block),
            ))?;
        }
        let inherits = ctx.inherits(&self.shared.arena).collect::<Vec<_>>();
        if let [first, second, rest @ ..] = inherits.as_slice() {
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
            Err(err)?;
        }

        // `inherit()` exists but no `inherits` in grammar config
        if let Some(&inherit_ref) = inherits.first()
            && config.inherits.is_none()
        {
            Err(LowerError::new(
                LowerErrorKind::InheritWithoutConfig,
                self.shared.arena.span(inherit_ref),
            ))?;
        }

        Ok(())
    }

    /// Validate that an `import`ed file only contains allowed items (No grammar
    /// block, `override` rules, or `inherit` calls).
    fn validate_import_items(&self, ctx: &ModuleContext) -> DslResult<()> {
        if let Some(inherit_id) = ctx.inherits(&self.shared.arena).next() {
            Err(LowerError::new(
                LowerErrorKind::ModuleDisallowedItem(DisallowedItemKind::Inherit),
                self.shared.arena.span(inherit_id),
            ))?;
        }
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

    fn find_cfg_drop<'b>(
        &'b self,
        current: &'b ModuleContext,
        name: StrId,
    ) -> Option<(NodeId, &'b ModuleContext)> {
        if let Some(&id) = current.cfg_dropped.get(&name) {
            return Some((id, current));
        }

        let mut visited = ModuleIdSet::default();
        let mut stack = Vec::new();
        let push_deps = |ctx: &ModuleContext, stack: &mut Vec<ModuleId>| {
            for &ref_id in ctx.module_refs.iter().rev() {
                if let &Node::Import {
                    module: Some(module),
                    ..
                } = self.shared.arena.get(ref_id)
                {
                    stack.push(module);
                }
            }
            if let Some((base, _)) = ctx.inherit_module(&self.shared.arena) {
                stack.push(base);
            }
        };

        // Search only modules whose declarations are visible from the failing module.
        push_deps(current, &mut stack);
        while let Some(module) = stack.pop() {
            if !visited.insert(module) {
                continue;
            }
            let ctx = self.modules[usize::from(module)].ctx();
            if let Some(&id) = ctx.cfg_dropped.get(&name) {
                return Some((id, ctx));
            }
            push_deps(ctx, &mut stack);
        }
        None
    }
}

fn resolve_path(path: &Path, span: Span) -> DslResult<PathBuf> {
    dunce::canonicalize(path).map_err(|error| {
        LowerError::new(
            LowerErrorKind::ModuleResolveFailed(IoError {
                error,
                path: Some(path.to_path_buf()),
            }),
            span,
        )
        .into()
    })
}
