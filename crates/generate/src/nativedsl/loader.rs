//! Module system and glue code for the native DSL pipeline.

use std::path::{Path, PathBuf};

use crate::IoError;
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
    /// Module dedup cache. Each (canonical path, kind) loads at most once
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
    #[expect(clippy::missing_panics_doc)]
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

        // Register this module's flag declarations into the global state. This module's
        // flag values win over flags in any modules it imports.
        self.cfg.merge_module_flags(self.shared, &mut ctx)?;

        // Apply cfg gating *before* loading children so cfg-disabled imports don't trigger file
        // loads and other side effects. Skip the whole walk when no cfgs are used.
        if ctx.has_cfg {
            apply_cfg(self.shared, &mut ctx, self.cfg, kind)?;
        }

        match kind {
            ModuleKind::Grammar => self.validate_grammar(&ctx)?,
            ModuleKind::Helper => self.validate_import_items(&ctx)?,
        }

        // Inline top-level rule-set macro invocations into `ExpandedRule`
        // decls. Runs before child loads so all nodes this module owns
        // sit in one contiguous arena range.
        expand_macro_calls::expand_macro_calls(self.shared, self.strings, &mut ctx)?;
        ctx.node_range.end = self.shared.arena.next_id().into();

        // Load inherited grammar (Grammar kind only, must happen before typecheck).
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
            self.strings,
            self.modules,
            base,
            &imported_rules,
        )
        .map_err(|e| self.enrich_resolve_error(&ctx, e))?;

        // Child modules already populated `env` during their own `load_module` calls.
        typecheck::check(self.shared, &ctx, self.env)?;
        // This module's id. Checked before lowering: the evaluator derives its
        // root id from `modules.len()`, which must fit u8.
        let global_id = u8::try_from(self.modules.len())
            .map(ModuleId::from)
            .map_err(|_| LowerError::without_span(LowerErrorKind::ModuleTooMany))?;
        let module = match kind {
            ModuleKind::Grammar => {
                let lowered = Box::new(lower::lower_with_base(
                    self.state,
                    self.strings,
                    self.shared,
                    self.modules,
                    &ctx,
                    &imported_rules,
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
        // The id is this module's final index; lowering must not grow the list.
        debug_assert!(usize::from(global_id) == self.modules.len());
        self.modules.push(module);

        Ok(global_id)
    }

    /// If `e` is `UnknownIdentifier(name)` and `name` matches a cfg-dropped declaration,
    /// attach a note pointing at the gated decl with the cfg flag name.
    fn enrich_resolve_error(&self, current: &ModuleContext, mut e: ResolveError) -> ResolveError {
        let ResolveErrorKind::UnknownIdentifier(name) = &e.kind else {
            return e;
        };
        // The module that recorded the drop also parsed it, so its ctx is where
        // the cfg flag span resolves. Prefer the failing module's own drops (the
        // collision case where two modules gate the same name), then any loaded
        // module so inherited / imported drops are still attributed correctly.
        let Some((cfg_id, owner)) = current
            .cfg_dropped
            .get(name)
            .map(|&id| (id, current))
            .or_else(|| {
                self.modules
                    .iter()
                    .map(Module::ctx)
                    .find_map(|m| m.cfg_dropped.get(name).map(|&id| (id, m)))
            })
        else {
            return e;
        };
        expect_pat!(Node::Cfg { name: flag, .. }, *self.shared.arena.get(cfg_id));
        let flag_name = owner.text(flag).to_owned();
        let decl_span = self.shared.arena.span(cfg_id);
        e.add_note(owner.note(NoteMessage::GatedByDisabledCfg(flag_name), decl_span));
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

    /// Resolve `ModuleRef` nodes, loading each child file.
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
                continue; // Already resolved
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

    /// Validate that grammar block exists, `inherits` field consistency, <= 1 `inherit`
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

        // `inherits` in config but doesn't resolve to an inherit() call
        if let Some(id) = config.inherits
            && inherits.is_empty()
        {
            Err(LowerError::new(
                LowerErrorKind::InheritsWithoutInherit,
                self.shared.arena.span(id),
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
