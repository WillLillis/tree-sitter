//! Module system and glue code for the native DSL.
//!
//! Dispatches the lexer, parser, and lowerer throughout the module system.

use std::path::{Path, PathBuf};

use crate::nativedsl::{
    DisallowedItemKind, DslError, DslResult, LexError, LexErrorKind, LowerError, LowerErrorKind,
    LoweringState, Module, ModuleError, ModuleId,
    ast::{ModuleContext, Node, SharedAst, Span},
    lexer, lower, parser,
    typecheck::{self, TypeEnv},
};

const MAX_MODULE_DEPTH: usize = 256;

/// Mutable pipeline state passed through the load/lower recursion.
pub(super) struct Loader<'a> {
    pub shared: &'a mut SharedAst,
    pub modules: &'a mut Vec<Module>,
    pub env: &'a mut TypeEnv,
    pub state: &'a mut LoweringState,
    pub ancestor_paths: Vec<PathBuf>,
}

#[derive(Clone, Copy)]
pub enum ModuleKind {
    /// Grammar file (root or inherited). Must have grammar block, may have rules.
    Grammar,
    /// Helper file (imported). Only let/macro/import allowed.
    Helper,
}

impl Loader<'_> {
    /// Load a tsg module, imported either via `import` or `inherit`.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the imported module isn't valid tsg source.
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
        let ctx = parser::Parser::new(&tokens, source.to_string(), path, self.shared).parse()?;

        match kind {
            ModuleKind::Grammar => self.validate_grammar(&ctx)?,
            ModuleKind::Helper => self.validate_import_items(&ctx)?,
        }

        // Load inherited grammar (Grammar kind only, must happen before typecheck).
        if let Some(inherit_id) = ctx.inherit_ref
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
        typecheck::resolve_and_check(self.shared, &ctx, self.env, self.modules, base, path)?;

        let global_id = u8::try_from(self.modules.len())
            .map_err(|_| LowerError::without_span(LowerErrorKind::ModuleTooMany))?;
        let module = match kind {
            ModuleKind::Grammar => {
                let lowered = Box::new(lower::lower_with_base(
                    self.state,
                    self.shared,
                    self.modules,
                    &ctx,
                )?);
                Module::Grammar { ctx, lowered }
            }
            ModuleKind::Helper => Module::Helper { ctx },
        };
        self.modules.push(module);

        Ok(global_id)
    }

    fn load_child_module(
        &mut self,
        module_path: &Path,
        span: Span,
        kind: ModuleKind,
    ) -> DslResult<ModuleId> {
        if self.ancestor_paths.len() >= MAX_MODULE_DEPTH {
            return Err(LowerError::new(LowerErrorKind::ModuleDepthExceeded, span).into());
        }

        let content = std::fs::read_to_string(module_path).map_err(|e| ModuleError {
            inner: Box::new(
                LowerError::new(
                    LowerErrorKind::ModuleReadFailed {
                        path: module_path.to_path_buf(),
                        error: e.to_string(),
                    },
                    span,
                )
                .into(),
            ),
            source_text: String::new(),
            path: module_path.to_path_buf(),
            reference_span: span,
        })?;

        if self.ancestor_paths.iter().any(|p| p == module_path) {
            return Err(ModuleError {
                inner: Box::new(LowerError::without_span(LowerErrorKind::ModuleCycle).into()),
                source_text: content,
                path: module_path.to_path_buf(),
                reference_span: span,
            })?;
        }

        self.ancestor_paths.push(module_path.to_path_buf());
        let result = self
            .load_module(&content, module_path, kind)
            .map_err(|inner| ModuleError {
                inner: Box::new(inner),
                source_text: content,
                path: module_path.to_path_buf(),
                reference_span: span,
            });
        self.ancestor_paths.pop();
        Ok(result?)
    }

    /// Resolve unresolved `ModuleRef` nodes (those still `module: None`),
    /// loading each child file. Deduplicates by canonical path.
    fn load_import_children(
        &mut self,
        ctx: &ModuleContext,
        module_dir: &Path,
    ) -> Result<(), DslError> {
        // Dedup cache: canonical path -> module index. Typically 0-3 entries,
        // so linear search beats hashing.
        let mut loaded: Vec<(PathBuf, ModuleId)> = Vec::new();

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

            let gid = if let Some(&(_, gid)) = loaded.iter().find(|(p, _)| p == &canonical) {
                gid
            } else {
                let gid = self.load_child_module(&canonical, path_span, kind)?;
                loaded.push((canonical, gid));
                gid
            };

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
        if ctx.grammar_config.is_none() {
            return Err(LowerError::without_span(LowerErrorKind::MissingGrammarBlock).into());
        }
        let config_inherits = ctx.grammar_config.as_ref().and_then(|c| c.inherits);

        // inherit() exists but no `inherits` in grammar config
        if let Some(inherit_ref) = ctx.inherit_ref
            && config_inherits.is_none()
        {
            Err(LowerError::new(
                LowerErrorKind::InheritWithoutConfig,
                self.shared.arena.span(inherit_ref),
            ))?;
        }

        // `inherits` in config but doesn't resolve to an inherit() call
        if let Some(id) = config_inherits
            && ctx.inherit_ref.is_none()
        {
            Err(LowerError::new(
                LowerErrorKind::InheritsWithoutInherit,
                self.shared.arena.span(id),
            ))?;
        }

        Ok(())
    }

    /// Validate that an imported file only contains allowed items.
    fn validate_import_items(&self, ctx: &ModuleContext) -> DslResult<()> {
        for &item_id in &ctx.root_items {
            let kind = match self.shared.arena.get(item_id) {
                Node::Grammar => DisallowedItemKind::GrammarBlock,
                Node::Rule { is_override, .. } => {
                    if *is_override {
                        DisallowedItemKind::OverrideRule
                    } else {
                        DisallowedItemKind::Rule
                    }
                }
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
