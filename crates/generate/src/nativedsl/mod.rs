//! Native DSL front-end for tree-sitter grammar definitions.
//!
//! Pipeline: lex -> parse -> `merge_module_flags` -> `apply_cfg` -> `validate` ->
//! `expand_macro_calls` -> `resolve` -> `typecheck` -> `lower` -> `build_exports`
//! (imported/inherited modules load before resolve). Produces an [`InputGrammar`].

/// Save the length of one or more `Vec`s used as stacks, run a body, then
/// truncate each back.
macro_rules! stack_scope {
    ($buf:expr, |$base:ident| $body:expr) => {{
        let $base = $buf.len();
        let result = {
            #[allow(clippy::redundant_closure_call, reason = "IIFE scopes `?` to the closure so truncate runs")]
            (|| $body)()
        };
        $buf.truncate($base);
        result
    }};
    ($($buf:expr => $base:ident),+; $body:expr) => {{
        $(let $base = $buf.len();)+
        let result = {
            #[allow(clippy::redundant_closure_call, reason = "IIFE scopes `?` to the closure so truncates run")]
            (|| $body)()
        };
        $($buf.truncate($base);)+
        result
    }};
}

/// Extract a node matching the given pattern from the expression
///
/// # Panics
///
/// Panics if the expression does not match the given pattern.
macro_rules! expect_pat {
    ($pat:pat, $expr:expr $(,)?) => {
        let $pat = $expr else {
            panic!("Expected {}, got {:?}", stringify!($pat), $expr);
        };
    };
}

pub mod apply_cfg;
pub mod ast;
pub mod diagnostic;
pub mod expand_macro_calls;
pub mod lexer;
pub mod loader;
pub mod lower;
pub mod parser;
pub mod resolve;
pub mod serialize;
#[cfg(test)]
mod tests;
pub mod typecheck;

pub use crate::{
    grammars::InputGrammar,
    rules::{Rule, RuleId, RulePool},
    strpool::{StrId, StrPool},
};
pub use diagnostic::{
    Diagnostic, DslError, DslResult, ExpandError, LexError, LowerError, ModuleError,
    NativeDslError, Note, NoteMessage, ParseError, ResolveError, TypeError,
};
pub use expand_macro_calls::ExpandErrorKind;
pub use lexer::{LexErrorKind, LexResult};
pub use lower::{DisallowedItemKind, LowerErrorKind, LowerResult, LoweringState};
pub use parser::{ParseErrorKind, ParseResult};
pub use resolve::{ResolveErrorKind, ResolveResult};
pub use typecheck::{
    Constraint, ContainerKind, DataTy, ElemTy, InnerTy, ModuleTy, ScalarTy, TupleSig, Ty,
    TypeErrorKind, TypeResult,
};

use std::path::Path;

use rustc_hash::{FxBuildHasher, FxHashMap};
use serde::{Deserialize, Serialize};

use crate::IoError;
use crate::grammars::{PrecedenceEntry, ReservedWordContext, Variable};

use ast::{IdentKind, ModuleContext, Node, RuleTarget, SharedAst, Span};
use loader::Loader;
use typecheck::TypeEnv;

/// Global module index. Every loaded module gets a unique `ModuleId`.
///
/// A newtype over the module-table index, distinct from the other `u8`/`u32`
/// indices in the pipeline so they can't be confused at a use site.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ModuleId(u8);

impl From<u8> for ModuleId {
    fn from(value: u8) -> Self {
        Self(value)
    }
}

impl From<ModuleId> for usize {
    fn from(id: ModuleId) -> Self {
        Self::from(id.0)
    }
}

/// Fixed bitset covering every value representable by `ModuleId`.
#[derive(Default)]
struct ModuleIdSet([u64; 4]);

impl ModuleIdSet {
    /// Inserts `id`, returning whether it was newly inserted.
    #[must_use]
    fn insert(&mut self, id: ModuleId) -> bool {
        let index = usize::from(id);
        let word_idx = index / 64;
        let bit_idx = index % 64;
        let mask = 1u64 << bit_idx;
        let newly_inserted = self.0[word_idx] & mask == 0;
        self.0[word_idx] |= mask;
        newly_inserted
    }
}

/// A module's lowered grammar, holding only IDs. The pool these IDs reference is
/// owned by the [`Loader`] and moved into the root [`InputGrammar`] once every
/// module has been lowered.
#[derive(Debug)]
pub struct LoweredGrammar {
    pub name: StrId,
    pub variables: Vec<Variable>,
    pub external_roots: Vec<RuleId>,
    pub extra_roots: Vec<RuleId>,
    pub reserved_sets: Vec<ReservedWordContext>,
    pub supertype_names: Vec<StrId>,
    pub conflict_names: Vec<Vec<StrId>>,
    pub inline_names: Vec<StrId>,
    pub word_name: Option<StrId>,
    pub precedence_orderings: Vec<Vec<PrecedenceEntry>>,
}

impl LoweredGrammar {
    /// Take ownership of the [`Loader`]-wide pool.
    fn into_input(self, pool: RulePool) -> InputGrammar {
        InputGrammar {
            pool,
            name: self.name,
            variables: self.variables,
            external_roots: self.external_roots,
            extra_roots: self.extra_roots,
            reserved_sets: self.reserved_sets,
            supertype_names: self.supertype_names,
            conflict_names: self.conflict_names,
            inline_names: self.inline_names,
            word_name: self.word_name,
            precedence_orderings: self.precedence_orderings,
        }
    }
}

/// A loaded and resolved module.
#[derive(Debug)]
pub enum Module {
    /// `Helper` modules come from `import(...)` and expose let/macro/rule/external
    /// bindings. Their rules are lowered eagerly into `lowered_rules`.
    Helper {
        ctx: ModuleContext,
        lowered_rules: Vec<(StrId, RuleId)>,
        exports: FxHashMap<StrId, Export>,
    },
    /// `Grammar` modules come from `inherit(...)` (or the root grammar) and carry
    /// a fully lowered grammar for rule merging and `grammar_config` access.
    Grammar {
        ctx: ModuleContext,
        lowered: Box<LoweredGrammar>,
        exports: FxHashMap<StrId, Export>,
    },
}

/// What a name exported by a module resolves to
#[derive(Clone, Copy, Debug)]
pub enum Export {
    /// An AST-level `let` or `macro` (resolves to `Ident(Var | Macro)`).
    Local(IdentKind),
    /// A rule / external in the module's lowered output (resolves to
    /// `Node::ModuleRule`).
    Rule(RuleTarget),
}

impl Module {
    #[must_use]
    pub const fn ctx(&self) -> &ModuleContext {
        match self {
            Self::Helper { ctx, .. } | Self::Grammar { ctx, .. } => ctx,
        }
    }

    #[must_use]
    pub fn lowered(&self) -> Option<&LoweredGrammar> {
        match self {
            Self::Grammar { lowered, .. } => Some(lowered),
            Self::Helper { .. } => None,
        }
    }

    /// Look up a name in this module's export table.
    #[must_use]
    pub fn export(&self, name: StrId) -> Option<Export> {
        let exports = match self {
            Self::Helper { exports, .. } | Self::Grammar { exports, .. } => exports,
        };
        exports.get(&name).copied()
    }

    /// The names this module exports, for "did you mean" suggestions.
    pub(crate) fn export_keys(&self) -> impl Iterator<Item = StrId> {
        let exports = match self {
            Self::Helper { exports, .. } | Self::Grammar { exports, .. } => exports,
        };
        exports.keys().copied()
    }
}

/// The lowered output a module exposes, passed to [`build_exports`].
#[derive(Clone, Copy)]
pub enum LoweredRef<'a> {
    Grammar(&'a LoweredGrammar),
    Helper(&'a [(StrId, RuleId)]),
}

/// Build a module's export table: each name this module exposes to `mod::name` refs,
/// mapped to an ID-based [`Export`]. Built once, when the [`Module`] is constructed.
#[must_use]
pub fn build_exports(
    shared: &SharedAst,
    ctx: &ModuleContext,
    rule_pool: &RulePool,
    lowered: LoweredRef,
) -> FxHashMap<StrId, Export> {
    let mut exports: FxHashMap<StrId, Export> =
        FxHashMap::with_capacity_and_hasher(ctx.root_items.len(), FxBuildHasher);
    // Every insertion below is first-wins. User names share one namespace that
    // `collect_decls` already deduped. The only name inserted twice is a symbol that
    // is both a rule and an external. Rules are inserted before externals, so it
    // resolves to the rule.
    // AST-level `let` / `macro` bindings.
    for &item_id in &ctx.root_items {
        let (name, kind) = match shared.arena.get(item_id) {
            Node::Let { name, .. } => (*name, IdentKind::Var(item_id)),
            Node::Macro(macro_id) => (
                shared.pools.get_macro(*macro_id).name.value,
                IdentKind::Macro(*macro_id),
            ),
            _ => continue,
        };
        exports.entry(name).or_insert(Export::Local(kind));
    }

    // Rules and externals
    match lowered {
        LoweredRef::Grammar(g) => {
            exports.reserve(g.variables.len() + g.external_roots.len());
            for (i, v) in g.variables.iter().enumerate() {
                exports
                    .entry(v.name)
                    .or_insert(Export::Rule(RuleTarget::GrammarRule(i as u32)));
            }
            for (i, r) in g.external_roots.iter().enumerate() {
                if let Rule::NamedSymbol(n) = rule_pool.node(*r) {
                    exports
                        .entry(n)
                        .or_insert(Export::Rule(RuleTarget::GrammarExternal(i as u32)));
                }
            }
        }
        LoweredRef::Helper(rules) => {
            exports.reserve(rules.len());
            for (i, (name, _)) in rules.iter().enumerate() {
                exports
                    .entry(*name)
                    .or_insert(Export::Rule(RuleTarget::HelperRule(i as u32)));
            }
        }
    }
    exports
}

/// A rule contributed by a transitively-imported helper, exposed as a handle
/// into the owning helper's `lowered_rules`.
///
/// Resolved once after child modules load and shared by resolve (bare-name
/// registration) and lower (materialization), so the two passes can't disagree
/// about what an import contributes.
#[derive(Clone, Copy)]
pub struct ImportedRule {
    pub module: ModuleId,
    pub index: u32,
    /// The import statement that brought the rule into scope, for error locations.
    pub ref_span: Span,
}

/// Flatten the transitive helper imports into an ordered handle list. Each helper
/// is visited once, yielding one handle per rule in source order.
pub(crate) fn collect_imported_rules(
    arena: &ast::NodeArena,
    initial_refs: &[ast::NodeId],
    modules: &[Module],
) -> Vec<ImportedRule> {
    let mut rules = Vec::new();
    let mut visited = ModuleIdSet::default();
    let mut stack: Vec<(ModuleId, Span)> = Vec::new();

    let seed = |stack: &mut Vec<(ModuleId, Span)>, refs: &[ast::NodeId]| {
        // Reverse so the LIFO stack pops imports in source order (a pre-order
        // DFS visits siblings left-to-right when pushed right-to-left).
        for &mref_id in refs.iter().rev() {
            if let &ast::Node::ModuleRef {
                import: true,
                module: Some(idx),
                ..
            } = arena.get(mref_id)
            {
                stack.push((idx, arena.span(mref_id)));
            }
        }
    };
    seed(&mut stack, initial_refs);

    while let Some((idx, ref_span)) = stack.pop() {
        if !visited.insert(idx) {
            continue;
        }
        let module = &modules[usize::from(idx)];
        if let Module::Helper { lowered_rules, .. } = module {
            for index in 0..lowered_rules.len() {
                rules.push(ImportedRule {
                    module: idx,
                    index: index as u32,
                    ref_span,
                });
            }
            seed(&mut stack, &module.ctx().module_refs);
        }
    }
    rules
}

/// Entry point. Parse a native DSL source file into an [`InputGrammar`].
///
/// # Errors
///
/// Returns [`DslError`] if any pipeline stage fails.
pub fn parse_native_dsl(input: &str, grammar_path: &Path) -> DslResult<InputGrammar> {
    let canonical = dunce::canonicalize(grammar_path).map_err(|error| {
        LowerError::without_span(LowerErrorKind::ModuleResolveFailed(IoError {
            error,
            path: Some(grammar_path.to_path_buf()),
        }))
    })?;
    let cap = input.len() / 10;
    let mut shared = SharedAst::new(cap);
    let mut modules: Vec<Module> = Vec::new();
    let mut env = TypeEnv::default();
    let mut state = LoweringState::default();
    let mut pool = RulePool::default();
    let mut cfg = apply_cfg::CfgState::default();
    let mut dsl_loader = Loader {
        shared: &mut shared,
        modules: &mut modules,
        env: &mut env,
        state: &mut state,
        pool: &mut pool,
        cfg: &mut cfg,
        ancestor_paths: vec![canonical.clone()],
        loaded: Vec::new(),
    };
    dsl_loader.load_module(input, &canonical, loader::ModuleKind::Grammar)?;
    // Root is the last-pushed module by construction.
    expect_pat!(Some(Module::Grammar { ctx, lowered, .. }), modules.pop());
    if lowered.variables.is_empty() {
        let g_span = ctx
            .root_items
            .iter()
            .find(|&&id| matches!(shared.arena.get(id), Node::Grammar))
            .map(|&id| shared.arena.span(id))
            .unwrap();
        Err(LowerError::new(LowerErrorKind::GrammarHasNoRules, g_span))?;
    }
    Ok(lowered.into_input(pool))
}
