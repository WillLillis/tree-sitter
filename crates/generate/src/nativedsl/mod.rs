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
pub mod string_pool;
#[cfg(test)]
mod tests;
pub mod typecheck;

pub use crate::grammars::InputGrammar;
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

use rustc_hash::{FxHashMap, FxHashSet};

use crate::IoError;
use crate::rules::Rule;

use ast::{AstPools, IdentKind, ModuleContext, Node, NodeArena, RuleTarget, SharedAst, Span};
use loader::Loader;
use typecheck::TypeEnv;

/// Global module index. Every loaded module gets a unique `ModuleId`.
pub type ModuleId = u8;

/// A loaded and resolved module.
#[derive(Debug)]
pub enum Module {
    /// `Helper` modules come from `import(...)` and expose let/macro/rule/external
    /// bindings. Their rules are lowered eagerly into `lowered_rules`.
    Helper {
        ctx: ModuleContext,
        lowered_rules: Vec<(String, Rule)>,
        exports: FxHashMap<Box<str>, Export>,
    },
    /// `Grammar` modules come from `inherit(...)` (or the root grammar) and carry
    /// a fully lowered grammar for rule merging and `grammar_config` access.
    Grammar {
        ctx: ModuleContext,
        lowered: Box<InputGrammar>,
        exports: FxHashMap<Box<str>, Export>,
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
    pub fn lowered(&self) -> Option<&InputGrammar> {
        match self {
            Self::Grammar { lowered, .. } => Some(lowered),
            Self::Helper { .. } => None,
        }
    }

    /// Look up a name in this module's export table.
    #[must_use]
    pub fn export(&self, name: &str) -> Option<Export> {
        let exports = match self {
            Self::Helper { exports, .. } | Self::Grammar { exports, .. } => exports,
        };
        exports.get(name).copied()
    }

    /// The names this module exports, for "did you mean" suggestions.
    pub(crate) fn export_keys(&self) -> impl Iterator<Item = &str> {
        let exports = match self {
            Self::Helper { exports, .. } | Self::Grammar { exports, .. } => exports,
        };
        exports.keys().map(|k| &**k)
    }
}

/// The lowered output a module exposes, passed to [`build_exports`].
#[derive(Clone, Copy)]
pub enum LoweredRef<'a> {
    Grammar(&'a InputGrammar),
    Helper(&'a [(String, Rule)]),
}

// TODO: questionable wording, precedence?
/// Build a module's export table.
///
/// The table maps each name a module exposes to `mod::name` references onto an
/// ID-based [`Export`], built once when the module is constructed. Insertion
/// order encodes precedence (first wins): AST-level let/macro, then top-level
/// externals, then lowered rules/externals.
#[must_use]
pub fn build_exports(
    arena: &NodeArena,
    pools: &AstPools,
    ctx: &ModuleContext,
    lowered: LoweredRef,
) -> FxHashMap<Box<str>, Export> {
    let mut exports: FxHashMap<Box<str>, Export> = FxHashMap::default();
    // First insertion wins, so this runs highest-precedence first.
    let mut add = |name: &str, export| {
        exports.entry(name.into()).or_insert(export);
    };
    // AST-level `let` / `macro` bindings.
    for &item_id in &ctx.root_items {
        let (name, kind) = match arena.get(item_id) {
            Node::Let { name, .. } => (ctx.text(*name), IdentKind::Var(item_id)),
            Node::Macro(macro_id) => (
                ctx.text(pools.get_macro(*macro_id).name),
                IdentKind::Macro(*macro_id),
            ),
            _ => continue,
        };
        add(name, Export::Local(kind));
    }
    // `expect` forward-decls are NOT exported: a symbol is referenced across
    // modules through the lowered output below (a rule or registered external),
    // by flat name. A bare forward-decl names something defined elsewhere and so
    // has nothing to export on its own.
    // Lowered output.
    match lowered {
        LoweredRef::Grammar(g) => {
            for (i, v) in g.variables.iter().enumerate() {
                add(&v.name, Export::Rule(RuleTarget::GrammarRule(i as u32)));
            }
            for (i, r) in g.external_tokens.iter().enumerate() {
                if let Rule::NamedSymbol(n) = r {
                    add(n, Export::Rule(RuleTarget::GrammarExternal(i as u32)));
                }
            }
        }
        LoweredRef::Helper(rules) => {
            for (i, (name, _)) in rules.iter().enumerate() {
                add(name, Export::Rule(RuleTarget::HelperRule(i as u32)));
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
    let mut visited = FxHashSet::default();
    let mut stack: Vec<(u8, Span)> = Vec::new();

    let seed = |stack: &mut Vec<(u8, Span)>, refs: &[ast::NodeId]| {
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
    let mut strings = string_pool::StringPool::default();
    let mut cfg = apply_cfg::CfgState::default();
    let mut dsl_loader = Loader {
        shared: &mut shared,
        modules: &mut modules,
        env: &mut env,
        state: &mut state,
        strings: &mut strings,
        cfg: &mut cfg,
        ancestor_paths: vec![canonical.clone()],
        loaded: Vec::new(),
    };
    dsl_loader.load_module(input, &canonical, loader::ModuleKind::Grammar)?;
    // Root is the last-pushed module by construction.
    expect_pat!(Some(Module::Grammar { ctx, lowered, .. }), modules.pop());
    if lowered.variables.is_empty() {
        let span = ctx
            .root_items
            .iter()
            .find(|&&id| matches!(shared.arena.get(id), Node::Grammar))
            .map(|&id| shared.arena.span(id))
            .unwrap();
        Err(LowerError::new(LowerErrorKind::GrammarHasNoRules, span))?;
    }
    Ok(*lowered)
}
