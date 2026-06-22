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
mod suggest;
#[cfg(test)]
mod tests;
pub mod typecheck;

pub use crate::grammars::InputGrammar;
pub use diagnostic::NativeDslError;
pub use expand_macro_calls::ExpandErrorKind;
pub use lexer::{LexErrorKind, LexResult};
pub use lower::{DisallowedItemKind, LowerErrorKind, LowerResult, LoweringState};
pub use parser::{ParseErrorKind, ParseResult};
pub use resolve::{ResolveErrorKind, ResolveResult};
pub use typecheck::{
    Constraint, ContainerKind, DataTy, ElemTy, InnerTy, ModuleTy, ScalarTy, TupleSig, Ty,
    TypeErrorKind, TypeResult,
};

use std::path::{Path, PathBuf};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use thiserror::Error;

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
    // TODO: Should external decls be exported from a module? Seems wrong but need to check
    // roundtrip tests
    // Top-level `external X` declarations.
    for &item_id in &ctx.root_items {
        if let Node::External { name } = *arena.get(item_id) {
            add(ctx.text(name), Export::Rule(RuleTarget::ExternalName(name)));
        }
    }
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
    let canonical = dunce::canonicalize(grammar_path).map_err(|e| {
        LowerError::without_span(LowerErrorKind::ModuleResolveFailed {
            path: grammar_path.to_path_buf(),
            error: e.to_string(),
        })
    })?;
    let cap = input.len() / 30;
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

pub type DslResult<T> = Result<T, DslError>;

pub type LexError = Diagnostic<LexErrorKind>;
pub type ParseError = Diagnostic<ParseErrorKind>;
pub type ResolveError = Diagnostic<ResolveErrorKind>;
pub type TypeError = Diagnostic<TypeErrorKind>;
pub type LowerError = Diagnostic<LowerErrorKind>;
pub type ExpandError = Diagnostic<ExpandErrorKind>;

/// Diagnostic error shared by all pipeline stages.
#[derive(Debug, Serialize, Deserialize, Error)]
pub struct Diagnostic<K> {
    pub kind: K,
    pub span: Option<Span>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<Note>,
    /// Source + path the primary `span` indexes into, when that differs from the
    /// module the error surfaces in (a lower error born evaluating an imported
    /// module's macro body). `None` uses the caller-supplied source.
    #[serde(skip)]
    pub src: Option<Box<(String, PathBuf)>>,
}

impl<K> Diagnostic<K> {
    pub const fn new(kind: K, span: Span) -> Self {
        Self {
            kind,
            span: Some(span),
            notes: Vec::new(),
            src: None,
        }
    }

    pub const fn without_span(kind: K) -> Self {
        Self {
            kind,
            span: None,
            notes: Vec::new(),
            src: None,
        }
    }

    pub fn with_note(kind: K, span: Span, note: Note) -> Self {
        Self {
            kind,
            span: Some(span),
            notes: vec![note],
            src: None,
        }
    }

    /// Stamp the source + path the primary span belongs to, so a cross-module
    /// error renders against the right file. See [`Diagnostic::src`].
    #[must_use]
    pub fn with_source(mut self, source: &str, path: &Path) -> Self {
        self.src = Some(Box::new((source.to_owned(), path.to_owned())));
        self
    }

    pub fn add_note(&mut self, note: Note) {
        self.notes.push(note);
    }
}

impl<K: std::fmt::Display> std::fmt::Display for Diagnostic<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}

#[derive(Debug, Error, Serialize, Deserialize)]
#[error(transparent)]
pub enum DslError {
    Lex(#[from] LexError),
    Parse(#[from] ParseError),
    Expand(#[from] ExpandError),
    Resolve(#[from] ResolveError),
    Type(#[from] TypeError),
    Lower(#[from] LowerError),
    Module(#[from] ModuleError),
}

/// Error from loading a child module (inherited or imported).
#[derive(Debug, Serialize, Deserialize, Error)]
#[error("{inner}")]
pub struct ModuleError {
    pub inner: Box<DslError>,
    #[serde(skip)]
    pub source_text: String,
    #[serde(skip)]
    pub path: PathBuf,
    pub reference_span: Span,
}

impl ModuleError {
    #[must_use]
    pub fn new(inner: DslError, source_text: String, path: &Path, reference_span: Span) -> Self {
        Self {
            inner: Box::new(inner),
            source_text,
            path: path.to_path_buf(),
            reference_span,
        }
    }
}

/// Secondary annotation on an error, pointing to a related source location.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Note {
    pub message: NoteMessage,
    pub span: Span,
    #[serde(skip)]
    pub path: PathBuf,
    #[serde(skip)]
    pub src: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoteMessage {
    FirstDefinedHere,
    ReferencedFromHere,
    /// An unmatched `override rule` declaration's location
    OverrideDeclaredHere,
    /// One per redundant `inherit()` beyond the first in a `MultipleInherits` error
    AlsoInheritedHere,
    /// Carries the cfg flag name, the note's span points at the gated decl.
    GatedByDisabledCfg(String),
    DidYouMean(String),
    /// Emitted alongside `GrammarConfigRequiresInherit`. Carries the imported path string.
    SwitchImportToInherit(String),
    /// Points at a let value's reference back to itself.
    SelfReferenceHere,
    /// Attached to the arg-count error when `prec_left`/`prec_right` is given a single arg.
    PrecNeedsExplicitPrecedence {
        is_left: bool,
        arg: String,
    },
}

impl std::fmt::Display for NoteMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FirstDefinedHere => write!(f, "first defined here"),
            Self::ReferencedFromHere => write!(f, "referenced from here"),
            Self::OverrideDeclaredHere => write!(f, "override declared here"),
            Self::AlsoInheritedHere => write!(f, "also inherited here"),
            Self::GatedByDisabledCfg(flag) => {
                write!(f, "this declaration is disabled by `#[cfg({flag})]`")
            }
            Self::DidYouMean(name) => write!(f, "did you mean `{name}`?"),
            Self::SwitchImportToInherit(path) => {
                write!(f, "switch `import(\"{path}\")` to `inherit(\"{path}\")`")
            }
            Self::SelfReferenceHere => write!(f, "self-reference here"),
            Self::PrecNeedsExplicitPrecedence { is_left, arg } => {
                let name = if *is_left { "prec_left" } else { "prec_right" };
                write!(
                    f,
                    "{name} requires a precedence, did you mean `{name}(0, {arg})`?"
                )
            }
        }
    }
}

impl DslError {
    #[must_use]
    pub const fn span(&self) -> Option<Span> {
        match self {
            Self::Lex(e) => e.span,
            Self::Parse(e) => e.span,
            Self::Expand(e) => e.span,
            Self::Resolve(e) => e.span,
            Self::Type(e) => e.span,
            Self::Lower(e) => e.span,
            Self::Module(e) => Some(e.reference_span),
        }
    }

    #[must_use]
    pub fn call_trace(&self) -> Option<&[(String, PathBuf, usize, usize)]> {
        if let Self::Lower(e) = self
            && let lower::LowerErrorKind::CallDepthExceeded(trace)
            | lower::LowerErrorKind::RecursionTooDeep(trace) = &e.kind
        {
            Some(trace)
        } else {
            None
        }
    }

    #[must_use]
    pub fn notes(&self) -> &[Note] {
        match self {
            Self::Lex(e) => &e.notes,
            Self::Parse(e) => &e.notes,
            Self::Expand(e) => &e.notes,
            Self::Resolve(e) => &e.notes,
            Self::Type(e) => &e.notes,
            Self::Lower(e) => &e.notes,
            Self::Module(e) => e.inner.notes(),
        }
    }

    /// Source + path the primary span belongs to, when the error carries one (a
    /// lower error from an imported module's macro body). Lets the renderer put
    /// the caret in that module's file instead of the root's.
    #[must_use]
    pub fn primary_source(&self) -> Option<(&str, &Path)> {
        let src = match self {
            Self::Lex(e) => &e.src,
            Self::Parse(e) => &e.src,
            Self::Expand(e) => &e.src,
            Self::Resolve(e) => &e.src,
            Self::Type(e) => &e.src,
            Self::Lower(e) => &e.src,
            Self::Module(e) => return e.inner.primary_source(),
        };
        src.as_deref().map(|(s, p)| (s.as_str(), p.as_path()))
    }
}
