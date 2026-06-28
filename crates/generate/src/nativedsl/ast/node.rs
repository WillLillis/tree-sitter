//! The core AST [`Node`] enum and the small payload enums it carries.

use super::{ChildRange, ConfigField, ExpandId, ForId, MacroId, NodeId, Span};
use crate::nativedsl::ModuleId;
use crate::nativedsl::typecheck::Ty;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PrecKind {
    Default,
    Left,
    Right,
    Dynamic,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RepeatKind {
    ZeroOrMore,
    OneOrMore,
    Optional,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum BinOp {
    Add,
    Sub,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum IdentKind {
    Unresolved,
    Rule,
    Var(NodeId),
    Macro(MacroId),
}

/// Where a [`Node::ModuleRule`] reference points within another module's
/// lowered output.
///
/// Each variant carries an index, so the lowerer dereferences directly instead
/// of scanning by name. Cross-module names that resolve to an AST-level
/// `let`/`macro` become [`IdentKind`] instead; see [`Export`](crate::nativedsl::Export).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RuleTarget {
    /// Index into the target grammar's `lowered.variables`.
    GrammarRule(u32),
    /// Index into the target grammar's `lowered.external_tokens`.
    GrammarExternal(u32),
    /// Index into the target helper's `lowered_rules`.
    HelperRule(u32),
}

#[derive(Clone, Copy, Debug)]
pub enum Node {
    Grammar,
    Rule {
        is_override: bool,
        name: Span,
        body: NodeId,
    },
    /// `rule @<str_expr> { ... }` inside a rule-set macro body.
    /// `name_expr` is `str_t`. The `expand_macro_calls` evaluates
    /// it and emits an `ExpandedRule` node.
    ComputedRule {
        is_override: bool,
        name_expr: NodeId,
        body: NodeId,
    },
    /// Body of a rule-set macro (sequence of `Rule` / `ComputedRule`).
    RuleSet(ChildRange),
    /// One instantiation of a rule-set macro decl at a `@name(args)` call site.
    /// The [`Expansion`](super::Expansion) (side table) carries the computed name, the macro's
    /// id, the *shared template* decl body, and the call's args.
    ExpandedRule(ExpandId),
    /// `let name = value` or `let name: ty = value`. An optional type
    /// annotation lives out-of-line in [`ModuleContext::let_types`](super::ModuleContext) (keyed by
    /// this node's id) so it doesn't widen `Node` past its 16-byte budget.
    Let {
        name: Span,
        value: NodeId,
    },
    Macro(MacroId),
    /// `expect <name>` top-level forward-declaration: names a symbol defined
    /// elsewhere (a rule, an `externals:` token, or an inherited rule) so this
    /// file can reference it before/without defining it here.
    Forward {
        name: Span,
    },
    StringLit,
    RawStringLit {
        hash_count: u8,
    },
    IntLit(i64),
    Ident(IdentKind),
    FieldAccess {
        obj: NodeId,
        field: Span,
    },
    QualifiedAccess {
        obj: NodeId,
        member: Span,
    },
    /// A `mod::name` reference resolved to a rule (or external symbol) in
    /// another module's lowered output. Replaces a `QualifiedAccess` during
    /// resolve once the target is found.
    ModuleRule {
        module: ModuleId,
        target: RuleTarget,
    },
    /// `seq(a, b, ...)` or `choice(a, b, ...)`. Children: `[member0, member1, ...]`
    SeqOrChoice {
        seq: bool,
        range: ChildRange,
    },
    Repeat {
        kind: RepeatKind,
        inner: NodeId,
    },
    Blank,
    Field {
        name: Span,
        content: NodeId,
    },
    Alias {
        content: NodeId,
        target: NodeId,
    },
    Token {
        immediate: bool,
        inner: NodeId,
    },
    Prec {
        kind: PrecKind,
        value: NodeId,
        content: NodeId,
    },
    Reserved {
        context: Span,
        content: NodeId,
    },
    /// `concat(a, b, ...)`. Children: `[part0, part1, ...]` - each must be `str_t`.
    Concat(ChildRange),
    /// `regexp(pattern)` or `regexp(pattern, flags)`.
    DynRegex {
        pattern: NodeId,
        flags: Option<NodeId>,
    },
    /// `inherit("path.tsg")` or `import("path.tsg")`. `module` is `None`
    /// after parsing, set to a global module index by the loading pre-pass.
    ModuleRef {
        import: bool,
        path: Span,
        module: Option<ModuleId>,
    },
    /// `grammar_config(module, field)` - access a specific config field from
    /// an inherited grammar module.
    GrammarConfig {
        module: NodeId,
        field: ConfigField,
    },
    /// `expr::name(args)` - macro call through `::` access.
    /// Children layout: `[obj, name, arg0, arg1, ...]` where obj is the
    /// namespace value and name is the macro identifier.
    QualifiedCall(ChildRange),
    Append {
        left: NodeId,
        right: NodeId,
    },
    /// `for (binding) in <iter> { <body> }`. Expression-context only.
    For {
        for_id: ForId,
        body: NodeId,
    },
    /// `@<str_expr>` inside a rule-set macro body: a reference to a (possibly
    /// computed-name) rule. `expr` is `str_t`.
    SymRef {
        expr: NodeId,
    },
    /// `name(arg0, arg1, ...)`. `args` children: `[arg0, arg1, ...]`.
    Call {
        name: NodeId,
        args: ChildRange,
    },
    /// `[elem0, elem1, ...]`. Children: `[elem0, elem1, ...]`.
    List(ChildRange),
    /// `(elem0, elem1, ...)`. Children: `[elem0, elem1, ...]`.
    Tuple(ChildRange),
    /// `{ key0: val0, key1: val1, ... }`. Fields stored in `SharedAst::object_fields`.
    Object(ChildRange),
    Neg(NodeId),
    /// Integer arithmetic: `lhs + rhs` or `lhs - rhs`. Both operands typecheck
    /// as `int_t`. No multiplication, division, parens, or precedence; the
    /// parser only emits left-associative chains of `+`/`-`.
    BinOp {
        op: BinOp,
        lhs: NodeId,
        rhs: NodeId,
    },
    /// Parameter reference in a macro body. `index` is the position into the
    /// call's arg list; `ty` lets the typechecker avoid a macro-context lookup.
    MacroParam {
        ty: Ty,
        index: u8,
    },
    /// Binding reference in a for-loop body. `for_id` disambiguates the
    /// enclosing frame (for-loops nest); `index` selects within it; `ty`
    /// lets the typechecker avoid a pool lookup.
    ForBinding {
        for_id: ForId,
        ty: Ty,
        index: u8,
    },
    /// `#[cfg(NAME)] ITEM`: gates `child` on the named flag. Survives parse,
    /// and is then dropped (or unwrapped) by the `apply_cfg` pass before resolve.
    Cfg {
        name: Span,
        child: NodeId,
    },
    /// Sentinel value occupying index 0 in the arena. Not part of the public API.
    #[doc(hidden)]
    Unreachable,
}

const _: () = assert!(std::mem::size_of::<Node>() == 16);

impl Node {
    /// Range of children for variadic nodes whose children are a flat list of
    /// `NodeId`s: `SeqOrChoice`, `List`, `Tuple`, `Concat`, and `RuleSet`.
    #[must_use]
    pub const fn child_range(&self) -> Option<ChildRange> {
        match self {
            Self::SeqOrChoice { range: r, .. }
            | Self::List(r)
            | Self::Tuple(r)
            | Self::Concat(r)
            | Self::RuleSet(r) => Some(*r),
            _ => None,
        }
    }

    /// Mutable counterpart to [`Self::child_range`].
    pub const fn child_range_mut(&mut self) -> Option<&mut ChildRange> {
        match self {
            Self::SeqOrChoice { range: r, .. }
            | Self::List(r)
            | Self::Tuple(r)
            | Self::Concat(r)
            | Self::RuleSet(r) => Some(r),
            _ => None,
        }
    }
}
