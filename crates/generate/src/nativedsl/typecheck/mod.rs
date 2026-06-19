//! Type checking for the native grammar DSL. Entry point is [`check`],
//! invoked after [`super::resolve`] has resolved identifiers.

mod check;
pub mod types;

pub use types::{Constraint, DataTy, ElemTy, InnerTy, ModuleTy, ScalarTy, TupleSig, Ty};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use check::check_item;

use super::TypeError;
use super::ast::{ModuleContext, NodeId, SharedAst};

#[derive(Clone, Default)]
pub struct TypeEnv {
    pub vars: FxHashMap<NodeId, Ty>,
    /// Field names for Object variables, keyed by the Let node's `NodeId`.
    pub object_fields: FxHashMap<NodeId, Vec<String>>,
    /// Lets currently being typed; reentry is a self-reference cycle.
    lets_in_progress: FxHashSet<NodeId>,
}

/// Walks root items and type-checks the now-resolved AST.
///
/// # Errors
///
/// Returns [`TypeError`] on typecheck failure.
pub fn check(shared: &mut SharedAst, ctx: &ModuleContext, env: &mut TypeEnv) -> TypeResult<()> {
    for &item_id in &ctx.root_items {
        check_item(shared, ctx, item_id, env)?;
    }
    Ok(())
}

pub type TypeResult<T> = Result<T, TypeError>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ContainerKind {
    List,
    Object,
}

impl std::fmt::Display for ContainerKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::List => "list",
            Self::Object => "object",
        })
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum TypeErrorKind {
    #[error("expected {expected}, got {got}")]
    TypeMismatch { expected: Ty, got: Ty },
    #[error("expected {expected}, got {got}")]
    ConstraintMismatch { expected: Constraint, got: Ty },
    #[error("undefined macro '{0}'")]
    UndefinedMacro(String),
    #[error("macro '{macro_name}': expected {expected} arguments, got {got}")]
    ArgCountMismatch {
        macro_name: String,
        expected: usize,
        got: usize,
    },
    #[error("no field '{field}' on {on_type}")]
    FieldNotFound { field: String, on_type: Ty },
    #[error("list elements have inconsistent types: {first} vs {got}")]
    ListElementTypeMismatch { first: Ty, got: Ty },
    #[error("object values must be rule_t, str_t, or int_t, got {0}")]
    InvalidObjectValue(Ty),
    #[error(
        "list elements cannot be {0}; lists allow rule_t, str_t, int_t, or list_t (max 2 levels deep)"
    )]
    InvalidListElement(Ty),
    #[error("empty {0} requires a type annotation")]
    EmptyContainerNeedsAnnotation(ContainerKind),
    #[error("annotation {declared} is not a {kind} type, but value is an empty {kind} literal")]
    EmptyContainerAnnotationMismatch { declared: Ty, kind: ContainerKind },
    #[error("for-expression requires a list, got {0}")]
    ForRequiresList(Ty),
    #[error("for with multiple bindings requires a list of tuples")]
    ForRequiresTuples,
    #[error("for has {bindings} bindings but tuples have {tuple_elements} elements")]
    ForBindingCountMismatch {
        bindings: usize,
        tuple_elements: usize,
    },
    #[error("append requires list arguments, got {0}")]
    AppendRequiresList(Ty),
    #[error("alias target must be a name or string, got {0}")]
    InvalidAliasTarget(Ty),
    #[error("expected a rule name")]
    ExpectedRuleName,
    #[error("grammar_config() requires an inherited grammar, not an imported module")]
    GrammarConfigRequiresInherit,
    #[error(
        "tuples must have {min} to {max} elements (there is no grouping operator, so `(x)` is not a value); got {0}",
        min = types::TUPLE_MIN_ARITY,
        max = types::TUPLE_MAX_ARITY
    )]
    TupleArityInvalid(usize),
    #[error("tuple elements must be rule_t, str_t, or int_t, got {0}")]
    TupleElementNotScalar(Ty),
    #[error(
        "a for-loop cannot be used here; for-loops are expanded inline and may only appear inside a sequence, choice, or list"
    )]
    BoundForLoop,
    #[error("imported module has no macro '{0}'")]
    ImportMacroNotFound(String),
    #[error("member access with `::` requires a module bound by import() or inherit(), got {0}")]
    MemberAccessRequiresModule(Ty),
    #[error(
        "module_t cannot be a macro parameter or return type; a module can only be used where it is bound by import() or inherit()"
    )]
    ModuleTypeNotAllowed,
    #[error("duplicate parameter name '{0}'")]
    DuplicateParameter(String),
    #[error("duplicate binding name '{0}'")]
    DuplicateBinding(String),
    #[error("duplicate object key '{0}'")]
    DuplicateObjectKey(String),
    #[error("for-loop requires at least one binding")]
    EmptyForBindings,
    #[error("'{0}' is a macro, not a value; call it with {0}(...)")]
    MacroUsedAsValue(String),
    #[error("rule-set macro '{0}' can only be invoked at top level, not in expression context")]
    RuleSetMacroInExpressionContext(String),
    #[error(
        "reserved must be an object literal `{{ name: [...] }}`; its first set is the default \
         applied to every token, so the set order must be explicit (not a computed value)"
    )]
    ReservedMustBeLiteral,
    #[error("let '{0}' is defined in terms of itself")]
    CircularLet(String),
}
