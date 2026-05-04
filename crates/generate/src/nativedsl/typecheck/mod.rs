//! Name resolution and type checking for the native grammar DSL.
//!
//! Entry point is [`resolve_and_check`]: collects declarations, resolves
//! identifiers, then type-checks the resolved AST.

mod check;
pub mod types;

pub use types::{DataTy, InnerTy, ModuleTy, ScalarTy, Ty};

use rustc_hash::FxHashMap;
use serde::Serialize;
use thiserror::Error;

use check::check_item;

use super::ast::{MacroId, ModuleContext, NodeId, SharedAst};

#[derive(Clone, Default)]
pub struct TypeEnv {
    pub vars: FxHashMap<NodeId, Ty>,
    /// Field names for Object variables, keyed by the Let node's `NodeId`.
    pub object_fields: FxHashMap<NodeId, Vec<String>>,
    current_macro: Option<MacroId>,
}

/// Walks root items and type-checks the now-resolved AST.
///
/// # Errors
///
/// Returns [`TypeError`] on typecheck failure.
pub fn check<'ast>(
    shared: &'ast mut SharedAst,
    ctx: &'ast ModuleContext,
    env: &mut TypeEnv,
) -> Result<(), TypeError> {
    for &item_id in &ctx.root_items {
        check_item(shared, ctx, item_id, env)?;
    }
    Ok(())
}

use super::TypeError;

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum TypeErrorKind {
    #[error("expected {expected}, got {got}")]
    TypeMismatch { expected: Ty, got: Ty },
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
    #[error("field access requires obj_t, got {0}")]
    FieldAccessOnNonObject(Ty),
    #[error("list elements have inconsistent types: {first} vs {got}")]
    ListElementTypeMismatch { first: Ty, got: Ty },
    #[error("object values must be rule_t, str_t, or int_t, got {0}")]
    InvalidObjectValue(Ty),
    #[error(
        "list elements cannot be {0}; lists allow rule_t, str_t, int_t, or list_t (max 2 levels deep)"
    )]
    InvalidListElement(Ty),
    #[error("empty list requires a type annotation")]
    EmptyContainerNeedsAnnotation,
    #[error("for-expression requires a list, got {0}")]
    ForRequiresList(Ty),
    #[error("for with multiple bindings requires a list of tuples")]
    ForRequiresTuples,
    #[error("for has {bindings} bindings but tuples have {tuple_elements} elements")]
    ForBindingCountMismatch {
        bindings: usize,
        tuple_elements: usize,
    },
    #[error("unresolved variable '{0}'")]
    UnresolvedVariable(String),
    #[error("append requires list arguments, got {0}")]
    AppendRequiresList(Ty),
    #[error("'::' requires module_t, got {0}")]
    QualifiedAccessOnInvalidType(Ty),
    #[error("alias target must be a name or string, got {0}")]
    InvalidAliasTarget(Ty),
    #[error("prec value must be int_t or str_t, got {0}")]
    PrecValueTypeMismatch(Ty),
    #[error("expected a rule name")]
    ExpectedRuleName,
    #[error("reserved config must be an object literal or inherited")]
    ExpectedReservedConfig,
    #[error("grammar_config() requires an inherited grammar, not an imported module")]
    GrammarConfigRequiresInherit,
    #[error("cannot infer type of this expression")]
    CannotInferType,
    #[error("{}", non_bindable_message(*.0))]
    NonBindableType(Ty),
    #[error("imported module has no macro '{0}'")]
    ImportMacroNotFound(String),
    #[error("'::' call requires module_t, got {0}")]
    QualifiedCallOnNonModule(Ty),
    #[error("'{0}' is a macro, not a value; call it with {0}(...)")]
    MacroUsedAsValue(String),
}

const fn non_bindable_message(ty: Ty) -> &'static str {
    match ty {
        Ty::Spread => {
            "cannot bind a for-loop expansion to a variable: for-loops are expanded inline"
        }
        Ty::Tuple => {
            "tuples can only appear as elements of a for-loop iterable, not as standalone values"
        }
        _ => "this expression cannot be bound to a variable",
    }
}
