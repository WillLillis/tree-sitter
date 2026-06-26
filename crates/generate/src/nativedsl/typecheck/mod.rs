//! Type checking for the native grammar DSL. Entry point is [`check`],
//! invoked after [`super::resolve`] has resolved identifiers.

mod check;
mod error;
pub mod types;

pub use error::{TypeErrorKind, TypeResult};
pub use types::{Constraint, DataTy, ElemTy, InnerTy, ModuleTy, ScalarTy, TupleSig, Ty};

use rustc_hash::{FxHashMap, FxHashSet};
use serde::{Deserialize, Serialize};

use check::check_item;

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
/// Returns a `TypeError` on typecheck failure.
pub fn check(shared: &mut SharedAst, ctx: &ModuleContext, env: &mut TypeEnv) -> TypeResult<()> {
    for &item_id in &ctx.root_items {
        check_item(shared, ctx, item_id, env)?;
    }
    Ok(())
}

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
