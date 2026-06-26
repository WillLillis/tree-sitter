//! Lower-stage error taxonomy.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{MAX_CALL_DEPTH, MAX_EVAL_DEPTH};
use crate::nativedsl::LowerError;

pub type LowerResult<T> = Result<T, LowerError>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisallowedItemKind {
    OverrideRule,
    GrammarBlock,
    Inherit,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
pub enum LowerErrorKind {
    #[error("missing grammar block")]
    MissingGrammarBlock,
    #[error("grammar must have at least one rule")]
    GrammarHasNoRules,
    #[error("grammar block must have a 'language' field")]
    MissingLanguageField,
    #[error("only one inherit() call is allowed per grammar")]
    MultipleInherits,
    #[error("override rule(s) not found in base grammar or imported helpers: {}", .0.join(", "))]
    OverrideRuleNotFound(Vec<String>),
    #[error("failed to resolve '{}': {error}", path.display())]
    ModuleResolveFailed { path: PathBuf, error: String },
    #[error("failed to read '{}': {error}", path.display())]
    ModuleReadFailed { path: PathBuf, error: String },
    #[error("too many modules (max 256)")]
    ModuleTooMany,
    #[error("module import chain too deep (max 256)")]
    ModuleDepthExceeded,
    #[error("imported files cannot contain a {}", format_disallowed(*.0))]
    ModuleDisallowedItem(DisallowedItemKind),
    #[error("cycle while loading module")]
    ModuleCycle,
    #[error("expected a rule name reference")]
    ExpectedRuleName,
    #[error(
        "external token '{0}' cannot be the start rule; the start symbol must be a grammar rule"
    )]
    ExternalCannotBeStart(String),
    #[error("symbol(s) referenced but never defined: {}: ", .0.join(", "))]
    UndefinedSymbols(Vec<String>),
    #[error("object has no field '{field}'; available: {}", available.join(", "))]
    FieldNotFound {
        field: String,
        available: Vec<String>,
    },
    #[error("config field is not set in the base grammar")]
    ConfigFieldUnset,
    #[error("inherit() requires 'inherits' to be set in the grammar config")]
    InheritWithoutConfig,
    #[error("'inherits' must reference a variable bound to inherit()")]
    InheritsWithoutInherit,
    #[error("too many elements ({0}, maximum {max})", max = u16::MAX)]
    TooManyChildren(usize),
    #[error("maximum macro call depth ({MAX_CALL_DEPTH}) exceeded")]
    CallDepthExceeded(Vec<(String, PathBuf, usize, usize)>), // name, path, line, col
    #[error("expression nesting too deep while expanding macros (maximum depth {MAX_EVAL_DEPTH})")]
    RecursionTooDeep(Vec<(String, PathBuf, usize, usize)>), // name, path, line, col
    #[error("integer overflow: {0} does not fit in i32")]
    IntegerOverflow(i64),
    #[error("let '{0}' is defined in terms of itself")]
    CircularLet(String),
}

const fn format_disallowed(kind: DisallowedItemKind) -> &'static str {
    match kind {
        DisallowedItemKind::OverrideRule => "override rule",
        DisallowedItemKind::GrammarBlock => "grammar block",
        DisallowedItemKind::Inherit => "call to inherit()",
    }
}
