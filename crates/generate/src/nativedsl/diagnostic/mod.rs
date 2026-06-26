//! Diagnostics for the native DSL.
//!
//! The error types shared by every pipeline stage ([`types`]), their terminal
//! rendering ([`render`]), and the Levenshtein-based name suggestions that feed
//! "did you mean" notes ([`suggest`]).

mod render;
mod suggest;
mod types;

pub use render::NativeDslError;
pub(crate) use suggest::suggest_name;
pub use types::{
    Diagnostic, DslError, DslResult, ExpandError, LexError, LowerError, ModuleError, Note,
    NoteMessage, ParseError, ResolveError, TypeError,
};
