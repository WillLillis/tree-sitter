//! Intermediate representation used during lowering.
//!
//! All types here are pure data: pools, IDs, and an interned-string store.
//! The algorithm that fills these pools lives in [`super::evaluator`]; the
//! final translation to [`crate::rules::Rule`] / [`crate::grammars::InputGrammar`]
//! lives in [`super`] (mod.rs).

use std::num::NonZeroU32;

use super::super::ast::{ChildRange, Span};

/// Interned string id. Indexes into [`StringPool::entries`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Str(pub(super) NonZeroU32);

pub(super) enum StrEntry<'src> {
    Unreachable,
    Source(Span, &'src str),
    Owned(String),
}

pub(super) struct StringPool<'src> {
    pub(super) entries: Vec<StrEntry<'src>>,
}

impl<'src> StringPool<'src> {
    pub(super) fn new() -> Self {
        Self {
            entries: vec![StrEntry::Unreachable],
        }
    }

    pub(super) fn intern_span(&mut self, span: Span, source: &'src str) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Source(span, source));
        id
    }

    pub(super) fn intern_owned(&mut self, s: String) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Owned(s));
        id
    }

    pub(super) fn resolve(&self, id: Str) -> &str {
        // Safety: id was produced by intern_span/intern_owned which return
        // sequential indices into self.entries.
        match unsafe { self.entries.get_unchecked(id.0.get() as usize) } {
            StrEntry::Source(span, source) => span.resolve(source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }

    pub(super) fn to_string(&self, id: Str) -> String {
        self.resolve(id).to_string()
    }
}

/// Index into the lowering rule pool.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct RuleId(pub(super) u32);

#[derive(Clone, Copy)]
pub(super) enum APrec {
    Integer(i32),
    Name(Str),
}

/// Intermediate rule shape. Built up during evaluation, materialized into
/// [`crate::rules::Rule`] at the end via `Evaluator::build_rule`.
pub(super) enum ARule {
    Blank,
    String(Str),
    Pattern(Str, Option<Str>),
    NamedSymbol(Str),
    SeqOrChoice(bool, u32, u16),
    Repeat(RuleId),
    Prec(APrec, RuleId),
    PrecLeft(APrec, RuleId),
    PrecRight(APrec, RuleId),
    PrecDynamic(i32, RuleId),
    Field(Str, RuleId),
    Alias(Str, bool, RuleId),
    Token(bool, RuleId),
    Reserved(Str, RuleId),
}

/// Index into the lowering value pool.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct ValueId(pub(super) u32);

/// Compact value representation. Compound data (List/Tuple/Object) stores
/// indices into external pools on the Evaluator rather than inline heap data.
#[derive(Clone, Copy)]
pub(super) enum Value {
    Int(i32),
    Str(Str),
    Rule(RuleId),
    /// Index into `Evaluator::object_pool`.
    Object(u32),
    /// Range into `Evaluator::value_children`.
    List(ChildRange),
    /// Range into `Evaluator::value_children`.
    Tuple(ChildRange),
    Module(u8),
}

const LOADED_MODULES_WORDS: usize = (u8::MAX as usize + 1) / u64::BITS as usize;

/// Tracks which modules have had their let bindings evaluated. Sized to
/// the `u8` module-id ceiling: 256 bits = 4 × u64.
#[derive(Default)]
pub(super) struct LoadedModules([u64; LOADED_MODULES_WORDS]);

impl LoadedModules {
    pub(super) const fn is_loaded(&self, idx: usize) -> bool {
        self.0[idx / 64] & (1 << (idx % 64)) != 0
    }
    pub(super) const fn set_loaded(&mut self, idx: usize) {
        self.0[idx / 64] |= 1 << (idx % 64);
    }
}
