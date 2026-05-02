//! Intermediate representation used during lowering.
//!
//! All types here are pure data: pools, IDs, and an interned-string store.
//! The algorithm that fills these pools lives in [`super::evaluator`]; the
//! final translation to [`crate::rules::Rule`] / [`crate::grammars::InputGrammar`]
//! lives in [`super`] (mod.rs).

use std::borrow::Cow;
use std::num::NonZeroU32;

use rustc_hash::FxHashMap;

use super::super::ModuleId;
use super::super::ast::{ChildRange, Span};

/// Interned string id. Indexes into [`StringPool::entries`].
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Str(pub(super) NonZeroU32);

pub(super) enum StrEntry {
    Unreachable,
    Source(Span, ModuleId),
    Owned(String),
}

pub(super) struct StringPool {
    pub(super) entries: Vec<StrEntry>,
    owned_lookup: FxHashMap<String, Str>,
}

impl Default for StringPool {
    fn default() -> Self {
        Self {
            entries: vec![StrEntry::Unreachable],
            owned_lookup: FxHashMap::default(),
        }
    }
}

impl StringPool {
    pub(super) fn intern_span(&mut self, span: Span, mod_id: ModuleId) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Source(span, mod_id));
        id
    }

    pub(super) fn intern_owned(&mut self, s: Cow<'_, str>) -> Str {
        if let Some(&id) = self.owned_lookup.get(s.as_ref()) {
            return id;
        }
        let owned = s.into_owned();
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Owned(owned.clone()));
        self.owned_lookup.insert(owned, id);
        id
    }

    /// Get the raw entry. Resolution to `&str` lives on the Evaluator since
    /// `Source` entries may reference the in-progress module whose source
    /// isn't in the `previous` slice.
    pub(super) fn entry(&self, id: Str) -> &StrEntry {
        // Safety: id was produced by intern_span/intern_owned which return
        // sequential indices into self.entries.
        unsafe { self.entries.get_unchecked(id.0.get() as usize) }
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
    Module(ModuleId),
}

const LOADED_MODULES_WORDS: usize = (u8::MAX as usize + 1) / u64::BITS as usize;

/// Tracks which modules have had their let bindings evaluated. Sized to
/// the `u8` module-id ceiling: 256 bits = 4 × u64.
#[derive(Default)]
pub(super) struct LoadedModules([u64; LOADED_MODULES_WORDS]);

impl LoadedModules {
    pub(super) const fn is_loaded(&self, idx: ModuleId) -> bool {
        let i = idx as usize;
        self.0[i / 64] & (1 << (i % 64)) != 0
    }
    pub(super) const fn set_loaded(&mut self, idx: ModuleId) {
        let i = idx as usize;
        self.0[i / 64] |= 1 << (i % 64);
    }
}
