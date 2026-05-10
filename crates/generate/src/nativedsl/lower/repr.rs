//! Intermediate representation used during lowering.

use std::borrow::Cow;
use std::num::NonZeroU32;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use super::super::ModuleId;
use super::super::ast::{ChildRange, Span};

/// Interned string id. Indexes into [`StringPool::entries`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Str(pub NonZeroU32);

pub enum StrEntry {
    Unreachable,
    Source(Span, ModuleId),
    Owned(Rc<str>),
}

pub struct StringPool {
    pub entries: Vec<StrEntry>,
    /// Dedup table for `Owned` entries. The `Rc<str>` key shares its allocation
    /// with the matching `StrEntry::Owned`, so each unique string only allocates
    /// once.
    owned_lookup: FxHashMap<Rc<str>, Str>,
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
    pub fn intern_span(&mut self, span: Span, mod_id: ModuleId) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Source(span, mod_id));
        id
    }

    pub fn intern_owned(&mut self, s: Cow<'_, str>) -> Str {
        if let Some(&id) = self.owned_lookup.get(s.as_ref()) {
            return id;
        }
        let owned: Rc<str> = Rc::from(s.as_ref());
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Owned(Rc::clone(&owned)));
        self.owned_lookup.insert(owned, id);
        id
    }

    /// Get the raw entry. Resolution to `&str` lives on the Evaluator since
    /// `Source` entries may reference the in-progress module whose source
    /// isn't in the `previous` slice.
    pub fn entry(&self, id: Str) -> &StrEntry {
        // Safety: id was produced by intern_span/intern_owned which return
        // sequential indices into self.entries.
        unsafe { self.entries.get_unchecked(id.0.get() as usize) }
    }
}

/// Index into the lowering rule pool.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RuleId(pub u32);

#[derive(Clone, Copy)]
pub enum APrec {
    Integer(i32),
    Name(Str),
}

/// Intermediate rule shape. Materialized into [`crate::rules::Rule`]
/// at the end via `Evaluator::build_rule`.
pub enum ARule {
    Blank,
    String(Str),
    Pattern(Str, Option<Str>),
    NamedSymbol(Str),
    SeqOrChoice(bool, ChildRange),
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
pub struct ValueId(pub u32);

#[derive(Clone, Copy, Debug)]
pub enum Value {
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

/// Pools holding the intermediate IR being built during lowering. Indices
/// (`RuleId`, `ValueId`, `Str`, `ChildRange`) are stable for the lifetime
/// of the parse call so cross-grammar caches like `LoweringState::let_values`
/// can keep referencing earlier-grammar entries.
#[derive(Default)]
pub struct IrPools {
    pub values: Vec<Value>,
    pub rules: Vec<ARule>,
    pub value_children: Vec<ValueId>,
    pub rule_children: Vec<RuleId>,
    pub object_pool: Vec<FxHashMap<String, ValueId>>,
    pub strings: StringPool,
}

const LOADED_MODULES_WORDS: usize = (u8::MAX as usize + 1) / u64::BITS as usize;

/// Tracks which modules have had their let bindings evaluated.
#[derive(Default)]
pub struct LoadedModules([u64; LOADED_MODULES_WORDS]);

impl LoadedModules {
    pub const fn is_loaded(&self, idx: ModuleId) -> bool {
        let i = idx as usize;
        self.0[i / 64] & (1 << (i % 64)) != 0
    }
    pub const fn set_loaded(&mut self, idx: ModuleId) {
        let i = idx as usize;
        self.0[i / 64] |= 1 << (i % 64);
    }
}
