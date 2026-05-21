//! Intermediate representation used during lowering.

use rustc_hash::FxHashMap;

use super::super::ModuleId;
use super::super::ast::ChildRange;
use super::super::string_pool::Str;

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
