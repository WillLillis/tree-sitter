use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
};

use rustc_hash::FxHashMap;

use crate::{
    grammars::{LexicalGrammar, ProdRef, ReservedWordSetId, SyntaxGrammar},
    rule_pool::{FStep, NO_RESERVED_WORDS, Prec},
    rules::{Associativity, Symbol, SymbolType, TokenSet},
};

/// The augmented start production: one step, the grammar's root nonterminal.
static START_STEPS: [FStep; 1] = [FStep {
    sym_index: 0,
    prec_val: 0,
    alias: 0,
    field: 0,
    reserved: NO_RESERVED_WORDS,
    flags: SymbolType::NonTerminal as u8,
}];

pub const START_PRODUCTION_ID: u32 = u32::MAX;

const fn start_production() -> ProdRef<'static> {
    ProdRef {
        steps: &START_STEPS,
        dynamic_precedence: 0,
    }
}

/// Precomputed identity keys for one `(production, dot)` pair. Dense ids
/// assigned in item-content order, so item comparisons are integer ops
/// instead of production-content walks (which compare strings).
///
/// `cmp` is the rank of the content tuple `Ord` compares (dynamic
/// precedence, length, precedence/associativity at the dot, then completed
/// steps' aliases and fields and remaining steps in full); equal ranks hold
/// exactly when that tuple compares equal, so it doubles as the equality
/// class for items without preceding inherited fields. `eq_with_syms`
/// subdivides `cmp` by the completed steps' symbols, which participate in
/// equality only when `has_preceding_inherited_fields` is set.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct DotKeys {
    pub cmp: u32,
    pub eq_with_syms: u32,
}

/// Identity keys for every `(production, dot)` in a grammar: the whole
/// pooled production table (grammar productions and inlining-created ones
/// alike) plus the augmented start production in slot 0.
pub struct ItemKeyMap {
    keys: Vec<Box<[DotKeys]>>,
    start: Box<[DotKeys]>,
}

impl ItemKeyMap {
    pub fn new(grammar: &SyntaxGrammar) -> Self {
        let prod = |slot: usize| -> ProdRef {
            if slot == 0 {
                start_production()
            } else {
                grammar.production(slot as u32 - 1)
            }
        };
        let slot_count = grammar.productions.len() + 1;

        let mut contents: Vec<(u32, u32)> = Vec::new();
        for pi in 0..slot_count {
            for dot in 0..=prod(pi).steps.len() {
                contents.push((pi as u32, dot as u32));
            }
        }
        let content = |&(pi, dot): &(u32, u32)| ItemContent {
            production: prod(pi as usize),
            dot: dot as usize,
        };
        contents.sort_unstable_by(|a, b| content(a).cmp(&content(b)));

        let mut slot_keys: Vec<Box<[DotKeys]>> = (0..slot_count)
            .map(|pi| vec![DotKeys::default(); prod(pi).steps.len() + 1].into_boxed_slice())
            .collect();
        let mut cmp_id = 0u32;
        let mut prev: Option<(u32, u32)> = None;
        for &(pi, dot) in &contents {
            if let Some(p) = prev
                && content(&p) != content(&(pi, dot))
            {
                cmp_id += 1;
            }
            slot_keys[pi as usize][dot as usize].cmp = cmp_id;
            prev = Some((pi, dot));
        }

        let mut sym_classes: FxHashMap<(u32, Vec<Symbol>), u32> = FxHashMap::default();
        for &(pi, dot) in &contents {
            let syms: Vec<Symbol> = prod(pi as usize).steps[..dot as usize]
                .iter()
                .map(|s| s.symbol())
                .collect();
            let next = sym_classes.len() as u32;
            let keys = &mut slot_keys[pi as usize][dot as usize];
            keys.eq_with_syms = *sym_classes.entry((keys.cmp, syms)).or_insert(next);
        }

        let mut slots = slot_keys.into_iter();
        let start = slots.next().unwrap();
        Self {
            keys: slots.collect(),
            start,
        }
    }

    /// The keys slice (indexed by dot) for a production of this grammar.
    pub fn keys_for(&self, id: u32) -> &[DotKeys] {
        if id == START_PRODUCTION_ID {
            &self.start
        } else {
            &self.keys[id as usize]
        }
    }

    pub fn start_keys(&self) -> &[DotKeys] {
        &self.start
    }
}

/// The content tuple that `ParseItem`'s `Ord` (and flagless `Eq`) observe,
/// evaluated on `(production, dot)` directly. Ranking the key universe by
/// this order is what makes the dense `cmp` ids order-preserving. Item
/// ordering only ever compares same-dot pairs; the dot comparison keeps
/// the order total across dots so ranks are well defined.
struct ItemContent<'a> {
    production: ProdRef<'a>,
    dot: usize,
}

impl ItemContent<'_> {
    fn prec(&self) -> Prec {
        if self.dot > 0 {
            self.production.steps[self.dot - 1].precedence()
        } else {
            Prec::None
        }
    }

    fn assoc(&self) -> Option<Associativity> {
        if self.dot > 0 {
            self.production.steps[self.dot - 1].associativity()
        } else {
            None
        }
    }
}

impl Ord for FStep {
    fn cmp(&self, other: &Self) -> Ordering {
        self.symbol()
            .cmp(&other.symbol())
            .then_with(|| self.precedence().cmp(&other.precedence()))
            .then_with(|| self.associativity().cmp(&other.associativity()))
            .then_with(|| self.alias().cmp(&other.alias()))
            .then_with(|| self.field().cmp(&other.field()))
            .then_with(|| self.reserved.cmp(&other.reserved))
    }
}

impl PartialOrd for FStep {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ItemContent<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.production
            .dynamic_precedence
            .cmp(&other.production.dynamic_precedence)
            .then_with(|| {
                self.production
                    .steps
                    .len()
                    .cmp(&other.production.steps.len())
            })
            .then_with(|| self.prec().cmp(&other.prec()))
            .then_with(|| self.assoc().cmp(&other.assoc()))
            .then_with(|| self.dot.cmp(&other.dot))
            .then_with(|| {
                let steps = self.production.steps.iter().zip(other.production.steps);
                for (i, (&sa, &sb)) in steps.enumerate() {
                    let o = if i < self.dot {
                        (sa.alias(), sa.field()).cmp(&(sb.alias(), sb.field()))
                    } else {
                        sa.cmp(&sb)
                    };
                    if o != Ordering::Equal {
                        return o;
                    }
                }
                Ordering::Equal
            })
    }
}

impl PartialOrd for ItemContent<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for ItemContent<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for ItemContent<'_> {}

/// A [`ParseItem`] represents an in-progress match of a single production in a grammar.
#[derive(Clone, Copy, Debug)]
pub struct ParseItem<'a> {
    /// The index of the parent rule within the grammar.
    pub variable_index: u32,
    /// The number of symbols that have already been matched.
    pub step_index: u32,
    /// Id of the production being matched, in the pooled production table;
    /// `START_PRODUCTION_ID` for the augmented start production, which lives
    /// outside the table.
    pub prod_id: u32,
    /// The production's identity keys, indexed by dot position. There is one
    /// key per dot, so `keys.len()` is the step count plus one.
    pub keys: &'a [DotKeys],
    /// A boolean indicating whether any of the already-matched children were
    /// hidden nodes and had fields. Ordinarily, a parse item's behavior is not
    /// affected by the symbols of its preceding children; it only needs to
    /// keep track of their fields and aliases.
    ///
    /// Take for example these two items:
    ///   X -> a b • c
    ///   X -> a g • c
    ///
    /// They can be considered equivalent, for the purposes of parse table
    /// generation, because they entail the same actions. But if this flag is
    /// true, then the item's set of inherited fields may depend on the specific
    /// symbols of its preceding children.
    pub has_preceding_inherited_fields: bool,
}

/// Interned lookahead set: an index into a [`LookaheadSetPool`]. Ids are
/// canonical, so id equality is exactly the underlying [`TokenSet`] equality.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LookaheadSetId(u32);

impl Default for LookaheadSetId {
    fn default() -> Self {
        LookaheadSetPool::EMPTY
    }
}

/// Interner for lookahead [`TokenSet`]s. Item-set entries store ids, so
/// entry hash/eq/clone are integer ops, and unions of the same two sets are
/// memoized instead of re-materialized per state.
pub struct LookaheadSetPool {
    sets: Vec<TokenSet>,
    ids: FxHashMap<TokenSet, LookaheadSetId>,
    union_memo: FxHashMap<(LookaheadSetId, LookaheadSetId), LookaheadSetId>,
    insert_memo: FxHashMap<(LookaheadSetId, Symbol), LookaheadSetId>,
}

impl LookaheadSetPool {
    pub const EMPTY: LookaheadSetId = LookaheadSetId(0);

    #[must_use]
    pub fn new() -> Self {
        let mut pool = Self {
            sets: Vec::new(),
            ids: FxHashMap::default(),
            union_memo: FxHashMap::default(),
            insert_memo: FxHashMap::default(),
        };
        let empty = pool.intern(TokenSet::new());
        debug_assert!(empty == Self::EMPTY);
        pool
    }

    #[must_use]
    pub fn get(&self, id: LookaheadSetId) -> &TokenSet {
        &self.sets[id.0 as usize]
    }

    pub fn intern(&mut self, set: TokenSet) -> LookaheadSetId {
        if let Some(&id) = self.ids.get(&set) {
            return id;
        }
        let id = LookaheadSetId(self.sets.len() as u32);
        self.sets.push(set.clone());
        self.ids.insert(set, id);
        id
    }

    pub fn intern_ref(&mut self, set: &TokenSet) -> LookaheadSetId {
        if let Some(&id) = self.ids.get(set) {
            return id;
        }
        self.intern(set.clone())
    }

    pub fn singleton(&mut self, symbol: Symbol) -> LookaheadSetId {
        self.insert(Self::EMPTY, symbol)
    }

    pub fn insert(&mut self, id: LookaheadSetId, symbol: Symbol) -> LookaheadSetId {
        if self.get(id).contains(&symbol) {
            return id;
        }
        if let Some(&result) = self.insert_memo.get(&(id, symbol)) {
            return result;
        }
        let mut set = self.get(id).clone();
        set.insert(symbol);
        let result = self.intern(set);
        self.insert_memo.insert((id, symbol), result);
        result
    }

    pub fn union(&mut self, left: LookaheadSetId, right: LookaheadSetId) -> LookaheadSetId {
        if left == right || right == Self::EMPTY {
            return left;
        }
        if left == Self::EMPTY {
            return right;
        }
        let key = (left.min(right), left.max(right));
        if let Some(&result) = self.union_memo.get(&key) {
            return result;
        }
        let mut set = self.get(key.0).clone();
        set.insert_all(self.get(key.1));
        let result = self.intern(set);
        self.union_memo.insert(key, result);
        result
    }
}

/// Represents a set of in-progress matches of productions in a grammar.
///
/// For each in-progress match, a set of "lookaheads" (tokens that are allowed to
/// *follow* the in-progress rule) are included. This object corresponds directly
/// to a state in the final parse table.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ParseItemSet<'a> {
    pub entries: Vec<ParseItemSetEntry<'a>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseItemSetEntry<'a> {
    pub item: ParseItem<'a>,
    pub lookaheads: LookaheadSetId,
    pub following_reserved_word_set: ReservedWordSetId,
}

/// A [`ParseItemSetCore`] is like a [`ParseItemSet`], but without the lookahead
/// information. Parse states with the same core are candidates for merging.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParseItemSetCore<'a> {
    pub entries: Vec<ParseItem<'a>>,
}

pub struct ParseItemDisplay<'a>(
    pub &'a ParseItem<'a>,
    pub &'a SyntaxGrammar,
    pub &'a LexicalGrammar,
);

pub struct TokenSetDisplay<'a>(
    pub &'a TokenSet,
    pub &'a SyntaxGrammar,
    pub &'a LexicalGrammar,
);

pub struct ParseItemSetDisplay<'a>(
    pub &'a ParseItemSet<'a>,
    pub &'a SyntaxGrammar,
    pub &'a LexicalGrammar,
    pub &'a LookaheadSetPool,
);

impl<'a> ParseItem<'a> {
    #[must_use]
    pub fn start(key_map: &'a ItemKeyMap) -> Self {
        ParseItem {
            variable_index: u32::MAX,
            prod_id: START_PRODUCTION_ID,
            keys: key_map.start_keys(),
            step_index: 0,
            has_preceding_inherited_fields: false,
        }
    }

    /// The production being matched.
    #[must_use]
    pub fn production<'g>(&self, grammar: &'g SyntaxGrammar) -> ProdRef<'g> {
        if self.prod_id == START_PRODUCTION_ID {
            start_production()
        } else {
            grammar.production(self.prod_id)
        }
    }

    #[must_use]
    pub fn step(&self, grammar: &SyntaxGrammar) -> Option<FStep> {
        self.production(grammar)
            .steps
            .get(self.step_index as usize)
            .copied()
    }

    #[must_use]
    pub fn symbol(&self, grammar: &SyntaxGrammar) -> Option<Symbol> {
        self.step(grammar).map(FStep::symbol)
    }

    #[must_use]
    pub fn associativity(&self, grammar: &SyntaxGrammar) -> Option<Associativity> {
        self.prev_step(grammar).and_then(FStep::associativity)
    }

    #[must_use]
    pub fn precedence(&self, grammar: &SyntaxGrammar) -> Prec {
        self.prev_step(grammar)
            .map_or(Prec::None, FStep::precedence)
    }

    #[must_use]
    pub fn prev_step(&self, grammar: &SyntaxGrammar) -> Option<FStep> {
        if self.step_index > 0 {
            Some(self.production(grammar).steps[self.step_index as usize - 1])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.step_index as usize + 1 == self.keys.len()
    }

    #[must_use]
    pub const fn is_augmented(&self) -> bool {
        self.variable_index == u32::MAX
    }

    /// Create an item like this one, but advanced by one step.
    #[must_use]
    pub const fn successor(&self) -> Self {
        ParseItem {
            variable_index: self.variable_index,
            prod_id: self.prod_id,
            keys: self.keys,
            step_index: self.step_index + 1,
            has_preceding_inherited_fields: self.has_preceding_inherited_fields,
        }
    }

    /// Create an item identical to this one, but with a different production.
    /// This is used when dynamically "inlining" certain symbols in a production.
    #[must_use]
    pub const fn substitute_production(&self, prod_id: u32, keys: &'a [DotKeys]) -> Self {
        let mut result = *self;
        result.prod_id = prod_id;
        result.keys = keys;
        result
    }

    /// This item's identity keys at the current dot.
    #[must_use]
    fn dot_keys(&self) -> DotKeys {
        self.keys[self.step_index as usize]
    }
}

impl<'a> ParseItemSet<'a> {
    #[inline]
    pub fn insert(&mut self, item: ParseItem<'a>) -> &mut ParseItemSetEntry<'a> {
        match self.entries.binary_search_by(|e| e.item.cmp(&item)) {
            Err(i) => {
                self.entries.insert(
                    i,
                    ParseItemSetEntry {
                        item,
                        lookaheads: LookaheadSetPool::EMPTY,
                        following_reserved_word_set: ReservedWordSetId::default(),
                    },
                );
                &mut self.entries[i]
            }
            Ok(i) => &mut self.entries[i],
        }
    }

    #[must_use]
    pub fn core(&self) -> ParseItemSetCore<'a> {
        ParseItemSetCore {
            entries: self.entries.iter().map(|e| e.item).collect(),
        }
    }
}

/// `Precedence`'s display, resolved from the pooled form.
pub fn prec_display(grammar: &SyntaxGrammar, prec: Prec) -> String {
    match prec {
        Prec::None => "none".to_string(),
        Prec::Integer(i) => i.to_string(),
        Prec::Name(sid) => format!("'{}'", grammar.resolve(sid)),
    }
}

impl fmt::Display for ParseItemDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        if self.0.is_augmented() {
            write!(f, "START →")?;
        } else {
            write!(
                f,
                "{} →",
                self.1.variables[self.0.variable_index as usize].name
            )?;
        }

        let production = self.0.production(self.1);
        for (i, step) in production.steps.iter().enumerate() {
            let symbol = step.symbol();
            if i == self.0.step_index as usize {
                write!(f, " •")?;
                if !matches!(step.precedence(), Prec::None)
                    || step.associativity().is_some()
                    || step.reserved != 0
                {
                    write!(f, " (")?;
                    if !matches!(step.precedence(), Prec::None) {
                        write!(f, " {}", prec_display(self.1, step.precedence()))?;
                    }
                    if let Some(associativity) = step.associativity() {
                        write!(f, " {associativity:?}")?;
                    }
                    if step.reserved != 0 {
                        write!(f, "reserved: {}", step.reserved)?;
                    }
                    write!(f, " )")?;
                }
            }

            write!(f, " ")?;
            if symbol.is_terminal() {
                if let Some(variable) = self.2.variables.get(symbol.index) {
                    write!(f, "{}", variable.name)?;
                } else {
                    write!(f, "terminal-{}", symbol.index)?;
                }
            } else if symbol.is_external() {
                write!(f, "{}", self.1.external_tokens[symbol.index].name)?;
            } else {
                write!(f, "{}", self.1.variables[symbol.index].name)?;
            }

            if let Some(alias) = step.alias() {
                write!(f, "@{}", self.1.resolve(alias.value))?;
            }
        }

        if self.0.is_done() {
            write!(f, " •")?;
            if let Some(&step) = production.steps.last() {
                if let Some(associativity) = step.associativity() {
                    if matches!(step.precedence(), Prec::None) {
                        write!(f, " ({associativity:?})")?;
                    } else {
                        write!(
                            f,
                            " ({} {associativity:?})",
                            prec_display(self.1, step.precedence())
                        )?;
                    }
                } else if !matches!(step.precedence(), Prec::None) {
                    write!(f, " ({})", prec_display(self.1, step.precedence()))?;
                }
            }
        }

        Ok(())
    }
}

const fn escape_invisible(c: char) -> Option<&'static str> {
    Some(match c {
        '\n' => "\\n",
        '\r' => "\\r",
        '\t' => "\\t",
        '\0' => "\\0",
        '\\' => "\\\\",
        '\x0b' => "\\v",
        '\x0c' => "\\f",
        _ => return None,
    })
}

fn display_variable_name(source: &str) -> String {
    source
        .chars()
        .fold(String::with_capacity(source.len()), |mut acc, c| {
            if let Some(esc) = escape_invisible(c) {
                acc.push_str(esc);
            } else {
                acc.push(c);
            }
            acc
        })
}

impl fmt::Display for TokenSetDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(f, "[")?;
        for (i, symbol) in self.0.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }

            if symbol.is_terminal() {
                if let Some(variable) = self.2.variables.get(symbol.index) {
                    write!(f, "{}", display_variable_name(&variable.name))?;
                } else {
                    write!(f, "terminal-{}", symbol.index)?;
                }
            } else if symbol.is_external() {
                write!(f, "{}", self.1.external_tokens[symbol.index].name)?;
            } else {
                write!(f, "{}", self.1.variables[symbol.index].name)?;
            }
        }
        write!(f, "]")?;
        Ok(())
    }
}

impl fmt::Display for ParseItemSetDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for entry in &self.0.entries {
            write!(
                f,
                "{}\t{}",
                ParseItemDisplay(&entry.item, self.1, self.2),
                TokenSetDisplay(self.3.get(entry.lookaheads), self.1, self.2),
            )?;
            if entry.following_reserved_word_set != ReservedWordSetId::default() {
                write!(
                    f,
                    "\treserved word set: {}",
                    entry.following_reserved_word_set
                )?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Hash for ParseItem<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        hasher.write_u32(self.variable_index);
        hasher.write_u32(self.step_index);
        // The already-matched children don't play any role in the parse state
        // for this item unless they have fields or aliases, or are hidden
        // rules that have fields (see `has_preceding_inherited_fields`); the
        // keys encode exactly that, with the preceding symbols participating
        // only in `eq_with_syms`.
        let keys = self.dot_keys();
        if self.has_preceding_inherited_fields {
            hasher.write_u8(1);
            hasher.write_u32(keys.eq_with_syms);
        } else {
            hasher.write_u8(0);
            hasher.write_u32(keys.cmp);
        }
    }
}

impl PartialEq for ParseItem<'_> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        if self.variable_index != other.variable_index
            || self.step_index != other.step_index
            || self.has_preceding_inherited_fields != other.has_preceding_inherited_fields
        {
            return false;
        }
        if self.has_preceding_inherited_fields {
            self.dot_keys().eq_with_syms == other.dot_keys().eq_with_syms
        } else {
            self.dot_keys().cmp == other.dot_keys().cmp
        }
    }
}

impl Ord for ParseItem<'_> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.step_index
            .cmp(&other.step_index)
            .then_with(|| self.variable_index.cmp(&other.variable_index))
            .then_with(|| self.dot_keys().cmp.cmp(&other.dot_keys().cmp))
    }
}

impl PartialOrd for ParseItem<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Eq for ParseItem<'_> {}

impl Hash for ParseItemSet<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        hasher.write_usize(self.entries.len());
        for entry in &self.entries {
            entry.item.hash(hasher);
            entry.lookaheads.hash(hasher);
            entry.following_reserved_word_set.hash(hasher);
        }
    }
}

impl Hash for ParseItemSetCore<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        hasher.write_usize(self.entries.len());
        for item in &self.entries {
            item.hash(hasher);
        }
    }
}
