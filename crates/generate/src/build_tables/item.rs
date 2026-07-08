use std::{
    cmp::Ordering,
    fmt,
    hash::{Hash, Hasher},
    rc::Rc,
};

use rustc_hash::FxHashMap;

use crate::{
    grammars::{LexicalGrammar, ProdRef, ReservedWordSetId, SyntaxGrammar},
    rule_pool::{FStep, NO_RESERVED_WORDS, Prec, StrId},
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
        id: START_PRODUCTION_ID,
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
            strs: &grammar.strs,
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
/// this order is what makes the dense `cmp` ids order-preserving; every
/// string-bearing field resolves and compares as a string here so the ranks
/// reproduce the historical `String`-ordered comparisons exactly. Item
/// ordering only ever compares same-dot pairs; the dot comparison keeps the
/// order total across dots so ranks are well defined.
struct ItemContent<'a> {
    production: ProdRef<'a>,
    dot: usize,
    strs: &'a [Rc<str>],
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

    fn resolve(&self, id: StrId) -> &str {
        &self.strs[id.index()]
    }

    /// `Precedence` order: `None`, then `Integer`, then `Name` by string.
    fn prec_cmp(&self, a: Prec, b: Prec) -> Ordering {
        match (a, b) {
            (Prec::None, Prec::None) => Ordering::Equal,
            (Prec::Integer(x), Prec::Integer(y)) => x.cmp(&y),
            (Prec::Name(x), Prec::Name(y)) => self.resolve(x).cmp(self.resolve(y)),
            (Prec::None, _) | (Prec::Integer(_), Prec::Name(_)) => Ordering::Less,
            (_, Prec::None) | (Prec::Name(_), Prec::Integer(_)) => Ordering::Greater,
        }
    }

    /// `Option<Alias>` order: `None` first, then value string, then
    /// `is_named`.
    fn alias_cmp(&self, a: FStep, b: FStep) -> Ordering {
        let key = |s: FStep| s.alias().map(|a| (self.resolve(a.value), a.is_named));
        key(a).cmp(&key(b))
    }

    fn field_cmp(&self, a: FStep, b: FStep) -> Ordering {
        let key = |s: FStep| s.field().map(|f| self.resolve(f));
        key(a).cmp(&key(b))
    }

    /// `ProductionStep` order: symbol, precedence, associativity, alias,
    /// field, reserved word set.
    fn step_cmp(&self, a: FStep, b: FStep) -> Ordering {
        a.symbol()
            .cmp(&b.symbol())
            .then_with(|| self.prec_cmp(a.precedence(), b.precedence()))
            .then_with(|| a.associativity().cmp(&b.associativity()))
            .then_with(|| self.alias_cmp(a, b))
            .then_with(|| self.field_cmp(a, b))
            .then_with(|| a.reserved.cmp(&b.reserved))
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
            .then_with(|| self.prec_cmp(self.prec(), other.prec()))
            .then_with(|| self.assoc().cmp(&other.assoc()))
            .then_with(|| self.dot.cmp(&other.dot))
            .then_with(|| {
                let steps = self.production.steps.iter().zip(other.production.steps);
                for (i, (&sa, &sb)) in steps.enumerate() {
                    let o = if i < self.dot {
                        self.alias_cmp(sa, sb).then_with(|| self.field_cmp(sa, sb))
                    } else {
                        self.step_cmp(sa, sb)
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
    /// The production being matched.
    pub production: ProdRef<'a>,
    /// The production's identity keys, indexed by `step_index`.
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
    pub lookaheads: TokenSet,
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
);

impl<'a> ParseItem<'a> {
    #[must_use]
    pub fn start(key_map: &'a ItemKeyMap) -> Self {
        ParseItem {
            variable_index: u32::MAX,
            production: start_production(),
            keys: key_map.start_keys(),
            step_index: 0,
            has_preceding_inherited_fields: false,
        }
    }

    #[must_use]
    pub fn step(&self) -> Option<FStep> {
        self.production.steps.get(self.step_index as usize).copied()
    }

    #[must_use]
    pub fn symbol(&self) -> Option<Symbol> {
        self.step().map(FStep::symbol)
    }

    #[must_use]
    pub fn associativity(&self) -> Option<Associativity> {
        self.prev_step().and_then(FStep::associativity)
    }

    #[must_use]
    pub fn precedence(&self) -> Prec {
        self.prev_step().map_or(Prec::None, FStep::precedence)
    }

    #[must_use]
    pub fn prev_step(&self) -> Option<FStep> {
        if self.step_index > 0 {
            Some(self.production.steps[self.step_index as usize - 1])
        } else {
            None
        }
    }

    #[must_use]
    pub const fn is_done(&self) -> bool {
        self.step_index as usize == self.production.steps.len()
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
            production: self.production,
            keys: self.keys,
            step_index: self.step_index + 1,
            has_preceding_inherited_fields: self.has_preceding_inherited_fields,
        }
    }

    /// Create an item identical to this one, but with a different production.
    /// This is used when dynamically "inlining" certain symbols in a production.
    #[must_use]
    pub const fn substitute_production(
        &self,
        production: ProdRef<'a>,
        keys: &'a [DotKeys],
    ) -> Self {
        let mut result = *self;
        result.production = production;
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
                        lookaheads: TokenSet::new(),
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

        for (i, step) in self.0.production.steps.iter().enumerate() {
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
            if let Some(&step) = self.0.production.steps.last() {
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
                TokenSetDisplay(&entry.lookaheads, self.1, self.2),
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

// TEMP SPIKE: identity-op counters (remove with the other spike blocks).
pub static ITEM_HASHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ITEM_EQS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
pub static ITEM_CMPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Hash for ParseItem<'_> {
    fn hash<H: Hasher>(&self, hasher: &mut H) {
        ITEM_HASHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        ITEM_EQS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
        ITEM_CMPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
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
