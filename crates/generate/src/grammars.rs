use std::{collections::BTreeMap, fmt};

use rustc_hash::FxHashMap;

use crate::{
    node_types::ChildType,
    rule_pool::{FProd, FStep, PoolPrecedenceEntry, StrId, StrPool},
};

use super::{
    nfa::Nfa,
    rules::{Alias, Associativity, Precedence, Rule, Symbol, TokenSet},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum VariableType {
    Hidden,
    Auxiliary,
    Anonymous,
    Named,
}

// Input grammar

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Variable {
    pub name: String,
    pub kind: VariableType,
    pub rule: Rule,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum PrecedenceEntry {
    Name(String),
    Symbol(String),
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct InputGrammar {
    pub name: String,
    pub variables: Vec<Variable>,
    pub extra_symbols: Vec<Rule>,
    pub expected_conflicts: Vec<Vec<String>>,
    pub precedence_orderings: Vec<Vec<PrecedenceEntry>>,
    pub external_tokens: Vec<Rule>,
    pub variables_to_inline: Vec<String>,
    pub supertype_symbols: Vec<String>,
    pub word_token: Option<String>,
    pub reserved_words: Vec<ReservedWordContext<Rule>>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct ReservedWordContext<T> {
    pub name: String,
    pub reserved_words: Vec<T>,
}

// Extracted lexical grammar

#[derive(Debug, PartialEq, Eq)]
pub struct LexicalVariable {
    pub name: String,
    pub kind: VariableType,
    pub implicit_precedence: i32,
    pub start_state: u32,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct LexicalGrammar {
    pub nfa: Nfa,
    pub variables: Vec<LexicalVariable>,
}

// Extracted syntax grammar

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProductionStep {
    pub symbol: Symbol,
    pub precedence: Precedence,
    pub associativity: Option<Associativity>,
    pub alias: Option<Alias>,
    pub field_name: Option<String>,
    pub reserved_word_set_id: ReservedWordSetId,
}

impl ProductionStep {
    pub fn child_type(&self, default_aliases: &BTreeMap<Symbol, Alias>) -> ChildType {
        if let Some(alias) = &self.alias {
            ChildType::Aliased(alias.clone())
        } else if let Some(alias) = default_aliases.get(&self.symbol) {
            ChildType::Aliased(alias.clone())
        } else {
            ChildType::Normal(self.symbol)
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ReservedWordSetId(pub usize);

impl fmt::Display for ReservedWordSetId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}

pub const NO_RESERVED_WORDS: ReservedWordSetId = ReservedWordSetId(usize::MAX);

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Production {
    pub steps: Vec<ProductionStep>,
    pub dynamic_precedence: i32,
}

#[derive(Debug, Default)]
pub struct InlinedProductionMap {
    pub map: FxHashMap<(u32, u32), Vec<u32>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxVariable {
    pub name: String,
    pub kind: VariableType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExternalToken {
    pub name: String,
    pub kind: VariableType,
    pub corresponding_internal_token: Option<Symbol>,
}

#[derive(Debug, Default)]
pub struct SyntaxGrammar {
    pub variables: Vec<SyntaxVariable>,
    pub extra_symbols: Vec<Symbol>,
    pub expected_conflicts: Vec<Vec<Symbol>>,
    pub external_tokens: Vec<ExternalToken>,
    pub supertype_symbols: Vec<Symbol>,
    pub variables_to_inline: Vec<Symbol>,
    pub word_token: Option<Symbol>,
    pub precedence_orderings: Vec<Vec<PoolPrecedenceEntry>>,
    pub reserved_word_sets: Vec<TokenSet>,

    pub steps: Vec<FStep>,
    pub productions: Vec<FProd>,
    pub var_prods: Vec<(u32, u32)>,
    pub interner: StrPool,
}

impl SyntaxGrammar {
    // NOTE: Still need to settle where the string pool lives
    #[must_use]
    pub fn resolve(&self, id: StrId) -> &str {
        self.interner.resolve(id)
    }

    #[must_use]
    pub fn production(&self, id: u32) -> ProdRef<'_> {
        let p = self.productions[id as usize];
        ProdRef {
            steps: &self.steps[p.step_range()],
            dynamic_precedence: p.dynamic_prec,
        }
    }

    /// The pooled production ids belonging to a variable
    #[must_use]
    pub fn variable_prod_ids(&self, variable_index: usize) -> std::ops::Range<u32> {
        let (start, end) = self.var_prods[variable_index];
        start..end
    }
}

/// A production in the pooled [`SyntaxGrammar`] storage
#[derive(Clone, Copy, Debug)]
pub struct ProdRef<'a> {
    pub steps: &'a [FStep],
    pub dynamic_precedence: i32,
}

impl ProdRef<'_> {
    #[must_use]
    pub fn first_symbol(&self) -> Option<Symbol> {
        self.steps.first().map(|s| s.symbol())
    }
}

#[cfg(test)]
impl ProductionStep {
    #[must_use]
    pub fn new(symbol: Symbol) -> Self {
        Self {
            symbol,
            precedence: Precedence::None,
            associativity: None,
            alias: None,
            field_name: None,
            reserved_word_set_id: ReservedWordSetId::default(),
        }
    }

    #[must_use]
    pub fn with_prec(
        mut self,
        precedence: Precedence,
        associativity: Option<Associativity>,
    ) -> Self {
        self.precedence = precedence;
        self.associativity = associativity;
        self
    }

    #[must_use]
    pub fn with_alias(mut self, value: &str, is_named: bool) -> Self {
        self.alias = Some(Alias {
            value: value.to_string(),
            is_named,
        });
        self
    }

    #[must_use]
    pub fn with_field_name(mut self, name: &str) -> Self {
        self.field_name = Some(name.to_string());
        self
    }
}

impl Production {
    #[must_use]
    pub fn first_symbol(&self) -> Option<Symbol> {
        self.steps.first().map(|s| s.symbol)
    }
}

#[cfg(test)]
impl Variable {
    #[must_use]
    pub fn named(name: &str, rule: Rule) -> Self {
        Self {
            name: name.to_string(),
            kind: VariableType::Named,
            rule,
        }
    }

    #[must_use]
    pub fn auxiliary(name: &str, rule: Rule) -> Self {
        Self {
            name: name.to_string(),
            kind: VariableType::Auxiliary,
            rule,
        }
    }

    #[must_use]
    pub fn hidden(name: &str, rule: Rule) -> Self {
        Self {
            name: name.to_string(),
            kind: VariableType::Hidden,
            rule,
        }
    }

    #[must_use]
    pub fn anonymous(name: &str, rule: Rule) -> Self {
        Self {
            name: name.to_string(),
            kind: VariableType::Anonymous,
            rule,
        }
    }
}

impl VariableType {
    #[must_use]
    pub fn is_visible(self) -> bool {
        self == Self::Named || self == Self::Anonymous
    }
}

impl LexicalGrammar {
    pub fn variable_indices_for_nfa_states<'a>(
        &'a self,
        state_ids: &'a [u32],
    ) -> impl Iterator<Item = usize> + 'a {
        let mut prev = None;
        state_ids.iter().filter_map(move |state_id| {
            let variable_id = self.variable_index_for_nfa_state(*state_id);
            if prev == Some(variable_id) {
                None
            } else {
                prev = Some(variable_id);
                prev
            }
        })
    }

    #[must_use]
    pub fn variable_index_for_nfa_state(&self, state_id: u32) -> usize {
        // The NFA is built in reverse (accept state first, entry state last), so
        // each variable's `start_state` is the last (highest) NFA state allocated
        // for it. These values are monotonically increasing, so we can binary
        // search for the first variable whose start_state >= state_id.
        self.variables.partition_point(|v| v.start_state < state_id)
    }
}

impl SyntaxVariable {
    #[must_use]
    pub fn is_auxiliary(&self) -> bool {
        self.kind == VariableType::Auxiliary
    }

    #[must_use]
    pub fn is_hidden(&self) -> bool {
        self.kind == VariableType::Hidden || self.kind == VariableType::Auxiliary
    }
}

impl InlinedProductionMap {
    #[must_use]
    pub fn inlined_prod_ids<'a>(&'a self, prod_id: u32, step_index: u32) -> Option<&[u32]> {
        self.map.get(&(prod_id, step_index)).map(Vec::as_slice)
    }
}

impl fmt::Display for PrecedenceEntry {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Name(n) => write!(f, "'{n}'"),
            Self::Symbol(s) => write!(f, "$.{s}"),
        }
    }
}
