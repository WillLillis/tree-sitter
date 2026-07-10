use std::fmt;

use crate::{grammars::LexicalGrammar, rules::Symbol, tables::ParseTable};

pub struct CoincidentTokenIndex<'a> {
    /// Flat bitset for fast [`contains()`](Self::contains) checks. Indexed as `a * n + b`
    /// (both `(a,b)` and `(b,a)` bits are set, so no min/max normalization needed).
    contains_bits: Vec<u64>,
    /// Same shape as `contains_bits`: bit `(a, b)` is set iff tokens `a` and
    /// `b` are coincident in some parse state where the grammar's word token
    /// is NOT a valid lookahead. Lets keyword identification ask "do all
    /// states containing this pair also contain the word token?" without
    /// storing per-pair state lists.
    without_word_bits: Vec<u64>,
    /// Word-aligned per-row bitsets for vectorized intersection checks.
    /// Row `a` spans `[a * row_words .. (a+1) * row_words]`.
    /// Bit `b` is set iff tokens `a` and `b` are coincident in some parse state.
    pub(crate) row_bits: Vec<u64>,
    grammar: &'a LexicalGrammar,
    n: usize,
}

impl<'a> CoincidentTokenIndex<'a> {
    #[must_use]
    pub fn new(
        table: &ParseTable,
        lexical_grammar: &'a LexicalGrammar,
        word_token: Option<Symbol>,
    ) -> Self {
        let n = lexical_grammar.variables.len();
        let row_words = n.div_ceil(64);
        let mut result = Self {
            n,
            grammar: lexical_grammar,
            contains_bits: vec![0u64; (n * n).div_ceil(64)],
            without_word_bits: vec![0u64; (n * n).div_ceil(64)],
            row_bits: vec![0u64; n * row_words],
        };
        // Pre-collect terminal indices up front rather than continuously recomputing within the
        // loop below.
        let mut terminal_indices = Vec::new();
        for state in &table.states {
            terminal_indices.clear();
            terminal_indices.extend(
                state
                    .terminal_entries
                    .keys()
                    .filter(|s| s.is_terminal())
                    .map(|s| s.index),
            );
            let has_word = word_token.is_some_and(|w| state.terminal_entries.contains_key(&w));
            for (j, &a) in terminal_indices.iter().enumerate() {
                for &b in &terminal_indices[j..] {
                    // Set both (a,b) and (b,a) bits so lookups need no
                    // min/max normalization.
                    let ab = a * n + b;
                    let ba = b * n + a;
                    result.contains_bits[ab / 64] |= 1u64 << (ab % 64);
                    result.contains_bits[ba / 64] |= 1u64 << (ba % 64);
                    if !has_word {
                        result.without_word_bits[ab / 64] |= 1u64 << (ab % 64);
                        result.without_word_bits[ba / 64] |= 1u64 << (ba % 64);
                    }
                    // Also populate the word-aligned row bitsets.
                    result.row_bits[a * row_words + b / 64] |= 1u64 << (b % 64);
                    result.row_bits[b * row_words + a / 64] |= 1u64 << (a % 64);
                }
            }
        }
        result
    }

    /// True iff every parse state where `a` and `b` are both valid lookaheads
    /// also has the word token as a valid lookahead (vacuously true when the
    /// tokens never coincide).
    #[must_use]
    pub fn all_coincident_states_have_word(&self, a: Symbol, b: Symbol) -> bool {
        let bit_index = a.index * self.n + b.index;
        self.without_word_bits[bit_index / 64] & (1u64 << (bit_index % 64)) == 0
    }

    #[must_use]
    pub fn contains(&self, a: Symbol, b: Symbol) -> bool {
        let bit_index = a.index * self.n + b.index;
        self.contains_bits[bit_index / 64] & (1u64 << (bit_index % 64)) != 0
    }
}

impl fmt::Debug for CoincidentTokenIndex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "CoincidentTokenIndex {{")?;
        for i in 0..self.n {
            let mut coincident = Vec::new();
            for j in 0..self.n {
                if self.contains(Symbol::terminal(i), Symbol::terminal(j)) {
                    coincident.push(&self.grammar.variables[j].name);
                }
            }
            if !coincident.is_empty() {
                writeln!(f, "  {}: {:?},", self.grammar.variables[i].name, coincident)?;
            }
        }
        write!(f, "}}")?;
        Ok(())
    }
}
