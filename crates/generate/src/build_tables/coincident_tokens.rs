use std::fmt;

use crate::{
    grammars::LexicalGrammar,
    rules::Symbol,
    tables::{ParseStateId, ParseTable},
};

pub struct CoincidentTokenIndex<'a> {
    entries: Vec<Vec<ParseStateId>>,
    /// Flat bitset for fast `contains()` checks. Indexed as `a * n + b`
    /// (both `(a,b)` and `(b,a)` bits are set, so no min/max normalization needed).
    contains_bits: Vec<u64>,
    grammar: &'a LexicalGrammar,
    n: usize,
}

impl<'a> CoincidentTokenIndex<'a> {
    pub fn new(table: &ParseTable, lexical_grammar: &'a LexicalGrammar) -> Self {
        let n = lexical_grammar.variables.len();
        let mut result = Self {
            n,
            grammar: lexical_grammar,
            entries: vec![Vec::new(); n * n],
            contains_bits: vec![0u64; (n * n).div_ceil(64)],
        };
        let mut terminal_indices = Vec::new();
        for (i, state) in table.states.iter().enumerate() {
            terminal_indices.clear();
            terminal_indices.extend(
                state
                    .terminal_entries
                    .keys()
                    .filter(|s| s.is_terminal())
                    .map(|s| s.index),
            );
            for (j, &a) in terminal_indices.iter().enumerate() {
                for &b in &terminal_indices[j..] {
                    let index = result.index(a, b);
                    if result.entries[index].last().copied() != Some(i) {
                        result.entries[index].push(i);
                    }
                    // Set both (a,b) and (b,a) bits so `contains()` needs
                    // no min/max normalization.
                    let ab = a * n + b;
                    result.contains_bits[ab / 64] |= 1u64 << (ab % 64);
                    let ba = b * n + a;
                    result.contains_bits[ba / 64] |= 1u64 << (ba % 64);
                }
            }
        }
        result
    }

    pub fn states_with(&self, a: Symbol, b: Symbol) -> &[ParseStateId] {
        &self.entries[self.index(a.index, b.index)]
    }

    pub fn contains(&self, a: Symbol, b: Symbol) -> bool {
        let bit_index = a.index * self.n + b.index;
        self.contains_bits[bit_index / 64] & (1u64 << (bit_index % 64)) != 0
    }

    #[must_use]
    const fn index(&self, a: usize, b: usize) -> usize {
        if a < b {
            a * self.n + b
        } else {
            b * self.n + a
        }
    }
}

impl fmt::Debug for CoincidentTokenIndex<'_> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "CoincidentTokenIndex {{")?;

        writeln!(f, "  entries: {{")?;
        for i in 0..self.n {
            writeln!(f, "    {}: {{", self.grammar.variables[i].name)?;
            for j in 0..self.n {
                writeln!(
                    f,
                    "      {}: {:?},",
                    self.grammar.variables[j].name,
                    self.entries[self.index(i, j)].len()
                )?;
            }
            writeln!(f, "    }},")?;
        }
        write!(f, "  }},")?;
        write!(f, "}}")?;
        Ok(())
    }
}
