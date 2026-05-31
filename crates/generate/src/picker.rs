//! Plan B picker: per-state representation choice between Dense, CSR, and
//! Small tiers for the generated parse table.
//!
//! The Dense decision is deferred to master's existing heuristic via the
//! `large_state_count` input - the picker does not promote states out of
//! Dense or move Dense states elsewhere. For non-Dense states, it picks
//! between CSR and Small per state by minimum byte cost, then runs a
//! dedup-promotion pass that moves CSR rows back to Small when many
//! states share a canonical body and one shared body + N map entries
//! beats N independent CSR rows.
//!
//! Earlier picker iterations added cost-model tunables (PICKER_LAMBDA /
//! DEDUP_LAMBDA) that weighted estimated lookup cycles against bytes.
//! A static corpus sweep confirmed that any non-zero value either
//! changed picks for ~90% of grammars (PICKER_LAMBDA) or ~25% of
//! grammars (DEDUP_LAMBDA) but with no aggregate size benefit and no
//! empirical evidence that the resulting tier shifts actually improved
//! parse-time. The simpler bytes-only picker matches or beats master on
//! all 126 validated grammars; the knobs were removed to keep the
//! shipped surface small.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::tables::{GotoAction, ParseTable, ParseTableEntry};

/// Which storage tier the picker assigned to a state.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Repr {
    /// `parse_table[state * sym_count + symbol]` 2D array. O(1) lookup.
    Dense,
    /// Sparse row of (sym, val) pairs. O(log nnz) binary search.
    Csr,
    /// Grouped sparse rep with deduplicated bodies. O(group_count + nnz)
    /// linear scan but heavily shared across states.
    Small,
}

/// Per-state byte cost estimate for each candidate tier.
///
///   * Dense: no per-state overhead. The state slots into the
///     `parse_table[LARGE_STATE_COUNT][SYMBOL_COUNT]` matrix.
///   * Csr:   one `row_offsets` u32 entry (4 bytes) plus
///     `nnz * 4` bytes for the interleaved (sym, val) pairs.
///   * Small: the body bytes (`4 * group_count + 2 * nnz`) plus the
///     `group_count` u16 header (2 bytes) plus the
///     `small_parse_table_map` u32 entry (4 bytes).
#[derive(Copy, Clone, Debug)]
pub struct StateCosts {
    pub dense_bytes: usize,
    pub csr_bytes: usize,
    pub small_bytes: usize,
}

/// Compute byte-cost estimates for each tier per state.
pub fn compute_standalone_costs(parse_table: &ParseTable) -> Vec<StateCosts> {
    let symbol_count = parse_table.symbols.len();
    let mut term_groups: FxHashSet<&ParseTableEntry> = FxHashSet::default();
    let mut nonterm_groups: FxHashSet<usize> = FxHashSet::default();
    let mut out = Vec::with_capacity(parse_table.states.len());

    for (state_idx, state) in parse_table.states.iter().enumerate() {
        let nnz = state.terminal_entries.len() + state.nonterminal_entries.len();
        term_groups.clear();
        nonterm_groups.clear();
        for entry in state.terminal_entries.values() {
            term_groups.insert(entry);
        }
        for action in state.nonterminal_entries.values() {
            let id = match action {
                GotoAction::Goto(s) => *s,
                GotoAction::ShiftExtra => state_idx,
            };
            nonterm_groups.insert(id);
        }
        let group_count = term_groups.len() + nonterm_groups.len();

        out.push(StateCosts {
            dense_bytes: symbol_count * 2,
            csr_bytes: nnz * 4 + 4,
            small_bytes: 2 * nnz + 4 * group_count + 6,
        });
    }
    out
}

/// Decide a tier per state.
///
/// State ids `0..large_state_count` are forced to Dense (this is master's
/// per-state heuristic - pinning to it guarantees no parse-time
/// regression on the hot path). For state ids >= `large_state_count`,
/// the picker chooses between CSR and Small by argmin of byte cost.
///
/// A dedup-promotion pass then groups non-Dense states by their canonical
/// small-body hash and demotes a whole group to Small when one shared
/// body plus N map entries beats N independent CSR rows. Dense states
/// never participate; they keep their O(1) lookup guarantee.
pub fn decide_state_representations(
    parse_table: &ParseTable,
    standalone_costs: &[StateCosts],
    large_state_count: usize,
) -> Vec<Repr> {
    let n = parse_table.states.len();
    let mut picks = Vec::with_capacity(n);

    for (i, c) in standalone_costs.iter().enumerate() {
        let pick = if i < large_state_count {
            Repr::Dense
        } else if c.csr_bytes <= c.small_bytes {
            Repr::Csr
        } else {
            Repr::Small
        };
        picks.push(pick);
    }

    // Group non-Dense states by canonical-small hash. FxHasher 64-bit
    // collisions across O(10K) states are negligible; a false dedup
    // would slightly mis-estimate cost but not produce incorrect picks
    // since we still verify the promotion is locally cheaper.
    let mut by_hash: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    for i in 0..n {
        by_hash
            .entry(canonical_small_hash(parse_table, i))
            .or_default()
            .push(i);
    }

    for (_, mut group) in by_hash.into_iter().filter(|(_, g)| g.len() > 1) {
        // Dense states stay Dense.
        group.retain(|&i| picks[i] != Repr::Dense);
        if group.len() < 2 {
            continue;
        }
        // Sum each member's current bytes (whatever tier the per-state
        // pick chose) and compare against the promoted layout: one shared
        // Small body plus (N - 1) u32 map entries for the duplicates.
        let current_bytes: usize = group
            .iter()
            .map(|&i| match picks[i] {
                Repr::Csr => standalone_costs[i].csr_bytes,
                Repr::Small => standalone_costs[i].small_bytes,
                Repr::Dense => unreachable!(),
            })
            .sum();
        let canonical = &standalone_costs[group[0]];
        let promoted_bytes = canonical.small_bytes + 4 * (group.len() - 1);
        if promoted_bytes <= current_bytes {
            for &i in &group {
                picks[i] = Repr::Small;
            }
        }
    }

    picks
}

/// Hash the canonical small-table representation of a state. Two states
/// with equal hashes are presumed to emit byte-identical small entries.
pub fn canonical_small_hash(parse_table: &ParseTable, state_idx: usize) -> u64 {
    let state = &parse_table.states[state_idx];

    // Group symbols by action. Terminals key on `&ParseTableEntry` (Eq
    // distinguishes distinct action sequences); nonterminals key on
    // resolved goto state ID. Renumbering preserves equality (both
    // states remap goto values the same way), so original IDs are fine.
    let mut term_groups: FxHashMap<&ParseTableEntry, Vec<(u8, u32)>> = FxHashMap::default();
    let mut nonterm_groups: FxHashMap<usize, Vec<(u8, u32)>> = FxHashMap::default();

    for (sym, entry) in &state.terminal_entries {
        term_groups
            .entry(entry)
            .or_default()
            .push((sym.kind as u8, sym.index as u32));
    }
    for (sym, action) in &state.nonterminal_entries {
        let id = match action {
            GotoAction::Goto(s) => *s,
            GotoAction::ShiftExtra => state_idx,
        };
        nonterm_groups
            .entry(id)
            .or_default()
            .push((sym.kind as u8, sym.index as u32));
    }

    for v in term_groups.values_mut() {
        v.sort_unstable();
    }
    for v in nonterm_groups.values_mut() {
        v.sort_unstable();
    }

    // Sort groups deterministically. Each group's symbol set is
    // disjoint from every other group's (a symbol belongs to exactly
    // one action group), so symbol-sequence ordering is total.
    let mut term_vec: Vec<_> = term_groups.into_iter().collect();
    let mut nonterm_vec: Vec<_> = nonterm_groups.into_iter().collect();
    term_vec.sort_by(|a, b| a.1.cmp(&b.1));
    nonterm_vec.sort_by(|a, b| a.1.cmp(&b.1));

    let mut h = FxHasher::default();
    term_vec.hash(&mut h);
    nonterm_vec.hash(&mut h);
    h.finish()
}
