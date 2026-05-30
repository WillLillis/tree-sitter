//! Plan B picker: per-state representation choice between Dense, CSR, and
//! Small tiers for the generated parse table.
//!
//! The Dense decision is deferred to master's existing heuristic via the
//! `large_state_count` input - the picker does not promote states out of
//! Dense or move Dense states elsewhere. For non-Dense states, it picks
//! between CSR and Small per state using a cost model
//! `total = bytes + LAMBDA * lookup_cycles`, then runs a dedup-promotion
//! pass that moves CSR rows back to Small when many states share a
//! canonical body and the byte savings outweigh the cycle cost.
//!
//! With both LAMBDAs at 0, the picker reduces to pure bytes minimization.
//! See the per-constant comments for the rationale behind the defaults.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};

use crate::tables::{GotoAction, ParseTable, ParseTableEntry};

/// Per-state cost weight on lookup cycles when choosing between CSR and
/// Small. Higher values bias the per-state pick toward faster tiers
/// (CSR's binary search over Small's linear scan). 0 reduces to pure
/// argmin-bytes, which is the validated default - the corpus comparison
/// (126 grammars) showed no per-grammar size regressions at LAMBDA=0,
/// and no clear evidence that non-zero values would beat that.
const PICKER_LAMBDA: f64 = 0.0;

/// Cost weight for the dedup-promotion comparison. Dedup converts N CSR
/// rows (each their own bytes) into one shared Small body plus N map
/// entries; per-state lookup goes from O(log nnz) to a Small linear scan.
/// A non-zero value scales that cycle penalty by group size and quickly
/// dominates any byte saving. At LAMBDA=0 dedup fires whenever bytes
/// drop, matching master's heavy Small dedup. Kept distinct from
/// PICKER_LAMBDA so the two decisions can be tuned independently.
const DEDUP_LAMBDA: f64 = 0.0;

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

/// Per-state cost estimate for each candidate tier. `dense_cycles` and
/// `dense_total()` are unused today (Dense is always pinned by the
/// master-derived `large_state_count`, not selected via cost) but are
/// kept for symmetry so the struct describes the full picker model.
#[derive(Copy, Clone, Debug)]
pub struct StateCosts {
    pub dense_bytes: usize,
    pub csr_bytes: usize,
    pub small_bytes: usize,
    pub dense_cycles: f64,
    pub csr_cycles: f64,
    pub small_cycles: f64,
}

impl StateCosts {
    #[expect(dead_code, reason = "kept for symmetry; see struct doc")]
    pub fn dense_total(&self) -> f64 {
        self.dense_bytes as f64 + PICKER_LAMBDA * self.dense_cycles
    }
    pub fn csr_total(&self) -> f64 {
        self.csr_bytes as f64 + PICKER_LAMBDA * self.csr_cycles
    }
    pub fn small_total(&self) -> f64 {
        self.small_bytes as f64 + PICKER_LAMBDA * self.small_cycles
    }
}

/// Compute byte cost and lookup-cycle estimate for each tier per state.
///
/// Lookup-cycle estimates:
///   * Dense: 1 indexed load.
///   * Csr:   `ceil(log2(nnz + 1))` binary search depth, clamped to >= 1
///            for the trivial one-entry case.
///   * Small: average-case scan `(group_count + nnz) / 2`. Walks half the
///            outer-loop groups on average to reach the matching group,
///            plus half the inner-loop symbol comparisons to the hit.
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

        let csr_cycles = if nnz <= 1 {
            1.0
        } else {
            ((nnz + 1) as f64).log2().ceil()
        };
        let small_cycles = (group_count as f64 + nnz as f64) / 2.0;

        out.push(StateCosts {
            dense_bytes: symbol_count * 2,
            csr_bytes: nnz * 4 + 4,
            small_bytes: 2 * nnz + 4 * group_count + 6,
            dense_cycles: 1.0,
            csr_cycles,
            small_cycles,
        });
    }
    out
}

/// Decide a tier per state.
///
/// State ids `0..large_state_count` are forced to Dense (this is master's
/// per-state heuristic - pinning to it guarantees no parse-time
/// regression on the hot path). For state ids >= `large_state_count`,
/// the picker chooses between CSR and Small by argmin of
/// `bytes + PICKER_LAMBDA * lookup_cycles`.
///
/// A dedup-promotion pass then groups non-Dense states by their canonical
/// small-body hash and demotes a whole group to Small when one shared
/// body plus N map entries beats N independent CSR rows under the
/// DEDUP_LAMBDA-weighted cost. Dense states never participate; they keep
/// their O(1) lookup guarantee.
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
        } else if c.csr_total() <= c.small_total() {
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
        let current_total: f64 = group
            .iter()
            .map(|&i| {
                let c = &standalone_costs[i];
                match picks[i] {
                    Repr::Csr => c.csr_bytes as f64 + DEDUP_LAMBDA * c.csr_cycles,
                    Repr::Small => c.small_bytes as f64 + DEDUP_LAMBDA * c.small_cycles,
                    Repr::Dense => unreachable!(),
                }
            })
            .sum();
        let canonical = &standalone_costs[group[0]];
        let promoted_total = canonical.small_bytes as f64
            + 4.0 * (group.len() - 1) as f64
            + DEDUP_LAMBDA * canonical.small_cycles * group.len() as f64;
        if promoted_total <= current_total {
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
