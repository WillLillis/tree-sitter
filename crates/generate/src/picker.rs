//! Plan B picker: per-state representation choice between Dense, CSR, and
//! Small tiers for the generated parse table.
//!
//! ## Safe mode (default)
//!
//! [`PickerConfig::Safe`] defers the Dense decision to master's existing
//! `large_state_count` heuristic - the picker does not promote states
//! out of Dense or move Dense states elsewhere. For non-Dense states,
//! it picks between CSR and Small per state by minimum byte cost. This
//! mode has been validated across 126 grammars to match or beat master
//! on every one with no per-grammar regressions.
//!
//! ## Decoupled mode (opt-in)
//!
//! [`PickerConfig::Decoupled`] ignores `large_state_count` and
//! evaluates each non-runtime-reserved state (i.e. i > 1) against three
//! independent thresholds:
//!
//!   - `dense_csr_bias`:   CSR wins over Dense only if
//!                         `csr_bytes <= dense_csr_bias * dense_bytes`.
//!   - `dense_small_bias`: Small wins over Dense only if
//!                         `small_bytes <= dense_small_bias * dense_bytes`.
//!   - internal `SMALL_VS_CSR_BIAS = 0.6`: Small wins over CSR only if
//!                         `small_bytes <= 0.6 * csr_bytes`.
//!
//! This mode is exposed via `tree-sitter.json` and the `tree-sitter
//! generate` CLI flags for grammar authors who want to trade per-grammar
//! consistency for aggregate .so savings. A 4x5 static sweep across
//! `(dense_csr_bias, dense_small_bias)` on the 126-grammar corpus showed
//! the best aggregate point at `(0.5, 0.5)`: ~14% aggregate savings, but
//! with up to ~13% per-grammar growth on grammars where the safe picker
//! was already optimal (cuda, arduino, gn, org-mode). Those values are
//! also the [`DEFAULT_DENSE_CSR_BIAS`] / [`DEFAULT_DENSE_SMALL_BIAS`]
//! fill-ins used when a config layer requests decoupled mode but
//! doesn't specify both biases.
//!
//! After the per-state tier decision, both modes run the same
//! dedup-promotion pass that moves CSR rows back to Small when many
//! states share a canonical body and one shared body + N map entries
//! beats N independent CSR rows.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use serde::Serialize;
use thiserror::Error;

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

/// Empirical-best biases from the 4x5 static sweep across the 126-grammar
/// corpus. Used to fill in `Decoupled` biases that no config layer set.
pub const DEFAULT_DENSE_CSR_BIAS: f64 = 0.5;
pub const DEFAULT_DENSE_SMALL_BIAS: f64 = 0.5;

/// Identity of a picker bias axis. Owns the kebab-case string used in
/// JSON keys, CLI flags, and error messages so the spelling lives in
/// exactly one place.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Bias {
    DenseCsr,
    DenseSmall,
}

impl Bias {
    pub const fn name(self) -> &'static str {
        match self {
            Self::DenseCsr => "dense-csr-bias",
            Self::DenseSmall => "dense-small-bias",
        }
    }
}

impl std::fmt::Display for Bias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl Serialize for Bias {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(self.name())
    }
}

/// Validated picker configuration handed to the picker. Splitting this
/// from [`RawPickerConfig`] makes "partial config reaches the picker" a
/// compile error: the picker code matches the enum directly and gets
/// concrete bias values in the `Decoupled` arm.
#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum PickerConfig {
    /// Defer Dense to master's `large_state_count` heuristic; argmin
    /// for non-Dense. Validated against the 126-grammar corpus.
    #[default]
    Safe,
    /// 3-way bias-based picker. Both biases are present and validated
    /// to `(0.0, 1.0]` by the time we reach this variant.
    Decoupled {
        dense_csr_bias: f64,
        dense_small_bias: f64,
    },
}

/// Layerable, unvalidated picker config. One of these is produced per
/// source layer (file top-level, file per-grammar, CLI). Layers compose
/// via [`Self::merge`] and the final result is validated via
/// [`Self::build`] into a [`PickerConfig`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum RawPickerConfig {
    /// This layer explicitly requests the safe picker. Resets any
    /// `Decoupled` state from earlier layers.
    Safe,
    /// This layer specifies decoupled-mode biases. `None` fields are
    /// inherited from earlier layers; if no layer sets them, they fall
    /// back to [`DEFAULT_DENSE_CSR_BIAS`] / [`DEFAULT_DENSE_SMALL_BIAS`].
    Decoupled {
        dense_csr_bias: Option<f64>,
        dense_small_bias: Option<f64>,
    },
}

impl RawPickerConfig {
    /// Compose a layer over the prior accumulated state.
    /// `None` for either input means "this layer is silent".
    pub fn merge(prior: Option<Self>, layer: Option<Self>) -> Option<Self> {
        match (prior, layer) {
            (p, None) => p,
            (_, Some(Self::Safe)) => Some(Self::Safe),
            (
                Some(Self::Decoupled {
                    dense_csr_bias: pdc,
                    dense_small_bias: pds,
                }),
                Some(Self::Decoupled {
                    dense_csr_bias: ldc,
                    dense_small_bias: lds,
                }),
            ) => Some(Self::Decoupled {
                dense_csr_bias: ldc.or(pdc),
                dense_small_bias: lds.or(pds),
            }),
            (_, Some(layer @ Self::Decoupled { .. })) => Some(layer),
        }
    }

    /// Validate and lower into a [`PickerConfig`]. Decoupled biases that
    /// no layer set fall back to the module-level empirical defaults.
    pub fn build(self) -> Result<PickerConfig, PickerConfigError> {
        match self {
            Self::Safe => Ok(PickerConfig::Safe),
            Self::Decoupled {
                dense_csr_bias,
                dense_small_bias,
            } => {
                let dc = dense_csr_bias.unwrap_or(DEFAULT_DENSE_CSR_BIAS);
                let ds = dense_small_bias.unwrap_or(DEFAULT_DENSE_SMALL_BIAS);
                check_range(Bias::DenseCsr, dc)?;
                check_range(Bias::DenseSmall, ds)?;
                Ok(PickerConfig::Decoupled {
                    dense_csr_bias: dc,
                    dense_small_bias: ds,
                })
            }
        }
    }
}

impl PickerConfig {
    /// Compose all layers in source order and validate. Layers with
    /// `None` are silent. Final result is `Safe` if no layer spoke.
    pub fn from_layers(
        layers: impl IntoIterator<Item = Option<RawPickerConfig>>,
    ) -> Result<Self, PickerConfigError> {
        let mut state: Option<RawPickerConfig> = None;
        for layer in layers {
            state = RawPickerConfig::merge(state, layer);
        }
        match state {
            None => Ok(Self::Safe),
            Some(raw) => raw.build(),
        }
    }
}

fn check_range(bias: Bias, value: f64) -> Result<(), PickerConfigError> {
    if value.is_finite() && (0.0..=1.0).contains(&value) {
        Ok(())
    } else {
        Err(PickerConfigError::OutOfRange { bias, value })
    }
}

/// Error returned when a [`RawPickerConfig`] fails validation.
#[derive(Debug, Error, Serialize)]
pub enum PickerConfigError {
    #[error("picker `{bias}` must be in [0.0, 1.0], got {value}")]
    OutOfRange { bias: Bias, value: f64 },
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

/// Decide a tier per state, then run the shared dedup-promotion pass.
///
/// Behavior depends on `config`: see module docs for the two modes.
pub fn decide_state_representations(
    parse_table: &ParseTable,
    standalone_costs: &[StateCosts],
    large_state_count: usize,
    config: &PickerConfig,
) -> Vec<Repr> {
    let n = parse_table.states.len();
    let mut picks = match *config {
        PickerConfig::Safe => safe_picks(standalone_costs, large_state_count),
        PickerConfig::Decoupled {
            dense_csr_bias,
            dense_small_bias,
        } => decoupled_picks(standalone_costs, dense_csr_bias, dense_small_bias),
    };

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

/// Safe per-state tier choice. States below `large_state_count` go to
/// Dense (master's heuristic); the rest pick CSR vs Small by argmin
/// bytes.
fn safe_picks(standalone_costs: &[StateCosts], large_state_count: usize) -> Vec<Repr> {
    standalone_costs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i < large_state_count {
                Repr::Dense
            } else if c.csr_bytes <= c.small_bytes {
                Repr::Csr
            } else {
                Repr::Small
            }
        })
        .collect()
}

/// Decoupled per-state tier choice. States 0 and 1 are pinned to Dense
/// (runtime convention); the rest evaluate against the two configured
/// biases plus the internal small/CSR threshold.
fn decoupled_picks(
    standalone_costs: &[StateCosts],
    dense_csr_bias: f64,
    dense_small_bias: f64,
) -> Vec<Repr> {
    const SMALL_VS_CSR_BIAS: f64 = 0.6;
    standalone_costs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i <= 1 {
                return Repr::Dense;
            }
            let dense = c.dense_bytes as f64;
            let csr = c.csr_bytes as f64;
            let small = c.small_bytes as f64;
            let csr_beats_dense = csr <= dense_csr_bias * dense;
            let small_beats_csr = small <= SMALL_VS_CSR_BIAS * csr;
            let small_beats_dense = small <= dense_small_bias * dense;
            match (csr_beats_dense, small_beats_csr, small_beats_dense) {
                (false, _, false) => Repr::Dense,
                (false, _, true) => Repr::Small,
                (true, false, _) => Repr::Csr,
                (true, true, _) => {
                    if c.csr_bytes <= c.small_bytes {
                        Repr::Csr
                    } else {
                        Repr::Small
                    }
                }
            }
        })
        .collect()
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
