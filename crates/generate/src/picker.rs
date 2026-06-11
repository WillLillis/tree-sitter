//! Per-state representation choice between Dense, CSR, and Small tiers
//! for the generated parse table.
//!
//! ## Safe mode (default)
//!
//! [`PickerConfig::Safe`] defers the Dense decision to the original
//! `large_state_count` heuristic. The picker does not promote states
//! out of Dense or move Dense states elsewhere. For non-Dense states,
//! it picks between CSR and Small per state by minimum estimated byte
//! cost.
//!
//! ## Decoupled mode
//!
//! [`PickerConfig::Decoupled`] ignores `large_state_count` and evaluates
//! each non-pinned state (i.e. `i >= PINNED_DENSE_STATE_COUNT`) against
//! three independent thresholds:
//!
//!   - `dense_csr_bias`: CSR wins over Dense only if
//!     `csr_bytes <= dense_csr_bias * dense_bytes`.
//!   - `dense_small_bias`: Small wins over Dense only if
//!     `small_bytes <= dense_small_bias * dense_bytes`.
//!   - `small_csr_bias`: Small wins over CSR only if
//!     `small_bytes <= small_csr_bias * csr_bytes`. The most subtle of
//!     the three; rarely needs tuning.
//!
//! This mode is exposed via `tree-sitter.json` and the `tree-sitter
//! generate` CLI flags for grammar authors who want to optimize parser size
//! and/or for runtime performance.
//!
//! After the per-state tier decision, both modes run the same
//! dedup-promotion pass that moves CSR rows back to Small when many
//! states share a canonical body and one shared body + N map entries
//! beats N independent CSR rows.

use std::hash::{Hash, Hasher};

use rustc_hash::{FxHashMap, FxHashSet, FxHasher};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    PINNED_DENSE_STATE_COUNT,
    tables::{GotoAction, ParseTable, ParseTableEntry},
};

/// Which storage tier the picker assigned to a state.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Repr {
    /// `parse_table[state * sym_count + symbol]` 2D array. O(1) lookup.
    Dense,
    /// Sparse row of (sym, val) pairs. `O(log nnz)` binary search.
    Csr,
    /// Grouped sparse rep with deduplicated bodies. `O(group_count + nnz)`
    /// linear scan but heavily shared across states.
    Small,
}

/// Per-state byte cost estimate for each candidate tier.
///
///   - Dense: no per-state overhead. The state slots into the
///     `parse_table[LARGE_STATE_COUNT][SYMBOL_COUNT]` matrix.
///   - Csr:   one `row_offsets` u32 entry (4 bytes) plus
///     `nnz * 4` bytes for the interleaved (sym, val) pairs.
///   - Small: the body bytes (`4 * group_count + 2 * nnz`) plus the
///     `group_count` u16 header (2 bytes) plus the
///     `small_parse_table_map` u32 entry (4 bytes).
#[derive(Copy, Clone, Debug)]
pub struct StateByteCosts {
    pub dense: usize,
    pub csr: usize,
    pub small: usize,
}

/// Empirically determined best biases for _most_ grammars.
pub const DEFAULT_DENSE_CSR_BIAS: f64 = 0.5;
pub const DEFAULT_DENSE_SMALL_BIAS: f64 = 0.5;
/// Threshold for promoting a state from `CSR` to `Small`. A state that already
/// qualified for `CSR` is sparse, so `Small` only wins when it's substantially cheaper.
pub const DEFAULT_SMALL_CSR_BIAS: f64 = 0.6;

/// `u16` cell in the dense `parse_table[state][symbol]` matrix.
pub const DENSE_ENTRY_BYTES: usize = 2;
/// `u32` per-state entry in CSR's `row_offsets` array.
pub const CSR_ROW_OFFSETS_ENTRY_BYTES: usize = 4;
/// One interleaved `(sym:u16, val:u16)` pair per non-zero entry in CSR.
pub const CSR_PAIR_BYTES: usize = 4;
/// `u16` per symbol listed inside a small-table group.
pub const SMALL_SYMBOL_BYTES: usize = 2;
/// `(kind:u16, value:u16)` header per group in the small table.
pub const SMALL_GROUP_HEADER_BYTES: usize = 4;
/// `u16` `group_count` header at the start of each small-table state.
pub const SMALL_GROUP_COUNT_BYTES: usize = 2;
/// `u32` per-state entry in `small_parse_table_map`.
pub const SMALL_MAP_ENTRY_BYTES: usize = 4;

/// Identity of a picker bias axis.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Bias {
    DenseCsr,
    DenseSmall,
    SmallCsr,
}

impl std::fmt::Display for Bias {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DenseCsr => write!(f, "dense-csr-bias"),
            Self::DenseSmall => write!(f, "dense-small-bias"),
            Self::SmallCsr => write!(f, "small-csr-bias"),
        }
    }
}

impl Serialize for Bias {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for Bias {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "dense-csr-bias" => Ok(Self::DenseCsr),
            "dense-small-bias" => Ok(Self::DenseSmall),
            "small-csr-bias" => Ok(Self::SmallCsr),
            other => Err(serde::de::Error::unknown_variant(
                other,
                &["dense-csr-bias", "dense-small-bias", "small-csr-bias"],
            )),
        }
    }
}

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub enum PickerConfig {
    /// Defer `Dense` to the old `large_state_count` heuristic with argmin
    /// for non-Dense.
    #[default]
    Safe,
    /// 3-way bias-based picker.
    Decoupled(DecoupledBiases),
}

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecoupledBiases {
    #[serde(rename = "dense-csr-bias", default)]
    pub(crate) dense_csr: Option<f64>,
    #[serde(rename = "dense-small-bias", default)]
    pub(crate) dense_small: Option<f64>,
    #[serde(rename = "small-csr-bias", default)]
    pub(crate) small_csr: Option<f64>,
}

impl PickerConfig {
    /// Construct a `Decoupled` config from raw (optionally-unset)
    /// biases. Validates that any `Some` value is in `[0.0, 1.0]` and
    /// preserves `None` so per-axis inheritance still works at layer
    /// composition time.
    pub fn decoupled(
        dense_csr_bias: Option<f64>,
        dense_small_bias: Option<f64>,
        small_csr_bias: Option<f64>,
    ) -> Result<Self, PickerConfigError> {
        check_range(Bias::DenseCsr, dense_csr_bias)?;
        check_range(Bias::DenseSmall, dense_small_bias)?;
        check_range(Bias::SmallCsr, small_csr_bias)?;
        Ok(Self::Decoupled(DecoupledBiases {
            dense_csr: dense_csr_bias,
            dense_small: dense_small_bias,
            small_csr: small_csr_bias,
        }))
    }

    /// Compose all layers in source order and normalize the result.
    /// Layers with `None` are silent. Inputs are pre-validated by
    /// [`Self::decoupled`] / [`Self::Safe`], so merging cannot
    /// introduce new invalid values. After merging, any remaining
    /// `None` biases on a `Decoupled` result are filled with the
    /// module-level defaults.
    #[must_use]
    pub fn from_layers(layers: impl IntoIterator<Item = Option<Self>>) -> Self {
        let mut state: Option<Self> = None;
        for layer in layers {
            state = Self::merge(state, layer);
        }
        match state.unwrap_or(Self::Safe) {
            Self::Safe => Self::Safe,
            Self::Decoupled(b) => Self::Decoupled(DecoupledBiases {
                dense_csr: Some(b.dense_csr.unwrap_or(DEFAULT_DENSE_CSR_BIAS)),
                dense_small: Some(b.dense_small.unwrap_or(DEFAULT_DENSE_SMALL_BIAS)),
                small_csr: Some(b.small_csr.unwrap_or(DEFAULT_SMALL_CSR_BIAS)),
            }),
        }
    }

    /// Compose a single layer over the prior accumulated state.
    /// `None` for either input means "this layer is silent".
    fn merge(prior: Option<Self>, layer: Option<Self>) -> Option<Self> {
        match (prior, layer) {
            (p, None) => p,
            (_, Some(Self::Safe)) => Some(Self::Safe),
            (Some(Self::Decoupled(p)), Some(Self::Decoupled(l))) => {
                Some(Self::Decoupled(DecoupledBiases {
                    dense_csr: l.dense_csr.or(p.dense_csr),
                    dense_small: l.dense_small.or(p.dense_small),
                    small_csr: l.small_csr.or(p.small_csr),
                }))
            }
            (_, Some(layer @ Self::Decoupled(_))) => Some(layer),
        }
    }

    /// Compute the per-state tier picks before the dedup-promotion
    /// pass. Dispatches to the safe or decoupled algorithm based on
    /// mode and fills any unset `Decoupled` biases with the
    /// module-level defaults.
    fn picks(&self, costs: &[StateByteCosts], large_state_count: usize) -> Vec<Repr> {
        match self {
            Self::Safe => safe_picks(costs, large_state_count),
            Self::Decoupled(b) => decoupled_picks(
                costs,
                b.dense_csr.unwrap_or(DEFAULT_DENSE_CSR_BIAS),
                b.dense_small.unwrap_or(DEFAULT_DENSE_SMALL_BIAS),
                b.small_csr.unwrap_or(DEFAULT_SMALL_CSR_BIAS),
            ),
        }
    }
}

fn check_range(bias: Bias, value: Option<f64>) -> Result<(), PickerConfigError> {
    match value {
        None => Ok(()),
        Some(v) => {
            if v.is_finite() && (0.0..=1.0).contains(&v) {
                Ok(())
            } else {
                Err(PickerConfigError { bias, value: v })
            }
        }
    }
}

#[derive(Debug, Error, Serialize, Deserialize)]
#[error("picker `{bias}` must be in [0.0, 1.0], got {value}")]
pub struct PickerConfigError {
    bias: Bias,
    value: f64,
}

#[derive(Serialize, Deserialize)]
struct PickerLayersJson {
    #[serde(default)]
    grammars: Vec<PickerGrammarEntryJson>,
    #[serde(default)]
    picker: Option<RawPickerJson>,
}

#[derive(Serialize, Deserialize)]
struct PickerGrammarEntryJson {
    name: String,
    #[serde(default)]
    picker: Option<RawPickerJson>,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum RawPickerJson {
    Mode(PickerModeJson),
    Biases(DecoupledBiases),
}

#[derive(Serialize, Deserialize)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
enum PickerModeJson {
    Safe,
}

#[derive(Debug, Error, Serialize)]
pub enum ParsePickerLayersError {
    #[error("failed to parse picker config in `{path}` -- {error}")]
    Json { path: String, error: String },
    #[error("invalid picker configuration -- {0}")]
    Validation(#[from] PickerConfigError),
}

/// Parse the top-level and per-grammar picker layers out of contents of tree-sitter.json
pub fn parse_picker_layers(
    contents: &str,
    source_path: &str,
    grammar_name: &str,
) -> Result<(Option<PickerConfig>, Option<PickerConfig>), ParsePickerLayersError> {
    let parsed: PickerLayersJson =
        serde_json::from_str(contents).map_err(|e| ParsePickerLayersError::Json {
            path: source_path.to_string(),
            error: e.to_string(),
        })?;
    let top = parsed.picker.map(raw_to_picker_config).transpose()?;
    let per_grammar = parsed
        .grammars
        .into_iter()
        .find(|g| g.name == grammar_name)
        .and_then(|g| g.picker)
        .map(raw_to_picker_config)
        .transpose()?;
    Ok((top, per_grammar))
}

#[expect(clippy::needless_pass_by_value, reason = "map ergonomics")]
fn raw_to_picker_config(value: RawPickerJson) -> Result<PickerConfig, PickerConfigError> {
    match value {
        RawPickerJson::Mode(PickerModeJson::Safe) => Ok(PickerConfig::Safe),
        RawPickerJson::Biases(b) => {
            PickerConfig::decoupled(b.dense_csr, b.dense_small, b.small_csr)
        }
    }
}

/// Compute byte-cost estimates for each tier per state.
pub fn compute_standalone_costs(parse_table: &ParseTable) -> Vec<StateByteCosts> {
    let symbol_count = parse_table.symbols.len();
    let mut out = Vec::with_capacity(parse_table.states.len());
    let mut terminal_groups: FxHashSet<&ParseTableEntry> = FxHashSet::default();
    let mut nonterminal_groups: FxHashSet<usize> = FxHashSet::default();

    for (state_idx, state) in parse_table.states.iter().enumerate() {
        terminal_groups.clear();
        nonterminal_groups.clear();
        let nnz = state.terminal_entries.len() + state.nonterminal_entries.len();

        for entry in state.terminal_entries.values() {
            terminal_groups.insert(entry);
        }
        for action in state.nonterminal_entries.values() {
            let id = match action {
                GotoAction::Goto(s) => *s,
                GotoAction::ShiftExtra => state_idx,
            };
            nonterminal_groups.insert(id);
        }
        let group_count = terminal_groups.len() + nonterminal_groups.len();

        out.push(StateByteCosts {
            dense: symbol_count * DENSE_ENTRY_BYTES,
            csr: nnz * CSR_PAIR_BYTES + CSR_ROW_OFFSETS_ENTRY_BYTES,
            small: SMALL_SYMBOL_BYTES * nnz
                + SMALL_GROUP_HEADER_BYTES * group_count
                + SMALL_GROUP_COUNT_BYTES
                + SMALL_MAP_ENTRY_BYTES,
        });
    }

    out
}

/// Decide a tier per state, then run the shared dedup-promotion pass.
pub fn decide_state_representations(
    parse_table: &ParseTable,
    large_state_count: usize,
    config: &PickerConfig,
) -> Vec<Repr> {
    let costs = compute_standalone_costs(parse_table);
    let mut picks = config.picks(&costs, large_state_count);

    // Group non-Dense states by canonical-small hash.
    let mut by_hash: FxHashMap<u64, Vec<usize>> = FxHashMap::default();
    let mut buf = CanonicalSmallHashBuf::new();
    for state_id in picks
        .iter()
        .enumerate()
        .filter_map(|(i, p)| if *p == Repr::Dense { None } else { Some(i) })
    {
        by_hash
            .entry(canonical_small_hash(parse_table, state_id, &mut buf))
            .or_default()
            .push(state_id);
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
                Repr::Csr => costs[i].csr,
                Repr::Small => costs[i].small,
                Repr::Dense => unreachable!(),
            })
            .sum();
        // Every state in `group` shares the same canonical_small_hash,
        // which means their `small` costs are all equal (the cost is a
        // pure function of `nnz` and `group_count`, both identical
        // across byte-equivalent states).
        let canonical = &costs[group[0]];
        let promoted_bytes = canonical.small + SMALL_MAP_ENTRY_BYTES * (group.len() - 1);
        if promoted_bytes <= current_bytes {
            for &i in &group {
                picks[i] = Repr::Small;
            }
        }
    }

    picks
}

/// Safe per-state tier choice. States below `large_state_count` go to
/// Dense (tree-sitter's original heuristic); the rest pick CSR vs Small
/// by argmin bytes.
fn safe_picks(standalone_costs: &[StateByteCosts], large_state_count: usize) -> Vec<Repr> {
    standalone_costs
        .iter()
        .enumerate()
        .map(|(i, c)| {
            if i < large_state_count {
                Repr::Dense
            } else if c.csr <= c.small {
                Repr::Csr
            } else {
                Repr::Small
            }
        })
        .collect()
}

/// Decoupled per-state tier choice. Pinned-Dense states stay Dense;
/// the rest evaluate against the three configured biases.
fn decoupled_picks(
    standalone_costs: &[StateByteCosts],
    dense_csr_bias: f64,
    dense_small_bias: f64,
    small_csr_bias: f64,
) -> Vec<Repr> {
    standalone_costs
        .iter()
        .enumerate()
        .map(|(state_idx, c)| {
            if state_idx < PINNED_DENSE_STATE_COUNT {
                return Repr::Dense;
            }
            let dense = c.dense as f64;
            let csr = c.csr as f64;
            let small = c.small as f64;
            let csr_beats_dense = csr <= dense_csr_bias * dense;
            let small_beats_csr = small <= small_csr_bias * csr;
            let small_beats_dense = small <= dense_small_bias * dense;
            match (csr_beats_dense, small_beats_csr, small_beats_dense) {
                (false, _, false) => Repr::Dense,
                (false, _, true) => Repr::Small,
                (true, false, _) => Repr::Csr,
                (true, true, _) => {
                    if c.csr <= c.small {
                        Repr::Csr
                    } else {
                        Repr::Small
                    }
                }
            }
        })
        .collect()
}

pub type CanonicalSmallHashBuf = Vec<(u8, u64, u8, u32)>;

/// Hash the canonical small-table representation of a state. Two states
/// with equal hashes are presumed to emit byte-identical small entries.
///
/// The small layout is a deterministic function of the set of
/// `(symbol, action)` pairs in the state. We hash that set
/// order-independently by flattening it into `buf` as
/// `(domain_tag, action_key, sym_kind, sym_index)` tuples, sorting
/// once, then hashing the sorted sequence.
pub fn canonical_small_hash(
    parse_table: &ParseTable,
    state_idx: usize,
    buf: &mut CanonicalSmallHashBuf,
) -> u64 {
    const TERMINAL_TAG: u8 = 0;
    const NONTERMINAL_TAG: u8 = 1;

    let state = &parse_table.states[state_idx];
    buf.clear();
    buf.reserve(state.terminal_entries.len() + state.nonterminal_entries.len());
    for (sym, entry) in &state.terminal_entries {
        let mut eh = FxHasher::default();
        entry.hash(&mut eh);
        buf.push((TERMINAL_TAG, eh.finish(), sym.kind as u8, sym.index as u32));
    }
    for (sym, action) in &state.nonterminal_entries {
        let id = match action {
            GotoAction::Goto(s) => *s,
            GotoAction::ShiftExtra => state_idx,
        };
        buf.push((NONTERMINAL_TAG, id as u64, sym.kind as u8, sym.index as u32));
    }
    buf.sort_unstable();
    let mut h = FxHasher::default();
    buf.hash(&mut h);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(contents: &str, name: &str) -> (Option<PickerConfig>, Option<PickerConfig>) {
        parse_picker_layers(contents, "test.json", name).unwrap()
    }

    #[test]
    fn no_picker_fields_yields_no_layers() {
        let json = r#"{"grammars": [{"name": "foo"}]}"#;
        let (top, per_grammar) = parse(json, "foo");
        assert!(top.is_none());
        assert!(per_grammar.is_none());
    }

    #[test]
    fn safe_string_at_top_level() {
        let json = r#"{
            "grammars": [{"name": "foo"}],
            "picker": "safe"
        }"#;
        let (top, per_grammar) = parse(json, "foo");
        assert_eq!(top, Some(PickerConfig::Safe));
        assert!(per_grammar.is_none());
    }

    #[test]
    fn decoupled_biases_at_per_grammar() {
        let json = r#"{
            "grammars": [{
                "name": "foo",
                "picker": {
                    "dense-csr-bias": 0.3,
                    "small-csr-bias": 0.7
                }
            }]
        }"#;
        let (top, per_grammar) = parse(json, "foo");
        assert!(top.is_none());
        let Some(PickerConfig::Decoupled(b)) = per_grammar else {
            panic!("expected Decoupled, got {per_grammar:?}");
        };
        assert_eq!(b.dense_csr, Some(0.3));
        assert_eq!(b.dense_small, None);
        assert_eq!(b.small_csr, Some(0.7));
    }

    #[test]
    fn per_grammar_name_mismatch_drops_layer() {
        let json = r#"{
            "grammars": [{
                "name": "foo",
                "picker": "safe"
            }]
        }"#;
        let (top, per_grammar) = parse(json, "bar");
        assert!(top.is_none());
        assert!(per_grammar.is_none());
    }

    #[test]
    fn invalid_bias_value_is_validation_error() {
        let json = r#"{
            "grammars": [{"name": "foo"}],
            "picker": {"dense-csr-bias": 1.5}
        }"#;
        let err = parse_picker_layers(json, "test.json", "foo").unwrap_err();
        assert!(
            matches!(err, ParsePickerLayersError::Validation(_)),
            "expected Validation error, got {err:?}"
        );
    }

    #[test]
    fn unknown_field_in_biases_is_json_error() {
        let json = r#"{
            "grammars": [{"name": "foo"}],
            "picker": {"dense-csr-bias": 0.5, "denze-csr-bias": 0.3}
        }"#;
        let err = parse_picker_layers(json, "test.json", "foo").unwrap_err();
        assert!(
            matches!(err, ParsePickerLayersError::Json { .. }),
            "expected Json error, got {err:?}"
        );
    }
}
