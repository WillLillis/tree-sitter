use rustc_hash::FxHashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[cfg(test)]
use super::ExtractedSyntaxGrammar;
#[cfg(test)]
use crate::{grammars::Variable, rules::Rule};
use crate::{
    grammars::{Production, ProductionStep, ReservedWordSetId, SyntaxGrammar, SyntaxVariable},
    rules::{Associativity, Precedence, Symbol, TokenSet},
};

pub type FlattenGrammarResult<T> = Result<T, FlattenGrammarError>;

#[derive(Debug, Error, Serialize, Deserialize)]
pub enum FlattenGrammarError {
    #[error("No such reserved word set: {0}")]
    NoReservedWordSet(String),
    #[error(
        "The rule `{0}` matches the empty string.

Tree-sitter does not support syntactic rules that match the empty string
unless they are used only as the grammar's start rule.
"
    )]
    EmptyString(String),
    #[error("Rule `{0}` cannot be inlined because it contains a reference to itself")]
    RecursiveInline(String),
}

use super::extract_tokens::PoolExtractedMeta;
use crate::rule_pool::{
    FProd, FSTEP_ASSOC_LEFT, FSTEP_ASSOC_MASK, FSTEP_ASSOC_RIGHT, FSTEP_PREC_INTEGER,
    FSTEP_PREC_MASK, FSTEP_PREC_NAME, FStep, FlatOut, Node as PoolNode, NodeId, PoolGrammar, Prec,
    RulePool, StrId,
};
use crate::rules::SymbolType;

/// One continuation frame, packed to 8 bytes (tag in the 2 high bits of `.0`).
///   `Seq`:          tag 0, `.0` = next child (abs index into `children`), `.1` = end
///   `MetaExit`:     tag 1, `.0` low bits = pushed-stack flags, `.1` = steps len at entry
///   `ReservedExit`: tag 2
#[derive(Clone, Copy)]
struct Frame(u32, u32);

const _: () = assert!(std::mem::size_of::<Frame>() == 8);

const FR_SEQ: u32 = 0;
const FR_META: u32 = 1;
const FR_RES: u32 = 2;
const FR_PAYLOAD: u32 = (1 << 30) - 1;

const MP_PREC: u32 = 1;
const MP_ASSOC: u32 = 2;
const MP_ALIAS: u32 = 4;
const MP_FIELD: u32 = 8;

impl Frame {
    fn seq(start: u32, end: u32) -> Self {
        debug_assert!(end < (1 << 30));
        Self(start, end)
    }
    const fn meta(flags: u32, steps_before: u32) -> Self {
        Self((FR_META << 30) | flags, steps_before)
    }
    const fn reserved() -> Self {
        Self(FR_RES << 30, 0)
    }
    const fn tag(self) -> u32 {
        self.0 >> 30
    }
    const fn seq_exhausted(self) -> bool {
        self.0 == self.1
    }
}

/// Reusable flatten scratch. Everything is truncate-restore or snapshot
/// scratch; steady state allocates only output-pool growth.
#[derive(Default)]
pub(super) struct FlattenState {
    frames: Vec<Frame>,
    steps: Vec<FStep>,
    prec: Vec<Prec>,
    assoc: Vec<Associativity>,
    alias: Vec<crate::rule_pool::Alias>,
    field: Vec<StrId>,
    reserved: Vec<u16>,
    /// Fork snapshots, stacked as segments. A branch can consume context
    /// pushed BEFORE the fork (an enclosing Metadata/Reserved exit pops it),
    /// so every stack that exits pop through needs a full snapshot; only
    /// `steps` grows monotonically within a branch and truncates back.
    snap_frames: Vec<Frame>,
    snap_prec: Vec<Prec>,
    snap_assoc: Vec<Associativity>,
    snap_alias: Vec<crate::rule_pool::Alias>,
    snap_field: Vec<StrId>,
    snap_reserved: Vec<u16>,
    dyn_prec: i32,
}

/// Snapshot offsets captured at a fork; lives on the Rust call stack.
#[derive(Clone, Copy)]
struct Mark {
    steps: u32,
    frames: u32,
    prec: u32,
    assoc: u32,
    alias: u32,
    field: u32,
    reserved: u32,
    dyn_prec: i32,
}

impl FlattenState {
    fn reset(&mut self) {
        self.frames.clear();
        self.steps.clear();
        self.prec.clear();
        self.assoc.clear();
        self.alias.clear();
        self.field.clear();
        self.reserved.clear();
        self.snap_frames.clear();
        self.snap_prec.clear();
        self.snap_assoc.clear();
        self.snap_alias.clear();
        self.snap_field.clear();
        self.snap_reserved.clear();
        self.dyn_prec = 0;
    }

    fn fork_save(&mut self) -> Mark {
        let mark = Mark {
            steps: self.steps.len() as u32,
            frames: self.snap_frames.len() as u32,
            prec: self.snap_prec.len() as u32,
            assoc: self.snap_assoc.len() as u32,
            alias: self.snap_alias.len() as u32,
            field: self.snap_field.len() as u32,
            reserved: self.snap_reserved.len() as u32,
            dyn_prec: self.dyn_prec,
        };
        self.snap_frames.extend_from_slice(&self.frames);
        self.snap_prec.extend_from_slice(&self.prec);
        self.snap_assoc.extend_from_slice(&self.assoc);
        self.snap_alias.extend_from_slice(&self.alias);
        self.snap_field.extend_from_slice(&self.field);
        self.snap_reserved.extend_from_slice(&self.reserved);
        mark
    }

    fn restore(&mut self, mark: Mark) {
        self.steps.truncate(mark.steps as usize);
        self.dyn_prec = mark.dyn_prec;
        self.frames.clear();
        self.frames
            .extend_from_slice(&self.snap_frames[mark.frames as usize..]);
        self.prec.clear();
        self.prec
            .extend_from_slice(&self.snap_prec[mark.prec as usize..]);
        self.assoc.clear();
        self.assoc
            .extend_from_slice(&self.snap_assoc[mark.assoc as usize..]);
        self.alias.clear();
        self.alias
            .extend_from_slice(&self.snap_alias[mark.alias as usize..]);
        self.field.clear();
        self.field
            .extend_from_slice(&self.snap_field[mark.field as usize..]);
        self.reserved.clear();
        self.reserved
            .extend_from_slice(&self.snap_reserved[mark.reserved as usize..]);
    }

    fn fork_done(&mut self, mark: Mark) {
        self.snap_frames.truncate(mark.frames as usize);
        self.snap_prec.truncate(mark.prec as usize);
        self.snap_assoc.truncate(mark.assoc as usize);
        self.snap_alias.truncate(mark.alias as usize);
        self.snap_field.truncate(mark.field as usize);
        self.snap_reserved.truncate(mark.reserved as usize);
    }

    fn push_step(&mut self, kind: SymbolType, index: u32) {
        self.steps.push(FStep::pack(
            Symbol {
                kind,
                index: index as usize,
            },
            self.prec.last().copied().unwrap_or(Prec::None),
            self.assoc.last().copied(),
            self.alias.last().copied(),
            self.field.last().copied(),
            self.reserved.last().copied().unwrap_or(0),
        ));
    }

    /// `true` if nothing after the current point can push a step (computed
    /// bottom-up from the remaining frames).
    fn at_end(&self) -> bool {
        self.frames
            .iter()
            .all(|f| f.tag() != FR_SEQ || f.seq_exhausted())
    }

    /// Metadata exit: pop what was pushed; if a step was pushed in the
    /// subtree and more steps can follow, rewrite the last step's
    /// precedence/associativity to the outer context.
    fn meta_exit(&mut self, flags: u32, steps_before: u32) {
        let did_push = self.steps.len() as u32 > steps_before;
        let fixup = did_push && !self.at_end();
        if flags & MP_PREC != 0 {
            self.prec.pop();
            if fixup {
                let (bits, val) = match self.prec.last() {
                    None | Some(Prec::None) => (0u8, 0i32),
                    Some(Prec::Integer(n)) => (FSTEP_PREC_INTEGER, *n),
                    Some(Prec::Name(sid)) => (FSTEP_PREC_NAME, sid.raw() as i32),
                };
                let step = self.steps.last_mut().unwrap();
                step.prec_val = val;
                step.flags = (step.flags & !FSTEP_PREC_MASK) | bits;
            }
        }
        if flags & MP_ASSOC != 0 {
            self.assoc.pop();
            if fixup {
                let bits = match self.assoc.last() {
                    None => 0u8,
                    Some(Associativity::Left) => FSTEP_ASSOC_LEFT,
                    Some(Associativity::Right) => FSTEP_ASSOC_RIGHT,
                };
                let step = self.steps.last_mut().unwrap();
                step.flags = (step.flags & !FSTEP_ASSOC_MASK) | bits;
            }
        }
        if flags & MP_ALIAS != 0 {
            self.alias.pop();
        }
        if flags & MP_FIELD != 0 {
            self.field.pop();
        }
    }
}

pub(super) fn flatten_grammar(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    st: &mut FlattenState,
    out: &mut FlatOut,
) -> FlattenGrammarResult<()> {
    // Last wins on duplicate names.
    let reserved_ids: FxHashMap<StrId, u16> = meta
        .reserved_sets
        .iter()
        .enumerate()
        .map(|(i, (name, _))| (*name, i as u16))
        .collect();
    out.clear();
    for v in &g.variables {
        let prod_start = out.productions.len() as u32;
        st.reset();
        run_path(&g.pool, &reserved_ids, v.root, st, out, prod_start)?;
        out.var_prods
            .push((prod_start, out.productions.len() as u32));
    }
    check(g, meta, out)
}

/// Post-flatten checks: empty productions in used variables, recursive
/// inlines. Same iteration order as the historical Rule-based pass.
fn check(g: &PoolGrammar, meta: &PoolExtractedMeta, out: &FlatOut) -> FlattenGrammarResult<()> {
    for (i, &(ps, pe)) in out.var_prods.iter().enumerate() {
        let symbol = Symbol::non_terminal(i);
        let used = out.steps.iter().any(|s| s.symbol() == symbol);
        let inlined = meta.inline.contains(&symbol);
        for p in &out.productions[ps as usize..pe as usize] {
            if used && p.steps_len == 0 {
                Err(FlattenGrammarError::EmptyString(
                    g.pool.resolve(g.variables[i].name).to_string(),
                ))?;
            }
            if inlined
                && out.steps[p.step_range()]
                    .iter()
                    .any(|s| s.symbol() == symbol)
            {
                Err(FlattenGrammarError::RecursiveInline(
                    g.pool.resolve(g.variables[i].name).to_string(),
                ))?;
            }
        }
    }
    Ok(())
}

/// Enter `node` and, unless it forked, drive the continuation to completion.
fn run_path(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    node: NodeId,
    st: &mut FlattenState,
    out: &mut FlatOut,
    prod_start: u32,
) -> FlattenGrammarResult<()> {
    if enter(pool, reserved_ids, node, st, out, prod_start)? {
        drive(pool, reserved_ids, st, out, prod_start)?;
    }
    Ok(())
}

/// Process one node. Returns `false` if it forked (a `Choice` drove every
/// branch to completion; the caller's continuation is consumed). Unary
/// chains iterate in place.
fn enter(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    mut node: NodeId,
    st: &mut FlattenState,
    out: &mut FlatOut,
    prod_start: u32,
) -> FlattenGrammarResult<bool> {
    loop {
        match pool.node(node) {
            PoolNode::Sym { kind, index } => {
                st.push_step(kind, index);
                return Ok(true);
            }
            PoolNode::Seq(range) => {
                st.frames
                    .push(Frame::seq(range.start, range.start + range.len));
                return Ok(true);
            }
            PoolNode::Metadata { params, rule } => {
                let p = pool.params(params);
                let mut flags = 0;
                if p.prec != Prec::None {
                    st.prec.push(p.prec);
                    flags |= MP_PREC;
                }
                if let Some(a) = p.assoc {
                    st.assoc.push(a);
                    flags |= MP_ASSOC;
                }
                if let Some(a) = p.alias {
                    st.alias.push(a);
                    flags |= MP_ALIAS;
                }
                if let Some(f) = p.field {
                    st.field.push(f);
                    flags |= MP_FIELD;
                }
                if p.dynamic_precedence.abs() > st.dyn_prec.abs() {
                    st.dyn_prec = p.dynamic_precedence;
                }
                st.frames.push(Frame::meta(flags, st.steps.len() as u32));
                node = rule;
            }
            PoolNode::Reserved { rule, ctx } => {
                let Some(&set) = reserved_ids.get(&ctx) else {
                    return Err(FlattenGrammarError::NoReservedWordSet(
                        pool.resolve(ctx).to_string(),
                    ));
                };
                st.reserved.push(set);
                st.frames.push(Frame::reserved());
                node = rule;
            }
            PoolNode::Choice(range) => {
                let mark = st.fork_save();
                for i in range.as_range() {
                    let child = pool.child_slice(range)[i - range.start as usize];
                    run_path(pool, reserved_ids, child, st, out, prod_start)?;
                    st.restore(mark);
                }
                st.fork_done(mark);
                return Ok(false);
            }
            // Blank pushes nothing; String/Pattern/NamedSymbol/Repeat cannot
            // appear post-extract/expand.
            _ => return Ok(true),
        }
    }
}

/// Consume frames until the continuation is exhausted, then emit. Stops
/// without emitting if a deeper fork took over.
fn drive(
    pool: &RulePool,
    reserved_ids: &FxHashMap<StrId, u16>,
    st: &mut FlattenState,
    out: &mut FlatOut,
    prod_start: u32,
) -> FlattenGrammarResult<()> {
    loop {
        let Some(&top) = st.frames.last() else {
            emit(st, out, prod_start);
            return Ok(());
        };
        match top.tag() {
            FR_SEQ => {
                if top.seq_exhausted() {
                    st.frames.pop();
                } else {
                    let cur = top.0 & FR_PAYLOAD;
                    st.frames.last_mut().unwrap().0 += 1;
                    let child =
                        pool.child_slice(crate::rule_pool::NodeRange { start: cur, len: 1 })[0];
                    if !enter(pool, reserved_ids, child, st, out, prod_start)? {
                        return Ok(());
                    }
                }
            }
            FR_META => {
                let (flags, steps_before) = (top.0 & FR_PAYLOAD, top.1);
                st.frames.pop();
                st.meta_exit(flags, steps_before);
            }
            _ => {
                st.frames.pop();
                st.reserved.pop();
            }
        }
    }
}

/// Append the completed path as a production unless this variable already
/// has an identical one (order-preserving dedup).
fn emit(st: &FlattenState, out: &mut FlatOut, prod_start: u32) {
    for p in &out.productions[prod_start as usize..] {
        if p.dynamic_precedence == st.dyn_prec && out.steps[p.step_range()] == st.steps[..] {
            return;
        }
    }
    let steps_start = out.steps.len() as u32;
    out.steps.extend_from_slice(&st.steps);
    out.productions.push(FProd {
        steps_start,
        steps_len: st.steps.len() as u32,
        dynamic_precedence: st.dyn_prec,
    });
}

/// Materialize one flat step. Deleted at the `SyntaxGrammar` flip.
pub(super) fn step_to_master(pool: &RulePool, s: &FStep) -> ProductionStep {
    ProductionStep {
        symbol: s.symbol(),
        precedence: match s.precedence() {
            Prec::None => Precedence::None,
            Prec::Integer(n) => Precedence::Integer(n),
            Prec::Name(sid) => Precedence::Name(pool.resolve(sid).to_string()),
        },
        associativity: s.associativity(),
        alias: s.alias().map(|a| crate::rules::Alias {
            value: pool.resolve(a.value).to_string(),
            is_named: a.is_named,
        }),
        field_name: s.field().map(|f| pool.resolve(f).to_string()),
        reserved_word_set_id: ReservedWordSetId(s.reserved as usize),
    }
}

/// Boundary materialization to the legacy-shaped [`SyntaxGrammar`].
/// Deleted at the `SyntaxGrammar` flip.
pub(super) fn materialize_flattened(
    g: &PoolGrammar,
    meta: &PoolExtractedMeta,
    out: &FlatOut,
) -> SyntaxGrammar {
    let step_to_master = |s: &FStep| step_to_master(&g.pool, s);
    let mut reserved_word_sets: Vec<TokenSet> = meta
        .reserved_sets
        .iter()
        .map(|(_, symbols)| symbols.iter().copied().collect())
        .collect();
    if reserved_word_sets.is_empty() {
        reserved_word_sets.push(TokenSet::default());
    }
    SyntaxGrammar {
        variables: g
            .variables
            .iter()
            .zip(&meta.kinds)
            .zip(&out.var_prods)
            .map(|((v, &kind), &(ps, pe))| SyntaxVariable {
                name: g.pool.resolve(v.name).to_string(),
                kind,
                productions: out.productions[ps as usize..pe as usize]
                    .iter()
                    .map(|p| Production {
                        dynamic_precedence: p.dynamic_precedence,
                        steps: out.steps[p.step_range()]
                            .iter()
                            .map(step_to_master)
                            .collect(),
                    })
                    .collect(),
            })
            .collect(),
        extra_symbols: meta.extra_symbols.clone(),
        expected_conflicts: meta.conflicts.clone(),
        variables_to_inline: meta.inline.clone(),
        precedence_orderings: g.precedence_orderings.clone(),
        external_tokens: meta.external_tokens.clone(),
        supertype_symbols: meta.supertypes.clone(),
        word_token: meta.word,
        reserved_word_sets,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammars::VariableType;

    #[test]
    fn test_flatten_grammar() {
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::seq(vec![
                Rule::non_terminal(1),
                Rule::prec_left(
                    Precedence::Integer(101),
                    Rule::seq(vec![
                        Rule::non_terminal(2),
                        Rule::choice(vec![
                            Rule::prec_right(
                                Precedence::Integer(102),
                                Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                            ),
                            Rule::non_terminal(5),
                        ]),
                        Rule::non_terminal(6),
                    ]),
                ),
                Rule::non_terminal(7),
            ]),
        });

        assert_eq!(
            result.productions,
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(3))
                            .with_prec(Precedence::Integer(102), Some(Associativity::Right)),
                        ProductionStep::new(Symbol::non_terminal(4))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ]
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(5))
                            .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ]
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_maximum_dynamic_precedence() {
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::seq(vec![
                Rule::non_terminal(1),
                Rule::prec_dynamic(
                    101,
                    Rule::seq(vec![
                        Rule::non_terminal(2),
                        Rule::choice(vec![
                            Rule::prec_dynamic(
                                102,
                                Rule::seq(vec![Rule::non_terminal(3), Rule::non_terminal(4)]),
                            ),
                            Rule::non_terminal(5),
                        ]),
                        Rule::non_terminal(6),
                    ]),
                ),
                Rule::non_terminal(7),
            ]),
        });

        assert_eq!(
            result.productions,
            vec![
                Production {
                    dynamic_precedence: 102,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::non_terminal(3)),
                        ProductionStep::new(Symbol::non_terminal(4)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ],
                },
                Production {
                    dynamic_precedence: 101,
                    steps: vec![
                        ProductionStep::new(Symbol::non_terminal(1)),
                        ProductionStep::new(Symbol::non_terminal(2)),
                        ProductionStep::new(Symbol::non_terminal(5)),
                        ProductionStep::new(Symbol::non_terminal(6)),
                        ProductionStep::new(Symbol::non_terminal(7)),
                    ],
                },
            ]
        );
    }

    #[test]
    fn test_flatten_grammar_with_final_precedence() {
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::prec_left(
                Precedence::Integer(101),
                Rule::seq(vec![Rule::non_terminal(1), Rule::non_terminal(2)]),
            ),
        });

        assert_eq!(
            result.productions,
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::non_terminal(1))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                    ProductionStep::new(Symbol::non_terminal(2))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                ]
            }]
        );

        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::prec_left(
                Precedence::Integer(101),
                Rule::seq(vec![Rule::non_terminal(1)]),
            ),
        });

        assert_eq!(
            result.productions,
            vec![Production {
                dynamic_precedence: 0,
                steps: vec![
                    ProductionStep::new(Symbol::non_terminal(1))
                        .with_prec(Precedence::Integer(101), Some(Associativity::Left)),
                ]
            }]
        );
    }

    #[test]
    fn test_flatten_grammar_with_field_names() {
        let result = flatten_variable(Variable {
            name: "test".to_string(),
            kind: VariableType::Named,
            rule: Rule::seq(vec![
                Rule::field("first-thing".to_string(), Rule::terminal(1)),
                Rule::terminal(2),
                Rule::choice(vec![
                    Rule::Blank,
                    Rule::field("second-thing".to_string(), Rule::terminal(3)),
                ]),
            ]),
        });

        assert_eq!(
            result.productions,
            vec![
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(1)).with_field_name("first-thing"),
                        ProductionStep::new(Symbol::terminal(2))
                    ]
                },
                Production {
                    dynamic_precedence: 0,
                    steps: vec![
                        ProductionStep::new(Symbol::terminal(1)).with_field_name("first-thing"),
                        ProductionStep::new(Symbol::terminal(2)),
                        ProductionStep::new(Symbol::terminal(3)).with_field_name("second-thing"),
                    ]
                },
            ]
        );
    }

    fn pool_from_extracted(g: &ExtractedSyntaxGrammar) -> (PoolGrammar, PoolExtractedMeta) {
        use crate::rule_pool::PoolVariable;
        let mut pool = RulePool::default();
        let variables = g
            .variables
            .iter()
            .map(|v| PoolVariable {
                name: pool.intern(&v.name),
                root: pool.add_rule(&v.rule),
            })
            .collect();
        let reserved_sets = g
            .reserved_word_sets
            .iter()
            .map(|s| (pool.intern(&s.name), s.reserved_words.clone()))
            .collect();
        let meta = PoolExtractedMeta {
            kinds: g.variables.iter().map(|v| v.kind).collect(),
            lexical_variables: Vec::new(),
            separator_roots: Vec::new(),
            extra_symbols: g.extra_symbols.clone(),
            external_tokens: g.external_tokens.clone(),
            reserved_sets,
            supertypes: g.supertype_symbols.clone(),
            conflicts: g.expected_conflicts.clone(),
            inline: g.variables_to_inline.clone(),
            word: g.word_token,
        };
        let pg = PoolGrammar {
            pool,
            name: String::new(),
            variables,
            external_roots: Vec::new(),
            extra_roots: Vec::new(),
            reserved_sets: Vec::new(),
            supertype_names: Vec::new(),
            conflict_names: Vec::new(),
            inline_names: Vec::new(),
            word_name: None,
            precedence_orderings: g.precedence_orderings.clone(),
        };
        (pg, meta)
    }

    fn pool_pass(g: &ExtractedSyntaxGrammar) -> FlattenGrammarResult<SyntaxGrammar> {
        let (pg, meta) = pool_from_extracted(g);
        let mut st = FlattenState::default();
        let mut out = crate::rule_pool::FlatOut::default();
        flatten_grammar(&pg, &meta, &mut st, &mut out)?;
        Ok(materialize_flattened(&pg, &meta, &out))
    }

    fn flatten_variable(variable: Variable) -> SyntaxVariable {
        let g = ExtractedSyntaxGrammar {
            variables: vec![variable],
            ..Default::default()
        };
        let mut grammar = pool_pass(&g).unwrap();
        grammar.variables.pop().unwrap()
    }

    #[test]
    fn test_errors() {
        // Unknown reserved context.
        let g = ExtractedSyntaxGrammar {
            variables: vec![Variable::named(
                "test",
                Rule::Reserved {
                    rule: Box::new(Rule::terminal(1)),
                    context_name: "nope".to_string(),
                },
            )],
            ..Default::default()
        };
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "No such reserved word set: nope"
        );

        // Empty production in a used variable.
        let g = ExtractedSyntaxGrammar {
            variables: vec![
                Variable::named("a", Rule::non_terminal(1)),
                Variable::named("b", Rule::Blank),
            ],
            ..Default::default()
        };
        assert_eq!(
            pool_pass(&g).unwrap_err().to_string(),
            "The rule `b` matches the empty string.

Tree-sitter does not support syntactic rules that match the empty string
unless they are used only as the grammar's start rule.
"
        );
    }

    #[test]
    fn test_flatten_grammar_with_recursive_inline_variable() {
        let result = pool_pass(&ExtractedSyntaxGrammar {
            extra_symbols: Vec::new(),
            expected_conflicts: Vec::new(),
            variables_to_inline: vec![Symbol::non_terminal(0)],
            precedence_orderings: Vec::new(),
            external_tokens: Vec::new(),
            supertype_symbols: Vec::new(),
            word_token: None,
            reserved_word_sets: Vec::new(),
            variables: vec![Variable {
                name: "test".to_string(),
                kind: VariableType::Named,
                rule: Rule::seq(vec![
                    Rule::non_terminal(0),
                    Rule::non_terminal(1),
                    Rule::non_terminal(2),
                ]),
            }],
        });

        assert_eq!(
            result.unwrap_err().to_string(),
            "Rule `test` cannot be inlined because it contains a reference to itself",
        );
    }
}
