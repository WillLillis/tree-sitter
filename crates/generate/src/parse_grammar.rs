use rustc_hash::{FxHashMap, FxHashSet};

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use thiserror::Error;

use crate::{
    Diagnostic,
    grammars::PrecedenceEntry,
    rule_pool::{Node, NodeId, PoolGrammar, PoolReservedSet, PoolVariable, Prec, RulePool, StrId},
};
#[cfg(feature = "nativedsl")]
use crate::{grammars::InputGrammar, rules::Rule};

#[derive(Deserialize, Serialize)]
#[serde(tag = "type")]
#[expect(
    non_camel_case_types,
    reason = "variant names match JSON grammar format"
)]
#[expect(
    clippy::upper_case_acronyms,
    reason = "variant names match JSON grammar format"
)]
pub(crate) enum RuleJSON {
    ALIAS {
        content: Box<Self>,
        named: bool,
        value: String,
    },
    BLANK,
    STRING {
        value: String,
    },
    PATTERN {
        value: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        flags: Option<String>,
    },
    SYMBOL {
        name: String,
    },
    CHOICE {
        members: Vec<Self>,
    },
    FIELD {
        name: String,
        content: Box<Self>,
    },
    SEQ {
        members: Vec<Self>,
    },
    REPEAT {
        content: Box<Self>,
    },
    REPEAT1 {
        content: Box<Self>,
    },
    PREC_DYNAMIC {
        value: i32,
        content: Box<Self>,
    },
    PREC_LEFT {
        value: PrecedenceValueJSON,
        content: Box<Self>,
    },
    PREC_RIGHT {
        value: PrecedenceValueJSON,
        content: Box<Self>,
    },
    PREC {
        value: PrecedenceValueJSON,
        content: Box<Self>,
    },
    TOKEN {
        content: Box<Self>,
    },
    IMMEDIATE_TOKEN {
        content: Box<Self>,
    },
    RESERVED {
        context_name: String,
        content: Box<Self>,
    },
}

#[derive(Deserialize, Serialize)]
#[serde(untagged)]
pub(crate) enum PrecedenceValueJSON {
    Integer(i32),
    Name(String),
}

#[derive(Deserialize)]
pub struct GrammarJSON {
    pub name: String,
    rules: Map<String, Value>,
    #[serde(default)]
    precedences: Vec<Vec<RuleJSON>>,
    #[serde(default)]
    conflicts: Vec<Vec<String>>,
    #[serde(default)]
    externals: Vec<RuleJSON>,
    #[serde(default)]
    extras: Vec<RuleJSON>,
    #[serde(default)]
    inline: Vec<String>,
    #[serde(default)]
    supertypes: Vec<String>,
    #[serde(default)]
    word: Option<String>,
    #[serde(default)]
    reserved: Map<String, Value>,
}

pub type ParseGrammarResult<T> = Result<T, ParseGrammarError>;

#[derive(Debug, PartialEq, Eq, Error, Serialize, Deserialize)]
pub enum ParseGrammarError {
    #[error("{0}")]
    Serialization(String),
    #[error("Rules in the `extras` array must not contain empty strings")]
    InvalidExtra,
    #[error("Invalid rule in precedences array. Only strings and symbols are allowed")]
    Unexpected,
    #[error("Reserved word sets must be arrays")]
    InvalidReservedWordSet,
    #[error("Grammar Error: Unexpected rule `{0}` in `token()` call")]
    UnexpectedRule(String),
}

impl From<serde_json::Error> for ParseGrammarError {
    fn from(value: serde_json::Error) -> Self {
        Self::Serialization(value.to_string())
    }
}

/// Check if a rule is referenced within the subtree at `id`.
///
/// This function is used to determine if a variable is used in a given rule,
/// and `is_external` indicates if the rule is an external, and if it is,
/// to not assume that a named symbol that is equal to itself means it's being referenced.
///
/// For example, if we have an external rule **and** a normal rule both called `foo`,
/// `foo` should not be thought of as directly used unless it's used within another rule.
fn is_referenced(pool: &RulePool, id: NodeId, target: StrId, is_external: bool) -> bool {
    match pool.node(id) {
        Node::NamedSymbol(name) => name == target && !is_external,
        Node::Choice(range) | Node::Seq(range) => pool
            .child_slice(range)
            .iter()
            .any(|&c| is_referenced(pool, c, target, false)),
        Node::Metadata { rule, .. } | Node::Reserved { rule, .. } => {
            is_referenced(pool, rule, target, is_external)
        }
        Node::Repeat(inner) => is_referenced(pool, inner, target, false),
        Node::Blank | Node::String(_) | Node::Pattern(..) | Node::Sym { .. } => false,
    }
}

/// Append every `NamedSymbol` id reachable from `id` to `out`. If
/// `skip_top_level` is true, a `NamedSymbol` at the root of the subtree is
/// ignored (used for externals entries, which name themselves).
fn collect_referenced_ids(pool: &RulePool, id: NodeId, skip_top_level: bool, out: &mut Vec<StrId>) {
    match pool.node(id) {
        Node::NamedSymbol(name) => {
            if !skip_top_level {
                out.push(name);
            }
        }
        Node::Choice(range) | Node::Seq(range) => {
            for &c in pool.child_slice(range) {
                collect_referenced_ids(pool, c, false, out);
            }
        }
        Node::Metadata { rule, .. } | Node::Reserved { rule, .. } => {
            collect_referenced_ids(pool, rule, skip_top_level, out);
        }
        Node::Repeat(inner) => collect_referenced_ids(pool, inner, false, out),
        Node::Blank | Node::String(_) | Node::Pattern(..) | Node::Sym { .. } => {}
    }
}

impl PoolGrammar {
    /// Strip unused rules from the grammar and clean up references to them
    /// in the surrounding config. (conflicts, supertypes, inline, extras,
    /// externals, precedences).
    ///
    /// A variable is "used" if it is the start rule, the word token, named in
    /// `extras`/`externals`, or transitively reachable via rule references from
    /// any of the above.
    #[must_use]
    pub fn normalize(mut self, diagnostics: &mut Vec<Diagnostic>) -> Self {
        // Compute the used set via forward DFS from the implicit roots
        // (start rule, word_token, refs in extras and externals).
        //
        // Extras count their top-level `NamedSymbol` as a use (so naming
        // a rule directly in `extras` keeps it), but externals do not (the
        // external entry is the rule itself, not a reference to one).
        let used: FxHashSet<StrId> = {
            let by_name: FxHashMap<StrId, NodeId> =
                self.variables.iter().map(|v| (v.name, v.root)).collect();
            let mut visited: FxHashSet<StrId> = FxHashSet::default();
            let mut stack: Vec<StrId> = Vec::new();
            if let Some(first) = self.variables.first() {
                stack.push(first.name);
            }
            if let Some(word) = self.word_name {
                stack.push(word);
            }
            for &root in &self.extra_roots {
                collect_referenced_ids(&self.pool, root, false, &mut stack);
            }
            for &root in &self.external_roots {
                collect_referenced_ids(&self.pool, root, true, &mut stack);
            }
            // Reserved-word entries are uses of the named rule (the entry
            // names a token to reserve in some context). Top-level
            // `NamedSymbol` counts, same as for extras.
            for set in &self.reserved_sets {
                for &root in &set.roots {
                    collect_referenced_ids(&self.pool, root, false, &mut stack);
                }
            }
            while let Some(name) = stack.pop() {
                if !visited.insert(name) {
                    continue;
                }
                if let Some(&root) = by_name.get(&name) {
                    collect_referenced_ids(&self.pool, root, false, &mut stack);
                }
            }
            visited
        };

        for v in &self.variables {
            if !used.contains(&v.name) {
                continue;
            }
            if !self
                .extra_roots
                .iter()
                .any(|&r| is_referenced(&self.pool, r, v.name, false))
            {
                continue;
            }
            let inner_root = match self.pool.node(v.root) {
                Node::Metadata { rule, .. } => rule,
                _ => v.root,
            };
            let matches_empty = match self.pool.node(inner_root) {
                Node::String(s) => self.pool.resolve(s).is_empty(),
                Node::Pattern(value, _) => {
                    Regex::new(self.pool.resolve(value)).is_ok_and(|reg| reg.is_match(""))
                }
                _ => false,
            };
            if matches_empty {
                diagnostics.push(Diagnostic::EmptyStringMatch(
                    self.pool.resolve(v.name).to_string(),
                ));
            }
        }

        // Drop unused variables and clean up any references to them in the
        // surrounding grammar config.
        let dropped: Vec<StrId> = self
            .variables
            .iter()
            .filter(|v| !used.contains(&v.name))
            .map(|v| v.name)
            .collect();
        self.variables.retain(|v| used.contains(&v.name));
        for &name in &dropped {
            self.conflict_names.retain(|c| !c.contains(&name));
            self.supertype_names.retain(|&s| s != name);
            self.inline_names.retain(|&s| s != name);
            let pool = &self.pool;
            self.extra_roots
                .retain(|&r| !is_referenced(pool, r, name, true));
            self.external_roots
                .retain(|&r| !is_referenced(pool, r, name, true));
            let name_str = pool.resolve(name);
            self.precedence_orderings.retain(|o| {
                !o.iter()
                    .any(|e| matches!(e, PrecedenceEntry::Symbol(s) if s == name_str))
            });
            // Prune entries but keep the context: an intentionally-empty
            // reserved-word context is a meaningful marker that rule bodies
            // may reference by name.
            for set in &mut self.reserved_sets {
                set.roots.retain(|&r| !is_referenced(pool, r, name, false));
            }
        }

        self
    }
}

pub fn parse_grammar(
    input: &str,
    diagnostics: &mut Vec<Diagnostic>,
) -> ParseGrammarResult<PoolGrammar> {
    let grammar_json = serde_json::from_str::<GrammarJSON>(input)?;
    let mut pool = RulePool::default();

    let extra_roots =
        grammar_json
            .extras
            .into_iter()
            .try_fold(Vec::<NodeId>::new(), |mut acc, item| {
                let root = parse_rule(item, false, &mut pool, diagnostics)?;
                if let Node::String(s) = pool.node(root)
                    && pool.resolve(s).is_empty()
                {
                    Err(ParseGrammarError::InvalidExtra)?;
                }
                acc.push(root);
                ParseGrammarResult::Ok(acc)
            })?;

    let external_roots = grammar_json
        .externals
        .into_iter()
        .map(|e| parse_rule(e, false, &mut pool, diagnostics))
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    let mut precedence_orderings = Vec::with_capacity(grammar_json.precedences.len());
    for list in grammar_json.precedences {
        let mut ordering = Vec::with_capacity(list.len());
        for entry in list {
            ordering.push(match entry {
                RuleJSON::STRING { value } => PrecedenceEntry::Name(value),
                RuleJSON::SYMBOL { name } => PrecedenceEntry::Symbol(name),
                _ => Err(ParseGrammarError::Unexpected)?,
            });
        }
        precedence_orderings.push(ordering);
    }

    let variables = grammar_json
        .rules
        .into_iter()
        .map(|(name, r)| {
            let name = pool.intern(&name);
            Ok(PoolVariable {
                name,
                root: parse_rule(serde_json::from_value(r)?, false, &mut pool, diagnostics)?,
            })
        })
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    let reserved_sets = grammar_json
        .reserved
        .into_iter()
        .map(|(name, rule_values)| {
            let Value::Array(rule_values) = rule_values else {
                Err(ParseGrammarError::InvalidReservedWordSet)?
            };

            let name = pool.intern(&name);
            let mut roots = Vec::with_capacity(rule_values.len());
            for value in rule_values {
                roots.push(parse_rule(
                    serde_json::from_value(value)?,
                    false,
                    &mut pool,
                    diagnostics,
                )?);
            }
            Ok(PoolReservedSet { name, roots })
        })
        .collect::<ParseGrammarResult<Vec<_>>>()?;

    let supertype_names = grammar_json
        .supertypes
        .iter()
        .map(|s| pool.intern(s))
        .collect();
    let conflict_names = grammar_json
        .conflicts
        .iter()
        .map(|c| c.iter().map(|n| pool.intern(n)).collect())
        .collect();
    let inline_names = grammar_json.inline.iter().map(|n| pool.intern(n)).collect();
    let word_name = grammar_json.word.as_deref().map(|w| pool.intern(w));

    let grammar = PoolGrammar {
        pool,
        name: grammar_json.name,
        variables,
        external_roots,
        extra_roots,
        reserved_sets,
        supertype_names,
        conflict_names,
        inline_names,
        word_name,
        precedence_orderings,
    }
    .normalize(diagnostics);
    Ok(grammar)
}

fn parse_rule(
    json: RuleJSON,
    is_token: bool,
    pool: &mut RulePool,
    diagnostics: &mut Vec<Diagnostic>,
) -> ParseGrammarResult<NodeId> {
    match json {
        RuleJSON::ALIAS {
            content,
            value,
            named,
        } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let value = pool.intern(&value);
            Ok(pool.alias(content, value, named))
        }
        RuleJSON::BLANK => Ok(pool.blank()),
        RuleJSON::STRING { value } => Ok(pool.string(&value)),
        RuleJSON::PATTERN { value, flags } => {
            let processed_flags = flags.map_or(String::new(), |f| {
                f.matches(|c| {
                    if c == 'i' {
                        true
                    } else {
                        // silently ignore unicode flags
                        if c != 'u' && c != 'v' {
                            diagnostics.push(Diagnostic::UnsupportedRegexFlag {
                                flag: c,
                                pattern: value.clone(),
                            });
                        }
                        false
                    }
                })
                .collect()
            });
            Ok(pool.pattern(&value, &processed_flags))
        }
        RuleJSON::SYMBOL { name } => {
            if is_token {
                Err(ParseGrammarError::UnexpectedRule(name))?
            } else {
                Ok(pool.named_symbol(&name))
            }
        }
        RuleJSON::CHOICE { members } => {
            let members = members
                .into_iter()
                .map(|m| parse_rule(m, is_token, pool, diagnostics))
                .collect::<ParseGrammarResult<Vec<_>>>()?;
            Ok(pool.choice(&members))
        }
        RuleJSON::FIELD { content, name } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let name = pool.intern(&name);
            Ok(pool.field(name, content))
        }
        RuleJSON::SEQ { members } => {
            let members = members
                .into_iter()
                .map(|m| parse_rule(m, is_token, pool, diagnostics))
                .collect::<ParseGrammarResult<Vec<_>>>()?;
            Ok(pool.seq(&members))
        }
        RuleJSON::REPEAT1 { content } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            Ok(pool.repeat(content))
        }
        RuleJSON::REPEAT { content } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let repeat = pool.repeat(content);
            let blank = pool.blank();
            Ok(pool.choice(&[repeat, blank]))
        }
        RuleJSON::PREC { value, content } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let value = value.intern(pool);
            Ok(pool.prec(value, content))
        }
        RuleJSON::PREC_LEFT { value, content } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let value = value.intern(pool);
            Ok(pool.prec_left(value, content))
        }
        RuleJSON::PREC_RIGHT { value, content } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let value = value.intern(pool);
            Ok(pool.prec_right(value, content))
        }
        RuleJSON::PREC_DYNAMIC { value, content } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            Ok(pool.prec_dynamic(value, content))
        }
        RuleJSON::RESERVED {
            content,
            context_name,
        } => {
            let content = parse_rule(*content, is_token, pool, diagnostics)?;
            let ctx = pool.intern(&context_name);
            Ok(pool.reserved(content, ctx))
        }
        RuleJSON::TOKEN { content } => {
            let content = parse_rule(*content, true, pool, diagnostics)?;
            Ok(pool.token(content))
        }
        RuleJSON::IMMEDIATE_TOKEN { content } => {
            let content = parse_rule(*content, true, pool, diagnostics)?;
            Ok(pool.immediate_token(content))
        }
    }
}

impl PrecedenceValueJSON {
    fn intern(self, pool: &mut RulePool) -> Prec {
        match self {
            Self::Integer(i) => Prec::Integer(i),
            Self::Name(n) => Prec::Name(pool.intern(&n)),
        }
    }
}

// --- transitional Rule-based normalize (nativedsl frontend only; deleted
// when nativedsl builds pools directly) ---

/// [`is_referenced`] over a [`Rule`] tree.
#[cfg(feature = "nativedsl")]
fn rule_is_referenced(rule: &Rule, target: &str, is_external: bool) -> bool {
    match rule {
        Rule::NamedSymbol(name) => name == target && !is_external,
        Rule::Choice(rules) | Rule::Seq(rules) => {
            rules.iter().any(|r| rule_is_referenced(r, target, false))
        }
        Rule::Metadata { rule, .. } | Rule::Reserved { rule, .. } => {
            rule_is_referenced(rule, target, is_external)
        }
        Rule::Repeat(inner) => rule_is_referenced(inner, target, false),
        Rule::Blank | Rule::String(_) | Rule::Pattern(_, _) | Rule::Symbol(_) => false,
    }
}

/// [`collect_referenced_ids`] over a [`Rule`] tree.
#[cfg(feature = "nativedsl")]
fn collect_referenced_names<'a>(rule: &'a Rule, skip_top_level: bool, out: &mut Vec<&'a str>) {
    match rule {
        Rule::NamedSymbol(name) => {
            if !skip_top_level {
                out.push(name.as_str());
            }
        }
        Rule::Choice(rules) | Rule::Seq(rules) => {
            for r in rules {
                collect_referenced_names(r, false, out);
            }
        }
        Rule::Metadata { rule, .. } | Rule::Reserved { rule, .. } => {
            collect_referenced_names(rule, skip_top_level, out);
        }
        Rule::Repeat(inner) => collect_referenced_names(inner, false, out),
        Rule::Blank | Rule::String(_) | Rule::Pattern(_, _) | Rule::Symbol(_) => {}
    }
}

#[cfg(feature = "nativedsl")]
impl InputGrammar {
    /// [`PoolGrammar::normalize`] over a [`Rule`]-based grammar.
    #[must_use]
    pub fn normalize(mut self, diagnostics: &mut Vec<Diagnostic>) -> Self {
        let used: FxHashSet<String> = {
            let by_name: FxHashMap<&str, &Rule> = self
                .variables
                .iter()
                .map(|v| (v.name.as_str(), &v.rule))
                .collect();
            let mut visited: FxHashSet<&str> = FxHashSet::default();
            let mut stack: Vec<&str> = Vec::new();
            if let Some(first) = self.variables.first() {
                stack.push(first.name.as_str());
            }
            if let Some(word) = self.word_token.as_deref() {
                stack.push(word);
            }
            for rule in &self.extra_symbols {
                collect_referenced_names(rule, false, &mut stack);
            }
            for rule in &self.external_tokens {
                collect_referenced_names(rule, true, &mut stack);
            }
            for ctx in &self.reserved_words {
                for rule in &ctx.reserved_words {
                    collect_referenced_names(rule, false, &mut stack);
                }
            }
            while let Some(name) = stack.pop() {
                if !visited.insert(name) {
                    continue;
                }
                if let Some(rule) = by_name.get(name) {
                    collect_referenced_names(rule, false, &mut stack);
                }
            }
            visited.into_iter().map(String::from).collect()
        };

        for v in &self.variables {
            if !used.contains(v.name.as_str()) {
                continue;
            }
            if !self
                .extra_symbols
                .iter()
                .any(|r| rule_is_referenced(r, &v.name, false))
            {
                continue;
            }
            let inner_rule = match &v.rule {
                Rule::Metadata { rule, .. } => rule.as_ref(),
                other => other,
            };
            let matches_empty = match inner_rule {
                Rule::String(s) => s.is_empty(),
                Rule::Pattern(value, _) => Regex::new(value).is_ok_and(|reg| reg.is_match("")),
                _ => false,
            };
            if matches_empty {
                diagnostics.push(Diagnostic::EmptyStringMatch(v.name.clone()));
            }
        }

        let dropped: Vec<String> = self
            .variables
            .iter()
            .filter(|v| !used.contains(v.name.as_str()))
            .map(|v| v.name.clone())
            .collect();
        self.variables.retain(|v| used.contains(v.name.as_str()));
        for name in &dropped {
            self.expected_conflicts.retain(|r| !r.contains(name));
            self.supertype_symbols.retain(|r| r != name);
            self.variables_to_inline.retain(|r| r != name);
            self.extra_symbols
                .retain(|r| !rule_is_referenced(r, name, true));
            self.external_tokens
                .retain(|r| !rule_is_referenced(r, name, true));
            self.precedence_orderings.retain(|r| {
                !r.iter()
                    .any(|e| matches!(e, PrecedenceEntry::Symbol(s) if s == name))
            });
            for ctx in &mut self.reserved_words {
                ctx.reserved_words
                    .retain(|r| !rule_is_referenced(r, name, false));
            }
        }

        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rules::Rule;

    #[test]
    fn test_parse_grammar() {
        let grammar = parse_grammar(
            r#"{
            "name": "my_lang",
            "rules": {
                "file": {
                    "type": "REPEAT1",
                    "content": {
                        "type": "SYMBOL",
                        "name": "statement"
                    }
                },
                "statement": {
                    "type": "STRING",
                    "value": "foo"
                }
            }
        }"#,
            &mut Vec::new(),
        )
        .unwrap();

        assert_eq!(grammar.name, "my_lang");
        let names: Vec<&str> = grammar
            .variables
            .iter()
            .map(|v| grammar.pool.resolve(v.name))
            .collect();
        assert_eq!(names, ["file", "statement"]);
        assert_eq!(
            grammar.pool.rule(grammar.variables[0].root),
            Rule::repeat(Rule::NamedSymbol("statement".to_string()))
        );
        assert_eq!(
            grammar.pool.rule(grammar.variables[1].root),
            Rule::String("foo".to_string())
        );
    }
}
