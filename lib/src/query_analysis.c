#include "./query_analysis.h"
#include "./alloc.h"
#include "./language.h"

#include <stdlib.h>

// The analysis works out from the parse table which children the nodes of each
// symbol can have. A node is built in a sequence of parse states: the parser
// moves to another state after each child, and the state after the last child
// reduces the node. So walking back from the symbol's reductions, along the
// transitions into each state, finds the symbol's states, and the symbols of
// those transitions are its children. A state can also be entered from states
// of other productions, so the walk can find children that no node has, which
// only makes the analysis accept more.

// A position in a production: the state that a node is in after `child_index`
// children of the production `production_id`.
typedef struct {
  TSStateId state;
  uint16_t child_index;
  uint16_t production_id;
} QueryAnalysisPosition;

// The symbol that the production `production_id` has at `child_index`.
typedef struct {
  uint16_t production_id;
  uint16_t child_index;
  TSSymbol symbol;
} QueryAnalysisChild;

// A child's symbol, and the field that it is in, or zero for none.
typedef struct {
  TSSymbol symbol;
  TSFieldId field;
} QueryAnalysisEntry;

typedef Array(QueryAnalysisEntry) QueryAnalysisEntries;

// A transition from one parse state to another on a symbol.
typedef struct {
  TSStateId from;
  TSStateId to;
  TSSymbol symbol;
} QueryAnalysisTransition;

// A transition into a known parse state: the state that it comes from, and its
// symbol.
typedef struct {
  TSStateId state;
  TSSymbol symbol;
} QueryAnalysisPredecessor;

// How much the analysis needs to know about the nodes of a symbol.
typedef enum {
  QueryAnalysisNeedNothing,
  // The children that a node of the symbol can have: the query has the symbol
  // as a parent, or the symbol is a repetition that a pattern with several
  // top-level nodes can start inside, or a node type that can be the parent of
  // a child whose parent the query does not name, or of such a repetition, or
  // it is hidden inside a needed symbol.
  QueryAnalysisNeedChildren,
  // The children, and which named children can come first and last: the query
  // has the symbol as a parent with an anchored first or last child, or the
  // symbol is hidden inside such a parent's symbol.
  QueryAnalysisNeedEnds,
} QueryAnalysisNeed;

// What the analysis works out about the nodes of one symbol: a node type that
// the analysis needs as a parent, or a hidden symbol, whose children become
// its parent's.
typedef struct {
  TSSymbol symbol;
  // The positions where the symbol's productions are reduced, each once.
  Array(QueryAnalysisPosition) reductions;
  // The children of the symbol's productions, sorted by production, then
  // position.
  Array(QueryAnalysisChild) direct;
  // The node types, and hidden supertypes, that a node of the symbol can have
  // as children, sorted.
  QueryAnalysisEntries children;
  // The hidden children of the symbol's productions, sorted.
  QueryAnalysisEntries hidden;
  // The fields that one node of the symbol can have more than one child in,
  // sorted.
  Array(TSFieldId) repeatable_fields;
  // The named children, and hidden supertypes, that can come first and last
  // among a node's named children, sorted.
  QueryAnalysisEntries first_named;
  QueryAnalysisEntries last_named;
  // The most visible nodes that a hidden node of the symbol can stand for: 0,
  // 1, or 2 for more than one.
  uint8_t max_visible;
  // Whether a hidden node of the symbol can contain no named node at all.
  bool may_be_unnamed;
  QueryAnalysisNeed need;
} QueryAnalysisSummary;

// Raises `summary`'s need to `need`, and returns whether that is more than
// before.
static bool query_analysis_raise_need(QueryAnalysisSummary *summary, QueryAnalysisNeed need) {
  if (summary->need >= need) return false;
  summary->need = need;
  return true;
}

// A set of 64-bit keys, by open addressing with linear probing. Zero marks an
// empty slot, so each key is stored plus one.
typedef struct {
  uint64_t *slots;
  uint32_t capacity;
  uint32_t size;
} QueryAnalysisSet;

// The state of one call to `ts_query_analyze`.
typedef struct {
  const TSLanguage *language;
  const QueryTokenList *tokens;
  const QueryAst *ast;
  const QueryAstIds *ids;
  QueryProgram *program;
  // Receives a description of the impossible pattern that stops the analysis.
  QueryDiagnostic *error;
  Array(QueryAnalysisSummary) summaries;
  // For each symbol, including aliases, the index of its summary, or -1.
  Array(int32_t) summary_index;
  // For each symbol, whether it, or one of its aliases, has a summary.
  Array(bool) has_summary;
  // For each symbol, whether it is an extra, which can be a child of any
  // non-terminal.
  Array(bool) is_extra;
  // Whether one of the extras is named, so that it can stand for `(_)`.
  bool has_named_extra;
  // The visible extras, as children without a field.
  QueryAnalysisEntries extras;
  // The predecessors of each parse state `s`, at `predecessors[o[s]..o[s + 1]]`
  // where `o` is `predecessor_offsets`.
  Array(uint32_t) predecessor_offsets;
  Array(QueryAnalysisPredecessor) predecessors;
  // The positions and children found so far, each keyed with its summary's
  // index.
  QueryAnalysisSet positions;
  QueryAnalysisSet children;
  // Whether the query has a pattern with several top-level nodes, and one
  // whose match can start inside a repetition without fitting inside it.
  bool has_non_rooted_pattern;
  bool needs_parents;
  // The summaries whose children are still to be found.
  Array(uint32_t) pending;
  // The summaries that have the symbol of a summary `s` as a hidden child, at
  // `parents[parent_offsets[s]..parent_offsets[s + 1]]`.
  Array(uint32_t) parent_offsets;
  Array(uint32_t) parents;
  // The summaries that `can_continue` has reached in its current search.
  Array(bool) visited;
  // What some visible node can have: the union of the summaries of the node
  // types that the nodes of a non-terminal can have. Its `reductions` keeps
  // one, since only whether there are any matters. Its `need` is for children
  // whose parent the query does not name: the children of `(_ ...)`, and the
  // parts of a pattern's top level that have a field.
  QueryAnalysisSummary any_parent;
} QueryAnalyzer;

// Adds `key` to `self`, and returns whether it was new.
static bool query_analysis_set_add(QueryAnalysisSet *self, uint64_t key) {
  if (2 * (self->size + 1) > self->capacity) {
    QueryAnalysisSet grown = {.capacity = self->capacity ? 2 * self->capacity : 1024};
    grown.slots = ts_calloc(grown.capacity, sizeof(uint64_t));
    for (uint32_t i = 0; i < self->capacity; i++) {
      if (self->slots[i]) query_analysis_set_add(&grown, self->slots[i] - 1);
    }
    ts_free(self->slots);
    *self = grown;
  }
  const uint64_t stored = key + 1;
  // Multiplying by 2^64 divided by the golden ratio spreads nearby keys apart.
  uint32_t i = (uint32_t)((stored * 0x9e3779b97f4a7c15ull) >> 32) & (self->capacity - 1);
  while (self->slots[i]) {
    if (self->slots[i] == stored) return false;
    i = (i + 1) & (self->capacity - 1);
  }
  self->slots[i] = stored;
  self->size++;
  return true;
}

static int query_analysis_compare_u16(const void *a, const void *b) {
  const uint16_t left = *(const uint16_t *)a;
  const uint16_t right = *(const uint16_t *)b;
  return left < right ? -1 : left > right;
}

static int query_analysis_compare_entries(const void *a, const void *b) {
  const QueryAnalysisEntry *left = a;
  const QueryAnalysisEntry *right = b;
  if (left->symbol != right->symbol) return left->symbol < right->symbol ? -1 : 1;
  return left->field < right->field ? -1 : left->field > right->field;
}

static int query_analysis_compare_reductions(const void *a, const void *b) {
  const QueryAnalysisPosition *left = a;
  const QueryAnalysisPosition *right = b;
  if (left->production_id != right->production_id) return left->production_id < right->production_id ? -1 : 1;
  if (left->child_index != right->child_index) return left->child_index < right->child_index ? -1 : 1;
  return left->state < right->state ? -1 : left->state > right->state;
}

static int query_analysis_compare_children(const void *a, const void *b) {
  const QueryAnalysisChild *left = a;
  const QueryAnalysisChild *right = b;
  if (left->production_id != right->production_id) return left->production_id < right->production_id ? -1 : 1;
  if (left->child_index != right->child_index) return left->child_index < right->child_index ? -1 : 1;
  return left->symbol < right->symbol ? -1 : left->symbol > right->symbol;
}

// Sorts `array` with `compare`, and removes duplicates.
#define query_analysis_sort_unique(array, compare) do {                             \
    if ((array)->size > 1) {                                                        \
      qsort((array)->contents, (array)->size, sizeof(*(array)->contents), compare); \
      uint32_t kept = 1;                                                            \
      for (uint32_t i = 1; i < (array)->size; i++) {                                \
        if (compare(&(array)->contents[i], &(array)->contents[kept - 1]) != 0) {    \
          (array)->contents[kept++] = (array)->contents[i];                         \
        }                                                                           \
      }                                                                             \
      (array)->size = kept;                                                         \
    }                                                                               \
  } while (0)

static QueryAnalysisSummary *query_analysis_summary(QueryAnalyzer *self, TSSymbol symbol) {
  const int32_t index = *array_get(&self->summary_index, symbol);
  return index < 0 ? NULL : array_get(&self->summaries, (uint32_t)index);
}

static QueryAnalysisSummary *query_analysis_add_summary(QueryAnalyzer *self, TSSymbol symbol) {
  if (*array_get(&self->summary_index, symbol) < 0) {
    *array_get(&self->summary_index, symbol) = (int32_t)self->summaries.size;
    array_push(&self->summaries, ((QueryAnalysisSummary) {.symbol = symbol}));
  }
  return query_analysis_summary(self, symbol);
}

// Returns the key of a position in a production of `summary`, in the set of
// positions found so far.
static uint64_t query_analysis_position_key(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  QueryAnalysisPosition position
) {
  const uint64_t index = (uint64_t)*array_get(&self->summary_index, summary->symbol);
  return index << 48 | (uint64_t)position.state << 32 | (uint64_t)position.child_index << 16 | position.production_id;
}

// Returns the key of a child in a production of `summary`, in the set of
// children found so far.
static uint64_t query_analysis_child_key(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  QueryAnalysisChild child
) {
  const uint64_t index = (uint64_t)*array_get(&self->summary_index, summary->symbol);
  return index << 48 | (uint64_t)child.production_id << 32 | (uint64_t)child.child_index << 16 | child.symbol;
}

// Returns `entry` in `field`, unless it has a field of its own, since a node's
// field is the innermost one around it.
static QueryAnalysisEntry query_analysis_entry_in_field(QueryAnalysisEntry entry, TSFieldId field) {
  if (!entry.field) entry.field = field;
  return entry;
}

// Returns the field of a production's child, or zero. Inherited fields belong
// to the children of a hidden child, and come from its own productions.
static TSFieldId query_analysis_field_at(const TSLanguage *language, uint16_t production_id, uint16_t child_index) {
  const TSFieldMapEntry *entry, *end;
  ts_language_field_map(language, production_id, &entry, &end);
  for (; entry != end; entry++) {
    if (!entry->inherited && entry->child_index == child_index) return entry->field_id;
  }
  return 0;
}

// Returns whether `symbol` is a repetition, a hidden symbol that the grammar
// adds for a repeated rule.
static bool query_analysis_is_repetition(const TSLanguage *language, TSSymbol symbol) {
  if (symbol < language->token_count || symbol >= language->symbol_count) return false;
  const TSSymbolMetadata metadata = ts_language_symbol_metadata(language, symbol);
  return !metadata.visible && !metadata.named;
}

// Scans the parse table once: where the nodes of each symbol with a summary are
// reduced, the predecessors of each state, and which symbols are extras.
static void query_analysis_scan(QueryAnalyzer *self) {
  const TSLanguage *language = self->language;
  Array(QueryAnalysisTransition) transitions = array_new();
  // For each non-terminal, how many of its gotos lead back to the state that
  // they start in, and how many lead elsewhere.
  Array(uint32_t) self_gotos = array_new();
  Array(uint32_t) other_gotos = array_new();
  array_grow_by(&self_gotos, language->symbol_count);
  array_grow_by(&other_gotos, language->symbol_count);

  for (TSStateId state = 1; state < language->state_count; state++) {
    LookaheadIterator iterator = ts_language_lookaheads(language, state);
    while (ts_lookahead_iterator__next(&iterator)) {
      if (iterator.action_count) {
        for (unsigned i = 0; i < iterator.action_count; i++) {
          const TSParseAction *action = &iterator.actions[i];
          if (action->type == TSParseActionTypeReduce) {
            if (!*array_get(&self->has_summary, action->reduce.symbol)) continue;
            const QueryAnalysisPosition reduction = {
              .state = state,
              .child_index = action->reduce.child_count,
              .production_id = action->reduce.production_id,
            };
            const TSSymbol *alias, *alias_end;
            ts_language_aliases_for_symbol(language, action->reduce.symbol, &alias, &alias_end);
            for (; alias < alias_end; alias++) {
              QueryAnalysisSummary *summary = query_analysis_summary(self, *alias);
              if (!summary) continue;
              // A state has the same reduction for each lookahead that it reduces on.
              if (query_analysis_set_add(&self->positions, query_analysis_position_key(self, summary, reduction))) {
                array_push(&summary->reductions, reduction);
              }
            }
          } else if (action->type == TSParseActionTypeShift) {
            if (action->shift.extra) {
              *array_get(&self->is_extra, iterator.symbol) = true;
            } else {
              array_push(&transitions, ((QueryAnalysisTransition) {
                .from = state,
                .to = action->shift.state,
                .symbol = iterator.symbol,
              }));
            }
          }
        }
      } else if (iterator.next_state != 0) {
        if (iterator.next_state == state) {
          (*array_get(&self_gotos, iterator.symbol))++;
        } else {
          (*array_get(&other_gotos, iterator.symbol))++;
        }
        array_push(&transitions, ((QueryAnalysisTransition) {
          .from = state,
          .to = iterator.next_state,
          .symbol = iterator.symbol,
        }));
      }
    }
  }

  // A non-terminal extra can occur in any state, and the parser treats it as
  // an extra where its goto leads back to the state that it started in. A
  // repetition's goto does that too, but only in the few states where it
  // repeats.
  for (TSSymbol symbol = 0; symbol < language->symbol_count; symbol++) {
    if (*array_get(&self_gotos, symbol) > *array_get(&other_gotos, symbol)) {
      *array_get(&self->is_extra, symbol) = true;
    }
    if (!*array_get(&self->is_extra, symbol)) continue;
    const TSSymbolMetadata metadata = ts_language_symbol_metadata(language, symbol);
    if (metadata.named) self->has_named_extra = true;
    if (metadata.visible) array_push(&self->extras, ((QueryAnalysisEntry) {.symbol = language->public_symbol_map[symbol]}));
  }
  array_delete(&self_gotos);
  array_delete(&other_gotos);

  // An extra's goto back to the state that it started in is not a step from one
  // child of a production to the next, unlike a repetition's.
  uint32_t kept = 0;
  for (uint32_t i = 0; i < transitions.size; i++) {
    const QueryAnalysisTransition transition = *array_get(&transitions, i);
    if (transition.from == transition.to && *array_get(&self->is_extra, transition.symbol)) continue;
    *array_get(&transitions, kept) = transition;
    kept++;
  }
  transitions.size = kept;

  // Bucket the transitions by the state that they lead to.
  array_grow_by(&self->predecessor_offsets, language->state_count + 1u);
  for (uint32_t i = 0; i < transitions.size; i++) {
    (*array_get(&self->predecessor_offsets, array_get(&transitions, i)->to + 1u))++;
  }
  for (uint32_t state = 1; state <= language->state_count; state++) {
    *array_get(&self->predecessor_offsets, state) += *array_get(&self->predecessor_offsets, state - 1);
  }
  array_grow_by(&self->predecessors, transitions.size);
  Array(uint32_t) next_slot = array_new();
  array_extend(&next_slot, language->state_count, self->predecessor_offsets.contents);
  for (uint32_t i = 0; i < transitions.size; i++) {
    const QueryAnalysisTransition transition = *array_get(&transitions, i);
    uint32_t *slot = array_get(&next_slot, transition.to);
    *array_get(&self->predecessors, *slot) = (QueryAnalysisPredecessor) {
      .state = transition.from,
      .symbol = transition.symbol,
    };
    (*slot)++;
  }
  array_delete(&next_slot);
  array_delete(&transitions);
}

// Walks back from each reduction of `summary`'s symbol, along the transitions
// into each state, to find its positions and children.
static void query_analysis_walk_back(QueryAnalyzer *self, QueryAnalysisSummary *summary) {
  Array(QueryAnalysisPosition) pending = array_new();
  array_push_all(&pending, &summary->reductions);
  while (pending.size > 0) {
    const QueryAnalysisPosition position = array_pop(&pending);
    if (position.child_index == 0) continue;
    const uint32_t start = *array_get(&self->predecessor_offsets, position.state);
    const uint32_t end = *array_get(&self->predecessor_offsets, position.state + 1u);
    for (uint32_t i = start; i < end; i++) {
      const QueryAnalysisPredecessor *predecessor = array_get(&self->predecessors, i);
      const QueryAnalysisChild child = {
        .production_id = position.production_id,
        .child_index = position.child_index - 1,
        .symbol = predecessor->symbol,
      };
      if (query_analysis_set_add(&self->children, query_analysis_child_key(self, summary, child))) {
        array_push(&summary->direct, child);
      }
      if (child.child_index == 0) continue;
      const QueryAnalysisPosition previous = {
        .state = predecessor->state,
        .child_index = child.child_index,
        .production_id = position.production_id,
      };
      if (query_analysis_set_add(&self->positions, query_analysis_position_key(self, summary, previous))) {
        array_push(&pending, previous);
      }
    }
  }
  array_delete(&pending);
}

// Sorts out the children of `summary`'s productions into the node types and
// hidden symbols that a node of the symbol can have as children.
static void query_analysis_find_children(QueryAnalyzer *self, QueryAnalysisSummary *summary) {
  const TSLanguage *language = self->language;
  query_analysis_sort_unique(&summary->direct, query_analysis_compare_children);
  for (uint32_t i = 0; i < summary->direct.size; i++) {
    const QueryAnalysisChild *child = array_get(&summary->direct, i);
    const TSFieldId field = query_analysis_field_at(language, child->production_id, child->child_index);
    const TSSymbol alias = ts_language_alias_at(language, child->production_id, child->child_index);
    const TSSymbolMetadata metadata = ts_language_symbol_metadata(language, child->symbol);
    if (alias || metadata.visible) {
      const TSSymbol node_type = alias ? alias : language->public_symbol_map[child->symbol];
      array_push(&summary->children, ((QueryAnalysisEntry) {.symbol = node_type, .field = field}));
    } else if (child->symbol >= language->token_count) {
      const QueryAnalysisEntry entry = {.symbol = child->symbol, .field = field};
      if (metadata.supertype) array_push(&summary->children, entry);
      array_push(&summary->hidden, entry);
    }
  }
  // An extra can be between two children of a hidden node, and then gets the
  // field that the hidden node has in its parent.
  if (!ts_language_symbol_metadata(language, summary->symbol).visible) {
    for (uint32_t r = 0; r < summary->reductions.size; r++) {
      if (array_get(&summary->reductions, r)->child_index < 2) continue;
      array_push_all(&summary->children, &self->extras);
      break;
    }
  }
  query_analysis_sort_unique(&summary->children, query_analysis_compare_entries);
  query_analysis_sort_unique(&summary->hidden, query_analysis_compare_entries);
}

// Adds the children of each hidden child to its parent's, until nothing
// changes. A field of the hidden child goes to each of its children that has
// none of its own, since a node's field is the innermost one around it.
static void query_analysis_inherit_children(QueryAnalyzer *self) {
  QueryAnalysisEntries added = array_new();
  for (bool changed = true; changed;) {
    changed = false;
    for (uint32_t s = 0; s < self->summaries.size; s++) {
      QueryAnalysisSummary *summary = array_get(&self->summaries, s);
      array_clear(&added);
      for (uint32_t h = 0; h < summary->hidden.size; h++) {
        const QueryAnalysisEntry entry = *array_get(&summary->hidden, h);
        const QueryAnalysisSummary *hidden = query_analysis_summary(self, entry.symbol);
        for (uint32_t c = 0; c < hidden->children.size; c++) {
          array_push(&added, query_analysis_entry_in_field(*array_get(&hidden->children, c), entry.field));
        }
      }
      if (added.size == 0) continue;
      const uint32_t size = summary->children.size;
      array_extend(&summary->children, added.size, added.contents);
      query_analysis_sort_unique(&summary->children, query_analysis_compare_entries);
      if (summary->children.size != size) changed = true;
    }
  }
  array_delete(&added);
}

// Returns the most visible nodes that a production's child can stand for: one
// for a visible or aliased child, the count in its summary for a hidden
// non-terminal, and none for a hidden token.
static uint8_t query_analysis_child_weight(QueryAnalyzer *self, const QueryAnalysisChild *child) {
  if (ts_language_alias_at(self->language, child->production_id, child->child_index)) return 1;
  if (ts_language_symbol_metadata(self->language, child->symbol).visible) return 1;
  if (child->symbol < self->language->token_count) return 0;
  return query_analysis_summary(self, child->symbol)->max_visible;
}

// Works out how many visible nodes each hidden symbol can stand for, until
// nothing changes: the most that one of its productions has, adding up the
// most that each position's children can stand for.
static void query_analysis_count_visible(QueryAnalyzer *self) {
  for (bool changed = true; changed;) {
    changed = false;
    for (uint32_t s = 0; s < self->summaries.size; s++) {
      QueryAnalysisSummary *summary = array_get(&self->summaries, s);
      unsigned most = 0;
      unsigned production_sum = 0;
      unsigned position_most = 0;
      for (uint32_t i = 0; i < summary->direct.size; i++) {
        const QueryAnalysisChild *child = array_get(&summary->direct, i);
        const QueryAnalysisChild *previous = i > 0 ? array_get(&summary->direct, i - 1) : NULL;
        if (previous && previous->production_id != child->production_id) {
          production_sum += position_most;
          if (production_sum > most) most = production_sum;
          production_sum = 0;
          position_most = 0;
        } else if (previous && previous->child_index != child->child_index) {
          production_sum += position_most;
          position_most = 0;
        }
        const uint8_t weight = query_analysis_child_weight(self, child);
        if (weight > position_most) position_most = weight;
      }
      production_sum += position_most;
      if (production_sum > most) most = production_sum;
      const uint8_t max_visible = most > 2 ? 2 : (uint8_t)most;
      if (max_visible > summary->max_visible) {
        summary->max_visible = max_visible;
        changed = true;
      }
    }
  }
}

// Works out which fields a node of each symbol can have more than one child in:
// a field that a production has at two positions, or at a position whose
// child can stand for several visible nodes, or that is inside a hidden child.
static void query_analysis_find_repeatable_fields(QueryAnalyzer *self) {
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    // A field map marks only some of the fields inside a hidden child as
    // inherited: older generators leave out those inside a supertype.
    for (uint32_t h = 0; h < summary->hidden.size; h++) {
      const QueryAnalysisSummary *hidden = query_analysis_summary(self, array_get(&summary->hidden, h)->symbol);
      for (uint32_t c = 0; c < hidden->children.size; c++) {
        const TSFieldId field = array_get(&hidden->children, c)->field;
        if (field) array_push(&summary->repeatable_fields, field);
      }
    }
    for (uint32_t begin = 0; begin < summary->direct.size;) {
      const uint16_t production_id = array_get(&summary->direct, begin)->production_id;
      uint32_t end = begin + 1;
      while (end < summary->direct.size && array_get(&summary->direct, end)->production_id == production_id) end++;
      const TSFieldMapEntry *entry, *entries_end;
      ts_language_field_map(self->language, production_id, &entry, &entries_end);
      for (; entry != entries_end; entry++) {
        bool repeats = false;
        for (const TSFieldMapEntry *other = entry + 1; other != entries_end; other++) {
          if (other->field_id == entry->field_id) repeats = true;
        }
        for (uint32_t i = begin; i < end && !repeats; i++) {
          const QueryAnalysisChild *child = array_get(&summary->direct, i);
          if (child->child_index == entry->child_index && query_analysis_child_weight(self, child) > 1) repeats = true;
        }
        if (repeats) array_push(&summary->repeatable_fields, entry->field_id);
      }
      begin = end;
    }
    query_analysis_sort_unique(&summary->repeatable_fields, query_analysis_compare_u16);
  }
}

// Adds to `ends` the named child of a production at a position, or the first
// or last named children of the hidden child there, and returns whether a
// node can pass that position without a named child.
static bool query_analysis_add_end(
  QueryAnalyzer *self,
  QueryAnalysisEntries *ends,
  const QueryAnalysisChild *child,
  bool first
) {
  const TSLanguage *language = self->language;
  const TSFieldId field = query_analysis_field_at(language, child->production_id, child->child_index);
  const TSSymbol alias = ts_language_alias_at(language, child->production_id, child->child_index);
  const TSSymbol symbol = alias ? alias : child->symbol;
  const TSSymbolMetadata metadata = ts_language_symbol_metadata(language, symbol);
  if (alias || metadata.visible) {
    if (!metadata.named) return true;
    const TSSymbol node_type = alias ? alias : language->public_symbol_map[symbol];
    array_push(ends, ((QueryAnalysisEntry) {.symbol = node_type, .field = field}));
    return false;
  }
  if (symbol < language->token_count) return true;
  const QueryAnalysisSummary *hidden = query_analysis_summary(self, symbol);
  if (metadata.supertype) array_push(ends, ((QueryAnalysisEntry) {.symbol = symbol, .field = field}));
  const QueryAnalysisEntries *hidden_ends = first ? &hidden->first_named : &hidden->last_named;
  // A symbol that starts or ends with itself, as a repetition does, has its own
  // ends already.
  if (hidden_ends == ends) return hidden->may_be_unnamed;
  for (uint32_t i = 0; i < hidden_ends->size; i++) {
    array_push(ends, query_analysis_entry_in_field(*array_get(hidden_ends, i), field));
  }
  return hidden->may_be_unnamed;
}

// Works out the first and last named children of the symbols that need them,
// and which of those can contain no named node, until nothing changes. A
// production's first named child comes from its first position, or from a
// later one if each position before it can be passed without one.
static void query_analysis_find_ends(QueryAnalyzer *self) {
  // With the reductions sorted by production, each production comes in a run
  // of reductions in different states.
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    if (summary->need != QueryAnalysisNeedEnds) continue;
    query_analysis_sort_unique(&summary->reductions, query_analysis_compare_reductions);
  }

  for (bool changed = true; changed;) {
    changed = false;
    for (uint32_t s = 0; s < self->summaries.size; s++) {
      QueryAnalysisSummary *summary = array_get(&self->summaries, s);
      if (summary->need != QueryAnalysisNeedEnds) continue;
      const uint32_t first_size = summary->first_named.size;
      const uint32_t last_size = summary->last_named.size;
      bool may_be_unnamed = summary->may_be_unnamed;
      uint32_t begin = 0;
      for (uint32_t r = 0; r < summary->reductions.size; r++) {
        const QueryAnalysisPosition *reduction = array_get(&summary->reductions, r);
        if (r > 0) {
          const QueryAnalysisPosition *previous = array_get(&summary->reductions, r - 1);
          if (previous->production_id == reduction->production_id && previous->child_index == reduction->child_index) continue;
        }
        const uint16_t production_id = reduction->production_id;
        const uint16_t child_count = reduction->child_index;
        while (begin < summary->direct.size && array_get(&summary->direct, begin)->production_id < production_id) begin++;
        uint32_t end = begin;
        while (end < summary->direct.size && array_get(&summary->direct, end)->production_id == production_id) end++;

        bool passes_all = true;
        for (uint16_t index = 0; index < child_count && passes_all; index++) {
          bool passes = false;
          for (uint32_t i = begin; i < end; i++) {
            const QueryAnalysisChild *child = array_get(&summary->direct, i);
            if (child->child_index != index) continue;
            if (query_analysis_add_end(self, &summary->first_named, child, true)) passes = true;
          }
          passes_all = passes;
        }
        if (passes_all) may_be_unnamed = true;

        for (uint16_t index = child_count; index > 0; index--) {
          bool passes = false;
          for (uint32_t i = begin; i < end; i++) {
            const QueryAnalysisChild *child = array_get(&summary->direct, i);
            if (child->child_index != index - 1) continue;
            if (query_analysis_add_end(self, &summary->last_named, child, false)) passes = true;
          }
          if (!passes) break;
        }
      }
      query_analysis_sort_unique(&summary->first_named, query_analysis_compare_entries);
      query_analysis_sort_unique(&summary->last_named, query_analysis_compare_entries);
      if (
        summary->first_named.size != first_size ||
        summary->last_named.size != last_size ||
        may_be_unnamed != summary->may_be_unnamed
      ) {
        summary->may_be_unnamed = may_be_unnamed;
        changed = true;
      }
    }
  }
}

// Finds the children of the summaries in `pending`, and then of the hidden
// symbols among those children, which it adds to `pending`.
static void query_analysis_find_pending_children(QueryAnalyzer *self) {
  while (self->pending.size > 0) {
    // `array_get` evaluates its index twice, so the index is popped first.
    const uint32_t index = array_pop(&self->pending);
    QueryAnalysisSummary *summary = array_get(&self->summaries, index);
    query_analysis_walk_back(self, summary);
    query_analysis_find_children(self, summary);
    for (uint32_t h = 0; h < summary->hidden.size; h++) {
      QueryAnalysisSummary *hidden = query_analysis_summary(self, array_get(&summary->hidden, h)->symbol);
      if (query_analysis_raise_need(hidden, QueryAnalysisNeedChildren)) {
        array_push(&self->pending, (uint32_t)*array_get(&self->summary_index, hidden->symbol));
      }
    }
  }
}

// Works out the children of the symbols that the analysis needs, and of the
// hidden symbols inside them, and the first and last named children of the
// symbols that need those. The other summaries are left empty.
static void query_analysis_summarize(QueryAnalyzer *self) {
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    if (array_get(&self->summaries, s)->need != QueryAnalysisNeedNothing) array_push(&self->pending, s);
  }
  query_analysis_find_pending_children(self);
  query_analysis_inherit_children(self);
  query_analysis_count_visible(self);
  query_analysis_find_repeatable_fields(self);

  for (uint32_t s = 0; s < self->summaries.size; s++) {
    if (array_get(&self->summaries, s)->need == QueryAnalysisNeedEnds) array_push(&self->pending, s);
  }
  while (self->pending.size > 0) {
    const uint32_t index = array_pop(&self->pending);
    const QueryAnalysisSummary *summary = array_get(&self->summaries, index);
    for (uint32_t h = 0; h < summary->hidden.size; h++) {
      QueryAnalysisSummary *hidden = query_analysis_summary(self, array_get(&summary->hidden, h)->symbol);
      if (query_analysis_raise_need(hidden, QueryAnalysisNeedEnds)) {
        array_push(&self->pending, (uint32_t)*array_get(&self->summary_index, hidden->symbol));
      }
    }
  }
  query_analysis_find_ends(self);
}

// Builds `any_parent` from the summaries of the visible node types that have
// reductions.
static void query_analysis_build_any_parent(QueryAnalyzer *self) {
  QueryAnalysisSummary *any = &self->any_parent;
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    const QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    if (!ts_language_symbol_metadata(self->language, summary->symbol).visible || summary->reductions.size == 0) continue;
    if (any->reductions.size == 0) array_push(&any->reductions, *array_get(&summary->reductions, 0));
    array_push_all(&any->children, &summary->children);
    array_push_all(&any->repeatable_fields, &summary->repeatable_fields);
    array_push_all(&any->first_named, &summary->first_named);
    array_push_all(&any->last_named, &summary->last_named);
  }
  query_analysis_sort_unique(&any->children, query_analysis_compare_entries);
  query_analysis_sort_unique(&any->repeatable_fields, query_analysis_compare_u16);
  query_analysis_sort_unique(&any->first_named, query_analysis_compare_entries);
  query_analysis_sort_unique(&any->last_named, query_analysis_compare_entries);
}

// Returns whether `summary` is a hidden symbol's. A hidden node's child without
// a field of its own gets the field that the hidden node has in its parent,
// which the summary does not know, so it can be in any field.
static bool query_analysis_is_hidden(const QueryAnalyzer *self, const QueryAnalysisSummary *summary) {
  return summary != &self->any_parent && !ts_language_symbol_metadata(self->language, summary->symbol).visible;
}

// Returns whether `entries`, some children that a node of `summary`'s symbol can
// have, include a `symbol` in `field`, or in any field if `field` is zero. A
// zero `symbol` stands for any named node. An ERROR can be any child, and an
// extra any child of a non-terminal, in any field that some child is in: an
// extra in a field is inside a hidden node in it, whose children include the
// extras.
static bool query_analysis_entries_have(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  const QueryAnalysisEntries *entries,
  TSSymbol symbol,
  TSFieldId field
) {
  if (symbol == ts_builtin_sym_error) return true;
  const bool is_extra = symbol ? *array_get(&self->is_extra, symbol) : self->has_named_extra;
  if (is_extra && !field) return summary->reductions.size > 0;
  const bool is_hidden = query_analysis_is_hidden(self, summary);
  for (uint32_t i = 0; i < entries->size; i++) {
    const QueryAnalysisEntry *entry = array_get(entries, i);
    if (field && entry->field != field && (entry->field || !is_hidden)) continue;
    if (is_extra) return true;
    if (symbol ? entry->symbol == symbol : ts_language_symbol_metadata(self->language, entry->symbol).named) return true;
  }
  return false;
}

// Returns the node type of the named node `node`: the subtype's, in
// `(supertype/subtype ...)`.
static TSSymbol query_analysis_node_type(const QueryAnalyzer *self, uint32_t node) {
  const uint32_t first = node + 1;
  if (first < query_ast_end(self->ast, node) && query_ast_kind(self->ast, first) == QueryAstKindSubtype) {
    return (TSSymbol)query_ast_ids_get(self->ids, first);
  }
  return (TSSymbol)query_ast_ids_get(self->ids, node);
}

// Returns whether `entries`, some children that a node of `summary`'s symbol can
// have, include the node of `atom`, a named or anonymous node, in `field`, or in
// any field if `field` is zero. The node of `(supertype/subtype)` is the
// subtype's, inside the supertype, so it gets the field of either.
static bool query_analysis_entries_have_node(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  const QueryAnalysisEntries *entries,
  uint32_t atom,
  TSFieldId field
) {
  const TSSymbol symbol = (TSSymbol)query_ast_ids_get(self->ids, atom);
  const TSSymbol node_type = query_analysis_node_type(self, atom);
  if (node_type != symbol && !query_analysis_entries_have(self, summary, entries, symbol, 0)) return false;
  return query_analysis_entries_have(self, summary, entries, node_type, field);
}

// Returns whether a match can skip `pattern`: it has a `?` or `*`, or it is a
// group whose patterns a match can all skip, or an alternation with a branch
// that a match can skip.
static bool query_analysis_is_optional(QueryAnalyzer *self, uint32_t pattern) {
  for (uint32_t node = pattern + 1; node < query_ast_end(self->ast, pattern); node = query_ast_end(self->ast, node)) {
    if (query_ast_kind(self->ast, node) != QueryAstKindQuantifier) continue;
    const QueryTokenKind kind = query_token_list_kind(self->tokens, query_ast_token(self->ast, node));
    if (kind == QueryTokenKindQuestion || kind == QueryTokenKindStar) return true;
  }
  uint32_t atom = pattern + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) atom++;
  const QueryAstKind atom_kind = query_ast_kind(self->ast, atom);
  if (atom_kind != QueryAstKindAlternation && atom_kind != QueryAstKindGroup) return false;
  for (uint32_t node = atom + 1; node < query_ast_end(self->ast, atom); node = query_ast_end(self->ast, node)) {
    if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
    const bool is_optional = query_analysis_is_optional(self, node);
    if (atom_kind == QueryAstKindGroup && !is_optional) return false;
    if (atom_kind == QueryAstKindAlternation && is_optional) return true;
  }
  return atom_kind == QueryAstKindGroup;
}

static bool query_analysis_is_possible(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  uint32_t pattern,
  TSFieldId field
);

// Returns whether each pattern that a match cannot skip, among the siblings
// from `first` up to `end`, can be a child of a node of `summary`'s symbol.
static bool query_analysis_are_possible(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  uint32_t first,
  uint32_t end,
  TSFieldId field
) {
  for (uint32_t node = first; node < end; node = query_ast_end(self->ast, node)) {
    if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
    if (query_analysis_is_optional(self, node)) continue;
    if (!query_analysis_is_possible(self, summary, node, field)) return false;
  }
  return true;
}

// Returns whether `pattern` can match a child of a node of `summary`'s symbol,
// in `field` unless the pattern has its own: some node, for an alternation, and
// each node that a match cannot skip, for a group.
static bool query_analysis_is_possible(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  uint32_t pattern,
  TSFieldId field
) {
  uint32_t atom = pattern + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) {
    field = (TSFieldId)query_ast_ids_get(self->ids, atom);
    atom++;
  }
  switch (query_ast_kind(self->ast, atom)) {
    case QueryAstKindNamedNode:
    case QueryAstKindAnonymousNode:
      return query_analysis_entries_have_node(self, summary, &summary->children, atom, field);
    case QueryAstKindWildcard:
      if (!field || query_analysis_is_hidden(self, summary)) return summary->reductions.size > 0;
      for (uint32_t i = 0; i < summary->children.size; i++) {
        if (array_get(&summary->children, i)->field == field) return true;
      }
      return false;
    case QueryAstKindAlternation:
      for (uint32_t node = atom + 1; node < query_ast_end(self->ast, atom); node = query_ast_end(self->ast, node)) {
        if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
        if (query_analysis_is_possible(self, summary, node, field)) return true;
      }
      return false;
    case QueryAstKindGroup:
      return query_analysis_are_possible(self, summary, atom + 1, query_ast_end(self->ast, atom), field);
    default:
      // A MISSING node can stand in for any node.
      return true;
  }
}

// Describes `node` as the impossible pattern that stops the analysis, and
// returns false.
static bool query_analysis_fail_at(QueryAnalyzer *self, uint32_t node) {
  *self->error = (QueryDiagnostic) {
    .span = query_token_list_span(self->tokens, query_ast_token(self->ast, node)),
    .kind = QueryDiagnosticKindImpossiblePattern,
  };
  return false;
}

// Checks that each part of `pattern` can be a child of a node of `summary`'s
// symbol, in `field` unless the pattern has its own: every branch of an
// alternation, and every node of a group, even one that a match can skip.
static bool query_analysis_check_child(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  uint32_t pattern,
  TSFieldId field
) {
  uint32_t atom = pattern + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) {
    field = (TSFieldId)query_ast_ids_get(self->ids, atom);
    atom++;
  }
  const QueryAstKind kind = query_ast_kind(self->ast, atom);
  if (kind != QueryAstKindAlternation && kind != QueryAstKindGroup) {
    return query_analysis_is_possible(self, summary, pattern, field) || query_analysis_fail_at(self, pattern);
  }
  for (uint32_t node = atom + 1; node < query_ast_end(self->ast, atom); node = query_ast_end(self->ast, node)) {
    if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
    if (!query_analysis_check_child(self, summary, node, field)) return false;
  }
  return true;
}

// Checks that the named child that each branch of `pattern` matches can be the
// first (or last) named child of a node of `summary`'s symbol, in `field`
// unless the pattern has its own. Only named nodes are checked.
static bool query_analysis_check_end(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  uint32_t pattern,
  TSFieldId field,
  bool first
) {
  uint32_t atom = pattern + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) {
    field = (TSFieldId)query_ast_ids_get(self->ids, atom);
    atom++;
  }
  const QueryAnalysisEntries *ends = first ? &summary->first_named : &summary->last_named;
  switch (query_ast_kind(self->ast, atom)) {
    case QueryAstKindNamedNode:
      return query_analysis_entries_have_node(self, summary, ends, atom, field) || query_analysis_fail_at(self, pattern);
    case QueryAstKindAlternation:
      for (uint32_t node = atom + 1; node < query_ast_end(self->ast, atom); node = query_ast_end(self->ast, node)) {
        if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
        if (!query_analysis_check_end(self, summary, node, field, first)) return false;
      }
      return true;
    default:
      return true;
  }
}

static bool query_analysis_field_repeats(const QueryAnalysisSummary *summary, TSFieldId field) {
  for (uint32_t i = 0; i < summary->repeatable_fields.size; i++) {
    if (*array_get(&summary->repeatable_fields, i) == field) return true;
  }
  return false;
}

// Returns whether the parser can complete a node of `summary`'s symbol, a
// non-terminal: the walk back from one of its reductions reached a first child,
// or it has an empty production. A rule that precedence always makes lose to
// another is never completed.
static bool query_analysis_is_completed(const QueryAnalysisSummary *summary) {
  for (uint32_t i = 0; i < summary->direct.size; i++) {
    if (array_get(&summary->direct, i)->child_index == 0) return true;
  }
  for (uint32_t i = 0; i < summary->reductions.size; i++) {
    if (array_get(&summary->reductions, i)->child_index == 0) return true;
  }
  return false;
}

// Checks the children of the named node `node`, the atom of `pattern`: against
// what a node of its type can have, or for `(_)`, what some visible node can
// have. Supertypes and ERROR are not checked.
static bool query_analysis_check_parent(QueryAnalyzer *self, uint32_t pattern, uint32_t node) {
  const TSLanguage *language = self->language;
  const QueryAst *ast = self->ast;
  const TSSymbol symbol = query_analysis_node_type(self, node);
  if (symbol == ts_builtin_sym_error) return true;
  if (ts_language_symbol_metadata(language, symbol).supertype) return true;
  const uint32_t first = node + 1;
  const uint32_t end = query_ast_end(ast, node);

  // The children that the pattern lists, and whether the first and last are
  // anchored.
  uint32_t first_child = 0;
  uint32_t last_child = 0;
  bool has_children = false;
  bool anchors_first = false;
  bool anchor_since_last = false;
  for (uint32_t child = first; child < end; child = query_ast_end(ast, child)) {
    const QueryAstKind kind = query_ast_kind(ast, child);
    if (kind == QueryAstKindAnchor) {
      if (!has_children) anchors_first = true;
      anchor_since_last = true;
    } else if (kind == QueryAstKindPattern) {
      if (!has_children) first_child = child;
      has_children = true;
      last_child = child;
      anchor_since_last = false;
    }
  }
  if (!has_children) return true;

  const QueryAnalysisSummary *summary = symbol ? query_analysis_summary(self, symbol) : &self->any_parent;
  if (symbol >= language->token_count && symbol < language->symbol_count && !query_analysis_is_completed(summary)) {
    return query_analysis_fail_at(self, pattern);
  }

  // Every child must be possible, and a field that two required children
  // share must be able to repeat.
  Array(TSFieldId) fields = array_new();
  bool possible = true;
  for (uint32_t child = first; child < end && possible; child = query_ast_end(ast, child)) {
    if (query_ast_kind(ast, child) != QueryAstKindPattern) continue;
    if (!query_analysis_check_child(self, summary, child, 0)) {
      possible = false;
    } else if (!query_analysis_is_optional(self, child) && query_ast_kind(ast, child + 1) == QueryAstKindField) {
      const TSFieldId field = (TSFieldId)query_ast_ids_get(self->ids, child + 1);
      for (uint32_t i = 0; i < fields.size && possible; i++) {
        if (*array_get(&fields, i) == field && !query_analysis_field_repeats(summary, field)) {
          possible = query_analysis_fail_at(self, child);
        }
      }
      array_push(&fields, field);
    }
  }
  array_delete(&fields);
  if (!possible) return false;

  // A child after a leading anchor must be able to be the first named child,
  // and one before a trailing anchor the last.
  if (anchors_first && !query_analysis_check_end(self, summary, first_child, 0, true)) return false;
  if (anchor_since_last && !query_analysis_check_end(self, summary, last_child, 0, false)) return false;
  return true;
}

// Returns whether a match of `pattern` can start with a child of a node of
// `summary`'s symbol, in `field` unless the pattern has its own: for a group,
// with one of the nodes up to the first that the match cannot skip.
static bool query_analysis_can_start(
  QueryAnalyzer *self,
  const QueryAnalysisSummary *summary,
  uint32_t pattern,
  TSFieldId field
) {
  uint32_t atom = pattern + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) {
    field = (TSFieldId)query_ast_ids_get(self->ids, atom);
    atom++;
  }
  const QueryAstKind kind = query_ast_kind(self->ast, atom);
  if (kind != QueryAstKindAlternation && kind != QueryAstKindGroup) {
    return query_analysis_is_possible(self, summary, pattern, field);
  }
  for (uint32_t node = atom + 1; node < query_ast_end(self->ast, atom); node = query_ast_end(self->ast, node)) {
    if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
    if (query_analysis_can_start(self, summary, node, field)) return true;
    if (kind == QueryAstKindGroup && !query_analysis_is_optional(self, node)) return false;
  }
  return false;
}

// Returns whether a match of `pattern` that starts inside a node of the
// repetition `summaries[repetition]` can be among the children of the visible
// node around it: of a symbol that has it as a hidden child, or has such a
// hidden symbol as one, and so on.
static bool query_analysis_can_continue(QueryAnalyzer *self, uint32_t repetition, uint32_t pattern) {
  for (uint32_t s = 0; s < self->visited.size; s++) *array_get(&self->visited, s) = false;
  Array(uint32_t) stack = array_new();
  array_push(&stack, repetition);
  bool result = false;
  while (stack.size > 0 && !result) {
    const uint32_t s = array_pop(&stack);
    if (*array_get(&self->visited, s)) continue;
    *array_get(&self->visited, s) = true;
    const QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    result = query_analysis_is_possible(self, summary, pattern, 0);
    if (ts_language_symbol_metadata(self->language, summary->symbol).visible) continue;
    const uint32_t end = *array_get(&self->parent_offsets, s + 1);
    for (uint32_t p = *array_get(&self->parent_offsets, s); p < end; p++) {
      array_push(&stack, *array_get(&self->parents, p));
    }
  }
  array_delete(&stack);
  return result;
}

// Finds the children of the visible node types, and records which summaries
// have each summary's symbol as a hidden child, for `can_continue`.
static void query_analysis_find_parents(QueryAnalyzer *self) {
  // INVARIANT: A match can start inside a repetition that it cannot fit inside
  // only for a pattern that `needs_parents` found, so the scan has found the
  // reductions of the visible symbols.
  ts_assert(self->needs_parents);
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    if (!ts_language_symbol_metadata(self->language, summary->symbol).visible) continue;
    if (query_analysis_raise_need(summary, QueryAnalysisNeedChildren)) array_push(&self->pending, s);
  }
  query_analysis_find_pending_children(self);
  query_analysis_inherit_children(self);

  // Each summary's count of parents, then where its parents end, and, once
  // they are filled in from the end, where they start.
  array_grow_by(&self->parent_offsets, self->summaries.size + 1);
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    const QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    for (uint32_t h = 0; h < summary->hidden.size; h++) {
      const uint32_t child = (uint32_t)*array_get(&self->summary_index, array_get(&summary->hidden, h)->symbol);
      (*array_get(&self->parent_offsets, child))++;
    }
  }
  for (uint32_t s = 1; s <= self->summaries.size; s++) {
    *array_get(&self->parent_offsets, s) += *array_get(&self->parent_offsets, s - 1);
  }
  array_grow_by(&self->parents, *array_get(&self->parent_offsets, self->summaries.size));
  for (uint32_t s = 0; s < self->summaries.size; s++) {
    const QueryAnalysisSummary *summary = array_get(&self->summaries, s);
    for (uint32_t h = 0; h < summary->hidden.size; h++) {
      const uint32_t child = (uint32_t)*array_get(&self->summary_index, array_get(&summary->hidden, h)->symbol);
      uint32_t *offset = array_get(&self->parent_offsets, child);
      (*offset)--;
      *array_get(&self->parents, *offset) = s;
    }
  }
  array_grow_by(&self->visited, self->summaries.size);
}

// Marks each pattern with several top-level nodes whose match can start inside
// a repetition, and records the repetition in the program: with `around`
// unset, if the whole match can fit inside it, and with `around` set, if not,
// but the match can go on among the children of the visible node around it.
// Returns whether a match can start inside a repetition that it cannot fit
// inside.
static bool query_analysis_find_repetitions(QueryAnalyzer *self, bool around) {
  bool has_rest = false;
  uint32_t pattern_index = 0;
  for (uint32_t pattern = 0; pattern < query_ast_size(self->ast); pattern = query_ast_end(self->ast, pattern)) {
    QueryProgramPattern *record = array_get(&self->program->patterns, pattern_index);
    pattern_index++;
    if (record->is_rooted) continue;
    for (uint32_t s = 0; s < self->summaries.size; s++) {
      const QueryAnalysisSummary *summary = array_get(&self->summaries, s);
      if (!query_analysis_is_repetition(self->language, summary->symbol)) continue;
      if (!query_analysis_can_start(self, summary, pattern, 0)) continue;
      const bool fits = query_analysis_is_possible(self, summary, pattern, 0);
      if (around) {
        if (fits || !query_analysis_can_continue(self, s, pattern)) continue;
      } else if (!fits) {
        has_rest = true;
        continue;
      }
      record->is_non_local = true;
      array_push(&self->program->repetition_symbols, summary->symbol);
    }
  }
  return has_rest;
}

// Marks each pattern with several top-level nodes whose match can start inside
// a repetition and go on among the children of the visible node around it, and
// records those repetitions in the program. Only a match that cannot fit
// inside the repetition needs the visible symbols around it.
static void query_analysis_find_non_local(QueryAnalyzer *self) {
  if (query_analysis_find_repetitions(self, false)) {
    query_analysis_find_parents(self);
    query_analysis_find_repetitions(self, true);
  }
  query_analysis_sort_unique(&self->program->repetition_symbols, query_analysis_compare_u16);
}

// Returns whether a match of `pattern`, a pattern with several top-level
// nodes, can start inside a repetition without fitting inside it: whether it
// has a group of several nodes, as itself, its only node, or a branch.
static bool query_analysis_needs_parents(QueryAnalyzer *self, uint32_t pattern) {
  uint32_t atom = pattern + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) atom++;
  const QueryAstKind kind = query_ast_kind(self->ast, atom);
  if (kind != QueryAstKindAlternation && kind != QueryAstKindGroup) return false;
  uint32_t count = 0;
  for (uint32_t node = atom + 1; node < query_ast_end(self->ast, atom); node = query_ast_end(self->ast, node)) {
    if (query_ast_kind(self->ast, node) != QueryAstKindPattern) continue;
    if (query_analysis_needs_parents(self, node)) return true;
    count++;
  }
  return kind == QueryAstKindGroup && count > 1;
}

// Adds a summary for each node type that a non-terminal's nodes can have, with
// its need raised to `need`: the non-terminal's public symbol if it is visible,
// which can be a token's of the same name, and each of its aliases. The scan
// records the non-terminal's reductions under those symbols.
static void query_analysis_add_node_types(QueryAnalyzer *self, QueryAnalysisNeed need) {
  const TSLanguage *language = self->language;
  for (TSSymbol symbol = (TSSymbol)language->token_count; symbol < language->symbol_count; symbol++) {
    const TSSymbol *alias, *alias_end;
    ts_language_aliases_for_symbol(language, symbol, &alias, &alias_end);
    for (; alias < alias_end; alias++) {
      if (!ts_language_symbol_metadata(language, *alias).visible) continue;
      query_analysis_raise_need(query_analysis_add_summary(self, *alias), need);
    }
  }
}

// Adds a summary for each hidden non-terminal, whose children become its
// parent's, for each parent in the query, and for each node type of a
// non-terminal if a child's parent can be any visible node, and records what
// the analysis needs to know about them and the repetitions.
static void query_analysis_add_summaries(QueryAnalyzer *self) {
  const TSLanguage *language = self->language;
  const QueryAst *ast = self->ast;
  for (TSSymbol symbol = (TSSymbol)language->token_count; symbol < language->symbol_count; symbol++) {
    if (!ts_language_symbol_metadata(language, symbol).visible) query_analysis_add_summary(self, symbol);
  }

  // The end of the outermost named node so far. A field after it is on a part
  // of a pattern's top level, which can be a child of any visible node.
  uint32_t named_end = 0;
  for (uint32_t node = 0; node < query_ast_size(ast); node++) {
    const QueryAstKind node_kind = query_ast_kind(ast, node);
    if (node_kind == QueryAstKindField && node >= named_end) {
      query_analysis_raise_need(&self->any_parent, QueryAnalysisNeedChildren);
    }
    if (node_kind != QueryAstKindNamedNode) continue;
    if (node >= named_end) named_end = query_ast_end(ast, node);
    const TSSymbol symbol = query_analysis_node_type(self, node);
    if (symbol == ts_builtin_sym_error) continue;
    bool has_children = false;
    bool has_end_anchor = false;
    bool anchor_since_last = false;
    for (uint32_t child = node + 1; child < query_ast_end(ast, node); child = query_ast_end(ast, child)) {
      const QueryAstKind kind = query_ast_kind(ast, child);
      if (kind == QueryAstKindAnchor) {
        if (!has_children) has_end_anchor = true;
        anchor_since_last = true;
      } else if (kind == QueryAstKindPattern) {
        has_children = true;
        anchor_since_last = false;
      }
    }
    if (!has_children) continue;
    if (anchor_since_last) has_end_anchor = true;
    const QueryAnalysisNeed need = has_end_anchor ? QueryAnalysisNeedEnds : QueryAnalysisNeedChildren;
    query_analysis_raise_need(symbol ? query_analysis_add_summary(self, symbol) : &self->any_parent, need);
  }
  if (self->any_parent.need != QueryAnalysisNeedNothing) query_analysis_add_node_types(self, self->any_parent.need);

  // A pattern with several top-level nodes can start inside any repetition,
  // and go on among the children of the visible node around it. The node types
  // get summaries, so that the scan finds their reductions, but their children
  // are found only if the analysis needs them.
  uint32_t pattern_index = 0;
  for (uint32_t pattern = 0; pattern < query_ast_size(ast); pattern = query_ast_end(ast, pattern)) {
    if (!array_get(&self->program->patterns, pattern_index)->is_rooted) {
      self->has_non_rooted_pattern = true;
      if (query_analysis_needs_parents(self, pattern)) self->needs_parents = true;
    }
    pattern_index++;
  }
  for (TSSymbol symbol = (TSSymbol)language->token_count; symbol < language->symbol_count; symbol++) {
    if (self->has_non_rooted_pattern && query_analysis_is_repetition(language, symbol)) {
      query_analysis_raise_need(query_analysis_summary(self, symbol), QueryAnalysisNeedChildren);
    }
  }
  if (self->needs_parents) query_analysis_add_node_types(self, QueryAnalysisNeedNothing);

  for (TSSymbol symbol = 0; symbol < language->symbol_count; symbol++) {
    const TSSymbol *alias, *alias_end;
    ts_language_aliases_for_symbol(language, symbol, &alias, &alias_end);
    for (; alias < alias_end; alias++) {
      if (query_analysis_summary(self, *alias)) *array_get(&self->has_summary, symbol) = true;
    }
  }
}

static void query_analysis_summary_delete(QueryAnalysisSummary *summary) {
  array_delete(&summary->reductions);
  array_delete(&summary->direct);
  array_delete(&summary->children);
  array_delete(&summary->hidden);
  array_delete(&summary->repeatable_fields);
  array_delete(&summary->first_named);
  array_delete(&summary->last_named);
}

bool ts_query_analyze(
  const TSLanguage *language,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  const QueryAstIds *ids,
  QueryProgram *program,
  QueryDiagnostic *error
) {
  QueryAnalyzer self = {
    .language = language,
    .tokens = tokens,
    .ast = ast,
    .ids = ids,
    .program = program,
    .error = error,
    .summaries = array_new(),
    .summary_index = array_new(),
    .has_summary = array_new(),
    .is_extra = array_new(),
    .extras = array_new(),
    .predecessor_offsets = array_new(),
    .predecessors = array_new(),
    .pending = array_new(),
    .parent_offsets = array_new(),
    .parents = array_new(),
    .visited = array_new(),
  };
  const uint32_t symbol_count = ts_language_symbol_count(language);
  array_grow_by(&self.summary_index, symbol_count);
  for (uint32_t i = 0; i < symbol_count; i++) *array_get(&self.summary_index, i) = -1;
  array_grow_by(&self.has_summary, symbol_count);
  array_grow_by(&self.is_extra, symbol_count);

  query_analysis_add_summaries(&self);
  query_analysis_scan(&self);
  query_analysis_summarize(&self);
  if (self.any_parent.need != QueryAnalysisNeedNothing) query_analysis_build_any_parent(&self);

  bool possible = true;
  // The end of the outermost named node so far, as in `add_summaries`.
  uint32_t named_end = 0;
  for (uint32_t node = 0; node < query_ast_size(ast) && possible; node++) {
    const QueryAstKind kind = query_ast_kind(ast, node);
    if (kind == QueryAstKindField && node >= named_end) {
      // The pattern before the field is a part of a pattern's top level.
      possible = query_analysis_check_child(&self, &self.any_parent, node - 1, 0);
    } else if (kind == QueryAstKindNamedNode) {
      if (node >= named_end) named_end = query_ast_end(ast, node);
      // A named node is the atom of the pattern before it, or before its field.
      uint32_t pattern = node - 1;
      if (query_ast_kind(ast, pattern) == QueryAstKindField) pattern--;
      possible = query_analysis_check_parent(&self, pattern, node);
    }
  }
  if (possible && self.has_non_rooted_pattern) query_analysis_find_non_local(&self);

  for (uint32_t s = 0; s < self.summaries.size; s++) {
    query_analysis_summary_delete(array_get(&self.summaries, s));
  }
  query_analysis_summary_delete(&self.any_parent);
  array_delete(&self.summaries);
  array_delete(&self.summary_index);
  array_delete(&self.has_summary);
  array_delete(&self.is_extra);
  array_delete(&self.extras);
  array_delete(&self.predecessor_offsets);
  array_delete(&self.predecessors);
  array_delete(&self.pending);
  array_delete(&self.parent_offsets);
  array_delete(&self.parents);
  array_delete(&self.visited);
  ts_free(self.positions.slots);
  ts_free(self.children.slots);
  return possible;
}
