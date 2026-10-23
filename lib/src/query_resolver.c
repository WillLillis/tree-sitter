#include "./query_resolver.h"
#include "./language.h"
#include "./ts_assert.h"

#include <string.h>

// The fewest and the most times that something occurs. `min` is 0 or 1, and
// `max` is 0, 1, or 2, which stands for any number more than one.
typedef struct {
  uint8_t min;
  uint8_t max;
} QueryQuantifierBounds;

static QueryQuantifierBounds query_quantifier_bounds(TSQuantifier quantifier) {
  switch (quantifier) {
    case TSQuantifierZero: return (QueryQuantifierBounds) {.min = 0, .max = 0};
    case TSQuantifierZeroOrOne: return (QueryQuantifierBounds) {.min = 0, .max = 1};
    case TSQuantifierZeroOrMore: return (QueryQuantifierBounds) {.min = 0, .max = 2};
    case TSQuantifierOne: return (QueryQuantifierBounds) {.min = 1, .max = 1};
    case TSQuantifierOneOrMore: return (QueryQuantifierBounds) {.min = 1, .max = 2};
  }
  ts_unreachable();
}

static TSQuantifier query_quantifier_from_bounds(QueryQuantifierBounds bounds) {
  if (bounds.max == 0) return TSQuantifierZero;
  if (bounds.min == 0) return bounds.max == 1 ? TSQuantifierZeroOrOne : TSQuantifierZeroOrMore;
  return bounds.max == 1 ? TSQuantifierOne : TSQuantifierOneOrMore;
}

// Returns how many times a capture occurs if it occurs `left` times and then
// `right` more times.
static TSQuantifier query_quantifier_add(TSQuantifier left, TSQuantifier right) {
  const QueryQuantifierBounds a = query_quantifier_bounds(left);
  const QueryQuantifierBounds b = query_quantifier_bounds(right);
  const uint8_t max = a.max + b.max;
  return query_quantifier_from_bounds((QueryQuantifierBounds) {
    .min = a.min || b.min,
    .max = max < 2 ? max : 2,
  });
}

// Returns how many times a capture occurs if it occurs either `left` times or
// `right` times.
static TSQuantifier query_quantifier_join(TSQuantifier left, TSQuantifier right) {
  const QueryQuantifierBounds a = query_quantifier_bounds(left);
  const QueryQuantifierBounds b = query_quantifier_bounds(right);
  return query_quantifier_from_bounds((QueryQuantifierBounds) {
    .min = a.min && b.min,
    .max = a.max > b.max ? a.max : b.max,
  });
}

// Returns how many times a capture occurs if it occurs `left` times in something
// that occurs `right` times.
static TSQuantifier query_quantifier_multiply(TSQuantifier left, TSQuantifier right) {
  const QueryQuantifierBounds a = query_quantifier_bounds(left);
  const QueryQuantifierBounds b = query_quantifier_bounds(right);
  const uint8_t max = a.max * b.max;
  return query_quantifier_from_bounds((QueryQuantifierBounds) {
    .min = a.min && b.min,
    .max = max < 2 ? max : 2,
  });
}

// A hash set of the strings in `table`, with linear probing. Each slot is zero,
// or one more than the id of a string. There are zero slots, or a power of two
// more than twice as many as strings.
typedef struct {
  QueryStringTable *table;
  Array(uint32_t) slots;
} QueryStringSet;

// Returns the FNV-1a hash of the given bytes.
static uint32_t query_string_hash(const char *bytes, uint32_t length) {
  uint32_t hash = 2166136261u;
  for (uint32_t i = 0; i < length; i++) hash = (hash ^ (uint8_t)bytes[i]) * 16777619u;
  return hash;
}

// Returns the slot that holds the string with the given bytes, or else the
// empty slot where it belongs.
static uint32_t query_string_set_probe(
  const QueryStringSet *self,
  const char *bytes,
  uint32_t length
) {
  const uint32_t mask = self->slots.size - 1;
  for (uint32_t slot = query_string_hash(bytes, length) & mask;; slot = (slot + 1) & mask) {
    const uint32_t entry = *array_get(&self->slots, slot);
    if (entry == 0) return slot;
    uint32_t candidate_length;
    const char *candidate = query_string_table_get(self->table, entry - 1, &candidate_length);
    if (candidate_length == length && memcmp(candidate, bytes, length) == 0) return slot;
  }
}

// Sets `id` to the id of the string with the given bytes, and returns whether
// the table has it.
static bool query_string_set_find(
  const QueryStringSet *self,
  const char *bytes,
  uint32_t length,
  uint32_t *id
) {
  if (self->slots.size == 0) return false;
  const uint32_t entry = *array_get(&self->slots, query_string_set_probe(self, bytes, length));
  if (entry == 0) return false;
  *id = entry - 1;
  return true;
}

// Doubles the slots, and puts every string back in.
static void query_string_set_grow(QueryStringSet *self) {
  const uint32_t capacity = self->slots.size > 0 ? 2 * self->slots.size : 16;
  array_clear(&self->slots);
  array_grow_by(&self->slots, capacity);
  for (uint32_t id = 0; id < query_string_table_size(self->table); id++) {
    uint32_t length;
    const char *bytes = query_string_table_get(self->table, id, &length);
    *array_get(&self->slots, query_string_set_probe(self, bytes, length)) = id + 1;
  }
}

// Returns the id of the string with the given bytes, adding it to the table if
// it is new.
static uint32_t query_string_set_intern(QueryStringSet *self, const char *bytes, uint32_t length) {
  if (2 * (query_string_table_size(self->table) + 1) >= self->slots.size) query_string_set_grow(self);
  uint32_t *slot = array_get(&self->slots, query_string_set_probe(self, bytes, length));
  if (*slot == 0) {
    QueryStringTable *table = self->table;
    *slot = query_string_table_size(table) + 1;
    array_push(&table->offsets, table->bytes.size);
    array_extend(&table->bytes, length, bytes);
    array_push(&table->bytes, '\0');
  }
  return *slot - 1;
}

// An id in the query-wide capture-name table.
typedef struct {
  uint32_t index;
} QueryCaptureId;

// A capture's position in the current pattern's captures.
typedef struct {
  uint32_t index;
} QueryPatternCaptureIndex;

// The state of one call to `ts_query_resolve`.
typedef struct {
  const TSLanguage *language;
  const char *source;
  const QueryTokenList *tokens;
  const QueryAst *ast;
  QueryAstIds *ids;
  QueryTables *tables;
  // Receives a description of the error that stops resolution.
  QueryDiagnostic *error;
  // Hash sets of `tables->captures` and `tables->strings`.
  QueryStringSet captures;
  QueryStringSet strings;
  // The value of the last string with escape sequences that was looked up.
  Array(char) decoded;
  // The captures that the current pattern defines, each once, in order of
  // first appearance.
  Array(QueryCaptureId) pattern_captures;
  // For each subtree that the current pattern's capture quantifiers are being
  // computed for, and that is not yet part of its parent's, the quantifiers of
  // the pattern's captures within it, innermost last.
  Array(uint8_t) quantifiers;
} QueryResolver;

// Describes the error that stops resolution as being at the token of `node`, and
// returns false.
static bool query_resolver_fail_at(
  const QueryResolver *self,
  QueryDiagnosticKind kind,
  QueryAstIndex node
) {
  *self->error = (QueryDiagnostic) {
    .span = query_token_list_span(self->tokens, query_ast_token(self->ast, node)),
    .kind = kind,
  };
  return false;
}

// Returns whether a token can be used as a word.
static bool query_resolver_is_word(QueryTokenKind kind) {
  return
    kind == QueryTokenKindIdentifier ||
    kind == QueryTokenKindUnderscore ||
    kind == QueryTokenKindMissing;
}

static void query_resolver_set_id(QueryResolver *self, QueryAstIndex node, uint32_t id) {
  *array_get(&self->ids->ids, node.index) = id;
}

// Returns the value of the string token `token`: the bytes between its quotes,
// with each escape sequence replaced by the byte that it stands for. Sets
// `length` to how many bytes there are. The value is in the query source if
// the string has no escape sequences, and otherwise in `self.decoded`, until
// the next call replaces it.
static const char *query_resolver_string_value(
  QueryResolver *self,
  QueryTokenIndex token,
  uint32_t *length
) {
  const QuerySpan span = query_token_list_span(self->tokens, token);
  const char *contents = &self->source[span.start + 1];
  const char *end = &self->source[span.end - 1];
  const char *backslash = memchr(contents, '\\', (size_t)(end - contents));
  if (!backslash) {
    *length = (uint32_t)(end - contents);
    return contents;
  }

  array_clear(&self->decoded);
  const char *run = contents;
  while (backslash) {
    array_extend(&self->decoded, (uint32_t)(backslash - run), run);
    char value;
    const bool is_valid = query_token_unescape((uint8_t)backslash[1], &value);
    // INVARIANT: The tokenizer only produces strings in which every backslash
    // begins a valid escape sequence.
    ts_assert(is_valid);
    array_push(&self->decoded, value);
    run = backslash + 2;
    backslash = memchr(run, '\\', (size_t)(end - run));
  }
  array_extend(&self->decoded, (uint32_t)(end - run), run);
  *length = self->decoded.size;
  return self->decoded.contents;
}

// Resolves `node` to the node type that its token names: a named node type for
// a name, and an anonymous one for a string.
static bool query_resolver_resolve_node_type(QueryResolver *self, QueryAstIndex node) {
  const QueryTokenIndex token = query_ast_token(self->ast, node);
  const QueryTokenKind kind = query_token_list_kind(self->tokens, token);
  ts_assert(kind == QueryTokenKindString || query_resolver_is_word(kind));
  TSSymbol symbol;
  if (kind == QueryTokenKindString) {
    uint32_t length;
    const char *value = query_resolver_string_value(self, token, &length);
    symbol = ts_language_symbol_for_name(self->language, value, length, false);
  } else {
    const QuerySpan span = query_token_list_span(self->tokens, token);
    symbol = ts_language_symbol_for_name(
      self->language,
      &self->source[span.start],
      span.end - span.start,
      true
    );
  }
  if (!symbol) return query_resolver_fail_at(self, QueryDiagnosticKindUnknownNodeType, node);
  query_resolver_set_id(self, node, symbol);
  return true;
}

// Returns whether `supertype` wraps nodes of type `type`, either as one of its
// subtypes or through a supertype among them.
static bool query_resolver_supertype_wraps(const TSLanguage *language, TSSymbol supertype, TSSymbol type) {
  uint32_t subtype_count;
  const TSSymbol *subtypes = ts_language_subtypes(language, supertype, &subtype_count);
  for (uint32_t i = 0; i < subtype_count; i++) {
    if (subtypes[i] == type) return true;
    if (
      ts_language_symbol_metadata(language, subtypes[i]).supertype &&
      query_resolver_supertype_wraps(language, subtypes[i], type)
    ) return true;
  }
  return false;
}

// Resolves `node`, the subtype in `(supertype/subtype ...)`, after checking that
// its named node is a supertype, and that the subtype is a node type that the
// supertype wraps.
static bool query_resolver_resolve_subtype(QueryResolver *self, QueryAstIndex node) {
  ts_assert(node.index > 0);
  ts_assert(query_ast_kind(self->ast, node) == QueryAstKindSubtype);
  // A subtype is the first child of its named node.
  const QueryAstIndex named_node = {.index = node.index - 1};
  ts_assert(query_ast_kind(self->ast, named_node) == QueryAstKindNamedNode);
  const TSSymbol supertype = (TSSymbol)query_ast_ids_get(self->ids, named_node);
  if (!supertype || !ts_language_symbol_metadata(self->language, supertype).supertype) {
    return query_resolver_fail_at(self, QueryDiagnosticKindNotASupertype, named_node);
  }
  if (!query_resolver_resolve_node_type(self, node)) return false;
  const TSSymbol subtype = (TSSymbol)query_ast_ids_get(self->ids, node);
  // No node has a supertype as its type.
  if (ts_language_symbol_metadata(self->language, subtype).supertype) {
    return query_resolver_fail_at(self, QueryDiagnosticKindImpossiblePattern, node);
  }
  if (
    self->language->abi_version >= LANGUAGE_VERSION_WITH_RESERVED_WORDS &&
    !query_resolver_supertype_wraps(self->language, supertype, subtype)
  ) return query_resolver_fail_at(self, QueryDiagnosticKindNotASubtype, node);
  return true;
}

// Returns the id of the capture that `node` defines, adding the capture to the
// query's captures, and to the current pattern's, if it is new to them.
static QueryCaptureId query_resolver_define_capture(QueryResolver *self, QueryAstIndex node) {
  const QueryTokenIndex token = query_ast_token(self->ast, node);
  ts_assert(query_ast_kind(self->ast, node) == QueryAstKindCapture);
  ts_assert(query_token_list_kind(self->tokens, token) == QueryTokenKindCapture);
  const QuerySpan span = query_token_list_span(self->tokens, token);
  // The name of a capture leaves out its `@`.
  const QueryCaptureId id = {
    .index = query_string_set_intern(
      &self->captures,
      &self->source[span.start + 1],
      span.end - span.start - 1
    ),
  };
  for (uint32_t i = 0; i < self->pattern_captures.size; i++) {
    if (array_get(&self->pattern_captures, i)->index == id.index) return id;
  }
  array_push(&self->pattern_captures, id);
  return id;
}

// Resolves the name of `node`, if it has one, other than those of predicates
// and their arguments.
static bool query_resolver_resolve_name(QueryResolver *self, QueryAstIndex node) {
  const QueryTokenIndex token = query_ast_token(self->ast, node);
  const QueryTokenKind token_kind = query_token_list_kind(self->tokens, token);
  switch (query_ast_kind(self->ast, node)) {
    case QueryAstKindNamedNode: {
      // `(_)` names no node type, so its id stays zero.
      if (token_kind == QueryTokenKindUnderscore) return true;
      ts_assert(token_kind == QueryTokenKindIdentifier);
      if (!query_resolver_resolve_node_type(self, node)) return false;
      const TSSymbol symbol = (TSSymbol)query_ast_ids_get(self->ids, node);
      // A supertype needs an explicit subtype to have child patterns.
      if (ts_language_symbol_metadata(self->language, symbol).supertype) {
        const QueryAstIndex end = query_ast_end(self->ast, node);
        for (
          QueryAstIndex child = query_ast_first_child(node);
          child.index < end.index;
          child = query_ast_next_sibling(self->ast, child)
        ) {
          const QueryAstKind kind = query_ast_kind(self->ast, child);
          if (kind == QueryAstKindSubtype) break;
          if (kind == QueryAstKindPattern) return query_resolver_fail_at(self, QueryDiagnosticKindImpossiblePattern, child);
        }
      }
      return true;
    }
    case QueryAstKindAnonymousNode:
      ts_assert(token_kind == QueryTokenKindString);
      return query_resolver_resolve_node_type(self, node);
    case QueryAstKindMissingType:
      ts_assert(token_kind == QueryTokenKindString || query_resolver_is_word(token_kind));
      return query_resolver_resolve_node_type(self, node);
    case QueryAstKindSubtype:
      ts_assert(token_kind == QueryTokenKindString || query_resolver_is_word(token_kind));
      return query_resolver_resolve_subtype(self, node);
    case QueryAstKindField:
    case QueryAstKindNegatedField: {
      ts_assert(query_resolver_is_word(token_kind));
      const QuerySpan span = query_token_list_span(self->tokens, token);
      const TSFieldId field = ts_language_field_id_for_name(
        self->language,
        &self->source[span.start],
        span.end - span.start
      );
      if (!field) return query_resolver_fail_at(self, QueryDiagnosticKindUnknownField, node);
      query_resolver_set_id(self, node, field);
      return true;
    }
    case QueryAstKindCapture: {
      ts_assert(token_kind == QueryTokenKindCapture);
      const QueryCaptureId capture = query_resolver_define_capture(self, node);
      if (capture.index > UINT16_MAX) return query_resolver_fail_at(self, QueryDiagnosticKindTooManyCaptures, node);
      // `QueryAstIds` stores ids for all AST kinds in on type-erased array.
      query_resolver_set_id(self, node, capture.index);
      return true;
    }
    // Predicates and their arguments are resolved after the rest of the pattern.
    case QueryAstKindPredicate:
      ts_assert(token_kind == QueryTokenKindPredicateName || token_kind == QueryTokenKindDot);
      return true;
    case QueryAstKindPredicateArgument:
      ts_assert(
        token_kind == QueryTokenKindCapture ||
        token_kind == QueryTokenKindString ||
        query_resolver_is_word(token_kind)
      );
      return true;
    case QueryAstKindPattern:
    case QueryAstKindMissingNode:
    case QueryAstKindWildcard:
    case QueryAstKindAlternation:
    case QueryAstKindGroup:
    case QueryAstKindAnchor:
    case QueryAstKindQuantifier:
      return true;
  }
  ts_unreachable();
}

// Sets `index` to the position of `capture` in the current pattern's captures,
// or to their count if the pattern does not define it, which is where it would
// go. Returns whether the pattern defines it.
static bool query_resolver_pattern_capture_index(
  const QueryResolver *self,
  QueryCaptureId capture,
  QueryPatternCaptureIndex *index
) {
  for (uint32_t i = 0; i < self->pattern_captures.size; i++) {
    if (array_get(&self->pattern_captures, i)->index == capture.index) {
      *index = (QueryPatternCaptureIndex) {.index = i};
      return true;
    }
  }
  *index = (QueryPatternCaptureIndex) {.index = self->pattern_captures.size};
  return false;
}

// Resolves `node`, an argument of a predicate, and appends its step.
static bool query_resolver_resolve_argument(QueryResolver *self, QueryAstIndex node) {
  ts_assert(query_ast_kind(self->ast, node) == QueryAstKindPredicateArgument);
  const QueryTokenIndex token = query_ast_token(self->ast, node);
  const QuerySpan span = query_token_list_span(self->tokens, token);
  const QueryTokenKind kind = query_token_list_kind(self->tokens, token);
  TSQueryPredicateStep step;
  switch (kind) {
    case QueryTokenKindCapture: {
      QueryCaptureId capture;
      QueryPatternCaptureIndex index;
      // The name of a capture leaves out its `@`.
      if (
        !query_string_set_find(
          &self->captures,
          &self->source[span.start + 1],
          span.end - span.start - 1,
          &capture.index
        ) ||
        !query_resolver_pattern_capture_index(self, capture, &index)
      ) {
        return query_resolver_fail_at(self, QueryDiagnosticKindUndefinedCapture, node);
      }
      step = (TSQueryPredicateStep) {
        .type = TSQueryPredicateStepTypeCapture,
        .value_id = capture.index,
      };
      break;
    }
    case QueryTokenKindString: {
      uint32_t length;
      const char *value = query_resolver_string_value(self, token, &length);
      step = (TSQueryPredicateStep) {
        .type = TSQueryPredicateStepTypeString,
        .value_id = query_string_set_intern(&self->strings, value, length),
      };
      break;
    }
    case QueryTokenKindIdentifier:
    case QueryTokenKindUnderscore:
    case QueryTokenKindMissing:
      step = (TSQueryPredicateStep) {
        .type = TSQueryPredicateStepTypeString,
        .value_id = query_string_set_intern(
          &self->strings,
          &self->source[span.start],
          span.end - span.start
        ),
      };
      break;
    case QueryTokenKindLeftParen:
    case QueryTokenKindRightParen:
    case QueryTokenKindLeftBracket:
    case QueryTokenKindRightBracket:
    case QueryTokenKindDot:
    case QueryTokenKindBang:
    case QueryTokenKindColon:
    case QueryTokenKindSlash:
    case QueryTokenKindQuestion:
    case QueryTokenKindStar:
    case QueryTokenKindPlus:
    case QueryTokenKindPredicateName:
    case QueryTokenKindEnd:
      ts_unreachable();
  }
  query_resolver_set_id(self, node, step.value_id);
  array_push(&self->tables->predicate_steps, step);
  return true;
}

// Resolves the predicate `node` and its arguments, and appends their steps.
static bool query_resolver_resolve_predicate(QueryResolver *self, QueryAstIndex node) {
  ts_assert(query_ast_kind(self->ast, node) == QueryAstKindPredicate);
  const QueryTokenIndex token = query_ast_token(self->ast, node);
  const QueryTokenKind kind = query_token_list_kind(self->tokens, token);
  ts_assert(kind == QueryTokenKindPredicateName || kind == QueryTokenKindDot);
  const QuerySpan span = query_token_list_span(self->tokens, token);
  // The name of a predicate leaves out its `#`, or the `.` of the legacy form,
  // whose `?` or `!` is two tokens after the `.`.
  uint32_t name_end = span.end;
  if (kind == QueryTokenKindDot) {
    const QueryTokenIndex word = query_token_next(token);
    const QueryTokenIndex suffix = query_token_next(word);
    name_end = query_token_list_span(self->tokens, suffix).end;
  }
  const uint32_t name = query_string_set_intern(
    &self->strings,
    &self->source[span.start + 1],
    name_end - span.start - 1
  );
  query_resolver_set_id(self, node, name);
  array_push(&self->tables->predicate_steps, ((TSQueryPredicateStep) {
    .type = TSQueryPredicateStepTypeString,
    .value_id = name,
  }));

  // The arguments of a predicate are its children, and they are leaves.
  const QueryAstIndex end = query_ast_end(self->ast, node);
  for (
    QueryAstIndex argument = query_ast_first_child(node);
    argument.index < end.index;
    argument = query_ast_next_sibling(self->ast, argument)
  ) {
    if (!query_resolver_resolve_argument(self, argument)) return false;
  }
  array_push(&self->tables->predicate_steps, ((TSQueryPredicateStep) {
    .type = TSQueryPredicateStepTypeDone,
    .value_id = 0,
  }));
  return true;
}

// Combines the quantifiers of the patterns that are the children of `node`,
// which are the last on the stack, into one entry for `node`.
static void query_resolver_combine_quantifiers(
  QueryResolver *self,
  QueryAstIndex node,
  TSQuantifier (*combine)(TSQuantifier, TSQuantifier)
) {
  const uint32_t capture_count = self->pattern_captures.size;
  uint32_t pattern_count = 0;
  const QueryAstIndex end = query_ast_end(self->ast, node);
  for (
    QueryAstIndex child = query_ast_first_child(node);
    child.index < end.index;
    child = query_ast_next_sibling(self->ast, child)
  ) {
    if (query_ast_kind(self->ast, child) == QueryAstKindPattern) pattern_count++;
  }
  if (pattern_count == 0) {
    array_grow_by(&self->quantifiers, capture_count);
    return;
  }

  uint8_t *combined = array_get(&self->quantifiers, self->quantifiers.size - pattern_count * capture_count);
  for (uint32_t i = 1; i < pattern_count; i++) {
    const uint8_t *other = &combined[i * capture_count];
    for (uint32_t j = 0; j < capture_count; j++) {
      combined[j] = (uint8_t)combine((TSQuantifier)combined[j], (TSQuantifier)other[j]);
    }
  }
  self->quantifiers.size -= (pattern_count - 1) * capture_count;
}

// Applies the captures and quantifiers of the pattern `node` to the quantifiers
// of its atom, the last entry on the stack, which becomes the pattern's.
static void query_resolver_finish_quantifiers(QueryResolver *self, QueryAstIndex node) {
  ts_assert(query_ast_kind(self->ast, node) == QueryAstKindPattern);
  const uint32_t capture_count = self->pattern_captures.size;
  uint8_t *quantifiers = array_get(&self->quantifiers, self->quantifiers.size - capture_count);
  const QueryAstIndex pattern_end = query_ast_end(self->ast, node);
  QueryAstIndex atom = query_ast_first_child(node);
  ts_assert(atom.index < pattern_end.index);
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) {
    // INVARIANT: A pattern has at most one field, before its atom.
    atom.index++;
    ts_assert(atom.index < pattern_end.index);
  }

  // A pattern's quantifiers and captures are leaves after its atom.
  TSQuantifier pattern_quantifier = TSQuantifierOne;
  for (
    QueryAstIndex suffix = query_ast_end(self->ast, atom);
    suffix.index < pattern_end.index;
    suffix.index++
  ) {
    if (query_ast_kind(self->ast, suffix) == QueryAstKindCapture) {
      const QueryCaptureId capture = {.index = query_ast_ids_get(self->ids, suffix)};
      QueryPatternCaptureIndex index;
      const bool is_defined = query_resolver_pattern_capture_index(
        self,
        capture,
        &index
      );
      // INVARIANT: Resolving the pattern's names defined each of its captures.
      ts_assert(is_defined);
      quantifiers[index.index] = (uint8_t)query_quantifier_add(
          (TSQuantifier)quantifiers[index.index],
          TSQuantifierOne
      );
      continue;
    }
    TSQuantifier quantifier;
    switch (query_token_list_kind(self->tokens, query_ast_token(self->ast, suffix))) {
      case QueryTokenKindQuestion: quantifier = TSQuantifierZeroOrOne; break;
      case QueryTokenKindStar: quantifier = TSQuantifierZeroOrMore; break;
      case QueryTokenKindPlus: quantifier = TSQuantifierOneOrMore; break;
      // INVARIANT: The parser only makes quantifiers of the above three tokens.
      default: ts_unreachable();
    }
    pattern_quantifier = query_quantifier_join(quantifier, pattern_quantifier);
  }
  for (uint32_t i = 0; i < capture_count; i++) {
    quantifiers[i] = (uint8_t)query_quantifier_multiply((TSQuantifier)quantifiers[i], pattern_quantifier);
  }
}

// Appends how many times each capture in the top-level pattern `pattern` occurs
// in each of its matches.
static void query_resolver_quantify_captures(QueryResolver *self, QueryAstIndex pattern) {
  const uint32_t capture_count = self->pattern_captures.size;
  if (capture_count == 0) return;

  // Visit each node after all of its descendants. Each atom leaves one entry on
  // the stack, with a quantifier for every capture in the pattern.
  array_clear(&self->quantifiers);
  for (
    QueryAstIndex node = query_ast_end(self->ast, pattern);
    node.index > pattern.index;
  ) {
    node.index--;
    switch (query_ast_kind(self->ast, node)) {
      case QueryAstKindAnonymousNode:
      case QueryAstKindWildcard:
      case QueryAstKindMissingNode:
        array_grow_by(&self->quantifiers, capture_count);
        break;
      case QueryAstKindNamedNode:
      case QueryAstKindGroup:
        query_resolver_combine_quantifiers(self, node, query_quantifier_add);
        break;
      case QueryAstKindAlternation:
        query_resolver_combine_quantifiers(self, node, query_quantifier_join);
        break;
      case QueryAstKindPattern:
        query_resolver_finish_quantifiers(self, node);
        break;
      // Handled with their containing pattern
      case QueryAstKindField:
      case QueryAstKindCapture:
      case QueryAstKindQuantifier:
      // Can't contain captures
      case QueryAstKindSubtype:
      case QueryAstKindMissingType:
      case QueryAstKindAnchor:
      case QueryAstKindNegatedField:
      case QueryAstKindPredicate:
      case QueryAstKindPredicateArgument:
        break;
    }
  }

  // INVARIANT: The one entry left is the top-level pattern's.
  ts_assert(self->quantifiers.size == capture_count);
  for (uint32_t i = 0; i < capture_count; i++) {
    array_push(&self->tables->capture_quantifiers, ((QueryCaptureQuantifier) {
      .capture = array_get(&self->pattern_captures, i)->index,
      .quantifier = *array_get(&self->quantifiers, i),
    }));
  }
}

// Resolves the top-level pattern `pattern`, and appends what the query records
// about it.
static bool query_resolver_resolve_pattern(QueryResolver *self, QueryAstIndex pattern) {
  ts_assert(query_ast_kind(self->ast, pattern) == QueryAstKindPattern);
  const QueryAstIndex end = query_ast_end(self->ast, pattern);
  array_clear(&self->pattern_captures);
  for (
    QueryAstIndex node = pattern;
    node.index < end.index;
    node.index++
  ) {
    if (!query_resolver_resolve_name(self, node)) return false;
  }

  // Predicates are resolved after the rest of the pattern, so that they can use
  // its captures wherever they are defined.
  const uint32_t predicate_step_offset = self->tables->predicate_steps.size;
  for (
    QueryAstIndex node = pattern;
    node.index < end.index;
    node.index++
  ) {
    if (
      query_ast_kind(self->ast, node) == QueryAstKindPredicate &&
      !query_resolver_resolve_predicate(self, node)
    ) return false;
  }

  const uint32_t capture_quantifier_offset = self->tables->capture_quantifiers.size;
  query_resolver_quantify_captures(self, pattern);

  // A top-level pattern's tokens run up to the first token of the next one, or
  // to the `End` token.
  const QueryTokenIndex next_token = end.index < query_ast_size(self->ast)
    ? query_ast_token(self->ast, end)
    : (QueryTokenIndex) {
      .index = query_token_list_size(self->tokens) - 1,
    };
  array_push(&self->tables->patterns, ((QueryPatternInfo) {
    .span = {
      .start = query_token_list_span(self->tokens, query_ast_token(self->ast, pattern)).start,
      .end = query_token_list_span(self->tokens, query_token_previous(next_token)).end,
    },
    .predicate_step_offset = predicate_step_offset,
    .predicate_step_count = self->tables->predicate_steps.size - predicate_step_offset,
    .capture_quantifier_offset = capture_quantifier_offset,
    .capture_quantifier_count = self->tables->capture_quantifiers.size - capture_quantifier_offset,
  }));
  return true;
}

bool ts_query_resolve(
  const TSLanguage *language,
  const char *source,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  QueryAstIds *ids,
  QueryTables *tables,
  QueryDiagnostic *error
) {
  QueryResolver self = {
    .language = language,
    .source = source,
    .tokens = tokens,
    .ast = ast,
    .ids = ids,
    .tables = tables,
    .error = error,
    .captures = {.table = &tables->captures, .slots = array_new()},
    .strings = {.table = &tables->strings, .slots = array_new()},
    .decoded = array_new(),
    .pattern_captures = array_new(),
    .quantifiers = array_new(),
  };
  array_clear(&ids->ids);
  array_grow_by(&ids->ids, query_ast_size(ast));
  bool succeeded = true;
  for (
    QueryAstIndex pattern = {.index = 0};
    succeeded && pattern.index < query_ast_size(ast);
    pattern = query_ast_end(ast, pattern)
  ) {
    succeeded = query_resolver_resolve_pattern(&self, pattern);
  }
  array_delete(&self.captures.slots);
  array_delete(&self.strings.slots);
  array_delete(&self.decoded);
  array_delete(&self.pattern_captures);
  array_delete(&self.quantifiers);
  return succeeded;
}

#ifdef DEBUG_QUERY_RESOLUTION
static const char *query_quantifier_name(TSQuantifier quantifier) {
  switch (quantifier) {
    case TSQuantifierZero: return "Zero";
    case TSQuantifierZeroOrOne: return "ZeroOrOne";
    case TSQuantifierZeroOrMore: return "ZeroOrMore";
    case TSQuantifierOne: return "One";
    case TSQuantifierOneOrMore: return "OneOrMore";
  }
  ts_unreachable();
}

void ts_query_resolution_dump(
  const TSLanguage *language,
  const char *source,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  const QueryAstIds *ids,
  const QueryTables *tables
) {
  printf("=== RESOLUTION ===\n");
  for (
    QueryAstIndex node = {.index = 0};
    node.index < query_ast_size(ast);
    node.index++
  ) {
    const QueryTokenIndex token = query_ast_token(ast, node);
    const uint32_t id = query_ast_ids_get(ids, node);
    const QuerySpan span = query_token_list_span(tokens, token);
    const char *kind = NULL;
    const char *name = NULL;
    uint32_t name_length = 0;
    switch (query_ast_kind(ast, node)) {
      case QueryAstKindNamedNode:
      case QueryAstKindAnonymousNode:
      case QueryAstKindSubtype:
      case QueryAstKindMissingType:
        kind = "symbol";
        name = id ? ts_language_symbol_name(language, (TSSymbol)id) : "_";
        name_length = (uint32_t)strlen(name);
        break;
      case QueryAstKindField:
      case QueryAstKindNegatedField:
        kind = "field";
        name = ts_language_field_name_for_id(language, (TSFieldId)id);
        name_length = name ? (uint32_t)strlen(name) : 0;
        break;
      case QueryAstKindCapture:
        kind = "capture";
        name = query_string_table_get(&tables->captures, id, &name_length);
        break;
      case QueryAstKindPredicate:
        kind = "string";
        name = query_string_table_get(&tables->strings, id, &name_length);
        break;
      case QueryAstKindPredicateArgument:
        if (query_token_list_kind(tokens, token) == QueryTokenKindCapture) {
          kind = "capture";
          name = query_string_table_get(&tables->captures, id, &name_length);
        } else {
          kind = "string";
          name = query_string_table_get(&tables->strings, id, &name_length);
        }
        break;
      case QueryAstKindPattern:
      case QueryAstKindMissingNode:
      case QueryAstKindWildcard:
      case QueryAstKindAlternation:
      case QueryAstKindGroup:
      case QueryAstKindAnchor:
      case QueryAstKindQuantifier:
        break;
    }
    if (!kind) continue;
    printf(
      "node %3u: %.*s -> %s id=%u %.*s\n",
      node,
      (int)(span.end - span.start),
      &source[span.start],
      kind,
      id,
      (int)name_length,
      name ? name : ""
    );
  }

  for (uint32_t i = 0; i < tables->patterns.size; i++) {
    const QueryPatternInfo *pattern = array_get(&tables->patterns, i);
    printf("pattern %u: %u..%u\n", i, pattern->span.start, pattern->span.end);
    printf("  capture quantifiers:\n");
    if (pattern->capture_quantifier_count == 0) {
      printf("    (none)\n");
    }
    for (uint32_t j = 0; j < pattern->capture_quantifier_count; j++) {
      const QueryCaptureQuantifier *entry = array_get(
        &tables->capture_quantifiers,
        pattern->capture_quantifier_offset + j
      );
      uint32_t length;
      const char *capture = query_string_table_get(&tables->captures, entry->capture, &length);
      printf(
        "    @%.*s: %s\n",
        (int)length,
        capture,
        query_quantifier_name((TSQuantifier)entry->quantifier)
      );
    }
    printf("  predicate steps:\n");
    if (pattern->predicate_step_count == 0) {
      printf("    (none)\n");
    }
    bool at_predicate_start = true;
    for (uint32_t j = 0; j < pattern->predicate_step_count; j++) {
      const TSQueryPredicateStep *step = array_get(&tables->predicate_steps, pattern->predicate_step_offset + j);
      uint32_t length;
      const char *value;
      switch (step->type) {
        case TSQueryPredicateStepTypeDone:
          printf("  done (end predicate)\n");
          at_predicate_start = true;
          break;
        case TSQueryPredicateStepTypeCapture:
          value = query_string_table_get(&tables->captures, step->value_id, &length);
          printf("    capture @%.*s\n", (int)length, value);
          break;
        case TSQueryPredicateStepTypeString:
          value = query_string_table_get(&tables->strings, step->value_id, &length);
          if (at_predicate_start) {
            printf("    operator %.*s\n", (int)length, value);
            at_predicate_start = false;
          } else {
            printf("    string %.*s\n", (int)length, value);
          }
          break;
      }
    }
  }
}
#endif
