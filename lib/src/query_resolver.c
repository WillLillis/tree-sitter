#include "./query_resolver.h"
#include "./language.h"

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
  return (QueryQuantifierBounds) {.min = 0, .max = 0};
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

// Returns the id of the string with the given bytes, adding it to `self` if it
// is new.
static uint32_t query_string_table_intern(QueryStringTable *self, const char *bytes, uint32_t length) {
  uint32_t id;
  if (query_string_table_find(self, bytes, length, &id)) return id;
  const uint32_t start = self->bytes.size;
  array_extend(&self->bytes, length, bytes);
  array_push(&self->bytes, '\0');
  array_push(&self->offsets, start);
  return query_string_table_size(self) - 1;
}

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
  // The value of the last string with escape sequences that was looked up.
  Array(char) decoded;
  // The captures that the current pattern defines, each once, in order of
  // first appearance.
  Array(uint32_t) pattern_captures;
  // For each subtree that the current pattern's capture quantifiers are being
  // computed for, and that is not yet part of its parent's, the quantifiers of
  // the pattern's captures within it, innermost last.
  Array(uint8_t) quantifiers;
} QueryResolver;

// Describes the error that stops resolution as being at the token of `node`, and
// returns false.
static bool query_resolver_fail_at(const QueryResolver *self, QueryDiagnosticKind kind, uint32_t node) {
  *self->error = (QueryDiagnostic) {
    .span = query_token_list_span(self->tokens, query_ast_token(self->ast, node)),
    .kind = kind,
  };
  return false;
}

static void query_resolver_set_id(QueryResolver *self, uint32_t node, uint32_t id) {
  *array_get(&self->ids->ids, node) = id;
}

// Returns the value of the string token `token`: the bytes between its quotes,
// with each escape sequence replaced by the byte that it stands for. Sets
// `length` to how many bytes there are. The value is in the query source if
// the string has no escape sequences, and otherwise in `self.decoded`, until
// the next call replaces it.
static const char *query_resolver_string_value(QueryResolver *self, uint32_t token, uint32_t *length) {
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
static bool query_resolver_resolve_node_type(QueryResolver *self, uint32_t node) {
  const uint32_t token = query_ast_token(self->ast, node);
  TSSymbol symbol;
  if (query_token_list_kind(self->tokens, token) == QueryTokenKindString) {
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
static bool query_resolver_resolve_subtype(QueryResolver *self, uint32_t node) {
  // A subtype is the first child of its named node.
  const uint32_t named_node = node - 1;
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
static uint32_t query_resolver_define_capture(QueryResolver *self, uint32_t node) {
  const QuerySpan span = query_token_list_span(self->tokens, query_ast_token(self->ast, node));
  // The name of a capture leaves out its `@`.
  const uint32_t id = query_string_table_intern(
    &self->tables->captures,
    &self->source[span.start + 1],
    span.end - span.start - 1
  );
  for (uint32_t i = 0; i < self->pattern_captures.size; i++) {
    if (*array_get(&self->pattern_captures, i) == id) return id;
  }
  array_push(&self->pattern_captures, id);
  return id;
}

// Resolves the name of `node`, if it has one, other than those of predicates
// and their arguments.
static bool query_resolver_resolve_name(QueryResolver *self, uint32_t node) {
  const uint32_t token = query_ast_token(self->ast, node);
  switch (query_ast_kind(self->ast, node)) {
    case QueryAstKindNamedNode: {
      // `(_)` names no node type, so its id stays zero.
      if (query_token_list_kind(self->tokens, token) == QueryTokenKindUnderscore) return true;
      if (!query_resolver_resolve_node_type(self, node)) return false;
      const TSSymbol symbol = (TSSymbol)query_ast_ids_get(self->ids, node);
      // A supertype needs an explicit subtype to have child patterns.
      if (ts_language_symbol_metadata(self->language, symbol).supertype) {
        for (uint32_t child = node + 1; child < query_ast_end(self->ast, node); child = query_ast_end(self->ast, child)) {
          const QueryAstKind kind = query_ast_kind(self->ast, child);
          if (kind == QueryAstKindSubtype) break;
          if (kind == QueryAstKindPattern) return query_resolver_fail_at(self, QueryDiagnosticKindImpossiblePattern, child);
        }
      }
      return true;
    }
    case QueryAstKindAnonymousNode:
    case QueryAstKindMissingType:
      return query_resolver_resolve_node_type(self, node);
    case QueryAstKindSubtype:
      return query_resolver_resolve_subtype(self, node);
    case QueryAstKindField:
    case QueryAstKindNegatedField: {
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
      const uint32_t capture = query_resolver_define_capture(self, node);
      if (capture > UINT16_MAX) return query_resolver_fail_at(self, QueryDiagnosticKindTooManyCaptures, node);
      query_resolver_set_id(self, node, capture);
      return true;
    }
    // Predicates and their arguments are resolved after the rest of the pattern.
    case QueryAstKindPredicate:
    case QueryAstKindPredicateArgument:
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
  return true;
}

// Sets `index` to the position of `capture` in the current pattern's captures,
// or to their count if the pattern does not define it, which is where it would
// go. Returns whether the pattern defines it.
static bool query_resolver_pattern_capture_index(
  const QueryResolver *self,
  uint32_t capture,
  uint32_t *index
) {
  for (uint32_t i = 0; i < self->pattern_captures.size; i++) {
    if (*array_get(&self->pattern_captures, i) == capture) {
      *index = i;
      return true;
    }
  }
  *index = self->pattern_captures.size;
  return false;
}

// Resolves `node`, an argument of a predicate, and appends its step.
static bool query_resolver_resolve_argument(QueryResolver *self, uint32_t node) {
  const uint32_t token = query_ast_token(self->ast, node);
  const QuerySpan span = query_token_list_span(self->tokens, token);
  TSQueryPredicateStep step = {.type = TSQueryPredicateStepTypeString};
  switch (query_token_list_kind(self->tokens, token)) {
    case QueryTokenKindCapture: {
      step.type = TSQueryPredicateStepTypeCapture;
      uint32_t index;
      // The name of a capture leaves out its `@`.
      if (
        !query_string_table_find(
          &self->tables->captures,
          &self->source[span.start + 1],
          span.end - span.start - 1,
          &step.value_id
        ) ||
        !query_resolver_pattern_capture_index(self, step.value_id, &index)
      ) {
        return query_resolver_fail_at(self, QueryDiagnosticKindUndefinedCapture, node);
      }
      break;
    }
    case QueryTokenKindString: {
      uint32_t length;
      const char *value = query_resolver_string_value(self, token, &length);
      step.value_id = query_string_table_intern(&self->tables->strings, value, length);
      break;
    }
    default:
      step.value_id = query_string_table_intern(
        &self->tables->strings,
        &self->source[span.start],
        span.end - span.start
      );
      break;
  }
  query_resolver_set_id(self, node, step.value_id);
  array_push(&self->tables->predicate_steps, step);
  return true;
}

// Resolves the predicate `node` and its arguments, and appends their steps.
static bool query_resolver_resolve_predicate(QueryResolver *self, uint32_t node) {
  const uint32_t token = query_ast_token(self->ast, node);
  const QuerySpan span = query_token_list_span(self->tokens, token);
  // The name of a predicate leaves out its `#`, or the `.` of the legacy form,
  // whose `?` or `!` is two tokens after the `.`.
  uint32_t name_end = span.end;
  if (query_token_list_kind(self->tokens, token) == QueryTokenKindDot) {
    name_end = query_token_list_span(self->tokens, token + 2).end;
  }
  const uint32_t name = query_string_table_intern(
    &self->tables->strings,
    &self->source[span.start + 1],
    name_end - span.start - 1
  );
  query_resolver_set_id(self, node, name);
  array_push(&self->tables->predicate_steps, ((TSQueryPredicateStep) {
    .type = TSQueryPredicateStepTypeString,
    .value_id = name,
  }));

  // The arguments of a predicate are its children, and they are leaves.
  for (uint32_t argument = node + 1; argument < query_ast_end(self->ast, node); argument++) {
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
  uint32_t node,
  TSQuantifier (*combine)(TSQuantifier, TSQuantifier)
) {
  const uint32_t capture_count = self->pattern_captures.size;
  uint32_t pattern_count = 0;
  for (
    uint32_t child = node + 1;
    child < query_ast_end(self->ast, node);
    child = query_ast_end(self->ast, child)
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
static void query_resolver_finish_quantifiers(QueryResolver *self, uint32_t node) {
  const uint32_t capture_count = self->pattern_captures.size;
  uint8_t *quantifiers = array_get(&self->quantifiers, self->quantifiers.size - capture_count);
  uint32_t atom = node + 1;
  if (query_ast_kind(self->ast, atom) == QueryAstKindField) atom++;

  // A pattern's quantifiers and captures are leaves after its atom.
  TSQuantifier pattern_quantifier = TSQuantifierOne;
  for (uint32_t suffix = query_ast_end(self->ast, atom); suffix < query_ast_end(self->ast, node); suffix++) {
    if (query_ast_kind(self->ast, suffix) == QueryAstKindCapture) {
      uint32_t index;
      const bool is_defined = query_resolver_pattern_capture_index(
        self,
        query_ast_ids_get(self->ids, suffix),
        &index
      );
      // INVARIANT: Resolving the pattern's names defined each of its captures.
      ts_assert(is_defined);
      quantifiers[index] = (uint8_t)query_quantifier_add((TSQuantifier)quantifiers[index], TSQuantifierOne);
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
static void query_resolver_quantify_captures(QueryResolver *self, uint32_t pattern) {
  const uint32_t capture_count = self->pattern_captures.size;
  if (capture_count == 0) return;

  // Visit each node after all of its descendants. Each atom leaves one entry on
  // the stack, with a quantifier for every capture in the pattern.
  array_clear(&self->quantifiers);
  for (uint32_t node = query_ast_end(self->ast, pattern); node > pattern;) {
    node--;
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
      // Fields, captures, and quantifiers are handled with their pattern, and
      // the rest can contain no captures.
      case QueryAstKindField:
      case QueryAstKindSubtype:
      case QueryAstKindMissingType:
      case QueryAstKindAnchor:
      case QueryAstKindNegatedField:
      case QueryAstKindQuantifier:
      case QueryAstKindCapture:
      case QueryAstKindPredicate:
      case QueryAstKindPredicateArgument:
        break;
    }
  }

  // INVARIANT: The one entry left is the top-level pattern's.
  ts_assert(self->quantifiers.size == capture_count);
  for (uint32_t i = 0; i < capture_count; i++) {
    array_push(&self->tables->capture_quantifiers, ((QueryCaptureQuantifier) {
      .capture = *array_get(&self->pattern_captures, i),
      .quantifier = *array_get(&self->quantifiers, i),
    }));
  }
}

// Resolves the top-level pattern `pattern`, and appends what the query records
// about it.
static bool query_resolver_resolve_pattern(QueryResolver *self, uint32_t pattern) {
  const uint32_t end = query_ast_end(self->ast, pattern);
  array_clear(&self->pattern_captures);
  for (uint32_t node = pattern; node < end; node++) {
    if (!query_resolver_resolve_name(self, node)) return false;
  }

  // Predicates are resolved after the rest of the pattern, so that they can use
  // its captures wherever they are defined.
  const uint32_t predicate_step_offset = self->tables->predicate_steps.size;
  for (uint32_t node = pattern; node < end; node++) {
    if (
      query_ast_kind(self->ast, node) == QueryAstKindPredicate &&
      !query_resolver_resolve_predicate(self, node)
    ) return false;
  }

  const uint32_t capture_quantifier_offset = self->tables->capture_quantifiers.size;
  query_resolver_quantify_captures(self, pattern);

  // A top-level pattern's tokens run up to the first token of the next one, or
  // to the `End` token.
  const uint32_t next_token = end < query_ast_size(self->ast)
    ? query_ast_token(self->ast, end)
    : query_token_list_size(self->tokens) - 1;
  array_push(&self->tables->patterns, ((QueryPatternInfo) {
    .span = {
      .start = query_token_list_span(self->tokens, query_ast_token(self->ast, pattern)).start,
      .end = query_token_list_span(self->tokens, next_token - 1).end,
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
    .decoded = array_new(),
    .pattern_captures = array_new(),
    .quantifiers = array_new(),
  };
  array_clear(&ids->ids);
  array_grow_by(&ids->ids, query_ast_size(ast));
  bool succeeded = true;
  for (
    uint32_t pattern = 0;
    succeeded && pattern < query_ast_size(ast);
    pattern = query_ast_end(ast, pattern)
  ) {
    succeeded = query_resolver_resolve_pattern(&self, pattern);
  }
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
  return "?";
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
  for (uint32_t node = 0; node < query_ast_size(ast); node++) {
    const uint32_t token = query_ast_token(ast, node);
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
      default:
        break;
    }
    if (!kind) continue;
    printf(
      "%3u: %.*s -> %s %u %.*s\n",
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
    for (uint32_t j = 0; j < pattern->capture_quantifier_count; j++) {
      const QueryCaptureQuantifier *entry = array_get(
        &tables->capture_quantifiers,
        pattern->capture_quantifier_offset + j
      );
      uint32_t length;
      const char *capture = query_string_table_get(&tables->captures, entry->capture, &length);
      printf("  @%.*s %s\n", (int)length, capture, query_quantifier_name((TSQuantifier)entry->quantifier));
    }
    for (uint32_t j = 0; j < pattern->predicate_step_count; j++) {
      const TSQueryPredicateStep *step = array_get(&tables->predicate_steps, pattern->predicate_step_offset + j);
      uint32_t length;
      const char *value;
      switch (step->type) {
        case TSQueryPredicateStepTypeDone:
          printf("  done\n");
          break;
        case TSQueryPredicateStepTypeCapture:
          value = query_string_table_get(&tables->captures, step->value_id, &length);
          printf("  capture @%.*s\n", (int)length, value);
          break;
        case TSQueryPredicateStepTypeString:
          value = query_string_table_get(&tables->strings, step->value_id, &length);
          printf("  string %.*s\n", (int)length, value);
          break;
      }
    }
  }
}
#endif
