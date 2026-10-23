#ifndef TREE_SITTER_QUERY_RESOLVER_H_
#define TREE_SITTER_QUERY_RESOLVER_H_

#include "./array.h"
#include "./query_diagnostic.h"
#include "./query_parser.h"
#include "./query_tokenizer.h"
#include "tree_sitter/api.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

// #define DEBUG_QUERY_RESOLUTION

// The id that each node of a query's syntax tree resolves to, by node index: a
// symbol, a field, a capture, or a string, depending on the node's kind. It is
// zero for `(_)`, and for nodes that name nothing.
typedef struct {
  Array(uint32_t) ids;
} QueryAstIds;

static inline QueryAstIds query_ast_ids_new(void) {
  return (QueryAstIds) {.ids = array_new()};
}

static inline void query_ast_ids_delete(QueryAstIds *self) {
  array_delete(&self->ids);
}

static inline uint32_t query_ast_ids_get(const QueryAstIds *self, uint32_t node) {
  return *array_get(&self->ids, node);
}

// Distinct strings, each identified by its index in `offsets`.
typedef struct {
  // The bytes of every string, each followed by a NUL.
  Array(char) bytes;
  // Where each string starts in `bytes`.
  Array(uint32_t) offsets;
} QueryStringTable;

static inline QueryStringTable query_string_table_new(void) {
  return (QueryStringTable) {.bytes = array_new(), .offsets = array_new()};
}

static inline void query_string_table_delete(QueryStringTable *self) {
  array_delete(&self->bytes);
  array_delete(&self->offsets);
}

static inline uint32_t query_string_table_size(const QueryStringTable *self) {
  return self->offsets.size;
}

// Returns the bytes of the string with the given id, and sets `length` to how
// many there are. The escape sequence `\0` puts NUL bytes inside strings, so a
// string's length comes from where the next one starts, not from its NUL.
static inline const char *query_string_table_get(
  const QueryStringTable *self,
  uint32_t id,
  uint32_t *length
) {
  const uint32_t start = *array_get(&self->offsets, id);
  const uint32_t end = id + 1 < self->offsets.size
    ? *array_get(&self->offsets, id + 1)
    : self->bytes.size;
  *length = end - start - 1;
  return array_get(&self->bytes, start);
}

// Sets `id` to the id of the string with the given bytes, and returns whether
// `self` has it.
static inline bool query_string_table_find(
  const QueryStringTable *self,
  const char *bytes,
  uint32_t length,
  uint32_t *id
) {
  for (uint32_t i = 0; i < query_string_table_size(self); i++) {
    uint32_t candidate_length;
    const char *candidate = query_string_table_get(self, i, &candidate_length);
    if (candidate_length == length && memcmp(candidate, bytes, length) == 0) {
      *id = i;
      return true;
    }
  }
  return false;
}

// How many times a capture occurs in each match of a pattern.
typedef struct {
  uint32_t capture;
  // A TSQuantifier.
  uint32_t quantifier;
} QueryCaptureQuantifier;

// What a query records about one of its patterns.
typedef struct {
  // The pattern's text in the source.
  QuerySpan span;
  // The pattern's range of `predicate_steps`.
  uint32_t predicate_step_offset;
  uint32_t predicate_step_count;
  // The pattern's range of `capture_quantifiers`.
  uint32_t capture_quantifier_offset;
  uint32_t capture_quantifier_count;
} QueryPatternInfo;

// What resolution produces that lasts as long as the query.
typedef struct {
  // The names of the captures, without their `@`, in order of first appearance.
  QueryStringTable captures;
  // The names of predicates and their arguments that are not captures.
  QueryStringTable strings;
  Array(TSQueryPredicateStep) predicate_steps;
  Array(QueryCaptureQuantifier) capture_quantifiers;
  Array(QueryPatternInfo) patterns;
} QueryTables;

static inline QueryTables query_tables_new(void) {
  return (QueryTables) {
    .captures = query_string_table_new(),
    .strings = query_string_table_new(),
    .predicate_steps = array_new(),
    .capture_quantifiers = array_new(),
    .patterns = array_new(),
  };
}

static inline void query_tables_delete(QueryTables *self) {
  query_string_table_delete(&self->captures);
  query_string_table_delete(&self->strings);
  array_delete(&self->predicate_steps);
  array_delete(&self->capture_quantifiers);
  array_delete(&self->patterns);
}

static inline void query_tables_copy(QueryTables *self, const QueryTables *other) {
  *self = query_tables_new();
  array_assign(&self->captures.bytes, &other->captures.bytes);
  array_assign(&self->captures.offsets, &other->captures.offsets);
  array_assign(&self->strings.bytes, &other->strings.bytes);
  array_assign(&self->strings.offsets, &other->strings.offsets);
  array_assign(&self->predicate_steps, &other->predicate_steps);
  array_assign(&self->capture_quantifiers, &other->capture_quantifiers);
  array_assign(&self->patterns, &other->patterns);
}

// Resolves the names in `ast`, the syntax tree of `source`, against `language`.
// Sets `ids` to the id of each node, and adds the query's captures, strings,
// predicates, and patterns to `tables`. Returns false at the first name that
// the language does not have, or capture that a predicate's pattern does not
// define, after describing it in `error`.
bool ts_query_resolve(
  const TSLanguage *language,
  const char *source,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  QueryAstIds *ids,
  QueryTables *tables,
  QueryDiagnostic *error
);

#ifdef DEBUG_QUERY_RESOLUTION
void ts_query_resolution_dump(
  const TSLanguage *language,
  const char *source,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  const QueryAstIds *ids,
  const QueryTables *tables
);
#endif

#endif  // TREE_SITTER_QUERY_RESOLVER_H_
