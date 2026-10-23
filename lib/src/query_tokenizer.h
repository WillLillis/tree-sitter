#ifndef TREE_SITTER_QUERY_TOKENIZER_H_
#define TREE_SITTER_QUERY_TOKENIZER_H_

#include "./array.h"
#include "./query_diagnostic.h"

#include <stdbool.h>
#include <stdint.h>

// #define DEBUG_QUERY_TOKENS

// A kind of token in the source of a query.
typedef enum {
  // `(`
  QueryTokenKindLeftParen,
  // `)`
  QueryTokenKindRightParen,
  // `[`
  QueryTokenKindLeftBracket,
  // `]`
  QueryTokenKindRightBracket,
  // `.`, an anchor. It also begins the legacy `(.eq? ...)` form of a predicate.
  QueryTokenKindDot,
  // `!`, which precedes the name of a negated field.
  QueryTokenKindBang,
  // `:`, which follows the name of a field.
  QueryTokenKindColon,
  // `/`, which separates a supertype from one of its subtypes.
  QueryTokenKindSlash,
  // `?`, the zero-or-one quantifier.
  QueryTokenKindQuestion,
  // `*`, the zero-or-more quantifier.
  QueryTokenKindStar,
  // `+`, the one-or-more quantifier.
  QueryTokenKindPlus,
  // `_`, the wildcard.
  QueryTokenKindUnderscore,
  // `MISSING`, which matches nodes that the parser inserted to recover from an error.
  QueryTokenKindMissing,
  // The name of a node or a field, or a bare argument to a predicate.
  QueryTokenKindIdentifier,
  // `@`, followed by the name of a capture.
  QueryTokenKindCapture,
  // `#`, followed by the name of a predicate, which ends with `?`, or of a
  // directive, which ends with `!`.
  QueryTokenKindPredicateName,
  // A double-quoted string, including its quotes.
  QueryTokenKindString,
  // The end of the source. It is always the last token, and it is always empty.
  QueryTokenKindEnd,
} QueryTokenKind;

_Static_assert(
  QueryTokenKindEnd <= UINT8_MAX,
  "QueryTokenList stores each QueryTokenKind in a byte"
);

// The most tokens that a query can have.
#define QUERY_TOKEN_LIST_MAX_SIZE (1u << 27)

// The tokens of the source of a query.
typedef struct {
  Array(uint8_t) kinds;
  Array(QuerySpan) spans;
} QueryTokenList;

static inline QueryTokenList query_token_list_new(void) {
  return (QueryTokenList) {
    .kinds = array_new(),
    .spans = array_new(),
  };
}

static inline void query_token_list_delete(QueryTokenList *self) {
  array_delete(&self->kinds);
  array_delete(&self->spans);
}

static inline void query_token_list_reserve(QueryTokenList *self, uint32_t new_capacity) {
  array_reserve(&self->kinds, new_capacity);
  array_reserve(&self->spans, new_capacity);
}

static inline uint32_t query_token_list_size(const QueryTokenList *self) {
  return self->kinds.size;
}

static inline QueryTokenKind query_token_list_kind(const QueryTokenList *self, uint32_t index) {
  return (QueryTokenKind)*array_get(&self->kinds, index);
}

static inline QuerySpan query_token_list_span(const QueryTokenList *self, uint32_t index) {
  return *array_get(&self->spans, index);
}

static inline void query_token_list_push(QueryTokenList *self, QueryTokenKind kind, QuerySpan span) {
  array_push(&self->kinds, (uint8_t)kind);
  array_push(&self->spans, span);
}

// Sets `value` to the byte that the escape sequence made of a backslash and
// `escaped` stands for, and returns whether that escape sequence is valid.
static inline bool query_token_unescape(uint8_t escaped, char *value) {
  switch (escaped) {
    case 'n': *value = '\n'; return true;
    case 'r': *value = '\r'; return true;
    case 't': *value = '\t'; return true;
    case '0': *value = '\0'; return true;
    case '\\': *value = '\\'; return true;
    case '"': *value = '"'; return true;
    default: *value = (char)escaped; return false;
  }
}

// Splits `source` into tokens, and appends them to `tokens`, ending with the
// `End` token. Returns false at the first error in the source, after
// describing it in `error`.
bool ts_query_tokenize(
  const char *source,
  uint32_t length,
  QueryTokenList *tokens,
  QueryDiagnostic *error
);

#ifdef DEBUG_QUERY_TOKENS
void ts_query_token_list_dump(const QueryTokenList *self, const char *source);
#endif

#endif  // TREE_SITTER_QUERY_TOKENIZER_H_
