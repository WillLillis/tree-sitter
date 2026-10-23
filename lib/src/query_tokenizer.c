#include "./query_tokenizer.h"
#include "./unicode.h"

#include <string.h>

// The state of one call to `ts_query_tokenize`.
typedef struct {
  const char *source;
  // The number of bytes in `source`.
  uint32_t length;
  // The offset of the next byte to tokenize.
  uint32_t offset;
  // Receives a description of the error that stops tokenization.
  QueryDiagnostic *error;
} QueryTokenizer;

// Returns whether `byte` can appear anywhere in a name.
static inline bool query_tokenizer_is_name_byte(uint8_t byte) {
  return
    (byte >= 'a' && byte <= 'z') ||
    (byte >= 'A' && byte <= 'Z') ||
    (byte >= '0' && byte <= '9') ||
    byte == '_' ||
    byte == '-';
}

// Returns whether `byte` is a whole ASCII character, rather than part of a longer
// UTF-8 character.
static inline bool query_tokenizer_is_ascii(uint8_t byte) {
  return byte < 0x80;
}

// Describes the error that stops tokenization, and returns false.
static bool query_tokenizer_fail(
  const QueryTokenizer *self,
  QueryDiagnosticKind kind,
  uint32_t start,
  uint32_t end
) {
  *self->error = (QueryDiagnostic) {
    .span = {.start = start, .end = end},
    .kind = kind,
  };
  return false;
}

// Measures the UTF-8 character that begins at `offset`. Sets `size` to the
// number of bytes it occupies, or, if the bytes there are not valid UTF-8, to
// the number of bytes in the invalid sequence. Returns whether they are valid.
static bool query_tokenizer_measure_character(
  const QueryTokenizer *self,
  uint32_t offset,
  uint32_t *size
) {
  int32_t code_point;
  *size = ts_decode_utf8(
    (const uint8_t *)&self->source[offset],
    self->length - offset,
    &code_point
  );
  return code_point >= 0;
}

// Describes the character at `offset`, which cannot start a token, as the
// error that stops tokenization, and returns false.
static bool query_tokenizer_fail_at_character(const QueryTokenizer *self, uint32_t offset) {
  if (query_tokenizer_is_ascii((uint8_t)self->source[offset])) {
    return query_tokenizer_fail(self, QueryDiagnosticKindUnexpectedCharacter, offset, offset + 1);
  }
  uint32_t size;
  if (query_tokenizer_measure_character(self, offset, &size)) {
    return query_tokenizer_fail(self, QueryDiagnosticKindNonAsciiCharacter, offset, offset + size);
  }
  return query_tokenizer_fail(self, QueryDiagnosticKindInvalidUtf8, offset, offset + size);
}

// Advances past any whitespace and comments. A comment runs from a `;` to the
// end of its line.
static void query_tokenizer_skip_trivia(QueryTokenizer *self) {
  while (self->offset < self->length) {
    switch (self->source[self->offset]) {
      case ' ':
      case '\t':
      case '\n':
      case '\v':
      case '\f':
      case '\r':
        self->offset++;
        break;
      case ';': {
        const char *newline = memchr(&self->source[self->offset], '\n', self->length - self->offset);
        self->offset = newline ? (uint32_t)(newline - self->source) : self->length;
        break;
      }
      default:
        return;
    }
  }
}

// Advances past the name that begins at the current offset, if there is one. A
// `.` can appear in a name, but only between two name bytes, so that a name
// directly followed by an anchor does not absorb it.
static void query_tokenizer_scan_name(QueryTokenizer *self) {
  const uint32_t start = self->offset;
  while (self->offset < self->length) {
    const uint8_t byte = (uint8_t)self->source[self->offset];
    if (query_tokenizer_is_name_byte(byte)) {
      self->offset++;
    } else if (
      byte == '.' &&
      self->offset > start &&
      self->offset + 1 < self->length &&
      query_tokenizer_is_name_byte((uint8_t)self->source[self->offset + 1])
    ) {
      self->offset += 2;
    } else {
      break;
    }
  }
}

// Advances past a sigil, such as `@` or `#`, and the name that follows it. If no
// name follows the sigil, describes that as an error of the given kind, unless
// the next character is a more specific error, and returns false.
static bool query_tokenizer_scan_sigil_name(
  QueryTokenizer *self,
  QueryDiagnosticKind missing_name_kind
) {
  const uint32_t start = self->offset;
  const uint32_t name_start = start + 1;
  self->offset = name_start;
  query_tokenizer_scan_name(self);
  if (self->offset > name_start) return true;
  if (
    name_start < self->length &&
    !query_tokenizer_is_ascii((uint8_t)self->source[name_start])
  ) {
    return query_tokenizer_fail_at_character(self, name_start);
  }
  return query_tokenizer_fail(self, missing_name_kind, start, name_start);
}

// Advances past a predicate name: a `#`, a name, and the `?` or `!` that ends it.
static bool query_tokenizer_scan_predicate_name(QueryTokenizer *self) {
  const uint32_t start = self->offset;
  if (!query_tokenizer_scan_sigil_name(self, QueryDiagnosticKindMissingPredicateName)) {
    return false;
  }
  if (
    self->offset == self->length ||
    (self->source[self->offset] != '?' && self->source[self->offset] != '!')
  ) {
    return query_tokenizer_fail(
      self,
      QueryDiagnosticKindMissingPredicateSuffix,
      start,
      self->offset
    );
  }
  self->offset++;
  return true;
}

// Advances past a string, from its opening quote to its closing quote. A string
// cannot contain a line break, and each backslash in it must begin a valid
// escape sequence.
static bool query_tokenizer_scan_string(QueryTokenizer *self) {
  const uint32_t start = self->offset;
  self->offset++;
  for (;;) {
    const uint32_t offset = self->offset;
    if (offset == self->length || self->source[offset] == '\n') {
      return query_tokenizer_fail(self, QueryDiagnosticKindUnterminatedString, start, offset);
    }
    const uint8_t byte = (uint8_t)self->source[offset];
    if (byte == '"') {
      self->offset++;
      return true;
    }

    uint32_t size = 1;
    if (byte == '\\') {
      if (offset + 1 == self->length) {
        return query_tokenizer_fail(self, QueryDiagnosticKindUnterminatedString, start, offset + 1);
      }
      char value;
      uint8_t escaped = (uint8_t)self->source[offset + 1];
      if (!query_token_unescape(escaped, &value)) {
        uint32_t escaped_size = 1;
        if (
          !query_tokenizer_is_ascii(escaped) &&
          !query_tokenizer_measure_character(self, offset + 1, &escaped_size)
        ) {
          return query_tokenizer_fail(
            self,
            QueryDiagnosticKindInvalidUtf8,
            offset + 1,
            offset + 1 + escaped_size
          );
        }
        return query_tokenizer_fail(
          self,
          QueryDiagnosticKindUnknownEscape,
          offset,
          offset + 1 + escaped_size
        );
      }
      size = 2;
    } else if (
      !query_tokenizer_is_ascii(byte) &&
      !query_tokenizer_measure_character(self, offset, &size)
    ) {
      return query_tokenizer_fail(self, QueryDiagnosticKindInvalidUtf8, offset, offset + size);
    }
    self->offset += size;
  }
}

// Sets `kind` to the kind of the one-byte token that `byte` makes up, and
// returns whether there is one.
static bool query_tokenizer_punctuation_kind(uint8_t byte, QueryTokenKind *kind) {
  switch (byte) {
    case '(': *kind = QueryTokenKindLeftParen; return true;
    case ')': *kind = QueryTokenKindRightParen; return true;
    case '[': *kind = QueryTokenKindLeftBracket; return true;
    case ']': *kind = QueryTokenKindRightBracket; return true;
    case '.': *kind = QueryTokenKindDot; return true;
    case '!': *kind = QueryTokenKindBang; return true;
    case ':': *kind = QueryTokenKindColon; return true;
    case '/': *kind = QueryTokenKindSlash; return true;
    case '?': *kind = QueryTokenKindQuestion; return true;
    case '*': *kind = QueryTokenKindStar; return true;
    case '+': *kind = QueryTokenKindPlus; return true;
    default: return false;
  }
}

// Returns the kind of the token whose text is the name `word`.
static QueryTokenKind query_tokenizer_word_kind(const char *word, uint32_t length) {
#define MISSING_IDENT_LEN (sizeof("MISSING") - 1)
  if (length == 1 && word[0] == '_') return QueryTokenKindUnderscore;
  if (length == MISSING_IDENT_LEN && memcmp(word, "MISSING", MISSING_IDENT_LEN) == 0) {
    return QueryTokenKindMissing;
  }
  return QueryTokenKindIdentifier;
#undef MISSING_IDENT_LEN
}

bool ts_query_tokenize(
  const char *source,
  uint32_t length,
  QueryTokenList *tokens,
  QueryDiagnostic *error
) {
  QueryTokenizer self = {.source = source, .length = length, .offset = 0, .error = error};
  // Real queries average about six bytes per token, and almost never fewer than
  // four, so this is enough to avoid growing the list for nearly every query.
  query_token_list_reserve(tokens, query_token_list_size(tokens) + length / 4 + 1);
  for (;;) {
    query_tokenizer_skip_trivia(&self);
    uint32_t start = self.offset;
    if (query_token_list_size(tokens) >= QUERY_TOKEN_LIST_MAX_SIZE) {
      return query_tokenizer_fail(&self, QueryDiagnosticKindTooManyTokens, start, start);
    }
    if (start == length) {
      query_token_list_push(tokens, QueryTokenKindEnd, (QuerySpan) {.start = length, .end = length});
      return true;
    }

    QueryTokenKind kind;
    uint8_t byte = (uint8_t)source[start];
    if (query_tokenizer_punctuation_kind(byte, &kind)) {
      self.offset++;
    } else if (byte == '"') {
      kind = QueryTokenKindString;
      if (!query_tokenizer_scan_string(&self)) return false;
    } else if (byte == '@') {
      kind = QueryTokenKindCapture;
      if (!query_tokenizer_scan_sigil_name(&self, QueryDiagnosticKindMissingCaptureName)) {
        return false;
      }
    } else if (byte == '#') {
      kind = QueryTokenKindPredicateName;
      if (!query_tokenizer_scan_predicate_name(&self)) return false;
    } else if (query_tokenizer_is_name_byte(byte)) {
      query_tokenizer_scan_name(&self);
      kind = query_tokenizer_word_kind(&source[start], self.offset - start);
    } else {
      return query_tokenizer_fail_at_character(&self, start);
    }
    query_token_list_push(tokens, kind, (QuerySpan) {.start = start, .end = self.offset});
  }
}

#ifdef DEBUG_QUERY_TOKENS
static const char *query_token_kind_name(QueryTokenKind kind) {
  switch (kind) {
    case QueryTokenKindLeftParen: return "LeftParen";
    case QueryTokenKindRightParen: return "RightParen";
    case QueryTokenKindLeftBracket: return "LeftBracket";
    case QueryTokenKindRightBracket: return "RightBracket";
    case QueryTokenKindDot: return "Dot";
    case QueryTokenKindBang: return "Bang";
    case QueryTokenKindColon: return "Colon";
    case QueryTokenKindSlash: return "Slash";
    case QueryTokenKindQuestion: return "Question";
    case QueryTokenKindStar: return "Star";
    case QueryTokenKindPlus: return "Plus";
    case QueryTokenKindUnderscore: return "Underscore";
    case QueryTokenKindMissing: return "Missing";
    case QueryTokenKindIdentifier: return "Identifier";
    case QueryTokenKindCapture: return "Capture";
    case QueryTokenKindPredicateName: return "PredicateName";
    case QueryTokenKindString: return "String";
    case QueryTokenKindEnd: return "End";
  }
  return "?";
}

void ts_query_token_list_dump(const QueryTokenList *self, const char *source) {
  printf("=== TOKENS ===\n");
  for (uint32_t i = 0; i < query_token_list_size(self); i++) {
    QueryTokenKind kind = query_token_list_kind(self, i);
    QuerySpan span = query_token_list_span(self, i);
    printf(
      "%3u: %s %u..%u %.*s\n",
      i,
      query_token_kind_name(kind),
      span.start,
      span.end,
      (int)(span.end - span.start),
      &source[span.start]
    );
  }
}
#endif
