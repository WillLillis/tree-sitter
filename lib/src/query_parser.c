#include "./query_parser.h"

// A node that has been opened and not yet closed: a pattern whose atom has not
// ended yet, or a named node, group, or alternation that has not reached its
// closing delimiter.
typedef struct {
  QueryAstIndex node;
  // Whether a pattern has been parsed inside the node.
  bool has_pattern;
} QueryParserFrame;

// The state of one call to `ts_query_parse`.
typedef struct {
  const QueryTokenList *tokens;
  QueryAst *ast;
  // Receives a description of the error that stops parsing.
  QueryDiagnostic *error;
  // The index of the next token to parse.
  QueryTokenIndex token;
  // The open nodes, innermost last.
  Array(QueryParserFrame) frames;
} QueryParser;

// Returns the kind of the token `offset` tokens after the next one, or `End` if
// that is past the last token.
static QueryTokenKind query_parser_peek(const QueryParser *self, uint32_t offset) {
  const QueryTokenIndex index = (QueryTokenIndex) {
    .index = self->token.index + offset,
  };
  if (index.index >= query_token_list_size(self->tokens)) return QueryTokenKindEnd;
  return query_token_list_kind(self->tokens, index);
}

static void query_parser_advance_tokens(QueryParser *self, uint32_t count) {
  self->token.index += count;
}

static QueryTokenIndex query_parser_take_token(QueryParser *self) {
  const QueryTokenIndex token = self->token;
  query_parser_advance_tokens(self, 1);
  return token;
}

// Returns the index of the next token, which must be of the given kind, and moves
// past it.
#define query_parser_consume(self, kind) \
  (ts_assert(query_parser_peek(self, 0) == (kind)), query_parser_take_token(self))

// Returns whether a token of this kind can be a plain name, such as the name of a
// field.
static inline bool query_parser_is_word(QueryTokenKind kind) {
  return
    kind == QueryTokenKindIdentifier ||
    kind == QueryTokenKindUnderscore ||
    kind == QueryTokenKindMissing;
}

// Returns whether a node of this kind holds patterns until its closing
// delimiter: a named node, a group, or an alternation.
static inline bool query_parser_is_list(QueryAstKind kind) {
  return
    kind == QueryAstKindNamedNode ||
    kind == QueryAstKindGroup ||
    kind == QueryAstKindAlternation;
}

// Describes the error that stops parsing, and returns false.
static bool query_parser_fail(const QueryParser *self, QueryDiagnosticKind kind, QuerySpan span) {
  *self->error = (QueryDiagnostic) {.span = span, .kind = kind};
  return false;
}

// Describes the error that stops parsing as being at the token with the given
// index, and returns false.
static bool query_parser_fail_at(
  const QueryParser *self,
  QueryDiagnosticKind kind,
  QueryTokenIndex token
) {
  return query_parser_fail(self, kind, query_token_list_span(self->tokens, token));
}

// Returns whether the token with the given index starts where the token before
// it ends. If it does not, describes the space between them as the error that
// stops parsing.
static bool query_parser_expect_adjacent(
  const QueryParser *self,
  QueryTokenIndex token
) {
  const QuerySpan before = query_token_list_span(self->tokens, query_token_previous(token));
  const QuerySpan after = query_token_list_span(self->tokens, token);
  if (before.end == after.start) return true;
  return query_parser_fail(
    self,
    QueryDiagnosticKindUnexpectedSpace,
    (QuerySpan) {.start = before.end, .end = after.start}
  );
}

static void query_parser_push_frame(QueryParser *self, QueryAstIndex node) {
  array_push(&self->frames, ((QueryParserFrame) {.node = node, .has_pattern = false}));
}

// Returns the index of the token that opens `node`, a named node, group, or
// alternation.
static QueryTokenIndex query_parser_opener(const QueryParser *self, QueryAstIndex node) {
  const QueryTokenIndex token = query_ast_token(self->ast, node);
  const QueryAstKind kind = query_ast_kind(self->ast, node);
  ts_assert(query_parser_is_list(kind));
  // The token of a named node directly follows its `(`.
  if (kind == QueryAstKindNamedNode) {
    return query_token_previous(token);
  } else {
    return token;
  }
}

// Starts the named node whose name (an identifier or `_`) is the next token, and
// parses its subtype, if it has one.
static bool query_parser_begin_named_node(QueryParser *self) {
  const QueryTokenKind name_kind = query_parser_peek(self, 0);
  ts_assert(name_kind == QueryTokenKindIdentifier || name_kind == QueryTokenKindUnderscore);
  const QueryTokenIndex name = query_parser_take_token(self);
  const QueryAstIndex node = query_ast_open(self->ast, QueryAstKindNamedNode, name);
  if (query_parser_peek(self, 0) == QueryTokenKindSlash) {
    const QueryTokenIndex subtype = query_token_next(self->token);
    const QueryTokenKind subtype_kind = query_parser_peek(self, 1);
    if (!query_parser_expect_adjacent(self, self->token)) return false;
    if (!query_parser_is_word(subtype_kind) && subtype_kind != QueryTokenKindString) {
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedSubtype, subtype);
    }
    if (!query_parser_expect_adjacent(self, subtype)) return false;
    query_ast_push_leaf(self->ast, QueryAstKindSubtype, subtype);
    self->token = query_token_next(subtype);
  }
  query_parser_push_frame(self, node);
  return true;
}

// Parses the MISSING node whose `MISSING` is the next token. It has no children,
// except for the node type that can follow `MISSING`.
static bool query_parser_parse_missing_node(QueryParser *self) {
  const QueryTokenIndex missing = query_parser_consume(self, QueryTokenKindMissing);
  const QueryAstIndex node = query_ast_open(self->ast, QueryAstKindMissingNode, missing);
  QueryTokenKind kind = query_parser_peek(self, 0);
  if (query_parser_is_word(kind) || kind == QueryTokenKindString) {
    query_ast_push_leaf(self->ast, QueryAstKindMissingType, self->token);
    query_parser_advance_tokens(self, 1);
    kind = query_parser_peek(self, 0);
  }
  if (kind != QueryTokenKindRightParen) {
    return query_parser_fail_at(self, QueryDiagnosticKindExpectedRightParen, self->token);
  }
  query_parser_advance_tokens(self, 1);
  query_ast_close(self->ast, node);
  return true;
}

// Starts the atom whose `(` is the next token: a group, a named node, or a
// MISSING node, which is parsed completely.
static bool query_parser_begin_parenthesized(QueryParser *self) {
  const QueryTokenIndex paren = query_parser_consume(self, QueryTokenKindLeftParen);
  switch (query_parser_peek(self, 0)) {
    case QueryTokenKindLeftParen:
    case QueryTokenKindLeftBracket:
    case QueryTokenKindString:
      query_parser_push_frame(self, query_ast_open(self->ast, QueryAstKindGroup, paren));
      return true;
    case QueryTokenKindIdentifier:
    case QueryTokenKindUnderscore:
      return query_parser_begin_named_node(self);
    case QueryTokenKindMissing:
      return query_parser_parse_missing_node(self);
    case QueryTokenKindPredicateName:
    case QueryTokenKindDot:
      return query_parser_fail_at(self, QueryDiagnosticKindMisplacedPredicate, self->token);
    default:
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedNodeName, self->token);
  }
}

// Returns whether the next two tokens are the name of a field and its `:`.
static bool query_parser_at_field(const QueryParser *self) {
  return
    query_parser_is_word(query_parser_peek(self, 0)) &&
    query_parser_peek(self, 1) == QueryTokenKindColon;
}

// Starts the pattern at the next token: its field, if it has one, and its atom.
// An atom that cannot contain patterns is parsed completely.
static bool query_parser_begin_pattern(QueryParser *self) {
  query_parser_push_frame(self, query_ast_open(self->ast, QueryAstKindPattern, self->token));
  if (query_parser_at_field(self)) {
    query_ast_push_leaf(self->ast, QueryAstKindField, self->token);
    query_parser_advance_tokens(self, 2);
    if (query_parser_at_field(self)) {
      return query_parser_fail_at(self, QueryDiagnosticKindMultipleFields, self->token);
    }
  }

  switch (query_parser_peek(self, 0)) {
    case QueryTokenKindLeftParen:
      return query_parser_begin_parenthesized(self);
    case QueryTokenKindLeftBracket:
      query_parser_push_frame(self, query_ast_open(self->ast, QueryAstKindAlternation, self->token));
      query_parser_advance_tokens(self, 1);
      return true;
    case QueryTokenKindString:
      query_ast_push_leaf(self->ast, QueryAstKindAnonymousNode, self->token);
      query_parser_advance_tokens(self, 1);
      return true;
    case QueryTokenKindUnderscore:
      query_ast_push_leaf(self->ast, QueryAstKindWildcard, self->token);
      query_parser_advance_tokens(self, 1);
      return true;
    default:
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedPattern, self->token);
  }
}

// Finishes the innermost open pattern, whose atom has ended, by parsing its
// quantifiers and captures.
static void query_parser_finish_pattern(QueryParser *self) {
  ts_assert(query_ast_kind(self->ast, array_back(&self->frames)->node) == QueryAstKindPattern);
  for (;;) {
    const QueryTokenKind kind = query_parser_peek(self, 0);
    if (kind == QueryTokenKindCapture) {
      query_ast_push_leaf(self->ast, QueryAstKindCapture, self->token);
    } else if (
      kind == QueryTokenKindQuestion ||
      kind == QueryTokenKindStar ||
      kind == QueryTokenKindPlus
    ) {
      query_ast_push_leaf(self->ast, QueryAstKindQuantifier, self->token);
    } else {
      break;
    }
    query_parser_advance_tokens(self, 1);
  }
  query_ast_close(self->ast, array_back(&self->frames)->node);
  (void)array_pop(&self->frames);
  if (self->frames.size > 0) array_back(&self->frames)->has_pattern = true;
}

// Parses the legacy form of a predicate's name, `.name?` or `.name!`, whose `.` is
// the next token. There is no space inside it.
static bool query_parser_parse_legacy_predicate_name(QueryParser *self) {
  const QueryTokenIndex dot = query_parser_consume(self, QueryTokenKindDot);
  const QueryTokenIndex word = query_token_next(dot);
  const QueryTokenIndex suffix = query_token_next(word);
  const QueryTokenKind word_kind = query_parser_peek(self, 0);
  const QueryTokenKind suffix_kind = query_parser_peek(self, 1);
  if (!query_parser_is_word(word_kind)) {
    return query_parser_fail_at(
      self,
      QueryDiagnosticKindExpectedPredicateName,
      word
    );
  }
  if (!query_parser_expect_adjacent(self, word)) return false;
  if (suffix_kind != QueryTokenKindQuestion && suffix_kind != QueryTokenKindBang) {
    return query_parser_fail(self, QueryDiagnosticKindMissingPredicateSuffix, (QuerySpan) {
      .start = query_token_list_span(self->tokens, dot).start,
      .end = query_token_list_span(self->tokens, word).end,
    });
  }
  if (!query_parser_expect_adjacent(self, suffix)) return false;
  query_parser_advance_tokens(self, 2);
  return true;
}

// Parses the predicate whose `(` is the next token, followed by its name: a
// `#name?` token, or the `.` of the legacy form.
static bool query_parser_parse_predicate(QueryParser *self) {
  const QueryTokenIndex paren = query_parser_consume(self, QueryTokenKindLeftParen);
  const QueryTokenIndex name = self->token;
  if (query_parser_peek(self, 0) == QueryTokenKindDot) {
    if (!query_parser_parse_legacy_predicate_name(self)) return false;
  } else {
    (void)query_parser_consume(self, QueryTokenKindPredicateName);
  }

  const QueryAstIndex predicate = query_ast_open(self->ast, QueryAstKindPredicate, name);
  for (;;) {
    const QueryTokenKind kind = query_parser_peek(self, 0);
    if (kind == QueryTokenKindRightParen) break;
    if (kind == QueryTokenKindEnd) {
      return query_parser_fail_at(self, QueryDiagnosticKindUnclosedDelimiter, paren);
    }
    if (
      kind != QueryTokenKindCapture &&
      kind != QueryTokenKindString &&
      !query_parser_is_word(kind)
    ) {
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedPredicateArgument, self->token);
    }
    query_ast_push_leaf(self->ast, QueryAstKindPredicateArgument, self->token);
    query_parser_advance_tokens(self, 1);
  }
  query_parser_advance_tokens(self, 1);
  query_ast_close(self->ast, predicate);
  return true;
}

// Parses the negated field whose `!` is the next token.
static bool query_parser_parse_negated_field(QueryParser *self) {
  (void)query_parser_consume(self, QueryTokenKindBang);
  if (!query_parser_is_word(query_parser_peek(self, 0))) {
    return query_parser_fail_at(self, QueryDiagnosticKindExpectedFieldName, self->token);
  }
  query_ast_push_leaf(self->ast, QueryAstKindNegatedField, self->token);
  query_parser_advance_tokens(self, 1);
  return true;
}

// Returns whether the next token begins a predicate inside a list of the given
// kind. Alternations only hold patterns.
static bool query_parser_at_predicate(const QueryParser *self, QueryAstKind list_kind) {
  if (list_kind == QueryAstKindAlternation) return false;
  if (query_parser_peek(self, 0) != QueryTokenKindLeftParen) return false;
  const QueryTokenKind next = query_parser_peek(self, 1);
  return next == QueryTokenKindPredicateName || next == QueryTokenKindDot;
}

// Returns whether the next token begins a negated field inside a list of the
// given kind. Only named nodes have fields.
static bool query_parser_at_negated_field(const QueryParser *self, QueryAstKind list_kind) {
  return list_kind == QueryAstKindNamedNode && query_parser_peek(self, 0) == QueryTokenKindBang;
}

// Returns whether an anchor can be followed by the next token inside the
// innermost open named node, group, or alternation.
static bool query_parser_is_valid_anchor(const QueryParser *self) {
  const QueryParserFrame *frame = array_back(&self->frames);
  ts_assert(query_parser_is_list(query_ast_kind(self->ast, frame->node)));
  const QueryTokenKind next = query_parser_peek(self, 0);
  switch (query_ast_kind(self->ast, frame->node)) {
    case QueryAstKindNamedNode:
      // An anchor before the `)` makes the pattern before it the last child.
      if (next == QueryTokenKindRightParen) return frame->has_pattern;
      return next != QueryTokenKindDot;
    case QueryAstKindGroup:
      // A group is not a node, so it has no first or last child to anchor.
      return
        frame->has_pattern &&
        next != QueryTokenKindRightParen &&
        next != QueryTokenKindDot;
    default:
      return false;
  }
}

// Parses the anchor whose `.` is the next token, and the predicates and negated
// fields after it. Neither is a sibling of the patterns around it, so the item
// after them decides whether the anchor is valid.
static bool query_parser_parse_anchor(QueryParser *self, QueryAstKind list_kind) {
  const QueryTokenIndex anchor = query_parser_consume(self, QueryTokenKindDot);
  query_ast_push_leaf(self->ast, QueryAstKindAnchor, anchor);
  for (;;) {
    if (query_parser_at_predicate(self, list_kind)) {
      if (!query_parser_parse_predicate(self)) return false;
    } else if (query_parser_at_negated_field(self, list_kind)) {
      if (!query_parser_parse_negated_field(self)) return false;
    } else {
      break;
    }
  }
  if (!query_parser_is_valid_anchor(self)) {
    return query_parser_fail_at(self, QueryDiagnosticKindMisplacedAnchor, anchor);
  }
  return true;
}

// Parses the next item inside the innermost open named node, group, or
// alternation, or closes it.
static bool query_parser_continue_list(QueryParser *self) {
  const QueryParserFrame *frame = array_back(&self->frames);
  const QueryAstIndex list = frame->node;
  const QueryAstKind list_kind = query_ast_kind(self->ast, list);
  ts_assert(query_parser_is_list(list_kind));
  const QueryTokenKind kind = query_parser_peek(self, 0);
  switch (kind) {
    case QueryTokenKindEnd:
      return query_parser_fail_at(
        self,
        QueryDiagnosticKindUnclosedDelimiter,
        query_parser_opener(self, list)
      );
    case QueryTokenKindRightParen:
    case QueryTokenKindRightBracket: {
      const QueryTokenKind closer = list_kind == QueryAstKindAlternation
        ? QueryTokenKindRightBracket
        : QueryTokenKindRightParen;
      if (kind != closer) {
        return query_parser_fail_at(self, QueryDiagnosticKindUnexpectedClosingDelimiter, self->token);
      }
      if (list_kind != QueryAstKindNamedNode && !frame->has_pattern) {
        return query_parser_fail_at(self, QueryDiagnosticKindExpectedPattern, self->token);
      }
      query_ast_close(self->ast, list);
      (void)array_pop(&self->frames);
      query_parser_advance_tokens(self, 1);
      return true;
    }
    case QueryTokenKindDot:
      return query_parser_parse_anchor(self, list_kind);
    case QueryTokenKindBang:
      if (query_parser_at_negated_field(self, list_kind)) return query_parser_parse_negated_field(self);
      break;
    case QueryTokenKindLeftParen:
      if (query_parser_at_predicate(self, list_kind)) return query_parser_parse_predicate(self);
      break;
    default:
      break;
  }
  return query_parser_begin_pattern(self);
}

// Starts the next pattern at the top level of the query.
static bool query_parser_begin_top_level_pattern(QueryParser *self) {
  ts_assert(self->frames.size == 0);
  switch (query_parser_peek(self, 0)) {
    case QueryTokenKindRightParen:
    case QueryTokenKindRightBracket:
      return query_parser_fail_at(self, QueryDiagnosticKindUnexpectedClosingDelimiter, self->token);
    case QueryTokenKindDot:
      return query_parser_fail_at(self, QueryDiagnosticKindMisplacedAnchor, self->token);
    default:
      return query_parser_begin_pattern(self);
  }
}

bool ts_query_parse(const QueryTokenList *tokens, QueryAst *ast, QueryDiagnostic *error) {
  ts_assert(query_token_list_size(tokens) <= QUERY_TOKEN_LIST_MAX_SIZE);
  QueryParser self = {
    .tokens = tokens,
    .ast = ast,
    .error = error,
    .token = {.index = 0},
    .frames = array_new(),
  };
  query_ast_reserve(ast, query_token_list_size(tokens));
  bool succeeded = true;
  while (succeeded && (self.frames.size > 0 || query_parser_peek(&self, 0) != QueryTokenKindEnd)) {
    if (self.frames.size == 0) {
      succeeded = query_parser_begin_top_level_pattern(&self);
    } else if (query_ast_kind(ast, array_back(&self.frames)->node) == QueryAstKindPattern) {
      query_parser_finish_pattern(&self);
    } else {
      succeeded = query_parser_continue_list(&self);
    }
  }
  array_delete(&self.frames);
  return succeeded;
}

#ifdef DEBUG_QUERY_AST
static const char *query_ast_kind_name(QueryAstKind kind) {
  switch (kind) {
    case QueryAstKindPattern: return "Pattern";
    case QueryAstKindField: return "Field";
    case QueryAstKindNamedNode: return "NamedNode";
    case QueryAstKindSubtype: return "Subtype";
    case QueryAstKindMissingNode: return "MissingNode";
    case QueryAstKindMissingType: return "MissingType";
    case QueryAstKindAnonymousNode: return "AnonymousNode";
    case QueryAstKindWildcard: return "Wildcard";
    case QueryAstKindAlternation: return "Alternation";
    case QueryAstKindGroup: return "Group";
    case QueryAstKindAnchor: return "Anchor";
    case QueryAstKindNegatedField: return "NegatedField";
    case QueryAstKindQuantifier: return "Quantifier";
    case QueryAstKindCapture: return "Capture";
    case QueryAstKindPredicate: return "Predicate";
    case QueryAstKindPredicateArgument: return "PredicateArgument";
  }
  return "?";
}

void ts_query_ast_dump(const QueryAst *self, const QueryTokenList *tokens, const char *source) {
  printf("=== AST ===\n");
  // The ends of the subtrees that contain the current node.
  Array(QueryAstIndex) ends = array_new();
  for (
    QueryAstIndex node = {.index = 0};
    node < query_ast_size(self);
    node++
  ) {
    while (
      ends.size > 0 &&
      array_back(&ends)->index <= node.index
    ) {
      (void)array_pop(&ends);
    }
    const QuerySpan span = query_token_list_span(tokens, query_ast_token(self, node));
    printf(
      "%3u: %*s%s %.*s\n",
      node.index,
      (int)(2 * ends.size),
      "",
      query_ast_kind_name(query_ast_kind(self, node)),
      (int)(span.end - span.start),
      &source[span.start]
    );
    array_push(&ends, query_ast_end(self, node));
  }
  array_delete(&ends);
}
#endif
