#include "./query_parser.h"

// A node that has been opened and not yet closed: a pattern whose atom has not
// ended yet, or a named node, group, or alternation that has not reached its
// closing delimiter.
typedef struct {
  uint32_t node;
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
  uint32_t token;
  // The open nodes, innermost last.
  Array(QueryParserFrame) frames;
} QueryParser;

// Returns the kind of the token `offset` tokens after the next one, or `End` if
// that is past the last token.
static QueryTokenKind query_parser_peek(const QueryParser *self, uint32_t offset) {
  const uint32_t index = self->token + offset;
  if (index >= query_token_list_size(self->tokens)) return QueryTokenKindEnd;
  return query_token_list_kind(self->tokens, index);
}

// Returns whether a token of this kind can be a plain name, such as the name of a
// field.
static inline bool query_parser_is_word(QueryTokenKind kind) {
  return
    kind == QueryTokenKindIdentifier ||
    kind == QueryTokenKindUnderscore ||
    kind == QueryTokenKindMissing;
}

// Describes the error that stops parsing, and returns false.
static bool query_parser_fail(const QueryParser *self, QueryDiagnosticKind kind, QuerySpan span) {
  *self->error = (QueryDiagnostic) {.span = span, .kind = kind};
  return false;
}

// Describes the error that stops parsing as being at the token with the given
// index, and returns false.
static bool query_parser_fail_at(const QueryParser *self, QueryDiagnosticKind kind, uint32_t token) {
  return query_parser_fail(self, kind, query_token_list_span(self->tokens, token));
}

// Returns whether the token with the given index starts where the token before
// it ends. If it does not, describes the space between them as the error that
// stops parsing.
static bool query_parser_expect_adjacent(const QueryParser *self, uint32_t token) {
  const QuerySpan before = query_token_list_span(self->tokens, token - 1);
  const QuerySpan after = query_token_list_span(self->tokens, token);
  if (before.end == after.start) return true;
  return query_parser_fail(
    self,
    QueryDiagnosticKindUnexpectedSpace,
    (QuerySpan) {.start = before.end, .end = after.start}
  );
}

static void query_parser_push_frame(QueryParser *self, uint32_t node) {
  array_push(&self->frames, ((QueryParserFrame) {.node = node, .has_pattern = false}));
}

// Returns the index of the token that opens `node`, a named node, MISSING node,
// group, or alternation.
static uint32_t query_parser_opener(const QueryParser *self, uint32_t node) {
  const uint32_t token = query_ast_token(self->ast, node);
  switch (query_ast_kind(self->ast, node)) {
    // The token of a named node or a MISSING node directly follows its `(`.
    case QueryAstKindNamedNode:
    case QueryAstKindMissingNode:
      return token - 1;
    default:
      return token;
  }
}

// Starts the named node whose name, an identifier or `_`, is the next token, and
// parses its subtype, if it has one.
static bool query_parser_begin_named_node(QueryParser *self) {
  const uint32_t name = self->token;
  const uint32_t node = query_ast_open(self->ast, QueryAstKindNamedNode, name);
  self->token = name + 1;
  if (query_parser_peek(self, 0) == QueryTokenKindSlash) {
    const uint32_t subtype = self->token + 1;
    const QueryTokenKind subtype_kind = query_parser_peek(self, 1);
    if (!query_parser_expect_adjacent(self, self->token)) return false;
    if (!query_parser_is_word(subtype_kind) && subtype_kind != QueryTokenKindString) {
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedSubtype, subtype);
    }
    if (!query_parser_expect_adjacent(self, subtype)) return false;
    query_ast_push_leaf(self->ast, QueryAstKindSubtype, subtype);
    self->token = subtype + 1;
  }
  query_parser_push_frame(self, node);
  return true;
}

// Parses the MISSING node whose `MISSING` is the next token. It has no children,
// except for the node type that can follow `MISSING`.
static bool query_parser_parse_missing_node(QueryParser *self) {
  const uint32_t node = query_ast_open(self->ast, QueryAstKindMissingNode, self->token);
  self->token++;
  QueryTokenKind kind = query_parser_peek(self, 0);
  if (query_parser_is_word(kind) || kind == QueryTokenKindString) {
    query_ast_push_leaf(self->ast, QueryAstKindMissingType, self->token);
    self->token++;
    kind = query_parser_peek(self, 0);
  }
  if (kind == QueryTokenKindEnd) {
    return query_parser_fail_at(
      self,
      QueryDiagnosticKindUnclosedDelimiter,
      query_parser_opener(self, node)
    );
  }
  if (kind != QueryTokenKindRightParen) {
    return query_parser_fail_at(self, QueryDiagnosticKindExpectedRightParen, self->token);
  }
  self->token++;
  query_ast_close(self->ast, node);
  return true;
}

// Starts the atom whose `(` is the next token: a group, a named node, or a
// MISSING node, which is parsed completely.
static bool query_parser_begin_parenthesized(QueryParser *self) {
  const uint32_t paren = self->token;
  switch (query_parser_peek(self, 1)) {
    case QueryTokenKindLeftParen:
    case QueryTokenKindLeftBracket:
    case QueryTokenKindString:
      query_parser_push_frame(self, query_ast_open(self->ast, QueryAstKindGroup, paren));
      self->token++;
      return true;
    case QueryTokenKindIdentifier:
    case QueryTokenKindUnderscore:
      self->token++;
      return query_parser_begin_named_node(self);
    case QueryTokenKindMissing:
      self->token++;
      return query_parser_parse_missing_node(self);
    case QueryTokenKindPredicateName:
    case QueryTokenKindDot:
      return query_parser_fail_at(self, QueryDiagnosticKindMisplacedPredicate, paren + 1);
    case QueryTokenKindEnd:
      return query_parser_fail_at(self, QueryDiagnosticKindUnclosedDelimiter, paren);
    default:
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedNodeName, paren + 1);
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
    self->token += 2;
    if (query_parser_at_field(self)) {
      return query_parser_fail_at(self, QueryDiagnosticKindMultipleFields, self->token);
    }
  }

  switch (query_parser_peek(self, 0)) {
    case QueryTokenKindLeftParen:
      return query_parser_begin_parenthesized(self);
    case QueryTokenKindLeftBracket:
      query_parser_push_frame(self, query_ast_open(self->ast, QueryAstKindAlternation, self->token));
      self->token++;
      return true;
    case QueryTokenKindString:
      query_ast_push_leaf(self->ast, QueryAstKindAnonymousNode, self->token);
      self->token++;
      return true;
    case QueryTokenKindUnderscore:
      query_ast_push_leaf(self->ast, QueryAstKindWildcard, self->token);
      self->token++;
      return true;
    default:
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedPattern, self->token);
  }
}

// Finishes the innermost open pattern, whose atom has ended, by parsing its
// quantifiers and captures.
static void query_parser_finish_pattern(QueryParser *self) {
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
    self->token++;
  }
  query_ast_close(self->ast, array_back(&self->frames)->node);
  (void)array_pop(&self->frames);
  if (self->frames.size > 0) array_back(&self->frames)->has_pattern = true;
}

// Parses the legacy form of a predicate's name, `.name?` or `.name!`, whose `.` is
// the next token. There is no space inside it.
static bool query_parser_parse_legacy_predicate_name(QueryParser *self) {
  const uint32_t dot = self->token;
  const QueryTokenKind word_kind = query_parser_peek(self, 1);
  const QueryTokenKind suffix_kind = query_parser_peek(self, 2);
  if (!query_parser_is_word(word_kind)) {
    return query_parser_fail_at(self, QueryDiagnosticKindExpectedPredicateName, dot + 1);
  }
  if (!query_parser_expect_adjacent(self, dot + 1)) return false;
  if (suffix_kind != QueryTokenKindQuestion && suffix_kind != QueryTokenKindBang) {
    return query_parser_fail(self, QueryDiagnosticKindMissingPredicateSuffix, (QuerySpan) {
      .start = query_token_list_span(self->tokens, dot).start,
      .end = query_token_list_span(self->tokens, dot + 1).end,
    });
  }
  if (!query_parser_expect_adjacent(self, dot + 2)) return false;
  self->token = dot + 3;
  return true;
}

// Parses the predicate whose `(` is the next token.
static bool query_parser_parse_predicate(QueryParser *self) {
  const uint32_t paren = self->token;
  const uint32_t name = paren + 1;
  self->token = name;
  switch (query_parser_peek(self, 0)) {
    case QueryTokenKindPredicateName:
      self->token++;
      break;
    case QueryTokenKindDot:
      if (!query_parser_parse_legacy_predicate_name(self)) return false;
      break;
    default:
      return query_parser_fail_at(self, QueryDiagnosticKindExpectedPredicateName, name);
  }

  const uint32_t predicate = query_ast_open(self->ast, QueryAstKindPredicate, name);
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
    self->token++;
  }
  self->token++;
  query_ast_close(self->ast, predicate);
  return true;
}

// Returns whether the next token, a `.`, can be an anchor inside the innermost
// open named node, group, or alternation, given the tokens around it.
static bool query_parser_is_valid_anchor(const QueryParser *self) {
  const QueryParserFrame *frame = array_back(&self->frames);
  const QueryTokenKind next = query_parser_peek(self, 1);
  switch (query_ast_kind(self->ast, frame->node)) {
    case QueryAstKindNamedNode:
      // An anchor before the `)` makes the pattern before it the last child.
      if (next == QueryTokenKindRightParen) return frame->has_pattern;
      return next != QueryTokenKindDot && next != QueryTokenKindBang;
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

// Parses the next item inside the innermost open named node, group, or
// alternation, or closes it.
static bool query_parser_continue_list(QueryParser *self) {
  const QueryParserFrame *frame = array_back(&self->frames);
  const uint32_t list = frame->node;
  const QueryAstKind list_kind = query_ast_kind(self->ast, list);
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
      self->token++;
      return true;
    }
    case QueryTokenKindDot:
      if (!query_parser_is_valid_anchor(self)) {
        return query_parser_fail_at(self, QueryDiagnosticKindMisplacedAnchor, self->token);
      }
      query_ast_push_leaf(self->ast, QueryAstKindAnchor, self->token);
      self->token++;
      return true;
    case QueryTokenKindBang:
      if (list_kind != QueryAstKindNamedNode) break;
      if (!query_parser_is_word(query_parser_peek(self, 1))) {
        return query_parser_fail_at(self, QueryDiagnosticKindExpectedFieldName, self->token + 1);
      }
      query_ast_push_leaf(self->ast, QueryAstKindNegatedField, self->token + 1);
      self->token += 2;
      return true;
    case QueryTokenKindLeftParen: {
      const QueryTokenKind next = query_parser_peek(self, 1);
      if (
        list_kind != QueryAstKindAlternation &&
        (next == QueryTokenKindPredicateName || next == QueryTokenKindDot)
      ) {
        return query_parser_parse_predicate(self);
      }
      break;
    }
    default:
      break;
  }
  return query_parser_begin_pattern(self);
}

// Starts the next pattern at the top level of the query.
static bool query_parser_begin_top_level_pattern(QueryParser *self) {
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
    .token = 0,
    .frames = array_new(),
  };
  // Real queries average fewer nodes than tokens, so this is enough to avoid
  // growing the tree for most of them.
  query_ast_reserve(ast, query_ast_size(ast) + query_token_list_size(tokens));
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
  if (!succeeded) {
    // Close the nodes that were open when parsing stopped, so that the nodes
    // parsed so far still form a tree.
    for (uint32_t node = 0; node < query_ast_size(ast); node++) {
      if (query_ast_end(ast, node) == QUERY_AST_UNCLOSED) query_ast_close(ast, node);
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
  Array(uint32_t) ends = array_new();
  for (uint32_t node = 0; node < query_ast_size(self); node++) {
    while (ends.size > 0 && *array_back(&ends) <= node) (void)array_pop(&ends);
    const QuerySpan span = query_token_list_span(tokens, query_ast_token(self, node));
    printf(
      "%3u: %*s%s %.*s\n",
      node,
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
