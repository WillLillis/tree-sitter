#ifndef TREE_SITTER_QUERY_PARSER_H_
#define TREE_SITTER_QUERY_PARSER_H_

#include "./array.h"
#include "./query_diagnostic.h"
#include "./query_tokenizer.h"

#include <stdbool.h>
#include <stdint.h>

// #define DEBUG_QUERY_AST

// A kind of node in the syntax tree of a query.
typedef enum {
  // A pattern: an optional field, an atom, and then any quantifiers and
  // captures, which are its children in that order. Its token is the first
  // token of the pattern.
  QueryAstKindPattern,
  // `name:`, before the atom of a pattern. Its token is the name.
  QueryAstKindField,
  // `(name ...)` or `(_ ...)`. Its token is the name.
  QueryAstKindNamedNode,
  // The name or string after the `/` in `(supertype/subtype ...)`.
  QueryAstKindSubtype,
  // `(MISSING ...)`. Its token is `MISSING`.
  QueryAstKindMissingNode,
  // The name or string after `MISSING`.
  QueryAstKindMissingType,
  // `"string"`.
  QueryAstKindAnonymousNode,
  // `_` on its own, rather than as the name in `(_ ...)`.
  QueryAstKindWildcard,
  // `[...]`, around patterns, any one of which can match. Its token is the `[`.
  QueryAstKindAlternation,
  // `(...)`, around patterns that match a sequence of siblings. Its token is
  // the `(`.
  QueryAstKindGroup,
  // `.`, which anchors the patterns on either side of it.
  QueryAstKindAnchor,
  // `!name`. Its token is the name.
  QueryAstKindNegatedField,
  // `?`, `*`, or `+`, after the atom of a pattern.
  QueryAstKindQuantifier,
  // `@name`, after the atom of a pattern.
  QueryAstKindCapture,
  // `(#name? ...)`, or the legacy `(.name? ...)`. Its token is the `#name?`, or
  // the `.`.
  QueryAstKindPredicate,
  // A capture, a string, or a bare word, inside a predicate.
  QueryAstKindPredicateArgument,
} QueryAstKind;

// The number of bits that a QueryAstNode uses to store the index of its token.
#define QUERY_AST_TOKEN_BITS 27

_Static_assert(
  QUERY_TOKEN_LIST_MAX_SIZE <= (1u << QUERY_AST_TOKEN_BITS),
  "QueryAstNode.token must fit the index of any token"
);

_Static_assert(
  QueryAstKindPredicateArgument < (1 << (32 - QUERY_AST_TOKEN_BITS)),
  "QueryAstKind must fit in QueryAstNode.kind"
);

// A node in the syntax tree of a query. Nodes are stored in preorder: each node
// comes before its children, and children come in the order that they appear in
// the source. So each node's subtree, the node and all of its descendants, is a
// contiguous range of nodes.
typedef struct {
  // A QueryAstKind.
  uint32_t kind : 32 - QUERY_AST_TOKEN_BITS;
  // The index of the node's token in the QueryTokenList.
  uint32_t token : QUERY_AST_TOKEN_BITS;
  // The index of the first node after this node's subtree.
  uint32_t end;
} QueryAstNode;

_Static_assert(sizeof(QueryAstNode) == 8, "QueryAstNode packs its kind and token into one word");

// The syntax tree of a query: the subtree of each pattern, one after another.
typedef struct {
  Array(QueryAstNode) nodes;
} QueryAst;

static inline QueryAst query_ast_new(void) {
  return (QueryAst) {.nodes = array_new()};
}

static inline void query_ast_delete(QueryAst *self) {
  array_delete(&self->nodes);
}

static inline void query_ast_reserve(QueryAst *self, uint32_t new_capacity) {
  array_reserve(&self->nodes, new_capacity);
}

static inline uint32_t query_ast_size(const QueryAst *self) {
  return self->nodes.size;
}

static inline QueryAstKind query_ast_kind(const QueryAst *self, uint32_t node) {
  return (QueryAstKind)array_get(&self->nodes, node)->kind;
}

static inline uint32_t query_ast_token(const QueryAst *self, uint32_t node) {
  return array_get(&self->nodes, node)->token;
}

static inline uint32_t query_ast_end(const QueryAst *self, uint32_t node) {
  return array_get(&self->nodes, node)->end;
}

// The end of a node that has been opened and not yet closed.
#define QUERY_AST_UNCLOSED UINT32_MAX

// Appends a node whose descendants come next, and returns its index.
static inline uint32_t query_ast_open(QueryAst *self, QueryAstKind kind, uint32_t token) {
  uint32_t node = self->nodes.size;
  array_push(&self->nodes, ((QueryAstNode) {.kind = kind, .token = token, .end = QUERY_AST_UNCLOSED}));
  return node;
}

// Ends the subtree of `node` after the last node so far.
static inline void query_ast_close(QueryAst *self, uint32_t node) {
  array_get(&self->nodes, node)->end = self->nodes.size;
}

// Appends a node with no children.
static inline void query_ast_push_leaf(QueryAst *self, QueryAstKind kind, uint32_t token) {
  // `array_push` increments `size` unsequenced with evaluating the new node, so
  // the end has to be computed first.
  uint32_t end = self->nodes.size + 1;
  array_push(&self->nodes, ((QueryAstNode) {.kind = kind, .token = token, .end = end}));
}

// Parses `tokens` into a syntax tree, and appends its nodes to `ast`. Returns
// false at the first error in the source, after describing it in `error`.
bool ts_query_parse(const QueryTokenList *tokens, QueryAst *ast, QueryDiagnostic *error);

#ifdef DEBUG_QUERY_AST
void ts_query_ast_dump(const QueryAst *self, const QueryTokenList *tokens, const char *source);
#endif

#endif  // TREE_SITTER_QUERY_PARSER_H_
