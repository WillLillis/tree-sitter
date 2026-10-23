#ifndef TREE_SITTER_QUERY_CAPTURE_STORE_H_
#define TREE_SITTER_QUERY_CAPTURE_STORE_H_

#include "./array.h"
#include "tree_sitter/api.h"

typedef Array(TSQueryCapture) QueryCaptureList;

// The captures of a match, by the capture store's cell with the last one,
// which is linked to the cells with the ones before it.
typedef struct {
  uint32_t index;
} QueryCaptureChain;

// The empty capture chain.
#define QUERY_CAPTURE_CHAIN_EMPTY ((QueryCaptureChain) {UINT32_MAX})

static inline bool query_capture_chain_is_empty(QueryCaptureChain self) {
  return self.index == UINT32_MAX;
}

// One capture of a match, linked to the capture before it. Threads that are
// copies of each other share the captures that they had when they were copied.
typedef struct {
  // The captured node, without its tree, as a `const Subtree *`.
  const void *node_id;
  uint32_t start_byte;
  TSPoint start_point;
  // The capture before this one.
  QueryCaptureChain previous;
  // The number of threads, finished matches, and later cells that refer to
  // this cell.
  uint32_t reference_count;
  uint16_t capture;
  TSSymbol alias;
} QueryCaptureCell;

// The capture cells of a cursor. Each chain is owned by the thread or the
// finished match that ends with it, and each cell owns the cell before it, so
// copies of a thread share their captures instead of copying them.
typedef struct {
  Array(QueryCaptureCell) cells;
  // The first free cell, or an empty chain if there is none.
  QueryCaptureChain free_head;
  // The tree that the captured nodes are in.
  const TSTree *tree;
} QueryCaptureStore;

static inline QueryCaptureStore query_capture_store_new(void) {
  return (QueryCaptureStore) {.cells = array_new(), .free_head = QUERY_CAPTURE_CHAIN_EMPTY};
}

static inline void query_capture_store_delete(QueryCaptureStore *self) {
  array_delete(&self->cells);
  self->free_head = QUERY_CAPTURE_CHAIN_EMPTY;
}

// Frees every cell at once, for a new walk of `tree`. The threads and finished
// matches that owned them must be gone as well.
static inline void query_capture_store_reset(QueryCaptureStore *self, const TSTree *tree) {
  array_clear(&self->cells);
  self->free_head = QUERY_CAPTURE_CHAIN_EMPTY;
  self->tree = tree;
}

// Returns the node that `cell` captured.
static inline TSNode query_capture_store_node(const QueryCaptureStore *self, const QueryCaptureCell *cell) {
  return (TSNode) {
    .context = {cell->start_byte, cell->start_point.row, cell->start_point.column, cell->alias},
    .id = cell->node_id,
    .tree = self->tree,
  };
}

// Returns a chain that ends with `node` as `capture`, after `chain`. The
// new cell takes over the caller's reference to `chain`.
static inline QueryCaptureChain query_capture_store_append(
  QueryCaptureStore *self,
  QueryCaptureChain chain,
  TSNode node,
  uint16_t capture
) {
  uint32_t index;
  if (!query_capture_chain_is_empty(self->free_head)) {
    index = self->free_head.index;
    self->free_head = array_get(&self->cells, index)->previous;
  } else {
    index = self->cells.size;
    array_push(&self->cells, ((QueryCaptureCell) {0}));
  }
  // INVARIANT: An alias is a symbol.
  ts_assert(node.context[3] <= UINT16_MAX);
  *array_get(&self->cells, index) = (QueryCaptureCell) {
    .node_id = node.id,
    .start_byte = node.context[0],
    .start_point = {.row = node.context[1], .column = node.context[2]},
    .previous = chain,
    .reference_count = 1,
    .capture = capture,
    .alias = (TSSymbol)node.context[3],
  };
  return (QueryCaptureChain) {index};
}

// Adds a reference to `chain`, for a copy of the thread that owns it.
static inline void query_capture_store_retain(QueryCaptureStore *self, QueryCaptureChain chain) {
  if (!query_capture_chain_is_empty(chain)) array_get(&self->cells, chain.index)->reference_count++;
}

// Drops a reference to `chain`, and frees each of its cells that nothing
// refers to anymore, from the last one back to the first that something
// still does.
static inline void query_capture_store_release(QueryCaptureStore *self, QueryCaptureChain chain) {
  while (!query_capture_chain_is_empty(chain)) {
    QueryCaptureCell *cell = array_get(&self->cells, chain.index);
    ts_assert(cell->reference_count > 0);
    if (--cell->reference_count > 0) return;
    const QueryCaptureChain previous = cell->previous;
    cell->previous = self->free_head;
    self->free_head = chain;
    chain = previous;
  }
}

// Returns the number of captures in `chain`.
static inline uint32_t query_capture_store_length(const QueryCaptureStore *self, QueryCaptureChain chain) {
  uint32_t length = 0;
  while (!query_capture_chain_is_empty(chain)) {
    length++;
    chain = array_get(&self->cells, chain.index)->previous;
  }
  return length;
}

// Resizes `output` to `length` and copies the captures of `chain` into it,
// first capture first.
static inline void query_capture_store_copy(
  const QueryCaptureStore *self,
  QueryCaptureChain chain,
  QueryCaptureList *output,
  uint32_t length
) {
  array_reserve(output, length);
  // Leave new slots unset rather than zeroed with `array_grow_by`.
  output->size = length;
  uint32_t remaining = length;
  while (!query_capture_chain_is_empty(chain)) {
    const QueryCaptureCell *cell = array_get(&self->cells, chain.index);
    ts_assert(remaining > 0);
    remaining--;
    *array_get(output, remaining) = (TSQueryCapture) {
      .node = query_capture_store_node(self, cell),
      .index = cell->capture,
    };
    chain = cell->previous;
  }
  ts_assert(remaining == 0);
}

// Compares captured nodes in depth-first traversal order. Returns -1 if `a`
// comes first, 1 if `b` comes first, and 0 for the same node or an unresolved
// tie.
//
// Rules, applied in order:
// - Identical node IDs compare equal.
// - Earlier start byte comes first.
// - For equal starts, later end byte comes first.
// - For equal spans, larger descendant count comes first.
//
// The last two rules put ancestors before descendants. With zero-width
// nodes, equal starts do not imply nesting, so the order can be ambiguous
// or wrong. This can make chain comparison miss a subset.
static inline int query_capture_store_compare_nodes(
  const QueryCaptureStore *self,
  const QueryCaptureCell *a,
  const QueryCaptureCell *b
) {
  if (a->node_id == b->node_id) return 0;
  if (a->start_byte != b->start_byte) return a->start_byte < b->start_byte ? -1 : 1;
  const TSNode a_node = query_capture_store_node(self, a);
  const TSNode b_node = query_capture_store_node(self, b);
  const uint32_t a_end = ts_node_end_byte(a_node);
  const uint32_t b_end = ts_node_end_byte(b_node);
  if (a_end != b_end) return a_end > b_end ? -1 : 1;
  const uint32_t a_count = ts_node_descendant_count(a_node);
  const uint32_t b_count = ts_node_descendant_count(b_node);
  if (a_count != b_count) return a_count > b_count ? -1 : 1;
  return 0;
}

// How the first capture chain relates to the second. Subset and superset exclude
// equality.
typedef enum {
  // No containment relation was established.
  QueryCaptureRelationNone,
  QueryCaptureRelationEqual,
  QueryCaptureRelationSubset,
  QueryCaptureRelationSuperset,
} QueryCaptureRelation;

// Returns the containment relation of `a` to `b`. Both are in document order,
// so they are walked back from their last captures together, until they reach
// a cell that they share.
static inline QueryCaptureRelation query_capture_store_compare(
  const QueryCaptureStore *self,
  QueryCaptureChain a,
  QueryCaptureChain b
) {
  bool a_in_b = true;
  bool b_in_a = true;
  while (a.index != b.index) {
    if (query_capture_chain_is_empty(a)) {
      b_in_a = false;
      break;
    }
    if (query_capture_chain_is_empty(b)) {
      a_in_b = false;
      break;
    }
    const QueryCaptureCell *a_cell = array_get(&self->cells, a.index);
    const QueryCaptureCell *b_cell = array_get(&self->cells, b.index);
    if (a_cell->node_id == b_cell->node_id && a_cell->capture == b_cell->capture) {
      a = a_cell->previous;
      b = b_cell->previous;
      continue;
    }
    // The capture of the later node cannot be in the other chain, whose
    // remaining captures all come before it.
    switch (query_capture_store_compare_nodes(self, a_cell, b_cell)) {
      case 1:
        a_in_b = false;
        a = a_cell->previous;
        break;
      case -1:
        b_in_a = false;
        b = b_cell->previous;
        break;
      case 0:
        return QueryCaptureRelationNone;
      default:
        ts_unreachable();
    }
    if (!a_in_b && !b_in_a) return QueryCaptureRelationNone;
  }
  if (a_in_b && b_in_a) return QueryCaptureRelationEqual;
  if (a_in_b) return QueryCaptureRelationSubset;
  if (b_in_a) return QueryCaptureRelationSuperset;
  return QueryCaptureRelationNone;
}

#endif  // TREE_SITTER_QUERY_CAPTURE_STORE_H_
