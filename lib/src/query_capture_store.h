#ifndef TREE_SITTER_QUERY_CAPTURE_STORE_H_
#define TREE_SITTER_QUERY_CAPTURE_STORE_H_

#include "./array.h"
#include "tree_sitter/api.h"

// The captures of a match, by the capture store's cell with the last one,
// which is linked to the cells with the ones before it.
typedef struct {
  uint32_t cell;
} QueryHistory;

// The history of a match without captures.
#define QUERY_HISTORY_EMPTY ((QueryHistory) {UINT32_MAX})

static inline bool query_history_is_empty(QueryHistory self) {
  return self.cell == UINT32_MAX;
}

static inline bool query_history_eq(QueryHistory a, QueryHistory b) {
  return a.cell == b.cell;
}

// One capture of a match, linked to the capture before it. Threads that are
// copies of each other share the captures that they had when they were copied.
typedef struct {
  // The captured node, without its tree.
  const void *node_id;
  uint32_t start_byte;
  TSPoint start_point;
  // The capture before this one.
  QueryHistory previous;
  // The number of threads, finished matches, and later cells that refer to
  // this cell.
  uint32_t reference_count;
  uint16_t capture;
  TSSymbol alias;
} QueryCaptureCell;

// The capture cells of a cursor. Each history is owned by the thread or the
// finished match that ends with it, and each cell owns the cell before it, so
// copies of a thread share their captures instead of copying them.
typedef struct {
  Array(QueryCaptureCell) cells;
  // The first free cell, which is linked to the others by `previous`, or an
  // empty history if there is none.
  QueryHistory free_head;
  // The tree that the captured nodes are in.
  const TSTree *tree;
} QueryCaptureStore;

static inline QueryCaptureStore query_capture_store_new(void) {
  return (QueryCaptureStore) {.cells = array_new(), .free_head = QUERY_HISTORY_EMPTY};
}

static inline void query_capture_store_delete(QueryCaptureStore *self) {
  array_delete(&self->cells);
  self->free_head = QUERY_HISTORY_EMPTY;
}

// Frees every cell at once, for a new walk of `tree`. The threads and finished
// matches that owned them must be gone as well.
static inline void query_capture_store_reset(QueryCaptureStore *self, const TSTree *tree) {
  array_clear(&self->cells);
  self->free_head = QUERY_HISTORY_EMPTY;
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

// Returns a history that ends with `node` as `capture`, after `history`. The
// new cell takes over the caller's reference to `history`.
static inline QueryHistory query_capture_store_append(
  QueryCaptureStore *self,
  QueryHistory history,
  TSNode node,
  uint16_t capture
) {
  uint32_t index;
  if (!query_history_is_empty(self->free_head)) {
    index = self->free_head.cell;
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
    .previous = history,
    .reference_count = 1,
    .capture = capture,
    .alias = (TSSymbol)node.context[3],
  };
  return (QueryHistory) {index};
}

// Adds a reference to `history`, for a copy of the thread that owns it.
static inline void query_capture_store_retain(QueryCaptureStore *self, QueryHistory history) {
  if (!query_history_is_empty(history)) array_get(&self->cells, history.cell)->reference_count++;
}

// Drops a reference to `history`, and frees each of its cells that nothing
// refers to anymore, from the last one back to the first that something
// still does.
static inline void query_capture_store_release(QueryCaptureStore *self, QueryHistory history) {
  while (!query_history_is_empty(history)) {
    QueryCaptureCell *cell = array_get(&self->cells, history.cell);
    ts_assert(cell->reference_count > 0);
    if (--cell->reference_count > 0) return;
    const QueryHistory previous = cell->previous;
    cell->previous = self->free_head;
    self->free_head = history;
    history = previous;
  }
}

// Returns the number of captures in `history`.
static inline uint32_t query_capture_store_length(const QueryCaptureStore *self, QueryHistory history) {
  uint32_t length = 0;
  while (!query_history_is_empty(history)) {
    length++;
    history = array_get(&self->cells, history.cell)->previous;
  }
  return length;
}

// Copies the `length` captures of `history` into `output`, first capture first.
static inline void query_capture_store_copy(
  const QueryCaptureStore *self,
  QueryHistory history,
  TSQueryCapture *output,
  uint32_t length
) {
  uint32_t remaining = length;
  while (!query_history_is_empty(history)) {
    const QueryCaptureCell *cell = array_get(&self->cells, history.cell);
    remaining--;
    output[remaining] = (TSQueryCapture) {.node = query_capture_store_node(self, cell), .index = cell->capture};
    history = cell->previous;
  }
  ts_assert(remaining == 0);
}

// Returns -1 if the node of `a` comes before the node of `b` in a depth-first
// walk, 1 if it comes after, and 0 if they are the same node, or if nothing
// tells them apart. Of two nodes that start at the same byte, the longer one is
// the ancestor, and comes first, and of two that also end at the same byte,
// the ancestor is the one with more descendants. Two empty nodes at the same
// place may be ordered wrongly, which only makes a subset test miss a subset.
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

// Sets whether each of two histories has all of the captures of the other.
// Both are in document order, so they are walked back from their last captures
// together, until they reach a cell that they share.
static inline void query_capture_store_compare(
  const QueryCaptureStore *self,
  QueryHistory a,
  QueryHistory b,
  bool *a_in_b,
  bool *b_in_a
) {
  *a_in_b = true;
  *b_in_a = true;
  while (!query_history_eq(a, b)) {
    if (query_history_is_empty(a)) {
      *b_in_a = false;
      return;
    }
    if (query_history_is_empty(b)) {
      *a_in_b = false;
      return;
    }
    const QueryCaptureCell *a_cell = array_get(&self->cells, a.cell);
    const QueryCaptureCell *b_cell = array_get(&self->cells, b.cell);
    if (a_cell->node_id == b_cell->node_id && a_cell->capture == b_cell->capture) {
      a = a_cell->previous;
      b = b_cell->previous;
      continue;
    }
    // The capture of the later node cannot be in the other history, whose
    // remaining captures all come before it.
    switch (query_capture_store_compare_nodes(self, a_cell, b_cell)) {
      case 1:
        *a_in_b = false;
        a = a_cell->previous;
        break;
      case -1:
        *b_in_a = false;
        b = b_cell->previous;
        break;
      default:
        *a_in_b = false;
        *b_in_a = false;
        a = a_cell->previous;
        b = b_cell->previous;
        break;
    }
    if (!*a_in_b && !*b_in_a) return;
  }
}

#endif  // TREE_SITTER_QUERY_CAPTURE_STORE_H_
