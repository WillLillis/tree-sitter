#ifndef TREE_SITTER_QUERY_VM_H_
#define TREE_SITTER_QUERY_VM_H_

#include "./array.h"
#include "./query_capture_store.h"
#include "./query_program.h"
#include "./query_schedule.h"
#include "./query_thread_store.h"
#include "tree_sitter/api.h"

// No match id.
#define QUERY_VM_NONE UINT32_MAX

// What a finished match holds its captures with, and whether `next_capture`
// has returned all of them.
typedef enum {
  // The match's chain, which is copied into the shared short list whenever
  // the match is shown.
  QueryCaptureViewChain,
  // A long list of the match's own, which its chain was copied into once.
  QueryCaptureViewList,
  // The same, once `next_capture` has returned all of the match's captures.
  QueryCaptureViewChainExhausted,
  QueryCaptureViewListExhausted,
} QueryCaptureViewState;

// A match that has finished, and has not been returned yet.
typedef struct {
  // The match's id. Ids increase in the order that matches finish, which
  // `next_match` and the order of `next_capture`'s ties rely on.
  uint32_t id;
  uint16_t pattern;

  // A QueryCaptureViewState.
  uint8_t view : 2;
  // The match's chain, or the index of its long list, as `view` says.
  union {
    QueryCaptureChain chain;
    uint32_t list;
  } captures;

  // How many of the match's captures `next_capture` has returned, and where
  // the next one starts.
  uint32_t consumed_capture_count;
  uint32_t next_start_byte;
} QueryFinishedMatch;

// The finished matches of a cursor, and the lists that show their captures.
typedef struct {
  // In the order that they finished, or, once `next_capture` is used, a heap
  // of the first `heap_size` of them by their next captures.
  Array(QueryFinishedMatch) finished;
  uint32_t heap_size;

  // The id of the next match to finish.
  uint32_t next_id;

  // The captures of the match with a short chain that was shown last, and
  // that match's id, or NONE.
  QueryCaptureList short_list;
  uint32_t short_list_id;

  // The lists of the matches with long chains. Empty lists are free.
  Array(QueryCaptureList) long_lists;
} QueryMatchStore;

// What the threads that run on the current node may need to know about it.
typedef struct {
  TSNode node;
  TSSymbol symbol;

  // Whether the fields below have been computed. They are lazily computed when
  // first needed.
  bool has_status;

  // The field that the node is in, whether it has later siblings, and whether
  // a later sibling can be in the same field.
  TSFieldId field;
  bool has_later_siblings;
  bool can_have_later_siblings_with_this_field;
} QueryVmNode;

// A query's program, containing what the matching engine determines when the
// query is created, and which of its patterns and captures are disabled.
typedef struct {
  QueryProgram program;
  // The patterns and captures that the API has disabled.
  Array(bool) disabled_patterns;
  Array(bool) disabled_captures;
} QueryVmProgram;

// What the walk does next at the cursor's node.
typedef enum {
  // Enter the node, if it is visible, and descend into it if the walk needs its
  // children.
  QueryVmWalkPhaseEntering,
  // Leave the node, if it is visible, and move to its next sibling or back up.
  QueryVmWalkPhaseLeaving,
  // The walk is over.
  QueryVmWalkPhaseHalted,
} QueryVmWalkPhase;

// The state of the matching engine for one query cursor.
typedef struct {
  // The query that the cursor runs.
  const QueryVmProgram *query;
  QueryThreadStore threads;
  QueryCaptureStore captures;
  QuerySchedule schedule;
  QueryMatchStore matches;

  // The unfinished thread whose first capture is earliest, or NONE, kept
  // between calls to `next_capture` until the threads change, and whether it
  // is known.
  QueryThreadId first_unfinished;
  bool has_first_unfinished;

  // The number of threads and finished matches that hold captures, which the
  // match limit bounds.
  uint32_t capture_holder_count;

  // The node that the walk is entering.
  QueryVmNode node;

  // The walk: its cursor, the depth of the cursor's node counted from the node
  // where the walk started, whether that node is visible, and what the walk
  // does next with it. A hidden node has the depth of its nearest visible
  // ancestor.
  TSTreeCursor cursor;
  uint32_t depth;
  bool on_visible_node;
  QueryVmWalkPhase phase;

  // The part of the tree that each match overlaps, by both bytes and points.
  TSRange included_range;
  // The part of the tree that each match is inside, by both bytes and points.
  TSRange containing_range;
  // The deepest that a match's first node can be, counted from the node where
  // the walk started.
  uint32_t max_start_depth;

  // The most threads and finished matches with captures that the cursor keeps,
  // and whether it has had to abandon one to keep to it.
  uint32_t match_limit;
  bool did_exceed_match_limit;

  // The progress callback, its state, and the work done since it was last
  // called.
  const TSQueryCursorOptions *options;
  TSQueryCursorState progress_state;
  uint32_t work_count;
} QueryVm;

static inline QueryMatchStore query_match_store_new(void) {
  return (QueryMatchStore) {
    .finished = array_new(),
    .short_list = array_new(),
    .short_list_id = QUERY_VM_NONE,
    .long_lists = array_new(),
  };
}

// Empties the store, for a new walk. The long lists stay allocated but empty,
// which marks them free.
static inline void query_match_store_reset(QueryMatchStore *self) {
  array_clear(&self->finished);
  self->heap_size = 0;
  self->next_id = 0;
  array_clear(&self->short_list);
  self->short_list_id = QUERY_VM_NONE;
  for (uint32_t i = 0; i < self->long_lists.size; i++) {
    array_clear(array_get(&self->long_lists, i));
  }
}

static inline void query_match_store_delete(QueryMatchStore *self) {
  array_delete(&self->finished);
  array_delete(&self->short_list);
  for (uint32_t i = 0; i < self->long_lists.size; i++) {
    array_delete(array_get(&self->long_lists, i));
  }
  array_delete(&self->long_lists);
}

// Makes a query's program, which then owns `program`, for a query with
// `capture_count` capture names.
void ts_query_vm_program_new(QueryVmProgram *self, QueryProgram program,
                             uint32_t capture_count);
void ts_query_vm_program_copy(QueryVmProgram *self,
                              const QueryVmProgram *other);
void ts_query_vm_program_delete(QueryVmProgram *self);

// A query cursor's matching engine: `exec` starts a walk of the tree under
// `node`, and the rest are the query cursor API's functions of the same names.
void ts_query_vm_new(QueryVm *self);
void ts_query_vm_delete(QueryVm *self);
void ts_query_vm_exec(QueryVm *self, const QueryVmProgram *query, TSNode node,
                      const TSQueryCursorOptions *options);
bool ts_query_vm_next_match(QueryVm *self, TSQueryMatch *match);
bool ts_query_vm_next_capture(QueryVm *self, TSQueryMatch *match,
                              uint32_t *capture_index);
void ts_query_vm_remove_match(QueryVm *self, uint32_t match_id);

#endif // TREE_SITTER_QUERY_VM_H_
