#ifndef TREE_SITTER_QUERY_VM_H_
#define TREE_SITTER_QUERY_VM_H_

#include "./array.h"
#include "./query_capture_store.h"
#include "./query_program.h"
#include "./query_schedule.h"
#include "./query_thread_store.h"
#include "tree_sitter/api.h"

// No match id, or no match.
#define QUERY_VM_NONE UINT32_MAX

// The most supertypes of one node that the matching engine learns about. The
// tree cursor leaves out any past this many, so a supertype test fails on
// them, as it does on master.
#define QUERY_VM_MAX_SUPERTYPES 8

// What a finished match holds its captures with, and whether `next_capture`
// has returned all of them.
typedef enum {
  // The match's history, which is copied into the shared short list whenever
  // the match is shown.
  QueryCaptureViewHistory,
  // A long list of the match's own, which its history was copied into once.
  QueryCaptureViewList,
  // The same, once `next_capture` has returned all of the match's captures.
  QueryCaptureViewHistoryExhausted,
  QueryCaptureViewListExhausted,
} QueryCaptureViewState;

typedef Array(TSQueryCapture) QueryCaptureList;

// A match that has finished, and has not been returned yet.
typedef struct {
  // The match's id, once the API has shown the match, or NONE.
  uint32_t id;
  uint16_t pattern;
  // A QueryCaptureViewState.
  uint8_t view : 2;
  // The match's history, or the index of its long list, as `view` says.
  union {
    QueryHistory history;
    uint32_t list;
  } captures;
  // The order in which the matches of one run of the cursor finished.
  uint32_t order;
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
  // The order of the next match to finish, and the id of the next match to be
  // shown.
  uint32_t next_order;
  uint32_t next_id;
  // The captures of the match with a short history that was shown last, and
  // that match's order, or NONE.
  QueryCaptureList short_list;
  uint32_t short_list_order;
  // The lists of the matches with long histories, and the free ones.
  Array(QueryCaptureList) long_lists;
  Array(uint32_t) free_long_lists;
} QueryMatchStore;

// What the threads that run on the current node may need to know about it.
// The fields after `has_status` are computed the first time they are needed.
typedef struct {
  TSNode node;
  TSSymbol symbol;
  bool is_named;
  bool is_missing;
  bool is_error;
  bool has_status;
  // The field that the node is in, whether it has later siblings, and whether
  // a later sibling can be in the same field.
  TSFieldId field;
  bool has_later_siblings;
  bool has_later_named_siblings;
  bool can_have_later_siblings_with_this_field;
  // The supertypes that the node is in.
  TSSymbol supertypes[QUERY_VM_MAX_SUPERTYPES];
  unsigned supertype_count;
} QueryVmNode;

// A query's program, with what the matching engine works out from it when the
// query is created, and which of its patterns and captures are disabled.
typedef struct {
  QueryProgram program;
  // The entries whose first test is a symbol, by symbol: those of the symbol
  // `s` are at `symbol_offsets[s]` up to `symbol_offsets[s + 1]`, in pattern
  // order.
  Array(QueryEntry) symbol_entries;
  Array(uint32_t) symbol_offsets;
  // The entries that an ERROR node is given, and the entries whose first test
  // is not a symbol, which every other node is given. Those all fail on an
  // ERROR node.
  Array(QueryEntry) error_entries;
  Array(QueryEntry) wildcard_entries;
  // The patterns and captures that the API has disabled.
  Array(bool) disabled_patterns;
  Array(bool) disabled_captures;
  // Whether the program tests whether a node is the last one, which needs the
  // path to the current node.
  bool needs_path;
} QueryVmProgram;

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
  // For each depth of the walk, the index in the tree cursor's stack of the
  // node at that depth on the path to the current node, for a query that tests
  // whether a node is the last one.
  Array(uint32_t) path;
  // The number of threads and finished matches that hold captures, which the
  // match limit bounds.
  uint32_t holder_count;
  // The node that the walk is on.
  QueryVmNode node;
  // The walk: its cursor, the depth of the cursor's node, counted from the
  // node where the walk started, whether that node is visible, whether the
  // walk is going back up past it, and whether the walk has stopped.
  TSTreeCursor cursor;
  uint32_t depth;
  bool on_visible_node;
  bool ascending;
  bool halted;
  // The parts of the tree that matches are in, and the deepest that a match
  // can start.
  TSRange included_range;
  TSRange containing_range;
  uint32_t max_start_depth;
  // The most threads and finished matches with captures that the cursor keeps,
  // and whether it has had to abandon one to keep to it.
  uint32_t match_limit;
  bool did_exceed_match_limit;
  // The progress callback, its state, and the work done since it was last
  // called.
  const TSQueryCursorOptions *options;
  TSQueryCursorState progress_state;
  unsigned work_count;
} QueryVm;

// Makes a query's program, which then owns `program`, for a query in
// `language` with `capture_count` capture names.
void ts_query_vm_program_new(
  QueryVmProgram *self,
  QueryProgram program,
  const TSLanguage *language,
  uint32_t capture_count
);
void ts_query_vm_program_copy(QueryVmProgram *self, const QueryVmProgram *other);
void ts_query_vm_program_delete(QueryVmProgram *self);

// A query cursor's matching engine: `exec` starts a walk of the tree under
// `node`, and the rest are the query cursor API's functions of the same names.
void ts_query_vm_new(QueryVm *self);
void ts_query_vm_delete(QueryVm *self);
void ts_query_vm_exec(
  QueryVm *self,
  const QueryVmProgram *query,
  TSNode node,
  const TSQueryCursorOptions *options
);
bool ts_query_vm_next_match(QueryVm *self, TSQueryMatch *match);
bool ts_query_vm_next_capture(QueryVm *self, TSQueryMatch *match, uint32_t *capture_index);
void ts_query_vm_remove_match(QueryVm *self, uint32_t match_id);

#endif  // TREE_SITTER_QUERY_VM_H_
