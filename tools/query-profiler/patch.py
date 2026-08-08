#!/usr/bin/env python3
"""Generate tools/query-profiler/query_instr.c: lib/src/query.c + measurement counters.

Run from the repo root. Every anchor asserts exactly one match, so this fails
loudly if query.c drifts rather than silently mis-patching."""
import sys, re

SRC = "lib/src/query.c"
p = "tools/query-profiler/query_instr.c"
src = open(SRC).read()

def sub1(old, new, why):
    global src
    assert src.count(old) == 1, f"{why}: found {src.count(old)} matches"
    src = src.replace(old, new)

# 1. Counter struct, right after the includes.
sub1(
    '#include <wctype.h>\n',
    '''#include <wctype.h>

// ---- instrumentation ----
typedef struct {
  unsigned long nodes_entered;      // visible nodes entered by the cursor
  unsigned long nodes_descended;    // times we descended into a child
  unsigned long states_started;     // ts_query_cursor__add_state
  unsigned long states_copied;      // ts_query_cursor__copy_state (fan-out)
  unsigned long state_visits;       // (node, in-progress state) pairs examined
  unsigned long state_visits_depth_skip; // ... of those, rejected by the depth test alone
  unsigned long dedup_compares;     // ts_query_cursor__compare_captures calls
  unsigned long dedup_capture_steps;// inner-loop iterations of the above
  unsigned long pool_acquires;      // capture_list_pool_acquire calls
  unsigned long pool_scan_steps;    // free-slot linear scan iterations
  unsigned long sort_compares;      // ts_query_cursor__state_precedes calls
  unsigned long descend_scan_steps; // should_descend state-scan iterations
  unsigned long max_states;         // peak concurrent in-progress states
  unsigned long max_finished;       // peak finished states
  unsigned long capture_drops;      // captures silently dropped: step already had 3
  unsigned long matches;            // finished states pushed
} QProbe;
QProbe qprobe = {0};
#define QP(f) (qprobe.f++)
#define QPADD(f, n) (qprobe.f += (unsigned long)(n))
#define QPMAX(f, n) do { if ((unsigned long)(n) > qprobe.f) qprobe.f = (unsigned long)(n); } while (0)
''',
    "counter struct",
)

# 2. Capture drop detection (MAX_STEP_CAPTURE_COUNT overflow).
sub1(
    '''static void query_step__add_capture(QueryStep *self, uint16_t capture_id) {
  for (unsigned i = 0; i < MAX_STEP_CAPTURE_COUNT; i++) {
    if (self->capture_ids[i] == NONE) {
      self->capture_ids[i] = capture_id;
      break;
    }
  }
}''',
    '''static void query_step__add_capture(QueryStep *self, uint16_t capture_id) {
  for (unsigned i = 0; i < MAX_STEP_CAPTURE_COUNT; i++) {
    if (self->capture_ids[i] == NONE) {
      self->capture_ids[i] = capture_id;
      return;
    }
  }
  QP(capture_drops);
}''',
    "capture drop",
)

# 3. Pool acquire scan cost.
sub1(
    '''  if (self->free_capture_list_count > 0) {
    for (uint32_t i = 0; i < self->list.size; i++) {
      if (array_get(&self->list, i)->size == UINT32_MAX) {''',
    '''  QP(pool_acquires);
  if (self->free_capture_list_count > 0) {
    for (uint32_t i = 0; i < self->list.size; i++) {
      QP(pool_scan_steps);
      if (array_get(&self->list, i)->size == UINT32_MAX) {''',
    "pool scan",
)

# 4. Dedup comparison cost.
sub1(
    '''  *left_contains_right = true;
  *right_contains_left = true;
  unsigned i = 0, j = 0;
  for (;;) {''',
    '''  *left_contains_right = true;
  *right_contains_left = true;
  QP(dedup_compares);
  unsigned i = 0, j = 0;
  for (;;) {
    QP(dedup_capture_steps);''',
    "dedup compare",
)

# 5. State ordering comparison cost.
sub1(
    '''  const QueryState *a,
  const QueryState *b
) {
  if (a->start_depth != b->start_depth) return a->start_depth < b->start_depth;''',
    '''  const QueryState *a,
  const QueryState *b
) {
  QP(sort_compares);
  if (a->start_depth != b->start_depth) return a->start_depth < b->start_depth;''',
    "sort compare",
)

# 6. State creation.
sub1(
    '''  LOG(
    "  start state. pattern:%u, step:%u\\n",
    pattern->pattern_index,
    pattern->step_index
  );''',
    '''  QP(states_started);
  LOG(
    "  start state. pattern:%u, step:%u\\n",
    pattern->pattern_index,
    pattern->step_index
  );''',
    "state started",
)

# 7. State copy (fan-out).
sub1(
    '''  array_insert(&self->states, state_index + 1, copy);
  *state_ref = array_get(&self->states, state_index);''',
    '''  QP(states_copied);
  array_insert(&self->states, state_index + 1, copy);
  *state_ref = array_get(&self->states, state_index);''',
    "state copied",
)

# 8. should_descend scan cost.
sub1(
    '''  for (unsigned i = 0; i < self->states.size; i++) {
    QueryState *state = array_get(&self->states, i);
    QueryStep *next_step = array_get(&self->query->steps, state->step_index);
    if (
      next_step->depth != PATTERN_DONE_MARKER &&
      state->start_depth + next_step->depth > self->depth
    ) {
      return true;
    }
  }''',
    '''  for (unsigned i = 0; i < self->states.size; i++) {
    QP(descend_scan_steps);
    QueryState *state = array_get(&self->states, i);
    QueryStep *next_step = array_get(&self->query->steps, state->step_index);
    if (
      next_step->depth != PATTERN_DONE_MARKER &&
      state->start_depth + next_step->depth > self->depth
    ) {
      return true;
    }
  }''',
    "descend scan",
)

# 9. Per-node state scan: total visits and depth-filtered visits.
sub1(
    '''          QueryState *state = array_get(&self->states, j);
          QueryStep *step = array_get(&self->query->steps, state->step_index);
          state->has_in_progress_alternatives = false;
          copy_count = 0;

          // Check that the node matches all of the criteria for the next
          // step of the pattern.
          if ((uint32_t)state->start_depth + (uint32_t)step->depth != self->depth) continue;''',
    '''          QueryState *state = array_get(&self->states, j);
          QueryStep *step = array_get(&self->query->steps, state->step_index);
          state->has_in_progress_alternatives = false;
          copy_count = 0;

          // Check that the node matches all of the criteria for the next
          // step of the pattern.
          QP(state_visits);
          if ((uint32_t)state->start_depth + (uint32_t)step->depth != self->depth) {
            QP(state_visits_depth_skip);
            continue;
          }''',
    "state visits",
)

# 10. Node entry + peak state tracking.
sub1(
    '''        bool node_is_error = symbol == ts_builtin_sym_error;''',
    '''        QP(nodes_entered);
        QPMAX(max_states, self->states.size);
        QPMAX(max_finished, self->finished_states.size);
        bool node_is_error = symbol == ts_builtin_sym_error;''',
    "node entered",
)

# 11. Descent counter.
sub1(
    '''        switch (ts_tree_cursor_goto_first_child_internal(&self->cursor)) {
          case TreeCursorStepVisible:
            self->depth++;''',
    '''        QP(nodes_descended);
        switch (ts_tree_cursor_goto_first_child_internal(&self->cursor)) {
          case TreeCursorStepVisible:
            self->depth++;''',
    "descend",
)

# 12. Match counter.
sub1(
    '''static void ts_query_cursor__push_finished_state(
  TSQueryCursor *self,
  QueryState *state
) {
  state->heap_insert_order = self->next_finished_state_id++;''',
    '''static void ts_query_cursor__push_finished_state(
  TSQueryCursor *self,
  QueryState *state
) {
  QP(matches);
  state->heap_insert_order = self->next_finished_state_id++;''',
    "match count",
)

# ---- compile-phase timers ----

sub1(
    'QProbe qprobe = {0};',
    """QProbe qprobe = {0};
double qp_t_analyze_ms = 0, qp_t_subgraph_scan_ms = 0, qp_t_perform_ms = 0;
unsigned long qp_predecessor_map_bytes = 0, qp_subgraph_count = 0, qp_subgraph_nodes = 0;
unsigned long qp_perform_calls = 0, qp_perform_iterations = 0, qp_analysis_aborts = 0;
#include <time.h>
static double qp_now(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}""",
    "phase timer globals",
)

sub1(
    "  if (!ts_query__analyze_patterns(self, error_offset)) {",
    "  double _qp_a0 = qp_now();\n  if (!ts_query__analyze_patterns(self, error_offset)) {\n    qp_t_analyze_ms += qp_now() - _qp_a0;",
    "analyze timer start",
)

sub1(
    '''  #ifdef DEBUG_DUMP_STEPS
    ts_query__dump_steps(self, "post-analysis");
  #endif

  array_delete(&self->string_buffer);''',
    '''  qp_t_analyze_ms += qp_now() - _qp_a0;
  #ifdef DEBUG_DUMP_STEPS
    ts_query__dump_steps(self, "post-analysis");
  #endif

  array_delete(&self->string_buffer);''',
    "analyze timer end",
)

sub1(
    """  return (StatePredecessorMap) {
    .contents = ts_calloc(""",
    """  qp_predecessor_map_bytes = (unsigned long)language->state_count * (MAX_STATE_PREDECESSOR_COUNT + 1) * sizeof(TSStateId);
  return (StatePredecessorMap) {
    .contents = ts_calloc(""",
    "predecessor map size",
)

sub1(
    """  StatePredecessorMap predecessor_map = state_predecessor_map_new(self->language);
  for (TSStateId state = 1; state < (uint16_t)self->language->state_count; state++) {""",
    """  StatePredecessorMap predecessor_map = state_predecessor_map_new(self->language);
  double _qp_s0 = qp_now();
  for (TSStateId state = 1; state < (uint16_t)self->language->state_count; state++) {""",
    "parse-table scan start",
)

sub1(
    """  // For each subgraph, compute the preceding states by walking backward
  // from the end states using the predecessor map.""",
    """  qp_t_subgraph_scan_ms += qp_now() - _qp_s0;
  // For each subgraph, compute the preceding states by walking backward
  // from the end states using the predecessor map.""",
    "parse-table scan end",
)

sub1(
    """  QueryAnalysis analysis = query_analysis__new();
  for (unsigned i = 0; i < parent_step_indices.size; i++) {""",
    """  qp_subgraph_count = subgraphs.size;
  for (unsigned _i = 0; _i < subgraphs.size; _i++) qp_subgraph_nodes += array_get(&subgraphs, _i)->nodes.size;
  QueryAnalysis analysis = query_analysis__new();
  for (unsigned i = 0; i < parent_step_indices.size; i++) {""",
    "subgraph stats",
)

sub1(
    """  unsigned recursion_depth_limit = 0;
  unsigned prev_final_step_count = 0;""",
    """  qp_perform_calls++;
  double _qp_p0 = qp_now();
  unsigned recursion_depth_limit = 0;
  unsigned prev_final_step_count = 0;""",
    "perform_analysis timer start",
)

sub1(
    """  for (unsigned iteration = 0;; iteration++) {
    if (iteration == MAX_ANALYSIS_ITERATION_COUNT) {
      analysis->did_abort = true;
      break;
    }""",
    """  for (unsigned iteration = 0;; iteration++) {
    qp_perform_iterations++;
    if (iteration == MAX_ANALYSIS_ITERATION_COUNT) {
      analysis->did_abort = true;
      qp_analysis_aborts++;
      break;
    }""",
    "perform_analysis iterations",
)

sub1(
    """    AnalysisStateSet _states = analysis->states;
    analysis->states = analysis->next_states;
    analysis->next_states = _states;
  }
}""",
    """    AnalysisStateSet _states = analysis->states;
    analysis->states = analysis->next_states;
    analysis->next_states = _states;
  }
  qp_t_perform_ms += qp_now() - _qp_p0;
}""",
    "perform_analysis timer end",
)


open(p, "w").write(src)
print(f"wrote {p} ({len(src.splitlines())} lines)")
