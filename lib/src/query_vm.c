#include "./query_vm.h"
#include "./point.h"
#include "./subtree.h"
#include "./tree_cursor.h"
#include "./ts_assert.h"

#include <string.h>

/**********
 * Threads
 **********/

// Releases the captures of the thread `index`, if it has any.
static void query_vm_thread_clear(QueryVm *self, QueryThreadId index) {
  QueryThread *thread = query_thread_store_get(&self->threads, index);
  if (!query_capture_chain_is_empty(thread->chain)) {
    ts_assert(self->capture_holder_count > 0);
    self->capture_holder_count--;
    query_capture_store_release(&self->captures, thread->chain);
    thread->chain = QUERY_CAPTURE_CHAIN_EMPTY;
  }
}

// Frees the thread `index`, which no list has anymore, and its captures.
static void query_vm_thread_free(QueryVm *self, QueryThreadId index) {
  query_vm_thread_clear(self, index);
  query_thread_store_free(&self->threads, index);
}

// Abandons the thread `index`, which a list or the candidates still have: it
// releases its captures now, and its slot is freed when the list next reaches
// it.
static void query_vm_thread_abandon(QueryVm *self, QueryThreadId index) {
  query_vm_thread_clear(self, index);
  query_thread_store_get(&self->threads, index)->state = QueryThreadStateAbandoned;
}

/***********
 * Schedule
 ***********/

// Stops the thread `index` to wait for the next node at `depth`, after the
// threads that are already waiting there.
static void query_vm_wait(QueryVm *self, QueryThreadId index, uint32_t depth) {
  QueryThreadList *list = query_schedule_waiting_at_or_new(&self->schedule, depth);
  query_thread_list_append(list, &self->threads, index);
  (*array_get(&self->schedule.waiting_counts, query_thread_store_get(&self->threads, index)->pattern))++;
  array_push(&self->schedule.arrivals, index);
}

// Starts to give the current node, at `depth`, to the threads that were
// waiting for it.
static void query_vm_begin_delivery(QueryVm *self, uint32_t depth) {
  self->schedule.delivering = query_schedule_take_waiting(&self->schedule, depth);
  self->schedule.staying = QUERY_THREAD_LIST_EMPTY;
}

// Returns the next thread to give the current node to, which is no longer
// waiting, or NONE.
static QueryThreadId query_vm_next_delivery(QueryVm *self) {
  const QueryThreadId index = query_thread_list_pop_front(&self->schedule.delivering, &self->threads);
  if (query_thread_id_is_none(index)) return QUERY_THREAD_NONE;
  (*array_get(&self->schedule.waiting_counts, query_thread_store_get(&self->threads, index)->pattern))--;
  return index;
}

// Keeps the thread `index`, which skips the current node, waiting for a later
// one.
static void query_vm_stay(QueryVm *self, QueryThreadId index) {
  query_thread_list_append(&self->schedule.staying, &self->threads, index);
  (*array_get(&self->schedule.waiting_counts, query_thread_store_get(&self->threads, index)->pattern))++;
}

// Ends giving the current node, at `depth`, to the threads that were waiting
// for it: the ones that stayed wait there again, ahead of the ones that
// started to wait there meanwhile.
static void query_vm_end_delivery(QueryVm *self, uint32_t depth) {
  QueryThreadList *staying = &self->schedule.staying;
  if (query_thread_id_is_none(staying->first)) return;
  QueryThreadList *waiting = query_schedule_waiting_at_or_new(&self->schedule, depth);
  query_thread_list_prepend_list(waiting, &self->threads, staying);
}

// Frees the threads in `list`, which were waiting.
static inline void query_vm_free_waiting_list(QueryVm *self, QueryThreadList list) {
  QueryThreadId index = list.first;
  while (!query_thread_id_is_none(index)) {
    const QueryThread *thread = query_thread_store_get(&self->threads, index);
    const QueryThreadId next = thread->next;
    (*array_get(&self->schedule.waiting_counts, thread->pattern))--;
    query_vm_thread_free(self, index);
    index = next;
  }
}

// Frees the threads that are waiting at `depth`.
static void query_vm_free_waiting_at(QueryVm *self, uint32_t depth) {
  query_vm_free_waiting_list(self, query_schedule_take_waiting(&self->schedule, depth));
}

// Frees every waiting thread.
static void query_vm_free_waiting(QueryVm *self) {
  for (uint32_t i = 0; i < self->schedule.waiting.size; i++) {
    query_vm_free_waiting_list(self, array_get(&self->schedule.waiting, i)->threads);
  }
  array_clear(&self->schedule.waiting);
}

// Records that the thread `index` finished a match on the current node.
static void query_vm_accept(QueryVm *self, QueryThreadId index) {
  array_push(&self->schedule.candidates, index);
}

// Frees the threads whose matches are held back.
static void query_vm_free_held(QueryVm *self) {
  // The walk is between nodes at this point, so every candidate must be held.
  ts_assert(self->schedule.candidates.size == self->schedule.held_count);
  for (uint32_t i = 0; i < self->schedule.held_count; i++) {
    query_vm_thread_free(self, *array_get(&self->schedule.candidates, i));
  }
  array_clear(&self->schedule.candidates);
  self->schedule.held_count = 0;
}

/***************
 * Match limit
 ***************/

// Replaces `*best` with the thread `index`, if it is unfinished, holds
// captures, and captured its first node before the thread `*best` did,
// or at the same byte with an earlier pattern.
static void query_vm_consider_unfinished(
  QueryVm *self,
  QueryThreadId index,
  QueryThreadId *best
) {
  const QueryThread *thread = query_thread_store_get(&self->threads, index);
  if (thread->state == QueryThreadStateAbandoned || query_capture_chain_is_empty(thread->chain)) return;
  if (!query_thread_id_is_none(*best)) {
    const QueryThread *other = query_thread_store_get(&self->threads, *best);
    if (thread->first_start_byte > other->first_start_byte) return;
    if (thread->first_start_byte == other->first_start_byte && thread->pattern >= other->pattern) return;
  }
  *best = index;
}

static inline void query_vm_consider_unfinished_list(
  QueryVm *self,
  QueryThreadList list,
  QueryThreadId *best
) {
  for (
    QueryThreadId index = list.first;
    !query_thread_id_is_none(index);
    index = query_thread_store_get(&self->threads, index)->next
  ) {
    query_vm_consider_unfinished(self, index, best);
  }
}

// Returns the unfinished thread with captures whose first capture is earliest
// in the document, or NONE. That is the match that the match limit gives up
// first, and the one whose captures `next_capture` must not pass.
static QueryThreadId query_vm_first_unfinished(QueryVm *self) {
  // A thread that has not finished is waiting, being given the current node,
  // or held back.
  QueryThreadId best = QUERY_THREAD_NONE;
  for (uint32_t i = 0; i < self->schedule.waiting.size; i++) {
    query_vm_consider_unfinished_list(self, array_get(&self->schedule.waiting, i)->threads, &best);
  }
  query_vm_consider_unfinished_list(self, self->schedule.delivering, &best);
  query_vm_consider_unfinished_list(self, self->schedule.staying, &best);
  for (uint32_t i = 0; i < self->schedule.held_count; i++) {
    query_vm_consider_unfinished(self, *array_get(&self->schedule.candidates, i), &best);
  }
  return best;
}

// Gives up the unfinished match of the thread `index`: a held one is freed at
// once, and one that a list has is abandoned.
static void query_vm_evict(QueryVm *self, QueryThreadId index) {
  self->has_first_unfinished = false;
  for (uint32_t i = 0; i < self->schedule.held_count; i++) {
    if (array_get(&self->schedule.candidates, i)->index == index.index) {
      query_schedule_remove_held(&self->schedule, i);
      query_vm_thread_free(self, index);
      return;
    }
  }
  query_vm_thread_abandon(self, index);
}

// Makes room for one more thread or match with captures, once the match limit
// is reached, by giving up the unfinished one that captured the earliest node.
// Returns false if there is none.
static bool query_vm_make_room_at_limit(QueryVm *self) {
  self->did_exceed_match_limit = true;
  const QueryThreadId victim = query_vm_first_unfinished(self);
  if (query_thread_id_is_none(victim)) return false;
  query_vm_evict(self, victim);
  return true;
}

// Counts one more thread or match that holds captures, against the match limit,
// after making room for it if the limit is reached. Returns false if there is no
// room. The thread that needs the room must be in no list.
static inline bool query_vm_add_capture_holder(QueryVm *self) {
  if (
    self->capture_holder_count >= self->match_limit &&
    !query_vm_make_room_at_limit(self)
  ) return false;
  self->capture_holder_count++;
  return true;
}

// Copies the thread `original`, which no list has, so that the copy continues
// at `pc`. Returns NONE if there is no room for it.
static QueryThreadId query_vm_copy(
  QueryVm *self,
  QueryThreadId original,
  QueryProgramCounter pc
) {
  if (!query_capture_chain_is_empty(query_thread_store_get(&self->threads, original)->chain)) {
    if (!query_vm_add_capture_holder(self)) return QUERY_THREAD_NONE;
  }
  const QueryThreadId copy = query_thread_store_take(&self->threads);
  QueryThread *thread = query_thread_store_get(&self->threads, copy);
  *thread = *query_thread_store_get(&self->threads, original);
  thread->pc = pc;
  thread->next = QUERY_THREAD_NONE;
  query_capture_store_retain(&self->captures, thread->chain);
  return copy;
}

/************
 * Running
 ************/

static bool query_vm_is_capture_disabled(const QueryVmProgram *query, uint16_t capture) {
  return *array_get(&query->disabled_captures, capture);
}

// Computes the fields of the current node that are computed the first time
// they are needed.
static void query_vm_load_status(QueryVm *self) {
  QueryVmNode *node = &self->node;
  if (node->has_status) return;
  ts_tree_cursor_current_status(
    &self->cursor,
    &node->field,
    &node->has_later_siblings,
    &node->can_have_later_siblings_with_this_field
  );
  node->has_status = true;
}

// Returns the depth that `thread`, which is stopped at a wait, waits at.
static uint32_t query_vm_wait_depth(const QueryVm *self, const QueryThread *thread) {
  const QueryInstruction wait = *array_get(
    &self->query->program.instructions,
    thread->pc.index
  );
  return thread->base + query_instruction_depth(wait);
}

// Runs the thread `index` on the current node, until it waits, fails, or
// finishes a match.
static void query_vm_execute(QueryVm *self, QueryThreadId index) {
  QueryVmNode *node = &self->node;
  // INVARIANT: Only the threads that wait or are held back are abandoned.
  ts_assert(query_thread_store_get(&self->threads, index)->state == QueryThreadStateLive);
  for (;;) {
    QueryThread *thread = query_thread_store_get(&self->threads, index);
    const QueryInstruction instruction = *array_get(
      &self->query->program.instructions,
      thread->pc.index
    );
    switch (query_instruction_opcode(instruction)) {
      case QueryOpcodeWait:
      case QueryOpcodeWaitAnchored:
      case QueryOpcodeWaitImmediate:
        query_vm_wait(self, index, thread->base + query_instruction_depth(instruction));
        return;
      case QueryOpcodeSymbol:
        if (node->symbol != query_instruction_symbol(instruction)) goto fail;
        break;
      case QueryOpcodeSupertype: {
        const TSSymbol supertype = query_instruction_symbol(instruction);
        if (!ts_tree_cursor_is_wrapped_in_supertype(&self->cursor, supertype)) goto fail;
        break;
      }
      case QueryOpcodeField:
        query_vm_load_status(self);
        if (node->field != query_instruction_field(instruction)) goto fail;
        break;
      case QueryOpcodeNoField:
        if (ts_node_child_by_field_id(node->node, query_instruction_field(instruction)).id) goto fail;
        break;
      case QueryOpcodeNamed:
        if (!ts_node_is_named(node->node)) goto fail;
        break;
      case QueryOpcodeNotError:
        if (node->symbol == ts_builtin_sym_error) goto fail;
        break;
      case QueryOpcodeMissing:
        if (!ts_node_is_missing(node->node)) goto fail;
        break;
      case QueryOpcodeLast: {
        const uint32_t depth = thread->base + query_instruction_depth(instruction);
        // INVARIANT: The node that the thread matched at `depth` is the current
        // node or one of its ancestors.
        ts_assert(depth <= self->depth);
        const uint32_t stack_index = ts_tree_cursor_visible_ancestor_index(&self->cursor, self->depth - depth);
        if (ts_tree_cursor_has_later_named_siblings_at(&self->cursor, stack_index)) goto fail;
        break;
      }
      case QueryOpcodeCapture: {
        const uint16_t capture = query_instruction_capture(instruction);
        if (query_vm_is_capture_disabled(self->query, capture)) break;
        if (query_capture_chain_is_empty(thread->chain)) {
          if (!query_vm_add_capture_holder(self)) goto fail;
          thread->first_start_byte = ts_node_start_byte(node->node);
        }
        thread->chain = query_capture_store_append(&self->captures, thread->chain, node->node, capture);
        break;
      }
      case QueryOpcodeSplit: {
        const QueryThreadId copy = query_vm_copy(self, index, query_instruction_target(instruction));
        if (!query_thread_id_is_none(copy)) array_push(&self->schedule.work, copy);
        thread = query_thread_store_get(&self->threads, index);
        break;
      }
      case QueryOpcodeJump:
        thread->pc = query_instruction_target(instruction);
        continue;
      case QueryOpcodeAccept:
        query_vm_accept(self, index);
        return;
    }
    thread->pc = query_program_counter_next(thread->pc);
  }
fail:
  query_vm_thread_free(self, index);
}

// Runs the thread `index` on the current node, and then each copy that it
// splits off, in the order that they are split off.
static void query_vm_run(QueryVm *self, QueryThreadId index) {
  // INVARIANT: The work is empty whenever threads start to run.
  ts_assert(self->schedule.work.size == 0);
  array_push(&self->schedule.work, index);
  for (uint32_t i = 0; i < self->schedule.work.size; i++) {
    query_vm_execute(self, *array_get(&self->schedule.work, i));
  }
  array_clear(&self->schedule.work);
}

// Records whether a thread that is given the current node reaches into the
// cursor's range, if the node is at the top level of its pattern.
static void query_vm_receive(QueryVm *self, QueryThreadId index, const TSRange *node_range) {
  QueryThread *thread = query_thread_store_get(&self->threads, index);
  if (!thread->intersects_range && thread->base == self->depth) {
    // The pattern's first node starts before the range ends, so the match
    // reaches into the range once one of its top-level nodes ends after the
    // range starts.
    thread->intersects_range =
      node_range->end_byte > self->included_range.start_byte &&
      point_gt(node_range->end_point, self->included_range.start_point);
  }
}

/****************
 * Longest match
 ****************/

// Returns whether a thread that waits at `pc` can still capture a node.
static bool query_vm_may_capture(const QueryVm *self, QueryProgramCounter pc) {
  return query_instruction_may_capture(
    *array_get(&self->query->program.instructions, pc.index)
  );
}

// Returns whether a thread that has not finished, with the same pattern and
// start depth as the finished match of the thread `index`, may still finish a
// match with more captures: it has all of the match's captures, and either has
// more already or can still capture something.
static bool query_vm_has_longer(QueryVm *self, QueryThreadId index) {
  const QueryThread *thread = query_thread_store_get(&self->threads, index);
  const uint32_t pattern = thread->pattern;
  const uint32_t base = thread->base;
  const QueryCaptureChain chain = thread->chain;
  if (query_schedule_waiting_count(&self->schedule, pattern) == 0) return false;
  for (
    uint32_t i = query_schedule_bucket_from(&self->schedule, base);
    i < self->schedule.waiting.size;
    i++
  ) {
    const QueryWaitingBucket *bucket = array_get(&self->schedule.waiting, i);
    // INVARIANT: Threads wait for nodes no deeper than the current node's
    // children.
    ts_assert(bucket->depth <= self->depth + 1);
    for (
      QueryThreadId other = bucket->threads.first;
      !query_thread_id_is_none(other);
      other = query_thread_store_get(&self->threads, other)->next
    ) {
      const QueryThread *candidate = query_thread_store_get(&self->threads, other);
      if (candidate->state == QueryThreadStateAbandoned || candidate->pattern != pattern || candidate->base != base) continue;
      const QueryCaptureRelation relation =
        query_capture_store_compare(&self->captures, chain, candidate->chain);
      if (relation == QueryCaptureRelationSubset) return true;
      if (
        relation == QueryCaptureRelationEqual &&
        query_vm_may_capture(self, candidate->pc)
      ) return true;
    }
  }
  return false;
}

// Abandons the threads with the same pattern and start depth as the finished
// match of the thread `index` that can only finish the same match again or a
// shorter one: their captures are all in the match, and they cannot capture
// anything more.
static void query_vm_drop_redundant(QueryVm *self, QueryThreadId index) {
  const QueryThread *thread = query_thread_store_get(&self->threads, index);
  const uint32_t pattern = thread->pattern;
  const uint32_t base = thread->base;
  const QueryCaptureChain chain = thread->chain;
  if (query_schedule_waiting_count(&self->schedule, pattern) == 0) return;
  for (
    uint32_t i = query_schedule_bucket_from(&self->schedule, base);
    i < self->schedule.waiting.size;
    i++
  ) {
    const QueryWaitingBucket *bucket = array_get(&self->schedule.waiting, i);
    // INVARIANT: Threads wait for nodes no deeper than the current node's
    // children.
    ts_assert(bucket->depth <= self->depth + 1);
    for (
      QueryThreadId other = bucket->threads.first;
      !query_thread_id_is_none(other);
      other = query_thread_store_get(&self->threads, other)->next
    ) {
      const QueryThread *candidate = query_thread_store_get(&self->threads, other);
      if (candidate->state == QueryThreadStateAbandoned || candidate->pattern != pattern || candidate->base != base) continue;
      if (query_vm_may_capture(self, candidate->pc)) continue;
      const QueryCaptureRelation relation =
        query_capture_store_compare(&self->captures, candidate->chain, chain);
      if (
        relation == QueryCaptureRelationSubset ||
        relation == QueryCaptureRelationEqual
      ) query_vm_thread_abandon(self, other);
    }
  }
}

// Decides which of two threads with the same next step, or two finished
// matches, the longest-match rule keeps, when one's captures are all in the
// other's. Returns the thread to drop, or NONE.
static QueryThreadId query_vm_shorter(QueryVm *self, QueryThreadId older, QueryThreadId newer) {
  const QueryThread *older_thread = query_thread_store_get(&self->threads, older);
  const QueryThread *newer_thread = query_thread_store_get(&self->threads, newer);
  const QueryCaptureRelation relation = query_capture_store_compare(
    &self->captures,
    older_thread->chain,
    newer_thread->chain
  );
  // A thread that does not reach into the range yet may never finish a match,
  // so it never replaces one that does.
  switch (relation) {
    case QueryCaptureRelationEqual:
      if (older_thread->intersects_range != newer_thread->intersects_range) {
        return older_thread->intersects_range ? newer : older;
      }
      return newer;
    case QueryCaptureRelationSubset:
      if (newer_thread->intersects_range || !older_thread->intersects_range) return older;
      break;
    case QueryCaptureRelationSuperset:
      if (older_thread->intersects_range || !newer_thread->intersects_range) return newer;
      break;
    case QueryCaptureRelationNone:
      break;
  }
  return QUERY_THREAD_NONE;
}

// Drops the shorter of two threads at the same wait, from the same start
// depth, when one has all of the other's captures. Only pairs with a thread
// that stopped there in this event are compared: older pairs already were.
static void query_vm_drop_shorter_arrivals(QueryVm *self) {
  for (uint32_t i = 0; i < self->schedule.arrivals.size; i++) {
    const QueryThreadId arrival = *array_get(&self->schedule.arrivals, i);
    const QueryThread *thread = query_thread_store_get(&self->threads, arrival);
    if (thread->state == QueryThreadStateAbandoned) continue;
    const QueryProgramCounter pc = thread->pc;
    const uint32_t base = thread->base;
    const QueryThreadList *list = query_schedule_waiting_at(&self->schedule, query_vm_wait_depth(self, thread));
    // INVARIANT: The arrival is still in the list of its wait.
    ts_assert(list != NULL);
    // Lists are oldest first, so the threads before the arrival are older.
    bool other_is_older = true;
    for (
      QueryThreadId other = list->first;
      !query_thread_id_is_none(other);
      other = query_thread_store_get(&self->threads, other)->next
    ) {
      if (other.index == arrival.index) {
        other_is_older = false;
        continue;
      }
      const QueryThread *candidate = query_thread_store_get(&self->threads, other);
      if (
        candidate->state == QueryThreadStateAbandoned ||
        candidate->pc.index != pc.index ||
        candidate->base != base
      ) continue;
      const QueryThreadId dropped = other_is_older
        ? query_vm_shorter(self, other, arrival)
        : query_vm_shorter(self, arrival, other);
      if (query_thread_id_is_none(dropped)) continue;
      query_vm_thread_abandon(self, dropped);
      if (dropped.index == arrival.index) break;
    }
  }
  array_clear(&self->schedule.arrivals);
}

/************
 * Finishing
 ************/

// The most captures that a finished match can have and still be copied into
// the shared short list each time that the API shows it. A match with more is
// copied once, into a long list of its own.
#define QUERY_VM_MAX_SHORT_LIST_LENGTH 32

// Returns the index of a long list to fill.
static uint32_t query_vm_long_list_acquire(QueryVm *self) {
  for (uint32_t i = 0; i < self->matches.long_lists.size; i++) {
    // A long list in use holds more than `QUERY_VM_MAX_SHORT_LIST_LENGTH`, so
    // empty lists are free.
    if (array_get(&self->matches.long_lists, i)->size == 0) return i;
  }
  array_push(&self->matches.long_lists, ((QueryCaptureList) array_new()));
  return self->matches.long_lists.size - 1;
}

// Frees a long list. Its contents stay where they are until it is acquired
// again, so a match that was just returned can still be read.
static void query_vm_long_list_release(QueryVm *self, uint32_t index) {
  QueryCaptureList *list = array_get(&self->matches.long_lists, index);
  // INVARIANT: Only a match with more captures than the short list can hold
  // has a long list.
  ts_assert(list->size > QUERY_VM_MAX_SHORT_LIST_LENGTH);
  array_clear(list);
}

// Returns whether `match` shows its captures with a long list of its own.
static bool query_capture_view_owns_list(const QueryFinishedMatch *match) {
  switch ((QueryCaptureViewState)match->view) {
    case QueryCaptureViewChain:
    case QueryCaptureViewChainExhausted: return false;
    case QueryCaptureViewList:
    case QueryCaptureViewListExhausted: return true;
  }
  ts_unreachable();
}

// Returns whether `next_capture` has returned all of the captures of `match`.
static bool query_capture_view_is_exhausted(const QueryFinishedMatch *match) {
  switch ((QueryCaptureViewState)match->view) {
    case QueryCaptureViewChain:
    case QueryCaptureViewList: return false;
    case QueryCaptureViewChainExhausted:
    case QueryCaptureViewListExhausted: return true;
  }
  ts_unreachable();
}

static void query_capture_view_set_exhausted(QueryFinishedMatch *match, bool exhausted) {
  if (query_capture_view_owns_list(match)) {
    match->view = exhausted ? QueryCaptureViewListExhausted : QueryCaptureViewList;
  } else {
    match->view = exhausted ? QueryCaptureViewChainExhausted : QueryCaptureViewChain;
  }
}

// Returns the list that shows the captures of `match`, which has been
// materialized.
static const QueryCaptureList *query_vm_match_captures(const QueryVm *self, const QueryFinishedMatch *match) {
  return query_capture_view_owns_list(match)
    ? array_get(&self->matches.long_lists, match->captures.list)
    : &self->matches.short_list;
}

// Copies the captures of `match` into a list that the API can show: the shared
// short list, or, for a long chain, which would be slow to copy again each
// time that `next_capture` shows the match, a list of the match's own.
static void query_vm_materialize(QueryVm *self, QueryFinishedMatch *match) {
  if (query_capture_view_owns_list(match)) return;
  if (self->matches.short_list_id == match->id) return;
  const QueryCaptureChain chain = match->captures.chain;
  const uint32_t length = query_capture_store_length(&self->captures, chain);
  if (length > QUERY_VM_MAX_SHORT_LIST_LENGTH) {
    const uint32_t index = query_vm_long_list_acquire(self);
    QueryCaptureList *list = array_get(&self->matches.long_lists, index);
    query_capture_store_copy(&self->captures, chain, list, length);
    // The match holds its captures with the list from now on.
    query_capture_store_release(&self->captures, chain);
    match->captures.list = index;
    match->view = query_capture_view_is_exhausted(match) ? QueryCaptureViewListExhausted : QueryCaptureViewList;
  } else {
    query_capture_store_copy(&self->captures, chain, &self->matches.short_list, length);
    self->matches.short_list_id = match->id;
  }
}

// Releases the captures of `match`, which is being removed.
static void query_vm_finished_release(QueryVm *self, QueryFinishedMatch *match) {
  if (query_capture_view_owns_list(match)) {
    query_vm_long_list_release(self, match->captures.list);
  } else if (!query_capture_chain_is_empty(match->captures.chain)) {
    query_capture_store_release(&self->captures, match->captures.chain);
  } else {
    return;
  }
  ts_assert(self->capture_holder_count > 0);
  self->capture_holder_count--;
}

// Sets `byte` to where the next capture of `match` for `next_capture` starts,
// and returns whether it has one.
static bool query_vm_finished_next_byte(const QueryFinishedMatch *match, uint32_t *byte) {
  if (query_capture_view_is_exhausted(match)) return false;
  *byte = match->next_start_byte;
  return true;
}

// Records that the next capture of `match` has been returned.
static void query_vm_finished_consumed(QueryVm *self, QueryFinishedMatch *match) {
  const QueryCaptureList *list = query_vm_match_captures(self, match);
  match->consumed_capture_count++;
  const bool exhausted = match->consumed_capture_count >= list->size;
  query_capture_view_set_exhausted(match, exhausted);
  if (!exhausted) {
    match->next_start_byte = ts_node_start_byte(array_get(list, match->consumed_capture_count)->node);
  }
}

// Orders threads whose matches finish at the same time: by start depth, then
// pattern, then position of their first capture, with no captures first.
static bool query_vm_finishes_before(QueryVm *self, QueryThreadId a, QueryThreadId b) {
  const QueryThread *left = query_thread_store_get(&self->threads, a);
  const QueryThread *right = query_thread_store_get(&self->threads, b);
  if (left->base != right->base) return left->base < right->base;
  if (left->pattern != right->pattern) return left->pattern < right->pattern;
  const bool left_is_empty = query_capture_chain_is_empty(left->chain);
  const bool right_is_empty = query_capture_chain_is_empty(right->chain);
  if (left_is_empty || right_is_empty) return left_is_empty && !right_is_empty;
  return left->first_start_byte < right->first_start_byte;
}

// Turns the threads in `work`, of which there is at least one, into finished
// matches, in order.
static void query_vm_finish_threads(QueryVm *self) {
  for (uint32_t i = 1; i < self->schedule.work.size; i++) {
    const QueryThreadId key = *array_get(&self->schedule.work, i);
    uint32_t j = i;
    while (j > 0 && query_vm_finishes_before(self, key, *array_get(&self->schedule.work, j - 1))) {
      *array_get(&self->schedule.work, j) = *array_get(&self->schedule.work, j - 1);
      j--;
    }
    *array_get(&self->schedule.work, j) = key;
  }
  for (uint32_t i = 0; i < self->schedule.work.size; i++) {
    const QueryThreadId index = *array_get(&self->schedule.work, i);
    QueryThread *thread = query_thread_store_get(&self->threads, index);
    ts_assert(self->matches.next_id != QUERY_VM_NONE);
    // A match without captures has none for `next_capture` to return, so its
    // next capture's start is never read.
    array_push(&self->matches.finished, ((QueryFinishedMatch) {
      .id = self->matches.next_id++,
      .pattern = thread->pattern,
      .view = query_capture_chain_is_empty(thread->chain) ? QueryCaptureViewChainExhausted : QueryCaptureViewChain,
      .captures.chain = thread->chain,
      .consumed_capture_count = 0,
      .next_start_byte = thread->first_start_byte,
    }));
    // The finished match takes over the thread's captures, and its place among
    // the capture holders that the match limit counts.
    thread->chain = QUERY_CAPTURE_CHAIN_EMPTY;
    query_vm_thread_free(self, index);
  }
  array_clear(&self->schedule.work);
}

// Turns the threads in `work` into finished matches, in order. Returns whether
// there were any.
static inline bool query_vm_finish(QueryVm *self) {
  if (self->schedule.work.size == 0) return false;
  query_vm_finish_threads(self);
  return true;
}

// Moves the held matches that no thread can beat anymore, or all of them if
// `all` is set, to `work`, to be finished.
static void query_vm_release_held(QueryVm *self, bool all) {
  uint32_t i = 0;
  while (i < self->schedule.held_count) {
    const QueryThreadId index = *array_get(&self->schedule.candidates, i);
    if (all || !query_vm_has_longer(self, index)) {
      array_push(&self->schedule.work, index);
      query_schedule_remove_held(&self->schedule, i);
    } else {
      i++;
    }
  }
}

// Applies the longest-match rule to the threads that stopped or finished in
// this event, and finishes the matches that are ready. Returns whether any
// match finished.
static bool query_vm_settle(QueryVm *self) {
  // INVARIANT: The work is empty once the threads have run.
  ts_assert(self->schedule.work.size == 0);
  query_vm_drop_shorter_arrivals(self);

  // A new match is dropped if it does not reach into the range, or if an
  // earlier one, held or new, has all of its captures, or the other way
  // around.
  for (uint32_t i = self->schedule.held_count; i < self->schedule.candidates.size; i++) {
    const QueryThreadId index = *array_get(&self->schedule.candidates, i);
    const QueryThread *thread = query_thread_store_get(&self->threads, index);
    if (!thread->intersects_range) {
      query_vm_thread_abandon(self, index);
      continue;
    }
    for (uint32_t j = 0; j < i && thread->state != QueryThreadStateAbandoned; j++) {
      const QueryThreadId other = *array_get(&self->schedule.candidates, j);
      const QueryThread *candidate = query_thread_store_get(&self->threads, other);
      if (candidate->state == QueryThreadStateAbandoned || candidate->pattern != thread->pattern || candidate->base != thread->base) continue;
      const QueryThreadId dropped = query_vm_shorter(self, other, index);
      if (!query_thread_id_is_none(dropped)) query_vm_thread_abandon(self, dropped);
    }
  }

  // The held matches that were dropped are freed, and the ones that no thread
  // can beat anymore are finished.
  uint32_t i = 0;
  while (i < self->schedule.held_count) {
    const QueryThreadId index = *array_get(&self->schedule.candidates, i);
    if (query_thread_store_get(&self->threads, index)->state == QueryThreadStateAbandoned) {
      query_vm_thread_free(self, index);
      query_schedule_remove_held(&self->schedule, i);
    } else {
      i++;
    }
  }
  query_vm_release_held(self, false);

  // The new matches that a waiting thread may still beat are held back, after
  // the ones that already are.
  uint32_t held_count = self->schedule.held_count;
  for (uint32_t j = self->schedule.held_count; j < self->schedule.candidates.size; j++) {
    const QueryThreadId index = *array_get(&self->schedule.candidates, j);
    if (query_thread_store_get(&self->threads, index)->state == QueryThreadStateAbandoned) {
      query_vm_thread_free(self, index);
      continue;
    }
    query_vm_drop_redundant(self, index);
    if (query_vm_has_longer(self, index)) {
      *array_get(&self->schedule.candidates, held_count) = index;
      held_count++;
    } else {
      array_push(&self->schedule.work, index);
    }
  }
  self->schedule.candidates.size = held_count;
  self->schedule.held_count = held_count;
  return query_vm_finish(self);
}

/*********
 * Events
 *********/

// The cursor calls the progress callback once it has done this much work.
// Entering or leaving a node costs one, plus one for each thread, because the
// cursor may visit all of them.
#define QUERY_VM_WORK_PER_CALLBACK_CHECK 1000

// Returns whether `a` reaches into `b`. An empty `a` reaches into a `b` that
// starts where it is.
static bool query_vm_range_intersects(const TSRange *a, const TSRange *b) {
  const bool is_empty = a->start_byte == a->end_byte;
  return (
    (a->end_byte > b->start_byte || (is_empty && a->end_byte == b->start_byte)) &&
    (point_gt(a->end_point, b->start_point) || (is_empty && point_eq(a->end_point, b->start_point))) &&
    a->start_byte < b->end_byte &&
    point_lt(a->start_point, b->end_point)
  );
}

// Returns whether `a` is inside `b`.
static bool query_vm_range_within(const TSRange *a, const TSRange *b) {
  return (
    a->start_byte >= b->start_byte &&
    point_gte(a->start_point, b->start_point) &&
    a->end_byte <= b->end_byte &&
    point_lte(a->end_point, b->end_point)
  );
}

static bool query_vm_is_pattern_disabled(const QueryVmProgram *query, uint32_t pattern) {
  return *array_get(&query->disabled_patterns, pattern);
}

// Starts a thread for `entry` on the current node, unless its pattern is
// disabled or a match from here cannot reach into the range.
static void query_vm_start(
  QueryVm *self,
  const QueryEntry *entry,
  bool node_intersects_range,
  bool can_start_unrooted
) {
  if (query_vm_is_pattern_disabled(self->query, entry->pattern)) return;
  if (!(entry->is_rooted ? node_intersects_range : can_start_unrooted)) return;
  const QueryThreadId index = query_thread_store_take(&self->threads);
  *query_thread_store_get(&self->threads, index) = (QueryThread) {
    .pc = query_entry_pc(*entry),
    .base = self->depth,
    .pattern = entry->pattern,
    .chain = QUERY_CAPTURE_CHAIN_EMPTY,
    .next = QUERY_THREAD_NONE,
    .intersects_range = entry->is_rooted || node_intersects_range,
    .state = QueryThreadStateLive,
  };
  query_vm_run(self, index);
}

// Returns whether a thread that waits at `pc` can skip the current node, to be
// given a later sibling of it instead.
static bool query_vm_can_skip(QueryVm *self, QueryProgramCounter pc) {
  const QueryInstruction wait = *array_get(
    &self->query->program.instructions,
    pc.index
  );
  const QueryOpcode opcode = query_instruction_opcode(wait);
  if (
    opcode != QueryOpcodeWait &&
    (opcode != QueryOpcodeWaitAnchored || ts_node_is_named(self->node.node))
  ) return false;
  // A thread that cannot be given a later sibling of this node, because there
  // is none, or none in the field that it needs, takes this node itself
  // instead of waiting in vain.
  query_vm_load_status(self);
  const TSFieldId field = query_instruction_wait_field(wait);
  return self->node.has_later_siblings &&
    !(field && field == self->node.field && !self->node.can_have_later_siblings_with_this_field);
}

// Enters the current node: starts the patterns that can start there, and gives
// it to the threads that are waiting for a node at its depth. Returns whether
// any match finished.
static bool query_vm_enter(
  QueryVm *self,
  TSNode node,
  const TSRange *node_range,
  bool node_intersects_range,
  bool parent_intersects_range,
  bool parent_is_error
) {
  const QueryVmProgram *query = self->query;
  const TSSymbol symbol = ts_node_symbol(node);
  self->node = (QueryVmNode) {
    .node = node,
    .symbol = symbol,
    .has_status = false,
  };
  const uint32_t depth = self->depth;

  // The threads already waiting at this depth get the node after the new ones
  // start, and stay ahead of every thread that stops here during this event.
  query_vm_begin_delivery(self, depth);

  if (depth <= self->max_start_depth) {
    // A pattern with several top-level nodes matches a sequence of siblings, so
    // it can start before the range and still reach into it, but not after it.
    const bool can_start_unrooted =
      parent_intersects_range && !parent_is_error &&
      node_range->start_byte < self->included_range.end_byte &&
      point_lt(node_range->start_point, self->included_range.end_point);
    const QueryProgram *program = &query->program;
    if (symbol == ts_builtin_sym_error) {
      const uint32_t group = query_program_entry_group(program, ts_builtin_sym_error);
      const uint32_t end = *array_get(&program->symbol_offsets, group + 1);
      for (uint32_t i = *array_get(&program->symbol_offsets, group); i < end; i++) {
        query_vm_start(self, array_get(&program->entries, i), node_intersects_range, can_start_unrooted);
      }
    } else {
      // The entries that check for no specific symbol are group zero, at the
      // start of `entries`, and every other symbol's group is the symbol itself.
      const uint32_t wildcard_end = *array_get(&program->symbol_offsets, 1);
      for (uint32_t i = 0; i < wildcard_end; i++) {
        query_vm_start(self, array_get(&program->entries, i), node_intersects_range, can_start_unrooted);
      }
      const uint32_t end = *array_get(&program->symbol_offsets, symbol + 1);
      for (uint32_t i = *array_get(&program->symbol_offsets, symbol); i < end; i++) {
        query_vm_start(self, array_get(&program->entries, i), node_intersects_range, can_start_unrooted);
      }
    }
  }

  for (;;) {
    const QueryThreadId index = query_vm_next_delivery(self);
    if (query_thread_id_is_none(index)) break;
    QueryThread *thread = query_thread_store_get(&self->threads, index);
    if (thread->state == QueryThreadStateAbandoned) {
      query_vm_thread_free(self, index);
      continue;
    }
    if (!query_vm_can_skip(self, thread->pc)) {
      thread->pc = query_program_counter_next(thread->pc);
      query_vm_receive(self, index, node_range);
      query_vm_run(self, index);
      continue;
    }
    // A copy takes the node, unless the copy's first tests would fail, and the
    // thread keeps waiting for a later one. The copy is made before the thread
    // stays, as no list may have a thread that is copied.
    const QueryProgramCounter next_pc = query_program_counter_next(thread->pc);
    const QueryInstruction first = *array_get(
      &query->program.instructions,
      next_pc.index
    );
    const TSFieldId field = query_instruction_wait_field(
      *array_get(&query->program.instructions, thread->pc.index)
    );
    query_vm_load_status(self);
    const bool first_tests_fail =
      (query_instruction_opcode(first) == QueryOpcodeSymbol && query_instruction_symbol(first) != symbol) ||
      (field && field != self->node.field);
    const QueryThreadId copy = first_tests_fail ? QUERY_THREAD_NONE : query_vm_copy(self, index, next_pc);
    query_vm_stay(self, index);
    if (query_thread_id_is_none(copy)) continue;
    query_vm_receive(self, copy, node_range);
    query_vm_run(self, copy);
  }

  query_vm_end_delivery(self, depth);
  return query_vm_settle(self);
}

// Leaves a node at `depth`: the threads waiting for a node inside it fail.
// Returns whether any match finished.
static bool query_vm_leave(QueryVm *self, uint32_t depth) {
  query_vm_free_waiting_at(self, depth + 1);
  // Leaving the node where the walk started ends every match.
  if (depth == 0) query_vm_free_waiting(self);
  query_vm_release_held(self, depth == 0);
  return query_vm_finish(self);
}

// Stops the walk: the matches that have not finished are dropped.
static void query_vm_halt(QueryVm *self) {
  query_vm_free_waiting(self);
  query_vm_free_held(self);
}

// Returns whether the walk goes into the current node, to start matches or to
// give its children to the threads that are waiting for them.
static inline bool query_vm_should_descend(QueryVm *self, bool node_intersects_range) {
  if (node_intersects_range && self->depth < self->max_start_depth) return true;
  if (query_schedule_waiting_at(&self->schedule, self->depth + 1)) return true;
  if (self->depth >= self->max_start_depth) return false;
  // A hidden node can contain the first node of a pattern with several
  // top-level nodes, whose later nodes are in the range. Of the repetitions,
  // which can have many children, only those that the analysis found can.
  if (!self->on_visible_node) {
    Subtree subtree = ts_tree_cursor_current_subtree(&self->cursor);
    if (ts_subtree_is_repetition(subtree)) {
      uint32_t index;
      bool exists;
      array_search_sorted_by(&self->query->program.repetition_symbols, , ts_subtree_symbol(subtree), &index, &exists);
      return exists;
    }
    return true;
  }
  return false;
}

// Walks the tree until a match finishes, or the walk ends. Returns whether a
// match finished.
static bool query_vm_advance(QueryVm *self) {
  self->has_first_unfinished = false;
  bool did_match = false;
  for (;;) {
    const bool is_halted = self->phase == QueryVmWalkPhaseHalted;
    if (is_halted) query_vm_halt(self);
    if (did_match || is_halted) return did_match;

    // Consult the progress callback after a bounded amount of work.
    self->work_count += 1 + self->threads.occupied_count;
    if (self->work_count >= QUERY_VM_WORK_PER_CALLBACK_CHECK) {
      self->work_count = 0;
      if (self->options && self->options->progress_callback) {
        self->progress_state.current_byte_offset = ts_node_start_byte(ts_tree_cursor_current_node(&self->cursor));
        if (self->options->progress_callback(&self->progress_state)) {
          self->phase = QueryVmWalkPhaseHalted;
          continue;
        }
      }
    }

    if (self->phase == QueryVmWalkPhaseLeaving) {
      if (self->on_visible_node && query_vm_leave(self, self->depth)) did_match = true;
      switch (ts_tree_cursor_goto_next_sibling_internal(&self->cursor)) {
        case TreeCursorStepVisible:
          if (!self->on_visible_node) {
            self->depth++;
            self->on_visible_node = true;
          }
          self->phase = QueryVmWalkPhaseEntering;
          break;
        case TreeCursorStepHidden:
          if (self->on_visible_node) {
            self->depth--;
            self->on_visible_node = false;
          }
          self->phase = QueryVmWalkPhaseEntering;
          break;
        default:
          // The parent is the nearest visible ancestor. A hidden node has the
          // depth of that ancestor, which must still be left.
          if (ts_tree_cursor_goto_parent(&self->cursor)) {
            if (self->on_visible_node) self->depth--;
            self->on_visible_node = true;
          } else {
            self->phase = QueryVmWalkPhaseHalted;
          }
      }
    } else {
      TSNode node = ts_tree_cursor_current_node(&self->cursor);
      TSNode parent_node = ts_tree_cursor_parent_node(&self->cursor);
      const bool parent_intersects_range =
        ts_node_is_null(parent_node) ||
        query_vm_range_intersects(&(TSRange) {
          .start_point = ts_node_start_point(parent_node),
          .end_point = ts_node_end_point(parent_node),
          .start_byte = ts_node_start_byte(parent_node),
          .end_byte = ts_node_end_byte(parent_node),
        }, &self->included_range);
      const TSRange node_range = {
        .start_point = ts_node_start_point(node),
        .end_point = ts_node_end_point(node),
        .start_byte = ts_node_start_byte(node),
        .end_byte = ts_node_end_byte(node),
      };
      const bool node_intersects_range =
        parent_intersects_range && query_vm_range_intersects(&node_range, &self->included_range);
      const bool node_intersects_containing_range = query_vm_range_intersects(&node_range, &self->containing_range);
      const bool node_within_containing_range = query_vm_range_within(&node_range, &self->containing_range);
      if (node_within_containing_range && self->on_visible_node) {
        const bool parent_is_error =
          !ts_node_is_null(parent_node) &&
          ts_node_symbol(parent_node) == ts_builtin_sym_error;
        if (query_vm_enter(self, node, &node_range, node_intersects_range, parent_intersects_range, parent_is_error)) {
          did_match = true;
        }
      }

      if (node_intersects_containing_range && query_vm_should_descend(self, node_intersects_range)) {
        switch (ts_tree_cursor_goto_first_child_internal(&self->cursor)) {
          case TreeCursorStepVisible:
            self->depth++;
            self->on_visible_node = true;
            continue;
          case TreeCursorStepHidden:
            self->on_visible_node = false;
            continue;
          default:
            break;
        }
      }
      self->phase = QueryVmWalkPhaseLeaving;
    }
  }
}

/******
 * API
 ******/

void ts_query_vm_new(QueryVm *self) {
  *self = (QueryVm) {
    .threads = query_thread_store_new(),
    .captures = query_capture_store_new(),
    .schedule = query_schedule_new(),
    .matches = query_match_store_new(),
    .included_range = {.start_point = {0, 0}, .end_point = POINT_MAX, .start_byte = 0, .end_byte = UINT32_MAX},
    .containing_range = {.start_point = {0, 0}, .end_point = POINT_MAX, .start_byte = 0, .end_byte = UINT32_MAX},
    .max_start_depth = UINT32_MAX,
    .match_limit = UINT32_MAX,
  };
}

void ts_query_vm_delete(QueryVm *self) {
  query_thread_store_delete(&self->threads);
  query_capture_store_delete(&self->captures);
  query_match_store_delete(&self->matches);
  query_schedule_delete(&self->schedule);
  ts_tree_cursor_delete(&self->cursor);
}

void ts_query_vm_exec(QueryVm *self, const QueryVmProgram *query, TSNode node, const TSQueryCursorOptions *options) {
  self->query = query;
  query_thread_store_reset(&self->threads);
  query_capture_store_reset(&self->captures, node.tree);
  query_schedule_reset(&self->schedule, query->program.patterns.size);
  query_match_store_reset(&self->matches);
  self->has_first_unfinished = false;
  self->capture_holder_count = 0;

  ts_tree_cursor_reset(&self->cursor, node);
  self->depth = 0;
  self->on_visible_node = true;
  self->phase = QueryVmWalkPhaseEntering;
  self->did_exceed_match_limit = false;
  self->options = options;
  self->progress_state = (TSQueryCursorState) {.payload = options ? options->payload : NULL};
  self->work_count = 0;
}

// Orders the finished matches at `a` and `b` in the heap: by where each one's
// next capture starts, then pattern, then the order they finished in.
static bool query_vm_finished_precedes(const QueryVm *self, uint32_t a, uint32_t b) {
  const QueryFinishedMatch *left = array_get(&self->matches.finished, a);
  const QueryFinishedMatch *right = array_get(&self->matches.finished, b);
  uint32_t left_byte, right_byte;
  if (!query_vm_finished_next_byte(left, &left_byte)) return false;
  if (!query_vm_finished_next_byte(right, &right_byte)) return true;
  if (left_byte != right_byte) return left_byte < right_byte;
  if (left->pattern != right->pattern) return left->pattern < right->pattern;
  return left->id < right->id;
}

static void query_vm_heap_swap(QueryVm *self, uint32_t a, uint32_t b) {
  QueryFinishedMatch tmp = *array_get(&self->matches.finished, a);
  *array_get(&self->matches.finished, a) = *array_get(&self->matches.finished, b);
  *array_get(&self->matches.finished, b) = tmp;
}

static void query_vm_sift_down(QueryVm *self, uint32_t index) {
  for (;;) {
    uint32_t smallest = index;
    const uint32_t left = 2 * index + 1, right = 2 * index + 2;
    if (left < self->matches.finished.size && query_vm_finished_precedes(self, left, smallest)) smallest = left;
    if (right < self->matches.finished.size && query_vm_finished_precedes(self, right, smallest)) smallest = right;
    if (smallest == index) return;
    query_vm_heap_swap(self, index, smallest);
    index = smallest;
  }
}

static void query_vm_sift_up(QueryVm *self, uint32_t index) {
  while (index > 0) {
    const uint32_t parent = (index - 1) / 2;
    if (!query_vm_finished_precedes(self, index, parent)) return;
    query_vm_heap_swap(self, index, parent);
    index = parent;
  }
}

// Adds the finished matches past the heap's end to the heap.
static void query_vm_heapify(QueryVm *self) {
  while (self->matches.heap_size < self->matches.finished.size) {
    query_vm_sift_up(self, self->matches.heap_size);
    self->matches.heap_size++;
  }
}

// Removes the finished match at `index`, keeping the rest in order: in the
// order that they finished, or as a heap if they are one.
static void query_vm_erase_finished(QueryVm *self, uint32_t index) {
  if (self->matches.heap_size == 0) {
    array_erase(&self->matches.finished, index);
    return;
  }
  // The matches are a heap at this point, so every one must be inside it.
  ts_assert(self->matches.heap_size == self->matches.finished.size);
  const uint32_t last = self->matches.finished.size - 1;
  if (index != last) {
    *array_get(&self->matches.finished, index) = *array_get(&self->matches.finished, last);
  }
  self->matches.finished.size--;
  self->matches.heap_size = self->matches.finished.size;
  if (index < self->matches.finished.size) {
    if (index > 0 && query_vm_finished_precedes(self, index, (index - 1) / 2)) {
      query_vm_sift_up(self, index);
    } else {
      query_vm_sift_down(self, index);
    }
  }
}

bool ts_query_vm_next_match(QueryVm *self, TSQueryMatch *match) {
  if (self->matches.finished.size == 0) query_vm_advance(self);
  if (self->matches.finished.size == 0) return false;
  // The first match to finish is at the front, unless the matches are a heap.
  uint32_t index = 0;
  if (self->matches.heap_size > 0) {
    query_vm_heapify(self);
    for (uint32_t i = 1; i < self->matches.finished.size; i++) {
      if (
        array_get(&self->matches.finished, i)->id <
        array_get(&self->matches.finished, index)->id
      ) index = i;
    }
  }
  QueryFinishedMatch *finished = array_get(&self->matches.finished, index);
  query_vm_materialize(self, finished);
  const QueryCaptureList *list = query_vm_match_captures(self, finished);
  match->id = finished->id;
  match->pattern_index = finished->pattern;
  match->captures = list->contents;
  match->capture_count = (uint16_t)list->size;
  // The list's contents stay where they are until the list is used again.
  query_vm_finished_release(self, finished);
  query_vm_erase_finished(self, index);
  // The list is reused only by a later finished match, which reaches it after
  // this match has been read.
  return true;
}

void ts_query_vm_remove_match(QueryVm *self, uint32_t match_id) {
  if (self->matches.heap_size > 0) query_vm_heapify(self);
  for (uint32_t i = 0; i < self->matches.finished.size; i++) {
    QueryFinishedMatch *finished = array_get(&self->matches.finished, i);
    if (finished->id == match_id) {
      query_vm_finished_release(self, finished);
      query_vm_erase_finished(self, i);
      return;
    }
  }
  // An unfinished match has no id: `next_capture` only returns the captures of
  // finished ones.
}

bool ts_query_vm_next_capture(QueryVm *self, TSQueryMatch *match, uint32_t *capture_index) {
  for (;;) {
    query_vm_heapify(self);

    if (!self->has_first_unfinished) {
      self->first_unfinished = query_vm_first_unfinished(self);
      self->has_first_unfinished = true;
    }
    const QueryThreadId unfinished = self->first_unfinished;
    uint32_t unfinished_byte = UINT32_MAX;
    uint32_t unfinished_pattern = UINT32_MAX;
    if (!query_thread_id_is_none(unfinished)) {
      const QueryThread *thread = query_thread_store_get(&self->threads, unfinished);
      unfinished_byte = thread->first_start_byte;
      unfinished_pattern = thread->pattern;
    }

    QueryFinishedMatch *first = NULL;
    while (self->matches.finished.size > 0) {
      QueryFinishedMatch *finished = array_get(&self->matches.finished, 0);
      query_vm_materialize(self, finished);
      const QueryCaptureList *list = query_vm_match_captures(self, finished);
      if (finished->consumed_capture_count >= list->size) {
        query_vm_finished_release(self, finished);
        query_vm_erase_finished(self, 0);
        continue;
      }
      TSNode node = array_get(list, finished->consumed_capture_count)->node;
      const TSRange node_range = {
        .start_point = ts_node_start_point(node),
        .end_point = ts_node_end_point(node),
        .start_byte = ts_node_start_byte(node),
        .end_byte = ts_node_end_byte(node),
      };
      // A capture outside the range is skipped, though its match is returned.
      if (!query_vm_range_intersects(&node_range, &self->included_range)) {
        query_vm_finished_consumed(self, finished);
        query_vm_sift_down(self, 0);
        continue;
      }
      const uint32_t node_start_byte = node_range.start_byte;
      if (
        node_start_byte < unfinished_byte ||
        (node_start_byte == unfinished_byte && finished->pattern < unfinished_pattern)
      ) first = finished;
      break;
    }

    if (first) {
      const QueryCaptureList *list = query_vm_match_captures(self, first);
      match->id = first->id;
      match->pattern_index = first->pattern;
      match->captures = list->contents;
      match->capture_count = (uint16_t)list->size;
      *capture_index = first->consumed_capture_count;
      query_vm_finished_consumed(self, first);
      query_vm_sift_down(self, 0);
      return true;
    }

    if (!query_thread_id_is_none(unfinished) && self->capture_holder_count >= self->match_limit) {
      self->did_exceed_match_limit = true;
      query_vm_evict(self, unfinished);
    }

    if (!query_vm_advance(self) && self->matches.finished.size == 0) return false;
  }
}
