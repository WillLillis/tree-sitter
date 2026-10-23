#ifndef TREE_SITTER_QUERY_SCHEDULE_H_
#define TREE_SITTER_QUERY_SCHEDULE_H_

#include "./array.h"
#include "./query_thread_store.h"

// The threads that are waiting for the next node at one depth.
typedef struct {
  uint32_t depth;
  QueryThreadList threads;
} QueryWaitingBucket;

// Which threads are to run next, which are waiting, and which have finished a
// match that is not settled yet.
typedef struct {
  // The depths where threads are waiting, in order.
  Array(QueryWaitingBucket) waiting;
  // While the current node is given to the threads that were waiting for it,
  // the ones that have not had it yet, and the ones that skipped it and keep
  // waiting.
  QueryThreadList delivering;
  QueryThreadList staying;
  // The threads that must still run on the current node, and then the threads
  // whose matches are being finished.
  Array(QueryThreadId) work;
  // The threads that stopped at a wait while running on the current node.
  Array(QueryThreadId) arrivals;
  // The threads that finished a match: first the `held_count` whose matches are
  // held back from earlier nodes, because a thread with the same pattern and
  // start may still finish a longer one, and then the ones from the current
  // node.
  Array(QueryThreadId) candidates;
  uint32_t held_count;
  // For each pattern, how many of its threads are in waiting lists, including
  // abandoned threads that a list still has.
  Array(uint32_t) waiting_counts;
} QuerySchedule;

static inline QuerySchedule query_schedule_new(void) {
  return (QuerySchedule) {
    .waiting = array_new(),
    .delivering = QUERY_THREAD_LIST_EMPTY,
    .staying = QUERY_THREAD_LIST_EMPTY,
    .work = array_new(),
    .arrivals = array_new(),
    .candidates = array_new(),
    .waiting_counts = array_new(),
  };
}

static inline void query_schedule_delete(QuerySchedule *self) {
  array_delete(&self->waiting);
  self->delivering = QUERY_THREAD_LIST_EMPTY;
  self->staying = QUERY_THREAD_LIST_EMPTY;
  array_delete(&self->work);
  array_delete(&self->arrivals);
  array_delete(&self->candidates);
  self->held_count = 0;
  array_delete(&self->waiting_counts);
}

// Empties the schedule, for a new walk with a query of `pattern_count`
// patterns.
static inline void query_schedule_reset(QuerySchedule *self, uint32_t pattern_count) {
  array_clear(&self->waiting);
  self->delivering = QUERY_THREAD_LIST_EMPTY;
  self->staying = QUERY_THREAD_LIST_EMPTY;
  array_clear(&self->work);
  array_clear(&self->arrivals);
  array_clear(&self->candidates);
  self->held_count = 0;
  array_clear(&self->waiting_counts);
  array_grow_by(&self->waiting_counts, pattern_count);
}

// Returns the position of the first bucket of threads that are waiting at
// `depth` or deeper.
static inline uint32_t query_schedule_bucket_from(const QuerySchedule *self, uint32_t depth) {
  uint32_t low = 0;
  uint32_t high = self->waiting.size;
  while (low < high) {
    const uint32_t middle = low + (high - low) / 2;
    if (array_get(&self->waiting, middle)->depth < depth) {
      low = middle + 1;
    } else {
      high = middle;
    }
  }
  return low;
}

// Returns the threads that are waiting at `depth`, or NULL if none are.
static inline QueryThreadList *query_schedule_waiting_at(QuerySchedule *self, uint32_t depth) {
  const uint32_t index = query_schedule_bucket_from(self, depth);
  if (index == self->waiting.size || array_get(&self->waiting, index)->depth != depth) return NULL;
  return &array_get(&self->waiting, index)->threads;
}

// Returns the threads that are waiting at `depth`, after adding a bucket for
// them if there is none, which moves the buckets after it.
static inline QueryThreadList *query_schedule_waiting_at_or_new(QuerySchedule *self, uint32_t depth) {
  const uint32_t index = query_schedule_bucket_from(self, depth);
  if (index == self->waiting.size || array_get(&self->waiting, index)->depth != depth) {
    array_insert(&self->waiting, index, ((QueryWaitingBucket) {.depth = depth, .threads = QUERY_THREAD_LIST_EMPTY}));
  }
  return &array_get(&self->waiting, index)->threads;
}

// Removes the bucket of the threads that are waiting at `depth`, and returns
// them.
static inline QueryThreadList query_schedule_take_waiting(QuerySchedule *self, uint32_t depth) {
  const uint32_t index = query_schedule_bucket_from(self, depth);
  if (index == self->waiting.size || array_get(&self->waiting, index)->depth != depth) {
    return QUERY_THREAD_LIST_EMPTY;
  }
  const QueryThreadList threads = array_get(&self->waiting, index)->threads;
  array_erase(&self->waiting, index);
  return threads;
}

// Returns how many threads of `pattern` are in waiting lists.
static inline uint32_t query_schedule_waiting_count(const QuerySchedule *self, uint32_t pattern) {
  return *array_get(&self->waiting_counts, pattern);
}

// Removes the held candidate at `i`, and keeps the rest in order.
static inline void query_schedule_remove_held(QuerySchedule *self, uint32_t i) {
  ts_assert(i < self->held_count);
  array_erase(&self->candidates, i);
  self->held_count--;
}

#endif  // TREE_SITTER_QUERY_SCHEDULE_H_
