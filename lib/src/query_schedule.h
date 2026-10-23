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
  // Waiting buckets, sorted by target depth. Each bucket holds one queue.
  //
  // Before processing a node at depth <d>, take that depth's existing queue
  // into `delivering`. New waits go into `waiting` for later nodes, outside
  // the _current_ delivery pass. Threads that skip this node move into
  // staying. At the end, prepend staying to the waiting queue at <d>.
  //
  // Example: A and B were waiting at d. A skips this node, B finishes its
  // match, and a new thread C starts waiting at d while processing the node.
  //
  //   phase             waiting at d  delivering  staying
  //   before delivery   A -> B        -           -
  //   begin delivery    -             A -> B      -
  //   new wait C        C             A -> B      -
  //   A skips           C             B           A
  //   B finishes        C             -           A
  //   end delivery      A -> C        -           -
  Array(QueryWaitingBucket) waiting;
  QueryThreadList delivering;
  QueryThreadList staying;

  // One thread and the copies it splits off while running on the current node.
  // Cleared after that thread and all of its copies have run.
  //
  // Also holds accepted threads ready to become finished matches.
  // Cleared after their captures are transferred and their slots are freed.
  Array(QueryThreadId) work;
  // Threads that reached a wait while running on the current node.
  // They are already linked into waiting and are checked for redundancy
  // after delivery finishes.
  Array(QueryThreadId) arrivals;

  // Threads that reached `Accept`. The first `held_count` entries are retained
  // from earlier nodes while a competing thread may produce a longer match.
  // The remaining entries were accepted on the current node.
  Array(QueryThreadId) candidates;
  uint32_t held_count;

  // For each pattern, how many threads are in `waiting`, `delivering`, or
  // `staying`. Abandoned threads remain counted until removed from their list.
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
static inline const QueryThreadList *query_schedule_waiting_at(const QuerySchedule *self, uint32_t depth) {
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
