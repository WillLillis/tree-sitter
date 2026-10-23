#ifndef TREE_SITTER_QUERY_THREAD_STORE_H_
#define TREE_SITTER_QUERY_THREAD_STORE_H_

#include "./array.h"
#include "./query_capture_store.h"

// A thread, by the slot that it keeps in the thread store for as long as it
// lives.
typedef struct {
  uint32_t slot;
} QueryThreadId;

// No thread.
#define QUERY_THREAD_NONE ((QueryThreadId) {UINT32_MAX})

static inline bool query_thread_id_is_none(QueryThreadId self) {
  return self.slot == UINT32_MAX;
}

static inline bool query_thread_id_eq(QueryThreadId a, QueryThreadId b) {
  return a.slot == b.slot;
}

// Where a thread's slot is in its life.
typedef enum {
  // The thread runs, or waits to run.
  QueryThreadStateLive,
  // The thread was abandoned, and has released its captures, but a list still
  // has it. The list frees it when it next reaches it.
  QueryThreadStateAbandoned,
  // The slot is free, and in the store's free list.
  QueryThreadStateFree,
} QueryThreadState;

// One way that one match of a pattern can go on.
typedef struct {
  // The next instruction to run, or the wait that the thread is stopped at.
  uint32_t pc;
  // The depth of the node where the match started.
  uint32_t base;
  // The match's captures.
  QueryHistory history;
  // Where the match's first capture starts, once it has one.
  uint32_t first_start_byte;
  // The next thread in the list that this thread is in, or in the free list.
  QueryThreadId next;
  // The match's id, once the API has shown the match, or NONE.
  uint32_t match_id;
  uint16_t pattern;
  // Whether one of the match's top-level nodes reaches into the cursor's range.
  bool intersects_range : 1;
  // A QueryThreadState.
  uint8_t state : 2;
} QueryThread;

// The threads of a cursor, in slots that keep their index for as long as the
// thread lives, so that lists can refer to them.
typedef struct {
  Array(QueryThread) slots;
  // The first free slot, or NONE.
  QueryThreadId free_head;
  // The number of slots that are not free, including abandoned threads that a
  // list still has. The cursor charges this much work for each node, to know
  // when to call the progress callback.
  uint32_t occupied_count;
} QueryThreadStore;

// Threads linked through their `next` field, oldest first.
typedef struct {
  QueryThreadId first;
  QueryThreadId last;
} QueryThreadList;

#define QUERY_THREAD_LIST_EMPTY ((QueryThreadList) {.first = QUERY_THREAD_NONE, .last = QUERY_THREAD_NONE})

static inline QueryThreadStore query_thread_store_new(void) {
  return (QueryThreadStore) {.slots = array_new(), .free_head = QUERY_THREAD_NONE};
}

static inline void query_thread_store_delete(QueryThreadStore *self) {
  array_delete(&self->slots);
  self->free_head = QUERY_THREAD_NONE;
}

// Frees every slot at once, for a new walk.
static inline void query_thread_store_reset(QueryThreadStore *self) {
  array_clear(&self->slots);
  self->free_head = QUERY_THREAD_NONE;
  self->occupied_count = 0;
}

static inline QueryThread *query_thread_store_get(QueryThreadStore *self, QueryThreadId index) {
  return array_get(&self->slots, index.slot);
}

// Returns a slot for a new thread: a free one, if there is one, or a new one,
// which can move every thread.
static inline QueryThreadId query_thread_store_take(QueryThreadStore *self) {
  uint32_t index;
  if (!query_thread_id_is_none(self->free_head)) {
    index = self->free_head.slot;
    self->free_head = array_get(&self->slots, index)->next;
  } else {
    index = self->slots.size;
    array_push(&self->slots, ((QueryThread) {0}));
  }
  self->occupied_count++;
  return (QueryThreadId) {index};
}

// Puts the slot of the thread `index`, which no list has anymore, in the free
// list.
static inline void query_thread_store_free(QueryThreadStore *self, QueryThreadId index) {
  QueryThread *thread = query_thread_store_get(self, index);
  ts_assert(thread->state != QueryThreadStateFree);
  thread->state = QueryThreadStateFree;
  thread->next = self->free_head;
  self->free_head = index;
  self->occupied_count--;
}

// Appends the thread `index` to the list.
static inline void query_thread_list_append(QueryThreadList *self, QueryThreadStore *threads, QueryThreadId index) {
  query_thread_store_get(threads, index)->next = QUERY_THREAD_NONE;
  if (query_thread_id_is_none(self->last)) {
    self->first = index;
  } else {
    query_thread_store_get(threads, self->last)->next = index;
  }
  self->last = index;
}

#endif  // TREE_SITTER_QUERY_THREAD_STORE_H_
