#ifndef TREE_SITTER_ARRAY_H_
#define TREE_SITTER_ARRAY_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "./alloc.h"
#include "./ts_assert.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#ifdef _MSC_VER
#pragma warning(push)
#pragma warning(disable : 4101)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

#define Array(T)       \
  struct {             \
    T *contents;       \
    uint32_t size;     \
    uint32_t capacity; \
  }

/// Initialize an array.
#define array_init(self) \
  ((self)->size = 0, (self)->capacity = 0, (self)->contents = NULL)

/// Create an empty array.
#define array_new() \
  { NULL, 0, 0 }

/// Get a pointer to the element at a given `index` in the array.
#define array_get(self, _index) \
  (ts_assert((uint32_t)(_index) < (self)->size), &(self)->contents[_index])

/// Get a pointer to the first element in the array.
#define array_front(self) array_get(self, 0)

/// Get a pointer to the last element in the array.
#define array_back(self) array_get(self, (self)->size - 1)

/// Clear the array, setting its size to zero. Note that this does not free any
/// memory allocated for the array's contents.
#define array_clear(self) ((self)->size = 0)

#ifdef __cplusplus
#define _array__cast(self, expr) (decltype((self)->contents))(expr)
#else
#define _array__cast(self, expr) (expr)
#endif

/// Reserve `new_capacity` elements of space in the array. If `new_capacity` is
/// less than the array's current capacity, this function has no effect.
#define array_reserve(self, new_capacity)                 \
  ((self)->contents = _array__cast(self, _array__reserve( \
    (void *)(self)->contents, &(self)->capacity,          \
    array_elem_size(self), new_capacity))                 \
  )

/// Free any memory allocated for this array. Note that this does not free any
/// memory allocated for the array's contents.
#define array_delete(self)                           \
  do {                                               \
    if ((self)->contents) ts_free((self)->contents); \
    (self)->contents = NULL;                         \
    (self)->size = 0;                                \
    (self)->capacity = 0;                            \
  } while (0)

/// Push a new `element` onto the end of the array.
#define array_push(self, element)                                 \
  do {                                                            \
    (self)->contents = _array__cast(self, _array__grow(           \
      (void *)(self)->contents, (self)->size, &(self)->capacity,  \
      1, array_elem_size(self)                                    \
    ));                                                           \
   (self)->contents[(self)->size++] = (element);                  \
  } while(0)

/// Increase the array's size by `count` elements.
/// New elements are zero-initialized.
#define array_grow_by(self, count)                                               \
  do {                                                                           \
    if ((count) == 0) break;                                                     \
    (self)->contents = _array__cast(self, _array__grow(                          \
      (self)->contents, (self)->size, &(self)->capacity,                         \
      count, array_elem_size(self)                                               \
    ));                                                                          \
    memset((self)->contents + (self)->size, 0, (count) * array_elem_size(self)); \
    (self)->size += (count);                                                     \
  } while (0)

/// Append all elements from one array to the end of another.
#define array_push_all(self, other) \
  array_extend((self), (other)->size, (other)->contents)

/// Append `count` elements to the end of the array, reading their values from the
/// `contents` pointer.
#define array_extend(self, count, other_contents)                 \
  ((self)->contents = _array__cast(self, _array__splice(          \
    (void*)(self)->contents, &(self)->size, &(self)->capacity,    \
    array_elem_size(self), (self)->size, 0, count, other_contents \
  )))

/// Remove `old_count` elements from the array starting at the given `index`. At
/// the same index, insert `new_count` new elements, reading their values from the
/// `new_contents` pointer.
#define array_splice(self, _index, old_count, new_count, new_contents) \
  ((self)->contents = _array__cast(self, _array__splice(              \
    (void *)(self)->contents, &(self)->size, &(self)->capacity,        \
    array_elem_size(self), _index, old_count, new_count, new_contents  \
  )))

/// Insert one `element` into the array at the given `index`.
#define array_insert(self, _index, element)                     \
  ((self)->contents = _array__cast(self, _array__splice(        \
    (void *)(self)->contents, &(self)->size, &(self)->capacity, \
    array_elem_size(self), _index, 0, 1, &(element)             \
  )))

/// Remove one element from the array at the given `index`.
#define array_erase(self, _index) \
  _array__erase((void *)(self)->contents, &(self)->size, array_elem_size(self), _index)

/// Pop the last element off the array, returning the element by value.
#define array_pop(self) ((self)->contents[--(self)->size])

/// Assign the contents of one array to another, reallocating if necessary.
#define array_assign(self, other)                                   \
  ((self)->contents = _array__cast(self, _array__assign(            \
    (void *)(self)->contents, &(self)->size, &(self)->capacity,     \
    (const void *)(other)->contents, (other)->size, array_elem_size(self) \
  )))

/// Swap one array with another
#define array_swap(self, other)                                     \
  do {                                                              \
    void *_array_swap_tmp = (void *)(self)->contents;               \
    (self)->contents = (other)->contents;                           \
    (other)->contents = _array__cast(other, _array_swap_tmp);       \
    _array__swap(&(self)->size, &(self)->capacity,                  \
                 &(other)->size, &(other)->capacity);               \
  } while (0)

/// Get the size of the array contents
#define array_elem_size(self) (sizeof *(self)->contents)

/// Sort the array in place using `int compare(const void *, const void *)`.
/// The comparator receives pointers to two elements and returns a negative
/// value if the first comes before the second, zero if they compare equal,
/// or a positive value if the first comes after the second.
/// Equal elements may change order.
#define array_sort(self, compare) \
  _array__sort((void *)(self)->contents, (self)->size, array_elem_size(self), compare)

/// Search a sorted array for a given `needle` value, using the given `compare`
/// callback to determine the order.
///
/// If an existing element is found to be equal to `needle`, then the `index`
/// out-parameter is set to the existing value's index, and the `exists`
/// out-parameter is set to true. Otherwise, `index` is set to an index where
/// `needle` should be inserted in order to preserve the sorting, and `exists`
/// is set to false.
#define array_search_sorted_with(self, compare, needle, _index, _exists) \
  _array__search_sorted(self, 0, compare, , needle, _index, _exists)

/// Search a sorted array for a given `needle` value, using integer comparisons
/// of a given struct field (specified with a leading dot) to determine the order.
///
/// See also `array_search_sorted_with`.
#define array_search_sorted_by(self, field, needle, _index, _exists) \
  _array__search_sorted(self, 0, _compare_int, field, needle, _index, _exists)

/// Insert a given `value` into a sorted array, using the given `compare`
/// callback to determine the order.
#define array_insert_sorted_with(self, compare, value) \
  do { \
    unsigned _index, _exists; \
    array_search_sorted_with(self, compare, &(value), &_index, &_exists); \
    if (!_exists) array_insert(self, _index, value); \
  } while (0)

/// Insert a given `value` into a sorted array, using integer comparisons of
/// a given struct field (specified with a leading dot) to determine the order.
///
/// See also `array_search_sorted_by`.
#define array_insert_sorted_by(self, field, value) \
  do { \
    unsigned _index, _exists; \
    array_search_sorted_by(self, field, (value) field, &_index, &_exists); \
    if (!_exists) array_insert(self, _index, value); \
  } while (0)

// Private

// Pointers to individual `Array` fields (rather than the entire `Array` itself)
// are passed to the various `_array__*` functions below to address strict aliasing
// violations that arises when the _entire_ `Array` struct is passed as `Array(void)*`.
//
// The `Array` type itself was not altered as a solution in order to avoid breakage
// with existing consumers (in particular, parsers with external scanners).

/// This is not what you're looking for, see `array_erase`.
static inline void _array__erase(void* self_contents, uint32_t *size,
                                size_t element_size, uint32_t index) {
  ts_assert(index < *size);
  char *contents = (char *)self_contents;
  memmove(contents + index * element_size, contents + (index + 1) * element_size,
          (*size - index - 1) * element_size);
  (*size)--;
}

/// This is not what you're looking for, see `array_reserve`.
static inline void *_array__reserve(void *contents, uint32_t *capacity,
                                  size_t element_size, uint32_t new_capacity) {
  void *new_contents = contents;
  if (new_capacity > *capacity) {
    if (contents) {
      new_contents = ts_realloc(contents, new_capacity * element_size);
    } else {
      new_contents = ts_malloc(new_capacity * element_size);
    }
    *capacity = new_capacity;
  }
  return new_contents;
}

/// This is not what you're looking for, see `array_assign`.
static inline void *_array__assign(void* self_contents, uint32_t *self_size, uint32_t *self_capacity,
                                 const void *other_contents, uint32_t other_size, size_t element_size) {
  void *new_contents = _array__reserve(self_contents, self_capacity, element_size, other_size);
  *self_size = other_size;
  memcpy(new_contents, other_contents, *self_size * element_size);
  return new_contents;
}

/// This is not what you're looking for, see `array_swap`.
static inline void _array__swap(uint32_t *self_size, uint32_t *self_capacity,
                               uint32_t *other_size, uint32_t *other_capacity) {
  uint32_t tmp_size = *self_size;
  uint32_t tmp_capacity = *self_capacity;
  *self_size = *other_size;
  *self_capacity = *other_capacity;
  *other_size = tmp_size;
  *other_capacity = tmp_capacity;
}

/// This is not what you're looking for, see `array_push` or `array_grow_by`.
static inline void *_array__grow(void *contents, uint32_t size, uint32_t *capacity,
                               uint32_t count, size_t element_size) {
  void *new_contents = contents;
  uint32_t new_size = size + count;
  if (new_size > *capacity) {
    uint32_t new_capacity = *capacity * 2;
    if (new_capacity < 8) new_capacity = 8;
    if (new_capacity < new_size) new_capacity = new_size;
    new_contents = _array__reserve(contents, capacity, element_size, new_capacity);
  }
  return new_contents;
}

/// This is not what you're looking for, see `array_splice`.
static inline void *_array__splice(void *self_contents, uint32_t *size, uint32_t *capacity,
                                 size_t element_size,
                                 uint32_t index, uint32_t old_count,
                                 uint32_t new_count, const void *elements) {
  uint32_t new_size = *size + new_count - old_count;
  uint32_t old_end = index + old_count;
  uint32_t new_end = index + new_count;
  ts_assert(old_end <= *size);

  void *new_contents = _array__reserve(self_contents, capacity, element_size, new_size);

  char *contents = (char *)new_contents;
  if (*size > old_end) {
    memmove(
      contents + new_end * element_size,
      contents + old_end * element_size,
      (*size - old_end) * element_size
    );
  }
  if (new_count > 0) {
    if (elements) {
      memcpy(
        (contents + index * element_size),
        elements,
        new_count * element_size
      );
    } else {
      memset(
        (contents + index * element_size),
        0,
        new_count * element_size
      );
    }
  }
  *size += new_count - old_count;

  return new_contents;
}

// Three-way quicksort based on Bentley and McIlroy's "Engineering a Sort
// Function" (1993), with Musser's introsort depth limit and heapsort fallback.

// Swaps up to 16 bytes between regions that are disjoint or identical.
static inline void _array__sort_swap_chunk(
  unsigned char *left,
  unsigned char *right,
  size_t length
) {
  unsigned char left_chunk[16];
  unsigned char right_chunk[16];
  memcpy(left_chunk, left, length);
  memcpy(right_chunk, right, length);
  memcpy(left, right_chunk, length);
  memcpy(right, left_chunk, length);
}

// Swaps two elements of `element_size` bytes.
static inline void _array__sort_swap(
  unsigned char *left,
  unsigned char *right,
  size_t element_size
) {
  // Fixed chunk sizes encourage inlining of the `memcpy` calls.
  size_t offset = 0;
  for (; element_size - offset >= 16; offset += 16) {
    _array__sort_swap_chunk(left + offset, right + offset, 16);
  }
  if (element_size - offset >= 8) {
    _array__sort_swap_chunk(left + offset, right + offset, 8);
    offset += 8;
  }
  if (element_size - offset >= 4) {
    _array__sort_swap_chunk(left + offset, right + offset, 4);
    offset += 4;
  }
  if (element_size - offset >= 2) {
    _array__sort_swap_chunk(left + offset, right + offset, 2);
    offset += 2;
  }
  if (element_size - offset >= 1) {
    _array__sort_swap_chunk(left + offset, right + offset, 1);
  }
}

// Swaps two disjoint runs of `count` elements each.
static inline void _array__sort_swap_run(
  unsigned char *left,
  unsigned char *right,
  size_t count,
  size_t element_size
) {
  for (size_t i = 0; i < count; i++) {
    _array__sort_swap(
      left + i * element_size,
      right + i * element_size,
      element_size
    );
  }
}

// Returns the pointer to the median element according to `compare`.
static inline unsigned char *_array__sort_median(
  unsigned char *a,
  unsigned char *b,
  unsigned char *c,
  int (*compare)(const void *, const void *)
) {
  if (compare(a, b) < 0) {
    if (compare(b, c) < 0) return b;
    return compare(a, c) < 0 ? c : a;
  }
  if (compare(b, c) > 0) return b;
  return compare(a, c) < 0 ? a : c;
}

// Sorts `count` elements in place using insertion sort.
static inline void _array__sort_insertion(
  unsigned char *contents,
  size_t count,
  size_t element_size,
  int (*compare)(const void *, const void *)
) {
  for (size_t i = 1; i < count; i++) {
    for (
      unsigned char *element = contents + i * element_size;
      element > contents && compare(element - element_size, element) > 0;
      element -= element_size
    ) {
      _array__sort_swap(element - element_size, element, element_size);
    }
  }
}

// Restores heap order at `root`. Its child subtrees must already be max-heaps.
static inline void _array__sort_sift(
  unsigned char *contents,
  size_t root,
  size_t count,
  size_t element_size,
  int (*compare)(const void *, const void *)
) {
  while (root < count / 2) {
    size_t child = root * 2 + 1;
    if (
      child + 1 < count &&
      compare(
        contents + child * element_size,
        contents + (child + 1) * element_size
      ) < 0
    ) {
      child++;
    }
    if (
      compare(
        contents + root * element_size,
        contents + child * element_size
      ) >= 0
    ) return;

    _array__sort_swap(
      contents + root * element_size,
      contents + child * element_size,
      element_size
    );
    root = child;
  }
}

// Sorts `count` elements in place using heapsort.
static inline void _array__sort_heap(
  unsigned char *contents,
  size_t count,
  size_t element_size,
  int (*compare)(const void *, const void *)
) {
  if (count < 2) return;

  // Build the max-heap from the last parent back to the root.
  for (size_t root = count / 2; root > 0; root--) {
    _array__sort_sift(
      contents,
      root - 1,
      count,
      element_size,
      compare
    );
  }

  // Move each maximum to the end and restore the remaining heap.
  for (size_t last = count - 1; last > 0; last--) {
    _array__sort_swap(
      contents,
      contents + last * element_size,
      element_size
    );
    _array__sort_sift(contents, 0, last, element_size, compare);
  }
}

// Ranges shorter than this use insertion sort.
#define TS_ARRAY_SORT_INSERTION_CUTOFF 7

// Ranges longer than this take their pivot from a median of three medians.
#define TS_ARRAY_SORT_NINTHER_CUTOFF 40

// Maximum number of pending ranges for a uint32_t element count.
#define TS_ARRAY_SORT_STACK_SIZE 32

// A contiguous range of elements still to be sorted.
typedef struct {
  unsigned char *start;
  uint32_t count;
  // Remaining partition levels before switching to heapsort.
  uint32_t depth;
} ArraySortRange;

// Sorts `count` elements in place according to `compare`.
static inline void _array__sort(
  void *contents,
  uint32_t count,
  size_t element_size,
  int (*compare)(const void *, const void *)
) {
  unsigned char *bytes = (unsigned char *)contents;
  if (count >= TS_ARRAY_SORT_INSERTION_CUTOFF) {
    uint32_t sorted_end = 1;
    while (
      sorted_end < count &&
      compare(
        bytes + (sorted_end - 1) * element_size,
        bytes + sorted_end * element_size
      ) <= 0
    ) sorted_end++;
    if (sorted_end == count) return;
  }

  uint32_t depth_limit = 0;
  for (uint32_t n = count; n > 1; n >>= 1) depth_limit += 2;
  ArraySortRange range = {bytes, count, depth_limit};
  // Save the larger partition and process the smaller one first. Each added
  // stack entry reduces the active range to at most half its size, so 32
  // entries cover any uint32_t element count.
  ArraySortRange pending[TS_ARRAY_SORT_STACK_SIZE];
  uint32_t pending_count = 0;
  for (;;) {
    unsigned char *const first = range.start;
    const size_t range_count = range.count;
    if (range_count < TS_ARRAY_SORT_INSERTION_CUTOFF) {
      _array__sort_insertion(first, range_count, element_size, compare);
    } else if (range.depth == 0) {
      // Avoid quadratic work from repeatedly unbalanced partitions.
      _array__sort_heap(first, range_count, element_size, compare);
    } else {
      unsigned char *const end = first + range_count * element_size;
      unsigned char *pivot = first + (range_count / 2) * element_size;
      if (range_count > TS_ARRAY_SORT_INSERTION_CUTOFF) {
        unsigned char *low = first;
        unsigned char *high = end - element_size;
        if (range_count > TS_ARRAY_SORT_NINTHER_CUTOFF) {
          const size_t step = (range_count / 8) * element_size;
          low = _array__sort_median(low, low + step, low + 2 * step, compare);
          pivot = _array__sort_median(pivot - step, pivot, pivot + step, compare);
          high = _array__sort_median(high - 2 * step, high - step, high, compare);
        }
        pivot = _array__sort_median(low, pivot, high, compare);
      }
      _array__sort_swap(first, pivot, element_size);

      // The pivot stays at `first`. The regions are:
      //   [first, equal_end)           equal to pivot
      //   [equal_end, less_end)        less than pivot
      //   [less_end, greater_start]    unclassified
      //   (greater_start, equal_start] greater than pivot
      //   (equal_start, end)           equal to pivot
      unsigned char *equal_end = first + element_size;
      unsigned char *less_end = equal_end;
      unsigned char *greater_start = end - element_size;
      unsigned char *equal_start = greater_start;
      for (;;) {
        int order;
        while (
          less_end <= greater_start &&
          (order = compare(less_end, first)) <= 0
        ) {
          if (order == 0) {
            _array__sort_swap(equal_end, less_end, element_size);
            equal_end += element_size;
          }
          less_end += element_size;
        }
        while (
          less_end <= greater_start &&
          (order = compare(greater_start, first)) >= 0
        ) {
          if (order == 0) {
            _array__sort_swap(greater_start, equal_start, element_size);
            equal_start -= element_size;
          }
          greater_start -= element_size;
        }
        if (less_end > greater_start) break;
        _array__sort_swap(less_end, greater_start, element_size);
        less_end += element_size;
        greater_start -= element_size;
      }

      // Move the elements equal to the pivot into the middle.
      const size_t less_count = (size_t)(less_end - equal_end) / element_size;
      const size_t greater_count =
        (size_t)(equal_start - greater_start) / element_size;
      const size_t left_equal = (size_t)(equal_end - first) / element_size;
      const size_t right_equal =
        (size_t)(end - equal_start) / element_size - 1;
      _array__sort_swap_run(
        first,
        less_end - (left_equal < less_count ? left_equal : less_count) * element_size,
        left_equal < less_count ? left_equal : less_count,
        element_size
      );
      _array__sort_swap_run(
        less_end,
        end - (right_equal < greater_count ? right_equal : greater_count) * element_size,
        right_equal < greater_count ? right_equal : greater_count,
        element_size
      );

      const uint32_t next_depth = range.depth - 1;
      const ArraySortRange left = {first, (uint32_t)less_count, next_depth};
      const ArraySortRange right = {
        end - greater_count * element_size,
        (uint32_t)greater_count,
        next_depth
      };
      const bool left_is_smaller = left.count <= right.count;
      const ArraySortRange smaller = left_is_smaller ? left : right;
      const ArraySortRange larger = left_is_smaller ? right : left;
      if (larger.count > 1) {
        ts_assert(pending_count < TS_ARRAY_SORT_STACK_SIZE);
        pending[pending_count++] = larger;
      }
      range = smaller;
      continue;
    }
    if (pending_count == 0) return;
    range = pending[--pending_count];
  }
}

/// A binary search routine, based on Rust's `std::slice::binary_search_by`.
/// This is not what you're looking for, see `array_search_sorted_with` or `array_search_sorted_by`.
#define _array__search_sorted(self, start, compare, suffix, needle, _index, _exists) \
  do { \
    *(_index) = start; \
    *(_exists) = false; \
    uint32_t size = (self)->size - *(_index); \
    if (size == 0) break; \
    int comparison; \
    while (size > 1) { \
      uint32_t half_size = size / 2; \
      uint32_t mid_index = *(_index) + half_size; \
      comparison = compare(&((self)->contents[mid_index] suffix), (needle)); \
      if (comparison <= 0) *(_index) = mid_index; \
      size -= half_size; \
    } \
    comparison = compare(&((self)->contents[*(_index)] suffix), (needle)); \
    if (comparison == 0) *(_exists) = true; \
    else if (comparison < 0) *(_index) += 1; \
  } while (0)

/// Helper macro for the `_sorted_by` routines below. This takes the left (existing)
/// parameter by reference in order to work with the generic sorting function above.
#define _compare_int(a, b) ((int)*(a) - (int)(b))

#ifdef _MSC_VER
#pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#ifdef __cplusplus
}
#endif

#endif  // TREE_SITTER_ARRAY_H_
