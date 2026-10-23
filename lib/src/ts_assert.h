#ifndef TREE_SITTER_ASSERT_H_
#define TREE_SITTER_ASSERT_H_

#ifdef NDEBUG
#define ts_assert(e) ((void)(e))
#else
#include <assert.h>
#define ts_assert(e) assert(e)
#endif

#if defined(__GNUC__) || defined(__clang__)
#define ts_unreachable() do { ts_assert(0); __builtin_unreachable(); } while (0)
#elif defined(_MSC_VER)
#define ts_unreachable() do { ts_assert(0); __assume(0); } while (0)
#else
#define ts_unreachable() ts_assert(0)
#endif

#endif // TREE_SITTER_ASSERT_H_
