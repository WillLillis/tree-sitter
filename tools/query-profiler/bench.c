// Uninstrumented wall-clock timer, for honest A/B of a change to query.c.
// Links the real lib/src/query.c (no counters), so measurements are not skewed
// by the profiler's instrumentation.
//   bench <lang> <query.scm> <source> <mode:match|capture> [reps]
#define _DEFAULT_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lib/src/alloc.c"
#include "lib/src/get_changed_ranges.c"
#include "lib/src/language.c"
#include "lib/src/lexer.c"
#include "lib/src/node.c"
#include "lib/src/parser.c"
#include "lib/src/point.c"
#include "lib/src/query.c"
#include "lib/src/stack.c"
#include "lib/src/subtree.c"
#include "lib/src/tree_cursor.c"
#include "lib/src/tree.c"

void ts_wasm_store_delete(TSWasmStore *s) { (void)s; }
void ts_wasm_store_reset(TSWasmStore *s) { (void)s; }
bool ts_language_is_wasm(const TSLanguage *s) { (void)s; return false; }
void ts_wasm_language_retain(const TSLanguage *s) { (void)s; }
void ts_wasm_language_release(const TSLanguage *s) { (void)s; }

#ifdef HAVE_JAVASCRIPT
const TSLanguage *tree_sitter_javascript(void);
#endif
#ifdef HAVE_RUST
const TSLanguage *tree_sitter_rust(void);
#endif
#ifdef HAVE_PYTHON
const TSLanguage *tree_sitter_python(void);
#endif
#ifdef HAVE_GO
const TSLanguage *tree_sitter_go(void);
#endif
#ifdef HAVE_C
const TSLanguage *tree_sitter_c(void);
#endif

// Allocation counters, installed via ts_set_allocator. These answer a question
// wall clock cannot on small inputs: does a change add allocator traffic?
static unsigned long n_malloc, n_calloc, n_realloc, n_free;
static void *c_malloc(size_t n) { n_malloc++; return malloc(n); }
static void *c_calloc(size_t a, size_t b) { n_calloc++; return calloc(a, b); }
static void *c_realloc(void *p, size_t n) { n_realloc++; return realloc(p, n); }
static void c_free(void *p) { if (p) n_free++; free(p); }

static double now_ms(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static char *slurp(const char *path, uint32_t *len) {
  FILE *f = fopen(path, "rb");
  if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *b = malloc(n + 1);
  size_t got = fread(b, 1, n, f);
  b[got] = 0; fclose(f); *len = (uint32_t)got;
  return b;
}

// Enumeration-delay mode: time each next_match call separately. When a query
// has an inherently large answer set, total time says little; what matters is
// whether the cost per emitted match stays flat as enumeration proceeds.
static void delay_mode(TSQuery *q, TSTree *tree, int captures_mode) {
  TSQueryCursor *c = ts_query_cursor_new();
  unsigned cap = 1 << 20, n = 0;
  double *d = malloc(sizeof(double) * cap);
  double t_prev = now_ms(), t_start = t_prev;
  ts_query_cursor_exec(c, q, ts_tree_root_node(tree));
  TSQueryMatch m; uint32_t ci;
  while (captures_mode ? ts_query_cursor_next_capture(c, &m, &ci)
                       : ts_query_cursor_next_match(c, &m)) {
    double now = now_ms();
    if (n < cap) d[n++] = now - t_prev;
    t_prev = now;
  }
  double total = now_ms() - t_start;
  if (n == 0) { printf("  no matches\n"); free(d); ts_query_cursor_delete(c); return; }

  // Decile means, to expose any trend in delay as enumeration proceeds.
  printf("  matches=%u  total=%.1f ms  first_match_latency=%.4f ms  mean_delay=%.4f ms\n",
         n, total, d[0], (total - d[0]) / (n > 1 ? n - 1 : 1));
  printf("  delay by decile (ms):");
  for (int k = 0; k < 10; k++) {
    unsigned lo = (unsigned)((double)k / 10 * n), hi = (unsigned)((double)(k + 1) / 10 * n);
    if (hi <= lo) hi = lo + 1;
    double sum = 0; unsigned cnt = 0;
    for (unsigned i = lo; i < hi && i < n; i++) { sum += d[i]; cnt++; }
    printf(" %.4f", cnt ? sum / cnt : 0.0);
  }
  printf("\n  => last decile / first decile = ");
  {
    double a = 0, b = 0; unsigned ca = 0, cb = 0;
    for (unsigned i = 0; i < n / 10 + 1 && i < n; i++) { a += d[i]; ca++; }
    for (unsigned i = n - n / 10 - 1; i < n; i++) { b += d[i]; cb++; }
    double fa = ca ? a / ca : 0, fb = cb ? b / cb : 0;
    printf("%.1fx %s\n", fa > 0 ? fb / fa : 0.0,
           (fa > 0 && fb / fa > 3) ? "  <-- delay GROWS: not constant-delay" : "  <-- roughly flat");
  }
  free(d); ts_query_cursor_delete(c);
}

static int cmp_double(const void *a, const void *b) {
  double x = *(const double *)a, y = *(const double *)b;
  return (x > y) - (x < y);
}

int main(int argc, char **argv) {
  if (argc < 5) { fprintf(stderr, "usage: bench <lang> <query> <src> <match|capture> [reps]\n"); return 1; }
  struct { const char *n; const TSLanguage *(*f)(void); } langs[] = {
#ifdef HAVE_JAVASCRIPT
    {"javascript", tree_sitter_javascript},
#endif
#ifdef HAVE_RUST
    {"rust", tree_sitter_rust},
#endif
#ifdef HAVE_PYTHON
    {"python", tree_sitter_python},
#endif
#ifdef HAVE_GO
    {"go", tree_sitter_go},
#endif
#ifdef HAVE_C
    {"c", tree_sitter_c},
#endif
  };
  const TSLanguage *lang = NULL;
  for (unsigned i = 0; i < sizeof(langs)/sizeof(*langs); i++)
    if (!strcmp(langs[i].n, argv[1])) { lang = langs[i].f(); break; }
  if (!lang) { fprintf(stderr, "unknown language %s\n", argv[1]); return 1; }

  int mode = strcmp(argv[4], "capture") == 0;
  int reps = argc > 5 ? atoi(argv[5]) : 5;
  int delay = argc > 6 && strcmp(argv[6], "delay") == 0;
  ts_set_allocator(c_malloc, c_calloc, c_realloc, c_free);

  uint32_t qlen, slen;
  char *qsrc = slurp(argv[2], &qlen);
  char *ssrc = slurp(argv[3], &slen);

  uint32_t off; TSQueryError err;
  TSQuery *q = ts_query_new(lang, qsrc, qlen, &off, &err);
  if (!q) { fprintf(stderr, "query failed: err=%d off=%u\n", err, off); return 1; }

  TSParser *p = ts_parser_new();
  ts_parser_set_language(p, lang);
  TSTree *tree = ts_parser_parse_string(p, NULL, ssrc, slen);

  if (delay) { delay_mode(q, tree, mode); ts_tree_delete(tree); ts_parser_delete(p);
               ts_query_delete(q); free(qsrc); free(ssrc); return 0; }

  double *times = malloc(sizeof(double) * reps);
  unsigned long returned = 0;
  // Reset after setup so the counts cover only cursor creation + execution,
  // which is what a per-edit query in an editor actually repeats.
  n_malloc = n_calloc = n_realloc = n_free = 0;
  for (int r = 0; r < reps; r++) {
    TSQueryCursor *c = ts_query_cursor_new();
    double t0 = now_ms();
    ts_query_cursor_exec(c, q, ts_tree_root_node(tree));
    returned = 0;
    if (mode) { TSQueryMatch m; uint32_t ci; while (ts_query_cursor_next_capture(c, &m, &ci)) returned++; }
    else      { TSQueryMatch m;              while (ts_query_cursor_next_match(c, &m)) returned++; }
    times[r] = now_ms() - t0;
    ts_query_cursor_delete(c);
  }
  qsort(times, reps, sizeof(double), cmp_double);
  double total = 0;
  for (int r = 0; r < reps; r++) total += times[r];
  printf("%-22s %-9s reps=%-6d min=%8.4f ms  median=%8.4f ms  mean=%8.4f ms  "
         "allocs/rep: m=%.1f c=%.1f re=%.1f f=%.1f  returned=%lu\n",
         strrchr(argv[3], '/') ? strrchr(argv[3], '/') + 1 : argv[3],
         argv[4], reps, times[0], times[reps / 2], total / reps,
         (double)n_malloc / reps, (double)n_calloc / reps,
         (double)n_realloc / reps, (double)n_free / reps, returned);

  free(times); ts_tree_delete(tree); ts_parser_delete(p); ts_query_delete(q);
  free(qsrc); free(ssrc);
  return 0;
}
