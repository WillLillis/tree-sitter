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

  uint32_t qlen, slen;
  char *qsrc = slurp(argv[2], &qlen);
  char *ssrc = slurp(argv[3], &slen);

  uint32_t off; TSQueryError err;
  TSQuery *q = ts_query_new(lang, qsrc, qlen, &off, &err);
  if (!q) { fprintf(stderr, "query failed: err=%d off=%u\n", err, off); return 1; }

  TSParser *p = ts_parser_new();
  ts_parser_set_language(p, lang);
  TSTree *tree = ts_parser_parse_string(p, NULL, ssrc, slen);

  double *times = malloc(sizeof(double) * reps);
  unsigned long returned = 0;
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
  printf("%-34s %-10s reps=%d  min=%.2f ms  median=%.2f ms  returned=%lu\n",
         strrchr(argv[2], '/') ? strrchr(argv[2], '/') + 1 : argv[2],
         argv[4], reps, times[0], times[reps / 2], returned);

  free(times); ts_tree_delete(tree); ts_parser_delete(p); ts_query_delete(q);
  free(qsrc); free(ssrc);
  return 0;
}
