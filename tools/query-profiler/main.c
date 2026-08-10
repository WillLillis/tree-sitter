// Static + dynamic profile of the tree-sitter query engine.
// Includes an instrumented copy of query.c so it can see TSQuery internals.
#define _DEFAULT_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// wasm stubs: we build without TREE_SITTER_FEATURE_WASM
#include "lib/src/alloc.c"
#include "lib/src/get_changed_ranges.c"
#include "lib/src/language.c"
#include "lib/src/lexer.c"
#include "lib/src/node.c"
#include "lib/src/parser.c"
#include "lib/src/point.c"
#include "query_instr.c"
#include "lib/src/stack.c"
#include "lib/src/subtree.c"
#include "lib/src/tree_cursor.c"
#include "lib/src/tree.c"

void ts_wasm_store_delete(TSWasmStore *self) { (void)self; }
void ts_wasm_store_reset(TSWasmStore *self) { (void)self; }
bool ts_language_is_wasm(const TSLanguage *self) { (void)self; return false; }
void ts_wasm_language_retain(const TSLanguage *self) { (void)self; }
void ts_wasm_language_release(const TSLanguage *self) { (void)self; }

// Only the grammars present in test/fixtures are linked; see the Makefile.
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
  fseek(f, 0, SEEK_END);
  long n = ftell(f);
  fseek(f, 0, SEEK_SET);
  char *buf = malloc(n + 1);
  size_t got = fread(buf, 1, n, f);
  buf[got] = 0;
  fclose(f);
  *len = (uint32_t)got;
  return buf;
}

typedef struct { const char *name; const TSLanguage *(*fn)(void); } LangEntry;

static void static_profile(const TSQuery *q, const char *label) {
  unsigned pass_through = 0, dead_end = 0, wildcard = 0, with_captures = 0;
  unsigned rpg = 0, ppg = 0, done_markers = 0, immediate = 0, supertype = 0;
  unsigned max_depth = 0;
  unsigned cap_hist[MAX_STEP_CAPTURE_COUNT + 1] = {0};
  for (unsigned i = 0; i < q->steps.size; i++) {
    const QueryStep *s = array_get(&q->steps, i);
    if (s->depth == PATTERN_DONE_MARKER) { done_markers++; continue; }
    if (s->is_pass_through) pass_through++;
    if (s->is_dead_end) dead_end++;
    if (s->symbol == WILDCARD_SYMBOL) wildcard++;
    if (s->is_immediate) immediate++;
    if (s->supertype_symbol) supertype++;
    if (s->contains_captures) with_captures++;
    if (s->root_pattern_guaranteed) rpg++;
    if (s->parent_pattern_guaranteed) ppg++;
    if (s->depth > max_depth) max_depth = s->depth;
    unsigned n = 0;
    while (n < MAX_STEP_CAPTURE_COUNT && s->capture_ids[n] != NONE) n++;
    cap_hist[n]++;
  }
  unsigned real_steps = q->steps.size - done_markers;
  printf("  [static] %s\n", label);
  printf("    patterns=%u steps=%u (real=%u done=%u) pattern_map=%u wildcard_roots=%u\n",
         q->patterns.size, q->steps.size, real_steps, done_markers,
         q->pattern_map.size, q->wildcard_root_pattern_count);
  printf("    captures=%u pred_values=%u pred_steps=%u negated_fields=%u step_offsets=%u\n",
         q->captures.slices.size, q->predicate_values.slices.size,
         q->predicate_steps.size, q->negated_fields.size, q->step_offsets.size);
  printf("    encoding overhead: pass_through=%u (%.1f%%) dead_end=%u (%.1f%%) => %.1f%% of steps are control-flow only\n",
         pass_through, 100.0 * pass_through / real_steps,
         dead_end, 100.0 * dead_end / real_steps,
         100.0 * (pass_through + dead_end) / real_steps);
  printf("    wildcard=%u immediate=%u supertype=%u max_depth=%u\n",
         wildcard, immediate, supertype, max_depth);
  printf("    guaranteed: root=%u (%.1f%%) parent=%u (%.1f%%)\n",
         rpg, 100.0 * rpg / real_steps, ppg, 100.0 * ppg / real_steps);
  printf("    captures/step histogram: 0=%u 1=%u 2=%u 3=%u  (dropped-at-compile=%lu)\n",
         cap_hist[0], cap_hist[1], cap_hist[2], cap_hist[3], qprobe.capture_drops);
  printf("    sizeof(QueryStep)=%zu bytes => steps array = %zu KiB\n",
         sizeof(QueryStep), (size_t)q->steps.size * sizeof(QueryStep) / 1024);
}

static void run_one(const char *lang_name, const TSLanguage *lang,
                    const char *query_path, const char *src_path, int mode) {
  uint32_t qlen, slen;
  char *qsrc = slurp(query_path, &qlen);
  char *ssrc = slurp(src_path, &slen);

  // --- compile ---
  uint32_t err_off = 0;
  TSQueryError err = TSQueryErrorNone;
  memset(&qprobe, 0, sizeof(qprobe));
  qp_t_analyze_ms = qp_t_subgraph_scan_ms = qp_t_perform_ms = 0;
  qp_perform_calls = qp_perform_iterations = qp_analysis_aborts = 0;
  qp_subgraph_count = qp_subgraph_nodes = 0;
  double t0 = now_ms();
  TSQuery *q = ts_query_new(lang, qsrc, qlen, &err_off, &err);
  double t_compile = now_ms() - t0;
  if (!q) {
    printf("  !! query compile failed: err=%d at byte %u\n", err, err_off);
    free(qsrc); free(ssrc);
    return;
  }

  printf("\n=== %s : %s (%u bytes) over %s (%u bytes) ===\n",
         lang_name, query_path, qlen, src_path, slen);
  printf("  [compile] ts_query_new = %.2f ms  (analyze=%.2f ms = %.0f%%)\n",
         t_compile, qp_t_analyze_ms, 100.0 * qp_t_analyze_ms / t_compile);
  printf("    analyze breakdown: full parse-table scan=%.2f ms, perform_analysis=%.2f ms (%lu calls, %lu iters, %lu aborts)\n",
         qp_t_subgraph_scan_ms, qp_t_perform_ms, qp_perform_calls, qp_perform_iterations, qp_analysis_aborts);
  printf("    analysis set: inserts=%lu (dedup_hits=%lu, %.0f%%) shift=%lu (avg %.1f) peak_set=%lu\n",
         qprobe.analysis_inserts, qprobe.analysis_dedup_hits,
         qprobe.analysis_inserts ? 100.0 * qprobe.analysis_dedup_hits / qprobe.analysis_inserts : 0.0,
         qprobe.analysis_shift,
         qprobe.analysis_inserts ? (double)qprobe.analysis_shift / qprobe.analysis_inserts : 0.0,
         qprobe.max_analysis_set);
  printf("      of dedup hits, %.0f%% matched the LAST element; of inserts, %.0f%% were pure appends\n",
         qprobe.analysis_dedup_hits ? 100.0 * qprobe.analysis_hit_at_back / qprobe.analysis_dedup_hits : 0.0,
         (qprobe.analysis_inserts - qprobe.analysis_dedup_hits)
           ? 100.0 * qprobe.analysis_ins_at_back / (qprobe.analysis_inserts - qprobe.analysis_dedup_hits) : 0.0);
  printf("    comparator: calls=%lu steps=%lu (avg %.1f stack entries/call, %.1f calls/insert)\n",
         qprobe.analysis_compares, qprobe.analysis_cmp_steps,
         qprobe.analysis_compares ? (double)qprobe.analysis_cmp_steps / qprobe.analysis_compares : 0.0,
         qprobe.analysis_inserts ? (double)qprobe.analysis_compares / qprobe.analysis_inserts : 0.0);
  printf("    language: states=%u symbols=%u  predecessor_map alloc=%lu KiB  subgraphs=%lu (%lu nodes)\n",
         lang->state_count, lang->symbol_count, qp_predecessor_map_bytes / 1024,
         qp_subgraph_count, qp_subgraph_nodes);
  static_profile(q, query_path);

  // --- parse the source ---
  TSParser *parser = ts_parser_new();
  ts_parser_set_language(parser, lang);
  t0 = now_ms();
  TSTree *tree = ts_parser_parse_string(parser, NULL, ssrc, slen);
  double t_parse = now_ms() - t0;
  // count nodes
  unsigned long node_count = 0;
  {
    TSTreeCursor c = ts_tree_cursor_new(ts_tree_root_node(tree));
    bool go = true;
    while (go) {
      node_count++;
      if (ts_tree_cursor_goto_first_child(&c)) continue;
      for (;;) {
        if (ts_tree_cursor_goto_next_sibling(&c)) break;
        if (!ts_tree_cursor_goto_parent(&c)) { go = false; break; }
      }
    }
    ts_tree_cursor_delete(&c);
  }
  printf("  [parse]   %.2f ms, %lu named+anon nodes\n", t_parse, node_count);

  // --- execute ---
  memset(&qprobe, 0, sizeof(qprobe));
  TSQueryCursor *cur = ts_query_cursor_new();
  t0 = now_ms();
  ts_query_cursor_exec(cur, q, ts_tree_root_node(tree));
  unsigned long returned = 0;
  if (mode == 0) {
    TSQueryMatch m;
    while (ts_query_cursor_next_match(cur, &m)) returned++;
  } else {
    TSQueryMatch m; uint32_t ci;
    while (ts_query_cursor_next_capture(cur, &m, &ci)) returned++;
  }
  double t_exec = now_ms() - t0;

  const char *mode_name = mode == 0 ? "next_match" : "next_capture";
  printf("  [exec/%s] %.2f ms, returned=%lu\n", mode_name, t_exec, returned);
  printf("    nodes_entered=%lu descends=%lu  matches_pushed=%lu\n",
         qprobe.nodes_entered, qprobe.nodes_descended, qprobe.matches);
  printf("    states: started=%lu copied=%lu (fan-out x%.2f)  peak_live=%lu peak_finished=%lu\n",
         qprobe.states_started, qprobe.states_copied,
         qprobe.states_started ? 1.0 + (double)qprobe.states_copied / qprobe.states_started : 0.0,
         qprobe.max_states, qprobe.max_finished);
  printf("    state_visits=%lu (%.1f per node entered), of which %.1f%% fail the depth test\n",
         qprobe.state_visits,
         qprobe.nodes_entered ? (double)qprobe.state_visits / qprobe.nodes_entered : 0.0,
         qprobe.state_visits ? 100.0 * qprobe.state_visits_depth_skip / qprobe.state_visits : 0.0);
  printf("    dedup: compares=%lu inner_steps=%lu (%.1f per node)\n",
         qprobe.dedup_compares, qprobe.dedup_capture_steps,
         qprobe.nodes_entered ? (double)qprobe.dedup_capture_steps / qprobe.nodes_entered : 0.0);
  printf("    sort_compares=%lu (%.1f per node)\n", qprobe.sort_compares,
         qprobe.nodes_entered ? (double)qprobe.sort_compares / qprobe.nodes_entered : 0.0);
  printf("    pool: acquires=%lu scan_steps=%lu (avg scan %.1f)\n",
         qprobe.pool_acquires, qprobe.pool_scan_steps,
         qprobe.pool_acquires ? (double)qprobe.pool_scan_steps / qprobe.pool_acquires : 0.0);
  printf("    should_descend scan_steps=%lu (%.1f per node)\n", qprobe.descend_scan_steps,
         qprobe.nodes_entered ? (double)qprobe.descend_scan_steps / qprobe.nodes_entered : 0.0);

  ts_query_cursor_delete(cur);
  ts_tree_delete(tree);
  ts_parser_delete(parser);
  ts_query_delete(q);
  free(qsrc); free(ssrc);
}

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s <lang> <query.scm> <source> <mode:match|capture>\n", argv[0]);
    return 1;
  }
  LangEntry langs[] = {
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
  for (unsigned i = 0; i < sizeof(langs)/sizeof(*langs); i++) {
    if (!strcmp(langs[i].name, argv[1])) { lang = langs[i].fn(); break; }
  }
  if (!lang) { fprintf(stderr, "unknown language %s\n", argv[1]); return 1; }
  int mode = strcmp(argv[4], "capture") == 0 ? 1 : 0;
  run_one(argv[1], lang, argv[2], argv[3], mode);
  return 0;
}
