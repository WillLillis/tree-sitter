// PROTOTYPE, NOT PRODUCTION. Sizes the prize for a structural-join style
// evaluation of the pathological sibling pattern:
//
//   ((attribute_item)* @attr (line_comment)* @doc
//    (function_item name: (identifier) @name) @fn)
//
// The current engine takes ~700 ms on crates/cli/src/tests/query_test.rs and
// does ~322M capture-comparison steps to produce ~8000 matches. The question
// this answers: how much of that is a bad algorithm, and how much is the
// semantics themselves demanding quadratic output?
//
// So it computes the match set two ways:
//
//   GREEDY   -- one match per function_item, taking the maximal preceding run
//               of line_comments and, before that, of attribute_items. This is
//               what a user almost certainly means.
//   ALLSPLIT -- every valid (attr-run start, doc-run start) pair, which is what
//               `*` meaning "zero or more" actually licenses, and what the
//               engine returns.
//
// Both are single-pass over the tree: O(nodes), output-sensitive.
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

const TSLanguage *tree_sitter_rust(void);

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

typedef struct {
  TSSymbol attr, comment, func, ident;
  TSFieldId name_field;
} Syms;

// Accumulated so the compiler cannot optimize the walk away, and so we can
// check the match count against the real engine.
typedef struct {
  unsigned long matches;
  unsigned long captures;
  unsigned long children_scanned;
} Result;

// One pass over a single sibling list.
static void scan_siblings(
  const TSNode *kids, unsigned n, const Syms *s, bool all_splits, Result *out
) {
  out->children_scanned += n;
  for (unsigned i = 0; i < n; i++) {
    if (ts_node_symbol(kids[i]) != s->func) continue;

    // The function must have a `name:` child for the pattern to match.
    TSNode name = ts_node_child_by_field_id(kids[i], s->name_field);
    if (ts_node_is_null(name) || ts_node_symbol(name) != s->ident) continue;

    // Maximal run of line_comments immediately before i.
    unsigned d = i;
    while (d > 0 && ts_node_symbol(kids[d - 1]) == s->comment) d--;
    // Maximal run of attribute_items immediately before that.
    unsigned a = d;
    while (a > 0 && ts_node_symbol(kids[a - 1]) == s->attr) a--;

    if (!all_splits) {
      out->matches++;
      // @fn + @name, plus one capture per node in each run.
      out->captures += 2 + (i - d) + (d - a);
    } else {
      // Every (attr-run start, doc-run start) pair is a distinct match under
      // `*` meaning zero-or-more. attr start ranges over [a..d], doc start
      // over [d..i], and an attr run may only be non-empty if it is adjacent.
      for (unsigned as = a; as <= d; as++) {
        for (unsigned ds = d; ds <= i; ds++) {
          out->matches++;
          out->captures += 2 + (i - ds) + (ds >= d ? (d - as) : 0);
        }
      }
    }
  }
}

// Depth-first walk, materializing each node's named-child list once.
static void walk(TSTreeCursor *c, const Syms *s, bool all_splits, Result *out) {
  TSNode kids[4096];
  unsigned n = 0;
  if (ts_tree_cursor_goto_first_child(c)) {
    do {
      TSNode k = ts_tree_cursor_current_node(c);
      if (ts_node_is_named(k) && n < 4096) kids[n++] = k;
    } while (ts_tree_cursor_goto_next_sibling(c));
    ts_tree_cursor_goto_parent(c);
  }
  if (n) scan_siblings(kids, n, s, all_splits, out);

  if (ts_tree_cursor_goto_first_child(c)) {
    do { walk(c, s, all_splits, out); } while (ts_tree_cursor_goto_next_sibling(c));
    ts_tree_cursor_goto_parent(c);
  }
}

static Result run_pass(TSTree *tree, const Syms *s, bool all_splits) {
  Result r = {0};
  TSTreeCursor c = ts_tree_cursor_new(ts_tree_root_node(tree));
  walk(&c, s, all_splits, &r);
  ts_tree_cursor_delete(&c);
  return r;
}

int main(int argc, char **argv) {
  if (argc < 2) { fprintf(stderr, "usage: join_proto <source.rs> [reps]\n"); return 1; }
  int reps = argc > 2 ? atoi(argv[2]) : 20;
  const TSLanguage *lang = tree_sitter_rust();

  uint32_t slen;
  char *ssrc = slurp(argv[1], &slen);
  TSParser *p = ts_parser_new();
  ts_parser_set_language(p, lang);
  TSTree *tree = ts_parser_parse_string(p, NULL, ssrc, slen);

  Syms s = {
    .attr    = ts_language_symbol_for_name(lang, "attribute_item", 14, true),
    .comment = ts_language_symbol_for_name(lang, "line_comment", 12, true),
    .func    = ts_language_symbol_for_name(lang, "function_item", 13, true),
    .ident   = ts_language_symbol_for_name(lang, "identifier", 10, true),
    .name_field = ts_language_field_id_for_name(lang, "name", 4),
  };
  if (!s.attr || !s.comment || !s.func || !s.ident || !s.name_field) {
    fprintf(stderr, "symbol lookup failed (attr=%u comment=%u func=%u ident=%u field=%u)\n",
            s.attr, s.comment, s.func, s.ident, s.name_field);
    return 1;
  }

  printf("source: %s (%u bytes)\n\n", argv[1], slen);
  for (int mode = 0; mode < 2; mode++) {
    bool all_splits = mode == 1;
    Result r = {0};
    double best = 1e18;
    for (int i = 0; i < reps; i++) {
      double t0 = now_ms();
      r = run_pass(tree, &s, all_splits);
      double dt = now_ms() - t0;
      if (dt < best) best = dt;
    }
    printf("%-9s  min=%7.3f ms   matches=%-8lu captures=%-9lu children_scanned=%lu\n",
           all_splits ? "ALLSPLIT" : "GREEDY", best, r.matches, r.captures, r.children_scanned);
  }

  ts_tree_delete(tree); ts_parser_delete(p); free(ssrc);
  return 0;
}
