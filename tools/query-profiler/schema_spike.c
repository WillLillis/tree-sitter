// SPIKE. Measures the cost of the schema-lookup path that would replace
// ts_query__perform_analysis, and checks its guarantees against the shipped
// analyzer's on the same query.
//
// Only the mandatory-children rule is implemented here -- enough to answer the
// open question, which is whether table lookups are as cheap as assumed. Parity
// of the field-based rule was established separately (see 13-node-types-analysis.md).
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
const TSLanguage *tree_sitter_javascript(void);
const TSLanguage *tree_sitter_python(void);
const TSLanguage *tree_sitter_go(void);
const TSLanguage *tree_sitter_c(void);
#include "schema_rust.h"
#include "schema_javascript.h"
#include "schema_python.h"
#include "schema_go.h"
#include "schema_c.h"



static double now_ms(void) {
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
static char *slurp(const char *p, uint32_t *len) {
  FILE *f = fopen(p, "rb"); if (!f) { fprintf(stderr, "open %s\n", p); exit(1); }
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
  char *b = malloc(n + 1); size_t g = fread(b, 1, n, f); b[g] = 0; fclose(f);
  *len = (uint32_t)g; return b;
}

// Resolved form: schema keyed by TSSymbol, with child names resolved to symbols.
// This is the load-time pass a real implementation would do -- names in the
// artifact, ids resolved against the language actually in hand.
typedef struct { TSFieldId field; TSSymbol *types; uint16_t count; } ResolvedField;
typedef struct {
  TSSymbol *mandatory; uint16_t count;
  ResolvedField *fields; uint16_t field_count;
} ResolvedNode;
static ResolvedNode *g_by_symbol;
static uint32_t g_symbol_count;

static TSSymbol resolve(const TSLanguage *l, const char *name) {
  TSSymbol s = ts_language_symbol_for_name(l, name, (uint32_t)strlen(name), true);
  if (!s) s = ts_language_symbol_for_name(l, name, (uint32_t)strlen(name), false);
  return s;
}

static const SchemaNode *g_schema; static unsigned g_schema_count;
static double build_resolved(const TSLanguage *l) {
  double t0 = now_ms();
  g_symbol_count = ts_language_symbol_count(l);
  g_by_symbol = calloc(g_symbol_count, sizeof(ResolvedNode));
  for (unsigned i = 0; i < g_schema_count; i++) {
    const SchemaNode *n = &g_schema[i];
    TSSymbol parent = resolve(l, n->name);
    if (!parent || parent >= g_symbol_count || n->mandatory_count == 0) continue;
    ResolvedNode *r = &g_by_symbol[parent];
    r->mandatory = malloc(sizeof(TSSymbol) * n->mandatory_count);
    r->count = 0;
    for (unsigned j = 0; j < n->mandatory_count; j++) {
      TSSymbol c = resolve(l, n->mandatory[j]);
      if (c) r->mandatory[r->count++] = c;
    }
    if (n->field_count) {
      r->fields = calloc(n->field_count, sizeof(ResolvedField));
      r->field_count = 0;
      for (unsigned j = 0; j < n->field_count; j++) {
        const SchemaField *f = &n->fields[j];
        TSFieldId fid = ts_language_field_id_for_name(l, f->name, (uint32_t)strlen(f->name));
        if (!fid) continue;
        ResolvedField *rf = &r->fields[r->field_count++];
        rf->field = fid;
        rf->types = malloc(sizeof(TSSymbol) * f->type_count);
        rf->count = 0;
        for (unsigned k = 0; k < f->type_count; k++) {
          TSSymbol t = resolve(l, f->types[k]);
          if (t) rf->types[rf->count++] = t;
        }
      }
    }
  }
  return now_ms() - t0;
}

// The replacement for perform_analysis, mandatory-children rule only:
// a step is guaranteed if its symbol is a child its parent must always have.
static double schema_analyze(const TSQuery *q, unsigned *agree, unsigned *lost, unsigned *unsound) {
  TSSymbol parent_at_depth[64] = {0};
  double t0 = now_ms();
  unsigned a = 0, l = 0, u = 0;
  for (uint32_t i = 0; i < q->steps.size; i++) {
    const QueryStep *st = array_get(&q->steps, i);
    if (st->depth == PATTERN_DONE_MARKER) { memset(parent_at_depth, 0, sizeof parent_at_depth); continue; }
    if (st->depth < 64) parent_at_depth[st->depth] = st->symbol;
    if (st->depth == 0 || st->is_pass_through || st->is_dead_end) continue;
    TSSymbol parent = st->depth < 64 ? parent_at_depth[st->depth - 1] : 0;
    bool guar = false;
    // The mandatory set records only THAT a child must be present, not which
    // field it fills. A step carrying a field constraint therefore needs the
    // field rule; applying the mandatory rule to it is unsound -- e.g.
    // `(scoped_type_identifier path: (identifier))`, where `path` is optional
    // but an `identifier` is mandatory elsewhere in the node.
    if (parent && parent < g_symbol_count) {
      const ResolvedNode *r = &g_by_symbol[parent];
      if (st->field == 0) {
        // Mandatory-children rule: sound only without a field constraint, since
        // the mandatory set records presence, not which field a child fills.
        for (unsigned k = 0; k < r->count; k++)
          if (r->mandatory[k] == st->symbol) { guar = true; break; }
      } else if (st->symbol != WILDCARD_SYMBOL && !st->supertype_symbol) {
        // Field rule: guaranteed iff the field is required and single-valued
        // (both already filtered at generation) and every type it admits is
        // the one this step matches.
        for (unsigned k = 0; k < r->field_count; k++) {
          if (r->fields[k].field != st->field) continue;
          bool all = r->fields[k].count > 0;
          for (unsigned m = 0; m < r->fields[k].count; m++)
            if (r->fields[k].types[m] != st->symbol) { all = false; break; }
          guar = all;
          break;
        }
      }
    }
    bool c = st->parent_pattern_guaranteed;
    if (c && guar) a++;
    else if (c && !guar) { l++; if (getenv("SPIKE_WHY")) fprintf(stderr,
        "  LOST     step %u depth %u  parent=%s  sym=%s  field=%s\n", i, st->depth,
        ts_language_symbol_name(q->language, parent),
        st->symbol ? ts_language_symbol_name(q->language, st->symbol) : "_",
        st->field ? ts_language_field_name_for_id(q->language, st->field) : "-"); }
    else if (!c && guar) { u++; if (getenv("SPIKE_WHY")) fprintf(stderr,
        "  UNSOUND  step %u depth %u  parent=%s  sym=%s  field=%s  super=%s  imm=%d last=%d\n",
        i, st->depth, ts_language_symbol_name(q->language, parent),
        st->symbol ? ts_language_symbol_name(q->language, st->symbol) : "_",
        st->field ? ts_language_field_name_for_id(q->language, st->field) : "-",
        st->supertype_symbol ? ts_language_symbol_name(q->language, st->supertype_symbol) : "-",
        st->is_immediate, st->is_last_child); }
  }
  double dt = now_ms() - t0;
  *agree = a; *lost = l; *unsound = u;
  return dt;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: schema_spike <lang> <query.scm> [reps]\n"); return 1; }
  int reps = argc > 3 ? atoi(argv[3]) : 200;
  const TSLanguage *lang = NULL;
  if (!strcmp(argv[1],"rust"))            { lang = tree_sitter_rust();       g_schema = rust_schema;       g_schema_count = rust_schema_count; }
  else if (!strcmp(argv[1],"javascript")) { lang = tree_sitter_javascript(); g_schema = javascript_schema; g_schema_count = javascript_schema_count; }
  else if (!strcmp(argv[1],"python"))     { lang = tree_sitter_python();     g_schema = python_schema;     g_schema_count = python_schema_count; }
  else if (!strcmp(argv[1],"go"))         { lang = tree_sitter_go();         g_schema = go_schema;         g_schema_count = go_schema_count; }
  else if (!strcmp(argv[1],"c"))          { lang = tree_sitter_c();          g_schema = c_schema;          g_schema_count = c_schema_count; }
  else { fprintf(stderr, "unknown language %s\n", argv[1]); return 1; }

  uint32_t qlen; char *qsrc = slurp(argv[2], &qlen);
  uint32_t off; TSQueryError err;

  double t0 = now_ms();
  TSQuery *q = ts_query_new(lang, qsrc, qlen, &off, &err);
  double t_stock = now_ms() - t0;
  if (!q) { fprintf(stderr, "query failed err=%d off=%u\n", err, off); return 1; }

  double t_resolve = build_resolved(lang);

  unsigned agree = 0, lost = 0, unsound = 0;
  double best = 1e18;
  for (int r = 0; r < reps; r++) {
    double d = schema_analyze(q, &agree, &lost, &unsound);
    if (d < best) best = d;
  }

  printf("query: %s   (%u steps, %u patterns)\n", argv[2], q->steps.size, q->patterns.size);
  printf("  stock ts_query_new (parse + full analysis) : %8.3f ms\n", t_stock);
  printf("  schema resolve, names -> symbols, once     : %8.3f ms\n", t_resolve);
  printf("  schema analysis pass over all steps        : %8.4f ms   (min of %d)\n", best, reps);
  printf("  => analysis replaced by a pass %.0fx cheaper than the whole compile\n",
         best > 0 ? t_stock / best : 0.0);
  printf("  guarantees vs shipped analyzer: agree=%u lost=%u unsound=%u\n", agree, lost, unsound);
  ts_query_delete(q); free(qsrc);
  return 0;
}
