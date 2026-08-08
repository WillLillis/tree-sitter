// Verify specific correctness defects in query.c.
#define _DEFAULT_SOURCE 1
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

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

void ts_wasm_store_delete(TSWasmStore *s) { (void)s; }
void ts_wasm_store_reset(TSWasmStore *s) { (void)s; }
bool ts_language_is_wasm(const TSLanguage *s) { (void)s; return false; }
void ts_wasm_language_retain(const TSLanguage *s) { (void)s; }
void ts_wasm_language_release(const TSLanguage *s) { (void)s; }

#ifndef HAVE_RUST
#error "these repros need the rust fixture grammar (test/fixtures/grammars/rust)"
#endif
const TSLanguage *tree_sitter_rust(void);

static const char *errname(TSQueryError e) {
  switch (e) {
    case TSQueryErrorNone: return "None";
    case TSQueryErrorSyntax: return "Syntax";
    case TSQueryErrorNodeType: return "NodeType";
    case TSQueryErrorField: return "Field";
    case TSQueryErrorCapture: return "Capture";
    case TSQueryErrorStructure: return "Structure";
    case TSQueryErrorLanguage: return "Language";
  }
  return "?";
}

static TSQuery *compile(const TSLanguage *l, const char *src, const char *label) {
  uint32_t off = 0; TSQueryError err = TSQueryErrorNone;
  qprobe.capture_drops = 0;
  TSQuery *q = ts_query_new(l, src, (uint32_t)strlen(src), &off, &err);
  printf("  %-46s -> %s", label, q ? "compiled" : errname(err));
  if (!q) printf(" @%u", off);
  printf("\n");
  return q;
}

int main(void) {
  const TSLanguage *rust = tree_sitter_rust();

  printf("\n### BUG 1: MAX_STEP_CAPTURE_COUNT = 3 silently drops the 4th capture\n");
  {
    TSQuery *q = compile(rust, "(identifier) @a @b @c @d", "(identifier) @a @b @c @d");
    if (q) {
      printf("    query reports capture_count=%u (names: ", ts_query_capture_count(q));
      for (uint32_t i = 0; i < ts_query_capture_count(q); i++) {
        uint32_t len; const char *n = ts_query_capture_name_for_id(q, i, &len);
        printf("%.*s ", (int)len, n);
      }
      printf(")\n    captures actually stored on the step: ");
      const QueryStep *s = array_get(&q->steps, 0);
      for (unsigned i = 0; i < MAX_STEP_CAPTURE_COUNT; i++) {
        if (s->capture_ids[i] == NONE) break;
        uint32_t len; const char *n = ts_query_capture_name_for_id(q, s->capture_ids[i], &len);
        printf("%.*s ", (int)len, n);
      }
      printf("\n    >>> @d silently dropped at compile time (drops counted: %lu)\n", qprobe.capture_drops);

      // Show it at runtime too.
      TSParser *p = ts_parser_new();
      ts_parser_set_language(p, rust);
      const char *src = "fn foo() {}";
      TSTree *t = ts_parser_parse_string(p, NULL, src, (uint32_t)strlen(src));
      TSQueryCursor *c = ts_query_cursor_new();
      ts_query_cursor_exec(c, q, ts_tree_root_node(t));
      TSQueryMatch m;
      while (ts_query_cursor_next_match(c, &m)) {
        printf("    runtime match has %u captures (expected 4)\n", m.capture_count);
      }
      ts_query_cursor_delete(c); ts_tree_delete(t); ts_parser_delete(p);
      ts_query_delete(q);
    }
  }

  printf("\n### BUG 2: `MISSING` is matched by prefix, so any prefix of it is treated as MISSING\n");
  {
    // "MISS" is not a Rust node type. It should be a NodeType error.
    // But strncmp(node_name, \"MISSING\", 4) == 0, so it is parsed as (MISSING _).
    TSQuery *a = compile(rust, "(MISS)", "(MISS)          [not a node type]");
    TSQuery *b = compile(rust, "(M)", "(M)             [not a node type]");
    TSQuery *c = compile(rust, "(NOPE)", "(NOPE)          [not a node type, control]");
    TSQuery *d = compile(rust, "(MISSING)", "(MISSING)       [genuine]");
    printf("    >>> (MISS) and (M) compile because strncmp() only compares `length` bytes;\n");
    printf("        they silently become `(MISSING _)`. (NOPE) correctly errors.\n");
    if (a) ts_query_delete(a);
    if (b) ts_query_delete(b);
    if (c) ts_query_delete(c);
    if (d) ts_query_delete(d);
  }

  printf("\n### BUG 3: MAX_NEGATED_FIELD_COUNT = 8 silently ignores the 9th negated field\n");
  {
    // Rust field names, 9 distinct ones on one node.
    const char *src9 =
      "(call_expression !function !arguments !type !name !value !pattern !body !alias !left) @x";
    TSQuery *q = compile(rust, src9, "9 negated fields on one node");
    if (q) {
      const QueryStep *s = array_get(&q->steps, 0);
      unsigned n = 0;
      if (s->negated_field_list_id) {
        const TSFieldId *f = array_get(&q->negated_fields, s->negated_field_list_id);
        while (*f) { n++; f++; }
      }
      printf("    negated fields actually recorded: %u (expected 9)\n", n);
      printf("    >>> the 9th is discarded with no diagnostic\n");
      ts_query_delete(q);
    }
  }

  printf("\n### BUG 4: capture list pool exhaustion drops matches silently by default?\n");
  {
    TSQuery *q = compile(rust, "((line_comment)* @doc (function_item) @fn)", "doc-run pattern");
    if (q) {
      TSParser *p = ts_parser_new();
      ts_parser_set_language(p, rust);
      char buf[64000]; buf[0] = 0;
      for (int i = 0; i < 200; i++) strcat(buf, "// c\n");
      strcat(buf, "fn a() {}\n");
      TSTree *t = ts_parser_parse_string(p, NULL, buf, (uint32_t)strlen(buf));
      for (uint32_t limit = 1; limit <= 32; limit *= 4) {
        TSQueryCursor *c = ts_query_cursor_new();
        ts_query_cursor_set_match_limit(c, limit);
        ts_query_cursor_exec(c, q, ts_tree_root_node(t));
        unsigned long n = 0; TSQueryMatch m;
        while (ts_query_cursor_next_match(c, &m)) n++;
        printf("    match_limit=%-3u -> %lu matches, did_exceed=%d\n",
               limit, n, ts_query_cursor_did_exceed_match_limit(c));
        ts_query_cursor_delete(c);
      }
      ts_tree_delete(t); ts_parser_delete(p); ts_query_delete(q);
      printf("    >>> results depend on match_limit; the default is UINT32_MAX (unbounded memory)\n");
    }
  }

  return 0;
}
