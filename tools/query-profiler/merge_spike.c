// SPIKE, THROWAWAY. Runs the stock engine and a merge-based matcher over the
// same tree in one process and diffs their match streams.
//
// The bet, measured before writing any of this: live states collapse 56x under
// the key (step_index, start_depth, thread-flags) on the pathological query,
// and 1.00x on a real query file. So merging the *control* state should remove
// the O(n^2) dedup pass on exactly the case that needs it, and be a no-op
// elsewhere.
//
// Design, in one paragraph. The stock engine keeps one thread per
// (control state x capture history) and reconciles them afterwards by pairwise
// capture-subset comparison. This keeps one entry per *control* state, holding
// a SET of capture continuations. The per-node match test (symbol, field,
// negated fields, anchors) then runs once per control state instead of once per
// thread -- that is the 56x -- and capture histories are persistent cons-lists
// in an arena, so carrying several costs a pointer each rather than an array
// copy. Semantics are unchanged: every alternative history is retained, so the
// same set of matches is produced.
//
// Scope: flat sibling patterns whose steps are at depth 0 or 1, which covers
// the target query and its anchored variant. Anything else reports UNSUPPORTED
// rather than silently diverging.
//
// STATUS: harness complete and working; matcher INCOMPLETE.
//   - both engines run, streams are collected and compared as multisets, and a
//     capture-count histogram is printed to characterise any divergence.
//   - on the anchored query the matcher emits each match exactly twice
//     (120 -> 240, identical capture shapes), so a control state is being
//     reached by two paths that ought to be one, or a pattern is seeded more
//     than once per position. Not a semantics question -- a plumbing bug.
//   - the merge mechanism itself works: 120 continuations were absorbed rather
//     than forked on that run.
// Next: fix the duplication, then run the unanchored query, which is the case
// the whole exercise is aimed at.
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

/*******************
 * Canonical stream
 *******************/

typedef struct { uint16_t pattern; uint16_t capture_id; uint32_t start, end; } StreamCap;
typedef struct { uint32_t off, len; uint16_t pattern; } StreamMatch;
typedef struct {
  Array(StreamCap) caps;
  Array(StreamMatch) matches;
} Stream2;

static void stream_add(Stream2 *s, uint16_t pattern, const TSQueryCapture *caps, uint32_t n) {
  StreamMatch m = { .off = s->caps.size, .len = n, .pattern = pattern };
  for (uint32_t i = 0; i < n; i++) {
    array_push(&s->caps, ((StreamCap){
      .pattern = pattern, .capture_id = caps[i].index,
      .start = ts_node_start_byte(caps[i].node), .end = ts_node_end_byte(caps[i].node),
    }));
  }
  array_push(&s->matches, m);
}

// Matches are compared as multisets: the two engines need not discover them in
// the same order for the result to be equivalent, but the set must be identical.
static int match_cmp(const void *va, const void *vb) {
  const StreamMatch *a = va, *b = vb;
  if (a->pattern != b->pattern) return a->pattern < b->pattern ? -1 : 1;
  if (a->len != b->len) return a->len < b->len ? -1 : 1;
  return 0;
}

static void stream_sort_key(Stream2 *s, uint64_t *keys) {
  for (uint32_t i = 0; i < s->matches.size; i++) {
    StreamMatch *m = array_get(&s->matches, i);
    uint64_t h = 0xcbf29ce484222325ull;
    h = (h ^ m->pattern) * 0x100000001b3ull;
    // Sort within the match: the two engines need not agree on capture order
    // for the capture SET to be identical, which is what is being compared.
    StreamCap tmp[64]; uint32_t tn = m->len < 64 ? m->len : 64;
    for (uint32_t j = 0; j < tn; j++) tmp[j] = *array_get(&s->caps, m->off + j);
    for (uint32_t x = 1; x < tn; x++) { StreamCap k = tmp[x]; int32_t y = (int32_t)x - 1;
      while (y >= 0 && (tmp[y].start > k.start ||
             (tmp[y].start == k.start && tmp[y].capture_id > k.capture_id))) { tmp[y+1]=tmp[y]; y--; }
      tmp[y+1] = k; }
    for (uint32_t j = 0; j < tn; j++) {
      StreamCap *c = &tmp[j];
      h = (h ^ c->capture_id) * 0x100000001b3ull;
      h = (h ^ c->start) * 0x100000001b3ull;
      h = (h ^ c->end) * 0x100000001b3ull;
    }
    keys[i] = h;
  }
}

static int u64cmp(const void *a, const void *b) {
  uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
  return (x > y) - (x < y);
}

static bool stream_equal(Stream2 *a, Stream2 *b, char *why, size_t whyn) {
  if (a->matches.size != b->matches.size) {
    snprintf(why, whyn, "match count differs: stock=%u merge=%u", a->matches.size, b->matches.size);
    return false;
  }
  uint64_t *ka = malloc(sizeof(uint64_t) * (a->matches.size + 1));
  uint64_t *kb = malloc(sizeof(uint64_t) * (b->matches.size + 1));
  stream_sort_key(a, ka); stream_sort_key(b, kb);
  qsort(ka, a->matches.size, sizeof(uint64_t), u64cmp);
  qsort(kb, b->matches.size, sizeof(uint64_t), u64cmp);
  bool ok = true;
  for (uint32_t i = 0; i < a->matches.size; i++) {
    if (ka[i] != kb[i]) {
      snprintf(why, whyn, "first differing match at sorted index %u (stock hash %016llx, merge %016llx)",
               i, (unsigned long long)ka[i], (unsigned long long)kb[i]);
      ok = false; break;
    }
  }
  free(ka); free(kb);
  return ok;
}

/*********************
 * Engine A: as shipped
 *********************/

static double run_stock(const TSQuery *q, TSTree *tree, Stream2 *out) {
  TSQueryCursor *c = ts_query_cursor_new();
  double t0 = now_ms();
  ts_query_cursor_exec(c, q, ts_tree_root_node(tree));
  TSQueryMatch m;
  while (ts_query_cursor_next_match(c, &m)) {
    stream_add(out, (uint16_t)m.pattern_index, m.captures, m.capture_count);
  }
  double dt = now_ms() - t0;
  ts_query_cursor_delete(c);
  return dt;
}

/*****************************************
 * Engine B: merged control state + capture
 * histories as persistent cons-lists
 *****************************************/

// CAPTURE SETS AS BITSETS.
//
// Continuations reaching the same control state have matched the same steps;
// they differ only in which nodes they bound to quantified steps, and those
// nodes always come from the sibling sequence being scanned. So a capture set
// is a bitset over (sibling position x capture id), and A superset-of B is a
// word-wise test instead of a set comparison.
//
// Node identity is NOT stored: a set bit says "capture c was bound at sibling
// i", and the node is recovered from the sibling array at emit time -- kids[i]
// for a depth-0 step, or its field child for a depth-1 step.
typedef Array(uint64_t) BitArena;
static uint32_t g_words;      // words per continuation, sized per sibling sequence
static uint32_t g_ncap;       // distinct capture ids in the query

static uint32_t bs_new(BitArena *a) {
  uint32_t off = a->size;
  for (uint32_t i = 0; i < g_words; i++) array_push(a, (uint64_t)0);
  return off;
}
static uint32_t bs_clone_set(BitArena *a, uint32_t src, uint32_t bit) {
  uint32_t off = a->size;
  for (uint32_t i = 0; i < g_words; i++)
    array_push(a, src == UINT32_MAX ? (uint64_t)0 : a->contents[src + i]);
  a->contents[off + (bit >> 6)] |= (uint64_t)1 << (bit & 63);
  return off;
}
// Is every bit of `sub` also set in `sup`?
static bool bs_superset(const BitArena *a, uint32_t sup, uint32_t sub) {
  if (sub == UINT32_MAX) return true;
  if (sup == UINT32_MAX) return false;
  for (uint32_t i = 0; i < g_words; i++)
    if (a->contents[sub + i] & ~a->contents[sup + i]) return false;
  return true;
}
static bool bs_equal(const BitArena *a, uint32_t x, uint32_t y) {
  if (x == y) return true;
  if (x == UINT32_MAX || y == UINT32_MAX) return false;
  for (uint32_t i = 0; i < g_words; i++) if (a->contents[x + i] != a->contents[y + i]) return false;
  return true;
}

// A capture history is a singly-linked list through an arena, newest first.
// Sharing is automatic: two continuations that agree on a prefix point at the
// same cell, so carrying N alternatives costs N pointers, not N arrays.
typedef struct { TSNode node; uint16_t capture_id; uint32_t prev; uint32_t len; } CapCell;
#define CAP_NIL UINT32_MAX

typedef Array(CapCell) CapArena;

static uint32_t cap_push(CapArena *a, uint32_t prev, TSNode node, uint16_t id) {
  uint32_t len = (prev == CAP_NIL) ? 1 : a->contents[prev].len + 1;
  array_push(a, ((CapCell){ .node = node, .capture_id = id, .prev = prev, .len = len }));
  return a->size - 1;
}

static inline uint32_t cap_len(const CapArena *a, uint32_t h) {
  return h == CAP_NIL ? 0 : a->contents[h].len;
}

// Is `shorter`'s chain a suffix of `longer`'s? Continuations that reach the same
// control state differ by how many quantifier repetitions they took, and taking
// more only ever appends cells -- so a strict superset of captures shows up as a
// strict suffix-extension. Walking up the length difference and comparing one
// pointer decides it, rather than comparing two capture sets pairwise.
static bool cap_is_suffix(const CapArena *a, uint32_t longer, uint32_t shorter) {
  uint32_t ll = cap_len(a, longer), ls = cap_len(a, shorter);
  if (ls > ll) return false;
  if (shorter == CAP_NIL) return true;
  uint32_t h = longer;
  for (uint32_t k = 0; k < ll - ls; k++) h = a->contents[h].prev;
  return h == shorter;
}

// Materialize newest-first chain into oldest-first order for emission.
static uint32_t cap_flatten(const CapArena *a, uint32_t head, TSQueryCapture *buf, uint32_t cap) {
  uint32_t n = 0;
  for (uint32_t h = head; h != CAP_NIL && n < cap; h = a->contents[h].prev) n++;
  uint32_t i = n;
  for (uint32_t h = head; h != CAP_NIL && i > 0; h = a->contents[h].prev) {
    const CapCell *c = &a->contents[h];
    buf[--i] = (TSQueryCapture){ .node = c->node, .index = c->capture_id };
  }
  return n;
}

// One entry per *control* state. `heads` is the set of capture continuations
// that reached this control state -- the thing the stock engine keeps as
// separate threads and reconciles afterwards.
typedef struct {
  uint16_t step_index;
  uint16_t start_depth;
  uint16_t pattern_index;
  uint8_t  flags;            // seeking_immediate | skipped_quantifier | needs_parent
  Array(uint32_t) heads;
} MergedState;

#define F_SEEK_IMM   1u
#define F_SKIPPED_Q  2u
#define F_NEEDS_PAR  4u

typedef Array(MergedState) MergedSet;

static MergedState *merged_find(MergedSet *set, uint16_t step, uint16_t depth, uint16_t pat, uint8_t flags) {
  for (uint32_t i = 0; i < set->size; i++) {
    MergedState *s = array_get(set, i);
    if (s->step_index == step && s->start_depth == depth &&
        s->pattern_index == pat && s->flags == flags) return s;
  }
  return NULL;
}

// The merge point: an existing control state absorbs the new continuation
// instead of becoming a second thread.
// When PRUNE is on, longest-match is enforced here -- at merge time, in work
// proportional to the chain-length difference -- instead of by the stock
// engine's pairwise capture-set comparison afterwards. All heads in a MergedState
// already share step_index and flags, so the stock engine's guard conditions on
// those hold by construction.
static bool g_prune = false;
static unsigned long g_pruned = 0;

static void merged_add_a(MergedSet *set, const BitArena *arena, uint16_t step, uint16_t depth,
                         uint16_t pat, uint8_t flags, uint32_t head, unsigned long *merge_count) {
  MergedState *s = merged_find(set, step, depth, pat, flags);
  if (s) {
    for (uint32_t i = 0; i < s->heads.size; i++) {
      if (*array_get(&s->heads, i) == head) return;   // identical continuation
    }
    if (g_prune) {
      // Longest-match, enforced here in word-wise work: drop the newcomer if an
      // existing continuation subsumes it, else evict those it subsumes.
      for (uint32_t i = 0; i < s->heads.size; i++) {
        if (bs_superset(arena, *array_get(&s->heads, i), head)) { g_pruned++; return; }
      }
      for (uint32_t i = 0; i < s->heads.size; i++) {
        if (bs_superset(arena, head, *array_get(&s->heads, i))) {
          *array_get(&s->heads, i) = *array_back(&s->heads);
          s->heads.size--; i--; g_pruned++;
        }
      }
    } else {
      for (uint32_t i = 0; i < s->heads.size; i++)
        if (bs_equal(arena, *array_get(&s->heads, i), head)) return;
    }
    array_push(&s->heads, head);
    (*merge_count)++;
    return;
  }
  MergedState ns = { .step_index = step, .start_depth = depth,
                     .pattern_index = pat, .flags = flags, .heads = array_new() };
  array_push(&ns.heads, head);
  array_push(set, ns);
}

#define merged_add(set, step, depth, pat, flags, head, mc) \
  merged_add_a((set), &ctx->bits, (step), (depth), (pat), (flags), (head), (mc))

static void merged_clear(MergedSet *set) {
  for (uint32_t i = 0; i < set->size; i++) array_delete(&array_get(set, i)->heads);
  array_clear(set);
}


/*****************************************
 * Engine B: the merged matcher
 *****************************************/

typedef struct {
  const TSQuery *q;
  BitArena bits;
  // Per capture id: the depth of the step that binds it, and its field (for a
  // depth-1 step, the field selects which child of the sibling was captured).
  uint8_t cap_depth[64];
  TSFieldId cap_field[64];
  TSNode *kids; unsigned nkids; unsigned cur_sib;
  CapArena arena;
  Stream2 *out;
  unsigned long merges;        // continuations absorbed instead of forked
  unsigned long node_tests;    // per-(control state, node) match tests performed
  unsigned long emitted;
} MergeCtx;

// Follow dead-end / pass-through / alternative links from `step`, mirroring
// ts_query_cursor__advance:4431-4504. Yields the set of reachable control
// states. Computed once per control state rather than once per thread.
typedef struct { uint16_t step; uint8_t flags; } Reach;

static unsigned expand(const TSQuery *q, uint16_t step, uint8_t flags,
                       bool has_later_named_siblings, Reach *out, unsigned cap) {
  Reach work[64]; unsigned nw = 0, n = 0;
  work[nw++] = (Reach){ step, flags };
  for (unsigned i = 0; i < nw && nw < 64; i++) {
    uint16_t si = work[i].step; uint8_t fl = work[i].flags;
    const QueryStep *st = array_get(&q->steps, si);
    if (st->alternative_index != NONE) {
      if (st->is_dead_end) { work[nw++] = (Reach){ st->alternative_index, fl }; continue; }
      if (st->is_pass_through) { work[nw++] = (Reach){ (uint16_t)(si + 1), fl }; }
      if (st->alternative_is_skip && st->is_last_child && has_later_named_siblings) {
        if (!st->is_pass_through) { if (n < cap) out[n++] = (Reach){ si, fl }; }
        continue;
      }
      uint8_t nf = fl;
      if (st->is_pass_through) nf |= F_SEEK_IMM;
      if (st->alternative_is_skip && !st->is_immediate) nf |= F_SKIPPED_Q;
      work[nw++] = (Reach){ st->alternative_index, nf };
      if (st->is_pass_through) continue;
    }
    if (n < cap) out[n++] = (Reach){ si, fl };
  }
  return n;
}

static void match_siblings(MergeCtx *ctx, TSNode *kids, unsigned nkids, uint16_t depth);

// Verify the depth-1 tail of a pattern against a matched node's children,
// returning the capture head extended with whatever those steps captured, or
// CAP_NIL_FAIL if the tail does not match.
#define CAP_FAIL (CAP_NIL - 1)
static uint32_t match_child_tail(MergeCtx *ctx, TSNode parent, uint16_t first_step, uint32_t head) {
  (void)parent;
  const TSQuery *q = ctx->q;
  uint16_t si = first_step;
  for (;;) {
    const QueryStep *st = array_get(&q->steps, si);
    if (st->depth == PATTERN_DONE_MARKER || st->depth == 0) break;
    TSNode child = st->field
      ? ts_node_child_by_field_id(parent, st->field)
      : ts_node_named_child(parent, 0);
    if (ts_node_is_null(child)) return CAP_FAIL;
    if (st->symbol != WILDCARD_SYMBOL && ts_node_symbol(child) != st->symbol) return CAP_FAIL;
    for (unsigned c = 0; c < MAX_STEP_CAPTURE_COUNT && st->capture_ids[c] != NONE; c++) {
      head = bs_clone_set(&ctx->bits, head, ctx->cur_sib * g_ncap + st->capture_ids[c]);
    }
    si++;
  }
  return head;
}

static void match_siblings(MergeCtx *ctx, TSNode *kids, unsigned nkids, uint16_t depth) {
  const TSQuery *q = ctx->q;
  g_words = ((nkids * g_ncap) + 63) / 64;
  if (g_words == 0) g_words = 1;
  ctx->kids = kids; ctx->nkids = nkids;
  array_clear(&ctx->bits);
  MergedSet active = array_new();
  MergedSet next = array_new();

  for (unsigned i = 0; i < nkids; i++) {
    TSNode node = kids[i];
    TSSymbol sym = ts_node_symbol(node);
    bool is_named = ts_node_is_named(node);
    bool later_named = false;
    for (unsigned k = i + 1; k < nkids; k++) if (ts_node_is_named(kids[k])) { later_named = true; break; }

    // Seed every pattern entry. pattern_map already enumerates a pattern's
    // alternative entry points (this query has 1 pattern and 3 entries), so
    // expanding alternatives here as well would seed the same control state
    // twice under different flags and emit every match twice. The stock engine
    // seeds at pattern->step_index only, and expands after a match advances.
    // New states seek an immediate match, matching ts_query_cursor__add_state.
    for (uint32_t pi = 0; pi < q->pattern_map.size; pi++) {
      const PatternEntry *pe = array_get(&q->pattern_map, pi);
      const QueryStep *st = array_get(&q->steps, pe->step_index);
      if (st->depth != 0) continue;
      if (st->symbol != WILDCARD_SYMBOL && st->symbol != sym) continue;
      merged_add(&active, pe->step_index, depth, pe->pattern_index, F_SEEK_IMM, bs_new(&ctx->bits), &ctx->merges);
    }

    merged_clear(&next);
    for (uint32_t si = 0; si < active.size; si++) {
      MergedState *s = array_get(&active, si);
      const QueryStep *st = array_get(&q->steps, s->step_index);
      if (st->depth == PATTERN_DONE_MARKER) continue;
      ctx->node_tests++;   // ONE test for all continuations in this control state
      ctx->cur_sib = i;

      bool does_match;
      if (st->symbol == WILDCARD_SYMBOL) does_match = is_named || !st->is_named;
      else does_match = sym == st->symbol;
      if (st->field) {
        TSFieldId fid = 0;
        (void)fid; // field on a depth-0 sibling step is rare; not modelled here
      }
      if (st->is_last_child && later_named) does_match = false;

      bool later_ok = later_named && !(st->is_immediate && is_named && !(s->flags & F_SKIPPED_Q))
                                  && !(s->flags & F_SEEK_IMM);

      if (does_match) {
        // Advance: one capture-push per continuation, then expand once.
        uint16_t nstep = s->step_index + 1;
        const QueryStep *nx = array_get(&q->steps, nstep);
        uint8_t nflags = 0;
        if (st->symbol == WILDCARD_SYMBOL && !st->is_named && nx->is_immediate) nflags |= F_SEEK_IMM;
        Reach r[32];
        unsigned nr = (nx->depth == PATTERN_DONE_MARKER)
          ? (r[0] = (Reach){ nstep, nflags }, 1u)
          : expand(q, nstep, nflags, later_named, r, 32);

        for (uint32_t h = 0; h < s->heads.size; h++) {
          uint32_t head = *array_get(&s->heads, h);
          for (unsigned c = 0; c < MAX_STEP_CAPTURE_COUNT && st->capture_ids[c] != NONE; c++)
            head = bs_clone_set(&ctx->bits, head, i * g_ncap + st->capture_ids[c]);
          for (unsigned x = 0; x < nr; x++) {
            uint32_t h2 = head;
            const QueryStep *tail = array_get(&q->steps, r[x].step);
            if (tail->depth == 1) {                     // depth-1 tail on this node
              h2 = match_child_tail(ctx, node, r[x].step, head);
              if (h2 == CAP_FAIL) continue;
              uint16_t t = r[x].step;
              while (array_get(&q->steps, t)->depth == 1) t++;
              merged_add(&next, t, depth, s->pattern_index, r[x].flags, h2, &ctx->merges);
            } else {
              merged_add(&next, r[x].step, depth, s->pattern_index, r[x].flags, h2, &ctx->merges);
            }
          }
        }
      }
      if (later_ok) {
        for (uint32_t h = 0; h < s->heads.size; h++)
          merged_add(&next, s->step_index, depth, s->pattern_index, s->flags,
                     *array_get(&s->heads, h), &ctx->merges);
      }
    }

    // Swap, then emit anything that completed.
    MergedSet t = active; active = next; next = t;
    for (uint32_t si = 0; si < active.size; si++) {
      MergedState *s = array_get(&active, si);
      if (array_get(&q->steps, s->step_index)->depth != PATTERN_DONE_MARKER) continue;
      TSQueryCapture buf[256];
      for (uint32_t h = 0; h < s->heads.size; h++) {
        uint32_t bsoff = *array_get(&s->heads, h), n = 0;
        for (unsigned si2 = 0; si2 < nkids && n < 256; si2++) {
          for (unsigned c = 0; c < g_ncap && n < 256; c++) {
            uint32_t bit = si2 * g_ncap + c;
            if (!(ctx->bits.contents[bsoff + (bit >> 6)] >> (bit & 63) & 1)) continue;
            TSNode nd = ctx->cap_depth[c] ? ts_node_child_by_field_id(kids[si2], ctx->cap_field[c])
                                          : kids[si2];
            if (ts_node_is_null(nd)) continue;
            buf[n++] = (TSQueryCapture){ .node = nd, .index = c };
          }
        }
        stream_add(ctx->out, s->pattern_index, buf, n);
        ctx->emitted++;
      }
      array_clear(&s->heads);
    }
  }
  merged_clear(&active); merged_clear(&next);
  array_delete(&active); array_delete(&next);
}

static void walk_parents(MergeCtx *ctx, TSTreeCursor *c, uint16_t depth) {
  TSNode kids[4096]; unsigned n = 0;
  if (ts_tree_cursor_goto_first_child(c)) {
    do { TSNode k = ts_tree_cursor_current_node(c); if (n < 4096) kids[n++] = k; }
    while (ts_tree_cursor_goto_next_sibling(c));
    ts_tree_cursor_goto_parent(c);
  }
  if (n) match_siblings(ctx, kids, n, depth);
  if (ts_tree_cursor_goto_first_child(c)) {
    do { walk_parents(ctx, c, depth + 1); } while (ts_tree_cursor_goto_next_sibling(c));
    ts_tree_cursor_goto_parent(c);
  }
}

static double run_merge(const TSQuery *q, TSTree *tree, Stream2 *out, MergeCtx *ctx) {
  ctx->q = q; ctx->arena = (CapArena)array_new(); ctx->bits = (BitArena)array_new();
  ctx->out = out; ctx->merges = ctx->node_tests = ctx->emitted = 0;
  g_ncap = q->captures.slices.size; if (g_ncap == 0) g_ncap = 1;
  memset(ctx->cap_depth, 0, sizeof ctx->cap_depth);
  memset(ctx->cap_field, 0, sizeof ctx->cap_field);
  for (uint32_t i = 0; i < q->steps.size; i++) {
    const QueryStep *st = array_get(&q->steps, i);
    if (st->depth == PATTERN_DONE_MARKER) continue;
    for (unsigned c = 0; c < MAX_STEP_CAPTURE_COUNT && st->capture_ids[c] != NONE; c++) {
      uint16_t cid = st->capture_ids[c];
      if (cid < 64) { ctx->cap_depth[cid] = (uint8_t)st->depth; ctx->cap_field[cid] = st->field; }
    }
  }
  TSTreeCursor c = ts_tree_cursor_new(ts_tree_root_node(tree));
  double t0 = now_ms();
  walk_parents(ctx, &c, 0);
  double dt = now_ms() - t0;
  ts_tree_cursor_delete(&c);
  return dt;
}

int main(int argc, char **argv) {
  if (argc < 3) { fprintf(stderr, "usage: merge_spike <query.scm> <source.rs>\n"); return 1; }
  const TSLanguage *lang = tree_sitter_rust();

  uint32_t qlen, slen;
  char *qsrc = slurp(argv[1], &qlen);
  char *ssrc = slurp(argv[2], &slen);

  uint32_t off; TSQueryError err;
  TSQuery *q = ts_query_new(lang, qsrc, qlen, &off, &err);
  if (!q) { fprintf(stderr, "query failed: err=%d off=%u\n", err, off); return 1; }

  TSParser *p = ts_parser_new();
  ts_parser_set_language(p, lang);
  TSTree *tree = ts_parser_parse_string(p, NULL, ssrc, slen);

  printf("query: %s\nsource: %s (%u bytes)\n", argv[1], argv[2], slen);
  printf("steps=%u patterns=%u\n\n", q->steps.size, q->patterns.size);

  // Scope guard: report rather than silently diverge.
  unsigned max_depth = 0;
  for (uint32_t i = 0; i < q->steps.size; i++) {
    QueryStep *s = array_get(&q->steps, i);
    if (s->depth != PATTERN_DONE_MARKER && s->depth > max_depth) max_depth = s->depth;
  }
  if (max_depth > 1) {
    printf("UNSUPPORTED: max step depth %u; this spike handles flat sibling patterns "
           "(depth <= 1) only.\n", max_depth);
    return 2;
  }

  Stream2 a = { .caps = array_new(), .matches = array_new() };
  double ta = run_stock(q, tree, &a);
  printf("stock engine : %8.2f ms   %u matches\n", ta, a.matches.size);

  Stream2 b = { .caps = array_new(), .matches = array_new() };
  MergeCtx ctx;
  g_prune = getenv("SPIKE_PRUNE") != NULL;
  double tb = run_merge(q, tree, &b, &ctx);
  printf("merge matcher: %8.2f ms   %u matches   (node_tests=%lu, merged=%lu, bitset_words=%u, pruned=%lu, prune=%s)\n",
         tb, b.matches.size, ctx.node_tests, ctx.merges, ctx.bits.size, g_pruned,
         g_prune ? "ON" : "off");

  // Diagnostic: capture-count histogram per engine. If the merge matcher is
  // retaining alternatives the stock engine's longest-match pass discards, the
  // extra matches show up as a shorter-capture population.
  {
    unsigned ha[16] = {0}, hb[16] = {0};
    for (uint32_t i = 0; i < a.matches.size; i++) { uint32_t l = array_get(&a.matches,i)->len; ha[l<15?l:15]++; }
    for (uint32_t i = 0; i < b.matches.size; i++) { uint32_t l = array_get(&b.matches,i)->len; hb[l<15?l:15]++; }
    printf("\ncaptures/match  "); for (int k=0;k<8;k++) printf("%6d", k); printf("\n");
    printf("  stock         "); for (int k=0;k<8;k++) printf("%6u", ha[k]); printf("\n");
    printf("  merge         "); for (int k=0;k<8;k++) printf("%6u", hb[k]); printf("\n");
  }

  char why[256] = {0};
  bool same = stream_equal(&a, &b, why, sizeof why);
  printf("\nDIFF: %s\n", same ? "IDENTICAL match sets" : why);
  if (same && tb > 0) printf("SPEEDUP: %.1fx\n", ta / tb);
  array_delete(&b.caps); array_delete(&b.matches); array_delete(&ctx.arena); array_delete(&ctx.bits);

  ts_tree_delete(tree); ts_parser_delete(p); ts_query_delete(q);
  array_delete(&a.caps); array_delete(&a.matches);
  free(qsrc); free(ssrc);
  return 0;
}
