#include "./query_lowerer.h"
#include "./language.h"

#include <string.h>

// No node, such as the pattern after the last pattern of a list. Also the
// first instruction of a block that has not been emitted.
#define QUERY_LOWERER_NONE UINT32_MAX

// How a thread that has matched a node, and has not matched the next one yet,
// may skip the siblings in between: which wait instruction it waits with, and
// whether that changes when it passes a gap between two patterns of a list
// that has no anchor, or a pattern that matches no nodes.
typedef enum {
  // Any sibling may be skipped.
  QueryEdgeAny,
  // Only anonymous siblings may be skipped, because every gap passed since the
  // last node matched has an anchor. A gap without one makes this `Any`.
  QueryEdgeAnchored,
  // No sibling may be skipped, because the last node matched was a `_` and
  // every gap passed since has an anchor. A gap without one makes this `Any`.
  QueryEdgeImmediate,
  // Only anonymous siblings may be skipped, because an anchor comes before the
  // first child of a list. Gaps do not change this: patterns that match no
  // nodes pass the anchor on to the first node that matches.
  QueryEdgeFirstAnchored,
  // No sibling may be skipped, to match the next repetition of a pattern.
  // Gaps do not change this.
  QueryEdgeRepeated,
} QueryEdge;

// The number of kinds of edge.
#define QUERY_EDGE_COUNT 5

// Whether a pattern or an atom can match no nodes, and if it can, whether
// every gap that such a match passes has an anchor. Where there is more than
// one way, the later kind wins, because waiting along `Any` also finds every
// node that waiting along `Anchored` would.
typedef enum {
  // It always matches at least one node.
  QueryEmptyMatchNever,
  // It can match no nodes, passing only gaps with anchors.
  QueryEmptyMatchAnchored,
  // It can match no nodes, passing a gap without an anchor.
  QueryEmptyMatchUnanchored,
} QueryEmptyMatch;

// A piece of a pattern's code. Each block is emitted once, and then reached by
// jumps and splits, or by falling through from the instruction before it. A
// block that would only fall through to another block emits nothing, and
// starts where that block does.
typedef enum {
  // A node's tests and captures, followed by its children, if it has any.
  QueryBlockTests,
  // What comes after a node's last child: a `Last` if the list ends with an
  // anchor, and then the exit of the node's pattern.
  QueryBlockAfterChildren,
  // A wait for a node, followed by its tests. A node has one for each wait
  // opcode.
  QueryBlockWait,
  // Waits along an edge for the first node of a pattern in a list, or of one
  // of the patterns after it in the list.
  QueryBlockEnterList,
  // Waits along an edge for the first node of one of an alternation's
  // patterns.
  QueryBlockEnterAlternation,
  // What can come after a match of a pattern, along an edge from its last
  // node: another repetition of it, the next patterns of its list, or what
  // comes after the pattern that contains it.
  QueryBlockExit,
  // The end of a match.
  QueryBlockAccept,
} QueryBlockKind;

// TODO: tighter packing?
typedef struct {
  QueryBlockKind kind;
  // The node that the block belongs to: a node for `Tests`, `AfterChildren`,
  // and `Wait`, an alternation for `EnterAlternation`, a pattern for
  // `EnterList` and `Exit`, and the pattern's root for `Accept`.
  uint32_t node;
  // The QueryOpcode of a `Wait`, or the QueryEdge of an `Enter` or `Exit`.
  uint32_t variant;
} QueryBlock;

// An instruction whose operand is the first instruction of a block, which is
// filled in once every block of the pattern has been emitted.
typedef struct {
  uint32_t pc;
  QueryBlock block;
} QueryLowererFixup;

// The most blocks that one node of a pattern can have: a pattern's five
// `EnterList` and three `Exit` blocks.
#define QUERY_LOWERER_BLOCK_SLOTS 8

// What the lowerer records about one node of the pattern that it is lowering.
typedef struct {
  // The node whose child this one is, or NONE for the pattern's root.
  uint32_t parent;
  // For a named node, a group, or an alternation, its first child pattern,
  // which the parser requires a group or alternation to have.
  uint32_t first_child;
  // For a pattern in a list, the next pattern in the list.
  uint32_t next;
  // For a pattern, the nearest pattern around it, through groups and
  // alternations, with a field or a capture, which its nodes also get.
  uint32_t outer_decorated;
  // For a pattern or an atom, the depth of the nodes that it matches, counted
  // from the depth of the pattern's root.
  uint32_t depth;
  // The first instruction of each of the node's blocks, by slot.
  uint32_t pcs[QUERY_LOWERER_BLOCK_SLOTS];
  // For a pattern or an atom, how it can match no nodes, and for a pattern in
  // a list, how it and the rest of the list can. Each is a QueryEmptyMatch.
  uint8_t empty_match : 2;
  uint8_t rest_empty_match : 2;
  // For a pattern in a list, whether an anchor comes before the next pattern.
  bool is_anchored_after : 1;
  // For a named node, whether an anchor comes before its first child pattern,
  // and whether one comes after its last.
  bool has_leading_anchor : 1;
  bool has_trailing_anchor : 1;
  // For a pattern, whether it can repeat.
  bool is_repeated : 1;
  // For an alternation, whether each of its patterns is a node.
  bool has_only_nodes : 1;
  // For a pattern, whether it has a field or a capture.
  bool is_decorated : 1;
  // For a pattern, whether it can contain the first node of a match.
  bool can_start : 1;
  // For a pattern at the top level, whether a match can go on to another node
  // at the top level after it: because it repeats, because a pattern comes
  // after it in a group, or because the same is true of the pattern around it.
  bool can_continue : 1;
} QueryLowererNode;

// The state of one call to `ts_query_lower`.
typedef struct {
  const TSLanguage *language;
  const QueryTokenList *tokens;
  const QueryAst *ast;
  const QueryAstIds *ids;
  QueryProgram *program;
  // Receives a description of the error that stops lowering.
  QueryDiagnostic *error;
  // The index of the pattern being lowered, and its range of nodes.
  uint32_t pattern;
  uint32_t start;
  uint32_t end;
  // What has been recorded about each node of the pattern, by its index from
  // `start`.
  Array(QueryLowererNode) nodes;
  // The first instruction of the pattern's `Accept` block.
  uint32_t accept_pc;
  // Blocks that have been referred to, and may not have been emitted yet, in
  // the order of the first reference, and the index of the next one to emit.
  Array(QueryBlock) pending;
  uint32_t pending_index;
  // Blocks that start at the next instruction, and have not emitted any yet.
  Array(QueryBlock) chain;
  // Instructions whose operands are blocks that may not have been emitted yet.
  Array(QueryLowererFixup) fixups;
  // The atoms whose nodes can be the first node of a match of the pattern.
  Array(uint32_t) entry_atoms;
  // For each instruction of the pattern, whether a thread there can still
  // capture a node.
  Array(bool) may_capture;
  // Whether the pattern needs an operand that is too large for an instruction.
  bool is_too_large;
#ifdef DEBUG_QUERY_BLOCKS
  // The pattern's blocks, in the order that they were placed.
  Array(QueryBlock) placed;
#endif
} QueryLowerer;

static QueryLowererNode *query_lowerer_node(QueryLowerer *self, uint32_t node) {
  return array_get(&self->nodes, node - self->start);
}

static QueryAstKind query_lowerer_kind(const QueryLowerer *self, uint32_t node) {
  return query_ast_kind(self->ast, node);
}

// Returns the atom of the pattern `pattern`, which follows its field, if it has
// one.
static uint32_t query_lowerer_atom(const QueryLowerer *self, uint32_t pattern) {
  const uint32_t atom = pattern + 1;
  return query_lowerer_kind(self, atom) == QueryAstKindField ? atom + 1 : atom;
}

// Returns whether `node` is an atom that matches one node: a named node, an
// anonymous node, a wildcard, or a MISSING node.
static bool query_lowerer_is_node(const QueryLowerer *self, uint32_t node) {
  switch (query_lowerer_kind(self, node)) {
    case QueryAstKindNamedNode:
    case QueryAstKindAnonymousNode:
    case QueryAstKindWildcard:
    case QueryAstKindMissingNode:
      return true;
    default:
      return false;
  }
}

// Returns the pattern that contains `pattern` in a group or an alternation, or
// NONE if `pattern` is in the list of a named node, or is the root.
static uint32_t query_lowerer_outer_pattern(QueryLowerer *self, uint32_t pattern) {
  const uint32_t container = query_lowerer_node(self, pattern)->parent;
  if (container == QUERY_LOWERER_NONE) return QUERY_LOWERER_NONE;
  switch (query_lowerer_kind(self, container)) {
    case QueryAstKindGroup:
    case QueryAstKindAlternation:
      return query_lowerer_node(self, container)->parent;
    default:
      return QUERY_LOWERER_NONE;
  }
}

// Returns the opcode of the wait that a thread waits with along `edge`.
static QueryOpcode query_edge_wait_opcode(QueryEdge edge) {
  switch (edge) {
    case QueryEdgeAny: return QueryOpcodeWait;
    case QueryEdgeAnchored:
    case QueryEdgeFirstAnchored: return QueryOpcodeWaitAnchored;
    case QueryEdgeImmediate:
    case QueryEdgeRepeated: return QueryOpcodeWaitImmediate;
  }
  ts_unreachable();
}

// Returns `edge` after it passes a gap or an empty match, which is anchored or
// not.
static QueryEdge query_edge_pass(QueryEdge edge, bool is_anchored) {
  if (!is_anchored && (edge == QueryEdgeAnchored || edge == QueryEdgeImmediate)) return QueryEdgeAny;
  return edge;
}

// Returns how a pattern that can match no nodes as `first`, a gap that is
// anchored or not, and the rest of its list, which can as `rest`, can match no
// nodes together.
static QueryEmptyMatch query_empty_match_then(QueryEmptyMatch first, bool is_anchored, QueryEmptyMatch rest) {
  if (first == QueryEmptyMatchNever || rest == QueryEmptyMatchNever) return QueryEmptyMatchNever;
  if (first == QueryEmptyMatchAnchored && is_anchored && rest == QueryEmptyMatchAnchored) {
    return QueryEmptyMatchAnchored;
  }
  return QueryEmptyMatchUnanchored;
}

// Records each node's parent and depth, and links the child patterns of each
// named node and group. Each node is visited before its children.
static void query_lowerer_link(QueryLowerer *self) {
  for (uint32_t node = self->start; node < self->end; node++) {
    QueryLowererNode *info = query_lowerer_node(self, node);
    const QueryAstKind kind = query_lowerer_kind(self, node);
    if (kind == QueryAstKindPattern) {
      // The depth of a pattern's nodes must fit in an instruction.
      if (info->depth > QUERY_DEPTH_MAX) self->is_too_large = true;
      // A pattern's field comes before its atom, and its captures after.
      const uint32_t atom = query_lowerer_atom(self, node);
      info->is_decorated = atom != node + 1;
      for (uint32_t suffix = query_ast_end(self->ast, atom); suffix < query_ast_end(self->ast, node); suffix++) {
        if (query_lowerer_kind(self, suffix) == QueryAstKindCapture) info->is_decorated = true;
      }
      // Patterns in a group or an alternation are linked to the nearest one
      // around them with a field or a capture, so that collecting those for a
      // node skips the levels with neither.
      const uint32_t outer = query_lowerer_outer_pattern(self, node);
      if (outer != QUERY_LOWERER_NONE) {
        const QueryLowererNode *outer_info = query_lowerer_node(self, outer);
        info->outer_decorated = outer_info->is_decorated ? outer : outer_info->outer_decorated;
      }
    }

    // The patterns in a named node match its children, one level deeper.
    const uint32_t child_depth = kind == QueryAstKindNamedNode ? info->depth + 1 : info->depth;
    // Predicates and negated fields are not siblings, so the patterns on
    // either side of them are adjacent in the list.
    uint32_t previous = QUERY_LOWERER_NONE;
    bool is_anchored = false;
    for (uint32_t child = node + 1; child < query_ast_end(self->ast, node); child = query_ast_end(self->ast, child)) {
      QueryLowererNode *child_info = query_lowerer_node(self, child);
      child_info->parent = node;
      child_info->depth = child_depth;
      switch (query_lowerer_kind(self, child)) {
        case QueryAstKindAnchor:
          is_anchored = true;
          break;
        case QueryAstKindPattern:
          if (previous == QUERY_LOWERER_NONE) {
            info->first_child = child;
            info->has_leading_anchor = is_anchored;
          } else if (kind != QueryAstKindAlternation) {
            QueryLowererNode *previous_info = query_lowerer_node(self, previous);
            previous_info->next = child;
            previous_info->is_anchored_after = is_anchored;
          }
          previous = child;
          is_anchored = false;
          break;
        default:
          break;
      }
    }
    info->has_trailing_anchor = previous != QUERY_LOWERER_NONE && is_anchored;
  }
}

// Records which patterns and atoms can match no nodes, and which patterns
// repeat. Each node is visited after its descendants and the patterns after
// it in its list.
static void query_lowerer_measure(QueryLowerer *self) {
  for (uint32_t node = self->end; node > self->start;) {
    node--;
    QueryLowererNode *info = query_lowerer_node(self, node);
    switch (query_lowerer_kind(self, node)) {
      case QueryAstKindPattern: {
        const uint32_t atom = query_lowerer_atom(self, node);
        const QueryLowererNode *atom_info = query_lowerer_node(self, atom);
        // A pattern's quantifiers and captures are leaves after its atom.
        bool can_skip = false;
        for (uint32_t suffix = query_ast_end(self->ast, atom); suffix < query_ast_end(self->ast, node); suffix++) {
          if (query_lowerer_kind(self, suffix) != QueryAstKindQuantifier) continue;
          switch (query_token_list_kind(self->tokens, query_ast_token(self->ast, suffix))) {
            case QueryTokenKindQuestion: can_skip = true; break;
            case QueryTokenKindStar: can_skip = true; info->is_repeated = true; break;
            case QueryTokenKindPlus: info->is_repeated = true; break;
            // INVARIANT: The parser only makes quantifiers of these three tokens.
            default: ts_unreachable();
          }
        }
        // Skipping a pattern whose atom must match a node passes no gaps; a
        // pattern whose atom can match no nodes passes the gaps inside it.
        if (atom_info->empty_match != QueryEmptyMatchNever) {
          info->empty_match = atom_info->empty_match;
        } else {
          info->empty_match = can_skip ? QueryEmptyMatchAnchored : QueryEmptyMatchNever;
        }
        info->rest_empty_match = info->next == QUERY_LOWERER_NONE
          ? info->empty_match
          : query_empty_match_then(
              info->empty_match,
              info->is_anchored_after,
              query_lowerer_node(self, info->next)->rest_empty_match
            );
        break;
      }
      case QueryAstKindGroup:
        info->empty_match = query_lowerer_node(self, info->first_child)->rest_empty_match;
        break;
      case QueryAstKindAlternation:
        info->has_only_nodes = true;
        for (uint32_t branch = node + 1; branch < query_ast_end(self->ast, node); branch = query_ast_end(self->ast, branch)) {
          const QueryLowererNode *branch_info = query_lowerer_node(self, branch);
          // The later kind of empty match wins.
          if (branch_info->empty_match > info->empty_match) info->empty_match = branch_info->empty_match;
          if (!query_lowerer_is_node(self, query_lowerer_atom(self, branch))) info->has_only_nodes = false;
        }
        break;
      default:
        break;
    }
  }
}

// Finds the atoms whose nodes can be the first node of a match of the pattern:
// the root's atom, an alternation's patterns, a group's first pattern, and the
// pattern after one that can match no nodes. Nodes inside a named node never
// can.
static void query_lowerer_find_entries(QueryLowerer *self) {
  array_clear(&self->entry_atoms);
  query_lowerer_node(self, self->start)->can_start = true;
  for (uint32_t node = self->start; node < self->end; node++) {
    if (query_lowerer_kind(self, node) != QueryAstKindPattern) continue;
    QueryLowererNode *info = query_lowerer_node(self, node);
    if (info->depth == 0) {
      const uint32_t outer = query_lowerer_outer_pattern(self, node);
      info->can_continue =
        info->is_repeated ||
        info->next != QUERY_LOWERER_NONE ||
        (outer != QUERY_LOWERER_NONE && query_lowerer_node(self, outer)->can_continue);
    }
    if (!info->can_start) continue;
    if (info->empty_match != QueryEmptyMatchNever && info->next != QUERY_LOWERER_NONE) {
      query_lowerer_node(self, info->next)->can_start = true;
    }
    const uint32_t atom = query_lowerer_atom(self, node);
    switch (query_lowerer_kind(self, atom)) {
      case QueryAstKindGroup:
        query_lowerer_node(self, query_lowerer_node(self, atom)->first_child)->can_start = true;
        break;
      case QueryAstKindAlternation:
        for (uint32_t branch = atom + 1; branch < query_ast_end(self->ast, atom); branch = query_ast_end(self->ast, branch)) {
          query_lowerer_node(self, branch)->can_start = true;
        }
        break;
      default:
        array_push(&self->entry_atoms, atom);
        break;
    }
  }
}

// Returns where the first instruction of `block` is recorded. Each kind of node
// has its own kinds of block, which share the node's slots: a node has its
// `Tests`, its `AfterChildren`, and a `Wait` for each wait opcode, an
// alternation has an `EnterAlternation` for each edge, and a pattern has an
// `EnterList` for each edge and an `Exit` for the first three.
static uint32_t *query_lowerer_block_pc(QueryLowerer *self, QueryBlock block) {
  switch (block.kind) {
    case QueryBlockAccept:
      return &self->accept_pc;
    case QueryBlockTests:
      return &query_lowerer_node(self, block.node)->pcs[0];
    case QueryBlockAfterChildren:
      return &query_lowerer_node(self, block.node)->pcs[1];
    case QueryBlockWait:
      // The wait opcodes are consecutive.
      return &query_lowerer_node(self, block.node)->pcs[2 + block.variant - QueryOpcodeWait];
    case QueryBlockEnterList:
    case QueryBlockEnterAlternation:
      return &query_lowerer_node(self, block.node)->pcs[block.variant];
    case QueryBlockExit:
      // INVARIANT: A thread only exits a pattern after matching a node, so
      // only these three edges reach an exit.
      ts_assert(block.variant <= QueryEdgeImmediate);
      return &query_lowerer_node(self, block.node)->pcs[QUERY_EDGE_COUNT + block.variant];
  }
  ts_unreachable();
}

// Emits an instruction. An operand that does not fit in an instruction, or an
// instruction past the last one that an operand can refer to, makes the
// pattern too large, and the pattern fails before its program is used.
static void query_lowerer_emit(QueryLowerer *self, QueryOpcode opcode, uint32_t operand) {
  if (operand > QUERY_OPERAND_MAX || self->program->instructions.size > QUERY_OPERAND_MAX) {
    self->is_too_large = true;
    operand = 0;
  }
  array_push(&self->program->instructions, query_instruction_new(opcode, operand));
}

// Emits an instruction whose operand is the first instruction of `block`.
static void query_lowerer_emit_reference(QueryLowerer *self, QueryOpcode opcode, QueryBlock block) {
  array_push(&self->fixups, ((QueryLowererFixup) {
    .pc = self->program->instructions.size,
    .block = block,
  }));
  if (*query_lowerer_block_pc(self, block) == QUERY_LOWERER_NONE) array_push(&self->pending, block);
  query_lowerer_emit(self, opcode, 0);
}

// Returns the block that waits along `edge` for the first node of `atom`.
static QueryBlock query_lowerer_enter_atom(QueryLowerer *self, uint32_t atom, QueryEdge edge) {
  switch (query_lowerer_kind(self, atom)) {
    case QueryAstKindGroup:
      return (QueryBlock) {
        .kind = QueryBlockEnterList,
        .node = query_lowerer_node(self, atom)->first_child,
        .variant = edge,
      };
    case QueryAstKindAlternation:
      return (QueryBlock) {.kind = QueryBlockEnterAlternation, .node = atom, .variant = edge};
    default:
      return (QueryBlock) {.kind = QueryBlockWait, .node = atom, .variant = query_edge_wait_opcode(edge)};
  }
}

// Returns the exit of the pattern of `atom`, a node, after it matches.
static QueryBlock query_lowerer_exit_node(QueryLowerer *self, uint32_t atom) {
  // An anchor after a node lets only anonymous nodes be skipped, and one after
  // a `_` lets none be, so that in `_ . (x)` the `_` is the node just before
  // the `(x)`, whichever kind it is.
  const QueryEdge edge = query_lowerer_kind(self, atom) == QueryAstKindWildcard
    ? QueryEdgeImmediate
    : QueryEdgeAnchored;
  return (QueryBlock) {
    .kind = QueryBlockExit,
    .node = query_lowerer_node(self, atom)->parent,
    .variant = edge,
  };
}

// Emits splits to every block in `blocks` but the first, and returns the first,
// which the current thread continues with.
static QueryBlock query_lowerer_emit_branches(QueryLowerer *self, const QueryBlock *blocks, uint32_t count) {
  for (uint32_t i = 1; i < count; i++) query_lowerer_emit_reference(self, QueryOpcodeSplit, blocks[i]);
  return blocks[0];
}

// Returns the field that the node of `atom` must be in: the field of its
// pattern, or of the nearest pattern around it, through groups and
// alternations, that has one. Returns zero if there is none.
static TSFieldId query_lowerer_field(QueryLowerer *self, uint32_t atom) {
  for (
    uint32_t pattern = query_lowerer_node(self, atom)->parent;
    pattern != QUERY_LOWERER_NONE;
    pattern = query_lowerer_node(self, pattern)->outer_decorated
  ) {
    if (query_lowerer_kind(self, pattern + 1) == QueryAstKindField) {
      return (TSFieldId)query_ast_ids_get(self->ids, pattern + 1);
    }
  }
  return 0;
}

// Emits the tests and captures of `atom`, a node: its own, and those of the
// groups and alternations around it, up to the nearest named node. Returns the
// block that follows, which is its children, if it has any, or the exit of its
// pattern.
static QueryBlock query_lowerer_emit_tests(QueryLowerer *self, uint32_t atom) {
  const uint32_t id = query_ast_ids_get(self->ids, atom);
  const uint32_t end = query_ast_end(self->ast, atom);
  // A test of the node's symbol, if it has one, comes first, because an entry
  // is found by the symbol of its first test.
  switch (query_lowerer_kind(self, atom)) {
    case QueryAstKindNamedNode:
      if (id == 0) {
        // `(_)` is any named node but an ERROR.
        query_lowerer_emit(self, QueryOpcodeNamed, 0);
        query_lowerer_emit(self, QueryOpcodeNotError, 0);
      } else if (ts_language_symbol_metadata(self->language, (TSSymbol)id).supertype) {
        // A subtype is the first child of its named node. Without one, a
        // supertype is any node that it wraps but an ERROR.
        if (atom + 1 < end && query_lowerer_kind(self, atom + 1) == QueryAstKindSubtype) {
          query_lowerer_emit(self, QueryOpcodeSymbol, query_ast_ids_get(self->ids, atom + 1));
          query_lowerer_emit(self, QueryOpcodeSupertype, id);
        } else {
          query_lowerer_emit(self, QueryOpcodeSupertype, id);
          query_lowerer_emit(self, QueryOpcodeNotError, 0);
        }
      } else {
        query_lowerer_emit(self, QueryOpcodeSymbol, id);
      }
      break;
    case QueryAstKindAnonymousNode:
      query_lowerer_emit(self, QueryOpcodeSymbol, id);
      break;
    case QueryAstKindWildcard:
      // `_` is any node but an ERROR, named or not.
      query_lowerer_emit(self, QueryOpcodeNotError, 0);
      break;
    case QueryAstKindMissingNode:
      // The type after `MISSING`, if there is one, is its only child.
      if (atom + 1 < end) {
        query_lowerer_emit(self, QueryOpcodeSymbol, query_ast_ids_get(self->ids, atom + 1));
      }
      query_lowerer_emit(self, QueryOpcodeMissing, 0);
      break;
    default:
      ts_unreachable();
  }

  // The node's fields and captures are those of its pattern, and of the
  // patterns around it, through groups and alternations, that have any: a
  // field or a capture of a group or an alternation applies to each node at
  // its top level. The captures come after the tests, so that a thread only
  // records the node once it has passed them.
  for (
    uint32_t pattern = query_lowerer_node(self, atom)->parent;
    pattern != QUERY_LOWERER_NONE;
    pattern = query_lowerer_node(self, pattern)->outer_decorated
  ) {
    if (query_lowerer_kind(self, pattern + 1) == QueryAstKindField) {
      query_lowerer_emit(self, QueryOpcodeField, query_ast_ids_get(self->ids, pattern + 1));
    }
  }
  // A named node's negated fields are among its children.
  for (uint32_t child = atom + 1; child < end; child = query_ast_end(self->ast, child)) {
    if (query_lowerer_kind(self, child) == QueryAstKindNegatedField) {
      query_lowerer_emit(self, QueryOpcodeNoField, query_ast_ids_get(self->ids, child));
    }
  }
  for (
    uint32_t pattern = query_lowerer_node(self, atom)->parent;
    pattern != QUERY_LOWERER_NONE;
    pattern = query_lowerer_node(self, pattern)->outer_decorated
  ) {
    const uint32_t pattern_atom = query_lowerer_atom(self, pattern);
    for (uint32_t suffix = query_ast_end(self->ast, pattern_atom); suffix < query_ast_end(self->ast, pattern); suffix++) {
      if (query_lowerer_kind(self, suffix) == QueryAstKindCapture) {
        query_lowerer_emit(self, QueryOpcodeCapture, query_ast_ids_get(self->ids, suffix));
      }
    }
  }

  // Only a named node can have child patterns.
  const QueryLowererNode *info = query_lowerer_node(self, atom);
  if (info->first_child == QUERY_LOWERER_NONE) return query_lowerer_exit_node(self, atom);
  // If the node's child patterns can all match nothing, a copy of the thread
  // skips them and exits the node's pattern: with no child matched, a trailing
  // anchor has no node to apply to.
  if (query_lowerer_node(self, info->first_child)->rest_empty_match != QueryEmptyMatchNever) {
    query_lowerer_emit_reference(self, QueryOpcodeSplit, query_lowerer_exit_node(self, atom));
  }
  return (QueryBlock) {
    .kind = QueryBlockEnterList,
    .node = info->first_child,
    .variant = info->has_leading_anchor ? QueryEdgeFirstAnchored : QueryEdgeAny,
  };
}

// Emits the exit of `pattern` along `edge`, which is how the thread may skip
// siblings after the last node that it matched, and returns the block that the
// thread continues with.
static QueryBlock query_lowerer_emit_exit(QueryLowerer *self, uint32_t pattern, QueryEdge edge) {
  const QueryLowererNode *info = query_lowerer_node(self, pattern);
  // Another repetition, the next patterns of the list, and what comes after
  // the list or the pattern around this one: at most three, which run in that
  // order.
  QueryBlock options[3];
  uint32_t count = 0;
  if (info->is_repeated) {
    options[count++] = query_lowerer_enter_atom(self, query_lowerer_atom(self, pattern), QueryEdgeRepeated);
  }

  // A match of the root is a match of the whole pattern.
  const uint32_t container = info->parent;
  if (container == QUERY_LOWERER_NONE) {
    options[count++] = (QueryBlock) {.kind = QueryBlockAccept, .node = self->start};
    return query_lowerer_emit_branches(self, options, count);
  }
  // A match of one of an alternation's patterns is a match of the
  // alternation's pattern, and ends with the same node.
  if (query_lowerer_kind(self, container) == QueryAstKindAlternation) {
    options[count++] = (QueryBlock) {
      .kind = QueryBlockExit,
      .node = query_lowerer_node(self, container)->parent,
      .variant = edge,
    };
    return query_lowerer_emit_branches(self, options, count);
  }

  // In a list, the thread goes on to the next pattern through the gap after
  // this one. If the rest of the list can match nothing, it can also go through
  // all of it, to the end of the list.
  if (info->next != QUERY_LOWERER_NONE) {
    const QueryLowererNode *next_info = query_lowerer_node(self, info->next);
    edge = query_edge_pass(edge, info->is_anchored_after);
    options[count++] = (QueryBlock) {.kind = QueryBlockEnterList, .node = info->next, .variant = edge};
    if (next_info->rest_empty_match == QueryEmptyMatchNever) {
      return query_lowerer_emit_branches(self, options, count);
    }
    edge = query_edge_pass(edge, next_info->rest_empty_match == QueryEmptyMatchAnchored);
  }
  // At the end of a group's list, the group's pattern has matched. At the end
  // of a named node's list, all of the node's children have.
  if (query_lowerer_kind(self, container) == QueryAstKindGroup) {
    options[count++] = (QueryBlock) {
      .kind = QueryBlockExit,
      .node = query_lowerer_node(self, container)->parent,
      .variant = edge,
    };
  } else {
    options[count++] = (QueryBlock) {.kind = QueryBlockAfterChildren, .node = container};
  }
  return query_lowerer_emit_branches(self, options, count);
}

// Emits the instructions of `block`, and returns whether the thread continues
// with another block, `next`, after them.
static bool query_lowerer_emit_block(QueryLowerer *self, QueryBlock block, QueryBlock *next) {
  const QueryLowererNode *info = query_lowerer_node(self, block.node);
  switch (block.kind) {
    case QueryBlockTests:
      *next = query_lowerer_emit_tests(self, block.node);
      return true;
    case QueryBlockAfterChildren:
      // An anchor at the end of the list requires the last child that matched
      // to be the node's last named child. Then the node's pattern exits, as it
      // does after a node without children.
      if (info->has_trailing_anchor) query_lowerer_emit(self, QueryOpcodeLast, info->depth + 1);
      *next = query_lowerer_exit_node(self, block.node);
      return true;
    case QueryBlockWait: {
      const TSFieldId field = query_lowerer_field(self, block.node);
      query_lowerer_emit(self, (QueryOpcode)block.variant, query_wait_operand(info->depth, field));
      *next = (QueryBlock) {.kind = QueryBlockTests, .node = block.node};
      return true;
    }
    case QueryBlockEnterList: {
      const uint32_t atom = query_lowerer_atom(self, block.node);
      // If the pattern can match nothing, a copy of the thread skips it, which
      // passes the gaps inside it, if it has any, and the gap after it.
      // Skipping the last pattern of a list is offered earlier, as the end of
      // the list.
      if (info->empty_match != QueryEmptyMatchNever && info->next != QUERY_LOWERER_NONE) {
        QueryEdge edge = query_edge_pass((QueryEdge)block.variant, info->empty_match == QueryEmptyMatchAnchored);
        edge = query_edge_pass(edge, info->is_anchored_after);
        query_lowerer_emit_reference(self, QueryOpcodeSplit, ((QueryBlock) {
          .kind = QueryBlockEnterList,
          .node = info->next,
          .variant = edge,
        }));
      }
      *next = query_lowerer_enter_atom(self, atom, (QueryEdge)block.variant);
      return true;
    }
    case QueryBlockEnterAlternation: {
      const QueryEdge edge = (QueryEdge)block.variant;
      const uint32_t end = query_ast_end(self->ast, block.node);
      // Patterns that are all nodes share one wait, and then each tests the
      // node that it gives them, in a thread of its own.
      if (info->has_only_nodes) {
        // The wait has a field if the patterns' nodes must all be in the same
        // one.
        TSFieldId field = query_lowerer_field(self, query_lowerer_atom(self, info->first_child));
        for (uint32_t branch = query_ast_end(self->ast, info->first_child); branch < end; branch = query_ast_end(self->ast, branch)) {
          if (query_lowerer_field(self, query_lowerer_atom(self, branch)) != field) field = 0;
        }
        query_lowerer_emit(self, query_edge_wait_opcode(edge), query_wait_operand(info->depth, field));
        for (uint32_t branch = query_ast_end(self->ast, info->first_child); branch < end; branch = query_ast_end(self->ast, branch)) {
          query_lowerer_emit_reference(self, QueryOpcodeSplit, ((QueryBlock) {
            .kind = QueryBlockTests,
            .node = query_lowerer_atom(self, branch),
          }));
        }
        *next = (QueryBlock) {.kind = QueryBlockTests, .node = query_lowerer_atom(self, info->first_child)};
        return true;
      }
      // Otherwise each pattern waits for its own first node.
      for (uint32_t branch = query_ast_end(self->ast, info->first_child); branch < end; branch = query_ast_end(self->ast, branch)) {
        query_lowerer_emit_reference(self, QueryOpcodeSplit, query_lowerer_enter_atom(self, query_lowerer_atom(self, branch), edge));
      }
      *next = query_lowerer_enter_atom(self, query_lowerer_atom(self, info->first_child), edge);
      return true;
    }
    case QueryBlockExit:
      *next = query_lowerer_emit_exit(self, block.node, (QueryEdge)block.variant);
      return true;
    case QueryBlockAccept:
      query_lowerer_emit(self, QueryOpcodeAccept, 0);
      return false;
  }
  ts_unreachable();
}

// Emits `block`, which has not been emitted, and then each block that it falls
// through to, until one has been emitted before or ends the thread.
static void query_lowerer_emit_chain(QueryLowerer *self, QueryBlock block) {
  // A block starts at the next instruction when it is placed, so a block that
  // emits nothing starts where the block after it does. The blocks placed
  // before the chain has emitted anything are recorded, in case it never does.
  const uint32_t start = self->program->instructions.size;
  array_clear(&self->chain);
  for (;;) {
    uint32_t *pc = query_lowerer_block_pc(self, block);
    if (*pc != QUERY_LOWERER_NONE) break;
    *pc = self->program->instructions.size;
    if (*pc == start) array_push(&self->chain, block);
#ifdef DEBUG_QUERY_BLOCKS
    array_push(&self->placed, block);
#endif
    QueryBlock next;
    if (self->is_too_large || !query_lowerer_emit_block(self, block, &next)) return;
    block = next;
  }

  // The chain has reached a block that was emitted before.
  const uint32_t pc = *query_lowerer_block_pc(self, block);
  if (self->program->instructions.size == start) {
    // Every chain ends with a jump or an accept, so nothing falls through to
    // the start of this one, and its blocks can start where that block does.
    for (uint32_t i = 0; i < self->chain.size; i++) {
      *query_lowerer_block_pc(self, *array_get(&self->chain, i)) = pc;
    }
  } else if (query_instruction_opcode(*array_get(&self->program->instructions, pc)) == QueryOpcodeAccept) {
    // Falling through to a block that only accepts is the same as accepting.
    query_lowerer_emit(self, QueryOpcodeAccept, 0);
  } else {
    query_lowerer_emit_reference(self, QueryOpcodeJump, block);
  }
}

// Marks each of the pattern's waits where a thread can still capture a node,
// because a `Capture` can be reached from it. A loop can reach back to an
// earlier instruction, so what each instruction can reach is found by going
// backwards over the pattern until nothing changes.
static void query_lowerer_mark_waits(QueryLowerer *self, uint32_t instruction_offset) {
  const uint32_t count = self->program->instructions.size - instruction_offset;
  array_clear(&self->may_capture);
  array_grow_by(&self->may_capture, count);
  for (bool changed = true; changed;) {
    changed = false;
    for (uint32_t i = count; i > 0;) {
      i--;
      const QueryInstruction instruction = *array_get(&self->program->instructions, instruction_offset + i);
      bool may_capture;
      switch (query_instruction_opcode(instruction)) {
        case QueryOpcodeCapture: may_capture = true; break;
        case QueryOpcodeAccept: may_capture = false; break;
        case QueryOpcodeJump:
          may_capture = *array_get(&self->may_capture, query_instruction_target(instruction) - instruction_offset);
          break;
        case QueryOpcodeSplit:
          may_capture =
            *array_get(&self->may_capture, query_instruction_target(instruction) - instruction_offset) ||
            *array_get(&self->may_capture, i + 1);
          break;
        default: may_capture = *array_get(&self->may_capture, i + 1); break;
      }
      if (may_capture && !*array_get(&self->may_capture, i)) {
        *array_get(&self->may_capture, i) = true;
        changed = true;
      }
    }
  }
  for (uint32_t i = 0; i < count; i++) {
    QueryInstruction *instruction = array_get(&self->program->instructions, instruction_offset + i);
    if (query_opcode_is_wait(query_instruction_opcode(*instruction)) && *array_get(&self->may_capture, i)) {
      *instruction = query_instruction_with_may_capture(*instruction);
    }
  }
}

// Describes the error that stops lowering as being in the current pattern, and
// returns false.
static bool query_lowerer_fail(QueryLowerer *self, QueryDiagnosticKind kind) {
  // A pattern's tokens run up to the first token of the next one, or to the
  // `End` token.
  const uint32_t next_token = self->end < query_ast_size(self->ast)
    ? query_ast_token(self->ast, self->end)
    : query_token_list_size(self->tokens) - 1;
  *self->error = (QueryDiagnostic) {
    .span = {
      .start = query_token_list_span(self->tokens, query_ast_token(self->ast, self->start)).start,
      .end = query_token_list_span(self->tokens, next_token - 1).end,
    },
    .kind = kind,
  };
  return false;
}

#ifdef DEBUG_QUERY_BLOCKS
static const char *query_edge_name(QueryEdge edge) {
  switch (edge) {
    case QueryEdgeAny: return "Any";
    case QueryEdgeAnchored: return "Anchored";
    case QueryEdgeImmediate: return "Immediate";
    case QueryEdgeFirstAnchored: return "FirstAnchored";
    case QueryEdgeRepeated: return "Repeated";
  }
  ts_unreachable();
}

// Labels the start of each of the pattern's blocks in the program, in the
// order that the blocks were placed, so that blocks that start at the same
// instruction are listed in the order that they fall through to each other.
static void query_lowerer_label_blocks(QueryLowerer *self) {
  const uint32_t offset = self->program->labels.size;
  for (uint32_t i = 0; i < self->placed.size; i++) {
    const QueryBlock block = *array_get(&self->placed, i);
    QueryProgramLabel label = {
      .pc = *query_lowerer_block_pc(self, block),
      .kind = "?",
      .variant = NULL,
      .node = block.node,
      .span = query_token_list_span(self->tokens, query_ast_token(self->ast, block.node)),
    };
    switch (block.kind) {
      case QueryBlockTests: label.kind = "Tests"; break;
      case QueryBlockAfterChildren: label.kind = "AfterChildren"; break;
      case QueryBlockWait:
        label.kind = "Wait";
        label.variant =
          block.variant == QueryOpcodeWait ? "Wait" :
          block.variant == QueryOpcodeWaitAnchored ? "WaitAnchored" : "WaitImmediate";
        break;
      case QueryBlockEnterList:
        label.kind = "EnterList";
        label.variant = query_edge_name((QueryEdge)block.variant);
        break;
      case QueryBlockEnterAlternation:
        label.kind = "EnterAlternation";
        label.variant = query_edge_name((QueryEdge)block.variant);
        break;
      case QueryBlockExit:
        label.kind = "Exit";
        label.variant = query_edge_name((QueryEdge)block.variant);
        break;
      case QueryBlockAccept: label.kind = "Accept"; break;
    }
    // A pattern's token is its first one, which is often just a `(`, so it is
    // shown by its atom's token instead.
    if (block.kind == QueryBlockAccept) {
      label.span = (QuerySpan) {.start = 0, .end = 0};
    } else if (query_lowerer_kind(self, block.node) == QueryAstKindPattern) {
      label.span = query_token_list_span(self->tokens, query_ast_token(self->ast, query_lowerer_atom(self, block.node)));
    }
    // Insert it after every label that starts at or before the same pc.
    uint32_t index = self->program->labels.size;
    while (index > offset && array_get(&self->program->labels, index - 1)->pc > label.pc) index--;
    array_insert(&self->program->labels, index, label);
  }
  array_clear(&self->placed);
}
#endif

// Lowers the pattern whose root is the node `self.start`.
static bool query_lowerer_lower_pattern(QueryLowerer *self) {
  // Entries and matches store a pattern's index in 16 bits.
  if (self->pattern > UINT16_MAX) return query_lowerer_fail(self, QueryDiagnosticKindTooManyPatterns);
  array_clear(&self->nodes);
  array_grow_by(&self->nodes, self->end - self->start);
  for (uint32_t i = 0; i < self->nodes.size; i++) {
    QueryLowererNode *info = array_get(&self->nodes, i);
    info->parent = QUERY_LOWERER_NONE;
    info->first_child = QUERY_LOWERER_NONE;
    info->next = QUERY_LOWERER_NONE;
    info->outer_decorated = QUERY_LOWERER_NONE;
    memset(info->pcs, (unsigned char)QUERY_LOWERER_NONE, sizeof(info->pcs));
  }
  self->accept_pc = QUERY_LOWERER_NONE;
  array_clear(&self->pending);
  self->pending_index = 0;
  array_clear(&self->fixups);
  self->is_too_large = false;

  query_lowerer_link(self);
  query_lowerer_measure(self);
  query_lowerer_find_entries(self);

  const uint32_t instruction_offset = self->program->instructions.size;
  for (uint32_t i = 0; i < self->entry_atoms.size; i++) {
    array_push(&self->pending, ((QueryBlock) {.kind = QueryBlockTests, .node = *array_get(&self->entry_atoms, i)}));
  }
  // Once the program is too large, no more of it is emitted, because the
  // pattern fails anyway.
  while (!self->is_too_large && self->pending_index < self->pending.size) {
    const QueryBlock block = *array_get(&self->pending, self->pending_index);
    self->pending_index++;
    if (*query_lowerer_block_pc(self, block) == QUERY_LOWERER_NONE) query_lowerer_emit_chain(self, block);
  }

  if (self->is_too_large) return query_lowerer_fail(self, QueryDiagnosticKindPatternTooLarge);

  for (uint32_t i = 0; i < self->fixups.size; i++) {
    const QueryLowererFixup *fixup = array_get(&self->fixups, i);
    QueryInstruction *instruction = array_get(&self->program->instructions, fixup->pc);
    const uint32_t pc = *query_lowerer_block_pc(self, fixup->block);
    // INVARIANT: Every block that is referred to is emitted.
    ts_assert(pc != QUERY_LOWERER_NONE);
    *instruction = query_instruction_new(query_instruction_opcode(*instruction), pc);
  }
  query_lowerer_mark_waits(self, instruction_offset);

#ifdef DEBUG_QUERY_BLOCKS
  query_lowerer_label_blocks(self);
#endif

  // A match has another node at the top level only if it waits at depth zero,
  // and the program only does that to repeat a pattern at the top level, or to
  // go on to the next pattern of a group at the top level.
  const uint32_t entry_offset = self->program->entries.size;
  bool is_rooted = true;
  for (uint32_t i = 0; i < self->entry_atoms.size; i++) {
    const QueryLowererNode *atom = query_lowerer_node(self, *array_get(&self->entry_atoms, i));
    const uint32_t pc = atom->pcs[0];
    const QueryInstruction first = *array_get(&self->program->instructions, pc);
    const bool is_entry_rooted = !query_lowerer_node(self, atom->parent)->can_continue;
    is_rooted = is_rooted && is_entry_rooted;
    array_push(&self->program->entries, ((QueryEntry) {
      .pc = pc,
      .pattern = (uint16_t)self->pattern,
      .symbol = query_instruction_opcode(first) == QueryOpcodeSymbol
        ? query_instruction_symbol(first)
        : 0,
      .is_rooted = is_entry_rooted,
    }));
  }
  array_push(&self->program->patterns, ((QueryProgramPattern) {
    .instruction_offset = instruction_offset,
    .is_rooted = is_rooted,
    .entry_offset = entry_offset,
  }));
  return true;
}

bool ts_query_lower(
  const TSLanguage *language,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  const QueryAstIds *ids,
  QueryProgram *program,
  QueryDiagnostic *error
) {
  QueryLowerer self = {
    .language = language,
    .tokens = tokens,
    .ast = ast,
    .ids = ids,
    .program = program,
    .error = error,
    .nodes = array_new(),
    .pending = array_new(),
    .chain = array_new(),
    .fixups = array_new(),
    .entry_atoms = array_new(),
    .may_capture = array_new(),
#ifdef DEBUG_QUERY_BLOCKS
    .placed = array_new(),
#endif
  };
  bool succeeded = true;
  for (
    uint32_t pattern = 0;
    succeeded && pattern < query_ast_size(ast);
    pattern = query_ast_end(ast, pattern)
  ) {
    self.pattern = program->patterns.size;
    self.start = pattern;
    self.end = query_ast_end(ast, pattern);
    succeeded = query_lowerer_lower_pattern(&self);
  }
  array_delete(&self.nodes);
  array_delete(&self.pending);
  array_delete(&self.chain);
  array_delete(&self.fixups);
  array_delete(&self.entry_atoms);
  array_delete(&self.may_capture);
#ifdef DEBUG_QUERY_BLOCKS
  array_delete(&self.placed);
#endif
  return succeeded;
}

#ifdef DEBUG_QUERY_PROGRAM
#include <stdio.h>

static const char *query_opcode_name(QueryOpcode opcode) {
  switch (opcode) {
    case QueryOpcodeWait: return "Wait";
    case QueryOpcodeWaitAnchored: return "WaitAnchored";
    case QueryOpcodeWaitImmediate: return "WaitImmediate";
    case QueryOpcodeSymbol: return "Symbol";
    case QueryOpcodeSupertype: return "Supertype";
    case QueryOpcodeField: return "Field";
    case QueryOpcodeNoField: return "NoField";
    case QueryOpcodeNamed: return "Named";
    case QueryOpcodeNotError: return "NotError";
    case QueryOpcodeMissing: return "Missing";
    case QueryOpcodeLast: return "Last";
    case QueryOpcodeCapture: return "Capture";
    case QueryOpcodeSplit: return "Split";
    case QueryOpcodeJump: return "Jump";
    case QueryOpcodeAccept: return "Accept";
  }
  ts_unreachable();
}

static void query_program_dump_symbol(const TSLanguage *language, TSSymbol symbol) {
  const char *name = ts_language_symbol_name(language, symbol);
  if (ts_language_symbol_metadata(language, symbol).named) {
    printf("%s", name);
  } else {
    printf("\"%s\"", name);
  }
}

void ts_query_program_dump(
  const QueryProgram *self,
  const TSLanguage *language,
  const QueryTables *tables,
  const char *source
) {
  printf("=== PROGRAM ===\n");
  for (uint32_t i = 0; i < self->patterns.size; i++) {
    const QueryProgramPattern *pattern = array_get(&self->patterns, i);
    const QuerySpan span = array_get(&tables->patterns, i)->span;
    printf(
      "pattern %u%s: %.*s\n",
      i,
      pattern->is_rooted ? ", rooted" : "",
      (int)(span.end - span.start),
      &source[span.start]
    );
    for (uint32_t j = pattern->entry_offset; j < query_program_entry_end(self, i); j++) {
      const QueryEntry *entry = array_get(&self->entries, j);
      printf("  entry ");
      if (entry->symbol) {
        query_program_dump_symbol(language, entry->symbol);
      } else {
        printf("_");
      }
      printf(" at %u%s\n", entry->pc, entry->is_rooted ? "" : ", not rooted");
    }
#ifdef DEBUG_QUERY_BLOCKS
    uint32_t label = 0;
    while (label < self->labels.size && array_get(&self->labels, label)->pc < pattern->instruction_offset) label++;
#endif
    for (uint32_t pc = pattern->instruction_offset; pc < query_program_instruction_end(self, i); pc++) {
#ifdef DEBUG_QUERY_BLOCKS
      // The blocks that start here, each one falling through to the next.
      bool has_label = false;
      while (label < self->labels.size && array_get(&self->labels, label)->pc == pc) {
        const QueryProgramLabel *block = array_get(&self->labels, label);
        printf(
          "%s %s #%u %.*s",
          has_label ? "," : "        ;",
          block->kind,
          block->node,
          (int)(block->span.end - block->span.start),
          &source[block->span.start]
        );
        if (block->variant) printf(" %s", block->variant);
        has_label = true;
        label++;
      }
      if (has_label) printf("\n");
#endif
      const QueryInstruction instruction = *array_get(&self->instructions, pc);
      const QueryOpcode opcode = query_instruction_opcode(instruction);
      const uint32_t operand = query_instruction_operand(instruction);
      printf("  %4u: %-14s", pc, query_opcode_name(opcode));
      switch (opcode) {
        case QueryOpcodeWait:
        case QueryOpcodeWaitAnchored:
        case QueryOpcodeWaitImmediate: {
          printf("%u", query_instruction_depth(instruction));
          const TSFieldId field = query_instruction_wait_field(instruction);
          if (field) printf(" in %s", ts_language_field_name_for_id(language, field));
          if (!query_instruction_may_capture(instruction)) printf(", no more captures");
          break;
        }
        case QueryOpcodeLast:
          printf("%u", query_instruction_depth(instruction));
          break;
        case QueryOpcodeSplit:
        case QueryOpcodeJump:
          printf("%u", operand);
          break;
        case QueryOpcodeSymbol:
        case QueryOpcodeSupertype:
          query_program_dump_symbol(language, (TSSymbol)operand);
          break;
        case QueryOpcodeField:
        case QueryOpcodeNoField:
          printf("%s", ts_language_field_name_for_id(language, (TSFieldId)operand));
          break;
        case QueryOpcodeCapture: {
          uint32_t length;
          const char *name = query_string_table_get(&tables->captures, operand, &length);
          printf("@%.*s", (int)length, name);
          break;
        }
        case QueryOpcodeNamed:
        case QueryOpcodeNotError:
        case QueryOpcodeMissing:
        case QueryOpcodeAccept:
          break;
      }
      printf("\n");
    }
  }
}
#endif
