#include "./query_lowerer.h"
#include "./language.h"
#include "query_parser.h"
#include "query_program.h"

#include <string.h>

#define QUERY_LOWERER_AST_NONE ((QueryAstIndex) {.index = UINT32_MAX})

// A `QueryEdge` describes the boundary between one matched pattern and the next
// pattern that a thread may match.
//
// A gap is the interval of siblings between those two matched nodes. Named
// _and_ anonymous sibling nodes are significant.
//
// Without an anchor between two patterns, any number of named or anonymous
// siblings may occur between their matched nodes. An anchor between two
// patterns requires the matched nodes to be consecutive named siblings, while still 
// allowing anonymous siblings between them:
//
//   (parent (child) @a (child) @b)
//   (parent (child) @a . (child) @b)
//
// The first pattern may skip named siblings before matching `@b`. The second
// may skip only anonymous siblings. An anchor next to `_` requires the next
// literal sibling, because `_` can match anonymous nodes:
//
//   (parent (_) @a . (child) @b)
//
// A pattern may match no nodes, so an edge can pass through it without matching
// a node. An anchor before such a pattern is vacuous when the pattern is empty
// unless another anchor follows it:
//
//   (parent (child) . (comment)* (child))
//
// If no comments match, the final `child` is not constrained by the preceding
// anchor. More generally, an edge carries the current sibling constraint to
// the next pattern, including across patterns that match no nodes. An
// unanchored boundary weakens `Anchored` and `Immediate` to `Any`.
// `FirstAnchored` preserves a leading first-child restriction, and `Repeated`
// preserves adjacency between iterations. A trailing `.` is enforced by a
// `Last` instruction after the final child rather than by an edge.
typedef enum {
  // No restriction at this boundary. Any named or anonymous sibling may be
  // skipped while searching for the next node in the pattern.
  QueryEdgeAny,
  // Only anonymous siblings may be skipped. The matched nodes must be
  // consecutive named siblings:
  //
  //   (parent (child) @a . (child) @b)
  QueryEdgeAnchored,
  // No sibling may be skipped, including anonymous siblings. This is used when
  // an anchored `_` must be followed by the next literal sibiling.
  QueryEdgeImmediate,
  // The first named node matched by the list must be the parent's first named
  // child. Patterns that match no nodes do not consume this restriction. In
  // this example, the comment run must begin at the first named child, and
  // `child` inherits the restriction when the run is empty:
  //
  //   (parent . (comment)* (child) @id)
  QueryEdgeFirstAnchored,
  // The next iteration of a repeated pattern must match the literal next
  // sibling. This keeps the repetitions contiguous, without allowing
  // unrelated siblings between them:
  //
  //   (parent (comment)+ @comment)
  QueryEdgeRepeated,
} QueryEdge;

// The number of kinds of edge.
#define QUERY_EDGE_COUNT 5

// Whether a pattern or atom can match without matching any nodes, and which
// sibling gaps an empty match may pass. An empty match is `Anchored` only when
// every gap it passes has an anchor. Passing any unanchored gap makes it
// `Unanchored`. If there is more than one empty-match path, the least
// restrictive result wins.
typedef enum {
  // It always matches at least one node.
  QueryEmptyMatchNever,
  // It can match no nodes, passing only gaps with anchors.
  QueryEmptyMatchAnchored,
  // It can match no nodes, passing a gap without an anchor.
  QueryEmptyMatchUnanchored,
} QueryEmptyMatch;

// The lowerer represents each pattern as a graph of blocks. A block is a
// control-flow location identified by its kind, AST node, and variant. The
// variant kinds distinguish matching phases whose control flow and emitted
// instructions differ, so each phase can be memoized and emitted separately.
// The lowerer may emit instructions and fall through to another block:
//
//   Tests(parent) -> EnterList(child) -> Wait(child) -> Tests(child)
//
// A block can also branch to another block. The current thread follows the
// fall-through path, while a copy follows the branch:
//
//   EnterList(optional)
//     ├── Split -> EnterList(next)
//     └── fall through -> Wait(optional)
//
// Each block is emitted at most once. When different paths refer to the same
// block, they converge on one program location:
//
//   skip optional ──────────────┐
//                               ├─> EnterList(next)
//   match optional -> Exit ─────┘
//
// A block that emits no instructions shares the program location of the block
// it falls through to. The final program is a linear instruction stream. The
// block graph supplies the control-flow structure used to construct it.
typedef enum {
  // The tests, fields, and captures for a selected node, followed by the
  // entry to its child list or by the exit of its pattern.
  QueryBlockTests,
  // The continuation after a named node's child list. It emits `Last` when
  // the list ends with an anchor, then enters the node pattern's `Exit` block.
  QueryBlockAfterChildren,
  // Waits for a candidate node at the atom's depth and field, using the wait
  // opcode in `QueryBlock.variant`, then enters its `Tests` block. An atom has
  // one block for each wait opcode it can use.
  QueryBlockWait,
  // Enters a pattern in a list, carrying the edge constraint to the pattern's
  // first matching node, if it has one. It delegates to the atom's entry block
  // and may split to the next pattern when this pattern can match empty.
  QueryBlockEnterList,
  // Enters an alternation along an edge. Simple node branches share one wait
  // and split to their tests. More complex branches split to their own entry
  // blocks.
  QueryBlockEnterAlternation,
  // Continues after a pattern has matched its last node, carrying the edge
  // constraint in `QueryBlock.variant`. It branches to another repetition, the
  // next pattern in its list, or the continuation of its enclosing pattern, which
  // may be `Accept`.
  QueryBlockExit,
  // Completes a pattern match by emitting `Accept`. This block is terminal and
  // has no fall-through continuation.
  QueryBlockAccept,
} QueryBlockKind;

// A symbolic location in the lowerer's control-flow graph. `kind`, `node`, and
// `variant` identify the block, and the lowerer records its first instruction
// address separately as it emits the graph. A block may emit no instructions
// and share its address with the block it falls through to.
typedef struct {
  // The global AST node associated with the block.
  // It is an atom for `Tests` and `Wait`, a named node for `AfterChildren`, a
  // pattern for `EnterList` and `Exit`, an alternation for
  // `EnterAlternation`, and the pattern's root for `Accept`.
  QueryAstIndex node;
  // A QueryBlockKind, checked by `query_lowerer_block_kind`.
  uint8_t kind;
  // A QueryOpcode for `Wait`, a QueryEdge for an `Enter` or `Exit`, or zero
  // when the block kind has no variant. Its interpretation is checked by the
  // typed accessors below.
  uint8_t variant;
} QueryBlock;

_Static_assert(QueryBlockAccept <= UINT8_MAX, "QueryBlockKind must fit in one byte");
_Static_assert(QueryOpcodeAccept <= UINT8_MAX, "QueryOpcode must fit in one byte");
_Static_assert(QueryEdgeRepeated <= UINT8_MAX, "QueryEdge must fit in one byte");
_Static_assert(sizeof(QueryBlock) == 8, "QueryBlock must fit in 8 bytes");

// An instruction whose operand is the first instruction of a block, which is
// filled in once every block of the pattern has been emitted.
typedef struct {
  QueryProgramCounter pc;
  QueryBlock block;
} QueryLowererFixup;

#define QUERY_LOWERER_EXIT_BLOCK_SLOTS 3
#define QUERY_LOWERER_MAX_EXIT_OPTIONS 3
#define QUERY_LOWERER_TESTS_PC_SLOT 0
#define QUERY_LOWERER_AFTER_CHILDREN_PC_SLOT 1
#define QUERY_LOWERER_WAIT_PC_SLOT 2
#define QUERY_LOWERER_EXIT_PC_SLOT QUERY_EDGE_COUNT
#define QUERY_LOWERER_BLOCK_SLOTS \
  (QUERY_EDGE_COUNT + QUERY_LOWERER_EXIT_BLOCK_SLOTS)

// What the lowerer records about one node of the pattern that it is lowering.
typedef struct {
  // The node whose child this one is, or NONE for the pattern's root.
  QueryAstIndex parent;
  // For a named node, a group, or an alternation, its first child pattern,
  // which the parser requires a group or alternation to have.
  QueryAstIndex first_child;
  // For a pattern in a list, the next pattern in the list.
  QueryAstIndex next;
  // For a pattern, the nearest pattern around it, through groups and
  // alternations, with a field or a capture, which its nodes also get.
  QueryAstIndex outer_decorated;
  // The first instruction of each of the node's blocks. This is a shared slot
  // table:
  //
  //   slot  EnterList/EnterAlternation  other block
  //   [0]   Any                         Tests
  //   [1]   Anchored                    AfterChildren
  //   [2]   Immediate                   Wait
  //   [3]   FirstAnchored               WaitAnchored
  //   [4]   Repeated                    WaitImmediate
  //   [5]   Exit Any
  //   [6]   Exit Anchored
  //   [7]   Exit Immediate
  //
  // Enter blocks use the edge value directly. These overlaps are safe because
  // the AST node kind determines which block kinds can belong to it.
  QueryProgramCounter pcs[QUERY_LOWERER_BLOCK_SLOTS];
  // For a pattern or an atom, the depth of the nodes that it matches, counted
  // from the depth of the pattern's root. This is limited by the depth field
  // in a wait instruction.
  uint8_t depth;
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
} QueryLowererNodeInfo;

_Static_assert(QUERY_DEPTH_MAX <= UINT8_MAX, "QueryLowererNode.depth must fit in one byte");
_Static_assert(sizeof(QueryLowererNodeInfo) == 52, "QueryLowererNode must fit in 52 bytes");

// The state of one call to `ts_query_lower`.
typedef struct {
  const TSLanguage *language;
  const QueryTokenList *tokens;
  const QueryAst *ast;
  const QueryAstIds *ids;
  QueryProgram *program;
  // Receives a description of the error that stops lowering.
  QueryDiagnostic *error;
  // The index of the current pattern in `program->patterns`.
  uint32_t pattern;
  // The AST node index of the current top-level pattern's root.
  QueryAstIndex start_ast;
  // Lowering data for every AST node in the current pattern's subtree,
  // `[start_ast, query_ast_end(ast, start_ast))`. Callers pass a global AST
  // AST node index to `query_lowerer_node`, which maps it to this array by
  // subtracting `start_ast`.
  Array(QueryLowererNodeInfo) nodes;

  // The first instruction of the pattern's `Accept` block.
  QueryProgramCounter accept_pc;
  // Blocks referred to by emitted branches, in first-reference order.
  Array(QueryBlock) pending;
  uint32_t pending_index;

  // Blocks that fall through from one another while emitting a chain.
  Array(QueryBlock) chain;

  // Branch instructions whose target block addresses are not known yet.
  Array(QueryLowererFixup) fixups;

  // The atoms whose nodes can be the first node of a match of the pattern.
  Array(QueryAstIndex) entry_atoms;

  // For each instruction of the pattern, whether a thread there can still
  // capture a node.
  Array(bool) may_capture;
#ifdef DEBUG_QUERY_BLOCKS
  // Blocks in placement order.
  Array(QueryBlock) placed;
#endif
} QueryLowerer;

static QueryLowererNodeInfo *query_lowerer_node_info(QueryLowerer *self, QueryAstIndex node) {
  return array_get(&self->nodes, node.index - self->start_ast.index);
}

static QueryAstIndex query_lowerer_ast_end(const QueryLowerer *self) {
  return query_ast_end(self->ast, self->start_ast);
}

static QueryAstKind query_lowerer_kind(const QueryLowerer *self, QueryAstIndex node) {
  return query_ast_kind(self->ast, node);
}

// Returns the atom of the pattern `pattern`, which follows its field, if it has
// one.
static QueryAstIndex query_lowerer_atom(const QueryLowerer *self, QueryAstIndex pattern) {
  // The parser lays out a pattern as `[Field] Atom [Quantifier/Capture...]`.
  ts_assert(query_lowerer_kind(self, pattern) == QueryAstKindPattern);
  QueryAstIndex atom = query_ast_first_child(pattern);
  if (query_lowerer_kind(self, atom) == QueryAstKindField) atom.index++;
  return atom;
}

// Returns whether `node` is an atom that matches one node: a named node, an
// anonymous node, a wildcard, or a MISSING node.
static bool query_lowerer_is_node(const QueryLowerer *self, QueryAstIndex node) {
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

static QueryBlockKind query_lowerer_block_kind(QueryBlock block) {
  ts_assert(block.kind <= QueryBlockAccept);
  return (QueryBlockKind)block.kind;
}

static QueryOpcode query_lowerer_block_wait_opcode(QueryBlock block) {
  ts_assert(query_lowerer_block_kind(block) == QueryBlockWait);
  ts_assert(block.variant >= QueryOpcodeWait && block.variant <= QueryOpcodeWaitImmediate);
  return (QueryOpcode)block.variant;
}

static QueryEdge query_lowerer_block_edge(QueryBlock block) {
  const QueryBlockKind kind = query_lowerer_block_kind(block);
  switch (kind) {
    case QueryBlockEnterList:
    case QueryBlockEnterAlternation:
      ts_assert(block.variant < QUERY_EDGE_COUNT);
      break;
    case QueryBlockExit:
      ts_assert(
        block.variant == QueryEdgeAny ||
        block.variant == QueryEdgeAnchored ||
        block.variant == QueryEdgeImmediate
      );
      break;
    default:
      ts_unreachable();
  }
  return (QueryEdge)block.variant;
}

// Validates that `block` refers to a valid AST node in the current pattern,
// with a node kind and variant appropriate for its block kind. Returns the
// block's global AST node index.
static QueryAstIndex query_lowerer_block_node(const QueryLowerer *self, QueryBlock block) {
  const QueryBlockKind kind = query_lowerer_block_kind(block);
  ts_assert(block.node.index >= self->start_ast.index && block.node.index < query_lowerer_ast_end(self).index);
  const QueryAstKind ast_kind = query_lowerer_kind(self, block.node);
  switch (kind) {
    case QueryBlockTests:
      ts_assert(query_lowerer_is_node(self, block.node));
      ts_assert(block.variant == 0);
      break;
    case QueryBlockWait:
      ts_assert(query_lowerer_is_node(self, block.node));
      query_lowerer_block_wait_opcode(block);
      break;
    case QueryBlockAfterChildren:
      ts_assert(ast_kind == QueryAstKindNamedNode);
      ts_assert(block.variant == 0);
      break;
    case QueryBlockEnterList:
      query_lowerer_block_edge(block);
      ts_assert(ast_kind == QueryAstKindPattern);
      break;
    case QueryBlockExit:
      query_lowerer_block_edge(block);
      ts_assert(ast_kind == QueryAstKindPattern);
      break;
    case QueryBlockAccept:
      ts_assert(ast_kind == QueryAstKindPattern);
      ts_assert(block.variant == 0);
      break;
    case QueryBlockEnterAlternation:
      query_lowerer_block_edge(block);
      ts_assert(ast_kind == QueryAstKindAlternation);
      break;
  }
  return block.node;
}

// Returns the pattern that contains `pattern` in a group or an alternation, or
// NONE if `pattern` is in the list of a named node, or is the root.
static QueryAstIndex query_lowerer_outer_pattern(
  QueryLowerer *self,
  QueryAstIndex pattern
) {
  const QueryAstIndex container = query_lowerer_node_info(self, pattern)->parent;
  if (query_ast_index_is_none(container)) return QUERY_LOWERER_AST_NONE;
  switch (query_lowerer_kind(self, container)) {
    case QueryAstKindGroup:
    case QueryAstKindAlternation:
      return query_lowerer_node_info(self, container)->parent;
    default:
      return QUERY_LOWERER_AST_NONE;
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

// Combines the empty-match summaries of two adjacent parts of a list. If
// either part cannot match empty, the combination cannot. Otherwise, the
// combination is anchored only when both parts and the gap between them are
// anchored.
static QueryEmptyMatch query_empty_match_then(QueryEmptyMatch first, bool is_anchored, QueryEmptyMatch rest) {
  if (first == QueryEmptyMatchNever || rest == QueryEmptyMatchNever) return QueryEmptyMatchNever;
  if (first == QueryEmptyMatchAnchored && is_anchored && rest == QueryEmptyMatchAnchored) {
    return QueryEmptyMatchAnchored;
  }
  return QueryEmptyMatchUnanchored;
}

// Records each node's parent and depth, and links the child patterns of each
// named node and group. Each node is visited before its children.
static bool query_lowerer_link(QueryLowerer *self) {
  const QueryAstIndex end_ast = query_lowerer_ast_end(self);
  for (
    QueryAstIndex node = self->start_ast;
    node.index < end_ast.index;
    node.index++
  ) {
    QueryLowererNodeInfo *info = query_lowerer_node_info(self, node);
    const QueryAstKind kind = query_lowerer_kind(self, node);
    if (kind == QueryAstKindPattern) {
      // A pattern's field comes before its atom, and its captures after.
      const QueryAstIndex atom = query_lowerer_atom(self, node);

      // Determine if this node is decorated: whether it contains a fields, or has captures on itself
      info->is_decorated = atom.index != query_ast_first_child(node).index;
      const QueryAstIndex pattern_end = query_ast_end(self->ast, node);
      for (
        QueryAstIndex suffix = query_ast_end(self->ast, atom);
        suffix.index < pattern_end.index && !info->is_decorated;
        suffix.index++
      ) {
        if (query_lowerer_kind(self, suffix) == QueryAstKindCapture) info->is_decorated = true;
      }

      // For a pattern inside a group or alternation, link to the nearest
      // enclosing pattern with a field or capture.
      const QueryAstIndex outer = query_lowerer_outer_pattern(self, node);
      if (!query_ast_index_is_none(outer)) {
        const QueryLowererNodeInfo *outer_info = query_lowerer_node_info(self, outer);
        info->outer_decorated = outer_info->is_decorated ? outer : outer_info->outer_decorated;
      }
    }

    // The patterns in a named node match its children, one level deeper.
    const uint32_t child_depth = kind == QueryAstKindNamedNode ? info->depth + 1 : info->depth;
    if (child_depth > QUERY_DEPTH_MAX) return false;
    // Predicates and negated fields are not siblings, so the patterns on
    // either side of them are adjacent in the list.
    QueryAstIndex previous = QUERY_LOWERER_AST_NONE;
    bool is_anchored = false;
    const QueryAstIndex node_end = query_ast_end(self->ast, node);
    for (
      QueryAstIndex child = query_ast_first_child(node);
      child.index < node_end.index;
      child = query_ast_end(self->ast, child)
    ) {
      QueryLowererNodeInfo *child_info = query_lowerer_node_info(self, child);
      child_info->parent = node;
      child_info->depth = (uint8_t)child_depth;
      switch (query_lowerer_kind(self, child)) {
        case QueryAstKindAnchor:
          is_anchored = true;
          break;
        case QueryAstKindPattern:
          if (query_ast_index_is_none(previous)) {
            info->first_child = child;
            info->has_leading_anchor = is_anchored;
          } else if (kind != QueryAstKindAlternation) {
            QueryLowererNodeInfo *previous_info = query_lowerer_node_info(self, previous);
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
    info->has_trailing_anchor =
      !query_ast_index_is_none(previous) && is_anchored;
  }
  return true;
}

// Records which patterns and atoms can match no nodes, and which patterns
// repeat. Each node is visited after its descendants and the patterns after
// it in its list.
static void query_lowerer_measure(QueryLowerer *self) {
  for (
    QueryAstIndex node = query_lowerer_ast_end(self);
    node.index > self->start_ast.index;
  ) {
    node.index--;
    QueryLowererNodeInfo *info = query_lowerer_node_info(self, node);
    switch (query_lowerer_kind(self, node)) {
      case QueryAstKindPattern: {
        const QueryAstIndex atom = query_lowerer_atom(self, node);
        const QueryLowererNodeInfo *atom_info = query_lowerer_node_info(self, atom);
        // A pattern's quantifiers and captures are leaves after its atom.
        bool can_skip = false;
        const QueryAstIndex pattern_end = query_ast_end(self->ast, node);
        for (
          QueryAstIndex suffix = query_ast_end(self->ast, atom);
          suffix.index < pattern_end.index;
          suffix.index++
        ) {
          if (query_lowerer_kind(self, suffix) != QueryAstKindQuantifier) continue;
          switch (query_token_list_kind(self->tokens, query_ast_token(self->ast, suffix))) {
            case QueryTokenKindQuestion: can_skip = true; break;
            case QueryTokenKindStar: can_skip = true; info->is_repeated = true; break;
            case QueryTokenKindPlus: info->is_repeated = true; break;
            // INVARIANT: The parser only makes quantifiers of these three tokens.
            default: ts_unreachable();
          }
        }
        // Skipping a pattern whose atom must match a node passes no gaps. A
        // pattern whose atom can match no nodes passes the gaps inside it.
        if (atom_info->empty_match != QueryEmptyMatchNever) {
          info->empty_match = atom_info->empty_match;
        } else {
          info->empty_match = can_skip ? QueryEmptyMatchAnchored : QueryEmptyMatchNever;
        }
        info->rest_empty_match = query_ast_index_is_none(info->next)
          ? info->empty_match
          : query_empty_match_then(
              info->empty_match,
              info->is_anchored_after,
              query_lowerer_node_info(self, info->next)->rest_empty_match
            );
        break;
      }
      case QueryAstKindGroup:
        info->empty_match = query_lowerer_node_info(self, info->first_child)->rest_empty_match;
        break;
      case QueryAstKindAlternation:
        info->has_only_nodes = true;
        const QueryAstIndex end = query_ast_end(self->ast, node);
        for (
          QueryAstIndex branch = query_ast_first_child(node);
          branch.index < end.index;
          branch = query_ast_next_sibling(self->ast, branch)
        ) {
          const QueryLowererNodeInfo *branch_info = query_lowerer_node_info(self, branch);
          // An alternation can match empty whenever _any_ branch can, so retain
          // the least restrictive empty-match result.
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
  query_lowerer_node_info(self, self->start_ast)->can_start = true;
  const QueryAstIndex end_ast = query_lowerer_ast_end(self);
  for (
    QueryAstIndex node = self->start_ast;
    node.index < end_ast.index;
    node.index++
  ) {
    if (query_lowerer_kind(self, node) != QueryAstKindPattern) continue;
    QueryLowererNodeInfo *info = query_lowerer_node_info(self, node);
    if (info->depth == 0) {
      const QueryAstIndex outer = query_lowerer_outer_pattern(self, node);
      info->can_continue =
        info->is_repeated ||
        !query_ast_index_is_none(info->next) ||
        (!query_ast_index_is_none(outer) &&
          query_lowerer_node_info(self, outer)->can_continue);

    }
    if (!info->can_start) continue;
    if (
      info->empty_match != QueryEmptyMatchNever &&
      !query_ast_index_is_none(info->next)
    ) {
      query_lowerer_node_info(self, info->next)->can_start = true;
    }
    const QueryAstIndex atom = query_lowerer_atom(self, node);
    switch (query_lowerer_kind(self, atom)) {
      case QueryAstKindGroup:
        query_lowerer_node_info(self, query_lowerer_node_info(self, atom)->first_child)->can_start = true;
        break;
      case QueryAstKindAlternation: {
        const QueryAstIndex end = query_ast_end(self->ast, atom);
        for (
          QueryAstIndex branch = query_ast_first_child(atom);
          branch.index < end.index;
          branch = query_ast_next_sibling(self->ast, branch)
        ) {
          query_lowerer_node_info(self, branch)->can_start = true;
        }
        break;
      }
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
static QueryProgramCounter *query_lowerer_block_pc(
  QueryLowerer *self,
  QueryBlock block
) {
  const QueryBlockKind kind = query_lowerer_block_kind(block);
  const QueryAstIndex node = query_lowerer_block_node(self, block);
  switch (kind) {
    case QueryBlockAccept:
      return &self->accept_pc;
    case QueryBlockTests:
      return &query_lowerer_node_info(self, node)->pcs[QUERY_LOWERER_TESTS_PC_SLOT];
    case QueryBlockAfterChildren:
      return &query_lowerer_node_info(self, node)->pcs[QUERY_LOWERER_AFTER_CHILDREN_PC_SLOT];
    case QueryBlockWait:
      // The wait opcodes are consecutive.
      return &query_lowerer_node_info(self, node)->pcs[
        QUERY_LOWERER_WAIT_PC_SLOT +
        query_lowerer_block_wait_opcode(block) - QueryOpcodeWait
      ];
    case QueryBlockEnterList:
    case QueryBlockEnterAlternation:
      return &query_lowerer_node_info(self, node)->pcs[query_lowerer_block_edge(block)];
    case QueryBlockExit:
      // INVARIANT: A thread only exits a pattern after matching a node, so
      // only these three edges reach an exit.
      return &query_lowerer_node_info(self, node)->pcs[
        QUERY_LOWERER_EXIT_PC_SLOT + query_lowerer_block_edge(block)
      ];
  }
  ts_unreachable();
}

// Emits an instruction. An operand that does not fit in an instruction, or an
// instruction at the reserved unresolved-target address, makes the pattern
// too large, and the pattern fails.
static bool query_lowerer_emit(QueryLowerer *self, QueryOpcode opcode, uint32_t operand) {
  if (
    operand > QUERY_OPERAND_MAX ||
    self->program->instructions.size >= QUERY_OPERAND_MAX
  ) {
    return false;
  }
  array_push(&self->program->instructions, query_instruction_new(opcode, operand));
  return true;
}

// Emits an instruction whose operand is the first instruction of `block`. The
// operand is initially `QUERY_OPERAND_UNRESOLVED`, and replaced with the correct
// address by the fixup pass.
static bool query_lowerer_emit_reference(QueryLowerer *self, QueryOpcode opcode, QueryBlock block) {
  const QueryProgramCounter pc = {
    .index = self->program->instructions.size,
  };
  if (!query_lowerer_emit(self, opcode, QUERY_OPERAND_UNRESOLVED)) return false;
  array_push(&self->fixups, ((QueryLowererFixup) {
    .pc = pc,
    .block = block,
  }));
  if (query_program_counter_is_none(*query_lowerer_block_pc(self, block))) {
    array_push(&self->pending, block);
  }
  return true;
}

// Returns the block that waits along `edge` for the first node of `atom`.
static QueryBlock query_lowerer_enter_atom(QueryLowerer *self, QueryAstIndex atom, QueryEdge edge) {
  switch (query_lowerer_kind(self, atom)) {
    case QueryAstKindGroup:
      return (QueryBlock) {
        .kind = QueryBlockEnterList,
        .node = query_lowerer_node_info(self, atom)->first_child,
        .variant = edge,
      };
    case QueryAstKindAlternation:
      return (QueryBlock) {.kind = QueryBlockEnterAlternation, .node = atom, .variant = edge};
    case QueryAstKindNamedNode:
    case QueryAstKindAnonymousNode:
    case QueryAstKindWildcard:
    case QueryAstKindMissingNode:
      return (QueryBlock) {.kind = QueryBlockWait, .node = atom, .variant = query_edge_wait_opcode(edge)};
    default:
      ts_unreachable();
  }
}

// Returns the exit of the pattern of `atom`, a node, after it matches.
static QueryBlock query_lowerer_exit_node(QueryLowerer *self, QueryAstIndex atom) {
  // An anchor after a node lets only anonymous nodes be skipped, and one after
  // a `_` lets none be, so that in `_ . (x)` the `_` is the node just before
  // the `(x)`, whichever kind it is.
  const QueryEdge edge = query_lowerer_kind(self, atom) == QueryAstKindWildcard
    ? QueryEdgeImmediate
    : QueryEdgeAnchored;
  return (QueryBlock) {
    .kind = QueryBlockExit,
    .node = query_lowerer_node_info(self, atom)->parent,
    .variant = edge,
  };
}

// Emits splits to every block in `blocks` but the first, and writes the first,
// which the current thread continues with, to `next`.
static bool query_lowerer_emit_branches(
  QueryLowerer *self,
  const QueryBlock *blocks,
  uint32_t count,
  QueryBlock *next
) {
  ts_assert(count > 0);
  ts_assert(count <= QUERY_LOWERER_MAX_EXIT_OPTIONS);
  for (uint32_t i = 1; i < count; i++) {
    if (!query_lowerer_emit_reference(self, QueryOpcodeSplit, blocks[i])) return false;
  }
  *next = blocks[0];
  return true;
}

// Returns the field that the node of `atom` must be in: the field of its
// pattern, or of the nearest pattern around it, through groups and
// alternations, that has one. Returns zero if there is none.
static TSFieldId query_lowerer_field(QueryLowerer *self, QueryAstIndex atom) {
  for (
    QueryAstIndex pattern = query_lowerer_node_info(self, atom)->parent;
    !query_ast_index_is_none(pattern);
    pattern = query_lowerer_node_info(self, pattern)->outer_decorated
  ) {
    const QueryAstIndex field = query_ast_first_child(pattern);
    if (query_lowerer_kind(self, field) == QueryAstKindField) {
      return (TSFieldId)query_ast_ids_get(self->ids, field);
    }
  }
  return 0;
}

// Emits the tests and captures of `atom`, a node: its own, and those of the
// groups and alternations around it, up to the nearest named node. Writes the
// block that follows, which is its children, if it has any, or the exit of its
// pattern, to `next`.
static bool query_lowerer_emit_tests(
  QueryLowerer *self,
  QueryAstIndex atom,
  QueryBlock *next
) {
  const uint32_t id = query_ast_ids_get(self->ids, atom);
  const QueryAstIndex end = query_ast_end(self->ast, atom);
  // A test of the node's symbol, if it has one, comes first, because an entry
  // is found by the symbol of its first test.
  switch (query_lowerer_kind(self, atom)) {
    case QueryAstKindNamedNode:
      if (id == 0) {
        // `(_)` is any named node but an ERROR.
        if (!query_lowerer_emit(self, QueryOpcodeNamed, QUERY_OPERAND_NONE)) return false;
        if (!query_lowerer_emit(self, QueryOpcodeNotError, QUERY_OPERAND_NONE)) return false;
      } else if (ts_language_symbol_metadata(self->language, (TSSymbol)id).supertype) {
        // A subtype is the first child of its named node. Without one, a
        // supertype is any node that it wraps but an ERROR.
        const QueryAstIndex maybe_subtype = query_ast_first_child(atom);
        if (
          maybe_subtype.index < end.index &&
          query_lowerer_kind(self, maybe_subtype) == QueryAstKindSubtype
        ) {
          if (!query_lowerer_emit(self, QueryOpcodeSymbol, query_ast_ids_get(self->ids, maybe_subtype))) return false;
          if (!query_lowerer_emit(self, QueryOpcodeSupertype, id)) return false;
        } else {
          if (!query_lowerer_emit(self, QueryOpcodeSupertype, id)) return false;
          if (!query_lowerer_emit(self, QueryOpcodeNotError, QUERY_OPERAND_NONE)) return false;
        }
      } else {
        if (!query_lowerer_emit(self, QueryOpcodeSymbol, id)) return false;
      }
      break;
    case QueryAstKindAnonymousNode:
      if (!query_lowerer_emit(self, QueryOpcodeSymbol, id)) return false;
      break;
    case QueryAstKindWildcard:
      // `_` is any node but an ERROR, named or not.
      if (!query_lowerer_emit(self, QueryOpcodeNotError, QUERY_OPERAND_NONE)) return false;
      break;
    case QueryAstKindMissingNode: {
      // The type after `MISSING`, if there is one, is its only child.
      const QueryAstIndex maybe_missing_type = query_ast_first_child(atom);
      if (maybe_missing_type.index < end.index) {
        ts_assert(query_lowerer_kind(self, maybe_missing_type) == QueryAstKindMissingType);
        if (!query_lowerer_emit(self, QueryOpcodeSymbol, query_ast_ids_get(self->ids, maybe_missing_type))) return false;
      }
      if (!query_lowerer_emit(self, QueryOpcodeMissing, QUERY_OPERAND_NONE)) return false;
      break;
    }
    default:
      ts_unreachable();
  }

  // Apply fields from the atom's pattern and its decorated enclosing patterns.
  for (
    QueryAstIndex pattern = query_lowerer_node_info(self, atom)->parent;
    !query_ast_index_is_none(pattern);
    pattern = query_lowerer_node_info(self, pattern)->outer_decorated
  ) {
    const QueryAstIndex maybe_field = query_ast_first_child(pattern);
    if (
      query_lowerer_kind(self, maybe_field) == QueryAstKindField &&
      !query_lowerer_emit(self, QueryOpcodeField, query_ast_ids_get(self->ids, maybe_field))
    ) {
      return false;
    }
  }
  // A named node's negated fields are among its children.
  for (
    QueryAstIndex child = query_ast_first_child(atom);
    child.index < end.index;
    child = query_ast_next_sibling(self->ast, child)
  ) {
    if (
      query_lowerer_kind(self, child) == QueryAstKindNegatedField &&
      !query_lowerer_emit(self, QueryOpcodeNoField, query_ast_ids_get(self->ids, child))
    ) {
      return false;
    }
  }
  // Apply captures after all tests, so a thread only records a node once it has
  // passed them.
  for (
    QueryAstIndex pattern = query_lowerer_node_info(self, atom)->parent;
    !query_ast_index_is_none(pattern);
    pattern = query_lowerer_node_info(self, pattern)->outer_decorated
  ) {
    const QueryAstIndex pattern_atom = query_lowerer_atom(self, pattern);
    const QueryAstIndex pattern_end = query_ast_end(self->ast, pattern);
    for (
      QueryAstIndex suffix = query_ast_end(self->ast, pattern_atom);
      suffix.index < pattern_end.index;
      suffix.index++
    ) {
      if (
        query_lowerer_kind(self, suffix) == QueryAstKindCapture &&
        !query_lowerer_emit(self, QueryOpcodeCapture, query_ast_ids_get(self->ids, suffix))
      ) {
        return false;
      }
    }
  }

  // Only a named node can have child patterns.
  const QueryLowererNodeInfo *info = query_lowerer_node_info(self, atom);
  if (query_ast_index_is_none(info->first_child)) {
    *next = query_lowerer_exit_node(self, atom);
    return true;
  }
  // If the node's child patterns can all match nothing, a copy of the thread
  // skips them and exits the node's pattern: with no child matched, a trailing
  // anchor has no node to apply to.
  if (query_lowerer_node_info(self, info->first_child)->rest_empty_match != QueryEmptyMatchNever) {
    if (!query_lowerer_emit_reference(self, QueryOpcodeSplit, query_lowerer_exit_node(self, atom))) return false;
  }
  *next = (QueryBlock) {
    .kind = QueryBlockEnterList,
    .node = info->first_child,
    .variant = info->has_leading_anchor ? QueryEdgeFirstAnchored : QueryEdgeAny,
  };
  return true;
}

// Emits the exit of `pattern` along `edge`, which is how the thread may skip
// siblings after the last node that it matched, and writes the block that the
// thread continues with to `next`.
static bool query_lowerer_emit_exit(
  QueryLowerer *self,
  QueryAstIndex pattern,
  QueryEdge edge,
  QueryBlock *next
) {
  const QueryLowererNodeInfo *info = query_lowerer_node_info(self, pattern);
  // Another repetition, the next patterns of the list, and what comes after
  // the list or the pattern around this one: at most three, which run in that
  // order.
  QueryBlock options[QUERY_LOWERER_MAX_EXIT_OPTIONS] = {0};
  uint32_t count = 0;
  if (info->is_repeated) {
    options[count++] = query_lowerer_enter_atom(self, query_lowerer_atom(self, pattern), QueryEdgeRepeated);
  }

  // A match of the root is a match of the whole pattern.
  const QueryAstIndex container = info->parent;
  if (query_ast_index_is_none(container)) {
    options[count++] = (QueryBlock) {.kind = QueryBlockAccept, .node = self->start_ast};
    return query_lowerer_emit_branches(self, options, count, next);
  }
  // A match of one of an alternation's patterns is a match of the
  // alternation's pattern, and ends with the same node.
  if (query_lowerer_kind(self, container) == QueryAstKindAlternation) {
    options[count++] = (QueryBlock) {
      .kind = QueryBlockExit,
      .node = query_lowerer_node_info(self, container)->parent,
      .variant = edge,
    };
    return query_lowerer_emit_branches(self, options, count, next);
  }

  // In a list, the thread goes on to the next pattern through the gap after
  // this one. If the rest of the list can match nothing, it can also go through
  // all of it, to the end of the list.
  if (!query_ast_index_is_none(info->next)) {
    const QueryLowererNodeInfo *next_info = query_lowerer_node_info(self, info->next);
    edge = query_edge_pass(edge, info->is_anchored_after);
    options[count++] = (QueryBlock) {.kind = QueryBlockEnterList, .node = info->next, .variant = edge};
    if (next_info->rest_empty_match == QueryEmptyMatchNever) {
      return query_lowerer_emit_branches(self, options, count, next);
    }
    edge = query_edge_pass(edge, next_info->rest_empty_match == QueryEmptyMatchAnchored);
  }
  // At the end of a group's list, the group's pattern has matched. At the end
  // of a named node's list, all of the node's children have.
  if (query_lowerer_kind(self, container) == QueryAstKindGroup) {
    options[count++] = (QueryBlock) {
      .kind = QueryBlockExit,
      .node = query_lowerer_node_info(self, container)->parent,
      .variant = edge,
    };
  } else {
    options[count++] = (QueryBlock) {.kind = QueryBlockAfterChildren, .node = container};
  }
  return query_lowerer_emit_branches(self, options, count, next);
}

typedef enum {
  QueryLowererBlockFailed,
  QueryLowererBlockFinished,
  QueryLowererBlockContinues,
} QueryLowererBlockResult;

// Emits the instructions of `block`, and writes whether the thread continues
// with another block, and if so, `next`, after them.
static QueryLowererBlockResult query_lowerer_emit_block(
  QueryLowerer *self,
  QueryBlock block,
  QueryBlock *next
) {
  const QueryBlockKind kind = query_lowerer_block_kind(block);
  const QueryAstIndex node = query_lowerer_block_node(self, block);
  const QueryLowererNodeInfo *info = query_lowerer_node_info(self, node);
  switch (kind) {
    case QueryBlockTests:
      if (!query_lowerer_emit_tests(self, node, next)) {
        return QueryLowererBlockFailed;
      }
      return QueryLowererBlockContinues;
    case QueryBlockAfterChildren:
      // An anchor at the end of the list requires the last child that matched
      // to be the node's last named child. Then the node's pattern exits, as it
      // does after a node without children.
      if (
        info->has_trailing_anchor &&
        !query_lowerer_emit(self, QueryOpcodeLast, info->depth + 1)
      ) {
        return QueryLowererBlockFailed;
      }
      *next = query_lowerer_exit_node(self, node);
      return QueryLowererBlockContinues;
    case QueryBlockWait: {
      const TSFieldId field = query_lowerer_field(self, node);
      if (
        !query_lowerer_emit(
          self,
          query_lowerer_block_wait_opcode(block),
          query_wait_operand(info->depth, field)
        )
      ) {
        return QueryLowererBlockFailed;
      }
      *next = (QueryBlock) {.kind = QueryBlockTests, .node = node};
      return QueryLowererBlockContinues;
    }
    case QueryBlockEnterList: {
      const QueryEdge edge_before = query_lowerer_block_edge(block);
      const QueryAstIndex atom = query_lowerer_atom(self, node);
      // If the pattern can match nothing, a copy of the thread skips it, which
      // passes the gaps inside it, if it has any, and the gap after it.
      // Skipping the last pattern of a list is offered earlier, as the end of
      // the list.
      if (
        info->empty_match != QueryEmptyMatchNever &&
        !query_ast_index_is_none(info->next)
      ) {
        QueryEdge edge = query_edge_pass(
          edge_before,
          info->empty_match == QueryEmptyMatchAnchored
        );
        edge = query_edge_pass(edge, info->is_anchored_after);
        const QueryBlock target = {
          .kind = QueryBlockEnterList,
          .node = info->next,
          .variant = edge,
        };
        if (!query_lowerer_emit_reference(self, QueryOpcodeSplit, target)) {
          return QueryLowererBlockFailed;
        }
      }
      *next = query_lowerer_enter_atom(self, atom, edge_before);
      return QueryLowererBlockContinues;
    }
    case QueryBlockEnterAlternation: {
      const QueryEdge edge = query_lowerer_block_edge(block);
      const QueryAstIndex end = query_ast_end(self->ast, node);
      const QueryAstIndex first_branch = query_ast_end(self->ast, info->first_child);
      // Patterns that are all nodes share one wait, and then each tests the
      // node that it gives them, in a thread of its own.
      if (info->has_only_nodes) {
        // The wait has a field if the patterns' nodes must all be in the same
        // one.
        TSFieldId field = query_lowerer_field(
          self,
          query_lowerer_atom(self, info->first_child)
        );
        for (
          QueryAstIndex branch = first_branch;
          branch.index < end.index;
          branch = query_ast_next_sibling(self->ast, branch)
        ) {
          if (query_lowerer_field(self, query_lowerer_atom(self, branch)) != field) field = 0;
        }
        if (!query_lowerer_emit(
          self,
          query_edge_wait_opcode(edge),
          query_wait_operand(info->depth, field)
        )) {
          return QueryLowererBlockFailed;
        }
        for (
          QueryAstIndex branch = first_branch;
          branch.index < end.index;
          branch = query_ast_next_sibling(self->ast, branch)
        ) {
          const QueryBlock target = {
            .kind = QueryBlockTests,
            .node = query_lowerer_atom(self, branch),
          };
          if (!query_lowerer_emit_reference(self, QueryOpcodeSplit, target)) {
            return QueryLowererBlockFailed;
          }
        }
        *next = (QueryBlock) {
          .kind = QueryBlockTests,
          .node = query_lowerer_atom(self, info->first_child),
        };
        return QueryLowererBlockContinues;
      }
      // Otherwise each pattern waits for its own first node.
      for (
        QueryAstIndex branch = first_branch;
        branch.index < end.index;
        branch = query_ast_next_sibling(self->ast, branch)
      ) {
        const QueryBlock target = query_lowerer_enter_atom(
          self,
          query_lowerer_atom(self, branch),
          edge
        );
        if (!query_lowerer_emit_reference(self, QueryOpcodeSplit, target)) {
          return QueryLowererBlockFailed;
        }
      }
      *next = query_lowerer_enter_atom(
        self,
        query_lowerer_atom(self, info->first_child),
        edge
      );
      return QueryLowererBlockContinues;
    }
    case QueryBlockExit:
      if (!query_lowerer_emit_exit(
        self,
        node,
        query_lowerer_block_edge(block),
        next
      )) {
        return QueryLowererBlockFailed;
      }
      return QueryLowererBlockContinues;
    case QueryBlockAccept:
      if (!query_lowerer_emit(self, QueryOpcodeAccept, QUERY_OPERAND_NONE)) return QueryLowererBlockFailed;
      return QueryLowererBlockFinished;
  }
  ts_unreachable();
}

// Emits a chain along the current thread's fall-through path:
//
//   pending block -> block A -> block B -> placed block C
//
// `next` from one block identifies the next block, which the thread reaches
// without a jump. Each block is assigned its PC before emission, so blocks
// that emit no instructions share the current instruction position.
//
// When the chain reaches a placed block, it reconnects to that block by
// aliasing the no-instruction blocks if the chain emitted nothing, or by
// appending `Accept` or a `Jump` if it emitted instructions.
//
// Returns false if emission needs an instruction operand that does not fit.
static bool query_lowerer_emit_chain(QueryLowerer *self, QueryBlock block) {
  // Assign each block its starting PC before emitting it. Blocks that emit no
  // instructions share the PC of the next block that does.
  // The blocks placed at `start` are recorded in case the entire chain emits
  // no instructions.
  const QueryProgramCounter start = {
    .index = self->program->instructions.size,
  };
  array_clear(&self->chain);
  for (;;) {
    QueryProgramCounter *pc = query_lowerer_block_pc(self, block);
    // A non-NONE PC means this block was already placed. The tail below handles
    // the chain's reconvergence with it.
    if (!query_program_counter_is_none(*pc)) break;
    // Reserve this block's starting PC before emitting its instructions.
    pc->index = self->program->instructions.size;
    if (pc->index == start.index) array_push(&self->chain, block);
#ifdef DEBUG_QUERY_BLOCKS
    array_push(&self->placed, block);
#endif
    QueryBlock next = {.node = QUERY_LOWERER_AST_NONE};
    switch (query_lowerer_emit_block(self, block, &next)) {
      case QueryLowererBlockFailed: return false;
      case QueryLowererBlockFinished: return true;
      case QueryLowererBlockContinues:
        ts_assert(!query_ast_index_is_none(next.node));
        // Continue emitting the next block in the chain
        block = next;
        break;
    }
  }

  // The chain has reached a block that was emitted before.
  const QueryProgramCounter pc = *query_lowerer_block_pc(self, block);
  if (self->program->instructions.size == start.index) {
    // Every chain ends with a jump or an accept, so nothing falls through to
    // the start of this one, and its blocks can start where that block does.
    for (uint32_t i = 0; i < self->chain.size; i++) {
      *query_lowerer_block_pc(self, *array_get(&self->chain, i)) = pc;
    }
  } else if (
    query_instruction_opcode(
      *array_get(&self->program->instructions, pc.index)
    ) == QueryOpcodeAccept
  ) {
    // Falling through to a block that only accepts is the same as accepting.
    if (!query_lowerer_emit(self, QueryOpcodeAccept, QUERY_OPERAND_NONE)) return false;
  } else {
    if (!query_lowerer_emit_reference(self, QueryOpcodeJump, block)) return false;
  }
  return true;
}

// Marks each of the pattern's waits where a thread can still capture a node,
// because a `Capture` can be reached from it.
static void query_lowerer_mark_waits(
  QueryLowerer *self,
  QueryProgramCounter pattern_start_pc
) {
  const uint32_t count = self->program->instructions.size - pattern_start_pc.index;
  array_clear(&self->may_capture);
  array_grow_by(&self->may_capture, count);
  for (bool changed = true; changed;) {
    changed = false;
    for (uint32_t i = count; i > 0;) {
      i--;
      const QueryInstruction instruction = *array_get(
        &self->program->instructions,
        pattern_start_pc.index + i
      );
      bool may_capture;
      switch (query_instruction_opcode(instruction)) {
        case QueryOpcodeCapture: may_capture = true; break;
        case QueryOpcodeAccept: may_capture = false; break;
        case QueryOpcodeJump: {
          const QueryProgramCounter target = query_instruction_target(instruction);
          may_capture = *array_get(
            &self->may_capture,
            target.index - pattern_start_pc.index
          );
          break;
        }
        case QueryOpcodeSplit: {
          const QueryProgramCounter target = query_instruction_target(instruction);
          may_capture =
            *array_get(
              &self->may_capture,
              target.index - pattern_start_pc.index
            ) ||
            *array_get(&self->may_capture, i + 1);
          break;
        }
        default: may_capture = *array_get(&self->may_capture, i + 1); break;
      }
      if (may_capture && !*array_get(&self->may_capture, i)) {
        *array_get(&self->may_capture, i) = true;
        changed = true;
      }
    }
  }
  for (uint32_t i = 0; i < count; i++) {
    QueryInstruction *instruction = array_get(
      &self->program->instructions,
      pattern_start_pc.index + i
    );
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
  const QueryAstIndex end_ast = query_lowerer_ast_end(self);
  const QueryTokenIndex next_token = end_ast.index < query_ast_size(self->ast)
    ? query_ast_token(self->ast, end_ast)
    : (QueryTokenIndex) {
      .index = query_token_list_size(self->tokens) - 1,
    };
  *self->error = (QueryDiagnostic) {
    .span = {
      .start = query_token_list_span(self->tokens, query_ast_token(self->ast, self->start_ast)).start,
      .end = query_token_list_span(
        self->tokens,
        query_token_previous(next_token)
      ).end,
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
    const QueryBlockKind kind = query_lowerer_block_kind(block);
    const QueryAstIndex node = query_lowerer_block_node(self, block);
    QueryProgramLabel label = {
      .pc = *query_lowerer_block_pc(self, block),
      .kind = "?",
      .variant = NULL,
      .node = node.index,
      .span = query_token_list_span(self->tokens, query_ast_token(self->ast, node)),
    };
    switch (kind) {
      case QueryBlockTests: label.kind = "Tests"; break;
      case QueryBlockAfterChildren: label.kind = "AfterChildren"; break;
      case QueryBlockWait:
        label.kind = "Wait";
        label.variant = query_lowerer_block_wait_opcode(block) == QueryOpcodeWait ? "Wait" :
          query_lowerer_block_wait_opcode(block) == QueryOpcodeWaitAnchored ? "WaitAnchored" : "WaitImmediate";
        break;
      case QueryBlockEnterList:
        label.kind = "EnterList";
        label.variant = query_edge_name(query_lowerer_block_edge(block));
        break;
      case QueryBlockEnterAlternation:
        label.kind = "EnterAlternation";
        label.variant = query_edge_name(query_lowerer_block_edge(block));
        break;
      case QueryBlockExit:
        label.kind = "Exit";
        label.variant = query_edge_name(query_lowerer_block_edge(block));
        break;
      case QueryBlockAccept: label.kind = "Accept"; break;
    }
    // A pattern's token is its first one, which is often just a `(`, so it is
    // shown by its atom's token instead.
    if (kind == QueryBlockAccept) {
      label.span = (QuerySpan) {.start = 0, .end = 0};
    } else if (query_lowerer_kind(self, node) == QueryAstKindPattern) {
      label.span = query_token_list_span(self->tokens, query_ast_token(self->ast, query_lowerer_atom(self, node)));
    }
    // Insert it after every label that starts at or before the same pc.
    uint32_t index = self->program->labels.size;
    while (
      index > offset &&
      array_get(&self->program->labels, index - 1)->pc.index > label.pc.index
    ) index--;
    array_insert(&self->program->labels, index, label);
  }
  array_clear(&self->placed);
}
#endif

// Lowers the pattern whose root is the node `self.start_ast`.
static bool query_lowerer_lower_pattern(QueryLowerer *self) {
  // Entries and matches store a pattern's index in 16 bits.
  if (self->pattern > UINT16_MAX) return query_lowerer_fail(self, QueryDiagnosticKindTooManyPatterns);
  array_clear(&self->nodes);
  array_grow_by(
    &self->nodes,
    query_lowerer_ast_end(self).index - self->start_ast.index
  );
  for (uint32_t i = 0; i < self->nodes.size; i++) {
    QueryLowererNodeInfo *info = array_get(&self->nodes, i);
    info->parent = QUERY_LOWERER_AST_NONE;
    info->first_child = QUERY_LOWERER_AST_NONE;
    info->next = QUERY_LOWERER_AST_NONE;
    info->outer_decorated = QUERY_LOWERER_AST_NONE;
    memset(
      info->pcs,
      (unsigned char)QUERY_PROGRAM_COUNTER_NONE_INDEX,
      sizeof(info->pcs)
    );
  }
  self->accept_pc = QUERY_PROGRAM_COUNTER_NONE;
  array_clear(&self->pending);
  self->pending_index = 0;
  array_clear(&self->fixups);

  if (!query_lowerer_link(self)) return query_lowerer_fail(self, QueryDiagnosticKindPatternTooLarge);
  query_lowerer_measure(self);
  query_lowerer_find_entries(self);

  const QueryProgramCounter pattern_start_pc = {
    .index = self->program->instructions.size,
  };
  for (uint32_t i = 0; i < self->entry_atoms.size; i++) {
    array_push(
        &self->pending,
        ((QueryBlock) {
         .kind = QueryBlockTests,
         .node = *array_get(&self->entry_atoms, i)
        })
      );
  }
  while (self->pending_index < self->pending.size) {
    const QueryBlock block = *array_get(&self->pending, self->pending_index);
    self->pending_index++;
    if (
      query_program_counter_is_none(*query_lowerer_block_pc(self, block)) &&
      !query_lowerer_emit_chain(self, block)
    ) {
      return query_lowerer_fail(self, QueryDiagnosticKindPatternTooLarge);
    }
  }

  for (uint32_t i = 0; i < self->fixups.size; i++) {
    const QueryLowererFixup *fixup = array_get(&self->fixups, i);
    QueryInstruction *instruction = array_get(
      &self->program->instructions,
      fixup->pc.index
    );
    const QueryProgramCounter pc = *query_lowerer_block_pc(self, fixup->block);
    // INVARIANT: Every block that is referred to is emitted.
    ts_assert(!query_program_counter_is_none(pc));
    ts_assert(query_instruction_operand(*instruction) == QUERY_OPERAND_UNRESOLVED);
    *instruction = query_instruction_new(
      query_instruction_opcode(*instruction),
      pc.index
    );
  }
  query_lowerer_mark_waits(self, pattern_start_pc);

#ifdef DEBUG_QUERY_BLOCKS
  query_lowerer_label_blocks(self);
#endif

  // A match has another node at the top level only if it waits at depth zero,
  // and the program only does that to repeat a pattern at the top level, or to
  // go on to the next pattern of a group at the top level.
  bool is_rooted = true;
  for (uint32_t i = 0; i < self->entry_atoms.size; i++) {
    const QueryLowererNodeInfo *atom = query_lowerer_node_info(self, *array_get(&self->entry_atoms, i));
    const QueryProgramCounter pc = atom->pcs[QUERY_LOWERER_TESTS_PC_SLOT];
    const QueryInstruction first =
      *array_get(&self->program->instructions, pc.index);
    const bool is_entry_rooted = !query_lowerer_node_info(self, atom->parent)->can_continue;
    is_rooted = is_rooted && is_entry_rooted;
    array_push(&self->program->entries, ((QueryEntry) {
      .pc = pc.index,
      .pattern = (uint16_t)self->pattern,
      // The VM groups symbol-first entries by this hint before starting them.
      .symbol = query_instruction_opcode(first) == QueryOpcodeSymbol
        ? query_instruction_symbol(first)
        : 0,
      .is_rooted = is_entry_rooted,
    }));
  }
  array_push(&self->program->patterns, ((QueryProgramPattern) {
    .instruction_offset = pattern_start_pc.index,
    .is_rooted = is_rooted,
  }));
  return true;
}

// Groups the program's entries by the symbol that their first test checks for,
// keeping pattern order within each group, and records where each group starts.
static void query_lowerer_group_entries(QueryLowerer *self) {
  QueryProgram *program = self->program;
  // An offset where each group starts, and one where the last group ends. There
  // is a group for each symbol of the language, symbol zero's holding the
  // entries that check for no specific symbol, and a last group for ERROR.
  array_grow_by(&program->symbol_offsets, ts_language_symbol_count(self->language) + 2);
  // Summing the sizes of the groups puts each group's offset where it ends.
  // Placing the entries from last to first then moves each offset back to
  // where its group starts, and keeps each group in pattern order.
  for (uint32_t i = 0; i < program->entries.size; i++) {
    const TSSymbol symbol = array_get(&program->entries, i)->symbol;
    (*array_get(&program->symbol_offsets, query_program_entry_group(program, symbol)))++;
  }
  for (uint32_t g = 1; g < program->symbol_offsets.size; g++) {
    *array_get(&program->symbol_offsets, g) += *array_get(&program->symbol_offsets, g - 1);
  }
  Array(QueryEntry) grouped = array_new();
  array_grow_by(&grouped, program->entries.size);
  for (int64_t i = (int64_t)program->entries.size - 1; i >= 0; i--) {
    const QueryEntry *entry = array_get(&program->entries, i);
    const uint32_t group = query_program_entry_group(program, entry->symbol);
    uint32_t *offset = array_get(&program->symbol_offsets, group);
    (*offset)--;
    *array_get(&grouped, *offset) = *entry;
  }
  array_swap(&program->entries, &grouped);
  array_delete(&grouped);
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
  const uint32_t ast_size = query_ast_size(ast);
  for (
    QueryAstIndex pattern = {.index = 0};
    succeeded && pattern.index < ast_size;
    pattern = query_ast_end(ast, pattern)
  ) {
    self.pattern = program->patterns.size;
    self.start_ast = pattern;
    succeeded = query_lowerer_lower_pattern(&self);
  }
  if (succeeded) query_lowerer_group_entries(&self);
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
    const QueryProgramCounter pattern_start =
      query_program_pattern_instruction_offset(*pattern);
    const QueryProgramCounter pattern_end =
      query_program_instruction_end(self, i);
    const QuerySpan span = array_get(&tables->patterns, i)->span;
    printf(
      "pattern %u%s: %.*s\n",
      i,
      pattern->is_rooted ? ", rooted" : "",
      (int)(span.end - span.start),
      &source[span.start]
    );
    for (uint32_t j = 0; j < self->entries.size; j++) {
      const QueryEntry *entry = array_get(&self->entries, j);
      if (entry->pattern != i) continue;
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
    while (
      label < self->labels.size &&
      array_get(&self->labels, label)->pc.index < pattern_start.index
    ) label++;
#endif
    for (
      QueryProgramCounter pc = pattern_start;
      pc.index < pattern_end.index;
      pc = query_program_counter_next(pc)
    ) {
#ifdef DEBUG_QUERY_BLOCKS
      // The blocks that start here, each one falling through to the next.
      bool has_label = false;
      while (
        label < self->labels.size &&
        array_get(&self->labels, label)->pc.index == pc.index
      ) {
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
      const QueryInstruction instruction = *array_get(
        &self->instructions,
        pc.index
      );
      const QueryOpcode opcode = query_instruction_opcode(instruction);
      const uint32_t operand = query_instruction_operand(instruction);
      printf("  %4u: %-14s", pc.index, query_opcode_name(opcode));
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
