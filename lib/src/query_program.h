#ifndef TREE_SITTER_QUERY_PROGRAM_H_
#define TREE_SITTER_QUERY_PROGRAM_H_

#include "./array.h"
#include "./query_diagnostic.h"
#include "tree_sitter/api.h"

#include <stdbool.h>
#include <stdint.h>

// #define DEBUG_QUERY_PROGRAM
// #define DEBUG_QUERY_BLOCKS

// The lowerer's blocks are shown in the dump of the program.
#if defined(DEBUG_QUERY_BLOCKS) && !defined(DEBUG_QUERY_PROGRAM)
#define DEBUG_QUERY_PROGRAM
#endif

// An operation of a query's program. A query cursor runs the program in
// threads, as it walks the tree depth first. A thread runs one instruction
// after another on the node where the walk is, until it waits for a later node,
// fails, or finishes a match. Depths are counted from the depth of the node
// where the thread's match started.
typedef enum {
  // Stop until the walk enters the next node at the operand's depth, and then
  // continue with that node. A node that may be skipped is given to a copy of
  // the thread instead, and the thread keeps waiting. Any node may be skipped.
  QueryOpcodeWait,           // depth, field, may capture
  // `Wait`, but only anonymous nodes may be skipped.
  QueryOpcodeWaitAnchored,   // depth, field, may capture
  // `Wait`, but no node may be skipped.
  QueryOpcodeWaitImmediate,  // depth, field, may capture
  // Fail unless the node's symbol is the operand.
  QueryOpcodeSymbol,         // symbol
  // Fail unless the node is wrapped in the supertype that is the operand.
  QueryOpcodeSupertype,      // symbol
  // Fail unless the node is in the field that is the operand.
  QueryOpcodeField,          // field
  // Fail if the node has a child in the field that is the operand.
  QueryOpcodeNoField,        // field
  // Fail unless the node is named.
  QueryOpcodeNamed,          // <no operand>
  // Fail if the node is an ERROR.
  QueryOpcodeNotError,       // <no operand>
  // Fail unless the node is MISSING.
  QueryOpcodeMissing,        // <no operand>
  // Fail if the node that the thread matched at the depth that is the operand,
  // the node itself or one of its ancestors, has a later named sibling.
  QueryOpcodeLast,           // depth
  // Add the node to the match, as the capture that is the operand.
  QueryOpcodeCapture,        // capture
  // Continue, and start a copy of the thread at the instruction that is the
  // operand, which runs on the same node after the thread stops.
  QueryOpcodeSplit,          // instruction
  // Continue at the instruction that is the operand.
  QueryOpcodeJump,           // instruction
  // Finish a match of the thread's pattern.
  QueryOpcodeAccept,         // <no operand>
} QueryOpcode;

// An instruction: an opcode in the low bits, and an operand in the rest.
//
//    31                              8 7        0
//   +---------------------------------+----------+
//   |             operand             |  opcode  |
//   +---------------------------------+----------+
//
// A wait's operand holds three values:
// - depth: the depth of the node that it waits for
// - field: a field that the node must be in, or zero if there is none
// - may capture: whether a thread that waits there can still capture a node.
//
// The field repeats a `Field` test after the wait, so that the wait alone tells
// whether a later sibling could still pass the tests.
//
//    31            30          15 14      8 7        0
//   +-------------+--------------+---------+----------+
//   | may capture |    field     |  depth  |  opcode  |
//   +-------------+--------------+---------+----------+
//
// A `Last` instruction stores its depth in the same bits, with the rest zero.
typedef uint32_t QueryInstruction;

// The number of bits that an instruction uses to store its opcode.
#define QUERY_OPCODE_BITS 8

// The number of bits that an instruction uses to store its operand, which are
// also enough for the index of any instruction.
#define QUERY_OPERAND_BITS (32 - QUERY_OPCODE_BITS)

// The largest operand that an instruction can store.
#define QUERY_OPERAND_MAX ((1u << QUERY_OPERAND_BITS) - 1)

// The number of bits that a wait's operand uses to store its depth.
#define QUERY_DEPTH_BITS 7

// The largest depth that an instruction can store, which limits how deeply a
// pattern's nodes can nest.
#define QUERY_DEPTH_MAX ((1u << QUERY_DEPTH_BITS) - 1)

// The highest bit of a wait's operand, which says whether a thread that waits
// there can still capture a node.
#define QUERY_MAY_CAPTURE_BIT (1u << (QUERY_OPERAND_BITS - 1))

_Static_assert(
  QueryOpcodeAccept < (1 << QUERY_OPCODE_BITS),
  "QueryOpcode must fit in the low bits of a QueryInstruction"
);

_Static_assert(
  QUERY_DEPTH_BITS + sizeof(TSFieldId) * 8 + 1 == QUERY_OPERAND_BITS,
  "A wait's operand must hold its depth, its field, and one flag"
);

static inline QueryInstruction query_instruction_new(QueryOpcode opcode, uint32_t operand) {
  ts_assert(operand <= QUERY_OPERAND_MAX);
  return (QueryInstruction)opcode | operand << QUERY_OPCODE_BITS;
}

static inline QueryOpcode query_instruction_opcode(QueryInstruction self) {
  return (QueryOpcode)(self & ((1u << QUERY_OPCODE_BITS) - 1));
}

static inline uint32_t query_instruction_operand(QueryInstruction self) {
  return self >> QUERY_OPCODE_BITS;
}

static inline bool query_opcode_is_wait(QueryOpcode self) {
  return self == QueryOpcodeWait || self == QueryOpcodeWaitAnchored || self == QueryOpcodeWaitImmediate;
}

// Returns the operand of a wait for a node at `depth` that must be in `field`,
// unless it is zero. Its flag is clear: a thread that waits there cannot
// capture a node.
static inline uint32_t query_wait_operand(uint32_t depth, TSFieldId field) {
  ts_assert(depth <= QUERY_DEPTH_MAX);
  return (uint32_t)field << QUERY_DEPTH_BITS | depth;
}

// Returns the wait `self` with its flag set: a thread that waits there can
// still capture a node.
static inline QueryInstruction query_instruction_with_may_capture(QueryInstruction self) {
  ts_assert(query_opcode_is_wait(query_instruction_opcode(self)));
  return query_instruction_new(
    query_instruction_opcode(self),
    query_instruction_operand(self) | QUERY_MAY_CAPTURE_BIT
  );
}

// The depth that a wait or a `Last` instruction is about.
static inline uint32_t query_instruction_depth(QueryInstruction self) {
  ts_assert(
    query_opcode_is_wait(query_instruction_opcode(self)) ||
    query_instruction_opcode(self) == QueryOpcodeLast
  );
  return query_instruction_operand(self) & QUERY_DEPTH_MAX;
}

// A field that the node of a wait must be in, or zero if there is none.
static inline TSFieldId query_instruction_wait_field(QueryInstruction self) {
  ts_assert(query_opcode_is_wait(query_instruction_opcode(self)));
  return (TSFieldId)((query_instruction_operand(self) & ~QUERY_MAY_CAPTURE_BIT) >> QUERY_DEPTH_BITS);
}

// Whether a thread that waits at a wait can still capture a node.
static inline bool query_instruction_may_capture(QueryInstruction self) {
  ts_assert(query_opcode_is_wait(query_instruction_opcode(self)));
  return query_instruction_operand(self) & QUERY_MAY_CAPTURE_BIT;
}

// The symbol that a `Symbol` or `Supertype` instruction tests for.
static inline TSSymbol query_instruction_symbol(QueryInstruction self) {
  ts_assert(
    query_instruction_opcode(self) == QueryOpcodeSymbol ||
    query_instruction_opcode(self) == QueryOpcodeSupertype
  );
  return (TSSymbol)query_instruction_operand(self);
}

// The field that a `Field` or `NoField` instruction tests for.
static inline TSFieldId query_instruction_field(QueryInstruction self) {
  ts_assert(
    query_instruction_opcode(self) == QueryOpcodeField ||
    query_instruction_opcode(self) == QueryOpcodeNoField
  );
  return (TSFieldId)query_instruction_operand(self);
}

// The capture that a `Capture` instruction adds the node to the match as.
static inline uint16_t query_instruction_capture(QueryInstruction self) {
  ts_assert(query_instruction_opcode(self) == QueryOpcodeCapture);
  // INVARIANT: The resolver limits a query's capture names to 16-bit ids.
  const uint32_t cap = query_instruction_operand(self);
  ts_assert(cap <= UINT16_MAX);
  return (uint16_t)cap;
}

// The instruction that a `Split` or `Jump` instruction goes to.
static inline uint32_t query_instruction_target(QueryInstruction self) {
  ts_assert(
    query_instruction_opcode(self) == QueryOpcodeSplit ||
    query_instruction_opcode(self) == QueryOpcodeJump
  );
  return query_instruction_operand(self);
}

// A node that can be the first node of a match of a pattern, where a thread
// can start.
typedef struct {
  // The first instruction that tests the node. Like the index of any
  // instruction, it fits in an operand.
  uint32_t pc : QUERY_OPERAND_BITS;
  // Whether no match that starts at this node has another node at the top
  // level of the pattern.
  uint32_t is_rooted : 1;
  // The index of the pattern.
  uint16_t pattern;
  // The symbol that the node must have, or zero if any symbol can pass the
  // node's tests.
  TSSymbol symbol;
} QueryEntry;

_Static_assert(
  sizeof(QueryEntry) == 8,
  "QueryEntry packs its instruction and flag into one word"
);

// Where one pattern's instructions and entries start in its program. They end
// where the next pattern's start, or at the end of the program.
typedef struct {
  // The pattern's first instruction.
  uint32_t instruction_offset : QUERY_OPERAND_BITS;
  // Whether every entry of the pattern is rooted.
  uint32_t is_rooted : 1;
  // Whether a match can start inside a hidden repetition node, with later
  // top-level nodes after it, so that a walk over a range must look inside such
  // nodes before the range.
  uint32_t is_non_local : 1;
  // The pattern's first entry.
  uint32_t entry_offset;
} QueryProgramPattern;

_Static_assert(
  sizeof(QueryProgramPattern) == 8,
  "QueryProgramPattern packs its offset and flag into one word"
);

#ifdef DEBUG_QUERY_BLOCKS
// Where one of the blocks that the lowerer cut a pattern into starts, so that
// the program's dump can show it.
typedef struct {
  uint32_t pc;
  // The kind of block, and the edge or wait that it is for, if any.
  const char *kind;
  const char *variant;
  // The index of the syntax tree node that the block belongs to, and the span
  // of its token, or of its atom's token if it is a pattern.
  uint32_t node;
  QuerySpan span;
} QueryProgramLabel;
#endif

// The instructions of a query's patterns, one pattern after another, and where
// each pattern can start.
typedef struct {
  Array(QueryInstruction) instructions;
  Array(QueryEntry) entries;
  Array(QueryProgramPattern) patterns;
  // The hidden repetition symbols that a non-local pattern can start inside,
  // sorted.
  Array(TSSymbol) repetition_symbols;
#ifdef DEBUG_QUERY_BLOCKS
  // Ordered by `pc`.
  Array(QueryProgramLabel) labels;
#endif
} QueryProgram;

static inline QueryProgram query_program_new(void) {
  return (QueryProgram) {
    .instructions = array_new(),
    .entries = array_new(),
    .patterns = array_new(),
    .repetition_symbols = array_new(),
#ifdef DEBUG_QUERY_BLOCKS
    .labels = array_new(),
#endif
  };
}

// Returns the index of the instruction after the last one of `pattern`.
static inline uint32_t query_program_instruction_end(const QueryProgram *self, uint32_t pattern) {
  return pattern + 1 < self->patterns.size
    ? array_get(&self->patterns, pattern + 1)->instruction_offset
    : self->instructions.size;
}

// Returns the index of the entry after the last one of `pattern`.
static inline uint32_t query_program_entry_end(const QueryProgram *self, uint32_t pattern) {
  return pattern + 1 < self->patterns.size
    ? array_get(&self->patterns, pattern + 1)->entry_offset
    : self->entries.size;
}

static inline void query_program_delete(QueryProgram *self) {
  array_delete(&self->instructions);
  array_delete(&self->entries);
  array_delete(&self->patterns);
  array_delete(&self->repetition_symbols);
#ifdef DEBUG_QUERY_BLOCKS
  array_delete(&self->labels);
#endif
}

#endif  // TREE_SITTER_QUERY_PROGRAM_H_
