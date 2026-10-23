#ifndef TREE_SITTER_QUERY_DIAGNOSTIC_H_
#define TREE_SITTER_QUERY_DIAGNOSTIC_H_

#include <stdint.h>

// #define DEBUG_QUERY_DIAGNOSTICS

// A range of bytes in the source of a query.
typedef struct {
  // The offset of the first byte in the range.
  uint32_t start;
  // The offset just past the last byte in the range.
  uint32_t end;
} QuerySpan;

// A kind of error in the source of a query.
typedef enum {
  // An ASCII character that cannot start a token.
  QueryDiagnosticKindUnexpectedCharacter,
  // A non-ASCII character outside of a string or a comment.
  QueryDiagnosticKindNonAsciiCharacter,
  // Bytes that are not valid UTF-8.
  QueryDiagnosticKindInvalidUtf8,
  // A string with no closing quote before the end of its line.
  QueryDiagnosticKindUnterminatedString,
  // A backslash in a string, followed by a character that it cannot escape.
  QueryDiagnosticKindUnknownEscape,
  // An `@` that is not followed by the name of a capture.
  QueryDiagnosticKindMissingCaptureName,
  // A `#` that is not followed by the name of a predicate.
  QueryDiagnosticKindMissingPredicateName,
  // The name of a predicate that does not end with `?` or `!`.
  QueryDiagnosticKindMissingPredicateSuffix,
  // The start of the first token past the most that a query can have.
  QueryDiagnosticKindTooManyTokens,

  // A token that cannot start a pattern, where a pattern must be.
  QueryDiagnosticKindExpectedPattern,
  // A token after a `(` that cannot start a node, a group, or a predicate.
  QueryDiagnosticKindExpectedNodeName,
  // A token after a `/` that cannot be the name of a subtype.
  QueryDiagnosticKindExpectedSubtype,
  // A token after a `!` that cannot be the name of a field.
  QueryDiagnosticKindExpectedFieldName,
  // A token after a `(.` that cannot be the name of a predicate.
  QueryDiagnosticKindExpectedPredicateName,
  // A token in a predicate that is not a capture, a string, or a bare word.
  QueryDiagnosticKindExpectedPredicateArgument,
  // A token where only a `)` can be, such as a child of a MISSING node.
  QueryDiagnosticKindExpectedRightParen,
  // A `(` or `[` that is never closed.
  QueryDiagnosticKindUnclosedDelimiter,
  // A `)` or `]` that does not close the innermost open `(` or `[`.
  QueryDiagnosticKindUnexpectedClosingDelimiter,
  // Space between two tokens that must touch, as in `supertype/subtype`.
  QueryDiagnosticKindUnexpectedSpace,
  // A second field before the same pattern, as in `a: b: (c)`.
  QueryDiagnosticKindMultipleFields,
  // An anchor where it has nothing to anchor.
  QueryDiagnosticKindMisplacedAnchor,
  // A predicate that is not inside a named node or a group, or that follows a
  // field.
  QueryDiagnosticKindMisplacedPredicate,

  // A name or string that is not a node type in the language.
  QueryDiagnosticKindUnknownNodeType,
  // A name that is not a field in the language.
  QueryDiagnosticKindUnknownField,
  // A capture in a predicate that the predicate's pattern does not define.
  QueryDiagnosticKindUndefinedCapture,
  // The name before the `/` in `(supertype/subtype)`, which is not a supertype.
  QueryDiagnosticKindNotASupertype,
  // The name after the `/` in `(supertype/subtype)`, which is not a node type
  // that the supertype wraps.
  QueryDiagnosticKindNotASubtype,
  // A child pattern that no node of its parent's type can have, a parent whose
  // type the language can never produce, or a supertype after the `/` in
  // `(supertype/subtype)`, which is never a node's type.
  QueryDiagnosticKindImpossiblePattern,

  // A pattern that nests too deeply, or that makes the query need more
  // instructions or captures than an instruction's operand can count.
  QueryDiagnosticKindPatternTooLarge,
  // The first pattern past the most that a query can have, which is as many
  // as a match's 16-bit pattern index can count.
  QueryDiagnosticKindTooManyPatterns,
  // The first capture name past the most that a query can have, which is as
  // many as a capture's 16-bit id can count.
  QueryDiagnosticKindTooManyCaptures,
} QueryDiagnosticKind;

// An error in the source of a query.
typedef struct {
  QuerySpan span;
  QueryDiagnosticKind kind;
} QueryDiagnostic;

#ifdef DEBUG_QUERY_DIAGNOSTICS
// Prints `self`: its kind, its span, and the text of the source that it covers.
void ts_query_diagnostic_dump(const QueryDiagnostic *self, const char *source);
#endif


#endif  // TREE_SITTER_QUERY_DIAGNOSTIC_H_
