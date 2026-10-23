#include "./query_diagnostic.h"

#ifdef DEBUG_QUERY_DIAGNOSTICS
#include <stdio.h>

static const char *query_diagnostic_kind_name(QueryDiagnosticKind kind) {
  switch (kind) {
    case QueryDiagnosticKindUnexpectedCharacter: return "UnexpectedCharacter";
    case QueryDiagnosticKindNonAsciiCharacter: return "NonAsciiCharacter";
    case QueryDiagnosticKindInvalidUtf8: return "InvalidUtf8";
    case QueryDiagnosticKindUnterminatedString: return "UnterminatedString";
    case QueryDiagnosticKindUnknownEscape: return "UnknownEscape";
    case QueryDiagnosticKindMissingCaptureName: return "MissingCaptureName";
    case QueryDiagnosticKindMissingPredicateName: return "MissingPredicateName";
    case QueryDiagnosticKindMissingPredicateSuffix: return "MissingPredicateSuffix";
    case QueryDiagnosticKindTooManyTokens: return "TooManyTokens";
    case QueryDiagnosticKindExpectedPattern: return "ExpectedPattern";
    case QueryDiagnosticKindExpectedNodeName: return "ExpectedNodeName";
    case QueryDiagnosticKindExpectedSubtype: return "ExpectedSubtype";
    case QueryDiagnosticKindExpectedFieldName: return "ExpectedFieldName";
    case QueryDiagnosticKindExpectedPredicateName: return "ExpectedPredicateName";
    case QueryDiagnosticKindExpectedPredicateArgument: return "ExpectedPredicateArgument";
    case QueryDiagnosticKindExpectedRightParen: return "ExpectedRightParen";
    case QueryDiagnosticKindUnclosedDelimiter: return "UnclosedDelimiter";
    case QueryDiagnosticKindUnexpectedClosingDelimiter: return "UnexpectedClosingDelimiter";
    case QueryDiagnosticKindUnexpectedSpace: return "UnexpectedSpace";
    case QueryDiagnosticKindMultipleFields: return "MultipleFields";
    case QueryDiagnosticKindMisplacedAnchor: return "MisplacedAnchor";
    case QueryDiagnosticKindMisplacedPredicate: return "MisplacedPredicate";
    case QueryDiagnosticKindUnknownNodeType: return "UnknownNodeType";
    case QueryDiagnosticKindUnknownField: return "UnknownField";
    case QueryDiagnosticKindUndefinedCapture: return "UndefinedCapture";
    case QueryDiagnosticKindNotASupertype: return "NotASupertype";
    case QueryDiagnosticKindNotASubtype: return "NotASubtype";
    case QueryDiagnosticKindImpossiblePattern: return "ImpossiblePattern";
    case QueryDiagnosticKindPatternTooLarge: return "PatternTooLarge";
    case QueryDiagnosticKindTooManyPatterns: return "TooManyPatterns";
    case QueryDiagnosticKindTooManyCaptures: return "TooManyCaptures";
  }
  return "?";
}

void ts_query_diagnostic_dump(const QueryDiagnostic *self, const char *source) {
  printf(
    "=== ERROR ===\n%s %u..%u %.*s\n",
    query_diagnostic_kind_name(self->kind),
    self->span.start,
    self->span.end,
    (int)(self->span.end - self->span.start),
    &source[self->span.start]
  );
}
#endif
