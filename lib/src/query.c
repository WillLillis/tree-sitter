/*
 * On NetBSD, defining standard requirements like this removes symbols
 * from the namespace; however, we need non-standard symbols for
 * endian.h.
 */
#if defined(__NetBSD__) && defined(_POSIX_C_SOURCE)
#undef _POSIX_C_SOURCE
#endif

#include "tree_sitter/api.h"
#include "./alloc.h"
#include "./array.h"
#include "./language.h"
#include "./point.h"
#include "./query_compiler.h"
#include "./query_vm.h"

/*
 * TSQuery - A tree query, compiled from a string of S-expressions. The query
 * itself is immutable. The mutable state used in the process of executing the
 * query is stored in a `TSQueryCursor`.
 */
struct TSQuery {
  const TSLanguage *language;
  // The tables that the query's API reads, and the program that cursors run.
  QueryTables tables;
  QueryVmProgram vm_program;
};

/*
 * TSQueryCursor - A stateful struct used to execute a query on a tree.
 */
struct TSQueryCursor {
  QueryVm vm;
};

// Reports a diagnostic of the compiler as the query API's error type and
// offset. The offset is where the diagnostic's span starts, except for:
// * An unknown name, which starts inside the quotes of a string and after the
//   `@` of a capture, since bindings read the name from the offset.
// * A missing name or suffix, which is reported where it belongs, at the end
//   of the span, and unexpected space, which is reported at the token after it.
static void ts_query__report_error(
  const QueryDiagnostic *diagnostic,
  const char *source,
  uint32_t *error_offset,
  TSQueryError *error_type
) {
  const QuerySpan span = diagnostic->span;
  *error_offset = span.start;
  switch (diagnostic->kind) {
    case QueryDiagnosticKindUnknownNodeType:
      *error_type = TSQueryErrorNodeType;
      if (source[span.start] == '"') *error_offset = span.start + 1;
      return;
    case QueryDiagnosticKindUnknownField:
      *error_type = TSQueryErrorField;
      return;
    case QueryDiagnosticKindUndefinedCapture:
      *error_type = TSQueryErrorCapture;
      *error_offset = span.start + 1;
      return;
    case QueryDiagnosticKindNotASupertype:
    case QueryDiagnosticKindNotASubtype:
    case QueryDiagnosticKindImpossiblePattern:
    case QueryDiagnosticKindPatternTooLarge:
    case QueryDiagnosticKindTooManyPatterns:
    case QueryDiagnosticKindTooManyCaptures:
      *error_type = TSQueryErrorStructure;
      return;
    case QueryDiagnosticKindMissingCaptureName:
    case QueryDiagnosticKindMissingPredicateName:
    case QueryDiagnosticKindMissingPredicateSuffix:
    case QueryDiagnosticKindUnexpectedSpace:
      *error_type = TSQueryErrorSyntax;
      *error_offset = span.end;
      return;
    case QueryDiagnosticKindUnexpectedCharacter:
    case QueryDiagnosticKindNonAsciiCharacter:
    case QueryDiagnosticKindInvalidUtf8:
    case QueryDiagnosticKindUnterminatedString:
    case QueryDiagnosticKindUnknownEscape:
    case QueryDiagnosticKindTooManyTokens:
    case QueryDiagnosticKindExpectedPattern:
    case QueryDiagnosticKindExpectedNodeName:
    case QueryDiagnosticKindExpectedSubtype:
    case QueryDiagnosticKindExpectedFieldName:
    case QueryDiagnosticKindExpectedPredicateName:
    case QueryDiagnosticKindExpectedPredicateArgument:
    case QueryDiagnosticKindExpectedRightParen:
    case QueryDiagnosticKindUnclosedDelimiter:
    case QueryDiagnosticKindUnexpectedClosingDelimiter:
    case QueryDiagnosticKindMultipleFields:
    case QueryDiagnosticKindMisplacedAnchor:
    case QueryDiagnosticKindMisplacedPredicate:
      *error_type = TSQueryErrorSyntax;
      return;
  }
}

TSQuery *ts_query_new(
  const TSLanguage *language,
  const char *source,
  uint32_t source_len,
  uint32_t *error_offset,
  TSQueryError *error_type
) {
  if (
    !language ||
    language->abi_version > TREE_SITTER_LANGUAGE_VERSION ||
    language->abi_version < TREE_SITTER_MIN_COMPATIBLE_LANGUAGE_VERSION
  ) {
    *error_type = TSQueryErrorLanguage;
    return NULL;
  }

  QueryTables tables;
  QueryProgram program;
  QueryDiagnostic diagnostic;
  if (!ts_query_compile(language, source, source_len, &tables, &program, &diagnostic)) {
    ts_query__report_error(&diagnostic, source, error_offset, error_type);
    return NULL;
  }

  TSQuery *self = ts_malloc(sizeof(TSQuery));
  *self = (TSQuery) {
    .language = ts_language_copy(language),
    .tables = tables,
  };
  ts_query_vm_program_new(&self->vm_program, program, self->language, ts_query_capture_count(self));
  *error_type = TSQueryErrorNone;
  return self;
}

void ts_query_delete(TSQuery *self) {
  if (self) {
    query_tables_delete(&self->tables);
    ts_query_vm_program_delete(&self->vm_program);
    ts_language_delete(self->language);
    ts_free(self);
  }
}

TSQuery *ts_query_copy(const TSQuery *self) {
  TSQuery *copy = ts_malloc(sizeof(TSQuery));
  *copy = (TSQuery) {.language = ts_language_copy(self->language)};
  query_tables_copy(&copy->tables, &self->tables);
  ts_query_vm_program_copy(&copy->vm_program, &self->vm_program);
  return copy;
}

uint32_t ts_query_pattern_count(const TSQuery *self) {
  return self->tables.patterns.size;
}

uint32_t ts_query_capture_count(const TSQuery *self) {
  return query_string_table_size(&self->tables.captures);
}

uint32_t ts_query_string_count(const TSQuery *self) {
  return query_string_table_size(&self->tables.strings);
}

const char *ts_query_capture_name_for_id(
  const TSQuery *self,
  uint32_t index,
  uint32_t *length
) {
  return query_string_table_get(&self->tables.captures, index, length);
}

TSQuantifier ts_query_capture_quantifier_for_id(
  const TSQuery *self,
  uint32_t pattern_index,
  uint32_t capture_index
) {
  const QueryPatternInfo *pattern = array_get(&self->tables.patterns, pattern_index);
  for (uint32_t i = 0; i < pattern->capture_quantifier_count; i++) {
    const QueryCaptureQuantifier *entry = array_get(
      &self->tables.capture_quantifiers,
      pattern->capture_quantifier_offset + i
    );
    if (entry->capture == capture_index) return (TSQuantifier)entry->quantifier;
  }
  return TSQuantifierZero;
}

const char *ts_query_string_value_for_id(
  const TSQuery *self,
  uint32_t index,
  uint32_t *length
) {
  return query_string_table_get(&self->tables.strings, index, length);
}

const TSQueryPredicateStep *ts_query_predicates_for_pattern(
  const TSQuery *self,
  uint32_t pattern_index,
  uint32_t *step_count
) {
  const QueryPatternInfo *pattern = array_get(&self->tables.patterns, pattern_index);
  *step_count = pattern->predicate_step_count;
  if (pattern->predicate_step_count == 0) return NULL;
  return array_get(&self->tables.predicate_steps, pattern->predicate_step_offset);
}

uint32_t ts_query_start_byte_for_pattern(
  const TSQuery *self,
  uint32_t pattern_index
) {
  return array_get(&self->tables.patterns, pattern_index)->span.start;
}

uint32_t ts_query_end_byte_for_pattern(
  const TSQuery *self,
  uint32_t pattern_index
) {
  return array_get(&self->tables.patterns, pattern_index)->span.end;
}

bool ts_query_is_pattern_rooted(
  const TSQuery *self,
  uint32_t pattern_index
) {
  const QueryProgram *program = &self->vm_program.program;
  if (pattern_index >= program->patterns.size) return true;
  return array_get(&program->patterns, pattern_index)->is_rooted;
}

bool ts_query_is_pattern_non_local(
  const TSQuery *self,
  uint32_t pattern_index
) {
  const QueryProgram *program = &self->vm_program.program;
  if (pattern_index >= program->patterns.size) return false;
  return array_get(&program->patterns, pattern_index)->is_non_local;
}

bool ts_query_is_pattern_guaranteed_at_step(
  const TSQuery *self,
  uint32_t byte_offset
) {
  (void)self;
  (void)byte_offset;
  return false;
}

void ts_query_disable_capture(
  TSQuery *self,
  const char *name,
  uint32_t length
) {
  uint32_t id;
  if (query_string_table_find(&self->tables.captures, name, length, &id)) {
    *array_get(&self->vm_program.disabled_captures, id) = true;
  }
}

void ts_query_disable_pattern(
  TSQuery *self,
  uint32_t pattern_index
) {
  if (pattern_index < self->vm_program.disabled_patterns.size) {
    *array_get(&self->vm_program.disabled_patterns, pattern_index) = true;
  }
}

/***************
 * QueryCursor
 ***************/

TSQueryCursor *ts_query_cursor_new(void) {
  TSQueryCursor *self = ts_malloc(sizeof(TSQueryCursor));
  ts_query_vm_new(&self->vm);
  return self;
}

void ts_query_cursor_delete(TSQueryCursor *self) {
  ts_query_vm_delete(&self->vm);
  ts_free(self);
}

bool ts_query_cursor_did_exceed_match_limit(const TSQueryCursor *self) {
  return self->vm.did_exceed_match_limit;
}

uint32_t ts_query_cursor_match_limit(const TSQueryCursor *self) {
  return self->vm.match_limit;
}

void ts_query_cursor_set_match_limit(TSQueryCursor *self, uint32_t limit) {
  self->vm.match_limit = limit;
}

void ts_query_cursor_exec(
  TSQueryCursor *self,
  const TSQuery *query,
  TSNode node
) {
  ts_query_vm_exec(&self->vm, &query->vm_program, node, NULL);
}

void ts_query_cursor_exec_with_options(
  TSQueryCursor *self,
  const TSQuery *query,
  TSNode node,
  const TSQueryCursorOptions *query_options
) {
  ts_query_vm_exec(&self->vm, &query->vm_program, node, query_options);
}

bool ts_query_cursor_set_byte_range(
  TSQueryCursor *self,
  uint32_t start_byte,
  uint32_t end_byte
) {
  if (end_byte == 0) {
    end_byte = UINT32_MAX;
  }
  if (start_byte > end_byte) {
    return false;
  }
  self->vm.included_range.start_byte = start_byte;
  self->vm.included_range.end_byte = end_byte;
  return true;
}

bool ts_query_cursor_set_point_range(
  TSQueryCursor *self,
  TSPoint start_point,
  TSPoint end_point
) {
  if (end_point.row == 0 && end_point.column == 0) {
    end_point = POINT_MAX;
  }
  if (point_gt(start_point, end_point)) {
    return false;
  }
  self->vm.included_range.start_point = start_point;
  self->vm.included_range.end_point = end_point;
  return true;
}

bool ts_query_cursor_set_containing_byte_range(
  TSQueryCursor *self,
  uint32_t start_byte,
  uint32_t end_byte
) {
  if (end_byte == 0) {
    end_byte = UINT32_MAX;
  }
  if (start_byte > end_byte) {
    return false;
  }
  self->vm.containing_range.start_byte = start_byte;
  self->vm.containing_range.end_byte = end_byte;
  return true;
}

bool ts_query_cursor_set_containing_point_range(
  TSQueryCursor *self,
  TSPoint start_point,
  TSPoint end_point
) {
  if (end_point.row == 0 && end_point.column == 0) {
    end_point = POINT_MAX;
  }
  if (point_gt(start_point, end_point)) {
    return false;
  }
  self->vm.containing_range.start_point = start_point;
  self->vm.containing_range.end_point = end_point;
  return true;
}

bool ts_query_cursor_next_match(
  TSQueryCursor *self,
  TSQueryMatch *match
) {
  return ts_query_vm_next_match(&self->vm, match);
}

void ts_query_cursor_remove_match(
  TSQueryCursor *self,
  uint32_t match_id
) {
  ts_query_vm_remove_match(&self->vm, match_id);
}

bool ts_query_cursor_next_capture(
  TSQueryCursor *self,
  TSQueryMatch *match,
  uint32_t *capture_index
) {
  return ts_query_vm_next_capture(&self->vm, match, capture_index);
}

void ts_query_cursor_set_max_start_depth(
  TSQueryCursor *self,
  uint32_t max_start_depth
) {
  self->vm.max_start_depth = max_start_depth;
}
