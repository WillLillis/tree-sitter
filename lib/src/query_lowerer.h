#ifndef TREE_SITTER_QUERY_LOWERER_H_
#define TREE_SITTER_QUERY_LOWERER_H_

#include "./query_diagnostic.h"
#include "./query_parser.h"
#include "./query_program.h"
#include "./query_resolver.h"
#include "./query_tokenizer.h"
#include "tree_sitter/api.h"

#include <stdbool.h>

// Lowers each pattern of `ast`, whose nodes resolved to `ids`, to instructions,
// and appends them, the pattern's entries, and where they are to `program`.
// Returns false at the first pattern that does not fit in a program, because it
// needs an operand larger than an instruction can store, or because the query
// already has as many patterns as it can, after describing it in `error`.
bool ts_query_lower(
  const TSLanguage *language,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  const QueryAstIds *ids,
  QueryProgram *program,
  QueryDiagnostic *error
);

#ifdef DEBUG_QUERY_PROGRAM
void ts_query_program_dump(
  const QueryProgram *self,
  const TSLanguage *language,
  const QueryTables *tables,
  const char *source
);
#endif

#endif  // TREE_SITTER_QUERY_LOWERER_H_
