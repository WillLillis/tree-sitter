#ifndef TREE_SITTER_QUERY_ANALYSIS_H_
#define TREE_SITTER_QUERY_ANALYSIS_H_

#include "./query_diagnostic.h"
#include "./query_parser.h"
#include "./query_program.h"
#include "./query_resolver.h"
#include "./query_tokenizer.h"
#include "tree_sitter/api.h"

#include <stdbool.h>

// Checks that each pattern can match some tree of `language`, from what the
// parse table says about the children that each node type can have, and marks
// the non-local patterns of `program`, the program lowered from `ast`, with the
// repetitions that they can start inside. Returns false at the first pattern
// that can never match, after describing it in `error`.
bool ts_query_analyze(
  const TSLanguage *language,
  const QueryTokenList *tokens,
  const QueryAst *ast,
  const QueryAstIds *ids,
  QueryProgram *program,
  QueryDiagnostic *error
);

#endif  // TREE_SITTER_QUERY_ANALYSIS_H_
