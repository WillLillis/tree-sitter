#ifndef TREE_SITTER_QUERY_COMPILER_H_
#define TREE_SITTER_QUERY_COMPILER_H_

#include "./query_diagnostic.h"
#include "./query_program.h"
#include "./query_resolver.h"
#include "tree_sitter/api.h"

#include <stdbool.h>
#include <stdint.h>

// Compiles the source of a query by running each stage of the compiler in turn,
// into `tables` and `program`, which the caller then owns. Returns false at the
// first error in the source, after describing it in `error`.
bool ts_query_compile(
    const TSLanguage *language,
    const char *source,
    uint32_t length,
    QueryTables *tables,
    QueryProgram *program,
    QueryDiagnostic *error
);

#endif  // TREE_SITTER_QUERY_COMPILER_H_
