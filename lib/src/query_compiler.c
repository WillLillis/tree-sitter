#include "./query_compiler.h"
#include "./query_analysis.h"
#include "./query_lowerer.h"
#include "./query_parser.h"
#include "./query_program.h"
#include "./query_resolver.h"
#include "./query_tokenizer.h"

bool ts_query_compile(
    const TSLanguage *language,
    const char *source,
    uint32_t length,
    QueryTables *tables,
    QueryProgram *program,
    QueryDiagnostic *error
) {
  QueryTokenList tokens = query_token_list_new();
  bool succeeded = ts_query_tokenize(source, length, &tokens, error);
#ifdef DEBUG_QUERY_TOKENS
  ts_query_token_list_dump(&tokens, source);
#endif

  QueryAst ast = query_ast_new();
  if (succeeded) {
    succeeded = ts_query_parse(&tokens, &ast, error);
#ifdef DEBUG_QUERY_AST
    ts_query_ast_dump(&ast, &tokens, source);
#endif
  }

  QueryAstIds ids = query_ast_ids_new();
  *tables = query_tables_new();
  if (succeeded) {
    succeeded = ts_query_resolve(language, source, &tokens, &ast, &ids, tables, error);
#ifdef DEBUG_QUERY_RESOLUTION
    ts_query_resolution_dump(language, source, &tokens, &ast, &ids, tables);
#endif
  }

  *program = query_program_new();
  if (succeeded) {
    succeeded = ts_query_lower(language, &tokens, &ast, &ids, program, error);
#ifdef DEBUG_QUERY_PROGRAM
    ts_query_program_dump(program, language, tables, source);
#endif
  }

  if (succeeded) succeeded = ts_query_analyze(language, &tokens, &ast, &ids, program, error);

#ifdef DEBUG_QUERY_DIAGNOSTICS
  if (!succeeded) ts_query_diagnostic_dump(error, source);
#endif
  query_token_list_delete(&tokens);
  query_ast_delete(&ast);
  query_ast_ids_delete(&ids);
  if (!succeeded) {
    query_tables_delete(tables);
    query_program_delete(program);
  }
  return succeeded;
}
