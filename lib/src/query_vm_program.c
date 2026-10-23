#include "./query_vm.h"
#include "./language.h"

void ts_query_vm_program_new(
  QueryVmProgram *self,
  QueryProgram program,
  const TSLanguage *language,
  uint32_t capture_count
) {
  *self = (QueryVmProgram) {
    .program = program,
    .symbol_entries = array_new(),
    .symbol_offsets = array_new(),
    .error_entries = array_new(),
    .wildcard_entries = array_new(),
    .disabled_patterns = array_new(),
    .disabled_captures = array_new(),
  };

  // A counting sort by symbol, which keeps each symbol's entries in pattern
  // order.
  const uint32_t symbol_count = ts_language_symbol_count(language);
  array_grow_by(&self->symbol_offsets, symbol_count + 1);
  for (uint32_t i = 0; i < self->program.entries.size; i++) {
    const QueryEntry *entry = array_get(&self->program.entries, i);
    if (entry->symbol == 0) {
      array_push(&self->wildcard_entries, *entry);
    } else if (entry->symbol == ts_builtin_sym_error) {
      array_push(&self->error_entries, *entry);
    } else {
      (*array_get(&self->symbol_offsets, entry->symbol + 1))++;
    }
  }
  for (uint32_t s = 1; s <= symbol_count; s++) {
    *array_get(&self->symbol_offsets, s) += *array_get(&self->symbol_offsets, s - 1);
  }
  array_grow_by(&self->symbol_entries, *array_get(&self->symbol_offsets, symbol_count));
  for (uint32_t i = 0; i < self->program.entries.size; i++) {
    const QueryEntry *entry = array_get(&self->program.entries, i);
    if (entry->symbol == 0 || entry->symbol == ts_builtin_sym_error) continue;
    // The offset of each symbol counts up as its entries are placed, and ends
    // where the next symbol's entries start; it is moved back below.
    uint32_t *offset = array_get(&self->symbol_offsets, entry->symbol);
    *array_get(&self->symbol_entries, *offset) = *entry;
    (*offset)++;
  }
  for (uint32_t s = symbol_count; s > 0; s--) {
    *array_get(&self->symbol_offsets, s) = *array_get(&self->symbol_offsets, s - 1);
  }
  *array_get(&self->symbol_offsets, 0) = 0;
  array_grow_by(&self->disabled_patterns, self->program.patterns.size);
  array_grow_by(&self->disabled_captures, capture_count);

  for (uint32_t pc = 0; pc < self->program.instructions.size; pc++) {
    const QueryInstruction instruction = *array_get(&self->program.instructions, pc);
    if (query_instruction_opcode(instruction) == QueryOpcodeLast) self->needs_path = true;
  }
}

void ts_query_vm_program_copy(QueryVmProgram *self, const QueryVmProgram *other) {
  *self = (QueryVmProgram) {
    .program = query_program_new(),
    .symbol_entries = array_new(),
    .symbol_offsets = array_new(),
    .error_entries = array_new(),
    .wildcard_entries = array_new(),
    .disabled_patterns = array_new(),
    .disabled_captures = array_new(),
    .needs_path = other->needs_path,
  };
  array_assign(&self->program.instructions, &other->program.instructions);
  array_assign(&self->program.entries, &other->program.entries);
  array_assign(&self->program.patterns, &other->program.patterns);
  array_assign(&self->program.repetition_symbols, &other->program.repetition_symbols);
  array_assign(&self->symbol_entries, &other->symbol_entries);
  array_assign(&self->symbol_offsets, &other->symbol_offsets);
  array_assign(&self->error_entries, &other->error_entries);
  array_assign(&self->wildcard_entries, &other->wildcard_entries);
  array_assign(&self->disabled_patterns, &other->disabled_patterns);
  array_assign(&self->disabled_captures, &other->disabled_captures);
}

void ts_query_vm_program_delete(QueryVmProgram *self) {
  query_program_delete(&self->program);
  array_delete(&self->symbol_entries);
  array_delete(&self->symbol_offsets);
  array_delete(&self->error_entries);
  array_delete(&self->wildcard_entries);
  array_delete(&self->disabled_patterns);
  array_delete(&self->disabled_captures);
}
