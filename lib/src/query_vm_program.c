#include "./query_vm.h"

void ts_query_vm_program_new(
  QueryVmProgram *self,
  QueryProgram program,
  uint32_t capture_count
) {
  *self = (QueryVmProgram) {
    .program = program,
    .disabled_patterns = array_new(),
    .disabled_captures = array_new(),
  };

  array_grow_by(&self->disabled_patterns, self->program.patterns.size);
  array_grow_by(&self->disabled_captures, capture_count);
}

void ts_query_vm_program_copy(QueryVmProgram *self, const QueryVmProgram *other) {
  *self = (QueryVmProgram) {
    .program = query_program_copy(&other->program),
    .disabled_patterns = array_new(),
    .disabled_captures = array_new(),
  };
  array_assign(&self->disabled_patterns, &other->disabled_patterns);
  array_assign(&self->disabled_captures, &other->disabled_captures);
}

void ts_query_vm_program_delete(QueryVmProgram *self) {
  query_program_delete(&self->program);
  array_delete(&self->disabled_patterns);
  array_delete(&self->disabled_captures);
}
