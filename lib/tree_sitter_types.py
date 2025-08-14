# TODO: See if we need the typing import
from typing import List
import re
import lldb
import lldb.formatters.Logger
from lldb import (
    SBValue,
    # SBData,
)


class TreeSitterType(object):
    # api.h
    # TODO: Move these type defs down once we find their actual definition files
    TS_PARSER = "TSParser"
    TS_TREE = "TSTree"
    TS_QUERY = "TSQuery"
    TS_QUERY_CURSOR = "TSQueryCursor"
    TS_LOOKAHEAD_ITERATOR = "TSLookaheadIterator"
    TS_INPUT_ENCODING = "TSInputEncoding"
    TS_SYMBOL_TYPE = "TSSymbolType"
    TS_POINT = "TSPoint"
    TS_RANGE = "TSRange"
    TS_INPUT = "TSInput"
    TS_PARSE_STATE = "TSParseState"
    TS_LOG_TYPE = "TSLogType"
    TS_LOGGER = "TSLogger"
    TS_INPUT_EDITOR = "TSInputEdit"
    TS_QUERY_CAPTURE = "TSQueryCapture"
    TS_QUANTIFIER = "TSQuantifier"
    TS_QUERY_MATCH = "TSQueryMatch"
    TS_QUERY_PREDICATE_STEP_TYPE = "TSQueryPredicateStepType"
    TS_QUERY_ERROR = "TSQueryError"
    TS_QUERY_CURSOR_STATE = "TSQueryCursorState"
    TS_QUERY_CURSOR_OPTIONS = "TSQueryCursorOptions"
    TS_LANGUAGE_METADATA = "TSLanguageMetadata"
    TS_WASM_ERROR_KIND = "TSWasmErrorKind"
    TS_WASM_ERROR = "TSWasmError"
    # tree_cursor.h
    TREE_CURSOR = "TreeCursor"
    TREE_CURSOR_STEP = "TreeCursorStep"
    PARENT_CACHE_ENTRY = "ParentCacheEntry"
    TS_TREE = "TSTree"
    # subtree.h
    EXTERNAL_SCANNER_STATE = "ExternalScannerState"
    SUBTREE_INLINE_DATA = "SubtreeInlineData"
    SUBTREE_HEAP_DATA = "SubtreeHeapData"
    SUBTREE = "Subtree"
    MUTABLE_SUBTREE = "MutableSubtree"
    SUBTREE_ARRAY = "SubtreeArray"
    MUTABLE_SUBTREE_ARRAY = "MutableSubtreeArray"
    SUBTREE_POOL = "SubtreePool"
    # stack.h
    STACK_SLICE = "StackSlice"
    STACK_SLICE_ARRAY = "StackSliceArray"
    STACK_SUMMARY_ENTRY = "StackSummaryEntry"
    STACK_SUMMARY = "StackSummary"
    # reusable_node.h
    STACK_ENTRY = "StackEntry"
    REUSABLE_NODE = "ReusableNode"
    # reduce_action.h
    REDUCE_ACTION = "ReduceAction"
    REDUCE_ACTION_SET = "ReduceActionSet" # Array(ReduceAction)
    # parser.h
    TS_FIELD_MAP_ENTRY = "TSFieldMapEntry"
    TS_MAP_SLICE = "TSMapSlice"
    TS_SYMBOL_METADATA = "TSSymbolMetadata"
    TS_LEXER = "TSLexer"
    TS_PARSE_ACTION_TYPE = "TSParseActionType"
    TS_PARSE_ACTION = "TSParseAction"
    TS_LEX_MODE = "TSLexMode"
    TS_LEXER_MODE = "TSLexerMode"
    TS_PARSE_ACTION_ENTRY = "TSParseActionEntry"
    TS_CHARACTER_RANGE = "TSCharacterRange"
    TS_LANGUAGE = "TSLanguage"
    # lexer.h
    COLUMN_DATA = "ColumnData"
    LEXER = "Lexer"
    # length.h
    LENGTH = "Length"
    # language.h
    TABLE_ENTRY = "TableEntry"
    LOOKAHEAD_ITERATOR = "LookaheadIterator"
    TS_RANGE_ARRAY = "TSRangeArray"

# Regexes for all the types
#TS_PARSER_REGEX = r"^TSParser$|^struct TSParser$"
TS_QUERY_REGEX = r"^TSQuery$|^struct TSQuery$|^typedef TSQuery$"
TS_QUERY_CURSOR_REGEX = r"^TSQueryCursor$|^struct TSQueryCursor$|^typedef TSQueryCursor$"
TS_LOOKAHEAD_ITERATOR_REGEX = r"^TSLookaheadIterator$|^struct TSLookaheadIterator$|^typedef TSLookaheadIterator$"
TS_INPUT_ENCODING_REGEX = r"^TSInputEncoding$|^struct TSInputEncoding$|^typedef TSInputEncoding$"
TS_SYMBOL_TYPE_REGEX = r"^TSSymbolType$|^struct TSSymbolType$|^typedef TSSymbolType$"
TS_POINT_REGEX = r"^TSPoint$|^struct TSPoint$|^typedef TSPoint$"
TS_RANGE_REGEX = r"^TSRange$|^struct TSRange$|^typedef TSRange$"
TS_INPUT_REGEX = r"^TSInput$|^struct TSInput$|^typedef TSInput$"
TS_PARSE_STATE_REGEX = r"^TSParseState$|^struct TSParseState$|^typedef TSParseState$"
TS_LOG_TYPE_REGEX = r"^TSLogType$|^struct TSLogType$|^typedef TSLogType$"
TS_LOGGER_REGEX = r"^TSLogger$|^struct TSLogger$|^typedef TSLogger$"
TS_INPUT_EDITOR_REGEX = r"^TSInputEdit$|^struct TSInputEdit$|^typedef TSInputEdit$"
TS_QUERY_CAPTURE_REGEX = r"^TSQueryCapture$|^struct TSQueryCapture$|^typedef TSQueryCapture$"
TS_QUANTIFIER_REGEX = r"^TSQuantifier$|^struct TSQuantifier$|^typedef TSQuantifier$"
TS_QUERY_MATCH_REGEX = r"^TSQueryMatch$|^struct TSQueryMatch$|^typedef TSQueryMatch$"
TS_QUERY_PREDICATE_STEP_TYPE_REGEX = r"^TSQueryPredicateStepType$|^struct TSQueryPredicateStepType$|^typedef TSQueryPredicateStepType$"
TS_QUERY_ERROR_REGEX = r"^TSQueryError$|^struct TSQueryError$|^typedef TSQueryError$"
TS_QUERY_CURSOR_STATE_REGEX = r"^TSQueryCursorState$|^struct TSQueryCursorState$|^typedef TSQueryCursorState$"
TS_QUERY_CURSOR_OPTIONS_REGEX = r"^TSQueryCursorOptions$|^struct TSQueryCursorOptions$|^typedef TSQueryCursorOptions$"
TS_LANGUAGE_METADATA_REGEX = r"^TSLanguageMetadata$|^struct TSLanguageMetadata$|^typedef TSLanguageMetadata$"
TS_WASM_ERROR_KIND_REGEX = r"^TSWasmErrorKind$|^struct TSWasmErrorKind$|^typedef TSWasmErrorKind$"
TS_WASM_ERROR_REGEX = r"^TSWasmError$|^struct TSWasmError$|^typedef TSWasmError$"
TREE_CURSOR_REGEX = r"^TreeCursor$|^struct TreeCursor$|^typedef TreeCursor$"
TREE_CURSOR_STEP_REGEX = r"^TreeCursorStep$|^struct TreeCursorStep$|^typedef TreeCursorStep$"
PARENT_CACHE_ENTRY_REGEX = r"^ParentCacheEntry$|^struct ParentCacheEntry$|^typedef ParentCacheEntry$"
TS_TREE_REGEX = r"^TSTree$|^struct TSTree$|^typedef TSTree$"
EXTERNAL_SCANNER_STATE_REGEX = r"^ExternalScannerState$|^struct ExternalScannerState$|^typedef ExternalScannerState$"
SUBTREE_INLINE_DATA_REGEX = r"^SubtreeInlineData$|^struct SubtreeInlineData$|^typedef SubtreeInlineData$"
SUBTREE_HEAP_DATA_REGEX = r"^SubtreeHeapData$|^struct SubtreeHeapData$|^typedef SubtreeHeapData$"
SUBTREE_REGEX = r"^Subtree$|^struct Subtree$|^typedef Subtree$"
MUTABLE_SUBTREE_REGEX = r"^MutableSubtree$|^struct MutableSubtree$|^typedef MutableSubtree$"
SUBTREE_ARRAY_REGEX = r"^SubtreeArray$|^struct SubtreeArray$|^typedef SubtreeArray$"
MUTABLE_SUBTREE_ARRAY_REGEX = r"^MutableSubtreeArray$|^struct MutableSubtreeArray$|^typedef MutableSubtreeArray$"
SUBTREE_POOL_REGEX = r"^SubtreePool$|^struct SubtreePool$|^typedef SubtreePool$"
STACK_SLICE_REGEX = r"^StackSlice$|^struct StackSlice$|^typedef StackSlice$"
STACK_SLICE_ARRAY_REGEX = r"^StackSliceArray$|^struct StackSliceArray$|^typedef StackSliceArray$"
STACK_SUMMARY_ENTRY_REGEX = r"^StackSummaryEntry$|^struct StackSummaryEntry$|^typedef StackSummaryEntry$"
STACK_SUMMARY_REGEX = r"^StackSummary$|^struct StackSummary$|^typedef StackSummary$"
STACK_ENTRY_REGEX = r"^StackEntry$|^struct StackEntry$|^typedef StackEntry$"
REUSABLE_NODE_REGEX = r"^ReusableNode$|^struct ReusableNode$|^typedef ReusableNode$"
REDUCE_ACTION_REGEX = r"^ReduceAction$|^struct ReduceAction$|^typedef ReduceAction$"
REDUCE_ACTION_SET_REGEX = r"^ReduceActionSet$|^struct ReduceActionSet$|^typedef ReduceActionSet$"
TS_FIELD_MAP_ENTRY_REGEX = r"^TSFieldMapEntry$|^struct TSFieldMapEntry$|^typedef TSFieldMapEntry$"
TS_MAP_SLICE_REGEX = r"^TSMapSlice$|^struct TSMapSlice$|^typedef TSMapSlice$"
TS_SYMBOL_METADATA_REGEX = r"^TSSymbolMetadata$|^struct TSSymbolMetadata$|^typedef TSSymbolMetadata$"
TS_LEXER_REGEX = r"^TSLexer$|^struct TSLexer$|^typedef TSLexer$"
TS_PARSE_ACTION_TYPE_REGEX = r"^TSParseActionType$|^struct TSParseActionType$|^typedef TSParseActionType$"
TS_PARSE_ACTION_REGEX = r"^TSParseAction$|^struct TSParseAction$|^typedef TSParseAction$"
TS_LEX_MODE_REGEX = r"^TSLexMode$|^struct TSLexMode$|^typedef TSLexMode$"
TS_LEXER_MODE_REGEX = r"^TSLexerMode$|^struct TSLexerMode$|^typedef TSLexerMode$"
TS_PARSE_ACTION_ENTRY_REGEX = r"^TSParseActionEntry$|^struct TSParseActionEntry$|^typedef TSParseActionEntry$"
TS_CHARACTER_RANGE_REGEX = r"^TSCharacterRange$|^struct TSCharacterRange$|^typedef TSCharacterRange$"
TS_LANGUAGE_REGEX = r"^TSLanguage$|^struct TSLanguage$|^typedef TSLanguage$"
COLUMN_DATA_REGEX = r"^ColumnData$|^struct ColumnData$|^typedef ColumnData$"
LEXER_REGEX = r"^Lexer$|^struct Lexer$|^typedef Lexer$"
LENGTH_REGEX = r"^Length$|^struct Length$|^typedef Length$"
TABLE_ENTRY_REGEX = r"^TableEntry$|^struct TableEntry$|^typedef TableEntry$"
LOOKAHEAD_ITERATOR_REGEX = r"^LookaheadIterator$|^struct LookaheadIterator$|^typedef LookaheadIterator$"
TS_RANGE_ARRAY_REGEX = r"^TSRangeArray$|^struct TSRangeArray$|^typedef TSRangeArray$"

# Mapping of type defs to regexes
TS_TYPE_TO_REGEX = {
    #TreeSitterType.TS_PARSER: TS_PARSER_REGEX,
    TreeSitterType.TS_QUERY: TS_QUERY_REGEX,
    TreeSitterType.TS_QUERY_CURSOR: TS_QUERY_CURSOR_REGEX,
    TreeSitterType.TS_LOOKAHEAD_ITERATOR: TS_LOOKAHEAD_ITERATOR_REGEX,
    TreeSitterType.TS_INPUT_ENCODING: TS_INPUT_ENCODING_REGEX,
    TreeSitterType.TS_SYMBOL_TYPE: TS_SYMBOL_TYPE_REGEX,
    TreeSitterType.TS_POINT: TS_POINT_REGEX,
    TreeSitterType.TS_RANGE: TS_RANGE_REGEX,
    TreeSitterType.TS_INPUT: TS_INPUT_REGEX,
    TreeSitterType.TS_PARSE_STATE: TS_PARSE_STATE_REGEX,
    TreeSitterType.TS_LOG_TYPE: TS_LOG_TYPE_REGEX,
    TreeSitterType.TS_LOGGER: TS_LOGGER_REGEX,
    TreeSitterType.TS_INPUT_EDITOR: TS_INPUT_EDITOR_REGEX,
    TreeSitterType.TS_QUERY_CAPTURE: TS_QUERY_CAPTURE_REGEX,
    TreeSitterType.TS_QUANTIFIER: TS_QUANTIFIER_REGEX,
    TreeSitterType.TS_QUERY_MATCH: TS_QUERY_MATCH_REGEX,
    TreeSitterType.TS_QUERY_PREDICATE_STEP_TYPE: TS_QUERY_PREDICATE_STEP_TYPE_REGEX,
    TreeSitterType.TS_QUERY_ERROR: TS_QUERY_ERROR_REGEX,
    TreeSitterType.TS_QUERY_CURSOR_STATE: TS_QUERY_CURSOR_STATE_REGEX,
    TreeSitterType.TS_QUERY_CURSOR_OPTIONS: TS_QUERY_CURSOR_OPTIONS_REGEX,
    TreeSitterType.TS_LANGUAGE_METADATA: TS_LANGUAGE_METADATA_REGEX,
    TreeSitterType.TS_WASM_ERROR_KIND: TS_WASM_ERROR_KIND_REGEX,
    TreeSitterType.TS_WASM_ERROR: TS_WASM_ERROR_REGEX,
    TreeSitterType.TREE_CURSOR: TREE_CURSOR_REGEX,
    TreeSitterType.TREE_CURSOR_STEP: TREE_CURSOR_STEP_REGEX,
    TreeSitterType.PARENT_CACHE_ENTRY: PARENT_CACHE_ENTRY_REGEX,
    TreeSitterType.TS_TREE: TS_TREE_REGEX,
    TreeSitterType.EXTERNAL_SCANNER_STATE: EXTERNAL_SCANNER_STATE_REGEX,
    TreeSitterType.SUBTREE_INLINE_DATA: SUBTREE_INLINE_DATA_REGEX,
    TreeSitterType.SUBTREE_HEAP_DATA: SUBTREE_HEAP_DATA_REGEX,
    TreeSitterType.SUBTREE: SUBTREE_REGEX,
    TreeSitterType.MUTABLE_SUBTREE: MUTABLE_SUBTREE_REGEX,
    TreeSitterType.SUBTREE_ARRAY: SUBTREE_ARRAY_REGEX,
    TreeSitterType.MUTABLE_SUBTREE_ARRAY: MUTABLE_SUBTREE_ARRAY_REGEX,
    TreeSitterType.SUBTREE_POOL: SUBTREE_POOL_REGEX,
    TreeSitterType.STACK_SLICE: STACK_SLICE_REGEX,
    TreeSitterType.STACK_SLICE_ARRAY: STACK_SLICE_ARRAY_REGEX,
    TreeSitterType.STACK_SUMMARY_ENTRY: STACK_SUMMARY_ENTRY_REGEX,
    TreeSitterType.STACK_SUMMARY: STACK_SUMMARY_REGEX,
    TreeSitterType.STACK_ENTRY: STACK_ENTRY_REGEX,
    TreeSitterType.REUSABLE_NODE: REUSABLE_NODE_REGEX,
    TreeSitterType.REDUCE_ACTION: REDUCE_ACTION_REGEX,
    TreeSitterType.REDUCE_ACTION_SET: REDUCE_ACTION_SET_REGEX,
    TreeSitterType.TS_FIELD_MAP_ENTRY: TS_FIELD_MAP_ENTRY_REGEX,
    TreeSitterType.TS_MAP_SLICE: TS_MAP_SLICE_REGEX,
    TreeSitterType.TS_SYMBOL_METADATA: TS_SYMBOL_METADATA_REGEX,
    TreeSitterType.TS_LEXER: TS_LEXER_REGEX,
    TreeSitterType.TS_PARSE_ACTION_TYPE: TS_PARSE_ACTION_TYPE_REGEX,
    TreeSitterType.TS_PARSE_ACTION: TS_PARSE_ACTION_REGEX,
    TreeSitterType.TS_LEX_MODE: TS_LEX_MODE_REGEX,
    TreeSitterType.TS_LEXER_MODE: TS_LEXER_MODE_REGEX,
    TreeSitterType.TS_PARSE_ACTION_ENTRY: TS_PARSE_ACTION_ENTRY_REGEX,
    TreeSitterType.TS_CHARACTER_RANGE: TS_CHARACTER_RANGE_REGEX,
    TreeSitterType.TS_LANGUAGE: TS_LANGUAGE_REGEX,
    TreeSitterType.COLUMN_DATA: COLUMN_DATA_REGEX,
    TreeSitterType.LEXER: LEXER_REGEX,
    TreeSitterType.LENGTH: LENGTH_REGEX,
    TreeSitterType.TABLE_ENTRY: TABLE_ENTRY_REGEX,
    TreeSitterType.LOOKAHEAD_ITERATOR: LOOKAHEAD_ITERATOR_REGEX,
    TreeSitterType.TS_RANGE_ARRAY: TS_RANGE_ARRAY_REGEX,
}

# Ok so here's a simple pretty printer that just returns a string
# Added with:
# regex_str = r"^TSPoint$|^struct TSPoint$|^typedef TSPoint$"
# debugger.HandleCommand(f"type summary add -F tree_sitter_types.TSPointSummaryProvider -x {regex_str}")
def TSPointSummaryProvider(valobj, _dict) -> str:
    return '"FOOBAR"'

# And here's a synthetic provider that fulfills the interface
class TSPointSyntheticProvider:
    def __init__(self, valobj, _dict):
        self.valobj = valobj
        self.row = self.valobj.GetChildMemberWithName('row')  # Cache original fields for efficiency.
        self.column = self.valobj.GetChildMemberWithName('column')
        self.update()

    def num_children(self):
        return 3  # Original row + modified column + new position.

    def get_child_index(self, name):
        if name == 'row':
            return 0
        if name == 'column':
            return 1
        if name == 'position':
            return 2
        return None

    def get_child_at_index(self, index):
        if index == 0:
            # Preserve original 'row' field.
            return self.row.GetNonSyntheticValue()  # Use GetNonSyntheticValue to avoid synthetic children.
            # return self.row
        if index == 1:
            # Override 'column' with a computed value (e.g., original * 2).
            original_col = self.column.GetValueAsUnsigned(0)
            modified_col = original_col * 2
            # Create a synthetic child with the new value (same type as original).
            target = self.valobj.GetProcess().GetTarget()
            endian = target.GetByteOrder()  # Use the target's endianness.
            addr_size = target.GetAddressByteSize()
            data = lldb.SBData.CreateDataFromUInt32Array(endian, addr_size, [modified_col])
            return self.valobj.CreateValueFromData('column', data, self.column.GetType())
        if index == 2:
            # Add a new synthetic child: a string like "row: X, col: Y".
            row_val = self.row.GetValueAsUnsigned(0)
            col_val = self.column.GetValueAsUnsigned(0)
            position_str = f"row: {row_val}, col: {col_val}"
            # Create it as a char* (string pointer type).
            target = self.valobj.GetProcess().GetTarget()
            endian = target.GetByteOrder()
            addr_size = target.GetAddressByteSize()
            char_type = target.GetBasicType(lldb.eBasicTypeChar)
            str_type = char_type.GetPointerType()
            data = lldb.SBData.CreateDataFromCString(endian, addr_size, position_str)
            # return self.valobj.CreateValueFromData('position', data, str_type)
            return self.valobj.CreateValueFromExpression("my_string", f'"{position_str}"')
        return None

    def update(self):
        # Optional: Recompute or refresh cached values if needed (e.g., if the underlying value changes).
        self.row = self.valobj.GetChildMemberWithName('row')  # Cache original fields for efficiency.
        self.column = self.valobj.GetChildMemberWithName('column')

# typedef Array(ReduceAction) ReduceActionSet;
#
# typedef struct {
#   uint32_t count;
#   TSSymbol symbol;
#   int dynamic_precedence;
#   unsigned short production_id;
# } ReduceAction;
#
# define Array(T)      \
# struct {             \
#   T *contents;       \
#   uint32_t size;     \
#   uint32_t capacity; \
# }


class ReduceActionSetSyntheticProvider:
    def __init__(self, valobj: SBValue, _dict):
        # logger = Logger.Logger()
        # logger >> "[StdVecSyntheticProvider] for " + str(valobj.GetName())
        self.valobj = valobj
        self.update()

    def num_children(self) -> int:
        return self.size

    def get_child_index(self, name: str) -> int:
        index = name.lstrip("[").rstrip("]")
        if index.isdigit():
            return int(index)
        else:
            return -1

    def get_child_at_index(self, index: int) -> SBValue:
        start = self.contents.GetValueAsUnsigned()
        address = start + index * self.element_type_size
        element = self.contents.CreateValueFromAddress(
            "[%s]" % index, address, self.element_type
        )
        return element

    def update(self):
        self.contents = self.valobj.GetChildMemberWithName("contents") #.GetNonSyntheticValue()
        self.size = self.valobj.GetChildMemberWithName("size").GetValueAsUnsigned()
        self.capacity = self.valobj.GetChildMemberWithName("capacity").GetValueAsUnsigned()

        self.element_type = self.contents.GetType().GetPointeeType()
        self.element_type_size = self.element_type.GetByteSize()

    def has_children(self) -> bool:
        return True

# Here's a simple summary function that uses the synthetic provider above
def TSPointSummaryFromSynthetic(valobj, _dict) -> str:
    # Example summary that uses the synthetic children.
    row_child = valobj.GetChildAtIndex(0)
    col_child = valobj.GetChildAtIndex(1)
    pos_child = valobj.GetChildAtIndex(2)
    if row_child and col_child and pos_child:
        return f'"Position (wut): {pos_child.GetSummary() or "bad"}, row: {row_child.GetValue()}, modified col: {col_child.GetValue()}"'
    return '"TSPoint"'

# TODO: Next steps:
# Need to figure out how to read a field that is a struct from within a synthetic provider.
# Both default rendering logic and the ones we define in other synthetic/summary providers
# 
# Maybe something like this?
# self.first_element = self.valueObject.GetChildMemberWithName('values').GetChildMemberWithName('elements').GetChildMemberWithName('data')
# https://github.com/jcredland/juce-toys/blob/master/juce_lldb_xcode.py
#
# This looks promising for our linked list instances: 
# https://yellowbrick.com/blog/yellowbrick-engineering/lldb-extension-for-structure-visualization/
#
# Look into using a commands file to source rather than the __lldb_init_module function as below

def __lldb_init_module(debugger, internal_dict):
    lldb.formatters.Logger._lldb_formatters_debug_level = 2
    lldb.formatters.Logger._lldb_formatters_debug_filename = "/home/lillis/projects/tree-sitter/lldb-tree-sitter.log"
    # TODO: We probably want a smart way to iterate over all the types and register rather than
    # enumerating them all
    print("Registering TreeSitterTypes pretty printer...")
    # for type_name, type_regex in TS_TYPE_TO_REGEX.items():
    #     # Register the type summary
    #     debugger.HandleCommand(f"type summary add -F tree_sitter_types.{type_name}SummaryProvider -x '{type_regex}'")
    #     # Register the synthetic provider
    #     debugger.HandleCommand(f"type synthetic add -l tree_sitter_types.{type_name}SyntheticProvider -x '{type_regex}'")
    # regex_str = r"^TSPoint$|^struct TSPoint$|^typedef TSPoint$"
    # debugger.HandleCommand(f"type synthetic add -l tree_sitter_types.TSPointSyntheticProvider -x '{regex_str}'")
    regex_str = TS_TYPE_TO_REGEX[TreeSitterType.REDUCE_ACTION_SET]
    debugger.HandleCommand(f"type synthetic add -l tree_sitter_types.ReduceActionSetSyntheticProvider -x '{regex_str}'")
    debugger.HandleCommand('type summary add -s "size=${svar%#}" ReduceActionSet')
    # debugger.HandleCommand(f"type summary add -F tree_sitter_types.TSPointSummaryFromSynthetic -x {regex_str}")
    print("TreeSitterTypes pretty printer registered")

print("Script loaded")
