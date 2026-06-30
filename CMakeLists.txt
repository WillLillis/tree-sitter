cmake_minimum_required(VERSION 3.13)

project(tree-sitter
        VERSION "0.27.0"
        DESCRIPTION "An incremental parsing system for programming tools"
        HOMEPAGE_URL "https://tree-sitter.github.io/tree-sitter/"
        LANGUAGES C)

option(BUILD_SHARED_LIBS "Build using shared libraries" ON)
option(TREE_SITTER_FEATURE_WASM "Enable the Wasm feature" OFF)
option(AMALGAMATED "Build using an amalgamated source" OFF)

if(AMALGAMATED)
  set(TS_SOURCE_FILES "${PROJECT_SOURCE_DIR}/lib/src/lib.c")
else()
  file(GLOB TS_SOURCE_FILES lib/src/*.c)
  list(REMOVE_ITEM TS_SOURCE_FILES "${PROJECT_SOURCE_DIR}/lib/src/lib.c")
endif()

add_library(tree-sitter ${TS_SOURCE_FILES})

target_include_directories(tree-sitter PRIVATE lib/src lib/src/wasm PUBLIC lib/include)

if(MSVC)
  target_compile_options(tree-sitter PRIVATE
                         /wd4018 # disable 'signed/unsigned mismatch'
                         /wd4232 # disable 'nonstandard extension used'
                         /wd4244 # disable 'possible loss of data'
                         /wd4267 # disable 'possible loss of data (size_t)'
                         /wd4701 # disable 'potentially uninitialized local variable'
                         /we4022 # treat 'incompatible types' as an error
                         /W4)
else()
  target_compile_options(tree-sitter PRIVATE
                         -Wall -Wextra -Wshadow -Wpedantic
                         -Werror=incompatible-pointer-types
                         -Werror=strict-aliasing -Wstrict-aliasing=2)
endif()

if(TREE_SITTER_FEATURE_WASM)
  if(NOT DEFINED CACHE{WASMTIME_INCLUDE_DIR})
    message(CHECK_START "Looking for wasmtime headers")
    find_path(WASMTIME_INCLUDE_DIR wasmtime.h
              PATHS ENV DEP_WASMTIME_C_API_INCLUDE)
    if(NOT WASMTIME_INCLUDE_DIR)
      unset(WASMTIME_INCLUDE_DIR CACHE)
      message(FATAL_ERROR "Could not find wasmtime headers.\nDid you forget to set CMAKE_INCLUDE_PATH?")
    endif()
    message(CHECK_PASS "found")
  endif()

  if(NOT DEFINED CACHE{WASMTIME_LIBRARY})
    message(CHECK_START "Looking for wasmtime library")
    find_library(WASMTIME_LIBRARY wasmtime)
    if(NOT WASMTIME_LIBRARY)
      unset(WASMTIME_LIBRARY CACHE)
      message(FATAL_ERROR "Could not find wasmtime library.\nDid you forget to set CMAKE_LIBRARY_PATH?")
    endif()
    message(CHECK_PASS "found")
  endif()

  target_compile_definitions(tree-sitter PUBLIC TREE_SITTER_FEATURE_WASM)
  target_include_directories(tree-sitter SYSTEM PRIVATE "${WASMTIME_INCLUDE_DIR}")
  target_link_libraries(tree-sitter PUBLIC "${WASMTIME_LIBRARY}")
  set_property(TARGET tree-sitter PROPERTY C_STANDARD_REQUIRED ON)

  if(NOT BUILD_SHARED_LIBS)
    if(WIN32)
      target_compile_definitions(tree-sitter PRIVATE WASM_API_EXTERN= WASI_API_EXTERN=)
      target_link_libraries(tree-sitter INTERFACE ws2_32 advapi32 userenv ntdll shell32 ole32 bcrypt)
    elseif(NOT APPLE)
      target_link_libraries(tree-sitter INTERFACE pthread dl m)
    endif()
  endif()
endif()

set_target_properties(tree-sitter
                      PROPERTIES
                      C_STANDARD 11
                      C_VISIBILITY_PRESET hidden
                      POSITION_INDEPENDENT_CODE ON
                      SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
                      DEFINE_SYMBOL "")

target_compile_definitions(tree-sitter PRIVATE _POSIX_C_SOURCE=200112L _DEFAULT_SOURCE _BSD_SOURCE _DARWIN_C_SOURCE)

# On Windows a DLL exports nothing unless its symbols are listed explicitly: the
# `#pragma GCC visibility` directives in the public headers are no-ops under
# MSVC (yielding an empty export table, #5190), while under MinGW the GNU linker
# instead auto-exports *every* symbol, internals included. An explicit module-
# definition file fixes both, giving the curated, hidden-by-default surface used
# on other platforms. tree_sitter.def is the canonical export list (meant to be
# shared with the Make and Zig builds too). The Wasm store entry points only
# exist when the Wasm feature is enabled, so append them to a generated copy
# when needed -- naming an undefined symbol in a .def is a hard link error under
# both MSVC and GNU ld.
if(WIN32 AND BUILD_SHARED_LIBS)
  set(TS_DEF_FILE "${PROJECT_SOURCE_DIR}/lib/src/tree_sitter.def")
  if(TREE_SITTER_FEATURE_WASM)
    file(READ "${TS_DEF_FILE}" TS_DEF_BASE)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/tree_sitter.def"
"${TS_DEF_BASE}    ts_wasm_store_language_count
    ts_wasm_store_load_language
    ts_wasm_store_new
")
    set(TS_DEF_FILE "${CMAKE_CURRENT_BINARY_DIR}/tree_sitter.def")
  endif()
  target_sources(tree-sitter PRIVATE "${TS_DEF_FILE}")
endif()

include(GNUInstallDirs)

configure_file(lib/tree-sitter.pc.in "${CMAKE_CURRENT_BINARY_DIR}/tree-sitter.pc" @ONLY)

install(FILES lib/include/tree_sitter/api.h
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/tree_sitter")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/tree-sitter.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
install(TARGETS tree-sitter
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
