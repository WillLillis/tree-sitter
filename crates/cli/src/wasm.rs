use std::{
    fs,
    path::{Path, PathBuf},
};

use tree_sitter::wasm_stdlib_symbols;
use tree_sitter_generate::{load_grammar_file, parse_grammar::GrammarJSON};
use tree_sitter_loader::Loader;
use wasmparser::Parser;

use crate::error::{IoError, WasmCmdError, WasmCmdResult};

pub fn load_language_wasm_file(language_dir: &Path) -> WasmCmdResult<(String, Vec<u8>)> {
    let grammar_name = get_grammar_name(language_dir).expect("Failed to get Wasm filename");
    let wasm_filename = format!("tree-sitter-{grammar_name}.wasm");
    let path = language_dir.join(&wasm_filename);
    let contents = fs::read(&path).map_err(|source| WasmCmdError::ReadWasmBinary {
        path: path.clone(),
        source,
    })?;
    Ok((grammar_name, contents))
}

pub fn get_grammar_name(language_dir: &Path) -> WasmCmdResult<String> {
    let src_dir = language_dir.join("src");
    let grammar_json_path = src_dir.join("grammar.json");
    let grammar_json =
        fs::read_to_string(&grammar_json_path).map_err(IoError::at(&grammar_json_path))?;
    let grammar: GrammarJSON =
        serde_json::from_str(&grammar_json).map_err(|source| WasmCmdError::ParseGrammarFile {
            path: grammar_json_path,
            source,
        })?;
    Ok(grammar.name)
}

pub fn compile_language_to_wasm(
    loader: &Loader,
    language_dir: &Path,
    output_dir: &Path,
    output_file: Option<PathBuf>,
) -> WasmCmdResult<()> {
    let grammar_name = match get_grammar_name(language_dir) {
        Ok(name) => name,
        Err(_) => load_grammar_file(&language_dir.join("grammar.js"), None)?,
    };
    let output_filename =
        output_file.unwrap_or_else(|| output_dir.join(format!("tree-sitter-{grammar_name}.wasm")));
    let src_path = language_dir.join("src");
    let scanner_path = loader.get_scanner_path(&src_path);
    loader.compile_parser_to_wasm(
        &grammar_name,
        &src_path,
        scanner_path
            .as_ref()
            .and_then(|p| Some(Path::new(p.file_name()?))),
        &output_filename,
    )?;

    let stdlib_symbols = wasm_stdlib_symbols().collect::<Vec<_>>();
    let dylink_symbols = [
        "__indirect_function_table",
        "__memory_base",
        "__stack_pointer",
        "__table_base",
        "__table_base",
        "memory",
    ];
    let builtin_symbols = [
        "__assert_fail",
        "__cxa_atexit",
        "abort",
        "emscripten_notify_memory_growth",
        "tree_sitter_debug_message",
        "proc_exit",
    ];

    let mut missing_symbols = Vec::new();
    let wasm_bytes = fs::read(&output_filename).map_err(IoError::at(&output_filename))?;
    let parser = Parser::new(0);
    for payload in parser.parse_all(&wasm_bytes) {
        if let wasmparser::Payload::ImportSection(reader) = payload? {
            for imports in reader {
                for import in imports? {
                    let (_, import) = import?;
                    let name = import.name;
                    if !builtin_symbols.contains(&name)
                        && !stdlib_symbols.contains(&name)
                        && !dylink_symbols.contains(&name)
                    {
                        missing_symbols.push(name);
                    }
                }
            }
        }
    }

    if !missing_symbols.is_empty() {
        return Err(WasmCmdError::DisallowedScannerSymbols {
            missing: missing_symbols.iter().map(|s| (*s).to_string()).collect(),
            available: stdlib_symbols.iter().map(|s| (*s).to_string()).collect(),
        });
    }

    Ok(())
}
