use std::path::Path;

use super::*;

#[test]
fn json_native_grammar_roundtrip() {
    let dir = native_grammar_path("json_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("json_native"), &dir);
    let js_json =
        std::fs::read_to_string(test_grammars_dir().join("../grammars/json/src/grammar.json"))
            .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_eq!(native, js);
}

#[test]
fn c_native_grammar_roundtrip() {
    let dir = native_grammar_path("c_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("c_native"), &dir);
    let js_json =
        std::fs::read_to_string(test_grammars_dir().join("../grammars/c/src/grammar.json"))
            .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_eq!(native, js);
}

#[test]
fn javascript_native_grammar_roundtrip() {
    let dir = native_grammar_path("javascript_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("javascript_native"), &dir);
    let js_json = std::fs::read_to_string(
        test_grammars_dir().join("../grammars/javascript/src/grammar.json"),
    )
    .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

// ===== C++ grammar roundtrip =====

#[test]
fn cpp_native_grammar_roundtrip() {
    let dir = native_grammar_path("cpp_native");
    let (_, native_json) = dsl_to_json(&read_native_grammar("cpp_native"), &dir);
    let js_json =
        std::fs::read_to_string(test_grammars_dir().join("../grammars/cpp/src/grammar.json"))
            .unwrap();
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

/// Verify the direct path (no JSON roundtrip) works through `prepare_grammar`.
/// This is the same path the CLI takes via `GrammarSource::Grammar`.
#[test]
fn cpp_direct_prepare_grammar() {
    let dir = native_grammar_path("cpp_native");
    let mut grammar = parse_native_dsl(&read_native_grammar("cpp_native"), &dir).unwrap();
    crate::parse_grammar::normalize_grammar(&mut grammar);
    crate::prepare_grammar::prepare_grammar(&grammar).unwrap();
}

// ===== External grammar roundtrips =====

#[test]
fn query_native_grammar_roundtrip() {
    let grammars_dir = Path::new("/home/lillis/projects/grammars/tree-sitter-query");
    let tsg_path = grammars_dir.join("grammar.tsg");
    let js_json_path = grammars_dir.join("src/grammar.json");
    let tsg_src = std::fs::read_to_string(&tsg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsg_path.display()));
    let js_json = std::fs::read_to_string(&js_json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", js_json_path.display()));
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

#[test]
fn rust_native_grammar_roundtrip() {
    let grammars_dir = Path::new("/home/lillis/projects/grammars/tree-sitter-rust");
    let tsg_path = grammars_dir.join("grammar.tsg");
    let js_json_path = grammars_dir.join("src/grammar.json");
    let tsg_src = std::fs::read_to_string(&tsg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsg_path.display()));
    let js_json = std::fs::read_to_string(&js_json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", js_json_path.display()));
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

#[test]
fn python_native_grammar_roundtrip() {
    let grammars_dir = Path::new("/home/lillis/projects/grammars/tree-sitter-python");
    let tsg_path = grammars_dir.join("grammar.tsg");
    let js_json_path = grammars_dir.join("src/grammar.json");
    let tsg_src = std::fs::read_to_string(&tsg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsg_path.display()));
    let js_json = std::fs::read_to_string(&js_json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", js_json_path.display()));
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

#[test]
fn go_native_grammar_roundtrip() {
    let grammars_dir = Path::new("/home/lillis/projects/grammars/tree-sitter-go");
    let tsg_path = grammars_dir.join("grammar.tsg");
    let js_json_path = grammars_dir.join("src/grammar.json");
    let tsg_src = std::fs::read_to_string(&tsg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsg_path.display()));
    let js_json = std::fs::read_to_string(&js_json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", js_json_path.display()));
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

#[test]
fn gomod_native_grammar_roundtrip() {
    let grammars_dir = Path::new("/home/lillis/projects/grammars/tree-sitter-go-mod");
    let tsg_path = grammars_dir.join("grammar.tsg");
    let js_json_path = grammars_dir.join("src/grammar.json");
    let tsg_src = std::fs::read_to_string(&tsg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsg_path.display()));
    let js_json = std::fs::read_to_string(&js_json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", js_json_path.display()));
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

#[test]
fn vim_native_grammar_roundtrip() {
    let grammars_dir =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../grammars/tree-sitter-vim");
    let tsg_path = grammars_dir.join("grammar.tsg");
    let js_json_path = grammars_dir.join("src/grammar.json");
    let tsg_src = std::fs::read_to_string(&tsg_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", tsg_path.display()));
    let js_json = std::fs::read_to_string(&js_json_path)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", js_json_path.display()));
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);
    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

/// Iterate over all `.tsg` files in `fuzz_regressions/` and verify none panic.
/// To add a new regression, just drop a `.tsg` file in that directory.
#[test]
fn fuzz_regressions() {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/nativedsl/fuzz_regressions");
    let mut entries: Vec<_> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", dir.display()))
        .filter_map(|e| {
            let path = e.ok()?.path();
            (path.extension()?.to_str()? == "tsg").then_some(path)
        })
        .collect();
    entries.sort();
    assert!(!entries.is_empty(), "no .tsg files in {}", dir.display());
    let dummy_path = dir.join("grammar.tsg");
    for path in &entries {
        let input = std::fs::read_to_string(path).unwrap();
        let result = std::panic::catch_unwind(|| {
            let _ = parse_native_dsl(&input, &dummy_path);
        });
        assert!(
            result.is_ok(),
            "panic on regression file: {}",
            path.display()
        );
    }
}
