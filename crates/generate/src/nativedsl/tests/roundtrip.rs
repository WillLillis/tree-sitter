use super::*;

/// Roundtrip test: generate grammar.json from both grammar.js (via Node.js)
/// and grammar.tsg (via native DSL pipeline), then compare the results.
fn roundtrip_external(name: &str) {
    let tsg_path = native_grammar_path(name);
    let repo_dir = tsg_path.parent().unwrap();
    let js_path = repo_dir.join("grammar.js");

    // Generate JSON from grammar.js via Node.js
    let js_json = crate::load_grammar_file(&js_path, None)
        .unwrap_or_else(|e| panic!("failed to load {}: {e}", js_path.display()))
        .into_json();

    // Generate JSON from grammar.tsg via native DSL pipeline
    let tsg_src = read_native_grammar(name);
    let (_, native_json) = dsl_to_json(&tsg_src, &tsg_path);

    let native = crate::parse_grammar::parse_grammar(&native_json).unwrap();
    let js = crate::parse_grammar::parse_grammar(&js_json).unwrap();
    assert_grammar_eq(&native, &js);
}

#[test]
fn c_native_grammar_roundtrip() {
    roundtrip_external("c_native");
}

#[test]
fn javascript_native_grammar_roundtrip() {
    roundtrip_external("javascript_native");
}

#[test]
fn cpp_native_grammar_roundtrip() {
    roundtrip_external("cpp_native");
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
fn json_native_grammar_roundtrip() {
    roundtrip_external("json_native");
}

#[test]
fn query_native_grammar_roundtrip() {
    roundtrip_external("query_native");
}

#[test]
fn rust_native_grammar_roundtrip() {
    roundtrip_external("rust_native");
}

#[test]
fn python_native_grammar_roundtrip() {
    roundtrip_external("python_native");
}

#[test]
fn go_native_grammar_roundtrip() {
    roundtrip_external("go_native");
}

#[test]
fn gomod_native_grammar_roundtrip() {
    roundtrip_external("gomod_native");
}

#[test]
fn vim_native_grammar_roundtrip() {
    roundtrip_external("vim_native");
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
