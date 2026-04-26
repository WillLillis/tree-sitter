#![no_main]

use libfuzzer_sys::fuzz_target;
use std::path::Path;
use std::sync::Once;

static SEED: Once = Once::new();

fuzz_target!(|data: &str| {
    SEED.call_once(seed_corpus);
    let _ = tree_sitter_generate::nativedsl::parse_native_dsl(data, Path::new("/tmp/grammar.tsg"));
});

/// Copy seed files into the corpus directory on first run.
fn seed_corpus() {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let corpus_dir = project_root.join("crates/generate/fuzz/corpus/native_dsl");
    if corpus_dir.exists() && std::fs::read_dir(&corpus_dir).is_ok_and(|mut d| d.next().is_some()) {
        return;
    }
    let _ = std::fs::create_dir_all(&corpus_dir);

    let sources = [
        project_root.join("crates/generate/fuzz/seeds"),
        project_root.join("crates/generate/src/nativedsl/fuzz_regressions"),
    ];
    for dir in &sources {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let _ = std::fs::copy(&path, corpus_dir.join(path.file_name().unwrap()));
                }
            }
        }
    }
}
