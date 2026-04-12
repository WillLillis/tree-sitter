use std::path::Path;

pub fn seed_corpus() {
    let project_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    let corpus_dir = project_root.join("crates/generate/fuzz/corpus/native_dsl");
    if corpus_dir.exists() && std::fs::read_dir(&corpus_dir).is_ok_and(|mut d| d.next().is_some())
    {
        return;
    }
    let _ = std::fs::create_dir_all(&corpus_dir);

    let sources = [
        project_root.join("crates/xtask/fuzz_corpus/inputs"),
        project_root.join("crates/generate/src/nativedsl/fuzz_regressions"),
    ];
    for dir in &sources {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().is_some_and(|e| e == "tsg") {
                    let _ = std::fs::copy(&path, corpus_dir.join(path.file_name().unwrap()));
                }
            }
        }
    }

    let test_grammars = project_root.join("test/fixtures/test_grammars");
    if let Ok(entries) = std::fs::read_dir(test_grammars) {
        for entry in entries.flatten() {
            let grammar = entry.path().join("grammar.tsg");
            if grammar.exists() {
                let name = entry.file_name();
                let _ = std::fs::copy(
                    &grammar,
                    corpus_dir.join(format!("{}.tsg", name.to_string_lossy())),
                );
            }
        }
    }
}
