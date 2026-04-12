#![no_main]

mod common;

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use std::path::Path;
use std::sync::Once;

static SEED: Once = Once::new();

#[derive(Arbitrary, Debug)]
struct InheritInput<'a> {
    base: &'a str,
    derived_body: &'a str,
}

fuzz_target!(|input: InheritInput| {
    SEED.call_once(common::seed_corpus);

    let dir = Path::new("/tmp/fuzz_dsl_inherit");
    let _ = std::fs::create_dir_all(dir);
    let base_path = dir.join("base.tsg");
    let _ = std::fs::write(&base_path, input.base);

    // Wrap the fuzzed body with the inherit boilerplate so we always
    // reach the inheritance code path.
    let derived = format!(
        "let base = inherit(\"{}\")\n\
         grammar {{ language: \"derived\", inherits: base }}\n\
         {}",
        base_path.display(),
        input.derived_body,
    );
    let grammar_path = dir.join("grammar.tsg");
    let _ = tree_sitter_generate::nativedsl::parse_native_dsl(&derived, &grammar_path);
});
