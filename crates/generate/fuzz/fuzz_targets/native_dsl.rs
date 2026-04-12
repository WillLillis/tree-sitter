#![no_main]

mod common;

use libfuzzer_sys::fuzz_target;
use std::path::Path;
use std::sync::Once;

static SEED: Once = Once::new();

fuzz_target!(|data: &str| {
    SEED.call_once(common::seed_corpus);
    let _ = tree_sitter_generate::nativedsl::parse_native_dsl(data, Path::new("/tmp/grammar.tsg"));
});
