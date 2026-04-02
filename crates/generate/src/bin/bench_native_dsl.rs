use std::hint::black_box;
use std::time::Instant;

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| {
        eprintln!("usage: bench-native-dsl <grammar.grammar>");
        std::process::exit(1);
    });
    let source = std::fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("failed to read {path}: {e}");
        std::process::exit(1);
    });

    let n: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5000);

    // Warmup
    for _ in 0..10 {
        black_box(tree_sitter_generate::nativedsl::parse_native_dsl(&source).unwrap());
    }

    let start = Instant::now();
    for _ in 0..n {
        black_box(tree_sitter_generate::nativedsl::parse_native_dsl(&source).unwrap());
    }
    let elapsed = start.elapsed();
    eprintln!("{n} iterations in {elapsed:?} ({:?}/iter)", elapsed / n);
}
