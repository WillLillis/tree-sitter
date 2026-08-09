//! Golden tests over the real query files in `test/fixtures/grammars/*/queries`.
//!
//! These pin *observed* behaviour, which is a weaker and different claim than the hand-written
//! assertions in `query_test.rs`: those say what the engine *should* do, these record what it
//! *does*, bugs included. Their job is to answer "did this change alter any observable output
//! anywhere in the corpus" for changes that claim to be semantics-preserving.
//!
//! Each (query, example, mode) triple contributes one section holding a hash of the **full**
//! match stream plus a bounded verbatim sample. The hash catches any change anywhere; the
//! sample makes the common regression readable directly in the diff.
//!
//! Regenerate after an intentional change:
//!
//! ```sh
//! TREE_SITTER_BLESS=1 cargo test -p tree-sitter-cli query_golden
//! ```
//!
//! If a hash moves but the sample does not, widen the sample to localize it:
//!
//! ```sh
//! TREE_SITTER_GOLDEN_SAMPLE=100000 TREE_SITTER_BLESS=1 cargo test -p tree-sitter-cli query_golden
//! ```
//!
//! Goldens depend on the grammar versions pinned in `test/fixtures/fixtures.json`; bumping a
//! grammar there is expected to require re-blessing.

use std::{
    env,
    fmt::Write as _,
    fs,
    path::{Path, PathBuf},
};

use streaming_iterator::StreamingIterator;
use tree_sitter::{Language, Parser, Query, QueryCursor};

use super::helpers::fixtures::{fixtures_dir, get_language};

/// Matches shown verbatim per section. Enough to localize a typical regression without
/// making the goldens unreviewable.
const DEFAULT_SAMPLE: usize = 40;

/// Grammars are skipped unless they have a parser directly at `src/`, a `queries/` directory,
/// and an `examples/` directory. This excludes `typescript`, whose parsers are nested one level
/// down (`typescript/`, `tsx/`).
fn corpus() -> Vec<(String, Vec<PathBuf>, Vec<PathBuf>)> {
    let grammars_dir = fixtures_dir().join("grammars");
    let mut out = Vec::new();

    let mut entries = fs::read_dir(&grammars_dir)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", grammars_dir.display()))
        .filter_map(Result::ok)
        .map(|e| e.path())
        .collect::<Vec<_>>();
    entries.sort();

    for dir in entries {
        if !dir.join("src/parser.c").exists() {
            continue;
        }
        let (queries, examples) = (dir.join("queries"), dir.join("examples"));
        if !queries.is_dir() || !examples.is_dir() {
            continue;
        }
        let name = dir.file_name().unwrap().to_string_lossy().into_owned();
        let mut query_paths = sorted_files(&queries);
        query_paths.retain(|p| p.extension().is_some_and(|e| e == "scm"));
        let example_paths = sorted_files(&examples);
        if query_paths.is_empty() || example_paths.is_empty() {
            continue;
        }
        out.push((name, query_paths, example_paths));
    }
    out
}

fn sorted_files(dir: &Path) -> Vec<PathBuf> {
    let mut v = fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(Result::ok)
                .map(|e| e.path())
                .filter(|p| p.is_file())
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    v.sort();
    v
}

/// FNV-1a. Hand-rolled deliberately: `DefaultHasher` is explicitly not stable across Rust
/// releases, and a checked-in golden needs a hash that will not change under us.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// One line per match: pattern index, then each capture as `name@start..end`.
///
/// Byte ranges only — never `node.id`, which is a pointer and so varies run to run under ASLR.
/// Node text is deliberately excluded: it duplicates the source, inflates the goldens, and can
/// contain newlines that would break the line-oriented format.
fn render_stream(language: &Language, query: &Query, source: &[u8], captures_mode: bool) -> Vec<String> {
    let mut parser = Parser::new();
    parser.set_language(language).unwrap();
    let tree = parser.parse(source, None).unwrap();
    let names = query.capture_names();
    let mut cursor = QueryCursor::new();
    let mut lines = Vec::new();

    if captures_mode {
        let mut it = cursor.captures(query, tree.root_node(), source);
        while let Some((m, idx)) = it.next() {
            let c = m.captures[*idx];
            let r = c.node.byte_range();
            lines.push(format!(
                "p{} {}@{}..{}",
                m.pattern_index, names[c.index as usize], r.start, r.end
            ));
        }
    } else {
        let mut it = cursor.matches(query, tree.root_node(), source);
        while let Some(m) = it.next() {
            let mut line = format!("p{}", m.pattern_index);
            for c in m.captures {
                let r = c.node.byte_range();
                write!(line, " {}@{}..{}", names[c.index as usize], r.start, r.end).unwrap();
            }
            lines.push(line);
        }
    }
    lines
}

fn build_golden(grammar: &str, query_paths: &[PathBuf], example_paths: &[PathBuf], sample: usize) -> String {
    let language = get_language(grammar);
    let mut out = String::new();
    writeln!(out, "# Query match goldens for `{grammar}`.").unwrap();
    writeln!(
        out,
        "# Generated by `TREE_SITTER_BLESS=1 cargo test -p tree-sitter-cli query_golden`."
    )
    .unwrap();
    writeln!(
        out,
        "# Per section: FNV-1a of the full stream, total count, then the first {sample} entries."
    )
    .unwrap();
    writeln!(out, "# Ranges are byte offsets. Order is the API's own, and is part of what is pinned.").unwrap();

    for query_path in query_paths {
        let query_name = query_path.file_name().unwrap().to_string_lossy();
        let query_src = fs::read_to_string(query_path)
            .unwrap_or_else(|e| panic!("cannot read {}: {e}", query_path.display()));
        let query = match Query::new(&language, &query_src) {
            Ok(q) => q,
            Err(e) => {
                writeln!(out, "\n== {query_name} DOES NOT COMPILE: {e:?}").unwrap();
                continue;
            }
        };

        for example_path in example_paths {
            let example_name = example_path.file_name().unwrap().to_string_lossy();
            let source = fs::read(example_path)
                .unwrap_or_else(|e| panic!("cannot read {}: {e}", example_path.display()));

            for (mode, captures_mode) in [("matches", false), ("captures", true)] {
                let lines = render_stream(&language, &query, &source, captures_mode);
                let joined = lines.join("\n");
                writeln!(
                    out,
                    "\n== {query_name} x {example_name} [{mode}] count={} hash={:016x}",
                    lines.len(),
                    fnv1a(joined.as_bytes())
                )
                .unwrap();
                for line in lines.iter().take(sample) {
                    writeln!(out, "{line}").unwrap();
                }
                if lines.len() > sample {
                    writeln!(out, "... {} more elided", lines.len() - sample).unwrap();
                }
            }
        }
    }
    out
}

fn report_mismatch(grammar: &str, path: &Path, expected: &str, actual: &str) -> String {
    let exp = expected.lines().collect::<Vec<_>>();
    let act = actual.lines().collect::<Vec<_>>();
    let mut msg = format!(
        "query goldens for `{grammar}` changed ({})\n\
         re-run with TREE_SITTER_BLESS=1 to accept, after confirming the change is intended\n",
        path.display()
    );
    let mut shown = 0;
    for i in 0..exp.len().max(act.len()) {
        let (e, a) = (exp.get(i), act.get(i));
        if e != a {
            if shown == 20 {
                writeln!(msg, "  ... further differences elided").unwrap();
                break;
            }
            writeln!(msg, "  line {}:\n    -{}\n    +{}", i + 1, e.unwrap_or(&""), a.unwrap_or(&"")).unwrap();
            shown += 1;
        }
    }
    msg
}

#[test]
fn test_query_goldens() {
    let sample = env::var("TREE_SITTER_GOLDEN_SAMPLE")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_SAMPLE);
    let bless = env::var("TREE_SITTER_BLESS").is_ok();

    let golden_dir = fixtures_dir().join("query-goldens");
    if bless {
        fs::create_dir_all(&golden_dir).unwrap();
    }

    let corpus = corpus();
    assert!(!corpus.is_empty(), "no grammars with both queries/ and examples/ were found");

    let mut failures = Vec::new();
    for (grammar, query_paths, example_paths) in corpus {
        let actual = build_golden(&grammar, &query_paths, &example_paths, sample);
        let path = golden_dir.join(format!("{grammar}.txt"));

        if bless {
            fs::write(&path, &actual).unwrap();
            continue;
        }

        match fs::read_to_string(&path) {
            Ok(expected) if expected == actual => {}
            Ok(expected) => failures.push(report_mismatch(&grammar, &path, &expected, &actual)),
            Err(_) => failures.push(format!(
                "missing golden for `{grammar}` ({})\nre-run with TREE_SITTER_BLESS=1 to create it",
                path.display()
            )),
        }
    }

    assert!(failures.is_empty(), "\n\n{}", failures.join("\n"));
}
