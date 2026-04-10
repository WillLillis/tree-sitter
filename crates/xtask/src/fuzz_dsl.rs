use std::{
    fmt::Write as _,
    io::Write,
    panic::{self, catch_unwind},
    path::{Path, PathBuf},
    sync::atomic::{AtomicBool, Ordering},
    time::Instant,
};

use anyhow::{Result, anyhow};
use rand::prelude::*;
use rand::rngs::SmallRng;
use tree_sitter_generate::nativedsl::parse_native_dsl;

const DISPLAY_INTERVAL: u64 = 10_000;
const CORPUS_DIR: &str = "crates/xtask/fuzz_corpus";

static STOP: AtomicBool = AtomicBool::new(false);

fn write_num(w: &mut impl std::io::Write, n: u64) {
    let s = n.to_string();
    let len = s.len();
    for (i, b) in s.bytes().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            w.write_all(b"_").unwrap();
        }
        w.write_all(&[b]).unwrap();
    }
}

fn load_dir(dir: &Path) -> Result<Vec<String>> {
    let mut files = Vec::new();
    if !dir.exists() {
        return Ok(files);
    }
    for entry in std::fs::read_dir(dir)? {
        let path = entry?.path();
        if path.extension().is_some_and(|e| e == "tsg") {
            files.push(std::fs::read_to_string(&path)?);
        }
    }
    files.sort();
    Ok(files)
}

#[rustfmt::skip]
const TOKENS: &[&str] = &[
    "grammar", "rule", "let", "fn", "for", "in", "seq", "choice", "repeat", "repeat1",
    "optional", "blank", "field", "alias", "token", "prec", "prec_left", "prec_right",
    "prec_dynamic", "reserved", "token_immediate", "concat", "regexp", "inherit", "override",
    "append", "{", "}", "(", ")", "[", "]", ",", ":", "::", ".", "->", "-", "=", "<", ">",
    "\"test\"", "\"hello\"", "\"a\"", "\"b\"", "\"\"", "\"\\n\"", "\"\\t\"", "\"\\\\\"",
    r#"r"raw""#, r##"r#"raw"#"##,
    "42", "0", "-1", "99999",
    "foo", "bar", "baz", "program", "name", "x", "_x", "_hidden", "expr", "ident",
    "rule", "str", "int", "list",
    " ", " ", " ", "\n", "\n", "//comment\n",
];

#[rustfmt::skip]
const KEYWORDS: &[&str] = &[
    "grammar", "rule", "let", "fn", "for", "in", "seq", "choice", "repeat", "repeat1",
    "optional", "blank", "field", "alias", "token", "prec", "prec_left", "prec_right",
    "prec_dynamic", "reserved", "token_immediate", "concat", "regexp", "inherit", "override",
    "append",
];

// Combinators that wrap a single expression
const UNARY_COMBINATORS: &[&str] = &["repeat", "repeat1", "optional", "token", "token_immediate"];

// -- Generators --

fn random_ascii(rng: &mut SmallRng, max_len: usize, buf: &mut String) {
    let len = rng.random_range(0..max_len + 1);
    for _ in 0..len {
        let b = rng.random::<u8>() % 128;
        let c = (if b < 32 && b != b'\n' && b != b'\t' {
            b' '
        } else {
            b
        }) as char;
        buf.push(c);
    }
}

fn truncate_grammar(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = &corpus[rng.random_range(0..corpus.len())];
    let cut = rng.random_range(0..base.len() + 1);
    buf.push_str(&base[..cut]);
}

fn mutate_bytes(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = &corpus[rng.random_range(0..corpus.len())];
    let mut bytes: Vec<u8> = base.bytes().collect();
    let mutations = 1 + rng.random_range(0..5);
    for _ in 0..mutations {
        if bytes.is_empty() {
            break;
        }
        match rng.random_range(0..4) {
            // flip a byte
            0 => {
                let idx = rng.random_range(0..bytes.len());
                bytes[idx] = rng.random::<u8>() % 128;
            }
            // insert a byte
            1 => {
                let idx = rng.random_range(0..bytes.len() + 1);
                bytes.insert(idx, rng.random::<u8>() % 128);
            }
            // delete a byte
            2 => {
                let idx = rng.random_range(0..bytes.len());
                bytes.remove(idx);
            }
            // swap two bytes
            _ => {
                let a = rng.random_range(0..bytes.len());
                let b = rng.random_range(0..bytes.len());
                bytes.swap(a, b);
            }
        }
    }
    buf.push_str(&String::from_utf8_lossy(&bytes));
}

fn token_soup(rng: &mut SmallRng, buf: &mut String) {
    let count = 1 + rng.random_range(0..50);
    for i in 0..count {
        if i > 0 && rng.random_bool(0.5) {
            buf.push(' ');
        }
        buf.push_str(TOKENS[rng.random_range(0..TOKENS.len())]);
    }
}

/// Generate deeply nested expressions: repeat(repeat(repeat(...)))
fn deep_nesting(rng: &mut SmallRng, buf: &mut String) {
    let depth = 5 + rng.random_range(0..50);
    buf.push_str(r#"grammar { language: "test" } rule program { "#);
    for _ in 0..depth {
        let combinator = UNARY_COMBINATORS[rng.random_range(0..UNARY_COMBINATORS.len())];
        write!(buf, "{combinator}(").unwrap();
    }
    buf.push_str("\"x\"");
    for _ in 0..depth {
        buf.push(')');
    }
    buf.push_str(" }");
}

/// Splice two corpus grammars at random byte offsets
fn splice(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let a = &corpus[rng.random_range(0..corpus.len())];
    let b = &corpus[rng.random_range(0..corpus.len())];
    let cut_a = rng.random_range(0..a.len() + 1);
    let cut_b = rng.random_range(0..b.len() + 1);
    write!(buf, "{}{}", &a[..cut_a], &b[cut_b..]).unwrap();
}

/// Replace random keywords in a valid grammar with other keywords
fn keyword_swap(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = &corpus[rng.random_range(0..corpus.len())];
    buf.push_str(base);
    let swaps = 1 + rng.random_range(0..3);
    for _ in 0..swaps {
        let from = KEYWORDS[rng.random_range(0..KEYWORDS.len())];
        let to = KEYWORDS[rng.random_range(0..KEYWORDS.len())];
        if let Some(pos) = buf.find(from) {
            buf.replace_range(pos..pos + from.len(), to);
        }
    }
}

/// Generate a grammar with many rules
fn many_rules(rng: &mut SmallRng, buf: &mut String) {
    let count = 10 + rng.random_range(0..100);
    buf.push_str(r#"grammar { language: "test" } "#);
    buf.push_str(r"rule program { choice(");
    for i in 0..count {
        if i > 0 {
            buf.push_str(", ");
        }
        write!(buf, "r{i}").unwrap();
    }
    buf.push_str(") } ");
    for i in 0..count {
        write!(buf, "rule r{i} {{  ").unwrap();
        match rng.random_range(0..5) {
            0 => write!(buf, r#""lit{i}""#).unwrap(),
            1 => write!(buf, r#"seq("a{i}", "b{i}")"#).unwrap(),
            2 => write!(buf, r#"repeat("x{i}")"#).unwrap(),
            3 => write!(buf, r#"choice("y{i}", "z{i}")"#).unwrap(),
            _ => write!(buf, r#"prec({}, "p{i}")"#, rng.random_range(0..10i32) - 5).unwrap(),
        }
        buf.push_str(" } ");
    }
}

/// Generate strings with edge-case content
fn string_edge_cases(rng: &mut SmallRng, buf: &mut String) {
    // Sometimes just return the string literals below alone (tests lexer),
    // otherwise place them in a simple tsg file
    let place_in_grammar = rng.random_range(0..8) < 2;
    if place_in_grammar {
        write!(buf, r#"grammar {{ language: "test" }} rule program {{ "#).unwrap();
    }
    match rng.random_range(0..8) {
        // empty string
        0 => buf.push_str(r#""""#),
        // all escape sequences
        1 => buf.push_str(r#""\n\t\r\\\"\0""#),
        // very long string
        2 => {
            let len = 100 + rng.random_range(0..500);
            write!(buf, r#""{}""#, "a".repeat(len)).unwrap();
        }
        // very long identifier
        3 => {
            let len = 50 + rng.random_range(0..200);
            write!(
                buf,
                r#"grammar {{ language: "test" }} let {} = "x" rule program {{ "x" }}"#,
                "a".repeat(len)
            )
            .unwrap();
        }
        // raw string with many hashes
        4 => {
            let hashes = "#".repeat(1 + rng.random_range(0..5));
            write!(buf, r#"r{hashes}"hello"{hashes}"#).unwrap();
        }
        // string with repeated escapes
        5 => {
            let n = 10 + rng.random_range(0..50);
            write!(buf, r#""{}""#, "\\n".repeat(n)).unwrap();
        }
        // unterminated string
        6 => buf.push_str(r#""unterminated"#),
        // newline in string
        _ => buf.push_str("\"line1\nline2\""),
    }
    if place_in_grammar {
        buf.push_str(" }");
    }
}

/// Duplicate a grammar fragment many times
fn repeat_fragment(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = &corpus[rng.random_range(0..corpus.len())];
    let repeats = 2 + rng.random_range(0..10);
    let place_in_grammar = rng.random_bool(0.5);
    // Extract the rule body and repeat it in a seq
    if place_in_grammar {
        buf.push_str(r#"grammar { language: "test" } rule program { seq("#);
    }
    for i in 0..repeats {
        if i > 0 {
            buf.push_str(", ");
        }
        write!(buf, r#""item{i}""#).unwrap();
    }
    // Use the repeated items inside a choice or seq
    if place_in_grammar {
        buf.push_str(") }");
    } else {
        // Also try repeating the entire grammar text
        for _ in 0..repeats {
            buf.push_str(base);
            buf.push('\n');
        }
    }
}

/// Generate input with unicode (multi-byte chars)
fn unicode_input(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let unicode_chars = [
        '\u{00e9}',
        '\u{00f1}',
        '\u{00fc}',
        '\u{03b1}',
        '\u{03b2}',
        '\u{4e16}',
        '\u{754c}',
        '\u{1f600}',
        '\u{1f4a9}',
        '\u{200b}',
        '\u{feff}',
        '\u{fffd}',
    ];
    buf.push_str(&corpus[rng.random_range(0..corpus.len())]);
    // let mut chars: Vec<char> = base.chars().collect();
    let insertions = 1 + rng.random_range(0..5);
    for _ in 0..insertions {
        if buf.is_empty() {
            break;
        }
        let mut idx = rng.random_range(0..buf.len());
        while !buf.is_char_boundary(idx) {
            idx -= 1;
        }
        let ch = unicode_chars[rng.random_range(0..unicode_chars.len())];
        buf.insert(idx, ch);
    }
}

pub fn run(options: &super::FuzzDsl) -> Result<()> {
    let root = super::root_dir();
    let corpus = load_dir(&root.join(CORPUS_DIR).join("inputs"))?;
    if !corpus.is_empty() {
        Err(anyhow!("no .tsg files found in {CORPUS_DIR}/inputs"))?;
    }

    let dummy_path = PathBuf::from("grammar.tsg");
    let seed = options.seed.unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as u64
    });
    let mut rng = SmallRng::seed_from_u64(seed);
    let mut stderr = std::io::stderr().lock();

    ctrlc::set_handler(|| STOP.store(true, Ordering::Relaxed))
        .expect("failed to set Ctrl+C handler");

    writeln!(stderr, "fuzzing with seed {seed}").unwrap();

    let mut iterations = 0u64;
    let mut panics = 0u64;
    let mut input = String::new();
    let start = Instant::now();
    let mut last_checkpoint = start;

    // Suppress panic messages during fuzzing
    let default_hook = panic::take_hook();
    panic::set_hook(Box::new(|_| {}));

    loop {
        if STOP.load(Ordering::Relaxed) {
            break;
        }
        if let Some(max) = options.iterations
            && iterations >= max
        {
            break;
        }

        input.clear();
        match rng.random_range(0..10) {
            0 => random_ascii(&mut rng, 512, &mut input),
            1 => truncate_grammar(&mut rng, &corpus, &mut input),
            2 => mutate_bytes(&mut rng, &corpus, &mut input),
            3 => token_soup(&mut rng, &mut input),
            4 => deep_nesting(&mut rng, &mut input),
            5 => splice(&mut rng, &corpus, &mut input),
            6 => keyword_swap(&mut rng, &corpus, &mut input),
            7 => many_rules(&mut rng, &mut input),
            8 => string_edge_cases(&mut rng, &mut input),
            _ => {
                // Rotate through less common strategies
                match iterations % 3 {
                    0 => repeat_fragment(&mut rng, &corpus, &mut input),
                    1 => unicode_input(&mut rng, &corpus, &mut input),
                    _ => random_ascii(&mut rng, 4096, &mut input),
                }
            }
        }

        let input_ref = input.as_str();
        let path_ref = dummy_path.as_path();
        let result = catch_unwind(|| {
            let _ = parse_native_dsl(input_ref, path_ref);
        });

        if result.is_err() {
            panics += 1;
            let filename = format!("fuzz_crash_{panics}.tsg");
            writeln!(stderr, "\nPANIC on iteration {iterations}!").unwrap();
            writeln!(stderr, "saving crash input to {filename}").unwrap();
            std::fs::write(&filename, &input).ok();
            writeln!(
                stderr,
                "input ({} bytes): {:?}",
                input.len(),
                &input[..input.len().min(200)]
            )
            .unwrap();
        }

        iterations += 1;
        if iterations.is_multiple_of(DISPLAY_INTERVAL) {
            let now = Instant::now();
            let rate =
                (DISPLAY_INTERVAL as f64) / now.duration_since(last_checkpoint).as_secs_f64();
            last_checkpoint = now;
            stderr.write_all(b"\r").unwrap();
            write_num(&mut stderr, iterations);
            stderr.write_all(b" iterations (").unwrap();
            write_num(&mut stderr, rate as u64);
            stderr.write_all(b"/s, ").unwrap();
            write_num(&mut stderr, panics);
            // clear to end of line
            stderr.write_all(b" panics)\x1b[K").unwrap();
        }
    }

    drop(panic::take_hook());
    panic::set_hook(default_hook);

    let elapsed = start.elapsed().as_secs_f64();
    let rate = iterations as f64 / elapsed;
    {
        stderr.write_all(b"\n").unwrap();
        write_num(&mut stderr, iterations);
        write!(stderr, " iterations in {elapsed:.1}s (").unwrap();
        write_num(&mut stderr, rate as u64);
        stderr.write_all(b"/s), ").unwrap();
        write_num(&mut stderr, panics);
        stderr.write_all(b" panics found\n").unwrap();
    }

    if panics > 0 {
        Err(anyhow!("{panics} panic(s) found"))?;
    }

    Ok(())
}
