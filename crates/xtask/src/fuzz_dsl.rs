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

fn pick<'a>(rng: &mut SmallRng, items: &'a [impl AsRef<str>]) -> &'a str {
    items[rng.random_range(0..items.len())].as_ref()
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

const UNARY_COMBINATORS: &[&str] = &["repeat", "repeat1", "optional", "token", "token_immediate"];

// Random expressions for generating adversarial inputs
#[rustfmt::skip]
const RANDOM_EXPRS: &[&str] = &[
    "\"x\"", "\"hello\"", "42", "-1", "0", "blank()", "foo", "bar", "program",
    "seq(\"a\", \"b\")", "choice(\"a\", \"b\")", "repeat(\"x\")", "optional(\"x\")",
    "prec(1, \"x\")", "prec_left(1, \"x\")", "token(\"x\")", "field(name, \"x\")",
    "regexp(\"[a-z]+\")", "concat(\"a\", \"b\")", "[\"a\", \"b\"]", "(1, \"a\")",
    "{ a: 1, b: 2 }",
];

// -- Generators (original) --

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
    let base = pick(rng, corpus);
    let cut = rng.random_range(0..base.len() + 1);
    buf.push_str(&base[..cut]);
}

fn mutate_bytes(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = pick(rng, corpus);
    let mut bytes: Vec<u8> = base.bytes().collect();
    let mutations = 1 + rng.random_range(0..5);
    for _ in 0..mutations {
        if bytes.is_empty() {
            break;
        }
        match rng.random_range(0..4) {
            0 => {
                let idx = rng.random_range(0..bytes.len());
                bytes[idx] = rng.random::<u8>() % 128;
            }
            1 => {
                let idx = rng.random_range(0..bytes.len() + 1);
                bytes.insert(idx, rng.random::<u8>() % 128);
            }
            2 => {
                let idx = rng.random_range(0..bytes.len());
                bytes.remove(idx);
            }
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

fn deep_nesting(rng: &mut SmallRng, buf: &mut String) {
    let depth = 5 + rng.random_range(0..50);
    buf.push_str(r#"grammar { language: "test" } rule program { "#);
    for _ in 0..depth {
        let combinator = pick(rng, UNARY_COMBINATORS);
        write!(buf, "{combinator}(").unwrap();
    }
    buf.push_str("\"x\"");
    for _ in 0..depth {
        buf.push(')');
    }
    buf.push_str(" }");
}

fn splice(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let a = pick(rng, corpus);
    let b = pick(rng, corpus);
    let cut_a = rng.random_range(0..a.len() + 1);
    let cut_b = rng.random_range(0..b.len() + 1);
    write!(buf, "{}{}", &a[..cut_a], &b[cut_b..]).unwrap();
}

fn keyword_swap(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    buf.push_str(pick(rng, corpus));
    let swaps = 1 + rng.random_range(0..3);
    for _ in 0..swaps {
        let from = pick(rng, KEYWORDS);
        let to = pick(rng, KEYWORDS);
        if let Some(pos) = buf.find(from) {
            buf.replace_range(pos..pos + from.len(), to);
        }
    }
}

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
        write!(buf, "rule r{i} {{ ").unwrap();
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

fn string_edge_cases(rng: &mut SmallRng, buf: &mut String) {
    let place_in_grammar = rng.random_range(0..8) < 2;
    if place_in_grammar {
        write!(buf, r#"grammar {{ language: "test" }} rule program {{ "#).unwrap();
    }
    match rng.random_range(0..8) {
        0 => buf.push_str(r#""""#),
        1 => buf.push_str(r#""\n\t\r\\\"\0""#),
        2 => {
            let len = 100 + rng.random_range(0..500);
            write!(buf, r#""{}""#, "a".repeat(len)).unwrap();
        }
        3 => {
            let len = 50 + rng.random_range(0..200);
            write!(
                buf,
                r#"grammar {{ language: "test" }} let {} = "x" rule program {{ "x" }}"#,
                "a".repeat(len)
            )
            .unwrap();
        }
        4 => {
            let hashes = "#".repeat(1 + rng.random_range(0..5));
            write!(buf, r#"r{hashes}"hello"{hashes}"#).unwrap();
        }
        5 => {
            let n = 10 + rng.random_range(0..50);
            write!(buf, r#""{}""#, "\\n".repeat(n)).unwrap();
        }
        6 => buf.push_str(r#""unterminated"#),
        _ => buf.push_str("\"line1\nline2\""),
    }
    if place_in_grammar {
        buf.push_str(" }");
    }
}

fn repeat_fragment(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = pick(rng, corpus);
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
    buf.push_str(pick(rng, corpus));
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

// -- Generators (new) --

/// Wrong-type arguments to builtin functions
fn wrong_type_args(rng: &mut SmallRng, buf: &mut String) {
    buf.push_str(r#"grammar { language: "test" } rule program { "#);
    match rng.random_range(0..10) {
        // prec with string value
        0 => buf.push_str(r#"prec("not_int", "x")"#),
        // prec with rule value
        1 => buf.push_str(r#"prec(repeat("x"), "y")"#),
        // field with int name
        2 => buf.push_str(r#"field(42, "x")"#),
        // repeat with int
        3 => buf.push_str("repeat(42)"),
        // token with int
        4 => buf.push_str("token(42)"),
        // alias with int target
        5 => buf.push_str(r#"alias("x", 42)"#),
        // seq with mixed types
        6 => buf.push_str(r#"seq(42, "x", repeat("y"))"#),
        // nested wrong types
        7 => buf.push_str(r#"prec_left("a", prec("b", "c"))"#),
        // for with non-list
        8 => write!(buf, r#"seq(for (x: rule) in "not_a_list" {{ x }})"#).unwrap(),
        // append with non-list
        _ => write!(buf, r#"seq(for (x: str) in append("a", "b") {{ x }})"#).unwrap(),
    }
    buf.push_str(" }");
}

/// Random config field combinations
fn random_config(rng: &mut SmallRng, buf: &mut String) {
    buf.push_str(r#"grammar { language: "test""#);

    // Random subset of config fields
    if rng.random_bool(0.3) {
        write!(buf, ", extras: [{}]", pick(rng, RANDOM_EXPRS)).unwrap();
    }
    if rng.random_bool(0.3) {
        write!(buf, ", externals: [{}]", pick(rng, RANDOM_EXPRS)).unwrap();
    }
    if rng.random_bool(0.3) {
        buf.push_str(", inline: [foo]");
    }
    if rng.random_bool(0.3) {
        buf.push_str(", supertypes: [bar]");
    }
    if rng.random_bool(0.3) {
        buf.push_str(", word: ident");
    }
    if rng.random_bool(0.2) {
        buf.push_str(", conflicts: [[foo, bar]]");
    }
    if rng.random_bool(0.2) {
        buf.push_str(r#", precedences: [["a", "b"]]"#);
    }
    if rng.random_bool(0.2) {
        buf.push_str(r#", reserved: { default: ["if", "else"] }"#);
    }

    buf.push_str(" } ");

    // Sometimes add rules, sometimes don't (tests missing rule errors)
    let rule_count = rng.random_range(0..5);
    if rule_count > 0 {
        buf.push_str(r#"rule program { "x" } "#);
    }
    for name in ["foo", "bar", "ident", "expr"].iter().take(rule_count) {
        write!(buf, r#"rule {name} {{ "{name}" }} "#).unwrap();
    }
}

/// Boundary conditions: huge values, long identifiers, many args
fn boundary_values(rng: &mut SmallRng, buf: &mut String) {
    match rng.random_range(0..8) {
        // i32 max
        0 => buf.push_str(r#"grammar { language: "test" } rule program { prec(2147483647, "x") }"#),
        // i32 min
        1 => {
            buf.push_str(r#"grammar { language: "test" } rule program { prec(-2147483648, "x") }"#);
        }
        // i32 overflow
        2 => buf.push_str(r#"grammar { language: "test" } rule program { prec(2147483648, "x") }"#),
        // Many-argument seq
        3 => {
            let count = 100 + rng.random_range(0..500);
            buf.push_str(r#"grammar { language: "test" } rule program { seq("#);
            for i in 0..count {
                if i > 0 {
                    buf.push_str(", ");
                }
                write!(buf, r#""a{i}""#).unwrap();
            }
            buf.push_str(") }");
        }
        // Very long identifier
        4 => {
            let len = rng.random_range(0..3) * 255 + rng.random_range(0..10);
            write!(
                buf,
                r#"grammar {{ language: "test" }} rule {} {{ "x" }}"#,
                "a".repeat(len)
            )
            .unwrap();
        }
        // Empty input
        5 => {}
        // Whitespace only
        6 => {
            let len = 1 + rng.random_range(0..100);
            for _ in 0..len {
                buf.push(if rng.random_bool(0.5) { ' ' } else { '\n' });
            }
        }
        // Comment only
        _ => {
            let lines = 1 + rng.random_range(0..10);
            for _ in 0..lines {
                buf.push_str("// this is a comment\n");
            }
        }
    }
}

/// Insert comments between random tokens in a corpus grammar
fn comment_insertion(rng: &mut SmallRng, corpus: &[String], buf: &mut String) {
    let base = pick(rng, corpus);
    let insertions = 1 + rng.random_range(0..5);
    buf.push_str(base);
    for _ in 0..insertions {
        if buf.is_empty() {
            break;
        }
        // Find a space to replace with a comment
        let bytes = buf.as_bytes();
        let spaces: Vec<usize> = bytes
            .iter()
            .enumerate()
            .filter(|&(_, &b)| b == b' ' || b == b'\n')
            .map(|(i, _)| i)
            .collect();
        if spaces.is_empty() {
            break;
        }
        let idx = spaces[rng.random_range(0..spaces.len())];
        buf.insert_str(idx, " //comment\n");
    }
}

/// Type system adversaries: recursive fns, shadowing, wrong-type appends
fn type_adversary(rng: &mut SmallRng, buf: &mut String) {
    match rng.random_range(0..8) {
        // Recursive function
        0 => buf.push_str(
            r#"grammar { language: "test" } fn f(x: rule) -> rule { f(x) } rule program { f("x") }"#,
        ),
        // Variable shadowing
        1 => buf.push_str(
            r#"grammar { language: "test" } let x = "a" let x = "b" rule program { x }"#,
        ),
        // Cross-type append
        2 => buf.push_str(
            r#"grammar { language: "test" } let a = [1, 2] let b = ["a"] rule program { seq(for (x: str) in append(a, b) { x }) }"#,
        ),
        // Deeply nested objects
        3 => {
            buf.push_str(r#"grammar { language: "test" } let o = { a: { b: { c: 1 } } } "#);
            buf.push_str(r#"rule program { prec(o.a.b.c, "x") }"#);
        }
        // Deeply nested lists
        4 => buf.push_str(
            r#"grammar { language: "test" } let x = [[[["a"]]]] rule program { "x" }"#,
        ),
        // Deeply nested tuples
        5 => buf.push_str(
            r#"grammar { language: "test" } let x = (((("a",),),),) rule program { "x" }"#,
        ),
        // Function with too many/few args
        6 => buf.push_str(
            r#"grammar { language: "test" } fn f(a: rule, b: rule) -> rule { seq(a, b) } rule program { f("x") }"#,
        ),
        // Use keyword in expression position
        _ => {
            let kw = pick(rng, KEYWORDS);
            write!(
                buf,
                r#"grammar {{ language: "test" }} rule program {{ {kw} }}"#
            )
            .unwrap();
        }
    }
}

/// Inheritance: valid derived grammars
fn inheritance_valid(rng: &mut SmallRng, base_path: &Path, buf: &mut String) {
    let abs = base_path.display();
    writeln!(buf, "let base = inherit(\"{abs}\")").unwrap();
    buf.push_str("grammar { language: \"derived\", inherits: base");

    // Random config overrides
    if rng.random_bool(0.3) {
        buf.push_str(", extras: base.extras");
    }
    if rng.random_bool(0.3) {
        buf.push_str(", word: base.word");
    }
    buf.push_str(" }\n");

    // Random override rules
    let overrides = rng.random_range(0..3);
    let override_names = ["program", "statement", "expression"];
    for name in override_names.iter().take(overrides) {
        writeln!(buf, "override rule {name} {{ \"{name}_v2\" }}").unwrap();
    }

    // Random new rules
    let new_rules = rng.random_range(0..3);
    for i in 0..new_rules {
        writeln!(buf, "rule new_rule_{i} {{ \"new_{i}\" }}").unwrap();
    }
}

/// Inheritance: invalid/broken patterns
fn inheritance_invalid(rng: &mut SmallRng, base_path: &Path, buf: &mut String) {
    let abs = base_path.display();
    match rng.random_range(0..7) {
        // Override without inherit
        0 => buf.push_str(
            r#"grammar { language: "test" } override rule foo { "x" } rule program { "y" }"#,
        ),
        // Override nonexistent rule
        1 => {
            writeln!(buf, "let base = inherit(\"{abs}\")").unwrap();
            buf.push_str("grammar { language: \"derived\", inherits: base }\n");
            buf.push_str("override rule nonexistent_rule { \"x\" }\n");
        }
        // Inherit from nonexistent file
        2 => {
            buf.push_str("let base = inherit(\"nonexistent_file.tsg\")\n");
            buf.push_str("grammar { language: \"derived\", inherits: base }\n");
        }
        // Inherit from empty path
        3 => {
            buf.push_str("let base = inherit(\"\")\n");
            buf.push_str("grammar { language: \"derived\", inherits: base }\n");
        }
        // :: on non-grammar
        4 => {
            buf.push_str(r#"grammar { language: "test" } let x = "hello" rule program { x::foo }"#);
        }
        // Double inherit
        5 => {
            writeln!(buf, "let a = inherit(\"{abs}\")").unwrap();
            writeln!(buf, "let b = inherit(\"{abs}\")").unwrap();
            buf.push_str("grammar { language: \"derived\", inherits: a }\n");
        }
        // Mutated valid inheritance
        _ => {
            inheritance_valid(rng, base_path, buf);
            // Corrupt random bytes
            let mutations = 1 + rng.random_range(0..3);
            let mut bytes = std::mem::take(buf).into_bytes();
            for _ in 0..mutations {
                if bytes.is_empty() {
                    break;
                }
                let idx = rng.random_range(0..bytes.len());
                bytes[idx] = rng.random::<u8>() % 128;
            }
            *buf = String::from_utf8_lossy(&bytes).into_owned();
        }
    }
}

/// Set up base grammars in a temp directory for inheritance testing
fn setup_inherit_dir() -> Result<PathBuf> {
    let dir = std::env::temp_dir().join("fuzz_dsl_inherit");
    std::fs::create_dir_all(&dir)?;

    std::fs::write(
        dir.join("base.tsg"),
        r#"grammar { language: "base", extras: [regexp("\\s")], word: identifier }
rule program { repeat(statement) }
rule statement { choice(expression, ";") }
rule expression { identifier }
rule identifier { regexp("[a-z]+") }
"#,
    )?;

    std::fs::write(
        dir.join("base_with_config.tsg"),
        r#"grammar {
    language: "configured",
    extras: [regexp("\\s"), "\n"],
    inline: [_inner],
    conflicts: [[expr, stmt]],
    precedences: [["*", "+"]],
}
rule program { repeat(stmt) }
rule stmt { choice(expr, ";") }
rule expr { choice(_inner, "x") }
rule _inner { "y" }
"#,
    )?;

    Ok(dir)
}

// -- Runner --

pub fn run(options: &super::FuzzDsl) -> Result<()> {
    let root = super::root_dir();
    let corpus = load_dir(&root.join(CORPUS_DIR).join("inputs"))?;
    if corpus.is_empty() {
        Err(anyhow!("no .tsg files found in {CORPUS_DIR}/inputs"))?;
    }

    let inherit_dir = setup_inherit_dir()?;
    let base_paths = [
        inherit_dir.join("base.tsg"),
        inherit_dir.join("base_with_config.tsg"),
    ];

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

        match rng.random_range(0..16) {
            0 => random_ascii(&mut rng, 512, &mut input),
            1 => truncate_grammar(&mut rng, &corpus, &mut input),
            2 => mutate_bytes(&mut rng, &corpus, &mut input),
            3 => token_soup(&mut rng, &mut input),
            4 => deep_nesting(&mut rng, &mut input),
            5 => splice(&mut rng, &corpus, &mut input),
            6 => keyword_swap(&mut rng, &corpus, &mut input),
            7 => many_rules(&mut rng, &mut input),
            8 => string_edge_cases(&mut rng, &mut input),
            9 => wrong_type_args(&mut rng, &mut input),
            10 => random_config(&mut rng, &mut input),
            11 => boundary_values(&mut rng, &mut input),
            12 => comment_insertion(&mut rng, &corpus, &mut input),
            13 => type_adversary(&mut rng, &mut input),
            14 => {
                let base = &base_paths[rng.random_range(0..base_paths.len())];
                inheritance_valid(&mut rng, base, &mut input);
            }
            15 => {
                let base = &base_paths[rng.random_range(0..base_paths.len())];
                inheritance_invalid(&mut rng, base, &mut input);
            }
            _ => match iterations % 3 {
                0 => repeat_fragment(&mut rng, &corpus, &mut input),
                1 => unicode_input(&mut rng, &corpus, &mut input),
                _ => random_ascii(&mut rng, 4096, &mut input),
            },
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
            display_progress(&mut stderr, iterations, panics, &mut last_checkpoint);
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

fn display_progress(
    stderr: &mut std::io::StderrLock<'_>,
    iterations: u64,
    panics: u64,
    last_checkpoint: &mut Instant,
) {
    let now = Instant::now();
    let rate = (DISPLAY_INTERVAL as f64) / now.duration_since(*last_checkpoint).as_secs_f64();
    *last_checkpoint = now;
    stderr.write_all(b"\r").unwrap();
    write_num(stderr, iterations);
    stderr.write_all(b" iterations (").unwrap();
    write_num(stderr, rate as u64);
    stderr.write_all(b"/s, ").unwrap();
    write_num(stderr, panics);
    // clear to end of line
    stderr.write_all(b" panics)\x1b[K").unwrap();
}
