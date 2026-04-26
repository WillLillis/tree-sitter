# Native DSL Fuzzer

Coverage-guided fuzzing for the native grammar DSL (`grammar.tsg`) pipeline using
[libFuzzer](https://llvm.org/docs/LibFuzzer.html) via
[cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Requirements

- Rust nightly toolchain: `rustup toolchain install nightly`
- cargo-fuzz: `cargo install cargo-fuzz`

## Targets

- **`native_dsl`** - Fuzzes the full pipeline (lex, parse, resolve, typecheck, lower) with arbitrary input.
- **`native_dsl_inherit`** - Same pipeline but with a base grammar on disk, exercising inheritance paths.

## Usage

All commands should be run from the project root.

```sh
# List available targets
cargo +nightly fuzz list

# Run indefinitely (ctrl+c to stop)
cargo +nightly fuzz run native_dsl

# Run for a fixed duration
cargo +nightly fuzz run native_dsl -- -max_total_time=60

# Run the inheritance target
cargo +nightly fuzz run native_dsl_inherit

# Reproduce a crash artifact
cargo +nightly fuzz run native_dsl fuzz/artifacts/native_dsl/<artifact_file>
```

## Corpus

The `corpus/` directory is gitignored since libFuzzer grows it rapidly during runs.
On the first run, each fuzz target automatically seeds the corpus from the xtask fuzzer
inputs, fuzz regression files, and real rewritten grammars.

## Handling crashes

When libFuzzer finds a crash, it saves the input to `fuzz/artifacts/native_dsl/`.

```sh
# Reproduce a crash
cargo +nightly fuzz run native_dsl fuzz/artifacts/native_dsl/<crash-file>

# Minimize the crash input
cargo +nightly fuzz tmin native_dsl /absolute/path/to/crash-file

# Note: tmin requires an absolute path to the artifact file.
```

After fixing the bug:

1. Verify the fix: re-run the crash artifact to confirm it no longer crashes.
2. Add a regression file to `src/nativedsl/fuzz_regressions/` (the `fuzz_regressions` test
   iterates all `.tsg` files there and verifies none panic).
3. Add a targeted test in `tests.rs` that asserts the expected error.
