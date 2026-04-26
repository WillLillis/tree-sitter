# Native DSL Fuzzer

Coverage-guided fuzzing for the native grammar DSL (`grammar.tsg`) pipeline using
[libFuzzer](https://llvm.org/docs/LibFuzzer.html) via
[cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).

## Requirements

- Rust nightly: `rustup toolchain install nightly`
- cargo-fuzz: `cargo install cargo-fuzz`

## Usage

Run from `crates/generate`:

```sh
cargo +nightly fuzz run native_dsl              # run indefinitely
cargo +nightly fuzz run native_dsl -- -max_total_time=60  # run for 60s
```

On first run, seed files from `fuzz/seeds/` and `src/nativedsl/fuzz_regressions/`
are copied into the corpus automatically.

## Handling crashes

Crash inputs are saved to `fuzz/artifacts/native_dsl/`. To reproduce:

```sh
cargo +nightly fuzz run native_dsl fuzz/artifacts/native_dsl/<crash-file>
```

After fixing: add a regression `.tsg` file to `src/nativedsl/fuzz_regressions/`
and a targeted test if appropriate.
