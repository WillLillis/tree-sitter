# Native DSL Fuzzer

Coverage-guided fuzzing via [cargo-fuzz](https://github.com/rust-fuzz/cargo-fuzz).
Requires Rust nightly (`rustup toolchain install nightly`) and cargo-fuzz (`cargo install cargo-fuzz`).

Run from `crates/generate`:

```sh
cargo +nightly fuzz run native_dsl              # run indefinitely
cargo +nightly fuzz run native_dsl -- -max_total_time=60  # run for 60s
```

Crash inputs are saved to `fuzz/artifacts/native_dsl/`. After fixing, add a
regression `.tsg` file to `src/nativedsl/fuzz_regressions/`.
