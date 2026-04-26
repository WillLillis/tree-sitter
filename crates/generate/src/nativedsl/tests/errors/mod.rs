/// Generate test functions that assert a specific error kind for a given input.
/// `$variant` is the `DslError` variant to match (Lex, Parse, Type, Lower).
macro_rules! error_tests {
    ($variant:ident { $($name:ident { $input:expr, $expected:expr })* }) => {
        $(#[test] fn $name() {
            let e = assert_err!(dsl_err($input), $variant);
            assert_eq!(e.kind, $expected);
        })*
    };
}

mod lex;
mod lower;
mod parse;
mod resolve;
mod typecheck;
