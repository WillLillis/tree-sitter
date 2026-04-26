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

/// Generate test functions for errors in inherited (child) modules.
/// Calls `inherit_err()` and asserts the error kind + path match.
macro_rules! inherit_error_tests {
    ($variant:ident { $($name:ident { $base:expr, $expected:expr })* }) => {
        $(#[test] fn $name() {
            let (err, base_path) = inherit_err($base);
            let DslError::Module(m) = &err else { panic!("expected Module, got {err:?}") };
            let DslError::$variant(e) = m.inner.as_ref() else {
                panic!(concat!("expected ", stringify!($variant), ", got {:?}"), m.inner)
            };
            assert_eq!(e.kind, $expected);
            assert_eq!(m.path, base_path);
        })*
    };
}

mod lex;
mod lower;
mod parse;
mod resolve;
mod typecheck;
