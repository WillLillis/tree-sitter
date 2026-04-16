use super::*;

// ===== Type sizes =====

#[test]
fn node_enum_is_16_bytes() {
    // 4 nodes per 64-byte cache line
    assert_eq!(std::mem::size_of::<ast::Node>(), 16);
}

#[test]
fn print_type_sizes() {
    use super::super::{lower::LowerError, typecheck::TypeError};
    use crate::nativedsl::lexer::LexError;
    use crate::nativedsl::parser::ParseError;
    use crate::nativedsl::resolve::ResolveError;
    let sizes = [
        ("LowerError", std::mem::size_of::<LowerError>()),
        ("TypeError", std::mem::size_of::<TypeError>()),
        ("LexError", std::mem::size_of::<LexError>()),
        ("ParseError", std::mem::size_of::<ParseError>()),
        ("ResolveError", std::mem::size_of::<ResolveError>()),
        (
            "Result<(), LowerError>",
            std::mem::size_of::<Result<(), LowerError>>(),
        ),
        (
            "Result<(), TypeError>",
            std::mem::size_of::<Result<(), TypeError>>(),
        ),
        ("Rule", std::mem::size_of::<Rule>()),
    ];
    for (name, size) in &sizes {
        eprintln!("{name:>30}: {size:>3} bytes");
    }
}

// ===== Minimal grammar =====

#[test]
fn minimal_grammar() {
    let g = dsl(r#"
        grammar { language: "test" }
        rule program { repeat("hello") }
    "#);
    assert_eq!(g.name, "test");
    assert_eq!(g.variables.len(), 1);
    assert_eq!(g.variables[0].name, "program");
    assert_eq!(g.variables[0].kind, VariableType::Named);
    assert_eq!(
        g.variables[0].rule,
        Rule::choice(vec![
            Rule::repeat(Rule::String("hello".into())),
            Rule::Blank
        ])
    );
}
