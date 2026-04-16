use super::*;

// ===== Node size =====

#[test]
fn node_enum_is_16_bytes() {
    // 4 nodes per 64-byte cache line
    assert_eq!(std::mem::size_of::<ast::Node>(), 16);
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
