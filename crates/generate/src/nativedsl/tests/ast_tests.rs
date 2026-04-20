use super::*;

// Parsing the DSL is fairly  straightforward, but may run into complications in the future
// Leave this here as a starting point when more specific parsing tests are needed

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
