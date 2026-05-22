use super::super::*;

error_tests! { Parse {
    error_rule_set_body_requires_rule_decl {
        r#"grammar { language: "test" } macro bad() { let x = 1 } rule program { "x" }"#,
        ParseErrorKind::RuleSetBodyRequiresRuleDecl
    }
    error_empty_rule_set_macro_body {
        r#"grammar { language: "test" } macro bad() {} rule program { "x" }"#,
        ParseErrorKind::EmptyRuleSetMacroBody
    }
}}

error_tests! { Expand {
    error_unknown_top_level_macro {
        r#"grammar { language: "test" } rule program { "x" } not_defined()"#,
        ExpandErrorKind::UnknownMacro("not_defined".into())
    }
    error_expression_macro_called_at_top_level {
        r#"macro expr_only() rule_t { "x" }
           grammar { language: "test" }
           rule program { "y" }
           expr_only()"#,
        ExpandErrorKind::ExpressionMacroAsItem("expr_only".into())
    }
    error_arg_count_mismatch {
        r#"macro one(s: str_t) { rule program { s } }
           grammar { language: "test" }
           one("a", "b")"#,
        ExpandErrorKind::ArgCountMismatch {
            macro_name: "one".into(),
            expected: 1,
            got: 2,
        }
    }
    error_computed_name_not_identifier {
        r#"macro bad(s: str_t) { rule @concat("with space", s) { "x" } }
           grammar { language: "test" }
           bad("foo")"#,
        ExpandErrorKind::InvalidRuleName("with spacefoo".into())
    }
    error_computed_name_non_string_node {
        r#"macro bad() { rule @42 { "x" } }
           grammar { language: "test" }
           bad()"#,
        ExpandErrorKind::NonStringInName
    }
}}

error_tests! { Type {
    error_rule_set_macro_in_expression_context {
        r#"macro rs() { rule a { "x" } }
           grammar { language: "test" }
           rule program { rs() }"#,
        TypeErrorKind::RuleSetMacroInExpressionContext("rs".into())
    }
}}
