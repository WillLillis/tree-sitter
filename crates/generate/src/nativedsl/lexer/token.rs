//! Token vocabulary: the [`TokenKind`] enum (plus its keyword table) and
//! [`Token`]. Shared with the parser, which consumes these directly.

use serde::{Deserialize, Serialize};

use crate::nativedsl::ast::Span;

/// The kind of a lexer token.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TokenKind {
    // Identifiers and literals
    Ident,
    /// String literal. Token span covers `"..."` including quotes.
    StringLit,
    /// Raw string literal. Token span covers `r"..."`, `r#"..."#`, etc.
    RawStringLit {
        hash_count: u8,
    },
    IntLit(u32),
    // Structural keywords
    KwGrammar,
    KwRule,
    KwRules,
    KwLet,
    KwMacro,
    KwFor,
    KwIn,
    // Combinator keywords
    KwSeq,
    KwChoice,
    KwRepeat,
    KwRepeat1,
    KwOptional,
    KwBlank,
    KwField,
    KwAlias,
    KwToken,
    KwPrec,
    KwPrecLeft,
    KwPrecRight,
    KwPrecDynamic,
    KwReserved,
    KwTokenImmediate,
    KwConcat,
    KwRegexp,
    KwInherit,
    KwImport,
    KwOverride,
    KwAppend,
    KwGrammarConfig,
    KwExpect,
    // Punctuation
    LBrace,
    RBrace,
    LParen,
    RParen,
    LBracket,
    RBracket,
    Comma,
    Colon,
    ColonColon,
    Dot,
    Minus,
    Plus,
    Eq,
    Lt,
    Gt,
    Pound,
    At,
    // Other
    Comment,
    Eof,
}

/// Maps keyword text to `TokenKind` variant. Used by the lexer, Display, and `is_keyword`.
macro_rules! keywords {
    (
        decls { $($d_variant:ident => $d_str:literal),* $(,)? }
        combinators { $($c_variant:ident => $c_str:literal),* $(,)? }
    ) => {
        impl TokenKind {
            /// Keywords callable in expression position. Used to seed
            /// "did you mean?" suggestions when an identifier in a rule body
            /// fails to resolve.
            pub(crate) const COMBINATOR_KEYWORD_NAMES: &'static [&'static str] = &[$($c_str),*];

            #[must_use]
            pub const fn is_keyword(self) -> bool {
                matches!(self, $(Self::$d_variant)|* $(| Self::$c_variant)*)
            }

            /// Return the keyword string if this is a keyword token, or `None`.
            const fn keyword_str(self) -> Option<&'static str> {
                match self {
                    $(Self::$d_variant => Some($d_str),)*
                    $(Self::$c_variant => Some($c_str),)*
                    _ => None,
                }
            }

            /// Map keyword text to a `TokenKind`, or return `Ident`.
            pub(super) fn from_keyword(text: &str) -> Self {
                match text {
                    $($d_str => Self::$d_variant,)*
                    $($c_str => Self::$c_variant,)*
                    _ => Self::Ident,
                }
            }
        }
    };
}

keywords! {
    decls {
        KwGrammar => "grammar", KwRule => "rule", KwRules => "rules", KwLet => "let",
        KwMacro => "macro", KwOverride => "override", KwExpect => "expect",
        KwIn => "in",
    }
    combinators {
        KwFor => "for", KwSeq => "seq", KwChoice => "choice",
        KwRepeat => "repeat", KwRepeat1 => "repeat1", KwOptional => "optional",
        KwBlank => "blank", KwField => "field", KwAlias => "alias", KwToken => "token",
        KwPrec => "prec", KwPrecLeft => "prec_left", KwPrecRight => "prec_right",
        KwPrecDynamic => "prec_dynamic", KwReserved => "reserved",
        KwTokenImmediate => "token_immediate", KwConcat => "concat", KwRegexp => "regexp",
        KwInherit => "inherit", KwImport => "import",
        KwAppend => "append", KwGrammarConfig => "grammar_config",
    }
}

impl std::fmt::Display for TokenKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(kw) = self.keyword_str() {
            return write!(f, "'{kw}'");
        }
        f.write_str(match self {
            Self::Ident => "identifier",
            Self::StringLit => "string literal",
            Self::RawStringLit { .. } => "raw string literal",
            Self::IntLit(_) => "integer literal",
            Self::LBrace => "'{'",
            Self::RBrace => "'}'",
            Self::LParen => "'('",
            Self::RParen => "')'",
            Self::LBracket => "'['",
            Self::RBracket => "']'",
            Self::Comma => "','",
            Self::Colon => "':'",
            Self::ColonColon => "'::'",
            Self::Dot => "'.'",
            Self::Minus => "'-'",
            Self::Plus => "'+'",
            Self::Eq => "'='",
            Self::Lt => "'<'",
            Self::Gt => "'>'",
            Self::Pound => "'#'",
            Self::At => "'@'",
            Self::Comment => "comment",
            Self::Eof => "end of file",
            _ => unreachable!(),
        })
    }
}

#[derive(Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}
