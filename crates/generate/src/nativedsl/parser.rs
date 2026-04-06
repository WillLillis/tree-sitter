//! Recursive descent parser for the native grammar DSL.
//!
//! Consumes the `Vec<Token>` produced by the lexer and builds an arena-based
//! [`Ast`]. The parser handles:
//!
//! - Top-level items: `grammar`, `rule`, `let`, `fn`
//! - Expressions: combinators (`seq`, `choice`, `repeat`, etc.), literals,
//!   identifiers, field access, function calls, `for` expressions
//! - Types: `rule`, `str`, `int`, `list<T>`, `(T, U, ...)`
//! - Grammar configuration fields: `language`, `extras`, `externals`,
//!   `conflicts`, `precedences`, `word`, `inline`, `supertypes`, `reserved`
//!
//! Trailing commas are allowed everywhere. All identifiers start as
//! `Node::Ident` and are resolved to `RuleRef`/`VarRef` in a later pass.

use serde::Serialize;
use thiserror::Error;

use super::{
    ast::{
        Ast, FnConfig, ForConfig, GrammarConfig, Node, NodeId, NodeList, Param, PrecEntry,
        ReservedWordSet, Span,
    },
    lexer::{Token, TokenKind},
};

/// Recursive descent parser. Consumes tokens and builds an [`Ast`].
pub struct Parser<'src> {
    tokens: Vec<Token>,
    pos: usize,
    pub ast: Ast<'src>,
}

impl<'src> Parser<'src> {
    /// Create a new parser from a token stream and source text.
    pub fn new(tokens: Vec<Token>, source: &'src str) -> Self {
        Self {
            tokens,
            pos: 0,
            ast: Ast::new(source),
        }
    }

    /// Parse the entire token stream into an [`Ast`].
    ///
    /// Consumes `self` and returns the completed AST. All `NodeId`s are valid
    /// indices into the returned `Ast::nodes` and `Ast::spans` vectors.
    ///
    /// # Errors
    ///
    /// Returns [`ParseError`] on syntax errors.
    pub fn parse(mut self) -> Result<Ast<'src>, ParseError> {
        while !self.at_eof() {
            let id = self.parse_item()?;
            self.ast.root_items.push(id);
        }
        Ok(self.ast)
    }

    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }

    fn at_eof(&self) -> bool {
        self.tokens[self.pos].kind == TokenKind::Eof
    }

    fn at(&self, kind: &TokenKind) -> bool {
        self.tokens[self.pos].kind == *kind
    }

    fn eat(&mut self, kind: &TokenKind) -> Option<Span> {
        if self.at(kind) {
            let span = self.span();
            self.pos += 1;
            Some(span)
        } else {
            None
        }
    }

    fn expect(&mut self, kind: &TokenKind) -> Result<Span, ParseError> {
        self.eat(kind).ok_or_else(|| {
            self.error(ParseErrorKind::ExpectedToken {
                expected: kind.clone(),
                got: self.tokens[self.pos].kind.clone(),
            })
        })
    }

    /// Eat a token matching `kind` exactly, returning its span.
    fn eat_kind(&mut self, kind: &TokenKind) -> Option<Span> {
        if self.tokens[self.pos].kind == *kind {
            let span = self.span();
            self.pos += 1;
            Some(span)
        } else {
            None
        }
    }

    fn expect_ident(&mut self) -> Result<Span, ParseError> {
        self.eat_kind(&TokenKind::Ident)
            .ok_or_else(|| self.error(ParseErrorKind::ExpectedIdent))
    }

    fn expect_string(&mut self) -> Result<Span, ParseError> {
        self.eat_kind(&TokenKind::StringLit)
            .ok_or_else(|| self.error(ParseErrorKind::ExpectedString))
    }

    /// Accept an identifier or keyword as a name (for field names that clash with keywords).
    /// Returns the span of the name token.
    fn expect_name(&mut self) -> Result<Span, ParseError> {
        let span = self.span();
        match &self.tokens[self.pos].kind {
            TokenKind::Ident
            | TokenKind::KwField
            | TokenKind::KwToken
            | TokenKind::KwChoice
            | TokenKind::KwAlias
            | TokenKind::KwRule
            | TokenKind::KwSeq
            | TokenKind::KwRepeat
            | TokenKind::KwRepeat1
            | TokenKind::KwOptional
            | TokenKind::KwBlank
            | TokenKind::KwPrec
            | TokenKind::KwReserved
            | TokenKind::KwTokenImmediate
            | TokenKind::KwConcat
            | TokenKind::KwRegex
            | TokenKind::KwLet
            | TokenKind::KwFn
            | TokenKind::KwFor
            | TokenKind::KwIn
            | TokenKind::KwGrammar => {
                self.pos += 1;
                Ok(span)
            }
            _ => Err(self.error(ParseErrorKind::ExpectedName)),
        }
    }

    fn error(&self, kind: ParseErrorKind) -> ParseError {
        ParseError {
            kind,
            span: self.span(),
        }
    }

    fn parse_item(&mut self) -> Result<NodeId, ParseError> {
        match &self.tokens[self.pos].kind {
            TokenKind::KwGrammar => self.parse_grammar_block(),
            TokenKind::KwRule => self.parse_rule_def(),
            TokenKind::KwLet => self.parse_let_def(),
            TokenKind::KwFn => self.parse_fn_def(),
            _ => Err(self.error(ParseErrorKind::ExpectedItem)),
        }
    }

    fn parse_grammar_block(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(&TokenKind::KwGrammar)?;
        self.expect(&TokenKind::LBrace)?;
        let mut config = GrammarConfig::default();

        loop {
            if let Some(end) = self.eat(&TokenKind::RBrace) {
                self.ast.context.grammar_config = Some(config);
                return Ok(self.ast.push(Node::Grammar, start.merge(end)));
            }
            let key_span = self.expect_ident()?;
            let key = self.ast.text(key_span).to_string();
            self.expect(&TokenKind::Colon)?;
            match key.as_str() {
                "language" => {
                    let span = self.expect_string()?;
                    // Strip quotes - language name is simple ASCII, no escapes
                    let inner = Span::new(span.start + 1, span.end - 1);
                    config.language = Some(self.ast.text(inner).to_string());
                }
                "extras" => config.extras = self.parse_expr_array()?,
                "externals" => config.externals = self.parse_expr_array()?,
                "supertypes" => config.supertypes = self.parse_ident_span_array()?,
                "inline" => config.inline = self.parse_ident_span_array()?,
                "conflicts" => config.conflicts = self.parse_conflicts()?,
                "precedences" => config.precedences = self.parse_precedences()?,
                "word" => config.word = Some(self.expect_ident()?),
                "reserved" => config.reserved = self.parse_reserved()?,
                _ => {
                    return Err(ParseError {
                        kind: ParseErrorKind::UnknownGrammarField(key),
                        span: key_span,
                    });
                }
            }
            if !self.at(&TokenKind::RBrace) {
                self.expect(&TokenKind::Comma)?;
            }
        }
    }

    fn parse_expr_array(&mut self) -> Result<NodeList, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let items = self.comma_sep(&TokenKind::RBracket, Self::parse_expr)?;
        self.expect(&TokenKind::RBracket)?;
        Ok(items)
    }

    /// `[ident, ident, ...]` - returns spans of the identifiers
    fn parse_ident_span_array(&mut self) -> Result<Vec<Span>, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let items = self.comma_sep(&TokenKind::RBracket, Self::expect_ident)?;
        self.expect(&TokenKind::RBracket)?;
        Ok(items)
    }

    fn parse_conflicts(&mut self) -> Result<Vec<Vec<Span>>, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let items = self.comma_sep(&TokenKind::RBracket, |this| {
            this.expect(&TokenKind::LBracket)?;
            let group = this.comma_sep(&TokenKind::RBracket, Self::expect_ident)?;
            this.expect(&TokenKind::RBracket)?;
            Ok(group)
        })?;
        self.expect(&TokenKind::RBracket)?;
        Ok(items)
    }

    fn parse_precedences(&mut self) -> Result<Vec<Vec<PrecEntry>>, ParseError> {
        self.expect(&TokenKind::LBracket)?;
        let items = self.comma_sep(&TokenKind::RBracket, |this| {
            this.expect(&TokenKind::LBracket)?;
            let group = this.comma_sep(&TokenKind::RBracket, |this| {
                if let Some(span) = this.eat_kind(&TokenKind::StringLit) {
                    let inner = Span::new(span.start + 1, span.end - 1);
                    Ok(PrecEntry::Name(this.ast.text(inner).to_string()))
                } else if let Some(span) = this.eat_kind(&TokenKind::Ident) {
                    Ok(PrecEntry::Symbol(span))
                } else {
                    Err(this.error(ParseErrorKind::ExpectedStringOrIdent))
                }
            })?;
            this.expect(&TokenKind::RBracket)?;
            Ok(group)
        })?;
        self.expect(&TokenKind::RBracket)?;
        Ok(items)
    }

    fn parse_reserved(&mut self) -> Result<Vec<ReservedWordSet>, ParseError> {
        self.expect(&TokenKind::LBrace)?;
        let items = self.comma_sep(&TokenKind::RBrace, |this| {
            let name = this.expect_ident()?;
            this.expect(&TokenKind::Colon)?;
            let words = this.parse_expr_array()?;
            Ok(ReservedWordSet { name, words })
        })?;
        self.expect(&TokenKind::RBrace)?;
        Ok(items)
    }

    fn parse_rule_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(&TokenKind::KwRule)?;
        let name = self.expect_ident()?;
        self.expect(&TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(&TokenKind::RBrace)?;
        Ok(self.ast.push(Node::Rule { name, body }, start.merge(end)))
    }

    fn parse_let_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(&TokenKind::KwLet)?;
        let name = self.expect_ident()?;
        let ty = if self.eat(&TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&TokenKind::Eq)?;
        let value = self.parse_expr()?;
        let end = self.ast.span(value);
        Ok(self
            .ast
            .push(Node::Let { name, ty, value }, start.merge(end)))
    }

    fn parse_fn_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(&TokenKind::KwFn)?;
        let name = self.expect_ident()?;

        self.expect(&TokenKind::LParen)?;
        let params = self.comma_sep(&TokenKind::RParen, |this| {
            let pname = this.expect_ident()?;
            this.expect(&TokenKind::Colon)?;
            let ty = this.parse_type()?;
            Ok(Param { name: pname, ty })
        })?;
        self.expect(&TokenKind::RParen)?;

        if self.eat(&TokenKind::Arrow).is_none() {
            return Err(self.error(ParseErrorKind::MissingReturnType));
        }
        let return_ty = self.parse_type()?;

        self.expect(&TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(&TokenKind::RBrace)?;

        let fn_idx = self.ast.push_fn(FnConfig {
            name,
            params,
            return_ty,
            body,
        });
        Ok(self.ast.push(Node::Fn(fn_idx), start.merge(end)))
    }

    fn parse_type(&mut self) -> Result<NodeId, ParseError> {
        let start = self.span();

        if let Some(id_span) = self.eat_kind(&TokenKind::Ident) {
            let id = self.ast.text(id_span).to_string();
            return match id.as_str() {
                "rule" => Ok(self.ast.push(Node::TypeRule, id_span)),
                "str" => Ok(self.ast.push(Node::TypeStr, id_span)),
                "int" => Ok(self.ast.push(Node::TypeInt, id_span)),
                "list" => {
                    self.expect(&TokenKind::Lt)?;
                    let inner = self.parse_type()?;
                    let end = self.expect(&TokenKind::Gt)?;
                    Ok(self.ast.push(Node::TypeList(inner), start.merge(end)))
                }
                _ => Err(ParseError {
                    kind: ParseErrorKind::UnknownType(id),
                    span: id_span,
                }),
            };
        }

        if self.eat(&TokenKind::LParen).is_some() {
            let elems = self.comma_sep(&TokenKind::RParen, Self::parse_type)?;
            let end = self.expect(&TokenKind::RParen)?;
            let range = self.ast.push_children(elems);
            return Ok(self.ast.push(Node::TypeTuple(range), start.merge(end)));
        }

        if self.eat(&TokenKind::KwRule).is_some() {
            return Ok(self.ast.push(Node::TypeRule, start));
        }

        Err(self.error(ParseErrorKind::ExpectedType))
    }

    fn parse_expr(&mut self) -> Result<NodeId, ParseError> {
        self.parse_primary()
    }

    fn parse_primary(&mut self) -> Result<NodeId, ParseError> {
        let start = self.span();
        match &self.tokens[self.pos].kind {
            TokenKind::KwSeq => self.parse_variadic_combinator(start, "seq"),
            TokenKind::KwChoice => self.parse_variadic_combinator(start, "choice"),
            TokenKind::KwRepeat => self.parse_unary_wrapper(start, Node::Repeat),
            TokenKind::KwRepeat1 => self.parse_unary_wrapper(start, Node::Repeat1),
            TokenKind::KwOptional => self.parse_unary_wrapper(start, Node::Optional),
            TokenKind::KwBlank => {
                self.pos += 1;
                self.expect(&TokenKind::LParen)?;
                let end = self.expect(&TokenKind::RParen)?;
                Ok(self.ast.push(Node::Blank, start.merge(end)))
            }
            TokenKind::KwField => self.parse_field_combinator(start),
            TokenKind::KwAlias => self.parse_alias_combinator(start),
            TokenKind::KwToken => self.parse_unary_wrapper(start, Node::Token),
            TokenKind::KwTokenImmediate => self.parse_unary_wrapper(start, Node::TokenImmediate),
            TokenKind::KwPrec => self.parse_prec_combinator(start),
            TokenKind::KwReserved => self.parse_reserved_combinator(start),
            TokenKind::KwConcat => self.parse_concat(start),
            TokenKind::KwRegex => self.parse_dyn_regex(start),
            TokenKind::KwFor => self.parse_for_expr(start),
            TokenKind::Ident => self.parse_ident_expr(start),
            TokenKind::StringLit => {
                let span = self.eat_kind(&TokenKind::StringLit).unwrap();
                Ok(self.ast.push(Node::StringLit, span))
            }
            TokenKind::IntLit(_) => {
                if let TokenKind::IntLit(n) = self.tokens[self.pos].kind {
                    let span = self.span();
                    self.pos += 1;
                    Ok(self.ast.push(Node::IntLit(n), span))
                } else {
                    unreachable!()
                }
            }
            TokenKind::RawStringLit { hash_count } => {
                let hash_count = *hash_count;
                let span = self.span();
                self.pos += 1;
                Ok(self.ast.push(Node::RawStringLit { hash_count }, span))
            }
            TokenKind::Minus => {
                self.pos += 1;
                let inner = self.parse_expr()?;
                let end = self.ast.span(inner);
                Ok(self.ast.push(Node::Neg(inner), start.merge(end)))
            }
            TokenKind::LBracket => self.parse_list_expr(start),
            TokenKind::LParen => self.parse_tuple_expr(start),
            TokenKind::LBrace => self.parse_object_expr(start),
            _ => Err(self.error(ParseErrorKind::ExpectedExpression)),
        }
    }

    fn parse_variadic_combinator(&mut self, start: Span, name: &str) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let args = self.comma_sep(&TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(&TokenKind::RParen)?;
        let range = self.ast.push_children(args);
        let node = match name {
            "seq" => Node::Seq(range),
            "choice" => Node::Choice(range),
            _ => unreachable!(),
        };
        Ok(self.ast.push(node, start.merge(end)))
    }

    fn parse_unary_wrapper(
        &mut self,
        start: Span,
        make_node: fn(NodeId) -> Node,
    ) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let inner = self.parse_expr()?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(self.ast.push(make_node(inner), start.merge(end)))
    }

    fn parse_field_combinator(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let name = self.expect_name()?;
        self.expect(&TokenKind::Comma)?;
        let content = self.parse_expr()?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(self
            .ast
            .push(Node::Field { name, content }, start.merge(end)))
    }

    fn parse_alias_combinator(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let content = self.parse_expr()?;
        self.expect(&TokenKind::Comma)?;
        let target = self.parse_expr()?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(self
            .ast
            .push(Node::Alias { content, target }, start.merge(end)))
    }

    fn parse_concat(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let args = self.comma_sep(&TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(&TokenKind::RParen)?;
        let range = self.ast.push_children(args);
        Ok(self.ast.push(Node::Concat(range), start.merge(end)))
    }

    fn parse_dyn_regex(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let pattern = self.parse_expr()?;
        let flags = if self.eat(&TokenKind::Comma).is_some() {
            Some(self.parse_expr()?)
        } else {
            None
        };
        let end = self.expect(&TokenKind::RParen)?;
        Ok(self
            .ast
            .push(Node::DynRegex { pattern, flags }, start.merge(end)))
    }

    fn parse_prec_combinator(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        let variant = if self.eat(&TokenKind::Dot).is_some() {
            let id_span = self.expect_ident()?;
            let id = self.ast.text(id_span);
            match id {
                "left" => PrecVariant::Left,
                "right" => PrecVariant::Right,
                "dynamic" => PrecVariant::Dynamic,
                _ => {
                    return Err(ParseError {
                        kind: ParseErrorKind::ExpectedPrecVariant(id.to_string()),
                        span: id_span,
                    });
                }
            }
        } else {
            PrecVariant::Default
        };

        self.expect(&TokenKind::LParen)?;
        let value = self.parse_expr()?;
        self.expect(&TokenKind::Comma)?;
        let content = self.parse_expr()?;
        let end = self.expect(&TokenKind::RParen)?;
        let span = start.merge(end);

        let node = match variant {
            PrecVariant::Default => Node::Prec { value, content },
            PrecVariant::Left => Node::PrecLeft { value, content },
            PrecVariant::Right => Node::PrecRight { value, content },
            PrecVariant::Dynamic => Node::PrecDynamic { value, content },
        };
        Ok(self.ast.push(node, span))
    }

    fn parse_reserved_combinator(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let context_span = self.expect_string()?;
        let context = Span::new(context_span.start + 1, context_span.end - 1);
        self.expect(&TokenKind::Comma)?;
        let content = self.parse_expr()?;
        let end = self.expect(&TokenKind::RParen)?;
        Ok(self
            .ast
            .push(Node::Reserved { context, content }, start.merge(end)))
    }

    fn parse_for_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        self.expect(&TokenKind::LParen)?;
        let bindings = self.comma_sep(&TokenKind::RParen, |this| {
            let name = this.expect_ident()?;
            this.expect(&TokenKind::Colon)?;
            let ty = this.parse_type()?;
            Ok((name, ty))
        })?;
        self.expect(&TokenKind::RParen)?;
        self.expect(&TokenKind::KwIn)?;
        let iterable = self.parse_expr()?;
        self.expect(&TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(&TokenKind::RBrace)?;
        let for_idx = self.ast.push_for(ForConfig {
            bindings,
            iterable,
            body,
        });
        Ok(self.ast.push(Node::For(for_idx), start.merge(end)))
    }

    fn parse_ident_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        let ident_span = self.expect_ident()?;
        let mut id = self.ast.push(Node::Ident(ident_span), ident_span);

        loop {
            if self.at(&TokenKind::Dot) {
                self.pos += 1;
                let field = self.expect_ident()?;
                let span = start.merge(field);
                id = self.ast.push(Node::FieldAccess { obj: id, field }, span);
            } else if self.at(&TokenKind::LParen) {
                if !matches!(self.ast.node(id), Node::Ident(_)) {
                    return Err(self.error(ParseErrorKind::OnlySimpleNamesCallable));
                }
                self.pos += 1;
                let args = self.comma_sep(&TokenKind::RParen, Self::parse_expr)?;
                let end = self.expect(&TokenKind::RParen)?;
                let args = self.ast.push_children(args);
                id = self.ast.push(
                    Node::Call {
                        name: ident_span,
                        args,
                    },
                    start.merge(end),
                );
                break;
            } else {
                break;
            }
        }
        Ok(id)
    }

    fn parse_list_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        let items = self.comma_sep(&TokenKind::RBracket, Self::parse_expr)?;
        let end = self.expect(&TokenKind::RBracket)?;
        let range = self.ast.push_children(items);
        Ok(self.ast.push(Node::List(range), start.merge(end)))
    }

    fn parse_tuple_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        let items = self.comma_sep(&TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(&TokenKind::RParen)?;
        let range = self.ast.push_children(items);
        Ok(self.ast.push(Node::Tuple(range), start.merge(end)))
    }

    fn parse_object_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.pos += 1;
        let fields = self.comma_sep(&TokenKind::RBrace, |this| {
            let key = this.expect_ident()?;
            this.expect(&TokenKind::Colon)?;
            let value = this.parse_expr()?;
            Ok((key, value))
        })?;
        let end = self.expect(&TokenKind::RBrace)?;
        let range = self.ast.push_object(fields);
        Ok(self.ast.push(Node::Object(range), start.merge(end)))
    }

    /// Parse a comma-separated list of items up to (but not consuming) `close`.
    ///
    /// Allows trailing commas: if a comma is followed by the closing token,
    /// the list ends without attempting to parse another item.
    fn comma_sep<T>(
        &mut self,
        close: &TokenKind,
        mut parse_item: impl FnMut(&mut Self) -> Result<T, ParseError>,
    ) -> Result<Vec<T>, ParseError> {
        let mut items = Vec::new();
        loop {
            if self.at(close) {
                break;
            }
            items.push(parse_item(self)?);
            if self.eat(&TokenKind::Comma).is_none() {
                break;
            }
            // Allow trailing comma before close
            if self.at(close) {
                break;
            }
        }
        Ok(items)
    }
}

/// Which `prec` variant is being parsed: `prec(...)`, `prec.left(...)`, etc.
enum PrecVariant {
    Default,
    Left,
    Right,
    Dynamic,
}

/// An error encountered during parsing, with the source span of the problem.
#[derive(Debug, Serialize, Error)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
}

/// The specific kind of parse error.
#[derive(Debug, Serialize)]
pub enum ParseErrorKind {
    ExpectedToken { expected: TokenKind, got: TokenKind },
    ExpectedIdent,
    ExpectedString,
    ExpectedName,
    ExpectedExpression,
    ExpectedType,
    ExpectedItem,
    UnknownType(String),
    UnknownGrammarField(String),
    MissingReturnType,
    ExpectedPrecVariant(String),
    OnlySimpleNamesCallable,
    ExpectedStringOrIdent,
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            ParseErrorKind::ExpectedToken { expected, got } => {
                write!(f, "expected {expected}, got {got}")
            }
            ParseErrorKind::ExpectedIdent => write!(f, "expected identifier"),
            ParseErrorKind::ExpectedString => write!(f, "expected string literal"),
            ParseErrorKind::ExpectedName => write!(f, "expected name"),
            ParseErrorKind::ExpectedExpression => write!(f, "expected expression"),
            ParseErrorKind::ExpectedType => write!(f, "expected type"),
            ParseErrorKind::ExpectedItem => {
                write!(f, "expected 'grammar', 'rule', 'let', or 'fn'")
            }
            ParseErrorKind::UnknownType(name) => write!(f, "unknown type '{name}'"),
            ParseErrorKind::UnknownGrammarField(name) => {
                write!(f, "unknown grammar field '{name}'")
            }
            ParseErrorKind::MissingReturnType => {
                write!(f, "function must have a return type annotation (-> type)")
            }
            ParseErrorKind::ExpectedPrecVariant(found) => {
                write!(f, "expected 'left', 'right', or 'dynamic', found '{found}'")
            }
            ParseErrorKind::OnlySimpleNamesCallable => {
                write!(f, "only simple names can be called")
            }
            ParseErrorKind::ExpectedStringOrIdent => {
                write!(f, "expected string or identifier")
            }
        }
    }
}
