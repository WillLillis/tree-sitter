//! Recursive descent parser for the native grammar DSL.
//!
//! Consumes tokens from the lexer and builds an arena-based [`Ast`].
//! Trailing commas are allowed everywhere. All identifiers start as
//! `Node::Ident` and are resolved to `RuleRef`/`VarRef` in a later pass.

use serde::Serialize;
use thiserror::Error;

use std::path::Path;

use super::{
    ast::{
        Ast, ChildRange, FnConfig, ForConfig, GrammarConfig, Node, NodeId, Note, NoteMessage,
        Param, Span,
    },
    lexer::{Token, TokenKind},
};

const MAX_PARSE_DEPTH: u16 = 256;

pub struct Parser<'src, 'tok, 'path> {
    tokens: &'tok [Token],
    pos: usize,
    grammar_path: &'path Path,
    pub ast: Ast<'src>,
    scratch: Vec<NodeId>,
    depth: u16,
}

impl<'src, 'tok, 'path> Parser<'src, 'tok, 'path> {
    #[must_use]
    pub fn new(tokens: &'tok [Token], source: &'src str, grammar_path: &'path Path) -> Self {
        Self {
            tokens,
            pos: 0,
            grammar_path,
            ast: Ast::new(source),
            scratch: Vec::new(),
            depth: 0,
        }
    }

    pub fn parse(mut self) -> Result<Ast<'src>, ParseError> {
        self.skip_comments();
        while !self.at_eof() {
            let id = self.parse_item()?;
            self.ast.root_items.push(id);
            self.skip_comments();
        }
        Ok(self.ast)
    }

    fn skip_comments(&mut self) {
        while self.tokens[self.pos].kind == TokenKind::Comment {
            self.pos += 1;
        }
    }

    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }
    fn at_eof(&self) -> bool {
        self.tokens[self.pos].kind == TokenKind::Eof
    }
    fn at(&self, kind: TokenKind) -> bool {
        self.tokens[self.pos].kind == kind
    }

    /// Advance past the current token, skipping any comments.
    /// Safe without bounds check: the token stream always ends with `Eof`.
    fn advance_pos(&mut self) {
        self.pos += 1;
        while self.tokens[self.pos].kind == TokenKind::Comment {
            self.pos += 1;
        }
    }

    fn eat(&mut self, kind: TokenKind) -> Option<Span> {
        if self.at(kind) {
            let s = self.span();
            self.advance_pos();
            Some(s)
        } else {
            None
        }
    }

    fn expect(&mut self, kind: TokenKind) -> Result<Span, ParseError> {
        self.eat(kind).ok_or_else(|| {
            self.error(ParseErrorKind::ExpectedToken {
                expected: kind,
                got: self.tokens[self.pos].kind,
            })
        })
    }

    fn expect_ident_node(&mut self) -> Result<NodeId, ParseError> {
        let span = self.expect_ident()?;
        Ok(self.ast.push(Node::Ident, span))
    }

    fn expect_ident(&mut self) -> Result<Span, ParseError> {
        self.eat(TokenKind::Ident).ok_or_else(|| {
            if self.tokens[self.pos].kind.is_keyword() {
                self.error(ParseErrorKind::KeywordAsIdent)
            } else {
                self.error(ParseErrorKind::ExpectedIdent)
            }
        })
    }

    fn expect_string(&mut self) -> Result<Span, ParseError> {
        self.eat(TokenKind::StringLit)
            .ok_or_else(|| self.error(ParseErrorKind::ExpectedString))
    }

    /// Like `expect_ident`, but also accepts keywords (for field names, config keys).
    fn expect_name(&mut self) -> Result<Span, ParseError> {
        let span = self.span();
        let kind = self.tokens[self.pos].kind;
        if kind == TokenKind::Ident || kind.is_keyword() {
            self.advance_pos();
            Ok(span)
        } else {
            Err(self.error(ParseErrorKind::ExpectedName))
        }
    }

    fn error(&self, kind: ParseErrorKind) -> ParseError {
        ParseError {
            kind,
            span: self.span(),
            note: None,
        }
    }

    fn err_arg_count(
        &self,
        name: &'static str,
        expected: u8,
        got: usize,
        start: Span,
    ) -> ParseError {
        ParseError {
            kind: ParseErrorKind::WrongArgumentCount {
                name,
                expected,
                got,
            },
            span: start.merge(self.span()),
            note: None,
        }
    }

    /// After parsing the expected number of args, check that the next token
    /// is `)` and not `,` (which would indicate extra arguments).
    fn expect_close_args(
        &mut self,
        name: &'static str,
        expected: u8,
        start: Span,
    ) -> Result<(), ParseError> {
        if self.at(TokenKind::Comma) {
            let mut got = expected as usize;
            // account for trailing comma with no extra args
            self.advance_pos();
            if self.at(TokenKind::RParen) {
                return Ok(());
            }
            self.parse_expr()?;
            got += 1;
            // if there are extra args, count them
            while self.eat(TokenKind::Comma).is_some() {
                if self.at(TokenKind::RParen) {
                    break;
                }
                self.parse_expr()?;
                got += 1;
            }
            return Err(self.err_arg_count(name, expected, got, start));
        }
        Ok(())
    }

    // -- Top-level items --

    fn parse_item(&mut self) -> Result<NodeId, ParseError> {
        match &self.tokens[self.pos].kind {
            TokenKind::KwGrammar => self.parse_grammar_block(),
            TokenKind::KwRule => self.parse_rule_def(),
            TokenKind::KwOverride => self.parse_override_rule_def(),
            TokenKind::KwLet => self.parse_let_def(),
            TokenKind::KwFn => self.parse_fn_def(),
            TokenKind::KwPrint => self.parse_print_item(),
            _ => Err(self.error(ParseErrorKind::ExpectedItem)),
        }
    }

    fn parse_print_item(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::KwPrint)?;
        self.expect(TokenKind::LParen)?;
        let arg = self.parse_expr()?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(Node::Print(arg), start.merge(end)))
    }

    fn parse_grammar_block(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::KwGrammar)?;
        if self.ast.context.grammar_config.is_some() {
            let first_span = self
                .ast
                .root_items
                .iter()
                .find(|&&id| matches!(self.ast.node(id), Node::Grammar))
                .map(|&id| self.ast.span(id))
                .unwrap();
            return Err(ParseError {
                kind: ParseErrorKind::DuplicateGrammarBlock,
                span: start,
                note: Some(Note {
                    message: NoteMessage::FirstDefinedHere,
                    span: first_span,
                    path: self.grammar_path.to_path_buf(),
                    source: self.ast.source().to_string(),
                }),
            });
        }
        self.expect(TokenKind::LBrace)?;
        let mut config = GrammarConfig::default();
        let mut seen = [None::<Span>; ConfigField::COUNT];
        loop {
            if let Some(end) = self.eat(TokenKind::RBrace) {
                self.ast.context.grammar_config = Some(config);
                return Ok(self.ast.push(Node::Grammar, start.merge(end)));
            }
            let key_span = self.expect_name()?;
            self.expect(TokenKind::Colon)?;
            let key = self.ast.text(key_span);
            let field = ConfigField::from_str(key).ok_or_else(|| ParseError {
                kind: ParseErrorKind::UnknownGrammarField(key.to_string()),
                span: key_span,
                note: None,
            })?;
            if let Some(first_span) = seen[field as usize] {
                return Err(ParseError {
                    kind: ParseErrorKind::DuplicateGrammarField(key.to_string()),
                    span: key_span,
                    note: Some(Note {
                        message: NoteMessage::FirstDefinedHere,
                        span: first_span,
                        path: self.grammar_path.to_path_buf(),
                        source: self.ast.source().to_string(),
                    }),
                });
            }
            seen[field as usize] = Some(key_span);
            match field {
                ConfigField::Language => {
                    let s = self.expect_string()?;
                    let inner = Span::new(s.start + 1, s.end - 1);
                    config.language = Some(self.ast.text(inner).to_string());
                }
                ConfigField::Inherits => config.inherits = Some(self.parse_expr()?),
                ConfigField::Extras => config.extras = Some(self.parse_expr()?),
                ConfigField::Externals => config.externals = Some(self.parse_expr()?),
                ConfigField::Supertypes => config.supertypes = Some(self.parse_expr()?),
                ConfigField::Inline => config.inline = Some(self.parse_expr()?),
                ConfigField::Word => config.word = Some(self.parse_expr()?),
                ConfigField::Conflicts => config.conflicts = Some(self.parse_expr()?),
                ConfigField::Precedences => config.precedences = Some(self.parse_expr()?),
                ConfigField::Reserved => config.reserved = Some(self.parse_expr()?),
                ConfigField::_Count => unreachable!(),
            }
            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }
    }

    fn parse_rule_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::KwRule)?;
        let name = self.expect_ident_node()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        Ok(self.ast.push(Node::Rule { name, body }, start.merge(end)))
    }

    fn parse_override_rule_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::KwOverride)?;
        self.expect(TokenKind::KwRule)?;
        let name = self.expect_ident_node()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        Ok(self
            .ast
            .push(Node::OverrideRule { name, body }, start.merge(end)))
    }

    fn parse_let_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::KwLet)?;
        let name = self.expect_ident_node()?;
        let ty = if self.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        Ok(self.ast.push(
            Node::Let { name, ty, value },
            start.merge(self.ast.span(value)),
        ))
    }

    fn parse_fn_def(&mut self) -> Result<NodeId, ParseError> {
        let start = self.expect(TokenKind::KwFn)?;
        let name = self.expect_ident_node()?;
        self.expect(TokenKind::LParen)?;
        let params = self.comma_sep(TokenKind::RParen, |this| {
            let pname = this.expect_ident_node()?;
            this.expect(TokenKind::Colon)?;
            Ok(Param {
                name: pname,
                ty: this.parse_type()?,
            })
        })?;
        self.expect(TokenKind::RParen)?;
        if self.eat(TokenKind::Arrow).is_none() {
            return Err(self.error(ParseErrorKind::MissingReturnType));
        }
        let return_ty = self.parse_type()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        let fn_idx = self.ast.push_fn(FnConfig {
            name,
            params,
            return_ty,
            body,
        });
        Ok(self.ast.push(Node::Fn(fn_idx), start.merge(end)))
    }

    fn parse_type(&mut self) -> Result<NodeId, ParseError> {
        if let Some(id_span) = self.eat(TokenKind::Ident) {
            return match self.ast.text(id_span) {
                "rule_t" => Ok(self.ast.push(Node::TypeRule, id_span)),
                "str_t" => Ok(self.ast.push(Node::TypeStr, id_span)),
                "int_t" => Ok(self.ast.push(Node::TypeInt, id_span)),
                "list_rule_t" => Ok(self.ast.push(Node::TypeListRule, id_span)),
                "list_str_t" => Ok(self.ast.push(Node::TypeListStr, id_span)),
                "list_int_t" => Ok(self.ast.push(Node::TypeListInt, id_span)),
                "list_list_rule_t" => Ok(self.ast.push(Node::TypeListListRule, id_span)),
                "list_list_str_t" => Ok(self.ast.push(Node::TypeListListStr, id_span)),
                "list_list_int_t" => Ok(self.ast.push(Node::TypeListListInt, id_span)),
                "void_t" => Ok(self.ast.push(Node::TypeVoid, id_span)),
                _ => Err(ParseError {
                    kind: ParseErrorKind::UnknownType(self.ast.text(id_span).to_string()),
                    span: id_span,
                    note: None,
                }),
            };
        }
        Err(self.error(ParseErrorKind::ExpectedType))
    }

    // -- Expressions --

    fn parse_expr(&mut self) -> Result<NodeId, ParseError> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(self.error(ParseErrorKind::NestingTooDeep));
        }
        let result = self.parse_primary();
        self.depth -= 1;
        result
    }

    fn parse_primary(&mut self) -> Result<NodeId, ParseError> {
        let start = self.span();
        match &self.tokens[self.pos].kind {
            TokenKind::KwSeq => self.parse_variadic(start, Node::Seq),
            TokenKind::KwChoice => self.parse_variadic(start, Node::Choice),
            TokenKind::KwRepeat => self.parse_unary(start, "repeat", Node::Repeat),
            TokenKind::KwRepeat1 => self.parse_unary(start, "repeat1", Node::Repeat1),
            TokenKind::KwOptional => self.parse_unary(start, "optional", Node::Optional),
            TokenKind::KwBlank => {
                self.advance_pos();
                self.expect(TokenKind::LParen)?;
                if !self.at(TokenKind::RParen) {
                    let mut got = 0usize;
                    loop {
                        self.parse_expr()?;
                        got += 1;
                        if self.eat(TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    return Err(self.err_arg_count("blank", 0, got, start));
                }
                let end = self.expect(TokenKind::RParen)?;
                Ok(self.ast.push(Node::Blank, start.merge(end)))
            }
            TokenKind::KwField => self.parse_field(start),
            TokenKind::KwAlias => self.parse_alias(start),
            TokenKind::KwToken => self.parse_unary(start, "token", Node::Token),
            TokenKind::KwTokenImmediate => {
                self.parse_unary(start, "token_immediate", Node::TokenImmediate)
            }
            TokenKind::KwPrec => self.parse_prec(start, PrecVariant::Default),
            TokenKind::KwPrecLeft => self.parse_prec(start, PrecVariant::Left),
            TokenKind::KwPrecRight => self.parse_prec(start, PrecVariant::Right),
            TokenKind::KwPrecDynamic => self.parse_prec(start, PrecVariant::Dynamic),
            TokenKind::KwReserved => self.parse_reserved_expr(start),
            TokenKind::KwConcat => self.parse_variadic(start, Node::Concat),
            TokenKind::KwRegexp => self.parse_regexp(start),
            TokenKind::KwInherit => self.parse_inherit(start),
            TokenKind::KwAppend => self.parse_append(start),
            TokenKind::KwFor => self.parse_for(start),
            TokenKind::Ident => self.parse_ident_expr(start),
            TokenKind::StringLit => {
                self.advance_pos();
                Ok(self
                    .ast
                    .push(Node::StringLit, Span::new(start.start + 1, start.end - 1)))
            }
            TokenKind::IntLit(n) => {
                let n = *n;
                self.advance_pos();
                Ok(self.ast.push(Node::IntLit(n), start))
            }
            TokenKind::RawStringLit { hash_count } => {
                let h = *hash_count;
                self.advance_pos();
                Ok(self.ast.push(Node::RawStringLit { hash_count: h }, start))
            }
            TokenKind::Minus => {
                self.advance_pos();
                let inner = self.parse_expr()?;
                Ok(self
                    .ast
                    .push(Node::Neg(inner), start.merge(self.ast.span(inner))))
            }
            TokenKind::LBracket => self.parse_list(start),
            TokenKind::LParen => self.parse_tuple(start),
            TokenKind::LBrace => self.parse_object(start),
            _ => Err(self.error(ParseErrorKind::ExpectedExpression)),
        }
    }

    fn parse_variadic(
        &mut self,
        start: Span,
        make: fn(ChildRange) -> Node,
    ) -> Result<NodeId, ParseError> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        let range = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(make(range), start.merge(end)))
    }

    fn parse_unary(
        &mut self,
        start: Span,
        name: &'static str,
        make: fn(NodeId) -> Node,
    ) -> Result<NodeId, ParseError> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(name, 1, 0, start));
        }
        let inner = self.parse_expr()?;
        self.expect_close_args(name, 1, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(make(inner), start.merge(end)))
    }

    /// Parse a two-argument builtin: `name(first, second)`.
    fn parse_binary(
        &mut self,
        start: Span,
        name: &'static str,
        parse_first: fn(&mut Self) -> Result<NodeId, ParseError>,
    ) -> Result<(NodeId, NodeId, Span), ParseError> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(name, 2, 0, start));
        }
        let first = parse_first(self)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(name, 2, 1, start));
        }
        self.expect(TokenKind::Comma)?;
        let second = self.parse_expr()?;
        self.expect_close_args(name, 2, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok((first, second, end))
    }

    fn parse_field(&mut self, start: Span) -> Result<NodeId, ParseError> {
        let (name, content, end) = self.parse_binary(start, "field", |this| {
            let s = this.expect_name()?;
            Ok(this.ast.push(Node::Ident, s))
        })?;
        Ok(self
            .ast
            .push(Node::Field { name, content }, start.merge(end)))
    }

    fn parse_alias(&mut self, start: Span) -> Result<NodeId, ParseError> {
        let (content, target, end) = self.parse_binary(start, "alias", Self::parse_expr)?;
        Ok(self
            .ast
            .push(Node::Alias { content, target }, start.merge(end)))
    }

    fn parse_prec(&mut self, start: Span, variant: PrecVariant) -> Result<NodeId, ParseError> {
        let name = match variant {
            PrecVariant::Default => "prec",
            PrecVariant::Left => "prec_left",
            PrecVariant::Right => "prec_right",
            PrecVariant::Dynamic => "prec_dynamic",
        };
        let (value, content, end) = self.parse_binary(start, name, Self::parse_expr)?;
        let node = match variant {
            PrecVariant::Default => Node::Prec { value, content },
            PrecVariant::Left => Node::PrecLeft { value, content },
            PrecVariant::Right => Node::PrecRight { value, content },
            PrecVariant::Dynamic => Node::PrecDynamic { value, content },
        };
        Ok(self.ast.push(node, start.merge(end)))
    }

    /// Parse `reserved("context", content)` expression (not the config block).
    fn parse_reserved_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        let (context, content, end) = self.parse_binary(start, "reserved", |this| {
            let cs = this.expect_string()?;
            Ok(this
                .ast
                .push(Node::StringLit, Span::new(cs.start + 1, cs.end - 1)))
        })?;
        Ok(self
            .ast
            .push(Node::Reserved { context, content }, start.merge(end)))
    }

    fn parse_regexp(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count("regexp", 1, 0, start));
        }
        let pattern = self.parse_expr()?;
        let flags = if self.eat(TokenKind::Comma).is_some() && !self.at(TokenKind::RParen) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect_close_args("regexp", if flags.is_some() { 2 } else { 1 }, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self
            .ast
            .push(Node::DynRegex { pattern, flags }, start.merge(end)))
    }

    fn parse_inherit(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count("inherit", 1, 0, start));
        }
        let path_span = self.expect_string()?;
        let path = self.ast.push(
            Node::StringLit,
            Span::new(path_span.start + 1, path_span.end - 1),
        );
        self.expect_close_args("inherit", 1, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(Node::Inherit { path }, start.merge(end)))
    }

    fn parse_append(&mut self, start: Span) -> Result<NodeId, ParseError> {
        let (left, right, end) = self.parse_binary(start, "append", Self::parse_expr)?;
        Ok(self
            .ast
            .push(Node::Append { left, right }, start.merge(end)))
    }

    fn parse_for(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        let bindings = self.comma_sep(TokenKind::RParen, |this| {
            let name = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok((name, this.parse_type()?))
        })?;
        self.expect(TokenKind::RParen)?;
        self.expect(TokenKind::KwIn)?;
        let iterable = self.parse_expr()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        let for_idx = self.ast.push_for(ForConfig {
            bindings,
            iterable,
            body,
        });
        Ok(self.ast.push(Node::For(for_idx), start.merge(end)))
    }

    fn parse_ident_expr(&mut self, start: Span) -> Result<NodeId, ParseError> {
        let name_id = self.expect_ident_node()?;
        let mut id = name_id;
        loop {
            if self.at(TokenKind::Dot) {
                self.advance_pos();
                let field_span = self.expect_name()?;
                let field = self.ast.push(Node::Ident, field_span);
                id = self.ast.push(
                    Node::FieldAccess { obj: id, field },
                    start.merge(self.ast.span(field)),
                );
            } else if self.at(TokenKind::ColonColon) {
                self.advance_pos();
                let rule = self.expect_ident_node()?;
                id = self.ast.push(
                    Node::RuleInline { obj: id, rule },
                    start.merge(self.ast.span(rule)),
                );
            } else if self.at(TokenKind::LParen) {
                if !matches!(self.ast.node(id), Node::Ident) {
                    return Err(self.error(ParseErrorKind::ExpectedFunctionName));
                }
                self.advance_pos();
                let args = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
                let end = self.expect(TokenKind::RParen)?;
                id = self.ast.push(
                    Node::Call {
                        name: name_id,
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

    fn parse_list(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.advance_pos();
        let range = self.comma_sep_children(TokenKind::RBracket, Self::parse_expr)?;
        let end = self.expect(TokenKind::RBracket)?;
        Ok(self.ast.push(Node::List(range), start.merge(end)))
    }

    fn parse_tuple(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.advance_pos();
        let range = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(Node::Tuple(range), start.merge(end)))
    }

    fn parse_object(&mut self, start: Span) -> Result<NodeId, ParseError> {
        self.advance_pos();
        let fields = self.comma_sep(TokenKind::RBrace, |this| {
            let key = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok((key, this.parse_expr()?))
        })?;
        // Check for duplicate keys
        for (i, &(span_a, _)) in fields.iter().enumerate() {
            let key_a = self.ast.text(span_a);
            for &(span_b, _) in &fields[..i] {
                if self.ast.text(span_b) == key_a {
                    return Err(ParseError {
                        kind: ParseErrorKind::DuplicateObjectKey(key_a.to_string()),
                        span: span_a,
                        note: Some(Note {
                            message: NoteMessage::FirstDefinedHere,
                            span: span_b,
                            path: self.grammar_path.to_path_buf(),
                            source: self.ast.source().to_string(),
                        }),
                    });
                }
            }
        }
        let end = self.expect(TokenKind::RBrace)?;
        let range = self
            .ast
            .push_object(fields)
            .map_err(|_| self.error(ParseErrorKind::TooManyChildren))?;
        Ok(self.ast.push(Node::Object(range), start.merge(end)))
    }

    /// Parse comma-separated children directly into a [`ChildRange`].
    ///
    /// Uses a scratch buffer as a stack to avoid per-call heap allocation.
    /// Nested calls push above the parent's items and truncate back, so
    /// interleaving is impossible.
    fn comma_sep_children(
        &mut self,
        close: TokenKind,
        mut parse_item: impl FnMut(&mut Self) -> Result<NodeId, ParseError>,
    ) -> Result<ChildRange, ParseError> {
        let saved = self.scratch.len();
        loop {
            if self.at(close) {
                break;
            }
            let id = parse_item(self)?;
            self.scratch.push(id);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
            if self.at(close) {
                break;
            }
        }
        let range = self
            .ast
            .push_children(&self.scratch[saved..])
            .map_err(|_| self.error(ParseErrorKind::TooManyChildren))?;
        self.scratch.truncate(saved);
        Ok(range)
    }

    fn comma_sep<T>(
        &mut self,
        close: TokenKind,
        mut parse_item: impl FnMut(&mut Self) -> Result<T, ParseError>,
    ) -> Result<Vec<T>, ParseError> {
        let mut items = Vec::new();
        loop {
            if self.at(close) {
                break;
            }
            items.push(parse_item(self)?);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
            if self.at(close) {
                break;
            }
        }
        Ok(items)
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum PrecVariant {
    Default,
    Left,
    Right,
    Dynamic,
}

#[derive(Clone, Copy)]
enum ConfigField {
    Language,
    Inherits,
    Extras,
    Externals,
    Supertypes,
    Inline,
    Word,
    Conflicts,
    Precedences,
    Reserved,
    // Sentinel - must be last. Used to derive COUNT.
    _Count,
}

impl ConfigField {
    const COUNT: usize = Self::_Count as usize;

    fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "language" => Self::Language,
            "inherits" => Self::Inherits,
            "extras" => Self::Extras,
            "externals" => Self::Externals,
            "supertypes" => Self::Supertypes,
            "inline" => Self::Inline,
            "word" => Self::Word,
            "conflicts" => Self::Conflicts,
            "precedences" => Self::Precedences,
            "reserved" => Self::Reserved,
            _ => return None,
        })
    }
}

#[derive(Debug, Serialize, Error)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
    pub note: Option<Note>,
}

#[derive(Debug, PartialEq, Eq, Serialize)]
pub enum ParseErrorKind {
    ExpectedToken {
        expected: TokenKind,
        got: TokenKind,
    },
    ExpectedIdent,
    ExpectedString,
    ExpectedName,
    ExpectedExpression,
    ExpectedType,
    ExpectedItem,
    UnknownType(String),
    UnknownGrammarField(String),
    DuplicateGrammarField(String),
    DuplicateObjectKey(String),
    MissingReturnType,
    ExpectedFunctionName,
    KeywordAsIdent,
    DuplicateGrammarBlock,
    NestingTooDeep,
    TooManyChildren,
    WrongArgumentCount {
        name: &'static str,
        expected: u8,
        got: usize,
    },
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
                write!(f, "expected 'grammar', 'rule', 'let', 'fn', or 'print'")
            }
            ParseErrorKind::UnknownType(n) => write!(f, "unknown type '{n}'"),
            ParseErrorKind::UnknownGrammarField(n) => write!(f, "unknown grammar field '{n}'"),
            ParseErrorKind::DuplicateGrammarField(n) => {
                write!(f, "duplicate grammar field '{n}'")
            }
            ParseErrorKind::DuplicateObjectKey(n) => write!(f, "duplicate object key '{n}'"),
            ParseErrorKind::MissingReturnType => {
                write!(f, "function must have a return type (-> type)")
            }
            ParseErrorKind::ExpectedFunctionName => {
                write!(f, "only identifiers can be used as function names")
            }
            ParseErrorKind::KeywordAsIdent => {
                write!(f, "keywords cannot be used as identifiers")
            }
            ParseErrorKind::DuplicateGrammarBlock => write!(f, "only one grammar block is allowed"),
            ParseErrorKind::NestingTooDeep => {
                write!(f, "expression nesting too deep (maximum {MAX_PARSE_DEPTH})")
            }
            ParseErrorKind::TooManyChildren => {
                write!(f, "too many elements (maximum {})", u16::MAX)
            }
            ParseErrorKind::WrongArgumentCount {
                name,
                expected,
                got,
            } => match expected {
                0 => write!(f, "'{name}' takes no arguments, got {got}"),
                1 => write!(f, "'{name}' takes 1 argument, got {got}"),
                _ => write!(f, "'{name}' takes {expected} arguments, got {got}"),
            },
        }
    }
}
