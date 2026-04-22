//! Recursive descent parser for the native grammar DSL.
//!
//! Consumes tokens from the lexer and builds an arena-based [`Ast`].
//! Trailing commas are allowed everywhere. All identifiers start as
//! `Node::Ident` and are resolved to `RuleRef`/`VarRef` in a later pass.

use serde::Serialize;
use thiserror::Error;

use std::path::Path;

use crate::nativedsl::InnerTy;

use super::typecheck::Ty;
use super::{
    Note, NoteMessage,
    ast::{Ast, ChildRange, FnConfig, ForConfig, GrammarConfig, Node, NodeId, Param, Span},
    lexer::{Token, TokenKind},
};

const MAX_PARSE_DEPTH: u16 = 256;

pub struct Parser<'tok, 'path> {
    tokens: &'tok [Token],
    pos: usize,
    grammar_path: &'path Path,
    pub ast: Ast,
    scratch: Vec<NodeId>,
    depth: u16,
}

impl<'tok, 'path> Parser<'tok, 'path> {
    #[must_use]
    pub fn new(tokens: &'tok [Token], source: String, grammar_path: &'path Path) -> Self {
        Self {
            tokens,
            pos: 0,
            grammar_path,
            ast: Ast::new(source),
            scratch: Vec::new(),
            depth: 0,
        }
    }

    pub fn parse(mut self) -> ParseResult<Ast> {
        self.skip_comments();
        while !self.at_eof() {
            let id = self.parse_item()?;
            self.ast.root_items.push(id);
            self.skip_comments();
        }
        Ok(self.ast)
    }

    #[inline]
    fn skip_comments(&mut self) {
        while self.tokens[self.pos].kind == TokenKind::Comment {
            self.pos += 1;
        }
    }

    #[inline]
    fn span(&self) -> Span {
        self.tokens[self.pos].span
    }

    #[inline]
    fn at_eof(&self) -> bool {
        self.tokens[self.pos].kind == TokenKind::Eof
    }

    #[inline]
    fn at(&self, kind: TokenKind) -> bool {
        self.tokens[self.pos].kind == kind
    }

    /// Check whether the next non-comment token after the current one is `kind`.
    fn next_is(&self, kind: TokenKind) -> bool {
        if self.at_eof() {
            return false;
        }
        let mut i = self.pos + 1;
        // SAFETY: token stream is terminated by Eof. Since pos is not at Eof
        // (checked above), pos + 1 is at most the Eof position, and the loop
        // stops when it hits any non-Comment (including Eof).
        while unsafe { self.tokens.get_unchecked(i) }.kind == TokenKind::Comment {
            i += 1;
        }
        unsafe { self.tokens.get_unchecked(i) }.kind == kind
    }

    /// Advance past the current token, skipping any comments.
    /// Safe without bounds check: the token stream always ends with `Eof`.
    #[inline]
    fn advance_pos(&mut self) {
        self.pos += 1;
        while self.tokens[self.pos].kind == TokenKind::Comment {
            self.pos += 1;
        }
    }

    #[inline]
    fn eat(&mut self, kind: TokenKind) -> Option<Span> {
        if self.at(kind) {
            let s = self.span();
            self.advance_pos();
            Some(s)
        } else {
            None
        }
    }

    #[inline]
    fn expect(&mut self, kind: TokenKind) -> ParseResult<Span> {
        self.eat(kind).ok_or_else(|| {
            self.error(ParseErrorKind::ExpectedToken {
                expected: kind,
                got: self.tokens[self.pos].kind,
            })
        })
    }

    fn expect_ident_node(&mut self) -> ParseResult<NodeId> {
        let span = self.expect_ident()?;
        Ok(self.ast.push(Node::Ident, span))
    }

    /// Accept an identifier or a keyword used as an identifier.
    /// Keywords are contextual: `seq(...)` is a builtin call, but `seq` as a
    /// bare name (e.g. a rule reference or binding name) is a valid identifier.
    fn expect_ident(&mut self) -> ParseResult<Span> {
        self.expect_ident_or_kw(ParseErrorKind::ExpectedIdent)
    }

    fn expect_string(&mut self) -> ParseResult<Span> {
        self.eat(TokenKind::StringLit)
            .ok_or_else(|| self.error(ParseErrorKind::ExpectedString))
    }

    /// Accept a name token (for object keys, config keys, field access members).
    fn expect_name(&mut self) -> ParseResult<Span> {
        self.expect_ident_or_kw(ParseErrorKind::ExpectedName)
    }

    fn expect_ident_or_kw(&mut self, err: ParseErrorKind) -> ParseResult<Span> {
        let kind = self.tokens[self.pos].kind;
        if kind == TokenKind::Ident || kind.is_keyword() {
            let span = self.span();
            self.advance_pos();
            Ok(span)
        } else {
            Err(self.error(err))
        }
    }

    #[cold]
    fn error(&self, kind: ParseErrorKind) -> ParseError {
        Self::error_at(kind, self.span())
    }

    #[cold]
    const fn error_at(kind: ParseErrorKind, span: Span) -> ParseError {
        ParseError {
            kind,
            span,
            note: None,
        }
    }

    #[cold]
    #[expect(
        clippy::unnecessary_wraps,
        reason = "matches Note's Option<Box<Note>> field"
    )]
    fn first_defined_note(&self, span: Span) -> Option<Box<Note>> {
        Some(Box::new(Note {
            message: NoteMessage::FirstDefinedHere,
            span,
            path: self.grammar_path.to_path_buf(),
            source: self.ast.source().to_string(),
        }))
    }

    #[cold]
    fn err_arg_count(&self, name: TokenKind, expected: u8, got: usize, start: Span) -> ParseError {
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

    fn expect_close_args(&mut self, name: TokenKind, expected: u8, start: Span) -> ParseResult<()> {
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

    fn parse_item(&mut self) -> ParseResult<NodeId> {
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

    fn parse_print_item(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwPrint)?;
        self.expect(TokenKind::LParen)?;
        let arg = self.parse_expr()?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(Node::Print(arg), start.merge(end)))
    }

    fn parse_grammar_block(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwGrammar)?;
        if self.ast.context.grammar_config.is_some() {
            let first_span = self
                .ast
                .root_items
                .iter()
                .find(|&&id| matches!(self.ast.node(id), Node::Grammar))
                .map(|&id| self.ast.span(id))
                .unwrap();
            Err(ParseError {
                kind: ParseErrorKind::DuplicateGrammarBlock,
                span: start,
                note: self.first_defined_note(first_span),
            })?;
        }
        self.expect(TokenKind::LBrace)?;
        let mut config = GrammarConfig::default();
        let mut seen = [None::<Span>; ConfigField::COUNT];
        loop {
            if let Some(end) = self.eat(TokenKind::RBrace) {
                if config.language.is_none() {
                    return Err(ParseError {
                        kind: ParseErrorKind::MissingLanguageField,
                        span: start.merge(end),
                        note: None,
                    });
                }
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
                Err(ParseError {
                    kind: ParseErrorKind::DuplicateGrammarField(key.to_string()),
                    span: key_span,
                    note: self.first_defined_note(first_span),
                })?;
            }
            seen[field as usize] = Some(key_span);
            match field {
                ConfigField::Language => {
                    let s = self.expect_string()?;
                    let inner = s.strip_quotes();
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

    fn parse_rule_def(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwRule)?;
        let name = self.expect_ident_node()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        Ok(self.ast.push(Node::Rule { name, body }, start.merge(end)))
    }

    fn parse_override_rule_def(&mut self) -> ParseResult<NodeId> {
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

    fn parse_let_def(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwLet)?;
        let name = self.expect_ident_node()?;
        let ty = if self.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?.0)
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

    fn parse_fn_def(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwFn)?;
        let name = self.expect_ident_node()?;
        self.expect(TokenKind::LParen)?;
        let params = self.comma_sep(TokenKind::RParen, |this| {
            let pname = this.expect_ident_node()?;
            this.expect(TokenKind::Colon)?;
            Ok(Param {
                name: pname,
                ty: this.parse_type()?.0,
            })
        })?;
        self.expect(TokenKind::RParen)?;
        if self.eat(TokenKind::Arrow).is_none() {
            return Err(self.error(ParseErrorKind::MissingReturnType));
        }
        let return_ty = self.parse_type()?.0;
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

    fn parse_type(&mut self) -> ParseResult<(Ty, Span)> {
        if let Some(id_span) = self.eat(TokenKind::Ident) {
            return match self.ast.text(id_span) {
                "rule_t" => Ok((Ty::Rule, id_span)),
                "str_t" => Ok((Ty::Str, id_span)),
                "int_t" => Ok((Ty::Int, id_span)),
                "module_t" => Ok((Ty::AnyModule, id_span)),
                "list_t" => {
                    self.expect(TokenKind::Lt)?;
                    let (inner_ty, inner_span) = self.parse_type()?;
                    let ty = match inner_ty {
                        Ty::Rule => Ok(Ty::ListRule),
                        Ty::Str => Ok(Ty::ListStr),
                        Ty::Int => Ok(Ty::ListInt),
                        Ty::ListRule => Ok(Ty::ListListRule),
                        Ty::ListStr => Ok(Ty::ListListStr),
                        Ty::ListInt => Ok(Ty::ListListInt),
                        // All other types (too-deep nesting, objects, modules,
                        // internal types) are rejected as list elements.
                        _ => Err(ParseError {
                            kind: ParseErrorKind::ListInnerType(inner_ty),
                            span: inner_span,
                            note: None,
                        }),
                    };
                    let gt = self.expect(TokenKind::Gt)?;
                    Ok((ty?, id_span.merge(gt)))
                }
                "obj_t" => {
                    self.expect(TokenKind::Lt)?;
                    let (inner_ty, inner_span) = self.parse_type()?;
                    let ty = InnerTy::try_from(inner_ty)
                        .map(Ty::Object)
                        .map_err(|()| ParseError {
                            kind: ParseErrorKind::ObjectInnerType(inner_ty),
                            span: inner_span,
                            note: None,
                        });
                    let gt = self.expect(TokenKind::Gt)?;
                    Ok((ty?, id_span.merge(gt)))
                }
                "grammar_config_t" => Ok((Ty::GrammarConfig, id_span)),
                "void_t" => Err(ParseError {
                    kind: ParseErrorKind::InternalTypeNotAllowed(Ty::Void),
                    span: id_span,
                    note: None,
                }),
                "spread_t" => Err(ParseError {
                    kind: ParseErrorKind::InternalTypeNotAllowed(Ty::Spread),
                    span: id_span,
                    note: None,
                }),
                _ => Err(ParseError {
                    kind: ParseErrorKind::UnknownType(self.ast.text(id_span).to_string()),
                    span: id_span,
                    note: None,
                }),
            };
        }
        Err(self.error(ParseErrorKind::ExpectedType))
    }

    fn parse_expr(&mut self) -> ParseResult<NodeId> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(self.error(ParseErrorKind::NestingTooDeep));
        }
        let mut result = self.parse_primary()?;
        // Post-primary chaining: allow `.field` after any expression
        // (e.g. `grammar_config(base).extras`).
        let start = self.ast.span(result);
        while self.at(TokenKind::Dot) {
            self.advance_pos();
            let field_span = self.expect_name()?;
            let field = self.ast.push(Node::Ident, field_span);
            result = self.ast.push(
                Node::FieldAccess { obj: result, field },
                start.merge(self.ast.span(field)),
            );
        }
        self.depth -= 1;
        Ok(result)
    }

    fn parse_primary(&mut self) -> ParseResult<NodeId> {
        let start = self.span();
        // Builtin keywords are contextual: they act as keywords only when
        // followed by `(` (or `{` for `for`). Otherwise they are identifiers
        // so grammars can use names like `import`, `field`, etc. as rules.
        let next_lparen = self.next_is(TokenKind::LParen);
        match &self.tokens[self.pos].kind {
            TokenKind::KwSeq if next_lparen => self.parse_variadic(start, Node::Seq),
            TokenKind::KwChoice if next_lparen => self.parse_variadic(start, Node::Choice),
            TokenKind::KwRepeat if next_lparen => {
                self.parse_unary(start, TokenKind::KwRepeat, Node::Repeat)
            }
            TokenKind::KwRepeat1 if next_lparen => {
                self.parse_unary(start, TokenKind::KwRepeat1, Node::Repeat1)
            }
            TokenKind::KwOptional if next_lparen => {
                self.parse_unary(start, TokenKind::KwOptional, Node::Optional)
            }
            TokenKind::KwBlank if next_lparen => {
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
                    return Err(self.err_arg_count(TokenKind::KwBlank, 0, got, start));
                }
                let end = self.expect(TokenKind::RParen)?;
                Ok(self.ast.push(Node::Blank, start.merge(end)))
            }
            TokenKind::KwField if next_lparen => self.parse_field(start),
            TokenKind::KwAlias if next_lparen => self.parse_alias(start),
            TokenKind::KwToken if next_lparen => {
                self.parse_unary(start, TokenKind::KwToken, Node::Token)
            }
            TokenKind::KwTokenImmediate if next_lparen => {
                self.parse_unary(start, TokenKind::KwTokenImmediate, Node::TokenImmediate)
            }
            TokenKind::KwPrec if next_lparen => self.parse_prec(start, PrecVariant::Default),
            TokenKind::KwPrecLeft if next_lparen => self.parse_prec(start, PrecVariant::Left),
            TokenKind::KwPrecRight if next_lparen => self.parse_prec(start, PrecVariant::Right),
            TokenKind::KwPrecDynamic if next_lparen => self.parse_prec(start, PrecVariant::Dynamic),
            TokenKind::KwReserved if next_lparen => self.parse_reserved_expr(start),
            TokenKind::KwConcat if next_lparen => self.parse_variadic(start, Node::Concat),
            TokenKind::KwRegexp if next_lparen => self.parse_regexp(start),
            TokenKind::KwInherit if next_lparen => {
                self.parse_module_path(start, TokenKind::KwInherit, |p| Node::Inherit {
                    path: p,
                    module: None,
                })
            }
            TokenKind::KwImport if next_lparen => {
                self.parse_module_path(start, TokenKind::KwImport, |p| Node::Import {
                    path: p,
                    module: None,
                })
            }
            TokenKind::KwAppend if next_lparen => self.parse_append(start),
            TokenKind::KwPrint if next_lparen => self.parse_print_item(),
            TokenKind::KwGrammarConfig if next_lparen => {
                self.parse_unary(start, TokenKind::KwGrammarConfig, Node::GrammarConfig)
            }
            TokenKind::KwFor => self.parse_for(start),
            TokenKind::Ident => self.parse_ident_expr(start),
            // Keyword used as identifier (e.g. `import` as a rule reference)
            k if k.is_keyword() => self.parse_ident_expr(start),
            TokenKind::StringLit => {
                self.advance_pos();
                Ok(self.ast.push(Node::StringLit, start.strip_quotes()))
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

    fn parse_variadic(&mut self, start: Span, make: fn(ChildRange) -> Node) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        let range = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(make(range), start.merge(end)))
    }

    fn parse_unary(
        &mut self,
        start: Span,
        name: TokenKind,
        make: fn(NodeId) -> Node,
    ) -> ParseResult<NodeId> {
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
        name: TokenKind,
        parse_first: fn(&mut Self) -> ParseResult<NodeId>,
    ) -> ParseResult<(NodeId, NodeId, Span)> {
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

    fn parse_field(&mut self, start: Span) -> ParseResult<NodeId> {
        let (name, content, end) = self.parse_binary(start, TokenKind::KwField, |this| {
            let s = this.expect_name()?;
            Ok(this.ast.push(Node::Ident, s))
        })?;
        Ok(self
            .ast
            .push(Node::Field { name, content }, start.merge(end)))
    }

    fn parse_alias(&mut self, start: Span) -> ParseResult<NodeId> {
        let (content, target, end) =
            self.parse_binary(start, TokenKind::KwAlias, Self::parse_expr)?;
        Ok(self
            .ast
            .push(Node::Alias { content, target }, start.merge(end)))
    }

    fn parse_prec(&mut self, start: Span, variant: PrecVariant) -> ParseResult<NodeId> {
        let (value, content, end) = self.parse_binary(start, variant.into(), Self::parse_expr)?;
        let node = match variant {
            PrecVariant::Default => Node::Prec { value, content },
            PrecVariant::Left => Node::PrecLeft { value, content },
            PrecVariant::Right => Node::PrecRight { value, content },
            PrecVariant::Dynamic => Node::PrecDynamic { value, content },
        };
        Ok(self.ast.push(node, start.merge(end)))
    }

    /// Parse `reserved("context", content)` expression (not the config block).
    fn parse_reserved_expr(&mut self, start: Span) -> ParseResult<NodeId> {
        let (context, content, end) = self.parse_binary(start, TokenKind::KwReserved, |this| {
            let cs = this.expect_string()?;
            Ok(this.ast.push(Node::StringLit, cs.strip_quotes()))
        })?;
        Ok(self
            .ast
            .push(Node::Reserved { context, content }, start.merge(end)))
    }

    fn parse_regexp(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(TokenKind::KwRegexp, 1, 0, start));
        }
        let pattern = self.parse_expr()?;
        let flags = if self.eat(TokenKind::Comma).is_some() && !self.at(TokenKind::RParen) {
            Some(self.parse_expr()?)
        } else {
            None
        };
        self.expect_close_args(TokenKind::KwRegexp, 1 + u8::from(flags.is_some()), start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self
            .ast
            .push(Node::DynRegex { pattern, flags }, start.merge(end)))
    }

    fn parse_module_path(
        &mut self,
        start: Span,
        kw: TokenKind,
        make_node: fn(NodeId) -> Node,
    ) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(kw, 1, 0, start));
        }
        let path_span = self.expect_string()?;
        let path = self.ast.push(Node::StringLit, path_span.strip_quotes());
        self.expect_close_args(kw, 1, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(make_node(path), start.merge(end)))
    }

    fn parse_append(&mut self, start: Span) -> ParseResult<NodeId> {
        let (left, right, end) = self.parse_binary(start, TokenKind::KwAppend, Self::parse_expr)?;
        Ok(self
            .ast
            .push(Node::Append { left, right }, start.merge(end)))
    }

    fn parse_for(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        let bindings = self.comma_sep(TokenKind::RParen, |this| {
            let name = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok((name, this.parse_type()?.0))
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

    fn parse_ident_expr(&mut self, start: Span) -> ParseResult<NodeId> {
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
                let member = self.expect_ident_node()?;
                if self.at(TokenKind::LParen) {
                    // h::fn_name(args) - qualified call
                    self.advance_pos();
                    let mut children = vec![id, member];
                    let arg_ids = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
                    children.extend_from_slice(self.ast.child_slice(arg_ids));
                    let end = self.expect(TokenKind::RParen)?;
                    let range = self
                        .ast
                        .push_children(&children)
                        .map_err(|_| self.error(ParseErrorKind::NestingTooDeep))?;
                    id = self.ast.push(Node::QualifiedCall(range), start.merge(end));
                    break;
                }
                id = self.ast.push(
                    Node::QualifiedAccess { obj: id, member },
                    start.merge(self.ast.span(member)),
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

    fn parse_list(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        let range = self.comma_sep_children(TokenKind::RBracket, Self::parse_expr)?;
        let end = self.expect(TokenKind::RBracket)?;
        Ok(self.ast.push(Node::List(range), start.merge(end)))
    }

    fn parse_tuple(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        let range = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(Node::Tuple(range), start.merge(end)))
    }

    fn parse_object(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        let fields = self.comma_sep(TokenKind::RBrace, |this| {
            let key = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok((key, this.parse_expr()?))
        })?;
        // Check for duplicate keys (O(n) via hash set).
        let mut seen = rustc_hash::FxHashMap::with_capacity_and_hasher(
            fields.len(),
            rustc_hash::FxBuildHasher,
        );
        for &(span, _) in &fields {
            let key = self.ast.text(span);
            if let Some(&first_span) = seen.get(key) {
                return Err(ParseError {
                    kind: ParseErrorKind::DuplicateObjectKey(key.to_string()),
                    span,
                    note: self.first_defined_note(first_span),
                });
            }
            seen.insert(key, span);
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
        mut parse_item: impl FnMut(&mut Self) -> ParseResult<NodeId>,
    ) -> ParseResult<ChildRange> {
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
        mut parse_item: impl FnMut(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<Vec<T>> {
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

#[derive(Copy, Clone, PartialEq, Eq)]
enum PrecVariant {
    Default,
    Left,
    Right,
    Dynamic,
}

impl From<PrecVariant> for TokenKind {
    fn from(value: PrecVariant) -> Self {
        match value {
            PrecVariant::Default => Self::KwPrec,
            PrecVariant::Left => Self::KwPrecLeft,
            PrecVariant::Right => Self::KwPrecRight,
            PrecVariant::Dynamic => Self::KwPrecDynamic,
        }
    }
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

pub type ParseResult<T> = Result<T, ParseError>;

#[derive(Debug, Serialize, Error)]
pub struct ParseError {
    pub kind: ParseErrorKind,
    pub span: Span,
    pub note: Option<Box<Note>>,
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
    InternalTypeNotAllowed(Ty),
    ListInnerType(Ty),
    ObjectInnerType(Ty),
    UnknownGrammarField(String),
    DuplicateGrammarField(String),
    DuplicateObjectKey(String),
    MissingReturnType,
    ExpectedFunctionName,
    DuplicateGrammarBlock,
    MissingLanguageField,
    NestingTooDeep,
    TooManyChildren,
    WrongArgumentCount {
        name: TokenKind,
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
            ParseErrorKind::ListInnerType(ty) => {
                write!(f, "{ty} cannot be stored inside a list")
            }
            ParseErrorKind::ObjectInnerType(ty) => {
                write!(f, "{ty} cannot be stored inside an object")
            }
            ParseErrorKind::InternalTypeNotAllowed(ty) => {
                write!(
                    f,
                    "{ty} is an internal type and cannot be used as a type annotation"
                )
            }
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
            ParseErrorKind::DuplicateGrammarBlock => write!(f, "only one grammar block is allowed"),
            ParseErrorKind::MissingLanguageField => {
                write!(f, "grammar block must have a 'language' field")
            }
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
                0 => write!(f, "{name} takes no arguments, got {got}"),
                1 => write!(f, "{name} takes 1 argument, got {got}"),
                _ => write!(f, "{name} takes {expected} arguments, got {got}"),
            },
        }
    }
}
