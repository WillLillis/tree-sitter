//! Recursive descent parser for the native grammar DSL.

use std::path::Path;

use serde::Serialize;
use thiserror::Error;

use super::{
    InnerTy, Note, NoteMessage,
    ast::{
        Ast, ChildRange, FnConfig, ForConfig, GrammarConfig, Node, NodeId, Param, PrecKind,
        RepeatKind, Span,
    },
    lexer::{Token, TokenKind},
    typecheck::Ty,
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
            source: self.ast.ctx.source.clone(),
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
            TokenKind::KwRule => self.parse_rule_def(false),
            TokenKind::KwOverride => self.parse_rule_def(true),
            TokenKind::KwLet => self.parse_let_def(),
            TokenKind::KwFn => self.parse_fn_def(),
            _ => Err(self.error(ParseErrorKind::ExpectedItem)),
        }
    }

    fn parse_grammar_block(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwGrammar)?;
        if self.ast.ctx.grammar_config.is_some() {
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
                    return Err(Self::error_at(
                        ParseErrorKind::MissingLanguageField,
                        start.merge(end),
                    ));
                }
                self.ast.ctx.grammar_config = Some(config);
                return Ok(self.ast.push(Node::Grammar, start.merge(end)));
            }
            let key_span = self.expect_name()?;
            self.expect(TokenKind::Colon)?;
            let key = self.ast.ctx.text(key_span);
            let field = ConfigField::from_str(key).ok_or_else(|| {
                Self::error_at(
                    ParseErrorKind::UnknownGrammarField(key.to_string()),
                    key_span,
                )
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
                    config.language = Some(self.ast.ctx.text(s.strip_quotes()).to_string());
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
            }
            if !self.at(TokenKind::RBrace) {
                self.expect(TokenKind::Comma)?;
            }
        }
    }

    fn parse_rule_def(&mut self, is_override: bool) -> ParseResult<NodeId> {
        let start = if is_override {
            let s = self.expect(TokenKind::KwOverride)?;
            self.expect(TokenKind::KwRule)?;
            s
        } else {
            self.expect(TokenKind::KwRule)?
        };
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        Ok(self.ast.push(
            Node::Rule {
                is_override,
                name,
                body,
            },
            start.merge(end),
        ))
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
        let name = self.expect_ident()?;
        self.expect(TokenKind::LParen)?;
        let params = self.comma_sep(TokenKind::RParen, |this| {
            let pname = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok(Param {
                name: pname,
                ty: this.parse_type()?.0,
            })
        })?;
        self.expect(TokenKind::RParen)?;
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
            return match self.ast.ctx.text(id_span) {
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
                        _ => Err(Self::error_at(
                            ParseErrorKind::ListInnerType(inner_ty),
                            inner_span,
                        )),
                    };
                    let gt = self.expect(TokenKind::Gt)?;
                    Ok((ty?, id_span.merge(gt)))
                }
                "obj_t" => {
                    self.expect(TokenKind::Lt)?;
                    let (inner_ty, inner_span) = self.parse_type()?;
                    let ty = InnerTy::try_from(inner_ty).map(Ty::Object).map_err(|()| {
                        Self::error_at(ParseErrorKind::ObjectInnerType(inner_ty), inner_span)
                    });
                    let gt = self.expect(TokenKind::Gt)?;
                    Ok((ty?, id_span.merge(gt)))
                }
                "grammar_config_t" => Ok((Ty::GrammarConfig, id_span)),
                "spread_t" => Err(Self::error_at(
                    ParseErrorKind::InternalTypeNotAllowed(Ty::Spread),
                    id_span,
                )),
                _ => Err(Self::error_at(
                    ParseErrorKind::UnknownType(self.ast.ctx.text(id_span).to_string()),
                    id_span,
                )),
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
            let field = self.expect_name()?;
            result = self
                .ast
                .push(Node::FieldAccess { obj: result, field }, start.merge(field));
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
        let kw = self.tokens[self.pos].kind;
        match kw {
            TokenKind::KwSeq if next_lparen => self.parse_variadic(start, Node::Seq),
            TokenKind::KwChoice if next_lparen => self.parse_variadic(start, Node::Choice),
            TokenKind::KwRepeat | TokenKind::KwRepeat1 | TokenKind::KwOptional if next_lparen => {
                let repeat_kind = match kw {
                    TokenKind::KwRepeat => RepeatKind::ZeroOrMore,
                    TokenKind::KwRepeat1 => RepeatKind::OneOrMore,
                    _ => RepeatKind::Optional,
                };
                self.parse_unary(start, kw, |inner| Node::Repeat {
                    kind: repeat_kind,
                    inner,
                })
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
            TokenKind::KwToken | TokenKind::KwTokenImmediate if next_lparen => {
                let immediate = kw == TokenKind::KwTokenImmediate;
                self.parse_unary(start, kw, |inner| Node::Token { immediate, inner })
            }
            TokenKind::KwPrec if next_lparen => self.parse_prec(start, PrecKind::Default),
            TokenKind::KwPrecLeft if next_lparen => self.parse_prec(start, PrecKind::Left),
            TokenKind::KwPrecRight if next_lparen => self.parse_prec(start, PrecKind::Right),
            TokenKind::KwPrecDynamic if next_lparen => self.parse_prec(start, PrecKind::Dynamic),
            TokenKind::KwReserved if next_lparen => self.parse_reserved_expr(start),
            TokenKind::KwConcat if next_lparen => self.parse_variadic(start, Node::Concat),
            TokenKind::KwRegexp if next_lparen => self.parse_regexp(start),
            TokenKind::KwInherit if next_lparen => {
                self.parse_module_path(start, TokenKind::KwInherit, false)
            }
            TokenKind::KwImport if next_lparen => {
                self.parse_module_path(start, TokenKind::KwImport, true)
            }
            TokenKind::KwAppend if next_lparen => self.parse_append(start),
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
                self.advance_pos();
                Ok(self.ast.push(Node::IntLit(n), start))
            }
            TokenKind::RawStringLit { hash_count } => {
                self.advance_pos();
                Ok(self.ast.push(Node::RawStringLit { hash_count }, start))
            }
            TokenKind::Minus => {
                self.advance_pos();
                let inner = self.parse_expr()?;
                Ok(self
                    .ast
                    .push(Node::Neg(inner), start.merge(self.ast.span(inner))))
            }
            TokenKind::LBracket => self.parse_delimited(start, TokenKind::RBracket, Node::List),
            TokenKind::LParen => self.parse_delimited(start, TokenKind::RParen, Node::Tuple),
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
        make: impl FnOnce(NodeId) -> Node,
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

    fn parse_prec(&mut self, start: Span, kind: PrecKind) -> ParseResult<NodeId> {
        let (value, content, end) = self.parse_binary(start, kind.into(), Self::parse_expr)?;
        Ok(self.ast.push(
            Node::Prec {
                kind,
                value,
                content,
            },
            start.merge(end),
        ))
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
        is_import: bool,
    ) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(kw, 1, 0, start));
        }
        let path_span = self.expect_string()?;
        self.expect_close_args(kw, 1, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.ast.push(
            Node::ModuleRef {
                import: is_import,
                path: path_span.strip_quotes(),
                module: None,
            },
            start.merge(end),
        ))
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
                let field = self.expect_name()?;
                id = self
                    .ast
                    .push(Node::FieldAccess { obj: id, field }, start.merge(field));
            } else if self.at(TokenKind::ColonColon) {
                self.advance_pos();
                let member_span = self.expect_ident()?;
                if self.at(TokenKind::LParen) {
                    // h::fn_name(args) - qualified call needs member as NodeId in ChildRange
                    let member_id = self.ast.push(Node::Ident, member_span);
                    self.advance_pos();
                    let mut children = vec![id, member_id];
                    let arg_ids = self.comma_sep_children(TokenKind::RParen, Self::parse_expr)?;
                    children.extend_from_slice(self.ast.ctx.child_slice(arg_ids));
                    let end = self.expect(TokenKind::RParen)?;
                    let range = self
                        .ast
                        .push_children(&children)
                        .ok_or_else(|| self.error(ParseErrorKind::NestingTooDeep))?;
                    id = self.ast.push(Node::QualifiedCall(range), start.merge(end));
                    break;
                }
                id = self.ast.push(
                    Node::QualifiedAccess {
                        obj: id,
                        member: member_span,
                    },
                    start.merge(member_span),
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

    fn parse_delimited(
        &mut self,
        start: Span,
        close: TokenKind,
        make: fn(ChildRange) -> Node,
    ) -> ParseResult<NodeId> {
        self.advance_pos();
        let range = self.comma_sep_children(close, Self::parse_expr)?;
        let end = self.expect(close)?;
        Ok(self.ast.push(make(range), start.merge(end)))
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
            let key = self.ast.ctx.text(span);
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
            .ok_or_else(|| self.error(ParseErrorKind::TooManyChildren))?;
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
            .ok_or_else(|| self.error(ParseErrorKind::TooManyChildren))?;
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

impl From<PrecKind> for TokenKind {
    fn from(value: PrecKind) -> Self {
        match value {
            PrecKind::Default => Self::KwPrec,
            PrecKind::Left => Self::KwPrecLeft,
            PrecKind::Right => Self::KwPrecRight,
            PrecKind::Dynamic => Self::KwPrecDynamic,
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
}

impl ConfigField {
    const COUNT: usize = 10; // `std::mem::variant_count` once stabilized

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

#[derive(Debug, PartialEq, Eq, Serialize, Error)]
pub enum ParseErrorKind {
    #[error("expected {expected}, got {got}")]
    ExpectedToken { expected: TokenKind, got: TokenKind },
    #[error("expected identifier")]
    ExpectedIdent,
    #[error("expected string literal")]
    ExpectedString,
    #[error("expected name")]
    ExpectedName,
    #[error("expected expression")]
    ExpectedExpression,
    #[error("expected type")]
    ExpectedType,
    #[error("expected 'grammar', 'rule', 'let', 'fn', or 'print'")]
    ExpectedItem,
    #[error("unknown type '{0}'")]
    UnknownType(String),
    #[error("{0} is an internal type and cannot be used as a type annotation")]
    InternalTypeNotAllowed(Ty),
    #[error("{0} cannot be stored inside a list")]
    ListInnerType(Ty),
    #[error("{0} cannot be stored inside an object")]
    ObjectInnerType(Ty),
    #[error("unknown grammar field '{0}'")]
    UnknownGrammarField(String),
    #[error("duplicate grammar field '{0}'")]
    DuplicateGrammarField(String),
    #[error("duplicate object key '{0}'")]
    DuplicateObjectKey(String),
    #[error("only identifiers can be used as function names")]
    ExpectedFunctionName,
    #[error("only one grammar block is allowed")]
    DuplicateGrammarBlock,
    #[error("grammar block must have a 'language' field")]
    MissingLanguageField,
    #[error("expression nesting too deep (maximum {MAX_PARSE_DEPTH})")]
    NestingTooDeep,
    #[error("too many elements (maximum 65535)")]
    TooManyChildren,
    #[error("{}", format_arg_count(*.expected, .name, *.got))]
    WrongArgumentCount {
        name: TokenKind,
        expected: u8,
        got: usize,
    },
}

#[expect(
    clippy::trivially_copy_pass_by_ref,
    reason = "called by thiserror with references"
)]
fn format_arg_count(expected: u8, name: &TokenKind, got: usize) -> String {
    match expected {
        0 => format!("{name} takes no arguments, got {got}"),
        1 => format!("{name} takes 1 argument, got {got}"),
        _ => format!("{name} takes {expected} arguments, got {got}"),
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.kind.fmt(f)
    }
}
