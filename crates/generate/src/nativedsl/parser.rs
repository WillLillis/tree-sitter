//! Recursive descent parser for the native grammar DSL.

use std::path::PathBuf;

use rustc_hash::{FxBuildHasher, FxHashMap};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::{
    InnerTy, NoteMessage, ParseError,
    ast::{
        BinOp, ChildRange, ConfigField, ForConfig, ForId, GrammarConfig, IdentKind, MacroConfig,
        MacroKind, ModuleContext, Node, NodeId, Param, PrecKind, RepeatKind, SharedAst, Span,
    },
    lexer::{Token, TokenKind},
    typecheck::{DataTy, Ty},
};

const MAX_PARSE_DEPTH: u16 = 192;

#[derive(Clone, Copy)]
enum LocalBinding {
    MacroParam(Ty, u8),
    ForBinding(ForId, Ty, u8),
}

pub struct Parser<'tok, 'shared> {
    tokens: &'tok [Token],
    pos: usize,
    shared: &'shared mut SharedAst,
    ctx: ModuleContext,
    scratch: Vec<NodeId>,
    /// Stack of local bindings (macro params and for-loop bindings).
    /// Innermost scope is at the end. Pushed before parsing a body,
    /// truncated after.
    locals: Vec<(Span, LocalBinding)>,
    depth: u16,
}

impl<'tok, 'shared> Parser<'tok, 'shared> {
    /// SAFETY:
    ///
    /// Caller must pass a token stream produced by
    /// [`super::lexer::Lexer::tokenize`] (in particular, terminated by
    /// `TokenKind::Eof`).
    #[must_use]
    pub fn new(
        tokens: &'tok [Token],
        source: String,
        grammar_path: PathBuf,
        shared: &'shared mut SharedAst,
    ) -> Self {
        let root_cap = tokens.len() / 30;
        Self {
            tokens,
            pos: 0,
            shared,
            ctx: ModuleContext {
                source,
                path: grammar_path,
                grammar_config: None,
                root_items: Vec::with_capacity(root_cap),
                inherit_ref: None,
                module_refs: Vec::new(),
                external_names: Vec::new(),
                node_range: 0..0,
                has_cfg: false,
                cfg_declared: FxHashMap::default(),
                cfg_dropped: FxHashMap::default(),
            },
            scratch: Vec::with_capacity(32),
            locals: Vec::new(),
            depth: 0,
        }
    }

    pub fn parse(mut self) -> ParseResult<ModuleContext> {
        // Capture this module's NodeId range so consumers (e.g. LSP) can
        // attribute nodes back to the right ModuleContext / source text.
        let start = self.shared.arena.next_id().into();
        self.skip_comments();
        while !self.at_eof() {
            let id = self.parse_item()?;
            self.ctx.root_items.push(id);
            self.skip_comments();
        }
        let end = self.shared.arena.next_id().into();
        debug_assert!(start <= end);
        self.ctx.node_range = start..end;
        Ok(self.ctx)
    }

    /// Current token. Eof terminates the stream and the parser never
    /// advances past it, so `pos` is always in bounds.
    #[inline]
    fn current(&self) -> &Token {
        debug_assert!(self.pos < self.tokens.len());
        unsafe { self.tokens.get_unchecked(self.pos) }
    }

    fn skip_comments(&mut self) {
        while self.current().kind == TokenKind::Comment {
            self.pos += 1;
        }
    }

    fn span(&self) -> Span {
        self.current().span
    }

    fn at_eof(&self) -> bool {
        self.current().kind == TokenKind::Eof
    }

    fn at(&self, kind: TokenKind) -> bool {
        self.current().kind == kind
    }

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
    fn advance_pos(&mut self) {
        self.pos += 1;
        while self.current().kind == TokenKind::Comment {
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
                got: self.current().kind,
            })
        })
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
        let kind = self.current().kind;
        if kind == TokenKind::Ident || kind.is_keyword() {
            let span = self.span();
            self.advance_pos();
            Ok(span)
        } else {
            Err(self.error(err))
        }
    }

    fn error(&self, kind: ParseErrorKind) -> ParseError {
        ParseError::new(kind, self.span())
    }

    fn dup_err(&self, kind: ParseErrorKind, span: Span, first_span: Span) -> ParseError {
        ParseError::with_note(
            kind,
            span,
            self.ctx.note(NoteMessage::FirstDefinedHere, first_span),
        )
    }

    fn check_duplicate_names<T>(
        &self,
        items: &[T],
        name_of: fn(&T) -> Span,
        kind_ctor: fn(String) -> ParseErrorKind,
    ) -> ParseResult<()> {
        for i in 1..items.len() {
            let span = name_of(&items[i]);
            let name = self.ctx.text(span);
            for item in items.iter().take(i) {
                if self.ctx.text(name_of(item)) == name {
                    return Err(self.dup_err(kind_ctor(name.to_string()), span, name_of(item)));
                }
            }
        }
        Ok(())
    }

    fn err_arg_count(&self, name: TokenKind, expected: u8, got: usize, start: Span) -> ParseError {
        ParseError::new(
            ParseErrorKind::WrongArgumentCount {
                name,
                expected,
                got,
            },
            start.merge(self.span()),
        )
    }

    fn expect_close_args(&mut self, name: TokenKind, expected: u8, start: Span) -> ParseResult<()> {
        if self.at(TokenKind::Comma) {
            let mut got = expected as usize;
            self.advance_pos();
            if self.at(TokenKind::RParen) {
                return Ok(());
            }
            self.parse_expr()?;
            got += 1;
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
        if let Some((cfg_start, name)) = self.try_parse_cfg_attribute()? {
            // Cfg attribute layers stack via recursion here and through
            // apply_cfg's walk_cfg later. Cap nesting via the shared parse
            // depth budget to keep both recursive paths safe.
            self.depth += 1;
            if self.depth > MAX_PARSE_DEPTH {
                return Err(self.error(ParseErrorKind::NestingTooDeep));
            }
            let child = self.parse_item()?;
            self.depth -= 1;
            // A cfg-gated grammar block is structurally meaningless: the
            // grammar's `flags` declaration is read before cfg gating runs,
            // so the block would either fail to declare its own gating flag
            // or self-disable inconsistently. Reject at parse time.
            if matches!(self.shared.arena.get(child), Node::Grammar) {
                return Err(ParseError::new(
                    ParseErrorKind::CfgOnGrammarBlock,
                    cfg_start,
                ));
            }
            let end = self.shared.arena.span(child);
            return Ok(self
                .shared
                .arena
                .push(Node::Cfg { name, child }, cfg_start.merge(end)));
        }
        match &self.current().kind {
            TokenKind::KwGrammar => self.parse_grammar_block(),
            TokenKind::KwRule => self.parse_rule_def(false),
            TokenKind::KwOverride => self.parse_rule_def(true),
            TokenKind::KwLet => self.parse_let_def(),
            TokenKind::KwMacro => self.parse_macro_def(),
            TokenKind::KwRules => self.parse_rule_set_def(),
            TokenKind::KwExternal => self.parse_external_decl(),
            // `@name(args...)` - rule-set macro invocation, expanded later.
            TokenKind::At => self.parse_top_level_call(),
            _ => Err(self.error(ParseErrorKind::ExpectedItem)),
        }
    }

    /// If the next token is `#`, consume `#[cfg(IDENT)]` and return the
    /// `(attribute span, flag-name span)` pair. Otherwise return `None`
    /// without consuming anything.
    fn try_parse_cfg_attribute(&mut self) -> ParseResult<Option<(Span, Span)>> {
        let Some(start) = self.eat(TokenKind::Pound) else {
            return Ok(None);
        };
        self.expect(TokenKind::LBracket)?;
        let kw = self.expect_ident()?;
        let kw_text = self.ctx.text(kw);
        if kw_text != "cfg" {
            return Err(ParseError::new(
                ParseErrorKind::ExpectedCfgKeyword(kw_text.to_string()),
                kw,
            ));
        }
        self.expect(TokenKind::LParen)?;
        let name = self.expect_ident()?;
        self.expect(TokenKind::RParen)?;
        let end = self.expect(TokenKind::RBracket)?;
        self.ctx.has_cfg = true;
        Ok(Some((start.merge(end), name)))
    }

    /// Parse an expression, optionally preceded by one or more `#[cfg(NAME)]`
    /// attributes. Used in list-member positions (seq/choice members, list
    /// literal elements). Nested cfgs are intersection: all must be active for
    /// the wrapped expression to survive.
    fn parse_expr_with_cfg(&mut self) -> ParseResult<NodeId> {
        if let Some((cfg_start, name)) = self.try_parse_cfg_attribute()? {
            self.depth += 1;
            if self.depth > MAX_PARSE_DEPTH {
                return Err(self.error(ParseErrorKind::NestingTooDeep));
            }
            let child = self.parse_expr_with_cfg()?;
            self.depth -= 1;
            let end = self.shared.arena.span(child);
            return Ok(self
                .shared
                .arena
                .push(Node::Cfg { name, child }, cfg_start.merge(end)));
        }
        self.parse_expr()
    }

    fn parse_external_decl(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwExternal)?;
        let name = self.expect_ident()?;
        self.ctx.external_names.push(name);
        Ok(self
            .shared
            .arena
            .push(Node::External { name }, start.merge(name)))
    }

    fn parse_grammar_block(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwGrammar)?;
        if self.ctx.grammar_config.is_some() {
            let first_span = self
                .ctx
                .root_items
                .iter()
                .find(|&&id| matches!(self.shared.arena.get(id), Node::Grammar))
                .map(|&id| self.shared.arena.span(id))
                .unwrap();
            Err(self.dup_err(ParseErrorKind::DuplicateGrammarBlock, start, first_span))?;
        }
        self.expect(TokenKind::LBrace)?;
        let mut config = GrammarConfig::default();
        let mut seen = [None::<Span>; ConfigField::COUNT];
        loop {
            if let Some(end) = self.eat(TokenKind::RBrace) {
                if config.language.is_none() {
                    return Err(ParseError::new(
                        ParseErrorKind::MissingLanguageField,
                        start.merge(end),
                    ));
                }
                self.ctx.grammar_config = Some(config);
                return Ok(self.shared.arena.push(Node::Grammar, start.merge(end)));
            }
            let key_span = self.expect_name()?;
            self.expect(TokenKind::Colon)?;
            let key = self.ctx.text(key_span);
            let field = ConfigField::try_from(key).map_err(|()| {
                ParseError::new(
                    ParseErrorKind::UnknownGrammarField(key.to_string()),
                    key_span,
                )
            })?;
            if let Some(first_span) = seen[field as usize].replace(key_span) {
                Err(self.dup_err(
                    ParseErrorKind::DuplicateGrammarField(key.to_string()),
                    key_span,
                    first_span,
                ))?;
            }
            match field {
                ConfigField::Language => {
                    let s = self.expect_string()?;
                    config.language = Some(self.ctx.text(s.strip_quotes()).to_string());
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
                ConfigField::Start => config.start = Some(self.parse_expr()?),
                ConfigField::Flags => config.flags = Some(self.parse_expr()?),
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
        // `rule @<expr> { ... }` - computed name; consumed by expand.
        if self.eat(TokenKind::At).is_some() {
            let name_expr = self.parse_postfix()?;
            self.expect(TokenKind::LBrace)?;
            let body = self.parse_expr()?;
            let end = self.expect(TokenKind::RBrace)?;
            return Ok(self.shared.arena.push(
                Node::ComputedRule {
                    is_override,
                    name_expr,
                    body,
                },
                start.merge(end),
            ));
        }
        let name = self.expect_ident()?;
        self.expect(TokenKind::LBrace)?;
        let body = self.parse_expr()?;
        let end = self.expect(TokenKind::RBrace)?;
        Ok(self.shared.arena.push(
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
        let name = self.expect_ident()?;
        let ty = if self.eat(TokenKind::Colon).is_some() {
            Some(self.parse_type()?.0)
        } else {
            None
        };
        self.expect(TokenKind::Eq)?;
        let value = self.parse_expr()?;
        Ok(self.shared.arena.push(
            Node::Let { name, ty, value },
            start.merge(self.shared.arena.span(value)),
        ))
    }

    fn parse_macro_def(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwMacro)?;
        self.parse_macro_like(start, /* rule_set */ false)
    }

    /// `rules NAME(params) { rule decls... }` - expanded at each top-level
    /// `@NAME(args)` call site by `expand_macro_calls`. Distinct keyword from
    /// `macro` so the file-local, top-level-emitting nature of the binding is
    /// visible at the def.
    fn parse_rule_set_def(&mut self) -> ParseResult<NodeId> {
        let start = self.expect(TokenKind::KwRules)?;
        self.parse_macro_like(start, /* rule_set */ true)
    }

    fn parse_macro_like(&mut self, start: Span, rule_set: bool) -> ParseResult<NodeId> {
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
        self.check_duplicate_names(&params, |p| p.name, ParseErrorKind::DuplicateParameter)?;
        if params.len() > u8::MAX as usize {
            return Err(self.error(ParseErrorKind::TooManyBindings));
        }
        let kind = if rule_set {
            MacroKind::RuleSet
        } else {
            MacroKind::Expression(self.parse_type()?.0)
        };
        let brace_start = self.expect(TokenKind::LBrace)?;
        let body = stack_scope!(self.locals, |_saved| {
            for (i, p) in params.iter().enumerate() {
                self.locals
                    .push((p.name, LocalBinding::MacroParam(p.ty, i as u8)));
            }
            match kind {
                MacroKind::Expression(_) => self.parse_expr(),
                MacroKind::RuleSet => self.parse_rule_set_body(brace_start),
            }
        })?;
        let end = self.expect(TokenKind::RBrace)?;
        let macro_idx = self.shared.pools.push_macro(MacroConfig {
            name,
            params,
            body,
            kind,
        });
        Ok(self
            .shared
            .arena
            .push(Node::Macro(macro_idx), start.merge(end)))
    }

    /// Emits `Node::Call` at item position; consumed by `expand_macro_calls`.
    /// Caller is positioned at the leading `@`.
    fn parse_top_level_call(&mut self) -> ParseResult<NodeId> {
        let at_span = self.expect(TokenKind::At)?;
        let name_span = self.expect_ident()?;
        if self.at(TokenKind::ColonColon) {
            return Err(self.error(ParseErrorKind::QualifiedRuleSetCall));
        }
        let name_id = self
            .shared
            .arena
            .push(Node::Ident(IdentKind::Unresolved), name_span);
        self.expect(TokenKind::LParen)?;
        let args = self.comma_sep_children(&[], TokenKind::RParen, Self::parse_expr)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.shared.arena.push(
            Node::Call {
                name: name_id,
                args,
            },
            at_span.merge(end),
        ))
    }

    /// Parse `rule` / `override rule` decls until `}`, wrap in `Node::RuleSet`.
    /// Caller consumes the closing brace.
    fn parse_rule_set_body(&mut self, brace_start: Span) -> ParseResult<NodeId> {
        let mut decls = Vec::new();
        loop {
            if self.at(TokenKind::RBrace) {
                break;
            }
            let id = match self.current().kind {
                TokenKind::KwRule => self.parse_rule_def(false)?,
                TokenKind::KwOverride => self.parse_rule_def(true)?,
                _ => return Err(self.error(ParseErrorKind::RuleSetBodyRequiresRuleDecl)),
            };
            decls.push(id);
        }
        // expand_macro_calls relies on >=1 rule for in-place slot placement.
        if decls.is_empty() {
            return Err(self.error(ParseErrorKind::EmptyRuleSetMacroBody));
        }
        let range = self
            .shared
            .pools
            .push_children(&decls)
            .ok_or_else(|| self.error(ParseErrorKind::TooManyChildren))?;
        let end = self.span();
        Ok(self
            .shared
            .arena
            .push(Node::RuleSet(range), brace_start.merge(end)))
    }

    fn parse_type(&mut self) -> ParseResult<(Ty, Span)> {
        if let Some(id_span) = self.eat(TokenKind::Ident) {
            return match self.ctx.text(id_span) {
                "rule_t" => Ok((Ty::RULE, id_span)),
                "str_t" => Ok((Ty::STR, id_span)),
                "int_t" => Ok((Ty::INT, id_span)),
                "module_t" => Ok((Ty::ANY_MODULE, id_span)),
                "list_t" => {
                    self.expect(TokenKind::Lt)?;
                    let (inner_ty, inner_span) = self.parse_type()?;
                    let ty = inner_ty.to_list().ok_or_else(|| {
                        ParseError::new(ParseErrorKind::ListInnerType(inner_ty), inner_span)
                    })?;
                    let gt = self.expect(TokenKind::Gt)?;
                    Ok((ty, id_span.merge(gt)))
                }
                "obj_t" => {
                    self.expect(TokenKind::Lt)?;
                    let (inner_ty, inner_span) = self.parse_type()?;
                    let inner = match inner_ty {
                        Ty::Data(d) => InnerTy::try_from(d).map_err(|()| {
                            ParseError::new(ParseErrorKind::ObjectInnerType(inner_ty), inner_span)
                        })?,
                        _ => {
                            return Err(ParseError::new(
                                ParseErrorKind::ObjectInnerType(inner_ty),
                                inner_span,
                            ));
                        }
                    };
                    let gt = self.expect(TokenKind::Gt)?;
                    Ok((Ty::Data(DataTy::Object(inner)), id_span.merge(gt)))
                }
                _ => Err(ParseError::new(
                    ParseErrorKind::UnknownType(self.ctx.text(id_span).to_string()),
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
        let mut lhs = self.parse_postfix()?;
        // Left-associative chain of `+` / `-` at int_t positions.
        while let Some(op) = match self.current().kind {
            TokenKind::Plus => Some(BinOp::Add),
            TokenKind::Minus => Some(BinOp::Sub),
            _ => None,
        } {
            self.advance_pos();
            let rhs = self.parse_postfix()?;
            let span = self
                .shared
                .arena
                .span(lhs)
                .merge(self.shared.arena.span(rhs));
            lhs = self.shared.arena.push(Node::BinOp { op, lhs, rhs }, span);
        }
        self.depth -= 1;
        Ok(lhs)
    }

    /// Primary expression + post-primary `.field` chaining. Used as the
    /// operand parser for the `+`/`-` chain in `parse_expr` so RHS doesn't
    /// recurse into more arithmetic.
    fn parse_postfix(&mut self) -> ParseResult<NodeId> {
        self.depth += 1;
        if self.depth > MAX_PARSE_DEPTH {
            return Err(self.error(ParseErrorKind::NestingTooDeep));
        }
        let mut result = self.parse_primary()?;
        // Post-primary `.field` chaining, e.g. `grammar_config(base).extras`.
        while self.at(TokenKind::Dot) {
            let start = self.shared.arena.span(result);
            self.advance_pos();
            let field = self.expect_name()?;
            result = self
                .shared
                .arena
                .push(Node::FieldAccess { obj: result, field }, start.merge(field));
        }
        self.depth -= 1;
        Ok(result)
    }

    fn parse_primary(&mut self) -> ParseResult<NodeId> {
        let start = self.span();
        // Builtin keywords are contextual: they act as keywords only when
        // followed by `(`. Otherwise they are identifiers so grammars can
        // use names like `import`, `field`, `for`, etc. as rules.
        let next_lparen = self.next_is(TokenKind::LParen);
        let kw = self.current().kind;
        match kw {
            TokenKind::KwSeq | TokenKind::KwChoice if next_lparen => {
                let seq = kw == TokenKind::KwSeq;
                self.parse_variadic(start, |r| Node::SeqOrChoice { seq, range: r })
            }
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
                        if self.at(TokenKind::RParen) {
                            break;
                        }
                        self.parse_expr()?;
                        got += 1;
                        if self.eat(TokenKind::Comma).is_none() {
                            break;
                        }
                    }
                    return Err(self.err_arg_count(TokenKind::KwBlank, 0, got, start));
                }
                let end = self.expect(TokenKind::RParen)?;
                Ok(self.shared.arena.push(Node::Blank, start.merge(end)))
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
            TokenKind::KwInherit | TokenKind::KwImport if next_lparen => {
                self.parse_module_path(start, kw, kw == TokenKind::KwImport)
            }
            TokenKind::KwAppend if next_lparen => self.parse_append(start),
            TokenKind::KwGrammarConfig if next_lparen => self.parse_grammar_config(start),
            TokenKind::KwFor if next_lparen => self.parse_for(start),
            TokenKind::Ident => self.parse_ident_expr(start),
            _ if kw.is_keyword() => self.parse_ident_expr(start),
            TokenKind::StringLit => {
                self.advance_pos();
                Ok(self
                    .shared
                    .arena
                    .push(Node::StringLit, start.strip_quotes()))
            }
            TokenKind::IntLit(n) => {
                self.advance_pos();
                Ok(self.shared.arena.push(Node::IntLit(i64::from(n)), start))
            }
            TokenKind::RawStringLit { hash_count } => {
                self.advance_pos();
                Ok(self
                    .shared
                    .arena
                    .push(Node::RawStringLit { hash_count }, start))
            }
            TokenKind::Minus => {
                self.advance_pos();
                // parse_postfix (not parse_expr) so unary `-` binds tighter
                // than the trailing `+`/`-` chain: `-X + Y` is `(-X) + Y`.
                let inner = self.parse_postfix()?;
                Ok(self
                    .shared
                    .arena
                    .push(Node::Neg(inner), start.merge(self.shared.arena.span(inner))))
            }
            TokenKind::At => {
                // `@<primary>` - computed-name rule ref; valid in rule-set
                // macro bodies, resolved by expand_macro_calls.
                self.advance_pos();
                let inner = self.parse_postfix()?;
                Ok(self.shared.arena.push(
                    Node::SymRef { expr: inner },
                    start.merge(self.shared.arena.span(inner)),
                ))
            }
            TokenKind::LBracket => self.parse_delimited(
                start,
                TokenKind::RBracket,
                Node::List,
                Self::parse_expr_with_cfg,
            ),
            TokenKind::LParen => {
                self.parse_delimited(start, TokenKind::RParen, Node::Tuple, Self::parse_expr)
            }
            TokenKind::LBrace => self.parse_object(start),
            _ => Err(self.error(ParseErrorKind::ExpectedExpression)),
        }
    }

    fn parse_variadic(
        &mut self,
        start: Span,
        make: impl FnOnce(ChildRange) -> Node,
    ) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        let range = self.comma_sep_children(&[], TokenKind::RParen, Self::parse_expr_with_cfg)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self.shared.arena.push(make(range), start.merge(end)))
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
        Ok(self.shared.arena.push(make(inner), start.merge(end)))
    }

    /// Parse `grammar_config(module_expr, field_name)`.
    fn parse_grammar_config(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(TokenKind::KwGrammarConfig, 2, 0, start));
        }
        let module = self.parse_expr()?;
        if self.at(TokenKind::RParen) {
            return Err(self.err_arg_count(TokenKind::KwGrammarConfig, 2, 1, start));
        }
        self.expect(TokenKind::Comma)?;
        let field_span = self.expect_name()?;
        let field_name = self.ctx.text(field_span);
        let field = ConfigField::try_from(field_name)
            .ok()
            .filter(|f| {
                !matches!(
                    f,
                    ConfigField::Language | ConfigField::Inherits | ConfigField::Flags
                )
            })
            .ok_or_else(|| {
                ParseError::new(
                    ParseErrorKind::UnknownGrammarField(field_name.to_string()),
                    field_span,
                )
            })?;
        self.expect_close_args(TokenKind::KwGrammarConfig, 2, start)?;
        let end = self.expect(TokenKind::RParen)?;
        Ok(self
            .shared
            .arena
            .push(Node::GrammarConfig { module, field }, start.merge(end)))
    }

    /// Parse a two-argument builtin: `name(first, second)`.
    fn parse_binary<T>(
        &mut self,
        start: Span,
        name: TokenKind,
        parse_first: fn(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<(T, NodeId, Span)> {
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
        let (name, content, end) =
            self.parse_binary(start, TokenKind::KwField, Self::expect_name)?;
        Ok(self
            .shared
            .arena
            .push(Node::Field { name, content }, start.merge(end)))
    }

    fn parse_alias(&mut self, start: Span) -> ParseResult<NodeId> {
        let (content, target, end) =
            self.parse_binary(start, TokenKind::KwAlias, Self::parse_expr)?;
        Ok(self
            .shared
            .arena
            .push(Node::Alias { content, target }, start.merge(end)))
    }

    fn parse_prec(&mut self, start: Span, kind: PrecKind) -> ParseResult<NodeId> {
        let kw = match kind {
            PrecKind::Default => TokenKind::KwPrec,
            PrecKind::Left => TokenKind::KwPrecLeft,
            PrecKind::Right => TokenKind::KwPrecRight,
            PrecKind::Dynamic => TokenKind::KwPrecDynamic,
        };
        let (value, content, end) = self.parse_binary(start, kw, Self::parse_expr)?;
        Ok(self.shared.arena.push(
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
            Ok(this.expect_string()?.strip_quotes())
        })?;
        Ok(self
            .shared
            .arena
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
            .shared
            .arena
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
        let id = self.shared.arena.push(
            Node::ModuleRef {
                import: is_import,
                path: path_span.strip_quotes(),
                module: None,
            },
            start.merge(end),
        );
        self.ctx.module_refs.push(id);
        if !is_import {
            if let Some(first) = self.ctx.inherit_ref {
                return Err(self.dup_err(
                    ParseErrorKind::MultipleInherits,
                    start.merge(end),
                    self.shared.arena.span(first),
                ));
            }
            self.ctx.inherit_ref = Some(id);
        }
        Ok(id)
    }

    fn parse_append(&mut self, start: Span) -> ParseResult<NodeId> {
        let (left, right, end) = self.parse_binary(start, TokenKind::KwAppend, Self::parse_expr)?;
        Ok(self
            .shared
            .arena
            .push(Node::Append { left, right }, start.merge(end)))
    }

    fn parse_for(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        self.expect(TokenKind::LParen)?;
        let bindings = self.comma_sep(TokenKind::RParen, |this| {
            let name = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok(Param {
                name,
                ty: this.parse_type()?.0,
            })
        })?;
        self.expect(TokenKind::RParen)?;
        self.check_duplicate_names(&bindings, |p| p.name, ParseErrorKind::DuplicateBinding)?;
        if bindings.is_empty() {
            return Err(self.error(ParseErrorKind::EmptyForBindings));
        }
        if bindings.len() > u8::MAX as usize {
            return Err(self.error(ParseErrorKind::TooManyBindings));
        }
        self.expect(TokenKind::KwIn)?;
        let iterable = self.parse_expr()?;
        let for_id = self.shared.pools.push_for(ForConfig { bindings, iterable });
        self.expect(TokenKind::LBrace)?;
        let (body, end) = stack_scope!(self.locals, |_saved| {
            let config = self.shared.pools.get_for(for_id);
            for (i, &Param { name, ty }) in config.bindings.iter().enumerate() {
                self.locals
                    .push((name, LocalBinding::ForBinding(for_id, ty, i as u8)));
            }
            let body = self.parse_expr()?;
            let end = self.expect(TokenKind::RBrace)?;
            ParseResult::Ok((body, end))
        })?;
        Ok(self
            .shared
            .arena
            .push(Node::For { for_id, body }, start.merge(end)))
    }

    fn parse_ident_expr(&mut self, start: Span) -> ParseResult<NodeId> {
        let span = self.expect_ident()?;
        let name = self.ctx.text(span);
        let name_id = if let Some(&(_, binding)) = self
            .locals
            .iter()
            .rev()
            .find(|(s, _)| self.ctx.text(*s) == name)
        {
            let node = match binding {
                LocalBinding::MacroParam(ty, index) => Node::MacroParam { ty, index },
                LocalBinding::ForBinding(for_id, ty, index) => {
                    Node::ForBinding { for_id, ty, index }
                }
            };
            self.shared.arena.push(node, span)
        } else {
            self.shared
                .arena
                .push(Node::Ident(IdentKind::Unresolved), span)
        };
        let mut id = name_id;
        loop {
            if self.at(TokenKind::Dot) {
                self.advance_pos();
                let field = self.expect_name()?;
                id = self
                    .shared
                    .arena
                    .push(Node::FieldAccess { obj: id, field }, start.merge(field));
            } else if self.at(TokenKind::ColonColon) {
                self.advance_pos();
                let member_span = self.expect_ident()?;
                if self.at(TokenKind::LParen) {
                    // h::macro_name(args): pack [obj, name, ...args] into one ChildRange
                    let member_id = self
                        .shared
                        .arena
                        .push(Node::Ident(IdentKind::Unresolved), member_span);
                    self.advance_pos();
                    let range = self.comma_sep_children(
                        &[id, member_id],
                        TokenKind::RParen,
                        Self::parse_expr,
                    )?;
                    let end = self.expect(TokenKind::RParen)?;
                    id = self
                        .shared
                        .arena
                        .push(Node::QualifiedCall(range), start.merge(end));
                    break;
                }
                id = self.shared.arena.push(
                    Node::QualifiedAccess {
                        obj: id,
                        member: member_span,
                    },
                    start.merge(member_span),
                );
            } else if self.at(TokenKind::LParen) {
                if !matches!(
                    self.shared.arena.get(id),
                    Node::Ident(IdentKind::Unresolved)
                ) {
                    return Err(self.error(ParseErrorKind::ExpectedMacroName));
                }
                self.advance_pos();
                let args = self.comma_sep_children(&[], TokenKind::RParen, Self::parse_expr)?;
                let end = self.expect(TokenKind::RParen)?;
                id = self.shared.arena.push(
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
        item: fn(&mut Self) -> ParseResult<NodeId>,
    ) -> ParseResult<NodeId> {
        self.advance_pos();
        let range = self.comma_sep_children(&[], close, item)?;
        let end = self.expect(close)?;
        Ok(self.shared.arena.push(make(range), start.merge(end)))
    }

    fn parse_object(&mut self, start: Span) -> ParseResult<NodeId> {
        self.advance_pos();
        let fields = self.comma_sep(TokenKind::RBrace, |this| {
            let key = this.expect_ident()?;
            this.expect(TokenKind::Colon)?;
            Ok((key, this.parse_expr()?))
        })?;
        let mut seen = FxHashMap::with_capacity_and_hasher(fields.len(), FxBuildHasher);
        for &(span, _) in &fields {
            let key = self.ctx.text(span);
            if let Some(first_span) = seen.insert(key, span) {
                return Err(self.dup_err(
                    ParseErrorKind::DuplicateObjectKey(key.to_string()),
                    span,
                    first_span,
                ));
            }
        }
        let end = self.expect(TokenKind::RBrace)?;
        let range = self
            .shared
            .pools
            .push_object(fields)
            .ok_or_else(|| self.error(ParseErrorKind::TooManyChildren))?;
        Ok(self
            .shared
            .arena
            .push(Node::Object(range), start.merge(end)))
    }

    /// Parse comma-separated children directly into a [`ChildRange`].
    /// `prefix` is prepended to the resulting range (e.g. `[obj, name]`
    /// for a qualified call).
    ///
    /// Uses a scratch buffer as a stack to avoid per-call heap allocation.
    /// Nested calls push above the parent's items and truncate back, so
    /// interleaving is impossible.
    fn comma_sep_children(
        &mut self,
        prefix: &[NodeId],
        close: TokenKind,
        mut parse_item: impl FnMut(&mut Self) -> ParseResult<NodeId>,
    ) -> ParseResult<ChildRange> {
        stack_scope!(self.scratch, |saved| {
            self.scratch.extend_from_slice(prefix);
            while !self.at(close) {
                let id = parse_item(self)?;
                self.scratch.push(id);
                if self.eat(TokenKind::Comma).is_none() {
                    break;
                }
            }
            self.shared
                .pools
                .push_children(&self.scratch[saved..])
                .ok_or_else(|| self.error(ParseErrorKind::TooManyChildren))
        })
    }

    fn comma_sep<T>(
        &mut self,
        close: TokenKind,
        mut parse_item: impl FnMut(&mut Self) -> ParseResult<T>,
    ) -> ParseResult<Vec<T>> {
        let mut items = Vec::new();
        while !self.at(close) {
            items.push(parse_item(self)?);
            if self.eat(TokenKind::Comma).is_none() {
                break;
            }
        }
        Ok(items)
    }
}

pub type ParseResult<T> = Result<T, super::ParseError>;

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize, Error)]
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
    #[error("expected a top-level item (grammar, rule, override, let, macro, rules, external)")]
    ExpectedItem,
    #[error("unknown type '{0}'")]
    UnknownType(String),
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
    #[error("only identifiers can be used as macro names")]
    ExpectedMacroName,
    #[error("duplicate parameter name '{0}'")]
    DuplicateParameter(String),
    #[error("duplicate binding name '{0}'")]
    DuplicateBinding(String),
    #[error("only one grammar block is allowed")]
    DuplicateGrammarBlock,
    #[error("grammar block must have a 'language' field")]
    MissingLanguageField,
    #[error("only one inherit() call is allowed per grammar")]
    MultipleInherits,
    #[error("expression nesting too deep (maximum {MAX_PARSE_DEPTH})")]
    NestingTooDeep,
    #[error("too many elements (maximum 65535)")]
    TooManyChildren,
    #[error("too many bindings (maximum 255)")]
    TooManyBindings,
    #[error("for-loop requires at least one binding")]
    EmptyForBindings,
    #[error("{}", format_arg_count(*.expected, .name, *.got))]
    WrongArgumentCount {
        name: TokenKind,
        expected: u8,
        got: usize,
    },
    #[error("expected 'cfg' inside attribute, got '{0}'")]
    ExpectedCfgKeyword(String),
    #[error("`#[cfg(...)]` is not allowed on a grammar block")]
    CfgOnGrammarBlock,
    #[error("rule-set macro body may only contain rule declarations")]
    RuleSetBodyRequiresRuleDecl,
    #[error("rule-set macro body must declare at least one rule")]
    EmptyRuleSetMacroBody,
    #[error("qualified calls to rule-set macros are not supported")]
    QualifiedRuleSetCall,
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
