use serde::Serialize;

/// Top-level type of any DSL expression. Splits into three structural classes:
/// first-class data values, module references, and non-bindable sentinels.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum Ty {
    Data(DataTy),
    Module(ModuleTy),
    /// Result of for-loop expansion. Inlines into the surrounding list; not
    /// a value, can't be bound to a variable.
    Spread,
    /// Syntactic tuple. Only valid as a direct element of a for-loop
    /// iterable's literal list. Not a value, can't be bound or returned.
    Tuple,
}

/// First-class data values: things that can be bound to a let, returned from
/// a macro, passed as an argument, or stored in a list/object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum DataTy {
    Scalar(ScalarTy),
    /// `list_t<X>` for `X` in `{Rule, Str, Int}`.
    List(ScalarTy),
    /// `list_t<list_t<X>>`. Triple nesting is rejected at parse time.
    ListList(ScalarTy),
    /// `obj_t<X>`. The inner type is structurally non-Object, so
    /// `obj_t<obj_t<X>>` is impossible.
    Object(InnerTy),
}

/// Atomic data types: `rule_t`, `str_t`, `int_t`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ScalarTy {
    Rule,
    Str,
    Int,
}

/// Element types allowed inside `obj_t<...>`. Mirrors [`DataTy`] minus the
/// `Object` variant, so `obj_t<obj_t<X>>` is structurally impossible.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum InnerTy {
    Scalar(ScalarTy),
    List(ScalarTy),
    ListList(ScalarTy),
}

/// Module types: results of `import(...)` / `inherit(...)` and the user-facing
/// `module_t` annotation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum ModuleTy {
    /// Result of `import(...)`. The `u8` indexes into the module list.
    Import(u8),
    /// Result of `inherit(...)`. The `u8` indexes into the module list.
    Grammar(u8),
    /// User-facing `module_t` annotation. Matches any concrete module via
    /// [`Ty::is_compatible`].
    Any,
}

impl Ty {
    /// Common scalar shorthands. Pattern matching still requires the long
    /// form, but value construction (errors, equality checks, mismatch)
    /// stays terse.
    pub const RULE: Self = Self::Data(DataTy::Scalar(ScalarTy::Rule));
    pub const STR: Self = Self::Data(DataTy::Scalar(ScalarTy::Str));
    pub const INT: Self = Self::Data(DataTy::Scalar(ScalarTy::Int));
    pub const LIST_RULE: Self = Self::Data(DataTy::List(ScalarTy::Rule));
    pub const LIST_STR: Self = Self::Data(DataTy::List(ScalarTy::Str));
    pub const LIST_INT: Self = Self::Data(DataTy::List(ScalarTy::Int));
    pub const LIST_LIST_RULE: Self = Self::Data(DataTy::ListList(ScalarTy::Rule));
    pub const LIST_LIST_STR: Self = Self::Data(DataTy::ListList(ScalarTy::Str));
    pub const LIST_LIST_INT: Self = Self::Data(DataTy::ListList(ScalarTy::Int));
    pub const ANY_MODULE: Self = Self::Module(ModuleTy::Any);

    /// True for `rule_t` or `str_t` (str widens to rule).
    #[must_use]
    pub const fn is_rule_like(self) -> bool {
        matches!(
            self,
            Self::Data(DataTy::Scalar(ScalarTy::Rule | ScalarTy::Str))
        )
    }

    #[must_use]
    pub const fn is_bindable(self) -> bool {
        matches!(self, Self::Data(_) | Self::Module(_))
    }

    /// Check if `self` is assignable to `expected`.
    ///
    /// Any module satisfies any module-typed expectation. The surface
    /// language only exposes `module_t` (= [`ModuleTy::Any`]) as an
    /// annotation, so specific-vs-specific module checks aren't reachable
    /// today; the broader rule keeps things simple and won't trip future
    /// macro signatures that declare specific module shapes.
    #[must_use]
    pub fn is_compatible(self, expected: Self) -> bool {
        match (self, expected) {
            (Self::Data(a), Self::Data(b)) => a.is_compatible(b),
            (Self::Module(_), Self::Module(_)) => true,
            _ => self == expected,
        }
    }

    /// True if this is any list type. Non-data types are never lists.
    #[must_use]
    pub const fn is_list(self) -> bool {
        matches!(self, Self::Data(d) if d.is_list())
    }

    /// Widen two types, delegating to [`DataTy::widen`] for data pairs.
    /// Equal non-data types widen to themselves; otherwise `None`.
    #[must_use]
    pub fn widen(self, other: Self) -> Option<Self> {
        match (self, other) {
            (Self::Data(a), Self::Data(b)) => a.widen(b).map(Self::Data),
            _ if self == other => Some(self),
            _ => None,
        }
    }

    /// Promote a data type to its list wrapper. Returns `None` for non-data
    /// or non-list-able types.
    pub fn to_list(self) -> Option<Self> {
        match self {
            Self::Data(d) => d.to_list().map(Self::Data),
            _ => None,
        }
    }

    /// Unwrap a list type to its element type. Returns `None` for non-data
    /// or non-list types.
    pub fn elem_type(self) -> Option<Self> {
        match self {
            Self::Data(d) => d.elem_type().map(Self::Data),
            _ => None,
        }
    }
}

impl DataTy {
    /// True if this is a list type at any depth.
    #[must_use]
    const fn is_list(self) -> bool {
        matches!(self, Self::List(_) | Self::ListList(_))
    }

    /// Check if `self` is assignable to `expected`.
    fn is_compatible(self, expected: Self) -> bool {
        self.widens_to(expected)
            || matches!(
                (self, expected),
                (Self::Object(a), Self::Object(b)) if Self::from(a).widens_to(Self::from(b))
            )
    }

    /// Whether `self` is assignable to `target` via equality or Str->Rule widening.
    fn widens_to(self, target: Self) -> bool {
        self == target
            || matches!(
                (self, target),
                (Self::Scalar(ScalarTy::Str), Self::Scalar(ScalarTy::Rule))
                    | (Self::List(ScalarTy::Str), Self::List(ScalarTy::Rule))
                    | (
                        Self::ListList(ScalarTy::Str),
                        Self::ListList(ScalarTy::Rule)
                    )
            )
    }

    /// Widen `self` to accommodate `other`. Returns the wider type, or `None`
    /// if the types are incompatible.
    #[must_use]
    fn widen(self, other: Self) -> Option<Self> {
        if other.is_compatible(self) {
            Some(self)
        } else if self.is_compatible(other) {
            Some(other)
        } else {
            None
        }
    }

    /// Promote to the list wrapper (e.g. `Scalar(Rule)` -> `List(Rule)`).
    /// Returns `None` for types that can't be list elements (objects, or
    /// list-of-list which would be triple-nested).
    const fn to_list(self) -> Option<Self> {
        Some(match self {
            Self::Scalar(s) => Self::List(s),
            Self::List(s) => Self::ListList(s),
            Self::ListList(_) | Self::Object(_) => return None,
        })
    }

    /// Unwrap a list type to its element type. Returns `None` for non-list types.
    const fn elem_type(self) -> Option<Self> {
        Some(match self {
            Self::List(s) => Self::Scalar(s),
            Self::ListList(s) => Self::List(s),
            Self::Scalar(_) | Self::Object(_) => return None,
        })
    }
}

impl From<InnerTy> for DataTy {
    fn from(i: InnerTy) -> Self {
        match i {
            InnerTy::Scalar(s) => Self::Scalar(s),
            InnerTy::List(s) => Self::List(s),
            InnerTy::ListList(s) => Self::ListList(s),
        }
    }
}

impl TryFrom<DataTy> for InnerTy {
    type Error = ();
    fn try_from(d: DataTy) -> Result<Self, ()> {
        Ok(match d {
            DataTy::Scalar(s) => Self::Scalar(s),
            DataTy::List(s) => Self::List(s),
            DataTy::ListList(s) => Self::ListList(s),
            DataTy::Object(_) => return Err(()),
        })
    }
}

impl From<DataTy> for Ty {
    fn from(d: DataTy) -> Self {
        Self::Data(d)
    }
}

impl From<InnerTy> for Ty {
    fn from(i: InnerTy) -> Self {
        Self::Data(DataTy::from(i))
    }
}

impl std::fmt::Display for Ty {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(d) => d.fmt(f),
            Self::Module(_) => f.write_str("module_t"),
            Self::Spread => f.write_str("spread_t"),
            Self::Tuple => f.write_str("tuple"),
        }
    }
}

impl std::fmt::Display for DataTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Scalar(s) => s.fmt(f),
            Self::List(s) => write!(f, "list_t<{s}>"),
            Self::ListList(s) => write!(f, "list_t<list_t<{s}>>"),
            Self::Object(i) => write!(f, "obj_t<{i}>"),
        }
    }
}

impl std::fmt::Display for ScalarTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Rule => "rule_t",
            Self::Str => "str_t",
            Self::Int => "int_t",
        })
    }
}

impl std::fmt::Display for InnerTy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        DataTy::from(*self).fmt(f)
    }
}
