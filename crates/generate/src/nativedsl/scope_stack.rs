//! Generic scope stack used by the typechecker and lowerer.
//!
//! A scope stack is a `Vec` of `FxHashMap`s where lookups search from the
//! innermost (last) scope outward. Push/pop manage nesting for function
//! bodies, for-loop bodies, etc.

use rustc_hash::FxHashMap;

pub struct ScopeStack<'src, V: Copy> {
    scopes: Vec<FxHashMap<&'src str, V>>,
}

impl<'src, V: Copy> ScopeStack<'src, V> {
    #[inline]
    pub fn new() -> Self {
        Self {
            scopes: vec![FxHashMap::default()],
        }
    }

    #[inline]
    pub fn insert(&mut self, name: &'src str, val: V) {
        self.scopes.last_mut().unwrap().insert(name, val);
    }

    #[inline]
    pub fn get(&self, name: &str) -> Option<V> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    #[inline]
    pub fn push(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    #[inline]
    pub fn pop(&mut self) {
        debug_assert!(self.scopes.len() > 1, "cannot pop the base scope");
        self.scopes.pop();
    }
}
