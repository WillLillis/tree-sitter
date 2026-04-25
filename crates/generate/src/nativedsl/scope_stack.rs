//! Scope stack: nested `FxHashMap`s with innermost-first lookup.

use rustc_hash::FxHashMap;

#[derive(Clone)]
pub struct ScopeStack<'src, V: Copy> {
    scopes: Vec<FxHashMap<&'src str, V>>,
}

impl<'src, V: Copy> ScopeStack<'src, V> {
    pub fn new() -> Self {
        Self {
            scopes: vec![FxHashMap::default()],
        }
    }

    pub fn insert(&mut self, name: &'src str, val: V) {
        self.scopes.last_mut().unwrap().insert(name, val);
    }

    pub fn get(&self, name: &str) -> Option<V> {
        self.scopes.iter().rev().find_map(|s| s.get(name).copied())
    }

    pub fn push(&mut self) {
        self.scopes.push(FxHashMap::default());
    }

    pub fn pop(&mut self) {
        debug_assert!(self.scopes.len() > 1, "cannot pop the base scope");
        self.scopes.pop();
    }
}
