//! Interned string pool, shared across parse / expand / lower so that all
//! stages can intern names (source-span references or computed strings)
//! through a single uniform handle.

use std::num::NonZeroU32;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use super::ModuleId;
use super::ast::Span;

/// Interned string id. Indexes into [`StringPool::entries`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Str(pub NonZeroU32);

pub enum StrEntry {
    Unreachable,
    Source(Span, ModuleId),
    Owned(Rc<str>),
}

pub struct StringPool {
    pub entries: Vec<StrEntry>,
    /// Dedup table for `Owned` entries. The `Rc<str>` key shares its allocation
    /// with the matching `StrEntry::Owned`.
    owned_lookup: FxHashMap<Rc<str>, Str>,
}

impl Default for StringPool {
    fn default() -> Self {
        Self {
            entries: vec![StrEntry::Unreachable],
            owned_lookup: FxHashMap::default(),
        }
    }
}

impl StringPool {
    pub fn intern_span(&mut self, span: Span, mod_id: ModuleId) -> Str {
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Source(span, mod_id));
        id
    }

    pub fn intern_owned(&mut self, s: &str) -> Str {
        if let Some(&id) = self.owned_lookup.get(s) {
            return id;
        }
        let owned: Rc<str> = Rc::from(s);
        // Safety: entries always starts with one sentinel, so len() >= 1.
        let id = Str(unsafe { NonZeroU32::new_unchecked(self.entries.len() as u32) });
        self.entries.push(StrEntry::Owned(Rc::clone(&owned)));
        self.owned_lookup.insert(owned, id);
        id
    }

    /// Get the raw entry. Resolution to `&str` lives on the caller since
    /// `Source` entries may reference an in-progress module whose source
    /// isn't yet in the `previous` modules slice.
    #[must_use]
    pub fn entry(&self, id: Str) -> &StrEntry {
        // Safety: id was produced by intern_span/intern_owned which return
        // sequential indices into self.entries.
        unsafe { self.entries.get_unchecked(id.0.get() as usize) }
    }

    /// Resolve a `Str` whose `Source` entries are known to reference
    /// `source` (caller asserts the local-module invariant).
    #[must_use]
    pub fn resolve_local<'a>(&'a self, id: Str, source: &'a str) -> &'a str {
        match self.entry(id) {
            StrEntry::Source(span, _) => span.resolve(source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }
}
