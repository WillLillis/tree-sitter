use std::num::NonZeroU32;
use std::rc::Rc;

use rustc_hash::FxHashMap;

use super::ModuleId;
use super::ast::Span;

/// Interned string id. Indexes into [`StringPool::entries`].
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Str(NonZeroU32);

impl Str {
    /// Raw index into [`StringPool::entries`].
    #[inline]
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0.get()
    }
}

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

    #[must_use]
    pub fn entry(&self, id: Str) -> &StrEntry {
        // Safety: id was produced by intern_span/intern_owned which return
        // sequential indices into self.entries.
        unsafe { self.entries.get_unchecked(id.0.get() as usize) }
    }

    /// Resolve a `Str` whose `Source` entries are known to reference `source`
    #[must_use]
    pub fn resolve_local<'a>(&'a self, id: Str, source: &'a str) -> &'a str {
        match self.entry(id) {
            StrEntry::Source(span, _) => span.resolve(source),
            StrEntry::Owned(s) => s,
            StrEntry::Unreachable => unreachable!(),
        }
    }
}
