//! Levenshtein-based suggestion helper for misspelled identifier diagnostics.

/// Pick the candidate closest to `target` if any is within a small edit
/// distance. Returns `None` if no candidate is close enough, ignoring exact
/// matches (callers should have looked those up first).
pub(super) fn suggest_name<'a>(
    target: &str,
    candidates: impl Iterator<Item = &'a str>,
) -> Option<String> {
    // ~1 edit per 3 chars, minimum 1. Catches typos like `prec_lef` ->
    // `prec_left` without suggesting unrelated short names.
    let threshold = target.chars().count() / 3 + 1;
    candidates
        .map(|c| (levenshtein(target, c), c))
        .filter(|&(d, _)| d > 0 && d <= threshold)
        // Tie-break on the name so equal-distance candidates are deterministic
        // even when callers iterate hash-ordered keys.
        .min_by_key(|&(d, c)| (d, c))
        .map(|(_, c)| c.to_string())
}

fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.is_empty() {
        return b.len();
    }
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        curr[0] = i;
        for j in 1..=b.len() {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", "abc"), 0);
        assert_eq!(levenshtein("", "abc"), 3);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("prec_lef", "prec_left"), 1);
        assert_eq!(levenshtein("café", "cafe"), 1);
    }

    #[test]
    fn suggest_picks_closest_within_threshold() {
        let candidates = ["prec_left", "prec_right", "seq", "choice"];
        assert_eq!(
            suggest_name("prec_lef", candidates.iter().copied()).as_deref(),
            Some("prec_left"),
        );
    }

    #[test]
    fn suggest_returns_none_when_no_candidate_is_close() {
        let candidates = ["seq", "choice", "prec"];
        assert_eq!(
            suggest_name("totally_unrelated", candidates.iter().copied()),
            None,
        );
    }

    #[test]
    fn suggest_skips_exact_matches() {
        // Callers should have already looked up exact matches; suggesting the
        // input back is unhelpful.
        let candidates = ["foo", "bar"];
        assert_eq!(suggest_name("foo", candidates.iter().copied()), None);
    }

    #[test]
    fn suggest_empty_candidates() {
        assert_eq!(suggest_name("foo", std::iter::empty()), None);
    }

    #[test]
    fn suggest_breaks_ties_lexicographically() {
        // Equal-distance candidates must resolve deterministically: callers feed
        // hash-ordered keys, so the smaller name must win regardless of order.
        assert_eq!(
            suggest_name("abc", ["abe", "abd"].iter().copied()).as_deref(),
            Some("abd"),
        );
        assert_eq!(
            suggest_name("abc", ["abd", "abe"].iter().copied()).as_deref(),
            Some("abd"),
        );
    }

    #[test]
    fn suggest_short_names_have_tight_threshold() {
        // Threshold for 2-char target is 2/3 + 1 = 1; "ab" -> "xy" is distance 2.
        let candidates = ["xy"];
        assert_eq!(suggest_name("ab", candidates.iter().copied()), None);
    }
}
