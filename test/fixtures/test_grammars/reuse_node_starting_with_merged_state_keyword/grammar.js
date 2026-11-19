// The states after `a (` and `b (` have the same items, so they're merged, and the merged state
// reduces `pair` on `if`, which starts a `cond` after `a`. After `b`, those reductions don't lead
// to a shift of `if`, so in `b ( if ; x` the parser reads `if` as an identifier. Replacing `a` with
// `b` in `a ( if ; x` must give the same tree, although the `cond` that starts with `if` can be
// reused right after the `(`.

export default grammar({
    name: 'reuse_node_starting_with_merged_state_keyword',

    word: $ => $.identifier,

    rules: {
        source_file: $ => choice(seq('a', $.pair, $.cond), seq('b', $.pair, ';', $.identifier)),
        pair: $ => seq('(', optional($.identifier)),
        cond: $ => seq('if', ';', $.identifier),
        identifier: _ => /[a-z]+/,
    },
});
