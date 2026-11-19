// The states after `a ( x` and `b ( x` have the same items, so they're merged, and the merged
// state reduces `list` on `if`, which follows it after `a`. After `b`, those reductions don't lead
// to a shift of `if`, so in `b ( x if ;` the parser reads `if` as the `list`'s second identifier.
// That `list` starts in the merged state after `(`, so replacing `b` with `a` must not reuse it.

export default grammar({
    name: 'reuse_node_containing_merged_state_keyword',

    word: $ => $.identifier,

    rules: {
        source_file: $ => choice(seq('a', $.pair, 'if', ';'), seq('b', $.pair, ';')),
        pair: $ => seq('(', optional($.list)),
        list: $ => seq($.identifier, optional($.identifier)),
        identifier: _ => /[a-z]+/,
    },
});
