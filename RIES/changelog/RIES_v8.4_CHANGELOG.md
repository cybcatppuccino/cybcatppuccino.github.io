# RIES v8.4 changelog

Built on v8.3 without rolling back any v8.x features.

## Fixed

- Decimal/L-function result display now keeps the accumulated result set through the final pass. Earlier RIES/log/algebraic/L-function matches are no longer discarded by the final low-precision grouping slice when later L-function quadratic algebraic matches arrive.
- Added a final **Sort by confidence** control. The default table preserves discovery/group order, and the button reorders all current results by a confidence-first heuristic: verified digits first, then lower-degree/smaller-height algebraic forms, simpler L-rational forms, and lower-complexity expressions.
- Collapses mathematically competing L-function explanations using the same modular form and same L-value, so direct rational, quadratic-algebraic, and log-derived L matches for the same underlying identity do not flood the table. The best-verified/simplest representative is retained.
- Result table layout no longer gives `formula`, `form (N.k.#)`, and `q-expansion` their own narrow columns. They now live inside the `value / root` cell with internal line breaks; q-expansion display is larger and easier to read.
- L-function rational reciprocal formulas are normalized to avoid denominator-with-fraction displays such as `π/(2/3·L)`, preferring simplified forms such as `3π/(2·L)`.

## Preserved

- v8.3 static deployment protections, `.nojekyll`, lazy integer shortform database loading, L-function search, algebraic reconstruction, log-combination matching, high-precision evaluation, integer factorization, and shortform search remain in place.

## Added tests

- `tools/test_ries_v8_4_startup.js` checks v8.4 static startup wiring and sort-button attachment.
- `tools/test_ries_v8_4_results_display.js` checks the new value/root cell layout, final result tools, and L-equivalence deduplication.
- `tools/test_ries_v8_4_lfunc_formula_simplify.js` checks simplified reciprocal L-function rational formulas.
