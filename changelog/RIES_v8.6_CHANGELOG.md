# RIES v8.6 changelog

Based on v8.5. This release keeps the v8.x feature set and focuses on decimal matching, L-function scheduling, LaTeX cleanup, and confidence ordering.

## L-function matching

- Low RIES levels now prioritize simple monomial shapes first: `x/L`, `x·π/L`, `x/(π·L)`, and `1/(x·L)`.
- Quadratic-algebraic L-function matching is restricted to very low height and simple monomial shapes at low effort, then expands with Continue levels.
- Log-product L-function matching starts with a tiny catalog and small denominator range at low effort; broader products, special constants, and high-precision log combinations unlock at higher effort.
- More L-function candidate rows are preserved from each category before final deduplication, reducing the chance that good early hits disappear from the visible result set.

## Display and sorting

- L-function LaTeX powers such as `2^(-2)`, `3^(2)`, and `5^(5/3)` now render as cleaner MathJax exponents like `2^{-2}`, `3^2`, and `5^{5/3}`.
- The confidence sort button now interleaves module results: each module's best row first, then each module's second-best row, and so on.
- Traditional RIES equations that already verify to the user-supplied precision, or one digit less, receive a sort boost so concise RIES forms are not buried by more complex-but-slightly-more-precise outputs.
- The sort preserves every accumulated row passed to the renderer; it only reorders rows.

## Tests

- Added `tools/test_ries_v8_6_startup.js`.
- Added `tools/test_ries_v8_6_lfunc_low_level_and_latex.js`.
- Added `tools/test_ries_v8_6_confidence_round_robin.js`.
