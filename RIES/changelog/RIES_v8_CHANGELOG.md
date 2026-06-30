# RIES v8 changelog

## Integer continue/debug fixes

- Enabled Stop during the full integer pipeline, including the factorization phase.
- Added per-input caches for integer factorization, static shortform rows, dynamic database rows, and exact shortform rows by effort.
- Continue now reuses cached lower-effort results and searches only the newly requested effort when possible.
- Cached rows are merged by candidate/value/form metadata so improved higher-scoring results update the displayed table without duplicating old rows.

## L-function matching

- Reworked the L-function matcher into an async incremental scanner.
- Initial decimal runs use bounded slices; Continue raises the L-function effort together with the RIES level and resumes from the previous scan state.
- Rational matches are scanned first, so the `x^i*pi^j/L0 in Q` check appears quickly.
- Quadratic matching has a fast surd path for `alpha^2 in Q` before falling back to the full quadratic catalog.
- Log matching keeps the high-precision extras `log(log(2))`, `log(log(3))`, `log(Gamma(1/3))`, and `log(Gamma(1/4))` for higher precision / higher effort.

## Modular-form display

- L-function formulas now use concise `f`, for example `L(f,1)` and `L(f,2)`.
- Result rows include a separate modular-form column in `N.k.#` form, for example `23.2.5`.
- Result rows include a MathJax-rendered q-expansion column.

## Tests

- Added `tools/test_lfunc_v8.js`.
- Added `tools/lfunc_v8_test_results.md`; current smoke suite passes 6/6 representative L-function cases.
