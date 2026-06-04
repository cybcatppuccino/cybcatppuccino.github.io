# RIES v10.8.1 Changelog

Patch release for the v10.8 low-precision constant database UI.

## Fixed

- Fixed LaTeX backslash escaping in constant-database result formulas.
  - `x \approx ...` now remains `\approx` instead of displaying as plain `approx`.
  - `\frac{...}{...}`, `\sqrt{...}`, `\left(...\right)`, `\log`, `\exp`, `\alpha`, `\Gamma`, and `\zeta` are now preserved in JS string literals.
  - Thin-space LaTeX commands such as `\,` and substring-result spacing `\;\text{...}\;` are also preserved.
- Updated RIES page and asset cache-busting labels to v10.8.1.
- Synchronized the external and inline RIES scripts so the same LaTeX escaping fix is present in both copies.

## Tests

- Added `tools/test_ries_v10_8_1_latex_escape.js` covering the reported cases:
  - `x \approx \frac{3\,c}{16}`
  - `x \approx 6 + c + c^{2}`
  - `x \approx \log\left(6 + c + c^{2}\right)`
- Re-ran the v10.8 constant database, relation-family, integer-validation, packed-DB, sorting, and substring-budget regression tests under v10.8.1.
