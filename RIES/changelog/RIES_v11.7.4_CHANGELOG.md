# RIES v11.7.4 changelog

This patch extends the three large lazy database matchers introduced/refactored in v11.5-v11.7:

- Hypergeometric pFq database (`hypdata`).
- Integral/sum candidate database (`intsumdb`).
- Pruned hard-constant database (`harddb`).

## Changed

- Each database now compares stored formulas against three target views:
  - direct target: `x`;
  - exponential target: `exp(x)`, enabled only when `x <= 10`;
  - logarithmic target: `log|x|`.
- Result rows keep the transformed left-hand side explicit, for example:
  - `x \approx M\,S`;
  - `\exp(x) \approx M\,S`;
  - `\log\left|x\right| \approx M\,S`.
- The displayed prediction text now includes the transformed prediction and, when applicable, the implied value of `x`.
- Level/depth 4 and 5 time defaults for the three large databases were multiplied by 3:
  - harddb: `1000 -> 3000 ms`, `5000 -> 15000 ms`;
  - hypdata: `1000 -> 3000 ms`, `5000 -> 15000 ms`;
  - intsumdb: `1000 -> 3000 ms`, `5000 -> 15000 ms`.
- The level/depth 6 defaults remain `50000 ms`.

## LaTeX notes

- `exp(x)` is rendered as `\exp(x)` rather than `\exp\left(x\right)` because the older generic LaTeX simplifier can over-simplify the latter into `\expx` when it appears on the left-hand side of a sanitized relation.
- `log|x|` is rendered as `\log\left|x\right|`.
- Closed-form/database results continue to use `\approx`, not `=`.

## Tests

- Added `tools/test_ries_v11_7_4_database_transforms.js`.
- Updated current packaging/LaTeX/harddb regression tests for the v11.7.4 page version and tripled level/depth 4-5 budgets.
