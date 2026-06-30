# RIES v7 change log

## Integer search / shortform

- Restored the stronger historical integer database families from the v6.2–v6.5 line, including power, factorial, binomial/coefficient, structured denominator, fractional-scale, signed pair, product-pair, and factorial-power templates.
- Kept the restored database responsive by running the large template scans in asynchronous chunks with hard time budgets and frequent UI yields. The search status keeps updating during `Checking precomputed and structured integer database…` instead of locking the page.
- Large integers with 16 or more digits are no longer excluded from stronger database and exact shortform attempts. v7 still applies strict budgets so the page remains interruptible.
- Fallback search now uses a richer structured pool with powers, factorials, and binomial coefficients, not only simple powers.
- Power-ratio fallback can compact denominators and constant offsets using the precomputed <=100000 shortform database.
  - Denominators are kept clean: no `/`, `floor`, `ceil`, or `round` is used inside compacted denominator expressions.
  - Constant offsets may use the fuller <=100000 database, including short forms with division or floor/ceil when they are actually shorter.
- Removed `round(...)` from generated shortform/fallback expressions. Rounded divisions are classified as exact, floor, or ceil; ambiguous nearest-only cases are dropped instead of displayed as `round`.
- Added a final result filter so selected shortform rows containing `round(` are not surfaced.
- Continue-at-effort now spends a little more time on integer shortform/database slices, aiming for better results while staying bounded and responsive.

## Validation notes

Local checks run for this package:

- `node --check ries-script.js`
- `node --check ries_inline.js`
- Stress tests on representative integers including 10-digit, 14-digit, 16-digit, 17-digit, and 20-digit values.
- Verified compact fallback formatting for the example pattern `floor(5^20/(4^7-90))+71·(5!-4)` with correct LaTeX floor output.
