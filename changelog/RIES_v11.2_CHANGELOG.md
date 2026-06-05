# RIES v11.2 changelog

Focused constant-database performance update on top of v11.1.4.

## Changes

- Added a constant-database-only fast floating LLL reducer (`constDbFastLLLReduce`).
  - Other modules still use the existing global LLL / exact LLL strategy.
  - Search forms and acceptance checks are unchanged; candidates still pass the same residual, height, degree, and root-nearness validation.
  - The fast reducer uses lazy/local Gram-Schmidt recomputation instead of rebuilding the full table after every row operation.
  - Exact BigInt-rational LLL remains as a bounded fallback when the fast pass does not validate a relation.
- Kept bounded safeguards for the constant-database fast LLL path to avoid oversized floating quotient row updates monopolizing the UI thread.
- Updated constant database budgets:
  - level 4: 10 seconds
  - level 5: 30 seconds
  - level 6: 100 seconds
- Tightened priority algebraic sweep deadline checks so expired constant-database budgets do not continue scanning in the synchronous test path.

## Validation

- Syntax checks pass for `ries-script.js` and `ries_inline.js`.
- Constant database regression smoke tests pass for:
  - `2.386110381167886 → exp((-12 + 4π + π²)/12)`
  - `-2.143596015846163 → α·π, α^3 + α + 1 = 0`
  - `0.03876817960292` no longer emits the previous `α^3 = 0` false positive.
- Low-precision regression test still passes.
- Microbenchmarks on constant-database LLL relation calls showed roughly 20× improvement on representative linear and algebraic LLL cases in the Node harness.
