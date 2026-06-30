# RIES v11.2.1 Changelog

Scope: small constant-database patch on top of v11.2. Other modules' LLL/PSLQ strategies are unchanged.

## Constant database filtering

- Added a target-dependence filter for implicit subset relations. Rows such as
  `2*x + x*c - x/c = 0`, where the relation is equivalent to `x*(2+c-1/c)=0`
  and therefore constrains only `c`, are now rejected before result insertion.
- Excluded generated exact trigonometric/arctangent algebraic constants from the
  prioritized deep/LLL scans. They remain available to the cheap full-catalog
  direct pass, but no longer consume the most valuable deep-relation budget.

## Transform list

- Restricted constant-database comparison variants to exactly five transformed
  values, in this order:
  1. `x`
  2. `exp(x)`
  3. `log(x)` for positive real `x`
  4. `1/x`
  5. `x^2`

## Constant-database LLL safety

- Kept the v11.2 constant-DB-only fast floating LLL path.
- Added additional short-deadline guards so degree-3 polynomial-ratio LLL fallback
  does not overrun the browser UI time slices during deep scans.
- Added BigInt row-size safety checks to the constant-DB fast LLL reducer to avoid
  runaway row growth.
- Kept exact BigInt LLL fallback for offline/no-deadline calls, while avoiding it
  inside short UI slices.

## Budget and progress diagnostics

- Kept the v11.2 budgets:
  - level 4: 10 seconds
  - level 5: 30 seconds
  - level 6: 100 seconds
- Added internal deep-scan completion counters on the settings object:
  `_constantDbDeepDone`, `_constantDbDeepTotal`, `_constantDbDeepFrac`.

## Regression checks

- `2.386110381167886` still finds `x ≈ exp((-12 + 4·π + π^2)/12)`.
- `-2.143596015846163` still finds the cubic ratio result `α^3 + α + 1 = 0`.
- `0.03876817960292` does not reintroduce the previous `α^3 = 0` false positive.
- Low-precision regression tests still pass.
