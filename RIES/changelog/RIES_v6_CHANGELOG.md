# RIES v6 changelog

## 1. Integer shortform search

- Kept the v5.9 exact-search pipeline: factorization, fast database match, digit-minimizing DB search, ratio search, rational-power search, and exact verification.
- Added a more diverse fallback layer instead of relying on one fixed `round/ceil/floor(A^B/C)+E` family.
- New fallback families include:
  - `A^B + C^D + E`
  - `A^B - C^D + E`
  - `A^B * C^D + E`
  - `floor((A/D)*B^C)+E`, `ceil((A/D)*B^C)+E`, and `round((A/D)*B^C)+E` with two-digit numerator/denominator scans
- Fallbacks are still exact: a candidate is only displayed after its BigInt value equals the target.
- Candidate selection now merges diverse fallback families with the older power-ratio fallback and deduplicates by expression family.

## 2. Algebraic approximation

- Added an exact finite-decimal algebraic check for low-height real and complex finite decimals:
  - real rational input gives a degree-1 polynomial when the height is meaningful;
  - complex rational input uses the exact quadratic over Q when the height is meaningful.
- Retained and improved the PSLQ/LLL-style ladder from v5.9: relation search uses multiple precision scales and exact BigInt residual verification.
- Kept conservative irreducibility filtering: rational-root checks, quadratic discriminant checks, quartic factor checks, and modular irreducibility certificates.
- Ranking remains biased toward simple low-height true relations, but nonminimal interesting approximations are still shown below the best candidates.

## 3. High-precision expression evaluation

- Added a local `assets/decimal.js` dependency and license file.
- Added a safe parser instead of `eval` for the high-precision path.
- Supports common expression syntax:
  - `+ - * / ^` and parentheses
  - implicit multiplication such as `6e` and `4pi`
  - factorial, e.g. `125!*7`
  - constants `pi`, `π`, `e`, `phi`, `i`
  - functions `sqrt`, `sin`, `cos`, `tan`, `log`/`ln`, `exp`, `abs`, `gamma`
- Exact BigInt is carried through integer-only expressions, so examples such as `3^257 * 6^2` and `125!*7` display the exact integer.
- Decimal or complex results display the first 100 significant digits first and can expand to 1000 significant digits.
- The browser-side gamma path supports positive integers, `1/2`, `1/3`, and `1/4`; this covers `gamma(1/4)` without requiring a heavy CAS backend.
- High-precision evaluation is bounded by a 5-second limit at the UI level.

## 4. Packaging

- Updated `RIES/ries.html`, `ries-script.js`, and `ries_inline.js` together.
- Updated visible version label to `v6`.
