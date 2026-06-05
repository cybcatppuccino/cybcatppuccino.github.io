# RIES v11.1 changelog

## Decimal recognition fixes

- Fixed a constant-database algebraic-ratio false positive where very small ratios could satisfy a residual-only monomial such as `α^3 = 0` and then be displayed as an algebraic multiplier of a database constant.
- Added a root-nearness validation step for algebraic ratio candidates: a polynomial relation must have at least two nonzero terms and an actual real root close to the observed ratio before it can be emitted.
- Algebraic-ratio rows now use the validated nearby root to compute/display the predicted value instead of echoing the raw ratio.

## Precision handling

- Added a dedicated `typedInputPrecisionForDouble(...)` cap for modules that convert decimal input to JavaScript `Number`.
- Double-based matchers now treat 16–20 significant input digits as the double-safe ceiling instead of over-tightening tolerance based on digits that are no longer available after `Number` conversion.
- Higher-precision Decimal-backed modules keep using their own higher-precision paths.

## UI responsiveness

- Added `constantDbRowsAsync(...)`, a cooperative constant-database scan used by `solve`.
- The solve pipeline now yields between constant-database scan batches, updates status/progress during the scan, and renders partial candidates so the progress bar and SO(4) cube animation can keep updating.

## Tests

- Updated the v11 constant-database regression test for v11.1 cache/version strings.
- Added regression checks for the `0.03876817960292` / `α^3 = 0` false positive and for the double-precision cap.
- Added an async cooperative constant-database test.
