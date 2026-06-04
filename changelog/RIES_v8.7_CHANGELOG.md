# RIES v8.7 changelog

Built on v8.6 without removing existing features.

## Integer database simplification

- Added recursive display simplification for database / shortform integer expressions.
- Pure multiplicative factors are now compressed even when they appear inside a top-level offset or inside a rounding wrapper.
- Examples now simplify as expected:
  - `35^7·35+9` → `35^8+9`
  - `9^49·27-5` → `3^101-5`
  - `16/4·2^35+1` → `2^37+1`

## Continue responsiveness

- Converted the <=10^8 extra exhaustive pass into an async, sliced pass with UI yields.
- Added progress phase labels for compact DB build, direct/reverse, rational, rational-power, and structured-backup sub-passes.
- Stop keeps returning the best rows accumulated so far.

## Integer caching

- Added a target-level integer cache for factorization, static precomputed shortforms, structured database rows, and exact shortform rows.
- Repeated runs for the same integer/settings reuse completed stages rather than recomputing them.

## Tests

- Added `tools/test_ries_v8_7_startup.js`.
- Added `tools/test_ries_v8_7_integer_simplify_cache_progress.js`.
