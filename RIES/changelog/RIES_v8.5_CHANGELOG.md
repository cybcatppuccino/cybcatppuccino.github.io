# RIES v8.5 changelog

Based on v8.4.  This release keeps the v8.x feature set and focuses on the integer module.

## Fixed

- Large integer Continue no longer blocks on the multi-megabyte `shortform100k.js` table.  The table is still loaded for targets within the 100k precomputed range; larger integers go directly to the deterministic structured database and exact search.
- Strengthened mathematical equivalence keys for integer expressions:
  - rational multiples of powers such as `40/17·6^13`, `20/51·6^14`, and `15/68·72^6` collapse to one family;
  - floor/ceil forms with the same rational core and compensating outside offsets collapse to one family;
  - pure multiplicative forms such as `16/4·2^35` simplify to the lower-digit representative when possible.
- Integer shortform selection now respects larger result limits up to 20 instead of always truncating to five internal rows.

## Improved

- Added a bounded extra exact-search pass for integers up to `10^8` at higher Continue efforts, so small targets can spend extra time looking for truly shorter forms without unbounded runtime.
- Expanded large-integer structured database limits for `>=10^16` targets at higher efforts while keeping all scans deadline-checked and UI-yielding.
- Kept all previously discovered integer result groups when merging factorization, static/precomputed, database, and shortform phases.

## Tests

- Added `tools/test_ries_v8_5_startup.js`.
- Added `tools/test_ries_v8_5_integer_dedupe.js`.
