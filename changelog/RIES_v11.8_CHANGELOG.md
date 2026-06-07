# RIES v11.8 changelog

Focused modular-form L-function data update from v11.7.4.

## Changes

- Added weight 1 modular-form data to the shared browser L-function asset, with both `L(f,1/2)` and `L(f,1)` values.
- Added weight 3 modular-form data to the shared browser L-function asset, with both `L(f,1)` and `L(f,3/2)` values.
- Extended the RIES L-function matcher so weight 1/3 entries participate in the same rational, quadratic, and logarithmic comparison passes used for existing weight 2/4 entries.
- Updated L-function formula LaTeX rendering so half-integer labels display as `L(f,\tfrac{1}{2})` and `L(f,\tfrac{3}{2})`.
- Merged the new weight 1/3 forms into the homepage random newform dataset.
- Adjusted homepage random-newform selection to mildly favor lower levels while still sampling the full combined database.
- Simplified homepage weight 2 display to show only `L(f,1)`.
- Bumped the RIES page title, visible version, and script/data cache keys to v11.8.
- Changed the default auto depth stage budget from 5s to 8s.

## Tests

- Added `tools/test_ries_v11_8_lfunctions.js` to verify the new weight 1/3 data arrays, RIES entry construction, half-integer LaTeX, direct L-value matching, and the 8s default stage budget.
- Updated current version assertions in the existing v11.7 regression smoke tests to expect v11.8.
