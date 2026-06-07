# RIES v11.7.3 changelog

v11.7.3 is a harddb cleanup and matching-strategy update built on v11.7.2.

## Harddb database pruning

Removed the overlapping/generated harddb categories requested for deletion:

| Category | Removed rows |
|---|---:|
| low-height hypergeometric pFq | 3,048 |
| Euler beta integral fast | 2,555 |
| incomplete beta integral fast | 30,000 |
| beta logarithmic integral fast | 15,000 |
| gamma log-laplace integral fast | 221 |
| rational Mellin integral fast | 5,500 |
| **Total removed** | **56,324** |

The active harddb now contains 23,608 rows from the remaining categories.

## Harddb matching strategy

- Replaced the old depth-4/depth-5 row split with one pruned lazy-loaded asset: `assets/ries-harddb-v11_7_3-level4.js`.
- Depth 4 now loads all remaining harddb rows and uses simple comparison constants.
- Depth 5 scans the same rows with core comparison constants.
- Depth 6 scans the same rows with extended comparison constants.
- Old v11.6 split assets are no longer referenced by `ries-script.js`.
- Harddb result source marker changed to `harddb-v11.7.3-pruned`.

## Parameters panel

- Updated the harddb UI wording so it no longer says “low-height 20%” or “remaining rows”.
- Added a depth-6 harddb toggle and budget field.
- Updated harddb default stage budgets to 1s / 5s / 50s for depths 4 / 5 / 6.
- Updated harddb rational multiplier height default to 20; stage caps are still applied internally so depth 4 remains simple.
- Confirmed new settings are read by `readSettings()` and included in the existing cache key through `hardDbOptions` and `stageBudgets`.

## LaTeX and display

- Kept v11.7.1/v11.7.2 LaTeX normalization active for the rebuilt harddb formulas.
- Added full active-harddb LaTeX scanning in the v11.7.3 harddb test.
- Closed-form harddb outputs continue to use `x \approx ...`; equation-mode equality remains separate.

## Tests

Added `tools/test_ries_v11_7_3_harddb.js` and updated version assertions in the v11.7 packaging/LaTeX regression tests.
