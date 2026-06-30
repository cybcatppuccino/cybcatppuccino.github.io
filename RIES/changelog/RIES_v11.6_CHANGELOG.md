# RIES v11.6 changelog

## Scope

This release continues from v11.5.2 and keeps default search behavior compatible unless the user changes Parameters.

## Changes

- Fixed hypergeometric pFq LaTeX escaping in explanation, formula, value HTML, and copy LaTeX paths so `\\approx`, inline delimiters, spacing commands, and fraction commands survive JavaScript string parsing.
- Split the hard constant database into two lazy chunks:
  - depth 4 loads a low-height representative subset of about 20% of harddb rows while covering every stored category.
  - depth 5 loads the remaining rows cumulatively and scans the full harddb set.
- Added hard constant database Parameters for candidate count, stage time budget, rational multiplier height, stored-parameter height cap, depth toggles, and relation-pass toggles.
- Exposed per-module candidate counts and practical stage time limits in Parameters for the major search modules.
- Moved the external/worker QS or ECM handoff control into the Integer search module.
- Extended log-combination relation search to cover `log|x|`, raw `x`, and `log|log|x||`, all enabled by default and exposed in Parameters.
- Updated cache keys so parameter changes invalidate affected cached results.
- Added UPDATE_GUIDELINES rules for Parameters: defaults must preserve previous behavior, new modules should expose reasonable interfaces, and any new parameter must be connected to both search logic and cache keys.

## Validation

- `node --check ries-script.js`
- `node --check ries_inline.js`
- `node tools/test_ries_v11_6_packaging.js`
- `node tools/test_ries_v11_6_parameters.js`
- `node tools/test_ries_v11_6_harddb.js`
- `node tools/test_ries_v11_6_hypdata_latex.js`
