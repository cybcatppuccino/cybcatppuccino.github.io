# RIES v11.7 changelog

## Integral/sum database module

- Added an independent **Integral/sum database** detector (`intsumdb-v11.7`) modeled after the v11.5.2 hypergeometric lazy database matcher.
- The new matcher tests real decimal targets against identities of the form `x ≈ M·S`, where `S` is a stored integral/sum/series value and `M` is a staged constant multiplier.
- Reused the hypergeometric-style multiplier strategy:
  - level 4: simple rational, π, radical, and low-height prime-power constants;
  - level 5: the level-4 set plus core Gamma quotients;
  - level 6: the full set plus deeper Gamma/extended constants.

## Database split

- Packaged 36,685 v11.7 candidate rows into lazy assets:
  - `assets/ries-intsumdb-v11_7-level4.js`: 6,831 low-height rows (`height_score <= 2`, about 18.6% of the database) plus 1,200 stage-1 multipliers.
  - `assets/ries-intsumdb-v11_7-level5.js`: the remaining 29,854 rows plus 5,300 stage-2 multipliers, making the row table complete at level 5.
  - `assets/ries-intsumdb-v11_7-level6.js`: no extra rows; adds 9,500 deeper stage-3 multipliers so level 6 mainly increases constant complexity.
- Added `assets/ries-intsumdb-v11_7-stats.json` to record row counts, family counts, asset byte sizes, and the selection policy.
- Added `tools/build_intsumdb_v11_7.py` so the assets can be rebuilt from the candidate JSONL package.

## UI and solver integration

- Bumped the RIES page and script cache key from v11.6.4 to v11.7.
- Added a new Parameters panel block for the integral/sum module with module toggle, candidate limit, per-depth budgets, depth controls, and multiplier-family controls.
- Integrated the new lazy detector into the solve pipeline after hypergeometric pFq detection and before L-function detection.
- Added a separate result category, confidence sorting hooks, progress messages, source marker, and test hook for the integral/sum module.

## LaTeX handling

- Stored the source LaTeX strings directly in escaped blob form and renders them through the existing `latex-render` path with `\(...\)` delimiters.
- Added multiplier composition rendering that preserves `\frac`, `\sqrt`, `\Gamma`, `\pi`, `\left`, `\right`, and thin-space commands after JavaScript string decoding.
- Normalized generated sign runs such as `+-`, `-+`, `++`, and `--` in packaged row LaTeX to avoid confusing output for negative or complex parameters.
- Added runtime and static tests that inspect every new family type and verify that complex formulas keep their backslash escapes and do not collapse into malformed text.

## Tests

- Added `tools/test_ries_v11_7_packaging.js`.
- Added `tools/test_ries_v11_7_intsumdb.js`.
- Added `tools/test_ries_v11_7_intsumdb_latex.js`.
