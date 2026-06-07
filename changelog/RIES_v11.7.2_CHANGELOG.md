# RIES v11.7.2 changelog

Date: 2026-06-07

## Focus

v11.7.2 is a conservative sign-equivalence cleanup pass for the v11.7 integral/sum database.  It does not add new mathematical families or multiplier stages.

## Integral/sum database sign dedupe

- Added a build-time sign-equivalence pass to `tools/build_intsumdb_v11_7.py`.
- The pass first scans value16 absolute-value groups for rows whose stored values differ only by an overall sign.
- It removes a row only when a family-specific canonicalization proves the two records are the same mathematical kernel up to a global sign.
- Nontrivial identities, different kernels, different summation shifts, and sign pairs whose equivalence is not structurally proved are retained.

## Conservative rules currently enabled

1. `unit-beta/rational-log sign convention`: bridges
   - `BETA_LOG_PLUS_MINUS/unit_beta_log_pm` rows using `(-log x)^m log(1-x)^n`, and
   - `RATIONAL_LOG_BETA/unit_rational_log_beta` rows using `log(x)^m log(1-x)^n`,
   when the denominator/factor kernel is exactly the same.  The sign difference is then only the convention `log(x) = -(-log x)`.
2. `trig-Fourier reflection x->pi-x`: canonicalizes `TRIG_RATIONAL_FOURIER/trig_rational_fourier` pairs where all nonzero `cos x` coefficients flip together under `x -> pi-x`; odd `k` accounts for the global negative sign of `cos(kx)`.

## Counts

- Pre-dedupe integral/sum rows: 36,685
- Post-dedupe integral/sum rows: 36,443
- Rows removed: 242
- Mixed-sign absolute-value groups scanned: 742
- Rows in mixed-sign groups: 1,484
- Mixed-sign groups retained as distinct/non-proved: 500

### Removed by rule

- trig-Fourier reflection x->pi-x: 170
- unit-beta/rational-log sign convention: 72

### Removed by family

- RATIONAL_LOG_BETA/unit_rational_log_beta: 72
- TRIG_RATIONAL_FOURIER/trig_rational_fourier: 170

## Additional LaTeX cleanup

- Extended the v11.7.1 LaTeX normalizer to remove `\binom{...}{...}^{1}`, `(1)^n`, and trailing unit factors created by neutral denominator powers in binomial/inverse-binomial sums.

## Assets and UI

- Rebuilt `assets/ries-intsumdb-v11_7-level4.js`, `level5.js`, `level6.js`, and stats.
- Updated RIES visible version and cache-busters to v11.7.2.
- Updated `constantDbSource` for integral/sum hits to `intsumdb-v11.7.2`.
- Synced `ries-script.js` and `ries_inline.js`.

## Tests

- Updated integral/sum packaging/runtime/LaTeX tests for the new row counts.
- Added `tools/test_ries_v11_7_2_sign_dedupe.js` to ensure removed rows do not remain in chunk blobs and kept representatives do remain.
