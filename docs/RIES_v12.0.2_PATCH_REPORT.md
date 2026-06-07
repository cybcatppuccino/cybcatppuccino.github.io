# RIES v12.0.2 patch report

This patch is based on v12.0.1 and keeps the visible/runtime version at `v12.0.2`. It focuses on safe loading and execution speedups without reducing the set of searchable records, target transforms, multipliers, or level gating.

## Search capability preservation

The hypergeometric and integral/sum databases were split into search-index packages and display-metadata packages. The search-index packages retain every field used for matching: stored Float64 values, row maps, component codes, complexity arrays, multiplier values, and multiplier complexity arrays. Display-only strings such as formula text, LaTeX, high-precision decimal strings, row ids, family labels, and source/status labels moved to metadata packages and are loaded only after a hit needs formatting.

Smoke regression against the v12.0.1 runtime was run on representative hypergeometric and integral/sum targets. The hypergeometric top candidate, score, and error were identical. The integral/sum matched rows, scores, and errors were identical; only the display string changed because v12.0.2 cleans unit/zero artifacts in the formula text.

## Loading changes

- HardDB, hypergeometric, and integral/sum index loads are prestarted in parallel when their modules are enabled.
- Hypergeometric complex-target search no longer decodes the real scalar projection arrays before searching complex arrays.
- Hypergeometric and integral/sum search progress now uses time-sliced progress/yield checks rather than a fixed high-frequency yield cadence.
- L-function `Decimal(L.value)` objects are cached per precision/configuration so repeated monomial/rational passes do not repeatedly construct the same Decimal values.
- L-function progress/yield loops were converted to time-sliced checks.
- Package-loading progress continues to map bytes to the exact configured progress span. When `Content-Length` is unavailable, the configured `expectedBytes` is used as the real fallback upper bound rather than an artificial percentage ceiling.

## Index/meta split

| Asset | v12.0.1 monolithic bytes | v12.0.2 index bytes | v12.0.2 metadata bytes | Initial search-load change |
|---|---:|---:|---:|---:|
| hypdata level4 | 3,721,007 | 1,515,316 | 2,206,054 | -59.3% |
| hypdata level5 | 5,916,621 | 2,097,796 | 3,819,188 | -64.5% |
| hypdata level6 | 13,862,408 | 4,207,848 | 9,654,934 | -69.6% |
| intsum level4 | 2,309,630 | 146,437 | 2,163,560 | -93.7% |
| intsum level5 | 10,766,885 | 638,865 | 10,128,389 | -94.1% |
| intsum level6 | 608,570 | 152,505 | 456,430 | -74.9% |

Gzip-level reference sizes for the new split packages:

| Asset | gzip bytes |
|---|---:|
| hypdata level4 index / meta | 767,107 / 523,831 |
| hypdata level5 index / meta | 1,160,022 / 770,699 |
| hypdata level6 index / meta | 2,407,102 / 1,954,344 |
| intsum level4 index / meta | 77,747 / 297,005 |
| intsum level5 index / meta | 337,363 / 1,274,541 |
| intsum level6 index / meta | 66,817 / 57,945 |

## Integral/sum display cleanup

The intsum display path now normalizes generated unit/zero artifacts without changing the stored numeric value or search key. Examples:

- `e^{0-x+0x^2}(1+0x+0x^2)` displays as `e^{-x}`.
- `x^1`, `(1-x)^1`, `(1+...)^1` display without the redundant exponent.
- `cos(1 x)` displays as `cos(x)`.
- `1 cos x` displays as `cos x`.
- scalar multipliers `1`, `-1`, and factors with exponent `0` are suppressed or converted to standard signs in LaTeX.

## Changed/new files in this patch

- `ries.html`
- `ries-script.js`
- `ries_inline.js`
- `assets/ries-hypdata-level4.js`
- `assets/ries-hypdata-level4-meta.js`
- `assets/ries-hypdata-level5.js`
- `assets/ries-hypdata-level5-meta.js`
- `assets/ries-hypdata-level6.js`
- `assets/ries-hypdata-level6-meta.js`
- `assets/ries-intsumdb-level4.js`
- `assets/ries-intsumdb-level4-meta.js`
- `assets/ries-intsumdb-level5.js`
- `assets/ries-intsumdb-level5-meta.js`
- `assets/ries-intsumdb-level6.js`
- `assets/ries-intsumdb-level6-meta.js`
- `tools/test_ries_packaging_startup.js`
- `tools/test_ries_database_modules.js`
- `tools/test_ries_all.js`
- `docs/RIES_v12.0.2_PATCH_REPORT.md`
- `changelog/RIES_v12.0.2_CHANGELOG.md`
- `changelog/RIES_CHANGELOG.md`

## Validation run

The following syntax checks passed:

```bash
node --check ries-script.js
node --check ries_inline.js
node --check tools/test_ries_all.js
node --check tools/test_ries_*.js
```

The following consolidated suites passed when run individually:

```bash
node tools/test_ries_packaging_startup.js
node tools/test_ries_latex_rendering.js
node tools/test_ries_database_modules.js
node tools/test_ries_constdb_lfunc_log.js
node tools/test_ries_precision_integer_sorting.js
```

Final suite counts:

- Packaging/startup: 5 tests
- LaTeX/rendering: 5 tests
- Database modules: 9 tests
- ConstDB/log/L-functions: 6 tests
- Precision/integer/sorting: 6 tests
