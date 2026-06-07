# RIES v12.0.1 patch report

## Scope

This patch is based on v12.0 and produces a complete v12.0.1 package. The main runtime/test assets that are active in this version have been renamed to versionless filenames and versionless global names, while the visible application version is now `v12.0.1`.

The requested `2F1/3F2`, `4F3/5F4`, and full-data split is applied to the hypergeometric pFq database chunks used by the `hypData` module. The L-function module remains covered by the consolidated L-function regression tests.

## Active hypergeometric pFq chunk split

Active files:

- `assets/ries-hypdata-level4.js`
- `assets/ries-hypdata-level5.js`
- `assets/ries-hypdata-level6.js`
- `assets/ries-hypdata-stats.json`

The cumulative loading behavior is now:

| Runtime level | Newly loaded chunk | Cumulative families | Newly loaded rows | Real scalar rows | Complex rows |
|---:|---|---|---:|---:|---:|
| 4 | level4 | `2F1`, `3F2` | 29,618 | 36,156 | 29,618 |
| 5 | level5 | `2F1`, `3F2`, `4F3`, `5F4` | 36,403 | 54,493 | 36,403 |
| 6 | level6 | all remaining families | 70,149 | 115,241 | 70,149 |

Full cumulative row count at level6 remains `136,170`.

Level6's newly loaded families are:

`0F1`, `1F0`, `1F1`, `1F2`, `2F2`, `2F3`, `3F3`, `3F4`, `4F4`, `5F5`, `5F6`, `6F5`, `7F6`, `8F7`.

## Active versioned names removed

The currently used lazy runtime assets no longer use versioned filenames:

- `assets/ries-harddb-level4.js`
- `assets/ries-hypdata-level4.js`
- `assets/ries-hypdata-level5.js`
- `assets/ries-hypdata-level6.js`
- `assets/ries-intsumdb-level4.js`
- `assets/ries-intsumdb-level5.js`
- `assets/ries-intsumdb-level6.js`

The active global chunk arrays are now:

- `window.RIES_HARDDB_CHUNKS`
- `window.RIES_HYPDATA_CHUNKS`
- `window.RIES_INTSUMDB_CHUNKS`

The active consolidated test files are also versionless:

- `tools/ries_test_utils.js`
- `tools/test_ries_all.js`
- `tools/test_ries_packaging_startup.js`
- `tools/test_ries_latex_rendering.js`
- `tools/test_ries_database_modules.js`
- `tools/test_ries_constdb_lfunc_log.js`
- `tools/test_ries_precision_integer_sorting.js`

## Progress bar corrections

The package loader now displays real byte progress when `Content-Length` is available. If the server does not provide `Content-Length`, it uses the exact packaged `expectedBytes` value as the displayed upper bound. The progress percentage is mapped exactly to the configured progress interval using `base + span * frac`, and the status display no longer floors in-progress values to 2%.

Updated expected byte upper bounds:

| Asset | expectedBytes |
|---|---:|
| `assets/ries-harddb-level4.js` | 1,142,631 |
| `assets/ries-hypdata-level4.js` | 3,721,007 |
| `assets/ries-hypdata-level5.js` | 5,916,621 |
| `assets/ries-hypdata-level6.js` | 13,862,408 |
| `assets/ries-intsumdb-level4.js` | 2,309,630 |
| `assets/ries-intsumdb-level5.js` | 10,766,885 |
| `assets/ries-intsumdb-level6.js` | 608,570 |

The L-function transformed-DB scan progress now stays within its intended progress slice, `67.5%` to `72%`, instead of over-reporting past the displayed upper bound.

## Old files to delete when applying this over an old tree

These files are not present in the v12.0.1 package. If applying changes manually over v12.0 or earlier, delete them:

- `assets/ries-harddb-v11_4_1-filtered.js`
- `assets/ries-harddb-v11_4_1-filtered-stats.json`
- `assets/ries-harddb-v11_6-level4.js`
- `assets/ries-harddb-v11_6-level5.js`
- `assets/ries-harddb-v11_6-stats.json`
- `assets/ries-harddb-v11_7_3-level4.js`
- `assets/ries-harddb-v11_7_3-stats.json`
- `assets/ries-hypdata-v11_5.js`
- `assets/ries-hypdata-v11_5-stats.json`
- `assets/ries-hypdata-v11_5_2-level4.js`
- `assets/ries-hypdata-v11_5_2-level5.js`
- `assets/ries-hypdata-v11_5_2-level6.js`
- `assets/ries-hypdata-v11_5_2-stats.json`
- `assets/ries-hypdata-v11_9_2-level4.js`
- `assets/ries-hypdata-v11_9_2-level5.js`
- `assets/ries-hypdata-v11_9_2-level6.js`
- `assets/ries-hypdata-v11_9_2-stats.json`
- `assets/ries-intsumdb-v11_7-level4.js`
- `assets/ries-intsumdb-v11_7-level5.js`
- `assets/ries-intsumdb-v11_7-level6.js`
- `assets/ries-intsumdb-v11_7-stats.json`
- `assets/shortform100k_v10_3_stats.json`
- `assets/shortform100k_v10_4_stats.json`
- `tools/ries_v12_test_utils.js`
- `tools/test_ries_v12_0_all.js`
- `tools/test_ries_v12_0_packaging_startup.js`
- `tools/test_ries_v12_0_latex_rendering.js`
- `tools/test_ries_v12_0_database_modules.js`
- `tools/test_ries_v12_0_constdb_lfunc_log.js`
- `tools/test_ries_v12_0_precision_integer_sorting.js`
- `tools/test_harddb_v11_4_assets.js`
- `tools/test_harddb_v11_4_direct_runtime.js`
- `tools/test_lfunc_v7_3.js`
- `tools/test_lfunc_v8.js`
- `tools/lfunc_v7_3_generated_test_data.json`
- `tools/lfunc_v7_3_test_results.md`
- `tools/lfunc_v8_1_test_results.md`
- `tools/lfunc_v8_test_results.md`
- `tools/build_harddb_v11_6_split.py`
- `tools/build_harddb_v11_7_3.py`
- `tools/build_hypdata_v11_5.py`
- `tools/build_hypdata_v11_5_1.py`
- `tools/build_hypdata_v11_5_2.py`
- `tools/build_hypdata_v11_9_2.py`
- `tools/build_intsumdb_v11_7.py`

## Validation run

Commands run successfully:

```bash
node --check ries-script.js
node --check ries_inline.js
node --check tools/ries_test_utils.js
node --check tools/test_ries_*.js
node tools/test_ries_all.js
```

Final result:

```text
PASS RIES consolidated test suite (5 files)
```
