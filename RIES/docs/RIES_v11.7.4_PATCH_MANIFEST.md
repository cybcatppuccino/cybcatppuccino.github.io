# RIES v11.7.4 patch manifest

This patch is intended to be applied on top of v11.7.3. It contains only files changed or added after v11.7.3.

## Modified runtime/UI files

- `RIES/ries.html`
- `ries-script.js`
- `ries_inline.js`

## Modified tests

- `tools/test_ries_v11_7_packaging.js`
- `tools/test_ries_v11_7_1_latex_comprehensive.js`
- `tools/test_ries_v11_7_3_harddb.js`

## Added tests

- `tools/test_ries_v11_7_4_database_transforms.js`

## Added docs

- `changelog/RIES_v11.7.4_CHANGELOG.md`
- `docs/RIES_v11.7.4_PATCH_MANIFEST.md`

## Unchanged assets

No database assets are changed in v11.7.4. The patch keeps using:

- `assets/ries-harddb-v11_7_3-level4.js`
- `assets/ries-hypdata-v11_5_2-level4.js`
- `assets/ries-hypdata-v11_5_2-level5.js`
- `assets/ries-hypdata-v11_5_2-level6.js`
- `assets/ries-intsumdb-v11_7-level4.js`
- `assets/ries-intsumdb-v11_7-level5.js`
- `assets/ries-intsumdb-v11_7-level6.js`

The behavioral change is in runtime target construction and result formatting.
