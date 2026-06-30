# RIES v11.7.3 patch manifest

This patch zip contains only files changed or added since v11.7.2.

## Replace / add

- `RIES/ries.html`
- `ries-script.js`
- `ries_inline.js`
- `assets/ries-harddb-v11_7_3-level4.js`
- `assets/ries-harddb-v11_7_3-stats.json`
- `tools/build_harddb_v11_7_3.py`
- `tools/test_ries_v11_7_3_harddb.js`
- `tools/test_ries_v11_7_packaging.js`
- `tools/test_ries_v11_7_1_latex_comprehensive.js`
- `changelog/RIES_v11.7.3_CHANGELOG.md`
- `docs/RIES_v11.7.3_HARDDB_PRUNE_REPORT.md`
- `docs/RIES_v11.7.3_PATCH_MANIFEST.md`

## Optional cleanup in a full checkout

The active v11.7.3 runtime no longer references these old split harddb assets:

- `assets/ries-harddb-v11_6-level4.js`
- `assets/ries-harddb-v11_6-level5.js`
- `assets/ries-harddb-v11_6-stats.json`

They are not included in this patch zip because the user requested a patch package containing only files modified since v11.7.2.
