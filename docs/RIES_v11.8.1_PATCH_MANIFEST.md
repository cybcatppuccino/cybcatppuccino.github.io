# RIES v11.8.1 patch manifest

This patch is intended to be applied on top of the complete v11.8 build.

Modified runtime files:

- `index.html`
- `assets/site.css`
- `ries.html`
- `ries-script.js`
- `ries_inline.js`

Modified/new test and documentation files:

- `tools/test_ries_v11_7_1_latex_comprehensive.js`
- `tools/test_ries_v11_7_3_harddb.js`
- `tools/test_ries_v11_7_4_database_transforms.js`
- `tools/test_ries_v11_7_packaging.js`
- `tools/test_ries_v11_8_lfunctions.js`
- `tools/test_ries_v11_8_1_lfunc_sort_and_ui.js`
- `changelog/RIES_v11.8.1_CHANGELOG.md`
- `docs/RIES_v11.8.1_PATCH_MANIFEST.md`

Validation run:

```bash
node --check ries-script.js
node --check ries_inline.js
node tools/test_ries_v11_7_1_latex_comprehensive.js
node tools/test_ries_v11_7_3_harddb.js
node tools/test_ries_v11_7_4_database_transforms.js
node tools/test_ries_v11_7_packaging.js
node tools/test_ries_v11_8_lfunctions.js
node tools/test_ries_v11_8_1_lfunc_sort_and_ui.js
```
