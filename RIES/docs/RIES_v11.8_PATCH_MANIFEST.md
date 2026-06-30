# RIES v11.8 patch manifest

This patch updates only the files needed for the modular-form L-function data merge and v11.8 browser release.

## Modified assets

- `assets/lfunctions-l2l4.js` — merged weight 1 and weight 3 L-function datasets into the existing browser L-function asset.
- `assets/newforms.js` — merged weight 1 and weight 3 forms into the homepage random-newform dataset.

## Modified runtime/UI files

- `ries-script.js`
- `ries_inline.js`
- `RIES/ries.html`
- `index.html`
- `assets/site.css`

## Test/changelog files

- `tools/test_ries_v11_8_lfunctions.js`
- Updated current RIES version assertions in v11.7 regression tests.
- `changelog/RIES_v11.8_CHANGELOG.md`
- `docs/RIES_v11.8_PATCH_MANIFEST.md`
