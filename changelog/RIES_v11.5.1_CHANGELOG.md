# RIES v11.5.1 Changelog

## Summary

v11.5.1 refines the v11.5 hypergeometric database integration for better web loading behavior and clearer level semantics.

## Changes

- Moved the filtered hard-constant database matcher to RIES level 5 only.
  - Level 4 no longer loads or scans harddb.
  - Level 6 and above also skip harddb, so the deeper level can spend its budget on pFq/full searches and other modules.
- Split the hypergeometric pFq database into three incremental lazy chunks:
  - level 4: common 2F1/3F2 rows and stage-1 multipliers;
  - level 5: additional 4F3/5F4 rows and stage-2 multipliers;
  - level 6: remaining pFq rows and stage-3 multipliers.
- Changed higher-level pFq search to be cumulative:
  - level 5 searches both level-4 and level-5 H rows against both stage-1 and stage-2 multipliers;
  - level 6 searches all loaded H rows against all loaded multiplier families.
  - This avoids missing 2F1/3F2 values that require more complex gamma/pi/radical factors.
- Reduced web-facing pFq load size at lower levels:
  - old v11.5 single pFq asset: about 18 MB uncompressed JS;
  - v11.5.1 level-4 first pFq load: about 0.44 MB;
  - cumulative level-5 pFq load: about 5.67 MB;
  - cumulative level-6 pFq load: about 17.83 MB.
- Improved package-load progress text for large database assets.
  - The loader can now show an estimated expanded JS size when the network/content length is compressed or otherwise smaller than the actual evaluated script size.
  - This avoids misleading cases where the UI appears to cap at a smaller number such as 4 MB while the evaluated package is closer to 10 MB.
- Added v11.5.1 hypdata stats and smoke tests.

## Files changed

- `ries-script.js`
- `ries_inline.js`
- `ries.html`
- `assets/ries-hypdata-v11_5_1-level4.js`
- `assets/ries-hypdata-v11_5_1-level5.js`
- `assets/ries-hypdata-v11_5_1-level6.js`
- `assets/ries-hypdata-v11_5_1-stats.json`
- `tools/build_hypdata_v11_5_1.py`
- `tools/test_ries_v11_5_1_hypdata.js`
- `tools/test_ries_v11_5_1_packaging.js`

## Tests run

```bash
node --check ries-script.js
node --check ries_inline.js
node tools/test_ries_v11_5_1_packaging.js
node tools/test_ries_v11_5_1_hypdata.js
```
