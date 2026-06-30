# RIES v11.5 changelog

- Updated visible/cache-busting RIES labels from v11.4.3 to v11.5.
- Added an independent lazy-loaded hypergeometric pFq database matcher, modelled after the hard DB package pattern but kept separate from harddb/constantdb/L-function modules.
- Added `assets/ries-hypdata-v11_5.js` and `assets/ries-hypdata-v11_5-stats.json` as the compact merged hypergeometric database package. The asset merges the two supplied data ZIP sources, deduplicates by `MK`, stores 20-decimal display strings, and uses Float64 typed-array mirrors for fast low-precision matching.
- Added a staged pFq search over `x ≈ M·H` with at most five hypergeometric database results. Stage 1 targets 2F1/3F2 with simple factors, stage 2 adds 4F3/5F4 and core Gamma-style multiplier families, and stage 3 uses the full merged pFq database with broader multipliers.
- Added search-volume and complexity penalties so broad/deep hypergeometric matches do not automatically outrank simpler RIES, harddb, constantdb, or L-function rows.
- Added progress/status messages for lazy loading and staged hypergeometric scans.
- Added `tools/build_hypdata_v11_5.py` so the merged asset can be regenerated from `data1.zip` and `data2.zip` with the same v11.5 packing rules.
- Added v11.5 hypdata smoke tests covering startup lazy loading, asset shape, staged search hooks, category integration, and result formatting.

## Data notes

- Input rows scanned: 163,808.
- Successful source records: 110,969.
- Deduplicated hypergeometric rows: 109,738.
- Real-valued rows available for real target search: 36,874.
- Complex rows available for complex target search: 109,738.
- Tier 1 rows: 3,159; tier 2 rows: 36,407; tier 3 rows: 70,172.
- Multiplier rows: 16,000, split as 1,200 stage-1, 5,300 stage-2, and 9,500 stage-3 multipliers.
- A small number of corrupt/empty JSON block files from the source ZIPs were skipped and recorded in the stats JSON rather than silently discarded.

## Validation

- `node --check ries-script.js`
- `node --check ries_inline.js`
- `node tools/test_ries_v11_5_packaging.js`
- `node tools/test_ries_v11_5_hypdata.js`
