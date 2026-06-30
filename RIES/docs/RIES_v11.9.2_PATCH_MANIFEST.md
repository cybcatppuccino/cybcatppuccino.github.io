# RIES v11.9.2 patch manifest

This package contains only files changed or added relative to v11.9.1.

## Main runtime changes

- `RIES/ries.html`
  - Visible version and cache buster updated to v11.9.2.
- `ries-script.js`
- `ries_inline.js`
  - Hypergeom loader switched to `RIES_HYPDATA_V1192_CHUNKS` and new v11.9.2 assets.
  - `RIES_HYPDATA_TOTAL_ROWS` updated to 136170.
  - Real-target hypergeom search now consumes `realCompB64` so scalar projections `H`, `Re(H)`, and `Im(H)` are distinguishable.
  - Result rendering now emits valid LaTeX for `\operatorname{Re}` and `\operatorname{Im}` projections.
  - Source labels support bitmask source combinations including `data.zip 2F1 grid`.

## New/changed hypergeom database assets

- `assets/ries-hypdata-v11_9_2-level4.js`
- `assets/ries-hypdata-v11_9_2-level5.js`
- `assets/ries-hypdata-v11_9_2-level6.js`
- `assets/ries-hypdata-v11_9_2-stats.json`

The new `data.zip` 2F1 grid rows are placed in level4. Since RIES loads hypdata cumulatively, level4, level5, and level6 searches all use those rows. The real-search table includes scalar projections for `H`, `Re(H)`, and `Im(H)` after rational exclusion and scalar de-duplication.

## Build/report/test files

- `tools/build_hypdata_v11_9_2.py`
- `tools/test_ries_v11_9_2_hypdata.js`
- `docs/RIES_v11.9.2_HYPDATA_REPORT.md`
- `docs/RIES_v11.9.2_PATCH_MANIFEST.md`

## Final counts

- Final H rows: 136170
- Final real-search scalar rows: 205890
- Final complex-search rows: 136170
- Row-level rational exclusions: 593
- Row-level duplicates removed: 2297
- Scalar projection rational exclusions: 5452
- Scalar projection duplicates removed: 3991

Validation run:

```bash
node --check ries-script.js
node --check ries_inline.js
node tools/test_ries_v11_9_2_hypdata.js
```
