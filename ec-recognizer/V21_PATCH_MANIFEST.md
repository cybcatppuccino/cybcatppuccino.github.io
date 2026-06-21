# V21 EC atlas patch, delta from V20

Apply this archive on top of the V20 EC atlas files.

## Modified files
- `ec-recognizer/app.js`
- `ec-recognizer/style.css`

## Added files
- `ec-recognizer/js/smoke_v21.mjs`
- `V21_PATCH_MANIFEST.md`
- `V21_OBSOLETE_FILES.md`

## Obsolete file
`ec-recognizer/data/curves_compact.json` is no longer used by the runtime code.
V20/V21 use `curve_shards/`, `search/`, `tau_index.json`, `plot_meta.json`, `top_points.json`, `tiles/`, and `sato_tate_groups.json` instead.
If your deployed directory still contains `ec-recognizer/data/curves_compact.json` from V19, it can be deleted safely. Leaving it in place does not affect correctness, but wastes about 19 MB and may slow deployment/upload.

## V21 changes
1. Selected marker replaced with a minimal rotating circular marker: one solid circular stroke plus four inward ticks separated by pi/2.
2. Initial camera now points near the zenith with about N=500 at the left/right screen edges on a typical landscape display.
3. Runtime no longer references `curves_compact.json`.
4. Rendering path now uses a tile-aware visible candidate pool instead of scanning all loaded points every frame.
5. Interaction rendering uses a fast dot path while dragging/pinching/zooming/traveling, then restores detailed stars when idle.
6. Detail rendering budget is lower and adaptive, reducing expensive per-star gradients/rays during broad views.
7. CSS containment added for canvas, HUD, tooltip and detail panels to reduce layout/paint invalidation.

## Tests run
- `node --check ec-recognizer/app.js`
- `node --check ec-recognizer/js/ec_core.js`
- `node ec-recognizer/js/smoke_v21.mjs`
- static HTTP fetch checks for key data files in the V19+V20+V21 overlay
