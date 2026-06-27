# Gardner MiniChess Lab — v12.1 patch

This patch is intended to be applied over a complete v12 installation. It contains only code/documentation files changed for v12.1; keep your existing `data/`, PGN files, and generated tablebase folders in place.

## Install

1. Back up the v12 folder.
2. Unzip this patch at the project root and allow files to overwrite existing files.
3. Keep the existing `tools/gardner_tablebase/tables/` directory or equivalent tablebase deployment path.
4. Restart the local server and force-refresh the browser.

v12.1 uses the engine identity `Orion JS 12.1` and a new persistent analysis-cache key, so stale v12 cached lines are not reused.

## What changed in v12.1

### TB-assisted mate length display

- Normal search results that reach an exact 2–4 piece tablebase position along the PV are now post-processed with the exact DTM.
- Instead of leaving these lines as a generic `+220.00` / `-220.00` WDL score, the UI can display a mate-distance upper bound such as `#18 · TB bound`.
- The bound is intentionally conservative: it is used for display and ordering clarity, but it is not marked as a fully verified mate unless the complete PV actually replays to checkmate.

### Exact tablebase score text

- Exact tablebase wins/losses no longer fall back to `TB win` / `TB loss` when the PV is too short to replay to mate.
- If exact DTM is available, the score text shows the mate distance and marks that the result comes from tablebase help.

### Stability and crash-risk reduction

- WDL warm-up is now non-duplicative after completion and yields between blocks, reducing startup pressure in VS Code/local Chromium webviews.
- Individual corrupt/missing WDL blocks no longer abort the whole warm-up pass. Search still safely uses only blocks already present in memory.
- Continuous worker analysis now applies the same TB-DTM annotation as the play worker without changing the core alpha-beta result.

## Changed files

- `js/engine/engine.js`
- `js/engine/tablebase.js`
- `js/engine/worker.js`
- `js/engine/play-worker.js`
- `js/engine/analysis-cache.js`
- `js/ui/analysis-panel.js`
- `index.html`
- `README.md`
- `CHANGELOG.md`
- `VERSION`
- `tests/v12_1-tablebase-dtm-bound-tests.mjs`
