# Gardner MiniChess Lab v17 patch notes

v17 is intended to be applied over v16.1. It keeps the core Gardner rules and evaluation meaning intact, while fixing boot/cache behavior, making <=5-piece tablebase use lazy and browser-safe, and adding root-level tactical safety against short opponent mates.

## What changed

- The app always opens in Local mode. Previous Human-vs-AI / AI-vs-AI mode selection is not restored on page load.
- Persistent AI analysis caches are cleared at page boot.
- A separate lightweight current-game cache stores only the active local game tree and current node, so refreshing the page restores the board and variations without restoring AI state.
- Tablebase loading is now exact and lazy:
  - v17 uses only <=5-piece exact Gardner tablebase entries from `tools/gardner_tablebase/tables/manifest.json`;
  - no all-table WDL warmup happens at worker startup;
  - workers prefetch only the WDL block for the current relevant position/neighborhood;
  - missing tablebase blocks fall back to normal Orion search rather than breaking analysis.
- Root candidates now receive a short forced-mate safety check from the opponent's perspective. If a candidate allows a verified short mate, it is scored and displayed as a mate loss.
- Deep cached/resumed analysis with a very short, non-terminal PV is rejected so the worker must calculate again instead of presenting a stale 4-5 ply line at high depth.

## Changed or added files

```text
CHANGELOG.md
PATCH_FILE_LIST.txt
README.md
VERSION
app.js
index.html
js/engine/analysis-cache.js
js/engine/engine.js
js/engine/play-worker.js
js/engine/tablebase.js
js/engine/worker.js
tests/v11-efficiency-and-tablebase-tests.mjs
tests/v15_2-ui-and-move-buffer-tests.mjs
tests/v16-live-top3-info-tests.mjs
tests/v16_1-black-perspective-tests.mjs
tests/v17-state-tablebase-tactical-tests.mjs
tools/gardner_tablebase/tables/manifest.json
```

## Validation run

```text
node --check app.js
node --check js/engine/engine.js
node --check js/engine/tablebase.js
node --check js/engine/worker.js
node --check js/engine/play-worker.js
node --check js/engine/analysis-cache.js
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v11-efficiency-and-tablebase-tests.mjs
node tests/v15_2-ui-and-move-buffer-tests.mjs
node tests/v16-live-top3-info-tests.mjs
node tests/v16_1-black-perspective-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/play-worker-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/v17-state-tablebase-tactical-tests.mjs
node tests/v10_2-mate-and-efficiency-tests.mjs
node tests/v13-closed-breakthrough-tests.mjs
node tests/pause-resume-worker-tests.mjs
```
