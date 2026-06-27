# Gardner MiniChess Lab — v12 patch

This patch is intended to be applied over a complete v11 installation. It contains only code/documentation files changed for v12; keep your existing `data/`, PGN files, and generated tablebase folders in place.

## Install

1. Back up the v11 folder.
2. Unzip this patch at the project root and allow files to overwrite existing files.
3. Keep the existing `tools/gardner_tablebase/tables/` directory or equivalent tablebase deployment path.
4. Restart the local server and force-refresh the browser.

v12 uses the engine identity `Orion JS 12` and a new persistent analysis-cache key, so stale v11 cached lines are not reused.

## What changed in v12

### Tablebase hardcoding and lightweight endings

- `KvK`, `KBvK`, and `KNvK` are handled as hardcoded draws.
- `KBvKB`, `KBvKN`, `KNNvK`, and `KNvKN` are handled by a lightweight rule: check terminal/mate-in-one; otherwise treat as draw.
- This avoids unnecessary exact table loads for positions that contain no useful long DTM information.

### Corrected DTM handling

- Fixed child DTM accounting in `chooseMoves()`: a child checkmate with `dtmPly = 0` now gives the parent `dtmPly = 1`, instead of being collapsed to zero.
- Draw DTM now stays at `0` instead of falling back to PV length.

### Practical manifest shortcut

- After the exact material signature is known, the tablebase now checks whether the practical manifest has that signature before doing practical canonical ranking.
- With the current tables, where no practical manifest is present, 5–6 piece misses return faster.

### WDL inside search

- Workers now warm exact 2–4 piece WDL blocks in the background.
- Alpha-beta and quiescence search can synchronously use already-warmed WDL hits as exact win/draw/loss cutoffs.
- Misses are safe: if a WDL block is not warm yet, search simply falls back to normal evaluation/search.
- Root move ordering and mate proof search also use WDL hints to reach forcing lines faster.

### Long mate search heuristics

- Mate search now prioritizes moves that keep the attacker on a WDL-winning path.
- Attacking moves that sharply reduce the defender's legal replies are searched earlier.
- Defender replies are not trimmed; the optimization is used for ordering and WDL-based refutation, not for unsound proof shortcuts.

## Changed files

- `js/engine/engine.js`
- `js/engine/tablebase.js`
- `js/engine/worker.js`
- `js/engine/play-worker.js`
- `js/engine/analysis-cache.js`
- `index.html`
- `README.md`
- `CHANGELOG.md`
- `VERSION`
- `tests/v11-efficiency-and-tablebase-tests.mjs`
