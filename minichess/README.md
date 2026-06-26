# Gardner MiniChess Lab — v6

A static, browser-native 5×5 chess application with legal Gardner rules, editable positions, merged research books, a local JavaScript Alpha-Beta engine, Player-vs-AI and AI-vs-AI modes.

## Run

The project uses ES modules and Web Workers, so serve the folder over HTTP:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. On Windows, `serve.bat` does the same.

## v6 highlights

- `Analysis`, `Book` and `Edit` are compact board-toolbar actions.
- Added independent top/bottom flip and left/right mirror controls.
- One unified Game Tree merges the local game, all PGN books and the current top-three AI lines.
- Non-current branches are compressed to their principal continuation.
- Phone and tablet layouts use a fixed viewport without whole-page scrolling, including a compact short-landscape layout.
- Player-vs-AI undo returns to the previous human-to-move position.
- Added six starting layouts: Standard, Central symmetry, MiniChess 60, Central MiniChess 60, Pure random and Mallett Chess.
- Added the supplied `MalletM25.pgn` to the preserved research archive.
- Levels 1–9 are substantially weakened with shallow limits and calibrated legal-move randomness; level 10 remains deterministic maximum strength.
- Lower play levels no longer become accidentally strong by reusing deep continuous-analysis caches.

## Engine separation

- **Analysis worker:** continuous iterative deepening, MultiPV 3, pause/resume and persistent position cache.
- **Play worker:** one finite search per AI turn, difficulty-specific limits and selection policy.

The engine remains a classical, non-NNUE searcher with PVS, quiescence, transposition tables, move ordering, conservative pruning and low-material mate proof support.

## Research files

Original supplied material is stored under:

```text
data/pgn/
data/reference/
```

`data/SOURCE_INTEGRITY.sha256` records SHA-256 hashes, including the added Mallett PGN.

## Tests

```bash
node tests/core-tests.mjs
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v5-engine-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/client-regression-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/play-worker-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/v6-layout-and-strength-tests.mjs
```

The real-position difficulty calibration report is stored in `data/level-calibration.json`. See `docs/V6-AUDIT.md` for implementation and validation details.
