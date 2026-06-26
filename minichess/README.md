# Gardner MiniChess Lab — v11

This patch is intended to be copied over a complete v10.2 installation. It contains only files changed or added after v10.2; it does not include generated tablebase blocks, PGN archives, PDFs, or other unchanged assets.

## Install

1. Back up the current v10.2 folder.
2. Unzip this patch at the project root and allow files to overwrite existing files.
3. Keep your existing `data/`, `tools/gardner_tablebase/tables/`, PGN, PDF and tablebase files in place.
4. Restart the local server and force-refresh the browser.

v11 uses a new engine identity and a new persistent analysis-cache key, so stale v10.2 cached lines are not reused.

## What changed in v11

### Faster tablebase checking

The previous tablebase path spent noticeable time in `analyze()` after the initial WDL probe: it enumerated legal root moves, probed each child, and then repeatedly probed positions while building PVs. On small tablebase positions that overhead could be visible as roughly a few tenths of a second.

v11 keeps the same tablebase semantics, but reduces repeated work:

- starts manifest loading as soon as the analysis/play worker is created;
- caches exact/practical probe results by board hash;
- caches completed tablebase analysis results by board hash, MultiPV and PV limit;
- reuses already-loaded metadata and blocks more aggressively;
- probes child positions in a batched async step instead of serially awaiting every child;
- removes several ranking allocations in exact tablebase indexing.

The engine still does not invent arbitrary tablebase moves when a sparse practical record cannot prove a legal continuation.

### Search hot-path efficiency

v11 also adds broader pure-efficiency changes while preserving search behavior:

- `EnginePosition` now carries an incremental piece count in addition to the v10.2 king-square cache;
- make/undo restores that counter from pooled state instead of rescanning for tablebase eligibility;
- material-profile results are cached by Zobrist key;
- insufficient-material detection no longer allocates arrays or sets;
- move sorting reuses per-ply score buffers;
- quiet-move history updates reuse per-ply typed buffers;
- tactical-move filtering and several helper scans avoid `filter()`, `some()`, `map()` or temporary arrays on hot paths.

These changes are intended to make the program faster without weakening the search, draw logic, mate proof logic, or tablebase result quality.

## Recommended tests

```bash
node tests/core-tests.mjs
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v10-low-progress-draw-tests.mjs
node tests/v10_1-mate-tests.mjs
node tests/v10_2-mate-and-efficiency-tests.mjs
node tests/v11-efficiency-and-tablebase-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/client-regression-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/play-worker-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/tablebase-loader-tests.mjs
```

`tablebase-loader-tests` may still skip when no local generated tables are present.
