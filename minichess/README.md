# Gardner MiniChess Lab v17.4 patch notes

v17.4 is intended to be applied over v17.3. It focuses on stabilizing analysis/cache behavior and making exact <=5-piece tablebase results take priority over stale cached analysis.

## Highlights

- Version labels, script cache busting, current-game storage, and analysis-cache storage moved to v17.4 / `Orion JS 17.4`.
- Persistent analysis cache starts fresh in v17.4 and removes older v17.2/v17.3 cache buckets instead of migrating them.
- Fixed the remaining `previous is undefined` cache edge case with defensive result selection in `AnalysisCache.set()` and migration ingestion.
- Analysis worker now re-probes exact <=5-piece tablebase positions before a cached/resumed solved result can stop the worker.
- Exact tablebase probing now takes priority over broad WDL neighborhood warming, avoiding extra network contention during the first solve.
- Tablebase WDL/analysis negative misses from transient network/decompression failures are no longer memoized for the whole session.
- UI-side cache validation rejects stale exact-tablebase/bound results for <=5-piece positions so the worker can re-solve them exactly.

## Why this matters

Older streamed analysis and tablebase-bound artifacts could make a refreshed page display a high-depth but short or stale principal variation. In <=5-piece positions, that was especially visible when moving from a 5-piece table to a 4-piece table. v17.4 treats exact tablebase states as authoritative and gives them a chance to refresh before accepting any cached result.

## Files changed

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
js/engine/tablebase-manifest.js
js/engine/tablebase.js
js/engine/worker.js
tests/v17_2-tablebase-and-cache-tests.mjs
tests/v17_4-cache-tablebase-stability-tests.mjs
```

## Suggested verification

```bash
node --check app.js
node --check js/engine/analysis-cache.js
node --check js/engine/engine.js
node --check js/engine/tablebase.js
node --check js/engine/worker.js
node --check js/engine/play-worker.js
node tests/v17_4-cache-tablebase-stability-tests.mjs
node tests/v17_3-cache-and-worker-stability-tests.mjs
node tests/v17_2-tablebase-and-cache-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/play-worker-tests.mjs
node tests/analysis-cache-tests.mjs
```

If you want to smoke-test the uploaded pawn tablebase subset, serve the repository locally and run:

```bash
TB_BASE=http://127.0.0.1:8125/tools/gardner_tablebase/tables/ node tests/v17_4-cache-tablebase-stability-tests.mjs
```
