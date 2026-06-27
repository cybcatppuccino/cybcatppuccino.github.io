# Gardner MiniChess Lab v17.3 patch notes

v17.3 is intended to be applied over v17.2. It focuses on stabilizing streamed analysis/cache handling and making the web tablebase path faster and less blocking.

## What changed

- Version labels, script cache busting, current-game storage, and analysis-cache storage moved to v17.3 / `Orion JS 17.3`.
- Fixed the reported analysis crash:
  - `TypeError: can't access property "result", previous is undefined`
  - Root cause: the persistent analysis cache tried to return `previous.result` while handling a first-position PV-incomplete stream where no previous entry existed.
- Incomplete analysis streams are still allowed to update the live panel, but they are no longer persisted as resume/cache artifacts unless they are solved/terminal/tablebase/proof results.
- Cache replacement now compares solved status, PV completeness, score depth, and PV depth before replacing an older artifact.
- Tablebase web loading is faster and less redundant:
  - exact tablebase analysis begins with WDL-only blocks;
  - DTM is loaded only for the WDL-relevant candidate pool;
  - full DTM block loading reuses an already-loaded WDL block;
  - concurrent metadata/WDL/full-block requests are de-duplicated;
  - WDL neighborhood warming no longer blocks the first analysis step.
- The full <=5-piece manifest remains included as `tools/gardner_tablebase/tables/manifest.json` and as the embedded JS fallback.

## Files in this patch

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
tests/v11-efficiency-and-tablebase-tests.mjs
tests/v15_2-ui-and-move-buffer-tests.mjs
tests/v16-live-top3-info-tests.mjs
tests/v16_1-black-perspective-tests.mjs
tests/v17-state-tablebase-tactical-tests.mjs
tests/v17_1-ai-pause-style-and-mate-order-tests.mjs
tests/v17_2-kqvkbb-tablebase-smoke-tests.mjs
tests/v17_2-tablebase-and-cache-tests.mjs
tests/v17_3-cache-and-worker-stability-tests.mjs
tools/gardner_tablebase/tables/manifest.json
```

## Suggested verification

```bash
node --check app.js
node --check js/engine/engine.js
node --check js/engine/tablebase.js
node --check js/engine/analysis-cache.js
node --check js/engine/worker.js
node --check js/engine/play-worker.js
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v17_3-cache-and-worker-stability-tests.mjs
node tests/v17_2-tablebase-and-cache-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/play-worker-tests.mjs
node tests/analysis-cache-tests.mjs
```

For the partial KQvKBB database smoke test, serve the tablebase directory and run:

```bash
TB_BASE=http://127.0.0.1:8123/tools/gardner_tablebase/tables/ node tests/v17_2-kqvkbb-tablebase-smoke-tests.mjs
```
