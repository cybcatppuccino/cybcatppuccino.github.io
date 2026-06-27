# Gardner MiniChess Lab v17.2 patch notes

v17.2 is intended to be applied over v17.1. It focuses on making the web tablebase path robust for <=5-piece exact databases, using tablebase information earlier in search, and preventing incomplete PV/cache artifacts from becoming the displayed or selected best line.

## What changed

- Version labels and cache keys were moved to v17.2 / `Orion JS 17.2`.
- Removed the stale `coi-serviceworker-register.js` script reference from `index.html`; Stockfish/COI is no longer used by the UI, and the missing script was producing a 404 on GitHub Pages.
- Added an embedded v17.2 exact tablebase manifest fallback. If GitHub Pages serves an older 36-table manifest from cache, the runtime now augments it to the full 111 known <=5-piece table list, including the 75 five-piece tables such as `KQvKBB` and `KRvKNP`.
- Manifest and metadata requests now use a v17.2 cache-busting URL and `no-store`, while large tablebase block files remain browser-cacheable.
- Tablebase binary downloads now detect HTML/404 pages and Git LFS pointer files early and report a clear error instead of silently falling back to normal search.
- Exact tablebase probing is now WDL-first: candidate classification loads only the needed WDL block first, and DTM blocks are loaded only for WDL-relevant candidates/PV construction.
- Already-warmed <=5-piece WDL blocks now participate in the synchronous search probe path, so search can use five-piece tablebase hits instead of only <=4-piece hits.
- Fixed the KQvKBB-style DTM jump during 5→4 / child-position transitions by deriving displayed candidate DTM from the child side's best tablebase continuation rather than trusting a raw child-position DTM value when it is inconsistent with legal best play.
- Root search now uses reusable move buffers instead of allocating a fresh root legal-move array for every root search.
- Analysis/play cache metadata now distinguishes score depth from PV completeness. PV-incomplete high-depth live artifacts are not allowed to overwrite complete cached lines.
- Added v17.2 regression coverage for manifest fallback, cache/PV completeness, and the KQvKBB DTM transition smoke case.

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
tools/gardner_tablebase/tables/manifest.json
```

## Validation run

```text
node --check app.js
node --check js/engine/analysis-cache.js
node --check js/engine/engine.js
node --check js/engine/play-worker.js
node --check js/engine/tablebase-manifest.js
node --check js/engine/tablebase.js
node --check js/engine/worker.js
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v15_2-ui-and-move-buffer-tests.mjs
node tests/v16-live-top3-info-tests.mjs
node tests/v16_1-black-perspective-tests.mjs
node tests/v17-state-tablebase-tactical-tests.mjs
node tests/v17_1-ai-pause-style-and-mate-order-tests.mjs
node tests/v17_2-tablebase-and-cache-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/play-worker-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/tablebase-loader-tests.mjs
```

`tests/v17_2-kqvkbb-tablebase-smoke-tests.mjs` is included and will run when `KQvKBB` data is available at `TB_BASE` or at the local served tablebase path; otherwise it skips with a message.
