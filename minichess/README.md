# Gardner MiniChess Lab v17.1 patch notes

v17.1 is intended to be applied over v17. It focuses on side-to-move correctness, stronger play-AI style selection, and making the analysis panel represent the currently thinking play AI instead of running a second analysis job during games.

## What changed

- Fixed the remaining black-to-move mate/proof-line ordering issue. Mate/proof candidates are now kept in side-to-move utility order, so Black positions prefer moves good for Black and White positions prefer moves good for White.
- Audited analysis, cached-result reuse, play-worker final selection and style selection paths for white-centric score leaks.
- Rebalanced play styles for strength first:
  - Balanced remains objective-best;
  - non-balanced styles preserve stable wins and clear advantages;
  - Cunning now looks for practical traps mostly in equal/worse positions, and only among near-equivalent candidates.
- Play workers now stream their own internal candidate lines to the analysis panel while thinking.
- In Human-vs-AI and AI-vs-AI modes, manual Analysis mode is disabled to avoid duplicate computation.
- The analysis panel Pause/Resume button pauses/resumes the play AI and freezes its active time budget while paused.
- Version labels and cache keys were moved to v17.1 / `Orion JS 17.1`.

## Changed or added files

```text
CHANGELOG.md
PATCH_FILE_LIST.txt
README.md
VERSION
app.js
index.html
js/engine/analysis-cache.js
js/engine/difficulty.js
js/engine/engine.js
js/engine/play-client.js
js/engine/play-worker.js
tests/v11-efficiency-and-tablebase-tests.mjs
tests/v15_2-ui-and-move-buffer-tests.mjs
tests/v16-live-top3-info-tests.mjs
tests/v16_1-black-perspective-tests.mjs
tests/v17-state-tablebase-tactical-tests.mjs
tests/v17_1-ai-pause-style-and-mate-order-tests.mjs
```

## Validation run

```text
node --check app.js
node --check js/engine/engine.js
node --check js/engine/play-worker.js
node --check js/engine/play-client.js
node --check js/engine/difficulty.js
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v11-efficiency-and-tablebase-tests.mjs
node tests/v15_2-ui-and-move-buffer-tests.mjs
node tests/v16-live-top3-info-tests.mjs
node tests/v16_1-black-perspective-tests.mjs
node tests/v17-state-tablebase-tactical-tests.mjs
node tests/v17_1-ai-pause-style-and-mate-order-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/play-worker-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/v10_2-mate-and-efficiency-tests.mjs
node tests/v13-closed-breakthrough-tests.mjs
node tests/pause-resume-worker-tests.mjs
```
