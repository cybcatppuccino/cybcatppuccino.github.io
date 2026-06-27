# Gardner MiniChess Lab v16.1 patch notes

v16.1 is intended to be applied over v16. It keeps Orion's rules, legal-move logic, evaluation function and search decision semantics unchanged, and fixes the black-side ordering regression introduced by v16's live top-three analysis merge.

## What changed

- Current user-facing version labels are updated to v16.1 / `Orion JS 16.1`.
- Live analysis candidate merging now sorts by **root side-to-move utility**, not by white-centric score alone.
- Black-to-move analysis no longer promotes lines that are better for White to the top of the candidate list.
- Cached/resumed analysis results are re-normalized against the current side to move before display, reuse, PV arrows, and AI-play fallback consumption.
- Play-worker cached/resumed Orion results are also re-sorted by side-to-move utility before style selection and caching.
- Persistent cache migration remains compatible with v15.2 and v16 Orion entries.

## Patch contents

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
js/engine/worker.js
tests/v15_2-ui-and-move-buffer-tests.mjs
tests/v16-live-top3-info-tests.mjs
tests/v16_1-black-perspective-tests.mjs
```

## Validation commands

```bash
node --check app.js
node --check js/engine/engine.js
node --check js/engine/worker.js
node --check js/engine/play-worker.js
node --check js/engine/analysis-cache.js
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v15_2-ui-and-move-buffer-tests.mjs
node tests/v16-live-top3-info-tests.mjs
node tests/v16_1-black-perspective-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/play-worker-tests.mjs
node tests/analysis-cache-tests.mjs
```
