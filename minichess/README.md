# Gardner MiniChess Lab v18.2

This is the full v18.2 source package. It is rebuilt on the v18 result-quality/tablebase architecture, keeps the v18.1 mate/bound display stabilization, and adds a stricter UI paint cadence without changing the core chess analysis algorithm.

## What changed in v18.2

- Version labels, script cache busting, current-game storage, analysis-cache storage, and tablebase cache tags moved to v18.2 / `Orion JS 18.2`.
- Analysis result painting now uses a fixed trailing 500 ms cadence: every worker/cache update replaces the pending result, and the panel renders only on the next scheduled tick.
- Important/solved/tablebase/mate results no longer bypass that 500 ms cadence, which prevents rapid line/button churn.
- The v18.1 bound-stability fix is retained: verified mate, exact tablebase, and tablebase-bound lines are preserved over later ordinary live centipawn estimates for the same root move.
- The UI-side stream path now also applies the shared result-quality selector before painting, so a stronger cached/bound result is not overwritten by a weaker live update while waiting for the next tick.
- Optional DTM annotation still runs asynchronously, but skips redundant annotation when visible lines already carry mate/tablebase-bound/exact tablebase information.
- v18.1 analysis-cache entries are retired in favor of a fresh v18.2 cache bucket, while current-game state can still restore from v18.1.

## Notes

- `clearAiCachesOnBoot()` remains intentionally unchanged.
- AI style/search policy is not intentionally changed.
- The package includes all project files, not only a differential patch.
