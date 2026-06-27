# Gardner MiniChess Lab v18.1 patch notes

v18.1 is intended to be applied over v18. It keeps the v18 result-quality/tablebase architecture and makes a small stability pass without changing the core chess analysis algorithm.

## What changed

- Version labels, script cache busting, current-game storage, and analysis-cache storage moved to v18.1 / `Orion JS 18.1`.
- Analysis result painting is now capped at one UI refresh every 500 ms. Important/solved/tablebase results no longer bypass the throttle; the latest result is held and rendered on the next scheduled paint.
- Worker result merging now preserves verified mate/tablebase-bound scores for the same root move. A later live centipawn estimate such as `+220` can no longer overwrite a proven/bounded mate display like `≤#10 · TB bound`.
- The worker now compares each streamed cumulative result against the last known result before posting it, so higher-quality solved/bound results remain stable while search continues.
- Optional DTM annotation skips positions whose visible lines already carry mate/tablebase-bound/exact tablebase information, reducing redundant async tablebase annotation churn.
- The v18 analysis cache bucket is retired in v18.1 so stale v18 live/bound transitions do not persist across the new stability rules.

## Changed/new files in this patch

See `PATCH_FILE_LIST.txt`.
