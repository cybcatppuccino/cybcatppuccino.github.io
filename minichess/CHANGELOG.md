# Changelog

## v17.4

- Updated all version labels and cache-busting tags to v17.4 / `Orion JS 17.4`.
- Started a fresh v17.4 persistent analysis cache and removed older analysis cache buckets on load to avoid stale v17.2/v17.3 PV/tablebase artifacts.
- Hardened `AnalysisCache.set()` and migration ingestion so a missing previous entry can never throw `previous.result` / `previous.updatedAt` errors.
- Changed analysis-worker startup so exact <=5-piece tablebase positions are re-probed before any cached solved result can terminate analysis.
- Moved broad tablebase WDL neighborhood warming behind the direct exact tablebase probe in both analysis and play workers, reducing web-side first-load contention.
- Avoided memoizing transient WDL/tablebase miss results caused by network, decompression, or GitHub Pages cache races.
- Rejected stale <=5-piece tablebase-bound cache entries at UI validation time so exact tablebase can refresh them.
- Added v17.4 cache/tablebase stability regression tests.
