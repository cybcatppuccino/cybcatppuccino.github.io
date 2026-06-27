# Changelog

## v15.1

- Restored the stable v14.1-style Fairy-Stockfish startup path: when Fairy is selected, the UI now sends `fairy-stockfish` to the engine worker directly instead of preemptively replacing it with Orion JS when the main page reports missing `SharedArrayBuffer`.
- Replaced the v14.3/v15 COI service-worker bootstrap with a cleanup-only helper.  It unregisters stale `coi-serviceworker.js` registrations and clears reload guards that could keep the UI stuck in fallback/preparing mode.
- Kept the required browser setup simple and explicit: use `serve.sh` / `serve.bat` so the HTTP server sends COOP/COEP/CORP headers directly.
- Updated engine identity to `Orion JS 15.1` and persistent cache key to `gardner-analysis-cache-v15_1`.  Compatible v14, v14.1, v14.2, v14.3 and v15 Orion cache entries migrate forward.
- Kept all v15 cache capacities unchanged.
- Added a small PV SAN-format memoization layer in the UI to reduce repeated formatting and rendering work without changing search, evaluation, move legality, or engine behavior.
- Updated regression tests for v15.1 Stockfish startup simplification, legacy COI cleanup, and cache migration.
