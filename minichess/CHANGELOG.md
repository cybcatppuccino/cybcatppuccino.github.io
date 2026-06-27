# Changelog

## v15

- Fixed the Fairy-Stockfish startup/fallback loop caused by stale v14 COI service-worker state.  The COI helper now registers a versioned `coi-serviceworker.js?v15`, bypasses cached workers, unregisters older same-scope COI workers, and performs a bounded reload sequence until `SharedArrayBuffer` is available.
- Added `tools/serve-coi.py` to complete the local COOP/COEP launch path referenced by `serve.sh` and `serve.bat`.  The server sends `Cross-Origin-Opener-Policy`, `Cross-Origin-Embedder-Policy`, `Cross-Origin-Resource-Policy`, no-store cache headers, and correct wasm MIME type.
- Updated engine identity to `Orion JS 15` and persistent cache key to `gardner-analysis-cache-v15`.  Compatible v14/v14.1/v14.2/v14.3 Orion cache entries migrate into v15.
- Kept the v14.3 cache capacities unchanged: 576 persistent entries, expanded eval/structural caches, and larger worker result caches remain in place.
- Reduced UI blocking without changing chess semantics by coalescing streamed analysis DOM rendering while preserving every cache write and final result update.
- Added a small worker-side recent-history FEN key cache in analysis and play workers to avoid repeated FEN parsing for identical history lists.
- Improved Fairy/COI status handling in the UI: selecting Fairy now actively requests isolated-mode preparation, shows clearer status, and uses Orion only until isolation is ready.
- Added v15 regression coverage for versioned COI registration, v14.3→v15 cache migration, unchanged 576-entry cache capacity, and analysis UI render coalescing.
