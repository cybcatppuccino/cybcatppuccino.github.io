# Changelog

## v16.1

- Updated all current version labels to v16.1 / `Orion JS 16.1`.
- Fixed the v16 live-analysis merge so candidate lines are ranked by the current side to move, not always by white-centric score.
- Re-normalized cached/resumed analysis lines before display and AI-play reuse, preventing stale v16 black-side cache entries from putting White-favorable moves first.
- Added black-perspective regression coverage for engine output, analysis worker streaming, and play-worker source ordering safeguards.

## v16

- Updated the current UI and engine identity to `Orion JS 16`.
- Added live root-candidate reporting so interrupted analysis refreshes can still show the strongest three known candidate moves instead of waiting for a full new depth to finish.
- Merged newly refreshed candidate information with the previous known lines in the analysis worker, keeping the evaluation bar and candidate list dynamic while preserving existing search rules, evaluation meaning and move choice logic.
- Kept persistent analysis-cache compatibility with v15.2 Orion entries after the engine identity bump to `Orion JS 16`.

## v15.2

- Removed the user-facing Kernel selector and Fairy-Stockfish/Stockfish labels from the UI path.  UI analysis and play requests now dispatch Orion JS directly.
- Kept existing Fairy-Stockfish vendor files untouched so older repositories can retain them, but v15.2 no longer requires deploying or selecting Stockfish from the interface.
- Added per-ply typed-array move buffers for ordinary Orion search and quiescence to avoid repeated JS array allocation at search nodes.
- Added legal-move metadata for capture, promotion, moved type, captured type and gives-check status, allowing sorting/pruning/extension code to reuse the information instead of repeating make/undo through `givesCheck()`.
- Preserved move legality, rules, evaluation meaning and search scoring logic; the changes are code-structure/performance-only.

## v15.1

- Restored the stable v14.1-style Fairy-Stockfish startup path: when Fairy is selected, the UI now sends `fairy-stockfish` to the engine worker directly instead of preemptively replacing it with Orion JS when the main page reports missing `SharedArrayBuffer`.
- Replaced the v14.3/v15 COI service-worker bootstrap with a cleanup-only helper.  It unregisters stale `coi-serviceworker.js` registrations and clears reload guards that could keep the UI stuck in fallback/preparing mode.
- Kept the required browser setup simple and explicit: use `serve.sh` / `serve.bat` so the HTTP server sends COOP/COEP/CORP headers directly.
- Updated engine identity to `Orion JS 15.1` and persistent cache key to `gardner-analysis-cache-v15_1`.  Compatible v14, v14.1, v14.2, v14.3 and v15 Orion cache entries migrate forward.
- Kept all v15 cache capacities unchanged.
- Added a small PV SAN-format memoization layer in the UI to reduce repeated formatting and rendering work without changing search, evaluation, move legality, or engine behavior.
- Updated regression tests for v15.1 Stockfish startup simplification, legacy COI cleanup, and cache migration.
