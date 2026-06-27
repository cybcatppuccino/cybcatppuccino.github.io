# Changelog

## v18.2

- Updated app/engine/cache/tablebase labels to v18.2 / `Orion JS 18.2`.
- Changed streamed analysis rendering to a fixed trailing 500 ms cadence; tablebase, mate, and solved updates also wait for the next scheduled paint.
- Stabilized mate/tablebase-bound display so later live centipawn scores do not replace a stronger bound for the same root move.
- Skipped redundant async DTM annotation when all visible lines already have mate/tablebase-bound/exact tablebase information.
- Retired v18.1/v18 persistent analysis cache buckets for a clean v18.2 stability baseline, while allowing current-game state restore from v18.1.
- Applied shared result-quality selection on the UI stream path so stronger cached/bound displays are not overwritten by weaker live updates before painting.

## v18

- Updated version labels, current-game storage, analysis-cache storage, and script/tablebase cache-busting tags to v18 / `Orion JS 18`.
- Added a shared result-quality module so the UI cache and analysis worker rank solved, tablebase, proof, live-complete, and live-thin results consistently.
- Trusted exact current-version <=5-piece tablebase resumes instead of always re-probing; stale, WDL-only, and DTM-bound tablebase artifacts still refresh.
- Resumed non-solved deep searches from their recorded next/search depth, reducing repeated work for 6/7-piece positions that are close to falling into the database.
- Made root opponent-mate-risk cache keys include root identity, move, halfmove, and repetition-history signature.
- Made tablebase DTM-bound annotation non-blocking for manual analysis. The main search result streams first; if TB annotation catches up and is higher quality, it posts a follow-up result.
- Improved online tablebase loading by using versioned cacheable manifest/metadata URLs, increasing exact-block cache capacity, avoiding permanent memoization of transient exact-block failures, and displaying bounds as `≤#N · TB bound`.
- Reworked the analysis panel line renderer to reuse keyed row buttons instead of rebuilding the whole list on every update, reducing flicker and preserving click targets.
- Kept the deliberate boot-time AI cache clearing behavior unchanged and avoided intentional changes to AI style/search policy.
