# v3 AI audit

## Critical functional defect

### First analysis request was ignored

In v2, the UI called `AnalysisClient.update()` after the user enabled analysis. `update()` returned immediately while `active` was still false, so no `start` message reached the Worker. The panel could therefore remain on “Searching…” forever.

v3 makes `update()` a real start/update operation. A dedicated fake-worker regression test verifies that the first request is queued before readiness and sent immediately after the `ready` event.

## Correctness defects fixed

1. **Repetition:** one prior matching hash was effectively enough to score a draw. v3 counts the current occurrence and requires three total occurrences.
2. **Quiescence while checked:** the depth-cap path could return a non-position score. v3 always resolves at least one legal check evasion and returns a finite value.
3. **Extension growth:** repeated checks/promotions could preserve or increase nominal depth for too many plies. v3 caps cumulative extensions.
4. **Worker mate stop:** any MultiPV line containing mate stopped analysis. v3 stops only when the primary/best line contains a forced mate.
5. **Malformed engine FEN:** padded pieces outside b2–f6 could be silently ignored. v3 rejects them and validates kings, side to move, and promotion ranks.
6. **Dead endings:** bare kings and standard insufficient-material sets now terminate as draws.

## Efficiency defects fixed

1. The v2 transposition table used `Map` entries containing objects. This produced allocation and garbage-collection pressure. v3 uses fixed typed arrays.
2. Attack geometry was repeatedly reconstructed from coordinates. v3 precomputes all 25-square targets/rays.
3. Evaluation repeatedly scanned attacks in separate passes. v3 fuses attack counts and mobility.
4. King safety allocated a `Set` at every evaluation. v3 counts attackers without allocation.
5. MultiPV searched every legal root move with a full window. v3 searches one PV at a time with root exclusion, PVS, aspiration, and prior-depth ordering.
6. Move ordering created avoidable temporary objects. v3 uses parallel typed score arrays and insertion sort for the small Gardner move lists.

## Endgame changes

- exact insufficient-material recognition
- lower pruning aggressiveness at seven pieces or fewer and in minor/pawn-only structures
- null move disabled in likely zugzwang and near-promotion cases
- reduced LMR on passed-pawn pushes
- king centralization and mating-pressure terms
- legal PV replay tests in elementary queen endings

No claim is made that v3 has a complete tablebase. Exact tablebases remain a later extension.

## Oracle-PGN check

The benchmark uses supplied PGN main continuations only as expected/reference moves. It sends no book candidates to search. In the bundled depth-8 sample, all 16 reference moves were within 100 centipawns of the engine preference, 15/16 were in MultiPV 3, and the mean deficit was 13.75 centipawns.

## Performance check

Initial position, 950 ms, MultiPV 3, same Node.js environment:

- v2 median after warm-up: 19,617 NPS, depth 3
- v3 median after warm-up: 112,283 NPS, depth 8

The relative throughput was 5.72× in this environment. Browser/device results will differ.
