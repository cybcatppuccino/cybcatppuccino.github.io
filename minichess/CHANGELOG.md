# Changelog

## v17

- Updated all current version labels to v17 / `Orion JS 17`.
- Page boot now always defaults the play mode to Local and removes persistent AI-analysis cache keys before the UI starts.
- Added a small independent current-game cache so refresh restores the previous board, current node and local game tree without reusing AI analysis state.
- Reworked Gardner tablebase wiring for browser use:
  - exact tablebase use is hard-limited to the uploaded `tools/gardner_tablebase/tables/manifest.json` style <=5-piece tables;
  - startup no longer warms all small WDL files;
  - workers lazily prefetch only the exact WDL block for the current position and immediately relevant legal children;
  - exact root tablebase analysis still loads the needed metadata/block on demand.
- Added root tactical-safety verification for short opponent forced mates. Root candidates that allow a verified short mate are marked and scored as mate losses, preventing moves like `b2-b3` in `1n2k/p1ppp/2p2/PP1PP/4K w - - 0 2` from being treated as normal candidates.
- Added PV quality safeguards: deep cached results with shallow non-terminal PV fragments are rejected, and root PV display is extended from existing TT information when available.
- Added v17 regression coverage for Local boot/cache behavior, lazy <=5-piece tablebase wiring, the uploaded manifest, the mate-trap FEN, and thin-PV cache rejection.

## v16.1

- Fixed v16 live top-three merge ordering for black-to-move positions by ranking lines by current side-to-move utility rather than white-centric score.
