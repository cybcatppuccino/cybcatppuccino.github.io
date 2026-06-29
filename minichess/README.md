# Gardner MiniChess v21 patch notes

v21 adds a second engine, **Minifish JS 21**, alongside the original Orion engine. Minifish is intentionally rebuilt as a simpler no-cache searcher: brute-force alpha-beta, tactical/quiescence search, simple pruning, force-move extensions, and direct Gardner tablebase cutoffs for <=5-piece leaves. When the root position is already in the database, it publishes the database result directly, including Mate in N where available.

Key changes:

- Added `js/engine/minifish.js` as a fully parallel AI/search implementation with no transposition/eval/proof cache.
- Added engine selection in the UI: Minifish v21, Orion v21, and optional Fairy-Stockfish. Minifish is the default for the new v21 line.
- Wired Minifish into both the continuous analysis worker and the playing-AI worker.
- Fixed the endgame stall around `5/4k/p1N1P/R4/4K b - - 2 14` by bounding Orion's root opponent-mate guard to the current iterative-deepening slice instead of spending a fresh fixed side-proof budget per root move.
- Bumped engine/cache/storage namespace to v21.

Validation target: Engine `Minifish JS 21` plus `Orion JS 21`; storage namespace `v21`.
