# Changelog

## v17.2

- Updated version labels and cache keys to v17.2 / `Orion JS 17.2`.
- Removed the stale COI service-worker registration script reference that caused a GitHub Pages 404 after Stockfish UI removal.
- Added an embedded full <=5-piece exact tablebase manifest fallback and cache-busted/no-store manifest loading, so a stale 36-table web manifest is augmented to the 111-table v17.2 manifest.
- Added tablebase payload validation for HTML/404 and Git LFS pointer responses, making web deployment mistakes visible instead of silently falling back to search.
- Optimized tablebase probing to use WDL-first lazy block loading, loading DTM blocks only for WDL-relevant candidate moves and PV construction.
- Let already-warmed <=5-piece WDL tablebase blocks participate in synchronous search probing, rather than restricting internal search probes to <=4 pieces.
- Fixed KQvKBB-style DTM display jumps by deriving candidate DTM from the child side's legal best continuation when raw child-position DTM conflicts with the actual tablebase continuation.
- Removed root-search legal-move array allocation by using reusable root move buffers.
- Split analysis/play cache trust into score depth and PV completeness, preventing short live PVs from overwriting complete cached best lines.
- Added v17.2 tests for manifest fallback, cache/PV completeness, and KQvKBB DTM transition behavior.

## v17

- Added Local-mode boot defaults, current-game cache restore, lazy <=5-piece tablebase wiring, root short-mate safety and thin-PV cache safeguards.

## v16.1

- Fixed v16 live top-three merge ordering for black-to-move positions by ranking lines by current side-to-move utility rather than white-centric score.
