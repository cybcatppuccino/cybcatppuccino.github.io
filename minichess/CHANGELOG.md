# Changelog

## v11

- Updated engine/cache identity to Orion JS 11 and invalidated v10.2 persistent analysis-cache entries.
- Optimized tablebase checking by prewarming manifests in workers, caching probe results, caching full tablebase analysis results, batching child probes, and reducing exact-rank allocation.
- Added incremental `pieceCount` to `EnginePosition`; make/undo now preserves it through pooled state for faster small-tablebase eligibility checks.
- Removed several allocation-heavy helpers from hot paths.
- Reused per-ply typed buffers for move-order scores and quiet-history updates.
- Added v11 regression coverage for incremental masks, tablebase probe/analysis caching, and cache-key invalidation.

## v10.2

- Fixed the remaining mate-score dip before verified mate publication in compact king-and-pawn races.
- Added exact low-material mate proof for tiny positions.
- Added conservative search hot-path optimizations from v10.1.
