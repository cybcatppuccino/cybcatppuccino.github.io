# Gardner MiniChess Lab v18.3

This is the v18.3 differential source package. It is designed to be overlaid on v18.2; only modified or new files are included in the release archive.

## What changed in v18.3

- `clearAiCachesOnBoot()` remains intentionally unchanged: refreshing the browser resets AI/cache state while preserving separately stored game state.
- The 50-move automatic draw rule is removed. The game and GTB tablebases now use the same no-50-move convention. Threefold repetition and other terminal rules remain.
- Direct results from the fixed complete 111-table GTB corpus are trusted as solved, including tablebase draws and results whose DTM is a display bound. Covered ≤5-piece roots no longer enter ordinary engine search.
- AI play now imports the same `result-quality.js` rules used by manual analysis, cache selection, and worker resumes.
- Analysis cache keys include the full reversible repetition context; transposition-table locks also include root/history and incremental path repetition salts.
- localStorage writes are dirty-gated and scheduled with `requestIdleCallback` where available, reducing main-thread serialization during streamed analysis.
- Analysis Worker is created only after local analysis is enabled. Play Worker is created only after entering an AI mode. Switching modes releases the inactive worker, so the two workers do not retain duplicate decompressed tablebase caches.
- Tablebase loading uses a shared priority request queue, sequential low-priority neighborhood warming, bounded WDL LRU caching, and no practical-seed fallback path.

## Notes

- The online tablebase deployment is treated as a fixed complete 111-table corpus.
- `data/practical-seeds.json.gz` is not referenced by v18.3 runtime code.
- Run the included `tests/v18_3-rules-cache-tablebase-tests.mjs` after applying the patch.
