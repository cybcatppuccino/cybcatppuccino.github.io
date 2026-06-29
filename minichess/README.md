# Gardner MiniChess v20.4 patch notes

This patch is intended to be applied on top of v20.3.

v20.4 tightens cache ownership around analysis: ordinary numeric search results, ordinary TT entries, eval-cache entries, proof misses, and root mate-risk caches are no longer carried across a newly played move. Only independently verified proof mechanisms remain reusable: exact root tablebase results through the proof-only durable cache, and replay-verified in-worker bridge subtrees/certificates while the worker session is alive.

Endgame scheduling is also adjusted for 6-8 piece pawn/rook endings. The main alpha-beta search still receives the dominant budget, while bounded side proof work gets enough time to catch short forced mates or defensive drawing resources without replacing real numeric scores with labels or sentinels.

Validation focus:

- `3k1/K2p1/3Pp/2P1P/5 w - - 4 7` still proves mate rather than showing a small `+1.xx` score.
- `3k1/3pp/5/2PPP/2K2 b - - 3 2` stays numeric when no short forced mate is proven, and the king defence remains visible.
- Ordinary results are not persisted or returned from the analysis cache.
- New-root `GardnerSearcher.beginPosition()` clears ordinary TT/eval/proof-miss caches.
