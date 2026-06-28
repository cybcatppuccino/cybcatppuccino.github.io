# Gardner MiniChess Lab v19.5

This v19.5 package is a conservative stability update based on v19.4. It keeps the browser UI's fixed 500 ms analysis paint cadence and retains the engine's transposition table, evaluation cache, move ordering and GTB data caches.

## What changed in v19.5

- Updated all runtime release identities and saved-state/cache namespaces to v19.5 / `Orion JS 19.5`.
- Ordinary centipawn/PV snapshots are no longer persisted or resumed across a fresh analysis root. Only an exact root tablebase result, independently verified forced-mate/endgame proof, fortress proof, or terminal-rule result may persist.
- Removed principal-variation corridor seeding: a parent PV can no longer create a high-score 1–2 ply child result.
- A tablebase hit later inside a speculative PV is now stored only as a conditional hint. It cannot rewrite a root score to mate, change ordering, terminate search, or enter a cache.
- Worker publication is atomic: an incomplete chunk updates metrics only; the last completed iteration keeps its own score, proof flags and PV intact.
- MultiPV lines must complete their final root score window before they are accepted as a stable result. TT-appended PV tails are marked reconstructed and do not qualify as complete analysis evidence.
- Low-progress/closed-root detection keeps its audit time allowance but no longer hard-forces every line to 0.00 without a real draw/fortress proof.
- Added a persisted **Board** style selector next to the existing piece selector. It exposes the shipped Standard, Green, Sand, Slate and Sketch board palettes.

## Deliberate non-changes

- Front-end analysis rendering remains throttled at exactly 500 ms.
- Transposition-table, evaluation-cache, tablebase-data-cache and move-ordering reuse remain active; these are search accelerators, not stale analysis conclusions.
- Existing Oracle/GTB root tablebase handling remains the authoritative exact endgame path.

## Suggested smoke tests

Run from the project root with a modern Node runtime:

```bash
node tests/v19_5-stability-tests.mjs
node tests/core-tests.mjs
node tests/engine-tests.mjs
```

Open `index.html` through the supplied local server script rather than directly from `file://` so Worker and tablebase loading can run normally.
