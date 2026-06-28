# Changelog

## v19.5

- Updated release labels, browser cache-buster, saved-game namespace, and engine identity to v19.5 / `Orion JS 19.5`.
- Kept the 500 ms UI paint cadence while making engine result publication atomic: live chunks now advance progress metrics without replacing a completed score/PV/proof tuple.
- Disabled ordinary analysis/play result persistence and cross-root resume. Exact root tablebase and independently verified proof artifacts remain eligible for reuse.
- Removed PV-child corridor cache seeding and disallowed TT-reconstructed tails from qualifying as complete MultiPV results.
- Converted future-PV tablebase DTM annotations into conditional hints; they no longer promote a root to mate or tablebase-solved status.
- Replaced hard root low-progress draw flattening with an extra search-audit budget.
- Added a persisted board-style selector using the existing Standard, Green, Sand, Slate, and Sketch palettes.

## v19.4

- Updated visible release labels, saved-game namespace, and engine identity to v19.4 / `Orion JS 19.4`.

## v19.3

- Updated visible release labels, browser cache-buster, game-state namespace, and Orion engine/cache namespace to v19.3 / `Orion JS 19.3`.
