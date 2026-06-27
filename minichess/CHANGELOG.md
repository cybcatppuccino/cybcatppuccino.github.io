# Changelog

## v12

- Updated engine/cache identity to `Orion JS 12` and `gardner-analysis-cache-v12`.
- Hardcoded trivial draw signatures: `KvK`, `KBvK`, `KNvK`.
- Replaced low-value one-ply-only signatures with lightweight handling: `KBvKB`, `KBvKN`, `KNNvK`, `KNvKN`.
- Fixed `chooseMoves()` child DTM accounting for mate-in-one continuations.
- Preserved draw DTM as zero instead of falling back to PV length.
- Avoided practical canonical ranking unless the practical manifest actually contains the exact material signature.
- Added WDL-only exact block loading and synchronous WDL probes for search.
- Wired WDL probes into analysis worker, play worker, alpha-beta, quiescence, root ordering, and mate proof move ordering.
- Added WDL-guided long-mate ordering that prioritizes WDL-winning corridors and moves that restrict defender replies.
