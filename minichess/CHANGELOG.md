# Changelog

## v14

- Updated engine identity to `Orion JS 14` and cache key to `gardner-analysis-cache-v14`; v13 persisted PV caches are removed on load.
- Added a queenful deadlock classifier for fully locked, low-mobility positions with no legal irreversible progress, so static material edges are compressed to practical draws only after legal-move verification.
- Kept the v13 closed-breakthrough path separate by blocking deadlock compression when contact captures/check resources exist.
- Added Fairy-Stockfish wasm 1.1.11 as an optional UCI provider under `vendor/fairy-stockfish/`, with legal-PV validation and Orion JS fallback for invalid or unavailable external output.
- Added a UI kernel selector used by both live analysis and AI play.
- Added v14 tests for queenful deadlock compression and external-engine PV validation.

## v13

- Updated engine identity to `Orion JS 13` and cache key to `gardner-analysis-cache-v13`; v12.1/v12.2 persisted PV caches are removed on load.
- Added queenful closed-position search safeguards, full-window root verification for closed roots, and quiet breakthrough/threat extensions.
- Changed in-search twofold repetition handling from unconditional 0.00 to a small side-to-move cycle-contempt score, while keeping formal/proven dead draws exact.
- Added v13 closed-breakthrough regression tests based on the supplied `1.c3 d3 2.a3 b3 3.e3 Bxa3 4.Rxa3 Nxa3 5.Nxa3` line.
- Regenerated derived benchmark/calibration data for `Orion JS 13`.

## v12.2

- Migrated public coordinates from legacy Gardnerfish b2–f6 to standard A1–E5 for UI labels, SAN, UCI, engine output, PV arrows, tooltips and copied FEN.
- Made compact 5×5 FEN the canonical displayed/exported FEN while retaining legacy b2–f6 padded FEN import support.
- Added explicit standard/legacy coordinate helpers and PGN parsing mode so old research PGNs load through a compatibility adapter but display as standard SAN.
- Updated engine identity to `Orion JS 12.2` and cache key to `gardner-analysis-cache-v12_2`; v12.1 persisted PV cache is removed on load.
- Updated `data/library.json` titles and regenerated benchmark/calibration/practical-seed derived data with standard A1–E5 UCI/FEN.
- Updated database/tablebase tooling comments and FEN parser compatibility without changing square-indexed database binaries.
- Updated regression tests and added coordinate round-trip, legacy FEN import and legacy PGN adapter coverage.

## v12.1

- Updated engine/cache identity to `Orion JS 12.1` and `gardner-analysis-cache-v12_1`.
- Added TB-assisted DTM bound annotation for normal search lines that enter exact 2–4 piece tablebase positions.
- Replaced generic exact-tablebase `TB win` / `TB loss` text with mate-distance score text when DTM is available.
- Added UI labels for `TB bound` lines without marking them as fully verified mates unless the PV replays to checkmate.
- Made WDL warm-up safer in local browser/webview environments by avoiding repeated warm-ups, yielding between blocks, and ignoring individual missing/corrupt WDL blocks.

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
