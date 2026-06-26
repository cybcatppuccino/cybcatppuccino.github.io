# Changelog

## v8

- Rebuilt the phone layout around a smaller guaranteed-square board, a thin vertical evaluation rail, compact two-row settings, and a non-scrolling viewport.
- Restored Analysis as an always-visible phone toolbar action and reduced mobile MultiPV display to the best two lines.
- Made the phone move list collapsed by default and removed the redundant mobile Redo control to preserve touch-friendly action widths.
- Added a lazy GardnerTB browser probe for generated exact-core and practical gzip blocks under `tools/gardner_tablebase/tables/`.
- Integrated tablebase probing into continuous analysis and high-level finite-play workers without blocking the UI thread.
- Added exact/practical tablebase labels, DTM/bound handling, persistent caching, and conservative fallback when a sparse record cannot supply a proved continuation.
- Added root repetition-resource ordering for worse positions and cached failed low-material/fortress probes to avoid repeating expensive side searches.
- Stabilized locked two-wing opposite-colour-bishop structures so material is not treated as a decisive advantage before an actual breakthrough is found.
- Added a regression for the supplied `5/3b1/p1k1p/P3P/1K3 b` position and tablebase-loader coverage.
- Updated the engine/cache identity to Orion JS 8.0.

## v7.1

- Replaced the recommended exhaustive six-piece workflow with a practical exact 2–3 core + verified sparse 4–6 overlay.
- Added `data/practical-seeds.json.gz`, extracted from the supplied Gardner and Mallett PGNs.
- Added reachability-, balance-, frequency- and late-game-guided 4–6-piece selection.
- Added safe file-mirror and colour-rotation canonicalization with best-move remapping.
- Added SQLite-WAL checkpoint/resume for sparse graph construction and retrograde solving.
- Fixed the prototype node-cap behavior so stored nodes continue to be expanded after new insertion stops.
- Replaced per-child SQLite existence queries with one batched lookup per parent.
- Added proof-conservative frontier semantics: unresolved children can never create a false LOSS.
- Added delta-coded sparse indexes and independently lazy-loadable gzip blocks.
- Added measured hard-size trimming; the default complete `tables/` directory is capped at 96 MiB.
- Added `quick-estimate`, `quick-generate`, `quick-probe`, and `quick-status` commands plus one-click build scripts.
- Added sparse symmetry, resume, mate replay, unknown-frontier, checksum and size-budget regression tests.
- Added a legal-root fallback for very short low-level AI searches that expire before completing depth 1.

## v7

- Promoted replay-verified mate results to durable solved cache entries.
- Rebased verified mate distance and PV data when a player follows a cached line.
- Prevented the continuous-analysis Worker from recomputing an already validated solved result.
- Added exact-root mate replay validation when restoring persistent cache entries.
- Collapsed Gardner rules and Game Tree by default.
- Made the complete PGN archive demand-loaded only after Book activation or Game Tree expansion.
- Added compact horizontal Game Tree collapse behavior on desktop and tablet.
- Added `tools/gardner_tablebase`, a resumable Numba-JIT Gardner 2–6-piece WDL+DTM retrograde generator with block compression, checksums, lazy probing and crash recovery.
- Updated the engine/cache identity to Orion JS 7.0.

## v6

- Moved Analysis, Book and Edit into the board toolbar.
- Added left/right board mirroring.
- Replaced selectable PGN trees with one merged local/book/AI neighbourhood.
- Added fixed phone and tablet workspaces with no document scrolling.
- Added six selectable initial-layout modes and imported `MalletM25.pgn`.
- Changed Player-vs-AI undo to return to the previous human turn.
- Recalibrated levels 1–9 with lower limits, legal stochastic errors and cache isolation; level 10 remains maximum strength.
- Updated the engine/cache identity to Orion JS 6.0.

## 5.0.0

### Correctness

- Fixed the false `...Bc5 #1` claim after `1.b4 cxb4 2.Rxb4`.
- Prevented selective pruning from skipping every legal child and returning an INF sentinel as a real mate score.
- Corrected excluded singular-search nodes with no remaining candidate.
- Added legal PV replay and terminal verification before any mate score is displayed or allowed to stop continuous analysis.
- Added root-position restoration after deadline exceptions unwind through make/unmake frames.
- Added strict v5 cache-version isolation and cached-mate revalidation.

### Endgames and repetition

- Added a bounded exact low-material DTM proof search for positions with at most six pieces.
- Added a 256-entry on-demand proof cache with history/halfmove-aware keys.
- Added shortest-mate/longest-defence proof selection and conservative cycle/draw handling.
- Replaced repeated root-history scans with O(1) occurrence lookup.
- Added stable cycle detection below the root while preserving formal threefold semantics at the root.
- Added bounded check-evasion extension and score-scaled aspiration windows.

### Interface and data

- Removed heuristic White/Draw/Black percentages.
- Replaced them with a White-relative evaluation scale and verified-mate/DTM labels.
- Updated the engine identifier to Orion JS 5.0 and the persistent storage namespace to v5.
- Regenerated the book calibration and performance reports.

### Validation

- Added a regression test for the exact reported position.
- Added legal DTM proof replay tests and assertions that W/D/L fields are absent.
- Re-ran all core, engine, Worker, cache, play-mode, and AI-vs-AI suites.
- Preserved all original research PGN/PDF files.

## 4.0.0

### Analysis lifecycle and cache

- Added Pause and Continue controls for continuous background analysis.
- Added a structured 96-position browser cache backed by `localStorage`.
- Preserved the latest analysis display after Analysis is stopped.
- Added an independent in-Worker position cache and PV/root-score restoration.
- Seeded a bounded multi-ply corridor from already searched principal variations so selected lines continue from retained work.
- Added immediate cached-result display before deeper search resumes.

### Play modes

- Added Local two-player, Player vs AI, and AI vs AI modes.
- Added selectable human side and ten finite-search difficulty levels.
- Added a separate finite-search Worker so play AI does not imply continuous background analysis.
- Made all three analysis recommendations clickable legal moves.
- Added safe cancellation/token handling when modes, positions, or branches change.
- Added cached finite-search reuse when a play position was already analyzed.

### Engine

- Updated engine identifier to Orion JS 4.0.
- Added a typed-array static-evaluation cache.
- Added capture-history move ordering and aging.
- Expanded Worker transposition tables.
- Added halfmove-clock context to transposition-table keys while leaving repetition hashes rule-correct.
- Reused stored PVs and prior root scores when resuming a position.
- Kept conservative Gardner endgame/null-move safeguards.

### Validation

- Re-ran the supplied PGN benchmark without book injection.
- Added persistent-cache, pause/resume, play-Worker resume, halfmove TT, and AI-vs-AI smoke tests.
- Reviewed the supplied MCTS/PPO archive; no code/model was integrated because its rules and runtime do not match this project and the supplied archive has no clear licence.
- Preserved all original research PGN/PDF files.

## 3.0.0

- Fixed first-click live analysis startup.
- Reworked the search hot path around typed-array TT storage and precomputed attack geometry.
- Added streamed iterative-deepening MultiPV, stronger pruning safeguards, PGN regression benchmarking, and endgame evaluation improvements.
