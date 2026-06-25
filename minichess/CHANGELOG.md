# Changelog

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
