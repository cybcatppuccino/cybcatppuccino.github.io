# Orion JS 4.0 engine design

## Purpose

Orion is a local classical Gardner Chess engine for interactive browser analysis and play. It runs in ES-module Web Workers, uses no NNUE or neural model, and keeps the board/UI thread free.

Two Workers are intentionally separate:

- `worker.js`: continuous, resumable MultiPV analysis;
- `play-worker.js`: one finite move search when an AI-controlled side is to move.

Enabling a play mode therefore does not silently enable permanent analysis.

## Representation

- native `Int8Array(25)` board;
- positive piece codes for White and negative codes for Black;
- packed integer moves: 5 origin bits, 5 destination bits, 3 promotion bits;
- in-place make/unmake restoring capture, counters, side, and two 32-bit Zobrist keys;
- precomputed knight, king, pawn, and sliding-ray geometry;
- strict compact 5×5 and Gardnerfish-padded FEN parsing.

## Hashing and transposition safety

The repetition hash contains board pieces and side to move, as required for threefold detection. The transposition-table key additionally mixes the capped halfmove clock. This prevents an otherwise identical board at halfmove 0 and halfmove 99 from sharing a search value near the fifty-move draw horizon.

The two-way typed-array TT stores:

- primary key and independent lock;
- depth, score, static evaluation, best move, bound type, and generation;
- mate scores normalized by ply;
- depth/exact/age-aware replacement.

A separate 65,536-entry direct-mapped evaluation cache avoids recomputing position-only evaluation terms.

## Search pipeline

1. A Worker restores a saved result for the requested position, when available.
2. The saved PV and root scores seed ordering; the next completed depth becomes the new start depth.
3. Root moves are ordered by restored PV/TT information, tactics, and history.
4. MultiPV uses repeated root searches excluding already selected root moves.
5. Recursive nodes use Negamax Alpha-Beta and Principal Variation Search.
6. At depth zero, quiescence resolves check evasions and searches captures, promotions, and bounded quiet checks.
7. Stable depths are published. If a deeper attempt times out, the last stable lines remain visible and the Worker retries with a larger slice.

## Ordering

- restored PV move;
- TT move;
- promotions;
- MVV-LVA plus SEE and capture history;
- killer moves;
- countermove;
- endgame checks where useful;
- quiet history.

Quiet and capture history tables are aged between root positions instead of being fully discarded.

## Selectivity

Implemented mechanisms include:

- mate-distance pruning;
- razoring;
- reverse futility;
- null move with high-depth verification;
- ProbCut on sound tactical moves;
- futility and Late Move Pruning;
- SEE pruning;
- Late Move Reductions with PV/improving/killer/passed-pawn adjustments;
- singular extension;
- check, recapture, promotion, and advanced passed-pawn extensions.

Gardner-specific safeguards:

- aggressive pruning is disabled in low-material or likely zugzwang endings;
- null move requires substantial non-pawn material and no near-promoting pawn;
- checks, promotions, passed pushes, TT moves, and forcing recaptures receive reduced or no LMR;
- extensions have a hard cumulative budget;
- being in check disables most forward pruning.

## Structured result cache

`analysis-cache.js` stores sanitized results rather than engine objects:

- engine version;
- completed and selective depth;
- nodes, elapsed time, NPS, and hash usage;
- up to five lines and 24 UCI plies per PV;
- next depth and completion state.

The cache key includes the compact board/side, halfmove clock, and a bounded recent-history signature. Up to 96 entries are retained. The deepest completed result wins over a shallower transient result.

For each displayed root PV, the UI stores a bounded corridor of legal continuation seeds, removing one additional move and reducing the inherited depth at each step. These are ordering/continuation seeds, not claims that every descendant independently completed the same search.

## Finite play engine

Difficulty levels combine:

- a time limit;
- maximum depth;
- MultiPV breadth;
- a centipawn margin for eligible alternatives;
- a temperature controlling near-best variation.

Levels 1–9 may choose among close completed lines to create meaningful strength separation. Level 10 is deterministic and always takes the highest-ranked line. Previously analyzed results may be reused, but the legal move is always resolved again against the current board before it is played.

## Evaluation

The evaluator is side-to-move relative internally and converted to White-relative scores for the UI. Terms include:

- Gardner-specific piece values and piece-square tables;
- tapered king placement;
- central control and space;
- mobility;
- king-zone attacked squares and attacking-piece count;
- absolute pins;
- loose and multiply attacked pieces;
- rook/queen open and half-open files;
- queen proximity to the king;
- doubled, isolated, chained, blocked, passed, and protected passed pawns;
- promotion distance;
- tempo;
- endgame king activity and bare-king mop-up pressure.

The supplied PGNs are never injected into live search. They are used only by the separate benchmark tool and the optional UI book overlay.

## Worker protocol

Every request has a monotonically increasing token. The main thread ignores stale results after a move, undo, mode change, or cancellation.

Continuous Worker messages:

- `state`: thinking, paused, complete, or idle;
- `info`: stable MultiPV result;
- `error`: surfaced exception.

Pause is cooperative at search-slice boundaries. JavaScript cannot process a new Worker message during a synchronous recursive slice, so slices are bounded and the UI changes to Paused immediately while the Worker reaches the next boundary.

## Current limitations

- no complete Gardner oracle integration in engine search;
- no exact multi-piece tablebase;
- one search thread per Worker;
- JavaScript speed varies by browser/JIT and device power policy;
- SEE is approximate and used conservatively;
- persistent cached PVs are bounded and may become stale after an engine-version change (the v4 storage key isolates this version);
- PGN calibration samples continuations and is not a proof of objective equivalence.
