# Orion JS 6.0 engine design

## Purpose

Orion is a local classical Gardner Chess engine for interactive browser analysis and play. It runs in ES-module Web Workers, uses no NNUE or neural model, and keeps recursive search off the UI thread.

Two Workers are intentionally separate:

- `worker.js`: continuous, resumable MultiPV analysis;
- `play-worker.js`: one finite move search when an AI-controlled side is to move.

## Representation

- native `Int8Array(25)` board;
- positive piece codes for White and negative codes for Black;
- packed integer moves: 5 origin bits, 5 destination bits, 3 promotion bits;
- in-place make/unmake restoring capture, counters, side, and two 32-bit Zobrist keys;
- precomputed knight, king, pawn, and sliding-ray geometry;
- strict compact 5×5 and Gardnerfish-padded FEN parsing.

## Hashing and transposition safety

The repetition hash contains pieces and side to move. The transposition key additionally mixes the capped halfmove clock, preventing an otherwise identical board at halfmove 0 and halfmove 99 from sharing an unsafe value near the fifty-move horizon.

The two-way typed-array TT stores key, independent lock, depth, score, static evaluation, best move, bound, and generation. Mate scores are normalized by ply. A separate direct-mapped evaluation cache avoids recomputing position-only evaluation terms.

Cached browser results are accepted only when their engine identifier exactly matches `Orion JS 6.0`. A saved mate line is also replayed and revalidated before it can terminate a new Worker search.

## Search pipeline

1. Restore a valid cached result for the requested position when available.
2. Reuse legal saved PV moves and root scores for move ordering.
3. Search with iterative deepening, aspiration windows, root MultiPV exclusion, Negamax Alpha-Beta, and PVS.
4. Use quiescence at the horizon for check evasions, captures, promotions, and bounded quiet checks.
5. Sanitize every stable root line. A mate score survives only if its PV legally ends in checkmate at the encoded distance.
6. Restore the exact root snapshot if a deadline exception unwound through outstanding make/unmake frames.
7. In low material, optionally run a separate bounded DTM proof search.
8. Publish the last complete stable result; an interrupted deeper iteration never replaces it.

## Correctness guards introduced in v5

### First-searched-move invariant

Selective pruning must never eliminate every child and allow the `-INF` sentinel to escape as a real score. Late-move/futility and losing-capture SEE pruning now require at least one legal child to have been fully searched first.

### Excluded-node semantics

A singular-extension verification search can exclude the TT move. If that leaves no candidate, the result is a failed bound (`-INF + ply`), not checkmate or stalemate.

### Verified mate contract

A displayed mate requires all of the following:

- the numerical value lies in the mate-score band;
- the PV length agrees with the encoded mate distance;
- every PV move is legal from the root;
- the final position has no legal moves and the side to move is in check.

Rejected claims fall back to a non-mate root estimate and increment the diagnostic `rejectedMateClaims` counter.

## Repetition and cycle handling

Historical positions are pre-counted in a root `Map`, making each history lookup O(1). Search-line occurrences are read from the per-ply Zobrist stacks.

- At the actual root, a draw requires the formal third occurrence.
- Below the root, the second occurrence is treated as a repeatable cycle because the sequence can be repeated until a claim is available.
- Repetition/cycle checks occur before TT use.
- Fifty-move and insufficient-material draws are handled independently.

## Ordering

- restored PV move;
- TT move;
- promotions;
- MVV-LVA plus SEE and capture history;
- killer moves;
- countermove;
- useful endgame checks;
- quiet history.

Quiet and capture histories are aged between positions rather than discarded.

## Selectivity

Implemented mechanisms include mate-distance pruning, razoring, reverse futility, null move with verification, ProbCut, futility and Late Move Pruning, SEE pruning, LMR, singular extension, and bounded check/recapture/promotion/passed-pawn extensions.

Gardner-specific safeguards:

- aggressive pruning is disabled in sparse or likely-zugzwang endings;
- null move requires meaningful non-pawn material and no near-promoting pawn;
- checks, promotions, passed pushes, TT moves, and forcing recaptures receive reduced or no LMR;
- a check-evasion extension is bounded by the global extension budget;
- being in check disables most forward pruning;
- aspiration windows widen with the magnitude of the prior score.

## Low-material DTM proof search

For at most six pieces, Orion can spend a separate small budget on an exact bounded AND/OR proof:

- attacker turn: one proven child is sufficient; choose the shortest mate;
- defender turn: every legal reply must remain proven; choose the longest delay;
- cycle, fifty-move, or insufficient-material outcomes refute the mate proof;
- iterative distance limits are 20 plies for at most four pieces, 15 for five, and 11 for six;
- successful results are stored in a 256-entry local proof cache keyed by board, side, halfmove context, and root-history signature.

This is not a retrograde-generated complete tablebase. A timeout or null result means “not proven in this budget,” not “draw.”

## Evaluation

The evaluator is side-to-move relative internally and converted to White-relative values for display. Terms include Gardner-specific material and piece-square tables, tapered king placement, center/space, mobility, king-zone pressure, pins, loose and multiply attacked pieces, open and half-open files, queen proximity, pawn structure, promotion distance, tempo, king activity, and bare-king mop-up pressure.

The UI no longer maps these values to heuristic White/Draw/Black percentages. It presents only the centipawn value or a verified mate/DTM result.

## Structured result cache

`analysis-cache.js` stores sanitized, versioned values:

- engine version;
- completed/selective depth;
- nodes, elapsed time, NPS, and hash usage;
- up to five lines and 24 UCI plies per PV;
- mate-verification and DTM-proof flags;
- next depth and completion state.

The cache key includes board/side, halfmove clock, and bounded recent-history signature. Up to 96 entries are retained. Old engine versions are ignored.

## Finite play engine

Difficulty levels combine time, maximum depth, MultiPV breadth, an eligible centipawn margin, and controlled near-best variation. Level 10 always chooses the highest-ranked completed line. Cached lines are resolved against the current legal move list before play.

Low-material finite searches reserve a bounded fraction of their move budget for DTM proof. Continuous Analysis uses its own independent proof slice.

## Current limitations

- no complete Gardner oracle is injected into live search;
- no complete retrograde endgame tablebase;
- the DTM solver is bounded and only active up to six pieces;
- one search thread per Worker;
- JavaScript throughput varies by browser/JIT and device power policy;
- SEE remains an approximate conservative ordering/pruning tool;
- PGN calibration is a regression sample, not a proof of objective equivalence.


## v6 play-strength policy

Continuous Analysis remains full-strength. Finite play levels 1–9 ignore deep analysis resumes and select among legal MultiPV candidates using level-specific shallow limits, Gaussian score uncertainty and rank-biased error probabilities. Level 10 is deterministic and may resume cached work.
