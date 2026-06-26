# Orion JS 8.0 engine design

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

Browser results are accepted only when their engine identifier exactly matches `Orion JS 8.0`. Every restored mate line is legally replayed from the exact root before it may be treated as solved.

## Search pipeline

1. Restore a valid cached result for the requested position when available.
2. If the cache contains a replay-verified solved mate, publish it immediately and do not launch another proof/search cycle.
3. Otherwise reuse legal saved PV moves and root scores for move ordering.
4. Search with iterative deepening, aspiration windows, root MultiPV exclusion, Negamax Alpha-Beta, and PVS.
5. Use quiescence at the horizon for check evasions, captures, promotions, and bounded quiet checks.
6. Sanitize every stable root line. A mate score survives only if its PV legally ends in checkmate at the encoded distance.
7. Restore the exact root snapshot if a deadline exception unwound through outstanding make/unmake frames.
8. In low material, optionally run a separate bounded DTM proof search.
9. Publish the last complete stable result; an interrupted deeper iteration never replaces it.

## Verified mate cache contract

A displayed or persisted mate requires all of the following:

- the numerical value lies in the mate-score band;
- the PV length agrees with the encoded mate distance;
- every PV move is legal from the exact cache-key position;
- the final position has no legal moves and the side to move is in check.

In v8, a valid mate is stored with `solved: true`. A later transient non-solved/deeper update cannot overwrite it. When the user follows one or more moves of that PV, the remaining line is rebased:

- consumed moves are removed;
- the encoded mate score is adjusted by the consumed ply count;
- DTM is reduced by the same count;
- the child line is replay-validated from the child position before being cached.

A solved resume produces one cached `info` message followed by `complete`; Pause/Continue does not restart it. Invalid or stale mate claims are discarded rather than trusted.

## Repetition and cycle handling

Historical positions are pre-counted in a root `Map`, making each history lookup O(1). Search-line occurrences are read from the per-ply Zobrist stacks.

- At the actual root, a draw requires the formal third occurrence.
- Below the root, the second occurrence is treated as a repeatable cycle because the sequence can be repeated until a claim is available.
- Repetition/cycle checks occur before TT use.
- Fifty-move and insufficient-material draws are handled independently.

## Ordering and selectivity

Ordering uses restored PV, TT move, promotions, MVV-LVA/SEE/capture history, killers, countermove, useful endgame checks and quiet history.

Implemented selectivity includes mate-distance pruning, razoring, reverse futility, null move with verification, ProbCut, futility and Late Move Pruning, SEE pruning, LMR, singular extension, and bounded check/recapture/promotion/passed-pawn extensions.

Gardner safeguards disable or reduce aggressive pruning in sparse, likely-zugzwang, checked and near-promotion positions.

## Low-material bounded DTM proof

For at most six pieces, Orion can spend a separate small budget on an exact bounded AND/OR proof:

- attacker turn: one proven child is sufficient; choose the shortest mate;
- defender turn: every legal reply must remain proven; choose the longest delay;
- cycle, fifty-move, or insufficient-material outcomes refute the mate proof;
- successful results enter the same replay-verified solved cache.

This online proof is deliberately bounded and is not a complete tablebase.

## Offline Gardner tablebase path

The v7.1 generator output is now probed directly by the v8 browser Workers:

- exact exhaustive WDL+DTM for every legal 2–3-piece position by default;
- proved sparse WDL plus verified DTM upper bounds for selected 4–6-piece positions;
- PGN and legal-play reachability seeds, balanced/late-game priority and safe symmetry reduction;
- SQLite-WAL graph checkpoints and resumable retrograde propagation;
- measured 96 MiB default hard cap;
- independently compressed/checksummed material blocks and lazy probing.

A missing sparse record means unknown, so Orion continues searching normally. The web probe requests only the two manifests, one material metadata file and the single exact or sparse block containing the position. The 50-move rule is not encoded and must remain a separate engine rule consideration.


## Browser tablebase query path

Before normal Alpha-Beta search, a <=6-piece position is offered to the lazy `GardnerTablebase` reader. Exact-core records select moves by probing every legal child. Practical records use their proved best move and/or child records; if no proved legal continuation is available, the hit is not converted into an arbitrary PV and normal search resumes. Gzip blocks are decompressed inside the Worker and retained in a bounded LRU. Tablebase results are persisted through the v8 analysis cache.

## Evaluation and play strength

The evaluator is side-to-move relative internally and converted to White-relative values for display. It uses Gardner-specific material/PSTs, tapered king placement, center/space, mobility, king pressure, pins, loose and multiply attacked pieces, files, pawn structure, promotion distance, tempo, king activity and mop-up pressure.

Continuous Analysis remains full-strength. Finite play levels 1–9 use lower limits and controlled legal-move errors; level 10 is deterministic and may resume cached work.

## Current limitations

- no complete Gardner oracle is injected into live search;
- tablebase coverage is limited to locally generated exact/sparse files; missing positions remain unknown;
- the online DTM solver remains bounded;
- one search thread per Worker;
- JavaScript throughput varies by browser/JIT and device power policy;
- SEE remains an approximate conservative ordering/pruning tool;
- PGN calibration is a regression sample, not a proof of objective equivalence.
