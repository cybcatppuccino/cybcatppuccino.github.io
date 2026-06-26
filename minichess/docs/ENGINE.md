# Orion JS 9.0 engine design

## Purpose

Orion is a local classical Gardner Chess engine for browser analysis and play. It runs in ES-module Web Workers, uses no neural model, and keeps recursive search off the UI thread.

- `worker.js`: continuous, resumable MultiPV analysis.
- `play-worker.js`: finite full-strength move search for AI-controlled sides.

## Representation and hash safety

- native `Int8Array(25)` board;
- packed integer moves;
- in-place make/unmake with two 32-bit Zobrist keys;
- precomputed non-sliding attacks and sliding rays;
- repetition key contains pieces and side to move;
- TT identity also mixes the capped halfmove clock;
- two-way typed-array transposition table plus direct-mapped evaluation cache.

Only results whose identifier is exactly `Orion JS 9.0` are restored. Cached mate lines must legally replay to the encoded checkmate before they are treated as solved.

## Core search

1. Restore a valid cached result and legal root ordering information.
2. Probe the lazy Gardner tablebase for positions with at most six pieces.
3. Search with iterative deepening, aspiration windows, root MultiPV exclusion, Negamax Alpha-Beta and PVS.
4. Use quiescence for check evasions, captures, promotions and bounded checks.
5. Apply TT, killer, history, countermove, capture-history, MVV-LVA and SEE ordering.
6. Use mate-distance pruning, razoring, reverse futility, verified null move, ProbCut, futility/LMP, SEE pruning, LMR, singular extension and bounded tactical extensions.
7. Replay-validate all claimed mate PVs and restore the exact root after deadline aborts.
8. Optionally run bounded low-material DTM proof when no tablebase result is available.

## Repetition and formal draws

Root history is pre-counted in a map, and search-line repetition uses the per-ply Zobrist stack. Repetition is checked before TT lookup.

- the actual root requires the formal third occurrence;
- below the root, a second occurrence is treated as a repeatable cycle;
- fifty-move and insufficient-material rules are handled separately;
- tablebase and verified mate results override heuristic evaluation scaling.

## Generic low-progress model

The v9 evaluator measures whether either side can make useful progress instead of matching one exact pawn formation. For both colours it derives:

- legal and safe mobility;
- pawn advances, captures and promotion threats;
- passed-pawn progress;
- sound captures and forcing checks;
- improving centralising/attacking moves;
- king-zone pressure and heavy-piece presence;
- locked-pawn ratio and available pawn breaks.

These terms produce a `closure` value and a conservative evaluation scale. Scaling is strongest only when both sides are constrained, no meaningful pawn break exists, tactical contact is low, and neither side has an advanced passer or promotion threat. Queens and heavy-piece activity sharply limit the reduction.

This is deliberately a draw *tendency*, not a solved result. It prevents a blocked extra bishop or nominal material edge from becoming several pawns of evaluation when neither side has a constructive route.

The same classifier protects search quality. In a low-progress node Orion disables or softens aggressive forward pruning and LMR, because waiting moves, repetitions and a single structural break are disproportionately important on a 5×5 board.

## Five full-strength play styles

Finite play no longer weakens the search. Every style receives a full-depth result (and tablebase result when available):

- **Balanced:** choose the objective top line.
- **Aggressive:** favour open play, checks, exchanges and sound sacrifices.
- **Conservative:** favour stable conversion, low volatility and drawing resources when worse.
- **Cunning:** run a bounded opponent MultiPV reply probe and prefer positions with a large best-reply gap or few acceptable replies.
- **Pressing:** favour space, king pressure and restriction of the opponent's safe moves.

Non-balanced styles require MultiPV. Each candidate gets an objective utility and a style profile. Selection first constructs a near-best safety pool:

- a searched win/draw cannot be exchanged for a clearly losing move;
- verified mate candidates stay within the mate class;
- exact tablebase WDL cannot be changed by style;
- margins shrink when the engine is clearly winning.

Only then is the secondary style score applied. The primary objective remains winning and, failing that, drawing.

## Tablebase path

`GardnerTablebase` lazily probes the local GardnerTB/GardnerPracticalTB files. It loads only the current material metadata and required gzip block, keeps a bounded LRU, and falls back to Alpha-Beta on missing or unknown sparse records. This is a project-specific format, not orthodox `.rtbw/.rtbz` Syzygy.

## Current limitations

- the generic closure model is heuristic and may still misjudge rare long manoeuvring wins;
- Cunning's response difficulty is based on a bounded search, not a human-error model trained from games;
- styles using MultiPV spend some nodes on breadth and may finish one iteration shallower than Balanced;
- sparse practical tablebase misses remain unknown;
- the online DTM solver is bounded;
- JavaScript speed varies by browser, device and power policy.
