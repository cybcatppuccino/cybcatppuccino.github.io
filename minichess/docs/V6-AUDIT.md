# v6 UI, unified tree, variants and play-strength audit

## Scope

v6 keeps the Orion classical analysis engine at full strength while redesigning the play experience around a fixed, responsive workspace. The continuous Analysis worker remains independent from the finite Play worker.

## Unified game tree

The former per-PGN selector was removed. The tree now builds one temporary neighbourhood from three sources:

1. the complete local game tree, including played branches;
2. book moves aggregated across every loaded PGN source;
3. the current three engine principal variations.

Nodes with the same move are merged. The current path receives priority, while unrelated side branches are reduced to their leading continuation. Clicking a local node navigates the local tree; clicking a synthetic book or engine node replays its legal path from the position at which the neighbourhood was built. The viewport centers only when the active local node changes, so live AI updates do not pull the tree away while the user is exploring it.

## Starting layouts

The start-layout selector supports:

- Standard Gardner;
- central symmetry;
- MiniChess 60 (uniform 5! permutation mirrored by rank);
- central MiniChess 60;
- independent pure-random 5! back ranks;
- Mallett Chess (`RNKQN` versus `RBKQB`).

The supplied `MalletM25.pgn` is preserved byte-for-byte and added to the unified archive. It parses without skipped moves.

## Play-level redesign

Levels 1–9 deliberately no longer inherit deep Analysis cache scores. They use shallower time/depth limits, wider MultiPV candidate sets, level-specific score noise and rank-biased blunder probabilities. Every selected action still comes from the legal Alpha-Beta root results. Level 10 remains deterministic and maximum strength.

A fixed-seed distribution regression test checks the expected separation. In the v6 test position, average selected candidate rank was approximately:

- level 1: 4.85;
- level 3: 3.00;
- level 5: 1.52;
- level 7: 0.72;
- level 9: 0.11;
- level 10: 0.00.

These figures are behavioral regression targets, not Elo estimates.

A second calibration pass searched six real positions sampled from the supplied Gardner and Mallett archives, then repeatedly applied each level's legal root-selection policy. Average zero-based candidate rank fell monotonically from 4.32 at level 1 to 0.38 at level 9 and 0.00 at level 10. The complete position-level report is stored in `data/level-calibration.json`; it is a regression aid, not an Elo claim.

## Human-versus-AI undo

Undo now returns to the nearest preceding node at which the human side is to move. This normally removes the player's move and the AI reply together; if the AI has not replied yet, it removes only the pending human move.

## Responsive layout

- Desktop: board, analysis/moves and unified tree in three fixed columns.
- Tablet: board and analysis above, unified tree below.
- Phone: board/basic controls, compact continuous analysis and current moves only.
- Short landscape screens: the same phone feature set is arranged in two compact columns.

The document itself remains non-scrollable. Individual analysis, move and tree panels own any necessary internal overflow.

## Validation

The release suite covers rules, notation, engine regressions, the v5 false-mate position, endgame proofs, Worker streaming, pause/resume, finite AI play, AI-vs-AI, six start-layout modes, Mallett PGN parsing and the stochastic strength gradient.
