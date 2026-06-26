# v9 AI audit

## Scope

v9 changes only the finite-play personality layer, the closed/low-progress evaluation model, and related search safeguards. No PGN, PDF or generated tablebase data is included in the patch.

## Style safety

All five exposed personalities run the maximum-strength engine. Balanced uses one objective PV. The four personalities that need choice use MultiPV and apply their preference only after a result-preserving filter.

The filter preserves:

- verified mate class;
- exact tablebase WDL;
- searched win/draw status within a conservative centipawn margin;
- a tighter margin when already clearly winning.

Synthetic regressions assert that each style picks its intended near-best candidate and rejects a superficially attractive blunder.

## Cunning response model

For each candidate root move, a small secondary search measures the opponent's first replies. The profile records:

- gap between best and second-best reply;
- number of replies within a narrow objective band;
- whether the only best reply is forcing or quiet.

This information is used only as a tiebreak among objectively safe root candidates.

## Closed-position audit

The prior exact blocked-pawn safeguard remains as a narrow emergency scale, but the main v9 model is structural and side-symmetric. It evaluates actual legal opportunities for both players and is active in opening, middlegame and endgame positions.

Regression positions include:

- the reported locked extra-bishop position, which remains close to equality;
- the normal initial position, which must not be compressed toward a draw;
- candidate style profiles and safety-pool behavior.

## Search safeguards

Low-progress nodes are treated as pruning-sensitive. Orion reduces or disables null move, ProbCut, razoring/futility, LMP, losing-capture pruning and large LMR. A rare pawn break in a closed position can receive an extension. This costs nodes only in the positions most vulnerable to horizon errors.

## Validation performed

- JavaScript syntax checks for the complete `js/` tree and `app.js`;
- core rules, FEN, SAN, PGN and make/unmake tests;
- engine regression and false-mate tests;
- tablebase/fortress wiring tests;
- generic low-progress evaluation tests;
- five-style objective-safety tests;
- finite play-Worker legal-move and cache-resume tests;
- multi-ply AI-vs-AI legal-play smoke test;
- continuous analysis, pause/resume and cache tests;
- patch-content inspection confirming no PGN/PDF/tablebase gzip payloads.

## Interpretation

A score moved toward `0.00` by the structural model means the engine sees little credible progress under its current horizon. It does not certify a mathematical draw. Formal repetition, rule draws, verified mates and local tablebase records remain distinct and authoritative.
