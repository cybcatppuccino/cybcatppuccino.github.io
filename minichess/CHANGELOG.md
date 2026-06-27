# Changelog

## v17.1

- Updated all current version labels to v17.1 / `Orion JS 17.1`.
- Fixed the remaining black-to-move mate/proof-line ordering bug: losing proof lines are no longer inserted ahead of better candidates, and analysis/play results are normalized by side-to-move utility before display or AI selection.
- Re-audited the analysis worker, play worker, cached-result reuse and style-selection paths so White maximizes White utility and Black maximizes Black utility consistently.
- Strengthened play styles without changing Gardner rules or the static evaluation meaning:
  - Balanced keeps the objective best line;
  - Aggressive, Conservative, Cunning and Pressing now preserve stable wins/large advantages before applying style preferences;
  - Cunning uses near-equivalent traps mainly in equal or worse positions, rather than sacrificing clear objective value.
- Improved play-worker efficiency under short time limits by streaming iterative internal analysis to the UI while preserving high depth ceilings and stronger MultiPV candidate information for styled AIs.
- During Human-vs-AI or AI-vs-AI play, manual Analysis mode is blocked so a second analysis worker does not waste compute.
- The analysis panel now shows the active play AI's internal candidate lines while it is thinking.
- The panel Pause/Resume button now pauses/resumes play-AI thinking and tracks active elapsed time separately, so the AI time limit is not consumed while paused.
- Added v17.1 regression coverage for black mate ordering, play-worker pause/resume, AI internal-info streaming and the stronger Cunning style policy.

## v17

- Added Local-mode boot defaults, current-game cache restore, lazy <=5-piece tablebase wiring, root short-mate safety and thin-PV cache safeguards.

## v16.1

- Fixed v16 live top-three merge ordering for black-to-move positions by ranking lines by current side-to-move utility rather than white-centric score.
