# Gardner MiniChess v20.5 patch notes

v20.5 focuses on side-to-move correctness and mate display integrity in endgames. It keeps the v20.4 cache hygiene policy, enforces side-to-move ordering before worker publication, and adds a bounded replay verification pass for mate scores discovered by ordinary alpha-beta so verified #N lines can be displayed when the searched mate is independently confirmed.

Key changes:

- Worker results are normalized by current side-to-move utility before publication, preventing a losing verified-mate line from jumping ahead of a stronger defensive move.
- Ordinary alpha-beta mate claims are not shown raw; when a completed search returns a mate score, v20.5 tries to replay-confirm the root move with an independent AND/OR proof. Confirmed claims display as `#N`; unconfirmed claims fall back to real numeric search scores or remain unpublished.
- Endgame search remains majority alpha-beta, with proof work used only for verification/refinement.
- Added regression coverage for the reported `r1k2/P1p1p/1pP1P/1B3/R2K1 b - - 12 15` position.

Validation target: Engine `Orion JS 20.5`; storage namespace `v20.5`.
