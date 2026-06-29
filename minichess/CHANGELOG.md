# Gardner MiniChess v20.4

## v20.4 — cache hygiene and endgame scheduling

- Ordinary alpha-beta TT/eval/proof-miss caches are now root-local. After a played move, a fresh Orion search starts from a clean ordinary cache unless a separate replay-verified mate/draw proof cache applies.
- Durable analysis cache remains proof-only, and `AnalysisCache.set()` no longer returns an older payload when the current result is ordinary numeric analysis.
- 6-8 piece pawn/rook endings receive a slightly larger bounded side-proof budget while ordinary alpha-beta remains the majority of the work.
- The v20.3 near-terminal mate proof and Kb1 bridge behaviours are preserved.
- Added v20.4 cache-policy/endgame regression coverage.

Runtime identifiers:

- Engine: `Orion JS 20.4`
- Application / storage namespace: `v20.4`
