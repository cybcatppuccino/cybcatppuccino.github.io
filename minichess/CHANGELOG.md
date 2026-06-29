# Gardner MiniChess v19.8

## v19.8 — Mixed WDL finite-bound analysis

### Analysis / endgame behavior

- Removed the runtime `TB bridge · verifying` display path.
- The normal search still uses the exact internal tablebase WDL sentinel (`±22000`) for move ordering and alpha-beta cutoffs; this is never exposed as a normal centipawn score.
- When a completed stable principal variation exposes a non-draw tablebase signal (a resident exact tail or an internal root WDL sentinel), the analysis worker preloads reachable exact blocks and launches a bounded **mixed WDL audit** for that candidate root move.
- The audit keeps exact WDL leaves decisive but caps them at `±10.00`. Any non-tablebase defensive branch is searched normally and may become the visible worst-case score.
- The audit is root-move restricted and enumerates every reply in the normal alpha-beta search. It is deliberately not a mate proof.
- A shallow initial audit (depth 5–8, 780 ms) is followed by at most one deeper refinement (up to depth 10, 1200 ms). This prevents a stream of normal iterative-deepening updates from starving the audit.
- A finite mixed-WDL audit has its own result kind. It is stable for display during the same analysis session, never persisted, never used as a fresh-root resume result, and never marked as a solved mate/draw.
- Exact root tablebase, verified forced mate, and v19.7 AND/OR bridge certificates remain the only sources permitted to publish exact tablebase outcomes, `0.00`, or `mate ≤ N`.
- If normal analysis later provides a stronger proof, that proof still outranks the finite audit.

### Stability / output

- Internal WDL scores cannot leak to the UI as `+220.00` / `-220.00`.
- The existing 500 ms UI paint throttle is unchanged.
- The capped audit publishes a complete score/PV object, rather than merging a score from one search with a PV from another.
- Ordinary search acceleration remains intact: TT, evaluation cache, tablebase block cache, root ordering, and normal iterative deepening are retained.

### Versioning

- Engine: `Orion JS 19.8`
- Application / storage namespace: `v19.8`

### Regression coverage

- v19.7 six-piece bridge sample: `5/k4/p1p2/2P1P/2K2 w - - 0 3`
  - normal analysis does not emit a placeholder or `+220.00`;
  - capped audit for `Kb1` returns `+10.00` without falsely claiming mate.
- Symmetric black-winning bridge sample returns `-10.00`.
- New Worker integration test verifies the actual asynchronous Worker publishes the finite audit result using resident tablebase files.
