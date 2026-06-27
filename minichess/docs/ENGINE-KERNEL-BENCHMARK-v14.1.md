# Engine kernel benchmark — v14.1

Depth: 8. Scores are white-centric centipawns. Times are milliseconds from this Node benchmark run.

| Position | Category | Orion score / move | Fairy score / move | ms Orion / Fairy | Note |
|---|---|---:|---:|---:|---|
| initial-balance | opening / baseline speed | +0.00 / c2c3 | -0.52 / d2d3 | 1386 / 64 | Baseline comparison. |
| closed-deadlock-qb1 | closed fortress / draw recognition | +0.00 / e5d5 | +1.35 / b5d5 | 937 / 10 | Orion recognizes the closed draw; Fairy keeps a static material edge. |
| closed-deadlock-qc1 | closed fortress / draw recognition | +0.00 / a1b1 | +1.00 / c1b1 | 761 / 14 | Orion recognizes the closed draw; Fairy keeps a static material edge. |
| quiet-breakthrough | closed tactic / quiet breakthrough | -0.18 / a5b5 | +0.00 / a5c5 | 474 / 11 | Checks whether quiet heavy-piece offers appear before repetition lines. |
| reported-cycle-line | repetition stability | -0.18 / b1a3 | +1.05 / b2a3 | 597 / 10 | Baseline comparison. |
| mate-proof | mate search | #+ / c2d3 | #+ / c2d3 | 313 / 9 | Orion proof-search verifies mate; Fairy searches it as a normal UCI engine. |
| pawn-race-proof | low-material conversion | #- / a3b2 | +0.00 / b1c1 | 253 / 9 | Baseline comparison. |
| book-random-like | middlegame judgement | +0.48 / c3c2 | +0.38 / c3c2 | 688 / 15 | Baseline comparison. |

## Aggregate speed

- Orion: 191,112 nodes in 5409 ms, about 35,332 nps.
- Fairy-Stockfish: 10,713 nodes in 142 ms, about 75,444 nps.

## Practical reading

- Orion JS is stronger for app-specific truth conditions: compact-Gardner legal validation, mate proof flags, tablebase/fortress hooks, and closed-deadlock compression.
- Fairy-Stockfish is much faster at raw alpha-beta node throughput and is useful as an independent tactical/open-position opinion, but it does not know Orion’s bespoke fortress/deadlock model.
- In browser deployment, Fairy requires cross-origin isolation headers because this wasm build uses pthreads/SharedArrayBuffer. Use `serve.sh`, `serve.bat`, or equivalent COOP/COEP headers.
