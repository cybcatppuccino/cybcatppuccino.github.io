# Gardner MiniChess Lab v19.7

v19.7 is a conservative tablebase-proof update on top of v19.6. It preserves the browser UI's fixed **500 ms** analysis paint cadence, transposition table, evaluation cache, move ordering, and resident GTB data caches.

## What changed in v19.7

- Added a six-piece **database bridge proof** pass to Analysis. It is activated only after a completed stable principal variation has already demonstrated a legal path into a supplied <=5-piece exact WDL+DTM tablebase position.
- A bridge result is not inferred from one PV. It is an AND/OR certificate: the side seeking the result chooses a policy; the opponent's legal replies are all covered; each leaf must be an immediate terminal result or a resident exact tablebase entry.
- Winning certificates display `≤#N` or `≤-#N`. This is a true finite DTM upper bound, not a claim that N is shortest. The main engine continues analysing, and may later replace the bound with a smaller one or a separately verified exact mate.
- A completed PV that enters an exact WDL=0 tablebase node now also starts a bridge pass, but is only a candidate. Draw certificates still require two proofs: White can force an exact-tablebase draw and Black can force an exact-tablebase draw. Only then does the bridge pass replace the normal score with `0.00`.
- Exact-tablebase territory is now a proof boundary. A missing / WDL-only / nonmatching <=5-piece position fails the bridge branch rather than being searched speculatively.
- The bridge pass pre-warms required exact tablebase blocks before proof search and uses synchronous resident probes during proof construction. Network/decompression timing cannot create a certificate.
- Candidate moves for the proving side are bounded and ordered heuristically; all resisting-side moves are exhaustive. This can miss a proof under a tight budget, but cannot create a false `mate ≤ N` result.

## Checked example

For:

```text
5/k4/p1p2/2P1P/2K2 w - - 0 3
```

Analysis can obtain a completed PV beginning `Kb1`. The bridge prover then covers every legal black reply after `Kb1` and establishes:

```text
White mate ≤ 41 ply  (≤ 21 White moves)
```

This is an upper bound, not an exact DTM. The direct branch `Kb1 ...a2 Kxa2` reaches `KPPvKP` at ply 3 with exact DTM 24; the three black king replies are independently covered by the same AND/OR proof and the worst branch reaches an exact `KPPvKP` winning leaf after at most 41 total plies.

## Deployment

Copy the v19.7 patch files over an existing v19.6 installation. Keep the existing `tools/gardner_tablebase/tables/` directory from v19.6. Serve the project through HTTP/HTTPS so browser workers can fetch and decompress tablebase blocks.
