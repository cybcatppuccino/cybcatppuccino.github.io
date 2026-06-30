# RIES v11.7.3 harddb prune report

## Summary

v11.7.3 removes the requested overlapping harddb groups and turns the active harddb into a single pruned lazy-loaded table.  Depths 4, 5, and 6 no longer differ by rows; they differ only by comparison constants.

| Metric | Rows |
|---|---:|
| v11.7.2 harddb filtered rows | 79,932 |
| Removed in v11.7.3 | 56,324 |
| Active in v11.7.3 | 23,608 |

## Deleted categories

| Category | Removed rows | Reason |
|---|---:|---|
| low-height hypergeometric pFq | 3,048 | Covered by the independent hypergeometric database module. |
| Euler beta integral fast | 2,555 | Overlaps with fast/generated beta data and the newer integral/sum style. |
| incomplete beta integral fast | 30,000 | Large generated fast block removed from harddb. |
| beta logarithmic integral fast | 15,000 | Large generated fast block removed from harddb. |
| gamma log-laplace integral fast | 221 | Fast generated gamma-log/Laplace block removed from harddb. |
| rational Mellin integral fast | 5,500 | Fast generated Mellin block removed from harddb. |

## Remaining active harddb categories

| Category | Rows |
|---|---:|
| common Log-Exp-Trig composition | 20,000 |
| Gauss hypergeometric value | 411 |
| generalized hypergeometric value | 411 |
| incomplete beta integral | 284 |
| trigonometric Mellin integral | 284 |
| elliptic Pi integral | 284 |
| Lerch transcendent sum | 284 |
| beta logarithmic integral | 213 |
| Bessel K Mellin integral | 212 |
| Barnes/Gamma products | 213 |
| special-function zeros | 143 |
| Bessel J Mellin integral | 94 |
| Euler beta integral | 86 |
| gamma log-laplace integral | 71 |
| exponential rational Laplace integral | 71 |
| log-one-minus integral | 71 |
| elliptic K integral | 71 |
| elliptic E integral | 71 |
| Hurwitz zeta sum | 71 |
| incomplete gamma integral | 71 |
| log-one-minus integral fast | 71 |
| trigonometric Mellin integral fast | 59 |
| rational Mellin integral | 32 |
| polylog root-of-unity sum | 30 |

## Matching constants by depth

| Depth | Rows scanned | Rational multiplier cap | Extra scalar constants |
|---|---:|---:|---|
| 4 | 23,608 | 8 | simple: ±1, ±1/2 |
| 5 | 23,608 | 12 | core: adds ±2, ±3, ±1/3 |
| 6 | 23,608 | 20 | extended: adds ±4, ±3/2, ±2/3 |

The UI exposes a maximum rational multiplier height.  The effective stage cap is `min(user height, stage cap)`, so lowering the UI value still constrains every depth.

## Files changed for active behavior

- `RIES/ries.html`
- `ries-script.js`
- `ries_inline.js`
- `assets/ries-harddb-v11_7_3-level4.js`
- `assets/ries-harddb-v11_7_3-stats.json`

The old `assets/ries-harddb-v11_6-level4.js` and `assets/ries-harddb-v11_6-level5.js` are no longer referenced by the active script.  They can be removed from a full repository checkout if desired; this v11.7.3 delivery is a patch package and only contains files changed since v11.7.2.
