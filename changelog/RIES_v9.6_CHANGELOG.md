# RIES v9.6

Sorting-only update from v9.5.

- Sort by confidence is the default final display order.
- Confidence sorting now treats precision as a typed-precision gate/bucket and primarily ranks accepted candidates by visible formula length, coefficient/height simplicity, and compactness.
- Short sparse log|c| products, small-coefficient L-rational formulas, compact RIES equations, and low-height algebraic equations are promoted over longer LLL/PSLQ artefacts with only modestly better residuals.
- Original discovery/group order remains available via the Original order button.
