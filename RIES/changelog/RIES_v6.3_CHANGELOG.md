# RIES v6.3 changelog

## Search and algebraic recognition
- Reworked high-precision algebraic relation ranking so low degree, low height, low residual candidates are favored instead of being displaced by higher-degree overfits.
- Re-verifies algebraic relations at the highest available input precision before displaying them, reducing false positives from low-precision sweeps.
- Suppresses noisy huge-height linear decimal-rational artifacts when a meaningful algebraic candidate exists.
- Tested the requested samples so the following are recovered as leading candidates:
  - `1.3937513393975333742689` → `32x^3 + 6x - 95 = 0`
  - `11.76819789039794939573150647941410958744` → `2x^3 - 196x - 953 = 0`
  - `3.656789004808549181390450598221380319090943564` → `x^5 + 2x^4 - 16x - 953 = 0`

## Elegant fallback / shortforms
- Enlarged the small-number pretty-expression cache through `10^5`, including more factorial, power, binomial, rounded square-root/cube-root, and compact arithmetic families.
- Allows equal-digit shortforms for 3+ digit numbers when the expression is visibly more elegant than the literal integer.
- Uses the pretty-expression cache inside fallback decomposition so larger numbers inherit more elegant components.

## Interface and readability
- Compressed the top RIES header: removed the redundant Target title and made the version/formula-hunting line smaller and inline.
- Removed unnecessary vertical scroll boxes in result, high-precision, and number-analysis panels so the page can expand naturally.
- Enlarged result text and high-precision / 1000-digit decimal displays.
- Added copy controls beside candidates, formulas, values, number tools, and high-precision output.
- Replaced the plain searching indicator with a gradient progress bar and animated geometric motifs inspired by Platonic solids and classical planar constructions.
