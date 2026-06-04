# RIES v11 changelog

- Gives the four decimal-side modules substantially more time by RIES level:
  - level 4: 5 seconds
  - level 5: 10 seconds
  - level 6: 30 seconds
- Applies the same budget policy to L-function, log relation, Möbius relation, and constant database searches.
- Tightens constant database output quality: rows must be within roughly 10^3 of the typed input precision envelope.
- Removes forced padding to eight constant database rows when only weak nearest-rational fallback rows are available.
- Fixes constant database fallback LaTeX escaping so commands such as `\approx`, `\alpha`, and `\frac` survive rendering/copy.
- Adds regression coverage for `-2.143596015846163 = α·π`, where `α^3 + α + 1 = 0`.
