# RIES v12.0.2 changelog

## Performance and loading

- Prestarts HardDB, hypergeometric, and integral/sum index loads in parallel when the corresponding modules are enabled.
- Splits hypergeometric and integral/sum assets into search-index packages and display-metadata packages.
- Keeps all search-relevant values, row maps, component codes, complexity data, and multipliers in the index packages.
- Loads metadata packages only after a database hit needs formula formatting.
- Skips hypergeometric real-projection decoding for complex targets.
- Adds time-sliced progress/yield checks for hypergeometric, integral/sum, and L-function scans.
- Caches `Decimal(L.value)` objects inside L-function passes.

## Display fixes

- Normalizes integral/sum display formulas by suppressing redundant `0`, `1`, `-1`, and exponent-1/0 artifacts where they arise from generated templates.
- Preserves the original numeric search value and database row identity while cleaning candidate text and LaTeX output.

## Tests

- Updates startup tests for index/meta lazy packages.
- Adds database-module tests confirming metadata is not eagerly loaded before search hits.
- Adds integral/sum display cleanup tests for zero/unit polynomial and trigonometric artifacts.
