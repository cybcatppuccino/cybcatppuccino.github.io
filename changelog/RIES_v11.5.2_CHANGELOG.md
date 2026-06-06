# RIES v11.5.2

Based on v11.5.1. This release focuses on exposing the existing search modules through a clearer Parameters panel while keeping the default search pipeline compatible with v11.5.1 unless the user changes settings.

## UI and Parameters

- Rebuilt the Parameters panel around module-level controls. Each major search module now has an on/off checkbox; disabling a module collapses its detailed options.
- Removed the old **General** block title. The external factorization handoff option remains available in the compact search-depth/display block.
- Displayed RIES internal levels as user-facing **depth** values.
- Set all visible RIES unary/binary operation checkboxes on by default, including trigonometric and root/log-base operations.
- Added UI controls for low-precision linear combinations, Möbius variants, constant database transforms/passes, harddb, hypergeometric pFq database depths and multiplier families, L-function result families, and integer search submodules.
- Added a maximum relative error display/filter control. Precision internals remain hidden.

## Search integration

- Connected the new parameter controls to `readSettings()` and to the relevant module gates so UI choices affect the actual search.
- Included parameter-state fields in cache keys to avoid reusing results from incompatible UI settings.
- Kept harddb depth behavior from v11.5.1: it remains available only at depth 5.
- Kept pFq database staged loading behavior from v11.5.1, while allowing users to disable stages or multiplier families.
- Added integer-search sub-controls for factorization, structured/precomputed database use, and deterministic shortform search.

## Confidence sorting

- Kept module-wise round-robin confidence ordering: all modules' first results, then all modules' second results, and so on.
- Adjusted the ordering inside each round-robin layer and module queue to be mostly precision-first.
- Added a simplicity promotion rule: very low-height or few-term candidates can move ahead when their residual is still within about one order of magnitude of the typed-input tolerance.

## Tests

- Added v11.5.2 packaging and hypdata smoke tests.
- Updated syntax checks for `ries-script.js` and `ries_inline.js`.
