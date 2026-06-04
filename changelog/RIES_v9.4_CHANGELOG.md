# RIES v9.4

- Hardened integer Continue/search slices so medium integers such as `9169995354` effort 4 terminate in a bounded time and keep yielding to the UI.
- Removed manual precision/error parameter controls from the RIES page; decimal matching uses the precision typed by the user, including trailing zeroes.
- Added a bounded 16–100 digit substring database check for `r·A^B` and `r·binom(A,B)` with small integer multiplier `r`.
- Added smoke tests for v9.4 integer responsiveness, precision-control removal, substring database, and typed-precision decimal matching.
