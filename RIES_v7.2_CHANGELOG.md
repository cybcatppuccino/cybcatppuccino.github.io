# RIES v7.2

Incremental update over v7.1 focused on decimal precision routing, stronger natural integer fallbacks, and stricter equivalence cleanup.

## Changes

- For decimal inputs with fewer than 20 significant digits, v7.2 now runs all three paths: low-precision algebraic recognition, traditional RIES equation search, and log-combination testing. The low-precision algebraic search no longer pads short decimals up to an artificial 8-digit floor.
- For decimal inputs with at least 20 significant digits, v7.2 keeps the high-precision algebraic-only behavior.
- Added additional natural integer fallback families: binomial-mix, binomial-multiple, near-square / near-triangular, and structured floor/ceil quotient forms.
- Removed misleading fallback labels such as `A^B+C^D` when the actual expression contains binomial or factorial terms; labels now describe the family generically while the candidate text shows the exact expression.
- Improved equivalence filtering for additive constants, cancelling offsets, multiplicative powers, and floor/ceil quotient shifts such as `floor((N+20)/3)` versus `floor(N/3)+6`.
- Fixed division display parenthesization so expressions like `A/(B·C)` are not rendered ambiguously as `A/B·C`.
- Expression targets that evaluate to exact integers continue through the integer pipeline, with case-insensitive support for `!`, `binom/C/choose/nCr`, `A/perm/nPr`, `gcd`, `lcm`, `fib/fibonacci`, and `catalan`.
