# RIES v7.1

Incremental update over v7 focused on integer-expression targets, cleaner integer output, and low-precision algebraic recognition.

## Changes

- Exact integer results from computable expressions now enter the full integer pipeline instead of stopping at the high-precision preview.
  - Examples supported by the evaluator include `5!`, `binom(m,n)`, `C(m,n)`, `A(m,n)`, `perm(m,n)`, `gcd`, `lcm`, `fib`, `fibonacci`, and `catalan`.
  - Function names are case-insensitive.
- Sub-20-significant-digit decimal inputs now run the low-precision algebraic-number recognizer.
  - High-precision reconstruction is still reserved for inputs with at least 20 significant digits.
  - The low-precision path uses a shorter precision ladder so simple algebraic targets such as `sqrt(2)` and `phi` are easier to hit.
- Integer output polishing was strengthened.
  - Obvious multiplicative equivalents are collapsed by canonical factor signature, so forms such as `3^10·9` are represented by cleaner equivalents such as `9^6` when available.
  - Rounded-ratio family keys continue to merge mathematically equivalent `floor`/`ceil` plus small-offset variants.
  - Final filtering removes `round(...)` and keeps only exact, floor, or ceil certificates.
- Fallback behavior is less mechanical.
  - The old decimal split style `A*10^B+C` is no longer used as the displayed fallback path.
  - Natural fallbacks based on power sums, products, binomial/factorial structures, and rounded power ratios are preferred.
- UI cleanup.
  - The large white rounded background card on the RIES page is removed.
  - The visible `Target` label line is hidden for a cleaner input area.

## Validation performed

- Syntax checked `ries-script.js`, `ries_inline.js`, and the inline script inside `ries.html`.
- Tested exact expression evaluation for `5!`, `C(10,3)`, `A(10,3)`, and `binom(10,3)`.
- Tested expression-to-integer routing with `3^10*9` and verified the database output simplifies to `9^6`.
- Tested low-precision algebraic recognition with `1.41421356237`, which returns `x^2 − 2 = 0` as the top relation.
- Tested fallback output filtering to avoid decimal-split forms.
