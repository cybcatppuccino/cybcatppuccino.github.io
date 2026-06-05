# RIES v11.1.1 changelog

## Decimal / constant database

- Added an early targeted transformed-polynomial pass for decimal inputs whose `log|x|` or `e^x` transform is a low-degree polynomial in a database constant with a small rational denominator.
  - This fixes `2.386110381167886`, now recognized as `x ≈ exp((-12 + 4·c + c^2)/12)` with `c = π`.
  - The pass runs before the broader catalog sweeps, avoiding starvation when later transforms would otherwise exceed the time budget.

## LaTeX output

- Added LaTeX output for `log|c| linear relation` rows.
  - Uses proper LaTeX commands such as `\approx`, `\exp`, `\pi`, and `\frac`.
  - Displays exponents as `2/3`, `-4`, etc., without textual parenthesized exponent strings.
- Added LaTeX output for `RIES equation` rows using the existing expression-to-LaTeX formatter.

## Precision regression protection

- Kept the v11.1 double-based precision cap for 16–20 digit decimal inputs.
- Added regression tests to ensure sub-16-digit inputs keep their original typed precision and existing low-precision log matching behavior.
