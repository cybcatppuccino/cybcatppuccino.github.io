# RIES v10.6

- Added a decimal-input Möbius / fractional-linear relation module for inputs with at most 20 typed significant digits.
- The module checks `x`, `exp(x)` for `x <= 10`, and `log(|x|)` against `(r1 A + r2 B (+ r3 C))/(r4 A + r5 B (+ r6 C))` using the existing BigInt LLL machinery.
- Default effort enumerates two-constant bases from `1, π, e, log(2), log(3), log(π), π², e², πe, e^π, ζ(3), √2, √3, √π, φ, γ, sin(π/5), sin(π/7), sin(π/8), cos(π/5), cos(π/7), cos(π/8)`.
- Continue / higher RIES level additionally attempts three-constant bases.
- Möbius candidates are verified against the visible transformed formula and returned as at most five length-sorted rows.
- Möbius rows are integrated into confidence round-robin sorting as a distinct module.
- Added LaTeX rendering for `x≈ratio`, `x≈log(ratio)`, and signed `x≈±exp(ratio)` displays.
