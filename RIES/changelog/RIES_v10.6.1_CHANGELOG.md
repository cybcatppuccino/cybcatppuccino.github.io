# RIES v10.6.1

Small Möbius-module bugfix release.

- Runs the Möbius decimal module independently of the Log/Algebraic checkboxes.
- Gives direct, log|x|, and exp(x) Möbius variants independent pair-search budgets so early variants cannot starve later ones.
- Adds a deterministic sparse low-height matcher before LLL, catching identities such as `1 + γ` and `exp(π/√3)` in the initial pass.
- Fixes coefficient formatting for the constant `1`, so `2·1` displays as `2`, not `21`.
- Normalizes common integer factors in Möbius numerator/denominator coefficients, preventing duplicate scaled forms such as `2π/(2√3)`.
- Adds regression coverage for `γ + 1 = 1.577215664901533` and `e^(π/√3) = 6.1337074062362276`.
