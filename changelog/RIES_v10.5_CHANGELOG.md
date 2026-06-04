# RIES v10.5

- Replaced the optional log-linear basis item `log(log 5)` with `log Γ(1/6)`.
- Added a wide large-integer structured database pass for 16+ digit targets: `C*A^B + D` and `C*binom(A,B) + D`, with `1 <= C <= 9`, `100 <= A <= 1000`, `7 <= B <= 100`, and `|D| <= 9999` (`B <= A/2` for binomial rows).
- Added exact validation and simplification for the new wide rows; pure products such as `8·128^7` simplify to `2^52` when shorter.
- Folded tiny additive correction tails so variants like `binom(A,B)+3-1` display as `binom(A,B)+2` and deduplicate correctly.
