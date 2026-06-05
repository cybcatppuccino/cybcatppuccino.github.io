# RIES v11.2.4 changelog

- Generalized the level-4 decimal Möbius matcher from `(aA+bB)/(cA+dB)` to `(aA+bB)/(cA+dB+e)`.
- The direct LLL relation now uses columns `[A, B, ..., -xA, -xB, ..., -x]`, corresponding to `aA+bB = x(cA+dB+e)`.
- Added Catalan's constant `G` to the Möbius constant queue and kept `π²` in the early queue.
- Updated text and LaTeX formatting so denominator constants are displayed as ordinary constants, e.g. `(π² + 8·G)/16`.
- Adjusted the Möbius variant budget so direct generalized scans do not starve `exp(x)` and `log|x|` checks.
- Added a regression test for `1.0748330721566944 ≈ (π² + 8G)/16`.
