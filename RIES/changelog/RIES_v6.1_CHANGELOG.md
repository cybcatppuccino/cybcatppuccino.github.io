# RIES v6.1 changelog

## UI and workflow

- The RIES page now opens with an empty input box and a plain prompt.
- Removed all example-number buttons from the initial target area.
- The Continue button now resets whenever the target input changes. For integer inputs it continues the deterministic shortform search at the next effort; for non-integer/RIES inputs it continues at the next RIES level.
- Replaced the mathematical-symbol loading decoration with a lightweight geometric/lattice animation using CSS gradients and clip-path only.

## LaTeX output

- Replaced the earlier regex-like LaTeX converter with a small expression parser.
- The converter now respects parentheses, operator precedence, factorials, powers, products, quotients, floor/ceil/round, and binomial forms.
- Compound bases and products such as `(2+3)^4`, `(2+3)·4`, and `floor(8/81·10^11)+2` now render with explicit LaTeX grouping.

## Integer shortform search

- Added a wide first-round structured search at default effort so that the first solve already tries broader combinations instead of waiting for the final safety net.
- Added a tiny pretty database for values up to 100000, keeping the best expressions with at most four digits.
- When a residual or correction term has at most five decimal digits, v6.1 tries to replace it with a compact expression such as a small factorial, power, binomial, or short combination.
- Existing exact verification remains unchanged: every displayed integer shortform is checked against the BigInt target before display.
