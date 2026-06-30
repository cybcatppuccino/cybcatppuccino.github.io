# RIES v11.2.5 changelog

- Added a low-precision sparse linear-combination search module after RIES equation generation and before L-function matching.
- The new module uses a Q-basis-pruned constant catalogue, coefficient height <= 36, at most three constants, and returns five candidates with LaTeX/copy output.
- Result rendering treats the module as its own round-robin group and includes English explanations plus numeric values for constants appearing in a hit.
