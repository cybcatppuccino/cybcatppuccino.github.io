# RIES v9.6

- Integer contiguous-substring database now starts at 10 decimal digits.
- Substring database displays at most one representative result, preferring the smallest structural A, then fewer formula digits, then lower digit sum.
- Tightened low digit-budget exact integer passes and rational-power scans so high-effort Continue runs remain bounded and responsive.
- Added a hard 1.5x budget envelope for integer shortform tail/fallback work.
- Kept existing v9.5 sorting, decimal matching, L-function, log, and integer search families intact.
