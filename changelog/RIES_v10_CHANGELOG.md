# RIES v10

- Based on v9.6A sorting behavior.
- Integer substring database now runs for every integer input with at least 9 digits.
- Substring database rows are collapsed to at most one best representative, preferring smaller A/N, then fewer expression digits, then lower digit sum.
- High-effort integer shortform passes have stricter per-stage budgets and hard-stop accounting so they terminate within the configured time envelope.
- Final reverse pass now reports progress before entering the slice, keeping Stop more responsive.
