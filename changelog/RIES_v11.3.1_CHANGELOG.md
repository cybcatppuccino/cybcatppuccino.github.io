# RIES v11.3.1 changelog

Small follow-up to v11.3 for the low-precision sparse linear-combination matcher.

- Increased the linear-combination module time budget from the former under-1s window to a fixed 3s budget, so the tiered height <= 36 three-term search can cover more low-priority constant pairs before stopping.
- Capped the module return value and live progress previews at `RIES_LOWPREC_LINEAR_COMBO_LIMIT = 5`, so this module never emits more than five displayed candidates.
- Updated the UI progress wording and regression checks to match the 3s budget and five-candidate cap.
