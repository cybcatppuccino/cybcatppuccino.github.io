# RIES v11.3.1 changelog

- Reworked the low-precision sparse linear-combination matcher tolerance: accepted residuals now track roughly 100x the user-visible decimal precision, with a small binary64 safety floor.
- Removed the hard five-candidate behavior for this module; it returns the rows found inside the tolerance window instead of padding toward five.
- Added priority-tiered meet-in-the-middle scans for three-term formulas with coefficient/denominator height <= 36, favoring constant pairs common in integrals and sums before lower-priority mixed special-function pairs.
- Fixed LaTeX escaping for linear-combination rows (`\approx`, `\frac`, and coefficient spacing `\,`).
- Added run metadata (`_linearComboPairDone`, `_linearComboPairTotal`, `_linearComboExhaustiveComplete`, `_linearComboTolerance`) for debugging coverage under the module time budget.
