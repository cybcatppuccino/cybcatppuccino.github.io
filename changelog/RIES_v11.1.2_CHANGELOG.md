# RIES v11.1.2 changelog

- Reordered constant database decimal transforms to try `x`, `log|x|`, `e^x` when finite/suitably sized, `1/x`, and `x^2` before the remaining inverse/sqrt variants.
- Removed the v11.1.1 priority transformed-polynomial mini-scan from the solve path; the helper is left in the file but is no longer called by `constantDbRows()` or `constantDbRowsAsync()`.
- Increased the constant database module wall-clock budget to 1.2× of the existing RIES-level module budget.
- Adjusted the constant database deep scan to visit prioritized constants first and apply the reordered transform list inside each constant, so constants such as π are not delayed behind a full pass over lower-priority constants.
