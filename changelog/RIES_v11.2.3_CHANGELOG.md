# RIES v11.2.3 changelog

- Removed irreducibility filtering from algebraic polynomial candidates in the PSLQ/LLL algebraic relation paths, including the low-precision algebraic pass.
- Changed visible algebraic candidate labels from `irreducible algebraic:` to `algebraic relation:` so results no longer claim irreducibility.
- Added `scaledPowersForAlgebraic()` with per-search caching for exact scaled-power data used by LLL construction and high-precision verification.
- Real finite decimal inputs now prefer `decimalScaledPowers()`; `complexScaledPowers()` is used only for genuinely complex decimal inputs, with a fallback for compatibility.
