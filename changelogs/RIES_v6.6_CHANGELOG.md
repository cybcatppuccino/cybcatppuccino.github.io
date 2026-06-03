# RIES v6.6 changelog

- Collected all RIES changelog files under `changelogs/`.
- Removed the last-resort decimal split fallback (`A*10^B+C`) from displayed exact integer shortforms.
- For integer inputs with at least 16 digits, skipped the generic exact digit-min shortform engine and used only the structured database/template search with expanded constants.
- Extended structured database ranges for large integers: larger bases, coefficients, denominators and offsets at higher effort levels.
- Added recursive prettification for six-digit constants and denominators used by ratio fallbacks; denominators avoid fractional and rounding forms while offsets may use them.
- Reworked the progress display to use a stable-color bar and a compositor-friendly projected tesseract SVG animation rather than the previous decorative canvas.
- Added an optional worker-isolated external quadratic-sieve attempt via the browser BigInt QuadraticSieveFactorization package for unresolved 40+ digit composite remainders, plus an Alpertron ECM/SIQS handoff link when local factoring does not finish.
