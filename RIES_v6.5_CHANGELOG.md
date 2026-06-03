# RIES v6.5 changelog

- Preserve RIES/log rows for normal precision decimal inputs while reserving algebraic-only display for long high-precision decimal strings.
- Default algebraic degree is now 10, max 14, with exact BigInt-rational LLL applied to more precision rungs.
- Log matching now also passes through the exact LLL reducer, which follows the uploaded Fraction/Gram-Schmidt LLL structure adapted to browser BigInt arithmetic.
- Added multi-candidate `RIES_SHORTFORM_100K_MULTI` precomputed table and database-early-stop behavior for compact integer matches.
- Expanded large-integer database residual constants according to digit length and effort.
- Replaced the progress ornament with a minimal rotating 4D hypercube projection on canvas.
- Fixed continued-fraction copy fallback and strengthened bounded integer factorization.
