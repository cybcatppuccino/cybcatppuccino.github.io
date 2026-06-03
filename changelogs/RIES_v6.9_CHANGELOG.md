# RIES v6.9 changelog

- Default algebraic reconstruction now starts only for direct decimal / finite decimal complex inputs with at least 20 significant digits. Computed-expression targets and shorter decimals run ordinary RIES/log searches without high-precision algebraic PSLQ by default.
- Replaced the blocking structured integer database phase with an async responsive phase that yields frequently and reports progress.
- Capped synchronous exact-shortform slices and fixed the undefined `baseLimit` bug in the structured backup search.
- Improved UI paint yielding so the progress bar and SO(4) tesseract animation continue during integer searches.
