# RIES v6.7 changelog

- Fixed the integer-input progress/status path that could appear frozen before the browser had a chance to paint the first progress update.
- Added explicit `nextPaint()` yielding before expensive integer factorization, structured database search, and exact shortform phases.
- Reworked local integer factorization into an async UI-yielding path for trial division and Pollard-Rho, so progress text can continue updating during longer integer runs.
- Split the progress bar and projected tesseract into separate visual components.
- Rebuilt the tesseract widget as a fixed-size normalized projected 4D hypercube. The projection is rescaled into a fixed square, avoids overly flat projections where possible, and rotates with compositor-friendly CSS animation while a search is active.
- Kept progress color stable across search modes and made progress updates monotone within each run.
- Updated page labels and README references from v6.6 to v6.7.
