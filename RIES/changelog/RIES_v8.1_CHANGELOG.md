# RIES v8.1 changelog

- Fixed the v8 packaging regression where `RIES/ries.html` could load with no DOM body and throw `Cannot read properties of null (reading 'addEventListener')`.
- Added startup guards around UI event binding so a missing control reports a visible error instead of blanking the page.
- Moved all RIES changelog files into the `changelog/` directory.
- Reordered decimal-output groups: low-precision decimals show RIES, log|c| linear combinations, algebraic approximations, then L-function matches; high-precision decimals show algebraic approximations, then L-function matches.
- Improved L-function formula display with compact `f` notation, separate `N.k.#` modular-form metadata, and MathJax-rendered q-expansions.
- Improved mobile layout so the top navigation and RIES panels use the same available width, with a continuous soft gradient background.
