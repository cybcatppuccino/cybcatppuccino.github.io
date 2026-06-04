# RIES v8.2 changelog

- Restored the missing `commandPreview` DOM node used by the RIES startup guard, fixing the v8.1 `previewEl` initialization failure without reverting any v8.1 functionality.
- Updated the visible page/version labels and guard fallback wording from v8.1 to v8.2.
- Kept the existing RIES, algebraic, log-relation, L-function, integer factorization, database, shortform, copy, progress, and mobile-layout behavior unchanged.
- Added a conservative loading improvement: preconnects to jsDelivr.
- Added a v8.2 startup smoke test that checks the actual `ries.html` document contains every DOM id required by the initializer.
