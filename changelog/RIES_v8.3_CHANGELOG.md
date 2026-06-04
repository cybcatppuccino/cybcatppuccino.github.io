# RIES v8.3 changelog

- Keeps v8.2/v8.1 functionality and does not roll back any search features.
- Switches the RIES page from a large inline startup script to a deferred external `ries-script.js`, so the DOM can parse and paint before the solver initializes.
- Makes the `commandPreview`/`previewEl` dependency self-healing: if the DOM node is missing, the script creates a safe fallback instead of aborting initialization.
- Lazy-loads `assets/shortform100k.js` only when an integer solve needs the precomputed shortform table, while preserving the same integer shortform/database functionality.
- Changes MathJax loading to async so CDN latency cannot block RIES startup.
- Adds `.nojekyll` so GitHub Pages can serve the repository as a static site without Jekyll transformations.
- Adds v8.3 startup/static-deploy smoke tests.
