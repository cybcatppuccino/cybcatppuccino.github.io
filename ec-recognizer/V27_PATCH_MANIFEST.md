# V27 Patch Manifest

Changed/new files only.

## Changed files
- `app.js`
  - Bumped version to v27.
  - Fixed the settled-render path: after drag, wheel zoom, pinch, or animated travel stops, the atlas now immediately invalidates the render pool and renders the detailed star shapes/starbursts without waiting for hover or another event.
  - Added JSON fetch promise de-duplication and curve-detail caching to prevent repeated clicks from launching duplicate loads.
  - Fixed the cubic-like search detector word-boundary regex.
  - Made curve-detail loading more parallel and removed duplicated reduction-table computation for q-expansion.
  - Added stale-request cancellation tokens for detail panel, C-isogeny search, integral-point search, and S-integral search.
  - Changed C-isogeny neighbour computation to yield more frequently, report progress, cache per curve, and stop updating stale panels.
  - Optimized relation-matrix generation by filtering det>512 and replacing BigInt gcd calls with integer gcd.
  - Enriched C-isogeny neighbours with full detail rows before rendering, so each item shows label, relation, Weierstrass form, group, discriminant, and j-invariant.
  - Stopped search-result clicks from being swallowed by global document click handling.
- `style.css`
  - Fixed search dropdown clipping caused by `contain: paint` on `.search-wrap`.
  - Increased search-result visibility and dropdown stacking.
  - Replaced compact one-line C-isogeny entries with full-height rich cards.
- `index.html`
  - Updated version badge from v26 to v27.

## New file
- `V27_PATCH_MANIFEST.md`
