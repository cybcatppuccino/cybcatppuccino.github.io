# EC atlas v34

Static browser version of the elliptic-curve star atlas.

## Run locally

From the homepage root:

```bash
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/ec-recognizer/
```

Do not use `file://`, because module imports and JSON tile loading need an HTTP server.

## v34 notes

- Keeps the v31/v32/v33 UI and data contract.
- Runs without a Python server and without external math libraries.
- Loads star-map metadata and top points first, then loads sky tiles lazily.
- Loads curve/detail/search data only when search, hover, or detail information is requested.
- Keeps cubic equation recognition through the JS BigInt core, lazy-loaded from `js/ec_core.js`.
- Computes detail-panel invariants, local reduction data, q-expansions, integral points, S-integral points, and C-isogeny neighbour diagnostics in the browser.

## v34 changes

- Fixes the integral-point boundary bug that skipped `x=0` in the first bounded x-scan.
- Treats `S=1` and `S=-1` as integral-point searches, so repeated S-integral computes continue the cached integral frontier while the list remains incomplete.
- Adds a low-height rational-point seed search. Rational points found by scanning `x=m/d²` with small `d` are fed into the same heuristic Mordell-Weil subgroup generator, and accepted only when generated points pass the integral or S-integral test.
- Keeps the brute-force integral and S-integral scans intact; low-height rational search and subgroup generation only add verified points and never replace the original bounded enumeration.
- Adds conductor-prefix search handling for queries such as `6552.` and `6552.a`, with cached conductor rows and incremental result rendering in small batches.
- Reuses the cached conductor result for narrower prefixes, so typing from `6552.` to `6552.a` filters already-loaded rows instead of starting over.
