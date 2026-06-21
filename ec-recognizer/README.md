# EC atlas v33

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

## v33 notes

- Keeps the v31/v32 UI and data contract.
- Runs without a Python server and without external math libraries.
- Loads star-map metadata and top points first, then loads sky tiles lazily.
- Loads curve/detail/search data only when search, hover, or detail information is requested.
- Keeps cubic equation recognition through the JS BigInt core, lazy-loaded from `js/ec_core.js`.
- Computes detail-panel invariants, local reduction data, q-expansions, integral points, S-integral points, and C-isogeny neighbour diagnostics in the browser.

## v33 changes

- Raises the integral and S-integral point search multiplier to 2.25x the original v31 time budget.
- Adds resumable in-memory cache for point searches. Repeated `Compute S-integral points` clicks continue from the previous frontier with half of the normal v33 budget, instead of starting over.
- Adds modular square-congruence sieves before BigInt square-root tests, reducing the number of expensive candidate checks without changing which candidates can be accepted.
- Strengthens the group-generated search layer: discovered points are tested against a small-span heuristic, promoted to an independent-looking Mordell-Weil subgroup basis when safe, and used to generate multiples and low-height linear combinations.
- Keeps the original brute-force x / denominator-height scans intact; group and basis heuristics only add accepted points and never replace the bounded scan.
- Expands S-integral denominator/height search in phases when a cached run reaches its current frontier but still cannot prove completeness.
