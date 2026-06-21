# EC atlas v35

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

## v35 notes

- Keeps the v31/v32/v33 UI and data contract.
- Runs without a Python server and without external math libraries.
- Loads star-map metadata and top points first, then loads sky tiles lazily.
- Loads curve/detail/search data only when search, hover, or detail information is requested.
- Keeps cubic equation recognition through the JS BigInt core, lazy-loaded from `js/ec_core.js`.
- Computes detail-panel invariants, local reduction data, q-expansions, integral points, S-integral points, and C-isogeny neighbour diagnostics in the browser.

## v35 changes

- S-integral searches now explicitly reuse the ordinary integral-point cache. Every S-integral Compute click imports all currently found integral points because they are automatically S-integral for every `S`.
- Each S-integral Compute click also spends a bounded part of the run continuing the integral enhanced search when that integral cache is still incomplete, then imports the new integral points and group seeds before continuing the S-denominator search.
- Integral low-height rational seeds and heuristic Mordell-Weil subgroup seeds are shared into the S-integral search state, so examples such as `5712.o1` with `S=2` no longer start from an empty S-integral result when the integral search has already found points.
- The S-integral denominator scan trims the `d=1` ranges already covered by the integral x-frontier, reducing repeated work while preserving the separate S-smooth denominator frontier.
- Repeated S-integral clicks remain resumable: both the integral cache and the S-integral denominator/rational-seed frontiers continue from previous state instead of intentionally restarting already scheduled ranges.
- Retains the v34 fixes: the integral x-scan includes `x=0`, `S=±1` is treated as ordinary integral search, low-height rational-point seeds feed the group generator, and conductor-prefix search supports cached incremental rendering.
