# EC atlas v32

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

## v32 notes

- Replaces the v18 form-style recognizer UI with the v16 full-screen star atlas UI.
- Runs without a Python server and without external math libraries.
- Loads star-map metadata and top points first, then loads sky tiles lazily.
- Loads the full compact curve database only when search, hover, or detail information is requested.
- Keeps cubic equation recognition through the JS BigInt core from v18.
- Computes detail-panel invariants, local reduction data, q-expansions, integral points, S-integral points, and C-isogeny neighbour diagnostics in the browser.


## v32 changes

- Keeps the v31 UI and data contract while reducing initial JS work through lazy cubic recognizer loading.
- Uses native JSON text loading for small files and throttled streaming progress for large indexes.
- Speeds repeated substring search work by caching lowercase row keys and avoiding duplicate conductor-bucket passes.
- Makes C-isogeny matrix warmup/yielding asynchronous to reduce long main-thread stalls.
- Extends integral and S-integral searches to 1.5x the previous time budget and couples the original brute-force scan with safe Mordell-Weil group operations generated from points found during the same run.
