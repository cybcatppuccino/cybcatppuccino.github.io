# EC atlas v19

Static browser version of the v16 elliptic-curve star atlas.

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

## v19 notes

- Replaces the v18 form-style recognizer UI with the v16 full-screen star atlas UI.
- Runs without a Python server and without external math libraries.
- Loads star-map metadata and top points first, then loads sky tiles lazily.
- Loads the full compact curve database only when search, hover, or detail information is requested.
- Keeps cubic equation recognition through the JS BigInt core from v18.
- Computes detail-panel invariants, local reduction data, q-expansions, integral points, S-integral points, and C-isogeny neighbour diagnostics in the browser.
