# Cubic → Elliptic Curve Recognizer v18

Static browser feature for the homepage.

## v18 architecture

- Primary engine: `js/ec_core.js`, a pure JavaScript BigInt exact-arithmetic implementation.
- Fallback/cross-check: `ec-worker.js` loads Pyodide and runs `py/ec_core.py` when available.
- Curve lookup: `data/curves_by_j.json`, a static atlas index grouped by exact `j` string.
- No server backend is required.

## Supported inputs

```text
[3,3]
[0,0,1,3,3]
y^2 + y = x^3 + 3*x + 3
x^3 + x^2*y + y^3 + y^2 - 2*x + 1 = 0
u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0
X^3 + 2Y^3 + 1 = 15XY
```

The JS engine implements coefficient-list parsing, standard long Weierstrass recognition, rational-point plane-cubic transformation, reduced Q-minimal model recovery from integral invariants, and Aronhold ternary-cubic j-invariant fallback.

## Local test

From the homepage root:

```bash
python3 -m http.server 8000
```

Open:

```text
http://127.0.0.1:8000/ec-recognizer/
```

Do not open the file directly with `file://`, because Web Worker/fetch behavior differs across browsers.

## JS smoke test

```bash
cd ec-recognizer/js
node smoke_test.mjs
```
