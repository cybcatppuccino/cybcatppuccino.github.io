# V18 homepage patch manifest

This patch is relative to the original uploaded homepage zip.

## Modified

- `index.html`

## Added

- `ec-recognizer/index.html`
- `ec-recognizer/style.css`
- `ec-recognizer/app.js`
- `ec-recognizer/ec-worker.js`
- `ec-recognizer/README.md`
- `ec-recognizer/py/ec_core.py`
- `ec-recognizer/js/ec_core.js`
- `ec-recognizer/js/smoke_test.mjs`
- `ec-recognizer/data/curves_by_j.json`

## Main v18 changes

- JS BigInt exact-arithmetic recognizer is now the primary engine.
- Pyodide/Python is retained only as fallback/cross-check.
- Curve data loading has multiple URL attempts plus an emergency fallback for key examples.
- The UI no longer blocks recognition while Pyodide loads.
- The JS path computes `j`, `c4`, `c6`, discriminant, coefficients, and Q-minimal model for coefficient lists, long Weierstrass equations, and the tested rational-point plane cubic examples.

## Tested cases

```text
[3,3]
[0,0,1,3,3]
y^2 + y = x^3 + 3*x + 3
x^3 + x^2*y + y^3 + y^2 - 2*x + 1 = 0
u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0
X^3 + 2Y^3 + 1 = 15XY
```

`node ec-recognizer/js/smoke_test.mjs` passed before packaging.
