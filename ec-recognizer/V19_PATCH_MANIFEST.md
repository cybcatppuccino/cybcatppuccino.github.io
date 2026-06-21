# V19 patch manifest

V19 changes the homepage entry name to **EC atlas** and replaces the v18 EC recognizer page with a static browser version of the v16 elliptic-curve star atlas.

## Modified

- `index.html`
- `ec-recognizer/index.html`
- `ec-recognizer/style.css`
- `ec-recognizer/app.js`
- `ec-recognizer/README.md`

## Added / regenerated data

- `ec-recognizer/data/plot_meta.json`
- `ec-recognizer/data/top_points.json`
- `ec-recognizer/data/curves_compact.json`
- `ec-recognizer/data/sato_tate_groups.json`
- `ec-recognizer/data/tiles/*.json`

## JS core retained

- `ec-recognizer/js/ec_core.js`
- `ec-recognizer/js/smoke_v19.mjs`

## Test commands run

```bash
node --check ec-recognizer/app.js
node --check ec-recognizer/js/ec_core.js
cd ec-recognizer/js && node smoke_v19.mjs
node /tmp/mock_import_v19.mjs
node /tmp/mock_search_v19.mjs
node /tmp/mock_detail_v19.mjs
node /tmp/mock_detail_multi_v19.mjs
```

Key verified examples:

- `x^3 + x^2*y + y^3 + y^2 - 2*x + 1 = 0` returns `6291.d1`, `j = 110592/233`, and `y² + y = x³ + 3x + 3`.
- `11.a1`, `6291.d1`, and `9690.n2` detail panels compute discriminants, local data, and q-coefficients without precision loss from large coefficients.
