Lattice Polygon Database static release v10

Open index.html through a local static server, for example:
  python3 -m http.server
then visit the printed localhost URL.

v10 changes:
- Removed the expected-Severi-by-genus table, the separate tropical-geometry panel, the A-discriminant/secondary-data panel, and the standalone enumerative package button.
- Renamed the algebraic panel to "Algebraic geometry and curve-counting data".
- Mirror period support now defaults to all lattice points.
- Constant-term periods now detect a common divisor in the nonzero powers and use the substitution z^d -> z before Picard-Fuchs guessing and related output.
- The period-order input no longer has the old 28 cap, and constant-term computation no longer has the old internal state cap.
- Picard-Fuchs guessing now prioritizes theta-degree <= 2 while allowing higher z-degree when enough coefficients are available, and computes more coefficients for guessing.
- Riemann-symbol data are shown as exponent rows for all detected special points, with exact LaTeX coordinates for finite singular points of degree <= 2.
- Classical period coefficients used by the PF package are displayed as a LaTeX generating function rather than as a table.
- Ehrhart and interior lattice-point counts are displayed through generating functions with a max-order input.

Validation performed for v10:
- node --check app.js passed.
- Removed-panel title search passed for the requested visible module titles and deleted table labels.
