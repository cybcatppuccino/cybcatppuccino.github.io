# cybcat homepage

Static GitHub Pages site.

## Pages

- `index.html` — homepage.
- `test.html` — HPDB introduction and database search.
- `hypergeometric-motives.html` — hypergeometric motives and paramodular forms.
- `ries.html` — browser RIES page.
- `puzzleday.html` — standalone puzzleday page.
- `pool.html` — pool simulator.
- `mine/index.html` — minesweeper.

`hadamard.html` redirects to `test.html` for old links.

## RIES v11.2.3 note

v11.2.3 is a focused low-precision algebraic relation-search performance update: it removes irreducibility filtering from algebraic polynomial candidates, no longer labels algebraic output as irreducible, adds cached scaled-power generation for PSLQ/LLL verification, and routes real finite decimal inputs through decimalScaledPowers() instead of the complex scaled-power path.

## RIES v11.2.2 note

v11.2.2 is a focused constant-database performance update: it keeps other modules' LLL/PSLQ strategy unchanged, adds bounded PSLQ/LLL probes for constant-database relations with coefficient bound |a_i| ≤ 100, and sets constant database budgets to level 4/5/6 = 15/45/135 seconds.
