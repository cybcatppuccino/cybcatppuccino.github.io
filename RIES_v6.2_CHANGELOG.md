# RIES v6.2 changelog

## Algebraic-number search

- Raised the default algebraic relation sweep to degree 8 for decimal and complex decimal inputs.
- Raised the default coefficient-height budget to `1e12` and the default residual slack to 2, making low-degree algebraic recovery less likely to miss genuine degree 1--8 relations.
- The solver now performs the small-degree sweep with a wider result cap before final ranking, while still filtering candidates through irreducibility and residual checks.

## High-precision expression results

- High-precision expression evaluation is no longer mixed into the RIES candidate table.
- Computed exact integers, decimals, and complex values now appear in a dedicated `High-precision value` card above the candidate table.
- The card shows the first 100 digits by default and can expand to 1000 digits when available.

## Extra number-analysis panel

- Added a collapsed panel: `Continued fraction, base expansions, and digit statistics`.
- The panel computes only after the user opens it, keeping the page responsive.
- For real integers, exact finite decimals, and real computable-expression results, it displays:
  - continued fraction expansion,
  - base 2, 3, 5, 10, and 16 representations,
  - digit statistics in those bases.

## Layout and mobile polish

- The RIES page is now height-constrained with internal scrolling, so long results do not force the page beyond the viewport.
- Result tables, high-precision cards, and analysis panels have independent scroll regions.
- Mobile layout now stacks controls cleanly, uses full-width buttons, and keeps long numeric output readable.
