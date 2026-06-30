# RIES v8.8 changelog

Based on v8.7. Existing features are preserved; changes are limited to the decimal log-linear-combination continuation path.

## Decimal log-combination continuation

- Default RIES level keeps the original `log|c|` selected-basis implementation.
- From the first Continue level, automatic enumeration replaces the single current-basis parameter combination for the log module.
- Continue level 1 enumerates one optional candidate; level 2 enumerates pairs; level 3 triples, and so on within bounded time.
- For each optional choice, the search also tries removing the first 0, 1, 2, 3, or 4 default basis terms from this ordered list: `log(log(3))`, `log(log(pi))`, `log(5)`, `log(log(2))`.
- New optional candidates: `log(11)` and raw `e`.
- Candidate scoring favors few nonzero terms, small coefficient height, small residual, and shorter displayed products.

## Safety

- The continuation sweep is bounded by a small per-level deadline and a combination cap, so Continue remains responsive.
- If the automatic continuation sweep finds no result under the constraints, the original selected-basis log path is used as a fallback.
