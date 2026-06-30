# RIES v11.4.1 changelog

- Replaced the previous 420k direct hard-constant asset with `assets/ries-harddb-v11_4_1-filtered.js`, a directly loaded filtered subset of 79,932 rows.
- Selection policy: parameter fraction/integer height `<= 15`, category caps for dominant generated classes, and scoring that favors smaller parameter height, lower formula complexity, and moderate magnitudes.
- The hard-constant matcher still runs in the same third decimal-search position and preserves the v11.4 checks: `|x/A|` height `<= 10`, `log|x|/log|A|`, `log|x|/A`, and `x/log|A|` against `{-2,-1,-1/2,1/2,1,2}`.
- Result metadata now reports filtered row and original source row in the uploaded 420k database.
- The low-precision algebraic relation pass now tests only the current typed input precision rather than sweeping lower/adjusted precision levels.
