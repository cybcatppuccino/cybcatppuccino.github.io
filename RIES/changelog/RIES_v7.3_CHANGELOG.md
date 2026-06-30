# RIES v7.3

- Added `assets/lfunctions-l2l4.js`, generated from the supplied L2/L4 database, for browser-side access to L(f2,1), L(f4,1), and L(f4,2) values.
- Added the v7.3 L-function decimal matcher for literal real decimal inputs. It scans all nonzero L(f2,1), L(f4,1), and L(f4,2) values and reports up to three candidates in each of three families: rational tests for `x^i * π^j / L0` with `i ∈ {-2,-1,1,2}` and `j ∈ [-3,3]`, log-product relations, and quadratic-algebraic tests for the same powered ratio.
- The matcher uses Decimal.js verification when the input carries more precision than a double, but keeps a faster Number-guided search for candidate discovery. Output is beautified into formulas of the form `x = ...` involving the matched L-value.
- Log matching includes the traditional 2, 3, 5, and π basis and, for higher-precision decimal inputs, also considers log(2), log(3), Γ(1/3), and Γ(1/4) product factors.
- Added `tools/test_lfunc_v7_3.js`, `tools/lfunc_v7_3_test_results.md`, and `tools/lfunc_v7_3_generated_test_data.json`. The generated data contains 108 low-to-high precision cases (7, 12, 16, and 25 significant digits) across the three L-value families and rational/log/quadratic relation families; the committed smoke test verifies representative cases in Node.
