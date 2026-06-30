# RIES v10.3

- Added an exact BigInt/rational validator for integer-mode display expressions. Every integer candidate is re-evaluated from the formula text shown to the user before it can be displayed.
- Guarded simplification: if a simplified expression no longer evaluates to the intended integer, the simplification is rejected.
- Filtered structured-product, structured-backup, database, digit, fallback, and static shortform rows through the same validator.
- Fixed the 768 regression: invalid rows such as `48^2`, `96^2`, `4^5·3`, and `2^8·6` are rejected for target 768.
- Fixed MathJax retry behavior for dynamically rendered rows and corrected LaTeX output for `\pi` and `\varphi`.
- Added `tools/test_ries_v10_3_integer_validation.js` and `tools/test_ries_v10_3_packed_db_integration.js`.
