# RIES v10.7

- Added a low-precision Constant Database module for direct decimal inputs up to 20 significant digits.
- Added `assets/constantdb300.js`, containing 190 uploaded named constants and 110 generated elementary constants.
- The module tests the transforms `x^-2`, `x^-1`, `x^-1/2`, `x^1/2`, `x`, `x^2`, `exp(x)`, and `log|x|` against database constants.
- It supports degree-1/degree-2 ratio recognition for `b/c` and one-constant Möbius recognition via integer relations in `1,b,c,bc`.
- Constant Database rows join the existing confidence round-robin presentation order and return at most five length-sorted candidates.
- Uploaded 190-source constants display their English descriptions in the value column.
- Kept v10.6.1 Möbius, v10.5 integer, and v10.4.1 confidence-sorting fixes.
