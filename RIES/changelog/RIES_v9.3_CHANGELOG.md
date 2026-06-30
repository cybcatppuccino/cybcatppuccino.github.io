# RIES v9.3 changelog

Focused decimal precision-policy fix on top of v9.2.

- Decimal/log/L-function matching now uses the precision literally typed by the user as the default acceptance scale.
- Blank log/LLL precision no longer derives from the canonical rationalized input string, which could silently use the wrong scale.
- PSLQ/LLL algebraic verification precision is clamped to the typed precision unless the user input itself has more digits.
- Low-precision log|c| and L-function prefilters are permissive enough not to reject true positives before final typed-precision verification.
- Added smoke/stress coverage for typed trailing zeroes, low-precision log|c| matches, L-function exact-value matches, Gamma constants, integer responsiveness, and state reset.
