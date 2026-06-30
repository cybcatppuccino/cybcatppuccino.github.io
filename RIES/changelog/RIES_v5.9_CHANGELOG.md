# RIES v5.9 changelog

## Integer shortform
- Preserves the v5.8 deterministic database/search pipeline and effort levels 0–7.
- Adds a final exact fallback based on compact `floor/ceil/round(a^b/c)` and denominator-power forms such as `a^b/c^d`.
- Adds a decimal-block exact fallback only when every compact symbolic search fails.
- The fallback is exact: expressions are checked against the target integer before display.

## Algebraic recognition
- Adds arbitrary-length decimal parsing into exact rational form before numerical approximation.
- Adds complex parsing for forms such as `a+b*i`, `a+-b I`, `-2.5 - 1.1i`, including whitespace and upper/lowercase `i`.
- Uses the maximum significant precision of the real/imaginary parts, capped by the UI working precision.
- Searches integer relations on `(1, z, z^2, ...)` using BigInt-scaled real and imaginary constraints.
- Uses a multi-precision ladder so high-precision inputs still reveal low-height true relations.
- Adds conservative exact irreducibility checks for degree 1–4, rational-root rejection, quadratic discriminants, quartic factorization checks, and modular certificates for higher degrees.

## Notes
- This is still a browser-only implementation with no external math backend. It is designed for useful recognition and exact candidate verification, not for formal proof certification of every high-degree case.
