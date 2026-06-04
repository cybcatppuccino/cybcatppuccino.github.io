# RIES v10.4.1

Small confidence-sorting fix for decimal/algebraic searches.

- Fixed `log10MagnitudeAny()` for small `BigInt` values.  The previous large-integer approximation was incorrectly applied to small coefficient heights, producing negative log10 penalties such as height 9 -> about -14.  This made many irreducible algebraic rows look artificially short.
- Changed confidence ordering to use visible formula/equation length as the primary key both inside each module and when ordering each round-robin layer.
- Final confidence display now explicitly collects complete module queues and renders layer by layer: each module's shortest row first, then each module's second-shortest row, etc.
- Added a regression test for the reported `0.418991077502...` ordering where `L(f,1)` and the first log row must not be buried under multiple algebraic rows.
- Preserved v10.3/v10.4 integer validation, packed DB integration, and MathJax/LaTeX behavior.
