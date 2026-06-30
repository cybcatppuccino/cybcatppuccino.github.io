# RIES v11.8.1 changelog

- Split long modular-form q-expansions into two display lines on the homepage random-newform card and in RIES L-function result metadata, preventing oversized coefficients from stretching the page.
- Changed L-function ranking so candidates inside the 100× typed-precision tolerance bucket are ordered by simpler shape and lower rational/algebraic height before tiny residual differences.
- Changed the L-function matcher to return one global candidate list capped by the L-function candidate limit, instead of returning that many candidates per rational/log/quadratic submatcher.
- Added Parameters panel master buttons: all module switches on, all module switches off, and restore default module switches. These buttons only change module-level toggles and preserve all submodule settings.
- Updated visible RIES version and cache-busters to v11.8.1.
