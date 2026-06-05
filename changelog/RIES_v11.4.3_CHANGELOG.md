# RIES v11.4.3 changelog

- Updated visible/cache-busting RIES version labels from v11.4.2 to v11.4.3.
- Changed the filtered hard-constant database from an initial page payload into a lazy-loaded package. The page now loads the 80k hard database only when a RIES search can actually use it.
- Added package-loading status text and progress updates so the existing tesseract animation panel is shown while the shortform or hard-constant packages are loading.
- Connected the hard-constant database to the confidence/result ordering system as an independent module. Its matches now participate in the same interleaved module ordering: first result from each module, then second result from each module, and so on.
- Expanded hard-constant database explanations. Result details now include the source family, formula, parameter meanings, local variable meanings, match rule, and acceptance tolerance in compact form.
- Replaced generic hard-database placeholders for common Log/Exp/Trig compositions with formulas reconstructed from the Mathematica generator templates.
- Added `UPDATE_GUIDELINES.md` at the package root to document versioning, changelog placement, README protection, UI wording, testing, performance, and release workflow rules for future updates.
- Updated the packaging smoke test for v11.4.3 to verify lazy hard-database loading, independent hard-database sorting category, explanatory formula rendering, root update guidelines, and unchanged README policy.
