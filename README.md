# cybcat's homepage

Static GitHub Pages site.

## Pages
- `index.html` — redesigned homepage with random weight 2 / weight 4 newform card.
- `test.html` — HPDB introduction plus database search page.
- `hypergeometric-motives.html` — table for Hodge type `(1,1,1,1)` and paramodular forms.
- `ries.html` — browser RIES-lite v6.9 page with level up to 9, exact integer shortform search up to effort 7, expanded elegant fallback/shortform coverage, safer high-precision algebraic-number recognition, nonblocking projected-tesseract progress, copyable results, high-precision expression evaluation, structured large-integer templates, and bounded/local-plus-optional-external integer factorization.
- `tools/LLL_reference.py` — the uploaded Fraction-based LLL reference kept with the package; the browser implementation mirrors its exact Gram-Schmidt/reduction structure in BigInt JavaScript for algebraic/log relation searches.
- `puzzleday.html` — standalone playable puzzleday page.
- `pool.html` — pool simulator.
- `mine/index.html` — AI minesweeper.

`hadamard.html` is retained only as a redirect to `test.html` for old links.


## RIES v6.3

RIES v6.3 improves fallback elegance for small and medium integers, upgrades the progress/status UI with animated geometric motifs, removes unnecessary vertical scroll regions, enlarges result/high-precision displays, adds copy controls for candidates/formulas/values, and hardens the high-precision algebraic recognizer so low-degree, low-height true relations are preferred over low-precision overfits.


## v6.4

This build strengthens algebraic-number recognition with an exact BigInt LLL fallback, adds high-precision algebraic residual display, adds a precomputed <=10^5 integer shortform table, fixes continued-fraction copy, limits log matches to two rows, and replaces the progress decoration with a stable rotating tesseract projection.


## v6.5

- Keeps ordinary finite decimals with about 25 or fewer significant digits in mixed mode, so RIES and log-combination rows remain visible beside algebraic candidates. Long high-precision decimal inputs still switch to algebraic-only display when a strong irreducible relation is found.
- Raises the default algebraic relation degree to 10 and uses exact BigInt-rational LLL on more precision rungs for algebraic and log-relation lattices.
- Adds a multi-candidate precomputed shortform table for integers up to 100000 and skips deeper shortform search when the database already provides five compact, diverse forms.
- Expands deterministic integer database offsets for 9--15 digit targets, allowing three- and four-digit residual constants at higher effort levels.
- Replaces the search ornament with a minimal canvas-rendered rotating 4D hypercube projection and keeps progress monotone within each solve.
- Makes continued-fraction copy more robust on local/file origins by falling back when the Clipboard API rejects the request.
- Extends integer factorization with trial division by primes up to 10000 and longer Pollard-Rho cutoffs at higher effort for 40--55 digit integers.

## v6.6

- Moves all RIES changelog files into `changelogs/`.
- Suppresses the mechanical decimal-split fallback `A*10^B+C`.
- For integer inputs with at least 16 digits, uses only structured database/template search with enlarged constants rather than the generic exact shortform engine.
- Recursively prettifies six-digit constants and clean denominators in ratio fallbacks.
- Replaces the canvas tesseract with a smoother SVG/CSS projected tesseract and keeps the progress bar color stable.
- Adds an optional worker-isolated external quadratic-sieve attempt for unresolved 40+ digit composite remainders, with an Alpertron ECM/SIQS handoff link when local factorization is incomplete.

## v6.7

- Fixes the integer-input progress bug by yielding a browser paint frame before expensive integer phases.
- Reworks local factorization into an async, UI-yielding path for trial division and Pollard-Rho.
- Separates the progress bar from the projected 4D cube widget.
- Normalizes the tesseract into a fixed square so it does not become tiny or overly flat in the 2D projection.
- Keeps the projected tesseract spinning continuously with compositor-friendly CSS while a search is active.
- Updates status progress labels and keeps monotone progress behavior across integer, RIES, algebraic, and log phases.

## v6.9
- Skips high-precision algebraic reconstruction by default for expression-evaluated targets and for direct decimal inputs with fewer than 20 significant digits; RIES/log search still runs normally.
- Replaced the synchronous structured integer database call in the solve path with a responsive async database phase that yields at nested-loop boundaries. This fixes searches getting stuck at `Checking precomputed and structured integer database…`.
- Reduced synchronous exact-shortform slices and fixed the undefined backup base limit used by the fallback structured search.
- Updated UI yielding so the SO(4) tesseract keeps animating during integer search phases.

## v6.8

- Uses a true canvas-rendered SO(4)-style hypercube animation whose rotation direction drifts smoothly while a search is running. Vertex dots are removed and edge colors shift slowly.
- Cleans copy buttons so factorization timings and non-math labels are not copied; integer-only result tables no longer show an error column.
- Adds a disabled-by-default option for external/worker QS or ECM handoff and keeps local factorization bounded when that option is off.
- Fixes remaining structured integer database stalls by adding inner-loop deadline checks and shorter synchronous slices.
- Refines mobile layout so controls and result tables stay within the viewport while retaining all functions.

