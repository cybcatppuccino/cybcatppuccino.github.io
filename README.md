# cybcat's homepage

Static GitHub Pages site.

## Pages
- `index.html` — redesigned homepage with random weight 2 / weight 4 newform card.
- `test.html` — HPDB introduction plus database search page.
- `hypergeometric-motives.html` — table for Hodge type `(1,1,1,1)` and paramodular forms.
- `ries.html` — browser RIES-lite v7.3 page with level up to 9, exact integer shortform search up to effort 7, expanded elegant fallback/shortform coverage, safer high-precision algebraic-number recognition, nonblocking projected-tesseract progress, copyable results, high-precision expression evaluation, structured large-integer templates, and bounded/local-plus-optional-external integer factorization, and L-function decimal matching for L(f2,1), L(f4,1), and L(f4,2).
- `tools/LLL_reference.py` — the uploaded Fraction-based LLL reference kept with the package; the browser implementation mirrors its exact Gram-Schmidt/reduction structure in BigInt JavaScript for algebraic/log relation searches.
- `puzzleday.html` — standalone playable puzzleday page.
- `pool.html` — pool simulator.
- `mine/index.html` — AI minesweeper.

`hadamard.html` is retained only as a redirect to `test.html` for old links.



## RIES v8.8 notes

- Based on v8.7; keeps existing integer, RIES, algebraic, and L-function features.
- Adjusts only the decimal `log|c|` linear-combination continuation path.
- At default RIES level the original selected-basis log search is preserved.
- From the first Continue level onward, the log module enumerates optional-basis additions and ordered default-basis removals, including new optional candidates `log(11)` and `e`.

## RIES v8.7 notes

- Based on v8.6; keeps the current feature set and changes only the integer shortform/database path.
- Database expressions now recursively simplify multiplicative main terms inside offsets/rounding wrappers, so forms such as `35^7·35+9`, `9^49·27-5`, and `16/4·2^35+1` display as cleaner equivalent forms.
- The <=10^8 Continue exhaustive pass is now sliced with progress labels and UI yields, so Stop can return the best rows already found instead of waiting for a long synchronous pass.
- Structured integer database progress now reports the current template family and elapsed/total budget.
- Integer factor/database/shortform rows are cached per exact integer target/settings, so repeated runs and Continue do not recompute completed effort levels.

## RIES v8.6 notes

- Based on v8.5; keeps the current feature set and focuses on decimal/L-function matching and final ordering.
- Low RIES levels try simple L-function shapes first (`x/L`, `x·π/L`, `x/(πL)`, `1/(xL)`) and keep quadratic/log L-function searches deliberately small until higher Continue levels.
- L-function formula LaTeX simplifies plain-text powers such as `2^(-2)`, `3^(2)`, and `5^(5/3)` into cleaner MathJax exponents.
- The confidence sort interleaves each module's best result, then each module's second-best result, while still keeping all accumulated result rows.
- Traditional RIES equations that verify to the user-typed precision, or one digit below it, receive a stronger sort boost so concise RIES hits are not buried by more elaborate explanations.

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

- Moves all RIES changelog files into `changelog/`.
- Suppresses the mechanical decimal-split fallback `A*10^B+C`.
- For integer inputs with at least 16 digits, uses only structured database/template search with enlarged constants rather than the generic exact shortform engine.
- Recursively prettifies six-digit constants and clean denominators in ratio fallbacks.
- Replaces the canvas tesseract with a smoother SVG/CSS projected tesseract and keeps the progress bar color stable.
- Adds an optional worker-isolated external quadratic-sieve attempt for unresolved 40+ digit composite remainders, with an Alpertron ECM/SIQS handoff link when local factorization is incomplete.


## v7.3

- v7.3 adds a browser-native L-function decimal matcher backed by the L2/L4 database. For literal real decimal inputs, it tests rational, logarithmic, and quadratic-algebraic relations involving L(f2,1), L(f4,1), L(f4,2), small powers of π, and powers x^±1 and x^±2.
- High-precision decimal inputs are checked with Decimal.js against the stored high-precision L-values, while short decimal inputs keep a looser low-precision path. L-function hits are displayed as direct formulas `x = ... L0 ...`.

## v7.2

- v7.2 refines low-precision decimal routing, adds natural integer fallback families, and strengthens equivalence cleanup for additive and floor/ceil forms.

## v7

- Restores the strongest historical integer database/shortform templates while keeping the scan asynchronous and bounded.
- Restores binomial/coefficient-aware database and fallback coverage, including stronger >=16-digit structured integer matching.
- Compacts fallback denominator and constant terms through the <=100000 precomputed shortform database, with clean denominators and correct LaTeX.
- Removes generated `round(...)` results; rounded divisions are emitted only as exact, floor, or ceil.

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



## RIES v8.2 notes

- v8.2 fixes the v8.1 startup guard regression by restoring the missing `commandPreview` element required by the RIES UI initializer.
- Keeps the v8.1 feature set intact; no historical rollback or feature removal.
- Adds a small load-path improvement for MathJax by preconnecting to the CDN.


## RIES v8.1 notes

- v8.1 keeps changelogs under `changelog/`, fixes the RIES page startup regression, reorders decimal result groups, and improves mobile layout continuity.
