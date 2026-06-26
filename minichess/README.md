# Gardner MiniChess Lab — v9

A static, browser-native 5×5 chess application with legal Gardner rules, editable positions, merged research books, lazy Gardner tablebase probing, and a local classical Alpha-Beta engine.

## Apply this patch

Copy the v9 patch over a complete v8 installation. The patch contains changed/new web and test files only. It deliberately excludes generated tablebase blocks and the original PGN/PDF archive.

Keep the existing database directory intact:

```text
tools/gardner_tablebase/tables/
```

After copying, force-refresh the browser so old v8 modules and cached analysis are not reused.

## Run

The project uses ES modules and Web Workers, so serve the project root over HTTP:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. On Windows, `serve.bat` provides the same local server.

## Five full-strength play styles

The old 1–10 weakness ladder is replaced by five result-aware styles. Every style uses the strongest Orion search and probes the local tablebase first.

- **Balanced** — pure best-line selection, equivalent to the former level 10 objective play.
- **Aggressive** — among near-equal moves, prefers checks, exchanges, open positions and sound sacrifices.
- **Conservative** — protects the searched result, values repetition/fortress resources and stable conversion.
- **Cunning** — prefers sound positions in which the opponent has few good or obvious replies.
- **Pressing** — prefers space, king pressure and moves that restrict the opponent's safe choices.

Style is applied only after objective MultiPV search. A hard result-preservation pool prevents a stylistic preference from replacing a searched win/draw with a clearly losing move. Balanced always selects the top objective line.

## Generic closed-position and draw tendency model

Orion JS 9.0 no longer depends only on one hard-coded pawn-wall pattern. It estimates whether useful play is drying up from both sides' actual legal resources:

- movable pawn breaks and promotion threats;
- safe mobility and improving moves;
- sound captures and forcing checks;
- king pressure and space;
- locked-pawn ratio;
- the number of reasonable moves available to both players.

When neither side has a credible breakthrough and useful choices are very limited, material and positional advantages are scaled toward equality. This is an evaluation tendency, not a mathematical tablebase claim. Exact tablebase, verified mate, repetition, fifty-move and insufficient-material results still take precedence.

Search selectivity is also reduced in these low-progress nodes: null move, ProbCut, aggressive futility/LMP and large LMR are disabled or softened so defensive waiting moves and rare pawn breaks are not discarded.

## Endgame tablebase probing

`js/engine/tablebase.js` continues to read the Gardner-specific files produced by the supplied Python generator:

```text
tools/gardner_tablebase/tables/
├── manifest.json
├── practical-manifest.json          # optional
├── KQvK/
├── ...
└── practical/                       # optional
```

The loader remains lazy and runs in the engine Workers. Missing or unproved sparse records fall back to Orion search.

## Tests

From the project root:

```bash
node tests/core-tests.mjs
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v5-engine-tests.mjs
node tests/v8-tablebase-and-fortress-tests.mjs
node tests/v9-style-and-draw-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/client-regression-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/play-worker-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/v6-layout-and-strength-tests.mjs
node tests/v7-cache-and-lazy-tests.mjs
node tests/v8-mobile-and-tablebase-wiring-tests.mjs
```

See `docs/V9-AUDIT.md` and `docs/ENGINE.md` for implementation and validation details.
