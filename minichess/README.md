# Gardner MiniChess Lab — v10

A static, browser-native 5×5 chess application with legal Gardner rules, editable positions, merged research books, lazy Gardner tablebase probing, and a local classical Alpha-Beta engine.

## Apply this patch

Copy the v10 patch over a complete v9 installation. The patch contains changed/new web and test files only. It deliberately excludes generated tablebase blocks and the original PGN/PDF archive.

Keep the existing database directory intact:

```text
tools/gardner_tablebase/tables/
```

After copying, force-refresh the browser so old v9 modules and cached analysis are not reused.

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

Orion JS 10.0 adds a second-stage effective-progress classifier on top of the v9 closed-position model. The first stage remains cheap and structural; the second stage only runs in plausible locked endgames and asks whether either side has a non-losing move that creates real progress:

- a genuine pawn break, passer or promotion race;
- a sound capture rather than a losing sacrifice;
- a king-entry resource;
- a quiet move that enables one of those resources next move.

Purely reversible rook, bishop and king shuffling no longer prevents draw compression. Pawn pushes into opposing pawn control are not treated as breakthroughs unless they also create a serious passer or promotion threat. If both sides only have waiting moves and no effective breakthrough, Orion reports the position as exact `+0.00` instead of preserving a cosmetic space or activity edge. Exact tablebase, verified mate, repetition, fifty-move and insufficient-material results still take precedence.

Search selectivity remains reduced in low-progress nodes: null move, ProbCut, aggressive futility/LMP and large LMR are disabled or softened so defensive waiting moves and rare real pawn breaks are not discarded.

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
node tests/v10-low-progress-draw-tests.mjs
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

See `docs/V9-AUDIT.md`, `docs/ENGINE.md`, and `CHANGELOG.md` for implementation and validation details.
