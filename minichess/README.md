# Gardner MiniChess Lab — v8

A static, browser-native 5×5 chess application with legal Gardner rules, editable positions, merged research books, a local JavaScript Alpha-Beta engine, Player-vs-AI/AI-vs-AI modes, and lazy Gardner endgame-table probing.

## Apply this patch

This v8 delivery is intended to be copied over the complete v7.1 project. It contains changed/new web files only and deliberately excludes generated tablebase blocks. Do not delete your existing:

```text
tools/gardner_tablebase/tables/
```

After extraction, that directory should contain `manifest.json`, the material folders, and—when generated—the optional `practical-manifest.json` plus `practical/` folders.

## Run

The project uses ES modules and Web Workers, so serve the project root over HTTP:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. On Windows, `serve.bat` provides the same local server.

## v8 mobile interface

- Smaller, guaranteed-square phone board sized at roughly two thirds of the former layout.
- Thin vertical engine-evaluation rail to the left of the board.
- Analysis, Book and Edit remain visible in the compact toolbar.
- Two-row, non-clipping play settings.
- Only the two strongest principal variations are shown on phones.
- Moves is collapsed by default on phones and expands inside the fixed viewport.
- The phone page uses `100dvh` and does not require whole-document vertical or horizontal scrolling.

## Endgame tablebase probing

`js/engine/tablebase.js` reads the Gardner-specific files produced by the supplied v7/v7.1 Python generator:

```text
tools/gardner_tablebase/tables/
├── manifest.json
├── practical-manifest.json          # optional
├── KQvK/
├── ...
└── practical/                       # optional
```

The loader runs inside the existing engine Workers. It:

- checks tablebases only for positions with at most six pieces;
- lazily fetches one material metadata file and the required gzip block;
- supports exhaustive exact-core WDL+DTM and verified sparse practical records;
- selects winning/drawing moves by probing legal child positions;
- falls back to Orion search when a file or sparse continuation is absent;
- keeps a bounded decompressed-block LRU cache.

The format is GardnerTB/GardnerPracticalTB, not orthodox `.rtbw/.rtbz` Syzygy data.

## Search changes

Orion JS 8.0 retains the v7.1 classical search stack and adds:

- tablebase results for continuous Analysis and levels 8–10 play;
- tablebase/fortress results in persistent analysis cache;
- earlier ordering of historical repetition resources when the side to move is worse;
- memoized failed bounded mate/fortress probes across iterative-deepening chunks;
- conservative draw scaling for directly locked two-wing pawn walls with a lone opposite-colour bishop.

The reported position

```text
5/3b1/p1k1p/P3P/1K3 b - - 0 1
```

is now kept near equality instead of temporarily displaying a bishop-sized advantage such as `Bf6 -4.3`. This is a draw-oriented evaluation safeguard, not a claim that every structurally similar position has been formally solved.

## Research data

The original supplied PGN/PDF files remain under:

```text
data/pgn/
data/reference/
```

They are still demand-loaded only after Book is enabled or Game Tree is expanded.

## Tests

Run the JavaScript suites from the project root:

```bash
node tests/core-tests.mjs
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v5-engine-tests.mjs
node tests/v8-tablebase-and-fortress-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/client-regression-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/play-worker-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/v6-layout-and-strength-tests.mjs
node tests/v7-cache-and-lazy-tests.mjs
```

To exercise the browser tablebase loader, serve a generated `tables/` directory and run:

```bash
node tests/tablebase-loader-tests.mjs
```

See `docs/V8-AUDIT.md` for format, fallback, mobile-layout, and validation details.
