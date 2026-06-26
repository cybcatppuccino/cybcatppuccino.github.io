# Gardner MiniChess Lab — v7.1

A static, browser-native 5×5 chess application with legal Gardner rules, editable positions, merged research books, a local JavaScript Alpha-Beta engine, Player-vs-AI and AI-vs-AI modes.

## Run the web app

The project uses ES modules and Web Workers, so serve the folder over HTTP:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`. On Windows, `serve.bat` does the same.

## v7.1 highlights

The browser UI and Orion JS 7.0 searcher remain compatible with v7. The main change is a redesigned offline endgame builder:

- exact exhaustive WDL+DTM for every legal 2–3-piece position;
- reachability-guided, verified sparse WDL for practical 4–6-piece positions;
- supplied Gardner/Mallett PGNs converted into a compressed practical seed corpus;
- legal capture-biased playouts and balanced/late-game priority sampling;
- collision-free combinatorial indexing with safe file/colour symmetries;
- resumable SQLite-WAL sparse graph and retrograde process;
- batched SQLite child lookup and continued expansion after the node cap is filled;
- delta-coded, checksummed, multi-file lazy blocks;
- a hard default final-directory cap of 96 MiB;
- expected normal query cache of roughly 2–20 MiB;
- one-click Windows and Linux/macOS build scripts.

The selective layer never guesses. Missing or unresolved positions are reported as unknown so the strong Alpha-Beta engine can continue searching.

## Existing v7 browser improvements

- Replay-verified mate results are durable solved cache entries.
- Cached mate lines are rebased after each consumed ply and replay-validated against the exact child position.
- Gardner rules and Game Tree are collapsed by default.
- The PGN archive is fetched only when Book is enabled or Game Tree is opened.
- Desktop and tablet Game Tree layouts use compact collapse behavior.

## Engine separation

- **Analysis worker:** continuous iterative deepening, MultiPV 3, pause/resume and persistent position/solved-line cache.
- **Play worker:** one finite search per AI turn, difficulty-specific limits and selection policy.

The engine remains a classical, non-NNUE searcher with PVS, quiescence, transposition tables, move ordering, conservative pruning and bounded low-material mate proof support.

## Recommended practical tablebase build

### Windows

```text
tools\gardner_tablebase\build_practical_windows.bat
```

### Linux/macOS

```bash
cd tools/gardner_tablebase
chmod +x build_practical_linux.sh
./build_practical_linux.sh
```

Equivalent command:

```bash
./run_linux.sh quick-generate --output tables --work work --hours 3 --target-size 96M --core-pieces 3 --node-limit 750000 --rollouts 3000
./run_linux.sh verify --tables tables
```

The command can be interrupted and rerun unchanged. Preserve the final `tools/gardner_tablebase/tables/` folder for v8 integration.

This is a Gardner-specific exact 2–3 core plus selective 4–6 overlay, not an orthodox `.rtbw/.rtbz` set and not a complete six-piece oracle. Exported sparse WDL records are proved; their DTM is a verified upper bound. Absent records remain unknown.

See:

- `tools/gardner_tablebase/README.md`
- `tools/gardner_tablebase/FORMAT.md`
- `docs/V7.1-TABLEBASE-AUDIT.md`

The original exhaustive commands are retained for research use.

## Research files

Original supplied material remains under:

```text
data/pgn/
data/reference/
```

`data/SOURCE_INTEGRITY.sha256` records supplied research-file hashes. `data/practical-seeds.json.gz` is a generated, compressed position corpus derived from those PGNs.

## Tests

Run the JavaScript suites from the project root:

```bash
node tests/core-tests.mjs
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/v5-engine-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/client-regression-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/play-worker-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/v6-layout-and-strength-tests.mjs
node tests/v7-cache-and-lazy-tests.mjs
```

Run Python generator tests:

```bash
cd tools/gardner_tablebase
python -m pip install -r requirements-dev.txt
python -m pytest -q
```
