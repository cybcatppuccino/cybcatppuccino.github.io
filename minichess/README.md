# Gardner MiniChess Lab — v4

A browser-native Gardner 5×5 chess board, research-tree explorer, branching move record, position editor, and local classical chess engine. Everything runs locally in the browser. No server-side engine, NNUE network, or neural model is required.

## Run locally

The PGN archive and ES-module Web Workers must be loaded over HTTP. Open the project through a local server rather than double-clicking `index.html`.

### Windows

Double-click `serve.bat`, then open:

```text
http://localhost:8000
```

### macOS / Linux

```bash
chmod +x serve.sh
./serve.sh
```

Or run:

```bash
python -m http.server 8000
```

## v4 highlights

### Persistent, resumable analysis

- The analysis panel now has **Pause** and **Continue** controls.
- Every stable MultiPV result is stored in a structured position cache.
- Up to 96 recent position results are persisted in browser `localStorage`; stopping Analysis or reloading the page does not erase them.
- The analysis Worker keeps an additional in-memory position cache and retains its transposition table.
- When the current position has already been searched, the saved result appears immediately and iterative deepening continues from the following depth.
- Up to ten plies of each principal variation receive legal continuation seeds. Choosing a searched line therefore preserves useful work for the next position instead of always starting at depth one.
- Cached lines remain visible when Analysis is off.

### Play directly from engine recommendations

The three analysis rows are buttons. Clicking one plays its first legal move, creates or follows the corresponding branch in the game tree, and immediately restores any cached continuation for the resulting position.

### Three play modes

Modes can be changed at any time:

1. **Local two-player**
2. **Player vs AI** — choose White or Black
3. **AI vs AI**

The play engine is separate from continuous Analysis. It searches only when an AI-controlled side is to move. Analysis, when enabled, remains a continuous independent background process.

Ten difficulty levels control finite move-search time, depth ceiling, MultiPV breadth, and deliberate near-best-move variability. Level 10 always chooses the highest-ranked completed line.

### Orion JS 4.0

v4 keeps the v3 classical Alpha-Beta design and adds:

- direct-mapped typed-array static-evaluation cache;
- capture-history ordering in addition to TT, MVV-LVA/SEE, killers, history, and countermoves;
- a larger persistent Worker transposition table;
- halfmove-clock-aware transposition identity, preventing unsafe score reuse near the fifty-move horizon;
- resumable root ordering and PV seeds from stored analysis;
- cached finite-search reuse for Player-vs-AI and AI-vs-AI modes;
- 96-entry analysis and play-position caches;
- separate Workers for continuous analysis and finite AI moves, keeping the board responsive.

On the same runtime and initial position, a five-run 950 ms MultiPV-3 comparison recorded a warm-run median of roughly **134k NPS for v4 versus 112k NPS for v3** (about 1.19×). Absolute speed varies by browser and hardware. The main v4 gain is not only raw throughput: revisited positions can start with an already completed depth and PV rather than discarding prior work. See `data/performance-benchmark.json`.

## Search architecture

- Negamax Alpha-Beta
- iterative deepening
- Principal Variation Search
- aspiration windows
- quiescence with check evasions, promotions, captures, and limited quiet checks
- two-key incremental Zobrist hashing
- two-way typed-array transposition table
- static-evaluation cache
- TT move, MVV-LVA/SEE, capture history, killer moves, history heuristic, and countermoves
- mate-distance pruning
- conservative null-move pruning with verification
- Late Move Reductions
- futility, reverse-futility, razoring, delta, SEE, and late-move pruning
- ProbCut
- bounded singular/check/recapture/promotion/passed-pawn extensions
- MultiPV root exclusion and prior-iteration ordering

The Gardner evaluator includes material, piece-square tables, center control, mobility, king-zone pressure, pins, loose and multiply attacked pieces, queen-to-king distance, open files, pawn chains, isolated/doubled/passed/protected passed pawns, promotion distance, blockades, space, tempo, and endgame mating pressure.

## Oracle-position benchmark

`tools/book-benchmark.mjs` samples main continuations from the supplied oracle PGNs and searches without providing book moves to the engine. The generated report is in `data/book-benchmark.json`.

This is a regression/calibration test, not proof that every recorded PGN continuation is the unique objective best move. The live engine does not use the PGN book unless the user independently turns on the visual **Show book** arrows.

## Review of the supplied MCTS/PPO project

The supplied `mcts-chess-master.zip` was inspected. Its Gardner implementation is not directly integrated because it uses a materially different game definition: incremental castling/en-passant/double-pawn support, king-capture termination, and automatic promotion behavior. No clear repository licence was present in the supplied archive. Its Python/Ray/PyTorch model is also unsuitable for a dependency-free browser JavaScript build.

Only general ideas were retained: explicit legal-action masking, bounded finite searches for difficulty levels, and controlled exploration among near-best moves. No source code, model weights, or sample games from that archive are bundled or used by Orion. See `docs/MCTS-REVIEW.md`.

## Controls

- Click a piece to show legal destinations; click it again to clear markers.
- Drag to a legal square; illegal drops return to the origin.
- Click an analysis recommendation to play it.
- Pause/Continue affects continuous Analysis only, not the board.
- `A`: toggle analysis.
- `B`: toggle archive-book arrows.
- `F`: flip the board.
- Left/Right arrows: undo/redo.

## Important interpretation

The Gardner starting position is weakly solved as a draw with correct play. Live W/D/L percentages are heuristic mappings of the local engine score, not tablebase probabilities. A searched mate score is stronger evidence than a centipawn estimate, but this build does not include a complete Gardner oracle or exact general endgame tablebase.

## Source archive

The supplied research files remain unchanged in:

```text
data/pgn/
data/reference/oracle.pdf
```

Their hashes are listed in `data/SOURCE_INTEGRITY.sha256`.

## Tests

```bash
node tests/core-tests.mjs
node tests/engine-tests.mjs
node tests/engine-regression-tests.mjs
node tests/client-regression-tests.mjs
node tests/worker-smoke-tests.mjs
node tests/analysis-cache-tests.mjs
node tests/play-worker-tests.mjs
node tests/pause-resume-worker-tests.mjs
node tests/ai-vs-ai-smoke-tests.mjs
```

The suites cover rules/FEN/SAN/PGN parsing, engine/UI move-generation parity, promotions, SEE, terminal positions, legal PV replay, incremental hash round-trips, exact repetition counting, halfmove-aware TT identity, quiescence in check, endgame material handling, first-click startup, streamed iterative deepening, pause/resume, persistent cache serialization, finite play searches, cached finite-search reuse, and an AI-vs-AI legal-move sequence.

## Project layout

```text
index.html                       Main interface
app.js                           Application coordinator and mode state
styles.css                       Visual system
assets/                          Branding assets
js/core/                         Rules, FEN, SAN, game tree, PGN parser
js/engine/engine.js              Classical 25-square search engine
js/engine/worker.js              Continuous resumable analysis Worker
js/engine/play-worker.js         Finite AI-move Worker
js/engine/analysis-cache.js      Persistent structured result cache
js/engine/difficulty.js          Levels 1–10
js/ui/                           Board, move list, analysis and study views
data/pgn/                        Original PGN archive
data/reference/                  Original reference PDF
data/book-benchmark.json         Oracle-position calibration report
data/performance-benchmark.json  v3/v4 throughput comparison
docs/ENGINE.md                   Engine architecture
docs/V4-AUDIT.md                 v4 correctness and optimization audit
docs/MCTS-REVIEW.md              Review of the supplied external project
tools/book-benchmark.mjs         Re-runnable PGN benchmark
tools/performance-benchmark.mjs  Re-runnable local throughput check
tests/                           Browser-independent regression tests
```
