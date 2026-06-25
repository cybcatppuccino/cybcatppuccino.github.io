# Gardner MiniChess Lab — v5

A browser-native Gardner 5×5 chess board, study-tree explorer, branching move record, position editor, and local classical chess engine. Everything runs locally in the browser. No server-side engine, NNUE network, or neural model is required.

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

## v5 highlights

### False-mate correction and verified mate output

v4 could occasionally display `...Bc5 #1` after `1.b4 cxb4 2.Rxb4`, even though `Bc5` is not check: White's pawn on e3 blocks the bishop diagonal. The fault was in the selective search, not in the board rules. At a non-PV node, pruning could skip every legal move, leave the internal score at the `-INF` sentinel, and turn it into `+INF` when Negamax returned to the parent.

v5 fixes the cause and adds independent protection around mate reporting:

- the first legal move at every searched node is never removed by late-move, futility, or SEE pruning;
- excluded singular-search nodes no longer convert “no remaining candidate” into a fabricated mate;
- mate values are published only when the reported PV can be legally replayed to checkmate at the encoded distance;
- stale v3/v4 cache entries are rejected by the new `Orion JS 5.0` cache namespace;
- cached mate lines are revalidated before reuse;
- timed-out recursion restores the exact root position before result validation.

A dedicated regression test covers the reported line and confirms that `Bc5` is legal, is not check, and has many legal White replies.

### Low-material DTM proof search

For positions with at most six pieces, v5 can run a separate bounded exact mate proof after the ordinary Alpha-Beta slice:

- iterative deepening over mate distance;
- attacker nodes choose the shortest proven continuation;
- defender nodes choose the longest delaying continuation;
- a single escaping defender reply refutes the proof at that bound;
- repetitions, fifty-move draws, and insufficient material are treated conservatively as non-mates;
- successful proofs are cached as exact local DTM entries and shown as `DTM N ply`.

This is an on-demand proof cache, not a complete Gardner tablebase. If the proof budget expires, Orion keeps the ordinary search evaluation and does not claim a solved mate.

### Repetition and cycle handling

- Historical root occurrences are counted through an O(1) map rather than rescanning the game history at every node.
- The actual root still requires a formal third occurrence for a draw.
- Inside a hypothetical search line, a second occurrence is treated as a repeatable cycle, avoiding repeated expansion of reversible loops.
- Repetition is checked before transposition-table cutoffs.
- The halfmove clock remains part of the TT context, while the repetition identity remains board-and-side only.

### Strong advantages and disadvantages

- Mate-distance scores still prefer the shortest mate for the winning side and the longest defence for the losing side.
- Aspiration windows expand with score magnitude, reducing repeated fail-high/fail-low work in clearly winning or losing positions.
- A bounded check-evasion extension improves forcing defence without allowing unbounded extension chains.
- Aggressive null-move/LMR/futility pruning remains disabled or reduced in sparse, promotion-sensitive, and likely-zugzwang positions.

### Evaluation display

The heuristic White/Draw/Black percentages have been removed. The panel now shows only:

- a White-relative centipawn or verified mate score;
- a non-probabilistic evaluation scale;
- depth, nodes, NPS, hash usage, and cache state;
- up to three principal variations;
- `Verified mate` or exact local `DTM N ply` labels when applicable.

## Existing v4 interaction features retained

- Pause and Continue for continuous analysis.
- Structured persistent analysis cache and resumed line searching.
- Click any engine recommendation to play its first move.
- Local two-player, Player vs AI, and AI vs AI modes.
- White/Black side choice and ten finite-search difficulty levels.
- Separate continuous-analysis and finite-play Workers.
- Optional research-book arrows and interactive PGN neighbourhood tree.
- Position editor, FEN import/export, promotion choice, undo/redo, board flip, and multiple piece styles.

## Orion JS 5.0 search architecture

- Negamax Alpha-Beta
- iterative deepening
- Principal Variation Search
- aspiration windows
- quiescence with check evasions, captures, promotions, and bounded quiet checks
- two-key incremental Zobrist hashing
- two-way typed-array transposition table
- static-evaluation cache
- TT move, MVV-LVA/SEE, capture history, killers, quiet history, and countermoves
- mate-distance pruning
- conservative null-move pruning with verification
- Late Move Reductions
- futility, reverse-futility, razoring, delta, SEE, and late-move pruning
- ProbCut
- bounded singular, check, check-evasion, recapture, promotion, and passed-pawn extensions
- MultiPV root exclusion and prior-iteration ordering
- verified-mate PV replay
- bounded low-material DTM proof search

The Gardner evaluator includes material, piece-square tables, center control, mobility, king-zone pressure, pins, loose and multiply attacked pieces, queen-to-king distance, open files, pawn chains, isolated/doubled/passed/protected passed pawns, promotion distance, blockades, space, tempo, and endgame mating pressure.

## Benchmarks

### Reported-line regression

The position after `1.b4 cxb4 2.Rxb4` is stored in `tests/v5-engine-tests.mjs`. Repeated searches at several time limits return ordinary centipawn evaluations and no mate claim for `Bc5`.

### Oracle-position benchmark

`tools/book-benchmark.mjs` samples continuations from the supplied oracle PGNs while withholding book moves from the live engine. The current fixed-depth report is in `data/book-benchmark.json`.

This is a regression/calibration sample, not proof that every recorded PGN continuation is uniquely best. The live engine does not consume the PGN book unless the user independently enables visual **Show book** arrows.

### Throughput

A same-runtime initial-position test at 950 ms and MultiPV 3 completed depth 8 with a v5 warm-run median near 136k NPS in the build environment. Throughput is hardware/JIT dependent and is not Elo. See `data/performance-benchmark.json`.

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

The supplied research archive weakly solves the Gardner starting position, but Orion does not inject the complete oracle into live search. Centipawn values remain heuristic. A mate label is now reserved for a legally replayed checkmate PV; a `DTM` label additionally means the low-material proof search established the bound within its local search horizon. There is no complete general Gardner endgame tablebase in this build.

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
node tests/v5-engine-tests.mjs
```

The suites cover rules/FEN/SAN/PGN parsing, engine/UI move-generation parity, promotion, SEE, terminal positions, legal PV replay, incremental hash restoration, repetition counting, halfmove-aware TT identity, quiescence in check, endgame material handling, continuous Worker updates, pause/resume, persistent cache serialization, finite play searches, AI-vs-AI legal play, the reported false-mate position, exact low-material DTM replay, and removal of W/D/L output.

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
js/engine/analysis-cache.js      Versioned persistent result cache
js/engine/difficulty.js          Levels 1–10
js/ui/                           Board, move list, analysis and study views
data/pgn/                        Original PGN archive
data/reference/                  Original reference PDF
data/book-benchmark.json         Oracle-position calibration report
data/performance-benchmark.json  Local throughput report
docs/ENGINE.md                   Engine architecture
docs/V5-AUDIT.md                 v5 bug analysis and validation
docs/V4-AUDIT.md                 Historical v4 audit
docs/MCTS-REVIEW.md              Review of the supplied external project
tools/                           Re-runnable benchmark tools
tests/                           Browser-independent regression tests
```
