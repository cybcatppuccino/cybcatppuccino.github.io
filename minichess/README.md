# Gardner MiniChess Lab — v14.1 patch

This patch is intended to be applied over a complete v14 installation. It contains only files changed or added for v14.1. Keep the original archival PGN files and generated tablebase folders in place unless a changed file with the same path is included in this patch.

## Compatibility notes

v14 keeps the v12.2 public notation policy: compact 5×5 FEN and standard A1–E5 SAN/UCI are canonical everywhere in the UI, engine output and generated data. Legacy b2–f6 archive PGN/FEN input is still accepted only through the explicit compatibility paths.

v14.1 uses the engine identity `Orion JS 14.1` and persistent cache key `gardner-analysis-cache-v14_1`. The app removes `gardner-analysis-cache-v12_1`, `gardner-analysis-cache-v12_2`, `gardner-analysis-cache-v13` and `gardner-analysis-cache-v14` localStorage entries on load because closed-position and kernel-selection semantics changed.


## What changed in v14.1

- Fixed the follow-up locked queen/rook deadlock `rq2k/p1p1p/PpPpP/1B1P1/RQ2K b - - 6 8`: unsound pseudo-contact no longer prevents the legal low-progress classifier from compressing material-only edges to `0.00`.
- Kept tactical closed positions separate by adding a concrete heavy-offer breakthrough verifier.  If the side to move can make a defended heavy-piece offer that the opponent can accept and the mover can soundly recapture, hard deadlock compression is blocked.
- Fixed Fairy-Stockfish startup hangs.  Initialization errors without a search token now reject the provider and trigger Orion JS fallback instead of leaving the UI at “Starting the local engine…”.
- Added `tools/serve-coi.py` and updated `serve.sh` / `serve.bat` to serve COOP/COEP headers and `application/wasm`, which are required for this pthread/SharedArrayBuffer Fairy-Stockfish wasm build.
- Added `tools/engine-kernel-benchmark.mjs` and `docs/ENGINE-KERNEL-BENCHMARK-v14.1.md` for repeatable Orion-vs-Fairy comparisons.

### Running Fairy-Stockfish locally

Use the included server scripts rather than `python -m http.server` or `file://`:

```sh
./serve.sh
```

On Windows, run:

```bat
serve.bat
```

Then open `http://127.0.0.1:8000`.  If you deploy elsewhere, configure equivalent response headers:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

## What changed in v14

- Added legal-move verified compression for queenful/rook deadlocks: if the pawn wall is locked and neither side has captures, checks, pawn breaks, promotions, king entries or enabling moves, material-only edges are treated as practical draws.
- Preserved v13 breakthrough behavior by refusing deadlock compression when contact attacks or irreversible resources exist.
- Added Fairy-Stockfish wasm 1.1.11 as an optional UCI provider, not a replacement for Orion JS.  External PVs are validated with Orion's legal move generator before analysis or AI play can use them; invalid/unavailable wasm output falls back to Orion JS 14.
- Added a `Kernel` selector in the AI/analysis controls.  `Orion JS 14` remains the default; `Fairy-Stockfish` can be selected for live analysis and AI play.
- Added v14 regression tests for the supplied `rq2k/p1p1p/PpPpP/1B1P1/R1Q1K w - - 5 8` deadlock and external-engine validation.

No original archive PGN or tablebase/database source file is changed by this patch.  The Fairy-Stockfish wasm files are added under `vendor/fairy-stockfish/`; keep their package metadata and checksums with the distribution.
