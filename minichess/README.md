# Gardner MiniChess Lab v15 patch notes

v15 is intended to be applied over the latest complete v14.3 package.  It keeps the v14.3 cache capacity unchanged while fixing the Fairy-Stockfish startup path and reducing avoidable UI/worker overhead.

## Fairy-Stockfish startup

Fairy-Stockfish remains an optional backup kernel.  The supplied wasm package is a pthread build, so the browser must expose `SharedArrayBuffer`.  v15 makes the COI helper versioned and more aggressive about replacing old v14 service workers, because a stale one-shot reload flag could leave the app permanently in Orion fallback mode.

Recommended launch path:

```sh
./serve.sh
```

Windows:

```bat
serve.bat
```

Then open:

```text
http://127.0.0.1:8000
```

Do not use `file://`.  The included `tools/serve-coi.py` sends COOP/COEP/CORP headers directly.  If a user runs the app from an ordinary same-origin HTTP server instead, `coi-serviceworker.js?v15` will try to inject the same headers and reload the page up to three times.  Once `crossOriginIsolated` is true, Fairy-Stockfish runs directly; otherwise Orion JS 15 remains available as safe fallback.

## Cache and performance

v15 uses:

```text
Orion JS 15
gardner-analysis-cache-v15
```

It migrates compatible v14, v14.1, v14.2 and v14.3 Orion cache entries into the v15 cache.  The persistent cache size remains 576 entries.  The eval cache, structural profile cache, analysis worker cache and play worker cache remain at the v14.3 sizes.

Efficiency changes are intentionally conservative:

- coalesced UI rendering for streamed analysis results, while still writing every cache update;
- small worker-side FEN→history-key cache to avoid reparsing the same recent-history FENs across analysis/play requests;
- versioned COI service worker registration with `updateViaCache: 'none'`, reducing stale-service-worker fallback loops.

These changes do not alter evaluation weights, legal move generation, search semantics or playing style.

## Changed areas

- `coi-serviceworker-register.js`
- `coi-serviceworker.js`
- `tools/serve-coi.py`
- `serve.sh` / `serve.bat`
- `app.js`
- `js/engine/analysis-cache.js`
- `js/engine/engine.js`
- `js/engine/external-engine.js`
- `js/engine/worker.js`
- `js/engine/play-worker.js`
- Fairy worker startup message handling
- v15 regression tests
