# Gardner MiniChess Lab v15.1 patch notes

v15.1 is intended to be applied over the latest complete v15 package.  It keeps the v15 cache capacity unchanged and intentionally simplifies the Fairy-Stockfish startup path by reverting to the stable v14.1 model: the app should be served with real COOP/COEP headers by `serve.sh` / `serve.bat`, and the Fairy provider should be allowed to attempt startup directly.

## Why Stockfish kept falling back

The v14.3-v15 COI service-worker helper tried to manufacture cross-origin isolation inside the app.  In practice this introduced a second moving part: stale service-worker registrations and reload guards could leave the page in a state where the UI believed Fairy was still “preparing COI” and therefore preemptively sent requests to Orion JS instead of even attempting Fairy-Stockfish.

v15.1 removes that preemptive UI fallback.  The UI now passes `fairy-stockfish` to the worker whenever Fairy is selected.  The provider itself tries to boot the wasm engine and falls back to Orion only if the browser really blocks the pthread wasm or if Fairy returns an illegal PV.

## How to run Fairy-Stockfish

Use the included COOP/COEP server:

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

Do not use `file://`.  Ordinary static servers may also fail unless they send:

```text
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
Cross-Origin-Resource-Policy: same-origin
```

## Legacy COI cleanup

The files `coi-serviceworker-register.js` and `coi-serviceworker.js` remain in the patch only as cleanup shims.  They unregister the older v14.3/v15 COI service workers and clear stale reload flags.  They no longer try to inject COOP/COEP or force reload loops.

## Cache and performance

v15.1 uses:

```text
Orion JS 15.1
gardner-analysis-cache-v15_1
```

It migrates compatible v14, v14.1, v14.2, v14.3 and v15 Orion cache entries into v15.1.  Cache capacity is unchanged from v15:

```text
persistent analysis cache: 576 entries
eval cache: 524288
structural profile cache: 24576
analysis worker cache: 216
play worker cache: 576
```

The conservative performance change in v15.1 is UI-side: repeated PV-to-SAN formatting is memoized for the current analysis root, reducing DOM/render overhead without touching search rules, evaluation, move legality, or engine strength.

## Changed areas

- Restored direct Fairy kernel dispatch from the UI.
- Removed UI-side `SharedArrayBuffer` gating that caused permanent Orion fallback.
- Added legacy COI service-worker cleanup.
- Updated version/cache identity to v15.1.
- Preserved all v15 cache sizes.
- Added tests for simplified Stockfish startup, cache migration, and unchanged cache capacity.
