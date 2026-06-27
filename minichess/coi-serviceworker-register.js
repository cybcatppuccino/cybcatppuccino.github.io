/* v15: robust cross-origin isolation bootstrap for the optional
 * Fairy-Stockfish pthread wasm kernel.
 *
 * The supplied Fairy-Stockfish build requires SharedArrayBuffer, which in turn
 * requires a crossOriginIsolated page.  Static hosts frequently lack COOP/COEP
 * headers, so this helper registers a same-origin service worker that injects
 * them and then performs a small bounded reload sequence.  Orion JS never
 * depends on this helper.
 */
(() => {
  const VERSION = 'v15';
  const SW_URL = `./coi-serviceworker.js?${VERSION}`;
  const SCOPE = './';
  const RELOAD_KEY = `gardner-coi-reload-attempts-${VERSION}`;
  const STATUS_KEY = 'gardner-coi-status';
  const MAX_RELOADS = 3;

  function publish(status, detail = '') {
    const payload = { version: VERSION, status, detail, isolated: Boolean(window.crossOriginIsolated), at: Date.now() };
    window.__gardnerCoiStatus = payload;
    try { sessionStorage.setItem(STATUS_KEY, JSON.stringify(payload)); } catch {}
    try { window.dispatchEvent(new CustomEvent('gardner-coi-status', { detail: payload })); } catch {}
  }

  function reloadAttempts() {
    try { return Number(sessionStorage.getItem(RELOAD_KEY) || 0); } catch { return 0; }
  }

  function setReloadAttempts(value) {
    try { sessionStorage.setItem(RELOAD_KEY, String(Math.max(0, Number(value || 0)))); } catch {}
  }

  function reloadSoon(reason) {
    const attempts = reloadAttempts();
    if (attempts >= MAX_RELOADS) {
      publish('reload-limit', `${reason}; reload limit reached. Use ./serve.sh or serve.bat so COOP/COEP headers are sent by the HTTP server.`);
      return;
    }
    setReloadAttempts(attempts + 1);
    publish('reloading', `${reason}; reloading to enter cross-origin isolated mode (${attempts + 1}/${MAX_RELOADS}).`);
    setTimeout(() => {
      try { location.reload(); } catch {}
    }, 80);
  }

  window.__gardnerRequestCoiReload = reason => {
    if (window.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined') return false;
    reloadSoon(reason || 'Fairy-Stockfish requires cross-origin isolated mode');
    return true;
  };

  window.__gardnerCoiStatus = window.__gardnerCoiStatus || {
    version: VERSION,
    status: 'starting',
    detail: 'COI helper script loaded.',
    isolated: Boolean(window.crossOriginIsolated),
    at: Date.now()
  };


  if (window.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined') {
    setReloadAttempts(0);
    publish('isolated', 'SharedArrayBuffer is available.');
    return;
  }

  if (!('serviceWorker' in navigator) || !/^https?:$/.test(location.protocol)) {
    publish('unsupported', 'Service workers require same-origin HTTP/HTTPS. Open the app through ./serve.sh or serve.bat, not file://.');
    return;
  }

  publish('installing', 'Preparing COOP/COEP helper for Fairy-Stockfish.');

  navigator.serviceWorker.addEventListener('controllerchange', () => {
    if (!window.crossOriginIsolated) reloadSoon('COI service worker became the page controller');
  });

  (async () => {
    try {
      // Remove older unversioned/v14 helpers for this scope.  Keeping them can
      // leave a sessionStorage one-shot flag set while the page is still not
      // isolated, which made Fairy-Stockfish permanently fall back in v14.3.
      const regs = await navigator.serviceWorker.getRegistrations();
      await Promise.all(regs
        .filter(reg => reg.scope === new URL(SCOPE, location.href).href && !String(reg.active?.scriptURL || '').includes(`coi-serviceworker.js?${VERSION}`))
        .map(reg => reg.unregister().catch(() => false)));

      const registration = await navigator.serviceWorker.register(SW_URL, {
        scope: SCOPE,
        updateViaCache: 'none'
      });
      try { await registration.update(); } catch {}
      await navigator.serviceWorker.ready;

      if (window.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined') {
        setReloadAttempts(0);
        publish('isolated', 'SharedArrayBuffer is available.');
        return;
      }

      if (!navigator.serviceWorker.controller) {
        reloadSoon('COI service worker installed but does not control this page yet');
        return;
      }

      // A controlled but non-isolated page means the main document was not yet
      // loaded with COOP/COEP headers.  Reload again; the service worker will
      // add headers to the navigation response.
      setTimeout(() => {
        if (!window.crossOriginIsolated) reloadSoon('COI service worker is active but the current document is not isolated');
      }, 120);
    } catch (error) {
      publish('failed', error?.message || String(error));
      console.warn('COI service worker registration failed; Fairy-Stockfish may fall back to Orion JS.', error);
    }
  })();
})();
