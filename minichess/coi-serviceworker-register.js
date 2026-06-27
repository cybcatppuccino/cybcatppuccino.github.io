/* v15.1: legacy COI helper cleanup.
 *
 * v14.3-v15 tried to create cross-origin isolation with a service worker.  That
 * made the optional Fairy-Stockfish path harder to reason about: stale service
 * workers and sessionStorage reload guards could keep the UI in a permanent
 * fallback/preparing state.  v15.1 returns to the simpler v14.1 model: run the
 * app through serve.sh/serve.bat so the HTTP server sends COOP/COEP directly,
 * and let the Fairy provider attempt startup normally.  This tiny helper only
 * unregisters old COI service workers and clears their reload flags.
 */
(() => {
  const VERSION = 'v15.1';
  const CLEANUP_RELOAD_KEY = 'gardner-coi-cleanup-reloaded-v15_1';

  function publish(status, detail = '') {
    const payload = { version: VERSION, status, detail, isolated: Boolean(window.crossOriginIsolated), at: Date.now() };
    window.__gardnerCoiStatus = payload;
    try { window.dispatchEvent(new CustomEvent('gardner-coi-status', { detail: payload })); } catch {}
  }

  function clearLegacyFlags() {
    try {
      for (let i = sessionStorage.length - 1; i >= 0; i -= 1) {
        const key = sessionStorage.key(i);
        if (key && (key.startsWith('gardner-coi-reload-attempts-') || key === 'gardner-coi-status')) {
          sessionStorage.removeItem(key);
        }
      }
    } catch {}
  }

  async function cleanup() {
    clearLegacyFlags();
    if (!('serviceWorker' in navigator) || !/^https?:$/.test(location.protocol)) {
      publish('cleanup-complete', 'No service-worker cleanup needed.');
      return;
    }
    try {
      const regs = await navigator.serviceWorker.getRegistrations();
      const coiRegs = regs.filter(reg => {
        const urls = [reg.active?.scriptURL, reg.installing?.scriptURL, reg.waiting?.scriptURL].filter(Boolean).map(String);
        return urls.some(url => url.includes('coi-serviceworker.js'));
      });
      await Promise.all(coiRegs.map(reg => reg.unregister().catch(() => false)));
      const controllerUrl = String(navigator.serviceWorker.controller?.scriptURL || '');
      if (controllerUrl.includes('coi-serviceworker.js') && !sessionStorage.getItem(CLEANUP_RELOAD_KEY)) {
        sessionStorage.setItem(CLEANUP_RELOAD_KEY, '1');
        publish('cleanup-reload', 'Removed old COI service worker; reloading once so server headers control Fairy-Stockfish.');
        setTimeout(() => location.reload(), 80);
        return;
      }
      publish('cleanup-complete', coiRegs.length ? 'Removed old COI service worker registrations.' : 'No old COI service worker registrations found.');
    } catch (error) {
      publish('cleanup-failed', error?.message || String(error));
    }
  }

  cleanup();
})();
