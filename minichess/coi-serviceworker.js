/* v15 cross-origin isolation service worker.
 * Adds COOP/COEP/CORP to same-origin responses so Fairy-Stockfish's pthread
 * wasm can use SharedArrayBuffer under local/static hosting.
 */
const COI_VERSION = 'v15';
const COOP = 'Cross-Origin-Opener-Policy';
const COEP = 'Cross-Origin-Embedder-Policy';
const CORP = 'Cross-Origin-Resource-Policy';

self.addEventListener('install', event => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', event => {
  event.waitUntil((async () => {
    try {
      if (self.registration.navigationPreload) await self.registration.navigationPreload.disable();
    } catch {}
    await self.clients.claim();
  })());
});

function isolatedResponse(response) {
  if (!response || response.type === 'opaque' || response.type === 'opaqueredirect') return response;
  const headers = new Headers(response.headers);
  headers.set(COOP, 'same-origin');
  headers.set(COEP, 'require-corp');
  headers.set(CORP, 'same-origin');
  headers.set('X-Gardner-COI-ServiceWorker', COI_VERSION);
  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers
  });
}

self.addEventListener('fetch', event => {
  const request = event.request;
  if (request.cache === 'only-if-cached' && request.mode !== 'same-origin') return;
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;
  event.respondWith((async () => {
    try {
      const response = await fetch(request, { cache: request.mode === 'navigate' ? 'no-store' : undefined });
      return isolatedResponse(response);
    } catch (error) {
      if (request.mode === 'navigate') {
        return isolatedResponse(new Response('<!doctype html><title>Gardner MiniChess</title><p>Unable to load the app while preparing cross-origin isolation. Please reload.</p>', {
          status: 503,
          headers: { 'Content-Type': 'text/html; charset=utf-8' }
        }));
      }
      throw error;
    }
  })());
});

self.addEventListener('message', event => {
  if (event.data?.type === 'coi-status') {
    event.source?.postMessage?.({ type: 'coi-status', version: COI_VERSION });
  }
});
