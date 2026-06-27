/* v15.1 legacy file.
 * This service worker is no longer registered.  It exists only so browsers with
 * cached v14.3/v15 registration URLs can fetch a harmless replacement before the
 * cleanup script unregisters them.  Use serve.sh/serve.bat for COOP/COEP.
 */
self.addEventListener('install', event => event.waitUntil(self.skipWaiting()));
self.addEventListener('activate', event => event.waitUntil((async () => {
  try { await self.registration.unregister(); } catch {}
})()));
self.addEventListener('fetch', event => {
  event.respondWith(fetch(event.request));
});
