(() => {
  "use strict";
  const candidates = [
    "https://unpkg.com/maplibre-gl@5.24.0/dist/maplibre-gl.js",
    "https://cdn.jsdelivr.net/npm/maplibre-gl@5.24.0/dist/maplibre-gl.js"
  ];
  const cfg = window.GBA_RAIL_CONFIG;
  const manifestPath = cfg?.lazyDataManifest;
  if (manifestPath && !window.GBA_RAIL_MANIFEST_PROMISE) {
    const manifestUrl = `${manifestPath}?v=${encodeURIComponent(cfg.version || "manifest")}`;
    window.GBA_RAIL_MANIFEST_PROMISE = fetch(manifestUrl, { cache:"force-cache", priority:"high" })
      .then(response => response.ok ? response.json() : null)
      .catch(() => null);
  }
  function fail() {
    const el = document.createElement("div");
    el.style.cssText = "position:fixed;z-index:9999;inset:20px auto auto 50%;transform:translateX(-50%);max-width:620px;padding:14px 18px;border-radius:14px;background:#fff;color:#263742;box-shadow:0 14px 40px rgba(20,40,52,.25);font:14px/1.6 system-ui,sans-serif";
    el.innerHTML = "<strong>地图引擎未能加载</strong><br>轨道数据可从本地包或永久缓存读取，但 MapLibre 引擎与在线底图仍需能访问 unpkg、jsDelivr 或 OpenFreeMap。";
    document.body.appendChild(el);
  }
  function loadApp() {
    const app = document.createElement("script");
    app.src = `assets/app.js?v=${encodeURIComponent(cfg?.version || "app")}`;
    app.onerror = fail;
    document.body.appendChild(app);
  }
  function tryCandidate(index) {
    if (index >= candidates.length) return fail();
    const script = document.createElement("script");
    script.src = candidates[index];
    script.crossOrigin = "anonymous";
    script.onload = () => window.maplibregl ? loadApp() : tryCandidate(index + 1);
    script.onerror = () => tryCandidate(index + 1);
    document.body.appendChild(script);
  }
  tryCandidate(0);
})();
