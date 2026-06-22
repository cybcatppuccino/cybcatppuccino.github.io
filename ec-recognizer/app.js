const EC_ATLAS_VERSION = 'v40';
const DATA_ROOT = new URL('./data/', import.meta.url);
const API_PROGRESS = { active: 0, text: '' };
const JSON_STREAM_PROGRESS_THRESHOLD = 262144;
const JSON_PROGRESS_THROTTLE_MS = 120;
const INTEGRAL_SEARCH_TIME_MULTIPLIER = 2.25;
const JSON_DECODER = new TextDecoder();
let EC_CORE_PROMISE = null;
const API_CACHE = {
  meta: null,
  top: null,
  searchManifestPromise: null,
  searchManifest: null,
  rowsById: new Map(),
  curveShardPromises: new Map(),
  loadedCurveShards: new Set(),
  exactPromises: new Map(),
  exactMaps: new Map(),
  conductorExactPromises: new Map(),
  conductorExactMaps: new Map(),
  conductorPromises: new Map(),
  conductorRows: new Map(),
  hashPromises: new Map(),
  tauPromise: null,
  tauRows: null,
  tauBuckets: null,
  stPromise: null,
  stGroups: null,
  cmDiscPromise: null,
  cmDiscIndex: null,
  jsonPromises: new Map(),
  curveDetailCache: new Map(),
  curveDetailPromises: new Map(),
  cisoCache: new Map(),
  pointSearches: new Map(),
  searchResultCache: new Map(),
  squareSieveCache: new Map(),
  tilesLoaded: 0,
  tilesRequested: 0,
};

function setAtlasProgress(text, frac = null, hide = false) {
  const bar = document.getElementById('atlas-progress');
  const fill = document.getElementById('atlas-progress-fill');
  const label = document.getElementById('atlas-progress-text');
  if (!bar || !fill || !label) return;
  if (hide) {
    bar.classList.add('quiet');
    fill.style.width = '100%';
    label.textContent = '';
    return;
  }
  bar.classList.remove('quiet');
  if (text) label.textContent = text;
  if (typeof frac === 'number' && Number.isFinite(frac)) fill.style.width = `${Math.max(1, Math.min(100, frac * 100)).toFixed(1)}%`;
}

async function fetchTextWithProgress(path, label) {
  const url = new URL(path, DATA_ROOT);
  const res = await fetch(url, { cache: 'force-cache' });
  if (!res.ok) throw new Error(`${label || path} failed: HTTP ${res.status}`);
  const len = Number(res.headers.get('content-length') || 0);

  // v32/v33: streaming progress is useful for multi-MB indexes, but for the many
  // small JSON tiles it adds extra chunk bookkeeping, a Uint8Array copy, and
  // frequent DOM writes.  Keep progress streaming only where it can pay for
  // itself; let the browser's native text path handle small files.
  if (!res.body || !len || len < JSON_STREAM_PROGRESS_THRESHOLD) {
    if (label) setAtlasProgress(label, 0.35);
    return await res.text();
  }

  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  let lastProgressAt = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const now = performance.now();
    if (now - lastProgressAt >= JSON_PROGRESS_THROTTLE_MS || received >= len) {
      lastProgressAt = now;
      setAtlasProgress(`${label || 'Loading data'} ${(received / 1048576).toFixed(1)} MB`, Math.min(0.98, received / len));
    }
  }
  const all = new Uint8Array(received);
  let pos = 0;
  for (const ch of chunks) { all.set(ch, pos); pos += ch.length; }
  return JSON_DECODER.decode(all);
}

async function fetchJSONData(path, label) {
  const key = String(path);
  if (!API_CACHE.jsonPromises.has(key)) {
    const promise = fetchTextWithProgress(path, label)
      .then(text => JSON.parse(text))
      .catch(err => { API_CACHE.jsonPromises.delete(key); throw err; });
    API_CACHE.jsonPromises.set(key, promise);
  }
  return API_CACHE.jsonPromises.get(key);
}
function idleYield(timeout = 16) {
  return new Promise(resolve => {
    if (typeof requestIdleCallback === 'function') requestIdleCallback(() => resolve(), { timeout });
    else setTimeout(resolve, 0);
  });
}

function normalizeSearchText(text) {
  return String(text || '').trim().replace(/[−—–]/g, '-');
}
function gcdBI(a, b) { a = a < 0n ? -a : a; b = b < 0n ? -b : b; while (b) { const t = a % b; a = b; b = t; } return a; }
function absBI(n) { return n < 0n ? -n : n; }
function modBI(a, m) { a = BigInt(a); m = BigInt(m); let r = a % m; if (r < 0n) r += m < 0n ? -m : m; return r; }
function parseRationalKey(text) {
  const raw = normalizeSearchText(text).replace(/\s+/g, '');
  const m = raw.match(/^([+-]?\d+)(?:\/([+-]?\d+))?$/);
  if (!m) return null;
  let n = BigInt(m[1]);
  let d = m[2] == null ? 1n : BigInt(m[2]);
  if (d === 0n) return null;
  if (d < 0n) { n = -n; d = -d; }
  const g = gcdBI(n, d);
  n /= g; d /= g;
  return d === 1n ? String(n) : `${n}/${d}`;
}
function rowToObject(columns, arr) {
  const o = {};
  for (let i = 0; i < columns.length; i++) o[columns[i]] = arr[i];
  o.id = Number(o.id);
  o.N = Number(o.N);
  o.rank = Number(o.rank || 0);
  o.cm = Number(o.cm) ? 1 : 0;
  return o;
}
function prepareSearchRow(row) {
  if (!row || row.__searchPrepared) return row;
  row.__labelKey = String(row.label || '').toLowerCase();
  row.__cremonaKey = String(row.cremona || '').toLowerCase();
  row.__isoKey = String(row.iso || '').toLowerCase();
  row.__searchPrepared = true;
  return row;
}
function rowMatchesNeedle(row, needle, compactNeedle, conductorOnly = false, conductor = null) {
  prepareSearchRow(row);
  const exact = row.__labelKey === compactNeedle || row.__cremonaKey === compactNeedle || row.__isoKey === compactNeedle;
  if (exact) return 2;
  if (row.__labelKey.includes(needle) || row.__cremonaKey.includes(needle) || row.__isoKey.includes(needle) || (conductorOnly && String(row.N) === String(conductor))) return 1;
  return 0;
}
function hasFullCurveRow(row) {
  return !!(row && row.tor_order != null && row.st != null && row.pts != null && row.deg != null && row.tau_re != null && row.tau_im != null && row.real_period != null && row.prime_signature != null && row.period_phase != null && row.weierstrass_equation != null);
}
function addMapList(map, key, id) {
  if (!map.has(key)) map.set(key, []);
  map.get(key).push(id);
}
function enrichClientCurve(row) {
  const d = { ...row };
  d.id = Number(d.id);
  d.N = Number(d.N || 0);
  d.rank = Number(d.rank || 0);
  d.cm = Number(d.cm) ? 1 : 0;
  if (d.az != null) d.lon_deg = ((Number(d.az) * 180 / Math.PI) % 360 + 360) % 360;
  if (d.alt != null) d.alt_deg = Number(d.alt) * 180 / Math.PI;
  return d;
}
function v20HashKey(key, n = 64) {
  let h = 0;
  const s = String(key || '');
  for (let i = 0; i < s.length; i++) h = (h * 131 + s.charCodeAt(i)) % n;
  return h;
}
function curveShardForId(id, mod = 512) { return ((Number(id) % mod) + mod) % mod; }
function conductorBucketForN(N) { return Math.max(0, Math.min(9, Math.floor(Number(N || 0) / 1000))); }
function exactSearchShardForKey(key) { return v20HashKey(String(key || '').toLowerCase(), 256); }
function conductorExactShardForN(N) { return v20HashKey(String(N || ''), 64); }
function compactExactKey(q) { return normalizeSearchText(q).replace(/\s+/g, '').toLowerCase(); }
function isCubicLikeQuery(q) {
  const t = normalizeSearchText(q);
  return /[\[\]=^*]/.test(t) || /\b[x-zuvw][a-z]?\b/i.test(t) || /[²³]/.test(t);
}
async function loadEcCore() {
  if (!EC_CORE_PROMISE) EC_CORE_PROMISE = import('./js/ec_core.js');
  return EC_CORE_PROMISE;
}
async function identifyCubicLazy(q) {
  if (!isCubicLikeQuery(q)) return null;
  try {
    const mod = await loadEcCore();
    return mod.identifyCubicJS(q);
  } catch (e) {
    console.warn('cubic recognizer failed to load or parse input', e);
    return null;
  }
}
async function loadSearchManifest() {
  if (API_CACHE.searchManifest) return API_CACHE.searchManifest;
  if (!API_CACHE.searchManifestPromise) {
    API_CACHE.searchManifestPromise = fetchJSONData('search/manifest.json', 'Loading search metadata')
      .then(d => (API_CACHE.searchManifest = d));
  }
  return API_CACHE.searchManifestPromise;
}
async function loadCurveShardById(id) {
  const shard = curveShardForId(id, 512);
  const key = `d:${shard}`;
  if (API_CACHE.loadedCurveShards.has(key)) return key;
  if (!API_CACHE.curveShardPromises.has(key)) {
    const fastName = `detail_shards/d${String(shard).padStart(3, '0')}.json`;
    API_CACHE.curveShardPromises.set(key, (async () => {
      let packed;
      try {
        packed = await fetchJSONData(fastName, `Loading curve ${id}`);
      } catch (err) {
        // v20/v22 fallback: older deployments only have 128 modulo shards.
        const oldShard = curveShardForId(id, 128);
        const oldKey = `s:${oldShard}`;
        if (API_CACHE.loadedCurveShards.has(oldKey)) return oldKey;
        const oldName = `curve_shards/s${String(oldShard).padStart(3, '0')}.json`;
        packed = await fetchJSONData(oldName, `Loading curve ${id}`);
        API_CACHE.loadedCurveShards.add(oldKey);
      }
      const cols = packed.columns || [];
      for (const arr of packed.rows || []) {
        const row = prepareSearchRow(rowToObject(cols, arr));
        row.__detailFull = true;
        API_CACHE.rowsById.set(Number(row.id), row);
      }
      API_CACHE.loadedCurveShards.add(key);
      return key;
    })());
  }
  return API_CACHE.curveShardPromises.get(key);
}
async function loadCurveById(id) {
  id = Number(id);
  const cached = API_CACHE.rowsById.get(id);
  if (hasFullCurveRow(cached) || cached?.__detailFull) return cached;
  await loadCurveShardById(id);
  const row = API_CACHE.rowsById.get(id) || null;
  if (cached && row && row !== cached) Object.assign(cached, row, { __detailFull: true });
  return API_CACHE.rowsById.get(id) || cached || null;
}
async function loadRowsByIds(ids, limit = 20, onRows = null) {
  const out = [];
  const shardIds = new Map();
  const maxRows = Number.isFinite(limit) ? Math.max(0, Number(limit)) : Infinity;
  const emitRows = rows => { if (rows.length && typeof onRows === 'function') onRows(rows.slice(), out.slice()); };
  for (const id0 of ids || []) {
    const id = Number(id0);
    if (API_CACHE.rowsById.has(id)) {
      const row = API_CACHE.rowsById.get(id);
      out.push(row);
      emitRows([row]);
      if (out.length >= maxRows) return out;
      continue;
    }
    const shard = curveShardForId(id);
    if (!shardIds.has(shard)) shardIds.set(shard, []);
    shardIds.get(shard).push(id);
  }
  const shardGroups = [...shardIds.values()];
  const BATCH = 4;
  for (let i = 0; i < shardGroups.length; i += BATCH) {
    const groupBatch = shardGroups.slice(i, i + BATCH);
    await Promise.all(groupBatch.map(ids => loadCurveShardById(ids[0]).catch(err => console.warn('curve shard failed', ids[0], err))));
    const newly = [];
    for (const ids0 of groupBatch) {
      for (const id of ids0) {
        const row = API_CACHE.rowsById.get(id);
        if (row) { out.push(row); newly.push(row); }
        if (out.length >= maxRows) { emitRows(newly); return out; }
      }
    }
    emitRows(newly);
    if (i + BATCH < shardGroups.length) await idleYield(8);
  }
  return Number.isFinite(limit) ? out.slice(0, maxRows) : out;
}
async function loadConductorRowsForN(N, onRows = null) {
  const key = `conductor:${Math.floor(Number(N))}:rows`;
  const cached = API_CACHE.searchResultCache.get(key);
  if (cached?.rows) {
    if (typeof onRows === 'function') onRows(cached.rows.slice(), cached.rows.slice());
    return cached.rows;
  }
  if (cached?.promise) return cached.promise;
  const promise = (async () => {
    const ids = await idsForConductorFast(N).catch(() => []);
    let rows = [];
    if (ids && ids.length) {
      const byIdSeen = new Set();
      rows = await loadRowsByIds(ids, Infinity, (newRows, soFar) => {
        const unique = [];
        for (const row of newRows) {
          if (!row || byIdSeen.has(Number(row.id))) continue;
          byIdSeen.add(Number(row.id));
          unique.push(row);
        }
        if (unique.length && typeof onRows === 'function') onRows(unique.slice().sort(naturalLabelCompare), soFar.slice().sort(naturalLabelCompare));
      });
    }
    if (!rows.length || (ids.length && rows.length < ids.length)) {
      const bucketRows = await loadConductorBucket(conductorBucketForN(N)).catch(() => []);
      const exactRows = bucketRows.filter(r => String(r.N) === String(Math.floor(Number(N))));
      const seen = new Set(rows.map(r => Number(r.id)));
      for (const r of exactRows) if (!seen.has(Number(r.id))) { rows.push(r); seen.add(Number(r.id)); }
    }
    rows = rows.map(prepareSearchRow).sort(naturalLabelCompare);
    API_CACHE.searchResultCache.set(key, { rows, ts:Date.now() });
    return rows;
  })().catch(err => { API_CACHE.searchResultCache.delete(key); throw err; });
  API_CACHE.searchResultCache.set(key, { promise, ts:Date.now() });
  return promise;
}
async function loadExactSearchMap(key) {
  const shard = exactSearchShardForKey(key);
  if (API_CACHE.exactMaps.has(shard)) return API_CACHE.exactMaps.get(shard);
  if (!API_CACHE.exactPromises.has(shard)) {
    API_CACHE.exactPromises.set(shard, fetchJSONData(`search/exact/e${String(shard).padStart(3, '0')}.json`, 'Loading exact search index')
      .then(d => {
        const map = d.map || {};
        API_CACHE.exactMaps.set(shard, map);
        return map;
      })
      .catch(err => {
        console.warn('exact search shard unavailable; falling back to conductor search', err);
        const map = {};
        API_CACHE.exactMaps.set(shard, map);
        return map;
      }));
  }
  return API_CACHE.exactPromises.get(shard);
}
async function idsForExactKey(key) {
  const k = compactExactKey(key);
  if (!k) return [];
  const map = await loadExactSearchMap(k);
  return map[k] || [];
}
async function loadConductorExactMap(N) {
  const shard = conductorExactShardForN(N);
  if (API_CACHE.conductorExactMaps.has(shard)) return API_CACHE.conductorExactMaps.get(shard);
  if (!API_CACHE.conductorExactPromises.has(shard)) {
    API_CACHE.conductorExactPromises.set(shard, fetchJSONData(`search/N/n${String(shard).padStart(2, '0')}.json`, 'Loading conductor index')
      .then(d => {
        const map = d.map || {};
        API_CACHE.conductorExactMaps.set(shard, map);
        return map;
      })
      .catch(err => {
        console.warn('conductor exact shard unavailable; falling back to bucket search', err);
        const map = {};
        API_CACHE.conductorExactMaps.set(shard, map);
        return map;
      }));
  }
  return API_CACHE.conductorExactPromises.get(shard);
}
async function idsForConductorFast(N) {
  if (N == null || Number.isNaN(Number(N))) return [];
  const key = String(Math.floor(Number(N)));
  const map = await loadConductorExactMap(key);
  return map[key] || [];
}

async function loadConductorBucket(bucket) {
  bucket = Math.max(0, Math.min(9, Number(bucket) || 0));
  if (!API_CACHE.conductorPromises.has(bucket)) {
    API_CACHE.conductorPromises.set(bucket, (async () => {
      const packed = await fetchJSONData(`search/conductors/c${String(bucket).padStart(2, '0')}.json`, `Loading conductor search ${bucket}`);
      const cols = packed.columns || [];
      const rows = (packed.rows || []).map(arr => prepareSearchRow(rowToObject(cols, arr)));
      API_CACHE.conductorRows.set(bucket, rows);
      for (const row of rows) {
        const old = API_CACHE.rowsById.get(row.id);
        if (!old) API_CACHE.rowsById.set(row.id, row);
      }
      return rows;
    })());
  }
  return API_CACHE.conductorPromises.get(bucket);
}
async function loadHashMap(type, key) {
  const shardCount = 64;
  const shard = v20HashKey(key, shardCount);
  const cacheKey = `${type}:${shard}`;
  if (!API_CACHE.hashPromises.has(cacheKey)) {
    const prefix = type === 'disc' ? 'disc' : 'j';
    API_CACHE.hashPromises.set(cacheKey, fetchJSONData(`search/${prefix}/${prefix}${String(shard).padStart(2, '0')}.json`, `Loading ${prefix} index`).then(d => d.map || {}));
  }
  return API_CACHE.hashPromises.get(cacheKey);
}
async function idsForJ(j) {
  const map = await loadHashMap('j', j);
  return map[String(j)] || [];
}
async function idsForDisc(disc) {
  const map = await loadHashMap('disc', disc);
  return map[String(disc)] || [];
}
function parseConductorHint(q) {
  const compact = normalizeSearchText(q).replace(/\s+/g, '');
  let m = compact.match(/^(\d{1,5})(?:\.|[a-zA-Z])/);
  if (m) return Number(m[1]);
  m = compact.match(/^N\s*=?\s*(\d+)$/i);
  if (m) return Number(m[1]);
  if (/^\d{1,5}$/.test(compact)) return Number(compact);
  return null;
}
function parseConductorListingQuery(q) {
  const compact = compactExactKey(q);
  let m = compact.match(/^(\d{1,5})\.$/);
  if (m) return { N:Number(m[1]), prefix:`${m[1]}.`, conductorOnly:true };
  m = compact.match(/^(\d{1,5})\.([a-z]+\d*)$/i);
  if (m) return { N:Number(m[1]), prefix:`${m[1]}.${m[2].toLowerCase()}`, conductorOnly:false };
  return null;
}
function naturalLabelCompare(a, b) {
  return String(a.label || '').localeCompare(String(b.label || ''), undefined, { numeric:true, sensitivity:'base' });
}
function filterConductorListingRows(rows, parsed) {
  if (!parsed) return rows || [];
  const pref = String(parsed.prefix || '').toLowerCase();
  return (rows || []).filter(row => {
    prepareSearchRow(row);
    return parsed.conductorOnly ? String(row.N) === String(parsed.N) : (row.__labelKey.startsWith(pref) || row.__isoKey.startsWith(pref));
  });
}
async function loadTauRows() {
  if (API_CACHE.tauRows) return API_CACHE.tauRows;
  if (!API_CACHE.tauPromise) {
    API_CACHE.tauPromise = (async () => {
      const packed = await fetchJSONData('tau_index.json', 'Loading C-isogeny index');
      const cols = packed.columns || [];
      API_CACHE.tauRows = (packed.rows || []).map(arr => prepareSearchRow(rowToObject(cols, arr)));
      return API_CACHE.tauRows;
    })();
  }
  return API_CACHE.tauPromise;
}
async function loadCurveDatabase(reason = 'curve data') {
  // Backward-compatible helper exposed for debugging; v20 no longer loads the
  // 19 MB full curve database for hover/search/detail.
  await loadSearchManifest();
  return [...API_CACHE.rowsById.values()];
}
async function apiMeta() {
  if (!API_CACHE.meta) API_CACHE.meta = await fetchJSONData('plot_meta.json', 'Loading star-map metadata');
  return API_CACHE.meta;
}
async function apiTop() {
  if (!API_CACHE.top) API_CACHE.top = await fetchJSONData('top_points.json', 'Loading first stars');
  return API_CACHE.top;
}
async function apiTile(tileId) {
  API_CACHE.tilesRequested += 1;
  const data = await fetchJSONData(`tiles/${tileId}.json`, `Loading sky tiles ${API_CACHE.tilesLoaded}/${Math.max(API_CACHE.tilesRequested, 1)}`);
  API_CACHE.tilesLoaded += 1;
  setAtlasProgress(`Sky tiles ${API_CACHE.tilesLoaded}/${Math.max(API_CACHE.tilesRequested, API_CACHE.tilesLoaded)}`, Math.min(0.92, API_CACHE.tilesLoaded / Math.max(API_CACHE.tilesRequested, 1)));
  return data;
}
async function loadSatoTateGroups() {
  if (API_CACHE.stGroups) return API_CACHE.stGroups;
  if (!API_CACHE.stPromise) API_CACHE.stPromise = fetchJSONData('sato_tate_groups.json', 'Loading Sato-Tate data').then(d => (API_CACHE.stGroups = d));
  return API_CACHE.stPromise;
}
async function loadCMDiscIndex() {
  if (API_CACHE.cmDiscIndex) return API_CACHE.cmDiscIndex;
  if (!API_CACHE.cmDiscPromise) {
    API_CACHE.cmDiscPromise = fetchJSONData('cm_disc.json')
      .then(packed => {
        const idx = { byId: new Map(), byLabel: new Map() };
        const cols = packed.columns || [];
        const idCol = cols.indexOf('id');
        const labelCol = cols.indexOf('label');
        const discCol = cols.indexOf('cm_disc');
        for (const row of packed.rows || []) {
          const id = idCol >= 0 ? Number(row[idCol]) : null;
          const label = labelCol >= 0 ? String(row[labelCol] || '') : '';
          const disc = discCol >= 0 ? Number(row[discCol]) : 0;
          if (!Number.isFinite(disc) || disc === 0) continue;
          if (Number.isFinite(id)) idx.byId.set(id, disc);
          if (label) idx.byLabel.set(label, disc);
        }
        API_CACHE.cmDiscIndex = idx;
        return idx;
      })
      .catch(err => {
        console.warn('CM discriminant index unavailable; showing None for CMdisc', err);
        const empty = { byId: new Map(), byLabel: new Map() };
        API_CACHE.cmDiscIndex = empty;
        return empty;
      });
  }
  return API_CACHE.cmDiscPromise;
}
function cmDiscForCurve(d, idx) {
  if (!d || !idx) return null;
  const byId = idx.byId instanceof Map ? idx.byId.get(Number(d.id)) : null;
  const byLabel = idx.byLabel instanceof Map ? idx.byLabel.get(String(d.label || '')) : null;
  const value = byId ?? byLabel ?? null;
  return Number.isFinite(Number(value)) && Number(value) !== 0 ? Number(value) : null;
}
function formatCMDiscValue(value) { return value == null ? 'None' : String(value); }
function toBI(v) { return BigInt(String(v)); }
function invariantsBigFromRow(d) {
  const a1 = toBI(d.a1), a2 = toBI(d.a2), a3 = toBI(d.a3), a4 = toBI(d.a4), a6 = toBI(d.a6);
  const b2 = a1*a1 + 4n*a2;
  const b4 = 2n*a4 + a1*a3;
  const b6 = a3*a3 + 4n*a6;
  const b8 = a1*a1*a6 + 4n*a2*a6 - a1*a3*a4 + a2*a3*a3 - a4*a4;
  const c4 = b2*b2 - 24n*b4;
  const c6 = -b2*b2*b2 + 36n*b2*b4 - 216n*b6;
  const disc = -b2*b2*b8 - 8n*b4*b4*b4 - 27n*b6*b6 + 9n*b2*b4*b6;
  return { b2, b4, b6, b8, c4, c6, disc };
}
function vPBig(n, p) { n = absBI(BigInt(n)); p = BigInt(p); if (n === 0n) return 0; let e = 0; while (n % p === 0n) { n /= p; e++; } return e; }
function primesBelow(n) {
  const sieve = Array(n).fill(true); sieve[0] = sieve[1] = false;
  for (let p = 2; p*p < n; p++) if (sieve[p]) for (let k = p*p; k < n; k += p) sieve[k] = false;
  return sieve.map((ok, i) => ok ? i : 0).filter(Boolean);
}
function modSmall(v, p) { return Number(modBI(BigInt(v), BigInt(p))); }
function modBIResidue(v, p) { return modBI(BigInt(v), BigInt(p)); }
function pAdicValBI(n, p) {
  n = absBI(BigInt(n)); p = BigInt(p);
  if (n === 0n) return 1000000000;
  let e = 0;
  while (n % p === 0n) { n /= p; e++; }
  return e;
}
function pDividesBI(n, p) { return BigInt(n) % BigInt(p) === 0n; }
function pPowBI(p, e) { let out = 1n; const q = BigInt(p); for (let i=0; i<e; i++) out *= q; return out; }
function invModSmall(a, p) {
  a = ((Number(a) % p) + p) % p;
  for (let x=1; x<p; x++) if ((a * x) % p === 1) return x;
  return 0;
}
function powModSmall(a, e, p) {
  let b = ((Number(a) % p) + p) % p, out = 1 % p;
  while (e > 0) { if (e & 1) out = (out * b) % p; b = (b * b) % p; e >>= 1; }
  return out;
}
function legendreSmall(a, p) {
  a = ((Number(a) % p) + p) % p;
  if (a === 0) return 0;
  const v = powModSmall(a, (p - 1) >> 1, p);
  return v === 1 ? 1 : -1;
}
function rootModPowerSmall(a, e, p) {
  const target = modBIResidue(a, p);
  for (let x=0; x<p; x++) {
    let v = 1n;
    for (let i=0; i<e; i++) v = (v * BigInt(x)) % BigInt(p);
    if (v === target) return BigInt(x);
  }
  return 0n;
}
function quadHasRootModBI(a, b, c, p) {
  const P = BigInt(p);
  a = modBI(a, P); b = modBI(b, P); c = modBI(c, P);
  for (let x=0; x<p; x++) {
    const X = BigInt(x);
    if (modBI(a*X*X + b*X + c, P) === 0n) return true;
  }
  return false;
}
function cubicRootCountModBI(b, c, d, p) {
  const P = BigInt(p);
  b = modBI(b, P); c = modBI(c, P); d = modBI(d, P);
  let count = 0;
  for (let x=0; x<p; x++) {
    const X = BigInt(x);
    if (modBI(X*X*X + b*X*X + c*X + d, P) === 0n) count += 1;
  }
  return count;
}
function normalizeKodairaSymbol(k) {
  return String(k || '').replace(/\s+/g, '').replace('^*', '*');
}
function kodairaHtml(k) {
  k = normalizeKodairaSymbol(k);
  const m = /^I(\d+)(\*)?$/.exec(k);
  if (m) return `I<sub>${m[1]}</sub>${m[2] ? '<sup>*</sup>' : ''}`;
  const s = k.endsWith('*') ? k.slice(0, -1) : k;
  return `${escapeHtml(s)}${k.endsWith('*') ? '<sup>*</sup>' : ''}`;
}
function rstTransformAInvariants(ai, r, ss, t) {
  r = BigInt(r); ss = BigInt(ss); t = BigInt(t);
  const [a1,a2,a3,a4,a6] = ai.map(BigInt);
  return [
    a1 + 2n*ss,
    a2 - ss*a1 + 3n*r - ss*ss,
    a3 + r*a1 + 2n*t,
    a4 - ss*a3 + 2n*r*a2 - (t + r*ss)*a1 + 3n*r*r - 2n*ss*t,
    a6 + r*a4 + r*r*a2 + r*r*r - t*a3 - t*t - r*t*a1,
  ];
}
function invariantsBigFromA(ai) {
  const [a1,a2,a3,a4,a6] = ai.map(BigInt);
  const b2 = a1*a1 + 4n*a2;
  const b4 = 2n*a4 + a1*a3;
  const b6 = a3*a3 + 4n*a6;
  const b8 = a1*a1*a6 + 4n*a2*a6 - a1*a3*a4 + a2*a3*a3 - a4*a4;
  const c4 = b2*b2 - 24n*b4;
  const c6 = -b2*b2*b2 + 36n*b2*b4 - 216n*b6;
  const disc = -b2*b2*b8 - 8n*b4*b4*b4 - 27n*b6*b6 + 9n*b2*b4*b6;
  return { b2, b4, b6, b8, c4, c6, disc };
}
function divExactBI(n, d) { return BigInt(n) / BigInt(d); }
function tateLocalData(d, p) {
  const P = BigInt(p);
  let ai = [toBI(d.a1), toBI(d.a2), toBI(d.a3), toBI(d.a4), toBI(d.a6)];
  let loopGuard = 0;
  while ((++loopGuard) < 16) {
    let inv = invariantsBigFromA(ai);
    let valDisc = pAdicValBI(inv.disc, p);
    if (valDisc === 0) {
      return { kodaira: 'I0', tamagawa: 1, conductor_exp: 0, split: null, vp_disc: 0, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: 0, minimal_ai: ai.slice() };
    }
    let r = 0n, t = 0n;
    if (p === 2) {
      if (pDividesBI(inv.b2, p)) {
        r = rootModPowerSmall(ai[3], 2, p);
        t = rootModPowerSmall(((r + ai[1]) * r + ai[3]) * r + ai[4], 2, p);
      } else {
        const a1i = BigInt(invModSmall(modSmall(ai[0], p), p));
        r = modBI(a1i * ai[2], P);
        t = modBI(a1i * (ai[3] + r*r), P);
      }
    } else if (p === 3) {
      if (pDividesBI(inv.b2, p)) r = rootModPowerSmall(-inv.b6, 3, p);
      else r = modBI(-BigInt(invModSmall(modSmall(inv.b2, p), p)) * inv.b4, P);
      t = ai[0] * r + ai[2];
    } else {
      if (pDividesBI(inv.c4, p)) r = modBI(-BigInt(invModSmall(12, p)) * inv.b2, P);
      else r = modBI(-BigInt(invModSmall(modSmall(12n * inv.c4, p), p)) * (inv.c6 + inv.b2 * inv.c4), P);
      const half = BigInt(invModSmall(2, p));
      t = modBI(-half * (ai[0] * r + ai[2]), P);
    }
    ai = rstTransformAInvariants(ai, r, 0n, t);
    inv = invariantsBigFromA(ai);
    valDisc = pAdicValBI(inv.disc, p);
    if (!pDividesBI(inv.c4, p)) {
      const split = quadHasRootModBI(1n, ai[0], -ai[1], p);
      const cp = split ? valDisc : (valDisc % 2 === 0 ? 2 : 1);
      return { kodaira: `I${valDisc}`, tamagawa: cp, conductor_exp: 1, split, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    if (pAdicValBI(ai[4], p) < 2) return { kodaira: 'II', tamagawa: 1, conductor_exp: valDisc, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    if (pAdicValBI(inv.b8, p) < 3) return { kodaira: 'III', tamagawa: 2, conductor_exp: valDisc - 1, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    if (pAdicValBI(inv.b6, p) < 3) {
      const a3t = divExactBI(ai[2], P);
      const a6t = divExactBI(ai[4], P*P);
      const cp = quadHasRootModBI(1n, a3t, -a6t, p) ? 3 : 1;
      return { kodaira: 'IV', tamagawa: cp, conductor_exp: valDisc - 2, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    let s = 0n;
    if (p === 2) {
      s = rootModPowerSmall(ai[1], 2, p);
      t = P * rootModPowerSmall(divExactBI(ai[4], P*P), 2, p);
    } else if (p === 3) {
      s = ai[0];
      t = ai[2];
    } else {
      const half = BigInt(invModSmall(2, p));
      s = modBI(-ai[0] * half, P);
      t = modBI(-ai[2] * half, P);
    }
    ai = rstTransformAInvariants(ai, 0n, s, t);
    inv = invariantsBigFromA(ai);
    valDisc = pAdicValBI(inv.disc, p);
    const b = divExactBI(ai[1], P);
    const c = divExactBI(ai[3], P*P);
    const dd = divExactBI(ai[4], P*P*P);
    const w = 27n*dd*dd - b*b*c*c + 4n*b*b*b*dd - 18n*b*c*dd + 4n*c*c*c;
    const x = 3n*c - b*b;
    let sw = 1;
    if (pDividesBI(w, p)) sw = pDividesBI(x, p) ? 3 : 2;
    if (sw === 1) {
      const cp = 1 + cubicRootCountModBI(b, c, dd, p);
      return { kodaira: 'I0*', tamagawa: cp, conductor_exp: valDisc - 4, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    if (sw === 2) {
      if (p === 2) r = rootModPowerSmall(c, 2, p);
      else if (p === 3) r = modBI(c * BigInt(invModSmall(modSmall(b, p), p)), P);
      else r = modBI((b*c - 9n*dd) * BigInt(invModSmall(modSmall(2n*x, p), p)), P);
      ai = rstTransformAInvariants(ai, P * r, 0n, 0n);
      inv = invariantsBigFromA(ai);
      let ix = 3, iy = 3;
      let mx = P * P, my = mx;
      let cp = 2;
      let guard = 0;
      while ((++guard) < 12) {
        const a2t = divExactBI(ai[1], P);
        const a3t = divExactBI(ai[2], my);
        const a4t = divExactBI(ai[3], P * mx);
        const a6t = divExactBI(ai[4], mx * my);
        if (pDividesBI(a3t*a3t + 4n*a6t, p)) {
          if (p === 2) t = my * rootModPowerSmall(a6t, 2, p);
          else t = my * modBI(-a3t * BigInt(invModSmall(2, p)), P);
          ai = rstTransformAInvariants(ai, 0n, 0n, t);
          inv = invariantsBigFromA(ai);
          my *= P; iy += 1;
          const a2u = divExactBI(ai[1], P);
          const a4u = divExactBI(ai[3], P * mx);
          const a6u = divExactBI(ai[4], mx * my);
          if (pDividesBI(a4u*a4u - 4n*a6u*a2u, p)) {
            if (p === 2) r = mx * rootModPowerSmall(a6u * BigInt(invModSmall(modSmall(a2u, p), p)), 2, p);
            else r = mx * modBI(-a4u * BigInt(invModSmall(modSmall(2n*a2u, p), p)), P);
            ai = rstTransformAInvariants(ai, r, 0n, 0n);
            inv = invariantsBigFromA(ai);
            mx *= P; ix += 1;
            continue;
          }
          cp = quadHasRootModBI(a2u, a4u, a6u, p) ? 4 : 2;
          break;
        }
        cp = quadHasRootModBI(1n, a3t, -a6t, p) ? 4 : 2;
        break;
      }
      valDisc = pAdicValBI(inv.disc, p);
      return { kodaira: `I${ix + iy - 5}*`, tamagawa: cp, conductor_exp: valDisc - ix - iy + 1, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    if (p === 2) r = modBI(b, P);
    else if (p === 3) r = rootModPowerSmall(-dd, 3, p);
    else r = modBI(-b * BigInt(invModSmall(3, p)), P);
    ai = rstTransformAInvariants(ai, P * r, 0n, 0n);
    inv = invariantsBigFromA(ai);
    const a3t = divExactBI(ai[2], P*P);
    const a6t = divExactBI(ai[4], P*P*P*P);
    if (!pDividesBI(a3t*a3t + 4n*a6t, p)) {
      const cp = quadHasRootModBI(1n, a3t, -a6t, p) ? 3 : 1;
      valDisc = pAdicValBI(inv.disc, p);
      return { kodaira: 'IV*', tamagawa: cp, conductor_exp: valDisc - 6, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    if (p === 2) t = -(P*P) * rootModPowerSmall(a6t, 2, p);
    else t = (P*P) * modBI(-a3t * BigInt(invModSmall(2, p)), P);
    ai = rstTransformAInvariants(ai, 0n, 0n, t);
    inv = invariantsBigFromA(ai);
    if (pAdicValBI(ai[3], p) < 4) {
      valDisc = pAdicValBI(inv.disc, p);
      return { kodaira: 'III*', tamagawa: 2, conductor_exp: valDisc - 7, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    if (pAdicValBI(ai[4], p) < 6) {
      valDisc = pAdicValBI(inv.disc, p);
      return { kodaira: 'II*', tamagawa: 1, conductor_exp: valDisc - 8, split: null, vp_disc: valDisc, vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, valDisc - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
    }
    ai = [
      divExactBI(ai[0], P),
      divExactBI(ai[1], P*P),
      divExactBI(ai[2], P*P*P),
      divExactBI(ai[3], P*P*P*P),
      divExactBI(ai[4], P*P*P*P*P*P),
    ];
  }
  const inv = invariantsBigFromA(ai);
  return { kodaira: '?', tamagawa: null, conductor_exp: vPBig(BigInt(d.N || 0), p), split: null, vp_disc: pAdicValBI(inv.disc, p), vp_c4: pAdicValBI(inv.c4, p), vp_den_j: Math.max(0, pAdicValBI(inv.disc, p) - 3 * pAdicValBI(inv.c4, p)), minimal_ai: ai.slice() };
}
function countReducedPoints(d, p) {
  const a1 = modSmall(d.a1, p), a2 = modSmall(d.a2, p), a3 = modSmall(d.a3, p), a4 = modSmall(d.a4, p), a6 = modSmall(d.a6, p);
  let nonsingular = 0, affine = 0; const singular = [];
  for (let x=0; x<p; x++) {
    const x2 = (x * x) % p;
    const x3 = (x2 * x) % p;
    for (let y=0; y<p; y++) {
      const lhs = (y*y + a1*x*y + a3*y) % p;
      const rhs = (x3 + a2*x2 + a4*x + a6) % p;
      if (((lhs - rhs) % p + p) % p !== 0) continue;
      affine += 1;
      const fx = ((a1*y - 3*x2 - 2*a2*x - a4) % p + p) % p;
      const fy = ((2*y + a1*x + a3) % p + p) % p;
      if (fx === 0 && fy === 0) singular.push([x,y]); else nonsingular += 1;
    }
  }
  return { total_points: affine + 1, smooth_points: nonsingular + 1, singular_points: singular };
}
function tangentConeRootCount(ai, p, x0) {
  const a1 = modSmall(ai[0], p), a2 = modSmall(ai[1], p);
  const xp = ((Number(x0) % p) + p) % p;
  const A = (p - ((3 * xp + a2) % p)) % p;
  const B = a1;
  const C = 1 % p;
  let roots = 0;
  // Projective tangent directions: X=1, Y=m, plus the vertical direction X=0.
  for (let m=0; m<p; m++) {
    const v = (C*m*m + B*m + A) % p;
    if (v === 0) roots += 1;
  }
  if (C % p === 0) roots += 1;
  return roots;
}
function reductionGeometryFromA(ai, p) {
  const a1 = modSmall(ai[0], p), a2 = modSmall(ai[1], p), a3 = modSmall(ai[2], p), a4 = modSmall(ai[3], p), a6 = modSmall(ai[4], p);
  const singular = [];
  for (let x=0; x<p; x++) {
    const x2 = (x * x) % p;
    const x3 = (x2 * x) % p;
    for (let y=0; y<p; y++) {
      const lhs = (y*y + a1*x*y + a3*y) % p;
      const rhs = (x3 + a2*x2 + a4*x + a6) % p;
      if (((lhs - rhs) % p + p) % p !== 0) continue;
      const fx = ((a1*y - 3*x2 - 2*a2*x - a4) % p + p) % p;
      const fy = ((2*y + a1*x + a3) % p + p) % p;
      if (fx === 0 && fy === 0) singular.push([x,y]);
    }
  }
  if (!singular.length) return { kind:'good', split:null, singular_points:[] };
  const [x0] = singular[0];
  const tangentRoots = tangentConeRootCount(ai, p, x0);
  if (tangentRoots >= 2) return { kind:'split multiplicative', split:true, singular_points:singular, tangent_roots:tangentRoots };
  if (tangentRoots === 0) return { kind:'nonsplit multiplicative', split:false, singular_points:singular, tangent_roots:tangentRoots };
  return { kind:'additive', split:null, singular_points:singular, tangent_roots:tangentRoots };
}
function checkedSplitFromLocalGeometry(local, p) {
  // The user-facing local table only asks for primes p<100.  For larger
  // bad primes used internally while inferring root numbers, use the cheap
  // Tate-algorithm split flag and avoid an O(p^2) finite-field scan.
  if (p > 101) return local.split;
  const geom = reductionGeometryFromA(local.minimal_ai || [], p);
  if (geom.kind === 'split multiplicative') return true;
  if (geom.kind === 'nonsplit multiplicative') return false;
  return local.split;
}
function badRootFromTate(local, p) {
  const k = normalizeKodairaSymbol(local.kodaira);
  if (k === 'I0') return 1;
  if (/^I\d+$/.test(k)) return local.split ? -1 : 1;
  if (p === 2 || p === 3) return null;
  if (/^I\d+\*$/.test(k) || k === 'I0*' || k === 'II' || k === 'II*') return legendreSmall(-1, p);
  if (k === 'III' || k === 'III*') return legendreSmall(-2, p);
  if (k === 'IV' || k === 'IV*') return legendreSmall(-3, p);
  return null;
}
function localRowFromTate(d, p, pointCounts = null) {
  const local = tateLocalData(d, p);
  const k = normalizeKodairaSymbol(local.kodaira);
  const vpN = vPBig(BigInt(d.N || 0), p);
  let a_p = 0, reduction = 'additive', typeLabel = 'add.', root = badRootFromTate(local, p);
  let pts = pointCounts || null;
  if (!pts && k === 'I0') pts = countReducedPoints(d, p);
  if (!pts) pts = { smooth_points: null, total_points: null, singular_points: [] };
  const geom = p <= 101 ? reductionGeometryFromA(local.minimal_ai || [toBI(d.a1), toBI(d.a2), toBI(d.a3), toBI(d.a4), toBI(d.a6)], p) : { kind: reduction, singular_points: [] };
  if (k === 'I0') {
    a_p = p + 1 - pts.total_points;
    reduction = 'good';
    typeLabel = (a_p % p === 0) ? 's.sing.' : 'ord.';
    root = 1;
  } else if (/^I\d+$/.test(k)) {
    const split = checkedSplitFromLocalGeometry(local, p);
    a_p = split ? 1 : -1;
    reduction = split ? 'split multiplicative' : 'nonsplit multiplicative';
    typeLabel = split ? 's.mul.' : 'n.mul.';
    root = split ? -1 : 1;
  } else {
    a_p = 0;
    reduction = 'additive';
    typeLabel = 'add.';
  }
  return {
    p,
    a_p,
    tamagawa: local.tamagawa,
    kodaira: k,
    type_label: typeLabel,
    reduction,
    root_number: root,
    ord_N: vpN,
    vp_disc: local.vp_disc,
    ord_disc: local.vp_disc,
    vp_c4: local.vp_c4,
    ord_den_j: local.vp_den_j,
    smooth_points: pts.smooth_points,
    total_points: pts.total_points,
    singular_points: geom.singular_points || pts.singular_points || [],
    local_check: { geometric_reduction: geom.kind, conductor_exp: local.conductor_exp }
  };
}
function inferMissingWildRootNumbers(d, rows) {
  const byP = new Map(rows.map(r => [r.p, r]));
  const badPrimes = [...factorIntSmall(d.N || 0).keys()].sort((a,b) => a-b);
  const globalRoot = (Number(d.rank || 0) % 2) ? -1 : 1;
  let knownProduct = -1; // w_infinity = -1, so include the archimedean sign.
  const missing = [];
  for (const p of badPrimes) {
    let row = byP.get(p);
    if (!row) row = localRowFromTate(d, p, null);
    if (row.root_number === null || row.root_number === undefined) missing.push({ p, row });
    else knownProduct *= Number(row.root_number);
  }
  if (missing.length === 1) {
    const inferred = globalRoot / knownProduct;
    const target = byP.get(missing[0].p);
    if (target && (inferred === 1 || inferred === -1)) target.root_number = inferred;
  }
  return rows;
}
function reductionData(d, bound = 100) {
  const rows = [];
  for (const p of primesBelow(bound)) rows.push(localRowFromTate(d, p, countReducedPoints(d, p)));
  return inferMissingWildRootNumbers(d, rows);
}
function displayRootNumber(v) { return v === null || v === undefined ? '?' : String(v); }
function localTypeTitle(v) {
  return ({ 's.sing.': 'supersingular good reduction', 'ord.': 'ordinary good reduction', 's.mul.': 'split multiplicative reduction', 'n.mul.': 'non-split multiplicative reduction', 'add.': 'additive reduction' })[v] || v || '';
}
function factorIntSmall(n) {
  n = Math.abs(Math.trunc(Number(n))); const out = new Map();
  let d = 2;
  while (d*d <= n) { while (n % d === 0) { out.set(d, (out.get(d)||0)+1); n = Math.floor(n/d); } d = d === 2 ? 3 : d + 2; }
  if (n > 1) out.set(n, (out.get(n)||0)+1);
  return out;
}
function localCoeffPrimePower(a_p, p, k, reduction) {
  if (k === 0) return 1;
  if (reduction === 'good') {
    if (k === 1) return a_p;
    let a0 = 1, a1 = a_p;
    for (let i=2; i<=k; i++) { const t = a_p*a1 - p*a0; a0 = a1; a1 = t; }
    return a1;
  }
  if (reduction === 'split multiplicative' || reduction === 'nonsplit multiplicative') return Math.pow(a_p, k);
  return 0;
}
function newformCoefficients(d, bound = 30, precomputedReduction = null) {
  const reduction = precomputedReduction || reductionData(d, Math.max(bound + 1, 100));
  const local = new Map(reduction.map(row => [row.p, row]));
  const coeffs = {1: 1};
  for (let n=2; n<=bound; n++) {
    let val = 1;
    for (const [p,e] of factorIntSmall(n)) {
      const row = local.get(p);
      val *= localCoeffPrimePower(row.a_p, p, e, row.reduction);
    }
    coeffs[n] = val;
  }
  return coeffs;
}
function formatQExpansion(coeffs, bound = null) {
  if (bound == null) bound = Math.max(...Object.keys(coeffs).map(Number));
  const terms = [];
  for (let n=1; n<=bound; n++) {
    const a = coeffs[n] || 0; if (!a) continue;
    if (n === 1) { terms.push('q'); continue; }
    const sign = a > 0 ? '+' : '-'; const mag = Math.abs(a); const coeff = mag === 1 ? '' : String(mag); const term = `${coeff}q^${n}`;
    terms.push(` ${sign} ${term}`);
  }
  return (terms.join('') || '0') + ` + O(q^${bound + 1})`;
}
function makeCompactMember(row) { return enrichClientCurve(row); }
async function buildTauBuckets() {
  if (API_CACHE.tauBuckets) return API_CACHE.tauBuckets;
  const rows = await loadTauRows();
  const buckets = new Map();
  const scale = 1000000;
  let i = 0;
  for (const row of rows || []) {
    const key = `${Math.round(Number(row.tau_re) * scale)},${Math.round(Number(row.tau_im) * scale)}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(row);
    if (((++i) & 4095) === 0) await idleYield(8);
  }
  API_CACHE.tauBuckets = buckets;
  return buckets;
}

function cApply(z, mat) {
  const [a,b,c,d] = mat;
  const den = { re: c*z.re + d, im: c*z.im };
  const den2 = den.re*den.re + den.im*den.im;
  if (den2 < 1e-24) return null;
  const num = { re: a*z.re + b, im: a*z.im };
  return { re: (num.re*den.re + num.im*den.im)/den2, im: (num.im*den.re - num.re*den.im)/den2 };
}
function matMul(L, R) { const [a,b,c,d]=L, [e,f,g,h]=R; return [a*e+b*g, a*f+b*h, c*e+d*g, c*f+d*h]; }
function reduceTau(z) {
  let Rm = [1,0,0,1], changed = false; z = { re:z.re, im:z.im };
  for (let iter=0; iter<40; iter++) {
    if (z.im <= 0) return [z,Rm,changed];
    const n = Math.round(z.re);
    if (n !== 0) { const T=[1,-n,0,1]; const z2=cApply(z,T); if (!z2) break; z=z2; Rm=matMul(T,Rm); changed=true; continue; }
    const abs = Math.hypot(z.re,z.im);
    if (abs < 1 - 1e-12 || (Math.abs(abs - 1) <= 1e-12 && z.re < -1e-12)) { const S=[0,-1,1,0]; const z2=cApply(z,S); if (!z2) break; z=z2; Rm=matMul(S,Rm); changed=true; continue; }
    break;
  }
  return [z,Rm,changed];
}
function relationString(mat) { const [a,b,c,d]=mat; return `τ′=(${a}τ${b>=0?'+':''}${b})/(${c}τ${d>=0?'+':''}${d})`; }
function gcdInt(a, b) {
  a = Math.abs(a | 0); b = Math.abs(b | 0);
  while (b) { const t = a % b; a = b; b = t; }
  return a;
}

function relationMatrices() {
  if (relationMatrices.cache) return relationMatrices.cache;
  const mats = [], H = 18, DET_MAX = 512;
  for (let a=-H; a<=H; a++) for (let b=-H; b<=H; b++) for (let c=-H; c<=H; c++) for (let d=-H; d<=H; d++) {
    if (a===0&&b===0&&c===0&&d===0) continue;
    const first = [a,b,c,d].find(x => x !== 0); if (first < 0) continue;
    const det = a*d - b*c; if (det <= 0 || det > DET_MAX) continue;
    if (gcdInt(gcdInt(gcdInt(Math.abs(a), Math.abs(b)), Math.abs(c)), Math.abs(d)) !== 1) continue;
    const h = Math.max(Math.abs(a),Math.abs(b),Math.abs(c),Math.abs(d)); mats.push([a,b,c,d,h,det]);
  }
  mats.sort((x,y) => (x[4]-y[4]) || (x[5]-y[5]) || ((Math.abs(x[0])+Math.abs(x[1])+Math.abs(x[2])+Math.abs(x[3])) - (Math.abs(y[0])+Math.abs(y[1])+Math.abs(y[2])+Math.abs(y[3]))));
  relationMatrices.cache = mats; return mats;
}
async function relationMatricesAsync() {
  if (relationMatrices.cache) return relationMatrices.cache;
  if (relationMatrices.promise) return relationMatrices.promise;
  relationMatrices.promise = (async () => {
    const mats = [], H = 18, DET_MAX = 512;
    for (let a=-H; a<=H; a++) {
      for (let b=-H; b<=H; b++) for (let c=-H; c<=H; c++) for (let d=-H; d<=H; d++) {
        if (a===0&&b===0&&c===0&&d===0) continue;
        const first = [a,b,c,d].find(x => x !== 0); if (first < 0) continue;
        const det = a*d - b*c; if (det <= 0 || det > DET_MAX) continue;
        if (gcdInt(gcdInt(gcdInt(Math.abs(a), Math.abs(b)), Math.abs(c)), Math.abs(d)) !== 1) continue;
        const h = Math.max(Math.abs(a),Math.abs(b),Math.abs(c),Math.abs(d)); mats.push([a,b,c,d,h,det]);
      }
      if ((a & 1) === 0) await idleYield(8);
    }
    mats.sort((x,y) => (x[4]-y[4]) || (x[5]-y[5]) || ((Math.abs(x[0])+Math.abs(x[1])+Math.abs(x[2])+Math.abs(x[3])) - (Math.abs(y[0])+Math.abs(y[1])+Math.abs(y[2])+Math.abs(y[3]))));
    relationMatrices.cache = mats;
    return mats;
  })().catch(err => { relationMatrices.promise = null; throw err; });
  return relationMatrices.promise;
}
function tauMatchEps(mat, tau) {
  const [a,b,c,d] = mat;
  const det = Math.abs(a*d - b*c);
  const height = Math.max(Math.abs(a), Math.abs(b), Math.abs(c), Math.abs(d));
  const denom = Math.hypot(c*tau.re + d, c*tau.im);
  // tau_index is stored to about 7 decimals, so the floor cannot be much
  // smaller than 1e-7.  Keep the tolerance tight enough to avoid accidental
  // high-height Möbius matches, but allow the derivative of the transform.
  const mapped = (det / Math.max(denom*denom, 1e-10)) * 1.0e-7;
  const coeffBoost = 1.0 + 0.012 * height;
  return Math.min(2.4e-6, Math.max(2.2e-7, 2.35*mapped*coeffBoost + 1.2e-7));
}
function bucketCandidates(buckets, target, eps) {
  const scale = 1000000, kr = Math.round(target.re*scale), ki = Math.round(target.im*scale), rad = Math.max(1, Math.min(12, Math.ceil(eps*scale)+1));
  const out = [];
  for (let dr=-rad; dr<=rad; dr++) for (let di=-rad; di<=rad; di++) {
    const arr = buckets.get(`${kr+dr},${ki+di}`); if (arr) out.push(...arr);
  }
  return out;
}

function conductorSanity(curve, cand, height, det, errSum, eps, sameQClass) {
  const N1 = Math.max(1, Number(curve.N || 0));
  const N2 = Math.max(1, Number(cand.N || 0));
  if (sameQClass) return { ok: true, tag: 'same Q-isogeny class' };
  const g = Number(gcdBI(BigInt(N1), BigInt(N2)));
  if (g <= 1) return { ok: false, tag: 'rejected: coprime conductors' };
  const mn = Math.min(N1, N2), mx = Math.max(N1, N2);
  const multiple = mx % mn === 0;
  const tight = errSum <= Math.max(2.6e-7, eps * 0.72);
  const veryTight = errSum <= Math.max(1.8e-7, eps * 0.42);
  // A numerical complex-isogeny test based only on rounded τ can otherwise
  // create fake matches at large height.  We therefore use conductor data as a
  // sanity filter rather than as a proof: coprime conductors are rejected,
  // non-multiple conductor pairs must be both low-height and very tight.
  if (multiple) {
    if (height <= 18 && det <= 512 && tight) return { ok: true, tag: 'conductor multiple sanity' };
    return { ok: false, tag: 'rejected: weak multiple-conductor match' };
  }
  if (height <= 8 && det <= 96 && veryTight) return { ok: true, tag: 'strict non-multiple sanity' };
  return { ok: false, tag: 'rejected: weak non-multiple conductor match' };
}
async function detectedCIsogenyNeighbours(curve, limit = Infinity, shouldCancel = null, onProgress = null) {
  if (shouldCancel && shouldCancel()) return { canceled: true, items: [] };
  const buckets = await buildTauBuckets();
  if (shouldCancel && shouldCancel()) return { canceled: true, items: [] };
  const mats = await relationMatricesAsync();
  const tau = { re:Number(curve.tau_re), im:Number(curve.tau_im) };
  const found = new Map();
  let checked = 0;
  for (const [a,b,c,d,h,det] of mats) {
    if (shouldCancel && shouldCancel()) return { canceled: true, items: [] };
    const base = [a,b,c,d]; const raw = cApply(tau, base); if (!raw || raw.im <= 0) continue;
    const [reducedTarget, redMat, reduced] = reduceTau(raw); if (reducedTarget.im <= 0) continue;
    const combined = reduced ? matMul(redMat, base) : base; const [ca,cb,cc,cd] = combined; const eps = tauMatchEps(combined, tau);
    const invDet = ca*cd - cb*cc; if (invDet <= 0) continue;
    for (const cand of bucketCandidates(buckets, reducedTarget, eps)) {
      if (Number(cand.id) === Number(curve.id)) continue;
      const candTau = { re:Number(cand.tau_re), im:Number(cand.tau_im) };
      const forwardErr = Math.hypot(candTau.re - reducedTarget.re, candTau.im - reducedTarget.im); if (forwardErr > eps) continue;
      const back = cApply(candTau, [cd, -cb, -cc, ca]); if (!back) continue;
      const inverseErr = Math.hypot(back.re - tau.re, back.im - tau.im); const invEps = Math.max(eps, tauMatchEps([cd,-cb,-cc,ca], candTau)); if (inverseErr > invEps) continue;
      const sameQClass = cand.iso === curve.iso;
      const errSum = forwardErr + inverseErr;
      const sanity = conductorSanity(curve, cand, h, det, errSum, eps, sameQClass);
      if (!sanity.ok) continue;
      const old = found.get(cand.id);
      const score = [sameQClass ? 0 : 1, h, det, errSum];
      if (old && (old._score[0] < score[0] || (old._score[0]===score[0] && (old._score[1] < score[1] || (old._score[1]===score[1] && (old._score[2] < score[2] || (old._score[2]===score[2] && old._score[3] <= score[3]))))))) continue;
      const item = makeCompactMember(cand); const sameTau = Math.hypot(candTau.re - tau.re, candTau.im - tau.im) <= Math.max(3.0e-7, eps * 0.55);
      Object.assign(item, { same_tau: sameTau, same_q_isogeny_class: sameQClass, conductor_sanity: sanity.tag, relation: relationString(combined), height: h, determinant: det, error: Number(forwardErr.toFixed(10)), inverse_error: Number(inverseErr.toFixed(10)), tau_match_eps: Number(eps.toFixed(10)), verified_tau_match: true, _score: score });
      found.set(cand.id, item);
    }
    if ((++checked & 1023) === 0) {
      if (onProgress) onProgress(Math.min(0.98, checked / Math.max(1, mats.length)), found.size);
      await new Promise(r => setTimeout(r, 0));
    }
  }
  const items = [...found.values()].map(x => { delete x._score; return x; }).sort((x,y) =>
    (Number(x.N || 0) - Number(y.N || 0)) ||
    (Number(x.height || 0) - Number(y.height || 0)) ||
    (Number(x.determinant || 0) - Number(y.determinant || 0)) ||
    String(x.label).localeCompare(String(y.label))
  );
  return { canceled: false, items: Number.isFinite(limit) ? items.slice(0, limit) : items };
}
async function apiHover(id) {
  const point = state && state.points ? state.points.get(Number(id)) : null;
  const base = point ? {
    id: point.i,
    label: point.l,
    group: `E(Q) ≅ Z^${point.r}${point.t === '0' ? '' : ' ⊕ ' + point.t}`,
    N: point.N,
    rank: point.r,
    tor_label: point.t,
    cm: point.cm,
  } : { id: Number(id) };
  // Fetch one small detail shard instead of the v19 19 MB full database.
  const row = await loadCurveById(id);
  if (!row) return base;
  const d = enrichClientCurve(row);
  return { ...base, id: d.id, label: d.label, group: `E(Q) ≅ Z^${d.rank}${d.tor_label === '0' ? '' : ' ⊕ ' + d.tor_label}`, weierstrass_equation: d.weierstrass_equation, disc: d.disc, j_str: d.j_str };
}

async function loadCurveByLabelAndConductor(label, conductor = null) {
  const rawLabel = normalizeSearchText(label);
  const wanted = compactExactKey(rawLabel);
  if (!wanted) return null;
  const ids = await idsForExactKey(wanted).catch(() => []);
  if (ids.length) {
    const rows = await loadRowsByIds(ids, Math.max(ids.length, 24)).catch(() => []);
    const exact = rows.find(r => compactExactKey(r.label) === wanted || compactExactKey(r.cremona) === wanted) || rows[0];
    if (exact) return await loadCurveById(exact.id).catch(() => exact);
  }
  const N = conductor != null && !Number.isNaN(Number(conductor)) ? Math.floor(Number(conductor)) : parseConductorHint(rawLabel);
  if (N != null) {
    const nids = await idsForConductorFast(N).catch(() => []);
    if (nids.length) {
      const rows = await loadRowsByIds(nids, Math.max(nids.length, 24)).catch(() => []);
      const exact = rows.find(r => compactExactKey(r.label) === wanted || compactExactKey(r.cremona) === wanted || compactExactKey(r.iso) === wanted);
      if (exact) return await loadCurveById(exact.id).catch(() => exact);
    }
    const bucketRows = await loadConductorBucket(conductorBucketForN(N)).catch(() => []);
    const exact = bucketRows.find(r => Number(r.N) === N && (compactExactKey(r.label) === wanted || compactExactKey(r.cremona) === wanted || compactExactKey(r.iso) === wanted));
    if (exact) return await loadCurveById(exact.id).catch(() => exact);
  }
  return null;
}

async function apiSearch(q, limit = 15, options = {}) {
  q = normalizeSearchText(q); if (!q) return [];
  const opts = options || {};
  const batchSize = Math.max(1, Number(opts.batchSize || 10));
  const isCanceled = () => {
    try { return typeof opts.shouldCancel === 'function' && !!opts.shouldCancel(); }
    catch { return false; }
  };
  const finishCanceled = () => out.slice();
  const cacheKey = `search:${compactExactKey(q)}:${Number.isFinite(limit) ? limit : 'all'}`;
  const cached = API_CACHE.searchResultCache.get(cacheKey);
  if (cached?.items) {
    if (!isCanceled() && typeof opts.onBatch === 'function') opts.onBatch(cached.items.slice(), { cached:true, done:true });
    return isCanceled() ? [] : cached.items.slice();
  }
  const out = [], seen = new Set();
  let lastEmitted = 0;
  function maybeEmit(force = false, status = 'partial') {
    if (isCanceled() || typeof opts.onBatch !== 'function') return;
    if (!force && out.length - lastEmitted < batchSize) return;
    lastEmitted = out.length;
    opts.onBatch(out.slice(), { status, done:false });
  }
  function addRows(rows, match, localLimit = limit) {
    if (isCanceled()) return 0;
    const maxRows = Number.isFinite(localLimit) ? Math.max(0, Number(localLimit)) : Infinity;
    let added = 0;
    for (const row of rows || []) {
      if (isCanceled()) break;
      if (!row || seen.has(Number(row.id)) || out.length >= maxRows) continue;
      const d = enrichClientCurve(row); d.search_match = match; seen.add(Number(d.id)); out.push(d); added++;
      maybeEmit(false, 'partial');
    }
    return added;
  }
  async function addIds(ids, match, localLimit = limit) {
    if (isCanceled()) return;
    const rows = await loadRowsByIds(ids || [], Math.max((Number.isFinite(localLimit) ? localLimit : 0) * 2, Number.isFinite(localLimit) ? 20 : Infinity), (newRows) => {
      if (!isCanceled()) addRows(newRows, match, localLimit);
    });
    if (isCanceled()) return;
    addRows(rows, match, localLimit);
  }

  if (isCanceled()) return finishCanceled();
  const listing = parseConductorListingQuery(q);
  if (listing) {
    const fullLimit = Math.max(Number.isFinite(limit) ? Number(limit) : 0, 1000);
    const allRows = await loadConductorRowsForN(listing.N, (newRows) => {
      if (isCanceled()) return;
      const filtered = filterConductorListingRows(newRows, listing);
      addRows(filtered, listing.conductorOnly ? 'exact conductor' : 'conductor prefix', fullLimit);
    });
    if (isCanceled()) return finishCanceled();
    addRows(filterConductorListingRows(allRows, listing), listing.conductorOnly ? 'exact conductor' : 'conductor prefix', fullLimit);
    maybeEmit(true, 'done');
    const final = out.slice();
    API_CACHE.searchResultCache.set(cacheKey, { items:final, ts:Date.now() });
    return final;
  }

  // v23 fast path: exact label / Cremona / isogeny-class search uses a tiny
  // hash shard and avoids downloading 0.7-1.1 MB conductor buckets.
  const exactKey = compactExactKey(q);
  if (exactKey && !isCubicLikeQuery(q)) {
    const exactIds = await idsForExactKey(exactKey).catch(() => []);
    if (isCanceled()) return finishCanceled();
    if (exactIds.length) {
      await addIds(exactIds, 'exact label / Cremona / isogeny class');
      if (isCanceled()) return finishCanceled();
      if (out.length) { maybeEmit(true, 'done'); API_CACHE.searchResultCache.set(cacheKey, { items:out.slice(), ts:Date.now() }); return out; }
    }
  }

  const cubic = await identifyCubicLazy(q);
  if (isCanceled()) return finishCanceled();
  if (cubic && cubic.ok && cubic.j) {
    const ids = await idsForJ(cubic.j).catch(() => []);
    if (isCanceled()) return finishCanceled();
    if (cubic.coeffs_int && ids.length) {
      const rows = await loadRowsByIds(ids, Math.max(limit * 3, 30));
      if (isCanceled()) return finishCanceled();
      const want = JSON.stringify(cubic.coeffs_int.map(String));
      addRows(rows.filter(r => JSON.stringify([r.a1,r.a2,r.a3,r.a4,r.a6].map(String)) === want), cubic.method || 'Q-minimal model from cubic input');
      if (out.length) { maybeEmit(true, 'done'); API_CACHE.searchResultCache.set(cacheKey, { items:out.slice(), ts:Date.now() }); return out; }
      addRows(rows, cubic.method || 'same j-invariant from cubic input');
    } else {
      await addIds(ids, cubic.method || 'same j-invariant from cubic input');
    }
    if (out.length) { maybeEmit(true, 'done'); API_CACHE.searchResultCache.set(cacheKey, { items:out.slice(), ts:Date.now() }); return out; }
  }

  const rational = parseRationalKey(q);
  if (rational) {
    const maybeN = /^\d{1,5}$/.test(rational) ? Number(rational) : null;
    const [discIds, jIds, conductorIds] = await Promise.all([
      idsForDisc(rational).catch(() => []),
      idsForJ(rational).catch(() => []),
      maybeN != null ? idsForConductorFast(maybeN).catch(() => []) : Promise.resolve([]),
    ]);
    if (isCanceled()) return finishCanceled();
    await addIds(conductorIds, 'exact conductor');
    if (isCanceled()) return finishCanceled();
    await addIds(discIds, 'exact discriminant');
    if (isCanceled()) return finishCanceled();
    await addIds(jIds, 'exact j-invariant');
    if (out.length) { maybeEmit(true, 'done'); API_CACHE.searchResultCache.set(cacheKey, { items:out.slice(), ts:Date.now() }); return out; }
  }

  const needle = q.toLowerCase();
  const compactNeedle = q.replace(/\s+/g, '').toLowerCase();
  const conductor = parseConductorHint(q);
  const compactQ = q.replace(/\s+/g, '');
  const conductorOnly = /^(?:n=?)?\d{1,5}\.?$/i.test(compactQ);
  const buckets = conductor != null ? [conductorBucketForN(conductor)] : [...API_CACHE.conductorRows.keys()];
  for (const bucket of buckets) {
    if (isCanceled()) return finishCanceled();
    const rows = await loadConductorBucket(bucket).catch(() => []);
    if (isCanceled()) return finishCanceled();
    const exactHits = [], fuzzyHits = [];
    for (const row of rows) {
      if (isCanceled()) break;
      const matchLevel = rowMatchesNeedle(row, needle, compactNeedle, conductorOnly, conductor);
      if (matchLevel === 2) exactHits.push(row);
      else if (matchLevel === 1) fuzzyHits.push(row);
      if (exactHits.length + fuzzyHits.length >= limit * 2) break;
    }
    addRows(exactHits, 'label / Cremona / isogeny class');
    addRows(fuzzyHits, conductorOnly ? 'exact conductor' : 'label / Cremona / isogeny class');
    if (out.length >= limit) { maybeEmit(true, 'done'); API_CACHE.searchResultCache.set(cacheKey, { items:out.slice(), ts:Date.now() }); return out; }
  }
  // Fast top-point fallback for non-numeric substring searches before scanning all conductor buckets.
  if (!out.length && state.points && needle.length >= 2) {
    const rows = [];
    for (const p of state.points.values()) {
      if (isCanceled()) break;
      if (String(p.l).toLowerCase().includes(needle) || String(p.iso).toLowerCase().includes(needle)) {
        rows.push({ id:p.i, label:p.l, cremona:'', iso:p.iso, N:p.N, rank:p.r, tor_label:p.t, cm:p.cm, disc:'', j_str:'', az:p.az, alt:p.al });
        if (rows.length >= limit) break;
      }
    }
    addRows(rows, 'visible star label / isogeny class');
  }
  // Last resort: scan conductor buckets lazily. This preserves v19 substring search without blocking ordinary exact searches.
  if (!out.length && needle.length >= 3) {
    for (let bucket = 0; bucket <= 9; bucket++) {
      if (isCanceled()) return finishCanceled();
      if (conductor != null && bucket === conductorBucketForN(conductor)) continue;
      const rows = await loadConductorBucket(bucket).catch(() => []);
      if (isCanceled()) return finishCanceled();
      const fuzzyHits = [];
      for (const row of rows) {
        if (isCanceled()) break;
        if (rowMatchesNeedle(row, needle, compactNeedle) > 0) fuzzyHits.push(row);
        if (fuzzyHits.length >= limit * 2) break;
      }
      addRows(fuzzyHits, 'label / Cremona / isogeny class');
      if (out.length >= limit) break;
    }
  }
  if (isCanceled()) return finishCanceled();
  maybeEmit(true, 'done');
  const final = out.slice();
  API_CACHE.searchResultCache.set(cacheKey, { items:final, ts:Date.now() });
  return final;
}
async function buildCurveDetail(id, qBound = 30) {
  let row = await loadCurveById(id);
  if (!row) {
    // v28 robustness: if a plotted point refers to a stale or mismatched id,
    // recover through the label/conductor indexes instead of leaving the side
    // panel stuck on Loading curve data.
    const point = state && state.points ? state.points.get(Number(id)) : null;
    if (point && point.l) row = await loadCurveByLabelAndConductor(point.l, point.N).catch(() => null);
  }
  if (!row) return { error: 'not found' };
  const d = enrichClientCurve(row);
  const [stGroups, isoIds, NIds, cmDiscIndex] = await Promise.all([
    loadSatoTateGroups().catch(() => ({})),
    idsForExactKey(d.iso).catch(() => []),
    idsForConductorFast(d.N).catch(() => []),
    loadCMDiscIndex().catch(() => ({ byId: new Map(), byLabel: new Map() })),
  ]);
  d.cm_disc = cmDiscForCurve(d, cmDiscIndex);
  let isoRows = [];
  let NRows = [];
  if (isoIds.length || NIds.length) {
    [isoRows, NRows] = await Promise.all([
      loadRowsByIds(isoIds, 320).catch(() => []),
      loadRowsByIds((NIds || []).slice(0, 240), 240).catch(() => []),
    ]);
  } else {
    // Older v20/v22 deployments do not have v23 exact indexes.
    const conductorRows = await loadConductorBucket(conductorBucketForN(d.N)).catch(() => []);
    isoRows = conductorRows.filter(r => r.iso === d.iso);
    NRows = conductorRows.filter(r => String(r.N) === String(d.N));
  }
  d.sato_tate = stGroups[d.st] || { label: d.st };
  d.members = isoRows.map(makeCompactMember).sort((a,b) => (Number(a.class_index||0)-Number(b.class_index||0)) || String(a.label).localeCompare(String(b.label)));
  d.same_conductor_count = NIds.length || NRows.length;
  d.same_conductor = NRows.slice(0,240).map(makeCompactMember);
  d.c_isogeny = null;
  const inv = invariantsBigFromRow(d);
  d.invariants = Object.fromEntries(Object.entries(inv).map(([k,v]) => [k, String(v)]));
  d.reduction_table = reductionData(d, 100);
  const coeffs = newformCoefficients(d, qBound, d.reduction_table);
  d.q_coefficients = Array.from({ length: qBound }, (_, i) => ({ n: i+1, a_n: coeffs[i+1] || 0 }));
  d.q_expansion = formatQExpansion(coeffs, qBound);
  return d;
}
async function apiCurve(id, qBound = 30) {
  id = Number(id);
  const cacheKey = `${id}:${qBound}`;
  if (API_CACHE.curveDetailCache.has(cacheKey)) return API_CACHE.curveDetailCache.get(cacheKey);
  if (!API_CACHE.curveDetailPromises.has(cacheKey)) {
    API_CACHE.curveDetailPromises.set(cacheKey, buildCurveDetail(id, qBound)
      .then(d => { if (d && !d.error) API_CACHE.curveDetailCache.set(cacheKey, d); return d; })
      .catch(err => { API_CACHE.curveDetailPromises.delete(cacheKey); throw err; }));
  }
  return API_CACHE.curveDetailPromises.get(cacheKey);
}
function isqrtBI(n) { if (n < 0n) return null; if (n < 2n) return n; let x0 = n, x1 = (n >> 1n) + 1n; while (x1 < x0) { x0 = x1; x1 = (x1 + n / x1) >> 1n; } return x0*x0 === n ? x0 : null; }
function formatQ(num, den = 1n) { num = BigInt(num); den = BigInt(den); if (den < 0n) { num=-num; den=-den; } const g = gcdBI(num,den); num/=g; den/=g; return den === 1n ? String(num) : `${num}/${den}`; }
function pointYFromY(a1,a3,xNum,d,Ynum) { const den = 2n*d**3n; let num = Ynum - a1*xNum*d - a3*d**3n; const g = gcdBI(absBI(num), den); return [num/g, den/g]; }
function makeRatBI(num, den = 1n) {
  let n = BigInt(num), d = BigInt(den);
  if (d === 0n) return null;
  if (d < 0n) { n = -n; d = -d; }
  const g = gcdBI(absBI(n), d);
  return { n: n / g, d: d / g };
}
function ratKey(r) { return r.d === 1n ? String(r.n) : `${r.n}/${r.d}`; }
function ratAdd(a,b){ return makeRatBI(a.n*b.d + b.n*a.d, a.d*b.d); }
function ratSub(a,b){ return makeRatBI(a.n*b.d - b.n*a.d, a.d*b.d); }
function ratMul(a,b){ return makeRatBI(a.n*b.n, a.d*b.d); }
function ratDiv(a,b){ if (b.n === 0n) return null; return makeRatBI(a.n*b.d, a.d*b.n); }
function ratNeg(a){ return { n:-a.n, d:a.d }; }
function ratEq(a,b){ return a && b && a.n === b.n && a.d === b.d; }
function ratFromInt(n){ return { n:BigInt(n), d:1n }; }
function ratSquare(a){ return makeRatBI(a.n*a.n, a.d*a.d); }
function ratCube(a){ return makeRatBI(a.n*a.n*a.n, a.d*a.d*a.d); }
function ratBitSize(a) { return Math.max(absBI(a.n).toString(2).length, a.d.toString(2).length); }
function ratIsSInteger(a, primeSet) {
  if (!a || a.d === 1n) return true;
  let d = a.d;
  for (const p of primeSet || []) {
    const P = BigInt(p);
    while (d % P === 0n) d /= P;
  }
  return d === 1n;
}
function ecPointKey(P) { return P.inf ? 'O' : `${ratKey(P.x)},${ratKey(P.y)}`; }
function ecNegPoint(curve, P) {
  if (!P || P.inf) return { inf:true };
  const a1 = ratFromInt(curve.a1), a3 = ratFromInt(curve.a3);
  return { x:P.x, y: ratSub(ratSub(ratNeg(P.y), ratMul(a1, P.x)), a3) };
}
function ecAddPoints(curve, P, Q) {
  if (!P || P.inf) return Q && !Q.inf ? { x:Q.x, y:Q.y } : { inf:true };
  if (!Q || Q.inf) return { x:P.x, y:P.y };
  const a1 = ratFromInt(curve.a1), a2 = ratFromInt(curve.a2), a3 = ratFromInt(curve.a3), a4 = ratFromInt(curve.a4), a6 = ratFromInt(curve.a6);
  let lambda, nu;
  if (ratEq(P.x, Q.x)) {
    const negP = ecNegPoint(curve, P);
    if (ratEq(negP.y, Q.y)) return { inf:true };
    const threeX2 = ratMul(ratFromInt(3), ratSquare(P.x));
    const twoA2X = ratMul(ratMul(ratFromInt(2), a2), P.x);
    const numerator = ratSub(ratAdd(ratAdd(threeX2, twoA2X), a4), ratMul(a1, P.y));
    const denominator = ratAdd(ratAdd(ratMul(ratFromInt(2), P.y), ratMul(a1, P.x)), a3);
    lambda = ratDiv(numerator, denominator);
    if (!lambda) return { inf:true };
    nu = ratSub(P.y, ratMul(lambda, P.x));
  } else {
    lambda = ratDiv(ratSub(Q.y, P.y), ratSub(Q.x, P.x));
    if (!lambda) return { inf:true };
    nu = ratDiv(ratSub(ratMul(P.y, Q.x), ratMul(Q.y, P.x)), ratSub(Q.x, P.x));
    if (!nu) return { inf:true };
  }
  const x3 = ratSub(ratSub(ratAdd(ratSquare(lambda), ratMul(a1, lambda)), a2), ratAdd(P.x, Q.x));
  const y3 = ratSub(ratSub(ratNeg(ratMul(ratAdd(lambda, a1), x3)), nu), a3);
  const R = { x:x3, y:y3 };
  // Avoid propagating accidental arithmetic overflow/mistakes; generated points
  // are only used as a supplement to the unchanged brute-force enumerator.
  return ecPointOnCurve(curve, R) ? R : { inf:true };
}
function ecPointOnCurve(curve, P) {
  if (!P || P.inf) return true;
  const a1 = ratFromInt(curve.a1), a2 = ratFromInt(curve.a2), a3 = ratFromInt(curve.a3), a4 = ratFromInt(curve.a4), a6 = ratFromInt(curve.a6);
  const lhs = ratAdd(ratAdd(ratSquare(P.y), ratMul(ratMul(a1, P.x), P.y)), ratMul(a3, P.y));
  const rhs = ratAdd(ratAdd(ratAdd(ratCube(P.x), ratMul(a2, ratSquare(P.x))), ratMul(a4, P.x)), a6);
  return ratEq(lhs, rhs);
}
function rationalPointFromIntegral(x, y) { return { x:makeRatBI(BigInt(x)), y:makeRatBI(BigInt(y)) }; }
function rationalPointFromStrings(x, y) {
  const parse = v => { const m = String(v).match(/^([+-]?\d+)(?:\/([+-]?\d+))?$/); return m ? makeRatBI(BigInt(m[1]), m[2] ? BigInt(m[2]) : 1n) : null; };
  const X = parse(x), Y = parse(y);
  return X && Y ? { x:X, y:Y } : null;
}
function generatedPointWithinCaps(P, bitCap = 96, denCap = 1000000n) {
  if (!P || P.inf) return false;
  return P.x.d <= denCap && P.y.d <= denCap && ratBitSize(P.x) <= bitCap && ratBitSize(P.y) <= bitCap;
}
function pointToIntegralOutput(P, generated = false) { return { x:ratKey(P.x), y:ratKey(P.y), generated:Boolean(generated) }; }
function pointToSIntegralOutput(P, generated = false) {
  return { x:ratKey(P.x), y:ratKey(P.y), den_x:Number(P.x.d<=9007199254740991n?P.x.d:9007199254740991n), den_y:Number(P.y.d<=9007199254740991n?P.y.d:9007199254740991n), generated:Boolean(generated) };
}
function ecSubPoints(curve, P, Q) { return ecAddPoints(curve, P, ecNegPoint(curve, Q)); }
function ecMulPoint(curve, P, n) {
  let k = BigInt(n);
  if (!P || P.inf || k === 0n) return { inf:true };
  let base = k < 0n ? ecNegPoint(curve, P) : P;
  if (k < 0n) k = -k;
  let out = { inf:true };
  while (k > 0n) {
    if (k & 1n) out = ecAddPoints(curve, out, base);
    k >>= 1n;
    if (k) base = ecAddPoints(curve, base, base);
  }
  return out;
}
function pointIsSmallTorsion(curve, P, maxOrder = 12) {
  if (!P || P.inf) return true;
  let R = { inf:true };
  for (let k = 1; k <= maxOrder; k++) {
    R = ecAddPoints(curve, R, P);
    if (R.inf) return true;
  }
  return false;
}
function pointInSmallSpan(curve, basis, P, coeffRadius = 2, maxCombos = 1400) {
  if (!basis.length || !P || P.inf) return false;
  const target = ecPointKey(P);
  const targetNeg = ecPointKey(ecNegPoint(curve, P));
  let combos = [{ inf:true }];
  for (const B of basis) {
    const multiples = [];
    for (let k = -coeffRadius; k <= coeffRadius; k++) multiples.push(ecMulPoint(curve, B, k));
    const next = [];
    for (const C of combos) {
      for (const M of multiples) {
        const R = ecAddPoints(curve, C, M);
        const key = ecPointKey(R);
        if (key === target || key === targetNeg) return true;
        next.push(R);
        if (next.length > maxCombos) break;
      }
      if (next.length > maxCombos) break;
    }
    combos = next;
    if (combos.length > maxCombos) coeffRadius = 1;
  }
  return false;
}
function addIntegralOutput(found, seen, P, generated = false) {
  if (P.x.d !== 1n || P.y.d !== 1n) return false;
  const key = `${P.x.n},${P.y.n}`;
  if (seen.has(key)) return false;
  seen.add(key);
  found.push(pointToIntegralOutput(P, generated));
  return true;
}
function addSIntegralOutput(found, seen, P, primeSet, generated = false) {
  if (!ratIsSInteger(P.x, primeSet) || !ratIsSInteger(P.y, primeSet)) return false;
  const key = `${ratKey(P.x)},${ratKey(P.y)}`;
  if (seen.has(key)) return false;
  seen.add(key);
  found.push(pointToSIntegralOutput(P, generated));
  return true;
}
function createGroupSearch(curve, acceptPoint, addOutput, opts = {}) {
  const points = [];
  const queued = [];
  const seenRational = new Set(['O']);
  const basis = [];
  const seedLimit = opts.seedLimit || 20;
  const bitCap = opts.bitCap || 112;
  const denCap = BigInt(opts.denCap || 1000000);
  const perPump = opts.perPump || 120;
  const multipleLimit = opts.multipleLimit || 10;
  const basisLimit = opts.basisLimit || Math.max(2, Math.min(7, Number(curve.rank || 0) + 3));
  const enqueue = (P, generated = true, markBasis = true) => {
    if (!generatedPointWithinCaps(P, bitCap, denCap)) return false;
    if (!ecPointOnCurve(curve, P)) return false;
    const key = ecPointKey(P);
    if (seenRational.has(key)) return false;
    seenRational.add(key);
    points.push(P);
    queued.push(P);
    if (markBasis) maybeAddBasis(P);
    if (acceptPoint(P)) addOutput(P, generated);
    return true;
  };
  const enqueueLinearCombos = (P) => {
    if (!P || P.inf) return;
    let R = { inf:true };
    for (let k = 1; k <= multipleLimit; k++) {
      R = ecAddPoints(curve, R, P);
      enqueue(R, true, false);
      enqueue(ecNegPoint(curve, R), true, false);
    }
    const peers = basis.slice(0, -1);
    for (const Q of peers) {
      for (const a of [-2, -1, 1, 2]) {
        for (const b of [-2, -1, 1, 2]) {
          if (Math.abs(a) + Math.abs(b) > 3) continue;
          const R1 = ecAddPoints(curve, ecMulPoint(curve, P, a), ecMulPoint(curve, Q, b));
          enqueue(R1, true, false);
        }
      }
    }
    if (basis.length >= 3) {
      const recent = basis.slice(Math.max(0, basis.length - 4));
      for (let i = 0; i < recent.length; i++) {
        for (let j = i + 1; j < recent.length; j++) {
          for (let k = j + 1; k < recent.length; k++) {
            enqueue(ecAddPoints(curve, ecAddPoints(curve, recent[i], recent[j]), recent[k]), true, false);
            enqueue(ecSubPoints(curve, ecAddPoints(curve, recent[i], recent[j]), recent[k]), true, false);
          }
        }
      }
    }
  };
  function maybeAddBasis(P) {
    if (!P || P.inf || basis.length >= basisLimit) return false;
    if (pointIsSmallTorsion(curve, P, 12)) return false;
    if (pointInSmallSpan(curve, basis, P, basis.length <= 3 ? 2 : 1)) return false;
    basis.push(P);
    enqueueLinearCombos(P);
    return true;
  }
  const seed = (P) => {
    if (P && !P.inf && ecPointOnCurve(curve, P)) {
      maybeAddBasis(P);
      enqueue(P, true, true);
      enqueue(ecNegPoint(curve, P), true, true);
    }
  };
  const pump = (deadline = Infinity) => {
    let steps = 0;
    while (queued.length && steps < perPump && performance.now() < deadline) {
      const P = queued.shift();
      enqueue(ecAddPoints(curve, P, P), true, true);
      const seeds = (basis.length ? basis.concat(points.slice(0, Math.max(0, seedLimit - basis.length))) : points.slice(0, seedLimit));
      for (const Q of seeds) {
        if (Q === P) continue;
        enqueue(ecAddPoints(curve, P, Q), true, true);
        enqueue(ecSubPoints(curve, P, Q), true, true);
        if (++steps >= perPump || performance.now() >= deadline) break;
      }
      steps++;
    }
  };
  return { seed, pump, size:() => seenRational.size, basisSize:() => basis.length, points:() => points.slice(), basis:() => basis.slice() };
}
const SQUARE_SIEVE_PRIMES = [7, 11, 13, 17, 19, 23, 29];
const SIEVE_YIELD_EVERY = 4096;
function modSmallBI(n, p) { let r = Number(BigInt(n) % BigInt(p)); if (r < 0) r += p; return r; }
function modSmallNumber(n, p) { let r = n % p; if (r < 0) r += p; return r; }
function squareResidueMask(p) {
  const key = `sq:${p}`;
  let mask = API_CACHE.squareSieveCache.get(key);
  if (mask) return mask;
  mask = Array(p).fill(false);
  for (let y = 0; y < p; y++) mask[(y * y) % p] = true;
  API_CACHE.squareSieveCache.set(key, mask);
  return mask;
}
function makeCubicSquareSieve(cacheKey, b2, b4, b6, d = 1n) {
  const fullKey = `cubic:${cacheKey}:${String(d)}`;
  const cached = API_CACHE.squareSieveCache.get(fullKey);
  if (cached) return cached;
  const sieve = [];
  for (const p of SQUARE_SIEVE_PRIMES) {
    const sq = squareResidueMask(p);
    const B2 = modSmallBI(b2, p), B4 = modSmallBI(b4, p), B6 = modSmallBI(b6, p), D = modSmallBI(d, p);
    const D2 = (D * D) % p, D4 = (D2 * D2) % p, D6 = (D4 * D2) % p;
    const allowed = Array(p).fill(false);
    let pass = 0;
    for (let x = 0; x < p; x++) {
      const x2 = (x * x) % p;
      const x3 = (x2 * x) % p;
      const val = (4 * x3 + B2 * x2 * D2 + 2 * B4 * x * D4 + B6 * D6) % p;
      if (sq[val]) { allowed[x] = true; pass++; }
    }
    // Keep only genuinely selective congruences; a bad reduction prime can be
    // unhelpful while still safe, so skip it to reduce per-candidate overhead.
    if (pass > 0 && pass < p) sieve.push({ p, allowed, pass });
  }
  sieve.sort((a,b) => (a.pass / a.p) - (b.pass / b.p));
  API_CACHE.squareSieveCache.set(fullKey, sieve);
  return sieve;
}
function passesCubicSquareSieve(n, sieve) {
  for (const test of sieve || []) if (!test.allowed[modSmallNumber(n, test.p)]) return false;
  return true;
}
function rationalSeedCaps(mode, phase) {
  const baseD = mode === 'S-integral' ? 18 : 14;
  const baseH = mode === 'S-integral' ? 95 : 120;
  return {
    maxD: Math.min(96, Math.ceil(baseD * Math.pow(1.38, phase))),
    xHeight: Math.min(2200, Math.ceil(baseH * Math.pow(1.32, phase)))
  };
}
function scheduleRationalSeedSegments(seed, maxD, xHeight) {
  seed.maxD = Math.max(seed.maxD || 0, maxD);
  seed.xHeight = Math.max(seed.xHeight || 0, xHeight);
  for (let d0 = 1; d0 <= maxD; d0++) {
    const nextBound = Math.floor(xHeight * d0 * d0);
    const prevBound = seed.scheduledMBounds.get(d0) || -1;
    if (nextBound <= prevBound) continue;
    if (prevBound < 0) {
      seed.segments.push({ d0, lo:-nextBound, hi:nextBound, m:-nextBound });
    } else {
      seed.segments.push({ d0, lo:-nextBound, hi:-prevBound - 1, m:-nextBound });
      seed.segments.push({ d0, lo:prevBound + 1, hi:nextBound, m:prevBound + 1 });
    }
    seed.scheduledMBounds.set(d0, nextBound);
  }
  seed.segments.sort((a,b) => (a.d0 - b.d0) || (Math.abs(a.lo) - Math.abs(b.lo)) || (a.lo - b.lo));
}
function createRationalSeedState(curve, inv, mode) {
  const seed = {
    mode, curveId:Number(curve.id), phase:0, segments:[], segIndex:0, scheduledMBounds:new Map(),
    maxD:0, xHeight:0, checked:0, hits:0
  };
  const caps = rationalSeedCaps(mode, 0);
  scheduleRationalSeedSegments(seed, caps.maxD, caps.xHeight);
  return seed;
}
function pumpRationalSeedSearch(state, deadline = Infinity, shouldCancel = null) {
  const seed = state.rationalSeed;
  if (!seed || !state.groupSearch) return false;
  const { a1, a3, b2, b4, b6 } = state;
  let local = 0;
  while (performance.now() < deadline && local < 1536) {
    if (seed.segIndex >= seed.segments.length) {
      seed.phase += 1;
      const caps = rationalSeedCaps(state.kind || seed.mode, seed.phase);
      scheduleRationalSeedSegments(seed, caps.maxD, caps.xHeight);
      if (seed.segIndex >= seed.segments.length) return false;
    }
    const seg = seed.segments[seed.segIndex];
    const d0 = seg.d0, d = BigInt(d0), d2 = d*d, d4 = d2*d2, d6 = d4*d2;
    const sieve = makeCubicSquareSieve(`R:${state.curveId}`, b2, b4, b6, d);
    for (let m = seg.m; m <= seg.hi; m++) {
      seg.m = m + 1;
      local++; seed.checked++;
      if (d0 > 1 && gcdBI(absBI(BigInt(m)), d) !== 1n) {
        if (local >= 1536 || performance.now() >= deadline) break;
        continue;
      }
      if (passesCubicSquareSieve(m, sieve)) {
        const M = BigInt(m);
        const rhs = 4n*M*M*M + b2*M*M*d2 + 2n*b4*M*d4 + b6*d6;
        const Yroot = isqrtBI(rhs);
        if (Yroot != null) {
          const Ys = Yroot === 0n ? [0n] : [Yroot, -Yroot];
          for (const Y of Ys) {
            let xNum = M, xDen = d2; const gx = gcdBI(absBI(xNum), xDen); xNum /= gx; xDen /= gx;
            const [yNum, yDen] = pointYFromY(a1, a3, M, d, Y);
            const P = { x:makeRatBI(xNum, xDen), y:makeRatBI(yNum, yDen) };
            if (ecPointOnCurve(state.curve, P)) { seed.hits++; state.groupSearch.seed(P); }
          }
        }
      }
      if (shouldCancel && shouldCancel()) return true;
      if (local >= 1536 || performance.now() >= deadline) break;
    }
    if (seg.m > seg.hi) seed.segIndex++;
    if (shouldCancel && shouldCancel()) return true;
  }
  return false;
}
function searchCacheKey(curveId, S) { return `${Number(curveId)}|${Math.max(1, Math.abs(Math.floor(Number(S)||1)))}`; }
function stripGenerated(points, limit = Infinity) { return points.slice(0, limit).map(({generated, ...p}) => p); }
function finishTimedStatus(state, startedAt, timedOut) {
  state.runs = (state.runs || 0) + 1;
  state.total_elapsed_ms = (state.total_elapsed_ms || 0) + (performance.now() - startedAt);
  state.last_run_elapsed_ms = performance.now() - startedAt;
  state.timed_out = Boolean(timedOut);
  state.lastTouched = Date.now();
}
function createIntegralSearchState(curve) {
  const inv = invariantsBigFromRow(curve);
  const found = [], seen = new Set();
  const target = Number(curve.pts || 0);
  const state = {
    kind:'integral', curveId:Number(curve.id), curve, inv,
    a1:toBI(curve.a1), a3:toBI(curve.a3), b2:inv.b2, b4:inv.b4, b6:inv.b6,
    target, found, seen, bound:256, x:-256, searched:-1, complete:false,
    runs:0, total_elapsed_ms:0, last_run_elapsed_ms:0, timed_out:false,
    sieve:makeCubicSquareSieve(`I:${curve.id}`, inv.b2, inv.b4, inv.b6, 1n),
    rationalSeed:null
  };
  state.groupSearch = createGroupSearch(curve,
    P => P.x.d === 1n && P.y.d === 1n,
    (P, generated) => addIntegralOutput(found, seen, P, generated),
    { seedLimit: 28, bitCap: 132, denCap: 20000000n, perPump: 180, multipleLimit: 18, basisLimit: Math.max(2, Math.min(8, Number(curve.rank || 0) + 4)) });
  state.rationalSeed = createRationalSeedState(curve, inv, 'integral');
  return state;
}
function getOrCreateIntegralSearchState(curve) {
  const key = searchCacheKey(curve.id, 1);
  let state = API_CACHE.pointSearches.get(key);
  if (!state || state.kind !== 'integral') {
    state = createIntegralSearchState(curve);
    API_CACHE.pointSearches.set(key, state);
  }
  return state;
}
function importIntegralSearchIntoSIntegralState(sState, integralState, opts = {}) {
  if (!sState || !integralState || sState.kind !== 'S-integral') return { points:0, seeds:0 };
  const seedCap = opts.seedCap ?? 96;
  let importedPoints = 0;
  let importedSeeds = 0;
  for (const row of integralState.found || []) {
    const P = rationalPointFromStrings(row.x, row.y);
    if (!P || !ecPointOnCurve(sState.curve, P)) continue;
    if (addSIntegralOutput(sState.found, sState.seen, P, sState.primeSet, Boolean(row.generated))) importedPoints++;
    sState.groupSearch.seed(P);
  }
  const basis = typeof integralState.groupSearch?.basis === 'function' ? integralState.groupSearch.basis() : [];
  const points = typeof integralState.groupSearch?.points === 'function' ? integralState.groupSearch.points() : [];
  for (const P of basis.concat(points).slice(0, seedCap)) {
    if (!P || P.inf || !ecPointOnCurve(sState.curve, P)) continue;
    sState.groupSearch.seed(P);
    importedSeeds++;
  }
  sState.integralBridge = {
    imported_integral_points: (sState.integralBridge?.imported_integral_points || 0) + importedPoints,
    imported_integral_seeds: (sState.integralBridge?.imported_integral_seeds || 0) + importedSeeds,
    integral_cache_runs: integralState.runs || 0,
    integral_complete: Boolean(integralState.complete),
    integral_searched_abs_x_up_to: Math.max(0, integralState.searched || 0),
    integral_found_count: (integralState.found || []).length,
    last_import_ms: Date.now()
  };
  return { points:importedPoints, seeds:importedSeeds };
}
function trimSIntegralIntegralOverlap(sState, integralState) {
  if (!sState || sState.kind !== 'S-integral' || !integralState) return 0;
  const cover = Math.max(0, Number(integralState.searched || 0));
  const oldCover = Math.max(0, Number(sState.integralOverlapCoveredX || 0));
  if (cover <= oldCover) return 0;
  let skipped = 0;
  const kept = [];
  const sourceSegments = (sState.segments || []).slice(sState.segIndex || 0);
  for (const seg of sourceSegments) {
    if (seg.d0 !== 1) { kept.push(seg); continue; }
    const lo = Math.max(seg.m, seg.lo);
    const hi = seg.hi;
    if (lo > hi) continue;
    if (hi < -cover || lo > cover) { kept.push({ ...seg, lo, m:lo }); continue; }
    const leftHi = Math.min(hi, -cover - 1);
    if (lo <= leftHi) kept.push({ d0:1, lo, hi:leftHi, m:lo });
    const rightLo = Math.max(lo, cover + 1);
    if (rightLo <= hi) kept.push({ d0:1, lo:rightLo, hi, m:rightLo });
    skipped += Math.max(0, Math.min(hi, cover) - Math.max(lo, -cover) + 1);
  }
  sState.segments = kept;
  sState.segIndex = 0;
  sState.integralOverlapCoveredX = cover;
  sState.integralOverlapSkipped = (sState.integralOverlapSkipped || 0) + skipped;
  return skipped;
}
function sIntegralCaps(primeCount, phase) {
  const baseMaxD = primeCount <= 2 ? 80 : 50;
  const baseX = 1200;
  return {
    maxD: Math.min(500, Math.ceil(baseMaxD * Math.pow(1.45, phase))),
    xBound: Math.min(20000, Math.ceil(baseX * Math.pow(1.35, phase)))
  };
}
function scheduleSIntegralSegments(state, maxD, xBound) {
  state.maxD = Math.max(state.maxD || 0, maxD);
  state.xBound = Math.max(state.xBound || 0, xBound);
  const denominators = smoothNumbersFromPrimes(state.primes, maxD);
  for (const d0 of denominators) {
    const nextBound = Math.floor(xBound * d0 * d0);
    const prevBound = state.scheduledMBounds.get(d0) || -1;
    if (nextBound <= prevBound) continue;
    if (prevBound < 0) {
      state.segments.push({ d0, lo:-nextBound, hi:nextBound, m:-nextBound });
    } else {
      state.segments.push({ d0, lo:-nextBound, hi:-prevBound - 1, m:-nextBound });
      state.segments.push({ d0, lo:prevBound + 1, hi:nextBound, m:prevBound + 1 });
    }
    state.scheduledMBounds.set(d0, nextBound);
  }
}
function createSIntegralSearchState(curve, S, primes) {
  const inv = invariantsBigFromRow(curve);
  const found = [], seen = new Set();
  const state = {
    kind:'S-integral', curveId:Number(curve.id), curve, S, primes, primeSet:new Set(primes), inv,
    a1:toBI(curve.a1), a3:toBI(curve.a3), b2:inv.b2, b4:inv.b4, b6:inv.b6,
    found, seen, checked:0, phase:0, segments:[], segIndex:0, scheduledMBounds:new Map(),
    maxD:0, xBound:0, complete:false, runs:0, total_elapsed_ms:0, last_run_elapsed_ms:0, timed_out:false,
    rationalSeed:null
  };
  const caps = sIntegralCaps(primes.length, 0);
  scheduleSIntegralSegments(state, caps.maxD, caps.xBound);
  state.groupSearch = createGroupSearch(curve,
    P => ratIsSInteger(P.x, state.primeSet) && ratIsSInteger(P.y, state.primeSet),
    (P, generated) => addSIntegralOutput(found, seen, P, state.primeSet, generated),
    { seedLimit: 34, bitCap: 128, denCap: 25000000n, perPump: 200, multipleLimit: 20, basisLimit: Math.max(2, Math.min(9, Number(curve.rank || 0) + 5)) });
  state.rationalSeed = createRationalSeedState(curve, inv, 'S-integral');
  return state;
}
async function runIntegralSearchState(state, budgetMs, shouldCancel = null) {
  const startedAt = performance.now();
  const deadline = startedAt + budgetMs;
  let timedOut = false;
  const { a1, a3, b2, b4, b6 } = state;
  if (state.target === 0) { state.complete = true; finishTimedStatus(state, startedAt, false); return state; }
  while (state.bound <= 2000000) {
    for (let x = state.x; x <= state.bound; x++) {
      state.x = x + 1;
      if (Math.abs(x) <= state.searched) continue;
      const shouldYield = (Math.abs(x) % SIEVE_YIELD_EVERY) === 0;
      if (passesCubicSquareSieve(x, state.sieve)) {
        const X = BigInt(x); const rhs = 4n*X*X*X + b2*X*X + 2n*b4*X + b6; const root = isqrtBI(rhs);
        if (root != null) {
          const Ys = root === 0n ? [0n] : [root, -root];
          for (const Y of Ys) {
            const yn = Y - a1*X - a3; if (yn % 2n !== 0n) continue;
            const y = yn/2n;
            const P = rationalPointFromIntegral(X, y);
            addIntegralOutput(state.found, state.seen, P, false);
            state.groupSearch.seed(P);
          }
        }
      }
      if (shouldYield) {
        state.groupSearch.pump(deadline);
        if (pumpRationalSeedSearch(state, deadline, shouldCancel)) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
        if (shouldCancel && shouldCancel()) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
        if (performance.now() > deadline) { timedOut = true; break; }
        await idleYield(8);
      }
      if (state.target && state.found.length >= state.target) { state.complete = true; break; }
    }
    state.groupSearch.pump(deadline);
    if (!state.complete && pumpRationalSeedSearch(state, deadline, shouldCancel)) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
    if (state.complete || timedOut) break;
    state.searched = state.bound;
    state.bound *= 2;
    state.x = -state.bound;
  }
  if (state.bound > 2000000 && state.target && state.found.length >= state.target) state.complete = true;
  if (performance.now() > deadline) timedOut = true;
  finishTimedStatus(state, startedAt, timedOut);
  return state;
}
async function runSIntegralSearchState(state, budgetMs, shouldCancel = null) {
  const startedAt = performance.now();
  const deadline = startedAt + budgetMs;
  let timedOut = false;
  const { a1, a3, b2, b4, b6 } = state;
  while (performance.now() <= deadline) {
    if (state.segIndex >= state.segments.length) {
      state.groupSearch.pump(deadline);
      if (pumpRationalSeedSearch(state, deadline, shouldCancel)) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
      state.phase += 1;
      const caps = sIntegralCaps(state.primes.length, state.phase);
      scheduleSIntegralSegments(state, caps.maxD, caps.xBound);
      if (state.segIndex >= state.segments.length) break;
    }
    const seg = state.segments[state.segIndex];
    const d0 = seg.d0, d = BigInt(d0), d2=d*d, d4=d**4n, d6=d**6n;
    const sieve = makeCubicSquareSieve(`S:${state.curveId}`, b2, b4, b6, d);
    for (let m = seg.m; m <= seg.hi; m++) {
      seg.m = m + 1;
      state.checked++;
      if (passesCubicSquareSieve(m, sieve)) {
        const M=BigInt(m); const rhs=4n*M*M*M + b2*M*M*d2 + 2n*b4*M*d4 + b6*d6; const Yroot=isqrtBI(rhs);
        if (Yroot != null) {
          const Ys=Yroot===0n?[0n]:[Yroot,-Yroot];
          for (const Y of Ys) {
            let xNum=M, xDen=d2; const gx=gcdBI(absBI(xNum),xDen); xNum/=gx; xDen/=gx;
            const [yNum,yDen]=pointYFromY(a1,a3,M,d,Y);
            const P = { x:makeRatBI(xNum,xDen), y:makeRatBI(yNum,yDen) };
            addSIntegralOutput(state.found, state.seen, P, state.primeSet, false);
            state.groupSearch.seed(P);
          }
        }
      }
      if (state.checked % SIEVE_YIELD_EVERY === 0) {
        state.groupSearch.pump(deadline);
        if (pumpRationalSeedSearch(state, deadline, shouldCancel)) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
        if (shouldCancel && shouldCancel()) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
        if (performance.now() > deadline) { timedOut = true; break; }
        await idleYield(8);
      }
    }
    state.groupSearch.pump(deadline);
    if (pumpRationalSeedSearch(state, deadline, shouldCancel)) { finishTimedStatus(state, startedAt, false); state.canceled = true; return state; }
    if (timedOut) break;
    if (seg.m > seg.hi) state.segIndex++;
  }
  if (performance.now() > deadline) timedOut = true;
  finishTimedStatus(state, startedAt, timedOut);
  return state;
}
async function apiIntegralPoints(curveId, timeout = 2.25, shouldCancel = null) {
  if (shouldCancel && shouldCancel()) return { canceled:true, points:[], count:0, elapsed_ms:0, note:'canceled because another curve was selected.' };
  const curve = await loadCurveById(curveId); if (!curve) return { error:'curve not found' };
  if (shouldCancel && shouldCancel()) return { canceled:true, points:[], count:0, elapsed_ms:0, note:'canceled because another curve was selected.' };
  const state = getOrCreateIntegralSearchState(curve);
  const requestedMs = Math.max(0.05, Number(timeout) || 2.25) * 1000;
  const budgetMs = state.runs && !state.complete ? requestedMs * 0.5 : requestedMs;
  await runIntegralSearchState(state, budgetMs, shouldCancel);
  state.found.sort((a,b) => (a.x.length-b.x.length) || String(a.x).localeCompare(String(b.x)) || String(a.y).localeCompare(String(b.y)));
  const generated = state.found.filter(p => p.generated).length;
  if (state.target === 0) {
    return { curve_id: curveId, label: curve.label, mode:'integral', target_count:0, points:[], count:0, generated_count:0, searched_abs_x_up_to:0, complete:true, timed_out:false, elapsed_ms:Number(state.last_run_elapsed_ms.toFixed(1)), total_elapsed_ms:Number(state.total_elapsed_ms.toFixed(1)), cache_runs:state.runs, continued:state.runs>1, note:'stored integral-point count is zero.' };
  }
  return { curve_id:curveId, label:curve.label, mode:'integral', target_count:state.target, points:stripGenerated(state.found), count:state.found.length, generated_count:generated, searched_abs_x_up_to:Math.max(0,state.searched), complete:Boolean(state.complete), timed_out:state.timed_out, elapsed_ms:Number(state.last_run_elapsed_ms.toFixed(1)), total_elapsed_ms:Number(state.total_elapsed_ms.toFixed(1)), cache_runs:state.runs, continued:state.runs>1, basis_rank_guess:state.groupSearch.basisSize(), rational_points_seen:state.groupSearch.size(), low_height_rational_candidates_checked:state.rationalSeed?.checked || 0, low_height_rational_hits:state.rationalSeed?.hits || 0, note:'complete=true means the stored integral-point count was reached; otherwise the list is a bounded resumable search result. v35 uses congruence square sieves plus a heuristic Mordell-Weil subgroup search generated from discovered integral and low-height rational points; the brute-force x-search is retained and the cache is shared with S-integral searches.' };
}
function primeFactorsSmall(n) { return [...factorIntSmall(n).keys()]; }
function smoothNumbersFromPrimes(primes, limit) { const vals = new Set([1]); for (const p of primes) { const current = [...vals].sort((a,b)=>a-b); let mul = p; while (mul <= limit) { for (const v of current) { const nv = v*mul; if (nv <= limit) vals.add(nv); } mul *= p; } } return [...vals].sort((a,b)=>a-b); }
async function apiSIntegralPoints(curveId, S, timeout = 7.5, shouldCancel = null) {
  if (shouldCancel && shouldCancel()) return { canceled:true, points:[], count:0, elapsed_ms:0, note:'canceled because another curve was selected.' };
  const curve = await loadCurveById(curveId); if (!curve) return { error:'curve not found' };
  if (shouldCancel && shouldCancel()) return { canceled:true, points:[], count:0, elapsed_ms:0, note:'canceled because another curve was selected.' };
  const Sabs = Math.max(1, Math.abs(Math.floor(Number(S)||1)));
  const primes = primeFactorsSmall(Sabs); if (!primes.length) return apiIntegralPoints(curveId, timeout, shouldCancel);
  const key = searchCacheKey(curveId, Sabs);
  let state = API_CACHE.pointSearches.get(key);
  if (!state || state.kind !== 'S-integral' || state.S !== Sabs) {
    state = createSIntegralSearchState(curve, Sabs, primes);
    API_CACHE.pointSearches.set(key, state);
  }

  // v35: every S-integral run is explicitly coupled to the ordinary integral
  // search cache.  Integral points are S-integral for every S, and integral
  // low-height rational/group seeds are also valuable Mordell-Weil subgroup
  // seeds.  Import first (instant reuse of previous integral searches), spend a
  // bounded slice of this click continuing the integral frontier if incomplete,
  // import again, then continue the S-denominator frontier.  Each state keeps
  // its own non-overlapping frontier, so repeated Compute clicks advance rather
  // than rescan the same denominator/x ranges.
  const integralState = getOrCreateIntegralSearchState(curve);
  importIntegralSearchIntoSIntegralState(state, integralState, { seedCap: 128 });
  trimSIntegralIntegralOverlap(state, integralState);

  const requestedMs = Math.max(0.05, Number(timeout)||7.5)*1000;
  const continued = state.runs > 0 && !state.complete;
  const budgetMs = continued ? requestedMs * 0.5 : requestedMs;
  let integralBudgetMs = 0;
  if (!integralState.complete) {
    integralBudgetMs = Math.max(80, Math.min(budgetMs * 0.42, budgetMs - 50));
  } else if ((integralState.rationalSeed && integralState.rationalSeed.segIndex < integralState.rationalSeed.segments.length) || (typeof integralState.groupSearch?.size === 'function' && integralState.groupSearch.size() > 1)) {
    integralBudgetMs = Math.max(35, Math.min(budgetMs * 0.12, 180));
  }
  let remainingBudgetMs = Math.max(50, budgetMs - integralBudgetMs);
  if (integralBudgetMs > 0) {
    await runIntegralSearchState(integralState, integralBudgetMs, shouldCancel);
    if (shouldCancel && shouldCancel()) return { canceled:true, points:[], count:0, elapsed_ms:0, note:'canceled because another curve was selected.' };
    importIntegralSearchIntoSIntegralState(state, integralState, { seedCap: 160 });
    trimSIntegralIntegralOverlap(state, integralState);
  }
  await runSIntegralSearchState(state, remainingBudgetMs, shouldCancel);
  importIntegralSearchIntoSIntegralState(state, integralState, { seedCap: 160 });
  state.groupSearch.pump(performance.now() + 25);
  state.found.sort((a,b) => (Math.max(a.den_x||1,a.den_y||1)-Math.max(b.den_x||1,b.den_y||1)) || (a.x.length-b.x.length) || String(a.x).localeCompare(String(b.x)) || String(a.y).localeCompare(String(b.y)));
  const generated = state.found.filter(p => p.generated).length;
  const maxDen = Math.max(1, ...state.scheduledMBounds.keys());
  const bridge = state.integralBridge || {};
  return { curve_id:curveId, label:curve.label, mode:'S-integral', S:state.S, S_primes:primes, points:stripGenerated(state.found, 500), count:state.found.length, generated_count:generated, returned_count:Math.min(state.found.length,500), denominators_checked:state.scheduledMBounds.size, max_denominator_checked:maxDen, x_height_bound:state.xBound, candidate_triples_checked:state.checked, timed_out:state.timed_out, complete:false, elapsed_ms:Number(state.last_run_elapsed_ms.toFixed(1)), total_elapsed_ms:Number(state.total_elapsed_ms.toFixed(1)), cache_runs:state.runs, continued:state.runs>1, resumed_extra_budget:continued, integral_bridge:bridge, integral_budget_ms:Number(integralBudgetMs.toFixed(1)), s_integral_budget_ms:Number(remainingBudgetMs.toFixed(1)), integral_cache_runs:bridge.integral_cache_runs || integralState.runs || 0, integral_found_count:bridge.integral_found_count || integralState.found.length || 0, imported_integral_points:bridge.imported_integral_points || 0, integral_overlap_skipped_x_candidates:state.integralOverlapSkipped || 0, contains_integral_search_results:true, basis_rank_guess:state.groupSearch.basisSize(), rational_points_seen:state.groupSearch.size(), low_height_rational_candidates_checked:state.rationalSeed?.checked || 0, low_height_rational_hits:state.rationalSeed?.hits || 0, note:'S-integral search is bounded by time and height; complete=false means additional points may be missing. v35 explicitly reuses and continues the integral-point cache on every S-integral Compute click, imports all currently found integral points because they are automatically S-integral, and shares integral low-height rational/group seeds with the S-integral search. Repeated clicks continue the cached integral and S-denominator frontiers without deliberately repeating already scheduled ranges.' };
}
async function apiPoints(curveId, S = 1, timeout = 1.0, shouldCancel = null) {
  const effectiveTimeout = Math.max(0.05, Number(timeout) || 1.0) * INTEGRAL_SEARCH_TIME_MULTIPLIER;
  const Sabs = Math.max(1, Math.abs(Math.floor(Number(S)||1)));
  return Sabs === 1 ? apiIntegralPoints(curveId, effectiveTimeout, shouldCancel) : apiSIntegralPoints(curveId, Sabs, effectiveTimeout, shouldCancel);
}
if (typeof window !== 'undefined') window.EC_ATLAS_API = { apiMeta, apiTop, apiTile, apiSearch, apiCurve, apiHover, apiPoints, loadCurveDatabase, version: EC_ATLAS_VERSION };


const TAU = Math.PI * 2;
const RAD = Math.PI / 180;
const MIN_ALT = 0.15 * RAD;
const MAX_ALT = 89.85 * RAD;

const state = {
  canvas: null,
  ctx: null,
  dpr: window.devicePixelRatio || 1,
  viewW: 0,
  viewH: 0,
  viewF: null,
  viewRight: null,
  viewUp: null,
  meta: null,
  points: new Map(),
  pointIdsByTile: new Map(),
  topPointIds: new Set(),
  activeTileIds: new Set(),
  visibleTileCandidates: [],
  loadedTiles: new Set(),
  loadingTiles: new Set(),
  rendered: [],
  renderPool: [],
  lastRenderKey: '',
  lastSelectionKey: '',
  selectedProjection: null,
  selectionEl: null,
  detailIds: new Set(),
  detailSeparation: new Map(),
  hover: null,
  hoverMouse: null,
  hoverCandidate: null,
  hoverTimer: null,
  hoverStillMs: 500,
  hoverInfoCache: new Map(),
  hoverInfoPending: new Set(),
  selected: null,
  az: 0,
  alt: 89.7 * RAD,
  fov: 30 * RAD,
  initialFov: 30 * RAD,
  visibleLevel: 2,
  dragging: false,
  drag: null,
  tooltip: null,
  raf: 0,
  dirty: true,
  detailMembers: new Map(),
  travel: null,
  lastTileCheck: 0,
  pointerDown: null,
  pointerDownTarget: null,
  clickMoved: false,
  preloadIndex: 0,
  preloadingAll: false,
  selectedPulseUntil: 0,
  currentDetailId: null,
  activePointers: new Map(),
  pinch: null,
  interactingUntil: 0,
  perfMode: false,
  bgCache: null,
  bgCacheKey: '',
  wasInteracting: false,
  detailLoadToken: 0,
  detailWheelUntil: 0,
  atlasWheelSuppressUntil: 0,
};

const VISIBLE_LEVELS = [500, 950, 1800, 3500, 5000];
const $ = id => document.getElementById(id);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const mod = (x, m) => ((x % m) + m) % m;
const screen = () => {
  if (state.viewW && state.viewH) return { w: state.viewW, h: state.viewH };
  const r = state.canvas.getBoundingClientRect();
  return { w: r.width, h: r.height };
};

function requestRender() {
  state.dirty = true;
  if (!state.raf) state.raf = requestAnimationFrame(frame);
}

function frame(ts) {
  state.raf = 0;
  const traveling = updateTravel(ts);
  const animating = traveling || state.dragging || !!state.pinch || (ts < (state.interactingUntil || 0));
  const justSettled = state.wasInteracting && !animating;
  if (animating) state.wasInteracting = true;
  if (justSettled) {
    state.wasInteracting = false;
    state.renderPool = [];
    state.lastRenderKey = '';
    updateVisibleTiles(true);
    state.dirty = true;
  }
  if (state.dirty || animating || justSettled) {
    render(ts);
    state.dirty = false;
  }
  if (animating || state.dirty) state.raf = requestAnimationFrame(frame);
}

function sphToVec(az, alt) {
  const c = Math.cos(alt);
  return [c * Math.cos(az), c * Math.sin(az), Math.sin(alt)];
}
function vecToAzAlt(v) {
  const x = v[0], y = v[1], z = clamp(v[2], -1, 1);
  return [mod(Math.atan2(y, x), TAU), Math.asin(z)];
}
function dot(a, b) { return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }
function cross(a, b) { return [a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]]; }
function norm(v) {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0]/n, v[1]/n, v[2]/n];
}
function rotateVec(v, axis, ang) {
  const a = norm(axis);
  const c = Math.cos(ang), s = Math.sin(ang);
  const av = dot(a, v);
  const axv = cross(a, v);
  return [
    v[0]*c + axv[0]*s + a[0]*av*(1-c),
    v[1]*c + axv[1]*s + a[1]*av*(1-c),
    v[2]*c + axv[2]*s + a[2]*av*(1-c),
  ];
}
function tangentPart(v, f) {
  const d = dot(v, f);
  return [v[0] - d*f[0], v[1] - d*f[1], v[2] - d*f[2]];
}
function installCamera(f, preferredRight = null, preferredUp = null) {
  f = norm(f);
  let [az, alt] = vecToAzAlt(f);
  if (alt > MAX_ALT || alt < MIN_ALT) {
    if (alt > MAX_ALT && Math.cos(alt) < 0.003) az = state.az;
    alt = clamp(alt, MIN_ALT, MAX_ALT);
    f = sphToVec(az, alt);
  }

  let right = preferredRight ? tangentPart(preferredRight, f) : null;
  if (!right || Math.hypot(right[0], right[1], right[2]) < 1e-8) {
    right = state.viewRight ? tangentPart(state.viewRight, f) : null;
  }
  if (!right || Math.hypot(right[0], right[1], right[2]) < 1e-8) {
    right = cross(f, [0, 0, 1]);
  }
  if (Math.hypot(right[0], right[1], right[2]) < 1e-8 && preferredUp) {
    right = cross(preferredUp, f);
  }
  if (Math.hypot(right[0], right[1], right[2]) < 1e-8) right = [0, -1, 0];
  right = norm(right);
  let up = norm(cross(right, f));
  // Keep the handedness stable if a previously rotated up vector was supplied.
  if (preferredUp && dot(up, preferredUp) < 0) {
    right = [-right[0], -right[1], -right[2]];
    up = [-up[0], -up[1], -up[2]];
  }

  state.viewF = f;
  state.viewRight = right;
  state.viewUp = up;
  const [rawAz, rawAlt] = vecToAzAlt(f);
  state.alt = clamp(rawAlt, MIN_ALT, MAX_ALT);
  // At the zenith azimuth is numerically undefined. Preserve the previous
  // azimuth there instead of letting tiny x/y noise become a large angle jump.
  if (Math.cos(state.alt) > 0.003) state.az = rawAz;
  else state.az = mod(state.az, TAU);
}
function setViewAzAlt(az, alt) {
  installCamera(sphToVec(mod(az, TAU), clamp(alt, MIN_ALT, MAX_ALT)), state.viewRight, state.viewUp);
}
function setViewFromVectors(f, right = null, up = null) {
  installCamera(f, right, up);
}
function camera() {
  if (!state.viewF || !state.viewRight || !state.viewUp) setViewAzAlt(state.az, state.alt);
  return { f: state.viewF, right: state.viewRight, up: state.viewUp };
}

function projectionScale() {
  const { w, h } = screen();
  return (Math.max(w, h) * 0.60) / Math.max(1e-6, Math.tan(state.fov / 3.3));
}

function projectAzAlt(az, alt) {
  const { w, h } = screen();
  const { f, right, up } = camera();
  const v = sphToVec(az, alt);
  const z = dot(v, f);
  if (z <= 0) return null;
  const xr = dot(v, right);
  const yu = dot(v, up);
  const denom = Math.max(1e-9, 1 + z);
  const rho = Math.sqrt(xr*xr + yu*yu) / denom;
  const scale = projectionScale();
  const x = w / 2 + xr / denom * scale;
  const y = h / 2 - yu / denom * scale;
  if (x < -260 || x > w + 260 || y < -260 || y > h + 260) return null;
  return { x, y, z, rho };
}

function screenToWorldFromCamera(sx, sy, cam) {
  const { w, h } = screen();
  const scale = projectionScale();
  let u = (sx - w / 2) / scale;
  let v = (h / 2 - sy) / scale;
  let rho2 = u*u + v*v;
  if (rho2 > 0.9801) {
    const fac = 0.99 / Math.sqrt(rho2);
    u *= fac; v *= fac; rho2 = u*u + v*v;
  }
  const denom = 1 + rho2;
  const xc = 2*u / denom;
  const yc = 2*v / denom;
  const zc = (1 - rho2) / denom;
  const { f, right, up } = cam;
  return norm([xc*right[0] + yc*up[0] + zc*f[0], xc*right[1] + yc*up[1] + zc*f[1], xc*right[2] + yc*up[2] + zc*f[2]]);
}
function screenToWorld(sx, sy) {
  return screenToWorldFromCamera(sx, sy, camera());
}

function angularDistance(a, b) {
  return Math.acos(clamp(dot(a, b), -1, 1));
}
function zoomLevel() {
  return Math.max(0, Math.log2(state.initialFov / state.fov));
}
function dynamicMax() {
  const z = zoomLevel();
  const base = VISIBLE_LEVELS[state.visibleLevel] || 1800;
  // v26: the selected visible-star budget is the dominant budget.  Zooming
  // adds only a small extra allowance so the default level really behaves
  // like ~1800 stars instead of growing into a heavy draw path immediately.
  const zoomExtra = z > 0.25 ? Math.min(520, 95 * Math.pow(z, 0.82)) : 0;
  return Math.round(base + zoomExtra);
}
function smoothScoreFromLargestPrime(lp) {
  return clamp((Math.log(10007) - Math.log(Math.max(lp, 2))) / (Math.log(10007) - Math.log(2)), 0, 1);
}
const STAR_STOPS = Object.freeze([
  [0.00, [86, 120, 255]],
  [0.10, [108, 171, 255]],
  [0.24, [146, 221, 255]],
  [0.40, [244, 249, 255]],
  [0.58, [255, 242, 176]],
  [0.76, [255, 174, 92]],
  [0.90, [255, 98, 112]],
  [1.00, [224, 92, 255]],
]);
const TORSION_STYLE_MAP = Object.freeze({
  '0':   { rays: 4, points: 0, product: false, innerPoints: 0, rot: 0 },
  'n2':  { rays: 4, points: 4, product: false, innerPoints: 0, rot: Math.PI/4 },
  'n3':  { rays: 6, points: 3, product: false, innerPoints: 0, rot: 0 },
  'n4':  { rays: 8, points: 4, product: false, innerPoints: 0, rot: 0 },
  'n5':  { rays: 5, points: 5, product: false, innerPoints: 0, rot: 0 },
  'n6':  { rays: 6, points: 6, product: false, innerPoints: 0, rot: 0 },
  'n7':  { rays: 7, points: 7, product: false, innerPoints: 0, rot: 0 },
  'n8':  { rays: 8, points: 8, product: false, innerPoints: 0, rot: 0 },
  'n9':  { rays: 9, points: 9, product: false, innerPoints: 0, rot: 0 },
  'n10': { rays: 10, points: 10, product: false, innerPoints: 0, rot: 0 },
  'n12': { rays: 12, points: 12, product: false, innerPoints: 0, rot: 0 },
  '2x2': { rays: 8, points: 4, product: true, innerPoints: 4, rot: Math.PI/4 },
  '2x4': { rays: 10, points: 8, product: true, innerPoints: 4, rot: Math.PI/8 },
  '2x6': { rays: 12, points: 6, product: true, innerPoints: 4, rot: Math.PI/6 },
  '2x8': { rays: 12, points: 8, product: true, innerPoints: 4, rot: Math.PI/8 },
  'prod':{ rays: 10, points: 6, product: true, innerPoints: 4, rot: Math.PI/8 },
});
const RANK_COLOR_BASE = Object.freeze({ 0:0.56, 1:0.70, 2:0.08, 3:0.02, 4:0.92, 5:0.84 });
const TORSION_COLOR_OFFSET = Object.freeze({
  '0':0.00,'n2':0.03,'n3':0.06,'n4':0.09,'n5':0.13,'n6':0.17,'n7':0.21,'n8':0.25,'n9':0.29,'n10':0.33,'n12':0.38,
  '2x2':0.45,'2x4':0.52,'2x6':0.60,'2x8':0.68,'prod':0.74
});
function starStops() { return STAR_STOPS; }
function lerp(a, b, t) { return a + (b - a) * t; }
function lerpAngle(a, b, t) { let d = mod(b - a + Math.PI, TAU) - Math.PI; return mod(a + d * t, TAU); }
function slerpVec(a, b, t) {
  const d = clamp(dot(a, b), -1, 1);
  const ang = Math.acos(d);
  if (ang < 1e-7) return norm([lerp(a[0], b[0], t), lerp(a[1], b[1], t), lerp(a[2], b[2], t)]);
  const s = Math.sin(ang);
  const w0 = Math.sin((1 - t) * ang) / s;
  const w1 = Math.sin(t * ang) / s;
  return norm([a[0]*w0 + b[0]*w1, a[1]*w0 + b[1]*w1, a[2]*w0 + b[2]*w1]);
}
function ease(t) { return t < 0.5 ? 4*t*t*t : 1 - Math.pow(-2*t + 2, 3) / 2; }
function mixColor(c1, c2, t) { return [Math.round(lerp(c1[0], c2[0], t)), Math.round(lerp(c1[1], c2[1], t)), Math.round(lerp(c1[2], c2[2], t))]; }
function colorFromScalar(t) {
  const stops = STAR_STOPS;
  t = clamp(t, 0, 1);
  for (let i = 0; i < stops.length - 1; i++) {
    const [p0, c0] = stops[i], [p1, c1] = stops[i + 1];
    if (t <= p1) return mixColor(c0, c1, (t - p0) / (p1 - p0 || 1));
  }
  return stops[stops.length - 1][1].slice();
}
function rgba(rgb, a) { return `rgba(${rgb[0]},${rgb[1]},${rgb[2]},${a})`; }

function torsionSignature(label, torOrder) {
  label = String(label || '0');
  if (label === '0' || torOrder <= 1) return '0';
  if (label.includes('×')) {
    if (label.includes('Z/2Z × Z/2Z')) return '2x2';
    if (label.includes('Z/2Z × Z/4Z')) return '2x4';
    if (label.includes('Z/2Z × Z/6Z')) return '2x6';
    if (label.includes('Z/2Z × Z/8Z')) return '2x8';
    return 'prod';
  }
  return `n${torOrder}`;
}

function styleFromSignature(sig) {
  return TORSION_STYLE_MAP[sig] || TORSION_STYLE_MAP['0'];
}

function deriveVisual(p) {
  const rank = Number(p.r || 0);
  const torOrder = Math.max(1, Number(p.to || 1));
  const lp = Number(p.lp || 10007);
  const classSize = Math.max(1, Number(p.cs || 1));
  const cm = !!Number(p.cm || 0);
  const smooth = smoothScoreFromLargestPrime(lp);
  const smallPrimeBoost = Math.pow(smooth, 0.36);
  const torNorm = clamp(Math.log2(torOrder + 1) / Math.log2(17), 0, 1);
  const rankNorm = clamp(rank / 4, 0, 1);
  const classNorm = clamp(Math.log2(classSize + 1) / 4, 0, 1);
  const rankBoost = rank >= 2 ? 0.18 + 0.10 * Math.min(rank - 2, 2) : 0;
  const importance = clamp(0.58*smallPrimeBoost + 0.24*torNorm + 0.12*classNorm + 0.18*rankNorm + rankBoost + (cm ? 0.16 : 0), 0, 2.25);
  // v29: stronger bright/dim and size separation.  Smooth low-conductor
  // curves now stand out much more clearly, while faint stars remain visible.
  const brightness = clamp(0.012 + 4.9*Math.pow(importance, 2.85), 0.012, 6.4);
  const baseRadius = 0.68 + 6.85*Math.pow(importance, 1.68) + 0.30*Math.sqrt(torOrder) + 0.50*rankNorm + (cm ? 0.55 : 0);
  const priority = 11.4*importance + 1.00*rankNorm + 0.52*torNorm + 0.26*classNorm;

  // v12 normalization: color depends only on rank and torsion.
  const sig = torsionSignature(p.t, torOrder);
  const rankBase = RANK_COLOR_BASE[Math.min(rank, 5)] ?? 0.84;
  const torsionOffset = TORSION_COLOR_OFFSET[sig] ?? 0.08;
  const temp = mod(rankBase + torsionOffset, 1);
  const rgb = colorFromScalar(temp);
  const glow = mixColor(rgb, [255,255,255], clamp(0.02 + 0.10*importance, 0, 0.16));
  const hot = mixColor(rgb, [255,255,255], clamp(0.30 + 0.22*importance, 0.30, 0.55));

  const st = styleFromSignature(sig);

  p.imp = importance;
  p.bri = brightness;
  p.baseR = baseRadius;
  p.priority = priority;
  p.smooth = smooth;
  p.rankNorm = rankNorm;
  p.torNorm = torNorm;
  p.rank = rank;
  p.rgb = rgb;
  p.glow = glow;
  p.hot = hot;
  p.sig = sig;
  p.rayCount = st.rays;
  p.shapeOrder = st.points;
  p.innerShapeOrder = st.innerPoints;
  p.productStyle = st.product;
  p.shapeRot = st.rot;
}

function addPoint(raw, tileId = null) {
  if (state.points.has(raw.i)) {
    const old = state.points.get(raw.i);
    if (tileId) {
      if (!state.pointIdsByTile.has(tileId)) state.pointIdsByTile.set(tileId, new Set());
      state.pointIdsByTile.get(tileId).add(old.i);
      old.tile = old.tile || tileId;
    }
    return old;
  }
  const p = { ...raw };
  p.tile = tileId || p.tile || null;
  p.az = Number(p.az);
  p.al = Number(p.al);
  p.vec = sphToVec(p.az, p.al);
  deriveVisual(p);
  state.points.set(p.i, p);
  if (tileId) {
    if (!state.pointIdsByTile.has(tileId)) state.pointIdsByTile.set(tileId, new Set());
    state.pointIdsByTile.get(tileId).add(p.i);
  } else {
    state.topPointIds.add(p.i);
  }
  return p;
}

async function loadTop() {
  const data = await apiTop();
  data.points.forEach(raw => addPoint(raw, null));
  state.renderPool = [];
  state.lastRenderKey = '';
  requestRender();
}

async function loadTile(tile) {
  if (state.loadedTiles.has(tile.id) || state.loadingTiles.has(tile.id)) return;
  state.loadingTiles.add(tile.id);
  try {
    const data = await apiTile(tile.id);
    data.points.forEach(raw => addPoint(raw, tile.id));
    state.loadedTiles.add(tile.id);
    state.renderPool = [];
    state.lastRenderKey = '';
    requestRender();
  } catch (e) {
    console.warn('tile failed', tile.id, e);
  } finally {
    state.loadingTiles.delete(tile.id);
  }
}

function updateVisibleTiles(force = false) {
  if (!state.meta) return;
  const now = performance.now();
  if (!force && now - state.lastTileCheck < 45) return;
  state.lastTileCheck = now;
  const { w, h } = screen();
  const aspect = w / Math.max(h, 1);
  const { f } = camera();
  const halfDiag = Math.atan(Math.tan(state.fov / 2) * Math.sqrt(1 + aspect*aspect));
  const buffer = state.fov < 8*RAD ? 0.38 : state.fov < 18*RAD ? 0.32 : state.fov < 45*RAD ? 0.28 : 0.34;
  const candidates = [];
  for (const t of state.meta.tiles) {
    const d = angularDistance(f, t.c);
    if (state.fov > 92*RAD || d < halfDiag + t.rad + buffer) candidates.push([d, t]);
  }
  candidates.sort((a,b) => a[0] - b[0]);
  state.visibleTileCandidates = candidates.map(v => v[1]);
  const moving = isInteracting();
  const activeBudget = moving ? (state.fov < 14*RAD ? 52 : state.fov < 32*RAD ? 44 : 36) : (state.fov < 14*RAD ? 96 : state.fov < 32*RAD ? 84 : 72);
  state.activeTileIds = new Set(candidates.slice(0, activeBudget).map(v => v[1].id));
  state.renderPool = [];
  state.lastRenderKey = '';
  const maxLoads = isInteracting() ? (state.fov < 16*RAD ? 16 : 10) : (state.fov < 7*RAD ? 58 : state.fov < 16*RAD ? 44 : state.fov < 45*RAD ? 32 : 22);
  for (const [, t] of candidates.slice(0, maxLoads)) loadTile(t);
}

function preloadAllTiles() {
  if (!state.meta || state.preloadingAll) return;
  state.preloadingAll = true;
  // v12: keep the atlas responsive by loading visible tiles first and
  // preloading only a small idle budget in the background instead of forcing
  // all 360 tiles into memory immediately.
  const tiles = [...state.meta.tiles].sort((a,b) => (b.n || 0) - (a.n || 0));
  const idleTileBudget = 36;
  const pump = () => {
    if (state.dragging || state.travel) { setTimeout(pump, 260); return; }
    let launched = 0;
    while (state.preloadIndex < tiles.length && launched < 1 && state.loadedTiles.size < idleTileBudget) {
      const t = tiles[state.preloadIndex++];
      if (!state.loadedTiles.has(t.id) && !state.loadingTiles.has(t.id)) { loadTile(t); launched++; }
    }
    if (state.preloadIndex < tiles.length && state.loadedTiles.size < idleTileBudget) setTimeout(pump, 520);
    else state.preloadingAll = false;
  };
  setTimeout(pump, 1800);
}

function resize() {
  const r = state.canvas.getBoundingClientRect();
  state.dpr = window.devicePixelRatio || 1;
  state.viewW = r.width;
  state.viewH = r.height;
  state.canvas.width = Math.round(r.width * state.dpr);
  state.canvas.height = Math.round(r.height * state.dpr);
  state.ctx = state.canvas.getContext('2d', { alpha: false });
  state.ctx.setTransform(state.dpr, 0, 0, state.dpr, 0, 0);
  state.bgCache = null;
  state.bgCacheKey = '';
  state.renderPool = [];
  state.lastRenderKey = '';
  requestRender();
}

function resetView() {
  state.viewF = state.viewRight = state.viewUp = null;
  setViewAzAlt(0, 89.7 * RAD);
  state.fov = 30 * RAD;
  state.travel = null;
  state.renderPool = [];
  state.lastRenderKey = '';
  updateVisibleTiles(true);
  preloadAllTiles();
  requestRender();
}

function zoom(factor) {
  markInteraction(260);
  state.fov = clamp(state.fov / factor, 1e-9, 170 * RAD);
  state.renderPool = [];
  state.lastRenderKey = '';
  updateVisibleTiles(true);
  requestRender();
}

function panToScreenPoint(sx, sy) {
  markInteraction(180);
  if (!state.drag || !state.drag.anchorWorld) return;
  const anchor = state.drag.anchorWorld;
  const cam = camera();
  const current = screenToWorldFromCamera(sx, sy, cam);
  if (!current) return;
  let axis = cross(current, anchor);
  const n = Math.hypot(axis[0], axis[1], axis[2]);
  if (n < 1e-10) return;
  axis = [axis[0]/n, axis[1]/n, axis[2]/n];
  const ang = Math.acos(clamp(dot(current, anchor), -1, 1));
  const f = rotateVec(cam.f, axis, ang);
  const right = rotateVec(cam.right, axis, ang);
  const up = rotateVec(cam.up, axis, ang);
  setViewFromVectors(f, right, up);
  updateVisibleTiles();
  requestRender();
}

function cameraFromVector(f, preferredRight = null) {
  f = norm(f);
  let right = preferredRight ? tangentPart(preferredRight, f) : cross(f, [0,0,1]);
  if (Math.hypot(right[0], right[1], right[2]) < 1e-8) right = [0, -1, 0];
  right = norm(right);
  const up = norm(cross(right, f));
  return { f, right, up };
}

function animateToPoint(p) {
  const cam = camera();
  state.travel = { start: performance.now(), dur: 720, f0: cam.f.slice(), right0: cam.right.slice(), up0: cam.up.slice(), f1: sphToVec(p.az, p.al) };
  markInteraction(780);
  requestRender();
}
function updateTravel(now) {
  const tr = state.travel;
  if (!tr) return false;
  const t = clamp((now - tr.start) / tr.dur, 0, 1);
  const u = ease(t);
  const f = slerpVec(tr.f0, tr.f1, u);
  setViewFromVectors(f, tr.right0, tr.up0);
  updateVisibleTiles();
  if (t >= 1) {
    state.travel = null;
    state.renderPool = [];
    state.lastRenderKey = '';
  }
  return true;
}

function buildBackgroundCache(w, h) {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(w));
  c.height = Math.max(1, Math.round(h));
  const b = c.getContext('2d', { alpha: false });
  const g = b.createRadialGradient(w*0.5, h*0.44, 10, w*0.5, h*0.52, Math.max(w, h) * 0.82);
  g.addColorStop(0, '#0b1530');
  g.addColorStop(0.40, '#050b1c');
  g.addColorStop(1, '#020510');
  b.fillStyle = g;
  b.fillRect(0, 0, w, h);
  b.save();
  b.fillStyle = '#fff';
  for (let i = 0; i < 130; i++) {
    const x = ((i * 977) % 1000) / 1000 * w;
    const y = ((i * 421) % 1000) / 1000 * h;
    const r = 0.25 + (((i * 47) % 100) / 100) * 1.15;
    b.globalAlpha = 0.02 + (((i * 61) % 100) / 100) * 0.08;
    b.beginPath();
    b.arc(x, y, r, 0, TAU);
    b.fill();
  }
  b.restore();
  return c;
}
function drawBackground(ctx, w, h) {
  const key = `${Math.round(w)}x${Math.round(h)}`;
  if (!state.bgCache || state.bgCacheKey !== key) {
    state.bgCache = buildBackgroundCache(w, h);
    state.bgCacheKey = key;
  }
  ctx.drawImage(state.bgCache, 0, 0, w, h);
}


function drawProjectedPath(ctx, pts, strokeStyle, lineWidth = 1, close = false, opts = {}) {
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false, last = null, first = null, firstStarted = false, visible = 0;
  const { w, h } = screen();
  const jumpLimit = opts.noSplit ? Infinity : Math.max(w, h) * (opts.jumpScale || 0.58);
  for (let i = 0; i < pts.length; i++) {
    const p = projectAzAlt(pts[i][0], pts[i][1]);
    if (!p) {
      if (!opts.bridgeGaps) { started = false; last = null; }
      continue;
    }
    visible++;
    if (last && Math.hypot(p.x - last.x, p.y - last.y) > jumpLimit) {
      started = false;
      last = null;
    }
    if (!started) {
      ctx.moveTo(p.x, p.y);
      started = true;
      if (!firstStarted) { first = p; firstStarted = true; }
    } else {
      ctx.lineTo(p.x, p.y);
    }
    last = p;
  }
  if (close && first && last && visible >= Math.max(8, pts.length * 0.72)) {
    const gap = Math.hypot(first.x - last.x, first.y - last.y);
    if (gap < Math.max(w, h) * 0.22 || opts.forceClose) ctx.closePath();
  }
  ctx.stroke();
}

function drawProjectedClosedLoop(ctx, pts, strokeStyle, lineWidth = 1, opts = {}) {
  // Stable closed-loop projection used for high latitude circles.  It avoids
  // the v23/v24 failure mode where disconnected visible arc segments were
  // joined across the whole viewport, while still explicitly closing smooth
  // near-zenith rings when the whole ring is visible.
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  const { w, h } = screen();
  const jumpLimit = Math.max(w, h) * (opts.jumpScale || 0.62);
  let first = null, last = null, started = false, visible = 0, segmentCount = 0;
  for (let i = 0; i <= pts.length; i++) {
    const src = pts[i % pts.length];
    const p = projectAzAlt(src[0], src[1]);
    if (!p) {
      started = false;
      last = null;
      continue;
    }
    if (i < pts.length) visible++;
    if (last && Math.hypot(p.x - last.x, p.y - last.y) > jumpLimit) {
      started = false;
      last = null;
    }
    if (!started) {
      ctx.moveTo(p.x, p.y);
      if (!first) first = p;
      started = true;
      segmentCount++;
    } else {
      ctx.lineTo(p.x, p.y);
    }
    last = p;
  }
  if (opts.forceClose && first && last && visible >= pts.length * 0.78 && segmentCount <= 2) {
    const gap = Math.hypot(first.x - last.x, first.y - last.y);
    if (gap < Math.max(w, h) * 0.32) ctx.closePath();
  }
  ctx.stroke();
}

function chooseNiceStep(spanDeg, targetLines = 6) {
  const opts = [60, 45, 30, 20, 15, 12, 10, 7.5, 6, 5, 4, 3, 2, 1.5, 1, 0.5];
  const maxStep = Math.max(0.5, spanDeg / targetLines);
  for (const s of opts) if (s <= maxStep) return s;
  return opts[opts.length - 1];
}

function gridStepDeg(fast = false) {
  const f = state.fov / RAD;
  const camAlt = vecToAzAlt(camera().f)[1] / RAD;
  const altSpan = Math.min(88, Math.max(10, f * 1.05));
  // v26: near the zenith many meridians collapse into visually similar radial
  // lines.  Draw fewer meridians there, but use denser latitude sampling so the
  // top cap remains smooth and closed without causing per-frame stalls.
  const targetAlt = fast ? 4 : (camAlt > 82 ? 9 : state.fov < 20 * RAD ? 8 : 6);
  const targetAz = fast ? (camAlt > 76 ? 4 : 5) : (camAlt > 82 ? 6 : camAlt > 70 ? 7 : 8);
  const altStep = chooseNiceStep(altSpan, targetAlt);
  const azSpan = Math.min(360, Math.max(70, f * 1.35 / Math.max(0.24, Math.cos(camAlt * RAD))));
  let azStep = chooseNiceStep(azSpan, targetAz);
  if (camAlt > 80) azStep = Math.max(azStep, fast ? 45 : 30);
  else if (fast) azStep = Math.max(azStep, 24);
  return { alt: altStep, az: azStep, camAlt };
}

function uniqueSortedDegrees(vals) {
  const seen = new Set();
  const out = [];
  for (const raw of vals) {
    const v = Math.round(raw * 1000) / 1000;
    if (!Number.isFinite(v) || seen.has(v)) continue;
    seen.add(v); out.push(v);
  }
  return out.sort((a, b) => a - b);
}

function projectedPolylineSegments(pts, maxJumpFactor = 0.40) {
  const { w, h } = screen();
  const jump = Math.max(120, Math.max(w, h) * maxJumpFactor);
  const segs = [];
  let seg = [], last = null;
  for (const src of pts) {
    const p = projectAzAlt(src[0], src[1]);
    if (!p) {
      if (seg.length > 1) segs.push(seg);
      seg = []; last = null;
      continue;
    }
    if (last && Math.hypot(p.x - last.x, p.y - last.y) > jump) {
      if (seg.length > 1) segs.push(seg);
      seg = [];
    }
    seg.push(p);
    last = p;
  }
  if (seg.length > 1) segs.push(seg);
  return segs;
}

function strokeSegments(ctx, segs) {
  ctx.beginPath();
  for (const seg of segs) {
    ctx.moveTo(seg[0].x, seg[0].y);
    for (let i = 1; i < seg.length; i++) ctx.lineTo(seg[i].x, seg[i].y);
  }
  ctx.stroke();
}

function drawGridRing(ctx, altDeg, samples, style, width, forceClose = false) {
  const alt = clamp(altDeg, 0.05, 89.35) * RAD;
  const pts = [];
  for (let i = 0; i < samples; i++) pts.push([i * TAU / samples, alt]);
  // Repeat the first point only for the top/high-altitude loops.  The segmented
  // draw path prevents accidental long vertical screen-spanning joins.
  if (forceClose) pts.push([0, alt]);
  const segs = projectedPolylineSegments(pts, altDeg > 82 ? 0.28 : 0.42);
  ctx.strokeStyle = style;
  ctx.lineWidth = width;
  strokeSegments(ctx, segs);
}

function drawGridMeridian(ctx, azDeg, samples, style, width) {
  const az = azDeg * RAD;
  const pts = [];
  for (let i = 0; i <= samples; i++) {
    const t = i / samples;
    // Ease the last part into the closed zenith cap.  This avoids jagged and
    // nearly vertical overshoots near the top while keeping axes visible.
    const altDeg = 0.25 + (89.25 - 0.25) * (1 - Math.pow(1 - t, 1.18));
    pts.push([az, altDeg * RAD]);
  }
  const segs = projectedPolylineSegments(pts, 0.34);
  ctx.strokeStyle = style;
  ctx.lineWidth = width;
  strokeSegments(ctx, segs);
}

function drawSkyGrid(ctx, fast = false) {
  // v26 grid: constant-time lightweight mesh during movement, denser smooth
  // rings only when still.  This deliberately avoids the v23/v24 style of
  // drawing many high-latitude meridian samples that caused zenith stalls and
  // occasional screen-spanning vertical joins.
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.shadowBlur = 0;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const step = gridStepDeg(fast);
  const topAltDeg = 89.25;
  const camAlt = step.camAlt;
  const ringSamples = fast ? 56 : (camAlt > 80 ? 168 : 128);
  const topSamples = fast ? 96 : 240;
  const meridianSamples = fast ? 24 : (camAlt > 80 ? 44 : 62);

  const latDegs = [];
  for (let altDeg = 0; altDeg < topAltDeg - step.alt * 0.5; altDeg += step.alt) latDegs.push(altDeg);
  latDegs.push(30, 45, 60, 75, 84, topAltDeg);
  for (const altDeg of uniqueSortedDegrees(latDegs)) {
    if (altDeg < 0 || altDeg > topAltDeg) continue;
    const top = Math.abs(altDeg - topAltDeg) < 1e-6;
    const high = altDeg >= 75;
    const rounded = Math.round(altDeg);
    const major = top || altDeg === 0 || rounded === 30 || rounded === 45 || rounded === 60 || rounded === 75 || rounded === 84;
    const samples = top ? topSamples : high ? Math.max(ringSamples, fast ? 88 : 176) : ringSamples;
    const color = top ? 'rgba(238,162,104,0.54)' : major ? 'rgba(218,132,82,0.35)' : 'rgba(198,116,72,0.145)';
    const width = top ? (fast ? 0.92 : 1.18) : major ? (fast ? 0.68 : 0.92) : (fast ? 0.34 : 0.50);
    drawGridRing(ctx, altDeg, samples, color, width, top || high);
  }

  const azDegs = [];
  for (let azDeg = 0; azDeg < 360; azDeg += step.az) azDegs.push(azDeg);
  azDegs.push(0, 90, 180, 270);
  for (const azDeg of uniqueSortedDegrees(azDegs)) {
    if (azDeg >= 360) continue;
    const axis = azDeg === 0 || azDeg === 90 || azDeg === 180 || azDeg === 270;
    // Near zenith, non-axis meridians are visually redundant and expensive;
    // keep axes plus coarse meridians only.
    if (step.camAlt > 82 && !axis && (azDeg % 60 !== 0)) continue;
    const major = axis || (azDeg % 90 === 0) || (azDeg % Math.max(step.az * 2, 1) === 0);
    const color = axis ? 'rgba(255,190,132,0.50)' : major ? 'rgba(218,132,82,0.26)' : 'rgba(198,116,72,0.12)';
    const width = axis ? (fast ? 0.82 : 1.05) : major ? (fast ? 0.50 : 0.72) : (fast ? 0.30 : 0.42);
    drawGridMeridian(ctx, azDeg, meridianSamples, color, width);
  }
  ctx.restore();
}

function pointRadius(p, proj) {
  const z = zoomLevel();
  // v29: stars can grow significantly larger, but the growth remains more
  // controlled near deep zoom so detailed starbursts do not collapse together.
  const growth = 0.72 + (0.46 + 0.96*p.imp) * Math.pow(Math.max(z, 0) + 0.16, 0.98);
  const cap = 34 + 58*p.imp + 10*Math.sqrt(Math.max(0, z));
  const minR = 0.72 + Math.min(4.8, 0.30 * z);
  return clamp(p.baseR * growth * Math.pow(proj.z, 0.19), minR, cap);
}

function detailThreshold(p) {
  const z = zoomLevel();
  return 1.45 + 3.75 * (1 - Math.min(1.35, p.imp)) + Math.max(0, 0.8 - z) * 0.72;
}

function starPolygonPath(ctx, x, y, rOuter, rInner, points, rotation = -Math.PI/2) {
  ctx.beginPath();
  for (let i = 0; i < points * 2; i++) {
    const r = i % 2 === 0 ? rOuter : rInner;
    const ang = rotation + i * Math.PI / points;
    const px = x + r * Math.cos(ang);
    const py = y + r * Math.sin(ang);
    if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
  }
  ctx.closePath();
}


function drawSelectedRing(ctx, proj, r, now) {
  // Canvas fallback only.  The live marker is a DOM overlay whose four cursors
  // orbit the selected star; no extra center dot is drawn so the selected star
  // itself remains fixed and visible.
  const t = now / 1000;
  const rot = t * 0.24;
  const rr = Math.max(10, r * 1.72 + 4.5);
  const tick = Math.max(3.0, Math.min(7.2, rr * 0.22));
  ctx.save();
  ctx.translate(proj.x, proj.y);
  ctx.rotate(rot);
  ctx.strokeStyle = 'rgba(255,255,255,0.92)';
  ctx.lineWidth = Math.max(1.15, Math.min(2.0, r * 0.09));
  ctx.lineCap = 'round';
  ctx.beginPath();
  for (let k = 0; k < 4; k++) {
    const a = k * Math.PI / 2;
    ctx.moveTo(Math.cos(a) * rr, Math.sin(a) * rr);
    ctx.lineTo(Math.cos(a) * (rr - tick), Math.sin(a) * (rr - tick));
  }
  ctx.stroke();
  ctx.restore();
}

function updateSelectionMarker() {
  const el = state.selectionEl;
  if (!el) return;
  const sp = state.selectedProjection;
  if (!state.selected || !sp || !sp.pr) {
    el.classList.remove('visible');
    return;
  }
  // v26: larger orbit so the four-cursor selection shape is legible and
  // does not crowd the selected star core.
  const rr = Math.max(34, Math.min(82, sp.r * 4.65 + 24));
  const rect = state.canvas ? state.canvas.getBoundingClientRect() : { left: 0, top: 0 };
  el.style.setProperty('--sel-x', `${(rect.left + sp.pr.x).toFixed(2)}px`);
  el.style.setProperty('--sel-y', `${(rect.top + sp.pr.y).toFixed(2)}px`);
  el.style.setProperty('--sel-size', `${rr.toFixed(2)}px`);
  el.classList.add('visible');
}

function drawPointStar(ctx, p, proj, r, selected) {
  const glowR = r * (2.0 + 1.65 * Math.min(1.55, p.imp));
  const dim = clamp((p.bri - 0.012) / (6.4 - 0.012), 0, 1);
  const g = ctx.createRadialGradient(proj.x, proj.y, 0, proj.x, proj.y, glowR);
  g.addColorStop(0, rgba(p.glow, clamp(0.010 + 0.38 * Math.pow(dim, 1.12), 0.01, 0.60)));
  g.addColorStop(0.22, rgba(p.glow, clamp(0.005 + 0.22 * Math.pow(dim, 1.10), 0.005, 0.34)));
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, glowR, 0, TAU);
  ctx.fill();

  ctx.fillStyle = rgba(p.hot, clamp(0.045 + 1.02 * Math.pow(dim, 1.24), 0.045, 1));
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, r, 0, TAU);
  ctx.fill();

  ctx.fillStyle = rgba([255,255,255], clamp(0.34 + 0.62 * Math.pow(dim, 0.86), 0.34, 1));
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, Math.max(0.68, r * (0.30 + 0.24 * Math.pow(dim, 0.82))), 0, TAU);
  ctx.fill();

}

function drawRays(ctx, p, proj, r) {
  const rays = Math.min(14, p.rayCount);
  const inner = r * (p.productStyle ? 1.55 : 1.62);
  const outer = inner + r * (0.62 + 0.36 * p.imp);
  const altLen = inner + r * (0.36 + 0.20 * p.imp);
  ctx.save();
  ctx.strokeStyle = rgba(p.glow, clamp(0.08 + 0.12 * p.bri, 0.08, 0.72));
  ctx.lineWidth = Math.max(0.68, r * 0.09);
  ctx.lineCap = 'round';
  const rot = p.shapeRot || 0;
  for (let i = 0; i < rays; i++) {
    const ang = -Math.PI/2 + rot + i * TAU / rays;
    const out = (i % 2 === 0 || !p.productStyle) ? outer : altLen;
    const x1 = proj.x + inner * Math.cos(ang);
    const y1 = proj.y + inner * Math.sin(ang);
    const x2 = proj.x + out * Math.cos(ang);
    const y2 = proj.y + out * Math.sin(ang);
    ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
  }
  ctx.restore();
}

function detailedBurstScale(p, proj, r) {
  const zoom = zoomLevel();
  // v30-redo: keep the v29 amount/level logic unchanged.  Only the decorative
  // starburst size is locally braked at deep zoom or in tight clusters.
  const zoomBrake = zoom <= 1.55 ? 1.0 : 1 / (1 + Math.pow(zoom - 1.55, 1.10) * 0.22);
  const count = state.rendered ? state.rendered.length : 0;
  const densityScale = count <= 260 ? 0.94 : count <= 900 ? 0.88 : 0.82;
  let scale = densityScale * zoomBrake;

  const gap = state.detailSeparation ? state.detailSeparation.get(Number(p.i)) : null;
  if (gap && Number.isFinite(gap)) {
    const desiredFootprint = r * scale * (2.78 + 0.18 * Math.min(2, p.imp || 0));
    const safeFootprint = Math.max(13, gap * 0.46);
    if (desiredFootprint > safeFootprint) scale *= safeFootprint / desiredFootprint;
  }

  if (state.selected && Number(state.selected.i) === Number(p.i)) scale = Math.max(scale, 0.56);
  return clamp(scale, 0.34, 1.0);
}

function drawDetailedCore(ctx, p, proj, r, selected) {
  const pts = p.shapeOrder;
  const z = zoomLevel();
  const detailScale = detailedBurstScale(p, proj, r);
  const rr = r * detailScale;
  const outlineOnly = rr > 5.8 || z > 2.1;

  const halo = ctx.createRadialGradient(proj.x, proj.y, 0, proj.x, proj.y, rr * (1.95 + 0.82 * Math.min(1.2, p.imp)));
  halo.addColorStop(0, rgba(p.glow, clamp(0.018 + 0.18 * p.bri, 0.018, 0.30)));
  halo.addColorStop(0.25, rgba(p.glow, clamp(0.010 + 0.10 * p.bri, 0.01, 0.16)));
  halo.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = halo;
  ctx.beginPath(); ctx.arc(proj.x, proj.y, rr * (2.35 + 0.78 * p.imp), 0, TAU); ctx.fill();

  drawRays(ctx, p, proj, rr);
  ctx.save();
  ctx.shadowBlur = 7 + 5 * p.imp;
  ctx.shadowColor = rgba(p.glow, 0.88);
  ctx.lineWidth = Math.max(1.0, rr * 0.16);
  ctx.strokeStyle = rgba(p.hot, 0.985);
  ctx.fillStyle = rgba(p.rgb, outlineOnly ? 0.08 : 0.30);

  if (!pts) {
    ctx.beginPath();
    ctx.arc(proj.x, proj.y, rr * 0.92, 0, TAU);
    if (!outlineOnly) ctx.fill();
    ctx.stroke();
  } else {
    const outer = rr * 0.98;
    const inner = outer * (pts <= 4 ? 0.62 : pts <= 6 ? 0.50 : 0.42);
    starPolygonPath(ctx, proj.x, proj.y, outer, inner, pts, -Math.PI/2 + (p.shapeRot || 0));
    if (!outlineOnly) ctx.fill();
    ctx.stroke();
    if (p.productStyle && p.innerShapeOrder) {
      ctx.lineWidth = Math.max(0.9, rr * 0.11);
      starPolygonPath(ctx, proj.x, proj.y, outer * 0.58, outer * 0.34, p.innerShapeOrder, -Math.PI/2 + (p.shapeRot || 0) + Math.PI/(2*Math.max(3,p.innerShapeOrder)));
      ctx.stroke();
    } else if (pts >= 8 || p.sig === 'n6') {
      ctx.lineWidth = Math.max(0.9, rr * 0.11);
      starPolygonPath(ctx, proj.x, proj.y, outer * 0.58, outer * 0.30, Math.max(3, Math.min(pts, 6)), Math.PI / Math.max(3, pts));
      ctx.stroke();
    }
  }

  const rings = Math.max(0, Math.min(4, Number(p.rank || 0)));
  if (rings > 0) {
    ctx.shadowBlur = 0;
    ctx.strokeStyle = rgba(p.hot, 0.82);
    ctx.lineWidth = Math.max(0.7, rr * 0.052);
    for (let k = 0; k < rings; k++) {
      ctx.beginPath();
      ctx.arc(proj.x, proj.y, rr * (1.16 + 0.17 * k), 0, TAU);
      ctx.stroke();
    }
  }

  if (Number(p.cm)) {
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(246,214,111,0.97)';
    ctx.lineWidth = Math.max(0.8, rr * 0.085);
    ctx.beginPath(); ctx.arc(proj.x, proj.y, rr * 1.30, 0, TAU); ctx.stroke();
    for (let k = 0; k < 4; k++) {
      const ang = Math.PI/4 + k * Math.PI/2;
      ctx.beginPath();
      ctx.moveTo(proj.x + Math.cos(ang) * rr * 1.40, proj.y + Math.sin(ang) * rr * 1.40);
      ctx.lineTo(proj.x + Math.cos(ang) * rr * 1.64, proj.y + Math.sin(ang) * rr * 1.64);
      ctx.stroke();
    }
  }

  ctx.beginPath();
  ctx.arc(proj.x, proj.y, Math.max(0.55, rr * 0.22), 0, TAU);
  ctx.fillStyle = rgba([255,255,255], 0.95);
  ctx.fill();

  ctx.restore();
}

const LABEL_RENDER_MAX = 80;
function drawVisibleLabels(ctx) {
  const items = state.rendered || [];
  if (!items.length || items.length > LABEL_RENDER_MAX) return;
  ctx.save();
  ctx.font = '500 10.5px Inter, Segoe UI, Arial, sans-serif';
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  ctx.lineJoin = 'round';
  for (const [p, pr, r] of items) {
    const label = String(p.l || p.label || '');
    if (!label) continue;
    const offset = Math.max(10, Math.min(32, r * 1.75 + 8));
    const x = pr.x + offset;
    const y = pr.y - offset * 0.42;
    const a = clamp(0.34 + 0.13 * Math.min(1, p.imp || 0), 0.34, 0.52);
    ctx.lineWidth = 3;
    ctx.strokeStyle = 'rgba(2,6,18,0.58)';
    ctx.strokeText(label, x, y);
    ctx.fillStyle = `rgba(220,236,255,${a})`;
    ctx.fillText(label, x, y);
  }
  ctx.restore();
}

function markInteraction(ms = 220) {
  state.interactingUntil = Math.max(state.interactingUntil || 0, performance.now() + ms);
}
function isInteracting(now = performance.now()) {
  return state.dragging || !!state.pinch || !!state.travel || now < (state.interactingUntil || 0);
}
function renderKey() {
  const cam = camera();
  const f = cam.f;
  return [
    Math.round(f[0] * 1200), Math.round(f[1] * 1200), Math.round(f[2] * 1200),
    Math.round(state.fov / RAD * 18), state.visibleLevel,
    state.loadedTiles.size, state.points.size
  ].join('|');
}
function buildRenderPool(now = performance.now()) {
  const key = renderKey();
  if (state.renderPool.length && state.lastRenderKey === key) return state.renderPool;
  state.lastSelectionKey = '';
  const cam = camera();
  const { w, h } = screen();
  const aspect = w / Math.max(h, 1);
  const halfDiag = Math.atan(Math.tan(state.fov / 2) * Math.sqrt(1 + aspect * aspect));
  const cosLimit = Math.cos(Math.min(Math.PI, halfDiag + 0.42));
  const ids = new Set(state.topPointIds);
  for (const tid of state.activeTileIds) {
    const bucket = state.pointIdsByTile.get(tid);
    if (!bucket) continue;
    for (const id of bucket) ids.add(id);
  }
  const special = [state.selected, state.hover].filter(Boolean);
  const specialIds = new Set();
  for (const p of special) { ids.add(p.i); specialIds.add(p.i); }
  const scale = projectionScale();
  const zLevel = zoomLevel();
  const pool = [];
  for (const id of ids) {
    const p = state.points.get(id);
    if (!p) continue;
    const z = dot(p.vec, cam.f);
    if (z < cosLimit && !specialIds.has(p.i)) continue;
    const xr = dot(p.vec, cam.right);
    const yu = dot(p.vec, cam.up);
    if (z <= 0) continue;
    const denom = Math.max(1e-9, 1 + z);
    const x = w / 2 + xr / denom * scale;
    const y = h / 2 - yu / denom * scale;
    if (x < -160 || x > w + 160 || y < -160 || y > h + 160) continue;
    const pr = { x, y, z, rho: Math.sqrt(xr*xr + yu*yu) / denom };
    const r = pointRadius(p, pr);
    const score = p.priority + 0.10 * zLevel * p.imp + Math.min(0.55, r * 0.018);
    pool.push([p, pr, r, score]);
  }
  state.renderPool = pool;
  state.lastRenderKey = key;
  return pool;
}
function drawFastDot(ctx, p, pr, r, selected) {
  const rr = Math.max(1.05, Math.min(r, 10.5));
  ctx.fillStyle = rgba(p.hot, clamp(0.26 + 0.62 * p.bri, 0.24, 1));
  ctx.beginPath();
  ctx.arc(pr.x, pr.y, rr, 0, TAU);
  ctx.fill();
  if (rr > 2.2) {
    ctx.fillStyle = rgba([255,255,255], 0.55);
    ctx.beginPath();
    ctx.arc(pr.x, pr.y, Math.max(0.45, rr * 0.24), 0, TAU);
    ctx.fill();
  }
}


function updateDetailSeparation() {
  state.detailSeparation = new Map();
  if (!state.detailIds || !state.detailIds.size) return;
  const stars = [];
  for (const [p, pr, r] of state.rendered || []) {
    if (state.detailIds.has(p.i)) stars.push({ id: Number(p.i), x: pr.x, y: pr.y, r });
  }
  if (stars.length <= 1) {
    for (const s of stars) state.detailSeparation.set(s.id, 9999);
    return;
  }

  const cell = 72;
  const grid = new Map();
  const key = (cx, cy) => `${cx},${cy}`;
  for (const s of stars) {
    const cx = Math.floor(s.x / cell), cy = Math.floor(s.y / cell);
    const k = key(cx, cy);
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(s);
    s.cx = cx; s.cy = cy;
  }

  for (const s of stars) {
    let best = Infinity;
    for (let dx = -2; dx <= 2; dx++) for (let dy = -2; dy <= 2; dy++) {
      const bucket = grid.get(key(s.cx + dx, s.cy + dy));
      if (!bucket) continue;
      for (const t of bucket) {
        if (t.id === s.id) continue;
        const d = Math.hypot(s.x - t.x, s.y - t.y);
        if (d < best) best = d;
      }
    }
    state.detailSeparation.set(s.id, Number.isFinite(best) ? best : 9999);
  }
}


function selectRenderable(now = performance.now()) {
  const pool = buildRenderPool(now);
  const interacting = isInteracting(now);
  const maxBase = dynamicMax();
  const max = interacting ? Math.max(90, Math.round(maxBase * 0.50)) : maxBase;
  const selectionKey = `${state.lastRenderKey}|${interacting ? 1 : 0}|${max}|${state.selected ? state.selected.i : ''}|${state.hover ? state.hover.i : ''}`;
  if (state.lastSelectionKey === selectionKey && state.rendered && state.rendered.length) return;
  state.lastSelectionKey = selectionKey;
  let sorted;
  if (interacting && pool.length > max * 2.6) {
    // During drag/pinch keep only high-priority objects before sorting.  This
    // mimics star-map LOD: broad context remains visible while expensive low
    // priority stars wait until the camera settles.
    const cutoff = pool.length > 3200 ? 4.1 : 3.2;
    sorted = pool.filter(v => v[3] >= cutoff || (state.selected && v[0].i === state.selected.i) || (state.hover && v[0].i === state.hover.i));
    if (sorted.length < max) sorted = pool.slice(0, Math.min(pool.length, max * 2));
  } else {
    sorted = pool.slice();
  }
  sorted.sort((A, B) => B[3] - A[3]);
  state.rendered = sorted.slice(0, max);
  const ids = new Set(state.rendered.map(v => v[0].i));
  for (const special of [state.selected, state.hover]) {
    if (special && !ids.has(special.i)) {
      const pr = projectAzAlt(special.az, special.al);
      if (pr) state.rendered.push([special, pr, pointRadius(special, pr), special.priority + 10]);
    }
  }
  state.selectedProjection = null;
  const detailCandidates = [];
  if (!interacting) {
    let order = 0;
    for (const [p, pr, r, priority] of state.rendered) {
      // state.rendered is already sorted by visual priority.  Give the first
      // ~100 visible stars their real torsion/rank shape even when they are not
      // large enough to pass the old radius threshold.
      detailCandidates.push([100000 - order + 0.15 * (priority || 0) + 0.08 * r, p.i]);
      order++;
      if (state.selected && p.i === state.selected.i) state.selectedProjection = { pr, r };
    }
  } else if (state.selected) {
    for (const [p, pr, r] of state.rendered) {
      if (p.i === state.selected.i) { state.selectedProjection = { pr, r }; break; }
    }
  }
  detailCandidates.sort((a, b) => b[0] - a[0]);
  const visibleCount = state.rendered.length;
  // v29: detailed starbursts use 30% of the selected visible-star budget,
  // capped only by however many stars are actually on screen.
  const detailBudget = Math.ceil((VISIBLE_LEVELS[state.visibleLevel] || 1800) * 0.30);
  const detailLimit = interacting ? 0 : Math.min(visibleCount, detailBudget);
  state.detailIds = new Set(detailCandidates.slice(0, detailLimit).map(v => v[1]));
  if (state.selected) state.detailIds.add(state.selected.i);
  updateDetailSeparation();
}

function render(now = performance.now()) {
  const { w, h } = screen();
  const ctx = state.ctx;
  const fast = isInteracting(now);
  ctx.clearRect(0, 0, w, h);
  drawBackground(ctx, w, h);
  drawSkyGrid(ctx, fast);
  selectRenderable(now);
  updateSelectionMarker();

  if (fast) {
    for (const [p, pr, r] of state.rendered) drawFastDot(ctx, p, pr, r, state.selected && state.selected.i === p.i);
    drawVisibleLabels(ctx);
    return;
  }

  // Draw all simple point stars first.
  for (const [p, pr, r] of state.rendered) {
    if (state.detailIds.has(p.i)) continue;
    drawPointStar(ctx, p, pr, r, state.selected && state.selected.i === p.i);
  }
  // Then detailed stars on top.
  for (const [p, pr, r] of state.rendered) {
    if (!state.detailIds.has(p.i)) continue;
    drawDetailedCore(ctx, p, pr, r, state.selected && state.selected.i === p.i);
  }
  drawVisibleLabels(ctx);
}

function projectedRenderableForPoint(p) {
  if (!p) return null;
  for (const item of state.rendered || []) {
    if (item[0] && Number(item[0].i) === Number(p.i)) return item;
  }
  const pr = projectAzAlt(p.az, p.al);
  if (!pr) return null;
  return [p, pr, pointRadius(p, pr), p.priority || 0];
}

function pointerIsNearPoint(p, sx, sy, extra = 18) {
  const item = projectedRenderableForPoint(p);
  if (!item) return false;
  const [, pr, r0] = item;
  const hitR = clamp((r0 || 0) * 0.74 + extra, 12, 38);
  return Math.hypot(pr.x - sx, pr.y - sy) <= hitR;
}

function activeHoverPointForClick(sx, sy) {
  if (state.hover && pointerIsNearPoint(state.hover, sx, sy, 24)) return state.hover;
  if (state.hoverCandidate && pointerIsNearPoint(state.hoverCandidate, sx, sy, 22)) return state.hoverCandidate;
  return null;
}

function nearest(sx, sy, threshold = 18, opts = {}) {
  const preferSmall = opts.preferSmall !== false;
  let best = null, bestScore = Infinity;
  const items = state.rendered || [];
  for (let idx = items.length - 1; idx >= 0; idx--) {
    const [p, pr, r0] = items[idx];
    const r = r0 || pointRadius(p, pr);
    const center = Math.hypot(pr.x - sx, pr.y - sy);
    const coreR = clamp(r * 0.58 + threshold, Math.max(10, threshold * 0.66), Math.max(22, threshold + 12));
    if (center > coreR) continue;
    let score = center / coreR;
    if (preferSmall) score += Math.min(0.22, r * 0.006);
    if (state.hover && Number(state.hover.i) === Number(p.i)) score -= 0.42;
    if (state.hoverCandidate && Number(state.hoverCandidate.i) === Number(p.i)) score -= 0.22;
    if (state.selected && Number(state.selected.i) === Number(p.i)) score -= 0.04;
    if (score < bestScore) {
      bestScore = score;
      best = p;
    }
  }
  return best;
}

function groupFromPoint(p) {
  const torsion = (!p.t || p.t === '0') ? '' : ` ⊕ ${p.t}`;
  return `E(Q) ≅ Z^${p.r ?? 0}${torsion}`;
}

function fetchHoverInfo(p) {
  if (!p || state.hoverInfoCache.has(p.i) || state.hoverInfoPending.has(p.i)) return;
  state.hoverInfoPending.add(p.i);
  apiHover(p.i)
    .then(data => data && !data.error ? data : null)
    .then(data => {
      state.hoverInfoPending.delete(p.i);
      if (data && !data.error) state.hoverInfoCache.set(p.i, data);
      if (state.hover && state.hover.i === p.i && state.hoverMouse) {
        showTooltip(state.hover, state.hoverMouse.x, state.hoverMouse.y);
      }
    })
    .catch(() => state.hoverInfoPending.delete(p.i));
}

function clearHoverTimer() {
  if (state.hoverTimer) {
    clearTimeout(state.hoverTimer);
    state.hoverTimer = null;
  }
}

function clearHoverState(hide = true) {
  clearHoverTimer();
  state.hoverCandidate = null;
  state.hoverMouse = null;
  if (state.hover) { state.hover = null; requestRender(); }
  if (hide && state.tooltip) state.tooltip.style.display = 'none';
}

function scheduleHover(p, clientX, clientY) {
  if (!p) { clearHoverState(true); return; }
  const sameCandidate = state.hoverCandidate && state.hoverCandidate.i === p.i;
  state.hoverMouse = { x: clientX, y: clientY };
  if (state.hover && state.hover.i === p.i) {
    showTooltip(p, clientX, clientY);
    return;
  }
  if (!sameCandidate) {
    clearHoverTimer();
    state.hoverCandidate = p;
    if (state.tooltip) state.tooltip.style.display = 'none';
    state.hoverTimer = setTimeout(() => {
      const cand = state.hoverCandidate;
      if (!cand || cand.i !== p.i || !state.hoverMouse) return;
      state.hover = cand;
      showTooltip(cand, state.hoverMouse.x, state.hoverMouse.y);
      requestRender();
    }, state.hoverStillMs);
  }
}

function showTooltip(p, sx, sy) {
  const t = state.tooltip;
  if (!p) { t.style.display = 'none'; return; }
  state.hoverMouse = { x: sx, y: sy };
  const info = state.hoverInfoCache.get(p.i);
  const label = info?.label || p.l;
  const group = info?.group || groupFromPoint(p);
  const weier = info?.weierstrass_equation || 'loading…';
  const disc = info?.disc || 'loading…';
  const jinv = info?.j_str || 'loading…';
  t.innerHTML = `
    <div class="tip-grid">
      <div class="tip-k">label</div><div><b>${escapeHtml(label)}</b></div>
      <div class="tip-k">Weierstrass form</div><div class="code tip-equation">${escapeHtml(weier)}</div>
      <div class="tip-k">group</div><div>${escapeHtml(group)}</div>
      <div class="tip-k">discriminant</div><div class="code">${escapeHtml(disc)}</div>
      <div class="tip-k">j-invariant</div><div class="code">${escapeHtml(jinv)}</div>
    </div>`;
  t.style.left = Math.min(window.innerWidth - 430, sx + 14) + 'px';
  t.style.top = Math.max(70, sy - 88) + 'px';
  t.style.display = 'block';
  fetchHoverInfo(p);
}

function closeDetail(clear = true) {
  $('detail').classList.add('hidden');
  $('detail-content').innerHTML = '';
  state.currentDetailId = null;
  state.detailLoadToken++;
  if (clear) { state.selected = null; state.selectedProjection = null; updateSelectionMarker(); }
  requestRender();
}

function ensureMemberPoint(m) {
  return addPoint({ i: m.id, l: m.label, iso: m.iso, N: m.N, r: m.rank, t: m.tor_label, to: m.tor_order, cm: m.cm ? 1 : 0, az: Number(m.az), al: Number(m.alt), lp: m.largest_prime, ci: m.class_index, cs: m.class_size });
}

async function travelAndOpen(id, member = null) {
  let p = state.points.get(id);
  if (!p && member) p = ensureMemberPoint(member);
  if (p) { state.selected = p; state.selectedPulseUntil = performance.now() + 1200; animateToPoint(p); }
  await openCurve(id, false);
}


function escapeHtml(v) {
  return String(v ?? '').replace(/[&<>"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[ch]));
}

function weierstrassEquation(d) {
  if (d.weierstrass_equation) return escapeHtml(d.weierstrass_equation);
  const term = (coef, body) => {
    coef = Number(coef || 0);
    if (!coef) return '';
    const sign = coef < 0 ? ' - ' : ' + ';
    const mag = Math.abs(coef);
    return sign + ((mag === 1 && body) ? body : `${mag}${body}`);
  };
  return escapeHtml(`y²${term(d.a1, 'xy')}${term(d.a3, 'y')} = x³${term(d.a2, 'x²')}${term(d.a4, 'x')}${term(d.a6, '')}`);
}

function pointsHtml(data) {
  if (!data || data.error) return '<div class="muted">Failed to compute points.</div>';
  if (data.canceled) return '<div class="muted">Computation canceled because another curve was selected.</div>';
  const pts = data.points || [];
  const rows = pts.length ? pts.map(pt => `<span class="pill">(${pt.x}, ${pt.y})</span>`).join('') : '<span class="muted">No points found in the current search range.</span>';
  const status = data.complete ? 'complete' : (data.timed_out ? 'timed out; may be incomplete' : 'bounded search; may be incomplete');
  const extra = [];
  if (data.generated_count) extra.push(`group-generated=${data.generated_count}`);
  if (data.low_height_rational_hits != null) extra.push(`low-height rational hits=${data.low_height_rational_hits}`);
  if (data.cache_runs && data.cache_runs > 1) extra.push(`continued run #${data.cache_runs}`);
  if (data.total_elapsed_ms && data.total_elapsed_ms !== data.elapsed_ms) extra.push(`total=${data.total_elapsed_ms} ms`);
  return `<div>${rows}</div><p class="muted">count=${data.count ?? pts.length}; ${status}; elapsed=${data.elapsed_ms} ms${extra.length ? '; ' + extra.join('; ') : ''}. ${data.note || ''}</p>`;
}

async function loadIntegralPoints(curveId, token = state.detailLoadToken) {
  const box = document.getElementById('integral-results');
  if (!box || token !== state.detailLoadToken || state.currentDetailId !== Number(curveId)) return;
  box.innerHTML = '<span class="muted">Computing integral points…</span>';
  const canceled = () => token !== state.detailLoadToken || state.currentDetailId !== Number(curveId);
  try {
    const data = await apiPoints(curveId, 1, 0.55, canceled);
    if (!box || canceled()) return;
    box.innerHTML = pointsHtml(data);
  } catch (err) {
    if (!box || canceled()) return;
    box.innerHTML = '<span class="muted">Integral-point computation failed.</span>';
  }
}

async function computeSIntegral(curveId, token = state.detailLoadToken) {
  const input = document.getElementById('s-integral-input');
  const box = document.getElementById('s-integral-results');
  if (!input || !box) return;
  const S = Math.max(1, Math.abs(Math.floor(Number(input.value || 1))));
  const canceled = () => token !== state.detailLoadToken || state.currentDetailId !== Number(curveId);
  box.innerHTML = `<span class="muted">Computing S-integral points for S=${S}…</span>`;
  try {
    const data = await apiPoints(curveId, S, 3.0, canceled);
    if (!box || canceled()) return;
    box.innerHTML = pointsHtml(data);
  } catch (err) {
    if (!box || canceled()) return;
    box.innerHTML = '<span class="muted">S-integral computation failed.</span>';
  }
}
function memberTooltipLikeHtml(m, active=false) {
  state.detailMembers.set(String(m.id), m);
  const group = `E(Q) ≅ Z^${m.rank}${m.tor_label === '0' ? '' : ' ⊕ ' + m.tor_label}`;
  const weier = m.weierstrass_equation || weierstrassEquation(m);
  return `<button class="member member-rich ${active ? 'active' : ''}" data-id="${m.id}">
    <b>${escapeHtml(m.label || '')}</b>
    <span class="muted">${escapeHtml(m.cremona || '')} · ${escapeHtml(m.iso || '')} · N=${m.N} · rank ${m.rank} · ${escapeHtml(m.tor_label || '')}${m.cm ? ' · CM' : ''}</span>
    <span class="member-grid"><span class="tip-k">Weierstrass form</span><span class="code">${escapeHtml(weier)}</span><span class="tip-k">group</span><span>${escapeHtml(group)}</span><span class="tip-k">discriminant</span><span class="code">${escapeHtml(String(m.disc ?? ''))}</span><span class="tip-k">j-invariant</span><span class="code">${escapeHtml(String(m.j_str ?? ''))}</span></span>
  </button>`;
}
async function enrichCIsogenyDetails(items, shouldCancel = null) {
  const compact = items || [];
  if (!compact.length) return compact;
  const ids = compact.map(x => Number(x.id));
  const detailRows = await loadRowsByIds(ids, compact.length).catch(() => []);
  if (shouldCancel && shouldCancel()) return [];
  const byId = new Map(detailRows.map(r => [Number(r.id), enrichClientCurve(r)]));
  return compact.map(item => ({ ...(byId.get(Number(item.id)) || {}), ...item, ...(byId.get(Number(item.id)) || {}) }));
}
function cIsogenyMemberHtml(m) {
  state.detailMembers.set(String(m.id), m);
  const rel = m.relation || '';
  return `<button class="member ciso ciso-line" data-id="${m.id}"><b>${escapeHtml(m.label || '')}</b><span class="relation">${escapeHtml(rel)}</span></button>`;
}
function cIsogenyHtml(items) {
  const arr = (items || []).slice().sort((x,y) =>
    (Number(x.N || 0) - Number(y.N || 0)) ||
    (Number(x.height || 0) - Number(y.height || 0)) ||
    (Number(x.determinant || 0) - Number(y.determinant || 0)) ||
    String(x.label || '').localeCompare(String(y.label || ''))
  );
  if (!arr.length) return '<div class="muted">No small-height relation found.</div>';
  return arr.map(cIsogenyMemberHtml).join('');
}
function bindMemberButtons(root = document) {
  root.querySelectorAll('.member').forEach(el => {
    if (el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('click', () => travelAndOpen(Number(el.dataset.id), state.detailMembers.get(el.dataset.id)));
  });
}
async function fillCIsogenyNeighbours(d, token = state.detailLoadToken) {
  const box = document.getElementById('ciso-members');
  const id = Number(d.id);
  const canceled = () => token !== state.detailLoadToken || state.currentDetailId !== id;
  if (!box || canceled()) return;
  if (API_CACHE.cisoCache.has(id)) {
    box.innerHTML = cIsogenyHtml(API_CACHE.cisoCache.get(id));
    bindMemberButtons(box);
    return;
  }
  box.innerHTML = '<div class="muted">Loading C-isogeny index lazily…</div>';
  await idleYield(24);
  if (!box || canceled()) return;
  try {
    const detected = await detectedCIsogenyNeighbours(d, Infinity, canceled, (frac, count) => {
      if (!box || canceled()) return;
      box.innerHTML = `<div class="muted">Searching C-isogeny neighbours… ${Math.round(frac * 100)}%; found ${count}</div>`;
    });
    if (!box || canceled() || detected.canceled) return;
    if (!box || canceled()) return;
    // v28: C-isogeny rows only need label and τ transform, so skip the
    // neighbour detail-shard fan-out that made repeated selections sluggish.
    const items = detected.items || [];
    API_CACHE.cisoCache.set(id, items);
    box.innerHTML = cIsogenyHtml(items);
    bindMemberButtons(box);
  } catch (err) {
    if (box && !canceled()) box.innerHTML = '<div class="muted">C-isogeny neighbour search failed.</div>';
    console.warn('C-isogeny neighbour search failed', err);
  }
}
async function openCurve(id, animate = true) {
  id = Number(id);
  const token = ++state.detailLoadToken;
  state.currentDetailId = id;
  clearHoverState(true);
  let p = state.points.get(id);
  if (p) {
    state.selected = p;
    state.selectedPulseUntil = performance.now() + 1200;
    if (animate) animateToPoint(p);
  }
  $('detail').classList.remove('hidden');
  const content = $('detail-content');
  content.innerHTML = '<div class="card">Loading curve data…</div>';
  requestRender();
  const d = await apiCurve(id, 30);
  if (token !== state.detailLoadToken || state.currentDetailId !== id) return;
  if (d.error) { content.innerHTML = '<div class="card">Failed to load.</div>'; return; }
  if (Number(d.id) !== id) {
    id = Number(d.id);
    state.currentDetailId = id;
  }
  if (!p) {
    p = ensureMemberPoint(d);
    state.selected = p;
    state.selectedPulseUntil = performance.now() + 1200;
    if (animate) animateToPoint(p);
  }
  state.detailMembers.clear();
  const st = d.sato_tate || {}, moms = st.moments || {};
  const members = (d.members || []).map(m => memberTooltipLikeHtml(m, Number(m.id) === Number(d.id))).join('');
  const redRows = d.reduction_table.map(r => `<tr><td>${r.p}</td><td>${r.a_p}</td><td>${r.tamagawa ?? ''}</td><td>${kodairaHtml(r.kodaira)}</td><td title="${escapeHtml(localTypeTitle(r.type_label))}">${escapeHtml(r.type_label || '')}</td><td>${displayRootNumber(r.root_number)}</td></tr>`).join('');
  const coeffs = d.q_coefficients.map(r => `a_${r.n}=${r.a_n}`).join(', ');
  content.innerHTML = `
    <div class="card"><h2 class="title">${d.label}</h2>
      <div class="grid">
        <div class="k">Cremona</div><div>${d.cremona}</div><div class="k">Isogeny class</div><div>${d.iso}</div>
        <div class="k">Conductor</div><div>${d.N} = ${d.prime_signature}</div><div class="k">Largest prime factor</div><div>${d.largest_prime}</div>
        <div class="k">Discriminant</div><div class="code">${d.disc}</div><div class="k">Rank</div><div>${d.rank}</div><div class="k">Torsion</div><div>${d.tor_label}</div>
        <div class="k">Group</div><div class="code">E(Q) ≅ Z^${d.rank}${d.tor_label === '0' ? '' : ' ⊕ ' + d.tor_label}</div>
        <div class="k">CMdisc</div><div title="CM discriminant">${escapeHtml(formatCMDiscValue(d.cm_disc))}</div><div class="k">Integral points</div><div>${d.pts}</div><div class="k">Modular degree</div><div>${d.deg}</div>
        <div class="k">j-invariant</div><div class="code">${d.j_str}</div><div class="k">Weierstrass coefficients</div><div class="code">[${d.a1}, ${d.a2}, ${d.a3}, ${d.a4}, ${d.a6}]</div>
        <div class="k">Weierstrass equation</div><div class="code equation">${weierstrassEquation(d)}</div>
        <div class="k">Standard τ</div><div class="code">${Number(d.tau_re).toFixed(8)} + ${Number(d.tau_im).toFixed(8)}i</div>
        <div class="k">Real period Ω</div><div>${Number(d.real_period).toPrecision(10)}</div>
        <div class="k">frac(log₂ Ω)</div><div>${Number(d.period_phase).toFixed(6)}</div>
        <div class="k">Sky longitude</div><div>${Number(d.lon_deg).toFixed(3)}°</div><div class="k">Altitude</div><div>${Number(d.alt_deg).toFixed(3)}°</div>
      </div>
    </div>
    <div class="card"><div class="section">Sato-Tate group</div>
      <div class="grid"><div class="k">Label</div><div>${st.label || d.st}</div><div class="k">Name</div><div>${st.name || ''}</div><div class="k">G⁰</div><div>${st.identity_component || ''}</div><div class="k">G/G⁰</div><div>${st.component_group || ''}</div><div class="k">dimℝ</div><div>${st.dimR ?? ''}</div><div class="k">Pr[t=0]</div><div>${st.prob_t_eq_0 || ''}</div><div class="k">Maximal</div><div>${st.maximal ? '✓' : ' '}</div><div class="k">Rational</div><div>${st.rational ? '✓' : ' '}</div></div>
      <div class="section">Moments</div><div>${Object.entries(moms).map(([k,v]) => `<span class="pill">${k}=${v}</span>`).join('')}</div>
    </div>
    <div class="card"><div class="section">Integral and S-integral points</div>
      <div id="integral-results" class="points-box"><span class="muted">Computing integral points…</span></div>
      <div class="controls-row">
        <label class="muted">S</label>
        <input id="s-integral-input" class="num-input" type="number" min="1" step="1" value="1" />
        <button id="s-integral-run">Compute S-integral points</button>
      </div>
      <div id="s-integral-results" class="points-box"><span class="muted">Enter S and click the button to search S-integral points.</span></div>
    </div>
    <div class="detail-stack"><div class="card"><div class="section">Curves in ${d.iso}</div><div class="members members-rich">${members}</div></div><div class="card"><div class="section">C-isogeny neighbours <span class="muted">(sorted by conductor, then height)</span></div><div id="ciso-members" class="members ciso-list"><div class="muted">Preparing lazy C-isogeny search…</div></div></div></div>
    <div class="card"><div class="section">Invariants</div><div class="code">c4=${d.invariants.c4}\nc6=${d.invariants.c6}\nΔ=${d.invariants.disc}</div></div>
    <div class="card"><div class="section">Local data for primes p&lt;100</div><div class="tablewrap"><table><thead><tr><th>p</th><th>a_p</th><th>Tamagawa</th><th>Kodaira</th><th>type</th><th>root</th></tr></thead><tbody>${redRows}</tbody></table></div><p class="muted">s.sing. = supersingular; ord. = ordinary; s.mul. = split multiplicative; n.mul. = non-split multiplicative; add. = additive.</p></div>
    <div class="card"><div class="section">Modular form q-expansion</div><div class="code">${d.q_expansion}</div><p class="muted">${coeffs}</p></div>`;
  bindMemberButtons(content);
  const sBtn = document.getElementById('s-integral-run');
  if (sBtn) sBtn.addEventListener('click', () => computeSIntegral(d.id, token));
  requestRender();
  window.setTimeout(() => loadIntegralPoints(d.id, token), 0);
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 80));
  idle(() => { if (token === state.detailLoadToken && state.currentDetailId === Number(d.id)) fillCIsogenyNeighbours(d, token); }, { timeout: 500 });
}

function pointerLocal(e) {
  const rect = state.canvas.getBoundingClientRect();
  return { sx: e.clientX - rect.left, sy: e.clientY - rect.top };
}
function pointerMidpoint() {
  const pts = [...state.activePointers.values()];
  if (pts.length < 2) return null;
  return { x: (pts[0].x + pts[1].x) / 2, y: (pts[0].y + pts[1].y) / 2, dist: Math.hypot(pts[0].x - pts[1].x, pts[0].y - pts[1].y) };
}
const DETAIL_WHEEL_REENTRY_MS = 720;
const ATLAS_WHEEL_QUIET_MS = 180;
function markDetailWheelGesture() {
  const now = performance.now();
  state.detailWheelUntil = Math.max(state.detailWheelUntil || 0, now + DETAIL_WHEEL_REENTRY_MS);
}
function markAtlasReentryAfterDetailWheel() {
  const now = performance.now();
  if (now <= (state.detailWheelUntil || 0)) {
    state.atlasWheelSuppressUntil = Math.max(state.atlasWheelSuppressUntil || 0, now + DETAIL_WHEEL_REENTRY_MS);
  }
}
function shouldSuppressAtlasWheelAfterDetail() {
  const now = performance.now();
  if (now <= (state.detailWheelUntil || 0) || now <= (state.atlasWheelSuppressUntil || 0)) {
    // Treat touchpad inertia as one wheel gesture: keep suppressing until there
    // has been a short quiet gap, then ordinary atlas wheel-zoom resumes.
    state.atlasWheelSuppressUntil = now + ATLAS_WHEEL_QUIET_MS;
    return true;
  }
  return false;
}
function setupEvents() {
  state.canvas.addEventListener('pointerenter', () => {
    markAtlasReentryAfterDetailWheel();
  });
  state.canvas.addEventListener('pointerleave', () => {
    if (!state.activePointers.size) clearHoverState(true);
  });
  state.canvas.addEventListener('pointerdown', e => {
    e.preventDefault();
    const { sx, sy } = pointerLocal(e);
    // v31: remember the clicked star before entering drag/performance mode.
    // Some tiny stars drop out of the fast render set after pointerdown, so
    // pointerup must not depend only on the reduced drag-frame render list.
    const downTarget = activeHoverPointForClick(sx, sy) || nearest(sx, sy, 32, { preferSmall: true });
    clearHoverState(true);
    markInteraction(260);
    state.canvas.setPointerCapture?.(e.pointerId);
    state.activePointers.set(e.pointerId, { x:e.clientX, y:e.clientY, sx, sy });
    state.travel = null;
    if (state.activePointers.size === 1) {
      state.dragging = true;
      state.drag = { sx, sy, anchorWorld: screenToWorld(sx, sy) };
      state.pointerDown = { x: e.clientX, y: e.clientY, sx, sy, time: performance.now() };
      state.pointerDownTarget = downTarget;
      state.clickMoved = false;
    } else if (state.activePointers.size === 2) {
      const mid = pointerMidpoint();
      if (mid) {
        state.dragging = false;
        state.drag = null;
        state.pinch = { dist: Math.max(1, mid.dist), fov: state.fov, midX: mid.x, midY: mid.y };
        state.pointerDownTarget = null;
        state.clickMoved = true;
      }
    }
    requestRender();
  }, { passive:false });
  state.canvas.addEventListener('pointermove', e => {
    if (state.activePointers.has(e.pointerId)) {
      const { sx, sy } = pointerLocal(e);
      state.activePointers.set(e.pointerId, { x:e.clientX, y:e.clientY, sx, sy });
    }
    if (state.activePointers.size >= 2) {
      e.preventDefault();
      const mid = pointerMidpoint();
      if (mid && state.pinch) {
        const factor = Math.max(0.05, Math.min(80, mid.dist / state.pinch.dist));
        state.fov = clamp(state.pinch.fov / factor, 1e-9, 170 * RAD);
        markInteraction(240);
        updateVisibleTiles();
        requestRender();
      }
      return;
    }
    const { sx, sy } = pointerLocal(e);
    if (state.dragging) {
      e.preventDefault();
      const total = state.pointerDown ? Math.hypot(e.clientX - state.pointerDown.x, e.clientY - state.pointerDown.y) : 0;
      if (total > 4) state.clickMoved = true;
      panToScreenPoint(sx, sy);
      return;
    }
    if (e.pointerType !== 'touch') {
      const p = nearest(sx, sy, 24, { preferSmall: true });
      scheduleHover(p, e.clientX, e.clientY);
    }
  }, { passive:false });
  const finishPointer = e => {
    const wasPrimary = state.activePointers.size <= 1;
    state.activePointers.delete(e.pointerId);
    state.canvas.releasePointerCapture?.(e.pointerId);
    if (state.activePointers.size === 0) {
      state.dragging = false;
      state.drag = null;
      state.pinch = null;
      updateVisibleTiles(true);
      requestRender();
    } else if (state.activePointers.size === 1) {
      const pt = [...state.activePointers.values()][0];
      state.pinch = null;
      state.dragging = true;
      state.drag = { sx: pt.sx, sy: pt.sy, anchorWorld: screenToWorld(pt.sx, pt.sy) };
      state.pointerDown = { x: pt.x, y: pt.y, sx: pt.sx, sy: pt.sy, time: performance.now() };
      state.pointerDownTarget = null;
      state.clickMoved = true;
    }
    return wasPrimary;
  };
  state.canvas.addEventListener('pointerup', e => {
    const heldMs = state.pointerDown ? performance.now() - state.pointerDown.time : 0;
    const { sx, sy } = pointerLocal(e);
    const single = finishPointer(e);
    if (single && !state.clickMoved && heldMs <= 360) {
      const stored = state.pointerDownTarget;
      const p = activeHoverPointForClick(sx, sy) || stored || nearest(sx, sy, 30, { preferSmall: true });
      if (p) openCurve(p.i, true);
    }
    state.pointerDownTarget = null;
    state.clickMoved = false;
  }, { passive:false });
  state.canvas.addEventListener('pointercancel', e => { state.pointerDownTarget = null; finishPointer(e); }, { passive:false });
  state.canvas.addEventListener('wheel', e => {
    e.preventDefault();
    if (shouldSuppressAtlasWheelAfterDetail()) return;
    zoom(e.deltaY < 0 ? 1.06 : 1/1.06);
  }, { passive: false });
  document.addEventListener('gesturestart', e => e.preventDefault(), { passive:false });
  document.addEventListener('gesturechange', e => e.preventDefault(), { passive:false });
  const searchBoxForZoom = $('search');
  $('zoom-in').addEventListener('click', () => { if (document.activeElement === searchBoxForZoom && searchBoxForZoom.value.trim()) { searchBoxForZoom.focus(); return; } zoom(1.10); });
  $('zoom-out').addEventListener('click', () => { if (document.activeElement === searchBoxForZoom && searchBoxForZoom.value.trim()) { searchBoxForZoom.focus(); return; } zoom(1/1.10); });
  $('reset').addEventListener('click', resetView);
  $('close-detail').addEventListener('click', () => closeDetail(true));
  const detailPanel = $('detail');
  detailPanel.addEventListener('pointerenter', () => clearHoverState(true));
  detailPanel.addEventListener('pointermove', () => clearHoverState(true));
  detailPanel.addEventListener('wheel', markDetailWheelGesture, { passive: true });
  $('visible-level').addEventListener('change', e => {
    state.visibleLevel = Number(e.target.value);
    $('limit-value').textContent = '≈' + VISIBLE_LEVELS[state.visibleLevel];
    requestRender();
  });
  window.addEventListener('resize', () => { resize(); updateVisibleTiles(true); });
  window.addEventListener('keydown', e => {
    if (e.target && ['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
    if (e.key === 'Escape') closeDetail(true);
    else if (e.key === '0') resetView();
    else if (e.key === '+' || e.key === '=') zoom(1.12);
    else if (e.key === '-' || e.key === '_') zoom(1/1.12);
  });
}

function setupSearch() {
  let timer = null, seq = 0, activeSearchCancel = null;
  const cancelActiveSearch = () => { if (activeSearchCancel) activeSearchCancel.canceled = true; };
  const box = $('search'), res = $('search-results');
  if (!box || !res) return;

  // v28: detach the dropdown from the HUD/search wrapper and position it as a
  // fixed body-level overlay.  This avoids every ancestor containment/overflow
  // edge case and fixes the invisible-result bug.
  if (res.parentElement !== document.body) document.body.appendChild(res);

  function positionResults() {
    const rect = box.getBoundingClientRect();
    const margin = 10;
    const left = clamp(rect.left, margin, Math.max(margin, window.innerWidth - rect.width - margin));
    const width = Math.min(Math.max(rect.width, 360), window.innerWidth - margin * 2);
    const maxH = Math.max(120, window.innerHeight - rect.bottom - 18);
    res.style.left = `${left.toFixed(1)}px`;
    res.style.top = `${(rect.bottom + 7).toFixed(1)}px`;
    res.style.width = `${width.toFixed(1)}px`;
    res.style.maxHeight = `${Math.min(maxH, Math.round(window.innerHeight * 0.56), 520)}px`;
  }
  function showResults() { positionResults(); res.style.display = 'block'; }
  function hideResults() { res.style.display = 'none'; }

  let latestResultItems = new Map();
  function renderSearchItems(items, status = '') {
    latestResultItems = new Map((items || []).map(it => [Number(it.id), it]));
    if (!items.length) {
      res.innerHTML = status ? `<div class="result muted">${escapeHtml(status)}</div>` : '<div class="result muted">No matching curve found.</div>';
      showResults();
      return;
    }
    const statusHtml = status ? `<div class="result muted">${escapeHtml(status)}</div>` : '';
    res.innerHTML = statusHtml + items.map(it => {
      const match = it.search_match ? `<small>${escapeHtml(it.search_match)} · Δ=${escapeHtml(it.disc || '')} · j=${escapeHtml(it.j_str || '')}</small>` : '';
      return `<div class="result" data-id="${it.id}"><b>${escapeHtml(it.label)}</b><small>${escapeHtml(it.cremona || '')} · ${escapeHtml(it.iso || '')} · N=${it.N} · rank ${it.rank} · ${escapeHtml(it.tor_label || '')}${it.cm ? ' · CM' : ''}</small>${match}</div>`;
    }).join('');
    showResults();
  }

  const run = async () => {
    const q = box.value.trim();
    const mySeq = ++seq;
    cancelActiveSearch();
    const cancelToken = { canceled: false };
    activeSearchCancel = cancelToken;
    const searchCanceled = () => cancelToken.canceled || mySeq !== seq;
    if (!q) { cancelToken.canceled = true; hideResults(); res.innerHTML = ''; return; }
    res.innerHTML = '<div class="result muted">Searching…</div>';
    showResults();
    let latestItems = [];
    try {
      const listing = parseConductorListingQuery(q);
      const searchLimit = listing ? 1000 : 15;
      const items = await apiSearch(q, searchLimit, {
        batchSize: listing ? 10 : 5,
        shouldCancel: searchCanceled,
        onBatch: (partial) => {
          if (searchCanceled()) return;
          latestItems = partial;
          renderSearchItems(latestItems, `Searching… ${partial.length} result${partial.length === 1 ? '' : 's'} loaded`);
        }
      });
      if (searchCanceled()) return;
      latestItems = items;
      if (!items.length) {
        renderSearchItems([], 'No matching curve found.');
        return;
      }
      renderSearchItems(items, listing && items.length >= 1000 ? 'Showing first 1000 matching curves.' : '');
    } catch (err) {
      console.error('search failed', err);
      if (!searchCanceled()) {
        res.innerHTML = `<div class="result muted">Search failed: ${escapeHtml(err.message || err)}</div>`;
        showResults();
      }
    }
  };
  box.addEventListener('pointerdown', e => e.stopPropagation());
  box.addEventListener('click', e => e.stopPropagation());
  res.addEventListener('pointerdown', e => e.stopPropagation());
  res.addEventListener('click', e => {
    e.stopPropagation();
    const el = e.target && e.target.closest ? e.target.closest('.result[data-id]') : null;
    if (!el || !res.contains(el)) return;
    const id = Number(el.dataset.id);
    const item = latestResultItems.get(id);
    hideResults();
    box.value = item ? item.label : '';
    travelAndOpen(id, item);
  });
  box.addEventListener('input', () => {
    clearTimeout(timer);
    cancelActiveSearch();
    ++seq;
    const q = box.value.trim();
    if (!q) { hideResults(); res.innerHTML = ''; return; }
    positionResults();
    timer = setTimeout(run, 90);
  });
  box.addEventListener('keydown', e => {
    e.stopPropagation();
    if (e.key === 'Enter') { clearTimeout(timer); run(); }
    if (e.key === 'Escape') { cancelActiveSearch(); ++seq; hideResults(); box.blur(); }
  });
  box.addEventListener('focus', () => { if (res.innerHTML.trim()) showResults(); });
  window.addEventListener('resize', () => { if (res.style.display !== 'none') positionResults(); });
  window.addEventListener('scroll', () => { if (res.style.display !== 'none') positionResults(); }, true);
  document.addEventListener('click', e => {
    if (!res.contains(e.target) && e.target !== box) hideResults();
  });
}

async function init() {
  state.canvas = $('atlas');
  const versionBadge = document.getElementById('version-badge');
  if (versionBadge) versionBadge.textContent = EC_ATLAS_VERSION;
  state.tooltip = document.createElement('div');
  state.tooltip.className = 'tooltip';
  document.body.appendChild(state.tooltip);
  state.selectionEl = document.createElement('div');
  state.selectionEl.className = 'selection-marker';
  state.selectionEl.innerHTML = '<span class="selection-orbit"><span class="selection-tick t0"></span><span class="selection-tick t1"></span><span class="selection-tick t2"></span><span class="selection-tick t3"></span></span>';
  document.body.appendChild(state.selectionEl);
  const visibleSelect = $('visible-level');
  if (visibleSelect) { visibleSelect.value = String(state.visibleLevel); $('limit-value').textContent = '≈' + VISIBLE_LEVELS[state.visibleLevel]; }
  resize();
  setViewAzAlt(state.az, state.alt);
  setupEvents();
  setupSearch();
  state.meta = await apiMeta();
  await loadTop();
  $('loader').classList.add('hidden');
  updateVisibleTiles(true);
  preloadAllTiles();
  // v38: do not warm C-isogeny relation matrices on startup.  They are
  // generated lazily only after a detail panel requests the C-isogeny section,
  // avoiding idle main-thread work during initial exploration.
  requestRender();
}

init().catch(err => {
  console.error(err);
  $('loader').textContent = 'Failed to load atlas: ' + err.message;
});
