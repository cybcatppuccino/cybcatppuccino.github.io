import { identifyCubicJS } from './js/ec_core.js';

const EC_ATLAS_VERSION = 'v23';
const DATA_ROOT = new URL('./data/', import.meta.url);
const API_PROGRESS = { active: 0, text: '' };
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
  if (!res.body || !len) {
    setAtlasProgress(label || 'Loading data…', 0.35);
    return await res.text();
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    setAtlasProgress(`${label || 'Loading data'} ${(received / 1048576).toFixed(1)} MB`, Math.min(0.98, received / len));
  }
  const all = new Uint8Array(received);
  let pos = 0;
  for (const ch of chunks) { all.set(ch, pos); pos += ch.length; }
  return new TextDecoder().decode(all);
}

async function fetchJSONData(path, label) {
  const text = await fetchTextWithProgress(path, label);
  return JSON.parse(text);
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
  return /[\[\]=^*]/.test(t) || /[x-zuvw][a-z]?/i.test(t) || /[²³]/.test(t);
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
        const row = rowToObject(cols, arr);
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
  if (API_CACHE.rowsById.has(id)) return API_CACHE.rowsById.get(id);
  await loadCurveShardById(id);
  return API_CACHE.rowsById.get(id) || null;
}
async function loadRowsByIds(ids, limit = 20) {
  const out = [];
  const shardIds = new Map();
  for (const id0 of ids || []) {
    const id = Number(id0);
    if (API_CACHE.rowsById.has(id)) { out.push(API_CACHE.rowsById.get(id)); continue; }
    const shard = curveShardForId(id);
    if (!shardIds.has(shard)) shardIds.set(shard, []);
    shardIds.get(shard).push(id);
  }
  const shardGroups = [...shardIds.values()];
  const BATCH = 4;
  for (let i = 0; i < shardGroups.length; i += BATCH) {
    const groupBatch = shardGroups.slice(i, i + BATCH);
    await Promise.all(groupBatch.map(ids => loadCurveShardById(ids[0]).catch(err => console.warn('curve shard failed', ids[0], err))));
    for (const ids0 of groupBatch) {
      for (const id of ids0) {
        const row = API_CACHE.rowsById.get(id);
        if (row) out.push(row);
        if (out.length >= limit) return out;
      }
    }
    if (i + BATCH < shardGroups.length) await idleYield(8);
  }
  return out.slice(0, limit);
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
      const rows = (packed.rows || []).map(arr => rowToObject(cols, arr));
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
  let m = compact.match(/^(\d+)(?:[.][A-Za-z]|[a-zA-Z])/);
  if (m) return Number(m[1]);
  m = compact.match(/^N\s*=?\s*(\d+)$/i);
  if (m) return Number(m[1]);
  if (/^\d{1,4}$/.test(compact)) return Number(compact);
  return null;
}
async function loadTauRows() {
  if (API_CACHE.tauRows) return API_CACHE.tauRows;
  if (!API_CACHE.tauPromise) {
    API_CACHE.tauPromise = (async () => {
      const packed = await fetchJSONData('tau_index.json', 'Loading C-isogeny index');
      const cols = packed.columns || [];
      API_CACHE.tauRows = (packed.rows || []).map(arr => rowToObject(cols, arr));
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
function curveEqMod(d, x, y, p) {
  const a1=modSmall(d.a1,p), a2=modSmall(d.a2,p), a3=modSmall(d.a3,p), a4=modSmall(d.a4,p), a6=modSmall(d.a6,p);
  const lhs = (y*y + a1*x*y + a3*y) % p;
  const rhs = (x*x%p*x + a2*x*x + a4*x + a6) % p;
  return ((lhs - rhs) % p + p) % p;
}
function derivsMod(d, x, y, p) {
  const a1=modSmall(d.a1,p), a2=modSmall(d.a2,p), a3=modSmall(d.a3,p), a4=modSmall(d.a4,p);
  return [((a1*y - 3*x*x - 2*a2*x - a4) % p + p) % p, ((2*y + a1*x + a3) % p + p) % p];
}
function reductionData(d, bound = 100) {
  const inv = invariantsBigFromRow(d);
  const rows = [];
  for (const p of primesBelow(bound)) {
    let nonsingular = 0; const singular = [];
    for (let x=0; x<p; x++) for (let y=0; y<p; y++) if (curveEqMod(d,x,y,p) === 0) {
      const [fx, fy] = derivsMod(d,x,y,p);
      if (fx === 0 && fy === 0) singular.push([x,y]); else nonsingular += 1;
    }
    const smooth = nonsingular + 1;
    const vpDisc = vPBig(inv.disc, BigInt(p));
    const vpC4 = vPBig(inv.c4, BigInt(p));
    const ap = p + 1 - smooth;
    let reduction = 'good';
    if (vpDisc !== 0) reduction = ap === 1 ? 'split multiplicative' : ap === -1 ? 'nonsplit multiplicative' : 'additive';
    rows.push({ p, smooth_points: smooth, a_p: ap, reduction, vp_disc: vpDisc, vp_c4: vpC4, singular_points: singular });
  }
  return rows;
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
function newformCoefficients(d, bound = 30) {
  const local = new Map(reductionData(d, Math.max(bound + 1, 100)).map(row => [row.p, row]));
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
  for (const row of rows || []) {
    const key = `${Math.round(Number(row.tau_re) * scale)},${Math.round(Number(row.tau_im) * scale)}`;
    if (!buckets.has(key)) buckets.set(key, []);
    buckets.get(key).push(row);
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
function relationMatrices() {
  if (relationMatrices.cache) return relationMatrices.cache;
  const mats = [], H = 18, DET_MAX = 512;
  for (let a=-H; a<=H; a++) for (let b=-H; b<=H; b++) for (let c=-H; c<=H; c++) for (let d=-H; d<=H; d++) {
    if (a===0&&b===0&&c===0&&d===0) continue;
    const first = [a,b,c,d].find(x => x !== 0); if (first < 0) continue;
    const det = a*d - b*c; if (det <= 0) continue;
    if (Number(gcdBI(gcdBI(gcdBI(BigInt(Math.abs(a)), BigInt(Math.abs(b))), BigInt(Math.abs(c))), BigInt(Math.abs(d)))) !== 1) continue;
    const h = Math.max(Math.abs(a),Math.abs(b),Math.abs(c),Math.abs(d)); mats.push([a,b,c,d,h,det]);
  }
  mats.sort((x,y) => (x[4]-y[4]) || (x[5]-y[5]) || ((Math.abs(x[0])+Math.abs(x[1])+Math.abs(x[2])+Math.abs(x[3])) - (Math.abs(y[0])+Math.abs(y[1])+Math.abs(y[2])+Math.abs(y[3]))));
  relationMatrices.cache = mats; return mats;
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
async function detectedCIsogenyNeighbours(curve, limit = 120) {
  const buckets = await buildTauBuckets(); const tau = { re:Number(curve.tau_re), im:Number(curve.tau_im) }; const found = new Map();
  let checked = 0;
  for (const [a,b,c,d,h,det] of relationMatrices()) {
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
    if ((++checked & 2047) === 0) await new Promise(r => setTimeout(r, 0));
  }
  return [...found.values()].map(x => { delete x._score; return x; }).sort((x,y) =>
    (Number(x.N || 0) - Number(y.N || 0)) ||
    (Number(x.height || 0) - Number(y.height || 0)) ||
    (Number(x.determinant || 0) - Number(y.determinant || 0)) ||
    String(x.label).localeCompare(String(y.label))
  ).slice(0, limit);
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
async function apiSearch(q, limit = 15) {
  q = normalizeSearchText(q); if (!q) return [];
  const out = [], seen = new Set();
  async function addRows(rows, match) {
    for (const row of rows || []) {
      if (!row || seen.has(Number(row.id)) || out.length >= limit) continue;
      const d = enrichClientCurve(row); d.search_match = match; seen.add(Number(d.id)); out.push(d);
    }
  }
  async function addIds(ids, match) {
    const rows = await loadRowsByIds(ids || [], Math.max(limit * 2, 20));
    await addRows(rows, match);
  }

  // v23 fast path: exact label / Cremona / isogeny-class search uses a tiny
  // hash shard and avoids downloading 0.7-1.1 MB conductor buckets.
  const exactKey = compactExactKey(q);
  if (exactKey && !isCubicLikeQuery(q)) {
    const exactIds = await idsForExactKey(exactKey).catch(() => []);
    if (exactIds.length) {
      await addIds(exactIds, 'exact label / Cremona / isogeny class');
      if (out.length) return out;
    }
  }

  let cubic = null;
  try { cubic = identifyCubicJS(q); } catch (e) { cubic = null; }
  if (cubic && cubic.ok && cubic.j) {
    const ids = await idsForJ(cubic.j).catch(() => []);
    if (cubic.coeffs_int && ids.length) {
      const rows = await loadRowsByIds(ids, Math.max(limit * 3, 30));
      const want = JSON.stringify(cubic.coeffs_int.map(String));
      await addRows(rows.filter(r => JSON.stringify([r.a1,r.a2,r.a3,r.a4,r.a6].map(String)) === want), cubic.method || 'Q-minimal model from cubic input');
      if (out.length) return out;
      await addRows(rows, cubic.method || 'same j-invariant from cubic input');
    } else {
      await addIds(ids, cubic.method || 'same j-invariant from cubic input');
    }
    if (out.length) return out;
  }

  const rational = parseRationalKey(q);
  if (rational) {
    const maybeN = /^\d{1,5}$/.test(rational) ? Number(rational) : null;
    const [discIds, jIds, conductorIds] = await Promise.all([
      idsForDisc(rational).catch(() => []),
      idsForJ(rational).catch(() => []),
      maybeN != null ? idsForConductorFast(maybeN).catch(() => []) : Promise.resolve([]),
    ]);
    await addIds(conductorIds, 'exact conductor');
    await addIds(discIds, 'exact discriminant');
    await addIds(jIds, 'exact j-invariant');
    if (out.length) return out;
  }

  const needle = q.toLowerCase();
  const compactNeedle = q.replace(/\s+/g, '').toLowerCase();
  const conductor = parseConductorHint(q);
  const conductorOnly = /^(?:n=?)?\d{1,4}$/i.test(q.replace(/\s+/g, ''));
  const buckets = conductor != null ? [conductorBucketForN(conductor)] : [...API_CACHE.conductorRows.keys()];
  for (const bucket of buckets) {
    const rows = await loadConductorBucket(bucket).catch(() => []);
    const hits = rows.filter(row => String(row.label).toLowerCase() === compactNeedle || String(row.cremona).toLowerCase() === compactNeedle || String(row.iso).toLowerCase() === compactNeedle);
    const fuzzy = rows.filter(row => !hits.includes(row) && (String(row.label).toLowerCase().includes(needle) || String(row.cremona).toLowerCase().includes(needle) || String(row.iso).toLowerCase().includes(needle) || (conductorOnly && String(row.N) === String(conductor))));
    await addRows([...hits, ...fuzzy], 'label / Cremona / isogeny class');
    if (out.length >= limit) return out;
  }
  // Fast top-point fallback for non-numeric substring searches before scanning all conductor buckets.
  if (!out.length && state.points && needle.length >= 2) {
    const rows = [];
    for (const p of state.points.values()) {
      if (String(p.l).toLowerCase().includes(needle) || String(p.iso).toLowerCase().includes(needle)) {
        rows.push({ id:p.i, label:p.l, cremona:'', iso:p.iso, N:p.N, rank:p.r, tor_label:p.t, cm:p.cm, disc:'', j_str:'', az:p.az, alt:p.al });
        if (rows.length >= limit) break;
      }
    }
    await addRows(rows, 'visible star label / isogeny class');
  }
  // Last resort: scan conductor buckets lazily. This preserves v19 substring search without blocking ordinary exact searches.
  if (!out.length && needle.length >= 3) {
    for (let bucket = 0; bucket <= 9; bucket++) {
      if (conductor != null && bucket === conductorBucketForN(conductor)) continue;
      const rows = await loadConductorBucket(bucket).catch(() => []);
      await addRows(rows.filter(row => String(row.label).toLowerCase().includes(needle) || String(row.cremona).toLowerCase().includes(needle) || String(row.iso).toLowerCase().includes(needle)), 'label / Cremona / isogeny class');
      if (out.length >= limit) break;
    }
  }
  return out;
}
async function apiCurve(id, qBound = 30) {
  const row = await loadCurveById(id);
  if (!row) return { error: 'not found' };
  const d = enrichClientCurve(row);
  const stGroups = await loadSatoTateGroups().catch(() => ({}));
  let isoIds = await idsForExactKey(d.iso).catch(() => []);
  let NIds = await idsForConductorFast(d.N).catch(() => []);
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
    NIds = NRows.map(r => r.id);
  }
  d.sato_tate = stGroups[d.st] || { label: d.st };
  d.members = isoRows.map(makeCompactMember).sort((a,b) => (Number(a.class_index||0)-Number(b.class_index||0)) || String(a.label).localeCompare(String(b.label)));
  d.same_conductor_count = NIds.length || NRows.length;
  d.same_conductor = NRows.slice(0,240).map(makeCompactMember);
  d.c_isogeny = null;
  const inv = invariantsBigFromRow(d);
  d.invariants = Object.fromEntries(Object.entries(inv).map(([k,v]) => [k, String(v)]));
  d.reduction_table = reductionData(d, 100);
  const coeffs = newformCoefficients(d, qBound);
  d.q_coefficients = Array.from({ length: qBound }, (_, i) => ({ n: i+1, a_n: coeffs[i+1] || 0 }));
  d.q_expansion = formatQExpansion(coeffs, qBound);
  return d;
}
function isqrtBI(n) { if (n < 0n) return null; if (n < 2n) return n; let x0 = n, x1 = (n >> 1n) + 1n; while (x1 < x0) { x0 = x1; x1 = (x1 + n / x1) >> 1n; } return x0*x0 === n ? x0 : null; }
function formatQ(num, den = 1n) { num = BigInt(num); den = BigInt(den); if (den < 0n) { num=-num; den=-den; } const g = gcdBI(num,den); num/=g; den/=g; return den === 1n ? String(num) : `${num}/${den}`; }
function pointYFromY(a1,a3,xNum,d,Ynum) { const den = 2n*d**3n; let num = Ynum - a1*xNum*d - a3*d**3n; const g = gcdBI(absBI(num), den); return [num/g, den/g]; }
async function apiIntegralPoints(curveId, timeout = 1.0) {
  const curve = await loadCurveById(curveId); if (!curve) return { error:'curve not found' };
  const start = performance.now(); const inv = invariantsBigFromRow(curve); const a1=toBI(curve.a1), a3=toBI(curve.a3), b2=inv.b2, b4=inv.b4, b6=inv.b6; const target = Number(curve.pts || 0);
  if (target === 0) return { curve_id: curveId, label: curve.label, mode:'integral', target_count:0, points:[], count:0, searched_abs_x_up_to:0, complete:true, timed_out:false, elapsed_ms:Number((performance.now()-start).toFixed(1)), note:'stored integral-point count is zero.' };
  const found=[], seen=new Set(); let bound=256, searched=0, complete=false, timed_out=false;
  while (bound <= 2000000) {
    for (let x=-bound; x<=bound; x++) {
      if (Math.abs(x) <= searched) continue;
      const X = BigInt(x); const rhs = 4n*X*X*X + b2*X*X + 2n*b4*X + b6; const root = isqrtBI(rhs); if (root == null) continue;
      const Ys = root === 0n ? [0n] : [root, -root];
      for (const Y of Ys) { const yn = Y - a1*X - a3; if (yn % 2n !== 0n) continue; const y = yn/2n; const key = `${X},${y}`; if (!seen.has(key)) { seen.add(key); found.push({ x:String(X), y:String(y) }); } }
      if ((x & 2047) === 0) { if (performance.now() - start > timeout*1000) { timed_out = true; break; } await new Promise(r => setTimeout(r, 0)); }
    }
    searched = bound; if (target && found.length >= target) { complete = true; break; } if (timed_out) break; bound *= 2;
  }
  found.sort((a,b) => (a.x.length-b.x.length) || String(a.x).localeCompare(String(b.x)) || String(a.y).localeCompare(String(b.y)));
  return { curve_id:curveId, label:curve.label, mode:'integral', target_count:target, points:found, count:found.length, searched_abs_x_up_to:searched, complete:Boolean(complete || (target===0 && !found.length && searched>=2000000)), timed_out, elapsed_ms:Number((performance.now()-start).toFixed(1)), note:'complete=true means the stored integral-point count was reached; otherwise the list is a bounded-time search result.' };
}
function primeFactorsSmall(n) { return [...factorIntSmall(n).keys()]; }
function smoothNumbersFromPrimes(primes, limit) { const vals = new Set([1]); for (const p of primes) { const current = [...vals].sort((a,b)=>a-b); let mul = p; while (mul <= limit) { for (const v of current) { const nv = v*mul; if (nv <= limit) vals.add(nv); } mul *= p; } } return [...vals].sort((a,b)=>a-b); }
async function apiSIntegralPoints(curveId, S, timeout = 5.0) {
  const curve = await loadCurveById(curveId); if (!curve) return { error:'curve not found' };
  const primes = primeFactorsSmall(S); if (!primes.length) return apiIntegralPoints(curveId, Math.min(timeout,1.0));
  const start=performance.now(), maxD=primes.length<=2?80:50, xBound=1200, denominators=smoothNumbersFromPrimes(primes,maxD); const inv=invariantsBigFromRow(curve); const a1=toBI(curve.a1), a3=toBI(curve.a3), b2=inv.b2, b4=inv.b4, b6=inv.b6; const found=[], seen=new Set(); let checked=0, timed_out=false;
  for (const d0 of denominators) { const d=BigInt(d0), d2=d*d, d4=d**4n, d6=d**6n, mBound=xBound*d0*d0; for (let m=-mBound; m<=mBound; m++) { checked++; const M=BigInt(m); const rhs=4n*M*M*M + b2*M*M*d2 + 2n*b4*M*d4 + b6*d6; const Yroot=isqrtBI(rhs); if (Yroot != null) { const Ys=Yroot===0n?[0n]:[Yroot,-Yroot]; for (const Y of Ys) { let xNum=M, xDen=d2; const gx=gcdBI(absBI(xNum),xDen); xNum/=gx; xDen/=gx; const [yNum,yDen]=pointYFromY(a1,a3,M,d,Y); const key=`${xNum},${xDen},${yNum},${yDen}`; if(!seen.has(key)){seen.add(key); found.push({ x:formatQ(xNum,xDen), y:formatQ(yNum,yDen), den_x:Number(xDen<=9007199254740991n?xDen:9007199254740991n), den_y:Number(yDen<=9007199254740991n?yDen:9007199254740991n) });} } } if (checked % 2048 === 0) { if (performance.now()-start > timeout*1000) { timed_out=true; break; } await new Promise(r => setTimeout(r, 0)); } } if (timed_out) break; }
  found.sort((a,b) => (Math.max(a.den_x||1,a.den_y||1)-Math.max(b.den_x||1,b.den_y||1)) || (a.x.length-b.x.length) || String(a.x).localeCompare(String(b.x)) || String(a.y).localeCompare(String(b.y)));
  return { curve_id:curveId, label:curve.label, mode:'S-integral', S, S_primes:primes, points:found.slice(0,500), count:found.length, returned_count:Math.min(found.length,500), denominators_checked:denominators.length, max_denominator_checked:denominators[denominators.length-1] || 1, x_height_bound:xBound, timed_out, complete:false, elapsed_ms:Number((performance.now()-start).toFixed(1)), note:'S-integral search is bounded by time and height; timed_out=true or complete=false means additional points may be missing.' };
}
async function apiPoints(curveId, S = 1, timeout = 1.0) { return Number(S) === 1 ? apiIntegralPoints(curveId, Math.min(timeout,1.0)) : apiSIntegralPoints(curveId, Math.max(1, Math.floor(Number(S)||1)), timeout); }
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
  selectedProjection: null,
  selectionEl: null,
  detailIds: new Set(),
  hover: null,
  hoverMouse: null,
  hoverInfoCache: new Map(),
  hoverInfoPending: new Set(),
  selected: null,
  az: 0,
  alt: 89.7 * RAD,
  fov: 30 * RAD,
  initialFov: 30 * RAD,
  visibleLevel: 3,
  dragging: false,
  drag: null,
  tooltip: null,
  raf: 0,
  dirty: true,
  detailMembers: new Map(),
  travel: null,
  lastTileCheck: 0,
  pointerDown: null,
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
};

const VISIBLE_LEVELS = [180, 260, 360, 500, 700, 950, 1300, 1800, 2500, 3500];
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
  if (state.dirty || animating) {
    render(ts);
    state.dirty = false;
  }
  if (animating) state.raf = requestAnimationFrame(frame);
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
  const base = VISIBLE_LEVELS[state.visibleLevel] || 500;
  return Math.round(base + 170 * Math.pow(z, 1.05) + 80 * Math.sqrt(z));
}
function smoothScoreFromLargestPrime(lp) {
  return clamp((Math.log(10007) - Math.log(Math.max(lp, 2))) / (Math.log(10007) - Math.log(2)), 0, 1);
}
function starStops() {
  return [
    [0.00, [86, 120, 255]],
    [0.10, [108, 171, 255]],
    [0.24, [146, 221, 255]],
    [0.40, [244, 249, 255]],
    [0.58, [255, 242, 176]],
    [0.76, [255, 174, 92]],
    [0.90, [255, 98, 112]],
    [1.00, [224, 92, 255]],
  ];
}
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
  const stops = starStops();
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
  const map = {
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
  };
  return map[sig] || map['0'];
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
  const importance = clamp(0.64*smallPrimeBoost + 0.20*torNorm + 0.10*classNorm + 0.16*rankNorm + rankBoost + (cm ? 0.12 : 0), 0, 2.05);
  const brightness = clamp(0.008 + 2.2*Math.pow(importance, 2.25), 0.008, 2.4);
  const baseRadius = 0.34 + 3.75*Math.pow(importance, 1.40) + 0.16*Math.sqrt(torOrder) + 0.28*rankNorm + (cm ? 0.26 : 0);
  const priority = 9.0*importance + 0.75*rankNorm + 0.34*torNorm + 0.18*classNorm;

  // v12 normalization: color depends only on rank and torsion.
  const sig = torsionSignature(p.t, torOrder);
  const rankBase = {0:0.56, 1:0.70, 2:0.08, 3:0.02, 4:0.92, 5:0.84}[Math.min(rank, 5)] ?? 0.84;
  const torsionOffset = {
    '0':0.00,'n2':0.03,'n3':0.06,'n4':0.09,'n5':0.13,'n6':0.17,'n7':0.21,'n8':0.25,'n9':0.29,'n10':0.33,'n12':0.38,
    '2x2':0.45,'2x4':0.52,'2x6':0.60,'2x8':0.68,'prod':0.74
  }[sig] ?? 0.08;
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
  if (t >= 1) state.travel = null;
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
  const projected = pts.map(([az, alt]) => projectAzAlt(az, alt));
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  let started = false, last = null, first = null, visible = 0;
  const jumpLimit = opts.noSplit ? Infinity : Math.max(screen().w, screen().h) * 0.84;
  for (const p of projected) {
    if (!p) { if (!opts.bridgeGaps) { started = false; last = null; } continue; }
    visible++;
    if (last && Math.hypot(p.x - last.x, p.y - last.y) > jumpLimit) started = false;
    if (!started) { ctx.moveTo(p.x, p.y); if (!first) first = p; started = true; }
    else ctx.lineTo(p.x, p.y);
    last = p;
  }
  if (close && first && last && visible > projected.length * 0.62 && Math.hypot(first.x - last.x, first.y - last.y) < Math.max(screen().w, screen().h) * 1.3) ctx.closePath();
  ctx.stroke();
}

function drawProjectedClosedLoop(ctx, pts, strokeStyle, lineWidth = 1) {
  const projected = [];
  for (const pt of pts) {
    const pr = projectAzAlt(pt[0], pt[1]);
    if (pr) projected.push(pr);
  }
  if (projected.length < 8) return;
  ctx.strokeStyle = strokeStyle;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(projected[0].x, projected[0].y);
  for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i].x, projected[i].y);
  ctx.closePath();
  ctx.stroke();
}

function chooseNiceStep(spanDeg, targetLines = 6) {
  const opts = [45, 30, 20, 15, 12, 10, 7.5, 6, 5, 4, 3, 2, 1.5, 1, 0.5, 0.25];
  const maxStep = Math.max(0.25, spanDeg / targetLines);
  for (const s of opts) if (s <= maxStep) return s;
  return opts[opts.length - 1];
}

function gridStepDeg(fast = false) {
  const f = state.fov / RAD;
  const camAlt = vecToAzAlt(camera().f)[1] / RAD;
  const altSpan = Math.min(89, Math.max(10, f * 1.04));
  const targetAlt = fast ? 8 : (state.fov < 18 * RAD ? 13 : state.fov < 42 * RAD ? 11 : 9);
  const targetAz = fast ? 10 : (camAlt > 78 ? 16 : camAlt > 66 ? 13 : camAlt > 48 ? 11 : 9);
  const altStep = chooseNiceStep(altSpan, targetAlt);
  const azSpan = Math.min(360, Math.max(36, f * 1.62 / Math.max(0.12, Math.cos(camAlt * RAD))));
  let azStep = chooseNiceStep(azSpan, targetAz);
  if (fast) azStep = Math.max(azStep, 10);
  return { alt: altStep, az: azStep };
}

function uniqueSortedDegrees(vals) {
  return [...new Set(vals.map(v => Math.round(v * 1000) / 1000))].sort((a, b) => a - b);
}

function drawSkyGrid(ctx, fast = false) {
  ctx.save();
  ctx.globalCompositeOperation = 'source-over';
  ctx.shadowBlur = fast ? 1 : 5;
  ctx.shadowColor = 'rgba(226,136,84,0.16)';
  const step = gridStepDeg(fast);
  const topAltDeg = 89.35;
  const ringSamples = fast ? 240 : 384;
  const topSamples = fast ? 320 : 512;
  const latDegs = [];
  for (let altDeg = 0; altDeg < topAltDeg - step.alt * 0.35; altDeg += step.alt) latDegs.push(altDeg);
  latDegs.push(30, 45, 60, 75, topAltDeg);
  for (const altDeg of uniqueSortedDegrees(latDegs)) {
    if (altDeg < 0 || altDeg > topAltDeg) continue;
    const alt = Math.max(0.05, Math.min(topAltDeg, altDeg)) * RAD;
    const samples = Math.abs(altDeg - topAltDeg) < 1e-6 || altDeg > 75 ? topSamples : ringSamples;
    const pts = [];
    for (let i = 0; i < samples; i++) pts.push([i * TAU / samples, alt]);
    const top = Math.abs(altDeg - topAltDeg) < 1e-6;
    const major = top || altDeg === 0 || Math.abs((altDeg / Math.max(step.alt, 1e-6)) % 3) < 1e-6 || [30,45,60,75].includes(Math.round(altDeg));
    const color = top ? 'rgba(238,162,104,0.72)' : major ? 'rgba(218,132,82,0.58)' : 'rgba(198,116,72,0.30)';
    const width = top ? 1.8 : major ? 1.35 : 0.72;
    // Closed-loop drawing keeps high-latitude rings circular and prevents the
    // near-zenith grid from looking like an open spiral during motion.
    drawProjectedClosedLoop(ctx, pts, color, width);
  }

  const azDegs = [];
  for (let azDeg = 0; azDeg < 360; azDeg += step.az) azDegs.push(azDeg);
  azDegs.push(0, 90, 180, 270);
  const meridianSamples = fast ? 96 : 150;
  for (const azDeg of uniqueSortedDegrees(azDegs)) {
    if (azDeg >= 360) continue;
    const az = azDeg * RAD;
    const pts = [];
    for (let i = 0; i <= meridianSamples; i++) {
      const u = i / meridianSamples;
      const altDeg = 0.05 + (topAltDeg - 0.05) * u;
      pts.push([az, altDeg * RAD]);
    }
    const axis = azDeg === 0 || azDeg === 90 || azDeg === 180 || azDeg === 270;
    const major = axis || Math.abs((azDeg / Math.max(step.az, 1e-6)) % 3) < 1e-6;
    drawProjectedPath(ctx, pts, axis ? 'rgba(255,190,132,0.72)' : major ? 'rgba(218,132,82,0.47)' : 'rgba(198,116,72,0.24)', axis ? 1.55 : major ? 1.12 : 0.66, false, { bridgeGaps: true });
  }
  ctx.restore();
}

function pointRadius(p, proj) {
  const z = zoomLevel();
  const growth = 0.42 + (0.22 + 0.54*p.imp) * Math.pow(Math.max(z, 0) + 0.16, 1.09);
  const cap = 16 + 28*p.imp + 8*Math.sqrt(Math.max(0, z));
  const minR = 0.42 + Math.min(2.8, 0.22 * z);
  return clamp(p.baseR * growth * Math.pow(proj.z, 0.24), minR, cap);
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
  // Fallback for screenshots/tests; the live UI uses the DOM marker so the
  // selected-star animation no longer forces a full canvas repaint every frame.
  const t = now / 1000;
  const rot = t * 0.24;
  const rr = Math.max(10, r * 1.72 + 4.5);
  const tick = Math.max(3.0, Math.min(7.2, rr * 0.22));
  ctx.save();
  ctx.translate(proj.x, proj.y);
  ctx.rotate(rot);
  ctx.fillStyle = 'rgba(255,255,255,0.96)';
  ctx.beginPath();
  ctx.arc(0, 0, Math.max(3.2, Math.min(6.2, rr * 0.16)), 0, TAU);
  ctx.fill();
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
  const rr = Math.max(18, Math.min(54, sp.r * 3.05 + 15));
  const rect = state.canvas ? state.canvas.getBoundingClientRect() : { left: 0, top: 0 };
  el.style.setProperty('--sel-x', `${(rect.left + sp.pr.x).toFixed(2)}px`);
  el.style.setProperty('--sel-y', `${(rect.top + sp.pr.y).toFixed(2)}px`);
  el.style.setProperty('--sel-size', `${rr.toFixed(2)}px`);
  el.classList.add('visible');
}

function drawPointStar(ctx, p, proj, r, selected) {
  const glowR = r * (2.2 + 1.05 * Math.min(1.3, p.imp));
  const g = ctx.createRadialGradient(proj.x, proj.y, 0, proj.x, proj.y, glowR);
  g.addColorStop(0, rgba(p.glow, clamp(0.02 + 0.18 * p.bri, 0.02, 0.30)));
  g.addColorStop(0.18, rgba(p.glow, clamp(0.01 + 0.11 * p.bri, 0.01, 0.18)));
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, glowR, 0, TAU);
  ctx.fill();

  ctx.fillStyle = rgba(p.hot, clamp(0.10 + 0.78 * p.bri, 0.10, 1));
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, r, 0, TAU);
  ctx.fill();

  ctx.fillStyle = rgba([255,255,255], clamp(0.75 + 0.20 * p.bri, 0.7, 1));
  ctx.beginPath();
  ctx.arc(proj.x, proj.y, Math.max(0.55, r * 0.42), 0, TAU);
  ctx.fill();

}

function drawRays(ctx, p, proj, r) {
  const rays = Math.min(14, p.rayCount);
  const inner = r * (p.productStyle ? 1.55 : 1.62);
  const outer = inner + r * (0.62 + 0.36 * p.imp);
  const altLen = inner + r * (0.36 + 0.20 * p.imp);
  ctx.save();
  ctx.strokeStyle = rgba(p.glow, 0.24 + 0.16 * p.bri);
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

function drawDetailedCore(ctx, p, proj, r, selected) {
  const pts = p.shapeOrder;
  const z = zoomLevel();
  const outlineOnly = r > 5.8 || z > 2.1;

  const halo = ctx.createRadialGradient(proj.x, proj.y, 0, proj.x, proj.y, r * (2.3 + 1.1 * Math.min(1.25, p.imp)));
  halo.addColorStop(0, rgba(p.glow, clamp(0.05 + 0.12 * p.bri, 0.04, 0.18)));
  halo.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = halo;
  ctx.beginPath(); ctx.arc(proj.x, proj.y, r * (2.8 + 1.0 * p.imp), 0, TAU); ctx.fill();

  drawRays(ctx, p, proj, r);
  ctx.save();
  ctx.shadowBlur = 8 + 6 * p.imp;
  ctx.shadowColor = rgba(p.glow, 0.82);
  ctx.lineWidth = Math.max(1.0, r * 0.16);
  ctx.strokeStyle = rgba(p.hot, 0.96);
  ctx.fillStyle = rgba(p.rgb, outlineOnly ? 0.10 : 0.34);

  if (!pts) {
    ctx.beginPath();
    ctx.arc(proj.x, proj.y, r * 0.92, 0, TAU);
    if (!outlineOnly) ctx.fill();
    ctx.stroke();
  } else {
    const outer = r * 0.98;
    const inner = outer * (pts <= 4 ? 0.62 : pts <= 6 ? 0.50 : 0.42);
    starPolygonPath(ctx, proj.x, proj.y, outer, inner, pts, -Math.PI/2 + (p.shapeRot || 0));
    if (!outlineOnly) ctx.fill();
    ctx.stroke();
    if (p.productStyle && p.innerShapeOrder) {
      ctx.lineWidth = Math.max(0.9, r * 0.11);
      starPolygonPath(ctx, proj.x, proj.y, outer * 0.58, outer * 0.34, p.innerShapeOrder, -Math.PI/2 + (p.shapeRot || 0) + Math.PI/(2*Math.max(3,p.innerShapeOrder)));
      ctx.stroke();
    } else if (pts >= 8 || p.sig === 'n6') {
      ctx.lineWidth = Math.max(0.9, r * 0.11);
      starPolygonPath(ctx, proj.x, proj.y, outer * 0.58, outer * 0.30, Math.max(3, Math.min(pts, 6)), Math.PI / Math.max(3, pts));
      ctx.stroke();
    }
  }

  // Rank-dependent decorative rings.
  const rings = Math.max(0, Math.min(4, Number(p.rank || 0)));
  if (rings > 0) {
    ctx.shadowBlur = 0;
    ctx.strokeStyle = rgba(p.hot, 0.76);
    ctx.lineWidth = Math.max(0.7, r * 0.052);
    for (let k = 0; k < rings; k++) {
      ctx.beginPath();
      ctx.arc(proj.x, proj.y, r * (1.16 + 0.17 * k), 0, TAU);
      ctx.stroke();
    }
  }

  if (Number(p.cm)) {
    ctx.shadowBlur = 0;
    ctx.strokeStyle = 'rgba(246,214,111,0.95)';
    ctx.lineWidth = Math.max(0.8, r * 0.085);
    ctx.beginPath(); ctx.arc(proj.x, proj.y, r * 1.30, 0, TAU); ctx.stroke();
    for (let k = 0; k < 4; k++) {
      const ang = Math.PI/4 + k * Math.PI/2;
      ctx.beginPath();
      ctx.moveTo(proj.x + Math.cos(ang) * r * 1.40, proj.y + Math.sin(ang) * r * 1.40);
      ctx.lineTo(proj.x + Math.cos(ang) * r * 1.64, proj.y + Math.sin(ang) * r * 1.64);
      ctx.stroke();
    }
  }

  ctx.beginPath();
  ctx.arc(proj.x, proj.y, Math.max(0.55, r * 0.22), 0, TAU);
  ctx.fillStyle = rgba([255,255,255], 0.92);
  ctx.fill();

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
  for (const p of special) ids.add(p.i);
  const pool = [];
  for (const id of ids) {
    const p = state.points.get(id);
    if (!p) continue;
    const z = dot(p.vec, cam.f);
    if (z < cosLimit && !special.some(s => s && s.i === p.i)) continue;
    const xr = dot(p.vec, cam.right);
    const yu = dot(p.vec, cam.up);
    if (z <= 0) continue;
    const denom = Math.max(1e-9, 1 + z);
    const scale = projectionScale();
    const x = w / 2 + xr / denom * scale;
    const y = h / 2 - yu / denom * scale;
    if (x < -160 || x > w + 160 || y < -160 || y > h + 160) continue;
    const pr = { x, y, z, rho: Math.sqrt(xr*xr + yu*yu) / denom };
    const r = pointRadius(p, pr);
    const score = p.priority + 0.10 * zoomLevel() * p.imp + Math.min(0.55, r * 0.018);
    pool.push([p, pr, r, score]);
  }
  state.renderPool = pool;
  state.lastRenderKey = key;
  return pool;
}
function drawFastDot(ctx, p, pr, r, selected) {
  const rr = Math.max(0.75, Math.min(r, 7.4));
  ctx.fillStyle = rgba(p.hot, clamp(0.24 + 0.56 * p.bri, 0.22, 0.96));
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


function selectRenderable(now = performance.now()) {
  const pool = buildRenderPool(now);
  const interacting = isInteracting(now);
  const maxBase = dynamicMax();
  const max = interacting ? Math.max(90, Math.round(maxBase * 0.50)) : maxBase;
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
    for (const [p, pr, r] of state.rendered) {
      const score = r - detailThreshold(p);
      if (score > 0) detailCandidates.push([score + 0.72 * p.imp, p.i]);
      if (state.selected && p.i === state.selected.i) state.selectedProjection = { pr, r };
    }
  } else if (state.selected) {
    for (const [p, pr, r] of state.rendered) {
      if (p.i === state.selected.i) { state.selectedProjection = { pr, r }; break; }
    }
  }
  detailCandidates.sort((a, b) => b[0] - a[0]);
  const visibleCount = state.rendered.length;
  const detailLimit = interacting ? 0 : (visibleCount <= 45 ? Math.min(34, visibleCount) : visibleCount <= 120 ? 26 : visibleCount <= 300 ? 18 : 12);
  state.detailIds = new Set(detailCandidates.slice(0, detailLimit).map(v => v[1]));
  if (state.selected) state.detailIds.add(state.selected.i);
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
}

function nearest(sx, sy, threshold = 18) {
  let best = null, bestD = threshold;
  for (const [p, pr, r0] of state.rendered) {
    const r = r0 || pointRadius(p, pr);
    const d = Math.hypot(pr.x - sx, pr.y - sy) - Math.max(2.2, r);
    if (d < bestD) {
      bestD = d;
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
  const pts = data.points || [];
  const rows = pts.length ? pts.map(pt => `<span class="pill">(${pt.x}, ${pt.y})</span>`).join('') : '<span class="muted">No points found in the current search range.</span>';
  const status = data.complete ? 'complete' : (data.timed_out ? 'timed out; may be incomplete' : 'bounded search; may be incomplete');
  return `<div>${rows}</div><p class="muted">count=${data.count ?? pts.length}; ${status}; elapsed=${data.elapsed_ms} ms. ${data.note || ''}</p>`;
}

async function loadIntegralPoints(curveId) {
  const box = document.getElementById('integral-results');
  if (!box) return;
  box.innerHTML = '<span class="muted">Computing integral points…</span>';
  try {
    const data = await apiPoints(curveId, 1, 1);
    box.innerHTML = pointsHtml(data);
  } catch (err) {
    box.innerHTML = '<span class="muted">Integral-point computation failed.</span>';
  }
}

async function computeSIntegral(curveId) {
  const input = document.getElementById('s-integral-input');
  const box = document.getElementById('s-integral-results');
  if (!input || !box) return;
  const S = Math.max(1, Math.floor(Number(input.value || 1)));
  box.innerHTML = `<span class="muted">Computing S-integral points for S=${S}…</span>`;
  try {
    const data = await apiPoints(curveId, S, 5);
    box.innerHTML = pointsHtml(data);
  } catch (err) {
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
function cIsogenyHtml(items) {
  const arr = (items || []).slice().sort((x,y) =>
    (Number(x.N || 0) - Number(y.N || 0)) ||
    (Number(x.height || 0) - Number(y.height || 0)) ||
    String(x.label || '').localeCompare(String(y.label || ''))
  );
  if (!arr.length) return '<div class="muted">No small-height relation found.</div>';
  return arr.map(m => {
    state.detailMembers.set(String(m.id), m);
    return `<button class="member ciso ciso-compact" data-id="${m.id}"><b>${escapeHtml(m.label || '')}</b> <span class="relation">${escapeHtml(m.relation || '')}</span></button>`;
  }).join('');
}
function bindMemberButtons(root = document) {
  root.querySelectorAll('.member').forEach(el => {
    if (el.dataset.bound) return;
    el.dataset.bound = '1';
    el.addEventListener('click', () => travelAndOpen(Number(el.dataset.id), state.detailMembers.get(el.dataset.id)));
  });
}
async function fillCIsogenyNeighbours(d) {
  const box = document.getElementById('ciso-members');
  if (!box || state.currentDetailId !== Number(d.id)) return;
  box.innerHTML = '<div class="muted">Loading C-isogeny index lazily…</div>';
  try {
    const items = await detectedCIsogenyNeighbours(d);
    if (!box || state.currentDetailId !== Number(d.id)) return;
    box.innerHTML = cIsogenyHtml(items);
    bindMemberButtons(box);
  } catch (err) {
    if (box) box.innerHTML = '<div class="muted">C-isogeny neighbour search failed.</div>';
    console.warn('C-isogeny neighbour search failed', err);
  }
}
async function openCurve(id, animate = true) {
  id = Number(id);
  state.currentDetailId = id;
  let p = state.points.get(id);
  if (p) { state.selected = p; state.selectedPulseUntil = performance.now() + 1200; if (animate) animateToPoint(p); }
  $('detail').classList.remove('hidden');
  const content = $('detail-content');
  content.innerHTML = '<div class="card">Loading curve data…</div>';
  requestRender();
  const d = await apiCurve(id, 30);
  if (d.error) { content.innerHTML = '<div class="card">Failed to load.</div>'; return; }
  if (state.currentDetailId !== id) return;
  if (!p) { p = ensureMemberPoint(d); state.selected = p; state.selectedPulseUntil = performance.now() + 1200; if (animate) animateToPoint(p); }
  state.detailMembers.clear();
  const st = d.sato_tate || {}, moms = st.moments || {};
  const members = (d.members || []).map(m => memberTooltipLikeHtml(m, Number(m.id) === Number(d.id))).join('');
  const redRows = d.reduction_table.map(r => `<tr><td>${r.p}</td><td>${r.smooth_points}</td><td>${r.a_p}</td><td>${r.reduction}</td><td>${r.vp_disc}</td><td>${r.vp_c4}</td></tr>`).join('');
  const coeffs = d.q_coefficients.map(r => `a_${r.n}=${r.a_n}`).join(', ');
  content.innerHTML = `
    <div class="card"><h2 class="title">${d.label}</h2>
      <div class="grid">
        <div class="k">Cremona</div><div>${d.cremona}</div><div class="k">Isogeny class</div><div>${d.iso}</div>
        <div class="k">Conductor</div><div>${d.N} = ${d.prime_signature}</div><div class="k">Largest prime factor</div><div>${d.largest_prime}</div>
        <div class="k">Discriminant</div><div class="code">${d.disc}</div><div class="k">Rank</div><div>${d.rank}</div><div class="k">Torsion</div><div>${d.tor_label}</div>
        <div class="k">Group</div><div class="code">E(Q) ≅ Z^${d.rank}${d.tor_label === '0' ? '' : ' ⊕ ' + d.tor_label}</div>
        <div class="k">CM</div><div>${d.cm}</div><div class="k">Integral points</div><div>${d.pts}</div><div class="k">Modular degree</div><div>${d.deg}</div>
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
    <div class="card"><div class="section">Local data for primes p&lt;100</div><div class="tablewrap"><table><thead><tr><th>p</th><th>#smooth</th><th>a_p</th><th>reduction</th><th>v_p(Δ)</th><th>v_p(c4)</th></tr></thead><tbody>${redRows}</tbody></table></div></div>
    <div class="card"><div class="section">Modular form q-expansion</div><div class="code">${d.q_expansion}</div><p class="muted">${coeffs}</p></div>`;
  bindMemberButtons(content);
  const sBtn = document.getElementById('s-integral-run');
  if (sBtn) sBtn.addEventListener('click', () => computeSIntegral(d.id));
  loadIntegralPoints(d.id);
  // Defer the 4.7 MB τ-index path so the main curve card is interactive first.
  // This keeps the C-isogeny feature intact while avoiding random-click stalls.
  const idle = window.requestIdleCallback || (fn => setTimeout(fn, 250));
  idle(() => setTimeout(() => fillCIsogenyNeighbours(d), 900), { timeout: 2600 });
  requestRender();
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
function setupEvents() {
  state.canvas.addEventListener('pointerdown', e => {
    e.preventDefault();
    markInteraction(260);
    state.canvas.setPointerCapture?.(e.pointerId);
    const { sx, sy } = pointerLocal(e);
    state.activePointers.set(e.pointerId, { x:e.clientX, y:e.clientY, sx, sy });
    state.travel = null;
    if (state.activePointers.size === 1) {
      state.dragging = true;
      state.drag = { sx, sy, anchorWorld: screenToWorld(sx, sy) };
      state.pointerDown = { x: e.clientX, y: e.clientY, time: performance.now() };
      state.clickMoved = false;
    } else if (state.activePointers.size === 2) {
      const mid = pointerMidpoint();
      if (mid) {
        state.dragging = false;
        state.drag = null;
        state.pinch = { dist: Math.max(1, mid.dist), fov: state.fov, midX: mid.x, midY: mid.y };
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
      const p = nearest(sx, sy, 22);
      if ((p && !state.hover) || (!p && state.hover) || (p && state.hover && p.i !== state.hover.i)) {
        state.hover = p;
        showTooltip(p, e.clientX, e.clientY);
        requestRender();
      } else if (p) showTooltip(p, e.clientX, e.clientY);
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
      state.pointerDown = { x: pt.x, y: pt.y, time: performance.now() };
      state.clickMoved = true;
    }
    return wasPrimary;
  };
  state.canvas.addEventListener('pointerup', e => {
    const heldMs = state.pointerDown ? performance.now() - state.pointerDown.time : 0;
    const { sx, sy } = pointerLocal(e);
    const single = finishPointer(e);
    if (single && !state.clickMoved && heldMs <= 360) {
      const p = nearest(sx, sy, 26);
      if (p) openCurve(p.i, true);
    }
    state.clickMoved = false;
  }, { passive:false });
  state.canvas.addEventListener('pointercancel', finishPointer, { passive:false });
  state.canvas.addEventListener('wheel', e => {
    e.preventDefault();
    zoom(e.deltaY < 0 ? 1.06 : 1/1.06);
  }, { passive: false });
  document.addEventListener('gesturestart', e => e.preventDefault(), { passive:false });
  document.addEventListener('gesturechange', e => e.preventDefault(), { passive:false });
  $('zoom-in').addEventListener('click', () => zoom(1.10));
  $('zoom-out').addEventListener('click', () => zoom(1/1.10));
  $('reset').addEventListener('click', resetView);
  $('close-detail').addEventListener('click', () => closeDetail(true));
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
  let timer = null, seq = 0;
  const box = $('search'), res = $('search-results');
  const run = async () => {
    const q = box.value.trim();
    const mySeq = ++seq;
    if (!q) { res.style.display = 'none'; res.innerHTML = ''; return; }
    res.innerHTML = '<div class="result muted">Searching…</div>';
    res.style.display = 'block';
    try {
      const items = await apiSearch(q);
      if (mySeq !== seq) return;
      if (!items.length) {
        res.innerHTML = '<div class="result muted">No matching curve found.</div>';
        return;
      }
      res.innerHTML = items.map(it => {
        const match = it.search_match ? `<small>${escapeHtml(it.search_match)} · Δ=${escapeHtml(it.disc || '')} · j=${escapeHtml(it.j_str || '')}</small>` : '';
        return `<div class="result" data-id="${it.id}"><b>${escapeHtml(it.label)}</b><small>${escapeHtml(it.cremona || '')} · ${escapeHtml(it.iso || '')} · N=${it.N} · rank ${it.rank} · ${escapeHtml(it.tor_label || '')}${it.cm ? ' · CM' : ''}</small>${match}</div>`;
      }).join('');
      res.style.display = 'block';
      res.querySelectorAll('.result').forEach(el => el.addEventListener('click', () => {
        const item = items.find(x => Number(x.id) === Number(el.dataset.id));
        res.style.display = 'none';
        box.value = item ? item.label : '';
        travelAndOpen(Number(el.dataset.id), item);
      }));
    } catch (err) {
      console.error('search failed', err);
      if (mySeq === seq) res.innerHTML = `<div class="result muted">Search failed: ${escapeHtml(err.message || err)}</div>`;
    }
  };
  box.addEventListener('input', () => {
    clearTimeout(timer);
    const q = box.value.trim();
    if (!q) { res.style.display = 'none'; res.innerHTML = ''; return; }
    timer = setTimeout(run, 120);
  });
  box.addEventListener('keydown', e => {
    if (e.key === 'Enter') { clearTimeout(timer); run(); }
    if (e.key === 'Escape') { res.style.display = 'none'; box.blur(); }
  });
  document.addEventListener('click', e => {
    if (!res.contains(e.target) && e.target !== box) res.style.display = 'none';
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
  state.selectionEl.innerHTML = '<span class="selection-dot"></span><span class="selection-tick t0"></span><span class="selection-tick t1"></span><span class="selection-tick t2"></span><span class="selection-tick t3"></span>';
  document.body.appendChild(state.selectionEl);
  resize();
  setViewAzAlt(state.az, state.alt);
  setupEvents();
  setupSearch();
  state.meta = await apiMeta();
  await loadTop();
  $('loader').classList.add('hidden');
  updateVisibleTiles(true);
  preloadAllTiles();
  requestRender();
}

init().catch(err => {
  console.error(err);
  $('loader').textContent = 'Failed to load atlas: ' + err.message;
});
