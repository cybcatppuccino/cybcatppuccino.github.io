(() => {
  const MANIFEST = window.SUNIT_MANIFEST;
  const DATA = window.SUNIT_DATASETS || (window.SUNIT_DATASETS = {});
  const FULL_DATA = window.SUNIT_FULL_DATASETS || (window.SUNIT_FULL_DATASETS = {});
  const ORDER = MANIFEST.order;
  const META = MANIFEST.datasets;
  const COLORS = ['#2563EB', '#E5484D', '#D97706', '#7C5CFC'];
  const TWO_PI = Math.PI * 2;
  const LAYOUT_X_SPAN = 300;
  const LAYOUT_Y_SCALE = 50;
  const SIZE_WARP_POWER = 0.82;
  const BUCKET_SIZE = 7.5;
  const DPR_CAP = 2;

  const home = document.getElementById('home');
  const viewer = document.getElementById('viewer');
  const selector = document.getElementById('selector');
  const back = document.getElementById('back');
  const zoomIn = document.getElementById('zoom-in');
  const zoomOut = document.getElementById('zoom-out');
  const stage = document.getElementById('stage');
  const canvas = document.getElementById('canvas');
  const overlay = document.getElementById('overlay');
  const detail = document.getElementById('detail');
  const detailClose = document.getElementById('detail-close');
  const detailExpression = document.getElementById('detail-expression');
  const detailFactor = document.getElementById('detail-factor');
  const tools = document.getElementById('tools');
  const toolButtons = tools.querySelector('.tool-buttons');
  const primeToggle = document.getElementById('prime-toggle');
  const searchToggle = document.getElementById('search-toggle');
  const maxToggle = document.getElementById('max-toggle');
  const minToggle = document.getElementById('min-toggle');
  const downloadButton = document.getElementById('download');
  const primePanel = document.getElementById('prime-panel');
  const searchPanel = document.getElementById('search-panel');
  const maxPanel = document.getElementById('max-panel');
  const minPanel = document.getElementById('min-panel');
  const primeOptions = document.getElementById('prime-options');
  const primeAll = document.getElementById('prime-all');
  const primeNone = document.getElementById('prime-none');
  const searchInput = document.getElementById('search-input');
  const maxInput = document.getElementById('max-input');
  const minInput = document.getElementById('min-input');
  const solutionCount = document.getElementById('solution-count');
  const searchResults = document.getElementById('search-results');
  const ctx = canvas.getContext('2d', { alpha: true, desynchronized: true });
  const overlayCtx = overlay.getContext('2d', { alpha: true, desynchronized: true });

  const datasetCache = new Map();
  const loadCache = new Map();
  const fullLoadCache = new Map();
  const choiceButtons = new Map();
  let openRequest = 0;
  let currentKey = null;
  let allPoints = [];
  let points = [];
  let activePrimeMask = 0;
  let searchTerms = [];
  let maxTerm = null;
  let minTerm = null;
  let spatial = new Map();
  let bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
  let width = 0, height = 0, dpr = 1;
  let zoom = 1, minZoom = 1, maxZoom = 30;
  let viewStretchX = 1, viewStretchY = 1;
  let panX = 0, panY = 0;
  let dragging = false;
  let didDrag = false;
  let dragPointerId = null;
  let dragX = 0, dragY = 0, dragPanX = 0, dragPanY = 0;
  let hoverPoint = null;
  let selectedPoint = null;
  let framePending = false;
  let overlayPending = false;
  let resizeTimer = 0;

  for (const key of ORDER) {
    const d = META[key];
    const b = document.createElement('button');
    b.className = 'choice';
    b.innerHTML = `<span class="choice-eq">${d.equation}</span><span class="choice-meta"><span class="choice-prime">${d.primeDisplay}</span><span class="choice-count">${Number(d.totalCount ?? 0).toLocaleString("en-US")}</span></span>`;
    b.addEventListener('click', () => openDataset(key));
    choiceButtons.set(key, b);
    selector.appendChild(b);
  }

  function loadDataset(key) {
    if (DATA[key]) return Promise.resolve(DATA[key]);
    if (loadCache.has(key)) return loadCache.get(key);
    const meta = META[key];
    if (!meta) return Promise.reject(new Error(`Unknown dataset: ${key}`));
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = meta.file;
      script.async = true;
      script.onload = () => DATA[key] ? resolve(DATA[key]) : reject(new Error(`Dataset did not register: ${key}`));
      script.onerror = () => reject(new Error(`Could not load ${meta.file}`));
      document.head.appendChild(script);
    });
    loadCache.set(key, promise);
    return promise;
  }

  function loadFullDataset(key) {
    const meta = META[key];
    if (!meta || !meta.fullFile || (DATA[key] && DATA[key].partialMax == null && DATA[key].partialMin == null)) return Promise.resolve(DATA[key]);
    if (fullLoadCache.has(key)) return fullLoadCache.get(key);
    const promise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = meta.fullFile;
      script.async = true;
      script.onload = () => {
        if (!FULL_DATA[key]) { reject(new Error(`Dataset did not register: ${key}`)); return; }
        DATA[key] = FULL_DATA[key];
        datasetCache.delete(key);
        resolve(DATA[key]);
      };
      script.onerror = () => reject(new Error(`Could not load ${meta.fullFile}`));
      document.head.appendChild(script);
    });
    fullLoadCache.set(key, promise);
    return promise;
  }

  function needsFullDataset(key) {
    const meta = META[key];
    const d = DATA[key];
    if (!meta || !meta.fullFile || !d) return false;
    if (d.partialMax != null) {
      if (maxTerm === null || !decimalLE(maxTerm, decimalString(d.partialMax))) return true;
    }
    if (d.partialMin != null) {
      if (minTerm === null || !decimalGE(minTerm, decimalString(d.partialMin))) return true;
    }
    return false;
  }

  function valueLog1p(value) {
    if (typeof value === 'number') return Math.log1p(value);
    const s = String(value);
    if (s.length < 290) return Math.log1p(Number(s));
    const take = Math.min(16, s.length);
    const lead = Number(s.slice(0, take)) / Math.pow(10, take - 1);
    return (s.length - 1) * Math.LN10 + Math.log(lead);
  }

  function logVector(row) {
    return row.map(valueLog1p);
  }

  // Layout rule (no randomness, no density field):
  // 1) overall logarithmic magnitude determines the strict left-to-right order;
  // 2) the dominant variation of the relative log proportions determines y.
  // The silhouette therefore comes from the solution set itself rather than
  // from a prescribed rectangle, spiral, grid, or noise process.
  function buildOrderedDataLayout(rows) {
    const n = rows.length;
    const dims = rows[0].length;
    const entries = new Array(n);
    const residualMean = new Array(dims).fill(0);

    for (let i = 0; i < n; i++) {
      const logs = logVector(rows[i]);
      let square = 0;
      let mean = 0;
      for (let j = 0; j < dims; j++) {
        square += logs[j] * logs[j];
        mean += logs[j];
      }
      mean /= dims;
      const residual = new Array(dims);
      for (let j = 0; j < dims; j++) {
        residual[j] = logs[j] - mean;
        residualMean[j] += residual[j];
      }
      entries[i] = {
        sourceIndex: i,
        size: Math.sqrt(square / dims),
        residual
      };
    }

    for (let j = 0; j < dims; j++) residualMean[j] /= n;

    // Principal direction of the scale-free residual vectors.  A short power
    // iteration is enough in 3D/4D and avoids any arbitrary visual jitter.
    const covariance = Array.from({ length: dims }, () => new Array(dims).fill(0));
    for (const e of entries) {
      for (let a = 0; a < dims; a++) {
        const va = e.residual[a] - residualMean[a];
        for (let b = 0; b < dims; b++) {
          covariance[a][b] += va * (e.residual[b] - residualMean[b]);
        }
      }
    }

    let axis = Array.from({ length: dims }, (_, j) => j - (dims - 1) * 0.5);
    for (let iter = 0; iter < 28; iter++) {
      const next = new Array(dims).fill(0);
      for (let a = 0; a < dims; a++) {
        for (let b = 0; b < dims; b++) next[a] += covariance[a][b] * axis[b];
      }
      const norm = Math.hypot(...next) || 1;
      axis = next.map(v => v / norm);
    }
    // Stable orientation across reloads: the first meaningful coefficient is negative.
    const first = axis.find(v => Math.abs(v) > 1e-8) || 1;
    if (first > 0) axis = axis.map(v => -v);

    let shapeSquare = 0;
    for (const e of entries) {
      let shape = 0;
      for (let j = 0; j < dims; j++) shape += (e.residual[j] - residualMean[j]) * axis[j];
      e.shape = shape;
      shapeSquare += shape * shape;
    }
    const shapeStd = Math.sqrt(shapeSquare / Math.max(1, n)) || 1;

    const ordered = entries.slice().sort((a, b) => a.size - b.size || a.sourceIndex - b.sourceIndex);
    const rank = new Int32Array(n);
    for (let i = 0; i < n; i++) rank[ordered[i].sourceIndex] = i;

    const positions = new Array(n);
    const denom = Math.max(1, n - 1);
    for (const e of entries) {
      const t = rank[e.sourceIndex] / denom;
      // A mild concave size map gives the smallest solutions more horizontal
      // breathing room while preserving the strict small-to-large order.
      const u = Math.pow(t, SIZE_WARP_POWER);
      positions[e.sourceIndex] = {
        x: (u - 0.5) * LAYOUT_X_SPAN,
        y: Math.asinh(e.shape / shapeStd) * LAYOUT_Y_SCALE,
        size: e.size
      };
    }
    return positions;
  }

  function buildScalableDataLayout(rows) {
    const n = rows.length;
    const dims = rows[0].length;
    const sizes = new Float32Array(n);
    const rawY = new Float32Array(n);
    let ySquare = 0;
    let axis = Array.from({ length: dims }, (_, j) => j - (dims - 1) * 0.5);
    const axisNorm = Math.hypot(...axis) || 1;
    axis = axis.map(v => v / axisNorm);

    for (let i = 0; i < n; i++) {
      const row = rows[i];
      const logs = new Array(dims);
      let mean = 0, square = 0;
      for (let j = 0; j < dims; j++) {
        const v = valueLog1p(row[j]);
        logs[j] = v;
        mean += v;
        square += v * v;
      }
      mean /= dims;
      let shape = 0;
      for (let j = 0; j < dims; j++) shape += (logs[j] - mean) * axis[j];
      sizes[i] = Math.sqrt(square / dims);
      rawY[i] = shape;
      ySquare += shape * shape;
    }

    const shapeStd = Math.sqrt(ySquare / Math.max(1, n)) || 1;
    const xs = new Float32Array(n);
    const ys = new Float32Array(n);
    const denom = Math.max(1, n - 1);
    for (let i = 0; i < n; i++) {
      // The seven-variable source files are already sorted by (max(row), row).
      // Using that order avoids a million-element browser sort while keeping
      // the horizontal direction strictly small-to-large by the row maximum.
      const t = i / denom;
      xs[i] = (Math.pow(t, SIZE_WARP_POWER) - 0.5) * LAYOUT_X_SPAN;
      ys[i] = Math.asinh(rawY[i] / shapeStd) * LAYOUT_Y_SCALE;
    }
    return { xs, ys, sizes };
  }

  function decimalString(value) {
    const s = String(value).replace(/^0+(?=\d)/, '');
    return s || '0';
  }

  function rowMaxString(row) {
    let best = '0';
    for (const value of row) {
      const s = decimalString(value);
      if (s.length > best.length || (s.length === best.length && s > best)) best = s;
    }
    return best;
  }

  function decimalLE(a, b) {
    if (a.length !== b.length) return a.length < b.length;
    return a <= b;
  }

  function decimalGE(a, b) {
    if (a.length !== b.length) return a.length > b.length;
    return a >= b;
  }

  function buildSpatial(result) {
    const buckets = new Map();
    for (const p of result) {
      const bx = Math.floor(p.x / BUCKET_SIZE);
      const by = Math.floor(p.y / BUCKET_SIZE);
      const k = `${bx},${by}`;
      let bucket = buckets.get(k);
      if (!bucket) buckets.set(k, bucket = []);
      bucket.push(p);
    }
    return buckets;
  }

  function buildDataset(key) {
    if (datasetCache.has(key)) return datasetCache.get(key);

    const d = DATA[key];
    const visual = d.visual;
    const scalable = d.rows.length >= 300000 ? buildScalableDataLayout(d.rows) : null;
    const raw = scalable ? null : buildOrderedDataLayout(d.rows);
    const result = new Array(d.rows.length);
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

    for (let sourceIndex = 0; sourceIndex < d.rows.length; sourceIndex++) {
      const styleCode = parseInt(visual[sourceIndex], 16);
      const q = scalable ? { x: scalable.xs[sourceIndex], y: scalable.ys[sourceIndex], size: scalable.sizes[sourceIndex] } : raw[sourceIndex];
      const row = d.rows[sourceIndex];
      const p = {
        row,
        maxValue: rowMaxString(row),
        primeMask: d.masks ? d.masks[sourceIndex] : 0,
        sourceIndex,
        x: q.x,
        y: q.y,
        size: q.size,
        shape: styleCode >> 2,
        dominant: styleCode & 3
      };
      result[sourceIndex] = p;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    // Center only by an affine translation; relative geometry is unchanged.
    const cx = (minX + maxX) * 0.5;
    const cy = (minY + maxY) * 0.5;
    minX = Infinity; maxX = -Infinity; minY = Infinity; maxY = -Infinity;
    for (const p of result) {
      p.x -= cx;
      p.y -= cy;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }

    const built = {
      points: result,
      spatial: buildSpatial(result),
      bounds: { minX, maxX, minY, maxY }
    };
    datasetCache.set(key, built);
    return built;
  }

  function boundsFor(result) {
    if (!result.length) return { minX: -1, maxX: 1, minY: -1, maxY: 1 };
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const p of result) {
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    return { minX, maxX, minY, maxY };
  }

  function normalizeSearch(text) {
    const raw = text.trim().split(/[\s,，]+/).filter(Boolean);
    const out = [];
    const seen = new Set();
    for (const token of raw) {
      if (!/^\d+$/.test(token)) continue;
      let q;
      try { q = BigInt(token).toString(); } catch (_) { continue; }
      if (!seen.has(q)) { seen.add(q); out.push(q); }
    }
    return out;
  }

  function normalizeMax(text) {
    const token = text.trim();
    if (!/^\d+$/.test(token)) return null;
    const value = decimalString(token);
    return value === '0' ? '0' : value;
  }

  function rowWithinRange(maxValue) {
    if (maxTerm !== null && !decimalLE(maxValue, maxTerm)) return false;
    if (minTerm !== null && !decimalGE(maxValue, minTerm)) return false;
    return true;
  }

  function rowHasTerms(row) {
    if (!searchTerms.length) return true;
    for (const q of searchTerms) {
      let found = false;
      for (const value of row) {
        if (String(value) === q) { found = true; break; }
      }
      if (!found) return false;
    }
    return true;
  }

  function renderSearchResults() {
    searchResults.replaceChildren();
    if (!searchTerms.length) return;
    const ordered = points.slice().sort((a, b) => a.size - b.size || a.sourceIndex - b.sourceIndex);
    const limit = Math.min(100, ordered.length);
    for (let i = 0; i < limit; i++) {
      const p = ordered[i];
      const b = document.createElement('button');
      b.className = 'search-result';
      b.textContent = rawExpression(p.row);
      b.addEventListener('click', () => {
        selectedPoint = p;
        hoverPoint = null;
        panX = -p.x * viewStretchX * zoom;
        panY = p.y * viewStretchY * zoom;
        syncDetail();
        scheduleDraw();
      });
      searchResults.appendChild(b);
    }
  }

  function buildDynamicFilteredPoints() {
    const d = DATA[currentKey];
    const meta = META[currentKey];
    const denyMask = ((1 << meta.primes.length) - 1) & ~activePrimeMask;
    const rows = [];
    const sourceIndices = [];
    const maxNumeric = maxTerm !== null && maxTerm.length <= 15 ? Number(maxTerm) : null;
    const minNumeric = minTerm !== null && minTerm.length <= 15 ? Number(minTerm) : null;

    for (let i = 0; i < d.rows.length; i++) {
      const row = d.rows[i];
      let rowMax;
      if (maxTerm !== null) {
        if (maxNumeric !== null) {
          rowMax = 0;
          for (const value of row) if (value > rowMax) rowMax = value;
          if (rowMax > maxNumeric) {
            if (meta.sortedByMax) break;
            continue;
          }
        } else {
          const rowMaxText = rowMaxString(row);
          if (!decimalLE(rowMaxText, maxTerm)) {
            if (meta.sortedByMax) break;
            continue;
          }
        }
      }
      if (minTerm !== null) {
        if (rowMax === undefined) {
          if (minNumeric !== null) {
            rowMax = 0;
            for (const value of row) if (value > rowMax) rowMax = value;
          } else {
            rowMax = rowMaxString(row);
          }
        }
        if (minNumeric !== null) {
          if (rowMax < minNumeric) continue;
        } else {
          const rowMaxText = typeof rowMax === 'string' ? rowMax : String(rowMax);
          if (!decimalGE(rowMaxText, minTerm)) continue;
        }
      }
      if ((d.masks[i] & denyMask) !== 0 || !rowHasTerms(row)) continue;
      rows.push(row);
      sourceIndices.push(i);
    }

    if (!rows.length) return [];
    const scalable = rows.length >= 300000 ? buildScalableDataLayout(rows) : null;
    const raw = scalable ? null : buildOrderedDataLayout(rows);
    const result = new Array(rows.length);
    for (let i = 0; i < rows.length; i++) {
      const sourceIndex = sourceIndices[i];
      const styleCode = parseInt(d.visual[sourceIndex], 16);
      const q = scalable ? { x: scalable.xs[i], y: scalable.ys[i], size: scalable.sizes[i] } : raw[i];
      result[i] = {
        row: rows[i],
        maxValue: rowMaxString(rows[i]),
        primeMask: d.masks[sourceIndex],
        sourceIndex,
        x: q.x, y: q.y, size: q.size,
        shape: styleCode >> 2,
        dominant: styleCode & 3
      };
    }
    return result;
  }

  function fullPrimeMask(meta = META[currentKey]) {
    return meta ? (1 << meta.primes.length) - 1 : 0;
  }

  function hasActiveConstraint() {
    if (!currentKey) return false;
    return activePrimeMask !== fullPrimeMask() || searchTerms.length > 0 || maxTerm !== null || minTerm !== null;
  }

  function syncConstraintButtons() {
    if (!currentKey) {
      primeToggle.classList.remove('active');
      searchToggle.classList.remove('active');
      maxToggle.classList.remove('active');
      minToggle.classList.remove('active');
      return;
    }
    primeToggle.classList.toggle('active', activePrimeMask !== fullPrimeMask());
    searchToggle.classList.toggle('active', searchTerms.length > 0);
    maxToggle.classList.toggle('active', maxTerm !== null);
    minToggle.classList.toggle('active', minTerm !== null);
  }

  function syncTopChromeLayout() {
    if (!viewer.classList.contains('active')) return;
    const viewerRect = viewer.getBoundingClientRect();
    const buttonsRect = toolButtons.getBoundingClientRect();
    const buttonsBottom = Math.max(0, Math.ceil(buttonsRect.bottom - viewerRect.top));
    viewer.style.setProperty('--tool-buttons-bottom', `${buttonsBottom}px`);
    const countRect = solutionCount.getBoundingClientRect();
    const countBottom = Math.max(0, Math.ceil(countRect.bottom - viewerRect.top));
    viewer.style.setProperty('--top-chrome-bottom', `${Math.max(buttonsBottom, countBottom)}px`);
  }

  function applyFilters(refit = true) {
    const meta = META[currentKey];
    if (meta.dynamicLayout) {
      points = buildDynamicFilteredPoints();
      allPoints = points;
    } else {
      const denyMask = fullPrimeMask(meta) & ~activePrimeMask;
      points = allPoints.filter(p => (p.primeMask & denyMask) === 0 && rowHasTerms(p.row) && rowWithinRange(p.maxValue));
    }
    spatial = buildSpatial(points);
    bounds = boundsFor(points);
    updateViewStretch();
    solutionCount.textContent = `|Sol|=${points.length.toLocaleString("en-US")}`;
    syncConstraintButtons();
    syncTopChromeLayout();
    hoverPoint = null;
    selectedPoint = null;
    syncDetail();
    renderSearchResults();
    if (refit && width > 0 && height > 0) fit();
    else scheduleDraw();
  }

  function updateViewStretch() {
    viewStretchX = 1;
    viewStretchY = 1;
    if (!hasActiveConstraint() || points.length < 2 || width <= 0 || height <= 0) return;

    const spanX = Math.max(1e-9, bounds.maxX - bounds.minX);
    const spanY = Math.max(1e-9, bounds.maxY - bounds.minY);
    const padX = Math.max(76, width * 0.085);
    const padY = Math.max(76, height * 0.095);
    const usableW = Math.max(1, width - 2 * padX);
    const usableH = Math.max(1, height - 2 * padY);
    const targetRatio = usableW / usableH;
    const cloudRatio = spanX / spanY;

    // For a filtered/partial solution set, use the screen aspect ratio directly.
    // There is intentionally no old ~30% stretch cap: expand whichever axis is
    // short so the current subset uses the available viewport much more fully.
    if (cloudRatio > targetRatio) viewStretchY = cloudRatio / targetRatio;
    else viewStretchX = targetRatio / cloudRatio;
  }

  function setupPrimeControls() {
    const meta = META[currentKey];
    const primes = meta.primes;
    const defaultPrimeMax = meta.defaultPrimeMax ?? primes[primes.length - 1];
    activePrimeMask = 0;
    primeOptions.replaceChildren();
    primes.forEach((p, i) => {
      const checked = p <= defaultPrimeMax;
      if (checked) activePrimeMask |= 1 << i;
      const label = document.createElement('label');
      label.className = 'prime-option';
      const input = document.createElement('input');
      input.type = 'checkbox'; input.checked = checked; input.dataset.bit = String(i);
      const span = document.createElement('span'); span.textContent = String(p);
      input.addEventListener('change', () => {
        let mask = 0;
        for (const box of primeOptions.querySelectorAll('input')) if (box.checked) mask |= 1 << Number(box.dataset.bit);
        activePrimeMask = mask;
        applyFilters(true);
      });
      label.append(input, span); primeOptions.appendChild(label);
    });
  }

  function setAllPrimes(checked) {
    for (const box of primeOptions.querySelectorAll('input')) box.checked = checked;
    activePrimeMask = checked ? (1 << META[currentKey].primes.length) - 1 : 0;
    applyFilters(true);
  }

  function resetTools() {
    primePanel.classList.remove('show');
    searchPanel.classList.remove('show');
    maxPanel.classList.remove('show');
    minPanel.classList.remove('show');
    primeToggle.classList.remove('active');
    searchToggle.classList.remove('active');
    maxToggle.classList.remove('active');
    minToggle.classList.remove('active');
    searchInput.value = '';
    searchTerms = [];
    const defaultMax = META[currentKey].defaultMaxValue;
    maxInput.value = defaultMax == null ? '' : String(defaultMax);
    maxTerm = normalizeMax(maxInput.value);
    const defaultMin = META[currentKey].defaultMinValue;
    minInput.value = defaultMin == null ? '' : String(defaultMin);
    minTerm = normalizeMax(minInput.value);
    setupPrimeControls();
    syncConstraintButtons();
  }

  async function openDataset(key) {
    if (!META[key]) return;
    const request = ++openRequest;
    const button = choiceButtons.get(key);
    if (button) button.classList.add('loading');
    try {
      await loadDataset(key);
      if (request !== openRequest) return;
      currentKey = key;
      if (META[key].dynamicLayout) {
        allPoints = [];
        spatial = new Map();
        bounds = { minX: -1, maxX: 1, minY: -1, maxY: 1 };
      } else {
        const built = buildDataset(key);
        allPoints = built.points;
      }
      home.classList.remove('active');
      viewer.classList.add('active');
      resetTools();
      applyFilters(false);
      location.hash = key;
      requestAnimationFrame(() => {
        resizeCanvas();
        fit();
      });
    } catch (error) {
      console.error(error);
      if (button) button.classList.add('load-error');
    } finally {
      if (button) button.classList.remove('loading');
    }
  }

  function closeDataset() {
    viewer.classList.remove('active');
    home.classList.add('active');
    currentKey = null;
    allPoints = [];
    points = [];
    primePanel.classList.remove('show');
    searchPanel.classList.remove('show');
    maxPanel.classList.remove('show');
    minPanel.classList.remove('show');
    spatial = new Map();
    hoverPoint = null;
    selectedPoint = null;
    syncDetail();
    if (location.hash) history.replaceState(null, '', location.pathname + location.search);
  }

  function resizeCanvas() {
    const rect = stage.getBoundingClientRect();
    width = Math.max(1, Math.floor(rect.width));
    height = Math.max(1, Math.floor(rect.height));
    dpr = Math.min(DPR_CAP, window.devicePixelRatio || 1);
    const cw = Math.max(1, Math.floor(width * dpr));
    const ch = Math.max(1, Math.floor(height * dpr));
    for (const layer of [canvas, overlay]) {
      if (layer.width !== cw || layer.height !== ch) {
        layer.width = cw;
        layer.height = ch;
        layer.style.width = width + 'px';
        layer.style.height = height + 'px';
      }
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scheduleDraw();
  }

  function fit() {
    if (!points.length) {
      zoom = 1; minZoom = 0.1; maxZoom = 30; panX = 0; panY = 0;
      scheduleDraw();
      return;
    }
    // Re-evaluate filtered-set aspect adaptation after every resize/refit.
    updateViewStretch();
    // Fit the actual filtered bounds, including their center.  This is important
    // because filtering can leave an off-centre subset of the full map.
    const padX = Math.max(76, width * 0.085);
    const padY = Math.max(76, height * 0.095);
    const spanX = Math.max(0, bounds.maxX - bounds.minX) * viewStretchX;
    const spanY = Math.max(0, bounds.maxY - bounds.minY) * viewStretchY;
    const worldW = Math.max(3.2, spanX + 3.2);
    const worldH = Math.max(3.2, spanY + 3.2);
    const sx = Math.max(0.04, (width - 2 * padX) / worldW);
    const sy = Math.max(0.04, (height - 2 * padY) / worldH);
    const fitZoom = Math.min(sx, sy);
    zoom = fitZoom;
    // The fitted view is the initial view, not the minimum zoom.  Thus − works
    // immediately even after a filter/search refit, while + can still go deep.
    minZoom = Math.max(0.025, fitZoom / 8);
    maxZoom = Math.max(24, fitZoom * 36);
    const cx = (bounds.minX + bounds.maxX) * 0.5;
    const cy = (bounds.minY + bounds.maxY) * 0.5;
    panX = -cx * viewStretchX * zoom;
    panY = cy * viewStretchY * zoom;
    scheduleDraw();
  }

  function worldToScreenXY(x, y) {
    return {
      x: width / 2 + panX + x * viewStretchX * zoom,
      y: height / 2 + panY - y * viewStretchY * zoom
    };
  }

  function screenToWorldXY(x, y) {
    return {
      x: (x - width / 2 - panX) / (zoom * viewStretchX),
      y: -(y - height / 2 - panY) / (zoom * viewStretchY)
    };
  }

  function pointRadius() {
    // A slightly larger floor keeps isolated points legible at the fitted overview.
    return Math.max(1.85, Math.min(8.4, zoom * 0.52));
  }

  function visibleWorldBounds(paddingPx = 12) {
    const a = screenToWorldXY(-paddingPx, height + paddingPx);
    const b = screenToWorldXY(width + paddingPx, -paddingPx);
    return {
      minX: Math.min(a.x, b.x), maxX: Math.max(a.x, b.x),
      minY: Math.min(a.y, b.y), maxY: Math.max(a.y, b.y)
    };
  }

  function collectVisible() {
    const wb = visibleWorldBounds(22);
    const minBX = Math.floor(wb.minX / BUCKET_SIZE);
    const maxBX = Math.floor(wb.maxX / BUCKET_SIZE);
    const minBY = Math.floor(wb.minY / BUCKET_SIZE);
    const maxBY = Math.floor(wb.maxY / BUCKET_SIZE);
    const visible = [];
    for (let bx = minBX; bx <= maxBX; bx++) {
      for (let by = minBY; by <= maxBY; by++) {
        const bucket = spatial.get(`${bx},${by}`);
        if (!bucket) continue;
        for (const p of bucket) {
          if (p.x >= wb.minX && p.x <= wb.maxX && p.y >= wb.minY && p.y <= wb.maxY) visible.push(p);
        }
      }
    }
    return visible;
  }

  function appendShape(path, x, y, r, shape) {
    if (shape === 0) {
      path.moveTo(x + r, y);
      path.arc(x, y, r, 0, TWO_PI);
    } else if (shape === 1) {
      path.rect(x - r, y - r, 2 * r, 2 * r);
    } else if (shape === 2) {
      path.moveTo(x, y - r * 1.22);
      path.lineTo(x + r * 1.06, y + r * 0.92);
      path.lineTo(x - r * 1.06, y + r * 0.92);
      path.closePath();
    } else {
      path.moveTo(x - r * 1.04, y);
      path.lineTo(x + r * 1.04, y);
      path.moveTo(x, y - r * 1.04);
      path.lineTo(x, y + r * 1.04);
    }
  }

  function draw() {
    framePending = false;
    ctx.clearRect(0, 0, width, height);
    if (!points.length) return;

    const r = pointRadius();
    const paths = Array.from({ length: 16 }, () => new Path2D());
    const visible = collectVisible();

    // Static visibility aid for sparse result sets: a pale disc + crisp ring behind
    // every point.  It does not animate and disappears automatically for dense sets.
    if (points.length <= 140) {
      const haloR = r + (points.length <= 24 ? 7.0 : 4.8);
      ctx.fillStyle = 'rgba(255,255,255,.78)';
      ctx.strokeStyle = 'rgba(17,17,15,.24)';
      ctx.lineWidth = points.length <= 24 ? 1.25 : 0.9;
      for (const p of visible) {
        const s = worldToScreenXY(p.x, p.y);
        ctx.beginPath();
        ctx.arc(s.x, s.y, haloR, 0, TWO_PI);
        ctx.fill();
        ctx.stroke();
      }
    }

    for (const p of visible) {
      const s = worldToScreenXY(p.x, p.y);
      appendShape(paths[p.dominant * 4 + p.shape], s.x, s.y, r, p.shape);
    }

    ctx.globalAlpha = 0.9;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let dominant = 0; dominant < 4; dominant++) {
      const pointColor = COLORS[dominant];
      ctx.fillStyle = pointColor;
      ctx.strokeStyle = pointColor;
      for (let shape = 0; shape < 4; shape++) {
        const path = paths[dominant * 4 + shape];
        if (shape === 3) {
          ctx.lineWidth = Math.max(1.05, r * 0.60);
          ctx.stroke(path);
        } else {
          ctx.fill(path);
        }
      }
    }
    ctx.globalAlpha = 1;
    scheduleOverlay();
  }

  function drawOverlay() {
    overlayPending = false;
    overlayCtx.clearRect(0, 0, width, height);
    const active = hoverPoint || selectedPoint;
    if (!active || !points.length) return;
    const r = pointRadius();
    const s = worldToScreenXY(active.x, active.y);
    overlayCtx.strokeStyle = 'rgba(17,17,15,.80)';
    overlayCtx.lineWidth = 1.1;
    overlayCtx.beginPath();
    overlayCtx.arc(s.x, s.y, r + Math.max(3.4, r * 0.68), 0, TWO_PI);
    overlayCtx.stroke();
  }

  function scheduleOverlay() {
    if (overlayPending) return;
    overlayPending = true;
    requestAnimationFrame(drawOverlay);
  }

  function scheduleDraw() {
    if (!framePending) {
      framePending = true;
      requestAnimationFrame(draw);
    }
    scheduleOverlay();
  }

  function setZoom(newZoom, anchorX = width / 2, anchorY = height / 2) {
    const z = Math.max(minZoom, Math.min(maxZoom, newZoom));
    if (Math.abs(z - zoom) < 1e-8) return;
    const anchorWorld = screenToWorldXY(anchorX, anchorY);
    zoom = z;
    panX = anchorX - width / 2 - anchorWorld.x * viewStretchX * zoom;
    panY = anchorY - height / 2 + anchorWorld.y * viewStretchY * zoom;
    hoverPoint = null;
    syncDetail();
    scheduleDraw();
  }

  function nearestPoint(mx, my) {
    if (!points.length) return null;
    const w = screenToWorldXY(mx, my);
    // The interaction target is deliberately larger than the painted glyph.
    const hitPx = Math.max(12.5, pointRadius() + 7.8);
    const hitWorld = hitPx / (zoom * Math.min(viewStretchX, viewStretchY));
    const range = Math.max(1, Math.ceil(hitWorld / BUCKET_SIZE));
    const bx = Math.floor(w.x / BUCKET_SIZE);
    const by = Math.floor(w.y / BUCKET_SIZE);
    let best = null;
    let bestD = hitPx * hitPx;

    for (let ix = bx - range; ix <= bx + range; ix++) {
      for (let iy = by - range; iy <= by + range; iy++) {
        const bucket = spatial.get(`${ix},${iy}`);
        if (!bucket) continue;
        for (const p of bucket) {
          const dx = (p.x - w.x) * viewStretchX * zoom;
          const dy = (p.y - w.y) * viewStretchY * zoom;
          const d = dx * dx + dy * dy;
          if (d <= bestD) {
            bestD = d;
            best = p;
          }
        }
      }
    }
    return best;
  }

  function rawExpression(row) {
    const lhsCount = META[currentKey].lhsCount;
    return `${row.slice(0, lhsCount).join(' + ')} = ${row.slice(lhsCount).join(' + ')}`;
  }

  function factorHTML(value) {
    let n = BigInt(value);
    if (n === 1n) return '1';
    const parts = [];
    for (const p of META[currentKey].primes) {
      const prime = BigInt(p);
      let exponent = 0;
      while (n % prime === 0n) {
        n /= prime;
        exponent++;
      }
      if (exponent) parts.push(exponent === 1 ? `${p}` : `${p}<sup>${exponent}</sup>`);
    }
    if (n > 1n) parts.push(n.toString());
    return parts.join('·');
  }

  function factorExpression(row) {
    const v = row.map(factorHTML);
    const lhsCount = META[currentKey].lhsCount;
    return `${v.slice(0, lhsCount).join(' + ')} = ${v.slice(lhsCount).join(' + ')}`;
  }

  function downloadFiltered() {
    if (!currentKey) return;
    const ordered = points.slice().sort((a, b) => a.size - b.size || a.sourceIndex - b.sourceIndex);
    const text = ordered.map(p => rawExpression(p.row)).join('\n') + (ordered.length ? '\n' : '');
    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    const meta = META[currentKey];
    const safeEq = meta.equation.replace(/\s+/g, '').replace(/\+/g, '+').replace(/=/g, '=');
    a.href = url;
    a.download = `${safeEq}.txt`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  }

  function syncDetail() {
    const p = hoverPoint || selectedPoint;
    if (!p || !currentKey) {
      detail.classList.remove('show');
      detail.setAttribute('aria-hidden', 'true');
      return;
    }
    detailExpression.textContent = rawExpression(p.row);
    detailFactor.innerHTML = factorExpression(p.row);
    detail.classList.add('show');
    detail.setAttribute('aria-hidden', 'false');
  }

  function closeDetail() {
    selectedPoint = null;
    hoverPoint = null;
    syncDetail();
    scheduleOverlay();
  }

  primeToggle.addEventListener('click', () => {
    const show = !primePanel.classList.contains('show');
    primePanel.classList.toggle('show', show);
  });
  searchToggle.addEventListener('click', () => {
    const show = !searchPanel.classList.contains('show');
    searchPanel.classList.toggle('show', show);
    if (show) searchInput.focus();
  });
  maxToggle.addEventListener('click', () => {
    const show = !maxPanel.classList.contains('show');
    maxPanel.classList.toggle('show', show);
    if (show) { maxInput.focus(); maxInput.select(); }
  });
  minToggle.addEventListener('click', () => {
    const show = !minPanel.classList.contains('show');
    minPanel.classList.toggle('show', show);
    if (show) { minInput.focus(); minInput.select(); }
  });
  primeAll.addEventListener('click', () => setAllPrimes(true));
  primeNone.addEventListener('click', () => setAllPrimes(false));
  searchInput.addEventListener('input', () => {
    searchTerms = normalizeSearch(searchInput.value);
    applyFilters(true);
  });
  maxInput.addEventListener('input', () => {
    maxTerm = normalizeMax(maxInput.value);
    const key = currentKey;
    applyFilters(true);
    if (needsFullDataset(key)) {
      maxToggle.classList.add('loading');
      loadFullDataset(key).then(() => {
        if (currentKey === key) applyFilters(true);
      }).catch(console.error).finally(() => maxToggle.classList.remove('loading'));
    }
  });
  minInput.addEventListener('input', () => {
    minTerm = normalizeMax(minInput.value);
    const key = currentKey;
    applyFilters(true);
    if (needsFullDataset(key)) {
      minToggle.classList.add('loading');
      loadFullDataset(key).then(() => {
        if (currentKey === key) applyFilters(true);
      }).catch(console.error).finally(() => minToggle.classList.remove('loading'));
    }
  });
  downloadButton.addEventListener('click', downloadFiltered);
  tools.addEventListener('pointerdown', e => e.stopPropagation());
  tools.addEventListener('wheel', e => e.stopPropagation(), { passive: true });

  stage.addEventListener('wheel', e => {
    e.preventDefault();
    const rect = stage.getBoundingClientRect();
    const factor = Math.exp(-e.deltaY * 0.00115);
    setZoom(zoom * factor, e.clientX - rect.left, e.clientY - rect.top);
  }, { passive: false });

  stage.addEventListener('pointerdown', e => {
    if (e.button !== undefined && e.button !== 0) return;
    dragging = true;
    didDrag = false;
    dragPointerId = e.pointerId;
    dragX = e.clientX;
    dragY = e.clientY;
    dragPanX = panX;
    dragPanY = panY;
    stage.classList.add('dragging');
    stage.setPointerCapture(e.pointerId);
  });

  stage.addEventListener('pointermove', e => {
    const rect = stage.getBoundingClientRect();
    const mx = e.clientX - rect.left;
    const my = e.clientY - rect.top;

    if (dragging && e.pointerId === dragPointerId) {
      const dx = e.clientX - dragX;
      const dy = e.clientY - dragY;
      if (!didDrag && dx * dx + dy * dy > 12) didDrag = true;
      if (didDrag) {
        panX = dragPanX + dx;
        panY = dragPanY + dy;
        if (hoverPoint) {
          hoverPoint = null;
          syncDetail();
        }
        scheduleDraw();
      }
      return;
    }

    if (e.pointerType === 'touch') return;
    const p = nearestPoint(mx, my);
    if (p !== hoverPoint) {
      hoverPoint = p;
      syncDetail();
      scheduleOverlay();
    }
  });

  function endPointer(e) {
    if (!dragging || e.pointerId !== dragPointerId) return;
    const wasDrag = didDrag;
    dragging = false;
    didDrag = false;
    dragPointerId = null;
    stage.classList.remove('dragging');
    try { stage.releasePointerCapture(e.pointerId); } catch (_) {}

    if (!wasDrag) {
      const rect = stage.getBoundingClientRect();
      const p = nearestPoint(e.clientX - rect.left, e.clientY - rect.top);
      if (!p) {
        selectedPoint = null;
      } else if (selectedPoint === p) {
        selectedPoint = null;
      } else {
        selectedPoint = p;
      }
      hoverPoint = p;
      syncDetail();
      scheduleOverlay();
    }
  }

  stage.addEventListener('pointerup', endPointer);
  stage.addEventListener('pointercancel', e => {
    if (dragging && e.pointerId === dragPointerId) {
      dragging = false;
      didDrag = false;
      dragPointerId = null;
      stage.classList.remove('dragging');
    }
  });
  stage.addEventListener('pointerleave', () => {
    if (!dragging && hoverPoint) {
      hoverPoint = null;
      syncDetail();
      scheduleOverlay();
    }
  });

  detailClose.addEventListener('click', closeDetail);
  detail.addEventListener('pointerdown', e => e.stopPropagation());
  zoomIn.addEventListener('click', () => setZoom(zoom * 1.42));
  zoomOut.addEventListener('click', () => setZoom(zoom / 1.42));
  back.addEventListener('click', closeDataset);

  window.addEventListener('keydown', e => {
    if (!currentKey) return;
    if (e.key === 'Escape') { closeDetail(); return; }
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const plus = e.key === '+' || e.key === '=' || e.code === 'NumpadAdd';
    const minus = e.key === '-' || e.key === '_' || e.code === 'NumpadSubtract';
    if (plus || minus) {
      e.preventDefault();
      setZoom(plus ? zoom * 1.35 : zoom / 1.35);
    }
  });

  window.addEventListener('resize', () => {
    if (!currentKey) return;
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      resizeCanvas();
      syncTopChromeLayout();
      fit();
    }, 90);
  });

  if (typeof ResizeObserver !== 'undefined') {
    const chromeObserver = new ResizeObserver(() => syncTopChromeLayout());
    chromeObserver.observe(toolButtons);
    chromeObserver.observe(solutionCount);
  }

  const initial = location.hash.replace('#', '');
  if (ORDER.includes(initial)) openDataset(initial);
})();
