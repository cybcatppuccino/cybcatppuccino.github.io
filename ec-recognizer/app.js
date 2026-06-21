import { identifyCubicJS, EC_CORE_JS_VERSION } from './js/ec_core.js';

const $ = (id) => document.getElementById(id);

const inputEl = $('cubicInput');
const recognizeBtn = $('recognizeBtn');
const clearBtn = $('clearBtn');
const copyBtn = $('copyBtn');
const resultBox = $('resultBox');
const candidateBox = $('candidateBox');
const candidateCount = $('candidateCount');
const kernelStatus = $('kernelStatus');
const kernelDetail = $('kernelDetail');
const kernelDot = $('kernelDot');
const dataStats = $('dataStats');

let worker = null;
let pyReady = false;
let pending = new Map();
let requestSeq = 0;
let lastResult = null;
let curveIndexPromise = null;
let lastEngineNote = 'JS BigInt engine ready.';

const CURVE_FIELDS = ['id','label','cremona','iso','N','rank','tor_label','tor_order','cm','class_index','class_size','a1','a2','a3','a4','a6','disc','real_period'];
const FALLBACK_INDEX = {
  version: 'v18-emergency-fallback',
  curve_fields: CURVE_FIELDS,
  count: 8,
  j_count: 3,
  by_j: {
    '110592/233': [
      [39649, '6291.a1', '6291d1', '6291.a', 6291, 1, '0', 1, 0, 0, 1, 0, 0, 1, 27, -88, '-4586139', 1.2722683966782533],
      [39652, '6291.d1', '6291a1', '6291.d', 6291, 1, '0', 1, 0, 0, 1, 0, 0, 1, 3, 3, '-6291', 2.9347188679499463]
    ],
    '6912/13': [
      [7610, '1404.b1', '1404b1', '1404.b', 1404, 1, '0', 1, 0, 0, 1, 0, 0, 0, 27, 81, '-4094064', 1.699933263208385],
      [7611, '1404.c1', '1404a1', '1404.c', 1404, 1, '0', 1, 0, 0, 1, 0, 0, 0, 3, -3, '-5616', 2.236818132616875],
      [35080, '5616.m1', '5616p1', '5616.m', 5616, 0, '0', 1, 0, 0, 1, 0, 0, 0, 27, -81, '-4094064', 1.2914275509945885],
      [35094, '5616.w1', '5616o1', '5616.w', 5616, 0, '0', 1, 0, 0, 1, 0, 0, 0, 3, 3, '-5616', 2.944370781353281]
    ],
    '350402625/137842': [
      [42268, '6642.f2', '6642i1', '6642.f', 6642, 1, 'Z/3Z', 3, 0, 1, 2, 1, -1, 0, -132, -298, '100486818', 0.7281225582492187],
      [42279, '6642.p1', '6642l2', '6642.p', 6642, 0, '0', 1, 0, 0, 2, 1, -1, 1, -1190, 9235, '73254890322', 0.49657881561950934]
    ]
  }
};

function setKernel(state, title, detail) {
  kernelDot.className = `dot ${state}`;
  kernelStatus.textContent = title;
  kernelDetail.textContent = detail || '';
}
function htmlEscape(s) {
  return String(s ?? '').replace(/[&<>"']/g, (ch) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
}
function coeffKey(a) {
  return Array.isArray(a) ? a.map((x) => String(x)).join(',') : '';
}
function curveEquation(c) {
  const [a1,a2,a3,a4,a6] = c.a.map(Number);
  const term = (coef, body) => {
    if (!Number.isFinite(coef) || coef === 0) return '';
    const sign = coef < 0 ? ' - ' : ' + ';
    const mag = Math.abs(coef);
    const text = mag === 1 && body ? body : `${mag}${body}`;
    return `${sign}${text}`;
  };
  return `y²${term(a1,'xy')}${term(a3,'y')} = x³${term(a2,'x²')}${term(a4,'x')}${term(a6,'')}`;
}
function arrayToCurve(fields, arr) {
  const obj = {};
  fields.forEach((f, i) => obj[f] = arr[i]);
  obj.a = [obj.a1, obj.a2, obj.a3, obj.a4, obj.a6];
  return obj;
}
function displayDataStats(data, sourceLabel = 'static index') {
  const count = Number(data.count || 0).toLocaleString();
  const jCount = Number(data.j_count || Object.keys(data.by_j || {}).length || 0).toLocaleString();
  dataStats.innerHTML = `<div><strong>${count}</strong><br><small>curves</small></div><div><strong>${jCount}</strong><br><small>j-values</small></div><div><strong>${htmlEscape(sourceLabel)}</strong><br><small>curve data</small></div>`;
}
async function fetchJsonWithTimeout(url, timeoutMs = 12000) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  try {
    const response = await fetch(url, { cache: 'force-cache', signal: controller.signal });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    return await response.json();
  } finally {
    clearTimeout(timer);
  }
}
async function loadCurveIndex() {
  if (!curveIndexPromise) {
    curveIndexPromise = (async () => {
      const urls = [
        new URL('./data/curves_by_j.json', import.meta.url).toString(),
        './data/curves_by_j.json',
        '/ec-recognizer/data/curves_by_j.json'
      ];
      let lastError = null;
      for (const url of urls) {
        try {
          const data = await fetchJsonWithTimeout(url);
          if (!data || !data.by_j || !data.curve_fields) throw new Error('curve index has invalid shape');
          displayDataStats(data, 'loaded');
          return data;
        } catch (err) {
          lastError = err;
        }
      }
      console.warn('Curve data failed to load; using small emergency fallback.', lastError);
      displayDataStats(FALLBACK_INDEX, 'fallback');
      return FALLBACK_INDEX;
    })();
  }
  return curveIndexPromise;
}

function initPyodideFallback() {
  try {
    worker = new Worker(new URL('./ec-worker.js', import.meta.url));
  } catch (err) {
    setKernel('ready', 'JS BigInt ready', `Pyodide fallback could not start: ${err?.message || err}`);
    return;
  }
  worker.onmessage = (event) => {
    const msg = event.data || {};
    if (msg.type === 'ready') {
      pyReady = true;
      lastEngineNote = 'JS BigInt is primary; Pyodide fallback is ready for cross-checks.';
      setKernel('ready', 'JS BigInt + Pyodide ready', lastEngineNote);
      return;
    }
    if (msg.type === 'error') {
      pyReady = false;
      lastEngineNote = `JS BigInt is primary; Pyodide fallback unavailable: ${msg.error || 'unknown error'}`;
      setKernel('ready', 'JS BigInt ready', lastEngineNote);
      return;
    }
    if (msg.type === 'result') {
      const resolve = pending.get(msg.requestId);
      pending.delete(msg.requestId);
      if (resolve) resolve(msg.result);
    }
  };
  worker.onerror = (event) => {
    pyReady = false;
    lastEngineNote = `JS BigInt is primary; Pyodide worker error: ${event.message || 'unknown error'}`;
    setKernel('ready', 'JS BigInt ready', lastEngineNote);
  };
  worker.postMessage({ type: 'ping' });
}
function identifyPy(input) {
  if (!worker || !pyReady) return Promise.resolve(null);
  return new Promise((resolve) => {
    const requestId = ++requestSeq;
    pending.set(requestId, resolve);
    worker.postMessage({ type: 'identify', requestId, input });
    setTimeout(() => {
      if (pending.has(requestId)) {
        pending.delete(requestId);
        resolve({ ok: false, error: 'Pyodide fallback timed out.' });
      }
    }, 25000);
  });
}
function preferPyResult(jsRes, pyRes) {
  if (!pyRes || !pyRes.ok) return false;
  if (!jsRes || !jsRes.ok) return true;
  if (jsRes.j && pyRes.j && jsRes.j !== pyRes.j) return false;
  if (jsRes.mode !== 'minimal_model' && pyRes.mode === 'minimal_model') return true;
  if (!jsRes.discriminant && pyRes.discriminant) return true;
  return false;
}
function renderResult(res, extra = '') {
  lastResult = res;
  copyBtn.disabled = !res;
  if (!res || !res.ok) {
    resultBox.className = 'empty-state error-box';
    resultBox.innerHTML = `<strong>Recognition failed.</strong><br>${htmlEscape(res?.error || 'Unknown error')}`;
    return;
  }
  resultBox.className = '';
  const coeffs = Array.isArray(res.coeffs) ? `[${res.coeffs.join(', ')}]` : '—';
  const notes = Array.isArray(res.notes) && res.notes.length
    ? `<dt>notes</dt><dd>${res.notes.map(htmlEscape).join('<br>')}</dd>`
    : '';
  resultBox.innerHTML = `
    <div class="badge-row">
      <span class="result-badge">${res.mode === 'minimal_model' ? 'Q-minimal model' : 'j-invariant only'}</span>
      <span class="engine-badge">${htmlEscape(res.engine || res.version || 'recognizer')}</span>
      ${extra ? `<span class="engine-badge soft">${htmlEscape(extra)}</span>` : ''}
    </div>
    <dl class="kv">
      <dt>method</dt><dd>${htmlEscape(res.method)}</dd>
      <dt>j-invariant</dt><dd class="mono">${htmlEscape(res.j)}</dd>
      <dt>Weierstrass</dt><dd class="mono">${htmlEscape(res.weierstrass || '—')}</dd>
      <dt>coefficients</dt><dd class="mono">${htmlEscape(coeffs)}</dd>
      <dt>discriminant</dt><dd class="mono">${htmlEscape(res.discriminant || '—')}</dd>
      <dt>c4, c6</dt><dd class="mono">${htmlEscape(res.c4 || '—')}, ${htmlEscape(res.c6 || '—')}</dd>
      ${notes}
    </dl>`;
}
async function renderCandidates(res) {
  if (!res || !res.ok || !res.j) {
    candidateCount.textContent = 'none';
    candidateBox.className = 'empty-state';
    candidateBox.textContent = 'No j-invariant was available for atlas lookup.';
    return;
  }
  let index;
  try { index = await loadCurveIndex(); }
  catch (err) {
    candidateCount.textContent = 'data error';
    candidateBox.className = 'empty-state error-box';
    candidateBox.textContent = String(err && err.message ? err.message : err);
    return;
  }
  const fields = index.curve_fields || [];
  const raw = (index.by_j && index.by_j[res.j]) || [];
  const curves = raw.map((arr) => arrayToCurve(fields, arr));
  const inputCoeffKey = coeffKey(res.coeffs_int);
  curves.sort((a, b) => {
    const ae = inputCoeffKey && coeffKey(a.a) === inputCoeffKey ? 0 : 1;
    const be = inputCoeffKey && coeffKey(b.a) === inputCoeffKey ? 0 : 1;
    return ae - be || Number(a.N) - Number(b.N) || String(a.label).localeCompare(String(b.label));
  });
  candidateCount.textContent = `${curves.length} candidate${curves.length === 1 ? '' : 's'}`;
  if (!curves.length) {
    candidateBox.className = 'empty-state';
    candidateBox.textContent = 'No same-j curve was found in the bundled static atlas index.';
    return;
  }
  candidateBox.className = 'candidate-list';
  candidateBox.innerHTML = curves.slice(0, 60).map((c) => {
    const exact = inputCoeffKey && coeffKey(c.a) === inputCoeffKey;
    const tags = [
      exact ? `<span class="exact">exact stored model match</span>` : `<span class="samej">same j</span>`,
      `<span>N=${htmlEscape(c.N)}</span>`,
      `<span>rank ${htmlEscape(c.rank)}</span>`,
      `<span>torsion ${htmlEscape(c.tor_label === '0' ? 'trivial' : c.tor_label)}</span>`,
      c.cm ? `<span>CM</span>` : ``
    ].join('');
    return `<article class="curve-card">
      <div class="curve-top">
        <div><div class="curve-label">${htmlEscape(c.label)} <small>(${htmlEscape(c.cremona)})</small></div><div class="curve-tags">${tags}</div></div>
        <div class="mono">[${c.a.map(htmlEscape).join(', ')}]</div>
      </div>
      <div class="curve-eq">${htmlEscape(curveEquation(c))}</div>
      <div class="curve-tags"><span>isogeny class ${htmlEscape(c.iso)}</span><span>disc ${htmlEscape(c.disc)}</span><span>class ${Number(c.class_index)+1}/${htmlEscape(c.class_size)}</span></div>
    </article>`;
  }).join('');
}
async function runRecognition() {
  recognizeBtn.disabled = true;
  resultBox.className = 'empty-state';
  resultBox.textContent = 'Computing exact invariants with JS BigInt…';
  candidateBox.className = 'empty-state';
  candidateBox.textContent = 'Waiting for j-invariant…';
  candidateCount.textContent = 'running';
  let jsRes;
  try {
    jsRes = identifyCubicJS(inputEl.value);
    renderResult(jsRes, EC_CORE_JS_VERSION);
    await renderCandidates(jsRes);
    if (pyReady) {
      resultBox.insertAdjacentHTML('beforeend', '<p class="checking">Checking Pyodide fallback for a possible certified upgrade…</p>');
      const pyRes = await identifyPy(inputEl.value);
      if (preferPyResult(jsRes, pyRes)) {
        pyRes.engine = 'pyodide-fallback';
        renderResult(pyRes, 'upgraded from JS result');
        await renderCandidates(pyRes);
      } else if (pyRes && pyRes.ok && pyRes.j === jsRes.j) {
        renderResult(jsRes, 'Pyodide cross-check matched j');
      } else if (pyRes && !pyRes.ok) {
        jsRes.notes = [...(jsRes.notes || []), `Pyodide fallback did not complete: ${pyRes.error || 'unknown error'}`];
        renderResult(jsRes, 'JS result retained');
      }
    }
  } catch (err) {
    renderResult({ ok: false, error: String(err && err.message ? err.message : err) });
  } finally {
    recognizeBtn.disabled = false;
  }
}

document.querySelectorAll('[data-example]').forEach((btn) => {
  btn.addEventListener('click', () => {
    inputEl.value = btn.getAttribute('data-example') || '';
    inputEl.focus();
  });
});
recognizeBtn.addEventListener('click', runRecognition);
clearBtn.addEventListener('click', () => {
  inputEl.value = '';
  resultBox.className = 'empty-state';
  resultBox.textContent = 'Run the recognizer to see the j-invariant, model, and matching atlas curves.';
  candidateBox.className = 'empty-state';
  candidateBox.textContent = 'Same-j candidates will appear here after recognition.';
  candidateCount.textContent = 'waiting';
  copyBtn.disabled = true;
  lastResult = null;
});
copyBtn.addEventListener('click', async () => {
  if (!lastResult) return;
  await navigator.clipboard.writeText(JSON.stringify(lastResult, null, 2));
  copyBtn.textContent = 'Copied';
  setTimeout(() => copyBtn.textContent = 'Copy JSON', 900);
});
inputEl.addEventListener('keydown', (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') runRecognition();
});

recognizeBtn.disabled = false;
setKernel('ready', 'JS BigInt ready', 'The primary recognizer runs locally in JavaScript BigInt. Loading Pyodide fallback in the background…');
loadCurveIndex().catch((err) => {
  console.warn(err);
  displayDataStats(FALLBACK_INDEX, 'fallback');
});
initPyodideFallback();
