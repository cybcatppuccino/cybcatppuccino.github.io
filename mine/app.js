// app.js (restored overlay + next-move + number colors, debuggable, JS+CPP kernels)
// Enhanced with kernel state migration

let pyodide = null;
let jsApi = null, cppApi = null;
let kernelType = "js";
let switchingKernel = false;

let H = 25, W = 40, M = 200;
let cellScale = 1.3, pageScale = 1.0;
let currentGameSeed = null;

let manualModeEnabled = false;
let undoState = null;
let stepping = false;

const DBG = true;
const dlog = (...a) => DBG && console.log("[DBG]", ...a);
const dwarn = (...a) => DBG && console.warn("[DBG]", ...a);
const derr = (...a) => console.error("[ERR]", ...a);

document.documentElement.style.setProperty("--board-cell-scale", cellScale.toFixed(2));
document.documentElement.style.setProperty("--page-scale", pageScale.toFixed(2));
document.body.style.zoom = pageScale;

const el = (id) => document.getElementById(id);
const statusEl = el("status"), boardEl = el("board");
const infoEl = document.querySelector(".info-panel");
const controlPanel = document.querySelector(".controls-container"), togglePanelBtn = el("togglePanel");
const cellScaleValueEl = el("cellScaleValue");

const inpH = el("inpH"), inpW = el("inpW"), inpM = el("inpM"), inpSeed = el("inpSeed");
const btnNewGame = el("btnNewGame"), btnStepSolve = el("btnStepSolve");
const btnUndo = el("btnUndo");
const btnSwitchKernel = el("btnSwitchKernel"); // optional if exists
const firstZeroCheckbox = el("firstzero");

const btnEasy = el("btnEasy"), btnNormal = el("btnNormal"), btnHard = el("btnHard"), btnTranspose = el("btnTranspose");
const btnCellScaleUp = el("btnCellScaleUp"), btnCellScaleDown = el("btnCellScaleDown");

const btnHMinus5 = el("btnHMinus5"), btnHMinus1 = el("btnHMinus1"), btnHPlus1 = el("btnHPlus1"), btnHPlus5 = el("btnHPlus5");
const btnWMinus5 = el("btnWMinus5"), btnWMinus1 = el("btnWMinus1"), btnWPlus1 = el("btnWPlus1"), btnWPlus5 = el("btnWPlus5");
const btnMMinus100 = el("btnMMinus100"), btnMMinus10 = el("btnMMinus10"), btnMMinus1 = el("btnMMinus1");
const btnMPlus1 = el("btnMPlus1"), btnMPlus10 = el("btnMPlus10"), btnMPlus100 = el("btnMPlus100");

const FLAG_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect x="7" y="2" width="1.5" height="12" fill="#000"/><polygon points="8,2 14,5 8,8" fill="#f00"/></svg>`;

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
const setStatus = (s) => (statusEl.textContent = s);

function toPlain(x) {
  try { if (x && typeof x.toJs === "function") return x.toJs(); } catch {}
  return x;
}
function normalizeDelta(d) {
  d = toPlain(d) || {};
  if (!("move" in d)) d.move = null;
  if (!Array.isArray(d.newly)) d.newly = [];
  if (!Array.isArray(d.ai_mines)) d.ai_mines = [];
  d.lost = !!d.lost; d.won = !!d.won; d.stuck = !!d.stuck;
  if (!Number.isFinite(d.revealed_count)) d.revealed_count = 0;
  return d;
}
function normalizeState(st) {
  st = toPlain(st) || {};
  if (!Number.isFinite(st.h)) st.h = H;
  if (!Number.isFinite(st.w)) st.w = W;
  if (!Number.isFinite(st.mines)) st.mines = M;
  if (!Array.isArray(st.revealed)) st.revealed = [];
  if (!Array.isArray(st.ai_mines)) st.ai_mines = [];
  if (!Number.isFinite(st.revealed_count)) st.revealed_count = st.revealed.length;
  st.lost = !!st.lost; st.won = !!st.won;
  return st;
}
function hasMove(d) {
  return Array.isArray(d?.move) && d.move.length === 2 &&
    Number.isFinite(d.move[0]) && Number.isFinite(d.move[1]);
}
function clampInt(x, lo, hi, fallback) {
  const n = Number.parseInt(x, 10);
  return !Number.isFinite(n) ? fallback : Math.max(lo, Math.min(hi, n));
}
function getApi() { return kernelType === "cpp" ? cppApi : jsApi; }
function assertApiReady(A) {
  if (!A) return false;
  for (const k of ["newGame","step","stepAt","getState","getAnalysis"]) {
    if (typeof A[k] !== "function") { 
      dwarn("API missing basic function:", k, "available keys:", Object.keys(A)); 
      return false; 
    }
  }
  // Check additional functions for state migration - 只在需要时检查
  if (kernelType === "cpp") {
    // 先检查函数是否存在，不存在也不报错（作为可选功能）
    dlog("Available C++ functions:", Object.keys(A));
  }
  return true;
}



const key = (r,c)=> `${r},${c}`;
const jsRevealed = new Set();
let jsCells = [];

const getProbColor = (p) => {
  const clampedP = Math.max(0, Math.min(1, p));
  const saturation = 90;
  const lightness = 40*(1-clampedP)+40;
  const hue = 120 * (1- clampedP)**1.5;
  const alpha = 0.4+0.3*(1- clampedP);
  return `hsl(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
};

// ---------- UI helpers ----------
function togglePanel() {
  const isCollapsed = controlPanel.classList.contains("collapsed");
  
  if (isCollapsed) {
    // 展开
    controlPanel.classList.remove("collapsed");
    controlPanel.style.display = "block";
    togglePanelBtn.textContent = "▲";
  } else {
    // 折叠
    controlPanel.classList.add("collapsed");
    controlPanel.style.display = "none";
    togglePanelBtn.textContent = "▼";
  }
  
  // 关键：重新调整棋盘大小以适应新的布局
  setTimeout(() => {
    const board = document.getElementById('board');
    if (board) {
      adjustBoardToFillScreen();
    }
  }, 50);
}



// 调整棋盘填充屏幕 - 考虑 zoom 的影响
function adjustBoardToFillScreen() {
  const board = document.getElementById('board');
  if (!board) return;
  
  // 获取所有相关元素的高度
  const header = document.querySelector('.persistent-controls');
  const panel = document.querySelector('.controls-container');
  const status = document.getElementById('status');
  
  const headerHeight = header ? header.offsetHeight : 0;
  const panelHeight = panel && !panel.classList.contains("collapsed") ? panel.offsetHeight : 0;
  const statusHeight = status ? status.offsetHeight : 0;
  
  // 计算可用高度（考虑 zoom 影响）
  const totalUsedHeight = headerHeight + panelHeight + statusHeight;
  const availableHeight = (window.innerHeight / pageScale) - totalUsedHeight - 32; // 32px 是 padding
  
  // 设置棋盘高度
  board.style.height = Math.max(availableHeight, 200) + 'px';
  board.style.width = '100%';
}


// 页面加载和窗口大小变化时调整
window.addEventListener('load', adjustBoardToFillScreen);
window.addEventListener('resize', adjustBoardToFillScreen);



function adjustPageToFit() {
  const viewportWidth = window.innerWidth;
  const viewportHeight = window.innerHeight;
  const boardWrap = document.querySelector('.boardWrap');
  if (!boardWrap) return;
  const boardWidth = boardWrap.offsetWidth || 1;
  const boardHeight = boardWrap.offsetHeight || 1;
  const scale = Math.min(viewportWidth/boardWidth, viewportHeight/boardHeight) * 0.95;
  if (scale < 1) {
    document.body.style.zoom = scale;
    pageScale = scale;
    pageScaleValueEl.textContent = Math.round(pageScale * 100) + "%";
  }
}

function adjustCellScale(delta) {
  cellScale = Math.max(0.2, Math.min(4.0, cellScale + delta));
  document.documentElement.style.setProperty("--board-cell-scale", cellScale.toFixed(2));
  cellScaleValueEl.textContent = Math.round(cellScale * 100) + "%";
}
function adjustPageScale(delta) {
  pageScale = Math.max(0.2, Math.min(4.0, pageScale + delta));
  document.body.style.zoom = pageScale;
  pageScaleValueEl.textContent = Math.round(pageScale * 100) + "%";
  
  // 关键：调整所有元素的宽度来匹配 zoom 后的视窗
  setTimeout(() => {
    // 计算 zoom 后的实际可视区域尺寸
    const zoomedViewportWidth = window.innerWidth / pageScale;
    const zoomedViewportHeight = window.innerHeight / pageScale;
    
    // 调整 body 尺寸
    document.body.style.width = zoomedViewportWidth + 'px';
    document.body.style.height = zoomedViewportHeight + 'px';
    
    // 调整 layout 尺寸 - 关键是要填满整个宽度
    const layout = document.querySelector('.layout');
    if (layout) {
      layout.style.width = zoomedViewportWidth + 'px';
      layout.style.height = zoomedViewportHeight + 'px';
      layout.style.margin = '0';
      layout.style.padding = '12px';
      layout.style.boxSizing = 'border-box';
    }
    
    // 调整 boardWrap 尺寸
    const boardWrap = document.querySelector('.boardWrap');
    if (boardWrap) {
      // boardWrap 应该是 layout 减去 padding
      boardWrap.style.width = (zoomedViewportWidth - 24) + 'px'; // 12px * 2 padding
      boardWrap.style.height = (zoomedViewportHeight - 24) + 'px';
      boardWrap.style.margin = '0';
      boardWrap.style.boxSizing = 'border-box';
    }
    
    adjustBoardToFillScreen();
  }, 10);
}

function updateGameInfo(st) {
  let revealedCount = 0, flaggedMines = 0;
  if (st) { revealedCount = st.revealed_count || 0; flaggedMines = st.ai_mines ? st.ai_mines.length : 0; }
  const density = H > 0 && W > 0 ? Math.round((M / (H * W)) * 100) : 0;
  const displaySeed = (st && st.seed !== undefined && st.seed !== null) ? st.seed : (currentGameSeed ?? "None");
  infoEl.innerHTML = `
    <div class="info-item"><span class="info-label">Size</span><span class="info-value">${H}×${W}</span></div>
    <div class="info-item"><span class="info-label">Density</span><span class="info-value">${density}%</span></div>
    <div class="info-item"><span class="info-label">Revealed</span><span class="info-value">${revealedCount}/${H * W - M}</span></div>
    <div class="info-item"><span class="info-label">Mines</span><span class="info-value">${flaggedMines}/${M}</span></div>
    <div class="info-item seed-info"><span class="info-label">Seed</span><span class="info-value">${displaySeed}</span></div>
  `;
}

// ---------- Board DOM ----------
function buildBoardDOM(h, w) {
  if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0 || h*w > 40000)
    throw new Error(`Invalid board size: h=${h}, w=${w}`);
  jsRevealed.clear();
  jsCells = new Array(h*w);
  boardEl.style.gridTemplateColumns = `repeat(${w}, calc(var(--cell) * var(--board-cell-scale)))`;
  boardEl.innerHTML = "";
  for (let r=0;r<h;r++) for (let c=0;c<w;c++) {
    const d = document.createElement("div");
    d.className = "cell"; d.dataset.r = r; d.dataset.c = c;
    boardEl.appendChild(d);
    jsCells[r*w + c] = d;
  }
}

function clearAnalysisEffects(cellElement) {
  if (!cellElement) return;
  cellElement.classList.remove('analyzed', 'next-move');
  cellElement.style.backgroundColor = '';
  cellElement.style.color = '';
  cellElement.style.fontWeight = '';
  cellElement.style.fontSize = '';
  cellElement.style.border = '';
  cellElement.style.boxShadow = '';
  cellElement.style.animation = '';
}

function setCellCovered(r,c) {
  const d = jsCells[r*W + c];
  d.className = "cell";
  d.textContent = "";
  d.innerHTML = "";
  delete d.dataset.number;
  clearAnalysisEffects(d);
}
function setCellOpen(r,c,n) {
  const d = jsCells[r*W + c];
  d.className = "cell open";
  if (n > 0) { d.textContent = String(n); d.dataset.number = String(n); }
  else { d.textContent = ""; delete d.dataset.number; }
  clearAnalysisEffects(d);
}
function setCellMine(r,c) {
  const d = jsCells[r*W + c];
  d.className = "cell mine";
  d.textContent = "X";
  delete d.dataset.number;
  clearAnalysisEffects(d);
}
function setCellFlag(r,c) {
  const d = jsCells[r*W + c];
  if (d.classList.contains("open") || d.classList.contains("mine")) return;
  d.className = "cell flag";
  d.innerHTML = FLAG_SVG;
  delete d.dataset.number;
  clearAnalysisEffects(d);
}

// ---------- Rendering ----------
function applyFullState(st0) {
  const st = normalizeState(st0);
  H = st.h; W = st.w; M = st.mines;
  buildBoardDOM(H, W);
  if (st.seed !== undefined) currentGameSeed = st.seed;

  for (let r=0;r<H;r++) for (let c=0;c<W;c++) setCellCovered(r,c);
  for (const [r,c,n] of st.revealed) {
    jsRevealed.add(key(r,c));
    (n === -1) ? setCellMine(r,c) : setCellOpen(r,c,n);
  }
  for (const [r,c] of st.ai_mines) setCellFlag(r,c);

  setStatus(`Ready | Revealed: ${st.revealed_count} | Lost: ${st.lost} | Won: ${st.won}`);
  updateGameInfo(st);

  // restore analysis overlay immediately
  refreshAnalysisOverlay();
}

function applyStepDelta(d0) {
  const d = normalizeDelta(d0);

  for (const [r,c,n] of d.newly) {
    jsRevealed.add(key(r,c));
    (n === -1) ? setCellMine(r,c) : setCellOpen(r,c,n);
  }
  for (const [r,c] of d.ai_mines) setCellFlag(r,c);

  if (d.lost) { setStatus("GAME OVER"); manualModeEnabled = false; }
  else if (d.won) { setStatus("YOU WIN"); manualModeEnabled = false; }
  else if (d.stuck) { setStatus("STUCK (no moves)"); manualModeEnabled = false; }
  else setStatus(`Running | Revealed: ${d.revealed_count}`);

  updateGameInfo({ revealed_count: d.revealed_count, ai_mines: d.ai_mines, seed: currentGameSeed });

  // 不再立即刷新分析覆盖层，改为在适当时机批量刷新
}

// ---------- Analysis Overlay (same behavior as your old applyAnalysisOverlay) ----------
function clearOverlayMarks() {
  const analyzedCells = document.querySelectorAll('.cell.analyzed');
  for (const cell of analyzedCells) {
    if (!cell.classList.contains('open') &&
        !cell.classList.contains('mine') &&
        !cell.classList.contains('flag')) {
      cell.classList.remove('analyzed', 'next-move');
      cell.textContent = '';
      clearAnalysisEffects(cell);
    } else {
      cell.classList.remove('analyzed', 'next-move');
      clearAnalysisEffects(cell);
    }
  }
}

function parseProbsObject(probs) {
  // Expect {"r,c": p}, but tolerate spaces and non-string keys from wasm/py
  probs = toPlain(probs) || {};
  const out = [];
  for (const [k,v] of Object.entries(probs)) {
    const s = String(k);
    const m = s.match(/(\d+)\s*,\s*(\d+)/);
    if (!m) continue;
    const r = parseInt(m[1], 10), c = parseInt(m[2], 10);
    const p0 = toPlain(v);
    const p = typeof p0 === "string" ? parseFloat(p0) : Number(p0);
    if (!Number.isFinite(r) || !Number.isFinite(c) || !Number.isFinite(p)) continue;
    out.push([r,c,p]);
  }
  return out;
}

function applyAnalysisOverlay(analysis0) {
  const d = toPlain(analysis0) || {};
  clearOverlayMarks();

  const triples = parseProbsObject(d.probs);
  for (const [r,c,p] of triples) {
    if (r<0 || r>=H || c<0 || c>=W) continue;
    const cellElement = jsCells[r*W + c];
    if (!cellElement ||
        cellElement.classList.contains('open') ||
        cellElement.classList.contains('mine') ||
        cellElement.classList.contains('flag')) continue;

    cellElement.classList.add('analyzed');

    const pp = Math.max(0, Math.min(1, +p));
    cellElement.style.backgroundColor = getProbColor(pp);
    cellElement.textContent = Math.round(pp * 100).toString().padStart(2, '0');
    cellElement.style.color = '#000';
    cellElement.style.fontWeight = 'normal';
    cellElement.style.fontSize = 'calc(var(--cell) * var(--board-cell-scale) * 0.6)';
  }

  if (d.next_move && Array.isArray(d.next_move) && d.next_move.length === 2) {
    const nr = +d.next_move[0], nc = +d.next_move[1];
    if (Number.isFinite(nr) && Number.isFinite(nc) && nr>=0 && nr<H && nc>=0 && nc<W) {
      const nextCellElement = jsCells[nr * W + nc];
      if (nextCellElement &&
          !nextCellElement.classList.contains('open') &&
          !nextCellElement.classList.contains('mine') &&
          !nextCellElement.classList.contains('flag')) {

        nextCellElement.classList.add('analyzed', 'next-move');
        nextCellElement.style.backgroundColor = '#00FF00';
        nextCellElement.style.color = '#FFFFFF';
        nextCellElement.style.fontWeight = 'bold';
        nextCellElement.style.border = '2px solid #FF0000';
        nextCellElement.style.boxShadow = '0 0 10px #00FF00';
      }
    }
  }

  dlog("analysis overlay", { probs: triples.length, next_move: d.next_move });
}

function refreshAnalysisOverlay() {
  const A = getApi();
  if (!assertApiReady(A)) return;
  try {
    const a = toPlain(A.getAnalysis());
    applyAnalysisOverlay(a);
  } catch (e) {
    dwarn("getAnalysis/apply overlay failed:", e);
  }
}

// ---------- Interaction ----------
boardEl.addEventListener("click", handleManualClick);

async function handleManualClick(event) {
  const statusText = statusEl.textContent;
  if (statusText === "GAME OVER" || statusText === "YOU WIN" || statusText === "STUCK (no moves)") return;
  if (!manualModeEnabled || stepping) return;

  const target = event.target.closest?.(".cell") || event.target;
  if (!target.classList.contains("cell") ||
      target.classList.contains("open") ||
      target.classList.contains("flag") ||
      target.classList.contains("mine")) return;

  const r = parseInt(target.dataset.r, 10);
  const c = parseInt(target.dataset.c, 10);
  if (!Number.isFinite(r) || !Number.isFinite(c)) return;

  const A = getApi();
  if (!assertApiReady(A)) return setStatus("No API available for current kernel");

  dlog("ManualClick", { kernelType, r, c });

  // snapshot for undo if supported
  undoState = null;
  if (typeof A.setState === "function") undoState = JSON.parse(JSON.stringify(normalizeState(A.getState())));
  btnUndo.style.display = (undoState && typeof A?.setState === "function") ? "block" : "none";

  applyStepDelta(A.stepAt(r, c));
  const s2 = statusEl.textContent;
  if (s2 === "GAME OVER" || s2 === "YOU WIN" || s2 === "STUCK (no moves)") return;

  // 执行所有安全移动但不立即显示分析覆盖层
  if (typeof A.makeSafeMove === "function") {
    while (true) {
      const ds = normalizeDelta(A.makeSafeMove());
      applyStepDelta(ds);
      if (ds.lost || ds.won || ds.stuck) {
        // 游戏结束时也显示分析覆盖层
        refreshAnalysisOverlay();
        return;
      }
      if (!hasMove(ds)) break;
      await sleep(10);
    }
    // 只在所有安全移动完成后刷新分析覆盖层
    refreshAnalysisOverlay();
  }
}

async function stepSolve() {
  const A = getApi();
  if (!assertApiReady(A) || stepping || !manualModeEnabled) return;
  stepping = true;
  manualModeEnabled = false;

  try {
    dlog("stepSolve start", { kernelType });

    // 执行所有安全移动但不立即显示分析覆盖层
    if (typeof A.makeSafeMove === "function") {
      while (true) {
        const r = normalizeDelta(A.makeSafeMove());
        applyStepDelta(r);
        if (r.lost || r.won || r.stuck) {
          refreshAnalysisOverlay();
          return;
        }
        if (!hasMove(r)) break;
        await sleep(10);
      }
    }

    // snapshot before risky step() if supported
    undoState = null;
    if (typeof A.setState === "function") undoState = JSON.parse(JSON.stringify(normalizeState(A.getState())));
    btnUndo.style.display = (undoState && typeof A?.setState === "function") ? "block" : "none";

    const s = normalizeDelta(A.step());
    applyStepDelta(s);
    if (s.lost || s.won || s.stuck) {
      refreshAnalysisOverlay();
      return;
    }

    // 执行所有安全移动但不立即显示分析覆盖层
    if (typeof A.makeSafeMove === "function") {
      while (true) {
        const r2 = normalizeDelta(A.makeSafeMove());
        applyStepDelta(r2);
        if (r2.lost || r2.won || r2.stuck) {
          refreshAnalysisOverlay();
          return;
        }
        if (!hasMove(r2)) break;
        await sleep(10);
      }
    }
    
    // 只在所有操作完成后刷新分析覆盖层
    refreshAnalysisOverlay();
  } catch (e) {
    derr(e);
    setStatus("stepSolve failed: " + (e?.message || String(e)));
  } finally {
    stepping = false;
    manualModeEnabled = true;
  }
}

async function undoLastMove() {
  const A = getApi();
  if (!undoState) return;
  if (!assertApiReady(A)) return setStatus("No API available for current kernel");
  if (typeof A.setState !== "function") return setStatus("Undo not supported in this kernel");

  try {
    dlog("undo restore", { kernelType });
    const restored = normalizeState(A.setState(undoState));
    applyFullState(restored);
    btnUndo.style.display = "none";
    setStatus("Ready");
    undoState = null;
    manualModeEnabled = true;

    setTimeout(() => {
      refreshAnalysisOverlay();
      try { updateGameInfo(normalizeState(A.getState())); } catch {}
    }, 10);
  } catch (e) {
    derr("undo failed", e);
    setStatus("Undo failed: " + (e?.message || String(e)));
  }
}

// ---------- Kernel loading ----------
async function loadPy() {
  if (pyodide && jsApi) return;

  const candidates = [
    "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
    "https://pyodide.org/pyodide/v0.26.4/full/",
    "https://unpkg.com/pyodide@0.26.4/pyodide/full/",
  ];

  setStatus("Loading Pyodide...");
  let lastErr = null;
  for (const indexURL of candidates) {
    try { pyodide = await loadPyodide({ indexURL }); lastErr = null; break; }
    catch (e) { lastErr = e; dwarn("loadPyodide failed", indexURL, e); }
  }
  if (lastErr) throw lastErr;

  setStatus("Loading python code...");
  const resp = await fetch("./py/minesweeper2.py", { cache: "no-cache" });
  if (!resp.ok) throw new Error(`fetch minesweeper2.py failed: ${resp.status} ${resp.statusText}`);
  await pyodide.runPythonAsync(await resp.text());

jsApi = {
  newGame: pyodide.globals.get("ms_new_game"),
  step: pyodide.globals.get("ms_step"),
  stepAt: pyodide.globals.get("ms_step_at"),
  getState: pyodide.globals.get("ms_get_state"),
  makeSafeMove: pyodide.globals.get("ms_make_safe_move"),
  setState: pyodide.globals.get("ms_set_state"),
  getAnalysis: pyodide.globals.get("ms_get_analysis"),
  ms_load_board: pyodide.globals.get("ms_load_board"),
  ms_board_info: pyodide.globals.get("ms_board_info")
};


  dlog("js api ready");
  setStatus("Ready.");
}

function bindCppApiFromModule() {
  const pick = (...names) => {
    for (const n of names) if (typeof Module?.[n] === "function") return Module[n];
    return undefined;
  };
  cppApi = {
    newGame: pick("ms_new_game"),
    step: pick("ms_step"),
    stepAt: pick("ms_step_at"),
    getState: pick("ms_get_state"),
    makeSafeMove: pick("ms_make_safe_move"),
    setState: pick("ms_set_state"),
    getAnalysis: pick("ms_get_analysis"),
    ms_load_board: pick("ms_load_board", "loadBoard"),
    ms_board_info: pick("ms_board_info", "boardInfo")
  };
  dlog("cpp api bound", Object.fromEntries(Object.keys(cppApi).map(k => [k, typeof cppApi[k]])));
}


window.addEventListener('load', adjustBoardToFillScreen);
window.addEventListener('resize', adjustBoardToFillScreen);

window.Module = window.Module || {};
window.Module.print = window.Module.print || ((t)=>console.log("[WASM]", t));
window.Module.printErr = window.Module.printErr || ((t)=>console.error("[WASM ERROR]", t));
window.Module.locateFile = window.Module.locateFile || function(path, prefix) {
  if (path.endsWith(".wasm")) return "./cpp/" + path;
  return prefix + path;
};
window.Module.onRuntimeInitialized = function() {
  dlog("C++ runtime initialized");
  bindCppApiFromModule();
};

async function ensureCppLoaded() {
  if (cppApi && typeof cppApi.newGame === "function") return;

  const src = "./cpp/minesweeper.js";
  let script = document.querySelector(`script[data-cpp="1"]`);
  if (!script) {
    script = document.createElement("script");
    script.dataset.cpp = "1";
    script.async = true;
    document.head.appendChild(script);
  }
  script.src = src;

  await new Promise((resolve, reject) => {
    script.onload = () => { dlog("minesweeper.js loaded"); resolve(); };
    script.onerror = () => reject(new Error("Failed to load " + src));
  });

  await new Promise((resolve, reject) => {
    const t0 = performance.now();
    const timer = setInterval(() => {
      if (typeof Module?.ms_new_game === "function") {
        if (!cppApi) bindCppApiFromModule();
        clearInterval(timer);
        resolve();
      } else if (performance.now() - t0 > 20000) {
        clearInterval(timer);
        reject(new Error("C++ exports not ready"));
      }
    }, 50);
  });
}

// ---------- Enhanced Kernel Switching with State Migration ----------
async function migrateGameState(fromApi, toApi) {
  if (!fromApi || !toApi) return false;
  
  try {
    // 获取当前游戏状态
    const currentState = normalizeState(fromApi.getState());
    dlog("Migrating game state from", kernelType, currentState);
    
    // 构建字段数据 - 只包含可见信息
    const h = currentState.h || H;
    const w = currentState.w || W;
    
    // 创建字段数组，初始化为隐藏状态
    const field = new Array(h).fill(null).map(() => new Array(w).fill('H'));
    
    // 标记已揭示的数字格子
    for (const [r, c, n] of currentState.revealed) {
      if (r >= 0 && r < h && c >= 0 && c < w) {
        if (n === -1) {
          // 爆雷的格子 - 保持为H（因为游戏结束了）
          field[r][c] = 'H';
        } else {
          // 数字格子
          field[r][c] = String(n);
        }
      }
    }
    
    // 标记AI标记的旗子
    const aiMines = currentState.ai_mines || [];
    for (const [r, c] of aiMines) {
      if (r >= 0 && r < h && c >= 0 && c < w) {
        field[r][c] = 'F';
      }
    }
    
    // 构造基础加载数据
    const loadData = {
      height: h,
      width: w,
      mines: currentState.mines || M,
      seed: currentState.seed !== undefined ? currentState.seed : null,
      field: field
    };
    
    // 如果源内核支持 boardInfo，则获取完整的雷区信息
    if (typeof fromApi.ms_board_info === "function") {
      try {
        const boardInfo = toPlain(fromApi.ms_board_info());
        if (boardInfo && boardInfo.mines_layout) {
          loadData.mines_layout = boardInfo.mines_layout;
          if (boardInfo.first_move_made !== undefined) {
            loadData.first_move_made = boardInfo.first_move_made;
          }
        }
      } catch (e) {
        dlog("Failed to get boardInfo, using basic migration");
      }
    }
    
    dlog("Load data for migration:", loadData);
    
    // 优先使用 ms_load_board
    if (typeof toApi.ms_load_board === "function") {
      const newState = normalizeState(toApi.ms_load_board(loadData));
      dlog("Migration successful via ms_load_board");
      return newState;
    }
    // 回退到 setState
    else if (typeof toApi.setState === "function") {
      const newState = normalizeState(toApi.setState(currentState));
      dlog("Migration successful via setState");
      return newState;
    }
    
    dlog("No suitable migration method found, trying setState as fallback");
    // 最后的回退
    if (typeof toApi.setState === "function") {
      const newState = normalizeState(toApi.setState(currentState));
      return newState;
    }
    
    dlog("No migration method available");
    return false;
  } catch (e) {
    derr("Game state migration failed:", e);
    return false;
  }
}

async function switchToCppKernel() {
  if (switchingKernel) return;
  switchingKernel = true;
  try {
    setStatus("Switching to C++...");
    await ensureCppLoaded();
    
    let migratedState = null;
    if (kernelType === "js" && jsApi) {
      try {
        dlog("Attempting JS to C++ migration...");
        
        if (typeof jsApi.ms_board_info === "function") {
          const boardInfoRaw = jsApi.ms_board_info();
          const boardInfo = toPlain(boardInfoRaw);
          dlog("Board info from JS:", boardInfo);
          
          if (boardInfo && boardInfo.field && !boardInfo.error) {
            // 构造完整的加载数据（包含 mines_layout）
            const loadData = {
              height: parseInt(boardInfo.height, 10),
              width: parseInt(boardInfo.width, 10),
              mines: parseInt(boardInfo.mines, 10),
              seed: boardInfo.seed !== undefined && boardInfo.seed !== null ? parseInt(boardInfo.seed, 10) : null,
              field: boardInfo.field.map(row => String(row)),
              first_move_made: Boolean(boardInfo.first_move_made)
            };
            
            // 添加 mines_layout（如果存在）
            if (boardInfo.mines_layout) {
              try {
                // 安全地处理 mines_layout 嵌套数组
                const minesLayout = [];
                const rawLayout = boardInfo.mines_layout;
                
                // 处理可能的 JsProxy 对象
                const layoutArray = Array.isArray(rawLayout) ? rawLayout : 
                                  (rawLayout.toJs ? rawLayout.toJs() : Object.values(rawLayout));
                
                for (let r = 0; r < layoutArray.length; r++) {
                  const rawRow = layoutArray[r];
                  const rowArray = Array.isArray(rawRow) ? rawRow :
                                 (rawRow.toJs ? rawRow.toJs() : Object.values(rawRow));
                  
                  const processedRow = [];
                  for (let c = 0; c < rowArray.length; c++) {
                    processedRow.push(parseInt(rowArray[c], 10) || 0);
                  }
                  minesLayout.push(processedRow);
                }
                
                loadData.mines_layout = minesLayout;
                dlog("Added mines_layout to load data");
              } catch (layoutError) {
                dlog("Failed to process mines_layout:", layoutError);
              }
            }
            
            dlog("Calling C++ ms_load_board with complete data:", loadData);
            
            if (cppApi && typeof cppApi.ms_load_board === "function") {
              const result = cppApi.ms_load_board(loadData);
              migratedState = normalizeState(result);
              dlog("Migration successful:", migratedState);
            }
          }
        }
      } catch (migrationError) {
        derr("Migration error:", migrationError);
      }
    }
    
    kernelType = "cpp";
    if (btnSwitchKernel) btnSwitchKernel.textContent = "Switch to JS Kernel";
    
    if (migratedState) {
      applyFullState(migratedState);
      setStatus("Switched to C++ kernel with game state preserved");
    } else {
      await createNewGame();
      setStatus("Switched to C++ kernel (new game created)");
    }
  } catch (e) {
    derr("Switch to C++ failed:", e);
    setStatus("Failed to switch to C++ kernel: " + (e?.message || String(e)));
  } finally {
    switchingKernel = false;
  }
}

async function switchToJsKernel() {
  let migratedState = null;
  if (kernelType === "cpp" && cppApi) {
    try {
      if (typeof cppApi.ms_board_info === "function") {
        const boardInfo = toPlain(cppApi.ms_board_info());
        dlog("Board info from C++:", boardInfo);
        
        if (boardInfo && boardInfo.field && !boardInfo.error) {
          // 构造完整的加载数据（包含 mines_layout）
          const loadData = {
            height: parseInt(boardInfo.height, 10),
            width: parseInt(boardInfo.width, 10),
            mines: parseInt(boardInfo.mines, 10),
            seed: boardInfo.seed !== undefined && boardInfo.seed !== null ? parseInt(boardInfo.seed, 10) : null,
            field: boardInfo.field.map(row => String(row)),
            first_move_made: Boolean(boardInfo.first_move_made)
          };
          
          // 添加 mines_layout
          if (boardInfo.mines_layout) {
            try {
              // 安全地处理 mines_layout 嵌套数组
              const minesLayout = [];
              const rawLayout = boardInfo.mines_layout;
              
              // 处理可能的 JsProxy 对象
              const layoutArray = Array.isArray(rawLayout) ? rawLayout : 
                                (rawLayout.toJs ? rawLayout.toJs() : Object.values(rawLayout));
              
              for (let r = 0; r < layoutArray.length; r++) {
                const rawRow = layoutArray[r];
                const rowArray = Array.isArray(rawRow) ? rawRow :
                               (rawRow.toJs ? rawRow.toJs() : Object.values(rawRow));
                
                const processedRow = [];
                for (let c = 0; c < rowArray.length; c++) {
                  processedRow.push(parseInt(rowArray[c], 10) || 0);
                }
                minesLayout.push(processedRow);
              }
              
              loadData.mines_layout = minesLayout;
              dlog("Added mines_layout to load data for Python");
            } catch (layoutError) {
              dlog("Failed to process mines_layout:", layoutError);
            }
          }
          
          dlog("Prepared complete load data:", loadData);
          
          // 序列化为 JSON 字符串
          const jsonString = JSON.stringify(loadData);
          dlog("JSON string length:", jsonString.length);
          
          if (jsApi && typeof jsApi.ms_load_board === "function") {
            migratedState = normalizeState(jsApi.ms_load_board(jsonString));
            dlog("Migration from C++ to JS successful");
          }
        }
      }
    } catch (e) {
      derr("C++ to JS migration failed:", e);
    }
  }
  
  kernelType = "js";
  if (btnSwitchKernel) btnSwitchKernel.textContent = "Switch to C++ Kernel";
  
  if (migratedState) {
    applyFullState(migratedState);
    setStatus("Switched to JS kernel with game state preserved");
  } else {
    await createNewGame();
    setStatus("Switched to JS kernel (new game created)");
  }
}


async function switchKernel() {
  if (kernelType === "js") await switchToCppKernel();
  else await switchToJsKernel();
}

// ---------- Game creation ----------
function setDifficulty(h, w, m) { inpH.value = h; inpW.value = w; inpM.value = m; }
function transposeBoard() {
  const currentH = parseInt(inpH.value, 10) || 25;
  const currentW = parseInt(inpW.value, 10) || 40;
  const currentM = parseInt(inpM.value, 10) || 200;
  inpH.value = currentW; inpW.value = currentH; inpM.value = currentM;
}
function adjustParam(inputId, delta, minVal, maxVal) {
  const input = el(inputId);
  if (!input) return;
  let val = parseInt(input.value, 10) || 0;
  val = Math.max(minVal, Math.min(maxVal, val + delta));
  input.value = val;
}

async function createNewGame() {
  manualModeEnabled = true;
  undoState = null;
  btnUndo.style.display = "none";

  const h = clampInt(inpH.value, 5, 200, 25);
  const w = clampInt(inpW.value, 5, 200, 40);
  const m = clampInt(inpM.value, 1, h*w - 1, Math.min(200, h*w-1));
  const seedStr = (inpSeed?.value ?? "").trim();
  
  let seed;
  if (seedStr !== "") {
    seed = clampInt(seedStr, -2147483648, 2147483647, null);
  } else {
    // 使用 Crypto API 生成高质量随机种子
    seed = await generateCryptoSeed();
    console.log("Generated crypto seed:", seed);
  }
  
  const firstmv = firstZeroCheckbox?.checked ? 1 : 2;

  inpH.value = String(h); inpW.value = String(w); inpM.value = String(m);

  const A = getApi();
  console.log("Creating new game with params:", { kernelType, h, w, m, seed, firstmv });
  if (!assertApiReady(A)) return setStatus("No API available for current kernel");

  const st = normalizeState(A.newGame(h, w, m, seed, firstmv));
  if (st.seed !== undefined) currentGameSeed = st.seed;

  applyFullState(st);
}

// 新增：生成加密级随机种子
async function generateCryptoSeed() {
  // 方法1: 使用 Crypto.getRandomValues (推荐)
  if (typeof crypto !== 'undefined' && crypto.getRandomValues) {
    const array = new Int32Array(1);
    crypto.getRandomValues(array);
    return array[0];
  }
  
  // 方法2: 使用 Math.random() + 时间戳 + 性能计时的组合
  const timestamp = Date.now();
  const performanceNow = performance.now();
  const randomFactor = Math.random() * 1000000;
  
  // 混合多种熵源
  let seed = Math.abs(
    (timestamp * 492357816) ^ 
    (Math.floor(performanceNow * 1000) * 73856093) ^ 
    (Math.floor(randomFactor) * 114007148193271)
  );
  
  // 确保在 32位整数范围内
  return seed % 2147483647;
}


// ---------- Wire ----------
btnNewGame?.addEventListener("click", createNewGame);
btnStepSolve?.addEventListener("click", stepSolve);
btnUndo?.addEventListener("click", undoLastMove);
btnSwitchKernel?.addEventListener("click", switchKernel);

btnEasy?.addEventListener("click", () => { setDifficulty(9,9,10); createNewGame(); });
btnNormal?.addEventListener("click", () => { setDifficulty(16,16,40); createNewGame(); });
btnHard?.addEventListener("click", () => { setDifficulty(16,30,99); createNewGame(); });
btnTranspose?.addEventListener("click", () => { transposeBoard(); createNewGame(); });

togglePanelBtn?.addEventListener("click", togglePanel);
btnCellScaleUp?.addEventListener("click", () => adjustCellScale(0.1));
btnCellScaleDown?.addEventListener("click", () => adjustCellScale(-0.1));

btnHMinus5?.addEventListener("click", () => adjustParam("inpH", -5, 5, 200));
btnHMinus1?.addEventListener("click", () => adjustParam("inpH", -1, 5, 200));
btnHPlus1?.addEventListener("click", () => adjustParam("inpH", 1, 5, 200));
btnHPlus5?.addEventListener("click", () => adjustParam("inpH", 5, 5, 200));

btnWMinus5?.addEventListener("click", () => adjustParam("inpW", -5, 5, 200));
btnWMinus1?.addEventListener("click", () => adjustParam("inpW", -1, 5, 200));
btnWPlus1?.addEventListener("click", () => adjustParam("inpW", 1, 5, 200));
btnWPlus5?.addEventListener("click", () => adjustParam("inpW", 5, 5, 200));

btnMMinus100?.addEventListener("click", () => adjustParam("inpM", -100, 1, 9999));
btnMMinus10?.addEventListener("click", () => adjustParam("inpM", -10, 1, 9999));
btnMMinus1?.addEventListener("click", () => adjustParam("inpM", -1, 1, 9999));
btnMPlus1?.addEventListener("click", () => adjustParam("inpM", 1, 1, 9999));
btnMPlus10?.addEventListener("click", () => adjustParam("inpM", 10, 1, 9999));
btnMPlus100?.addEventListener("click", () => adjustParam("inpM", 100, 1, 9999));

cellScaleValueEl && (cellScaleValueEl.textContent = "130%");
togglePanelBtn && (togglePanelBtn.textContent = "▼");

// ---------- Keyboard Shortcuts ----------
document.addEventListener('keydown', function(event) {
  // 防止在输入框中触发快捷键
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
    return;
  }
  
  // 阻止默认行为（除了特殊键）
  const specialKeys = [' ', 'Enter', 'PageUp', 'PageDown', '+', '-'];
  const keyChar = event.key.toLowerCase();
  
  // N - New Game
  if (keyChar === 'n') {
    event.preventDefault();
    createNewGame();
    return;
  }
  
  // Z - Undo
  if (keyChar === 'z') {
    event.preventDefault();
    if (btnUndo && btnUndo.style.display !== 'none') {
      undoLastMove();
    }
    return;
  }
  
  // S - Switch Kernel
  if (keyChar === 's') {
    event.preventDefault();
    if (btnSwitchKernel) {
      switchKernel();
    }
    return;
  }
  
  // Space or Enter - Step Solve
  if (event.key === ' ' || event.key === 'Enter') {
    event.preventDefault();
    if (btnStepSolve) {
      stepSolve();
    }
    return;
  }
  
  // + or PageUp or Numpad+ - Increase Cell Scale
  if (event.key === '=' || event.key === 'PageUp' || event.key === 'Add') {
    event.preventDefault();
    adjustCellScale(0.1);
    return;
  }
  
  // - or PageDown or Numpad- - Decrease Cell Scale
  if (event.key === '-' || event.key === 'PageDown' || event.key === 'Subtract') {
    event.preventDefault();
    adjustCellScale(-0.1);
    return;
  }
});

// 确保空格键在按钮上不会触发页面滚动
btnStepSolve?.addEventListener('keydown', function(event) {
  if (event.key === ' ') {
    event.preventDefault();
    stepSolve();
  }
});


// ---------- bootstrap (JS kernel default) ----------
(async () => {
  try {
    await loadPy();
    kernelType = "js";
    if (btnSwitchKernel) btnSwitchKernel.textContent = "Switch to C++ Kernel";
    setDifficulty(16, 30, 99);
    await new Promise(requestAnimationFrame);
    await createNewGame();
  } catch (e) {
    derr(e);
    setStatus("Failed to load Pyodide: " + (e?.stack || String(e)));
  }
})();
