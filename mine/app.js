// app.js (restored overlay + next-move + number colors, debuggable, JS+CPP kernels)
// Enhanced with kernel state migration

let pyodide = null;
let jsApi = null, cppApi = null;
let kernelType = "cpp";  // 🔴 改为默认使用C++内核
// 🔴 添加一个标记，记录是否已经加载了Pyodide
let pyodideLoaded = false;
let pyodideLoaderPromise = null;
let pyodideLoadPromise = null;
let switchingKernel = false;


// 添加全局游戏状态变量
let globalGameState = "READY"; // 初始状态

// 🔴 检查游戏是否结束
function isGameEnded() {
  return globalGameState === "GAME OVER" ||
         globalGameState === "YOU WIN" ||
         globalGameState === "STUCK" ||
         globalGameState === "WRONG FLAG";
}


let H = 25, W = 40, M = 200;
let cellScale = 1.3, pageScale = 1.0;
let currentGameSeed = null;

let manualModeEnabled = false;
let undoState = null;
let stepping = false;
let allowHoverEffect = true;
let showWinRateEnabled = true;
let playByAIEnabled = false;
let boardFrozen = false;
let frozenBoardHasError = false;
let victoryProbabilityMode = null; // "flags" => remaining hidden cells are safe; "remaining-mines" => remaining hidden cells are mines


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
const btnNewGame = el("btnNewGame");
const btnUndo = el("btnUndo");
const btnSwitchKernel = el("btnSwitchKernel"); // optional if exists
const firstZeroCheckbox = el("firstzero");
const showWinRateButton = el("showWinRate");
const playByAIButton = el("playByAI");
const btnDownloadMine = el("btnDownloadMine");
const btnUploadMine = el("btnUploadMine");
const mineUploadInput = el("mineUploadInput");

const btnEasy = el("btnEasy"), btnNormal = el("btnNormal"), btnHard = el("btnHard"), btnTranspose = el("btnTranspose");
const btnCellScaleUp = el("btnCellScaleUp"), btnCellScaleDown = el("btnCellScaleDown");

const btnHMinus5 = el("btnHMinus5"), btnHMinus1 = el("btnHMinus1"), btnHPlus1 = el("btnHPlus1"), btnHPlus5 = el("btnHPlus5");
const btnWMinus5 = el("btnWMinus5"), btnWMinus1 = el("btnWMinus1"), btnWPlus1 = el("btnWPlus1"), btnWPlus5 = el("btnWPlus5");
const btnMMinus100 = el("btnMMinus100"), btnMMinus10 = el("btnMMinus10"), btnMMinus1 = el("btnMMinus1");
const btnMPlus1 = el("btnMPlus1"), btnMPlus10 = el("btnMPlus10"), btnMPlus100 = el("btnMPlus100");

const FLAG_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" aria-hidden="true"><g transform="translate(-1 0)"><rect x="7.2" y="2.6" width="1.7" height="13.2" rx="0.7" fill="#202020"/><path d="M8.6 3.1 L17 5.15 L8.6 8.55 Z" fill="#d81724"/><path d="M8.6 6.05 L17 5.15 L8.6 8.55 Z" fill="#a91119" opacity="0.82"/><rect x="5.2" y="15.1" width="6" height="1.7" rx="0.8" fill="#252525"/></g></svg>`;

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
function getApi() { 
    if (kernelType === "cpp") {
        return cppApi; 
    } else {
        return jsApi; 
    }
}
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
    //dlog("Available C++ functions:", Object.keys(A));
  }
  return true;
}




const key = (r,c)=> `${r},${c}`;
function parseCellKey(k) {
  const [r, c] = String(k).split(',').map(v => parseInt(v, 10));
  return [r, c];
}
const jsRevealed = new Set();
let jsCells = [];

// Tracks only cells currently carrying the probability / AI recommendation overlay.
// Keeping this explicit map lets overlay refreshes update only changed cells instead
// of querying and repainting the whole board every time.
const overlayCellsByKey = new Map();
const overlayStateByKey = new Map();


// Visible flags are the shared, user-visible board state.
// The engine may infer mines internally for probabilities, but those must not
// become visible flags unless the player marks them or AI play is explicitly used.
let visibleFlagKeys = new Set();
let wrongFlagKey = null;

function isVisibleFlag(r, c) { return visibleFlagKeys.has(key(r, c)); }
function addVisibleFlag(r, c) {
  const k = key(r, c);
  visibleFlagKeys.add(k);
  setCellFlag(r, c);
}
function visibleFlagsAsArray() {
  return Array.from(visibleFlagKeys).map(parseCellKey);
}

function readMineLayout(A) {
  const info = getBoardInfo(A);
  if (!info || info.error || info.first_move_made === false) return null;
  const layout = rowToPlainArray(info.mines_layout);
  if (!layout.length) return null;
  return layout.map(rowToPlainArray);
}

function allActualMinesAreVisibleFlags(A) {
  if (!Number.isFinite(M) || M <= 0) return false;
  if (visibleFlagKeys.size < M) return false;

  const layout = readMineLayout(A);
  if (layout && layout.length) {
    let mineCount = 0;
    for (let r = 0; r < H; r++) {
      const row = rowToPlainArray(layout[r]);
      for (let c = 0; c < W; c++) {
        if (Number(row[c]) === 1) {
          mineCount++;
          if (!visibleFlagKeys.has(key(r, c))) return false;
        }
      }
    }
    return mineCount > 0 && mineCount === M;
  }

  return visibleFlagKeys.size === M && !wrongFlagKey;
}

function checkFlagWinCondition(A) {
  if (isGameEnded()) return false;
  if (!allActualMinesAreVisibleFlags(A)) return false;
  globalGameState = "YOU WIN";
  manualModeEnabled = false;
  victoryProbabilityMode = "flags";
  setStatus("YOU WIN");
  try {
    updateGameInfo(normalizeState(A?.getState?.() || { revealed_count: jsRevealed.size, seed: currentGameSeed }));
  } catch {
    updateGameInfo({ revealed_count: jsRevealed.size, seed: currentGameSeed });
  }
  if (showWinRateEnabled) applyWinningProbabilityOverlay(victoryProbabilityMode);
  return true;
}

function confirmGameEndFromKernel(A) {
  if (isGameEnded() || !A || typeof A.getState !== "function") return false;
  try {
    const st = normalizeState(A.getState());
    if (st.lost) {
      setStatus("GAME OVER");
      manualModeEnabled = false;
      globalGameState = "GAME OVER";
      updateGameInfo(st);
      return true;
    }
    if (st.won) {
      setStatus("YOU WIN");
      manualModeEnabled = false;
      globalGameState = "YOU WIN";
      victoryProbabilityMode = allActualMinesAreVisibleFlags(A) ? "flags" : "remaining-mines";
      updateGameInfo(st);
      if (showWinRateEnabled) applyWinningProbabilityOverlay(victoryProbabilityMode);
      return true;
    }
    return checkFlagWinCondition(A);
  } catch (e) {
    dwarn("confirmGameEndFromKernel failed:", e);
    return false;
  }
}

const getProbColor = (p) => {
  // Softer, darker old-style green -> yellow/orange -> red gradient.
  // Keep probability differences clear while separating hidden/probability cells
  // from the opened board visually.
  const x = Math.max(0, Math.min(1, Number(p) || 0));
  const hue = 120 * Math.pow(1 - x, 1.38);
  const saturation = 55 + 16 * x;
  const lightness = 55 - 27 * Math.pow(x, 1.04);
  const alpha = 0.80 + 0.08 * x;
  return `hsla(${hue.toFixed(1)}, ${saturation.toFixed(1)}%, ${lightness.toFixed(1)}%, ${alpha.toFixed(3)})`;
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
  if (st) { revealedCount = st.revealed_count || 0; }
  flaggedMines = visibleFlagKeys.size;
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
  overlayCellsByKey.clear();
  overlayStateByKey.clear();
  currentHoverCell = null;
  jsCells = new Array(h*w);
  boardEl.style.gridTemplateColumns = `repeat(${w}, calc(var(--cell) * var(--board-cell-scale)))`;

  const frag = document.createDocumentFragment();
  for (let r=0;r<h;r++) for (let c=0;c<w;c++) {
    const d = document.createElement("div");
    d.className = "cell";
    d.dataset.r = r;
    d.dataset.c = c;

    // 🔴 添加鼠标事件监听器
    d.addEventListener('mouseenter', handleCellHover);
    d.addEventListener('mouseleave', handleCellLeave);

    frag.appendChild(d);
    jsCells[r*w + c] = d;
  }
  boardEl.replaceChildren(frag);
}

// 🔴 添加全局变量跟踪当前悬停的单元格
let currentHoverCell = null;

// 🔴 修改鼠标进入单元格事件
function handleCellHover(event) {
  // 🔴 如果不允许悬停效果，直接返回
  if (!allowHoverEffect) return;
  
  const cell = event.target;
  // 只对隐藏的格子（不是已揭示、不是雷、不是旗子）添加悬停效果
  if (!cell.classList.contains('open') && 
      !cell.classList.contains('mine') && 
      !cell.classList.contains('flag') &&
      !cell.classList.contains('wrong-flag')) {
    
    // 移除之前悬停格子的效果
    if (currentHoverCell && currentHoverCell !== cell) {
      handleCellLeave({ target: currentHoverCell });
    }
    
    // 添加悬停效果（红色边框）
    cell.style.border = '2px solid #FF0000';
    cell.style.boxShadow = '0 0 5px rgba(255, 0, 0, 0.5)';
    
    currentHoverCell = cell;
  }
}

// 🔴 鼠标离开单元格事件保持不变
function handleCellLeave(event) {
  const cell = event.target;
  // 只移除悬停效果，不影响分析覆盖层的样式
  if (cell === currentHoverCell) {
    // 检查是否是分析覆盖层的格子，如果是则恢复原有样式
    if (cell.classList.contains('analyzed')) {
      // 恢复分析覆盖层样式
      if (cell.classList.contains('next-move')) {
        cell.style.borderTop = '2px solid rgba(216, 236, 247, 0.96)';
        cell.style.borderLeft = '2px solid rgba(216, 236, 247, 0.96)';
        cell.style.borderRight = '2px solid rgba(44, 86, 128, 0.95)';
        cell.style.borderBottom = '2px solid rgba(44, 86, 128, 0.95)';
        cell.style.outline = 'none';
        cell.style.boxShadow = '0 0 10px rgba(91, 183, 255, 0.95), 0 0 24px rgba(45, 132, 224, 0.78), inset 0 0 9px rgba(218, 243, 255, 0.74), inset 0 2px 4px rgba(0,0,0,0.24)';
      } else {
        cell.style.border = '';
        cell.style.boxShadow = '';
      }
    } else {
      // 普通隐藏格子，清除悬停效果
      cell.style.border = '';
      cell.style.boxShadow = '';
    }
    currentHoverCell = null;
  }
}

function clearAnalysisEffects(cellElement) {
  if (!cellElement) return;
  cellElement.classList.remove('analyzed', 'next-move');
  cellElement.style.removeProperty('--prob-color');
  cellElement.style.color = '';
  cellElement.style.fontWeight = '';
  cellElement.style.fontSize = '';
  cellElement.style.fontFamily = '';
  cellElement.style.outline = '';
  cellElement.style.borderTop = '';
  cellElement.style.borderLeft = '';
  cellElement.style.borderRight = '';
  cellElement.style.borderBottom = '';
  
  // 保留悬停效果，但不保留概率/AI高亮层。
  if (cellElement !== currentHoverCell) {
    cellElement.style.border = '';
    cellElement.style.boxShadow = '';
  } else {
    cellElement.style.border = '2px solid #FF0000';
    cellElement.style.boxShadow = '0 0 5px rgba(255, 0, 0, 0.5)';
  }
}


function setCellCovered(r,c) {
  const d = jsCells[r*W + c];
  d.className = "cell";
  d.textContent = "";
  d.innerHTML = "";
  delete d.dataset.number;
  clearAnalysisEffects(d);
  
  // 🔴 重新添加鼠标事件监听器
  d.addEventListener('mouseenter', handleCellHover);
  d.addEventListener('mouseleave', handleCellLeave);
  
  // 🔴 如果这是当前悬停的格子，恢复悬停效果
  if (d === currentHoverCell) {
    d.style.border = '2px solid #FF0000';
    d.style.boxShadow = '0 0 5px rgba(255, 0, 0, 0.5)';
  }
}

function setCellOpen(r,c,n) {
  const d = jsCells[r*W + c];
  d.className = "cell open";
  if (n > 0) { d.textContent = String(n); d.dataset.number = String(n); }
  else { d.textContent = ""; delete d.dataset.number; }
  clearAnalysisEffects(d);
  
  // 🔴 移除悬停效果（因为现在是打开的了）
  if (d === currentHoverCell) {
    currentHoverCell = null;
  }
}

function setCellMine(r,c) {
  const d = jsCells[r*W + c];
  d.className = "cell mine";
  d.textContent = "";
  d.innerHTML = "";
  delete d.dataset.number;
  clearAnalysisEffects(d);
  
  // 🔴 移除悬停效果（因为现在是雷了）
  if (d === currentHoverCell) {
    currentHoverCell = null;
  }
}

function setCellFlag(r,c) {
  const d = jsCells[r*W + c];
  if (d.classList.contains("open") || d.classList.contains("mine")) return;
  d.className = "cell flag";
  d.innerHTML = FLAG_SVG;
  delete d.dataset.number;
  clearAnalysisEffects(d);
  
  // 🔴 移除悬停效果（因为现在是旗子了）
  if (d === currentHoverCell) {
    currentHoverCell = null;
  }
}

function setCellWrongFlag(r,c) {
  const d = jsCells[r*W + c];
  if (!d || d.classList.contains("open") || d.classList.contains("mine")) return;
  d.className = "cell wrong-flag";
  d.innerHTML = FLAG_SVG;
  delete d.dataset.number;
  clearAnalysisEffects(d);
  d.classList.add("wrong-flag");
  if (d === currentHoverCell) currentHoverCell = null;
}

function snapshotForUndo(A) {
  if (!A || typeof A.getState !== "function" || typeof A.setState !== "function") return null;
  return {
    engine: JSON.parse(JSON.stringify(normalizeState(A.getState()))),
    visibleFlags: Array.from(visibleFlagKeys),
    wrongFlagKey,
    globalGameState,
    boardFrozen,
    frozenBoardHasError,
  };
}

function setUndoSnapshot(A) {
  undoState = snapshotForUndo(A);
  btnUndo.style.display = (undoState && typeof A?.setState === "function") ? "block" : "none";
}

function hasFlagInState(st, r, c) {
  return (st.ai_mines || []).some(([rr, cc]) => Number(rr) === r && Number(cc) === c);
}

function getBoardInfo(A) {
  if (!A || typeof A.ms_board_info !== "function") return null;
  try { return toPlain(A.ms_board_info()); }
  catch (e) { dwarn("ms_board_info failed:", e); return null; }
}

function rowToPlainArray(row) {
  row = toPlain(row);
  if (Array.isArray(row)) return row;
  if (row && typeof row.length === "number") return Array.from(row);
  if (row && typeof row === "object") return Object.values(row);
  return [];
}

function isActualMine(A, r, c) {
  const info = getBoardInfo(A);
  if (!info || info.error) return null;
  if (info.first_move_made === false) return "not-ready";
  const layout = rowToPlainArray(info.mines_layout);
  const row = rowToPlainArray(layout[r]);
  if (!row.length) return null;
  return Number(row[c]) === 1;
}

function addFlagToSharedState(A, r, c) {
  // First update the visible/shared UI state. This is the only source of
  // user-visible flags while Play by AI is off.
  visibleFlagKeys.add(key(r, c));

  if (!A || typeof A.getState !== "function") {
    setCellFlag(r, c);
    checkFlagWinCondition(A);
    return;
  }
  const st = normalizeState(A.getState());
  if (!hasFlagInState(st, r, c)) st.ai_mines.push([r, c]);
  if (typeof A.setState === "function") {
    const restored = normalizeState(A.setState(st));
    applyFullState(restored);
    checkFlagWinCondition(A);
  } else {
    setCellFlag(r, c);
    updateGameInfo(st);
    if (!checkFlagWinCondition(A)) refreshAnalysisOverlay();
  }
}

function addFlagsToSharedState(A, cells) {
  const coords = [];
  for (const item of cells || []) {
    const r = Number(item?.[0]), c = Number(item?.[1]);
    if (!Number.isFinite(r) || !Number.isFinite(c)) continue;
    if (r < 0 || r >= H || c < 0 || c >= W) continue;
    const cell = jsCells[r * W + c];
    if (!cell || cell.classList.contains("open") || cell.classList.contains("mine") || cell.classList.contains("flag")) continue;
    visibleFlagKeys.add(key(r, c));
    coords.push([r, c]);
  }
  if (!coords.length) return;

  if (!A || typeof A.getState !== "function") {
    for (const [r, c] of coords) setCellFlag(r, c);
    checkFlagWinCondition(A);
    return;
  }

  const st = normalizeState(A.getState());
  for (const [r, c] of coords) {
    if (!hasFlagInState(st, r, c)) st.ai_mines.push([r, c]);
  }
  if (typeof A.setState === "function") {
    const restored = normalizeState(A.setState(st));
    applyFullState(restored);
    checkFlagWinCondition(A);
  } else {
    for (const [r, c] of coords) setCellFlag(r, c);
    updateGameInfo(st);
    if (!checkFlagWinCondition(A)) refreshAnalysisOverlay();
  }
}

function absorbAIFlagsIntoVisibleState(aiMines) {
  if (!Array.isArray(aiMines)) return;
  for (const item of aiMines) {
    const r = Number(item?.[0]), c = Number(item?.[1]);
    if (Number.isFinite(r) && Number.isFinite(c)) visibleFlagKeys.add(key(r, c));
  }
}

// ---------- Rendering ----------
function applyFullState(st0) {
  const st = normalizeState(st0);
  H = st.h; W = st.w; M = st.mines;
  buildBoardDOM(H, W);
  if (st.seed !== undefined) currentGameSeed = st.seed;

  for (const [r,c,n] of st.revealed) {
    jsRevealed.add(key(r,c));
    (n === -1) ? setCellMine(r,c) : setCellOpen(r,c,n);
  }
  for (const flagKey of visibleFlagKeys) {
    const [r, c] = parseCellKey(flagKey);
    if (Number.isFinite(r) && Number.isFinite(c) && r >= 0 && r < H && c >= 0 && c < W) {
      setCellFlag(r, c);
    }
  }
  if (wrongFlagKey) {
    const [wr, wc] = parseCellKey(wrongFlagKey);
    if (Number.isFinite(wr) && Number.isFinite(wc) && wr >= 0 && wr < H && wc >= 0 && wc < W) {
      setCellWrongFlag(wr, wc);
    }
  }

  setStatus(`Ready | Revealed: ${st.revealed_count} | Lost: ${st.lost} | Won: ${st.won}`);
  updateGameInfo(st);

  // restore analysis overlay immediately
  refreshAnalysisOverlay();
}

function applyStepDelta(d0, options = {}) {
  const d = normalizeDelta(d0);
  const showAIFlags = !!options.showAIFlags;

  // On a losing click, do not redraw the board or refresh probability analysis.
  // Only mark the single cell that caused the failure and enter GAME OVER.
  // This keeps the previous win-rate overlay exactly as it was.
  if (d.lost) {
    let lossCell = Array.isArray(options.failureCell) ? options.failureCell : null;
    if (!lossCell && Array.isArray(d.newly)) {
      const mineEntry = d.newly.find(item => Number(item?.[2]) === -1);
      if (mineEntry) lossCell = [mineEntry[0], mineEntry[1]];
    }
    if (!lossCell && hasMove(d)) lossCell = d.move;
    if (lossCell) {
      const rr = Number(lossCell[0]), cc = Number(lossCell[1]);
      if (Number.isFinite(rr) && Number.isFinite(cc) && rr >= 0 && rr < H && cc >= 0 && cc < W) {
        jsRevealed.add(key(rr, cc));
        setCellMine(rr, cc);
      }
    }
    setStatus("GAME OVER");
    manualModeEnabled = false;
    globalGameState = "GAME OVER";
    updateGameInfo({ revealed_count: d.revealed_count, seed: currentGameSeed });
    return;
  }

  for (const [r,c,n] of d.newly) {
    jsRevealed.add(key(r,c));
    (n === -1) ? setCellMine(r,c) : setCellOpen(r,c,n);
  }
  if (showAIFlags) {
    absorbAIFlagsIntoVisibleState(d.ai_mines);
    for (const [r,c] of d.ai_mines) setCellFlag(r,c);
  }

  const flaggedWin = !d.won && !d.stuck && checkFlagWinCondition(getApi());

  if (d.won) { setStatus("YOU WIN"); manualModeEnabled = false; globalGameState = "YOU WIN"; victoryProbabilityMode = allActualMinesAreVisibleFlags(getApi()) ? "flags" : "remaining-mines"; if (showWinRateEnabled) applyWinningProbabilityOverlay(victoryProbabilityMode);}
  else if (d.stuck) { setStatus("STUCK (no moves)"); manualModeEnabled = false; globalGameState = "STUCK";}
  else if (flaggedWin) { setStatus("YOU WIN"); manualModeEnabled = false; globalGameState = "YOU WIN"; victoryProbabilityMode = "flags"; if (showWinRateEnabled) applyWinningProbabilityOverlay(victoryProbabilityMode); }
  else {setStatus(`Running | Revealed: ${d.revealed_count}`); globalGameState = "READY";}

  updateGameInfo({ revealed_count: d.revealed_count, seed: currentGameSeed });

  // 不再立即刷新分析覆盖层，改为在适当时机批量刷新
}

// ---------- Analysis Overlay (same behavior as your old applyAnalysisOverlay) ----------
function clearOverlayCell(cell) {
  if (!cell) return;
  if (!cell.classList.contains('analyzed') && !cell.classList.contains('next-move')) return;
  const keepVisibleText = cell.classList.contains('open') ||
    cell.classList.contains('mine') ||
    cell.classList.contains('flag');
  cell.classList.remove('analyzed', 'next-move');
  if (!keepVisibleText) cell.textContent = '';
  clearAnalysisEffects(cell);
}

function clearOverlayMarks() {
  for (const cell of overlayCellsByKey.values()) clearOverlayCell(cell);
  overlayCellsByKey.clear();
  overlayStateByKey.clear();
}

function overlaySignature(item) {
  return `${item.text || ''}|${item.probColor || ''}|${item.nextMove ? 1 : 0}`;
}

function setOverlayDesired(desired, r, c, item) {
  if (!Number.isFinite(r) || !Number.isFinite(c) || r < 0 || r >= H || c < 0 || c >= W) return;
  const cell = jsCells[r * W + c];
  if (!isHiddenPlayableCell(cell)) return;
  const k = key(r, c);
  const current = desired.get(k) || { text: '', probColor: '', nextMove: false };
  desired.set(k, { ...current, ...item });
}

function applyOverlayItem(cell, item) {
  cell.classList.add('analyzed');
  cell.classList.toggle('next-move', !!item.nextMove);
  cell.style.setProperty('--prob-color', item.probColor || 'transparent');
  cell.textContent = item.text || '';
}

function applyOverlayPatch(desired) {
  for (const [k, cell] of Array.from(overlayCellsByKey.entries())) {
    if (!desired.has(k) || !isHiddenPlayableCell(cell)) {
      clearOverlayCell(cell);
      overlayCellsByKey.delete(k);
      overlayStateByKey.delete(k);
    }
  }

  for (const [k, item] of desired.entries()) {
    const [r, c] = parseCellKey(k);
    const cell = jsCells[r * W + c];
    if (!isHiddenPlayableCell(cell)) continue;

    const sig = overlaySignature(item);
    if (overlayStateByKey.get(k) === sig &&
        overlayCellsByKey.get(k) === cell &&
        cell.classList.contains('analyzed') &&
        cell.classList.contains('next-move') === !!item.nextMove) {
      continue;
    }

    applyOverlayItem(cell, item);
    overlayCellsByKey.set(k, cell);
    overlayStateByKey.set(k, sig);
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

function addCoordArrayToKeySet(target, arr) {
  arr = toPlain(arr);
  if (!arr || typeof arr[Symbol.iterator] !== 'function') return;
  for (const item0 of arr) {
    const item = toPlain(item0);
    if (!item || typeof item[Symbol.iterator] !== 'function') continue;
    const vals = Array.from(item).map(toPlain);
    if (vals.length < 2) continue;
    const r = Number(vals[0]), c = Number(vals[1]);
    if (!Number.isFinite(r) || !Number.isFinite(c)) continue;
    if (r < 0 || r >= H || c < 0 || c >= W) continue;
    target.add(key(r, c));
  }
}

function engineKnownMineKeysFromState() {
  const out = new Set();
  const A = getApi();
  if (!A || typeof A.getState !== 'function') return out;
  try {
    const st = normalizeState(A.getState());
    addCoordArrayToKeySet(out, st.ai_mines || []);
  } catch (e) {
    dwarn('failed to read engine-known mines for overlay:', e);
  }
  return out;
}

function analysisKnownMineKeys(d) {
  const out = engineKnownMineKeysFromState();
  // Future-proof against either kernel returning deterministic mine fields directly.
  for (const field of ['ai_mines', 'mines', 'known_mines', 'certain_mines', 'forced_mines']) {
    addCoordArrayToKeySet(out, d?.[field] || []);
  }
  return out;
}

function cellIsOpenByKey(k) {
  const [r, c] = parseCellKey(k);
  const cell = jsCells[r * W + c];
  return !!cell?.classList.contains('open');
}

function cellIsVisibleFlagByKey(k) {
  const [r, c] = parseCellKey(k);
  return isVisibleFlag(r, c);
}

function hiddenNeighborsOf(r, c) {
  const out = [];
  for (let rr = r - 1; rr <= r + 1; rr++) {
    if (rr < 0 || rr >= H) continue;
    for (let cc = c - 1; cc <= c + 1; cc++) {
      if (cc < 0 || cc >= W || (rr === r && cc === c)) continue;
      const cell = jsCells[rr * W + cc];
      if (!cell || cell.classList.contains('open') || cell.classList.contains('mine')) continue;
      out.push(key(rr, cc));
    }
  }
  return out;
}

function knownMinesFromEngineState() {
  const out = new Set(visibleFlagKeys);
  const A = getApi();
  if (!A || typeof A.getState !== 'function') return out;
  try {
    const st = normalizeState(A.getState());
    for (const [r, c] of st.ai_mines || []) {
      if (Number.isFinite(Number(r)) && Number.isFinite(Number(c))) out.add(key(Number(r), Number(c)));
    }
  } catch (e) {
    dwarn('failed to read inferred mines for overlay:', e);
  }
  return out;
}

function sortedKeyList(setObj) {
  return Array.from(setObj).sort();
}

function sentenceSignature(cells, count) {
  return `${count}|${sortedKeyList(cells).join(';')}`;
}

function addSentenceUnique(sentences, seen, cells, count) {
  const clean = new Set(cells);
  if (clean.size === 0) return false;
  if (count < 0 || count > clean.size) return false;
  const sig = sentenceSignature(clean, count);
  if (seen.has(sig)) return false;
  seen.add(sig);
  sentences.push({ cells: clean, count });
  return true;
}

function inferDeterministicCertainties() {
  // Supplements the probability overlay with cells the engine/visible board can
  // already prove as 100% mine or 0% mine. This does not reveal or flag anything.
  const knownMines = knownMinesFromEngineState();
  const knownSafes = new Set();
  const sentences = [];
  const seen = new Set();

  const rebuildBaseSentences = () => {
    for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) {
      const cell = jsCells[r * W + c];
      if (!cell?.classList.contains('open')) continue;
      const n = Number(cell.dataset.number || 0);
      if (!Number.isFinite(n)) continue;

      let knownMineCount = 0;
      const unknown = [];
      for (const nb of hiddenNeighborsOf(r, c)) {
        if (knownMines.has(nb)) knownMineCount++;
        else if (!knownSafes.has(nb)) unknown.push(nb);
      }
      addSentenceUnique(sentences, seen, unknown, n - knownMineCount);
    }
  };

  rebuildBaseSentences();

  for (let iter = 0; iter < 20; iter++) {
    let changed = false;

    // Normalize sentences after any newly inferred certainties.
    for (const s of sentences) {
      for (const m of knownMines) {
        if (s.cells.delete(m)) { s.count -= 1; changed = true; }
      }
      for (const sf of knownSafes) {
        if (s.cells.delete(sf)) changed = true;
      }
    }

    for (const s of sentences) {
      if (s.cells.size === 0) continue;
      if (s.count === 0) {
        for (const c of s.cells) if (!knownSafes.has(c) && !knownMines.has(c)) { knownSafes.add(c); changed = true; }
      } else if (s.count === s.cells.size) {
        for (const c of s.cells) if (!knownMines.has(c)) { knownMines.add(c); changed = true; }
      }
    }

    // Subset inference: if A is subset of B, then B-A has count(B)-count(A).
    const base = sentences.filter(s => s.cells.size > 0);
    const limit = Math.min(base.length, 500);
    for (let i = 0; i < limit; i++) {
      for (let j = 0; j < limit; j++) {
        if (i === j) continue;
        const a = base[i], b = base[j];
        if (a.cells.size >= b.cells.size) continue;
        let subset = true;
        for (const x of a.cells) if (!b.cells.has(x)) { subset = false; break; }
        if (!subset) continue;
        const diff = new Set([...b.cells].filter(x => !a.cells.has(x)));
        if (addSentenceUnique(sentences, seen, diff, b.count - a.count)) {
          changed = true;
          if (sentences.length > 2000) break;
        }
      }
      if (sentences.length > 2000) break;
    }

    if (!changed) break;
  }

  // Do not put overlays on opened cells or visible flags.
  for (const k of Array.from(knownMines)) {
    if (cellIsOpenByKey(k) || cellIsVisibleFlagByKey(k)) knownMines.delete(k);
  }
  for (const k of Array.from(knownSafes)) {
    if (cellIsOpenByKey(k) || cellIsVisibleFlagByKey(k) || knownMines.has(k)) knownSafes.delete(k);
  }

  return { mines: knownMines, safes: knownSafes };
}

function isHiddenPlayableCell(cellElement) {
  return !!cellElement &&
    !cellElement.classList.contains('open') &&
    !cellElement.classList.contains('mine') &&
    !cellElement.classList.contains('flag') &&
    !cellElement.classList.contains('wrong-flag');
}

function chooseBestMoveFromProbMap(probMap, preferredMove) {
  const candidateFromMove = (move) => {
    if (!Array.isArray(move) || move.length !== 2) return null;
    const r = Number(move[0]), c = Number(move[1]);
    if (!Number.isFinite(r) || !Number.isFinite(c) || r < 0 || r >= H || c < 0 || c >= W) return null;
    const cell = jsCells[r * W + c];
    return isHiddenPlayableCell(cell) ? [r, c] : null;
  };

  const preferred = candidateFromMove(preferredMove);
  if (preferred) return preferred;

  let best = null;
  let bestP = Infinity;
  for (const [k, p0] of probMap.entries()) {
    const [r, c] = parseCellKey(k);
    if (!Number.isFinite(r) || !Number.isFinite(c) || r < 0 || r >= H || c < 0 || c >= W) continue;
    const cell = jsCells[r * W + c];
    if (!isHiddenPlayableCell(cell)) continue;
    const p = Math.max(0, Math.min(1, Number(p0)));
    if (!Number.isFinite(p)) continue;
    if (p < bestP) {
      bestP = p;
      best = [r, c];
      if (p === 0) break;
    }
  }
  return best;
}

function styleProbabilityCell(cellElement, pp) {
  cellElement.classList.add('analyzed');
  cellElement.style.setProperty('--prob-color', getProbColor(pp));
  cellElement.textContent = Math.round(pp * 100).toString().padStart(2, '0');
}

function hiddenPlayableCoordsForCompletion() {
  const out = [];
  for (let r = 0; r < H; r++) {
    for (let c = 0; c < W; c++) {
      const cellElement = jsCells[r * W + c];
      if (isHiddenPlayableCell(cellElement)) out.push([r, c]);
    }
  }
  return out;
}

function adjacentMineCountForCompletion(A, r, c) {
  const layout = readMineLayout(A);
  let count = 0;
  for (let rr = r - 1; rr <= r + 1; rr++) {
    if (rr < 0 || rr >= H) continue;
    for (let cc = c - 1; cc <= c + 1; cc++) {
      if (cc < 0 || cc >= W || (rr === r && cc === c)) continue;
      if (layout && layout.length) {
        const row = rowToPlainArray(layout[rr]);
        if (Number(row[cc]) === 1) count++;
      } else if (visibleFlagKeys.has(key(rr, cc))) {
        // Frozen uploaded boards may not have a verified seed layout. If the
        // board is already won by all mines being flagged, the visible flags are
        // enough to compute the safe cells' displayed numbers.
        count++;
      }
    }
  }
  return count;
}

function syncCompletedWinToEngine(mode, completedCells) {
  const A = getApi();
  if (!A || typeof A.getState !== "function" || typeof A.setState !== "function") return;
  try {
    const st = normalizeState(A.getState());
    const revealedKeys = new Set((st.revealed || []).map(([r, c]) => key(Number(r), Number(c))));
    const flagKeys = new Set((st.ai_mines || []).map(([r, c]) => key(Number(r), Number(c))));

    if (mode === "flags") {
      for (const [r, c, n] of completedCells || []) {
        const k = key(r, c);
        if (!revealedKeys.has(k)) {
          st.revealed.push([r, c, n]);
          revealedKeys.add(k);
        }
      }
      st.revealed_count = st.revealed.length;
    } else if (mode === "remaining-mines") {
      for (const [r, c] of completedCells || []) {
        const k = key(r, c);
        if (!flagKeys.has(k)) {
          st.ai_mines.push([r, c]);
          flagKeys.add(k);
        }
      }
    }
    st.won = true;
    st.lost = false;
    A.setState(st);
  } catch (e) {
    dwarn("failed to sync completed win state:", e);
  }
}

function applyWinningProbabilityOverlay(mode) {
  // Historical name kept because many win paths call this function.
  // New behavior: when Win rate is on and the game is won, finish the visible
  // board instead of leaving 00/100 probability text on the remaining cells.
  clearOverlayMarks();
  if (!showWinRateEnabled) return;

  const A = getApi();
  const hidden = hiddenPlayableCoordsForCompletion();
  if (!hidden.length) return;

  if (mode === "flags") {
    const opened = [];
    for (const [r, c] of hidden) {
      const n = adjacentMineCountForCompletion(A, r, c);
      jsRevealed.add(key(r, c));
      setCellOpen(r, c, n);
      opened.push([r, c, n]);
    }
    syncCompletedWinToEngine("flags", opened);
  } else {
    const flagged = [];
    for (const [r, c] of hidden) {
      visibleFlagKeys.add(key(r, c));
      setCellFlag(r, c);
      flagged.push([r, c]);
    }
    syncCompletedWinToEngine("remaining-mines", flagged);
  }

  setStatus("YOU WIN");
  manualModeEnabled = false;
  globalGameState = "YOU WIN";
  try {
    updateGameInfo(normalizeState(A?.getState?.() || { revealed_count: jsRevealed.size, seed: currentGameSeed }));
  } catch {
    updateGameInfo({ revealed_count: jsRevealed.size, seed: currentGameSeed });
  }
}

function parsedVictoryProbabilityMode(parsed, boardError = false) {
  if (!parsed || boardError) return null;
  if (parsed.flags?.length === parsed.mines) return "flags";
  if (parsed.revealed?.length === parsed.h * parsed.w - parsed.mines) return "remaining-mines";
  return null;
}

function applyAnalysisOverlay(analysis0) {
  const d = toPlain(analysis0) || {};

  const probMap = new Map();
  const forcedMines = analysisKnownMineKeys(d);
  const forcedSafes = new Set();

  for (const [r, c, p] of parseProbsObject(d.probs)) {
    const k = key(r, c);
    const pp = Math.max(0, Math.min(1, Number(p)));
    probMap.set(k, pp);
    // Exact 0/100 returned by the probability engine should be treated as
    // deterministic, but mines always have priority over safes below.
    if (pp >= 1 - 1e-9) forcedMines.add(k);
    else if (pp <= 1e-9) forcedSafes.add(k);
  }

  // Even if the engine did not assign probabilities to deterministic cells,
  // fill them in for display/highlight: certain mine = 100, certain safe = 0.
  const certainty = inferDeterministicCertainties();
  for (const k of certainty.mines) forcedMines.add(k);
  for (const k of certainty.safes) forcedSafes.add(k);

  // Priority is important: a mine proven by the engine/front-end inference must
  // never be overwritten by a later safe inference or missing probability entry.
  for (const k of forcedSafes) {
    if (!forcedMines.has(k)) probMap.set(k, 0);
  }
  for (const k of forcedMines) {
    if (!cellIsOpenByKey(k) && !cellIsVisibleFlagByKey(k)) probMap.set(k, 1);
  }

  const desiredOverlay = new Map();

  if (showWinRateEnabled) {
    for (const [k, p] of probMap.entries()) {
      const [r, c] = parseCellKey(k);
      const pp = Math.max(0, Math.min(1, Number(p)));
      if (!Number.isFinite(pp)) continue;
      setOverlayDesired(desiredOverlay, r, c, {
        text: Math.round(pp * 100).toString().padStart(2, '0'),
        probColor: getProbColor(pp),
      });
    }
  }

  if (playByAIEnabled) {
    const bestMove = chooseBestMoveFromProbMap(probMap, d.next_move || d.move || d.best_move);
    if (bestMove) {
      const [nr, nc] = bestMove;
      setOverlayDesired(desiredOverlay, nr, nc, {
        nextMove: true,
        probColor: 'rgba(45, 126, 196, 0.62)',
      });
    }
  }

  applyOverlayPatch(desiredOverlay);
}


function refreshAnalysisOverlay() {
  if (globalGameState === "YOU WIN" && showWinRateEnabled && victoryProbabilityMode) {
    applyWinningProbabilityOverlay(victoryProbabilityMode);
    return;
  }
  if (boardFrozen && frozenBoardHasError) {
    clearOverlayMarks();
    return;
  }
  if (!showWinRateEnabled && !playByAIEnabled) {
    clearOverlayMarks();
    return;
  }
  const A = getApi();
  if (!assertApiReady(A)) return;
  try {
    const a = toPlain(A.getAnalysis());
    applyAnalysisOverlay(a);
  } catch (e) {
    dwarn("getAnalysis/apply overlay failed:", e);
  }
}

function updateModeButtons() {
  if (showWinRateButton) {
    showWinRateButton.textContent = showWinRateEnabled ? "Win rate on" : "Win rate off";
    showWinRateButton.classList.toggle("mode-on", showWinRateEnabled);
    showWinRateButton.classList.toggle("mode-off", !showWinRateEnabled);
    showWinRateButton.setAttribute("aria-pressed", String(showWinRateEnabled));
  }
  if (playByAIButton) {
    playByAIButton.textContent = playByAIEnabled ? "AI on" : "AI off";
    playByAIButton.classList.toggle("mode-on", playByAIEnabled);
    playByAIButton.classList.toggle("mode-off", !playByAIEnabled);
    playByAIButton.setAttribute("aria-pressed", String(playByAIEnabled));
  }
}

function updateModeFromButtons() {
  updateModeButtons();
}

// ---------- Naive opened-number shortcuts ----------
function neighborCoords(r, c) {
  const out = [];
  for (let rr = r - 1; rr <= r + 1; rr++) {
    if (rr < 0 || rr >= H) continue;
    for (let cc = c - 1; cc <= c + 1; cc++) {
      if (cc < 0 || cc >= W || (rr === r && cc === c)) continue;
      out.push([rr, cc]);
    }
  }
  return out;
}

function hiddenUnflaggedNeighborCoords(r, c) {
  const out = [];
  for (const [rr, cc] of neighborCoords(r, c)) {
    const cell = jsCells[rr * W + cc];
    if (!cell) continue;
    if (cell.classList.contains("open") || cell.classList.contains("mine") ||
        cell.classList.contains("flag") || cell.classList.contains("wrong-flag")) continue;
    out.push([rr, cc]);
  }
  return out;
}

function visibleFlagNeighborCount(r, c) {
  let count = 0;
  for (const [rr, cc] of neighborCoords(r, c)) {
    if (isVisibleFlag(rr, cc)) count++;
  }
  return count;
}

async function runAISafeMovesIfEnabled(A) {
  if (!playByAIEnabled || typeof A?.makeSafeMove !== "function") return;
  while (true) {
    const ds = normalizeDelta(A.makeSafeMove());
    applyStepDelta(ds, { showAIFlags: true });
    if (ds.lost || ds.won || ds.stuck || isGameEnded()) break;
    if (!hasMove(ds)) break;
  }
  // AI moves can finish the game through a final reveal or by completing all flags.
  // Confirm the engine state after the automatic move batch before refreshing overlays.
  confirmGameEndFromKernel(A);
}

async function handleOpenedNumberClick(target, r, c) {
  const n = Number(target.dataset.number || 0);
  if (!Number.isFinite(n)) return false;

  const hidden = hiddenUnflaggedNeighborCoords(r, c);
  if (!hidden.length) return false;

  const flagged = visibleFlagNeighborCount(r, c);
  const A = getApi();
  if (!assertApiReady(A)) {
    setStatus("No API available for current kernel");
    return true;
  }

  // Standard chording: enough flags around this number => reveal all other hidden neighbors.
  if (flagged === n) {
    setUndoSnapshot(A);
    allowHoverEffect = false;
    try {
      for (const [rr, cc] of hidden) {
        const cell = jsCells[rr * W + cc];
        if (!cell || cell.classList.contains("open") || cell.classList.contains("flag") || cell.classList.contains("mine")) continue;
        const d = normalizeDelta(A.stepAt(rr, cc));
        applyStepDelta(d, { showAIFlags: playByAIEnabled, failureCell: [rr, cc] });
        if (d.lost || d.won || d.stuck || isGameEnded()) break;
      }
      if (!isGameEnded()) {
        await runAISafeMovesIfEnabled(A);
        if (!isGameEnded()) refreshAnalysisOverlay();
      }
    } finally {
      allowHoverEffect = true;
    }
    return true;
  }

  // Naive mine marking: all remaining hidden neighbors must be mines.
  if (flagged + hidden.length === n) {
    setUndoSnapshot(A);
    addFlagsToSharedState(A, hidden);
    if (!isGameEnded()) {
      setStatus(`Ready | Auto-marked ${hidden.length} mine${hidden.length > 1 ? "s" : ""}`);
      refreshAnalysisOverlay();
      try { updateGameInfo(normalizeState(A.getState())); } catch {}
    }
    return true;
  }

  return false;
}

// ---------- Interaction ----------
boardEl.addEventListener("click", handleManualClick);
boardEl.addEventListener("contextmenu", handleManualFlag);

async function handleManualClick(event) {
    if (isGameEnded() || boardFrozen) return;
    if (!manualModeEnabled || stepping) return;

    const target = event.target.closest?.(".cell") || event.target;
    if (!target.classList.contains("cell") ||
        target.classList.contains("flag") ||
        target.classList.contains("mine") ||
        target.classList.contains("wrong-flag")) return;

    const r = parseInt(target.dataset.r, 10);
    const c = parseInt(target.dataset.c, 10);
    if (!Number.isFinite(r) || !Number.isFinite(c)) return;

    if (target.classList.contains("open")) {
        await handleOpenedNumberClick(target, r, c);
        return;
    }

    // 🔴 立即清理被点击格子的红色悬停效果
    if (target === currentHoverCell) {
        target.style.border = '';
        target.style.boxShadow = '';
        currentHoverCell = null;
    }

    allowHoverEffect = false;

    const A = getApi();
    if (!assertApiReady(A)) {
        allowHoverEffect = true;
        return setStatus("No API available for current kernel");
    }

    setUndoSnapshot(A);

    try {
        applyStepDelta(A.stepAt(r, c), { showAIFlags: playByAIEnabled, failureCell: [r, c] });
        if (isGameEnded()) {
            return;
        }

        // 只有勾选 Play by AI 时，左键点击才恢复旧版的 AI 自动安全步行为。
        await runAISafeMovesIfEnabled(A);

        if (!isGameEnded()) refreshAnalysisOverlay();
    } finally {
        allowHoverEffect = true;
    }
}

async function handleManualFlag(event) {
    event.preventDefault();
    if (isGameEnded() || boardFrozen) return;
    if (!manualModeEnabled || stepping) return;

    const target = event.target.closest?.(".cell") || event.target;
    if (!target.classList.contains("cell") ||
        target.classList.contains("open") ||
        target.classList.contains("mine") ||
        target.classList.contains("wrong-flag")) return;

    const r = parseInt(target.dataset.r, 10);
    const c = parseInt(target.dataset.c, 10);
    if (!Number.isFinite(r) || !Number.isFinite(c)) return;

    // 信息只新增不减少：已经标记的格子不再取消标记。
    if (target.classList.contains("flag")) return;

    const A = getApi();
    if (!assertApiReady(A)) return setStatus("No API available for current kernel");

    const mineCheck = isActualMine(A, r, c);
    if (mineCheck === "not-ready") {
        setStatus("Reveal one cell first before flagging, because the mine layout is created after the first move.");
        return;
    }
    if (mineCheck === null) {
        setStatus("Cannot verify the flag with the current kernel state.");
        return;
    }

    setUndoSnapshot(A);

    if (mineCheck === true) {
        addFlagToSharedState(A, r, c);
        if (!isGameEnded()) {
          setStatus(`Ready | Correct flag: ${r},${c}`);
          refreshAnalysisOverlay();
          try { updateGameInfo(normalizeState(A.getState())); } catch {}
        }
    } else {
        // Wrong flag failure should only update the problematic cell.
        // Keep the existing probability overlay on every other cell unchanged.
        wrongFlagKey = key(r, c);
        setCellWrongFlag(r, c);
        manualModeEnabled = false;
        globalGameState = "WRONG FLAG";
        setStatus("GAME OVER - Wrong flag");
    }
}

async function undoLastMove() {
  const A = getApi();
  if (!undoState) return;
  if (!assertApiReady(A)) return setStatus("No API available for current kernel");
  if (typeof A.setState !== "function") return setStatus("Undo not supported in this kernel");

  try {
    dlog("undo restore", { kernelType });
    const snap = undoState.engine ? undoState : { engine: undoState, visibleFlags: [] };
    visibleFlagKeys = new Set(snap.visibleFlags || []);
    wrongFlagKey = snap.wrongFlagKey || null;
    globalGameState = snap.globalGameState || "READY";
    boardFrozen = !!snap.boardFrozen;
    frozenBoardHasError = !!snap.frozenBoardHasError;
    const restored = normalizeState(A.setState(snap.engine));
    applyFullState(restored);
    btnUndo.style.display = "none";
    setStatus("Ready");
    if (globalGameState === "WRONG FLAG") setStatus("GAME OVER - Wrong flag");
    undoState = null;
    manualModeEnabled = !boardFrozen && !isGameEnded();

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
function loadScriptOnce(src, selector) {
    return new Promise((resolve, reject) => {
        const existing = selector ? document.querySelector(selector) : null;
        if (existing && existing.dataset.loaded === "1") return resolve();
        if (existing && existing.dataset.loading === "1") {
            existing.addEventListener("load", () => resolve(), { once: true });
            existing.addEventListener("error", () => reject(new Error("Failed to load " + src)), { once: true });
            return;
        }

        const script = existing || document.createElement("script");
        if (selector) script.dataset.pyodideLoader = "1";
        script.dataset.loading = "1";
        script.async = true;
        script.onload = () => {
            script.dataset.loading = "0";
            script.dataset.loaded = "1";
            resolve();
        };
        script.onerror = () => {
            script.remove();
            reject(new Error("Failed to load " + src));
        };
        if (!existing) document.head.appendChild(script);
        script.src = src;
    });
}

async function ensurePyodideLoader() {
    if (typeof window.loadPyodide === "function") return;
    if (pyodideLoaderPromise) return pyodideLoaderPromise;

    pyodideLoaderPromise = (async () => {
        const loaderCandidates = [
            "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js",
            "https://pyodide.org/pyodide/v0.26.4/full/pyodide.js",
            "https://unpkg.com/pyodide@0.26.4/pyodide/full/pyodide.js",
        ];
        let lastErr = null;
        for (const src of loaderCandidates) {
            try {
                await loadScriptOnce(src, 'script[data-pyodide-loader="1"]');
                if (typeof window.loadPyodide === "function") return;
                throw new Error("Pyodide loader did not expose loadPyodide");
            } catch (e) {
                lastErr = e;
                document.querySelector('script[data-pyodide-loader="1"]')?.remove();
                dwarn("pyodide loader failed", src, e);
            }
        }
        throw lastErr || new Error("Failed to load Pyodide loader");
    })();

    try {
        await pyodideLoaderPromise;
    } catch (e) {
        pyodideLoaderPromise = null;
        throw e;
    }
}

async function loadPy() {
    // 🔴 如果已经加载过，直接返回
    if (pyodideLoaded && jsApi) return;
    if (pyodideLoadPromise) return pyodideLoadPromise;

    pyodideLoadPromise = (async () => {
        await ensurePyodideLoader();

        const candidates = [
            "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
            "https://pyodide.org/pyodide/v0.26.4/full/",
            "https://unpkg.com/pyodide@0.26.4/pyodide/full/",
        ];

        setStatus("Loading Pyodide...");
        let lastErr = null;
        for (const indexURL of candidates) {
            try { 
                pyodide = await window.loadPyodide({ indexURL }); 
                lastErr = null; 
                break; 
            }
            catch (e) { 
                lastErr = e; 
                dwarn("loadPyodide failed", indexURL, e); 
            }
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

        pyodideLoaded = true;  // 🔴 标记Pyodide已加载
        dlog("js api ready");
        setStatus("Ready.");
    })();

    try {
        await pyodideLoadPromise;
    } catch (e) {
        if (!pyodideLoaded || !jsApi) pyodideLoadPromise = null;
        throw e;
    }
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
    // 🔴 检查C++ API是否已准备好
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
    
    // 只迁移用户界面中已经可见的旗子。
    // 内核内部推理出的 ai_mines 不能在 Play by AI 关闭时泄露成旗子。
    for (const [r, c] of visibleFlagsAsArray()) {
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
        
        // 🔴 确保C++模块已加载
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
    // 🔴 按需加载Pyodide
    if (!pyodideLoaded) {
        await loadPy();
    }
    
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
    if (switchingKernel) return;
    
    if (isGameEnded()) {
        setStatus("Cannot switch kernel: Game has ended");
        return;
    }
    
    const A = getApi();
    if (!assertApiReady(A)) {
        setStatus("Cannot switch: current kernel not ready");
        return;
    }
    
    // 执行内核切换
    if (kernelType === "js") {
        // 从JS切换到C++
        await switchToCppKernel();
    } else {
        // 从C++切换到JS（按需加载Pyodide）
        await switchToJsKernel();
    }
    undoState = null;
    btnUndo.style.display = "none";
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


function currentReplaySeed() {
  if (currentGameSeed !== undefined && currentGameSeed !== null && String(currentGameSeed).trim() !== "") return currentGameSeed;
  const seedStr = (inpSeed?.value ?? "").trim();
  if (seedStr !== "") return clampInt(seedStr, -2147483648, 2147483647, null);
  return null;
}

function askNewGameChoice() {
  return new Promise((resolve) => {
    let modal = document.getElementById("newGameModal");
    if (!modal) {
      modal = document.createElement("div");
      modal.id = "newGameModal";
      modal.className = "modal-overlay";
      modal.innerHTML = `
        <div class="modal-card" role="dialog" aria-modal="true" aria-labelledby="newGameModalTitle">
          <div id="newGameModalTitle" class="modal-title">New game?</div>
          <div class="modal-actions">
            <button type="button" class="btn-primary" data-choice="new">new seed</button>
            <button type="button" class="btn-secondary" data-choice="old">old seed</button>
            <button type="button" class="btn-secondary" data-choice="resume">resume</button>
          </div>
        </div>`;
      document.body.appendChild(modal);
    }
    modal.style.display = "flex";
    const done = (choice) => {
      modal.style.display = "none";
      modal.removeEventListener("click", onClick);
      document.removeEventListener("keydown", onKey);
      resolve(choice);
    };
    const onClick = (ev) => {
      const btn = ev.target.closest?.("button[data-choice]");
      if (btn) done(btn.dataset.choice || "resume");
    };
    const onKey = (ev) => {
      if (ev.key === "Escape") done("resume");
    };
    modal.addEventListener("click", onClick);
    document.addEventListener("keydown", onKey);
  });
}

async function handleNewGameButtonClick() {
  const replaySeed = currentReplaySeed();
  const choice = await askNewGameChoice();
  if (choice === "resume") return;
  if (choice === "old") {
    if (replaySeed === null) return createNewGame({ forceRandomSeed: true });
    return createNewGame({ seedOverride: replaySeed });
  }
  return createNewGame({ forceRandomSeed: true });
}

async function createNewGame(options = {}) {
  boardFrozen = false;
  frozenBoardHasError = false;
  victoryProbabilityMode = null;
  manualModeEnabled = true;
  undoState = null;
  btnUndo.style.display = "none";
  visibleFlagKeys.clear();
  wrongFlagKey = null;
  globalGameState = "READY"; // 初始状态

  const h = clampInt(inpH.value, 5, 200, 25);
  const w = clampInt(inpW.value, 5, 200, 40);
  const m = clampInt(inpM.value, 1, h*w - 1, Math.min(200, h*w-1));
  const seedStr = (inpSeed?.value ?? "").trim();
  
  let seed;
  if (options && Object.prototype.hasOwnProperty.call(options, "seedOverride")) {
    seed = clampInt(options.seedOverride, -2147483648, 2147483647, null);
  } else if (options?.forceRandomSeed) {
    seed = await generateCryptoSeed();
    console.log("Generated crypto seed:", seed);
  } else if (seedStr !== "") {
    seed = clampInt(seedStr, -2147483648, 2147483647, null);
  } else {
    // 使用 Crypto API 生成高质量随机种子
    seed = await generateCryptoSeed();
    console.log("Generated crypto seed:", seed);
  }
  
  const firstmv = firstZeroCheckbox?.checked ? 1 : 2;

  inpH.value = String(h); inpW.value = String(w); inpM.value = String(m);
  if (inpSeed) inpSeed.value = String(seed);

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


// ---------- .mine import / export ----------
function showMineFileError(message) {
  setStatus(message);
  try { window.alert(message); } catch {}
}

function stateRevealedMap(st) {
  const out = new Map();
  for (const item of (normalizeState(st).revealed || [])) {
    const r = Number(item?.[0]), c = Number(item?.[1]), n = Number(item?.[2]);
    if (!Number.isFinite(r) || !Number.isFinite(c) || !Number.isFinite(n)) continue;
    out.set(key(r, c), n);
  }
  return out;
}

function makeMineFileText() {
  const A = getApi();
  if (!assertApiReady(A)) throw new Error("No API available for current kernel");
  const st = normalizeState(A.getState());
  const revealed = stateRevealedMap(st);
  const lines = [`${W}x${H}x${M}`];

  for (let r = 0; r < H; r++) {
    let row = "";
    for (let c = 0; c < W; c++) {
      const k = key(r, c);
      const cell = jsCells[r * W + c];
      if (visibleFlagKeys.has(k) || wrongFlagKey === k || cell?.classList.contains("mine")) {
        row += "F";
      } else if (revealed.has(k)) {
        const n = revealed.get(k);
        row += (Number.isInteger(n) && n >= 0 && n <= 8) ? String(n) : "H";
      } else {
        row += "H";
      }
    }
    lines.push(row);
  }
  return lines.join("\n") + "\n";
}

function downloadMineFile() {
  try {
    const seed = currentGameSeed ?? inpSeed?.value?.trim();
    if (seed === undefined || seed === null || String(seed).trim() === "") {
      showMineFileError("No seed available.");
      return;
    }
    const content = makeMineFileText();
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${seed}.mine`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
    setStatus(`Downloaded ${seed}.mine`);
  } catch (e) {
    derr("download .mine failed", e);
    showMineFileError(e?.message || "Download failed.");
  }
}

function parseMineFileText(text, filename = "") {
  const seedMatch = String(filename || "").match(/^(-?\d+)\.mine$/i);
  if (!seedMatch) throw new Error("Invalid file format.");
  const seed = Number.parseInt(seedMatch[1], 10);
  if (!Number.isFinite(seed)) throw new Error("Invalid file format.");

  const rawLines = String(text ?? "").replace(/^\uFEFF/, "").split(/\r?\n/);
  while (rawLines.length && rawLines[rawLines.length - 1].trim() === "") rawLines.pop();
  if (!rawLines.length) throw new Error("Invalid file format.");

  const header = rawLines[0].trim();
  const m = header.match(/^(\d+)x(\d+)x(\d+)$/i);
  if (!m) throw new Error("Invalid file format.");
  const w = Number.parseInt(m[1], 10);
  const h = Number.parseInt(m[2], 10);
  const mines = Number.parseInt(m[3], 10);
  if (!Number.isInteger(w) || !Number.isInteger(h) || !Number.isInteger(mines) || w < 5 || h < 5 || mines < 1 || mines >= w * h) {
    throw new Error("Invalid file format.");
  }
  if (rawLines.length !== h + 1) throw new Error("Invalid file format.");

  const cells = [];
  const revealed = [];
  const flags = [];
  const openedCandidates = [];
  const hiddenCandidates = [];

  for (let r = 0; r < h; r++) {
    const row = rawLines[r + 1].trim();
    if (row.length !== w) throw new Error("Invalid file format.");
    const arr = [];
    for (let c = 0; c < w; c++) {
      const ch = row[c].toUpperCase();
      if (!/[0-8HF]/.test(ch)) throw new Error("Invalid file format.");
      arr.push(ch);
      if (/[0-8]/.test(ch)) {
        const n = Number.parseInt(ch, 10);
        revealed.push([r, c, n]);
        openedCandidates.push([r, c, n]);
      } else if (ch === "F") {
        flags.push([r, c]);
      } else {
        hiddenCandidates.push([r, c]);
      }
    }
    cells.push(arr);
  }

  return { seed, w, h, mines, cells, revealed, flags, openedCandidates, hiddenCandidates };
}

function mineSetFromState(st) {
  const out = new Set();
  const arr = toPlain(st?.mines_pos) || [];
  for (const item0 of arr) {
    const item = toPlain(item0);
    if (!item || typeof item[Symbol.iterator] !== "function") continue;
    const vals = Array.from(item).map(toPlain);
    if (vals.length < 2) continue;
    const r = Number(vals[0]), c = Number(vals[1]);
    if (Number.isFinite(r) && Number.isFinite(c)) out.add(key(r, c));
  }
  return out;
}

function countAdjacentMinesForSet(mineSet, h, w, r, c) {
  let count = 0;
  for (let rr = r - 1; rr <= r + 1; rr++) {
    if (rr < 0 || rr >= h) continue;
    for (let cc = c - 1; cc <= c + 1; cc++) {
      if (cc < 0 || cc >= w || (rr === r && cc === c)) continue;
      if (mineSet.has(key(rr, cc))) count++;
    }
  }
  return count;
}

function validateMineFileAgainstLayout(parsed, st) {
  const mineSet = mineSetFromState(st);
  if (mineSet.size !== parsed.mines) return false;

  for (const [r, c, n] of parsed.revealed) {
    if (mineSet.has(key(r, c))) return false;
    if (countAdjacentMinesForSet(mineSet, parsed.h, parsed.w, r, c) !== n) return false;
  }
  for (const [r, c] of parsed.flags) {
    if (!mineSet.has(key(r, c))) return false;
  }
  return true;
}


function validateVisibleMineFileConstraints(parsed) {
  const flagSet = new Set(parsed.flags.map(([r, c]) => key(r, c)));
  const hiddenSet = new Set(parsed.hiddenCandidates.map(([r, c]) => key(r, c)));
  for (const [r, c, n] of parsed.revealed) {
    let flagged = 0;
    let hidden = 0;
    for (let rr = r - 1; rr <= r + 1; rr++) {
      if (rr < 0 || rr >= parsed.h) continue;
      for (let cc = c - 1; cc <= c + 1; cc++) {
        if (cc < 0 || cc >= parsed.w || (rr === r && cc === c)) continue;
        const k = key(rr, cc);
        if (flagSet.has(k)) flagged++;
        else if (hiddenSet.has(k)) hidden++;
      }
    }
    if (flagged > n) return false;
    if (flagged + hidden < n) return false;
  }
  return true;
}

function firstMoveCandidatesForMineFile(parsed) {
  const zeros = parsed.openedCandidates.filter(item => item[2] === 0).map(([r, c]) => [r, c]);
  const nonZeroOpened = parsed.openedCandidates.filter(item => item[2] !== 0).map(([r, c]) => [r, c]);
  if (zeros.length || nonZeroOpened.length) return [...zeros, ...nonZeroOpened];
  return parsed.hiddenCandidates.slice(0, 2000);
}

function firstMoveModesForUpload() {
  const current = firstZeroCheckbox?.checked ? 1 : 2;
  return current === 1 ? [1, 2] : [2, 1];
}

function restoreSnapshotAfterFailedUpload(A, snapshot) {
  if (!snapshot || !A || typeof A.setState !== "function") return;
  try {
    visibleFlagKeys = new Set(snapshot.visibleFlags || []);
    wrongFlagKey = snapshot.wrongFlagKey || null;
    globalGameState = snapshot.globalGameState || "READY";
    boardFrozen = !!snapshot.boardFrozen;
    frozenBoardHasError = !!snapshot.frozenBoardHasError;
    currentGameSeed = snapshot.engine?.seed ?? currentGameSeed;
    const restored = normalizeState(A.setState(snapshot.engine));
    applyFullState(restored);
    if (globalGameState === "GAME OVER") setStatus("GAME OVER");
    else if (globalGameState === "YOU WIN") setStatus("YOU WIN");
    else if (globalGameState === "WRONG FLAG") setStatus("GAME OVER - Wrong flag");
  } catch (e) {
    dwarn("failed to restore board after upload error:", e);
  }
}

function parsedFieldRows(parsed) {
  return parsed.cells.map(row => row.join(''));
}

function visibleBoardStateFromParsed(parsed, options = {}) {
  const flags = parsed.flags.map(([r, c]) => [r, c]);
  const wonByReveal = parsed.revealed.length === parsed.h * parsed.w - parsed.mines;
  const wonByFlags = !options.boardError && flags.length === parsed.mines;
  return {
    h: parsed.h,
    w: parsed.w,
    mines: parsed.mines,
    first: true,
    firstmv: Number.isFinite(options.firstmv) ? options.firstmv : (firstZeroCheckbox?.checked ? 1 : 2),
    lost: false,
    won: wonByReveal || wonByFlags,
    seed: parsed.seed,
    mines_pos: options.mines_pos || [],
    revealed: parsed.revealed,
    ai_mines: flags,
  };
}

function applyUploadedBoardUi(parsed, restoredState, options = {}) {
  H = parsed.h;
  W = parsed.w;
  M = parsed.mines;
  currentGameSeed = parsed.seed;
  inpH.value = String(H);
  inpW.value = String(W);
  inpM.value = String(M);
  inpSeed.value = String(parsed.seed);
  if (firstZeroCheckbox && Number.isFinite(options.firstmv)) firstZeroCheckbox.checked = options.firstmv === 1;

  visibleFlagKeys = new Set(parsed.flags.map(([r, c]) => key(r, c)));
  wrongFlagKey = null;
  boardFrozen = !!options.frozen;
  frozenBoardHasError = !!options.boardError;
  victoryProbabilityMode = restoredState?.won ? (options.winOverlayMode || parsedVictoryProbabilityMode(parsed, frozenBoardHasError) || "remaining-mines") : null;
  globalGameState = restoredState?.won ? "YOU WIN" : "READY";
  manualModeEnabled = !boardFrozen && !restoredState?.won;
  undoState = null;
  btnUndo.style.display = "none";

  const stForRender = normalizeState({ ...(restoredState || {}), seed: parsed.seed });
  applyFullState(stForRender);

  if (frozenBoardHasError) {
    showWinRateEnabled = false;
    updateModeFromButtons();
    clearOverlayMarks();
  } else if (showWinRateEnabled && restoredState?.won && victoryProbabilityMode) {
    applyWinningProbabilityOverlay(victoryProbabilityMode);
  } else if (showWinRateEnabled) {
    refreshAnalysisOverlay();
  } else {
    clearOverlayMarks();
  }

  updateGameInfo({ ...stForRender, seed: parsed.seed });

  if (stForRender.won) setStatus("YOU WIN");
  else if (boardFrozen && frozenBoardHasError) setStatus(`Loaded ${parsed.seed}.mine | Board frozen | Board error.`);
  else if (boardFrozen) setStatus(`Loaded ${parsed.seed}.mine | Board frozen`);
  else setStatus(`Loaded ${parsed.seed}.mine`);
}

function loadVisibleBoardWithoutSeedLayout(A, parsed, boardError) {
  const loadData = {
    height: parsed.h,
    width: parsed.w,
    mines: parsed.mines,
    seed: parsed.seed,
    field: parsedFieldRows(parsed),
    first_move_made: true,
  };

  let restored;
  if (typeof A.ms_load_board === "function") {
    restored = normalizeState(A.ms_load_board(loadData));
  } else if (typeof A.setState === "function") {
    // Very old fallback: this can render the board but should remain frozen.
    restored = normalizeState(A.setState(visibleBoardStateFromParsed(parsed, { mines_pos: [], exactMatch: false })));
  } else {
    throw new Error("Upload is not supported in this kernel.");
  }
  restored.seed = parsed.seed;
  restored.won = !!visibleBoardStateFromParsed(parsed, { boardError }).won;
  return restored;
}

async function loadMineFile(parsed) {
  const A = getApi();
  if (!assertApiReady(A)) throw new Error("No API available for current kernel");
  if (typeof A.setState !== "function" && typeof A.ms_load_board !== "function") {
    throw new Error("Upload is not supported in this kernel.");
  }

  const visibleBoardError = !validateVisibleMineFileConstraints(parsed);
  const candidates = firstMoveCandidatesForMineFile(parsed);
  if (!candidates.length) throw new Error("Invalid file format.");

  let matchedState = null;
  let matchedMode = null;
  const modes = firstMoveModesForUpload();

  // Try to prove that this uploaded board exactly matches the requested seed.
  // Only a proven exact match stays playable; every other parsed board is loaded frozen.
  if (!visibleBoardError && typeof A.newGame === "function") {
    for (const mode of modes) {
      for (const [fr, fc] of candidates) {
        normalizeState(A.newGame(parsed.h, parsed.w, parsed.mines, parsed.seed, mode));
        const d = normalizeDelta(A.stepAt(fr, fc));
        if (d.lost) continue;
        const st = normalizeState(A.getState());
        if (validateMineFileAgainstLayout(parsed, st)) {
          matchedState = st;
          matchedMode = mode;
          break;
        }
      }
      if (matchedState) break;
    }
  }

  if (matchedState) {
    const targetState = visibleBoardStateFromParsed(parsed, {
      exactMatch: true,
      boardError: false,
      firstmv: matchedMode,
      mines_pos: normalizeState(matchedState).mines_pos || [],
    });
    const restored = normalizeState(A.setState(targetState));
    restored.seed = parsed.seed;
    applyUploadedBoardUi(parsed, restored, { frozen: false, boardError: false, firstmv: matchedMode, winOverlayMode: parsedVictoryProbabilityMode(parsed, false) });
    return;
  }

  // Not an exact seed match: still upload the board, but freeze it.
  // If the visible constraints are contradictory, force Win rate off and block it later.
  if (visibleBoardError) {
    showWinRateEnabled = false;
    updateModeFromButtons();
  }

  const restored = loadVisibleBoardWithoutSeedLayout(A, parsed, visibleBoardError);
  applyUploadedBoardUi(parsed, restored, {
    frozen: true,
    boardError: visibleBoardError,
    firstmv: firstZeroCheckbox?.checked ? 1 : 2,
    winOverlayMode: parsedVictoryProbabilityMode(parsed, visibleBoardError),
  });
}

async function uploadMineFile(file) {
  if (!file) return;
  try {
    const text = await file.text();
    const parsed = parseMineFileText(text, file.name);
    await loadMineFile(parsed);
  } catch (e) {
    derr("upload .mine failed", e);
    const msg = (e?.message === "Seed mismatch." || e?.message === "Invalid file format.") ? e.message : "Invalid file format.";
    showMineFileError(msg);
  } finally {
    if (mineUploadInput) mineUploadInput.value = "";
  }
}


// ---------- Wire ----------
btnNewGame?.addEventListener("click", handleNewGameButtonClick);
btnUndo?.addEventListener("click", undoLastMove);
btnSwitchKernel?.addEventListener("click", switchKernel);
btnDownloadMine?.addEventListener("click", downloadMineFile);
btnUploadMine?.addEventListener("click", () => mineUploadInput?.click());
mineUploadInput?.addEventListener("change", () => uploadMineFile(mineUploadInput.files?.[0]));
showWinRateButton?.addEventListener("click", () => {
  if (!showWinRateEnabled && boardFrozen && frozenBoardHasError) {
    showWinRateEnabled = false;
    updateModeFromButtons();
    clearOverlayMarks();
    showMineFileError("Board error.");
    return;
  }
  showWinRateEnabled = !showWinRateEnabled;
  updateModeFromButtons();
  refreshAnalysisOverlay();
});
playByAIButton?.addEventListener("click", () => {
  playByAIEnabled = !playByAIEnabled;
  updateModeFromButtons();
  refreshAnalysisOverlay();
});

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
showWinRateEnabled = true;
playByAIEnabled = false;
updateModeFromButtons();

// ---------- Keyboard Shortcuts ----------
document.addEventListener('keydown', function(event) {
  // 防止在输入框中触发快捷键
  if (event.target.tagName === 'INPUT' || event.target.tagName === 'TEXTAREA') {
    return;
  }
  
  // 阻止默认行为（除了特殊键）
  const specialKeys = ['PageUp', 'PageDown', '+', '-'];
  const keyChar = event.key.toLowerCase();
  
  // N - New Game
  if (keyChar === 'n') {
    event.preventDefault();
    handleNewGameButtonClick();
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
  
  // + or PageUp or Numpad+ - Increase Cell Scale
  if (event.key === 'PageUp') {
    event.preventDefault();
    adjustCellScale(0.1);
    return;
  }
  
  // - or PageDown or Numpad- - Decrease Cell Scale
  if (event.key === 'PageDown') {
    event.preventDefault();
    adjustCellScale(-0.1);
    return;
  }
});

// ---------- bootstrap (C++ kernel default) ----------
(async () => {
    try {
        // 🔴 首先确保C++模块已加载
        await ensureCppLoaded();
        
        kernelType = "cpp";  // 🔴 确保使用C++内核
        if (btnSwitchKernel) btnSwitchKernel.textContent = "Switch to JS Kernel";
        
        setDifficulty(16, 30, 99);
        await sleep(0);
        await createNewGame();
    } catch (e) {
        derr(e);
        // 🔴 如果C++加载失败，回退到JS
        try {
            await loadPy();
            kernelType = "js";
            if (btnSwitchKernel) btnSwitchKernel.textContent = "Switch to C++ Kernel";
            
            setDifficulty(16, 30, 99);
            await sleep(0);
            await createNewGame();
        } catch (jsError) {
            derr("Failed to load both C++ and JS kernels:", jsError);
            setStatus("Failed to load kernels: " + (jsError?.stack || String(jsError)));
        }
    }
})();
