let pyodide = null, api = null;
let H = 25, W = 40, M = 200;
let solvingTimer = null, startTime = null, gameStarted = false;
let viewScale = 1.0, cellScale = 1.0;

const el = (id) => document.getElementById(id);
const statusEl = el("status"), boardEl = el("board");
const infoEl = document.querySelector(".info-panel");
const controlPanel = document.querySelector(".controls-container"), togglePanelBtn = el("togglePanel");
const viewScaleValueEl = el("viewScaleValue"), cellScaleValueEl = el("cellScaleValue");

const inpH = el("inpH"), inpW = el("inpW"), inpM = el("inpM"), inpSeed = el("inpSeed"), inpSpeed = el("inpSpeed");
const btnNew = el("btnNew"), btnViewScaleUp = el("btnViewScaleUp"), btnViewScaleDown = el("btnViewScaleDown");
const firstZeroCheckbox = el("firstzero");
const btnEasy = el("btnEasy"), btnNormal = el("btnNormal"), btnHard = el("btnHard"), btnTranspose = el("btnTranspose");
const btnCellScaleUp = el("btnCellScaleUp"), btnCellScaleDown = el("btnCellScaleDown");

const btnHMinus5 = el("btnHMinus5"), btnHMinus1 = el("btnHMinus1"), btnHPlus1 = el("btnHPlus1"), btnHPlus5 = el("btnHPlus5");
const btnWMinus5 = el("btnWMinus5"), btnWMinus1 = el("btnWMinus1"), btnWPlus1 = el("btnWPlus1"), btnWPlus5 = el("btnWPlus5");
const btnMMinus100 = el("btnMMinus100"), btnMMinus10 = el("btnMMinus10"), btnMMinus1 = el("btnMMinus1");
const btnMPlus1 = el("btnMPlus1"), btnMPlus10 = el("btnMPlus10"), btnMPlus100 = el("btnMPlus100");

const key = (r,c)=> `${r},${c}`;
const jsRevealed = new Set();
let jsCells = [];
const FLAG_SVG = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect x="7" y="2" width="1.5" height="12" fill="#000"/><polygon points="8,2 14,5 8,8" fill="#f00"/></svg>`;

function logLine(s) { infoEl.textContent = s; infoEl.scrollTop = infoEl.scrollHeight; }
function setStatus(s) { statusEl.textContent = s; }

function togglePanel() {
  const isCollapsed = controlPanel.classList.contains("collapsed");
  controlPanel.classList.toggle("collapsed", !isCollapsed);
  togglePanelBtn.textContent = isCollapsed ? "▼" : "▲";
}

function adjustViewScale(delta) {
  viewScale = Math.max(0.2, Math.min(3.0, viewScale + delta));
  document.querySelector('.boardWrap').style.transform = `scale(${viewScale})`;
  document.querySelector('.boardWrap').style.transformOrigin = 'top left';
  viewScaleValueEl.textContent = Math.round(viewScale * 100) + "%";
}

function adjustCellScale(delta) {
  cellScale = Math.max(0.2, Math.min(3.0, cellScale + delta));
  document.documentElement.style.setProperty("--board-cell-scale", cellScale.toFixed(2));
  cellScaleValueEl.textContent = Math.round(cellScale * 100) + "%";
}

function updateGameInfo(st) {
  const elapsed = gameStarted ? Math.floor((Date.now() - startTime) / 1000) : 0;
  const minutes = Math.floor(elapsed / 60), seconds = elapsed % 60;
  const timeStr = `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
  let revealedCount = 0, flaggedMines = 0;
  if (st) { revealedCount = st.revealed_count || 0; flaggedMines = st.ai_mines ? st.ai_mines.length : 0; }
  infoEl.innerHTML = `
    <div class="info-item"><span class="info-label">Time</span><span class="info-value">${timeStr}</span></div>
    <div class="info-item"><span class="info-label">Size</span><span class="info-value">${H}×${W}</span></div>
    <div class="info-item"><span class="info-label">Revealed</span><span class="info-value">${revealedCount}/${H * W - M}</span></div>
    <div class="info-item"><span class="info-label">Mines</span><span class="info-value">${flaggedMines}/${M}</span></div>
  `;
}

function buildBoardDOM(h, w) {
  if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0 || h*w > 40000) 
    throw new Error(`Invalid board size: h=${h}, w=${w}`);
  jsRevealed.clear(); jsCells = new Array(h * w);
  boardEl.style.gridTemplateColumns = `repeat(${w}, calc(var(--cell) * var(--board-cell-scale)))`;
  boardEl.innerHTML = "";
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) {
    const d = document.createElement("div");
    d.className = "cell"; d.dataset.r = r; d.dataset.c = c;
    boardEl.appendChild(d); jsCells[r * w + c] = d;
  }
}

function setCellCovered(r,c) { const d = jsCells[r*W + c]; d.className = "cell"; d.textContent = ""; delete d.dataset.number; }
function setCellOpen(r,c,n) { const d = jsCells[r*W + c]; d.className = "cell open"; if (n > 0) { d.textContent = String(n); d.dataset.number = String(n); } else { d.textContent = ""; delete d.dataset.number; } }
function setCellMine(r,c) { const d = jsCells[r*W + c]; d.className = "cell mine"; d.textContent = "X"; delete d.dataset.number; }
function setCellFlag(r,c) { const d = jsCells[r*W + c]; if (!d.classList.contains("open") && !d.classList.contains("mine")) { d.className = "cell flag"; d.innerHTML = FLAG_SVG; delete d.dataset.number; } }

function applyFullState(st) {
  H = st.h; W = st.w; M = st.mines; buildBoardDOM(H, W);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) setCellCovered(r,c);
  for (const [r,c,n] of st.revealed) { jsRevealed.add(key(r,c)); if (n === -1) setCellMine(r,c); else setCellOpen(r,c,n); }
  for (const [r,c] of st.ai_mines) setCellFlag(r,c);
  setStatus(`Ready | Revealed: ${st.revealed_count} | Lost: ${st.lost} | Won: ${st.won}`);
  updateGameInfo(st);
}

function applyStepDelta(delta) {
  for (const [r,c,n] of delta.newly) { jsRevealed.add(key(r,c)); if (n === -1) setCellMine(r,c); else setCellOpen(r,c,n); }
  for (const [r,c] of delta.ai_mines) setCellFlag(r,c);
  if (delta.lost) { setStatus("GAME OVER"); gameStarted = false; }
  else if (delta.won) { setStatus("YOU WIN"); gameStarted = false; }
  else if (delta.stuck) { setStatus("STUCK (no moves)"); gameStarted = false; }
  else setStatus(`Running | Revealed: ${delta.revealed_count}`);
  updateGameInfo({ revealed_count: delta.revealed_count, ai_mines: delta.ai_mines });
}

async function loadPy() {
  try {
    setStatus("Loading Pyodide...");
    pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
    setStatus("Loading python code...");
    const code = await fetch("./py/minesweeper.py").then(r => r.text());
    await pyodide.runPythonAsync(code);
    api = { newGame: pyodide.globals.get("ms_new_game"), step: pyodide.globals.get("ms_step"), getState: pyodide.globals.get("ms_get_state") };
    setStatus("Ready."); await newGameFromUI();
  } catch (e) { console.error(e); setStatus("Failed to load: " + (e?.stack || String(e))); }
}

function clampInt(x, lo, hi, fallback) {
  const n = Number.parseInt(x, 10);
  return !Number.isFinite(n) ? fallback : Math.max(lo, Math.min(hi, n));
}

function adjustParam(inputId, delta, minVal, maxVal) {
  const input = el(inputId);
  let val = parseInt(input.value) || 0;
  val += delta; val = Math.max(minVal, Math.min(maxVal, val));
  input.value = val;
}

function setDifficulty(h, w, m) {
  inpH.value = h;
  inpW.value = w;
  inpM.value = m;
}

function transposeBoard() {
  const currentH = parseInt(inpH.value) || 25;
  const currentW = parseInt(inpW.value) || 40;
  const currentM = parseInt(inpM.value) || 200;
  
  // 交换 H 和 W，保持雷数不变
  inpH.value = currentW;
  inpW.value = currentH;
  // 雷数保持不变
  inpM.value = currentM;
}

async function newGameFromUI() {
  stopSolving();
  const h = clampInt(inpH.value, 5, 200, 25);
  const w = clampInt(inpW.value, 5, 200, 40);
  const m = clampInt(inpM.value, 1, h*w - 1, Math.min(200, h*w-1));
  const seedStr = (inpSeed.value || "").trim();
  const seed = seedStr === "" ? null : clampInt(seedStr, -2147483648, 2147483647, 0);
  
  // 读取复选框状态：勾选为 true -> firstmv=1, 未勾选为 false -> firstmv=2
  const isFirstMoveZero = firstZeroCheckbox.checked;
  const firstmv = isFirstMoveZero ? 1 : 2;
  
  inpH.value = String(h); inpW.value = String(w); inpM.value = String(m);
  startTime = Date.now(); gameStarted = true; updateGameInfo(null);
  
  // 将 firstmv 作为额外参数传递给 Python 的 ms_new_game 函数
  const stProxy = api.newGame(h, w, m, seed, firstmv);
  
  const st = stProxy.toJs(); stProxy.destroy?.(); applyFullState(st);
  startSolving(); // 自动开始求解
}


async function stepOnce() {
  if (!api) return;
  const dProxy = api.step(); const d = dProxy.toJs(); dProxy.destroy?.();
  applyStepDelta(d);
  if (d.lost || d.won || d.stuck) stopSolving();
}

function startSolving() {
  stopSolving(); const tick = async () => {
    const speed = parseInt(inpSpeed.value, 10);
    await stepOnce();
    if (!solvingTimer) return;
    solvingTimer = setTimeout(tick, speed);
  };
  solvingTimer = setTimeout(tick, 0);
}

function stopSolving() {
  if (solvingTimer) { clearTimeout(solvingTimer); solvingTimer = null; }
}

btnNew.addEventListener("click", newGameFromUI);
togglePanelBtn.addEventListener("click", togglePanel);
btnViewScaleUp.addEventListener("click", () => adjustViewScale(0.2));
btnViewScaleDown.addEventListener("click", () => adjustViewScale(-0.2));
btnCellScaleUp.addEventListener("click", () => adjustCellScale(0.2));
btnCellScaleDown.addEventListener("click", () => adjustCellScale(-0.2));

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

// 难度按钮事件监听器
btnEasy.addEventListener("click", () => {
  setDifficulty(9, 9, 10);
  newGameFromUI();
});

btnNormal.addEventListener("click", () => {
  setDifficulty(16, 16, 40);
  newGameFromUI();
});

btnHard.addEventListener("click", () => {
  setDifficulty(16, 30, 99);
  newGameFromUI();
});

btnTranspose.addEventListener("click", () => {
  transposeBoard();
  newGameFromUI();
});


viewScaleValueEl.textContent = "100%";
cellScaleValueEl.textContent = "100%";

// Initialize with expanded panel
togglePanelBtn.textContent = "▲";

loadPy().catch(err => { console.error(err); setStatus("Failed to load: " + String(err)); });
