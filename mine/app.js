let pyodide = null, api = null;
let H = 25, W = 40, M = 200;
let solvingTimer = null, startTime = null, gameStarted = false;
let viewScale = 1.0, cellScale = 1.0;

const el = (id) => document.getElementById(id);
const statusEl = el("status"), boardEl = el("board");
const infoEl = document.querySelector(".info-panel");
const controlPanel = document.querySelector(".controls-container"), togglePanelBtn = el("togglePanel");
const viewScaleValueEl = el("viewScaleValue"), cellScaleValueEl = el("viewScaleValue");

const inpH = el("inpH"), inpW = el("inpW"), inpM = el("inpM"), inpSeed = el("inpSeed"), inpSpeed = el("inpSpeed");
const btnNewGame = el("btnNewGame"), btnStepSolve = el("btnStepSolve"), btnAllSolve = el("btnAllSolve"),
      btnViewScaleUp = el("btnViewScaleUp"), btnViewScaleDown = el("btnViewScaleDown");
const firstZeroCheckbox = el("firstzero");
const btnEasy = el("btnEasy"), btnNormal = el("btnNormal"), btnHard = el("btnHard"), btnTranspose = el("btnTranspose");
const btnCellScaleUp = el("btnCellScaleUp"), btnCellScaleDown = el("btnCellScaleDown");

const btnHMinus5 = el("btnHMinus5"), btnHMinus1 = el("btnHMinus1"), btnHPlus1 = el("btnHPlus1"), btnHPlus5 = el("btnHPlus5");
const btnWMinus5 = el("btnWMinus5"), btnWMinus1 = el("btnWMinus1"), btnWPlus1 = el("btnWPlus1"), btnWPlus5 = el("btnWPlus5");
const btnMMinus100 = el("btnMMinus100"), btnMMinus10 = el("btnMMinus10"), btnMMinus1 = el("btnMMinus1");
const btnMPlus1 = el("btnMPlus1"), btnMPlus10 = el("btnMPlus10"), btnMPlus100 = el("btnMPlus100");

const key = (r,c)=> `${r},${c}`;
const jsRevealed = new Set();

const getProbColor = (p) => {
    const clampedP = Math.max(0, Math.min(1, p));
    const saturation = 90; // 饱和度，避免过于鲜艳
    const lightness = 40*(1-clampedP)+40;  // 亮度，使其看起来是“淡”色
    const hue = 120 * (1- clampedP)**1.5;
    const alpha = 0.4+0.3*(1- clampedP);
    
    return `hsl(${hue}, ${saturation}%, ${lightness}%, ${alpha})`;
};

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

function clearAnalysisEffects(cellElement) {
    if (cellElement) {
        cellElement.classList.remove('analyzed', 'next-move');
        cellElement.style.backgroundColor = '';
        cellElement.style.color = '';
        cellElement.style.fontWeight = '';
        cellElement.style.fontSize = '';
        cellElement.style.border = '';
        cellElement.style.boxShadow = '';
        cellElement.style.animation = '';
    }
}

function setCellCovered(r,c) { 
    const d = jsCells[r*W + c]; 
    d.className = "cell"; 
    d.textContent = ""; 
    delete d.dataset.number; 
    clearAnalysisEffects(d);
}

function setCellOpen(r,c,n) { 
    const d = jsCells[r*W + c]; 
    d.className = "cell open"; 
    if (n > 0) { 
        d.textContent = String(n); 
        d.dataset.number = String(n); 
    } else { 
        d.textContent = ""; 
        delete d.dataset.number; 
    }
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
    if (!d.classList.contains("open") && !d.classList.contains("mine")) { 
        d.className = "cell flag"; 
        d.innerHTML = FLAG_SVG; 
        delete d.dataset.number; 
        clearAnalysisEffects(d);
    }
}

function applyFullState(st) {
  H = st.h; W = st.w; M = st.mines; buildBoardDOM(H, W);
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) setCellCovered(r,c);
  for (const [r,c,n] of st.revealed) { jsRevealed.add(key(r,c)); if (n === -1) setCellMine(r,c); else setCellOpen(r,c,n); }
  for (const [r,c] of st.ai_mines) setCellFlag(r,c);
  setStatus(`Ready | Revealed: ${st.revealed_count} | Lost: ${st.lost} | Won: ${st.won}`);
  updateGameInfo(st);
}

function applyStepDelta(delta) {
    for (const [r,c,n] of delta.newly) {
        jsRevealed.add(key(r,c));
        if (n === -1) setCellMine(r,c);
        else setCellOpen(r,c,n);
        
        // 清除该格子可能存在的概率分析样式
        const cellElement = jsCells[r * W + c];
        clearAnalysisEffects(cellElement);
    }
    for (const [r,c] of delta.ai_mines) {
        setCellFlag(r,c);
        
        // 清除该格子可能存在的概率分析样式
        const cellElement = jsCells[r * W + c];
        clearAnalysisEffects(cellElement);
    }
    if (delta.lost) { setStatus("GAME OVER"); gameStarted = false; }
    else if (delta.won) { setStatus("YOU WIN"); gameStarted = false; }
    else if (delta.stuck) { setStatus("STUCK (no moves)"); gameStarted = false; }
    else setStatus(`Running | Revealed: ${delta.revealed_count}`);
    updateGameInfo({ revealed_count: delta.revealed_count, ai_mines: delta.ai_mines });
}

function applyAnalysisOverlay(d) {
    // --- 1. 精准清理：只清理未翻开的、之前被分析过的格子 ---
    const analyzedCells = document.querySelectorAll('.cell.analyzed');
    for(const cell of analyzedCells) {
        // 确保不 touch 已经翻开的格子
        if (!cell.classList.contains('open') && 
            !cell.classList.contains('mine') && 
            !cell.classList.contains('flag')) {
            
            cell.classList.remove('analyzed', 'next-move');
            // 只清除分析相关的动态样式和内容
            cell.textContent = '';
            cell.style.backgroundColor = '';
            cell.style.color = '';
            cell.style.fontWeight = '';
            cell.style.fontSize = '';
            cell.style.border = '';
            cell.style.boxShadow = '';
            cell.style.animation = '';
        }
    }

    // --- 2. 应用新的概率分析 ---
    for (const [keyStr, probability] of Object.entries(d.probs)) {
        const match = keyStr.match(/\((\d+),\s*(\d+)\)/);
        if (!match) continue;
        const r = parseInt(match[1], 10);
        const c = parseInt(match[2], 10);

        if (r < 0 || r >= H || c < 0 || c >= W) continue;

        const cellElement = jsCells[r * W + c];
        
        // 关键：只对未翻开的格子应用分析样式
        if (cellElement && 
            !cellElement.classList.contains('open') && 
            !cellElement.classList.contains('mine') && 
            !cellElement.classList.contains('flag')) {
            
            cellElement.classList.add('analyzed'); // 标记为已分析
            
            const p = parseFloat(probability);
            if (isNaN(p)) continue;

            // 应用概率背景色（使用优化后的颜色函数）
            cellElement.style.backgroundColor = getProbColor(p);
            
            // 应用概率文本
            cellElement.textContent = Math.round(p * 100).toString().padStart(2, '0');
            cellElement.style.color = '#000';
            cellElement.style.fontWeight = 'normal';
            cellElement.style.fontSize = 'calc(var(--cell) * var(--board-cell-scale) * 0.6)';
        }
    }

    // --- 3. 高亮下一步要点击的格子 ---
    if (d.next_move && Array.isArray(d.next_move) && d.next_move.length === 2) {
        const [nr, nc] = d.next_move;

        if (nr >= 0 && nr < H && nc >= 0 && nc < W) {
            const nextCellElement = jsCells[nr * W + nc];
            
            // 再次确认是未翻开的格子
            if (nextCellElement && 
                !nextCellElement.classList.contains('open') && 
                !nextCellElement.classList.contains('mine') && 
                !nextCellElement.classList.contains('flag')) {
                
                // 确保它被标记为已分析（它应该已经在 d.probs 里了）
                nextCellElement.classList.add('analyzed');
                // 应用更明显的高亮样式
                nextCellElement.classList.add('next-move'); // 添加类名
                
                // 更明显的视觉效果
                nextCellElement.style.backgroundColor = '#00FF00'; // 鲜艳绿色
                nextCellElement.style.color = '#FFFFFF'; // 白色文字
                nextCellElement.style.fontWeight = 'bold'; // 粗体
                nextCellElement.style.border = '2px solid #FF0000'; // 红色边框
                nextCellElement.style.boxShadow = '0 0 10px #00FF00'; // 绿色发光效果
            }
        }
    }
}

async function loadPy() {
  try {
    setStatus("Loading Pyodide...");
    pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });
    setStatus("Loading python code...");
    const code = await fetch("./py/minesweeper.py").then(r => r.text());
    await pyodide.runPythonAsync(code);
    // --- 在 loadPy 函数内修改 api 初始化 ---
    api = {
        newGame: pyodide.globals.get("ms_new_game"),
        step: pyodide.globals.get("ms_step"),
        getState: pyodide.globals.get("ms_get_state"),
        makeSafeMove: pyodide.globals.get("ms_make_safe_move"),
        getAnalysis: pyodide.globals.get("ms_get_analysis") // <-- 新增这一行
    };
    // --- 修改结束 ---

    setStatus("Ready."); 
    await createNewGame(); // 页面加载完成后调用新的函数名
  } catch (e) { 
    console.error(e); 
    setStatus("Failed to load: " + (e?.stack || String(e))); 
  }
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

// --- 修改 1: 重命名并修改 newGameFromUI 为 createNewGame ---
async function createNewGame() {
  stopSolving(); // 确保停止任何正在进行的自动求解
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
  
  // 调用 Python 的 ms_new_game 函数创建新游戏
  const stProxy = api.newGame(h, w, m, seed, firstmv);
  
  const st = stProxy.toJs(); stProxy.destroy?.(); applyFullState(st);
  // 注意：这里不再调用 startSolving()
  // setStatus("New game ready. Click 'Step Solve' or 'All Solve' to begin.");
}

// 全局状态标志
let isStepSolving = false;
let isAllSolving = false;

// --- 修复后的 stepSolve ---
async function stepSolve() {
    if (!api || isStepSolving || isAllSolving || !gameStarted) return;
    isStepSolving = true;
    try {
        let rP, r;
        // 1. 清空所有当前的安全移动
        do {
            rP = api.makeSafeMove(); 
            r = rP.toJs(); 
            rP.destroy?.();
            applyStepDelta(r);
            if (r.lost || r.won) { 
                gameStarted = false; 
                return; 
            }
            if (r.move) await new Promise(res => setTimeout(res, 10));
        } while (r.move);
        
        // 2. 执行一次随机移动（仅当还有未知格子时）
        const sP = api.step(); 
        const s = sP.toJs(); 
        sP.destroy?.();
        applyStepDelta(s);
        if (s.lost || s.won || s.stuck) { 
            gameStarted = false; 
            return; 
        }
        
        // 3. 清空这次随机移动带来的新安全移动
        do {
            rP = api.makeSafeMove(); 
            r = rP.toJs(); 
            rP.destroy?.();
            applyStepDelta(r);
            if (r.lost || r.won) { 
                gameStarted = false; 
                return; 
            }
            if (r.move) await new Promise(res => setTimeout(res, 10));
        } while (r.move);
        
        // 4. 显示分析
        if (gameStarted) {
            const aP = api.getAnalysis(); 
            const a = aP.toJs(); 
            aP.destroy?.();
            applyAnalysisOverlay(a);
        }
    } finally { 
        isStepSolving = false; 
    }
}

// --- 修复后的 stepOnce（用于 All Solve）---
async function stepOnce() {
    if (!api || isStepSolving) return; // 增加互斥检查
    
    // 先尝试执行所有安全移动
    let rP, r;
    do {
        rP = api.makeSafeMove(); 
        r = rP.toJs(); 
        rP.destroy?.();
        applyStepDelta(r);
        if (r.lost || r.won) { 
            stopSolving();
            return; 
        }
        if (r.move) await new Promise(res => setTimeout(res, 0));
    } while (r.move);
    
    // 如果没有安全移动了，执行一次随机移动
    const dProxy = api.step(); 
    const d = dProxy.toJs(); 
    dProxy.destroy?.();
    applyStepDelta(d);
    if (d.lost || d.won || d.stuck) {
        stopSolving();
    }
}

// --- 修复后的 startAllSolve ---
function startAllSolve() {
    stopSolving(); // 停止之前的定时器
    isStepSolving = false; // 确保 Step Solve 不会干扰
    isAllSolving = true; // 设置 All Solve 标志
    
    const tick = async () => {
        if (!isAllSolving) return; // 双重检查
        const speed = parseInt(inpSpeed.value, 10);
        await stepOnce();
        if (solvingTimer && isAllSolving) { // 确保状态一致
            solvingTimer = setTimeout(tick, speed);
        }
    };
    solvingTimer = setTimeout(tick, 0);
}

// --- 修复后的 stopSolving ---
function stopSolving() {
    if (solvingTimer) { 
        clearTimeout(solvingTimer); 
        solvingTimer = null; 
    }
    isAllSolving = false;
}

btnNewGame.addEventListener("click", createNewGame);     // 连接到我们重命名的 createNewGame 函数
btnStepSolve.addEventListener("click", stepSolve);       // 连接到新创建的 stepSolve 函数
btnAllSolve.addEventListener("click", startAllSolve);    // 连接到我们重命名的 startAllSolve 函数

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
  createNewGame();
});

btnNormal.addEventListener("click", () => {
  setDifficulty(16, 16, 40);
  createNewGame();
});

btnHard.addEventListener("click", () => {
  setDifficulty(16, 30, 99);
  createNewGame();
});

btnTranspose.addEventListener("click", () => {
  transposeBoard();
  createNewGame();
});

viewScaleValueEl.textContent = "100%";
cellScaleValueEl.textContent = "100%";

// Initialize with expanded panel
togglePanelBtn.textContent = "▲";

loadPy().catch(err => { console.error(err); setStatus("Failed to load: " + String(err)); });
