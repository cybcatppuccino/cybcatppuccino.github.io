let pyodide = null, api = null;
let H = 25, W = 40, M = 200;
let solvingTimer = null;
let viewScale = 1.0, cellScale = 1.0;
let currentGameSeed = null;

// 新增：人工模式开关
let manualModeEnabled = false;
// 新增：撤回状态存储
let undoState = null;

const el = (id) => document.getElementById(id);
const statusEl = el("status"), boardEl = el("board");
const infoEl = document.querySelector(".info-panel");
const controlPanel = document.querySelector(".controls-container"), togglePanelBtn = el("togglePanel");
const viewScaleValueEl = el("viewScaleValue"), cellScaleValueEl = el("cellScaleValue");


const inpH = el("inpH"), inpW = el("inpW"), inpM = el("inpM"), inpSeed = el("inpSeed");
const btnNewGame = el("btnNewGame"), btnStepSolve = el("btnStepSolve"),
      btnViewScaleUp = el("btnViewScaleUp"), btnViewScaleDown = el("btnViewScaleDown");
const firstZeroCheckbox = el("firstzero");
const btnEasy = el("btnEasy"), btnNormal = el("btnNormal"), btnHard = el("btnHard"), btnTranspose = el("btnTranspose");
const btnCellScaleUp = el("btnCellScaleUp"), btnCellScaleDown = el("btnCellScaleDown");
const btnUndo = el("btnUndo"); // 新增：撤回按钮

const btnHMinus5 = el("btnHMinus5"), btnHMinus1 = el("btnHMinus1"), btnHPlus1 = el("btnHPlus1"), btnHPlus5 = el("btnHPlus5");
const btnWMinus5 = el("btnWMinus5"), btnWMinus1 = el("btnWMinus1"), btnWPlus1 = el("btnWPlus1"), btnWPlus5 = el("btnWPlus5");
const btnMMinus100 = el("btnMMinus100"), btnMMinus10 = el("btnMMinus10"), btnMMinus1 = el("btnMMinus1");
const btnMPlus1 = el("btnMPlus1"), btnMPlus10 = el("btnMPlus10"), btnMPlus100 = el("btnMPlus100");

const key = (r,c)=> `${r},${c}`;
const jsRevealed = new Set();

const getProbColor = (p) => {
    const clampedP = Math.max(0, Math.min(1, p));
    const saturation = 90; // 饱和度，避免过于鲜艳
    const lightness = 40*(1-clampedP)+40;  // 亮度，使其看起来是"淡"色
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
  let revealedCount = 0, flaggedMines = 0;
  if (st) { revealedCount = st.revealed_count || 0; flaggedMines = st.ai_mines ? st.ai_mines.length : 0; }

  const density = H > 0 && W > 0 ? Math.round((M / (H * W)) * 100) : 0;
  const displaySeed = (st && st.seed !== undefined && st.seed !== null) ? st.seed : "None";

  infoEl.innerHTML = `
    <div class="info-item"><span class="info-label">Size</span><span class="info-value">${H}×${W}</span></div>
    <div class="info-item"><span class="info-label">Density</span><span class="info-value">${density}%</span></div>
    <div class="info-item"><span class="info-label">Revealed</span><span class="info-value">${revealedCount}/${H * W - M}</span></div>
    <div class="info-item"><span class="info-label">Mines</span><span class="info-value">${flaggedMines}/${M}</span></div>
    <div class="info-item seed-info"><span class="info-label">Seed</span><span class="info-value">${displaySeed}</span></div>
  `;
}

boardEl.addEventListener("click", handleManualClick);
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

// 新增：处理人工点击事件
async function handleManualClick(event) {
    if (!manualModeEnabled) return;

    const target = event.target;
    if (!target.classList.contains("cell") || 
        target.classList.contains("open") ||
        target.classList.contains("flag") ||
        target.classList.contains("mine")) return;

    const r = parseInt(target.dataset.r);
    const c = parseInt(target.dataset.c);

    if (isNaN(r) || isNaN(c)) return;

    // 保存当前状态用于撤回（在执行点击前）
    const currentStateP = api.getState();
    const currentState = currentStateP.toJs();
    currentStateP.destroy?.();
    undoState = JSON.parse(JSON.stringify(currentState)); // 深拷贝当前状态

    // 执行一次人工点击
    const resultP = api.stepAt(r, c);
    const result = resultP.toJs();
    resultP.destroy?.();
    applyStepDelta(result);

    if (result.lost || result.won || result.stuck) {
        return;
    }

    // 清理 safe moves（包括超低概率安全格）
    let rP, rData;
    do {
        rP = api.makeSafeMove();
        rData = rP.toJs();
        rP.destroy?.();
        applyStepDelta(rData); // 不再跳过更新信息
        if (rData.lost || rData.won) {
            return;
        }
        if (rData.move) await new Promise(res => setTimeout(res, 10));
    } while (rData.move);

    // 显示分析overlay
    const aP = api.getAnalysis();
    const analysis = aP.toJs();
    aP.destroy?.();
    applyAnalysisOverlay(analysis);
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
  if (st.seed !== undefined) {
    currentGameSeed = st.seed;  // 保存种子
  }
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
        const cellElement = jsCells[r * W + c];
        clearAnalysisEffects(cellElement);
    }
    for (const [r,c] of delta.ai_mines) {
        setCellFlag(r,c);
        const cellElement = jsCells[r * W + c];
        clearAnalysisEffects(cellElement);
    }

    if (delta.lost) { 
        setStatus("GAME OVER"); 
        btnUndo.style.display = undoState ? "block" : "none";
        updateGameInfo({
            revealed_count: delta.revealed_count,
            ai_mines: delta.ai_mines,
            seed: currentGameSeed !== null ? currentGameSeed : "None"
        });
    }
    else if (delta.won) { 
        setStatus("YOU WIN"); 
        btnUndo.style.display = "none";
        updateGameInfo({
            revealed_count: delta.revealed_count,
            ai_mines: delta.ai_mines,
            seed: currentGameSeed !== null ? currentGameSeed : "None"
        });
    }
    else if (delta.stuck) { 
        setStatus("STUCK (no moves)"); 
        btnUndo.style.display = "none";
        updateGameInfo({
            revealed_count: delta.revealed_count,
            ai_mines: delta.ai_mines,
            seed: currentGameSeed !== null ? currentGameSeed : "None"
        });
    }
    else {
        setStatus(`Running | Revealed: ${delta.revealed_count}`);
        btnUndo.style.display = "none";
        updateGameInfo({
            revealed_count: delta.revealed_count,
            ai_mines: delta.ai_mines,
            seed: currentGameSeed !== null ? currentGameSeed : "None"
        });
    }
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
    if (pyodide && api) return;

    const candidates = [
      "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
      "https://pyodide.org/pyodide/v0.26.4/full/",
      "https://unpkg.com/pyodide@0.26.4/pyodide/full/",
    ];

    let lastErr = null;
    for (const indexURL of candidates) {
      try {
        setStatus(`Loading Pyodide... (${indexURL})`);
        pyodide = await loadPyodide({ indexURL });
        lastErr = null;
        break;
      } catch (e) {
        console.warn("loadPyodide failed with", indexURL, e);
        lastErr = e;
      }
    }
    if (lastErr) throw lastErr;

    setStatus("Loading python code...");
    const resp = await fetch("./py/minesweeper.py", { cache: "no-cache" });
    if (!resp.ok) throw new Error(`fetch minesweeper.py failed: ${resp.status} ${resp.statusText}`);
    const code = await resp.text();

    setStatus("Initializing python runtime...");
    await pyodide.runPythonAsync(code);

    api = {
      newGame: pyodide.globals.get("ms_new_game"),
      step: pyodide.globals.get("ms_step"),
      stepAt: pyodide.globals.get("ms_step_at"),
      getState: pyodide.globals.get("ms_get_state"),
      makeSafeMove: pyodide.globals.get("ms_make_safe_move"),
      setState: pyodide.globals.get("ms_set_state"),
      getAnalysis: pyodide.globals.get("ms_get_analysis"),
    };

    setStatus("Ready.");
    setDifficulty(16, 30, 99);
    await new Promise(requestAnimationFrame);
    await createNewGame();
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

async function createNewGame() {
  stopSolving(); // 确保停止任何正在进行的自动求解
  manualModeEnabled = true; // 开启人工模式
  // 清空撤回状态
  undoState = null;
  // 隐藏撤回按钮
  btnUndo.style.display = "none";
  
  const h = clampInt(inpH.value, 5, 200, 25);
  const w = clampInt(inpW.value, 5, 200, 40);
  const m = clampInt(inpM.value, 1, h*w - 1, Math.min(200, h*w-1));
  const seedInputValue = inpSeed.value;
  let seed = null;

  if (seedInputValue !== null && seedInputValue !== undefined && seedInputValue.trim() !== "") {
      const trimmedValue = seedInputValue.trim();
      // 使用 null 作为 fallback，这样无效输入也会返回 null
      seed = clampInt(trimmedValue, -2147483648, 2147483647, null);
  }

  // 读取复选框状态：勾选为 true -> firstmv=1, 未勾选为 false -> firstmv=2
  const isFirstMoveZero = firstZeroCheckbox.checked;
  const firstmv = isFirstMoveZero ? 1 : 2;
  
  inpH.value = String(h); inpW.value = String(w); inpM.value = String(m);
  
  // 调用 Python 的 ms_new_game 函数创建新游戏
  const stProxy = api.newGame(h, w, m, seed, firstmv);
  
  const st = stProxy.toJs(); 
  stProxy.destroy?.(); 
  
  // 确保在这里设置 currentGameSeed
  if (st.seed !== undefined) {
    currentGameSeed = st.seed;
  }
  
  applyFullState(st);
}

// 全局状态标志
let isStepSolving = false;

async function stepSolve() {
    if (!api || isStepSolving || !manualModeEnabled) return;
    isStepSolving = true;
    manualModeEnabled = false;
    try {
        let rP, r;
        // 1. 清空所有当前的安全移动
        do {
            rP = api.makeSafeMove(); 
            r = rP.toJs(); 
            rP.destroy?.();
            applyStepDelta(r); // 不再跳过更新信息
            if (r.lost || r.won) { 
                // 获取完整状态来保持seed等信息
                const fullStateP = api.getState();
                const fullState = fullStateP.toJs();
                fullStateP.destroy?.();
                updateGameInfo(fullState);
                return; 
            }
            if (r.move) await new Promise(res => setTimeout(res, 10));
        } while (r.move);
        
        // 2. 执行一次随机移动前保存状态用于撤回
        const currentStateP = api.getState();
        const currentState = currentStateP.toJs();
        currentStateP.destroy?.();
        undoState = JSON.parse(JSON.stringify(currentState)); // 深拷贝当前状态
        
        // 执行一次随机移动
        const sP = api.step(); 
        const s = sP.toJs(); 
        sP.destroy?.();
        applyStepDelta(s); // 不再跳过更新信息
        if (s.lost || s.won || s.stuck) { 
            // 获取完整状态来保持seed等信息
            const fullStateP = api.getState();
            const fullState = fullStateP.toJs();
            fullStateP.destroy?.();
            updateGameInfo(fullState);
            return; 
        }
        
        // 3. 清空这次随机移动带来的新安全移动
        do {
            rP = api.makeSafeMove(); 
            r = rP.toJs(); 
            rP.destroy?.();
            applyStepDelta(r); // 不再跳过更新信息
            if (r.lost || r.won) { 
                // 获取完整状态来保持seed等信息
                const fullStateP = api.getState();
                const fullState = fullStateP.toJs();
                fullStateP.destroy?.();
                updateGameInfo(fullState);
                return; 
            }
            if (r.move) await new Promise(res => setTimeout(res, 10));
        } while (r.move);
        
        // 4. 显示分析和更新完整信息
        // 如果 currentGameSeed 为空，先获取完整状态
        if (currentGameSeed === null) {
            const fullStateP = api.getState();
            const fullState = fullStateP.toJs();
            fullStateP.destroy?.();
            if (fullState.seed !== undefined) {
                currentGameSeed = fullState.seed;
            }
        }
        
        // 获取完整状态来保持seed等信息
        const fullStateP = api.getState();
        const fullState = fullStateP.toJs();
        fullStateP.destroy?.();
        updateGameInfo(fullState);
        
        // 显示分析overlay
        const aP = api.getAnalysis(); 
        const a = aP.toJs(); 
        aP.destroy?.();
        applyAnalysisOverlay(a);
    } finally { 
        isStepSolving = false; 
        manualModeEnabled = true;
    }
}

function stopSolving() {
    if (solvingTimer) { 
        clearTimeout(solvingTimer); 
        solvingTimer = null; 
    }
    manualModeEnabled = true; // 恢复人工模式
}

async function undoLastMove() {
    if (!undoState) {
        console.warn("No state to undo");
        return;
    }

    try {
        // 1) 使用Python的setState函数完全恢复状态
        const restoredP = api.setState(undoState);
        const restored = restoredP.toJs();
        restoredP.destroy?.();

        // 2) 应用恢复的状态到前端
        applyFullState(restored);

        // 3) 隐藏撤回按钮
        btnUndo.style.display = "none";
        
        // 4) 更新状态
        setStatus("Ready");
        undoState = null;
        manualModeEnabled = true;

        // 5) 重新获取并显示分析overlay（这是关键步骤）
        // 添加一个小延迟确保状态完全同步
        setTimeout(() => {
            const aP = api.getAnalysis();
            const analysis = aP.toJs();
            aP.destroy?.();
            applyAnalysisOverlay(analysis);
            
            // 6) 确保信息面板更新
            const fullStateP = api.getState();
            const fullState = fullStateP.toJs();
            fullStateP.destroy?.();
            updateGameInfo(fullState);
        }, 10);
    } catch (error) {
        console.error("Error during undo:", error);
        setStatus("Undo failed: " + (error.message || String(error)));
    }
}


btnNewGame.addEventListener("click", createNewGame);
btnStepSolve.addEventListener("click", stepSolve);
btnUndo.addEventListener("click", undoLastMove); // 新增：撤回按钮事件监听器

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
