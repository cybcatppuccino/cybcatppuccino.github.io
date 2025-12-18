let pyodide = null;
let api = null;

let H = 35, W = 35, M = 260;
let solvingTimer = null;

const el = (id) => document.getElementById(id);
const statusEl = el("status");
const boardEl  = el("board");
const logEl    = el("log");

const inpH = el("inpH");
const inpW = el("inpW");
const inpM = el("inpM");
const inpSeed = el("inpSeed");
const inpSpeed = el("inpSpeed");

const btnNew = el("btnNew");
const btnStep = el("btnStep");
const btnSolve = el("btnSolve");
const btnStop = el("btnStop");

const key = (r,c)=> `${r},${c}`;
const jsRevealed = new Set();
let jsCells = []; // DOM nodes

function logLine(s) {
  logEl.textContent += s + "\n";
  logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(s) { statusEl.textContent = s; }

function buildBoardDOM(h, w) {
  if (!Number.isFinite(h) || !Number.isFinite(w) || h <= 0 || w <= 0 || h*w > 40000) {
    throw new Error(`Invalid board size: h=${h}, w=${w}`);
  }
  jsRevealed.clear();
  jsCells = new Array(h * w);

  // 动态调格子大小（UI不重要，这里只是别太大）
  const cellPx = Math.max(14, Math.min(26, Math.floor(700 / Math.max(h, w))));
  document.documentElement.style.setProperty("--cell", `${cellPx}px`);

  boardEl.style.gridTemplateColumns = `repeat(${w}, var(--cell))`;
  boardEl.innerHTML = "";

  for (let r = 0; r < h; r++) {
    for (let c = 0; c < w; c++) {
      const d = document.createElement("div");
      d.className = "cell";
      d.dataset.r = r;
      d.dataset.c = c;
      boardEl.appendChild(d);
      jsCells[r * w + c] = d;
    }
  }
}

function setCellCovered(r,c) {
  const d = jsCells[r*W + c];
  d.className = "cell";
  d.textContent = "";
}

function setCellOpen(r,c,n) {
  const d = jsCells[r*W + c];
  d.className = "cell open";
  d.textContent = (n > 0 ? String(n) : "");
}

function setCellMine(r,c) {
  const d = jsCells[r*W + c];
  d.className = "cell mine";
  d.textContent = "X";
}

function setCellFlag(r,c) {
  const d = jsCells[r*W + c];
  // 不覆盖 open/mine 的展示优先级；这里简单粗暴
  if (!d.classList.contains("open") && !d.classList.contains("mine")) {
    d.className = "cell flag";
    d.textContent = "F";
  }
}

function applyFullState(st) {
  H = st.h; W = st.w; M = st.mines;
  buildBoardDOM(H, W);

  // 全量清
  for (let r = 0; r < H; r++) for (let c = 0; c < W; c++) setCellCovered(r,c);

  // revealed: [ [r,c,n], ... ]
  for (const [r,c,n] of st.revealed) {
    jsRevealed.add(key(r,c));
    if (n === -1) setCellMine(r,c);
    else setCellOpen(r,c,n);
  }

  // ai_mines: [ [r,c], ... ]（只用于展示flag）
  for (const [r,c] of st.ai_mines) setCellFlag(r,c);

  setStatus(`ready | revealed=${st.revealed_count} | lost=${st.lost} | won=${st.won}`);
}

function applyStepDelta(delta) {
  // newly: [ [r,c,n], ... ]
  for (const [r,c,n] of delta.newly) {
    jsRevealed.add(key(r,c));
    if (n === -1) setCellMine(r,c);
    else setCellOpen(r,c,n);
  }

  for (const [r,c] of delta.ai_mines) setCellFlag(r,c);

  if (delta.lost) setStatus("GAME OVER");
  else if (delta.won) setStatus("YOU WIN");
  else if (delta.stuck) setStatus("STUCK (no moves)");
  else setStatus(`running | revealed=${delta.revealed_count}`);
}

async function loadPy() {
  try {
    setStatus("Loading Pyodide runtime...");
    pyodide = await loadPyodide({ indexURL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/" });

    setStatus("Loading python code...");
    const code = await fetch("./py/minesweeper.py").then(r => r.text());
    await pyodide.runPythonAsync(code);

    api = {
      newGame: pyodide.globals.get("ms_new_game"),
      step: pyodide.globals.get("ms_step"),
      getState: pyodide.globals.get("ms_get_state"),
    };

    setStatus("Ready.");
    btnStep.disabled = false;
    btnSolve.disabled = false;

    await newGameFromUI();
  } catch (e) {
    console.error(e);
    setStatus("Failed to load: " + (e?.stack || String(e)));
  }
}


function clampInt(x, lo, hi, fallback) {
  const n = Number.parseInt(x, 10);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(lo, Math.min(hi, n));
}

async function newGameFromUI() {
  stopSolving();

  const h = clampInt(inpH.value, 5, 200, 35);
  const w = clampInt(inpW.value, 5, 200, 35);
  const m = clampInt(inpM.value, 1, h*w - 1, Math.min(260, h*w-1));

  const seedStr = (inpSeed.value || "").trim();
  const seed = seedStr === "" ? null : clampInt(seedStr, -2147483648, 2147483647, 0);

  inpH.value = String(h);
  inpW.value = String(w);
  inpM.value = String(m);

  logEl.textContent = "";
  logLine(`New game: ${h}x${w}, mines=${m}, seed=${seedStr || "(none)"}`);

const stProxy = api.newGame(h, w, m, seed);
const st = stProxy.toJs();   // <-- 不要传 dict_converter
stProxy.destroy?.();
console.log("state from python:", st); // 临时调试
applyFullState(st);

}


async function stepOnce() {
  if (!api) return;

  const dProxy = api.step();
  const d = dProxy.toJs();

  dProxy.destroy?.();

  applyStepDelta(d);

  if (d.lost || d.won || d.stuck) stopSolving();
}

function startSolving() {
  stopSolving();
  btnStop.disabled = false;

  const tick = async () => {
    const speed = parseInt(inpSpeed.value, 10);
    await stepOnce();
    if (!solvingTimer) return;
    solvingTimer = setTimeout(tick, speed);
  };
  solvingTimer = setTimeout(tick, 0);
}

function stopSolving() {
  if (solvingTimer) {
    clearTimeout(solvingTimer);
    solvingTimer = null;
  }
  btnStop.disabled = true;
}

btnNew.addEventListener("click", newGameFromUI);
btnStep.addEventListener("click", stepOnce);
btnSolve.addEventListener("click", startSolving);
btnStop.addEventListener("click", stopSolving);

loadPy().catch(err => {
  console.error(err);
  setStatus("Failed to load: " + String(err));
});
