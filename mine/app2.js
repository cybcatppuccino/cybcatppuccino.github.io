// app.js (deterministic JS board-gen + safe undo + minimal freeze risk)

let pyodide=null, jsApi=null, cppApi=null, kernelType="js", switchingKernel=false;
let H=25,W=40,M=200, viewScale=1.8, cellScale=1.7, pageScale=0.8, currentGameSeed=null;
let manualModeEnabled=false, undoSnap=null, stepping=false;

const DBG=true;
const dlog=(...a)=>DBG&&console.log("[DBG]",...a);
const dwarn=(...a)=>DBG&&console.warn("[DBG]",...a);
const derr=(...a)=>console.error("[ERR]",...a);

document.documentElement.style.setProperty("--board-cell-scale", cellScale.toFixed(2));
document.documentElement.style.setProperty("--page-scale", pageScale.toFixed(2));
document.body.style.zoom = pageScale;

const el=(id)=>document.getElementById(id);
const statusEl=el("status"), boardEl=el("board");
const infoEl=document.querySelector(".info-panel");
const controlPanel=document.querySelector(".controls-container"), togglePanelBtn=el("togglePanel");
const viewScaleValueEl=el("viewScaleValue"), cellScaleValueEl=el("cellScaleValue"), pageScaleValueEl=el("pageScaleValue");

const inpH=el("inpH"), inpW=el("inpW"), inpM=el("inpM"), inpSeed=el("inpSeed");
const btnNewGame=el("btnNewGame"), btnStepSolve=el("btnStepSolve"), btnUndo=el("btnUndo");
const btnSwitchKernel=el("btnSwitchKernel");
const firstZeroCheckbox=el("firstzero");

const btnEasy=el("btnEasy"), btnNormal=el("btnNormal"), btnHard=el("btnHard"), btnTranspose=el("btnTranspose");
const btnViewScaleUp=el("btnViewScaleUp"), btnViewScaleDown=el("btnViewScaleDown");
const btnCellScaleUp=el("btnCellScaleUp"), btnCellScaleDown=el("btnCellScaleDown");
const btnPageScaleUp=el("btnPageScaleUp"), btnPageScaleDown=el("btnPageScaleDown");
const btnHMinus5=el("btnHMinus5"), btnHMinus1=el("btnHMinus1"), btnHPlus1=el("btnHPlus1"), btnHPlus5=el("btnHPlus5");
const btnWMinus5=el("btnWMinus5"), btnWMinus1=el("btnWMinus1"), btnWPlus1=el("btnWPlus1"), btnWPlus5=el("btnWPlus5");
const btnMMinus100=el("btnMMinus100"), btnMMinus10=el("btnMMinus10"), btnMMinus1=el("btnMMinus1");
const btnMPlus1=el("btnMPlus1"), btnMPlus10=el("btnMPlus10"), btnMPlus100=el("btnMPlus100");

const FLAG_SVG=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect x="7" y="2" width="1.5" height="12" fill="#000"/><polygon points="8,2 14,5 8,8" fill="#f00"/></svg>`;
const sleep=(ms)=>new Promise(r=>setTimeout(r,ms));
const setStatus=(s)=>statusEl.textContent=s;

const toPlain=(x)=>{ try{ if(x&&typeof x.toJs==="function") return x.toJs(); }catch{} return x; };
const clampInt=(x,lo,hi,f)=>{ const n=Number.parseInt(x,10); return !Number.isFinite(n)?f:Math.max(lo,Math.min(hi,n)); };
const getApi=()=>kernelType==="cpp"?cppApi:jsApi;

function assertApiReady(A){
  if(!A) return false;
  for(const k of ["newGame","step","stepAt","getState","getAnalysis"]) if(typeof A[k]!=="function") return false;
  return true;
}

function normalizeDelta(d){
  d=toPlain(d)||{};
  if(!("move" in d)) d.move=null;
  if(!Array.isArray(d.newly)) d.newly=[];
  if(!Array.isArray(d.ai_mines)) d.ai_mines=[];
  d.lost=!!d.lost; d.won=!!d.won; d.stuck=!!d.stuck;
  if(!Number.isFinite(d.revealed_count)) d.revealed_count=0;
  return d;
}
function normalizeState(st){
  st=toPlain(st)||{};
  st.h=Number.isFinite(st.h)?st.h:H;
  st.w=Number.isFinite(st.w)?st.w:W;
  st.mines=Number.isFinite(st.mines)?st.mines:M;
  if(!Array.isArray(st.revealed)) st.revealed=[];
  if(!Array.isArray(st.ai_mines)) st.ai_mines=[];
  st.revealed_count=Number.isFinite(st.revealed_count)?st.revealed_count:st.revealed.length;
  st.lost=!!st.lost; st.won=!!st.won;
  return st;
}
const hasMove=(d)=>Array.isArray(d?.move)&&d.move.length===2&&Number.isFinite(d.move[0])&&Number.isFinite(d.move[1]);
const key=(r,c)=>`${r},${c}`;

function makeHiddenFieldStrings(h, w) {
  const row = "H".repeat(w);
  return Array.from({ length: h }, () => row);
}

// 如果你需要从“可见状态”生成 field（用于 undo/migrate，不泄漏额外信息）
function stateToFieldStrings(st) {
  const h = st.h, w = st.w;
  const grid = Array.from({ length: h }, () => Array.from({ length: w }, () => "H"));

  // revealed: [r,c,n] 其中 n=-1 代表爆雷格；对 undo 来说可按 H 处理或保留 X（但 C++ 只认 digit/F/H）
  for (const [r, c, n] of (st.revealed || [])) {
    if (n >= 0 && n <= 8) grid[r][c] = String(n);
    else grid[r][c] = "H";
  }

  // 你的规则：F 是 AI 100% 确认真雷 -> 可以写入 'F'
  for (const [r, c] of (st.ai_mines || [])) grid[r][c] = "F";

  return grid.map(row => row.join(""));
}


// ---------------- UI / Board ----------------
const jsRevealed=new Set();
let jsCells=[];

const getProbColor=(p)=>{
  const pp=Math.max(0,Math.min(1,p));
  const sat=90, light=40*(1-pp)+40, hue=120*(1-pp)**1.5, alpha=0.4+0.3*(1-pp);
  return `hsl(${hue}, ${sat}%, ${light}%, ${alpha})`;
};

function buildBoardDOM(h,w){
  jsRevealed.clear();
  jsCells=new Array(h*w);
  boardEl.style.gridTemplateColumns=`repeat(${w}, calc(var(--cell) * var(--board-cell-scale)))`;
  boardEl.innerHTML="";
  for(let r=0;r<h;r++)for(let c=0;c<w;c++){
    const d=document.createElement("div");
    d.className="cell"; d.dataset.r=r; d.dataset.c=c;
    boardEl.appendChild(d);
    jsCells[r*w+c]=d;
  }
}

function clearAnalysisEffects(d){
  if(!d) return;
  d.classList.remove("analyzed","next-move");
  d.style.backgroundColor=""; d.style.color=""; d.style.fontWeight=""; d.style.fontSize="";
  d.style.border=""; d.style.boxShadow=""; d.style.animation="";
}
function setCellCovered(r,c){
  const d=jsCells[r*W+c]; d.className="cell"; d.textContent=""; d.innerHTML="";
  delete d.dataset.number; clearAnalysisEffects(d);
}
function setCellOpen(r,c,n){
  const d=jsCells[r*W+c]; d.className="cell open";
  if(n>0){ d.textContent=String(n); d.dataset.number=String(n); } else { d.textContent=""; delete d.dataset.number; }
  clearAnalysisEffects(d);
}
function setCellMine(r,c){
  const d=jsCells[r*W+c]; d.className="cell mine"; d.textContent="X";
  delete d.dataset.number; clearAnalysisEffects(d);
}
function setCellFlag(r,c){
  const d=jsCells[r*W+c];
  if(d.classList.contains("open")||d.classList.contains("mine")) return;
  d.className="cell flag"; d.innerHTML=FLAG_SVG; delete d.dataset.number; clearAnalysisEffects(d);
}

function updateGameInfo(st){
  const rc=st?.revealed_count||0;
  const fm=(st?.ai_mines||[]).length;
  const density=H>0&&W>0?Math.round((M/(H*W))*100):0;
  const seed=(st&&st.seed!=null)?st.seed:(currentGameSeed??"None");
  infoEl.innerHTML=`
    <div class="info-item"><span class="info-label">Size</span><span class="info-value">${H}×${W}</span></div>
    <div class="info-item"><span class="info-label">Density</span><span class="info-value">${density}%</span></div>
    <div class="info-item"><span class="info-label">Revealed</span><span class="info-value">${rc}/${H*W-M}</span></div>
    <div class="info-item"><span class="info-label">Mines</span><span class="info-value">${fm}/${M}</span></div>
    <div class="info-item seed-info"><span class="info-label">Seed</span><span class="info-value">${seed}</span></div>`;
}

function applyFullState(st0){
  const st=normalizeState(st0);
  H=st.h; W=st.w; M=st.mines;
  if(st.seed!==undefined) currentGameSeed=st.seed;
  buildBoardDOM(H,W);
  for(let r=0;r<H;r++)for(let c=0;c<W;c++) setCellCovered(r,c);
  for(const [r,c,n] of st.revealed){
    jsRevealed.add(key(r,c));
    (n===-1)?setCellMine(r,c):setCellOpen(r,c,n);
  }
  for(const [r,c] of st.ai_mines) setCellFlag(r,c);
  setStatus(st.lost?"GAME OVER":st.won?"YOU WIN":"Ready");
  updateGameInfo(st);
  refreshAnalysisOverlay();
}

function applyStepDelta(d0){
  const d=normalizeDelta(d0);
  for(const [r,c,n] of d.newly){
    jsRevealed.add(key(r,c));
    (n===-1)?setCellMine(r,c):setCellOpen(r,c,n);
  }
  for(const [r,c] of d.ai_mines) setCellFlag(r,c);

  if(d.lost){ setStatus("GAME OVER"); manualModeEnabled=false; }
  else if(d.won){ setStatus("YOU WIN"); manualModeEnabled=false; }
  else if(d.stuck){ setStatus("STUCK (no moves)"); manualModeEnabled=false; }
  else setStatus(`Running | Revealed: ${d.revealed_count}`);

  updateGameInfo({revealed_count:d.revealed_count, ai_mines:d.ai_mines, seed:currentGameSeed});
}

// ---------------- Analysis overlay ----------------
function clearOverlayMarks(){
  for(const cell of document.querySelectorAll(".cell.analyzed")){
    cell.classList.remove("analyzed","next-move");
    if(!cell.classList.contains("open")&&!cell.classList.contains("mine")&&!cell.classList.contains("flag")){
      cell.textContent=""; clearAnalysisEffects(cell);
    } else clearAnalysisEffects(cell);
  }
}
function parseProbsObject(probs){
  probs=toPlain(probs)||{};
  const out=[];
  for(const [k,v] of Object.entries(probs)){
    const m=String(k).match(/(\d+)\s*,\s*(\d+)/); if(!m) continue;
    const r=+m[1], c=+m[2], p=Number(toPlain(v));
    if(Number.isFinite(r)&&Number.isFinite(c)&&Number.isFinite(p)) out.push([r,c,p]);
  }
  return out;
}
function applyAnalysisOverlay(a0){
  const a=toPlain(a0)||{};
  clearOverlayMarks();
  for(const [r,c,p] of parseProbsObject(a.probs)){
    if(r<0||r>=H||c<0||c>=W) continue;
    const d=jsCells[r*W+c];
    if(!d||d.classList.contains("open")||d.classList.contains("mine")||d.classList.contains("flag")) continue;
    d.classList.add("analyzed");
    const pp=Math.max(0,Math.min(1,p));
    d.style.backgroundColor=getProbColor(pp);
    d.textContent=String(Math.round(pp*100)).padStart(2,"0");
    d.style.color="#000"; d.style.fontWeight="normal";
    d.style.fontSize="calc(var(--cell) * var(--board-cell-scale) * 0.6)";
  }
  if(Array.isArray(a.next_move)&&a.next_move.length===2){
    const nr=+a.next_move[0], nc=+a.next_move[1];
    if(nr>=0&&nr<H&&nc>=0&&nc<W){
      const d=jsCells[nr*W+nc];
      if(d&&!d.classList.contains("open")&&!d.classList.contains("mine")&&!d.classList.contains("flag")){
        d.classList.add("analyzed","next-move");
        d.style.backgroundColor="#00FF00"; d.style.color="#FFF"; d.style.fontWeight="bold";
        d.style.border="2px solid #F00"; d.style.boxShadow="0 0 10px #00FF00";
      }
    }
  }
}
function refreshAnalysisOverlay(){
  const A=getApi(); if(!assertApiReady(A)) return;
  try{ applyAnalysisOverlay(toPlain(A.getAnalysis())); } catch(e){ dwarn("analysis failed",e); }
}

// ---------------- Deterministic board generation (JS owns mines) ----------------
// PRNG: xorshift32
function rng32(seed){
  let x=(seed|0)||0x12345678;
  return ()=>{ x|=0; x^=x<<13; x^=x>>>17; x^=x<<5; return x|0; };
}
function randInt(rng, n){
  // unbiased-ish for UI; for strictness you can do rejection sampling
  const u=(rng()>>>0);
  return u % n;
}
function neighbors(h,w,r,c){
  const out=[];
  for(let rr=r-1;rr<=r+1;rr++) if(0<=rr&&rr<h)
    for(let cc=c-1;cc<=c+1;cc++) if(0<=cc&&cc<w&&(rr!==r||cc!==c)) out.push([rr,cc]);
  return out;
}
function genMinesLayout(h,w,m,seed,firstClick,mode){
  const [sr,sc]=firstClick;
  const safe=new Set([key(sr,sc)]);
  if(mode===1){ for(const [rr,cc] of neighbors(h,w,sr,sc)) safe.add(key(rr,cc)); }
  // mode===2 => only the clicked cell is safe (already added)

  const cells=[];
  for(let r=0;r<h;r++) for(let c=0;c<w;c++){
    if(!safe.has(key(r,c))) cells.push([r,c]);
  }
  if(m>cells.length) throw new Error("Too many mines for given safe zone");

  const rng=rng32(seed??0);
  // Fisher-Yates partial shuffle
  for(let i=0;i<m;i++){
    const j=i+randInt(rng, cells.length-i);
    const tmp=cells[i]; cells[i]=cells[j]; cells[j]=tmp;
  }
  const layout=Array.from({length:h},()=>Array(w).fill(0));
  for(let i=0;i<m;i++){
    const [r,c]=cells[i]; layout[r][c]=1;
  }
  return layout;
}

// "field" snapshot for undo/migration: only visible info (H: hidden, F: flag, "0-8": open number)
function stateToField(st){
  const h=st.h,w=st.w;
  const field=Array.from({length:h},()=>Array(w).fill("H"));
  for(const [r,c,n] of st.revealed){
    if(n===-1) field[r][c]="H"; // do NOT expose mine even in lost state
    else field[r][c]=String(n);
  }
  for(const [r,c] of (st.ai_mines||[])) field[r][c]="F";
  return field;
}

// Game session board ownership
let jsBoard = { ready:false, seed:null, firstmv:2, mines_layout:null, first_click:null };

// Push board to kernel (if supported). Returns true if load succeeded.
function kernelLoadBoard(A, payload) {
  if (typeof A?.ms_load_board !== "function") return null;
  try {
    // C++(embind) 通常能吃 object；Python 多半吃 json string
    try { return A.ms_load_board(payload); }
    catch { return A.ms_load_board(JSON.stringify(payload)); }
  } catch (e) {
    console.warn("ms_load_board failed", e);
    return null;
  }
}


// Ensure kernel has our mines_layout before first reveal
function ensureBoardInKernel(A, clickCell) {
  // 你自己原来应当有 jsBoard / ready / seed / firstmv 之类的全局
  if (jsBoard?.ready) return true;

  const seed = (jsBoard?.seed ?? currentGameSeed ?? 0) | 0;
  const mode = (jsBoard?.firstmv ?? 2) | 0;

  jsBoard.first_click = clickCell;
  jsBoard.mines_layout = genMinesLayout(H, W, M, seed, clickCell, mode);
  jsBoard.ready = true;

  const payload = {
    height: H,
    width: W,
    mines: M,
    seed: seed,
    first_move_made: true,               // 已经有真实雷布局了
    field: makeHiddenFieldStrings(H, W), // !!! string[] !!!
    mines_layout: jsBoard.mines_layout
  };

  const st = kernelLoadBoard(A, payload);
  if (st) { applyFullState(st); return true; }
  // 若 load 失败，不阻断（仍可退回让内核自己生成雷，但你要求 2）一般希望这里成功）
  return true;
}


// ---------------- Interaction & Undo ----------------
boardEl.addEventListener("click", handleManualClick);

async function handleManualClick(ev){
  const over = ["GAME OVER","YOU WIN","STUCK (no moves)"].includes(statusEl.textContent);
  if(over||!manualModeEnabled||stepping) return;

  const t=ev.target.closest?.(".cell")||ev.target;
  if(!t.classList.contains("cell")||t.classList.contains("open")||t.classList.contains("flag")||t.classList.contains("mine")) return;
  const r=+t.dataset.r, c=+t.dataset.c; if(!Number.isFinite(r)||!Number.isFinite(c)) return;

  const A=getApi(); if(!assertApiReady(A)) return setStatus("No API");

  // SAFE UNDO SNAPSHOT: visible-only, never mines_layout
  const st0=normalizeState(A.getState());
  undoSnap = { kernel:kernelType, h:st0.h,w:st0.w,mines:st0.mines,seed:currentGameSeed,
               field: stateToField(st0), firstmv: jsBoard.firstmv, jsBoardReady: jsBoard.ready,
               jsBoardSeed: jsBoard.seed, jsMines: jsBoard.mines_layout, jsFirst: jsBoard.first_click };
  btnUndo.style.display = (typeof A?.ms_load_board==="function") ? "block" : (typeof A?.setState==="function"?"block":"none");

  // ensure JS-owned board exists in kernel before first click
  if(!jsBoard.ready) ensureBoardInKernel(A,[r,c]);

  applyStepDelta(A.stepAt(r,c));
  if(["GAME OVER","YOU WIN","STUCK (no moves)"].includes(statusEl.textContent)){ refreshAnalysisOverlay(); return; }

  if(typeof A.makeSafeMove==="function"){
    while(true){
      const ds=normalizeDelta(A.makeSafeMove());
      applyStepDelta(ds);
      if(ds.lost||ds.won||ds.stuck){ refreshAnalysisOverlay(); return; }
      if(!hasMove(ds)) break;
      await sleep(10);
    }
  }
  refreshAnalysisOverlay();
}

async function stepSolve(){
  const A=getApi();
  if(!assertApiReady(A)||stepping||!manualModeEnabled) return;
  stepping=true; manualModeEnabled=false;
  try{
    // drain safe moves first
    if(typeof A.makeSafeMove==="function"){
      while(true){
        const r=normalizeDelta(A.makeSafeMove());
        applyStepDelta(r);
        if(r.lost||r.won||r.stuck){ refreshAnalysisOverlay(); return; }
        if(!hasMove(r)) break;
        await sleep(10);
      }
    }

    // snapshot for undo (visible-only)
    const st0=normalizeState(A.getState());
    undoSnap = { kernel:kernelType, h:st0.h,w:st0.w,mines:st0.mines,seed:currentGameSeed,
                 field: stateToField(st0), firstmv: jsBoard.firstmv, jsBoardReady: jsBoard.ready,
                 jsBoardSeed: jsBoard.seed, jsMines: jsBoard.mines_layout, jsFirst: jsBoard.first_click };
    btnUndo.style.display = (typeof A?.ms_load_board==="function") ? "block" : (typeof A?.setState==="function"?"block":"none");

    const s=normalizeDelta(A.step());
    applyStepDelta(s);
    if(s.lost||s.won||s.stuck){ refreshAnalysisOverlay(); return; }

    if(typeof A.makeSafeMove==="function"){
      while(true){
        const r2=normalizeDelta(A.makeSafeMove());
        applyStepDelta(r2);
        if(r2.lost||r2.won||r2.stuck){ refreshAnalysisOverlay(); return; }
        if(!hasMove(r2)) break;
        await sleep(10);
      }
    }
    refreshAnalysisOverlay();
  }catch(e){
    derr(e); setStatus("stepSolve failed: "+(e?.message||String(e)));
  }finally{
    stepping=false; manualModeEnabled=true;
  }
}

async function undoLastMove() {
  const A = getApi();
  if (!undoState) return;
  if (!assertApiReady(A)) return setStatus("No API available");

  try {
    // undoState 建议你存的是当时的 getState() + 我们的 jsBoard 快照
    // 如果你现在 undoState 只存 getState()，也能用，但建议加 jsBoard 快照（如下）
    const st = normalizeState(undoState);

    // 用“可见信息 field”恢复，不直接 setState（更不容易在 lost 状态泄漏/错乱）
    if (typeof A.ms_load_board === "function") {
      const payload = {
        height: st.h, width: st.w, mines: st.mines,
        seed: (st.seed ?? currentGameSeed ?? 0) | 0,
        first_move_made: !!st.first,
        field: stateToFieldStrings(st),      // !!! string[] !!!
      };

      // 如果当时已经生成过雷版（首击之后），把 mines_layout 一并带回，确保完全一致
      if (jsBoard?.ready && jsBoard?.mines_layout) {
        payload.first_move_made = true;
        payload.mines_layout = jsBoard.mines_layout;
      }

      const restored = kernelLoadBoard(A, payload);
      if (restored) applyFullState(restored);
      else applyFullState(normalizeState(A.setState(st))); // fallback
    } else if (typeof A.setState === "function") {
      applyFullState(normalizeState(A.setState(st)));
    } else {
      return setStatus("Undo not supported");
    }

    btnUndo.style.display = "none";
    undoState = null;
    manualModeEnabled = true;

    setTimeout(() => { refreshAnalysisOverlay(); }, 10);
  } catch (e) {
    derr("undo failed", e);
    setStatus("Undo failed: " + (e?.message || String(e)));
  }
}


// ---------------- Kernel loading ----------------
async function loadPy(){
  if(pyodide&&jsApi) return;
  const urls=[
    "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
    "https://pyodide.org/pyodide/v0.26.4/full/",
    "https://unpkg.com/pyodide@0.26.4/pyodide/full/",
  ];
  setStatus("Loading Pyodide...");
  let last=null;
  for(const u of urls){
    try{ pyodide=await loadPyodide({indexURL:u}); last=null; break; }
    catch(e){ last=e; dwarn("loadPyodide failed",u,e); }
  }
  if(last) throw last;

  setStatus("Loading python code...");
  const resp=await fetch("./py/minesweeper2.py",{cache:"no-cache"});
  if(!resp.ok) throw new Error(`fetch minesweeper2.py failed: ${resp.status}`);
  await pyodide.runPythonAsync(await resp.text());

  jsApi={
    newGame: pyodide.globals.get("ms_new_game"),
    step: pyodide.globals.get("ms_step"),
    stepAt: pyodide.globals.get("ms_step_at"),
    getState: pyodide.globals.get("ms_get_state"),
    makeSafeMove: pyodide.globals.get("ms_make_safe_move"),
    setState: pyodide.globals.get("ms_set_state"),
    getAnalysis: pyodide.globals.get("ms_get_analysis"),
    ms_load_board: pyodide.globals.get("ms_load_board"),
    ms_board_info: pyodide.globals.get("ms_board_info"),
  };
  setStatus("Ready.");
}

function bindCppApiFromModule(){
  const pick=(...names)=>{ for(const n of names) if(typeof Module?.[n]==="function") return Module[n]; };
  cppApi={
    newGame: pick("ms_new_game"),
    step: pick("ms_step"),
    stepAt: pick("ms_step_at"),
    getState: pick("ms_get_state"),
    makeSafeMove: pick("ms_make_safe_move"),
    setState: pick("ms_set_state"),
    getAnalysis: pick("ms_get_analysis"),
    ms_load_board: pick("ms_load_board","loadBoard"),
    ms_board_info: pick("ms_board_info","boardInfo"),
  };
  dlog("cpp api", Object.fromEntries(Object.keys(cppApi).map(k=>[k,typeof cppApi[k]])));
}

window.Module=window.Module||{};
Module.print=Module.print||((t)=>console.log("[WASM]",t));
Module.printErr=Module.printErr||((t)=>console.error("[WASM]",t));
Module.locateFile=Module.locateFile||((path,prefix)=>path.endsWith(".wasm")?"./cpp/"+path:prefix+path);
Module.onRuntimeInitialized=function(){ dlog("C++ runtime initialized"); bindCppApiFromModule(); };

async function ensureCppLoaded(){
  if(cppApi&&typeof cppApi.newGame==="function") return;
  const src="./cpp/minesweeper.js";
  let s=document.querySelector(`script[data-cpp="1"]`);
  if(!s){ s=document.createElement("script"); s.dataset.cpp="1"; s.async=true; document.head.appendChild(s); }
  s.src=src;
  await new Promise((res,rej)=>{ s.onload=res; s.onerror=()=>rej(new Error("Failed to load "+src)); });
  await new Promise((res,rej)=>{
    const t0=performance.now();
    const timer=setInterval(()=>{
      if(typeof Module?.ms_new_game==="function"){ if(!cppApi) bindCppApiFromModule(); clearInterval(timer); res(); }
      else if(performance.now()-t0>20000){ clearInterval(timer); rej(new Error("C++ exports not ready")); }
    },50);
  });
}

// ---------------- Kernel switch (no mines leakage) ----------------
function makeVisibleLoadPayload(fromApi){
  const st=normalizeState(fromApi.getState());
  return {
    height: st.h, width: st.w, mines: st.mines,
    seed: (st.seed!=null)?(st.seed|0):(currentGameSeed==null?0:currentGameSeed|0),
    field: stateToField(st),
    first_move_made: jsBoard.ready,
    mines_layout: jsBoard.mines_layout || undefined
  };
}
async function switchToCppKernel(){
  if(switchingKernel) return;
  switchingKernel=true;
  try{
    setStatus("Switching to C++...");
    await ensureCppLoaded();
    kernelType="cpp";
    btnSwitchKernel && (btnSwitchKernel.textContent="Switch to JS Kernel");
    const A=getApi();
    // try migrate visible state + (optional) JS mines_layout (only if we already have it)
    if(typeof A?.ms_load_board==="function" && jsApi){
      kernelLoadBoard(A, makeVisibleLoadPayload(jsApi));
    } else {
      await createNewGame();
    }
  }catch(e){
    derr(e); setStatus("Switch failed: "+(e?.message||String(e)));
  }finally{ switchingKernel=false; }
}
async function switchToJsKernel(){
  if(switchingKernel) return;
  switchingKernel=true;
  try{
    setStatus("Switching to JS...");
    await loadPy();
    kernelType="js";
    btnSwitchKernel && (btnSwitchKernel.textContent="Switch to C++ Kernel");
    const A=getApi();
    if(typeof A?.ms_load_board==="function" && cppApi){
      kernelLoadBoard(A, makeVisibleLoadPayload(cppApi));
    } else {
      await createNewGame();
    }
  }catch(e){
    derr(e); setStatus("Switch failed: "+(e?.message||String(e)));
  }finally{ switchingKernel=false; }
}
async function switchKernel(){ if(kernelType==="js") await switchToCppKernel(); else await switchToJsKernel(); }

// ---------------- Game creation ----------------
function setDifficulty(h,w,m){ inpH.value=h; inpW.value=w; inpM.value=m; }
function transposeBoard(){
  const h=+inpH.value||25, w=+inpW.value||40, m=+inpM.value||200;
  inpH.value=String(w); inpW.value=String(h); inpM.value=String(m);
}
function adjustParam(id,delta,minVal,maxVal){
  const input=el(id); if(!input) return;
  let v=+input.value||0; v=Math.max(minVal,Math.min(maxVal,v+delta)); input.value=String(v);
}

async function createNewGame() {
  manualModeEnabled = true;
  undoState = null;
  btnUndo.style.display = "none";

  const h = clampInt(inpH.value, 5, 200, 25);
  const w = clampInt(inpW.value, 5, 200, 40);
  const m = clampInt(inpM.value, 1, h*w - 1, Math.min(200, h*w - 1));
  const seedStr = (inpSeed?.value ?? "").trim();
  const seed = seedStr !== "" ? clampInt(seedStr, -2147483648, 2147483647, 0) : 0;
  const firstmv = firstZeroCheckbox?.checked ? 1 : 2;

  inpH.value = String(h); inpW.value = String(w); inpM.value = String(m);
  H = h; W = w; M = m; currentGameSeed = seed;

  // reset JS board ownership
  jsBoard = { ready: false, seed, firstmv, mines_layout: null, first_click: null };

  const A = getApi();
  if (!assertApiReady(A)) return setStatus("No API available for current kernel");

  // 如果内核支持 ms_load_board：先加载一个“全H”的可见字段，让两边起点一致（雷布局稍后首击生成）
  if (typeof A.ms_load_board === "function") {
    const payload = {
      height: H, width: W, mines: M, seed: seed,
      first_move_made: false,
      field: makeHiddenFieldStrings(H, W)   // !!! string[] !!!
    };
    const st = kernelLoadBoard(A, payload);
    if (st) { applyFullState(st); return; }
  }

  // fallback: 仍走旧 newGame
  const st2 = normalizeState(A.newGame(H, W, M, seed, firstmv));
  applyFullState(st2);
}


// ---------------- Wire UI ----------------
btnNewGame?.addEventListener("click", createNewGame);
btnStepSolve?.addEventListener("click", stepSolve);
btnUndo?.addEventListener("click", undoLastMove);
btnSwitchKernel?.addEventListener("click", switchKernel);

btnEasy?.addEventListener("click", ()=>{ setDifficulty(9,9,10); createNewGame(); });
btnNormal?.addEventListener("click", ()=>{ setDifficulty(16,16,40); createNewGame(); });
btnHard?.addEventListener("click", ()=>{ setDifficulty(16,30,99); createNewGame(); });
btnTranspose?.addEventListener("click", ()=>{ transposeBoard(); createNewGame(); });

togglePanelBtn?.addEventListener("click", ()=>{
  const collapsed=controlPanel.classList.contains("collapsed");
  controlPanel.classList.toggle("collapsed", !collapsed);
  togglePanelBtn.textContent=collapsed?"▼":"▲";
});
btnViewScaleUp?.addEventListener("click", ()=>{ viewScale=Math.min(4,viewScale+0.1); viewScaleValueEl.textContent=Math.round(viewScale*100)+"%"; });
btnViewScaleDown?.addEventListener("click", ()=>{ viewScale=Math.max(0.2,viewScale-0.1); viewScaleValueEl.textContent=Math.round(viewScale*100)+"%"; });
btnCellScaleUp?.addEventListener("click", ()=>{ cellScale=Math.min(4,cellScale+0.1); document.documentElement.style.setProperty("--board-cell-scale",cellScale.toFixed(2)); cellScaleValueEl.textContent=Math.round(cellScale*100)+"%"; });
btnCellScaleDown?.addEventListener("click", ()=>{ cellScale=Math.max(0.2,cellScale-0.1); document.documentElement.style.setProperty("--board-cell-scale",cellScale.toFixed(2)); cellScaleValueEl.textContent=Math.round(cellScale*100)+"%"; });
btnPageScaleUp?.addEventListener("click", ()=>{ pageScale=Math.min(4,pageScale+0.1); document.body.style.zoom=pageScale; pageScaleValueEl.textContent=Math.round(pageScale*100)+"%"; });
btnPageScaleDown?.addEventListener("click", ()=>{ pageScale=Math.max(0.2,pageScale-0.1); document.body.style.zoom=pageScale; pageScaleValueEl.textContent=Math.round(pageScale*100)+"%"; });

btnHMinus5?.addEventListener("click", ()=>adjustParam("inpH",-5,5,200));
btnHMinus1?.addEventListener("click", ()=>adjustParam("inpH",-1,5,200));
btnHPlus1?.addEventListener("click", ()=>adjustParam("inpH", 1,5,200));
btnHPlus5?.addEventListener("click", ()=>adjustParam("inpH", 5,5,200));

btnWMinus5?.addEventListener("click", ()=>adjustParam("inpW",-5,5,200));
btnWMinus1?.addEventListener("click", ()=>adjustParam("inpW",-1,5,200));
btnWPlus1?.addEventListener("click", ()=>adjustParam("inpW", 1,5,200));
btnWPlus5?.addEventListener("click", ()=>adjustParam("inpW", 5,5,200));

btnMMinus100?.addEventListener("click", ()=>adjustParam("inpM",-100,1,9999));
btnMMinus10?.addEventListener("click", ()=>adjustParam("inpM",-10,1,9999));
btnMMinus1?.addEventListener("click", ()=>adjustParam("inpM", -1,1,9999));
btnMPlus1?.addEventListener("click", ()=>adjustParam("inpM",  1,1,9999));
btnMPlus10?.addEventListener("click", ()=>adjustParam("inpM", 10,1,9999));
btnMPlus100?.addEventListener("click", ()=>adjustParam("inpM",100,1,9999));

viewScaleValueEl&&(viewScaleValueEl.textContent="180%");
cellScaleValueEl&&(cellScaleValueEl.textContent="170%");
pageScaleValueEl&&(pageScaleValueEl.textContent="80%");
togglePanelBtn&&(togglePanelBtn.textContent="▲");

// ---------------- bootstrap ----------------
(async()=>{
  try{
    await loadPy();
    kernelType="js";
    btnSwitchKernel && (btnSwitchKernel.textContent="Switch to C++ Kernel");
    setDifficulty(16,30,99);
    await new Promise(requestAnimationFrame);
    await createNewGame();
  }catch(e){
    derr(e);
    setStatus("Failed to load Pyodide: "+(e?.stack||String(e)));
  }
})();
