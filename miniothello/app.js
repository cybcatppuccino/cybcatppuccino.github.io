const $ = (selector) => document.querySelector(selector);
const BOARD_SIZE = 6;
const CELL_COUNT = BOARD_SIZE * BOARD_SIZE;
const PASS = -1;
const FILES = ['A', 'B', 'C', 'D', 'E', 'F'];
const ROWS = ['1', '2', '3', '4', '5', '6'];
const DIRS = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0],           [1, 0],
  [-1, 1],  [0, 1],  [1, 1]
];
const BB_BITS = Array.from({ length: CELL_COUNT }, (_, i) => 1n << BigInt(i));
const BB_RAYS = Array.from({ length: CELL_COUNT }, (_, idx) => {
  const x0 = idx % BOARD_SIZE;
  const y0 = Math.floor(idx / BOARD_SIZE);
  return DIRS.map(([dx, dy]) => {
    const ray = [];
    let x = x0 + dx;
    let y = y0 + dy;
    while (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) {
      ray.push(y * BOARD_SIZE + x);
      x += dx;
      y += dy;
    }
    return ray;
  });
});

function bitCount(mask) {
  let n = mask;
  let count = 0;
  while (n) {
    n &= n - 1n;
    count += 1;
  }
  return count;
}

function attachBitboard(b, blackMask = null, whiteMask = null) {
  let black = blackMask;
  let white = whiteMask;
  if (black === null || white === null) {
    black = 0n;
    white = 0n;
    for (let i = 0; i < CELL_COUNT; i += 1) {
      if (b[i] === 'black') black |= BB_BITS[i];
      else if (b[i] === 'white') white |= BB_BITS[i];
    }
  }
  b.__black = black;
  b.__white = white;
  b.__blackCount = bitCount(black);
  b.__whiteCount = bitCount(white);
  b.__key = `${black.toString(36)}:${white.toString(36)}`;
  return b;
}

function ensureBitboard(b) {
  if (!b || typeof b.__black !== 'bigint' || typeof b.__white !== 'bigint') return attachBitboard(b);
  return b;
}

function cloneBoard(b) {
  ensureBitboard(b);
  return attachBitboard(b.slice(0, CELL_COUNT), b.__black, b.__white);
}

function flipsMaskForMove(b, side, idx) {
  if (idx === PASS) return 0n;
  ensureBitboard(b);
  const bit = BB_BITS[idx];
  const occupied = b.__black | b.__white;
  if ((occupied & bit) !== 0n) return 0n;
  const own = side === 'black' ? b.__black : b.__white;
  const enemy = side === 'black' ? b.__white : b.__black;
  let flips = 0n;
  for (const ray of BB_RAYS[idx]) {
    let line = 0n;
    for (const pos of ray) {
      const p = BB_BITS[pos];
      if ((enemy & p) !== 0n) {
        line |= p;
        continue;
      }
      if ((own & p) !== 0n) {
        if (line !== 0n) flips |= line;
        break;
      }
      break;
    }
  }
  return flips;
}

function indicesFromMask(mask) {
  const out = [];
  for (let i = 0; i < CELL_COUNT; i += 1) {
    if ((mask & BB_BITS[i]) !== 0n) out.push(i);
  }
  return out;
}

const CORNERS = new Set([0, 5, 30, 35]);
const EDGES = new Set(Array.from({ length: CELL_COUNT }, (_, i) => i).filter((i) => {
  const x = i % BOARD_SIZE;
  const y = Math.floor(i / BOARD_SIZE);
  return x === 0 || x === BOARD_SIZE - 1 || y === 0 || y === BOARD_SIZE - 1;
}));
const CORNER_ADJACENT = new Set([1, 6, 7, 4, 10, 11, 24, 25, 31, 28, 29, 34]);
const STORAGE = 'miniothello-v6-state';
const BOARD_STYLE_STORAGE = 'miniothello-board-style';
const AI_DELAY_STORAGE = 'miniothello-ai-delay';
const HUMAN_SIDE_STORAGE = 'miniothello-human-side';
const EFFORT_STORAGE = 'miniothello-v6-effort';
const AI_KERNEL_STORAGE = 'miniothello-v6-ai-kernel';
const ORION_ANALYSIS_TIMEOUT_MS = 3500;
const ORION_PLAY_TIMEOUT_MS = 2200;
const ENDGAME_START_DISCS = 18;

const AI_KERNELS = {
  egaroucid: {
    id: 'egaroucid',
    name: 'Egaroucid',
    uiLabel: 'Egaroucid Oracle',
    dbLabel: 'Egaroucid DB',
    baseEngine: 'Egaroucid Oracle · 6×6 WASM',
    fallbackEngine: 'Egaroucid 6×6 WASM',
    searchWorker: 'assets/egaroucid-worker.js',
    searchLabel: 'Egaroucid WASM',
    searchSource: 'Egaroucid WASM iterative',
    solvedSource: 'Egaroucid WASM exact',
    mixedSource: 'Oracle DB + Egaroucid WASM',
    mixedSolvedSource: 'Oracle DB + Egaroucid WASM exact',
    candidateSource: 'Egaroucid WASM',
    fallbackState: 'WASM search',
    fallbackSearchingTitle: 'Egaroucid 6×6 WASM is searching…',
    fallbackSearchingSub: 'Complete-depth evaluations are updated every 0.5s; searches continue without a deadline. Terminal-solved scores are integer discs.',
    fallbackErrorTitle: 'Egaroucid WASM search failed.',
    fallbackErrorSub: 'The Egaroucid WASM worker could not analyze this position.',
    fallbackStartErrorSub: 'Could not start Egaroucid WASM worker.',
    unavailableLog: 'Outside both Oracle databases; starting Egaroucid 6×6 WASM.',
    fullSearchHint: 'No Oracle move; use Egaroucid 6×6 search.'
  },
  poseidon: {
    id: 'poseidon',
    name: 'Poseidon',
    uiLabel: 'Poseidon oracle',
    dbLabel: 'Poseidon DB',
    baseEngine: 'Poseidon oracle v6',
    fallbackEngine: 'Poseidon NNUE policy2',
    searchWorker: 'assets/poseidon-worker.js',
    searchLabel: 'Poseidon NNUE',
    searchSource: 'Poseidon NNUE iterative',
    solvedSource: 'Poseidon NNUE solved',
    mixedSource: 'Poseidon DB + Poseidon NNUE iterative',
    mixedSolvedSource: 'Poseidon DB + Poseidon NNUE solved',
    candidateSource: 'candidate Poseidon NNUE',
    fallbackState: 'NNUE search',
    fallbackSearchingTitle: 'Poseidon NNUE is searching…',
    fallbackSearchingSub: 'Depth, nodes, raw candidate scores and 80ms round-robin terminal predictions publish together; Poseidon uses nnue_policy2_best for move ordering, static eval, and selective pruning; exact endgame remains full-width.',
    fallbackErrorTitle: 'Poseidon NNUE search failed.',
    fallbackErrorSub: 'The Poseidon NNUE worker could not analyze this position.',
    fallbackStartErrorSub: 'The Poseidon NNUE worker could not be started.',
    unavailableLog: 'Poseidon unavailable for this position; starting Poseidon NNUE search.',
    fullSearchHint: 'No white Poseidon/database move; use full Poseidon NNUE search for all legal moves.'
  },
  orion: {
    id: 'orion',
    name: 'Orion',
    uiLabel: 'Orion oracle',
    dbLabel: 'Orion DB',
    baseEngine: 'Orion oracle v5',
    fallbackEngine: 'JS fallback v8 pattern',
    searchWorker: 'assets/fallback-worker.js',
    searchLabel: 'JS fallback',
    searchSource: 'JS fallback iterative',
    solvedSource: 'JS fallback solved',
    mixedSource: 'Orion DB + JS fallback iterative',
    mixedSolvedSource: 'Orion DB + JS fallback solved',
    candidateSource: 'candidate JS fallback',
    fallbackState: 'JS search',
    fallbackSearchingTitle: 'JS fallback is searching…',
    fallbackSearchingSub: 'Depth, nodes, raw candidate scores and 80ms round-robin terminal predictions publish together; Orion keeps the original pattern-based fallback search and full-width exact endgame.',
    fallbackErrorTitle: 'JS fallback search failed.',
    fallbackErrorSub: 'The Orion JS fallback worker could not analyze this position.',
    fallbackStartErrorSub: 'The Orion JS fallback worker could not be started.',
    unavailableLog: 'Orion unavailable for this position; starting JS fallback search.',
    fullSearchHint: 'No white Orion/database move; use full JS fallback search for all legal moves.'
  }
};

function normalizeAiKernel(value) { return AI_KERNELS[value] ? value : 'egaroucid'; }

const elements = {
  board: $('#board'),
  statusText: $('#statusText'),
  turnToken: $('#turnToken'),
  modePill: $('#modePill'),
  undo: $('#undoButton'),
  redo: $('#redoButton'),
  flip: $('#flipButton'),
  mirror: $('#mirrorButton'),
  pass: $('#passButton'),
  boardStyle: $('#boardStyleSelect'),
  analysis: $('#analysisButton'),
  newGame: $('#newGameButton'),
  gameMode: $('#gameModeSelect'),
  humanSide: $('#humanSideSelect'),
  humanSideField: $('#humanSideField'),
  aiKernel: $('#difficultySelect'),
  aiDelay: $('#aiThinkTimeSelect'),
  effort: $('#effortSelect'),
  playEngineStatus: $('#playEngineStatus'),
  moveCount: $('#moveCount'),
  moveTree: $('#moveTree'),
  copyLine: $('#copyLineButton'),
  fenDisplay: $('#fenDisplay'),
  copyFen: $('#copyFenButton'),
  analysisPanel: $('#analysisPanel'),
  analysisKernelEyebrow: $('#analysisKernelEyebrow'),
  analysisState: $('[data-ai-state]'),
  analysisLines: $('#analysisLines'),
  aiEval: $('[data-ai-eval]'),
  aiBalance: $('[data-ai-balance]'),
  aiMarker: $('[data-ai-marker]'),
  mobileEvalFill: $('[data-mobile-eval-fill]'),
  mobileEvalMarker: $('[data-mobile-eval-marker]'),
  mobileEvalLabel: $('[data-mobile-eval-label]'),
  metricTurn: $('#metricTurn'),
  metricLegal: $('#metricLegal'),
  metricSource: $('#metricSource'),
  metricPass: $('#metricPass'),
  metricCache: $('#metricCache'),
  metricFinals: $('#metricFinals'),
  aiEngine: $('[data-ai-engine]'),
  pauseAnalysis: $('#pauseAnalysisButton'),
  resumeAnalysis: $('#resumeAnalysisButton'),
  libraryStatus: $('#libraryStatus'),
  toast: $('#toast')
};

function setOptionalText(el, text) {
  if (el) el.textContent = text;
}
function addOptionalClass(el, className) {
  if (el) el.classList.add(className);
}

class OracleWorkerClient {
  constructor() {
    this.worker = null;
    this.ready = false;
    this.pending = new Map();
    this.cache = new Map();
    this.available = false;
    this.init();
  }

  init() {
    try {
      this.worker = new Worker('assets/orion-worker.js');
      this.worker.addEventListener('message', (event) => {
        this.ready = true;
        this.available = true;
        setOptionalText(elements.libraryStatus, 'Ready');
        addOptionalClass(elements.libraryStatus, 'ready');
        const [moves, result] = event.data;
        const key = keyOf(moves);
        const normalized = normalizeOracleResult(moves, result);
        this.cache.set(key, normalized);
        const queue = this.pending.get(key);
        if (queue) {
          this.pending.delete(key);
          queue.forEach(({ resolve }) => resolve(normalized));
        }
      });
      this.worker.addEventListener('error', (event) => {
        this.available = false;
        setOptionalText(elements.libraryStatus, 'Error');
        addOptionalClass(elements.libraryStatus, 'error');
        console.error('Oracle worker error', event);
        this.rejectAll(new Error('Oracle worker failed'));
      });
    } catch (error) {
      this.available = false;
      setOptionalText(elements.libraryStatus, 'Fallback');
      addOptionalClass(elements.libraryStatus, 'error');
      console.error(error);
    }
  }

  request(moves, options = {}) {
    const cleanMoves = moves.slice();
    const key = keyOf(cleanMoves);
    const timeoutMs = Math.max(0, Number(options.timeoutMs || 0));
    if (this.cache.has(key)) return Promise.resolve({ ...this.cache.get(key), cache: 'Hit', source: kernelSearchSource() });
    if (!this.worker) return Promise.reject(new Error('Oracle worker is not available'));
    return new Promise((resolve, reject) => {
      let timer = null;
      const entry = {
        resolve: (value) => {
          if (timer) clearTimeout(timer);
          resolve(value);
        },
        reject: (error) => {
          if (timer) clearTimeout(timer);
          reject(error);
        }
      };
      const removePending = () => {
        const queue = this.pending.get(key) || [];
        const next = queue.filter((item) => item !== entry);
        if (next.length) this.pending.set(key, next);
        else this.pending.delete(key);
      };
      if (timeoutMs > 0) {
        timer = setTimeout(() => {
          removePending();
          reject(new Error(`${kernelName()} request timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }
      const queue = this.pending.get(key) || [];
      queue.push(entry);
      this.pending.set(key, queue);
      this.worker.postMessage(cleanMoves);
    });
  }

  rejectAll(error) {
    for (const queue of this.pending.values()) queue.forEach(({ reject }) => reject(error));
    this.pending.clear();
  }
}

class BlackOracleWorkerClient {
  constructor() {
    this.worker = null;
    this.pending = new Map();
    this.cache = new Map();
    this.available = false;
    this.nextId = 0;
    this.init();
  }

  init() {
    try {
      this.worker = new Worker('assets/black-oracle-worker.js');
      this.worker.addEventListener('message', (event) => {
        const payload = event.data || {};
        if (payload.type === 'ready') {
          this.available = true;
          setOptionalText(elements.libraryStatus, 'Ready');
          addOptionalClass(elements.libraryStatus, 'ready');
          return;
        }
        if (payload.type === 'error') {
          this.available = false;
          console.warn('Black Oracle WASM failed to load:', payload.message);
          this.rejectAll(new Error(payload.message || 'Black Oracle WASM failed to load'));
          return;
        }
        const entry = this.pending.get(payload.id);
        if (!entry) return;
        this.pending.delete(payload.id);
        if (entry.timer) clearTimeout(entry.timer);
        const result = { ...payload, cache: 'Fresh', source: payload.source || 'Black Oracle DB' };
        this.cache.set(entry.key, result);
        entry.resolve(result);
      });
      this.worker.addEventListener('error', (event) => {
        this.available = false;
        console.error('Black Oracle worker error', event);
        this.rejectAll(new Error('Black Oracle worker failed'));
      });
    } catch (error) {
      this.available = false;
      console.warn('Black Oracle worker unavailable', error);
    }
  }

  request(moves, options = {}) {
    return this.requestBook(moves, 'lookup', options);
  }

  requestWhiteBook(moves, options = {}) {
    return this.requestBook(moves, 'white-book', options);
  }

  requestBook(moves, type, options = {}) {
    const cleanMoves = moves.slice();
    const key = `${type}:${keyOf(cleanMoves)}`;
    if (this.cache.has(key)) return Promise.resolve({ ...this.cache.get(key), cache: 'Hit' });
    if (!this.worker) return Promise.reject(new Error('Oracle book worker is not available'));
    const timeoutMs = Math.max(0, Number(options.timeoutMs ?? ORION_ANALYSIS_TIMEOUT_MS));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const entry = { key, resolve, reject, timer: null };
      if (timeoutMs > 0) {
        entry.timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`${type} book request timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }
      this.pending.set(id, entry);
      try {
        this.worker.postMessage({ type, id, moves: cleanMoves });
      } catch (error) {
        this.pending.delete(id);
        if (entry.timer) clearTimeout(entry.timer);
        reject(error);
      }
    });
  }

  solve(board, side, options = {}) {
    const cleanBoard = Array.from({ length: CELL_COUNT }, (_, i) => {
      const cell = board[i];
      return cell === 'black' ? 1 : (cell === 'white' ? -1 : 0);
    });
    const key = `solve:${side}:${cleanBoard.join(',')}`;
    if (this.cache.has(key)) return Promise.resolve({ ...this.cache.get(key), cache: 'Hit' });
    if (!this.worker) return Promise.reject(new Error('Black Oracle WASM solver is not available'));
    const timeoutMs = Math.max(0, Number(options.timeoutMs ?? 20000));
    const id = ++this.nextId;
    return new Promise((resolve, reject) => {
      const entry = { key, resolve, reject, timer: null };
      if (timeoutMs > 0) {
        entry.timer = setTimeout(() => {
          this.pending.delete(id);
          reject(new Error(`Black Oracle WASM search timed out after ${timeoutMs}ms`));
        }, timeoutMs);
      }
      this.pending.set(id, entry);
      try {
        this.worker.postMessage({ type: 'solve', id, board: cleanBoard, side });
      } catch (error) {
        this.pending.delete(id);
        if (entry.timer) clearTimeout(entry.timer);
        reject(error);
      }
    });
  }

  rejectAll(error) {
    for (const entry of this.pending.values()) {
      if (entry.timer) clearTimeout(entry.timer);
      entry.reject(error);
    }
    this.pending.clear();
  }
}

class TreeNode {
  constructor({ id, parent = null, move = null, source = 'local' }) {
    this.id = id;
    this.parent = parent;
    this.move = move;
    this.source = source;
    this.children = [];
    this.createdAt = Date.now();
  }
}

let nextNodeId = 1;
let root = new TreeNode({ id: 'root' });
let current = root;
let board = createInitialBoard();
let legal = [];
let lastMove = null;
let analysisEnabled = true;
let analysisPaused = false;
let latestAnalysis = null;
let latestAnalysisKey = '';
let pendingAnalysisKey = '';
let aiBusy = false;
let aiTimer = null;
let gameMode = 'local';
let humanSide = localStorage.getItem(HUMAN_SIDE_STORAGE) || 'black';
let aiKernel = normalizeAiKernel(localStorage.getItem(AI_KERNEL_STORAGE) || 'egaroucid');
let aiDelay = Number(localStorage.getItem(AI_DELAY_STORAGE) || '700');
let boardStyle = localStorage.getItem(BOARD_STYLE_STORAGE) || 'standard';
let searchEffort = localStorage.getItem(EFFORT_STORAGE) || 'balanced';
let flipped = false;
let mirrored = false;
let renderSeq = 0;
const oracleKernel = new OracleWorkerClient();
const blackOracleKernel = new BlackOracleWorkerClient();
let fallbackWorker = null;
let fallbackToken = 0;
let activeFallbackKey = '';
let activeAiWasmCancel = null;
const egaroucidAiWaiters = new Set();
let lastFallbackResult = null;
let endgameWorker = null;
let endgameToken = 0;
let activeEndgameKey = '';
const endgameCompletedCache = new Map();
const endgameAiWaiters = new Set();


function currentKernel() { return AI_KERNELS[normalizeAiKernel(aiKernel)]; }
function kernelName() { return currentKernel().name; }
function kernelDbLabel() { return currentKernel().dbLabel; }
function kernelSearchSource() { return `${kernelName()} search`; }
function kernelAnalysisKey(moves = getLineMoves()) { return `${currentKernel().id}:${keyOf(moves)}`; }
function activeOracle() { return oracleKernel; }
function selectedSearchWorkerPath() { return currentKernel().searchWorker; }
function searchWorkerFailureMessage() { return `${currentKernel().searchLabel} worker failed`; }

function createInitialBoard() {
  const b = Array(CELL_COUNT).fill(null);
  b[2 + 2 * BOARD_SIZE] = 'white';
  b[3 + 2 * BOARD_SIZE] = 'black';
  b[2 + 3 * BOARD_SIZE] = 'black';
  b[3 + 3 * BOARD_SIZE] = 'white';
  return attachBitboard(b);
}

function setCell(b, x, y, value) {
  b[y * BOARD_SIZE + x] = value;
  return attachBitboard(b);
}
function getCell(b, x, y) { return b[y * BOARD_SIZE + x]; }
function inBounds(x, y) { return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE; }
function opponent(side) { return side === 'black' ? 'white' : 'black'; }
function sideToMove(moves = getLineMoves()) { return moves.length % 2 === 0 ? 'black' : 'white'; }
function keyOf(moves) { return moves.length ? moves.join(',') : 'start'; }
function coordOf(idx) {
  if (idx === PASS) return 'pass';
  const x = idx % BOARD_SIZE;
  const y = Math.floor(idx / BOARD_SIZE);
  return `${FILES[x]}${ROWS[y]}`;
}
function normalizeOracleResult(moves, result) {
  if (moves.length % 2 === 0) {
    const raw = Array.from(result || []);
    const movesOut = [];
    for (let i = 0; i < raw.length; i += 2) movesOut.push({ move: raw[i], blackValue: raw[i + 1] });
    return { type: 'black-moves', moves: movesOut, cache: 'Fresh', source: kernelSearchSource() };
  }
  return { type: 'white-move', move: Number(result), cache: 'Fresh', source: kernelSearchSource() };
}
function pathTo(node) {
  const out = [];
  let n = node;
  while (n && n.parent) {
    out.push(n.move);
    n = n.parent;
  }
  return out.reverse();
}
function getLineMoves() { return pathTo(current); }
function sourceForPosition(moves) { return moves.length <= 16 ? `${kernelDbLabel()}/search` : kernelSearchSource(); }
function isDatabasePosition(moves) { return moves.length <= 16; }
function isIntegerLike(value) {
  const n = Number(value);
  return Number.isFinite(n) && Math.abs(n - Math.round(n)) < 1e-9;
}
async function requestBlackOraclePosition(moves, timeoutMs = ORION_ANALYSIS_TIMEOUT_MS) {
  try {
    const response = await blackOracleKernel.request(moves, { timeoutMs });
    return response?.found ? response : null;
  } catch (error) {
    console.debug('Black Oracle database miss; continuing with the other database/search.', error);
    return null;
  }
}
async function requestWhiteOracleBookPosition(moves, timeoutMs = ORION_ANALYSIS_TIMEOUT_MS) {
  try {
    const response = await blackOracleKernel.requestWhiteBook(moves, { timeoutMs });
    return response?.found && response.exact ? response : null;
  } catch (error) {
    console.debug('White Oracle database unavailable; using normal analysis.', error);
    return null;
  }
}

// Both books report exact final White-relative values. Never infer database
// membership from depth or from the original Orion worker's search results.
function oracleWhiteScoreAtPosition(response) {
  if (!response || response.exact === false) return null;
  if (response.side === 'black') {
    if (response.bestMove !== undefined && response.scoreBlack !== undefined) {
      return -Number(response.scoreBlack);
    }
    if (Array.isArray(response.options) && response.options.length) {
      return -Math.max(...response.options.map(item => Number(item.blackValue)));
    }
  }
  if (response.side === 'white') {
    if (response.scoreWhite !== undefined) return Number(response.scoreWhite);
    if (Array.isArray(response.options) && response.options.length) {
      return -Math.min(...response.options.map(item => Number(item.blackValue)));
    }
  }
  return null;
}

function whiteOracleBookAnalysis(response, legalNow) {
  if (!response || response.exact === false) return null;
  const source = 'White Oracle DB';
  if (response.side === 'black') {
    const options = (response.options || [])
      .filter(item => legalNow.includes(Number(item.move)) && Number.isFinite(Number(item.blackValue)));
    if (!options.length) return null;
    const lines = options.map(item => ({
      move: Number(item.move), evalWhite: -Number(item.blackValue),
      blackValue: Number(item.blackValue), source,
      pv: [Number(item.move)], exact: true, cache: response.cache || 'Fresh'
    })).sort((a, b) => a.evalWhite - b.evalWhite || a.move - b.move);
    lines[0].oracleBest = true;
    return { side: 'black', lines, evalWhite: lines[0].evalWhite,
      evalExact: true, bestMove: lines[0].move, source, cache: response.cache,
      oracleBestMove: lines[0].move, depth: 'Oracle', exact: lines.length === legalNow.length };
  }
  if (response.side !== 'white') return null;
  const move = Number(response.bestMove), score = Number(response.scoreWhite);
  if (!legalNow.includes(move) || !Number.isFinite(score)) return null;
  const line = { move, evalWhite: score, source, pv: [move],
    exact: true, oracleBest: true, cache: response.cache || 'Fresh' };
  return { side: 'white', lines: [line], evalWhite: score, evalExact: true,
    bestMove: move, source, cache: response.cache || 'Fresh',
    oracleBestMove: move, depth: 'Oracle', exact: true };
}

async function lookupOracleInOrder(moves) {
  const black = await requestBlackOraclePosition(moves);
  if (black) return { response: black, source: 'black' };
  const white = await requestWhiteOracleBookPosition(moves);
  return white ? { response: white, source: 'white' } : null;
}

async function oracleLineAfterMove(moves, b, side, move) {
  const after = applyMoveToBoard(b, side, move);
  if (isGameOver(moves.concat(move), after)) {
    return { move, evalWhite: finalWhiteDiff(after), source: 'terminal',
      pv: [move], exact: true, terminal: true };
  }
  const hit = await lookupOracleInOrder(moves.concat(move));
  if (!hit) return null;
  const evalWhite = oracleWhiteScoreAtPosition(hit.response);
  if (!Number.isFinite(evalWhite)) return null;
  const source = hit.source === 'black' ? 'Black Oracle DB' : 'White Oracle DB';
  return { move, evalWhite, source, pv: [move], exact: true,
    cache: hit.response.cache || 'Fresh' };
}

function hasEndgameThreshold(b) {
  const counts = countsOf(b);
  return counts.black + counts.white >= ENDGAME_START_DISCS;
}
function endgameSource() { return `${kernelName()} endgame WASM`; }
function pendingEndgameAnalysis(b, side, legalNow) {
  return {
    side, lines: legalNow.map(move => makePendingOracleLine(b, side, move)),
    source: endgameSource(), cache: 'WASM', depth: 'Solving 0/' + legalNow.length,
    bestMove: null, evalWhite: null, exact: false, endgameWasm: true,
    backgroundFallback: { type: 'endgame', side, bestMove: null,
      oracleLines: [], legalMoves: legalNow.slice(), source: null }
  };
}
function normalizeEndgameAnalysis(raw, overlay = null) {
  if (!raw || !Array.isArray(raw.lines)) return null;
  const side = overlay?.side || raw.side;
  const legalMoves = overlay?.legalMoves || raw.moves || raw.lines.map(line => line.move);
  const map = new Map();
  for (const line of raw.lines) {
    if (!legalMoves.includes(Number(line.move)) || !Number.isFinite(Number(line.evalWhite))) continue;
    map.set(Number(line.move), { ...line, move: Number(line.move),
      source: endgameSource(), cache: 'WASM', exact: Boolean(line.exact), pending: false });
  }
  // Move-wise precedence: Black DB -> White DB -> endgame WASM -> ordinary.
  for (const line of overlay?.oracleLines || []) map.set(line.move, { ...line, exact: true, pending: false });
  const missing = legalMoves.filter(move => !map.has(move));
  for (const move of missing) map.set(move, { move, evalWhite: null, pending: true,
    source: 'Searching', pv: [move], exact: false });
  const candidates = [...map.values()].filter(line => !line.pending);
  if (!candidates.length) return null;
  candidates.sort((a,b) => side === 'black' ? a.evalWhite-b.evalWhite : b.evalWhite-a.evalWhite);
  // Until every uncovered candidate is terminal-solved, do not announce a
  // WASM 'best move': a partially completed set cannot prove global optimality.
  // A book-proven root best move remains valid immediately.
  const allExact = !missing.length && [...map.values()].every(line => line.exact);
  const bestMove = overlay?.bestMove ?? (allExact ? candidates[0].move : null);
  const compare = oracleLineSort(side, bestMove);
  const sorted = [...map.values()].sort((a,z) => a.pending !== z.pending
    ? (a.pending ? 1 : -1) : compare(a,z));
  const best = bestMove === null ? null : map.get(bestMove);
  return {
    side, lines: sorted, bestMove,
    evalWhite: best?.evalWhite ?? candidates[0].evalWhite,
    evalExact: Boolean(best?.exact), exact: allExact,
    source: overlay?.source ? `${overlay.source} / ${endgameSource()}` : endgameSource(),
    cache: overlay?.source ? 'Mixed' : 'WASM', endgameWasm: true,
    depth: allExact ? 'Endgame exact' : `Solving ${raw.completed ?? candidates.length}/${raw.total ?? legalMoves.length}`,
    nodes: raw.nodes, backgroundFallback: null,
    _missing: missing
  };
}
function cancelEndgameAiWaiters() {
  for (const waiter of endgameAiWaiters) waiter.reject(new Error('Exact endgame WASM search canceled'));
  endgameAiWaiters.clear();
}
function deliverEndgameAiResult(key, raw) {
  for (const waiter of [...endgameAiWaiters]) {
    if (waiter.key === key) {
      endgameAiWaiters.delete(waiter);
      waiter.resolve(raw);
    }
  }
}
function stopEndgameSearch() {
  if (endgameWorker) endgameWorker.terminate();
  endgameWorker = null;
  activeEndgameKey = '';
  // A waiter is only kept alive while its specific UI Worker is active.
  cancelEndgameAiWaiters();
}
function waitForLiveEndgameAi(b, side) {
  if (aiKernel !== 'egaroucid' || !analysisEnabled || analysisPaused ||
      !endgameWorker || activeEndgameKey !== kernelAnalysisKey(getLineMoves()) ||
      sideToMove() !== side || !b.every((disc, i) => disc === board[i])) return null;
  const key = `${side}:${b.slice(0,CELL_COUNT).join(',')}`;
  return new Promise((resolve, reject) => endgameAiWaiters.add({ key, resolve, reject }));
}
function startEndgameAnalysis(moves, b, analysisKey, overlay = null) {
  stopEndgameSearch();
  stopFallbackSearch(true);
  activeEndgameKey = analysisKey;
  const token = ++endgameToken;
  const side = sideToMove(moves);
  elements.analysisState.textContent = 'Searching';
  try {
    endgameWorker = new Worker('assets/endgame-worker.js');
    endgameWorker.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.token !== token || payload.key !== analysisKey || pendingAnalysisKey !== analysisKey) return;
      if (payload.type === 'error') {
        console.error('Exact endgame WASM failed:', payload.message);
        stopEndgameSearch();
        elements.analysisState.textContent = 'Error';
        // Never substitute a limited-depth ordinary score for an exact one.
        return;
      }
      if (payload.type !== 'update' && payload.type !== 'done') return;
      const result = normalizeEndgameAnalysis(payload.result, overlay);
      if (result) {
        latestAnalysis = result;
        latestAnalysisKey = analysisKey;
        lastFallbackResult = result;
        renderStatus();
        renderAnalysis(result);
        renderBoard();
      }
      if (payload.type === 'done') {
        if (!result?.exact || result._missing.length) {
          elements.analysisState.textContent = 'Error';
          console.error('Exact endgame WASM returned an incomplete position.');
        } else {
          const key = `${side}:${b.slice(0,CELL_COUNT).join(',')}`;
          endgameCompletedCache.set(key, payload.result);
          if (endgameCompletedCache.size > 120) endgameCompletedCache.delete(endgameCompletedCache.keys().next().value);
          deliverEndgameAiResult(key, payload.result);
          elements.analysisState.textContent = 'Ready';
        }
        stopEndgameSearch();
      }
    };
    endgameWorker.onerror = (error) => {
      console.error('Exact endgame worker error:', error);
      stopEndgameSearch();
      if (pendingAnalysisKey === analysisKey) elements.analysisState.textContent = 'Error';
    };
    endgameWorker.postMessage({ type: 'analyze', token, key: analysisKey,
      board: b.slice(0,CELL_COUNT), side });
  } catch(error) {
    console.error('Exact endgame worker startup error:', error);
    stopEndgameSearch();
    elements.analysisState.textContent = 'Error';
  }
}
function requestEndgameForAi(b, side, options = {}) {
  const key = `${side}:${b.slice(0,CELL_COUNT).join(',')}`;
  const cached = endgameCompletedCache.get(key);
  if (cached?.exact) return Promise.resolve(cached);
  // With Egaroucid selected, AI can await an already running exact-analysis
  // WASM Worker. Do not start another 32 MiB instance for the same board.
  if (options.aiRequest) {
    const live = waitForLiveEndgameAi(b, side);
    if (live) return live;
  }
  // No deadline, node budget or provisional estimate. Only a complete root
  // solve may release a final AI move. Board changes cancel AI-owned workers.
  return new Promise((resolve,reject) => {
    let worker;
    const cancel = options.aiRequest ? () => {
      cleanup(); reject(new Error('Exact endgame WASM AI canceled after board change'));
    } : null;
    const cleanup = () => {
      if (activeAiWasmCancel === cancel) activeAiWasmCancel = null;
      worker?.terminate(); worker = null;
    };
    try {
      worker = new Worker('assets/endgame-worker.js');
      if (cancel) activeAiWasmCancel = cancel;
      worker.onmessage = (event) => {
        const payload=event.data||{};
        if(payload.type === 'update') return; // exact individual moves, incomplete root
        if(payload.type === 'error') {
          cleanup();reject(new Error(payload.message||'Exact endgame WASM failed'));
          return;
        }
        if(payload.type === 'done') {
          cleanup();
          if(payload.result?.exact && payload.result.lines?.length === payload.result.total) {
            endgameCompletedCache.set(key,payload.result);
            resolve(payload.result);
          } else reject(new Error('Exact endgame WASM returned an incomplete result'));
        }
      };
      worker.onerror = (error) => {cleanup();reject(error);};
      worker.postMessage({type:'analyze',board:b.slice(0,CELL_COUNT),side});
    } catch(error) {cleanup();reject(error);}
  });
}

function makePendingOracleLine(b, side, move) {
  return { move, evalWhite: null, pending: true, exact: false,
    source: 'Searching', pv: [move], cache: 'Fallback' };
}

function oracleLineSort(side, bestMove) {
  return (a, b) => {
    if (a.move === bestMove && b.move !== bestMove) return -1;
    if (b.move === bestMove && a.move !== bestMove) return 1;
    if (a.pending !== b.pending) return a.pending ? 1 : -1;
    return side === 'black' ? Number(a.evalWhite) - Number(b.evalWhite) : Number(b.evalWhite) - Number(a.evalWhite);
  };
}

async function completeOracleAnalysis(moves, b, side, legalNow, primary) {
  if (!primary || primary.bestMove === null || primary.bestMove === undefined) return null;
  const lines = new Map(primary.lines.map(line => [Number(line.move), { ...line, exact: true }]));
  const otherMoves = legalNow.filter(move => !lines.has(move));
  const otherLines = await Promise.all(otherMoves.map(move => oracleLineAfterMove(moves, b, side, move)));
  for (let i = 0; i < otherMoves.length; i++) {
    if (otherLines[i]) lines.set(otherMoves[i], otherLines[i]);
  }
  const uncovered = legalNow.filter(move => !lines.has(move));
  for (const move of uncovered) lines.set(move, makePendingOracleLine(b, side, move));
  const sorted = [...lines.values()].sort(oracleLineSort(side, primary.bestMove));
  const isMixed = uncovered.length > 0;
  return {
    ...primary,
    lines: sorted,
    exact: !isMixed && sorted.every(line => line.exact),
    evalExact: Boolean(sorted.find(line => line.move === primary.bestMove)?.exact),
    cache: isMixed ? 'Mixed' : primary.cache,
    depth: isMixed ? 'Mixed' : 'Oracle',
    backgroundFallback: isMixed ? {
      type: 'oracle-mixed', side, bestMove: primary.bestMove,
      oracleLines: sorted.filter(line => line.exact),
      legalMoves: legalNow.slice(), source: primary.source
    } : null
  };
}

function mergeOracleWithFallback(fallbackResult, overlay) {
  const lines = new Map();
  for (const line of fallbackResult.lines || []) {
    if (!overlay.legalMoves.includes(line.move)) continue;
    lines.set(line.move, {
      ...line, exact: Boolean(line.exact || fallbackResult.exact),
      cache: 'Fallback', source: line.source || fallbackResult.source || currentKernel().searchSource
    });
  }
  // Book values always win over fallback search, even when the two trees overlap.
  for (const line of overlay.oracleLines) lines.set(line.move, { ...line, exact: true });
  for (const move of overlay.legalMoves) {
    if (!lines.has(move)) lines.set(move, { move, evalWhite: null,
      pending: true, exact: false, pv: [move], source: 'Searching' });
  }
  const sorted = [...lines.values()].sort(oracleLineSort(overlay.side, overlay.bestMove));
  const best = sorted.find(line => line.move === overlay.bestMove);
  return {
    ...fallbackResult, side: overlay.side, lines: sorted,
    bestMove: overlay.bestMove, oracleBestMove: overlay.bestMove,
    evalWhite: best?.evalWhite ?? fallbackResult.evalWhite,
    evalExact: Boolean(best?.exact),
    exact: sorted.every(line => line.exact),
    source: `${overlay.source} / ${fallbackResult.source || currentKernel().searchSource}`,
    cache: 'Mixed'
  };
}

async function requestBlackOracleSolve(b, side, timeoutMs = 20000) {
  try {
    const response = await blackOracleKernel.solve(b, side, { timeoutMs });
    return response?.found && response.exact ? response : null;
  } catch (error) {
    console.debug('Black Oracle WASM search unavailable; continuing with the fallback search.', error);
    return null;
  }
}
function blackOracleBlackAnalysis(response, legalNow) {
  if (!response || response.exact === false || response.side !== 'black' || response.bestMove === null || response.bestMove === undefined) return null;
  const move = Number(response.bestMove);
  if (!legalNow.includes(move)) return null;
  const blackValue = Number(response.scoreBlack);
  if (!Number.isFinite(blackValue)) return null;
  const exact = isIntegerLike(blackValue);
  const line = {
    move,
    evalWhite: -blackValue,
    blackValue,
    source: response.source || 'Black Oracle DB',
    pv: [move],
    cache: response.cache || 'Fresh',
    exact,
    oracleBest: true
  };
  return {
    side: 'black',
    lines: [line],
    evalWhite: line.evalWhite,
    evalExact: exact,
    bestMove: move,
    source: response.source || 'Black Oracle DB',
    cache: response.cache || 'Fresh',
    oracleBestMove: move,
    depth: 'Oracle',
    nodes: response.nodes ?? '—',
    exact
  };
}
function blackOracleWhiteAnalysis(response, legalNow) {
  if (!response || response.exact === false || response.side !== 'white') return null;
  const options = (response.options || []).filter((item) => legalNow.includes(Number(item.move)) && Number.isFinite(Number(item.blackValue)));
  if (!options.length) return null;
  const best = options.reduce((currentBest, item) => Number(item.blackValue) < Number(currentBest.blackValue) ? item : currentBest, options[0]);
  const lines = options.map((item) => ({
    move: Number(item.move),
    evalWhite: -Number(item.blackValue),
    blackValue: Number(item.blackValue),
    source: response.source || 'Black Oracle DB',
    pv: [Number(item.move)].concat(item.blackReply !== null && item.blackReply !== undefined ? [Number(item.blackReply)] : []),
    cache: response.cache || 'Fresh',
    exact: isIntegerLike(item.blackValue),
    terminal: item.blackReply === null || item.blackReply === undefined,
    oracleBest: Number(item.move) === Number(best.move)
  })).sort(whiteLineSort(Number(best.move)));
  const exact = lines.length > 0 && lines.every((line) => line.exact);
  const bestLine = lines.find((line) => line.move === Number(best.move)) || lines[0];
  return {
    side: 'white',
    lines,
    evalWhite: bestLine.evalWhite,
    evalExact: Boolean(bestLine.exact),
    bestMove: bestLine.move,
    source: response.source || 'Black Oracle DB',
    cache: response.cache || 'Fresh',
    oracleBestMove: bestLine.move,
    depth: 'Oracle',
    nodes: response.nodes ?? '—',
    exact
  };
}
function whiteLineSort(oracleMove = null) {
  return (a, z) => {
    if (oracleMove !== null && oracleMove !== undefined) {
      if (a.move === oracleMove && z.move !== oracleMove) return -1;
      if (z.move === oracleMove && a.move !== oracleMove) return 1;
    }
    return Number(z.evalWhite || 0) - Number(a.evalWhite || 0);
  };
}
function legalMovesFor(b, side) {
  ensureBitboard(b);
  const occupied = b.__black | b.__white;
  const moves = [];
  for (let idx = 0; idx < CELL_COUNT; idx += 1) {
    if ((occupied & BB_BITS[idx]) !== 0n) continue;
    if (flipsMaskForMove(b, side, idx) !== 0n) moves.push(idx);
  }
  return moves;
}

function flipsForMove(b, side, idx) {
  return indicesFromMask(flipsMaskForMove(b, side, idx));
}

function applyMoveToBoard(b, side, move) {
  ensureBitboard(b);
  const next = b.slice(0, CELL_COUNT);
  if (move === PASS) return attachBitboard(next, b.__black, b.__white);
  const moveBit = BB_BITS[move];
  const flips = flipsMaskForMove(b, side, move);
  const flipIndices = indicesFromMask(flips);
  next[move] = side;
  for (const idx of flipIndices) next[idx] = side;
  if (side === 'black') {
    return attachBitboard(next, b.__black | moveBit | flips, b.__white & ~flips);
  }
  return attachBitboard(next, b.__black & ~flips, b.__white | moveBit | flips);
}

function rebuildBoard(moves = getLineMoves()) {
  let b = createInitialBoard();
  for (let ply = 0; ply < moves.length; ply++) {
    b = applyMoveToBoard(b, ply % 2 === 0 ? 'black' : 'white', moves[ply]);
  }
  return b;
}

function countsOf(b = board) {
  ensureBitboard(b);
  const black = b.__blackCount;
  const white = b.__whiteCount;
  return { black, white, empty: CELL_COUNT - black - white, whiteDiff: white - black, blackDiff: black - white };
}

function finalWhiteDiff(b = board) {
  const c = countsOf(b);
  const diff = c.white - c.black;
  if (diff > 0) return diff + c.empty;
  if (diff < 0) return diff - c.empty;
  return 0;
}

function isGameOver(moves = getLineMoves(), b = board) {
  if (countsOf(b).empty === 0) return true;
  if (moves.length >= 2 && moves[moves.length - 1] === PASS && moves[moves.length - 2] === PASS) return true;
  const side = sideToMove(moves);
  return legalMovesFor(b, side).length === 0 && legalMovesFor(b, opponent(side)).length === 0;
}

function makeChild(move, source = 'local') {
  let child = current.children.find((n) => n.move === move);
  if (!child) {
    child = new TreeNode({ id: `n${nextNodeId++}`, parent: current, move, source });
    current.children.push(child);
  } else if (source !== 'local' && child.source === 'local') {
    child.source = source;
  }
  current = child;
  lastMove = move;
  board = rebuildBoard();
}

function playMove(move, source = 'local') {
  if (isGameOver()) return false;
  const side = sideToMove();
  legal = legalMovesFor(board, side);
  if (move === PASS) {
    if (legal.length > 0) {
      toast('Pass is only legal when there are no playable squares.');
      return false;
    }
  } else if (!legal.includes(move)) {
    toast(`${coordOf(move)} is not legal for ${side}.`);
    return false;
  }
  makeChild(move, source);
  latestAnalysis = null;
  latestAnalysisKey = '';
  stopFallbackSearch();
  cancelAiTimer();
  updateUI({ requestAnalysis: true, scheduleAi: true });
  return true;
}

function navigate(node) {
  current = node;
  const moves = getLineMoves();
  lastMove = moves.length ? moves[moves.length - 1] : null;
  board = rebuildBoard(moves);
  latestAnalysis = null;
  latestAnalysisKey = '';
  stopFallbackSearch();
  cancelAiTimer();
  updateUI({ requestAnalysis: true, scheduleAi: true });
}

function boardIndexFromVisualIndex(vIdx) {
  let x = vIdx % BOARD_SIZE;
  let y = Math.floor(vIdx / BOARD_SIZE);
  if (mirrored) x = BOARD_SIZE - 1 - x;
  if (flipped) y = BOARD_SIZE - 1 - y;
  return y * BOARD_SIZE + x;
}

function candidateMapForCurrentPosition() {
  const moves = getLineMoves();
  if (latestAnalysisKey !== kernelAnalysisKey(moves) || !latestAnalysis?.lines) return new Map();
  const map = new Map();
  latestAnalysis.lines.forEach((line, index) => {
    if (!line.pending && Number.isFinite(Number(line.evalWhite))) {
      map.set(line.move, { ...line, index, exact: Boolean(line.exact || latestAnalysis.exact || latestAnalysis.terminal) });
    }
  });
  return map;
}

function renderBoard() {
  const seq = ++renderSeq;
  const moves = getLineMoves();
  const side = sideToMove(moves);
  const over = isGameOver(moves, board);
  const currentLegal = over ? [] : legalMovesFor(board, side);
  const candidateMap = candidateMapForCurrentPosition();
  const bestMove = latestAnalysisKey === kernelAnalysisKey(moves) ? latestAnalysis?.bestMove : null;
  const canClick = isHumanTurn();

  elements.board.innerHTML = '';
  elements.board.dataset.boardStyle = boardStyle;
  elements.board.classList.toggle('ai-locked', !canClick && !over);

  for (let visual = 0; visual < CELL_COUNT; visual++) {
    const idx = boardIndexFromVisualIndex(visual);
    const x = idx % BOARD_SIZE;
    const y = Math.floor(idx / BOARD_SIZE);
    const cell = document.createElement('button');
    cell.type = 'button';
    cell.className = 'square light';
    cell.dataset.idx = String(idx);
    cell.setAttribute('aria-label', `${coordOf(idx)} ${board[idx] || 'empty'}`);
    if (lastMove === idx) cell.classList.add('last-move');

    const isCandidate = !over && currentLegal.includes(idx);
    const candidate = candidateMap.get(idx);
    if (isCandidate) {
      cell.classList.add('candidate-square');
      if (!candidate || candidate.pending) cell.classList.add('candidate-pending');
      if (candidate && !candidate.pending) cell.classList.add('scored-candidate');
      if (bestMove === idx) cell.classList.add('best-candidate');
    }

    const visualX = visual % BOARD_SIZE;
    const visualY = Math.floor(visual / BOARD_SIZE);
    if (visualY === BOARD_SIZE - 1) {
      const file = document.createElement('span');
      file.className = 'coord-file';
      file.textContent = FILES[x];
      cell.appendChild(file);
    }
    if (visualX === 0) {
      const rank = document.createElement('span');
      rank.className = 'coord-rank';
      rank.textContent = ROWS[y];
      cell.appendChild(rank);
    }
    if (board[idx]) {
      const disc = document.createElement('span');
      disc.className = `disc ${board[idx]}`;
      cell.appendChild(disc);
    } else if (candidate && !candidate.pending) {
      if (!candidate.exact && Number.isFinite(Number(candidate.predictedFinalWhite))) {
        const predicted = document.createElement('span');
        predicted.className = 'candidate-final-score';
        predicted.textContent = formatScore(candidate.predictedFinalWhite, false);
        const samples = Number(candidate.predictedFinalSamples || 0);
        predicted.title = `Predicted final score ${formatScore(candidate.predictedFinalWhite, false)}${samples ? ` · ${samples} rollout${samples === 1 ? '' : 's'}` : ''}`;
        cell.appendChild(predicted);
      }
      const score = document.createElement('span');
      score.className = 'candidate-score';
      score.textContent = formatScore(candidate.evalWhite, candidate.exact);
      score.title = `${candidate.exact ? 'Final' : 'Raw'} White-relative score ${formatScore(candidate.evalWhite, candidate.exact)}`;
      cell.appendChild(score);
    }
    cell.addEventListener('click', () => {
      if (seq !== renderSeq) return;
      if (!canClick) return toast(`${kernelName()} is thinking.`);
      playMove(idx);
    });
    elements.board.appendChild(cell);
  }
}

function renderStatus() {
  const moves = getLineMoves();
  const side = sideToMove(moves);
  const counts = countsOf(board);
  const over = isGameOver(moves, board);
  legal = over ? [] : legalMovesFor(board, side);
  const rawDiffLabel = `B ${counts.black} / W ${counts.white} / diff ${counts.whiteDiff > 0 ? '+' : ''}${counts.whiteDiff}`;
  if (over) {
    const score = finalWhiteDiff(board);
    const scoreLabel = `B ${counts.black} / W ${counts.white} / empty ${counts.empty} / score ${score > 0 ? '+' : ''}${score}`;
    const result = score > 0 ? 'White win' : score < 0 ? 'Black win' : 'Draw';
    elements.statusText.textContent = `${result} · ${scoreLabel}`;
    elements.turnToken.textContent = 'FINISHED';
    elements.turnToken.className = 'turn-token finished';
  } else {
    elements.statusText.textContent = `${capitalize(side)} move${legal.length === 0 ? ' · only pass' : ''} · ${rawDiffLabel}`;
    elements.turnToken.textContent = legal.length === 0 ? 'PASS' : side.toUpperCase();
    elements.turnToken.className = `turn-token ${side}`;
  }
  elements.metricTurn.textContent = over ? 'Game over' : capitalize(side);
  elements.metricLegal.textContent = over ? '0' : String(legal.length);
  elements.metricSource.textContent = latestAnalysis?.source || sourceForPosition(moves);
  elements.metricPass.textContent = latestAnalysis?.depth ?? '—';
  elements.metricCache.textContent = latestAnalysis?.nodes ? formatNodes(latestAnalysis.nodes) : (latestAnalysis?.cache || '—');
  if (elements.metricFinals) elements.metricFinals.textContent = latestAnalysis?.uiCalibrationTerminalSamples !== undefined ? formatNodes(latestAnalysis.uiCalibrationTerminalSamples) : '—';
  elements.pass.disabled = over || legal.length > 0 || !isHumanTurn();
  elements.pass.classList.toggle('pass-ready', !elements.pass.disabled);
  elements.undo.disabled = !current.parent;
  elements.redo.disabled = current.children.length === 0;
  elements.moveCount.textContent = `${moves.length} ${moves.length === 1 ? 'ply' : 'plies'}`;
  elements.fenDisplay.textContent = makePositionKey(moves, board);
  updateModePill();
}

function updateModePill() {
  const labels = { local: 'Local two-player', 'human-ai': `Player vs ${kernelName()} · ${capitalize(humanSide)}`, 'ai-ai': `${kernelName()} self-play` };
  elements.modePill.innerHTML = `<i></i> ${labels[gameMode] || labels.local}`;
  if (elements.analysisKernelEyebrow) elements.analysisKernelEyebrow.textContent = `${kernelName()} local oracle`;
  if (elements.aiKernel && elements.aiKernel.value !== aiKernel) elements.aiKernel.value = aiKernel;
  elements.humanSideField.style.display = gameMode === 'human-ai' ? '' : 'none';
  elements.playEngineStatus.textContent = aiBusy ? `${kernelName()} thinking` : 'Ready';
  elements.playEngineStatus.className = aiBusy ? 'play-engine-status thinking' : 'play-engine-status';
}

function renderMoves() {
  const moves = getLineMoves();
  if (!moves.length) {
    elements.moveTree.className = 'move-tree empty-state';
    elements.moveTree.innerHTML = `<div class="empty-illustration">●</div><strong>No moves yet</strong><span>Play a move or choose a ${kernelName()} candidate.</span>`;
    return;
  }
  elements.moveTree.className = 'move-tree';
  const line = document.createElement('div');
  line.className = 'move-line';
  for (let i = 0; i < moves.length; i += 2) {
    const num = document.createElement('span');
    num.className = 'move-number';
    num.textContent = `${i / 2 + 1}.`;
    line.appendChild(num);
    line.appendChild(makeMoveButton(i, moves[i], 'black-move'));
    if (moves[i + 1] !== undefined) line.appendChild(makeMoveButton(i + 1, moves[i + 1], 'white-move'));
  }
  elements.moveTree.innerHTML = '';
  elements.moveTree.appendChild(line);
}

function makeMoveButton(ply, move, colorClass) {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `move-entry ${colorClass}${move === PASS ? ' pass' : ''}`;
  button.textContent = coordOf(move);
  button.title = `Go to ply ${ply + 1}`;
  button.addEventListener('click', () => {
    let node = root;
    const moves = getLineMoves();
    for (let i = 0; i <= ply; i++) {
      const next = node.children.find((child) => child.move === moves[i]);
      if (!next) return;
      node = next;
    }
    navigate(node);
  });
  return button;
}

function renderAnalysisPlaceholder(text, sub = '') {
  elements.analysisLines.innerHTML = `<div class="analysis-placeholder"><strong>${escapeHtml(text)}</strong><span>${escapeHtml(sub)}</span></div>`;
}

async function requestAnalysis() {
  if (!analysisEnabled || analysisPaused) return;
  stopFallbackSearch();
  stopEndgameSearch();
  const moves = getLineMoves();
  const analysisKey = kernelAnalysisKey(moves);
  pendingAnalysisKey = analysisKey;
  latestAnalysisKey = '';
  elements.analysisState.textContent = 'Thinking';
  elements.analysisPanel.classList.remove('analysis-disabled');
  renderAnalysisPlaceholder(`${kernelName()} is thinking…`, 'Candidate scores will appear on the board.');
  renderBoard();

  try {
    const result = await analyzePosition(moves, board);
    if (pendingAnalysisKey !== analysisKey) return;
    stopFallbackSearch();
    latestAnalysis = result;
    latestAnalysisKey = analysisKey;
    renderStatus();
    renderAnalysis(result);
    renderBoard();
    if (result.backgroundFallback?.type === 'oracle-mixed') {
      if (hasEndgameThreshold(board)) startEndgameAnalysis(moves, board, analysisKey, result.backgroundFallback);
      else startOracleMixedFallbackAnalysis(moves, board, analysisKey, result.backgroundFallback);
    } else if (result.backgroundFallback?.type === 'endgame') {
      startEndgameAnalysis(moves, board, analysisKey);
    }
  } catch (error) {
    console.warn(currentKernel().unavailableLog, error);
    if (pendingAnalysisKey !== analysisKey) return;
    if (hasEndgameThreshold(board)) {
      // A transient Oracle failure must not downgrade 18+ disc analysis to
      // an inexact ordinary search. Keep the exact WASM route.
      const legalNow = legalMovesFor(board,sideToMove(moves));
      const pending = pendingEndgameAnalysis(board,sideToMove(moves),legalNow);
      latestAnalysis = pending;latestAnalysisKey=analysisKey;
      renderAnalysis(pending);renderBoard();
      startEndgameAnalysis(moves,board,analysisKey);
      return;
    }
    startFallbackAnalysis(moves, board, analysisKey, error);
  }
}

async function analyzePosition(moves, b) {
  const side = sideToMove(moves);
  const legalNow = legalMovesFor(b, side);
  if (isGameOver(moves, b)) {
    return { side, lines: [], evalWhite: finalWhiteDiff(b), bestMove: null,
      source: 'terminal', cache: 'Fresh', terminal: true, exact: true };
  }
  if (!legalNow.length) {
    // Early Egaroucid deepening handles passes itself. At 18+ discs,
    // hand the forced PASS to the dedicated exact endgame WASM instead.
    if (aiKernel === 'egaroucid') {
      if (hasEndgameThreshold(b)) return pendingEndgameAnalysis(b, side, [PASS]);
      throw new Error('WASM continuous forced-pass analysis');
    }
    const downstream = await analyzePositionShallow(moves.concat(PASS), b);
    return { side, source: 'forced pass', cache: downstream.cache || 'Fresh',
      bestMove: PASS, evalWhite: downstream.evalWhite,
      lines: [{ move: PASS, evalWhite: downstream.evalWhite, source: 'pass',
        pv: [PASS].concat(downstream.bestMove !== null && downstream.bestMove !== undefined ? [downstream.bestMove] : []),
        exact: Boolean(downstream.exact) }], exact: Boolean(downstream.exact),
      terminal: Boolean(downstream.terminal) };
  }

  // Symmetric precedence for either side to move: Black book, then White book.
  // The old Orion API searches when the book misses; never treat it as a DB hit.
  const hit = await lookupOracleInOrder(moves);
  if (!hit) {
    if (hasEndgameThreshold(b)) return pendingEndgameAnalysis(b,side,legalNow);
    throw new Error('Position is outside both Oracle trees; run ordinary analysis.');
  }
  const primary = hit.source === 'black'
    ? (side === 'black' ? blackOracleBlackAnalysis(hit.response, legalNow) : blackOracleWhiteAnalysis(hit.response, legalNow))
    : whiteOracleBookAnalysis(hit.response, legalNow);
  if (!primary) {
    if (hasEndgameThreshold(b)) return pendingEndgameAnalysis(b, side, legalNow);
    throw new Error('No legal oracle line at this position; run ordinary analysis.');
  }
  return completeOracleAnalysis(moves, b, side, legalNow, primary);
}

async function analyzePositionShallow(moves, b) {
  if (isGameOver(moves, b)) return { evalWhite: finalWhiteDiff(b),
    bestMove: null, cache: 'Fresh', terminal: true, exact: true };
  const side = sideToMove(moves);
  const legalNow = legalMovesFor(b, side);
  if (!legalNow.length) return analyzePositionShallow(moves.concat(PASS), b);
  const hit = await lookupOracleInOrder(moves);
  if (hit) {
    const analysis = hit.source === 'black'
      ? (side === 'black' ? blackOracleBlackAnalysis(hit.response, legalNow) : blackOracleWhiteAnalysis(hit.response, legalNow))
      : whiteOracleBookAnalysis(hit.response, legalNow);
    if (analysis) return analysis;
  }
  if (hasEndgameThreshold(b)) {
    try {
      const raw = await requestEndgameForAi(b,side);
      const result = normalizeEndgameAnalysis(raw);
      if (result && !result._missing.length) return result;
    } catch(error) { throw new Error(`Exact endgame WASM was unavailable on forced pass: ${error.message || error}`); }
    throw new Error('Exact endgame WASM failed to solve the forced-pass position');
  }
  const result = await requestFallbackResult(moves, b, side, {
    resolveOnUpdate: true, minDepth: fallbackCandidateMinDepth()
  });
  return { ...result, cache: 'Fallback' };
}

function fallbackCandidateMinDepth() {
  if (searchEffort === 'fast') return 3;
  if (searchEffort === 'deep') return 5;
  return 4;
}

// Before 18 discs, the AI can reuse a live Egaroucid WASM analysis worker,
// avoiding a second 48 MiB instance with its own 24 MiB TT.
function waitForLiveEgaroucidAi(key, minDepth, requireExact) {
  if (aiKernel !== 'egaroucid' || !analysisEnabled || analysisPaused ||
      activeFallbackKey !== key || !fallbackWorker) return null;
  const acceptable = (result) => result?.wasmEgaroucid &&
    result.bestMove !== null && result.bestMove !== undefined &&
    (requireExact ? result.exact : Number(result.depth || 0) >= minDepth);
  if (latestAnalysisKey === key && acceptable(latestAnalysis)) return Promise.resolve(latestAnalysis);
  return new Promise((resolve, reject) => {
    const waiter = { key, acceptable, resolve, reject };
    egaroucidAiWaiters.add(waiter);
  });
}
function deliverEgaroucidAiUpdate(key, result) {
  for (const waiter of [...egaroucidAiWaiters]) {
    if (waiter.key === key && waiter.acceptable(result)) {
      egaroucidAiWaiters.delete(waiter);
      waiter.resolve(result);
    }
  }
}
function cancelEgaroucidAiWaiters() {
  for (const waiter of egaroucidAiWaiters) waiter.reject(new Error('Egaroucid AI search canceled'));
  egaroucidAiWaiters.clear();
}

function requestFallbackResult(moves, b, side = sideToMove(moves), options = {}) {
  return new Promise((resolve, reject) => {
    let worker;
    const key = `fallback-once-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const token = `fallback-once-${++fallbackToken}`;
    let cancel = null;
    const cleanup = () => {
      if (activeAiWasmCancel === cancel) activeAiWasmCancel = null;
      if (worker) {
        try { worker.postMessage({ type: 'stop' }); } catch {}
        worker.terminate();
      }
      worker = null;
    };
    if (options.aiRequest) {
      cancel = () => { cleanup(); reject(new Error('AI WASM search canceled after board change')); };
      activeAiWasmCancel = cancel;
    }
    try {
      worker = new Worker(selectedSearchWorkerPath());
      worker.onmessage = (event) => {
        const payload = event.data || {};
        if (payload.key !== key || payload.token !== token) return;
        if (payload.type === 'error') {
          cleanup();
          reject(new Error(payload.message || searchWorkerFailureMessage()));
          return;
        }
        if (payload.result && (payload.type === 'done' || (options.resolveOnUpdate && payload.type === 'update' && Number(payload.result.depth || 0) >= Number(options.minDepth || 1)))) {
          cleanup();
          resolve(payload.result);
        }
      };
      worker.onerror = (error) => {
        cleanup();
        reject(error instanceof Error ? error : new Error(searchWorkerFailureMessage()));
      };
      worker.postMessage({ type: 'start', key, token, board: b.slice(0, CELL_COUNT), side, effort: searchEffort });
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

function startOracleMixedFallbackAnalysis(moves, b, analysisKey, overlay) {
  stopFallbackSearch(true);
  activeFallbackKey = analysisKey;
  const token = `fallback-mixed-${++fallbackToken}`;
  elements.analysisState.textContent = 'Searching';
  updateAnalysisMetrics({ ...(latestAnalysis || {}), source: `${overlay.source} / ${currentKernel().searchSource}`, cache: 'Mixed' });

  try {
    fallbackWorker = new Worker(selectedSearchWorkerPath());
    fallbackWorker.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.key !== analysisKey || payload.token !== token || pendingAnalysisKey !== analysisKey) return;
      if (payload.type === 'error') {
        console.warn(payload.message);
        elements.analysisState.textContent = 'Error';
        stopFallbackSearch(true);
        return;
      }
      if (!payload.result) return;
      const merged = mergeOracleWithFallback(payload.result, overlay);
      latestAnalysis = merged;
      lastFallbackResult = merged;
      latestAnalysisKey = analysisKey;
      deliverEgaroucidAiUpdate(analysisKey, merged);
      renderStatus();
      renderAnalysis(merged);
      renderBoard();
      if (payload.type === 'done') {
        elements.analysisState.textContent = merged.terminal ? 'Game over' : 'Ready';
        stopFallbackSearch(true);
      }
    };
    fallbackWorker.onerror = (workerError) => {
      console.error(workerError);
      elements.analysisState.textContent = 'Error';
      stopFallbackSearch(true);
    };
    fallbackWorker.postMessage({ type: 'start', key: analysisKey, token, board: b.slice(0, CELL_COUNT), side: overlay.side, effort: searchEffort, uiCalibration: true });
  } catch (workerError) {
    console.error(workerError);
    elements.analysisState.textContent = 'Error';
  }
}

function startFallbackAnalysis(moves, b, analysisKey, error) {
  stopFallbackSearch();
  latestAnalysis = null;
  latestAnalysisKey = analysisKey;
  activeFallbackKey = analysisKey;
  const token = `fallback-${++fallbackToken}`;
  elements.analysisState.textContent = currentKernel().fallbackState;
  renderAnalysisPlaceholder(currentKernel().fallbackSearchingTitle, currentKernel().fallbackSearchingSub);
  updateAnalysisMetrics({ source: currentKernel().searchSource, cache: 'Fallback', depth: 0, nodes: 0 });

  try {
    fallbackWorker = new Worker(selectedSearchWorkerPath());
    fallbackWorker.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.key !== analysisKey || payload.token !== token || pendingAnalysisKey !== analysisKey) return;
      if (payload.type === 'error') {
        console.warn(payload.message);
        elements.analysisState.textContent = 'Error';
        renderAnalysisPlaceholder(currentKernel().fallbackErrorTitle, payload.message || currentKernel().fallbackErrorSub);
        stopFallbackSearch(true);
        return;
      }
      const result = payload.result;
      if (!result) return;
      latestAnalysis = result;
      lastFallbackResult = result;
      latestAnalysisKey = analysisKey;
      deliverEgaroucidAiUpdate(analysisKey, result);
      renderStatus();
      renderAnalysis(result);
      renderBoard();
      if (payload.type === 'done') {
        elements.analysisState.textContent = result.terminal ? 'Game over' : 'Solved';
        stopFallbackSearch(true);
      }
    };
    fallbackWorker.onerror = (workerError) => {
      console.error(workerError);
      elements.analysisState.textContent = 'Error';
      renderAnalysisPlaceholder(currentKernel().fallbackErrorTitle, currentKernel().fallbackErrorSub);
      stopFallbackSearch(true);
    };
    fallbackWorker.postMessage({ type: 'start', key: analysisKey, token, board: b.slice(0, CELL_COUNT), side: sideToMove(moves), effort: searchEffort, uiCalibration: true });
  } catch (workerError) {
    console.error(workerError || error);
    elements.analysisState.textContent = 'Error';
    renderAnalysisPlaceholder(currentKernel().fallbackErrorTitle, currentKernel().fallbackStartErrorSub);
  }
}

function stopFallbackSearch(keepResult = false) {
  cancelEgaroucidAiWaiters();
  if (fallbackWorker) {
    try { fallbackWorker.postMessage({ type: 'stop' }); } catch {}
    fallbackWorker.terminate();
  }
  fallbackWorker = null;
  activeFallbackKey = '';
  if (!keepResult) lastFallbackResult = null;
}

function updateAnalysisMetrics(result = {}) {
  if (elements.aiEngine) elements.aiEngine.textContent = result.endgameWasm ? endgameSource() : (result.cache === 'Fallback' ? currentKernel().fallbackEngine : currentKernel().baseEngine);
  elements.metricSource.textContent = result.source || '—';
  const depth = result.searchingDepth && !result.exact ? `${result.depth || 0}→${result.searchingDepth}` : (result.depth ?? 'Oracle');
  elements.metricPass.textContent = depth;
  elements.metricCache.textContent = result.nodes ? formatNodes(result.nodes) : (result.cache || '—');
  if (elements.metricFinals) elements.metricFinals.textContent = result.uiCalibrationTerminalSamples !== undefined ? formatNodes(result.uiCalibrationTerminalSamples) : '—';
}

function renderAnalysis(result) {
  const displayEval = displayEvalForAnalysis(result);
  if (result.endgameWasm && !result.exact && result.bestMove == null) {
    // Partial exact per-square values do not prove an overall best score.
    // Show no made-up headline evaluation while any candidate is unsolved.
    elements.aiEval.textContent = '…';
    elements.aiMarker.dataset.score = '…';
    if (elements.mobileEvalLabel) elements.mobileEvalLabel.textContent = '…';
    if (elements.mobileEvalMarker) elements.mobileEvalMarker.dataset.score = '…';
  } else updateEval(displayEval.score, displayEval.exact);
  elements.analysisState.textContent = result.terminal ? 'Game over' : ((result.endgameWasm || result.wasmEgaroucid || result.cache === 'Fallback') && !result.exact ? 'Searching' : 'Ready');
  updateAnalysisMetrics(result);
  elements.analysisPanel.classList.toggle('analysis-disabled', !analysisEnabled);
  if (result.terminal) return renderAnalysisPlaceholder('Game over', `Final score from White perspective: ${formatScore(displayEval.score, true)}.`);
  if (!result.lines.length) return renderAnalysisPlaceholder('No legal moves', 'The only legal choice is pass.');
  elements.analysisLines.innerHTML = '';
  result.lines.forEach((line, index) => {
    const row = document.createElement('button');
    row.type = 'button';
    const noBest = result.bestMove === null || result.bestMove === undefined;
    const isBest = line.move === result.bestMove || (noBest && index === 0 && !result.endgameWasm);
    row.className = `analysis-line${isBest ? ' best' : ''}${line.move === PASS ? ' pass-line' : ''}`;
    const lineExact = Boolean(line.exact || result.exact || result.terminal);
    const predicted = !lineExact && Number.isFinite(Number(line.predictedFinalWhite))
      ? `<span class="analysis-final" title="Cumulative terminal prediction from ${formatNodes(line.predictedFinalSamples || 0)} finals">Finals ${formatScore(line.predictedFinalWhite, false)} · ${formatNodes(line.predictedFinalSamples || 0)}</span>`
      : '';
    row.innerHTML = `
      <span class="analysis-rank">${index + 1}</span>
      <span class="analysis-move">
        <span class="analysis-move-row"><strong>${coordOf(line.move)}</strong><span class="analysis-score">${line.pending ? '…' : formatScore(line.evalWhite, lineExact)}</span>${predicted}<span class="move-tag">${escapeHtml(line.source || result.source)}</span></span>
        <span class="analysis-pv">${formatPv(line.pv)}</span>
      </span>
      <span class="analysis-play">Play</span>`;
    row.addEventListener('click', () => playMove(line.move, line.move === PASS ? 'pass' : 'ai'));
    elements.analysisLines.appendChild(row);
  });
}


function bestLineForAnalysis(result = {}) {
  const lines = Array.isArray(result.lines) ? result.lines : [];
  if (!lines.length) return null;
  if (result.bestMove !== null && result.bestMove !== undefined) {
    const byMove = lines.find((line) => line.move === result.bestMove);
    if (byMove) return byMove;
  }
  return lines[0];
}

function displayEvalForAnalysis(result = {}) {
  const bestLine = bestLineForAnalysis(result);
  const exact = Boolean(result.terminal || result.exact || result.evalExact || bestLine?.terminal || bestLine?.exact);
  const exactCandidate = exact && Number.isFinite(Number(bestLine?.evalWhite)) ? Number(bestLine.evalWhite) : Number(result.evalWhite);
  if (exact && Number.isFinite(exactCandidate)) return { score: exactCandidate, exact: true };
  if (bestLine && Number.isFinite(Number(bestLine.predictedFinalWhite))) {
    return { score: Number(bestLine.predictedFinalWhite), exact: false };
  }
  const raw = Number(bestLine?.evalWhite ?? result.evalWhite ?? 0);
  return { score: Number.isFinite(raw) ? raw : 0, exact: false };
}

function updateEval(score, exact = false) {
  const percent = ((clamp(score, -36, 36) + 36) / 72) * 100;
  const label = formatScore(score, exact);
  elements.aiEval.textContent = label;
  elements.aiBalance.style.width = `${percent}%`;
  elements.aiMarker.style.left = `${percent}%`;
  elements.aiMarker.dataset.score = label;
  if (elements.mobileEvalFill) elements.mobileEvalFill.style.width = `${percent}%`;
  if (elements.mobileEvalMarker) elements.mobileEvalMarker.style.left = `${percent}%`;
  if (elements.mobileEvalLabel) elements.mobileEvalLabel.textContent = label;
}

function scheduleAiMove() {
  cancelAiTimer();
  if (isGameOver()) return;
  if (isHumanTurn()) return;
  aiBusy = true;
  updateModePill();
  const token = kernelAnalysisKey(getLineMoves());
  aiTimer = setTimeout(async () => {
    if (token !== kernelAnalysisKey(getLineMoves())) return;
    try {
      const move = await chooseAiMove();
      aiBusy = false;
      if (token === kernelAnalysisKey(getLineMoves())) playMove(move, 'ai');
    } catch (error) {
      aiBusy = false;
      updateModePill();
      if (token !== kernelAnalysisKey(getLineMoves())) return;
      console.error(error);
      // No non-WASM substitute: report a real engine failure rather than
      // silently restarting a stale or wrong-position search.
      if (aiKernel === 'egaroucid') {
        elements.analysisState.textContent = 'Error';
        return;
      }
      if (hasEndgameThreshold(board)) {
        // Never play a non-perfect fallback AI move at 18+ discs.
        elements.analysisState.textContent = 'Error';
        return;
      }
      try {
        const moves = getLineMoves();
        const fallback = await requestFallbackResult(moves, board, sideToMove(moves), { resolveOnUpdate: true, minDepth: fallbackCandidateMinDepth() });
        if (token === kernelAnalysisKey(getLineMoves()) && fallback.bestMove !== null && fallback.bestMove !== undefined) playMove(fallback.bestMove, fallback.bestMove === PASS ? 'pass' : 'ai');
      } catch (fallbackError) {
        console.error(fallbackError);
      }
    }
  }, aiDelay);
}

async function chooseAiMove() {
  const moves = getLineMoves();
  const side = sideToMove(moves);
  const legalNow = legalMovesFor(board, side);
  if (!legalNow.length) return PASS;
  try {
    // A hit in either actual opening database takes precedence over all search.
    // Black Oracle is checked first for BOTH Black and White turns.
    const hit = await lookupOracleInOrder(moves);
    if (hit) {
      const result = hit.source === 'black'
        ? (side === 'black' ? blackOracleBlackAnalysis(hit.response, legalNow) : blackOracleWhiteAnalysis(hit.response, legalNow))
        : whiteOracleBookAnalysis(hit.response, legalNow);
      if (result && legalNow.includes(result.bestMove)) return result.bestMove;
    }
    // At 18 or more discs, ALWAYS use the existing, separate, exact endgame
    // WASM, including with Egaroucid Oracle selected. No heuristic score is
    // accepted as proof; all legal moves must finish solving before AI acts.
    if (hasEndgameThreshold(board)) {
      const raw = await requestEndgameForAi(board, side, { aiRequest: true });
      const solved = normalizeEndgameAnalysis(raw);
      if (solved?.exact && !solved._missing.length && legalNow.includes(solved.bestMove)) return solved.bestMove;
      throw new Error('The 18+ disc WASM solution was not fully exact');
    }
    // Only before 18 discs: use the C/WASM Egaroucid-inspired evaluator.
    // It deepens without a deadline while the board remains unchanged; AI
    // may play after one complete depth >=7 iteration, not a partial depth.
    if (aiKernel === 'egaroucid') {
      const requiredDepth = Math.min(7, 36 - countsOf(board).black - countsOf(board).white);
      const reuse = waitForLiveEgaroucidAi(kernelAnalysisKey(moves), requiredDepth, false);
      const result = await (reuse || requestFallbackResult(moves, board, side,
        { resolveOnUpdate: true, minDepth: requiredDepth, aiRequest: true }));
      if (legalNow.includes(result.bestMove)) return result.bestMove;
      throw new Error('Egaroucid WASM did not produce a legal best move');
    }
    // Preserve the original AI search behavior outside both databases.
    const response = await activeOracle().request(moves, { timeoutMs: ORION_PLAY_TIMEOUT_MS });
    if (side === 'white') {
      if (legalNow.includes(response.move)) return response.move;
    } else {
      const legalCandidates = (response.moves || []).filter(item => legalNow.includes(item.move));
      if (legalCandidates.length) {
        legalCandidates.sort((a, b) => Number(b.blackValue || 0) - Number(a.blackValue || 0));
        return legalCandidates[0].move;
      }
    }
    throw new Error('No legal candidate from original AI');
  } catch (error) {
    if (aiKernel === 'egaroucid' || hasEndgameThreshold(board)) throw error;
    console.warn('AI move fell back to JS search', error);
    const currentKey = kernelAnalysisKey(moves);
    if (latestAnalysisKey === currentKey && latestAnalysis?.bestMove !== null && latestAnalysis?.bestMove !== undefined && legalNow.includes(latestAnalysis.bestMove)) {
      return latestAnalysis.bestMove;
    }
    const fallback = await requestFallbackResult(moves, board, side, {
      resolveOnUpdate: true, minDepth: fallbackCandidateMinDepth()
    });
    return fallback.bestMove ?? legalNow[0];
  }
}

function isHumanTurn() {
  if (gameMode === 'local') return true;
  if (gameMode === 'ai-ai') return false;
  return sideToMove() === humanSide;
}

function cancelAiTimer() {
  cancelEgaroucidAiWaiters();
  if (activeAiWasmCancel) activeAiWasmCancel();
  if (aiTimer) clearTimeout(aiTimer);
  aiTimer = null;
  aiBusy = false;
}

function updateUI(options = {}) {
  renderStatus();
  renderBoard();
  renderMoves();
  saveState();
  if (options.requestAnalysis && analysisEnabled) requestAnalysis();
  if (options.scheduleAi) scheduleAiMove();
}

function newGame() {
  cancelAiTimer();
  nextNodeId = 1;
  root = new TreeNode({ id: 'root' });
  current = root;
  board = createInitialBoard();
  lastMove = null;
  latestAnalysis = null;
  latestAnalysisKey = '';
  pendingAnalysisKey = '';
  stopFallbackSearch();
  stopEndgameSearch();
  updateEval(0);
  updateUI({ requestAnalysis: true, scheduleAi: true });
  toast('Started a new MiniOthello game.');
}

function makePositionKey(moves, b) {
  const rows = [];
  for (let y = 0; y < BOARD_SIZE; y++) {
    let row = '';
    let empties = 0;
    for (let x = 0; x < BOARD_SIZE; x++) {
      const cell = getCell(b, x, y);
      if (!cell) empties++;
      else {
        if (empties) row += String(empties), empties = 0;
        row += cell === 'black' ? 'B' : 'W';
      }
    }
    if (empties) row += String(empties);
    rows.push(row);
  }
  return `${rows.join('/')} ${sideToMove(moves)[0]} moves:${moves.map(coordOf).join(' ') || '-'}`;
}

function serializeNode(node) {
  return { id: node.id, move: node.move, source: node.source, children: node.children.map(serializeNode) };
}
function deserializeNode(raw, parent = null) {
  const node = new TreeNode({ id: raw.id || (parent ? `n${nextNodeId++}` : 'root'), parent, move: raw.move ?? null, source: raw.source || 'local' });
  node.children = (raw.children || []).map((child) => deserializeNode(child, node));
  return node;
}
function findNodeByPath(node, moves, ply = 0) {
  if (ply === moves.length) return node;
  const child = node.children.find((n) => n.move === moves[ply]);
  return child ? findNodeByPath(child, moves, ply + 1) : node;
}
function saveState() {
  try {
    localStorage.setItem(STORAGE, JSON.stringify({ tree: serializeNode(root), path: getLineMoves(), nextNodeId, flipped, mirrored, gameMode, aiKernel, analysisEnabled, searchEffort }));
  } catch {}
}
function loadState() {
  try {
    const raw = JSON.parse(localStorage.getItem(STORAGE) || 'null');
    if (!raw?.tree) return;
    root = deserializeNode(raw.tree);
    nextNodeId = Math.max(Number(raw.nextNodeId) || nextNodeId, countNodes(root) + 1);
    current = findNodeByPath(root, raw.path || []);
    flipped = Boolean(raw.flipped);
    mirrored = Boolean(raw.mirrored);
    gameMode = raw.gameMode || 'local';
    aiKernel = normalizeAiKernel(localStorage.getItem(AI_KERNEL_STORAGE) || raw.aiKernel || aiKernel || 'egaroucid');
    analysisEnabled = raw.analysisEnabled !== false;
    searchEffort = raw.searchEffort || searchEffort;
    board = rebuildBoard();
    const moves = getLineMoves();
    lastMove = moves.length ? moves[moves.length - 1] : null;
  } catch {}
}
function countNodes(node) { return 1 + node.children.reduce((sum, child) => sum + countNodes(child), 0); }

function copyText(text, message) {
  navigator.clipboard?.writeText(text).then(() => toast(message)).catch(() => {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
    toast(message);
  });
}

function toast(text) {
  elements.toast.textContent = text;
  elements.toast.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { elements.toast.hidden = true; }, 2200);
}
function capitalize(text) { return text ? text[0].toUpperCase() + text.slice(1) : text; }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function formatScore(value, exact = false) {
  const n = Number(value);
  if (!Number.isFinite(n)) return '0.00';
  if (exact) {
    const rounded = Math.round(n);
    return `${rounded > 0 ? '+' : ''}${rounded}`;
  }
  return `${n > 0 ? '+' : ''}${n.toFixed(2)}`;
}
function formatNodes(value) {
  if (value === '—') return '—';
  const n = Number(value) || 0;
  if (n >= 1000000) return `${(n / 1000000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(Math.round(n));
}
function formatPv(pv = []) { return pv.length ? pv.map(coordOf).join(' ') : '—'; }
function escapeHtml(text) { return String(text).replace(/[&<>"]/g, (ch) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[ch])); }

function wireEvents() {
  elements.undo.addEventListener('click', () => current.parent && navigate(current.parent));
  elements.redo.addEventListener('click', () => current.children[0] && navigate(current.children[0]));
  elements.flip.addEventListener('click', () => { flipped = !flipped; updateUI({ requestAnalysis: false }); });
  elements.mirror.addEventListener('click', () => { mirrored = !mirrored; updateUI({ requestAnalysis: false }); });
  elements.pass.addEventListener('click', () => playMove(PASS));
  elements.newGame.addEventListener('click', newGame);
  elements.analysis.addEventListener('click', () => {
    analysisEnabled = !analysisEnabled;
    analysisPaused = false;
    elements.analysis.setAttribute('aria-pressed', String(analysisEnabled));
    elements.analysis.classList.toggle('active', analysisEnabled);
    if (analysisEnabled) requestAnalysis();
    else {
      stopFallbackSearch();
      stopEndgameSearch();
      elements.analysisState.textContent = 'Idle';
      elements.analysisPanel.classList.add('analysis-disabled');
      latestAnalysis = null;
      latestAnalysisKey = '';
      renderAnalysisPlaceholder('Analysis is off', 'Turn it on to show candidate scores on the board.');
      renderBoard();
    }
    saveState();
  });
  elements.pauseAnalysis?.addEventListener('click', () => {
    analysisPaused = true;
    stopFallbackSearch(true);
    stopEndgameSearch();
    elements.analysisState.textContent = 'Paused';
    elements.pauseAnalysis.hidden = true;
    elements.resumeAnalysis.hidden = false;
  });
  elements.resumeAnalysis?.addEventListener('click', () => {
    analysisPaused = false;
    elements.pauseAnalysis.hidden = false;
    elements.resumeAnalysis.hidden = true;
    requestAnalysis();
  });
  elements.boardStyle.value = boardStyle;
  elements.boardStyle.addEventListener('change', () => {
    boardStyle = elements.boardStyle.value;
    localStorage.setItem(BOARD_STYLE_STORAGE, boardStyle);
    renderBoard();
  });
  elements.gameMode.value = gameMode;
  elements.gameMode.addEventListener('change', () => {
    gameMode = elements.gameMode.value;
    updateUI({ scheduleAi: true, requestAnalysis: true });
  });
  if (elements.aiKernel) {
    elements.aiKernel.value = aiKernel;
    elements.aiKernel.addEventListener('change', () => {
      aiKernel = normalizeAiKernel(elements.aiKernel.value);
      localStorage.setItem(AI_KERNEL_STORAGE, aiKernel);
      latestAnalysis = null;
      latestAnalysisKey = '';
      pendingAnalysisKey = '';
      stopFallbackSearch();
      cancelAiTimer();
      updateUI({ scheduleAi: true, requestAnalysis: true });
      toast(`${kernelName()} kernel selected.`);
    });
  }
  elements.humanSide.value = humanSide;
  elements.humanSide.addEventListener('change', () => {
    humanSide = elements.humanSide.value;
    localStorage.setItem(HUMAN_SIDE_STORAGE, humanSide);
    updateUI({ scheduleAi: true, requestAnalysis: true });
  });
  elements.aiDelay.value = String(aiDelay);
  elements.aiDelay.addEventListener('change', () => {
    aiDelay = Number(elements.aiDelay.value);
    localStorage.setItem(AI_DELAY_STORAGE, String(aiDelay));
  });
  elements.effort.value = searchEffort;
  elements.effort.addEventListener('change', () => {
    searchEffort = elements.effort.value;
    localStorage.setItem(EFFORT_STORAGE, searchEffort);
    stopFallbackSearch();
    requestAnalysis();
  });
  elements.copyLine.addEventListener('click', () => copyText(getLineMoves().map(coordOf).join(' ') || '-', 'Copied current line.'));
  elements.copyFen.addEventListener('click', () => copyText(elements.fenDisplay.textContent, 'Copied position key.'));
  window.addEventListener('keydown', (event) => {
    if (event.target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) return;
    if (event.key === 'ArrowLeft' && current.parent) navigate(current.parent);
    if (event.key === 'ArrowRight' && current.children[0]) navigate(current.children[0]);
    if (event.key.toLowerCase() === 'p') playMove(PASS);
  });
}

loadState();
wireEvents();
elements.analysis.setAttribute('aria-pressed', String(analysisEnabled));
elements.analysis.classList.toggle('active', analysisEnabled);
if (elements.pauseAnalysis) elements.pauseAnalysis.disabled = false;
updateEval(countsOf(board).whiteDiff);
updateUI({ requestAnalysis: true, scheduleAi: true });
