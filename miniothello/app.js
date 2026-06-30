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
const CORNERS = new Set([0, 5, 30, 35]);
const EDGES = new Set(Array.from({ length: CELL_COUNT }, (_, i) => i).filter((i) => {
  const x = i % BOARD_SIZE;
  const y = Math.floor(i / BOARD_SIZE);
  return x === 0 || x === BOARD_SIZE - 1 || y === 0 || y === BOARD_SIZE - 1;
}));
const CORNER_ADJACENT = new Set([1, 6, 7, 4, 10, 11, 24, 25, 31, 28, 29, 34]);
const STORAGE = 'miniothello-v5-state';
const BOARD_STYLE_STORAGE = 'miniothello-board-style';
const AI_DELAY_STORAGE = 'miniothello-ai-delay';
const HUMAN_SIDE_STORAGE = 'miniothello-human-side';
const EFFORT_STORAGE = 'miniothello-v5-effort';

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
  aiDelay: $('#aiThinkTimeSelect'),
  effort: $('#effortSelect'),
  playEngineStatus: $('#playEngineStatus'),
  moveCount: $('#moveCount'),
  moveTree: $('#moveTree'),
  copyLine: $('#copyLineButton'),
  fenDisplay: $('#fenDisplay'),
  copyFen: $('#copyFenButton'),
  analysisPanel: $('#analysisPanel'),
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

class OrionOracle {
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
        console.error('Orion worker error', event);
        this.rejectAll(new Error('Orion worker failed'));
      });
    } catch (error) {
      this.available = false;
      setOptionalText(elements.libraryStatus, 'Fallback');
      addOptionalClass(elements.libraryStatus, 'error');
      console.error(error);
    }
  }

  request(moves) {
    const cleanMoves = moves.slice();
    const key = keyOf(cleanMoves);
    if (this.cache.has(key)) return Promise.resolve({ ...this.cache.get(key), cache: 'Hit' });
    if (!this.worker) return Promise.reject(new Error('Orion worker is not available'));
    return new Promise((resolve, reject) => {
      const queue = this.pending.get(key) || [];
      queue.push({ resolve, reject });
      this.pending.set(key, queue);
      this.worker.postMessage(cleanMoves);
    });
  }

  rejectAll(error) {
    for (const queue of this.pending.values()) queue.forEach(({ reject }) => reject(error));
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
let aiDelay = Number(localStorage.getItem(AI_DELAY_STORAGE) || '700');
let boardStyle = localStorage.getItem(BOARD_STYLE_STORAGE) || 'standard';
let searchEffort = localStorage.getItem(EFFORT_STORAGE) || 'balanced';
let flipped = false;
let mirrored = false;
let renderSeq = 0;
const orion = new OrionOracle();
let fallbackWorker = null;
let fallbackToken = 0;
let activeFallbackKey = '';
let lastFallbackResult = null;

function createInitialBoard() {
  const b = Array(CELL_COUNT).fill(null);
  setCell(b, 2, 2, 'white');
  setCell(b, 3, 2, 'black');
  setCell(b, 2, 3, 'black');
  setCell(b, 3, 3, 'white');
  return b;
}

function setCell(b, x, y, value) { b[y * BOARD_SIZE + x] = value; }
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
    return { type: 'black-moves', moves: movesOut, cache: 'Fresh', source: 'Orion search' };
  }
  return { type: 'white-move', move: Number(result), cache: 'Fresh', source: 'Orion search' };
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
function sourceForPosition(moves) { return moves.length <= 16 ? 'Orion DB/search' : 'Orion search'; }
function isDatabasePosition(moves) { return moves.length <= 16; }

function legalMovesFor(b, side) {
  const moves = [];
  for (let idx = 0; idx < CELL_COUNT; idx++) {
    if (b[idx] !== null) continue;
    if (flipsForMove(b, side, idx).length) moves.push(idx);
  }
  return moves;
}

function flipsForMove(b, side, idx) {
  if (idx === PASS || b[idx] !== null) return [];
  const x0 = idx % BOARD_SIZE;
  const y0 = Math.floor(idx / BOARD_SIZE);
  const enemy = opponent(side);
  const flips = [];
  for (const [dx, dy] of DIRS) {
    let x = x0 + dx;
    let y = y0 + dy;
    const line = [];
    while (inBounds(x, y) && getCell(b, x, y) === enemy) {
      line.push(y * BOARD_SIZE + x);
      x += dx;
      y += dy;
    }
    if (line.length && inBounds(x, y) && getCell(b, x, y) === side) flips.push(...line);
  }
  return flips;
}

function applyMoveToBoard(b, side, move) {
  const next = b.slice();
  if (move === PASS) return next;
  const flips = flipsForMove(b, side, move);
  next[move] = side;
  flips.forEach((idx) => { next[idx] = side; });
  return next;
}

function rebuildBoard(moves = getLineMoves()) {
  let b = createInitialBoard();
  for (let ply = 0; ply < moves.length; ply++) {
    b = applyMoveToBoard(b, ply % 2 === 0 ? 'black' : 'white', moves[ply]);
  }
  return b;
}

function countsOf(b = board) {
  let black = 0;
  let white = 0;
  for (const cell of b) {
    if (cell === 'black') black++;
    if (cell === 'white') white++;
  }
  return { black, white, empty: CELL_COUNT - black - white, whiteDiff: white - black, blackDiff: black - white };
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
  if (latestAnalysisKey !== keyOf(moves) || !latestAnalysis?.lines) return new Map();
  const map = new Map();
  latestAnalysis.lines.forEach((line, index) => map.set(line.move, { ...line, index, exact: Boolean(line.exact || latestAnalysis.exact || latestAnalysis.terminal) }));
  return map;
}

function renderBoard() {
  const seq = ++renderSeq;
  const moves = getLineMoves();
  const side = sideToMove(moves);
  const over = isGameOver(moves, board);
  const currentLegal = over ? [] : legalMovesFor(board, side);
  const candidateMap = candidateMapForCurrentPosition();
  const bestMove = latestAnalysisKey === keyOf(moves) ? latestAnalysis?.bestMove : null;
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
      if (!candidate) cell.classList.add('candidate-pending');
      if (candidate) cell.classList.add('scored-candidate');
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
    } else if (candidate) {
      const score = document.createElement('span');
      score.className = 'candidate-score';
      score.textContent = formatScore(candidate.evalWhite, candidate.exact);
      score.title = `White-relative score ${formatScore(candidate.evalWhite, candidate.exact)}`;
      cell.appendChild(score);
    }
    cell.addEventListener('click', () => {
      if (seq !== renderSeq) return;
      if (!canClick) return toast('Orion is thinking.');
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
  if (over) {
    const result = counts.whiteDiff > 0 ? 'White win' : counts.whiteDiff < 0 ? 'Black win' : 'Draw';
    elements.statusText.textContent = result;
    elements.turnToken.textContent = 'FINISHED';
    elements.turnToken.className = 'turn-token finished';
  } else {
    elements.statusText.textContent = `${capitalize(side)} move${legal.length === 0 ? ' · only pass' : ''}`;
    elements.turnToken.textContent = legal.length === 0 ? 'PASS' : side.toUpperCase();
    elements.turnToken.className = `turn-token ${side}`;
  }
  elements.metricTurn.textContent = over ? 'Game over' : capitalize(side);
  elements.metricLegal.textContent = over ? '0' : String(legal.length);
  elements.metricSource.textContent = latestAnalysis?.source || sourceForPosition(moves);
  elements.metricPass.textContent = latestAnalysis?.depth ?? '—';
  elements.metricCache.textContent = latestAnalysis?.nodes ? formatNodes(latestAnalysis.nodes) : (latestAnalysis?.cache || '—');
  elements.pass.disabled = over || legal.length > 0 || !isHumanTurn();
  elements.pass.classList.toggle('pass-ready', !elements.pass.disabled);
  elements.undo.disabled = !current.parent;
  elements.redo.disabled = current.children.length === 0;
  elements.moveCount.textContent = `${moves.length} ${moves.length === 1 ? 'ply' : 'plies'}`;
  elements.fenDisplay.textContent = makePositionKey(moves, board);
  updateModePill();
}

function updateModePill() {
  const labels = { local: 'Local two-player', 'human-ai': `Player vs Orion · ${capitalize(humanSide)}`, 'ai-ai': 'Orion self-play' };
  elements.modePill.innerHTML = `<i></i> ${labels[gameMode] || labels.local}`;
  elements.humanSideField.style.display = gameMode === 'human-ai' ? '' : 'none';
  elements.playEngineStatus.textContent = aiBusy ? 'Orion thinking' : 'Ready';
  elements.playEngineStatus.className = aiBusy ? 'play-engine-status thinking' : 'play-engine-status';
}

function renderMoves() {
  const moves = getLineMoves();
  if (!moves.length) {
    elements.moveTree.className = 'move-tree empty-state';
    elements.moveTree.innerHTML = '<div class="empty-illustration">●</div><strong>No moves yet</strong><span>Play a move or choose an Orion candidate.</span>';
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
  const moves = getLineMoves();
  const analysisKey = keyOf(moves);
  pendingAnalysisKey = analysisKey;
  latestAnalysisKey = '';
  elements.analysisState.textContent = 'Thinking';
  elements.analysisPanel.classList.remove('analysis-disabled');
  renderAnalysisPlaceholder('Orion is thinking…', 'Candidate scores will appear on the board.');
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
  } catch (error) {
    console.warn('Orion unavailable for this position; starting JS fallback search.', error);
    if (pendingAnalysisKey !== analysisKey) return;
    startFallbackAnalysis(moves, board, analysisKey, error);
  }
}

async function analyzePosition(moves, b) {
  const side = sideToMove(moves);
  const legalNow = legalMovesFor(b, side);
  const over = isGameOver(moves, b);
  if (over) {
    const c = countsOf(b);
    return { side, lines: [], evalWhite: c.whiteDiff, bestMove: null, source: 'terminal', cache: 'Fresh', terminal: true, exact: true };
  }
  if (legalNow.length === 0) {
    const afterPass = moves.concat(PASS);
    const downstream = await analyzePositionShallow(afterPass, b);
    return {
      side,
      source: 'forced pass',
      cache: downstream.cache || 'Fresh',
      bestMove: PASS,
      evalWhite: downstream.evalWhite,
      lines: [{ move: PASS, evalWhite: downstream.evalWhite, source: 'pass', pv: [PASS].concat(downstream.bestMove !== null && downstream.bestMove !== undefined ? [downstream.bestMove] : []), exact: Boolean(downstream.exact) }],
      exact: Boolean(downstream.exact),
      terminal: Boolean(downstream.terminal)
    };
  }

  try {
    if (side === 'black') {
      const response = await orion.request(moves);
      const oracleMoves = response.moves || [];
      const filtered = oracleMoves.filter((item) => legalNow.includes(item.move));
      if (!filtered.length) throw new Error('Orion returned no legal black candidates');
      const lines = filtered
        .map((item) => ({
          move: item.move,
          evalWhite: -Number(item.blackValue || 0),
          blackValue: Number(item.blackValue || 0),
          source: sourceForPosition(moves),
          pv: [item.move],
          exact: isDatabasePosition(moves)
        }))
        .sort((a, z) => a.evalWhite - z.evalWhite);
      const best = lines[0];
      return { side, lines, evalWhite: best.evalWhite, bestMove: best.move, source: sourceForPosition(moves), cache: response.cache, oracleBestMove: best.move, depth: 'Oracle', nodes: '—', exact: isDatabasePosition(moves) };
    }

    const lineResults = [];
    for (const move of legalNow) {
      const score = await evaluateWhiteCandidateSafely(moves, b, move);
      lineResults.push({
        move,
        evalWhite: score.evalWhite,
        source: score.source,
        pv: [move].concat(score.reply !== null && score.reply !== undefined ? [score.reply] : []),
        cache: score.cache,
        exact: Boolean(score.exact),
        terminal: Boolean(score.terminal)
      });
    }
    const lines = lineResults.sort((a, z) => z.evalWhite - a.evalWhite);
    const best = lines[0];
    if (best) best.oracleBest = best.cache !== 'Fallback';
    return { side, lines, evalWhite: best?.evalWhite ?? countsOf(b).whiteDiff, bestMove: best?.move ?? null, source: sourceForPosition(moves), cache: lines.some((line) => line.cache === 'Fallback') ? 'Mixed' : 'Fresh', oracleBestMove: best?.move ?? null, depth: lines.every((line) => line.cache !== 'Fallback') ? 'Oracle' : 'Mixed', nodes: '—', exact: lines.length > 0 && lines.every((line) => line.exact) };
  } catch (error) {
    throw error;
  }
}

async function evaluateWhiteCandidateSafely(moves, b, move) {
  try {
    return await evaluateWhiteCandidateWithOracle(moves, b, move);
  } catch (error) {
    console.debug('White candidate oracle look-ahead unavailable; using JS fallback worker for candidate score.', move, error);
    return evaluateWhiteCandidateWithFallback(moves, b, move);
  }
}

async function evaluateWhiteCandidateWithOracle(moves, b, move) {
  const afterMoves = moves.concat(move);
  const afterBoard = applyMoveToBoard(b, 'white', move);
  const blackLegal = legalMovesFor(afterBoard, 'black');
  if (isGameOver(afterMoves, afterBoard)) return { evalWhite: countsOf(afterBoard).whiteDiff, reply: null, source: 'terminal', cache: 'Fresh', terminal: true, exact: true };
  if (!isDatabasePosition(afterMoves)) throw new Error('White candidate is outside the database');
  if (!blackLegal.length) {
    const downstream = await analyzePositionShallow(afterMoves.concat(PASS), afterBoard);
    return { evalWhite: downstream.evalWhite, reply: PASS, source: 'candidate search', cache: downstream.cache || 'Fresh', exact: Boolean(downstream.exact), terminal: Boolean(downstream.terminal) };
  }
  const blackReply = await orion.request(afterMoves);
  const replies = (blackReply.moves || []).filter((item) => blackLegal.includes(item.move));
  if (!replies.length) throw new Error('No legal black reply from Orion');
  replies.sort((a, z) => Number(z.blackValue || 0) - Number(a.blackValue || 0));
  const bestReply = replies[0];
  return { evalWhite: -Number(bestReply.blackValue || 0), reply: bestReply.move, source: 'candidate search', cache: blackReply.cache || 'Fresh', exact: isDatabasePosition(afterMoves) };
}

async function evaluateWhiteCandidateWithFallback(moves, b, move) {
  const afterMoves = moves.concat(move);
  const afterBoard = applyMoveToBoard(b, 'white', move);
  if (isGameOver(afterMoves, afterBoard)) {
    return { evalWhite: countsOf(afterBoard).whiteDiff, reply: null, source: 'terminal', cache: 'Fresh', terminal: true, exact: true };
  }
  const result = await requestFallbackResult(afterMoves, afterBoard, sideToMove(afterMoves), { resolveOnUpdate: true, minDepth: fallbackCandidateMinDepth() });
  return {
    evalWhite: result.evalWhite,
    reply: result.bestMove,
    source: 'candidate fallback',
    cache: 'Fallback',
    terminal: Boolean(result.terminal),
    exact: Boolean(result.exact)
  };
}

async function analyzePositionShallow(moves, b) {
  if (isGameOver(moves, b)) return { evalWhite: countsOf(b).whiteDiff, bestMove: null, cache: 'Fresh', terminal: true, exact: true };
  const side = sideToMove(moves);
  const legalNow = legalMovesFor(b, side);
  if (!legalNow.length) return analyzePositionShallow(moves.concat(PASS), b);
  try {
    if (side === 'black') {
      const r = await orion.request(moves);
      const lines = (r.moves || []).filter((item) => legalNow.includes(item.move));
      const best = lines.sort((a, z) => Number(z.blackValue || 0) - Number(a.blackValue || 0))[0];
      if (!best) throw new Error('No black reply');
      return { evalWhite: -Number(best.blackValue || 0), bestMove: best.move, cache: r.cache, exact: isDatabasePosition(moves) };
    }
    const candidates = [];
    for (const move of legalNow) {
      const score = await evaluateWhiteCandidateSafely(moves, b, move);
      candidates.push({ move, ...score });
    }
    candidates.sort((a, z) => z.evalWhite - a.evalWhite);
    const best = candidates[0];
    if (!best) throw new Error('No white reply');
    return { evalWhite: best.evalWhite, bestMove: best.move, cache: best.cache, exact: Boolean(best.exact), terminal: Boolean(best.terminal) };
  } catch (error) {
    throw error;
  }
}



function fallbackCandidateMinDepth() {
  if (searchEffort === 'fast') return 3;
  if (searchEffort === 'deep') return 5;
  return 4;
}

function requestFallbackResult(moves, b, side = sideToMove(moves), options = {}) {
  return new Promise((resolve, reject) => {
    let worker;
    const key = `fallback-once-${Date.now()}-${Math.random().toString(16).slice(2)}`;
    const token = `fallback-once-${++fallbackToken}`;
    const cleanup = () => {
      if (worker) {
        try { worker.postMessage({ type: 'stop' }); } catch {}
        worker.terminate();
      }
      worker = null;
    };
    try {
      worker = new Worker('assets/fallback-worker.js');
      worker.onmessage = (event) => {
        const payload = event.data || {};
        if (payload.key !== key || payload.token !== token) return;
        if (payload.type === 'error') {
          cleanup();
          reject(new Error(payload.message || 'JS fallback worker failed'));
          return;
        }
        if (payload.result && (payload.type === 'done' || (options.resolveOnUpdate && payload.type === 'update' && Number(payload.result.depth || 0) >= Number(options.minDepth || 1)))) {
          cleanup();
          resolve(payload.result);
        }
      };
      worker.onerror = (error) => {
        cleanup();
        reject(error instanceof Error ? error : new Error('JS fallback worker failed'));
      };
      worker.postMessage({ type: 'start', key, token, board: b, side, effort: searchEffort });
    } catch (error) {
      cleanup();
      reject(error);
    }
  });
}

function startFallbackAnalysis(moves, b, analysisKey, error) {
  stopFallbackSearch();
  latestAnalysis = null;
  latestAnalysisKey = analysisKey;
  activeFallbackKey = analysisKey;
  const token = `fallback-${++fallbackToken}`;
  elements.analysisState.textContent = 'Fallback';
  renderAnalysisPlaceholder('JS fallback is searching…', 'Depth, nodes and candidate scores update every 500ms while analysis remains on.');
  updateAnalysisMetrics({ source: 'JS fallback iterative', cache: 'Fallback', depth: 0, nodes: 0 });

  try {
    fallbackWorker = new Worker('assets/fallback-worker.js');
    fallbackWorker.onmessage = (event) => {
      const payload = event.data || {};
      if (payload.key !== analysisKey || payload.token !== token || pendingAnalysisKey !== analysisKey) return;
      if (payload.type === 'error') {
        console.warn(payload.message);
        elements.analysisState.textContent = 'Error';
        renderAnalysisPlaceholder('JS fallback failed.', payload.message || 'The fallback worker could not analyze this position.');
        stopFallbackSearch(true);
        return;
      }
      const result = payload.result;
      if (!result) return;
      latestAnalysis = result;
      lastFallbackResult = result;
      latestAnalysisKey = analysisKey;
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
      renderAnalysisPlaceholder('JS fallback failed.', 'The fallback worker could not analyze this position.');
      stopFallbackSearch(true);
    };
    fallbackWorker.postMessage({ type: 'start', key: analysisKey, token, board: b, side: sideToMove(moves), effort: searchEffort });
  } catch (workerError) {
    console.error(workerError || error);
    elements.analysisState.textContent = 'Error';
    renderAnalysisPlaceholder('JS fallback failed.', 'The fallback worker could not be started.');
  }
}

function stopFallbackSearch(keepResult = false) {
  if (fallbackWorker) {
    try { fallbackWorker.postMessage({ type: 'stop' }); } catch {}
    fallbackWorker.terminate();
  }
  fallbackWorker = null;
  activeFallbackKey = '';
  if (!keepResult) lastFallbackResult = null;
}

function updateAnalysisMetrics(result = {}) {
  if (elements.aiEngine) elements.aiEngine.textContent = result.cache === 'Fallback' ? 'JS fallback v5' : 'Orion oracle v5';
  elements.metricSource.textContent = result.source || '—';
  const depth = result.searchingDepth && !result.exact ? `${result.depth || 0}→${result.searchingDepth}` : (result.depth ?? 'Oracle');
  elements.metricPass.textContent = depth;
  elements.metricCache.textContent = result.nodes ? formatNodes(result.nodes) : (result.cache || '—');
}

function renderAnalysis(result) {
  const score = Number(result.evalWhite || 0);
  const exactScore = Boolean(result.terminal || result.exact);
  updateEval(score, exactScore);
  elements.analysisState.textContent = result.terminal ? 'Game over' : (result.cache === 'Fallback' && !result.exact ? 'Searching' : 'Ready');
  updateAnalysisMetrics(result);
  elements.analysisPanel.classList.toggle('analysis-disabled', !analysisEnabled);
  if (result.terminal) return renderAnalysisPlaceholder('Game over', `Final score from White perspective: ${formatScore(score, true)}.`);
  if (!result.lines.length) return renderAnalysisPlaceholder('No legal moves', 'The only legal choice is pass.');
  elements.analysisLines.innerHTML = '';
  result.lines.forEach((line, index) => {
    const row = document.createElement('button');
    row.type = 'button';
    const noBest = result.bestMove === null || result.bestMove === undefined;
    const isBest = line.move === result.bestMove || (noBest && index === 0);
    row.className = `analysis-line${isBest ? ' best' : ''}${line.move === PASS ? ' pass-line' : ''}`;
    row.innerHTML = `
      <span class="analysis-rank">${index + 1}</span>
      <span class="analysis-move">
        <span class="analysis-move-row"><strong>${coordOf(line.move)}</strong><span class="analysis-score">${formatScore(line.evalWhite, line.exact || result.exact || result.terminal)}</span><span class="move-tag">${escapeHtml(line.source || result.source)}</span></span>
        <span class="analysis-pv">${formatPv(line.pv)}</span>
      </span>
      <span class="analysis-play">Play</span>`;
    row.addEventListener('click', () => playMove(line.move, line.move === PASS ? 'pass' : 'ai'));
    elements.analysisLines.appendChild(row);
  });
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
  const token = keyOf(getLineMoves());
  aiTimer = setTimeout(async () => {
    if (token !== keyOf(getLineMoves())) return;
    try {
      const move = await chooseAiMove();
      aiBusy = false;
      if (token === keyOf(getLineMoves())) playMove(move, 'ai');
    } catch (error) {
      aiBusy = false;
      updateModePill();
      console.error(error);
      try {
        const moves = getLineMoves();
        const fallback = await requestFallbackResult(moves, board, sideToMove(moves));
        if (token === keyOf(getLineMoves()) && fallback.bestMove !== null && fallback.bestMove !== undefined) playMove(fallback.bestMove, fallback.bestMove === PASS ? 'pass' : 'ai');
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
    if (side === 'black') {
      const response = await orion.request(moves);
      const legalCandidates = (response.moves || []).filter((item) => legalNow.includes(item.move));
      if (legalCandidates.length) {
        legalCandidates.sort((a, z) => Number(z.blackValue || 0) - Number(a.blackValue || 0));
        return legalCandidates[0].move;
      }
      throw new Error('No legal black candidate');
    }
    const response = await orion.request(moves);
    if (legalNow.includes(response.move)) return response.move;
    throw new Error('Illegal white candidate');
  } catch (error) {
    console.warn('AI move fell back to JS search', error);
    const currentKey = keyOf(moves);
    if (latestAnalysisKey === currentKey && latestAnalysis?.cache === 'Fallback' && latestAnalysis.bestMove !== null && latestAnalysis.bestMove !== undefined) {
      return latestAnalysis.bestMove;
    }
    const fallback = await requestFallbackResult(moves, board, side);
    return fallback.bestMove ?? legalNow[0];
  }
}

function isHumanTurn() {
  if (gameMode === 'local') return true;
  if (gameMode === 'ai-ai') return false;
  return sideToMove() === humanSide;
}

function cancelAiTimer() {
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
    localStorage.setItem(STORAGE, JSON.stringify({ tree: serializeNode(root), path: getLineMoves(), nextNodeId, flipped, mirrored, gameMode, analysisEnabled, searchEffort }));
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
