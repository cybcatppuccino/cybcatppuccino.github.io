import {
  COLORS,
  PIECE_NAMES,
  TYPES,
  COORD_SYSTEMS,
  parseStandardCoord
} from './js/core/constants.js';
import { GameTree } from './js/core/game-tree.js';
import { Position, validateEditedPosition } from './js/core/position.js';
import { StudyLibrary, exportCurrentLineMovetext, exportGameTreePGN, parsePGN } from './js/core/pgn.js';
import { Rules, gameStatus, legalMoves } from './js/core/rules.js';
import { START_LAYOUTS, createStartPosition } from './js/core/start-positions.js';
import { moveToSAN, moveToUci } from './js/core/notation.js';
import { AnalysisCache, buildAnalysisKey } from './js/engine/analysis-cache.js';
import { AnalysisClient } from './js/engine/client.js';
import { AI_STYLES } from './js/engine/difficulty.js';
import { ENGINE_VERSION, EnginePosition, validateMateResult } from './js/engine/engine.js';
import { compareAnalysisResults, isTrustedExactTablebaseResult, withResultQuality } from './js/engine/result-quality.js';
import { PlayEngineClient } from './js/engine/play-client.js';
import { AnalysisPanelView } from './js/ui/analysis-panel.js';
import { BoardView } from './js/ui/board.js';
import { BOARD_STYLES } from './js/ui/board-styles.js';
import { MoveListView } from './js/ui/move-list.js';
import { PIECE_STYLES, applyPieceStyle } from './js/ui/pieces.js';
import { StudyTreeView } from './js/ui/tree-view.js';

const $ = selector => document.querySelector(selector);
const GAME_STATE_STORAGE_KEY = 'gardner-current-game-v21.1';
const GAME_STATE_FALLBACK_KEYS = Object.freeze(['gardner-current-game-v20', 'gardner-current-game-v19.4', 'gardner-current-game-v19.3', 'gardner-current-game-v19.2', 'gardner-current-game-v19.1', 'gardner-current-game-v19', 'gardner-current-game-v18.4', 'gardner-current-game-v18.3', 'gardner-current-game-v18.2', 'gardner-current-game-v18.1', 'gardner-current-game-v17']);

// Intentional product behavior: a browser refresh starts a clean AI session.
// Do not remove this reset merely to preserve persistent analysis entries; game
// state remains separate, while AI workers/cache are deliberately disposable.
function clearAiCachesOnBoot(storage = globalThis.localStorage) {
  try {
    for (let index = storage.length - 1; index >= 0; index -= 1) {
      const key = storage.key(index);
      if (key && key.startsWith('gardner-analysis-cache-')) storage.removeItem(key);
    }
    storage.setItem('gardner-game-mode', 'local');
  } catch {}
}

clearAiCachesOnBoot();
const AI_THINK_OPTIONS = Object.freeze([1000, 2000, 3000, 5000, 10000, 20000, 30000]);
const ENGINE_KERNEL_OPTIONS = Object.freeze([
  { id: 'minifish-js', label: 'Minifish v21.1', description: 'Simple brute-force endgame AI: no cache, direct GTB leaves, force-move extensions.' },
  { id: 'orion-js', label: 'Orion v21.1', description: 'Original cached proof/search engine with tablebase bridge and mate/foundation proofs.' },
  { id: 'fairy-stockfish', label: 'Fairy-Stockfish', description: 'Optional wasm reference engine when available.' }
]);
function normalizeAiThinkMs(value) {
  const numeric = Number(value);
  return AI_THINK_OPTIONS.includes(numeric) ? numeric : 3000;
}
function formatThinkTime(ms) {
  return ms < 10000 ? `${ms / 1000}s` : `${Math.round(ms / 1000)}s`;
}
let startLayout = localStorage.getItem('gardner-start-layout') || 'standard';
if (!START_LAYOUTS.some(layout => layout.id === startLayout)) startLayout = 'standard';
let currentStart = createStartPosition(startLayout);
const game = new GameTree(currentStart.position);
const library = new StudyLibrary();
let libraryLoaded = false;
let libraryLoadingPromise = null;
let libraryLoadFailed = false;
let activeMatches = [];
let preferredMatchId = null;
let unifiedNodeSequence = 1;
let toastTimer = null;
let editorPosition = currentStart.position.clone();
let editorPiece = { color: COLORS.WHITE, type: TYPES.QUEEN };
let editorTool = 'move';
let editorMoveFrom = null;
let newGameUndoSnapshot = null;
let showBook = false;
let analysisEnabled = false;
let analysisPaused = false;
let analysisResult = null;
let lastAnalysisKey = '';
let currentAnalysisKey = '';
let gameMode = 'local';
let humanSide = localStorage.getItem('gardner-human-side') || COLORS.WHITE;
let aiStyle = localStorage.getItem('gardner-ai-style') || 'balanced';
if (!AI_STYLES.some(style => style.id === aiStyle)) aiStyle = 'balanced';
let aiThinkMs = normalizeAiThinkMs(localStorage.getItem('gardner-ai-think-ms'));
let engineKernel = localStorage.getItem('gardner-engine-kernel') || 'minifish-js';
if (!ENGINE_KERNEL_OPTIONS.some(option => option.id === engineKernel)) engineKernel = 'minifish-js';
let aiThinking = false;
let aiPaused = false;
let aiRequestKey = '';
let aiTimer = null;
const analysisCache = new AnalysisCache();
let pieceStyle = localStorage.getItem('gardner-piece-style') || 'standard';
if (!PIECE_STYLES.some(style => style.id === pieceStyle)) pieceStyle = 'standard';
let boardStyle = localStorage.getItem('gardner-board-style') || 'standard';
if (!BOARD_STYLES.some(style => style.id === boardStyle)) boardStyle = 'standard';

const elements = {
  statusText: $('#statusText'),
  turnToken: $('#turnToken'),
  moveCount: $('#moveCount'),
  copyLine: $('#copyLineButton'),
  copyTree: $('#copyTreeButton'),
  fenDisplay: $('#fenDisplay'),
  undo: $('#undoButton'),
  redo: $('#redoButton'),
  flip: $('#flipButton'),
  mirror: $('#mirrorButton'),
  edit: $('#editButton'),
  newGame: $('#newGameButton'),
  analysis: $('#analysisButton'),
  book: $('#bookButton'),
  pieceStyle: $('#pieceStyleSelect'),
  boardStyle: $('#boardStyleSelect'),
  effort: $('#effortSelect'),
  pauseAnalysis: $('#pauseAnalysisButton'),
  resumeAnalysis: $('#resumeAnalysisButton'),
  gameMode: $('#gameModeSelect'),
  humanSide: $('#humanSideSelect'),
  engineKernel: $('#engineKernelSelect'),
  difficulty: $('#difficultySelect'),
  aiThinkTime: $('#aiThinkTimeSelect'),
  startLayout: $('#startLayoutSelect'),
  humanSideField: $('#humanSideField'),
  engineKernelField: $('#engineKernelField'),
  difficultyField: $('#difficultyField'),
  aiThinkTimeField: $('#aiThinkTimeField'),
  playEngineStatus: $('#playEngineStatus'),
  modePill: $('#modePill'),
  copyFen: $('#copyFenButton'),
  libraryStatus: $('#libraryStatus'),
  matchStatus: $('#matchStatus'),
  matchCounter: $('#matchCounter'),
  promotionDialog: $('#promotionDialog'),
  promotionChoices: $('#promotionChoices'),
  editorDialog: $('#editorDialog'),
  editorBoard: $('#editorBoard'),
  piecePalette: $('#piecePalette'),
  editorClear: $('#editorClear'),
  editorStart: $('#editorStart'),
  editorCopyFen: $('#editorCopyFen'),
  fenInput: $('#fenInput'),
  loadFen: $('#loadFenButton'),
  editorErrors: $('#editorErrors'),
  applyEditor: $('#applyEditor'),
  toast: $('#toast'),
  workspace: $('.workspace'),
  gameTreePanel: $('#gameTreePanel'),
  treeEmpty: $('#treeEmpty')
};

const analysisPanel = new AnalysisPanelView($('#analysisPanel'), {
  onSelectMove: (uci, line) => playAnalysisMove(uci, line)
});

const boardView = new BoardView($('#board'), {
  getPosition: () => game.current.position,
  getLegalMoves: from => legalMoves(game.current.position, { from }),
  onAttemptMove: attemptMove
});
boardView.setPieceStyle(pieceStyle, false);
boardView.setBoardStyle(boardStyle, false);

const editorBoardView = new BoardView(elements.editorBoard, {
  getPosition: () => editorPosition,
  getLegalMoves: () => [],
  onAttemptMove: async () => {},
  onEditorSquare: sq => handleEditorSquare(sq)
});
editorBoardView.setPieceStyle(pieceStyle, false);
editorBoardView.setBoardStyle(boardStyle, false);
editorBoardView.setEditorMode(true);

const moveListView = new MoveListView($('#moveTree'), nodeId => {
  const node = findLocalNode(game.root, nodeId);
  if (!node) return;
  preferredMatchId = null;
  game.navigate(node);
  boardView.clearSelection(false);
  updateUI();
});

const studyTreeView = new StudyTreeView($('#studyTree'), $('#treeEmpty'), node => {
  cancelAiTurn({ quiet: true });
  preferredMatchId = node.id;
  if (node.localNode) {
    game.navigate(node.localNode);
  } else if (node.baseNodeId && Array.isArray(node.pathUci)) {
    const base = findLocalNode(game.root, node.baseNodeId);
    if (!base) return;
    game.navigate(base);
    for (const uci of node.pathUci) {
      const move = findMoveByUci(game.current.position, uci);
      if (!move) break;
      game.play(move, node.primarySource || 'tree');
    }
  }
  boardView.clearSelection(false);
  toast(`Opened ${node.parent ? node.san : 'the starting position'} from the unified tree.`);
  updateUI();
});

let analysisClient = null;
let playClient = null;

function ensureAnalysisClient() {
  if (analysisClient) return analysisClient;
  analysisClient = new AnalysisClient({
    onReady: message => {
      if (message.engine) $('#analysisPanel [data-ai-engine]').textContent = message.engine;
    },
    onState: message => {
      const state = !analysisEnabled && analysisResult
        ? 'stopped'
        : analysisPaused
          ? 'paused'
          : message.state;
      analysisPanel.setState(state, message.engine, message);
      const paused = state === 'paused';
      const thinking = state === 'thinking' || state === 'probing';
      elements.pauseAnalysis.hidden = paused;
      elements.pauseAnalysis.disabled = !analysisEnabled || !thinking;
      elements.resumeAnalysis.hidden = !paused;
      elements.resumeAnalysis.disabled = !analysisEnabled || !paused;
    },
    onInfo: result => {
      const rawKey = result?.cacheKey || currentAnalysisKey;
      if (rawKey && rawKey !== currentAnalysisKey) {
        analysisCache.set(rawKey, result);
        return;
      }
      const normalized = result?.lines?.length
        ? { ...result, lines: sortAnalysisLinesForPosition(game.current.position, result.lines) }
        : result;
      const key = normalized?.cacheKey || currentAnalysisKey;
      analysisResult = chooseVisibleAnalysisResult(normalized, key);
      scheduleAnalysisPaint(analysisResult, key || currentAnalysisKey);
    },
    onError: message => {
      console.error(message);
      analysisPanel.renderError(message);
      toast('The local analysis worker could not continue.');
    }
  });
  return analysisClient;
}

function releaseAnalysisClient() {
  analysisClient?.dispose();
  analysisClient = null;
}

function ensurePlayClient() {
  if (playClient) return playClient;
  playClient = new PlayEngineClient({
    onState: message => {
      const aiLabel = AI_STYLES.find(item => item.id === aiStyle)?.shortLabel || 'AI';
      if (message.state === 'thinking') {
        aiPaused = false;
        elements.playEngineStatus.textContent = `${aiLabel} thinking`;
        elements.playEngineStatus.className = 'play-engine-status thinking';
        analysisPanel.setState('thinking', message.engine || ENGINE_VERSION, message);
      } else if (message.state === 'paused') {
        aiPaused = true;
        elements.playEngineStatus.textContent = `${aiLabel} paused`;
        elements.playEngineStatus.className = 'play-engine-status thinking';
        analysisPanel.setState('paused', message.engine || ENGINE_VERSION, message);
      } else if (message.state === 'complete') {
        aiPaused = false;
        analysisPanel.setState('complete', message.engine || ENGINE_VERSION, message);
      }
      syncPauseControls();
    },
    onInfo: result => handleAiInfoResult(result),
    onResult: result => handleAiMoveResult(result),
    onError: message => {
      console.error(message);
      aiThinking = false;
      aiPaused = false;
      aiRequestKey = '';
      elements.playEngineStatus.textContent = 'AI error';
      elements.playEngineStatus.className = 'play-engine-status error';
      syncPauseControls();
      toast('The play engine could not complete its move.');
      updateUI();
    }
  });
  return playClient;
}

function releasePlayClient() {
  playClient?.dispose();
  playClient = null;
}


const ANALYSIS_PAINT_INTERVAL_MS = 500;
let pendingAnalysisPaint = null;
let analysisPaintTimer = 0;
let analysisPaintClockKey = '';

function clearAnalysisPaintTimer() {
  if (analysisPaintTimer) {
    clearTimeout(analysisPaintTimer);
    analysisPaintTimer = 0;
  }
}

function resetAnalysisPaintCadence(key = currentAnalysisKey, { clearPending = false } = {}) {
  clearAnalysisPaintTimer();
  if (clearPending) pendingAnalysisPaint = null;
  analysisPaintClockKey = key || '';
}

function flushAnalysisPaint() {
  clearAnalysisPaintTimer();
  const pending = pendingAnalysisPaint;
  pendingAnalysisPaint = null;
  if (!pending || pending.key !== currentAnalysisKey) return;
  analysisPaintClockKey = pending.key || '';
  // Principal-variation scores are only synchronized here, on the fixed 500 ms UI paint.
  analysisPanel.render(pending.result, formatAnalysisLines(pending.result), {
    state: aiThinking ? (aiPaused ? 'paused' : 'thinking') : analysisPaused ? 'paused' : analysisEnabled ? '' : 'stopped'
  });
  boardView.setArrows(composeBoardArrows());
  refreshStudyMatch();
}

function scheduleAnalysisPaint(result, key) {
  // v19: analysis paints are a fixed trailing cadence.  Every worker/cache
  // update replaces the pending payload, but nothing renders until the next
  // 500 ms tick, including tablebase, mate, and other "important" results.
  // Resetting the clock on root-key changes keeps stale queued paints from a
  // previous board from shortening the first visible tick on the new board.
  pendingAnalysisPaint = { result, key };
  if (key && key !== analysisPaintClockKey) {
    resetAnalysisPaintCadence(key, { clearPending: false });
    pendingAnalysisPaint = { result, key };
  }
  if (!analysisPaintTimer) analysisPaintTimer = setTimeout(flushAnalysisPaint, ANALYSIS_PAINT_INTERVAL_MS);
}

const TABLEBASE_PSEUDO_SCORE = 22000;

function hasDatabaseMateLine(result) {
  return Boolean(result?.lines?.some(line => Boolean(line?.tablebase)
    && Math.abs(Number(line?.score || 0)) >= 29000
    && Number(line?.dtm || 0) > 0));
}

function hasTablebasePseudoScore(result) {
  return Boolean(result?.lines?.some(line => !line?.tablebase
    && Math.abs(Number(line?.score || 0)) === TABLEBASE_PSEUDO_SCORE));
}

function preserveDatabaseDisplay(previous, candidate) {
  if (!previous?.lines?.length || !candidate) return candidate;
  const databaseLocked = isTrustedExactTablebaseResult(previous) || hasDatabaseMateLine(previous);
  if (!databaseLocked) return candidate;
  // Worker progress is deliberately lightweight and can still carry the
  // synchronous WDL sentinel (+/-220.00). Once a database mate/exact root
  // result has been published, retain its PV and DTM display while continuing
  // to refresh NODES/NPS from the live snapshot.
  if (candidate.liveProgress || (!isTrustedExactTablebaseResult(candidate)
      && !hasDatabaseMateLine(candidate) && hasTablebasePseudoScore(candidate))) {
    return {
      ...candidate,
      lines: previous.lines.map(line => ({ ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] })),
      terminal: Boolean(previous.terminal),
      tablebase: Boolean(previous.tablebase),
      tablebaseSource: previous.tablebaseSource || candidate.tablebaseSource || '',
      tablebaseSignature: previous.tablebaseSignature || candidate.tablebaseSignature || '',
      tablebaseWdl: Number(previous.tablebaseWdl ?? candidate.tablebaseWdl ?? 0),
      tablebaseDtmBound: Boolean(previous.tablebaseDtmBound),
      solved: Boolean(previous.solved),
      tablebaseDisplayLocked: true
    };
  }
  return candidate;
}

function mergePublishedAnalysisMetrics(prior, candidate, chosen) {
  if (!chosen) return chosen;
  const nodes = Math.max(0, Number(prior?.nodes || 0), Number(candidate?.nodes || 0), Number(chosen.nodes || 0));
  const elapsed = Math.max(0, Number(prior?.elapsed || 0), Number(candidate?.elapsed || 0), Number(chosen.elapsed || 0));
  let nodeTarget = Math.max(0, Number(prior?.nodeTarget || 0), Number(candidate?.nodeTarget || 0), Number(chosen.nodeTarget || 0));
  // nodeTarget is the public depth+1 work target, not an iteration-local
  // estimate. Do not let a new depth or an asynchronous proof/audit make the
  // numerator or denominator appear to go backward at the 500 ms paint.
  if (nodeTarget && nodeTarget <= nodes) {
    nodeTarget = nodes + Math.max(1_000, Math.round(nodes * 0.18));
  }
  return {
    ...chosen,
    nodes,
    elapsed,
    nps: Math.round(nodes * 1000 / Math.max(1, elapsed)),
    selDepth: Math.max(Number(prior?.selDepth || 0), Number(candidate?.selDepth || 0), Number(chosen.selDepth || 0)),
    searchDepth: Math.max(Number(prior?.searchDepth || 0), Number(candidate?.searchDepth || 0), Number(chosen.searchDepth || 0)),
    nextDepth: Math.max(Number(prior?.nextDepth || 0), Number(candidate?.nextDepth || 0), Number(chosen.nextDepth || 0)),
    nodeTarget
  };
}

function chooseVisibleAnalysisResult(candidate, key) {
  if (!candidate) return candidate;
  const prior = analysisResult?.cacheKey === key ? analysisResult : null;
  candidate = preserveDatabaseDisplay(prior, candidate);
  // v19 progress snapshots intentionally bypass durable-result ranking. They
  // are display-only: every 500 ms paint should show the newest node count and
  // current top three, while cache/proof quality remains reserved for completed
  // results below.
  if (candidate.liveProgress) {
    // Keep score/PV/proof fields from one completed iteration. A live snapshot
    // only advances metrics, so the 500 ms UI paint cannot splice a short PV
    // into an older score or tablebase/proof badge.
    if (prior?.lines?.length) {
      return mergePublishedAnalysisMetrics(prior, candidate, {
        ...prior,
        liveProgress: true,
        liveUpdate: true,
        cached: false
      });
    }
    return candidate;
  }
  if (!candidate.lines?.length) return candidate;
  const qualified = withResultQuality(candidate);
  if (!key || qualified.engine !== ENGINE_VERSION) return qualified;
  const visible = compareAnalysisResults(analysisResult, qualified);
  const cachedChoice = analysisCache.set(key, qualified);
  const chosen = compareAnalysisResults(visible, cachedChoice, { preferNextOnTie: false }) || visible || qualified;
  return mergePublishedAnalysisMetrics(prior, qualified, chosen);
}


function findLocalNode(root, id) {
  const stack = [root];
  while (stack.length) {
    const node = stack.pop();
    if (node.id === id) return node;
    stack.push(...node.children);
  }
  return null;
}

function cloneLocalTree(localNode, parent = null, map = new Map()) {
  const node = {
    id: `unified-${localNode.id}`,
    parent,
    children: [],
    position: localNode.position,
    positionKey: localNode.position.canonicalKey(),
    move: localNode.move,
    uci: localNode.move ? moveToUci(localNode.move) : '',
    san: localNode.san,
    ply: localNode.ply,
    localNode,
    sourceKinds: new Set(['local']),
    sourceLabel: localNode.source === 'ai' ? 'Played by AI' : 'Current game',
    primarySource: localNode.source || 'local',
    priority: 120
  };
  map.set(localNode.id, node);
  node.children = localNode.children.map(child => cloneLocalTree(child, node, map));
  return node;
}

function mergeSyntheticLine(anchor, uciLine, kind, { rank = 0, label = '', priority = 0 } = {}) {
  if (!anchor || !Array.isArray(uciLine) || !uciLine.length) return;
  let parent = anchor;
  let cursor = anchor.position.clone();
  const pathUci = [];
  for (const uciText of uciLine.slice(0, 7)) {
    const move = findMoveByUci(cursor, uciText);
    if (!move) break;
    const uci = moveToUci(move);
    const san = moveToSAN(cursor, move);
    const next = cursor.makeMove(move);
    pathUci.push(uci);
    let child = parent.children.find(candidate => candidate.uci === uci);
    if (!child) {
      child = {
        id: `unified-synthetic-${unifiedNodeSequence++}`,
        parent,
        children: [],
        position: next,
        positionKey: next.canonicalKey(),
        move: { ...move },
        uci,
        san,
        ply: parent.ply + 1,
        localNode: null,
        baseNodeId: game.current.id,
        pathUci: [...pathUci],
        sourceKinds: new Set([kind]),
        sourceLabel: label,
        sourceBadge: kind === 'ai' && pathUci.length === 1 ? `A${rank}` : kind === 'book' && pathUci.length === 1 ? 'B' : '',
        primarySource: kind,
        priority
      };
      parent.children.push(child);
    } else {
      child.sourceKinds.add(kind);
      child.priority = Math.max(child.priority || 0, priority);
      child.sourceLabel = [child.sourceLabel, label].filter(Boolean).filter((value, index, values) => values.indexOf(value) === index).join(' · ');
      if (pathUci.length === 1 && kind === 'ai') child.sourceBadge = child.sourceKinds.has('book') ? `A${rank}+B` : `A${rank}`;
      else if (pathUci.length === 1 && kind === 'book' && !child.sourceBadge) child.sourceBadge = 'B';
      if (!child.localNode) {
        child.baseNodeId = game.current.id;
        child.pathUci = [...pathUci];
      }
    }
    parent = child;
    cursor = next;
  }
}

function principalBookLine(position, firstEntry, maxPlies = 6) {
  const line = [];
  let cursor = position.clone();
  let entry = firstEntry;
  for (let ply = 0; ply < maxPlies && entry; ply += 1) {
    const move = findMoveByUci(cursor, moveToUci(entry.move));
    if (!move) break;
    line.push(moveToUci(move));
    cursor = cursor.makeMove(move);
    entry = library.bookMoves(cursor)[0] || null;
  }
  return line;
}

function buildUnifiedTree() {
  const map = new Map();
  const root = cloneLocalTree(game.root, null, map);
  const anchor = map.get(game.current.id) || root;

  const bookEntries = library.bookMoves(game.current.position).slice(0, 5);
  bookEntries.forEach((entry, index) => {
    mergeSyntheticLine(anchor, principalBookLine(game.current.position, entry), 'book', {
      rank: index + 1,
      label: `${entry.count} archive occurrence${entry.count === 1 ? '' : 's'}`,
      priority: 90 - index
    });
  });

  (analysisResult?.lines || []).slice(0, 3).forEach((line, index) => {
    mergeSyntheticLine(anchor, line.pv || [], 'ai', {
      rank: index + 1,
      label: `AI principal variation ${index + 1} · ${line.scoreText || ''}`,
      priority: 110 - index
    });
  });
  return { root, anchor };
}


function gameCurrentIndexPath() {
  const path = [];
  let cursor = game.current;
  while (cursor?.parent) {
    path.unshift(Math.max(0, cursor.parent.children.indexOf(cursor)));
    cursor = cursor.parent;
  }
  return path;
}

function serializeGameNode(node, budget) {
  if (!node || budget.count >= 220) return { children: [] };
  budget.count += 1;
  const children = node.children.slice(0, Math.max(0, 220 - budget.count)).map(child => ({
    move: child.move ? moveToUci(child.move) : '',
    source: child.source || 'local',
    comment: child.comment || '',
    ...serializeGameNode(child, budget)
  }));
  const preferredIndex = node.preferredChildId
    ? node.children.findIndex(child => child.id === node.preferredChildId)
    : -1;
  return { preferredIndex, children };
}

function saveGameState() {
  try {
    const payload = {
      schema: 1,
      version: 'v21.1',
      savedAt: Date.now(),
      startLayout,
      rootFen: game.root.position.toCompactFEN(),
      currentPath: gameCurrentIndexPath(),
      tree: serializeGameNode(game.root, { count: 0 })
    };
    localStorage.setItem(GAME_STATE_STORAGE_KEY, JSON.stringify(payload));
  } catch {}
}

function restoreSerializedChildren(parent, data) {
  const children = Array.isArray(data?.children) ? data.children : [];
  const restored = [];
  for (const childData of children) {
    if (!childData?.move) continue;
    const move = findMoveByUci(parent.position, childData.move);
    if (!move) continue;
    game.navigate(parent);
    const child = game.play(move, childData.source || 'local');
    child.comment = childData.comment || '';
    restored.push(child);
    restoreSerializedChildren(child, childData);
  }
  const preferredIndex = Number(data?.preferredIndex);
  if (Number.isInteger(preferredIndex) && restored[preferredIndex]) parent.preferredChildId = restored[preferredIndex].id;
}

function restoreSavedGameState() {
  try {
    let payload = JSON.parse(localStorage.getItem(GAME_STATE_STORAGE_KEY) || 'null');
    if (!payload) {
      for (const key of GAME_STATE_FALLBACK_KEYS) {
        payload = JSON.parse(localStorage.getItem(key) || 'null');
        if (payload) break;
      }
    }
    if (!payload || payload.schema !== 1 || !payload.rootFen || !payload.tree) return false;
    const rootPosition = Position.fromFEN(payload.rootFen);
    game.reset(rootPosition);
    restoreSerializedChildren(game.root, payload.tree);
    let cursor = game.root;
    for (const index of Array.isArray(payload.currentPath) ? payload.currentPath : []) {
      const next = cursor.children[Number(index)];
      if (!next) break;
      cursor = next;
    }
    game.navigate(cursor);
    preferredMatchId = null;
    return true;
  } catch (error) {
    try { localStorage.removeItem(GAME_STATE_STORAGE_KEY); } catch {}
    console.warn('Saved Gardner game state could not be restored.', error);
    return false;
  }
}

function sideIsAI(color) {
  if (gameMode === 'ai-ai') return true;
  if (gameMode === 'human-ai') return color !== humanSide;
  return false;
}

function analysisModeAllowed() {
  return gameMode === 'local';
}

function stopManualAnalysisForPlayMode() {
  if (analysisModeAllowed()) return;
  if (analysisEnabled || analysisClient?.active) {
    analysisEnabled = false;
    analysisPaused = false;
    analysisClient?.stop();
    lastAnalysisKey = '';
    elements.analysis.setAttribute('aria-pressed', 'false');
    elements.analysis.classList.remove('active');
    analysisPanel.setEnabled(false, true);
  }
  // Keep at most one worker-resident tablebase cache alive: entering AI play
  // releases the manual-analysis worker instead of leaving it dormant.
  releaseAnalysisClient();
}

function canHumanMove() {
  return !aiThinking && !sideIsAI(game.current.position.turn) && !isFinished(currentStatus());
}

function currentAnalysisContext(position = game.current.position, historyFens = engineHistoryFens()) {
  return {
    historyFens,
    // v14.2: Orion search/mate cache is keyed only by position + recent
    // history.  Kernel/style/mode affect request scheduling, not the reusable
    // search artifact, so analysis and AI play can share cached work.
    key: buildAnalysisKey(position, historyFens)
  };
}

function cancelAiTurn({ quiet = false } = {}) {
  clearTimeout(aiTimer);
  aiTimer = null;
  playClient?.cancel();
  aiThinking = false;
  aiPaused = false;
  aiRequestKey = '';
  syncPauseControls();
  if (!quiet) {
    elements.playEngineStatus.textContent = 'Ready';
    elements.playEngineStatus.className = 'play-engine-status';
  }
}

function syncPauseControls() {
  if (aiThinking) {
    elements.pauseAnalysis.hidden = aiPaused;
    elements.resumeAnalysis.hidden = !aiPaused;
    elements.pauseAnalysis.disabled = aiPaused;
    elements.resumeAnalysis.disabled = !aiPaused;
    return;
  }
  if (!analysisModeAllowed()) {
    elements.pauseAnalysis.hidden = false;
    elements.resumeAnalysis.hidden = true;
    elements.pauseAnalysis.disabled = true;
    elements.resumeAnalysis.disabled = true;
  }
}

function lineUtilityForPosition(position, line) {
  const score = Number(line?.score || 0);
  return position.turn === COLORS.BLACK ? -score : score;
}

function sortAnalysisLinesForPosition(position, lines) {
  return (Array.isArray(lines) ? lines : [])
    .slice()
    .sort((a, b) => lineUtilityForPosition(position, b) - lineUtilityForPosition(position, a));
}

function validateCachedAnalysis(position, key, cached) {
  if (!cached?.lines?.length) return null;
  const enginePosition = EnginePosition.fromFEN(position.toCompactFEN());
  const directTablebase = isTrustedExactTablebaseResult(cached);
  const leading = cached.lines[0] || {};
  const proofBackedMate = Boolean(leading.mateVerified && (leading.mateProof || leading.endgameProof || cached.mateProof || cached.endgameProof));
  const trustedProof = Boolean(cached.fortressProof || cached.endgameProof || proofBackedMate || cached.terminal);
  if (cached.engine !== ENGINE_VERSION || (!directTablebase && !trustedProof)) {
    analysisCache.delete(key);
    return null;
  }
  const lines = sortAnalysisLinesForPosition(
    position,
    cached.lines.filter(line => !line.mateVerified || (Boolean(line.mateProof || line.endgameProof || cached.mateProof || cached.endgameProof) && validateMateResult(enginePosition, line)))
  );
  if (cached.lines[0]?.mateVerified && lines[0] !== cached.lines[0]) {
    analysisCache.delete(key);
    return null;
  }
  return withResultQuality({
    ...cached,
    lines,
    solved: Boolean(directTablebase || cached.fortressProof || cached.endgameProof || (lines[0]?.mateVerified && (lines[0]?.mateProof || lines[0]?.endgameProof || cached.mateProof || cached.endgameProof)))
  });
}

async function playAnalysisMove(uci) {
  const move = findMoveByUci(game.current.position, uci);
  if (!move || isFinished(currentStatus())) {
    toast('That engine line is no longer legal in the current position.');
    return;
  }
  const san = moveToSAN(game.current.position, move);
  cancelAiTurn({ quiet: true });
  preferredMatchId = null;
  game.play(move, 'analysis');
  boardView.clearSelection(false);
  toast(`Played the engine recommendation ${san}.`);
  updateUI();
}

function maybeStartAiTurn(status = currentStatus()) {
  const shouldMove = !isFinished(status) && sideIsAI(game.current.position.turn);
  if (!shouldMove) {
    if (aiThinking || aiTimer) cancelAiTurn();
    return false;
  }
  const context = currentAnalysisContext();
  const key = `${context.key}|${gameMode}|${humanSide}|K${engineKernel}|S${aiStyle}|T${aiThinkMs}`;
  if (aiThinking && aiRequestKey === key) return true;
  cancelAiTurn({ quiet: true });
  aiThinking = true;
  aiPaused = false;
  aiRequestKey = key;
  currentAnalysisKey = context.key;
  resetAnalysisPaintCadence(context.key, { clearPending: true });
  analysisResult = null;
  analysisPanel.setEnabled(true, true);
  analysisPanel.renderSearching();
  syncPauseControls();
  elements.playEngineStatus.textContent = `${AI_STYLES.find(item => item.id === aiStyle)?.shortLabel || 'AI'} queued`;
  elements.playEngineStatus.className = 'play-engine-status thinking';
  const delay = gameMode === 'ai-ai' ? 320 : 180;
  aiTimer = setTimeout(() => {
    aiTimer = null;
    if (!aiThinking || aiRequestKey !== key) return;
    ensurePlayClient().search({
      fen: game.current.position.toCompactFEN(),
      historyFens: engineHistoryFens(),
      style: aiStyle,
      kernel: engineKernel,
      thinkTimeMs: aiThinkMs,
      cacheKey: context.key,
      resumeResult: validateCachedAnalysis(game.current.position, context.key, analysisCache.get(context.key))
    });
  }, delay);
  return true;
}

function handleAiInfoResult(result) {
  if (!aiThinking || !sideIsAI(game.current.position.turn)) return;
  const context = currentAnalysisContext();
  currentAnalysisKey = context.key;
  const normalized = result?.lines?.length
    ? { ...result, lines: sortAnalysisLinesForPosition(game.current.position, result.lines) }
    : result;
  const stored = {
    ...normalized,
    cacheKey: context.key,
    cached: Boolean(normalized?.cached),
    aiInternal: true
  };
  analysisResult = chooseVisibleAnalysisResult(stored, context.key);
  analysisPanel.setEnabled(true, true);
  scheduleAnalysisPaint(analysisResult, context.key);
}

function handleAiMoveResult(result) {
  const expectedKey = `${currentAnalysisContext().key}|${gameMode}|${humanSide}|K${engineKernel}|S${aiStyle}|T${aiThinkMs}`;
  if (!aiThinking || aiRequestKey !== expectedKey || !sideIsAI(game.current.position.turn)) return;
  const context = currentAnalysisContext();
  const stored = {
    ...result,
    lines: sortAnalysisLinesForPosition(game.current.position, result.lines || []),
    cacheKey: context.key,
    cached: false,
    searchDepth: Math.max(1, Number(result.depth || 0) + 1),
    nextDepth: Math.max(1, Number(result.depth || 0) + 1)
  };
  analysisCache.set(context.key, stored);
  analysisResult = stored;
  const move = findMoveByUci(game.current.position, result.selectedMove || result.lines?.[0]?.move);
  aiThinking = false;
  aiPaused = false;
  aiRequestKey = '';
  syncPauseControls();
  elements.playEngineStatus.textContent = `${result.styleLabel || AI_STYLES.find(item => item.id === aiStyle)?.shortLabel || 'AI'} · d${result.depth || 0}`;
  elements.playEngineStatus.className = 'play-engine-status';
  if (!move) {
    updateUI();
    return;
  }
  preferredMatchId = null;
  game.play(move, 'ai');
  boardView.clearSelection(false);
  updateUI();
}

function buildEngineKernelSelect() {
  elements.engineKernel.innerHTML = '';
  for (const kernel of ENGINE_KERNEL_OPTIONS) {
    const option = document.createElement('option');
    option.value = kernel.id;
    option.textContent = kernel.label;
    option.title = kernel.description;
    option.selected = kernel.id === engineKernel;
    elements.engineKernel.appendChild(option);
  }
}

function selectedEngineKernelLabel() {
  return ENGINE_KERNEL_OPTIONS.find(option => option.id === engineKernel)?.label || engineKernel;
}

function buildDifficultySelect() {
  elements.difficulty.innerHTML = '';
  for (const config of AI_STYLES) {
    const option = document.createElement('option');
    option.value = config.id;
    option.textContent = config.label;
    option.title = config.description;
    option.selected = config.id === aiStyle;
    elements.difficulty.appendChild(option);
  }
}

function buildAiThinkTimeSelect() {
  elements.aiThinkTime.innerHTML = '';
  for (const ms of AI_THINK_OPTIONS) {
    const option = document.createElement('option');
    option.value = String(ms);
    option.textContent = formatThinkTime(ms);
    option.selected = ms === aiThinkMs;
    elements.aiThinkTime.appendChild(option);
  }
}


function buildStartLayoutSelect() {
  elements.startLayout.innerHTML = '';
  for (const layout of START_LAYOUTS) {
    const option = document.createElement('option');
    option.value = layout.id;
    option.textContent = layout.label;
    option.title = layout.description;
    option.selected = layout.id === startLayout;
    elements.startLayout.appendChild(option);
  }
}

function cloneStartPositionState(start) {
  if (!start) return null;
  return {
    ...start,
    white: Array.isArray(start.white) ? start.white.slice() : start.white,
    black: Array.isArray(start.black) ? start.black.slice() : start.black,
    position: start.position?.clone?.() || null
  };
}

function saveNewGameUndoSnapshot() {
  newGameUndoSnapshot = {
    tree: game.captureSnapshot(),
    start: cloneStartPositionState(currentStart),
    startLayout: START_LAYOUTS.some(layout => layout.id === currentStart?.id) ? currentStart.id : startLayout
  };
}

function restoreNewGameUndoSnapshot() {
  const snapshot = newGameUndoSnapshot;
  if (!snapshot?.tree || !game.restoreSnapshot(snapshot.tree)) return false;
  currentStart = cloneStartPositionState(snapshot.start) || currentStart;
  startLayout = START_LAYOUTS.some(layout => layout.id === snapshot.startLayout) ? snapshot.startLayout : startLayout;
  elements.startLayout.value = startLayout;
  newGameUndoSnapshot = null;
  return true;
}

function startNewGame({ announce = true } = {}) {
  cancelAiTurn({ quiet: true });
  saveNewGameUndoSnapshot();
  preferredMatchId = null;
  currentStart = createStartPosition(startLayout);
  game.reset(currentStart.position);
  analysisResult = null;
  lastAnalysisKey = '';
  boardView.clearSelection(false);
  if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK, false);
  if (announce) toast(`${START_LAYOUTS.find(layout => layout.id === startLayout)?.label || 'Selected'} layout · ${currentStart.signature}. Undo restores the previous game.`);
  updateUI();
}

function syncModeControls() {
  if (!['local', 'human-ai', 'ai-ai'].includes(gameMode)) gameMode = 'local';
  if (![COLORS.WHITE, COLORS.BLACK].includes(humanSide)) humanSide = COLORS.WHITE;
  elements.gameMode.value = gameMode;
  elements.humanSide.value = humanSide;
  elements.engineKernel.value = engineKernel;
  elements.difficulty.value = aiStyle;
  elements.aiThinkTime.value = String(aiThinkMs);
  elements.startLayout.value = startLayout;
  elements.humanSideField.classList.toggle('is-hidden', gameMode !== 'human-ai');
  elements.difficultyField.classList.toggle('is-hidden', gameMode === 'local');
  elements.aiThinkTimeField.classList.toggle('is-hidden', gameMode === 'local');
  elements.analysis.disabled = !analysisModeAllowed();
  elements.analysis.title = analysisModeAllowed() ? 'Toggle local continuous analysis' : 'Analysis mode is disabled during AI play; this panel shows AI thinking instead.';
  const labels = {
    local: 'Local two-player',
    'human-ai': `Player (${humanSide === COLORS.WHITE ? 'White' : 'Black'}) vs AI`,
    'ai-ai': 'AI vs AI'
  };
  elements.modePill.innerHTML = `<i></i> ${labels[gameMode]}`;
}

async function attemptMove(from, to, candidates) {
  const status = currentStatus();
  if (isFinished(status)) return;
  if (!canHumanMove()) {
    toast(sideIsAI(game.current.position.turn) ? 'The AI is moving for this side.' : 'Please wait for the AI search to finish.');
    return;
  }
  let move = candidates[0];
  if (candidates.length > 1 && candidates.some(candidate => candidate.promotion)) {
    const promotion = await choosePromotion(candidates[0].color);
    if (!promotion) return;
    move = candidates.find(candidate => candidate.promotion === promotion);
  }
  if (!move) return;
  cancelAiTurn({ quiet: true });
  preferredMatchId = null;
  game.play(move);
  updateUI();
}

function choosePromotion(color) {
  return new Promise(resolve => {
    elements.promotionChoices.innerHTML = '';
    [TYPES.QUEEN, TYPES.ROOK, TYPES.BISHOP, TYPES.KNIGHT].forEach(type => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'promotion-choice';
      const pieceNode = document.createElement('strong');
      applyPieceStyle(pieceNode, pieceStyle, color, type);
      const label = document.createElement('span');
      label.textContent = PIECE_NAMES[type];
      button.append(pieceNode, label);
      button.addEventListener('click', () => elements.promotionDialog.close(type));
      elements.promotionChoices.appendChild(button);
    });
    const onClose = () => {
      elements.promotionDialog.removeEventListener('close', onClose);
      resolve(['q', 'r', 'b', 'n'].includes(elements.promotionDialog.returnValue) ? elements.promotionDialog.returnValue : null);
    };
    elements.promotionDialog.addEventListener('close', onClose);
    elements.promotionDialog.showModal();
  });
}

function currentStatus() {
  return gameStatus(game.current.position, game.repetitionCount());
}

function isFinished(status) {
  return ['checkmate', 'stalemate', 'draw-insufficient', 'draw-repetition'].includes(status.state);
}

function currentBookEntries() {
  return libraryLoaded ? library.bookMoves(game.current.position) : [];
}

function arrowFromUci(uci, kind, title = '') {
  const match = String(uci || '').match(/^([a-e][1-5])([a-e][1-5])/i);
  if (!match) return null;
  const from = parseStandardCoord(match[1].toLowerCase());
  const to = parseStandardCoord(match[2].toLowerCase());
  if (from < 0 || to < 0) return null;
  return { from, to, kind, title };
}

function composeBoardArrows() {
  const arrows = [];
  if (showBook) {
    const entries = currentBookEntries();
    const maxCount = Math.max(1, ...entries.map(entry => entry.count));
    for (const entry of entries) {
      arrows.push({
        from: entry.move.from,
        to: entry.move.to,
        kind: 'book',
        opacity: 0.15 + 0.18 * entry.count / maxCount,
        title: `${entry.san} · ${entry.count} archive occurrence${entry.count === 1 ? '' : 's'}`
      });
    }
  }
  const resultMatchesBoard = analysisResult?.cacheKey && analysisResult.cacheKey === currentAnalysisContext().key;
  if (resultMatchesBoard && analysisResult?.lines?.[0]?.pv?.length) {
    const best = analysisResult.lines[0];
    const first = arrowFromUci(best.pv[0], 'engine', `Engine choice ${best.scoreText}`);
    const response = arrowFromUci(best.pv[1], 'response', 'Best reply');
    if (first) arrows.push(first);
    if (response) arrows.push(response);
  }
  return arrows;
}

function findMoveByUci(position, uci) {
  return legalMoves(position).find(move => moveToUci(move).toLowerCase() === String(uci).toLowerCase()) || null;
}

const pvSanFormatCache = new Map();
const PV_SAN_FORMAT_CACHE_LIMIT = 384;
function rememberPvSan(cacheKey, formatted) {
  pvSanFormatCache.set(cacheKey, formatted);
  if (pvSanFormatCache.size > PV_SAN_FORMAT_CACHE_LIMIT) {
    const oldest = pvSanFormatCache.keys().next().value;
    if (oldest !== undefined) pvSanFormatCache.delete(oldest);
  }
  return formatted;
}

function formatPV(position, pv) {
  let cursor = position.clone();
  const san = [];
  for (const uci of pv || []) {
    const move = findMoveByUci(cursor, uci);
    if (!move) break;
    san.push(moveToSAN(cursor, move));
    cursor = cursor.makeMove(move);
  }
  return san;
}

function formatAnalysisLines(result) {
  const rootKey = currentAnalysisKey || game.current.position.canonicalKey();
  return (result.lines || []).map(line => {
    const pv = Array.isArray(line.pv) ? line.pv : [];
    const cacheKey = `${rootKey}|${pv.join(' ')}`;
    const cached = pvSanFormatCache.get(cacheKey);
    if (cached) return { ...line, ...cached };
    const pvSan = formatPV(game.current.position, pv);
    return {
      ...line,
      ...rememberPvSan(cacheKey, {
        firstSan: pvSan[0] || line.move,
        pvSan: pvSan.join(' ')
      })
    };
  });
}

function engineHistoryFens() {
  const fens = [];
  let cursor = game.current.parent;
  while (cursor) {
    fens.unshift(cursor.position.toCompactFEN());
    cursor = cursor.parent;
  }
  return fens;
}

function restartAnalysis(force = false) {
  if (!analysisModeAllowed()) {
    stopManualAnalysisForPlayMode();
    currentAnalysisKey = currentAnalysisContext().key;
    resetAnalysisPaintCadence(currentAnalysisKey, { clearPending: true });
    if (!aiThinking && !analysisResult) analysisPanel.renderIdle();
    return;
  }
  const context = currentAnalysisContext();
  currentAnalysisKey = context.key;
  resetAnalysisPaintCadence(context.key, { clearPending: true });
  const resumeResult = validateCachedAnalysis(game.current.position, context.key, analysisCache.get(context.key));
  const cached = resumeResult;
  if (cached) {
    analysisResult = cached;
    scheduleAnalysisPaint(cached, context.key);
  } else {
    analysisResult = null;
    if (analysisEnabled && !analysisPaused) analysisPanel.renderSearching();
    else if (analysisEnabled && analysisPaused) {
      analysisPanel.renderSearching();
      analysisPanel.setState('paused');
    } else analysisPanel.renderIdle();
  }
  boardView.setArrows(composeBoardArrows(), false);

  if (!analysisEnabled || analysisPaused) return;
  const client = ensureAnalysisClient();
  if (!force && context.key === lastAnalysisKey && client.active) return;
  lastAnalysisKey = context.key;
  client.update({
    fen: game.current.position.toCompactFEN(),
    bookMoves: [],
    historyFens: context.historyFens,
    effortMs: Number(elements.effort.value),
    kernel: engineKernel,
    multipv: 3,
    cacheKey: context.key,
    resumeResult
  });
}

function updateUI() {
  const position = game.current.position;
  const status = currentStatus();
  const finished = isFinished(status);
  const colorName = position.turn === COLORS.WHITE ? 'White' : 'Black';

  stopManualAnalysisForPlayMode();
  restartAnalysis();
  const aiTurn = maybeStartAiTurn(status);

  const messages = {
    playing: `${colorName} to move`,
    check: `${colorName} is in check`,
    checkmate: `Checkmate — ${status.winner === COLORS.WHITE ? 'White' : 'Black'} wins`,
    stalemate: 'Draw by stalemate',
    'draw-insufficient': 'Draw by insufficient material',
    'draw-repetition': 'Draw by threefold repetition'
  };
  elements.statusText.textContent = aiTurn
    ? `${colorName} AI is thinking`
    : messages[status.state] || `${colorName} to move`;
  elements.turnToken.textContent = finished ? 'FINISHED' : aiTurn ? 'AI' : colorName.toUpperCase();
  elements.turnToken.className = `turn-token ${finished ? 'finished' : position.turn === COLORS.WHITE ? 'white' : 'black'}`;

  boardView.setArrows(composeBoardArrows(), false);
  boardView.setState({
    lastMove: game.current.move,
    status,
    locked: finished || aiTurn || aiThinking
  });
  $('#board').classList.toggle('ai-locked', aiTurn || aiThinking);
  moveListView.render(game);
  const plies = game.currentPath().length;
  elements.moveCount.textContent = `${plies} ${plies === 1 ? 'ply' : 'plies'}`;
  elements.fenDisplay.textContent = position.toStandardFEN();
  elements.undo.disabled = !game.current.parent && !newGameUndoSnapshot;
  elements.redo.disabled = !game.current.children.length;
  elements.edit.disabled = aiThinking;
  if (finished) {
    cancelAiTurn();
    elements.playEngineStatus.textContent = 'Game over';
  } else if (gameMode === 'local' && !aiThinking) {
    elements.playEngineStatus.textContent = 'Local';
    elements.playEngineStatus.className = 'play-engine-status';
  }
  refreshStudyMatch();
  saveGameState();
}

function refreshStudyMatch() {
  if (!elements.gameTreePanel?.open) return;
  activeMatches = libraryLoaded ? library.matches(game.current.position) : [];
  if (!libraryLoaded) {
    elements.matchStatus.classList.remove('matched');
    elements.matchStatus.innerHTML = `<i></i> ${libraryLoadingPromise ? 'Loading archive…' : 'Archive not loaded · local and engine branches only'}`;
    elements.matchCounter.textContent = 'Lazy archive';
  } else if (activeMatches.length) {
    elements.matchStatus.classList.add('matched');
    elements.matchStatus.innerHTML = `<i></i> Exact archive position · ${activeMatches.length} merged match${activeMatches.length === 1 ? '' : 'es'}`;
    elements.matchCounter.textContent = `${activeMatches.length} archive match${activeMatches.length === 1 ? '' : 'es'}`;
  } else {
    elements.matchStatus.classList.remove('matched');
    elements.matchStatus.innerHTML = '<i></i> No exact archive node · showing local and engine branches';
    elements.matchCounter.textContent = '0 archive matches';
  }
  const { root, anchor } = buildUnifiedTree();
  studyTreeView.render(root, anchor, game.current.position.canonicalKey());
}

function toast(message) {
  clearTimeout(toastTimer);
  elements.toast.textContent = message;
  elements.toast.classList.add('visible');
  toastTimer = setTimeout(() => elements.toast.classList.remove('visible'), 2300);
}

function buildBoardStyleSelect() {
  elements.boardStyle.innerHTML = '';
  for (const style of BOARD_STYLES) {
    const option = document.createElement('option');
    option.value = style.id;
    option.textContent = style.label;
    option.selected = style.id === boardStyle;
    elements.boardStyle.appendChild(option);
  }
}

function buildPieceStyleSelect() {
  elements.pieceStyle.innerHTML = '';
  for (const style of PIECE_STYLES) {
    const option = document.createElement('option');
    option.value = style.id;
    option.textContent = style.label;
    option.selected = style.id === pieceStyle;
    elements.pieceStyle.appendChild(option);
  }
}

function clearEditorMoveSelection({ render = false } = {}) {
  editorMoveFrom = null;
  editorBoardView.clearSelection(render);
}

function handleEditorSquare(sq) {
  if (editorTool === 'move') {
    if (editorMoveFrom === null) {
      if (!editorPosition.pieceAt(sq)) return;
      editorMoveFrom = sq;
      editorBoardView.selectedSquare = sq;
      editorBoardView.legalTargets = [];
      editorBoardView.render();
      clearEditorErrors();
      return;
    }
    if (sq === editorMoveFrom) {
      clearEditorMoveSelection({ render: true });
      return;
    }
    const moving = editorPosition.pieceAt(editorMoveFrom);
    if (moving) {
      // Editor move mode intentionally ignores legal-move rules: it is a
      // board-construction tool and may overwrite the target square.
      editorPosition.setPiece(sq, moving);
      editorPosition.setPiece(editorMoveFrom, null);
    }
    clearEditorMoveSelection({ render: true });
    clearEditorErrors();
    return;
  }

  if (editorTool === 'erase') editorPosition.setPiece(sq, null);
  else if (editorTool === 'piece' && editorPiece) editorPosition.setPiece(sq, editorPiece);
  clearEditorMoveSelection({ render: true });
  clearEditorErrors();
}

function buildPalette() {
  elements.piecePalette.innerHTML = '';
  for (const color of [COLORS.WHITE, COLORS.BLACK]) {
    for (const type of [TYPES.KING, TYPES.QUEEN, TYPES.ROOK, TYPES.BISHOP, TYPES.KNIGHT, TYPES.PAWN]) {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = 'palette-piece';
      applyPieceStyle(button, pieceStyle, color, type);
      button.title = `${color === COLORS.WHITE ? 'White' : 'Black'} ${PIECE_NAMES[type]}`;
      button.dataset.color = color;
      button.dataset.type = type;
      button.addEventListener('click', () => {
        editorTool = 'piece';
        editorPiece = { color, type };
        clearEditorMoveSelection({ render: true });
        updatePaletteSelection();
      });
      elements.piecePalette.appendChild(button);
    }
  }

  const moveTool = document.createElement('button');
  moveTool.type = 'button';
  moveTool.className = 'palette-piece editor-tool move-tool';
  moveTool.textContent = '↔  Move piece';
  moveTool.dataset.tool = 'move';
  moveTool.addEventListener('click', () => {
    editorTool = 'move';
    clearEditorMoveSelection({ render: true });
    updatePaletteSelection();
  });
  elements.piecePalette.appendChild(moveTool);

  const eraser = document.createElement('button');
  eraser.type = 'button';
  eraser.className = 'palette-piece editor-tool eraser';
  eraser.textContent = '⌫  Erase square';
  eraser.dataset.tool = 'erase';
  eraser.addEventListener('click', () => {
    editorTool = 'erase';
    clearEditorMoveSelection({ render: true });
    updatePaletteSelection();
  });
  elements.piecePalette.appendChild(eraser);
  updatePaletteSelection();
}

function updatePaletteSelection() {
  elements.piecePalette.querySelectorAll('.palette-piece').forEach(button => {
    const selected = button.dataset.tool
      ? button.dataset.tool === editorTool
      : editorTool === 'piece'
        && button.dataset.color === editorPiece?.color
        && button.dataset.type === editorPiece?.type;
    button.classList.toggle('selected', selected);
  });
}

function openEditor() {
  editorPosition = game.current.position.clone();
  editorPiece = { color: COLORS.WHITE, type: TYPES.QUEEN };
  editorTool = 'move';
  clearEditorMoveSelection({ render: false });
  elements.fenInput.value = editorPosition.toStandardFEN();
  elements.editorDialog.querySelector(`input[name="editorTurn"][value="${editorPosition.turn}"]`).checked = true;
  clearEditorErrors();
  buildPalette();
  editorBoardView.render();
  elements.editorDialog.showModal();
}

function clearEditorErrors() {
  elements.editorErrors.textContent = '';
}

function showEditorErrors(errors) {
  elements.editorErrors.innerHTML = errors.map(error => `• ${escapeHtml(error)}`).join('<br>');
}

function escapeHtml(text) {
  return String(text).replace(/[&<>'"]/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' })[char]);
}

function syncEditorTurn() {
  const selected = elements.editorDialog.querySelector('input[name="editorTurn"]:checked');
  editorPosition.turn = selected?.value || COLORS.WHITE;
}

async function copyText(text, successMessage) {
  try {
    await navigator.clipboard.writeText(text);
    toast(successMessage);
  } catch {
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    document.execCommand('copy');
    textarea.remove();
    toast(successMessage);
  }
}

async function loadLibrary() {
  if (libraryLoaded) return library;
  if (libraryLoadingPromise) return libraryLoadingPromise;
  if (libraryLoadFailed) {
    library.studies.length = 0;
    library.positionIndex.clear();
  }
  libraryLoadingPromise = (async () => {
  try {
    libraryLoadFailed = false;
    const manifest = await fetch('data/library.json').then(response => {
      if (!response.ok) throw new Error(`Library manifest returned ${response.status}.`);
      return response.json();
    });

    let totalMoves = 0;
    let totalErrors = 0;
    for (const source of manifest.sources) {
      elements.libraryStatus.textContent = `Loading ${library.studies.length + 1}/${manifest.sources.length}`;
      await new Promise(resolve => requestAnimationFrame(resolve));
      const text = await fetch(source.path).then(response => {
        if (!response.ok) throw new Error(`${source.path} returned ${response.status}.`);
        return response.text();
      });
      const study = parsePGN(text, source.id, { coordSystem: source.coordinateSystem || manifest.coordinateSystem || COORD_SYSTEMS.STANDARD });
      study.title = source.title;
      study.kind = source.kind;
      study.path = source.path;
      library.addStudy(study);
      totalMoves += study.parsedMoves;
      totalErrors += study.errors.length;
    }

    libraryLoaded = true;
    elements.libraryStatus.textContent = `${totalMoves.toLocaleString()} nodes`;
    elements.libraryStatus.className = 'loading-dot ready';
    if (totalErrors) {
      console.warn(`PGN archive loaded with ${totalErrors} skipped tokens.`, library.studies.map(s => ({ source: s.sourceName, errors: s.errors.slice(0, 12) })));
    }
    boardView.setArrows(composeBoardArrows());
    refreshStudyMatch();
    return library;
  } catch (error) {
    console.error(error);
    elements.libraryStatus.textContent = 'Unavailable';
    elements.libraryStatus.className = 'loading-dot error';
    studyTreeView.clear('The PGN archive could not be loaded.');
    $('#treeEmpty').innerHTML += '<span>Run this project through a local web server instead of opening index.html directly.</span>';
    libraryLoadFailed = true;
    toast('Start a local web server to load the PGN archive.');
    throw error;
  } finally {
    libraryLoadingPromise = null;
  }
  })();
  return libraryLoadingPromise;
}

async function ensureLibraryLoaded() {
  if (libraryLoaded) return true;
  if (libraryLoadFailed) {
    elements.libraryStatus.textContent = 'Retrying…';
    elements.libraryStatus.className = 'loading-dot';
  }
  try {
    await loadLibrary();
    return true;
  } catch {
    return false;
  }
}

elements.undo.addEventListener('click', () => {
  cancelAiTurn({ quiet: true });
  let changed = false;
  let restoredNewGame = false;
  if (gameMode === 'human-ai') {
    if (game.undo()) changed = true;
    while (changed && game.current.parent && game.current.position.turn !== humanSide) game.undo();
  } else {
    changed = Boolean(game.undo());
  }
  if (!changed && newGameUndoSnapshot) {
    changed = restoreNewGameUndoSnapshot();
    restoredNewGame = changed;
  }
  if (changed) {
    preferredMatchId = null;
    boardView.clearSelection(false);
    if (restoredNewGame) toast('Restored the game from before New game.');
    updateUI();
  }
});
elements.redo.addEventListener('click', () => {
  cancelAiTurn({ quiet: true });
  if (game.redo()) {
    preferredMatchId = null;
    boardView.clearSelection(false);
    updateUI();
  }
});
elements.flip.addEventListener('click', () => boardView.flip());
elements.mirror.addEventListener('click', () => boardView.mirror());
elements.edit.addEventListener('click', () => {
  cancelAiTurn({ quiet: true });
  openEditor();
});
elements.newGame.addEventListener('click', () => startNewGame());

elements.analysis.addEventListener('click', () => {
  if (!analysisModeAllowed()) {
    stopManualAnalysisForPlayMode();
    toast('Analysis mode is disabled during AI play; the panel will show AI internal thinking.');
    return;
  }
  analysisEnabled = !analysisEnabled;
  analysisPaused = false;
  elements.analysis.setAttribute('aria-pressed', String(analysisEnabled));
  elements.analysis.classList.toggle('active', analysisEnabled);
  elements.pauseAnalysis.disabled = !analysisEnabled;
  elements.pauseAnalysis.hidden = false;
  elements.resumeAnalysis.hidden = true;
  analysisPanel.setEnabled(analysisEnabled, true);
  if (analysisEnabled) {
    lastAnalysisKey = '';
    resetAnalysisPaintCadence(currentAnalysisContext().key, { clearPending: true });
    restartAnalysis(true);
    toast('Local continuous analysis started.');
  } else {
    analysisClient?.stop();
    releaseAnalysisClient();
    lastAnalysisKey = '';
    resetAnalysisPaintCadence(currentAnalysisKey, { clearPending: true });
    if (analysisResult) analysisPanel.renderStopped(analysisResult, formatAnalysisLines(analysisResult));
    else analysisPanel.renderIdle();
    boardView.setArrows(composeBoardArrows());
    toast('Analysis stopped; the latest result remains cached.');
  }
});

elements.pauseAnalysis.addEventListener('click', () => {
  if (aiThinking) {
    if (aiPaused) return;
    aiPaused = true;
    playClient?.pause();
    syncPauseControls();
    if (analysisResult) analysisPanel.render(analysisResult, formatAnalysisLines(analysisResult), { state: 'paused' });
    else analysisPanel.setState('paused');
    toast('AI thinking paused. The move clock is paused too.');
    return;
  }
  if (!analysisEnabled || analysisPaused) return;
  analysisPaused = true;
  analysisClient?.pause();
  elements.pauseAnalysis.hidden = true;
  elements.resumeAnalysis.hidden = false;
  resetAnalysisPaintCadence(currentAnalysisKey, { clearPending: true });
  if (analysisResult) analysisPanel.render(analysisResult, formatAnalysisLines(analysisResult), { state: 'paused' });
  else analysisPanel.setState('paused');
  toast('Analysis paused. The current search result is preserved.');
});

elements.resumeAnalysis.addEventListener('click', () => {
  if (aiThinking) {
    if (!aiPaused) return;
    aiPaused = false;
    playClient?.resume();
    syncPauseControls();
    if (analysisResult) analysisPanel.render(analysisResult, formatAnalysisLines(analysisResult), { state: 'thinking' });
    else analysisPanel.renderSearching();
    toast('AI thinking resumed.');
    return;
  }
  if (!analysisEnabled || !analysisPaused) return;
  analysisPaused = false;
  elements.pauseAnalysis.hidden = false;
  elements.resumeAnalysis.hidden = true;
  if (analysisClient?.active && currentAnalysisKey === lastAnalysisKey) analysisClient.resume();
  else restartAnalysis(true);
  toast('Analysis resumed from the saved depth.');
});

elements.book.addEventListener('click', async () => {
  showBook = !showBook;
  elements.book.setAttribute('aria-pressed', String(showBook));
  elements.book.classList.toggle('active', showBook);
  elements.book.textContent = 'Book';
  if (showBook) {
    elements.book.disabled = true;
    const ready = await ensureLibraryLoaded();
    elements.book.disabled = false;
    if (!ready) {
      showBook = false;
      elements.book.setAttribute('aria-pressed', 'false');
      elements.book.classList.remove('active');
    }
  }
  boardView.setArrows(composeBoardArrows());
  const count = currentBookEntries().length;
  toast(showBook ? (count ? `${count} archive move${count === 1 ? '' : 's'} shown.` : 'No archive move matches this position.') : 'Book arrows hidden.');
});

elements.pieceStyle.addEventListener('change', () => {
  pieceStyle = elements.pieceStyle.value;
  localStorage.setItem('gardner-piece-style', pieceStyle);
  boardView.setPieceStyle(pieceStyle);
  editorBoardView.setPieceStyle(pieceStyle);
  buildPalette();
});

elements.boardStyle.addEventListener('change', () => {
  boardStyle = elements.boardStyle.value;
  localStorage.setItem('gardner-board-style', boardStyle);
  boardView.setBoardStyle(boardStyle);
  editorBoardView.setBoardStyle(boardStyle);
});

elements.effort.addEventListener('change', () => {
  if (analysisEnabled && !analysisPaused) restartAnalysis(true);
});

elements.gameMode.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  gameMode = elements.gameMode.value;
  localStorage.setItem('gardner-game-mode', gameMode);
  syncModeControls();
  stopManualAnalysisForPlayMode();
  if (gameMode === 'local') releasePlayClient();
  else ensurePlayClient();
  if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK);
  toast(`Mode changed to ${elements.gameMode.selectedOptions[0]?.textContent || gameMode}.`);
  updateUI();
});

elements.humanSide.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  humanSide = elements.humanSide.value === COLORS.BLACK ? COLORS.BLACK : COLORS.WHITE;
  localStorage.setItem('gardner-human-side', humanSide);
  syncModeControls();
  stopManualAnalysisForPlayMode();
  if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK);
  updateUI();
});


elements.engineKernel.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  engineKernel = ENGINE_KERNEL_OPTIONS.some(option => option.id === elements.engineKernel.value) ? elements.engineKernel.value : 'minifish-js';
  localStorage.setItem('gardner-engine-kernel', engineKernel);
  elements.engineKernel.value = engineKernel;
  analysisResult = null;
  lastAnalysisKey = '';
  if (analysisEnabled && !analysisPaused) restartAnalysis(true);
  toast(`Engine selected: ${selectedEngineKernelLabel()}.`);
  updateUI();
});

elements.difficulty.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  aiStyle = AI_STYLES.some(style => style.id === elements.difficulty.value) ? elements.difficulty.value : 'balanced';
  localStorage.setItem('gardner-ai-style', aiStyle);
  const style = AI_STYLES.find(item => item.id === aiStyle) || AI_STYLES[0];
  elements.playEngineStatus.textContent = style.shortLabel;
  toast(`${style.label} AI selected · ${style.description}`);
  updateUI();
});

elements.aiThinkTime.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  aiThinkMs = normalizeAiThinkMs(elements.aiThinkTime.value);
  localStorage.setItem('gardner-ai-think-ms', String(aiThinkMs));
  elements.aiThinkTime.value = String(aiThinkMs);
  toast(`AI thinking time set to ${formatThinkTime(aiThinkMs)}.`);
  updateUI();
});

elements.startLayout.addEventListener('change', () => {
  startLayout = START_LAYOUTS.some(layout => layout.id === elements.startLayout.value) ? elements.startLayout.value : 'standard';
  localStorage.setItem('gardner-start-layout', startLayout);
  startNewGame();
});

elements.copyFen.addEventListener('click', () => copyText(game.current.position.toStandardFEN(), 'Standard FEN copied.'));

function preventMovePanelToggle(event) {
  event.preventDefault();
  event.stopPropagation();
}

elements.copyLine.addEventListener('click', event => {
  preventMovePanelToggle(event);
  copyText(exportCurrentLineMovetext(game), 'Current line copied as standard PGN movetext.');
});
elements.copyTree.addEventListener('click', event => {
  preventMovePanelToggle(event);
  copyText(exportGameTreePGN(game), 'Complete game tree copied as PGN variations.');
});

elements.editorDialog.querySelectorAll('input[name="editorTurn"]').forEach(input => {
  input.addEventListener('change', () => {
    syncEditorTurn();
    clearEditorMoveSelection({ render: false });
    editorBoardView.render();
  });
});
elements.editorClear.addEventListener('click', () => {
  syncEditorTurn();
  editorPosition = Position.empty(editorPosition.turn);
  clearEditorMoveSelection({ render: false });
  editorBoardView.render();
  clearEditorErrors();
});
elements.editorCopyFen.addEventListener('click', () => {
  syncEditorTurn();
  copyText(editorPosition.toStandardFEN(), 'Editor FEN copied.');
});
elements.editorStart.addEventListener('click', () => {
  editorPosition = currentStart.position.clone();
  editorTool = 'move';
  elements.editorDialog.querySelector('input[name="editorTurn"][value="w"]').checked = true;
  elements.fenInput.value = editorPosition.toStandardFEN();
  clearEditorMoveSelection({ render: false });
  updatePaletteSelection();
  editorBoardView.render();
  clearEditorErrors();
});
elements.loadFen.addEventListener('click', () => {
  try {
    editorPosition = Position.fromFEN(elements.fenInput.value);
    elements.editorDialog.querySelector(`input[name="editorTurn"][value="${editorPosition.turn}"]`).checked = true;
    clearEditorMoveSelection({ render: false });
    editorBoardView.render();
    clearEditorErrors();
  } catch (error) {
    showEditorErrors([error.message]);
  }
});
elements.applyEditor.addEventListener('click', () => {
  syncEditorTurn();
  const errors = validateEditedPosition(editorPosition, Rules);
  if (errors.length) {
    showEditorErrors(errors);
    return;
  }
  preferredMatchId = null;
  newGameUndoSnapshot = null;
  game.reset(editorPosition);
  boardView.clearSelection(false);
  elements.editorDialog.close('apply');
  toast('The edited position is now active.');
  updateUI();
});

document.addEventListener('keydown', event => {
  if (event.target.matches('textarea, input, select') || elements.editorDialog.open || elements.promotionDialog.open) return;
  if (event.key === 'ArrowLeft') elements.undo.click();
  if (event.key === 'ArrowRight') elements.redo.click();
  if (event.key.toLowerCase() === 'f') elements.flip.click();
  if (event.key.toLowerCase() === 'a') elements.analysis.click();
  if (event.key.toLowerCase() === 'b') elements.book.click();
});


window.addEventListener('error', event => {
  const message = event?.message || 'Unexpected browser error.';
  console.error(event?.error || message);
  try {
    toast(`Engine/UI error: ${message}. Stop and restart AI if play is stuck.`);
  } catch {}
});

window.addEventListener('unhandledrejection', event => {
  const message = event?.reason?.message || String(event?.reason || 'Unhandled async error.');
  console.error(event?.reason || message);
  try {
    toast(`Async engine error: ${message}. Stop and restart AI if play is stuck.`);
  } catch {}
});

window.addEventListener('beforeunload', () => {
  saveGameState();
  try { analysisCache.flush(); } catch {}
});


buildBoardStyleSelect();
buildPieceStyleSelect();
buildEngineKernelSelect();
buildDifficultySelect();
buildAiThinkTimeSelect();
buildStartLayoutSelect();
restoreSavedGameState();
editorPosition = game.current.position.clone();
syncModeControls();
if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK, false);
buildPalette();
analysisPanel.renderIdle();
elements.pauseAnalysis.disabled = true;
const movePanel = document.querySelector('#movePanel');
const mobileLayout = window.matchMedia('(max-width: 640px)');
function syncMobileMovePanel(event = mobileLayout) {
  if (!movePanel) return;
  if (event.matches && !movePanel.dataset.mobileInitialized) {
    movePanel.open = true;
    movePanel.dataset.mobileInitialized = 'true';
  } else if (!event.matches) {
    movePanel.open = true;
    delete movePanel.dataset.mobileInitialized;
  }
}
syncMobileMovePanel();
mobileLayout.addEventListener?.('change', syncMobileMovePanel);
elements.workspace?.classList.toggle('tree-collapsed', !elements.gameTreePanel?.open);
elements.gameTreePanel?.addEventListener('toggle', async () => {
  const open = elements.gameTreePanel.open;
  elements.workspace?.classList.toggle('tree-collapsed', !open);
  if (!open) return;
  refreshStudyMatch();
  await ensureLibraryLoaded();
  refreshStudyMatch();
});
updateUI();
