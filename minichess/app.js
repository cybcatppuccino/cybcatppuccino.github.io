import {
  COLORS,
  PIECE_NAMES,
  TYPES,
  parseStudyCoord
} from './js/core/constants.js';
import { GameTree } from './js/core/game-tree.js';
import { Position, validateEditedPosition } from './js/core/position.js';
import { StudyLibrary, parsePGN } from './js/core/pgn.js';
import { Rules, gameStatus, legalMoves } from './js/core/rules.js';
import { START_LAYOUTS, createStartPosition } from './js/core/start-positions.js';
import { moveToSAN, moveToUci } from './js/core/notation.js';
import { AnalysisCache, buildAnalysisKey, rebaseVerifiedMateLine } from './js/engine/analysis-cache.js';
import { AnalysisClient } from './js/engine/client.js';
import { AI_STYLES } from './js/engine/difficulty.js';
import { EnginePosition, validateMateResult } from './js/engine/engine.js';
import { PlayEngineClient } from './js/engine/play-client.js';
import { AnalysisPanelView } from './js/ui/analysis-panel.js';
import { BoardView } from './js/ui/board.js';
import { MoveListView } from './js/ui/move-list.js';
import { PIECE_STYLES, applyPieceStyle } from './js/ui/pieces.js';
import { StudyTreeView } from './js/ui/tree-view.js';

const $ = selector => document.querySelector(selector);
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
let showBook = false;
let analysisEnabled = false;
let analysisPaused = false;
let analysisResult = null;
let lastAnalysisKey = '';
let currentAnalysisKey = '';
let gameMode = localStorage.getItem('gardner-game-mode') || 'local';
let humanSide = localStorage.getItem('gardner-human-side') || COLORS.WHITE;
let aiStyle = localStorage.getItem('gardner-ai-style') || 'balanced';
if (!AI_STYLES.some(style => style.id === aiStyle)) aiStyle = 'balanced';
let aiThinking = false;
let aiRequestKey = '';
let aiTimer = null;
const analysisCache = new AnalysisCache();
let pieceStyle = localStorage.getItem('gardner-piece-style') || 'standard';
if (!PIECE_STYLES.some(style => style.id === pieceStyle)) pieceStyle = 'standard';

const elements = {
  statusText: $('#statusText'),
  turnToken: $('#turnToken'),
  moveCount: $('#moveCount'),
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
  effort: $('#effortSelect'),
  pauseAnalysis: $('#pauseAnalysisButton'),
  resumeAnalysis: $('#resumeAnalysisButton'),
  gameMode: $('#gameModeSelect'),
  humanSide: $('#humanSideSelect'),
  difficulty: $('#difficultySelect'),
  startLayout: $('#startLayoutSelect'),
  humanSideField: $('#humanSideField'),
  difficultyField: $('#difficultyField'),
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

const editorBoardView = new BoardView(elements.editorBoard, {
  getPosition: () => editorPosition,
  getLegalMoves: () => [],
  onAttemptMove: async () => {},
  onEditorSquare: sq => {
    editorPosition.setPiece(sq, editorPiece);
    editorBoardView.render();
    clearEditorErrors();
  }
});
editorBoardView.setPieceStyle(pieceStyle, false);
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

const analysisClient = new AnalysisClient({
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
    const key = result.cacheKey || currentAnalysisKey;
    if (key) analysisCache.set(key, result);
    if (key && key !== currentAnalysisKey) return;
    analysisResult = result;
    cachePrincipalVariationChildren(result);
    analysisPanel.render(result, formatAnalysisLines(result), {
      state: analysisPaused ? 'paused' : analysisEnabled ? '' : 'stopped'
    });
    boardView.setArrows(composeBoardArrows());
    refreshStudyMatch();
  },
  onError: message => {
    console.error(message);
    analysisPanel.renderError(message);
    toast('The local analysis worker could not continue.');
  }
});

const playClient = new PlayEngineClient({
  onState: message => {
    if (message.state === 'thinking') {
      elements.playEngineStatus.textContent = `${AI_STYLES.find(item => item.id === aiStyle)?.shortLabel || 'AI'} thinking`;
      elements.playEngineStatus.className = 'play-engine-status thinking';
    }
  },
  onResult: result => handleAiMoveResult(result),
  onError: message => {
    console.error(message);
    aiThinking = false;
    aiRequestKey = '';
    elements.playEngineStatus.textContent = 'AI error';
    elements.playEngineStatus.className = 'play-engine-status error';
    toast('The play engine could not complete its move.');
    updateUI();
  }
});

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

function sideIsAI(color) {
  if (gameMode === 'ai-ai') return true;
  if (gameMode === 'human-ai') return color !== humanSide;
  return false;
}

function canHumanMove() {
  return !aiThinking && !sideIsAI(game.current.position.turn) && !isFinished(currentStatus());
}

function currentAnalysisContext(position = game.current.position, historyFens = engineHistoryFens()) {
  return {
    historyFens,
    key: buildAnalysisKey(position, historyFens)
  };
}

function cancelAiTurn({ quiet = false } = {}) {
  clearTimeout(aiTimer);
  aiTimer = null;
  playClient.cancel();
  aiThinking = false;
  aiRequestKey = '';
  if (!quiet) {
    elements.playEngineStatus.textContent = 'Ready';
    elements.playEngineStatus.className = 'play-engine-status';
  }
}

function validateCachedAnalysis(position, key, cached) {
  if (!cached?.lines?.length) return cached || null;
  const enginePosition = EnginePosition.fromFEN(position.toCompactFEN());
  const lines = cached.lines.filter(line => !line.mateVerified || validateMateResult(enginePosition, line));
  // The leading line defines a solved result. If it no longer replays to mate
  // from this exact root, discard the entry instead of silently downgrading it.
  if (cached.lines[0]?.mateVerified && lines[0] !== cached.lines[0]) {
    analysisCache.delete(key);
    return null;
  }
  return {
    ...cached,
    lines,
    solved: Boolean(cached.tablebase || cached.fortressProof || lines[0]?.mateVerified || cached.endgameProof)
  };
}

function cachePrincipalVariationChildren(result) {
  if (!result?.lines?.length || (result.tablebase && !result.lines[0]?.mateVerified) || result.fortressProof) return;
  const rootPosition = game.current.position;
  const rootHistory = engineHistoryFens();
  for (const line of result.lines) {
    const pv = Array.isArray(line.pv) ? line.pv : [];
    if (pv.length < 2) continue;
    let cursor = rootPosition.clone();
    const history = [...rootHistory];
    // Seed a bounded corridor, not only the immediate child. Verified mate
    // distances are rebased at every child so the cache remains replay-valid.
    const corridor = Math.min(pv.length - 1, 10);
    for (let offset = 0; offset < corridor; offset += 1) {
      const move = findMoveByUci(cursor, pv[offset]);
      if (!move) break;
      const parentFen = cursor.toCompactFEN();
      const child = cursor.makeMove(move);
      history.push(parentFen);
      const continuation = pv.slice(offset + 1);
      if (continuation.length) {
        const childKey = buildAnalysisKey(child, history);
        const childDepth = Math.max(1, Number(result.depth || 1) - offset - 1);
        const consumed = offset + 1;
        const childLine = line.mateVerified
          ? rebaseVerifiedMateLine(line, consumed)
          : { ...line, move: continuation[0], pv: continuation };
        if (!childLine) {
          cursor = child;
          continue;
        }
        if (childLine.mateVerified) {
          const engineChild = EnginePosition.fromFEN(child.toCompactFEN());
          if (!validateMateResult(engineChild, childLine)) {
            cursor = child;
            continue;
          }
        }
        analysisCache.set(childKey, {
          ...result,
          depth: childDepth,
          selDepth: Math.max(childDepth, Number(result.selDepth || childDepth) - consumed),
          nodes: 0,
          elapsed: 0,
          nps: 0,
          cached: true,
          solved: Boolean(childLine.mateVerified || result.tablebase || result.fortressProof),
          endgameProof: Boolean(childLine.endgameProof),
          tablebase: Boolean(result.tablebase),
          tablebaseSource: result.tablebaseSource || '',
          fortressProof: Boolean(result.fortressProof),
          searchDepth: childLine.mateVerified ? childDepth : childDepth + 1,
          nextDepth: childLine.mateVerified ? childDepth : childDepth + 1,
          lines: [childLine]
        });
      }
      cursor = child;
    }
  }
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
  const key = `${context.key}|${gameMode}|${humanSide}|S${aiStyle}`;
  if (aiThinking && aiRequestKey === key) return true;
  cancelAiTurn({ quiet: true });
  aiThinking = true;
  aiRequestKey = key;
  elements.playEngineStatus.textContent = `${AI_STYLES.find(item => item.id === aiStyle)?.shortLabel || 'AI'} queued`;
  elements.playEngineStatus.className = 'play-engine-status thinking';
  const delay = gameMode === 'ai-ai' ? 320 : 180;
  aiTimer = setTimeout(() => {
    aiTimer = null;
    if (!aiThinking || aiRequestKey !== key) return;
    playClient.search({
      fen: game.current.position.toCompactFEN(),
      historyFens: engineHistoryFens(),
      style: aiStyle,
      cacheKey: context.key,
      resumeResult: analysisCache.get(context.key)
    });
  }, delay);
  return true;
}

function handleAiMoveResult(result) {
  const expectedKey = `${currentAnalysisContext().key}|${gameMode}|${humanSide}|S${aiStyle}`;
  if (!aiThinking || aiRequestKey !== expectedKey || !sideIsAI(game.current.position.turn)) return;
  const context = currentAnalysisContext();
  const stored = {
    ...result,
    cacheKey: context.key,
    cached: true,
    searchDepth: Math.max(1, Number(result.depth || 0) + 1),
    nextDepth: Math.max(1, Number(result.depth || 0) + 1)
  };
  analysisCache.set(context.key, stored);
  analysisResult = stored;
  cachePrincipalVariationChildren(stored);
  const move = findMoveByUci(game.current.position, result.selectedMove || result.lines?.[0]?.move);
  aiThinking = false;
  aiRequestKey = '';
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

function startNewGame({ announce = true } = {}) {
  cancelAiTurn({ quiet: true });
  preferredMatchId = null;
  currentStart = createStartPosition(startLayout);
  game.reset(currentStart.position);
  analysisResult = null;
  lastAnalysisKey = '';
  boardView.clearSelection(false);
  if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK, false);
  if (announce) toast(`${START_LAYOUTS.find(layout => layout.id === startLayout)?.label || 'Selected'} layout · ${currentStart.signature}.`);
  updateUI();
}

function syncModeControls() {
  if (!['local', 'human-ai', 'ai-ai'].includes(gameMode)) gameMode = 'local';
  if (![COLORS.WHITE, COLORS.BLACK].includes(humanSide)) humanSide = COLORS.WHITE;
  elements.gameMode.value = gameMode;
  elements.humanSide.value = humanSide;
  elements.difficulty.value = aiStyle;
  elements.startLayout.value = startLayout;
  elements.humanSideField.classList.toggle('is-hidden', gameMode !== 'human-ai');
  elements.difficultyField.classList.toggle('is-hidden', gameMode === 'local');
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
  return ['checkmate', 'stalemate', 'draw-insufficient', 'draw-50', 'draw-repetition'].includes(status.state);
}

function currentBookEntries() {
  return libraryLoaded ? library.bookMoves(game.current.position) : [];
}

function arrowFromUci(uci, kind, title = '') {
  const match = String(uci || '').match(/^([b-f][2-6])([b-f][2-6])/i);
  if (!match) return null;
  const from = parseStudyCoord(match[1].toLowerCase());
  const to = parseStudyCoord(match[2].toLowerCase());
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
  if (analysisResult?.lines?.[0]?.pv?.length) {
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
  return (result.lines || []).map(line => {
    const pvSan = formatPV(game.current.position, line.pv);
    return {
      ...line,
      firstSan: pvSan[0] || line.move,
      pvSan: pvSan.join(' ')
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
  const context = currentAnalysisContext();
  currentAnalysisKey = context.key;
  const cached = validateCachedAnalysis(game.current.position, context.key, analysisCache.get(context.key));
  if (cached) {
    analysisResult = cached;
    analysisPanel.render(cached, formatAnalysisLines(cached), {
      state: analysisPaused ? 'paused' : analysisEnabled ? 'thinking' : 'stopped'
    });
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
  if (!force && context.key === lastAnalysisKey && analysisClient.active) return;
  lastAnalysisKey = context.key;
  analysisClient.update({
    fen: game.current.position.toCompactFEN(),
    bookMoves: [],
    historyFens: context.historyFens,
    effortMs: Number(elements.effort.value),
    multipv: 3,
    cacheKey: context.key,
    resumeResult: cached
  });
}

function updateUI() {
  const position = game.current.position;
  const status = currentStatus();
  const finished = isFinished(status);
  const colorName = position.turn === COLORS.WHITE ? 'White' : 'Black';

  restartAnalysis();
  const aiTurn = maybeStartAiTurn(status);

  const messages = {
    playing: `${colorName} to move`,
    check: `${colorName} is in check`,
    checkmate: `Checkmate — ${status.winner === COLORS.WHITE ? 'White' : 'Black'} wins`,
    stalemate: 'Draw by stalemate',
    'draw-insufficient': 'Draw by insufficient material',
    'draw-50': 'Draw by the 50-move rule',
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
  elements.fenDisplay.textContent = position.toStudyFEN();
  elements.undo.disabled = !game.current.parent;
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
        editorPiece = { color, type };
        updatePaletteSelection();
      });
      elements.piecePalette.appendChild(button);
    }
  }
  const eraser = document.createElement('button');
  eraser.type = 'button';
  eraser.className = 'palette-piece eraser';
  eraser.textContent = '⌫  Erase square';
  eraser.dataset.eraser = 'true';
  eraser.addEventListener('click', () => {
    editorPiece = null;
    updatePaletteSelection();
  });
  elements.piecePalette.appendChild(eraser);
  updatePaletteSelection();
}

function updatePaletteSelection() {
  elements.piecePalette.querySelectorAll('.palette-piece').forEach(button => {
    const selected = editorPiece
      ? button.dataset.color === editorPiece.color && button.dataset.type === editorPiece.type
      : button.dataset.eraser === 'true';
    button.classList.toggle('selected', selected);
  });
}

function openEditor() {
  editorPosition = game.current.position.clone();
  editorPiece = { color: COLORS.WHITE, type: TYPES.QUEEN };
  elements.fenInput.value = editorPosition.toStudyFEN();
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
      const study = parsePGN(text, source.id);
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
  if (gameMode === 'human-ai') {
    if (game.undo()) changed = true;
    while (changed && game.current.parent && game.current.position.turn !== humanSide) game.undo();
  } else {
    changed = Boolean(game.undo());
  }
  if (changed) {
    preferredMatchId = null;
    boardView.clearSelection(false);
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
    restartAnalysis(true);
    toast('Local continuous analysis started.');
  } else {
    analysisClient.stop();
    lastAnalysisKey = '';
    if (analysisResult) analysisPanel.renderStopped(analysisResult, formatAnalysisLines(analysisResult));
    else analysisPanel.renderIdle();
    boardView.setArrows(composeBoardArrows());
    toast('Analysis stopped; the latest result remains cached.');
  }
});

elements.pauseAnalysis.addEventListener('click', () => {
  if (!analysisEnabled || analysisPaused) return;
  analysisPaused = true;
  analysisClient.pause();
  elements.pauseAnalysis.hidden = true;
  elements.resumeAnalysis.hidden = false;
  if (analysisResult) analysisPanel.render(analysisResult, formatAnalysisLines(analysisResult), { state: 'paused' });
  else analysisPanel.setState('paused');
  toast('Analysis paused. The current search result is preserved.');
});

elements.resumeAnalysis.addEventListener('click', () => {
  if (!analysisEnabled || !analysisPaused) return;
  analysisPaused = false;
  elements.pauseAnalysis.hidden = false;
  elements.resumeAnalysis.hidden = true;
  if (analysisClient.active && currentAnalysisKey === lastAnalysisKey) analysisClient.resume();
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

elements.effort.addEventListener('change', () => {
  if (analysisEnabled && !analysisPaused) restartAnalysis(true);
});

elements.gameMode.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  gameMode = elements.gameMode.value;
  localStorage.setItem('gardner-game-mode', gameMode);
  syncModeControls();
  if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK);
  toast(`Mode changed to ${elements.gameMode.selectedOptions[0]?.textContent || gameMode}.`);
  updateUI();
});

elements.humanSide.addEventListener('change', () => {
  cancelAiTurn({ quiet: true });
  humanSide = elements.humanSide.value === COLORS.BLACK ? COLORS.BLACK : COLORS.WHITE;
  localStorage.setItem('gardner-human-side', humanSide);
  syncModeControls();
  if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK);
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

elements.startLayout.addEventListener('change', () => {
  startLayout = START_LAYOUTS.some(layout => layout.id === elements.startLayout.value) ? elements.startLayout.value : 'standard';
  localStorage.setItem('gardner-start-layout', startLayout);
  startNewGame();
});

elements.copyFen.addEventListener('click', () => copyText(game.current.position.toStudyFEN(), 'Study FEN copied.'));

elements.editorDialog.querySelectorAll('input[name="editorTurn"]').forEach(input => {
  input.addEventListener('change', () => {
    syncEditorTurn();
    editorBoardView.render();
  });
});
elements.editorClear.addEventListener('click', () => {
  syncEditorTurn();
  editorPosition = Position.empty(editorPosition.turn);
  editorBoardView.render();
  clearEditorErrors();
});
elements.editorCopyFen.addEventListener('click', () => {
  syncEditorTurn();
  copyText(editorPosition.toStudyFEN(), 'Editor FEN copied.');
});
elements.editorStart.addEventListener('click', () => {
  editorPosition = currentStart.position.clone();
  elements.editorDialog.querySelector('input[name="editorTurn"][value="w"]').checked = true;
  elements.fenInput.value = editorPosition.toStudyFEN();
  editorBoardView.render();
  clearEditorErrors();
});
elements.loadFen.addEventListener('click', () => {
  try {
    editorPosition = Position.fromFEN(elements.fenInput.value);
    elements.editorDialog.querySelector(`input[name="editorTurn"][value="${editorPosition.turn}"]`).checked = true;
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

buildPieceStyleSelect();
buildDifficultySelect();
buildStartLayoutSelect();
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
    movePanel.open = false;
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
