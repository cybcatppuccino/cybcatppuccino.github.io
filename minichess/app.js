import {
  COLORS,
  INITIAL_STUDY_FEN,
  PIECE_NAMES,
  TYPES,
  parseStudyCoord
} from './js/core/constants.js';
import { GameTree } from './js/core/game-tree.js';
import { Position, validateEditedPosition } from './js/core/position.js';
import { StudyLibrary, parsePGN, pathToNode } from './js/core/pgn.js';
import { Rules, gameStatus, legalMoves } from './js/core/rules.js';
import { moveToSAN, moveToUci } from './js/core/notation.js';
import { AnalysisCache, buildAnalysisKey } from './js/engine/analysis-cache.js';
import { AnalysisClient } from './js/engine/client.js';
import { AI_LEVELS } from './js/engine/difficulty.js';
import { PlayEngineClient } from './js/engine/play-client.js';
import { AnalysisPanelView } from './js/ui/analysis-panel.js';
import { BoardView } from './js/ui/board.js';
import { MoveListView } from './js/ui/move-list.js';
import { PIECE_STYLES, applyPieceStyle } from './js/ui/pieces.js';
import { StudyTreeView } from './js/ui/tree-view.js';

const $ = selector => document.querySelector(selector);
const game = new GameTree(Position.initial());
const library = new StudyLibrary();
let activeStudy = null;
let activeMatches = [];
let matchIndex = 0;
let preferredMatchId = null;
let toastTimer = null;
let editorPosition = Position.initial();
let editorPiece = { color: COLORS.WHITE, type: TYPES.QUEEN };
let showBook = false;
let analysisEnabled = false;
let analysisPaused = false;
let analysisResult = null;
let lastAnalysisKey = '';
let currentAnalysisKey = '';
let gameMode = localStorage.getItem('gardner-game-mode') || 'local';
let humanSide = localStorage.getItem('gardner-human-side') || COLORS.WHITE;
let aiLevel = Math.max(1, Math.min(10, Number(localStorage.getItem('gardner-ai-level') || 5)));
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
  humanSideField: $('#humanSideField'),
  difficultyField: $('#difficultyField'),
  playEngineStatus: $('#playEngineStatus'),
  modePill: $('#modePill'),
  copyFen: $('#copyFenButton'),
  libraryStatus: $('#libraryStatus'),
  studySelect: $('#studySelect'),
  matchStatus: $('#matchStatus'),
  matchCounter: $('#matchCounter'),
  previousMatch: $('#previousMatch'),
  nextMatch: $('#nextMatch'),
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
  toast: $('#toast')
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
  preferredMatchId = node.id;
  const path = pathToNode(node);
  game.importPath(path, node);
  boardView.clearSelection(false);
  toast(`Loaded ${node.parent ? node.san : 'the starting position'} from ${studyTitle(node.source)}.`);
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
    const thinking = state === 'thinking';
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
      elements.playEngineStatus.textContent = `L${aiLevel} thinking`;
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

function studyTitle(id) {
  return library.studies.find(study => study.sourceName === id)?.title || id || 'the study archive';
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

function cachePrincipalVariationChildren(result) {
  if (!result?.lines?.length) return;
  const rootPosition = game.current.position;
  const rootHistory = engineHistoryFens();
  for (const line of result.lines) {
    const pv = Array.isArray(line.pv) ? line.pv : [];
    if (pv.length < 2) continue;
    let cursor = rootPosition.clone();
    const history = [...rootHistory];
    // Seed a bounded corridor, not only the immediate child. This lets a user
    // follow several already-calculated PV plies while preserving useful work.
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
        analysisCache.set(childKey, {
          ...result,
          depth: childDepth,
          selDepth: Math.max(childDepth, Number(result.selDepth || childDepth) - offset - 1),
          nodes: 0,
          elapsed: 0,
          nps: 0,
          cached: true,
          searchDepth: childDepth + 1,
          nextDepth: childDepth + 1,
          lines: [{ ...line, move: continuation[0], pv: continuation }]
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
  const key = `${context.key}|${gameMode}|${humanSide}|L${aiLevel}`;
  if (aiThinking && aiRequestKey === key) return true;
  cancelAiTurn({ quiet: true });
  aiThinking = true;
  aiRequestKey = key;
  elements.playEngineStatus.textContent = `L${aiLevel} queued`;
  elements.playEngineStatus.className = 'play-engine-status thinking';
  const delay = gameMode === 'ai-ai' ? 320 : 180;
  aiTimer = setTimeout(() => {
    aiTimer = null;
    if (!aiThinking || aiRequestKey !== key) return;
    playClient.search({
      fen: game.current.position.toCompactFEN(),
      historyFens: engineHistoryFens(),
      level: aiLevel,
      cacheKey: context.key,
      resumeResult: analysisCache.get(context.key)
    });
  }, delay);
  return true;
}

function handleAiMoveResult(result) {
  const expectedKey = `${currentAnalysisContext().key}|${gameMode}|${humanSide}|L${aiLevel}`;
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
  elements.playEngineStatus.textContent = `L${aiLevel} · d${result.depth || 0}`;
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
  for (const config of AI_LEVELS) {
    const option = document.createElement('option');
    option.value = String(config.level);
    option.textContent = config.label;
    option.selected = config.level === aiLevel;
    elements.difficulty.appendChild(option);
  }
}

function syncModeControls() {
  if (!['local', 'human-ai', 'ai-ai'].includes(gameMode)) gameMode = 'local';
  if (![COLORS.WHITE, COLORS.BLACK].includes(humanSide)) humanSide = COLORS.WHITE;
  elements.gameMode.value = gameMode;
  elements.humanSide.value = humanSide;
  elements.difficulty.value = String(aiLevel);
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
  return library.bookMoves(game.current.position);
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
  const cached = analysisCache.get(context.key);
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
  if (!activeStudy) return;
  activeMatches = library.matches(game.current.position).filter(node => node.source === activeStudy.sourceName);
  const preferredIndex = preferredMatchId ? activeMatches.findIndex(node => node.id === preferredMatchId) : -1;
  if (preferredIndex >= 0) matchIndex = preferredIndex;
  else if (matchIndex >= activeMatches.length) matchIndex = 0;
  const match = activeMatches[matchIndex] || null;

  if (match) {
    elements.matchStatus.classList.add('matched');
    elements.matchStatus.innerHTML = `<i></i> Exact archive match at ply ${match.ply}`;
    elements.matchCounter.textContent = `${matchIndex + 1} / ${activeMatches.length}`;
  } else {
    elements.matchStatus.classList.remove('matched');
    elements.matchStatus.innerHTML = '<i></i> No exact node in this source — showing its opening root';
    elements.matchCounter.textContent = '0 / 0';
  }
  elements.previousMatch.disabled = activeMatches.length < 2;
  elements.nextMatch.disabled = activeMatches.length < 2;
  studyTreeView.render(activeStudy.root, match || activeStudy.root, game.current.position.canonicalKey());
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
  try {
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

    elements.studySelect.innerHTML = '';
    library.studies.forEach((study, index) => {
      const option = document.createElement('option');
      option.value = study.sourceName;
      option.textContent = study.title;
      if (index === 0) option.selected = true;
      elements.studySelect.appendChild(option);
    });
    activeStudy = library.studies[0] || null;
    elements.libraryStatus.textContent = `${totalMoves.toLocaleString()} nodes`;
    elements.libraryStatus.className = 'loading-dot ready';
    if (totalErrors) {
      console.warn(`PGN archive loaded with ${totalErrors} skipped tokens.`, library.studies.map(s => ({ source: s.sourceName, errors: s.errors.slice(0, 12) })));
    }
    updateUI();
    if (analysisEnabled) restartAnalysis(true);
  } catch (error) {
    console.error(error);
    elements.libraryStatus.textContent = 'Unavailable';
    elements.libraryStatus.className = 'loading-dot error';
    studyTreeView.clear('The PGN archive could not be loaded.');
    $('#treeEmpty').innerHTML += '<span>Run this project through a local web server instead of opening index.html directly.</span>';
    toast('Start a local web server to load the PGN archive.');
  }
}

elements.undo.addEventListener('click', () => {
  cancelAiTurn({ quiet: true });
  if (game.undo()) {
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
elements.edit.addEventListener('click', () => {
  cancelAiTurn({ quiet: true });
  openEditor();
});
elements.newGame.addEventListener('click', () => {
  cancelAiTurn({ quiet: true });
  preferredMatchId = null;
  game.reset(Position.initial());
  boardView.clearSelection(false);
  toast('A new Gardner game is ready.');
  updateUI();
});

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

elements.book.addEventListener('click', () => {
  showBook = !showBook;
  elements.book.setAttribute('aria-pressed', String(showBook));
  elements.book.classList.toggle('active', showBook);
  elements.book.innerHTML = `<span class="button-icon">⌁</span> ${showBook ? 'Hide book' : 'Show book'}`;
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
  aiLevel = Math.max(1, Math.min(10, Number(elements.difficulty.value || 5)));
  localStorage.setItem('gardner-ai-level', String(aiLevel));
  elements.playEngineStatus.textContent = `Level ${aiLevel}`;
  updateUI();
});

elements.copyFen.addEventListener('click', () => copyText(game.current.position.toStudyFEN(), 'Study FEN copied.'));

elements.studySelect.addEventListener('change', () => {
  preferredMatchId = null;
  activeStudy = library.studies.find(study => study.sourceName === elements.studySelect.value) || library.studies[0];
  matchIndex = 0;
  refreshStudyMatch();
});
elements.previousMatch.addEventListener('click', () => {
  if (!activeMatches.length) return;
  matchIndex = (matchIndex - 1 + activeMatches.length) % activeMatches.length;
  refreshStudyMatch();
});
elements.nextMatch.addEventListener('click', () => {
  if (!activeMatches.length) return;
  matchIndex = (matchIndex + 1) % activeMatches.length;
  refreshStudyMatch();
});

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
  editorPosition = Position.initial();
  elements.editorDialog.querySelector('input[name="editorTurn"][value="w"]').checked = true;
  elements.fenInput.value = INITIAL_STUDY_FEN;
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
syncModeControls();
if (gameMode === 'human-ai') boardView.setFlipped(humanSide === COLORS.BLACK, false);
buildPalette();
analysisPanel.renderIdle();
elements.pauseAnalysis.disabled = true;
updateUI();
loadLibrary();
