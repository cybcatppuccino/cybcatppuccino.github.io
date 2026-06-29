import {
  EngineInternals,
  EnginePosition,
  evaluate,
  generateLegalMoves,
  isInCheck,
  isInsufficientMaterial,
  moveToUci,
  scoreToDisplay,
  staticExchangeEval
} from './engine.js';

const {
  PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  PIECE_VALUE, MATE, MATE_BOUND, TB_WIN_SCORE, TB_WIN_MIN_SCORE,
  makeMove, undoMove, moveFrom, moveTo, movePromotion, sideOf, typeOf,
  givesCheck, isCapture, isPromotion, rankOf
} = EngineInternals;

export const MINIFISH_VERSION = 'Minifish JS 23';

const INF = 32000;
const DRAW = 0;
const MAX_PLY = 96;
const PROGRESS_NODE_INTERVAL = 8192;
const PROGRESS_MIN_INTERVAL_MS = 120;
const ABORT = Object.freeze({ aborted: true });

function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function isMateScore(score) { return Math.abs(Number(score || 0)) >= MATE_BOUND; }
function isTablebaseBoundScore(score) {
  const value = Math.abs(Number(score));
  return Number.isFinite(value) && value >= TB_WIN_MIN_SCORE && value <= TB_WIN_SCORE;
}
function mateDistancePly(score) { return isMateScore(score) ? Math.max(1, MATE - Math.abs(score)) : 0; }
function rootScoreToWhite(score, rootSide) { return rootSide === WHITE ? score : -score; }
function sideScoreToWhite(pos, score) { return pos.turn === WHITE ? score : -score; }

function cloneLine(line) {
  return line ? { ...line, pv: Array.isArray(line.pv) ? line.pv.slice() : [] } : null;
}

function immediateTerminalScore(pos, ply) {
  const legal = generateLegalMoves(pos, false);
  if (legal.length) return null;
  return {
    score: isInCheck(pos) ? -MATE + ply : DRAW,
    legal
  };
}

function probeToScore(probe, ply = 0) {
  if (!probe || probe.wdl === 2 || probe.wdl === undefined) return null;
  const wdl = Number(probe.wdl || 0);
  if (wdl === 0) return DRAW;
  const plyOffset = Math.min(MAX_PLY, Math.max(0, Math.floor(Number(ply) || 0)));
  const dtm = Math.max(1, Number(probe.dtmPly || 0) + plyOffset);
  if (probe.exactDtm || probe.dtmUpperBound === false) return wdl > 0 ? MATE - dtm : -MATE + dtm;
  return wdl > 0 ? TB_WIN_SCORE - Math.min(MAX_PLY, plyOffset) : -TB_WIN_SCORE + Math.min(MAX_PLY, plyOffset);
}

function materialKey(pos) {
  let key = 0;
  for (const piece of pos.board) if (piece) key += (typeOf(piece) * 2 + (sideOf(piece) === WHITE ? 0 : 1));
  return key;
}

function moveTypeScore(pos, move) {
  const movingType = typeOf(pos.board[moveFrom(move)]);
  const target = pos.board[moveTo(move)];
  const capturedType = typeOf(target);
  const promo = movePromotion(move);
  let score = 0;
  if (promo) score += 900_000 + PIECE_VALUE[promo] * 20;
  if (target) score += 500_000 + PIECE_VALUE[capturedType] * 24 - PIECE_VALUE[movingType] * 4;
  if (givesCheck(pos, move)) score += 360_000;
  if (movingType === PAWN && (rankOf(moveTo(move)) === 1 || rankOf(moveTo(move)) === 3)) score += 30_000;
  return score;
}

function pvEndsInMate(root, pv, rootScore) {
  if (!isMateScore(rootScore) || !Array.isArray(pv) || !pv.length) return false;
  const targetPlies = mateDistancePly(rootScore);
  if (pv.length < targetPlies) return false;
  const cursor = root.clone();
  for (let ply = 0; ply < targetPlies; ply += 1) {
    const move = pv[ply];
    const legal = generateLegalMoves(cursor, false);
    if (!legal.includes(move)) return false;
    makeMove(cursor, move);
  }
  return isInCheck(cursor) && generateLegalMoves(cursor, false).length === 0;
}

export class MinifishSearcher {
  constructor({ tablebaseProbe = null } = {}) {
    this.tablebaseProbe = typeof tablebaseProbe === 'function' ? tablebaseProbe : null;
    this.nodes = 0;
    this.selDepth = 0;
    this.deadline = Infinity;
    this.startedAt = 0;
    this.completedDepth = 0;
    this.lastLines = [];
    this.pvTable = Array.from({ length: MAX_PLY + 2 }, () => new Uint16Array(MAX_PLY + 2));
    this.pvLength = new Int16Array(MAX_PLY + 2);
    this.hashStackA = new Uint32Array(MAX_PLY + 2);
    this.hashStackB = new Uint32Array(MAX_PLY + 2);
    this.rootHistory = [];
    this.rootRepetition = new Map();
    this.progressCallback = null;
    this.progressRootSide = WHITE;
    this.progressMultipv = 3;
    this.progressLastAt = 0;
    this.progressNextNode = PROGRESS_NODE_INTERVAL;
    this.rootOrdering = [];
    this.tablebaseProbeHits = 0;
    this.exactTablebaseHits = 0;
  }

  setTablebaseProbe(probe) {
    this.tablebaseProbe = typeof probe === 'function' ? probe : null;
  }

  clear() {
    this.lastLines = [];
    this.rootOrdering = [];
    this.rootRepetition.clear();
  }

  beginPosition() {
    this.nodes = 0;
    this.selDepth = 0;
    this.completedDepth = 0;
    this.lastLines = [];
    this.rootOrdering = [];
    this.tablebaseProbeHits = 0;
    this.exactTablebaseHits = 0;
  }

  probeTablebase(pos) {
    if (!this.tablebaseProbe || !Number.isInteger(pos?.pieceCount) || pos.pieceCount > 5) return null;
    try {
      const hit = this.tablebaseProbe(pos);
      if (!hit || hit.wdl === 2 || hit.wdl === undefined) return null;
      this.tablebaseProbeHits += 1;
      if (hit.exactDtm || hit.dtmUpperBound === false) this.exactTablebaseHits += 1;
      return hit;
    } catch {
      return null;
    }
  }

  shouldStop() {
    if ((this.nodes & 255) !== 0) return;
    const now = performance.now();
    if (this.progressCallback
      && (this.nodes >= this.progressNextNode)
      && now - this.progressLastAt >= PROGRESS_MIN_INTERVAL_MS) {
      this.emitProgress(now);
    }
    if (now >= this.deadline) throw ABORT;
  }

  recordPath(ply, pos) {
    const index = Math.max(0, Math.min(MAX_PLY, ply));
    this.hashStackA[index] = pos.hashA >>> 0;
    this.hashStackB[index] = pos.hashB >>> 0;
  }

  repetitionCount(pos, ply) {
    let count = this.rootRepetition.get(`${pos.hashA}:${pos.hashB}`) || 0;
    count += 1;
    const limit = Math.min(ply, pos.halfmove || 0);
    for (let back = 2; back <= limit; back += 2) {
      const index = ply - back;
      if (index >= 0 && this.hashStackA[index] === pos.hashA && this.hashStackB[index] === pos.hashB) count += 1;
    }
    return count;
  }

  orderedMoves(pos, moves, ply, depth, previousPvMove = 0, tacticalOnly = false) {
    const scored = [];
    const inCheck = isInCheck(pos);
    const lowBranchBonus = depth >= 2;
    for (const move of moves) {
      const capture = isCapture(pos, move);
      const promotion = isPromotion(move);
      const check = givesCheck(pos, move);
      if (tacticalOnly && !inCheck && !capture && !promotion && !check) continue;
      let score = moveTypeScore(pos, move);
      if (move === previousPvMove) score += 2_000_000;
      if (capture || promotion) score += Math.max(-80_000, staticExchangeEval(pos, move) * 700);
      const state = makeMove(pos, move);
      const hit = this.probeTablebase(pos);
      if (hit) {
        const childScore = -probeToScore(hit, ply + 1);
        if (childScore > 0) score += 1_500_000 + Math.min(250_000, childScore);
        else if (childScore < 0) score -= 1_500_000 + Math.min(250_000, -childScore);
      }
      if (lowBranchBonus || check) {
        const replies = generateLegalMoves(pos, false).length;
        if (check || replies <= 3) score += Math.max(0, 8 - Math.min(8, replies)) * 45_000;
      }
      undoMove(pos, move, state);
      scored.push({ move, score });
    }
    scored.sort((a, b) => b.score - a.score || moveToUci(a.move).localeCompare(moveToUci(b.move)));
    return scored.map(item => item.move);
  }

  qsearch(pos, alpha, beta, ply, qDepth = 0) {
    this.nodes += 1;
    this.selDepth = Math.max(this.selDepth, ply);
    this.pvLength[ply] = ply;
    this.shouldStop();
    if (ply >= MAX_PLY - 2) return evaluate(pos);
    if (isInsufficientMaterial(pos)) return DRAW;
    if (this.repetitionCount(pos, ply) >= 3) return DRAW;
    const tb = this.probeTablebase(pos);
    const tbScore = probeToScore(tb, ply);
    if (tbScore !== null) return tbScore;

    const inCheck = isInCheck(pos);
    let standPat = inCheck ? -INF : evaluate(pos);
    if (!inCheck) {
      if (standPat >= beta) return standPat;
      if (standPat > alpha) alpha = standPat;
      if (qDepth >= 3) return standPat;
    }
    const legal = generateLegalMoves(pos, false);
    if (!legal.length) return inCheck ? -MATE + ply : standPat;
    const moves = this.orderedMoves(pos, legal, ply, 0, 0, true);
    if (!moves.length) return standPat;
    for (const move of moves) {
      const capture = isCapture(pos, move);
      const promotion = isPromotion(move);
      const check = givesCheck(pos, move);
      if (!inCheck && !promotion && !check && !capture) continue;
      if (!inCheck && capture && !promotion) {
        const victim = typeOf(pos.board[moveTo(move)]);
        if (standPat + PIECE_VALUE[victim] + 70 < alpha) continue;
        if (staticExchangeEval(pos, move) < -45) continue;
      }
      const state = makeMove(pos, move);
      this.recordPath(ply + 1, pos);
      const score = -this.qsearch(pos, -beta, -alpha, ply + 1, qDepth + 1);
      undoMove(pos, move, state);
      if (score >= beta) return score;
      if (score > alpha) {
        alpha = score;
        this.pvTable[ply][ply] = move;
        for (let i = ply + 1; i < this.pvLength[ply + 1]; i += 1) this.pvTable[ply][i] = this.pvTable[ply + 1][i];
        this.pvLength[ply] = this.pvLength[ply + 1];
      }
    }
    return alpha;
  }

  search(pos, depth, alpha, beta, ply, previousMove = 0, extensionBudget = 0) {
    this.nodes += 1;
    this.selDepth = Math.max(this.selDepth, ply);
    this.pvLength[ply] = ply;
    this.shouldStop();
    if (ply >= MAX_PLY - 2) return evaluate(pos);
    if (isInsufficientMaterial(pos)) return DRAW;
    if (this.repetitionCount(pos, ply) >= 3) return DRAW;

    const tb = this.probeTablebase(pos);
    const tbScore = probeToScore(tb, ply);
    if (tbScore !== null) return tbScore;

    alpha = Math.max(alpha, -MATE + ply);
    beta = Math.min(beta, MATE - ply - 1);
    if (alpha >= beta) return alpha;

    const inCheck = isInCheck(pos);
    if (depth <= 0) return this.qsearch(pos, alpha, beta, ply, 0);
    const legal = generateLegalMoves(pos, false);
    if (!legal.length) return inCheck ? -MATE + ply : DRAW;

    const staticEval = inCheck ? -INF : evaluate(pos);
    const rootishEndgame = Number(pos.pieceCount || 0) <= 8;
    if (!inCheck && !rootishEndgame && depth <= 2 && staticEval + 155 * depth <= alpha) {
      const razor = this.qsearch(pos, alpha, beta, ply, 0);
      if (razor <= alpha) return razor;
    }
    if (!inCheck && !rootishEndgame && depth <= 3 && staticEval - (120 + 65 * depth) >= beta) return staticEval;

    const previousPvMove = this.pvTable[ply][ply] || 0;
    const moves = this.orderedMoves(pos, legal, ply, depth, previousPvMove || previousMove, false);
    const originalAlpha = alpha;
    let bestScore = -INF;
    let bestMove = 0;
    let legalIndex = 0;
    let quietIndex = 0;
    for (const move of moves) {
      const capture = isCapture(pos, move);
      const promotion = isPromotion(move);
      const check = givesCheck(pos, move);
      const quiet = !capture && !promotion;
      if (quiet) quietIndex += 1;

      if (legalIndex > 0 && quiet && !check && !inCheck && depth <= 3 && !rootishEndgame) {
        if (quietIndex > (depth === 1 ? 4 : depth === 2 ? 7 : 10)) continue;
        if (staticEval + 90 * depth <= alpha) continue;
      }
      if (legalIndex > 0 && capture && !promotion && !inCheck && depth <= 3 && staticExchangeEval(pos, move) < -70 * depth) continue;

      const state = makeMove(pos, move);
      this.recordPath(ply + 1, pos);
      const replyCount = depth >= 2 || check ? generateLegalMoves(pos, false).length : 99;
      const movedType = typeOf(state.moving);
      const nearPromotion = movedType === PAWN && (rankOf(moveTo(move)) === 1 || rankOf(moveTo(move)) === 3);
      const forceExtension = extensionBudget < 4 && (inCheck || promotion || check || nearPromotion || replyCount <= 1 || (replyCount <= 2 && depth >= 4));
      const newDepth = depth - 1 + (forceExtension ? 1 : 0);
      const nextBudget = extensionBudget + (forceExtension ? 1 : 0);
      let reduction = 0;
      if (legalIndex > 0 && quiet && !check && !promotion && !inCheck && depth >= 4 && replyCount > 3) {
        reduction = depth >= 7 ? 2 : 1;
        if (rootishEndgame || legal.length <= 5) reduction = Math.max(0, reduction - 1);
      }
      let score = -this.search(pos, Math.max(0, newDepth - reduction), -alpha - 1, -alpha, ply + 1, move, nextBudget);
      if (reduction && score > alpha) score = -this.search(pos, newDepth, -alpha - 1, -alpha, ply + 1, move, nextBudget);
      if (score > alpha && score < beta) score = -this.search(pos, newDepth, -beta, -alpha, ply + 1, move, nextBudget);
      undoMove(pos, move, state);
      legalIndex += 1;

      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
      if (score > alpha) {
        alpha = score;
        this.pvTable[ply][ply] = move;
        for (let i = ply + 1; i < this.pvLength[ply + 1]; i += 1) this.pvTable[ply][i] = this.pvTable[ply + 1][i];
        this.pvLength[ply] = this.pvLength[ply + 1];
        if (alpha >= beta) break;
      }
    }
    if (!bestMove) return originalAlpha;
    return bestScore;
  }

  rootMoveTablebaseLine(move, childHit, plyFromRoot = 1) {
    if (!childHit || childHit.wdl === 2 || childHit.wdl === undefined) return null;
    const childWdl = Math.sign(Number(childHit.wdl || 0));
    const rootWdl = -childWdl;
    if (rootWdl === 0) {
      return {
        score: DRAW,
        move,
        pv: [move],
        tablebase: true,
        rootMoveTablebase: true,
        tablebaseExactDtm: Boolean(childHit.exactDtm),
        tablebaseWdl: 0,
        mateVerified: false,
        dtm: 0
      };
    }
    if (!childHit.exactDtm) return null;
    const dtm = Math.max(1, Number(childHit.dtmPly || 0) + Math.max(1, Math.floor(Number(plyFromRoot) || 1)));
    return {
      score: rootWdl > 0 ? MATE - dtm : -MATE + dtm,
      move,
      pv: [move],
      tablebase: true,
      rootMoveTablebase: true,
      tablebaseExactDtm: true,
      tablebaseWdl: rootWdl,
      mateVerified: true,
      dtm
    };
  }

  scoreRootMove(pos, move, depth) {
    const rootCheck = givesCheck(pos, move);
    const state = makeMove(pos, move);
    this.recordPath(1, pos);
    this.pvLength[1] = 1;
    const childTbHit = this.probeTablebase(pos);
    const tbLine = this.rootMoveTablebaseLine(move, childTbHit, 1);
    if (tbLine) {
      undoMove(pos, move, state);
      return tbLine;
    }
    const replyCount = generateLegalMoves(pos, false).length;
    const extension = (rootCheck || replyCount <= 1) ? 1 : 0;
    const score = -this.search(pos, Math.max(0, depth - 1 + extension), -INF, INF, 1, move, extension ? 1 : 0);
    const pv = [move];
    for (let i = 1; i < this.pvLength[1]; i += 1) pv.push(this.pvTable[1][i]);
    undoMove(pos, move, state);
    return { score, move, pv };
  }

  searchRoot(pos, depth, multipv = 3) {
    const legal = generateLegalMoves(pos, false);
    if (!legal.length) return [];
    const previousBest = this.lastLines?.[0]?.move || 0;
    const moves = this.orderedMoves(pos, legal, 0, depth, previousBest, false);
    const lines = [];
    for (const move of moves) {
      const line = this.scoreRootMove(pos, move, depth);
      lines.push(line);
      this.noteProgressLines(pos, lines, depth, multipv);
    }
    lines.sort((a, b) => b.score - a.score || moveToUci(a.move).localeCompare(moveToUci(b.move)));
    this.rootOrdering = lines.map(line => line.move);
    return lines.slice(0, Math.max(1, Math.min(multipv, lines.length)));
  }

  noteProgressLines(pos, lines, depth, multipv) {
    if (!this.progressCallback || !lines?.length) return;
    const sorted = lines.slice().sort((a, b) => b.score - a.score).slice(0, multipv);
    this.lastLiveLines = this.buildPublicLines(pos, sorted, depth, false);
  }

  buildPublicLines(root, rootLines, depth, exact = true) {
    const rootSide = root.turn;
    return (rootLines || []).map(line => {
      const rootScore = Number(line.score || 0);
      const whiteScore = rootScoreToWhite(rootScore, rootSide);
      const pv = Array.isArray(line.pv) && line.pv.length ? line.pv.slice() : [line.move];
      // v23: mate-like scores are trusted and published directly. WDL-only
      // tablebase bounds remain internal because they do not carry a display score.
      const internalTablebase = isTablebaseBoundScore(rootScore);
      const internalMate = isMateScore(rootScore);
      const trustedMate = internalMate;
      const internalBound = internalTablebase;
      return {
        move: moveToUci(line.move),
        score: whiteScore,
        scoreText: internalBound ? '' : scoreToDisplay(whiteScore),
        scoreKind: trustedMate ? 'mate' : internalTablebase ? 'tablebase-wdl' : exact ? 'evaluation' : 'live',
        scoreNumeric: !internalBound,
        pv: pv.map(moveToUci),
        mateVerified: trustedMate,
        mateRejected: false,
        dtm: trustedMate ? Number(line.dtm || mateDistancePly(rootScore)) : 0,
        tablebase: Boolean(line.tablebase && !internalTablebase),
        tablebaseRoot: false,
        tablebaseWdl: Number(line.tablebaseWdl || 0),
        tablebaseExactDtm: Boolean(line.tablebaseExactDtm),
        tablebaseBound: internalTablebase,
        rootScoreExact: exact,
        pvComplete: exact,
        liveUpdate: !exact,
        liveDepth: !exact ? depth : 0,
        resultContract: trustedMate ? 'mate' : internalBound ? 'empty' : exact ? 'evaluation' : 'live',
        resultKindV2: trustedMate ? 'mate' : internalBound ? 'empty' : exact ? 'evaluation' : 'live'
      };
    });
  }

  emitProgress(now = performance.now()) {
    if (!this.progressCallback) return;
    const lines = this.lastLiveLines || this.buildPublicLines({ turn: this.progressRootSide }, [], 0, false);
    try {
      this.progressCallback({
        engine: MINIFISH_VERSION,
        engineLabel: MINIFISH_VERSION,
        depth: this.completedDepth,
        selDepth: this.selDepth,
        nodes: this.nodes,
        nps: Math.round(this.nodes * 1000 / Math.max(1, now - this.startedAt)),
        elapsed: Math.round(Math.max(0, now - this.startedAt)),
        scoreDepth: this.completedDepth,
        attemptedDepth: Math.max(1, this.completedDepth + 1),
        searchDepth: Math.max(1, this.completedDepth + 1),
        nextDepth: Math.max(1, this.completedDepth + 1),
        hashfull: 0,
        rootTurn: this.progressRootSide,
        lines,
        completed: false,
        liveUpdate: true,
        liveProgress: true,
        pvIncomplete: true,
        pvComplete: false,
        terminal: false,
        tablebase: false,
        minifish: true
      });
    } catch {}
    this.progressLastAt = now;
    this.progressNextNode = this.nodes + PROGRESS_NODE_INTERVAL;
  }

  analyze(pos, {
    timeMs = 950,
    maxDepth = 32,
    multipv = 3,
    startDepth = 1,
    historyKeys = [],
    newPosition = true,
    onProgress = null
  } = {}) {
    const rootSide = pos.turn;
    if (newPosition) this.beginPosition();
    this.nodes = 0;
    this.selDepth = 0;
    this.startedAt = performance.now();
    this.deadline = this.startedAt + Math.max(35, Number(timeMs || 0));
    this.progressCallback = typeof onProgress === 'function' ? onProgress : null;
    this.progressRootSide = rootSide;
    this.progressMultipv = Math.max(1, Number(multipv || 1));
    this.progressLastAt = this.startedAt;
    this.progressNextNode = PROGRESS_NODE_INTERVAL;
    this.lastLiveLines = [];
    this.rootRepetition.clear();
    for (const key of Array.isArray(historyKeys) ? historyKeys : []) {
      if (!key) continue;
      const identity = `${key.a >>> 0}:${key.b >>> 0}`;
      this.rootRepetition.set(identity, (this.rootRepetition.get(identity) || 0) + 1);
    }
    this.recordPath(0, pos);

    const terminal = immediateTerminalScore(pos, 0);
    if (terminal && !terminal.legal.length) {
      const elapsed = Math.max(1, performance.now() - this.startedAt);
      return {
        engine: MINIFISH_VERSION,
        engineLabel: MINIFISH_VERSION,
        depth: 0,
        selDepth: 0,
        nodes: 0,
        nps: 0,
        elapsed: Math.round(elapsed),
        scoreDepth: 0,
        pvDepth: 0,
        pvTarget: 0,
        pvComplete: true,
        lines: [],
        terminal: true,
        completed: true,
        multiPvVerified: true,
        rootTurn: rootSide,
        minifish: true,
        nextDepth: 0,
        searchDepth: 0,
        hashfull: 0
      };
    }

    let completed = null;
    let depth = Math.max(1, Number(startDepth || 1));
    try {
      for (; depth <= Math.max(1, Number(maxDepth || 1)); depth += 1) {
        const rootLines = this.searchRoot(pos, depth, multipv);
        if (!rootLines.length) break;
        const lines = this.buildPublicLines(pos, rootLines, depth, true);
        completed = { depth, lines, rootLines };
        this.lastLines = rootLines.map(cloneLine);
        this.completedDepth = depth;
        this.shouldStop();
      }
    } catch (error) {
      if (error !== ABORT) throw error;
    }

    const source = completed || (this.lastLines.length ? {
      depth: this.completedDepth,
      rootLines: this.lastLines,
      lines: this.buildPublicLines(pos, this.lastLines, this.completedDepth, false)
    } : null);
    const elapsed = Math.max(1, performance.now() - this.startedAt);
    const resultDepth = Math.max(0, Number(source?.depth || 0));
    const lines = source?.lines || [];
    const pvDepth = Math.max(0, ...lines.map(line => Array.isArray(line.pv) ? line.pv.length : 0), 0);
    const completedIteration = Boolean(completed && resultDepth >= Number(startDepth || 1));
    return {
      engine: MINIFISH_VERSION,
      engineLabel: MINIFISH_VERSION,
      depth: resultDepth,
      selDepth: this.selDepth,
      nodes: this.nodes,
      nps: Math.round(this.nodes * 1000 / elapsed),
      elapsed: Math.round(elapsed),
      scoreDepth: resultDepth,
      pvDepth,
      pvTarget: 0,
      pvComplete: completedIteration,
      lines,
      terminal: false,
      completed: completedIteration,
      liveUpdate: !completedIteration,
      pvIncomplete: !completedIteration,
      multiPvVerified: completedIteration,
      attemptedDepth: Math.max(1, Number(startDepth || 1)),
      hashfull: 0,
      rootTurn: rootSide,
      minifish: true,
      tablebaseProbeHits: this.tablebaseProbeHits,
      exactTablebaseHits: this.exactTablebaseHits,
      nextDepth: completedIteration ? resultDepth + 1 : Math.max(1, Number(startDepth || 1)),
      searchDepth: completedIteration ? resultDepth + 1 : Math.max(1, Number(startDepth || 1))
    };
  }
}

export function analyzeMinifishOnce(fen, options = {}) {
  const searcher = new MinifishSearcher(options);
  return searcher.analyze(EnginePosition.fromFEN(fen), options);
}
