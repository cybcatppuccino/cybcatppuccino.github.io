import {
  EngineInternals,
  analyzePositionActivity,
  evaluate,
  generateLegalMoves,
  isInCheck,
  staticExchangeEval,
  uciToMove
} from './engine.js';

const {
  PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  PIECE_VALUE, MATE_BOUND, makeMove, undoMove, moveFrom, moveTo, givesCheck,
  movePromotion, sideOf, typeOf
} = EngineInternals;

export const AI_STYLES = Object.freeze([
  {
    id: 'balanced', label: 'Balanced', shortLabel: 'Balanced',
    description: 'Maximum-strength objective play with no secondary preference.',
    timeMs: 3800, maxDepth: 40, multipv: 1, marginCp: 0, endgameProbeMs: 200, fortressProbeMs: 190
  },
  {
    id: 'aggressive', label: 'Aggressive', shortLabel: 'Aggressive',
    description: 'Prefers open fights, exchanges and sound sacrifices when the objective cost is small.',
    timeMs: 4800, maxDepth: 40, multipv: 5, marginCp: 30, endgameProbeMs: 200, fortressProbeMs: 190
  },
  {
    id: 'conservative', label: 'Conservative', shortLabel: 'Conservative',
    description: 'Protects the result, favours stable conversion and seeks defensive drawing resources.',
    timeMs: 4800, maxDepth: 40, multipv: 5, marginCp: 18, endgameProbeMs: 240, fortressProbeMs: 240
  },
  {
    id: 'cunning', label: 'Cunning', shortLabel: 'Cunning',
    description: 'Chooses sound moves whose accurate reply is narrow or unobvious.',
    timeMs: 5200, maxDepth: 40, multipv: 5, marginCp: 24, endgameProbeMs: 210, fortressProbeMs: 200,
    responseProbeMs: 160
  },
  {
    id: 'pressing', label: 'Pressing', shortLabel: 'Pressing',
    description: 'Restricts space and legal choices while maintaining the objective result.',
    timeMs: 5000, maxDepth: 40, multipv: 5, marginCp: 24, endgameProbeMs: 210, fortressProbeMs: 200
  }
]);

export function styleConfig(style) {
  const id = String(style || 'balanced').toLowerCase();
  return AI_STYLES.find(item => item.id === id) || AI_STYLES[0];
}

function whiteScoreFromStatic(pos) {
  const score = evaluate(pos);
  return pos.turn === WHITE ? score : -score;
}

function materialSnapshot(pos) {
  const result = {
    white: 0, black: 0, whitePieces: 0, blackPieces: 0,
    whiteNonKing: 0, blackNonKing: 0
  };
  for (const piece of pos.board) {
    if (!piece) continue;
    const side = sideOf(piece);
    const type = typeOf(piece);
    if (side === WHITE) {
      result.white += PIECE_VALUE[type];
      result.whitePieces += 1;
      if (type !== KING) result.whiteNonKing += 1;
    } else {
      result.black += PIECE_VALUE[type];
      result.blackPieces += 1;
      if (type !== KING) result.blackNonKing += 1;
    }
  }
  return result;
}

function lineVolatility(position, pv = []) {
  const cursor = position.clone();
  const values = [whiteScoreFromStatic(cursor)];
  for (const uci of pv.slice(0, 6)) {
    const move = uciToMove(cursor, uci);
    if (!move) break;
    makeMove(cursor, move);
    values.push(whiteScoreFromStatic(cursor));
  }
  let swing = 0;
  for (let i = 1; i < values.length; i += 1) swing += Math.abs(values[i] - values[i - 1]);
  return values.length > 1 ? swing / (values.length - 1) : 0;
}

export function buildMoveStyleProfile(position, line) {
  const move = uciToMove(position, line?.move);
  if (!move) return null;
  const side = position.turn;
  const movingPiece = position.board[moveFrom(move)];
  const capturedPiece = position.board[moveTo(move)];
  const movingType = typeOf(movingPiece);
  const capturedType = typeOf(capturedPiece);
  const beforeActivity = analyzePositionActivity(position);
  const beforeMaterial = materialSnapshot(position);
  const capture = Boolean(capturedPiece);
  const promotion = Boolean(movePromotion(move));
  const see = capture || promotion ? staticExchangeEval(position, move) : 0;
  const check = givesCheck(position, move);
  const state = makeMove(position, move);
  const afterActivity = analyzePositionActivity(position);
  const afterMaterial = materialSnapshot(position);
  const opponentKey = position.turn === WHITE ? 'white' : 'black';
  const moverKey = side === WHITE ? 'white' : 'black';
  const opponentActivity = afterActivity.exact[opponentKey];
  const moverStructural = afterActivity[moverKey];
  const opponentStructural = afterActivity[opponentKey];
  const opponentInCheck = isInCheck(position);
  undoMove(position, move, state);

  const nonKingBefore = beforeMaterial.whiteNonKing + beforeMaterial.blackNonKing;
  const nonKingAfter = afterMaterial.whiteNonKing + afterMaterial.blackNonKing;
  const closureDelta = beforeActivity.closure - afterActivity.closure;
  const restriction = beforeActivity.exact[opponentKey].sound - opponentActivity.sound;
  const spaceEdge = moverStructural.safeMoves - opponentStructural.safeMoves;
  const pressureEdge = moverStructural.kingPressure - opponentStructural.kingPressure;
  const sacrifice = Math.max(0, -see);
  const materialExchange = Math.max(0, nonKingBefore - nonKingAfter);
  const irreversible = movingType === PAWN || capture || promotion;
  const quiet = !capture && !promotion && !check;

  return {
    capture,
    promotion,
    check,
    quiet,
    irreversible,
    movingType,
    capturedType,
    see,
    sacrifice,
    materialExchange,
    closureBefore: beforeActivity.closure,
    closureAfter: afterActivity.closure,
    opensPosition: Math.max(0, closureDelta),
    closesPosition: Math.max(0, -closureDelta),
    opponentLegal: opponentActivity.legal,
    opponentSound: opponentActivity.sound,
    opponentForcing: opponentActivity.captures + opponentActivity.checks,
    opponentInCheck,
    restriction,
    spaceEdge,
    pressureEdge,
    volatility: lineVolatility(position, line.pv || []),
    replyGap: Number(line.replyProfile?.gapCp || 0),
    goodReplyCount: Number(line.replyProfile?.goodReplyCount || 0),
    bestReplyForcing: Boolean(line.replyProfile?.bestReplyForcing),
    bestReplyQuiet: Boolean(line.replyProfile?.bestReplyQuiet)
  };
}

function utilityFor(line, sideToMove) {
  const sign = sideToMove === 'b' || sideToMove === BLACK ? -1 : 1;
  return sign * Number(line?.score || 0);
}

function outcomeClass(line, utility) {
  if (line?.mateVerified && Math.abs(utility) >= MATE_BOUND) return utility > 0 ? 3 : -3;
  if (Number.isFinite(Number(line?.tablebaseWdl))) return Math.sign(Number(line.tablebaseWdl));
  if (utility > 85) return 1;
  if (utility < -85) return -1;
  return 0;
}

function safeStylePool(lines, config, sideToMove) {
  const ranked = lines
    .filter(Boolean)
    .map((line, index) => ({ line, index, utility: utilityFor(line, sideToMove) }))
    .sort((a, b) => b.utility - a.utility || a.index - b.index);
  if (!ranked.length) return [];
  if (config.id === 'balanced') return [ranked[0]];

  const best = ranked[0];
  const bestClass = outcomeClass(best.line, best.utility);
  const bestIsSolvedWin = bestClass === 3 || Number(best.line?.tablebaseWdl) > 0;
  let margin = Number(config.marginCp || 0);

  // v17.1: style is a secondary preference, never the primary objective.
  // When the engine already sees a stable advantage, all personalities stay
  // close to the best move.  Tricky/pressing choices get more freedom only in
  // approximately equal or worse positions, where practical chances matter.
  if (bestIsSolvedWin || best.utility >= 650) margin = Math.min(margin, 8);
  else if (best.utility >= 260) margin = Math.min(margin, 14);
  else if (best.utility >= 110) margin = Math.min(margin, 20);
  else if (best.utility <= -180) margin = Math.min(58, margin + (config.id === 'cunning' ? 30 : 14));
  else if (Math.abs(best.utility) <= 90 && config.id === 'cunning') margin = Math.min(54, margin + 16);

  let pool = ranked.filter(item => best.utility - item.utility <= margin);
  if (bestClass >= 0) {
    // Do not turn a searched win/draw into a clearly losing choice merely for
    // personality.  In winning positions, insist on retaining the advantage.
    const floor = best.utility >= 110 ? Math.max(70, best.utility - margin) : -35;
    pool = pool.filter(item => outcomeClass(item.line, item.utility) >= 0 || item.utility >= floor);
  }
  if (bestClass === 3) pool = pool.filter(item => outcomeClass(item.line, item.utility) === 3);
  if (Number.isFinite(Number(best.line?.tablebaseWdl))) {
    pool = pool.filter(item => Number(item.line?.tablebaseWdl) === Number(best.line.tablebaseWdl));
  }
  return pool.length ? pool : [best];
}

function styleBonus(item, config) {
  const p = {
    materialExchange: 0, sacrifice: 0, opensPosition: 0, closesPosition: 0,
    pressureEdge: 0, volatility: 0, opponentForcing: 0, restriction: 0,
    replyGap: 0, goodReplyCount: 0, opponentLegal: 0, opponentSound: 0,
    spaceEdge: 0,
    ...(item.line.styleProfile || {})
  };
  switch (config.id) {
    case 'aggressive':
      return (p.check ? 36 : 0)
        + (p.capture ? 18 : 0)
        + p.materialExchange * 24
        + Math.min(30, p.sacrifice * 0.09)
        + p.opensPosition * 72
        + Math.max(0, p.pressureEdge) * 5
        + Math.max(0, p.volatility - 25) * 0.08
        - p.closesPosition * 55;
    case 'conservative':
      return (item.utility < 0 && item.line.repetition ? 90 : 0)
        + (item.line.fortressProof || item.line.tablebaseWdl === 0 ? 120 : 0)
        + Math.max(0, -p.volatility + 75) * 0.32
        + Math.max(0, -p.opponentForcing + 3) * 7
        + Math.max(0, p.restriction) * 1
        + (p.sacrifice ? -Math.min(55, p.sacrifice * 0.16) : 8)
        + (item.utility > 120 ? p.materialExchange * 16 : -p.materialExchange * 5)
        - p.opensPosition * 26;
    case 'cunning':
      return Math.min(90, p.replyGap * 0.55)
        + Math.max(0, 3 - p.goodReplyCount) * 28
        + (p.bestReplyQuiet ? 24 : 0)
        - (p.bestReplyForcing ? 18 : 0)
        + Math.min(28, p.opponentLegal * 2)
        + (p.quiet ? 10 : 0)
        + Math.max(0, p.pressureEdge) * 3;
    case 'pressing':
      return Math.max(0, p.restriction) * 15
        + Math.max(0, 8 - p.opponentSound) * 7
        + Math.max(0, p.spaceEdge) * 5
        + Math.max(0, p.pressureEdge) * 8
        + (p.check ? 28 : 0)
        + (p.opponentInCheck ? 12 : 0)
        - p.sacrifice * 0.08
        - p.closesPosition * 10;
    default:
      return 0;
  }
}

export function selectLineForStyle(lines, configOrId, sideToMove = 'w') {
  const config = typeof configOrId === 'string' ? styleConfig(configOrId) : (configOrId || AI_STYLES[0]);
  const pool = safeStylePool(lines, config, sideToMove);
  if (!pool.length) return null;
  if (config.id === 'balanced') return pool[0].line;

  const bestUtility = pool[0].utility;
  const objectivePenalty = bestUtility >= 110 ? 2.15 : bestUtility <= -180 ? 0.82 : 1.35;
  const scored = pool.map(item => ({
    ...item,
    styleScore: styleBonus(item, config) - (bestUtility - item.utility) * objectivePenalty
  }));
  scored.sort((a, b) => b.styleScore - a.styleScore || b.utility - a.utility || a.index - b.index);
  return scored[0].line;
}
