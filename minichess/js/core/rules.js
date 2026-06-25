import {
  BOARD_SIZE,
  COLORS,
  MOVE_FLAGS,
  TYPES,
  fileOf,
  opposite,
  rankOf,
  square
} from './constants.js';

const KNIGHT_OFFSETS = [
  [1, 2], [2, 1], [2, -1], [1, -2],
  [-1, -2], [-2, -1], [-2, 1], [-1, 2]
];
const KING_OFFSETS = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0], [1, 0],
  [-1, 1], [0, 1], [1, 1]
];
const BISHOP_DIRS = [[1, 1], [1, -1], [-1, 1], [-1, -1]];
const ROOK_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1]];
const QUEEN_DIRS = [...BISHOP_DIRS, ...ROOK_DIRS];
const PROMOTIONS = [TYPES.QUEEN, TYPES.ROOK, TYPES.BISHOP, TYPES.KNIGHT];

function inside(file, rank) {
  return file >= 0 && file < BOARD_SIZE && rank >= 0 && rank < BOARD_SIZE;
}

function makeMove(from, to, moving, captured = null, promotion = null) {
  const isCapture = Boolean(captured);
  return {
    from,
    to,
    piece: moving.type,
    color: moving.color,
    captured: captured?.type || null,
    promotion,
    flag: promotion
      ? (isCapture ? MOVE_FLAGS.PROMOTION_CAPTURE : MOVE_FLAGS.PROMOTION)
      : (isCapture ? MOVE_FLAGS.CAPTURE : MOVE_FLAGS.QUIET)
  };
}

function pseudoMoves(position, color = position.turn, onlyCaptures = false) {
  const moves = [];
  position.board.forEach((moving, from) => {
    if (!moving || moving.color !== color) return;
    const file = fileOf(from);
    const rank = rankOf(from);

    if (moving.type === TYPES.PAWN) {
      const direction = color === COLORS.WHITE ? 1 : -1;
      const promotionRank = color === COLORS.WHITE ? BOARD_SIZE - 1 : 0;
      const nextRank = rank + direction;
      if (!onlyCaptures && inside(file, nextRank)) {
        const to = square(file, nextRank);
        if (!position.board[to]) {
          if (nextRank === promotionRank) {
            PROMOTIONS.forEach(p => moves.push(makeMove(from, to, moving, null, p)));
          } else {
            moves.push(makeMove(from, to, moving));
          }
        }
      }
      for (const df of [-1, 1]) {
        const captureFile = file + df;
        if (!inside(captureFile, nextRank)) continue;
        const to = square(captureFile, nextRank);
        const target = position.board[to];
        if (target && target.color !== color) {
          if (nextRank === promotionRank) {
            PROMOTIONS.forEach(p => moves.push(makeMove(from, to, moving, target, p)));
          } else {
            moves.push(makeMove(from, to, moving, target));
          }
        }
      }
      return;
    }

    if (moving.type === TYPES.KNIGHT || moving.type === TYPES.KING) {
      const offsets = moving.type === TYPES.KNIGHT ? KNIGHT_OFFSETS : KING_OFFSETS;
      for (const [df, dr] of offsets) {
        const tf = file + df;
        const tr = rank + dr;
        if (!inside(tf, tr)) continue;
        const to = square(tf, tr);
        const target = position.board[to];
        if (!target && !onlyCaptures) moves.push(makeMove(from, to, moving));
        else if (target && target.color !== color) moves.push(makeMove(from, to, moving, target));
      }
      return;
    }

    const dirs = moving.type === TYPES.BISHOP
      ? BISHOP_DIRS
      : moving.type === TYPES.ROOK
        ? ROOK_DIRS
        : QUEEN_DIRS;
    for (const [df, dr] of dirs) {
      let tf = file + df;
      let tr = rank + dr;
      while (inside(tf, tr)) {
        const to = square(tf, tr);
        const target = position.board[to];
        if (!target) {
          if (!onlyCaptures) moves.push(makeMove(from, to, moving));
        } else {
          if (target.color !== color) moves.push(makeMove(from, to, moving, target));
          break;
        }
        tf += df;
        tr += dr;
      }
    }
  });
  return moves;
}

export function isSquareAttacked(position, targetSquare, byColor) {
  const targetFile = fileOf(targetSquare);
  const targetRank = rankOf(targetSquare);

  for (let sq = 0; sq < position.board.length; sq += 1) {
    const p = position.board[sq];
    if (!p || p.color !== byColor) continue;
    const file = fileOf(sq);
    const rank = rankOf(sq);

    if (p.type === TYPES.PAWN) {
      const direction = byColor === COLORS.WHITE ? 1 : -1;
      if (targetRank === rank + direction && Math.abs(targetFile - file) === 1) return true;
      continue;
    }

    if (p.type === TYPES.KNIGHT) {
      if (KNIGHT_OFFSETS.some(([df, dr]) => file + df === targetFile && rank + dr === targetRank)) return true;
      continue;
    }

    if (p.type === TYPES.KING) {
      if (Math.abs(targetFile - file) <= 1 && Math.abs(targetRank - rank) <= 1) return true;
      continue;
    }

    const dirs = p.type === TYPES.BISHOP ? BISHOP_DIRS : p.type === TYPES.ROOK ? ROOK_DIRS : QUEEN_DIRS;
    for (const [df, dr] of dirs) {
      let tf = file + df;
      let tr = rank + dr;
      while (inside(tf, tr)) {
        const sq2 = square(tf, tr);
        if (sq2 === targetSquare) return true;
        if (position.board[sq2]) break;
        tf += df;
        tr += dr;
      }
    }
  }
  return false;
}

export function isInCheck(position, color = position.turn) {
  const king = position.kingSquare(color);
  if (king < 0) return false;
  return isSquareAttacked(position, king, opposite(color));
}

export function legalMoves(position, { from = null, capturesOnly = false } = {}) {
  return pseudoMoves(position, position.turn, capturesOnly).filter(move => {
    if (from !== null && move.from !== from) return false;
    const next = position.makeMove(move);
    return !isInCheck(next, move.color);
  });
}


export function isInsufficientMaterial(position) {
  const material = position.board
    .map((piece, sq) => piece ? { ...piece, sq } : null)
    .filter(piece => piece && piece.type !== TYPES.KING);

  if (!material.length) return true;
  if (material.some(piece => [TYPES.PAWN, TYPES.ROOK, TYPES.QUEEN].includes(piece.type))) return false;
  if (material.length === 1) return [TYPES.BISHOP, TYPES.KNIGHT].includes(material[0].type);

  if (material.every(piece => piece.type === TYPES.BISHOP)) {
    const squareColors = new Set(material.map(piece => (fileOf(piece.sq) + rankOf(piece.sq)) % 2));
    return squareColors.size === 1;
  }
  return false;
}

export function gameStatus(position, repetitionCount = 1) {
  const moves = legalMoves(position);
  const check = isInCheck(position, position.turn);
  if (!moves.length) {
    return check
      ? { state: 'checkmate', check: true, winner: opposite(position.turn), legalMoves: [] }
      : { state: 'stalemate', check: false, winner: null, legalMoves: [] };
  }
  if (isInsufficientMaterial(position)) return { state: 'draw-insufficient', check, winner: null, legalMoves: moves };
  if (position.halfmove >= 100) return { state: 'draw-50', check, winner: null, legalMoves: moves };
  if (repetitionCount >= 3) return { state: 'draw-repetition', check, winner: null, legalMoves: moves };
  return { state: check ? 'check' : 'playing', check, winner: null, legalMoves: moves };
}

export const Rules = Object.freeze({
  pseudoMoves,
  legalMoves,
  isSquareAttacked,
  isInCheck,
  gameStatus,
  isInsufficientMaterial
});
