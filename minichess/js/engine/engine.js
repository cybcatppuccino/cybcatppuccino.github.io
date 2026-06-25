// Gardner MiniChess classical analysis engine.
// Native 25-square board, iterative deepening PVS, quiescence, TT and
// conservative selective pruning tuned for the tactical 5×5 game.

export const ENGINE_VERSION = 'Orion JS 4.0';

const EMPTY = 0;
const PAWN = 1;
const KNIGHT = 2;
const BISHOP = 3;
const ROOK = 4;
const QUEEN = 5;
const KING = 6;
const WHITE = 1;
const BLACK = -1;
const BOARD_N = 25;
const MAX_PLY = 64;
const INF = 32000;
const MATE = 30000;
const MATE_BOUND = 29000;
const DRAW = 0;

const TYPE_TO_CHAR = ['', 'p', 'n', 'b', 'r', 'q', 'k'];
const CHAR_TO_TYPE = Object.freeze({ p: PAWN, n: KNIGHT, b: BISHOP, r: ROOK, q: QUEEN, k: KING });
const PIECE_VALUE = Int16Array.from([0, 100, 305, 330, 515, 895, 20000]);
const PHASE_VALUE = Int8Array.from([0, 0, 1, 1, 2, 4, 0]);
const MAX_PHASE = 16;
const PROMOTIONS = Int8Array.from([QUEEN, ROOK, BISHOP, KNIGHT]);
const KNIGHT_OFFSETS = Object.freeze([[1,2],[2,1],[2,-1],[1,-2],[-1,-2],[-2,-1],[-2,1],[-1,2]]);
const KING_OFFSETS = Object.freeze([[-1,-1],[0,-1],[1,-1],[-1,0],[1,0],[-1,1],[0,1],[1,1]]);
const BISHOP_DIRS = Object.freeze([[1,1],[1,-1],[-1,1],[-1,-1]]);
const ROOK_DIRS = Object.freeze([[1,0],[-1,0],[0,1],[0,-1]]);
const QUEEN_DIRS = Object.freeze([...BISHOP_DIRS, ...ROOK_DIRS]);

// Precomputed geometry avoids repeated coordinate arithmetic in the hot path.
const KNIGHT_TARGETS = Array.from({ length: BOARD_N }, (_, sq) => {
  const file = sq % 5, rank = (sq / 5) | 0;
  return Int8Array.from(KNIGHT_OFFSETS
    .map(([df, dr]) => [file + df, rank + dr])
    .filter(([f, r]) => f >= 0 && f < 5 && r >= 0 && r < 5)
    .map(([f, r]) => r * 5 + f));
});
const KING_TARGETS = Array.from({ length: BOARD_N }, (_, sq) => {
  const file = sq % 5, rank = (sq / 5) | 0;
  return Int8Array.from(KING_OFFSETS
    .map(([df, dr]) => [file + df, rank + dr])
    .filter(([f, r]) => f >= 0 && f < 5 && r >= 0 && r < 5)
    .map(([f, r]) => r * 5 + f));
});
const PAWN_TARGETS = {
  [WHITE]: Array.from({ length: BOARD_N }, (_, sq) => {
    const file = sq % 5, rank = (sq / 5) | 0;
    return Int8Array.from([-1, 1]
      .map(df => [file + df, rank + 1])
      .filter(([f, r]) => f >= 0 && f < 5 && r >= 0 && r < 5)
      .map(([f, r]) => r * 5 + f));
  }),
  [BLACK]: Array.from({ length: BOARD_N }, (_, sq) => {
    const file = sq % 5, rank = (sq / 5) | 0;
    return Int8Array.from([-1, 1]
      .map(df => [file + df, rank - 1])
      .filter(([f, r]) => f >= 0 && f < 5 && r >= 0 && r < 5)
      .map(([f, r]) => r * 5 + f));
  })
};
const RAYS = Array.from({ length: BOARD_N }, (_, sq) => {
  const file = sq % 5, rank = (sq / 5) | 0;
  return QUEEN_DIRS.map(([df, dr]) => {
    const targets = [];
    let f = file + df, r = rank + dr;
    while (f >= 0 && f < 5 && r >= 0 && r < 5) {
      targets.push(r * 5 + f);
      f += df;
      r += dr;
    }
    return Int8Array.from(targets);
  });
});
const LMR_TABLE = Array.from({ length: 33 }, (_, depth) =>
  Int8Array.from({ length: 33 }, (_, moveIndex) =>
    depth < 3 || moveIndex < 2 ? 0 : Math.max(0, Math.floor(Math.log(depth) * Math.log(moveIndex + 1) / 2.35))
  )
);

// Flat tables are indexed from White's side: rank 0 is White's back rank.
const PST = Object.freeze({
  [PAWN]: Int16Array.from([
      0,   0,   0,   0,   0,
     10,  14,  16,  14,  10,
     20,  25,  30,  25,  20,
     42,  50,  58,  50,  42,
      0,   0,   0,   0,   0
  ]),
  [KNIGHT]: Int16Array.from([
    -32, -15, -10, -15, -32,
    -18,   5,  12,   5, -18,
    -10,  15,  26,  15, -10,
    -18,   7,  14,   7, -18,
    -32, -15, -10, -15, -32
  ]),
  [BISHOP]: Int16Array.from([
    -14,  -6,  -4,  -6, -14,
     -6,   8,  11,   8,  -6,
     -4,  12,  17,  12,  -4,
     -6,   8,  11,   8,  -6,
    -14,  -6,  -4,  -6, -14
  ]),
  [ROOK]: Int16Array.from([
      0,   3,   5,   3,   0,
      4,   6,   8,   6,   4,
      5,   8,  10,   8,   5,
      7,  10,  12,  10,   7,
      8,  10,  12,  10,   8
  ]),
  [QUEEN]: Int16Array.from([
    -10,  -4,  -2,  -4, -10,
     -4,   3,   7,   3,  -4,
     -2,   7,  12,   7,  -2,
     -4,   3,   7,   3,  -4,
    -10,  -4,  -2,  -4, -10
  ]),
  kingMid: Int16Array.from([
     30,  20,   8,  18,  30,
     12,   2, -10,   0,  12,
      0, -12, -22, -12,   0,
    -10, -18, -28, -18, -10,
    -18, -24, -34, -24, -18
  ]),
  kingEnd: Int16Array.from([
    -20, -10,  -6, -10, -20,
    -10,   8,  14,   8, -10,
     -6,  14,  28,  14,  -6,
    -10,   8,  14,   8, -10,
    -20, -10,  -6, -10, -20
  ])
});

const TT_EXACT = 0;
const TT_LOWER = 1;
const TT_UPPER = 2;
const ABORT = Object.freeze({ aborted: true });

function fileOf(sq) { return sq % 5; }
function rankOf(sq) { return (sq / 5) | 0; }
function square(file, rank) { return rank * 5 + file; }
function inside(file, rank) { return file >= 0 && file < 5 && rank >= 0 && rank < 5; }
function sideOf(piece) { return piece > 0 ? WHITE : piece < 0 ? BLACK : 0; }
function typeOf(piece) { return Math.abs(piece); }
function mirrorSquare(sq) { return square(fileOf(sq), 4 - rankOf(sq)); }
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function moveFrom(move) { return move & 31; }
function moveTo(move) { return (move >>> 5) & 31; }
function movePromotion(move) { return (move >>> 10) & 7; }
function encodeMove(from, to, promotion = 0) { return from | (to << 5) | (promotion << 10); }
function moveKey(move) { return move >>> 0; }
function isMateScore(score) { return Math.abs(score) >= MATE_BOUND; }
function scoreToTT(score, ply) { return score > MATE_BOUND ? score + ply : score < -MATE_BOUND ? score - ply : score; }
function scoreFromTT(score, ply) { return score > MATE_BOUND ? score - ply : score < -MATE_BOUND ? score + ply : score; }

function xorshift(seed) {
  let x = seed >>> 0;
  x ^= x << 13;
  x ^= x >>> 17;
  x ^= x << 5;
  return x >>> 0;
}

const ZOBRIST_A = Array.from({ length: 12 }, () => new Uint32Array(25));
const ZOBRIST_B = Array.from({ length: 12 }, () => new Uint32Array(25));
let za = 0x9e3779b9;
let zb = 0x243f6a88;
for (let p = 0; p < 12; p += 1) {
  for (let sq = 0; sq < 25; sq += 1) {
    za = xorshift(za + p * 31 + sq + 1);
    zb = xorshift(zb + p * 47 + sq + 11);
    ZOBRIST_A[p][sq] = za;
    ZOBRIST_B[p][sq] = zb;
  }
}
const TURN_A = xorshift(za ^ 0xa5a5a5a5);
const TURN_B = xorshift(zb ^ 0x5a5a5a5a);

// The board/repetition hash deliberately excludes the fifty-move counter.
// TT entries do not: otherwise an identical board reached at halfmove 0 and
// halfmove 99 could incorrectly share a value near the automatic-draw horizon.
function ttContextKeyA(pos) {
  const clock = Math.min(100, Math.max(0, pos.halfmove | 0)) + 1;
  return (pos.hashA ^ Math.imul(clock, 0x9e3779b1)) >>> 0;
}

function ttContextKeyB(pos) {
  const clock = Math.min(100, Math.max(0, pos.halfmove | 0)) + 1;
  return (pos.hashB ^ Math.imul(clock, 0x85ebca6b)) >>> 0;
}

function zobristIndex(piece) {
  const type = typeOf(piece) - 1;
  return piece > 0 ? type : 6 + type;
}

function coordToSquare(coord) {
  const file = coord.charCodeAt(0) - 98; // study files b..f
  const rank = coord.charCodeAt(1) - 50; // study ranks 2..6
  return inside(file, rank) ? square(file, rank) : -1;
}

function squareToCoord(sq) {
  return String.fromCharCode(98 + fileOf(sq)) + String.fromCharCode(50 + rankOf(sq));
}

export function moveToUci(move) {
  return `${squareToCoord(moveFrom(move))}${squareToCoord(moveTo(move))}${TYPE_TO_CHAR[movePromotion(move)] || ''}`;
}

export function uciToMove(position, uci) {
  const match = String(uci || '').trim().match(/^([b-f][2-6])([b-f][2-6])([qrbn])?$/i);
  if (!match) return 0;
  const from = coordToSquare(match[1].toLowerCase());
  const to = coordToSquare(match[2].toLowerCase());
  const promo = match[3] ? CHAR_TO_TYPE[match[3].toLowerCase()] : 0;
  const wanted = encodeMove(from, to, promo);
  return generateLegalMoves(position).find(move => move === wanted) || 0;
}

export class EnginePosition {
  constructor(board = new Int8Array(25), turn = WHITE, halfmove = 0, fullmove = 1) {
    this.board = board;
    this.turn = turn;
    this.halfmove = halfmove;
    this.fullmove = fullmove;
    this.hashA = 0;
    this.hashB = 0;
    this.recomputeHash();
  }

  static fromFEN(fen) {
    const parts = String(fen || '').trim().split(/\s+/);
    if (!parts[0]) throw new Error('Engine FEN is empty.');
    let rows = parts[0].split('/');
    if (rows.length === 8) {
      const expanded = rows.map(row => expandRow(row, 8));
      if ([0, 1, 7].some(index => expanded[index].some(Boolean))) {
        throw new Error('Pieces outside the Gardner 5×5 area are not supported.');
      }
      const playable = expanded.slice(2, 7);
      if (playable.some(row => row[0] || row[6] || row[7])) {
        throw new Error('A piece is outside the Gardner 5×5 area.');
      }
      rows = playable.map(row => compressRow(row.slice(1, 6)));
    }
    if (rows.length !== 5) throw new Error('Engine expects a compact 5×5 or padded Gardner FEN.');
    if (parts[1] && parts[1] !== 'w' && parts[1] !== 'b') throw new Error('Active color must be w or b.');
    const board = new Int8Array(25);
    rows.forEach((row, rowIndex) => {
      const cells = expandRow(row, 5);
      const rank = 4 - rowIndex;
      cells.forEach((symbol, file) => {
        if (!symbol) return;
        const type = CHAR_TO_TYPE[symbol.toLowerCase()];
        if (!type) throw new Error(`Unknown piece in FEN: ${symbol}`);
        board[square(file, rank)] = symbol === symbol.toUpperCase() ? type : -type;
      });
    });
    let whiteKings = 0, blackKings = 0;
    for (let sq = 0; sq < BOARD_N; sq += 1) {
      if (board[sq] === KING) whiteKings += 1;
      else if (board[sq] === -KING) blackKings += 1;
      if (typeOf(board[sq]) === PAWN && (rankOf(sq) === 0 || rankOf(sq) === 4)) {
        throw new Error('A pawn cannot remain on a promotion rank.');
      }
    }
    if (whiteKings !== 1 || blackKings !== 1) throw new Error('Engine positions require exactly one king per side.');
    return new EnginePosition(board, parts[1] === 'b' ? BLACK : WHITE, Number(parts[4] || 0), Number(parts[5] || 1));
  }

  clone() {
    const copy = new EnginePosition(this.board.slice(), this.turn, this.halfmove, this.fullmove);
    copy.hashA = this.hashA;
    copy.hashB = this.hashB;
    return copy;
  }

  recomputeHash() {
    let a = this.turn === BLACK ? TURN_A : 0;
    let b = this.turn === BLACK ? TURN_B : 0;
    for (let sq = 0; sq < 25; sq += 1) {
      const piece = this.board[sq];
      if (!piece) continue;
      const index = zobristIndex(piece);
      a ^= ZOBRIST_A[index][sq];
      b ^= ZOBRIST_B[index][sq];
    }
    this.hashA = a >>> 0;
    this.hashB = b >>> 0;
  }

  key() { return `${this.hashA}:${this.hashB}`; }
}

function expandRow(row, width) {
  const cells = [];
  for (const char of row) {
    if (/\d/.test(char)) {
      for (let i = 0; i < Number(char); i += 1) cells.push(null);
    } else cells.push(char);
  }
  if (cells.length !== width) throw new Error(`FEN row ${row} has width ${cells.length}, expected ${width}.`);
  return cells;
}

function compressRow(cells) {
  let text = '';
  let empty = 0;
  for (const cell of cells) {
    if (!cell) empty += 1;
    else {
      if (empty) text += empty;
      empty = 0;
      text += cell;
    }
  }
  return text + (empty || '');
}

function hashTogglePiece(pos, piece, sq) {
  const index = zobristIndex(piece);
  pos.hashA = (pos.hashA ^ ZOBRIST_A[index][sq]) >>> 0;
  pos.hashB = (pos.hashB ^ ZOBRIST_B[index][sq]) >>> 0;
}

function makeMove(pos, move) {
  const from = moveFrom(move);
  const to = moveTo(move);
  const promotion = movePromotion(move);
  const moving = pos.board[from];
  const captured = pos.board[to];
  const state = {
    captured,
    moving,
    halfmove: pos.halfmove,
    fullmove: pos.fullmove,
    hashA: pos.hashA,
    hashB: pos.hashB,
    turn: pos.turn
  };
  hashTogglePiece(pos, moving, from);
  if (captured) hashTogglePiece(pos, captured, to);
  const placed = promotion ? sideOf(moving) * promotion : moving;
  hashTogglePiece(pos, placed, to);
  pos.board[from] = EMPTY;
  pos.board[to] = placed;
  pos.halfmove = typeOf(moving) === PAWN || captured ? 0 : pos.halfmove + 1;
  if (pos.turn === BLACK) pos.fullmove += 1;
  pos.turn = -pos.turn;
  pos.hashA = (pos.hashA ^ TURN_A) >>> 0;
  pos.hashB = (pos.hashB ^ TURN_B) >>> 0;
  return state;
}

function undoMove(pos, move, state) {
  pos.board[moveFrom(move)] = state.moving;
  pos.board[moveTo(move)] = state.captured;
  pos.turn = state.turn;
  pos.halfmove = state.halfmove;
  pos.fullmove = state.fullmove;
  pos.hashA = state.hashA;
  pos.hashB = state.hashB;
}

function makeNullMove(pos) {
  const state = { turn: pos.turn, halfmove: pos.halfmove, hashA: pos.hashA, hashB: pos.hashB };
  pos.turn = -pos.turn;
  pos.halfmove += 1;
  pos.hashA = (pos.hashA ^ TURN_A) >>> 0;
  pos.hashB = (pos.hashB ^ TURN_B) >>> 0;
  return state;
}

function undoNullMove(pos, state) {
  pos.turn = state.turn;
  pos.halfmove = state.halfmove;
  pos.hashA = state.hashA;
  pos.hashB = state.hashB;
}

function kingSquare(pos, side) {
  const king = side * KING;
  for (let sq = 0; sq < 25; sq += 1) if (pos.board[sq] === king) return sq;
  return -1;
}

function attacksSquare(board, from, target) {
  const piece = board[from];
  if (!piece) return false;
  const side = sideOf(piece);
  const type = typeOf(piece);
  if (type === PAWN) {
    for (const sq of PAWN_TARGETS[side][from]) if (sq === target) return true;
    return false;
  }
  if (type === KNIGHT) {
    for (const sq of KNIGHT_TARGETS[from]) if (sq === target) return true;
    return false;
  }
  if (type === KING) {
    for (const sq of KING_TARGETS[from]) if (sq === target) return true;
    return false;
  }
  const start = type === ROOK ? 4 : 0;
  const end = type === BISHOP ? 4 : 8;
  for (let dir = start; dir < end; dir += 1) {
    for (const sq of RAYS[from][dir]) {
      if (sq === target) return true;
      if (board[sq]) break;
    }
  }
  return false;
}

export function isSquareAttacked(pos, target, bySide) {
  const board = pos.board;
  // Pawn attackers are one rank behind the target from the attacker's view.
  const pawnRank = rankOf(target) - bySide;
  for (const df of [-1, 1]) {
    const f = fileOf(target) + df;
    if (inside(f, pawnRank) && board[square(f, pawnRank)] === bySide * PAWN) return true;
  }
  for (const from of KNIGHT_TARGETS[target]) if (board[from] === bySide * KNIGHT) return true;
  for (const from of KING_TARGETS[target]) if (board[from] === bySide * KING) return true;

  for (let dir = 0; dir < 8; dir += 1) {
    for (const from of RAYS[target][dir]) {
      const piece = board[from];
      if (!piece) continue;
      if (sideOf(piece) === bySide) {
        const type = typeOf(piece);
        if (dir < 4 ? (type === BISHOP || type === QUEEN) : (type === ROOK || type === QUEEN)) return true;
      }
      break;
    }
  }
  return false;
}

export function isInCheck(pos, side = pos.turn) {
  const king = kingSquare(pos, side);
  return king >= 0 && isSquareAttacked(pos, king, -side);
}

function pushPawnMoves(pos, moves, from, side, capturesOnly) {
  const file = fileOf(from), rank = rankOf(from), nextRank = rank + side;
  const promotionRank = side === WHITE ? 4 : 0;
  if (!capturesOnly && inside(file, nextRank)) {
    const to = square(file, nextRank);
    if (!pos.board[to]) {
      if (nextRank === promotionRank) for (const promo of PROMOTIONS) moves.push(encodeMove(from, to, promo));
      else moves.push(encodeMove(from, to));
    }
  }
  for (const to of PAWN_TARGETS[side][from]) {
    const target = pos.board[to];
    if (target && sideOf(target) === -side && typeOf(target) !== KING) {
      if (rankOf(to) === promotionRank) for (const promo of PROMOTIONS) moves.push(encodeMove(from, to, promo));
      else moves.push(encodeMove(from, to));
    }
  }
}

export function generatePseudoMoves(pos, capturesOnly = false) {
  const moves = [];
  const side = pos.turn;
  for (let from = 0; from < BOARD_N; from += 1) {
    const moving = pos.board[from];
    if (sideOf(moving) !== side) continue;
    const type = typeOf(moving);
    if (type === PAWN) {
      pushPawnMoves(pos, moves, from, side, capturesOnly);
      continue;
    }
    if (type === KNIGHT || type === KING) {
      const targets = type === KNIGHT ? KNIGHT_TARGETS[from] : KING_TARGETS[from];
      for (const to of targets) {
        const target = pos.board[to];
        if (!target) {
          if (!capturesOnly) moves.push(encodeMove(from, to));
        } else if (sideOf(target) === -side && typeOf(target) !== KING) moves.push(encodeMove(from, to));
      }
      continue;
    }
    const start = type === ROOK ? 4 : 0;
    const end = type === BISHOP ? 4 : 8;
    for (let dir = start; dir < end; dir += 1) {
      for (const to of RAYS[from][dir]) {
        const target = pos.board[to];
        if (!target) {
          if (!capturesOnly) moves.push(encodeMove(from, to));
          continue;
        }
        if (sideOf(target) === -side && typeOf(target) !== KING) moves.push(encodeMove(from, to));
        break;
      }
    }
  }
  return moves;
}

export function generateLegalMoves(pos, capturesOnly = false) {
  const side = pos.turn;
  const legal = [];
  for (const move of generatePseudoMoves(pos, capturesOnly)) {
    const state = makeMove(pos, move);
    const valid = !isInCheck(pos, side);
    undoMove(pos, move, state);
    if (valid) legal.push(move);
  }
  return legal;
}

function generateLegalTacticalMoves(pos, includeQuietChecks = false) {
  if (includeQuietChecks) {
    return generateLegalMoves(pos, false).filter(move => isCapture(pos, move) || isPromotion(move) || givesCheck(pos, move));
  }
  const candidates = generatePseudoMoves(pos, true);
  const side = pos.turn;
  // Captures-only generation omits non-capturing promotions, which are always tactical.
  for (let from = 0; from < BOARD_N; from += 1) {
    if (pos.board[from] !== side * PAWN) continue;
    const toRank = rankOf(from) + side;
    if (toRank !== (side === WHITE ? 4 : 0)) continue;
    const to = square(fileOf(from), toRank);
    if (!pos.board[to]) for (const promo of PROMOTIONS) candidates.push(encodeMove(from, to, promo));
  }
  const legal = [];
  for (const move of candidates) {
    const state = makeMove(pos, move);
    const valid = !isInCheck(pos, side);
    undoMove(pos, move, state);
    if (valid) legal.push(move);
  }
  return legal;
}

function givesCheck(pos, move) {
  const state = makeMove(pos, move);
  const check = isInCheck(pos, pos.turn);
  undoMove(pos, move, state);
  return check;
}

function isCapture(pos, move) { return pos.board[moveTo(move)] !== EMPTY; }
function isPromotion(move) { return movePromotion(move) !== 0; }
function movePieceType(pos, move) { return typeOf(pos.board[moveFrom(move)]); }
function capturedType(pos, move) { return typeOf(pos.board[moveTo(move)]); }

function leastValuableAttacker(board, target, side) {
  let bestSq = -1, bestValue = INF;
  for (let sq = 0; sq < 25; sq += 1) {
    const piece = board[sq];
    if (sideOf(piece) !== side || !attacksSquare(board, sq, target)) continue;
    const value = PIECE_VALUE[typeOf(piece)];
    if (value < bestValue) { bestValue = value; bestSq = sq; }
  }
  return bestSq;
}

export function staticExchangeEval(pos, move) {
  const target = moveTo(move);
  const moving = pos.board[moveFrom(move)];
  const captured = pos.board[target];
  if (!captured && !isPromotion(move)) return 0;
  const board = pos.board.slice();
  const gains = new Int32Array(32);
  const promo = movePromotion(move);
  gains[0] = PIECE_VALUE[typeOf(captured)] + (promo ? PIECE_VALUE[promo] - PIECE_VALUE[PAWN] : 0);
  board[moveFrom(move)] = EMPTY;
  board[target] = promo ? sideOf(moving) * promo : moving;
  let side = -sideOf(moving);
  let depth = 0;
  while (depth < 30) {
    const attackerSq = leastValuableAttacker(board, target, side);
    if (attackerSq < 0) break;
    depth += 1;
    gains[depth] = PIECE_VALUE[typeOf(board[target])] - gains[depth - 1];
    board[target] = board[attackerSq];
    board[attackerSq] = EMPTY;
    side = -side;
  }
  while (depth > 0) {
    gains[depth - 1] = -Math.max(-gains[depth - 1], gains[depth]);
    depth -= 1;
  }
  return gains[0];
}

function buildAttackInfo(pos, side) {
  const counts = new Int8Array(BOARD_N);
  let mobilityScore = 0;
  const board = pos.board;
  for (let from = 0; from < BOARD_N; from += 1) {
    const piece = board[from];
    if (sideOf(piece) !== side) continue;
    const type = typeOf(piece);
    if (type === PAWN) {
      for (const to of PAWN_TARGETS[side][from]) counts[to] += 1;
      continue;
    }
    if (type === KNIGHT || type === KING) {
      const targets = type === KNIGHT ? KNIGHT_TARGETS[from] : KING_TARGETS[from];
      for (const to of targets) {
        counts[to] += 1;
        if (type !== KING && sideOf(board[to]) !== side) mobilityScore += 3;
      }
      continue;
    }
    const start = type === ROOK ? 4 : 0;
    const end = type === BISHOP ? 4 : 8;
    const unit = type === QUEEN ? 1 : type === ROOK ? 2 : 3;
    for (let dir = start; dir < end; dir += 1) {
      for (const to of RAYS[from][dir]) {
        counts[to] += 1;
        if (sideOf(board[to]) !== side) mobilityScore += unit;
        if (board[to]) break;
      }
    }
  }
  return { counts, mobilityScore };
}

function countPinned(pos, side) {
  const ksq = kingSquare(pos, side);
  if (ksq < 0) return 0;
  let pinned = 0;
  const kf = fileOf(ksq), kr = rankOf(ksq);
  for (const [df, dr] of QUEEN_DIRS) {
    let f = kf + df, r = kr + dr, candidate = -1;
    while (inside(f, r)) {
      const sq = square(f, r), piece = pos.board[sq];
      if (piece) {
        if (candidate < 0 && sideOf(piece) === side) candidate = sq;
        else {
          if (candidate >= 0 && sideOf(piece) === -side) {
            const type = typeOf(piece);
            const diagonal = df !== 0 && dr !== 0;
            if ((diagonal && (type === BISHOP || type === QUEEN)) || (!diagonal && (type === ROOK || type === QUEEN))) pinned += 1;
          }
          break;
        }
      }
      f += df; r += dr;
    }
  }
  return pinned;
}

function pawnStructure(pos, side, ownAttacks, enemyPawnsByFile) {
  const pawnsByFile = [[], [], [], [], []];
  for (let sq = 0; sq < 25; sq += 1) if (pos.board[sq] === side * PAWN) pawnsByFile[fileOf(sq)].push(sq);
  let score = 0;
  for (let file = 0; file < 5; file += 1) {
    const pawns = pawnsByFile[file];
    if (pawns.length > 1) score -= 18 * (pawns.length - 1);
    for (const sq of pawns) {
      const rank = rankOf(sq);
      const advance = side === WHITE ? rank : 4 - rank;
      const distance = 4 - advance;
      const isolated = (file === 0 || !pawnsByFile[file - 1].length) && (file === 4 || !pawnsByFile[file + 1].length);
      if (isolated) score -= 13;
      let passed = true;
      for (let ef = Math.max(0, file - 1); ef <= Math.min(4, file + 1); ef += 1) {
        for (const enemySq of enemyPawnsByFile[ef]) {
          if ((side === WHITE && rankOf(enemySq) > rank) || (side === BLACK && rankOf(enemySq) < rank)) passed = false;
        }
      }
      const protectedPawn = ownAttacks[sq] > 0;
      if (protectedPawn) score += 8;
      if (passed) {
        score += 20 + advance * 15 + (distance <= 1 ? 30 : 0);
        if (protectedPawn) score += 16;
      }
      const aheadRank = rank + side;
      if (inside(file, aheadRank) && pos.board[square(file, aheadRank)]) score -= 9 + advance * 2;
    }
  }
  return score;
}

function kingSafety(pos, side, enemyAttacks, phase) {
  const ksq = kingSquare(pos, side);
  if (ksq < 0) return -MATE;
  const zone = KING_TARGETS[ksq];
  let attackedZone = 0;
  for (const target of zone) attackedZone += enemyAttacks[target];

  // Count each attacking piece once without allocating a Set in the evaluation hot path.
  let attackers = 0;
  for (let from = 0; from < BOARD_N; from += 1) {
    if (sideOf(pos.board[from]) !== -side) continue;
    for (const target of zone) {
      if (attacksSquare(pos.board, from, target)) {
        attackers += 1;
        break;
      }
    }
  }
  const midWeight = phase / MAX_PHASE;
  return -Math.round((attackedZone * 10 + attackers * attackers * 9) * midWeight);
}

function edgeDistance(sq) {
  return Math.min(fileOf(sq), 4 - fileOf(sq), rankOf(sq), 4 - rankOf(sq));
}

function kingDistance(a, b) {
  return Math.max(Math.abs(fileOf(a) - fileOf(b)), Math.abs(rankOf(a) - rankOf(b)));
}

export function isInsufficientMaterial(pos) {
  const material = [];
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const type = typeOf(pos.board[sq]);
    if (!type || type === KING) continue;
    material.push({ type, sq });
  }
  if (!material.length) return true;
  if (material.some(piece => piece.type === PAWN || piece.type === ROOK || piece.type === QUEEN)) return false;
  if (material.length === 1) return material[0].type === BISHOP || material[0].type === KNIGHT;
  if (material.every(piece => piece.type === BISHOP)) {
    const colors = new Set(material.map(piece => (fileOf(piece.sq) + rankOf(piece.sq)) & 1));
    return colors.size === 1;
  }
  return false;
}

function materialProfile(pos) {
  const profile = {
    pieces: 0,
    pawns: 0,
    nonPawnPieces: 0,
    heavyPieces: 0,
    whiteNonKing: 0,
    blackNonKing: 0,
    whitePawns: 0,
    blackPawns: 0
  };
  for (const piece of pos.board) {
    if (!piece) continue;
    profile.pieces += 1;
    const type = typeOf(piece);
    if (type === KING) continue;
    const value = PIECE_VALUE[type];
    if (piece > 0) profile.whiteNonKing += value;
    else profile.blackNonKing += value;
    if (type === PAWN) {
      profile.pawns += 1;
      if (piece > 0) profile.whitePawns += 1;
      else profile.blackPawns += 1;
    } else {
      profile.nonPawnPieces += 1;
      if (type === ROOK || type === QUEEN) profile.heavyPieces += 1;
    }
  }
  return profile;
}

function endgameMopUp(pos, side, ownMaterial, enemyMaterial, ownKing, enemyKing) {
  if (ownKing < 0 || enemyKing < 0 || ownMaterial <= enemyMaterial) return 0;
  const advantage = ownMaterial - enemyMaterial;
  if (advantage < PIECE_VALUE[ROOK] - 40) return 0;
  const driveToEdge = (2 - edgeDistance(enemyKing)) * 18;
  const kingApproach = (4 - Math.min(4, kingDistance(ownKing, enemyKing))) * 9;
  return driveToEdge + kingApproach;
}

export function evaluate(pos) {
  const whiteInfo = buildAttackInfo(pos, WHITE);
  const blackInfo = buildAttackInfo(pos, BLACK);
  const whiteAttacks = whiteInfo.counts;
  const blackAttacks = blackInfo.counts;
  const pawnFiles = { [WHITE]: [[],[],[],[],[]], [BLACK]: [[],[],[],[],[]] };
  let white = 0, black = 0, phase = 0;
  let whiteMaterial = 0, blackMaterial = 0;
  const whiteKing = kingSquare(pos, WHITE), blackKing = kingSquare(pos, BLACK);

  for (const piece of pos.board) if (piece) phase += PHASE_VALUE[typeOf(piece)];
  phase = clamp(phase, 0, MAX_PHASE);
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq];
    if (!piece) continue;
    const side = sideOf(piece), type = typeOf(piece), relSq = side === WHITE ? sq : mirrorSquare(sq);
    let value = PIECE_VALUE[type];
    if (type === KING) {
      const mid = PST.kingMid[relSq], end = PST.kingEnd[relSq];
      value = Math.round((mid * phase + end * (MAX_PHASE - phase)) / MAX_PHASE);
    } else {
      value += PST[type][relSq];
      if (side === WHITE) whiteMaterial += PIECE_VALUE[type];
      else blackMaterial += PIECE_VALUE[type];
    }
    if (side === WHITE) white += value; else black += value;
    if (type === PAWN) pawnFiles[side][fileOf(sq)].push(sq);
  }

  white += whiteInfo.mobilityScore * 3;
  black += blackInfo.mobilityScore * 3;
  white += pawnStructure(pos, WHITE, whiteAttacks, pawnFiles[BLACK]);
  black += pawnStructure(pos, BLACK, blackAttacks, pawnFiles[WHITE]);
  white += kingSafety(pos, WHITE, blackAttacks, phase);
  black += kingSafety(pos, BLACK, whiteAttacks, phase);
  white -= countPinned(pos, WHITE) * 22;
  black -= countPinned(pos, BLACK) * 22;

  // Central control, space, loose pieces and overloaded defenders.
  const central = [6, 7, 8, 11, 12, 13, 16, 17, 18];
  for (const sq of central) {
    white += whiteAttacks[sq] * (sq === 12 ? 5 : 2);
    black += blackAttacks[sq] * (sq === 12 ? 5 : 2);
  }
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    if (rankOf(sq) >= 2) white += whiteAttacks[sq] ? 1 : 0;
    if (rankOf(sq) <= 2) black += blackAttacks[sq] ? 1 : 0;
    const piece = pos.board[sq];
    if (!piece || typeOf(piece) === KING) continue;
    const side = sideOf(piece), own = side === WHITE ? whiteAttacks : blackAttacks, enemy = side === WHITE ? blackAttacks : whiteAttacks;
    const value = PIECE_VALUE[typeOf(piece)];
    let adjustment = 0;
    if (enemy[sq] && !own[sq]) adjustment -= Math.min(72, Math.round(value / 10));
    if (enemy[sq] >= 2 && own[sq] <= 1) adjustment -= Math.min(38, Math.round(value / 22));
    if (side === WHITE) white += adjustment; else black += adjustment;
  }

  // Rooks and queens on files without friendly pawns; fully open files are best.
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq], type = typeOf(piece);
    if (!piece || (type !== ROOK && type !== QUEEN)) continue;
    const side = sideOf(piece), file = fileOf(sq);
    const ownPawns = pawnFiles[side][file].length;
    const enemyPawns = pawnFiles[-side][file].length;
    const bonus = ownPawns ? 0 : enemyPawns ? 10 : 18;
    if (side === WHITE) white += bonus; else black += bonus;
  }

  // Queen pressure is useful on a compact board, but exposed queens are also tactical targets.
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq];
    if (typeOf(piece) !== QUEEN) continue;
    const side = sideOf(piece);
    const enemyKing = side === WHITE ? blackKing : whiteKing;
    if (enemyKing < 0) continue;
    const distance = Math.abs(fileOf(sq) - fileOf(enemyKing)) + Math.abs(rankOf(sq) - rankOf(enemyKing));
    const bonus = Math.max(0, 5 - distance) * 4;
    if (side === WHITE) white += bonus; else black += bonus;
  }

  // Endgame technique: king activity and driving a bare king toward the rim.
  const endWeight = (MAX_PHASE - phase) / MAX_PHASE;
  if (whiteKing >= 0 && blackKing >= 0) {
    white += Math.round(endgameMopUp(pos, WHITE, whiteMaterial, blackMaterial, whiteKing, blackKing) * endWeight);
    black += Math.round(endgameMopUp(pos, BLACK, blackMaterial, whiteMaterial, blackKing, whiteKing) * endWeight);
    if (phase <= 5) {
      white += Math.round((2 - edgeDistance(whiteKing)) * -4 + (4 - kingDistance(whiteKing, blackKing)) * 2);
      black += Math.round((2 - edgeDistance(blackKing)) * -4 + (4 - kingDistance(whiteKing, blackKing)) * 2);
    }
  }

  const tempo = phase <= 4 ? 7 : 12;
  const score = white - black + (pos.turn === WHITE ? tempo : -tempo);
  return pos.turn === WHITE ? score : -score;
}

function isPruningEndgame(pos) {
  const profile = materialProfile(pos);
  return profile.pieces <= 7 || profile.heavyPieces === 0 && profile.nonPawnPieces <= 2;
}

function hasNonPawnMaterial(pos, side) {
  for (const piece of pos.board) if (sideOf(piece) === side && ![PAWN, KING].includes(typeOf(piece))) return true;
  return false;
}

function nullMoveSafe(pos) {
  let nonPawn = 0, pieces = 0, advancedPawn = false;
  for (let sq = 0; sq < 25; sq += 1) {
    const piece = pos.board[sq];
    if (sideOf(piece) !== pos.turn) continue;
    pieces += 1;
    const type = typeOf(piece);
    if (![PAWN, KING].includes(type)) nonPawn += type >= ROOK ? 2 : 1;
    if (type === PAWN) {
      const advance = pos.turn === WHITE ? rankOf(sq) : 4 - rankOf(sq);
      if (advance >= 3) advancedPawn = true;
    }
  }
  return pieces >= 5 && nonPawn >= 2 && !advancedPawn;
}

function isPassedPawnMove(pos, move) {
  const moving = pos.board[moveFrom(move)];
  if (typeOf(moving) !== PAWN) return false;
  const side = sideOf(moving), to = moveTo(move), file = fileOf(to), rank = rankOf(to);
  for (let sq = 0; sq < 25; sq += 1) {
    if (pos.board[sq] !== -side * PAWN) continue;
    if (Math.abs(fileOf(sq) - file) > 1) continue;
    if ((side === WHITE && rankOf(sq) > rank) || (side === BLACK && rankOf(sq) < rank)) return false;
  }
  return true;
}

function moveOrderScore(pos, move, ttMove, ply, searcher, previousMove, endgameChecks = false) {
  if (move === ttMove) return 2_000_000;
  const promo = movePromotion(move);
  const capture = pos.board[moveTo(move)];
  if (promo) return 1_300_000 + PIECE_VALUE[promo] * 10 + PIECE_VALUE[typeOf(capture)];
  if (capture) {
    const sideIndex = pos.turn === WHITE ? 0 : 1;
    const movingType = typeOf(pos.board[moveFrom(move)]);
    const historyIndex = movingType * 25 + moveTo(move);
    return 1_000_000 + PIECE_VALUE[typeOf(capture)] * 16 - PIECE_VALUE[movingType] + staticExchangeEval(pos, move) + searcher.captureHistory[sideIndex][historyIndex];
  }
  if (searcher.killers[ply][0] === move) return 900_000;
  if (searcher.killers[ply][1] === move) return 850_000;
  if (previousMove && searcher.countermoves[previousMove & 8191] === move) return 820_000;
  if (endgameChecks && givesCheck(pos, move)) return 780_000;
  const sideIndex = pos.turn === WHITE ? 0 : 1;
  return searcher.history[sideIndex][moveFrom(move) * 25 + moveTo(move)];
}

function insertionSortMoves(pos, moves, ttMove, ply, searcher, previousMove, endgameChecks = false) {
  const scores = new Int32Array(moves.length);
  for (let i = 0; i < moves.length; i += 1) scores[i] = moveOrderScore(pos, moves[i], ttMove, ply, searcher, previousMove, endgameChecks);
  for (let i = 1; i < moves.length; i += 1) {
    const move = moves[i];
    const score = scores[i];
    let j = i - 1;
    while (j >= 0 && scores[j] < score) {
      moves[j + 1] = moves[j];
      scores[j + 1] = scores[j];
      j -= 1;
    }
    moves[j + 1] = move;
    scores[j + 1] = score;
  }
  return moves;
}

function wdlFromScore(score, phaseHint = 0.5) {
  if (score >= MATE_BOUND) return { win: 100, draw: 0, loss: 0 };
  if (score <= -MATE_BOUND) return { win: 0, draw: 0, loss: 100 };
  const abs = Math.abs(score);
  const draw = clamp(Math.round(58 * Math.exp(-abs / (235 + phaseHint * 80))), 3, 72);
  const decisive = 100 - draw;
  const whiteShare = 1 / (1 + Math.exp(-score / 145));
  const win = Math.round(decisive * whiteShare);
  return { win, draw, loss: 100 - draw - win };
}

export function scoreToDisplay(score) {
  if (score >= MATE_BOUND) return `#${Math.max(1, Math.ceil((MATE - score) / 2))}`;
  if (score <= -MATE_BOUND) return `-#${Math.max(1, Math.ceil((MATE + score) / 2))}`;
  const pawns = score / 100;
  return `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}`;
}

export class GardnerSearcher {
  constructor({ hashEntries = 180000 } = {}) {
    const requested = Math.max(16_384, Number(hashEntries) || 180_000);
    let buckets = 1;
    while (buckets < requested / 2) buckets <<= 1;
    this.ttBuckets = buckets;
    this.ttMask = buckets - 1;
    this.hashEntries = buckets * 2;
    this.ttUsed = new Uint8Array(this.hashEntries);
    this.ttKey = new Uint32Array(this.hashEntries);
    this.ttLock = new Uint32Array(this.hashEntries);
    this.ttDepth = new Int8Array(this.hashEntries);
    this.ttScore = new Int16Array(this.hashEntries);
    this.ttEval = new Int16Array(this.hashEntries);
    this.ttMove = new Uint16Array(this.hashEntries);
    this.ttFlag = new Uint8Array(this.hashEntries);
    this.ttGeneration = new Uint8Array(this.hashEntries);
    this.ttOccupied = 0;
    this.generation = 0;
    this.history = [new Int32Array(625), new Int32Array(625)];
    this.killers = Array.from({ length: MAX_PLY }, () => new Int32Array(2));
    this.countermoves = new Uint16Array(8192);
    this.captureHistory = [new Int32Array(175), new Int32Array(175)];
    this.evalMask = 65535;
    this.evalUsed = new Uint8Array(this.evalMask + 1);
    this.evalKey = new Uint32Array(this.evalMask + 1);
    this.evalLock = new Uint32Array(this.evalMask + 1);
    this.evalScore = new Int16Array(this.evalMask + 1);
    this.pvTable = Array.from({ length: MAX_PLY + 1 }, () => new Int32Array(MAX_PLY + 1));
    this.pvLength = new Int16Array(MAX_PLY + 1);
    this.staticEvalStack = new Int32Array(MAX_PLY + 2);
    this.hashStackA = new Uint32Array(MAX_PLY + 128);
    this.hashStackB = new Uint32Array(MAX_PLY + 128);
    this.nodes = 0;
    this.selDepth = 0;
    this.deadline = Infinity;
    this.rootBookMoves = new Set();
    this.rootHistory = [];
    this.previousRootScores = new Map();
    this.previousPVScore = new Int32Array(8);
    this.completedDepth = 0;
    this.lastLines = [];
    this.startedAt = 0;
  }

  clear() {
    this.ttUsed.fill(0);
    this.ttOccupied = 0;
    this.history.forEach(table => table.fill(0));
    this.killers.forEach(row => row.fill(0));
    this.countermoves.fill(0);
    this.captureHistory.forEach(table => table.fill(0));
    this.evalUsed.fill(0);
    this.previousRootScores.clear();
    this.previousPVScore.fill(0);
    this.completedDepth = 0;
    this.lastLines = [];
  }

  beginPosition() {
    // Retain the transposition table, but age volatile ordering heuristics.
    for (const table of this.history) {
      for (let i = 0; i < table.length; i += 1) table[i] = table[i] >> 1;
    }
    this.killers.forEach(row => row.fill(0));
    this.countermoves.fill(0);
    for (const table of this.captureHistory) {
      for (let i = 0; i < table.length; i += 1) table[i] = table[i] >> 1;
    }
    this.previousRootScores.clear();
    this.previousPVScore.fill(0);
    this.completedDepth = 0;
    this.lastLines = [];
  }

  setBookMoves(uciMoves = []) {
    this.rootBookMoves = new Set(uciMoves);
  }

  staticEvaluate(pos) {
    const index = pos.hashA & this.evalMask;
    if (this.evalUsed[index] && this.evalKey[index] === pos.hashA && this.evalLock[index] === pos.hashB) {
      return this.evalScore[index];
    }
    const score = evaluate(pos);
    this.evalUsed[index] = 1;
    this.evalKey[index] = pos.hashA;
    this.evalLock[index] = pos.hashB;
    this.evalScore[index] = clamp(score, -32767, 32767);
    return score;
  }

  seedFromResult(pos, result) {
    if (!result || !Array.isArray(result.lines) || !result.lines.length) return;
    const rootSide = pos.turn;
    const seeded = [];
    for (const line of result.lines.slice(0, 5)) {
      const cursor = pos.clone();
      const pv = [];
      for (const uci of line.pv || []) {
        const move = uciToMove(cursor, uci);
        if (!move) break;
        pv.push(move);
        makeMove(cursor, move);
      }
      if (!pv.length) continue;
      const whiteScore = Number(line.score || 0);
      seeded.push({
        move: pv[0],
        score: rootSide === WHITE ? whiteScore : -whiteScore,
        pv
      });
    }
    if (!seeded.length) return;
    this.lastLines = seeded;
    this.completedDepth = Math.max(this.completedDepth, Number(result.depth || 0));
    seeded.forEach((line, index) => {
      this.previousPVScore[index] = line.score;
      this.previousRootScores.set(line.move, line.score);
    });
  }

  shouldStop() {
    if ((this.nodes & 255) === 0 && performance.now() >= this.deadline) throw ABORT;
  }

  repetitionCount(pos, ply) {
    let count = 1; // Current position.
    const limit = Math.min(ply, pos.halfmove);
    for (let back = 2; back <= limit; back += 2) {
      const index = ply - back;
      if (this.hashStackA[index] === pos.hashA && this.hashStackB[index] === pos.hashB) count += 1;
    }
    for (let index = this.rootHistory.length - 1; index >= 0; index -= 1) {
      const key = this.rootHistory[index];
      if (key.a === pos.hashA && key.b === pos.hashB) count += 1;
    }
    return count;
  }

  isRepetition(pos, ply) {
    return this.repetitionCount(pos, ply) >= 3;
  }

  probeTT(pos, ply) {
    const keyA = ttContextKeyA(pos);
    const keyB = ttContextKeyB(pos);
    const base = (keyA & this.ttMask) << 1;
    for (let slot = base; slot <= base + 1; slot += 1) {
      if (!this.ttUsed[slot] || this.ttKey[slot] !== keyA || this.ttLock[slot] !== keyB) continue;
      return {
        depth: this.ttDepth[slot],
        score: scoreFromTT(this.ttScore[slot], ply),
        staticEval: this.ttEval[slot],
        move: this.ttMove[slot],
        flag: this.ttFlag[slot],
        generation: this.ttGeneration[slot]
      };
    }
    return null;
  }

  storeTT(pos, depth, score, flag, move, staticEval, ply) {
    const keyA = ttContextKeyA(pos);
    const keyB = ttContextKeyB(pos);
    const base = (keyA & this.ttMask) << 1;
    let target = -1;
    for (let slot = base; slot <= base + 1; slot += 1) {
      if (this.ttUsed[slot] && this.ttKey[slot] === keyA && this.ttLock[slot] === keyB) {
        if (this.ttDepth[slot] > depth && this.ttGeneration[slot] === this.generation && flag !== TT_EXACT) return;
        target = slot;
        break;
      }
      if (!this.ttUsed[slot] && target < 0) target = slot;
    }
    if (target < 0) {
      const a = base, b = base + 1;
      const ageA = (this.generation - this.ttGeneration[a]) & 255;
      const ageB = (this.generation - this.ttGeneration[b]) & 255;
      const qualityA = this.ttDepth[a] - ageA * 4 + (this.ttFlag[a] === TT_EXACT ? 3 : 0);
      const qualityB = this.ttDepth[b] - ageB * 4 + (this.ttFlag[b] === TT_EXACT ? 3 : 0);
      target = qualityA <= qualityB ? a : b;
    }
    if (!this.ttUsed[target]) {
      this.ttUsed[target] = 1;
      this.ttOccupied += 1;
    }
    this.ttKey[target] = keyA;
    this.ttLock[target] = keyB;
    this.ttDepth[target] = clamp(depth, -1, 127);
    this.ttScore[target] = scoreToTT(score, ply);
    this.ttEval[target] = clamp(staticEval, -32767, 32767);
    this.ttMove[target] = move || 0;
    this.ttFlag[target] = flag;
    this.ttGeneration[target] = this.generation;
  }

  qsearch(pos, alpha, beta, ply, qDepth = 0, previousMove = 0) {
    this.nodes += 1;
    this.selDepth = Math.max(this.selDepth, ply);
    this.shouldStop();
    if (ply >= MAX_PLY - 2) return this.staticEvaluate(pos);
    if (pos.halfmove >= 100 || this.isRepetition(pos, ply) || isInsufficientMaterial(pos)) return DRAW;

    const inCheck = isInCheck(pos);
    let standPat = inCheck ? -INF : this.staticEvaluate(pos);
    if (!inCheck) {
      if (standPat >= beta) return standPat;
      if (standPat > alpha) alpha = standPat;
      if (qDepth >= 12) return standPat;
    }

    let moves;
    if (inCheck) moves = generateLegalMoves(pos, false);
    else moves = generateLegalTacticalMoves(pos, qDepth < 3);
    if (!moves.length) return inCheck ? -MATE + ply : standPat;
    moves = insertionSortMoves(pos, moves, 0, ply, this, previousMove);

    // At extreme quiescence depth, still resolve check with one legal evasion.
    if (qDepth >= 16) {
      let best = -INF;
      for (const move of moves) {
        const state = makeMove(pos, move);
        const score = -this.staticEvaluate(pos);
        undoMove(pos, move, state);
        if (score > best) best = score;
      }
      return best;
    }

    for (const move of moves) {
      const capture = capturedType(pos, move);
      const promotion = isPromotion(move);
      let checking = false;
      if (!inCheck && !promotion && !capture) checking = true; // Quiet moves here are prefiltered checks.
      if (!inCheck && !checking && !promotion) {
        if (standPat + PIECE_VALUE[capture] + 80 < alpha) continue;
        if (staticExchangeEval(pos, move) < -35) continue;
      }
      const state = makeMove(pos, move);
      this.hashStackA[ply + 1] = pos.hashA;
      this.hashStackB[ply + 1] = pos.hashB;
      const score = -this.qsearch(pos, -beta, -alpha, ply + 1, qDepth + 1, move);
      undoMove(pos, move, state);
      if (score >= beta) return score;
      if (score > alpha) alpha = score;
    }
    return alpha;
  }

  search(pos, depth, alpha, beta, ply, pvNode, previousMove = 0, excludedMove = 0, allowNull = true, extensions = 0) {
    this.nodes += 1;
    this.selDepth = Math.max(this.selDepth, ply);
    this.pvLength[ply] = ply;
    this.shouldStop();

    if (ply >= MAX_PLY - 2) return this.staticEvaluate(pos);
    if (pos.halfmove >= 100 || this.isRepetition(pos, ply) || isInsufficientMaterial(pos)) return DRAW;
    alpha = Math.max(alpha, -MATE + ply);
    beta = Math.min(beta, MATE - ply - 1);
    if (alpha >= beta) return alpha;
    if (depth <= 0) return this.qsearch(pos, alpha, beta, ply, 0, previousMove);

    const inCheck = isInCheck(pos);
    const pruningEndgame = isPruningEndgame(pos);
    const tt = excludedMove ? null : this.probeTT(pos, ply);
    const ttMove = tt?.move || 0;
    if (tt && tt.depth >= depth && !pvNode) {
      if (tt.flag === TT_EXACT) return tt.score;
      if (tt.flag === TT_LOWER && tt.score >= beta) return tt.score;
      if (tt.flag === TT_UPPER && tt.score <= alpha) return tt.score;
    }

    const staticEval = inCheck ? -INF : (tt?.staticEval ?? this.staticEvaluate(pos));
    const improving = ply >= 2 && staticEval > this.staticEvalStack[ply - 2];
    this.staticEvalStack[ply] = staticEval;

    // Razoring: shallow hopeless nodes fall directly into quiescence.
    if (!pvNode && !inCheck && !pruningEndgame && depth <= 2 && staticEval + 175 * depth < alpha) {
      const razor = this.qsearch(pos, alpha, beta, ply, 0, previousMove);
      if (razor <= alpha) return razor;
    }

    // Reverse futility pruning, deliberately conservative on the compact board.
    if (!pvNode && !inCheck && !pruningEndgame && depth <= 5 && !isMateScore(beta) && staticEval - (105 + 58 * depth) >= beta && hasNonPawnMaterial(pos, pos.turn)) {
      return staticEval;
    }

    // Null move with verification at high depth. Disabled in likely 5×5 zugzwangs.
    if (allowNull && !pvNode && !inCheck && !pruningEndgame && depth >= 4 && staticEval >= beta && nullMoveSafe(pos)) {
      const reduction = 2 + Math.floor(depth / 4) + Math.min(2, Math.floor((staticEval - beta) / 180));
      const state = makeNullMove(pos);
      this.hashStackA[ply + 1] = pos.hashA;
      this.hashStackB[ply + 1] = pos.hashB;
      const nullScore = -this.search(pos, depth - reduction - 1, -beta, -beta + 1, ply + 1, false, 0, 0, false, extensions);
      undoNullMove(pos, state);
      if (nullScore >= beta && !isMateScore(nullScore)) {
        if (depth < 8) return nullScore;
        const verify = this.search(pos, depth - reduction, beta - 1, beta, ply, false, previousMove, 0, false, extensions);
        if (verify >= beta) return nullScore;
      }
    }

    // ProbCut with good captures only.
    if (!pvNode && !inCheck && !pruningEndgame && depth >= 5 && !isMateScore(beta)) {
      const probBeta = beta + 140;
      let tactical = generateLegalMoves(pos, true)
        .filter(move => isPromotion(move) || staticExchangeEval(pos, move) >= 40);
      tactical = insertionSortMoves(pos, tactical, ttMove, ply, this, previousMove).slice(0, 5);
      for (const move of tactical) {
        const state = makeMove(pos, move);
        this.hashStackA[ply + 1] = pos.hashA;
        this.hashStackB[ply + 1] = pos.hashB;
        let score = -this.qsearch(pos, -probBeta, -probBeta + 1, ply + 1, 0, move);
        if (score >= probBeta) score = -this.search(pos, depth - 4, -probBeta, -probBeta + 1, ply + 1, false, move, 0, true, extensions);
        undoMove(pos, move, state);
        if (score >= probBeta) return score;
      }
    }

    let moves = generateLegalMoves(pos, false);
    if (excludedMove) moves = moves.filter(move => move !== excludedMove);
    if (!moves.length) return inCheck ? -MATE + ply : DRAW;
    moves = insertionSortMoves(pos, moves, ttMove, ply, this, previousMove, pruningEndgame);

    let singularMove = 0;
    if (!excludedMove && depth >= 7 && ttMove && tt && tt.depth >= depth - 2 && tt.flag !== TT_UPPER && !isMateScore(tt.score)) {
      const singularBeta = tt.score - 36 - depth * 2;
      const singularDepth = Math.max(1, Math.floor((depth - 1) / 2));
      const value = this.search(pos, singularDepth, singularBeta - 1, singularBeta, ply, false, previousMove, ttMove, false, extensions);
      if (value < singularBeta) singularMove = ttMove;
    }

    const originalAlpha = alpha;
    let bestScore = -INF, bestMove = 0, legalIndex = 0, quietTried = 0;
    const searchedQuiets = [];
    for (const move of moves) {
      const capture = isCapture(pos, move);
      const promotion = isPromotion(move);
      const quiet = !capture && !promotion;
      if (quiet) quietTried += 1;

      // Late move and futility pruning. Never applied in check or to root/book moves.
      if (!pvNode && !inCheck && !pruningEndgame && depth <= 3 && quiet && move !== ttMove) {
        const lmpLimit = depth === 1 ? 5 : depth === 2 ? 9 : 15;
        if (quietTried > lmpLimit) continue;
        const margin = 85 * depth + (improving ? 55 : 0);
        if (staticEval + margin <= alpha && !givesCheck(pos, move) && !isPassedPawnMove(pos, move)) continue;
      }
      if (!pvNode && !inCheck && !pruningEndgame && depth <= 4 && capture && !promotion && staticExchangeEval(pos, move) < -55 * depth) continue;
      if (quiet) searchedQuiets.push(move);

      const movedType = movePieceType(pos, move);
      const recapture = previousMove && capture && moveTo(previousMove) === moveTo(move);
      const passedPush = movedType === PAWN && isPassedPawnMove(pos, move);
      const checking = givesCheck(pos, move);
      let extension = 0;
      if (extensions < 2) {
        const nearPromotion = passedPush && (rankOf(moveTo(move)) === 3 || rankOf(moveTo(move)) === 1);
        const forcingCheck = checking && (depth <= 5 || moves.length <= 4);
        const soundRecapture = recapture && depth <= 3 && staticExchangeEval(pos, move) >= -20;
        if (move === singularMove || promotion || nearPromotion || forcingCheck || soundRecapture) extension = 1;
      }
      const nextExtensions = extensions + extension;
      const newDepth = depth - 1 + extension;

      const state = makeMove(pos, move);
      this.hashStackA[ply + 1] = pos.hashA;
      this.hashStackB[ply + 1] = pos.hashB;
      let score;
      if (legalIndex === 0) {
        score = -this.search(pos, newDepth, -beta, -alpha, ply + 1, pvNode, move, 0, true, nextExtensions);
      } else {
        let reduction = 0;
        if (depth >= 3 && quiet && !inCheck && !checking && !promotion && move !== ttMove && !passedPush) {
          reduction = LMR_TABLE[Math.min(32, depth)][Math.min(32, legalIndex + 1)];
          if (pvNode) reduction -= 1;
          if (improving) reduction -= 1;
          if (pruningEndgame) reduction -= 1;
          if (this.killers[ply][0] === move || this.killers[ply][1] === move) reduction -= 1;
          reduction = clamp(reduction, 0, Math.max(0, newDepth - 1));
        }
        score = -this.search(pos, newDepth - reduction, -alpha - 1, -alpha, ply + 1, false, move, 0, true, nextExtensions);
        if (reduction && score > alpha) score = -this.search(pos, newDepth, -alpha - 1, -alpha, ply + 1, false, move, 0, true, nextExtensions);
        if (score > alpha && score < beta) score = -this.search(pos, newDepth, -beta, -alpha, ply + 1, pvNode, move, 0, true, nextExtensions);
      }
      undoMove(pos, move, state);
      legalIndex += 1;

      if (score > bestScore) { bestScore = score; bestMove = move; }
      if (score > alpha) {
        alpha = score;
        this.pvTable[ply][ply] = move;
        for (let i = ply + 1; i < this.pvLength[ply + 1]; i += 1) this.pvTable[ply][i] = this.pvTable[ply + 1][i];
        this.pvLength[ply] = this.pvLength[ply + 1];
        if (alpha >= beta) {
          if (capture) {
            const sideIndex = pos.turn === WHITE ? 0 : 1;
            const historyIndex = movedType * 25 + moveTo(move);
            this.captureHistory[sideIndex][historyIndex] = clamp(
              this.captureHistory[sideIndex][historyIndex] + depth * depth * 10,
              -12000,
              12000
            );
          }
          if (quiet) {
            if (this.killers[ply][0] !== move) {
              this.killers[ply][1] = this.killers[ply][0];
              this.killers[ply][0] = move;
            }
            const sideIndex = pos.turn === WHITE ? 0 : 1;
            const bonus = depth * depth * 18;
            const index = moveFrom(move) * 25 + moveTo(move);
            this.history[sideIndex][index] = clamp(this.history[sideIndex][index] + bonus, -30000, 30000);
            const malus = Math.max(8, Math.floor(bonus / 3));
            for (const tried of searchedQuiets) {
              if (tried === move) continue;
              const triedIndex = moveFrom(tried) * 25 + moveTo(tried);
              this.history[sideIndex][triedIndex] = clamp(this.history[sideIndex][triedIndex] - malus, -30000, 30000);
            }
            if (previousMove) this.countermoves[previousMove & 8191] = move;
          }
          break;
        }
      }
    }

    const flag = bestScore <= originalAlpha ? TT_UPPER : bestScore >= beta ? TT_LOWER : TT_EXACT;
    if (!excludedMove) this.storeTT(pos, depth, bestScore, flag, bestMove, staticEval, ply);
    return bestScore;
  }

  rootSearch(pos, depth, alpha, beta, excluded, preferredMove = 0) {
    let moves = generateLegalMoves(pos, false).filter(move => !excluded.has(move));
    if (!moves.length) return null;
    const ttMove = preferredMove || this.probeTT(pos, 0)?.move || 0;
    moves = insertionSortMoves(pos, moves, ttMove, 0, this, 0, isPruningEndgame(pos));
    moves.sort((a, b) => {
      const aBook = this.rootBookMoves.has(moveToUci(a)) ? 1 : 0;
      const bBook = this.rootBookMoves.has(moveToUci(b)) ? 1 : 0;
      if (aBook !== bBook) return bBook - aBook;
      return (this.previousRootScores.get(b) ?? -INF) - (this.previousRootScores.get(a) ?? -INF);
    });

    let bestScore = -INF, bestMove = 0, bestPV = [], index = 0;
    for (const move of moves) {
      const state = makeMove(pos, move);
      this.hashStackA[1] = pos.hashA;
      this.hashStackB[1] = pos.hashB;
      this.pvLength[1] = 1;
      let score;
      if (index === 0) score = -this.search(pos, depth - 1, -beta, -alpha, 1, true, move);
      else {
        score = -this.search(pos, depth - 1, -alpha - 1, -alpha, 1, false, move);
        if (score > alpha && score < beta) score = -this.search(pos, depth - 1, -beta, -alpha, 1, true, move);
      }
      undoMove(pos, move, state);
      this.previousRootScores.set(move, score);
      index += 1;
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
        bestPV = [move];
        for (let i = 1; i < this.pvLength[1]; i += 1) bestPV.push(this.pvTable[1][i]);
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }
    return { score: bestScore, move: bestMove, pv: bestPV };
  }

  searchMultiPV(pos, depth, multipv = 3) {
    const legalCount = generateLegalMoves(pos, false).length;
    if (!legalCount) return [];
    const limit = Math.min(Math.max(1, multipv), legalCount);
    const excluded = new Set();
    const results = [];

    for (let pvIndex = 0; pvIndex < limit; pvIndex += 1) {
      const priorLine = this.lastLines[pvIndex];
      const preferredMove = priorLine && !excluded.has(priorLine.move) ? priorLine.move : 0;
      const previous = priorLine?.score ?? this.previousPVScore[pvIndex] ?? 0;
      let delta = depth >= 4 ? 34 : INF;
      let alpha = delta === INF ? -INF : previous - delta;
      let beta = delta === INF ? INF : previous + delta;
      let result = null;

      while (true) {
        result = this.rootSearch(pos, depth, alpha, beta, excluded, preferredMove);
        if (!result) break;
        if (alpha > -INF && result.score <= alpha) {
          delta = Math.min(INF, delta * 2);
          alpha = delta >= INF ? -INF : Math.max(-INF, result.score - delta);
          beta = delta >= INF ? INF : Math.min(INF, result.score + Math.floor(delta / 2));
          continue;
        }
        if (beta < INF && result.score >= beta) {
          delta = Math.min(INF, delta * 2);
          alpha = delta >= INF ? -INF : Math.max(-INF, result.score - Math.floor(delta / 2));
          beta = delta >= INF ? INF : Math.min(INF, result.score + delta);
          continue;
        }
        break;
      }
      if (!result) break;
      results.push(result);
      excluded.add(result.move);
      this.previousPVScore[pvIndex] = result.score;
    }

    results.sort((a, b) => b.score - a.score || Number(this.rootBookMoves.has(moveToUci(b.move))) - Number(this.rootBookMoves.has(moveToUci(a.move))));
    return results;
  }

  analyze(pos, {
    timeMs = 950,
    maxDepth = 32,
    multipv = 3,
    startDepth = 1,
    bookMoves = [],
    historyKeys = [],
    newPosition = true,
    resumeResult = null
  } = {}) {
    const rootSide = pos.turn;
    if (newPosition) this.beginPosition();
    if (resumeResult) this.seedFromResult(pos, resumeResult);
    this.generation = (this.generation + 1) & 255;
    this.nodes = 0;
    this.selDepth = 0;
    this.deadline = performance.now() + Math.max(80, timeMs);
    this.startedAt = performance.now();
    this.setBookMoves(bookMoves);
    this.rootHistory = Array.isArray(historyKeys) ? historyKeys : [];
    this.hashStackA[0] = pos.hashA;
    this.hashStackB[0] = pos.hashB;
    const rootMoves = generateLegalMoves(pos);
    const rootTerminal = !rootMoves.length || pos.halfmove >= 100 || isInsufficientMaterial(pos) || this.repetitionCount(pos, 0) >= 3;
    let completed = null;
    let depth = Math.max(1, startDepth);
    try {
      if (!rootTerminal) for (; depth <= maxDepth; depth += 1) {
        const lines = this.searchMultiPV(pos, depth, multipv);
        if (!lines.length) break;
        completed = { depth, lines };
        this.lastLines = lines;
        this.completedDepth = depth;
        if (isMateScore(lines[0].score) && depth >= 3) break;
        this.shouldStop();
      }
    } catch (error) {
      if (error !== ABORT) throw error;
    }
    const elapsed = Math.max(1, performance.now() - this.startedAt);
    const lines = (completed?.lines || this.lastLines || []).map(line => {
      const whiteScore = rootSide === WHITE ? line.score : -line.score;
      return {
        move: moveToUci(line.move),
        score: whiteScore,
        scoreText: scoreToDisplay(whiteScore),
        wdl: wdlFromScore(whiteScore),
        pv: line.pv.map(moveToUci)
      };
    });
    return {
      engine: ENGINE_VERSION,
      depth: completed?.depth || this.completedDepth || 0,
      selDepth: this.selDepth,
      nodes: this.nodes,
      nps: Math.round(this.nodes * 1000 / elapsed),
      elapsed: Math.round(elapsed),
      lines,
      terminal: rootTerminal,
      completed: Boolean(completed),
      attemptedDepth: Math.max(1, startDepth),
      hashfull: Math.round(this.ttOccupied * 1000 / this.hashEntries),
      nextDepth: completed
        ? Math.max(1, completed.depth + 1)
        : Math.max(1, startDepth)
    };
  }
}

export function analyzeOnce(fen, options = {}) {
  const searcher = new GardnerSearcher(options);
  return searcher.analyze(EnginePosition.fromFEN(fen), options);
}

export const EngineInternals = Object.freeze({
  EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  encodeMove, moveFrom, moveTo, movePromotion, makeMove, undoMove, givesCheck,
  isCapture, isPromotion, kingSquare, isPruningEndgame, materialProfile,
  PIECE_VALUE, MATE, MATE_BOUND
});
