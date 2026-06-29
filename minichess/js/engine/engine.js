// Gardner MiniChess classical analysis engine.
// Native 25-square board, iterative deepening PVS, quiescence and a
// Stockfish-style tablebase integration: tablebase information is consumed
// inside the ordinary alpha-beta tree, never by a separate search path.

export const ENGINE_VERSION = 'Orion JS 22.2';

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
const TB_WIN_SCORE = 22000;
const TB_BOUND_MAX_PLY = MAX_PLY;
const TB_WIN_MIN_SCORE = TB_WIN_SCORE - TB_BOUND_MAX_PLY;
// WDL-only tablebase probes are internal alpha-beta bounds. Exact DTM probes
// are translated to mate-distance scores; WDL bounds are never shown as a user
// score, because they do not contain an exact mate distance.
const MIN_TABLEBASE_AUDIT_SCORE = 800;
const MAX_TABLEBASE_AUDIT_SCORE = 6000;
// Display-only cap for optional TT PV reconstruction. It is never a search
// completion requirement: a valid PV may stop at a draw by repetition.
const PV_DISPLAY_TAIL_TARGET = 64;
// v18.4 streams lightweight live root snapshots from the synchronous search.
// The UI still paints at a fixed 500 ms cadence; this only keeps the queued
// snapshot current while a longer depth slice is running.
const PROGRESS_NODE_INTERVAL = 4096;
const PROGRESS_MIN_INTERVAL_MS = 90;
// UI paints at 500 ms. Emit a lightweight snapshot sooner if a slow branch has
// not yet crossed the usual node interval, so the displayed NODES stays fresh.
const PROGRESS_MAX_STALENESS_MS = 180;
// Low-progress roots retain their hard draw compression, but receive a bounded
// audit budget for sacrificial attacks, king entries, and zugzwang resources.
const LOW_PROGRESS_AUDIT_MULTIPLIER = 1.25;

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

const MAX_GENERATED_MOVES = 256;
const MOVE_FLAG_CAPTURE = 1;
const MOVE_FLAG_PROMOTION = 2;
const MOVE_FLAG_CHECK = 4;

function createMoveList() {
  return {
    moves: new Uint16Array(MAX_GENERATED_MOVES),
    flags: new Uint8Array(MAX_GENERATED_MOVES),
    movedTypes: new Uint8Array(MAX_GENERATED_MOVES),
    capturedTypes: new Uint8Array(MAX_GENERATED_MOVES),
    scores: new Int32Array(MAX_GENERATED_MOVES),
    count: 0
  };
}

function resetMoveList(list) {
  list.count = 0;
  return list;
}

function pushBufferedMove(list, move, movedType, capturedType = 0, extraFlags = 0) {
  const index = list.count;
  if (index >= list.moves.length) throw new Error('Move buffer overflow.');
  const promotion = movePromotion(move);
  list.moves[index] = move;
  list.flags[index] = extraFlags
    | (capturedType ? MOVE_FLAG_CAPTURE : 0)
    | (promotion ? MOVE_FLAG_PROMOTION : 0);
  list.movedTypes[index] = movedType;
  list.capturedTypes[index] = capturedType;
  list.count = index + 1;
  return move;
}

function copyMoveEntry(list, dst, src) {
  list.moves[dst] = list.moves[src];
  list.flags[dst] = list.flags[src];
  list.movedTypes[dst] = list.movedTypes[src];
  list.capturedTypes[dst] = list.capturedTypes[src];
}

const PUBLIC_PSEUDO_MOVE_LIST = createMoveList();
const PUBLIC_LEGAL_MOVE_LIST = createMoveList();

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
function tablebaseBoundDistance(ply = 0) {
  return Math.min(TB_BOUND_MAX_PLY, Math.max(0, Math.floor(Number(ply) || 0)));
}
function isTablebaseBoundScore(score) {
  const value = Math.abs(Number(score));
  return Number.isFinite(value) && value >= TB_WIN_MIN_SCORE && value <= TB_WIN_SCORE;
}
function isOrdinaryEvaluationScore(score) {
  const value = Number(score);
  return Number.isFinite(value)
    && !isMateScore(value)
    && !isTablebaseBoundScore(value)
    && Math.abs(value) < TB_WIN_MIN_SCORE;
}

function clampOrdinaryEvaluation(score) {
  const value = Math.trunc(Number(score) || 0);
  return clamp(value, -TB_WIN_MIN_SCORE + 1, TB_WIN_MIN_SCORE - 1);
}

function scoreToTT(score, ply) { return score >= MATE_BOUND ? score + ply : score <= -MATE_BOUND ? score - ply : score; }
function scoreFromTT(score, ply) { return score >= MATE_BOUND ? score - ply : score <= -MATE_BOUND ? score + ply : score; }

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

// The board hash excludes move history, while threefold adjudication does not.
// Fold a stable two-word repetition-context fingerprint into TT locks so cached
// values from a visually identical board cannot leak across different histories.
function ttContextKeyA(pos, historySaltA = 0, repetition = 0) {
  const clock = (Math.max(0, Number(pos.halfmove || 0)) >>> 0) + 1;
  return (pos.hashA ^ Math.imul(clock, 0x9e3779b1) ^ (historySaltA >>> 0) ^ Math.imul(repetition + 1, 0x27d4eb2d)) >>> 0;
}

function ttContextKeyB(pos, historySaltB = 0, repetition = 0) {
  const clock = (Math.max(0, Number(pos.halfmove || 0)) >>> 0) + 1;
  return (pos.hashB ^ Math.imul(clock, 0x85ebca6b) ^ (historySaltB >>> 0) ^ Math.imul(repetition + 1, 0x165667b1)) >>> 0;
}

function repetitionContextFingerprint(repetitions) {
  let a = 0x811c9dc5;
  let b = 0x9e3779b9;
  for (const [identity, count] of [...(repetitions || new Map()).entries()].sort(([left], [right]) => left.localeCompare(right))) {
    const text = `${identity}:${count}|`;
    for (let index = 0; index < text.length; index += 1) {
      const code = text.charCodeAt(index);
      a = Math.imul(a ^ code, 0x01000193) >>> 0;
      b = xorshift((b ^ code ^ (a >>> 13)) >>> 0);
    }
  }
  return { a, b };
}

function zobristIndex(piece) {
  const type = typeOf(piece) - 1;
  return piece > 0 ? type : 6 + type;
}

function coordToSquare(coord) {
  // v12.2: engine UCI is standard A1–E5. Legacy b2–f6 UCI is intentionally
  // not accepted here; old archive coordinates are converted in the PGN layer.
  const file = coord.charCodeAt(0) - 97; // standard files a..e
  const rank = coord.charCodeAt(1) - 49; // standard ranks 1..5
  return inside(file, rank) ? square(file, rank) : -1;
}

function squareToCoord(sq) {
  return String.fromCharCode(97 + fileOf(sq)) + String.fromCharCode(49 + rankOf(sq));
}

export function moveToUci(move) {
  return `${squareToCoord(moveFrom(move))}${squareToCoord(moveTo(move))}${TYPE_TO_CHAR[movePromotion(move)] || ''}`;
}

export function uciToMove(position, uci) {
  const match = String(uci || '').trim().match(/^([a-e][1-5])([a-e][1-5])([qrbn])?$/i);
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
    // v10.2: keep king squares incrementally.  The 5×5 board is small, but
    // kingSquare()/isInCheck() is on the search hot path and was previously
    // scanning all 25 squares at every node.
    this.whiteKingSq = -1;
    this.blackKingSq = -1;
    this.hashA = 0;
    this.hashB = 0;
    this.recomputeHash();
  }

  static fromFEN(fen) {
    const parts = String(fen || '').trim().split(/\s+/);
    if (!parts[0]) throw new Error('Engine FEN is empty.');
    let rows = parts[0].split('/');
    if (rows.length === 8) {
      // Compatibility only. The engine's canonical FEN is compact A1–E5;
      // legacy b2–f6 padded FENs are converted on input for old saved studies.
      rows = paddedRowsToCompactRows(rows);
    }
    if (rows.length !== 5) throw new Error('Engine expects a compact 5×5 or compatible padded Gardner FEN.');
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
    copy.whiteKingSq = this.whiteKingSq;
    copy.blackKingSq = this.blackKingSq;
    copy.pieceCount = this.pieceCount;
    return copy;
  }

  recomputeHash() {
    let a = this.turn === BLACK ? TURN_A : 0;
    let b = this.turn === BLACK ? TURN_B : 0;
    this.whiteKingSq = -1;
    this.blackKingSq = -1;
    this.pieceCount = 0;
    for (let sq = 0; sq < BOARD_N; sq += 1) {
      const piece = this.board[sq];
      if (!piece) continue;
      this.pieceCount += 1;
      if (piece === KING) this.whiteKingSq = sq;
      else if (piece === -KING) this.blackKingSq = sq;
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

function paddedRowsToCompactRows(rows) {
  const expandedRows = rows.map(row => expandRow(row, 8));
  const legacy = extractPaddedRectangle(expandedRows, { top: 2, bottom: 6, left: 1, right: 5 });
  const standard = extractPaddedRectangle(expandedRows, { top: 3, bottom: 7, left: 0, right: 4 });
  if (legacy.valid) return legacy.rows;
  if (standard.valid) return standard.rows;
  throw new Error('Pieces outside supported Gardner 5×5 areas. Use compact A1–E5 FEN, or legacy b2–f6/standard A1–E5 padded FEN.');
}

function extractPaddedRectangle(expandedRows, { top, bottom, left, right }) {
  const inside = [];
  let outsidePieces = 0;
  expandedRows.forEach((row, rowIndex) => {
    const rowInside = rowIndex >= top && rowIndex <= bottom;
    if (rowInside) inside.push(row.slice(left, right + 1));
    row.forEach((symbol, file) => {
      if (!symbol) return;
      const fileInside = file >= left && file <= right;
      if (!rowInside || !fileInside) outsidePieces += 1;
    });
  });
  return {
    valid: outsidePieces === 0 && inside.length === 5,
    rows: inside.map(cells => compressRow(cells))
  };
}

const STATE_POOL = Array.from({ length: MAX_PLY + 512 }, (_, index) => ({ _poolIndex: index }));
let statePoolCursor = 0;
function allocateState() {
  if (statePoolCursor < STATE_POOL.length) return STATE_POOL[statePoolCursor++];
  return { _poolIndex: -1 };
}
function releaseState(state) {
  if (state && state._poolIndex >= 0 && state._poolIndex === statePoolCursor - 1) statePoolCursor -= 1;
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
  const state = allocateState();
  state.captured = captured;
  state.moving = moving;
  state.halfmove = pos.halfmove;
  state.fullmove = pos.fullmove;
  state.hashA = pos.hashA;
  state.hashB = pos.hashB;
  state.turn = pos.turn;
  state.whiteKingSq = pos.whiteKingSq;
  state.blackKingSq = pos.blackKingSq;
  state.pieceCount = pos.pieceCount;
  hashTogglePiece(pos, moving, from);
  if (captured) hashTogglePiece(pos, captured, to);
  const placed = promotion ? sideOf(moving) * promotion : moving;
  hashTogglePiece(pos, placed, to);
  pos.board[from] = EMPTY;
  pos.board[to] = placed;
  if (captured) pos.pieceCount -= 1;
  if (moving === KING) pos.whiteKingSq = to;
  else if (moving === -KING) pos.blackKingSq = to;
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
  pos.whiteKingSq = state.whiteKingSq;
  pos.blackKingSq = state.blackKingSq;
  pos.pieceCount = state.pieceCount;
  releaseState(state);
}

function makeNullMove(pos) {
  const state = allocateState();
  state.turn = pos.turn;
  state.halfmove = pos.halfmove;
  state.hashA = pos.hashA;
  state.hashB = pos.hashB;
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
  releaseState(state);
}

function restorePosition(pos, snapshot) {
  pos.board.set(snapshot.board);
  pos.turn = snapshot.turn;
  pos.halfmove = snapshot.halfmove;
  pos.fullmove = snapshot.fullmove;
  pos.hashA = snapshot.hashA;
  pos.hashB = snapshot.hashB;
  pos.whiteKingSq = snapshot.whiteKingSq;
  pos.blackKingSq = snapshot.blackKingSq;
  pos.pieceCount = snapshot.pieceCount;
}

function kingSquare(pos, side) {
  const cached = side === WHITE ? pos.whiteKingSq : pos.blackKingSq;
  if (cached >= 0 && cached < BOARD_N && pos.board[cached] === side * KING) return cached;
  // Defensive fallback for external tests that may construct a position-like
  // object without the v10.2 king-square fields.
  const king = side * KING;
  for (let sq = 0; sq < BOARD_N; sq += 1) if (pos.board[sq] === king) return sq;
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

function pushPawnMovesInto(pos, list, from, side, capturesOnly) {
  const file = fileOf(from), rank = rankOf(from), nextRank = rank + side;
  const promotionRank = side === WHITE ? 4 : 0;
  if (!capturesOnly && inside(file, nextRank)) {
    const to = square(file, nextRank);
    if (!pos.board[to]) {
      if (nextRank === promotionRank) for (const promo of PROMOTIONS) pushBufferedMove(list, encodeMove(from, to, promo), PAWN, 0);
      else pushBufferedMove(list, encodeMove(from, to), PAWN, 0);
    }
  }
  for (const to of PAWN_TARGETS[side][from]) {
    const target = pos.board[to];
    if (target && sideOf(target) === -side && typeOf(target) !== KING) {
      const captured = typeOf(target);
      if (rankOf(to) === promotionRank) for (const promo of PROMOTIONS) pushBufferedMove(list, encodeMove(from, to, promo), PAWN, captured);
      else pushBufferedMove(list, encodeMove(from, to), PAWN, captured);
    }
  }
}

function generatePseudoMovesInto(pos, capturesOnly = false, list = createMoveList()) {
  resetMoveList(list);
  const side = pos.turn;
  const board = pos.board;
  for (let from = 0; from < BOARD_N; from += 1) {
    const moving = board[from];
    if (sideOf(moving) !== side) continue;
    const type = typeOf(moving);
    if (type === PAWN) {
      pushPawnMovesInto(pos, list, from, side, capturesOnly);
      continue;
    }
    if (type === KNIGHT || type === KING) {
      const targets = type === KNIGHT ? KNIGHT_TARGETS[from] : KING_TARGETS[from];
      for (const to of targets) {
        const target = board[to];
        if (!target) {
          if (!capturesOnly) pushBufferedMove(list, encodeMove(from, to), type, 0);
        } else if (sideOf(target) === -side && typeOf(target) !== KING) pushBufferedMove(list, encodeMove(from, to), type, typeOf(target));
      }
      continue;
    }
    const start = type === ROOK ? 4 : 0;
    const end = type === BISHOP ? 4 : 8;
    for (let dir = start; dir < end; dir += 1) {
      for (const to of RAYS[from][dir]) {
        const target = board[to];
        if (!target) {
          if (!capturesOnly) pushBufferedMove(list, encodeMove(from, to), type, 0);
          continue;
        }
        if (sideOf(target) === -side && typeOf(target) !== KING) pushBufferedMove(list, encodeMove(from, to), type, typeOf(target));
        break;
      }
    }
  }
  return list.count;
}

export function generatePseudoMoves(pos, capturesOnly = false) {
  const list = PUBLIC_PSEUDO_MOVE_LIST;
  const count = generatePseudoMovesInto(pos, capturesOnly, list);
  return Array.from(list.moves.subarray(0, count));
}

function generateLegalMovesInto(pos, capturesOnly = false, list = createMoveList(), includeCheckMeta = false) {
  const side = pos.turn;
  generatePseudoMovesInto(pos, capturesOnly, list);
  let write = 0;
  const pseudoCount = list.count;
  for (let read = 0; read < pseudoCount; read += 1) {
    const move = list.moves[read];
    const flags = list.flags[read];
    const movedType = list.movedTypes[read];
    const captured = list.capturedTypes[read];
    const state = makeMove(pos, move);
    const valid = !isInCheck(pos, side);
    const checking = valid && includeCheckMeta && isInCheck(pos, pos.turn);
    undoMove(pos, move, state);
    if (!valid) continue;
    if (write !== read) {
      list.moves[write] = move;
      list.flags[write] = flags;
      list.movedTypes[write] = movedType;
      list.capturedTypes[write] = captured;
    }
    if (checking) list.flags[write] |= MOVE_FLAG_CHECK;
    else list.flags[write] &= ~MOVE_FLAG_CHECK;
    write += 1;
  }
  list.count = write;
  return write;
}

export function generateLegalMoves(pos, capturesOnly = false) {
  const list = PUBLIC_LEGAL_MOVE_LIST;
  const count = generateLegalMovesInto(pos, capturesOnly, list, false);
  return Array.from(list.moves.subarray(0, count));
}

function generateLegalTacticalMovesInto(pos, includeQuietChecks = false, list = createMoveList(), includeCheckMeta = false) {
  if (includeQuietChecks) {
    generateLegalMovesInto(pos, false, list, true);
    let write = 0;
    const count = list.count;
    for (let read = 0; read < count; read += 1) {
      const flags = list.flags[read];
      if (!(flags & (MOVE_FLAG_CAPTURE | MOVE_FLAG_PROMOTION | MOVE_FLAG_CHECK))) continue;
      if (write !== read) copyMoveEntry(list, write, read);
      write += 1;
    }
    list.count = write;
    return write;
  }

  generatePseudoMovesInto(pos, true, list);
  const side = pos.turn;
  // Captures-only generation omits non-capturing promotions, which are always tactical.
  for (let from = 0; from < BOARD_N; from += 1) {
    if (pos.board[from] !== side * PAWN) continue;
    const toRank = rankOf(from) + side;
    if (toRank !== (side === WHITE ? 4 : 0)) continue;
    const to = square(fileOf(from), toRank);
    if (!pos.board[to]) for (const promo of PROMOTIONS) pushBufferedMove(list, encodeMove(from, to, promo), PAWN, 0);
  }
  let write = 0;
  const candidateCount = list.count;
  for (let read = 0; read < candidateCount; read += 1) {
    const move = list.moves[read];
    const flags = list.flags[read];
    const movedType = list.movedTypes[read];
    const captured = list.capturedTypes[read];
    const state = makeMove(pos, move);
    const valid = !isInCheck(pos, side);
    const checking = valid && includeCheckMeta && isInCheck(pos, pos.turn);
    undoMove(pos, move, state);
    if (!valid) continue;
    if (write !== read) {
      list.moves[write] = move;
      list.flags[write] = flags;
      list.movedTypes[write] = movedType;
      list.capturedTypes[write] = captured;
    }
    if (checking) list.flags[write] |= MOVE_FLAG_CHECK;
    else list.flags[write] &= ~MOVE_FLAG_CHECK;
    write += 1;
  }
  list.count = write;
  return write;
}

function generateLegalTacticalMoves(pos, includeQuietChecks = false) {
  const list = PUBLIC_LEGAL_MOVE_LIST;
  const count = generateLegalTacticalMovesInto(pos, includeQuietChecks, list, false);
  return Array.from(list.moves.subarray(0, count));
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

const SEE_BOARD = new Int8Array(BOARD_N);
const SEE_GAINS = new Int32Array(32);

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
  const board = SEE_BOARD;
  board.set(pos.board);
  const gains = SEE_GAINS;
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
  let minorCount = 0;
  let knightCount = 0;
  let bishopCount = 0;
  let bishopColorMask = 0;
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const type = typeOf(pos.board[sq]);
    if (!type || type === KING) continue;
    if (type === PAWN || type === ROOK || type === QUEEN) return false;
    minorCount += 1;
    if (type === KNIGHT) knightCount += 1;
    else if (type === BISHOP) {
      bishopCount += 1;
      bishopColorMask |= 1 << ((fileOf(sq) + rankOf(sq)) & 1);
    }
  }
  if (minorCount === 0) return true;
  if (minorCount === 1) return bishopCount === 1 || knightCount === 1;
  return bishopCount === minorCount && (bishopColorMask === 1 || bishopColorMask === 2);
}

function materialProfile(pos) {
  const profile = {
    pieces: Number.isInteger(pos.pieceCount) ? pos.pieceCount : 0,
    pawns: 0,
    nonPawnPieces: 0,
    heavyPieces: 0,
    whiteNonKing: 0,
    blackNonKing: 0,
    whitePawns: 0,
    blackPawns: 0
  };
  let countedPieces = 0;
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq];
    if (!piece) continue;
    countedPieces += 1;
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
  if (!profile.pieces) profile.pieces = countedPieces;
  return profile;
}


function boardIdentity(pos) {
  // Collision-free for the tiny board and only used by the bounded fortress
  // verifier, not by the hot Alpha-Beta path.
  let key = pos.turn === WHITE ? 'w:' : 'b:';
  for (let square = 0; square < BOARD_N; square += 1) key += String.fromCharCode(pos.board[square] + 7);
  return key;
}

function isIrreversibleMove(pos, move) {
  return typeOf(pos.board[moveFrom(move)]) === PAWN || Boolean(pos.board[moveTo(move)]) || Boolean(movePromotion(move));
}

function endgameMopUp(pos, side, ownMaterial, enemyMaterial, ownKing, enemyKing) {
  if (ownKing < 0 || enemyKing < 0 || ownMaterial <= enemyMaterial) return 0;
  const advantage = ownMaterial - enemyMaterial;
  if (advantage < PIECE_VALUE[ROOK] - 40) return 0;
  const driveToEdge = (2 - edgeDistance(enemyKing)) * 18;
  const kingApproach = (4 - Math.min(4, kingDistance(ownKing, enemyKing))) * 9;
  return driveToEdge + kingApproach;
}



function pieceCentrality(sq) {
  return 4 - (Math.abs(fileOf(sq) - 2) + Math.abs(rankOf(sq) - 2));
}

function pawnIsPassed(pos, side, sq) {
  const file = fileOf(sq), rank = rankOf(sq);
  for (let enemySq = 0; enemySq < BOARD_N; enemySq += 1) {
    if (pos.board[enemySq] !== -side * PAWN) continue;
    if (Math.abs(fileOf(enemySq) - file) > 1) continue;
    if ((side === WHITE && rankOf(enemySq) > rank) || (side === BLACK && rankOf(enemySq) < rank)) return false;
  }
  return true;
}


function pawnWouldBePassedAfterMove(pos, side, to) {
  const file = fileOf(to), rank = rankOf(to);
  for (let enemySq = 0; enemySq < BOARD_N; enemySq += 1) {
    if (pos.board[enemySq] !== -side * PAWN) continue;
    if (Math.abs(fileOf(enemySq) - file) > 1) continue;
    if ((side === WHITE && rankOf(enemySq) > rank) || (side === BLACK && rankOf(enemySq) < rank)) return false;
  }
  return true;
}

function enemyPawnAttacksSquare(pos, side, target) {
  const enemy = -side;
  const pawnRank = rankOf(target) - enemy;
  for (const df of [-1, 1]) {
    const file = fileOf(target) + df;
    if (inside(file, pawnRank) && pos.board[square(file, pawnRank)] === enemy * PAWN) return true;
  }
  return false;
}

function directlyBlockedByEnemyPawn(pos, side, sq) {
  const ahead = rankOf(sq) + side;
  return inside(fileOf(sq), ahead) && pos.board[square(fileOf(sq), ahead)] === -side * PAWN;
}

function forceSideToMove(pos, side) {
  const cursor = pos.clone();
  if (cursor.turn !== side) {
    cursor.turn = side;
    cursor.recomputeHash();
  }
  return cursor;
}

function moveLandsSafely(pos, move) {
  const side = pos.turn;
  const to = moveTo(move);
  const movingType = typeOf(pos.board[moveFrom(move)]);
  const state = makeMove(pos, move);
  const enemy = pos.turn;
  const safe = movingType === KING
    ? !isSquareAttacked(pos, to, enemy)
    : (!isSquareAttacked(pos, to, enemy) || isSquareAttacked(pos, to, side));
  undoMove(pos, move, state);
  return safe;
}

function moveDropsMaterialBadly(pos, move) {
  const moving = pos.board[moveFrom(move)];
  const movingType = typeOf(moving);
  if (isPromotion(move)) return false;
  if (isCapture(pos, move)) return staticExchangeEval(pos, move) < -55;
  if (movingType === KING || movingType === PAWN) return false;
  const side = sideOf(moving), to = moveTo(move);
  const state = makeMove(pos, move);
  const enemyCanTake = isSquareAttacked(pos, to, -side);
  const ownDefends = isSquareAttacked(pos, to, side);
  undoMove(pos, move, state);
  return enemyCanTake && !ownDefends && PIECE_VALUE[movingType] >= PIECE_VALUE[BISHOP];
}

function isKingEntryMove(pos, move, side = pos.turn) {
  const moving = pos.board[moveFrom(move)];
  if (sideOf(moving) !== side || typeOf(moving) !== KING) return false;
  const from = moveFrom(move), to = moveTo(move);
  const fromRank = rankOf(from), toRank = rankOf(to);
  const enemyCampBefore = side === WHITE ? fromRank >= 3 : fromRank <= 1;
  const enemyCampAfter = side === WHITE ? toRank >= 3 : toRank <= 1;
  if (!enemyCampAfter || enemyCampBefore) return false;
  return moveLandsSafely(pos, move);
}

function isRealPawnBreakMove(pos, move) {
  const moving = pos.board[moveFrom(move)];
  if (typeOf(moving) !== PAWN) return false;
  const side = sideOf(moving);
  const from = moveFrom(move), to = moveTo(move);
  const advance = side === WHITE ? rankOf(to) : 4 - rankOf(to);
  if (isPromotion(move) || advance >= 4) return true;
  if (isCapture(pos, move)) {
    if (staticExchangeEval(pos, move) >= -35) return true;
    const state = makeMove(pos, move);
    const passed = pawnIsPassed(pos, side, to);
    undoMove(pos, move, state);
    return passed && advance >= 3;
  }

  // In already locked endings, a push into an enemy-pawn-controlled square is
  // usually just a reversible concession: the pawn can be exchanged or fixed
  // without opening a new route. Do not count it as a breakthrough unless it is
  // also a serious passer/promotion event.
  const contestedByEnemyPawn = enemyPawnAttacksSquare(pos, side, to);
  const state = makeMove(pos, move);
  const passed = pawnIsPassed(pos, side, to);
  const blockedAhead = directlyBlockedByEnemyPawn(pos, side, to);
  const safe = !isSquareAttacked(pos, to, -side) || isSquareAttacked(pos, to, side);
  undoMove(pos, move, state);
  if (contestedByEnemyPawn && !(passed && advance >= 3 && safe)) return false;
  if (passed && (advance >= 2 || !blockedAhead) && safe) return true;
  if (advance >= 3 && !blockedAhead && safe) return true;
  return false;
}

function sideImmediateProgressStats(pos, side) {
  const cursor = forceSideToMove(pos, side);
  const moves = generateLegalMoves(cursor, false);
  const stats = { realPawnBreaks: 0, soundCaptures: 0, promotions: 0, kingEntries: 0 };
  for (const move of moves) {
    if (isPromotion(move)) stats.promotions += 1;
    if (isCapture(cursor, move) && staticExchangeEval(cursor, move) >= -35) stats.soundCaptures += 1;
    if (isRealPawnBreakMove(cursor, move)) stats.realPawnBreaks += 1;
    if (isKingEntryMove(cursor, move, side) && !moveDropsMaterialBadly(cursor, move)) stats.kingEntries += 1;
  }
  return stats;
}

function sideEffectiveLowProgress(pos, side) {
  const cursor = forceSideToMove(pos, side);
  const moves = generateLegalMoves(cursor, false);
  const before = sideImmediateProgressStats(cursor, side);
  const metrics = {
    legal: moves.length,
    nonLosing: 0,
    safeQuiet: 0,
    realPawnBreaks: 0,
    soundCaptures: 0,
    promotionMoves: 0,
    kingEntries: 0,
    enablingMoves: 0,
    progressMoves: 0,
    reversibleWaiting: 0
  };

  for (const move of moves) {
    const movingType = typeOf(cursor.board[moveFrom(move)]);
    const capture = isCapture(cursor, move);
    const promotion = isPromotion(move);
    const badlyLosing = moveDropsMaterialBadly(cursor, move);
    if (!badlyLosing) metrics.nonLosing += 1;

    let progress = false;
    if (promotion) {
      metrics.promotionMoves += 1;
      progress = true;
    }
    if (capture && staticExchangeEval(cursor, move) >= -35) {
      metrics.soundCaptures += 1;
      progress = true;
    }
    if (isRealPawnBreakMove(cursor, move)) {
      metrics.realPawnBreaks += 1;
      progress = true;
    }
    if (isKingEntryMove(cursor, move, side) && !badlyLosing) {
      metrics.kingEntries += 1;
      progress = true;
    }

    if (!progress && !capture && !promotion && !badlyLosing) {
      const pawnPushIntoPawnControl = movingType === PAWN && enemyPawnAttacksSquare(cursor, side, moveTo(move));
      const state = makeMove(cursor, move);
      const after = sideImmediateProgressStats(cursor, side);
      undoMove(cursor, move, state);
      const enablesForcingResource = !pawnPushIntoPawnControl && (
        after.realPawnBreaks > before.realPawnBreaks ||
        after.soundCaptures > before.soundCaptures ||
        after.promotions > before.promotions ||
        after.kingEntries > before.kingEntries
      );
      if (enablesForcingResource) {
        metrics.enablingMoves += 1;
        progress = true;
      }
    }

    if (progress && !badlyLosing) metrics.progressMoves += 1;
    else if (!capture && !promotion && !badlyLosing && moveLandsSafely(cursor, move)) {
      metrics.safeQuiet += 1;
      if (movingType !== PAWN) metrics.reversibleWaiting += 1;
    }
  }
  return metrics;
}

function lowProgressLegalProfile(pos) {
  const white = sideEffectiveLowProgress(pos, WHITE);
  const black = sideEffectiveLowProgress(pos, BLACK);
  return {
    white,
    black,
    realPawnBreaks: white.realPawnBreaks + black.realPawnBreaks,
    soundCaptures: white.soundCaptures + black.soundCaptures,
    promotionMoves: white.promotionMoves + black.promotionMoves,
    kingEntries: white.kingEntries + black.kingEntries,
    enablingMoves: white.enablingMoves + black.enablingMoves,
    progressMoves: white.progressMoves + black.progressMoves,
    nonLosing: white.nonLosing + black.nonLosing,
    safeQuiet: white.safeQuiet + black.safeQuiet,
    reversibleWaiting: white.reversibleWaiting + black.reversibleWaiting
  };
}

function sideStructuralActivity(pos, side, ownInfo, enemyInfo) {
  const metrics = {
    pieces: 0,
    pawns: 0,
    lockedPawns: 0,
    pawnPushes: 0,
    pawnCaptures: 0,
    promotionThreats: 0,
    passedPawns: 0,
    advancedPassedPawns: 0,
    pseudoMoves: 0,
    safeMoves: 0,
    improvingMoves: 0,
    soundCaptures: 0,
    contactAttacks: 0,
    kingPressure: 0,
    heavyPieces: 0
  };
  const board = pos.board;
  const ownAttacks = ownInfo.counts;
  const enemyAttacks = enemyInfo.counts;
  const enemyKing = kingSquare(pos, -side);
  const enemyKingZone = enemyKing >= 0 ? KING_TARGETS[enemyKing] : [];

  for (let from = 0; from < BOARD_N; from += 1) {
    const piece = board[from];
    if (sideOf(piece) !== side) continue;
    const type = typeOf(piece);
    metrics.pieces += 1;
    if (type === ROOK || type === QUEEN) metrics.heavyPieces += 1;

    if (type === PAWN) {
      metrics.pawns += 1;
      const aheadRank = rankOf(from) + side;
      let hasCapture = false;
      if (inside(fileOf(from), aheadRank)) {
        const ahead = square(fileOf(from), aheadRank);
        if (!board[ahead]) {
          metrics.pawnPushes += 1;
          metrics.pseudoMoves += 1;
          metrics.safeMoves += enemyAttacks[ahead] <= ownAttacks[ahead] + 1 ? 1 : 0;
          const advance = side === WHITE ? rankOf(ahead) : 4 - rankOf(ahead);
          if (advance >= 4) metrics.promotionThreats += 1;
          if (pieceCentrality(ahead) > pieceCentrality(from)) metrics.improvingMoves += 1;
        }
      }
      for (const to of PAWN_TARGETS[side][from]) {
        if (sideOf(board[to]) !== -side) continue;
        hasCapture = true;
        metrics.pawnCaptures += 1;
        metrics.pseudoMoves += 1;
        metrics.contactAttacks += 1;
        const victim = PIECE_VALUE[typeOf(board[to])];
        if (victim + 35 >= PIECE_VALUE[PAWN] || ownAttacks[to] >= enemyAttacks[to]) metrics.soundCaptures += 1;
        metrics.safeMoves += ownAttacks[to] >= enemyAttacks[to] ? 1 : 0;
      }
      const ahead = inside(fileOf(from), aheadRank) ? board[square(fileOf(from), aheadRank)] : EMPTY;
      if (ahead === -side * PAWN && !hasCapture) metrics.lockedPawns += 1;
      if (pawnIsPassed(pos, side, from)) {
        metrics.passedPawns += 1;
        const advance = side === WHITE ? rankOf(from) : 4 - rankOf(from);
        if (advance >= 3) metrics.advancedPassedPawns += 1;
      }
      continue;
    }

    let targets = null;
    if (type === KNIGHT) targets = KNIGHT_TARGETS[from];
    else if (type === KING) targets = KING_TARGETS[from];
    if (targets) {
      for (const to of targets) {
        if (sideOf(board[to]) === side) continue;
        metrics.pseudoMoves += 1;
        const enemyPiece = sideOf(board[to]) === -side;
        if (enemyPiece) {
          metrics.contactAttacks += 1;
          const victim = PIECE_VALUE[typeOf(board[to])];
          if (victim + 35 >= PIECE_VALUE[type] || ownAttacks[to] > enemyAttacks[to]) metrics.soundCaptures += 1;
        }
        const safe = type === KING ? enemyAttacks[to] === 0 : enemyAttacks[to] === 0 || ownAttacks[to] >= enemyAttacks[to];
        if (safe) {
          metrics.safeMoves += 1;
          const centralGain = pieceCentrality(to) - pieceCentrality(from);
          const kingGain = enemyKing >= 0 ? kingDistance(from, enemyKing) - kingDistance(to, enemyKing) : 0;
          if (centralGain > 0 || kingGain > 0) metrics.improvingMoves += 1;
        }
        if (enemyKingZone.includes(to)) metrics.kingPressure += 1;
      }
      continue;
    }

    const start = type === ROOK ? 4 : 0;
    const end = type === BISHOP ? 4 : 8;
    for (let dir = start; dir < end; dir += 1) {
      for (const to of RAYS[from][dir]) {
        if (sideOf(board[to]) === side) break;
        metrics.pseudoMoves += 1;
        const enemyPiece = sideOf(board[to]) === -side;
        if (enemyPiece) {
          metrics.contactAttacks += 1;
          const victim = PIECE_VALUE[typeOf(board[to])];
          if (victim + 35 >= PIECE_VALUE[type] || ownAttacks[to] > enemyAttacks[to]) metrics.soundCaptures += 1;
        }
        const safe = enemyAttacks[to] === 0 || ownAttacks[to] >= enemyAttacks[to];
        if (safe) {
          metrics.safeMoves += 1;
          const centralGain = pieceCentrality(to) - pieceCentrality(from);
          const kingGain = enemyKing >= 0 ? kingDistance(from, enemyKing) - kingDistance(to, enemyKing) : 0;
          if (centralGain > 0 || kingGain > 0) metrics.improvingMoves += 1;
        }
        if (enemyKingZone.includes(to)) metrics.kingPressure += 1;
        if (board[to]) break;
      }
    }
  }
  metrics.pawnBreaks = metrics.pawnPushes + metrics.pawnCaptures;
  metrics.progressMoves = metrics.pawnBreaks + metrics.soundCaptures + Math.min(4, metrics.improvingMoves);
  metrics.reasonableMoves = metrics.safeMoves + metrics.soundCaptures + metrics.pawnPushes;
  return metrics;
}

const STRUCTURAL_PROFILE_CACHE = new Map();
function structuralProfileCacheKey(pos) {
  return `${pos.hashA}:${pos.hashB}:t${pos.turn}`;
}
function rememberStructuralProfile(key, profile) {
  STRUCTURAL_PROFILE_CACHE.set(key, profile);
  if (STRUCTURAL_PROFILE_CACHE.size > 24576) STRUCTURAL_PROFILE_CACHE.delete(STRUCTURAL_PROFILE_CACHE.keys().next().value);
  return profile;
}


function quietHeavyOfferBreakthroughThreat(pos, move) {
  if (isCapture(pos, move) || isPromotion(move)) return false;
  const movingType = movePieceType(pos, move);
  if (movingType < ROOK) return false;
  const side = pos.turn;
  const to = moveTo(move);
  const state = makeMove(pos, move);
  const attackedByEnemy = isSquareAttacked(pos, to, pos.turn);
  const defendedByMover = isSquareAttacked(pos, to, side);
  if (!attackedByEnemy || !defendedByMover) {
    undoMove(pos, move, state);
    return false;
  }
  // For structural draw classification, require more than a visually defended
  // heavy-piece shuffle.  The opponent must be able to accept the offer, and
  // the mover must then have a sound recapture on that same square.  This keeps
  // v13-style ...Qb5 breakthroughs tactical without treating harmless rook/queen
  // repositioning in a locked fortress as real progress.
  let concrete = false;
  const replies = generateLegalMoves(pos, false);
  for (const reply of replies) {
    if (!isCapture(pos, reply) || moveTo(reply) !== to) continue;
    const replyState = makeMove(pos, reply);
    const recaptures = generateLegalMoves(pos, false);
    for (const recapture of recaptures) {
      if (!isCapture(pos, recapture) || moveTo(recapture) !== to) continue;
      if (staticExchangeEval(pos, recapture) >= -45) {
        concrete = true;
        break;
      }
    }
    undoMove(pos, reply, replyState);
    if (concrete) break;
  }
  undoMove(pos, move, state);
  return concrete;
}

function sideQuietProgressThreatCount(pos, side, limit = 2) {
  const cursor = forceSideToMove(pos, side);
  const moves = generateLegalMoves(cursor, false);
  let count = 0;
  for (const move of moves) {
    if (!quietHeavyOfferBreakthroughThreat(cursor, move)) continue;
    count += 1;
    if (count >= limit) return count;
  }
  return count;
}

/**
 * A generic low-progress/closed-position model. It deliberately does not
 * recognise one exact pawn pattern. Instead it combines available pawn breaks,
 * safe mobility, improving moves, contact captures and king pressure. The
 * result is an evaluation scale, not a mathematical draw claim.
 */
function structuralDrawProfile(pos, whiteInfo = null, blackInfo = null) {
  const cacheKey = structuralProfileCacheKey(pos);
  const cached = STRUCTURAL_PROFILE_CACHE.get(cacheKey);
  if (cached) return cached;
  const wInfo = whiteInfo || buildAttackInfo(pos, WHITE);
  const bInfo = blackInfo || buildAttackInfo(pos, BLACK);
  const white = sideStructuralActivity(pos, WHITE, wInfo, bInfo);
  const black = sideStructuralActivity(pos, BLACK, bInfo, wInfo);
  const pawns = white.pawns + black.pawns;
  const locked = white.lockedPawns + black.lockedPawns;
  const rawPawnBreaks = white.pawnBreaks + black.pawnBreaks;
  const rawSoundCaptures = white.soundCaptures + black.soundCaptures;
  const rawProgress = white.progressMoves + black.progressMoves;
  const safeMobility = white.safeMoves + black.safeMoves;
  const rawImproving = white.improvingMoves + black.improvingMoves;
  const rawPressure = white.kingPressure + black.kingPressure;
  const rawContacts = white.contactAttacks + black.contactAttacks;
  const heavy = white.heavyPieces + black.heavyPieces;
  const advancedPassed = white.advancedPassedPawns + black.advancedPassedPawns;
  const profile = materialProfile(pos);
  let queens = 0;
  for (const piece of pos.board) if (typeOf(piece) === QUEEN) queens += 1;

  const inCheck = isInCheck(pos, WHITE) || isInCheck(pos, BLACK);
  const lockRatio = pawns ? locked / pawns : 0;

  // The old v9 model was intentionally cheap, but in compact locked endings it
  // over-valued reversible rook/bishop shuffling and pawn pushes into opposing
  // pawn control. Only run the more expensive legal-move classifier after a
  // fast structural gate says the position is a plausible low-progress ending.
  const queenfulLockedCandidate = queens > 0
    // v14: queen/rook material does not by itself create progress on a 5×5
    // board. If the pawn wall is almost fully locked, neither side has a
    // capture/check/pawn break now, and mobility is tiny, run the exact legal
    // progress classifier instead of preserving a large static material edge.
    // This stays generic: the gate uses only mobility, pawn locks and available
    // irreversible resources, not a fixed square pattern.
    && pawns >= 6
    && lockRatio >= 0.72
    && rawPawnBreaks === 0
    && rawSoundCaptures === 0
    && rawPressure === 0
    && rawContacts <= 2
    && safeMobility <= 12
    && heavy <= 4
    && profile.pieces <= 18;
  const legalCandidate = !inCheck
    && pawns >= 4
    && (queens === 0 || queenfulLockedCandidate)
    && profile.pieces <= (queenfulLockedCandidate ? 18 : 14)
    && advancedPassed === 0
    && rawSoundCaptures <= 1
    && rawPressure <= (queenfulLockedCandidate ? 0 : 4)
    && (lockRatio >= 0.45 || rawPawnBreaks <= 2 && safeMobility <= 16)
    && heavy <= (queenfulLockedCandidate ? 4 : 2);
  const legal = legalCandidate ? lowProgressLegalProfile(pos) : null;
  // v14.1: distinguish a real closed deadlock from a visually similar
  // closed tactic.  Pseudo-contact can be a self-losing capture that should
  // not block draw compression, but a quiet heavy-piece offer that creates a
  // defended forcing resource (the v13 ...Qb5 class) must keep the position
  // tactical.  This is still pattern-free: the side to move's legal heavy-piece offers are checked by a concrete
  // accept-and-recapture verifier, while the broader search extender remains free to investigate
  // looser quiet progress threats.
  const quietProgressThreats = legal ? sideQuietProgressThreatCount(pos, pos.turn) : 0;

  const pawnBreaks = legal ? legal.realPawnBreaks : rawPawnBreaks;
  const soundCaptures = legal ? legal.soundCaptures : rawSoundCaptures;
  const progress = legal ? legal.progressMoves + legal.enablingMoves : rawProgress;
  const improving = legal ? Math.min(rawImproving, legal.progressMoves + legal.enablingMoves) : rawImproving;
  const pressure = rawPressure;

  const noBreaks = 1 - Math.min(1, pawnBreaks / Math.max(1, pawns * 0.35));
  const lowProgress = 1 - Math.min(1, progress / (legal ? 5 : 10));
  const lowMobility = legal
    ? 1 - Math.min(1, (legal.progressMoves * 2 + legal.safeQuiet * 0.25) / 8)
    : 1 - Math.min(1, safeMobility / 18);
  const lowImprovement = 1 - Math.min(1, improving / (legal ? 4 : 8));
  const lowTactics = 1 - Math.min(1, (soundCaptures * 2 + pressure) / 8);
  const closure = clamp(
    lockRatio * 0.31 + noBreaks * 0.24 + lowProgress * 0.19 + lowMobility * 0.14 + lowImprovement * 0.08 + lowTactics * 0.04,
    0,
    1
  );

  let scale = 1;
  let lowProgressDraw = false;
  const tacticallyQuiet = soundCaptures <= 1 && pressure <= 4 && advancedPassed === 0 && quietProgressThreats === 0;
  const bothConstrained = legal
    ? Math.max(white.reasonableMoves, black.reasonableMoves) <= 14
      && Math.min(white.reasonableMoves, black.reasonableMoves) <= 7
      && Math.max(legal.white.progressMoves, legal.black.progressMoves) <= 1
    : Math.max(white.reasonableMoves, black.reasonableMoves) <= 10
      && Math.min(white.reasonableMoves, black.reasonableMoves) <= 6;

  if (!inCheck && pawns >= 2 && tacticallyQuiet && closure > 0.47) {
    let maxReduction = heavy >= 2 ? 0.48 : heavy === 1 ? 0.66 : 0.90;
    if (pawnBreaks === 0 && bothConstrained) maxReduction = Math.min(0.96, maxReduction + 0.14);
    if (white.promotionThreats || black.promotionThreats || legal?.promotionMoves) maxReduction *= 0.35;
    const normalized = clamp((closure - 0.47) / 0.53, 0, 1);
    scale = clamp(1 - maxReduction * Math.pow(normalized, 1.25), 0.035, 1);
    if (pawnBreaks === 0 && soundCaptures === 0 && bothConstrained && improving <= 2) scale = Math.min(scale, 0.12);
  }

  if (legal) {
    const materialGap = Math.abs(profile.whiteNonKing - profile.blackNonKing);
    const bothHaveWaiting = legal.white.reversibleWaiting >= 1 && legal.black.reversibleWaiting >= 1;
    const noEffectiveBreakthrough = legal.realPawnBreaks === 0
      && legal.soundCaptures === 0
      && legal.promotionMoves === 0
      && legal.kingEntries === 0
      && legal.progressMoves === 0
      && legal.enablingMoves === 0
      && quietProgressThreats === 0;
    const lockedEnough = lockRatio >= 0.50 || locked >= 3 && rawPawnBreaks <= 2;

    // This is the v10 hard draw-compression gate. It is not a pattern matcher:
    // it asks whether either side has a non-losing move that changes the pawn
    // structure, creates a sound capture, enters with the king, or enables such
    // a resource. If all legal play is reversible waiting or self-contesting
    // pawn pushes, a static edge must be treated as no edge at all.
    const queenfulDeadlock = queens > 0
      && lockRatio >= 0.72
      && pawnBreaks === 0
      && rawSoundCaptures === 0
      && rawContacts <= 2
      && pressure === 0
      && legal.white.legal <= 8
      && legal.black.legal <= 8;
    if (lockedEnough && noEffectiveBreakthrough && bothHaveWaiting && materialGap <= PIECE_VALUE[ROOK] + 120) {
      scale = 0;
      lowProgressDraw = true;
    } else if (queenfulDeadlock && noEffectiveBreakthrough && materialGap <= PIECE_VALUE[ROOK] + 120) {
      // A high-material deadlock with queens/rooks can be just as drawn as a
      // minor-piece fortress when every legal non-losing move is reversible
      // waiting and no side can create an irreversible resource. Compress to a
      // practical draw, but only after the legal classifier confirms there are
      // no captures, pawn breaks, king entries, promotions or enabling moves.
      scale = Math.min(scale, 0.02);
      lowProgressDraw = true;
    } else if (lockedEnough
      && legal.realPawnBreaks === 0
      && legal.soundCaptures === 0
      && legal.promotionMoves === 0
      && legal.kingEntries === 0
      && legal.progressMoves <= 1
      && quietProgressThreats === 0
      && materialGap <= PIECE_VALUE[ROOK] + 120) {
      scale = Math.min(scale, 0.05);
    }
  }

  return rememberStructuralProfile(cacheKey, {
    white,
    black,
    closure,
    scale,
    pawnBreaks,
    rawPawnBreaks,
    soundCaptures,
    rawSoundCaptures,
    safeMobility,
    improving,
    rawImproving,
    rawContacts,
    quietProgressThreats,
    pressure,
    advancedPassed,
    bothConstrained,
    legal,
    lowProgressDraw
  });
}

export function analyzePositionActivity(pos) {
  const whiteInfo = buildAttackInfo(pos, WHITE);
  const blackInfo = buildAttackInfo(pos, BLACK);
  const profile = structuralDrawProfile(pos, whiteInfo, blackInfo);
  const exact = {};
  for (const side of [WHITE, BLACK]) {
    const cursor = pos.clone();
    if (cursor.turn !== side) {
      cursor.turn = side;
      cursor.recomputeHash();
    }
    const moves = generateLegalMoves(cursor, false);
    let captures = 0, checks = 0, irreversible = 0, sound = 0;
    for (const move of moves) {
      const capture = isCapture(cursor, move);
      const promotion = isPromotion(move);
      if (capture) captures += 1;
      if (givesCheck(cursor, move)) checks += 1;
      if (capture || promotion || movePieceType(cursor, move) === PAWN) irreversible += 1;
      if (promotion || givesCheck(cursor, move) || capture && staticExchangeEval(cursor, move) >= -45) sound += 1;
      else if (!capture) {
        const state = makeMove(cursor, move);
        const to = moveTo(move);
        const movedSide = -cursor.turn;
        const safe = !isSquareAttacked(cursor, to, cursor.turn) || isSquareAttacked(cursor, to, movedSide);
        undoMove(cursor, move, state);
        if (safe) sound += 1;
      }
    }
    exact[side === WHITE ? 'white' : 'black'] = { legal: moves.length, captures, checks, irreversible, sound };
  }
  return { ...profile, exact };
}

function lowProgressSearchNode(pos, legalMoves = null) {
  // v13: queenful positions can still be search-sensitive closed systems on a
  // 5×5 board. Earlier versions disabled the low-progress protections whenever
  // a queen was present; that let PVS/LMR accept reversible repetition lines
  // before fully verifying quiet breakthrough resources.
  let pawns = 0, pawnBreaks = 0, contacts = 0, heavy = 0, queens = 0, nonKing = 0;
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq];
    if (!piece) continue;
    const side = sideOf(piece), type = typeOf(piece);
    if (type !== KING) nonKing += 1;
    if (type === QUEEN) { queens += 1; heavy += 2; }
    else if (type === ROOK) heavy += 1;
    if (type === PAWN) {
      pawns += 1;
      const aheadRank = rankOf(sq) + side;
      if (inside(fileOf(sq), aheadRank) && !pos.board[square(fileOf(sq), aheadRank)]) pawnBreaks += 1;
      for (const to of PAWN_TARGETS[side][sq]) if (sideOf(pos.board[to]) === -side) pawnBreaks += 1;
    } else if (type !== KING) {
      if (type === KNIGHT) {
        for (const to of KNIGHT_TARGETS[sq]) if (sideOf(pos.board[to]) === -side) contacts += 1;
      } else {
        const start = type === ROOK ? 4 : 0;
        const end = type === BISHOP ? 4 : 8;
        for (let dir = start; dir < end; dir += 1) {
          for (const to of RAYS[sq][dir]) {
            const target = pos.board[to];
            if (!target) continue;
            if (sideOf(target) === -side) contacts += 1;
            break;
          }
        }
      }
    }
  }
  const legalCount = Array.isArray(legalMoves) ? legalMoves.length : (legalMoves && typeof legalMoves.count === 'number' ? legalMoves.count : -1);
  const narrow = legalCount >= 0 && legalCount <= 8;
  const closedPawnMass = pawns >= 6 && pawnBreaks <= 2 && contacts <= 2;
  if (queens) return closedPawnMass && (narrow || nonKing <= 12);
  return pawns >= 4 && pawnBreaks <= 1 && contacts <= 1 && heavy <= 1;
}

function progressStatsValue(stats) {
  return stats.realPawnBreaks * 5 + stats.promotions * 5 + stats.soundCaptures * 3 + stats.kingEntries * 2;
}

function quietMoveCreatesProgressThreat(pos, move) {
  if (isCapture(pos, move) || isPromotion(move)) return false;
  const movingType = movePieceType(pos, move);
  const side = pos.turn;
  if (movingType === PAWN) return isRealPawnBreakMove(pos, move);
  const before = sideImmediateProgressStats(pos, side);
  const to = moveTo(move);
  const state = makeMove(pos, move);
  const after = sideImmediateProgressStats(pos, side);
  const attackedByEnemy = isSquareAttacked(pos, to, pos.turn);
  const defendedByMover = isSquareAttacked(pos, to, side);
  const check = isInCheck(pos, pos.turn);
  undoMove(pos, move, state);
  if (check) return true;
  if (progressStatsValue(after) > progressStatsValue(before)) return true;
  // A defended heavy-piece offer is often the only way to crack a closed
  // low-mobility structure. This is intentionally generic: it does not check a
  // fixed square/pattern, only whether a queen/rook quiet move can be captured
  // while being tactically supported by the mover.
  return movingType >= ROOK && attackedByEnemy && defendedByMover;
}

function rootPriorityCompare(pos, left, right, repetitionPriority, tablebasePriority, searcher) {
  const leftRepeat = repetitionPriority.get(left) || 0;
  const rightRepeat = repetitionPriority.get(right) || 0;
  if (leftRepeat !== rightRepeat) return rightRepeat - leftRepeat;
  const leftTb = tablebasePriority.get(left) || 0;
  const rightTb = tablebasePriority.get(right) || 0;
  if (leftTb !== rightTb) return rightTb - leftTb;
  const leftBook = searcher.rootBookMoves.has(moveToUci(left)) ? 1 : 0;
  const rightBook = searcher.rootBookMoves.has(moveToUci(right)) ? 1 : 0;
  if (leftBook !== rightBook) return rightBook - leftBook;
  const leftClosed = quietMoveCreatesProgressThreat(pos, left) ? 1 : 0;
  const rightClosed = quietMoveCreatesProgressThreat(pos, right) ? 1 : 0;
  if (leftClosed !== rightClosed) return rightClosed - leftClosed;
  return (searcher.previousRootScores.get(right) ?? -INF) - (searcher.previousRootScores.get(left) ?? -INF);
}

function shouldUseFullWidthRoot(pos, moves) {
  // Full-window root verification is reserved for structurally closed nodes.
  // Low branching alone is not enough: simple K+P races also have few moves and
  // need the normal PVS efficiency to reach long mate/distancing horizons.
  return lowProgressSearchNode(pos, moves);
}

function closedWrongBishopFortressScale(pos) {
  if (isInCheck(pos, WHITE) || isInCheck(pos, BLACK)) return 1;
  const pawns = { [WHITE]: [], [BLACK]: [] };
  const extras = { [WHITE]: [], [BLACK]: [] };
  let whiteMaterial = 0, blackMaterial = 0;
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq];
    if (!piece) continue;
    const side = sideOf(piece), type = typeOf(piece);
    if (type === KING) continue;
    if (type === PAWN) pawns[side].push(sq);
    else extras[side].push({ type, sq });
    if (side === WHITE) whiteMaterial += PIECE_VALUE[type];
    else blackMaterial += PIECE_VALUE[type];
  }
  if (pawns[WHITE].length < 2 || pawns[WHITE].length !== pawns[BLACK].length) return 1;
  const stronger = whiteMaterial > blackMaterial ? WHITE : BLACK;
  const weaker = -stronger;
  if (Math.abs(whiteMaterial - blackMaterial) < PIECE_VALUE[BISHOP] - 25) return 1;
  if (extras[stronger].length !== 1 || extras[stronger][0].type !== BISHOP || extras[weaker].length !== 0) return 1;

  // Every pawn must be directly locked by its opposite number and no pawn may
  // currently capture. This is the common Gardner two-wing deadlock.
  for (const side of [WHITE, BLACK]) {
    for (const sq of pawns[side]) {
      const aheadRank = rankOf(sq) + side;
      if (!inside(fileOf(sq), aheadRank) || pos.board[square(fileOf(sq), aheadRank)] !== -side * PAWN) return 1;
      for (const target of PAWN_TARGETS[side][sq]) if (sideOf(pos.board[target]) === -side) return 1;
    }
  }

  const bishop = extras[stronger][0];
  const bishopColor = (fileOf(bishop.sq) + rankOf(bishop.sq)) & 1;
  // The extra bishop cannot ever take the locked defender pawns.
  if (pawns[weaker].some(sq => ((fileOf(sq) + rankOf(sq)) & 1) === bishopColor)) return 1;
  // Keep the deadlock scaling while the stronger king approaches. On a 5×5
  // board that approach can otherwise create a large horizon bonus many plies
  // before a pawn can actually be won. The scale disappears immediately after
  // a pawn move/capture breaks the locked wall, so a genuinely forced
  // breakthrough is still visible to deeper search.
  return 0.025;
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
  const fortressScale = closedWrongBishopFortressScale(pos);
  const activityProfile = structuralDrawProfile(pos, whiteInfo, blackInfo);
  const activityScale = activityProfile.scale;
  const drawScale = Math.min(fortressScale, activityScale);
  if (drawScale <= 0) return 0;
  const positional = Math.round((white - black) * drawScale);
  const tempoBonus = drawScale <= 0.02 ? 0 : Math.max(1, Math.round(tempo * drawScale));
  const score = positional + (pos.turn === WHITE ? tempoBonus : -tempoBonus);
  return clampOrdinaryEvaluation(pos.turn === WHITE ? score : -score);
}

function isPruningEndgame(pos) {
  const profile = materialProfile(pos);
  return profile.pieces <= 7 || profile.heavyPieces === 0 && profile.nonPawnPieces <= 2;
}

function hasNonPawnMaterial(pos, side) {
  for (let sq = 0; sq < BOARD_N; sq += 1) {
    const piece = pos.board[sq];
    if (sideOf(piece) !== side) continue;
    const type = typeOf(piece);
    if (type !== PAWN && type !== KING) return true;
  }
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

function moveOrderScore(pos, move, ttMove, ply, searcher, previousMove, endgameChecks = false, flags = 0, movedTypeHint = 0, capturedTypeHint = 0) {
  if (move === ttMove) return 2_000_000;
  const promo = movePromotion(move);
  const captured = capturedTypeHint || typeOf(pos.board[moveTo(move)]);
  const movedType = movedTypeHint || typeOf(pos.board[moveFrom(move)]);
  if (promo) return 1_300_000 + PIECE_VALUE[promo] * 10 + PIECE_VALUE[captured];
  if (captured) {
    const sideIndex = pos.turn === WHITE ? 0 : 1;
    const historyIndex = movedType * 25 + moveTo(move);
    return 1_000_000 + PIECE_VALUE[captured] * 16 - PIECE_VALUE[movedType] + staticExchangeEval(pos, move) + searcher.captureHistory[sideIndex][historyIndex];
  }
  if (searcher.killers[ply][0] === move) return 900_000;
  if (searcher.killers[ply][1] === move) return 850_000;
  if (previousMove && searcher.countermoves[previousMove & 8191] === move) return 820_000;
  if (endgameChecks && (flags & MOVE_FLAG_CHECK || givesCheck(pos, move))) return 780_000;
  if (endgameChecks && quietMoveCreatesProgressThreat(pos, move)) {
    return 760_000 + PIECE_VALUE[movedType] + pieceCentrality(moveTo(move)) * 8;
  }
  const sideIndex = pos.turn === WHITE ? 0 : 1;
  return searcher.history[sideIndex][moveFrom(move) * 25 + moveTo(move)];
}

function insertionSortMoveList(pos, list, ttMove, ply, searcher, previousMove, endgameChecks = false) {
  const count = list.count;
  const scores = list.scores;
  for (let i = 0; i < count; i += 1) {
    scores[i] = moveOrderScore(pos, list.moves[i], ttMove, ply, searcher, previousMove, endgameChecks, list.flags[i], list.movedTypes[i], list.capturedTypes[i]);
  }
  for (let i = 1; i < count; i += 1) {
    const move = list.moves[i];
    const flags = list.flags[i];
    const movedType = list.movedTypes[i];
    const captured = list.capturedTypes[i];
    const score = scores[i];
    let j = i - 1;
    while (j >= 0 && scores[j] < score) {
      list.moves[j + 1] = list.moves[j];
      list.flags[j + 1] = list.flags[j];
      list.movedTypes[j + 1] = list.movedTypes[j];
      list.capturedTypes[j + 1] = list.capturedTypes[j];
      scores[j + 1] = scores[j];
      j -= 1;
    }
    list.moves[j + 1] = move;
    list.flags[j + 1] = flags;
    list.movedTypes[j + 1] = movedType;
    list.capturedTypes[j + 1] = captured;
    scores[j + 1] = score;
  }
  return list;
}

function insertionSortMoves(pos, moves, ttMove, ply, searcher, previousMove, endgameChecks = false) {
  if (moves && typeof moves.count === 'number') return insertionSortMoveList(pos, moves, ttMove, ply, searcher, previousMove, endgameChecks);
  const scratch = searcher?.moveScoreStack?.[Math.min(ply, MAX_PLY)] || null;
  const scores = scratch && scratch.length >= moves.length ? scratch : new Int32Array(moves.length);
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

export function scoreToDisplay(score) {
  if (score >= MATE_BOUND) return `#${Math.max(1, Math.ceil((MATE - score) / 2))}`;
  if (score <= -MATE_BOUND) return `-#${Math.max(1, Math.ceil((MATE + score) / 2))}`;
  const pawns = score / 100;
  return `${pawns >= 0 ? '+' : ''}${pawns.toFixed(2)}`;
}

export function validateMateResult(position, externalLine) {
  if (!position || !externalLine?.mateVerified || !Array.isArray(externalLine.pv)) return false;
  const cursor = position.clone();
  const pv = [];
  for (const uci of externalLine.pv) {
    const move = uciToMove(cursor, uci);
    if (!move) return false;
    pv.push(move);
    makeMove(cursor, move);
  }
  const whiteScore = Number(externalLine.score || 0);
  if (!Number.isFinite(whiteScore)) return false;
  const rootScore = position.turn === WHITE ? whiteScore : -whiteScore;
  return verifyMatePV(position, { score: rootScore, pv });
}

function mateDistancePly(score) {
  return isMateScore(score) ? Math.max(1, MATE - Math.abs(score)) : 0;
}

// A mate score is only exposed to the UI when its PV can be replayed to an
// actual checkmate at the encoded distance. This prevents a stale cache entry,
// an interrupted aspiration re-search, or a corrupted TT bound from being
// presented as a solved line.
function verifyMatePV(root, line) {
  if (!line || !isMateScore(line.score) || !Array.isArray(line.pv)) return false;
  const distance = mateDistancePly(line.score);
  if (!distance || line.pv.length < distance) return false;
  const rootSide = root.turn;
  const matingSide = line.score > 0 ? rootSide : -rootSide;
  const cursor = root.clone();
  for (let ply = 0; ply < distance; ply += 1) {
    const move = line.pv[ply];
    if (!generateLegalMoves(cursor, false).includes(move)) return false;
    makeMove(cursor, move);
    const replies = generateLegalMoves(cursor, false);
    if (ply + 1 < distance && !replies.length) return false;
  }
  return cursor.turn === -matingSide
    && isInCheck(cursor)
    && generateLegalMoves(cursor, false).length === 0;
}

export class GardnerSearcher {
  constructor({ hashEntries = 180000, tablebaseProbe = null } = {}) {
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
    // v15.1: keep the v14.3 eval-cache budget while reducing avoidable worker/UI churn.
    // The power-of-two mask keeps lookup cheap while reducing repeat static
    // evaluation in closed low-mobility searches without changing semantics.
    this.evalMask = 524287;
    this.evalUsed = new Uint8Array(this.evalMask + 1);
    this.evalKey = new Uint32Array(this.evalMask + 1);
    this.evalLock = new Uint32Array(this.evalMask + 1);
    this.evalScore = new Int16Array(this.evalMask + 1);
    this.pvTable = Array.from({ length: MAX_PLY + 1 }, () => new Int32Array(MAX_PLY + 1));
    this.pvLength = new Int16Array(MAX_PLY + 1);
    this.staticEvalStack = new Int32Array(MAX_PLY + 2);
    this.moveScoreStack = Array.from({ length: MAX_PLY + 1 }, () => new Int32Array(96));
    this.moveStack = Array.from({ length: MAX_PLY + 1 }, () => createMoveList());
    this.excludedMoveStack = Array.from({ length: MAX_PLY + 1 }, () => createMoveList());
    this.quietMoveStack = Array.from({ length: MAX_PLY + 1 }, () => new Uint16Array(96));
    // Dedicated scratch avoids allocating while checking a legal immediate
    // third-repetition draw for the Stockfish-style alpha bound.
    this.repetitionMoveStack = Array.from({ length: MAX_PLY + 1 }, () => createMoveList());
    this.hashStackA = new Uint32Array(MAX_PLY + 128);
    this.hashStackB = new Uint32Array(MAX_PLY + 128);
    // Incremental order-independent path fingerprints keep TT entries aware
    // of repetition-relevant history while still allowing equivalent move-order
    // transpositions to share entries.
    this.ttPathSaltA = new Uint32Array(MAX_PLY + 128);
    this.ttPathSaltB = new Uint32Array(MAX_PLY + 128);
    this.nodes = 0;
    this.selDepth = 0;
    this.deadline = Infinity;
    this.rootBookMoves = new Set();
    this.rootHistory = [];
    this.rootRepetition = new Map();
    this.ttHistorySaltA = 0;
    this.ttHistorySaltB = 0;
    this.previousRootScores = new Map();
    // Stable, UI-safe root scores from fully sanitized non-mate or verified
    // mate lines.  Unlike previousRootScores, this map is never polluted by a
    // raw unverified mate bound from the current iteration.
    this.stableRootScores = new Map();
    this.previousPVScore = new Int32Array(8);
    this.previousPVScoreValid = new Uint8Array(8);
    this.completedDepth = 0;
    this.lastLines = [];
    this.liveRootLines = [];
    this.liveRootDepth = 0;
    this.startedAt = 0;
    this.rejectedMateClaims = 0;
    this.tablebaseProbe = typeof tablebaseProbe === 'function' ? tablebaseProbe : null;
    this.tablebaseProbeHits = 0;
    this.tablebaseExactDtmHits = 0;
    this.tablebaseWdlOnlyHits = 0;
    this.tablebaseGuideHits = 0;
    this.tablebaseGuideNodes = 0;
    this.rootMoveList = createMoveList();
    this.rootCountMoveList = createMoveList();
    this.progressCallback = null;
    this.progressRootSide = WHITE;
    this.progressMultipv = 3;
    this.progressStartDepth = 1;
    this.progressLastAt = 0;
    this.progressNextNode = PROGRESS_NODE_INTERVAL;
    this.lowProgressAudit = false;
  }


  setTablebaseProbe(probe) {
    this.tablebaseProbe = typeof probe === 'function' ? probe : null;
  }

  probeTablebase(pos) {
    if (!this.tablebaseProbe || !Number.isInteger(pos?.pieceCount) || pos.pieceCount > 5) return null;
    try {
      const result = this.tablebaseProbe(pos);
      if (!result || result.wdl === 2 || result.wdl === undefined) return null;
      this.tablebaseProbeHits += 1;
      return result;
    } catch {
      return null;
    }
  }

  tablebaseBound(hit, ply = 0) {
    const wdl = Math.sign(Number(hit?.wdl || 0));
    if (!wdl) return { score: DRAW, flag: TT_EXACT, exact: true, wdl: 0 };
    const dtm = Number(hit?.dtmPly || 0);
    if (hit?.exactDtm && Number.isFinite(dtm) && dtm > 0) {
      this.tablebaseExactDtmHits += 1;
      const distance = Math.min(MATE - 2, Math.max(1, Math.floor(ply) + Math.floor(dtm)));
      return {
        score: wdl > 0 ? MATE - distance : -MATE + distance,
        flag: TT_EXACT,
        exact: true,
        wdl
      };
    }
    this.tablebaseWdlOnlyHits += 1;
    // Same lower/upper-bound encoding used by Stockfish-style Syzygy probing:
    // a winning WDL node cannot score below the winning band, and a losing WDL
    // node cannot score above the losing band.  It is deliberately internal.
    return {
      score: wdl > 0 ? TB_WIN_SCORE - tablebaseBoundDistance(ply) : -TB_WIN_SCORE + tablebaseBoundDistance(ply),
      flag: wdl > 0 ? TT_LOWER : TT_UPPER,
      exact: false,
      wdl
    };
  }

  applyTablebaseBounds(pos, depth, alpha, beta, ply, { store = true } = {}) {
    const hit = this.probeTablebase(pos);
    if (!hit) return { hit: null, alpha, beta, cutoff: false, score: null, exact: false, bound: null };
    const bound = this.tablebaseBound(hit, ply);
    if (store) this.storeTT(pos, Math.max(0, Number(depth || 0)), bound.score, bound.flag, 0, 0, ply);
    if (bound.exact) {
      return { hit, alpha, beta, cutoff: true, score: bound.score, exact: true, bound };
    }
    if (bound.flag === TT_LOWER) {
      const nextAlpha = Math.max(alpha, bound.score);
      return {
        hit,
        alpha: nextAlpha,
        beta,
        cutoff: nextAlpha >= beta,
        score: bound.score,
        exact: false,
        bound
      };
    }
    const nextBeta = Math.min(beta, bound.score);
    return {
      hit,
      alpha,
      beta: nextBeta,
      cutoff: alpha >= nextBeta,
      score: bound.score,
      exact: false,
      bound
    };
  }

  reorderMoveListByScore(list) {
    for (let i = 1; i < list.count; i += 1) {
      const move = list.moves[i];
      const flags = list.flags[i];
      const movedType = list.movedTypes[i];
      const capturedType = list.capturedTypes[i];
      const score = list.scores[i];
      let j = i - 1;
      while (j >= 0 && list.scores[j] < score) {
        copyMoveEntry(list, j + 1, j);
        j -= 1;
      }
      list.moves[j + 1] = move;
      list.flags[j + 1] = flags;
      list.movedTypes[j + 1] = movedType;
      list.capturedTypes[j + 1] = capturedType;
      list.scores[j + 1] = score;
    }
  }

  // This is a one-ply move-ordering hint only.  It never scans a two-ply
  // second search tree, and never suppresses an opponent reply.
  // Unknown children stay in the normal alpha-beta tree.
  applyTablebaseMoveOrdering(pos, moves, ply = 0) {
    if (!this.tablebaseProbe || !moves?.count) return;
    let touched = false;
    for (let index = 0; index < moves.count; index += 1) {
      const move = moves.moves[index];
      const state = makeMove(pos, move);
      const hit = this.probeTablebase(pos);
      undoMove(pos, move, state);
      if (!hit) continue;
      const outcomeForMover = -Math.sign(Number(hit.wdl || 0));
      const dtm = Math.max(0, Number(hit.dtmPly || 0) + 1);
      let bonus = 0;
      if (outcomeForMover > 0) {
        bonus = hit.exactDtm ? 1_400_000 - Math.min(180_000, dtm * 80) : 850_000;
      } else if (outcomeForMover === 0) {
        bonus = 450_000;
      } else {
        bonus = hit.exactDtm ? -850_000 + Math.min(180_000, dtm * 80) : -700_000;
      }
      moves.scores[index] += bonus;
      touched = true;
    }
    if (touched) {
      this.tablebaseGuideNodes += 1;
      this.tablebaseGuideHits += 1;
      this.reorderMoveListByScore(moves);
    }
  }

  rootMoveTablebaseLine(move, childHit, plyFromRoot = 1) {
    if (!childHit || childHit.wdl === 2 || childHit.wdl === undefined) return null;
    const childWdl = Math.sign(Number(childHit.wdl || 0));
    const rootWdl = -childWdl;
    const source = childHit.source || 'gardner-tablebase';
    const signature = childHit.signature || '';
    if (rootWdl === 0) {
      return {
        move,
        score: DRAW,
        pv: [move],
        scoreKind: 'evaluation',
        scoreNumeric: true,
        mateVerified: false,
        tablebase: true,
        tablebaseRoot: false,
        rootMoveTablebase: true,
        tablebaseExactDtm: Boolean(childHit.exactDtm),
        tablebaseWdl: 0,
        dtm: 0,
        source,
        tablebaseSignature: signature,
        rootScoreExact: true,
        pvComplete: true,
        resultContract: 'evaluation',
        resultKindV2: 'evaluation'
      };
    }
    if (!childHit.exactDtm) return null;
    const dtm = Math.max(1, Number(childHit.dtmPly || 0) + Math.max(1, Math.floor(Number(plyFromRoot) || 1)));
    const score = rootWdl > 0 ? MATE - dtm : -MATE + dtm;
    return {
      move,
      score,
      pv: [move],
      scoreKind: 'mate',
      scoreNumeric: true,
      mateVerified: true,
      tablebase: true,
      tablebaseRoot: false,
      rootMoveTablebase: true,
      tablebaseExactDtm: true,
      tablebaseWdl: rootWdl,
      dtm,
      source,
      tablebaseSignature: signature,
      rootScoreExact: true,
      pvComplete: true,
      resultContract: 'mate',
      resultKindV2: 'mate'
    };
  }

  rootTablebaseChoices(pos, multipv = 3) {
    const root = this.probeTablebase(pos);
    if (!root || root.wdl === 2 || root.wdl === undefined) return null;
    const rootWdl = Math.sign(Number(root.wdl || 0));
    // Decisive root tablebase publication needs DTM so the visible result can
    // be mate-in-N. Draw roots are exact with WDL alone and display as 0.00.
    if (rootWdl !== 0 && !root.exactDtm) return null;
    const legal = generateLegalMoves(pos, false);
    if (!legal.length) return null;
    const candidates = [];
    for (const move of legal) {
      const state = makeMove(pos, move);
      const child = this.probeTablebase(pos);
      undoMove(pos, move, state);
      // Stockfish ranks root moves only after probing every legal child. If a
      // child is unavailable, fall back to ordinary alpha-beta instead of
      // publishing a partial tablebase root result.
      const line = this.rootMoveTablebaseLine(move, child, 1);
      if (!line) return null;
      candidates.push(line);
    }
    if (!candidates.length) return null;
    candidates.sort((left, right) => {
      if (right.score !== left.score) return right.score - left.score;
      return moveToUci(left.move).localeCompare(moveToUci(right.move));
    });
    return { root, candidates: candidates.slice(0, Math.max(1, Math.min(candidates.length, Number(multipv || 1)))) };
  }

  makeRootTablebaseResult(pos, choices) {
    if (!choices?.root || !choices?.candidates?.length) return null;
    const rootSide = pos.turn;
    const rootWdl = Math.sign(Number(choices.root.wdl || 0));
    const lines = choices.candidates.map(candidate => {
      const whiteScore = rootSide === WHITE ? candidate.score : -candidate.score;
      const mateVerified = Boolean(candidate.mateVerified && isMateScore(candidate.score));
      return {
        move: moveToUci(candidate.move),
        score: whiteScore,
        scoreText: scoreToDisplay(whiteScore),
        scoreKind: mateVerified ? 'mate' : 'evaluation',
        scoreNumeric: true,
        pv: [moveToUci(candidate.move)],
        mateVerified,
        tablebase: true,
        tablebaseRoot: true,
        rootMoveTablebase: false,
        tablebaseExactDtm: Boolean(candidate.tablebaseExactDtm),
        tablebaseWdl: Number(candidate.tablebaseWdl || 0),
        dtm: mateVerified ? Number(candidate.dtm || mateDistancePly(candidate.score)) : 0,
        source: candidate.source || choices.root.source || 'gardner-tablebase',
        tablebaseSignature: candidate.tablebaseSignature || choices.root.signature || '',
        rootScoreExact: true,
        pvComplete: true,
        resultContract: mateVerified ? 'mate' : 'evaluation',
        resultKindV2: mateVerified ? 'mate' : 'evaluation'
      };
    });
    return {
      engine: ENGINE_VERSION,
      engineLabel: `${ENGINE_VERSION} + GTB`,
      depth: 0,
      selDepth: 0,
      nodes: this.nodes,
      nps: 0,
      elapsed: 0,
      scoreDepth: 0,
      pvDepth: 1,
      pvTarget: 0,
      pvComplete: true,
      lines,
      terminal: true,
      completed: true,
      tablebase: true,
      tablebaseRoot: true,
      tablebaseSource: choices.root.source || 'gardner-tablebase',
      tablebaseSignature: choices.root.signature || '',
      tablebaseWdl: rootWdl,
      rootTurn: rootSide,
      solved: true,
      multiPvVerified: true,
      nextDepth: 0,
      searchDepth: 0,
      hashfull: Math.round(this.ttOccupied * 1000 / this.hashEntries),
      resultContract: lines[0]?.resultContract || 'evaluation',
      resultKindV2: lines[0]?.resultKindV2 || 'evaluation'
    };
  }

  clearVolatileSearchCaches({ clearOrdering = false } = {}) {
    // Ordinary TT/eval caches are root-local. A played move starts a fresh
    // minimax search; only a direct exact-root tablebase answer or a verified
    // mate can be reused outside the current worker iteration.
    this.ttUsed.fill(0);
    this.ttOccupied = 0;
    this.ttPathSaltA.fill(0);
    this.ttPathSaltB.fill(0);
    this.evalUsed.fill(0);
    this.previousRootScores.clear();
    this.stableRootScores.clear();
    this.previousPVScore.fill(0);
    this.previousPVScoreValid.fill(0);
    this.completedDepth = 0;
    this.lastLines = [];
    this.liveRootLines = [];
    this.liveRootDepth = 0;
    this.rejectedMateClaims = 0;
    this.tablebaseProbeHits = 0;
    this.tablebaseExactDtmHits = 0;
    this.tablebaseWdlOnlyHits = 0;
    this.tablebaseGuideHits = 0;
    this.tablebaseGuideNodes = 0;
    this.ttHistorySaltA = 0;
    this.ttHistorySaltB = 0;
    this.progressCallback = null;
    this.progressLastAt = 0;
    this.progressNextNode = PROGRESS_NODE_INTERVAL;
    this.lowProgressAudit = false;
    if (clearOrdering) {
      this.history.forEach(table => table.fill(0));
      this.killers.forEach(row => row.fill(0));
      this.countermoves.fill(0);
      this.captureHistory.forEach(table => table.fill(0));
    }
  }

  clear() {
    this.clearVolatileSearchCaches({ clearOrdering: true });
  }


  beginPosition({ reuseOrdinaryCache = false } = {}) {
    // A played move starts a fresh ordinary search; only heuristic ordering is aged.
    if (!reuseOrdinaryCache) this.clearVolatileSearchCaches({ clearOrdering: false });
    // Age move-ordering heuristics only.  They do not encode a score/PV for a
    // position and therefore do not violate the no-cross-root-cache policy.
    for (const table of this.history) {
      for (let i = 0; i < table.length; i += 1) table[i] = table[i] >> 1;
    }
    this.killers.forEach(row => row.fill(0));
    this.countermoves.fill(0);
    for (const table of this.captureHistory) {
      for (let i = 0; i < table.length; i += 1) table[i] = table[i] >> 1;
    }
    this.previousRootScores.clear();
    this.stableRootScores.clear();
    this.previousPVScore.fill(0);
    this.previousPVScoreValid.fill(0);
    this.completedDepth = 0;
    this.lastLines = [];
    this.liveRootLines = [];
    this.liveRootDepth = 0;
    this.tablebaseProbeHits = 0;
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
    if (!result || result.engine !== ENGINE_VERSION || !Array.isArray(result.lines) || !result.lines.length) return;
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
      if (!Number.isFinite(whiteScore)) continue;
      const rootScore = rootSide === WHITE ? whiteScore : -whiteScore;
      const trustedMate = Boolean(line.mateVerified) && isMateScore(rootScore);
      if (!trustedMate && !isOrdinaryEvaluationScore(rootScore)) continue;
      seeded.push({
        move: pv[0],
        score: rootScore,
        pv,
        mateVerified: trustedMate,
        dtm: trustedMate ? Number(line.dtm || mateDistancePly(rootScore)) : 0
      });
    }
    if (!seeded.length) return;
    this.lastLines = seeded;
    this.completedDepth = Math.max(this.completedDepth, Number(result.depth || 0));
    seeded.forEach((line, index) => {
      this.previousPVScore[index] = line.score;
      this.previousPVScoreValid[index] = 1;
      this.previousRootScores.set(line.move, line.score);
      this.stableRootScores.set(line.move, line.score);
    });
  }

  shouldStop() {
    if ((this.nodes & 255) !== 0) return;
    const now = performance.now();
    const progressAge = now - this.progressLastAt;
    if (this.progressCallback
      && (this.nodes >= this.progressNextNode || progressAge >= PROGRESS_MAX_STALENESS_MS)
      && progressAge >= PROGRESS_MIN_INTERVAL_MS) {
      this.emitProgressSnapshot(now);
    }
    if (now >= this.deadline) throw ABORT;
  }

  emitProgressSnapshot(now = performance.now()) {
    if (!this.progressCallback) return;
    const rootSide = this.progressRootSide;
    const source = this.mergeKnownRootLines(this.lastLines, this.progressMultipv);
    const toUci = move => typeof move === 'number' ? moveToUci(move) : String(move || '');
    const lines = source.filter(line => !isTablebaseBoundScore(line?.score)).map(line => {
      const rootScore = Number(line?.score || 0);
      const whiteScore = rootSide === WHITE ? rootScore : -rootScore;
      const rawPv = Array.isArray(line?.pv) && line.pv.length ? line.pv : [line?.move];
      return {
        move: toUci(line?.move),
        score: whiteScore,
        scoreText: scoreToDisplay(whiteScore),
        pv: rawPv.map(toUci).filter(Boolean),
        mateVerified: Boolean(line?.mateVerified),
        mateRejected: Boolean(line?.mateRejected),
        tablebase: Boolean(line?.tablebase),
        tablebaseWdl: Number(line?.tablebaseWdl || 0),
        tablebaseBound: Boolean(line?.tablebaseBound),
        tablebaseExactDtm: Boolean(line?.tablebaseExactDtm),
        liveUpdate: true,
        liveDepth: Math.max(0, Number(line?.liveDepth || this.liveRootDepth || this.progressStartDepth || 0)),
        dtm: Number(line?.dtm || 0)
      };
    }).filter(line => line.move);
    try {
      this.progressCallback({
        engine: ENGINE_VERSION,
        depth: this.completedDepth,
        selDepth: this.selDepth,
        nodes: this.nodes,
        nps: Math.round(this.nodes * 1000 / Math.max(1, now - this.startedAt)),
        elapsed: Math.round(Math.max(0, now - this.startedAt)),
        scoreDepth: this.completedDepth,
        attemptedDepth: Math.max(1, this.progressStartDepth),
        searchDepth: Math.max(1, this.liveRootDepth || this.progressStartDepth),
        nextDepth: Math.max(1, this.liveRootDepth || this.progressStartDepth),
        hashfull: Math.round(this.ttOccupied * 1000 / this.hashEntries),
        rootTurn: rootSide,
          lines,
        completed: false,
        liveUpdate: true,
        liveProgress: true,
        pvIncomplete: true,
        pvComplete: false,
        terminal: false,
        tablebase: false
      });
    } catch {
      // Progress reporting is optional and must never interrupt the search.
    }
    this.progressLastAt = now;
    this.progressNextNode = this.nodes + PROGRESS_NODE_INTERVAL;
  }

  repetitionCount(pos, ply) {
    let count = 1; // Current position.
    const limit = Math.min(ply, pos.halfmove);
    for (let back = 2; back <= limit; back += 2) {
      const index = ply - back;
      if (this.hashStackA[index] === pos.hashA && this.hashStackB[index] === pos.hashB) count += 1;
    }
    count += this.rootRepetition.get(`${pos.hashA}:${pos.hashB}`) || 0;
    return count;
  }

  isRepetition(pos, ply) {
    return this.repetitionCount(pos, ply) >= 3;
  }

  searchCycleScore(pos, ply) {
    // Like Stockfish's Position::is_draw(), only a formal third occurrence is
    // an exact terminal draw. A merely repeated position is not a leaf: the
    // side to move may still choose a different continuation.
    return this.repetitionCount(pos, ply) >= 3 ? DRAW : null;
  }

  hasImmediateRepetitionDraw(pos, ply) {
    // Stockfish also treats an available repetition as an alpha bound. This
    // compact implementation checks only a *real, legal next move* that would
    // create the third occurrence, so the bound is exact rather than a static
    // twofold-cycle heuristic. Never apply it at the root: root move search
    // must still score and display every legal candidate independently.
    if (ply <= 0 || pos.halfmove < 4) return false;
    const moves = this.repetitionMoveStack[Math.min(ply, MAX_PLY)];
    const moveCount = generateLegalMovesInto(pos, false, moves, true);
    for (let index = 0; index < moveCount; index += 1) {
      const flags = moves.flags[index];
      // A capture, pawn move or promotion changes the irreversible state and
      // cannot restore an earlier position on this variant.
      if (flags & (MOVE_FLAG_CAPTURE | MOVE_FLAG_PROMOTION)) continue;
      const move = moves.moves[index];
      if (typeOf(pos.board[moveFrom(move)]) === PAWN) continue;
      const state = makeMove(pos, move);
      this.recordSearchPath(ply + 1, pos);
      const repeats = this.repetitionCount(pos, ply + 1) >= 3;
      undoMove(pos, move, state);
      if (repeats) {
        this.recordSearchPath(ply, pos);
        return true;
      }
    }
    this.recordSearchPath(ply, pos);
    return false;
  }

  applyUpcomingRepetitionBound(pos, alpha, beta, ply) {
    // A legal move that creates the third occurrence guarantees at least a
    // draw for the side to move. This mirrors Stockfish's upcoming_repetition
    // alpha tightening, while leaving all ordinary alpha-beta search intact.
    if (ply <= 0 || alpha >= DRAW || !this.hasImmediateRepetitionDraw(pos, ply)) {
      return { alpha, beta, cutoff: false, score: null };
    }
    const nextAlpha = Math.max(alpha, DRAW);
    return {
      alpha: nextAlpha,
      beta,
      cutoff: nextAlpha >= beta,
      score: DRAW
    };
  }

  stableRejectedMateScore(pos, line, index) {
    // v20.3: an unverified mate-like TT/PV value is not a real score.  Reuse an
    // already completed non-mate value only when it belongs to the same move or
    // prior completed PV slot. If no such completed score exists, mark the
    // line non-publishable so the worker keeps the previous real score visible.
    const byStableMove = line?.move ? this.stableRootScores.get(line.move) : undefined;
    if (isOrdinaryEvaluationScore(byStableMove)) return byStableMove;
    const byIndex = index < this.previousPVScore.length && this.previousPVScoreValid[index]
      ? this.previousPVScore[index]
      : undefined;
    if (isOrdinaryEvaluationScore(byIndex)) return byIndex;
    const byPreviousMove = line?.move ? this.previousRootScores.get(line.move) : undefined;
    if (isOrdinaryEvaluationScore(byPreviousMove) && Math.abs(byPreviousMove) < 3000) {
      return byPreviousMove;
    }
    return null;
  }


  sanitizeRootLines(pos, lines, { recordStable = true } = {}) {
    const clean = [];
    for (const [index, source] of (lines || []).entries()) {
      const line = { ...source, pv: Array.isArray(source.pv) ? source.pv.slice() : [] };

      // A WDL-only tablebase bound is valid for alpha-beta, but it is not an
      // exact user-facing evaluation or mate distance. Keep a previously
      // completed ordinary score for the same move if one exists; otherwise do
      // not publish the bound.
      if (isTablebaseBoundScore(line.score)) {
        const replacement = this.stableRejectedMateScore(pos, line, index);
        if (!Number.isFinite(replacement)) continue;
        line.score = replacement;
        line.scoreKind = 'evaluation';
        line.scoreNumeric = true;
        line.tablebaseBound = true;
      }

      if (isMateScore(line.score)) {
        const exactTablebaseMate = Boolean(line.tablebase && line.tablebaseExactDtm);
        const mateVerified = exactTablebaseMate || Boolean(line.mateVerified && verifyMatePV(pos, line));
        if (!mateVerified) {
          const replacement = this.stableRejectedMateScore(pos, line, index);
          if (!Number.isFinite(replacement)) continue;
          line.score = replacement;
          line.scoreKind = 'evaluation';
          line.scoreNumeric = true;
          line.mateRejected = true;
          line.mateVerified = false;
          line.dtm = 0;
          this.rejectedMateClaims += 1;
        } else {
          line.mateVerified = true;
          line.dtm = Math.max(1, Number(line.dtm || mateDistancePly(line.score)));
        }
      } else {
        line.mateVerified = false;
        line.dtm = 0;
      }
      clean.push(line);
    }
    clean.sort((a, b) => b.score - a.score);
    if (recordStable) {
      clean.forEach((line, index) => {
        if (isOrdinaryEvaluationScore(line.score)) {
          if (index < this.previousPVScore.length) {
            this.previousPVScore[index] = line.score;
            this.previousPVScoreValid[index] = 1;
          }
          if (line.move) {
            this.previousRootScores.set(line.move, line.score);
            this.stableRootScores.set(line.move, line.score);
          }
        }
      });
    }
    return clean;
  }

  resetLiveRootLines(depth = 0) {
    this.liveRootDepth = Math.max(0, Number(depth || 0));
    this.liveRootLines = [];
  }

  mergeRootLinePv(previous, next) {
    if (!next?.move) return previous || next;
    const nextPv = Array.isArray(next.pv) && next.pv.length ? next.pv.slice() : [next.move];
    if (!previous?.move || previous.move !== next.move) return { ...next, pv: nextPv };
    const previousPv = Array.isArray(previous.pv) && previous.pv.length ? previous.pv : [previous.move];
    // A narrow PVS re-search often knows the new root score before it rebuilds
    // every continuation ply. Preserve a previously verified legal prefix only
    // when the new variation is literally that prefix; never splice divergent
    // branches together merely to make a line look longer.
    const nextIsPrefix = nextPv.length <= previousPv.length
      && nextPv.every((move, index) => move === previousPv[index]);
    if (nextIsPrefix && previousPv.length > nextPv.length) {
      return {
        ...next,
        pv: previousPv.slice(),
        pvPreservedFromLive: true
      };
    }
    return { ...next, pv: nextPv };
  }

  noteLiveRootLine(move, score, pv, depth, metadata = null) {
    if (!move || !Number.isFinite(score)) return;
    const line = {
      move,
      score,
      pv: Array.isArray(pv) && pv.length ? pv.slice() : [move],
      liveDepth: Math.max(0, Number(depth || this.liveRootDepth || 0)),
      liveUpdate: true,
      ...(metadata || {})
    };
    const index = this.liveRootLines.findIndex(item => item.move === move);
    if (index >= 0) this.liveRootLines[index] = this.mergeRootLinePv(this.liveRootLines[index], line);
    else this.liveRootLines.push(line);
    this.liveRootLines.sort((a, b) => b.score - a.score);
    if (this.liveRootLines.length > 8) this.liveRootLines.length = 8;
  }

  mergeKnownRootLines(baseLines = [], limit = 3) {
    const merged = new Map();
    for (const source of baseLines || []) {
      if (!source?.move) continue;
      merged.set(source.move, { ...source, pv: Array.isArray(source.pv) ? source.pv.slice() : [source.move] });
    }
    for (const source of this.liveRootLines || []) {
      if (!source?.move) continue;
      const live = {
        ...source,
        pv: Array.isArray(source.pv) && source.pv.length ? source.pv.slice() : [source.move],
        liveUpdate: true
      };
      merged.set(source.move, this.mergeRootLinePv(merged.get(source.move), live));
    }
    return [...merged.values()]
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(1, Number(limit || 3)));
  }

  recordSearchPath(ply, pos) {
    const index = Math.max(0, Math.min(this.ttPathSaltA.length - 1, Number(ply) || 0));
    this.hashStackA[index] = pos.hashA;
    this.hashStackB[index] = pos.hashB;
    const previousA = index ? this.ttPathSaltA[index - 1] : this.ttHistorySaltA;
    const previousB = index ? this.ttPathSaltB[index - 1] : this.ttHistorySaltB;
    // v18.4: additions are commutative, so equivalent transpositions that have
    // seen the same repetition-relevant position multiset share TT entries.
    // Counts remain encoded (a repeated position contributes twice), preserving
    // the history distinction that v18.3 introduced for threefold correctness.
    const contributionA = xorshift((pos.hashA ^ Math.imul(pos.hashB, 0x9e3779b1)) >>> 0);
    const contributionB = xorshift((pos.hashB ^ Math.imul(pos.hashA, 0x85ebca6b)) >>> 0);
    this.ttPathSaltA[index] = (previousA + contributionA) >>> 0;
    this.ttPathSaltB[index] = (previousB + contributionB) >>> 0;
  }

  probeTT(pos, ply) {
    const repetition = this.repetitionCount(pos, ply);
    const keyA = ttContextKeyA(pos, this.ttPathSaltA[ply] || this.ttHistorySaltA, repetition);
    const keyB = ttContextKeyB(pos, this.ttPathSaltB[ply] || this.ttHistorySaltB, repetition);
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
    const repetition = this.repetitionCount(pos, ply);
    const keyA = ttContextKeyA(pos, this.ttPathSaltA[ply] || this.ttHistorySaltA, repetition);
    const keyB = ttContextKeyB(pos, this.ttPathSaltB[ply] || this.ttHistorySaltB, repetition);
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
    if (isInsufficientMaterial(pos)) return DRAW;
    const cycleScore = this.searchCycleScore(pos, ply);
    if (cycleScore !== null) return cycleScore;

    const inCheck = isInCheck(pos);
    const tablebase = this.applyTablebaseBounds(pos, 0, alpha, beta, ply);
    if (tablebase.cutoff) return tablebase.score;
    alpha = tablebase.alpha;
    beta = tablebase.beta;
    let standPat = inCheck ? -INF : this.staticEvaluate(pos);
    // A WDL upper/lower bound is a game-theoretic constraint, not a centipawn
    // estimate. Keep quiescence inside that bound until it can cut the window.
    if (tablebase.bound?.flag === TT_LOWER) standPat = Math.max(standPat, tablebase.bound.score);
    if (tablebase.bound?.flag === TT_UPPER) standPat = Math.min(standPat, tablebase.bound.score);
    if (!inCheck) {
      if (standPat >= beta) return standPat;
      if (standPat > alpha) alpha = standPat;
      if (qDepth >= 12) return standPat;
    }

    const moves = this.moveStack[Math.min(ply, MAX_PLY)];
    const moveCount = inCheck
      ? generateLegalMovesInto(pos, false, moves, true)
      : generateLegalTacticalMovesInto(pos, qDepth < 3, moves, true);
    if (!moveCount) return inCheck ? -MATE + ply : standPat;
    insertionSortMoveList(pos, moves, 0, ply, this, previousMove);

    // At extreme quiescence depth, still resolve check with one legal evasion.
    if (qDepth >= 16) {
      let best = -INF;
      for (let moveIndex = 0; moveIndex < moveCount; moveIndex += 1) {
        const move = moves.moves[moveIndex];
        const state = makeMove(pos, move);
        const score = -this.staticEvaluate(pos);
        undoMove(pos, move, state);
        if (score > best) best = score;
      }
      return best;
    }

    for (let moveIndex = 0; moveIndex < moveCount; moveIndex += 1) {
      const move = moves.moves[moveIndex];
      const flags = moves.flags[moveIndex];
      const capture = moves.capturedTypes[moveIndex];
      const promotion = Boolean(flags & MOVE_FLAG_PROMOTION);
      const checking = !inCheck && !promotion && !capture && Boolean(flags & MOVE_FLAG_CHECK); // Quiet moves here are prefiltered checks.
      if (!inCheck && !checking && !promotion) {
        if (standPat + PIECE_VALUE[capture] + 80 < alpha) continue;
        if (staticExchangeEval(pos, move) < -35) continue;
      }
      const state = makeMove(pos, move);
      this.recordSearchPath(ply + 1, pos);
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
    if (isInsufficientMaterial(pos)) return DRAW;
    const cycleScore = this.searchCycleScore(pos, ply);
    if (cycleScore !== null) return cycleScore;
    alpha = Math.max(alpha, -MATE + ply);
    beta = Math.min(beta, MATE - ply - 1);
    if (alpha >= beta) return alpha;
    // Stockfish applies the optional upcoming-repetition bound only to the
    // full search. Quiescence keeps the cheaper exact threefold leaf test.
    if (depth <= 0) return this.qsearch(pos, alpha, beta, ply, 0, previousMove);
    const repetitionBound = this.applyUpcomingRepetitionBound(pos, alpha, beta, ply);
    if (repetitionBound.cutoff) return repetitionBound.score;
    alpha = repetitionBound.alpha;
    beta = repetitionBound.beta;

    const inCheck = isInCheck(pos);
    // Tablebase is part of this same alpha-beta node. An excluded-move search
    // is a synthetic node used by singular extension, so it must not inherit a
    // bound for the full legal position.
    if (!excludedMove) {
      const tablebase = this.applyTablebaseBounds(pos, depth, alpha, beta, ply);
      if (tablebase.cutoff) return tablebase.score;
      alpha = tablebase.alpha;
      beta = tablebase.beta;
    }
    const tt = excludedMove ? null : this.probeTT(pos, ply);
    const ttMove = tt?.move || 0;
    const localMoveStack = excludedMove ? this.excludedMoveStack : this.moveStack;
    if (tt && tt.depth >= depth && !pvNode) {
      if (tt.flag === TT_EXACT) return tt.score;
      if (tt.flag === TT_LOWER && tt.score >= beta) return tt.score;
      if (tt.flag === TT_UPPER && tt.score <= alpha) return tt.score;
    }

    const staticEval = inCheck ? -INF : (tt?.staticEval ?? this.staticEvaluate(pos));
    const improving = ply >= 2 && staticEval > this.staticEvalStack[ply - 2];
    this.staticEvalStack[ply] = staticEval;

    // Razoring: shallow hopeless nodes fall directly into quiescence.
    if (!pvNode && !inCheck && depth <= 2 && staticEval + 175 * depth < alpha) {
      const razor = this.qsearch(pos, alpha, beta, ply, 0, previousMove);
      if (razor <= alpha) return razor;
    }

    // Reverse futility pruning, deliberately conservative on the compact board.
    if (!pvNode && !inCheck && depth <= 5 && !isMateScore(beta) && staticEval - (105 + 58 * depth) >= beta && hasNonPawnMaterial(pos, pos.turn)) {
      return staticEval;
    }

    // Null move with verification at high depth. Disabled in likely 5×5 zugzwangs.
    if (allowNull && !pvNode && !inCheck && depth >= 4 && staticEval >= beta && nullMoveSafe(pos)) {
      const reduction = 2 + Math.floor(depth / 4) + Math.min(2, Math.floor((staticEval - beta) / 180));
      const state = makeNullMove(pos);
      this.recordSearchPath(ply + 1, pos);
      const nullScore = -this.search(pos, depth - reduction - 1, -beta, -beta + 1, ply + 1, false, 0, 0, false, extensions);
      undoNullMove(pos, state);
      if (nullScore >= beta && !isMateScore(nullScore)) {
        if (depth < 8) return nullScore;
        const verify = this.search(pos, depth - reduction, beta - 1, beta, ply, false, previousMove, 0, false, extensions);
        if (verify >= beta) return nullScore;
      }
    }

    // ProbCut with good captures only.
    if (!pvNode && !inCheck && depth >= 5 && !isMateScore(beta)) {
      const probBeta = beta + 140;
      const tactical = localMoveStack[Math.min(ply, MAX_PLY)];
      generateLegalMovesInto(pos, true, tactical, false);
      let tacticalCount = 0;
      const generatedCount = tactical.count;
      for (let read = 0; read < generatedCount; read += 1) {
        const move = tactical.moves[read];
        if (!(tactical.flags[read] & MOVE_FLAG_PROMOTION) && staticExchangeEval(pos, move) < 40) continue;
        if (tacticalCount !== read) copyMoveEntry(tactical, tacticalCount, read);
        tacticalCount += 1;
      }
      tactical.count = tacticalCount;
      insertionSortMoveList(pos, tactical, ttMove, ply, this, previousMove);
      const tacticalLimit = Math.min(5, tacticalCount);
      for (let tacticalIndex = 0; tacticalIndex < tacticalLimit; tacticalIndex += 1) {
        const move = tactical.moves[tacticalIndex];
        const state = makeMove(pos, move);
        this.recordSearchPath(ply + 1, pos);
        let score = -this.qsearch(pos, -probBeta, -probBeta + 1, ply + 1, 0, move);
        if (score >= probBeta) score = -this.search(pos, depth - 4, -probBeta, -probBeta + 1, ply + 1, false, move, 0, true, extensions);
        undoMove(pos, move, state);
        if (score >= probBeta) return score;
      }
    }

    const moves = localMoveStack[Math.min(ply, MAX_PLY)];
    let moveCount = generateLegalMovesInto(pos, false, moves, true);
    if (excludedMove) {
      let write = 0;
      for (let read = 0; read < moveCount; read += 1) {
        if (moves.moves[read] === excludedMove) continue;
        if (write !== read) copyMoveEntry(moves, write, read);
        write += 1;
      }
      moves.count = write;
      moveCount = write;
    }
    if (!moveCount) return excludedMove ? -INF + ply : inCheck ? -MATE + ply : DRAW;
    insertionSortMoveList(pos, moves, ttMove, ply, this, previousMove, false);
    this.applyTablebaseMoveOrdering(pos, moves, ply);

    let singularMove = 0;
    if (!excludedMove && depth >= 7 && ttMove && tt && tt.depth >= depth - 2 && tt.flag !== TT_UPPER && !isMateScore(tt.score)) {
      const singularBeta = tt.score - 36 - depth * 2;
      const singularDepth = Math.max(1, Math.floor((depth - 1) / 2));
      const value = this.search(pos, singularDepth, singularBeta - 1, singularBeta, ply, false, previousMove, ttMove, false, extensions);
      if (value < singularBeta) singularMove = ttMove;
    }

    const originalAlpha = alpha;
    let bestScore = -INF, bestMove = 0, legalIndex = 0, quietTried = 0;
    const searchedQuiets = this.quietMoveStack[Math.min(ply, MAX_PLY)];
    let searchedQuietCount = 0;
    for (let moveIndex = 0; moveIndex < moveCount; moveIndex += 1) {
      const move = moves.moves[moveIndex];
      const flags = moves.flags[moveIndex];
      const capture = Boolean(flags & MOVE_FLAG_CAPTURE);
      const promotion = Boolean(flags & MOVE_FLAG_PROMOTION);
      const quiet = !capture && !promotion;
      if (quiet) quietTried += 1;

      // Late move and futility pruning. Never applied in check or to root/book moves.
      if (legalIndex > 0 && !pvNode && !inCheck && depth <= 3 && quiet && move !== ttMove) {
        const lmpLimit = depth === 1 ? 5 : depth === 2 ? 9 : 15;
        if (quietTried > lmpLimit) continue;
        const margin = 85 * depth + (improving ? 55 : 0);
        if (staticEval + margin <= alpha && !(flags & MOVE_FLAG_CHECK) && !isPassedPawnMove(pos, move)) continue;
      }
      if (legalIndex > 0 && !pvNode && !inCheck && depth <= 4 && capture && !promotion && staticExchangeEval(pos, move) < -55 * depth) continue;
      if (quiet && searchedQuietCount < searchedQuiets.length) searchedQuiets[searchedQuietCount++] = move;

      const movedType = moves.movedTypes[moveIndex] || movePieceType(pos, move);
      const recapture = previousMove && capture && moveTo(previousMove) === moveTo(move);
      const passedPush = movedType === PAWN && isPassedPawnMove(pos, move);
      const checking = Boolean(flags & MOVE_FLAG_CHECK);
      let extension = 0;
      const extensionLimit = 2;
      if (extensions < extensionLimit) {
        const nearPromotion = passedPush && (rankOf(moveTo(move)) === 3 || rankOf(moveTo(move)) === 1);
        const forcingCheck = checking && (depth <= 5 || moveCount <= 4);
        const soundRecapture = recapture && depth <= 3 && staticExchangeEval(pos, move) >= -20;
        const checkEvasion = inCheck && depth <= 7;
        if (move === singularMove || promotion || nearPromotion || forcingCheck || soundRecapture || checkEvasion) extension = 1;
      }
      const nextExtensions = extensions + extension;
      const newDepth = depth - 1 + extension;

      const state = makeMove(pos, move);
      this.recordSearchPath(ply + 1, pos);
      let score;
      if (legalIndex === 0) {
        score = -this.search(pos, newDepth, -beta, -alpha, ply + 1, pvNode, move, 0, true, nextExtensions);
      } else {
        let reduction = 0;
        if (depth >= 3 && quiet && !inCheck && !checking && !promotion && move !== ttMove && !passedPush) {
          reduction = LMR_TABLE[Math.min(32, depth)][Math.min(32, legalIndex + 1)];
          if (pvNode) reduction -= 1;
          if (improving) reduction -= 1;
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
            for (let triedCursor = 0; triedCursor < searchedQuietCount; triedCursor += 1) {
              const tried = searchedQuiets[triedCursor];
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


  isLegalMoveForPosition(pos, move) {
    if (!move) return false;
    return generateLegalMoves(pos, false).some(candidate => candidate === move);
  }

  extendPvWithTt(pos, line, targetPlies = PV_DISPLAY_TAIL_TARGET) {
    if (!line?.move) return line;
    // v19.3: PV display is allowed to follow the entire currently searched
    // depth. PV_DISPLAY_TAIL_TARGET only limits optional display reconstruction;
    // it never changes whether an alpha-beta iteration is complete.
    const target = Math.max(1, Math.min(MAX_PLY - 2, Number(targetPlies || PV_DISPLAY_TAIL_TARGET)));
    const pv = Array.isArray(line.pv) && line.pv.length ? line.pv.slice() : [line.move];
    let pvReconstructed = Boolean(line.pvReconstructed);
    const cursor = pos.clone();
    const seen = new Set([`${cursor.hashA}:${cursor.hashB}:${cursor.turn}`]);
    // TT locks include the repetition-sensitive path context. Rebuild that
    // context while replaying the PV; using stale salts from the most recently
    // searched root branch was why short MultiPV lines could not recover their
    // already-computed TT continuation.
    this.recordSearchPath(0, cursor);
    let consumed = 0;
    for (; consumed < pv.length; consumed += 1) {
      const move = pv[consumed];
      if (!this.isLegalMoveForPosition(cursor, move)) {
        pv.length = consumed;
        break;
      }
      makeMove(cursor, move);
      this.recordSearchPath(Math.min(MAX_PLY - 2, consumed + 1), cursor);
      const identity = `${cursor.hashA}:${cursor.hashB}:${cursor.turn}`;
      if (seen.has(identity)) return { ...line, pv, pvReconstructed };
      seen.add(identity);
      if (!generateLegalMoves(cursor, false).length || isInsufficientMaterial(cursor)) {
        return { ...line, pv, pvReconstructed };
      }
    }
    while (pv.length < target) {
      const ply = Math.min(MAX_PLY - 2, pv.length);
      const ttMove = this.probeTT(cursor, ply)?.move || 0;
      if (!ttMove || !this.isLegalMoveForPosition(cursor, ttMove)) break;
      pv.push(ttMove);
      pvReconstructed = true;
      makeMove(cursor, ttMove);
      this.recordSearchPath(Math.min(MAX_PLY - 2, pv.length), cursor);
      const identity = `${cursor.hashA}:${cursor.hashB}:${cursor.turn}`;
      if (seen.has(identity)) break;
      seen.add(identity);
      if (!generateLegalMoves(cursor, false).length || isInsufficientMaterial(cursor)) break;
    }
    return { ...line, pv, pvReconstructed };
  }

  completeRootLines(pos, lines, depth = 0, multipv = 3) {
    if (!Array.isArray(lines) || !lines.length) return [];
    const target = Math.max(1, Number(depth || 1));
    return lines
      .map(source => {
        let line = { ...source, pv: Array.isArray(source.pv) && source.pv.length ? source.pv.slice() : [source.move] };
        const exactTablebaseMate = Boolean(line.tablebase && line.tablebaseExactDtm && isMateScore(line.score));
        if (!line.mateVerified && !exactTablebaseMate) line = this.extendPvWithTt(pos, line, Math.max(target, line.pv.length));
        if (exactTablebaseMate) {
          line = { ...line, mateVerified: true, dtm: Math.max(1, Number(line.dtm || mateDistancePly(line.score))) };
        } else if (line.mateVerified && !verifyMatePV(pos, line)) {
          line = { ...line, mateVerified: false, mateRejected: true, dtm: 0 };
        }
        return line;
      })
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(1, Number(multipv || 3)));
  }

  rootSearch(pos, depth, alpha, beta, excluded, preferredMove = 0, allowedRootMoves = null) {
    const moves = this.rootMoveList;
    generateLegalMovesInto(pos, false, moves, true);
    let write = 0;
    for (let read = 0; read < moves.count; read += 1) {
      const move = moves.moves[read];
      if (excluded.has(move)) continue;
      if (allowedRootMoves && !allowedRootMoves.has(move)) continue;
      if (write !== read) copyMoveEntry(moves, write, read);
      write += 1;
    }
    moves.count = write;
    if (!moves.count) return null;
    const ttMove = preferredMove || this.probeTT(pos, 0)?.move || 0;
    insertionSortMoves(pos, moves, ttMove, 0, this, 0, false);

    // When the side to move is worse, inspect historical repetitions before
    // ordering the root. This does not change the score of any move; it merely
    // lets Alpha-Beta prove a perpetual/threefold resource earlier instead of
    // spending the first time slice on attractive but losing continuations.
    const repetitionPriority = new Map();
    if (pos.halfmove >= 4 && this.staticEvaluate(pos) < -120 && this.rootRepetition.size) {
      for (let i = 0; i < moves.count; i += 1) {
        const move = moves.moves[i];
        const state = makeMove(pos, move);
        const repeats = this.rootRepetition.get(`${pos.hashA}:${pos.hashB}`) || 0;
        undoMove(pos, move, state);
        if (repeats) repetitionPriority.set(move, repeats);
      }
    }
    const scores = moves.scores;
    for (let i = 0; i < moves.count; i += 1) {
      const move = moves.moves[i];
      scores[i] = 0;
      scores[i] += (repetitionPriority.get(move) || 0) * 1_000_000;
      if (this.rootBookMoves.has(moveToUci(move))) scores[i] += 10_000;
      scores[i] += Math.max(-900, Math.min(900, this.previousRootScores.get(move) ?? -900));
    }
    for (let i = 1; i < moves.count; i += 1) {
      const move = moves.moves[i];
      const flags = moves.flags[i];
      const movedType = moves.movedTypes[i];
      const capturedType = moves.capturedTypes[i];
      const score = scores[i];
      let j = i - 1;
      while (j >= 0 && scores[j] < score) {
        copyMoveEntry(moves, j + 1, j);
        scores[j + 1] = scores[j];
        j -= 1;
      }
      moves.moves[j + 1] = move;
      moves.flags[j + 1] = flags;
      moves.movedTypes[j + 1] = movedType;
      moves.capturedTypes[j + 1] = capturedType;
      scores[j + 1] = score;
    }
    this.applyTablebaseMoveOrdering(pos, moves, 0);
    const fullWidthRoot = false;

    let bestScore = -INF, bestMove = 0, bestPV = [], bestMetadata = null, index = 0;
    for (let i = 0; i < moves.count; i += 1) {
      const move = moves.moves[i];
      const state = makeMove(pos, move);
      this.recordSearchPath(1, pos);
      this.pvLength[1] = 1;
      const childTbHit = this.probeTablebase(pos);
      const tablebaseLine = this.rootMoveTablebaseLine(move, childTbHit, 1);
      let score;
      let metadata = null;
      let candidatePV = [move];
      if (tablebaseLine) {
        score = tablebaseLine.score;
        metadata = tablebaseLine;
      } else if (index === 0 || fullWidthRoot) {
        score = -this.search(pos, depth - 1, -beta, -alpha, 1, true, move);
      } else {
        score = -this.search(pos, depth - 1, -alpha - 1, -alpha, 1, false, move);
        if (score > alpha && score < beta) score = -this.search(pos, depth - 1, -beta, -alpha, 1, true, move);
      }
      if (!metadata) {
        for (let pvIndex = 1; pvIndex < this.pvLength[1]; pvIndex += 1) candidatePV.push(this.pvTable[1][pvIndex]);
      }
      undoMove(pos, move, state);
      // A null-window MultiPV candidate may have an exact root score but only a
      // short local pvTable tail. Recover its legal TT continuation before the
      // 500 ms progress publisher sees it. Exact tablebase-DTM root moves do not
      // need a replayable PV to publish mate-in-N: the DTM itself is the proof.
      if (!metadata?.tablebaseExactDtm) {
        candidatePV = this.extendPvWithTt(pos, { move, pv: candidatePV }, Math.max(1, depth)).pv;
      }
      this.noteLiveRootLine(move, score, candidatePV, depth, metadata);
      if (isOrdinaryEvaluationScore(score)) this.previousRootScores.set(move, score);
      index += 1;
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
        bestPV = candidatePV.slice();
        bestMetadata = metadata;
      }
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }
    return { score: bestScore, move: bestMove, pv: bestPV, ...(bestMetadata || {}) };
  }

  searchMultiPV(pos, depth, multipv = 3, allowedRootMoves = null) {
    const rootCounts = this.rootCountMoveList;
    let legalCount = generateLegalMovesInto(pos, false, rootCounts, false);
    if (allowedRootMoves) {
      let write = 0;
      for (let read = 0; read < legalCount; read += 1) {
        const move = rootCounts.moves[read];
        if (!allowedRootMoves.has(move)) continue;
        if (write !== read) copyMoveEntry(rootCounts, write, read);
        write += 1;
      }
      rootCounts.count = write;
      legalCount = write;
    }
    if (!legalCount) return [];
    const limit = Math.min(Math.max(1, multipv), legalCount);
    const excluded = new Set();
    const results = [];

    for (let pvIndex = 0; pvIndex < limit; pvIndex += 1) {
      const priorLine = this.lastLines[pvIndex];
      const preferredMove = priorLine && !excluded.has(priorLine.move) ? priorLine.move : 0;
      const previous = priorLine?.score ?? (this.previousPVScoreValid[pvIndex] ? this.previousPVScore[pvIndex] : 0);
      let delta = depth >= 4 && !isMateScore(previous)
        ? 34 + Math.min(280, Math.floor(Math.abs(previous) / 8))
        : INF;
      let alpha = delta === INF ? -INF : previous - delta;
      let beta = delta === INF ? INF : previous + delta;
      let result = null;

      while (true) {
        result = this.rootSearch(pos, depth, alpha, beta, excluded, preferredMove, allowedRootMoves);
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
      // The aspiration loop only reaches this point after the root score is
      // inside its final window; mark this line as a complete root comparison.
      result.rootScoreExact = true;
      results.push(result);
      excluded.add(result.move);
      if (isOrdinaryEvaluationScore(result.score)) {
        this.previousPVScore[pvIndex] = result.score;
        this.previousPVScoreValid[pvIndex] = 1;
      }
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
    resumeResult = null,
    onProgress = null,
    restrictedRootMoves = null,
    reuseOrdinaryCache = false
  } = {}) {
    const rootSide = pos.turn;
    const allowedRootMoves = Array.isArray(restrictedRootMoves) && restrictedRootMoves.length
      ? new Set(restrictedRootMoves.filter(move => Number.isInteger(move) && move > 0))
      : null;
    statePoolCursor = 0;
    const rootSnapshot = pos.clone();
    if (newPosition) this.beginPosition({ reuseOrdinaryCache: Boolean(reuseOrdinaryCache) });
    if (resumeResult) this.seedFromResult(pos, resumeResult);
    this.resetLiveRootLines(0);
    this.generation = (this.generation + 1) & 255;
    this.nodes = 0;
    this.selDepth = 0;
    this.startedAt = performance.now();
    this.deadline = this.startedAt + Math.max(80, timeMs);
    this.progressCallback = typeof onProgress === 'function' ? onProgress : null;
    this.progressRootSide = rootSide;
    this.progressMultipv = Math.max(1, Number(multipv || 3));
    this.progressStartDepth = Math.max(1, Number(startDepth || 1));
    this.progressLastAt = this.startedAt;
    this.progressNextNode = PROGRESS_NODE_INTERVAL;
    this.lowProgressAudit = false;
    this.setBookMoves(bookMoves);
    this.rootHistory = Array.isArray(historyKeys) ? historyKeys : [];
    this.rootRepetition.clear();
    for (const key of this.rootHistory) {
      const identity = `${key.a >>> 0}:${key.b >>> 0}`;
      this.rootRepetition.set(identity, (this.rootRepetition.get(identity) || 0) + 1);
    }
    const repetitionFingerprint = repetitionContextFingerprint(this.rootRepetition);
    this.ttHistorySaltA = repetitionFingerprint.a;
    this.ttHistorySaltB = repetitionFingerprint.b;
    this.recordSearchPath(0, pos);
    const rootMoves = generateLegalMoves(pos);
    const eligibleRootMoves = allowedRootMoves ? rootMoves.filter(move => allowedRootMoves.has(move)) : rootMoves;
    this.lowProgressAudit = false;
    const noLegalRootMove = !eligibleRootMoves.length;
    const repetitionDraw = this.repetitionCount(pos, 0) >= 3;
    // A synchronously resident exact root tablebase ranks every legal root
    // move before iterative deepening. This remains true for material-draw
    // positions, so a loaded tablebase root displays its exact 0.00 answer.
    if (!noLegalRootMove && !repetitionDraw && !allowedRootMoves) {
      const rootTablebase = this.rootTablebaseChoices(pos, multipv);
      const direct = this.makeRootTablebaseResult(pos, rootTablebase);
      if (direct) {
        restorePosition(pos, rootSnapshot);
        statePoolCursor = 0;
        return direct;
      }
    }
    const rootTerminal = noLegalRootMove || isInsufficientMaterial(pos) || repetitionDraw;

    let completed = null;
    let depth = Math.max(1, startDepth);
    try {
      if (!rootTerminal) for (; depth <= maxDepth; depth += 1) {
        this.resetLiveRootLines(depth);
        const lines = this.sanitizeRootLines(pos, this.searchMultiPV(pos, depth, multipv, allowedRootMoves));
        if (!lines.length) break;
        completed = { depth, lines };
        this.lastLines = lines;
        this.completedDepth = depth;
        // A verified mate line is complete for this iterative root search.
        if (lines[0]?.mateVerified && lines[0].score > 0 && isMateScore(lines[0].score) && depth >= 3) break;
        this.shouldStop();
      }
    } catch (error) {
      if (error !== ABORT) throw error;
    }

    restorePosition(pos, rootSnapshot);
    statePoolCursor = 0;
    const liveIncomplete = Boolean(this.liveRootLines.length && (!completed || Number(completed.depth || 0) < this.liveRootDepth));
    const resultDepth = completed?.depth || this.completedDepth || 0;
    const finalSourceLines = liveIncomplete
      ? this.mergeKnownRootLines(completed?.lines || this.lastLines || [], multipv)
      : (completed?.lines || this.lastLines || []);
    let finalLines = this.sanitizeRootLines(pos, finalSourceLines, { recordStable: !liveIncomplete });
    if (!rootTerminal && finalLines.length) {
      finalLines = this.completeRootLines(pos, finalLines, completed?.depth || this.completedDepth || this.liveRootDepth || maxDepth, multipv);
      finalLines = this.sanitizeRootLines(pos, finalLines, { recordStable: !liveIncomplete });
      this.lastLines = finalLines;
    }
    // A completed alpha-beta iteration is complete even when the displayed PV
    // stops early at a repetition draw, terminal node or TT boundary. Search
    // interruption—not PV length—is the only reason to retry a depth.
    const bestPvDepth = Math.max(0, ...finalLines.map(line => Array.isArray(line?.pv) ? line.pv.length : 0), 0);
    const pvTarget = 0;
    const storedPvDepth = bestPvDepth;
    const elapsed = Math.max(1, performance.now() - this.startedAt);
    const lines = finalLines.map(line => {
      const rawRootScore = Number(line.score || 0);
      const mateVerified = Boolean(line.mateVerified && isMateScore(rawRootScore));
      const whiteScore = rootSide === WHITE ? rawRootScore : -rawRootScore;
      return {
        move: moveToUci(line.move),
        score: whiteScore,
        scoreText: scoreToDisplay(whiteScore),
        scoreKind: mateVerified ? 'mate' : 'evaluation',
        scoreNumeric: true,
        pv: line.pv.map(moveToUci),
        mateVerified,
        mateRejected: Boolean(line.mateRejected),
        tablebase: Boolean(line.tablebase && !line.tablebaseBound),
        tablebaseRoot: Boolean(line.tablebaseRoot),
        tablebaseWdl: Number(line.tablebaseWdl || 0),
        tablebaseExactDtm: Boolean(line.tablebaseExactDtm),
        pvReconstructed: Boolean(line.pvReconstructed),
        rootScoreExact: line.rootScoreExact !== false,
        liveUpdate: Boolean(line.liveUpdate),
        liveDepth: Number(line.liveDepth || 0),
        dtm: mateVerified ? Number(line.dtm || mateDistancePly(rawRootScore)) : 0,
        // PV length is not a validity criterion; a legal root score from a
        // completed iteration remains complete when its PV ends in repetition.
        pvComplete: !liveIncomplete && (line.rootScoreExact !== false || mateVerified),
        resultContract: mateVerified ? 'mate' : 'evaluation',
        resultKindV2: mateVerified ? 'mate' : 'evaluation'
      };
    });
    const visibleRootCount = Math.min(Math.max(1, multipv), lines.length);
    const multiPvVerified = Boolean(lines.length) && lines.slice(0, visibleRootCount).every(line => line.rootScoreExact === true || line.tablebase || line.mateVerified);
    const iterationComplete = Boolean(completed) && !liveIncomplete && multiPvVerified;
    const pvIncomplete = !iterationComplete;
    const pvComplete = iterationComplete;
    return {
      engine: ENGINE_VERSION,
      depth: resultDepth,
      selDepth: this.selDepth,
      nodes: this.nodes,
      nps: Math.round(this.nodes * 1000 / elapsed),
      elapsed: Math.round(elapsed),
      scoreDepth: resultDepth,
      pvDepth: storedPvDepth,
      pvTarget,
      pvComplete,
      lines,
      terminal: rootTerminal,
      multiPvVerified,
      completed: iterationComplete,
      liveUpdate: !iterationComplete,
      pvIncomplete,
      liveDepth: liveIncomplete ? this.liveRootDepth : 0,
      attemptedDepth: Math.max(1, startDepth),
      hashfull: Math.round(this.ttOccupied * 1000 / this.hashEntries),
      rejectedMateClaims: this.rejectedMateClaims,
      tablebaseProbeHits: this.tablebaseProbeHits,
      tablebaseExactDtmHits: this.tablebaseExactDtmHits,
      tablebaseWdlOnlyHits: this.tablebaseWdlOnlyHits,
      tablebaseGuideHits: this.tablebaseGuideHits,
      tablebaseGuideNodes: this.tablebaseGuideNodes,
      rootTurn: rootSide,
      nextDepth: iterationComplete ? Math.max(1, completed.depth + 1) : Math.max(1, startDepth)
    };
  }
}

export function analyzeOnce(fen, options = {}) {
  const searcher = new GardnerSearcher(options);
  return searcher.analyze(EnginePosition.fromFEN(fen), options);
}

export const EngineInternals = Object.freeze({
  EMPTY, PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, WHITE, BLACK,
  encodeMove, moveFrom, moveTo, movePromotion, makeMove, undoMove, givesCheck, isInCheck,
  isCapture, isPromotion, kingSquare, materialProfile,
  PIECE_VALUE, MATE, MATE_BOUND, TB_WIN_SCORE, TB_WIN_MIN_SCORE, MIN_TABLEBASE_AUDIT_SCORE, MAX_TABLEBASE_AUDIT_SCORE,
  sideOf, typeOf, fileOf, rankOf, square, inside,
  structuralDrawProfile, lowProgressLegalProfile, lowProgressSearchNode, quietHeavyOfferBreakthroughThreat
});
