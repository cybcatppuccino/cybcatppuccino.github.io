export const BOARD_SIZE = 5;
export const SQUARE_COUNT = 25;

export const COLORS = Object.freeze({ WHITE: 'w', BLACK: 'b' });
export const TYPES = Object.freeze({
  PAWN: 'p', KNIGHT: 'n', BISHOP: 'b', ROOK: 'r', QUEEN: 'q', KING: 'k'
});

export const PIECE_GLYPHS = Object.freeze({
  w: { p: '♙', n: '♘', b: '♗', r: '♖', q: '♕', k: '♔' },
  b: { p: '♟', n: '♞', b: '♝', r: '♜', q: '♛', k: '♚' }
});

export const PIECE_NAMES = Object.freeze({
  p: 'Pawn', n: 'Knight', b: 'Bishop', r: 'Rook', q: 'Queen', k: 'King'
});

export const STUDY_FILES = Object.freeze(['b', 'c', 'd', 'e', 'f']);
export const STUDY_RANKS = Object.freeze(['2', '3', '4', '5', '6']);
export const COMPACT_FILES = Object.freeze(['a', 'b', 'c', 'd', 'e']);
export const COMPACT_RANKS = Object.freeze(['1', '2', '3', '4', '5']);

export const INITIAL_COMPACT_FEN = 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1';
export const INITIAL_STUDY_FEN = '8/8/1rnbqk2/1ppppp2/8/1PPPPP2/1RNBQK2/8 w - - 0 1';

export const MOVE_FLAGS = Object.freeze({
  QUIET: 0,
  CAPTURE: 1,
  PROMOTION: 2,
  PROMOTION_CAPTURE: 3
});

export function opposite(color) {
  return color === COLORS.WHITE ? COLORS.BLACK : COLORS.WHITE;
}

export function square(file, rank) {
  return rank * BOARD_SIZE + file;
}

export function fileOf(sq) {
  return sq % BOARD_SIZE;
}

export function rankOf(sq) {
  return Math.floor(sq / BOARD_SIZE);
}

export function studyCoord(sq) {
  return `${STUDY_FILES[fileOf(sq)]}${STUDY_RANKS[rankOf(sq)]}`;
}

export function compactCoord(sq) {
  return `${COMPACT_FILES[fileOf(sq)]}${COMPACT_RANKS[rankOf(sq)]}`;
}

export function parseStudyCoord(coord) {
  if (!coord || coord.length < 2) return -1;
  const file = STUDY_FILES.indexOf(coord[0].toLowerCase());
  const rank = STUDY_RANKS.indexOf(coord[1]);
  return file >= 0 && rank >= 0 ? square(file, rank) : -1;
}

export function parseCompactCoord(coord) {
  if (!coord || coord.length < 2) return -1;
  const file = COMPACT_FILES.indexOf(coord[0].toLowerCase());
  const rank = COMPACT_RANKS.indexOf(coord[1]);
  return file >= 0 && rank >= 0 ? square(file, rank) : -1;
}
