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

// v12.2: A1–E5 is the canonical public coordinate system everywhere
// (UI, SAN, UCI, engine output, compact FEN). The old Gardnerfish study
// b2–f6 coordinates remain available only for reading legacy research PGNs
// and pasted legacy 8×8 study FENs.
export const STANDARD_FILES = Object.freeze(['a', 'b', 'c', 'd', 'e']);
export const STANDARD_RANKS = Object.freeze(['1', '2', '3', '4', '5']);
export const LEGACY_STUDY_FILES = Object.freeze(['b', 'c', 'd', 'e', 'f']);
export const LEGACY_STUDY_RANKS = Object.freeze(['2', '3', '4', '5', '6']);
export const COORD_SYSTEMS = Object.freeze({
  STANDARD: 'standard',
  LEGACY_STUDY: 'legacy-study'
});

// Backward-compatible aliases. New code should use STANDARD_* explicitly.
export const COMPACT_FILES = STANDARD_FILES;
export const COMPACT_RANKS = STANDARD_RANKS;
export const STUDY_FILES = STANDARD_FILES;
export const STUDY_RANKS = STANDARD_RANKS;

export const INITIAL_COMPACT_FEN = 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1';
export const INITIAL_STANDARD_FEN = INITIAL_COMPACT_FEN;
export const INITIAL_LEGACY_STUDY_FEN = '8/8/1rnbqk2/1ppppp2/8/1PPPPP2/1RNBQK2/8 w - - 0 1';
// Backward-compatible name for external callers that still import it.
export const INITIAL_STUDY_FEN = INITIAL_LEGACY_STUDY_FEN;

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

function coordFromTables(sq, files, ranks) {
  return `${files[fileOf(sq)]}${ranks[rankOf(sq)]}`;
}

function parseCoordFromTables(coord, files, ranks) {
  if (!coord || coord.length < 2) return -1;
  const file = files.indexOf(coord[0].toLowerCase());
  const rank = ranks.indexOf(coord[1]);
  return file >= 0 && rank >= 0 ? square(file, rank) : -1;
}

export function standardCoord(sq) {
  return coordFromTables(sq, STANDARD_FILES, STANDARD_RANKS);
}

export function parseStandardCoord(coord) {
  return parseCoordFromTables(coord, STANDARD_FILES, STANDARD_RANKS);
}

export function legacyStudyCoord(sq) {
  return coordFromTables(sq, LEGACY_STUDY_FILES, LEGACY_STUDY_RANKS);
}

export function parseLegacyStudyCoord(coord) {
  return parseCoordFromTables(coord, LEGACY_STUDY_FILES, LEGACY_STUDY_RANKS);
}

export function coord(sq, coordSystem = COORD_SYSTEMS.STANDARD) {
  return coordSystem === COORD_SYSTEMS.LEGACY_STUDY ? legacyStudyCoord(sq) : standardCoord(sq);
}

export function parseCoord(coordText, coordSystem = COORD_SYSTEMS.STANDARD) {
  return coordSystem === COORD_SYSTEMS.LEGACY_STUDY ? parseLegacyStudyCoord(coordText) : parseStandardCoord(coordText);
}

export function compactCoord(sq) {
  return standardCoord(sq);
}

export function parseCompactCoord(coordText) {
  return parseStandardCoord(coordText);
}

// v12.2 compatibility: the public "study" coordinate now follows the
// standard A1–E5 system. Use legacyStudyCoord/parseLegacyStudyCoord when
// reading old b2–f6 archival material.
export function studyCoord(sq) {
  return standardCoord(sq);
}

export function parseStudyCoord(coordText) {
  return parseStandardCoord(coordText);
}
