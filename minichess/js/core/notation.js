import { TYPES, fileOf, studyCoord } from './constants.js';
import { gameStatus, legalMoves } from './rules.js';

const SAN_PIECE = Object.freeze({ p: '', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' });

export function moveKey(move) {
  return `${move.from}-${move.to}-${move.promotion || ''}`;
}

export function moveToUci(move) {
  return `${studyCoord(move.from)}${studyCoord(move.to)}${move.promotion || ''}`;
}

export function moveToSAN(position, move) {
  const moving = position.pieceAt(move.from);
  if (!moving) return moveToUci(move);
  let text = SAN_PIECE[moving.type];
  const isCapture = Boolean(move.captured || position.pieceAt(move.to));

  if (moving.type === TYPES.PAWN) {
    if (isCapture) text += studyCoord(move.from)[0];
  } else {
    const alternatives = legalMoves(position).filter(other =>
      other.to === move.to &&
      other.from !== move.from &&
      position.pieceAt(other.from)?.type === moving.type
    );
    if (alternatives.length) {
      const sameFile = alternatives.some(other => fileOf(other.from) === fileOf(move.from));
      text += sameFile ? studyCoord(move.from)[1] : studyCoord(move.from)[0];
    }
  }

  if (isCapture) text += 'x';
  text += studyCoord(move.to);
  if (move.promotion) text += `=${move.promotion.toUpperCase()}`;

  const next = position.makeMove(move);
  const status = gameStatus(next);
  if (status.state === 'checkmate') text += '#';
  else if (status.check) text += '+';
  return text;
}

export function normalizeSAN(token) {
  return String(token || '')
    .trim()
    .replace(/\u00a0/g, '')
    .replace(/(?:e\.p\.)/gi, '')
    .replace(/[!?]+/g, '')
    .replace(/[+#]+$/g, '')
    .replace(/^0-0-0$|^O-O-O$/i, 'O-O-O')
    .replace(/^0-0$|^O-O$/i, 'O-O')
    .replace(/\s+/g, '');
}

export function findMoveBySAN(position, token) {
  const wanted = normalizeSAN(token);
  if (!wanted || wanted === '--') return null;
  const moves = legalMoves(position);

  // Standard SAN match generated from the current legal position.
  let match = moves.find(move => normalizeSAN(moveToSAN(position, move)) === wanted);
  if (match) return match;

  // Tolerate omitted disambiguation and check markers in historical PGNs.
  const strippedWanted = wanted.replace(/^([KQRBN])?[a-f2-6]?x?/, prefix => prefix);
  match = moves.find(move => {
    const san = normalizeSAN(moveToSAN(position, move));
    return san === strippedWanted || san.replace(/^([KQRBN])[a-f2-6](?=x?[b-f][2-6])/, '$1') === wanted;
  });
  if (match) return match;

  // Coordinate notation fallback, e.g. b3b4 or b3-b4.
  const coord = wanted.match(/^([b-f][2-6])[-x]?([b-f][2-6])(?:=)?([QRBNqrbn])?$/);
  if (coord) {
    const promotion = coord[3]?.toLowerCase() || null;
    return moves.find(move =>
      studyCoord(move.from) === coord[1] &&
      studyCoord(move.to) === coord[2] &&
      (move.promotion || null) === promotion
    ) || null;
  }
  return null;
}
