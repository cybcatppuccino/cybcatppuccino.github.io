import { COORD_SYSTEMS, TYPES, coord, fileOf, parseCoord } from './constants.js';
import { gameStatus, legalMoves } from './rules.js';

const SAN_PIECE = Object.freeze({ p: '', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' });

function normalizeCoordSystem(coordSystem) {
  return coordSystem === COORD_SYSTEMS.LEGACY_STUDY || coordSystem === 'legacy' ? COORD_SYSTEMS.LEGACY_STUDY : COORD_SYSTEMS.STANDARD;
}

function coordPattern(coordSystem) {
  return normalizeCoordSystem(coordSystem) === COORD_SYSTEMS.LEGACY_STUDY ? '[b-f][2-6]' : '[a-e][1-5]';
}

export function moveKey(move) {
  return `${move.from}-${move.to}-${move.promotion || ''}`;
}

export function moveToUci(move, { coordSystem = COORD_SYSTEMS.STANDARD } = {}) {
  const system = normalizeCoordSystem(coordSystem);
  return `${coord(move.from, system)}${coord(move.to, system)}${move.promotion || ''}`;
}

export function moveToSAN(position, move, { coordSystem = COORD_SYSTEMS.STANDARD } = {}) {
  const system = normalizeCoordSystem(coordSystem);
  const moving = position.pieceAt(move.from);
  if (!moving) return moveToUci(move, { coordSystem: system });
  const fromCoord = coord(move.from, system);
  const toCoord = coord(move.to, system);
  let text = SAN_PIECE[moving.type];
  const isCapture = Boolean(move.captured || position.pieceAt(move.to));

  if (moving.type === TYPES.PAWN) {
    if (isCapture) text += fromCoord[0];
  } else {
    const alternatives = legalMoves(position).filter(other =>
      other.to === move.to &&
      other.from !== move.from &&
      position.pieceAt(other.from)?.type === moving.type
    );
    if (alternatives.length) {
      const sameFile = alternatives.some(other => fileOf(other.from) === fileOf(move.from));
      text += sameFile ? fromCoord[1] : fromCoord[0];
    }
  }

  if (isCapture) text += 'x';
  text += toCoord;
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

export function findMoveBySAN(position, token, { coordSystem = COORD_SYSTEMS.STANDARD } = {}) {
  const system = normalizeCoordSystem(coordSystem);
  const wanted = normalizeSAN(token);
  if (!wanted || wanted === '--') return null;
  const moves = legalMoves(position);

  // SAN match generated from the current legal position. v12.2 defaults to
  // standard A1–E5 output; legacy mode exists only for old archived PGNs.
  let match = moves.find(move => normalizeSAN(moveToSAN(position, move, { coordSystem: system })) === wanted);
  if (match) return match;

  // Tolerate omitted disambiguation and check markers in historical PGNs.
  const targetPattern = coordPattern(system);
  const disambiguation = new RegExp(`^([KQRBN])[a-f1-6](?=x?${targetPattern})`);
  match = moves.find(move => {
    const san = normalizeSAN(moveToSAN(position, move, { coordSystem: system }));
    return san.replace(disambiguation, '$1') === wanted;
  });
  if (match) return match;

  // Coordinate notation fallback, e.g. a2a3/a2-a3 in v12.2 or b3b4 in legacy PGNs.
  const coordRegex = new RegExp(`^(${targetPattern})[-x]?(${targetPattern})(?:=)?([QRBNqrbn])?$`, 'i');
  const coordMatch = wanted.match(coordRegex);
  if (coordMatch) {
    const from = parseCoord(coordMatch[1].toLowerCase(), system);
    const to = parseCoord(coordMatch[2].toLowerCase(), system);
    const promotion = coordMatch[3]?.toLowerCase() || null;
    return moves.find(move =>
      move.from === from &&
      move.to === to &&
      (move.promotion || null) === promotion
    ) || null;
  }
  return null;
}
