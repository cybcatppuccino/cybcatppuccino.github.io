import { Position } from './position.js';

const STANDARD = Object.freeze(['r', 'n', 'b', 'q', 'k']);

export const START_LAYOUTS = Object.freeze([
  { id: 'standard', label: 'Standard', description: 'R N B Q K for both sides.' },
  { id: 'central', label: 'Central symmetry', description: 'Black is the 180° rotation of White.' },
  { id: 'mini60', label: 'MiniChess 60', description: 'One random 5! back rank, vertically mirrored.' },
  { id: 'mini60-central', label: 'Central MiniChess 60', description: 'One random 5! back rank, centrally mirrored.' },
  { id: 'random', label: 'Pure random', description: 'Independent random 5! back ranks.' },
  { id: 'mallet', label: 'Mallett Chess', description: 'White RNKQN; Black RBKQB.' }
]);

function shuffle(values, rng = Math.random) {
  const result = [...values];
  for (let i = result.length - 1; i > 0; i -= 1) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j], result[i]];
  }
  return result;
}

function placement(whiteBack, blackBack) {
  const white = whiteBack.join('').toUpperCase();
  const black = blackBack.join('').toLowerCase();
  return `${black}/ppppp/5/PPPPP/${white} w - - 0 1`;
}

export function layoutDefinition(id, rng = Math.random) {
  const key = START_LAYOUTS.some(layout => layout.id === id) ? id : 'standard';
  if (key === 'standard') return { id: key, white: [...STANDARD], black: [...STANDARD] };
  if (key === 'central') return { id: key, white: [...STANDARD], black: [...STANDARD].reverse() };
  if (key === 'mini60') {
    const white = shuffle(STANDARD, rng);
    return { id: key, white, black: [...white] };
  }
  if (key === 'mini60-central') {
    const white = shuffle(STANDARD, rng);
    return { id: key, white, black: [...white].reverse() };
  }
  if (key === 'random') return { id: key, white: shuffle(STANDARD, rng), black: shuffle(STANDARD, rng) };
  return { id: 'mallet', white: ['r', 'n', 'k', 'q', 'n'], black: ['r', 'b', 'k', 'q', 'b'] };
}

export function createStartPosition(id, rng = Math.random) {
  const definition = layoutDefinition(id, rng);
  return {
    ...definition,
    position: Position.fromFEN(placement(definition.white, definition.black)),
    signature: `${definition.white.join('').toUpperCase()} / ${definition.black.join('').toUpperCase()}`
  };
}
