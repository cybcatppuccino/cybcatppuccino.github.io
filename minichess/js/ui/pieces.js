import { PIECE_GLYPHS } from '../core/constants.js';

// Standard is the local Syzygy-style SVG set bundled with this patch.  All
// other sets remain text based so they can be recoloured without extra assets.
export const PIECE_STYLES = Object.freeze([
  { id: 'standard', label: 'Standard' },
  { id: 'classic', label: 'Classic' },
  { id: 'neo', label: 'Neo' },
  { id: 'ivory', label: 'Ivory' },
  { id: 'letter', label: 'Minimal' }
]);

const LETTERS = Object.freeze({ p: 'P', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' });

export function isPieceStyle(style) {
  return PIECE_STYLES.some(item => item.id === style);
}

export function pieceText(style, color, type) {
  return style === 'letter' ? LETTERS[type] : style === 'standard' ? '' : PIECE_GLYPHS[color][type];
}

export function applyPieceStyle(node, style, color, type) {
  const resolved = isPieceStyle(style) ? style : 'standard';
  node.classList.add('piece', color === 'w' ? 'white' : 'black', `piece-style-${resolved}`);
  node.dataset.piece = `${color}${type}`;
  node.dataset.pieceStyle = resolved;
  node.textContent = pieceText(resolved, color, type);
}
