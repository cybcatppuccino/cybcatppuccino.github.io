import { PIECE_GLYPHS } from '../core/constants.js';

export const PIECE_STYLES = Object.freeze([
  { id: 'standard', label: 'Standard' },
  { id: 'neo', label: 'Neo' },
  { id: 'letter', label: 'Minimal' }
]);

const LETTERS = Object.freeze({ p: 'P', n: 'N', b: 'B', r: 'R', q: 'Q', k: 'K' });

export function pieceText(style, color, type) {
  return style === 'letter' ? LETTERS[type] : PIECE_GLYPHS[color][type];
}

export function applyPieceStyle(node, style, color, type) {
  node.classList.add('piece', color === 'w' ? 'white' : 'black', `piece-style-${style}`);
  node.dataset.piece = `${color}${type}`;
  node.dataset.pieceStyle = style;
  node.textContent = pieceText(style, color, type);
}
