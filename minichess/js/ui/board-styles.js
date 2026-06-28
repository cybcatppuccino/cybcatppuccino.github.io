// Board and piece styles are deliberately independent.  These palettes mirror
// the supplied references while keeping a high-contrast accessible default.
export const BOARD_STYLES = Object.freeze([
  { id: 'standard', label: 'Standard · Syzygy' },
  { id: 'green', label: 'Green' },
  { id: 'sand', label: 'Sand' },
  { id: 'slate', label: 'Slate' },
  { id: 'sketch', label: 'Sketch' }
]);

export function isBoardStyle(style) {
  return BOARD_STYLES.some(item => item.id === style);
}
