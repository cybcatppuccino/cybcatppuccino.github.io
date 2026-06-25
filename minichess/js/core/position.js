import {
  BOARD_SIZE,
  COLORS,
  INITIAL_COMPACT_FEN,
  INITIAL_STUDY_FEN,
  TYPES,
  fileOf,
  rankOf,
  square
} from './constants.js';

function piece(color, type) {
  return { color, type };
}

export class Position {
  constructor({ board = Array(25).fill(null), turn = COLORS.WHITE, halfmove = 0, fullmove = 1 } = {}) {
    this.board = board;
    this.turn = turn;
    this.halfmove = halfmove;
    this.fullmove = fullmove;
  }

  static initial() {
    return Position.fromFEN(INITIAL_COMPACT_FEN);
  }

  static empty(turn = COLORS.WHITE) {
    return new Position({ turn });
  }

  static fromFEN(fen) {
    const text = String(fen || '').trim();
    if (!text) throw new Error('FEN is empty.');
    const [placement, active = 'w', , , halfmove = '0', fullmove = '1'] = text.split(/\s+/);
    const rows = placement.split('/');
    let boardRows;

    if (rows.length === 5) {
      boardRows = rows;
    } else if (rows.length === 8) {
      // Gardnerfish/Stockfish study format: playable area b2–f6.
      const expandedRows = rows.map(row => expandFenRow(row, 8));
      if ([0, 1, 7].some(index => expandedRows[index].some(Boolean))) {
        throw new Error('Pieces outside the Gardner 5×5 area are not supported.');
      }
      boardRows = expandedRows.slice(2, 7).map((expanded, index) => {
        const playable = expanded.slice(1, 6);
        const outside = [...expanded.slice(0, 1), ...expanded.slice(6)];
        if (outside.some(Boolean)) throw new Error(`A piece is outside the playable area on study row ${index + 1}.`);
        return compressFenRow(playable);
      });
    } else {
      throw new Error('Gardner FEN must contain either 5 ranks or the study-compatible 8 ranks.');
    }

    const board = Array(25).fill(null);
    boardRows.forEach((row, rowIndex) => {
      const expanded = expandFenRow(row, BOARD_SIZE);
      const rank = BOARD_SIZE - 1 - rowIndex;
      expanded.forEach((symbol, file) => {
        if (!symbol) return;
        const lower = symbol.toLowerCase();
        if (!'pnbrqk'.includes(lower)) throw new Error(`Unknown piece symbol: ${symbol}`);
        board[square(file, rank)] = piece(symbol === lower ? COLORS.BLACK : COLORS.WHITE, lower);
      });
    });

    if (!['w', 'b'].includes(active)) throw new Error('Active color must be w or b.');
    return new Position({
      board,
      turn: active,
      halfmove: Number.isFinite(Number(halfmove)) ? Number(halfmove) : 0,
      fullmove: Number.isFinite(Number(fullmove)) ? Number(fullmove) : 1
    });
  }

  clone() {
    return new Position({
      board: this.board.map(p => (p ? { ...p } : null)),
      turn: this.turn,
      halfmove: this.halfmove,
      fullmove: this.fullmove
    });
  }

  pieceAt(sq) {
    return this.board[sq] || null;
  }

  setPiece(sq, value) {
    this.board[sq] = value ? { ...value } : null;
  }

  kingSquare(color) {
    return this.board.findIndex(p => p && p.color === color && p.type === TYPES.KING);
  }

  toCompactFEN() {
    const ranks = [];
    for (let rank = BOARD_SIZE - 1; rank >= 0; rank -= 1) {
      const cells = [];
      for (let file = 0; file < BOARD_SIZE; file += 1) {
        const p = this.board[square(file, rank)];
        cells.push(p ? (p.color === COLORS.WHITE ? p.type.toUpperCase() : p.type) : null);
      }
      ranks.push(compressFenRow(cells));
    }
    return `${ranks.join('/')} ${this.turn} - - ${this.halfmove} ${this.fullmove}`;
  }

  toStudyFEN() {
    const internalRows = this.toCompactFEN().split(' ')[0].split('/');
    const padded = internalRows.map(row => {
      const cells = expandFenRow(row, BOARD_SIZE);
      return compressFenRow([null, ...cells, null, null]);
    });
    return `8/8/${padded.join('/')} /8`.replace(' /8', '/8') + ` ${this.turn} - - ${this.halfmove} ${this.fullmove}`;
  }

  canonicalKey() {
    return `${this.toCompactFEN().split(' ').slice(0, 2).join(' ')}`;
  }

  makeMove(move) {
    const next = this.clone();
    const moving = next.board[move.from];
    if (!moving) throw new Error('No piece on the move origin.');
    next.board[move.to] = { ...moving, type: move.promotion || moving.type };
    next.board[move.from] = null;
    next.halfmove = moving.type === TYPES.PAWN || move.captured ? 0 : next.halfmove + 1;
    if (moving.color === COLORS.BLACK) next.fullmove += 1;
    next.turn = moving.color === COLORS.WHITE ? COLORS.BLACK : COLORS.WHITE;
    return next;
  }
}

export function expandFenRow(row, width) {
  const cells = [];
  for (const char of row) {
    if (/\d/.test(char)) {
      const count = Number(char);
      for (let i = 0; i < count; i += 1) cells.push(null);
    } else {
      cells.push(char);
    }
  }
  if (cells.length !== width) throw new Error(`FEN row “${row}” expands to ${cells.length}, expected ${width}.`);
  return cells;
}

export function compressFenRow(cells) {
  let text = '';
  let empties = 0;
  for (const value of cells) {
    if (!value) {
      empties += 1;
    } else {
      if (empties) text += String(empties);
      empties = 0;
      text += value;
    }
  }
  if (empties) text += String(empties);
  return text;
}

export function validateEditedPosition(position, rules) {
  const errors = [];
  const whiteKings = position.board.filter(p => p?.color === COLORS.WHITE && p.type === TYPES.KING).length;
  const blackKings = position.board.filter(p => p?.color === COLORS.BLACK && p.type === TYPES.KING).length;
  if (whiteKings !== 1) errors.push('Place exactly one White king.');
  if (blackKings !== 1) errors.push('Place exactly one Black king.');

  position.board.forEach((p, sq) => {
    if (p?.type === TYPES.PAWN && (rankOf(sq) === 0 || rankOf(sq) === BOARD_SIZE - 1)) {
      errors.push(`A pawn cannot remain on the promotion rank (${fileOf(sq) + 1}, ${rankOf(sq) + 1}).`);
    }
  });

  if (!errors.length) {
    const wk = position.kingSquare(COLORS.WHITE);
    const bk = position.kingSquare(COLORS.BLACK);
    const df = Math.abs(fileOf(wk) - fileOf(bk));
    const dr = Math.abs(rankOf(wk) - rankOf(bk));
    if (df <= 1 && dr <= 1) errors.push('The kings may not stand next to each other.');

    const nonMoving = position.turn === COLORS.WHITE ? COLORS.BLACK : COLORS.WHITE;
    if (rules.isInCheck(position, nonMoving)) {
      errors.push('The side that just moved may not have left its own king in check.');
    }
  }
  return [...new Set(errors)];
}

export { INITIAL_STUDY_FEN };
