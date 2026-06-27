import {
  BOARD_SIZE,
  COLORS,
  STANDARD_FILES,
  STANDARD_RANKS,
  fileOf,
  rankOf,
  square
} from '../core/constants.js';
import { applyPieceStyle, pieceText } from './pieces.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
let boardInstanceId = 0;

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

export class BoardView {
  constructor(element, options) {
    this.element = element;
    this.getPosition = options.getPosition;
    this.getLegalMoves = options.getLegalMoves;
    this.onAttemptMove = options.onAttemptMove;
    this.onEditorSquare = options.onEditorSquare;
    this.flipped = false;
    this.mirrored = false;
    this.selectedSquare = null;
    this.legalTargets = [];
    this.lastMove = null;
    this.status = { state: 'playing', check: false };
    this.editorMode = false;
    this.locked = false;
    this.drag = null;
    this.pieceStyle = 'standard';
    this.arrows = [];
    this.instanceId = ++boardInstanceId;

    this.element.addEventListener('pointerdown', event => this.onPointerDown(event));
    window.addEventListener('pointermove', event => this.onPointerMove(event));
    window.addEventListener('pointerup', event => this.onPointerUp(event));
    window.addEventListener('pointercancel', () => this.cancelDrag());
  }

  setState({ lastMove = this.lastMove, status = this.status, locked = this.locked } = {}) {
    this.lastMove = lastMove;
    this.status = status;
    this.locked = locked;
    this.render();
  }

  setArrows(arrows = [], render = true) {
    this.arrows = Array.isArray(arrows) ? arrows.filter(arrow => Number.isInteger(arrow.from) && Number.isInteger(arrow.to)) : [];
    if (render) this.render();
  }

  setPieceStyle(style = 'standard', render = true) {
    this.pieceStyle = ['standard', 'neo', 'letter'].includes(style) ? style : 'standard';
    if (render) this.render();
  }

  setEditorMode(enabled) {
    this.editorMode = enabled;
    this.clearSelection(false);
    this.render();
  }

  flip() {
    this.flipped = !this.flipped;
    this.render();
  }

  mirror() {
    this.mirrored = !this.mirrored;
    this.render();
  }

  setMirrored(mirrored, render = true) {
    this.mirrored = Boolean(mirrored);
    if (render) this.render();
  }

  setFlipped(flipped, render = true) {
    this.flipped = Boolean(flipped);
    if (render) this.render();
  }

  clearSelection(render = true) {
    this.selectedSquare = null;
    this.legalTargets = [];
    if (render) this.render();
  }

  selectSquare(sq) {
    const position = this.getPosition();
    const p = position.pieceAt(sq);
    if (!p || p.color !== position.turn || this.locked || this.editorMode) {
      this.clearSelection();
      return;
    }
    if (this.selectedSquare === sq) {
      this.clearSelection();
      return;
    }
    this.selectedSquare = sq;
    this.legalTargets = this.getLegalMoves(sq);
    this.render();
  }

  async handleSquareActivation(sq) {
    if (this.editorMode) {
      this.onEditorSquare?.(sq);
      return;
    }
    if (this.locked) return;
    const position = this.getPosition();
    const p = position.pieceAt(sq);

    if (this.selectedSquare !== null) {
      if (sq === this.selectedSquare) {
        this.clearSelection();
        return;
      }
      const candidates = this.legalTargets.filter(move => move.to === sq);
      if (candidates.length) {
        await this.onAttemptMove(this.selectedSquare, sq, candidates);
        this.clearSelection();
        return;
      }
      if (p?.color === position.turn) {
        this.selectSquare(sq);
        return;
      }
      this.clearSelection();
      return;
    }

    if (p?.color === position.turn) this.selectSquare(sq);
  }

  onPointerDown(event) {
    if (event.button !== 0) return;
    const squareElement = event.target.closest('.square');
    if (!squareElement || !this.element.contains(squareElement)) return;
    const sq = Number(squareElement.dataset.square);
    const pieceElement = event.target.closest('.piece');
    const position = this.getPosition();
    const p = position.pieceAt(sq);

    this.drag = {
      pointerId: event.pointerId,
      from: sq,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
      validPiece: Boolean(pieceElement && p && p.color === position.turn && !this.locked && !this.editorMode),
      pieceElement,
      ghost: null
    };
    squareElement.setPointerCapture?.(event.pointerId);
    event.preventDefault();
  }

  onPointerMove(event) {
    if (!this.drag || this.drag.pointerId !== event.pointerId) return;
    const distance = Math.hypot(event.clientX - this.drag.startX, event.clientY - this.drag.startY);
    if (!this.drag.moved && distance > 7 && this.drag.validPiece) {
      this.drag.moved = true;
      this.beginDragGhost(event);
      this.selectedSquare = this.drag.from;
      this.legalTargets = this.getLegalMoves(this.drag.from);
      this.render();
      this.drag.pieceElement = this.element.querySelector(`.square[data-square="${this.drag.from}"] .piece`);
      this.drag.pieceElement?.classList.add('drag-source');
    }
    if (this.drag.moved && this.drag.ghost) {
      this.drag.ghost.style.left = `${event.clientX}px`;
      this.drag.ghost.style.top = `${event.clientY}px`;
    }
  }

  async onPointerUp(event) {
    if (!this.drag || this.drag.pointerId !== event.pointerId) return;
    const drag = this.drag;
    this.drag = null;

    drag.ghost?.remove();
    drag.pieceElement?.classList.remove('drag-source');

    if (drag.moved) {
      const target = document.elementFromPoint(event.clientX, event.clientY)?.closest('.square');
      const to = target && this.element.contains(target) ? Number(target.dataset.square) : -1;
      const candidates = this.legalTargets.filter(move => move.to === to);
      if (candidates.length) await this.onAttemptMove(drag.from, to, candidates);
      this.clearSelection();
      return;
    }

    await this.handleSquareActivation(drag.from);
  }

  beginDragGhost(event) {
    const p = this.getPosition().pieceAt(this.drag.from);
    if (!p) return;
    const ghost = document.createElement('div');
    ghost.className = `drag-ghost piece ${p.color === COLORS.WHITE ? 'white' : 'black'} piece-style-${this.pieceStyle}`;
    ghost.textContent = pieceText(this.pieceStyle, p.color, p.type);
    ghost.style.left = `${event.clientX}px`;
    ghost.style.top = `${event.clientY}px`;
    document.body.appendChild(ghost);
    this.drag.ghost = ghost;
  }

  visualPoint(sq) {
    const file = fileOf(sq);
    const rank = rankOf(sq);
    return {
      x: ((this.flipped !== this.mirrored) ? 4 - file : file) + 0.5,
      y: (this.flipped ? rank : 4 - rank) + 0.5
    };
  }

  renderArrows() {
    if (!this.arrows.length || this.editorMode) return;
    const layer = svg('svg', {
      class: 'board-arrow-layer',
      viewBox: '0 0 5 5',
      preserveAspectRatio: 'none',
      'aria-hidden': 'true'
    });
    const defs = svg('defs');
    const styles = {
      book: { color: '#94a9ff', opacity: 0.24, width: 0.075, marker: 0.19 },
      engine: { color: '#58e6bd', opacity: 0.9, width: 0.105, marker: 0.25 },
      response: { color: '#ffd166', opacity: 0.78, width: 0.085, marker: 0.21 }
    };
    for (const kind of Object.keys(styles)) {
      const marker = svg('marker', {
        id: `arrow-${this.instanceId}-${kind}`,
        markerWidth: 5,
        markerHeight: 5,
        refX: 4.1,
        refY: 2.5,
        orient: 'auto',
        markerUnits: 'strokeWidth'
      });
      marker.appendChild(svg('path', { d: 'M0,0 L5,2.5 L0,5 Z', fill: styles[kind].color }));
      defs.appendChild(marker);
    }
    layer.appendChild(defs);

    this.arrows.forEach((arrow, index) => {
      if (arrow.from === arrow.to) return;
      const kind = styles[arrow.kind] ? arrow.kind : 'book';
      const style = styles[kind];
      const start = this.visualPoint(arrow.from);
      const end = this.visualPoint(arrow.to);
      const dx = end.x - start.x;
      const dy = end.y - start.y;
      const length = Math.hypot(dx, dy) || 1;
      const shortenStart = kind === 'response' ? 0.13 : 0.12;
      const shortenEnd = kind === 'book' ? 0.28 : 0.31;
      const x1 = start.x + dx / length * shortenStart;
      const y1 = start.y + dy / length * shortenStart;
      const x2 = end.x - dx / length * shortenEnd;
      const y2 = end.y - dy / length * shortenEnd;
      const line = svg('line', {
        x1, y1, x2, y2,
        class: `board-arrow board-arrow-${kind}`,
        stroke: style.color,
        'stroke-width': style.width,
        'stroke-linecap': 'round',
        opacity: arrow.opacity ?? style.opacity,
        'marker-end': `url(#arrow-${this.instanceId}-${kind})`
      });
      if (kind === 'response') line.setAttribute('stroke-dasharray', '.18 .10');
      if (arrow.title) {
        const title = svg('title');
        title.textContent = arrow.title;
        line.appendChild(title);
      }
      layer.appendChild(line);
    });
    this.element.appendChild(layer);
  }

  render() {
    const position = this.getPosition();
    this.element.innerHTML = '';
    this.element.classList.toggle('locked', this.locked);
    this.element.classList.toggle('editor-mode', this.editorMode);
    this.element.dataset.pieceStyle = this.pieceStyle;

    const displayRanks = this.flipped ? [0, 1, 2, 3, 4] : [4, 3, 2, 1, 0];
    const reverseFiles = this.flipped !== this.mirrored;
    const displayFiles = reverseFiles ? [4, 3, 2, 1, 0] : [0, 1, 2, 3, 4];
    const checkedKing = this.status.check ? position.kingSquare(position.turn) : -1;

    for (const rank of displayRanks) {
      for (const file of displayFiles) {
        const sq = square(file, rank);
        const cell = document.createElement('div');
        cell.className = `square ${(file + rank) % 2 === 0 ? 'light' : 'dark'}`;
        cell.dataset.square = String(sq);
        cell.setAttribute('role', 'button');
        cell.setAttribute('aria-label', `${STANDARD_FILES[file]}${STANDARD_RANKS[rank]}`);

        if (this.selectedSquare === sq) cell.classList.add('selected');
        if (this.lastMove && (this.lastMove.from === sq || this.lastMove.to === sq)) cell.classList.add('last-move');
        if (checkedKing === sq) cell.classList.add(this.status.state === 'checkmate' ? 'checkmate' : 'check');
        const targetMoves = this.legalTargets.filter(move => move.to === sq);
        if (targetMoves.length) {
          const capture = targetMoves.some(move => move.captured || position.pieceAt(sq));
          cell.classList.add(capture ? 'legal-capture' : 'legal-move');
        }

        const p = position.pieceAt(sq);
        if (p) {
          const pieceNode = document.createElement('span');
          applyPieceStyle(pieceNode, this.pieceStyle, p.color, p.type);
          cell.appendChild(pieceNode);
        }

        const isBottomDisplayRank = rank === displayRanks[displayRanks.length - 1];
        const isLeftDisplayFile = file === displayFiles[0];
        if (isBottomDisplayRank) {
          const fileLabel = document.createElement('span');
          fileLabel.className = 'coord-file';
          fileLabel.textContent = STANDARD_FILES[file];
          cell.appendChild(fileLabel);
        }
        if (isLeftDisplayFile) {
          const rankLabel = document.createElement('span');
          rankLabel.className = 'coord-rank';
          rankLabel.textContent = STANDARD_RANKS[rank];
          cell.appendChild(rankLabel);
        }
        this.element.appendChild(cell);
      }
    }
    this.renderArrows();
  }
}
