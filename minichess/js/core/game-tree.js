import { Position } from './position.js';
import { moveKey, moveToSAN } from './notation.js';

let nextId = 1;

export class GameNode {
  constructor({ parent = null, position, move = null, san = '', source = 'local', comment = '' }) {
    this.id = `node-${nextId++}`;
    this.parent = parent;
    this.position = position;
    this.move = move;
    this.san = san;
    this.source = source;
    this.comment = comment;
    this.children = [];
    this.preferredChildId = null;
    this.ply = parent ? parent.ply + 1 : 0;
  }
}

export class GameTree {
  constructor(position = Position.initial()) {
    this.root = new GameNode({ position: position.clone() });
    this.current = this.root;
  }

  reset(position = Position.initial()) {
    this.root = new GameNode({ position: position.clone() });
    this.current = this.root;
  }

  play(move, source = 'local') {
    const key = moveKey(move);
    let child = this.current.children.find(node => node.move && moveKey(node.move) === key);
    if (!child) {
      const san = moveToSAN(this.current.position, move);
      child = new GameNode({
        parent: this.current,
        position: this.current.position.makeMove(move),
        move: { ...move },
        san,
        source
      });
      this.current.children.push(child);
    }
    this.current.preferredChildId = child.id;
    this.current = child;
    return child;
  }

  undo() {
    if (!this.current.parent) return null;
    this.current.parent.preferredChildId = this.current.id;
    this.current = this.current.parent;
    return this.current;
  }

  redo() {
    if (!this.current.children.length) return null;
    const next = this.current.children.find(c => c.id === this.current.preferredChildId) || this.current.children[0];
    this.current = next;
    return this.current;
  }

  navigate(node) {
    if (!node) return;
    this.current = node;
    let child = node;
    while (child.parent) {
      child.parent.preferredChildId = child.id;
      child = child.parent;
    }
  }

  currentPath() {
    const nodes = [];
    let cursor = this.current;
    while (cursor && cursor.parent) {
      nodes.unshift(cursor);
      cursor = cursor.parent;
    }
    return nodes;
  }

  repetitionCount(position = this.current.position) {
    const key = position.canonicalKey();
    let count = 0;
    let cursor = this.current;
    while (cursor) {
      if (cursor.position.canonicalKey() === key) count += 1;
      cursor = cursor.parent;
    }
    return count;
  }

  importPath(nodes, targetNode) {
    if (!nodes?.length) return;
    this.reset(nodes[0].position);
    let local = this.root;
    for (let i = 1; i < nodes.length; i += 1) {
      const source = nodes[i];
      const child = new GameNode({
        parent: local,
        position: source.position.clone(),
        move: source.move ? { ...source.move } : null,
        san: source.san,
        source: source.source || 'library',
        comment: source.comment || ''
      });
      local.children.push(child);
      local.preferredChildId = child.id;
      local = child;
      if (source === targetNode) this.current = child;
    }
    if (!this.current || this.current === this.root) this.current = local;
  }
}
