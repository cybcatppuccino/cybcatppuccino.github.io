import { COORD_SYSTEMS } from './constants.js';
import { Position } from './position.js';
import { findMoveBySAN, moveToSAN } from './notation.js';

let libraryId = 1;

export class LibraryNode {
  constructor({ parent = null, position, move = null, san = '', source = '', comment = '' }) {
    this.id = `lib-${libraryId++}`;
    this.parent = parent;
    this._position = position;
    // v12.2: internal node keys/fens are canonical compact A1–E5 FENs.
    this.positionFen = position.toCompactFEN();
    this.positionKey = position.canonicalKey();
    this.move = move;
    this.san = san;
    this.source = source;
    this.comment = comment;
    this.children = [];
    this.ply = parent ? parent.ply + 1 : 0;
  }

  get position() {
    return this._position || Position.fromFEN(this.positionFen);
  }

  releasePosition() {
    this._position = null;
  }
}

export function parseTags(text) {
  const tags = {};
  for (const match of text.matchAll(/^\s*\[([^\s]+)\s+"((?:\\"|[^"])*)"\]\s*$/gm)) {
    tags[match[1]] = match[2].replace(/\\"/g, '"');
  }
  return tags;
}

export function tokenizeMovetext(text) {
  const start = text.search(/(?:^|\n)\s*\d+\.(?:\.\.)?/m);
  const input = start >= 0 ? text.slice(start) : text.replace(/^\s*\[[^\n]+\]\s*$/gm, ' ');
  const tokens = [];
  let i = 0;
  while (i < input.length) {
    const ch = input[i];
    if (/\s/.test(ch)) {
      i += 1;
      continue;
    }
    if (ch === '{') {
      let end = i + 1;
      while (end < input.length && input[end] !== '}') end += 1;
      tokens.push({ type: 'comment', value: input.slice(i + 1, end).trim() });
      i = Math.min(end + 1, input.length);
      continue;
    }
    if (ch === ';') {
      let end = input.indexOf('\n', i + 1);
      if (end < 0) end = input.length;
      tokens.push({ type: 'comment', value: input.slice(i + 1, end).trim() });
      i = end;
      continue;
    }
    if (ch === '(' || ch === ')') {
      tokens.push({ type: ch, value: ch });
      i += 1;
      continue;
    }
    let end = i + 1;
    while (end < input.length && !/\s|[(){};]/.test(input[end])) end += 1;
    tokens.push({ type: 'word', value: input.slice(i, end) });
    i = end;
  }
  return tokens;
}

function isIgnorableWord(word) {
  return /^\d+\.(?:\.\.)?$/.test(word) ||
    /^\d+\.\.\.$/.test(word) ||
    /^\$\d+$/.test(word) ||
    /^(?:1-0|0-1|1\/2-1\/2|\*)$/.test(word) ||
    /^(?:[!?]+|\.\.\.|\+\-|-\+|-\/\+|\+\/=|=\+|=\.|=|±|∓|∞)$/.test(word);
}

export function parsePGN(text, sourceName = 'PGN', { coordSystem = COORD_SYSTEMS.STANDARD } = {}) {
  const tags = parseTags(text);
  const system = coordSystem === COORD_SYSTEMS.LEGACY_STUDY || coordSystem === 'legacy' ? COORD_SYSTEMS.LEGACY_STUDY : COORD_SYSTEMS.STANDARD;
  const startPosition = Position.fromFEN(tags.FEN || 'rnbqk/ppppp/5/PPPPP/RNBQK w - - 0 1');
  const root = new LibraryNode({ position: startPosition, source: sourceName });
  const tokens = tokenizeMovetext(text);
  const errors = [];
  let parsedMoves = 0;

  function parseSequence(index, baseNode, stopAtClose = false) {
    let current = baseNode;
    let lastMoveNode = null;

    while (index < tokens.length) {
      const token = tokens[index];
      if (token.type === ')') {
        return { index: index + 1, current };
      }
      if (token.type === '(') {
        const branchBase = lastMoveNode?.parent || current;
        const branch = parseSequence(index + 1, branchBase, true);
        index = branch.index;
        continue;
      }
      if (token.type === 'comment') {
        if (lastMoveNode) {
          lastMoveNode.comment = [lastMoveNode.comment, token.value].filter(Boolean).join(' ');
        }
        index += 1;
        continue;
      }

      const word = token.value;
      if (isIgnorableWord(word)) {
        index += 1;
        continue;
      }

      const move = findMoveBySAN(current.position, word, { coordSystem: system });
      if (!move) {
        errors.push({ token: word, ply: current.ply, context: current.position.toStandardFEN() });
        index += 1;
        continue;
      }

      const san = moveToSAN(current.position, move, { coordSystem: COORD_SYSTEMS.STANDARD });
      const next = new LibraryNode({
        parent: current,
        position: current.position.makeMove(move),
        move: { ...move },
        san,
        source: sourceName
      });
      current.children.push(next);
      current = next;
      lastMoveNode = next;
      parsedMoves += 1;
      index += 1;
    }

    return { index, current };
  }

  parseSequence(0, root, false);
  for (const node of flattenTree(root)) node.releasePosition();
  return { root, tags, errors, parsedMoves, sourceName };
}

export function flattenTree(root, max = Infinity) {
  const nodes = [];
  const stack = [root];
  while (stack.length && nodes.length < max) {
    const node = stack.pop();
    nodes.push(node);
    for (let i = node.children.length - 1; i >= 0; i -= 1) stack.push(node.children[i]);
  }
  return nodes;
}

export function pathToNode(node) {
  const path = [];
  let cursor = node;
  while (cursor) {
    path.unshift(cursor);
    cursor = cursor.parent;
  }
  return path;
}

export class StudyLibrary {
  constructor() {
    this.studies = [];
    this.positionIndex = new Map();
  }

  addStudy(study) {
    this.studies.push(study);
    for (const node of flattenTree(study.root)) {
      const key = node.positionKey;
      if (!this.positionIndex.has(key)) this.positionIndex.set(key, []);
      this.positionIndex.get(key).push(node);
    }
  }

  matches(position) {
    return this.positionIndex.get(position.canonicalKey()) || [];
  }

  bookMoves(position) {
    const grouped = new Map();
    for (const node of this.matches(position)) {
      for (const child of node.children) {
        if (!child.move) continue;
        const key = `${child.move.from}-${child.move.to}-${child.move.promotion || ''}`;
        if (!grouped.has(key)) {
          grouped.set(key, {
            move: { ...child.move },
            san: child.san,
            count: 0,
            sources: new Set(),
            comments: []
          });
        }
        const entry = grouped.get(key);
        entry.count += 1;
        entry.sources.add(child.source);
        if (child.comment) entry.comments.push(child.comment);
      }
    }
    return [...grouped.values()]
      .map(entry => ({ ...entry, sources: [...entry.sources] }))
      .sort((a, b) => b.count - a.count || a.san.localeCompare(b.san));
  }

  findNode(id) {
    for (const study of this.studies) {
      const node = flattenTree(study.root).find(n => n.id === id);
      if (node) return node;
    }
    return null;
  }
}

function preferredChild(node) {
  if (!node?.children?.length) return null;
  return node.children.find(child => child.id === node.preferredChildId) || node.children[0] || null;
}

function sanForNode(node) {
  if (node?.san) return node.san;
  if (node?.parent?.position && node?.move) return moveToSAN(node.parent.position, node.move, { coordSystem: COORD_SYSTEMS.STANDARD });
  return '…';
}

function movePrefix(node, firstInLine) {
  const parent = node?.parent;
  const fullmove = Math.max(1, Number(parent?.position?.fullmove || Math.ceil(Number(node?.ply || 1) / 2)));
  if (parent?.position?.turn === 'w') return `${fullmove}.`;
  return firstInLine ? `${fullmove}...` : '';
}

function tokensForNode(node, firstInLine) {
  const prefix = movePrefix(node, firstInLine);
  return [prefix, sanForNode(node)].filter(Boolean);
}

function serializeFromParent(parent, { includeVariations = false } = {}) {
  const main = preferredChild(parent);
  if (!main) return '';
  const tokens = [...tokensForNode(main, false)];
  // A recursive-annotation variation is written after the main move it
  // replaces. This keeps the exported text compatible with ordinary PGN
  // readers and with this project's own branch parser.
  if (includeVariations && parent.children?.length > 1) {
    for (const alternative of parent.children) {
      if (alternative === main) continue;
      const variation = serializeBranch(alternative, { includeVariations: true, firstInLine: true });
      if (variation) tokens.push(`(${variation})`);
    }
  }
  const tail = serializeFromParent(main, { includeVariations });
  if (tail) tokens.push(tail);
  return tokens.join(' ').replace(/\s+/g, ' ').trim();
}

function serializeBranch(startNode, { includeVariations = false, firstInLine = true } = {}) {
  if (!startNode) return '';
  const tokens = [...tokensForNode(startNode, firstInLine)];
  const tail = serializeFromParent(startNode, { includeVariations });
  if (tail) tokens.push(tail);
  return tokens.join(' ').replace(/\s+/g, ' ').trim();
}

function serializeTreeMovetext(tree) {
  const root = tree?.root;
  if (!root?.children?.length) return '';
  const main = preferredChild(root);
  if (!main) return '';
  const tokens = [...tokensForNode(main, true)];
  // Root alternatives replace the first move, so they follow that first move.
  if (root.children.length > 1) {
    for (const alternative of root.children) {
      if (alternative === main) continue;
      const variation = serializeBranch(alternative, { includeVariations: true, firstInLine: true });
      if (variation) tokens.push(`(${variation})`);
    }
  }
  const tail = serializeFromParent(main, { includeVariations: true });
  if (tail) tokens.push(tail);
  return tokens.join(' ').replace(/\s+/g, ' ').trim();
}

export function exportCurrentLineMovetext(tree, { result = '*' } = {}) {
  const nodes = typeof tree?.currentPath === 'function' ? tree.currentPath() : [];
  if (!nodes.length) return String(result || '*');
  const tokens = [];
  nodes.forEach((node, index) => {
    tokens.push(...tokensForNode(node, index === 0));
  });
  return `${tokens.join(' ').replace(/\s+/g, ' ').trim()} ${String(result || '*')}`.trim();
}

export function exportGameTreeMovetext(tree, { result = '*' } = {}) {
  const body = serializeTreeMovetext(tree);
  return body ? `${body} ${String(result || '*')}`.trim() : String(result || '*');
}


export function exportCurrentLinePGN(tree, {
  event = 'Gardner MiniChess',
  result = '*'
} = {}) {
  const rootFen = tree?.root?.position?.toStandardFEN?.() || '';
  const tags = [
    `[Event "${escapeTagValue(event)}"]`,
    '[Variant "Gardner MiniChess"]',
    '[SetUp "1"]',
    `[FEN "${escapeTagValue(rootFen)}"]`,
    `[Result "${escapeTagValue(result)}"]`
  ];
  return `${tags.join('\n')}\n\n${exportCurrentLineMovetext(tree, { result })}`;
}

function escapeTagValue(value) {
  return String(value ?? '').replace(/\\/g, '\\\\').replace(/"/g, '\\"');
}

export function exportGameTreePGN(tree, {
  event = 'Gardner MiniChess',
  result = '*'
} = {}) {
  const rootFen = tree?.root?.position?.toStandardFEN?.() || '';
  const tags = [
    `[Event "${escapeTagValue(event)}"]`,
    '[Variant "Gardner MiniChess"]',
    '[SetUp "1"]',
    `[FEN "${escapeTagValue(rootFen)}"]`,
    `[Result "${escapeTagValue(result)}"]`
  ];
  return `${tags.join('\n')}\n\n${exportGameTreeMovetext(tree, { result })}`;
}

