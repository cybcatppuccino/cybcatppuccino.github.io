const SVG_NS = 'http://www.w3.org/2000/svg';
const PIECE_COLORS = Object.freeze({
  p: '#4fd1b3', n: '#9a7cff', b: '#f4b95f', r: '#ff7183', q: '#ed78d8', k: '#6ca7ff'
});

function svg(tag, attrs = {}) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs).forEach(([key, value]) => node.setAttribute(key, String(value)));
  return node;
}

function isAncestor(candidate, node) {
  let cursor = node;
  while (cursor) {
    if (cursor === candidate) return true;
    cursor = cursor.parent;
  }
  return false;
}

export class StudyTreeView {
  constructor(svgElement, emptyElement, onNodeClick) {
    this.svg = svgElement;
    this.viewport = svgElement.parentElement;
    this.empty = emptyElement;
    this.onNodeClick = onNodeClick;
    this.nodeMap = new Map();
    this.svg.addEventListener('click', event => {
      const group = event.target.closest('.tree-node');
      if (!group) return;
      const node = this.nodeMap.get(group.dataset.nodeId);
      if (node) this.onNodeClick(node);
    });
  }

  clear(message = 'No study nodes are available for this position.') {
    this.svg.innerHTML = '';
    this.svg.removeAttribute('viewBox');
    this.empty.classList.remove('hidden');
    this.empty.innerHTML = `<strong>${message}</strong><span>Choose another source or play a move that appears in the archive.</span>`;
  }

  render(studyRoot, anchor, currentPositionKey) {
    if (!studyRoot || !anchor) {
      this.clear();
      return;
    }
    this.empty.classList.add('hidden');
    this.svg.innerHTML = '';
    this.nodeMap.clear();

    let start = anchor;
    for (let i = 0; i < 2 && start.parent; i += 1) start = start.parent;
    const maxDepth = start.ply + 5;
    const maxNodes = 86;
    let count = 0;

    const build = node => {
      if (count >= maxNodes) return null;
      count += 1;
      const item = { node, children: [], depth: node.ply - start.ply, x: 0, y: 0 };
      if (node.ply >= maxDepth) return item;

      let children = [...node.children];
      const pathChild = children.find(child => isAncestor(child, anchor));
      if (pathChild) children = [pathChild, ...children.filter(child => child !== pathChild)];
      if (children.length > 10) children = children.slice(0, 10);
      for (const child of children) {
        const built = build(child);
        if (built) item.children.push(built);
        if (count >= maxNodes) break;
      }
      return item;
    };

    const visibleRoot = build(start);
    if (!visibleRoot) {
      this.clear();
      return;
    }

    const xGap = 138;
    const yGap = 44;
    const paddingX = 52;
    const paddingY = 38;
    let leafIndex = 0;
    let maxVisibleDepth = 0;

    const layout = item => {
      maxVisibleDepth = Math.max(maxVisibleDepth, item.depth);
      item.x = paddingX + item.depth * xGap;
      if (!item.children.length) {
        item.y = paddingY + leafIndex * yGap;
        leafIndex += 1;
      } else {
        item.children.forEach(layout);
        item.y = item.children.reduce((sum, child) => sum + child.y, 0) / item.children.length;
      }
    };
    layout(visibleRoot);

    const height = Math.max(520, paddingY * 2 + Math.max(1, leafIndex - 1) * yGap + 60);
    const width = Math.max(640, paddingX * 2 + maxVisibleDepth * xGap + 190);
    this.svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
    this.svg.setAttribute('width', width);
    this.svg.setAttribute('height', height);

    const edgeLayer = svg('g', { class: 'edge-layer' });
    const nodeLayer = svg('g', { class: 'node-layer' });
    this.svg.append(edgeLayer, nodeLayer);

    const draw = item => {
      for (const child of item.children) {
        const path = svg('path', {
          class: 'tree-edge',
          d: `M ${item.x} ${item.y} C ${item.x + 50} ${item.y}, ${child.x - 50} ${child.y}, ${child.x} ${child.y}`,
          stroke: PIECE_COLORS[child.node.move?.piece] || '#64748b'
        });
        edgeLayer.appendChild(path);
        draw(child);
      }

      const node = item.node;
      this.nodeMap.set(node.id, node);
      const classes = ['tree-node'];
      if (!node.parent) classes.push('root');
      if (node === anchor) classes.push('current');
      if (node.positionKey === currentPositionKey && node !== anchor) classes.push('match');
      const group = svg('g', {
        class: classes.join(' '),
        transform: `translate(${item.x}, ${item.y})`,
        'data-node-id': node.id,
        tabindex: '0',
        role: 'button'
      });
      const circle = svg('circle', { r: node === anchor ? 7.4 : 6 });
      const label = svg('text', { x: 12, y: 3 });
      const moveLabel = node.parent ? node.san : 'Start';
      label.textContent = moveLabel.length > 17 ? `${moveLabel.slice(0, 15)}…` : moveLabel;
      const title = svg('title');
      title.textContent = [
        node.parent ? `${Math.ceil(node.ply / 2)}${node.ply % 2 ? '.' : '…'} ${node.san}` : 'Starting position',
        node.comment || '',
        node.position.toStudyFEN()
      ].filter(Boolean).join('\n');
      group.append(circle, label, title);
      nodeLayer.appendChild(group);
    };
    draw(visibleRoot);

    queueMicrotask(() => this.centerCurrentNode());
  }

  centerCurrentNode() {
    const current = this.svg.querySelector('.tree-node.current');
    if (!current || !this.viewport) return;
    const transform = current.getAttribute('transform') || '';
    const match = transform.match(/translate\(([-\d.]+)[ ,]+([-\d.]+)\)/);
    if (!match) return;
    const x = Number(match[1]);
    const y = Number(match[2]);
    const maxLeft = Math.max(0, this.svg.clientWidth - this.viewport.clientWidth);
    const maxTop = Math.max(0, this.svg.clientHeight - this.viewport.clientHeight);
    const left = Math.max(0, Math.min(maxLeft, x - this.viewport.clientWidth / 2));
    const top = Math.max(0, Math.min(maxTop, y - this.viewport.clientHeight / 2));
    this.viewport.scrollTo({ left, top, behavior: 'smooth' });
  }
}
