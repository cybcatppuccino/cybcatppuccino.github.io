export class MoveListView {
  constructor(element, onNavigate) {
    this.element = element;
    this.onNavigate = onNavigate;
    this.element.addEventListener('click', event => {
      const button = event.target.closest('[data-node-id]');
      if (button) this.onNavigate(button.dataset.nodeId);
    });
  }

  render(tree) {
    if (!tree.root.children.length) {
      this.element.className = 'move-tree empty-state';
      this.element.innerHTML = '<div class="empty-illustration">♟</div><strong>No moves yet</strong><span>Play a move or open a node from the study tree.</span>';
      return;
    }

    this.element.className = 'move-tree';
    this.element.innerHTML = '';
    const preferred = tree.root.children.find(c => c.id === tree.root.preferredChildId) || tree.root.children[0];
    this.renderLine(preferred, this.element, tree.current, 0);

    tree.root.children.filter(c => c !== preferred).forEach(child => {
      const branch = this.makeVariationContainer('Opening alternative');
      this.renderLine(child, branch, tree.current, 1);
      this.element.appendChild(branch);
    });
    const currentButton = this.element.querySelector('.move-entry.current');
    if (currentButton) {
      const top = currentButton.offsetTop - this.element.clientHeight / 2 + currentButton.offsetHeight / 2;
      this.element.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
    }
  }

  renderLine(startNode, container, current, depth) {
    const line = document.createElement('div');
    line.className = 'move-line';
    container.appendChild(line);

    let node = startNode;
    while (node) {
      this.appendMove(line, node, current);

      if (node.children.length > 1) {
        const preferred = node.children.find(c => c.id === node.preferredChildId) || node.children[0];
        node.children.filter(c => c !== preferred).forEach(child => {
          const branch = this.makeVariationContainer('Branch');
          this.renderLine(child, branch, current, depth + 1);
          container.appendChild(branch);
        });
        node = preferred;
      } else {
        node = node.children[0] || null;
      }
    }
  }

  appendMove(line, node, current) {
    const moveNo = Math.ceil(node.ply / 2);
    const number = document.createElement('span');
    number.className = 'move-number';
    number.textContent = node.ply % 2 === 1 ? `${moveNo}.` : `${moveNo}…`;
    line.appendChild(number);

    const button = document.createElement('button');
    button.type = 'button';
    button.className = `move-entry${node === current ? ' current' : ''}`;
    button.dataset.nodeId = node.id;
    button.textContent = node.san || '…';
    button.title = node.comment || node.position.toStandardFEN();
    line.appendChild(button);
  }

  makeVariationContainer(labelText) {
    const branch = document.createElement('div');
    branch.className = 'variation';
    const label = document.createElement('div');
    label.className = 'variation-label';
    label.textContent = labelText;
    branch.appendChild(label);
    return branch;
  }
}
