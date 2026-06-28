function preferredChild(node) {
  if (!node?.children?.length) return null;
  return node.children.find(child => child.id === node.preferredChildId) || node.children[0] || null;
}

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
      this.element.innerHTML = '<div class="empty-illustration">♟</div><strong>No moves yet</strong><span>Play a move or choose a tree node.</span>';
      return;
    }

    this.element.className = 'move-tree';
    this.element.innerHTML = '';
    const main = preferredChild(tree.root);
    this.renderLine(main, this.element, tree.current);

    for (const child of tree.root.children) {
      if (child === main) continue;
      const branch = this.makeVariationContainer('Variation');
      this.renderLine(child, branch, tree.current);
      this.element.appendChild(branch);
    }

    const currentButton = this.element.querySelector('.move-entry.current');
    if (currentButton) {
      const top = currentButton.offsetTop - this.element.clientHeight / 2 + currentButton.offsetHeight / 2;
      this.element.scrollTo({ top: Math.max(0, top), behavior: 'smooth' });
    }
  }

  renderLine(startNode, container, current) {
    if (!startNode) return;
    const line = document.createElement('div');
    line.className = 'move-line';
    container.appendChild(line);

    let node = startNode;
    let activeTurn = null;
    let firstInLine = true;
    while (node) {
      activeTurn = this.appendMove(line, node, current, { activeTurn, firstInLine });

      const main = preferredChild(node);
      if (node.children.length > 1) {
        for (const child of node.children) {
          if (child === main) continue;
          const branch = this.makeVariationContainer('Variation');
          this.renderLine(child, branch, current);
          container.appendChild(branch);
        }
      }
      node = main;
      firstInLine = false;
    }
  }

  appendMove(line, node, current, { activeTurn, firstInLine }) {
    const parent = node.parent;
    const sideToMove = parent?.position?.turn || 'w';
    const fullmove = Math.max(1, Number(parent?.position?.fullmove || Math.ceil(node.ply / 2)));
    const shouldStartTurn = sideToMove === 'w'
      || !activeTurn
      || activeTurn.dataset.fullmove !== String(fullmove);

    let turn = activeTurn;
    if (shouldStartTurn) {
      turn = document.createElement('span');
      turn.className = 'move-turn';
      turn.dataset.fullmove = String(fullmove);
      const number = document.createElement('span');
      number.className = 'move-number';
      number.textContent = sideToMove === 'w' ? `${fullmove}.` : `${fullmove}…`;
      turn.appendChild(number);
      line.appendChild(turn);
    } else if (sideToMove === 'b' && firstInLine) {
      // A variation can begin with Black. It still needs the conventional
      // `N...` prefix even though it owns a separate visual turn container.
      const number = document.createElement('span');
      number.className = 'move-number';
      number.textContent = `${fullmove}…`;
      turn.insertBefore(number, turn.firstChild);
    }

    const button = document.createElement('button');
    button.type = 'button';
    button.className = `move-entry${node === current ? ' current' : ''}`;
    button.dataset.nodeId = node.id;
    button.textContent = node.san || '…';
    button.title = node.comment || node.position.toStandardFEN();
    turn.appendChild(button);
    return turn;
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
