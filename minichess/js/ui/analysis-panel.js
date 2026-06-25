function formatNumber(value) {
  const number = Number(value || 0);
  if (number >= 1_000_000) return `${(number / 1_000_000).toFixed(number >= 10_000_000 ? 0 : 1)}M`;
  if (number >= 1_000) return `${(number / 1_000).toFixed(number >= 100_000 ? 0 : 1)}k`;
  return String(number);
}

function safeText(value) {
  return String(value ?? '').replace(/[&<>"']/g, character => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
  }[character]));
}

export class AnalysisPanelView {
  constructor(root, { onSelectMove } = {}) {
    this.root = root;
    this.onSelectMove = onSelectMove;
    this.state = root.querySelector('[data-ai-state]');
    this.engine = root.querySelector('[data-ai-engine]');
    this.depth = root.querySelector('[data-ai-depth]');
    this.nodes = root.querySelector('[data-ai-nodes]');
    this.nps = root.querySelector('[data-ai-nps]');
    this.hash = root.querySelector('[data-ai-hash]');
    this.cache = root.querySelector('[data-ai-cache]');
    this.whiteLabel = root.querySelector('[data-ai-white]');
    this.drawLabel = root.querySelector('[data-ai-draw]');
    this.blackLabel = root.querySelector('[data-ai-black]');
    this.balance = root.querySelector('[data-ai-balance]');
    this.marker = root.querySelector('[data-ai-marker]');
    this.lines = root.querySelector('[data-ai-lines]');
    this.root.classList.add('analysis-disabled');
  }

  setEnabled(enabled, preserve = false) {
    this.root.classList.toggle('analysis-disabled', !enabled);
    this.root.classList.toggle('analysis-enabled', enabled);
    if (!enabled && !preserve) this.renderIdle();
  }

  setState(state, engine = '', details = {}) {
    const nextDepth = Number(details.searchDepth || 0);
    const completedDepth = Number(details.depth || 0);
    let label = 'Idle';
    if (state === 'thinking') {
      label = nextDepth
        ? `Thinking · d${nextDepth}`
        : completedDepth
          ? `Thinking · d${completedDepth + 1}`
          : 'Thinking';
    } else if (state === 'paused') label = completedDepth ? `Paused · d${completedDepth}` : 'Paused';
    else if (state === 'stopped') label = completedDepth ? `Stopped · d${completedDepth}` : 'Stopped';
    else if (state === 'complete') label = 'Solved line';
    this.state.textContent = label;
    this.state.className = `analysis-state ${['thinking', 'paused'].includes(state) ? state : ''}`;
    if (engine) this.engine.textContent = engine;
  }

  renderSearching() {
    this.setState('thinking', '', { searchDepth: 1 });
    this.depth.textContent = '0/0 → d1';
    this.nodes.textContent = '0';
    this.nps.textContent = '0';
    this.hash.textContent = '0%';
    if (this.cache) this.cache.textContent = 'Fresh';
    this.whiteLabel.textContent = 'White —';
    this.drawLabel.textContent = 'Draw —';
    this.blackLabel.textContent = 'Black —';
    this.balance.style.width = '50%';
    this.marker.style.left = '50%';
    this.marker.dataset.score = '…';
    this.lines.innerHTML = '<div class="analysis-placeholder"><strong>Starting the local engine…</strong><span>Completed depths and principal variations will stream here.</span></div>';
  }

  renderIdle() {
    this.setState('idle');
    this.depth.textContent = '—';
    this.nodes.textContent = '—';
    this.nps.textContent = '—';
    this.hash.textContent = '—';
    if (this.cache) this.cache.textContent = '—';
    this.whiteLabel.textContent = 'White —';
    this.drawLabel.textContent = 'Draw —';
    this.blackLabel.textContent = 'Black —';
    this.balance.style.width = '50%';
    this.marker.style.left = '50%';
    this.marker.dataset.score = '0.00';
    this.lines.innerHTML = '<div class="analysis-placeholder"><strong>Analysis is off</strong><span>Start the local engine to receive three principal variations.</span></div>';
  }

  renderStopped(result, formattedLines = []) {
    this.render(result, formattedLines, { state: 'stopped' });
  }

  render(result, formattedLines = [], { state = '' } = {}) {
    const searchDepth = Number(result.searchDepth || result.attemptedDepth || 0);
    this.setState(state || (result.terminal ? 'complete' : 'thinking'), result.engine, {
      depth: result.depth,
      searchDepth
    });
    const completed = result.completed !== false;
    const retryMark = completed ? '' : ' ↻';
    const nextMark = !result.terminal && searchDepth > Number(result.depth || 0) ? ` → d${searchDepth}` : '';
    this.depth.textContent = `${result.depth || 0}/${result.selDepth || 0}${nextMark}${retryMark}`;
    this.depth.title = completed
      ? `Completed depth ${result.depth || 0}; selective depth ${result.selDepth || 0}`
      : `Depth ${searchDepth || result.attemptedDepth || 0} is being retried with a larger time slice`;
    this.nodes.textContent = formatNumber(result.nodes);
    this.nps.textContent = formatNumber(result.nps);
    this.hash.textContent = `${Math.round((result.hashfull || 0) / 10)}%`;
    if (this.cache) this.cache.textContent = result.cached ? 'Resumed' : 'Live';

    const best = result.lines?.[0];
    if (best) {
      const { win, draw, loss } = best.wdl;
      this.whiteLabel.textContent = `White ${win}%`;
      this.drawLabel.textContent = `Draw ${draw}%`;
      this.blackLabel.textContent = `Black ${loss}%`;
      const midpoint = Math.max(1, Math.min(99, win + draw / 2));
      this.balance.style.width = `${midpoint}%`;
      this.marker.style.left = `${midpoint}%`;
      this.marker.dataset.score = best.scoreText;
    }

    if (!formattedLines.length) {
      this.lines.innerHTML = '<div class="analysis-placeholder"><strong>Searching…</strong><span>The first complete depth will appear here.</span></div>';
      return;
    }
    this.lines.innerHTML = '';
    formattedLines.forEach((line, index) => {
      const item = document.createElement('button');
      item.type = 'button';
      item.className = `analysis-line ${index === 0 ? 'best' : ''}`;
      item.dataset.move = line.move || '';
      item.title = `Play ${line.firstSan || line.move}`;
      item.innerHTML = `
        <span class="analysis-rank">${index + 1}</span>
        <span class="analysis-main">
          <span class="analysis-move-row">
            <strong>${safeText(line.firstSan || line.move)}</strong>
            <span class="analysis-score">${safeText(line.scoreText)}</span>
            <span class="analysis-wdl">W ${line.wdl.win} · D ${line.wdl.draw} · B ${line.wdl.loss}</span>
          </span>
          <span class="analysis-pv" title="${safeText(line.pvSan || line.pv.join(' '))}">${safeText(line.pvSan || line.pv.join(' '))}</span>
        </span>
        <span class="analysis-play" aria-hidden="true">Play ›</span>`;
      item.addEventListener('click', () => this.onSelectMove?.(line.move, line));
      this.lines.appendChild(item);
    });
  }

  renderError(message) {
    this.setState('idle');
    this.lines.innerHTML = `<div class="analysis-placeholder error"><strong>Engine unavailable</strong><span>${safeText(message)}</span></div>`;
  }
}
