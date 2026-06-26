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

function scoreToMarker(score) {
  if (Math.abs(score) >= 29000) return score > 0 ? 97 : 3;
  return Math.max(3, Math.min(97, 50 + 46 * Math.tanh(Number(score || 0) / 650)));
}

function evaluationLabel(line) {
  if (!line) return '—';
  if (line.mateVerified) return `${line.scoreText} · proven`;
  return line.scoreText || '0.00';
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
    this.evalText = root.querySelector('[data-ai-eval]');
    this.balance = root.querySelector('[data-ai-balance]');
    this.marker = root.querySelector('[data-ai-marker]');
    this.lines = root.querySelector('[data-ai-lines]');
    this.mobileFill = document.querySelector('[data-mobile-eval-fill]');
    this.mobileMarker = document.querySelector('[data-mobile-eval-marker]');
    this.mobileLabel = document.querySelector('[data-mobile-eval-label]');
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
    } else if (state === 'probing') label = 'Checking tablebase';
    else if (state === 'paused') label = completedDepth ? `Paused · d${completedDepth}` : 'Paused';
    else if (state === 'stopped') label = completedDepth ? `Stopped · d${completedDepth}` : 'Stopped';
    else if (state === 'complete') label = 'Solved line';
    this.state.textContent = label;
    this.state.className = `analysis-state ${['thinking', 'paused'].includes(state) ? state : ''}`;
    if (engine) this.engine.textContent = engine;
  }

  resetEvaluation(text = '—') {
    this.evalText.textContent = text;
    this.balance.style.width = '50%';
    this.marker.style.left = '50%';
    this.marker.dataset.score = text;
    if (this.mobileFill) this.mobileFill.style.height = '50%';
    if (this.mobileMarker) this.mobileMarker.style.bottom = '50%';
    if (this.mobileLabel) this.mobileLabel.textContent = text === '—' || text === '…' ? text : '0.0';
  }

  renderSearching() {
    this.setState('thinking', '', { searchDepth: 1 });
    this.depth.textContent = '0/0 → d1';
    this.nodes.textContent = '0';
    this.nps.textContent = '0';
    this.hash.textContent = '0%';
    if (this.cache) this.cache.textContent = 'Fresh';
    this.resetEvaluation('…');
    this.lines.innerHTML = '<div class="analysis-placeholder"><strong>Starting the local engine…</strong><span>Completed depths and principal variations will stream here.</span></div>';
  }

  renderIdle() {
    this.setState('idle');
    this.depth.textContent = '—';
    this.nodes.textContent = '—';
    this.nps.textContent = '—';
    this.hash.textContent = '—';
    if (this.cache) this.cache.textContent = '—';
    this.resetEvaluation('0.00');
    this.lines.innerHTML = '<div class="analysis-placeholder"><strong>Analysis is off</strong><span>Start the local engine to receive principal variations.</span></div>';
  }

  renderStopped(result, formattedLines = []) {
    this.render(result, formattedLines, { state: 'stopped' });
  }

  render(result, formattedLines = [], { state = '' } = {}) {
    const searchDepth = Number(result.searchDepth || result.attemptedDepth || 0);
    this.setState(state || (result.terminal || result.endgameProof || result.tablebase || result.fortressProof ? 'complete' : 'thinking'), result.engineLabel || result.engine, {
      depth: result.depth,
      searchDepth
    });
    if (result.tablebase) this.state.textContent = result.tablebaseWdl === 0 ? 'Tablebase draw' : 'Tablebase result';
    else if (result.fortressProof) this.state.textContent = 'Draw proof';
    const completed = result.completed !== false;
    const retryMark = completed ? '' : ' ↻';
    const nextMark = !result.terminal && !result.endgameProof && !result.tablebase && !result.fortressProof && searchDepth > Number(result.depth || 0) ? ` → d${searchDepth}` : '';
    if (result.tablebase) {
      this.depth.textContent = 'TB';
      this.depth.title = `${result.tablebaseSource || 'Gardner tablebase'} · ${result.tablebaseSignature || ''}`;
      this.nodes.textContent = '—';
      this.nps.textContent = '—';
      this.hash.textContent = '—';
    } else if (result.fortressProof) {
      this.depth.textContent = 'Draw proof';
      this.depth.title = `Closed-position proof over ${formatNumber(result.fortressNodes || 0)} states`;
      this.nodes.textContent = formatNumber(result.fortressNodes || result.nodes);
      this.nps.textContent = '—';
      this.hash.textContent = '—';
    } else {
      this.depth.textContent = `${result.depth || 0}/${result.selDepth || 0}${nextMark}${retryMark}`;
      this.depth.title = completed
        ? `Completed depth ${result.depth || 0}; selective depth ${result.selDepth || 0}`
        : `Depth ${searchDepth || result.attemptedDepth || 0} is being retried with a larger time slice`;
      this.nodes.textContent = formatNumber(result.nodes);
      this.nps.textContent = formatNumber(result.nps);
      this.hash.textContent = `${Math.round((result.hashfull || 0) / 10)}%`;
    }
    if (this.cache) this.cache.textContent = result.tablebase
      ? (result.tablebaseSource === 'exact-core' ? 'Exact TB' : 'Practical TB')
      : result.fortressProof ? 'Fortress'
        : result.cached ? 'Resumed'
          : result.endgameProof ? 'DTM proof' : 'Live';

    const best = result.lines?.[0];
    if (best) {
      const midpoint = scoreToMarker(best.score);
      this.balance.style.width = `${midpoint}%`;
      this.marker.style.left = `${midpoint}%`;
      this.marker.dataset.score = best.scoreText;
      this.evalText.textContent = result.tablebase && result.tablebaseWdl === 0 ? 'Draw · tablebase' : result.fortressProof ? 'Draw · closed position' : evaluationLabel(best);
      if (this.mobileFill) this.mobileFill.style.height = `${midpoint}%`;
      if (this.mobileMarker) this.mobileMarker.style.bottom = `${midpoint}%`;
      if (this.mobileLabel) this.mobileLabel.textContent = best.mateVerified ? best.scoreText : (Number(best.score || 0) / 100).toFixed(1);
    } else this.resetEvaluation('—');

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
      const proof = line.tablebase
        ? `<span class="analysis-proof">${line.tablebaseWdl === 0 ? 'TB draw' : line.dtmUpperBound ? 'TB bound' : 'Exact TB'}</span>`
        : line.fortressProof
          ? '<span class="analysis-proof">Fortress draw</span>'
          : line.endgameProof
        ? `<span class="analysis-proof">DTM ${Math.max(1, line.dtm || 1)} ply</span>`
        : line.mateVerified
          ? '<span class="analysis-proof">Verified mate</span>'
          : '';
      item.innerHTML = `
        <span class="analysis-rank">${index + 1}</span>
        <span class="analysis-main">
          <span class="analysis-move-row">
            <strong>${safeText(line.firstSan || line.move)}</strong>
            <span class="analysis-score">${safeText(line.scoreText)}</span>
            ${proof}
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
