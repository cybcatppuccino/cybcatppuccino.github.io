// Public result contract for v23.2.
// Mate-like scores are trusted and published directly. WDL-only tablebase bounds remain internal.

export const RESULT_CONTRACT_KIND = Object.freeze({
  TABLEBASE_ROOT: 'tablebase-root',
  MATE: 'mate',
  EVALUATION: 'evaluation',
  TERMINAL: 'terminal',
  LIVE: 'live',
  EMPTY: 'empty'
});

const MATE_DISPLAY_THRESHOLD = 29_000;
const TB_WIN_SCORE = 22_000;
const TB_BOUND_MAX_PLY = 64;
const TB_WIN_MIN_SCORE = TB_WIN_SCORE - TB_BOUND_MAX_PLY;
const finite = value => Number.isFinite(Number(value));
const evalText = score => `${Number(score) >= 0 ? '+' : ''}${(Number(score) / 100).toFixed(2)}`;
function isInternalTablebaseScore(score) {
  const value = Math.abs(Number(score));
  return Number.isFinite(value) && value >= TB_WIN_MIN_SCORE && value <= TB_WIN_SCORE;
}
function isInternalTablebaseLine(line = {}) {
  return Boolean(line?.tablebaseBound) || isInternalTablebaseScore(line?.score);
}

export function contractKindForLine(line = {}, result = {}) {
  if (result?.terminal && !line?.move) return RESULT_CONTRACT_KIND.TERMINAL;
  if (isInternalTablebaseLine(line)) return RESULT_CONTRACT_KIND.EMPTY;
  if (line?.tablebase && (line?.tablebaseRoot || result?.tablebaseRoot) && line?.tablebaseExactDtm) {
    return RESULT_CONTRACT_KIND.TABLEBASE_ROOT;
  }
  if (Math.abs(Number(line?.score || 0)) >= MATE_DISPLAY_THRESHOLD) {
    return RESULT_CONTRACT_KIND.MATE;
  }
  if (result?.liveProgress || line?.liveUpdate) return RESULT_CONTRACT_KIND.LIVE;
  return finite(line?.score) ? RESULT_CONTRACT_KIND.EVALUATION : RESULT_CONTRACT_KIND.EMPTY;
}

export function canDisplayMateIn(line = {}, result = {}) {
  const kind = contractKindForLine(line, result);
  return kind === RESULT_CONTRACT_KIND.MATE
    || (kind === RESULT_CONTRACT_KIND.TABLEBASE_ROOT && Math.sign(Number(line?.tablebaseWdl || 0)) !== 0);
}

export function canDisplayMateBound() { return false; }

export function canDisplayAsSolved(line = {}, result = {}) {
  const kind = contractKindForLine(line, result);
  return kind === RESULT_CONTRACT_KIND.TABLEBASE_ROOT
    || kind === RESULT_CONTRACT_KIND.MATE
    || kind === RESULT_CONTRACT_KIND.TERMINAL;
}

export function normalizeLineContract(line = {}, result = {}) {
  if (!line || typeof line !== 'object') return line;
  const score = Number(line.score || 0);
  let next = { ...line };
  if (isInternalTablebaseLine(next)) {
    return {
      ...next,
      scoreText: '',
      scoreKind: 'tablebase-wdl',
      scoreNumeric: false,
      tablebaseBound: true,
      mateVerified: false,
      dtm: 0,
      resultContract: RESULT_CONTRACT_KIND.EMPTY,
      resultKindV2: RESULT_CONTRACT_KIND.EMPTY
    };
  }
  const mateLike = Math.abs(score) >= MATE_DISPLAY_THRESHOLD || String(next.scoreText || '').includes('#');
  if (mateLike) {
    next = {
      ...next,
      scoreKind: 'mate',
      scoreNumeric: true,
      mateVerified: true,
      mateRejected: false,
      dtm: Math.max(1, Number(next.dtm || (MATE_DISPLAY_THRESHOLD ? 30000 - Math.abs(score) : 0)) || 1)
    };
  }
  const kind = contractKindForLine(next, result);
  return { ...next, resultContract: kind, resultKindV2: kind };
}

export function normalizeResultContract(result) {
  if (!result || typeof result !== 'object') return result;
  const lines = (Array.isArray(result.lines) ? result.lines : []).map(line => normalizeLineContract(line, result));
  const first = lines[0];
  const kind = first ? contractKindForLine(first, result) : contractKindForLine({}, result);
  const solved = Boolean(result.solved && first && canDisplayAsSolved(first, result));
  return { ...result, lines, resultContract: kind, resultKindV2: kind, solved };
}

export function isPublishableLine(line = {}, result = {}) {
  if (isInternalTablebaseLine(line)) return false;
  const kind = contractKindForLine(line, result);
  return kind !== RESULT_CONTRACT_KIND.EMPTY && line?.scoreNumeric !== false && finite(line?.score);
}
