// v21.1 public-result contract for endgame analysis.
//
// This module is intentionally dependency-free.  It is used by searchers,
// workers, caches and UI code to prevent old compatibility flags such as
// `mateVerified` from being interpreted as a complete AND/OR proof unless the
// line also carries the required proof provenance.

export const RESULT_CONTRACT_KIND = Object.freeze({
  DB_EXACT_ROOT: 'db_exact_root',
  FORCED_MATE_EXACT: 'forced_mate_exact',
  FORCED_MATE_BOUND: 'forced_mate_bound',
  DB_BRIDGE_MATE_BOUND: 'db_bridge_mate_bound',
  DB_BRIDGE_DRAW: 'db_bridge_draw',
  FORCED_DRAW: 'forced_draw',
  TERMINAL: 'terminal',
  ORDINARY_SEARCH: 'ordinary_search',
  MATE_CANDIDATE: 'mate_candidate',
  PROOF_SEED_INTERNAL: 'proof_seed_internal',
  LIVE_PROGRESS: 'live_progress',
  EMPTY: 'empty'
});

const MATE_DISPLAY_THRESHOLD = 29_000;
const FINITE_MATE_CANDIDATE_SCORE = 25_000;

function finiteNumber(value, fallback = 0) {
  const number = Number(value);
  return Number.isFinite(number) ? number : fallback;
}

function finiteEvalText(score) {
  const value = finiteNumber(score, 0) / 100;
  const sign = value > 0 ? '+' : '';
  return `${sign}${value.toFixed(2)}`;
}

function hasFullMateProof(line = {}, result = {}) {
  if (!line?.mateVerified) return false;
  if (line?.pvMateLineOnly || line?.mateCandidate || line?.unverifiedMate) return false;
  if (line?.tablebaseBridgeProof || line?.mateUpperBound) return false;
  return Boolean(
    line?.mateProof || line?.endgameProof || result?.mateProof || result?.endgameProof
  );
}

function hasMateBoundProof(line = {}, result = {}) {
  if (line?.tablebaseBridgeProof || result?.tablebaseBridgeProof) return false;
  return Boolean(
    line?.mateUpperBound
    && line?.mateVerified
    && (line?.mateProof || line?.endgameProof || result?.mateProof || result?.endgameProof)
  );
}

export function contractKindForLine(line = {}, result = {}) {
  if (!line && !result) return RESULT_CONTRACT_KIND.EMPTY;
  const explicit = String(line?.resultContract || line?.resultKindV2 || result?.resultContract || result?.resultKindV2 || '');
  if (Object.values(RESULT_CONTRACT_KIND).includes(explicit)) return explicit;
  if (result?.terminal || line?.terminal) return RESULT_CONTRACT_KIND.TERMINAL;
  if ((result?.tablebase || line?.tablebase) && (result?.tablebaseScope === 'root-exact' || line?.tablebaseScope === 'root-exact' || line?.tablebaseExactRoot)) {
    return RESULT_CONTRACT_KIND.DB_EXACT_ROOT;
  }
  if (result?.tablebaseBridgeDraw || line?.tablebaseBridgeDraw) return RESULT_CONTRACT_KIND.DB_BRIDGE_DRAW;
  if (result?.tablebaseBridgeProof || line?.tablebaseBridgeProof) return RESULT_CONTRACT_KIND.DB_BRIDGE_MATE_BOUND;
  if (line?.fortressProof || result?.fortressProof) return RESULT_CONTRACT_KIND.FORCED_DRAW;
  if (hasFullMateProof(line, result)) return RESULT_CONTRACT_KIND.FORCED_MATE_EXACT;
  if (hasMateBoundProof(line, result)) return RESULT_CONTRACT_KIND.FORCED_MATE_BOUND;
  if (line?.tablebaseBridgeCandidate || line?.scoreKind === 'tablebase-wdl-proof-seed' || line?.scoreNumeric === false) {
    return RESULT_CONTRACT_KIND.PROOF_SEED_INTERNAL;
  }
  if (line?.pvMateLineOnly || line?.mateCandidate || line?.unverifiedMate || String(line?.scoreText || '').includes('#')) {
    return RESULT_CONTRACT_KIND.MATE_CANDIDATE;
  }
  if (result?.liveProgress || line?.liveUpdate) return RESULT_CONTRACT_KIND.LIVE_PROGRESS;
  if (Number.isFinite(Number(line?.score))) return RESULT_CONTRACT_KIND.ORDINARY_SEARCH;
  return RESULT_CONTRACT_KIND.EMPTY;
}

export function canDisplayMateIn(line = {}, result = {}) {
  return contractKindForLine(line, result) === RESULT_CONTRACT_KIND.FORCED_MATE_EXACT;
}

export function canDisplayMateBound(line = {}, result = {}) {
  const kind = contractKindForLine(line, result);
  return kind === RESULT_CONTRACT_KIND.FORCED_MATE_BOUND || kind === RESULT_CONTRACT_KIND.DB_BRIDGE_MATE_BOUND;
}

export function canDisplayAsSolved(line = {}, result = {}) {
  const kind = contractKindForLine(line, result);
  return [
    RESULT_CONTRACT_KIND.DB_EXACT_ROOT,
    RESULT_CONTRACT_KIND.FORCED_MATE_EXACT,
    RESULT_CONTRACT_KIND.FORCED_MATE_BOUND,
    RESULT_CONTRACT_KIND.DB_BRIDGE_MATE_BOUND,
    RESULT_CONTRACT_KIND.DB_BRIDGE_DRAW,
    RESULT_CONTRACT_KIND.FORCED_DRAW,
    RESULT_CONTRACT_KIND.TERMINAL
  ].includes(kind);
}

export function normalizeLineContract(line = {}, result = {}) {
  if (!line || typeof line !== 'object') return line;
  let next = { ...line };
  let kind = contractKindForLine(next, result);

  if (kind === RESULT_CONTRACT_KIND.DB_BRIDGE_MATE_BOUND) {
    next = {
      ...next,
      mateVerified: false,
      mateProof: false,
      endgameProof: false,
      mateUpperBound: true,
      tablebaseBridgeProof: true,
      rootScoreExact: false,
      scoreNumeric: true,
      resultContract: kind,
      resultKindV2: kind
    };
    return next;
  }

  if (kind === RESULT_CONTRACT_KIND.DB_BRIDGE_DRAW) {
    next = {
      ...next,
      mateVerified: false,
      mateProof: false,
      endgameProof: false,
      tablebaseBridgeDraw: true,
      tablebaseBridgeProof: false,
      score: 0,
      scoreText: next.scoreText || '0.00',
      scoreNumeric: true,
      resultContract: kind,
      resultKindV2: kind
    };
    return next;
  }

  const score = finiteNumber(next.score, 0);
  const looksLikeUnbackedMate = Math.abs(score) >= MATE_DISPLAY_THRESHOLD || String(next.scoreText || '').includes('#');
  if (looksLikeUnbackedMate && !canDisplayMateIn(next, result) && !canDisplayMateBound(next, result) && kind !== RESULT_CONTRACT_KIND.DB_EXACT_ROOT) {
    const safeScore = score < 0 ? -FINITE_MATE_CANDIDATE_SCORE : FINITE_MATE_CANDIDATE_SCORE;
    next = {
      ...next,
      score: safeScore,
      scoreText: finiteEvalText(safeScore),
      scoreKind: next.scoreKind === 'ordinary-search' ? 'mate-candidate-eval' : String(next.scoreKind || 'mate-candidate-eval'),
      scoreNumeric: true,
      mateVerified: false,
      mateProof: false,
      endgameProof: false,
      mateUpperBound: false,
      mateCandidate: true,
      pvMateLineOnly: Boolean(next.pvMateLineOnly || looksLikeUnbackedMate),
      rootScoreExact: false
    };
    kind = RESULT_CONTRACT_KIND.MATE_CANDIDATE;
  }

  return {
    ...next,
    resultContract: kind,
    resultKindV2: kind
  };
}

export function normalizeResultContract(result) {
  if (!result || typeof result !== 'object') return result;
  const sourceLines = Array.isArray(result.lines) ? result.lines : [];
  const lines = sourceLines.map(line => normalizeLineContract(line, result));
  const firstLine = lines.length ? lines[0] : null;
  const firstKind = firstLine ? contractKindForLine(firstLine, result) : contractKindForLine({}, result);
  // A secondary PV may contain a replay-verified mate after a blunder.  That is
  // displayable line evidence, not a solved root result.  Root-level solved
  // status is assigned by result-quality after checking the first/best line and
  // its root-side utility.
  const solvedByContract = Boolean(firstLine && canDisplayAsSolved(firstLine, result));
  return {
    ...result,
    lines,
    resultContract: firstKind,
    resultKindV2: firstKind,
    solved: Boolean(result.solved && solvedByContract)
  };
}

export function isPublishableLine(line = {}, result = {}) {
  const kind = contractKindForLine(line, result);
  if ([RESULT_CONTRACT_KIND.PROOF_SEED_INTERNAL, RESULT_CONTRACT_KIND.EMPTY].includes(kind)) return false;
  if ([RESULT_CONTRACT_KIND.MATE_CANDIDATE].includes(kind)) return Number.isFinite(Number(line?.score));
  if (canDisplayAsSolved(line, result)) return true;
  if (line?.scoreNumeric === false || line?.unverifiedMate || line?.matePendingUnscored) return false;
  return Number.isFinite(Number(line?.score));
}
