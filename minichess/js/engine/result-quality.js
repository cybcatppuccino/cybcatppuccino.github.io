// Shared result-quality helpers for analysis cache and workers.
// Keep this module dependency-free so it can be imported by browser workers,
// node tests, and UI-side cache code without creating engine/tablebase cycles.

export const RESULT_KIND = Object.freeze({
  EMPTY: 'empty',
  LIVE_THIN: 'live-thin',
  LIVE_COMPLETE: 'live-complete',
  TABLEBASE_BOUND: 'tablebase-bound',
  WDL_ONLY: 'wdl-only',
  ENDGAME_PROOF: 'endgame-proof',
  FORTRESS_PROOF: 'fortress-proof',
  EXACT_TABLEBASE: 'exact-tablebase',
  VERIFIED_MATE: 'verified-mate',
  TERMINAL: 'terminal'
});

export function pvTargetForDepth(depth) {
  const d = Math.max(0, Number(depth || 0));
  return d >= 10 ? Math.min(14, Math.max(8, d - 2)) : d >= 8 ? 6 : 1;
}

export function lineHasCompletePv(line, result = {}) {
  if (!line) return false;
  if (result?.terminal || result?.fortressProof || result?.endgameProof) return true;
  if (line.mateVerified || line.fortressProof || line.endgameProof) return true;
  if (isExactTablebaseLine(line) || isTrustedExactTablebaseResult(result)) return true;
  if (result?.tablebase && !result?.tablebaseDtmBound && !line.tablebaseBound && !line.dtmUpperBound) return true;
  const depth = Number(result?.depth || line.depth || 0);
  if (depth < 8) return true;
  const pvLength = Array.isArray(line.pv) ? line.pv.length : 0;
  return pvLength >= pvTargetForDepth(depth);
}

export function resultPvProfile(result, lines = result?.lines || []) {
  const depth = Math.max(0, Number(result?.depth || 0));
  const visible = Array.isArray(lines) ? lines : [];
  const target = pvTargetForDepth(depth);
  if (!visible.length) {
    return { pvComplete: false, pvDepth: 0, pvTarget: target, scoreDepth: depth };
  }
  if (classifyResult(result).rank >= RESULT_RANKS[RESULT_KIND.TABLEBASE_BOUND]) {
    const pvDepth = Math.max(0, ...visible.map(line => Array.isArray(line?.pv) ? line.pv.length : 0));
    return { pvComplete: true, pvDepth, pvTarget: 0, scoreDepth: Math.max(0, Number(result?.scoreDepth || depth)) };
  }
  const bestPvLength = Array.isArray(visible[0]?.pv) ? visible[0].pv.length : 0;
  const bestComplete = lineHasCompletePv(visible[0], result);
  const allVisibleComplete = visible.slice(0, Math.min(3, visible.length)).every(line => lineHasCompletePv(line, result));
  return {
    pvComplete: Boolean(bestComplete && allVisibleComplete),
    pvDepth: bestComplete ? depth : Math.min(depth, Math.max(0, bestPvLength + 2)),
    pvTarget: target,
    scoreDepth: Math.max(0, Number(result?.scoreDepth || depth))
  };
}

export function isExactTablebaseLine(line) {
  if (!line?.tablebase) return false;
  if (line.tablebaseBound || line.dtmUpperBound) return false;
  if (line.tablebaseExactDtm) return true;
  // Exact tablebase draws have no meaningful DTM, but are still solved if the
  // result/line is marked as tablebase and not as a bound.
  return Number(line.tablebaseWdl || 0) === 0 && !line.tablebaseBound && !line.dtmUpperBound;
}

export function isTrustedExactTablebaseResult(result) {
  if (!result?.tablebase || !Array.isArray(result.lines) || !result.lines.length) return false;
  if (result.tablebaseDtmBound || result.tablebaseBound || result.dtmUpperBound) return false;
  if (result.lines.some(line => line?.tablebaseBound || line?.dtmUpperBound)) return false;
  if (result.tablebaseWdl === 0 || result.lines[0]?.tablebaseWdl === 0) return true;
  return result.lines.some(isExactTablebaseLine) || String(result.tablebaseSource || result.lines[0]?.tablebaseSource || result.lines[0]?.source || '').includes('exact');
}

export const RESULT_RANKS = Object.freeze({
  [RESULT_KIND.EMPTY]: 0,
  [RESULT_KIND.LIVE_THIN]: 10,
  [RESULT_KIND.LIVE_COMPLETE]: 20,
  [RESULT_KIND.WDL_ONLY]: 30,
  [RESULT_KIND.TABLEBASE_BOUND]: 40,
  [RESULT_KIND.ENDGAME_PROOF]: 52,
  [RESULT_KIND.FORTRESS_PROOF]: 55,
  [RESULT_KIND.EXACT_TABLEBASE]: 70,
  [RESULT_KIND.VERIFIED_MATE]: 80,
  [RESULT_KIND.TERMINAL]: 90
});

export function classifyResult(result) {
  if (!result || !Array.isArray(result.lines) || !result.lines.length) {
    return { kind: RESULT_KIND.EMPTY, rank: RESULT_RANKS[RESULT_KIND.EMPTY], solved: false, exact: false };
  }
  const first = result.lines[0] || {};
  if (result.terminal && !result.tablebase) {
    return { kind: RESULT_KIND.TERMINAL, rank: RESULT_RANKS[RESULT_KIND.TERMINAL], solved: true, exact: true };
  }
  if (first.mateVerified || (result.solved && first.mateVerified)) {
    return { kind: RESULT_KIND.VERIFIED_MATE, rank: RESULT_RANKS[RESULT_KIND.VERIFIED_MATE], solved: true, exact: true };
  }
  if (isTrustedExactTablebaseResult(result)) {
    return { kind: RESULT_KIND.EXACT_TABLEBASE, rank: RESULT_RANKS[RESULT_KIND.EXACT_TABLEBASE], solved: true, exact: true };
  }
  if (result.fortressProof || first.fortressProof) {
    return { kind: RESULT_KIND.FORTRESS_PROOF, rank: RESULT_RANKS[RESULT_KIND.FORTRESS_PROOF], solved: true, exact: true };
  }
  if (result.endgameProof || first.endgameProof) {
    return { kind: RESULT_KIND.ENDGAME_PROOF, rank: RESULT_RANKS[RESULT_KIND.ENDGAME_PROOF], solved: true, exact: true };
  }
  if (result.tablebase || result.tablebaseDtmBound || first.tablebase || first.tablebaseBound || first.dtmUpperBound) {
    const onlyWdl = Boolean(result.tablebase || first.tablebase)
      && !result.tablebaseDtmBound
      && !first.tablebaseBound
      && !first.dtmUpperBound
      && !Number(first.dtm || 0)
      && !first.mateVerified
      && !isExactTablebaseLine(first);
    const kind = onlyWdl ? RESULT_KIND.WDL_ONLY : RESULT_KIND.TABLEBASE_BOUND;
    return { kind, rank: RESULT_RANKS[kind], solved: Boolean(result.tablebase && !result.tablebaseDtmBound), exact: false };
  }
  const profile = resultPvProfileShallow(result);
  const kind = profile.pvComplete ? RESULT_KIND.LIVE_COMPLETE : RESULT_KIND.LIVE_THIN;
  return { kind, rank: RESULT_RANKS[kind], solved: false, exact: false };
}

function resultPvProfileShallow(result) {
  const depth = Math.max(0, Number(result?.scoreDepth || result?.depth || 0));
  const firstPv = Array.isArray(result?.lines?.[0]?.pv) ? result.lines[0].pv.length : 0;
  const target = depth >= 8 ? pvTargetForDepth(depth) : 0;
  const explicitComplete = result?.pvComplete !== false && !result?.pvIncomplete && result?.completed !== false;
  return { pvComplete: explicitComplete && (!target || firstPv >= target), pvDepth: firstPv, pvTarget: target };
}

export function isSolvedResult(result) {
  return classifyResult(result).solved;
}

export function isPvCompleteResult(result) {
  if (!result) return false;
  if (classifyResult(result).rank >= RESULT_RANKS[RESULT_KIND.TABLEBASE_BOUND]) return true;
  if (result.pvComplete === false || result.pvIncomplete) return false;
  return resultPvProfile(result).pvComplete;
}

export function isThinPvResult(result) {
  if (!result || classifyResult(result).rank >= RESULT_RANKS[RESULT_KIND.TABLEBASE_BOUND]) return false;
  const depth = Number(result.scoreDepth || result.depth || 0);
  const profile = resultPvProfile(result);
  return depth >= 10 && !profile.pvComplete && profile.pvDepth > 0;
}

export function shouldCacheResult(result) {
  if (!result?.lines?.length) return false;
  const quality = classifyResult(result);
  if (quality.rank >= RESULT_RANKS[RESULT_KIND.TABLEBASE_BOUND] || result.terminal) return true;
  const profile = resultPvProfile(result);
  return profile.pvComplete && result.completed !== false;
}

export function withResultQuality(result) {
  if (!result) return result;
  const profile = resultPvProfile(result);
  const quality = classifyResult({ ...result, ...profile });
  return {
    ...result,
    ...profile,
    resultKind: quality.kind,
    resultRank: quality.rank,
    solved: Boolean(result.solved || quality.solved)
  };
}

export function compareAnalysisResults(previous, next, { preferNextOnTie = true } = {}) {
  if (!previous) return next;
  if (!next) return previous;
  const previousQuality = classifyResult(previous);
  const nextQuality = classifyResult(next);
  if (previousQuality.rank !== nextQuality.rank) return nextQuality.rank > previousQuality.rank ? next : previous;

  const previousProfile = resultPvProfile(previous);
  const nextProfile = resultPvProfile(next);
  if (previousProfile.pvComplete !== nextProfile.pvComplete) return nextProfile.pvComplete ? next : previous;

  const previousScoreDepth = Number(previous.scoreDepth || previous.depth || 0);
  const nextScoreDepth = Number(next.scoreDepth || next.depth || 0);
  if (previousScoreDepth !== nextScoreDepth) return nextScoreDepth > previousScoreDepth ? next : previous;

  const previousPvDepth = Number(previous.pvDepth || previousProfile.pvDepth || 0);
  const nextPvDepth = Number(next.pvDepth || nextProfile.pvDepth || 0);
  if (previousPvDepth !== nextPvDepth) return nextPvDepth > previousPvDepth ? next : previous;

  const previousNodes = Number(previous.nodes || 0);
  const nextNodes = Number(next.nodes || 0);
  if (previousNodes !== nextNodes && previousQuality.kind === nextQuality.kind) return nextNodes >= previousNodes ? next : previous;
  return preferNextOnTie ? next : previous;
}
