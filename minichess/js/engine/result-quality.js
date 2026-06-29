// Shared result-quality helpers for analysis cache and workers.
// Keep this module dependency-free so it can be imported by browser workers,
// node tests, and UI-side cache code without creating engine/tablebase cycles.
//
// v19.8 policy:
// - A tablebase answer is exact only when the *root* position was probed.
// - A future tablebase hit along a speculative PV is a hint, never a mate proof.
// - Ordinary search snapshots are useful for display, but are never persisted
//   or resumed across a fresh root analysis.

export const RESULT_KIND = Object.freeze({
  EMPTY: 'empty',
  LIVE_THIN: 'live-thin',
  LIVE_COMPLETE: 'live-complete',
  TABLEBASE_HINT: 'tablebase-hint',
  // Kept as a compatibility enum for older callers. v19.8 never awards it
  // solved/exact status for a conditional PV annotation.
  TABLEBASE_BOUND: 'tablebase-bound',
  TABLEBASE_BRIDGE_MATE: 'tablebase-bridge-mate',
  TABLEBASE_BRIDGE_DRAW: 'tablebase-bridge-draw',
  // A completed root-restricted normal search in which exact WDL leaves are
  // deliberately capped. It is a numeric worst-defence estimate, not a mate
  // certificate and never cross-root cacheable.
  TABLEBASE_MIXED_BOUND: 'tablebase-mixed-bound',
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

export function isExactTablebaseLine(line) {
  return Boolean(line?.tablebase && line?.tablebaseScope === 'root-exact' && !line?.tablebaseBound);
}

export function isTrustedExactTablebaseResult(result) {
  if (!result?.tablebase || result?.tablebaseScope !== 'root-exact') return false;
  return Boolean(Array.isArray(result.lines) && result.lines.length && result.lines.every(line => isExactTablebaseLine(line)));
}

function lineHasProof(line, result = {}) {
  return Boolean(
    line?.fortressProof || line?.endgameProof ||
    (line?.mateVerified && (line?.mateProof || line?.endgameProof || line?.tablebaseBridgeProof || result?.mateProof || result?.endgameProof || result?.tablebaseBridgeProof))
  );
}

export function lineHasCompletePv(line, result = {}) {
  if (!line) return false;
  if (result?.terminal || result?.fortressProof || result?.endgameProof || result?.tablebaseBridgeDraw || line?.tablebaseBridgeDraw) return true;
  if (lineHasProof(line, result)) return true;
  if (isExactTablebaseLine(line) || isTrustedExactTablebaseResult(result)) return true;
  // A TT-appended tail is display help only. It must not convert a partial root
  // search into a cacheable / stable PV.
  if (line?.pvReconstructed) return false;
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
  const quality = classifyResultShallow(result);
  if (quality.exact) {
    const pvDepth = Math.max(0, ...visible.map(line => Array.isArray(line?.pv) ? line.pv.length : 0));
    return { pvComplete: true, pvDepth, pvTarget: 0, scoreDepth: Math.max(0, Number(result?.scoreDepth || depth)) };
  }
  const bestPvLength = Array.isArray(visible[0]?.pv) ? visible[0].pv.length : 0;
  const bestComplete = lineHasCompletePv(visible[0], result);
  const allVisibleComplete = visible.slice(0, Math.min(3, visible.length)).every(line => lineHasCompletePv(line, result));
  const rootsExact = result?.multiPvVerified !== false && visible.slice(0, Math.min(3, visible.length)).every(line => line?.rootScoreExact !== false);
  return {
    pvComplete: Boolean(bestComplete && allVisibleComplete && rootsExact),
    pvDepth: bestComplete ? depth : Math.min(depth, Math.max(0, bestPvLength + 2)),
    pvTarget: target,
    scoreDepth: Math.max(0, Number(result?.scoreDepth || depth))
  };
}

export const RESULT_RANKS = Object.freeze({
  [RESULT_KIND.EMPTY]: 0,
  [RESULT_KIND.LIVE_THIN]: 10,
  [RESULT_KIND.TABLEBASE_HINT]: 12,
  [RESULT_KIND.TABLEBASE_BOUND]: 12,
  [RESULT_KIND.LIVE_COMPLETE]: 20,
  // Above a normal completed iteration, below any proof. This keeps a
  // completed worst-defence audit stable while the background bridge prover
  // continues to seek a true mate/draw certificate.
  [RESULT_KIND.TABLEBASE_MIXED_BOUND]: 22,
  [RESULT_KIND.WDL_ONLY]: 25,
  [RESULT_KIND.TABLEBASE_BRIDGE_DRAW]: 60,
  [RESULT_KIND.TABLEBASE_BRIDGE_MATE]: 78,
  [RESULT_KIND.ENDGAME_PROOF]: 52,
  [RESULT_KIND.FORTRESS_PROOF]: 55,
  [RESULT_KIND.VERIFIED_MATE]: 80,
  [RESULT_KIND.EXACT_TABLEBASE]: 85,
  [RESULT_KIND.TERMINAL]: 90
});

function classifyResultShallow(result) {
  if (!result || !Array.isArray(result.lines) || !result.lines.length) {
    return { kind: RESULT_KIND.EMPTY, rank: RESULT_RANKS[RESULT_KIND.EMPTY], solved: false, exact: false };
  }
  const first = result.lines[0] || {};
  if (result.terminal && !result.tablebase) {
    return { kind: RESULT_KIND.TERMINAL, rank: RESULT_RANKS[RESULT_KIND.TERMINAL], solved: true, exact: true };
  }
  if (isTrustedExactTablebaseResult(result)) {
    return { kind: RESULT_KIND.EXACT_TABLEBASE, rank: RESULT_RANKS[RESULT_KIND.EXACT_TABLEBASE], solved: true, exact: true };
  }
  if (result.tablebaseBridgeDraw || first.tablebaseBridgeDraw) {
    return { kind: RESULT_KIND.TABLEBASE_BRIDGE_DRAW, rank: RESULT_RANKS[RESULT_KIND.TABLEBASE_BRIDGE_DRAW], solved: true, exact: false };
  }
  if (first.tablebaseBridgeProof && first.mateVerified && lineHasProof(first, result)) {
    // The winner is forced, but the displayed distance is an AND/OR upper
    // bound rather than a shortest-DTM claim.  It must outrank ordinary
    // search while remaining below an independently minimized mate proof.
    return { kind: RESULT_KIND.TABLEBASE_BRIDGE_MATE, rank: RESULT_RANKS[RESULT_KIND.TABLEBASE_BRIDGE_MATE], solved: true, exact: false };
  }
  if (first.mateVerified && lineHasProof(first, result)) {
    return { kind: RESULT_KIND.VERIFIED_MATE, rank: RESULT_RANKS[RESULT_KIND.VERIFIED_MATE], solved: true, exact: true };
  }
  if (result.fortressProof || first.fortressProof) {
    return { kind: RESULT_KIND.FORTRESS_PROOF, rank: RESULT_RANKS[RESULT_KIND.FORTRESS_PROOF], solved: true, exact: true };
  }
  if (result.tablebaseMixedAudit || first.tablebaseMixedAudit) {
    return { kind: RESULT_KIND.TABLEBASE_MIXED_BOUND, rank: RESULT_RANKS[RESULT_KIND.TABLEBASE_MIXED_BOUND], solved: false, exact: false };
  }
  if (result.endgameProof || first.endgameProof) {
    return { kind: RESULT_KIND.ENDGAME_PROOF, rank: RESULT_RANKS[RESULT_KIND.ENDGAME_PROOF], solved: true, exact: true };
  }
  if (result.tablebaseDtmHint || first.tablebaseHint || first.tablebaseBound || first.dtmUpperBound) {
    return { kind: RESULT_KIND.TABLEBASE_HINT, rank: RESULT_RANKS[RESULT_KIND.TABLEBASE_HINT], solved: false, exact: false };
  }
  if (result.tablebase || first.tablebase) {
    // A non-root tablebase flag from legacy payloads is informational only.
    return { kind: RESULT_KIND.WDL_ONLY, rank: RESULT_RANKS[RESULT_KIND.WDL_ONLY], solved: false, exact: false };
  }
  return { kind: RESULT_KIND.EMPTY, rank: RESULT_RANKS[RESULT_KIND.EMPTY], solved: false, exact: false };
}

export function classifyResult(result) {
  const shallow = classifyResultShallow(result);
  if (shallow.kind !== RESULT_KIND.EMPTY) return shallow;
  const profile = resultPvProfileShallow(result);
  const kind = profile.pvComplete ? RESULT_KIND.LIVE_COMPLETE : RESULT_KIND.LIVE_THIN;
  return { kind, rank: RESULT_RANKS[kind], solved: false, exact: false };
}

function resultPvProfileShallow(result) {
  const depth = Math.max(0, Number(result?.scoreDepth || result?.depth || 0));
  const firstPv = Array.isArray(result?.lines?.[0]?.pv) ? result.lines[0].pv.length : 0;
  const target = depth >= 8 ? pvTargetForDepth(depth) : 0;
  const explicitComplete = result?.pvComplete !== false && !result?.pvIncomplete && result?.completed !== false && result?.multiPvVerified !== false;
  const reconstructed = Boolean(result?.lines?.slice(0, Math.min(3, result?.lines?.length || 0)).some(line => line?.pvReconstructed));
  return { pvComplete: explicitComplete && !reconstructed && (!target || firstPv >= target), pvDepth: firstPv, pvTarget: target };
}

export function isSolvedResult(result) {
  return classifyResult(result).solved;
}

export function isPvCompleteResult(result) {
  if (!result) return false;
  if (classifyResult(result).exact) return true;
  if (result.pvComplete === false || result.pvIncomplete || result.multiPvVerified === false) return false;
  return resultPvProfile(result).pvComplete;
}

export function isThinPvResult(result) {
  if (!result || classifyResult(result).exact) return false;
  const depth = Number(result.scoreDepth || result.depth || 0);
  const profile = resultPvProfile(result);
  return (depth >= 10 && !profile.pvComplete && profile.pvDepth > 0) || Boolean(result?.lines?.some(line => line?.pvReconstructed));
}

// Persistence is deliberately narrower than display quality. Cross-root reuse
// is allowed only for an exact root tablebase / verified proof / terminal rule.
export function shouldCacheResult(result) {
  if (!result?.lines?.length) return false;
  const quality = classifyResult(result);
  if (quality.kind === RESULT_KIND.TABLEBASE_BRIDGE_MATE || quality.kind === RESULT_KIND.TABLEBASE_BRIDGE_DRAW) return false;
  return quality.exact || quality.kind === RESULT_KIND.TERMINAL;
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
