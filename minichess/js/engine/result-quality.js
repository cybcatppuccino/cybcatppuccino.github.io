import { RESULT_CONTRACT_KIND, contractKindForLine, normalizeResultContract } from './result-contract.js';

export const RESULT_KIND = Object.freeze({
  EMPTY: 'empty',
  LIVE: 'live',
  SEARCH: 'search',
  TABLEBASE_ROOT: 'tablebase-root',
  VERIFIED_MATE: 'mate',
  TERMINAL: 'terminal'
});

// PVs are display lines, not proof-depth certificates. A valid line may end
// at a repetition draw before the completed search depth.
export function pvTargetForDepth(_depth) { return 0; }
export function isExactTablebaseLine(line) {
  return Boolean(line?.tablebase && line?.tablebaseRoot && line?.tablebaseExactDtm);
}
export function isTrustedExactTablebaseResult(result) {
  return Boolean(result?.tablebase && result?.tablebaseRoot
    && Array.isArray(result.lines) && result.lines.length
    && result.lines.every(isExactTablebaseLine));
}
export function rootUtilityForLine(line = {}, result = {}) {
  const score = Number(line.score || 0);
  return Number(result?.rootTurn || 1) > 0 ? score : -score;
}
export function lineHasCompletePv(line = {}, result = {}) {
  if (!line || typeof line !== 'object') return false;
  if (line.pvComplete === true) return true;
  const contract = contractKindForLine(line, result);
  if (result?.terminal || contract === RESULT_CONTRACT_KIND.TERMINAL) return true;
  if (contract === RESULT_CONTRACT_KIND.TABLEBASE_ROOT || contract === RESULT_CONTRACT_KIND.MATE) return true;
  // Ordinary alpha-beta lines are complete when their root score was produced
  // by a completed iteration. PV length has no bearing on this: repetition may
  // terminate a legal best line before the nominal depth.
  return Boolean(result?.completed !== false && result?.pvComplete !== false
    && !result?.pvIncomplete && result?.multiPvVerified !== false
    && line?.rootScoreExact !== false && !line?.liveUpdate);
}
export function resultPvProfile(result, linesOverride = null) {
  const lines = Array.isArray(linesOverride) ? linesOverride : (Array.isArray(result?.lines) ? result.lines : []);
  const pvDepth = Math.max(0, ...lines.slice(0, 3).map(line => Array.isArray(line?.pv) ? line.pv.length : 0));
  const pvComplete = Boolean(result?.pvComplete !== false && !result?.pvIncomplete
    && result?.multiPvVerified !== false && result?.completed !== false);
  return { pvComplete, pvDepth, pvTarget: 0 };
}
export function classifyResult(result) {
  if (!result) return { kind: RESULT_KIND.EMPTY, rank: 0, exact: false, solved: false };
  if (isTrustedExactTablebaseResult(result)) return { kind: RESULT_KIND.TABLEBASE_ROOT, rank: 100, exact: true, solved: true };
  const first = result?.lines?.[0] || {};
  const contract = contractKindForLine(first, result);
  if (result.terminal || contract === RESULT_CONTRACT_KIND.TERMINAL) return { kind: RESULT_KIND.TERMINAL, rank: 95, exact: true, solved: true };
  if (contract === RESULT_CONTRACT_KIND.MATE && first.mateVerified) return { kind: RESULT_KIND.VERIFIED_MATE, rank: 90, exact: true, solved: true };
  const profile = resultPvProfile(result);
  if (result.liveProgress || result.liveUpdate || !profile.pvComplete || result.completed === false) {
    return { kind: RESULT_KIND.LIVE, rank: 30, exact: false, solved: false };
  }
  return { kind: RESULT_KIND.SEARCH, rank: 50, exact: false, solved: false };
}
export function isSolvedResult(result) { return classifyResult(result).solved; }
export function isPvCompleteResult(result) { return classifyResult(result).exact || resultPvProfile(result).pvComplete; }
export function isThinPvResult(result) { return !classifyResult(result).exact && !resultPvProfile(result).pvComplete; }
export function shouldCacheResult(result) {
  const q = classifyResult(result);
  return Boolean(result?.lines?.length) && q.exact && (q.kind === RESULT_KIND.TABLEBASE_ROOT || q.kind === RESULT_KIND.VERIFIED_MATE);
}
export function withResultQuality(result) {
  if (!result) return result;
  const normalized = normalizeResultContract(result);
  const profile = resultPvProfile(normalized);
  const q = classifyResult({ ...normalized, ...profile });
  return normalizeResultContract({ ...normalized, ...profile, resultKind: q.kind, resultRank: q.rank, solved: q.solved });
}
export function compareAnalysisResults(previous, next, { preferNextOnTie = true } = {}) {
  if (!previous) return next;
  if (!next) return previous;
  const a = classifyResult(previous), b = classifyResult(next);
  if (a.rank !== b.rank) return b.rank > a.rank ? next : previous;
  const ad = Number(previous.scoreDepth || previous.depth || 0), bd = Number(next.scoreDepth || next.depth || 0);
  if (ad !== bd) return bd > ad ? next : previous;
  const an = Number(previous.nodes || 0), bn = Number(next.nodes || 0);
  if (an !== bn) return bn > an ? next : previous;
  return preferNextOnTie ? next : previous;
}
