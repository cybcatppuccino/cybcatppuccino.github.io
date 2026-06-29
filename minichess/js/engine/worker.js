import { EnginePosition, GardnerSearcher, ENGINE_VERSION, validateMateResult, uciToMove } from './engine.js';
import { GardnerTablebase } from './tablebase.js';
import { compareAnalysisResults, isSolvedResult, isTrustedExactTablebaseResult, resultPvProfile, withResultQuality } from './result-quality.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';

// v19.8 analysis worker
// Result ownership rule: every published score/PV pair comes from one completed
// iteration (or one exact proof). Incomplete chunks may update progress only.
const MAX_DEPTH = 48;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
searcher.setTablebaseProbe(position => tablebase.probeSync(position));
tablebase.init().catch(() => {});

let activeToken = 0;
let running = false;
let paused = false;
let current = null;
let nextDepth = 1;
let effortMs = 950;
let currentBudgetMs = 90;
let multipv = 3;
let firstChunk = true;
let totalNodes = 0;
let totalElapsed = 0;
let currentKernel = ENGINE_KERNELS.ORION;

const TABLEBASE_PROMOTION_RETRY_MS = 220;
const TABLEBASE_FRONTIER_MAX_PLY = 4;
const TABLEBASE_FRONTIER_MAX_STATES = 320;
const TABLEBASE_TAIL_MAX_PLY = 96;
const TABLEBASE_BRIDGE_MAX_PLY = 48;
const TABLEBASE_BRIDGE_MAX_NODES = 64_000;
const TABLEBASE_BRIDGE_TIME_MS = 1_400;
const TABLEBASE_BRIDGE_IMPROVE_TIME_MS = 700;
// v19.8: a capped-WDL audit keeps exact tablebase leaves decisive for
// pruning while exposing the strongest ordinary defensive branch as a
// numeric score when the AND/OR bridge certificate is still incomplete.
const TABLEBASE_MIXED_AUDIT_SCORE_CAP = 1_000;
const TABLEBASE_MIXED_AUDIT_MIN_DEPTH = 5;
// Keep the first audit shallow enough to finish before a stream of normal
// iterative-deepening chunks can make it obsolete. A single refinement gets
// two more plies, but never turns the bounded score into a proof.
const TABLEBASE_MIXED_AUDIT_INITIAL_MAX_DEPTH = 8;
const TABLEBASE_MIXED_AUDIT_MAX_DEPTH = 10;
const TABLEBASE_MIXED_AUDIT_TIME_MS = 780;
const TABLEBASE_MIXED_AUDIT_HASH_ENTRIES = 196_608;
const historyFenKeyCache = new Map();
const HISTORY_FEN_KEY_CACHE_LIMIT = 256;

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function reportFatalWorkerError(error, token = activeToken) {
  const message = error?.stack || error?.message || String(error || 'Unknown worker error.');
  try { post('error', { token: Number(token || activeToken || 0), message }); } catch {}
}

self.addEventListener('error', event => reportFatalWorkerError(event?.error || event?.message || 'Worker script error.'));
self.addEventListener('unhandledrejection', event => reportFatalWorkerError(event?.reason || 'Unhandled worker promise rejection.'));

function historyKeyFromFen(fen) {
  const key = String(fen || '');
  const cached = historyFenKeyCache.get(key);
  if (cached) return cached;
  const position = EnginePosition.fromFEN(key);
  const value = { a: position.hashA, b: position.hashB };
  historyFenKeyCache.set(key, value);
  if (historyFenKeyCache.size > HISTORY_FEN_KEY_CACHE_LIMIT) {
    const oldest = historyFenKeyCache.keys().next().value;
    if (oldest !== undefined) historyFenKeyCache.delete(oldest);
  }
  return value;
}

function lineUtilityForSide(line, sideToMove) {
  const score = Number(line?.score || 0);
  return sideToMove === -1 ? -score : score;
}

function sortResultLinesForSide(result, sideToMove, limit = 3) {
  if (!result || !Array.isArray(result.lines)) return result;
  const maxLines = Math.max(1, Math.min(5, Number(limit || result.lines.length || 3)));
  const lines = result.lines
    .map(line => ({ ...line, pv: Array.isArray(line?.pv) ? line.pv.slice() : [] }))
    .sort((a, b) => lineUtilityForSide(b, sideToMove) - lineUtilityForSide(a, sideToMove))
    .slice(0, maxLines);
  return { ...result, lines };
}

function initialBudget(depth) {
  if (depth <= 1) return 70;
  if (depth === 2) return 100;
  if (depth === 3) return 145;
  return Math.min(effortMs, 190 + (depth - 4) * 82);
}

function schedule(token, delay = 0) {
  setTimeout(() => void runChunk(token), delay);
}

function isTrustedResume(position, candidate) {
  if (!candidate?.lines?.length || candidate.engine !== ENGINE_VERSION || !isSolvedResult(candidate)) return null;
  const normalized = withResultQuality(sortResultLinesForSide(candidate, position.turn, multipv));
  if (isTrustedExactTablebaseResult(normalized)) return normalized;
  const first = normalized.lines[0];
  const verifiedMate = Boolean(first?.mateVerified && (first?.mateProof || first?.endgameProof || normalized?.mateProof || normalized?.endgameProof));
  if (verifiedMate && validateMateResult(position, first)) return normalized;
  if (normalized.fortressProof || normalized.endgameProof || normalized.terminal) return normalized;
  return null;
}

function isStableSearchResult(result) {
  if (!result?.lines?.length) return false;
  if (isSolvedResult(result)) return true;
  return Boolean(
    result.completed !== false &&
    result.pvComplete !== false &&
    !result.pvIncomplete &&
    result.multiPvVerified !== false
  );
}

function isTablebaseBridgeUpperBound(result) {
  return Boolean(result?.tablebaseBridgeProof || result?.lines?.[0]?.tablebaseBridgeProof || result?.lines?.[0]?.mateUpperBound);
}

function isTablebaseBridgeDraw(result) {
  return Boolean(result?.tablebaseBridgeDraw || result?.lines?.[0]?.tablebaseBridgeDraw);
}

function findBridgeTailLine(result, { nonDrawOnly = false } = {}) {
  if (!Array.isArray(result?.lines)) return null;
  return result.lines.find(line => {
    const tail = line?.tablebaseTail;
    if (tail && Boolean(tail.bridgeable || tail.terminal || tail.exactWdl)) {
      const wdl = Number(tail.wdl || 0);
      return Number.isFinite(wdl) && (!nonDrawOnly || wdl !== 0);
    }
    // Tail hydration is asynchronous and can lose a race to the next normal
    // iteration. A completed root line that already used a WDL sentinel is
    // still a valid trigger for the bounded audit; the audit will preload the
    // exact reachable blocks before it starts scoring.
    const raw = Number(line?.tablebaseRawScore || 0);
    return Boolean(line?.tablebaseBridgeCandidate)
      && Number.isFinite(raw)
      && (!nonDrawOnly || raw !== 0);
  }) || null;
}

function currentStillPrefersBridgeMove(move) {
  const active = current?.lastResult;
  const first = active?.lines?.[0];
  return Boolean(move && first?.move === move && (first?.tablebaseBridgeCandidate || first?.tablebaseTail));
}

function auditBaselineForMove(fallback, move) {
  return currentStillPrefersBridgeMove(move) && current?.lastResult?.lines?.length
    ? current.lastResult
    : fallback;
}

function currentBridgeLineForMove(fallback, move) {
  const active = current?.lastResult;
  const found = active?.lines?.find(line => line?.move === move && (line?.tablebaseBridgeCandidate || line?.tablebaseTail));
  return found || fallback;
}

function auditTargetDepth(baseline, refinement = 0) {
  const base = Math.max(0, Number(baseline?.scoreDepth || baseline?.depth || 0));
  const initial = Math.max(
    TABLEBASE_MIXED_AUDIT_MIN_DEPTH,
    Math.min(TABLEBASE_MIXED_AUDIT_INITIAL_MAX_DEPTH, base || TABLEBASE_MIXED_AUDIT_MIN_DEPTH)
  );
  return Math.min(
    TABLEBASE_MIXED_AUDIT_MAX_DEPTH,
    initial + Math.max(0, Number(refinement || 0)) * 2
  );
}

function isSaturatedTablebaseAudit(audit, line) {
  const cap = Math.max(1, Number(audit?.tablebaseScoreCap || 0));
  if (!audit?.tablebaseBoundedSearch || Number(audit?.tablebaseBoundedProbeHits || 0) <= 0 || !line) return false;
  return Math.abs(Number(line.score || 0)) >= cap;
}

function sortAuditedRootLines(lines, rootTurn) {
  return lines
    .filter(line => line?.move)
    .sort((a, b) => lineUtilityForSide(b, rootTurn) - lineUtilityForSide(a, rootTurn));
}

function buildTablebaseMixedAuditResult(root, baseline, sourceLine, audit) {
  const auditLine = audit?.lines?.[0];
  if (!auditLine?.move) return null;
  const baselineLines = Array.isArray(baseline?.lines) ? baseline.lines : [];
  const line = {
    ...auditLine,
    tablebaseTail: sourceLine?.tablebaseTail ? { ...sourceLine.tablebaseTail } : undefined,
    tablebaseTailComplete: Boolean(sourceLine?.tablebaseTailComplete),
    tablebaseBridgeCandidate: false,
    tablebaseDisplayFallback: false,
    tablebaseRawScore: Number(sourceLine?.tablebaseRawScore || 0),
    tablebaseMixedAudit: true,
    tablebaseMixedAuditCap: Number(audit.tablebaseScoreCap || TABLEBASE_MIXED_AUDIT_SCORE_CAP),
    tablebaseMixedAuditDepth: Number(audit.scoreDepth || audit.depth || 0),
    tablebaseMixedAuditNodes: Number(audit.nodes || 0),
    tablebaseMixedAuditElapsed: Number(audit.elapsed || 0),
    tablebaseMixedAuditProbeHits: Number(audit.tablebaseBoundedProbeHits || 0),
    tablebaseMixedAuditRootMove: auditLine.move,
    tablebaseMixedAuditSaturated: isSaturatedTablebaseAudit(audit, auditLine),
    tablebaseScope: 'mixed-wdl-audit',
    // The restricted root window completed; RESULT_KIND keeps this distinct
    // from an exact root-tablebase / mate theorem.
    rootScoreExact: true,
    pvComplete: auditLine.pvComplete !== false
  };
  return withResultQuality({
    ...baseline,
    // The audit re-scores one root move against every reply. Re-sort the
    // completed baseline after replacing that move so an ordinary alternative
    // is never hidden behind a formerly-infinite tablebase sentinel.
    lines: sortAuditedRootLines(
      [line, ...baselineLines.filter(item => item?.move && item.move !== line.move)],
      Number(root?.turn || baseline.rootTurn || 1)
    ),
    // Do not pretend every displayed MultiPV line was re-searched by the audit.
    // The audited root line is exact for its capped-WDL search; the remaining
    // lines stay attached to the completed normal iteration.
    completed: baseline.completed !== false,
    multiPvVerified: baseline.multiPvVerified !== false,
    pvComplete: baseline.pvComplete !== false,
    pvIncomplete: Boolean(baseline.pvIncomplete),
    solved: false,
    cached: false,
    tablebaseMixedAudit: true,
    tablebaseMixedAuditCap: Number(audit.tablebaseScoreCap || TABLEBASE_MIXED_AUDIT_SCORE_CAP),
    tablebaseMixedAuditDepth: Number(audit.scoreDepth || audit.depth || 0),
    tablebaseMixedAuditNodes: Number(audit.nodes || 0),
    tablebaseMixedAuditElapsed: Number(audit.elapsed || 0),
    tablebaseMixedAuditProbeHits: Number(audit.tablebaseBoundedProbeHits || 0),
    tablebaseMixedAuditRootMove: auditLine.move,
    tablebaseMixedAuditSaturated: isSaturatedTablebaseAudit(audit, auditLine),
    tablebaseMixedAuditWdl: Number(sourceLine?.tablebaseTail?.wdl || Math.sign(Number(sourceLine?.tablebaseRawScore || 0))),
    rootTurn: Number(root?.turn || baseline.rootTurn || 1)
  });
}

function bridgeScoreText(whiteScore, dtmPly) {
  const moves = Math.max(1, Math.ceil(Math.max(1, Number(dtmPly || 1)) / 2));
  return `${whiteScore < 0 ? '≤-#' : '≤#'}${moves}`;
}

function buildTablebaseBridgeResult(root, baseline, proof) {
  const rootTurn = Number(root.turn || 1);
  const baselineLines = Array.isArray(baseline?.lines) ? baseline.lines : [];
  if (proof?.tablebaseBridgeDraw) {
    const line = {
      move: proof.pv?.[0] || baselineLines[0]?.move || '',
      score: 0,
      scoreText: '0.00',
      pv: Array.isArray(proof.pv) ? proof.pv.slice() : [],
      mateVerified: false,
      tablebaseBridgeDraw: true,
      tablebaseBridgeProof: false,
      tablebaseScope: 'bridge-proof',
      rootScoreExact: true,
      pvComplete: true
    };
    return withResultQuality({
      ...baseline,
      lines: [line, ...baselineLines.filter(item => item?.move && item.move !== line.move)],
      tablebaseBridgeDraw: true,
      tablebaseBridgeNodes: Number(proof.proofNodes || 0),
      tablebaseBridgeLeaves: Number(proof.proofLeaves || 0),
      tablebaseBridgeElapsed: Number(proof.elapsed || 0),
      completed: true,
      multiPvVerified: true,
      pvComplete: true,
      pvIncomplete: false,
      solved: true,
      cached: false
    });
  }
  const dtm = Math.max(1, Number(proof?.dtmPly || 0));
  const rootScore = (Number(proof?.controller) === rootTurn ? 1 : -1) * (30_000 - dtm);
  const whiteScore = rootTurn === 1 ? rootScore : -rootScore;
  const line = {
    move: proof.pv?.[0] || baselineLines[0]?.move || '',
    score: whiteScore,
    scoreText: bridgeScoreText(whiteScore, dtm),
    pv: Array.isArray(proof.pv) ? proof.pv.slice() : [],
    mateVerified: true,
    mateUpperBound: true,
    tablebaseBridgeProof: true,
    tablebaseBridgeDraw: false,
    tablebaseBridgeDtm: dtm,
    tablebaseBridgeController: Number(proof?.controller || 0),
    tablebaseScope: 'bridge-proof',
    rootScoreExact: true,
    pvComplete: true,
    dtm
  };
  return withResultQuality({
    ...baseline,
    lines: [line, ...baselineLines.filter(item => item?.move && item.move !== line.move)],
    tablebaseBridgeProof: true,
    tablebaseBridgeDtm: dtm,
    tablebaseBridgeNodes: Number(proof.proofNodes || 0),
    tablebaseBridgeLeaves: Number(proof.proofLeaves || 0),
    tablebaseBridgeElapsed: Number(proof.elapsed || 0),
    completed: true,
    multiPvVerified: true,
    pvComplete: true,
    pvIncomplete: false,
    solved: true,
    cached: false
  });
}

function progressFromStable(stable, snapshot, requestedDepth) {
  if (!stable?.lines?.length) return null;
  const nodes = totalNodes + Math.max(0, Number(snapshot?.nodes || 0));
  const elapsed = totalElapsed + Math.max(0, Number(snapshot?.elapsed || 0));
  return {
    ...stable,
    // Keep score/PV/proof fields unchanged. Only progress metrics advance.
    nodes,
    elapsed,
    nps: Math.round(nodes * 1000 / Math.max(1, elapsed)),
    selDepth: Math.max(Number(stable.selDepth || 0), Number(snapshot?.selDepth || 0)),
    searchDepth: Math.max(1, Number(requestedDepth || nextDepth)),
    nextDepth: Math.max(1, Number(requestedDepth || nextDepth)),
    nodeTarget: updateDepthNodeEstimate(current, nodes, requestedDepth),
    cacheKey: current.cacheKey,
    cached: false,
    liveProgress: true,
    liveUpdate: true
  };
}

function estimateDepthNodes(state, depth) {
  const requested = Math.max(1, Number(depth || 1));
  const costs = state?.depthNodeCosts || new Map();
  const previous = Math.max(0, Number(costs.get(requested - 1) || 0));
  const beforePrevious = Math.max(0, Number(costs.get(requested - 2) || 0));
  let estimate;
  if (previous > 0 && beforePrevious > 0) {
    const growth = Math.max(1.25, Math.min(3.25, previous / Math.max(1, beforePrevious)));
    estimate = previous * growth;
  } else if (previous > 0) {
    estimate = previous * (requested <= 3 ? 1.55 : 1.85);
  } else if (totalNodes > 0) {
    estimate = Math.max(2_000, totalNodes * 0.45);
  } else {
    estimate = 2_000 * Math.pow(1.62, Math.max(0, requested - 1));
  }
  return Math.max(1_000, Math.min(250_000_000, Math.round(estimate)));
}

function beginDepthNodeEstimate(state, depth) {
  if (!state) return 0;
  const requested = Math.max(1, Number(depth || 1));
  if (state.progressDepth !== requested || state.progressTargetNodes <= totalNodes) {
    state.progressDepth = requested;
    state.progressTargetNodes = totalNodes + estimateDepthNodes(state, requested);
  }
  return state.progressTargetNodes;
}

function updateDepthNodeEstimate(state, visibleNodes, depth) {
  const requested = Math.max(1, Number(depth || state?.progressDepth || 1));
  let target = beginDepthNodeEstimate(state, requested);
  const currentNodes = Math.max(0, Number(visibleNodes || 0));
  if (currentNodes >= target) {
    target = Math.max(currentNodes + 1_000, Math.round(currentNodes * 1.28));
    state.progressTargetNodes = target;
  }
  return target;
}

function postLiveProgress(token, snapshot, requestedDepth) {
  if (!current || token !== activeToken || !running || paused || !snapshot) return;
  // No raw live PV reaches the UI. Until the first completed iteration exists,
  // the panel remains in its existing searching state instead of displaying a
  // one-ply / transient score.
  const live = progressFromStable(current.lastResult, snapshot, requestedDepth);
  if (live) post('info', { token, result: live });
}

function publishExactTablebaseResult(token, result) {
  if (!current || token !== activeToken || !result || !isTrustedExactTablebaseResult(result)) return false;
  const solved = withResultQuality({
    ...result,
    cacheKey: current.cacheKey,
    cached: false,
    solved: true,
    multiPvVerified: true,
    searchDepth: 0,
    nextDepth: 0
  });
  current.lastResult = solved;
  current.tablebasePromotionPending = false;
  current.tablebasePromotionQueued = false;
  current.tablebasePromotionResolved = true;
  post('info', { token, result: solved });
  running = false;
  paused = false;
  post('state', { token, state: 'complete', engine: result.engineLabel || ENGINE_VERSION, depth: 0, searchDepth: 0, tablebase: true });
  return true;
}

async function probeTablebase(token, { announce = true } = {}) {
  if (!current || token !== activeToken) return false;
  try {
    if (announce) post('state', { token, state: 'probing', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    const result = await tablebase.analyze(current.position.clone(), { multipv });
    if (!result || token !== activeToken || !current) return false;
    return publishExactTablebaseResult(token, result);
  } catch {
    return false;
  }
}

function queueExactTablebasePromotion(token) {
  if (!current || token !== activeToken || current.tablebasePromotionQueued || current.tablebasePromotionResolved) return;
  const root = current.position.clone();
  current.tablebasePromotionQueued = true;
  current.tablebasePromotionPending = true;
  void (async () => {
    let promoted = false;
    try {
      const warmed = await tablebase.warmExactWdlNeighborhood(root.clone(), { includeLegalChildren: true });
      if (!current || token !== activeToken) return;
      if (warmed) promoted = await probeTablebase(token, { announce: false });
      if (!promoted && current && token === activeToken) {
        await new Promise(resolve => setTimeout(resolve, TABLEBASE_PROMOTION_RETRY_MS));
        if (current && token === activeToken) promoted = await probeTablebase(token, { announce: false });
      }
    } finally {
      if (!current || token !== activeToken || promoted || isTrustedExactTablebaseResult(current.lastResult)) return;
      current.tablebasePromotionQueued = false;
      current.tablebasePromotionPending = false;
      current.tablebasePromotionResolved = true;
    }
  })();
}

function queueTablebaseFrontierWarmup(token) {
  if (!current || token !== activeToken || current.tablebaseFrontierQueued || current.tablebaseFrontierResolved) return;
  const root = current.position.clone();
  if (Number(root.pieceCount || 0) !== 6) return;
  current.tablebaseFrontierQueued = true;
  void (async () => {
    try {
      const summary = await tablebase.warmExactFrontier(root, {
        maxPly: TABLEBASE_FRONTIER_MAX_PLY,
        maxStates: TABLEBASE_FRONTIER_MAX_STATES,
        priority: 1
      });
      if (current && token === activeToken) current.tablebaseFrontier = summary;
    } finally {
      if (current && token === activeToken) {
        current.tablebaseFrontierQueued = false;
        current.tablebaseFrontierResolved = true;
      }
    }
  })();
}

function queueTablebaseMixedBoundAudit(token, revision, baseline, refinement = 0) {
  if (!current || token !== activeToken || !baseline?.lines?.length) return;
  const root = current.position.clone();
  if (Number(root.pieceCount || 0) !== 6) return;
  const bridgeLine = findBridgeTailLine(baseline, { nonDrawOnly: true });
  if (!bridgeLine?.move) return;
  const rootMove = uciToMove(root, bridgeLine.move);
  if (!rootMove) return;
  const targetDepth = auditTargetDepth(baseline, refinement);
  const auditBudget = TABLEBASE_MIXED_AUDIT_TIME_MS + Math.max(0, Number(refinement || 0)) * 420;
  const bridgeIdentity = Number(bridgeLine.tablebaseTail?.wdl || bridgeLine.tablebaseRawScore || 0);
  const key = `${revision}:${bridgeLine.move}:${Math.sign(bridgeIdentity)}:${targetDepth}:${refinement}`;
  if (current.tablebaseMixedAuditQueued || current.tablebaseMixedAuditQueuedKey === key) return;
  current.tablebaseMixedAuditQueued = true;
  current.tablebaseMixedAuditQueuedKey = key;
  void (async () => {
    try {
      // The normal search keeps the exact ±22000 tablebase sentinel for move
      // ordering and alpha-beta cuts. This independent audit caps it at a
      // finite score, but first makes the reachable exact blocks resident.
      // Starting from the completed root WDL sentinel avoids relying on tail
      // hydration winning a race against the next normal iteration.
      if (!current || token !== activeToken) return;
      if (!currentStillPrefersBridgeMove(bridgeLine.move)) return;
      if (isTablebaseBridgeUpperBound(current.lastResult) || isTablebaseBridgeDraw(current.lastResult)) return;
      await tablebase.warmExactBridgeTables(root, {
        maxPly: TABLEBASE_FRONTIER_MAX_PLY,
        maxStates: TABLEBASE_FRONTIER_MAX_STATES,
        maxBlocks: 36,
        priority: 1,
        seedSignatures: [bridgeLine.tablebaseTail?.signature].filter(Boolean)
      });
      if (!current || token !== activeToken) return;
      if (!currentStillPrefersBridgeMove(bridgeLine.move)) return;
      if (isTablebaseBridgeUpperBound(current.lastResult) || isTablebaseBridgeDraw(current.lastResult)) return;

      const auditor = new GardnerSearcher({ hashEntries: TABLEBASE_MIXED_AUDIT_HASH_ENTRIES });
      auditor.setTablebaseProbe(position => tablebase.probeExactSync(position));
      const audit = auditor.analyze(root.clone(), {
        timeMs: auditBudget,
        maxDepth: targetDepth,
        multipv: 1,
        startDepth: 1,
        historyKeys: current.historyKeys,
        newPosition: true,
        endgameProbeMs: 0,
        fortressProbeMs: 0,
        mateProbeMs: 0,
        tablebaseScoreCap: TABLEBASE_MIXED_AUDIT_SCORE_CAP,
        restrictedRootMoves: [rootMove]
      });
      const auditLine = audit?.lines?.[0];
      // A saturated finite cap means all searched defensive routes still land
      // in a decisive exact-WDL leaf. Publish the finite cap rather than the
      // internal ±22000 sentinel: it is a stable winning/losing evaluation,
      // never a mate claim. A non-saturated audit is stronger still because the
      // opponent's best ordinary branch supplies the displayed score directly.
      if (!audit?.completed || !auditLine || auditLine.move !== bridgeLine.move) return;
      if (Number(audit.tablebaseBoundedProbeHits || 0) <= 0) return;
      if (!current || token !== activeToken) return;
      if (!currentStillPrefersBridgeMove(bridgeLine.move)) return;
      if (isTablebaseBridgeUpperBound(current.lastResult) || isTablebaseBridgeDraw(current.lastResult)) return;

      const sourceLine = currentBridgeLineForMove(bridgeLine, bridgeLine.move);
      const candidate = buildTablebaseMixedAuditResult(root, auditBaselineForMove(baseline, bridgeLine.move), sourceLine, audit);
      if (!candidate) return;
      const existing = current.tablebaseMixedAuditResult;
      const existingDepth = Number(existing?.tablebaseMixedAuditDepth || 0);
      const candidateDepth = Number(candidate.tablebaseMixedAuditDepth || 0);
      if (existing && existingDepth > candidateDepth) return;
      current.tablebaseMixedAuditResult = candidate;
      current.lastResult = candidate;
      post('info', { token, result: candidate });
      // A single deeper audit pass is deliberately bounded. It may discover a
      // lower ordinary defensive score, but it never becomes a mate proof and
      // it never preempts the independent AND/OR bridge certificate.
      if (Number(refinement || 0) === 0 && Number(current.tablebaseMixedAuditImproveAttempts || 0) < 1) {
        current.tablebaseMixedAuditImproveAttempts = Number(current.tablebaseMixedAuditImproveAttempts || 0) + 1;
        setTimeout(() => {
          if (current && token === activeToken && currentStillPrefersBridgeMove(bridgeLine.move)) {
            queueTablebaseMixedBoundAudit(token, revision, baseline, 1);
          }
        }, 18);
      }
    } catch {
      // The audit is advisory and bounded. A cache miss or timeout leaves the
      // existing normal analysis untouched; it is never interpreted as a draw
      // or as a failed win.
    } finally {
      if (current && token === activeToken && current.tablebaseMixedAuditQueuedKey === key) {
        current.tablebaseMixedAuditQueued = false;
      }
    }
  })();
}

function queueTablebaseBridgeProof(token, revision, baseline) {
  if (!current || token !== activeToken || !baseline?.lines?.length) return;
  const root = current.position.clone();
  if (Number(root.pieceCount || 0) !== 6) return;
  // A non-draw tail must reach an actual GTB checkmate; an exact WDL=0
  // entry is also bridgeable, but only the dual-controller prover can promote
  // it to a final 0.00 result.
  const bridgeLine = baseline.lines.find(line => {
    const tail = line?.tablebaseTail;
    return Boolean(tail?.bridgeable || tail?.terminal || tail?.exactWdl)
      && Number.isFinite(Number(tail?.wdl));
  });
  if (!bridgeLine) return;
  const existing = current.tablebaseBridgeResult;
  const existingDtm = Number(existing?.lines?.[0]?.tablebaseBridgeDtm || 0);
  const nextLimit = existingDtm > 1 ? existingDtm - 1 : TABLEBASE_BRIDGE_MAX_PLY;
  const key = `${revision}:${bridgeLine.move || ''}:${existingDtm || 0}:${bridgeLine.tablebaseTail?.wdl || 0}`;
  if (current.tablebaseBridgeQueued || current.tablebaseBridgeQueuedKey === key) return;
  current.tablebaseBridgeQueued = true;
  current.tablebaseBridgeQueuedKey = key;
  const rootWdl = Number(bridgeLine.tablebaseTail?.wdl || 0);
  const preferredMoves = Array.isArray(bridgeLine.pv) ? bridgeLine.pv.slice(0, 16) : [];
  void (async () => {
    try {
      // Fetch/decompress candidate GTB families before proof construction.
      // The proof then uses resident exact blocks only, so its time budget is
      // deterministic and never waits on network I/O.
      await tablebase.warmExactBridgeTables(root, {
        maxPly: TABLEBASE_FRONTIER_MAX_PLY,
        maxStates: TABLEBASE_FRONTIER_MAX_STATES,
        maxBlocks: 36,
        priority: 1,
        seedSignatures: [bridgeLine.tablebaseTail?.signature].filter(Boolean)
      });
      if (!current || token !== activeToken) return;
      let proof = null;
      if (rootWdl === 0) {
        proof = await tablebase.proveExactBridgeDraw(root, {
          preferredMoves,
          maxPlies: Math.min(TABLEBASE_BRIDGE_MAX_PLY, nextLimit),
          maxNodes: TABLEBASE_BRIDGE_MAX_NODES,
          timeMs: TABLEBASE_BRIDGE_TIME_MS,
          controllerMoveLimit: 4,
          priority: 1
        });
      } else {
        const controller = rootWdl > 0 ? root.turn : -root.turn;
        proof = await tablebase.proveExactBridgeOutcome(root, {
          controller,
          outcome: 'win',
          preferredMoves,
          maxPlies: Math.min(TABLEBASE_BRIDGE_MAX_PLY, nextLimit),
          maxNodes: TABLEBASE_BRIDGE_MAX_NODES,
          timeMs: existingDtm > 1 ? TABLEBASE_BRIDGE_IMPROVE_TIME_MS : TABLEBASE_BRIDGE_TIME_MS,
          controllerMoveLimit: 4,
          priority: 1
        });
      }
      if (!proof || !current || token !== activeToken) return;
      const candidate = buildTablebaseBridgeResult(root, baseline, proof);
      const incumbent = current.lastResult;
      const chosen = compareAnalysisResults(incumbent, candidate, { preferNextOnTie: true });
      // Bridge bounds are root certificates, not speculative PV annotations.
      // A shorter newly found bound replaces an older one; ordinary analysis can
      // never overwrite it, while a stronger exact mate proof still can.
      if (chosen !== candidate && !(candidate.tablebaseBridgeProof && Number(candidate.tablebaseBridgeDtm || Infinity) < Number(incumbent?.tablebaseBridgeDtm || Infinity))) return;
      current.tablebaseBridgeResult = candidate;
      current.lastResult = candidate;
      post('info', { token, result: candidate });
      // One bounded refinement pass starts from the newly established upper
      // bound.  It may lower M≤N, while the ordinary mate prover continues to
      // seek a shortest/exact result in parallel with later iterations.
      if (candidate.tablebaseBridgeProof && Number(candidate.tablebaseBridgeDtm || 0) > 1
        && Number(current.tablebaseBridgeImproveAttempts || 0) < 1) {
        current.tablebaseBridgeImproveAttempts = Number(current.tablebaseBridgeImproveAttempts || 0) + 1;
        setTimeout(() => {
          if (current && token === activeToken) queueTablebaseBridgeProof(token, revision + 10_000, baseline);
        }, 16);
      }
    } catch {
      // A failed budget/GTB request is not a negative conclusion. Normal
      // analysis remains authoritative and a later completed depth can retry.
    } finally {
      if (current && token === activeToken && current.tablebaseBridgeQueuedKey === key) {
        current.tablebaseBridgeQueued = false;
      }
    }
  })();
}

function queueTablebaseTailHydration(token, revision, baseline) {
  if (!current || token !== activeToken || !baseline?.lines?.length || baseline.tablebase || baseline.fortressProof) return;
  if (current.tablebaseTailQueuedRevision === revision) return;
  const root = current.position.clone();
  current.tablebaseTailQueuedRevision = revision;
  void (async () => {
    try {
      const hydrated = await tablebase.extendResultWithExactTablebaseTails(root, baseline, {
        maxLines: multipv,
        maxProbePly: 36,
        maxTailPly: TABLEBASE_TAIL_MAX_PLY
      });
      // A tail hydration may only amend the exact GTB continuation of the same
      // stable iteration. It can never replace the score, proof flags or root
      // result after a newer depth has arrived.
      if (!hydrated?.tablebaseTailHydrated || !current || token !== activeToken) return;
      if (current.analysisRevision !== revision || current.lastResult !== baseline) return;
      current.lastResult = hydrated;
      post('info', { token, result: hydrated });
      queueTablebaseMixedBoundAudit(token, revision, hydrated);
      queueTablebaseBridgeProof(token, revision, hydrated);
    } catch {
      // GTB enrichment is opportunistic. The completed alpha-beta result remains
      // valid if a block is unavailable or a browser aborts a background fetch.
    } finally {
      if (current && token === activeToken && current.tablebaseTailQueuedRevision === revision) {
        current.tablebaseTailQueuedRevision = -1;
      }
    }
  })();
}

async function startFairyPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.FAIRY;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  current = { position, cacheKey, lastResult: null, bookMoves: [], historyKeys: [] };
  running = true;
  paused = false;
  effortMs = Math.max(200, Math.min(30000, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  post('state', { token, state: 'thinking', engine: FAIRY_STOCKFISH_LABEL, depth: 0, searchDepth: 0 });
  try {
    const raw = await fairyProvider.search({ token, fen: String(message.fen || '').trim(), timeMs: effortMs, multipv });
    if (token !== activeToken || !current) return;
    const result = validateExternalAnalysisResult(position, raw, { maxLines: multipv });
    if (!result) throw new Error('Fairy-Stockfish returned no fully legal Gardner PV.');
    const finalResult = withResultQuality(sortResultLinesForSide({
      ...result,
      cacheKey,
      cached: false,
      rootTurn: position.turn,
      searchDepth: 0,
      nextDepth: 0,
      completed: true,
      multiPvVerified: true,
      solved: false
    }, position.turn, multipv));
    current.lastResult = finalResult;
    running = false;
    paused = false;
    post('info', { token, result: finalResult });
    post('state', { token, state: 'complete', engine: FAIRY_STOCKFISH_LABEL, depth: finalResult.depth, searchDepth: 0 });
  } catch (error) {
    if (token !== activeToken) return;
    await startOrionPosition({ ...message, kernel: ENGINE_KERNELS.ORION });
  }
}

async function startOrionPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.ORION;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  effortMs = Math.max(200, Math.min(2400, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  const resumeResult = isTrustedResume(position, message.resumeResult);
  const tablebaseEligible = Number(position.pieceCount || 0) <= 5;
  current = {
    position,
    cacheKey,
    lastResult: resumeResult,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen),
    depthNodeCosts: new Map(),
    progressDepth: 0,
    progressTargetNodes: 0,
    tablebasePromotionPending: false,
    tablebasePromotionQueued: false,
    tablebasePromotionResolved: false,
    tablebaseFrontierQueued: false,
    tablebaseFrontierResolved: false,
    tablebaseFrontier: null,
    analysisRevision: 0,
    tablebaseTailQueuedRevision: -1,
    tablebaseBridgeQueued: false,
    tablebaseBridgeQueuedKey: '',
    tablebaseBridgeResult: null,
    tablebaseBridgeImproveAttempts: 0,
    tablebaseMixedAuditQueued: false,
    tablebaseMixedAuditQueuedKey: '',
    tablebaseMixedAuditResult: null,
    tablebaseMixedAuditImproveAttempts: 0
  };
  nextDepth = 1;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = 0;
  totalElapsed = 0;
  paused = Boolean(message.startPaused);
  const solvedResume = isSolvedResult(resumeResult);
  running = !paused && !solvedResume;

  if (resumeResult?.lines?.length) {
    post('info', { token, result: { ...resumeResult, cacheKey, cached: true, searchDepth: 0, nextDepth: 0 } });
  }
  post('state', {
    token,
    state: paused ? 'paused' : solvedResume ? 'complete' : 'thinking',
    engine: resumeResult?.engineLabel || ENGINE_VERSION,
    depth: Number(resumeResult?.depth || 0),
    searchDepth: solvedResume ? 0 : nextDepth
  });
  if (!running) return;
  if (tablebaseEligible) {
    if (await probeTablebase(token, { announce: true })) return;
    if (token !== activeToken || !current) return;
    queueExactTablebasePromotion(token);
  } else {
    // v19.8: for a 6-piece root, preload first tablebase-entry leaves without
    // delaying the first iterative chunk. Later alpha-beta nodes then terminate
    // directly from GTB instead of re-searching the resolved ending.
    queueTablebaseFrontierWarmup(token);
  }
  if (token === activeToken && running && !paused) schedule(token);
}

async function startPosition(message) {
  return selectedKernel(message.kernel) === ENGINE_KERNELS.FAIRY ? startFairyPosition(message) : startOrionPosition(message);
}

async function runChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const mateBudget = Math.min(520, Math.max(70, Math.round(currentBudgetMs * 0.28)));
    const mainBudget = Math.max(70, currentBudgetMs - mateBudget);
    const raw = searcher.analyze(current.position.clone(), {
      timeMs: mainBudget,
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      bookMoves: current.bookMoves,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
      endgameProbeMs: 70,
      fortressProbeMs: 150,
      mateProbeMs: mateBudget,
      mateMaxPlies: 81,
      onProgress: snapshot => postLiveProgress(token, snapshot, requestedDepth)
    });
    firstChunk = false;
    if (!running || paused || token !== activeToken) return;

    const chunkNodes = Math.max(0, Number(raw.nodes || 0));
    totalNodes += chunkNodes;
    totalElapsed += Math.max(0, Number(raw.elapsed || 0));
    if (raw.completed) {
      if (Number(raw.depth || 0) > 0) current.depthNodeCosts.set(Number(raw.depth || 0), chunkNodes);
      nextDepth = Math.max(requestedDepth + 1, Number(raw.nextDepth || requestedDepth + 1));
      currentBudgetMs = initialBudget(nextDepth);
      current.progressDepth = 0;
      beginDepthNodeEstimate(current, nextDepth);
    } else {
      currentBudgetMs = Math.min(Math.max(effortMs, 320), Math.round(currentBudgetMs * 1.5));
      updateDepthNodeEstimate(current, totalNodes, requestedDepth);
    }

    const profile = resultPvProfile(raw);
    const cumulative = withResultQuality({
      ...raw,
      ...profile,
      nodes: totalNodes,
      elapsed: totalElapsed,
      nps: Math.round(totalNodes * 1000 / Math.max(1, totalElapsed)),
      scoreDepth: Number(raw.scoreDepth || raw.depth || 0),
      searchDepth: nextDepth,
      searchBudget: currentBudgetMs,
      nodeTarget: updateDepthNodeEstimate(current, totalNodes, nextDepth),
      rootTurn: Number(raw.rootTurn || current.position.turn),
      lowProgressAudit: Boolean(raw.lowProgressAudit),
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(raw)
    });

    // Atomic publication: never merge PVs, proof labels or scores from prior
    // and current iterations. A partial depth only refreshes stable metrics.
    let visible = null;
    if (isStableSearchResult(cumulative)) {
      // A bridge certificate outranks ordinary alpha-beta output.  Keep the
      // engine searching for a shorter/exact mate, but never let a later raw
      // +220-style WDL sentinel overwrite the certified upper bound.
      const chosen = compareAnalysisResults(current.lastResult, cumulative, { preferNextOnTie: true });
      if (chosen === cumulative) {
        current.lastResult = cumulative;
        current.tablebaseMixedAuditResult = null;
        current.tablebaseMixedAuditImproveAttempts = 0;
        current.analysisRevision += 1;
        visible = cumulative;
        // The finite audit may start directly from a completed root WDL
        // sentinel. This makes it robust even when asynchronous PV-tail
        // hydration is superseded by the next normal iteration.
        queueTablebaseMixedBoundAudit(token, current.analysisRevision, cumulative);
        queueTablebaseTailHydration(token, current.analysisRevision, cumulative);
      } else {
        visible = progressFromStable(current.lastResult, cumulative, requestedDepth);
      }
    } else {
      visible = progressFromStable(current.lastResult, cumulative, requestedDepth);
    }
    if (visible) post('info', { token, result: visible });

    const stable = current.lastResult;
    const bridgeUpperBound = isTablebaseBridgeUpperBound(stable);
    const mateFound = Boolean(stable?.lines?.[0]?.mateVerified && stable?.solved && !bridgeUpperBound);
    // A bridge mate bound is intentionally not terminal for the iterative
    // engine: subsequent depths and the independent mate prover may tighten it
    // to a smaller bound or an exact mate.  A dual-controller draw proof is a
    // final 0.00 result and may stop normally.
    if (stable?.terminal || stable?.fortressProof || mateFound || (isSolvedResult(stable) && !bridgeUpperBound) || nextDepth > MAX_DEPTH) {
      running = false;
      post('state', { token, state: 'complete', engine: ENGINE_VERSION, depth: Number(stable?.depth || raw.depth || 0), searchDepth: nextDepth });
      return;
    }
    post('state', { token, state: 'thinking', engine: ENGINE_VERSION, depth: Number(stable?.depth || 0), searchDepth: nextDepth });
    schedule(token, 7);
  } catch (error) {
    running = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

self.addEventListener('message', event => {
  const message = event.data || {};
  if (message.type === 'start' || message.type === 'position') {
    void startPosition(message).catch(error => {
      running = false;
      post('error', { token: Number(message.token || activeToken), message: error?.stack || error?.message || String(error) });
    });
    return;
  }
  if (message.type === 'pause') {
    if (Number(message.token) !== activeToken || !current) return;
    paused = true;
    running = false;
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    post('state', { token: activeToken, state: 'paused', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    return;
  }
  if (message.type === 'resume') {
    if (Number(message.token) !== activeToken || !current) return;
    if (isSolvedResult(current.lastResult) && !isTablebaseBridgeUpperBound(current.lastResult)) {
      paused = false;
      running = false;
      post('state', { token: activeToken, state: 'complete', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
      return;
    }
    paused = false;
    running = true;
    post('state', { token: activeToken, state: 'thinking', engine: ENGINE_VERSION, depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    schedule(activeToken);
    return;
  }
  if (message.type === 'stop') {
    activeToken = Number(message.token || activeToken + 1);
    running = false;
    paused = false;
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    current = null;
    post('state', { token: activeToken, state: 'idle', engine: ENGINE_VERSION });
    return;
  }
  if (message.type === 'clear') {
    // Clear search accelerators only on an explicit user reset. A normal root
    // transition retains TT/eval ordering but never retains analysis conclusions.
    searcher.clear();
    fairyProvider.stop();
    nextDepth = 1;
    currentBudgetMs = initialBudget(1);
    post('state', { token: activeToken, state: paused ? 'paused' : running ? 'thinking' : 'idle', engine: ENGINE_VERSION });
  }
});

post('ready', { engine: ENGINE_VERSION });
