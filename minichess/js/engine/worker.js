import { EnginePosition, GardnerSearcher, ENGINE_VERSION, EngineInternals, generateLegalMoves, moveToUci, validateMateResult, uciToMove } from './engine.js';
import { MinifishSearcher, MINIFISH_VERSION } from './minifish.js';
import { GardnerTablebase } from './tablebase.js';
import { compareAnalysisResults, isSolvedResult, isTrustedExactTablebaseResult, resultPvProfile, withResultQuality } from './result-quality.js';
import { ENGINE_KERNELS, FAIRY_STOCKFISH_LABEL, FairyStockfishProvider, selectedKernel, validateExternalAnalysisResult } from './external-engine.js';

// v21 analysis worker
// Result ownership rule: every published score/PV pair comes from one completed
// iteration (or one exact proof). Incomplete chunks may update progress only.
const MAX_DEPTH = 48;
const { makeMove, undoMove } = EngineInternals;
const searcher = new GardnerSearcher({ hashEntries: 786432 });
const minifish = new MinifishSearcher();
const tablebase = new GardnerTablebase();
const fairyProvider = new FairyStockfishProvider({
  onState: message => {
    if (Number(message.token || 0) === activeToken) post('state', { ...message, engine: FAIRY_STOCKFISH_LABEL });
  }
});
searcher.setTablebaseProbe(position => tablebase.probeSync(position));
minifish.setTablebaseProbe(position => tablebase.probeSync(position));
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
const TABLEBASE_WIDE_BRIDGE_TIME_MS = 2_600;
const TABLEBASE_WIDE_BRIDGE_MAX_NODES = 1_000_000;
const TABLEBASE_WIDE_BRIDGE_MAX_STATES = 5_000;
const TABLEBASE_WIDE_BRIDGE_MAX_BLOCKS = 128;
const TABLEBASE_WIDE_BRIDGE_TIME_SLICES = [360, 850, 1_700, TABLEBASE_WIDE_BRIDGE_TIME_MS];
const TABLEBASE_WIDE_BRIDGE_NODE_SLICES = [120_000, 300_000, 700_000, TABLEBASE_WIDE_BRIDGE_MAX_NODES];
// v20.5 display policy: the UI may show only a numeric score or a verified
// mate bound. Interior tablebase WDL hits remain proof seeds; if they are not
// converted into a verified bridge/root tablebase/mate result, the worker keeps
// the previous completed real score visible instead of publishing a TB label.
const historyFenKeyCache = new Map();
const HISTORY_FEN_KEY_CACHE_LIMIT = 256;
const VERIFIED_BRIDGE_PROOF_CACHE_LIMIT = 18;
const VERIFIED_BRIDGE_SUBTREE_CACHE_LIMIT = 32768;
// Full AND/OR bridge certificates live in the worker, separate from the
// ordinary result cache. They keep every defending reply and matching policy
// answer available for later refinement. A shallow subtree index lets a played
// variation reuse the already verified branch without re-discovering it.
const verifiedBridgeProofCache = new Map();
const verifiedBridgeSubtreeCache = new Map();

function post(type, payload = {}) {
  self.postMessage({ type, ...payload });
}

function activeEngineLabel() {
  if (currentKernel === ENGINE_KERNELS.MINIFISH) return MINIFISH_VERSION;
  if (currentKernel === ENGINE_KERNELS.FAIRY) return FAIRY_STOCKFISH_LABEL;
  return ENGINE_VERSION;
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

function bridgeProofCacheKey(cacheKey) {
  return `${ENGINE_VERSION}|${String(cacheKey || '')}`;
}

function bridgeSubtreeCacheKey(position) {
  return `${ENGINE_VERSION}|${position?.key?.() || ''}|hm${Number(position?.halfmove || 0)}`;
}

function cacheBridgeProofSubtrees(root, proof) {
  if (!root?.clone || !proof?.proof || proof.tablebaseBridgeDraw) return;
  const controller = Number(proof.controller || root.turn);
  const outcome = 'win';
  const seen = new Set();
  const put = (position, node, depth) => {
    if (!position || !node || depth > 10) return;
    const key = bridgeSubtreeCacheKey(position);
    if (!key || seen.has(`${key}:${depth}`)) return;
    seen.add(`${key}:${depth}`);
    const record = {
      tablebaseBridgeProof: true,
      tablebaseBridgeDraw: false,
      controller,
      wdl: controller === position.turn ? 1 : -1,
      dtmPly: Math.max(0, Number(node.distance || 0)),
      exactDtm: false,
      upperBound: true,
      proof: node,
      proofNodes: Number(proof.proofNodes || 0),
      proofLeaves: Number(proof.proofLeaves || 0),
      elapsed: Number(proof.elapsed || 0),
      cachedAt: Date.now(),
      usedAt: Date.now()
    };
    verifiedBridgeSubtreeCache.delete(key);
    verifiedBridgeSubtreeCache.set(key, record);
    while (verifiedBridgeSubtreeCache.size > VERIFIED_BRIDGE_SUBTREE_CACHE_LIMIT) {
      const oldest = verifiedBridgeSubtreeCache.keys().next().value;
      if (oldest === undefined) break;
      verifiedBridgeSubtreeCache.delete(oldest);
    }
    if (node.kind === 'choice' && node.move) {
      const state = makeMove(position, node.move);
      put(position, node.child, depth + 1);
      undoMove(position, node.move, state);
    } else if (node.kind === 'all' && Array.isArray(node.children)) {
      for (const entry of node.children) {
        if (!entry?.move || !entry?.child) continue;
        const state = makeMove(position, entry.move);
        put(position, entry.child, depth + 1);
        undoMove(position, entry.move, state);
      }
    }
  };
  put(root.clone(), proof.proof, 0);
}

function getVerifiedBridgeSubtree(position) {
  const key = bridgeSubtreeCacheKey(position);
  const record = verifiedBridgeSubtreeCache.get(key);
  if (!record) return null;
  record.usedAt = Date.now();
  verifiedBridgeSubtreeCache.delete(key);
  verifiedBridgeSubtreeCache.set(key, record);
  return record;
}

function bridgeProofTreeSize(node, seen = new Set()) {
  if (!node || typeof node !== 'object' || seen.has(node)) return 0;
  seen.add(node);
  if (node.kind === 'choice') return 1 + bridgeProofTreeSize(node.child, seen);
  if (node.kind === 'all') return 1 + (node.children || []).reduce((sum, entry) => sum + bridgeProofTreeSize(entry?.child, seen), 0);
  return 1;
}

function cacheVerifiedBridgeProof(cacheKey, root, proof) {
  if (!cacheKey || !root || !proof?.proof) return false;
  const outcome = proof.tablebaseBridgeDraw ? 'draw' : 'win';
  // The proof was constructed from resident exact blocks.  Validate once at
  // the cache boundary so a later UI/worker path can never retain a partial PV
  // masquerading as a complete win/draw/loss certificate.
  if (!tablebase.verifyExactBridgeProof(root.clone(), proof, {
    controller: Number(proof.controller || root.turn),
    outcome
  })) return false;
  const key = bridgeProofCacheKey(cacheKey);
  const snapshot = {
    tablebaseBridgeProof: Boolean(proof.tablebaseBridgeProof),
    tablebaseBridgeDraw: Boolean(proof.tablebaseBridgeDraw),
    controller: Number(proof.controller || root.turn),
    wdl: Number(proof.wdl || 0),
    dtmPly: Math.max(0, Number(proof.dtmPly || 0)),
    exactDtm: Boolean(proof.exactDtm),
    upperBound: Boolean(proof.upperBound),
    pv: Array.isArray(proof.pv) ? proof.pv.slice() : [],
    proof: proof.proof,
    drawStrategies: proof.drawStrategies || null,
    proofNodes: Math.max(0, Number(proof.proofNodes || 0)),
    proofLeaves: Math.max(0, Number(proof.proofLeaves || 0)),
    elapsed: Math.max(0, Number(proof.elapsed || 0)),
    treeNodes: bridgeProofTreeSize(proof.proof),
    cachedAt: Date.now(),
    usedAt: Date.now()
  };
  verifiedBridgeProofCache.delete(key);
  verifiedBridgeProofCache.set(key, snapshot);
  while (verifiedBridgeProofCache.size > VERIFIED_BRIDGE_PROOF_CACHE_LIMIT) {
    const oldest = verifiedBridgeProofCache.keys().next().value;
    if (oldest === undefined) break;
    verifiedBridgeProofCache.delete(oldest);
  }
  cacheBridgeProofSubtrees(root, snapshot);
  return true;
}

function getVerifiedBridgeProof(cacheKey) {
  const key = bridgeProofCacheKey(cacheKey);
  const record = verifiedBridgeProofCache.get(key);
  if (!record) return null;
  record.usedAt = Date.now();
  verifiedBridgeProofCache.delete(key);
  verifiedBridgeProofCache.set(key, record);
  return record;
}

function bridgeSeedBaseline(root, proof) {
  const pv = Array.isArray(proof?.pv) ? proof.pv.slice() : [];
  return withResultQuality({
    engine: ENGINE_VERSION,
    engineLabel: ENGINE_VERSION,
    depth: 0,
    scoreDepth: 0,
    selDepth: 0,
    nodes: 0,
    elapsed: 0,
    nps: 0,
    rootTurn: Number(root?.turn || 1),
    completed: true,
    multiPvVerified: true,
    pvComplete: true,
    lines: [{
      move: pv[0] || '',
      score: 0,
      scoreText: '0.00',
      pv,
      rootScoreExact: true,
      pvComplete: true
    }]
  });
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


function relabelTablebaseResult(result, engineLabel) {
  if (!result) return result;
  return withResultQuality({
    ...result,
    engine: engineLabel,
    engineLabel: `${engineLabel} + GTB`,
    lines: Array.isArray(result.lines) ? result.lines.map(line => ({
      ...line,
      sourceEngine: result.engine || line.sourceEngine || '',
      scoreKind: line.scoreKind || 'exact-tablebase',
      tablebase: true,
      tablebaseScope: line.tablebaseScope || 'root-exact'
    })) : []
  });
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

function lineHasPublishableScore(line) {
  if (!line) return false;
  if (line.mateVerified || line.tablebaseExactRoot || line.tablebase || line.fortressProof) return true;
  if (line.scoreNumeric === false || line.unverifiedMate || line.matePendingUnscored) return false;
  return Number.isFinite(Number(line.score));
}

function isStableSearchResult(result) {
  if (!result?.lines?.length) return false;
  if (isSolvedResult(result)) return true;
  if (
    result.completed === false ||
    result.pvComplete === false ||
    result.pvIncomplete ||
    result.multiPvVerified === false
  ) return false;
  // v20.5: a completed iteration is publishable only when every visible line is
  // either a real numeric score or a verified mate/tablebase/fortress result.
  // Interior WDL seeds and unverified mate candidates remain internal proof
  // material; they must never replace a visible real score with a blank label.
  const visibleCount = Math.max(1, Math.min(Number(result.multipv || multipv || 1), result.lines.length));
  return result.lines.slice(0, visibleCount).every(lineHasPublishableScore);
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
      const wdl = Number(tail.wdl);
      return Number.isFinite(wdl) && (!nonDrawOnly || wdl !== 0);
    }
    // v20.1: a raw WDL sentinel may seed a non-draw bridge proof, but it must
    // never fabricate a draw proof.  Draw bridge attempts require a hydrated
    // exact tail with wdl === 0 so that missing/zero fallback data cannot hide
    // an ordinary search evaluation.
    const raw = Number(line?.tablebaseRawScore || 0);
    return Boolean(line?.tablebaseBridgeCandidate)
      && Number.isFinite(raw)
      && raw !== 0
      && (!nonDrawOnly || raw !== 0);
  }) || null;
}

function bridgeRootWdl(root, line) {
  if (!root || !line) return NaN;
  if (line.tablebaseTail && Number.isFinite(Number(line.tablebaseTail.wdl))) {
    return Math.sign(Number(line.tablebaseTail.wdl));
  }
  const rawWhite = Math.sign(Number(line.tablebaseRawScore || 0));
  if (!rawWhite) return NaN;
  return Number(root.turn || 1) === 1 ? rawWhite : -rawWhite;
}

function bridgeAttemptKey(move, rootWdl, existingDtm = 0) {
  return `${String(move || '')}:${Math.sign(Number(rootWdl || 0))}:dtm${Math.max(0, Number(existingDtm || 0))}`;
}

function bridgeAttemptCount(state, key) {
  return Number(state?.tablebaseBridgeAttemptsByKey?.get(key) || 0);
}

function bumpBridgeAttempt(state, key) {
  if (!state?.tablebaseBridgeAttemptsByKey) return 0;
  const next = bridgeAttemptCount(state, key) + 1;
  state.tablebaseBridgeAttemptsByKey.set(key, next);
  return next;
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
      scoreKind: 'tablebase-bridge-draw',
      scoreNumeric: true,
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
    scoreKind: 'tablebase-bridge-mate-upper-bound',
    scoreNumeric: true,
    pv: Array.isArray(proof.pv) ? proof.pv.slice() : [],
    mateVerified: true,
    mateUpperBound: true,
    tablebaseBridgeProof: true,
    tablebaseBridgeDraw: false,
    tablebaseBridgeDtm: dtm,
    tablebaseBridgeController: Number(proof?.controller || 0),
    tablebaseScope: 'bridge-proof',
    // A bridge proof guarantees mate within this bound, but it does not claim
    // shortest DTM.  Keep the exactness flag false so score consumers do not
    // treat ≤#N as an exact root mate distance.
    rootScoreExact: false,
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

function stabilizePublishedMetrics(state, result) {
  if (!state || !result) return result;
  const nodes = Math.max(0, Number(result.nodes || 0));
  const elapsed = Math.max(0, Number(result.elapsed || 0));
  const priorNodes = Math.max(0, Number(state.maxPublishedNodes || 0));
  const priorElapsed = Math.max(0, Number(state.maxPublishedElapsed || 0));
  const visibleNodes = Math.max(priorNodes, nodes);
  const visibleElapsed = Math.max(priorElapsed, elapsed);
  const proposedTarget = Math.max(0, Number(result.nodeTarget || 0));
  let target = Math.max(proposedTarget, Number(state.maxPublishedNodeTarget || 0));
  // The denominator is an extendable depth+1 work target.  Never reset it to a
  // smaller per-chunk estimate as iterative deepening advances; once crossed,
  // extend it above the current cumulative work rather than showing a lower
  // nodes/target pair.
  if (target && target <= visibleNodes) {
    target = Math.max(target, visibleNodes + Math.max(1_000, Math.round(visibleNodes * 0.18)));
  }
  state.maxPublishedNodes = visibleNodes;
  state.maxPublishedElapsed = visibleElapsed;
  state.maxPublishedNodeTarget = target;
  return {
    ...result,
    nodes: visibleNodes,
    elapsed: visibleElapsed,
    nps: Math.round(visibleNodes * 1000 / Math.max(1, visibleElapsed)),
    nodeTarget: target
  };
}

function progressFromStable(stable, snapshot, requestedDepth) {
  if (!stable?.lines?.length) return null;
  const nodes = totalNodes + Math.max(0, Number(snapshot?.nodes || 0));
  const elapsed = totalElapsed + Math.max(0, Number(snapshot?.elapsed || 0));
  return stabilizePublishedMetrics(current, {
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
  });
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
  const cumulativeFloor = Math.max(totalNodes, Number(state.maxPublishedNodes || 0));
  if (state.progressDepth !== requested || state.progressTargetNodes <= cumulativeFloor) {
    state.progressDepth = requested;
    state.progressTargetNodes = Math.max(
      Number(state.progressTargetNodes || 0),
      Number(state.maxPublishedNodeTarget || 0),
      cumulativeFloor + estimateDepthNodes(state, requested)
    );
  }
  return state.progressTargetNodes;
}

function updateDepthNodeEstimate(state, visibleNodes, depth) {
  const requested = Math.max(1, Number(depth || state?.progressDepth || 1));
  let target = beginDepthNodeEstimate(state, requested);
  const currentNodes = Math.max(0, Number(visibleNodes || 0), Number(state?.maxPublishedNodes || 0));
  if (currentNodes >= target) {
    target = Math.max(
      target,
      currentNodes + 1_000,
      Math.round(currentNodes * 1.28),
      Number(state?.maxPublishedNodeTarget || 0)
    );
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
  current.lastResult = stabilizePublishedMetrics(current, solved);
  current.tablebasePromotionPending = false;
  current.tablebasePromotionQueued = false;
  current.tablebasePromotionResolved = true;
  post('info', { token, result: current.lastResult });
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
  const attempts = Math.max(0, Number(current.tablebasePromotionAttempts || 0));
  if (attempts >= 4) {
    current.tablebasePromotionResolved = true;
    return;
  }
  current.tablebasePromotionAttempts = attempts + 1;
  current.tablebasePromotionQueued = true;
  current.tablebasePromotionPending = true;
  void (async () => {
    let promoted = false;
    try {
      const warmed = await tablebase.warmExactWdlNeighborhood(root.clone(), { includeLegalChildren: true });
      if (!current || token !== activeToken) return;
      if (warmed) promoted = await probeTablebase(token, { announce: false });
      if (!promoted && current && token === activeToken) {
        await new Promise(resolve => setTimeout(resolve, TABLEBASE_PROMOTION_RETRY_MS * Math.min(3, current.tablebasePromotionAttempts || 1)));
        if (current && token === activeToken) promoted = await probeTablebase(token, { announce: false });
      }
    } finally {
      if (!current || token !== activeToken || promoted || isTrustedExactTablebaseResult(current.lastResult)) return;
      current.tablebasePromotionQueued = false;
      current.tablebasePromotionPending = false;
      // Do not mark a transient miss as permanently resolved.  A later completed
      // depth can retry root exact promotion after async tablebase requests settle.
      if (Number(current.tablebasePromotionAttempts || 0) >= 4) current.tablebasePromotionResolved = true;
    }
  })();
}

async function tryCachedBridgeSubtree(token) {
  if (!current || token !== activeToken) return false;
  const root = current.position.clone();
  const cached = getVerifiedBridgeSubtree(root);
  if (!cached) return false;
  if (!tablebase.verifyExactBridgeProof(root, cached, {
    controller: Number(cached.controller || root.turn),
    outcome: 'win'
  })) return false;
  try {
    const pv = await tablebase.buildBridgePrincipalVariation(root, cached.proof, {
      priority: 0,
      maxTailPly: TABLEBASE_TAIL_MAX_PLY
    });
    if (!pv?.length || !current || token !== activeToken) return false;
    const proof = { ...cached, pv };
    const baseline = current.lastResult || bridgeSeedBaseline(root, proof);
    const candidate = stabilizePublishedMetrics(current, buildTablebaseBridgeResult(root, baseline, proof));
    current.tablebaseBridgeResult = candidate;
    current.lastResult = candidate;
    post('info', { token, result: candidate });
    return true;
  } catch {
    return false;
  }
}

// Fast path for a six-piece root that can immediately capture/promote into an
// exact tablebase. It is a real one-move AND/OR certificate only when the side
// to move itself can choose the exact winning entry. This prevents a child such
// as Kb1 ...a2 from first displaying a stale ordinary negative score before
// Kxa2 is recognized as a database-backed win.
async function tryImmediateExactEntryBridge(token) {
  if (!current || token !== activeToken || current.tablebaseImmediateEntryQueued) return false;
  const root = current.position.clone();
  if (Number(root.pieceCount || 0) !== 6) return false;
  current.tablebaseImmediateEntryQueued = true;
  try {
    const moves = generateLegalMoves(root, false);
    let preferred = null;
    for (const move of moves) {
      const child = root.clone();
      const childMove = uciToMove(child, moveToUci(move));
      if (!childMove) continue;
      makeMove(child, childMove);
      if (Number(child.pieceCount || 0) > 5) continue;
      const probe = await tablebase.probe(child, { priority: 0 });
      if (!probe || probe.dtmUpperBound || Number(probe.wdl || 0) === 0) continue;
      const winner = probe.wdl > 0 ? child.turn : -child.turn;
      if (winner !== current.position.turn) continue;
      preferred = moveToUci(move);
      break;
    }
    if (!preferred || !current || token !== activeToken) return false;
    const proof = await tablebase.proveExactBridgeOutcome(current.position.clone(), {
      controller: current.position.turn,
      outcome: 'win',
      preferredMoves: [preferred],
      maxPlies: Math.min(TABLEBASE_BRIDGE_MAX_PLY, 32),
      maxNodes: 6_000,
      timeMs: 420,
      controllerMoveLimit: 1,
      priority: 0
    });
    if (!proof || !current || token !== activeToken) return false;
    if (!cacheVerifiedBridgeProof(current.cacheKey, current.position.clone(), proof)) return false;
    const cachedProof = getVerifiedBridgeProof(current.cacheKey) || proof;
    const baseline = current.lastResult || bridgeSeedBaseline(current.position, cachedProof);
    const candidate = stabilizePublishedMetrics(
      current,
      buildTablebaseBridgeResult(current.position, baseline, cachedProof)
    );
    const chosen = compareAnalysisResults(current.lastResult, candidate, { preferNextOnTie: true });
    if (chosen === candidate || !current.lastResult) {
      current.tablebaseBridgeResult = candidate;
      current.lastResult = candidate;
      post('info', { token, result: candidate });
      return true;
    }
  } catch {
    // A direct-entry miss is not a negative conclusion. Normal analysis and
    // the broader bridge prover continue unchanged.
  } finally {
    if (current && token === activeToken) current.tablebaseImmediateEntryQueued = false;
  }
  return false;
}


function bridgePreferredMovesFromResult(result, limit = 16) {
  const first = Array.isArray(result?.lines) ? result.lines[0] : null;
  return Array.isArray(first?.pv) ? first.pv.slice(0, Math.max(1, Number(limit || 16))) : [];
}

function queueWideTablebaseBridgeProof(token, revision = 0, baseline = null, { delay = 0 } = {}) {
  if (!current || token !== activeToken || current.tablebaseWideBridgeQueued || current.tablebaseWideBridgeResolved) return;
  const root = current.position.clone();
  // v20.4: the wide 6->5 tablebase bridge is useful but comparatively heavy.
  // Keep it to six-piece roots and run it as progressive side work; seven-piece
  // pawn endings are handled by the short full-width mate prover plus ordinary
  // alpha-beta so a failed broad bridge cannot starve the main search.
  if (Number(root.pieceCount || 0) !== 6) return;

  const attempt = Math.max(0, Number(current.tablebaseWideBridgeAttempts || 0));
  const incumbentHasMate = Boolean(current.lastResult?.lines?.[0]?.mateVerified || current.tablebaseBridgeResult);
  const maxAttempts = 4;
  if (attempt >= maxAttempts) {
    current.tablebaseWideBridgeResolved = !incumbentHasMate;
    return;
  }

  current.tablebaseWideBridgeQueued = true;
  current.tablebaseWideBridgeAttempts = attempt + 1;
  const sliceIndex = Number(root.pieceCount || 0) === 6 && !incumbentHasMate && attempt === 0
    ? TABLEBASE_WIDE_BRIDGE_TIME_SLICES.length - 1
    : Math.min(attempt, TABLEBASE_WIDE_BRIDGE_TIME_SLICES.length - 1);
  const timeSlice = TABLEBASE_WIDE_BRIDGE_TIME_SLICES[sliceIndex];
  const nodeSlice = TABLEBASE_WIDE_BRIDGE_NODE_SLICES[sliceIndex];
  const stateSlice = Math.max(900, Math.round(TABLEBASE_WIDE_BRIDGE_MAX_STATES * Math.min(1, timeSlice / TABLEBASE_WIDE_BRIDGE_TIME_MS)));
  const preferredMoves = bridgePreferredMovesFromResult(baseline || current.lastResult, attempt < 2 ? 12 : 20);

  const run = async () => {
    let published = false;
    try {
      const warmSummary = await tablebase.warmExactBridgeTables(root, {
        maxPly: Math.max(6, TABLEBASE_FRONTIER_MAX_PLY),
        maxStates: stateSlice,
        maxBlocks: TABLEBASE_WIDE_BRIDGE_MAX_BLOCKS,
        priority: 0,
        seedSignatures: ['KPPvKP', 'KQPvKP']
      });
      if (!warmSummary?.warmed || !current || token !== activeToken) return;
      const proof = await tablebase.proveExactBridgeOutcome(root, {
        controller: root.turn,
        outcome: 'win',
        preferredMoves,
        maxPlies: TABLEBASE_BRIDGE_MAX_PLY,
        maxNodes: nodeSlice,
        timeMs: timeSlice,
        controllerMoveLimit: attempt < 2 ? 3 : 4,
        priority: 0
      });
      if (!proof || !current || token !== activeToken) return;
      if (!cacheVerifiedBridgeProof(current.cacheKey, root, proof)) return;
      const cachedProof = getVerifiedBridgeProof(current.cacheKey) || proof;
      const seed = current.lastResult || baseline || bridgeSeedBaseline(root, cachedProof);
      const candidate = stabilizePublishedMetrics(current, buildTablebaseBridgeResult(root, seed, cachedProof));
      const incumbent = current.lastResult;
      const chosen = compareAnalysisResults(incumbent, candidate, { preferNextOnTie: true });
      if (chosen !== candidate && !(candidate.tablebaseBridgeProof && Number(candidate.tablebaseBridgeDtm || Infinity) < Number(incumbent?.tablebaseBridgeDtm || Infinity))) return;
      current.tablebaseBridgeResult = candidate;
      current.lastResult = candidate;
      published = true;
      post('info', { token, result: candidate });
    } catch {
      // A wide bridge miss is never a negative conclusion. Ordinary completed
      // scores stay visible, and later completed iterations may start a larger
      // slice if the root still looks promising.
    } finally {
      if (current && token === activeToken) {
        current.tablebaseWideBridgeQueued = false;
        if (published) {
          current.tablebaseWideBridgeResolved = false; // allow later improvement.
        } else if (Number(current.tablebaseWideBridgeAttempts || 0) >= maxAttempts) {
          current.tablebaseWideBridgeResolved = !incumbentHasMate;
        } else if (!current.tablebaseBridgeResult) {
          const nextDelay = 420 + Math.min(600, 120 * Number(current.tablebaseWideBridgeAttempts || 0));
          setTimeout(() => {
            if (current && token === activeToken && !current.tablebaseBridgeResult) {
              queueWideTablebaseBridgeProof(token, revision + 50_000, current.lastResult || baseline, { delay: 0 });
            }
          }, nextDelay);
        }
      }
    }
  };
  if (delay > 0) setTimeout(() => { if (current && token === activeToken) void run(); }, delay);
  else void run();
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

function queueTablebaseBridgeProof(token, revision, baseline) {
  if (!current || token !== activeToken || !baseline?.lines?.length) return;
  const root = current.position.clone();
  if (Number(root.pieceCount || 0) !== 6) return;
  // A hydrated tail is a useful seed, but a completed root WDL sentinel is
  // already enough to attempt a proof.  This matters after a played move such
  // as Kb1 ...a2: Kxa2 immediately enters an exact five-piece tablebase and
  // must not spend a whole ordinary iteration displaying an unrelated score.
  const bridgeLine = findBridgeTailLine(baseline, { nonDrawOnly: false });
  if (!bridgeLine) return;
  const rootWdl = bridgeRootWdl(root, bridgeLine);
  if (!Number.isFinite(rootWdl)) return;
  if (rootWdl === 0 && !bridgeLine.tablebaseTail) return;
  const existing = current.tablebaseBridgeResult;
  const existingDtm = Number(existing?.lines?.[0]?.tablebaseBridgeDtm || 0);
  const nextLimit = existingDtm > 1 ? existingDtm - 1 : TABLEBASE_BRIDGE_MAX_PLY;
  const key = bridgeAttemptKey(bridgeLine.move, rootWdl, existingDtm);
  if (current.tablebaseBridgeQueuedKey === key) return;
  current.tablebaseBridgeQueued = true;
  current.tablebaseBridgeQueuedKey = key;
  const preferredMoves = Array.isArray(bridgeLine.pv) ? bridgeLine.pv.slice(0, 16) : [];
  void (async () => {
    let published = false;
    try {
      // Fetch/decompress candidate GTB families before proof construction.
      // The proof then uses resident exact blocks only, so its time budget is
      // deterministic and never waits on network I/O.
      const warmSummary = await tablebase.warmExactBridgeTables(root, {
        maxPly: TABLEBASE_FRONTIER_MAX_PLY,
        maxStates: TABLEBASE_FRONTIER_MAX_STATES,
        maxBlocks: 36,
        priority: 1,
        seedSignatures: [bridgeLine.tablebaseTail?.signature].filter(Boolean)
      });
      if (!warmSummary?.warmed) return;
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
      // A bridge result is publishable only after its *entire* AND/OR tree has
      // been replay-verified against the resident exact tablebases.  Keep the
      // tree in the worker cache before it reaches the display path so later
      // iterations and navigated child positions cannot lose a defender reply.
      if (!cacheVerifiedBridgeProof(current.cacheKey, root, proof)) return;
      const cachedProof = getVerifiedBridgeProof(current.cacheKey) || proof;
      const candidate = stabilizePublishedMetrics(current, buildTablebaseBridgeResult(root, baseline, cachedProof));
      const incumbent = current.lastResult;
      const chosen = compareAnalysisResults(incumbent, candidate, { preferNextOnTie: true });
      // Bridge bounds are root certificates, not speculative PV annotations.
      // A shorter newly found bound replaces an older one; ordinary analysis can
      // never overwrite it, while a stronger exact mate proof still can.
      if (chosen !== candidate && !(candidate.tablebaseBridgeProof && Number(candidate.tablebaseBridgeDtm || Infinity) < Number(incumbent?.tablebaseBridgeDtm || Infinity))) return;
      current.tablebaseBridgeResult = candidate;
      current.lastResult = candidate;
      published = true;
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
        // A failed/too-early GTB proof is not a negative conclusion. Retry a
        // small bounded number of times after the asynchronous tablebase warm
        // work has settled; the current WDL signal remains the only trigger.
        if (!published && !current.tablebaseBridgeResult
          && bridgeAttemptCount(current, key) < 2
          && findBridgeTailLine(current.lastResult, { nonDrawOnly: false })) {
          const attempts = bumpBridgeAttempt(current, key);
          setTimeout(() => {
            if (current && token === activeToken && !current.tablebaseBridgeResult) {
              queueTablebaseBridgeProof(token, revision + 70_000 + attempts, current.lastResult);
            }
          }, 120 + 100 * attempts);
        }
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
      current.lastResult = stabilizePublishedMetrics(current, hydrated);
      post('info', { token, result: current.lastResult });
      queueTablebaseBridgeProof(token, revision, current.lastResult);
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
    // Per-root published counters. These are monotone even when a deeper
    // iteration, a proof task, or an audit reports from a smaller local node
    // counter than the cumulative analysis stream.
    maxPublishedNodes: 0,
    maxPublishedElapsed: 0,
    maxPublishedNodeTarget: 0,
    tablebasePromotionPending: false,
    tablebasePromotionQueued: false,
    tablebasePromotionResolved: false,
    tablebasePromotionAttempts: 0,
    tablebaseFrontierQueued: false,
    tablebaseFrontierResolved: false,
    tablebaseFrontier: null,
    tablebaseImmediateEntryQueued: false,
    tablebaseWideBridgeQueued: false,
    tablebaseWideBridgeResolved: false,
    tablebaseWideBridgeAttempts: 0,
    analysisRevision: 0,
    tablebaseTailQueuedRevision: -1,
    tablebaseBridgeQueued: false,
    tablebaseBridgeQueuedKey: '',
    tablebaseBridgeResult: null,
    tablebaseBridgeImproveAttempts: 0,
    tablebaseBridgeAttemptsByKey: new Map()
  };
  nextDepth = 1;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = 0;
  totalElapsed = 0;

  // Reuse only a previously replay-verified full bridge tree.  Unlike an
  // ordinary PV cache, this record contains every defending reply and remains
  // valid for the identical root/history key. A mate upper bound is displayed
  // immediately but does not stop iterative search for a shorter/exact proof.
  const cachedBridgeProof = Number(position.pieceCount || 0) === 6
    ? getVerifiedBridgeProof(cacheKey)
    : null;
  if (cachedBridgeProof && tablebase.verifyExactBridgeProof(position.clone(), cachedBridgeProof, {
    controller: Number(cachedBridgeProof.controller || position.turn),
    outcome: cachedBridgeProof.tablebaseBridgeDraw ? 'draw' : 'win'
  })) {
    const cachedBaseline = resumeResult || bridgeSeedBaseline(position, cachedBridgeProof);
    const cachedResult = stabilizePublishedMetrics(
      current,
      buildTablebaseBridgeResult(position, cachedBaseline, cachedBridgeProof)
    );
    current.tablebaseBridgeResult = cachedResult;
    current.lastResult = cachedResult;
  }

  paused = Boolean(message.startPaused);
  const solvedResume = isSolvedResult(current.lastResult) && !isTablebaseBridgeUpperBound(current.lastResult);
  running = !paused && !solvedResume;

  if (current.lastResult?.lines?.length) {
    post('info', { token, result: { ...current.lastResult, cacheKey, cached: Boolean(cachedBridgeProof || resumeResult), searchDepth: 0, nextDepth: 0 } });
  }
  post('state', {
    token,
    state: paused ? 'paused' : solvedResume ? 'complete' : 'thinking',
    engine: current.lastResult?.engineLabel || ENGINE_VERSION,
    depth: Number(current.lastResult?.depth || 0),
    searchDepth: solvedResume ? 0 : nextDepth
  });
  if (!running) return;
  if (tablebaseEligible) {
    if (await probeTablebase(token, { announce: true })) return;
    if (token !== activeToken || !current) return;
    queueExactTablebasePromotion(token);
  } else {
    // Reuse a previously verified proof subtree when the user navigates into a
    // defended variation. This reconstructs the displayed PV from the cached
    // AND/OR tree; no defender reply is discarded.
    await tryCachedBridgeSubtree(token);
    if (token !== activeToken || !current) return;
    // A one-move exact entry is cheap to check and may immediately establish a
    // complete upper-bound certificate (for example Kb1 ...a2 Kxa2). Run this
    // before the first ordinary chunk so a stale opposite score is never shown
    // for a position already resolved by a direct tablebase transition.
    await tryImmediateExactEntryBridge(token);
    if (token !== activeToken || !current) return;
    // Preload six-to-five frontier blocks and start the broad bridge prover as
    // delayed side work.  The first ordinary chunks get the UI moving, while
    // Kb1-style roots no longer wait for a lucky PV-tablebase contact.
    queueTablebaseFrontierWarmup(token);
    queueWideTablebaseBridgeProof(token, 0, current.lastResult, { delay: 350 });
  }
  if (token === activeToken && running && !paused) schedule(token);
}


async function startMinifishPosition(message) {
  activeToken = Number(message.token || activeToken + 1);
  const token = activeToken;
  currentKernel = ENGINE_KERNELS.MINIFISH;
  const position = EnginePosition.fromFEN(message.fen);
  const cacheKey = String(message.cacheKey || position.key());
  effortMs = Math.max(120, Math.min(3000, Number(message.effortMs || effortMs)));
  multipv = Math.max(1, Math.min(5, Number(message.multipv || multipv)));
  current = {
    position,
    cacheKey,
    lastResult: null,
    bookMoves: Array.isArray(message.bookMoves) ? message.bookMoves : [],
    historyKeys: (Array.isArray(message.historyFens) ? message.historyFens : []).map(historyKeyFromFen),
    depthNodeCosts: new Map(),
    progressDepth: 0,
    progressTargetNodes: 0,
    maxPublishedNodes: 0,
    maxPublishedElapsed: 0,
    maxPublishedNodeTarget: 0,
    minifish: true
  };
  nextDepth = 1;
  currentBudgetMs = initialBudget(nextDepth);
  firstChunk = true;
  totalNodes = 0;
  totalElapsed = 0;
  paused = Boolean(message.startPaused);
  running = !paused;
  post('state', { token, state: paused ? 'paused' : 'thinking', engine: MINIFISH_VERSION, depth: 0, searchDepth: nextDepth });
  if (!running) return;

  if (Number(position.pieceCount || 0) <= 5) {
    try {
      const tb = await tablebase.analyze(position.clone(), { multipv });
      if (token !== activeToken || !current) return;
      if (tb) {
        const exact = stabilizePublishedMetrics(current, relabelTablebaseResult({ ...tb, cacheKey, rootTurn: position.turn }, MINIFISH_VERSION));
        current.lastResult = exact;
        running = false;
        post('info', { token, result: exact });
        post('state', { token, state: 'complete', engine: exact.engineLabel || MINIFISH_VERSION, depth: 0, searchDepth: 0, tablebase: true });
        return;
      }
    } catch {}
  }

  // A non-blocking preload lets Minifish cut 6→5 captures/promotions with the
  // exact database once resident, but the main brute-force search never waits
  // behind the old bridge-prover pipeline.
  if (Number(position.pieceCount || 0) <= 6) {
    void tablebase.warmExactWdlNeighborhood(position.clone(), { includeLegalChildren: true }).catch(() => false);
  }
  schedule(token, 0);
}

async function runMinifishChunk(token) {
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const raw = minifish.analyze(current.position.clone(), {
      timeMs: Math.max(45, currentBudgetMs),
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
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
      currentBudgetMs = Math.min(Math.max(effortMs, 900), Math.max(260, Math.round(currentBudgetMs * 1.28)));
      updateDepthNodeEstimate(current, totalNodes, requestedDepth);
    }

    const cumulative = withResultQuality(sortResultLinesForSide({
      ...raw,
      nodes: totalNodes,
      elapsed: totalElapsed,
      nps: Math.round(totalNodes * 1000 / Math.max(1, totalElapsed)),
      scoreDepth: Number(raw.scoreDepth || raw.depth || 0),
      searchDepth: nextDepth,
      searchBudget: currentBudgetMs,
      nodeTarget: updateDepthNodeEstimate(current, totalNodes, nextDepth),
      rootTurn: Number(raw.rootTurn || current.position.turn),
      cacheKey: current.cacheKey,
      cached: false,
      solved: isSolvedResult(raw),
      completed: raw.completed !== false,
      multiPvVerified: raw.multiPvVerified !== false,
      pvComplete: raw.pvComplete !== false,
      minifish: true
    }, current.position.turn, multipv));
    const visibleResult = stabilizePublishedMetrics(current, cumulative);
    let visible = null;
    if (isStableSearchResult(visibleResult)) {
      current.lastResult = visibleResult;
      visible = visibleResult;
    } else {
      visible = progressFromStable(current.lastResult, visibleResult, requestedDepth);
    }
    if (visible) post('info', { token, result: visible });

    const stable = current.lastResult;
    const mateFound = Boolean(stable?.lines?.[0]?.mateVerified && isSolvedResult(stable));
    if (stable?.terminal || mateFound || nextDepth > MAX_DEPTH) {
      running = false;
      post('state', { token, state: 'complete', engine: MINIFISH_VERSION, depth: Number(stable?.depth || raw.depth || 0), searchDepth: nextDepth });
      return;
    }
    post('state', { token, state: 'thinking', engine: MINIFISH_VERSION, depth: Number(stable?.depth || 0), searchDepth: nextDepth });
    schedule(token, 0);
  } catch (error) {
    running = false;
    post('error', { token, message: error?.stack || error?.message || String(error) });
  }
}

async function startPosition(message) {
  const kernel = selectedKernel(message.kernel);
  if (kernel === ENGINE_KERNELS.FAIRY) return startFairyPosition(message);
  if (kernel === ENGINE_KERNELS.MINIFISH) return startMinifishPosition(message);
  return startOrionPosition(message);
}

async function runChunk(token) {
  if (currentKernel === ENGINE_KERNELS.MINIFISH) return runMinifishChunk(token);
  if (!running || paused || token !== activeToken || !current) return;
  try {
    const requestedDepth = nextDepth;
    beginDepthNodeEstimate(current, requestedDepth);
    const stableMate = Boolean(current.lastResult?.lines?.[0]?.mateVerified || current.tablebaseBridgeResult);
    const pieceCount = Number(current.position?.pieceCount || 0);
    const compactEndgame = pieceCount >= 6 && pieceCount <= 8;
    const pawnOrRookEndgame = compactEndgame && current.position.board
      ? current.position.board.every(piece => {
        const type = Math.abs(Number(piece || 0));
        return type === 0 || type === 1 || type === 4 || type === 6;
      })
      : false;
    const mateBudget = stableMate
      ? Math.min(380, Math.max(90, Math.round(currentBudgetMs * 0.18)))
      : pawnOrRookEndgame
        ? Math.min(240, Math.max(135, Math.round(currentBudgetMs * 0.16)))
        : compactEndgame
          ? Math.min(210, Math.max(120, Math.round(currentBudgetMs * 0.14)))
          : Math.min(150, Math.max(50, Math.round(currentBudgetMs * 0.10)));
    const endgameProbeBudget = compactEndgame
      ? Math.min(130, Math.max(70, Math.round(currentBudgetMs * 0.07)))
      : 70;
    const fortressProbeBudget = compactEndgame
      ? Math.min(150, Math.max(100, Math.round(currentBudgetMs * 0.08)))
      : 150;
    // v20.5: ordinary alpha-beta keeps the majority of the analysis cadence.
    // Endgame proof work is still bounded side work, but 6-8 piece pawn/rook
    // endings receive enough coverage to catch short forced mates/draw holds.
    const mainBudget = Math.max(90, currentBudgetMs);
    const raw = searcher.analyze(current.position.clone(), {
      timeMs: mainBudget,
      maxDepth: requestedDepth,
      multipv,
      startDepth: requestedDepth,
      bookMoves: current.bookMoves,
      historyKeys: current.historyKeys,
      newPosition: firstChunk,
      endgameProbeMs: endgameProbeBudget,
      fortressProbeMs: fortressProbeBudget,
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
      const hasProof = Boolean(current.lastResult?.lines?.[0]?.mateVerified || current.tablebaseBridgeResult);
      const incompleteCap = hasProof ? Math.max(effortMs, 900) : Math.min(Math.max(effortMs, 900), 900);
      currentBudgetMs = Math.min(incompleteCap, Math.max(320, Math.round(currentBudgetMs * 1.35)));
      updateDepthNodeEstimate(current, totalNodes, requestedDepth);
    }

    const profile = resultPvProfile(raw);
    const cumulative = withResultQuality(sortResultLinesForSide({
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
    }, current.position.turn, multipv));

    // v20.5: freeze metrics only.  No capped WDL audit, historical fallback, or
    // sentinel replacement is allowed to modify the score/PV pair published by
    // this completed iteration.
    const boundedCumulative = stabilizePublishedMetrics(
      current,
      cumulative
    );

    if (findBridgeTailLine(boundedCumulative, { nonDrawOnly: false })) {
      // Even when an interior-WDL line is not publishable as a numeric score, it
      // is still a valid proof seed.  Start the exact bridge/tail pipeline from
      // the owning iteration without displaying a fallback score.
      queueTablebaseBridgeProof(token, current.analysisRevision + 1, boundedCumulative);
      queueTablebaseTailHydration(token, current.analysisRevision + 1, boundedCumulative);
    }

    // Atomic publication: never merge PVs, proof labels or scores from prior
    // and current iterations. A partial depth only refreshes stable metrics.
    let visible = null;
    if (isStableSearchResult(boundedCumulative)) {
      // A bridge certificate outranks ordinary alpha-beta output.  Keep the
      // engine searching for a shorter/exact mate, but never let a later raw
      // +220-style WDL sentinel overwrite the certified upper bound.
      const chosen = compareAnalysisResults(current.lastResult, boundedCumulative, { preferNextOnTie: true });
      if (chosen === boundedCumulative) {
        current.lastResult = boundedCumulative;
        current.analysisRevision += 1;
        visible = boundedCumulative;
        // Begin a true AND/OR proof as soon as a completed line has a WDL
        // signal.  Tail hydration remains a PV-quality enhancement, not the
        // gate that decides whether a forced result can be proven.
        queueTablebaseBridgeProof(token, current.analysisRevision, boundedCumulative);
        queueTablebaseTailHydration(token, current.analysisRevision, boundedCumulative);
        if (!boundedCumulative?.lines?.[0]?.mateVerified) {
          queueWideTablebaseBridgeProof(token, current.analysisRevision, boundedCumulative, { delay: 90 });
        } else if (Number(current.position?.pieceCount || 0) === 6) {
          queueWideTablebaseBridgeProof(token, current.analysisRevision, boundedCumulative, { delay: 180 });
        }
      } else {
        visible = progressFromStable(current.lastResult, boundedCumulative, requestedDepth);
      }
    } else {
      visible = progressFromStable(current.lastResult, boundedCumulative, requestedDepth);
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
    schedule(token, current?.tablebaseWideBridgeQueued ? 3_200 : 7);
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
    post('state', { token: activeToken, state: 'paused', engine: activeEngineLabel(), depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    return;
  }
  if (message.type === 'resume') {
    if (Number(message.token) !== activeToken || !current) return;
    if (isSolvedResult(current.lastResult) && !isTablebaseBridgeUpperBound(current.lastResult)) {
      paused = false;
      running = false;
      post('state', { token: activeToken, state: 'complete', engine: activeEngineLabel(), depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
      return;
    }
    paused = false;
    running = true;
    post('state', { token: activeToken, state: 'thinking', engine: activeEngineLabel(), depth: Number(current.lastResult?.depth || 0), searchDepth: nextDepth });
    schedule(activeToken);
    return;
  }
  if (message.type === 'stop') {
    activeToken = Number(message.token || activeToken + 1);
    running = false;
    paused = false;
    if (currentKernel === ENGINE_KERNELS.FAIRY) fairyProvider.stop();
    current = null;
    post('state', { token: activeToken, state: 'idle', engine: activeEngineLabel() });
    return;
  }
  if (message.type === 'clear') {
    // Clear search accelerators only on an explicit user reset. A normal root
    // transition retains TT/eval ordering but never retains analysis conclusions.
    searcher.clear();
    fairyProvider.stop();
    nextDepth = 1;
    currentBudgetMs = initialBudget(1);
    post('state', { token: activeToken, state: paused ? 'paused' : running ? 'thinking' : 'idle', engine: activeEngineLabel() });
  }
});

post('ready', { engine: ENGINE_VERSION });
