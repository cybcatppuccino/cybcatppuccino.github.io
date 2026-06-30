const BOARD_SIZE = 6;
const CELL_COUNT = BOARD_SIZE * BOARD_SIZE;
const PASS = -1;
const INF = 1e9;
const EPS = 0.01;
const DIRS = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0],           [1, 0],
  [-1, 1],  [0, 1],  [1, 1]
];
const CORNERS = new Set([0, 5, 30, 35]);
const EDGES = new Set(Array.from({ length: CELL_COUNT }, (_, i) => {
  const x = i % BOARD_SIZE;
  const y = Math.floor(i / BOARD_SIZE);
  return x === 0 || x === BOARD_SIZE - 1 || y === 0 || y === BOARD_SIZE - 1;
}).map((keep, i) => keep ? i : -1).filter((i) => i >= 0));
const CORNER_GROUPS = [
  { corner: 0, adjacent: [1, 6, 7], edgeA: [1, 2, 3, 4, 5], edgeB: [6, 12, 18, 24, 30] },
  { corner: 5, adjacent: [4, 10, 11], edgeA: [4, 3, 2, 1, 0], edgeB: [11, 17, 23, 29, 35] },
  { corner: 30, adjacent: [24, 25, 31], edgeA: [24, 18, 12, 6, 0], edgeB: [31, 32, 33, 34, 35] },
  { corner: 35, adjacent: [28, 29, 34], edgeA: [29, 23, 17, 11, 5], edgeB: [34, 33, 32, 31, 30] }
];
const CORNER_ADJACENT = new Set(CORNER_GROUPS.flatMap((g) => g.adjacent));
const PST = [
   120, -40,  16,  16, -40, 120,
   -40, -64, -10, -10, -64, -40,
    16, -10,   5,   5, -10,  16,
    16, -10,   5,   5, -10,  16,
   -40, -64, -10, -10, -64, -40,
   120, -40,  16,  16, -40, 120
];
const FILES = ['A', 'B', 'C', 'D', 'E', 'F'];
const ROWS = ['1', '2', '3', '4', '5', '6'];

let activeToken = 0;
const legalCache = new Map();
const flipCache = new Map();
const evalCache = new Map();
const featureCache = new Map();

self.onmessage = (event) => {
  const data = event.data || {};
  if (data.type === 'stop') {
    activeToken += 1;
    return;
  }
  if (data.type !== 'start') return;
  const token = data.token || `${Date.now()}`;
  activeToken += 1;
  const localToken = activeToken;
  const board = normalizeBoard(data.board || []);
  const side = data.side || 'black';
  const effort = data.effort || 'balanced';
  const key = data.key || '';
  const searcher = new FallbackSearcher({ board, side, effort, key, token, localToken });
  searcher.run().catch((error) => {
    self.postMessage({ type: 'error', key, token, message: error?.message || String(error) });
  });
};

function normalizeBoard(board) {
  return Array.from({ length: CELL_COUNT }, (_, i) => {
    const cell = board[i];
    if (cell === 'black' || cell === 1 || cell === 'b' || cell === 'B') return 'black';
    if (cell === 'white' || cell === -1 || cell === 'w' || cell === 'W') return 'white';
    return null;
  });
}

class FallbackSearcher {
  constructor({ board, side, effort, key, token, localToken }) {
    this.rootBoard = board;
    this.rootSide = side;
    this.effort = effort;
    this.key = key;
    this.token = token;
    this.localToken = localToken;
    this.tt = new Map();
    this.nodes = 0;
    this.qnodes = 0;
    this.cutoffs = 0;
    this.researches = 0;
    this.startedAt = Date.now();
    this.lastPostAt = 0;
    this.bestResult = null;
    this.previousOrder = [];
    this.previousScore = sideSign(side) * heuristicEvalWhite(board);
    this.killers = Array.from({ length: 96 }, () => []);
    this.history = new Map();
    this.counterMoves = new Map();
    this.fullWidthSearch = false;
    const empty = countsOf(board).empty;
    // 6x6 is small enough that analysis mode may continue to exact endgame.
    // Depth counts placed discs; forced passes do not consume depth.
    this.hardMaxDepth = Math.max(1, empty);
  }

  async run() {
    const legal = legalMovesFor(this.rootBoard, this.rootSide);
    if (isGameOver(this.rootBoard)) {
      const c = countsOf(this.rootBoard);
      this.bestResult = this.decorate({
        side: this.rootSide,
        lines: [],
        evalWhite: c.whiteDiff,
        bestMove: null,
        source: 'terminal',
        cache: 'Fallback',
        terminal: true,
        depth: 0,
        nodes: 0,
        nps: 0,
        searchingDepth: 0,
        exact: true
      });
      this.post('done');
      return;
    }

    if (!legal.length) {
      for (let depth = 1; depth <= this.hardMaxDepth; depth += 1) {
        if (this.cancelled()) return;
        this.fullWidthSearch = depth >= this.hardMaxDepth;
        const childPv = [];
        const scorePerspective = -this.negamax(this.rootBoard, opponent(this.rootSide), depth, -INF, INF, 1, 1, childPv, true);
        const evalWhite = this.rootSide === 'white' ? scorePerspective : -scorePerspective;
        this.bestResult = this.decorate({
          side: this.rootSide,
          lines: [{ move: PASS, evalWhite, source: 'forced pass', pv: [PASS, ...childPv] }],
          evalWhite,
          bestMove: PASS,
          source: depth >= this.hardMaxDepth ? 'JS fallback solved' : 'JS fallback iterative',
          cache: 'Fallback',
          depth,
          searchingDepth: depth + 1,
          exact: depth >= this.hardMaxDepth
        });
        this.post(depth >= this.hardMaxDepth ? 'done' : 'update');
        await yieldToLoop();
      }
      return;
    }

    for (let depth = 1; depth <= this.hardMaxDepth; depth += 1) {
      if (this.cancelled()) return;
      this.fullWidthSearch = depth >= this.hardMaxDepth;
      let result;
      const window = aspirationWindow(depth, this.previousScore, countsOf(this.rootBoard).empty);
      try {
        result = this.rootSearch(depth, this.previousScore - window, this.previousScore + window);
        const perspective = this.rootSide === 'white' ? result.evalWhite : -result.evalWhite;
        if (perspective <= this.previousScore - window + EPS || perspective >= this.previousScore + window - EPS) {
          this.researches += 1;
          result = this.rootSearch(depth, -INF, INF);
        }
      } catch (error) {
        result = this.rootSearch(depth, -INF, INF);
      }
      this.bestResult = this.decorate({
        ...result,
        source: depth >= this.hardMaxDepth ? 'JS fallback solved' : 'JS fallback iterative',
        cache: 'Fallback',
        depth,
        searchingDepth: depth + 1,
        exact: depth >= this.hardMaxDepth
      });
      this.previousOrder = this.bestResult.lines.map((line) => line.move);
      this.previousScore = this.rootSide === 'white' ? this.bestResult.evalWhite : -this.bestResult.evalWhite;
      this.post(depth >= this.hardMaxDepth ? 'done' : 'update');
      await yieldToLoop();
    }
  }

  rootSearch(depth, alphaStart = -INF, betaStart = INF) {
    const legal = legalMovesFor(this.rootBoard, this.rootSide);
    const moves = orderMoves(this.rootBoard, this.rootSide, legal, {
      preferred: this.previousOrder,
      killers: this.killers[0],
      history: this.history,
      counterMoves: this.counterMoves
    });
    const lines = [];
    let bestMove = null;
    let bestScore = -INF;
    let alpha = alphaStart;
    const beta = betaStart;
    let first = true;

    for (const move of moves) {
      if (this.cancelled()) break;
      const nb = applyMoveToBoard(this.rootBoard, this.rootSide, move);
      const childPv = [];
      let score;
      if (first || beta - alpha > 1e8) {
        score = -this.negamax(nb, opponent(this.rootSide), depth - 1, -beta, -alpha, 0, 1, childPv, true, move);
        first = false;
      } else {
        score = -this.negamax(nb, opponent(this.rootSide), depth - 1, -alpha - EPS, -alpha, 0, 1, childPv, false, move);
        if (score > alpha + EPS && score < beta - EPS) {
          this.researches += 1;
          childPv.length = 0;
          score = -this.negamax(nb, opponent(this.rootSide), depth - 1, -beta, -alpha, 0, 1, childPv, true, move);
        }
      }
      const evalWhite = this.rootSide === 'white' ? score : -score;
      lines.push({ move, evalWhite, source: 'JS fallback iterative', pv: [move, ...childPv] });
      if (score > bestScore) {
        bestScore = score;
        bestMove = move;
      }
      alpha = Math.max(alpha, score);
      lines.sort((a, z) => compareLines(a, z, this.rootSide));
      this.maybePost(depth);
    }

    lines.sort((a, z) => compareLines(a, z, this.rootSide));
    const bestLine = lines.find((line) => line.move === bestMove) || lines[0];
    return {
      side: this.rootSide,
      lines,
      evalWhite: bestLine?.evalWhite ?? heuristicEvalWhite(this.rootBoard),
      bestMove: bestMove ?? bestLine?.move ?? null,
      depth
    };
  }

  negamax(b, side, depth, alpha, beta, passCount, ply, outPv, pvNode, previousMove = null) {
    this.nodes += 1;
    if ((this.nodes & 8191) === 0) this.maybePost(depth);

    const c = countsOf(b);
    if (c.empty === 0 || passCount >= 2) return sideSign(side) * c.whiteDiff;
    const legalNow = legalMovesFor(b, side);
    const oppLegal = legalNow.length ? null : legalMovesFor(b, opponent(side));
    if (!legalNow.length && !oppLegal.length) return sideSign(side) * c.whiteDiff;
    if (depth <= 0) {
      this.qnodes += 1;
      return sideSign(side) * heuristicEvalWhite(b);
    }

    const alphaOrig = alpha;
    const betaOrig = beta;
    const ttKey = `${boardKey(b)}|${side}|${passCount}`;
    const hit = this.tt.get(ttKey);
    if (hit && hit.depth >= depth && (!this.fullWidthSearch || hit.fullWidth)) {
      if (hit.flag === 'exact') {
        if (outPv && hit.bestMove !== null && hit.bestMove !== undefined) outPv.push(hit.bestMove);
        return hit.value;
      }
      if (hit.flag === 'lower') alpha = Math.max(alpha, hit.value);
      else if (hit.flag === 'upper') beta = Math.min(beta, hit.value);
      if (alpha >= beta) return hit.value;
    }

    if (!legalNow.length) {
      // A forced pass is a normal move choice, but it should not consume a placed-disc depth.
      const childPv = [];
      const value = -this.negamax(b, opponent(side), depth, -beta, -alpha, passCount + 1, ply + 1, childPv, pvNode, PASS);
      this.storeTT(ttKey, depth, value, alphaOrig, betaOrig, PASS);
      if (outPv) outPv.push(PASS, ...childPv.slice(0, 12));
      return value;
    }

    const staticValue = sideSign(side) * heuristicEvalWhite(b);
    const preferred = [];
    if (hit?.bestMove !== undefined) preferred.push(hit.bestMove);
    const cm = previousMove !== null && previousMove !== undefined ? this.counterMoves.get(`${opponent(side)}:${previousMove}`) : null;
    if (cm !== null && cm !== undefined) preferred.push(cm);
    const moves = orderMoves(b, side, legalNow, {
      preferred,
      killers: this.killers[ply] || [],
      history: this.history,
      counterMoves: this.counterMoves
    });

    let bestValue = -INF;
    let bestMove = null;
    let bestChildPv = [];
    let searched = 0;

    for (const move of moves) {
      if (this.cancelled()) break;
      const tactical = isTacticalMove(b, side, move);
      const empty = c.empty;

      // Conservative frontier-style futility: never prune corners, forced pass moves, or endgames.
      if (!this.fullWidthSearch && !pvNode && depth <= 2 && searched >= 2 && !tactical && empty > 9) {
        const margin = depth === 1 ? 1.2 : 2.2;
        if (staticValue + margin <= alpha) continue;
      }

      const nb = applyMoveToBoard(b, side, move);
      let childDepth = depth - 1;
      let reduced = false;
      const oppMobility = legalMovesFor(nb, opponent(side)).length;
      if (!this.fullWidthSearch && !pvNode && depth >= 4 && searched >= 3 && !tactical && empty > 10 && oppMobility > 1) {
        const reduction = (searched >= 6 && depth >= 6 && oppMobility > 3) ? 2 : 1;
        childDepth = Math.max(0, childDepth - reduction);
        reduced = true;
      }

      const childPv = [];
      let score;
      if (searched === 0) {
        score = -this.negamax(nb, opponent(side), childDepth, -beta, -alpha, 0, ply + 1, childPv, pvNode, move);
      } else {
        score = -this.negamax(nb, opponent(side), childDepth, -alpha - EPS, -alpha, 0, ply + 1, childPv, false, move);
        if (score > alpha + EPS && score < beta - EPS) {
          this.researches += 1;
          childPv.length = 0;
          score = -this.negamax(nb, opponent(side), childDepth, -beta, -alpha, 0, ply + 1, childPv, pvNode, move);
        }
      }
      if (reduced && score > alpha + EPS) {
        this.researches += 1;
        childPv.length = 0;
        score = -this.negamax(nb, opponent(side), depth - 1, -beta, -alpha, 0, ply + 1, childPv, pvNode, move);
      }

      searched += 1;
      if (score > bestValue) {
        bestValue = score;
        bestMove = move;
        bestChildPv = childPv.slice(0, 12);
      }
      alpha = Math.max(alpha, score);
      if (alpha >= beta) {
        this.cutoffs += 1;
        this.recordKiller(ply, move);
        this.bumpHistory(side, move, depth * depth);
        if (previousMove !== null && previousMove !== undefined && move !== PASS) {
          this.counterMoves.set(`${opponent(side)}:${previousMove}`, move);
        }
        break;
      }
    }

    if (bestMove === null) bestValue = staticValue;
    this.storeTT(ttKey, depth, bestValue, alphaOrig, betaOrig, bestMove);
    if (outPv && bestMove !== null && bestMove !== undefined) outPv.push(bestMove, ...bestChildPv);
    return bestValue;
  }

  storeTT(key, depth, value, alphaOrig, betaOrig, bestMove) {
    let flag = 'exact';
    if (value <= alphaOrig) flag = 'upper';
    else if (value >= betaOrig) flag = 'lower';
    this.tt.set(key, { depth, value, flag, bestMove, fullWidth: this.fullWidthSearch });
    if (this.tt.size > 300000) this.tt.clear();
  }

  recordKiller(ply, move) {
    if (move === PASS || CORNERS.has(move)) return;
    const arr = this.killers[ply] || (this.killers[ply] = []);
    if (arr[0] === move) return;
    if (arr[1] === move) arr.splice(1, 1);
    arr.unshift(move);
    arr.length = Math.min(arr.length, 2);
  }

  bumpHistory(side, move, amount) {
    if (move === PASS) return;
    const key = `${side}:${move}`;
    this.history.set(key, (this.history.get(key) || 0) + amount);
  }

  decorate(result) {
    const elapsed = Math.max(1, Date.now() - this.startedAt);
    const exact = Boolean(result.exact || result.terminal);
    const normalizedLines = (result.lines || []).map((line) => ({
      ...line,
      exact: Boolean(line.exact || exact),
      evalWhite: normalizeReportedEval(line.evalWhite, line.exact || exact)
    }));
    return {
      ...result,
      lines: normalizedLines,
      evalWhite: normalizeReportedEval(result.evalWhite, exact),
      nodes: this.nodes,
      qnodes: this.qnodes,
      cutoffs: this.cutoffs,
      researches: this.researches,
      nps: Math.round(this.nodes * 1000 / elapsed),
      elapsedMs: elapsed,
      source: result.source || 'JS fallback iterative',
      cache: 'Fallback'
    };
  }

  maybePost(searchingDepth) {
    const now = Date.now();
    if (!this.bestResult || now - this.lastPostAt < 500) return;
    this.bestResult = this.decorate({ ...this.bestResult, searchingDepth });
    this.post('update');
  }

  post(kind) {
    if (this.cancelled() || !this.bestResult) return;
    this.lastPostAt = Date.now();
    self.postMessage({ type: kind, key: this.key, token: this.token, result: this.bestResult });
  }

  cancelled() {
    return this.localToken !== activeToken;
  }
}

function yieldToLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

function setCell(b, x, y, value) { b[y * BOARD_SIZE + x] = value; }
function getCell(b, x, y) { return b[y * BOARD_SIZE + x]; }
function inBounds(x, y) { return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE; }
function opponent(side) { return side === 'black' ? 'white' : 'black'; }
function sideSign(side) { return side === 'white' ? 1 : -1; }
function countsOf(b) {
  let black = 0;
  let white = 0;
  for (const cell of b) {
    if (cell === 'black') black++;
    if (cell === 'white') white++;
  }
  return { black, white, empty: CELL_COUNT - black - white, whiteDiff: white - black, blackDiff: black - white };
}
function legalMovesFor(b, side) {
  const key = `${boardKey(b)}|${side}`;
  const cached = legalCache.get(key);
  if (cached) return cached.slice();
  const moves = [];
  for (let idx = 0; idx < CELL_COUNT; idx++) {
    if (b[idx] !== null) continue;
    if (flipsForMove(b, side, idx).length) moves.push(idx);
  }
  if (legalCache.size > 300000) legalCache.clear();
  legalCache.set(key, moves);
  return moves.slice();
}
function flipsForMove(b, side, idx) {
  if (idx === PASS || b[idx] !== null) return [];
  const key = `${boardKey(b)}|${side}|${idx}`;
  const cached = flipCache.get(key);
  if (cached) return cached.slice();
  const x0 = idx % BOARD_SIZE;
  const y0 = Math.floor(idx / BOARD_SIZE);
  const enemy = opponent(side);
  const flips = [];
  for (const [dx, dy] of DIRS) {
    let x = x0 + dx;
    let y = y0 + dy;
    const line = [];
    while (inBounds(x, y) && getCell(b, x, y) === enemy) {
      line.push(y * BOARD_SIZE + x);
      x += dx;
      y += dy;
    }
    if (line.length && inBounds(x, y) && getCell(b, x, y) === side) flips.push(...line);
  }
  if (flipCache.size > 300000) flipCache.clear();
  flipCache.set(key, flips);
  return flips.slice();
}
function applyMoveToBoard(b, side, move) {
  const next = b.slice();
  if (move === PASS) return next;
  const flips = flipsForMove(b, side, move);
  next[move] = side;
  flips.forEach((idx) => { next[idx] = side; });
  return next;
}
function isGameOver(b) {
  if (countsOf(b).empty === 0) return true;
  return legalMovesFor(b, 'black').length === 0 && legalMovesFor(b, 'white').length === 0;
}
function boardKey(b) {
  return b.map((cell) => cell ? cell[0] : '.').join('');
}
function orderMoves(b, side, moves, context = {}) {
  const preferredMap = new Map((context.preferred || [])
    .filter((m) => m !== null && m !== undefined)
    .map((m, i) => [m, 1000000 - i * 1000]));
  const killerMap = new Map((context.killers || [])
    .filter((m) => m !== null && m !== undefined)
    .map((m, i) => [m, 450000 - i * 50000]));
  return moves.slice().sort((a, z) => {
    const pa = preferredMap.get(a) || 0;
    const pz = preferredMap.get(z) || 0;
    if (pa !== pz) return pz - pa;
    const ka = killerMap.get(a) || 0;
    const kz = killerMap.get(z) || 0;
    if (ka !== kz) return kz - ka;
    const ha = context.history?.get(`${side}:${a}`) || 0;
    const hz = context.history?.get(`${side}:${z}`) || 0;
    if (ha !== hz) return hz - ha;
    return moveOrderingScore(b, side, z) - moveOrderingScore(b, side, a);
  });
}
function moveOrderingScore(b, side, move) {
  const flips = flipsForMove(b, side, move).length;
  const c = countsOf(b);
  let score = 0;
  if (CORNERS.has(move)) score += 180000;
  if (EDGES.has(move)) score += 2500;
  if (isBadCornerAdjacent(b, move)) score -= 38000;

  const nb = applyMoveToBoard(b, side, move);
  const enemy = opponent(side);
  const enemyMob = legalMovesFor(nb, enemy).length;
  const selfNextMob = legalMovesFor(nb, side).length;
  const enemyCornerMoves = legalMovesFor(nb, enemy).filter((m) => CORNERS.has(m)).length;
  const stableGain = approximateStabilityDiff(nb) - approximateStabilityDiff(b);
  const frontier = frontierDiffAfterMove(b, side, move);
  const parity = parityOrderingBonus(nb, side);

  // Search first the moves that leave the opponent fewer replies; this usually improves alpha-beta cutoffs.
  score += (12 - enemyMob) * 780;
  score += selfNextMob * 70;
  score -= enemyCornerMoves * 45000;
  score += stableGain * sideSign(side) * 950;
  score -= frontier * sideSign(side) * 65;
  score += parity;
  score += flips * (c.empty <= 10 ? 130 : -26);
  score += PST[move] * 10;
  if (enemyMob === 0) score += 6000;
  return score;
}
function isBadCornerAdjacent(b, move) {
  if (!CORNER_ADJACENT.has(move)) return false;
  return CORNER_GROUPS.some((group) => group.adjacent.includes(move) && !b[group.corner]);
}
function isTacticalMove(b, side, move) {
  if (move === PASS || CORNERS.has(move)) return true;
  if (EDGES.has(move) && !isBadCornerAdjacent(b, move)) return true;
  const nb = applyMoveToBoard(b, side, move);
  return legalMovesFor(nb, opponent(side)).length === 0 || legalMovesFor(nb, opponent(side)).some((m) => CORNERS.has(m));
}
function compareLines(a, z, side) {
  if (a.move === PASS) return -1;
  if (z.move === PASS) return 1;
  return side === 'white' ? z.evalWhite - a.evalWhite : a.evalWhite - z.evalWhite;
}
function heuristicEvalWhite(b) {
  const key = boardKey(b);
  const hit = evalCache.get(key);
  if (hit !== undefined) return hit;
  const c = countsOf(b);
  if (c.empty === 0 || (legalMovesFor(b, 'black').length === 0 && legalMovesFor(b, 'white').length === 0)) return c.whiteDiff;

  const features = featuresFor(b);
  const empty = c.empty;
  const phase = (CELL_COUNT - empty) / CELL_COUNT;
  const materialWeight = empty <= 6 ? 1.75 : empty <= 12 ? 0.95 : empty <= 20 ? 0.20 : 0.025;
  const mobilityWeight = empty <= 6 ? 0.22 : empty <= 14 ? 0.95 : 1.75;
  const potentialWeight = empty <= 10 ? 0.06 : 0.54;
  const stabilityWeight = empty <= 8 ? 3.3 : empty <= 16 ? 2.55 : 2.05;
  const frontierWeight = empty <= 8 ? 0.14 : 0.72;
  const dangerWeight = empty <= 10 ? 1.2 : 4.3;
  const edgeWeight = empty <= 10 ? 0.55 : 0.24;
  const pstWeight = empty <= 10 ? 0.010 : 0.050;
  const parityWeight = empty <= 12 ? 0.42 : 0.05;
  const cornerAccessWeight = empty <= 14 ? 3.6 : 5.5;

  const raw =
    c.whiteDiff * materialWeight +
    features.mobility * mobilityWeight +
    features.potential * potentialWeight +
    features.cornerDiff * 10.5 +
    features.cornerAccess * cornerAccessWeight +
    features.stability * stabilityWeight +
    features.edgeDiff * edgeWeight -
    features.danger * dangerWeight -
    features.frontier * frontierWeight +
    features.parity * parityWeight +
    features.pst * pstWeight +
    features.safeEdge * (0.28 + phase * 0.18);
  if (evalCache.size > 300000) evalCache.clear();
  evalCache.set(key, raw);
  return raw;
}
function featuresFor(b) {
  const key = boardKey(b);
  const hit = featureCache.get(key);
  if (hit) return hit;
  const whiteMob = legalMovesFor(b, 'white').length;
  const blackMob = legalMovesFor(b, 'black').length;
  const mobility = scaledDiff(whiteMob, blackMob, 10);
  const potential = potentialMobilityDiff(b);
  const stability = approximateStabilityDiff(b);
  const parity = parityDiff(b);
  let cornerDiff = 0;
  let edgeDiff = 0;
  let danger = 0;
  let frontier = 0;
  let pst = 0;
  let safeEdge = 0;
  for (let i = 0; i < CELL_COUNT; i++) {
    const cell = b[i];
    if (!cell) continue;
    const sign = cell === 'white' ? 1 : -1;
    if (CORNERS.has(i)) cornerDiff += sign;
    if (EDGES.has(i)) edgeDiff += sign;
    if (isDangerDiscNearOpenCorner(b, i)) danger += sign;
    if (isFrontierDisc(b, i)) frontier += sign;
    if (EDGES.has(i) && !isVolatileEdgeDisc(b, i)) safeEdge += sign;
    pst += sign * PST[i];
  }
  const whiteCornerAccess = legalMovesFor(b, 'white').filter((m) => CORNERS.has(m)).length;
  const blackCornerAccess = legalMovesFor(b, 'black').filter((m) => CORNERS.has(m)).length;
  const cornerAccess = whiteCornerAccess - blackCornerAccess;
  const value = { mobility, potential, stability, parity, cornerDiff, edgeDiff, danger, frontier, pst, safeEdge, cornerAccess };
  if (featureCache.size > 300000) featureCache.clear();
  featureCache.set(key, value);
  return value;
}
function scaledDiff(a, b, scale) {
  return ((a - b) / Math.max(1, a + b)) * scale;
}
function potentialMobilityDiff(b) {
  let whitePotential = 0;
  let blackPotential = 0;
  for (let idx = 0; idx < CELL_COUNT; idx++) {
    if (b[idx]) continue;
    const x0 = idx % BOARD_SIZE;
    const y0 = Math.floor(idx / BOARD_SIZE);
    let touchesBlack = false;
    let touchesWhite = false;
    for (const [dx, dy] of DIRS) {
      const x = x0 + dx;
      const y = y0 + dy;
      if (!inBounds(x, y)) continue;
      const cell = getCell(b, x, y);
      if (cell === 'black') touchesBlack = true;
      if (cell === 'white') touchesWhite = true;
    }
    if (touchesBlack) whitePotential += 1;
    if (touchesWhite) blackPotential += 1;
  }
  return scaledDiff(whitePotential, blackPotential, 8);
}
function approximateStabilityDiff(b) {
  const stable = new Set();
  for (const group of CORNER_GROUPS) {
    const owner = b[group.corner];
    if (!owner) continue;
    stable.add(group.corner);
    for (const ray of [group.edgeA, group.edgeB]) {
      for (const idx of ray) {
        if (b[idx] !== owner) break;
        stable.add(idx);
      }
    }
  }
  let diff = 0;
  for (const idx of stable) diff += b[idx] === 'white' ? 1 : -1;
  return diff;
}
function parityDiff(b) {
  const empties = [];
  for (let i = 0; i < CELL_COUNT; i++) if (!b[i]) empties.push(i);
  const seen = new Set();
  let odd = 0;
  let even = 0;
  for (const start of empties) {
    if (seen.has(start)) continue;
    let size = 0;
    const stack = [start];
    seen.add(start);
    while (stack.length) {
      const idx = stack.pop();
      size += 1;
      const x0 = idx % BOARD_SIZE;
      const y0 = Math.floor(idx / BOARD_SIZE);
      for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
        const x = x0 + dx;
        const y = y0 + dy;
        const n = y * BOARD_SIZE + x;
        if (inBounds(x, y) && !b[n] && !seen.has(n)) {
          seen.add(n);
          stack.push(n);
        }
      }
    }
    if (size % 2) odd += 1;
    else even += 1;
  }
  // In Othello endgames, access to odd empty regions is often valuable for the player to move last.
  return odd - even;
}
function parityOrderingBonus(nb, side) {
  const p = parityDiff(nb);
  return side === 'white' ? p * 55 : -p * 55;
}
function frontierDiffAfterMove(b, side, move) {
  const before = frontierSignedDiff(b);
  const after = frontierSignedDiff(applyMoveToBoard(b, side, move));
  return after - before;
}
function frontierSignedDiff(b) {
  let diff = 0;
  for (let i = 0; i < CELL_COUNT; i++) {
    if (!b[i]) continue;
    if (isFrontierDisc(b, i)) diff += b[i] === 'white' ? 1 : -1;
  }
  return diff;
}
function isDangerDiscNearOpenCorner(b, idx) {
  return CORNER_GROUPS.some((group) => !b[group.corner] && group.adjacent.includes(idx));
}
function isFrontierDisc(b, idx) {
  const x0 = idx % BOARD_SIZE;
  const y0 = Math.floor(idx / BOARD_SIZE);
  for (const [dx, dy] of DIRS) {
    const x = x0 + dx;
    const y = y0 + dy;
    if (inBounds(x, y) && getCell(b, x, y) === null) return true;
  }
  return false;
}
function isVolatileEdgeDisc(b, idx) {
  if (!EDGES.has(idx)) return false;
  return CORNER_GROUPS.some((group) => group.edgeA.includes(idx) || group.edgeB.includes(idx)) &&
    CORNER_GROUPS.some((group) => !b[group.corner] && (group.edgeA.includes(idx) || group.edgeB.includes(idx)));
}
function aspirationWindow(depth, previousScore, empty) {
  if (depth <= 2) return INF;
  if (empty <= 10) return 5.0;
  if (empty <= 18) return 4.0;
  return 3.0 + Math.min(4, depth * 0.15 + Math.abs(previousScore) * 0.04);
}
function normalizeReportedEval(value, exact = false) {
  const n = Number(value);
  if (!Number.isFinite(n)) return 0;
  return exact ? clamp(n, -36, 36) : n;
}
function clamp(value, min, max) { return Math.max(min, Math.min(max, value)); }
function coordOf(idx) {
  if (idx === PASS) return 'pass';
  const x = idx % BOARD_SIZE;
  const y = Math.floor(idx / BOARD_SIZE);
  return `${FILES[x]}${ROWS[y]}`;
}
