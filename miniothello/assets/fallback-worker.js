const BOARD_SIZE = 6;
const CELL_COUNT = BOARD_SIZE * BOARD_SIZE;
const PASS = -1;
const INF = 1e9;
const EPS = 0.01;
const UI_SEARCH_WINDOW_MS = 500;
const UI_CALIBRATION_WINDOW_MS = 80;
const UI_PUBLISH_INTERVAL_MS = UI_SEARCH_WINDOW_MS + UI_CALIBRATION_WINDOW_MS;
const UI_CALIBRATION_EXACT_ENDGAME_EMPTY = 5;
const DIRS = [
  [-1, -1], [0, -1], [1, -1],
  [-1, 0],           [1, 0],
  [-1, 1],  [0, 1],  [1, 1]
];
const BB_BITS = Array.from({ length: CELL_COUNT }, (_, i) => 1n << BigInt(i));
const BB_RAYS = Array.from({ length: CELL_COUNT }, (_, idx) => {
  const x0 = idx % BOARD_SIZE;
  const y0 = Math.floor(idx / BOARD_SIZE);
  return DIRS.map(([dx, dy]) => {
    const ray = [];
    let x = x0 + dx;
    let y = y0 + dy;
    while (x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE) {
      ray.push(y * BOARD_SIZE + x);
      x += dx;
      y += dy;
    }
    return ray;
  });
});

function bitCount(mask) {
  let n = mask;
  let count = 0;
  while (n) {
    n &= n - 1n;
    count += 1;
  }
  return count;
}

function attachBitboard(b, blackMask = null, whiteMask = null) {
  let black = blackMask;
  let white = whiteMask;
  if (black === null || white === null) {
    black = 0n;
    white = 0n;
    for (let i = 0; i < CELL_COUNT; i += 1) {
      if (b[i] === 'black') black |= BB_BITS[i];
      else if (b[i] === 'white') white |= BB_BITS[i];
    }
  }
  b.__black = black;
  b.__white = white;
  b.__blackCount = bitCount(black);
  b.__whiteCount = bitCount(white);
  b.__key = `${black.toString(36)}:${white.toString(36)}`;
  return b;
}

function ensureBitboard(b) {
  if (!b || typeof b.__black !== 'bigint' || typeof b.__white !== 'bigint') return attachBitboard(b);
  return b;
}

function cloneBoard(b) {
  ensureBitboard(b);
  return attachBitboard(b.slice(0, CELL_COUNT), b.__black, b.__white);
}

function flipsMaskForMove(b, side, idx) {
  if (idx === PASS) return 0n;
  ensureBitboard(b);
  const bit = BB_BITS[idx];
  const occupied = b.__black | b.__white;
  if ((occupied & bit) !== 0n) return 0n;
  const own = side === 'black' ? b.__black : b.__white;
  const enemy = side === 'black' ? b.__white : b.__black;
  let flips = 0n;
  for (const ray of BB_RAYS[idx]) {
    let line = 0n;
    for (const pos of ray) {
      const p = BB_BITS[pos];
      if ((enemy & p) !== 0n) {
        line |= p;
        continue;
      }
      if ((own & p) !== 0n) {
        if (line !== 0n) flips |= line;
        break;
      }
      break;
    }
  }
  return flips;
}

function indicesFromMask(mask) {
  const out = [];
  for (let i = 0; i < CELL_COUNT; i += 1) {
    if ((mask & BB_BITS[i]) !== 0n) out.push(i);
  }
  return out;
}

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


// Stage-aware 6x6 pattern/n-tuple evaluation.  These tables are generated lazily
// from compact pattern codes, so the worker gets a Logistello-style representation
// without shipping a huge learned weight file.  The score function is intentionally
// conservative; the table layout can later be replaced by learned weights from
// self-play/exact-label data without touching the search code.
const PATTERN_PHASE_COUNT = 10;
const PATTERN_TABLE_LIMIT = 220000;
const patternEvalCache = new Map();
const patternTableCache = new Map();
const PATTERN_DEFS = buildPatternDefs();
const MPC_ENABLED = false; // enable only after stage-wise shallow/deep error calibration.

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
  const searcher = new FallbackSearcher({ board, side, effort, key, token, localToken, uiCalibration: Boolean(data.uiCalibration) });
  searcher.run().catch((error) => {
    self.postMessage({ type: 'error', key, token, message: error?.message || String(error) });
  });
};

function normalizeBoard(board) {
  const cells = Array.from({ length: CELL_COUNT }, (_, i) => {
    const cell = board[i];
    if (cell === 'black' || cell === 1 || cell === 'b' || cell === 'B') return 'black';
    if (cell === 'white' || cell === -1 || cell === 'w' || cell === 'W') return 'white';
    return null;
  });
  return attachBitboard(cells);
}

class FallbackSearcher {
  constructor({ board, side, effort, key, token, localToken, uiCalibration = false }) {
    this.rootBoard = board;
    this.rootSide = side;
    this.effort = effort;
    this.key = key;
    this.token = token;
    this.localToken = localToken;
    this.uiCalibration = Boolean(uiCalibration);
    this.tt = new Map();
    this.nodes = 0;
    this.qnodes = 0;
    this.cutoffs = 0;
    this.researches = 0;
    this.startedAt = Date.now();
    this.lastPostAt = Date.now();
    this.nextPostAt = this.lastPostAt + UI_SEARCH_WINDOW_MS;
    this.pendingPostKind = 'update';
    this.bestResult = null;
    this.previousOrder = [];
    this.previousScore = sideSign(side) * heuristicEvalWhite(board);
    this.uiPredictionStats = new Map();
    this.uiPredictionMemo = new Map();
    this.uiPredictionCursor = 0;
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
      this.bestResult = this.decorate({
        side: this.rootSide,
        lines: [],
        evalWhite: finalWhiteDiff(this.rootBoard),
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
      this.flushFinal();
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
        if (depth >= this.hardMaxDepth) this.flushFinal();
        else this.publishIfDue('update');
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
      if (depth >= this.hardMaxDepth) this.flushFinal();
      else this.publishIfDue('update');
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
    if (c.empty === 0 || passCount >= 2) return sideSign(side) * finalWhiteDiff(b);
    const legalNow = legalMovesFor(b, side);
    const oppLegal = legalNow.length ? null : legalMovesFor(b, opponent(side));
    if (!legalNow.length && !oppLegal.length) return sideSign(side) * finalWhiteDiff(b);
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
    if (!this.bestResult) return;
    this.bestResult = this.decorate({ ...this.bestResult, searchingDepth });
    this.publishIfDue('update');
  }

  publishIfDue(kind = 'update', force = false) {
    if (this.cancelled() || !this.bestResult) return false;
    const now = Date.now();
    if (!force && now < this.nextPostAt) {
      if (kind === 'done') this.pendingPostKind = 'done';
      return false;
    }
    this.post(kind === 'done' || this.pendingPostKind === 'done' ? 'done' : 'update');
    return true;
  }

  flushFinal() {
    if (this.cancelled() || !this.bestResult) return;
    const delay = Math.max(0, this.nextPostAt - Date.now());
    if (delay <= 0) {
      this.post('done');
      return;
    }
    setTimeout(() => {
      if (!this.cancelled() && this.bestResult) this.post('done');
    }, delay);
  }

  post(kind) {
    if (this.cancelled() || !this.bestResult) return;
    this.lastPostAt = Date.now();
    this.nextPostAt = this.lastPostAt + UI_PUBLISH_INTERVAL_MS;
    this.pendingPostKind = 'update';
    const uiResult = this.uiCalibration ? this.addUiTerminalPredictions(this.bestResult) : this.bestResult;
    self.postMessage({ type: kind, key: this.key, token: this.token, result: uiResult });
  }

  addUiTerminalPredictions(result) {
    const exact = Boolean(result.exact || result.terminal);
    const cumulativeSamplesBefore = this.totalUiPredictionSamples();
    if (!result?.lines?.length) {
      return {
        ...result,
        uiCalibrationSamples: cumulativeSamplesBefore,
        uiCalibrationTerminalSamples: cumulativeSamplesBefore,
        uiCalibrationBatchTerminalSamples: 0,
        uiCalibrationMs: 0
      };
    }

    if (exact) {
      return {
        ...result,
        lines: result.lines.map((line) => ({ ...line, exact: Boolean(line.exact || exact) })),
        uiCalibrationSamples: cumulativeSamplesBefore,
        uiCalibrationTerminalSamples: cumulativeSamplesBefore,
        uiCalibrationBatchTerminalSamples: 0,
        uiCalibrationMs: 0
      };
    }

    const targets = result.lines
      .map((line) => ({
        line,
        move: line.move,
        frontier: frontierAfterCandidateMove(this.rootBoard, this.rootSide, line.move)
      }))
      .filter(({ line, frontier }) => !Boolean(line.exact || exact) && frontier);

    if (!targets.length) {
      return {
        ...result,
        uiCalibrationSamples: cumulativeSamplesBefore,
        uiCalibrationTerminalSamples: cumulativeSamplesBefore,
        uiCalibrationBatchTerminalSamples: 0,
        uiCalibrationMs: 0
      };
    }

    for (const { move } of targets) {
      if (!this.uiPredictionStats.has(move)) this.uiPredictionStats.set(move, { total: 0, samples: 0 });
      if (!this.uiPredictionMemo.has(move)) this.uiPredictionMemo.set(move, new Map());
    }

    const started = Date.now();
    const deadline = started + UI_CALIBRATION_WINDOW_MS;
    let batchSamples = 0;

    // Display-only terminal prediction: spend the full 80 ms budget in round-robin order.
    // There is no total sample cap. Stats are cumulative for this worker/position, so every
    // publish adds newly completed finals to the running average shown in the UI.
    while (Date.now() < deadline && targets.length) {
      const target = targets[this.uiPredictionCursor % targets.length];
      const stat = this.uiPredictionStats.get(target.move);
      const memo = this.uiPredictionMemo.get(target.move) || new Map();
      const value = deterministicSinglePlayout(
        target.frontier.board,
        target.frontier.side,
        target.frontier.passCount,
        stat.samples,
        deadline,
        memo
      );
      if (!Number.isFinite(value)) break;
      stat.total += value;
      stat.samples += 1;
      batchSamples += 1;
      this.uiPredictionStats.set(target.move, stat);
      this.uiPredictionMemo.set(target.move, memo);
      this.uiPredictionCursor = (this.uiPredictionCursor + 1) % targets.length;
    }

    const adjustedLines = result.lines.map((line) => {
      const stat = this.uiPredictionStats.get(line.move);
      if (!stat || !stat.samples || Boolean(line.exact || exact)) return line;
      return {
        ...line,
        predictedFinalWhite: stat.total / stat.samples,
        predictedFinalSamples: stat.samples
      };
    });

    const cumulativeSamples = this.totalUiPredictionSamples();

    return {
      ...result,
      lines: adjustedLines,
      uiCalibrationSamples: cumulativeSamples,
      uiCalibrationTerminalSamples: cumulativeSamples,
      uiCalibrationBatchTerminalSamples: batchSamples,
      uiCalibrationMs: Date.now() - started
    };
  }

  totalUiPredictionSamples() {
    let total = 0;
    for (const stat of this.uiPredictionStats.values()) total += stat.samples || 0;
    return total;
  }

  cancelled() {
    return this.localToken !== activeToken;
  }
}

function yieldToLoop() {
  return new Promise((resolve) => setTimeout(resolve, 0));
}


function rawLegalMovesFor(b, side) {
  return legalMovesFor(b, side);
}

function rawFlipsForMove(b, side, idx) {
  return flipsForMove(b, side, idx);
}

function rawApplyMoveToBoard(b, side, move) {
  return applyMoveToBoard(b, side, move);
}

function frontierAfterCandidateMove(rootBoard, rootSide, move) {
  const legal = rawLegalMovesFor(rootBoard, rootSide);
  if (!legal.length) {
    if (move !== PASS) return null;
    return { board: cloneBoard(rootBoard), side: opponent(rootSide), passCount: 1 };
  }
  if (move === PASS || !legal.includes(move)) return null;
  return { board: rawApplyMoveToBoard(rootBoard, rootSide, move), side: opponent(rootSide), passCount: 0 };
}

function deterministicSinglePlayout(startBoard, startSide, startPassCount, seed, deadline, memo) {
  let b = cloneBoard(startBoard);
  let side = startSide;
  let passCount = startPassCount || 0;
  let ply = 0;
  while (countsOf(b).empty > UI_CALIBRATION_EXACT_ENDGAME_EMPTY && passCount < 2 && ply < 80) {
    if (Date.now() >= deadline) return NaN;
    const legal = rawLegalMovesFor(b, side);
    if (!legal.length) {
      side = opponent(side);
      passCount += 1;
      ply += 1;
      continue;
    }
    const ordered = calibrationOrderedMoves(b, side, legal);
    const width = calibrationBreadthWidth(ordered.length, ply, countsOf(b).empty);
    const move = ordered[Math.min(deterministicRank(seed, ply, width), ordered.length - 1)];
    b = rawApplyMoveToBoard(b, side, move);
    side = opponent(side);
    passCount = 0;
    ply += 1;
  }
  return solveLastPliesPerfect(b, side, passCount, deadline, memo);
}

function calibrationOrderedMoves(b, side, moves) {
  return moves
    .map((move) => ({ move, score: calibrationMoveScore(b, side, move) }))
    .sort((a, z) => z.score - a.score)
    .map((item) => item.move);
}

function calibrationMoveScore(b, side, move) {
  const flips = rawFlipsForMove(b, side, move).length;
  const empty = countsOf(b).empty;
  let score = 0;
  if (CORNERS.has(move)) score += 1200;
  else if (EDGES.has(move) && !isBadCornerAdjacent(b, move)) score += 90;
  if (isBadCornerAdjacent(b, move)) score -= 620;
  score += flips * (empty <= 10 ? 22 : empty <= 18 ? 3 : -2.5);
  score += PST[move] * 0.16;
  // Deterministic tie-breaker keeps broad sampling stable without pseudo-randomness.
  score += (35 - move) * 0.001;
  return score;
}

function frontierDiffAfterMoveRaw(before, after) {
  return frontierSignedDiff(after) - frontierSignedDiff(before);
}

function calibrationBreadthWidth(moveCount, ply, empty) {
  if (moveCount <= 1) return 1;
  if (empty <= 5) return moveCount;
  // Broad near the current frontier, then increasingly depth-oriented.
  if (ply <= 1) return Math.min(moveCount, 6);
  if (ply <= 4) return Math.min(moveCount, 5);
  if (ply <= 10) return Math.min(moveCount, 4);
  if (empty <= 12) return Math.min(moveCount, 3);
  return Math.min(moveCount, 2);
}

function deterministicRank(seed, ply, width) {
  if (width <= 1 || seed === 0) return 0;
  const h = hash32((seed + 1) * 1103515245 + (ply + 17) * 2654435761);
  if (ply <= 1) return h % width;
  const bucket = h % 100;
  if (bucket < 52) return 0;
  if (bucket < 77) return Math.min(1, width - 1);
  if (bucket < 91) return Math.min(2, width - 1);
  return h % width;
}

function solveLastPliesPerfect(board, side, passCount, deadline, memo = new Map()) {
  if (Date.now() >= deadline) return NaN;
  const c = countsOf(board);
  if (c.empty === 0 || passCount >= 2 || isGameOver(board)) return finalWhiteDiff(board);

  const key = `${boardKey(board)}|${side}|${passCount || 0}`;
  if (memo.has(key)) return memo.get(key);

  const legal = rawLegalMovesFor(board, side);
  let value;
  if (!legal.length) {
    value = solveLastPliesPerfect(board, opponent(side), (passCount || 0) + 1, deadline, memo);
  } else {
    const ordered = calibrationOrderedMoves(board, side, legal);
    value = side === 'white' ? -INF : INF;
    for (const move of ordered) {
      if (Date.now() >= deadline) return NaN;
      const child = rawApplyMoveToBoard(board, side, move);
      const childValue = solveLastPliesPerfect(child, opponent(side), 0, deadline, memo);
      if (!Number.isFinite(childValue)) return NaN;
      if (side === 'white') value = Math.max(value, childValue);
      else value = Math.min(value, childValue);
    }
  }

  memo.set(key, value);
  return value;
}

function hash32(x) {
  x |= 0;
  x ^= x >>> 16;
  x = Math.imul(x, 0x7feb352d);
  x ^= x >>> 15;
  x = Math.imul(x, 0x846ca68b);
  x ^= x >>> 16;
  return x >>> 0;
}


function idxOf(x, y) { return y * BOARD_SIZE + x; }

function buildPatternDefs() {
  const defs = [];
  const seen = new Set();
  const add = (id, type, cells, extra = {}) => {
    const clean = cells.filter((v, i, arr) => v >= 0 && v < CELL_COUNT && arr.indexOf(v) === i);
    if (clean.length < 2) return;
    const key = `${type}:${clean.join(',')}`;
    if (seen.has(key)) return;
    seen.add(key);
    defs.push({ id, type, cells: clean, ...extra });
  };

  // Four 3x3 corner regions: the highest-value patterns on a 6x6 board.
  const cornerSpecs = [
    { id: 'c3_tl', corner: idxOf(0,0), cells: [idxOf(0,0),idxOf(1,0),idxOf(2,0),idxOf(0,1),idxOf(1,1),idxOf(2,1),idxOf(0,2),idxOf(1,2),idxOf(2,2)], rays: [[idxOf(1,0),idxOf(2,0),idxOf(3,0),idxOf(4,0),idxOf(5,0)], [idxOf(0,1),idxOf(0,2),idxOf(0,3),idxOf(0,4),idxOf(0,5)]], adjacent: [idxOf(1,0),idxOf(0,1),idxOf(1,1)] },
    { id: 'c3_tr', corner: idxOf(5,0), cells: [idxOf(5,0),idxOf(4,0),idxOf(3,0),idxOf(5,1),idxOf(4,1),idxOf(3,1),idxOf(5,2),idxOf(4,2),idxOf(3,2)], rays: [[idxOf(4,0),idxOf(3,0),idxOf(2,0),idxOf(1,0),idxOf(0,0)], [idxOf(5,1),idxOf(5,2),idxOf(5,3),idxOf(5,4),idxOf(5,5)]], adjacent: [idxOf(4,0),idxOf(5,1),idxOf(4,1)] },
    { id: 'c3_bl', corner: idxOf(0,5), cells: [idxOf(0,5),idxOf(1,5),idxOf(2,5),idxOf(0,4),idxOf(1,4),idxOf(2,4),idxOf(0,3),idxOf(1,3),idxOf(2,3)], rays: [[idxOf(1,5),idxOf(2,5),idxOf(3,5),idxOf(4,5),idxOf(5,5)], [idxOf(0,4),idxOf(0,3),idxOf(0,2),idxOf(0,1),idxOf(0,0)]], adjacent: [idxOf(1,5),idxOf(0,4),idxOf(1,4)] },
    { id: 'c3_br', corner: idxOf(5,5), cells: [idxOf(5,5),idxOf(4,5),idxOf(3,5),idxOf(5,4),idxOf(4,4),idxOf(3,4),idxOf(5,3),idxOf(4,3),idxOf(3,3)], rays: [[idxOf(4,5),idxOf(3,5),idxOf(2,5),idxOf(1,5),idxOf(0,5)], [idxOf(5,4),idxOf(5,3),idxOf(5,2),idxOf(5,1),idxOf(5,0)]], adjacent: [idxOf(4,5),idxOf(5,4),idxOf(4,4)] }
  ];
  for (const spec of cornerSpecs) add(spec.id, 'corner3x3', spec.cells, spec);

  // Six rows and six columns.
  for (let y = 0; y < BOARD_SIZE; y += 1) add(`row${y}`, 'line6', Array.from({ length: BOARD_SIZE }, (_, x) => idxOf(x, y)));
  for (let x = 0; x < BOARD_SIZE; x += 1) add(`col${x}`, 'line6', Array.from({ length: BOARD_SIZE }, (_, y) => idxOf(x, y)));

  // All long diagonals of length 4+ in both directions.
  const collectDiag = (x, y, dx, dy) => {
    const cells = [];
    while (inBounds(x, y)) { cells.push(idxOf(x, y)); x += dx; y += dy; }
    return cells;
  };
  for (let x = 0; x < BOARD_SIZE; x += 1) {
    const d1 = collectDiag(x, 0, 1, 1); if (d1.length >= 4) add(`dse_t${x}`, `diag${d1.length}`, d1);
    const d2 = collectDiag(x, 0, -1, 1); if (d2.length >= 4) add(`dsw_t${x}`, `diag${d2.length}`, d2);
  }
  for (let y = 1; y < BOARD_SIZE; y += 1) {
    const d1 = collectDiag(0, y, 1, 1); if (d1.length >= 4) add(`dse_l${y}`, `diag${d1.length}`, d1);
    const d2 = collectDiag(BOARD_SIZE - 1, y, -1, 1); if (d2.length >= 4) add(`dsw_r${y}`, `diag${d2.length}`, d2);
  }

  // Edge + adjacent inner line, useful for edge traps and stable edge detection.
  add('edge_top_inner', 'edge2', [...Array.from({ length: BOARD_SIZE }, (_, x) => idxOf(x, 0)), ...Array.from({ length: BOARD_SIZE }, (_, x) => idxOf(x, 1))]);
  add('edge_bottom_inner', 'edge2', [...Array.from({ length: BOARD_SIZE }, (_, x) => idxOf(x, 5)), ...Array.from({ length: BOARD_SIZE }, (_, x) => idxOf(x, 4))]);
  add('edge_left_inner', 'edge2', [...Array.from({ length: BOARD_SIZE }, (_, y) => idxOf(0, y)), ...Array.from({ length: BOARD_SIZE }, (_, y) => idxOf(1, y))]);
  add('edge_right_inner', 'edge2', [...Array.from({ length: BOARD_SIZE }, (_, y) => idxOf(5, y)), ...Array.from({ length: BOARD_SIZE }, (_, y) => idxOf(4, y))]);

  // Systematic short straight n-tuples.  They are cheap and make the evaluator less
  // brittle than a pure hand-weighted feature sum.
  const dirs = [[1,0], [0,1], [1,1], [1,-1]];
  for (const len of [2, 3]) {
    for (let y = 0; y < BOARD_SIZE; y += 1) {
      for (let x = 0; x < BOARD_SIZE; x += 1) {
        for (const [dx, dy] of dirs) {
          const cells = [];
          for (let k = 0; k < len; k += 1) {
            const xx = x + dx * k;
            const yy = y + dy * k;
            if (!inBounds(xx, yy)) break;
            cells.push(idxOf(xx, yy));
          }
          if (cells.length === len) add(`nt${len}_${x}_${y}_${dx}_${dy}`, `ntuple${len}`, cells);
        }
      }
    }
  }
  return defs;
}

function patternPhaseIndex(empty) {
  const occupied = CELL_COUNT - empty;
  const span = Math.max(1, CELL_COUNT - 4);
  return clamp(Math.floor(((occupied - 4) / span) * PATTERN_PHASE_COUNT), 0, PATTERN_PHASE_COUNT - 1);
}

function encodePattern(b, cells) {
  let code = 0;
  let mul = 1;
  for (const idx of cells) {
    const cell = b[idx];
    const v = cell === 'white' ? 1 : cell === 'black' ? 2 : 0;
    code += v * mul;
    mul *= 3;
  }
  return code;
}

function stateAtCode(code, pos) {
  for (let i = 0; i < pos; i += 1) code = Math.floor(code / 3);
  return code % 3;
}

function patternEvalWhite(b) {
  const key = boardKey(b);
  const hit = patternEvalCache.get(key);
  if (hit !== undefined) return hit;
  const phase = patternPhaseIndex(countsOf(b).empty);
  let total = 0;
  for (const def of PATTERN_DEFS) {
    const code = encodePattern(b, def.cells);
    const tableKey = `${phase}|${def.id}|${code}`;
    let value = patternTableCache.get(tableKey);
    if (value === undefined) {
      value = scorePatternCode(def, code, phase);
      if (patternTableCache.size > PATTERN_TABLE_LIMIT) patternTableCache.clear();
      patternTableCache.set(tableKey, value);
    }
    total += value;
  }
  // Avoid double counting from many overlapping n-tuples while preserving stage signal.
  const normalized = total / 7.5;
  if (patternEvalCache.size > 300000) patternEvalCache.clear();
  patternEvalCache.set(key, normalized);
  return normalized;
}

function scorePatternCode(def, code, phase) {
  const maturity = phase / Math.max(1, PATTERN_PHASE_COUNT - 1);
  const states = def.cells.map((_, i) => stateAtCode(code, i));
  const signs = states.map((v) => v === 1 ? 1 : v === 2 ? -1 : 0);
  const diff = signs.reduce((a, z) => a + z, 0);
  let score = diff * (0.06 + maturity * 0.04);

  for (let i = 0; i < def.cells.length; i += 1) {
    const sign = signs[i];
    if (!sign) continue;
    const idx = def.cells[i];
    if (CORNERS.has(idx)) score += sign * (2.2 + 3.6 * maturity);
    else if (EDGES.has(idx)) score += sign * (0.32 + 0.62 * maturity);
    else score += sign * (0.03 + 0.04 * maturity);
  }

  if (def.type === 'corner3x3') {
    const cornerState = states[0];
    const cornerSign = cornerState === 1 ? 1 : cornerState === 2 ? -1 : 0;
    if (cornerSign) {
      score += cornerSign * (7.8 + 6.4 * maturity);
      for (const ray of def.rays || []) {
        let run = 0;
        for (const idx of ray) {
          const pos = def.cells.indexOf(idx);
          if (pos < 0) break;
          if (signs[pos] !== cornerSign) break;
          run += 1;
        }
        score += cornerSign * run * (1.55 + 1.35 * maturity);
      }
    } else {
      for (const adj of def.adjacent || []) {
        const pos = def.cells.indexOf(adj);
        if (pos >= 0 && signs[pos]) score -= signs[pos] * (3.9 + 2.8 * (1 - maturity));
      }
    }
    // Quiet corner shapes are better than noisy mixed 3x3 shapes around an open corner.
    score -= transitionPenalty(signs) * 0.10;
  } else if (def.type === 'edge2') {
    const edgeSigns = signs.slice(0, BOARD_SIZE);
    const innerSigns = signs.slice(BOARD_SIZE);
    const edgeDiff = edgeSigns.reduce((a, z) => a + z, 0);
    const innerDiff = innerSigns.reduce((a, z) => a + z, 0);
    score += edgeDiff * (0.65 + 0.85 * maturity);
    score -= innerDiff * (0.06 + 0.10 * (1 - maturity));
    score -= transitionPenalty(edgeSigns) * (0.14 + 0.10 * (1 - maturity));
    score += anchoredEdgeRunBonus(edgeSigns, maturity);
  } else if (def.type.startsWith('line') || def.type.startsWith('diag')) {
    score += linePatternScore(signs, maturity);
  } else if (def.type === 'ntuple2' || def.type === 'ntuple3') {
    const occupied = signs.filter(Boolean).length;
    if (occupied === signs.length && Math.abs(diff) === signs.length) score += diff * (0.04 + 0.08 * maturity);
    if (occupied === 1) score -= diff * (0.015 + 0.03 * (1 - maturity));
  }
  return score;
}

function transitionPenalty(signs) {
  let penalty = 0;
  for (let i = 1; i < signs.length; i += 1) {
    if (signs[i] && signs[i - 1] && signs[i] !== signs[i - 1]) penalty += 1;
  }
  return penalty;
}

function anchoredEdgeRunBonus(edgeSigns, maturity) {
  let score = 0;
  for (const start of [0, edgeSigns.length - 1]) {
    const sign = edgeSigns[start];
    if (!sign) continue;
    const step = start === 0 ? 1 : -1;
    let run = 0;
    for (let i = start; i >= 0 && i < edgeSigns.length; i += step) {
      if (edgeSigns[i] !== sign) break;
      run += 1;
    }
    score += sign * run * (0.18 + 0.32 * maturity);
  }
  return score;
}

function linePatternScore(signs, maturity) {
  let score = 0;
  let whiteRuns = 0;
  let blackRuns = 0;
  for (let i = 0; i < signs.length;) {
    if (!signs[i]) { i += 1; continue; }
    const sign = signs[i];
    let len = 0;
    while (i < signs.length && signs[i] === sign) { len += 1; i += 1; }
    if (sign > 0) whiteRuns += len * len;
    else blackRuns += len * len;
  }
  score += (whiteRuns - blackRuns) * (0.025 + 0.045 * maturity);
  score -= transitionPenalty(signs) * (0.07 + 0.05 * (1 - maturity));
  const left = signs[0];
  const right = signs[signs.length - 1];
  if (left && left === right) score += left * (0.22 + 0.32 * maturity);
  return score;
}

function patternOrderingDelta(b, side, move) {
  const before = patternEvalWhite(b);
  const after = patternEvalWhite(applyMoveToBoard(b, side, move));
  return (after - before) * sideSign(side);
}

function setCell(b, x, y, value) {
  b[y * BOARD_SIZE + x] = value;
  return attachBitboard(b);
}
function getCell(b, x, y) { return b[y * BOARD_SIZE + x]; }
function inBounds(x, y) { return x >= 0 && x < BOARD_SIZE && y >= 0 && y < BOARD_SIZE; }
function opponent(side) { return side === 'black' ? 'white' : 'black'; }
function sideSign(side) { return side === 'white' ? 1 : -1; }
function countsOf(b) {
  ensureBitboard(b);
  const black = b.__blackCount;
  const white = b.__whiteCount;
  return { black, white, empty: CELL_COUNT - black - white, whiteDiff: white - black, blackDiff: black - white };
}

function finalWhiteDiff(b) {
  const c = countsOf(b);
  const diff = c.white - c.black;
  if (diff > 0) return diff + c.empty;
  if (diff < 0) return diff - c.empty;
  return 0;
}
function legalMovesFor(b, side) {
  const key = `${boardKey(b)}|${side}`;
  const cached = legalCache.get(key);
  if (cached) return cached.slice();
  ensureBitboard(b);
  const occupied = b.__black | b.__white;
  const moves = [];
  for (let idx = 0; idx < CELL_COUNT; idx += 1) {
    if ((occupied & BB_BITS[idx]) !== 0n) continue;
    if (flipsMaskForMove(b, side, idx) !== 0n) moves.push(idx);
  }
  if (legalCache.size > 300000) legalCache.clear();
  legalCache.set(key, moves);
  return moves.slice();
}
function flipsForMove(b, side, idx) {
  const key = `${boardKey(b)}|${side}|${idx}`;
  const cached = flipCache.get(key);
  if (cached) return cached.slice();
  const flips = indicesFromMask(flipsMaskForMove(b, side, idx));
  if (flipCache.size > 300000) flipCache.clear();
  flipCache.set(key, flips);
  return flips.slice();
}
function applyMoveToBoard(b, side, move) {
  ensureBitboard(b);
  const next = b.slice(0, CELL_COUNT);
  if (move === PASS) return attachBitboard(next, b.__black, b.__white);
  const moveBit = BB_BITS[move];
  const flips = flipsMaskForMove(b, side, move);
  const flipIndices = indicesFromMask(flips);
  next[move] = side;
  for (const idx of flipIndices) next[idx] = side;
  if (side === 'black') {
    return attachBitboard(next, b.__black | moveBit | flips, b.__white & ~flips);
  }
  return attachBitboard(next, b.__black & ~flips, b.__white | moveBit | flips);
}
function isGameOver(b) {
  if (countsOf(b).empty === 0) return true;
  return legalMovesFor(b, 'black').length === 0 && legalMovesFor(b, 'white').length === 0;
}
function boardKey(b) {
  ensureBitboard(b);
  return b.__key;
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
  score += patternOrderingDelta(b, side, move) * 520;
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
  if (c.empty === 0 || (legalMovesFor(b, 'black').length === 0 && legalMovesFor(b, 'white').length === 0)) return finalWhiteDiff(b);

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
  const patternWeight = empty <= 6 ? 0.22 : empty <= 12 ? 0.46 : empty <= 20 ? 0.82 : 1.05;

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
    features.safeEdge * (0.28 + phase * 0.18) +
    features.pattern * patternWeight;

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
  const pattern = patternEvalWhite(b);
  const value = { mobility, potential, stability, parity, cornerDiff, edgeDiff, danger, frontier, pst, safeEdge, cornerAccess, pattern };
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
