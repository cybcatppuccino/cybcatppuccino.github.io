// Stage2 full-teacher PUCT. Network estimates guide unproved branches; certified
// WDL bounds are separate minimax facts and always override heuristic values.
const CPuctBase = 19652;
const CPuctInit = 1.45;
const UNIFORM_PRIOR_MIX = 0.025;
const WDL_SCORE = 29999;

function outcome(game) {
  if (game.winner === 3) return 0;
  return game.winner === game.turn ? 1 : -1;
}

class Node {
  constructor(game, prior = 1, move = -1, parent = null) {
    this.game = game;
    this.prior = prior;
    this.move = move;
    this.parent = parent;
    this.n = 0;
    this.sum = 0;
    this.edges = null;
    this.estimated = 0;
    this.evalMs = 0;
    this.evaluated = false;
    this.neuralEvaluated = false;
    // Bounds are from THIS node's side to move, not from X or the root.
    // A certificate intersects the minimax bounds deduced from children.
    this.certLower = game.winner ? outcome(game) : -1;
    this.certUpper = game.winner ? outcome(game) : 1;
    this.lower = this.certLower;
    this.upper = this.certUpper;
    this.proofBest = -1;
    this.proofAttempts = 0;
    this.proofSpentMs = 0;
    this.attackResults = new Int8Array(3).fill(-1);
  }
  get mean() { return this.n ? this.sum / this.n : this.estimated; }
  get solved() { return this.lower === this.upper; }
}

export class Stage2MCTS {
  constructor(game, net) {
    this.root = new Node(game.clone());
    this.net = net;
    this.evaluations = 0;
    this.evaluationTimeMs = 0;
    this.maxDepth = 0;
    this.simulations = 0;
    this.lastEvaluationMs = 0;
    this.boundsUpdated = 0;
  }

  evaluate(node) {
    if (node.game.winner || node.solved) {
      node.estimated = node.lower;
      node.edges = [];
      node.evaluated = true;
      return node.estimated;
    }
    const legal = node.game.legal();
    if (!legal.length) {
      node.estimated = 0;
      node.edges = [];
      node.evaluated = true;
      node.certLower = node.certUpper = node.lower = node.upper = 0;
      return 0;
    }

    const t0 = performance.now();
    const { value, logits } = this.net.predict(node.game, legal);
    const elapsed = Math.max(0, performance.now() - t0);
    node.evalMs = elapsed;
    node.evaluated = true;
    node.neuralEvaluated = true;
    this.lastEvaluationMs = elapsed;
    this.evaluationTimeMs += elapsed;
    this.evaluations++;
    node.estimated = Math.max(-1, Math.min(1, Number(value) || 0));

    let top = -Infinity;
    for (const move of legal) if (logits[move] > top) top = logits[move];
    const raw = new Array(legal.length);
    let sum = 0;
    for (let i = 0; i < legal.length; i++) {
      const p = Math.exp(Math.max(-32, Math.min(0, logits[legal[i]] - top)));
      raw[i] = p;
      sum += p;
    }
    if (!(sum > 0) || !Number.isFinite(sum)) sum = 1;
    const uniform = 1 / legal.length;
    node.edges = legal.map((move, i) => ({
      move,
      prior: (1 - UNIFORM_PRIOR_MIX) * (raw[i] / sum) + UNIFORM_PRIOR_MIX * uniform,
      child: null,
    }));
    return node.estimated;
  }

  child(node, edge) {
    if (!edge.child) {
      const next = node.game.clone();
      next.play(edge.move);
      edge.child = new Node(next, edge.prior, edge.move, node);
      this.updateBounds(node);
    }
    return edge.child;
  }

  updateBounds(node) {
    let lower = node.certLower, upper = node.certUpper;
    if (node.edges && node.edges.length) {
      let childLower = -1, childUpper = -1;
      for (const edge of node.edges) {
        const child = edge.child;
        const lo = child ? -child.upper : -1;
        const hi = child ? -child.lower : 1;
        if (lo > childLower) childLower = lo;
        if (hi > childUpper) childUpper = hi;
      }
      lower = Math.max(lower, childLower);
      upper = Math.min(upper, childUpper);
    }
    if (lower === 0) lower = 0;
    if (upper === 0) upper = 0;
    if (lower > upper) throw Error('Contradictory strict WDL certificates');
    if (lower !== node.lower || upper !== node.upper) {
      node.lower = lower;
      node.upper = upper;
      this.boundsUpdated++;
      return true;
    }
    return false;
  }

  propagateBounds(node) {
    while (node) {
      const changed = this.updateBounds(node);
      if (!changed) break;
      node = node.parent;
    }
  }

  // Result of "can attacker force a win?" is 1 (yes), 0 (no), -1 (unknown).
  // In particular, a single NO is a real one-sided certificate, not a draw.
  certifyAttack(node, attacker, result, witness = -1) {
    if (result !== 0 && result !== 1) return false;
    if (attacker !== 1 && attacker !== 2) throw Error('Invalid attacker');
    if (node.attackResults[attacker] >= 0 && node.attackResults[attacker] !== result)
      throw Error('Conflicting AND/OR proof results');
    node.attackResults[attacker] = result;
    if (attacker === node.game.turn) {
      if (result === 1) { node.certLower = 1; node.certUpper = 1; }
      else node.certUpper = Math.min(node.certUpper, 0);
      if (result === 1 && node.game.isLegal(witness)) node.proofBest = witness;
    } else {
      if (result === 1) { node.certLower = -1; node.certUpper = -1; }
      else {
        node.certLower = Math.max(node.certLower, 0);
        // Defender's witness refutes the opponent's forced-win claim.
        if (node.game.isLegal(witness)) node.proofBest = witness;
      }
    }
    // Root/child witness is itself a valid proof fact, not just a move hint.
    if (node.game.isLegal(witness) && node.edges) {
      const edge = node.edges.find(e => e.move === witness);
      if (edge && ((attacker === node.game.turn && result === 1) ||
        (attacker !== node.game.turn && result === 0))) {
        const child = this.child(node, edge);
        if (attacker === node.game.turn) child.certLower = child.certUpper = -1;
        else child.certUpper = Math.min(child.certUpper, 0);
        this.propagateBounds(child);
      }
    }
    this.propagateBounds(node);
    return true;
  }

  certifyWdl(node, wdl, witness = -1) {
    if (![1,0,-1].includes(wdl)) throw Error('Invalid exact WDL');
    node.certLower = node.certUpper = wdl;
    if (node.game.isLegal(witness) && wdl >= 0) node.proofBest = witness;
    this.propagateBounds(node);
  }

  start() {
    if (this.root.edges === null) this.evaluate(this.root);
    return this.snapshot();
  }

  cpuct(node) {
    return CPuctInit + 1.65 * Math.log((node.n + CPuctBase + 1) / CPuctBase);
  }

  simulation() {
    let node = this.root;
    let depth = 0;
    let value;
    const path = this.path || (this.path = []);
    path.length = 0;
    path.push(node);

    while (true) {
      if (node.solved) { value = node.lower; break; }
      if (node.edges === null) { value = this.evaluate(node); break; }
      if (node.game.winner || !node.edges.length) { value = outcome(node.game); break; }

      let best = -Infinity, selected = null;
      const scale = this.cpuct(node) * Math.sqrt(Math.max(1, node.n));
      const fpu = node.mean - 0.18;
      for (const edge of node.edges) {
        const child = edge.child;
        const lo = child ? -child.upper : -1;
        const hi = child ? -child.lower : 1;
        // If a parent has a certified achievable result, any child whose upper
        // bound is worse cannot improve the decision and may safely be skipped.
        if (hi < node.lower) continue;
        const visits = child ? child.n : 0;
        const q = child?.solved ? -child.lower : visits ? -child.mean : fpu;
        const certifiedQ = Math.min(hi, Math.max(lo, q));
        const u = scale * edge.prior / (1 + visits);
        const score = certifiedQ + u;
        if (score > best) { best = score; selected = edge; }
      }
      if (!selected) { value = node.lower; break; }
      node = this.child(node, selected);
      path.push(node);
      depth++;
    }
    if (depth > this.maxDepth) this.maxDepth = depth;
    for (let i = path.length - 1; i >= 0; i--) {
      path[i].n++;
      path[i].sum += value;
      value = -value;
    }
    this.simulations++;
    // This is cheap and also incorporates newly reached terminal children.
    for (let i = path.length - 1; i >= 0; i--) this.updateBounds(path[i]);
  }

  adoptSubtree(root) {
    this.root = root;
    root.parent = null;
    const stats = this.subtreeStats(root);
    this.simulations = root.n;
    this.evaluations = stats.evaluations;
    this.evaluationTimeMs = stats.evaluationTimeMs;
    this.maxDepth = stats.maxDepth;
  }

  subtreeStats(root = this.root) {
    let evaluations = 0, evaluationTimeMs = 0, maxDepth = 0;
    const stack = [[root, 0]];
    while (stack.length) {
      const [node, depth] = stack.pop();
      if (depth > maxDepth) maxDepth = depth;
      if (node.neuralEvaluated) { evaluations++; evaluationTimeMs += node.evalMs; }
      if (node.edges) for (const edge of node.edges) if (edge.child) stack.push([edge.child, depth + 1]);
    }
    return { evaluations, evaluationTimeMs, maxDepth };
  }

  snapshot() {
    const root = this.root, sign = root.game.turn === 1 ? 1 : -1;
    const list = (root.edges || []).map(edge => {
      const child = edge.child, visits = child?.n || 0;
      const lowerWdl = child ? -child.upper : -1;
      const upperWdl = child ? -child.lower : 1;
      const exact = lowerWdl === upperWdl;
      const parentValue = exact ? lowerWdl : visits ?
        Math.min(upperWdl, Math.max(lowerWdl, -child.mean)) :
        Math.min(upperWdl, Math.max(lowerWdl, root.mean));
      return {move: edge.move, visits, prior: edge.prior,
        valueX: Math.max(-1, Math.min(1, sign * parentValue)),
        score: exact ? lowerWdl * WDL_SCORE : null,
        bound: false, solved: exact, lowerWdl, upperWdl,
        proofKind: exact ? 'WDL (distance unproven)' : undefined};
    });
    list.sort((a, b) => b.visits - a.visits || b.prior - a.prior);
    // A certified win outranks any heuristic; a certified safe draw is preferred
    // when no unproved alternative has a convincing win assessment.
    const win = list.find(c => c.lowerWdl === 1);
    const safe = list.filter(c => c.lowerWdl >= 0);
    const guessed = list[0];
    const draw = safe.sort((a,b) => b.lowerWdl-a.lowerWdl || b.visits-a.visits)[0];
    const best = win?.move ?? (root.lower >= 0 && root.proofBest >= 0 ? root.proofBest :
      (draw && (!guessed || guessed.valueX * sign < 0.35) ? draw.move : guessed?.move)) ?? -1;
    const rootValue = root.solved ? root.lower : Math.max(root.lower, Math.min(root.upper, root.mean));
    return {depth:this.maxDepth,best,
      score:root.solved?root.lower*WDL_SCORE:null,
      valueX:Math.max(-1,Math.min(1,sign*rootValue)),
      lowerWdl:root.lower,upperWdl:root.upper,
      candidates:list,visits:root.n,simulations:this.simulations,
      model:{format:'Stage2 FP16',evaluations:this.evaluations,inferenceMs:this.evaluationTimeMs},
      solved:root.solved};
  }
}
