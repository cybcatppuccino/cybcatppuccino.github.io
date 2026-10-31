// Stage2 full-teacher PUCT. Every newly expanded non-terminal leaf is evaluated
// by the unchanged original Stage2 network; values are backed up with side-to-move
// sign changes. The tree is persistent and may be reused only for an identical state.
const CPuctBase = 19652;
const CPuctInit = 1.45;
const UNIFORM_PRIOR_MIX = 0.025;

function outcome(game) {
  if (game.winner === 3) return 0;
  // Game.turn is already the side to move after the terminal move.
  return game.winner === game.turn ? 1 : -1;
}

class Node {
  constructor(game, prior = 1, move = -1) {
    this.game = game;
    this.prior = prior;
    this.move = move;
    this.n = 0;
    this.sum = 0;
    this.edges = null;
    this.estimated = 0;
    this.evalMs = 0;
    this.evaluated = false;
    this.neuralEvaluated = false;
  }
  get mean() { return this.n ? this.sum / this.n : this.estimated; }
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
  }

  evaluate(node) {
    if (node.game.winner) {
      node.estimated = outcome(node.game);
      node.edges = [];
      node.evaluated = true;
      return node.estimated;
    }
    const legal = node.game.legal();
    if (!legal.length) {
      node.estimated = 0;
      node.edges = [];
      node.evaluated = true;
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
      const move = legal[i];
      const p = Math.exp(Math.max(-32, Math.min(0, logits[move] - top)));
      raw[i] = p;
      sum += p;
    }
    if (!(sum > 0) || !Number.isFinite(sum)) sum = 1;
    const uniform = 1 / legal.length;
    const edges = new Array(legal.length);
    for (let i = 0; i < legal.length; i++) {
      // A small uniform floor keeps legal tactical moves discoverable even if
      // the policy head assigns them very little probability.
      const prior = (1 - UNIFORM_PRIOR_MIX) * (raw[i] / sum) + UNIFORM_PRIOR_MIX * uniform;
      edges[i] = { move: legal[i], prior, child: null };
    }
    node.edges = edges;
    return node.estimated;
  }

  start() {
    if (this.root.edges === null) this.evaluate(this.root);
    return this.snapshot();
  }

  cpuct(node) {
    // AlphaZero-style logarithmic exploration growth. It lets the policy guide
    // early visits without freezing low-prior moves out at larger visit counts.
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
      if (node.edges === null) {
        value = this.evaluate(node);
        break;
      }
      if (node.game.winner || !node.edges.length) {
        value = outcome(node.game);
        break;
      }

      let best = -Infinity;
      let selected = null;
      const parentVisits = Math.max(1, node.n);
      const scale = this.cpuct(node) * Math.sqrt(parentVisits);
      const fpu = node.mean - 0.18;
      for (const edge of node.edges) {
        const child = edge.child;
        const visits = child ? child.n : 0;
        // Child values are stored from the child's side-to-move perspective.
        const q = visits ? -child.mean : fpu;
        const u = scale * edge.prior / (1 + visits);
        const score = q + u;
        if (score > best) {
          best = score;
          selected = edge;
        }
      }

      if (!selected.child) {
        const next = node.game.clone();
        next.play(selected.move);
        selected.child = new Node(next, selected.prior, selected.move);
      }
      node = selected.child;
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
  }

  adoptSubtree(root) {
    this.root = root;
    const stats = this.subtreeStats(root);
    this.simulations = root.n;
    this.evaluations = stats.evaluations;
    this.evaluationTimeMs = stats.evaluationTimeMs;
    this.maxDepth = stats.maxDepth;
  }

  subtreeStats(root = this.root) {
    let evaluations = 0;
    let evaluationTimeMs = 0;
    let maxDepth = 0;
    const stack = [[root, 0]];
    while (stack.length) {
      const [node, depth] = stack.pop();
      if (depth > maxDepth) maxDepth = depth;
      if (node.neuralEvaluated) {
        evaluations++;
        evaluationTimeMs += node.evalMs;
      }
      if (node.edges) {
        for (const edge of node.edges) if (edge.child) stack.push([edge.child, depth + 1]);
      }
    }
    return { evaluations, evaluationTimeMs, maxDepth };
  }

  snapshot() {
    const root = this.root;
    const sign = root.game.turn === 1 ? 1 : -1;
    const list = (root.edges || []).map(edge => {
      const child = edge.child;
      const visits = child?.n || 0;
      const parentValue = visits ? -child.mean : root.mean;
      const valueX = Math.max(-1, Math.min(1, sign * parentValue));
      const candidate = {
        move: edge.move,
        visits,
        prior: edge.prior,
        valueX,
        score: null,
        bound: false,
      };
      if (child?.game.winner) {
        candidate.score = child.game.winner === 3 ? 0 :
          child.game.winner === root.game.turn ? 29999 : -29999;
        candidate.solved = true;
      }
      return candidate;
    });
    list.sort((a, b) => b.visits - a.visits || b.prior - a.prior);

    // A terminal winning child is authoritative even before it accumulates
    // enough visits to become the most-visited edge.
    const immediateWin = list.find(c => c.solved && c.score > 0);
    const best = immediateWin?.move ?? list[0]?.move ?? -1;
    const valueX = Math.max(-1, Math.min(1, sign * root.mean));
    return {
      depth: this.maxDepth,
      best,
      score: null,
      valueX,
      candidates: list,
      visits: root.n,
      simulations: this.simulations,
      model: {
        format: 'Stage2 FP16',
        evaluations: this.evaluations,
        inferenceMs: this.evaluationTimeMs,
      },
      solved: !!immediateWin,
    };
  }
}
