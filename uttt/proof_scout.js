// V14: rolling wall-clock allocator and neural-PV proof frontier selection.
// These are scheduling hints, NEVER evidence that any unsearched move is bad.
const clamp01 = x => Math.max(0, Math.min(1, x));
const sigmoid = x => 1 / (1 + Math.exp(-x));

export class RollingAllocator {
  constructor(initialModelMs = 0) {
    this.used = {model:initialModelMs, pvs:0, proof:0};
    this.otherMs = 0;
  }
  add(channel, elapsed) { this.used[channel] += Math.max(0, elapsed); }
  choose(shares) {
    // A weighted fair queue, with carry-over debt across ALL search cycles.
    // It has no "reset to model" bias, including in indefinite analysis.
    return ['model','pvs','proof'].reduce((best, key) =>
      this.used[key] / shares[key] < this.used[best] / shares[best] ? key : best, 'model');
  }
  total() { return this.used.model + this.used.pvs + this.used.proof; }
}

export function adaptiveShares(phase, modelShare, frontier = null, root = null, proofStats = null) {
  const signal = clamp01(phase?.urgency || 0);
  const depth = frontier?.depth || 0;
  const closing = frontier?.closing || 0;
  const narrow = clamp01(frontier?.narrow || 0);
  const progress = root?.certLower > -1 || root?.certUpper < 1 ? 1 : 0;
  const evidence = clamp01(.50*signal + .28*Math.tanh(depth/5) +
    .14*Math.tanh(closing/2) + .08*narrow + .30*progress);
  const yields = (proofStats?.usefulCertificates || 0) / (1 + (proofStats?.attempts || 0));
  const failures = (proofStats?.attempts || 0) - (proofStats?.certificates || 0);
  // Low-yield proof work gradually loses its allocation; new structural
  // evidence and partial WDL can still restore it. No sudden phase cutoffs.
  const proofShare = .47 * sigmoid(8*(evidence - .42) + 2*yields - 1.05*Math.log1p(Math.max(0, failures)));
  const nonProof = 1 - proofShare;
  const m = Math.max(.18, Math.min(.90, modelShare || .75));
  return {model:nonProof*m, pvs:nonProof*(1-m), proof:proofShare};
}

const rankedEdges = node => (node.edges || []).slice().sort((a,b)=>
  (b.child?.n||0) + b.prior * (node.n + 3) - (a.child?.n||0) - a.prior * (node.n + 3));

export class PVScout {
  constructor(mcts) { this.mcts = mcts; this.expansions = 0; this.lastFrontier = null; }

  // Expand at most two network leaves per invocation, maintaining a continuous
  // frontier in a long-running analysis rather than an expensive initial burst.
  advance() {
    const root = this.mcts.root;
    if (!root.edges || root.solved) return;
    const candidates = rankedEdges(root);
    // Rotate between the strongest principal line and plausible alternatives.
    // The second root move is explored only if network/MCTS assigns it weight.
    const alternate = candidates.length > 1 &&
      candidates[1].prior >= candidates[0].prior * .17;
    const rootChoice = alternate && (this.expansions % 5 === 4) ? candidates[1] : candidates[0];
    if (!rootChoice) return;
    let node = this.mcts.child(root,rootChoice);
    let built = 0;
    const goal = 3 + Math.floor(5 * sigmoid((this.expansions - 5)/5));
    for (let depth=1; depth<=goal && !node.solved && !node.game.winner; depth++) {
      if (node.edges === null) {
        if (built >= 2) break;
        this.mcts.evaluate(node);
        built++;
      }
      const edges = rankedEdges(node);
      if (!edges.length) break;
      // Deep alternative replies deserve active scouting as well as the main
      // neural PV: a seemingly forced line is not a mathematical proof.
      const alt = edges.length>1 && edges[1].prior >= edges[0].prior*.09 &&
        (this.expansions + depth*3)%7===0;
      node = this.mcts.child(node, alt ? edges[1] : edges[0]);
    }
    this.expansions++;
    return built;
  }

  frontiers() {
    const root = this.mcts.root;
    const closed0 = root.game.boards.reduce((s,x)=>s+!!x,0);
    const open0 = root.game.legal().length;
    const found = [];
    const stack = [[root,0,1]];
    while (stack.length && found.length < 400) {
      const [node,depth,pathSupport] = stack.pop();
      if (!node.solved && !node.game.winner) {
        const closing = node.game.boards.reduce((s,x)=>s+!!x,0)-closed0;
        const nlegal = node.game.legal().length;
        const narrow = clamp01(1 - nlegal / Math.max(1,open0));
        const parentStrength = pathSupport;
        // Future solved nodes are useful only when they can influence a
        // plausible continuation. Score reflects structural compression,
        // potential root impact, and prior failed work on this same node.
        const oneSided=(node.attackResults[1]===0 || node.attackResults[2]===0) ? 1 : 0;
        const priority = (.6 + 1.0*Math.tanh(depth/4) + .62*closing + .65*narrow +
          .6*Math.sqrt(Math.max(.01,parentStrength)) + 1.55*oneSided) /
          (1 + .24*Math.log1p(node.proofSpentMs) + .18*node.proofAttempts);
        found.push({node,depth,closing,narrow,priority});
      }
      if (depth >= 10 || !node.edges) continue;
      const ranked = rankedEdges(node);
      for (let k=ranked.length-1;k>=0;k--) {
        const e=ranked[k];
        if (!e.child) continue;
        const confidence = pathSupport * e.prior;
        if (k>1 && e.child.n<3) continue;
        stack.push([e.child,depth+1,Math.max(.005,confidence)]);
      }
    }
    found.sort((a,b)=>b.priority-a.priority);
    this.lastFrontier = found[0] || null;
    return found;
  }

  pick() {
    const items=this.frontiers();
    if (!items.length) return null;
    // A slight rotation prevents repeated failed proof calls on just one leaf.
    const head=items.slice(0,Math.min(6,items.length));
    return head[this.expansions % head.length];
  }
}
