import {
  TARGETS,
  angleError,
  clamp,
  cloneState,
  derivative,
  energyAtTarget,
  stepRK4,
  targetError,
  totalEnergy
} from "./physics.js";

const LQR_DT = 1 / 60;
const LQR_ITERATIONS = 190;

function zeros(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

function identity(n) {
  const I = zeros(n, n);
  for (let i = 0; i < n; i++) I[i][i] = 1;
  return I;
}

function transpose(A) {
  const rows = A.length;
  const cols = A[0].length;
  const T = zeros(cols, rows);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) T[c][r] = A[r][c];
  }
  return T;
}

function matMul(A, B) {
  const rows = A.length;
  const cols = B[0].length;
  const inner = B.length;
  const C = zeros(rows, cols);
  for (let r = 0; r < rows; r++) {
    for (let k = 0; k < inner; k++) {
      const a = A[r][k];
      if (a === 0) continue;
      for (let c = 0; c < cols; c++) C[r][c] += a * B[k][c];
    }
  }
  return C;
}

function matAdd(A, B) {
  return A.map((row, r) => row.map((v, c) => v + B[r][c]));
}

function matSub(A, B) {
  return A.map((row, r) => row.map((v, c) => v - B[r][c]));
}

function matScale(A, s) {
  return A.map(row => row.map(v => v * s));
}

function vectorToColumn(v) {
  return v.map(x => [x]);
}

function columnToVector(A) {
  return A.map(row => row[0]);
}

function diag(values) {
  const D = zeros(values.length, values.length);
  values.forEach((v, i) => { D[i][i] = v; });
  return D;
}

function discreteLqr(A, B, Q, RScalar) {
  let P = Q;
  const AT = transpose(A);
  const BT = transpose(B);

  for (let i = 0; i < LQR_ITERATIONS; i++) {
    const AT_P = matMul(AT, P);
    const AT_P_A = matMul(AT_P, A);
    const AT_P_B = matMul(AT_P, B);
    const BTP = matMul(BT, P);
    const BTPA = matMul(BTP, A);
    const BTPB = matMul(BTP, B)[0][0];
    const gainDen = 1 / Math.max(1e-9, RScalar + BTPB);
    const correction = matScale(matMul(AT_P_B, BTPA), gainDen);
    P = matAdd(Q, matSub(AT_P_A, correction));
  }

  const den = RScalar + matMul(matMul(BT, P), B)[0][0];
  // BT * P * A is a 1 x n row vector. The previous build accidentally
  // converted this row with columnToVector(), leaving only the first gain.
  // Returning the entire row is essential: the local balancer must feedback
  // both angles, both angular rates, support position and support velocity.
  return matScale(matMul(matMul(BT, P), A), 1 / Math.max(1e-9, den))[0];
}

function linearizeAtTarget(target, params) {
  const n = 6;
  const eps = 1e-5;

  function transition(z, a) {
    const s = {
      x: z[4],
      vx: z[5],
      th1: target.angles[0] + z[0],
      th2: target.angles[1] + z[1],
      om1: z[2],
      om2: z[3]
    };
    const next = stepRK4(s, a, params, 0, LQR_DT, false);
    return [
      angleError(next.th1, target.angles[0]),
      angleError(next.th2, target.angles[1]),
      next.om1,
      next.om2,
      next.x,
      next.vx
    ];
  }

  const Ad = zeros(n, n);
  const base = Array(n).fill(0);
  for (let c = 0; c < n; c++) {
    const zp = base.slice();
    const zm = base.slice();
    zp[c] += eps;
    zm[c] -= eps;
    const fp = transition(zp, 0);
    const fm = transition(zm, 0);
    for (let r = 0; r < n; r++) Ad[r][c] = (fp[r] - fm[r]) / (2 * eps);
  }

  const fp = transition(base, eps);
  const fm = transition(base, -eps);
  const Bd = zeros(n, 1);
  for (let r = 0; r < n; r++) Bd[r][0] = (fp[r] - fm[r]) / (2 * eps);

  return { Ad, Bd };
}

function makeLqrGain(target, params) {
  const { Ad, Bd } = linearizeAtTarget(target, params);

  // Use stronger local stabilization for the three unstable equilibria.
  // The cart terms are intentionally nonzero: without them, a finite rail
  // quickly loses authority even if the angular feedback is mathematically
  // stabilizing around an infinite cart track.
  const q1 = target.id === 0 ? 42 : (target.id === 3 ? 125 : 120);
  const q2 = target.id === 0 ? 42 : (target.id === 3 ? 125 : 135);
  const qOm1 = target.id === 0 ? 7 : 18;
  const qOm2 = target.id === 0 ? 7 : 19;
  const qCenter = target.id === 0 ? 1.8 : 9.5;
  const qCartVelocity = target.id === 0 ? 1.35 : 5.6;
  const Q = diag([q1, q2, qOm1, qOm2, qCenter, qCartVelocity]);
  const R = target.id === 0 ? 0.11 : (target.id === 2 ? 0.045 : 0.18);
  return discreteLqr(Ad, Bd, Q, R);
}

function lqrStateVector(state, target) {
  return [
    angleError(state.th1, target.angles[0]),
    angleError(state.th2, target.angles[1]),
    state.om1,
    state.om2,
    state.x,
    state.vx
  ];
}

function dot(a, b) {
  let out = 0;
  for (let i = 0; i < a.length; i++) out += a[i] * b[i];
  return out;
}

function closenessToTarget(state, target) {
  const [e1, e2] = targetError(state, target);
  const angleNorm = Math.sqrt(e1 * e1 + e2 * e2);
  const speedNorm = Math.sqrt(state.om1 * state.om1 + state.om2 * state.om2);
  return { angleNorm, speedNorm };
}

function energyPumpAcceleration(state, target, params) {
  const currentE = totalEnergy(state, params);
  const targetE = energyAtTarget(target, params);
  const energyError = targetE - currentE;

  const phasePower =
    (params.m1 + params.m2) * params.l1 * state.om1 * Math.cos(state.th1) +
    params.m2 * params.l2 * state.om2 * Math.cos(state.th2);

  if (Math.abs(phasePower) < 1e-5 || Math.abs(energyError) < 1e-4) return 0;

  // The same sign rule can pump energy upward for states 1/2/3 or drain
  // energy quickly for state 0, whose target energy is the minimum.
  const duty = target.id === 0 ? 0.86 : 1.0;
  return -duty * params.maxAcc * Math.sign(energyError * phasePower);
}


function centerReturnAcceleration(state, params, gainScale = 1.0) {
  const half = Math.max(0.01, params.segmentHalfLength);
  const ratio = Math.abs(state.x) / half;

  // PD return-to-center term; gains rise close to the rail edges.
  const edgeBoost = ratio > 0.55 ? 1 + 3.2 * Math.pow((ratio - 0.55) / 0.45, 2) : 1;
  const raw = (-1.35 * state.x - 1.85 * state.vx) * gainScale * edgeBoost;
  return clamp(raw, -0.72 * params.maxAcc, 0.72 * params.maxAcc);
}

function applyTrackSafety(state, raw, params) {
  const half = Math.max(0.01, params.segmentHalfLength);
  const ratio = Math.abs(state.x) / half;
  const outward = Math.sign(state.x || state.vx || 1);
  let safe = raw;

  if (ratio > 0.58) {
    const barrier = Math.pow((ratio - 0.58) / 0.42, 2);
    safe += -outward * params.maxAcc * 0.85 * barrier;
    safe += -2.25 * state.vx * (1 + 2.0 * barrier);
  }

  // Hard guard: near an edge, never allow acceleration that keeps driving outward.
  if (ratio > 0.88 && Math.sign(safe) === outward) {
    safe = Math.min(Math.abs(safe), 0.18 * params.maxAcc) * -outward;
  }

  // If already pushing the stop, brake strongly toward the center.
  if (ratio > 0.96) {
    safe = -outward * Math.max(0.35 * params.maxAcc, Math.abs(safe));
  }

  return clamp(safe, -params.maxAcc, params.maxAcc);
}

function downStabilizationAcceleration(state, params) {
  const e1 = angleError(state.th1, 0);
  const e2 = angleError(state.th2, 0);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(state.x) + 0.7 * Math.abs(state.vx);

  // The exact downward equilibrium should remain exactly still. A larger quiet
  // zone prevents numerical/controller noise from pumping energy into state 0.
  if (angleNorm < 0.018 && speedNorm < 0.025 && supportNorm < 0.020) return 0;

  const mixedAngle = 0.60 * e1 + 0.40 * e2;
  const mixedSpeed = 0.62 * state.om1 + 0.38 * state.om2;
  const centerA = centerReturnAcceleration(state, params);

  // For the downward equilibrium, the pivot acceleration sign is opposite to
  // upright balancing. Keep the angle term gentle and let gravity do most work.
  const raw =
    8.8 * mixedAngle +
    5.2 * mixedSpeed +
    1.12 * centerA;

  return clamp(raw, -0.66 * params.maxAcc, 0.66 * params.maxAcc);
}

let heuristicLqrCache = { id: -1, g: NaN, friction: NaN, gain: null };

function pseudoLqrAcceleration(state, target, params) {
  if (
    !heuristicLqrCache.gain ||
    heuristicLqrCache.id !== target.id ||
    Math.abs(heuristicLqrCache.g - params.g) > 1e-9 ||
    Math.abs(heuristicLqrCache.friction - (params.friction || 0)) > 1e-9
  ) {
    heuristicLqrCache = {
      id: target.id,
      g: params.g,
      friction: params.friction || 0,
      gain: makeLqrGain(target, params)
    };
  }
  const z = lqrStateVector(state, target);
  return clamp(-dot(heuristicLqrCache.gain, z), -params.maxAcc, params.maxAcc);
}

function targetAlignmentAcceleration(state, target, params) {
  if (target.id === 0) return downStabilizationAcceleration(state, params);

  const [e1, e2] = targetError(state, target);
  const supportBias = centerReturnAcceleration(state, params, 0.64);

  // During the capture phase use the real LQR direction whenever the state is
  // moderately close; otherwise use a coarse target-seeking PD term to seed MPC.
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const captureAngleLimit = target.id === 2 ? 1.20 : 0.85;
  const captureSpeedLimit = target.id === 2 ? 7.5 : 5.2;
  if (angleNorm < captureAngleLimit && speedNorm < captureSpeedLimit) {
    const pseudo = pseudoLqrAcceleration(state, target, params);
    return clamp(0.78 * pseudo + 0.22 * supportBias, -params.maxAcc, params.maxAcc);
  }

  const w1 = target.id === 1 ? 0.44 : 0.58;
  const w2 = target.id === 2 ? 0.44 : 0.58;
  const mixedAngle = w1 * e1 + w2 * e2;
  const mixedSpeed = 0.42 * state.om1 + 0.38 * state.om2;
  const raw = -10.5 * mixedAngle - 3.4 * mixedSpeed + supportBias;
  return clamp(raw, -params.maxAcc, params.maxAcc);
}

function scoreState(state, target, params, terminal = false) {
  const [e1, e2] = targetError(state, target);
  const angleCost = e1 * e1 + e2 * e2;
  const speedCost = state.om1 * state.om1 + state.om2 * state.om2;
  const half = Math.max(0.01, params.segmentHalfLength);
  const centerCost = (state.x / half) * (state.x / half) + 0.34 * state.vx * state.vx;

  const energyScale = Math.max(1, params.g * (params.m1 + params.m2) * params.l1 + params.g * params.m2 * params.l2);
  const energyCost = Math.pow((totalEnergy(state, params) - energyAtTarget(target, params)) / energyScale, 2);

  const edgeRatio = Math.abs(state.x) / half;
  const outwardSpeed = state.x * state.vx > 0 ? Math.abs(state.vx) : 0;
  const softEdge = edgeRatio > 0.62 ? Math.pow((edgeRatio - 0.62) / 0.38, 4) : 0;
  const hardEdge = edgeRatio > 0.84 ? Math.pow((edgeRatio - 0.84) / 0.16, 8) : 0;
  const edgeCost = 32 * softEdge + 260 * hardEdge + 3.2 * outwardSpeed * outwardSpeed * Math.max(0, edgeRatio - 0.52);

  if (terminal) {
    return 16.0 * angleCost + 2.2 * speedCost + 6.8 * centerCost + 0.75 * energyCost + 3.5 * edgeCost;
  }
  return 0.34 * angleCost + 0.060 * speedCost + 0.85 * centerCost + 0.20 * energyCost + 1.10 * edgeCost;
}

let controllerRngState = 0x6d2b79f5;

function resetControllerRandomForTarget(targetId, stateHint = null) {
  // Per-target deterministic seeds improve repeatability. State 0 also looks
  // at the current pose, because draining energy from state 1, state 2 and
  // state 3 benefits from different CEM exploration basins.
  let seed;
  if (targetId === 0 && stateHint) {
    const th1Down = Math.abs(angleError(stateHint.th1, 0));
    const th2Down = Math.abs(angleError(stateHint.th2, 0));
    const th1Up = Math.abs(angleError(stateHint.th1, Math.PI));
    const th2Up = Math.abs(angleError(stateHint.th2, Math.PI));
    if (th1Up < 0.82 && th2Up < 0.82) seed = 0xabcdef01;
    else if (th1Down < 0.82 && th2Up < 0.82) seed = 0x00003039;
    else if (th1Up < 0.82 && th2Down < 0.82) seed = 0x00000001;
    else seed = 0x9e3779b9;
  } else {
    const seeds = [0x9e3779b9, 0x00000001, 0xdeadbeef, 0x000001c8];
    seed = seeds[targetId] ?? 0x6d2b79f5;
  }
  controllerRngState = seed;
}

function randomUnit() {
  // Deterministic PRNG: repeatable control behavior across browsers and runs.
  controllerRngState = (1664525 * controllerRngState + 1013904223) >>> 0;
  return controllerRngState / 0x100000000;
}

function randomBetween(min, max) {
  return min + randomUnit() * (max - min);
}

function gaussianRandom() {
  let u = 0;
  let v = 0;
  while (u === 0) u = randomUnit();
  while (v === 0) v = randomUnit();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

function makeCandidateSequence(index, horizon, A, energyA, alignA) {
  const seq = new Array(horizon);

  if (index === 0) return seq.fill(energyA);
  if (index === 1) return seq.fill(alignA);
  if (index === 2) return seq.fill(0);
  if (index === 3) return seq.fill(A);
  if (index === 4) return seq.fill(-A);
  if (index === 5) {
    for (let i = 0; i < horizon; i++) seq[i] = i < horizon / 2 ? A : -A;
    return seq;
  }
  if (index === 6) {
    for (let i = 0; i < horizon; i++) seq[i] = i < horizon / 2 ? -A : A;
    return seq;
  }

  const pieces = 5;
  const pieceLen = Math.ceil(horizon / pieces);
  const bias = 0.35 * energyA + 0.25 * alignA;
  for (let p = 0; p < pieces; p++) {
    const value = clamp(bias + randomBetween(-A, A), -A, A);
    for (let i = p * pieceLen; i < Math.min(horizon, (p + 1) * pieceLen); i++) seq[i] = value;
  }
  return seq;
}

function runPrediction(state, target, params, sequence, t0) {
  let s = cloneState(state);
  let cost = 0;
  const dt = 0.036;
  for (let i = 0; i < sequence.length; i++) {
    // RK4 prediction is more expensive than semi-implicit Euler, but it greatly
    // reduces artificial drift in the model predictive controller.
    const safeA = applyTrackSafety(s, sequence[i], params);
    s = stepRK4(s, safeA, params, t0 + i * dt, dt, false);
    cost += scoreState(s, target, params, false);
    cost += 0.0020 * safeA * safeA / Math.max(1, params.maxAcc * params.maxAcc);
    if (s.x * safeA > 0) cost += 0.0025 * Math.abs(s.x / Math.max(0.01, params.segmentHalfLength));
  }
  cost += scoreState(s, target, params, true);
  return cost;
}

function vectorMean(vectors) {
  const out = Array(vectors[0].length).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < out.length; i++) out[i] += v[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= vectors.length;
  return out;
}

function vectorStd(vectors, mean, floor) {
  const out = Array(mean.length).fill(0);
  for (const v of vectors) {
    for (let i = 0; i < out.length; i++) {
      const d = v[i] - mean[i];
      out[i] += d * d;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = Math.max(floor, Math.sqrt(out[i] / vectors.length));
  return out;
}

function blocksToSequence(blocks, horizon, blockLen) {
  const seq = new Array(horizon);
  for (let i = 0; i < horizon; i++) seq[i] = blocks[Math.min(blocks.length - 1, Math.floor(i / blockLen))];
  return seq;
}

function sequenceToBlocks(sequence, blockCount, blockLen, fallback) {
  const blocks = new Array(blockCount);
  for (let b = 0; b < blockCount; b++) {
    const idx = b * blockLen;
    blocks[b] = sequence && idx < sequence.length ? sequence[idx] : fallback;
  }
  return blocks;
}

function makeHeuristicBlocks(index, blockCount, A, energyA, alignA, centerA, target) {
  const blocks = new Array(blockCount);
  const bias = clamp(0.46 * energyA + 0.23 * alignA + 0.31 * centerA, -A, A);

  if (index === 0) return blocks.fill(energyA);
  if (index === 1) return blocks.fill(alignA);
  if (index === 2) return blocks.fill(centerA);
  if (index === 3) return blocks.fill(bias);
  if (index === 4) return blocks.fill(0);
  if (index === 5) return blocks.fill(A);
  if (index === 6) return blocks.fill(-A);

  for (let i = 0; i < blockCount; i++) {
    const phase = i / Math.max(1, blockCount - 1);
    if (index === 7) blocks[i] = A * Math.sin(2 * Math.PI * (1.0 + 0.15 * target.id) * phase);
    else if (index === 8) blocks[i] = A * Math.sin(2 * Math.PI * (1.5 + 0.2 * target.id) * phase + Math.PI / 2);
    else if (index === 9) blocks[i] = i % 2 === 0 ? A : -A;
    else if (index === 10) blocks[i] = i < blockCount / 2 ? A : -A;
    else if (index === 11) blocks[i] = i < blockCount / 2 ? -A : A;
    else blocks[i] = clamp(bias + randomBetween(-0.65 * A, 0.65 * A), -A, A);
  }
  return blocks;
}

function chooseCemAcceleration(state, target, params, t, previousPlan) {
  const A = params.maxAcc;
  const horizon = 42;
  const blockLen = 3;
  const blockCount = Math.ceil(horizon / blockLen);
  const sampleCount = 54;
  const eliteCount = 8;
  const generations = 2;
  const energyA = energyPumpAcceleration(state, target, params);
  const alignA = targetAlignmentAcceleration(state, target, params);
  const centerA = centerReturnAcceleration(state, params);

  let mean = sequenceToBlocks(previousPlan, blockCount, blockLen, clamp(0.46 * energyA + 0.23 * alignA + 0.31 * centerA, -A, A));
  let std = Array(blockCount).fill(Math.max(0.18 * A, 1.5));

  let bestBlocks = mean.slice();
  let bestCost = Infinity;

  for (let gen = 0; gen < generations; gen++) {
    const candidates = [];

    for (let i = 0; i < sampleCount; i++) {
      let blocks;
      if (gen === 0 && i < 13) {
        blocks = makeHeuristicBlocks(i, blockCount, A, energyA, alignA, centerA, target);
      } else {
        blocks = mean.map((m, j) => clamp(m + gaussianRandom() * std[j], -A, A));
      }

      const seq = blocksToSequence(blocks, horizon, blockLen);
      const cost = runPrediction(state, target, params, seq, t);
      candidates.push({ blocks, cost });

      if (cost < bestCost) {
        bestCost = cost;
        bestBlocks = blocks.slice();
      }
    }

    candidates.sort((a, b) => a.cost - b.cost);
    const elites = candidates.slice(0, eliteCount).map(c => c.blocks);
    mean = vectorMean(elites);
    std = vectorStd(elites, mean, Math.max(0.055 * A, 0.35));
  }

  const bestSequence = blocksToSequence(bestBlocks, horizon, blockLen);
  return {
    acceleration: clamp(bestSequence[0], -A, A),
    plan: bestSequence.slice(1)
  };
}


export class PendulumController {
  constructor(params) {
    this.target = TARGETS[0];
    this.cachedTargetId = -1;
    this.cachedGravity = NaN;
    this.cachedFriction = NaN;
    this.cachedGain = null;
    this.commandAcc = 0;
    this.controlAccumulator = 0;
    this.plan = [];
  }

  setTarget(id, stateHint = null) {
    this.target = TARGETS[id] || TARGETS[0];
    this.controlAccumulator = 0;
    this.plan = [];
    resetControllerRandomForTarget(this.target.id, stateHint);
    if (this.target.id === 0) this.commandAcc = 0;
  }

  recomputeGainIfNeeded(params) {
    if (
      !this.cachedGain ||
      this.cachedTargetId !== this.target.id ||
      Math.abs(this.cachedGravity - params.g) > 1e-9 ||
      Math.abs(this.cachedFriction - (params.friction || 0)) > 1e-9
    ) {
      this.cachedGain = makeLqrGain(this.target, params);
      this.cachedTargetId = this.target.id;
      this.cachedGravity = params.g;
      this.cachedFriction = params.friction || 0;
    }
  }

  lqrAcceleration(state, params) {
    this.recomputeGainIfNeeded(params);
    const z = lqrStateVector(state, this.target);
    return clamp(-dot(this.cachedGain, z), -params.maxAcc, params.maxAcc);
  }

  update(state, params, dt, t) {
    this.controlAccumulator += dt;
    const shouldUpdate = this.controlAccumulator >= 0.045;
    if (!shouldUpdate) return this.commandAcc;
    this.controlAccumulator = 0;

    let raw;
    if (this.target.id === 0) {
      const nearDown = closenessToTarget(state, this.target);
      const quietEnough = nearDown.angleNorm < 0.020 && nearDown.speedNorm < 0.030 && Math.abs(state.x) + 0.7 * Math.abs(state.vx) < 0.025;
      if (quietEnough) {
        raw = 0;
        this.plan = [];
      } else if (nearDown.angleNorm < 0.58 && nearDown.speedNorm < 2.25) {
        raw = downStabilizationAcceleration(state, params);
        this.plan = [];
      } else {
        const planned = chooseCemAcceleration(state, this.target, params, t, this.plan);
        raw = 0.86 * planned.acceleration + 0.14 * downStabilizationAcceleration(state, params);
        this.plan = planned.plan;
      }
    } else {
      const near = closenessToTarget(state, this.target);
      const lqrAngleLimit = this.target.id === 2 ? 0.92 : 0.55;
      const lqrSpeedLimit = this.target.id === 2 ? 7.2 : 4.2;
      const useLqr = near.angleNorm < lqrAngleLimit && near.speedNorm < lqrSpeedLimit;
      if (useLqr) {
        raw = 0.93 * this.lqrAcceleration(state, params) + 0.07 * centerReturnAcceleration(state, params);
        this.plan = [];
      } else {
        const planned = chooseCemAcceleration(state, this.target, params, t, this.plan);
        raw = planned.acceleration;
        this.plan = planned.plan;
      }
    }

    raw = applyTrackSafety(state, raw, params);

    // Slew-limited actuator smoothing keeps motion physical while still obeying acceleration-only control.
    const maxDelta = Math.max(6.0, params.maxAcc * 1.20);
    const blended = 0.96 * raw + 0.04 * this.commandAcc;
    this.commandAcc = clamp(blended, this.commandAcc - maxDelta, this.commandAcc + maxDelta);
    this.commandAcc = applyTrackSafety(state, clamp(this.commandAcc, -params.maxAcc, params.maxAcc), params);
    if (Math.abs(this.commandAcc) < 1e-5) this.commandAcc = 0;
    return this.commandAcc;
  }
}
