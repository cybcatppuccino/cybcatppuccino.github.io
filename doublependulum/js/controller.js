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
  const q1 = target.id === 0 ? 42 : (target.id === 3 ? 155 : 135);
  const q2 = target.id === 0 ? 42 : (target.id === 3 ? 165 : 145);
  const qOm1 = target.id === 0 ? 7 : (target.id === 3 ? 25 : 20);
  const qOm2 = target.id === 0 ? 7 : (target.id === 3 ? 27 : 21);

  // The finite rail still matters, but near the unstable target the local
  // controller must first keep the links balanced. A large cart-centering term
  // was the main cause of the old target-3 overshoot: when the links were
  // already almost upright, LQR would spend too much authority trying to move
  // x back to zero and the links fell out of the capture basin.
  const roughEnvironment = (params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15;
  const qCenter = target.id === 0 ? 8.0 : (target.id === 3 ? (roughEnvironment ? 1.15 : 1.75) : 2.25);
  const qCartVelocity = target.id === 0 ? 4.6 : (target.id === 3 ? (roughEnvironment ? 0.82 : 1.20) : 1.65);
  const Q = diag([q1, q2, qOm1, qOm2, qCenter, qCartVelocity]);
  const R = target.id === 0 ? 0.08 : (target.id === 2 ? 0.050 : (target.id === 3 ? 0.090 : 0.13));
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

function supportPhasePower(state, params) {
  // ∂E/∂x_dot-like coupling term. Choosing a support acceleration with the
  // same sign gives dE/dt ≈ -a * phasePower, so near state 0 it removes
  // pendulum energy continuously instead of bang-bang pumping it.
  return (params.m1 + params.m2) * params.l1 * state.om1 * Math.cos(state.th1) +
    params.m2 * params.l2 * state.om2 * Math.cos(state.th2);
}

function energyPumpAcceleration(state, target, params) {
  const currentE = totalEnergy(state, params);
  const targetE = energyAtTarget(target, params);
  const energyError = targetE - currentE;

  const phasePower = supportPhasePower(state, params);

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

function balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, base = 0.08) {
  if (target.id === 0) return 1.0;
  const half = Math.max(0.01, params.segmentHalfLength);
  const edgeRatio = Math.abs(state.x) / half;
  const outwardSpeed = state.x * state.vx > 0 ? Math.abs(state.vx) / Math.max(0.5, params.maxAcc * 0.18) : 0;
  const roughEnvironment = target.id === 3 && ((params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15);

  // When close to a target equilibrium, spend almost all authority on angular
  // balance unless the slider is actually approaching the rail boundary.  In
  // low-damping or windy settings the same rule is made more conservative:
  // don't chase x=0 while the links are still being captured.
  const capture = clamp(1 - (angleNorm / (roughEnvironment ? 0.66 : 0.62) + speedNorm / (roughEnvironment ? 5.6 : 5.4)) * 0.5, 0, 1);
  if (!roughEnvironment) {
    const edgeNeed = clamp((edgeRatio - 0.62) / 0.22, 0, 1);
    const velocityNeed = clamp(outwardSpeed, 0, 1);
    const normalCenter = base * (1 - 0.82 * capture);
    const edgeCenter = (0.16 + 0.56 * Math.max(edgeNeed, velocityNeed)) * Math.max(edgeNeed, velocityNeed);
    return clamp(Math.max(normalCenter, edgeCenter), 0, 0.72);
  }

  const quiet = clamp(1 - (angleNorm / 0.16 + speedNorm / 0.75) * 0.5, 0, 1);
  const edgeStart = 0.74;
  const edgeNeed = clamp((edgeRatio - edgeStart) / Math.max(0.05, 0.92 - edgeStart), 0, 1);
  const velocityNeed = edgeRatio > edgeStart ? clamp(outwardSpeed, 0, 1) : 0;
  const normalCenter = base * ((1 - 0.86 * capture) + 0.45 * quiet);
  const edgeCenter = (0.09 + 0.50 * Math.max(edgeNeed, velocityNeed)) * Math.max(edgeNeed, velocityNeed);
  return clamp(Math.max(normalCenter, edgeCenter), 0, 0.60);
}

function downTerminalBrakeAcceleration(state, params) {
  const e1 = angleError(state.th1, 0);
  const e2 = angleError(state.th2, 0);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(state.x) + 0.7 * Math.abs(state.vx);
  const mixedAngle = 0.66 * e1 + 0.34 * e2;
  const mixedSpeed = 0.68 * state.om1 + 0.32 * state.om2;
  const centerA = centerReturnAcceleration(state, params, supportNorm < 0.20 ? 1.20 : 1.00);
  const phaseAbsorber = clamp(4.6 * supportPhasePower(state, params), -0.14 * params.maxAcc, 0.14 * params.maxAcc);
  const raw = 11.0 * mixedAngle + 8.4 * mixedSpeed + 0.32 * centerA + 0.26 * phaseAbsorber;
  const limit = (angleNorm < 0.10 && speedNorm < 0.24) ? 0.18 * params.maxAcc : 0.38 * params.maxAcc;
  return clamp(raw, -limit, limit);
}

function applyTrackSafety(state, raw, params) {
  const half = Math.max(0.01, params.segmentHalfLength);
  const ratio = Math.abs(state.x) / half;
  const outward = Math.sign(state.x || state.vx || 1);
  let safe = raw;

  if (ratio > 0.66) {
    const barrier = Math.pow((ratio - 0.66) / 0.34, 2);
    safe += -outward * params.maxAcc * 1.05 * barrier;
    safe += -2.85 * state.vx * (1 + 2.4 * barrier);
  }

  // Hard guard: near an edge, never allow acceleration that keeps driving outward.
  if (ratio > 0.88 && Math.sign(safe) === outward) {
    safe = Math.min(Math.abs(safe), 0.18 * params.maxAcc) * -outward;
  }

  // If already pushing the stop, brake strongly toward the center.
  if (ratio > 0.96) {
    safe = -outward * Math.max(0.42 * params.maxAcc, Math.abs(safe));
  }

  return clamp(safe, -params.maxAcc, params.maxAcc);
}


function downStabilizationAcceleration(state, params) {
  const e1 = angleError(state.th1, 0);
  const e2 = angleError(state.th2, 0);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(state.x) + 0.7 * Math.abs(state.vx);

  // State 0 is the natural energy minimum. If it is already quiet, the best
  // controller is no controller: do not inject support motion into the joints.
  if (angleNorm < 0.030 && speedNorm < 0.040 && supportNorm < 0.030) return 0;

  const mixedAngle = 0.60 * e1 + 0.40 * e2;
  const mixedSpeed = 0.62 * state.om1 + 0.38 * state.om2;
  const centerA = centerReturnAcceleration(state, params, supportNorm < 0.18 ? 0.85 : 1.10);

  // High-speed down-crossing absorber. The previous build often passed near
  // state 0 with a large angular rate, then spent multiple visible cycles
  // ringing down. This phase-power term applies damping exactly in that
  // zero-crossing band, while leaving far swing-down and targets 1/2/3 alone.
  if (angleNorm < 0.72 && speedNorm < 6.60 && speedNorm > 1.75 && supportNorm < 0.95) {
    const phasePower = supportPhasePower(state, params);
    const gate = clamp((speedNorm - 1.55) / 4.40, 0, 1);
    const phaseA = clamp((7.60 + 4.20 * gate) * phasePower, -(0.42 + 0.22 * gate) * params.maxAcc, (0.42 + 0.22 * gate) * params.maxAcc);
    const lqrA = pseudoLqrAcceleration(state, TARGETS[0], params);
    const centerMix = supportNorm < 0.36 ? 0.08 : 0.16;
    const raw = (0.58 + 0.10 * gate) * phaseA + (0.28 - 0.10 * gate) * lqrA + centerMix * centerA;
    return clamp(raw, -(0.52 + 0.12 * gate) * params.maxAcc, (0.52 + 0.12 * gate) * params.maxAcc);
  }

  // Fast terminal absorber: removes the final small visible oscillation around
  // state 0 instead of waiting for passive friction to finish the job.
  if (angleNorm < 0.42 && speedNorm < 2.20 && supportNorm < 0.85) {
    const terminalA = downTerminalBrakeAcceleration(state, params);
    const lqrA = pseudoLqrAcceleration(state, TARGETS[0], params);
    const raw = angleNorm < 0.15 && speedNorm < 0.42
      ? 0.58 * lqrA + 0.42 * terminalA
      : 0.34 * lqrA + 0.66 * terminalA;
    return clamp(raw, -0.44 * params.maxAcc, 0.44 * params.maxAcc);
  }

  // If the links are essentially settled but the support is still returning,
  // prioritize centering the support with a small authority cap.
  if (angleNorm < 0.045 && speedNorm < 0.070) {
    return clamp(centerA, -0.20 * params.maxAcc, 0.20 * params.maxAcc);
  }

  // Baseline downward controller: close to the previous build, because the
  // down equilibrium is naturally stable and excessive support motion creates
  // a visible limit-cycle.
  const baseline = 8.8 * mixedAngle + 5.2 * mixedSpeed + 1.12 * centerA;

  const currentE = totalEnergy(state, params);
  const targetE = energyAtTarget(TARGETS[0], params);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const energyExcess = Math.max(0, currentE - targetE);
  const energyGate = clamp(energyExcess / (0.042 * energyScale), 0, 1);
  const phasePower = supportPhasePower(state, params);
  const dampingLimit = (0.10 + 0.20 * energyGate) * params.maxAcc;
  const energyDampingA = clamp((5.2 + 3.6 * energyGate) * phasePower, -dampingLimit, dampingLimit);

  // Only add the Lyapunov-style absorber in the small oscillation region. For
  // larger deviations the normal PD/energy planner is safer and avoids cart drift.
  if (angleNorm < 0.56 && speedNorm < 3.00 && supportNorm < 0.90) {
    const lqrA = pseudoLqrAcceleration(state, TARGETS[0], params);
    const finalBand = angleNorm < 0.18 && speedNorm < 0.52;
    const raw = finalBand
      ? 0.62 * lqrA + 0.22 * energyDampingA + 0.16 * centerA
      : 0.54 * lqrA + 0.28 * baseline + 0.18 * energyDampingA;
    return clamp(raw, -0.50 * params.maxAcc, 0.50 * params.maxAcc);
  }

  return clamp(baseline, -0.66 * params.maxAcc, 0.66 * params.maxAcc);
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

function target3SingleLinkBridgeAcceleration(state, params) {
  const e1 = angleError(state.th1, Math.PI);
  const e2 = angleError(state.th2, Math.PI);
  const a1 = Math.abs(e1);
  const a2 = Math.abs(e2);
  let focus = 0;
  const speedNorm = Math.hypot(state.om1, state.om2);
  // Only assist the hard state-2→state-3 basin: upper link already near
  // upright, lower link still close to down, and the system is not moving fast.
  // The mirror case state-1→state-3 is already handled well by the base CEM/LQR
  // handoff, so injecting this bridge there tends to over-pump it.
  if (a1 < 0.58 && a2 > 1.75 && speedNorm < 3.4) focus = 2;
  else return null;
  const err = focus === 1 ? e1 : e2;
  const theta = focus === 1 ? state.th1 : state.th2;
  const omega = focus === 1 ? state.om1 : state.om2;
  const phasePower = omega * Math.cos(theta);
  const seed = -Math.sign(err || 1) * params.maxAcc;
  const pump = Math.abs(phasePower) < 0.18 ? 0.75 * seed : -Math.sign(phasePower) * params.maxAcc;
  const centerA = centerReturnAcceleration(state, params, 0.72);
  return clamp(0.84 * pump + 0.16 * centerA, -params.maxAcc, params.maxAcc);
}

function targetAlignmentAcceleration(state, target, params) {
  if (target.id === 0) return downStabilizationAcceleration(state, params);

  const [e1, e2] = targetError(state, target);
  const supportBiasRaw = centerReturnAcceleration(state, params, 0.64);

  // During the capture phase use the real LQR direction whenever the state is
  // moderately close; otherwise use a coarse target-seeking PD term to seed MPC.
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const captureAngleLimit = target.id === 2 ? 1.20 : 0.85;
  const captureSpeedLimit = target.id === 2 ? 7.5 : 5.2;
  if (angleNorm < captureAngleLimit && speedNorm < captureSpeedLimit) {
    const pseudo = pseudoLqrAcceleration(state, target, params);
    const centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, 0.09);
    return clamp((1 - centerMix) * pseudo + centerMix * supportBiasRaw, -params.maxAcc, params.maxAcc);
  }

  const w1 = target.id === 1 ? 0.44 : 0.58;
  const w2 = target.id === 2 ? 0.44 : 0.58;
  const mixedAngle = w1 * e1 + w2 * e2;
  const mixedSpeed = 0.42 * state.om1 + 0.38 * state.om2;
  const centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, 0.18);
  const raw = -10.5 * mixedAngle - 3.4 * mixedSpeed + centerMix * supportBiasRaw;
  return clamp(raw, -params.maxAcc, params.maxAcc);
}

function makeScoreContext(target, params) {
  return {
    half: Math.max(0.01, params.segmentHalfLength),
    targetEnergy: energyAtTarget(target, params),
    energyScale: Math.max(1, params.g * (params.m1 + params.m2) * params.l1 + params.g * params.m2 * params.l2)
  };
}

function scoreState(state, target, params, terminal = false, ctx = null) {
  const [e1, e2] = targetError(state, target);
  const angleCost = e1 * e1 + e2 * e2;
  const speedCost = state.om1 * state.om1 + state.om2 * state.om2;
  const half = ctx ? ctx.half : Math.max(0.01, params.segmentHalfLength);
  const centerCost = (state.x / half) * (state.x / half) + 0.34 * state.vx * state.vx;
  const terminalAngleWeight = target.id === 3 ? 17.5 : 16.0;
  const terminalSpeedWeight = target.id === 3 ? 2.4 : 2.2;
  const nearBalance = angleCost < 0.55 && speedCost < 16.0;
  const roughEnvironment = target.id === 3 && ((params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15);
  const centerWeight = target.id === 3 ? (roughEnvironment ? (nearBalance ? 1.05 : 2.85) : (nearBalance ? 1.8 : 3.2)) : 6.8;
  const edgeWeight = target.id === 3 ? (roughEnvironment ? 6.2 : 5.8) : 3.5;

  const energyScale = ctx ? ctx.energyScale : Math.max(1, params.g * (params.m1 + params.m2) * params.l1 + params.g * params.m2 * params.l2);
  const targetEnergy = ctx ? ctx.targetEnergy : energyAtTarget(target, params);
  const dE = (totalEnergy(state, params) - targetEnergy) / energyScale;
  const energyCost = dE * dE;

  const edgeRatio = Math.abs(state.x) / half;
  const outwardSpeed = state.x * state.vx > 0 ? Math.abs(state.vx) : 0;
  const softEdge = edgeRatio > 0.62 ? Math.pow((edgeRatio - 0.62) / 0.38, 4) : 0;
  const hardEdge = edgeRatio > 0.84 ? Math.pow((edgeRatio - 0.84) / 0.16, 8) : 0;
  const edgeCost = 32 * softEdge + 260 * hardEdge + 3.2 * outwardSpeed * outwardSpeed * Math.max(0, edgeRatio - 0.52);

  if (terminal) {
    // Target 0 uses a different MPC/CEM score from the unstable targets: energy
    // and angular-rate removal should dominate the rollout, otherwise the
    // optimizer keeps choosing gentle center-preserving plans that look safe
    // locally but take many extra cycles to become almost still.
    if (target.id === 0) {
      const fastDown = angleCost < 1.10 ? 1.0 : 0.0;
      return 18.50 * angleCost + (2.70 + 0.90 * fastDown) * speedCost + 4.00 * centerCost + 1.45 * energyCost + 3.70 * edgeCost;
    }
    return terminalAngleWeight * angleCost + terminalSpeedWeight * speedCost + centerWeight * centerCost + 0.75 * energyCost + edgeWeight * edgeCost;
  }
  if (target.id === 3) {
    const centerRunWeight = roughEnvironment ? (angleCost < 0.55 && speedCost < 16.0 ? 0.09 : 0.34) : (angleCost < 0.55 && speedCost < 16.0 ? 0.18 : 0.42);
    const angleWeight = roughEnvironment ? 0.52 : 0.46;
    const speedWeight = roughEnvironment ? 0.086 : 0.078;
    const energyWeight = roughEnvironment ? 0.18 : 0.20;
    const edgeRunWeight = roughEnvironment ? 1.42 : 1.36;
    return angleWeight * angleCost + speedWeight * speedCost + centerRunWeight * centerCost + energyWeight * energyCost + edgeRunWeight * edgeCost;
  }
  if (target.id === 0) {
    // Running cost for the swing-down phase: keep rail safety, but bias the
    // search toward energy decay and lower angular velocity rather than
    // over-centering the support during early capture.
    const nearDown = angleCost < 0.75;
    return 0.42 * angleCost + (nearDown ? 0.130 : 0.080) * speedCost + 0.46 * centerCost + 0.50 * energyCost + 1.34 * edgeCost;
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
    const seeds = [0x9e3779b9, 0x00000001, 0xdeadbeef, 0x27182818];
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

function runPrediction(state, target, params, sequence, t0, ctx = null) {
  let s = cloneState(state);
  let cost = 0;
  const dt = 0.036;
  const half = ctx ? ctx.half : Math.max(0.01, params.segmentHalfLength);
  const invA2 = 1 / Math.max(1, params.maxAcc * params.maxAcc);
  for (let i = 0; i < sequence.length; i++) {
    const safeA = applyTrackSafety(s, sequence[i], params);
    s = stepRK4(s, safeA, params, t0 + i * dt, dt, false);
    cost += scoreState(s, target, params, false, ctx);
    cost += 0.0020 * safeA * safeA * invA2;
    if (s.x * safeA > 0) cost += 0.0025 * Math.abs(s.x / half);
  }
  cost += scoreState(s, target, params, true, ctx);
  return cost;
}

function runPredictionBlocks(state, target, params, blocks, horizon, blockLen, t0, ctx) {
  let s = cloneState(state);
  let cost = 0;
  const dt = 0.036;
  const half = ctx.half;
  const invA2 = 1 / Math.max(1, params.maxAcc * params.maxAcc);
  let stepIndex = 0;

  // Same rollout semantics as the previous Math.floor(i / blockLen) loop, but
  // avoids repeated block-index calculations inside the CEM hot path.
  for (let blockIndex = 0; blockIndex < blocks.length && stepIndex < horizon; blockIndex++) {
    const rawBlockA = blocks[blockIndex];
    const blockEnd = Math.min(horizon, stepIndex + blockLen);
    for (; stepIndex < blockEnd; stepIndex++) {
      const safeA = applyTrackSafety(s, rawBlockA, params);
      s = stepRK4(s, safeA, params, t0 + stepIndex * dt, dt, false);
      cost += scoreState(s, target, params, false, ctx);
      cost += 0.0020 * safeA * safeA * invA2;
      if (s.x * safeA > 0) cost += 0.0025 * Math.abs(s.x / half);
    }
  }

  cost += scoreState(s, target, params, true, ctx);
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

function makeHeuristicBlocks(index, blockCount, A, energyA, alignA, centerA, bridgeA, target) {
  const blocks = new Array(blockCount);
  const bias = clamp(0.46 * energyA + 0.23 * alignA + 0.31 * centerA, -A, A);

  if (index === 0) return blocks.fill(energyA);
  if (index === 1) return blocks.fill(alignA);
  if (index === 2) return blocks.fill(centerA);
  if (index === 3) return blocks.fill(bias);
  if (index === 4 && bridgeA !== null) return blocks.fill(bridgeA);
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
    else if (target.id === 3 && index === 12) blocks[i] = i < blockCount / 3 ? A : (i < 2 * blockCount / 3 ? -0.90 * A : 0.55 * centerA);
    else if (target.id === 3 && index === 13) blocks[i] = i < blockCount / 3 ? -A : (i < 2 * blockCount / 3 ? 0.90 * A : 0.55 * centerA);
    else if (target.id === 3 && index === 14) blocks[i] = clamp(0.70 * alignA + 0.30 * energyA, -A, A);
    else if (target.id === 3 && index === 15) blocks[i] = clamp(0.62 * energyA + 0.24 * alignA + 0.14 * A * Math.sin(Math.PI * phase), -A, A);
    else blocks[i] = clamp(bias + randomBetween(-0.65 * A, 0.65 * A), -A, A);
  }
  return blocks;
}

function chooseCemAcceleration(state, target, params, t, previousPlan) {
  const A = params.maxAcc;
  const downRoughEnvironment = target.id === 0 && ((params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.12);
  const downHighAuthority = target.id === 0 && params.maxAcc > 20 && !downRoughEnvironment;
  const horizon = target.id === 0 ? (downHighAuthority ? 46 : 50) : 42;
  const blockLen = 3;
  const blockCount = Math.ceil(horizon / blockLen);
  const sampleCount = target.id === 0 ? (downHighAuthority ? 50 : 54) : 54;
  const eliteCount = target.id === 0 ? 8 : 8;
  const generations = 2;
  const energyA = energyPumpAcceleration(state, target, params);
  const alignA = targetAlignmentAcceleration(state, target, params);
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.18 : 1.0);
  const bridgeA = target.id === 3 ? target3SingleLinkBridgeAcceleration(state, params) : null;
  const scoreCtx = makeScoreContext(target, params);

  let mean = sequenceToBlocks(previousPlan, blockCount, blockLen, clamp(bridgeA !== null ? (0.40 * bridgeA + 0.32 * energyA + 0.16 * alignA + 0.12 * centerA) : (0.46 * energyA + 0.23 * alignA + 0.31 * centerA), -A, A));
  let std = Array(blockCount).fill(Math.max(0.18 * A, 1.5));

  let bestBlocks = mean.slice();
  let bestCost = Infinity;

  for (let gen = 0; gen < generations; gen++) {
    const candidates = [];

    for (let i = 0; i < sampleCount; i++) {
      let blocks;
      if (gen === 0 && i < (target.id === 3 ? 16 : 13)) {
        blocks = makeHeuristicBlocks(i, blockCount, A, energyA, alignA, centerA, bridgeA, target);
      } else {
        blocks = mean.map((m, j) => clamp(m + gaussianRandom() * std[j], -A, A));
      }

      const cost = runPredictionBlocks(state, target, params, blocks, horizon, blockLen, t, scoreCtx);
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


function robustLocalAcceleration(state, target, params, lqrA) {
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.20 : 0.85);
  if (target.id !== 3) {
    const [e1, e2] = targetError(state, target);
    const centerMix = balanceFirstCenterBlend(state, target, params, Math.hypot(e1, e2), Math.hypot(state.om1, state.om2), 0.06);
    return clamp((1 - centerMix) * lqrA + centerMix * centerA, -params.maxAcc, params.maxAcc);
  }

  const alignA = targetAlignmentAcceleration(state, target, params);
  const [e1, e2] = targetError(state, target);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(state.x) + 0.7 * Math.abs(state.vx);
  const dampingA = clamp(
    -0.75 * Math.sin(e1) - 0.95 * Math.sin(e2) - 0.18 * state.om1 - 0.24 * state.om2,
    -0.22 * params.maxAcc,
    0.22 * params.maxAcc
  );
  const roughEnvironment = (params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15;
  let centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, roughEnvironment ? 0.040 : 0.045);
  if (roughEnvironment) {
    const half = Math.max(0.01, params.segmentHalfLength);
    const edgeRatio = Math.abs(state.x) / half;
    const outward = state.x * state.vx > 0;
    if (!(angleNorm < 0.10 && speedNorm < 0.42) && !(edgeRatio > 0.86 && outward)) {
      centerMix = Math.min(centerMix, 0.16);
    }
    // Once the links are truly quiet, finish the job by recentering the support.
    // This remains gated by a tight angular/rate band so the recentering term
    // cannot steal authority during capture.
    if (angleNorm < 0.045 && speedNorm < 0.16) centerMix = Math.max(centerMix, 0.24);
    if (angleNorm < 0.025 && speedNorm < 0.08) centerMix = Math.max(centerMix, 0.34);
  }
  const lqrMix = roughEnvironment ? clamp(0.965 - centerMix, 0.48, 0.98) : clamp(0.94 - centerMix, 0.40, 0.96);
  const alignMix = angleNorm < 0.36 && speedNorm < 2.5 ? (roughEnvironment ? 0.018 : 0.025) : (roughEnvironment ? 0.045 : 0.055);
  return clamp(lqrMix * lqrA + alignMix * alignA + centerMix * centerA + (roughEnvironment ? 0.014 : 0.01) * dampingA, -params.maxAcc, params.maxAcc);
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
    this.localCaptureActive = false;
  }

  setTarget(id, stateHint = null) {
    this.target = TARGETS[id] || TARGETS[0];
    this.controlAccumulator = 0;
    this.plan = [];
    this.localCaptureActive = false;
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
    const preNear = closenessToTarget(state, this.target);
    const updatePeriod = (this.target.id === 0 && preNear.angleNorm < 0.58 && preNear.speedNorm < 2.20) ? 0.026 :
      (this.target.id === 3 && preNear.angleNorm < 0.42 && preNear.speedNorm < 3.2) ? 0.022 :
      (this.target.id === 2 && preNear.angleNorm < 0.82 && preNear.speedNorm < 6.2) ? 0.026 : 0.045;
    const shouldUpdate = this.controlAccumulator >= updatePeriod;
    if (!shouldUpdate) return this.commandAcc;
    this.controlAccumulator = 0;

    let raw;
    if (this.target.id === 0) {
      const nearDown = preNear;
      const supportNorm = Math.abs(state.x) + 0.7 * Math.abs(state.vx);
      const quietEnough = nearDown.angleNorm < 0.030 && nearDown.speedNorm < 0.040 && supportNorm < 0.030;
      if (quietEnough) {
        this.commandAcc = 0;
        this.plan = [];
        return 0;
      } else if (nearDown.angleNorm < 1.12 && nearDown.speedNorm < 5.30) {
        raw = downStabilizationAcceleration(state, params);
        this.plan = [];
      } else {
        const planned = chooseCemAcceleration(state, this.target, params, t, this.plan);
        raw = 0.78 * planned.acceleration + 0.22 * downStabilizationAcceleration(state, params);
        this.plan = planned.plan;
      }
    } else {
      const near = preNear;
      if (near.angleNorm < (this.target.id === 3 ? 0.26 : 0.14) && near.speedNorm < (this.target.id === 3 ? 2.60 : 0.70)) this.localCaptureActive = true;
      if (near.angleNorm > 2.45 || near.speedNorm > 15.0) this.localCaptureActive = false;
      const lqrAngleLimit = this.target.id === 3 && this.localCaptureActive ? 2.10 : (this.target.id === 3 ? 0.64 : (this.target.id === 2 ? 0.92 : 0.55));
      const lqrSpeedLimit = this.target.id === 3 && this.localCaptureActive ? 13.0 : (this.target.id === 3 ? 5.6 : (this.target.id === 2 ? 7.2 : 4.2));
      const useLqr = near.angleNorm < lqrAngleLimit && near.speedNorm < lqrSpeedLimit;
      if (useLqr) {
        raw = robustLocalAcceleration(state, this.target, params, this.lqrAcceleration(state, params));
        this.plan = [];
      } else {
        const planned = chooseCemAcceleration(state, this.target, params, t, this.plan);
        raw = planned.acceleration;
        this.plan = planned.plan;
      }
    }

    let terminalEdgeCapture = false;
    if (this.target.id === 3 && params.maxAcc > 28 && (params.friction || 0) < 0.025) {
      const half = Math.max(0.01, params.segmentHalfLength);
      const edgeRatio = Math.abs(state.x) / half;
      const outward = state.x * state.vx > 0;
      terminalEdgeCapture = outward && preNear.angleNorm < 0.32 && preNear.speedNorm < 2.7 && edgeRatio > 0.64;
      if (terminalEdgeCapture) {
        const centerA = centerReturnAcceleration(state, params, 2.05);
        const edgeNeed = clamp((edgeRatio - 0.64) / 0.22, 0, 1);
        const quiet = clamp(1 - (preNear.angleNorm / 0.32 + preNear.speedNorm / 2.7) * 0.5, 0, 1);
        const centerMix = clamp(0.70 + 0.18 * edgeNeed + 0.10 * quiet, 0.70, 0.94);
        const cap = Math.max(6.0, Math.min(params.maxAcc, 10.0 + 4.0 * preNear.angleNorm + 1.4 * preNear.speedNorm));
        raw = clamp((1 - centerMix) * raw + centerMix * centerA, -cap, cap);
      }
    }

    raw = applyTrackSafety(state, raw, params);

    if (terminalEdgeCapture) {
      this.commandAcc = clamp(raw, -params.maxAcc, params.maxAcc);
      if (Math.abs(this.commandAcc) < 1e-5) this.commandAcc = 0;
      return this.commandAcc;
    }

    // Slew-limited actuator smoothing keeps motion physical while still obeying acceleration-only control.
    const maxDelta = Math.max(6.0, params.maxAcc * 1.20);
    const blended = 0.96 * raw + 0.04 * this.commandAcc;
    this.commandAcc = clamp(blended, this.commandAcc - maxDelta, this.commandAcc + maxDelta);
    this.commandAcc = applyTrackSafety(state, clamp(this.commandAcc, -params.maxAcc, params.maxAcc), params);
    if (Math.abs(this.commandAcc) < 1e-5) this.commandAcc = 0;
    return this.commandAcc;
  }
}
