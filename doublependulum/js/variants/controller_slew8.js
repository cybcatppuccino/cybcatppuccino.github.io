import { wasmRollout } from "../wasm_rollout.js";
import { adaptiveAI } from "../ai_learning.js";
import {
  TARGETS,
  angleError,
  clamp,
  copyStateInto,
  energyAtTarget,
  stepRK4,
  stepRK4Into,
  supportBounds,
  supportCenter,
  supportCenterError,
  supportEdgeRatio,
  supportHalfSpan,
  supportOutwardSign,
  targetError,
  totalEnergy,
  windAcceleration
} from "../physics.js";

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
      x: supportCenter(params) + z[4],
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
      next.x - supportCenter(params),
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
  const state3LowAuthorityWindy = target.id === 3 && (params.maxAcc || 0) < 19 && (params.windAmp || 0) > 0.075 && (params.friction || 0) >= 0.035;
  const q1 = target.id === 0 ? 42 : (target.id === 3 ? (state3LowAuthorityWindy ? 155 : 178) : (target.id === 2 ? 144 : 135));
  const q2 = target.id === 0 ? 42 : (target.id === 3 ? (state3LowAuthorityWindy ? 165 : 188) : (target.id === 2 ? 150 : 145));
  const qOm1 = target.id === 0 ? 7 : (target.id === 3 ? (state3LowAuthorityWindy ? 25 : 38) : (target.id === 2 ? 22 : 20));
  const qOm2 = target.id === 0 ? 7 : (target.id === 3 ? (state3LowAuthorityWindy ? 27 : 42) : (target.id === 2 ? 23 : 21));

  // The finite rail still matters, but near the unstable target the local
  // controller must first keep the links balanced. A large cart-centering term
  // was the main cause of the old target-3 overshoot: when the links were
  // already almost upright, LQR would spend too much authority trying to move
  // x back to zero and the links fell out of the capture basin.
  const roughEnvironment = (params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15;
  const qCenter = target.id === 0 ? 8.0 : (target.id === 3 ? (roughEnvironment ? 1.15 : 1.75) : 2.25);
  const qCartVelocity = target.id === 0 ? 4.6 : (target.id === 3 ? (roughEnvironment ? 0.82 : 1.20) : 1.65);
  const Q = diag([q1, q2, qOm1, qOm2, qCenter, qCartVelocity]);
  const R = target.id === 0 ? 0.08 : (target.id === 2 ? 0.046 : (target.id === 3 ? (state3LowAuthorityWindy ? 0.090 : 0.076) : 0.13));
  return discreteLqr(Ad, Bd, Q, R);
}

function lqrFeedback(gain, state, target, params) {
  return gain[0] * angleError(state.th1, target.angles[0]) +
    gain[1] * angleError(state.th2, target.angles[1]) +
    gain[2] * state.om1 +
    gain[3] * state.om2 +
    gain[4] * supportCenterError(state, params) +
    gain[5] * state.vx;
}

function closenessToTarget(state, target) {
  const e1 = angleError(state.th1, target.angles[0]);
  const e2 = angleError(state.th2, target.angles[1]);
  return {
    angleNorm: Math.hypot(e1, e2),
    speedNorm: Math.hypot(state.om1, state.om2)
  };
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
  // energy quickly for state 0, whose target energy is the minimum.  With high
  // support authority, scale the pump by energy error so the controller does not
  // continue injecting large kinetic energy after reaching the right energy band.
  const duty = target.id === 0 ? 0.86 : 1.0;
  if (params.maxAcc <= 20) return -duty * params.maxAcc * Math.sign(energyError * phasePower);
  const near = closenessToTarget(state, target);
  const cap = planningAuthority(state, target, params, near.angleNorm, near.speedNorm);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const eGate = clamp(Math.abs(energyError) / (0.55 * energyScale), 0.20, 1.0);
  const farBoost = target.id === 3 ? clamp((near.angleNorm - 1.45) / 1.35, 0, 0.34) : (near.angleNorm > 0.88 || near.speedNorm > 6.5 ? 0.20 : 0);
  return -duty * Math.min(params.maxAcc, cap * (1 + farBoost)) * (0.40 + 0.60 * eGate) * Math.sign(energyError * phasePower);
}


function centerReturnAcceleration(state, params, gainScale = 1.0) {
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const ratio = Math.abs(xErr) / half;

  // PD return-to-center term; gains rise close to the rail edges.
  const edgeBoost = ratio > 0.55 ? 1 + 3.2 * Math.pow((ratio - 0.55) / 0.45, 2) : 1;
  const raw = (-1.18 * xErr - 1.55 * state.vx) * gainScale * edgeBoost;
  return clamp(raw, -0.72 * params.maxAcc, 0.72 * params.maxAcc);
}

function sameSideBoundaryRisk(state, candidateA, target, params, near = null, leadOverride = null) {
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const edgeRatio = Math.abs(xErr) / half;
  const side = supportOutwardSign(state, params);
  const lead = leadOverride ?? (target.id === 3 ? 0.50 : 0.42);
  const predictedErr = xErr + state.vx * lead + 0.5 * candidateA * lead * lead;
  const predictedRatio = Math.abs(predictedErr) / half;
  const predictedSide = Math.sign(predictedErr) || side;
  const commandOutward = side * candidateA > Math.max(0.45, 0.035 * params.maxAcc);
  const velocityOutward = side * state.vx > 0.035;
  const edgeStart = target.id === 3 ? 0.70 : 0.62;
  const predictedStart = target.id === 3 ? 0.84 : 0.78;
  const severeStart = target.id === 3 ? 0.93 : 0.89;
  const sameSide = predictedSide === side;
  const commandConflict = sameSide && commandOutward && edgeRatio > edgeStart && predictedRatio > predictedStart;
  const inertiaConflict = sameSide && velocityOutward && edgeRatio > (target.id === 3 ? 0.84 : 0.78) && predictedRatio > severeStart;
  const severe = sameSide && predictedRatio > 0.965 && edgeRatio > (target.id === 3 ? 0.88 : 0.82) && (commandOutward || velocityOutward);

  return {
    conflict: commandConflict || inertiaConflict || severe,
    severe,
    side,
    edgeRatio,
    predictedRatio,
    commandOutward,
    velocityOutward
  };
}

function balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, base = 0.08, balanceA = null) {
  if (target.id === 0) return 1.0;
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const edgeRatio = Math.abs(xErr) / half;
  const outwardSpeed = xErr * state.vx > 0 ? Math.abs(state.vx) / Math.max(0.5, params.maxAcc * 0.18) : 0;
  const roughEnvironment = target.id === 3 && ((params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15);
  const risk = balanceA === null ? null : sameSideBoundaryRisk(state, balanceA, target, params, null);

  // Give the local balancer priority.  Centering/braking is mixed in only when
  // the predicted balance command still drives the support into the same-side
  // rail.  If the balance command is inward or otherwise not edge-conflicting,
  // keep the older strong-balance behavior even with outward support velocity.
  const capture = clamp(1 - (angleNorm / (roughEnvironment ? 0.66 : 0.62) + speedNorm / (roughEnvironment ? 5.6 : 5.4)) * 0.5, 0, 1);
  if (!roughEnvironment) {
    const edgeNeed = clamp((edgeRatio - 0.66) / 0.22, 0, 1);
    const velocityNeed = clamp(outwardSpeed, 0, 1);
    const riskNeed = risk === null ? Math.max(edgeNeed, velocityNeed) : (risk.conflict ? Math.max(edgeNeed, velocityNeed, clamp((risk.predictedRatio - 0.82) / 0.16, 0, 1)) : 0);
    const normalCenter = base * (1 - 0.82 * capture);
    const edgeCenter = (0.10 + 0.42 * riskNeed) * riskNeed;
    return clamp(Math.max(normalCenter, edgeCenter), 0, 0.64);
  }

  const quiet = clamp(1 - (angleNorm / 0.16 + speedNorm / 0.75) * 0.5, 0, 1);
  const edgeStart = 0.78;
  const edgeNeed = clamp((edgeRatio - edgeStart) / Math.max(0.05, 0.94 - edgeStart), 0, 1);
  const velocityNeed = edgeRatio > edgeStart ? clamp(outwardSpeed, 0, 1) : 0;
  const riskNeed = risk === null ? Math.max(edgeNeed, velocityNeed) : (risk.conflict ? Math.max(edgeNeed, velocityNeed, clamp((risk.predictedRatio - 0.88) / 0.11, 0, 1)) : 0);
  const normalCenter = base * ((1 - 0.86 * capture) + 0.42 * quiet);
  const edgeCenter = (0.07 + 0.38 * riskNeed) * riskNeed;
  return clamp(Math.max(normalCenter, edgeCenter), 0, 0.54);
}

function downTerminalBrakeAcceleration(state, params) {
  const e1 = angleError(state.th1, 0);
  const e2 = angleError(state.th2, 0);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(supportCenterError(state, params)) + 0.7 * Math.abs(state.vx);
  const oppositeGate = e1 * e2 < -0.010 ? clamp((-e1 * e2) / 0.16, 0, 1) : 0;
  const angleLead = 0.66 + 0.17 * oppositeGate;
  const speedLead = 0.68 + 0.14 * oppositeGate;
  const mixedAngle = angleLead * e1 + (1 - angleLead) * e2;
  const mixedSpeed = speedLead * state.om1 + (1 - speedLead) * state.om2;
  const centerA = centerReturnAcceleration(state, params, supportNorm < 0.20 ? 1.20 : 1.00);
  const phaseAbsorber = clamp(4.6 * supportPhasePower(state, params), -0.14 * params.maxAcc, 0.14 * params.maxAcc);
  const antiPhaseA = oppositeGate * (2.8 * (e1 - e2) + 1.15 * (state.om1 - state.om2));
  const raw = 9.2 * mixedAngle + 6.8 * mixedSpeed + 0.76 * antiPhaseA + 0.16 * centerA + 0.16 * phaseAbsorber;
  const limit = (angleNorm < 0.10 && speedNorm < 0.24) ? 0.18 * params.maxAcc : 0.38 * params.maxAcc;
  return clamp(raw, -limit, limit);
}

function applyTrackSafety(state, raw, params) {
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const ratio = Math.abs(xErr) / half;
  const outward = supportOutwardSign(state, params);
  const movingOutward = xErr * state.vx > 0;
  let safe = raw;

  // Softer rail policy for the enlarged segment: the controller may use the
  // endpoint regions for swing-up/capture, but outward velocity is damped before
  // the physical stop can remove all support authority.
  if (ratio > 0.84) {
    const barrier = Math.pow((ratio - 0.84) / 0.16, 2);
    const outwardCmd = Math.max(0, safe * outward);
    safe -= outward * Math.min(outwardCmd, params.maxAcc * 0.72 * barrier);
    if (movingOutward) {
      safe += -outward * params.maxAcc * 0.38 * barrier;
      safe += -1.38 * state.vx * (1 + 1.75 * barrier);
    } else {
      safe += -0.24 * state.vx * barrier;
    }
  }

  // Very close to an endpoint, prevent commands that keep pushing into the stop;
  // this threshold is intentionally late so endpoints are no longer avoided too early.
  if (ratio > 0.955 && Math.sign(safe) === outward && movingOutward) {
    safe = Math.min(Math.abs(safe), 0.12 * params.maxAcc) * -outward;
  }

  if (ratio > 0.992) {
    safe = -outward * Math.max(0.30 * params.maxAcc, Math.abs(safe));
  }

  return clamp(safe, -params.maxAcc, params.maxAcc);
}

function predictiveLandingGuardAcceleration(state, raw, target, params, near) {
  if (target.id === 0) return raw;

  // Only intervene in the final capture band.  The important distinction is the
  // direction of the balance command: if strong balance wants to move away from
  // the same-side rail, leave it untouched.  Guarding is reserved for cases
  // where the predicted strong-balance command itself still pushes into the
  // boundary, or inertia is severe enough that the inward command cannot save it.
  const highAuthority = params.maxAcc > 20;
  const angleLimit = target.id === 3 ? (highAuthority ? 0.52 : 0.34) : (highAuthority ? 0.30 : 0.23);
  const speedLimit = target.id === 3 ? (highAuthority ? 4.60 : 2.25) : (highAuthority ? 2.70 : 1.70);
  if (near.angleNorm > angleLimit || near.speedNorm > speedLimit) return raw;

  const risk = sameSideBoundaryRisk(state, raw, target, params, near);
  if (!risk.conflict) return raw;
  if (risk.predictedRatio < (target.id === 3 ? 0.90 : 0.84) || risk.edgeRatio < (target.id === 3 ? 0.76 : 0.70)) return raw;

  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.10 : 1.20);
  const edgeGate = clamp((Math.max(risk.edgeRatio, risk.predictedRatio) - (target.id === 3 ? 0.86 : 0.80)) / 0.13, 0, 1);
  const quiet = clamp(1 - (near.angleNorm / angleLimit + near.speedNorm / speedLimit) * 0.5, 0, 1);
  const brakeA = clamp(centerA - risk.side * params.maxAcc * (0.035 + 0.10 * edgeGate) - 0.18 * state.vx, -params.maxAcc, params.maxAcc);
  const mix = clamp(0.06 + 0.24 * edgeGate + 0.05 * quiet, 0.06, target.id === 3 ? 0.34 : 0.42);
  const cap = planningAuthority(state, target, params, near.angleNorm, near.speedNorm);
  return clamp((1 - mix) * raw + mix * brakeA, -cap, cap);
}


function planningAuthority(state, target, params, angleNorm = null, speedNorm = null) {
  const maxA = Math.max(1, params.maxAcc);
  if (maxA <= 20) return maxA;
  if (angleNorm === null || speedNorm === null) {
    const near = closenessToTarget(state, target);
    angleNorm = near.angleNorm;
    speedNorm = near.speedNorm;
  }

  // Treat supp acc as an actuator ceiling, not as the default swing amplitude.
  // Above ~20 m/s², full-bang CEM rollouts often keep the double pendulum in a
  // high-energy orbit.  This sublinear search radius keeps the extra authority
  // available for emergencies while making high maxAcc behave monotonically better.
  let cap = Math.min(maxA,
    (target.id === 3 ? 8.2 : 12.4) +
    (target.id === 3 ? 1.68 : 3.12) * Math.sqrt(maxA) +
    (target.id === 3 ? 0.022 : 0.040) * maxA
  );
  const farGate = clamp((angleNorm - (target.id === 3 ? 0.95 : 1.15)) / (target.id === 3 ? 1.55 : 1.75), 0, 1);
  cap = Math.min(maxA, cap * (1 + (target.id === 3 ? 0.16 : 0.20) * farGate));

  const captureGate = clamp(1 - (angleNorm / (target.id === 3 ? 0.82 : 0.70) + speedNorm / (target.id === 3 ? 5.2 : 4.4)) * 0.5, 0, 1);
  const highGate = clamp((maxA - 20) / 40, 0, 1);
  const captureCap = Math.min(maxA, (target.id === 3 ? 5.0 : 5.5) + (target.id === 3 ? 0.38 : 0.46) * cap + highGate * (target.id === 3 ? 2.2 : 2.0));
  cap = cap * (1 - 0.46 * captureGate) + captureCap * (0.46 * captureGate);

  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const edgeRatio = Math.abs(xErr) / half;
  const outward = xErr * state.vx > 0;
  if (edgeRatio > 0.74 || outward) {
    const edgeGate = Math.max(clamp((edgeRatio - 0.74) / 0.23, 0, 1), outward ? clamp(Math.abs(state.vx) / Math.max(0.75, 0.18 * maxA), 0, 1) : 0);
    cap = cap * (1 - 0.16 * edgeGate) + Math.min(maxA, 8.5 + 0.50 * cap) * (0.16 * edgeGate);
  }
  return clamp(cap, Math.min(maxA, 5.5), maxA);
}

function downStabilizationAcceleration(state, params) {
  const e1 = angleError(state.th1, 0);
  const e2 = angleError(state.th2, 0);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(supportCenterError(state, params)) + 0.7 * Math.abs(state.vx);

  // State 0 is the natural energy minimum. If it is already quiet, the best
  // controller is no controller: do not inject support motion into the joints.
  if (angleNorm < 0.030 && speedNorm < 0.040 && supportNorm < 0.030) return 0;

  const oppositeGate = e1 * e2 < -0.010 ? clamp((-e1 * e2) / 0.16, 0, 1) : 0;
  const angleLead = 0.60 + 0.20 * oppositeGate;
  const speedLead = 0.62 + 0.16 * oppositeGate;
  const mixedAngle = angleLead * e1 + (1 - angleLead) * e2;
  const mixedSpeed = speedLead * state.om1 + (1 - speedLead) * state.om2;
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
      ? 0.70 * lqrA + 0.30 * terminalA
      : 0.46 * lqrA + 0.54 * terminalA;
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
  const antiPhaseA = oppositeGate * (3.4 * (e1 - e2) + 1.35 * (state.om1 - state.om2));
  const baseline = 8.8 * mixedAngle + 5.2 * mixedSpeed + antiPhaseA + 1.12 * centerA;

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
  return clamp(-lqrFeedback(heuristicLqrCache.gain, state, target, params), -params.maxAcc, params.maxAcc);
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
    const centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, 0.09, pseudo);
    return clamp((1 - centerMix) * pseudo + centerMix * supportBiasRaw, -params.maxAcc, params.maxAcc);
  }

  const w1 = target.id === 1 ? 0.44 : 0.58;
  const w2 = target.id === 2 ? 0.44 : 0.58;
  const mixedAngle = w1 * e1 + w2 * e2;
  const mixedSpeed = 0.42 * state.om1 + 0.38 * state.om2;
  const balanceOnlyA = -10.5 * mixedAngle - 3.4 * mixedSpeed;
  const centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, 0.18, balanceOnlyA);
  const raw = balanceOnlyA + centerMix * supportBiasRaw;
  return clamp(raw, -params.maxAcc, params.maxAcc);
}

function predictedStabilizationDirection(state, target, params, alignA = null) {
  if (target.id === 0) return 0;
  const [e1, e2] = targetError(state, target);
  const angularNeed = -0.52 * Math.sin(e1) - 0.58 * Math.sin(e2) - 0.16 * state.om1 - 0.18 * state.om2;
  const balanceNeed = alignA === null ? targetAlignmentAcceleration(state, target, params) : alignA;
  const combined = balanceNeed + (target.id === 3 ? 2.8 : 2.1) * angularNeed;
  if (Math.abs(combined) > 0.20) return Math.sign(combined);
  const phasePower = supportPhasePower(state, params);
  return Math.abs(phasePower) > 1e-4 ? -Math.sign(phasePower) : 0;
}

function reservePrepositionAcceleration(state, target, params, near, alignA = null) {
  if (target.id === 0) return centerReturnAcceleration(state, params, 1.0);

  const dir = predictedStabilizationDirection(state, target, params, alignA);
  if (dir === 0) return centerReturnAcceleration(state, params, target.id === 3 ? 1.04 : 0.94);

  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const edgeRatio = Math.abs(xErr) / half;
  const finalBand = target.id === 3 ? 0.46 : 0.30;
  const approachGate = clamp((near.angleNorm - finalBand) / (target.id === 3 ? 1.55 : 1.15), 0, 1);
  const speedGate = clamp((near.speedNorm - (target.id === 3 ? 1.0 : 0.7)) / (target.id === 3 ? 6.5 : 4.6), 0, 1);
  const gate = Math.max(0.25 * speedGate, approachGate);
  const reserveFrac = target.id === 3 ? 0.33 : (target.id === 2 ? 0.26 : 0.23);
  const desiredErr = clamp(-dir * reserveFrac * half, -0.46 * half, 0.46 * half);
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.08 : 1.00);
  const raw = -1.05 * (xErr - desiredErr) - 1.28 * state.vx;
  const cap = Math.min(params.maxAcc, Math.max(4.0, 0.42 * planningAuthority(state, target, params, near.angleNorm, near.speedNorm)));
  let prepositionA = clamp(raw, -cap, cap);

  if (edgeRatio > 0.72 || xErr * state.vx > 0) {
    const edgeGate = Math.max(clamp((edgeRatio - 0.72) / 0.22, 0, 1), xErr * state.vx > 0 ? clamp(Math.abs(state.vx) / Math.max(0.7, 0.16 * params.maxAcc), 0, 1) : 0);
    prepositionA = (1 - 0.70 * edgeGate) * prepositionA + 0.70 * edgeGate * centerA;
  }

  return clamp((1 - gate) * centerA + gate * prepositionA, -cap, cap);
}

function makeScoreContext(target, params) {
  return {
    half: supportHalfSpan(params),
    center: supportCenter(params),
    targetEnergy: energyAtTarget(target, params),
    energyScale: Math.max(1, params.g * (params.m1 + params.m2) * params.l1 + params.g * params.m2 * params.l2)
  };
}

function scoreState(state, target, params, terminal = false, ctx = null) {
  const e1 = angleError(state.th1, target.angles[0]);
  const e2 = angleError(state.th2, target.angles[1]);
  const angleCost = e1 * e1 + e2 * e2;
  const speedCost = state.om1 * state.om1 + state.om2 * state.om2;
  const half = ctx ? ctx.half : supportHalfSpan(params);
  const center = ctx ? ctx.center : supportCenter(params);
  const xErr = state.x - center;
  const centerCost = (xErr / half) * (xErr / half) + 0.26 * state.vx * state.vx;
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

  const edgeRatio = Math.abs(xErr) / half;
  const outwardSpeed = xErr * state.vx > 0 ? Math.abs(state.vx) : 0;
  const softEdge = edgeRatio > 0.74 ? Math.pow((edgeRatio - 0.74) / 0.26, 4) : 0;
  const hardEdge = edgeRatio > 0.93 ? Math.pow((edgeRatio - 0.93) / 0.07, 8) : 0;
  const edgeCost = 12 * softEdge + 120 * hardEdge + 1.45 * outwardSpeed * outwardSpeed * Math.max(0, edgeRatio - 0.68);
  const arrivalGate = target.id === 0 ? clamp(1 - Math.sqrt(angleCost) / 0.72, 0, 1) :
    clamp(1 - Math.sqrt(angleCost) / (target.id === 3 ? 0.96 : 0.76), 0, 1);
  const captureGate = target.id === 3 ? arrivalGate : 0;
  const positiveDE = Math.max(0, dE);
  // Predictive landing cost: once a rollout reaches the target neighborhood,
  // prefer trajectories that arrive with manageable angular velocity and the
  // right energy, instead of blasting through the capture basin and relying on
  // later correction.
  const landingCost = arrivalGate * ((target.id === 3 ? 0.55 : 0.24) * speedCost + (target.id === 3 ? 1.05 : 0.62) * energyCost + 0.36 * positiveDE * positiveDE);
  const flybyCost = captureGate * (0.34 * speedCost + 0.18 * positiveDE * positiveDE);

  if (terminal) {
    // Target 0 uses a different MPC/CEM score from the unstable targets: energy
    // and angular-rate removal should dominate the rollout, otherwise the
    // optimizer keeps choosing gentle center-preserving plans that look safe
    // locally but take many extra cycles to become almost still.
    if (target.id === 0) {
      const fastDown = angleCost < 1.10 ? 1.0 : 0.0;
      return 18.50 * angleCost + (2.70 + 0.90 * fastDown) * speedCost + 4.00 * centerCost + 1.45 * energyCost + 3.70 * edgeCost + 0.45 * landingCost;
    }
    return terminalAngleWeight * angleCost + (terminalSpeedWeight + 0.65 * captureGate) * speedCost + centerWeight * centerCost + (0.75 + 0.07 * captureGate) * energyCost + edgeWeight * edgeCost + 1.15 * flybyCost + (target.id === 3 ? 3.25 : 1.35) * landingCost;
  }
  if (target.id === 3) {
    const centerRunWeight = roughEnvironment ? (angleCost < 0.55 && speedCost < 16.0 ? 0.09 : 0.34) : (angleCost < 0.55 && speedCost < 16.0 ? 0.18 : 0.42);
    const angleWeight = roughEnvironment ? 0.52 : 0.46;
    const speedWeight = roughEnvironment ? 0.086 : 0.078;
    const energyWeight = roughEnvironment ? 0.18 : 0.20;
    const edgeRunWeight = roughEnvironment ? 1.42 : 1.36;
    return angleWeight * angleCost + (speedWeight + 0.12 * captureGate) * speedCost + centerRunWeight * centerCost + energyWeight * energyCost + edgeRunWeight * edgeCost + 0.48 * flybyCost + 0.70 * landingCost;
  }
  if (target.id === 0) {
    // Running cost for the swing-down phase: keep rail safety, but bias the
    // search toward energy decay and lower angular velocity rather than
    // over-centering the support during early capture.
    const nearDown = angleCost < 0.75;
    return 0.42 * angleCost + (nearDown ? 0.130 : 0.080) * speedCost + 0.46 * centerCost + 0.50 * energyCost + 1.34 * edgeCost + 0.10 * landingCost;
  }
  return 0.34 * angleCost + (0.060 + 0.10 * captureGate) * speedCost + 0.85 * centerCost + 0.20 * energyCost + 1.10 * edgeCost + 0.36 * flybyCost + 0.16 * landingCost;
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

function makeRolloutScratch() {
  return {
    s: {},
    next: {},
    scratch1: {},
    scratch2: {},
    scratch3: {},
    k1: {},
    k2: {},
    k3: {},
    k4: {},
    firstValues: new Array(13),
    followValues: new Array(4)
  };
}

const sequenceRolloutScratch = makeRolloutScratch();
const blockRolloutScratch = makeRolloutScratch();
const captureRolloutScratch = makeRolloutScratch();

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
  const ws = sequenceRolloutScratch;
  let s = copyStateInto(state, ws.s);
  let next = ws.next;
  let cost = 0;
  const dt = 0.036;
  const half = ctx ? ctx.half : supportHalfSpan(params);
  const center = ctx ? ctx.center : supportCenter(params);
  const invA2 = 1 / Math.max(1, params.maxAcc * params.maxAcc);
  for (let i = 0; i < sequence.length; i++) {
    const safeA = applyTrackSafety(s, sequence[i], params);
    stepRK4Into(s, safeA, params, t0 + i * dt, dt, false, next, ws.scratch1, ws.scratch2, ws.scratch3, ws.k1, ws.k2, ws.k3, ws.k4);
    const old = s;
    s = next;
    next = old;
    cost += scoreState(s, target, params, false, ctx);
    cost += 0.0020 * safeA * safeA * invA2;
    const xErr = s.x - center;
    if (xErr * safeA > 0) cost += 0.0012 * Math.abs(xErr / half);
  }
  cost += scoreState(s, target, params, true, ctx);
  return cost;
}

function runPredictionBlocksWasm(state, target, params, blocks, horizon, blockLen) {
  return wasmRollout.evaluateBlocks(state, target, params, blocks, horizon, blockLen);
}

function runPredictionBlocksBatchWasm(state, target, params, candidates, horizon, blockLen) {
  return wasmRollout.evaluateBatch(state, target, params, candidates, horizon, blockLen);
}

function scoreCandidateBlocks(state, target, params, candidates, horizon, blockLen, t, scoreCtx, wasmReady) {
  if (wasmReady) {
    const costs = runPredictionBlocksBatchWasm(state, target, params, candidates, horizon, blockLen);
    if (costs) {
      for (let i = 0; i < candidates.length; i++) candidates[i].cost = costs[i];
      return;
    }
  }
  for (const candidate of candidates) {
    candidate.cost = runPredictionBlocks(state, target, params, candidate.blocks, horizon, blockLen, t, scoreCtx);
  }
}

function runPredictionBlocks(state, target, params, blocks, horizon, blockLen, t0, ctx) {
  const ws = blockRolloutScratch;
  let s = copyStateInto(state, ws.s);
  let next = ws.next;
  let cost = 0;
  const dt = 0.036;
  const half = ctx.half;
  const center = ctx.center;
  const invA2 = 1 / Math.max(1, params.maxAcc * params.maxAcc);
  let stepIndex = 0;

  // Same rollout semantics as the previous Math.floor(i / blockLen) loop, but
  // avoids repeated block-index calculations inside the CEM hot path.
  for (let blockIndex = 0; blockIndex < blocks.length && stepIndex < horizon; blockIndex++) {
    const rawBlockA = blocks[blockIndex];
    const blockEnd = Math.min(horizon, stepIndex + blockLen);
    for (; stepIndex < blockEnd; stepIndex++) {
      const safeA = applyTrackSafety(s, rawBlockA, params);
      stepRK4Into(s, safeA, params, t0 + stepIndex * dt, dt, false, next, ws.scratch1, ws.scratch2, ws.scratch3, ws.k1, ws.k2, ws.k3, ws.k4);
      const old = s;
      s = next;
      next = old;
      cost += scoreState(s, target, params, false, ctx);
      cost += 0.0020 * safeA * safeA * invA2;
      const xErr = s.x - center;
      if (xErr * safeA > 0) cost += 0.0012 * Math.abs(xErr / half);
    }
  }

  cost += scoreState(s, target, params, true, ctx);
  return cost;
}

function localLandingCap(state, target, params, angleNorm, speedNorm) {
  const maxA = Math.max(1, params.maxAcc);
  if (maxA <= 20) return maxA;
  const highGate = clamp((maxA - 20) / 40, 0, 1);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const dE = Math.abs(totalEnergy(state, params) - energyAtTarget(target, params)) / energyScale;

  if (target.id === 0) {
    const closeGate = clamp(1 - angleNorm / 0.95, 0, 1);
    return Math.min(maxA, 8.5 + 0.24 * maxA + 2.8 * speedNorm + 6.0 * dE + 5.0 * highGate * closeGate);
  }
  if (target.id === 3) {
    const closeGate = clamp(1 - angleNorm / 0.85, 0, 1);
    return Math.min(maxA, 9.0 + 0.30 * maxA + 3.0 * speedNorm + 8.0 * dE + 7.0 * highGate * closeGate);
  }
  const closeGate = clamp(1 - angleNorm / 0.65, 0, 1);
  return Math.min(maxA, 7.0 + 0.24 * maxA + 2.0 * speedNorm + 5.0 * dE + 4.0 * highGate * closeGate);
}


function captureTerminalCost(state, target, params, ctx = null) {
  const e1 = angleError(state.th1, target.angles[0]);
  const e2 = angleError(state.th2, target.angles[1]);
  const angleCost = e1 * e1 + e2 * e2;
  const speedCost = state.om1 * state.om1 + state.om2 * state.om2;
  const center = ctx ? ctx.center : supportCenter(params);
  const half = ctx ? ctx.half : supportHalfSpan(params);
  const xErr = state.x - center;
  const centerCost = xErr ** 2 + 0.42 * state.vx * state.vx;
  const energyScale = ctx ? ctx.energyScale : Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const targetEnergy = ctx ? ctx.targetEnergy : energyAtTarget(target, params);
  const dE = (totalEnergy(state, params) - targetEnergy) / energyScale;
  const edge = Math.max(0, Math.abs(xErr) / Math.max(0.01, half) - 0.82);
  return 24.0 * angleCost + 2.8 * speedCost + 0.85 * centerCost + 1.35 * dE * dE + 18.0 * edge * edge;
}

function capturePolishAcceleration(state, target, params, t, baseA, near) {
  // Short-horizon capture enumeration for target fly-bys. It remains dormant
  // far away, but once the links are near a target it compares lower-displacement
  // braking, prepositioning and LQR-follow candidates before committing.
  if (target.id === 0 || params.maxAcc <= 20) return baseA;

  const radial = targetRadialVelocity(state, target);
  const edgeStress = supportEdgeRatio(state, params) > (target.id === 3 ? 0.76 : 0.72) && supportCenterError(state, params) * state.vx > 0;
  const danger = (radial > (target.id === 3 ? 0.18 : 0.14) && near.speedNorm > (target.id === 3 ? 1.65 : 1.10)) ||
    near.speedNorm > (target.id === 3 ? 4.6 : 3.4) ||
    edgeStress;
  const angleLimit = target.id === 3 ? 0.72 : (target.id === 2 ? 0.56 : 0.48);
  const speedLimit = target.id === 3 ? 8.2 : (target.id === 2 ? 6.0 : 5.2);
  if (!danger || near.angleNorm > angleLimit || near.speedNorm > speedLimit) return baseA;

  const cap = localLandingCap(state, target, params, near.angleNorm, near.speedNorm);
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.08 : 1.04);
  const alignA = targetAlignmentAcceleration(state, target, params);
  const reserveA = reservePrepositionAcceleration(state, target, params, near, alignA);
  const phaseA = clamp(4.8 * supportPhasePower(state, params), -0.38 * cap, 0.38 * cap);
  const lqrA = pseudoLqrAcceleration(state, target, params);
  const ws = captureRolloutScratch;
  const firstValues = ws.firstValues;
  firstValues[0] = clamp(baseA, -cap, cap);
  firstValues[1] = clamp(0.62 * baseA, -cap, cap);
  firstValues[2] = clamp(1.10 * baseA, -cap, cap);
  firstValues[3] = clamp(alignA, -cap, cap);
  firstValues[4] = clamp(lqrA, -cap, cap);
  firstValues[5] = clamp(0.62 * alignA + 0.38 * centerA, -cap, cap);
  firstValues[6] = clamp(0.62 * alignA + 0.38 * reserveA, -cap, cap);
  firstValues[7] = clamp(0.70 * baseA + 0.30 * phaseA, -cap, cap);
  firstValues[8] = clamp(0.48 * baseA + 0.30 * alignA + 0.22 * reserveA, -cap, cap);
  firstValues[9] = clamp(centerA, -cap, cap);
  firstValues[10] = clamp(reserveA, -cap, cap);
  firstValues[11] = clamp(-0.30 * baseA + 0.72 * alignA, -cap, cap);
  firstValues[12] = clamp(0.50 * phaseA + 0.30 * alignA + 0.20 * centerA, -cap, cap);
  const followValues = ws.followValues;
  followValues[0] = clamp(alignA, -cap, cap);
  followValues[1] = clamp(lqrA, -cap, cap);
  followValues[2] = clamp(0.72 * alignA + 0.28 * centerA, -cap, cap);
  followValues[3] = clamp(0.72 * alignA + 0.28 * reserveA, -cap, cap);

  const dt = 1 / 90;
  const horizon = target.id === 3 ? 16 : 13;
  let bestA = clamp(baseA, -cap, cap);
  let bestCost = Infinity;
  let baseCost = Infinity;
  const terminalCtx = {
    center: supportCenter(params),
    half: supportHalfSpan(params),
    targetEnergy: energyAtTarget(target, params),
    energyScale: Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2))
  };
  const horizonDenom = Math.max(1, horizon - 1);
  const tailStart = horizon - 5;

  let candidateIndex = 0;
  for (let firstIndex = 0; firstIndex < firstValues.length; firstIndex++) {
    const firstA = firstValues[firstIndex];
    for (let settleIndex = 0; settleIndex < followValues.length; settleIndex++) {
      const settleA = followValues[settleIndex];
      let s = copyStateInto(state, ws.s);
      let next = ws.next;
      let runningCost = 0;
      for (let k = 0; k < horizon; k++) {
        const phase = k / horizonDenom;
        const blend = phase < 0.24 ? 0 : clamp((phase - 0.24) / 0.76, 0, 1);
        const followA = clamp((1 - blend) * firstA + blend * settleA, -cap, cap);
        stepRK4Into(s, followA, params, t + k * dt, dt, false, next, ws.scratch1, ws.scratch2, ws.scratch3, ws.k1, ws.k2, ws.k3, ws.k4);
        const old = s;
        s = next;
        next = old;
        if (k >= tailStart) runningCost += 0.24 * captureTerminalCost(s, target, params, terminalCtx);
      }
      const edgePenalty = Math.max(0, Math.abs(s.x - terminalCtx.center) / Math.max(0.01, terminalCtx.half) - 0.78);
      const cost = runningCost + captureTerminalCost(s, target, params, terminalCtx) + 0.0022 * firstA * firstA + 6.0 * edgePenalty * edgePenalty;
      if (candidateIndex === 0) baseCost = cost;
      if (cost < bestCost) {
        bestCost = cost;
        bestA = firstA;
      }
      candidateIndex += 1;
    }
  }

  if (bestCost < baseCost * 0.985 - 0.0010) {
    const highGate = clamp((params.maxAcc - 20) / 40, 0, 1);
    const mix = 0.56 + 0.24 * highGate;
    return clamp((1 - mix) * baseA + mix * bestA, -cap, cap);
  }
  return baseA;
}

function updateEliteMoments(candidates, eliteCount, mean, std, floor) {
  const blockCount = mean.length;
  mean.fill(0);
  for (let c = 0; c < eliteCount; c++) {
    const blocks = candidates[c].blocks;
    for (let i = 0; i < blockCount; i++) mean[i] += blocks[i];
  }
  const invElite = 1 / eliteCount;
  for (let i = 0; i < blockCount; i++) mean[i] *= invElite;

  std.fill(0);
  for (let c = 0; c < eliteCount; c++) {
    const blocks = candidates[c].blocks;
    for (let i = 0; i < blockCount; i++) {
      const d = blocks[i] - mean[i];
      std[i] += d * d;
    }
  }
  for (let i = 0; i < blockCount; i++) std[i] = Math.max(floor, Math.sqrt(std[i] * invElite));
}

function blocksToSequence(blocks, horizon, blockLen) {
  const seq = new Array(horizon);
  let i = 0;
  for (let b = 0; b < blocks.length && i < horizon; b++) {
    const value = blocks[b];
    const end = Math.min(horizon, i + blockLen);
    for (; i < end; i++) seq[i] = value;
  }
  const last = blocks.length > 0 ? blocks[blocks.length - 1] : 0;
  for (; i < horizon; i++) seq[i] = last;
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

function extendBlocks(blocks, blockCount) {
  const out = new Array(blockCount);
  const last = blocks.length > 0 ? blocks[blocks.length - 1] : 0;
  for (let i = 0; i < blockCount; i++) out[i] = i < blocks.length ? blocks[i] : last;
  return out;
}

function firstBlockDeltaSafe(baseBlocks, candidateBlocks, A) {
  if (!baseBlocks.length || !candidateBlocks.length) return true;
  // Extra-search candidates are meant to polish the old plan, not replace it
  // with an unrelated bang-bang command. A first-action guard avoids the few
  // cases where a longer horizon gives an attractive rollout cost but kicks the
  // real system out of the already good capture basin.
  return Math.abs(candidateBlocks[0] - baseBlocks[0]) <= Math.max(0.62 * A, 3.2);
}

function improvePlanWithExtraSearch(state, target, params, t, baseBlocks, baseHorizon, blockLen, A, energyA, alignA, centerA, reserveA, bridgeA, scoreCtx, allowState3Extra = false) {
  if (!wasmRollout.isReady() || !baseBlocks || baseBlocks.length === 0 || target.id === 0 || (target.id === 3 && !allowState3Extra)) return baseBlocks;

  // Spend only the WASM headroom here. The primary CEM geometry remains the
  // proven previous version; this extra pass is currently enabled for states 1
  // and 2 only. Regression tests showed that state 0 and state 3 are more
  // sensitive to a longer horizon, so they keep the stable existing path.
  // A replacement is accepted only when exact JS rescoring is clearly better
  // than the original plan.
  const extraHorizon = baseHorizon + (target.id === 3 ? 6 : 6);
  const extraBlockCount = Math.ceil(extraHorizon / blockLen);
  const baseExtended = extendBlocks(baseBlocks, extraBlockCount);
  const baseCost = runPredictionBlocks(state, target, params, baseExtended, extraHorizon, blockLen, t, scoreCtx);
  const extraCount = target.id === 3 ? 64 : 40;
  const refineCount = target.id === 3 ? 8 : 8;
  const mutationScale = target.id === 3 ? 0.12 : 0.155;

  const candidates = [{ blocks: baseExtended, cost: baseCost, exact: true, base: true }];
  const newCandidates = [];
  for (let i = 0; i < extraCount; i++) {
    let blocks;
    if (i < (target.id === 3 ? 18 : 12)) {
      blocks = makeHeuristicBlocks(i, extraBlockCount, A, energyA, alignA, centerA, reserveA, bridgeA, target);
    } else {
      const taperPower = i < 22 ? 1.0 : 1.35;
      blocks = new Array(extraBlockCount);
      for (let j = 0; j < extraBlockCount; j++) {
        const phase = extraBlockCount <= 1 ? 0 : j / (extraBlockCount - 1);
        const taper = Math.pow(1 - 0.45 * phase, taperPower);
        const bias = i % 4 === 0 ? 0.10 * reserveA : (i % 4 === 1 ? 0.10 * centerA : (i % 4 === 2 ? 0.08 * alignA : 0.08 * energyA));
        blocks[j] = clamp(baseExtended[j] + bias + gaussianRandom() * mutationScale * A * taper, -A, A);
      }
    }
    newCandidates.push({ blocks, cost: Infinity, exact: false, base: false });
  }
  scoreCandidateBlocks(state, target, params, newCandidates, extraHorizon, blockLen, t, scoreCtx, wasmRollout.isReady());
  candidates.push(...newCandidates);

  candidates.sort((a, b) => a.cost - b.cost);
  for (let i = 0; i < Math.min(refineCount, candidates.length); i++) {
    if (!candidates[i].exact) {
      candidates[i].cost = runPredictionBlocks(state, target, params, candidates[i].blocks, extraHorizon, blockLen, t, scoreCtx);
      candidates[i].exact = true;
    }
  }
  candidates.sort((a, b) => a.cost - b.cost);

  const best = candidates[0];
  const improvement = baseCost - best.cost;
  const requiredGain = Math.max((target.id === 3 ? 0.030 : 0.018) * Math.max(1, Math.abs(baseCost)), target.id === 3 ? 0.035 : 0.020);
  if (!best.base && best.exact && improvement > requiredGain && firstBlockDeltaSafe(baseExtended, best.blocks, A)) {
    return best.blocks;
  }
  return baseExtended;
}

function makeHeuristicBlocks(index, blockCount, A, energyA, alignA, centerA, reserveA, bridgeA, target) {
  const blocks = new Array(blockCount);
  const bias = clamp(0.42 * energyA + 0.22 * alignA + 0.18 * centerA + 0.18 * reserveA, -A, A);

  if (index === 0) return blocks.fill(energyA);
  if (index === 1) return blocks.fill(alignA);
  if (index === 2) return blocks.fill(centerA);
  if (index === 3) return blocks.fill(reserveA);
  if (index === 4 && bridgeA !== null) return blocks.fill(bridgeA);
  if (index === 4) return blocks.fill(0);
  if (index === 5) return blocks.fill(A);
  if (index === 6) return blocks.fill(-A);
  if (index === 7) {
    for (let i = 0; i < blockCount; i++) blocks[i] = i < blockCount * 0.42 ? reserveA : alignA;
    return blocks;
  }
  if (index === 8) {
    for (let i = 0; i < blockCount; i++) blocks[i] = i < blockCount * 0.36 ? reserveA : (i < blockCount * 0.70 ? energyA : alignA);
    return blocks;
  }

  const phaseDenom = Math.max(1, blockCount - 1);
  const twoPi = 2 * Math.PI;
  const cutThird = blockCount / 3;
  const cutTwoThirds = 2 * blockCount / 3;
  for (let i = 0; i < blockCount; i++) {
    const phase = i / phaseDenom;
    if (index === 9) blocks[i] = A * Math.sin(twoPi * (1.0 + 0.15 * target.id) * phase);
    else if (index === 10) blocks[i] = A * Math.sin(twoPi * (1.5 + 0.2 * target.id) * phase + Math.PI / 2);
    else if (index === 11) blocks[i] = i % 2 === 0 ? A : -A;
    else if (index === 12) blocks[i] = i < blockCount / 2 ? A : -A;
    else if (index === 13) blocks[i] = i < blockCount / 2 ? -A : A;
    else if (target.id === 3 && index === 14) blocks[i] = i < cutThird ? reserveA : (i < cutTwoThirds ? -0.90 * A : 0.55 * centerA);
    else if (target.id === 3 && index === 15) blocks[i] = i < cutThird ? reserveA : (i < cutTwoThirds ? 0.90 * A : 0.55 * alignA);
    else if (target.id === 3 && index === 16) blocks[i] = clamp(0.70 * alignA + 0.30 * energyA, -A, A);
    else if (target.id === 3 && index === 17) blocks[i] = clamp(0.52 * energyA + 0.22 * alignA + 0.18 * reserveA + 0.08 * A * Math.sin(Math.PI * phase), -A, A);
    else blocks[i] = clamp(bias + randomBetween(-0.65 * A, 0.65 * A), -A, A);
  }
  return blocks;
}

function chooseCemAcceleration(state, target, params, t, previousPlan, state3SearchMode = 0) {
  const authorityNear = closenessToTarget(state, target);
  const A = planningAuthority(state, target, params, authorityNear.angleNorm, authorityNear.speedNorm);
  const downRoughEnvironment = target.id === 0 && ((params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.12);
  const downHighAuthority = target.id === 0 && params.maxAcc > 20 && !downRoughEnvironment;
  const state3RightExtra = target.id === 3 && state3SearchMode === 1;
  const state3WideLeft = target.id === 3 && state3SearchMode === 2;
  const highAuthority = params.maxAcc > 20;
  const horizon = target.id === 0 ? (downHighAuthority ? 46 : 50) : (state3RightExtra && highAuthority ? 50 : (highAuthority && target.id === 3 ? 45 : 42));
  const blockLen = 3;
  const blockCount = Math.ceil(horizon / blockLen);
  const sampleCount = target.id === 0 ? (downHighAuthority ? 52 : 56) : (state3RightExtra && highAuthority ? 96 : (state3WideLeft ? 84 : (highAuthority ? (target.id === 3 ? 76 : 64) : 56)));
  const eliteCount = target.id === 0 ? 8 : 8;
  const generations = 2;
  const energyA = energyPumpAcceleration(state, target, params);
  const alignA = targetAlignmentAcceleration(state, target, params);
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.18 : 1.0);
  const reserveA = reservePrepositionAcceleration(state, target, params, authorityNear, alignA);
  const bridgeA = target.id === 3 ? target3SingleLinkBridgeAcceleration(state, params) : null;
  const scoreCtx = makeScoreContext(target, params);

  let mean = sequenceToBlocks(previousPlan, blockCount, blockLen, clamp(bridgeA !== null ? (0.34 * bridgeA + 0.28 * energyA + 0.14 * alignA + 0.10 * centerA + 0.14 * reserveA) : (0.40 * energyA + 0.21 * alignA + 0.20 * centerA + 0.19 * reserveA), -A, A));
  let std = Array(blockCount).fill(Math.max(0.18 * A, 1.5));

  let bestBlocks = mean.slice();
  let bestCost = Infinity;

  const wasmReady = wasmRollout.isReady();

  for (let gen = 0; gen < generations; gen++) {
    const candidates = new Array(sampleCount);

    for (let i = 0; i < sampleCount; i++) {
      let blocks;
      if (gen === 0 && i < (target.id === 3 ? 18 : 15)) {
        blocks = makeHeuristicBlocks(i, blockCount, A, energyA, alignA, centerA, reserveA, bridgeA, target);
      } else {
        blocks = new Array(blockCount);
        for (let j = 0; j < blockCount; j++) blocks[j] = clamp(mean[j] + gaussianRandom() * std[j], -A, A);
      }
      candidates[i] = { blocks, cost: Infinity };
    }

    scoreCandidateBlocks(state, target, params, candidates, horizon, blockLen, t, scoreCtx, wasmReady);
    candidates.sort((a, b) => a.cost - b.cost);

    // The WASM path uses a fast self-contained trig approximation.  Keep the
    // speedup, but rescore a small elite prefix with the exact JS rollout before
    // updating CEM statistics.  This makes the optimization low-risk: WASM does
    // the broad pruning; JS preserves the final ranking near the top.
    if (wasmReady) {
      const refineCount = Math.min(candidates.length, target.id === 3 ? (state3WideLeft ? 8 : 14) : 12);
      for (let i = 0; i < refineCount; i++) {
        candidates[i].cost = runPredictionBlocks(state, target, params, candidates[i].blocks, horizon, blockLen, t, scoreCtx);
      }
      candidates.sort((a, b) => a.cost - b.cost);
    }

    if (candidates[0].cost < bestCost) {
      bestCost = candidates[0].cost;
      bestBlocks = candidates[0].blocks.slice();
    }

    updateEliteMoments(candidates, eliteCount, mean, std, Math.max(0.055 * A, 0.35));
  }

  if (wasmReady) {
    bestBlocks = improvePlanWithExtraSearch(state, target, params, t, bestBlocks, horizon, blockLen, A, energyA, alignA, centerA, reserveA, bridgeA, scoreCtx, state3RightExtra);
  }

  const bestSequence = blocksToSequence(bestBlocks, Math.max(horizon, bestBlocks.length * blockLen), blockLen);
  return {
    acceleration: clamp(bestSequence[0], -A, A),
    plan: bestSequence.slice(1)
  };
}


function robustLocalAcceleration(state, target, params, lqrA) {
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.20 : 0.85);
  if (target.id !== 3) {
    const [e1, e2] = targetError(state, target);
    const angleNorm = Math.hypot(e1, e2);
    const speedNorm = Math.hypot(state.om1, state.om2);
    let centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, 0.06, lqrA);
    if (target.id === 2 && angleNorm < 0.050 && speedNorm < 0.16) centerMix = Math.max(centerMix, 0.20);
    if (target.id === 2 && angleNorm < 0.026 && speedNorm < 0.08) centerMix = Math.max(centerMix, 0.32);
    const localCapBase = params.maxAcc <= 20 ? params.maxAcc : Math.min(params.maxAcc, ((params.friction || 0) >= 0.04 ? 7.0 : 5.0) + ((params.friction || 0) >= 0.04 ? 6.2 : 4.2) * angleNorm + ((params.friction || 0) >= 0.04 ? 1.8 : 1.1) * speedNorm);
    const localCap = params.maxAcc <= 20 ? localCapBase : Math.max(localCapBase, localLandingCap(state, target, params, angleNorm, speedNorm) * 0.62);
    return clamp((1 - centerMix) * lqrA + centerMix * centerA, -localCap, localCap);
  }

  const alignA = targetAlignmentAcceleration(state, target, params);
  const [e1, e2] = targetError(state, target);
  const angleNorm = Math.hypot(e1, e2);
  const speedNorm = Math.hypot(state.om1, state.om2);
  const supportNorm = Math.abs(supportCenterError(state, params)) + 0.7 * Math.abs(state.vx);
  const dampingA = clamp(
    -0.75 * Math.sin(e1) - 0.95 * Math.sin(e2) - 0.18 * state.om1 - 0.24 * state.om2,
    -0.22 * params.maxAcc,
    0.22 * params.maxAcc
  );
  const energyDampingA = clamp(5.4 * supportPhasePower(state, params), -0.30 * params.maxAcc, 0.30 * params.maxAcc);
  const roughEnvironment = (params.friction || 0) < 0.015 || (params.windAmp || 0) > 0.15;
  let centerMix = balanceFirstCenterBlend(state, target, params, angleNorm, speedNorm, roughEnvironment ? 0.040 : 0.045, lqrA);
  if (roughEnvironment) {
    const half = supportHalfSpan(params);
    const xErr = supportCenterError(state, params);
    const edgeRatio = Math.abs(xErr) / half;
    const outward = xErr * state.vx > 0;
    if (!(angleNorm < 0.10 && speedNorm < 0.42) && !(edgeRatio > 0.86 && outward)) {
      centerMix = Math.min(centerMix, 0.16);
    }
  }
  // Once the links are truly quiet, finish the job by recentering the support.
  // This remains gated by a tight angular/rate band so the recentering term
  // cannot steal authority during capture.
  if (angleNorm < 0.045 && speedNorm < 0.16) centerMix = Math.max(centerMix, roughEnvironment ? 0.24 : 0.18);
  if (angleNorm < 0.025 && speedNorm < 0.08) centerMix = Math.max(centerMix, roughEnvironment ? 0.34 : 0.28);
  const lqrMix = roughEnvironment ? clamp(0.965 - centerMix, 0.48, 0.98) : clamp(0.94 - centerMix, 0.40, 0.96);
  const state3LowAuthorityWindy = target.id === 3 && (params.maxAcc || 0) < 19 && (params.windAmp || 0) > 0.075 && (params.friction || 0) >= 0.035;
  const alignMix = state3LowAuthorityWindy ? (angleNorm < 0.36 && speedNorm < 2.5 ? (roughEnvironment ? 0.018 : 0.025) : (roughEnvironment ? 0.045 : 0.055)) : (angleNorm < 0.36 && speedNorm < 2.5 ? (roughEnvironment ? 0.024 : 0.035) : (roughEnvironment ? 0.052 : 0.068));
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const energyExcess = Math.max(0, (totalEnergy(state, params) - energyAtTarget(target, params)) / energyScale);
  const dampingGate = clamp((0.88 - angleNorm) / 0.88, 0, 1) * clamp((speedNorm - 0.22) / 4.8, 0, 1) * clamp(energyExcess / 0.18, 0, 1);
  const energyDampingMix = (state3LowAuthorityWindy ? (roughEnvironment ? 0.050 : 0.065) : (roughEnvironment ? 0.075 : 0.105)) * dampingGate;
  const localCapBase = params.maxAcc <= 20 ? params.maxAcc : Math.min(params.maxAcc, ((params.friction || 0) >= 0.04 ? 6.0 : 5.2) + ((params.friction || 0) >= 0.04 ? 5.0 : 5.0) * angleNorm + ((params.friction || 0) >= 0.04 ? 3.0 : 1.25) * speedNorm);
  const landingCap = localLandingCap(state, target, params, angleNorm, speedNorm);
  const localCap = params.maxAcc <= 20 ? localCapBase : Math.max(localCapBase, landingCap * (roughEnvironment ? 0.72 : 0.66));
  return clamp(lqrMix * lqrA + alignMix * alignA + centerMix * centerA + (state3LowAuthorityWindy ? (roughEnvironment ? 0.014 : 0.010) : (roughEnvironment ? 0.030 : 0.040)) * dampingA + energyDampingMix * energyDampingA, -localCap, localCap);
}


const DEFAULT_INDEX_AI_CENTER = Object.freeze({ maxAcc: 20.0, g: 9.0, windAmp: 0.03, friction: 0.03 });
const DEFAULT_INDEX_AI_RADIUS = Object.freeze({
  // 10% of each index.html slider axis: maxAcc [1,60], g [0,12], wind/friction [0,1].
  maxAcc: 5.9,
  g: 1.2,
  windAmp: 0.10,
  friction: 0.10
});

function isDefaultIndexState3AIRange(target, params) {
  if (!target || target.id !== 3 || !params) return false;
  return Math.abs((Number(params.maxAcc) || 0) - DEFAULT_INDEX_AI_CENTER.maxAcc) <= DEFAULT_INDEX_AI_RADIUS.maxAcc &&
    Math.abs((Number(params.g) || 0) - DEFAULT_INDEX_AI_CENTER.g) <= DEFAULT_INDEX_AI_RADIUS.g &&
    Math.abs((Number(params.windAmp) || 0) - DEFAULT_INDEX_AI_CENTER.windAmp) <= DEFAULT_INDEX_AI_RADIUS.windAmp &&
    Math.abs((Number(params.friction) || 0) - DEFAULT_INDEX_AI_CENTER.friction) <= DEFAULT_INDEX_AI_RADIUS.friction;
}
function adaptiveStrength(params, profile) {
  if (!profile) return 0;
  const maxA = Math.max(1, params.maxAcc || 1);
  const highAStress = clamp((maxA - 26) / 18, 0, 1);
  const lowFrictionStress = clamp((0.025 - (params.friction || 0)) / 0.025, 0, 1);
  const lightGStress = clamp((8.2 - (params.g || 0)) / 8.2, 0, 1);
  const windStress = clamp(Math.abs(params.windAmp || 0) / 0.18, 0, 1);
  const learned = clamp((profile.learnedVisits || 0) / 5, 0, 1);
  return clamp(Math.max(learned, highAStress, 0.45 * lowFrictionStress, 0.40 * lightGStress, 0.35 * windStress), 0, 1);
}

function blendProfileValue(value, strength) {
  return 1 + (value - 1) * clamp(strength, 0, 1);
}

function adaptiveCaptureEnvelope(target, params, profile) {
  if (target.id === 0) return { angleScale: 1, speedScale: 1, lqrSpeedScale: 1 };
  const maxA = Math.max(1, params.maxAcc || 1);
  const highA = clamp((maxA - 20) / 40, 0, 1);
  const lowFriction = clamp((0.045 - (params.friction || 0)) / 0.045, 0, 1);
  const lightG = clamp((8.6 - (params.g || 0)) / 8.6, 0, 1);
  const conservatism = clamp(profile?.captureConservatism || 1, 0.70, 1.75);
  const damping = clamp(profile?.speedDamping || 1, 0.70, 1.90);

  const strength = adaptiveStrength(params, profile);
  const rawAngleScale = clamp((1 - 0.08 * highA - 0.04 * lowFriction + 0.03 * lightG) / Math.sqrt(conservatism), 0.70, 1.10);
  const rawSpeedScale = clamp((1 - 0.14 * highA - 0.055 * lowFriction - 0.035 * lightG) / (0.78 + 0.22 * conservatism + 0.06 * (damping - 1)), 0.62, 1.10);
  const rawLqrSpeedScale = clamp(1 + 0.18 * highA - 0.10 * (conservatism - 1), 0.76, 1.18);
  const angleScale = blendProfileValue(rawAngleScale, strength);
  const speedScale = blendProfileValue(rawSpeedScale, strength);
  const lqrSpeedScale = blendProfileValue(rawLqrSpeedScale, strength);
  return { angleScale, speedScale, lqrSpeedScale };
}

function adaptiveLocalCap(state, target, params, near, profile) {
  const base = localLandingCap(state, target, params, near.angleNorm, near.speedNorm);
  if (!profile || target.id === 0) return base;
  const strength = adaptiveStrength(params, profile);
  if (strength <= 0.001) return base;
  const highA = clamp((Math.max(1, params.maxAcc || 1) - 20) / 40, 0, 1);
  const brake = clamp(profile.brakeGain || 1, 0.70, 1.85);
  const conservatism = clamp(profile.captureConservatism || 1, 0.70, 1.75);
  const rawScale = clamp((profile.authorityScale || 1) * (1 - 0.12 * highA * (brake - 1)) / (0.90 + 0.10 * conservatism), 0.55, 1.18);
  const scale = blendProfileValue(rawScale, strength);
  return clamp(base * scale, Math.min(params.maxAcc, 4.0), params.maxAcc);
}

function adaptiveCaptureBrakeAcceleration(state, raw, target, params, near, profile) {
  if (!profile || target.id === 0) return raw;
  const maxA = Math.max(1, params.maxAcc || 1);
  const strength = adaptiveStrength(params, profile);
  if (maxA < 28 && strength < 0.45) return raw;
  const radial = targetRadialVelocity(state, target);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const dE = (totalEnergy(state, params) - energyAtTarget(target, params)) / energyScale;
  const angleLimit = target.id === 3 ? 1.04 : (target.id === 2 ? 0.74 : 0.66);
  const speedLimit = target.id === 3 ? 7.8 : (target.id === 2 ? 5.8 : 5.0);
  const captureGate = clamp(1 - (near.angleNorm / angleLimit + near.speedNorm / speedLimit) * 0.5, 0, 1);
  if (captureGate <= 0) return raw;
  const quietFinal = near.angleNorm < (target.id === 3 ? 0.24 : 0.18) && near.speedNorm < (target.id === 3 ? 0.82 : 0.62) && supportEdgeRatio(state, params) < 0.82;
  if (quietFinal) return raw;

  const speedSoft = target.id === 3 ? 2.75 : 1.80;
  const speedGate = clamp((near.speedNorm - speedSoft) / Math.max(0.6, speedLimit - speedSoft), 0, 1);
  const flybyGate = clamp(radial / (target.id === 3 ? 0.56 : 0.40), 0, 1);
  const energyGate = clamp(Math.max(0, dE - 0.05) / (target.id === 3 ? 0.34 : 0.26), 0, 1);
  const edgeGate = clamp((supportEdgeRatio(state, params) - 0.70) / 0.25, 0, 1);
  const need = Math.max(speedGate, flybyGate, energyGate, 0.65 * edgeGate);
  if (need <= 0.02) return raw;

  const lqrA = pseudoLqrAcceleration(state, target, params);
  const alignA = targetAlignmentAcceleration(state, target, params);
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.18 * profile.centerBias : 1.04 * profile.centerBias);
  const phaseA = clamp(5.6 * profile.brakeGain * supportPhasePower(state, params), -0.34 * maxA, 0.34 * maxA);
  const reserveA = reservePrepositionAcceleration(state, target, params, near, alignA);
  const brakingA = clamp(
    0.42 * lqrA + 0.24 * alignA + 0.16 * phaseA + 0.10 * centerA + 0.08 * reserveA,
    -maxA,
    maxA
  );
  const highA = clamp((maxA - 20) / 40, 0, 1);
  const mix = clamp((0.06 + 0.22 * need + 0.06 * highA) * blendProfileValue(profile.brakeGain * profile.speedDamping, strength), 0.04, target.id === 3 ? 0.44 : 0.40);
  const cap = adaptiveLocalCap(state, target, params, near, profile);
  return clamp((1 - mix) * raw + mix * brakingA, -cap, cap);
}

function adaptiveEdgeReserveAcceleration(state, raw, target, params, near, profile) {
  if (!profile || target.id === 0) return raw;
  const maxA = Math.max(1, params.maxAcc || 1);
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const ratio = Math.abs(xErr) / Math.max(0.01, half);
  const outward = supportOutwardSign(state, params);
  const movingOutward = xErr * state.vx > 0;
  const risk = sameSideBoundaryRisk(state, raw, target, params, near, target.id === 3 ? 0.58 : 0.48);
  const strength = adaptiveStrength(params, profile);
  if (maxA < 28 && strength < 0.45 && Math.max(ratio, risk.predictedRatio || ratio) < 0.92) return raw;
  const highA = clamp((maxA - 20) / 40, 0, 1);
  const start = clamp((target.id === 3 ? 0.76 : 0.72) - 0.055 * (profile.edgeReserve - 1) - 0.035 * highA, 0.60, 0.82);
  const predicted = Math.max(ratio, risk.predictedRatio || ratio);
  const edgeNeed = clamp((predicted - start) / Math.max(0.06, 0.98 - start), 0, 1);
  const outwardNeed = movingOutward ? clamp(Math.abs(state.vx) / Math.max(0.55, 0.14 * params.maxAcc), 0, 1) : 0;
  const commandNeed = risk.commandOutward ? 0.60 : 0;
  const need = Math.max(edgeNeed, outwardNeed, commandNeed);
  if (need <= 0.01) return raw;

  const centerA = centerReturnAcceleration(state, params, (target.id === 3 ? 1.20 : 1.08) * profile.centerBias);
  const inwardBrake = clamp(centerA - outward * params.maxAcc * (0.045 + 0.16 * edgeNeed) * profile.edgeReserve - 0.22 * state.vx, -params.maxAcc, params.maxAcc);
  const mix = clamp((0.06 + 0.28 * need) * blendProfileValue(profile.landingGuard, strength), 0.04, target.id === 3 ? 0.48 : 0.46);
  const cap = adaptiveLocalCap(state, target, params, near, profile);
  return clamp((1 - mix) * raw + mix * inwardBrake, -cap, cap);
}



function phaseFeaturesForLearning(state, target, params, near = null) {
  const n = near || closenessToTarget(state, target);
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const energyDelta = (totalEnergy(state, params) - energyAtTarget(target, params)) / energyScale;
  const delta = state.th1 - state.th2;
  const angularMomentum =
    (params.m1 + params.m2) * params.l1 * params.l1 * state.om1 +
    params.m2 * params.l2 * params.l2 * state.om2 +
    0.5 * params.m2 * params.l1 * params.l2 * Math.cos(delta) * (state.om1 + state.om2);
  const edgeSide = Math.abs(xErr) < 1e-6 ? 0 : Math.sign(xErr);
  return {
    angleNorm: n.angleNorm,
    speedNorm: n.speedNorm,
    energyDelta,
    edgeRatio: Math.abs(xErr) / Math.max(0.01, half),
    edgeSide,
    outward: xErr * state.vx > 0,
    xNorm: xErr / Math.max(0.01, half),
    vxNorm: state.vx / Math.max(0.6, 0.16 * Math.max(1, params.maxAcc)),
    radial: targetRadialVelocity(state, target),
    phasePower: supportPhasePower(state, params) / Math.max(0.5, params.g * 0.12),
    angularMomentum: angularMomentum / Math.max(1, params.g * (params.m1 + params.m2) * params.l1)
  };
}

function learnedPhasePolicyAcceleration(state, raw, target, params, near, profile, advice) {
  if (!advice || target.id === 0) return raw;
  const maxA = Math.max(1, params.maxAcc || 1);
  const strength = adaptiveStrength(params, profile) || clamp((advice.confidence || 0) / 1.2, 0, 1);
  if (strength <= 0.02 && (advice.confidence || 0) < 0.25) return raw;

  const cap = adaptiveLocalCap(state, target, params, near, profile);
  const centerA = centerReturnAcceleration(state, params, (target.id === 3 ? 1.06 : 1.00) * (profile?.centerBias || 1));
  const alignA = targetAlignmentAcceleration(state, target, params);
  const reserveA = reservePrepositionAcceleration(state, target, params, near, alignA);
  const phaseA = clamp(4.9 * supportPhasePower(state, params) * (1 + Math.max(0, advice.brakeBias || 0)), -0.42 * cap, 0.42 * cap);

  const captureGate = clamp(1 - (near.angleNorm / (target.id === 3 ? 1.20 : 0.86) + near.speedNorm / (target.id === 3 ? 8.0 : 5.8)) * 0.5, 0, 1);
  const edgeGate = clamp((supportEdgeRatio(state, params) - 0.68) / 0.26, 0, 1);
  const activeGate = clamp(0.35 + 0.45 * captureGate + 0.20 * edgeGate, 0.24, 1.0);

  const ruleA = clamp(
    raw * (advice.authorityScale || 1) +
    maxA * (advice.actionBias || 0) +
    centerA * (0.20 + 1.15 * Math.max(0, advice.centerBias || 0)) +
    reserveA * (0.14 + 1.05 * Math.max(0, advice.reserveBias || 0)) +
    alignA * (0.10 + 0.95 * Math.max(0, advice.alignBias || 0)) +
    phaseA * (0.10 + 0.95 * Math.max(0, advice.brakeBias || 0)),
    -cap,
    cap
  );
  // Learned phase rules are advisory, not a replacement for the hand-built
  // stabilizer.  Keep the blend deliberately small so a coarse offline rule can
  // help with repeated fly-by/edge cases without overpowering a working LQR/MPC
  // command in neighboring parameter ranges.
  const blend = clamp((advice.blend || 0) * (0.22 + 0.38 * strength) * activeGate * Math.min(1.0, advice.confidence || 1), 0, target.id === 3 ? 0.085 : 0.070);
  if (blend <= 0.002) return raw;
  return clamp((1 - blend) * raw + blend * ruleA, -cap, cap);
}

function nearestEquilibrium(state) {
  let best = { id: -1, angleNorm: Infinity, speedNorm: Math.hypot(state.om1, state.om2) };
  const speedNorm = Math.hypot(state.om1, state.om2);
  for (const target of TARGETS) {
    const e1 = angleError(state.th1, target.angles[0]);
    const e2 = angleError(state.th2, target.angles[1]);
    const angleNorm = Math.hypot(e1, e2);
    if (angleNorm < best.angleNorm) best = { id: target.id, angleNorm, speedNorm };
  }
  return best;
}

function escapeKickDirection(state, params) {
  const rail = supportBounds(params);
  const rightRoom = rail.right - state.x;
  const leftRoom = state.x - rail.left;
  if (Math.abs(rightRoom - leftRoom) > 0.20 * rail.half) return rightRoom > leftRoom ? 1 : -1;
  // Deterministic tie-breaker: with the asymmetric extended right rail, the
  // right side has more useful workspace from the default x=0 start.
  return 1;
}

function escapeKickAcceleration(state, target, params, direction) {
  const maxA = params.maxAcc;
  const near = closenessToTarget(state, target);
  const baseCap = maxA <= 20 ? maxA : planningAuthority(state, target, params, near.angleNorm, near.speedNorm);
  const rail = supportBounds(params);
  const xErr = state.x - rail.center;
  const edgeRatio = Math.abs(xErr) / rail.half;
  const edgeRoomGate = direction * xErr > 0 ? clamp(1 - (edgeRatio - 0.58) / 0.34, 0.28, 1.0) : 1.0;
  return clamp(direction * baseCap * 0.78 * edgeRoomGate, -maxA, maxA);
}

function escapePatternDirection(state, target, params, sourceId) {
  let direction = escapeKickDirection(state, params);
  // Preserve the empirically useful mirror case from the older one-shot kick.
  if (sourceId === 3 && target.id === 1) direction = -direction;
  if (sourceId === 1 && target.id === 2) direction = -direction;
  return direction || 1;
}

function escapePatternDuration(target, params, sourceId) {
  const maxA = Math.max(16, params.maxAcc || 0);
  if (target.id === 3) return clamp(9.5 / maxA, 0.24, 0.48);
  if (sourceId === 3 && target.id !== 0) return clamp(8.6 / maxA, 0.22, 0.44);
  return clamp(7.0 / maxA, 0.18, 0.44);
}


function escapePatternAcceleration(state, target, params, elapsed, total, direction, sourceId) {
  // Fixed, repeatable escape pulse.  For the low-energy state0->state3 case,
  // use a short two-pulse launch: first create angular motion, then recover
  // support reserve before the cart reaches the rail.  Other sources keep the
  // proven one-sided kick.
  if (target.id === 3 && sourceId === 0 && total > 0) {
    const phase = clamp(elapsed / Math.max(1e-6, total), 0, 1);
    if (phase > 0.58) return escapeKickAcceleration(state, target, params, -0.46 * direction);
  }
  return escapeKickAcceleration(state, target, params, direction);
}



function targetRadialVelocity(state, target) {
  const [e1, e2] = targetError(state, target);
  const angleNorm = Math.hypot(e1, e2);
  return angleNorm < 1e-6 ? 0 : (e1 * state.om1 + e2 * state.om2) / angleNorm;
}

function missedTargetAttempt(state, target, params, near, prevNear) {
  if (!prevNear || target.id === 0 || target.id === 2) return false;
  const edgeStress = supportEdgeRatio(state, params) > (target.id === 3 ? 0.86 : 0.82) && supportCenterError(state, params) * state.vx > 0;
  const closeRescueBand = !edgeStress && near.angleNorm < (target.id === 3 ? 0.44 : 0.30) && near.speedNorm < (target.id === 3 ? 5.8 : 4.0);
  const radial = targetRadialVelocity(state, target);
  const movingAway = !closeRescueBand && near.angleNorm > prevNear.angleNorm + 0.010 && radial > 0.24;
  const edgeBad = supportEdgeRatio(state, params) > (target.id === 3 ? 0.91 : 0.87) && supportCenterError(state, params) * state.vx > 0 && near.angleNorm > (target.id === 3 ? 0.34 : 0.25);
  if (target.id === 3) {
    const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
    const dE = (totalEnergy(state, params) - energyAtTarget(target, params)) / energyScale;
    const fastFlyby = near.angleNorm < 0.52 && near.speedNorm > (params.maxAcc > 20 ? 8.2 : 6.2);
    const wrongEnergy = dE > 0.20 && near.angleNorm < 0.88 && near.speedNorm > 4.8;
    return fastFlyby || wrongEnergy || edgeBad || (movingAway && prevNear.angleNorm < 0.58 && near.speedNorm > 2.9);
  }
  const fastFlyby = near.angleNorm < (target.id === 1 ? 0.34 : 0.38) && near.speedNorm > (params.maxAcc > 20 ? 6.4 : 5.0);
  if (target.id === 2) return fastFlyby || edgeBad;
  return fastFlyby || edgeBad || (movingAway && prevNear.angleNorm < 0.40 && near.speedNorm > 2.4);
}

function retryPreparationAcceleration(state, target, params, direction = 0) {
  const near = closenessToTarget(state, target);
  const A = planningAuthority(state, target, params, near.angleNorm, near.speedNorm);
  const energyA = energyPumpAcceleration(state, target, params);
  const alignA = targetAlignmentAcceleration(state, target, params);
  const centerA = centerReturnAcceleration(state, params, target.id === 3 ? 1.12 : 1.02);
  const relaunchSign = direction || escapeKickDirection(state, params);
  const relaunchA = relaunchSign * A * (target.id === 3 ? 0.46 : 0.34);
  let raw;
  if (target.id === 3) {
    const bridgeA = target3SingleLinkBridgeAcceleration(state, params);
    raw = bridgeA !== null
      ? 0.52 * bridgeA + 0.20 * energyA + 0.14 * centerA + 0.14 * relaunchA
      : 0.44 * energyA + 0.20 * alignA + 0.18 * centerA + 0.18 * relaunchA;
    if (near.angleNorm < 0.60 && near.speedNorm > 5.2) {
      const phaseDump = clamp(5.0 * supportPhasePower(state, params), -0.28 * params.maxAcc, 0.28 * params.maxAcc);
      raw = 0.55 * raw + 0.25 * phaseDump + 0.20 * centerA;
    }
  } else {
    raw = 0.42 * energyA + 0.26 * alignA + 0.18 * relaunchA + 0.14 * centerA;
  }
  const edgeRatio = supportEdgeRatio(state, params);
  const outward = supportCenterError(state, params) * state.vx > 0;
  if (edgeRatio > 0.76 || outward) {
    const edgeGate = Math.max(clamp((edgeRatio - 0.76) / 0.20, 0, 1), outward ? clamp(Math.abs(state.vx) / Math.max(0.8, 0.18 * params.maxAcc), 0, 1) : 0);
    raw = (1 - 0.55 * edgeGate) * raw + 0.55 * edgeGate * centerA;
  }
  return clamp(raw, -A, A);
}

export class PendulumController {
  constructor(params) {
    this.params = params;
    this.target = TARGETS[0];
    this.cachedTargetId = -1;
    this.cachedGravity = NaN;
    this.cachedFriction = NaN;
    this.cachedGain = null;
    this.commandAcc = 0;
    this.controlAccumulator = 0;
    this.plan = [];
    this.localCaptureActive = false;
    this.escapeTimer = 0;
    this.escapeDuration = 0;
    this.escapeDirection = 0;
    this.escapeSourceId = -1;
    this.escapeCooldown = 0;
    this.wrongEquilibriumId = -1;
    this.wrongEquilibriumDwell = 0;
    this.prevNear = null;
    this.retryTimer = 0;
    this.retryDirection = 0;
    this.state3InitialEdgeBoostTimer = 0;
    this.state3WideLeftSearchTimer = 0;
    this.aiProfile = null;
    this.specialState3AIActive = false;
  }

  setTarget(id, stateHint = null) {
    this.target = TARGETS[id] || TARGETS[0];
    this.controlAccumulator = 0;
    this.plan = [];
    this.localCaptureActive = false;
    this.escapeTimer = 0;
    this.escapeDuration = 0;
    this.escapeDirection = 0;
    this.escapeSourceId = -1;
    this.escapeCooldown = 0;
    this.wrongEquilibriumId = -1;
    this.wrongEquilibriumDwell = 0;
    this.prevNear = null;
    this.retryTimer = 0;
    this.retryDirection = 0;
    this.state3InitialEdgeBoostTimer = 0;
    this.state3WideLeftSearchTimer = 0;
    this.aiProfile = null;
    this.specialState3AIActive = false;
    if (this.target.id === 3 && stateHint) {
      const edgeRatio = supportEdgeRatio(stateHint, this.params);
      const xErr = supportCenterError(stateHint, this.params);
      const smallAuthority = (this.params.maxAcc || 0) <= 20;
      const smallWind = (this.params.windAmp || 0) > 0 && (this.params.windAmp || 0) <= 0.06;
      if (edgeRatio >= 0.18 && edgeRatio <= 0.48 && ((xErr > 0 && (smallAuthority || Math.abs(this.params.windAmp || 0) < 1e-9)) || (smallAuthority && smallWind))) this.state3InitialEdgeBoostTimer = smallAuthority ? 1.20 : 10.0;
      if ((this.params.maxAcc || 0) <= 20 && edgeRatio >= 0.18 && edgeRatio <= 0.48 && xErr < 0 && Math.abs(this.params.windAmp || 0) < 1e-9) this.state3WideLeftSearchTimer = 14.0;
    }
    resetControllerRandomForTarget(this.target.id, stateHint);

    // If the user switches between exact equilibria, a symmetric controller can
    // dither around the old state for several cycles.  Start a deterministic
    // one-sided escape pulse so the system leaves the old basin immediately.
    if (stateHint) {
      const source = nearestEquilibrium(stateHint);
      const state3State2RemoteSource = this.target.id === 3 && source.id === 2 && source.angleNorm < 0.68 && source.speedNorm < 1.18;
      const state3RemoteSource0 = this.target.id === 3 && source.id === 0 && source.angleNorm < 0.50 && source.speedNorm < 0.92 && Math.abs(angleError(stateHint.th1, stateHint.th2)) < 0.78 && Math.abs(stateHint.om1 - stateHint.om2) < 1.35;
      if (source.id !== this.target.id && ((source.angleNorm < 0.24 && source.speedNorm < 0.62) || state3State2RemoteSource || state3RemoteSource0)) {
        this.escapeDirection = escapePatternDirection(stateHint, this.target, this.params, source.id);
        this.escapeSourceId = source.id;
        this.escapeDuration = state3State2RemoteSource ? Math.max(escapePatternDuration(this.target, this.params, source.id), 0.42) : (state3RemoteSource0 ? Math.max(escapePatternDuration(this.target, this.params, source.id), 0.30) : escapePatternDuration(this.target, this.params, source.id));
        this.escapeTimer = this.escapeDuration;
        this.escapeCooldown = state3State2RemoteSource ? 0.24 : (state3RemoteSource0 ? 0.22 : 0.36);
      }
    }

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
    return clamp(-lqrFeedback(this.cachedGain, state, this.target, params), -params.maxAcc, params.maxAcc);
  }

  suspendForPhysicalMode() {
    this.commandAcc = 0;
    this.plan = [];
    this.controlAccumulator = 0;
  }

  resumeFromPhysicalMode(stateHint = null) {
    this.commandAcc = 0;
    this.plan = [];
    this.controlAccumulator = 0;
    this.prevNear = null;
    this.wrongEquilibriumId = -1;
    this.wrongEquilibriumDwell = 0;
    if (stateHint) resetControllerRandomForTarget(this.target.id, stateHint);
  }

  maybeStartEquilibriumEscape(state, params) {
    if (this.escapeTimer > 0 || this.escapeCooldown > 0) return;
    const source = nearestEquilibrium(state);
    if (source.id === this.target.id || source.id < 0) {
      this.wrongEquilibriumId = -1;
      this.wrongEquilibriumDwell = 0;
      return;
    }

    const maxA = Math.max(1, params.maxAcc || 0);
    const source2Remote = this.target.id === 3 && source.id === 2;
    const source0Remote = this.target.id === 3 && source.id === 0 && Math.abs(angleError(state.th1, state.th2)) < 0.78 && Math.abs(state.om1 - state.om2) < 1.35;
    const angleBand = source2Remote ? 0.64 : (source0Remote ? 0.50 : (this.target.id === 3 ? 0.22 : 0.18));
    const speedBand = source2Remote ? 1.22 : (source0Remote ? 0.96 : (maxA > 20 ? 0.42 : 0.34));
    const stuck = source.angleNorm <= angleBand && source.speedNorm <= speedBand;
    if (!stuck) {
      this.wrongEquilibriumId = -1;
      this.wrongEquilibriumDwell = 0;
      return;
    }

    if (this.wrongEquilibriumId !== source.id) {
      this.wrongEquilibriumId = source.id;
      this.wrongEquilibriumDwell = 0;
    }
    this.wrongEquilibriumDwell += 0.045;
    if (this.wrongEquilibriumDwell < (this.target.id === 3 && source.id === 2 ? 0.12 : (this.target.id === 3 && source.id === 0 ? 0.16 : 0.22))) return;

    this.escapeDirection = escapePatternDirection(state, this.target, params, source.id);
    this.escapeSourceId = source.id;
    this.escapeDuration = this.target.id === 3 && source.id === 2 ? Math.max(escapePatternDuration(this.target, params, source.id), 0.42) : (this.target.id === 3 && source.id === 0 ? Math.max(escapePatternDuration(this.target, params, source.id), 0.30) : escapePatternDuration(this.target, params, source.id));
    this.escapeTimer = this.escapeDuration;
    this.escapeCooldown = this.target.id === 3 && source.id === 2 ? 0.26 : (this.target.id === 3 && source.id === 0 ? 0.24 : 0.42);
    this.wrongEquilibriumDwell = 0;
    this.plan = [];
    this.localCaptureActive = false;
  }


  update(state, params, dt, t) {
    this.controlAccumulator += dt;
    if (this.escapeTimer > 0) this.escapeTimer = Math.max(0, this.escapeTimer - dt);
    if (this.escapeCooldown > 0) this.escapeCooldown = Math.max(0, this.escapeCooldown - dt);
    if (this.retryTimer > 0) this.retryTimer = Math.max(0, this.retryTimer - dt);
    if (this.state3InitialEdgeBoostTimer > 0) this.state3InitialEdgeBoostTimer = Math.max(0, this.state3InitialEdgeBoostTimer - dt);
    if (this.state3WideLeftSearchTimer > 0) this.state3WideLeftSearchTimer = Math.max(0, this.state3WideLeftSearchTimer - dt);
    const specialState3AIActive = isDefaultIndexState3AIRange(this.target, params);
    this.specialState3AIActive = specialState3AIActive;
    this.aiProfile = specialState3AIActive ? adaptiveAI.getProfile(this.target.id, params) : null;
    const preNear = closenessToTarget(state, this.target);
    if (this.target.id === 0) this.prevNear = null;
    const updatePeriod = (this.target.id === 0 && preNear.angleNorm < 0.84 && preNear.speedNorm < 3.60) ? 0.023 :
      (this.target.id === 3 && params.maxAcc > 20 && preNear.angleNorm < 0.70 && preNear.speedNorm < 5.2) ? 0.020 :
      (this.target.id === 3 && preNear.angleNorm < 0.42 && preNear.speedNorm < 3.2) ? 0.022 :
      (this.target.id === 1 && preNear.angleNorm < 0.82 && preNear.speedNorm < 5.4) ? 0.024 :
      (this.target.id === 2 && preNear.angleNorm < 0.82 && preNear.speedNorm < 6.2) ? 0.026 : 0.045;
    const shouldUpdate = this.controlAccumulator >= updatePeriod;
    if (!shouldUpdate) {
      if (this.target.id !== 0) this.prevNear = preNear;
      return this.commandAcc;
    }
    this.controlAccumulator = 0;
    this.maybeStartEquilibriumEscape(state, params);

    let raw;
    const escapeActive = this.escapeTimer > 0 && preNear.angleNorm > 0.10 && preNear.speedNorm < 2.65;
    if (escapeActive) {
      const elapsed = Math.max(0, this.escapeDuration - this.escapeTimer);
      raw = escapePatternAcceleration(state, this.target, params, elapsed, this.escapeDuration || 0.35, this.escapeDirection || 1, this.escapeSourceId);
      this.plan = [];
    } else if (this.target.id === 0) {
      const nearDown = preNear;
      const supportNorm = Math.abs(supportCenterError(state, params)) + 0.7 * Math.abs(state.vx);
      const quietEnough = nearDown.angleNorm < 0.030 && nearDown.speedNorm < 0.040 && supportNorm < 0.030;
      if (quietEnough) {
        this.commandAcc = 0;
        this.plan = [];
        return 0;
      } else if (nearDown.angleNorm < 1.18 && nearDown.speedNorm < 5.80) {
        raw = downStabilizationAcceleration(state, params);
        this.plan = [];
      } else {
        const planned = chooseCemAcceleration(state, this.target, params, t, this.plan);
        raw = 0.74 * planned.acceleration + 0.26 * downStabilizationAcceleration(state, params);
        this.plan = planned.plan;
      }
    } else {
      const near = preNear;
      const missedAttempt = this.retryTimer <= 0 && missedTargetAttempt(state, this.target, params, near, this.prevNear);
      if (missedAttempt) {
        this.localCaptureActive = false;
        this.plan = [];
        this.retryDirection = supportCenterError(state, params) * state.vx > 0 ? -supportOutwardSign(state, params) : escapeKickDirection(state, params);
        this.retryTimer = this.target.id === 3 ? (params.maxAcc > 20 ? 0.22 : 0.30) : (params.maxAcc > 20 ? 0.16 : 0.22);
      }
      const adaptiveEnvelope = specialState3AIActive
        ? adaptiveCaptureEnvelope(this.target, params, this.aiProfile)
        : { angleScale: 1, speedScale: 1, lqrSpeedScale: 1 };
      const captureStartAngle = (this.target.id === 3 ? (params.maxAcc > 20 ? 0.66 : 0.56) : (this.target.id === 2 ? (params.maxAcc > 20 ? 0.38 : 0.24) : (params.maxAcc > 20 ? 0.40 : 0.30))) * adaptiveEnvelope.angleScale;
      const captureStartSpeed = (this.target.id === 3 ? (params.maxAcc > 20 ? 6.20 : 4.30) : (this.target.id === 2 ? (params.maxAcc > 20 ? 2.30 : 1.32) : (params.maxAcc > 20 ? 2.15 : 1.65))) * adaptiveEnvelope.speedScale;
      if (near.angleNorm < captureStartAngle && near.speedNorm < captureStartSpeed) this.localCaptureActive = true;
      if (near.angleNorm > (this.target.id === 3 ? 2.65 : 2.30) || near.speedNorm > (this.target.id === 3 ? 14.0 : 12.5)) this.localCaptureActive = false;
      const lqrAngleLimit = this.target.id === 3 && this.localCaptureActive ? 2.10 : (this.target.id === 3 ? (params.maxAcc > 20 ? 0.86 : 0.64) : (this.target.id === 2 ? (params.maxAcc > 20 ? 1.08 : 0.96) : (params.maxAcc > 20 ? 0.72 : 0.62)));
      const lqrSpeedLimit = (this.target.id === 3 && this.localCaptureActive ? (params.maxAcc > 20 ? 15.0 : 13.0) : (this.target.id === 3 ? (params.maxAcc > 20 ? 8.4 : 5.6) : (this.target.id === 2 ? (params.maxAcc > 20 ? 8.5 : 7.4) : (params.maxAcc > 20 ? 5.5 : 4.7)))) * adaptiveEnvelope.lqrSpeedScale;
      const useLqr = near.angleNorm < lqrAngleLimit && near.speedNorm < lqrSpeedLimit;
      if (this.retryTimer > 0) {
        raw = retryPreparationAcceleration(state, this.target, params, this.retryDirection);
        this.plan = [];
      } else if (useLqr) {
        raw = robustLocalAcceleration(state, this.target, params, this.lqrAcceleration(state, params));
        this.plan = [];
      } else {
        const state3SearchMode = this.state3WideLeftSearchTimer > 0 ? 2 : (this.state3InitialEdgeBoostTimer > 0 ? 1 : 0);
        const planned = chooseCemAcceleration(state, this.target, params, t, this.plan, state3SearchMode);
        raw = planned.acceleration;
        this.plan = planned.plan;
      }
    }

    if (!escapeActive) {
      raw = capturePolishAcceleration(state, this.target, params, t, raw, preNear);
    }

    if (specialState3AIActive && this.aiProfile) {
      const aiFeatures = phaseFeaturesForLearning(state, this.target, params, preNear);
      const aiAdvice = adaptiveAI.getPhaseAdvice(this.target.id, params, aiFeatures);
      // v5 actual-success replay integration:
      // Actual controller rollouts showed that the learned phase rules improve
      // state-3 settling, while the old broad learned brake/edge overlays were
      // too conservative and often pushed already-captured states back out of
      // the strict stable basin.  Keep the learned phase policy as the main AI
      // adapter and reserve rail protection for the hard endpoint guard below.
      raw = learnedPhasePolicyAcceleration(state, raw, this.target, params, preNear, this.aiProfile, aiAdvice);
    }

    let terminalEdgeCapture = false;
    if (this.target.id === 3) {
      const risk = sameSideBoundaryRisk(state, raw, this.target, params, preNear);
      const edgeCaptureStart = params.maxAcc > 20 ? 0.90 : 0.78;
      terminalEdgeCapture = preNear.angleNorm < (params.maxAcc > 20 ? 0.52 : 0.40) && preNear.speedNorm < (params.maxAcc > 20 ? 4.5 : 3.0) && risk.conflict && risk.edgeRatio > edgeCaptureStart;
      if (terminalEdgeCapture) {
        const centerA = centerReturnAcceleration(state, params, params.maxAcc > 28 ? 1.22 : 1.12);
        const edgeNeed = clamp((Math.max(risk.edgeRatio, risk.predictedRatio) - edgeCaptureStart) / Math.max(0.06, 0.99 - edgeCaptureStart), 0, 1);
        const quiet = clamp(1 - (preNear.angleNorm / (params.maxAcc > 20 ? 0.52 : 0.40) + preNear.speedNorm / (params.maxAcc > 20 ? 4.5 : 3.0)) * 0.5, 0, 1);
        const centerMix = clamp(0.18 + 0.22 * edgeNeed + 0.05 * quiet, 0.18, 0.48);
        const cap = Math.max(Math.min(params.maxAcc, 5.2), Math.min(params.maxAcc, 6.2 + 3.4 * preNear.angleNorm + 1.15 * preNear.speedNorm));
        raw = clamp((1 - centerMix) * raw + centerMix * centerA, -cap, cap);
      }
    } else if (this.target.id === 1 || this.target.id === 2) {
      const risk = sameSideBoundaryRisk(state, raw, this.target, params, preNear);
      terminalEdgeCapture = preNear.angleNorm < 0.28 && preNear.speedNorm < 2.25 && risk.conflict && risk.edgeRatio > 0.80;
      if (terminalEdgeCapture) {
        const centerA = centerReturnAcceleration(state, params, params.maxAcc > 24 ? 1.24 : 1.12);
        const edgeNeed = clamp((Math.max(risk.edgeRatio, risk.predictedRatio) - 0.80) / 0.18, 0, 1);
        const quiet = clamp(1 - (preNear.angleNorm / 0.28 + preNear.speedNorm / 2.25) * 0.5, 0, 1);
        const centerMix = clamp(0.22 + 0.24 * edgeNeed + 0.05 * quiet, 0.22, 0.56);
        const cap = planningAuthority(state, this.target, params, preNear.angleNorm, preNear.speedNorm);
        raw = clamp((1 - centerMix) * raw + centerMix * centerA, -cap, cap);
      }
    }


    if (this.target.id === 3 && preNear.angleNorm < 0.30 && preNear.speedNorm < 0.72) {
      const risk = sameSideBoundaryRisk(state, raw, this.target, params, preNear, 0.44);
      if (params.maxAcc > 20 && risk.conflict && risk.edgeRatio > 0.80 && risk.predictedRatio > 0.90) {
        // Only soften a near-upright command when that command is predicted to
        // keep driving into the same-side rail.  Otherwise preserve the strong
        // local balancer; this fixes the earlier premature weakening near edges.
        raw = clamp(0.62 * raw, -5.2, 5.2);
        terminalEdgeCapture = false;
      }
    }


    if (specialState3AIActive) {
      const edgeRatioNow = supportEdgeRatio(state, params);
      const xErrNow = supportCenterError(state, params);
      const movingOutwardNow = xErrNow * state.vx > 0;
      if (edgeRatioNow > 0.905 && movingOutwardNow) {
        const profile = this.aiProfile || {};
        const side = supportOutwardSign(state, params);
        const edgeNeed = clamp((edgeRatioNow - 0.895) / 0.09, 0, 1);
        const centerA = centerReturnAcceleration(state, params, 1.04 * clamp(profile.centerBias || 1, 0.9, 1.35));
        const inwardA = clamp(centerA - side * params.maxAcc * (0.038 + 0.078 * edgeNeed) * clamp(profile.edgeReserve || 1, 0.9, 1.45) - 0.20 * state.vx, -params.maxAcc, params.maxAcc);
        const mix = clamp(0.05 + 0.15 * edgeNeed, 0.05, 0.20);
        const cap = planningAuthority(state, this.target, params, preNear.angleNorm, preNear.speedNorm);
        raw = clamp((1 - mix) * raw + mix * inwardA, -cap, cap);
      }
    }
    raw = predictiveLandingGuardAcceleration(state, raw, this.target, params, preNear);
    raw = applyTrackSafety(state, raw, params);

    if (terminalEdgeCapture) {
      this.commandAcc = clamp(raw, -params.maxAcc, params.maxAcc);
      if (Math.abs(this.commandAcc) < 1e-5) this.commandAcc = 0;
      if (this.target.id !== 0) this.prevNear = preNear;
      return this.commandAcc;
    }

    // Slew-limited actuator smoothing keeps motion physical while still obeying acceleration-only control.
    const maxDelta = 8.0;
    const blended = 0.96 * raw + 0.04 * this.commandAcc;
    this.commandAcc = clamp(blended, this.commandAcc - maxDelta, this.commandAcc + maxDelta);
    this.commandAcc = applyTrackSafety(state, clamp(this.commandAcc, -params.maxAcc, params.maxAcc), params);
    if (Math.abs(this.commandAcc) < 1e-5) this.commandAcc = 0;
    if (this.target.id !== 0) this.prevNear = preNear;
    return this.commandAcc;
  }
}
