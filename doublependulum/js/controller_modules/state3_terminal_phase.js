import {
  TARGETS,
  angleError,
  clamp,
  copyStateInto,
  energyAtTarget,
  stepRK4Into,
  supportCenterError,
  supportEdgeRatio,
  supportHalfSpan,
  totalEnergy
} from "../physics.js";


function authorityAtOrBelow(params, level) {
  return 1 - clamp(((params.maxAcc || 0) - (level + 0.001)) / 0.001, 0, 1);
}

function nearUpright(state) {
  const e1 = angleError(state.th1, Math.PI);
  const e2 = angleError(state.th2, Math.PI);
  return { angleNorm: Math.hypot(e1, e2), speedNorm: Math.hypot(state.om1, state.om2) };
}

function radialVelocityToUpright(state) {
  const e1 = angleError(state.th1, Math.PI);
  const e2 = angleError(state.th2, Math.PI);
  const n = Math.hypot(e1, e2);
  return n < 1e-6 ? 0 : (e1 * state.om1 + e2 * state.om2) / n;
}

function supportPhasePower(state, params) {
  return (params.m1 + params.m2) * params.l1 * state.om1 * Math.cos(state.th1) +
    params.m2 * params.l2 * state.om2 * Math.cos(state.th2);
}

function terminalSupportMetrics(state, params, lqrA = 0) {
  const half = supportHalfSpan(params);
  const xErr = supportCenterError(state, params);
  const vx = state.vx;
  const maxA = Math.max(1, params.maxAcc || 1);
  const brakeAuthority = Math.max(2.4, Math.min(maxA, 0.42 * maxA));
  const stopErr = xErr + Math.sign(vx || xErr || 1) * vx * vx / Math.max(1.0, 2.0 * brakeAuthority);
  const stopRatio = Math.abs(stopErr) / Math.max(0.01, half);
  const edgeRatio = supportEdgeRatio(state, params);
  const movingOutward = xErr * vx > 0;
  const lqrOutward = xErr * lqrA > 0 && Math.abs(lqrA) > Math.max(2.0, 0.28 * maxA);
  return { half, xErr, vx, stopErr, stopRatio, edgeRatio, movingOutward, lqrOutward };
}

/**
 * Tests whether the current upright approach is safe to hand to the local
 * stabilizer.  It rejects the classic bad terminal phase: the links are near
 * upright, but the LQR action would accelerate the cart further toward the same
 * rail while it still has outward velocity or insufficient stopping reserve.
 */
export function state3TerminalPhaseReady(state, params, near, lqrA = 0) {
  const m = terminalSupportMetrics(state, params, lqrA);
  const radial = radialVelocityToUpright(state);
  const maxA = Math.max(1, params.maxAcc || 1);
  const strictStop = m.lqrOutward ? 0.19 : 0.36;
  const fastOutward = m.movingOutward && Math.abs(m.vx) > Math.max(0.55, 0.055 * maxA);
  const flyby = radial > 0.34 && near.speedNorm > 1.6;
  if (m.edgeRatio > 0.84) return false;
  if (m.stopRatio > strictStop) return false;
  if (m.lqrOutward && (fastOutward || m.edgeRatio > 0.16 || flyby)) return false;
  if (fastOutward && m.stopRatio > 0.24 && near.angleNorm > 0.30) return false;
  return true;
}

function terminalLandingCost(state, params, lqrA = 0) {
  const near = nearUpright(state);
  const m = terminalSupportMetrics(state, params, lqrA);
  const radial = radialVelocityToUpright(state);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const dE = (totalEnergy(state, params) - energyAtTarget(TARGETS[3], params)) / energyScale;
  const angle = near.angleNorm;
  const speed = near.speedNorm;
  const xNorm = m.xErr / Math.max(0.01, m.half);
  const vxNorm = m.vx / Math.max(0.7, 0.13 * Math.max(1, params.maxAcc || 1));
  const edgeSoft = Math.max(0, m.edgeRatio - 0.70);
  const outward = m.movingOutward ? Math.abs(vxNorm) : 0;
  const lqrConflict = m.lqrOutward ? 1 : 0;
  const readyBonus = state3TerminalPhaseReady(state, params, near, lqrA) && angle < 0.56 && speed < 3.9 ? -2.25 : 0;
  return readyBonus +
    19.0 * angle * angle +
    2.35 * speed * speed +
    3.15 * xNorm * xNorm +
    1.80 * vxNorm * vxNorm +
    16.0 * m.stopRatio * m.stopRatio +
    0.95 * Math.max(0, dE - 0.03) * Math.max(0, dE - 0.03) +
    1.80 * Math.max(0, radial) * Math.max(0, radial) +
    2.10 * outward * outward +
    1.55 * lqrConflict * (m.stopRatio + Math.max(0, m.edgeRatio - 0.12)) +
    36.0 * edgeSoft * edgeSoft;
}

/**
 * Terminal phase planner for State 3.  It is active before local capture, when
 * the pendulums are already approaching upright.  Rather than minimizing only
 * angular error, it enumerates short two-stage support commands and scores the
 * best predicted handoff state in (x, vx, theta, theta_dot).  This makes the
 * first capture more likely to enter the LQR basin with braking reserve.
 */
export function state3TerminalPhasePlannerAcceleration(state, raw, params, t, near, lqrAcceleration = null) {
  if (!near || near.angleNorm > 1.12 || near.speedNorm > 7.2) return raw;
  if (near.angleNorm < 0.20 && near.speedNorm < 0.80) return raw;
  const maxA = Math.max(1, params.maxAcc || 1);
  const lqrA0 = typeof lqrAcceleration === "function" ? lqrAcceleration(state, TARGETS[3], params) : 0;
  const metrics = terminalSupportMetrics(state, params, lqrA0);
  const energyScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
  const dE = (totalEnergy(state, params) - energyAtTarget(TARGETS[3], params)) / energyScale;

  // Keep this planner surgical.  The old controller is already good when the
  // cart is nearly centered with modest velocity.  The planner only takes over
  // for a genuine terminal-phase mismatch: useful angular arrival but support
  // position/velocity would make the first LQR handoff burn workspace.
  const absX = Math.abs(metrics.xErr);
  const absV = Math.abs(metrics.vx);
  const directionalPhaseMismatch = (metrics.xErr < 0 && metrics.vx > 0) ? absX > 1.25 : absX > 0.22;
  const phaseMismatch = near.angleNorm < 0.86 && directionalPhaseMismatch && absX < 1.86 && absV > 1.00;
  const outwardReserveMismatch = near.angleNorm < 0.92 && metrics.movingOutward && metrics.stopRatio > 0.18 && absV > 0.80 && !(metrics.xErr < 0 && metrics.vx > 0 && absX < 1.25);
  const lqrConflictMismatch = near.angleNorm < 0.82 && metrics.lqrOutward && metrics.stopRatio > 0.10 && absX > 0.18 && !(metrics.xErr < 0 && metrics.vx > 0 && absX < 1.25);
  const energyFlybyMismatch = near.angleNorm < 0.76 && dE > 0.16 && absX > 0.22 && absV > 0.80 && !(metrics.xErr < 0 && metrics.vx > 0 && absX < 1.25);
  if (!(phaseMismatch || outwardReserveMismatch || lqrConflictMismatch || energyFlybyMismatch)) return raw;

  const baseEngage = clamp((0.95 - near.angleNorm) / 0.50, 0, 1) * clamp((7.2 - near.speedNorm) / 5.4, 0, 1);
  const engage = Math.max(0.35 * baseEngage, phaseMismatch ? 0.62 : 0, outwardReserveMismatch ? 0.58 : 0, lqrConflictMismatch ? 0.66 : 0, energyFlybyMismatch ? 0.52 : 0);
  if (engage <= 0.02) return raw;

  const cap = Math.min(maxA, Math.max(5.0, authorityAtOrBelow(params, 20) >= 0.5 ? maxA : 0.58 * maxA));
  const centerA = clamp(-1.25 * metrics.xErr - 1.65 * state.vx, -0.44 * cap, 0.44 * cap);
  const stopA = clamp(-1.65 * metrics.stopErr - 1.35 * state.vx, -0.55 * cap, 0.55 * cap);
  const phaseA = clamp(4.7 * supportPhasePower(state, params), -0.42 * cap, 0.42 * cap);
  const lqrA = clamp(lqrA0, -cap, cap);
  const antiLqrOutward = metrics.lqrOutward ? clamp(centerA - Math.sign(metrics.xErr || 1) * 0.18 * cap, -cap, cap) : centerA;
  const firstValues = [
    raw,
    0.72 * raw,
    0.42 * raw + 0.58 * centerA,
    0.35 * raw + 0.65 * stopA,
    centerA,
    stopA,
    phaseA,
    0.50 * phaseA + 0.50 * centerA,
    0.40 * phaseA + 0.35 * stopA + 0.25 * lqrA,
    lqrA,
    antiLqrOutward,
    -0.45 * lqrA + 0.70 * centerA
  ].map(a => clamp(a, -cap, cap));
  const followValues = [
    centerA,
    stopA,
    0.55 * centerA + 0.45 * stopA,
    0.55 * lqrA + 0.45 * centerA,
    0.45 * phaseA + 0.55 * stopA,
    0
  ].map(a => clamp(a, -cap, cap));

  const ws = state3TerminalPhasePlannerAcceleration._ws || (state3TerminalPhasePlannerAcceleration._ws = {
    s: {}, next: {}, scratch1: {}, scratch2: {}, scratch3: {}, k1: {}, k2: {}, k3: {}, k4: {}
  });
  const horizon = 0.88;
  const dt = 1 / 90;
  const switchT = 0.28;
  let bestA = clamp(raw, -cap, cap);
  let bestCost = Infinity;
  let baseCost = Infinity;
  let idx = 0;

  for (const firstA of firstValues) {
    for (const followA of followValues) {
      let s = copyStateInto(state, ws.s);
      let next = ws.next;
      let tt = 0;
      let minLanding = Infinity;
      let running = 0;
      while (tt < horizon - 1e-12) {
        const blend = clamp((tt - switchT) / Math.max(1e-6, horizon - switchT), 0, 1);
        const a = clamp((1 - blend) * firstA + blend * followA, -cap, cap);
        stepRK4Into(s, a, params, t + tt, dt, true, next, ws.scratch1, ws.scratch2, ws.scratch3, ws.k1, ws.k2, ws.k3, ws.k4);
        const old = s; s = next; next = old;
        const n = nearUpright(s);
        const tailGate = clamp((tt - 0.22) / 0.45, 0, 1);
        const landing = terminalLandingCost(s, params, lqrA);
        minLanding = Math.min(minLanding, landing + 0.20 * Math.max(0, n.angleNorm - near.angleNorm));
        running += tailGate * (0.025 * landing + 0.0009 * a * a / Math.max(1, cap * cap));
        tt += dt;
      }
      const finalCost = terminalLandingCost(s, params, lqrA);
      const effort = 0.0025 * firstA * firstA / Math.max(1, cap * cap) + 0.0008 * Math.abs(firstA - clamp(raw, -cap, cap));
      const cost = 0.72 * minLanding + 0.28 * finalCost + running + effort;
      if (idx === 0) baseCost = cost;
      if (cost < bestCost) { bestCost = cost; bestA = firstA; }
      idx += 1;
    }
  }

  const required = Math.max(0.035 * Math.max(1, Math.abs(baseCost)), metrics.lqrOutward ? 0.05 : 0.16);
  if (baseCost - bestCost <= required) return raw;
  const mix = clamp(0.38 + 0.34 * engage + 0.12 * (metrics.lqrOutward ? 1 : 0), 0.42, 0.82);
  return clamp((1 - mix) * raw + mix * bestA, -cap, cap);
}
