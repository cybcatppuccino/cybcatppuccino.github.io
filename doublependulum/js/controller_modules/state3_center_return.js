import {
  TARGETS,
  angleError,
  clamp,
  copyStateInto,
  stepRK4Into,
  supportCenterError
} from "../physics.js";

function nearUpright(state) {
  const e1 = angleError(state.th1, TARGETS[3].angles[0]);
  const e2 = angleError(state.th2, TARGETS[3].angles[1]);
  return { angleNorm: Math.hypot(e1, e2), speedNorm: Math.hypot(state.om1, state.om2) };
}

/**
 * Balance-preserving center-return module.
 * Engages only after State 3 is quiet; uses a short-horizon candidate search so
 * the cart can brake/return without re-opening the swing-up/capture problem.
 */
export function state3CenterReturnBalanceAcceleration(state, raw, params, near, phaseName = "hold", phaseTimer = 0, localBalance = null) {
  if (near.angleNorm > 0.135 || near.speedNorm > 0.52) return raw;
  const xErr = supportCenterError(state, params);
  const absX = Math.abs(xErr);
  const absV = Math.abs(state.vx);
  if (absX < 0.17 && absV < 0.075) return raw;

  const maxA = Math.max(1, params.maxAcc || 1);
  const quiet = clamp(1 - (near.angleNorm / 0.135 + near.speedNorm / 0.52) * 0.5, 0, 1);
  const phaseGate = phaseName === "center_return" || phaseName === "hold" ? 1 : clamp((phaseTimer - 0.55) / 0.75, 0, 1);
  const engage = quiet * phaseGate;
  if (engage <= 0.03) return raw;

  const dirToCenter = absX > 1e-6 ? -Math.sign(xErr) : 0;
  const brakingComfort = Math.max(0.32, Math.min(0.95, 0.045 * maxA));
  const stopLimitedV = Math.sqrt(Math.max(0, 2 * brakingComfort * Math.max(0, absX - 0.06)));
  const vCruise = Math.min(0.22, Math.max(0.14, 0.010 * maxA));
  const vRef = dirToCenter * Math.min(vCruise, stopLimitedV);

  const centerA = clamp(-1.75 * xErr - 2.35 * state.vx, -0.38 * maxA, 0.38 * maxA);
  const velocityA = clamp(2.10 * (vRef - state.vx) - 0.22 * xErr, -0.36 * maxA, 0.36 * maxA);
  const lqrA = typeof localBalance === "function" ? localBalance(state, TARGETS[3], params) : raw;
  const cap = Math.min(maxA, Math.max(3.4, 0.42 * maxA));
  const candidates = [
    raw,
    centerA,
    velocityA,
    0.70 * raw + 0.30 * centerA,
    0.55 * raw + 0.45 * velocityA,
    0.64 * lqrA + 0.36 * centerA,
    0.58 * lqrA + 0.42 * velocityA
  ];

  const horizon = 0.52;
  const dt = 1 / 120;
  const scratch1 = {}, scratch2 = {}, scratch3 = {}, k1 = {}, k2 = {}, k3 = {}, k4 = {};
  let bestA = raw, bestScore = Infinity;
  let trial = {}, next = {};
  for (const c of candidates) {
    const a = clamp(c, -cap, cap);
    copyStateInto(state, trial);
    let tt = 0;
    let maxAngle = near.angleNorm;
    let maxSpeed = near.speedNorm;
    while (tt < horizon - 1e-12) {
      stepRK4Into(trial, a, params, tt, dt, true, next, scratch1, scratch2, scratch3, k1, k2, k3, k4);
      const n = nearUpright(next);
      maxAngle = Math.max(maxAngle, n.angleNorm);
      maxSpeed = Math.max(maxSpeed, n.speedNorm);
      const tmp = trial; trial = next; next = tmp;
      tt += dt;
    }
    const nFinal = nearUpright(trial);
    const xFinal = supportCenterError(trial, params);
    const crossingFast = xErr * xFinal < 0 && Math.abs(trial.vx) > 0.12;
    const score =
      34.0 * Math.max(0, maxAngle - 0.155) +
      7.5 * Math.max(0, maxSpeed - 0.70) +
      8.0 * Math.max(0, nFinal.angleNorm - near.angleNorm - 0.025) +
      2.0 * Math.max(0, nFinal.speedNorm - near.speedNorm - 0.075) +
      1.95 * Math.abs(xFinal) +
      1.65 * Math.abs(trial.vx - vRef) +
      0.55 * Math.abs(trial.vx) +
      (crossingFast ? 1.2 : 0) +
      0.035 * Math.abs(a - raw);
    if (score < bestScore) { bestScore = score; bestA = a; }
  }

  const mix = clamp(0.18 + 0.48 * engage + 0.10 * clamp(absX / 0.50, 0, 1), 0.18, 0.72);
  return clamp((1 - mix) * raw + mix * bestA, -cap, cap);
}
