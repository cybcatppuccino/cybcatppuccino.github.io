export const DEFAULT_PARAMS = Object.freeze({
  m1: 1.0,
  m2: 1.0,
  l1: 1.0,
  l2: 1.0,
  g: 9.81,
  maxAcc: 15.0,
  windAmp: 0.0,
  friction: 0.03,
  // segmentHalfLength is the half length of the original symmetric rail.
  // The effective rail extends the right endpoint by 1/4 of the original
  // full segment length: [-H, H + 0.25*(2H)] = [-H, 1.5H].
  segmentHalfLength: 3.20,
  rightSegmentExtensionFraction: 0.25,
  topY: 0.0
});

export const TARGETS = Object.freeze([
  { id: 0, name: "State 0 · Down", angles: [0, 0] },
  { id: 1, name: "State 1 · V", angles: [0, Math.PI] },
  { id: 2, name: "State 2 · Lambda", angles: [Math.PI, 0] },
  { id: 3, name: "State 3 · Upright", angles: [Math.PI, Math.PI] }
]);

export function supportBounds(params) {
  const baseHalf = Math.max(0.01, params.segmentHalfLength);
  const extensionFraction = Math.max(0, params.rightSegmentExtensionFraction ?? 0.25);
  const left = -baseHalf;
  const right = baseHalf + 2 * baseHalf * extensionFraction;
  const center = 0.5 * (left + right);
  const half = 0.5 * (right - left);
  return { left, right, center, half };
}

export function supportLeft(params) {
  return supportBounds(params).left;
}

export function supportRight(params) {
  return supportBounds(params).right;
}

export function supportCenter(params) {
  return supportBounds(params).center;
}

export function supportHalfSpan(params) {
  return supportBounds(params).half;
}

export function supportCenterError(state, params) {
  return state.x - supportCenter(params);
}

export function supportEdgeRatio(state, params) {
  const bounds = supportBounds(params);
  return Math.abs(state.x - bounds.center) / Math.max(0.01, bounds.half);
}

export function supportOutwardSign(state, params) {
  const bounds = supportBounds(params);
  const xErr = state.x - bounds.center;
  if (Math.abs(xErr) > 1e-6) return Math.sign(xErr);
  if (Math.abs(state.vx) > 1e-6) return Math.sign(state.vx);
  return 1;
}

export function makeInitialState() {
  // Start at the midpoint of the current effective rail.  With the v2 asymmetric
  // extension this is x = 0.8 for the default [-3.2, 4.8] segment, not the old
  // symmetric-segment origin x = 0.
  return {
    x: supportCenter(DEFAULT_PARAMS),
    vx: 0,
    th1: 0,
    th2: 0,
    om1: 0,
    om2: 0
  };
}

export function cloneState(s) {
  return {
    x: s.x,
    vx: s.vx,
    th1: s.th1,
    th2: s.th2,
    om1: s.om1,
    om2: s.om2
  };
}

export function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

export function wrapAngle(angle) {
  let a = (angle + Math.PI) % (2 * Math.PI);
  if (a < 0) a += 2 * Math.PI;
  return a - Math.PI;
}

export function wrapUnit(value) {
  let v = value % 1;
  if (v < 0) v += 1;
  return v;
}

export function angleToPhaseCoord(angle) {
  // Downward angle 0 maps to 1/4; upright angle pi maps to 3/4.
  return wrapUnit(angle / (2 * Math.PI) + 0.25);
}

export function phaseCoordToAngle(coord) {
  return wrapAngle((wrapUnit(coord) - 0.25) * 2 * Math.PI);
}

export function angleError(angle, target) {
  return wrapAngle(angle - target);
}

export function targetError(state, target) {
  return [
    angleError(state.th1, target.angles[0]),
    angleError(state.th2, target.angles[1])
  ];
}

export function windAcceleration(t, params) {
  const a = params.windAmp || 0;
  if (a === 0) return 0;

  // Smooth zero-mean field. Each component is sinusoidal, so the long-run mean is zero.
  const raw =
    0.56 * Math.sin(0.73 * t + 0.6) +
    0.31 * Math.sin(1.37 * t + 2.1) +
    0.13 * Math.sin(2.11 * t + 4.2);

  return a * raw;
}

function constrainSupportAcceleration(state, commandAcc, params) {
  const a = clamp(commandAcc, -params.maxAcc, params.maxAcc);
  const { left, right } = supportBounds(params);

  if (state.x <= left && state.vx <= 0 && a < 0) return 0;
  if (state.x >= right && state.vx >= 0 && a > 0) return 0;
  return a;
}

export function derivative(state, commandAcc, params, t, useWind = true) {
  return derivativeInto(state, commandAcc, params, t, useWind, {});
}

export function derivativeInto(state, commandAcc, params, t, useWind = true, out = {}) {
  const aBase = constrainSupportAcceleration(state, commandAcc, params);
  const aWind = useWind ? windAcceleration(t, params) : 0;

  // The wind acts as a horizontal body acceleration on both point masses.
  // In generalized coordinates this is equivalent to replacing x_pivot_ddot by x_pivot_ddot - a_wind.
  const effectiveHorizontalAcc = aBase - aWind;

  const { m1, m2, l1, l2, g } = params;
  const { th1, th2, om1, om2 } = state;
  const delta = th1 - th2;
  const cosDelta = Math.cos(delta);
  const sinDelta = Math.sin(delta);

  const M11 = (m1 + m2) * l1 * l1;
  const M12 = m2 * l1 * l2 * cosDelta;
  const M22 = m2 * l2 * l2;
  const det = M11 * M22 - M12 * M12;

  let rhs1 =
    -m2 * l1 * l2 * sinDelta * om2 * om2 -
    (m1 + m2) * g * l1 * Math.sin(th1) -
    (m1 + m2) * l1 * Math.cos(th1) * effectiveHorizontalAcc;

  let rhs2 =
    m2 * l1 * l2 * sinDelta * om1 * om1 -
    m2 * g * l2 * Math.sin(th2) -
    m2 * l2 * Math.cos(th2) * effectiveHorizontalAcc;

  const friction = Math.max(0, params.friction || 0);
  if (friction > 0) {
    // Viscous joint damping. The slider itself is still acceleration-controlled;
    // this only removes mechanical energy from the pendulum joints.
    const c1 = 0.34 * friction * (m1 + m2) * l1 * l1;
    const c2 = 0.28 * friction * m2 * l2 * l2;
    rhs1 -= c1 * om1;
    rhs2 -= c2 * om2;
  }

  const dom1 = (rhs1 * M22 - rhs2 * M12) / det;
  const dom2 = (M11 * rhs2 - M12 * rhs1) / det;

  out.x = state.vx;
  out.vx = aBase;
  out.th1 = om1;
  out.th2 = om2;
  out.om1 = dom1;
  out.om2 = dom2;
  return out;
}

function addScaledState(base, d, scale) {
  return {
    x: base.x + d.x * scale,
    vx: base.vx + d.vx * scale,
    th1: base.th1 + d.th1 * scale,
    th2: base.th2 + d.th2 * scale,
    om1: base.om1 + d.om1 * scale,
    om2: base.om2 + d.om2 * scale
  };
}

function addScaledStateInto(base, d, scale, out) {
  out.x = base.x + d.x * scale;
  out.vx = base.vx + d.vx * scale;
  out.th1 = base.th1 + d.th1 * scale;
  out.th2 = base.th2 + d.th2 * scale;
  out.om1 = base.om1 + d.om1 * scale;
  out.om2 = base.om2 + d.om2 * scale;
  return out;
}

function combineRK4(state, k1, k2, k3, k4, dt) {
  return {
    x: state.x + (dt / 6) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
    vx: state.vx + (dt / 6) * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx),
    th1: state.th1 + (dt / 6) * (k1.th1 + 2 * k2.th1 + 2 * k3.th1 + k4.th1),
    th2: state.th2 + (dt / 6) * (k1.th2 + 2 * k2.th2 + 2 * k3.th2 + k4.th2),
    om1: state.om1 + (dt / 6) * (k1.om1 + 2 * k2.om1 + 2 * k3.om1 + k4.om1),
    om2: state.om2 + (dt / 6) * (k1.om2 + 2 * k2.om2 + 2 * k3.om2 + k4.om2)
  };
}

function applySupportStop(state, params) {
  const { left, right } = supportBounds(params);
  if (state.x < left) {
    state.x = left;
    state.vx = 0;
  } else if (state.x > right) {
    state.x = right;
    state.vx = 0;
  }
  return state;
}

export function stepRK4(state, commandAcc, params, t, dt, useWind = true) {
  return stepRK4Into(state, commandAcc, params, t, dt, useWind, {}, {}, {}, {}, {}, {}, {});
}

export function stepRK4Into(state, commandAcc, params, t, dt, useWind = true, out = {}, scratch1 = {}, scratch2 = {}, scratch3 = {}, k1 = {}, k2 = {}, k3 = {}, k4 = {}) {
  derivativeInto(state, commandAcc, params, t, useWind, k1);
  addScaledStateInto(state, k1, dt * 0.5, scratch1);
  derivativeInto(scratch1, commandAcc, params, t + dt * 0.5, useWind, k2);
  addScaledStateInto(state, k2, dt * 0.5, scratch2);
  derivativeInto(scratch2, commandAcc, params, t + dt * 0.5, useWind, k3);
  addScaledStateInto(state, k3, dt, scratch3);
  derivativeInto(scratch3, commandAcc, params, t + dt, useWind, k4);

  const w = dt / 6;
  out.x = state.x + w * (k1.x + 2 * k2.x + k3.x * 2 + k4.x);
  out.vx = state.vx + w * (k1.vx + 2 * k2.vx + k3.vx * 2 + k4.vx);
  out.th1 = wrapAngle(state.th1 + w * (k1.th1 + 2 * k2.th1 + k3.th1 * 2 + k4.th1));
  out.th2 = wrapAngle(state.th2 + w * (k1.th2 + 2 * k2.th2 + k3.th2 * 2 + k4.th2));
  out.om1 = state.om1 + w * (k1.om1 + 2 * k2.om1 + k3.om1 * 2 + k4.om1);
  out.om2 = state.om2 + w * (k1.om2 + 2 * k2.om2 + k3.om2 * 2 + k4.om2);
  return applySupportStop(out, params);
}

export function stepSemiImplicit(state, commandAcc, params, t, dt, useWind = false) {
  const d = derivative(state, commandAcc, params, t, useWind);
  const next = {
    vx: state.vx + d.vx * dt,
    om1: state.om1 + d.om1 * dt,
    om2: state.om2 + d.om2 * dt,
    x: state.x,
    th1: state.th1,
    th2: state.th2
  };
  next.x += next.vx * dt;
  next.th1 = wrapAngle(next.th1 + next.om1 * dt);
  next.th2 = wrapAngle(next.th2 + next.om2 * dt);
  return applySupportStop(next, params);
}

export function pointPositions(state, params) {
  const x0 = state.x;
  const y0 = params.topY;
  const x1 = x0 + params.l1 * Math.sin(state.th1);
  const y1 = y0 + params.l1 * Math.cos(state.th1);
  const x2 = x1 + params.l2 * Math.sin(state.th2);
  const y2 = y1 + params.l2 * Math.cos(state.th2);
  return { x0, y0, x1, y1, x2, y2 };
}

export function totalEnergy(state, params) {
  const { m1, m2, l1, l2, g } = params;
  const v1x = state.vx + l1 * Math.cos(state.th1) * state.om1;
  const v1y = -l1 * Math.sin(state.th1) * state.om1;
  const v2x = state.vx + l1 * Math.cos(state.th1) * state.om1 + l2 * Math.cos(state.th2) * state.om2;
  const v2y = -l1 * Math.sin(state.th1) * state.om1 - l2 * Math.sin(state.th2) * state.om2;
  const kinetic = 0.5 * m1 * (v1x * v1x + v1y * v1y) + 0.5 * m2 * (v2x * v2x + v2y * v2y);
  const potential = -(m1 + m2) * g * l1 * Math.cos(state.th1) - m2 * g * l2 * Math.cos(state.th2);
  return kinetic + potential;
}

export function energyAtTarget(target, params) {
  return totalEnergy({
    x: 0,
    vx: 0,
    th1: target.angles[0],
    th2: target.angles[1],
    om1: 0,
    om2: 0
  }, params);
}
