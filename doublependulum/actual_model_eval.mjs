import fs from 'node:fs';
import { DEFAULT_PARAMS, TARGETS, angleError, stepRK4, supportBounds, supportCenter, totalEnergy, energyAtTarget } from './js/physics.js';
import { PendulumController } from './js/controller.js';
import { adaptiveAI } from './js/ai_learning.js';
import { wasmRollout } from './js/wasm_rollout.js';

const TWO_PI = Math.PI * 2;
function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
function wrap(a) { a = (a + Math.PI) % TWO_PI; if (a < 0) a += TWO_PI; return a - Math.PI; }
function rngMulberry(seed) {
  let t = seed >>> 0;
  return () => {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}
function uniform(rng, a, b) { return a + (b - a) * rng(); }
function triangular(rng, lo, hi) { return (uniform(rng, lo, hi) + uniform(rng, lo, hi)) * 0.5; }
function closeness(s, target) {
  const e1 = angleError(s.th1, target.angles[0]);
  const e2 = angleError(s.th2, target.angles[1]);
  return { angleNorm: Math.hypot(e1, e2), speedNorm: Math.hypot(s.om1, s.om2), e1, e2 };
}
function kineticProxy(s, params) {
  const E = totalEnergy(s, params) - energyAtTarget(TARGETS[3], params);
  return Math.max(0, E);
}
function defaultParams(rng) {
  return {
    ...DEFAULT_PARAMS,
    maxAcc: Math.round(triangular(rng, 14.1, 25.9) * 2) / 2,
    g: Math.round(triangular(rng, 7.8, 10.2) * 100) / 100,
    windAmp: Math.round(triangular(rng, 0.0, 0.13) * 200) / 200,
    friction: Math.round(triangular(rng, 0.0, 0.13) * 200) / 200,
  };
}
function randomState(rng, params) {
  const rail = supportBounds(params);
  const modeRoll = rng();
  let x, vx, th1, th2, om1, om2;
  if (modeRoll < 0.28) { // broad random, moderate momentum
    x = uniform(rng, rail.center - 0.72 * rail.half, rail.center + 0.72 * rail.half);
    vx = uniform(rng, -1.45, 1.45);
    th1 = uniform(rng, -Math.PI, Math.PI);
    th2 = uniform(rng, -Math.PI, Math.PI);
    om1 = uniform(rng, -3.1, 3.1);
    om2 = uniform(rng, -3.1, 3.1);
  } else if (modeRoll < 0.54) { // near target 3 but too energetic
    x = uniform(rng, rail.center - 0.52 * rail.half, rail.center + 0.52 * rail.half);
    vx = uniform(rng, -1.05, 1.05);
    th1 = wrap(Math.PI + uniform(rng, -0.95, 0.95));
    th2 = wrap(Math.PI + uniform(rng, -0.95, 0.95));
    om1 = uniform(rng, -3.4, 3.4);
    om2 = uniform(rng, -3.4, 3.4);
  } else if (modeRoll < 0.76) { // from down-ish state, common browser reset/drag
    x = uniform(rng, rail.center - 0.58 * rail.half, rail.center + 0.58 * rail.half);
    vx = uniform(rng, -1.25, 1.25);
    th1 = uniform(rng, -0.65, 0.65);
    th2 = uniform(rng, -0.65, 0.65);
    om1 = uniform(rng, -2.2, 2.2);
    om2 = uniform(rng, -2.2, 2.2);
  } else { // rail recovery
    const side = rng() < 0.5 ? -1 : 1;
    x = rail.center + side * uniform(rng, 0.62 * rail.half, 0.91 * rail.half);
    vx = rng() < 0.70 ? side * uniform(rng, 0.15, 1.25) : uniform(rng, -0.85, 0.85);
    if (rng() < 0.55) {
      th1 = wrap(Math.PI + uniform(rng, -0.95, 0.95));
      th2 = wrap(Math.PI + uniform(rng, -0.95, 0.95));
    } else {
      th1 = uniform(rng, -Math.PI, Math.PI);
      th2 = uniform(rng, -Math.PI, Math.PI);
    }
    om1 = uniform(rng, -2.9, 2.9);
    om2 = uniform(rng, -2.9, 2.9);
  }
  return { x, vx, th1, th2, om1, om2 };
}
function runEpisode(rng, seconds = 15, dt = 1 / 360) {
  const params = defaultParams(rng);
  let state = randomState(rng, params);
  const controller = new PendulumController(params);
  controller.setTarget(3, state);
  const target = TARGETS[3];
  let t = 0;
  let stableDwell = 0;
  let captureDwell = 0;
  let firstCapture = null;
  let stableTime = null;
  let bestAngle = Infinity;
  let bestSpeed = Infinity;
  let maxEdge = 0;
  let edgeRisk = false;
  const rail = supportBounds(params);
  let command = 0;
  for (let i = 0; i < Math.floor(seconds / dt); i++) {
    const near = closeness(state, target);
    bestAngle = Math.min(bestAngle, near.angleNorm);
    bestSpeed = Math.min(bestSpeed, near.speedNorm);
    const edge = Math.abs(state.x - rail.center) / rail.half;
    maxEdge = Math.max(maxEdge, edge);
    if (edge > 0.965 && (state.x - rail.center) * state.vx > 0) edgeRisk = true;
    const eScale = Math.max(1, params.g * ((params.m1 + params.m2) * params.l1 + params.m2 * params.l2));
    const kineticSmall = kineticProxy(state, params) / eScale < 0.16;
    const stable = near.angleNorm < 0.34 && near.speedNorm < 1.20 && edge < 0.90 && kineticSmall;
    const capture = near.angleNorm < 0.72 && near.speedNorm < 3.60 && edge < 0.92;
    if (capture) {
      captureDwell += dt;
      if (captureDwell > 0.12 && firstCapture === null) firstCapture = t;
    } else {
      captureDwell = Math.max(0, captureDwell - 2 * dt);
    }
    if (stable) {
      stableDwell += dt;
      if (stableDwell > 0.36) { stableTime = t; break; }
    } else {
      stableDwell = Math.max(0, stableDwell - 3 * dt);
    }
    command = controller.update(state, params, dt, t);
    state = stepRK4(state, command, params, t, dt, true);
    t += dt;
    if (!Number.isFinite(state.x + state.vx + state.th1 + state.th2 + state.om1 + state.om2)) break;
  }
  let event = stableTime !== null ? 'stable' : (firstCapture !== null ? 'capture-only' : 'timeout');
  if (edgeRisk) event = stableTime !== null ? 'stable-edge-risk' : 'edge-risk';
  return { success: stableTime !== null, captured: firstCapture !== null || stableTime !== null, stableTime: stableTime ?? seconds, firstCapture: firstCapture ?? seconds, bestAngle, bestSpeed, maxEdge, event };
}
function summarize(results, seconds) {
  const n = results.length || 1;
  const succ = results.filter(r => r.success);
  const cap = results.filter(r => r.captured);
  const events = {};
  for (const r of results) events[r.event] = (events[r.event] || 0) + 1;
  const mean = arr => arr.reduce((a, b) => a + b, 0) / Math.max(1, arr.length);
  return {
    episodes: results.length,
    successRate: succ.length / n,
    captureRate: cap.length / n,
    meanStableTime: mean(succ.map(r => r.stableTime)).toFixed(3),
    meanFirstCaptureTime: mean(cap.map(r => r.firstCapture)).toFixed(3),
    meanBestAngle: mean(results.map(r => r.bestAngle)).toFixed(3),
    edgeRiskRate: (results.filter(r => r.event.includes('edge-risk')).length / n),
    events,
  };
}

const args = Object.fromEntries(process.argv.slice(2).map((x, i, arr) => x.startsWith('--') ? [x.slice(2), arr[i + 1] && !arr[i + 1].startsWith('--') ? arr[i + 1] : true] : [String(i), x]));
const dbPath = args.db || 'ai_data/training_db.json';
const episodes = Number(args.episodes || 80);
const seed = Number(args.seed || 40421);
const seconds = Number(args.seconds || 15);
const data = JSON.parse(fs.readFileSync(dbPath, 'utf-8'));
adaptiveAI.importDatabase(data, 'actual-eval');
try { await wasmRollout.ready; } catch {}
const rng = rngMulberry(seed);
const results = [];
for (let i = 0; i < episodes; i++) results.push(runEpisode(rng, seconds));
const summary = summarize(results, seconds);
console.log(JSON.stringify(summary, null, 2));
