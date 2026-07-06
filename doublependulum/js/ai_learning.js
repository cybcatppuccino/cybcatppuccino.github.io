// Lightweight adaptive memory for the pendulum controller.
//
// Important design rule: this module never changes player-selected physics
// parameters. It stores small internal controller multipliers and coarse
// phase-rule advice per parameter/range bucket.  The same database can be
// updated online in the browser or offline by train_ai.py.

const DB_NAME = "dp-adaptive-ai-v2";
const STORE_NAME = "profiles";
const PHASE_STORE_NAME = "phaseRules";
const LOCAL_KEY = "dp.adaptiveAi.profiles.v2";
const PHASE_LOCAL_KEY = "dp.adaptiveAi.phaseRules.v2";
const SAVE_DEBOUNCE_MS = 450;
const LEARNING_RATE = 0.58;
const STATIC_DB_URL = "ai_data/training_db.json";
// This package is configured for inference-only browser use: the browser loads
// the static training result and does not keep training / overriding it locally.
// Set both flags to true only when intentionally collecting online data.
const BROWSER_ONLINE_LEARNING_ENABLED = false;
const USE_BROWSER_LOCAL_AI_CACHE = false;

const PROFILE_FIELDS = [
  "captureConservatism",
  "brakeGain",
  "edgeReserve",
  "authorityScale",
  "centerBias",
  "retryBoost",
  "speedDamping",
  "landingGuard",
  "reserveBias"
];

function clamp01(x) {
  return Math.max(0, Math.min(1, x));
}

function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

function roundTo(value, step, fallback = 0) {
  const v = Number.isFinite(value) ? value : fallback;
  return Math.round(v / step) * step;
}

function nowMs() {
  return typeof performance !== "undefined" && performance.now ? performance.now() : Date.now();
}

function defaultProfile(params, targetId) {
  const maxA = Math.max(1, Number(params.maxAcc) || 1);
  const highA = clamp01((maxA - 20) / 40);
  const lowA = clamp01((18 - maxA) / 17);
  const friction = Math.max(0, Number(params.friction) || 0);
  const lowFriction = clamp01((0.045 - friction) / 0.045);
  const highFriction = clamp01((friction - 0.08) / 0.22);
  const wind = Math.abs(Number(params.windAmp) || 0);
  const windStress = clamp01(wind / 0.35);
  const g = Math.max(0, Number(params.g) || 0);
  const lightG = clamp01((8.6 - g) / 8.6);
  const heavyG = clamp01((g - 9.8) / 2.2);
  const upright = targetId === 3 ? 1 : 0;

  return {
    captureConservatism: clamp(1 + 0.56 * highA + 0.14 * lowFriction + 0.08 * windStress + 0.08 * lightG + 0.16 * upright * highA - 0.06 * highFriction, 0.82, 1.58),
    brakeGain: clamp(1 + 0.62 * highA + 0.14 * lowFriction + 0.07 * lightG + 0.06 * windStress + 0.18 * upright * highA - 0.06 * highFriction, 0.82, 1.64),
    edgeReserve: clamp(1 + 0.36 * highA + 0.12 * lowFriction + 0.10 * windStress + 0.05 * lightG, 0.86, 1.64),
    authorityScale: clamp(1 - 0.42 * highA + 0.08 * lowA - 0.04 * windStress - 0.04 * lightG + 0.04 * heavyG, 0.60, 1.18),
    centerBias: clamp(1 + 0.12 * highA + 0.12 * lowFriction + 0.08 * windStress, 0.84, 1.48),
    retryBoost: clamp(1 + 0.14 * highA + 0.08 * lowFriction + 0.05 * windStress, 0.82, 1.45),
    speedDamping: clamp(1 + 0.34 * highA + 0.10 * lowFriction + 0.06 * lightG + 0.06 * upright, 0.84, 1.65),
    landingGuard: clamp(1 + 0.34 * highA + 0.10 * lowFriction + 0.08 * windStress + 0.04 * upright, 0.84, 1.72),
    reserveBias: clamp(1 + 0.12 * highA + 0.08 * windStress + 0.08 * upright, 0.86, 1.45)
  };
}

function neutralDelta() {
  return {
    captureConservatism: 0,
    brakeGain: 0,
    edgeReserve: 0,
    authorityScale: 0,
    centerBias: 0,
    retryBoost: 0,
    speedDamping: 0,
    landingGuard: 0,
    reserveBias: 0
  };
}

function profileWithDelta(base, delta, meta = null) {
  const profile = {
    captureConservatism: clamp(base.captureConservatism + (delta.captureConservatism || 0), 0.68, 1.75),
    brakeGain: clamp(base.brakeGain + (delta.brakeGain || 0), 0.70, 1.85),
    edgeReserve: clamp(base.edgeReserve + (delta.edgeReserve || 0), 0.70, 1.90),
    authorityScale: clamp(base.authorityScale + (delta.authorityScale || 0), 0.55, 1.30),
    centerBias: clamp(base.centerBias + (delta.centerBias || 0), 0.70, 1.75),
    retryBoost: clamp(base.retryBoost + (delta.retryBoost || 0), 0.70, 1.70),
    speedDamping: clamp(base.speedDamping + (delta.speedDamping || 0), 0.70, 1.90),
    landingGuard: clamp(base.landingGuard + (delta.landingGuard || 0), 0.70, 1.95),
    reserveBias: clamp(base.reserveBias + (delta.reserveBias || 0), 0.70, 1.75)
  };
  profile.learnedVisits = meta?.visits || 0;
  profile.learnedWeight = meta?.weight || 0;
  return profile;
}

function bucketParts(targetId, params) {
  return {
    targetId: Number(targetId) || 0,
    maxAcc: clamp(roundTo(Number(params.maxAcc), 5, 20), 1, 60),
    g: roundTo(Number(params.g), 0.75, 9),
    friction: roundTo(Number(params.friction), 0.05, 0.03),
    windAmp: roundTo(Number(params.windAmp), 0.05, 0.03)
  };
}

function bucketKey(targetId, params) {
  const p = bucketParts(targetId, params);
  return `t${p.targetId}|a${p.maxAcc.toFixed(0)}|g${p.g.toFixed(2)}|f${p.friction.toFixed(2)}|w${p.windAmp.toFixed(2)}`;
}

function band(value, cuts, labels) {
  for (let i = 0; i < cuts.length; i++) if (value < cuts[i]) return labels[i];
  return labels[labels.length - 1];
}

function paramRegime(partsOrParams) {
  const maxAcc = Number(partsOrParams?.maxAcc ?? 20);
  const g = Number(partsOrParams?.g ?? 9);
  const friction = Number(partsOrParams?.friction ?? 0.03);
  const wind = Math.abs(Number(partsOrParams?.windAmp ?? 0.03));
  return {
    accRegime: band(maxAcc, [10, 22, 38, 52], ["tiny", "low", "normal", "high", "extreme"]),
    gRegime: band(g, [1.5, 5.5, 8.2, 10.2], ["zero", "low", "soft", "normal", "heavy"]),
    frictionRegime: band(friction, [0.015, 0.075, 0.22, 0.65], ["none", "low", "medium", "high", "extreme"]),
    windRegime: band(wind, [0.025, 0.10, 0.35, 0.70], ["none", "low", "medium", "high", "extreme"])
  };
}

function phaseParamSuffix(params) {
  if (!params) return "";
  const r = paramRegime(params);
  return `|pa${r.accRegime}|pg${r.gRegime}|pf${r.frictionRegime}|pw${r.windRegime}`;
}

function stripPhaseParamSuffix(key) {
  return String(key || "").split("|").filter(part => !(part.startsWith("pa") || part.startsWith("pg") || part.startsWith("pf") || part.startsWith("pw"))).join("|");
}

function parseBucketKey(key) {
  const m = /^t(-?\d+)\|a(-?\d+(?:\.\d+)?)\|g(-?\d+(?:\.\d+)?)\|f(-?\d+(?:\.\d+)?)\|w(-?\d+(?:\.\d+)?)$/.exec(String(key || ""));
  if (!m) return null;
  return {
    targetId: Number(m[1]),
    maxAcc: Number(m[2]),
    g: Number(m[3]),
    friction: Number(m[4]),
    windAmp: Number(m[5])
  };
}

function paramDistance(a, b) {
  if (!a || !b || a.targetId !== b.targetId) return Infinity;
  // Unitless coarse distance.  One bucket in maxAcc/gravity/friction/wind is
  // still useful, so those records are blended instead of requiring exact values.
  const dA = Math.abs(a.maxAcc - b.maxAcc) / 7.5;
  const dG = Math.abs(a.g - b.g) / 1.15;
  const dF = Math.abs(a.friction - b.friction) / 0.075;
  const dW = Math.abs(a.windAmp - b.windAmp) / 0.075;
  return Math.sqrt(dA * dA + dG * dG + dF * dF + dW * dW);
}

function mergeDelta(records, wantedParts) {
  const out = neutralDelta();
  let weightSum = 0;
  let visits = 0;
  for (const rec of records) {
    if (!rec?.key || !rec?.delta) continue;
    const parts = rec.parts || parseBucketKey(rec.key);
    const dist = paramDistance(wantedParts, parts);
    if (!Number.isFinite(dist) || dist > 2.15) continue;
    const exactBoost = dist < 0.001 ? 1.55 : 1.0;
    const visitBoost = clamp(Math.log1p(rec.visits || 0) / 3.0, 0.25, 1.45);
    const w = exactBoost * visitBoost / (1 + dist * dist);
    for (const name of PROFILE_FIELDS) out[name] += (rec.delta[name] || 0) * w;
    weightSum += w;
    visits += rec.visits || 0;
  }
  if (weightSum <= 0) return { delta: neutralDelta(), visits: 0, weight: 0 };
  for (const name of PROFILE_FIELDS) out[name] = clamp(out[name] / weightSum, -0.35, 0.42);
  return { delta: out, visits, weight: weightSum };
}

function ruleForEvent(type) {
  switch (type) {
    case "fast-flyby":
      return { brakeGain: 0.035, captureConservatism: 0.032, speedDamping: 0.028, landingGuard: 0.026, authorityScale: -0.018, retryBoost: 0.016 };
    case "wrong-energy":
      return { brakeGain: 0.030, speedDamping: 0.030, captureConservatism: 0.020, authorityScale: -0.014, retryBoost: 0.012 };
    case "edge-risk":
      return { edgeReserve: 0.060, centerBias: 0.052, landingGuard: 0.050, reserveBias: 0.036, speedDamping: 0.012, authorityScale: -0.024 };
    case "slow-arrival":
      return { authorityScale: 0.018, captureConservatism: -0.010, brakeGain: -0.006, reserveBias: 0.006 };
    case "slow-capture":
      return { brakeGain: 0.018, speedDamping: 0.018, landingGuard: 0.014, centerBias: 0.008 };
    case "stable":
      return { brakeGain: -0.003, captureConservatism: -0.002, edgeReserve: -0.002, centerBias: -0.002, speedDamping: -0.002, landingGuard: -0.002 };
    default:
      return null;
  }
}

function safeJsonParse(text, fallback) {
  try {
    return JSON.parse(text);
  } catch {
    return fallback;
  }
}

function speedBin(v) {
  if (v < 0.8) return "calm";
  if (v < 2.4) return "low";
  if (v < 5.2) return "mid";
  if (v < 9.0) return "high";
  return "extreme";
}

function zoneBin(angleNorm, targetId) {
  if (angleNorm < (targetId === 3 ? 0.18 : 0.14)) return "terminal";
  if (angleNorm < (targetId === 3 ? 0.60 : 0.44)) return "capture";
  if (angleNorm < (targetId === 3 ? 1.30 : 1.00)) return "approach";
  return "swing";
}

function energyBin(e) {
  if (e < -0.18) return "low";
  if (e > 0.18) return "high";
  return "ok";
}

function edgeBin(edgeRatio, edgeSide, outward) {
  if (edgeRatio > 0.90) return edgeSide < 0 ? "left-danger" : "right-danger";
  if (edgeRatio > 0.72) return outward ? (edgeSide < 0 ? "left-out" : "right-out") : (edgeSide < 0 ? "left" : "right");
  return "center";
}

function signBin(v, small = 0.04) {
  if (v > small) return "pos";
  if (v < -small) return "neg";
  return "zero";
}

function phaseRuleKey(targetId, features, params = null) {
  const zone = zoneBin(features.angleNorm || 0, targetId);
  const speed = speedBin(features.speedNorm || 0);
  const energy = energyBin(features.energyDelta || 0);
  const edge = edgeBin(features.edgeRatio || 0, features.edgeSide || 0, !!features.outward);
  const radial = signBin(features.radial || 0, 0.12);
  const phasePower = signBin(features.phasePower || 0, 0.04);
  return `t${targetId}|z${zone}|s${speed}|e${energy}|x${edge}|r${radial}|p${phasePower}${phaseParamSuffix(params)}`;
}

function defaultPhaseAction() {
  return {
    blend: 0.0,
    actionBias: 0.0,
    brakeBias: 0.0,
    centerBias: 0.0,
    reserveBias: 0.0,
    alignBias: 0.0,
    authorityScale: 1.0
  };
}

function phaseActionForEvent(type, features) {
  const out = defaultPhaseAction();
  const side = features?.edgeSide || 0;
  const radial = features?.radial || 0;
  const phasePower = features?.phasePower || 0;
  const energy = features?.energyDelta || 0;
  switch (type) {
    case "fast-flyby":
      out.blend = 0.060;
      out.brakeBias = 0.085;
      out.alignBias = 0.020;
      out.authorityScale = 0.985;
      out.actionBias = -0.014 * Math.sign(radial || phasePower || 1);
      break;
    case "wrong-energy":
      out.blend = 0.052;
      out.brakeBias = 0.070;
      out.alignBias = 0.018;
      out.actionBias = energy > 0 ? 0.018 * Math.sign(phasePower || 1) : -0.012 * Math.sign(phasePower || 1);
      out.authorityScale = 0.990;
      break;
    case "edge-risk":
      out.blend = 0.070;
      out.centerBias = 0.095;
      out.reserveBias = 0.075;
      out.brakeBias = 0.025;
      out.actionBias = -0.030 * Math.sign(side || 1);
      out.authorityScale = 0.985;
      break;
    case "slow-arrival":
      out.blend = 0.035;
      out.alignBias = 0.050;
      out.reserveBias = 0.020;
      out.actionBias = 0.018 * Math.sign(phasePower || 1);
      out.authorityScale = 1.012;
      break;
    case "slow-capture":
      out.blend = 0.046;
      out.brakeBias = 0.042;
      out.centerBias = 0.018;
      out.alignBias = 0.018;
      break;
    case "stable":
      out.blend = -0.006;
      out.brakeBias = -0.006;
      out.centerBias = -0.003;
      out.reserveBias = -0.003;
      break;
    default:
      return null;
  }
  return out;
}

function actionAdd(a, b, scale = 1) {
  const out = { ...a };
  for (const [k, v] of Object.entries(b || {})) {
    if (k === "authorityScale") {
      out[k] = 1 + ((out[k] ?? 1) - 1) + ((v ?? 1) - 1) * scale;
    } else {
      out[k] = (out[k] || 0) + v * scale;
    }
  }
  return out;
}

function actionClamp(a) {
  return {
    blend: clamp(a.blend || 0, 0, 0.24),
    actionBias: clamp(a.actionBias || 0, -0.16, 0.16),
    brakeBias: clamp(a.brakeBias || 0, -0.10, 0.24),
    centerBias: clamp(a.centerBias || 0, -0.10, 0.24),
    reserveBias: clamp(a.reserveBias || 0, -0.10, 0.24),
    alignBias: clamp(a.alignBias || 0, -0.10, 0.20),
    authorityScale: clamp(a.authorityScale || 1, 0.88, 1.12)
  };
}

function ruleMatches(rule, targetId, features, wantedParts) {
  if (!rule || Number(rule.targetId) !== Number(targetId)) return 0;
  const zone = zoneBin(features.angleNorm || 0, targetId);
  const speed = speedBin(features.speedNorm || 0);
  const energy = energyBin(features.energyDelta || 0);
  const edge = edgeBin(features.edgeRatio || 0, features.edgeSide || 0, !!features.outward);
  const radial = signBin(features.radial || 0, 0.12);
  const phasePower = signBin(features.phasePower || 0, 0.04);

  let specificity = 0;
  function match(field, value, weight) {
    const r = rule[field];
    if (r === undefined || r === null || r === "any") return 0.25 * weight;
    if (String(r) === String(value)) return weight;
    return -Infinity;
  }
  specificity += match("zone", zone, 1.00);
  specificity += match("speedBin", speed, 0.60);
  specificity += match("energyBin", energy, 0.50);
  specificity += match("edgeBin", edge, 0.75);
  specificity += match("radialBin", radial, 0.40);
  specificity += match("phasePowerBin", phasePower, 0.35);
  if (!Number.isFinite(specificity)) return 0;

  let paramWeight = 1;
  if (rule.center) {
    const center = {
      targetId,
      maxAcc: Number(rule.center.maxAcc ?? wantedParts.maxAcc),
      g: Number(rule.center.g ?? wantedParts.g),
      friction: Number(rule.center.friction ?? wantedParts.friction),
      windAmp: Number(rule.center.windAmp ?? wantedParts.windAmp)
    };
    const span = rule.span || {};
    const dA = Math.abs(center.maxAcc - wantedParts.maxAcc) / Math.max(7.5, Number(span.maxAcc) || 12);
    const dG = Math.abs(center.g - wantedParts.g) / Math.max(1.25, Number(span.g) || 2.0);
    const dF = Math.abs(center.friction - wantedParts.friction) / Math.max(0.075, Number(span.friction) || 0.14);
    const dW = Math.abs(center.windAmp - wantedParts.windAmp) / Math.max(0.075, Number(span.windAmp) || 0.14);
    const d = Math.sqrt(dA * dA + dG * dG + dF * dF + dW * dW);
    if (d > 2.5) return 0;
    paramWeight = 1 / (1 + d * d);
  }
  const visits = clamp(Math.log1p(rule.visits || 0) / 2.2, 0.35, 1.75);
  return Math.max(0, specificity) * paramWeight * visits;
}

export class AdaptiveAIStore {
  constructor() {
    this.records = new Map();
    this.phaseRules = new Map();
    this.phaseRuleGroups = new Map();
    this.loaded = false;
    this.staticLoaded = false;
    this.db = null;
    this.saveTimer = 0;
    this.lastObserveMs = 0;
    this.flushOnExitInstalled = false;
    this.load();
    this.installFlushOnExit();
  }

  basePhaseKey(key) {
    return stripPhaseParamSuffix(key);
  }

  storePhaseRule(rule) {
    if (!rule?.key) return;
    this.phaseRules.set(rule.key, rule);
    const base = this.basePhaseKey(rule.key);
    let group = this.phaseRuleGroups.get(base);
    if (!group) {
      group = new Set();
      this.phaseRuleGroups.set(base, group);
    }
    group.add(rule.key);
  }

  getKey(targetId, params) {
    return bucketKey(targetId, params);
  }

  getProfile(targetId, params) {
    const base = defaultProfile(params, targetId);
    const wantedParts = bucketParts(targetId, params);
    const merged = mergeDelta(this.records.values(), wantedParts);
    return profileWithDelta(base, merged.delta, merged);
  }

  getPhaseAdvice(targetId, params, features) {
    if (!features || this.phaseRules.size === 0) return null;
    const wantedParts = bucketParts(targetId, params);
    const exactKey = phaseRuleKey(targetId, features, params);
    const baseKey = phaseRuleKey(targetId, features, null);
    const keys = new Set([exactKey, baseKey]);
    const group = this.phaseRuleGroups.get(baseKey);
    if (group) for (const key of group) keys.add(key);

    const scored = [];
    for (const key of keys) {
      const rule = this.phaseRules.get(key);
      if (!rule) continue;
      let w = ruleMatches(rule, targetId, features, wantedParts);
      if (key === exactKey) w *= 1.35;
      else if (key === baseKey) w *= 0.72;
      if (w > 0.10) scored.push({ rule, w });
    }
    if (!scored.length) return null;
    scored.sort((a, b) => b.w - a.w);
    const top = scored.slice(0, 5);
    let total = 0;
    let action = defaultPhaseAction();
    for (const { rule, w } of top) {
      action = actionAdd(action, rule.action || defaultPhaseAction(), w);
      total += w;
    }
    if (total <= 0) return null;
    for (const key of Object.keys(action)) {
      if (key === "authorityScale") action[key] = 1 + ((action[key] ?? 1) - 1) / total;
      else action[key] = (action[key] || 0) / total;
    }
    action = actionClamp(action);
    return { ...action, confidence: clamp(total / 4.0, 0.10, 1.40), matches: top.length };
  }

  observe(targetId, params, type, severity = 1) {
    if (!BROWSER_ONLINE_LEARNING_ENABLED) return;
    const rule = ruleForEvent(type);
    if (!rule || !params) return;
    const key = this.getKey(targetId, params);
    const rec = this.records.get(key) || { key, parts: bucketParts(targetId, params), delta: neutralDelta(), stats: {}, visits: 0, updatedAt: Date.now() };
    const s = clamp(Number(severity) || 1, 0.15, 2.50);

    for (const [name, step] of Object.entries(rule)) {
      const current = rec.delta[name] || 0;
      rec.delta[name] = clamp(current + step * s * LEARNING_RATE, -0.35, 0.42);
    }

    if (type === "stable") {
      for (const name of Object.keys(rec.delta)) rec.delta[name] *= 0.997;
    }

    rec.stats[type] = (rec.stats[type] || 0) + 1;
    rec.visits = (rec.visits || 0) + 1;
    rec.updatedAt = Date.now();
    this.records.set(key, rec);
    this.scheduleSave();
  }

  observePhase(targetId, params, features, type, severity = 1) {
    if (!BROWSER_ONLINE_LEARNING_ENABLED) return;
    const delta = phaseActionForEvent(type, features);
    if (!delta || !params || !features) return;
    const key = phaseRuleKey(targetId, features, params);
    const bins = this.decodePhaseRuleKey(key);
    const rec = this.phaseRules.get(key) || {
      key,
      targetId,
      ...bins,
      center: bucketParts(targetId, params),
      span: { maxAcc: 12, g: 2.0, friction: 0.14, windAmp: 0.14 },
      action: defaultPhaseAction(),
      stats: {},
      visits: 0,
      updatedAt: Date.now(),
      source: "browser-online"
    };
    const s = clamp(Number(severity) || 1, 0.15, 2.50);
    const stepScale = 0.46 * s;
    rec.action = actionClamp(actionAdd(rec.action, delta, stepScale));
    if (type === "stable") {
      rec.action.blend *= 0.998;
      rec.action.brakeBias *= 0.998;
      rec.action.centerBias *= 0.998;
      rec.action.reserveBias *= 0.998;
    }
    rec.stats[type] = (rec.stats[type] || 0) + 1;
    rec.visits = (rec.visits || 0) + 1;
    rec.updatedAt = Date.now();
    this.storePhaseRule(rec);
    this.scheduleSave();
  }

  decodePhaseRuleKey(key) {
    const out = {};
    for (const part of String(key || "").split("|")) {
      if (part.startsWith("z")) out.zone = part.slice(1);
      else if (part.startsWith("s")) out.speedBin = part.slice(1);
      else if (part.startsWith("e")) out.energyBin = part.slice(1);
      else if (part.startsWith("x")) out.edgeBin = part.slice(1);
      else if (part.startsWith("r")) out.radialBin = part.slice(1);
      else if (part.startsWith("pa")) out.accRegime = part.slice(2);
      else if (part.startsWith("pg")) out.gRegime = part.slice(2);
      else if (part.startsWith("pf")) out.frictionRegime = part.slice(2);
      else if (part.startsWith("pw")) out.windRegime = part.slice(2);
      else if (part.startsWith("p")) out.phasePowerBin = part.slice(1);
    }
    return out;
  }

  load() {
    this.loadFromStaticSeedDatabase();
    if (USE_BROWSER_LOCAL_AI_CACHE) {
      this.loadFromLocalStorage();
      this.loadFromIndexedDb();
    }
  }

  loadFromStaticSeedDatabase() {
    if (typeof fetch === "undefined") return;
    fetch(STATIC_DB_URL, { cache: "no-store" })
      .then(response => response.ok ? response.json() : null)
      .then(data => {
        if (data) this.importDatabase(data, "static-json");
        this.staticLoaded = true;
      })
      .catch(() => { this.staticLoaded = true; });
  }

  importDatabase(data, source = "external") {
    if (!data || typeof data !== "object") return;
    const profiles = Array.isArray(data.profiles) ? data.profiles : [];
    for (const rec of profiles) {
      if (!rec?.key || !rec?.delta) continue;
      const existing = this.records.get(rec.key);
      const merged = existing ? this.mergeRecord(existing, rec) : { ...rec };
      merged.parts = merged.parts || parseBucketKey(merged.key);
      merged.source = merged.source || source;
      this.records.set(merged.key, merged);
    }
    const rules = Array.isArray(data.phaseRules) ? data.phaseRules : [];
    for (const rule of rules) {
      if (!rule?.key && Number.isFinite(rule?.targetId)) {
        rule.key = `ext-${source}-${this.phaseRules.size}-${Date.now()}`;
      }
      if (!rule?.key || !Number.isFinite(Number(rule.targetId))) continue;
      const existing = this.phaseRules.get(rule.key);
      const normalized = {
        ...rule,
        targetId: Number(rule.targetId),
        action: actionClamp(rule.action || defaultPhaseAction()),
        visits: Number(rule.visits) || 1,
        stats: rule.stats || {},
        source: rule.source || source
      };
      this.storePhaseRule(existing ? this.mergePhaseRule(existing, normalized) : normalized);
    }
  }

  mergeRecord(a, b) {
    const out = { ...a, delta: { ...a.delta }, stats: { ...(a.stats || {}) } };
    const bVisits = Math.max(1, Number(b.visits) || 1);
    const aVisits = Math.max(1, Number(a.visits) || 1);
    for (const name of PROFILE_FIELDS) {
      const av = out.delta[name] || 0;
      const bv = b.delta?.[name] || 0;
      out.delta[name] = clamp((av * aVisits + bv * bVisits) / (aVisits + bVisits), -0.35, 0.42);
    }
    for (const [k, v] of Object.entries(b.stats || {})) out.stats[k] = (out.stats[k] || 0) + v;
    out.visits = (a.visits || 0) + (b.visits || 0);
    out.updatedAt = Math.max(a.updatedAt || 0, b.updatedAt || 0, Date.now());
    return out;
  }

  mergePhaseRule(a, b) {
    const out = { ...a, action: { ...(a.action || {}) }, stats: { ...(a.stats || {}) } };
    const bVisits = Math.max(1, Number(b.visits) || 1);
    const aVisits = Math.max(1, Number(a.visits) || 1);
    const keys = new Set([...Object.keys(defaultPhaseAction()), ...Object.keys(b.action || {})]);
    for (const k of keys) {
      const av = k === "authorityScale" ? (out.action[k] ?? 1) : (out.action[k] || 0);
      const bv = k === "authorityScale" ? (b.action?.[k] ?? 1) : (b.action?.[k] || 0);
      out.action[k] = (av * aVisits + bv * bVisits) / (aVisits + bVisits);
    }
    out.action = actionClamp(out.action);
    for (const [k, v] of Object.entries(b.stats || {})) out.stats[k] = (out.stats[k] || 0) + v;
    out.visits = (a.visits || 0) + (b.visits || 0);
    out.updatedAt = Math.max(a.updatedAt || 0, b.updatedAt || 0, Date.now());
    return out;
  }

  loadFromLocalStorage() {
    if (typeof localStorage === "undefined") return;
    const raw = localStorage.getItem(LOCAL_KEY);
    const parsed = safeJsonParse(raw || "[]", []);
    if (Array.isArray(parsed)) this.importDatabase({ profiles: parsed }, "localStorage");
    const rawPhase = localStorage.getItem(PHASE_LOCAL_KEY);
    const parsedPhase = safeJsonParse(rawPhase || "[]", []);
    if (Array.isArray(parsedPhase)) this.importDatabase({ phaseRules: parsedPhase }, "localStorage");
  }

  loadFromIndexedDb() {
    if (typeof indexedDB === "undefined") return;
    try {
      const request = indexedDB.open(DB_NAME, 2);
      request.onupgradeneeded = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains(STORE_NAME)) db.createObjectStore(STORE_NAME, { keyPath: "key" });
        if (!db.objectStoreNames.contains(PHASE_STORE_NAME)) db.createObjectStore(PHASE_STORE_NAME, { keyPath: "key" });
      };
      request.onsuccess = () => {
        this.db = request.result;
        const tx = this.db.transaction([STORE_NAME, PHASE_STORE_NAME], "readonly");
        const profileStore = tx.objectStore(STORE_NAME);
        const phaseStore = tx.objectStore(PHASE_STORE_NAME);
        const getProfiles = profileStore.getAll();
        getProfiles.onsuccess = () => {
          const records = Array.isArray(getProfiles.result) ? getProfiles.result : [];
          this.importDatabase({ profiles: records }, "indexedDB");
        };
        const getRules = phaseStore.getAll();
        getRules.onsuccess = () => {
          const rules = Array.isArray(getRules.result) ? getRules.result : [];
          this.importDatabase({ phaseRules: rules }, "indexedDB");
          this.loaded = true;
        };
      };
    } catch {
      // localStorage/in-memory fallback is enough in restricted environments.
    }
  }

  scheduleSave() {
    const t = nowMs();
    if (t - this.lastObserveMs < 50) return;
    this.lastObserveMs = t;
    if (this.saveTimer) return;
    if (typeof setTimeout === "undefined") {
      this.flush();
      return;
    }
    this.saveTimer = setTimeout(() => {
      this.saveTimer = 0;
      this.flush();
    }, SAVE_DEBOUNCE_MS);
  }

  installFlushOnExit() {
    if (this.flushOnExitInstalled || typeof window === "undefined") return;
    this.flushOnExitInstalled = true;
    window.addEventListener("pagehide", () => this.flush());
    if (typeof document !== "undefined") {
      document.addEventListener("visibilitychange", () => {
        if (document.visibilityState === "hidden") this.flush();
      });
    }
  }

  flush() {
    if (!BROWSER_ONLINE_LEARNING_ENABLED || !USE_BROWSER_LOCAL_AI_CACHE) return;
    const records = Array.from(this.records.values());
    const rules = Array.from(this.phaseRules.values());
    if (typeof localStorage !== "undefined") {
      try {
        localStorage.setItem(LOCAL_KEY, JSON.stringify(records));
        localStorage.setItem(PHASE_LOCAL_KEY, JSON.stringify(rules));
      } catch {
        // Ignore quota/private-mode failures; runtime memory still works.
      }
    }
    if (!this.db) return;
    try {
      const tx = this.db.transaction([STORE_NAME, PHASE_STORE_NAME], "readwrite");
      const profileStore = tx.objectStore(STORE_NAME);
      const phaseStore = tx.objectStore(PHASE_STORE_NAME);
      for (const rec of records) profileStore.put(rec);
      for (const rule of rules) phaseStore.put(rule);
    } catch {
      // IndexedDB may be blocked on file://. localStorage fallback already ran.
    }
  }

  exportDatabase() {
    return {
      version: 2,
      generatedAt: new Date().toISOString(),
      profiles: Array.from(this.records.values()),
      phaseRules: Array.from(this.phaseRules.values())
    };
  }
}

export const adaptiveAI = new AdaptiveAIStore();
