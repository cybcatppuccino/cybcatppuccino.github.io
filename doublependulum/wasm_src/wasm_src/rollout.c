// Minimal standalone WASM rollout evaluator for the CEM/MPC hot path.
// No libc/libm dependency: fast trigonometry is implemented locally so clang
// can emit a self-contained wasm32 module without Emscripten glue.

#define PI 3.141592653589793238462643383279502884
#define TWO_PI 6.283185307179586476925286766559005768
#define HALF_PI 1.570796326794896619231321691639751442

#define MAX_BLOCKS 64
#define MAX_CANDIDATES 256

static double block_buffer[256];
static double candidate_buffer[MAX_CANDIDATES * MAX_BLOCKS];
static double cost_buffer[MAX_CANDIDATES];

static double f_abs(double x) { return x < 0.0 ? -x : x; }
static double f_min(double a, double b) { return a < b ? a : b; }
static double f_max(double a, double b) { return a > b ? a : b; }
static double f_clamp(double x, double lo, double hi) { return f_max(lo, f_min(hi, x)); }
static double f_sign(double x) { return (x > 0.0) - (x < 0.0); }
static double f_sqrt(double x) { return __builtin_sqrt(x); }

static double wrap_angle(double angle) {
  // Avoid fmod/libm. Rollout angles remain close enough that a few reductions
  // are sufficient; the while fallback handles unusually high angular rates.
  double a = angle + PI;
  if (a >= TWO_PI || a < 0.0) {
    long k = (long)(a / TWO_PI);
    a -= (double)k * TWO_PI;
    while (a >= TWO_PI) a -= TWO_PI;
    while (a < 0.0) a += TWO_PI;
  }
  return a - PI;
}

static double fast_sin_core(double x) {
  const double x2 = x * x;
  // 9th-order Taylor on [-pi/2, pi/2].  Good enough for CEM ranking while
  // avoiding imported Math.sin calls inside the innermost physics loop.
  return x * (1.0 + x2 * (-0.16666666666666666 + x2 * (0.008333333333333333 + x2 * (-0.0001984126984126984 + x2 * 0.0000027557319223985893))));
}

static double fast_sin(double x) {
  x = wrap_angle(x);
  if (x > HALF_PI) return fast_sin_core(PI - x);
  if (x < -HALF_PI) return -fast_sin_core(PI + x);
  return fast_sin_core(x);
}

static double fast_cos(double x) {
  return fast_sin(x + HALF_PI);
}

typedef struct {
  double x;
  double vx;
  double th1;
  double th2;
  double om1;
  double om2;
} State;

typedef struct {
  double x;
  double vx;
  double th1;
  double th2;
  double om1;
  double om2;
} Deriv;

typedef struct {
  double m1;
  double m2;
  double l1;
  double l2;
  double g;
  double maxAcc;
  double windAmp;
  double friction;
  double segmentHalfLength;
  double rightSegmentExtensionFraction;
} Params;

typedef struct {
  double left;
  double right;
  double center;
  double half;
} Bounds;

static Bounds support_bounds(const Params* p) {
  const double baseHalf = f_max(0.01, p->segmentHalfLength);
  const double extensionFraction = f_max(0.0, p->rightSegmentExtensionFraction);
  Bounds b;
  b.left = -baseHalf;
  b.right = baseHalf + 2.0 * baseHalf * extensionFraction;
  b.center = 0.5 * (b.left + b.right);
  b.half = 0.5 * (b.right - b.left);
  return b;
}

static double target_angle(int targetId, int which) {
  if (targetId == 0) return 0.0;
  if (targetId == 1) return which == 0 ? 0.0 : PI;
  if (targetId == 2) return which == 0 ? PI : 0.0;
  return PI;
}

static double constrain_support_acceleration(const State* s, double commandAcc, const Params* p, const Bounds* b) {
  double a = f_clamp(commandAcc, -p->maxAcc, p->maxAcc);
  if (s->x <= b->left && s->vx <= 0.0 && a < 0.0) return 0.0;
  if (s->x >= b->right && s->vx >= 0.0 && a > 0.0) return 0.0;
  return a;
}

static Deriv derivative(const State* s, double commandAcc, const Params* p, const Bounds* b) {
  const double aBase = constrain_support_acceleration(s, commandAcc, p, b);
  const double effectiveHorizontalAcc = aBase; // CEM rollouts intentionally use useWind=false.

  const double m1 = p->m1;
  const double m2 = p->m2;
  const double l1 = p->l1;
  const double l2 = p->l2;
  const double g = p->g;
  const double th1 = s->th1;
  const double th2 = s->th2;
  const double om1 = s->om1;
  const double om2 = s->om2;
  const double delta = th1 - th2;
  const double cosDelta = fast_cos(delta);
  const double sinDelta = fast_sin(delta);

  const double M11 = (m1 + m2) * l1 * l1;
  const double M12 = m2 * l1 * l2 * cosDelta;
  const double M22 = m2 * l2 * l2;
  const double det = M11 * M22 - M12 * M12;

  double rhs1 =
    -m2 * l1 * l2 * sinDelta * om2 * om2 -
    (m1 + m2) * g * l1 * fast_sin(th1) -
    (m1 + m2) * l1 * fast_cos(th1) * effectiveHorizontalAcc;

  double rhs2 =
    m2 * l1 * l2 * sinDelta * om1 * om1 -
    m2 * g * l2 * fast_sin(th2) -
    m2 * l2 * fast_cos(th2) * effectiveHorizontalAcc;

  const double friction = f_max(0.0, p->friction);
  if (friction > 0.0) {
    const double c1 = 0.34 * friction * (m1 + m2) * l1 * l1;
    const double c2 = 0.28 * friction * m2 * l2 * l2;
    rhs1 -= c1 * om1;
    rhs2 -= c2 * om2;
  }

  const double dom1 = (rhs1 * M22 - rhs2 * M12) / det;
  const double dom2 = (M11 * rhs2 - M12 * rhs1) / det;

  Deriv d;
  d.x = s->vx;
  d.vx = aBase;
  d.th1 = om1;
  d.th2 = om2;
  d.om1 = dom1;
  d.om2 = dom2;
  return d;
}

static State add_scaled_state(const State* s, const Deriv* d, double scale) {
  State out;
  out.x = s->x + d->x * scale;
  out.vx = s->vx + d->vx * scale;
  out.th1 = s->th1 + d->th1 * scale;
  out.th2 = s->th2 + d->th2 * scale;
  out.om1 = s->om1 + d->om1 * scale;
  out.om2 = s->om2 + d->om2 * scale;
  return out;
}

static State step_rk4(const State* s, double commandAcc, const Params* p, const Bounds* b, double dt) {
  const Deriv k1 = derivative(s, commandAcc, p, b);
  const State s2 = add_scaled_state(s, &k1, dt * 0.5);
  const Deriv k2 = derivative(&s2, commandAcc, p, b);
  const State s3 = add_scaled_state(s, &k2, dt * 0.5);
  const Deriv k3 = derivative(&s3, commandAcc, p, b);
  const State s4 = add_scaled_state(s, &k3, dt);
  const Deriv k4 = derivative(&s4, commandAcc, p, b);

  State next;
  const double w = dt / 6.0;
  next.x = s->x + w * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
  next.vx = s->vx + w * (k1.vx + 2.0 * k2.vx + 2.0 * k3.vx + k4.vx);
  next.th1 = wrap_angle(s->th1 + w * (k1.th1 + 2.0 * k2.th1 + 2.0 * k3.th1 + k4.th1));
  next.th2 = wrap_angle(s->th2 + w * (k1.th2 + 2.0 * k2.th2 + 2.0 * k3.th2 + k4.th2));
  next.om1 = s->om1 + w * (k1.om1 + 2.0 * k2.om1 + 2.0 * k3.om1 + k4.om1);
  next.om2 = s->om2 + w * (k1.om2 + 2.0 * k2.om2 + 2.0 * k3.om2 + k4.om2);

  if (next.x < b->left) {
    next.x = b->left;
    next.vx = 0.0;
  } else if (next.x > b->right) {
    next.x = b->right;
    next.vx = 0.0;
  }
  return next;
}

static double apply_track_safety(const State* s, double raw, const Params* p, const Bounds* b) {
  const double xErr = s->x - b->center;
  const double ratio = f_abs(xErr) / b->half;
  double outward = f_sign(xErr);
  if (f_abs(xErr) <= 1e-6) outward = f_abs(s->vx) > 1e-6 ? f_sign(s->vx) : 1.0;
  const int movingOutward = xErr * s->vx > 0.0;
  double safe = raw;

  if (ratio > 0.84) {
    const double q = (ratio - 0.84) / 0.16;
    const double barrier = q * q;
    const double outwardCmd = f_max(0.0, safe * outward);
    safe -= outward * f_min(outwardCmd, p->maxAcc * 0.72 * barrier);
    if (movingOutward) {
      safe += -outward * p->maxAcc * 0.38 * barrier;
      safe += -1.38 * s->vx * (1.0 + 1.75 * barrier);
    } else {
      safe += -0.24 * s->vx * barrier;
    }
  }

  if (ratio > 0.955 && f_sign(safe) == outward && movingOutward) {
    safe = f_min(f_abs(safe), 0.12 * p->maxAcc) * -outward;
  }

  if (ratio > 0.992) {
    safe = -outward * f_max(0.30 * p->maxAcc, f_abs(safe));
  }

  return f_clamp(safe, -p->maxAcc, p->maxAcc);
}

static double total_energy(const State* s, const Params* p) {
  const double m1 = p->m1;
  const double m2 = p->m2;
  const double l1 = p->l1;
  const double l2 = p->l2;
  const double g = p->g;
  const double v1x = s->vx + l1 * fast_cos(s->th1) * s->om1;
  const double v1y = -l1 * fast_sin(s->th1) * s->om1;
  const double v2x = s->vx + l1 * fast_cos(s->th1) * s->om1 + l2 * fast_cos(s->th2) * s->om2;
  const double v2y = -l1 * fast_sin(s->th1) * s->om1 - l2 * fast_sin(s->th2) * s->om2;
  const double kinetic = 0.5 * m1 * (v1x * v1x + v1y * v1y) + 0.5 * m2 * (v2x * v2x + v2y * v2y);
  const double potential = -(m1 + m2) * g * l1 * fast_cos(s->th1) - m2 * g * l2 * fast_cos(s->th2);
  return kinetic + potential;
}

static double target_energy(int targetId, const Params* p) {
  State t;
  t.x = 0.0;
  t.vx = 0.0;
  t.th1 = target_angle(targetId, 0);
  t.th2 = target_angle(targetId, 1);
  t.om1 = 0.0;
  t.om2 = 0.0;
  return total_energy(&t, p);
}

static double score_state(const State* s, int targetId, const Params* p, const Bounds* b, int terminal, double targetEnergy, double energyScale) {
  const double e1 = wrap_angle(s->th1 - target_angle(targetId, 0));
  const double e2 = wrap_angle(s->th2 - target_angle(targetId, 1));
  const double angleCost = e1 * e1 + e2 * e2;
  const double speedCost = s->om1 * s->om1 + s->om2 * s->om2;
  const double xErr = s->x - b->center;
  const double xNorm = xErr / b->half;
  const double centerCost = xNorm * xNorm + 0.26 * s->vx * s->vx;
  const double terminalAngleWeight = targetId == 3 ? 17.5 : 16.0;
  const double terminalSpeedWeight = targetId == 3 ? 2.4 : 2.2;
  const int nearBalance = angleCost < 0.55 && speedCost < 16.0;
  const int roughEnvironment = targetId == 3 && (p->friction < 0.015 || p->windAmp > 0.15);
  const double centerWeight = targetId == 3 ? (roughEnvironment ? (nearBalance ? 1.05 : 2.85) : (nearBalance ? 1.8 : 3.2)) : 6.8;
  const double edgeWeight = targetId == 3 ? (roughEnvironment ? 6.2 : 5.8) : 3.5;

  const double dE = (total_energy(s, p) - targetEnergy) / energyScale;
  const double energyCost = dE * dE;

  const double edgeRatio = f_abs(xErr) / b->half;
  const double outwardSpeed = xErr * s->vx > 0.0 ? f_abs(s->vx) : 0.0;
  double softEdge = 0.0;
  if (edgeRatio > 0.74) {
    const double u = (edgeRatio - 0.74) / 0.26;
    const double u2 = u * u;
    softEdge = u2 * u2;
  }
  double hardEdge = 0.0;
  if (edgeRatio > 0.93) {
    const double u = (edgeRatio - 0.93) / 0.07;
    const double u2 = u * u;
    const double u4 = u2 * u2;
    hardEdge = u4 * u4;
  }
  const double edgeCost = 12.0 * softEdge + 120.0 * hardEdge + 1.45 * outwardSpeed * outwardSpeed * f_max(0.0, edgeRatio - 0.68);
  const double sqrtAngle = f_sqrt(angleCost);
  const double arrivalGate = targetId == 0 ? f_clamp(1.0 - sqrtAngle / 0.72, 0.0, 1.0) :
    f_clamp(1.0 - sqrtAngle / (targetId == 3 ? 0.96 : 0.76), 0.0, 1.0);
  const double captureGate = targetId == 3 ? arrivalGate : 0.0;
  const double positiveDE = f_max(0.0, dE);
  const double landingCost = arrivalGate * ((targetId == 3 ? 0.55 : 0.24) * speedCost + (targetId == 3 ? 1.05 : 0.62) * energyCost + 0.36 * positiveDE * positiveDE);
  const double flybyCost = captureGate * (0.34 * speedCost + 0.18 * positiveDE * positiveDE);

  if (terminal) {
    if (targetId == 0) {
      const double fastDown = angleCost < 1.10 ? 1.0 : 0.0;
      return 18.50 * angleCost + (2.70 + 0.90 * fastDown) * speedCost + 4.00 * centerCost + 1.45 * energyCost + 3.70 * edgeCost + 0.45 * landingCost;
    }
    return terminalAngleWeight * angleCost + (terminalSpeedWeight + 0.65 * captureGate) * speedCost + centerWeight * centerCost + (0.75 + 0.07 * captureGate) * energyCost + edgeWeight * edgeCost + 1.15 * flybyCost + (targetId == 3 ? 3.25 : 1.35) * landingCost;
  }

  if (targetId == 3) {
    const double centerRunWeight = roughEnvironment ? (angleCost < 0.55 && speedCost < 16.0 ? 0.09 : 0.34) : (angleCost < 0.55 && speedCost < 16.0 ? 0.18 : 0.42);
    const double angleWeight = roughEnvironment ? 0.52 : 0.46;
    const double speedWeight = roughEnvironment ? 0.086 : 0.078;
    const double energyWeight = roughEnvironment ? 0.18 : 0.20;
    const double edgeRunWeight = roughEnvironment ? 1.42 : 1.36;
    return angleWeight * angleCost + (speedWeight + 0.12 * captureGate) * speedCost + centerRunWeight * centerCost + energyWeight * energyCost + edgeRunWeight * edgeCost + 0.48 * flybyCost + 0.70 * landingCost;
  }

  if (targetId == 0) {
    const int nearDown = angleCost < 0.75;
    return 0.42 * angleCost + (nearDown ? 0.130 : 0.080) * speedCost + 0.46 * centerCost + 0.50 * energyCost + 1.34 * edgeCost + 0.10 * landingCost;
  }

  return 0.34 * angleCost + (0.060 + 0.10 * captureGate) * speedCost + 0.85 * centerCost + 0.20 * energyCost + 1.10 * edgeCost + 0.36 * flybyCost + 0.16 * landingCost;
}

__attribute__((visibility("default")))
double* getBlockBufferPtr(void) {
  return block_buffer;
}


static double evaluate_blocks_ptr(
  const double* blocks,
  double x,
  double vx,
  double th1,
  double th2,
  double om1,
  double om2,
  int targetId,
  const Params* p,
  const Bounds* b,
  int blockCount,
  int horizon,
  int blockLen
) {
  if (blockCount <= 0 || blockCount > MAX_BLOCKS || horizon <= 0 || blockLen <= 0 || p->maxAcc <= 0.0) return 1.0e100;

  State s;
  s.x = x;
  s.vx = vx;
  s.th1 = th1;
  s.th2 = th2;
  s.om1 = om1;
  s.om2 = om2;

  const double dt = 0.036;
  const double invA2 = 1.0 / f_max(1.0, p->maxAcc * p->maxAcc);
  const double tEnergy = target_energy(targetId, p);
  const double energyScale = f_max(1.0, p->g * (p->m1 + p->m2) * p->l1 + p->g * p->m2 * p->l2);
  double cost = 0.0;
  int stepIndex = 0;

  for (int blockIndex = 0; blockIndex < blockCount && stepIndex < horizon; blockIndex++) {
    const double rawBlockA = blocks[blockIndex];
    const int blockEnd = horizon < stepIndex + blockLen ? horizon : stepIndex + blockLen;
    for (; stepIndex < blockEnd; stepIndex++) {
      const double safeA = apply_track_safety(&s, rawBlockA, p, b);
      s = step_rk4(&s, safeA, p, b, dt);
      cost += score_state(&s, targetId, p, b, 0, tEnergy, energyScale);
      cost += 0.0020 * safeA * safeA * invA2;
      if ((s.x - b->center) * safeA > 0.0) cost += 0.0012 * f_abs((s.x - b->center) / b->half);
    }
  }

  cost += score_state(&s, targetId, p, b, 1, tEnergy, energyScale);
  return cost;
}

__attribute__((visibility("default")))
double* getCandidateBufferPtr(void) {
  return candidate_buffer;
}

__attribute__((visibility("default")))
double* getCostBufferPtr(void) {
  return cost_buffer;
}

__attribute__((visibility("default")))
double evaluateBlocks(
  double x,
  double vx,
  double th1,
  double th2,
  double om1,
  double om2,
  int targetId,
  double m1,
  double m2,
  double l1,
  double l2,
  double g,
  double maxAcc,
  double windAmp,
  double friction,
  double segmentHalfLength,
  double rightSegmentExtensionFraction,
  int blockCount,
  int horizon,
  int blockLen
) {
  Params p;
  p.m1 = m1;
  p.m2 = m2;
  p.l1 = l1;
  p.l2 = l2;
  p.g = g;
  p.maxAcc = maxAcc;
  p.windAmp = windAmp;
  p.friction = friction;
  p.segmentHalfLength = segmentHalfLength;
  p.rightSegmentExtensionFraction = rightSegmentExtensionFraction;
  Bounds b = support_bounds(&p);
  return evaluate_blocks_ptr(block_buffer, x, vx, th1, th2, om1, om2, targetId, &p, &b, blockCount, horizon, blockLen);
}

__attribute__((visibility("default")))
int evaluateBatch(
  double x,
  double vx,
  double th1,
  double th2,
  double om1,
  double om2,
  int targetId,
  double m1,
  double m2,
  double l1,
  double l2,
  double g,
  double maxAcc,
  double windAmp,
  double friction,
  double segmentHalfLength,
  double rightSegmentExtensionFraction,
  int candidateCount,
  int blockCount,
  int horizon,
  int blockLen
) {
  if (candidateCount <= 0 || candidateCount > MAX_CANDIDATES || blockCount <= 0 || blockCount > MAX_BLOCKS || horizon <= 0 || blockLen <= 0 || maxAcc <= 0.0) return 0;

  Params p;
  p.m1 = m1;
  p.m2 = m2;
  p.l1 = l1;
  p.l2 = l2;
  p.g = g;
  p.maxAcc = maxAcc;
  p.windAmp = windAmp;
  p.friction = friction;
  p.segmentHalfLength = segmentHalfLength;
  p.rightSegmentExtensionFraction = rightSegmentExtensionFraction;
  Bounds b = support_bounds(&p);

  for (int candidateIndex = 0; candidateIndex < candidateCount; candidateIndex++) {
    const double* blocks = candidate_buffer + candidateIndex * MAX_BLOCKS;
    cost_buffer[candidateIndex] = evaluate_blocks_ptr(blocks, x, vx, th1, th2, om1, om2, targetId, &p, &b, blockCount, horizon, blockLen);
  }
  return 1;
}
