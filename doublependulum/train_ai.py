#!/usr/bin/env python3
"""
Offline/local trainer for the double-pendulum adaptive AI.

What it trains
--------------
This program does not change the player's physical parameters.  It improves the
existing learned buckets first, only using a tiny amount of optional exploration.
It simulates target capture / settling, classifies failures, and updates a compact
database consumed directly by js/ai_learning.js:

  * profile deltas per coarse parameter bucket
  * phase-rule advice shared across ranges, such as capture / swing / edge zones

Run examples
------------
  python train_ai.py                         # endless resume-training, publishing only accepted improvements
  python train_ai.py --episodes 2000          # finite resume-training run
  python train_ai.py --episodes 1000 --fresh  # start a new database intentionally
  python train_ai.py --eval-only --validate-episodes 160

For browser use, serve the folder locally after training, for example:
  python -m http.server 8000
Then open http://localhost:8000/ so the browser can fetch ai_data/training_db.json.
The browser build in this package does not train online by default; it uses the
published static training result directly.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from functools import lru_cache
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

PI = math.pi
TWO_PI = 2.0 * math.pi

TARGETS = {
    0: (0.0, 0.0),
    1: (0.0, math.pi),
    2: (math.pi, 0.0),
    3: (math.pi, math.pi),
}

PROFILE_FIELDS = [
    "captureConservatism",
    "brakeGain",
    "edgeReserve",
    "authorityScale",
    "centerBias",
    "retryBoost",
    "speedDamping",
    "landingGuard",
    "reserveBias",
]

DEFAULT_PARAMS = {
    "m1": 1.0,
    "m2": 1.0,
    "l1": 1.0,
    "l2": 1.0,
    "g": 9.0,
    "maxAcc": 20.0,
    "windAmp": 0.03,
    "friction": 0.03,
    "segmentHalfLength": 3.20,
    "rightSegmentExtensionFraction": 0.25,
}

PARAM_RANGES = {
    "maxAcc": (1.0, 60.0),
    "g": (0.0, 12.0),
    "windAmp": (0.0, 1.0),
    "friction": (0.0, 1.0),
}

# Success now means practical capture: if the links enter a broad, low-speed
# basin that the terminal local stabilizer can quickly polish, the episode is
# considered successful.  This matches the browser goal better than requiring
# mathematically tiny angle/rate errors for nearly a full second.
SUCCESS_ANGLE = {0: 0.30, 1: 0.70, 2: 0.72, 3: 0.96}
SUCCESS_SPEED = {0: 1.40, 1: 3.85, 2: 4.05, 3: 5.80}
SUCCESS_DWELL = {0: 0.10, 1: 0.12, 2: 0.12, 3: 0.14}
# Browser-facing handoff success: once the system spends a short time inside a
# broad, safe capture basin, the local stabilizer has enough time/authority to
# polish the motion.  This is stricter than a one-frame near miss, but much less
# brittle than demanding nearly zero angles and rates during offline evaluation.
HANDOFF_DWELL = {0: 0.055, 1: 0.070, 2: 0.070, 3: 0.085}
CAPTURE_ANGLE = {0: 0.36, 1: 0.82, 2: 0.84, 3: 1.10}
CAPTURE_SPEED = {0: 1.80, 1: 4.65, 2: 4.85, 3: 6.60}

PRACTICAL_BUCKETS = [
    # Handful of common playable settings used when no existing database is
    # available.  Once records exist, training strongly prefers those records.
    {"maxAcc": 15.0, "g": 9.0, "friction": 0.03, "windAmp": 0.00},
    {"maxAcc": 20.0, "g": 9.0, "friction": 0.03, "windAmp": 0.03},
    {"maxAcc": 25.0, "g": 9.0, "friction": 0.03, "windAmp": 0.05},
    {"maxAcc": 35.0, "g": 9.0, "friction": 0.03, "windAmp": 0.05},
    {"maxAcc": 45.0, "g": 8.25, "friction": 0.02, "windAmp": 0.08},
    {"maxAcc": 55.0, "g": 10.5, "friction": 0.08, "windAmp": 0.12},
    {"maxAcc": 30.0, "g": 7.5, "friction": 0.00, "windAmp": 0.05},
    {"maxAcc": 30.0, "g": 11.25, "friction": 0.10, "windAmp": 0.00},
]



# Named training curricula.  These are deliberately not full-random slider
# samples: each scenario corresponds to a practical regime the web UI can
# encounter, and the trainer tries to lift every regime instead of optimizing
# only the average case.  Ranges are slider-space ranges and are quantized by
# params_from_specialty() to the same resolution used by the browser controls.
SPECIALTY_CASES = [
    {"name": "default_index10_state3", "weight": 4.00, "maxAcc": (14.1, 25.9), "g": (7.8, 10.2), "friction": (0.0, 0.13), "windAmp": (0.0, 0.13)},
    {"name": "default_play", "weight": 2.80, "maxAcc": (16.0, 30.0), "g": (8.2, 10.0), "friction": (0.0, 0.075), "windAmp": (0.0, 0.090)},
    {"name": "high_acc", "weight": 1.45, "maxAcc": (42.0, 60.0), "g": (7.6, 11.2), "friction": (0.0, 0.120), "windAmp": (0.0, 0.140)},
    {"name": "low_acc", "weight": 1.30, "maxAcc": (1.0, 12.0), "g": (6.0, 11.2), "friction": (0.0, 0.090), "windAmp": (0.0, 0.080)},
    {"name": "low_g_low_acc", "weight": 1.30, "maxAcc": (1.0, 14.0), "g": (0.0, 4.5), "friction": (0.0, 0.080), "windAmp": (0.0, 0.060)},
    {"name": "low_g_normal_acc", "weight": 1.10, "maxAcc": (16.0, 35.0), "g": (0.0, 5.0), "friction": (0.0, 0.080), "windAmp": (0.0, 0.090)},
    {"name": "high_g", "weight": 1.05, "maxAcc": (18.0, 46.0), "g": (10.2, 12.0), "friction": (0.0, 0.110), "windAmp": (0.0, 0.090)},
    {"name": "large_friction", "weight": 1.35, "maxAcc": (12.0, 50.0), "g": (7.0, 11.5), "friction": (0.22, 0.65), "windAmp": (0.0, 0.100)},
    {"name": "very_large_friction", "weight": 0.70, "maxAcc": (18.0, 60.0), "g": (7.0, 12.0), "friction": (0.65, 1.00), "windAmp": (0.0, 0.080)},
    {"name": "low_friction_high_acc", "weight": 1.20, "maxAcc": (38.0, 60.0), "g": (6.5, 11.2), "friction": (0.0, 0.025), "windAmp": (0.0, 0.110)},
    {"name": "windy", "weight": 1.10, "maxAcc": (18.0, 52.0), "g": (6.5, 11.2), "friction": (0.0, 0.100), "windAmp": (0.18, 0.55)},
    {"name": "extreme_wind", "weight": 0.45, "maxAcc": (30.0, 60.0), "g": (5.0, 11.2), "friction": (0.0, 0.160), "windAmp": (0.55, 1.00)},
    {"name": "low_acc_high_friction", "weight": 0.95, "maxAcc": (1.0, 16.0), "g": (4.0, 10.5), "friction": (0.18, 0.75), "windAmp": (0.0, 0.070)},
    {"name": "high_acc_high_friction", "weight": 0.85, "maxAcc": (42.0, 60.0), "g": (6.5, 12.0), "friction": (0.20, 0.90), "windAmp": (0.0, 0.120)},
]
SPECIALTY_BY_NAME = {case["name"]: case for case in SPECIALTY_CASES}


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def wrap_angle(a: float) -> float:
    a = (a + PI) % TWO_PI
    return a - PI


def angle_error(angle: float, target: float) -> float:
    return wrap_angle(angle - target)


def round_to(value: float, step: float, fallback: float = 0.0) -> float:
    if not math.isfinite(value):
        value = fallback
    return round(value / step) * step


@dataclass
class State:
    x: float
    vx: float
    th1: float
    th2: float
    om1: float
    om2: float

    def copy(self) -> "State":
        return State(self.x, self.vx, self.th1, self.th2, self.om1, self.om2)


def support_bounds(params: Dict[str, float]) -> Tuple[float, float, float, float]:
    base = max(0.01, params.get("segmentHalfLength", 3.2))
    ext = max(0.0, params.get("rightSegmentExtensionFraction", 0.25))
    left = -base
    right = base + 2.0 * base * ext
    center = 0.5 * (left + right)
    half = 0.5 * (right - left)
    return left, right, center, half


def support_center(params: Dict[str, float]) -> float:
    return support_bounds(params)[2]


def support_half(params: Dict[str, float]) -> float:
    return support_bounds(params)[3]


def support_edge_ratio(s: State, params: Dict[str, float]) -> float:
    return abs(s.x - support_center(params)) / max(0.01, support_half(params))


def support_outward_sign(s: State, params: Dict[str, float]) -> int:
    err = s.x - support_center(params)
    if abs(err) > 1e-8:
        return 1 if err > 0 else -1
    if abs(s.vx) > 1e-8:
        return 1 if s.vx > 0 else -1
    return 1


def constrain_support_acc(s: State, a: float, params: Dict[str, float]) -> float:
    a = clamp(a, -params["maxAcc"], params["maxAcc"])
    left, right, _, _ = support_bounds(params)
    if s.x <= left and s.vx <= 0 and a < 0:
        return 0.0
    if s.x >= right and s.vx >= 0 and a > 0:
        return 0.0
    return a


def wind_acceleration(t: float, params: Dict[str, float]) -> float:
    amp = params.get("windAmp", 0.0)
    if amp == 0:
        return 0.0
    raw = (
        0.56 * math.sin(0.73 * t + 0.6)
        + 0.31 * math.sin(1.37 * t + 2.1)
        + 0.13 * math.sin(2.11 * t + 4.2)
    )
    return amp * raw


def derivative(s: State, command_acc: float, params: Dict[str, float], t: float, use_wind: bool = True) -> State:
    a_base = constrain_support_acc(s, command_acc, params)
    a_wind = wind_acceleration(t, params) if use_wind else 0.0
    a_eff = a_base - a_wind

    m1, m2 = params["m1"], params["m2"]
    l1, l2 = params["l1"], params["l2"]
    g = params["g"]
    th1, th2, om1, om2 = s.th1, s.th2, s.om1, s.om2
    delta = th1 - th2
    cd = math.cos(delta)
    sd = math.sin(delta)

    m11 = (m1 + m2) * l1 * l1
    m12 = m2 * l1 * l2 * cd
    m22 = m2 * l2 * l2
    det = m11 * m22 - m12 * m12
    if abs(det) < 1e-12:
        det = 1e-12 if det >= 0 else -1e-12

    rhs1 = (
        -m2 * l1 * l2 * sd * om2 * om2
        - (m1 + m2) * g * l1 * math.sin(th1)
        - (m1 + m2) * l1 * math.cos(th1) * a_eff
    )
    rhs2 = (
        m2 * l1 * l2 * sd * om1 * om1
        - m2 * g * l2 * math.sin(th2)
        - m2 * l2 * math.cos(th2) * a_eff
    )

    friction = max(0.0, params.get("friction", 0.0))
    if friction > 0:
        c1 = 0.34 * friction * (m1 + m2) * l1 * l1
        c2 = 0.28 * friction * m2 * l2 * l2
        rhs1 -= c1 * om1
        rhs2 -= c2 * om2

    dom1 = (rhs1 * m22 - rhs2 * m12) / det
    dom2 = (m11 * rhs2 - m12 * rhs1) / det
    return State(s.vx, a_base, om1, om2, dom1, dom2)


def rk4_step(s: State, a: float, params: Dict[str, float], t: float, dt: float, use_wind: bool = True) -> State:
    k1 = derivative(s, a, params, t, use_wind)
    s2 = State(
        s.x + 0.5 * dt * k1.x,
        s.vx + 0.5 * dt * k1.vx,
        s.th1 + 0.5 * dt * k1.th1,
        s.th2 + 0.5 * dt * k1.th2,
        s.om1 + 0.5 * dt * k1.om1,
        s.om2 + 0.5 * dt * k1.om2,
    )
    k2 = derivative(s2, a, params, t + 0.5 * dt, use_wind)
    s3 = State(
        s.x + 0.5 * dt * k2.x,
        s.vx + 0.5 * dt * k2.vx,
        s.th1 + 0.5 * dt * k2.th1,
        s.th2 + 0.5 * dt * k2.th2,
        s.om1 + 0.5 * dt * k2.om1,
        s.om2 + 0.5 * dt * k2.om2,
    )
    k3 = derivative(s3, a, params, t + 0.5 * dt, use_wind)
    s4 = State(
        s.x + dt * k3.x,
        s.vx + dt * k3.vx,
        s.th1 + dt * k3.th1,
        s.th2 + dt * k3.th2,
        s.om1 + dt * k3.om1,
        s.om2 + dt * k3.om2,
    )
    k4 = derivative(s4, a, params, t + dt, use_wind)
    out = State(
        s.x + (dt / 6.0) * (k1.x + 2 * k2.x + 2 * k3.x + k4.x),
        s.vx + (dt / 6.0) * (k1.vx + 2 * k2.vx + 2 * k3.vx + k4.vx),
        wrap_angle(s.th1 + (dt / 6.0) * (k1.th1 + 2 * k2.th1 + 2 * k3.th1 + k4.th1)),
        wrap_angle(s.th2 + (dt / 6.0) * (k1.th2 + 2 * k2.th2 + 2 * k3.th2 + k4.th2)),
        s.om1 + (dt / 6.0) * (k1.om1 + 2 * k2.om1 + 2 * k3.om1 + k4.om1),
        s.om2 + (dt / 6.0) * (k1.om2 + 2 * k2.om2 + 2 * k3.om2 + k4.om2),
    )
    left, right, _, _ = support_bounds(params)
    if out.x < left:
        out.x = left
        out.vx = max(0.0, out.vx)
    elif out.x > right:
        out.x = right
        out.vx = min(0.0, out.vx)
    return out


def potential_energy(s: State, params: Dict[str, float]) -> float:
    m1, m2, l1, l2, g = params["m1"], params["m2"], params["l1"], params["l2"], params["g"]
    y1 = -l1 * math.cos(s.th1)
    y2 = y1 - l2 * math.cos(s.th2)
    return m1 * g * y1 + m2 * g * y2


def kinetic_energy(s: State, params: Dict[str, float]) -> float:
    m1, m2, l1, l2 = params["m1"], params["m2"], params["l1"], params["l2"]
    vx1 = s.vx + l1 * s.om1 * math.cos(s.th1)
    vy1 = l1 * s.om1 * math.sin(s.th1)
    vx2 = vx1 + l2 * s.om2 * math.cos(s.th2)
    vy2 = vy1 + l2 * s.om2 * math.sin(s.th2)
    return 0.5 * m1 * (vx1 * vx1 + vy1 * vy1) + 0.5 * m2 * (vx2 * vx2 + vy2 * vy2)


def total_energy(s: State, params: Dict[str, float]) -> float:
    return potential_energy(s, params) + kinetic_energy(s, params)


def target_energy(target: int, params: Dict[str, float]) -> float:
    a1, a2 = TARGETS[target]
    s = State(support_center(params), 0.0, a1, a2, 0.0, 0.0)
    return total_energy(s, params)


def closeness(s: State, target: int) -> Tuple[float, float, float, float]:
    a1, a2 = TARGETS[target]
    e1 = angle_error(s.th1, a1)
    e2 = angle_error(s.th2, a2)
    return math.hypot(e1, e2), math.hypot(s.om1, s.om2), e1, e2


def support_phase_power(s: State, params: Dict[str, float]) -> float:
    return (
        (params["m1"] + params["m2"]) * params["l1"] * s.om1 * math.cos(s.th1)
        + params["m2"] * params["l2"] * s.om2 * math.cos(s.th2)
    )


def target_radial_velocity(s: State, target: int) -> float:
    angle_norm, _, e1, e2 = closeness(s, target)
    if angle_norm < 1e-8:
        return 0.0
    return (e1 * s.om1 + e2 * s.om2) / angle_norm


def base_profile(params: Dict[str, float], target: int) -> Dict[str, float]:
    max_a = max(1.0, params["maxAcc"])
    high_a = clamp((max_a - 20.0) / 40.0, 0.0, 1.0)
    low_a = clamp((18.0 - max_a) / 17.0, 0.0, 1.0)
    f = max(0.0, params["friction"])
    low_f = clamp((0.045 - f) / 0.045, 0.0, 1.0)
    high_f = clamp((f - 0.08) / 0.22, 0.0, 1.0)
    wind = abs(params["windAmp"])
    wind_stress = clamp(wind / 0.35, 0.0, 1.0)
    g = max(0.0, params["g"])
    light_g = clamp((8.6 - g) / 8.6, 0.0, 1.0)
    heavy_g = clamp((g - 9.8) / 2.2, 0.0, 1.0)
    upright = 1.0 if target == 3 else 0.0
    return {
        "captureConservatism": clamp(1 + 0.56 * high_a + 0.14 * low_f + 0.08 * wind_stress + 0.08 * light_g + 0.16 * upright * high_a - 0.06 * high_f, 0.82, 1.58),
        "brakeGain": clamp(1 + 0.62 * high_a + 0.14 * low_f + 0.07 * light_g + 0.06 * wind_stress + 0.18 * upright * high_a - 0.06 * high_f, 0.82, 1.64),
        "edgeReserve": clamp(1 + 0.36 * high_a + 0.12 * low_f + 0.10 * wind_stress + 0.05 * light_g, 0.86, 1.64),
        "authorityScale": clamp(1 - 0.42 * high_a + 0.08 * low_a - 0.04 * wind_stress - 0.04 * light_g + 0.04 * heavy_g, 0.60, 1.18),
        "centerBias": clamp(1 + 0.12 * high_a + 0.12 * low_f + 0.08 * wind_stress, 0.84, 1.48),
        "retryBoost": clamp(1 + 0.14 * high_a + 0.08 * low_f + 0.05 * wind_stress, 0.82, 1.45),
        "speedDamping": clamp(1 + 0.34 * high_a + 0.10 * low_f + 0.06 * light_g + 0.06 * upright, 0.84, 1.65),
        "landingGuard": clamp(1 + 0.34 * high_a + 0.10 * low_f + 0.08 * wind_stress + 0.04 * upright, 0.84, 1.72),
        "reserveBias": clamp(1 + 0.12 * high_a + 0.08 * wind_stress + 0.08 * upright, 0.86, 1.45),
    }


PROFILE_BOUNDS = {
    "captureConservatism": (0.68, 1.75),
    "brakeGain": (0.70, 1.85),
    "edgeReserve": (0.70, 1.90),
    "authorityScale": (0.55, 1.30),
    "centerBias": (0.70, 1.75),
    "retryBoost": (0.70, 1.70),
    "speedDamping": (0.70, 1.90),
    "landingGuard": (0.70, 1.95),
    "reserveBias": (0.70, 1.75),
}


def bucket_parts(target: int, params: Dict[str, float]) -> Dict[str, float]:
    return {
        "targetId": int(target),
        "maxAcc": clamp(round_to(params["maxAcc"], 5.0, 20.0), 1.0, 60.0),
        "g": round_to(params["g"], 0.75, 9.0),
        "friction": round_to(params["friction"], 0.05, 0.03),
        "windAmp": round_to(params["windAmp"], 0.05, 0.03),
    }


def param_distance(wanted: Dict[str, float], parts: Optional[Dict[str, float]]) -> float:
    if not parts or int(parts.get("targetId", -999)) != int(wanted.get("targetId", -998)):
        return float("inf")
    d_a = abs(float(parts.get("maxAcc", 20.0)) - wanted["maxAcc"]) / 7.5
    d_g = abs(float(parts.get("g", 9.0)) - wanted["g"]) / 1.15
    d_f = abs(float(parts.get("friction", 0.03)) - wanted["friction"]) / 0.075
    d_w = abs(float(parts.get("windAmp", 0.03)) - wanted["windAmp"]) / 0.075
    return math.sqrt(d_a * d_a + d_g * d_g + d_f * d_f + d_w * d_w)


def merge_delta(records, wanted_parts: Dict[str, float]) -> Tuple[Dict[str, float], int, float]:
    # Same neighborhood blending idea as js/ai_learning.js.  This is the most
    # important trainer fix: random continuous slider values should still reuse
    # nearby learned buckets instead of starting from zero every episode.
    out = neutral_delta()
    weight_sum = 0.0
    visits = 0
    for rec in records:
        if not rec or not rec.get("delta"):
            continue
        parts = rec.get("parts")
        d = param_distance(wanted_parts, parts)
        if not math.isfinite(d) or d > 2.15:
            continue
        exact_boost = 1.55 if d < 0.001 else 1.0
        visit_boost = clamp(math.log1p(float(rec.get("visits", 0))) / 3.0, 0.25, 1.45)
        w = exact_boost * visit_boost / (1.0 + d * d)
        for name in PROFILE_FIELDS:
            out[name] += float(rec.get("delta", {}).get(name, 0.0)) * w
        weight_sum += w
        visits += int(rec.get("visits", 0) or 0)
    if weight_sum <= 0:
        return neutral_delta(), 0, 0.0
    for name in PROFILE_FIELDS:
        out[name] = clamp(out[name] / weight_sum, -0.35, 0.42)
    return out, visits, weight_sum


def profile_with_delta(params: Dict[str, float], target: int, delta: Dict[str, float]) -> Dict[str, float]:
    p = base_profile(params, target)
    for k in PROFILE_FIELDS:
        lo, hi = PROFILE_BOUNDS[k]
        p[k] = clamp(p[k] + float(delta.get(k, 0.0)), lo, hi)
    return p


def profile_from_db(db: Dict[str, Any], params: Dict[str, float], target: int) -> Dict[str, float]:
    wanted = bucket_parts(target, params)
    delta, visits, weight = merge_delta(db.get("profiles_by_key", {}).values(), wanted)
    p = profile_with_delta(params, target, delta)
    p["learnedVisits"] = visits
    p["learnedWeight"] = weight
    return p


def profile_from_record(params: Dict[str, float], target: int, record: Optional[Dict[str, Any]]) -> Dict[str, float]:
    return profile_with_delta(params, target, (record or {}).get("delta", {}))


def bucket_key(target: int, params: Dict[str, float]) -> str:
    p = bucket_parts(target, params)
    return f"t{target}|a{p['maxAcc']:.0f}|g{p['g']:.2f}|f{p['friction']:.2f}|w{p['windAmp']:.2f}"


def _band(value: float, cuts: List[float], labels: List[str]) -> str:
    for cut, label in zip(cuts, labels):
        if value < cut:
            return label
    return labels[-1]


def param_regime(parts_or_params: Dict[str, Any]) -> Dict[str, str]:
    # Coarser than profile buckets.  This prevents phase-rule advice learned in
    # high-acc / low-g / high-friction regions from overwriting advice for the
    # normal browser regime, while still sharing inside a broad specialty.
    max_acc = float(parts_or_params.get("maxAcc", 20.0))
    g = float(parts_or_params.get("g", 9.0))
    friction = float(parts_or_params.get("friction", 0.03))
    wind = float(parts_or_params.get("windAmp", 0.03))
    return {
        "accRegime": _band(max_acc, [10.0, 22.0, 38.0, 52.0], ["tiny", "low", "normal", "high", "extreme"]),
        "gRegime": _band(g, [1.5, 5.5, 8.2, 10.2], ["zero", "low", "soft", "normal", "heavy"]),
        "frictionRegime": _band(friction, [0.015, 0.075, 0.22, 0.65], ["none", "low", "medium", "high", "extreme"]),
        "windRegime": _band(abs(wind), [0.025, 0.10, 0.35, 0.70], ["none", "low", "medium", "high", "extreme"]),
    }


def phase_param_suffix(params: Optional[Dict[str, float]]) -> str:
    if not params:
        return ""
    r = param_regime(params)
    return f"|pa{r['accRegime']}|pg{r['gRegime']}|pf{r['frictionRegime']}|pw{r['windRegime']}"


def neutral_delta() -> Dict[str, float]:
    return {k: 0.0 for k in PROFILE_FIELDS}


def rule_for_event(event: str) -> Optional[Dict[str, float]]:
    if event == "fast-flyby":
        return {"brakeGain": 0.035, "captureConservatism": 0.032, "speedDamping": 0.028, "landingGuard": 0.026, "authorityScale": -0.018, "retryBoost": 0.016}
    if event == "wrong-energy":
        return {"brakeGain": 0.030, "speedDamping": 0.030, "captureConservatism": 0.020, "authorityScale": -0.014, "retryBoost": 0.012}
    if event == "edge-risk":
        return {"edgeReserve": 0.060, "centerBias": 0.052, "landingGuard": 0.050, "reserveBias": 0.036, "speedDamping": 0.012, "authorityScale": -0.024}
    if event == "slow-arrival":
        return {"authorityScale": 0.018, "captureConservatism": -0.010, "brakeGain": -0.006, "reserveBias": 0.006}
    if event == "slow-capture":
        return {"brakeGain": 0.018, "speedDamping": 0.018, "landingGuard": 0.014, "centerBias": 0.008}
    if event == "stable":
        return {"brakeGain": -0.003, "captureConservatism": -0.002, "edgeReserve": -0.002, "centerBias": -0.002, "speedDamping": -0.002, "landingGuard": -0.002}
    return None


def update_profile(db: Dict[str, Any], target: int, params: Dict[str, float], event: str, severity: float) -> None:
    rule = rule_for_event(event)
    if not rule:
        return
    key = bucket_key(target, params)
    rec = db["profiles_by_key"].setdefault(key, {
        "key": key,
        "parts": bucket_parts(target, params),
        "delta": neutral_delta(),
        "stats": {},
        "visits": 0,
        "source": "python-trainer",
        "updatedAt": int(time.time() * 1000),
    })
    s = clamp(severity, 0.15, 2.5)
    lr = 0.58
    for k, step in rule.items():
        rec["delta"][k] = clamp(rec["delta"].get(k, 0.0) + step * s * lr, -0.35, 0.42)
    if event == "stable":
        for k in list(rec["delta"].keys()):
            rec["delta"][k] *= 0.997
    rec["stats"][event] = rec["stats"].get(event, 0) + 1
    rec["visits"] = rec.get("visits", 0) + 1
    rec["updatedAt"] = int(time.time() * 1000)


def speed_bin(v: float) -> str:
    if v < 0.8: return "calm"
    if v < 2.4: return "low"
    if v < 5.2: return "mid"
    if v < 9.0: return "high"
    return "extreme"


def zone_bin(angle_norm: float, target: int) -> str:
    if angle_norm < (0.18 if target == 3 else 0.14): return "terminal"
    if angle_norm < (0.60 if target == 3 else 0.44): return "capture"
    if angle_norm < (1.30 if target == 3 else 1.00): return "approach"
    return "swing"


def energy_bin(e: float) -> str:
    if e < -0.18: return "low"
    if e > 0.18: return "high"
    return "ok"


def edge_bin(edge_ratio: float, edge_side: int, outward: bool) -> str:
    if edge_ratio > 0.90:
        return "left-danger" if edge_side < 0 else "right-danger"
    if edge_ratio > 0.72:
        if outward:
            return "left-out" if edge_side < 0 else "right-out"
        return "left" if edge_side < 0 else "right"
    return "center"


def sign_bin(v: float, small: float = 0.04) -> str:
    if v > small: return "pos"
    if v < -small: return "neg"
    return "zero"


def phase_features(s: State, target: int, params: Dict[str, float]) -> Dict[str, float]:
    angle_norm, speed_norm, _, _ = closeness(s, target)
    x_err = s.x - support_center(params)
    half = support_half(params)
    e_scale = max(1.0, params["g"] * ((params["m1"] + params["m2"]) * params["l1"] + params["m2"] * params["l2"]))
    e_delta = (total_energy(s, params) - target_energy(target, params)) / e_scale
    edge_side = 0 if abs(x_err) < 1e-8 else (1 if x_err > 0 else -1)
    delta = s.th1 - s.th2
    angular_momentum = (
        (params["m1"] + params["m2"]) * params["l1"] * params["l1"] * s.om1 +
        params["m2"] * params["l2"] * params["l2"] * s.om2 +
        0.5 * params["m2"] * params["l1"] * params["l2"] * math.cos(delta) * (s.om1 + s.om2)
    )
    return {
        "angleNorm": angle_norm,
        "speedNorm": speed_norm,
        "energyDelta": e_delta,
        "edgeRatio": abs(x_err) / max(0.01, half),
        "edgeSide": edge_side,
        "outward": 1.0 if x_err * s.vx > 0 else 0.0,
        "radial": target_radial_velocity(s, target),
        "phasePower": support_phase_power(s, params) / max(0.5, params["g"] * 0.12),
        "angularMomentum": angular_momentum / max(1.0, params["g"] * (params["m1"] + params["m2"]) * params["l1"]),
    }


def phase_rule_key(target: int, f: Dict[str, float], params: Optional[Dict[str, float]] = None) -> Tuple[str, Dict[str, Any]]:
    zone = zone_bin(f["angleNorm"], target)
    sb = speed_bin(f["speedNorm"])
    eb = energy_bin(f["energyDelta"])
    xb = edge_bin(f["edgeRatio"], int(f["edgeSide"]), bool(f["outward"]))
    rb = sign_bin(f["radial"], 0.12)
    pb = sign_bin(f["phasePower"], 0.04)
    base_key = f"t{target}|z{zone}|s{sb}|e{eb}|x{xb}|r{rb}|p{pb}"
    bins = {"targetId": target, "zone": zone, "speedBin": sb, "energyBin": eb, "edgeBin": xb, "radialBin": rb, "phasePowerBin": pb}
    if params:
        bins.update(param_regime(params))
    return base_key + phase_param_suffix(params), bins


def default_action() -> Dict[str, float]:
    return {"blend": 0.0, "actionBias": 0.0, "brakeBias": 0.0, "centerBias": 0.0, "reserveBias": 0.0, "alignBias": 0.0, "authorityScale": 1.0}


def event_action(event: str, f: Dict[str, float]) -> Optional[Dict[str, float]]:
    out = default_action()
    side = int(f.get("edgeSide", 0)) or 1
    radial = f.get("radial", 0.0)
    phase_power = f.get("phasePower", 0.0)
    energy = f.get("energyDelta", 0.0)
    sign = 1.0 if (radial or phase_power or 1.0) > 0 else -1.0
    psign = 1.0 if (phase_power or 1.0) > 0 else -1.0
    if event == "fast-flyby":
        out.update({"blend": 0.060, "brakeBias": 0.085, "alignBias": 0.020, "authorityScale": 0.985, "actionBias": -0.014 * sign})
    elif event == "wrong-energy":
        out.update({"blend": 0.052, "brakeBias": 0.070, "alignBias": 0.018, "authorityScale": 0.990, "actionBias": (0.018 if energy > 0 else -0.012) * psign})
    elif event == "edge-risk":
        out.update({"blend": 0.092, "centerBias": 0.130, "reserveBias": 0.105, "brakeBias": 0.035, "authorityScale": 0.972, "actionBias": -0.044 * side})
    elif event == "slow-arrival":
        out.update({"blend": 0.035, "alignBias": 0.050, "reserveBias": 0.020, "authorityScale": 1.012, "actionBias": 0.018 * psign})
    elif event == "slow-capture":
        out.update({"blend": 0.046, "brakeBias": 0.042, "centerBias": 0.018, "alignBias": 0.018})
    elif event == "stable":
        out.update({"blend": -0.006, "brakeBias": -0.006, "centerBias": -0.003, "reserveBias": -0.003})
    else:
        return None
    return out


def action_clamp(a: Dict[str, float]) -> Dict[str, float]:
    return {
        "blend": clamp(a.get("blend", 0.0), 0.0, 0.24),
        "actionBias": clamp(a.get("actionBias", 0.0), -0.16, 0.16),
        "brakeBias": clamp(a.get("brakeBias", 0.0), -0.10, 0.24),
        "centerBias": clamp(a.get("centerBias", 0.0), -0.10, 0.24),
        "reserveBias": clamp(a.get("reserveBias", 0.0), -0.10, 0.24),
        "alignBias": clamp(a.get("alignBias", 0.0), -0.10, 0.20),
        "authorityScale": clamp(a.get("authorityScale", 1.0), 0.88, 1.12),
    }


def action_add(a: Dict[str, float], b: Dict[str, float], scale: float) -> Dict[str, float]:
    out = dict(a)
    for k, v in b.items():
        if k == "authorityScale":
            out[k] = 1.0 + (out.get(k, 1.0) - 1.0) + (v - 1.0) * scale
        else:
            out[k] = out.get(k, 0.0) + v * scale
    return action_clamp(out)


def update_phase_rule(db: Dict[str, Any], target: int, params: Dict[str, float], f: Dict[str, float], event: str, severity: float) -> None:
    action = event_action(event, f)
    if not action:
        return
    key, bins = phase_rule_key(target, f, params)
    rec = db["phase_by_key"].setdefault(key, {
        "key": key,
        **bins,
        "center": bucket_parts(target, params),
        "span": {"maxAcc": 12.0, "g": 2.0, "friction": 0.14, "windAmp": 0.14},
        "action": default_action(),
        "stats": {},
        "visits": 0,
        "source": "python-trainer",
        "updatedAt": int(time.time() * 1000),
    })
    rec["action"] = action_add(rec["action"], action, 0.46 * clamp(severity, 0.15, 2.5))
    rec["stats"][event] = rec["stats"].get(event, 0) + 1
    rec["visits"] = rec.get("visits", 0) + 1
    rec["updatedAt"] = int(time.time() * 1000)


def _phase_match_value(rule: Dict[str, Any], field: str, value: str, weight: float) -> float:
    r = rule.get(field)
    if r is None or r == "any":
        return 0.25 * weight
    if str(r) == str(value):
        return weight
    return -float("inf")


def phase_rule_weight(rule: Dict[str, Any], target: int, params: Dict[str, float], f: Dict[str, float]) -> float:
    if int(rule.get("targetId", -999)) != int(target):
        return 0.0
    zone = zone_bin(f["angleNorm"], target)
    sb = speed_bin(f["speedNorm"])
    eb = energy_bin(f["energyDelta"])
    xb = edge_bin(f["edgeRatio"], int(f["edgeSide"]), bool(f["outward"]))
    rb = sign_bin(f["radial"], 0.12)
    pb = sign_bin(f["phasePower"], 0.04)
    specificity = 0.0
    for field, value, weight in [
        ("zone", zone, 1.00), ("speedBin", sb, 0.60), ("energyBin", eb, 0.50),
        ("edgeBin", xb, 0.75), ("radialBin", rb, 0.40), ("phasePowerBin", pb, 0.35),
    ]:
        specificity += _phase_match_value(rule, field, value, weight)
    if not math.isfinite(specificity):
        return 0.0

    wanted = bucket_parts(target, params)
    param_weight = 1.0
    if rule.get("center"):
        center = rule.get("center", {})
        span = rule.get("span", {})
        d_a = abs(float(center.get("maxAcc", wanted["maxAcc"])) - wanted["maxAcc"]) / max(7.5, float(span.get("maxAcc", 12.0) or 12.0))
        d_g = abs(float(center.get("g", wanted["g"])) - wanted["g"]) / max(1.25, float(span.get("g", 2.0) or 2.0))
        d_f = abs(float(center.get("friction", wanted["friction"])) - wanted["friction"]) / max(0.075, float(span.get("friction", 0.14) or 0.14))
        d_w = abs(float(center.get("windAmp", wanted["windAmp"])) - wanted["windAmp"]) / max(0.075, float(span.get("windAmp", 0.14) or 0.14))
        d = math.sqrt(d_a * d_a + d_g * d_g + d_f * d_f + d_w * d_w)
        if d > 2.5:
            return 0.0
        param_weight = 1.0 / (1.0 + d * d)
    visit_weight = clamp(math.log1p(float(rule.get("visits", 0))) / 2.2, 0.35, 1.75)
    return max(0.0, specificity) * param_weight * visit_weight


def get_phase_advice(db: Dict[str, Any], target: int, params: Dict[str, float], f: Dict[str, float]) -> Optional[Dict[str, float]]:
    # Fast path: trainer-generated phase rules are keyed by exact coarse phase
    # bins, so most lookups do not need to scan hundreds/thousands of rules.
    phase_rules = db.get("phase_by_key", {})
    exact_key, _ = phase_rule_key(target, f, params)
    legacy_key, _ = phase_rule_key(target, f, None)
    scored = []
    # Prefer parameter-regime-specialized advice, then fall back to legacy
    # phase-only rules from older databases.  This keeps lookups fast while
    # avoiding cross-contamination between very different slider regimes.
    for key, boost in ((exact_key, 1.35), (legacy_key, 0.72)):
        rule = phase_rules.get(key)
        if not rule:
            continue
        w = phase_rule_weight(rule, target, params, f) * boost
        if w > 0.10:
            scored.append((w, rule))
    if not scored:
        return None
    if not scored:
        return None
    action = default_action()
    total = 0.0
    for w, rule in scored[:5]:
        action = action_add(action, rule.get("action", default_action()), w)
        total += w
    if total <= 0:
        return None
    for k in list(action.keys()):
        if k == "authorityScale":
            action[k] = 1.0 + ((action.get(k, 1.0) - 1.0) / total)
        else:
            action[k] = action.get(k, 0.0) / total
    action = action_clamp(action)
    action["confidence"] = clamp(total / 4.0, 0.10, 1.40)
    action["matches"] = min(5, len(scored))
    return action


def apply_phase_advice(s: State, target: int, params: Dict[str, float], profile: Dict[str, float], raw: float, f: Dict[str, float], advice: Optional[Dict[str, float]]) -> float:
    if not advice:
        return raw
    cap = max(0.25, params["maxAcc"] * clamp(profile.get("authorityScale", 1.0), 0.55, 1.30) * clamp(advice.get("authorityScale", 1.0), 0.88, 1.12))
    center_a = center_return(s, params, profile.get("centerBias", 1.0))
    align_a = align_acceleration(s, target, params, profile)
    phase_a = clamp(4.9 * support_phase_power(s, params) * (1.0 + max(0.0, advice.get("brakeBias", 0.0))), -0.42 * cap, 0.42 * cap)
    edge_pressure = clamp((float(f.get("edgeRatio", 0.0)) - 0.66) / 0.30, 0.0, 1.0)
    side = int(f.get("edgeSide", 0)) or support_outward_sign(s, params)
    reserve_a = -side * params["maxAcc"] * edge_pressure * advice.get("reserveBias", 0.0)
    learned = raw
    learned += advice.get("actionBias", 0.0) * cap
    learned += advice.get("brakeBias", 0.0) * phase_a
    learned += advice.get("alignBias", 0.0) * align_a
    learned += advice.get("centerBias", 0.0) * center_a
    learned += reserve_a
    blend = clamp(advice.get("blend", 0.0) * clamp(advice.get("confidence", 0.0), 0.0, 1.25), 0.0, 0.22)
    return clamp((1.0 - blend) * raw + blend * learned, -cap, cap)


def center_return(s: State, params: Dict[str, float], gain_scale: float = 1.0) -> float:
    half = support_half(params)
    x_err = s.x - support_center(params)
    ratio = abs(x_err) / max(0.01, half)
    edge_boost = 1.0 + 3.8 * ((ratio - 0.48) / 0.52) ** 2 if ratio > 0.48 else 1.0
    raw = (-1.34 * x_err - 1.82 * s.vx) * gain_scale * edge_boost
    return clamp(raw, -0.84 * params["maxAcc"], 0.84 * params["maxAcc"])


def energy_pump(s: State, target: int, params: Dict[str, float], profile: Dict[str, float]) -> float:
    e_err = target_energy(target, params) - total_energy(s, params)
    phase = support_phase_power(s, params)
    if abs(phase) < 1e-5 or abs(e_err) < 1e-4:
        return 0.0
    cap = params["maxAcc"] * profile["authorityScale"]
    e_scale = max(1.0, params["g"] * ((params["m1"] + params["m2"]) * params["l1"] + params["m2"] * params["l2"]))
    gate = clamp(abs(e_err) / (0.55 * e_scale), 0.20, 1.0)
    return -clamp(cap * (0.40 + 0.60 * gate), 0.0, params["maxAcc"]) * (1.0 if e_err * phase > 0 else -1.0)


@lru_cache(maxsize=512)
def make_lqr_gain_cached(target: int, g: float, friction: float) -> Tuple[float, ...]:
    import numpy as np

    params = dict(DEFAULT_PARAMS)
    params["g"] = float(g)
    params["friction"] = float(friction)
    params["windAmp"] = 0.0
    dt = 1.0 / 60.0
    eps = 1e-5
    n = 6
    a1, a2 = TARGETS[target]

    def transition(z, a):
        s0 = State(
            support_center(params) + float(z[4]),
            float(z[5]),
            wrap_angle(a1 + float(z[0])),
            wrap_angle(a2 + float(z[1])),
            float(z[2]),
            float(z[3]),
        )
        nxt = rk4_step(s0, float(a), params, 0.0, dt, use_wind=False)
        return np.array([
            angle_error(nxt.th1, a1), angle_error(nxt.th2, a2),
            nxt.om1, nxt.om2, nxt.x - support_center(params), nxt.vx
        ], dtype=float)

    base = np.zeros(n)
    A = np.zeros((n, n))
    for c in range(n):
        zp = base.copy(); zm = base.copy()
        zp[c] += eps; zm[c] -= eps
        A[:, c] = (transition(zp, 0.0) - transition(zm, 0.0)) / (2.0 * eps)
    B = ((transition(base, eps) - transition(base, -eps)) / (2.0 * eps)).reshape((n, 1))

    q1 = 42 if target == 0 else (155 if target == 3 else (144 if target == 2 else 135))
    q2 = 42 if target == 0 else (165 if target == 3 else (150 if target == 2 else 145))
    qom1 = 7 if target == 0 else (25 if target == 3 else (22 if target == 2 else 20))
    qom2 = 7 if target == 0 else (27 if target == 3 else (23 if target == 2 else 21))
    rough = friction < 0.015
    qcenter = 8.0 if target == 0 else (1.15 if (target == 3 and rough) else (1.75 if target == 3 else 2.25))
    qcart = 4.6 if target == 0 else (0.82 if (target == 3 and rough) else (1.20 if target == 3 else 1.65))
    Q = np.diag([q1, q2, qom1, qom2, qcenter, qcart])
    R = 0.08 if target == 0 else (0.046 if target == 2 else (0.090 if target == 3 else 0.13))
    P = Q.copy()
    AT = A.T; BT = B.T
    for _ in range(190):
        den = float(R + (BT @ P @ B)[0, 0])
        P = Q + AT @ P @ A - (AT @ P @ B @ BT @ P @ A) / max(1e-9, den)
    den = float(R + (BT @ P @ B)[0, 0])
    K = (BT @ P @ A / max(1e-9, den)).reshape(-1)
    return tuple(float(x) for x in K)


def lqr_acceleration(s: State, target: int, params: Dict[str, float]) -> float:
    # Coarse cache keys keep endless training fast; the LQR gain varies smoothly
    # with these parameters and is blended with robust nonlinear terms anyway.
    gain = make_lqr_gain_cached(target, round_to(float(params["g"]), 0.75, 9.0), round_to(float(params.get("friction", 0.0)), 0.05, 0.03))
    a1, a2 = TARGETS[target]
    z = [angle_error(s.th1, a1), angle_error(s.th2, a2), s.om1, s.om2, s.x - support_center(params), s.vx]
    feedback = sum(g * v for g, v in zip(gain, z))
    return clamp(-feedback, -params["maxAcc"], params["maxAcc"])


def align_acceleration(s: State, target: int, params: Dict[str, float], profile: Dict[str, float]) -> float:
    _, _, e1, e2 = closeness(s, target)
    # Match the browser controller's target-seeking direction: for the non-down
    # equilibria a positive angular error generally requires a negative support
    # acceleration in the coarse capture controller.  The old trainer used the
    # opposite sign, so offline learning often reinforced corrections generated
    # by a policy that was fighting its own stabilizer.
    if target == 0:
        angular = 4.8 * e1 + 3.9 * e2 + 2.1 * s.om1 + 1.7 * s.om2
    else:
        angular = -(4.8 * e1 + 3.9 * e2 + 2.1 * s.om1 + 1.7 * s.om2)
    angular *= profile["brakeGain"] * profile["speedDamping"]
    return clamp(angular, -params["maxAcc"], params["maxAcc"])


def apply_track_safety(s: State, raw: float, params: Dict[str, float], profile: Dict[str, float]) -> float:
    half = support_half(params)
    x_err = s.x - support_center(params)
    ratio = abs(x_err) / max(0.01, half)
    outward = support_outward_sign(s, params)
    safe = clamp(raw, -params["maxAcc"], params["maxAcc"])
    if ratio > 0.64:
        barrier = clamp((ratio - 0.64) / 0.34, 0.0, 1.0) ** 2
        safe += -outward * params["maxAcc"] * 0.36 * barrier * profile["edgeReserve"]
        safe += -1.42 * s.vx * barrier * profile["centerBias"]
    if ratio > 0.88 and x_err * s.vx > 0:
        safe += -outward * params["maxAcc"] * 0.48 * profile["landingGuard"]
    return clamp(safe, -params["maxAcc"], params["maxAcc"])


def trainer_policy(s: State, target: int, params: Dict[str, float], profile: Dict[str, float], advice: Optional[Dict[str, float]] = None, features: Optional[Dict[str, float]] = None) -> float:
    angle_norm, speed_norm, _, _ = closeness(s, target)
    edge = support_edge_ratio(s, params)
    center_a = center_return(s, params, profile["centerBias"])
    pump_a = energy_pump(s, target, params, profile)
    align_a = align_acceleration(s, target, params, profile)
    lqr_a = lqr_acceleration(s, target, params)

    capture_angle = (0.58 if target == 3 else 0.40) / math.sqrt(max(0.70, profile["captureConservatism"]))
    capture_speed = (5.1 if target == 3 else 2.2) / (0.85 + 0.15 * profile["speedDamping"])

    if target == 0:
        raw = 0.58 * align_a + 0.30 * center_a + 0.12 * pump_a
    elif angle_norm < capture_angle and speed_norm < capture_speed:
        phase_brake = clamp(4.9 * support_phase_power(s, params) * profile["brakeGain"], -0.45 * params["maxAcc"], 0.45 * params["maxAcc"])
        raw = 0.58 * lqr_a + 0.18 * align_a + 0.12 * phase_brake + 0.08 * center_a + 0.04 * pump_a
    elif angle_norm < (1.25 if target == 3 else 0.92):
        raw = 0.26 * pump_a + 0.34 * align_a + 0.27 * lqr_a + 0.13 * center_a
    else:
        raw = 0.72 * pump_a + 0.12 * align_a + 0.16 * center_a

    if edge > 0.62:
        mix = clamp((edge - 0.62) / 0.31, 0.0, 1.0) * 0.54 * profile["edgeReserve"]
        raw = (1.0 - min(0.75, mix)) * raw + min(0.75, mix) * center_a
    if advice and features:
        raw = apply_phase_advice(s, target, params, profile, raw, features, advice)
    return apply_track_safety(s, raw, params, profile)


def quantize_slider_value(name: str, value: float) -> float:
    if name == "maxAcc":
        return clamp(round_to(value, 0.5, 20.0), *PARAM_RANGES[name])
    if name in ("friction", "windAmp"):
        return clamp(round_to(value, 0.005, 0.03), *PARAM_RANGES[name])
    if name == "g":
        return clamp(round_to(value, 0.01, 9.0), *PARAM_RANGES[name])
    return value


def params_from_bucket_parts(parts: Dict[str, Any], rng: Optional[random.Random] = None, jitter: bool = True) -> Dict[str, float]:
    p = dict(DEFAULT_PARAMS)
    rng = rng or random.Random(0)
    max_acc = float(parts.get("maxAcc", 20.0))
    gravity = float(parts.get("g", 9.0))
    friction = float(parts.get("friction", 0.03))
    wind = float(parts.get("windAmp", 0.03))
    if jitter:
        max_acc += rng.uniform(-2.0, 2.0)
        gravity += rng.uniform(-0.28, 0.28)
        friction += rng.uniform(-0.018, 0.018)
        wind += rng.uniform(-0.018, 0.018)
    p["maxAcc"] = quantize_slider_value("maxAcc", max_acc)
    p["g"] = quantize_slider_value("g", gravity)
    p["friction"] = quantize_slider_value("friction", friction)
    p["windAmp"] = quantize_slider_value("windAmp", wind)
    return p


def params_from_specialty(spec: Dict[str, Any], rng: random.Random, jitter: bool = True) -> Dict[str, float]:
    p = dict(DEFAULT_PARAMS)
    for name in ("maxAcc", "g", "friction", "windAmp"):
        lo, hi = spec[name]
        # Triangular sampling emphasizes the middle of each named regime while
        # still exercising the edges.  This is closer to users moving sliders
        # around a setting than to full-random parameter generation.
        v = rng.triangular(float(lo), float(hi), 0.5 * (float(lo) + float(hi)))
        if jitter:
            if name == "maxAcc": v += rng.uniform(-0.50, 0.50)
            elif name == "g": v += rng.uniform(-0.035, 0.035)
            else: v += rng.uniform(-0.006, 0.006)
        p[name] = quantize_slider_value(name, v)
    return p


def specialty_match_score(spec: Dict[str, Any], parts: Dict[str, Any]) -> float:
    score = 1.0
    for name in ("maxAcc", "g", "friction", "windAmp"):
        lo, hi = spec[name]
        v = float(parts.get(name, DEFAULT_PARAMS[name]))
        width = max(1e-6, float(hi) - float(lo))
        center = 0.5 * (float(lo) + float(hi))
        radius = 0.5 * width
        d = abs(v - center) / max(radius, 1e-6)
        # Keep nearby bucket reuse, but strongly prefer records inside a scenario.
        score *= max(0.0, 1.0 - 0.38 * max(0.0, d - 1.0)) / (1.0 + 0.24 * d * d)
        if d > 2.35:
            return 0.0
    return score


def choose_specialty(rng: random.Random, specialty_stats: Optional[Dict[str, Dict[str, float]]] = None, names: Optional[List[str]] = None) -> Dict[str, Any]:
    allowed = [SPECIALTY_BY_NAME[n] for n in (names or list(SPECIALTY_BY_NAME)) if n in SPECIALTY_BY_NAME]
    if not allowed:
        allowed = SPECIALTY_CASES
    weights = []
    for spec in allowed:
        st = (specialty_stats or {}).get(spec["name"], {})
        seen = float(st.get("episodes", 0.0))
        fail = float(st.get("failures", 0.0))
        edge = float(st.get("edge", 0.0))
        slow = float(st.get("slow", 0.0))
        # Underperforming and under-sampled scenarios get more attention.
        fail_rate = fail / max(1.0, seen)
        edge_rate = edge / max(1.0, seen)
        slow_rate = slow / max(1.0, seen)
        scarcity = 1.0 / math.sqrt(1.0 + seen / 45.0)
        weights.append(float(spec.get("weight", 1.0)) * (0.70 + scarcity + 1.8 * fail_rate + 0.65 * edge_rate + 0.40 * slow_rate))
    return rng.choices(allowed, weights=weights, k=1)[0]


def infer_specialty_name(params_or_parts: Dict[str, Any]) -> str:
    best_name = "default_play"
    best = -1.0
    for spec in SPECIALTY_CASES:
        sc = specialty_match_score(spec, params_or_parts) * float(spec.get("weight", 1.0))
        if sc > best:
            best = sc
            best_name = spec["name"]
    return best_name


def sample_specialty_replay_case(rng: random.Random, db: Dict[str, Any], target_ids: List[int], spec: Dict[str, Any], jitter: bool = True) -> Optional[Tuple[int, Dict[str, float]]]:
    candidates = []
    weights = []
    allowed = set(target_ids)
    for rec in db.get("profiles_by_key", {}).values():
        parts = rec.get("parts") or {}
        target = int(parts.get("targetId", -1))
        if target not in allowed:
            continue
        match = specialty_match_score(spec, parts)
        if match <= 0:
            continue
        candidates.append((target, parts))
        weights.append(replay_weight(rec) * (0.20 + 1.80 * match))
    if not candidates:
        return None
    target, parts = rng.choices(candidates, weights=weights, k=1)[0]
    return target, params_from_bucket_parts(parts, rng, jitter=jitter)


def replay_weight(rec: Dict[str, Any]) -> float:
    stats = rec.get("stats", {}) or {}
    failures = sum(float(v) for k, v in stats.items() if k != "stable")
    stable = float(stats.get("stable", 0.0))
    visits = max(1.0, float(rec.get("visits", 0) or 0))
    # Revisit buckets that have failures but avoid locking onto one noisy bucket forever.
    return max(0.20, 1.0 + failures - 0.65 * stable) / math.sqrt(visits)


def sample_replay_case(rng: random.Random, db: Dict[str, Any], target_ids: List[int], jitter: bool = True) -> Optional[Tuple[int, Dict[str, float]]]:
    candidates = []
    weights = []
    allowed = set(target_ids)
    for rec in db.get("profiles_by_key", {}).values():
        parts = rec.get("parts") or {}
        target = int(parts.get("targetId", -1))
        if target not in allowed:
            continue
        candidates.append((target, parts))
        weights.append(replay_weight(rec))
    if not candidates:
        return None
    target, parts = rng.choices(candidates, weights=weights, k=1)[0]
    return target, params_from_bucket_parts(parts, rng, jitter=jitter)


def practical_case(rng: random.Random, target_ids: List[int], jitter: bool = True) -> Tuple[int, Dict[str, float]]:
    target = rng.choices(target_ids, weights=[0.16 if t == 0 else (0.36 if t == 3 else 0.24) for t in target_ids], k=1)[0]
    base = dict(DEFAULT_PARAMS)
    base.update(rng.choice(PRACTICAL_BUCKETS))
    if jitter:
        base["maxAcc"] = clamp(base["maxAcc"] + rng.uniform(-1.25, 1.25), *PARAM_RANGES["maxAcc"])
        base["g"] = clamp(base["g"] + rng.uniform(-0.18, 0.18), *PARAM_RANGES["g"])
        base["friction"] = clamp(base["friction"] + rng.uniform(-0.010, 0.010), *PARAM_RANGES["friction"])
        base["windAmp"] = clamp(base["windAmp"] + rng.uniform(-0.010, 0.010), *PARAM_RANGES["windAmp"])
    return target, base


def sample_training_case(
    rng: random.Random,
    db: Dict[str, Any],
    target_ids: List[int],
    explore_rate: float = 0.0,
    jitter: bool = True,
    specialty_stats: Optional[Dict[str, Dict[str, float]]] = None,
    specialty_names: Optional[List[str]] = None,
    replay_rate: float = 0.58,
) -> Tuple[int, Dict[str, float], str]:
    # Specialized curriculum is the default.  We select a named regime first,
    # then improve nearby existing buckets whenever possible.  Full random
    # exploration is available but disabled by default because it dilutes the
    # ability to specialize in difficult web-facing regimes.
    spec = choose_specialty(rng, specialty_stats, specialty_names)
    if db.get("profiles_by_key") and rng.random() < clamp(replay_rate, 0.0, 0.95):
        replay = sample_specialty_replay_case(rng, db, target_ids, spec, jitter=jitter)
        if replay:
            return replay[0], replay[1], f"replay:{spec['name']}"
    target = rng.choices(target_ids, weights=[0.12 if t == 0 else (0.42 if t == 3 else 0.23) for t in target_ids], k=1)[0]
    if rng.random() < clamp(explore_rate, 0.0, 0.25):
        # Still constrained to the chosen specialty, not full-slider random.
        params = params_from_specialty(spec, rng, jitter=jitter)
        return target, params, f"specialty-explore:{spec['name']}"
    params = params_from_specialty(spec, rng, jitter=jitter)
    return target, params, f"specialty:{spec['name']}"


def random_params(rng: random.Random) -> Dict[str, float]:
    p = dict(DEFAULT_PARAMS)
    # Mixture sampling: more samples near practical values, but still covers the full sliders.
    if rng.random() < 0.65:
        p["maxAcc"] = rng.choice([rng.uniform(12, 30), rng.uniform(20, 42), rng.uniform(1, 60)])
        p["g"] = clamp(rng.gauss(9.0, 1.8), 0.0, 12.0)
        p["friction"] = clamp(abs(rng.gauss(0.045, 0.045)), 0.0, 1.0)
        p["windAmp"] = clamp(abs(rng.gauss(0.05, 0.08)), 0.0, 1.0)
    else:
        p["maxAcc"] = rng.uniform(*PARAM_RANGES["maxAcc"])
        p["g"] = rng.uniform(*PARAM_RANGES["g"])
        p["friction"] = rng.uniform(*PARAM_RANGES["friction"])
        p["windAmp"] = rng.uniform(*PARAM_RANGES["windAmp"])
    return p


def random_state(rng: random.Random, params: Dict[str, float], mode: str = "mixed", target: Optional[int] = None) -> State:
    left, right, center, half = support_bounds(params)
    # Browser-facing initial conditions: users mostly switch targets, drag the
    # phase point, or recover near the rails.  We still keep a small mixed mode,
    # but the curriculum no longer measures success mainly on extreme random
    # states that the page rarely produces.
    if mode in ("browser-target", "near-target"):
        x = rng.uniform(center - 0.46 * half, center + 0.46 * half)
        vx = rng.uniform(-0.95, 0.95)
        target_id = target if target in TARGETS else rng.choice([1, 2, 3])
        a1, a2 = TARGETS[target_id]
        spread = 0.58 if mode == "browser-target" else 0.70
        th1 = wrap_angle(a1 + rng.uniform(-spread, spread))
        th2 = wrap_angle(a2 + rng.uniform(-spread, spread))
        om1, om2 = rng.uniform(-2.55, 2.55), rng.uniform(-2.55, 2.55)
    elif mode == "browser-drag":
        x = rng.uniform(center - 0.50 * half, center + 0.50 * half)
        vx = rng.uniform(-1.05, 1.05)
        # Dragging changes angles but preserves velocities; use broad angles with
        # moderate angular rates rather than completely random high rates.
        th1 = rng.uniform(-PI, PI)
        th2 = rng.uniform(-PI, PI)
        om1, om2 = rng.uniform(-2.35, 2.35), rng.uniform(-2.35, 2.35)
    elif mode == "edge-recovery":
        side = -1 if rng.random() < 0.5 else 1
        x = center + side * rng.uniform(0.66 * half, 0.93 * half)
        vx = side * rng.uniform(0.10, 1.45) if rng.random() < 0.70 else rng.uniform(-0.75, 0.75)
        target_id = target if target in TARGETS else rng.choice([1, 2, 3])
        a1, a2 = TARGETS[target_id]
        if rng.random() < 0.62:
            th1 = wrap_angle(a1 + rng.uniform(-0.85, 0.85))
            th2 = wrap_angle(a2 + rng.uniform(-0.85, 0.85))
        else:
            th1, th2 = rng.uniform(-0.75, 0.75), rng.uniform(-0.75, 0.75)
        om1, om2 = rng.uniform(-2.8, 2.8), rng.uniform(-2.8, 2.8)
    elif mode == "near-down":
        x = rng.uniform(center - 0.58 * half, center + 0.58 * half)
        vx = rng.uniform(-1.25, 1.25)
        th1, th2 = rng.uniform(-0.55, 0.55), rng.uniform(-0.55, 0.55)
        om1, om2 = rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8)
    else:
        x = rng.uniform(center - 0.70 * half, center + 0.70 * half)
        vx = rng.uniform(-1.55, 1.55)
        th1 = rng.uniform(-PI, PI)
        th2 = rng.uniform(-PI, PI)
        om1, om2 = rng.uniform(-3.6, 3.6), rng.uniform(-3.6, 3.6)
    return State(x, vx, th1, th2, om1, om2)


def classify_failure(samples: List[Tuple[float, State, Dict[str, float]]], target: int, params: Dict[str, float], reached_near: bool) -> Tuple[str, float, Dict[str, float]]:
    # Use the worst/most informative recent sample.
    worst = samples[-1]
    for item in reversed(samples[-90:]):
        _, s, f = item
        if f["edgeRatio"] > 0.88 and f["outward"]:
            return "edge-risk", clamp((f["edgeRatio"] - 0.78) / 0.18, 0.6, 2.0), f
        if f["angleNorm"] < (0.64 if target == 3 else 0.46) and f["speedNorm"] > (5.5 if target == 3 else 3.8) and f["radial"] > 0.12:
            return "fast-flyby", clamp(f["speedNorm"] / (7.0 if target == 3 else 5.0), 0.6, 2.0), f
        if f["angleNorm"] < (0.88 if target == 3 else 0.62) and f["energyDelta"] > 0.18 and f["speedNorm"] > (4.8 if target == 3 else 3.6):
            return "wrong-energy", clamp(f["energyDelta"] / 0.35, 0.5, 2.0), f
    _, _, last_f = worst
    if reached_near:
        return "slow-capture", 0.85, last_f
    return "slow-arrival", 0.85, last_f


def run_episode(rng: random.Random, db: Dict[str, Any], target: int, params: Dict[str, float], initial: State, seconds: float, dt: float, learn: bool = True) -> Dict[str, Any]:
    profile = profile_from_db(db, params, target)
    s = initial.copy()
    stable_time = 0.0
    capture_time = 0.0
    reached_near = False
    best_angle = 999.0
    samples: List[Tuple[float, State, Dict[str, float]]] = []
    t = 0.0
    steps = int(seconds / dt)
    command = 0.0
    for i in range(steps):
        f = phase_features(s, target, params)
        samples.append((t, s.copy(), f))
        best_angle = min(best_angle, f["angleNorm"])
        handoff_ready = (
            f["angleNorm"] < CAPTURE_ANGLE[target]
            and f["speedNorm"] < CAPTURE_SPEED[target]
            and f["edgeRatio"] < 0.90
            and f["radial"] < (3.20 if target == 3 else 2.45)
        )
        if handoff_ready:
            reached_near = True
            capture_time += dt
        else:
            capture_time = max(0.0, capture_time - 2.0 * dt)
        if capture_time > HANDOFF_DWELL[target]:
            if learn:
                update_profile(db, target, params, "stable", 0.24)
                update_phase_rule(db, target, params, f, "stable", 0.24)
            return {"success": True, "time": t, "event": "handoff", "bestAngle": best_angle, "reachedNear": True}
        practical_stable = (
            f["angleNorm"] < SUCCESS_ANGLE[target]
            and f["speedNorm"] < SUCCESS_SPEED[target]
            and f["edgeRatio"] < 0.92
            and f["radial"] < (3.40 if target == 3 else 2.80)
        )
        tight_stable = f["angleNorm"] < (0.070 if target == 3 else 0.055) and f["speedNorm"] < (0.30 if target == 3 else 0.24) and f["edgeRatio"] < 0.80
        stable = practical_stable or tight_stable
        stable_time = stable_time + dt if stable else max(0.0, stable_time - 3 * dt)
        if stable_time > SUCCESS_DWELL[target]:
            if learn:
                update_profile(db, target, params, "stable", 0.45 if tight_stable else 0.35)
                update_phase_rule(db, target, params, f, "stable", 0.45 if tight_stable else 0.35)
            return {"success": True, "time": t, "event": "settled" if practical_stable else "stable", "bestAngle": best_angle, "reachedNear": True}

        # Online-looking but compact: only update phase rules on sparse bad events.
        if i % 18 == 0:
            if f["edgeRatio"] > 0.90 and f["outward"]:
                if learn:
                    update_phase_rule(db, target, params, f, "edge-risk", clamp((f["edgeRatio"] - 0.80) / 0.18, 0.5, 1.8))
            elif f["angleNorm"] < (0.64 if target == 3 else 0.44) and f["speedNorm"] > (5.5 if target == 3 else 3.8) and f["radial"] > 0.12:
                if learn:
                    update_phase_rule(db, target, params, f, "fast-flyby", clamp(f["speedNorm"] / (7.0 if target == 3 else 5.0), 0.5, 1.8))

        advice = get_phase_advice(db, target, params, f)
        raw = trainer_policy(s, target, params, profile, advice, f)
        max_delta = max(6.0, params["maxAcc"] * 1.2) * dt
        command = clamp(0.96 * raw + 0.04 * command, command - max_delta, command + max_delta)
        s = rk4_step(s, command, params, t, dt, use_wind=True)
        t += dt
        if not all(math.isfinite(v) for v in (s.x, s.vx, s.th1, s.th2, s.om1, s.om2)):
            break
    event, severity, f = classify_failure(samples, target, params, reached_near)
    if learn:
        update_profile(db, target, params, event, severity)
        update_phase_rule(db, target, params, f, event, severity)
    return {"success": False, "time": seconds, "event": event, "bestAngle": best_angle, "reachedNear": reached_near}



# ---------------------------------------------------------------------------
# Hard-case mining and local profile surgery
# ---------------------------------------------------------------------------
# When ordinary curriculum training plateaus, most additional random episodes
# only add noisy phase rules.  The functions below mine deterministic failures
# from the browser-like evaluation distribution, then try small, bounded profile
# changes on the exact failing case before accepting them.  This makes continued
# training much more like regression fixing: every saved update must improve a
# concrete difficult scenario, and the global validation gate still prevents
# overfitting from replacing the published database.

EVENT_PENALTY = {
    "settled": 0.0,
    "stable": 0.0,
    "slow-capture": 4.0,
    "slow-arrival": 8.0,
    "wrong-energy": 6.0,
    "fast-flyby": 7.0,
    "edge-risk": 10.0,
}


def state_to_dict(s: State) -> Dict[str, float]:
    return {"x": s.x, "vx": s.vx, "th1": s.th1, "th2": s.th2, "om1": s.om1, "om2": s.om2}


def state_from_dict(d: Dict[str, float]) -> State:
    return State(float(d["x"]), float(d["vx"]), float(d["th1"]), float(d["th2"]), float(d["om1"]), float(d["om2"]))


def jitter_state_for_browser_case(rng: random.Random, s: State, params: Dict[str, float], scale: float = 1.0) -> State:
    left, right, center, half = support_bounds(params)
    return State(
        clamp(s.x + rng.uniform(-0.035, 0.035) * half * scale, left, right),
        s.vx + rng.uniform(-0.06, 0.06) * scale,
        wrap_angle(s.th1 + rng.uniform(-0.045, 0.045) * scale),
        wrap_angle(s.th2 + rng.uniform(-0.045, 0.045) * scale),
        s.om1 + rng.uniform(-0.10, 0.10) * scale,
        s.om2 + rng.uniform(-0.10, 0.10) * scale,
    )


def case_objective(result: Dict[str, Any], seconds: float) -> float:
    # Stability-focused target-3 objective.  The main reward is simply making a
    # safe handoff / settled capture under a wider set of browser-like states.
    # Speed is now a weak tie-breaker only; edge-risk and wrong-energy failures
    # are penalized heavily because they hurt practical success.
    success = 1.0 if result.get("success") else 0.0
    reached = 1.0 if result.get("reachedNear") else 0.0
    best = float(result.get("bestAngle", 9.0))
    t = float(result.get("time", seconds))
    event = str(result.get("event", "timeout"))
    event_penalty = EVENT_PENALTY.get(event, 10.0)
    if event == "edge-risk":
        event_penalty += 20.0
    elif event == "wrong-energy":
        event_penalty += 12.0
    elif event == "fast-flyby":
        event_penalty += 8.0
    # Weak time penalty: late success is much better than fast failure.
    return 245.0 * success + 28.0 * reached - 5.0 * best - 2.2 * t * success - 4.0 * t * (1.0 - success) - event_penalty


def collect_hard_cases(db: Dict[str, Any], episodes: int, seconds: float, dt: float, seed: int, target_ids: List[int], specialty_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    names = [n for n in (specialty_names or [c["name"] for c in SPECIALTY_CASES]) if n in SPECIALTY_BY_NAME]
    if not names:
        names = ["default_play"]
    modes = ["browser-target", "near-down", "edge-recovery", "browser-drag", "mixed"]
    mode_weights = [0.36, 0.18, 0.20, 0.11, 0.15]
    hard: List[Dict[str, Any]] = []
    for i in range(max(0, episodes)):
        name = names[i % len(names)]
        spec = SPECIALTY_BY_NAME[name]
        target = rng.choices(target_ids, weights=[0.10 if t == 0 else (0.44 if t == 3 else 0.23) for t in target_ids], k=1)[0]
        params = params_from_specialty(spec, rng, jitter=True)
        mode = rng.choices(modes, weights=mode_weights, k=1)[0]
        initial = random_state(rng, params, mode, target)
        result = run_episode(rng, db, target, params, initial, seconds, dt, learn=False)
        # Keep every failure and risky success as a regression test.  Slow but
        # safe successes are no longer treated as hard cases; this build
        # prioritizes success rate and capture stability over speed.
        if (not result.get("success")) or result.get("event") in ("edge-risk", "wrong-energy", "fast-flyby"):
            hard.append({
                "specialty": name,
                "target": target,
                "params": {k: float(params[k]) for k in DEFAULT_PARAMS.keys() if k in params},
                "initial": state_to_dict(initial),
                "mode": mode,
                "event": result.get("event"),
                "bestAngle": float(result.get("bestAngle", 9.0)),
                "time": float(result.get("time", seconds)),
                "score": case_objective(result, seconds),
            })
    # Highest priority first: unsafe/uncaptured cases, then poor case objective.
    event_rank = {"edge-risk": 0, "wrong-energy": 1, "fast-flyby": 2, "slow-arrival": 3, "slow-capture": 4, "settled": 5, "stable": 5}
    hard.sort(key=lambda c: (event_rank.get(str(c.get("event")), 6), c.get("score", -999.0)))
    return hard


def profile_mutation_vectors(event: str, target: int, params: Dict[str, float]) -> List[Dict[str, float]]:
    high_wind = abs(float(params.get("windAmp", 0.0))) > 0.18
    low_g = float(params.get("g", 9.0)) < 5.5
    high_f = float(params.get("friction", 0.0)) > 0.22
    low_acc = float(params.get("maxAcc", 20.0)) < 14.0
    high_acc = float(params.get("maxAcc", 20.0)) > 42.0
    base: List[Dict[str, float]] = []
    if event == "edge-risk":
        base += [
            {"edgeReserve": 0.13, "centerBias": 0.11, "landingGuard": 0.12, "reserveBias": 0.10, "authorityScale": -0.050, "speedDamping": 0.030},
            {"centerBias": 0.16, "reserveBias": 0.11, "edgeReserve": 0.08, "authorityScale": -0.035},
            {"landingGuard": 0.16, "speedDamping": 0.07, "brakeGain": 0.04, "authorityScale": -0.050},
        ]
    elif event == "wrong-energy":
        base += [
            {"brakeGain": 0.10, "speedDamping": 0.09, "captureConservatism": 0.06, "authorityScale": -0.045, "landingGuard": 0.05},
            {"captureConservatism": 0.11, "brakeGain": 0.05, "authorityScale": -0.060, "retryBoost": 0.04},
        ]
    elif event == "fast-flyby":
        base += [
            {"brakeGain": 0.13, "speedDamping": 0.11, "landingGuard": 0.07, "authorityScale": -0.040},
            {"captureConservatism": 0.10, "brakeGain": 0.08, "speedDamping": 0.07, "authorityScale": -0.030},
        ]
    elif event == "slow-capture":
        base += [
            {"brakeGain": 0.07, "speedDamping": 0.06, "landingGuard": 0.05, "centerBias": 0.025},
            {"captureConservatism": -0.04, "authorityScale": 0.035, "brakeGain": 0.025, "retryBoost": 0.030},
        ]
    else:  # slow-arrival / timeout
        base += [
            {"authorityScale": 0.090, "captureConservatism": -0.065, "retryBoost": 0.060, "brakeGain": -0.025},
            {"authorityScale": 0.055, "reserveBias": 0.045, "centerBias": 0.030, "captureConservatism": -0.035},
            {"retryBoost": 0.080, "authorityScale": 0.045, "speedDamping": -0.030},
        ]
    if high_wind:
        base.append({"edgeReserve": 0.10, "centerBias": 0.08, "landingGuard": 0.10, "speedDamping": 0.06, "authorityScale": -0.035, "reserveBias": 0.08})
    if low_g:
        base.append({"authorityScale": 0.055, "captureConservatism": -0.040, "brakeGain": 0.035, "speedDamping": 0.025})
    if high_f:
        base.append({"authorityScale": 0.070, "captureConservatism": -0.035, "retryBoost": 0.045, "brakeGain": -0.020})
    if low_acc:
        base.append({"authorityScale": 0.115, "captureConservatism": -0.080, "retryBoost": 0.070, "reserveBias": 0.035})
    if high_acc:
        base.append({"authorityScale": -0.045, "edgeReserve": 0.075, "brakeGain": 0.080, "speedDamping": 0.070, "landingGuard": 0.070})
    # Safe generic fallback probes.
    base += [
        {"authorityScale": 0.040},
        {"brakeGain": 0.045, "speedDamping": 0.040},
        {"edgeReserve": 0.050, "centerBias": 0.050, "reserveBias": 0.035},
    ]
    return base


def ensure_profile_record(db: Dict[str, Any], target: int, params: Dict[str, float]) -> Dict[str, Any]:
    key = bucket_key(target, params)
    return db["profiles_by_key"].setdefault(key, {
        "key": key,
        "parts": bucket_parts(target, params),
        "delta": neutral_delta(),
        "stats": {},
        "visits": 0,
        "source": "python-hard-miner",
        "updatedAt": int(time.time() * 1000),
    })


def add_delta_patch(delta: Dict[str, float], patch: Dict[str, float], scale: float) -> Dict[str, float]:
    out = dict(delta)
    for k, v in patch.items():
        if k not in PROFILE_FIELDS:
            continue
        out[k] = clamp(float(out.get(k, 0.0)) + float(v) * scale, -0.35, 0.42)
    return out


def optimize_hard_case_profile(db: Dict[str, Any], case: Dict[str, Any], seconds: float, dt: float, rng: random.Random, probes: int = 16) -> bool:
    target = int(case["target"])
    params = dict(DEFAULT_PARAMS)
    params.update({k: float(v) for k, v in (case.get("params") or {}).items()})
    initial = jitter_state_for_browser_case(rng, state_from_dict(case["initial"]), params, scale=0.50)
    probe_seconds = min(seconds, 6.5)
    probe_dt = max(dt, 0.0125)
    before = run_episode(rng, db, target, params, initial, probe_seconds, probe_dt, learn=False)
    before_score = case_objective(before, probe_seconds)
    rec = ensure_profile_record(db, target, params)
    original_delta = dict(rec.get("delta", neutral_delta()))
    event = str(before.get("event") or case.get("event") or "slow-arrival")
    vectors = profile_mutation_vectors(event, target, params)
    rng.shuffle(vectors)
    # Include both moderate and aggressive scales; exact-case acceptance keeps it safe.
    trials: List[Tuple[float, Dict[str, float]]] = []
    for vec in vectors[:max(1, probes)]:
        trials.append((1.00, vec))
        if len(trials) < probes:
            trials.append((0.55, vec))
        if len(trials) < probes and before.get("event") in ("edge-risk", "slow-arrival", "wrong-energy"):
            trials.append((1.35, vec))
        if len(trials) >= probes:
            break
    best_delta = original_delta
    best_result = before
    best_score = before_score
    improved = False
    for scale, patch in trials[:probes]:
        rec["delta"] = add_delta_patch(original_delta, patch, scale)
        result = run_episode(rng, db, target, params, initial, probe_seconds, probe_dt, learn=False)
        score = case_objective(result, probe_seconds)
        if score > best_score + 0.20:
            best_score = score
            best_delta = dict(rec["delta"])
            best_result = result
            improved = True
            if result.get("success") and not before.get("success"):
                break
    rec["delta"] = best_delta if improved else original_delta
    if improved:
        rec["visits"] = int(rec.get("visits", 0) or 0) + 1
        rec.setdefault("stats", {})["hard-surgery"] = rec.setdefault("stats", {}).get("hard-surgery", 0) + 1
        rec.setdefault("stats", {})[str(best_result.get("event", "improved"))] = rec.setdefault("stats", {}).get(str(best_result.get("event", "improved")), 0) + 1
        rec["updatedAt"] = int(time.time() * 1000)
        # Also add one phase-rule hint from the new trajectory when the case is
        # still imperfect, so the browser receives local context advice.
        if not best_result.get("success"):
            replay = run_episode(rng, db, target, params, initial, probe_seconds, probe_dt, learn=True)
            _ = replay
    return improved

def evaluation_score(summary: Dict[str, Any]) -> float:
    # Balanced practical score: macro-average across named specialties, plus a
    # floor term so a single weak regime cannot be hidden by easy defaults.
    success = summary.get("macroSuccessRate", summary.get("successRate", 0.0))
    capture = summary.get("macroCaptureRate", summary.get("captureRate", 0.0))
    worst_success = summary.get("worstSpecialtySuccess", success)
    mean_best = summary.get("macroMeanBestAngle", summary.get("meanBestAngle", 9.0))
    mean_time = summary.get("macroMeanSuccessTime", summary.get("meanSuccessTime", summary.get("seconds", 20.0)))
    edge = summary.get("macroEdgeRiskRate", summary.get("edgeRiskRate", 0.0))
    wrong = summary.get("macroWrongEnergyRate", summary.get("wrongEnergyRate", 0.0))
    timeout = summary.get("macroTimeoutRate", summary.get("timeoutRate", 0.0))
    spread = summary.get("specialtySuccessSpread", 0.0)
    return (
        185.0 * success
        + 72.0 * worst_success
        + 44.0 * capture
        - 3.5 * mean_best
        - 24.0 * edge
        - 16.0 * wrong
        - 0.55 * mean_time * max(0.20, success)
        - 10.0 * timeout
        - 22.0 * spread
    )


def _empty_eval_stats() -> Dict[str, Any]:
    return {"success": 0, "capture": 0, "events": {}, "bestAngles": [], "successTimes": [], "episodes": 0}


def _accumulate_eval(stats: Dict[str, Any], result: Dict[str, Any], seconds: float) -> None:
    stats["episodes"] += 1
    stats["success"] += 1 if result["success"] else 0
    stats["capture"] += 1 if result.get("reachedNear") or result["success"] or result["event"] in ("slow-capture", "handoff", "settled", "stable") else 0
    stats["events"][result["event"]] = stats["events"].get(result["event"], 0) + 1
    stats["bestAngles"].append(float(result.get("bestAngle", 9.0)))
    if result["success"]:
        stats["successTimes"].append(float(result.get("time", seconds)))


def _finalize_eval_stats(stats: Dict[str, Any], seconds: float) -> Dict[str, Any]:
    n = max(1, int(stats.get("episodes", 0)))
    mean_success_time = sum(stats["successTimes"]) / max(1, len(stats["successTimes"])) if stats["successTimes"] else seconds
    return {
        "episodes": int(stats.get("episodes", 0)),
        "successRate": stats["success"] / n,
        "captureRate": stats["capture"] / n,
        "meanSuccessTime": mean_success_time,
        "meanBestAngle": sum(stats["bestAngles"]) / max(1, len(stats["bestAngles"])),
        "edgeRiskRate": stats["events"].get("edge-risk", 0) / n,
        "wrongEnergyRate": stats["events"].get("wrong-energy", 0) / n,
        "timeoutRate": (n - stats["success"]) / n,
        "events": dict(stats["events"]),
    }


def evaluate_db(db: Dict[str, Any], episodes: int, seconds: float, dt: float, seed: int, target_ids: List[int], specialty_names: Optional[List[str]] = None) -> Dict[str, Any]:
    rng = random.Random(seed)
    overall = _empty_eval_stats()
    names = [n for n in (specialty_names or [c["name"] for c in SPECIALTY_CASES]) if n in SPECIALTY_BY_NAME]
    if not names:
        names = ["default_play"]
    by_stats = {name: _empty_eval_stats() for name in names}
    total = max(0, episodes)
    modes = ["browser-target", "near-down", "edge-recovery", "browser-drag", "mixed"]
    mode_weights = [0.36, 0.18, 0.20, 0.11, 0.15]
    for i in range(total):
        name = names[i % len(names)]
        spec = SPECIALTY_BY_NAME[name]
        target = rng.choices(target_ids, weights=[0.10 if t == 0 else (0.44 if t == 3 else 0.23) for t in target_ids], k=1)[0]
        params = params_from_specialty(spec, rng, jitter=True)
        mode = rng.choices(modes, weights=mode_weights, k=1)[0]
        result = run_episode(rng, db, target, params, random_state(rng, params, mode, target), seconds, dt, learn=False)
        _accumulate_eval(overall, result, seconds)
        _accumulate_eval(by_stats[name], result, seconds)
    out = _finalize_eval_stats(overall, seconds)
    out["episodes"] = total
    out["seconds"] = seconds
    by_specialty = {name: _finalize_eval_stats(st, seconds) for name, st in by_stats.items()}
    out["bySpecialty"] = by_specialty
    if by_specialty:
        vals = list(by_specialty.values())
        out["macroSuccessRate"] = sum(v["successRate"] for v in vals) / len(vals)
        out["macroCaptureRate"] = sum(v["captureRate"] for v in vals) / len(vals)
        out["macroMeanBestAngle"] = sum(v["meanBestAngle"] for v in vals) / len(vals)
        out["macroMeanSuccessTime"] = sum(v["meanSuccessTime"] for v in vals) / len(vals)
        out["macroEdgeRiskRate"] = sum(v["edgeRiskRate"] for v in vals) / len(vals)
        out["macroWrongEnergyRate"] = sum(v["wrongEnergyRate"] for v in vals) / len(vals)
        out["macroTimeoutRate"] = sum(v["timeoutRate"] for v in vals) / len(vals)
        out["worstSpecialtySuccess"] = min(v["successRate"] for v in vals)
        out["specialtySuccessSpread"] = max(v["successRate"] for v in vals) - min(v["successRate"] for v in vals)
    else:
        out["macroSuccessRate"] = out["successRate"]
        out["macroCaptureRate"] = out["captureRate"]
        out["worstSpecialtySuccess"] = out["successRate"]
        out["specialtySuccessSpread"] = 0.0
    out["score"] = evaluation_score(out)
    return out


def format_eval(summary: Dict[str, Any]) -> str:
    success = summary.get("macroSuccessRate", summary.get("successRate", 0.0))
    capture = summary.get("macroCaptureRate", summary.get("captureRate", 0.0))
    settle = summary.get("macroMeanSuccessTime", summary.get("meanSuccessTime", summary.get("seconds", 0.0)))
    mean_best = summary.get("macroMeanBestAngle", summary.get("meanBestAngle", 9.0))
    edge = summary.get("macroEdgeRiskRate", summary.get("edgeRiskRate", 0.0))
    worst = summary.get("worstSpecialtySuccess", success)
    return (f"macro-success {success:.1%} | worst-specialty {worst:.1%} | capture {capture:.1%} | "
            f"settle {settle:.2f}s | mean-best-angle {mean_best:.3f} | edge {edge:.1%} | "
            f"score {summary['score']:.2f}")


def empty_db() -> Dict[str, Any]:
    return {"profiles_by_key": {}, "phase_by_key": {}}


def load_db(path: Path) -> Dict[str, Any]:
    db = empty_db()
    if not path.exists():
        return db
    data = json.loads(path.read_text(encoding="utf-8"))
    for rec in data.get("profiles", []):
        if rec.get("key"):
            rec.setdefault("delta", neutral_delta())
            rec.setdefault("stats", {})
            rec.setdefault("visits", 0)
            db["profiles_by_key"][rec["key"]] = rec
    for rule in data.get("phaseRules", []):
        if rule.get("key"):
            rule.setdefault("action", default_action())
            rule.setdefault("stats", {})
            rule.setdefault("visits", 0)
            db["phase_by_key"][rule["key"]] = rule
    return db


def export_db(db: Dict[str, Any], path: Path, episodes: int, seed: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    profiles = sorted(db["profiles_by_key"].values(), key=lambda r: (r.get("key", "")))
    rules = sorted(db["phase_by_key"].values(), key=lambda r: (r.get("targetId", 0), -r.get("visits", 0), r.get("key", "")))
    data = {
        "version": 2,
        "generatedAt": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "generator": "train_ai.py",
        "episodes": episodes,
        "seed": seed,
        "description": "Stability/success-focused adaptive profiles and phase-rule advice for target state 3 in the default index ±10% range. Browser online learning is disabled in this build; player-selected physics parameters are never modified.",
        "browserOnlineLearning": False,
        "trainingMode": "default_index10_state3_success_stability_focus_v3",
        "specialties": [case["name"] for case in SPECIALTY_CASES],
        "phaseRuleSpecialization": "phase rules may include parameter-regime suffixes pa/pg/pf/pw so high-acc, low-g, friction and wind advice do not overwrite each other",
        "successDefinition": "browser handoff: a short dwell inside a broad safe capture basin that the terminal local controller can quickly stabilize",
        "scoreDefinition": "success-focused target-3 score: success rate, capture rate, worst-seed robustness and low rail/wrong-energy risk are prioritized; speed is only a weak tie-breaker",
        "profiles": profiles,
        "phaseRules": rules,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously improve adaptive AI profiles and phase-rule database locally.")
    parser.add_argument("--episodes", type=int, default=0, help="number of episodes to train; 0 means run forever")
    parser.add_argument("--seconds", type=float, default=17.0, help="max simulated seconds per episode")
    parser.add_argument("--dt", type=float, default=1.0 / 180.0, help="simulation step for trainer; default is closer to browser fixed-step behavior")
    parser.add_argument("--seed", type=int, default=114513, help="random seed")
    parser.add_argument("--db", type=Path, default=Path("ai_data/training_db.json"), help="published JSON database path used by the browser")
    parser.add_argument("--fresh", action="store_true", help="start from an empty database instead of improving the existing one")
    parser.add_argument("--resume", action="store_true", help="kept for compatibility; resume is now the default unless --fresh is used")
    parser.add_argument("--targets", default="3", help="comma-separated target ids to train; this package is specialized for target state 3")
    parser.add_argument("--save-every", type=int, default=50, help="candidate autosave interval in episodes")
    parser.add_argument("--save-seconds", type=float, default=8.0, help="also autosave candidate after this many wall-clock seconds")
    parser.add_argument("--validate-episodes", type=int, default=98, help="fixed-seed validation episodes; set 0 to disable validation")
    parser.add_argument("--validate-every", type=int, default=100, help="in endless mode, publish only after this many episodes if score improves")
    parser.add_argument("--validation-seed", type=int, default=99021, help="seed for acceptance-gate validation")
    parser.add_argument("--acceptance-tolerance", type=float, default=0.0, help="required validation-score improvement margin before publishing")
    parser.add_argument("--explore-rate", type=float, default=0.0, help="probability of extra within-specialty exploration; 0 keeps training focused on named regimes")
    parser.add_argument("--replay-rate", type=float, default=0.58, help="probability of improving existing buckets within the selected specialty")
    parser.add_argument("--hard-mining-rate", type=float, default=0.34, help="probability of training a mined browser-like validation failure instead of a fresh curriculum case")
    parser.add_argument("--hard-pool-episodes", type=int, default=192, help="episodes used to mine hard cases from the selected browser-like specialties")
    parser.add_argument("--hard-refresh-every", type=int, default=450, help="refresh the hard-case pool every N training episodes; 0 disables refresh")
    parser.add_argument("--hard-probes", type=int, default=14, help="bounded profile-surgery probes for each mined hard case")
    parser.add_argument("--specialties", default="all", help="comma-separated specialty names to train/evaluate, or all")
    parser.add_argument("--no-acceptance-gate", action="store_true", help="publish trained database even if validation score regresses")
    parser.add_argument("--eval-only", action="store_true", help="only evaluate the selected database without training")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    target_ids = [int(x.strip()) for x in args.targets.split(",") if x.strip()]
    if any(t not in TARGETS for t in target_ids):
        raise SystemExit("targets must be chosen from 0,1,2,3")
    if args.specialties.strip().lower() == "all":
        specialty_names = [case["name"] for case in SPECIALTY_CASES]
    else:
        specialty_names = [x.strip() for x in args.specialties.split(",") if x.strip()]
        unknown = [x for x in specialty_names if x not in SPECIALTY_BY_NAME]
        if unknown:
            raise SystemExit(f"unknown specialties: {', '.join(unknown)}")

    db = empty_db() if args.fresh and not args.eval_only else load_db(args.db)
    if args.eval_only:
        summary = evaluate_db(db, args.validate_episodes or 160, args.seconds, args.dt, args.validation_seed, target_ids, specialty_names)
        print("eval:", format_eval(summary))
        print("events:", summary["events"])
        return

    infinite = args.episodes <= 0
    gate_enabled = args.validate_episodes > 0 and not args.no_acceptance_gate
    candidate_path = args.db.with_name(f"{args.db.stem}.candidate{args.db.suffix}") if gate_enabled else args.db

    best_db = copy.deepcopy(db)
    best_eval = evaluate_db(best_db, args.validate_episodes, args.seconds, args.dt, args.validation_seed, target_ids, specialty_names) if gate_enabled else None
    best_score = best_eval["score"] if best_eval else float("-inf")
    if best_eval:
        print("baseline:", format_eval(best_eval), flush=True)

    hard_cases: List[Dict[str, Any]] = []
    if args.hard_mining_rate > 0 and args.hard_pool_episodes > 0:
        hard_cases = collect_hard_cases(db, args.hard_pool_episodes, args.seconds, args.dt, args.validation_seed + 137, target_ids, specialty_names)
        print(f"hard cases mined: {len(hard_cases)}", flush=True)

    stats = {"success": 0, "events": {}, "bestAngles": [], "sources": {}, "specialties": {}, "hardImproved": 0, "hardTried": 0}
    start = time.time()
    last_save = start
    ep = 0
    last_validation_ep = 0

    while infinite or ep < args.episodes:
        ep += 1
        use_hard = bool(hard_cases) and rng.random() < clamp(args.hard_mining_rate, 0.0, 0.95)
        if use_hard:
            weights = [1.0 + max(0.0, -float(c.get("score", 0.0))) * 0.04 + (2.0 if c.get("event") == "edge-risk" else 0.0) for c in hard_cases]
            hard = rng.choices(hard_cases, weights=weights, k=1)[0]
            target = int(hard["target"])
            params = dict(DEFAULT_PARAMS); params.update({k: float(v) for k, v in hard.get("params", {}).items()})
            specialty = str(hard.get("specialty", infer_specialty_name(params)))
            source = f"hard:{specialty}"
            stats["hardTried"] += 1
            improved = optimize_hard_case_profile(db, hard, args.seconds, args.dt, rng, probes=args.hard_probes)
            stats["hardImproved"] += 1 if improved else 0
            initial = jitter_state_for_browser_case(rng, state_from_dict(hard["initial"]), params, scale=0.75)
            result = run_episode(rng, db, target, params, initial, args.seconds, args.dt, learn=True)
        else:
            target, params, source = sample_training_case(
                rng, db, target_ids, explore_rate=args.explore_rate, jitter=True,
                specialty_stats=stats["specialties"], specialty_names=specialty_names, replay_rate=args.replay_rate
            )
            specialty = source.split(":", 1)[1] if ":" in source else infer_specialty_name(params)
            if source.startswith("replay:"):
                mode = rng.choices(["browser-target", "near-down", "edge-recovery", "browser-drag", "mixed"], weights=[0.36, 0.18, 0.22, 0.10, 0.14], k=1)[0]
            else:
                mode = rng.choices(["browser-target", "near-down", "edge-recovery", "browser-drag", "mixed"], weights=[0.34, 0.18, 0.22, 0.10, 0.16], k=1)[0]

            initial = random_state(rng, params, mode, target)
            result = run_episode(rng, db, target, params, initial, args.seconds, args.dt, learn=True)
        stats["success"] += 1 if result["success"] else 0
        stats["events"][result["event"]] = stats["events"].get(result["event"], 0) + 1
        stats["sources"][source] = stats["sources"].get(source, 0) + 1
        stats["bestAngles"].append(float(result.get("bestAngle", 9.0)))
        sp = stats["specialties"].setdefault(specialty, {"episodes": 0, "failures": 0, "edge": 0, "slow": 0})
        sp["episodes"] += 1
        if not result["success"]: sp["failures"] += 1
        if result["event"] == "edge-risk": sp["edge"] += 1
        if result["event"] in ("slow-arrival", "slow-capture"): sp["slow"] += 1

        now = time.time()
        should_save = (ep % max(1, args.save_every) == 0) or (args.save_seconds > 0 and now - last_save >= args.save_seconds)
        if should_save:
            export_db(db, candidate_path, ep, args.seed)
            last_save = now
            elapsed = max(1e-6, now - start)
            rate = ep / elapsed
            window = stats["bestAngles"][-min(100, len(stats["bestAngles"])):]
            mean_best = sum(window) / max(1, len(window))
            limit = "∞" if infinite else str(args.episodes)
            hard_msg = f" | hard {stats['hardImproved']}/{stats['hardTried']}" if stats.get('hardTried') else ""
            print(f"episode {ep}/{limit} | practical-success {stats['success']/ep:.1%} | mean-best-angle(last100) {mean_best:.3f} | profiles {len(db['profiles_by_key'])} | rules {len(db['phase_by_key'])}{hard_msg} | {rate:.1f} ep/s", flush=True)

        do_periodic_validation = gate_enabled and args.validate_every > 0 and ep - last_validation_ep >= args.validate_every
        if do_periodic_validation:
            last_validation_ep = ep
            candidate_eval = evaluate_db(db, args.validate_episodes, args.seconds, args.dt, args.validation_seed, target_ids, specialty_names)
            improved = candidate_eval["score"] >= best_score + args.acceptance_tolerance
            print("validation:", format_eval(candidate_eval), "|", "published" if improved else "not published", flush=True)
            if improved or not gate_enabled:
                best_score = candidate_eval["score"]
                best_eval = candidate_eval
                best_db = copy.deepcopy(db)
                export_db(db, args.db, ep, args.seed)
            else:
                # Keep the public database monotonic.  Roll the in-memory learner
                # back to the best accepted version, then continue searching.
                db = copy.deepcopy(best_db)

        if args.hard_refresh_every > 0 and ep % args.hard_refresh_every == 0 and args.hard_mining_rate > 0 and args.hard_pool_episodes > 0:
            hard_cases = collect_hard_cases(db, args.hard_pool_episodes, args.seconds, args.dt, args.validation_seed + 137 + ep, target_ids, specialty_names)
            print(f"hard cases refreshed: {len(hard_cases)}", flush=True)

    accepted = True
    after_eval = None
    if gate_enabled:
        after_eval = evaluate_db(db, args.validate_episodes, args.seconds, args.dt, args.validation_seed, target_ids, specialty_names)
        accepted = after_eval["score"] >= best_score + args.acceptance_tolerance
        print("final validation:", format_eval(after_eval), flush=True)

    if accepted or not gate_enabled:
        export_db(db, args.db, ep, args.seed)
        database_path = args.db
    else:
        export_db(db, candidate_path, ep, args.seed)
        export_db(best_db, args.db, 0, args.seed)
        database_path = candidate_path

    print("done")
    if accepted or not gate_enabled:
        print(f"database: {database_path}")
    else:
        print(f"candidate not published because validation did not improve; saved candidate: {database_path}")
        print(f"kept best database: {args.db}")
    print(f"profiles: {len(db['profiles_by_key'])}, phase rules: {len(db['phase_by_key'])}")
    print("events:", stats["events"])
    print("sources:", stats["sources"])
    print("specialties:", stats["specialties"])
    if best_eval:
        print("best validation:", format_eval(best_eval))
    if after_eval:
        print("final validation:", format_eval(after_eval))


if __name__ == "__main__":
    main()
