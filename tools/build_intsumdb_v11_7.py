#!/usr/bin/env python3
"""Build the lazy RIES v11.7 integral/sum candidate database chunks.

Input is the v11.7 candidate JSONL generated from the uploaded integral/sum
sources.  The web-facing asset is split like hypdata:
  level4: simple low-height rows (about one fifth) + stage-1 multipliers
  level5: remaining rows so the cumulative table is complete + stage-2 multipliers
  level6: no extra rows, only deeper stage-3 multipliers

The browser matcher compares x ≈ M·S, where S is a stored integral/sum value and
M is one of the same rational/π/radical/Gamma multiplier families used by the
hypergeometric matcher.
"""
from __future__ import annotations

import argparse
import base64
import json
import math
import os
import struct
import re
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from fractions import Fraction
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
VERSION = '11.7.2-intsumdb-sign-dedupe-20260607'
DEFAULT_INPUTS = [
    ROOT / 'assets' / 'ries-intsumdb-v11_7-candidates.jsonl',
    ROOT.parent / 'candidates' / 'assets' / 'ries-intsumdb-v11_7-candidates.jsonl',
    Path('/mnt/data/work_v117/candidates/assets/ries-intsumdb-v11_7-candidates.jsonl'),
]
OUT_STATS = ROOT / 'assets' / 'ries-intsumdb-v11_7-stats.json'


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def pack_f64(xs: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<d', float(x)) for x in xs)


def pack_f32(xs: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<f', float(x)) for x in xs)


def pack_u32(xs: Iterable[int]) -> bytes:
    return b''.join(struct.pack('<I', int(x)) for x in xs)


def pack_u16(xs: Iterable[int]) -> bytes:
    return b''.join(struct.pack('<H', max(0, min(65535, int(x)))) for x in xs)


def pack_u8(xs: Iterable[int]) -> bytes:
    return bytes(max(0, min(255, int(x))) for x in xs)


@dataclass(frozen=True)
class Mult:
    value: float
    text: str
    latex: str
    family: str
    stage: int
    complexity: float


def rat_cost(n: int, d: int) -> float:
    h = max(abs(n), abs(d))
    if h <= 1:
        return 0.0
    return math.log2(h + 1) * 0.95


def rat_text_latex(n: int, d: int):
    if d == 1:
        return str(abs(n)), str(abs(n))
    return f'{abs(n)}/{d}', f'\\frac{{{abs(n)}}}{{{d}}}'


def pow_text(base: str, exp: int):
    if exp == 0:
        return '', ''
    if exp == 1:
        return base, base
    return f'{base}^{exp}', f'{base}^{{{exp}}}'


def pi_text(d2: int):
    if d2 == 0:
        return '', ''
    if d2 == 2:
        return 'π', '\\pi'
    if d2 == -2:
        return 'π^-1', '\\pi^{-1}'
    if d2 == 1:
        return 'sqrt(π)', '\\sqrt{\\pi}'
    if d2 == -1:
        return 'π^(-1/2)', '\\pi^{-1/2}'
    if d2 % 2 == 0:
        return f'π^{d2//2}', f'\\pi^{{{d2//2}}}'
    return f'π^({d2}/2)', f'\\pi^{{{d2}/2}}'


def gamma_text(r: str, exp: int):
    if exp == 0:
        return '', ''
    if exp == 1:
        return f'Γ({r})', f'\\Gamma({r})'
    return f'Γ({r})^{exp}', f'\\Gamma({r})^{{{exp}}}'


GAMMA_VALS = {
    '1/3': math.gamma(1/3), '1/4': math.gamma(1/4), '1/6': math.gamma(1/6),
    '1/5': math.gamma(1/5), '2/5': math.gamma(2/5), '1/8': math.gamma(1/8), '3/8': math.gamma(3/8),
    '1/10': math.gamma(1/10), '3/10': math.gamma(3/10), '1/12': math.gamma(1/12), '5/12': math.gamma(5/12),
}


def regular_comps(stage: int):
    if stage == 1:
        prime_specs = [('2', 2, -6, 6, 1.0), ('3', 3, -6, 6, 1.0), ('5', 5, -4, 4, 1.5)]
        pi_range = range(-6, 7)
        sqrt_opts = [(1, '', '', 0.0)]
        for n, c in [(2, 1.7), (3, 1.7), (5, 2.0)]:
            sqrt_opts += [(math.sqrt(n), f'√{n}', f'\\sqrt{{{n}}}', c), (1 / math.sqrt(n), f'1/√{n}', f'1/\\sqrt{{{n}}}', c)]
        max_cost, cap = 7.2, 900
    elif stage == 2:
        prime_specs = [('2', 2, -9, 9, 1.0), ('3', 3, -9, 9, 1.0), ('5', 5, -6, 6, 1.5), ('7', 7, -3, 3, 2.2)]
        pi_range = range(-12, 13)
        sqrt_opts = [(1, '', '', 0.0)]
        for n, c in [(2, 1.6), (3, 1.6), (5, 1.8), (6, 2.0), (7, 2.2), (10, 2.4)]:
            sqrt_opts += [(math.sqrt(n), f'√{n}', f'\\sqrt{{{n}}}', c), (1 / math.sqrt(n), f'1/√{n}', f'1/\\sqrt{{{n}}}', c)]
        max_cost, cap = 10.6, 2600
    else:
        prime_specs = [('2', 2, -12, 12, 1.0), ('3', 3, -12, 12, 1.0), ('5', 5, -9, 9, 1.5), ('7', 7, -5, 5, 2.2), ('11', 11, -2, 2, 3.5)]
        pi_range = range(-20, 21)
        sqrt_opts = [(1, '', '', 0.0)]
        for n, c in [(2, 1.5), (3, 1.5), (5, 1.7), (6, 1.9), (7, 2.1), (10, 2.2), (15, 2.5)]:
            sqrt_opts += [(math.sqrt(n), f'√{n}', f'\\sqrt{{{n}}}', c), (1 / math.sqrt(n), f'1/√{n}', f'1/\\sqrt{{{n}}}', c)]
        max_cost, cap = 13.5, 5200

    prime_comps = []

    def rec(i, value, cost, texts, latexs):
        if cost > max_cost:
            return
        if i == len(prime_specs):
            prime_comps.append((value, cost, texts[:], latexs[:]))
            return
        sym, val, lo, hi, w = prime_specs[i]
        for e in range(lo, hi + 1):
            ec = abs(e) * w
            if cost + ec > max_cost:
                continue
            if e == 0:
                rec(i + 1, value, cost, texts, latexs)
            else:
                t, l = pow_text(sym, e)
                rec(i + 1, value * math.pow(val, e), cost + ec, texts + [t], latexs + [l])

    rec(0, 1.0, 0.0, [], [])
    seen = {}
    for pv, pc, pt, pl in prime_comps:
        for d2 in pi_range:
            pic = abs(d2) * 1.2
            if pc + pic > max_cost:
                continue
            pit, pil = pi_text(d2)
            piv = math.pow(math.pi, d2 / 2)
            for sv, st, sl, sc in sqrt_opts:
                cost = pc + pic + sc
                if cost > max_cost:
                    continue
                val = pv * piv * sv
                texts = pt + ([pit] if pit else []) + ([st] if st else [])
                latexs = pl + ([pil] if pil else []) + ([sl] if sl else [])
                key = f'{val:.15e}'
                item = (val, cost, '·'.join(texts), '\\,'.join(latexs), 'pi-radical-prime factor')
                old = seen.get(key)
                if old is None or (cost, len(item[2])) < (old[1], len(old[2])):
                    seen[key] = item
    out = list(seen.values())
    out.sort(key=lambda x: (x[1], len(x[2]), abs(math.log(abs(x[0])) if x[0] else 999)))
    return out[:cap]


def gamma_comps(stage: int):
    if stage == 1:
        return [(1.0, 0.0, '', '', 'no gamma')]
    if stage == 2:
        bases = [('1/3', 3, 4.0), ('1/4', 3, 4.0), ('1/6', 3, 4.2)]
        max_cost, cap = 8.4, 160
    else:
        bases = [('1/3', 4, 3.8), ('1/4', 4, 3.8), ('1/6', 4, 4.0), ('1/5', 2, 5.2), ('2/5', 2, 5.2), ('1/8', 2, 5.5), ('3/8', 2, 5.5), ('1/10', 1, 6.0), ('3/10', 1, 6.0), ('1/12', 1, 6.2), ('5/12', 1, 6.2)]
        max_cost, cap = 12.5, 420
    out = [(1.0, 0.0, '', '', 'no gamma')]
    for r, mx, w in bases:
        for e in range(-mx, mx + 1):
            if not e:
                continue
            c = abs(e) * w
            if c <= max_cost:
                t, l = gamma_text(r, e)
                out.append((math.pow(GAMMA_VALS[r], e), c, t, l, 'gamma quotient'))
    for i, (r1, mx1, w1) in enumerate(bases):
        for r2, mx2, w2 in bases[i + 1:]:
            for e1 in range(-mx1, mx1 + 1):
                if not e1:
                    continue
                for e2 in range(-mx2, mx2 + 1):
                    if not e2:
                        continue
                    c = abs(e1) * w1 + abs(e2) * w2
                    if c > max_cost:
                        continue
                    t1, l1 = gamma_text(r1, e1)
                    t2, l2 = gamma_text(r2, e2)
                    out.append((math.pow(GAMMA_VALS[r1], e1) * math.pow(GAMMA_VALS[r2], e2), c, t1 + '·' + t2, l1 + '\\,' + l2, 'gamma quotient'))
    core = bases[:3]
    for e1 in range(-2, 3):
        for e2 in range(-2, 3):
            for e3 in range(-2, 3):
                if not (e1 or e2 or e3):
                    continue
                c = abs(e1) * core[0][2] + abs(e2) * core[1][2] + abs(e3) * core[2][2]
                if c > max_cost:
                    continue
                texts, latexs, val = [], [], 1.0
                for (r, _, _), e in zip(core, [e1, e2, e3]):
                    if e:
                        t, l = gamma_text(r, e)
                        texts.append(t)
                        latexs.append(l)
                        val *= math.pow(GAMMA_VALS[r], e)
                out.append((val, c, '·'.join(texts), '\\,'.join(latexs), 'gamma quotient'))
    seen = {}
    for v, c, t, l, f in out:
        key = f'{v:.15e}'
        old = seen.get(key)
        if old is None or (c, len(t)) < (old[1], len(old[2])):
            seen[key] = (v, c, t, l, f)
    out = list(seen.values())
    out.sort(key=lambda x: (x[1], len(x[2]), abs(math.log(abs(x[0])) if x[0] else 999)))
    return out[:cap]


def factor_combos(stage: int):
    regs = regular_comps(stage)
    gams = gamma_comps(stage)
    seen = {}
    raw = list(regs)
    if stage > 1:
        reg_for_gamma = regs[:650 if stage == 2 else 1100]
        for rv, rc, rt, rl, rf in reg_for_gamma:
            for gv, gc, gt, gl, gf in gams:
                if not gt:
                    continue
                cost = rc + gc
                if cost > (13.2 if stage == 2 else 18.2):
                    continue
                text = '·'.join([x for x in (rt, gt) if x])
                latex = '\\,'.join([x for x in (rl, gl) if x])
                raw.append((rv * gv, cost, text, latex, 'gamma quotient'))
    for v, c, t, l, f in raw:
        if not (math.isfinite(v) and v != 0):
            continue
        key = f'{v:.15e}'
        old = seen.get(key)
        if old is None or (c, len(t)) < (old[1], len(old[2])):
            seen[key] = (v, c, t, l, f)
    out = list(seen.values())
    out.sort(key=lambda x: (x[1], len(x[2]), abs(math.log(abs(x[0])) if x[0] else 999)))
    return out[:{1: 900, 2: 3600, 3: 9000}[stage]]


def rational_options(max_h: int):
    out = []
    for n in range(1, max_h + 1):
        for d in range(1, max_h + 1):
            if math.gcd(n, d) != 1:
                continue
            c = rat_cost(n, d)
            t, l = rat_text_latex(n, d)
            out.append((n / d, c, t, l, n, d))
    out.sort(key=lambda x: (x[1], x[5], x[4]))
    return out


def build_multipliers():
    stage_cfg = {
        1: {'max_h': 12, 'max_cost': 10.0, 'cap': 1200, 'rat_cap': 80, 'factor_cap': 500},
        2: {'max_h': 30, 'max_cost': 18.0, 'cap': 6500, 'rat_cap': 220, 'factor_cap': 1200},
        3: {'max_h': 100, 'max_cost': 24.0, 'cap': 16000, 'rat_cap': 350, 'factor_cap': 2000},
    }
    by_key = {}
    for stage in (1, 2, 3):
        comps = factor_combos(stage)[:stage_cfg[stage]['factor_cap']]
        rats = rational_options(stage_cfg[stage]['max_h'])[:stage_cfg[stage]['rat_cap']]
        for fval, fcost, ftext, flatex, family in comps:
            for rval, rcost, rtext, rlatex, n, d in rats:
                cost = fcost + rcost
                if cost > stage_cfg[stage]['max_cost']:
                    break
                for sign in (1, -1):
                    val = sign * rval * fval
                    if not (math.isfinite(val) and val != 0 and 1e-300 < abs(val) < 1e300):
                        continue
                    parts, lparts = [], []
                    if not (n == 1 and d == 1):
                        parts.append(rtext)
                        lparts.append(rlatex)
                    if ftext:
                        parts.append(ftext)
                        lparts.append(flatex)
                    text = '·'.join(parts) if parts else '1'
                    latex = '\\,'.join(lparts) if lparts else '1'
                    if sign < 0:
                        text = '-' + text if text != '1' else '-1'
                        latex = '-' + latex if latex != '1' else '-1'
                    key = f'{val:.15e}'
                    m = Mult(val, text, latex, family, stage, cost + (0.4 if sign < 0 else 0))
                    old = by_key.get(key)
                    if old is None or (m.stage, m.complexity, len(m.text)) < (old.stage, old.complexity, len(old.text)):
                        by_key[key] = m
        cur = [m for m in by_key.values() if m.stage <= stage]
        cur.sort(key=lambda m: (m.stage, m.complexity, len(m.text), abs(math.log(abs(m.value)))))
        keep = set(f'{m.value:.15e}' for m in cur[:stage_cfg[stage]['cap']])
        by_key = {k: v for k, v in by_key.items() if k in keep or v.stage < stage}
    mults = list(by_key.values())
    mults.sort(key=lambda m: (m.stage, m.complexity, len(m.text), abs(math.log(abs(m.value)))))
    return mults


def clean_line(s) -> str:
    return str(s if s is not None else '').replace('\r', ' ').replace('\n', ' ').strip()


def normalize_latex_signs(s: str) -> str:
    """Clean generated sign runs that would render as confusing TeX."""
    out = clean_line(s).replace('−', '-')
    old = None
    while out != old:
        old = out
        out = (out.replace('+-', '-')
                  .replace('-+', '-')
                  .replace('++', '+')
                  .replace('--', '+'))
    out = out.replace('^{+1}', '^{1}').replace('^{+0}', '^{0}')
    return out


def _matching_paren(s: str, i: int) -> int:
    depth = 0
    for j in range(i, len(s)):
        if s[j] == '(':
            depth += 1
        elif s[j] == ')':
            depth -= 1
            if depth == 0:
                return j
    return -1


def _neutral_power_at(s: str, i: int):
    m = re.match(r'\^\{\s*([+-]?\d+)\s*\}|\^([+-]?\d+)(?!\d)', s[i:])
    if not m:
        return None
    v = int(m.group(1) if m.group(1) is not None else m.group(2))
    if v in (0, 1):
        return v, len(m.group(0))
    return None


def _simplify_balanced_group_powers(s: str) -> str:
    out = s
    for _ in range(10):
        changed = False
        res = []
        i = 0
        while i < len(out):
            if out[i] == '(':
                j = _matching_paren(out, i)
                if j > i:
                    p = _neutral_power_at(out, j + 1)
                    if p:
                        v, plen = p
                        res.append(out[i:j + 1] if v == 1 else '1')
                        i = j + 1 + plen
                        changed = True
                        continue
            res.append(out[i])
            i += 1
        out = ''.join(res)
        if not changed:
            break
    return out


def normalize_latex_display(s: str) -> str:
    """Normalize common generated LaTeX display artifacts before packaging."""
    out = normalize_latex_signs(s)
    out = re.sub(r'\^\{\s*([+-]?\d+)\s*([+-])\s*([+-]?\d+)\s*\}', lambda m: f'^{{{int(m.group(1)) + (int(m.group(3)) if m.group(2)=="+" else -int(m.group(3)))}}}', out)
    out = re.sub(r'(\\(?:sin|cos|tan|log|exp|arctan|arsinh|sinh|cosh)\s*\((?:[^()]|\([^()]*\))*\))\^\{1\}', r'\1', out)
    out = re.sub(r'(\\(?:sin|cos|tan|log|exp|arctan|arsinh|sinh|cosh)\s*\((?:[^()]|\([^()]*\))*\))\^\{0\}', '1', out)
    out = re.sub(r'(\\binom\{[^{}]+\}\{[^{}]+\})\^\{1\}', r'\1', out)
    out = re.sub(r'\(1\)\^\{?n\}?', '', out)
    out = re.sub(r'(\\log\s*x|\\log\s*t|[A-Za-z]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)\^\{1\}', r'\1', out)
    out = re.sub(r'(\\binom\{[^{}]+\}\{[^{}]+\})\^\{1\}', r'\1', out)
    out = re.sub(r'\(1\)\^\{?n\}?', '', out)
    out = re.sub(r'(\\log\s*x|\\log\s*t|[A-Za-z]|\\[A-Za-z]+|\\pi|\\theta|\\varphi)\^\{0\}', '', out)
    out = _simplify_balanced_group_powers(out)
    out = re.sub(r'([\{+\-(])\s*1(?=(?:[A-Za-z]|\\(?!right\b)[A-Za-z]+|\\left|\())', r'\1', out)
    out = re.sub(r'(?<=[\)\}])1(?=[\}\)])', '', out)
    out = re.sub(r'(?<![0-9])1+\\,d', r'\\,d', out)
    return normalize_latex_signs(out)

def input_path(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    for p in DEFAULT_INPUTS:
        if p.exists():
            return p
    return DEFAULT_INPUTS[0]


def normalize_row(raw: dict, idx: int) -> dict | None:
    value = float(raw.get('value'))
    if not (math.isfinite(value) and value != 0):
        return None
    height = raw.get('height_score')
    if height is None:
        height = 9
    try:
        height = int(height)
    except Exception:
        height = 9
    return {
        'value': value,
        'valueText': clean_line(raw.get('value') or raw.get('value16') or value),
        'value16': clean_line(raw.get('value16') or raw.get('value') or value),
        'latex': normalize_latex_display(raw.get('latex')),
        'plain': clean_line(raw.get('plain') or raw.get('latex') or raw.get('db_id') or raw.get('id') or f'intsum row {idx+1}'),
        'family': clean_line(raw.get('family_id') or raw.get('family') or 'INTSUM'),
        'sub': clean_line(raw.get('subfamily_id') or raw.get('sub') or ''),
        'id': clean_line(raw.get('db_id') or raw.get('id') or f'INTSUMDB_{idx+1:06d}'),
        'status': clean_line(raw.get('source_status') or raw.get('sourceStatus') or 'verified'),
        'dataset': clean_line(raw.get('source_dataset') or raw.get('sourceDataset') or ''),
        'verifiedDigits': int(raw.get('verified_digits') or raw.get('verifiedDigits') or 0),
        'height': max(0, min(65535, height)),
        'formulaKey': clean_line(raw.get('formula_key') or raw.get('formulaKey') or ''),
        'params': raw.get('parameters') if isinstance(raw.get('parameters'), dict) else {},
        'sourceIndex': idx,
        'simple': height <= 2,
    }


def _param_fraction_text(v) -> str:
    try:
        f = Fraction(str(v)).limit_denominator(48)
        if abs(float(f) - float(v)) <= 1e-12:
            return str(f.numerator) if f.denominator == 1 else f'{f.numerator}/{f.denominator}'
    except Exception:
        pass
    return clean_line(v)


def _param_int(v, default=0) -> int:
    try:
        return int(round(float(v)))
    except Exception:
        return default


def _param_float(v, default=0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _abs_value_key(value_text: str, value: float) -> str:
    try:
        d = Decimal(str(value_text)).copy_abs()
    except (InvalidOperation, ValueError):
        d = Decimal(str(value)).copy_abs()
    # Match the precision used by the displayed value16 field, but avoid grouping
    # merely adjacent constants that differ beyond the stored detection precision.
    return format(d.quantize(Decimal('1e-16'), rounding=ROUND_HALF_UP), 'f')


def _unit_beta_rational_den_key(r: dict):
    p = r.get('params') or {}
    fam = (r.get('family'), r.get('sub'))
    if fam == ('RATIONAL_LOG_BETA', 'unit_rational_log_beta'):
        A, B, q = _param_float(p.get('A')), _param_float(p.get('B')), _param_float(p.get('q'))
        if abs(A - 1.0) > 1e-12 or abs(B - 1.0) > 1e-12 or q <= 0:
            return None
        m, n, rr = _param_int(p.get('m')), _param_int(p.get('n')), _param_int(p.get('r'), 1)
        semantic_sign = -1 if ((m + n) & 1) else 1
        key = ('unit_beta_rational_den', _param_fraction_text(p.get('a')), _param_fraction_text(p.get('b')), str(m), str(n), str(rr), _param_fraction_text(q))
        return '|'.join(key), semantic_sign
    if fam == ('BETA_LOG_PLUS_MINUS', 'unit_beta_log_pm'):
        c, sgn, t = _param_float(p.get('c')), _param_float(p.get('s')), _param_float(p.get('t'))
        e, f, g = _param_int(p.get('e'), 1), _param_int(p.get('f'), 1), _param_int(p.get('g'), 1)
        # Conservative bridge to RATIONAL_LOG_BETA only when the formula is exactly
        # x^a(1-x)^b(1+x^f)^(-q)(-log x)^m log(1-x)^n.
        if e != 1 or f <= 0 or g != 1 or abs(sgn - 1.0) > 1e-12 or abs(t + 1.0) > 1e-12 or c >= 0:
            return None
        m, n = _param_int(p.get('m')), _param_int(p.get('n'))
        semantic_sign = -1 if (n & 1) else 1
        key = ('unit_beta_rational_den', _param_fraction_text(p.get('a')), _param_fraction_text(p.get('b')), str(m), str(n), str(f), _param_fraction_text(-c))
        return '|'.join(key), semantic_sign
    return None


def _trig_fourier_reflection_key(r: dict):
    if (r.get('family'), r.get('sub')) != ('TRIG_RATIONAL_FOURIER', 'trig_rational_fourier'):
        return None
    p = r.get('params') or {}
    B, D = _param_float(p.get('B')), _param_float(p.get('D'))
    orient = 0
    for z in (B, D):
        if abs(z) > 1e-12:
            orient = 1 if z > 0 else -1
            break
    if orient == 0:
        return None
    # x -> pi-x flips every cos(x) coefficient.  Only normalize rows where all
    # non-zero coefficients share the same orientation; mixed signs are kept.
    if B * orient < -1e-12 or D * orient < -1e-12:
        return None
    k = _param_int(p.get('k'))
    semantic_sign = -1 if (orient < 0 and (k & 1)) else 1
    key = ('trig_fourier_reflection', _param_fraction_text(p.get('A')), _param_fraction_text(abs(B)), _param_fraction_text(p.get('C')), _param_fraction_text(abs(D)), str(k), _param_fraction_text(p.get('m')), _param_fraction_text(p.get('q')))
    return '|'.join(key), semantic_sign


def _sign_equivalence_keys(r: dict):
    keys = []
    for fn in (_unit_beta_rational_den_key, _trig_fourier_reflection_key):
        item = fn(r)
        if item:
            keys.append(item)
    return keys


def _row_preference_key(r: dict):
    verified_rank = 0 if str(r.get('status', '')).lower() == 'verified' else 1
    sign_rank = 0 if r.get('value', 0) > 0 else 1
    return (verified_rank, r.get('height', 9999), len(r.get('latex', '')), len(r.get('plain', '')), sign_rank, r.get('id', ''))



def scan_value_sign_pairs(rows: list[dict]):
    groups = defaultdict(list)
    for r in rows:
        groups[_abs_value_key(r.get('value16') or r.get('valueText'), r.get('value'))].append(r)
    mixed = [items for items in groups.values() if any(r['value'] > 0 for r in items) and any(r['value'] < 0 for r in items)]
    return {
        'absValue16MixedSignGroups': len(mixed),
        'rowsInMixedSignGroups': sum(len(items) for items in mixed),
        'mixedSignGroupsByFamily': dict(sorted(Counter(f"{r['family']}/{r['sub']}" for items in mixed for r in items).items())),
    }

def dedupe_sign_equivalent_rows(rows: list[dict]):
    groups = defaultdict(list)
    for r in rows:
        abs_key = _abs_value_key(r.get('value16') or r.get('valueText'), r.get('value'))
        for key, semantic_sign in _sign_equivalence_keys(r):
            groups[(abs_key, key)].append((r, semantic_sign))
    remove_ids = set()
    cases = []
    by_rule = Counter()
    for (abs_key, key), items in groups.items():
        if len(items) < 2:
            continue
        numeric_signs = {1 if r['value'] > 0 else -1 for r, _ in items}
        semantic_signs = {sgn for _, sgn in items}
        if len(numeric_signs) < 2 or len(semantic_signs) < 2:
            continue
        # Same absolute value and same canonical kernel.  Keep one representative;
        # the ± multiplier table can recover either target sign.
        ordered = sorted((r for r, _ in items), key=_row_preference_key)
        keeper = ordered[0]
        removed = ordered[1:]
        for r in removed:
            remove_ids.add(r['id'])
        rule = 'unit-beta/rational-log sign convention' if key.startswith('unit_beta_rational_den') else 'trig-Fourier reflection x->pi-x'
        by_rule[rule] += len(removed)
        cases.append({
            'absValue16': abs_key,
            'rule': rule,
            'canonicalKey': key,
            'kept': keeper['id'],
            'removed': [r['id'] for r in removed],
            'families': sorted(set(f"{r['family']}/{r['sub']}" for r, _ in items)),
        })
    filtered = [r for r in rows if r['id'] not in remove_ids]
    by_family = Counter(f"{r['family']}/{r['sub']}" for r in rows if r['id'] in remove_ids)
    report = {
        'preDedupeRows': len(rows),
        'rowsRemoved': len(remove_ids),
        'postDedupeRows': len(filtered),
        'ruleCounts': dict(sorted(by_rule.items())),
        'removedByFamily': dict(sorted(by_family.items())),
        'candidateGroups': len(cases),
        'sampleCases': cases[:24],
        'removedIds': sorted(remove_ids),
        'policy': 'Conservative sign-equivalence pass: remove only rows whose value is the negative of another row at value16 precision and whose family-specific canonical kernel proves they differ by a global sign convention. Nontrivial equal-absolute-value identities remain in the database.',
    }
    return filtered, report


def make_chunk(stage: int, level: int, label: str, rows: list[dict], mults: list[Mult], total_rows: int, source_rows: int):
    real_pairs = sorted((r['value'], i) for i, r in enumerate(rows) if math.isfinite(r['value']) and r['value'] != 0)
    fam_dict, fam_codes = [], []
    fam_index = {}
    for m in mults:
        f = m.family or 'multiplier'
        if f not in fam_index:
            fam_index[f] = len(fam_dict)
            fam_dict.append(f)
        fam_codes.append(fam_index[f])
    status_dict, status_codes = [], []
    status_index = {}
    dataset_dict, dataset_codes = [], []
    dataset_index = {}
    for r in rows:
        for val, dictionary, index, codes in [(r['status'], status_dict, status_index, status_codes), (r['dataset'], dataset_dict, dataset_index, dataset_codes)]:
            if val not in index:
                index[val] = len(dictionary)
                dictionary.append(val)
            codes.append(index[val])
    asset = {
        'version': VERSION,
        'stage': stage,
        'level': level,
        'label': label,
        'rows': len(rows),
        'globalRows': total_rows,
        'sourceRows': source_rows,
        'realRows': len(real_pairs),
        'decimalPlaces': [15, 16, 20],
        'idBlob': '\n'.join(r['id'] for r in rows),
        'plainBlob': '\n'.join(r['plain'] for r in rows),
        'latexBlob': '\n'.join(r['latex'] for r in rows),
        'valueBlob': '\n'.join(r['valueText'] for r in rows),
        'value16Blob': '\n'.join(r['value16'] for r in rows),
        'familyBlob': '\n'.join(r['family'] for r in rows),
        'subBlob': '\n'.join(r['sub'] for r in rows),
        'formulaKeyBlob': '\n'.join(r['formulaKey'] for r in rows),
        'valuesB64': b64(pack_f64(v for v, _ in real_pairs)),
        'rowB64': b64(pack_u32(i for _, i in real_pairs)),
        'complexityB64': b64(pack_u16(r['height'] for r in rows)),
        'verifiedDigitsB64': b64(pack_u16(r['verifiedDigits'] for r in rows)),
        'statusDict': status_dict,
        'statusB64': b64(pack_u8(status_codes)),
        'datasetDict': dataset_dict,
        'datasetB64': b64(pack_u8(dataset_codes)),
        'multiplierRows': len(mults),
        'multValuesB64': b64(pack_f64(m.value for m in mults)),
        'multComplexityB64': b64(pack_f32(m.complexity for m in mults)),
        'multTextBlob': '\n'.join(m.text for m in mults),
        'multLatexBlob': '\n'.join(m.latex for m in mults),
        'multFamilyDict': fam_dict,
        'multFamilyB64': b64(pack_u8(fam_codes)),
    }
    name = f'assets/ries-intsumdb-v11_7-level{level}.js'
    out = ROOT / name
    js = '(function(){\nwindow.RIES_INTSUMDB_V117_CHUNKS=window.RIES_INTSUMDB_V117_CHUNKS||[];\n'
    js += f'window.RIES_INTSUMDB_V117_CHUNKS[{stage-1}]='
    js += json.dumps(asset, ensure_ascii=False, separators=(',', ':'))
    js += ';\n})();\n'
    out.write_text(js, encoding='utf-8')
    return {
        'stage': stage,
        'level': level,
        'file': name,
        'assetBytes': len(js.encode('utf-8')),
        'rows': len(rows),
        'realRows': len(real_pairs),
        'multiplierRows': len(mults),
        'latexBlobBytes': len(asset['latexBlob'].encode('utf-8')),
        'plainBlobBytes': len(asset['plainBlob'].encode('utf-8')),
        'valueBlobBytes': len(asset['valueBlob'].encode('utf-8')),
    }


def main():
    ap = argparse.ArgumentParser(description='Build RIES v11.7 integral/sum database chunks.')
    ap.add_argument('--input', default=None, help='Path to ries-intsumdb-v11_7-candidates.jsonl')
    args = ap.parse_args()
    src = input_path(args.input)
    if not src.exists():
        raise SystemExit(f'Missing input JSONL: {src}')
    rows = []
    raw_count = 0
    for line in src.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        raw_count += 1
        r = normalize_row(json.loads(line), raw_count - 1)
        if r:
            rows.append(r)
    value_sign_pair_scan = scan_value_sign_pairs(rows)
    rows, sign_dedupe_report = dedupe_sign_equivalent_rows(rows)
    sign_dedupe_report['retainedMixedSignGroupsAsDistinct'] = max(0, value_sign_pair_scan['absValue16MixedSignGroups'] - sign_dedupe_report['candidateGroups'])
    rows.sort(key=lambda r: (0 if r['simple'] else 1, r['height'], r['family'], r['sub'], abs(r['value']), r['id']))
    simple = [r for r in rows if r['simple']]
    rest = [r for r in rows if not r['simple']]
    mults = build_multipliers()
    assets = []
    assets.append(make_chunk(1, 4, 'integral/sum level 4 simple low-height chunk', simple, [m for m in mults if m.stage == 1], len(rows), raw_count))
    assets.append(make_chunk(2, 5, 'integral/sum level 5 remaining full-data chunk', rest, [m for m in mults if m.stage == 2], len(rows), raw_count))
    assets.append(make_chunk(3, 6, 'integral/sum level 6 deep multiplier chunk', [], [m for m in mults if m.stage == 3], len(rows), raw_count))
    stats = {
        'version': VERSION,
        'sourceFile': str(src),
        'sourceRows': raw_count,
        'rows': len(rows),
        'valueSignPairScan': value_sign_pair_scan,
        'signDedupe': sign_dedupe_report,
        'level4Rows': len(simple),
        'level5AdditionalRows': len(rest),
        'level4Fraction': round(len(simple) / max(1, len(rows)), 6),
        'level5CumulativeRows': len(rows),
        'level6CumulativeRows': len(rows),
        'heightCounts': dict(sorted(Counter(str(r['height']) for r in rows).items(), key=lambda kv: (int(kv[0]), kv[1]))),
        'familyCounts': dict(sorted(Counter(f"{r['family']}/{r['sub']}" for r in rows).items())),
        'multiplierRows': len(mults),
        'multiplierStageCounts': {str(k): sum(1 for m in mults if m.stage == k) for k in (1, 2, 3)},
        'assets': assets,
        'cumulativeAssetBytes': {
            'level4': sum(a['assetBytes'] for a in assets if a['level'] <= 4),
            'level5': sum(a['assetBytes'] for a in assets if a['level'] <= 5),
            'level6': sum(a['assetBytes'] for a in assets if a['level'] <= 6),
        },
        'selectionPolicy': 'level4 uses rows with height_score <= 2 (about one fifth of the database); level5 loads all remaining rows; level6 adds only deeper multipliers.',
        'matchingPolicy': 'browser matcher compares x ≈ M·S, where S is a stored integral/sum value and M is a stage-gated rational, pi/radical/prime-power, or Gamma multiplier.',
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({
        'rows': stats['rows'],
        'level4Rows': stats['level4Rows'],
        'level5AdditionalRows': stats['level5AdditionalRows'],
        'signDedupeRowsRemoved': stats['signDedupe']['rowsRemoved'],
        'multiplierStageCounts': stats['multiplierStageCounts'],
        'cumulativeAssetBytes': stats['cumulativeAssetBytes'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
