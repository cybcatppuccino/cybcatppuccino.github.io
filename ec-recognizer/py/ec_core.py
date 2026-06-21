from __future__ import annotations

# EC Recognizer v17 core extracted from ec_atlas_package_v16.
# Pure-Python standard-library arithmetic: no SymPy, Sage, PARI/GP, NumPy, or server API.

import json
import math
import re
import functools
try:
    import sqlite3  # not used by the browser recognizer; kept only for compatibility with copied helper names
except Exception:  # pragma: no cover
    sqlite3 = None
from fractions import Fraction
from functools import lru_cache, reduce
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path('.')
DB_PATH = ROOT / 'data' / 'atlas_v15.sqlite'
META_PATH = ROOT / 'data' / 'plot_meta.json'
TOP_PATH = ROOT / 'data' / 'top_points.json.gz'
TILE_DIR = ROOT / 'data' / 'tiles'




def primes_below(n: int) -> List[int]:
    sieve = [True] * n
    sieve[:2] = [False, False]
    for p in range(2, int(n ** 0.5) + 1):
        if sieve[p]:
            step = p
            start = p * p
            sieve[start:n:step] = [False] * (((n - 1) - start) // step + 1)
    return [i for i, ok in enumerate(sieve) if ok]


PRIMES_LT_100 = primes_below(100)


def v_p(n: int, p: int) -> int:
    if n == 0:
        return 0
    n = abs(n)
    e = 0
    while n % p == 0:
        n //= p
        e += 1
    return e


def invariants(a1: int, a2: int, a3: int, a4: int, a6: int):
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    c4 = b2 * b2 - 24 * b4
    c6 = -b2 * b2 * b2 + 36 * b2 * b4 - 216 * b6
    disc = -b2 * b2 * b8 - 8 * b4 * b4 * b4 - 27 * b6 * b6 + 9 * b2 * b4 * b6
    return {
        'b2': b2, 'b4': b4, 'b6': b6, 'b8': b8,
        'c4': c4, 'c6': c6, 'disc': disc,
    }


def curve_eq_mod(a1, a2, a3, a4, a6, x, y, p):
    return (y * y + a1 * x * y + a3 * y - (x * x * x + a2 * x * x + a4 * x + a6)) % p


def derivs_mod(a1, a2, a3, a4, x, y, p):
    fx = (a1 * y - 3 * x * x - 2 * a2 * x - a4) % p
    fy = (2 * y + a1 * x + a3) % p
    return fx, fy


@functools.lru_cache(maxsize=2048)
def reduction_data(a1: int, a2: int, a3: int, a4: int, a6: int, bound: int = 100):
    inv = invariants(a1, a2, a3, a4, a6)
    disc = inv['disc']
    c4 = inv['c4']
    rows = []
    for p in primes_below(bound):
        nonsingular_affine = 0
        singular_points = []
        for x in range(p):
            for y in range(p):
                if curve_eq_mod(a1, a2, a3, a4, a6, x, y, p) == 0:
                    fx, fy = derivs_mod(a1, a2, a3, a4, x, y, p)
                    if fx == 0 and fy == 0:
                        singular_points.append((x, y))
                    else:
                        nonsingular_affine += 1
        smooth_points = nonsingular_affine + 1  # point at infinity
        vp_disc = v_p(disc, p)
        vp_c4 = v_p(c4, p)
        a_p = p + 1 - smooth_points
        if vp_disc == 0:
            rtype = 'good'
        else:
            if a_p == 1:
                rtype = 'split multiplicative'
            elif a_p == -1:
                rtype = 'nonsplit multiplicative'
            else:
                rtype = 'additive'
        rows.append({
            'p': p,
            'smooth_points': smooth_points,
            'a_p': a_p,
            'reduction': rtype,
            'vp_disc': vp_disc,
            'vp_c4': vp_c4,
            'singular_points': singular_points,
        })
    return rows


def local_coeff_prime_power(a_p: int, p: int, k: int, reduction: str) -> int:
    if k == 0:
        return 1
    if reduction == 'good':
        if k == 1:
            return a_p
        a0, a1 = 1, a_p
        for _ in range(2, k + 1):
            a0, a1 = a1, a_p * a1 - p * a0
        return a1
    if reduction in ('split multiplicative', 'nonsplit multiplicative'):
        return a_p ** k
    return 0


def factorint(n: int) -> Dict[int, int]:
    out: Dict[int, int] = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            out[d] = out.get(d, 0) + 1
            n //= d
        d = 3 if d == 2 else d + 2
    if n > 1:
        out[n] = out.get(n, 0) + 1
    return out


def newform_coefficients(a1: int, a2: int, a3: int, a4: int, a6: int, bound: int = 30):
    local = {row['p']: row for row in reduction_data(a1, a2, a3, a4, a6, max(bound + 1, 100))}
    coeffs = {1: 1}
    for n in range(2, bound + 1):
        fac = factorint(n)
        val = 1
        for p, e in fac.items():
            if p not in local:
                # should not happen with bound small, but keep robust
                extra = {row['p']: row for row in reduction_data(a1, a2, a3, a4, a6, p + 1)}
                local.update(extra)
            row = local[p]
            val *= local_coeff_prime_power(row['a_p'], p, e, row['reduction'])
        coeffs[n] = val
    return coeffs


def format_q_expansion(coeffs: Dict[int, int], bound: int | None = None) -> str:
    if bound is None:
        bound = max(coeffs)
    terms = []
    for n in range(1, bound + 1):
        a = coeffs[n]
        if a == 0:
            continue
        if n == 1:
            terms.append('q')
            continue
        sign = '+' if a > 0 else '-'
        mag = abs(a)
        coeff_str = '' if mag == 1 else str(mag)
        term = f'{coeff_str}q^{n}' if coeff_str else f'q^{n}'
        terms.append(f' {sign} {term}')
    return ''.join(terms) + ' + O(q^{%d})' % (bound + 1)

# Stored tau values are rounded in the source data, so matches are accepted only
# after a forward and inverse residual check against an adaptive precision budget.
TAU_BUCKET_SCALE = 1_000_000
TAU_SOURCE_ABS_EPS = 8.0e-8
TAU_MIN_MATCH_EPS = 7.5e-7
TAU_MAX_MATCH_EPS = 7.0e-6
RELATION_HEIGHT = 10


def _gcd4(a: int, b: int, c: int, d: int) -> int:
    return reduce(math.gcd, (abs(a), abs(b), abs(c), abs(d)))


def _mat_mul(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    a, b, c, d = left
    e, f, g, h = right
    return (a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)


def _apply_matrix(z: complex, mat: Tuple[int, int, int, int]) -> Optional[complex]:
    a, b, c, d = mat
    denom = c * z + d
    if abs(denom) < 1e-14:
        return None
    return (a * z + b) / denom


def _reduce_tau(z: complex) -> Tuple[complex, Tuple[int, int, int, int], bool]:
    # Reduce a point of the upper half-plane to the standard SL2Z domain.
    # Returns (reduced_z, R, changed) where reduced_z = R(z).
    R = (1, 0, 0, 1)
    changed = False
    z = complex(z)
    for _ in range(40):
        if z.imag <= 0:
            return z, R, changed
        n = int(round(z.real))
        if n != 0:
            T = (1, -n, 0, 1)
            z2 = _apply_matrix(z, T)
            if z2 is None:
                break
            z, R = z2, _mat_mul(T, R)
            changed = True
            continue
        # Standard fundamental-domain condition: |tau| >= 1, with boundary tie.
        if abs(z) < 1 - 1e-12 or (abs(abs(z) - 1) <= 1e-12 and z.real < -1e-12):
            S = (0, -1, 1, 0)
            z2 = _apply_matrix(z, S)
            if z2 is None:
                break
            z, R = z2, _mat_mul(S, R)
            changed = True
            continue
        break
    return z, R, changed


def _relation_string(mat: Tuple[int, int, int, int], reduced: bool) -> str:
    a, b, c, d = mat
    expr = f"({a}τ {b:+d}) / ({c}τ {d:+d})"
    return f"τ′ ≈ red({expr})" if reduced else f"τ′ ≈ {expr}"


def _tau_match_eps(mat: Tuple[int, int, int, int], tau: complex) -> float:
    a, b, c, d = mat
    det = abs(a * d - b * c)
    denom = abs(c * tau + d)
    # Propagate the stored tau rounding error through the fractional transform.
    mapped = (det / max(denom * denom, 1e-10)) * TAU_SOURCE_ABS_EPS
    coeff_boost = 1.0 + 0.025 * max(abs(a), abs(b), abs(c), abs(d))
    return min(TAU_MAX_MATCH_EPS, max(TAU_MIN_MATCH_EPS, 4.0 * mapped * coeff_boost + 2.5e-7))


@lru_cache(maxsize=1)
def relation_matrices() -> List[Tuple[int, int, int, int, int, int]]:
    mats: List[Tuple[int, int, int, int, int, int]] = []
    H = RELATION_HEIGHT
    for a in range(-H, H + 1):
        for b in range(-H, H + 1):
            for c in range(-H, H + 1):
                for d in range(-H, H + 1):
                    if a == b == c == d == 0:
                        continue
                    first = next(x for x in (a, b, c, d) if x != 0)
                    if first < 0:
                        continue
                    det = a * d - b * c
                    if det <= 0:
                        continue
                    if _gcd4(a, b, c, d) != 1:
                        continue
                    h = max(abs(a), abs(b), abs(c), abs(d))
                    mats.append((a, b, c, d, h, det))
    mats.sort(key=lambda z: (z[4], z[5], abs(z[0]) + abs(z[1]) + abs(z[2]) + abs(z[3])))
    return mats


@lru_cache(maxsize=8192)
def conductor_factors(N: int) -> Tuple[str, int]:
    fac = factorint(abs(int(N)))
    if not fac:
        return ('1', 1)
    parts = []
    for pnum in sorted(fac):
        e = fac[pnum]
        parts.append(str(pnum) if e == 1 else f'{pnum}^{e}')
    return ('·'.join(parts), max(fac))


def format_j_invariant(inv: Dict[str, int]) -> str:
    disc = int(inv['disc'])
    if disc == 0:
        return 'undefined'
    frac = Fraction(int(inv['c4']) ** 3, disc)
    return str(frac.numerator) if frac.denominator == 1 else f'{frac.numerator}/{frac.denominator}'


def weierstrass_equation(a1: int, a2: int, a3: int, a4: int, a6: int) -> str:
    def signed_term(coef: int, body: str) -> str:
        if coef == 0:
            return ''
        sign = '-' if coef < 0 else '+'
        mag = abs(coef)
        text = body if mag == 1 and body else f'{mag}{body}'
        return f' {sign} {text}'

    lhs = 'y²' + signed_term(a1, 'xy') + signed_term(a3, 'y')
    rhs = 'x³' + signed_term(a2, 'x²') + signed_term(a4, 'x') + signed_term(a6, '')
    return f'{lhs} = {rhs}'


def enrich_curve(d: Dict[str, Any]) -> Dict[str, Any]:
    a1, a2, a3, a4, a6 = int(d['a1']), int(d['a2']), int(d['a3']), int(d['a4']), int(d['a6'])
    inv = invariants(a1, a2, a3, a4, a6)
    sig, lp = conductor_factors(int(d['N']))
    d['prime_signature'] = sig
    d['largest_prime'] = lp
    d['disc'] = str(inv['disc'])
    d['j_str'] = format_j_invariant(inv)
    real_period = float(d.get('real_period') or 0.0)
    d['period_phase'] = (math.log2(real_period) % 1.0) if real_period > 0 else 0.0
    d['weierstrass_equation'] = weierstrass_equation(a1, a2, a3, a4, a6)
    return d


def _frac_key(value: Fraction) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f'{value.numerator}/{value.denominator}'


def _normalize_query_text(text: str) -> str:
    return (text or '').strip().replace('−', '-').replace('—', '-').replace('–', '-')


def parse_fraction_literal(text: str) -> Optional[Fraction]:
    raw = _normalize_query_text(text)
    compact = re.sub(r'\s+', '', raw)
    if not re.fullmatch(r'[+-]?\d+(?:/[+-]?\d+)?', compact):
        return None
    try:
        if '/' in compact:
            a, b = compact.split('/', 1)
            den = int(b)
            if den == 0:
                return None
            return Fraction(int(a), den)
        return Fraction(int(compact), 1)
    except Exception:
        return None


def _to_fraction(v: Any) -> Optional[Fraction]:
    if isinstance(v, Fraction):
        return v
    try:
        if hasattr(v, 'p') and hasattr(v, 'q'):
            return Fraction(int(v.p), int(v.q))
        return Fraction(v)
    except Exception:
        return None


def parse_coefficient_list(text: str) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    raw = _normalize_query_text(text)
    m = re.fullmatch(r'\[\s*(.*?)\s*\]', raw)
    if not m:
        return None
    parts = [x.strip() for x in m.group(1).split(',')]
    if len(parts) not in (2, 5):
        return None
    vals = [parse_fraction_literal(part) for part in parts]
    if any(v is None for v in vals):
        return None
    if len(vals) == 2:
        a4, a6 = vals
        return (Fraction(0), Fraction(0), Fraction(0), a4, a6)
    return tuple(vals)  # type: ignore[return-value]


def invariants_fraction(a1: Fraction, a2: Fraction, a3: Fraction, a4: Fraction, a6: Fraction) -> Dict[str, Fraction]:
    b2 = a1 * a1 + 4 * a2
    b4 = 2 * a4 + a1 * a3
    b6 = a3 * a3 + 4 * a6
    b8 = a1 * a1 * a6 + 4 * a2 * a6 - a1 * a3 * a4 + a2 * a3 * a3 - a4 * a4
    c4 = b2 * b2 - 24 * b4
    c6 = -b2 * b2 * b2 + 36 * b2 * b4 - 216 * b6
    disc = -b2 * b2 * b8 - 8 * b4 * b4 * b4 - 27 * b6 * b6 + 9 * b2 * b4 * b6
    return {'b2': b2, 'b4': b4, 'b6': b6, 'b8': b8, 'c4': c4, 'c6': c6, 'disc': disc}


def j_from_invariants_fraction(inv: Dict[str, Fraction]) -> Optional[Fraction]:
    disc = inv['disc']
    if disc == 0:
        return None
    return Fraction(inv['c4'] ** 3, disc)


def j_key_from_coeffs(coeffs: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]) -> Optional[str]:
    j = j_from_invariants_fraction(invariants_fraction(*coeffs))
    return None if j is None else _frac_key(j)


def _perfect_nth_root_int(n: int, k: int) -> Optional[int]:
    if n < 0 and k % 2 == 0:
        return None
    sign = -1 if n < 0 else 1
    n_abs = abs(n)
    if n_abs in (0, 1):
        return sign * n_abs
    lo, hi = 0, int(round(n_abs ** (1.0 / k))) + 3
    while lo <= hi:
        mid = (lo + hi) // 2
        p = mid ** k
        if p == n_abs:
            return sign * mid
        if p < n_abs:
            lo = mid + 1
        else:
            hi = mid - 1
    return None


def _perfect_nth_root_fraction(fr: Fraction, k: int) -> List[Fraction]:
    fr = Fraction(fr)
    num = _perfect_nth_root_int(fr.numerator, k)
    den = _perfect_nth_root_int(fr.denominator, k)
    if num is None or den is None or den == 0:
        return []
    root = Fraction(num, den)
    if k % 2 == 0 and root != 0:
        return [root, -root]
    return [root]


def _transform_coeffs(coeffs: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction], u: Fraction, r: Fraction, s: Fraction, t: Fraction) -> Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    a1, a2, a3, a4, a6 = coeffs
    return (
        (a1 + 2 * s) / u,
        (a2 - s * a1 + 3 * r - s * s) / (u ** 2),
        (a3 + r * a1 + 2 * t) / (u ** 3),
        (a4 - s * a3 + 2 * r * a2 - (t + r * s) * a1 + 3 * r * r - 2 * s * t) / (u ** 4),
        (a6 + r * a4 + r * r * a2 + r ** 3 - t * a3 - t * t - r * t * a1) / (u ** 6),
    )


def q_isomorphic_coeffs(source: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction], target: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]) -> bool:
    if source == target:
        return True
    inv_s = invariants_fraction(*source)
    inv_t = invariants_fraction(*target)
    js = j_from_invariants_fraction(inv_s)
    jt = j_from_invariants_fraction(inv_t)
    if js is None or jt is None or js != jt:
        return False
    c4, c6, c4p, c6p = inv_s['c4'], inv_s['c6'], inv_t['c4'], inv_t['c6']
    u_candidates: List[Fraction] = []
    if c4 != 0 and c6 != 0 and c4p != 0 and c6p != 0:
        for u in _perfect_nth_root_fraction(Fraction(c6 * c4p, c6p * c4), 2):
            u_candidates.append(u)
    elif c4 == 0 and c4p == 0 and c6 != 0 and c6p != 0:
        for u in _perfect_nth_root_fraction(Fraction(c6, c6p), 6):
            u_candidates.append(u)
    elif c6 == 0 and c6p == 0 and c4 != 0 and c4p != 0:
        for u in _perfect_nth_root_fraction(Fraction(c4, c4p), 4):
            u_candidates.append(u)
    seen = set()
    for u in u_candidates:
        if u == 0 or u in seen:
            continue
        seen.add(u)
        a1, a2, a3, a4, a6 = source
        b1, b2, b3, b4, b6 = target
        ss = (u * b1 - a1) / 2
        rr = (u * u * b2 - a2 + ss * a1 + ss * ss) / 3
        tt = (u ** 3 * b3 - a3 - rr * a1) / 2
        if _transform_coeffs(source, u, rr, ss, tt) == target:
            return True
    return False



def _normalize_math_input(text: str) -> str:
    s = _normalize_query_text(text)
    trans = str.maketrans({
        '²': '^2', '³': '^3', '⁴': '^4', '⁵': '^5', '⁶': '^6',
        '＋': '+', '－': '-', '＊': '*', '／': '/', '（': '(', '）': ')', '＝': '=',
        '·': '*', '×': '*', '–': '-', '—': '-', '−': '-',
    })
    return s.translate(trans)


def _fraction_to_expr(fr: Fraction) -> str:
    fr = Fraction(fr)
    return str(fr.numerator) if fr.denominator == 1 else f'({fr.numerator}/{fr.denominator})'


def _coeffs_to_weierstrass_affine_expr(coeffs: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]) -> str:
    a1, a2, a3, a4, a6 = coeffs
    # Affine cubic F(x,y)=0 for y^2+a1xy+a3y = x^3+a2x^2+a4x+a6.
    return (
        f'y^2 + ({_fraction_to_expr(a1)})*x*y + ({_fraction_to_expr(a3)})*y '
        f'- x^3 - ({_fraction_to_expr(a2)})*x^2 - ({_fraction_to_expr(a4)})*x - ({_fraction_to_expr(a6)})'
    )


# ---------------------------------------------------------------------------
# Pure-Python rational polynomial parser and cubic normalizer.
# v14 used SymPy/PARI when available. v15 deliberately keeps this path inside
# the standard library: exact Fraction arithmetic, a small recursive-descent
# parser, and the Cremona/Laska-Kraus-Connell reduced-minimal-model formulas.
# ---------------------------------------------------------------------------

Poly = Dict[Tuple[int, ...], Fraction]


def _poly_clean(poly: Poly) -> Poly:
    return {m: Fraction(c) for m, c in poly.items() if c}


def _poly_const(value: Fraction, nvars: int) -> Poly:
    value = Fraction(value)
    return {} if value == 0 else {(0,) * nvars: value}


def _poly_var(index: int, nvars: int) -> Poly:
    exp = [0] * nvars
    exp[index] = 1
    return {tuple(exp): Fraction(1)}


def _poly_add(a: Poly, b: Poly) -> Poly:
    out = dict(a)
    for m, c in b.items():
        out[m] = out.get(m, Fraction(0)) + c
        if out[m] == 0:
            out.pop(m, None)
    return out


def _poly_neg(a: Poly) -> Poly:
    return {m: -c for m, c in a.items()}


def _poly_sub(a: Poly, b: Poly) -> Poly:
    return _poly_add(a, _poly_neg(b))


def _poly_mul(a: Poly, b: Poly, max_degree: int = 12) -> Poly:
    out: Poly = {}
    if not a or not b:
        return out
    for ma, ca in a.items():
        for mb, cb in b.items():
            m = tuple(x + y for x, y in zip(ma, mb))
            if sum(m) > max_degree:
                raise ValueError('degree too large')
            out[m] = out.get(m, Fraction(0)) + ca * cb
    return _poly_clean(out)


def _poly_scale(a: Poly, k: Fraction) -> Poly:
    k = Fraction(k)
    return {} if k == 0 else {m: c * k for m, c in a.items() if c * k}


def _poly_pow(a: Poly, exponent: int, max_degree: int = 12) -> Poly:
    if exponent < 0:
        raise ValueError('negative polynomial power')
    nvars = len(next(iter(a))) if a else 1
    out = _poly_const(Fraction(1), nvars)
    base = dict(a)
    e = exponent
    while e:
        if e & 1:
            out = _poly_mul(out, base, max_degree=max_degree)
        e >>= 1
        if e:
            base = _poly_mul(base, base, max_degree=max_degree)
    return out


def _poly_degree(poly: Poly) -> int:
    return max((sum(m) for m in poly), default=0)


def _poly_coeff(poly: Poly, monom: Tuple[int, ...]) -> Fraction:
    return poly.get(tuple(monom), Fraction(0))


def _candidate_symbols(raw: str) -> List[str]:
    names: List[str] = []
    for ch in re.findall(r'[A-Za-z]', raw):
        if ch.lower() == 'e':
            continue
        if ch not in names:
            names.append(ch)
    lowered = {n.lower(): n for n in names}
    ordered: List[str] = []
    for preferred in ('x', 'y'):
        if preferred in lowered and lowered[preferred] not in ordered:
            ordered.append(lowered[preferred])
    for n in names:
        if n not in ordered:
            ordered.append(n)
    return ordered[:6]


_TOKEN_RE = re.compile(r'\s*(?:(\d+(?:/\d+)?)|([A-Za-z])|([+\-*/^()]))')


def _tokenize_math(raw: str) -> Optional[List[Tuple[str, str]]]:
    out: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(raw):
        if raw[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(raw, pos)
        if not m:
            return None
        if m.group(1) is not None:
            out.append(('num', m.group(1)))
        elif m.group(2) is not None:
            out.append(('var', m.group(2)))
        else:
            out.append(('op', m.group(3)))
        pos = m.end()
    out.append(('eof', ''))
    return out


def _parse_number_token(value: str) -> Fraction:
    if '/' in value:
        a, b = value.split('/', 1)
        if int(b) == 0:
            raise ValueError('zero denominator')
        return Fraction(int(a), int(b))
    return Fraction(int(value), 1)


class _MathParser:
    def __init__(self, raw: str, names: List[str]):
        toks = _tokenize_math(raw)
        if toks is None:
            raise ValueError('bad token')
        self.toks = toks
        self.i = 0
        self.names = names
        self.index = {name: j for j, name in enumerate(names)}
        self.nvars = len(names)

    def _peek(self) -> Tuple[str, str]:
        return self.toks[self.i]

    def _take(self) -> Tuple[str, str]:
        tok = self.toks[self.i]
        self.i += 1
        return tok

    def parse(self) -> Poly:
        poly = self.expr()
        if self._peek()[0] != 'eof':
            raise ValueError('trailing tokens')
        return poly

    def expr(self) -> Poly:
        out = self.term()
        while self._peek() == ('op', '+') or self._peek() == ('op', '-'):
            op = self._take()[1]
            rhs = self.term()
            out = _poly_add(out, rhs) if op == '+' else _poly_sub(out, rhs)
        return out

    def _starts_factor(self, tok: Tuple[str, str]) -> bool:
        typ, val = tok
        return typ in {'num', 'var'} or (typ == 'op' and val == '(')

    def term(self) -> Poly:
        out = self.power()
        while True:
            tok = self._peek()
            if tok == ('op', '*'):
                self._take()
                out = _poly_mul(out, self.power(), max_degree=12)
            elif tok == ('op', '/'):
                self._take()
                rhs = self.power()
                if len(rhs) != 1:
                    raise ValueError('division by non-constant')
                coeff = next(iter(rhs.values()))
                if coeff == 0:
                    raise ValueError('zero division')
                out = _poly_scale(out, Fraction(1, 1) / coeff)
            elif self._starts_factor(tok):
                # Common mathematical shorthand: 7820x, 15XY, x(y+1).
                out = _poly_mul(out, self.power(), max_degree=12)
            else:
                break
        return out

    def power(self) -> Poly:
        base = self.factor()
        if self._peek() == ('op', '^'):
            self._take()
            sign = 1
            if self._peek() == ('op', '+'):
                self._take()
            elif self._peek() == ('op', '-'):
                self._take(); sign = -1
            tok = self._take()
            if tok[0] != 'num' or '/' in tok[1]:
                raise ValueError('non-integer exponent')
            exp = sign * int(tok[1])
            base = _poly_pow(base, exp, max_degree=12)
        return base

    def factor(self) -> Poly:
        tok = self._take()
        typ, val = tok
        if typ == 'num':
            return _poly_const(_parse_number_token(val), self.nvars)
        if typ == 'var':
            if val not in self.index:
                raise ValueError('unknown variable')
            return _poly_var(self.index[val], self.nvars)
        if tok == ('op', '+'):
            return self.factor()
        if tok == ('op', '-'):
            return _poly_neg(self.factor())
        if tok == ('op', '('):
            out = self.expr()
            if self._take() != ('op', ')'):
                raise ValueError('missing right parenthesis')
            return out
        raise ValueError('bad factor')


def _parse_poly_expr(raw: str, names: List[str]) -> Optional[Poly]:
    try:
        return _MathParser(raw, names).parse()
    except Exception:
        return None


def _poly_from_input(text: str) -> Optional[Tuple[Poly, Tuple[str, str]]]:
    raw = _normalize_math_input(text)
    if not raw:
        return None
    names = _candidate_symbols(raw)
    if len(names) < 2:
        return None
    if '=' in raw:
        left, right = raw.split('=', 1)
        lp = _parse_poly_expr(left, names)
        rp = _parse_poly_expr(right, names)
        if lp is None or rp is None:
            return None
        poly = _poly_sub(lp, rp)
    else:
        poly = _parse_poly_expr(raw, names)
        if poly is None:
            return None
    used = [name for i, name in enumerate(names) if any(m[i] for m in poly)]
    if len(used) != 2:
        return None
    # Repack to exactly two variables. Preserve x/y preference; otherwise preserve first appearance.
    idx = [names.index(name) for name in used]
    repacked: Poly = {}
    for m, c in poly.items():
        new_m = (m[idx[0]], m[idx[1]])
        repacked[new_m] = repacked.get(new_m, Fraction(0)) + c
    repacked = _poly_clean(repacked)
    deg = _poly_degree(repacked)
    if deg > 3 or deg < 2:
        return None
    return repacked, (used[0], used[1])


# Backward-compatible internal name. It no longer imports SymPy.
def _sympy_poly_from_input(text: str):
    return _poly_from_input(text)


def _standard_weierstrass_from_poly(poly_obj) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    try:
        poly, symbols = poly_obj
        # Try both variable orders. This makes v^2+v=u^3-u^2... work even when
        # the non-standard variable names are alphabetically reversed.
        for swap in (False, True):
            if not swap:
                P = dict(poly)
            else:
                P = {(j, i): c for (i, j), c in poly.items()}
            allowed = {(0, 2), (1, 1), (0, 1), (3, 0), (2, 0), (1, 0), (0, 0)}
            if any(monom not in allowed and coeff != 0 for monom, coeff in P.items()):
                continue
            cy2 = _poly_coeff(P, (0, 2))
            cx3 = _poly_coeff(P, (3, 0))
            if cy2 == 0 or cx3 == 0:
                continue
            Pn = _poly_scale(P, Fraction(1, 1) / cy2)
            if _poly_coeff(Pn, (0, 2)) != 1 or _poly_coeff(Pn, (3, 0)) != -1:
                continue
            vals = (
                _poly_coeff(Pn, (1, 1)),
                -_poly_coeff(Pn, (2, 0)),
                _poly_coeff(Pn, (0, 1)),
                -_poly_coeff(Pn, (1, 0)),
                -_poly_coeff(Pn, (0, 0)),
            )
            return q_minimal_model(vals)
    except Exception:
        return None
    return None


def _lcm_int(a: int, b: int) -> int:
    if a == 0 or b == 0:
        return abs(a or b)
    return abs(a // math.gcd(a, b) * b)


def _denominator_lcm(values: Tuple[Fraction, ...]) -> int:
    d = 1
    for v in values:
        d = _lcm_int(d, Fraction(v).denominator)
    return max(d, 1)


def _centered_mod(n: int, m: int) -> int:
    r = n % m
    if r > m // 2:
        r -= m
    return r


def _prime_divisors(n: int) -> List[int]:
    return sorted(factorint(abs(int(n))).keys())


def _reduced_model_from_integral_invariants(c4: int, c6: int) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    delta_num = c4 ** 3 - c6 ** 2
    if delta_num == 0 or delta_num % 1728 != 0:
        return None
    delta = delta_num // 1728
    g = math.gcd(c6 * c6, delta)
    u = 1
    for p in _prime_divisors(g):
        d = v_p(g, p) // 12
        if d <= 0:
            continue
        if p == 2:
            p4 = p ** (4 * d)
            p6 = p ** (6 * d)
            if c4 % p4 or c6 % p6:
                d -= 1
            else:
                a = (c4 // p4) % 16
                b = (c6 // p6) % 32
                # Kraus condition at 2: c6 == -1 mod 4, or c4 == 0 mod 16 and c6 == 0 or 8 mod 32.
                if (b % 4 != 3) and not (a == 0 and b in (0, 8)):
                    d -= 1
        elif p == 3:
            if v_p(c6, 3) == 6 * d + 2:
                d -= 1
        if d > 0:
            u *= p ** d
    if c4 % (u ** 4) or c6 % (u ** 6):
        return None
    c4m = c4 // (u ** 4)
    c6m = c6 // (u ** 6)
    if (c4m ** 3 - c6m ** 2) == 0 or (c4m ** 3 - c6m ** 2) % 1728 != 0:
        return None
    b2 = _centered_mod(-c6m, 12)
    b4_num = b2 * b2 - c4m
    if b4_num % 24:
        return None
    b4 = b4_num // 24
    b6_num = -b2 ** 3 + 36 * b2 * b4 - c6m
    if b6_num % 216:
        return None
    b6 = b6_num // 216
    a1 = b2 % 2
    a3 = b6 % 2
    if (b2 - a1) % 4 or (b4 - a1 * a3) % 2 or (b6 - a3) % 4:
        return None
    a2 = (b2 - a1) // 4
    a4 = (b4 - a1 * a3) // 2
    a6 = (b6 - a3) // 4
    return (Fraction(a1), Fraction(a2), Fraction(a3), Fraction(a4), Fraction(a6))


def q_minimal_model(coeffs: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]) -> Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]:
    coeffs = tuple(Fraction(v) for v in coeffs)  # type: ignore[assignment]
    inv0 = invariants_fraction(*coeffs)
    if inv0['disc'] == 0:
        return coeffs
    # Clear denominators by the scaling x=u^2*x', y=u^3*y' with u=1/D.
    D = _denominator_lcm(coeffs)
    weights = (1, 2, 3, 4, 6)
    scaled = tuple(coeffs[i] * (D ** weights[i]) for i in range(5))
    if all(v.denominator == 1 for v in scaled):
        inv = invariants_fraction(*scaled)
        c4, c6 = inv['c4'], inv['c6']
        if c4.denominator == 1 and c6.denominator == 1:
            red = _reduced_model_from_integral_invariants(int(c4), int(c6))
            if red is not None:
                return red
    return coeffs


def _diagonal_hesse_j_from_poly(poly_obj) -> Optional[Fraction]:
    try:
        poly, symbols = poly_obj
        allowed = {(3, 0), (0, 3), (1, 1), (0, 0)}
        if any(monom not in allowed and coeff != 0 for monom, coeff in poly.items()):
            return None
        A = _poly_coeff(poly, (3, 0))
        B = _poly_coeff(poly, (0, 3))
        C = _poly_coeff(poly, (0, 0))
        E = _poly_coeff(poly, (1, 1))
        if A == 0 or B == 0 or C == 0 or E == 0:
            return None
        D = -E
        k3 = Fraction(D ** 3, 27 * A * B * C)
        if k3 == 1:
            return None
        return Fraction(27 * k3 * (k3 + 8) ** 3, (k3 - 1) ** 3)
    except Exception:
        return None



def _ternary_cubic_coefficients_aronhold(poly: Poly) -> Tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]:
    """Return coefficients (a,b,c,d,e,f,g,h,i,j) in Aronhold's basis.

    The homogeneous ternary cubic is written as

        aX^3 + bY^3 + cZ^3 + 3dX^2Y + 3eY^2Z + 3fZ^2X
        + 3gXY^2 + 3hYZ^2 + 3iZX^2 + 6jXYZ.

    The divisions by 3 and 6 are intentional: this is the classical normalized
    coordinate system in which the S and T formulae are shortest.
    """
    return (
        _poly_coeff(poly, (3, 0, 0)),
        _poly_coeff(poly, (0, 3, 0)),
        _poly_coeff(poly, (0, 0, 3)),
        _poly_coeff(poly, (2, 1, 0)) / 3,
        _poly_coeff(poly, (0, 2, 1)) / 3,
        _poly_coeff(poly, (1, 0, 2)) / 3,
        _poly_coeff(poly, (1, 2, 0)) / 3,
        _poly_coeff(poly, (0, 1, 2)) / 3,
        _poly_coeff(poly, (2, 0, 1)) / 3,
        _poly_coeff(poly, (1, 1, 1)) / 6,
    )


def _aronhold_st_invariants(poly: Poly) -> Optional[Tuple[Fraction, Fraction]]:
    """Compute Aronhold S,T invariants for a homogeneous ternary cubic.

    This pure-Python exact formula gives the Jacobian j-invariant even when no
    rational point or rational flex has been found on the plane cubic. It is a
    fallback invariant computation, not a minimal-model construction.
    """
    try:
        if any(sum(m) != 3 for m in poly):
            return None
        a, b, c, d, e, f, g, h, i, j = _ternary_cubic_coefficients_aronhold(poly)
        S = (
            a*g*e*c - a*g*h**2 - a*j*b*c + a*j*e*h + a*f*b*h - a*f*e**2
            - d**2*e*c + d**2*h**2 + d*i*b*c - d*i*e*h + d*g*j*c
            - d*g*f*h - 2*d*j**2*h + 3*d*j*f*e - d*f**2*b
            - i**2*b*h + i**2*e**2 - i*g**2*c + 3*i*g*j*h - i*g*f*e
            - 2*i*j**2*e + i*j*f*b + g**2*f**2 - 2*g*j**2*f + j**4
        )
        T = (
            a**2*b**2*c**2 - 3*a**2*e**2*h**2 - 6*a**2*b*e*h*c
            + 4*a**2*b*h**3 + 4*a**2*e**3*c - 6*a*d*g*b*c**2
            + 18*a*d*g*e*h*c - 12*a*d*g*h**3 + 12*a*d*j*b*h*c
            - 24*a*d*j*e**2*c + 12*a*d*j*e*h**2 - 12*a*d*f*b*h**2
            + 6*a*d*f*b*e*c + 6*a*d*f*e**2*h + 6*a*i*g*b*h*c
            - 12*a*i*g*e**2*c + 6*a*i*g*e*h**2 + 12*a*i*j*b*e*c
            + 12*a*i*j*e**2*h - 6*a*i*f*b**2*c + 18*a*i*f*b*e*h
            - 24*a*g**2*j*h*c - 24*a*i*j*b*h**2 - 12*a*i*f*e**3
            + 4*a*g**3*c**2 - 12*a*g**2*f*e*c + 24*a*g**2*f*h**2
            + 36*a*g*j**2*e*c + 12*a*g*j**2*h**2 + 12*a*g*j*f*b*c
            - 60*a*g*j*f*e*h - 12*a*g*f**2*b*h + 24*a*g*f**2*e**2
            - 20*a*j**3*b*c - 12*a*j**3*e*h + 36*a*j**2*f*b*h
            + 12*a*j**2*f*e**2 - 24*a*j*f**2*b*e + 4*a*f**3*b**2
            + 4*d**3*b*c**2 - 12*d**3*e*h*c + 8*d**3*h**3
            + 24*d**2*i*e**2*c - 12*d**2*i*e*h**2 + 12*d**2*g*j*h*c
            + 6*d**2*g*f*e*c - 24*d**2*j**2*h**2 - 12*d**2*i*b*h*c
            - 3*d**2*g**2*c**2 - 24*g**2*j**2*f**2 + 24*g*j**4*f
            - 12*d**2*g*f*h**2 + 12*d**2*j**2*e*c - 24*d**2*j*f*b*c
            - 27*d**2*f**2*e**2 + 36*d**2*j*f*e*h + 24*d**2*f**2*b*h
            + 24*d*i**2*b*h**2 - 12*d*i**2*b*e*c - 12*d*i**2*e**2*h
            + 6*d*i*g**2*h*c - 60*d*i*g*j*e*c + 36*d*i*g*j*h**2
            + 18*d*i*g*f*b*c - 6*d*i*g*f*e*h + 36*d*i*j**2*b*c
            - 12*d*i*j**2*e*h - 60*d*i*j*f*b*h + 36*d*i*j*f*e**2
            + 6*d*i*f**2*b*e + 12*d*g**2*j*f*c - 12*d*g*j**3*c
            - 12*d*g*j**2*f*h + 36*d*g*j*f**2*e - 12*d*g*f**3*b
            + 24*d*j**4*h + 12*d*j**2*f**2*b + 4*i**3*b**2*c
            + 24*i**2*g**2*e*c - 27*i**2*g**2*h**2 - 36*d*j**3*f*e
            - 12*i**3*b*e*h + 8*i**3*e**3 - 24*i**2*g*j*b*c
            + 36*i**2*g*j*e*h + 6*i**2*g*f*b*h + 12*i**2*j**2*b*h
            - 3*i**2*f**2*b**2 - 12*d*g**2*f**2*h - 12*i**2*g*f*e**2
            - 24*i**2*j**2*e**2 + 12*i**2*j*f*b*e - 12*i*g**3*f*c
            + 12*i*g**2*j**2*c + 36*i*g**2*j*f*h - 12*i*g**2*f**2*e
            - 36*i*g*j**3*h - 12*i*g*j**2*f*e + 12*i*g*j*f**2*b
            + 24*i*j**4*e - 12*i*j**3*f*b + 8*g**3*f**3 - 8*j**6
        )
        return Fraction(S), Fraction(T)
    except Exception:
        return None


def _ternary_cubic_j_from_aronhold(poly_obj) -> Optional[Fraction]:
    try:
        poly2, _symbols = poly_obj
        if _poly_degree(poly2) != 3:
            return None
        F = _homogenize_affine_cubic(poly2)
        st = _aronhold_st_invariants(F)
        if st is None:
            return None
        S, T = st
        denom = (4 * S) ** 3 - T ** 2
        if denom == 0:
            return None
        return Fraction(1728 * (4 * S) ** 3, denom)
    except Exception:
        return None

def _homogenize_affine_cubic(poly: Poly) -> Poly:
    out: Poly = {}
    for (i, j), c in poly.items():
        k = 3 - i - j
        if k < 0:
            raise ValueError('not cubic')
        out[(i, j, k)] = out.get((i, j, k), Fraction(0)) + c
    return _poly_clean(out)


def _eval_hom3(poly: Poly, P: Tuple[Fraction, Fraction, Fraction]) -> Fraction:
    X, Y, Z = P
    total = Fraction(0)
    for (i, j, k), c in poly.items():
        total += c * (X ** i) * (Y ** j) * (Z ** k)
    return total


def _deriv_hom3(poly: Poly, idx: int) -> Poly:
    out: Poly = {}
    for m, c in poly.items():
        if m[idx] == 0:
            continue
        mm = list(m)
        mm[idx] -= 1
        out[tuple(mm)] = out.get(tuple(mm), Fraction(0)) + c * m[idx]
    return _poly_clean(out)


def _grad_hom3(poly: Poly, P: Tuple[Fraction, Fraction, Fraction]) -> Tuple[Fraction, Fraction, Fraction]:
    return tuple(_eval_hom3(_deriv_hom3(poly, i), P) for i in range(3))  # type: ignore[return-value]


def _primitive_point(P: Tuple[Fraction, Fraction, Fraction]) -> Tuple[Fraction, Fraction, Fraction]:
    # Normalize only sign and common integer scale for stable projective comparisons.
    dens = [v.denominator for v in P]
    D = 1
    for d in dens:
        D = _lcm_int(D, d)
    ints = [int(v * D) for v in P]
    g = 0
    for a in ints:
        g = math.gcd(g, abs(a))
    if g:
        ints = [a // g for a in ints]
    first = next((a for a in ints if a), 1)
    if first < 0:
        ints = [-a for a in ints]
    return (Fraction(ints[0]), Fraction(ints[1]), Fraction(ints[2]))


def _same_projective(P: Tuple[Fraction, Fraction, Fraction], Q: Tuple[Fraction, Fraction, Fraction]) -> bool:
    return (
        P[0] * Q[1] == P[1] * Q[0]
        and P[0] * Q[2] == P[2] * Q[0]
        and P[1] * Q[2] == P[2] * Q[1]
    )


def _smooth_at(poly: Poly, P: Tuple[Fraction, Fraction, Fraction]) -> bool:
    return any(g != 0 for g in _grad_hom3(poly, P))


def _find_small_rational_point(poly: Poly, bound: int = 8) -> Optional[Tuple[Fraction, Fraction, Fraction]]:
    seen = set()
    # Prioritize points at infinity: many Weierstrass/Hesse-style inputs expose them.
    for B in range(1, bound + 1):
        candidates = []
        for z_values in ([0], range(-B, B + 1)):
            for X in range(-B, B + 1):
                for Y in range(-B, B + 1):
                    for Z in z_values:
                        if X == Y == Z == 0 or max(abs(X), abs(Y), abs(Z)) != B:
                            continue
                        g = math.gcd(math.gcd(abs(X), abs(Y)), abs(Z))
                        if g != 1:
                            continue
                        P = _primitive_point((Fraction(X), Fraction(Y), Fraction(Z)))
                        if P in seen:
                            continue
                        seen.add(P)
                        candidates.append(P)
        for P in candidates:
            if _eval_hom3(poly, P) == 0 and _smooth_at(poly, P):
                return P
    return None


def _point_on_line_not_multiple(line: Tuple[Fraction, Fraction, Fraction], P: Tuple[Fraction, Fraction, Fraction], require_not_on_curve: Optional[Poly] = None) -> Optional[Tuple[Fraction, Fraction, Fraction]]:
    A, B, C = line
    candidates = [
        (B, -A, Fraction(0)),
        (C, Fraction(0), -A),
        (Fraction(0), C, -B),
        (B + C, -A, -A),
        (C, C, -A - B),
    ]
    for Q in candidates:
        if Q == (0, 0, 0):
            continue
        if A * Q[0] + B * Q[1] + C * Q[2] != 0:
            continue
        Q = _primitive_point(Q)
        if _same_projective(Q, P):
            continue
        if require_not_on_curve is not None and _eval_hom3(require_not_on_curve, Q) == 0:
            continue
        return Q
    return None


def _line_substitution_coeffs(poly: Poly, P: Tuple[Fraction, Fraction, Fraction], Q: Tuple[Fraction, Fraction, Fraction]) -> List[Fraction]:
    # F(P + t*(Q-P)) as a polynomial in t, coefficient list degree 0..3.
    D = tuple(Q[i] - P[i] for i in range(3))
    coeffs = [Fraction(0), Fraction(0), Fraction(0), Fraction(0)]

    def lin_pow(a: Fraction, b: Fraction, e: int) -> List[Fraction]:
        vals = [Fraction(0)] * (e + 1)
        for r in range(e + 1):
            vals[r] = Fraction(math.comb(e, r)) * (a ** (e - r)) * (b ** r)
        return vals

    for (i, j, k), c in poly.items():
        px = lin_pow(P[0], D[0], i)
        py = lin_pow(P[1], D[1], j)
        pz = lin_pow(P[2], D[2], k)
        tmp = [Fraction(0)] * 4
        for a_i, ca in enumerate(px):
            for b_i, cb in enumerate(py):
                for c_i, cc in enumerate(pz):
                    tmp[a_i + b_i + c_i] += c * ca * cb * cc
        for n in range(4):
            coeffs[n] += tmp[n]
    return coeffs


def _tangent_third_point(poly: Poly, P: Tuple[Fraction, Fraction, Fraction]) -> Tuple[Fraction, Fraction, Fraction]:
    grad = _grad_hom3(poly, P)
    Q = _point_on_line_not_multiple(grad, P)
    if Q is None:
        return P
    c = _line_substitution_coeffs(poly, P, Q)
    # Tangency gives c0=c1=0. Remaining root is -c2/c3 when c3 != 0.
    if c[3] == 0:
        return P
    t3 = -c[2] / c[3]
    R = tuple(P[i] + t3 * (Q[i] - P[i]) for i in range(3))
    return _primitive_point(R)  # type: ignore[arg-type]


def _mat_det3(M: Tuple[Tuple[Fraction, Fraction, Fraction], Tuple[Fraction, Fraction, Fraction], Tuple[Fraction, Fraction, Fraction]]) -> Fraction:
    return (
        M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
        - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
        + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0])
    )


def _matrix_from_columns(cols: List[Tuple[Fraction, Fraction, Fraction]]) -> Tuple[Tuple[Fraction, Fraction, Fraction], Tuple[Fraction, Fraction, Fraction], Tuple[Fraction, Fraction, Fraction]]:
    return tuple(tuple(cols[j][i] for j in range(3)) for i in range(3))  # type: ignore[return-value]


def _substitute_linear_hom3(poly: Poly, M: Tuple[Tuple[Fraction, Fraction, Fraction], Tuple[Fraction, Fraction, Fraction], Tuple[Fraction, Fraction, Fraction]]) -> Poly:
    # old variables are rows of M applied to new variables (U,V,T).
    forms = []
    for i in range(3):
        forms.append({(1, 0, 0): M[i][0], (0, 1, 0): M[i][1], (0, 0, 1): M[i][2]})
    out: Poly = {}
    for m, c in poly.items():
        term = _poly_const(c, 3)
        for idx, exp in enumerate(m):
            if exp:
                term = _poly_mul(term, _poly_pow(forms[idx], exp, max_degree=3), max_degree=3)
        out = _poly_add(out, term)
    return _poly_clean(out)


def _choose_R_not_on_line(line: Tuple[Fraction, Fraction, Fraction]) -> Optional[Tuple[Fraction, Fraction, Fraction]]:
    A, B, C = line
    for R in ((Fraction(1), Fraction(0), Fraction(0)), (Fraction(0), Fraction(1), Fraction(0)), (Fraction(0), Fraction(0), Fraction(1)), (Fraction(1), Fraction(1), Fraction(1))):
        if A * R[0] + B * R[1] + C * R[2] != 0:
            return R
    return None


def _weierstrass_from_flex(poly: Poly, P: Tuple[Fraction, Fraction, Fraction]) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    tangent = _grad_hom3(poly, P)
    for _ in range(6):
        Q = _point_on_line_not_multiple(tangent, P, require_not_on_curve=poly)
        R = _choose_R_not_on_line(tangent)
        if Q is None or R is None:
            return None
        M = _matrix_from_columns([Q, P, R])
        if _mat_det3(M) == 0:
            return None
        G = _substitute_linear_hom3(poly, M)
        k = _poly_coeff(G, (3, 0, 0))
        c0raw = _poly_coeff(G, (0, 2, 1))
        if k == 0 or c0raw == 0:
            return None
        c2 = _poly_coeff(G, (2, 0, 1)) / k
        c1 = _poly_coeff(G, (1, 1, 1)) / k
        c0 = c0raw / k
        c4 = _poly_coeff(G, (1, 0, 2)) / k
        c3 = _poly_coeff(G, (0, 1, 2)) / k
        c6 = _poly_coeff(G, (0, 0, 3)) / k
        coeffs = (
            c1 / c0,
            -c2 / c0,
            -c3 / (c0 ** 2),
            c4 / (c0 ** 2),
            -c6 / (c0 ** 3),
        )
        return q_minimal_model(coeffs)
    return None


def _nonflex_to_weierstrass(poly: Poly, P: Tuple[Fraction, Fraction, Fraction], P2: Tuple[Fraction, Fraction, Fraction], P3: Tuple[Fraction, Fraction, Fraction]) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    M = _matrix_from_columns([P, P2, P3])
    if _mat_det3(M) == 0:
        return None
    F2 = _substitute_linear_hom3(poly, M)
    F3: Poly = {}
    for (i, j, k), c in F2.items():
        # x=U^2, y=V*W, z=U*W; divide by U^2*W.
        m = (2 * i + k - 2, j, j + k - 1)
        if any(e < 0 for e in m):
            return None
        F3[m] = F3.get(m, Fraction(0)) + c
    F3 = _poly_clean(F3)
    a = _poly_coeff(F3, (3, 0, 0))
    if a == 0:
        return None
    F4 = _poly_scale(F3, Fraction(1, 1) / a)
    b = _poly_coeff(F4, (0, 2, 1))
    if b == 0:
        return None
    F5: Poly = {}
    zval = -Fraction(1, 1) / b
    for (i, j, k), c in F4.items():
        m = (i, j)
        F5[m] = F5.get(m, Fraction(0)) + c * (zval ** k)
    F5 = _poly_clean(F5)
    coeffs = _standard_weierstrass_from_poly((F5, ('x', 'y')))
    return coeffs


def _general_cubic_to_weierstrass(poly_obj) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    try:
        poly2, _symbols = poly_obj
        if _poly_degree(poly2) != 3:
            return None
        F = _homogenize_affine_cubic(poly2)
        P = _find_small_rational_point(F, bound=8)
        if P is None:
            return None
        P2 = _tangent_third_point(F, P)
        if _same_projective(P2, P):
            return _weierstrass_from_flex(F, P)
        P3 = _tangent_third_point(F, P2)
        if _same_projective(P3, P2):
            return _weierstrass_from_flex(F, P2)
        return _nonflex_to_weierstrass(F, P, P2, P3)
    except Exception:
        return None


def parse_general_cubic_query(text: str) -> Optional[Dict[str, Any]]:
    # Unified entry: coefficient lists are first converted to the same affine cubic
    # polynomial path used by typed equations. All recognition below is standard-library only.
    coeffs = parse_coefficient_list(text)
    if coeffs is not None:
        poly_obj = _poly_from_input(_coeffs_to_weierstrass_affine_expr(coeffs))
        if poly_obj is not None:
            parsed = _standard_weierstrass_from_poly(poly_obj)
            if parsed is not None:
                return {'coeffs': parsed, 'match': 'Q-minimal model from Weierstrass coefficients'}
        return {'coeffs': q_minimal_model(coeffs), 'match': 'Q-minimal model from Weierstrass coefficients'}

    poly_obj = _poly_from_input(text)
    if poly_obj is None:
        return None

    coeffs = _standard_weierstrass_from_poly(poly_obj)
    if coeffs is not None:
        return {'coeffs': coeffs, 'match': 'Q-minimal model from Weierstrass equation'}

    coeffs = _general_cubic_to_weierstrass(poly_obj)
    if coeffs is not None:
        return {'coeffs': coeffs, 'match': 'Q-minimal model from rational-point plane cubic'}

    aronhold_j = _ternary_cubic_j_from_aronhold(poly_obj)
    if aronhold_j is not None:
        return {'j': aronhold_j, 'match': 'same j-invariant from Aronhold ternary-cubic invariants'}

    hesse_j = _diagonal_hesse_j_from_poly(poly_obj)
    if hesse_j is not None:
        return {'j': hesse_j, 'match': 'same j-invariant from diagonal Hesse cubic'}

    return None


# Backward-compatible names retained for any local scripts that imported v13 helpers.
def parse_weierstrass_polynomial(text: str) -> Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]]:
    parsed = parse_general_cubic_query(text)
    if parsed and parsed.get('coeffs') is not None:
        return parsed['coeffs']
    return None


def parse_diagonal_hesse_j(text: str) -> Optional[Fraction]:
    parsed = parse_general_cubic_query(text)
    if parsed and parsed.get('j') is not None:
        return parsed['j']
    return None


SEARCH_COLUMNS = 'id,label,cremona,iso,N,rank,tor_label,tor_order,cm,st,az,alt,class_index,class_size,a1,a2,a3,a4,a6,real_period'


@lru_cache(maxsize=1)
def search_index() -> Dict[str, Any]:
    con = get_connection()
    rows: Dict[int, Dict[str, Any]] = {}
    disc_map: Dict[str, List[int]] = {}
    j_map: Dict[str, List[int]] = {}
    for r in con.execute(f'select {SEARCH_COLUMNS} from curves'):
        d = dict(r)
        inv = invariants(int(d['a1']), int(d['a2']), int(d['a3']), int(d['a4']), int(d['a6']))
        disc_key = _frac_key(Fraction(inv['disc'], 1))
        j_key = format_j_invariant(inv)
        rows[int(d['id'])] = d
        disc_map.setdefault(disc_key, []).append(int(d['id']))
        j_map.setdefault(j_key, []).append(int(d['id']))
    con.close()
    return {'rows': rows, 'disc': disc_map, 'j': j_map}


def _search_result(row: Dict[str, Any], match: str = '') -> Dict[str, Any]:
    d = enrich_curve(dict(row))
    if match:
        d['search_match'] = match
    return d


def _add_unique(out: List[Dict[str, Any]], seen: set, row: Dict[str, Any], match: str, limit: int) -> None:
    rid = int(row['id'])
    if rid in seen or len(out) >= limit:
        return
    seen.add(rid)
    out.append(_search_result(row, match))


def _search_by_j_key(j_key: str, limit: int, match: str, prefer_qiso: Optional[Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]] = None, qiso_match: str = 'Q-isomorphic Weierstrass model') -> List[Dict[str, Any]]:
    idx = search_index()
    ids = idx['j'].get(j_key, [])
    rows = idx['rows']
    out: List[Dict[str, Any]] = []
    seen = set()
    if prefer_qiso is not None:
        for rid in ids:
            row = rows[rid]
            target = tuple(Fraction(int(row[k]), 1) for k in ('a1', 'a2', 'a3', 'a4', 'a6'))
            if q_isomorphic_coeffs(prefer_qiso, target):
                _add_unique(out, seen, row, qiso_match, limit)
    if not out:
        for rid in ids:
            _add_unique(out, seen, rows[rid], match, limit)
    return out[:limit]


@lru_cache(maxsize=1)
def tau_index():
    con = get_connection()
    rows = [dict(r) for r in con.execute('''
      select id,label,cremona,iso,N,rank,tor_label,tor_order,cm,class_index,class_size,az,alt,tau_re,tau_im,a1,a2,a3,a4,a6,real_period
      from curves
    ''')]
    con.close()
    buckets: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
    for r in rows:
        key = (round(float(r['tau_re']) * TAU_BUCKET_SCALE), round(float(r['tau_im']) * TAU_BUCKET_SCALE))
        buckets.setdefault(key, []).append(r)
    return rows, buckets


def _bucket_candidates(buckets: Dict[Tuple[int, int], List[Dict[str, Any]]], target: complex, eps: float):
    kr = round(target.real * TAU_BUCKET_SCALE)
    ki = round(target.imag * TAU_BUCKET_SCALE)
    rad = max(1, min(12, int(math.ceil(eps * TAU_BUCKET_SCALE)) + 1))
    for dr in range(-rad, rad + 1):
        for di in range(-rad, rad + 1):
            yield from buckets.get((kr + dr, ki + di), [])


def detected_c_isogeny_neighbours(curve: Dict[str, Any], limit: int = 120) -> List[Dict[str, Any]]:
    _, buckets = tau_index()
    tau = complex(float(curve['tau_re']), float(curve['tau_im']))
    found: Dict[int, Dict[str, Any]] = {}
    for a, b, c, d, h, det in relation_matrices():
        base_mat = (a, b, c, d)
        raw_target = _apply_matrix(tau, base_mat)
        if raw_target is None or raw_target.imag <= 0:
            continue
        reduced_target, red_mat, reduced = _reduce_tau(raw_target)
        if reduced_target.imag <= 0:
            continue
        combined = _mat_mul(red_mat, base_mat) if reduced else base_mat
        ca, cb, cc, cd = combined
        eps = _tau_match_eps(combined, tau)
        inv_det = ca * cd - cb * cc
        if inv_det <= 0:
            continue
        for cand in _bucket_candidates(buckets, reduced_target, eps):
            if cand['id'] == curve['id']:
                continue
            cand_tau = complex(float(cand['tau_re']), float(cand['tau_im']))
            forward_err = abs(cand_tau - reduced_target)
            if forward_err > eps:
                continue
            # Inverse residual validates that the same displayed tau' maps back to tau.
            back = _apply_matrix(cand_tau, (cd, -cb, -cc, ca))
            if back is None:
                continue
            inverse_err = abs(back - tau)
            inv_eps = max(eps, _tau_match_eps((cd, -cb, -cc, ca), cand_tau))
            if inverse_err > inv_eps:
                continue
            old = found.get(cand['id'])
            score = (h, det, forward_err + inverse_err)
            if old and old.get('_score', (999, 999, 999.0)) <= score:
                continue
            item = compact_member(cand)
            item['same_tau'] = abs(cand_tau - tau) <= max(TAU_MIN_MATCH_EPS, eps)
            item['same_q_isogeny_class'] = (cand['iso'] == curve['iso'])
            item['relation'] = _relation_string(base_mat, reduced)
            item['height'] = h
            item['determinant'] = det
            item['error'] = round(forward_err, 10)
            item['inverse_error'] = round(inverse_err, 10)
            item['tau_match_eps'] = round(eps, 10)
            item['verified_tau_match'] = True
            item['_score'] = score
            found[cand['id']] = item
    out = list(found.values())
    for item in out:
        item.pop('_score', None)
    out.sort(key=lambda z: (z['N'], z['height'], z['determinant'], z['label']))
    return out[:limit]


def get_connection():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    return con


@lru_cache(maxsize=1)
def metadata() -> Dict[str, str]:
    con = get_connection()
    data = {r['key']: r['value'] for r in con.execute('select key,value from metadata')}
    con.close()
    data['version'] = 'v16'
    return data


@lru_cache(maxsize=1)
def plot_meta() -> Dict[str, Any]:
    data = json.loads(META_PATH.read_text(encoding='utf-8'))
    data['version'] = 'v16'
    return data


@lru_cache(maxsize=1)
def sato_tate_groups() -> Dict[str, Any]:
    raw = metadata().get('sato_tate_groups', '{}')
    return json.loads(raw)


def plot_stub(row: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    r = enrich_curve(dict(row))
    return {
        'i': r['id'], 'l': r['label'], 'iso': r['iso'], 'N': r['N'], 'r': r['rank'], 't': r['tor_label'],
        'to': r['tor_order'], 'cm': 1 if r['cm'] else 0, 'az': r['az'], 'al': r['alt'],
        'lp': r['largest_prime'], 'lon': (float(r['az']) * 180.0 / 3.141592653589793) % 360,
        'alt': float(r['alt']) * 180.0 / 3.141592653589793,
        'ci': r.get('class_index', 0), 'cs': r.get('class_size', 1),
    }


def search_curves(q: str, limit: int = 15) -> List[Dict[str, Any]]:
    q = _normalize_query_text(q)
    if not q:
        return []

    cubic = parse_general_cubic_query(q)
    if cubic is not None:
        coeffs = cubic.get('coeffs')
        if coeffs is not None:
            j_key = j_key_from_coeffs(coeffs)
            if j_key is None:
                return []
            return _search_by_j_key(j_key, limit, 'same j-invariant fallback', prefer_qiso=coeffs, qiso_match=cubic.get('match', 'Q-isomorphic Weierstrass model'))
        if cubic.get('j') is not None:
            return _search_by_j_key(_frac_key(cubic['j']), limit, cubic.get('match', 'same j-invariant from cubic input'))

    out: List[Dict[str, Any]] = []
    seen = set()

    rational = parse_fraction_literal(q)
    if rational is not None:
        idx = search_index()
        key = _frac_key(rational)
        for rid in idx['disc'].get(key, []):
            _add_unique(out, seen, idx['rows'][rid], 'exact discriminant', limit)
        for rid in idx['j'].get(key, []):
            _add_unique(out, seen, idx['rows'][rid], 'exact j-invariant', limit)
        if out:
            return out[:limit]

    like = f'%{q}%'
    con = get_connection()
    rows = [dict(r) for r in con.execute(f'''
      select {SEARCH_COLUMNS}
      from curves
      where label like ? or cremona like ? or iso like ?
      order by N,label
      limit ?
    ''', (like, like, like, limit))]
    con.close()
    for row in rows:
        _add_unique(out, seen, row, 'label / Cremona / isogeny class', limit)
    return out[:limit]


@lru_cache(maxsize=4096)
def get_hover_curve(curve_id: int) -> Optional[Dict[str, Any]]:
    con = get_connection()
    row = con.execute(f'select {SEARCH_COLUMNS} from curves where id=?', (curve_id,)).fetchone()
    con.close()
    if not row:
        return None
    d = enrich_curve(dict(row))
    group = f"E(Q) ≅ Z^{d['rank']}" + ('' if d['tor_label'] == '0' else f" ⊕ {d['tor_label']}")
    return {
        'id': d['id'],
        'label': d['label'],
        'group': group,
        'weierstrass_equation': d['weierstrass_equation'],
        'disc': d['disc'],
        'j_str': d['j_str'],
    }


def get_system(iso: str) -> Optional[Dict[str, Any]]:
    con = get_connection()
    members = [compact_member(r) for r in con.execute('''
      select id,label,cremona,iso,N,rank,tor_label,tor_order,cm,st,class_index,class_size,az,alt,a1,a2,a3,a4,a6,real_period
      from curves where iso=? order by class_index,label
    ''', (iso,))]
    con.close()
    if not members:
        return None
    N = members[0]['N']
    return {
        'iso': iso,
        'N': N,
        'count': len(members),
        'rank_max': max(m['rank'] for m in members),
        'cm_count': sum(1 for m in members if m['cm']),
        'members': members,
    }


def compact_member(r: sqlite3.Row | Dict[str, Any]) -> Dict[str, Any]:
    d = enrich_curve(dict(r))
    d['lon_deg'] = (float(d['az']) * 180.0 / 3.141592653589793) % 360
    d['alt_deg'] = float(d['alt']) * 180.0 / 3.141592653589793
    return d


@lru_cache(maxsize=256)
def get_curve(curve_id: int, q_bound: int = 30) -> Optional[Dict[str, Any]]:
    con = get_connection()
    row = con.execute('select * from curves where id=?', (curve_id,)).fetchone()
    if not row:
        con.close()
        return None
    data = enrich_curve(dict(row))
    members = [compact_member(r) for r in con.execute('''
      select id,label,cremona,iso,N,rank,tor_label,tor_order,cm,class_index,class_size,az,alt,a1,a2,a3,a4,a6,real_period
      from curves where iso=? order by class_index,label
    ''', (data['iso'],))]
    same_count = con.execute('select count(*) from curves where N=?', (data['N'],)).fetchone()[0]
    same_conductor = [compact_member(r) for r in con.execute('''
      select id,label,cremona,iso,N,rank,tor_label,tor_order,cm,class_index,class_size,az,alt,a1,a2,a3,a4,a6,real_period
      from curves where N=? order by iso,class_index,label limit 240
    ''', (data['N'],))]
    con.close()

    a1, a2, a3, a4, a6 = data['a1'], data['a2'], data['a3'], data['a4'], data['a6']
    inv = invariants(a1, a2, a3, a4, a6)
    red = reduction_data(a1, a2, a3, a4, a6, 100)
    coeffs = newform_coefficients(a1, a2, a3, a4, a6, q_bound)
    st_groups = sato_tate_groups()
    data['sato_tate'] = st_groups.get(data['st'], {'label': data['st']})
    data['members'] = members
    data['same_conductor'] = same_conductor
    data['same_conductor_count'] = same_count
    data['lon_deg'] = (float(data['az']) * 180.0 / 3.141592653589793) % 360
    data['alt_deg'] = float(data['alt']) * 180.0 / 3.141592653589793
    data['c_isogeny'] = detected_c_isogeny_neighbours(data)
    data['invariants'] = inv
    data['reduction_table'] = red
    data['q_coefficients'] = [{'n': n, 'a_n': coeffs[n]} for n in range(1, q_bound + 1)]
    data['q_expansion'] = format_q_expansion(coeffs, q_bound)
    return data

# ---------------------------------------------------------------------------
# Integral and S-integral point search utilities
# ---------------------------------------------------------------------------
import time
from math import isqrt


def _curve_by_id(curve_id: int) -> Optional[Dict[str, Any]]:
    con = get_connection()
    row = con.execute('select * from curves where id=?', (curve_id,)).fetchone()
    con.close()
    return dict(row) if row else None


def _square_root_int(n: int) -> Optional[int]:
    if n < 0:
        return None
    r = isqrt(n)
    return r if r * r == n else None


def _point_y_from_Y(a1: int, a3: int, x_num: int, d: int, Y_num: int) -> Optional[Tuple[int, int]]:
    # x = x_num / d^2, Y = Y_num / d^3, Y = 2y + a1*x + a3
    den = 2 * d**3
    num = Y_num - a1 * x_num * d - a3 * d**3
    if num % den != 0:
        # Return as rational numerator/denominator anyway for S-integral use.
        g = math.gcd(abs(num), den)
        return (num // g, den // g)
    return (num // den, 1)


def _format_q(num: int, den: int = 1) -> str:
    return str(num) if den == 1 else f'{num}/{den}'


def _smooth_numbers_from_primes(primes: List[int], limit: int) -> List[int]:
    vals = {1}
    for p in primes:
        current = sorted(vals)
        mul = p
        while mul <= limit:
            for v in current:
                nv = v * mul
                if nv <= limit:
                    vals.add(nv)
            mul *= p
    return sorted(vals)


def _prime_factors(n: int) -> List[int]:
    n = abs(int(n))
    if n <= 1:
        return []
    out = []
    d = 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d = 3 if d == 2 else d + 2
    if n > 1:
        out.append(n)
    return out


def search_integral_points(curve_id: int, timeout: float = 1.0) -> Dict[str, Any]:
    curve = _curve_by_id(curve_id)
    if not curve:
        return {'error': 'curve not found'}
    start = time.perf_counter()
    a1, a2, a3, a4, a6 = curve['a1'], curve['a2'], curve['a3'], curve['a4'], curve['a6']
    inv = invariants(a1, a2, a3, a4, a6)
    b2, b4, b6 = inv['b2'], inv['b4'], inv['b6']
    target = int(curve.get('pts') or 0)
    if target == 0:
        return {
            'curve_id': curve_id, 'label': curve['label'], 'mode': 'integral',
            'target_count': 0, 'points': [], 'count': 0, 'searched_abs_x_up_to': 0,
            'complete': True, 'timed_out': False,
            'elapsed_ms': round((time.perf_counter() - start) * 1000, 1),
            'note': 'stored integral-point count is zero.'
        }
    found = []
    seen = set()

    # Expand symmetrically. If the stored LMFDB count is reached, mark complete.
    bound = 256
    max_bound = 2_000_000
    searched = 0
    complete = False
    timed_out = False
    while bound <= max_bound:
        lo, hi = -bound, bound
        for x in range(lo, hi + 1):
            if abs(x) <= searched:
                continue
            rhs = 4*x*x*x + b2*x*x + 2*b4*x + b6
            yroot = _square_root_int(rhs)
            if yroot is None:
                continue
            for Y in ({0} if yroot == 0 else {yroot, -yroot}):
                yn = Y - a1*x - a3
                if yn % 2:
                    continue
                y = yn // 2
                key = (x, y)
                if key not in seen:
                    seen.add(key)
                    found.append({'x': str(x), 'y': str(y)})
        searched = bound
        if target and len(found) >= target:
            complete = True
            break
        if time.perf_counter() - start > timeout:
            timed_out = True
            break
        bound *= 2
    found.sort(key=lambda z: (len(z['x']), z['x'], z['y']))
    return {
        'curve_id': curve_id,
        'label': curve['label'],
        'mode': 'integral',
        'target_count': target,
        'points': found,
        'count': len(found),
        'searched_abs_x_up_to': searched,
        'complete': bool(complete or (target == 0 and not found and searched >= max_bound)),
        'timed_out': timed_out,
        'elapsed_ms': round((time.perf_counter() - start) * 1000, 1),
        'note': 'complete=true means the stored integral-point count was reached; otherwise the list is a bounded-time search result.'
    }


def search_s_integral_points(curve_id: int, S: int, timeout: float = 5.0) -> Dict[str, Any]:
    curve = _curve_by_id(curve_id)
    if not curve:
        return {'error': 'curve not found'}
    start = time.perf_counter()
    primes = _prime_factors(S)
    if not primes:
        return search_integral_points(curve_id, min(timeout, 1.0))
    # Keep this responsive. These bounds are intentionally conservative.
    max_d = 80 if len(primes) <= 2 else 50
    x_bound = 1200
    denominators = _smooth_numbers_from_primes(primes, max_d)

    a1, a2, a3, a4, a6 = curve['a1'], curve['a2'], curve['a3'], curve['a4'], curve['a6']
    inv = invariants(a1, a2, a3, a4, a6)
    b2, b4, b6 = inv['b2'], inv['b4'], inv['b6']
    found = []
    seen = set()
    timed_out = False
    checked = 0
    for d in denominators:
        d2, d4, d6 = d*d, d**4, d**6
        m_bound = x_bound * d2
        for m in range(-m_bound, m_bound + 1):
            checked += 1
            rhs_num = 4*m*m*m + b2*m*m*d2 + 2*b4*m*d4 + b6*d6
            Yroot = _square_root_int(rhs_num)
            if Yroot is not None:
                for Y in ({0} if Yroot == 0 else {Yroot, -Yroot}):
                    y_num, y_den = _point_y_from_Y(a1, a3, m, d, Y)
                    x_num, x_den = m, d2
                    gx = math.gcd(abs(x_num), x_den)
                    x_num //= gx; x_den //= gx
                    key = (x_num, x_den, y_num, y_den)
                    if key not in seen:
                        seen.add(key)
                        found.append({'x': _format_q(x_num, x_den), 'y': _format_q(y_num, y_den), 'den_x': x_den, 'den_y': y_den})
            if checked % 4096 == 0 and time.perf_counter() - start > timeout:
                timed_out = True
                break
        if timed_out:
            break
    found.sort(key=lambda z: (max(z.get('den_x', 1), z.get('den_y', 1)), len(z['x']), z['x'], z['y']))
    return {
        'curve_id': curve_id,
        'label': curve['label'],
        'mode': 'S-integral',
        'S': S,
        'S_primes': primes,
        'points': found[:500],
        'count': len(found),
        'returned_count': min(len(found), 500),
        'denominators_checked': len(denominators),
        'max_denominator_checked': denominators[-1] if denominators else 1,
        'x_height_bound': x_bound,
        'timed_out': timed_out,
        'complete': False,
        'elapsed_ms': round((time.perf_counter() - start) * 1000, 1),
        'note': 'S-integral search is bounded by time and height; timed_out=true or complete=false means additional points may be missing.'
    }

# ---------------------------------------------------------------------------
# Browser-facing clean API
# ---------------------------------------------------------------------------

def _frac_to_string(value: Fraction | int | str) -> str:
    value = Fraction(value)
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _coeffs_to_json(coeffs: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]) -> List[str]:
    return [_frac_to_string(c) for c in coeffs]


def _coeffs_to_int_json(coeffs: Tuple[Fraction, Fraction, Fraction, Fraction, Fraction]) -> Optional[List[int]]:
    if all(Fraction(c).denominator == 1 for c in coeffs):
        return [int(Fraction(c)) for c in coeffs]
    return None


def weierstrass_equation_fraction(a1: Fraction, a2: Fraction, a3: Fraction, a4: Fraction, a6: Fraction) -> str:
    def term(coef: Fraction, body: str) -> str:
        coef = Fraction(coef)
        if coef == 0:
            return ''
        sign = '-' if coef < 0 else '+'
        mag = abs(coef)
        if mag == 1 and body:
            text = body
        else:
            text = f"{_frac_to_string(mag)}{body}"
        return f" {sign} {text}"
    lhs = 'y²' + term(a1, 'xy') + term(a3, 'y')
    rhs = 'x³' + term(a2, 'x²') + term(a4, 'x') + term(a6, '')
    return f"{lhs} = {rhs}"


def _safe_symbols(text: str) -> List[str]:
    try:
        poly_obj = _poly_from_input(text)
        if poly_obj is None:
            return []
        return list(poly_obj[1])
    except Exception:
        return []


def identify_cubic(text: str) -> Dict[str, Any]:
    """Recognize a typed cubic and return a JSON-serializable exact result.

    The function is designed for Pyodide/Web Worker use. It performs only
    standard-library exact rational arithmetic and does not read external files.
    """
    raw = _normalize_query_text(text)
    result: Dict[str, Any] = {
        'ok': False,
        'input': raw,
        'version': 'v17',
        'notes': [],
    }
    if not raw:
        result['error'] = 'Empty input.'
        return result
    try:
        parsed = parse_general_cubic_query(raw)
    except Exception as exc:
        result['error'] = f'Parser/recognizer error: {exc}'
        return result
    if parsed is None:
        result['error'] = 'Input was not recognized as a supported two-variable cubic, Weierstrass equation, or coefficient list.'
        return result

    result['ok'] = True
    result['method'] = parsed.get('match', 'recognized cubic')
    result['symbols'] = _safe_symbols(raw)

    coeffs = parsed.get('coeffs')
    if coeffs is not None:
        coeffs = tuple(Fraction(c) for c in coeffs)
        inv = invariants_fraction(*coeffs)
        j = j_from_invariants_fraction(inv)
        if j is None:
            result['ok'] = False
            result['error'] = 'The recognized model has zero discriminant; j-invariant is undefined.'
            return result
        result.update({
            'mode': 'minimal_model',
            'j': _frac_to_string(j),
            'coeffs': _coeffs_to_json(coeffs),
            'coeffs_int': _coeffs_to_int_json(coeffs),
            'weierstrass': weierstrass_equation_fraction(*coeffs),
            'discriminant': _frac_to_string(inv['disc']),
            'c4': _frac_to_string(inv['c4']),
            'c6': _frac_to_string(inv['c6']),
        })
        return result

    j = parsed.get('j')
    if j is not None:
        result.update({
            'mode': 'j_only',
            'j': _frac_to_string(Fraction(j)),
            'coeffs': None,
            'coeffs_int': None,
            'weierstrass': None,
        })
        result['notes'].append('A smooth ternary-cubic j-invariant was computed, but no Q-minimal model was certified by the current rational-point transformation path.')
        return result

    result['ok'] = False
    result['error'] = 'Recognition reached no usable j-invariant.'
    return result


def identify_cubic_json(text: str) -> str:
    return json.dumps(identify_cubic(text), ensure_ascii=False, separators=(',', ':'))


if __name__ == '__main__':
    examples = [
        '[3,3]',
        '[0,0,1,3,3]',
        'y^2 + y = x^3 + 3*x + 3',
        'x^3 + x^2*y + y^3 + y^2 - 2*x + 1 = 0',
        'u^3 + u^2*v + v^3 + v^2 - 2*u + 1 = 0',
        'X^3 + 2Y^3 + 1 = 15XY',
    ]
    for ex in examples:
        print(ex)
        print(json.dumps(identify_cubic(ex), ensure_ascii=False, indent=2))
