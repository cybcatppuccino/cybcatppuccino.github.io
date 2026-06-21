
// EC Recognizer v18 JS BigInt core.
// Exact rational arithmetic with BigInt. No external libraries.

function absBig(n) { return n < 0n ? -n : n; }
function gcdBig(a, b) {
  a = absBig(BigInt(a)); b = absBig(BigInt(b));
  while (b !== 0n) { const t = a % b; a = b; b = t; }
  return a;
}
function lcmBig(a, b) {
  a = BigInt(a); b = BigInt(b);
  if (a === 0n || b === 0n) return absBig(a || b);
  return absBig((a / gcdBig(a, b)) * b);
}
function modBig(a, m) {
  a = BigInt(a); m = BigInt(m);
  let r = a % m;
  if (r < 0n) r += absBig(m);
  return r;
}
function centeredModBig(n, m) {
  m = BigInt(m);
  let r = modBig(n, m);
  if (r > m / 2n) r -= m;
  return r;
}
function powBig(a, e) {
  a = BigInt(a); e = BigInt(e);
  if (e < 0n) throw new Error('negative integer power');
  let out = 1n, base = a;
  while (e) { if (e & 1n) out *= base; e >>= 1n; if (e) base *= base; }
  return out;
}
function comb(n, r) {
  if (r < 0 || r > n) return 0n;
  r = Math.min(r, n - r);
  let out = 1n;
  for (let k = 1; k <= r; k++) out = (out * BigInt(n - r + k)) / BigInt(k);
  return out;
}

export class Rat {
  constructor(num, den = 1n) {
    if (num instanceof Rat) { this.num = num.num; this.den = num.den; return; }
    let n = typeof num === 'bigint' ? num : BigInt(num);
    let d = typeof den === 'bigint' ? den : BigInt(den);
    if (d === 0n) throw new Error('zero denominator');
    if (d < 0n) { n = -n; d = -d; }
    const g = gcdBig(n, d);
    this.num = n / g; this.den = d / g;
  }
  static from(v) { return v instanceof Rat ? v : new Rat(v); }
  static parse(s) {
    s = String(s).trim();
    const m = s.match(/^([+-]?\d+)(?:\/([+-]?\d+))?$/);
    if (!m) throw new Error('bad rational literal: ' + s);
    return new Rat(BigInt(m[1]), m[2] == null ? 1n : BigInt(m[2]));
  }
  isZero() { return this.num === 0n; }
  isOne() { return this.num === this.den; }
  neg() { return new Rat(-this.num, this.den); }
  abs() { return new Rat(absBig(this.num), this.den); }
  add(b) { b = Rat.from(b); return new Rat(this.num * b.den + b.num * this.den, this.den * b.den); }
  sub(b) { b = Rat.from(b); return new Rat(this.num * b.den - b.num * this.den, this.den * b.den); }
  mul(b) { b = Rat.from(b); return new Rat(this.num * b.num, this.den * b.den); }
  div(b) { b = Rat.from(b); if (b.num === 0n) throw new Error('division by zero'); return new Rat(this.num * b.den, this.den * b.num); }
  pow(e) {
    e = Number(e);
    if (!Number.isInteger(e)) throw new Error('non-integer exponent');
    if (e < 0) return new Rat(powBig(this.den, -e), powBig(this.num, -e));
    return new Rat(powBig(this.num, e), powBig(this.den, e));
  }
  eq(b) { b = Rat.from(b); return this.num === b.num && this.den === b.den; }
  toString() { return this.den === 1n ? String(this.num) : `${this.num}/${this.den}`; }
  toJSON() { return this.toString(); }
  toNumber() { return Number(this.num) / Number(this.den); }
}
function R(n, d = 1n) { return new Rat(BigInt(n), BigInt(d)); }
const ZERO = R(0), ONE = R(1);

function normalizeQueryText(text) {
  return String(text || '').trim().replace(/[−—–]/g, '-');
}
function normalizeMathInput(text) {
  return normalizeQueryText(text)
    .replace(/²/g, '^2').replace(/³/g, '^3').replace(/⁴/g, '^4').replace(/⁵/g, '^5').replace(/⁶/g, '^6')
    .replace(/＋/g, '+').replace(/[－−—–]/g, '-').replace(/[＊×·]/g, '*').replace(/／/g, '/')
    .replace(/（/g, '(').replace(/）/g, ')').replace(/＝/g, '=');
}
function parseFractionLiteral(text) {
  const raw = normalizeQueryText(text).replace(/\s+/g, '');
  if (!/^[+-]?\d+(?:\/[+-]?\d+)?$/.test(raw)) return null;
  try { return Rat.parse(raw); } catch { return null; }
}
function parseCoefficientList(text) {
  const m = normalizeQueryText(text).match(/^\[\s*(.*?)\s*\]$/);
  if (!m) return null;
  const parts = m[1].split(',').map(x => x.trim());
  if (![2, 5].includes(parts.length)) return null;
  const vals = parts.map(parseFractionLiteral);
  if (vals.some(v => v == null)) return null;
  if (vals.length === 2) return [ZERO, ZERO, ZERO, vals[0], vals[1]];
  return vals;
}
function fractionToExpr(fr) { fr = Rat.from(fr); return fr.den === 1n ? String(fr.num) : `(${fr.num}/${fr.den})`; }
function coeffsToWeierstrassAffineExpr(coeffs) {
  const [a1,a2,a3,a4,a6] = coeffs;
  return `y^2 + (${fractionToExpr(a1)})*x*y + (${fractionToExpr(a3)})*y - x^3 - (${fractionToExpr(a2)})*x^2 - (${fractionToExpr(a4)})*x - (${fractionToExpr(a6)})`;
}

function key(m) { return m.join(','); }
function unkey(k) { return k.split(',').map(Number); }
function polyClean(poly) {
  const out = new Map();
  for (const [k, v] of poly) { const r = Rat.from(v); if (!r.isZero()) out.set(k, r); }
  return out;
}
function polyConst(value, nvars) {
  value = Rat.from(value); const out = new Map(); if (!value.isZero()) out.set(key(Array(nvars).fill(0)), value); return out;
}
function polyVar(index, nvars) {
  const e = Array(nvars).fill(0); e[index] = 1; const out = new Map(); out.set(key(e), ONE); return out;
}
function polyAdd(a, b) {
  const out = new Map(a);
  for (const [k, c] of b) out.set(k, (out.get(k) || ZERO).add(c));
  return polyClean(out);
}
function polyNeg(a) { const out = new Map(); for (const [k, c] of a) out.set(k, c.neg()); return out; }
function polySub(a, b) { return polyAdd(a, polyNeg(b)); }
function polyScale(a, s) { s = Rat.from(s); const out = new Map(); if (s.isZero()) return out; for (const [k, c] of a) out.set(k, c.mul(s)); return polyClean(out); }
function polyMul(a, b, maxDegree = 12) {
  const out = new Map();
  for (const [ka, ca] of a) for (const [kb, cb] of b) {
    const ma = unkey(ka), mb = unkey(kb); const m = ma.map((x, i) => x + mb[i]);
    if (m.reduce((p, q) => p + q, 0) > maxDegree) throw new Error('degree too large');
    const kk = key(m); out.set(kk, (out.get(kk) || ZERO).add(ca.mul(cb)));
  }
  return polyClean(out);
}
function polyPow(a, exponent, maxDegree = 12) {
  exponent = Number(exponent); if (exponent < 0) throw new Error('negative polynomial power');
  const nvars = a.size ? unkey(a.keys().next().value).length : 1;
  let out = polyConst(ONE, nvars), base = new Map(a), e = exponent;
  while (e) { if (e & 1) out = polyMul(out, base, maxDegree); e >>= 1; if (e) base = polyMul(base, base, maxDegree); }
  return out;
}
function polyDegree(poly) { let d = 0; for (const k of poly.keys()) d = Math.max(d, unkey(k).reduce((a,b)=>a+b,0)); return d; }
function polyCoeff(poly, monom) { return poly.get(key(monom)) || ZERO; }

function candidateSymbols(raw) {
  const names = [];
  for (const ch of raw.match(/[A-Za-z]/g) || []) {
    if (ch.toLowerCase() === 'e') continue;
    if (!names.includes(ch)) names.push(ch);
  }
  const lowered = new Map(names.map(n => [n.toLowerCase(), n]));
  const ordered = [];
  for (const preferred of ['x', 'y']) if (lowered.has(preferred) && !ordered.includes(lowered.get(preferred))) ordered.push(lowered.get(preferred));
  for (const n of names) if (!ordered.includes(n)) ordered.push(n);
  return ordered.slice(0, 6);
}
function tokenizeMath(raw) {
  const out = []; let pos = 0;
  const re = /\s*(?:(\d+(?:\/\d+)?)|([A-Za-z])|([+\-*\/^()]))/y;
  while (pos < raw.length) {
    if (/\s/.test(raw[pos])) { pos++; continue; }
    re.lastIndex = pos; const m = re.exec(raw);
    if (!m) return null;
    if (m[1] != null) out.push(['num', m[1]]);
    else if (m[2] != null) out.push(['var', m[2]]);
    else out.push(['op', m[3]]);
    pos = re.lastIndex;
  }
  out.push(['eof', '']); return out;
}
class MathParser {
  constructor(raw, names) {
    const toks = tokenizeMath(raw); if (!toks) throw new Error('bad token');
    this.toks = toks; this.i = 0; this.names = names; this.nvars = names.length; this.index = new Map(names.map((n, j) => [n, j]));
  }
  peek() { return this.toks[this.i]; }
  take() { return this.toks[this.i++]; }
  parse() { const p = this.expr(); if (this.peek()[0] !== 'eof') throw new Error('trailing tokens'); return p; }
  expr() { let out = this.term(); while (this.peek()[0] === 'op' && ['+','-'].includes(this.peek()[1])) { const op = this.take()[1]; const rhs = this.term(); out = op === '+' ? polyAdd(out, rhs) : polySub(out, rhs); } return out; }
  startsFactor(tok) { return tok[0] === 'num' || tok[0] === 'var' || (tok[0] === 'op' && tok[1] === '('); }
  term() {
    let out = this.power();
    while (true) {
      const tok = this.peek();
      if (tok[0] === 'op' && tok[1] === '*') { this.take(); out = polyMul(out, this.power(), 12); }
      else if (tok[0] === 'op' && tok[1] === '/') {
        this.take(); const rhs = this.power(); if (rhs.size !== 1) throw new Error('division by non-constant');
        const coeff = [...rhs.values()][0]; if (coeff.isZero()) throw new Error('zero division'); out = polyScale(out, ONE.div(coeff));
      } else if (this.startsFactor(tok)) out = polyMul(out, this.power(), 12);
      else break;
    }
    return out;
  }
  power() {
    let base = this.factor();
    if (this.peek()[0] === 'op' && this.peek()[1] === '^') {
      this.take(); let sign = 1;
      if (this.peek()[0] === 'op' && this.peek()[1] === '+') this.take();
      else if (this.peek()[0] === 'op' && this.peek()[1] === '-') { this.take(); sign = -1; }
      const tok = this.take(); if (tok[0] !== 'num' || tok[1].includes('/')) throw new Error('non-integer exponent');
      base = polyPow(base, sign * Number(tok[1]), 12);
    }
    return base;
  }
  factor() {
    const tok = this.take(), typ = tok[0], val = tok[1];
    if (typ === 'num') return polyConst(Rat.parse(val), this.nvars);
    if (typ === 'var') { if (!this.index.has(val)) throw new Error('unknown variable'); return polyVar(this.index.get(val), this.nvars); }
    if (typ === 'op' && val === '+') return this.factor();
    if (typ === 'op' && val === '-') return polyNeg(this.factor());
    if (typ === 'op' && val === '(') { const out = this.expr(); const close = this.take(); if (close[0] !== 'op' || close[1] !== ')') throw new Error('missing right parenthesis'); return out; }
    throw new Error('bad factor');
  }
}
function parsePolyExpr(raw, names) { try { return new MathParser(raw, names).parse(); } catch { return null; } }
function polyFromInput(text) {
  const raw = normalizeMathInput(text); if (!raw) return null;
  const names = candidateSymbols(raw); if (names.length < 2) return null;
  let poly;
  if (raw.includes('=')) {
    const [left, ...rest] = raw.split('='); const right = rest.join('=');
    const lp = parsePolyExpr(left, names), rp = parsePolyExpr(right, names); if (!lp || !rp) return null;
    poly = polySub(lp, rp);
  } else {
    poly = parsePolyExpr(raw, names); if (!poly) return null;
  }
  const used = names.filter((name, i) => [...poly.keys()].some(k => unkey(k)[i]));
  if (used.length !== 2) return null;
  const idx = used.map(n => names.indexOf(n)); const repacked = new Map();
  for (const [k, c] of poly) { const m = unkey(k); const nm = [m[idx[0]], m[idx[1]]]; const kk = key(nm); repacked.set(kk, (repacked.get(kk) || ZERO).add(c)); }
  const clean = polyClean(repacked); const deg = polyDegree(clean); if (deg > 3 || deg < 2) return null;
  return { poly: clean, symbols: used };
}

function invariantsFraction(a1, a2, a3, a4, a6) {
  const b2 = a1.mul(a1).add(R(4).mul(a2));
  const b4 = R(2).mul(a4).add(a1.mul(a3));
  const b6 = a3.mul(a3).add(R(4).mul(a6));
  const b8 = a1.mul(a1).mul(a6).add(R(4).mul(a2).mul(a6)).sub(a1.mul(a3).mul(a4)).add(a2.mul(a3).mul(a3)).sub(a4.mul(a4));
  const c4 = b2.mul(b2).sub(R(24).mul(b4));
  const c6 = b2.mul(b2).mul(b2).neg().add(R(36).mul(b2).mul(b4)).sub(R(216).mul(b6));
  const disc = b2.mul(b2).mul(b8).neg().sub(R(8).mul(b4).mul(b4).mul(b4)).sub(R(27).mul(b6).mul(b6)).add(R(9).mul(b2).mul(b4).mul(b6));
  return { b2, b4, b6, b8, c4, c6, disc };
}
function jFromInvariants(inv) { if (inv.disc.isZero()) return null; return inv.c4.pow(3).div(inv.disc); }
function denominatorLcm(values) { let d = 1n; for (const v of values) d = lcmBig(d, Rat.from(v).den); return d === 0n ? 1n : d; }
function vP(n, p) { n = absBig(n); p = BigInt(p); if (n === 0n) return 0; let e = 0; while (n % p === 0n) { n /= p; e++; } return e; }
function primeDivisors(n) {
  n = absBig(n); const out = [];
  if (n < 2n) return out;
  if (n % 2n === 0n) { out.push(2n); while (n % 2n === 0n) n /= 2n; }
  let p = 3n;
  while (p * p <= n && p <= 1000000n) {
    if (n % p === 0n) { out.push(p); while (n % p === 0n) n /= p; }
    p += 2n;
  }
  if (n > 1n) out.push(n);
  return out;
}
function reducedModelFromIntegralInvariants(c4, c6) {
  c4 = BigInt(c4); c6 = BigInt(c6);
  const deltaNum = c4 ** 3n - c6 ** 2n;
  if (deltaNum === 0n || deltaNum % 1728n !== 0n) return null;
  const delta = deltaNum / 1728n;
  const g = gcdBig(c6 * c6, delta);
  let u = 1n;
  for (const p of primeDivisors(g)) {
    let d = Math.floor(vP(g, p) / 12);
    if (d <= 0) continue;
    if (p === 2n) {
      const p4 = p ** BigInt(4 * d), p6 = p ** BigInt(6 * d);
      if (c4 % p4 !== 0n || c6 % p6 !== 0n) d -= 1;
      else {
        const aa = modBig(c4 / p4, 16n), bb = modBig(c6 / p6, 32n);
        if ((bb % 4n !== 3n) && !(aa === 0n && (bb === 0n || bb === 8n))) d -= 1;
      }
    } else if (p === 3n) {
      if (vP(c6, 3n) === 6 * d + 2) d -= 1;
    }
    if (d > 0) u *= p ** BigInt(d);
  }
  if (c4 % (u ** 4n) !== 0n || c6 % (u ** 6n) !== 0n) return null;
  const c4m = c4 / (u ** 4n), c6m = c6 / (u ** 6n);
  const chk = c4m ** 3n - c6m ** 2n;
  if (chk === 0n || chk % 1728n !== 0n) return null;
  const b2 = centeredModBig(-c6m, 12n);
  const b4Num = b2 * b2 - c4m; if (b4Num % 24n !== 0n) return null;
  const b4 = b4Num / 24n;
  const b6Num = -(b2 ** 3n) + 36n * b2 * b4 - c6m; if (b6Num % 216n !== 0n) return null;
  const b6 = b6Num / 216n;
  const a1 = modBig(b2, 2n), a3 = modBig(b6, 2n);
  if ((b2 - a1) % 4n !== 0n || (b4 - a1 * a3) % 2n !== 0n || (b6 - a3) % 4n !== 0n) return null;
  const a2 = (b2 - a1) / 4n, a4 = (b4 - a1 * a3) / 2n, a6 = (b6 - a3) / 4n;
  return [a1,a2,a3,a4,a6].map(x => new Rat(x));
}
function qMinimalModel(coeffs) {
  coeffs = coeffs.map(Rat.from);
  const inv0 = invariantsFraction(...coeffs); if (inv0.disc.isZero()) return coeffs;
  const D = denominatorLcm(coeffs); const weights = [1,2,3,4,6];
  const scaled = coeffs.map((c, i) => c.mul(new Rat(D ** BigInt(weights[i]))));
  if (scaled.every(v => v.den === 1n)) {
    const inv = invariantsFraction(...scaled);
    if (inv.c4.den === 1n && inv.c6.den === 1n) {
      const red = reducedModelFromIntegralInvariants(inv.c4.num, inv.c6.num);
      if (red) return red;
    }
  }
  return coeffs;
}
function standardWeierstrassFromPoly(polyObj) {
  try {
    const source = polyObj.poly;
    for (const swap of [false, true]) {
      const P = new Map();
      for (const [k, c] of source) { const [i,j] = unkey(k); P.set(key(swap ? [j,i] : [i,j]), c); }
      const allowed = new Set(['0,2','1,1','0,1','3,0','2,0','1,0','0,0']);
      if ([...P.entries()].some(([k,c]) => !allowed.has(k) && !c.isZero())) continue;
      const cy2 = polyCoeff(P, [0,2]), cx3 = polyCoeff(P, [3,0]);
      if (cy2.isZero() || cx3.isZero()) continue;
      const Pn = polyScale(P, ONE.div(cy2));
      if (!polyCoeff(Pn,[0,2]).eq(ONE) || !polyCoeff(Pn,[3,0]).eq(R(-1))) continue;
      const vals = [polyCoeff(Pn,[1,1]), polyCoeff(Pn,[2,0]).neg(), polyCoeff(Pn,[0,1]), polyCoeff(Pn,[1,0]).neg(), polyCoeff(Pn,[0,0]).neg()];
      return qMinimalModel(vals);
    }
  } catch {}
  return null;
}
function diagonalHesseJFromPoly(polyObj) {
  try {
    const poly = polyObj.poly;
    const allowed = new Set(['3,0','0,3','1,1','0,0']);
    if ([...poly.entries()].some(([k,c]) => !allowed.has(k) && !c.isZero())) return null;
    const A = polyCoeff(poly,[3,0]), B = polyCoeff(poly,[0,3]), C = polyCoeff(poly,[0,0]), E = polyCoeff(poly,[1,1]);
    if (A.isZero() || B.isZero() || C.isZero() || E.isZero()) return null;
    const D = E.neg();
    const k3 = D.pow(3).div(R(27).mul(A).mul(B).mul(C));
    if (k3.eq(ONE)) return null;
    return R(27).mul(k3).mul(k3.add(R(8)).pow(3)).div(k3.sub(ONE).pow(3));
  } catch { return null; }
}
function homogenizeAffineCubic(poly) {
  const out = new Map();
  for (const [kk, c] of poly) { const [i,j] = unkey(kk); const k = 3 - i - j; if (k < 0) throw new Error('not cubic'); const nk = key([i,j,k]); out.set(nk, (out.get(nk)||ZERO).add(c)); }
  return polyClean(out);
}
function ternaryCubicCoefficientsAronhold(poly) {
  return [
    polyCoeff(poly,[3,0,0]), polyCoeff(poly,[0,3,0]), polyCoeff(poly,[0,0,3]),
    polyCoeff(poly,[2,1,0]).div(R(3)), polyCoeff(poly,[0,2,1]).div(R(3)), polyCoeff(poly,[1,0,2]).div(R(3)),
    polyCoeff(poly,[1,2,0]).div(R(3)), polyCoeff(poly,[0,1,2]).div(R(3)), polyCoeff(poly,[2,0,1]).div(R(3)),
    polyCoeff(poly,[1,1,1]).div(R(6))
  ];
}

// GENERATED_ARONHOLD_START
function aronholdSTFromCoefficients(vals) {
  const [a,b,c,d,e,f,g,h,i,j] = vals;
  const S = (((((((((((((((((((((((((((a).mul(g)).mul(e)).mul(c)).sub(((a).mul(g)).mul((h).pow(2)))).sub((((a).mul(j)).mul(b)).mul(c))).add((((a).mul(j)).mul(e)).mul(h))).add((((a).mul(f)).mul(b)).mul(h))).sub(((a).mul(f)).mul((e).pow(2)))).sub((((d).pow(2)).mul(e)).mul(c))).add(((d).pow(2)).mul((h).pow(2)))).add((((d).mul(i)).mul(b)).mul(c))).sub((((d).mul(i)).mul(e)).mul(h))).add((((d).mul(g)).mul(j)).mul(c))).sub((((d).mul(g)).mul(f)).mul(h))).sub((((R(2)).mul(d)).mul((j).pow(2))).mul(h))).add(((((R(3)).mul(d)).mul(j)).mul(f)).mul(e))).sub(((d).mul((f).pow(2))).mul(b))).sub((((i).pow(2)).mul(b)).mul(h))).add(((i).pow(2)).mul((e).pow(2)))).sub(((i).mul((g).pow(2))).mul(c))).add(((((R(3)).mul(i)).mul(g)).mul(j)).mul(h))).sub((((i).mul(g)).mul(f)).mul(e))).sub((((R(2)).mul(i)).mul((j).pow(2))).mul(e))).add((((i).mul(j)).mul(f)).mul(b))).add(((g).pow(2)).mul((f).pow(2)))).sub((((R(2)).mul(g)).mul((j).pow(2))).mul(f))).add((j).pow(4));
  const T = (((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((((a).pow(2)).mul((b).pow(2))).mul((c).pow(2))).sub((((R(3)).mul((a).pow(2))).mul((e).pow(2))).mul((h).pow(2)))).sub((((((R(6)).mul((a).pow(2))).mul(b)).mul(e)).mul(h)).mul(c))).add((((R(4)).mul((a).pow(2))).mul(b)).mul((h).pow(3)))).add((((R(4)).mul((a).pow(2))).mul((e).pow(3))).mul(c))).sub((((((R(6)).mul(a)).mul(d)).mul(g)).mul(b)).mul((c).pow(2)))).add(((((((R(18)).mul(a)).mul(d)).mul(g)).mul(e)).mul(h)).mul(c))).sub(((((R(12)).mul(a)).mul(d)).mul(g)).mul((h).pow(3)))).add(((((((R(12)).mul(a)).mul(d)).mul(j)).mul(b)).mul(h)).mul(c))).sub((((((R(24)).mul(a)).mul(d)).mul(j)).mul((e).pow(2))).mul(c))).add((((((R(12)).mul(a)).mul(d)).mul(j)).mul(e)).mul((h).pow(2)))).sub((((((R(12)).mul(a)).mul(d)).mul(f)).mul(b)).mul((h).pow(2)))).add(((((((R(6)).mul(a)).mul(d)).mul(f)).mul(b)).mul(e)).mul(c))).add((((((R(6)).mul(a)).mul(d)).mul(f)).mul((e).pow(2))).mul(h))).add(((((((R(6)).mul(a)).mul(i)).mul(g)).mul(b)).mul(h)).mul(c))).sub((((((R(12)).mul(a)).mul(i)).mul(g)).mul((e).pow(2))).mul(c))).add((((((R(6)).mul(a)).mul(i)).mul(g)).mul(e)).mul((h).pow(2)))).add(((((((R(12)).mul(a)).mul(i)).mul(j)).mul(b)).mul(e)).mul(c))).add((((((R(12)).mul(a)).mul(i)).mul(j)).mul((e).pow(2))).mul(h))).sub((((((R(6)).mul(a)).mul(i)).mul(f)).mul((b).pow(2))).mul(c))).add(((((((R(18)).mul(a)).mul(i)).mul(f)).mul(b)).mul(e)).mul(h))).sub((((((R(24)).mul(a)).mul((g).pow(2))).mul(j)).mul(h)).mul(c))).sub((((((R(24)).mul(a)).mul(i)).mul(j)).mul(b)).mul((h).pow(2)))).sub(((((R(12)).mul(a)).mul(i)).mul(f)).mul((e).pow(3)))).add((((R(4)).mul(a)).mul((g).pow(3))).mul((c).pow(2)))).sub((((((R(12)).mul(a)).mul((g).pow(2))).mul(f)).mul(e)).mul(c))).add(((((R(24)).mul(a)).mul((g).pow(2))).mul(f)).mul((h).pow(2)))).add((((((R(36)).mul(a)).mul(g)).mul((j).pow(2))).mul(e)).mul(c))).add(((((R(12)).mul(a)).mul(g)).mul((j).pow(2))).mul((h).pow(2)))).add(((((((R(12)).mul(a)).mul(g)).mul(j)).mul(f)).mul(b)).mul(c))).sub(((((((R(60)).mul(a)).mul(g)).mul(j)).mul(f)).mul(e)).mul(h))).sub((((((R(12)).mul(a)).mul(g)).mul((f).pow(2))).mul(b)).mul(h))).add(((((R(24)).mul(a)).mul(g)).mul((f).pow(2))).mul((e).pow(2)))).sub(((((R(20)).mul(a)).mul((j).pow(3))).mul(b)).mul(c))).sub(((((R(12)).mul(a)).mul((j).pow(3))).mul(e)).mul(h))).add((((((R(36)).mul(a)).mul((j).pow(2))).mul(f)).mul(b)).mul(h))).add(((((R(12)).mul(a)).mul((j).pow(2))).mul(f)).mul((e).pow(2)))).sub((((((R(24)).mul(a)).mul(j)).mul((f).pow(2))).mul(b)).mul(e))).add((((R(4)).mul(a)).mul((f).pow(3))).mul((b).pow(2)))).add((((R(4)).mul((d).pow(3))).mul(b)).mul((c).pow(2)))).sub(((((R(12)).mul((d).pow(3))).mul(e)).mul(h)).mul(c))).add(((R(8)).mul((d).pow(3))).mul((h).pow(3)))).add(((((R(24)).mul((d).pow(2))).mul(i)).mul((e).pow(2))).mul(c))).sub(((((R(12)).mul((d).pow(2))).mul(i)).mul(e)).mul((h).pow(2)))).add((((((R(12)).mul((d).pow(2))).mul(g)).mul(j)).mul(h)).mul(c))).add((((((R(6)).mul((d).pow(2))).mul(g)).mul(f)).mul(e)).mul(c))).sub((((R(24)).mul((d).pow(2))).mul((j).pow(2))).mul((h).pow(2)))).sub((((((R(12)).mul((d).pow(2))).mul(i)).mul(b)).mul(h)).mul(c))).sub((((R(3)).mul((d).pow(2))).mul((g).pow(2))).mul((c).pow(2)))).sub((((R(24)).mul((g).pow(2))).mul((j).pow(2))).mul((f).pow(2)))).add((((R(24)).mul(g)).mul((j).pow(4))).mul(f))).sub(((((R(12)).mul((d).pow(2))).mul(g)).mul(f)).mul((h).pow(2)))).add(((((R(12)).mul((d).pow(2))).mul((j).pow(2))).mul(e)).mul(c))).sub((((((R(24)).mul((d).pow(2))).mul(j)).mul(f)).mul(b)).mul(c))).sub((((R(27)).mul((d).pow(2))).mul((f).pow(2))).mul((e).pow(2)))).add((((((R(36)).mul((d).pow(2))).mul(j)).mul(f)).mul(e)).mul(h))).add(((((R(24)).mul((d).pow(2))).mul((f).pow(2))).mul(b)).mul(h))).add(((((R(24)).mul(d)).mul((i).pow(2))).mul(b)).mul((h).pow(2)))).sub((((((R(12)).mul(d)).mul((i).pow(2))).mul(b)).mul(e)).mul(c))).sub(((((R(12)).mul(d)).mul((i).pow(2))).mul((e).pow(2))).mul(h))).add((((((R(6)).mul(d)).mul(i)).mul((g).pow(2))).mul(h)).mul(c))).sub(((((((R(60)).mul(d)).mul(i)).mul(g)).mul(j)).mul(e)).mul(c))).add((((((R(36)).mul(d)).mul(i)).mul(g)).mul(j)).mul((h).pow(2)))).add(((((((R(18)).mul(d)).mul(i)).mul(g)).mul(f)).mul(b)).mul(c))).sub(((((((R(6)).mul(d)).mul(i)).mul(g)).mul(f)).mul(e)).mul(h))).add((((((R(36)).mul(d)).mul(i)).mul((j).pow(2))).mul(b)).mul(c))).sub((((((R(12)).mul(d)).mul(i)).mul((j).pow(2))).mul(e)).mul(h))).sub(((((((R(60)).mul(d)).mul(i)).mul(j)).mul(f)).mul(b)).mul(h))).add((((((R(36)).mul(d)).mul(i)).mul(j)).mul(f)).mul((e).pow(2)))).add((((((R(6)).mul(d)).mul(i)).mul((f).pow(2))).mul(b)).mul(e))).add((((((R(12)).mul(d)).mul((g).pow(2))).mul(j)).mul(f)).mul(c))).sub(((((R(12)).mul(d)).mul(g)).mul((j).pow(3))).mul(c))).sub((((((R(12)).mul(d)).mul(g)).mul((j).pow(2))).mul(f)).mul(h))).add((((((R(36)).mul(d)).mul(g)).mul(j)).mul((f).pow(2))).mul(e))).sub(((((R(12)).mul(d)).mul(g)).mul((f).pow(3))).mul(b))).add((((R(24)).mul(d)).mul((j).pow(4))).mul(h))).add(((((R(12)).mul(d)).mul((j).pow(2))).mul((f).pow(2))).mul(b))).add((((R(4)).mul((i).pow(3))).mul((b).pow(2))).mul(c))).add(((((R(24)).mul((i).pow(2))).mul((g).pow(2))).mul(e)).mul(c))).sub((((R(27)).mul((i).pow(2))).mul((g).pow(2))).mul((h).pow(2)))).sub(((((R(36)).mul(d)).mul((j).pow(3))).mul(f)).mul(e))).sub(((((R(12)).mul((i).pow(3))).mul(b)).mul(e)).mul(h))).add(((R(8)).mul((i).pow(3))).mul((e).pow(3)))).sub((((((R(24)).mul((i).pow(2))).mul(g)).mul(j)).mul(b)).mul(c))).add((((((R(36)).mul((i).pow(2))).mul(g)).mul(j)).mul(e)).mul(h))).add((((((R(6)).mul((i).pow(2))).mul(g)).mul(f)).mul(b)).mul(h))).add(((((R(12)).mul((i).pow(2))).mul((j).pow(2))).mul(b)).mul(h))).sub((((R(3)).mul((i).pow(2))).mul((f).pow(2))).mul((b).pow(2)))).sub(((((R(12)).mul(d)).mul((g).pow(2))).mul((f).pow(2))).mul(h))).sub(((((R(12)).mul((i).pow(2))).mul(g)).mul(f)).mul((e).pow(2)))).sub((((R(24)).mul((i).pow(2))).mul((j).pow(2))).mul((e).pow(2)))).add((((((R(12)).mul((i).pow(2))).mul(j)).mul(f)).mul(b)).mul(e))).sub(((((R(12)).mul(i)).mul((g).pow(3))).mul(f)).mul(c))).add(((((R(12)).mul(i)).mul((g).pow(2))).mul((j).pow(2))).mul(c))).add((((((R(36)).mul(i)).mul((g).pow(2))).mul(j)).mul(f)).mul(h))).sub(((((R(12)).mul(i)).mul((g).pow(2))).mul((f).pow(2))).mul(e))).sub(((((R(36)).mul(i)).mul(g)).mul((j).pow(3))).mul(h))).sub((((((R(12)).mul(i)).mul(g)).mul((j).pow(2))).mul(f)).mul(e))).add((((((R(12)).mul(i)).mul(g)).mul(j)).mul((f).pow(2))).mul(b))).add((((R(24)).mul(i)).mul((j).pow(4))).mul(e))).sub(((((R(12)).mul(i)).mul((j).pow(3))).mul(f)).mul(b))).add(((R(8)).mul((g).pow(3))).mul((f).pow(3)))).sub((R(8)).mul((j).pow(6)));
  return [S,T];
}
// GENERATED_ARONHOLD_END


function ternaryCubicJFromAronhold(polyObj) {
  try {
    if (polyDegree(polyObj.poly) !== 3) return null;
    const F = homogenizeAffineCubic(polyObj.poly);
    if ([...F.keys()].some(k => unkey(k).reduce((a,b)=>a+b,0) !== 3)) return null;
    const [S, T] = aronholdSTFromCoefficients(ternaryCubicCoefficientsAronhold(F));
    const denom = R(4).mul(S).pow(3).sub(T.pow(2));
    if (denom.isZero()) return null;
    return R(1728).mul(R(4).mul(S).pow(3)).div(denom);
  } catch { return null; }
}
function evalHom3(poly, P) {
  let total = ZERO; const [X,Y,Z] = P;
  for (const [kk, c] of poly) { const [i,j,k] = unkey(kk); total = total.add(c.mul(X.pow(i)).mul(Y.pow(j)).mul(Z.pow(k))); }
  return total;
}
function derivHom3(poly, idx) {
  const out = new Map();
  for (const [kk, c] of poly) { const m = unkey(kk); if (m[idx] === 0) continue; const mm = [...m]; mm[idx] -= 1; const nk = key(mm); out.set(nk, (out.get(nk)||ZERO).add(c.mul(R(m[idx])))); }
  return polyClean(out);
}
function gradHom3(poly, P) { return [0,1,2].map(i => evalHom3(derivHom3(poly, i), P)); }
function primitivePoint(P) {
  let D = 1n; for (const v of P) D = lcmBig(D, v.den);
  let ints = P.map(v => v.num * (D / v.den)); let g = 0n; for (const a of ints) g = gcdBig(g, absBig(a)); if (g) ints = ints.map(a => a / g);
  const first = ints.find(a => a !== 0n) || 1n; if (first < 0n) ints = ints.map(a => -a);
  return ints.map(a => new Rat(a));
}
function sameProjective(P, Q) {
  return P[0].mul(Q[1]).eq(P[1].mul(Q[0])) && P[0].mul(Q[2]).eq(P[2].mul(Q[0])) && P[1].mul(Q[2]).eq(P[2].mul(Q[1]));
}
function smoothAt(poly, P) { return gradHom3(poly, P).some(g => !g.isZero()); }
function findSmallRationalPoint(poly, bound = 8) {
  const seen = new Set();
  for (let B = 0; B <= bound; B++) {
    const candidates = [];
    for (let X = -B; X <= B; X++) for (let Y = -B; Y <= B; Y++) for (let Z = -B; Z <= B; Z++) {
      if (X === 0 && Y === 0 && Z === 0) continue;
      if (Math.max(Math.abs(X), Math.abs(Y), Math.abs(Z)) !== B) continue;
      if (gcdBig(gcdBig(BigInt(Math.abs(X)), BigInt(Math.abs(Y))), BigInt(Math.abs(Z))) !== 1n) continue;
      const P = primitivePoint([R(X),R(Y),R(Z)]); const sig = P.map(v=>v.toString()).join(','); if (seen.has(sig)) continue; seen.add(sig); candidates.push(P);
    }
    for (const P of candidates) if (evalHom3(poly, P).isZero() && smoothAt(poly, P)) return P;
  }
  return null;
}
function pointOnLineNotMultiple(line, P, requireNotOnCurve = null) {
  const [A,B,C] = line;
  const candidates = [[B,A.neg(),ZERO],[C,ZERO,A.neg()],[ZERO,C,B.neg()],[B.add(C),A.neg(),A.neg()],[C,C,A.add(B).neg()]];
  for (let Q of candidates) {
    if (Q.every(v=>v.isZero())) continue;
    if (!A.mul(Q[0]).add(B.mul(Q[1])).add(C.mul(Q[2])).isZero()) continue;
    Q = primitivePoint(Q); if (sameProjective(Q, P)) continue;
    if (requireNotOnCurve && evalHom3(requireNotOnCurve, Q).isZero()) continue;
    return Q;
  }
  return null;
}
function lineSubstitutionCoeffs(poly, P, Q) {
  const D = Q.map((q,i)=>q.sub(P[i])); const coeffs = [ZERO,ZERO,ZERO,ZERO];
  function linPow(a,b,e) { const vals = Array(e+1).fill(null).map(()=>ZERO); for (let r=0;r<=e;r++) vals[r] = R(comb(e,r)).mul(a.pow(e-r)).mul(b.pow(r)); return vals; }
  for (const [kk, c] of poly) {
    const [i,j,k] = unkey(kk); const px=linPow(P[0],D[0],i), py=linPow(P[1],D[1],j), pz=linPow(P[2],D[2],k);
    const tmp = [ZERO,ZERO,ZERO,ZERO];
    for (let ai=0; ai<px.length; ai++) for (let bi=0; bi<py.length; bi++) for (let ci=0; ci<pz.length; ci++) tmp[ai+bi+ci] = tmp[ai+bi+ci].add(c.mul(px[ai]).mul(py[bi]).mul(pz[ci]));
    for (let n=0; n<4; n++) coeffs[n] = coeffs[n].add(tmp[n]);
  }
  return coeffs;
}
function tangentThirdPoint(poly, P) {
  const grad = gradHom3(poly, P); const Q = pointOnLineNotMultiple(grad, P); if (!Q) return P;
  const c = lineSubstitutionCoeffs(poly, P, Q); if (c[3].isZero()) return P;
  const t3 = c[2].neg().div(c[3]); const Rpt = P.map((p,i)=>p.add(t3.mul(Q[i].sub(p)))); return primitivePoint(Rpt);
}
function matDet3(M) {
  return M[0][0].mul(M[1][1].mul(M[2][2]).sub(M[1][2].mul(M[2][1])))
    .sub(M[0][1].mul(M[1][0].mul(M[2][2]).sub(M[1][2].mul(M[2][0]))))
    .add(M[0][2].mul(M[1][0].mul(M[2][1]).sub(M[1][1].mul(M[2][0]))));
}
function matrixFromColumns(cols) { return [0,1,2].map(i => [0,1,2].map(j => cols[j][i])); }
function substituteLinearHom3(poly, M) {
  const forms = [];
  for (let i=0; i<3; i++) forms.push(new Map([[key([1,0,0]), M[i][0]],[key([0,1,0]), M[i][1]],[key([0,0,1]), M[i][2]]].filter(([,v])=>!v.isZero())));
  let out = new Map();
  for (const [kk, c] of poly) {
    const m = unkey(kk); let term = polyConst(c, 3);
    for (let idx=0; idx<3; idx++) if (m[idx]) term = polyMul(term, polyPow(forms[idx], m[idx], 3), 3);
    out = polyAdd(out, term);
  }
  return polyClean(out);
}
function chooseRNotOnLine(line) {
  const [A,B,C] = line;
  const candidates = [[ONE,ZERO,ZERO],[ZERO,ONE,ZERO],[ZERO,ZERO,ONE],[ONE,ONE,ONE]];
  return candidates.find(Rp => !A.mul(Rp[0]).add(B.mul(Rp[1])).add(C.mul(Rp[2])).isZero()) || null;
}
function weierstrassFromFlex(poly, P) {
  const tangent = gradHom3(poly, P);
  for (let t=0; t<6; t++) {
    const Q = pointOnLineNotMultiple(tangent, P, poly); const RR = chooseRNotOnLine(tangent); if (!Q || !RR) return null;
    const M = matrixFromColumns([Q, P, RR]); if (matDet3(M).isZero()) return null;
    const G = substituteLinearHom3(poly, M); const k = polyCoeff(G,[3,0,0]), c0raw = polyCoeff(G,[0,2,1]); if (k.isZero() || c0raw.isZero()) return null;
    const c2 = polyCoeff(G,[2,0,1]).div(k), c1 = polyCoeff(G,[1,1,1]).div(k), c0 = c0raw.div(k), c4 = polyCoeff(G,[1,0,2]).div(k), c3 = polyCoeff(G,[0,1,2]).div(k), c6 = polyCoeff(G,[0,0,3]).div(k);
    const coeffs = [c1.div(c0), c2.neg().div(c0), c3.neg().div(c0.pow(2)), c4.div(c0.pow(2)), c6.neg().div(c0.pow(3))];
    return qMinimalModel(coeffs);
  }
  return null;
}
function nonflexToWeierstrass(poly, P, P2, P3) {
  const M = matrixFromColumns([P, P2, P3]); if (matDet3(M).isZero()) return null;
  const F2 = substituteLinearHom3(poly, M); const F3 = new Map();
  for (const [kk, c] of F2) {
    const [i,j,k] = unkey(kk); const m = [2*i + k - 2, j, j + k - 1]; if (m.some(e=>e<0)) return null;
    const nk = key(m); F3.set(nk, (F3.get(nk)||ZERO).add(c));
  }
  const F3c = polyClean(F3); const a = polyCoeff(F3c,[3,0,0]); if (a.isZero()) return null;
  const F4 = polyScale(F3c, ONE.div(a)); const b = polyCoeff(F4,[0,2,1]); if (b.isZero()) return null;
  const F5 = new Map(); const zval = ONE.neg().div(b);
  for (const [kk, c] of F4) { const [i,j,k] = unkey(kk); const nk = key([i,j]); F5.set(nk, (F5.get(nk)||ZERO).add(c.mul(zval.pow(k)))); }
  return standardWeierstrassFromPoly({poly: polyClean(F5), symbols: ['x','y']});
}
function generalCubicToWeierstrass(polyObj) {
  try {
    if (polyDegree(polyObj.poly) !== 3) return null;
    const F = homogenizeAffineCubic(polyObj.poly); const P = findSmallRationalPoint(F, 8); if (!P) return null;
    const P2 = tangentThirdPoint(F, P); if (sameProjective(P2, P)) return weierstrassFromFlex(F, P);
    const P3 = tangentThirdPoint(F, P2); if (sameProjective(P3, P2)) return weierstrassFromFlex(F, P2);
    return nonflexToWeierstrass(F, P, P2, P3);
  } catch { return null; }
}
function weierstrassEquationFraction(a1,a2,a3,a4,a6) {
  function term(coef, body) {
    coef = Rat.from(coef); if (coef.isZero()) return '';
    const sign = coef.num < 0n ? '-' : '+'; const mag = coef.abs();
    const text = mag.isOne() && body ? body : `${mag.toString()}${body}`;
    return ` ${sign} ${text}`;
  }
  return `y²${term(a1,'xy')}${term(a3,'y')} = x³${term(a2,'x²')}${term(a4,'x')}${term(a6,'')}`;
}
function coeffsJson(coeffs) { return coeffs.map(c => c.toString()); }
function coeffsIntJson(coeffs) { return coeffs.map(c => c.den === 1n && c.num <= BigInt(Number.MAX_SAFE_INTEGER) && c.num >= BigInt(Number.MIN_SAFE_INTEGER) ? Number(c.num) : c.toString()); }
function safeSymbols(text) { const p = polyFromInput(text); return p ? p.symbols : []; }
function parseGeneralCubicQuery(text) {
  const coeffs = parseCoefficientList(text);
  if (coeffs) {
    const polyObj = polyFromInput(coeffsToWeierstrassAffineExpr(coeffs));
    const parsed = polyObj && standardWeierstrassFromPoly(polyObj);
    return { coeffs: parsed || qMinimalModel(coeffs), match: 'Q-minimal model from Weierstrass coefficients' };
  }
  const polyObj = polyFromInput(text); if (!polyObj) return null;
  let c = standardWeierstrassFromPoly(polyObj); if (c) return { coeffs: c, match: 'Q-minimal model from Weierstrass equation' };
  c = generalCubicToWeierstrass(polyObj); if (c) return { coeffs: c, match: 'Q-minimal model from rational-point plane cubic' };
  const jA = ternaryCubicJFromAronhold(polyObj); if (jA) return { j: jA, match: 'same j-invariant from Aronhold ternary-cubic invariants' };
  const jH = diagonalHesseJFromPoly(polyObj); if (jH) return { j: jH, match: 'same j-invariant from diagonal Hesse cubic' };
  return null;
}
export function identifyCubicJS(text) {
  const raw = normalizeQueryText(text);
  const result = { ok: false, input: raw, version: 'v18-js-bigint', engine: 'js-bigint', notes: [] };
  if (!raw) { result.error = 'Empty input.'; return result; }
  let parsed;
  try { parsed = parseGeneralCubicQuery(raw); } catch (e) { result.error = 'JS parser/recognizer error: ' + (e && e.message ? e.message : String(e)); return result; }
  if (!parsed) { result.error = 'Input was not recognized as a supported two-variable cubic, Weierstrass equation, or coefficient list.'; return result; }
  result.ok = true; result.method = parsed.match || 'recognized cubic'; result.symbols = safeSymbols(raw);
  if (parsed.coeffs) {
    const coeffs = parsed.coeffs.map(Rat.from); const inv = invariantsFraction(...coeffs); const jj = jFromInvariants(inv);
    if (!jj) { result.ok = false; result.error = 'The recognized model has zero discriminant; j-invariant is undefined.'; return result; }
    Object.assign(result, { mode: 'minimal_model', j: jj.toString(), coeffs: coeffsJson(coeffs), coeffs_int: coeffsIntJson(coeffs), weierstrass: weierstrassEquationFraction(...coeffs), discriminant: inv.disc.toString(), c4: inv.c4.toString(), c6: inv.c6.toString() });
    return result;
  }
  if (parsed.j) {
    Object.assign(result, { mode: 'j_only', j: parsed.j.toString(), coeffs: null, coeffs_int: null, weierstrass: null, discriminant: null, c4: null, c6: null });
    result.notes.push('JS computed a smooth ternary-cubic j-invariant. No Q-minimal model was certified by the JS path; Pyodide fallback may upgrade this result if available.');
    return result;
  }
  result.ok = false; result.error = 'Recognition reached no usable j-invariant.'; return result;
}

export const EC_CORE_JS_VERSION = 'v18-js-bigint';
