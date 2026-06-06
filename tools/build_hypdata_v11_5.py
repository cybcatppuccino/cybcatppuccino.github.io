#!/usr/bin/env python3
"""Build the lazy RIES v11.5 hypergeometric pFq database asset.

Inputs are the two user-provided block zip files.  The output follows the
v11.4.3 harddb direct-JS pattern: compact typed-array payloads plus lightweight
string blobs, loaded lazily by ries-script.js only when a decimal search can use
it.
"""
from __future__ import annotations
import argparse, base64, json, math, os, re, struct, zipfile
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, getcontext, ROUND_HALF_EVEN
from fractions import Fraction
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA1_CANDIDATES = [ROOT / 'data1.zip', ROOT.parent / 'data1.zip', Path('/mnt/data/data1.zip')]
DEFAULT_DATA2_CANDIDATES = [ROOT / 'data2.zip', ROOT.parent / 'data2.zip', Path('/mnt/data/data2.zip')]
DATA1 = None
DATA2 = None
OUT_JS = ROOT / 'assets' / 'ries-hypdata-v11_5.js'
OUT_STATS = ROOT / 'assets' / 'ries-hypdata-v11_5-stats.json'
VERSION = '11.5'
getcontext().prec = 90
Q20 = Decimal('0.00000000000000000001')


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def pack_f64(xs: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<d', float(x)) for x in xs)


def pack_f32(xs: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<f', float(x)) for x in xs)


def pack_u32(xs: Iterable[int]) -> bytes:
    return b''.join(struct.pack('<I', int(x)) for x in xs)


def pack_u16(xs: Iterable[int]) -> bytes:
    return b''.join(struct.pack('<H', int(max(0, min(65535, x)))) for x in xs)


def pack_u8(xs: Iterable[int]) -> bytes:
    return bytes(int(max(0, min(255, x))) for x in xs)


def dec20(s: str) -> str:
    try:
        return format(Decimal(str(s)).quantize(Q20, rounding=ROUND_HALF_EVEN), 'f')
    except Exception:
        return str(s)


def finite_float(s: str) -> float | None:
    try:
        x = float(s)
    except Exception:
        return None
    return x if math.isfinite(x) else None


def frac_height(tok: str) -> int:
    tok = str(tok).strip()
    if not tok:
        return 0
    try:
        f = Fraction(tok)
        return max(abs(f.numerator), abs(f.denominator))
    except Exception:
        return max(1, len(tok))


def parse_mk(mk: str):
    parts = str(mk).split('|')
    # Expected: P|prefactor exponent|upper params|lower params|z
    if len(parts) < 5:
        return {'pref':'0','upper':[], 'lower':[], 'z':'', 'p':0, 'q':0, 'height':999, 'tier':3}
    pref, upper, lower, z = parts[1], parts[2], parts[3], parts[4]
    ups = [x for x in upper.split(',') if x != '']
    lows = [x for x in lower.split(',') if x != '']
    p, q = len(ups), len(lows)
    hs = [frac_height(x) for x in [pref, z] + ups + lows]
    height = max(hs or [1])
    # Stage tags: 1 = common 2F1/3F2; 2 = classical 4F3/5F4; 3 = all others.
    if (p, q) in {(2,1), (3,2)}:
        tier = 1
    elif (p, q) in {(4,3), (5,4)}:
        tier = 2
    else:
        tier = 3
    zscore = 0 if z in {'1','-1','1/2','-1/2','1/4','-1/4','1/3','-1/3'} else 4
    complexity = min(65535, 10 + 4*p + 4*q + int(math.log10(max(1,height))*18) + zscore)
    return {'pref':pref, 'upper':ups, 'lower':lows, 'z':z, 'p':p, 'q':q, 'height':height, 'tier':tier, 'complexity':complexity}


def read_zip(path: Path, source_bit: int):
    bad_files = []
    total = ok = failed = 0
    rows = {}
    with zipfile.ZipFile(path) as z:
        for name in sorted(z.namelist()):
            if not name.endswith('.json'):
                continue
            raw = z.read(name)
            if not raw.strip():
                bad_files.append({'file':name, 'reason':'empty'})
                continue
            try:
                obj = json.loads(raw)
            except Exception as e:
                bad_files.append({'file':name, 'reason':str(e)[:120]})
                continue
            for rec in obj.get('records', []):
                total += 1
                mk = rec.get('MK')
                ev = rec.get('eval') or {}
                if not mk or not ev.get('ok'):
                    failed += 1
                    continue
                re_s, im_s = str(ev.get('re','0')), str(ev.get('im','0'))
                re_f, im_f = finite_float(re_s), finite_float(im_s)
                if re_f is None or im_f is None:
                    failed += 1
                    continue
                ok += 1
                rows[mk] = {'mk':mk, 're_s':re_s, 'im_s':im_s, 're':re_f, 'im':im_f, 'source_bit':source_bit}
    return rows, {'zip':path.name, 'sourceBit':source_bit, 'totalRecords':total, 'okRecords':ok, 'failedRecords':failed, 'badFiles':bad_files[:12], 'badFileCount':len(bad_files)}


def merge_rows():
    merged = {}
    source_flags = defaultdict(int)
    stats = []
    for path, bit in [(DATA1, 1), (DATA2, 2)]:
        rows, st = read_zip(path, bit)
        stats.append(st)
        for mk, r in rows.items():
            source_flags[mk] |= bit
            if mk not in merged:
                merged[mk] = dict(r)
            else:
                # Prefer the first successful value, but keep source information.
                pass
    out = []
    for mk, r in merged.items():
        meta = parse_mk(mk)
        src = source_flags[mk]
        out.append({**r, **meta, 'source':src, 're20':dec20(r['re_s']), 'im20':dec20(r['im_s'])})
    return out, stats


@dataclass(frozen=True)
class Mult:
    value: float
    text: str
    latex: str
    family: str
    stage: int
    complexity: float


def rat_cost(n:int,d:int)->float:
    h=max(abs(n),abs(d))
    if h<=1: return 0
    return math.log2(h+1)*0.95


def rat_text_latex(n:int,d:int):
    if d == 1:
        return str(abs(n)), str(abs(n))
    return f'{abs(n)}/{d}', f'\\frac{{{abs(n)}}}{{{d}}}'


def pow_text(base:str, exp:int):
    if exp == 0: return '', ''
    if exp == 1: return base, base
    return f'{base}^{exp}', f'{base}^{{{exp}}}'


def pi_text(d2:int):
    if d2 == 0: return '', ''
    if d2 == 2: return 'π', '\\pi'
    if d2 == -2: return 'π^-1', '\\pi^{-1}'
    if d2 == 1: return 'sqrt(π)', '\\sqrt{\\pi}'
    if d2 == -1: return 'π^(-1/2)', '\\pi^{-1/2}'
    if d2 % 2 == 0:
        return f'π^{d2//2}', f'\\pi^{{{d2//2}}}'
    return f'π^({d2}/2)', f'\\pi^{{{d2}/2}}'


def sqrt_text(n:int, exp:int):
    if exp == 0: return '', ''
    if exp == 1: return f'√{n}', f'\\sqrt{{{n}}}'
    if exp == -1: return f'1/√{n}', f'1/\\sqrt{{{n}}}'
    return f'(√{n})^{exp}', f'(\\sqrt{{{n}}})^{{{exp}}}'


def gamma_text(r:str, exp:int):
    if exp == 0: return '', ''
    if exp == 1: return f'Γ({r})', f'\\Gamma({r})'
    return f'Γ({r})^{exp}', f'\\Gamma({r})^{{{exp}}}'

GAMMA_VALS = {
    '1/3': math.gamma(1/3), '1/4': math.gamma(1/4), '1/6': math.gamma(1/6),
    '1/5': math.gamma(1/5), '2/5': math.gamma(2/5), '1/8': math.gamma(1/8), '3/8': math.gamma(3/8),
    '1/10': math.gamma(1/10), '3/10': math.gamma(3/10), '1/12': math.gamma(1/12), '5/12': math.gamma(5/12),
}


def regular_comps(stage:int):
    if stage == 1:
        prime_specs=[('2',2,-6,6,1.0),('3',3,-6,6,1.0),('5',5,-4,4,1.5)]
        pi_range=range(-6,7); sqrt_opts=[(1,'','',0.0),(math.sqrt(2),'√2','\\sqrt{2}',1.7),(1/math.sqrt(2),'1/√2','1/\\sqrt{2}',1.7),(math.sqrt(3),'√3','\\sqrt{3}',1.7),(1/math.sqrt(3),'1/√3','1/\\sqrt{3}',1.7),(math.sqrt(5),'√5','\\sqrt{5}',2.0),(1/math.sqrt(5),'1/√5','1/\\sqrt{5}',2.0)]
        max_cost, cap = 7.2, 900
    elif stage == 2:
        prime_specs=[('2',2,-9,9,1.0),('3',3,-9,9,1.0),('5',5,-6,6,1.5),('7',7,-3,3,2.2)]
        pi_range=range(-12,13); sqrt_opts=[(1,'','',0.0)]
        for n,c in [(2,1.6),(3,1.6),(5,1.8),(6,2.0),(7,2.2),(10,2.4)]:
            sqrt_opts += [(math.sqrt(n),f'√{n}',f'\\sqrt{{{n}}}',c),(1/math.sqrt(n),f'1/√{n}',f'1/\\sqrt{{{n}}}',c)]
        max_cost, cap = 10.6, 2600
    else:
        prime_specs=[('2',2,-12,12,1.0),('3',3,-12,12,1.0),('5',5,-9,9,1.5),('7',7,-5,5,2.2),('11',11,-2,2,3.5)]
        pi_range=range(-20,21); sqrt_opts=[(1,'','',0.0)]
        for n,c in [(2,1.5),(3,1.5),(5,1.7),(6,1.9),(7,2.1),(10,2.2),(15,2.5)]:
            sqrt_opts += [(math.sqrt(n),f'√{n}',f'\\sqrt{{{n}}}',c),(1/math.sqrt(n),f'1/√{n}',f'1/\\sqrt{{{n}}}',c)]
        max_cost, cap = 13.5, 5200
    # Generate prime-power products recursively; pi and one radical are attached afterwards.
    prime_comps=[]
    def rec(i,value,cost,texts,latexs):
        if cost>max_cost: return
        if i==len(prime_specs):
            prime_comps.append((value,cost,texts[:],latexs[:]))
            return
        sym,val,lo,hi,w=prime_specs[i]
        for e in range(lo,hi+1):
            ec=abs(e)*w
            if cost+ec>max_cost: continue
            if e==0:
                rec(i+1,value,cost,texts,latexs)
            else:
                t,l=pow_text(sym,e)
                rec(i+1,value*math.pow(val,e),cost+ec,texts+[t],latexs+[l])
    rec(0,1.0,0.0,[],[])
    out=[]; seen={}
    for pv,pc,pt,pl in prime_comps:
        for d2 in pi_range:
            pic=abs(d2)*1.2
            if pc+pic>max_cost: continue
            pit,pil=pi_text(d2); piv=math.pow(math.pi,d2/2)
            for sv,st,sl,sc in sqrt_opts:
                cost=pc+pic+sc
                if cost>max_cost: continue
                val=pv*piv*sv
                texts=pt+([pit] if pit else [])+([st] if st else [])
                latexs=pl+([pil] if pil else [])+([sl] if sl else [])
                key=f'{val:.15e}'
                item=(val,cost,'·'.join(texts),'\\,'.join(latexs),'pi-radical-prime factor')
                old=seen.get(key)
                if old is None or (cost,len(item[2]))<(old[1],len(old[2])):
                    seen[key]=item
    out=list(seen.values())
    out.sort(key=lambda x:(x[1], len(x[2]), abs(math.log(abs(x[0])) if x[0] else 999)))
    return out[:cap]


def gamma_comps(stage:int):
    if stage == 1:
        return [(1.0,0.0,'','','no gamma')]
    if stage == 2:
        bases=[('1/3',3,4.0),('1/4',3,4.0),('1/6',3,4.2)]
        max_cost, cap = 8.4, 160
    else:
        bases=[('1/3',4,3.8),('1/4',4,3.8),('1/6',4,4.0),('1/5',2,5.2),('2/5',2,5.2),('1/8',2,5.5),('3/8',2,5.5),('1/10',1,6.0),('3/10',1,6.0),('1/12',1,6.2),('5/12',1,6.2)]
        max_cost, cap = 12.5, 420
    out=[(1.0,0.0,'','','no gamma')]
    # singles
    for r,mx,w in bases:
        for e in range(-mx,mx+1):
            if not e: continue
            c=abs(e)*w
            if c<=max_cost:
                t,l=gamma_text(r,e); out.append((math.pow(GAMMA_VALS[r],e),c,t,l,'gamma quotient'))
    # pairs; this is intentionally limited to avoid noisy high-dimensional gamma products.
    for i,(r1,mx1,w1) in enumerate(bases):
        for r2,mx2,w2 in bases[i+1:]:
            for e1 in range(-mx1,mx1+1):
                if not e1: continue
                for e2 in range(-mx2,mx2+1):
                    if not e2: continue
                    c=abs(e1)*w1+abs(e2)*w2
                    if c>max_cost: continue
                    t1,l1=gamma_text(r1,e1); t2,l2=gamma_text(r2,e2)
                    out.append((math.pow(GAMMA_VALS[r1],e1)*math.pow(GAMMA_VALS[r2],e2),c,t1+'·'+t2,l1+'\\,'+l2,'gamma quotient'))
    # core triples only for stage 3/2 if cost permits.
    core=bases[:3]
    for e1 in range(-2,3):
        for e2 in range(-2,3):
            for e3 in range(-2,3):
                if not (e1 or e2 or e3): continue
                c=abs(e1)*core[0][2]+abs(e2)*core[1][2]+abs(e3)*core[2][2]
                if c>max_cost: continue
                texts=[]; latexs=[]; val=1.0
                for (r,_,_),e in zip(core,[e1,e2,e3]):
                    if e:
                        t,l=gamma_text(r,e); texts.append(t); latexs.append(l); val*=math.pow(GAMMA_VALS[r],e)
                out.append((val,c,'·'.join(texts),'\\,'.join(latexs),'gamma quotient'))
    seen={}
    for v,c,t,l,f in out:
        key=f'{v:.15e}'
        old=seen.get(key)
        if old is None or (c,len(t))<(old[1],len(old[2])): seen[key]=(v,c,t,l,f)
    out=list(seen.values())
    out.sort(key=lambda x:(x[1], len(x[2]), abs(math.log(abs(x[0])) if x[0] else 999)))
    return out[:cap]


def factor_combos(stage:int):
    regs=regular_comps(stage)
    gams=gamma_comps(stage)
    out=[]; seen={}
    if stage == 1:
        raw=regs
    else:
        raw=list(regs)
        reg_for_gamma=regs[:650 if stage==2 else 1100]
        for rv,rc,rt,rl,rf in reg_for_gamma:
            for gv,gc,gt,gl,gf in gams:
                if not gt: continue
                cost=rc+gc
                if cost>(13.2 if stage==2 else 18.2): continue
                text='·'.join([x for x in (rt,gt) if x])
                latex='\\,'.join([x for x in (rl,gl) if x])
                raw.append((rv*gv,cost,text,latex,'gamma quotient'))
    for v,c,t,l,f in raw:
        if not (math.isfinite(v) and v!=0): continue
        key=f'{v:.15e}'
        old=seen.get(key)
        if old is None or (c,len(t))<(old[1],len(old[2])):
            seen[key]=(v,c,t,l,f)
    out=list(seen.values())
    out.sort(key=lambda x:(x[1], len(x[2]), abs(math.log(abs(x[0])) if x[0] else 999)))
    return out[:{1:900,2:3600,3:9000}[stage]]

def rational_options(max_h:int):
    out=[]
    for n in range(1,max_h+1):
        for d in range(1,max_h+1):
            if math.gcd(n,d) != 1: continue
            c=rat_cost(n,d)
            t,l=rat_text_latex(n,d)
            out.append((n/d,c,t,l,n,d))
    out.sort(key=lambda x:(x[1], x[5], x[4]))
    return out


def build_multipliers():
    stage_cfg = {
        1: {'max_h':12, 'max_cost':10.0, 'cap':1200, 'rat_cap':80, 'factor_cap':500},
        2: {'max_h':30, 'max_cost':18.0, 'cap':6500, 'rat_cap':220, 'factor_cap':1200},
        3: {'max_h':100,'max_cost':24.0, 'cap':16000, 'rat_cap':350, 'factor_cap':2000},
    }
    by_key={}
    for stage in (1,2,3):
        comps=factor_combos(stage)[:stage_cfg[stage]['factor_cap']]
        rats=rational_options(stage_cfg[stage]['max_h'])[:stage_cfg[stage]['rat_cap']]
        for fval,fcost,ftext,flatex,family in comps:
            for rval,rcost,rtext,rlatex,n,d in rats:
                cost=fcost+rcost
                if cost > stage_cfg[stage]['max_cost']:
                    break
                for sign in (1,-1):
                    val = sign * rval * fval
                    if not (math.isfinite(val) and val != 0 and 1e-300 < abs(val) < 1e300):
                        continue
                    parts=[]; lparts=[]
                    if not (n == 1 and d == 1):
                        parts.append(rtext); lparts.append(rlatex)
                    if ftext:
                        parts.append(ftext); lparts.append(flatex)
                    text='·'.join(parts) if parts else '1'
                    latex='\\,'.join(lparts) if lparts else '1'
                    if sign < 0:
                        text='-'+text if text!='1' else '-1'
                        latex='-'+latex if latex!='1' else '-1'
                    key=f'{val:.15e}'
                    old=by_key.get(key)
                    m=Mult(val,text,latex,family,stage,cost+(0.4 if sign<0 else 0))
                    if old is None or (m.stage, m.complexity, len(m.text)) < (old.stage, old.complexity, len(old.text)):
                        by_key[key]=m
        # Cap cumulatively per stage to keep the asset predictable.
        cur=[m for m in by_key.values() if m.stage<=stage]
        cur.sort(key=lambda m:(m.stage, m.complexity, len(m.text), abs(math.log(abs(m.value)))))
        keep=set(f'{m.value:.15e}' for m in cur[:stage_cfg[stage]['cap']])
        by_key={k:v for k,v in by_key.items() if k in keep or v.stage<stage}
    mults=list(by_key.values())
    mults.sort(key=lambda m:(m.stage, m.complexity, len(m.text), abs(math.log(abs(m.value)))))
    # Re-stage cumulative order: first 1500 are stage1, next up to 14k stage2, rest stage3.
    return mults


def js_string(s: str) -> str:
    return json.dumps(s, ensure_ascii=False)


def first_existing(paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[0])

def main():
    global DATA1, DATA2
    import sys, time
    ap = argparse.ArgumentParser(description='Build the RIES v11.5 lazy hypergeometric pFq database asset.')
    ap.add_argument('--data1', default=None, help='Path to data1.zip; defaults to data1.zip near the project, then /mnt/data/data1.zip')
    ap.add_argument('--data2', default=None, help='Path to data2.zip; defaults to data2.zip near the project, then /mnt/data/data2.zip')
    args = ap.parse_args()
    DATA1 = Path(args.data1) if args.data1 else first_existing(DEFAULT_DATA1_CANDIDATES)
    DATA2 = Path(args.data2) if args.data2 else first_existing(DEFAULT_DATA2_CANDIDATES)
    if not DATA1.exists() or not DATA2.exists():
        raise SystemExit(f'Missing input zip(s): {DATA1} {DATA2}')
    print(f'hypdata build: using {DATA1} and {DATA2}', flush=True)
    print('hypdata build: merge rows...', flush=True)
    _t=time.time()
    rows, input_stats = merge_rows()
    print('hypdata build: merged', len(rows), 'in', round(time.time()-_t,2), flush=True)
    rows.sort(key=lambda r:(r['tier'], r['p'], r['q'], r['height'], r['mk']))
    print('hypdata build: sorted', flush=True)
    for i,r in enumerate(rows):
        r['id'] = i
    real_pairs = [(r['re'], r['id']) for r in rows if abs(r['im']) <= 5e-21 and r['re'] != 0 and math.isfinite(r['re'])]
    real_pairs.sort(key=lambda x:x[0])
    complex_rows = [(r['re'], r['im'], r['id']) for r in rows if not (r['re']==0 and r['im']==0)]
    complex_rows.sort(key=lambda x:(x[0], x[1]))
    print('hypdata build: build multipliers...', flush=True)
    _t=time.time()
    mults = build_multipliers()
    print('hypdata build: multipliers', len(mults), 'in', round(time.time()-_t,2), flush=True)
    stage_counts = {str(s):sum(1 for m in mults if m.stage==s) for s in (1,2,3)}

    mk_blob='\n'.join(r['mk'] for r in rows)
    value20_blob='\n'.join(r['re20']+'|'+r['im20'] for r in rows)
    mult_text_blob='\n'.join(m.text for m in mults)
    mult_latex_blob='\n'.join(m.latex for m in mults)
    mult_family_blob='\n'.join(m.family for m in mults)

    p_arr=[r['p'] for r in rows]
    q_arr=[r['q'] for r in rows]
    tier_arr=[r['tier'] for r in rows]
    source_arr=[r['source'] for r in rows]
    comp_arr=[r['complexity'] for r in rows]

    asset = {
        'version': VERSION,
        'rows': len(rows),
        'realRows': len(real_pairs),
        'complexRows': len(complex_rows),
        'decimalPlaces': 20,
        'sourceRows': sum(s['totalRecords'] for s in input_stats),
        'okInputRows': sum(s['okRecords'] for s in input_stats),
        'inputStats': input_stats,
        'mkBlob': mk_blob,
        'value20Blob': value20_blob,
        'pB64': b64(pack_u8(p_arr)),
        'qB64': b64(pack_u8(q_arr)),
        'tierB64': b64(pack_u8(tier_arr)),
        'sourceB64': b64(pack_u8(source_arr)),
        'complexityB64': b64(pack_u16(comp_arr)),
        'realValuesB64': b64(pack_f64(v for v,_ in real_pairs)),
        'realRowB64': b64(pack_u32(i for _,i in real_pairs)),
        'complexReB64': b64(pack_f64(re for re,_,_ in complex_rows)),
        'complexImB64': b64(pack_f64(im for _,im,_ in complex_rows)),
        'complexRowB64': b64(pack_u32(i for _,_,i in complex_rows)),
        'multiplierRows': len(mults),
        'multiplierStageCounts': stage_counts,
        'multValuesB64': b64(pack_f64(m.value for m in mults)),
        'multStageB64': b64(pack_u8(m.stage for m in mults)),
        'multComplexityB64': b64(pack_f32(m.complexity for m in mults)),
        'multTextBlob': mult_text_blob,
        'multLatexBlob': mult_latex_blob,
        'multFamilyBlob': mult_family_blob,
    }
    print('hypdata build: constructing js...', flush=True)
    js = '(function(){\n  window.RIES_HYPDATA_V115 = '
    # Use ensure_ascii=False so π/Γ in multiplier text remains readable.
    js += json.dumps(asset, ensure_ascii=False, separators=(',',':'))
    js += ';\n})();\n'
    print('hypdata build: writing js', len(js), flush=True)
    OUT_JS.write_text(js, encoding='utf-8')
    print('hypdata build: wrote js', flush=True)
    from collections import Counter
    tier_counter = Counter(str(r['tier']) for r in rows)
    pfq_counter = Counter(f"{r['p']}F{r['q']}" for r in rows)
    stats={
        'version': VERSION,
        'rows': len(rows),
        'realRows': len(real_pairs),
        'complexRows': len(complex_rows),
        'sourceRows': asset['sourceRows'],
        'okInputRows': asset['okInputRows'],
        'inputStats': input_stats,
        'tierCounts': dict(sorted(tier_counter.items())),
        'pFqCounts': dict(sorted(pfq_counter.items())),
        'multiplierRows': len(mults),
        'multiplierStageCounts': stage_counts,
        'assetBytes': len(js.encode('utf-8')),
        'mkBlobBytes': len(mk_blob.encode('utf-8')),
        'value20BlobBytes': len(value20_blob.encode('utf-8')),
        'notes': 'Values are deduplicated by MK; successful re/im values are rounded into value20Blob at 20 decimal places for display, while Float64 mirrors are used for fast low-precision search.'
    }
    print('hypdata build: writing stats', flush=True)
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print('hypdata build: wrote stats', flush=True)
    print('hypdata build: done', stats['rows'], 'rows', stats['multiplierRows'], 'multipliers')

if __name__ == '__main__':
    main()
