#!/usr/bin/env python3
"""Build RIES v11.9.2 hypergeometric pFq assets.

This merges the existing v11.5.2 level4/5/6 hypergeometric chunks with
``data.zip``'s ``hyper2f1_grid_v2_blocks_v2_json`` records, then performs:

* 20-decimal continued-fraction rational exclusion;
* row-level de-duplication of complex H values;
* scalar de-duplication for real-search projections, including Re(H)/Im(H);
* level4 placement for the new 2F1 grid, so level4/5/6 cumulative searches see it.
"""
from __future__ import annotations

import base64
import json
import math
import re
import struct
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from decimal import Decimal, ROUND_HALF_UP, getcontext
from fractions import Fraction
from pathlib import Path
from typing import Iterable

getcontext().prec = 90

ROOT = Path(__file__).resolve().parents[1]
OLD_ASSET_GLOB = 'ries-hypdata-v11_5_2-level*.js'
OUT_STATS = ROOT / 'assets' / 'ries-hypdata-v11_9_2-stats.json'
REPORT = ROOT / 'docs' / 'RIES_v11.9.2_HYPDATA_REPORT.md'
VERSION = '11.9.2'
WINDOW_NAME = 'RIES_HYPDATA_V1192_CHUNKS'
DATA_SOURCE_BIT = 4
DATA_SOURCE_NAME = 'data.zip hyper2f1-grid-v2'
DECIMAL_PLACES = 20
Q20 = Decimal('1e-20')
RATIONAL_TOL = Decimal('5e-21')
RATIONAL_MAX_DENOM = 1_000_000
ZERO20 = '0.00000000000000000000'

# Values whose absolute binary64 magnitude is below this are useless for the
# multiplier scan and cause division/noise issues. They are also rational by the
# continued-fraction rule, so this is mostly a safety belt.
FLOAT_ZERO_EPS = 0.0


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')


def unb64(s: str) -> bytes:
    return base64.b64decode((s or '').encode('ascii'))


def unpack_u8(s: str):
    return list(unb64(s))


def unpack_u16(s: str):
    data = unb64(s)
    return list(struct.unpack('<' + 'H' * (len(data) // 2), data))


def unpack_u32(s: str):
    data = unb64(s)
    return list(struct.unpack('<' + 'I' * (len(data) // 4), data))


def unpack_f32(s: str):
    data = unb64(s)
    return list(struct.unpack('<' + 'f' * (len(data) // 4), data))


def unpack_f64(s: str):
    data = unb64(s)
    return list(struct.unpack('<' + 'd' * (len(data) // 8), data))


def pack_u8(xs: Iterable[int]) -> bytes:
    return bytes(int(max(0, min(255, x))) for x in xs)


def pack_u16(xs: Iterable[int]) -> bytes:
    return b''.join(struct.pack('<H', int(max(0, min(65535, x)))) for x in xs)


def pack_u32(xs: Iterable[int]) -> bytes:
    return b''.join(struct.pack('<I', int(x)) for x in xs)


def pack_f32(xs: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<f', float(x)) for x in xs)


def pack_f64(xs: Iterable[float]) -> bytes:
    return b''.join(struct.pack('<d', float(x)) for x in xs)


def parse_js_chunk(path: Path) -> dict:
    s = path.read_text(encoding='utf-8')
    m = re.search(r'=\s*(\{.*\})\s*;\s*\n\}\)\(\);\s*$', s, re.S)
    if not m:
        raise SystemExit(f'Cannot locate chunk JSON in {path}')
    return json.loads(m.group(1))


def decimal20(s: str | Decimal | int | float) -> str:
    d = Decimal(str(s)) if not isinstance(s, Decimal) else s
    if not d.is_finite():
        raise ValueError(f'non-finite decimal {s!r}')
    # Avoid -0.000... in keys/display.
    q = d.quantize(Q20, rounding=ROUND_HALF_UP)
    if q == 0:
        q = abs(q)
    return format(q, f'.{DECIMAL_PLACES}f')


def decimal_key(s: str) -> str:
    # Normalize +0/-0 and exponent forms after 20-place rounding.
    return decimal20(s)


def is_zero20(s: str) -> bool:
    try:
        return Decimal(s) == 0
    except Exception:
        return False


def rational_cf_approx(s20: str):
    """Return a trusted small-denominator rational approximation, else None.

    The input is the 20-decimal display string. Fraction.limit_denominator uses
    continued fractions internally; we then require an absolute residual within
    half of the 20th decimal place.
    """
    x = Decimal(s20)
    frac = Fraction(s20).limit_denominator(RATIONAL_MAX_DENOM)
    approx = Decimal(frac.numerator) / Decimal(frac.denominator)
    if abs(x - approx) <= RATIONAL_TOL:
        return frac
    return None


def is_rational20(s20: str) -> bool:
    return rational_cf_approx(s20) is not None


def parse_rat(tok: str):
    tok = str(tok or '').strip()
    if not tok:
        return None
    if '/' in tok:
        a, b = tok.split('/', 1)
        try:
            return int(a), int(b)
        except Exception:
            return None
    try:
        return int(tok), 1
    except Exception:
        return None


def rat_abs_sum(tok: str) -> int:
    r = parse_rat(tok)
    if not r:
        return 999_999
    n, d = r
    return abs(n) + abs(d)


def mk_parts(mk: str):
    p = str(mk or '').split('|')
    if len(p) < 5:
        return '0', [], [], ''
    return p[1] or '0', [x for x in p[2].split(',') if x], [x for x in p[3].split(',') if x], p[4]


def mk_param_score(mk: str) -> int:
    pref, up, lo, z = mk_parts(mk)
    xs = []
    if pref and pref != '0':
        xs.append(pref)
    xs.extend(up)
    xs.extend(lo)
    if z:
        xs.append(z)
    return sum(rat_abs_sum(x) for x in xs) or 0


def pfq_from_mk(mk: str) -> str:
    _, up, lo, _ = mk_parts(mk)
    return f'{len(up)}F{len(lo)}'


def mk_to_pq(mk: str):
    _, up, lo, _ = mk_parts(mk)
    return len(up), len(lo)


def normalize_new_mk(mk: str) -> str:
    # Input example: 2F1|a=1/12|b=1/12|c=-1/2|z=-1
    parts = str(mk or '').split('|')
    if not parts or parts[0] != '2F1':
        raise ValueError(f'Unsupported MK format: {mk!r}')
    d = {}
    for part in parts[1:]:
        if '=' in part:
            k, v = part.split('=', 1)
            d[k.strip()] = v.strip()
    return f"P|0|{d.get('a','')},{d.get('b','')}|{d.get('c','')}|{d.get('z','')}"


@dataclass
class Row:
    mk: str
    re20: str
    im20: str
    re: float
    im: float
    p: int
    q: int
    source: int
    complexity: int
    stage: int
    origin: str
    score: int

    def row_key(self) -> str:
        return f'{self.re20}|{self.im20}'

    def selector(self):
        # Prefer lower parameter score, then lower complexity/stage, then earlier
        # source. This approximates "sum of absolute coefficients" for pFq rows.
        return (self.score, self.complexity, self.stage, self.source, self.mk)


def load_old_rows() -> tuple[list[Row], dict, list[dict]]:
    rows: list[Row] = []
    old_stats = {}
    mult_by_stage = {}
    input_stats = []
    for path in sorted((ROOT / 'assets').glob(OLD_ASSET_GLOB)):
        ch = parse_js_chunk(path)
        stage = int(ch.get('stage') or 1)
        mk_lines = ch.get('mkBlob', '').split('\n') if ch.get('mkBlob') else []
        value_lines = ch.get('value20Blob', '').split('\n') if ch.get('value20Blob') else []
        p_arr = unpack_u8(ch.get('pB64', ''))
        q_arr = unpack_u8(ch.get('qB64', ''))
        source_arr = unpack_u8(ch.get('sourceB64', ''))
        comp_arr = unpack_u16(ch.get('complexityB64', ''))
        cre = unpack_f64(ch.get('complexReB64', ''))
        cim = unpack_f64(ch.get('complexImB64', ''))
        crow = unpack_u32(ch.get('complexRowB64', ''))
        row_re = [0.0] * len(mk_lines)
        row_im = [0.0] * len(mk_lines)
        for a, b, ri in zip(cre, cim, crow):
            if 0 <= ri < len(row_re):
                row_re[ri] = float(a)
                row_im[ri] = float(b)
        for i, mk in enumerate(mk_lines):
            parts = (value_lines[i] if i < len(value_lines) else f'{row_re[i]:.20f}|{row_im[i]:.20f}').split('|')
            re20 = decimal_key(parts[0] if parts else row_re[i])
            im20 = decimal_key(parts[1] if len(parts) > 1 else row_im[i])
            p, q = (p_arr[i] if i < len(p_arr) else mk_to_pq(mk)[0], q_arr[i] if i < len(q_arr) else mk_to_pq(mk)[1])
            score = mk_param_score(mk)
            rows.append(Row(
                mk=mk,
                re20=re20,
                im20=im20,
                re=row_re[i],
                im=row_im[i],
                p=int(p), q=int(q),
                source=int(source_arr[i]) if i < len(source_arr) else 0,
                complexity=int(comp_arr[i]) if i < len(comp_arr) else min(65535, score),
                stage=stage,
                origin='v11.5.2',
                score=score,
            ))
        # Multipliers are preserved verbatim by stage.
        mult_by_stage[stage] = {
            'multValues': unpack_f64(ch.get('multValuesB64', '')),
            'multComplexity': unpack_f32(ch.get('multComplexityB64', '')),
            'multText': ch.get('multTextBlob', '').split('\n') if ch.get('multTextBlob') else [],
            'multLatex': ch.get('multLatexBlob', '').split('\n') if ch.get('multLatexBlob') else [],
            'multFamilyDict': ch.get('multFamilyDict') or [],
            'multFamilyCodes': unpack_u8(ch.get('multFamilyB64', '')),
        }
        if stage == 1:
            input_stats = ch.get('inputStats') or []
    old_stats['inputStats'] = input_stats
    return rows, old_stats, mult_by_stage


def read_data_zip(data_zip: Path) -> tuple[list[Row], dict]:
    rows: list[Row] = []
    stats = {
        'zip': data_zip.name,
        'sourceBit': DATA_SOURCE_BIT,
        'sourceName': DATA_SOURCE_NAME,
        'totalRecords': 0,
        'okRecords': 0,
        'failedRecords': 0,
        'badFiles': [],
        'blockFiles': 0,
    }
    with zipfile.ZipFile(data_zip) as zf:
        names = sorted(n for n in zf.namelist() if n.endswith('.json'))
        stats['blockFiles'] = len(names)
        for name in names:
            try:
                data = json.loads(zf.read(name).decode('utf-8'))
            except Exception as e:
                stats['badFiles'].append({'file': name, 'reason': str(e)})
                continue
            recs = data.get('records') or []
            for rec in recs:
                stats['totalRecords'] += 1
                ev = rec.get('eval') or {}
                if not ev.get('ok'):
                    stats['failedRecords'] += 1
                    continue
                try:
                    mk = normalize_new_mk(rec.get('MK', ''))
                    re20 = decimal20(ev.get('re', '0'))
                    im20 = decimal20(ev.get('im', '0'))
                    re_v = float(Decimal(ev.get('re', '0')))
                    im_v = float(Decimal(ev.get('im', '0')))
                    score = mk_param_score(mk)
                    rows.append(Row(
                        mk=mk, re20=re20, im20=im20, re=re_v, im=im_v,
                        p=2, q=1, source=DATA_SOURCE_BIT,
                        complexity=max(1, min(65535, score)),
                        stage=1, origin='data.zip', score=score,
                    ))
                    stats['okRecords'] += 1
                except Exception as e:
                    stats['failedRecords'] += 1
                    if len(stats['badFiles']) < 50:
                        stats['badFiles'].append({'file': name, 'recordMK': rec.get('MK'), 'reason': str(e)})
    stats['badFileCount'] = len(stats['badFiles'])
    return rows, stats


def merge_and_filter_rows(old_rows: list[Row], new_rows: list[Row]):
    all_rows = old_rows + new_rows
    raw_count = len(all_rows)
    origin_counts_raw = Counter(r.origin for r in all_rows)

    kept_after_rational: list[Row] = []
    rational_rows = []
    for r in all_rows:
        re_rat = is_rational20(r.re20)
        im_rat = is_rational20(r.im20)
        if re_rat and im_rat:
            rational_rows.append(r)
        else:
            kept_after_rational.append(r)

    groups: dict[str, list[Row]] = defaultdict(list)
    for r in kept_after_rational:
        groups[r.row_key()].append(r)

    final_rows: list[Row] = []
    dup_removed = 0
    duplicate_groups = 0
    source_merge_count = 0
    for key, items in groups.items():
        if len(items) > 1:
            duplicate_groups += 1
            dup_removed += len(items) - 1
        chosen = min(items, key=lambda x: x.selector())
        src_or = 0
        origins = set()
        for it in items:
            src_or |= int(it.source or 0)
            origins.add(it.origin)
        if src_or != chosen.source:
            source_merge_count += 1
        chosen.source = src_or or chosen.source
        # When a lower-stage duplicate exists, keep it visible in lower levels.
        chosen.stage = min(it.stage for it in items)
        final_rows.append(chosen)

    # Stable level order: lower stage first, then pFq/score/MK/value.
    final_rows.sort(key=lambda r: (r.stage, r.p, r.q, r.score, r.mk, Decimal(r.re20), Decimal(r.im20)))
    stats = {
        'rawRows': raw_count,
        'rawOriginCounts': dict(origin_counts_raw),
        'rowRationalExcluded': len(rational_rows),
        'rowRationalExcludedByOrigin': dict(Counter(r.origin for r in rational_rows)),
        'rowDuplicateGroups': duplicate_groups,
        'rowDuplicatesRemoved': dup_removed,
        'sourceMergedDuplicateGroups': source_merge_count,
        'finalRows': len(final_rows),
    }
    return final_rows, stats


def scalar_projection_entries(rows: list[Row]):
    """Create real-search entries: H for real rows, Re(H)/Im(H) for complex rows.

    component code: 0 = H, 1 = Re(H), 2 = Im(H)
    """
    raw = []
    rational_excluded = 0
    rational_excluded_by_component = Counter()
    for i, r in enumerate(rows):
        im_zero = Decimal(r.im20) == 0
        candidates = []
        if im_zero:
            candidates.append((r.re20, r.re, i, 0))
        else:
            candidates.append((r.re20, r.re, i, 1))
            candidates.append((r.im20, r.im, i, 2))
        for v20, v, ri, comp in candidates:
            if is_rational20(v20):
                rational_excluded += 1
                rational_excluded_by_component[str(comp)] += 1
                continue
            if not (v != 0 and math.isfinite(v)):
                continue
            raw.append({'v20': v20, 'v': float(v), 'row': ri, 'comp': comp, 'selector': rows[ri].selector() + (comp,)})

    groups: dict[str, list[dict]] = defaultdict(list)
    for e in raw:
        groups[e['v20']].append(e)

    entries = []
    dup_removed = 0
    dup_groups = 0
    for key, items in groups.items():
        if len(items) > 1:
            dup_groups += 1
            dup_removed += len(items) - 1
        entries.append(min(items, key=lambda e: e['selector']))
    entries.sort(key=lambda e: e['v'])
    stats = {
        'scalarProjectionRawBeforeRational': sum(1 if Decimal(r.im20) == 0 else 2 for r in rows),
        'scalarProjectionRationalExcluded': rational_excluded,
        'scalarProjectionRationalExcludedByComponent': dict(rational_excluded_by_component),
        'scalarProjectionDuplicateGroups': dup_groups,
        'scalarProjectionDuplicatesRemoved': dup_removed,
        'scalarProjectionFinal': len(entries),
        'componentCodeMeaning': {'0': 'real H', '1': 'Re(H)', '2': 'Im(H)'},
    }
    return entries, stats


def complex_entries(rows: list[Row]):
    # Row-level de-dup already guarantees unique 20-decimal complex display keys.
    entries = []
    for i, r in enumerate(rows):
        if math.isfinite(r.re) and math.isfinite(r.im):
            entries.append({'re': r.re, 'im': r.im, 'row': i})
    # Search uses lower_bound on Re. Secondary sort by Im gives deterministic order.
    entries.sort(key=lambda e: (e['re'], e['im'], e['row']))
    return entries


def copy_multipliers(mult_by_stage: dict[int, dict], stage: int) -> dict:
    m = mult_by_stage.get(stage) or {}
    return {
        'values': list(m.get('multValues') or []),
        'complexity': list(m.get('multComplexity') or []),
        'text': list(m.get('multText') or []),
        'latex': list(m.get('multLatex') or []),
        'familyDict': list(m.get('multFamilyDict') or []),
        'familyCodes': list(m.get('multFamilyCodes') or []),
    }


def build_assets(rows: list[Row], scalar_entries: list[dict], mult_by_stage: dict[int, dict], old_stats: dict, data_stats: dict, filter_stats: dict, scalar_stats: dict):
    stage_ranges = {}
    for stage in (1, 2, 3):
        ids = [i for i, r in enumerate(rows) if r.stage == stage]
        if ids:
            stage_ranges[stage] = (min(ids), max(ids) + 1)
        else:
            stage_ranges[stage] = (0, 0)

    assets = []
    level_for_stage = {1: 4, 2: 5, 3: 6}
    labels = {
        1: '2F1/3F2 fast pFq chunk + data.zip 2F1 grid',
        2: '4F3/5F4 classical pFq chunk',
        3: 'full/deep remaining pFq chunk',
    }
    source_rows = filter_stats['rawRows']
    ok_input_rows = filter_stats['rawRows']
    input_stats = list(old_stats.get('inputStats') or []) + [data_stats]

    # Index scalar entries by stage-local ranges.
    for stage in (1, 2, 3):
        lo, hi = stage_ranges[stage]
        stage_rows = rows[lo:hi]
        row_count = len(stage_rows)
        scalar = [e for e in scalar_entries if lo <= e['row'] < hi]
        comp_entries = [e for e in complex_entries(rows) if lo <= e['row'] < hi]
        mult = copy_multipliers(mult_by_stage, stage)
        local_scalar_rows = [e['row'] - lo for e in scalar]
        local_scalar_comp = [e['comp'] for e in scalar]
        local_complex_rows = [e['row'] - lo for e in comp_entries]
        asset = {
            'version': VERSION,
            'level': level_for_stage[stage],
            'stage': stage,
            'label': labels[stage],
            'rowOffset': lo,
            'rows': row_count,
            'globalRows': len(rows),
            'realRows': len(scalar),
            'complexRows': len(comp_entries),
            'decimalPlaces': DECIMAL_PLACES,
            'sourceRows': source_rows,
            'okInputRows': ok_input_rows,
            'inputStats': input_stats if stage == 1 else [],
            'mkBlob': '\n'.join(r.mk for r in stage_rows),
            'value20Blob': '\n'.join(f'{r.re20}|{r.im20}' for r in stage_rows),
            'pB64': b64(pack_u8(r.p for r in stage_rows)),
            'qB64': b64(pack_u8(r.q for r in stage_rows)),
            'sourceB64': b64(pack_u8(r.source for r in stage_rows)),
            'complexityB64': b64(pack_u16(r.complexity for r in stage_rows)),
            'realValuesB64': b64(pack_f64(e['v'] for e in scalar)),
            'realRowB64': b64(pack_u32(local_scalar_rows)),
            'realCompB64': b64(pack_u8(local_scalar_comp)),
            'complexReB64': b64(pack_f64(e['re'] for e in comp_entries)),
            'complexImB64': b64(pack_f64(e['im'] for e in comp_entries)),
            'complexRowB64': b64(pack_u32(local_complex_rows)),
            'multiplierRows': len(mult['values']),
            'multValuesB64': b64(pack_f64(mult['values'])),
            'multComplexityB64': b64(pack_f32(mult['complexity'])),
            'multTextBlob': '\n'.join(mult['text']),
            'multLatexBlob': '\n'.join(mult['latex']),
            'multFamilyDict': mult['familyDict'],
            'multFamilyB64': b64(pack_u8(mult['familyCodes'])),
        }
        out = ROOT / 'assets' / f'ries-hypdata-v11_9_2-level{level_for_stage[stage]}.js'
        js = f'(function(){{\nwindow.{WINDOW_NAME}=window.{WINDOW_NAME}||[];\n'
        js += f'window.{WINDOW_NAME}[{stage - 1}]='
        js += json.dumps(asset, ensure_ascii=False, separators=(',', ':'))
        js += ';\n})();\n'
        out.write_text(js, encoding='utf-8')
        assets.append({
            'stage': stage,
            'level': level_for_stage[stage],
            'file': f'assets/{out.name}',
            'assetBytes': len(js.encode('utf-8')),
            'rows': row_count,
            'realRows': len(scalar),
            'complexRows': len(comp_entries),
            'multiplierRows': len(mult['values']),
            'mkBlobBytes': len(asset['mkBlob'].encode('utf-8')),
            'value20BlobBytes': len(asset['value20Blob'].encode('utf-8')),
        })

    stats = {
        'version': VERSION,
        'sourceVersion': '11.5.2 + data.zip hyper2f1-grid-v2',
        'rows': len(rows),
        'realRows': len(scalar_entries),
        'complexRows': len(rows),
        'decimalPlaces': DECIMAL_PLACES,
        'sourceRows': source_rows,
        'okInputRows': ok_input_rows,
        'continuedFractionRationalExclusion': {
            'decimalPlacesUsed': DECIMAL_PLACES,
            'maxDenominator': RATIONAL_MAX_DENOM,
            'absoluteTolerance': str(RATIONAL_TOL),
        },
        'inputStats': input_stats,
        'mergeFilterStats': filter_stats,
        'scalarProjectionStats': scalar_stats,
        'tierCounts': {str(k): sum(1 for r in rows if r.stage == k) for k in (1, 2, 3)},
        'pFqCounts': dict(sorted(Counter(pfq_from_mk(r.mk) for r in rows).items())),
        'sourceBitNames': {'1': 'data1', '2': 'data2', str(DATA_SOURCE_BIT): DATA_SOURCE_NAME},
        'sourceCounts': dict(sorted(Counter(str(r.source) for r in rows).items(), key=lambda kv: int(kv[0]))),
        'originCountsFinal': dict(Counter(r.origin for r in rows)),
        'multiplierRows': sum(a['multiplierRows'] for a in assets),
        'multiplierStageCounts': {str(a['stage']): a['multiplierRows'] for a in assets},
        'assets': assets,
        'cumulativeAssetBytes': {
            'level4': sum(a['assetBytes'] for a in assets if a['level'] <= 4),
            'level5': sum(a['assetBytes'] for a in assets if a['level'] <= 5),
            'level6': sum(a['assetBytes'] for a in assets if a['level'] <= 6),
        },
        'notes': 'v11.9.2 merges data.zip 2F1 grid into level4, keeps cumulative level4/5/6 loading, excludes trusted 20-decimal continued-fraction rationals, de-duplicates complex rows and real scalar projections, and exposes Re(H)/Im(H) scalar searches via realCompB64.',
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    return stats


def write_report(stats: dict):
    f = stats['mergeFilterStats']
    s = stats['scalarProjectionStats']
    data = next((x for x in stats['inputStats'] if x.get('sourceBit') == DATA_SOURCE_BIT), {})
    lines = []
    lines.append('# RIES v11.9.2 hypergeom 数据库合并、排除与去重报告')
    lines.append('')
    lines.append('## 输入')
    lines.append(f"- 旧库：v11.5.2 三段 hypergeom chunk，原始行数 {f['rawOriginCounts'].get('v11.5.2', 0)}。")
    lines.append(f"- 新数据：`data.zip` / `hyper2f1_grid_v2_blocks_v2_json`，JSON block {data.get('blockFiles', 0)} 个，记录 {data.get('totalRecords', 0)} 条，其中 ok {data.get('okRecords', 0)} 条，失败/跳过 {data.get('failedRecords', 0)} 条。")
    lines.append('')
    lines.append('## 有理数排除')
    lines.append(f"- 方法：所有数先四舍五入保留 {DECIMAL_PLACES} 位小数，再用连分数展开寻找分母 ≤ {RATIONAL_MAX_DENOM} 的有理数；若与 20 位小数值的绝对误差 ≤ {RATIONAL_TOL}，判定为有理数并排除。")
    lines.append(f"- 行级 H 值排除：{f['rowRationalExcluded']} 行；其中旧库 {f['rowRationalExcludedByOrigin'].get('v11.5.2', 0)} 行，新 data.zip {f['rowRationalExcludedByOrigin'].get('data.zip', 0)} 行。")
    lines.append(f"- 实数搜索投影排除：{s['scalarProjectionRationalExcluded']} 个 scalar 投影（component 0=H，1=Re(H)，2=Im(H)；分布 {s['scalarProjectionRationalExcludedByComponent']}）。")
    lines.append('')
    lines.append('## 去重')
    lines.append(f"- 行级复数 H 去重：发现重复组 {f['rowDuplicateGroups']} 个，移除重复行 {f['rowDuplicatesRemoved']} 条；保留规则为参数/系数绝对值求和较小者优先，其次 complexity、stage。")
    lines.append(f"- 实数搜索投影去重：发现重复组 {s['scalarProjectionDuplicateGroups']} 个，移除重复 scalar 投影 {s['scalarProjectionDuplicatesRemoved']} 个。")
    lines.append('')
    lines.append('## 输出')
    lines.append(f"- 最终 H 行数：{stats['rows']}。")
    lines.append(f"- 最终实数搜索 scalar 数量（含 H、Re(H)、Im(H)）：{stats['realRows']}。")
    lines.append(f"- 最终复数搜索 H 数量：{stats['complexRows']}。")
    lines.append(f"- tierCounts：{stats['tierCounts']}。")
    lines.append(f"- pFqCounts：{stats['pFqCounts']}。")
    lines.append('')
    lines.append('## 资产')
    for a in stats['assets']:
        lines.append(f"- {a['file']}：level {a['level']} / stage {a['stage']}，H rows {a['rows']}，real scalar rows {a['realRows']}，complex rows {a['complexRows']}，bytes {a['assetBytes']}。")
    lines.append('')
    lines.append('## 搜索与 LaTeX')
    lines.append('- 新 data.zip 的 2F1 记录放入 level4 chunk，因此 level4/5/6 的累计加载都会使用。')
    lines.append('- 实数搜索新增 `realCompB64`：0 表示 H 本身，1 表示 `\\operatorname{Re}(H)`，2 表示 `\\operatorname{Im}(H)`；前端显示时会生成 `\\operatorname{Re}` / `\\operatorname{Im}` 的 LaTeX。')
    lines.append('- 复数目标搜索继续使用完整 H = Re(H)+i Im(H) 表。')
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv:
        data_zip = Path(argv[0])
    else:
        data_zip = ROOT.parent / 'data.zip'
    if not data_zip.exists():
        raise SystemExit(f'data.zip not found: {data_zip}')
    old_rows, old_stats, mult_by_stage = load_old_rows()
    new_rows, data_stats = read_data_zip(data_zip)
    rows, filter_stats = merge_and_filter_rows(old_rows, new_rows)
    scalar_entries, scalar_stats = scalar_projection_entries(rows)
    stats = build_assets(rows, scalar_entries, mult_by_stage, old_stats, data_stats, filter_stats, scalar_stats)
    write_report(stats)
    print(json.dumps({
        'rows': stats['rows'],
        'realRows': stats['realRows'],
        'complexRows': stats['complexRows'],
        'rowRationalExcluded': filter_stats['rowRationalExcluded'],
        'rowDuplicatesRemoved': filter_stats['rowDuplicatesRemoved'],
        'scalarProjectionRationalExcluded': scalar_stats['scalarProjectionRationalExcluded'],
        'scalarProjectionDuplicatesRemoved': scalar_stats['scalarProjectionDuplicatesRemoved'],
        'assets': stats['assets'],
    }, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
