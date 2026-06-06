#!/usr/bin/env python3
"""Split the RIES v11.5 hypergeometric database into v11.5.1 level chunks.

This script consumes the already merged v11.5 asset, keeps the same 20-decimal
value display strings, and emits three incremental lazy-load assets:
  level4: 2F1/3F2 rows + stage-1 multipliers
  level5: 4F3/5F4 rows + stage-2 multipliers
  level6: all remaining pFq rows + stage-3 multipliers

Higher RIES levels load all lower chunks too, so level5/6 searches compare
low-tier H values with the higher-tier multiplier families rather than only
new H rows.
"""
from __future__ import annotations
import base64, json, re, struct
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
IN_JS = ROOT / 'assets' / 'ries-hypdata-v11_5.js'
OUT_STATS = ROOT / 'assets' / 'ries-hypdata-v11_5_1-stats.json'
VERSION = '11.5.1'

def b64(data: bytes) -> str:
    return base64.b64encode(data).decode('ascii')

def unb64(s: str) -> bytes:
    return base64.b64decode(s.encode('ascii'))

def unpack_u8(s: str):
    return list(unb64(s))

def unpack_u16(s: str):
    data=unb64(s); return list(struct.unpack('<'+'H'*(len(data)//2), data))

def unpack_u32(s: str):
    data=unb64(s); return list(struct.unpack('<'+'I'*(len(data)//4), data))

def unpack_f32(s: str):
    data=unb64(s); return list(struct.unpack('<'+'f'*(len(data)//4), data))

def unpack_f64(s: str):
    data=unb64(s); return list(struct.unpack('<'+'d'*(len(data)//8), data))

def pack_u8(xs): return bytes(int(max(0,min(255,x))) for x in xs)
def pack_u16(xs): return b''.join(struct.pack('<H', int(max(0,min(65535,x)))) for x in xs)
def pack_u32(xs): return b''.join(struct.pack('<I', int(x)) for x in xs)
def pack_f32(xs): return b''.join(struct.pack('<f', float(x)) for x in xs)
def pack_f64(xs): return b''.join(struct.pack('<d', float(x)) for x in xs)

def parse_js_asset(path: Path):
    s=path.read_text(encoding='utf-8')
    m=re.search(r'window\.RIES_HYPDATA_V115\s*=\s*(\{.*\})\s*;\s*\n\}\)\(\);\s*$', s, re.S)
    if not m:
        raise SystemExit('Cannot locate v11.5 hypdata payload in '+str(path))
    return json.loads(m.group(1))

def pfq_from_mk(mk: str):
    parts=str(mk).split('|')
    if len(parts)<5: return '0F0'
    p=len([x for x in parts[2].split(',') if x])
    q=len([x for x in parts[3].split(',') if x])
    return f'{p}F{q}'

def main():
    obj=parse_js_asset(IN_JS)
    mk_lines=obj['mkBlob'].split('\n') if obj.get('mkBlob') else []
    value20_lines=obj['value20Blob'].split('\n') if obj.get('value20Blob') else []
    p_arr=unpack_u8(obj['pB64']); q_arr=unpack_u8(obj['qB64']); tier_arr=unpack_u8(obj['tierB64'])
    source_arr=unpack_u8(obj['sourceB64']); comp_arr=unpack_u16(obj['complexityB64'])
    real_values=unpack_f64(obj['realValuesB64']); real_rows=unpack_u32(obj['realRowB64'])
    complex_re=unpack_f64(obj['complexReB64']); complex_im=unpack_f64(obj['complexImB64']); complex_rows=unpack_u32(obj['complexRowB64'])
    mult_values=unpack_f64(obj['multValuesB64']); mult_stage=unpack_u8(obj['multStageB64']); mult_comp=unpack_f32(obj['multComplexityB64'])
    mult_text=obj['multTextBlob'].split('\n') if obj.get('multTextBlob') else []
    mult_latex=obj['multLatexBlob'].split('\n') if obj.get('multLatexBlob') else []
    mult_family=obj['multFamilyBlob'].split('\n') if obj.get('multFamilyBlob') else []

    if len(mk_lines)!=obj['rows'] or len(value20_lines)!=obj['rows']:
        raise SystemExit('row string blob length mismatch')
    offsets={}
    for stage in (1,2,3):
        ids=[i for i,t in enumerate(tier_arr) if t==stage]
        offsets[stage]=(min(ids), len(ids)) if ids else (0,0)

    assets=[]
    level_for_stage={1:4,2:5,3:6}
    label_for_stage={1:'2F1/3F2 fast pFq chunk',2:'4F3/5F4 classical pFq chunk',3:'full/deep remaining pFq chunk'}
    for stage in (1,2,3):
        row_offset,nrows=offsets[stage]
        row_ids=list(range(row_offset,row_offset+nrows))
        row_set=set(row_ids)
        # Local sorted mirrors.
        r_vals=[]; r_rows=[]
        for v,ri in zip(real_values, real_rows):
            if ri in row_set:
                r_vals.append(v); r_rows.append(ri-row_offset)
        c_re=[]; c_im=[]; c_rows=[]
        for re_v,im_v,ri in zip(complex_re, complex_im, complex_rows):
            if ri in row_set:
                c_re.append(re_v); c_im.append(im_v); c_rows.append(ri-row_offset)
        mult_ids=[i for i,s in enumerate(mult_stage) if s==stage]
        fam_dict=[]; fam_index={}; fam_codes=[]
        for i in mult_ids:
            f=mult_family[i] if i < len(mult_family) else 'multiplier'
            if f not in fam_index:
                fam_index[f]=len(fam_dict); fam_dict.append(f)
            fam_codes.append(fam_index[f])
        asset={
            'version':VERSION,
            'level':level_for_stage[stage],
            'stage':stage,
            'label':label_for_stage[stage],
            'rowOffset':row_offset,
            'rows':nrows,
            'globalRows':obj['rows'],
            'realRows':len(r_vals),
            'complexRows':len(c_rows),
            'decimalPlaces':20,
            'sourceRows':obj.get('sourceRows',0),
            'okInputRows':obj.get('okInputRows',0),
            'inputStats':obj.get('inputStats',[]) if stage==1 else [],
            'mkBlob':'\n'.join(mk_lines[row_offset:row_offset+nrows]),
            'value20Blob':'\n'.join(value20_lines[row_offset:row_offset+nrows]),
            'pB64':b64(pack_u8(p_arr[row_offset:row_offset+nrows])),
            'qB64':b64(pack_u8(q_arr[row_offset:row_offset+nrows])),
            'sourceB64':b64(pack_u8(source_arr[row_offset:row_offset+nrows])),
            'complexityB64':b64(pack_u16(comp_arr[row_offset:row_offset+nrows])),
            'realValuesB64':b64(pack_f64(r_vals)),
            'realRowB64':b64(pack_u32(r_rows)),
            'complexReB64':b64(pack_f64(c_re)),
            'complexImB64':b64(pack_f64(c_im)),
            'complexRowB64':b64(pack_u32(c_rows)),
            'multiplierRows':len(mult_ids),
            'multValuesB64':b64(pack_f64(mult_values[i] for i in mult_ids)),
            'multComplexityB64':b64(pack_f32(mult_comp[i] for i in mult_ids)),
            'multTextBlob':'\n'.join(mult_text[i] for i in mult_ids),
            'multLatexBlob':'\n'.join(mult_latex[i] for i in mult_ids),
            'multFamilyDict':fam_dict,
            'multFamilyB64':b64(pack_u8(fam_codes)),
        }
        name=f'assets/ries-hypdata-v11_5_1-level{level_for_stage[stage]}.js'
        out=ROOT/name
        js='(function(){\nwindow.RIES_HYPDATA_V1151_CHUNKS=window.RIES_HYPDATA_V1151_CHUNKS||[];\n'
        js+=f'window.RIES_HYPDATA_V1151_CHUNKS[{stage-1}]='
        js+=json.dumps(asset, ensure_ascii=False, separators=(',',':'))
        js+=';\n})();\n'
        out.write_text(js, encoding='utf-8')
        assets.append({
            'stage':stage,
            'level':level_for_stage[stage],
            'file':name,
            'assetBytes':len(js.encode('utf-8')),
            'rows':nrows,
            'realRows':len(r_vals),
            'complexRows':len(c_rows),
            'multiplierRows':len(mult_ids),
            'mkBlobBytes':len(asset['mkBlob'].encode('utf-8')),
            'value20BlobBytes':len(asset['value20Blob'].encode('utf-8')),
        })
    stats={
        'version':VERSION,
        'sourceVersion':obj.get('version'),
        'rows':obj['rows'],
        'realRows':obj['realRows'],
        'complexRows':obj['complexRows'],
        'sourceRows':obj.get('sourceRows',0),
        'okInputRows':obj.get('okInputRows',0),
        'inputStats':obj.get('inputStats',[]),
        'tierCounts':{str(k):sum(1 for t in tier_arr if t==k) for k in (1,2,3)},
        'pFqCounts':dict(sorted(Counter(pfq_from_mk(mk) for mk in mk_lines).items())),
        'multiplierRows':len(mult_values),
        'multiplierStageCounts':{str(k):sum(1 for s in mult_stage if s==k) for k in (1,2,3)},
        'assets':assets,
        'cumulativeAssetBytes':{
            'level4':sum(a['assetBytes'] for a in assets if a['level']<=4),
            'level5':sum(a['assetBytes'] for a in assets if a['level']<=5),
            'level6':sum(a['assetBytes'] for a in assets if a['level']<=6),
        },
        'oldSingleAssetBytes':IN_JS.stat().st_size,
        'notes':'v11.5.1 splits the pFq database into incremental level4/5/6 chunks. Higher levels load all lower chunks, then compare all loaded H rows against all loaded multipliers, so 2F1/3F2 rows are still tested with stage-2 and stage-3 coefficients.'
    }
    OUT_STATS.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stats['cumulativeAssetBytes'], indent=2))

if __name__=='__main__':
    main()
