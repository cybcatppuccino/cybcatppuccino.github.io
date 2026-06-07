import base64, json, re, struct
from pathlib import Path

VERSION='11.7.3-harddb-pruned-full-20260607'
SRC=Path('assets/ries-harddb-v11_4_1-filtered.js')
OUT=Path('assets/ries-harddb-v11_7_3-level4.js')
STATS=Path('assets/ries-harddb-v11_7_3-stats.json')
REMOVE_CATEGORIES={
    'low-height hypergeometric pFq',
    'Euler beta integral fast',
    'incomplete beta integral fast',
    'beta logarithmic integral fast',
    'gamma log-laplace integral fast',
    'rational Mellin integral fast',
}

def load_direct(path):
    s=path.read_text()
    m=re.search(r'window\.RIES_HARDDB_V114_DIRECT\s*=\s*(\{.*\})\s*;\s*\}\)\(\);', s, re.S)
    if not m:
        raise SystemExit(f'Cannot locate direct harddb object in {path}')
    return json.loads(m.group(1))

def b64_values(values, indices):
    return base64.b64encode(struct.pack('<%sd'%len(indices), *[values[i] for i in indices])).decode()

def b64_orig(orig, indices):
    return base64.b64encode(struct.pack('<%sI'%len(indices), *[orig[i] for i in indices])).decode()

def b64_rowmap(rowmap, indices):
    out=bytearray()
    for i in indices:
        out.extend(rowmap[i*3:i*3+3])
    return base64.b64encode(out).decode()

def height_for_param_text(txt):
    h=1
    txt=str(txt or '').strip().strip('"')
    for a,b in re.findall(r'(-?\d+)/(\d+)', txt):
        h=max(h, abs(int(a)), abs(int(b)))
    for n in re.findall(r'(?<![A-Za-z])-?\d+(?![A-Za-z])', txt):
        try:
            h=max(h, abs(int(n)))
        except Exception:
            pass
    return h

def params_for(row, d, rowmap, param_bytes, dict_vals):
    off=row*3
    cid=rowmap[off]
    ord_=rowmap[off+1] | (rowmap[off+2]<<8)
    keys=d['paramKeys'][cid]
    start=(int(d['catParamOffsets'][cid]) + ord_*len(keys))*2
    p={}
    for kidx,k in enumerate(keys):
        code=param_bytes[start+2*kidx] | (param_bytes[start+2*kidx+1]<<8)
        p[k]=dict_vals[code] if code < len(dict_vals) else ''
    return cid,p

d=load_direct(SRC)
rows=int(d['rows'])
values=list(struct.unpack('<%sd'%rows, base64.b64decode(d['valuesB64'])))
rowmap=base64.b64decode(d['rowMapB64'])
orig=list(struct.unpack('<%sI'%rows, base64.b64decode(d['origRowsB64'])))
param_bytes=base64.b64decode(d['paramB64'])
dict_vals=str(d['dict']).split('\t')
cat_names=list(d['categories'])
removed_cids={i for i,c in enumerate(cat_names) if c in REMOVE_CATEGORIES}
if len(removed_cids)!=len(REMOVE_CATEGORIES):
    missing=REMOVE_CATEGORIES-set(cat_names)
    raise SystemExit(f'Missing categories: {sorted(missing)}')
source_counts=[0]*len(cat_names)
for i in range(rows):
    source_counts[rowmap[i*3]]+=1
keep=[i for i in range(rows) if rowmap[i*3] not in removed_cids]
# Preserve sorted numeric table for stable scanning/reporting.
keep.sort(key=lambda i:(values[i], i))
kept_counts=[0]*len(cat_names)
for i in keep:
    kept_counts[rowmap[i*3]]+=1
removed_counts={cat_names[cid]: source_counts[cid] for cid in sorted(removed_cids)}
heights=[]
for i in keep:
    _,p=params_for(i,d,rowmap,param_bytes,dict_vals)
    h=1
    for v in p.values():
        h=max(h, height_for_param_text(v))
    heights.append(h)
obj={
    'version': VERSION,
    'stage': 1,
    'level': 4,
    'rows': len(keep),
    'sourceRows': rows,
    'removedRows': rows-len(keep),
    'selectionPolicy': 'v11.7.3 full remaining harddb after removing overlapping fast/hypergeometric/beta/Mellin categories; depths 4/5/6 share these rows and differ only by comparison constant families',
    'removedCategories': sorted(REMOVE_CATEGORIES),
    'categories': d['categories'],
    'paramKeys': d['paramKeys'],
    'catCounts': kept_counts,
    'sourceCatCounts': source_counts,
    'removedCatCounts': removed_counts,
    'catParamOffsets': d['catParamOffsets'],
    'dict': d['dict'],
    'valuesB64': b64_values(values, keep),
    'rowMapB64': b64_rowmap(rowmap, keep),
    'origRowsB64': b64_orig(orig, keep),
    'paramB64': d['paramB64'],
    'valueBytes': len(keep)*8,
    'rowMapBytes': len(keep)*3,
    'origRowBytes': len(keep)*4,
    'paramBytes': d['paramBytes'],
    'maxParamHeight': max(heights) if heights else 0,
    'medianParamHeight': sorted(heights)[len(heights)//2] if heights else 0,
}
text='// RIES v11.7.3 pruned full hard-constant database asset.\n(function(){\n  window.RIES_HARDDB_V1173_CHUNKS = window.RIES_HARDDB_V1173_CHUNKS || [];\n  window.RIES_HARDDB_V1173_CHUNKS[0] = %s;\n})();\n' % json.dumps(obj,separators=(',',':'))
OUT.write_text(text)
stats={
    'version': VERSION,
    'sourceAsset': SRC.name,
    'sourceRows': rows,
    'removedRows': rows-len(keep),
    'remainingRows': len(keep),
    'activeAsset': OUT.name,
    'assetBytes': len(text.encode()),
    'removedCategories': removed_counts,
    'remainingCategories': {cat_names[i]: kept_counts[i] for i in range(len(cat_names)) if kept_counts[i]},
    'zeroedCategories': [cat_names[i] for i in range(len(cat_names)) if source_counts[i] and not kept_counts[i]],
    'maxParamHeight': obj['maxParamHeight'],
    'medianParamHeight': obj['medianParamHeight'],
    'comparisonPolicy': {
        'level4': 'all remaining rows; simple rational multipliers and simple exponent/log constants',
        'level5': 'all remaining rows; core rational multipliers and core exponent/log constants',
        'level6': 'all remaining rows; extended rational multipliers and extended exponent/log constants',
    }
}
STATS.write_text(json.dumps(stats, indent=2, ensure_ascii=False))
print(json.dumps(stats, indent=2, ensure_ascii=False))
