import base64, json, re, struct, math
from pathlib import Path
src=Path('assets/ries-harddb-v11_4_1-filtered.js')
s=src.read_text()
m=re.search(r'window\.RIES_HARDDB_V114_DIRECT\s*=\s*(\{.*\})\s*;\s*\}\)\(\);',s,re.S)
d=json.loads(m.group(1))
rows=int(d['rows'])
values=list(struct.unpack('<%sd'%rows, base64.b64decode(d['valuesB64'])))
rowmap=base64.b64decode(d['rowMapB64'])
orig=list(struct.unpack('<%sI'%rows, base64.b64decode(d['origRowsB64'])))
param_bytes=base64.b64decode(d['paramB64'])
dict_vals=str(d['dict']).split('\t')
param_keys=d['paramKeys']
offsets=d['catParamOffsets']
cat_counts=[0]*len(d['categories'])
cat_indices={i:[] for i in range(len(d['categories']))}

def params_for(i):
    off=i*3; cid=rowmap[off]; ord_=rowmap[off+1] | (rowmap[off+2]<<8)
    keys=param_keys[cid]
    start=(int(offsets[cid]) + ord_*len(keys))*2
    p={}
    for kidx,k in enumerate(keys):
        code=param_bytes[start+2*kidx] | (param_bytes[start+2*kidx+1]<<8)
        p[k]=dict_vals[code] if code < len(dict_vals) else ''
    return cid,p

def height_of_params(p):
    h=1
    for v in p.values():
        t=str(v).strip().strip('"')
        for a,b in re.findall(r'(-?\d+)/(\d+)', t):
            h=max(h, abs(int(a)), abs(int(b)))
        # include standalone integer-ish values, avoid exponents from words only minimal
        for n in re.findall(r'(?<![A-Za-z])-?\d+(?![A-Za-z])', t):
            try: h=max(h, abs(int(n)))
            except Exception: pass
    return h
heights=[]
for i in range(rows):
    cid,p=params_for(i)
    h=height_of_params(p)
    heights.append(h)
    cat_indices[cid].append(i)

selected=set()
target=max(1, rows//5)
# per-category floor: select the lowest-height 20%, at least one, capped by category size
for cid,inds in cat_indices.items():
    if not inds: continue
    k=max(1, int(round(len(inds)*0.2)))
    inds_sorted=sorted(inds, key=lambda i:(heights[i], len(str(params_for(i)[1])), abs(values[i]), i))
    selected.update(inds_sorted[:k])
# adjust to exactly near target by global height if needed
if len(selected)<target:
    for i in sorted(range(rows), key=lambda i:(heights[i], abs(values[i]), i)):
        selected.add(i)
        if len(selected)>=target: break
elif len(selected)>target:
    # keep at least one per cat, remove highest height extras
    must=set()
    for cid,inds in cat_indices.items():
        sel=[i for i in inds if i in selected]
        if sel: must.add(min(sel, key=lambda i:(heights[i], abs(values[i]), i)))
    removable=sorted([i for i in selected if i not in must], key=lambda i:(-heights[i], -abs(values[i]), -i))
    for i in removable:
        if len(selected)<=target: break
        selected.remove(i)
low=sorted(selected, key=lambda i:(values[i], i))
high=sorted([i for i in range(rows) if i not in selected], key=lambda i:(values[i], i))
print(rows, len(low), len(high), 'categories in low', len({params_for(i)[0] for i in low}), 'of', len(d['categories']))

def b64_values(indices):
    return base64.b64encode(struct.pack('<%sd'%len(indices), *[values[i] for i in indices])).decode()
def b64_orig(indices):
    return base64.b64encode(struct.pack('<%sI'%len(indices), *[orig[i] for i in indices])).decode()
def b64_rowmap(indices):
    out=bytearray()
    for i in indices:
        off=i*3; out.extend(rowmap[off:off+3])
    return base64.b64encode(out).decode()
def cat_counts_for(indices):
    counts=[0]*len(d['categories'])
    for i in indices:
        counts[rowmap[i*3]]+=1
    return counts

def write_chunk(indices, stage, level, name):
    obj={
        'version':'11.6-harddb-split-20260606',
        'stage':stage,
        'level':level,
        'rows':len(indices),
        'sourceRows':d.get('sourceRows',420000),
        'selectionPolicy': ('level4 low-height representative 20% subset across all categories' if stage==1 else 'level5 remaining filtered harddb rows'),
        'maxParamHeight': max(heights[i] for i in indices) if indices else 0,
        'medianParamHeight': sorted(heights[i] for i in indices)[len(indices)//2] if indices else 0,
        'categories':d['categories'],
        'paramKeys':d['paramKeys'],
        'catCounts':cat_counts_for(indices),
        'catParamOffsets':d['catParamOffsets'],
        'dict':d['dict'],
        'valuesB64':b64_values(indices),
        'rowMapB64':b64_rowmap(indices),
        'origRowsB64':b64_orig(indices),
        'paramB64':d['paramB64'],
        'valueBytes':len(indices)*8,
        'rowMapBytes':len(indices)*3,
        'origRowBytes':len(indices)*4,
        'paramBytes':d['paramBytes']
    }
    text='// RIES v11.6 split filtered hard-constant database asset.\n(function(){\n  window.RIES_HARDDB_V116_CHUNKS = window.RIES_HARDDB_V116_CHUNKS || [];\n  window.RIES_HARDDB_V116_CHUNKS[%d] = %s;\n})();\n' % (stage-1, json.dumps(obj,separators=(',',':')))
    Path(name).write_text(text)
    return len(text.encode())
size4=write_chunk(low,1,4,'assets/ries-harddb-v11_6-level4.js')
size5=write_chunk(high,2,5,'assets/ries-harddb-v11_6-level5.js')
stats={
    'version':'11.6-harddb-split-20260606','sourceAsset':src.name,'sourceRows':rows,
    'level4Rows':len(low),'level5AdditionalRows':len(high),'level4AssetBytes':size4,'level5AssetBytes':size5,
    'level4MaxHeight':max(heights[i] for i in low),'level4MedianHeight':sorted(heights[i] for i in low)[len(low)//2],
    'level5MaxHeight':max(heights[i] for i in high),'level5MedianHeight':sorted(heights[i] for i in high)[len(high)//2],
    'level4CategoriesCovered':len({params_for(i)[0] for i in low}),'totalCategories':len(d['categories'])
}
Path('assets/ries-harddb-v11_6-stats.json').write_text(json.dumps(stats,indent=2))
print(stats)
