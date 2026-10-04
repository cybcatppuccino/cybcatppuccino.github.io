import json, os
ROOT=os.path.dirname(os.path.dirname(__file__))
D=os.path.join(ROOT,'data')
R=os.path.join(D,'runtime')
os.makedirs(R,exist_ok=True)

def load(rel, default=None):
    p=os.path.join(D,rel)
    if not os.path.exists(p): return default
    with open(p,encoding='utf8') as f:return json.load(f)

heesch={r['id']:r for r in load('known/heesch-n7-n10.json',{'records':[]}).get('records',[])}
complexity=load('known/complexity-index.json',{'records':{}}).get('records',{})
rect=load('known/rectangle-seed-index.json',{'records':{}}).get('records',{})
facts={r['id']:r for r in load('known/tiling-facts.json',{'records':[]}).get('records',[])}
show=load('known/complex-showcase-seeds.json',{'records':{}}).get('records',{})
summary=[]
for n in range(1,13):
    cat=load(f'catalog/free-holeless-n{n}.json')
    seeds=load(f'known/periodic-seeds-n{n}.json',{'seeds':{}}).get('seeds',{}) if n<=10 else {}
    out_shapes=[]
    for s in cat['shapes']:
        sid=s['id']; x={k:s[k] for k in ('id','c','w','h','p','o')}
        meta={}
        if sid in heesch: meta['h']=[heesch[sid]['Hc'],heesch[sid]['Hh']]
        if sid in complexity: meta['x']=complexity[sid]
        if sid in rect: meta['r']=rect[sid]
        if sid in facts: meta['f']=facts[sid].get('facts',{})
        if meta: x['m']=meta
        out_shapes.append(x)
    payload={'v':4,'n':n,'count':cat['count'],'shapes':out_shapes,'seeds':seeds}
    if n<=10:
        c={sid:rec for sid,rec in show.items() if sid.startswith(f'P{n}-')}
        if c:payload['complex']=c
    js='window.__TL_PACKS__=window.__TL_PACKS__||{};window.__TL_PACKS__[%d]=%s;\n'%(n,json.dumps(payload,separators=(',',':'),ensure_ascii=False))
    p=os.path.join(R,f'catalog-pack-n{n}.js')
    with open(p,'w',encoding='utf8') as f:f.write(js)
    summary.append((n,cat['count'],len(js),len(seeds),sum(len(v) for v in seeds.values())))
with open(os.path.join(R,'runtime-summary.json'),'w') as f:json.dump({'packs':[{'n':a,'count':b,'bytes':c,'seedShapes':d,'seedEntries':e} for a,b,c,d,e in summary]},f,indent=2)
print('\n'.join(map(str,summary)))
