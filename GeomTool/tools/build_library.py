#!/usr/bin/env python3
"""GeomTool v4.7: deterministic, reproducible, loss-audited AG1/AG2/Newclid problem compiler.
Usage: python tools/build_library.py --ag1 ZIP --ag2 ZIP --newclid ZIP
Only problem datasets are indexed; source text and explicit coordinates are preserved.
"""
from __future__ import annotations
import argparse, ast, collections, hashlib, json, pathlib, re, zipfile
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT/'assets'/'library-data.js'
POINT = re.compile(r'^([A-Za-z][A-Za-z0-9_]*)@(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)_(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)$')
IDENT = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
# AG2's core is restricted; no invented AG2 predicate synonyms for unsupported ones.
AG2_PRED = {'coll','angeq','para','perp','s_angle','aconst','eqangle','distmeq','cong','eqratio','rconst','distseq','cyclic','cyclic_with_centers','overlap','acompute'}
entries=[]; rejects=[]

def record_point(token):
    m=POINT.fullmatch(token)
    if m:return m[1],[float(m[2]),float(m[3])]
    return token,None

def goal_rows(text):
    goals=[s.strip() for s in text.split(';') if s.strip()]
    # Convert names only if they truly differ between source formats. Retain literals verbatim.
    return [{'ag1':g,'ag2':g,'ag2Supported':g.split()[0] in AG2_PRED} for g in goals]

def key(source,name,problem):return hashlib.sha256((source+'\n'+name+'\n'+problem).encode()).hexdigest()[:16]

def high_problem(src,name,text,where):
    if '?' not in text:return False
    body,goals=text.split('?',1)
    if not goals.strip():return False
    base,sep,aux=body.partition('|')
    steps=[];coords={};error=[]
    for phase,sub in [('construction',base),('auxiliary',aux)]:
      for item in sub.split(';'):
        item=item.strip()
        if not item:continue
        if '=' not in item:
            error.append('malformed clause '+item[:60]);continue
        outs,rhs=item.split('=',1)
        ids=[]
        for t in outs.strip().split():
            p,xy=record_point(t)
            if not IDENT.fullmatch(p):error.append('bad point '+t);continue
            ids.append(p)
            if xy:coords[p]=xy
        constraints=[p.strip() for p in rhs.split(',') if p.strip()]
        operations=[]
        for con in constraints:
            toks=con.split()
            if not toks:continue
            op=toks[0];args=toks[1:]
            # Source archives are inconsistent: AG1 sometimes repeats output names
            # in RHS while Newclid omits them. Only strip exact complete prefix.
            if len(args)>=len(ids) and args[:len(ids)]==ids:
                args=args[len(ids):]
            operations.append({'op':op,'args':args,'raw':con})
        steps.append({'id':' '.join(ids),'outputs':ids,'operations':operations,'phase':phase,'raw':item})
    rows=goal_rows(goals)
    entries.append({'id':key(src,name,text),'source':src,'category':src,'title':name,'kind':'construction',
      'steps':steps,'coords':coords,'goal':rows[0]['ag1'],'goals':rows,
      'raw':text,'sourcePath':where,'parseIssues':error})
    return True

def low_problem(src,name,text,where,aux=False):
    if '?' not in text:return False
    body,goals=text.split('?',1)
    if not goals.strip():return False
    points=[];coords={};preds=[];errors=[]
    for clause in body.split(';'):
        if not clause.strip():continue
        if '=' not in clause:errors.append('Missing =');continue
        left,right=clause.split('=',1)
        for token in left.strip().split():
            p,xy=record_point(token)
            if not IDENT.fullmatch(p) or xy is None:errors.append('Invalid explicit-coordinate point: '+token);continue
            points.append(p);coords[p]=xy
        preds.extend([p.strip() for p in right.split(',') if p.strip()])
    rows=goal_rows(goals)
    entries.append({'id':key(src,name,text),'source':src,'category':src,'title':name,'kind':'coordinates',
      'steps':[], 'points':points,'coords':coords,'predicates':preds,
      'goal':rows[0]['ag1'],'goals':rows,'raw':text,'sourcePath':where,
      'hasAuxiliary':aux,'parseIssues':errors})
    return True

def process_pairs(z,inner,source):
    contents=z.read(inner).decode('utf-8-sig','replace').splitlines()
    lines=[x.strip() for x in contents if x.strip() and not x.lstrip().startswith('#')]
    i=0
    while i<len(lines):
        if '?' in lines[i]:
            rejects.append({'source':source,'path':inner,'reason':'missing title','line':i+1});i+=1;continue
        name=lines[i];i+=1
        if i>=len(lines):break
        if '?' not in lines[i]:
            # File may contain a prose preamble or a multi-line example.
            rejects.append({'source':source,'path':inner,'title':name,'reason':'not a problem line','line':i+1});continue
        problem=lines[i];i+=1
        if not high_problem(source,name,problem,inner):rejects.append({'source':source,'path':inner,'title':name,'reason':'parse failed'})

parser=argparse.ArgumentParser()
parser.add_argument('--ag1',required=True,type=pathlib.Path)
parser.add_argument('--ag2',required=True,type=pathlib.Path)
parser.add_argument('--newclid',required=True,type=pathlib.Path)
args=parser.parse_args()
with zipfile.ZipFile(args.ag1) as z:
    for path in ('examples.txt','jgex_ag_231.txt','imo_ag_30.txt'):
        inner='alphageometry-main/'+path
        process_pairs(z,inner,'AlphaGeometry 1')
with zipfile.ZipFile(args.newclid) as z:
    pths=sorted(p for p in z.namelist() if re.fullmatch(r'Newclid-main/newclid/problems_datasets/[^/]+\.txt',p))
    for inner in pths:process_pairs(z,inner,'Newclid')
with zipfile.ZipFile(args.ag2) as z:
    program=z.read('alphageometry2-main/test.py').decode('utf-8-sig')
    tree=ast.parse(program)
    for statement in tree.body:
      if not isinstance(statement,ast.Assign):continue
      if not any(isinstance(t,ast.Name) and t.id in ('problems_without_aux','problems_with_aux') for t in statement.targets):continue
      tag=next(t.id for t in statement.targets if isinstance(t,ast.Name) and t.id.startswith('problems_'))
      for name,text in ast.literal_eval(statement.value).items():
        if not low_problem('AlphaGeometry 2',name,text,'alphageometry2-main/test.py:'+tag,tag=='problems_with_aux'):
            rejects.append({'source':'AlphaGeometry 2','title':name,'reason':'failed low-level parse'})
# Do not mask source duplicates; they matter for source-complete inventory.
summary=dict(collections.Counter(x['source'] for x in entries))
obj={'schema':'geomtool.library/v4.7','summary':summary,'entries':entries,
 'inventory':{'sourceProblemCount':len(entries),'unparsed':rejects,
 'note':'AG2 includes original predicate constraints and exact sampled coordinates, not reconstructed unique constructions.'}}
OUT.write_text('/* Generated by tools/build_library.py; use the bundled source ZIPs to reproduce. */\nwindow.GeomLibraryData='+json.dumps(obj,ensure_ascii=False,separators=(',',':'))+';\n',encoding='utf-8')
(ROOT/'docs'/'V47_IMPORT_REPORT.json').write_text(json.dumps({'summary':summary,'count':len(entries),'unparsed':rejects},ensure_ascii=False,indent=2),encoding='utf-8')
print('v4.7 imported',len(entries),summary,'parse anomalies',len(rejects),'bytes',OUT.stat().st_size)
