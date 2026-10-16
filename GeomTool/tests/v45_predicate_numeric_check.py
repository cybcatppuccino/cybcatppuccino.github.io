"""Check that every emitted construction axiom holds in its own Kernel diagram.
Numerical checking is a compiler sanity test, NEVER an inference engine.
"""
import json,math,subprocess,tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as d:
    target=Path(d)/'cases.json'
    subprocess.run(['node',str(ROOT/'tests/v45_construction_cases.cjs'),str(target)],check=True)
    rows=json.loads(target.read_text())
errors=[];n=0

def good_angle(d):return abs((d+math.pi/2)%math.pi-math.pi/2)<1e-6
for c in rows:
 if not c.get('valid'):continue
 xy={p['name']:p['value'] for p in c['output']['points']}
 for x in c['output']['predicates']:
  ps=[xy[y] for y in x['points']];name=x['name'];n+=1
  def d(a,b):return math.dist(ps[a],ps[b])
  def a(i,j):return math.atan2(ps[j][1]-ps[i][1],ps[j][0]-ps[i][0])
  def cross(a0,b0,c0,d0):
   u=[ps[b0][k]-ps[a0][k] for k in [0,1]];v=[ps[d0][k]-ps[c0][k] for k in [0,1]];return u[0]*v[1]-u[1]*v[0]
  def dot(a0,b0,c0,d0):
   return sum((ps[b0][k]-ps[a0][k])*(ps[d0][k]-ps[c0][k]) for k in [0,1])
  ok=True
  if name=='coll':
   ok=abs(cross(0,1,0,2))<1e-6
  elif name=='cong':ok=abs(d(0,1)-d(2,3))<1e-6
  elif name=='rconst':ok=abs(d(0,1)-d(2,3)*x['constants'][0])<1e-6
  elif name=='para':ok=abs(cross(0,1,2,3))<1e-6
  elif name=='perp':ok=abs(dot(0,1,2,3))<1e-6
  elif name=='eqangle':ok=good_angle(a(0,1)-a(2,3)-a(4,5)+a(6,7))
  elif name=='aconst':ok=good_angle(a(0,1)-a(2,3)-x['constants'][0]*math.pi/180)
  elif name=='overlap':ok=d(0,1)<1e-6
  elif name=='cyclic_with_centers':
   z=x['constants'][0];ok=all(abs(d(0,i)-d(0,z))<1e-6 for i in range(z+1,len(ps)))
  elif name=='cyclic':
   if len(ps)>3:
    aa,bb,cc,dd=ps[:4]
    def row(x):return [x[0]**2+x[1]**2,x[0],x[1],1]
    import numpy as np
    ok=abs(float(np.linalg.det(np.array(list(map(row,(aa,bb,cc,dd)))))))<1e-5
  else:print('unhandled',name)
  if not ok:errors.append((c['op'],name,x['points'],x['constants'],x['source']))
print('CHECKED',n,'AXIOMS; ERRORS:',len(errors))
for e in errors[:40]:print(e)
assert not errors
