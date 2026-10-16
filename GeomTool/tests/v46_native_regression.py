"""v4.6 both official AG implementations, proof provenance, generic sketches."""
import json, subprocess, sys, signal, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as directory:
    dest=Path(directory)/'fixtures.json'
    out=subprocess.run(['node',str(ROOT/'tests/v46_prepare_cases.cjs'),str(dest)],capture_output=True,text=True,check=True)
    fixtures=json.loads(dest.read_text())
    print(out.stdout.strip())

sys.path.insert(0,str(ROOT/'vendor/alphageometry1'))
import geomtool_v45_session as ag1
class Timed(Exception):pass
def timeout(*args):raise Timed('AG1 原生初始化超时')
signal.signal(signal.SIGALRM,timeout)
for row in fixtures:
    try:
        signal.alarm(3)
        sess=ag1.Session({'ag1':row['ag1'],'queries':[],'mode':'closure','maxLevel':1,'levelTimeout':1})
        assert sess.progress()['engine']=='ag1'
    finally:signal.alarm(0)
assert len(fixtures)==36
assert any(x['op']=='reflectLine' and any('通用数值示意' in v for v in x['limitations']) for x in fixtures)
print('PASS: 36 AG1 constructor imports, including coincident reflection generic sketch')

# Build an independent midline case including the always-present reference axes.
with tempfile.TemporaryDirectory() as directory:
    target=Path(directory)/'midline.json'
    subprocess.run(['node',str(ROOT/'tests/v46_make_midline.cjs'),str(target)],capture_output=True,check=True)
    sample=json.loads(target.read_text())

sess=ag1.Session(sample)
for i in range(12):
    if sess.done:break
    sess.step()
assert sess.goal_found
steps=sess.proof('midline')['steps']
assert any(step['rule']=='r06' and len(step['premises'])==2 for step in steps)
assert sum(step['rule']=='AR' for step in steps)>=2
assert sum(step['rule']=='构造 / midpoint' for step in steps)>=4
print('PASS: AG1 midpoint line r06 trace, construction/AR/DD provenance')

# AG2 needs its own Python modules in the same process, without AG1 module shadowing.
sys.path.insert(0,str(ROOT/'vendor/alphageometry2'))
for module in ('ddar','numericals','geomtool_v45_session'):
    sys.modules.pop(module,None)
import geomtool_v45_session as ag2
sess=ag2.Session(sample)
for i in range(30):
    if sess.done:break
    sess.step()
assert sess.goal_found
trace=sess.proof('midline')
assert trace['available'] and trace['verified'],trace
assert trace['stage']=='search_similar',trace
assert trace['premiseCount']==4,trace
assert any(x.get('sourceType')=='dd' or x.get('rule','').startswith('DD') for x in trace['steps']),trace
assert trace['steps'][-1]['predicate'].startswith('para '),trace
assert trace['steps'][-1]['premises'] and all(isinstance(i,int) for i in trace['steps'][-1]['premises']),trace
print('PASS: AG2 native replay certificate, four midpoint premises -> similar triangle search')
