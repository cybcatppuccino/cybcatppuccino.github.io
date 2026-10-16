"""AG2 provenance smoke: logs have concrete numeric references, no rule rewrite."""
import json, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'/'alphageometry2'))
from geomtool_v45_session import Session
from parse import AGPredicate
import numpy as np
script=r'''
const {Kernel}=require('./assets/kernel.js'),cmp=require('./assets/ddar-compiler.js');
for(const kind of ['similarPoint','equalAngleLine']){
 const k=new Kernel();for(const [id,x,y] of [['A',0,0],['B',2,0],['C',1,2],['D',4,0],['E',6,1]])
  k.execute({op:'free',id,params:{x,y}});
 k.execute({op:kind,id:kind==='similarPoint'?'F':'l',inputs:['A','B','C','D','E']});
 console.log(JSON.stringify({kind,compiled:cmp.compile(k.snapshot())}));
}
'''
run=subprocess.run(['node','-e',script],cwd=ROOT,capture_output=True,text=True,check=True)
for case in map(json.loads,run.stdout.splitlines()):
    sess=Session({'compiled':case['compiled'],'queries':[],'mode':'closure'})
    assert len(sess.premise_rows)>0
    for row in sess.premise_rows:
        assert sess.engine.check_pred(sess._pred(row)),(case['kind'],row)
    assert all(e.get('premises') or e['type'] in ('INPUT','PREMISE') for e in sess.audit_index.values())
    sess.audit_file.close()
    print('AG2 native inputs:',case['kind'],len(sess.premise_rows),'accepted')
subprocess.run(['node',str(ROOT/'tests'/'v46_make_midline.cjs'),'/tmp/v52_native_mid.json'],cwd=ROOT,check=True,capture_output=True)
sess=Session(json.load(open('/tmp/v52_native_mid.json')))
for _ in range(50):
    if sess.done:break
    sess.step()
assert sess.goal_found
for row in sess.audit_index.values():
    if row['type'] not in ('INPUT','PREMISE'):
        assert row.get('premises'),row
        for ref in row['premises']:
            assert isinstance(ref,int) and ref in sess.audit_index and ref<row['id'],row
cert=sess.proof('midline')
assert cert['available'] and cert['verified']
assert any('similarity' in str(s['rule']) for s in cert['steps']),cert['steps']
print('AG2 midline with provenance PASS',len(cert['steps']),'steps',sess.audit_count,'events')
