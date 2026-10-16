"""v5.1 integration smoke: source predicates, native closure and audit."""
import json, os, subprocess, sys, tempfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor/alphageometry2'))
from geomtool_v45_session import Session
with tempfile.TemporaryDirectory() as td:
    dest=Path(td)/'cases.json'
    subprocess.run(['node',str(ROOT/'tests/v51_compilation.cjs'),str(dest)],check=True,
                   stdout=subprocess.DEVNULL)
    cases=json.loads(dest.read_text())
assert len(cases)==36
for case in cases:
    sess=Session({'compiled':case['compiled'],'queries':[],'mode':'closure'})
    assert sess.audit_count>=1
    assert any(e['type']=='PREMISE' for e in sess.audit_index.values()) if case['compiled']['predicates'] else sess.audit_count==1
    assert json.loads(sess.log_text().splitlines()[0])['type']=='INPUT'
    sess.audit_file.close()
# Reuse the old canonical midline fixture to guard native proof and logged DD.
with tempfile.TemporaryDirectory() as td:
    dest=Path(td)/'midline.json'
    subprocess.run(['node',str(ROOT/'tests/v46_make_midline.cjs'),str(dest)],check=True,
                   stdout=subprocess.DEVNULL)
    payload=json.loads(dest.read_text())
sess=Session(payload)
for _ in range(40):
    if sess.done:break
    sess.step()
assert sess.goal_found
assert sess.audit_count>1
records=[json.loads(line) for line in sess.log_text().splitlines()]
assert records[0]['type']=='INPUT'
assert any(row['type']=='AR' for row in records)
assert any(row['type']=='DD' and 'similar triangles' in row['rule'] for row in records)
assert any(row['queryId']=='midline' for row in records if row.get('type')=='DDAR')
cert=sess.proof('midline')
assert cert['available'] and cert['verified']
assert cert['events']
print('v5.1 PASS: 36 AG2 initializations; midline proof; DD + AR + query evidence JSONL')
