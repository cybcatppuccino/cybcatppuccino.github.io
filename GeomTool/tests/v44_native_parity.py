"""Native checks of bundled ORIGINAL AG1/AG2 inference via GeomTool adapters.

These do not assert browser/native WASM parity (Pyodide not installed here).
Run: python tests/v44_native_parity.py
"""
from pathlib import Path
import subprocess
import sys
import json
import tempfile
import os
import importlib

ROOT = Path(__file__).resolve().parents[1]
node = ROOT / 'tests' / 'v44_make_cases.cjs'
node1 = ROOT / 'tests' / 'v44_make_ag1.cjs'

with tempfile.TemporaryDirectory() as tmp:
    a = Path(tmp) / 'ag2.json'; b = Path(tmp) / 'ag1.json'
    subprocess.run(['node', str(node), str(a)], check=True)
    subprocess.run(['node', str(node1), str(b)], check=True)
    cases2 = json.loads(a.read_text()); cases1 = json.loads(b.read_text())
    sys.path.insert(0, str(ROOT / 'vendor' / 'alphageometry2'))
    import geomtool_bridge as bridge2
    result2 = {c['name']:bridge2.execute(c) for c in cases2}
    expected2 = {
        'mid':['unknown','proved','proved'],
        'free':['unknown','unknown'],
        'foot':['proved','unknown'],
        'circle':['proved','proved'],
    }
    for name, expected in expected2.items():
        actual = [q['status'] for q in result2[name]['queries']]
        assert actual==expected,(name,actual,expected)
    print('PASS AG2 native GeomTool translations:',list(expected2))

    # Modules are isolated by captured Python imports; this explicitly checks
    # AG2 -> AG1 -> AG2 in the SAME interpreter.
    sys.path.insert(0, str(ROOT / 'vendor' / 'alphageometry1'))
    for module in ('ddar','numericals'):
        sys.modules.pop(module,None)
    import geomtool_ag1_bridge as bridge1
    result1 = {c['name']:bridge1.execute(c) for c in cases1}
    expected1 = {
        'mid':['proved','proved'],
        'free':['unknown'],
        'foot':['proved'],
        'circle':['proved'],
        'orthocenter':['proved'],
    }
    for name, expected in expected1.items():
        actual=[q['status'] for q in result1[name]['queries']]
        assert actual==expected,(name,actual,expected)
    assert [x['status'] for x in bridge2.execute(cases2[0])['queries']]==expected2['mid']
    print('PASS AG1 native GeomTool translations:',list(expected1))
    print('PASS AG2 -> AG1 -> AG2 module isolation')

# Upstream AG2 data test: its own official parsing and closure without any JS
# compiler; verifies the vendored mathematical source is executable.
sys.path.insert(0, str(ROOT / 'vendor' / 'alphageometry2'))
for module in ('ddar','numericals'):
    sys.modules.pop(module,None)
from ddar import DDAR
from parse import AGProblem
sample = (ROOT.parent / 'alphageometry2-main' / 'test.py')
if sample.exists():
    sys.path.insert(0, str(sample.parent))
    import test as upstream_cases
    for name in ['2003_p4','2007_p4']:
        p=AGProblem.parse(upstream_cases.problems_without_aux[name])
        engine=DDAR(p.points)
        for pred in p.preds:engine.force_pred(pred)
        engine.deduction_closure(progress_dot=False)
        assert engine.check_pred(p.goal),name
    print('PASS two AlphaGeometry2 published IMO problem fixtures')

print('v4.4 native checks complete (browser WASM parity not yet certified)')
