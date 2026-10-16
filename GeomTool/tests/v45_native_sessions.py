"""v4.5 regression: original AG kernels, resumable stages and AG1 proof DAG.

Does NOT test Pyodide/WebAssembly parity; that requires a network-enabled browser.
"""
import json
import subprocess
import tempfile
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory() as tmp:
    a = Path(tmp) / 'ag1.json'
    b = Path(tmp) / 'constructors.json'
    subprocess.run(['node', str(ROOT/'tests/v44_make_ag1.cjs'), str(a)], check=True)
    subprocess.run(['node', str(ROOT/'tests/v45_construction_cases.cjs'), str(b)], check=True)
    ag1cases = json.loads(a.read_text())
    cases = json.loads(b.read_text())

    sys.path.insert(0, str(ROOT/'vendor/alphageometry2'))
    import geomtool_v45_session as ag2
    checked = 0
    for case in cases:
        if not case['valid']:
            continue
        sess = ag2.Session({'compiled': case['output'], 'queries': [], 'mode':'closure'})
        assert sess.progress()['engine'] == 'ag2'
        checked += 1
    assert checked == 36, checked
    print('PASS: AG2 accepts all 36 generated construction fixtures')

    # AG2 stage resumability, with two separate state observations.
    for case in cases:
        if case['op'] == 'midpoint':
            sess = ag2.Session({'compiled': case['output'], 'queries': [], 'mode':'closure'})
            a = sess.progress()['depth']
            b = sess.step()['depth']
            assert b >= a
            while not sess.done:
                sess.step()
            assert sess.progress()['done']
            break
    print('PASS: AG2 cooperative closure reaches terminal state')

    # Re-import AG1's ORIGINAL Python packages, exactly as the Pyodide switch.
    sys.path.insert(0, str(ROOT/'vendor/alphageometry1'))
    for module in ('ddar', 'numericals','geomtool_v45_session'):
        sys.modules.pop(module, None)
    import geomtool_v45_session as ag1
    case = ag1cases[0]  # A, B, C, two midpoints M,N; deduce MN // BC
    sess = ag1.Session({**case, 'mode':'goal', 'targetId':'mid:1'})
    assert sess.progress()['done'] is False
    for _ in range(10):
        if sess.done: break
        sess.step()
    assert sess.done and sess.goal_found, sess.progress()
    chain = sess.proof('mid:1')
    assert chain['available'] and len(chain['steps']) >= 3, chain
    assert any(step['rule'].startswith('r') and step['premises'] for step in chain['steps']), chain
    assert any(step['rule'] == 'AR' for step in chain['steps']), chain
    print('PASS: AG1 goal early stopping and numbered proof DAG with AR + theorem reference')

    # Switching back must retain AG2's own DDAR class (not AG1's import).
    assert ag2.Session({'compiled':cases[0]['output'], 'queries':[],'mode':'closure'}).progress()['engine']=='ag2'
    print('PASS: AG1 -> AG2 engine/module isolation')
print('v4.5 native session tests complete; browser runtime not exercised')
