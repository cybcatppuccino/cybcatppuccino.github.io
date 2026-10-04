import json, os, re, subprocess
ROOT=os.path.dirname(os.path.dirname(__file__))
expected={1:1,2:1,3:2,4:5,5:12,6:35,7:107,8:363,9:1248,10:4460,11:16094,12:58937}
total=0
for n,c in expected.items():
    d=json.load(open(os.path.join(ROOT,f'data/catalog/free-holeless-n{n}.json'),encoding='utf-8'))
    assert d['count']==c==len(d['shapes']), (n,d['count'],c); total+=c
h=json.load(open(os.path.join(ROOT,'data/known/heesch-n7-n10.json'),encoding='utf-8')); assert len(h['records'])==1611
p=json.load(open(os.path.join(ROOT,'data/known/periodic-index.json'),encoding='utf-8')); assert p['summary']['entries']==5109 and p['summary']['shapes']==1846
cx=json.load(open(os.path.join(ROOT,'data/known/complexity-index.json'),encoding='utf-8')); assert cx['topEntries']==600
for f in ['app.js','solver.worker.js','worker-source.js']:
    subprocess.run(['node','--check',os.path.join(ROOT,f)],check=True,stdout=subprocess.DEVNULL)
for n in range(1,13): subprocess.run(['node','--check',os.path.join(ROOT,f'data/runtime/catalog-pack-n{n}.js')],check=True,stdout=subprocess.DEVNULL)
html=open(os.path.join(ROOT,'index.html'),encoding='utf-8').read(); app=open(os.path.join(ROOT,'app.js'),encoding='utf-8').read(); solver=open(os.path.join(ROOT,'solver.worker.js'),encoding='utf-8').read(); embedded=open(os.path.join(ROOT,'worker-source.js'),encoding='utf-8').read().strip()
prefix='window.__TL_SOLVER_SOURCE__='; assert embedded.startswith(prefix) and embedded.endswith(';'); assert json.loads(embedded[len(prefix):-1])==solver
html_ids=set(re.findall(r'\bid=["\']([^"\']+)',html)); js_ids=set(re.findall(r"\$\(['\"]([^'\"]+)['\"]\)",app)); assert not (js_ids-html_ids), sorted(js_ids-html_ids)
assert '<html lang="en">' in html and 'β9' in html
assert "group: 'D4', mode: 'cell', algorithm: 'auto'" in app
assert re.findall(r'data-algorithm="([^"]+)"',html)==['auto','dense','constructive']
assert re.findall(r'data-mode="([^"]+)"',html)==['cell','companion']
assert 'id="cellPieces" type="number" min="8" max="20" step="1" value="10"' in html
assert 'id="cellPiecesDec"' in html and 'id="cellPiecesInc"' in html
for x in ['aggression','complexity','regularity','hueWheel','paletteStyles','dispersion','neighborBias','orbitSpan','luma','edge','cellEdge','materialStrength','tileMaterials','patternTransform','patternScale','patternWidth','patternOpacity','patternContrast','edgeLineStyle','edgeColorMode','voidMode','downloadBtn','focusBtn']: assert f'id="{x}"' in html,x
materials=re.findall(r'data-material="([^"]+)"',html); assert len(materials)==32 and len(set(materials))==len(materials)
styles=re.findall(r'data-style="([^"]+)"',html); assert len(styles)==30 and len(set(styles))==len(styles)
for x in ['rotateCCWBtn','rotateCWBtn','flipVBtn','flipHBtn','zoomOutBtn','zoomInBtn','resetViewBtn','downloadBtn','focusBtn','infoBtn']: assert f'id="{x}"' in html,x
assert 'id="fitBtn"' not in html and 'tilePatterns' not in html
assert 'solvePeriodicSpectrum' in solver and 'companionFromVoid' in solver and 'companionResidualPattern' in solver and 'mineSubstructureResults' in solver and 'latticeCandidatesForTarget' in solver
assert 'mode===\'companion\'' in solver and "mode==='growth'" not in solver
assert 'max="40"' not in html and 'max="30"' not in html and 'organic growth' not in html.lower()
assert '最小伴生块' not in html and '>minimal companion<' not in html
assert 'drawAxes(){/* Intentionally empty' in app and 'drawPeriodic' in app and 'safeColorPeriod' in app and 'twoColorVoltage' in app
assert 'searchMode' in app and 'searchGroup' in app and 'SvgCanvasContext' in app and 'exportCurrent' in app
print(f'OK: {total:,} shapes; {len(h["records"]):,} Heesch; {p["summary"]["entries"]:,} exact seeds; {cx["topEntries"]} complexity entries')
print(f'OK: β9 spectrum+companion modes; HNF lattice enumeration; UI period limit 8–20; {len(styles)} palette styles; {len(materials)} material choices; worker byte sync')
