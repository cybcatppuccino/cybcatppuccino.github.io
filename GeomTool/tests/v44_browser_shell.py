"""Browser DOM + worker-request preparation smoke, NOT Pyodide parity."""
from pathlib import Path
import re
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1400,'height':850})
    errors=[]
    page.on('pageerror',lambda e:errors.append(str(e)))
    src=(ROOT/'index.html').read_text()
    src=re.sub(r'<link[^>]+href="assets/app.css"[^>]*>', '<style>'+(ROOT/'assets/app.css').read_text()+'</style>',src)
    scripts=re.findall(r'<script src="([^"]+)"></script>',src)
    page.set_content(re.sub(r'<script src="[^"]+"></script>','',src),wait_until='domcontentloaded')
    for script in scripts:page.add_script_tag(content=(ROOT/script).read_text())
    assert not errors,errors
    assert page.locator('#ddar-engine option').count()==2
    for cmd in ('point A 0 0','point B 4 0','point C 0 3','midpoint M A B','midpoint N A C'):
        page.evaluate('(s)=>GeomTool.runConsole(s)',cmd)
    prepared=page.evaluate('''()=>GeomDDAR.prepare(GeomTool.kernel.snapshot(),{
      enumerate:false,goal:{kind:'para',args:['P:B|C','P:M|N']}})''')
    assert prepared['queries'][0]['predicates'][0]['name']=='para',prepared
    assert prepared['compiled']['predicates'],prepared
    assert prepared['backend']=='ag2'
    page.locator('#acc-proof > button.accordion-trigger').click()
    page.locator('#ddar-discover-goals').click()
    page.wait_for_function('GeomTool.getGoals()?.goals?.length > 0',timeout=15000)
    page.locator('#ddar-goal-manual').fill('para M N B C')
    page.locator('#ddar-add-goal').click()
    assert page.locator('#ddar-goal-select').input_value()!=''
    page.locator('#ddar-engine').select_option('ag1')
    assert page.evaluate('GeomTool.getGoalProof()') is None
    a1=page.evaluate('''()=>GeomDDAR.prepare(GeomTool.kernel.snapshot(),{
      enumerate:false,engine:'ag1',goal:{kind:'para',args:['P:B|C','P:M|N']}})''')
    assert a1['ag1']['problem']
    assert a1['queries'][0]['ag1Goal'].startswith('para ')
    assert not errors,errors
    print('PASS v4.4 browser shell: both backends, unchanged goal discovery, manual goal, state invalidation')
    browser.close()
