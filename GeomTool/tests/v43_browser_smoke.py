"""End-to-end static HTML test: goal discovery, custom goal, DDAR proof, stale invalidation.
Requires Playwright and Chromium; no network or Python computation in the app.
"""
from pathlib import Path
import re
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
    page=browser.new_page(viewport={'width':1480,'height':960})
    errors=[]
    page.on('pageerror',lambda e: errors.append(str(e)))
    source=(ROOT/'index.html').read_text(encoding='utf8')
    source=re.sub(r'<link[^>]+href="assets/app.css"[^>]*>', '<style>'+ (ROOT/'assets/app.css').read_text(encoding='utf8') +'</style>',source)
    scripts=re.findall(r'<script src="([^"]+)"></script>',source)
    page.set_content(re.sub(r'<script src="[^"]+"></script>','',source),wait_until='domcontentloaded')
    for script in scripts:page.add_script_tag(content=(ROOT/script).read_text(encoding='utf8'))
    assert not errors,errors
    for cmd in ('point A 0 0','point B 4 0','point C 1 3','midpoint M A B','midpoint N A C'):
        page.evaluate('(s)=>GeomTool.runConsole(s)',cmd)
    page.locator('#acc-proof > button.accordion-trigger').click()
    page.locator('#ddar-discover-goals').click()
    page.wait_for_function('GeomTool.getGoals()?.goals?.length > 0',timeout=15000)
    assert '待证命题' in page.locator('#ddar-goal-stats').inner_text()
    page.locator('#ddar-goal-manual').fill('para M N B C')
    page.locator('#ddar-add-goal').click()
    assert page.locator('#ddar-goal-select').input_value()!=''
    page.locator('#ddar-prove-goal').click()
    page.wait_for_function('GeomTool.getGoalProof()?.status === "proved"',timeout=15000)
    assert '✓ DDAR 已证' in page.locator('#ddar-goal-result').inner_text()
    assert not errors,errors
    # Every diagram edit invalidates stale proofs and goal lists.
    page.evaluate('(s)=>GeomTool.runConsole(s)','point Z 6 5')
    assert page.evaluate('GeomTool.getGoalProof()') is None
    assert page.evaluate('GeomTool.getGoals()') is None
    page.locator('#ddar-goal-manual').fill('cong A B B C')
    page.locator('#ddar-add-goal').click()
    page.locator('#ddar-prove-goal').click()
    page.wait_for_function('GeomTool.getGoalProof()?.status === "unknown"',timeout=15000)
    assert '尚未证明' in page.locator('#ddar-goal-result').inner_text()
    assert not errors,errors
    print('PASS v4.3 browser UI: discovery, goal selection, proof check, invalidation, unproved safety')
    browser.close()
