# Optional integration smoke test. Requires `pip install playwright` and local Chromium.
# Uses set_content + script injection to work even in environments blocking file:// navigation.
from playwright.sync_api import sync_playwright
from pathlib import Path
import re,math
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':960})
 errs=[];page.on('pageerror',lambda x:errs.append(str(x)))
 html=(root/'index.html').read_text()
 html=re.sub(r'<link[^>]+href="assets/app.css"[^>]*>', '<style>'+ (root/'assets/app.css').read_text()+'</style>', html)
 scripts=re.findall(r'<script src="([^"]+)"></script>',html)
 page.set_content(re.sub(r'<script src="[^"]+"></script>','',html),wait_until='load')
 for src in scripts:page.add_script_tag(content=(root/src).read_text())
 c=lambda x:page.evaluate('(s)=>window.GeomTool.runConsole(s)',x)
 assert not errs,errs
 assert page.locator('.version').inner_text()=='v3.5'
 assert len(page.locator('#object-area [data-id]').all())==6
 c('point A 2 0');c('point O 0 0');c('tool rotate')
 assert page.locator('#param-value').count()==1
 # config is inside selected tool card, not at distant bottom
 assert page.locator('.tool-card.active #param-value').count()==1
 page.locator('#param-angle-unit').select_option('rad')
 assert not page.locator('#param-pi').is_hidden()
 page.locator('#param-value').fill('2/3')
 page.locator('#param-value').press('Home');page.locator('#param-pi').click()
 assert page.locator('#param-value').input_value()=='π2/3'
 page.locator('#param-value').fill('2π/3')
 assert math.isclose(page.evaluate('GeomExpression.parse(document.querySelector("#param-value").value)'),math.pi*2/3)
 page.locator('#param-angle-unit').select_option('deg')
 assert page.locator('#param-pi').is_hidden()
 page.locator('#param-angle-unit').select_option('rad')
 assert page.locator('#param-value').input_value()=='2π/3'
 c('step A');c('step O')
 n=page.evaluate('window.GeomTool.kernel.list().filter(n=>n.op==="rotate").at(-1).geom')
 assert math.isclose(n['x'],-1,abs_tol=1e-6) and math.isclose(n['y'],math.sqrt(3),abs_tol=1e-6),n
 c('hide A');page.locator('.obj-row[data-id="A"]').click()
 assert page.locator('.details-area #rename-field').input_value()=='A'
 assert page.locator('#visibility-btn').inner_text()=='显示'
 assert page.locator('.obj-row[data-id="A"]').count()==1
 # toggle via properties while hidden
 page.locator('#visibility-btn').click()
 assert page.evaluate('window.GeomTool.kernel.node("A").style.visible')==True
 c('settings axes on');c('settings complex on');c('hide GTREF_origin');c('hide GTREF_real')
 c('select GTREF_real')
 assert page.locator('#visibility-btn').is_enabled()
 assert page.locator('#rename-btn').is_disabled()
 assert page.locator('#delete-btn').is_disabled()
 c('settings axes off')
 assert not page.evaluate('window.GeomTool.settings.complex')
 assert page.locator('.obj-row[data-id="GTREF_real"]').count()==1
 assert page.locator('#rename-field').input_value()=='Re'
 # each independently visible when axes restored
 c('show GTREF_origin');c('settings axes on')
 assert page.evaluate('window.GeomTool.kernel.node("GTREF_origin").style.visible')
 assert not page.evaluate('window.GeomTool.kernel.node("GTREF_real").style.visible')
 c('help'); assert page.locator('#console-help-modal').is_visible()
 assert page.locator('.help-category').count()>=10
 page.locator('#console-help-search').fill('位似')
 assert page.locator('#console-help-count').inner_text().split('/')[0].strip()!='0'
 assert page.locator('.help-category').first.get_attribute('open') is not None
 page.locator('#console-help-close').click()
 c('settings labels off');assert not page.evaluate('window.GeomTool.settings.labels')
 c('settings precision 10');assert page.evaluate('window.GeomTool.settings.precision')==10
 c('relations kinds all');assert page.locator('#radar-options input:checked').count()>=40
 c('relations kinds none');assert page.locator('#radar-options input:checked').count()==0
 c('relations kinds parallel on'); assert page.locator('#radar-options input:checked').count()==1
 c('library search Euler');assert int(page.locator('#library-result-count').inner_text().split()[0])>=1
 c('panel pretty');c('pretty strength 50');assert page.locator('#pretty-strength').input_value()=='50'
 c('export json'); data=c('export json'); assert 'geomtool.document/v1' in data
 c('import json '+data.replace('\n',''));assert page.evaluate('window.GeomTool.kernel.node("A")') is not None
 c('A');assert page.locator('#rename-field').input_value()=='A'
 page.locator('#console-toggle').click()
 page.locator('#console-input').fill('beaut')
 assert page.locator('#console-suggestions').is_visible()
 page.locator('#console-input').fill('help')
 page.locator('#console-input').press('Enter')
 assert page.locator('#console-help-modal').is_visible()
 assert not errs,errs
 print('PASS v3.5 browser: transform rad+π, hidden/system selection, settings, searchable help, relations, library, import/export; help entries:',len(page.locator('.help-entry').all()))
 b.close()
