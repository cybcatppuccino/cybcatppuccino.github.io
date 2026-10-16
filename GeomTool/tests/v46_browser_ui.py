"""Browser UI smoke test; Pyodide CDN unavailable in local regression sandbox."""
from pathlib import Path
from http.server import ThreadingHTTPServer,SimpleHTTPRequestHandler
from threading import Thread
from functools import partial
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
server=ThreadingHTTPServer(('127.0.0.1',0),partial(SimpleHTTPRequestHandler,directory=str(ROOT)))
Thread(target=server.serve_forever,daemon=True).start()
with sync_playwright() as p:
 browser=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=browser.new_page(viewport={'width':1440,'height':900})
 errors=[]
 page.on('pageerror',lambda e:errors.append(str(e)))
 try:
  page.goto('http://127.0.0.1:%d'%server.server_port,wait_until='domcontentloaded',timeout=30000)
 except Exception as exc:
  if 'BLOCKED_BY_ADMINISTRATOR' in str(exc):
   print('SKIP: browser local HTTP navigation blocked by sandbox security policy')
   browser.close();server.shutdown();raise SystemExit(0)
  raise
 page.wait_for_timeout(1300)
 assert page.locator('#acc-proof').count()==1
 assert page.locator('#ddar-engine').count()==1
 assert page.get_by_text('结论探知',exact=True).count()==1
 assert page.locator('#aux-run').count()==0
 assert page.locator('#split-analysis').count()==0
 assert 'v4.6' in page.title().lower(),page.title()
 page.locator('#ddar-discover-goals').click()
 page.wait_for_timeout(1300)
 print('UI title',page.title(),'goals status',page.locator('#ddar-goal-stats').inner_text(), 'errors', errors[:4])
 assert not errors,errors
 browser.close()
server.shutdown()
