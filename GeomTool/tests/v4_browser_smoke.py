from playwright.sync_api import sync_playwright
from pathlib import Path
import re,math
root=Path(__file__).resolve().parents[1]
with sync_playwright() as p:
 b=p.chromium.launch(headless=True,executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 page=b.new_page(viewport={'width':1440,'height':960})
 errs=[];page.on('pageerror',lambda e:errs.append(str(e)))
 html=(root/'index.html').read_text()
 html=re.sub(r'<link[^>]+href="assets/app.css"[^>]*>','<style>'+ (root/'assets/app.css').read_text()+'</style>', html)
 scripts=re.findall(r'<script src="([^"]+)"></script>',html)
 page.set_content(re.sub(r'<script src="[^"]+"></script>','',html),wait_until='load')
 for script in scripts:page.add_script_tag(content=(root/script).read_text())
 assert not errs,errs
 c=lambda s:page.evaluate('(s)=>GeomTool.runConsole(s)',s)
 assert page.locator('.version').inner_text()=='v4'
 assert page.evaluate('GeomTool.kernel.node("p_0").label')=='p_0'
 assert page.evaluate('GeomTool.kernel.node("l_Re").style.system')
 assert len(page.locator('#object-area [data-id]').all())==6
 c('point A -2 -1');c('point B 2 -1');c('point C 0 2');
 # Last (third) point of the circumcircle: preview uses a smart snapped point.
 c('tool circumcircle');c('step A');c('step B')
 box=page.locator('#board').bounding_box()
 assert box and box['width']>300
 screen=lambda x,y:(box['x']+box['width']/2+88*x,box['y']+box['height']/2-88*y)
 x,y=screen(0,2);page.mouse.move(x,y)
 preview=page.evaluate('GeomTool.getPreview()')
 assert preview and preview['kind']=='circle' and preview['r']>2,preview
 ghost=page.evaluate('GeomTool.getGhost()');assert ghost and ghost['kind']=='existing' and ghost['existing']['id']=='C',ghost
 # Real resulting circle should be identical to the preview.
 expected_radius=preview['r'];page.mouse.click(x,y)
 circles=page.evaluate('GeomTool.kernel.list().filter(n=>n.op==="circumcircle")')
 assert len(circles)==1 and abs(circles[0]['geom']['r']-expected_radius)<1e-8
 # Polygon: click first vertex to close. Produces three actual segments.
 c('tool polygon');c('step A');c('step B');c('step C');page.mouse.move(*screen(-2,-1))
 assert page.evaluate('GeomTool.getPreview()?.kind')=='polygon'
 page.mouse.click(*screen(-2,-1))
 polys=page.evaluate('GeomTool.kernel.list().filter(n=>n.op==="polygon")')
 assert len(polys)==1,polys
 edges=page.evaluate('GeomTool.kernel.list().filter(n=>n.params.polygonParent===GeomTool.kernel.list().find(p=>p.op==="polygon").id)')
 assert len(edges)==3 and all(e['geom']['extent']=='segment' for e in edges),edges
 edge_id=edges[0]['id']
 c('tool midpoint');c('step A');c('step B')
 assert page.evaluate('GeomTool.kernel.node("D")?.op')=='midpoint'
 c('hide '+edge_id);assert page.evaluate('(x)=>GeomTool.kernel.node(x).style.visible',edge_id)==False
 c('show '+edge_id)
 c('rename A Z');assert page.evaluate('GeomTool.kernel.node("A")') is None
 assert page.evaluate('GeomTool.kernel.node("Z").label')=='Z'
 assert page.evaluate('GeomTool.kernel.node("sZB")?.inputs.includes("Z")')
 c('help');assert page.locator('#console-help-modal').is_visible()
 page.locator('#console-help-search').fill('polygon')
 assert int(page.locator('#console-help-count').inner_text().split('/')[0])>0
 assert not errs,errs
 print('PASS v4 browser: references, circle prediction, smart point snap, polygon close and edges, rename, hidden segment, help')
 b.close()
