"""Create a self-contained offline GeomTool v4.3 HTML from the source tree."""
from pathlib import Path
import re
root=Path(__file__).resolve().parents[1]
html=(root/'index.html').read_text(encoding='utf8')
html=re.sub(r'<link\b[^>]*href="assets/app.css"[^>]*>',lambda _: '<style>\n'+(root/'assets/app.css').read_text(encoding='utf8').replace('</style','<\\/style')+'\n</style>',html)
def embed(m):
    src=m.group(1)
    contents=(root/src).read_text(encoding='utf8').replace('</script','<\\/script')
    return f'<script>\n/* bundled: {src} */\n'+contents+'\n</script>'
html=re.sub(r'<script\s+src="([^"]+)"\s*></script>',embed,html)
assert 'src="assets/' not in html
output=root/'GeomToolv4.3-Standalone.html'
output.write_text(html,encoding='utf8')
print(output,output.stat().st_size)
