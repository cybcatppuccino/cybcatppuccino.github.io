/* GeomTool v4 — static view/controller. All edits go through GeomKernel.Kernel. */
(()=>{'use strict';
const {Kernel,METHODS,runText,previewConstruction,intersections,intersectionBranches,dist,project}=window.GeomKernel;
const Pretty=window.GeomPretty, ToolGuides=window.GeomToolGuides;
const $=s=>document.querySelector(s), $$=s=>[...document.querySelectorAll(s)];
const ICON={smartPoint:'point',select:'cursor',pan:'cursor',free:'point',real:'point',onCarrier:'point',line:'line',segment:'line',ray:'line',circle:'circle',circleRadius:'circle',circumcircle:'circle',intersection:'target',midpoint:'point',perpendicular:'line',parallel:'line',perpBisector:'line',angleBisector:'construct',foot:'point',reflectPoint:'target',reflectLine:'target',translate:'target',rotate:'target',homothety:'target',centroid:'point',circumcenter:'point',orthocenter:'point',incenter:'point',tangent:'line',tangentFrom:'line',polygon:'construct',similarPoint:'target',equalAngleLine:'line'};
const groups={
 select:{title:'选择工具',subtitle:'智能选中、拖点与平移背景',tools:[['select','选择 / 移动','点按选中 · 空白拖动平移']]},
 point:{title:'点',subtitle:'基础点与精确定位',tools:[['smartPoint','智能点','自由 · 约束 · 交点，一次点击'],['midpoint','中点','选择两个点'],['foot','垂足','点 → 直线'],['centroid','重心','三角形三个顶点'],['circumcenter','外心','三个顶点'],['incenter','内心','三个顶点'],['orthocenter','垂心','三个顶点'],['excenter','旁心','顶点 A → B → C'],['parallelogram','平行四边形','已知 A、B、C'],['similarPoint','相似点','A → B → C → D → E，生成 F']]},
 line:{title:'线与射线',subtitle:'直线、垂线和角度构造',tools:[['line','直线','通过两点'],['segment','线段','两个端点'],['ray','射线','起点 → 方向点'],['parallel','平行线','点 → 直线'],['perpendicular','垂线','点 → 直线'],['perpBisector','中垂线','选择两点'],['angleBisector','角平分线','A → 顶点 B → C'],['tangent','圆上切线','点 → 圆'],['tangentFrom','外点切线','点 → 圆'],['parallel3','三点平行','P → A → B'],['perpendicular3','三点垂线','P → A → B'],['outerAngleBisector','外角平分线','A → 顶点 B → C'],['equalAngleLine','等角线','A → B → C → D → E']]},
 circle:{title:'圆与多边形',subtitle:'圆规与常用曲线',tools:[['circle','圆','圆心 → 过点'],['circleRadius','定半径圆','圆心 + 半径'],['circumcircle','三点圆','经过三个点'],['polygon','多边形','点首顶点闭合 / 双击结束'],['circleBySegment','线段作半径','圆心 → A → B'],['diameterCircle','直径圆','直径两个端点'],['equilateral','等边顶点','选择两端点']]},
 construct:{title:'几何构造',subtitle:'证明器常用辅助构造',tools:[['midpoint','中点','AB 中点'],['foot','垂足','过点向直线'],['perpBisector','中垂线','线段的中垂线'],['angleBisector','角平分线','A → 顶点 B → C'],['circumcenter','外心','三顶点'],['orthocenter','垂心','三顶点'],['incenter','内心','三顶点'],['centroid','重心','三顶点'],['foot3','三点垂足','P → A → B'],['similarPoint','相似点','A → B → C → D → E，生成 F']]},
 transform:{title:'几何变换',subtitle:'对称、平移、旋转与位似',tools:[['reflectPoint','中心对称','原点 → 对称中心'],['reflectLine','轴对称','点 → 对称轴'],['translate','平移点','P → A → B'],['rotate','旋转点','点 → 中心'],['homothety','位似点','点 → 中心'],['rotateLeft','左旋 90°','点 → 中心'],['rotateRight','右旋 90°','点 → 中心'],['parallelogram','平行四边形','A → B → C']]},
};
const toolToOp={onCarrier:null,select:null,pan:null};
const namedKinds={point:'点',line:'直线',circle:'圆',polygon:'多边形'};
const RelationEngine=window.GeomRelationEngine, RelationOverlay=window.GeomRelationOverlay;
const relationLabels=Object.fromEntries(RelationEngine.registry.map(x=>[x.id,x.label]));
const settingsDefault={grid:false,labels:true,axes:false,snap:true,wheel:true,precision:3,palette:'blue',complex:false,theme:'light',leftWidth:228,rightWidth:312,leftCollapsed:false,rightCollapsed:false,ddarSplit:false};
let settings={...settingsDefault};try{settings={...settings,...JSON.parse(localStorage.getItem('geomtool.settings')||'{}')}}catch(e){}
settings.precision=Number.isInteger(Number(settings.precision))?Math.max(0,Math.min(15,Number(settings.precision))):3;
let kernel=new Kernel(),activeGroup='select',activeTool='select',steps=[],selected=null,hovered=null,mouse={x:0,y:0},drag=null,spaceDown=false;
let raster=null,ctx=$('#board').getContext('2d'),view={ox:0,oy:0,scale:88},W=800,H=600,dpr=1;
let radarQuadraticOptions={quadraticProductLimit:66,quadraticExpressionLimit:20};
let radarResult=null,radarEntry=null,radarPage=1,radarFiltered=[],objTab='objects',proofFacts=[],clipboardSupported=true,consoleHistory=[],historyIndex=0,saveJob=0,paintPending=false,lastCanvasCenter=true;
let bgCacheDirty=true,expandedGuide=null,prettyPlan=null,prettyPending=null,prettyConditions=[],ghost=null,hoverCarriers=[],libraryCurrent=null,libraryVisualId=null,libraryVisualMarks=[],proofOverlay=null,toolPreview=null;
const transformPrefs={rotate:{unit:'deg',deg:'60',rad:'π/3'},homothety:{factor:'2'}};
const fmt=n=>!Number.isFinite(n)?'—':(Math.abs(n)<1e-10?'0':n.toFixed(settings.precision).replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1'));
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function svg(name){return `<svg><use href="#i-${name}"/></svg>`}
function notice(text,error=false){const host=$('#toasts'),el=document.createElement('div');el.className='toast'+(error?' error':'');el.textContent=text;host.appendChild(el);setTimeout(()=>el.remove(),2600)}
function setHint(s){$('#task-tip').textContent=s;$('#board-instruction').textContent=s}
function command(c,silent=false){try{const v=kernel.execute(c);if(!silent)notice(c.op==='delete'?'已删除相关对象':c.op==='clear'?'已清空画板':'构造成功');return v}catch(e){notice(e.message,true);return null}}
function save(){try{localStorage.setItem('geomtool.document',JSON.stringify({name:$('#doc-name').value,graph:kernel.serialize(),libraryVisualId,prettyConditions:prettyConditions.map(c=>c.text)}));localStorage.setItem('geomtool.settings',JSON.stringify(settings))}catch(e){$('#save-state').textContent='存储不可用';}}
function requestSave(){clearTimeout(saveJob);saveJob=setTimeout(save,180)}
function effectiveAxes(){return !!(settings.axes||settings.complex);}
function invalidateRelationVisibility(){radarEntry=null;radarResult=null;radarPage=1;renderRadar();}
function initSettings(){
 for(const [key,id] of Object.entries({grid:'setting-grid',labels:'setting-labels',snap:'setting-snap',wheel:'setting-wheel'})){
  const input=$('#'+id);input.checked=!!settings[key];input.onchange=()=>{settings[key]=input.checked;requestSave();render()};
 }
 // Axis preference is remembered independently of the complex reference.
 // Turning complex on requests axes temporarily; turning it off restores the preference.
 // Turning the visible axes switch OFF while complex is on turns both off.
 $('#complex-toggle').onchange=e=>{settings.complex=e.target.checked;syncPalette();invalidateRelationVisibility();requestSave();render()};
 $('#setting-axes').onchange=e=>{settings.axes=e.target.checked;if(!settings.axes&&settings.complex)settings.complex=false;syncPalette();invalidateRelationVisibility();requestSave();render()};
 $('#setting-precision').value=String(settings.precision);
 $('#setting-precision').onchange=e=>{settings.precision=Number(e.target.value);requestSave();render()};
 $('#setting-palette').value=settings.palette;$('#setting-palette').onchange=e=>{settings.palette=e.target.value;syncPalette();requestSave();draw()};syncPalette();
}
function syncPalette(){document.body.dataset.palette=settings.palette;document.body.dataset.theme=settings.theme==='dark'?'dark':'light';$('#complex-toggle').checked=!!settings.complex;$('#setting-axes').checked=effectiveAxes();$('#theme-symbol').textContent=settings.theme==='dark'?'☀':'☾';$('#theme-text').textContent=settings.theme==='dark'?'日间':'夜间';$('#theme-toggle').title=settings.theme==='dark'?'切换至白天模式':'切换至黑夜模式';}
const refs=[
 {id:'p_0',op:'referencePoint',params:{x:0,y:0}},
 {id:'p_1',op:'referencePoint',params:{x:1,y:0}},
 {id:'p_i',op:'referencePoint',params:{x:0,y:1}},
 {id:'l_Re',op:'line',inputs:['p_0','p_1']},
 {id:'l_Im',op:'line',inputs:['p_0','p_i']},
 {id:'c_unit',op:'circleRadius',inputs:['p_0'],params:{radius:1}}
];
function ensureReferences(){for(const r of refs){if(kernel.node(r.id))continue;try{kernel._create({...r,system:true,style:{system:true,visible:true,reference:true,dashed:r.id==='c_unit'}})}catch(e){console.warn('reference:',r.id,e);}}}
function referenceVisible(n){if(!n?.style.system)return true;
 if(!effectiveAxes())return false;
 if(['c_unit','p_1','p_i'].includes(n.id))return !!settings.complex;
 return true;
}
function isVisible(n){return !!(n?.valid&&n.style.visible&&referenceVisible(n));}
// Keep even invisible reference nodes in the object list so they can be managed individually.
function exposedNodes(){return kernel.list();}
function invalidatePretty(){prettyPlan=null;$('#pretty-progress').disabled=true;$('#pretty-progress').value=0;$('#pretty-progress-label').textContent='0%';$('#pretty-status').textContent='图形已更新，点击“计算 Pretty”重新优化。';}

function restore(){try{const s=JSON.parse(localStorage.getItem('geomtool.document')||'null');if(s?.graph?.schema==='geomtool.ir/v1'){kernel.load(s.graph,false);$('#doc-name').value=s.name||'未命名画板';setLibraryVisuals(s.libraryVisualId);prettyConditions=(s.prettyConditions||[]).flatMap(raw=>{try{return [window.GeomPrettyConstraints.parse(raw,kernel)]}catch(e){console.warn('Condition restore:',raw,e.message);return []}});renderPrettyConditions();return true}}catch(e){console.warn('Failed to restore document:',e);}return false}
function setLibraryVisuals(id){
 libraryVisualId=null;libraryVisualMarks=[];
 const e=window.GeomLibraryData?.entries?.find(x=>x.id===id);
 if(e&&window.GeomLibraryVisuals){libraryVisualId=e.id;libraryVisualMarks=window.GeomLibraryVisuals.compile(e);}
}
function drawLibraryVisuals(){
 if(!libraryVisualMarks.length||!window.GeomLibraryVisuals)return;
 window.GeomLibraryVisuals.draw(ctx,view,libraryVisualMarks,id=>kernel.geom(id),settings.theme);
}
function seed(){const cmds=[['free','A',[],{x:-2.35,y:-1.55}],['free','B',[],{x:2.45,y:-1.35}],['free','C',[],{x:.45,y:1.9}],['segment','s1',['A','B'],{}],['segment','s2',['B','C'],{}],['segment','s3',['C','A'],{}],['midpoint','M',['A','B'],{}],['line','l1',['A','B'],{}],['foot','H',['C','l1'],{}],['segment','s4',['C','H'],{}],['circumcircle','c1',['A','B','C'],{}]];
 cmds.forEach(([op,id,inputs,params])=>kernel.execute({op,id,inputs,params,style:{dashed:['l1','s4'].includes(id),visible:id!=='l1'}}));kernel.undoStack=[];kernel.redoStack=[];
}
function groupUI(group){activeGroup=group;const def=groups[group],root=$('#tool-list'),offset=root.scrollTop;$('#tool-group-title').textContent=def.title;$('#tool-group-subtitle').textContent=def.subtitle;$$('.rail-btn[data-group]').forEach(b=>b.classList.toggle('active',b.dataset.group===group));
 root.innerHTML=def.tools.map(([id,name,sub])=>{const g=ToolGuides.guide(id);return `<div class="tool-card ${activeTool===id?'active':''}"><div class="tool-card-row"><button class="tool-item ${activeTool===id?'active':''}" data-tool="${id}" title="${esc(name)}"><span class="tool-glyph">${ToolGuides.sketch(id)}</span><span class="tool-txt"><strong>${esc(name)}</strong><small class="tool-mini">${esc(sub)}</small></span></button><button class="tool-help ${expandedGuide===id?'active':''}" data-help="${id}" title="查看${esc(name)}操作说明" aria-label="查看${esc(name)}操作说明" aria-expanded="${expandedGuide===id}">?</button></div>${expandedGuide===id?`<div class="tool-help-body">${g.svg}<ol>${g.steps.map(t=>`<li>${esc(t)}</li>`).join('')}</ol><p><b>效果</b> ${esc(g.effect)}</p></div>`:''}${activeTool===id?'<div id="tool-config"></div>':''}</div>`;}).join('');
 $$('[data-tool]').forEach(b=>b.addEventListener('click',()=>activate(b.dataset.tool)));
 $$('[data-help]').forEach(b=>b.addEventListener('click',()=>{expandedGuide=expandedGuide===b.dataset.help?null:b.dataset.help;groupUI(activeGroup)}));
 root.scrollTop=offset;renderToolConfig();}

function activate(tool){if(['intersection','secondIntersection'].includes(tool))throw Error('该工具已由智能点替代');if(tool==='select')activeGroup='select';activeTool=tool;steps=[];hovered=null;ghost=null;toolPreview=null;$('#active-tool-label').textContent=toolName(tool);if(activeGroup==='select'&&tool!=='select'&&tool!=='pan'&&tool!=='onCarrier'){const match=Object.keys(groups).find(g=>groups[g].tools.some(i=>i[0]===tool));if(match)activeGroup=match;}
 groupUI(activeGroup);if(window.innerWidth<=620)$('#tools-panel').classList.remove('mobile-tools-show');setHint(tool==='select'?'点击对象选中 · 拖动活动点 · 拖动空白平移':tool==='smartPoint'?'单击空白/曲线/交会处，一次放置点':tool==='intersection'?'点击交点放一个，或依次选择两个对象创建所有交点':'选择 '+(METHODS[tool]?.inputs?.map(x=>x==='point'?'点':x==='line'?'直线':x==='circle'?'圆':'线或圆').join(' → ')|| (tool==='free'?'画布任意位置':tool==='onCarrier'?'一条直线或圆':tool==='pan'?'拖动画布':'所需对象')));draw()}
function toolName(tool){for(const g of Object.values(groups))for(const t of g.tools)if(t[0]===tool)return t[1];return tool}
function renderToolConfig(){const el=$('#tool-config');if(!el)return;
 if(activeTool==='rotate'||activeTool==='homothety'){
  const rotation=activeTool==='rotate',prefs=transformPrefs[activeTool];
  el.innerHTML=`<div class="tool-config"><label for="param-value">${rotation?'旋转角度':'位似倍数'}</label><div class="param-input-row"><input id="param-value" type="text" value="${esc(rotation?prefs[prefs.unit]:prefs.factor)}" autocomplete="off" spellcheck="false" aria-label="${rotation?'旋转角度':'位似倍数'}" />${rotation?'<button type="button" id="param-pi" class="param-pi" aria-label="输入 π" title="在光标处插入 π">π</button>':''}</div>${rotation?'<label class="param-unit-label" for="param-angle-unit">角度单位</label><select id="param-angle-unit" aria-label="角度单位"><option value="deg">角度 (°)</option><option value="rad">弧度 (rad)</option></select>':''}<small class="param-help">${rotation?'弧度支持 π、pi、2π/3、2pi/3；角度支持 60、180/3。':'支持 2、-sqrt(2)、5/7、2π/3 等算式。'}</small></div>`;
  const inp=$('#param-value');inp.oninput=()=>{if(rotation)prefs[prefs.unit]=inp.value;else prefs.factor=inp.value;};
  if(rotation){const unit=$('#param-angle-unit');unit.value=prefs.unit;unit.onchange=()=>{const from=prefs.unit,to=unit.value;try{const converted=window.GeomExpression.convertAngle(inp.value,from,to);prefs[from]=inp.value;prefs[to]=converted;prefs.unit=to;inp.value=converted;}catch(error){unit.value=from;notice('请先修正角度公式：'+error.message,true);}};
   $('#param-pi').onclick=()=>{const a=Number.isInteger(inp.selectionStart)?inp.selectionStart:inp.value.length,b=Number.isInteger(inp.selectionEnd)?inp.selectionEnd:a;inp.value=inp.value.slice(0,a)+'π'+inp.value.slice(b);inp.focus();inp.setSelectionRange(a+1,a+1);inp.dispatchEvent(new Event('input',{bubbles:true}));};
   $('#param-pi').hidden=unit.value!=='rad';unit.addEventListener('change',()=>{$('#param-pi').hidden=unit.value!=='rad';});}
  return;
 }
 const extras={circleRadius:['半径',2],tangentFrom:['切线分支 0/1',0],equilateral:['方向 0/1',0]};const item=extras[activeTool];
 el.innerHTML=item?`<div class="tool-config"><span>${item[0]}</span><input id="param-value" type="number" step="any" value="${item[1]}" aria-label="${item[0]}"></div>`:activeTool==='polygon'?'<p class="hint-in-tools">点击第一个顶点闭合；也可双击或按 Enter。</p>':'';
}
$('#tool-list').addEventListener('input',e=>{if(e.target.matches('#param-value, #param-angle-unit')){updateToolPreview(mouse);draw();}});
$('#tool-list').addEventListener('change',e=>{if(e.target.matches('#param-angle-unit')){updateToolPreview(mouse);draw();}});
function parameter(){const el=$('#param-value');if(!el)return 0;return window.GeomExpression.parse(el.value);}
function resize(){const cv=$('#board');const r=cv.getBoundingClientRect(),previousW=W,previousH=H;W=Math.max(1,r.width);H=Math.max(1,r.height);dpr=Math.min(3,window.devicePixelRatio||1);cv.width=Math.round(W*dpr);cv.height=Math.round(H*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);if(lastCanvasCenter){view.ox=W/2;view.oy=H/2;lastCanvasCenter=false;}else{view.ox+=(W-previousW)/2;view.oy+=(H-previousH)/2;}draw()}
function toScreen(p){return {x:view.ox+p.x*view.scale,y:view.oy-p.y*view.scale}}
function toWorld(p){return {x:(p.x-view.ox)/view.scale,y:(view.oy-p.y)/view.scale}}
function eventPos(e){const r=$('#board').getBoundingClientRect();return {x:e.clientX-r.left,y:e.clientY-r.top}}
function scrDist(a,b){return Math.hypot(a.x-b.x,a.y-b.y)}
function canvasColor(){return getComputedStyle(document.documentElement).getPropertyValue('--point').trim()||'#347ed3'}
function drawLine(g,style={}){const p=toScreen(g.p),d={x:g.d.x*view.scale,y:-g.d.y*view.scale},len=Math.hypot(d.x,d.y);if(len<.00001)return;const u={x:d.x/len,y:d.y/len};const b=g.extent==='segment'?{x:p.x+d.x,y:p.y+d.y}:g.extent==='ray'?{x:p.x+u.x*5000,y:p.y+u.y*5000}:{x:p.x+u.x*5000,y:p.y+u.y*5000};const a=g.extent==='line'?{x:p.x-u.x*5000,y:p.y-u.y*5000}:p;
 ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke();}
function pointGlyph(p,name,color,focus=false){const s=toScreen(p);ctx.beginPath();ctx.arc(s.x,s.y,focus?5.5:4.1,0,Math.PI*2);ctx.fillStyle=settings.theme==='dark'?'#172439':'#fff';ctx.fill();ctx.lineWidth=focus?2.7:2.1;ctx.strokeStyle=color;ctx.stroke();if(settings.labels){ctx.font='600 12px "Segoe UI",sans-serif';ctx.textAlign='left';ctx.fillStyle=focus?'#e9ab68':settings.theme==='dark'?'#c5d4e7':'#475d7a';ctx.fillText(name,s.x+10,s.y-10)}}
function drawGrid(){ctx.fillStyle=settings.theme==='dark'?'#121c2a':'#fbfdff';ctx.fillRect(0,0,W,H);if(!settings.grid)return;const u=view.scale;const step=u<34?Math.ceil(34/u)*u:u>150?u/2:u;const opacity=settings.theme==='dark'?'#202d3f':'#e9eff7';ctx.save();ctx.lineWidth=1;ctx.strokeStyle=opacity;let x=((view.ox%step)+step)%step;ctx.beginPath();for(;x<W;x+=step){ctx.moveTo(Math.round(x)+.5,0);ctx.lineTo(Math.round(x)+.5,H)}let y=((view.oy%step)+step)%step;for(;y<H;y+=step){ctx.moveTo(0,Math.round(y)+.5);ctx.lineTo(W,Math.round(y)+.5)}ctx.stroke();ctx.restore()}
function drawAxes(){const real=kernel.node('l_Re'),imag=kernel.node('l_Im');if(!isVisible(real)&&!isVisible(imag))return;
 ctx.save();ctx.font='11px Consolas,monospace';ctx.fillStyle=settings.theme==='dark'?'#8194b0':'#91a5c0';if(isVisible(real))ctx.fillText('Re',W-30,view.oy-8);if(isVisible(imag))ctx.fillText('Im',view.ox+8,19);
 if(isVisible(real)){const delta=view.scale<45?2:1;for(let k=Math.ceil(-view.ox/(view.scale*delta));view.ox+k*view.scale*delta<W;k++){if(k===0)continue;const x=view.ox+k*view.scale*delta;ctx.strokeStyle=settings.theme==='dark'?'#677e9e':'#a6bbd2';ctx.beginPath();ctx.moveTo(x,view.oy-4);ctx.lineTo(x,view.oy+4);ctx.stroke();}}ctx.restore();}
function drawOverlay(){const entry=proofOverlay||(radarEntry&&(radarResult||radarEntry.proofFact)?radarEntry:null);if(!entry)return;try{
 RelationOverlay.draw(ctx,view,entry,id=>kernel.geom(id),settings.theme);
 }catch(e){console.warn('关系高亮不可用',e)}}
// Predict the exact construction result from the same Kernel evaluator as commit.
// No temporary objects are written into the graph and no undo event is generated.
function previewParams(op){
 try{
  const value=$('#param-value')?.value;
  if(op==='circleRadius')return {radius:window.GeomExpression.parse(value)};
  if(op==='rotate'){const theta=window.GeomExpression.parse(value);return {degrees:theta*(($('#param-angle-unit')?.value||'deg')==='rad'?180/Math.PI:1)};}
  if(op==='homothety')return {factor:window.GeomExpression.parse(value)};
  if(op==='tangentFrom'||op==='equilateral')return {branch:Number(value)===1?1:0};
 }catch(e){return null;}return {};
}
function updateToolPreview(pos){
 toolPreview=null;ghost=null;hoverCarriers=[];
 const spec=METHODS[activeTool];if(!spec||activeTool==='intersection'){
  if(activeTool==='smartPoint'||activeTool==='intersection')updateGhost(pos);
  return;
 }
 const next=expected(activeTool,steps.length);
 let candidate=null;
 if(next.includes('point')){const current=!settings.snap?closest(pos,['point']):null;candidate=settings.snap?pointCandidate(pos):current?{kind:'existing',point:current.geom,existing:current}:{kind:'free',point:toWorld(pos)};}
 else if(next.length)candidate=closest(pos,next);
 if(candidate?.point){ghost=candidate;hoverCarriers=candidate.carriers||[];}
 const geoms=steps.map(id=>kernel.geom(id)).filter(Boolean);
 if(candidate){const g=candidate.point?{kind:'point',x:candidate.point.x,y:candidate.point.y}:candidate.geom;
  if(g)geoms.push(g);
 }
 const params=previewParams(activeTool);
 if(params!==null&&geoms.length>=(spec.variadic?3:spec.inputs.length)){
  try{toolPreview=previewConstruction(activeTool,geoms,params);}catch(e){}
 }
 // Before three vertices are available, outline the open polygon chain.
 if(activeTool==='polygon'&&geoms.length>1)toolPreview=toolPreview||{kind:'draftPolygon',points:geoms.filter(g=>g.kind==='point')};
 // For unfinished 2/3-point constructions, connect the chosen vertices.
 if(!toolPreview&&geoms.length>=2&&['circumcircle','centroid','circumcenter','incenter','orthocenter','excenter','angleBisector','outerAngleBisector','circleBySegment','foot3','perpendicular3','parallel3','translate','onCircumcircle','secondIntersection','similarPoint','equalAngleLine'].includes(activeTool)){
  toolPreview={kind:'draftPolygon',points:geoms.filter(g=>g.kind==='point')};
 }
}
function drawToolPreview(){
 const g=toolPreview;
 if(!g&&!ghost)return;
 ctx.save();ctx.strokeStyle='#159eaf';ctx.fillStyle='#159eaf22';ctx.lineWidth=2.2;ctx.setLineDash([7,5]);
 if(g?.kind==='line')drawLine(g);
 if(g?.kind==='circle'&&Number.isFinite(g.r)&&g.r>0){const c=toScreen(g.c);ctx.beginPath();ctx.arc(c.x,c.y,g.r*view.scale,0,Math.PI*2);ctx.stroke();}
 if(g?.kind==='point'){const s=toScreen(g);ctx.beginPath();ctx.arc(s.x,s.y,7,0,Math.PI*2);ctx.stroke();}
 if(g?.kind==='polygon'||g?.kind==='draftPolygon'){
  ctx.beginPath();g.points.forEach((p,i)=>{const s=toScreen(p);i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y)});
  if(g.kind==='polygon')ctx.closePath();ctx.stroke();if(g.kind==='polygon')ctx.fill();
 }
 if(ghost?.point){const s=toScreen(ghost.point);ctx.setLineDash([3,3]);ctx.beginPath();ctx.arc(s.x,s.y,7,0,2*Math.PI);ctx.stroke();ctx.beginPath();ctx.arc(s.x,s.y,2.5,0,2*Math.PI);ctx.fill();}
 ctx.restore();
}
function draw(){const canvas=$('#board');if(!canvas||!W)return;ctx.save();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,W,H);drawGrid();const color=canvasColor();for(const n of kernel.list()){const g=n.geom;if(!isVisible(n)||g?.kind==='point')continue;const focus=n.id===selected||n.id===hovered;ctx.save();ctx.lineWidth=focus?2.5:1.55;ctx.strokeStyle=focus?'#efa65a':n.style?.color|| (n.style.system?(settings.theme==='dark'?'#58799d':'#b3c2d3'):g.kind==='circle'?(settings.theme==='dark'?'#84b4dd':'#7da8c8'):(settings.theme==='dark'?'#90abc9':'#8195ad'));if(g.kind==='circle')ctx.lineWidth=focus?2.3:1.6;if(n.style.dashed)ctx.setLineDash([6,5]);if(g.kind==='line')drawLine(g);else if(g.kind==='circle'){const p=toScreen(g.c);ctx.beginPath();ctx.arc(p.x,p.y,g.r*view.scale,0,Math.PI*2);ctx.stroke()}else if(g.kind==='polygon'){ctx.fillStyle='#72b5e115';ctx.beginPath();g.points.forEach((p,i)=>{const s=toScreen(p);i?ctx.lineTo(s.x,s.y):ctx.moveTo(s.x,s.y)});ctx.closePath();ctx.fill();ctx.stroke()}ctx.restore();}
 drawAxes();drawLibraryVisuals();drawOverlay();if(auxPreview){ctx.save();ctx.lineWidth=2.4;ctx.strokeStyle='#c88b4b';ctx.fillStyle='#c88b4b';ctx.setLineDash([7,5]);const g=auxPreview;if(g.kind==='point'){const q=toScreen(g);ctx.beginPath();ctx.arc(q.x,q.y,9,0,Math.PI*2);ctx.stroke();}else if(g.kind==='line')drawLine(g);else if(g.kind==='circle'){const q=toScreen(g.c);ctx.beginPath();ctx.arc(q.x,q.y,g.r*view.scale,0,Math.PI*2);ctx.stroke();}ctx.restore();}for(const n of kernel.list()){if(!isVisible(n)||n.geom?.kind!=='point')continue;const focus=n.id===selected||n.id===hovered;pointGlyph(n.geom,n.id,focus?'#f2a14f':n.style.system?'#a78bcc':n.op==='free'||n.op==='real'?color:'#638cab',focus)}
 if(steps.length&&activeTool!=='select'&&activeTool!=='smartPoint'&&activeTool!=='intersection'){const start=kernel.geom(steps[0]);if(start?.kind==='point'){const s=toScreen(start);ctx.save();ctx.setLineDash([4,5]);ctx.strokeStyle='#9fbfdf';ctx.lineWidth=1.6;ctx.beginPath();ctx.moveTo(s.x,s.y);ctx.lineTo(mouse.x,mouse.y);ctx.stroke();ctx.restore()}steps.forEach(id=>{const g=kernel.geom(id);if(g?.kind==='point'){const s=toScreen(g);ctx.beginPath();ctx.arc(s.x,s.y,9,0,2*Math.PI);ctx.strokeStyle='#5fb6ab';ctx.lineWidth=2;ctx.stroke()}})}// Preview and carriers use the exact same candidate that a click would consume.
 if((activeTool==='smartPoint'||activeTool==='intersection')&&ghost){
  ctx.save();const palette=ghost.kind==='intersection'?'#e5a24d':'#4a9bda';
  for(const n of hoverCarriers){const g=n.geom;ctx.lineWidth=3;ctx.strokeStyle=palette;ctx.setLineDash([7,4]);
   if(g.kind==='line')drawLine(g);else if(g.kind==='circle'){const c=toScreen(g.c);ctx.beginPath();ctx.arc(c.x,c.y,g.r*view.scale,0,2*Math.PI);ctx.stroke();}}
  const pt=toScreen(ghost.point);ctx.setLineDash([4,4]);ctx.lineWidth=1.5;ctx.strokeStyle=palette;ctx.beginPath();ctx.arc(pt.x,pt.y,9,0,2*Math.PI);ctx.stroke();ctx.setLineDash([]);
  ctx.fillStyle=settings.theme==='dark'?'#182439dd':'#ffffffcc';ctx.beginPath();ctx.arc(pt.x,pt.y,4,0,2*Math.PI);ctx.fill();ctx.stroke();ctx.restore();
 }
 if(METHODS[activeTool])drawToolPreview();
 ctx.restore()}
function pointInPolygon(p,vertices){let inside=false;for(let i=0,j=vertices.length-1;i<vertices.length;j=i++){const a=vertices[i],b=vertices[j];if((a.y>p.y)!==(b.y>p.y)&&p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y)+a.x)inside=!inside;}return inside;}
function closest(pos,accept=['point','line','circle','polygon'],excluded=[]){
 const hits=[],list=kernel.list(),limit=14;
 for(let k=list.length-1;k>=0;k--){const n=list[k];if(!isVisible(n)||excluded.includes(n.id)||!accept.includes(n.geom.kind))continue;
  const g=n.geom;let d=Infinity,inside=false;
  if(g.kind==='point')d=scrDist(toScreen(g),pos);
  else if(g.kind==='circle')d=Math.abs(scrDist(toScreen(g.c),pos)-g.r*view.scale);
  else if(g.kind==='line'){const p=toScreen(g.p),v={x:g.d.x*view.scale,y:-g.d.y*view.scale},len=v.x*v.x+v.y*v.y;if(len<1e-10)continue;let t=((pos.x-p.x)*v.x+(pos.y-p.y)*v.y)/len;if(g.extent==='segment')t=Math.max(0,Math.min(1,t));if(g.extent==='ray')t=Math.max(0,t);d=scrDist(pos,{x:p.x+v.x*t,y:p.y+v.y*t});}
  else if(g.kind==='polygon'){const pts=g.points.map(toScreen);inside=pointInPolygon(pos,pts);d=Math.min(...pts.map((p,i)=>{const q=pts[(i+1)%pts.length],v={x:q.x-p.x,y:q.y-p.y},len=v.x*v.x+v.y*v.y,t=len?Math.max(0,Math.min(1,((pos.x-p.x)*v.x+(pos.y-p.y)*v.y)/len)):0;return scrDist(pos,{x:p.x+t*v.x,y:p.y+t*v.y})}));if(inside)d=Math.min(d,9);}
  if(d<=limit)hits.push({n,d,inside,order:k});
 }
 // Prioritize point glyphs in their entire interaction halo: derived points,
 // references, and free points must not be eclipsed by lines passing through them.
 hits.sort((a,b)=>{const rank=x=>x.n.geom.kind==='point'?0:x.n.geom.kind==='polygon'&&x.inside?2:1;return rank(a)-rank(b)||a.d-b.d||b.order-a.order;});
 return hits[0]?.n||null;
}
function expected(tool,index){if(tool==='intersection')return ['line','circle'];const spec=METHODS[tool];if(!spec)return [];const t=spec.inputs[Math.min(index,spec.inputs.length-1)]||'point';return t==='carrier'?['line','circle']:[t];}
function carrierHits(pos,limit=16){const hits=[];const here=toWorld(pos);
 for(const n of kernel.list()){if(!isVisible(n)||!['line','circle'].includes(n.geom.kind))continue;
  const g=n.geom;let screenPoint,d;
  if(g.kind==='circle'){const v={x:here.x-g.c.x,y:here.y-g.c.y},norm=Math.hypot(v.x,v.y);screenPoint=toScreen({x:g.c.x+g.r*(norm>1e-12?v.x/norm:1),y:g.c.y+g.r*(norm>1e-12?v.y/norm:0)});}
  else {const dd=g.d.x*g.d.x+g.d.y*g.d.y;let t=((here.x-g.p.x)*g.d.x+(here.y-g.p.y)*g.d.y)/dd;if(g.extent==='segment')t=Math.min(1,Math.max(0,t));if(g.extent==='ray')t=Math.max(0,t);screenPoint=toScreen({x:g.p.x+g.d.x*t,y:g.p.y+g.d.y*t});}
  d=scrDist(pos,screenPoint);if(d<=limit)hits.push({n,point:toWorld(screenPoint),d});
 }return hits.sort((a,b)=>a.d-b.d).slice(0,12);
}
function carrierPair(a,b){return [a,b].sort((x,y)=>(x.geom.kind==='line'?0:1)-(y.geom.kind==='line'?0:1)||x.id.localeCompare(y.id));}
function intersectionTarget(pos,hits=carrierHits(pos,19),radius=17){let best=null;
 for(let a=0;a<hits.length;a++)for(let b=a+1;b<hits.length;b++){
  const carriers=carrierPair(hits[a].n,hits[b].n),[first,second]=carriers;
  const pts=intersectionBranches(first.geom,second.geom);
  for(let branch=0;branch<pts.length;branch++){
   const point=pts[branch];if(!point)continue;const distance=scrDist(toScreen(point),pos);
   if(distance<=radius&&(!best||distance<best.distance))best={kind:'intersection',point,carriers,branch,distance};
  }
 }return best;
}
function pointCandidate(pos,onlyIntersection=false,snapCarriers=true){const hits=carrierHits(pos,19);
 const current=snapCarriers?closest(pos,['point']):null;if(current&&!onlyIntersection)return {kind:'existing',point:current.geom,existing:current};
 const intersection=snapCarriers?intersectionTarget(pos,hits):null;
 if(intersection)return intersection;
 if(onlyIntersection)return null;
 if(hits.length&&snapCarriers)return {kind:'carrier',point:hits[0].point,carriers:[hits[0].n]};
 return {kind:'free',point:toWorld(pos),carriers:[]};
}
function updateGhost(pos){ghost=activeTool==='smartPoint'?pointCandidate(pos):activeTool==='intersection'?pointCandidate(pos,true):null;
 hoverCarriers=ghost?.carriers||[];
 if(activeTool==='intersection'&&steps.length){const first=kernel.node(steps[0]);if(first&&!hoverCarriers.some(n=>n.id===first.id))hoverCarriers=[first,...hoverCarriers];}
}
function createIntersection(carrierA,carrierB,branch){
 [carrierA,carrierB]=carrierPair(carrierA,carrierB);
 const pts=intersectionBranches(carrierA.geom,carrierB.geom);if(!pts.some(Boolean))return [];
 const indices=branch==null?pts.flatMap((p,i)=>p?[i]:[]):[branch],result=[];
 for(const i of indices){if(!pts[i])continue;
  const existing=kernel.list().find(n=>n.valid&&n.op==='intersection'&&n.inputs[0]===carrierA.id&&n.inputs[1]===carrierB.id&&n.params.branch===i);
  if(existing){result.push(existing);continue;}
  const v=command({op:'intersection',inputs:[carrierA.id,carrierB.id],params:{branch:i,canonicalInputs:true}},true);if(v)result.push(v);
 }return result;
}
function createSmart(pos,snapCarriers=true){const target=pointCandidate(pos,false,snapCarriers);if(!target)return null;
 if(target.kind==='existing')return target.existing;
 if(target.kind==='intersection')return createIntersection(...target.carriers,target.branch)[0]||null;
 const w=target.point;
 if(target.kind==='carrier'){
  const n=target.carriers[0],g=n.geom;let params={};
  if(g.kind==='line'){let t=((w.x-g.p.x)*g.d.x+(w.y-g.p.y)*g.d.y)/(g.d.x*g.d.x+g.d.y*g.d.y);
   if(g.extent==='segment')t=Math.max(0,Math.min(1,t));if(g.extent==='ray')t=Math.max(0,t);params.t=t;
  }else params.theta=Math.atan2(w.y-g.c.y,w.x-g.c.x);
  return command({op:g.kind==='line'?'onLine':'onCircle',inputs:[n.id],params},true);
 }
 return command({op:'free',params:{x:w.x,y:w.y}},true);
}
function allocatePoint(pos){const v=createSmart(pos,settings.snap);return v?.id||null;}
function startToolClick(pos){mouse=pos;const w=toWorld(pos);if(activeTool==='select'){const n=closest(pos);selected=n?.id||null;objTab=n?'details':'objects';$('#acc-objects').classList.add('open');render();return;}
 if(activeTool==='smartPoint'){const v=createSmart(pos);if(v)selected=v.id;updateGhost(pos);render();return;}
 if(activeTool==='intersection'){
  // Clicking a visible crossing adds only the indicated branch. Otherwise,
  // pick two carriers and add every branch (one or two) of their intersection.
  const direct=intersectionTarget(pos);
  if(direct){const v=createIntersection(direct.carriers[0],direct.carriers[1],direct.branch);if(v.length)selected=v[0].id;steps=[];updateGhost(pos);render();return;}
  const hits=carrierHits(pos);
  const carrier=hits.find(x=>!steps.includes(x.n.id))?.n;
  if(!carrier){notice('请点击相交位置，或依次选择两条直线/圆',true);return;}
  if(!steps.length){steps=[carrier.id];setHint('已选第一条载体 '+carrier.label+' · 再点击第二条载体');updateGhost(pos);draw();return;}
  const first=kernel.node(steps[0]);steps=[];
  const result=first?createIntersection(first,carrier):[];
  if(result.length){selected=result[0].id;setHint('已生成 '+result.length+' 个交点 · 固定构造分支');}
  else notice('两对象没有有效交点',true);
  updateGhost(pos);render();return;
 }
 const allow=expected(activeTool,steps.length),isPt=allow.length===1&&allow[0]==='point';
 let pick=isPt?(settings.snap?pointCandidate(pos)?.existing:closest(pos,['point']))||null:closest(pos,allow,steps);
 // A click near a line/circle/crossing becomes a constrained/intersection point.
 if(!pick&&isPt){const id=allocatePoint(pos);pick=kernel.node(id)}
 if(!pick){notice('请选择'+(isPt?'点':allow.join('/')),true);return;}
 if(steps.includes(pick.id)){if(activeTool==='polygon'&&steps.length>=3&&pick.id===steps[0])finalizeTool(pos);else notice('请选择不同的对象',true);return;}
 steps.push(pick.id);
 if(activeTool==='polygon'){setHint(`${steps.length} 个顶点 · 点击首顶点闭合 / 双击 / Enter 完成`);updateToolPreview(pos);draw();return;}
 const spec=METHODS[activeTool];if(steps.length<spec.inputs.length){setHint(`已选择 ${steps.join(' → ')} · 继续选择第 ${steps.length+1} 个对象`);draw();return;}
 finalizeTool(pos);
}
function finalizeTool(pos){const op=activeTool;if(op==='polygon'&&steps.length<3){notice('至少需要三个顶点',true);return;}
 let params={};try{
 if(op==='circleRadius')params.radius=parameter();
 if(op==='rotate'){
  const raw=parameter(),unit=$('#param-angle-unit')?.value||'deg';
  params.degrees=unit==='rad'?raw*180/Math.PI:raw;params.angleExpr=$('#param-value').value;params.angleUnit=unit;
 }
 if(op==='homothety'){params.factor=parameter();params.factorExpr=$('#param-value').value;}
 if(op==='tangentFrom'||op==='equilateral')params.branch=parameter()===1?1:0;
 }catch(e){notice('公式错误：'+e.message,true);return;}
 if(op==='intersection'){const pts=intersectionBranches(kernel.geom(steps[0]),kernel.geom(steps[1]));if(!pts.some(Boolean)){notice('没有有效的相交点',true);steps=[];draw();return;}const distances=pts.map(p=>p?scrDist(toScreen(p),pos||mouse):Infinity);params.branch=distances.indexOf(Math.min(...distances));}
 const value=command({op,inputs:steps,params},true);steps=[];toolPreview=null;ghost=null;if(value){selected=value.id;setHint(`已创建 ${value.label} · ${toolName(op)}`);}render()}
function zoomAt(pos,factor){const w=toWorld(pos);view.scale=Math.max(18,Math.min(420,view.scale*factor));view.ox=pos.x-w.x*view.scale;view.oy=pos.y+w.y*view.scale;$('#zoom-label').textContent=Math.round(view.scale/88*100)+'%';draw()}
function fit(){const arr=[];kernel.list().forEach(n=>{if(!isVisible(n))return;const g=n.geom;if(g.kind==='point')arr.push(g);else if(g.kind==='circle')arr.push({x:g.c.x-g.r,y:g.c.y-g.r},{x:g.c.x+g.r,y:g.c.y+g.r});else if(g.kind==='polygon')arr.push(...g.points);else if(g.kind==='line'&&g.extent==='segment')arr.push(g.p,{x:g.p.x+g.d.x,y:g.p.y+g.d.y})});if(!arr.length){view.scale=88;view.ox=W/2;view.oy=H/2;draw();return;}let x0=Math.min(...arr.map(p=>p.x)),x1=Math.max(...arr.map(p=>p.x)),y0=Math.min(...arr.map(p=>p.y)),y1=Math.max(...arr.map(p=>p.y));const width=Math.max(x1-x0,2),height=Math.max(y1-y0,2);view.scale=Math.min(155,Math.max(20,Math.min((W-130)/width,(H-170)/height)));view.ox=W/2-(x0+x1)/2*view.scale;view.oy=H/2+(y0+y1)/2*view.scale;$('#zoom-label').textContent=Math.round(view.scale/88*100)+'%';draw()}
function nearestMovablePoint(pos,halo=15){
 let best=null;
 // Prefer movable points even if a derived point, a circle or a line overlaps them.
 // No invisible or fixed reference objects may hijack a drag.
 for(const n of kernel.list()){
  if(!isVisible(n)||n.geom?.kind!=='point'||!['free','real','onLine','onCircle'].includes(n.op)||n.style.system)continue;
  const d=scrDist(toScreen(n.geom),pos);
  if(d<=halo&&(!best||d<best.distance-0.1||(Math.abs(d-best.distance)<=0.1&&n.id===selected)))best={node:n,distance:d};
 }return best?.node||null;
}
function pointerSetup(){const cv=$('#board');cv.addEventListener('pointerdown',e=>{if(e.button===2)return;cv.setPointerCapture(e.pointerId);const pos=eventPos(e);mouse=pos;const chosen=nearestMovablePoint(pos,15);if((activeTool==='select'&&!spaceDown&&e.button===0)&&chosen){
   drag={type:'point',id:chosen.id,start:pos,before:kernel.serialize(),moved:false};selected=chosen.id;objTab='details';render();return;}
  if((activeTool==='select'&&(!chosen||chosen.style.system||!['free','real','onLine','onCircle'].includes(chosen.op)))||spaceDown||e.button===1){drag={type:'pan',start:pos,ox:view.ox,oy:view.oy,moved:false};$('#board-wrap').classList.add('is-panning');return;}
  drag={type:'click',start:pos,moved:false};});
 cv.addEventListener('pointermove',e=>{const pos=eventPos(e);mouse=pos;const world=toWorld(pos);$('#coords').innerHTML=`x ${fmt(world.x)} &nbsp; y ${fmt(world.y)}`;
  if(drag){if(scrDist(pos,drag.start)>3)drag.moved=true;if(drag.type==='point'&&drag.moved){kernel.previewMove(drag.id,world.x,world.y);$('#board-wrap').classList.add('is-dragging');draw();return;}if(drag.type==='pan'&&drag.moved){view.ox=drag.ox+pos.x-drag.start.x;view.oy=drag.oy+pos.y-drag.start.y;draw();return;}}
  const n=closest(pos);const newId=n?.id||null;if(activeTool==='smartPoint'||activeTool==='intersection'){updateGhost(pos);hovered=newId;draw();}else if(METHODS[activeTool]){updateToolPreview(pos);hovered=newId;draw();}else if(newId!==hovered){hovered=newId;draw()}else if(steps.length)draw();});
 cv.addEventListener('pointerup',e=>{if(!drag)return;const d=drag;drag=null;$('#board-wrap').classList.remove('is-panning','is-dragging');const pos=eventPos(e);if(d.type==='point'){if(d.moved){kernel.checkpoint(d.before);render()}else{selected=d.id;objTab='details';render();}return;}
 if((d.type==='click'||d.type==='pan')&&!d.moved)startToolClick(pos);else draw();});
 cv.addEventListener('pointerleave',()=>{if(!drag){ghost=null;toolPreview=null;hoverCarriers=[];draw()}});
 cv.addEventListener('pointercancel',()=>{if(drag?.type==='point'){kernel.load(drag.before,false)}drag=null;$('#board-wrap').classList.remove('is-panning','is-dragging');render()});
 cv.addEventListener('dblclick',e=>{if(activeTool==='polygon'&&steps.length>=3)finalizeTool(eventPos(e))});
 cv.addEventListener('contextmenu',e=>{e.preventDefault();steps=[];draw()});cv.addEventListener('wheel',e=>{if(!settings.wheel)return;e.preventDefault();zoomAt(eventPos(e),Math.exp(-e.deltaY*.0013))},{passive:false});
 $('#zoom-in').onclick=()=>zoomAt({x:W/2,y:H/2},1.23);$('#zoom-out').onclick=()=>zoomAt({x:W/2,y:H/2},1/1.23);$('#fit-btn').onclick=fit;
}
function formatDef(n){const ins=n.inputs.map(x=>kernel.node(x)?.id||x).join(', ');return `${METHODS[n.op].label}${ins?' · '+ins:''}`}
function renderObjects(){const nodes=exposedNodes(),root=$('#object-area');$('#object-count').textContent=nodes.length;$('#board-objects').textContent=nodes.filter(n=>!n.style.system).length+' 对象';$('#revision').textContent='REV '+kernel.revision;
 root.innerHTML=nodes.length?nodes.slice().reverse().map(n=>{const g=n.geom,kind=METHODS[n.op].kind;return `<button class="obj-row ${n.id===selected?'selected':''} ${n.valid?'':'invalid'} ${!isVisible(n)?'obj-invisible':''}" data-id="${esc(n.id)}"><span class="obj-ico ${kind}">${kind==='point'?'•':kind==='line'?'╱':kind==='circle'?'○':'△'}</span><span class="obj-info"><strong>${esc(n.id)}</strong><small>${esc(formatDef(n))}</small></span><span class="obj-hide">${isVisible(n)?'':'◌'}</span></button>`}).join(''):'<div class="empty-small">画板是空的。请从左侧开始作图。</div>';
 $$('.obj-row').forEach(b=>b.onclick=()=>{selected=b.dataset.id;objTab='details';render()});$$('[data-objtab]').forEach(b=>b.classList.toggle('on',b.dataset.objtab===objTab));root.hidden=objTab!=='objects';$('#details-area').hidden=objTab!=='details';$('#pretty-area').hidden=objTab!=='pretty';renderDetails()}
function renderDetails(){const root=$('#details-area'),n=kernel.node(selected);if(!n){root.innerHTML='<p class="empty-small">选择一个对象查看属性</p>';return}const g=n.geom;let data=[['ID',n.id],['类型',METHODS[n.op].label],['构造',n.op],['依赖',n.inputs.join(', ')||'—'],['状态',n.valid?'有效':'退化 / 失效'],['显示状态',isVisible(n)?'正在显示':n.style.visible?'受坐标轴/复平面开关控制而隐藏':'已独立隐藏']];if(n.valid&&g.kind==='point')data.push(['x',fmt(g.x)],['y',fmt(g.y)]);if(n.valid&&g.kind==='circle')data.push(['圆心',`(${fmt(g.c.x)}, ${fmt(g.c.y)})`],['半径',fmt(g.r)]);if(n.valid&&g.kind==='line')data.push(['方向',`(${fmt(g.d.x)}, ${fmt(g.d.y)})`]);
 root.innerHTML=`${n.style.system?'<p class="reference-note">系统参考对象可独立隐藏/显示；坐标轴关闭时仍不会出现在画布中。0 / 实轴 / 虚轴随坐标轴显示，1 / i / 单位圆还需开启复平面。</p>':''}<div class="detail-title">${esc(n.id)}</div><input class="detail-rename" id="rename-field" value="${esc(n.id)}" aria-label="对象标签"/>${data.map(([k,v])=>`<div class="detail-line"><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('')}<div class="detail-actions"><button id="rename-btn" ${n.style.system?'disabled title="固定参考对象不可更名"':''}>重命名</button><button id="visibility-btn">${n.style.visible?'隐藏':'显示'}</button><button class="danger" id="delete-btn" ${n.style.system?'disabled title="固定参考对象不可删除"':''}>删除</button></div>`;
 $('#rename-btn').onclick=()=>{const v=command({op:'rename',id:n.id,label:$('#rename-field').value},true);if(v)selected=v.id;render()};$('#rename-field').onkeydown=e=>{if(e.key==='Enter')$('#rename-btn').click()};$('#visibility-btn').onclick=()=>{command({op:'style',id:n.id,style:{visible:!n.style.visible}},true);render()};$('#delete-btn').onclick=()=>{const v=command({op:'delete',id:n.id},true);if(v){selected=null;radarEntry=null;}render()}}
function complexNumber(x,y,sign=false){
 const re=fmt(x),im=fmt(Math.abs(y));return `(${re} ${y<0?'−':'+'} ${im}i)`;
}
function renderComplex(){const n=kernel.node(selected),g=n?.geom,root=$('#complex-values');
 if(!n||!n.valid){root.innerHTML='<p class="empty-small">选择一个点、直线或圆，查看复方程</p>';return;}
 let lines=[];
 if(g.kind==='point'){
  lines=[`<strong>z(${esc(n.id)})</strong> = ${esc(complexNumber(g.x,g.y))}`,`<strong>z̄</strong> = ${esc(complexNumber(g.x,-g.y))}`,`|z| = ${fmt(Math.hypot(g.x,g.y))} · arg = ${fmt(Math.atan2(g.y,g.x)*180/Math.PI)}°`];
 }else if(g.kind==='line'){
  const norm=Math.hypot(g.d.x,g.d.y);let nx=-g.d.y/norm,ny=g.d.x/norm;
  if(nx<0||(Math.abs(nx)<1e-12&&ny<0)){nx=-nx;ny=-ny;}
  const constant=-2*(nx*g.p.x+ny*g.p.y);
  lines=[`<strong>${esc(n.id)}：</strong> αz + ᾱz̄ + c = 0`,
   `α = ${esc(complexNumber(nx,-ny))}, |α| = 1`,
   `ᾱ = ${esc(complexNumber(nx,ny))}, c = ${fmt(constant)}`,
   `<span class="complex-equation">${esc(complexNumber(nx,-ny))}z + ${esc(complexNumber(nx,ny))}z̄ + (${fmt(constant)}) = 0</span>`];
 }else if(g.kind==='circle'){
  const x=g.c.x,y=g.c.y,k=x*x+y*y-g.r*g.r;
  lines=[`<strong>${esc(n.id)}：</strong> (z−a)(z̄−ā) = r²`,
   `a = ${esc(complexNumber(x,y))}, r = ${fmt(g.r)}`,
   `<span class="complex-equation">z z̄ − ${esc(complexNumber(x,-y))}z − ${esc(complexNumber(x,y))}z̄ + (${fmt(k)}) = 0</span>`];
 }else{root.innerHTML='<p class="empty-small">该对象类型暂无单一复方程</p>';return;}
 root.innerHTML=lines.map(t=>'<div class="complex-eq-row">'+t+'</div>').join('');
}
function radarOptions(){
 const root=$('#radar-options'),groups=new Map();for(const r of RelationEngine.registry){if(!groups.has(r.group))groups.set(r.group,[]);groups.get(r.group).push(r)}
 root.innerHTML=[...groups].map(([group,items])=>`<details class="radar-category" ${['基础位置','线与角','长度与面积'].includes(group)?'open':''}><summary>${esc(group)} <span>${items.length} 类</span></summary><div class="radar-category-options">${items.map(r=>`<label class="radar-option" title="${esc(r.help)}"><input type="checkbox" data-kind="${r.id}" ${r.enabled?'checked':''}/><span>${esc(r.label)}</span></label>`).join('')}</div></details>`).join('');
 const update=()=>$('#radar-selected-count').textContent=$$('#radar-options [data-kind]:checked').length+' / '+RelationEngine.registry.length+' 类';
 $$('#radar-options [data-kind]').forEach(el=>el.onchange=update);
 $('#radar-check-all').onclick=()=>{$$('#radar-options [data-kind]').forEach(el=>el.checked=true);update()};
 $('#radar-check-none').onclick=()=>{$$('#radar-options [data-kind]').forEach(el=>el.checked=false);update()};update();
 $('#radar-query').oninput=()=>{radarPage=1;renderRadar()};$('#radar-sort').onchange=()=>{radarPage=1;renderRadar()};
 $('#radar-more').onclick=()=>{radarPage++;renderRadar()};
}
function relationBoard(includeHidden){
 const pts=[],lines=[],circles=[],construction=[];
 for(const n of kernel.snapshot().nodes){if(!n.valid||!n.geom)continue;
  if(!includeHidden&&!isVisible(n))continue;
  const g=n.geom;construction.push({id:n.id,label:n.label,op:n.op,inputs:[...n.inputs],params:{...n.params}});
  if(g.kind==='point')pts.push({id:n.id,label:n.label,x:g.x,y:g.y});
  if(g.kind==='line')lines.push({id:n.id,label:n.label,x1:g.p.x,y1:g.p.y,x2:g.p.x+g.d.x,y2:g.p.y+g.d.y,drawType:g.extent==='segment'?'lineSegment':g.extent});
  if(g.kind==='circle')circles.push({id:n.id,label:n.label,cx:g.c.x,cy:g.c.y,r:g.r,centerId:['circle','circleRadius'].includes(n.op)?n.inputs[0]:null});
 }
 return {points:pts,lines,circles,construction,revision:kernel.revision};
}
// DDAR is the only source of the word "proved". Numeric matches alone never grant it.
function ddarLine(ref){
 if(typeof ref!=='string')return null;
 if(ref.startsWith('P:')){const x=ref.slice(2).split('|');return x.length===2?{type:'pair',a:x[0],b:x[1]}:null;}
 if(ref.startsWith('L:')){const id=ref.slice(2),n=kernel.node(id);return n?.geom?.kind==='line'?{type:'object',id}:null;}
 const p=ref.split('|');return p.length===2?{type:'pair',a:p[0],b:p[1]}:null;
}
function markDDAR(kind,args){
 const a=args||[],marks=[],pt=x=>({type:'point',id:x}),seg=(u,v)=>({type:'segment',a:u,b:v});
 const addLine=ref=>{const l=ddarLine(ref);if(l)marks.push({type:'line',ref:l});};
 const segment=x=>{const t=String(x||'').split('|');if(t.length===2)marks.push(seg(...t));};
 if(['para','perp'].includes(kind)&&a.length===2){a.forEach(addLine);}
 else if(['coll','cyclic'].includes(kind)&&a.length>=3){marks.push(...a.map(pt));if(kind==='coll')addLine('P:'+a[0]+'|'+a[1]);else if(a.length>=3)marks.unshift({type:'circumcircle',ids:a.slice(0,3)});}
 else if(kind==='midp'&&a.length===3){marks.push(pt(a[0]),seg(a[1],a[2]),seg(a[0],a[1]),seg(a[0],a[2]));}
 else if(kind==='cong'&&a.length===2){a.forEach(segment);}
 else if(kind==='eqratio'&&a.length>=2){segment(a[0]);segment(a[1]);}
 else if(kind==='eqangleLines'&&a.length===4){for(const x of a)addLine(x);}
 else if(kind==='eqangle6'&&a.length===6){marks.push({type:'angle',a:a[0],v:a[1],b:a[2]},{type:'angle',a:a[3],v:a[4],b:a[5]});}
 else if(kind==='eqratio'&&a.length===4){a.forEach(segment);}
 else if(kind==='rconst'&&a.length>=2){segment(a[0]);segment(a[1]);}
 else if(kind==='onl'&&a.length===2){marks.push(pt(a[0]));addLine(a[1]);}
 else if(kind==='onc'&&a.length===2){marks.push(pt(a[0]));if(kernel.node(a[1])?.geom?.kind==='circle')marks.push({type:'circle',ref:{type:'object',id:a[1]}});}
 else if(kind==='isos'&&a.length===3){marks.push(seg(a[0],a[1]),seg(a[0],a[2]));}
 return marks;
}
function displayDDARFact(f){
 const original=f.args||[],reverse=new Map();
 for(const [name,internal] of Object.entries(v45Task?.prepared?.ag1?.nameMap||{}))
  if(!reverse.has(internal)||(!name.startsWith('@')&&reverse.get(internal).startsWith('@')))reverse.set(internal,name);
 const human=x=>reverse.get(x)||x;
 let a=original.map(human),kind=f.kind;
 if(kind==='collx')kind='coll';
 // The original AlphaGeometry backend reports DD facts as point-token arrays,
 // whereas the goal enumerator stores segments/line carriers as strings.
 if(['para','perp','cong'].includes(kind)&&a.length===4){
  a=[a.slice(0,2).join('|'),a.slice(2,4).join('|')];
  if(kind!=='cong')a=a.map(x=>'P:'+x);
 }else if(kind==='eqangle'&&a.length===8){kind='eqangleLines';a=[0,2,4,6].map(i=>'P:'+a[i]+'|'+a[i+1]);}
 else if(kind==='eqratio'&&a.length===8){a=[a.slice(0,2).join('|'),a.slice(2,4).join('|'),a.slice(4,6).join('|'),a.slice(6,8).join('|')];}
 else if(kind==='eqangle6'&&a.length===6){a=a.slice();}
 else if(kind==='rconst'&&a.length>=4){a=[a.slice(0,2).join('|'),a.slice(2,4).join('|'),...(a.slice(4))];}
 const source=String(f.text||f.id||'');
 const readable=source.replace(/\b[A-Za-z][A-Za-z0-9_]*\b/g,m=>human(m));
 return {...f,kind,args:a,text:readable};
}
function proofEntryFromFact(f){
 f=displayDDARFact(f);const marks=markDDAR(f.kind,f.args);
 return {id:'v5-proof:'+f.id,kind:({para:'parallel',perp:'perpendicular',coll:'collinear',cyclic:'concyclic',cong:'distanceRatio',midp:'midpoint',onl:'pointOnLine',onc:'pointOnCircle'}[f.kind]||f.kind),title:f.text||f.id,
  searchText:[f.kind,f.text,(f.args||[]).join(' '),'形式证明 已证'].join(' '),detail:'AlphaGeometry '+($('#ddar-engine')?.value==='ag1'?'1':'2')+' · '+(f.rule||'DDAR'),
  score:100,marks,proofFact:f,source:'formal'};
}
function getProvedEntries(){return (ddarState?.facts||[]).map(proofEntryFromFact);}
function normalizedMarks(marks){return (marks||[]).map(m=>{
 if(m.type==='point')return 'P:'+m.id;
 if(m.type==='line'){const r=m.ref;if(r?.type==='pair')return 'L:'+([r.a,r.b].sort().join('|'));if(r?.type==='object'){
  const n=kernel.node(r.id);if(n&&['line','segment','ray'].includes(n.op)&&n.inputs.length===2)return 'L:'+n.inputs.slice().sort().join('|');
  return 'Lobj:'+r.id;}}
 if(m.type==='segment')return 'S:'+([m.a,m.b].sort().join('|'));
 return null;
 }).filter(Boolean).sort().join(' / ');}
function relationProvedBy(r,proven){
 if(r.proofFact)return true;
 const kind={parallel:'para',perpendicular:'perp',collinear:'coll',concyclic:'cyclic',midpoint:'midp',pointOnLine:'onl',pointOnCircle:'onc'}[r.kind];
 if(!kind)return false;
 const signature=normalizedMarks(r.marks);
 if(!signature)return false;
 return proven.has(kind+'|'+signature);
}
function provedMarkSignatures(data){
 const out=new Set();for(const raw of data){const f=displayDDARFact(raw),marks=markDDAR(f.kind,f.args);
  const signature=normalizedMarks(marks);if(signature)out.add(f.kind+'|'+signature);
 }return out;
}
function v52DisplayStep(raw){
 const s=String(raw||''),p=s.trim().split(/\s+/);
 if(p[0]==='coll'&&p.length>=4)return p.slice(1).join('、')+' 共线';
 if(p[0]==='para'&&p.length>=5)return p[1]+p[2]+' ∥ '+p[3]+p[4];
 if(p[0]==='perp'&&p.length>=5)return p[1]+p[2]+' ⟂ '+p[3]+p[4];
 if(p[0]==='cong'&&p.length>=5)return p[1]+p[2]+' = '+p[3]+p[4];
 if(p[0]==='eqratio'&&p.length>=9)return p[1]+p[2]+':'+p[3]+p[4]+' = '+p[5]+p[6]+':'+p[7]+p[8];
 if(p[0]==='eqangle'&&p.length>=9)return '∠('+p[1]+p[2]+','+p[3]+p[4]+') = ∠('+p[5]+p[6]+','+p[7]+p[8]+')';
 if(p[0]==='similar'&&p.length>=7)return '△'+p.slice(1,4).join('')+' ∼ △'+p.slice(4,7).join('');
 if(s.startsWith('angle:'))return '角度 AR：'+s.slice(6).replace(/d\((\S+) (\S+)\)/g,'θ($1$2)').replace(/\+ -1\*/g,' − ').trim();
 if(s.startsWith('length-mul:'))return '长度比例 AR：'+s.slice(11).replace(/log\(\|(\S+) (\S+)\|\)/g,'log|$1$2|').replace(/\+ -1\*/g,' − ').trim();
 if(s.startsWith('length-add:'))return '线段加法 AR：'+s.slice(11).trim();
 return s;
}
function v52DisplayRule(raw){const s=String(raw||'');
 if(s.includes('triangle similarity'))return 'DD · '+s.replace('triangle similarity','相似判定');
 if(s.includes('Gaussian equation insertion'))return 'AR · 精确高斯消元'+(s.includes('angle')?'（角度）':s.includes('length')?'（长度）':'');
 if(s==='GeomTool construction')return '作图已知前提';
 if(s==='AG2 符号关系检验')return '符号检查 · 目标成立';
 return s;
}
function v52OverlayEquivalent(raw){const s=String(raw||'');
 if(!s.startsWith('angle:'))return s;
 const ds=[...s.matchAll(/d\(([^ )]+) ([^ )]+)\)/g)];
 if(ds.length===2&&!/pi/.test(s))return 'para '+ds.map(m=>m[1]+' '+m[2]).join(' ');
 return s;
}
function proofOverlayFromPredicate(source){
 const str=String(source||'').trim(),tokens=str.replace(/[(),]/g,' ').split(/\s+/).filter(Boolean),kind=tokens[0]?.toLowerCase(),a=tokens.slice(1);
 let marks=[];
 const fromPairs=(k,ps)=>{const values=[];for(let i=0;i<ps.length-1;i+=2)values.push('P:'+ps[i]+'|'+ps[i+1]);return markDDAR(k,values);};
 if(['para','perp'].includes(kind)&&a.length>=4)marks=fromPairs(kind,a.slice(0,4));
 else if(kind==='cong'&&a.length>=4)marks=markDDAR('cong',[a.slice(0,2).join('|'),a.slice(2,4).join('|')]);
 else if(['coll','cyclic'].includes(kind))marks=markDDAR(kind,a.slice(0,kind==='coll'?3:4));
 else if(kind==='midp'&&a.length>=3)marks=markDDAR('midp',a.slice(0,3));
 else if(kind==='eqangle'&&a.length>=8)marks=fromPairs('eqangleLines',a.slice(0,8));
 else if(kind==='similar'&&a.length>=6)marks=[{type:'segment',a:a[0],b:a[1]},{type:'segment',a:a[1],b:a[2]},{type:'segment',a:a[0],b:a[2]},{type:'segment',a:a[3],b:a[4]},{type:'segment',a:a[4],b:a[5]},{type:'segment',a:a[3],b:a[5]}];
 else if(kind==='eqratio'&&a.length>=8)marks=[{type:'segment',a:a[0],b:a[1]},{type:'segment',a:a[2],b:a[3]},{type:'segment',a:a[4],b:a[5]},{type:'segment',a:a[6],b:a[7]}];
 else if(kind==='rconst'&&a.length>=4)marks=markDDAR('cong',[a.slice(0,2).join('|'),a.slice(2,4).join('|')]);
 else if(kind==='perp'&&a.length===2)marks=markDDAR('perp',a);
 else if(kind==='para'&&a.length===2)marks=markDDAR('para',a);
 // AlphaGeometry human-readable proof lines can use conventional angle,
 // perpendicular, and ratio notation rather than its raw predicate syntax.
 if(!marks.length){
  const names=new Set(kernel.list().filter(n=>n.geom?.kind==='point'&&!n.style?.system).map(n=>n.id));
  const parseToken=(t,n)=>{const walk=(v,arr)=>{if(arr.length===n)return v===''?arr:null;
   for(const id of [...names].sort((x,y)=>y.length-x.length))if(v.startsWith(id)){
    const found=walk(v.slice(id.length),[...arr,id]);if(found)return found;
   }return null;};return walk(t,[]);};
  const angleMatches=[...str.matchAll(/∠\s*([A-Za-z][A-Za-z0-9_]*)/g)];
  for(const m of angleMatches){const t=parseToken(m[1],3);if(t)marks.push({type:'angle',a:t[0],v:t[1],b:t[2]});}
  const segmentTokens=[...str.matchAll(/\b[A-Za-z][A-Za-z0-9_]*\b/g)].map(m=>m[0]);
  if(/[⊥⟂∥=：:]/.test(str))for(const t of segmentTokens){const ab=parseToken(t,2);
   if(ab&&!marks.some(m=>m.type==='segment'&&m.a===ab[0]&&m.b===ab[1]))marks.push({type:'segment',a:ab[0],b:ab[1]});}
  // In ∠(AB-CD), draw both carrier lines as well as the highlighted angle.
  for(const m of str.matchAll(/\(\s*([A-Za-z][A-Za-z0-9_]*)\s*[-–]\s*([A-Za-z][A-Za-z0-9_]*)\s*\)/g)){
   for(const part of [m[1],m[2]]){const ids=parseToken(part,2);if(ids)marks.push({type:'line',ref:{type:'pair',a:ids[0],b:ids[1]}});}
  }
 }
 // Other proof languages (especially AG1 readable angle equality): show only
 // existing point references, never fabricate a relation from ambiguous prose.
 if(!marks.length){const names=new Set(kernel.list().filter(n=>n.geom?.kind==='point'&&!n.style?.system).map(n=>n.id));
  const hits=str.match(/[A-Za-z][A-Za-z0-9_]*/g)||[];
  marks=[...new Set(hits.filter(x=>names.has(x)))].map(x=>({type:'point',id:x}));}
 return {marks,id:'proof-step',kind:'proof-step'};
}
function scanRadar(){
 const chosen=$$('#radar-options [data-kind]:checked').map(el=>el.dataset.kind);
 if(!chosen.length){notice('请先勾选至少一种关系',true);return;}
 const run=$('#radar-run');run.disabled=true;run.textContent='正在分析候选关系…';
 // Yield one frame so the UI can paint the busy state before CPU-intensive work.
 requestAnimationFrame(()=>setTimeout(()=>{try{
  const snapshot=relationBoard($('#radar-hidden').checked);
  radarResult=RelationEngine.scan(snapshot,{kinds:chosen,perturb:$('#radar-perturb').checked,...radarQuadraticOptions});radarEntry=null;radarPage=1;renderRadar();draw();
  notice(`扫描完成：${radarResult.count.toLocaleString('zh-CN')} 条候选关系`);
 }catch(e){notice('关系扫描失败：'+e.message,true);console.error(e)}finally{run.disabled=false;run.innerHTML=svg('spark')+' 扫描已勾选关系';}},0));
}
function renderRadar(){
 const root=$('#relations-list'),more=$('#radar-more'),stat=$('#radar-stats');
 const provenFacts=ddarState?.facts||[],formal=getProvedEntries(),provedSignatures=provedMarkSignatures(provenFacts);
 const candidates=radarResult?.entries||[];
 // A formally proved fact must remain visible even if the numeric scanner did
 // not enumerate its category or its maximal-cluster representative.
 const entries=[...candidates,...formal],query=($('#radar-query')?.value||'').trim().toLowerCase();
 const filtered=query?entries.filter(r=>GeomRelationSearch.matches(r,query,relationLabels[r.kind]||r.kind)):entries;
 radarFiltered=$('#radar-sort')?.value==='kind'?filtered.slice().sort((a,b)=>a.kind.localeCompare(b.kind)||b.score-a.score||a.id.localeCompare(b.id)):filtered;
 const shown=radarFiltered.slice(0,radarPage*80),matchStatus=matchedMap();
 $('#relation-count').textContent=entries.length.toLocaleString('zh-CN');
 if(radarResult)stat.textContent=`${candidates.length.toLocaleString('zh-CN')} 条数值候选 · ${formal.length} 条 AG 已证 · 匹配 ${filtered.length.toLocaleString('zh-CN')} · ${radarResult.objects.originalPoints} 点→${radarResult.objects.pointClusters} 簇 / ${radarResult.objects.supportLines} 支撑线 / ${radarResult.objects.circleGroups} 圆簇 · ${radarResult.perturbations?'微扰剔除':'微扰关闭'} ${radarResult.perturbations?radarResult.unstableRejected:''} · 逻辑合并 ${radarResult.semanticRemoved||0} · ${radarResult.elapsedMs} ms`;
 else stat.textContent=formal.length?`${formal.length} 条 AG 已证明结论；可直接点击高亮。运行扫描可同时查看数值关系。`:'等待扫描 · 不设结果数上限';
 root.innerHTML=shown.length?shown.map((r,i)=>{const proved=r.proofFact||matchStatus.get(r.id)==='proved'||relationProvedBy(r,provedSignatures);
 return `<button type="button" class="relation-item ${radarEntry?.id===r.id?'on':''} ${proved?'v5-formally-proven':''}" title="${esc(r.searchText||r.title)}" data-ri="${i}" aria-pressed="${radarEntry?.id===r.id}"><span>${esc(relationLabels[r.kind]||r.kind)} ${proved?'<small class="ddarmatch">✓ AG 已证明</small>':r.source==='construction'?'<small>· 作图条件</small>':r.source==='mixed'?'<small>· 含已知条件</small>':''} <em>${r.score} 分</em></span><strong>${esc(r.title)}</strong>${r.detail?`<small>${esc(r.detail)}</small>`:''}</button>`;}).join(''):
 `<div class="empty-small">${radarResult||formal.length?'没有符合搜索条件的关系':'选择关系类别并开始扫描'}</div>`;
 $$('.relation-item').forEach(b=>b.onclick=()=>{const r=radarFiltered[Number(b.dataset.ri)];radarEntry=radarEntry?.id===r.id?null:r;proofOverlay=null;renderRadar();draw()});
 more.hidden=shown.length>=radarFiltered.length;more.textContent=`显示更多（已显示 ${shown.length.toLocaleString('zh-CN')} / ${radarFiltered.length.toLocaleString('zh-CN')}）`;
}
function facts(){const all=[];const push=(id,conclusion,rule,premises)=>all.push({id:id+':'+all.length,objectId:id,conclusion,rule,premises});for(const n of kernel.list()){
 if(!n.valid)continue;const t=n.inputs.map(x=>kernel.node(x)?.id||x);const id=n.id;
 switch(n.op){case'midpoint':push(n.id,`${id} 是 ${t[0]}${t[1]} 的中点`,'MIDPOINT_DEF',[n.op+' '+t.join(' ')]);break;
 case'foot':push(n.id,`${id} 在 ${t[1]} 上，${t[0]}${id} ⟂ ${t[1]}`,'FOOT_DEF',t);break;
 case'foot3':push(n.id,`${id} 在直线 ${t[1]}${t[2]} 上，${t[0]}${id} ⟂ ${t[1]}${t[2]}`,'FOOT3_DEF',t);break;
 case'parallel':push(n.id,`${id} ∥ ${t[1]}，且过 ${t[0]}`,'PARALLEL_DEF',t);break;
 case'parallel3':push(n.id,`${id} ∥ ${t[1]}${t[2]}，且过 ${t[0]}`,'PARALLEL3_DEF',t);break;
 case'perpendicular':push(n.id,`${id} ⟂ ${t[1]}，且过 ${t[0]}`,'PERPENDICULAR_DEF',t);break;
 case'perpendicular3':push(n.id,`${id} ⟂ ${t[1]}${t[2]}，且过 ${t[0]}`,'PERPENDICULAR3_DEF',t);break;
 case'perpBisector':push(n.id,`${id} 是 ${t[0]}${t[1]} 的中垂线`,'PERP_BISECTOR_DEF',t);break;
 case'angleBisector':push(n.id,`${id} 平分 ∠${t.join('')}`,'ANGLE_BISECTOR_DEF',t);break;
 case'outerAngleBisector':push(n.id,`${id} 是 ∠${t.join('')} 的外角平分线`,'OUTER_ANGLE_BISECTOR_DEF',t);break;
 case'onLine':push(n.id,`${id} ∈ ${t[0]}`,'ON_LINE_DEF',t);break;
 case'onCircle':push(n.id,`${id} ∈ ${t[0]}`,'ON_CIRCLE_DEF',t);break;
 case'circle':push(n.id,`${id} 以 ${t[0]} 为圆心，并过 ${t[1]}`,'CIRCLE_DEF',t);break;
 case'circumcircle':push(n.id,`${t.join(', ')} 均在 ${id} 上`,'CIRCUMCIRCLE_DEF',t);break;
 case'centroid':push(n.id,`${id} 是 △${t.join('')} 的重心`,'CENTROID_DEF',t);break;
 case'circumcenter':push(n.id,`${id} 是 △${t.join('')} 的外心`,'CIRCUMCENTER_DEF',t);break;
 case'incenter':push(n.id,`${id} 是 △${t.join('')} 的内心`,'INCENTER_DEF',t);break;
 case'orthocenter':push(n.id,`${id} 是 △${t.join('')} 的垂心`,'ORTHOCENTER_DEF',t);break;
 case'intersection':push(n.id,`${id} 属于 ${t.join(' 与 ')}`,'INTERSECTION_DEF',t);break;
 case'tangent':push(n.id,`${id} 在 ${t[0]} 处与 ${t[1]} 相切`,'TANGENT_DEF',t);break;}
 }return all}
function renderProof(){proofFacts=facts();const select=$('#proof-goal'),current=select.value;select.innerHTML='<option value="">选择已知构造结论</option>'+proofFacts.map(f=>`<option value="${esc(f.id)}">${esc(f.conclusion)}</option>`).join('');if(proofFacts.some(f=>f.id===current))select.value=current}
function proofCheck(){const f=proofFacts.find(f=>f.id===$('#proof-goal').value),out=$('#proof-output');if(!f){out.classList.remove('success');out.textContent='尚未选择目标';return}const n=kernel.node(f.objectId);if(!n?.valid){out.classList.remove('success');out.textContent='构造当前无效，无法应用定义';return}out.classList.add('success');out.textContent=`✓ 构造定义已验证\n${f.conclusion}\n规则：${f.rule}\n依赖：${f.premises.join(', ')||'无'}\n类型：局部构造定义（不是完整自动证明）`}
function proofPayload(){return {schema:'geomtool.ddar/v4.4',source:'GeomTool v4.4',graph:kernel.serialize(),goal:proofFacts.find(f=>f.id===$('#proof-goal').value)||null,constructionFacts:proofFacts,computedFacts:proofFacts,engine:ddarState?.version||($('#ddar-engine')?.value==='ag1'?'AlphaGeometry1-DDAR/Pyodide-v4.4':'AlphaGeometry2-DDAR/Pyodide-v4.4'),selectedDDARGoal:goalSelection,goalProof:goalResult,goalCatalog:goalCatalog?.goals||[],ddAR:ddarState?{version:ddarState.version,facts:ddarState.facts,matches:ddarState.matches,counts:ddarState.counts,truncated:ddarState.truncated}:null,auxiliary:auxState?.candidates||[],note:'Official AlphaGeometry1 or AlphaGeometry2 DDAR runs via Pyodide; no constructed proof trace is claimed; discovery is conjectural.'}}
async function copyText(text){try{await navigator.clipboard.writeText(text);notice('已复制到剪贴板')}catch(e){const t=document.createElement('textarea');t.value=text;document.body.appendChild(t);t.select();try{document.execCommand('copy');notice('已复制到剪贴板')}catch(e){notice('复制失败，请导出文件',true)}t.remove()}}
// Offline catalog is an indexed set of source constructions and proof goals.
// Raw source coordinates are absent; every load initializes independent degrees
// of freedom from deterministic generated seeds and runs the native Pretty solver.
let libraryShown=36;
function renderLibrary(){
 const data=window.GeomLibraryData,api=window.GeomLibrary,root=$('#library-results');if(!data||!api){root.innerHTML='<p class="empty-small">题库文件未能加载</p>';return;}
 $('#library-count').textContent=data.entries.length;
 const q=$('#library-search').value,group=$('#library-filter').value;
 const found=api.search(data,q,group,0,libraryShown);
 $('#library-result-count').textContent=found.total+' 条';
 root.innerHTML=found.entries.map(item=>`<button class="library-entry ${libraryCurrent===item.id?'on':''}" data-lib-id="${item.id}"><strong>${esc(item.title)}</strong><small>${esc(item.source)} · ${item.kind==='coordinates'?'原始坐标与约束':item.steps.length+' 道构造步骤'} · ${esc(item.sourcePath.split('/').pop())}</small></button>`).join('')+(found.total>libraryShown?'<button class="ghost-wide" id="library-more">再显示 36 条</button>':'')||'<p class="empty-small">没有匹配的条目</p>';
 $$('[data-lib-id]').forEach(b=>b.onclick=()=>{libraryCurrent=b.dataset.libId;renderLibrary();renderLibraryDetail()});
 const more=$('#library-more');if(more)more.onclick=()=>{libraryShown+=36;renderLibrary()};
}
function renderLibraryDetail(){const root=$('#library-details'),data=window.GeomLibraryData,entry=data?.entries.find(e=>e.id===libraryCurrent);
 root.hidden=!entry;if(!entry)return;
 const relations=entry.kind==='coordinates'?(entry.predicates||[]):(entry.steps||[]).map(s=>s.raw);
 const goals=(entry.goals||[]).map((g,i)=>`<div class="helper-note"><strong>目标 ${i+1} · AlphaGeometry 1：</strong><code>${esc(g.ag1)}</code></div><div class="helper-note"><strong>目标 ${i+1} · AlphaGeometry 2：</strong><code>${esc(g.ag2)}</code>${g.ag2Supported?'':' <small>（AG2 原生 DDAR 不支持此目标谓词）</small>'}</div>`).join('');
 root.innerHTML=`<h4>${esc(entry.title)}</h4><div class="helper-note">${esc(entry.source)} · ${esc(entry.sourcePath)} · ${entry.kind==='coordinates'?'原始坐标实例':entry.steps.length+' 条源构造'}</div>`+
 `<div class="library-relations">${relations.slice(0,140).map(esc).join('\n')}${relations.length>140?'\n…其余约束仍保存在题库数据内':''}</div>`+goals+
 `<button id="library-load" class="wide-btn library-load">${entry.kind==='coordinates'?'按原坐标载入':'载入构造并自动 Pretty'}</button>`+
 `<p id="library-load-note" class="helper-note">${entry.kind==='coordinates'?'此来源仅含点坐标与几何谓词；载入原坐标实例并显示约束，不会捏造构造依赖。':'按原语义逐条构造。未支持的联合约束会明确跳过，不会用假自由点顶替。'}</p>`;
 $('#library-load').onclick=()=>loadLibrary(entry);
}
function loadLibrary(entry){
 const api=window.GeomLibrary;
 if(!api){notice('题库模块不可用',true);return;}
 if(kernel.list().some(n=>!n.style.system)&&!confirm('载入题库将替换当前画板。确认继续？'))return;
 const result=api.convert(entry,{attempts:5});
 if(!result?.graph?.nodes?.length){notice('此题未能建立有效的原生图形，原始题目和目标仍可查看',true);return;}
 try{
  const trial=new Kernel(result.graph);
  // Never move source-provided coordinates. Only geometry without sample positions
  // receives GeomTool's existing Pretty numeric layout optimization.
  if(!Object.keys(entry.coords||{}).length){
   const controls=Pretty.controls(trial).length;
   if(controls){const iterations=controls<=14?31:controls<=30?15:7;
    const planned=Pretty.optimize(trial,{iterations,strength:58});
    if(planned.steps)Pretty.apply(trial,planned,100);
   }
  }
  prettyConditions=[];renderPrettyConditions();setLibraryVisuals(entry.id);kernel.load(trial.serialize());ensureReferences();
  $('#doc-name').value=entry.title.slice(0,36);selected=null;proofOverlay=null;steps=[];ghost=null;hoverCarriers=[];invalidatePretty();fit();render();requestSave();
  const note=$('#library-load-note');if(note)note.textContent=result.coordinateOnly?
   `原坐标载入 ${result.converted}/${result.total} 个点。原始 ${entry.predicates.length} 条谓词仅作文本展示，不与 DDAR 绑定。`:
   `原生作图已转换 ${result.converted}/${result.total} 条源构造；尚有 ${result.skipped.length} 条无法保真表示。${result.skipped.length?'未转换内容：'+result.skipped.slice(0,8).join(' | '):'全部源构造完成。'}`;
  notice(result.skipped.length?'题目已部分载入 · 请检查未转换步骤':'题库图形已载入');
 }catch(e){notice('题库载入失败：'+e.message,true);console.error(e);}
}
function librarySetup(){
 if(!window.GeomLibraryData)return;
 $('#library-search').addEventListener('input',()=>{libraryShown=36;renderLibrary()});
 $('#library-filter').addEventListener('change',()=>{libraryShown=36;renderLibrary()});renderLibrary();
}
function render(){if(selected&&!kernel.node(selected))selected=null;renderObjects();renderComplex();if($('#proof-goal'))renderProof();draw();}
/* v4.3 controller bridge: all state is invalidated when Kernel revision changes. */
let ddarState=null,auxState=null,ddarJob=0,ddarPage=1,activeDDAR=null,activeAux=-1,auxPreview=null;
let goalCatalog=null,goalPage=[],goalSelection=null,goalResult=null,goalJob=0;
function matchedMap(){return new Map((ddarState?.matches||[]).map(x=>[x.id,x.status]));}
function clearDDAR(){++ddarJob;++goalJob;window.GeomDDAR?.cancel?.();if($('#ddar-run')){$('#ddar-run').disabled=false;$('#ddar-run').textContent='⊢ 运行 DDAR 推理'}if($('#aux-run')){$('#aux-run').disabled=false;$('#aux-run').textContent='✦ 推测辅助构造'}ddarState=null;auxState=null;activeDDAR=null;activeAux=-1;auxPreview=null;goalCatalog=null;goalPage=[];goalSelection=null;goalResult=null;ddarPage=1;renderDDAR();renderAuxiliary();renderDDARGoals();}
function showDDARFact(f){activeDDAR=f.id;renderDDAR();const detail=$('#ddar-detail');if(!detail)return;
 const byId=new Map((ddarState?.facts||[]).map(x=>[x.id,x]));
 const chain=[],seen=new Set();function walk(q,depth=0){if(depth>7||seen.has(q))return;seen.add(q);const v=byId.get(q);if(!v){chain.push({depth,text:q,rule:'构造定义 / 支持前提'});return;}for(const p of v.premises||[])walk(p,depth+1);chain.push({depth,text:v.text,rule:v.rule});}
 walk(f.id);detail.hidden=false;detail.innerHTML=`<strong>原版 DDAR 验证 · ${esc(f.text)}</strong><span class="ddar-proof-tag">${ddarState?.proofTraceAvailable===false?'已验证（官方未提供逐步回溯）':'推导路径'}</span><div class="ddar-proof-steps">${chain.slice(-24).map((x,i)=>`<div><small>${i+1}. ${esc(x.rule)}</small><span>${esc(x.text)}</span></div>`).join('')}</div><button type="button" id="ddar-copy-detail">复制此证据链</button>`;
 $('#ddar-copy-detail').onclick=()=>copyText(chain.map((x,i)=>`${i+1}. [${x.rule}] ${x.text}`).join('\n'));
}
function renderDDAR(){const root=$('#ddar-list'),stat=$('#ddar-stats'),more=$('#ddar-more');if(!root)return;
 const facts=ddarState?.facts||[],q=($('#ddar-query')?.value||'').trim().toLowerCase(),kind=$('#ddar-type')?.value||'';
 const filtered=facts.filter(f=>(!kind||kind===f.kind)&&(!q||(f.text+' '+f.rule+' '+f.args.join(' ')).toLowerCase().includes(q)));
 const shown=filtered.slice(0,ddarPage*60),match=ddarState?.matches||[],proved=match.filter(x=>x.status==='proved').length,unknown=match.filter(x=>x.status==='unknown').length;
 stat.textContent=ddarState?`${facts.length.toLocaleString()} 条抽样确认 · ${proved} 条数值候选获形式证明 / ${unknown} 条尚未证明 · 闭包推理 · ${ddarState.elapsedMs} ms${ddarState.truncated?' · 触及安全上限':''}`:'尚未运行 · 不将数值发现冒充形式证明';
 root.innerHTML=shown.length?shown.map((f,i)=>`<button type="button" data-df="${i}" class="ddar-item ${f.id===activeDDAR?'on':''}"><span><b>${esc(({coll:'共线',para:'平行',perp:'垂直',cong:'等长',eqangleLines:'等夹角',midp:'中点',cyclic:'共圆',isos:'等腰',thales:'直角'}[f.kind]||f.kind))}</b><small>${esc(f.rule)}</small></span><strong>${esc(f.text)}</strong><small>${ddarState?.proofTraceAvailable===false?'官方内核闭包确认 · 无逐步证明':'证据链可展开'}</small></button>`).join(''):'<p class="empty-small">'+(ddarState?'没有匹配的已证结论':'点击“运行 DDAR 推理”')+'</p>';
 $$('#ddar-list [data-df]').forEach(b=>b.onclick=()=>showDDARFact(filtered[Number(b.dataset.df)]));
 more.hidden=shown.length>=filtered.length;more.textContent=`展开更多（${shown.length} / ${filtered.length}）`;
 if(!ddarState){$('#ddar-detail').hidden=true;return;}
}
function renderAuxiliary(){if(!$('#aux-list')||!$('#aux-stats'))return;const root=$('#aux-list'),stat=$('#aux-stats');if(!root)return;
 const suggestions=auxState?.candidates||[];
 stat.textContent=auxState?`生成 ${auxState.generated} 个可审查方向 · 按证明关联度排序，展示 ${suggestions.length} 个 · ${auxState.elapsedMs} ms`:'先运行 DDAR 或点击“推测辅助构造”';
 root.innerHTML=suggestions.length?suggestions.map((c,i)=>`<button class="aux-row ${i===activeAux?'on':''}" data-ai="${i}"><span><b>${esc(c.category)}</b><small>启发分 ${c.score}</small></span><strong>${esc(c.title)}</strong><small>${esc(c.why)}</small></button>`).join(''):'<p class="empty-small">'+(auxState?'暂时没有非重复、有效的辅助构造候选':'等待候选分析')+'</p>';
 $$('#aux-list [data-ai]').forEach(el=>el.onclick=()=>selectAuxiliary(Number(el.dataset.ai)));
 if(!auxState&&$('#aux-detail'))$('#aux-detail').hidden=true;
}
function explainAuxiliary(c){return [c.title,'结构依据：'+c.why,'形式安全：辅助构造只是建议。提交后由 Kernel 执行，并重新运行 DDAR 才能得到新增已证事实。',`Kernel: ${c.op} (${c.inputs.join(', ')})${c.op==='compoundIntersection'?'，自动分三步并原子提交':''}`].join('\n');}
function selectAuxiliary(i){const c=auxState?.candidates[i];if(!c)return;
 activeAux=i;auxPreview=null;let previewError='';
 try{const trial=new Kernel(kernel.serialize());const n=window.GeomAuxiliary.execute(trial,c);if(n.valid)auxPreview=n.geom;else previewError='此时构造失效';}catch(e){previewError=e.message;}
 renderAuxiliary();const root=$('#aux-detail');if(!root)return;root.hidden=false;
 root.innerHTML=`<strong>${esc(c.title)}</strong><p>${esc(c.why)}</p><code>${esc(c.op+' '+c.inputs.join(' '))}</code>${previewError?`<p class="aux-warning">${esc(previewError)}</p>`:''}<div class="aux-commands"><button type="button" id="aux-probe" ${previewError?'disabled':''}>预测 DDAR 增益</button><button type="button" id="aux-apply" ${previewError?'disabled':''}>加入画板</button></div><div class="aux-gain" id="aux-gain">点击“预测 DDAR 增益”进行一次隔离式试算，不修改原画板。</div>`;
 if(!previewError){$('#aux-apply').onclick=()=>applyAuxiliary(i);$('#aux-probe').onclick=()=>probeAuxiliary(i);}
 draw();
}
function applyAuxiliary(i){const c=auxState?.candidates[i];if(!c)return;try{let n;if(c.op==='compoundIntersection'){const trial=new Kernel(kernel.serialize());n=window.GeomAuxiliary.execute(trial,c);kernel.load(trial.serialize());}else{n=window.GeomAuxiliary.execute(kernel,c);}selected=n.id;objTab='details';$('#acc-objects').classList.add('open');notice(`已构造 ${n.id} · ${c.title}`);clearDDAR();render();}catch(e){notice('无法添加该辅助构造：'+e.message,true)}}
async function probeAuxiliary(i){const c=auxState?.candidates[i];if(!c)return;const target=$('#aux-gain'),rev=kernel.revision;target.textContent='正在隔离试算…';
 setTimeout(async()=>{try{const trial=new Kernel(kernel.serialize());window.GeomAuxiliary.execute(trial,c);
  const raw=(radarResult?.entries||[]).map(e=>({id:e.id,kind:e.kind,marks:e.marks,metric:e.metric}));
  const newer=await window.GeomDDAR.infer(trial.snapshot(),{entries:raw,maxQueries:120,engine:$('#ddar-engine')?.value||'ag2'});
  if(rev!==kernel.revision)return;
  const prev=new Set((ddarState?.facts||[]).map(f=>f.id)),novel=newer.facts.filter(f=>!prev.has(f.id));
  const oldProved=new Set((ddarState?.matches||[]).filter(x=>x.status==='proved').map(x=>x.id));
  const newlyMatched=newer.matches.filter(x=>x.status==='proved'&&!oldProved.has(x.id));
  target.textContent=`隔离试算：新增 ${novel.length} 条形式结论，额外匹配 ${newlyMatched.length} 条原数值发现。${novel.slice(0,4).map(x=>'\n· '+x.text).join('')||'\n目前没有找到额外非平凡的证明；该构造仍可能有几何价值。'}\n试算用时 ${newer.elapsedMs} ms。`;
 }catch(e){target.textContent='试算无法完成：'+e.message}},15);
}
async function runDDAR(makeAux=false,goal=null){const btn=$('#ddar-run'),auxBtn=$('#aux-run'),token=++ddarJob,rev=kernel.revision;
 btn.disabled=true;auxBtn.disabled=true;btn.textContent='⊢ 推理运行中…';if(makeAux)auxBtn.textContent='✦ 生成中…';
 const snapshot=kernel.snapshot(),entries=(radarResult?.entries||[]).map(e=>({id:e.id,kind:e.kind,marks:e.marks,metric:e.metric}));
 // v4.4 calls the persistent Pyodide Worker through GeomDDAR.infer;
 // there is deliberately NO fallback to the retired JS DDAR engine.
 async function compute(){
  const deduction=await window.GeomDDAR.infer(snapshot,{entries,goal,maxQueries:190,engine:$('#ddar-engine')?.value||'ag2',userPretty:prettyAG2(snapshot)});
  const aux=makeAux?window.GeomAuxiliary.suggest(snapshot,deduction,entries,{limit:25}):null;
  return {deduction,aux};
 }
 try{const result=await compute();if(token!==ddarJob||rev!==kernel.revision)return;ddarState=result.deduction;
  if(goal){goalResult=ddarState.goal;renderDDARGoals();}
  if(makeAux){auxState=result.aux;activeAux=-1;auxPreview=null;renderAuxiliary();}
  ddarPage=1;activeDDAR=null;renderDDAR();renderRadar();draw();notice(`${ddarState.version}：已形式确认 ${ddarState.facts.length} 条抽样关系`);
 }catch(e){if(token===ddarJob){notice('DDAR 分析失败：'+e.message,true);console.error(e)}}
 finally{if(token===ddarJob){btn.disabled=false;auxBtn.disabled=false;btn.textContent='⊢ 运行 DDAR 推理';auxBtn.textContent='✦ 推测辅助构造';}}
}
function renderDDARGoals(){
 const sel=$('#ddar-goal-select'),stat=$('#ddar-goal-stats'),output=$('#ddar-goal-result');if(!sel)return;
 const all=goalCatalog?.goals||[],query=($('#ddar-goal-search')?.value||'').trim().toLowerCase();
 goalPage=all.filter(g=>!query||(g.text+' '+g.kind+' '+g.args.join(' ')).toLowerCase().includes(query)).slice(0,1200);
 const active=goalSelection?.id||sel.value;
 sel.innerHTML='<option value="">'+(goalCatalog?'选择目标（当前展示前 '+goalPage.length+' 条）':'请先发掘目标')+'</option>'+goalPage.map((g,i)=>`<option value="${i}">${esc(g.text)}</option>`).join('');
 const idx=goalPage.findIndex(g=>g.id===active);sel.value=idx>=0?String(idx):'';
 stat.textContent=goalCatalog?`找到 ${all.length} 个待证命题（筛选后 ${goalPage.length} 个可选） · ${goalCatalog.points} 点 / ${goalCatalog.lines} 线候选 · ${goalCatalog.elapsedMs} ms${goalCatalog.truncated?' · 达到枚举安全上限':''}`:'还未发掘待证命题';
 $('#ddar-prove-goal').disabled=!goalSelection;
 if(goalResult){const g=goalResult;
  const proof=(g.evidence||[]).map(id=>{
   const f=ddarState?.facts.find(x=>x.id===id);
   return f?'· '+f.rule+': '+f.text:'· '+id;
  });
  output.textContent=(g.status==='proved'?'✓ DDAR 已证':g.status==='unsupported'?'! 当前后端暂不支持':'? 尚未证明')+' · '+g.text+'\n'+g.reason+
    (proof.length?'\n相关证据：\n'+proof.slice(0,14).join('\n'):'')+
    (ddarState?.truncated?'\n推理触及事实安全上限，未知不代表假。':'');
  output.classList.toggle('success',g.status==='proved');
 }else{output.textContent=goalSelection?'待证：'+goalSelection.text+'\n点击“全力搜索选中目标”查询证明。':'发掘并选择一个待证目标。';output.classList.remove('success');}
}
async function discoverDDARGoals(){
 const btn=$('#ddar-discover-goals'),token=++goalJob,rev=kernel.revision,snapshot=kernel.snapshot();
 btn.disabled=true;btn.textContent='⌕ 正在筛选命题…';
 try{
  let result;
  if(typeof Worker!=='undefined'&&location.protocol!=='file:'){
   try{result=await new Promise((resolve,reject)=>{const w=new Worker('assets/ddar.worker.js');
    w.onmessage=e=>{w.terminate();e.data.error?reject(Error(e.data.error)):resolve(e.data.goals)};
    w.onerror=e=>{w.terminate();reject(Error(e.message))};
    w.postMessage({token,snapshot,discoverGoals:true});});}catch(e){console.warn('Goals worker fallback',e)}
  }
  if(!result)result=await new Promise(resolve=>setTimeout(()=>resolve(window.GeomDDARGoals.discover(snapshot,{maxGoals:2200})),15));
  if(token!==goalJob||rev!==kernel.revision)return;
  goalCatalog=result;goalSelection=null;goalResult=null;renderDDARGoals();
 }catch(e){if(token===goalJob)notice('目标发掘失败：'+e.message,true)}
 finally{if(token===goalJob){btn.disabled=false;btn.textContent='⌕ 发掘可设置的待证目标';}}
}
function addManualDDARGoal(){
 try{const g=window.GeomDDARGoals.parseGoal($('#ddar-goal-manual').value,kernel.snapshot());
  if(!goalCatalog)goalCatalog={goals:[],points:0,lines:0,elapsedMs:0,truncated:false};
  if(!goalCatalog.goals.some(x=>x.id===g.id))goalCatalog.goals.unshift(g);
  $('#ddar-goal-search').value='';goalSelection=g;goalResult=null;renderDDARGoals();
 }catch(e){notice('目标语法错误：'+e.message,true);$('#ddar-goal-result').textContent=e.message;}
}
function chooseDDARGoal(){const sel=$('#ddar-goal-select'),i=Number(sel.value);
 goalSelection=sel.value!==''?goalPage[i]||null:null;goalResult=null;renderDDARGoals();
}
function setupDDAR(){const a=$('#ddar-run');if(!a)return;
 a.onclick=()=>runDDAR(false);$('#aux-run').onclick=()=>runDDAR(true);
 $('#ddar-discover-goals').onclick=discoverDDARGoals;$('#ddar-add-goal').onclick=addManualDDARGoal;$('#ddar-goal-search').oninput=renderDDARGoals;
 $('#ddar-goal-select').onchange=chooseDDARGoal;$('#ddar-prove-goal').onclick=()=>goalSelection&&runDDAR(false,goalSelection);
 $('#ddar-engine').onchange=()=>{clearDDAR();};
 $('#ddar-query').oninput=()=>{ddarPage=1;renderDDAR()};$('#ddar-type').onchange=()=>{ddarPage=1;renderDDAR()};
 $('#ddar-more').onclick=()=>{ddarPage++;renderDDAR()};
 $('#split-analysis').onclick=()=>{settings.ddarSplit=!settings.ddarSplit;if(settings.ddarSplit){settings.rightWidth=Math.max(630,settings.rightWidth||312);settings.rightCollapsed=false;$('#acc-relations').classList.add('open');$('#acc-proof').classList.add('open');}else{settings.rightWidth=Math.min(settings.rightWidth,390);}applyPanels();requestSave();};
 renderDDAR();renderAuxiliary();renderDDARGoals();
}

/* v4.6: focused DDAR UI controller. Original auxiliary-construction functions
 * above remain in source, but their UI has been removed intentionally.
 */
let v45GoalShown=false,v45Task=null,v45RunId=0,v45Paused=false,v45Compiled=null,v45Ag1=null,v45LastPaint=0,v45GoalTimer=null,v45RunState='idle';
const v45Aliases={cong:'等长 等距 距离相等 长度相等 相等 equal length',para:'平行 平行线 parallel',perp:'垂直 正交 perpendicular orthogonal',coll:'共线 同线 collinear',cyclic:'共圆 同圆 concyclic circle',midp:'中点 中心 midpoint',eqratio:'比例 线段比 距离比 ratio',eqangleLines:'等角 角相等 角度相等 angle',angconst:'定角 固定角度',isos:'等腰 等边',onl:'在线上 点在线上',onc:'在圆上 点在圆上'};
function v45String(g){
 if(!g)return '';
 let text;
 if($('#ddar-engine').value==='ag1')text=window.GeomAG1Compiler.goal(g,v45Ag1||{nameMap:{},lines:{}});
 else text=window.GeomDDAR.transformGoal(g,v45Compiled||{lines:{},points:[],circles:{}})?.predicates?.map(p=>p.name+' '+p.points.join(' ')+(p.constants?.length?' '+p.constants.join(' '):'')).join(' ; ');
 // AG1's internal p0/p1 names are mapped back to human-visible GeomTool names.
 if(text&&v45Ag1){const rev=Object.fromEntries(Object.entries(v45Ag1.nameMap).map(([k,v])=>[v,k]));text=text.replace(/\bp\d+\b/g,m=>rev[m]||m);}
 return text||'不支持该形式目标';
}
function v45Matches(g,q){
 if(!q)return 1;
 const src=(g.kind+' '+g.args.join(' ')+' '+v45String(g)+' '+(g.text||'')+' '+(v45Aliases[g.kind]||'')).toLowerCase();
 const terms=q.toLowerCase().trim().split(/\s+/).filter(Boolean);
 let score=0;
 for(const token of terms){const ix=src.indexOf(token);if(ix>=0){score+=100-Math.min(ix,80);continue;}
  let pos=0;for(const c of token)if((pos=src.indexOf(c,pos))>=0)pos++;else return -1;
  score+=7;
 }
 return score;
}
function clearDDAR(){
 proofOverlay=null;++ddarJob;++goalJob;++v45RunId;window.GeomDDAR?.cancel?.();v45Task=null;v45Paused=false;v45RunState='idle';v45Compiled=null;v45Ag1=null;v45LastPaint=0;
 ddarState=null;auxState=null;goalCatalog=null;goalSelection=null;goalResult=null;activeDDAR=null;ddarPage=1;
 if($('#ddar-run'))$('#ddar-run').disabled=false;
 if($('#ddar-prove-goal'))$('#ddar-prove-goal').disabled=true;
 if($('#ddar-toggle')){$('#ddar-toggle').disabled=true;$('#ddar-toggle').textContent='停止搜索';}
 renderDDAR();renderDDARGoals();renderRadar();draw();
}
function renderDDARGoals(){
 const sel=$('#ddar-goal-select'),stat=$('#ddar-goal-stats'),output=$('#ddar-goal-result');if(!sel)return;
 const catalog=goalCatalog?.goals||[],search=($('#ddar-goal-manual')?.value||'').trim();
 let ranked=[];
 for(const g of catalog){const score=v45Matches(g,search);if(score>=0)ranked.push({g,score});}
 ranked.sort((a,b)=>b.score-a.score||(a.g.kind+a.g.args.join('')).localeCompare(b.g.kind+b.g.args.join('')));
 goalPage=ranked.slice(0,180).map(x=>x.g);
 sel.innerHTML=goalPage.map((g,i)=>`<option value="${i}">${esc(v45String(g))}</option>`).join('');
 const idx=goalPage.findIndex(x=>x.id===goalSelection?.id);if(idx>=0)sel.value=String(idx);else sel.selectedIndex=-1;
 stat.textContent=goalCatalog?`${catalog.length.toLocaleString('zh-CN')} 条形式命题 · 当前匹配 ${ranked.length.toLocaleString('zh-CN')} 条${ranked.length>180?' · 请继续输入以缩小范围':''}`:'尚未发掘形式命题';
 $('#ddar-prove-goal').disabled=!goalSelection||v45RunState==='running'||v45RunState==='paused';
 if(goalSelection&&goalResult){
  const status=goalResult.status==='proved'?'✓ 已证':goalResult.status==='unsupported'?'不支持':'尚未证明';
  output.textContent=status+' · '+v45String(goalSelection)+'\n'+(goalResult.reason||'');
  output.classList.toggle('success',goalResult.status==='proved');
 }else{
  output.textContent=goalSelection?'当前目标：'+v45String(goalSelection):'在搜索框中输入中文、形式谓词或点名，选择匹配结果作为目标。';
  output.classList.remove('success');
 }
}
function chooseDDARGoal(){const i=Number($('#ddar-goal-select').value);goalSelection=$('#ddar-goal-select').value!==''?goalPage[i]||null:null;goalResult=null;renderDDARGoals();}
function v45NormalizeInput(){
 const input=$('#ddar-goal-manual');const src=input.value.trim();if(!src)return;
 try{
  const g=window.GeomDDARGoals.parseGoal(src,kernel.snapshot());
  if(!goalCatalog)goalCatalog={goals:[],points:0,lines:0,elapsedMs:0};
  if(!goalCatalog.goals.some(x=>x.id===g.id))goalCatalog.goals.push(g);
  goalSelection=g;goalResult=null;input.value='';renderDDARGoals();
 }catch(e){
  // A fuzzy Chinese query is not a grammar error: retain its ranked matches.
  if(!goalPage.length)notice('无匹配命题：'+e.message,true);
 }
}
async function discoverDDARGoals(){
 const btn=$('#ddar-discover-goals'),token=++goalJob,rev=kernel.revision,snapshot=kernel.snapshot();
 btn.disabled=true;btn.textContent='发掘中…';
 try{
  let results;
  if(typeof Worker!=='undefined'){
   results=await new Promise((resolve,reject)=>{const w=new Worker('assets/ddar.worker.js');
    w.onmessage=e=>{w.terminate();e.data.error?reject(Error(e.data.error)):resolve(e.data.goals)};
    w.onerror=e=>{w.terminate();reject(Error(e.message||'发现 Worker 错误'))};w.postMessage({token,snapshot,discoverGoals:true});
   });
  }else results=window.GeomDDARGoals.discover(snapshot);
  if(token!==goalJob||rev!==kernel.revision)return;
  v45Compiled=window.GeomDDAR.compile(snapshot);
  v45Ag1=window.GeomAG1Compiler.compile(snapshot,v45Compiled);
  // Do not display a goal that cannot be expressed by the selected engine.
  results.goals=results.goals.filter(g=>$('#ddar-engine').value==='ag1'?!!window.GeomAG1Compiler.goal(g,v45Ag1):!!window.GeomDDAR.transformGoal(g,v45Compiled).predicates.length);
  goalCatalog=results;goalResult=null;goalSelection=null;renderDDARGoals();
 }catch(e){if(token===goalJob)notice('目标发掘失败：'+e.message,true)}
 finally{if(token===goalJob){btn.disabled=false;btn.textContent='发掘结论';}}
}
function renderDDAR(){
 const root=$('#ddar-list'),stat=$('#ddar-stats');if(!root)return;
 const data=ddarState?.facts||[],query=($('#ddar-query')?.value||'').trim(),filtered=data.filter(f=>{const shown=displayDDARFact(f);return v45Matches({kind:shown.kind,args:shown.args,text:shown.text},query)>=0;});
 const count=ddarPage*90,shown=filtered.slice(0,count),oldTop=root.scrollTop;
 root.innerHTML=shown.length?shown.map((f,i)=>`<button type="button" data-df="${i}" class="ddar-item ${activeDDAR===f.id?'on':''}"><span><b>${esc(f.kind)}</b><small>${esc(f.rule||'DDAR')}</small></span><strong>${esc(displayDDARFact(f).text)}</strong></button>`).join(''):'<p class="empty-small">'+(ddarState?'暂无符合查询的已证结论':'尚无已证结论')+'</p>';
 root.scrollTop=oldTop;
 root.querySelectorAll('[data-df]').forEach(b=>b.onclick=()=>showDDARFact(shown[Number(b.dataset.df)]));
 renderRadar();
 const p=ddarState?.progress;
 if(p)stat.textContent=`${v45RunState==='paused'?'已暂停 · ':v45RunState==='running'?'推理中 · ':''}${p.engine?.toUpperCase()} · 已证 ${data.length}（候选池 ${p.totalCandidates}） · 已检查 ${p.checked} · 查询尝试 ${p.attempts} · 深度 ${p.depth} · ${p.phase} · ${(p.elapsedMs/1000).toFixed(1)}s${p.done?' · 已终止':''}`;
 else stat.textContent='等待推理';
}
async function showDDARFact(f){
 activeDDAR=f.id;proofOverlay=null;const panel=$('#ddar-detail');panel.hidden=false;
 let chain=null;
 if(v45Task){try{chain=await v45Task.proof(f.queryId||f.id)}catch(e){chain={available:false,reason:e.message}}}
 if(activeDDAR!==f.id)return;
 const readable=displayDDARFact(f);
 const steps=chain?.available?chain.steps:[];
 const reverse=new Map();for(const [name,internal] of Object.entries(v45Task?.prepared?.ag1?.nameMap||{}))
  if(!reverse.has(internal)||(!name.startsWith('@')&&reverse.get(internal).startsWith('@')))reverse.set(internal,name);
 const human=t=>$('#ddar-engine').value==='ag1'?String(t).replace(/\bp\d+\b/g,x=>reverse.get(x)||x):String(t);
 const rows=steps.map((x,i)=>`<button type="button" class="v5-proof-step" data-proof-step="${i}" aria-pressed="false" title="点击在画板中高亮该步"><small>#${esc(x.id)} · ${esc(v52DisplayRule(x.rule))}${x.premises?.length?' · 依据 '+esc(x.premises.map(p=>'#'+p).join(', ')):''}${x.complete===false||x.dependencyLevel?.startsWith('macro')?' · 宏级/未完全追踪':''}</small><span>${esc(v52DisplayStep(human(x.predicate)))}</span></button>`).join('');
 const pending=chain?.reason||'该结论已证，但目前不能恢复逐项证明步骤。';
 const fallback=`<button type="button" class="v5-proof-step" data-proof-step="goal"><small>已证明的最终结论</small><span>${esc(readable.text||f.id)}</span></button>`;
 panel.innerHTML=`<strong>${esc(readable.text)}</strong><span class="ddar-proof-tag">${esc(f.rule||'DDAR')} · 推理深度 ${f.depth||0}</span>${chain?.certificate?`<small>${esc(chain.certificate)}</small>`:''}<div class="ddar-proof-steps">${rows||fallback}${!rows?`<div>${esc(pending)}</div>`:''}</div>`;
 panel.querySelectorAll('[data-proof-step]').forEach(button=>button.onclick=()=>{
  const n=button.dataset.proofStep,text=n==='goal'?readable.text:human(steps[Number(n)]?.predicate||'');
  const entry=n==='goal'?proofEntryFromFact(f):proofOverlayFromPredicate(v52OverlayEquivalent(text));
  proofOverlay=entry;
  panel.querySelectorAll('[data-proof-step]').forEach(x=>{x.classList.toggle('on',x===button);x.setAttribute('aria-pressed',String(x===button));});
  draw();
 });
 // Also allow clicking the theorem itself, even if the engine has no DAG.
 const first=panel.querySelector('[data-proof-step]');if(first)first.click();
}
async function runDDAR(makeAux=false,goal=null){
 const token=++v45RunId,rev=kernel.revision;
 v45GoalShown=false;
 if(!goalCatalog)await discoverDDARGoals();
 if(token!==v45RunId||rev!==kernel.revision)return;
 if(!goal&&!goalSelection)goalResult=null;
 const snapshot=kernel.snapshot(),catalog=goalCatalog?.goals||[];
 v45RunState='running';v45Paused=false;v45LastPaint=0;$('#ddar-toggle').disabled=false;$('#ddar-toggle').textContent='停止搜索';
 $('#ddar-run').disabled=true;$('#ddar-prove-goal').disabled=true;
 try{
  const task=window.GeomDDAR.startSession(snapshot,{
   engine:$('#ddar-engine').value,goals:catalog,goal,mode:goal?'goal':'closure',userPretty:prettyAG2(snapshot),
   onProgress:(state,event)=>{
    if(token!==v45RunId||rev!==kernel.revision)return;
    ddarState=state;
    if(goal){goalResult=state.goal;if(state.goal?.status==='proved'&&!v45GoalShown){v45GoalShown=true;showDDARFact(state.facts.find(f=>f.queryId===goal.id)||{id:goal.id,text:v45String(goal),queryId:goal.id,rule:'DDAR'});}}
    if(event==='paused')v45RunState='paused';
    if(event==='complete')v45RunState='complete';
    if(Date.now()-v45LastPaint>950||event!=='progress'){
     v45LastPaint=Date.now();renderDDAR();renderDDARGoals();
    }
   },onError:e=>{if(token===v45RunId)notice('推理失败：'+e.message,true)}
  });
  v45Task=task;
  if(task.prepared)task.prepared.startedRevision=rev;
  if(!$('#v51-submission-panel').hidden)v51UpdateSubmission(task.prepared);
  await task.completion;
  if(!$('#v52-audit-panel').hidden)await v52ShowAudit();
  if(makeAux&&token===v45RunId){
   const entries=(radarResult?.entries||[]).map(e=>({id:e.id,kind:e.kind,marks:e.marks,metric:e.metric}));
   auxState=window.GeomAuxiliary.suggest(snapshot,ddarState,entries,{limit:25});
   renderAuxiliary();
  }
 }catch(e){if(token===v45RunId){v45RunState='error';notice('DDAR 推理失败：'+e.message,true);}}
 finally{if(token===v45RunId){$('#ddar-run').disabled=false;$('#ddar-toggle').disabled=true;v45RunState='complete';renderDDAR();renderDDARGoals();}}
}
let v52AuditText='';
async function v52ShowAudit(){
 const pre=$('#v52-audit-text');if(!pre)return;
 if(!v45Task||v45Task.prepared?.backend!=='ag2'){pre.textContent='请先选择 AG2 并运行证明或推理闭包。';return;}
 try{const log=await v45Task.audit();v52AuditText=log.content||'';
  const lines=v52AuditText.trim().split('\n').filter(Boolean);
  pre.textContent=lines.map(line=>{try{const x=JSON.parse(line);if(x.type==='INPUT'&&x.points)
   return `[${x.id}] INPUT：点 ${x.points.length} 个；作图前提 ${(x.premises||[]).length} 条\n`+
    (x.premises||[]).map((p,i)=>`  假设${i+1} ${p.name} ${(p.points||[]).join(' ')} ${(p.constants||[]).join(' ')} 来源:${p.source||'作图'}`).join('\n');
   return `[${x.id}] ${x.type} · ${x.rule||x.stage||''}\n  结论：${(x.conclusions||[]).join('；')||x.predicate||''}\n  依据：${(x.premises||[]).map(p=>typeof p==='object'?JSON.stringify(p):p).join('；')|| (x.type==='INPUT'?'输入公设':'未追踪（不可视为完整证明）')}${x.complete===false?'\n  ⚠ 此条无法追踪全部前提':''}`;
  }catch(e){return line}}).join('\n\n');
 }catch(e){pre.textContent='读取日志失败：'+e.message;}
}
function prettyAG2(snapshot){
 if(!prettyConditions.length)return {predicates:[],limitations:[]};
 const base=window.GeomDDARCompiler.compile(snapshot);
 return window.GeomPrettyConstraints.ag2(prettyConditions,kernel,base);
}
let v51SubmissionText='';
function v51UpdateSubmission(prepared=null){
 const panel=$('#v51-submission-panel');if(!panel)return;
 try{
  const compiled=prepared?.compiled||window.GeomDDARCompiler.compile(kernel.snapshot());
  if(!prepared&&prettyConditions.length){const extra=prettyAG2(kernel.snapshot());compiled.predicates.push(...extra.predicates);compiled.limitations.push(...extra.limitations);}
  const actualAG2=prepared?.backend==='ag2';
  const parts=[
   '# GeomTool v5.4 → AlphaGeometry 2 / DDAR',
   '# '+(actualAG2?'此次推理实际提交的编译内容':'根据当前画板新编译的 AG2 预览（尚未运行）'),
   '# 作图前提来自构造定义；USER-PRETTY 是用户显式声明、经精度校验的附加假设（非已证明结论）。',
   '', '# 点坐标 / POINTS ('+compiled.points.length+')',
   ...compiled.points.map(x=>[x.virtual?'AUX':'POINT',x.name,JSON.stringify(x.value)].join(' ')),
   '', '# 虚拟点用途 / INTERNAL WITNESSES',
   ...(compiled.virtualDescriptions||[]).map(x=>x.id+' = '+x.reason),
   '', '# 构造前提 / ASSUMPTIONS ('+compiled.predicates.length+')',
   ...compiled.predicates.map((p,i)=>String(i+1).padStart(4,'0')+'. '+p.name+' '+p.points.join(' ')+(p.constants?.length?' '+p.constants.join(','):'')+'  // '+p.source),
   '', '# 派生直线 → 两点表示 / LINE CARRIERS',
   ...Object.entries(compiled.lines).map(([name,p])=>name+': '+p.join(' ')),
   '', '# 源圆定义 / CIRCLE DEPENDENCIES',
   ...Object.entries(compiled.circles).map(([name,c])=>name+': '+c.op+' '+c.inputs.join(' ')+'; center='+String(c.center||'?')+'; members='+c.points.join(' ')),
   '', '# 尚未精确表达的条件 / LIMITATIONS',
   ...(compiled.limitations.length?compiled.limitations:['无']),
   '', '# 注意：AG2 使用上面的坐标决定方向和非退化分支；分支选择不等于额外形式证明。'
  ];
  if(prepared?.backend==='ag2')parts.push('', '# AG2 目标与探知查询',...prepared.queries.slice(0,600).map(q=>q.id+': '+(q.predicates||[]).map(p=>p.name+' '+p.points.join(' ')+(p.constants?.length?' '+p.constants.join(','):'')).join(' & ')));
  v51SubmissionText=parts.join('\n');$('#v51-submission-text').textContent=v51SubmissionText;
 }catch(err){v51SubmissionText='编译失败：'+err.message;$('#v51-submission-text').textContent=v51SubmissionText;}
}
function v51Download(name,text,mime='application/x-ndjson;charset=utf-8'){
 const file=new Blob([text],{type:mime}),url=URL.createObjectURL(file),a=document.createElement('a');
 a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1500);
}
function setupDDAR(){
 $('#v52-audit-toggle').onclick=()=>{const box=$('#v52-audit-panel');box.hidden=!box.hidden;$('#v52-audit-toggle').setAttribute('aria-expanded',String(!box.hidden));if(!box.hidden)v52ShowAudit();};
 $('#v52-audit-refresh').onclick=v52ShowAudit;
 $('#v52-audit-copy').onclick=()=>copyText(v52AuditText);
 $('#v51-submission-toggle').onclick=()=>{
  const p=$('#v51-submission-panel'),opening=p.hidden;p.hidden=!opening;
  $('#v51-submission-toggle').setAttribute('aria-expanded',String(opening));
  if(opening)v51UpdateSubmission(v45Task?.prepared?.backend==='ag2'&&v45Task?.prepared?.startedRevision===kernel.revision?v45Task.prepared:null);
 };
 $('#v51-submission-copy').onclick=()=>copyText(v51SubmissionText);
 $('#v51-audit-download').onclick=async()=>{try{
  if(!v45Task||v45Task.prepared?.backend!=='ag2')throw Error('请先运行 AlphaGeometry 2');
  const log=await v45Task.audit();if(!log.available)throw Error(log.reason||'尚无日志');
  v51Download(log.filename||'geomtool_ag2_proof_log.jsonl',log.content);
  if(log.truncated)notice('日志事件已达到上限，文件内有截断说明');
 }catch(e){notice('日志下载失败：'+e.message,true)}};
 $('#ddar-run').onclick=()=>runDDAR(false);
 $('#ddar-prove-goal').onclick=()=>goalSelection&&runDDAR(false,goalSelection);
 $('#ddar-toggle').onclick=()=>{if(!v45Task)return;v45Paused=!v45Paused;if(v45Paused){v45Task.pause();v45RunState='paused';}else{v45Task.resume();v45RunState='running';}$('#ddar-toggle').textContent=v45Paused?'继续搜索':'停止搜索';renderDDAR();};
 $('#ddar-engine').onchange=()=>clearDDAR();
 $('#ddar-discover-goals').onclick=discoverDDARGoals;
 $('#ddar-goal-manual').oninput=()=>{clearTimeout(v45GoalTimer);v45GoalTimer=setTimeout(renderDDARGoals,100)};
 $('#ddar-goal-manual').onkeydown=e=>{if(e.key==='Enter'){e.preventDefault();v45NormalizeInput();}};
 $('#ddar-goal-select').onchange=chooseDDARGoal;
 $('#ddar-query').oninput=()=>{ddarPage=1;renderDDAR();};
 $('#ddar-list').onscroll=e=>{const list=e.target;if(list.scrollTop+list.clientHeight>list.scrollHeight-35){ddarPage++;renderDDAR();}};
 if($('#split-analysis'))$('#split-analysis').onclick=()=>{settings.ddarSplit=!settings.ddarSplit;applyPanels();requestSave();};
 renderDDAR();renderDDARGoals();
}

function bind(){initSettings();panelSetup();prettySetup();librarySetup();setupConsole();setupDDAR();$('#theme-toggle').onclick=()=>{settings.theme=settings.theme==='dark'?'light':'dark';syncPalette();draw();requestSave()};radarOptions();$$('.rail-btn[data-group]').forEach(b=>b.onclick=()=>{groupUI(b.dataset.group);activate(groups[b.dataset.group].tools[0][0]);if(window.innerWidth<=620)$('#tools-panel').classList.add('mobile-tools-show')});$$('.accordion-trigger').forEach(b=>b.onclick=()=>{$('#'+b.dataset.accordion).classList.toggle('open')});$$('[data-objtab]').forEach(b=>b.onclick=()=>{objTab=b.dataset.objtab;renderObjects()});$('#radar-run').onclick=scanRadar;if($('#proof-check'))$('#proof-check').onclick=proofCheck;if($('#proof-goal'))$('#proof-goal').onchange=()=>{$('#proof-output').textContent='选择目标后运行检查';$('#proof-output').classList.remove('success')};if($('#proof-export'))$('#proof-export').onclick=()=>copyText(JSON.stringify(proofPayload(),null,2));$('#undo-btn').onclick=()=>{kernel.undo();render()};$('#redo-btn').onclick=()=>{kernel.redo();render()};$('#fit-btn').onclick=fit;
 $('#export-btn').onclick=()=>{const contents=JSON.stringify({schema:'geomtool.document/v1',name:$('#doc-name').value,graph:kernel.serialize(),prettyConditions:prettyConditions.map(c=>c.text),settings:{complex:settings.complex,axes:settings.axes}},null,2);const blob=new Blob([contents],{type:'application/json;charset=utf-8'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download=($('#doc-name').value.trim()||'GeomTool')+'.geom.json';a.click();setTimeout(()=>URL.revokeObjectURL(a.href),500);notice('文件已导出')};
 $('#import-btn').onclick=()=>$('#file-input').click();$('#file-input').onchange=async e=>{const f=e.target.files?.[0];if(!f)return;try{if(f.size>2e6)throw Error('文件大小不能超过 2MB');importJSONDocument(await f.text());notice('导入成功')}catch(err){notice('导入失败：'+err.message,true)}e.target.value=''};
 $('#doc-name').addEventListener('input',requestSave);
 $('#console-toggle').onclick=()=>{$('#console').classList.toggle('closed');if(!$('#console').classList.contains('closed'))$('#console-input').focus()};$('#terminal-rail').onclick=()=>{$('#console').classList.remove('closed');$('#console-input').focus()};$('#console-form').onsubmit=e=>{e.preventDefault();const input=$('#console-input'),line=input.value.trim();if(!line)return;consoleHistory.push(line);historyIndex=consoleHistory.length;log('› '+line,'log-ok');try{const result=runConsole(line);if(result!==undefined&&result!==null)log(showConsoleResult(result),'log-muted');render()}catch(err){log('✕ '+err.message,'log-err');notice(err.message,true)}input.value='';$('#console-suggestions').hidden=true;};
 $('#console-input').onkeydown=e=>{if(e.key==='ArrowUp'&&consoleHistory.length){historyIndex=Math.max(0,historyIndex-1);e.target.value=consoleHistory[historyIndex];e.preventDefault()}if(e.key==='ArrowDown'){historyIndex=Math.min(consoleHistory.length,historyIndex+1);e.target.value=consoleHistory[historyIndex]||'';e.preventDefault()}};
 const modal=$('#settings-modal');const showModal=on=>{modal.hidden=!on};$('#settings-rail').onclick=()=>showModal(true);$('#settings-close').onclick=()=>showModal(false);$('#settings-done').onclick=()=>showModal(false);modal.addEventListener('mousedown',e=>{if(e.target===modal)showModal(false)});$('#new-board').onclick=()=>{if(confirm('清空当前画板？此操作可撤销。')){setLibraryVisuals(null);prettyConditions=[];renderPrettyConditions();command({op:'clear'},true);selected=null;radarResult=null;radarEntry=null;showModal(false);render()}};$('#load-demo').onclick=()=>{if(confirm('以示例图形替换当前画板？')){setLibraryVisuals(null);prettyConditions=[];renderPrettyConditions();const k=new Kernel();const cur=kernel;kernel=k;seed();const sample=k.serialize();kernel=cur;kernel.load(sample);ensureReferences();selected=null;fit();render();showModal(false)}};
 $('#mobile-side-toggle').onclick=()=>$('#inspector').classList.toggle('mobile-show');
 window.addEventListener('keydown',e=>{if(e.key==='Escape'&&!$('#console-help-modal').hidden){modalCloseHelp();return;}const target=e.target;if(target instanceof HTMLInputElement||target instanceof HTMLSelectElement||target instanceof HTMLTextAreaElement)return;
  const mod=e.metaKey||e.ctrlKey;if(mod&&e.key.toLowerCase()==='z'){e.preventDefault();e.shiftKey?kernel.redo():kernel.undo();render();return;}if(mod&&e.key.toLowerCase()==='y'){e.preventDefault();kernel.redo();render();return;}if(e.key==='Escape'){if(!$('#console-help-modal').hidden){modalCloseHelp();return;}if(!modal.hidden){showModal(false);return}steps=[];activate('select');return;}if(e.code==='Space'){spaceDown=true;e.preventDefault()}if(e.key==='Enter'&&activeTool==='polygon'&&steps.length>=3){finalizeTool(mouse);return}if(e.key==='Delete'||e.key==='Backspace'){if(selected){command({op:'delete',id:selected},true);selected=null;render()}}});window.addEventListener('keyup',e=>{if(e.code==='Space')spaceDown=false});
 window.addEventListener('resize',()=>{applyPanels();resize()});if(typeof ResizeObserver!=='undefined')new ResizeObserver(resize).observe($('#board-wrap'));pointerSetup();kernel.subscribe(ev=>{if(ev.type==='preview'){if(radarResult){radarEntry=null;radarResult=null;renderRadar();}draw();return;}if(ev.type==='clear')ensureReferences();renderPrettyConditions();if(!$('#v51-submission-panel').hidden)v51UpdateSubmission();if(ev.type!=='pretty')invalidatePretty();radarEntry=null;radarResult=null;clearDDAR();renderRadar();if($('#proof-output')){$('#proof-output').textContent='图形已更新，请重新检查目标。';$('#proof-output').classList.remove('success');}requestSave();$('#undo-btn').disabled=!kernel.undoStack.length;$('#redo-btn').disabled=!kernel.redoStack.length;render()});
}

function applyPanels(){const wide=window.innerWidth>900,leftDesktop=window.innerWidth>620;
 document.documentElement.style.setProperty('--tools-width',Math.max(170,Math.min(500,Number(settings.leftWidth)||228))+'px');
 document.documentElement.style.setProperty('--inspector-width',Math.max(240,Math.min(800,Number(settings.rightWidth)||312))+'px');
 $('#tools-panel').classList.toggle('collapsed',!!settings.leftCollapsed&&leftDesktop);
 $('#inspector').classList.toggle('collapsed',!!settings.rightCollapsed&&wide);$('#inspector').classList.toggle('split-view',!!settings.ddarSplit&&wide);$('#split-analysis')?.classList.toggle('split-on',!!settings.ddarSplit);
 $('#left-resizer').hidden=!!settings.leftCollapsed||!wide;
 $('#right-resizer').hidden=!!settings.rightCollapsed||!wide;
 $('#restore-inspector').hidden=!settings.rightCollapsed||!wide;
 $('#toggle-tools').title=settings.leftCollapsed?'展开工具侧栏':'收起工具侧栏';
 resize();
}
function panelSetup(){
 const toggleLeft=()=>{if(window.innerWidth<=620){$('#tools-panel').classList.toggle('mobile-tools-show');return;}settings.leftCollapsed=!settings.leftCollapsed;applyPanels();requestSave()};
 const toggleRight=()=>{if(window.innerWidth<=900){$('#inspector').classList.toggle('mobile-show');return;}settings.rightCollapsed=!settings.rightCollapsed;applyPanels();requestSave()};
 $('#toggle-tools').onclick=toggleLeft;$('#collapse-tools').onclick=toggleLeft;
 $('#collapse-inspector').onclick=toggleRight;$('#restore-inspector').onclick=toggleRight;
 for(const [id,field,sign,min,max] of [['left-resizer','leftWidth',1,170,500],['right-resizer','rightWidth',-1,240,560]]){
  const el=$('#'+id);
  el.addEventListener('pointerdown',e=>{if(e.button!==0)return;e.preventDefault();el.setPointerCapture(e.pointerId);const start=e.clientX,original=Number(settings[field]);
   const move=ev=>{settings[field]=Math.round(Math.max(min,Math.min(max,original+(ev.clientX-start)*sign)));applyPanels()};
   const up=()=>{el.removeEventListener('pointermove',move);el.removeEventListener('pointerup',up);el.removeEventListener('pointercancel',up);requestSave()};
   el.addEventListener('pointermove',move);el.addEventListener('pointerup',up);el.addEventListener('pointercancel',up);
  });
  el.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){settings[field]=Math.max(min,Math.min(max,Number(settings[field])+(e.key==='ArrowRight'?10:-10)*sign));applyPanels();requestSave();e.preventDefault()}});
 }
}
function renderPrettyConditions(){
 const list=$('#pretty-condition-list');if(!list)return;
 $('#pretty-condition-count').textContent=prettyConditions.length+' 条';
 list.innerHTML=prettyConditions.map((c,i)=>{
  const data=window.GeomPrettyConstraints.assess([c],kernel)[0];
  return `<div class="pretty-condition-entry"><span>${esc(c.text)}<small class="${data.satisfied?'':'unmet'}">${data.satisfied?'✓ 当前已满足':'待优化 · 误差 '+(data.error<1e5?data.error.toExponential(2):'对象无效')}</small></span><button type="button" data-pretty-delete="${i}" aria-label="删除条件 ${i+1}" title="删除条件">×</button></div>`;
 }).join('');
 list.querySelectorAll('[data-pretty-delete]').forEach(b=>b.onclick=()=>{prettyConditions.splice(Number(b.dataset.prettyDelete),1);invalidatePretty();clearDDAR();renderPrettyConditions();requestSave();});
}
function prettySetup(){
 const strength=$('#pretty-strength'),progress=$('#pretty-progress'),input=$('#pretty-condition-input'),feedback=$('#pretty-condition-feedback');
 const toggle=$('#pretty-syntax-toggle'),panel=$('#pretty-syntax-panel');
 toggle.onclick=()=>{panel.hidden=!panel.hidden;toggle.setAttribute('aria-expanded',String(!panel.hidden));};
 toggle.addEventListener('keydown',e=>{if(e.key==='Escape'&&!panel.hidden){panel.hidden=true;toggle.setAttribute('aria-expanded','false');toggle.focus();}});
 strength.oninput=()=>$('#pretty-strength-label').textContent=strength.value;
 const add=()=>{
  const raw=input.value.trim();if(!raw)return;
  try{
   const parsed=window.GeomPrettyConstraints.parse(raw,kernel);
   if(prettyConditions.some(c=>c.text.toLowerCase()===parsed.text.toLowerCase()))throw Error('条件已经添加');
   prettyConditions.push(parsed);input.value='';feedback.classList.remove('error');feedback.textContent='✓ 已识别并添加 '+parsed.kind+' 条件';
   invalidatePretty();clearDDAR();renderPrettyConditions();requestSave();
  }catch(e){feedback.classList.add('error');feedback.textContent='✕ '+e.message;}
 };
 $('#pretty-add-condition').onclick=add;
 input.addEventListener('keydown',e=>{if(e.key==='Enter'){e.preventDefault();add();}});
 input.addEventListener('input',()=>{feedback.textContent='';feedback.classList.remove('error');});
 const setProgress=()=>{if(!prettyPlan)return;const before=kernel.serialize();Pretty.apply(kernel,prettyPlan,Number(progress.value));$('#pretty-progress-label').textContent=progress.value+'%';prettyPending=before;renderPrettyConditions();requestSave();};
 progress.addEventListener('input',setProgress);
 progress.addEventListener('change',()=>{if(prettyPending){kernel.checkpoint(prettyPending,'pretty');prettyPending=null;render();}});
 $('#pretty-run').onclick=()=>{
  const free=Pretty.controls(kernel);if(!free.length){notice('请先放置至少一个可移动的点',true);return;}
  const button=$('#pretty-run');button.disabled=true;button.textContent='正在优化…';
  try{
   const before=kernel.serialize();prettyPlan=Pretty.optimize(kernel,{strength:Number(strength.value),iterations:prettyConditions.length?115:65,constraints:prettyConditions});
   if(!prettyPlan.steps){progress.disabled=true;const info=prettyPlan.feasibility;
    $('#pretty-status').textContent=info?`未找到更优下降；已满足 ${info.satisfied}/${info.total} 项，最大归一化误差 ${info.maxError.toExponential(2)}。可调整初始作图位置或移除冲突条件。`:'图形已接近局部最优。';renderPrettyConditions();return;}
   progress.disabled=false;progress.value=100;Pretty.apply(kernel,prettyPlan,100);kernel.checkpoint(before,'pretty');
   $('#pretty-progress-label').textContent='100%';
   const inf=prettyPlan.feasibility;
   $('#pretty-status').textContent=inf?
    `条件 ${inf.satisfied}/${inf.total} 项达到精度（目标 ${inf.tolerance.toExponential(1)}），最大归一化误差 ${inf.maxError.toExponential(3)}；${prettyPlan.steps} 个路径关键帧。${inf.satisfied<inf.total?'部分条件无法同时达到；已保留最优可达解。':'随后仅在保持条件成立时优化外观。'} 拖动滑块可回看路径（中间帧可能不满足全部条件）。`:
    `已接受 ${prettyPlan.steps} 次下降 · 能量 ${prettyPlan.energyStart.toFixed(2)} → ${prettyPlan.energyEnd.toFixed(2)}。拖动滑块可沿实际优化轨迹回看与微调。`;
   renderPrettyConditions();fit();render();
  }catch(e){notice('Pretty 计算失败：'+e.message,true);console.error(e)}finally{button.disabled=false;button.innerHTML=svg('spark')+' 计算 Pretty';}
 };
 renderPrettyConditions();
}
// v4 UI command layer: one command path controls native GUI and Kernel actions.
const ConsoleCatalog=window.GeomConsole||{words:s=>String(s).split(/\s+/),search:()=>[],sections:[],commands:[]};
ConsoleCatalog.registerMethods?.(METHODS);
const numeric=x=>window.GeomExpression.parse(String(x));
function required(parts,help){if(!parts.length||parts.some(x=>!x))throw Error('用法：'+help);}
function onoff(value){const v=String(value||'').toLowerCase();if(['on','true','1','yes','show','开','开启'].includes(v))return true;if(['off','false','0','no','hide','关','关闭'].includes(v))return false;throw Error('请使用 on 或 off');}
function objectRef(name){const aliases={origin:'p_0',zero:'p_0','0':'p_0',one:'p_1','1':'p_1',i:'p_i',realaxis:'l_Re',imagaxis:'l_Im',re:'l_Re',im:'l_Im',unit:'c_unit','单位圆':'c_unit','实轴':'l_Re','虚轴':'l_Im'};return kernel.node(name)||kernel.node(aliases[String(name).toLowerCase()]||'');}
function setSelected(name){const n=objectRef(name);if(!n)throw Error('对象不存在：'+name);selected=n.id;objTab='details';$('#acc-objects').classList.add('open');render();return `${n.id} · ${n.valid?'有效':'无效'} · ${isVisible(n)?'可见':'隐藏'}`;}
function setWorkspace(field,value){const aliases={background:'grid',label:'labels',axis:'axes',colours:'palette',color:'palette',decimal:'precision',left:'leftWidth',right:'rightWidth',leftcollapsed:'leftCollapsed',rightcollapsed:'rightCollapsed'};const key=aliases[String(field).toLowerCase()]||Object.keys(settingsDefault).find(x=>x.toLowerCase()===String(field).toLowerCase());if(!key)throw Error('未知设置：'+field+'（输入 settings 查看全部）');
 if(['grid','labels','axes','snap','wheel','complex'].includes(key)){const id={grid:'setting-grid',labels:'setting-labels',axes:'setting-axes',snap:'setting-snap',wheel:'setting-wheel',complex:'complex-toggle'}[key],el=$('#'+id);el.checked=onoff(value);el.dispatchEvent(new Event('change',{bubbles:true}));}
 else if(key==='precision'){const v=Number(value);if(!Number.isInteger(v)||v<0||v>15)throw Error('精度必须在 0 到 15 位');$('#setting-precision').value=String(v);$('#setting-precision').dispatchEvent(new Event('change',{bubbles:true}));}
 else if(key==='palette'){if(!['blue','graphite','emerald'].includes(value))throw Error('颜色可选 blue / graphite / emerald');$('#setting-palette').value=value;$('#setting-palette').dispatchEvent(new Event('change',{bubbles:true}));}
 else if(key==='theme'){if(!['dark','light','toggle'].includes(value))throw Error('主题可选 dark / light / toggle');if(value==='toggle'||value!==settings.theme)$('#theme-toggle').click();}
 else if(['leftCollapsed','rightCollapsed'].includes(key)){settings[key]=onoff(value);applyPanels();requestSave();}
 else if(['leftWidth','rightWidth'].includes(key)){const n=Number(value);if(!Number.isFinite(n))throw Error('侧栏宽度无效');settings[key]=Math.round(Math.max(key==='leftWidth'?170:240,Math.min(key==='leftWidth'?500:560,n)));applyPanels();requestSave();}
 else throw Error('不可修改设置：'+key);
 return `${key} = ${settings[key]}`;
}
function setPane(name){const p=String(name||'').toLowerCase();const byName={objects:'acc-objects',details:'acc-objects',pretty:'acc-objects',relations:'acc-relations',radar:'acc-relations',complex:'acc-complex',proof:'acc-proof',library:'acc-library'};
 if(['left','right'].includes(p)){settings[p==='left'?'leftCollapsed':'rightCollapsed']=false;applyPanels();if(p==='left'&&window.innerWidth<=620)$('#tools-panel').classList.add('mobile-tools-show');if(p==='right'&&window.innerWidth<=900)$('#inspector').classList.add('mobile-show');requestSave();return '已展开'+p+'侧栏';}
 if(p==='settings'){$('#settings-rail').click();return '已打开设置';}
 byName.ddar='acc-proof';byName.aux='acc-proof';const id=byName[p];if(!id)throw Error('面板：objects/details/pretty/relations/complex/proof/library/left/right/settings');$('#'+id).classList.add('open');if(['objects','details','pretty'].includes(p)){objTab=p;renderObjects();}return '已打开 '+p;}
function importJSONDocument(text){if(text.length>2e6)throw Error('JSON 大小不能超过 2MB');const data=JSON.parse(text);setLibraryVisuals(null);kernel.load(data.schema==='geomtool.document/v1'?data.graph:data);ensureReferences();prettyConditions=(data.prettyConditions||[]).flatMap(raw=>{try{return [window.GeomPrettyConstraints.parse(raw,kernel)]}catch(e){return []}});renderPrettyConditions();if(data.name)$('#doc-name').value=String(data.name).slice(0,36);if(data.settings){settings.axes=!!data.settings.axes;settings.complex=!!data.settings.complex;syncPalette();}selected=null;radarResult=null;radarEntry=null;fit();render();return '已导入 '+kernel.list().length+' 个对象';}
function consoleHelp(query=''){const modal=$('#console-help-modal'),field=$('#console-help-search');modal.hidden=false;field.value=query;renderConsoleHelp(query);field.focus();}
function renderConsoleHelp(query=''){
 const root=$('#console-help-groups'),matches=ConsoleCatalog.search(query),byCategory=new Map();for(const c of matches){if(!byCategory.has(c.category))byCategory.set(c.category,[]);byCategory.get(c.category).push(c);}
 $('#console-help-count').textContent=`${matches.length} / ${ConsoleCatalog.commands.length} 条`;
 root.innerHTML=[...byCategory].map(([group,items])=>`<details class="help-category" ${query||group==='常用命令与对象'?'open':''}><summary>${esc(group)}<span>${items.length} 条</span></summary>${items.map(c=>`<div class="help-entry"><button type="button" data-help-use="${esc(c.example)}" title="填入控制台">填入</button><code>${esc(c.syntax)}</code><div>${esc(c.description)}</div><div class="help-example">示例 › ${esc(c.example)}</div></div>`).join('')}</details>`).join('')||'<div class="help-empty">未找到匹配命令。可使用功能名、中文、英文或部分关键词继续搜索。</div>';
 $$('[data-help-use]').forEach(btn=>btn.onclick=()=>{$('#console-input').value=btn.dataset.helpUse;modalCloseHelp();$('#console').classList.remove('closed');$('#console-input').focus()});
}
function modalCloseHelp(){$('#console-help-modal').hidden=true;}
function commandSuggest(){const el=$('#console-input'),root=$('#console-suggestions'),q=el.value.trim();if(!q||q.length<2||q==='help'){root.hidden=true;return;}const hits=ConsoleCatalog.search(q).slice(0,5);root.hidden=!hits.length;root.innerHTML=hits.map(c=>`<button type="button" class="console-suggestion" data-suggest="${esc(c.example)}"><code>${esc(c.syntax)}</code>${esc(c.category)}</button>`).join('');$$('[data-suggest]').forEach(b=>b.onclick=()=>{el.value=b.dataset.suggest;root.hidden=true;el.focus()});}
function showConsoleResult(result){if(result===null||result===undefined)return;if(typeof result==='string')return result;if(typeof result==='boolean')return result?'成功':'没有可撤销或可重做的操作';if(result.id)return `${result.id} · ${METHODS[result.op]?.label||result.op}`;return JSON.stringify(result);}
function runConsole(line){const words=ConsoleCatalog.words(line),[verb,...p]=words,a=(verb||'').toLowerCase(),b=(p[0]||'').toLowerCase();if(!a||a.startsWith('#'))return null;
 if(a==='help'||a==='?'||a==='帮助'){consoleHelp(p.join(' '));return '已打开可搜索、可折叠的完整命令手册';}
 if(objectRef(line))return setSelected(line);
 if(a==='select'||a==='pick'||a==='选中'){required(p,'select 对象名');return setSelected(p.join(' '));}
 if(a==='list'||a==='ls'){const mode=b||'all';if(!['all','hidden','visible','selected'].includes(mode))throw Error('list all|hidden|visible|selected');return kernel.list().filter(n=>mode==='all'||mode==='hidden'&&!isVisible(n)||mode==='visible'&&isVisible(n)||mode==='selected'&&n.id===selected).map(n=>`${n.id} ${METHODS[n.op].label} ${n.valid?'':'失效 '}${isVisible(n)?'可见':'隐藏'}`).join('\n')||'没有符合条件的对象';}
 if(['hide','show','toggle'].includes(a)){required(p,a+' 对象名');const n=objectRef(p.join(' '));if(!n)throw Error('对象不存在');const visible=a==='toggle'?!n.style.visible:a==='show';kernel.execute({op:'style',id:n.id,style:{visible}});return `${n.label}: ${visible?'显示':'隐藏'}${n.style.system&&!referenceVisible(n)?'（仍受坐标轴/复平面总开关控制）':''}`;}
 if(a==='style'||a==='color'){const id=p[0],field=a==='color'?'color':p[1],v=a==='color'?p[1]:p[2],n=objectRef(id);if(!n||!field||v===undefined)throw Error('style 对象 dashed|visible|color 值');if(!['dashed','visible','color'].includes(field))throw Error('仅可修改 dashed / visible / color');const value=field==='color'?v:onoff(v);if(field==='color'&&!/^#[0-9a-f]{3}([0-9a-f]{3})?$/i.test(value))throw Error('颜色需为 #RGB 或 #RRGGBB');kernel.execute({op:'style',id:n.id,style:{[field]:value}});return `${n.label}.${field} = ${value}`;}
 if(a==='rename'){required(p,'rename 对象名 新名');const n=objectRef(p[0]);if(!n)throw Error('找不到对象');const v=kernel.execute({op:'rename',id:n.id,label:p.slice(1).join(' ')});if(selected===p[0])selected=v.id;return v;}
 if(['move','drag'].includes(a)){if(p.length!==3)throw Error('move A x y');const n=objectRef(p[0]);if(!n)throw Error('找不到对象');return kernel.execute({op:'move',id:n.id,x:numeric(p[1]),y:numeric(p[2])});}
 if(['undo','redo','clear','delete','del'].includes(a)){if(a==='delete'||a==='del'){const n=objectRef(p[0]);if(!n)throw Error('找不到对象');kernel.execute({op:'delete',id:n.id});if(selected===n.id)selected=null;return `已删除 ${n.label} 及依赖对象`;}if(a==='clear'){selected=null;steps=[];}return kernel.execute({op:a});}
 if(a==='demo'){$('#load-demo').click();return '已打开示例图确认';}
 if(a==='settings'||a==='setting'||a==='workspace'){if(!p.length||['show','open'].includes(b)){$('#settings-rail').click();return '已打开工作区设置';}if(b==='close'){$('#settings-close').click();return '已关闭工作区设置';}if(b==='list')return Object.keys(settingsDefault).map(k=>`${k}: ${settings[k]}`).join('\n');required(p.slice(1),'settings 属性 值');return setWorkspace(p[0],p.slice(1).join(' '));}
 if(a==='theme')return setWorkspace('theme',b||'toggle');
 if(a==='zoom'){if(!b)throw Error('zoom in|out|125%|1.5');if(b==='in')zoomAt({x:W/2,y:H/2},1.23);else if(b==='out')zoomAt({x:W/2,y:H/2},1/1.23);else{const factor=b.endsWith('%')?numeric(b.slice(0,-1))/100:numeric(b);if(factor<=0)throw Error('缩放比例必须为正数');zoomAt({x:W/2,y:H/2},factor*88/view.scale);}return `缩放 ${(view.scale/88*100).toFixed(1)}%`;}
 if(a==='fit'){fit();return '已适配视图';}
 if(a==='pan'){if(p.length!==2)throw Error('pan 水平像素 垂直像素');view.ox+=numeric(p[0]);view.oy+=numeric(p[1]);draw();return '画布已平移';}
 if(a==='center'){if(p.length!==2)throw Error('center 世界x 世界y');view.ox=W/2-numeric(p[0])*view.scale;view.oy=H/2+numeric(p[1])*view.scale;draw();return '已将世界坐标点居中';}
 if(a==='resetview'||a==='home'){view.scale=88;view.ox=W/2;view.oy=H/2;$('#zoom-label').textContent='100%';draw();return '视图已复位';}
 if(a==='panel'||a==='tab'){required(p,'panel 面板名');return setPane(p[0]);}
 if(a==='group'){const g=Object.keys(groups).find(key=>key.toLowerCase()===b||groups[key].title===p[0]);if(!g)throw Error('未知工具组：'+b);groupUI(g);return groups[g].title;}
 if(a==='tool'){required(p,'tool rotate');const tool=Object.keys(METHODS).find(x=>x.toLowerCase()===b)||Object.values(groups).flatMap(g=>g.tools).find(x=>x[1]===p[0])?.[0]||['select','smartPoint'].find(x=>x.toLowerCase()===b);if(!tool)throw Error('未知工具：'+p[0]);activate(tool);return '已选择工具 '+toolName(tool);}
 if(a==='methods')return Object.entries(METHODS).filter(([op])=>!['intersection','secondIntersection'].includes(op)).map(([op,spec])=>`${op} — ${spec.syntax}`).join('\n');
 if(a==='guide'){required(p,'guide 工具名');const id=Object.keys(METHODS).find(x=>x.toLowerCase()===b)||b;const g=ToolGuides.guide(id);return `${toolName(id)}：${g.steps.join('；')}。${g.effect}`;}
 if(a==='param'){required(p,'param 公式');if(!$('#param-value'))throw Error('当前工具没有参数框');const v=p.join(' ');numeric(v);$('#param-value').value=v;$('#param-value').dispatchEvent(new Event('input',{bubbles:true}));return '参数 = '+v;}
 if(a==='unit'){if(activeTool!=='rotate')throw Error('请先 tool rotate');const v=b==='degree'?'deg':b==='radian'?'rad':b;if(!['deg','rad'].includes(v))throw Error('unit deg|rad');$('#param-angle-unit').value=v;$('#param-angle-unit').dispatchEvent(new Event('change',{bubbles:true}));return `旋转单位：${v}`;}
 if(a==='step'){required(p,'step 对象名');const n=objectRef(p[0]);if(!n)throw Error('找不到对象');const possible=expected(activeTool,steps.length);if(possible.length&&!possible.includes(n.geom?.kind)&&!['polygon'].includes(activeTool))throw Error('当前工具需要 '+possible.join('/'));steps.push(n.id);const spec=METHODS[activeTool];if(spec&&!spec.variadic&&steps.length>=spec.inputs.length)finalizeTool(mouse);else draw();return '当前工具输入：'+steps.join(' → ');}
 if(a==='finish'){finalizeTool(mouse);return '已结束当前构造';}
 if(a==='cancel'){steps=[];activate('select');return '已取消当前构造';}
 if(a==='click'||a==='smart'){if(p.length!==2)throw Error(a+' 世界x 世界y');const pos=toScreen({x:numeric(p[0]),y:numeric(p[1])});if(a==='smart'){const n=createSmart(pos);if(n){selected=n.id;render();return n;}throw Error('无法放点');}startToolClick(pos);return '已在世界坐标执行单击';}
 if(a==='beautify'||a==='pretty'){if(a==='beautify'||!b||b==='run'){if(!Pretty.controls(kernel).length)throw Error('请先创建至少一个可移动点');const strength=a==='beautify'?p[0]:p[1];if(strength!==undefined){const n=Number(strength);if(!Number.isFinite(n)||n<5||n>100)throw Error('Pretty 力度范围 5–100');$('#pretty-strength').value=n;$('#pretty-strength').dispatchEvent(new Event('input'));}$('#pretty-run').click();return $('#pretty-status').textContent;}if(b==='strength'){const n=Number(p[1]);if(!Number.isFinite(n)||n<5||n>100)throw Error('力度 5–100');$('#pretty-strength').value=n;$('#pretty-strength').dispatchEvent(new Event('input'));return 'Pretty 力度 '+n;}if(b==='progress'){const n=Number(p[1]);if(!Number.isFinite(n)||n<0||n>100)throw Error('进度 0–100');const el=$('#pretty-progress');if(el.disabled)throw Error('请先运行 beautify');el.value=n;el.dispatchEvent(new Event('input'));el.dispatchEvent(new Event('change'));return 'Pretty 进度 '+n+'%';}if(b==='status')return $('#pretty-status').textContent;throw Error('pretty run|strength|progress|status');}
 if(a==='relations'||a==='relation'||a==='radar'){
  if(!p.length||b==='list')return `候选 ${radarFiltered.length} 条\n`+radarFiltered.slice(0,40).map((x,i)=>`${i+1}. ${x.title}`).join('\n');
  if(b==='scan'){if(!$$('#radar-options [data-kind]:checked').length)throw Error('请先选中至少一种关系类别');$('#radar-run').click();return '已开始关系扫描（完成后在关系面板查看）';}
  if(b==='quadratic'){
   const which=p[1]?.toLowerCase(),field=which==='product'?'quadraticProductLimit':which==='expression'?'quadraticExpressionLimit':null;
   if(!field)return `二次量默认关闭；手动勾选 quadraticRatio 后扫描。乘积候选线段上限 ${radarQuadraticOptions.quadraticProductLimit}，平方和差候选量上限 ${radarQuadraticOptions.quadraticExpressionLimit}。设置用 relations quadratic product|expression 数字`;
   if(p[2]===undefined)return `${which}: ${radarQuadraticOptions[field]}`;
   const n=Number(p[2]);if(!Number.isInteger(n)||n<0||n>200)throw Error('二次量枚举上限必须是 0—200 的整数');
   radarQuadraticOptions[field]=n;return `二次量 ${which} 上限 = ${n}；下次扫描生效`;
  }
  if(b==='clear'){radarResult=null;radarEntry=null;renderRadar();draw();return '已清除关系结果';}
  if(b==='kinds'){if(!p[1]||p[1]==='list')return RelationEngine.registry.map(r=>`${r.id} · ${r.label} ${$('#radar-options [data-kind="'+r.id+'"]').checked?'on':'off'}`).join('\n');const action=p[1].toLowerCase();if(action==='all'||action==='none'){$('#radar-'+(action==='all'?'check-all':'check-none')).click();return '关系勾选：'+action;}const kind=RelationEngine.registry.find(x=>x.id.toLowerCase()===action||x.label===p[1]);if(!kind)throw Error('未知关系类别：'+p[1]);const el=$('#radar-options [data-kind="'+kind.id+'"]');el.checked=onoff(p[2]);el.dispatchEvent(new Event('change'));return `${kind.label}: ${el.checked?'on':'off'}`;}
  if(['hidden','perturb'].includes(b)){const el=$('#radar-'+b);if(p[1]===undefined)return b+': '+el.checked;el.checked=onoff(p[1]);return b+': '+el.checked;}
  if(b==='search'){const q=p.slice(1).join(' ');$('#radar-query').value=q;$('#radar-query').dispatchEvent(new Event('input'));return '关系搜索：'+q;}
  if(b==='sort'){if(!['score','kind'].includes(p[1]))throw Error('relations sort score|kind');$('#radar-sort').value=p[1];$('#radar-sort').dispatchEvent(new Event('change'));return '关系排序：'+p[1];}
  if(b==='select'){const i=Number(p[1])-1;if(!Number.isInteger(i)||i<0||i>=radarFiltered.length)throw Error('关系序号不存在');radarEntry=radarFiltered[i];proofOverlay=null;radarPage=Math.max(radarPage,Math.ceil((i+1)/80));renderRadar();draw();return radarEntry.title;}
  if(b==='more'){$('#radar-more').click();return `显示 ${Math.min(radarPage*80,radarFiltered.length)} 条关系`;}
  throw Error('relations scan|kinds|hidden|perturb|search|sort|select|more|clear');
 }
 if(a==='ddar'){if(b==='goals'){discoverDDARGoals();return 'DDAR 待证命题发掘已开始';}if(b==='prove'){const i=Number(p[1]);if(Number.isInteger(i)&&i>=1&&i<=goalCatalog?.goals.length){goalSelection=goalCatalog.goals[i-1];goalResult=null;runDDAR(false,goalSelection);return '正在搜索 '+goalSelection.text;}if(goalSelection){runDDAR(false,goalSelection);return '正在搜索 '+goalSelection.text;}throw Error('请先发掘并选择目标，或使用 ddar prove 1');}if(!p.length||b==='run'){runDDAR(false);return 'DDAR 推理已发起';}if(b==='aux'){runDDAR(true);return 'DDAR 辅助构造分析已发起';}if(b==='stats')return ddarState?JSON.stringify({facts:ddarState.facts.length,counts:ddarState.counts,matched:ddarState.matches.filter(x=>x.status==='proved').length,elapsedMs:ddarState.elapsedMs}):'尚未运行 DDAR';if(b==='export')return JSON.stringify(proofPayload(),null,2);throw Error('ddar run|goals|prove [序号]|aux|stats|export');}
 if(a==='proof'){if(!p.length||b==='list')return proofFacts.map((x,i)=>`${i+1}. ${x.conclusion}`).join('\n')||'没有可检查的定义结论';if(b==='select'){const n=Number(p[1]),f=Number.isInteger(n)?proofFacts[n-1]:proofFacts.find(x=>x.id===p[1]);if(!f)throw Error('证明目标不存在');$('#proof-goal').value=f.id;return f.conclusion;}if(b==='check'){proofCheck();return $('#proof-output').textContent;}if(b==='export'){$('#proof-export').click();return '已请求复制 DD+AR 对接数据';}throw Error('proof list|select|check|export');}
 if(a==='library'){if(!p.length||b==='list'){const result=window.GeomLibrary.search(window.GeomLibraryData,$('#library-search').value,$('#library-filter').value,0,Math.min(50,libraryShown));return `${result.total} 条匹配\n`+result.entries.map((x,i)=>`${i+1}. ${x.title} [${x.id}]`).join('\n');}if(b==='search'){$('#library-search').value=p.slice(1).join(' ');$('#library-search').dispatchEvent(new Event('input'));return `题库匹配 ${$('#library-result-count').textContent}`;}if(b==='filter'){$('#library-filter').value=p.slice(1).join(' ');if($('#library-filter').value!==p.slice(1).join(' '))throw Error('未知题库分类，可在下拉框查看');$('#library-filter').dispatchEvent(new Event('change'));return '分类 = '+p[1];}if(b==='more'){const btn=$('#library-more');if(btn)btn.click();return '已展开更多题库记录';}if(b==='select'||b==='load'){let id=p[1],entry;if(id==='selected')id=libraryCurrent;if(/^\d+$/.test(id||'')){const matches=window.GeomLibrary.search(window.GeomLibraryData,$('#library-search').value,$('#library-filter').value,0,Math.max(50,libraryShown));id=matches.entries[Number(id)-1]?.id;}entry=window.GeomLibraryData.entries.find(x=>x.id===id);if(!entry)throw Error('找不到题库记录：'+(id||''));libraryCurrent=entry.id;renderLibrary();renderLibraryDetail();if(b==='load')loadLibrary(entry);else $('#acc-library').classList.add('open');return `${b==='load'?'载入':'选择'}：${entry.title}`;}throw Error('library search|list|select|load|filter|more');}
 if(a==='import'){if(!p.length||b==='file'){$('#import-btn').click();return '已打开文件选择器';}if(b==='json')return importJSONDocument(line.slice(/^import\s+json\s+/i.exec(line)?.[0].length||0));throw Error('import file 或 import json {JSON}');}
 if(a==='export'){if(!p.length||b==='file'){$('#export-btn').click();return '已请求导出 .geom.json 文件';}if(b==='json')return JSON.stringify({schema:'geomtool.document/v1',name:$('#doc-name').value,graph:kernel.serialize(),prettyConditions:prettyConditions.map(c=>c.text),settings:{complex:settings.complex,axes:settings.axes}},null,2);if(b==='proof')return JSON.stringify(proofPayload(),null,2);throw Error('export file|json|proof');}
 if(a==='copy'){if(b==='graph')copyText(JSON.stringify(kernel.serialize(),null,2));else if(b==='proof')copyText(JSON.stringify(proofPayload(),null,2));else throw Error('copy graph|proof');return '已请求复制至剪贴板';}
 if(a==='doc'||a==='document'){if(!p.length)return $('#doc-name').value;$('#doc-name').value=p.join(' ').slice(0,36);requestSave();return '文档名：'+$('#doc-name').value;}
 if(a==='shortcuts')return 'Ctrl+Z：undo，Ctrl+Shift+Z/Ctrl+Y：redo，Delete：delete 已选对象，Esc：cancel，空格拖动：pan，Enter：finish 多边形，滚轮：zoom。命令行另支持 ↑ / ↓ 浏览历史。';
 if(a==='console'){if(b==='clear'){$('#console-output').innerHTML='';return null;}if(b==='history')return consoleHistory.map((s,i)=>`${i+1}  ${s}`).join('\n');throw Error('console clear|history');}
 if(a==='ui'){const id=p[1]?.replace(/^#/,'');const el=id&&document.getElementById(id);if(!el)throw Error('找不到界面元素：'+(id||''));if(b==='click'){el.click();return '已触发 '+id;}if(b==='set'){const v=p.slice(2).join(' ');if(el.type==='checkbox')el.checked=onoff(v);else el.value=v;el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));return `${id} = ${v}`;}throw Error('ui click|set 元素ID [值]');}
 // Geometry operations, all METHOD aliases and construction parameters go through Kernel.
 return runText(kernel,line);
}
function setupConsole(){
 const input=$('#console-input');$('#console-help-btn').onclick=()=>consoleHelp();$('#console-help-close').onclick=modalCloseHelp;$('#console-help-done').onclick=modalCloseHelp;$('#console-help-search').oninput=e=>renderConsoleHelp(e.target.value);$('#console-help-modal').addEventListener('mousedown',e=>{if(e.target===$('#console-help-modal'))modalCloseHelp();});
 input.addEventListener('input',commandSuggest);
}
function log(line,type=''){const target=$('#console-output'),el=document.createElement('div');el.className=type;el.textContent=line;target.appendChild(el);target.scrollTop=target.scrollHeight;while(target.childNodes.length>100)target.firstChild.remove()}
bind();restore();ensureReferences();applyPanels();groupUI('select');activate('select');resize();render();
// Intentionally limited public UI bridge: use commands, not ad-hoc DOM mutations.
window.GeomTool={kernel,runConsole,searchCommands:query=>ConsoleCatalog.search(query),get settings(){return {...settings}},get prettyConditions(){return prettyConditions.map(c=>c.text)},addPrettyCondition:s=>{const c=window.GeomPrettyConstraints.parse(s,kernel);prettyConditions.push(c);invalidatePretty();renderPrettyConditions();requestSave();return c},removePrettyCondition:i=>{prettyConditions.splice(i,1);invalidatePretty();renderPrettyConditions();requestSave();},command:cmd=>kernel.execute(cmd),serialize:()=>kernel.serialize(),runText:line=>runText(kernel,line),getProofPayload:proofPayload,getDDAR:()=>ddarState,getGoals:()=>goalCatalog,getGoalProof:()=>goalResult,discoverDDARGoals,proveGoal:g=>runDDAR(false,g),getAuxiliary:()=>auxState,runDDAR,applyAuxiliary,probeAuxiliary,fit,scanRelations:scanRadar,pretty:()=>$('#pretty-run').click(),getGhost:()=>ghost,getPreview:()=>toolPreview,searchLibrary:(q,cat)=>window.GeomLibrary.search(window.GeomLibraryData,q,cat),loadLibrary:id=>{const entry=window.GeomLibraryData.entries.find(x=>x.id===id);if(!entry)throw Error('未知题库记录');loadLibrary(entry);},registerTool:(group,{id,name,description=''})=>{if(['intersection','secondIntersection'].includes(id))throw Error('已停用的工具：'+id);if(!groups[group]||!METHODS[id])throw Error('未知分组或构造');groups[group].tools.push([id,name||METHODS[id].label,description]);if(group===activeGroup)groupUI(group)}};
})();
