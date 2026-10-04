(() => {
  'use strict';

  const GRID_N = 11;
  const PAGE_SIZE = 40;
  const EXPECTED = {1:1,2:1,3:2,4:5,5:12,6:35,7:107,8:363,9:1248,10:4460,11:16094,12:58937};
  const I18N = {
    en: {
      ready:'ready', group:'group', target:'target', cellMode:'periodic spectrum', companionMode:'companion', generate:'generate', continue:'continue', stop:'stop', advanced:'search character', cellPieces:'period limit', aggression:'aggression', complexity:'complexity', regularity:'regularity', continueHint:'▶ again continues without discarding earlier results', palette:'art direction', voidColor:'void', boundary:'edge', edgeWeight:'edge weight', cellEdge:'cell joints', edgeStyle:'edge style', edgeColor:'edge color', globalSaturation:'global saturation', globalBrightness:'global lightness', extraColors:'extra colors', dispersion:'color spacing', neighborBias:'neighbor relation', orbitSpan:'orbit span', luma:'luma contrast', materialStrength:'material intensity', colorHint:'Palette style defines a color orbit; color controls tune its spacing, while material and edge controls change the decorative finish without changing the palette family.',
      navHint:'drag / wheel', emptyHint:'choose a shape → ▶', connected:'connected · holeless', disconnected:'disconnected', holed:'connected · holed', empty:'empty',
      loading:'loading', searching:'searching', stopped:'stopped', noResult:'no result in budget', result:'result', results:'results', catalog:'catalog',
      phasePrepare:'prepare', phaseCache:'cache', phaseExact:'exact cover', phaseMutate:'local moves', phasePack:'packing', phaseSeed:'exact seed', phaseKnowledge:'database',
      gC1:'translations only', gK4:'translations + 180° + axis mirrors', gC4:'translations + quarter turns', gD4:'translations + rotations + mirrors',
      aAuto:'adaptive portfolio', aDense:'density optimizer', aConstructive:'constructive heuristic', aHuman:'human-like constructive search',
      infoGroups:'<b>C₁</b> translation; <b>K₄</b> I,R₂,Mₓ,Mᵧ; <b>C₄</b> quarter turns; <b>D₄</b> rotations + mirrors.',
      infoMetrics:'<b>μ</b> is the reduced fundamental-domain area detected from translation symmetries. <b>Hₐ</b> and <b>Hₒ</b> are adjacency and orientation entropies.',
      infoFinite:'Periodic Spectrum searches the best primary-tile coverage separately for primitive periods 1…m, with extra exact-search effort at 1/2/3/4/6/8 and substructure mining from larger exact cells. Companion analyzes the residual region itself: one connected residual component, or several congruent residual components, becomes the companion shape directly. Optimize combined coverage first, then primary coverage, then companion area, while still exploring non-trivial near-optimal variants.',
      infoDb:'𝒟 is offline-first: 81,265 free connected holeless polyominoes (n=1…12), 1,611 Heesch non-tilers (n=7…10), plus 5,109 exact periodic seeds and a 600-entry complex-periodic leaderboard. IDs Pn-#### are local canonical IDs.',
      infoEngine:'<b>A</b> adaptive portfolio; <b>D</b> density-first; <b>H</b> human-like constructive search. Exact cover, MRV, repair and LNS are internal tools.',
      knownNonTiler:'known non-tiler · Heesch DB', labelKnownDensity:'known non-tiler · density',
      infoSources:'Sources and generated-data notes are in data/known/SOURCES.md and RESEARCH.md.',
      labelPeriodic:'periodic', labelComplex:'complex periodic', labelNear:'near packing', labelDrift:'irregular field', labelCached:'cached exact', labelLifted:'lifted periodic', labelFallback:'near fallback', labelKnownNear:'known non-tiler · near', labelDensity:'density field', labelDensityDrift:'density drift', labelLattice:'lattice density', labelKnownDensity:'known non-tiler · density',
      tipCenter:'center', tipClear:'clear', tipRandom:'random catalog shape', tipDownload:'download artwork', tipFullscreen:'full-screen tiling', exportFormat:'format', exportRatio:'page ratio', exportRegions:'basic regions', exportLongSide:'long side px', exportGo:'export current tiling', patternTransform:'orientation', patternScale:'motif size', patternWidth:'line width', patternOpacity:'opacity', patternContrast:'contrast', tipRotateCCW:'rotate 90° counterclockwise', tipRotateCW:'rotate 90° clockwise', tipFlipV:'mirror top / bottom', tipFlipH:'mirror left / right', tipZoomOut:'zoom out', tipZoomIn:'zoom in', tipReset:'reset view', tipInfo:'info', tipPrevious:'previous', tipNext:'next'
    },
    zh: {
      ready:'就绪', group:'变换群', target:'目标', cellMode:'周期谱', companionMode:'伴生块', generate:'生成', continue:'继续', stop:'停止', advanced:'搜索性格', cellPieces:'周期上限', aggression:'激进度', complexity:'复杂度', regularity:'规律性', continueHint:'再次 ▶ 将继续搜索，不会丢弃之前结果', palette:'艺术配色', voidColor:'空缺', boundary:'边界', edgeWeight:'边界粗细', cellEdge:'格内接缝', edgeStyle:'线型', edgeColor:'边界颜色', globalSaturation:'整体饱和度', globalBrightness:'整体明度', extraColors:'额外颜色', dispersion:'颜色间距', neighborBias:'邻色关系', orbitSpan:'色域范围', luma:'明度差', materialStrength:'材质强度', colorHint:'每种风格保留自己的连续颜色轨道；颜色控制决定配色关系，材质与边界控制只改变装饰表现，不改变配色家族。',
      navHint:'拖动 / 滚轮', emptyHint:'选择形状 → ▶', connected:'连通 · 无洞', disconnected:'非连通', holed:'连通 · 有洞', empty:'空',
      loading:'载入', searching:'搜索', stopped:'已停止', noResult:'预算内未找到', result:'个结果', results:'个结果', catalog:'图鉴',
      phasePrepare:'准备', phaseCache:'缓存', phaseExact:'精确覆盖', phaseMutate:'局部变换', phasePack:'近密铺', phaseSeed:'精确种子', phaseKnowledge:'数据库',
      gC1:'仅平移', gK4:'平移 + 180° + 横纵镜像', gC4:'平移 + 90°旋转', gD4:'平移 + 旋转 + 镜像',
      aAuto:'自适应组合', aDense:'高密度优化', aConstructive:'构造式启发', aHuman:'类人构造搜索',
      infoGroups:'<b>C₁</b> = 平移；<b>K₄</b> = I,R₂,Mₓ,Mᵧ；<b>C₄</b> = 四分之一转动；<b>D₄</b> = 旋转 + 镜像。',
      infoMetrics:'<b>μ</b>：由当前密铺的平移对称约化得到的基本域面积。<b>Hₐ</b> 与 <b>Hₒ</b> 分别是邻接与方向熵。',
      infoFinite:'周期谱分别搜索最小周期块数 1…m 的主 tile 最大覆盖率，并对 1/2/3/4/6/8 单体加强精确搜索，还会从较大的满铺周期中反向提取可独立延拓的更小子结构。伴生块直接分析剩余区域：剩余恰为一个连通分支，或多个彼此全等的连通分支时，以整个分支作为伴生块，不再把剩余区域强行切成小碎块。目标依次为：联合覆盖率最大、主 tile 覆盖率最大、伴生块面积更小，同时继续保留非平凡的相近最优构型。',
      infoDb:'𝒟 为离线优先数据库：n=1…12 共 81,265 个 free 连通无洞 polyomino；n=7…10 共 1,611 个 Heesch non-tiler，并附 5,109 条周期精确种子与 600 条复杂周期榜单。Pn-#### 为本站 canonical ID。',
      infoEngine:'<b>A</b> 自适应组合；<b>D</b> 密度优先；<b>H</b> 类人构造搜索。Exact cover、MRV、局部修补与 LNS 都作为内部工具自动调用。',
      knownNonTiler:'已知不可密铺 · Heesch 数据库', labelKnownDensity:'已知不可密铺 · 高密度场',
      infoSources:'数据来源与预计算说明见 data/known/SOURCES.md 与 RESEARCH.md。',
      labelPeriodic:'周期密铺', labelComplex:'复杂周期', labelNear:'近密铺', labelDrift:'乱序场', labelCached:'缓存精确解', labelLifted:'提升周期解', labelFallback:'近似回退', labelKnownNear:'已知不可密铺 · 近似', labelDensity:'高密度场', labelDensityDrift:'高密度乱序场', labelLattice:'格点高密度场', labelKnownDensity:'已知不可密铺 · 高密度场',
      tipCenter:'居中', tipClear:'清空', tipRandom:'随机图鉴形状', tipDownload:'下载作品', tipFullscreen:'全屏密铺', exportFormat:'格式', exportRatio:'页面长宽比', exportRegions:'基本区域数', exportLongSide:'长边像素', exportGo:'导出当前密铺', patternTransform:'纹样方向', patternScale:'纹样尺度', patternWidth:'线条宽度', patternOpacity:'透明度', patternContrast:'对比度', tipRotateCCW:'逆时针旋转 90°', tipRotateCW:'顺时针旋转 90°', tipFlipV:'上下对称', tipFlipH:'左右对称', tipZoomOut:'缩小', tipZoomIn:'放大', tipReset:'重置视图', tipInfo:'说明', tipPrevious:'上一页', tipNext:'下一页'
    }
  };

  const state = {
    lang: 'en',
    cells: new Set(['4,4','5,4','6,4','4,5','4,6']),
    group: 'D4', mode: 'cell', algorithm: 'auto', budget: 60000,
    searchSettings:{cellPieces:10,aggression:58,complexity:56,regularity:48},
    colorSettings:{style:'morandi',hue:0,saturation:100,brightness:100,extraColors:0,dispersion:66,neighborBias:8,orbitSpan:-8,luma:26,edge:48,cellEdge:0,materialStrength:34,material:'plain',patternTransform:'d4',patternScale:54,patternWidth:44,patternOpacity:54,patternContrast:48,edgeLineStyle:'single',edgeColorMode:'auto',voidMode:'auto'},
    searchRun:0,lastSearchSignature:'',resultShapeSignature:'',workerMode:'',inflightMode:'',inflightGroup:'',
    catalogN: 5, catalogPage: 0, catalogPages: 1, catalogCount: 0, catalogShapes: [], catalogId: null, catalogMeta: null,
    results: [], activeResult: null, previewResult: null,
    catalogWorker: null, solverWorker: null, solverReady:false, solverUrl:null, busy: false, previewFitted: false,
    view: {x:0,y:0,scale:34,m:[1,0,0,1]}, drag:null, pointers:new Map(), lastPinch:null, dpr:1,
    drawQueued:false, resizeObserver:null, colorCacheKey:'', colorMap:null, rasterCacheKey:'', rasterCache:null,
    searchDebounce:null, lastProgress:null, catalogLoadToken:0, workerBootTimer:null, pendingSolveMessage:null
  };

  const $ = id => document.getElementById(id);
  const els = {
    editorGrid:$('editorGrid'),tileArea:$('tileArea'),tileBox:$('tileBox'),shapeMessage:$('shapeMessage'),
    groupModes:$('groupModes'),searchModes:$('searchModes'),algorithmModes:$('algorithmModes'),budgetModes:$('budgetModes'),solveBtn:$('solveBtn'),stopBtn:$('stopBtn'),
    metricCoverage:$('metricCoverage'),metricFund:$('metricFund'),metricAdj:$('metricAdj'),metricOri:$('metricOri'),metricPeriod:$('metricPeriod'),metricScore:$('metricScore'),
    canvas:$('tilingCanvas'),emptyCanvas:$('emptyCanvas'),resultName:$('resultName'),resultDims:$('resultDims'),zoomLabel:$('zoomLabel'),liveHud:$('liveHud'),
    nSelector:$('nSelector'),catalogCount:$('catalogCount'),catalogGrid:$('catalogGrid'),catalogSearch:$('catalogSearch'),catalogCheck:$('catalogCheck'),dbHeesch:$('dbHeesch'),dbSeeds:$('dbSeeds'),dbComplex:$('dbComplex'),prevPage:$('prevPage'),nextPage:$('nextPage'),pageLabel:$('pageLabel'),
    resultList:$('resultList'),resultCount:$('resultCount'),statusDot:$('statusDot'),statusText:$('statusText'),
    clearBtn:$('clearBtn'),normalizeBtn:$('normalizeBtn'),sampleBtn:$('sampleBtn'),rotateCCWBtn:$('rotateCCWBtn'),rotateCWBtn:$('rotateCWBtn'),flipVBtn:$('flipVBtn'),flipHBtn:$('flipHBtn'),zoomOutBtn:$('zoomOutBtn'),zoomInBtn:$('zoomInBtn'),resetViewBtn:$('resetViewBtn'),infoBtn:$('infoBtn'),infoDialog:$('infoDialog'),langBtn:$('langBtn'),
    searchProgress:$('searchProgress'),progressPhase:$('progressPhase'),progressPct:$('progressPct'),progressBar:$('progressBar'),progressDims:$('progressDims'),progressNodes:$('progressNodes'),progressBest:$('progressBest'),progressTime:$('progressTime'),
    solveLabel:$('solveLabel'),advancedSearch:$('advancedSearch'),cellPieces:$('cellPieces'),cellPiecesDec:$('cellPiecesDec'),cellPiecesInc:$('cellPiecesInc'),aggression:$('aggression'),complexity:$('complexity'),regularity:$('regularity'),pieceLimitRow:$('pieceLimitRow'),resetSearchSettings:$('resetSearchSettings'),
    advancedColor:$('advancedColor'),hueWheel:$('hueWheel'),colorPreview:$('colorPreview'),paletteStyles:$('paletteStyles'),paletteStyleName:$('paletteStyleName'),paletteStyleKind:$('paletteStyleKind'),saturation:$('saturation'),brightness:$('brightness'),extraColorsDec:$('extraColorsDec'),extraColorsInc:$('extraColorsInc'),extraColorsOut:$('extraColorsOut'),dispersion:$('dispersion'),neighborBias:$('neighborBias'),orbitSpan:$('orbitSpan'),luma:$('luma'),edge:$('edge'),cellEdge:$('cellEdge'),materialStrength:$('materialStrength'),tileMaterials:$('tileMaterials'),materialName:$('materialName'),edgeLineStyle:$('edgeLineStyle'),edgeColorMode:$('edgeColorMode'),voidMode:$('voidMode'),resetColorSettings:$('resetColorSettings'),patternTransform:$('patternTransform'),patternScale:$('patternScale'),patternWidth:$('patternWidth'),patternOpacity:$('patternOpacity'),patternContrast:$('patternContrast'),downloadBtn:$('downloadBtn'),downloadMenu:$('downloadMenu'),exportFormat:$('exportFormat'),exportRatio:$('exportRatio'),exportRegions:$('exportRegions'),exportLongSide:$('exportLongSide'),exportGo:$('exportGo'),focusBtn:$('focusBtn')
  };

  function t(k){return I18N[state.lang][k] ?? k;}

  function displayLabel(r){
    const x=r?.label||r?.kind||'';
    if(x==='complex periodic')return t('labelComplex');
    if(x==='near packing')return t('labelNear');
    if(x==='aperiodic-looking / finite')return t('labelDrift');
    if(x==='cached exact')return t('labelCached');
    if(x==='lifted periodic')return t('labelLifted');
    if(x==='near fallback'||x==='finite fallback')return t('labelFallback');
    if(x==='known non-tiler · near')return t('labelKnownNear');
    if(x==='known non-tiler · density')return t('labelKnownDensity');
    if(x==='lattice density')return t('labelLattice');
    if(x==='known non-tiler · density')return t('labelKnownDensity');
    if(x==='density field'||x==='density fallback'||r?.kind==='field'&&x!=='density drift')return t('labelDensity');
    if(x==='density drift')return t('labelDensityDrift');
    if(x==='minimal companion'||x==='companion')return t('companionMode');if(x==='periodic cell')return t('cellMode');if(x==='periodic'||r?.kind==='periodic')return t('labelPeriodic');
    return x||'—';
  }
  function displayEngine(e){return ({cache:'⊞','complex-cache':'★',exact:'⊞',DLX:'⊞','bit-X':'⊞',local:'≈',resume:'↻',greedy:'D',lift:'A·↗',frontier:'C',repair:'D·↻',hybrid:'A',LNS:'D·↻',block:'▭',Q:'A',D:'D','D·LNS':'D·↻',L:'Λ',M:'H',C:'H','C·R':'H·↻'}[e]||e||'—');}
  function phaseLabel(p){return t({prepare:'phasePrepare',cache:'phaseCache',exact:'phaseExact',mutate:'phaseMutate',pack:'phasePack','exact-seed':'phaseSeed',knowledge:'phaseKnowledge'}[p]||p);}
  function applyLanguage(){
    document.documentElement.lang = state.lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(el=>{const k=el.dataset.i18n;const v=t(k);if(v.includes('<'))el.innerHTML=v;else el.textContent=v;});
    document.querySelectorAll('[data-i18n-title]').forEach(el=>{const v=t(el.dataset.i18nTitle);el.title=v;if(el.hasAttribute('aria-label'))el.setAttribute('aria-label',v);});
    els.langBtn.textContent = state.lang === 'en' ? '中' : 'EN'; els.langBtn.title = state.lang === 'en' ? '中文' : 'English'; els.langBtn.setAttribute('aria-label',els.langBtn.title);
    renderEditor();renderResults();updateSolveLabel();if(!state.busy)setStatus(t('ready'));if(state.lastProgress)updateProgress(state.lastProgress);scheduleDraw();
  }
  function setStatus(text,kind='idle'){els.statusText.textContent=text;els.statusDot.className='status-dot'+(kind==='busy'?' busy':kind==='error'?' error':'');}
  function key(x,y){return `${x},${y}`;}
  function parseKey(k){return k.split(',').map(Number);}
  function mod(a,n){return ((a%n)+n)%n;}
  function fmt(v,d=2){return Number.isFinite(v)?Number(v).toFixed(d).replace(/\.0+$/,'').replace(/(\.\d*?)0+$/,'$1'):'—';}

  function shapeArray(){return [...state.cells].map(parseKey);}
  function shapeBounds(cells=shapeArray()){
    if(!cells.length)return{minX:0,minY:0,maxX:0,maxY:0,w:0,h:0};let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;
    for(const[x,y]of cells){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);}return{minX,minY,maxX,maxY,w:maxX-minX+1,h:maxY-minY+1};
  }
  function isConnected(cells=shapeArray()){
    if(!cells.length)return false;const s=new Set(cells.map(([x,y])=>key(x,y))),q=[cells[0]],seen=new Set([key(...cells[0])]);
    for(let i=0;i<q.length;i++){const[x,y]=q[i];for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]){const k=key(x+dx,y+dy);if(s.has(k)&&!seen.has(k)){seen.add(k);q.push([x+dx,y+dy]);}}}return seen.size===s.size;
  }
  function hasHole(cells=shapeArray()){
    if(!cells.length)return false;const b=shapeBounds(cells),occ=new Set(cells.map(([x,y])=>key(x,y))),minX=b.minX-1,minY=b.minY-1,maxX=b.maxX+1,maxY=b.maxY+1,q=[[minX,minY]],seen=new Set([key(minX,minY)]);
    for(let i=0;i<q.length;i++){const[x,y]=q[i];for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]){const nx=x+dx,ny=y+dy,k=key(nx,ny);if(nx<minX||nx>maxX||ny<minY||ny>maxY||occ.has(k)||seen.has(k))continue;seen.add(k);q.push([nx,ny]);}}
    for(let y=b.minY;y<=b.maxY;y++)for(let x=b.minX;x<=b.maxX;x++)if(!occ.has(key(x,y))&&!seen.has(key(x,y)))return true;return false;
  }
  function normalizeShape(cells=shapeArray()){
    if(!cells.length)return[];const b=shapeBounds(cells);return cells.map(([x,y])=>[x-b.minX,y-b.minY]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]);
  }
  function editorShapeCentered(cells){const n=normalizeShape(cells),b=shapeBounds(n),ox=Math.floor((GRID_N-b.w)/2),oy=Math.floor((GRID_N-b.h)/2);return n.map(([x,y])=>[x+ox,y+oy]);}
  function canonicalShapeSig(cells=normalizeShape()){return cells.map(p=>p.join(',')).join(';');}
  function clearResultsForNewShape(){state.results=[];state.activeResult=null;state.previewResult=null;state.lastSearchSignature='';state.resultShapeSignature='';renderResults();updateMetrics();scheduleDraw();}
  function setShape(cells,id=null,meta=null){const old=canonicalShapeSig();const next=canonicalShapeSig(normalizeShape(cells));state.cells=new Set(editorShapeCentered(cells).map(([x,y])=>key(x,y)));state.catalogId=id;state.catalogMeta=meta;if(old!==next)clearResultsForNewShape();renderEditor();renderCatalog();updateSolveLabel();}

  function initEditor(){
    const frag=document.createDocumentFragment();for(let y=0;y<GRID_N;y++)for(let x=0;x<GRID_N;x++){const b=document.createElement('button');b.type='button';b.className='editor-cell';b.dataset.x=x;b.dataset.y=y;b.setAttribute('role','gridcell');frag.appendChild(b);}els.editorGrid.appendChild(frag);
    let painting=null;const paint=(target,value)=>{if(!target.classList.contains('editor-cell'))return;const k=key(+target.dataset.x,+target.dataset.y);const before=canonicalShapeSig();if(value)state.cells.add(k);else state.cells.delete(k);state.catalogId=null;state.catalogMeta=null;if(before!==canonicalShapeSig())clearResultsForNewShape();renderEditor();updateSolveLabel();};
    els.editorGrid.addEventListener('pointerdown',e=>{if(!e.target.classList.contains('editor-cell'))return;e.preventDefault();const k=key(+e.target.dataset.x,+e.target.dataset.y);painting=!state.cells.has(k);paint(e.target,painting);els.editorGrid.setPointerCapture?.(e.pointerId);});
    els.editorGrid.addEventListener('pointermove',e=>{if(painting===null||!(e.buttons&1))return;const target=document.elementFromPoint(e.clientX,e.clientY);if(target&&els.editorGrid.contains(target))paint(target,painting);});
    const end=()=>painting=null;els.editorGrid.addEventListener('pointerup',end);els.editorGrid.addEventListener('pointercancel',end);renderEditor();
  }
  function renderEditor(){
    for(const node of els.editorGrid.children)node.classList.toggle('on',state.cells.has(key(+node.dataset.x,+node.dataset.y)));
    const cells=shapeArray(),b=shapeBounds(cells),connected=isConnected(cells),hole=hasHole(cells);els.tileArea.textContent=`n=${cells.length}`;els.tileBox.textContent=cells.length?`${b.w}×${b.h}`:'0×0';
    if(!cells.length){els.shapeMessage.textContent=t('empty');els.shapeMessage.className='shape-message bad';}
    else if(!connected){els.shapeMessage.textContent=t('disconnected');els.shapeMessage.className='shape-message bad';}
    else if(hole){els.shapeMessage.textContent=t('holed');els.shapeMessage.className='shape-message bad';}
    else{let extra='';if(state.catalogMeta?.heesch)extra+=` · Hc=${state.catalogMeta.heesch.Hc}/Hh=${state.catalogMeta.heesch.Hh}`;const tor=state.catalogMeta?.fact?.facts?.smallestKnownTorus;if(tor)extra+=` · T ${tor[0]}×${tor[1]}`;els.shapeMessage.textContent=t('connected')+extra;els.shapeMessage.className='shape-message';}
  }

  function initSelectors(){
    const setup=(root,attr,stateKey,onChange=null)=>root.addEventListener('click',e=>{const b=e.target.closest(`[data-${attr}]`);if(!b)return;state[stateKey]=b.dataset[attr];[...root.children].forEach(x=>{const on=x===b;x.classList.toggle('active',on);x.setAttribute('aria-checked',on);});onChange?.();});
    setup(els.groupModes,'group','group',()=>{renderCatalog();updateSolveLabel();});setup(els.searchModes,'mode','mode',()=>{updateModeUI();updateSolveLabel();scheduleDraw();});setup(els.algorithmModes,'algorithm','algorithm',updateSolveLabel);setup(els.budgetModes,'budget','budget',()=>{state.budget=Number(state.budget)||60000;});
    for(let n=1;n<=12;n++){const b=document.createElement('button');b.type='button';b.className='n-btn'+(n===state.catalogN?' active':'');b.dataset.n=n;b.textContent=n;b.title=`${n}-omino`;els.nSelector.appendChild(b);}
    els.nSelector.addEventListener('click',e=>{const b=e.target.closest('[data-n]');if(!b)return;state.catalogN=+b.dataset.n;state.catalogPage=0;[...els.nSelector.children].forEach(x=>x.classList.toggle('active',x===b));requestCatalog();});
  }

  const SEARCH_DEFAULTS={cellPieces:10,aggression:58,complexity:56,regularity:48};
  function clampPeriodLimit(v){return Math.max(8,Math.min(20,Math.round(Number(v)||10)));}
  function readSearchSettings(){
    if(els.cellPieces)els.cellPieces.value=String(clampPeriodLimit(els.cellPieces.value));
    state.searchSettings={cellPieces:clampPeriodLimit(els.cellPieces?.value),aggression:+els.aggression.value,complexity:+els.complexity.value,regularity:+els.regularity.value};updateSearchOutputs();return state.searchSettings;
  }
  function updateSearchOutputs(){const q=state.searchSettings,pairs=[['cellPieces',`1–${q.cellPieces}`],['aggression',q.aggression],['complexity',q.complexity],['regularity',q.regularity]];for(const[id,v]of pairs){const o=$(id+'Out');if(o)o.value=o.textContent=String(v);}}
  function updateModeUI(){if(els.pieceLimitRow)els.pieceLimitRow.classList.remove('soft-disabled');}
  function initAdvancedSearch(){for(const el of [els.cellPieces,els.aggression,els.complexity,els.regularity])el?.addEventListener('input',readSearchSettings);const bump=d=>{if(!els.cellPieces)return;els.cellPieces.value=String(clampPeriodLimit(+els.cellPieces.value+d));readSearchSettings();};els.cellPiecesDec?.addEventListener('click',()=>bump(-1));els.cellPiecesInc?.addEventListener('click',()=>bump(1));els.resetSearchSettings?.addEventListener('click',()=>{for(const[k,v]of Object.entries(SEARCH_DEFAULTS))$(k).value=String(v);readSearchSettings();});readSearchSettings();updateModeUI();}

  const COLOR_DEFAULTS={style:'morandi',hue:0,saturation:100,brightness:100,extraColors:0,dispersion:66,neighborBias:8,orbitSpan:-8,luma:26,edge:48,cellEdge:0,materialStrength:34,material:'plain',patternTransform:'d4',patternScale:54,patternWidth:44,patternOpacity:54,patternContrast:48,edgeLineStyle:'single',edgeColorMode:'auto',voidMode:'auto'};
  const COLOR_PROFILES={
    morandi:{name:'Morandi Studio',kind:'muted blue-green daylight',mode:'orbit',span:.72,contrast:9,neighborTarget:.76,voidS:.05,voidL:97,orbit:[[138,25,69],[154,27,62],[171,28,71],[188,30,64],[204,28,73],[219,27,66],[232,24,75]]},
    mondrian:{name:'De Stijl Daylight',kind:'primary softened',mode:'orbit',span:.90,contrast:9,neighborTarget:1.24,voidS:.05,voidL:98,orbit:[[4,67,60],[48,72,65],[216,54,57],[0,0,25]]},
    suprematist:{name:'Suprematism Air',kind:'paper + primary',mode:'orbit',span:.84,contrast:11,neighborTarget:1.10,voidS:.04,voidL:97,orbit:[[0,0,24],[6,58,58],[45,62,64],[215,45,55],[0,0,89]]},
    constructivist:{name:'Constructivist Paper',kind:'brick / ink / linen',mode:'orbit',span:.72,contrast:13,neighborTarget:1.15,voidS:.05,voidL:96,orbit:[[4,61,57],[0,0,22],[39,19,88],[17,42,61]]},
    bauhaus:{name:'Bauhaus Daylight',kind:'primary + charcoal',mode:'orbit',span:.88,contrast:9,neighborTarget:1.22,voidS:.05,voidL:97,orbit:[[7,66,62],[47,70,65],[215,55,59],[0,0,26]]},
    albers:{name:'Albers / Interaction',kind:'warm interaction',mode:'orbit',span:.68,contrast:9,neighborTarget:.80,voidS:.08,voidL:96,orbit:[[19,51,61],[37,55,65],[55,48,68],[166,33,54],[205,36,56],[321,32,57]]},
    herrera:{name:'Carmen Herrera Soft Edge',kind:'hard-edge softened',mode:'orbit',span:.52,contrast:9,neighborTarget:1.02,voidS:.04,voidL:97,orbit:[[151,44,48],[45,15,92],[10,62,64],[0,0,24]]},
    kelly:{name:'Ellsworth Kelly Daylight',kind:'spectral daylight',mode:'full',span:.62,s:57,l:62,contrast:8,neighborTarget:1.20,voidS:.05,voidL:98},
    delaunay:{name:'Delaunay Pastel Orphism',kind:'chromatic wheel',mode:'orbit',span:.92,contrast:8,neighborTarget:1.00,voidS:.06,voidL:97,orbit:[[5,62,64],[35,68,65],[55,69,67],[143,39,57],[190,53,58],[226,52,62],[286,44,64],[337,53,67]]},
    matisse:{name:'Matisse Cut-outs',kind:'paper color orbit',mode:'orbit',span:.82,contrast:8,neighborTarget:1.00,voidS:.06,voidL:98,orbit:[[218,56,59],[3,62,65],[48,68,67],[139,40,57],[328,50,72]]},
    swiss:{name:'Swiss Warm Paper',kind:'signal red / neutral',mode:'orbit',span:.50,contrast:14,neighborTarget:1.12,voidS:.02,voidL:98,orbit:[[3,62,58],[0,0,23],[42,12,91]]},
    deco:{name:'Art Deco Pearl',kind:'emerald / champagne / ink',mode:'orbit',span:.70,contrast:12,neighborTarget:.98,voidS:.05,voidL:96,orbit:[[164,48,43],[42,48,62],[0,0,22],[35,22,88],[337,36,48]]},
    ukiyoe:{name:'Ukiyo-e Wash',kind:'indigo / vermilion / paper',mode:'orbit',span:.70,contrast:10,neighborTarget:.92,voidS:.07,voidL:95,orbit:[[210,39,47],[7,49,57],[39,44,67],[42,20,86],[0,0,25]]},
    barragan:{name:'Barragán Sunwashed',kind:'architectural sunlight',mode:'orbit',span:.86,contrast:8,neighborTarget:1.10,voidS:.06,voidL:97,orbit:[[329,54,59],[343,49,75],[43,62,64],[213,48,52],[14,51,66]]},
    midcentury:{name:'Mid-century Linen',kind:'muted modern',mode:'orbit',span:.76,contrast:7,neighborTarget:.72,voidS:.08,voidL:95,orbit:[[14,36,58],[42,46,61],[78,25,53],[177,27,52],[33,22,81]]},
    memphis:{name:'Memphis Pastel',kind:'candy geometry',mode:'orbit',span:.92,contrast:7,neighborTarget:1.08,voidS:.05,voidL:98,orbit:[[330,61,74],[194,62,70],[51,72,69],[145,39,71],[271,45,70],[8,59,70]]},
    acid:{name:'Neo Rave Pastel',kind:'fluorescent daylight',mode:'orbit',span:.94,contrast:8,neighborTarget:1.32,voidS:.04,voidL:96,orbit:[[73,76,67],[328,73,72],[188,70,66],[263,65,72],[39,78,68]]},
    jewel:{name:'Jewel Silk',kind:'jewel softened',mode:'orbit',span:.84,contrast:10,neighborTarget:1.00,voidS:.06,voidL:94,orbit:[[159,49,45],[216,50,52],[342,46,52],[286,41,52],[43,51,57]]},
    mono:{name:'Monochrome Tonal',kind:'single-hue tonal',mode:'mono',span:.66,s:39,l:65,contrast:14,neighborTarget:.38,voidS:.07,voidL:97},
    earth:{name:'Earth / Mineral',kind:'mineral daylight',mode:'orbit',span:.76,contrast:8,neighborTarget:.62,voidS:.08,voidL:95,orbit:[[13,32,57],[31,31,67],[71,21,55],[151,19,54],[206,20,55],[35,20,80]]},
    opart:{name:'Op Art Soft',kind:'warm neutral tonal',mode:'orbit',span:.92,contrast:18,neighborTarget:1.44,voidS:.01,voidL:97,orbit:[[0,0,20],[0,0,95],[0,0,44],[0,0,82]]},
    scandi:{name:'Scandinavian Daylight',kind:'birch / sky / sage',mode:'orbit',span:.62,contrast:7,neighborTarget:.76,voidS:.05,voidL:98,orbit:[[38,26,82],[202,31,73],[151,24,66],[26,30,69],[8,28,73]]},
    morris:{name:'Morris Garden',kind:'Arts & Crafts botanical',mode:'orbit',span:.70,contrast:9,neighborTarget:.82,voidS:.07,voidL:95,orbit:[[92,28,52],[146,29,47],[38,41,67],[8,37,58],[214,28,55],[331,28,62]]},
    secession:{name:'Vienna Secession',kind:'ivory / sage / gilt',mode:'orbit',span:.66,contrast:11,neighborTarget:.94,voidS:.04,voidL:97,orbit:[[44,42,70],[86,25,55],[158,27,50],[0,0,27],[32,20,88]]},
    nouveau:{name:'Art Nouveau Pastel',kind:'botanical pastel',mode:'orbit',span:.72,contrast:8,neighborTarget:.78,voidS:.06,voidL:97,orbit:[[112,29,68],[160,27,63],[196,33,70],[282,26,73],[345,34,74],[25,38,75]]},
    porcelain:{name:'Porcelain & Celadon',kind:'cobalt / celadon / porcelain',mode:'orbit',span:.58,contrast:10,neighborTarget:.92,voidS:.03,voidL:99,orbit:[[216,48,50],[196,34,68],[151,22,72],[43,16,88],[222,24,76]]},
    sorbet:{name:'Sorbet Modern',kind:'peach / mint / lemon / lilac',mode:'orbit',span:.78,contrast:6,neighborTarget:.86,voidS:.04,voidL:98,orbit:[[12,49,78],[43,55,80],[92,35,76],[157,35,76],[203,40,79],[278,35,80],[334,43,80]]},
    coastal:{name:'Sea Glass',kind:'aqua / sage / sand / sky',mode:'orbit',span:.64,contrast:7,neighborTarget:.74,voidS:.04,voidL:98,orbit:[[184,33,70],[200,37,74],[151,25,67],[45,28,79],[24,25,75]]},
    botanical:{name:'Botanical Linen',kind:'eucalyptus / fern / clay',mode:'orbit',span:.60,contrast:8,neighborTarget:.70,voidS:.06,voidL:96,orbit:[[136,25,58],[161,22,66],[88,22,62],[20,34,65],[38,29,78]]},
    rococo:{name:'Rococo Powder',kind:'powdered salon pastel',mode:'orbit',span:.74,contrast:6,neighborTarget:.82,voidS:.04,voidL:99,orbit:[[205,38,80],[334,38,82],[108,30,79],[273,30,82],[42,38,84],[168,28,80]]}
  };
  function selectSegment(root,attr,value){if(!root)return;for(const b of root.querySelectorAll(`[data-${attr}]`))b.classList.toggle('active',b.dataset[attr]===value);}
  function colorProfile(){return COLOR_PROFILES[state.colorSettings?.style]||COLOR_PROFILES.morandi;}
  function mixHue(a,b,t){const d=mod(b-a+180,360)-180;return mod(a+d*t,360);}
  function orbitSpec(profile,t){const a=profile.orbit||[[0,0,50]],n=a.length,x=mod(t,1)*n,i=Math.floor(x),u=x-i,f=u*u*(3-2*u),A=a[i%n],B=a[(i+1)%n];return{h:mixHue(A[0],B[0],f),s:A[1]+(B[1]-A[1])*f,l:A[2]+(B[2]-A[2])*f};}
  function softenSpec(z){
    // v9 daylight treatment: preserve every palette's hue/orbit identity while
    // reducing chroma and lifting shadows so large fields feel airy rather than heavy.
    const neutral=(z.s??0)<2,s=neutral?(z.s??0):Math.min(64,(z.s??0)*.72+4),l=Math.min(97.5,(z.l??50)+(100-(z.l??50))*.12);
    return{h:mod(z.h??0,360),s,l};
  }
  function globalTone(z){const q=state.colorSettings||COLOR_DEFAULTS,sMul=(q.saturation??100)/100,b=(q.brightness??100)-100;return{h:z.h,s:Math.max(0,Math.min(92,z.s*sMul)),l:Math.max(16,Math.min(98,z.l+b*.42))};}
  function ringSpec(t,profile=colorProfile()){
    const raw=(profile.mode==='full'||profile.mode==='mono')?{h:mod(t,1)*360,s:profile.s??72,l:profile.l??54}:orbitSpec(profile,t);
    return globalTone(softenSpec(raw));
  }
  function selectedStyleSpec(){const q=state.colorSettings||COLOR_DEFAULTS,p=colorProfile();return ringSpec((q.hue||0)/360,p);}
  function paletteSpec(i,count=6){
    i=Math.max(0,i|0);count=Math.max(1,count|0);const q=state.colorSettings||COLOR_DEFAULTS,p=colorProfile(),center=(q.hue||0)/360,spread=.28+.72*(q.dispersion??72)/100,spanScale=Math.pow(2,((q.orbitSpan??0)/100)*1.15),neighbor=(q.neighborBias??0)/100,neighborScale=neighbor<0?1+neighbor*.72:1+neighbor*.75,span=Math.max(.025,Math.min(.98,(p.span??.72)*spread*spanScale*neighborScale)),rank=count<=1?0:(i/(count-1)*2-1);let z;
    if(p.mode==='mono'){const tonal=Math.max(.18,Math.min(1.15,span/.72)),l=(p.l??55)+rank*23*tonal,s=(p.s??54)+(1-Math.abs(rank))*7;z=globalTone(softenSpec({h:mod(q.hue||0,360),s,l}));}
    else z=ringSpec(center+rank*span*.5,p);
    z.s=Math.max(0,Math.min(66,z.s));z.l=Math.max(34,Math.min(97.5,z.l+rank*(q.luma??26)/100*(p.contrast??9)*.58));
    z.css=`hsl(${z.h.toFixed(1)} ${z.s.toFixed(1)}% ${z.l.toFixed(1)}%)`;return z;
  }
  function paletteColor(i,count=6){i=Math.max(0,i|0);count=Math.max(1,count|0);const q=state.colorSettings||COLOR_DEFAULTS,cacheKey=`${q.style}|${q.hue}|${q.saturation}|${q.brightness}|${q.dispersion}|${q.neighborBias}|${q.orbitSpan}|${q.luma}|${i}|${count}`;if(paletteMemo.has(cacheKey))return paletteMemo.get(cacheKey);const out=paletteSpec(i,count).css;paletteMemo.set(cacheKey,out);return out;}
  function paletteDistance(a,b){const dh=Math.abs(mod(a.h-b.h+180,360)-180)*Math.PI/180,dc=2*Math.sin(dh/2),ds=(a.s-b.s)/100,dl=(a.l-b.l)/100;return Math.sqrt(dc*dc*.82+ds*ds*.16+dl*dl*1.8);}
  function decorateStyleButtons(){if(!els.paletteStyles)return;for(const b of els.paletteStyles.querySelectorAll('[data-style]')){const p=COLOR_PROFILES[b.dataset.style];if(!p)continue;const stops=[0,.34,.68].map(t=>{const z=ringSpec(t,p);return `hsl(${z.h.toFixed(0)} ${z.s.toFixed(0)}% ${z.l.toFixed(0)}%)`;});b.style.setProperty('--swatch',`linear-gradient(135deg,${stops.join(',')})`);}}
  const MATERIAL_NAMES={"plain":"plain color","ceramic":"glazed ceramic / azulejo","stained":"stained glass / leaded glass","mosaic":"mosaic tesserae","textile":"square textile medallion","garden":"square garden parterre","vintage":"vintage encaustic tile","gothic":"gothic tracery","terrazzo":"terrazzo chips","deco":"art deco fan","lacquer":"lacquer / enamel","paper":"cut paper / marquetry","pinstripe":"fine pinstripe","herringbone":"herringbone","chevron":"chevron stripe","gingham":"gingham check","trellis":"diamond trellis","arabesque":"arabesque loops","seigaiha":"seigaiha waves","asanoha":"asanoha hemp leaf","quatrefoil":"quatrefoil","guilloche":"guilloche","rosette":"rosette","sunburst":"sunburst","basketweave":"basket weave","linen":"linen weave","confetti":"restrained confetti","marble":"marble vein","leaf":"leaf sprig","clover":"clover","checker":"micro checker","maze":"Greek-key maze"};
  function refreshStyleCaption(){const p=colorProfile();if(els.paletteStyleName)els.paletteStyleName.textContent=p.name;if(els.paletteStyleKind)els.paletteStyleKind.textContent=p.kind;if(els.materialName)els.materialName.textContent=MATERIAL_NAMES[state.colorSettings?.material]||MATERIAL_NAMES.plain;}
  function updateColorOutputs(){const q=state.colorSettings||COLOR_DEFAULTS;for(const id of ['dispersion','neighborBias','orbitSpan','luma','edge','cellEdge','materialStrength','patternScale','patternWidth','patternOpacity','patternContrast']){const o=$(id+'Out');if(o)o.textContent=String(q[id]);}if($('saturationOut'))$('saturationOut').textContent=`${q.saturation}%`;if($('brightnessOut'))$('brightnessOut').textContent=`${q.brightness}%`;if(els.extraColorsOut)els.extraColorsOut.textContent=`+${q.extraColors||0}`;}
  function invalidateArt(){paletteMemo.clear();state.colorCacheKey='';state.colorMap=null;state.rasterCacheKey='';state.rasterCache=null;refreshStyleCaption();updateColorOutputs();drawHueWheel();if(els.colorPreview)els.colorPreview.style.background=`linear-gradient(90deg,${Array.from({length:7},(_,i)=>paletteColor(i,7)).join(',')})`;scheduleDraw();}
  function readColorSettings(){
    for(const id of ['saturation','brightness','dispersion','neighborBias','orbitSpan','luma','edge','cellEdge','materialStrength','patternScale','patternWidth','patternOpacity','patternContrast'])if(els[id])state.colorSettings[id]=+els[id].value;
    if(els.patternTransform)state.colorSettings.patternTransform=els.patternTransform.value;if(els.voidMode)state.colorSettings.voidMode=els.voidMode.value;if(els.edgeLineStyle)state.colorSettings.edgeLineStyle=els.edgeLineStyle.value;if(els.edgeColorMode)state.colorSettings.edgeColorMode=els.edgeColorMode.value;
    updateColorOutputs();invalidateArt();return state.colorSettings;
  }
  function setHueFromPointer(e){const c=els.hueWheel,rect=c.getBoundingClientRect(),sx=c.width/rect.width,sy=c.height/rect.height,x=(e.clientX-rect.left)*sx,y=(e.clientY-rect.top)*sy,cx=c.width*.28,cy=c.height*.5,a=Math.atan2(y-cy,x-cx)*180/Math.PI+90;state.colorSettings.hue=mod(a,360);invalidateArt();}
  function initColorSettings(){
    decorateStyleButtons();
    for(const el of [els.saturation,els.brightness,els.dispersion,els.neighborBias,els.orbitSpan,els.luma,els.edge,els.cellEdge,els.materialStrength,els.patternScale,els.patternWidth,els.patternOpacity,els.patternContrast])el?.addEventListener('input',readColorSettings);for(const el of [els.voidMode,els.edgeLineStyle,els.edgeColorMode,els.patternTransform])el?.addEventListener('change',readColorSettings);
    els.extraColorsDec?.addEventListener('click',()=>{state.colorSettings.extraColors=Math.max(0,(state.colorSettings.extraColors||0)-1);invalidateArt();});
    els.extraColorsInc?.addEventListener('click',()=>{state.colorSettings.extraColors=Math.min(12,(state.colorSettings.extraColors||0)+1);invalidateArt();});
    els.paletteStyles?.addEventListener('click',e=>{const b=e.target.closest('[data-style]');if(!b)return;state.colorSettings.style=b.dataset.style;selectSegment(els.paletteStyles,'style',state.colorSettings.style);invalidateArt();});
    els.tileMaterials?.addEventListener('click',e=>{const b=e.target.closest('[data-material]');if(!b)return;state.colorSettings.material=b.dataset.material;selectSegment(els.tileMaterials,'material',state.colorSettings.material);invalidateArt();});
    let hueDrag=false;els.hueWheel?.addEventListener('pointerdown',e=>{hueDrag=true;els.hueWheel.setPointerCapture?.(e.pointerId);setHueFromPointer(e);});els.hueWheel?.addEventListener('pointermove',e=>{if(hueDrag)setHueFromPointer(e);});const end=()=>hueDrag=false;els.hueWheel?.addEventListener('pointerup',end);els.hueWheel?.addEventListener('pointercancel',end);
    els.resetColorSettings?.addEventListener('click',()=>{state.colorSettings={...COLOR_DEFAULTS};for(const id of ['saturation','brightness','dispersion','neighborBias','orbitSpan','luma','edge','cellEdge','materialStrength','patternScale','patternWidth','patternOpacity','patternContrast'])if(els[id])els[id].value=String(COLOR_DEFAULTS[id]);if(els.patternTransform)els.patternTransform.value=COLOR_DEFAULTS.patternTransform;if(els.voidMode)els.voidMode.value=COLOR_DEFAULTS.voidMode;if(els.edgeLineStyle)els.edgeLineStyle.value=COLOR_DEFAULTS.edgeLineStyle;if(els.edgeColorMode)els.edgeColorMode.value=COLOR_DEFAULTS.edgeColorMode;selectSegment(els.paletteStyles,'style',COLOR_DEFAULTS.style);selectSegment(els.tileMaterials,'material',COLOR_DEFAULTS.material);readColorSettings();});
    selectSegment(els.paletteStyles,'style',state.colorSettings.style);selectSegment(els.tileMaterials,'material',state.colorSettings.material);for(const id of ['saturation','brightness'])if(els[id])els[id].value=String(state.colorSettings[id]);updateColorOutputs();readColorSettings();
  }

  function searchSignature(cells=normalizeShape()){const q=state.searchSettings;return `${canonicalShapeSig(cells)}|${state.group}|${state.mode}|${state.algorithm}|${q.cellPieces}|${q.aggression}|${q.complexity}|${q.regularity}`;}
  function updateSolveLabel(){if(!els.solveLabel)return;const canContinue=!!state.results.length&&state.lastSearchSignature===searchSignature();els.solveLabel.textContent=canContinue?t('continue'):t('generate');els.solveBtn?.classList.toggle('continue',canContinue);}

  const packLoads = new Map(), packOrder=[];
  function touchPack(n){const i=packOrder.indexOf(n);if(i>=0)packOrder.splice(i,1);packOrder.push(n);const keepSelected=state.catalogId?Number((/^P(\d+)-/.exec(state.catalogId)||[])[1]||0):0;while(packOrder.length>4){const victim=packOrder.find(x=>x!==state.catalogN&&x!==keepSelected&&x!==5);if(victim==null)break;packOrder.splice(packOrder.indexOf(victim),1);delete window.__TL_PACKS__?.[victim];}}
  function loadScript(src){
    return new Promise((resolve,reject)=>{const el=document.createElement('script'),timer=setTimeout(()=>{el.remove();reject(new Error(`load timeout: ${src}`));},8000);el.src=src;el.async=true;el.onload=()=>{clearTimeout(timer);el.remove();resolve();};el.onerror=()=>{clearTimeout(timer);el.remove();reject(new Error(`load failed: ${src}`));};document.head.appendChild(el);});
  }
  async function ensurePack(n){
    window.__TL_PACKS__=window.__TL_PACKS__||{};if(window.__TL_PACKS__[n]){touchPack(n);return window.__TL_PACKS__[n];}if(packLoads.has(n))return packLoads.get(n);
    const p=(async()=>{await loadScript(`data/runtime/catalog-pack-n${n}.js`);const pack=window.__TL_PACKS__[n];if(!pack)throw new Error(`catalog pack ${n} unavailable`);touchPack(n);return pack;})();packLoads.set(n,p);try{return await p;}finally{packLoads.delete(n);}
  }
  function parseCells(encoded){return String(encoded||'').split(';').filter(Boolean).map(pair=>pair.split(',').map(Number));}
  function expandPacked(s,pack){
    const seedRec=pack.seeds?.[s.id]||{},groups=Object.keys(seedRec),dims={};for(const g of groups){const z=seedRec[g];if(z)dims[g]=[z.w,z.h];}
    const m=s.m||{},h=m.h?{Hc:m.h[0],Hh:m.h[1]}:null;
    return{id:s.id,cells:parseCells(s.c),w:s.w,h:s.h,perimeter:s.p,orientations:s.o,heesch:h,fact:m.f?{facts:m.f}:null,seedGroups:groups,seedDims:dims,complexity:m.x||null,rectangle:m.r||null};
  }
  function filteredPack(pack,q){
    q=String(q||'').trim().toUpperCase();let arr=pack.shapes||[];if(!q)return arr;
    const hm=q.match(/^H(?:>=?)?(\d+)$/);if(hm){const k=+hm[1];return arr.filter(s=>s.m?.h&&Math.max(s.m.h[0],s.m.h[1])>=k);}
    if(q==='⊞'||q==='TILE'||q==='TILER')return arr.filter(s=>pack.seeds?.[s.id]);
    if(q==='★'||q==='*'||q==='MU'||q==='COMPLEX')return arr.filter(s=>s.m?.x).slice().sort((a,b)=>maxMu(b)-maxMu(a));
    const mm=q.match(/^(?:★|MU)(?:>=?)?(\d+)$/);if(mm){const k=+mm[1];return arr.filter(s=>maxMu(s)>=k).slice().sort((a,b)=>maxMu(b)-maxMu(a));}
    return arr.filter(s=>s.id.includes(q));
  }
  function maxMu(s){const x=s.m?.x||{};let m=0;for(const v of Object.values(x))m=Math.max(m,Number(v?.mu)||0);return m;}
  async function requestCatalog(){
    const token=++state.catalogLoadToken,n=state.catalogN;state.catalogShapes=[];els.catalogGrid.innerHTML=`<div class="catalog-loading">${t('loading')}…</div>`;els.catalogCount.textContent=`n=${n}`;els.catalogCheck.textContent='·';
    try{const pack=await ensurePack(n);if(token!==state.catalogLoadToken||n!==state.catalogN)return;state.catalogCount=pack.count;els.catalogCheck.textContent=pack.count===EXPECTED[n]?'✓':'!';els.catalogCheck.title=`${pack.count} / ${EXPECTED[n]??'?'}`;requestCatalogPage(0);setStatus(`${t('catalog')} ${n} · ${pack.count}`);}catch(err){if(token!==state.catalogLoadToken)return;els.catalogGrid.innerHTML=`<div class="catalog-loading">${err.message}</div>`;els.catalogCheck.textContent='!';setStatus(err.message,'error');}
  }
  function requestCatalogPage(page=0){
    const pack=window.__TL_PACKS__?.[state.catalogN];if(!pack){requestCatalog();return;}const arr=filteredPack(pack,els.catalogSearch.value.trim()),pages=Math.max(1,Math.ceil(arr.length/PAGE_SIZE));page=Math.max(0,Math.min(pages-1,Number(page)||0));state.catalogPage=page;state.catalogPages=pages;state.catalogCount=arr.length;state.catalogShapes=arr.slice(page*PAGE_SIZE,(page+1)*PAGE_SIZE).map(s=>expandPacked(s,pack));renderCatalog();
  }
  function sampleCatalog(){const pack=window.__TL_PACKS__?.[state.catalogN];if(!pack?.shapes?.length)return;const s=pack.shapes[(Math.random()*pack.shapes.length)|0];const x=expandPacked(s,pack);setShape(x.cells,x.id,x);}
  function renderCatalog(){
    els.catalogCount.textContent=`n=${state.catalogN} · ${state.catalogCount}`;els.pageLabel.textContent=`${state.catalogPage+1} / ${state.catalogPages}`;els.prevPage.disabled=state.catalogPage<=0;els.nextPage.disabled=state.catalogPage>=state.catalogPages-1;els.catalogGrid.innerHTML='';
    const frag=document.createDocumentFragment();for(const s of state.catalogShapes){const b=document.createElement('button');b.type='button';b.className='catalog-card'+(s.id===state.catalogId?' active':'');b.dataset.id=s.id;const tags=[];if(s.seedGroups?.includes(state.group)){const d=s.seedDims?.[state.group];tags.push(d?`⊞${d[0]}×${d[1]}`:'⊞');}if(s.rectangle?.[state.group])tags.push('▭');const cx=s.complexity?.[state.group];if(cx?.mu>1)tags.push(`★${fmt(cx.mu,0)}`);if(s.heesch)tags.push(`H${Math.max(s.heesch.Hc,s.heesch.Hh)}`);if(s.fact)tags.push('T');const badge=tags.length?` · ${tags.join(' ')}`:'';b.title=`${s.id} · ${s.w}×${s.h} · p=${s.perimeter} · o=${s.orientations}${badge}`;const c=document.createElement('canvas');c.width=100;c.height=64;drawMini(c,s.cells);const id=document.createElement('span');id.className='catalog-id';id.textContent=s.id+badge;b.append(c,id);frag.appendChild(b);}els.catalogGrid.appendChild(frag);
  }
  function drawMini(canvas,cells){const ctx=canvas.getContext('2d'),b=shapeBounds(cells),pad=8,s=Math.min((canvas.width-2*pad)/b.w,(canvas.height-2*pad)/b.h),ox=(canvas.width-b.w*s)/2,oy=(canvas.height-b.h*s)/2;ctx.clearRect(0,0,canvas.width,canvas.height);ctx.fillStyle='#657067';for(const[x,y]of cells)ctx.fillRect(ox+(x-b.minX)*s+.8,oy+(y-b.minY)*s+.8,s-1.6,s-1.6);}

  function localNormalize(cells){if(!cells?.length)return[];let minX=Infinity,minY=Infinity;for(const[x,y]of cells){minX=Math.min(minX,x);minY=Math.min(minY,y);}return cells.map(([x,y])=>[x-minX,y-minY]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]);}
  function localOris(shape,group){const fs={I:(x,y)=>[x,y],R90:(x,y)=>[-y,x],R180:(x,y)=>[-x,-y],R270:(x,y)=>[y,-x],MX:(x,y)=>[-x,y],MY:(x,y)=>[x,-y],D:(x,y)=>[y,x],AD:(x,y)=>[-y,-x]},names=group==='C1'?['I']:group==='K4'?['I','R180','MX','MY']:group==='C4'?['I','R90','R180','R270']:['I','R90','R180','R270','MX','MY','D','AD'],out=[],seen=new Set();for(const n of names){const c=localNormalize(shape.map(([x,y])=>fs[n](x,y))),k=c.map(v=>v.join(',')).join(';');if(!seen.has(k)){seen.add(k);out.push(c);}}return out;}
  function reduceByLattice(x,y,L){const a=Math.max(1,Math.round(Number(L?.a)||1)),c=Math.max(1,Math.round(Number(L?.c)||1)),b=Math.round(Number(L?.b)||0),q=Math.floor(y/c),yy=y-q*c,x1=x-q*b,p=Math.floor(x1/a),xx=x1-p*a;return{x:xx,y:yy,p,q};}
  function reduceLattice(x,y,r){const L=r?.lattice||{a:r.w,b:0,c:r.h};return reduceByLattice(x,y,L);}
  function latticeRawCells(p){return Array.isArray(p?.rawCells)?p.rawCells:(Array.isArray(p?.cells)?p.cells:[]);}
  function quotientCellSig(cells,L,prefix='P',dx=0,dy=0){const ids=[];for(const[x,y]of cells){const z=reduceByLattice(x+dx,y+dy,L);ids.push(z.y*L.a+z.x);}ids.sort((a,b)=>a-b);return`${prefix}${cells.length}:${ids.join('.')}`;}
  function liftedTileSig(cells,L,prefix='P',dx=0,dy=0){const pts=(cells||[]).map(([x,y])=>[x+dx,y+dy]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]);if(!pts.length)return`${prefix}0:`;const z=reduceByLattice(pts[0][0],pts[0][1],L),lx=z.p*L.a+z.q*L.b,ly=z.q*L.c;return`${prefix}${pts.length}:`+pts.map(([x,y])=>`${x-lx},${y-ly}`).join(';');}
  function periodicElements(r){const out=[];for(const p of r.placements||[])out.push({kind:'P',cells:latticeRawCells(p)});for(const c of r.companion?.components||[])out.push({kind:'C',cells:Array.isArray(c?.cells)?c.cells:(c?.quotientCells||[])});return out.filter(x=>x.cells.length);}
  function minimalTranslationLattice(r){
    const old={a:Math.max(1,Math.round(Number(r?.lattice?.a)||r.w||1)),b:Math.round(Number(r?.lattice?.b)||0),c:Math.max(1,Math.round(Number(r?.lattice?.c)||r.h||1))},els0=periodicElements(r);if(!els0.length)return old;const base=new Set(els0.map(e=>liftedTileSig(e.cells,old,e.kind))),memo=new Map();
    const symmetry=(tx,ty)=>{const zr=reduceByLattice(tx,ty,old),mk=`${zr.x},${zr.y}`;if(memo.has(mk))return memo.get(mk);if(zr.x===0&&zr.y===0){memo.set(mk,true);return true;}for(const e of els0)if(!base.has(liftedTileSig(e.cells,old,e.kind,tx,ty))){memo.set(mk,false);return false;}memo.set(mk,true);return true;};
    let a=old.a;for(let x=1;x<=old.a;x++)if(symmetry(x,0)){a=x;break;}let c=old.c,b=mod(old.b,a),found=false;for(let y=1;y<=old.c&&!found;y++)for(let x=0;x<a;x++)if(symmetry(x,y)){c=y;b=x;found=true;break;}const det=a*c,oldDet=old.a*old.c;if(det<1||oldDet%det!==0)return old;return{a,b,c,det};
  }
  function shiftPlacementToLattice(p,L){const raw=latticeRawCells(p);if(!raw.length)return null;const z=reduceByLattice(raw[0][0],raw[0][1],L),dx=z.p*L.a+z.q*L.b,dy=z.q*L.c,rawCells=raw.map(([x,y])=>[x-dx,y-dy]),cells=rawCells.map(([x,y])=>{const q=reduceByLattice(x,y,L);return[q.x,q.y];}),anchor=Array.isArray(p.anchor)?[p.anchor[0]-dx,p.anchor[1]-dy]:rawCells[0].slice();return{...p,rawCells,cells,anchor};}
  function shortestLatticeVector(L){let best=Infinity;for(let q=-5;q<=5;q++)for(let p=-5;p<=5;p++){if(!p&&!q)continue;const x=p*L.a+q*L.b,y=q*L.c,d=Math.hypot(x,y);if(d>1e-9&&d<best)best=d;}return Number.isFinite(best)?best:Math.min(L.a,L.c);}
  function gaussReducedBasis(L){let u=[Math.round(L.a),0],v=[Math.round(L.b||0),Math.round(L.c)];for(let it=0;it<24;it++){const nu=u[0]*u[0]+u[1]*u[1],nv=v[0]*v[0]+v[1]*v[1];if(nv<nu){[u,v]=[v,u];continue;}const mu=Math.round((u[0]*v[0]+u[1]*v[1])/Math.max(1,nu));if(mu===0)break;v=[v[0]-mu*u[0],v[1]-mu*u[1]];}if(u[0]*v[1]-u[1]*v[0]<0)v=[-v[0],-v[1]];return[u,v];}
  function reducePeriodicResult(r){
    if(!r||r._latticeReduced||!(r.kind==='periodic'||r.kind==='cell'||r.kind==='field'))return r;const old={a:Number(r?.lattice?.a)||r.w,b:Number(r?.lattice?.b)||0,c:Number(r?.lattice?.c)||r.h},L=minimalTranslationLattice(r),oldDet=old.a*old.c,newDet=L.a*L.c;if(!(newDet<oldDet-1e-9)){r._latticeReduced=true;r.lattice={...(r.lattice||{}),a:old.a,b:old.b,c:old.c,det:oldDet,key:`${old.a},${old.b},${old.c}`,basis:gaussReducedBasis(old)};if(r.metrics){r.metrics.latticeArea=oldDet;r.metrics.shortestPeriod=shortestLatticeVector(old);}return r;}
    const seen=new Set(),placements=[];for(const p of r.placements||[]){const q=shiftPlacementToLattice(p,L);if(!q)continue;const k=liftedTileSig(q.rawCells,L,'P');if(seen.has(k))continue;seen.add(k);placements.push(q);}let companion=r.companion;if(companion?.components){const cs=[],cseen=new Set();for(const p of companion.components){const q=shiftPlacementToLattice(p,L);if(!q)continue;const k=liftedTileSig(q.rawCells,L,'C');if(cseen.has(k))continue;cseen.add(k);q.quotientCells=q.cells.map(x=>x.slice());cs.push(q);}companion={...companion,components:cs,placements:cs,copies:cs.length,totalCells:cs.reduce((n,x)=>n+x.cells.length,0)};}
    const metrics={...(r.metrics||{})},primaryCells=placements.reduce((n,p)=>n+p.cells.length,0),compCells=companion?.components?.reduce((n,p)=>n+(p.cells?.length||0),0)||0;metrics.coverage=primaryCells/newDet;metrics.combinedCoverage=Math.min(1,(primaryCells+compCells)/newDet);metrics.fundArea=newDet;metrics.latticeArea=newDet;metrics.primitiveTileCount=placements.length;metrics.periodPieces=placements.length;metrics.tileCount=placements.length;metrics.shortestPeriod=shortestLatticeVector(L);metrics.translationSymmetries=Math.max(1,Math.round(oldDet/newDet));if(companion){metrics.companionCopies=companion.components?.length||0;metrics.companionCoverage=compCells/Math.max(1,newDet-primaryCells);metrics.unmatchedVoid=Math.max(0,newDet-primaryCells-compCells);}
    const out={...r,w:L.a,h:L.c,lattice:{...(r.lattice||{}),a:L.a,b:L.b,c:L.c,det:newDet,key:`${L.a},${L.b},${L.c}`,basis:gaussReducedBasis(L)},placements,companion,metrics,_latticeReduced:true,_latticeReducedFrom:oldDet};return validateResultPacking(out)?out:r;
  }
  function validateResultPacking(r){
    if(!r||!Number.isInteger(r.w)||!Number.isInteger(r.h)||r.w<1||r.h<1||!Array.isArray(r.placements))return false;const periodic=r.kind==='periodic'||r.kind==='cell'||r.kind==='field',area=r.w*r.h,owner=new Int32Array(area),expected=normalizeShape().length;owner.fill(-1);
    for(let i=0;i<r.placements.length;i++){const p=r.placements[i],raw=Array.isArray(p?.rawCells)?p.rawCells:p?.cells,seen=new Set();if(!Array.isArray(raw)||!raw.length||(expected&&raw.length!==expected))return false;for(const c of raw){if(!Array.isArray(c)||c.length<2)return false;const rx=Number(c[0]),ry=Number(c[1]);if(!Number.isInteger(rx)||!Number.isInteger(ry))return false;let x=rx,y=ry;if(periodic){const z=reduceLattice(x,y,r);x=z.x;y=z.y;}else if(x<0||x>=r.w||y<0||y>=r.h)return false;const id=y*r.w+x;if(seen.has(id)||owner[id]>=0)return false;seen.add(id);owner[id]=i;}if(expected&&seen.size!==expected)return false;}
    if(periodic){const a=r.lattice?.a||r.w,b=r.lattice?.b||0,c=r.lattice?.c||r.h,lifted=new Set();for(let q=-1;q<=1;q++)for(let p=-1;p<=1;p++)for(const tile of r.placements){const raw=Array.isArray(tile.rawCells)?tile.rawCells:tile.cells;for(const[x,y]of raw){const k=`${x+p*a+q*b},${y+q*c}`;if(lifted.has(k))return false;lifted.add(k);}}if(r.companion?.components){for(const comp of r.companion.components)for(const cell of comp.quotientCells||[]){const z=reduceLattice(cell[0],cell[1],r),id=z.y*r.w+z.x;if(owner[id]>=0)return false;}}}
    return true;
  }

  function localFallbackSolve(m,emit,dead){
    const shape=localNormalize(m.shape||[]),n=shape.length,oris=localOris(shape,m.group||'C1'),opt=m.options||{},start=performance.now();
    const send=x=>{if(!dead())emit(x);};send({type:'progress',phase:'prepare',engine:'L',elapsed:0,percent:0});
    const seed=m.cachedSeed;if(seed?.p?.length){const ps=[],occ=new Uint8Array(seed.w*seed.h);let ok=true;for(const[oi,ax,ay]of seed.p){if(!oris[oi]){ok=false;break;}const rawCells=oris[oi].map(([dx,dy])=>[ax+dx,ay+dy]),cells=rawCells.map(([x,y])=>[mod(x,seed.w),mod(y,seed.h)]);if(new Set(cells.map(v=>v.join(','))).size!==n){ok=false;break;}for(const[x,y]of cells){const id=y*seed.w+x;if(occ[id]){ok=false;break;}occ[id]=1;}if(!ok)break;ps.push({cells,rawCells,ori:oi,anchor:[ax,ay]});}if(ok){let covered=0;for(const v of occ)covered+=v;const r={id:'L-cache',kind:covered===seed.w*seed.h?'periodic':'cell',label:'periodic cell',w:seed.w,h:seed.h,engine:'cache',placements:ps,colorClasses:ps.map((_,i)=>i),metrics:{coverage:covered/(seed.w*seed.h),fundArea:seed.w*seed.h,primitiveTileCount:ps.length,adjEntropy:0,oriEntropy:0,shortestPeriod:Math.min(seed.w,seed.h),score:1000+covered}};if(validateResultPacking(r)){send({type:'preview',result:r,engine:'cache',elapsed:performance.now()-start});send({type:'result',results:[r],elapsed:performance.now()-start,reason:'local-cache'});return;}}}
    const bb=shapeBounds(shape),size=Math.min(46,Math.max(26,Math.round(Math.sqrt(Math.max(1,n)*72)),bb.w+8,bb.h+8)),w=size,h=size,occ=new Uint8Array(w*h),placements=[];for(let oi=0;oi<oris.length;oi++){let maxX=0,maxY=0;for(const[x,y]of oris[oi]){maxX=Math.max(maxX,x);maxY=Math.max(maxY,y);}for(let y=0;y<h-maxY;y++)for(let x=0;x<w-maxX;x++){const cells=oris[oi].map(([dx,dy])=>[x+dx,y+dy]),idx=cells.map(([xx,yy])=>yy*w+xx);placements.push({cells,idx,ori:oi,anchor:[x,y]});}}
    let z=(Date.now()^n^size)>>>0;for(let i=placements.length-1;i>0;i--){z=(Math.imul(z,1664525)+1013904223)>>>0;const j=z%(i+1);[placements[i],placements[j]]=[placements[j],placements[i]];}const chosen=[];for(const p of placements){let ok=true;for(const id of p.idx)if(occ[id]){ok=false;break;}if(!ok)continue;for(const id of p.idx)occ[id]=1;chosen.push(p);}let covered=chosen.length*n;const holes=[];for(let i=0;i<occ.length;i++)if(!occ[i])holes.push([i%w,(i/w)|0]);const coverage=covered/(w*h),r={id:'L-approx',kind:'cell',label:m.mode==='companion'?'companion':'periodic cell',w,h,engine:'local',placements:chosen.map(p=>({cells:p.cells,rawCells:p.cells,ori:p.ori,anchor:p.anchor})),holes,metrics:{coverage,fundArea:NaN,adjEntropy:NaN,oriEntropy:NaN,shortestPeriod:NaN,holeSpread:0,score:coverage*1000}};send({type:'preview',result:r,engine:'local',elapsed:performance.now()-start});send({type:'result',results:[r],elapsed:performance.now()-start,reason:'approximate'});
  }
  function createLocalSolverAdapter(){let dead=false;const a={onmessage:null,onerror:null,postMessage(m){if(dead)return;if(m.type==='ping'){setTimeout(()=>a.onmessage?.({data:{type:'ready',version:'7-local'}}),0);return;}if(m.type==='solve'){setTimeout(()=>{try{localFallbackSolve(m,x=>a.onmessage?.({data:x}),()=>dead);}catch(err){a.onerror?.(err);}},0);}},terminate(){dead=true;}};return a;}
  function cleanupSolverWorker(){clearTimeout(state.workerBootTimer);state.workerBootTimer=null;try{state.solverWorker?.terminate();}catch{}state.solverWorker=null;if(state.solverUrl){URL.revokeObjectURL(state.solverUrl);state.solverUrl=null;}state.solverReady=false;}
  function bindSolverWorker(w,mode){
    state.solverWorker=w;state.workerMode=mode;state.solverReady=false;clearTimeout(state.workerBootTimer);
    const fail=e=>{console.warn('solver backend failed',mode,e);if(state.solverReady){setBusy(false);setStatus(`solver error${e?.message?` · ${e.message}`:''}`,'error');return;}tryNextSolverBackend(mode);};
    w.onmessage=e=>{if(e.data?.type==='ready'){state.solverReady=true;clearTimeout(state.workerBootTimer);}onSolverMessage(e);};w.onerror=fail;
    state.workerBootTimer=setTimeout(()=>{if(!state.solverReady)fail(new Error('worker handshake timeout'));},900);try{w.postMessage({type:'ping'});}catch(e){fail(e);}
  }
  function tryNextSolverBackend(previous=''){
    cleanupSolverWorker();const src=window.__TL_SOLVER_SOURCE__;
    if(previous!=='url'&&location.protocol!=='file:'){try{bindSolverWorker(new Worker('solver.worker.js'),'url');return;}catch(e){console.warn(e);previous='url';}}
    if(previous!=='blob'&&src){try{const blob=new Blob([src],{type:'text/javascript'});state.solverUrl=URL.createObjectURL(blob);bindSolverWorker(new Worker(state.solverUrl),'blob');return;}catch(e){console.warn(e);previous='blob';}}
    bindSolverWorker(createLocalSolverAdapter(),'local');
  }
  function createSolverWorker(){cleanupSolverWorker();tryNextSolverBackend('');}
  function resultCoverage(r){return Number.isFinite(r?.metrics?.coverage)?r.metrics.coverage:0;}
  function resultCombined(r){const q=r?.metrics?.combinedCoverage;return Number.isFinite(q)?q:resultCoverage(r);}
  function resultPeriod(r){const p=Number(r?.metrics?.primitiveTileCount??r?.metrics?.periodPieces??r?.metrics?.tileCount??r?.placements?.length);return Number.isFinite(p)?Math.max(0,Math.round(p)):0;}
  function resultCompare(a,b){const ca=resultCoverage(a),cb=resultCoverage(b),comp=(a?.searchMode||state.mode)==='companion'||(b?.searchMode||state.mode)==='companion';if(comp){const xa=a?.metrics?.combinedCoverage??ca,xb=b?.metrics?.combinedCoverage??cb;if(Math.abs(xb-xa)>1e-10)return xb-xa;if(Math.abs(cb-ca)>1e-10)return cb-ca;const aa=Number.isFinite(a?.metrics?.companionArea)?a.metrics.companionArea:1e9,ab=Number.isFinite(b?.metrics?.companionArea)?b.metrics.companionArea:1e9;if(aa!==ab)return aa-ab;}else if(Math.abs(cb-ca)>1e-10)return cb-ca;const sa=a?.metrics?.score||0,sb=b?.metrics?.score||0;if(Math.abs(sb-sa)>1e-9)return sb-sa;return (a?.metrics?.latticeArea||a.w*a.h)-(b?.metrics?.latticeArea||b.w*b.h);}
  const EQUIV_TRANSFORMS=[(x,y)=>[x,y],(x,y)=>[-y,x],(x,y)=>[-x,-y],(x,y)=>[y,-x],(x,y)=>[-x,y],(x,y)=>[x,-y],(x,y)=>[y,x],(x,y)=>[-y,-x]];
  function finiteEquivalenceSig(r){
    const groups=[...(r.placements||[]).map(p=>({kind:'P',cells:p.cells||[]})),...(r.companion?.components||[]).map(p=>({kind:'C',cells:p.cells||p.quotientCells||[]}))].filter(x=>x.cells.length);if(!groups.length)return'';let best=null;
    for(const f of EQUIV_TRANSFORMS){const tg=groups.map(g=>({kind:g.kind,cells:g.cells.map(([x,y])=>f(x,y))}));let minX=Infinity,minY=Infinity;for(const g of tg)for(const[x,y]of g.cells){minX=Math.min(minX,x);minY=Math.min(minY,y);}const parts=tg.map(g=>g.kind+':'+g.cells.map(([x,y])=>[x-minX,y-minY]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]).map(v=>v.join(',')).join(';')).sort(),sig=parts.join('|');if(best===null||sig<best)best=sig;}return best||'';
  }
  function gcdInt(a,b){a=Math.abs(Math.round(a));b=Math.abs(Math.round(b));while(b){const t=a%b;a=b;b=t;}return a;}
  function xgcdInt(a,b){const sa=a<0?-1:1,sb=b<0?-1:1;let x0=1,y0=0,x1=0,y1=1,A=Math.abs(Math.round(a)),B=Math.abs(Math.round(b));while(B){const q=Math.floor(A/B),t=A%B;A=B;B=t;[x0,x1]=[x1,x0-q*x1];[y0,y1]=[y1,y0-q*y1];}return[A,x0*sa,y0*sb];}
  function hnfFromBasis(u,v){const ux=Math.round(u[0]),uy=Math.round(u[1]),vx=Math.round(v[0]),vy=Math.round(v[1]),det=Math.abs(ux*vy-uy*vx);if(!det)return{a:1,b:0,c:1,det:1};const c=Math.max(1,gcdInt(uy,vy)),eg=xgcdInt(uy,vy),a=Math.max(1,Math.round(det/c)),x=eg[1]*ux+eg[2]*vx,b=mod(x,a);return{a,b,c,det};}
  function periodicEquivalenceSig(r){
    const old={a:Number(r?.lattice?.a)||r.w,b:Number(r?.lattice?.b)||0,c:Number(r?.lattice?.c)||r.h},base=periodicElements(r);if(!base.length)return'';let best=null;
    for(const f of EQUIV_TRANSFORMS){const u=f(old.a,0),v=f(old.b,old.c),L=hnfFromBasis(u,v),tiles=base.map(e=>({kind:e.kind,cells:e.cells.map(([x,y])=>f(x,y))}));for(const root of tiles){const pts=root.cells.slice().sort((a,b)=>a[1]-b[1]||a[0]-b[0]);if(!pts.length)continue;const rx=pts[0][0],ry=pts[0][1],parts=tiles.map(e=>liftedTileSig(e.cells,L,e.kind,-rx,-ry)).sort(),sig=`L${L.a},${L.b},${L.c}|${parts.join('|')}`;if(best===null||sig<best)best=sig;}}
    return best||'';
  }
  function resultKey(r){if(r?._equivKey)return r._equivKey;const periodic=r&&(r.kind==='periodic'||r.kind==='cell'||r.kind==='field'),body=periodic?periodicEquivalenceSig(r):finiteEquivalenceSig(r),comp=(r?.searchMode==='companion'||r?.companion)?`|C${r?.metrics?.companionArea??'x'}`:'';const k=`${r?.searchMode||''}|${periodic?'T':'F'}|m${resultPeriod(r)}|ρ${Math.round(resultCoverage(r)*1e8)}${comp}|${body}`;try{Object.defineProperty(r,'_equivKey',{value:k,writable:true,configurable:true});}catch{r._equivKey=k;}return k;}
  function mergeResults(incoming){const all=[...state.results,...(incoming||[])].filter(Boolean).map(cleanResult).filter(Boolean),seen=new Set(),out=[];for(const r of all.sort(resultCompare)){const k=resultKey(r);if(seen.has(k))continue;seen.add(k);out.push(r);if(out.length>=64)break;}return out;}
  function mergedRepresentative(candidate){if(!candidate)return state.results[0]||null;const k=resultKey(candidate);return state.results.find(r=>resultKey(r)===k)||state.results[0]||null;}
  function tagResult(r){if(!r)return r;r.searchMode=r.searchMode||state.inflightMode||state.mode;r.searchGroup=r.searchGroup||state.inflightGroup||state.group;return r;}
  function shouldShowPreview(r){if(!r)return false;const base=[state.previewResult,state.activeResult,...state.results.slice(0,8)].filter(x=>x&&(x.searchMode||state.mode)===(r.searchMode||state.mode)).sort(resultCompare)[0];return !base||resultCompare(r,base)<=0;}
  function rememberExactResult(r){if(!state.catalogId||r?.kind!=='periodic'||!Array.isArray(r.placements)||!r.placements.length||r.lattice?.b)return;const pack=selectedPack();if(!pack)return;pack.seeds=pack.seeds||{};pack.seeds[state.catalogId]=pack.seeds[state.catalogId]||{};const prev=pack.seeds[state.catalogId][state.group],rec={w:r.w,h:r.h,p:r.placements.map(p=>[Number(p.ori)||0,Number(p.anchor?.[0])||0,Number(p.anchor?.[1])||0])};if(!prev||r.w*r.h<prev.w*prev.h)pack.seeds[state.catalogId][state.group]=rec;if(state.catalogMeta){state.catalogMeta.seedGroups=Object.keys(pack.seeds[state.catalogId]);state.catalogMeta.seedDims=state.catalogMeta.seedDims||{};state.catalogMeta.seedDims[state.group]=[r.w,r.h];}}

  function onSolverMessage(e){const m=e.data||{};
    if(m.type==='ready'){state.solverReady=true;clearTimeout(state.workerBootTimer);if(state.pendingSolveMessage){const q=state.pendingSolveMessage;state.pendingSolveMessage=null;try{state.solverWorker.postMessage(q);}catch(err){setBusy(false);setStatus(`solver error · ${err.message}`,'error');}}else if(!state.busy)setStatus(state.workerMode==='local'?`${t('ready')} · local`:t('ready'));}
    else if(m.type==='progress'){state.lastProgress=m;updateProgress(m);setStatus(`${phaseLabel(m.phase)}${m.w?` · ${m.w}×${m.h}`:''}`,'busy');}
    else if(m.type==='preview'){const r=tagResult(cleanResult(m.result));if(shouldShowPreview(r)){state.previewResult=r;state.colorCacheKey='';if(!state.previewFitted){centerViewOnResult(r,true);state.previewFitted=true;}updateMetrics(r);scheduleDraw();}updateLiveHud(m);}
    else if(m.type==='partial'){const incoming=(m.results||[]).map(r=>tagResult(cleanResult(r))).filter(Boolean),best=incoming.slice().sort(resultCompare)[0]||null;state.results=mergeResults(incoming);if(best)state.activeResult=mergedRepresentative(best);renderResults();updateMetrics(currentResult());scheduleDraw();}
    else if(m.type==='result'){const incoming=(m.results||[]).map(r=>tagResult(cleanResult(r))).filter(Boolean),best=incoming.slice().sort(resultCompare)[0]||null;state.results=mergeResults(incoming);state.previewResult=null;state.activeResult=best?mergedRepresentative(best):(state.activeResult||state.results[0]||null);if(state.activeResult?.kind==='periodic')rememberExactResult(state.activeResult);state.colorCacheKey='';state.rasterCacheKey='';state.rasterCache=null;renderResults();updateMetrics();if(state.activeResult)centerViewOnResult(state.activeResult,true);scheduleDraw();setBusy(false);const prefix=m.reason==='known-non-tiler'?'H · ':m.reason==='approximate'?'≈ · ':'';setStatus(state.results.length?`${prefix}${state.results.length} ${state.results.length===1?t('result'):t('results')}`:t('noResult'),'idle');updateSolveLabel();}
    else if(m.type==='stopped'){setBusy(false);updateSolveLabel();setStatus(t('stopped'));}
    else if(m.type==='error'){setBusy(false);setStatus(m.message||'solver error','error');if(m.stack)console.error(m.stack);}
  }
  function cleanResult(r){if(!r||!validateResultPacking(r)){if(r)console.warn('discarded invalid/overlapping solver result',r?.id||r?.kind);return null;}r=reducePeriodicResult(r);if(!validateResultPacking(r)){console.warn('discarded result after lattice reduction',r?.id||r?.kind);return null;}if('_sig'in r)delete r._sig;return r;}
  function setBusy(v){state.busy=v;els.solveBtn.classList.toggle('hidden',v);els.stopBtn.classList.toggle('hidden',!v);els.searchProgress.classList.toggle('hidden',!v);els.liveHud?.classList.toggle('hidden',!v);if(!v){state.lastProgress=null;if(els.liveHud)els.liveHud.classList.add('hidden');}}
  function selectedPack(){if(!state.catalogId)return null;const m=/^P(\d+)-/.exec(state.catalogId);return m?window.__TL_PACKS__?.[Number(m[1])]||null:null;}
  function solve(){
    const cells=normalizeShape();if(!cells.length||!isConnected(cells)||hasHole(cells)){setStatus(!cells.length?t('empty'):!isConnected(cells)?t('disconnected'):t('holed'),'error');return;}if(!state.solverWorker){createSolverWorker();if(!state.solverWorker)return;}
    const sig=searchSignature(cells),shapeSig=canonicalShapeSig(cells),sameShape=state.resultShapeSignature===shapeSig||!state.resultShapeSignature,continuing=state.lastSearchSignature===sig&&state.results.length>0;
    if(!continuing){state.results=[];state.activeResult=null;}const resumeResults=sameShape&&continuing?state.results.filter(r=>(r.searchMode||state.mode)===state.mode&&(r.searchGroup||state.group)===state.group).slice(0,16):[];state.lastSearchSignature=sig;state.resultShapeSignature=shapeSig;state.inflightMode=state.mode;state.inflightGroup=state.group;state.searchRun++;
    state.previewResult=null;state.previewFitted=continuing;state.colorCacheKey='';renderResults();updateMetrics();scheduleDraw();setBusy(true);setStatus(t('searching'),'busy');resetProgress();
    const pack=selectedPack(),seedRec=pack?.seeds?.[state.catalogId]||null,cachedSeed=seedRec?.[state.group]||null,complexSeed=pack?.complex?.[state.catalogId]?.[state.group]||null,q=readSearchSettings();const timeMs=Math.min(60000,Math.max(350,Number(state.budget)||60000));
    const message={type:'solve',shape:cells,catalogId:state.catalogId,group:state.group,mode:state.mode,algorithm:state.algorithm,runNonce:state.searchRun,resumeResults,knownNonTiler:!!state.catalogMeta?.heesch,heesch:state.catalogMeta?.heesch||null,cachedSeed,complexSeed,knownComplexity:state.catalogMeta?.complexity?.[state.group]||null,options:{timeMs,maxResults:48,cellPieces:q.cellPieces,aggression:q.aggression/100,complexity:q.complexity/100,regularity:q.regularity/100,coverageBand:.03,targetCoverage:.9995,previewMs:500}};
    if(state.solverReady){state.pendingSolveMessage=null;state.solverWorker.postMessage(message);}else{state.pendingSolveMessage=message;if(!state.solverWorker)createSolverWorker();}
  }
  function stop(){if(!state.busy)return;state.pendingSolveMessage=null;if(state.previewResult){state.results=mergeResults([state.previewResult]);state.activeResult=state.results[0]||state.activeResult;}cleanupSolverWorker();createSolverWorker();state.previewResult=null;renderResults();updateMetrics(state.activeResult);setBusy(false);updateSolveLabel();setStatus(t('stopped'));scheduleDraw();}
  function resetProgress(){els.progressPhase.textContent='—';els.progressPct.textContent='0%';els.progressBar.style.width='0%';els.progressDims.textContent='—';els.progressNodes.textContent='0';els.progressBest.textContent='—';els.progressTime.textContent='0.0s';if(els.liveHud)els.liveHud.textContent='A · ρ — · 0.0s';}
  function updateLiveHud(m={}){if(!els.liveHud)return;const r=currentResult(),rho=Number.isFinite(r?.metrics?.coverage)?`${fmt(r.metrics.coverage*100,1)}%`:'—',engine=m.engine||displayEngine(r?.engine)||'A';els.liveHud.textContent=`${engine} · ρ ${rho} · ${((m.elapsed||0)/1000).toFixed(1)}s`;}
  function updateProgress(m){
    const pct=Number.isFinite(m.percent)?Math.max(0,Math.min(1,m.percent)):null;els.progressPhase.textContent=`${phaseLabel(m.phase)} · ${m.engine||'A'}`;els.progressPct.textContent=pct===null?'—':`${Math.round(pct*100)}%`;els.progressBar.style.width=pct===null?'0%':`${Math.round(pct*100)}%`;els.progressDims.textContent=m.w?`${m.w}×${m.h}`:'—';els.progressNodes.textContent=m.nodes?compact(m.nodes):m.current&&m.total?`${m.current}/${m.total}`:'0';els.progressBest.textContent=Number.isFinite(m.bestCoverage)?`ρ ${fmt(m.bestCoverage,3)}`:Number.isFinite(m.bestScore)?`S ${fmt(m.bestScore,1)}`:'—';els.progressTime.textContent=`${((m.elapsed||0)/1000).toFixed(1)}s`;updateLiveHud(m);
  }
  function compact(v){return v>=1e6?`${(v/1e6).toFixed(1)}m`:v>=1e3?`${(v/1e3).toFixed(1)}k`:String(v);}

  function latticeDisplay(L){if(!L)return'';const B=Array.isArray(L.basis)&&L.basis.length===2?L.basis:[[L.a,0],[L.b||0,L.c]];return`Λ (${B[0][0]},${B[0][1]}),(${B[1][0]},${B[1][1]})`; }
  function makeResultCard(r,tag=''){
    const i=state.results.indexOf(r),b=document.createElement('button');b.type='button';b.className='result-card'+(r===state.activeResult?' active':'');if(i>=0)b.dataset.i=i;
    const m=r.metrics||{},k=resultPeriod(r),rho=resultCoverage(r),plus=resultCombined(r),L=r.lattice||{a:r.w,b:0,c:r.h};
    const main=document.createElement('span');main.className='r-main';main.textContent=`${tag?tag+' · ':''}m ${k} · ρ ${fmt(rho*100,2)}%`+(r.searchMode==='companion'?` · ρ+ ${fmt(plus*100,2)}%`:``);
    const score=document.createElement('span');score.className='r-score';score.textContent=r.searchMode==='companion'?(Number.isFinite(m.companionArea)?`C ${m.companionArea}×${m.companionCopies||0}`:'C —'):`S ${fmt(m.score,1)}`;
    const sub=document.createElement('span');sub.className='r-sub';const basis=latticeDisplay(L);const rem=r.searchMode==='companion'&&Number.isFinite(m.unmatchedVoid)?` · rem ${m.unmatchedVoid}`:'';sub.textContent=`${displayEngine(r.engine)} · ${basis} · det ${L.det||L.a*L.c}${rem}`;b.append(main,score,sub);return b;
  }
  function periodCardText(r){const m=resultPeriod(r),rho=`ρ ${fmt(resultCoverage(r)*100,2)}%`,L=r.lattice?latticeDisplay(r.lattice):`${r.w}×${r.h}`;if((r.searchMode||state.mode)==='companion'){const plus=`ρ+ ${fmt(resultCombined(r)*100,2)}%`,q=Number.isFinite(r.metrics?.companionArea)?` · C ${fmt(r.metrics.companionArea,0)}×${r.metrics?.companionCopies||0}`:'',kind=r.metrics?.companionPattern==='single-component'?(state.lang==='zh'?' · 单分支':' · 1 branch'):r.metrics?.companionPattern==='congruent-components'?(state.lang==='zh'?' · ≅同形分支':' · ≅ branches'):'';return `m ${m} · ${rho} · ${plus}${q}${kind} · ${L}`;}return `m ${m} · ${rho} · ${L}`;}
  function appendResultCard(frag,r,i,extra=''){const b=document.createElement('button');b.type='button';b.className='result-card'+(r===state.activeResult?' active':'')+(extra?` ${extra}`:'');b.dataset.i=i;const main=document.createElement('span');main.className='r-main';main.textContent=periodCardText(r);const score=document.createElement('span');score.className='r-score';score.textContent=(r.searchMode||state.mode)==='companion'?`rem ${r.metrics?.unmatchedVoid??'—'} · S ${fmt(r.metrics?.score,1)}`:`S ${fmt(r.metrics?.score,1)}`;const sub=document.createElement('span');sub.className='r-sub';const voids=Number.isFinite(r.metrics?.voidComponents)?` · ∅${r.metrics.voidComponents}`:'',mu=Number.isFinite(r.metrics?.fundArea)?` · μ ${fmt(r.metrics.fundArea,0)}`:'';sub.textContent=`${displayEngine(r.engine)}${mu} · Hₐ ${fmt(r.metrics?.adjEntropy,2)} · Hₒ ${fmt(r.metrics?.oriEntropy,2)}${voids}`;b.append(main,score,sub);frag.appendChild(b);}
  function renderResults(){const limit=Math.max(8,Math.min(20,state.searchSettings.cellPieces||10)),eligible=state.results.filter(r=>(r.searchMode||state.mode)===state.mode&&(r.searchGroup||state.group)===state.group),by=new Map();for(const r of eligible){const m=resultPeriod(r);if(m<1||m>limit)continue;const old=by.get(m);if(!old||resultCompare(r,old)<0)by.set(m,r);}els.resultCount.textContent=`${by.size}/${limit}`;els.resultList.innerHTML='';const frag=document.createDocumentFragment(),head=document.createElement('div');head.className='result-group-title';head.textContent=state.lang==='zh'?(state.mode==='companion'?`伴生块周期谱 · 1…${limit}`:`周期谱 · 1…${limit}`):(state.mode==='companion'?`companion spectrum · 1…${limit}`:`primitive-period spectrum · 1…${limit}`);frag.appendChild(head);for(let m=1;m<=limit;m++){const r=by.get(m);if(r){appendResultCard(frag,r,state.results.indexOf(r),'period-best');}else{const b=document.createElement('div');b.className='result-card result-placeholder';b.innerHTML=`<span class="r-main">m ${m} · —</span><span class="r-score">ρ —</span><span class="r-sub">${state.busy?(state.lang==='zh'?'搜索中…':'searching…'):(state.lang==='zh'?'本预算内未找到':'not found in this budget')}</span>`;frag.appendChild(b);}}const bestSet=new Set(by.values()),alts=eligible.filter(r=>!bestSet.has(r)).sort(resultCompare).slice(0,24);if(alts.length){const h=document.createElement('div');h.className='result-group-title alternatives';h.textContent=state.lang==='zh'?'相近最优 · 其它格子 / 构型':'near-optimal alternatives · other lattices / packings';frag.appendChild(h);for(const r of alts)appendResultCard(frag,r,state.results.indexOf(r),'alternative');}els.resultList.appendChild(frag);}
  function currentResult(){return state.busy&&state.previewResult?state.previewResult:state.activeResult;}
  function updateMetrics(result=currentResult()){const m=result?.metrics||{};els.metricCoverage.textContent=Number.isFinite(m.coverage)?`${fmt(m.coverage*100,2)}%`:'—';els.metricFund.textContent=fmt(m.fundArea,0);els.metricAdj.textContent=fmt(m.adjEntropy,2);els.metricOri.textContent=fmt(m.oriEntropy,2);els.metricPeriod.textContent=result?String(resultPeriod(result)||'—'):'—';els.metricScore.textContent=fmt(m.score,1);els.resultName.textContent=result?`${displayLabel(result)} · ${displayEngine(result.engine)}`:'—';if(!result){els.resultDims.textContent='—';return;}const L=result.lattice;els.resultDims.textContent=L?`${latticeDisplay(L)} · det ${L.det||L.a*L.c}`:`${result.w}×${result.h}`;}


  const VIEW_BASE_SCALE=34;
  function viewMatrix(){const m=state.view.m;return Array.isArray(m)&&m.length===4?m:[1,0,0,1];}
  function screenDeltaToBase(dx,dy){const m=viewMatrix();return[m[0]*dx+m[2]*dy,m[1]*dx+m[3]*dy];}
  function composeViewTransform(t){const m=viewMatrix(),a=t[0]*m[0]+t[1]*m[2],b=t[0]*m[1]+t[1]*m[3],c=t[2]*m[0]+t[3]*m[2],d=t[2]*m[1]+t[3]*m[3];state.view.m=[Math.round(a),Math.round(b),Math.round(c),Math.round(d)];scheduleDraw();}
  function updateZoomLabel(){if(els.zoomLabel)els.zoomLabel.textContent=`${Math.round(state.view.scale/VIEW_BASE_SCALE*100)}%`;}
  function centerViewOnResult(result=currentResult(),preserveScale=true){if(!result)return;let x=0,y=0;if(result.kind==='periodic'||result.kind==='cell'||result.kind==='field'){const L=latticeOf(result);x=(L.a+(L.b||0))/2;y=L.c/2;}else{x=(Number(result.w)||0)/2;y=(Number(result.h)||0)/2;}state.view.x=x;state.view.y=y;if(!preserveScale)state.view.scale=VIEW_BASE_SCALE;updateZoomLabel();}
  function initCanvas(){
    const c=els.canvas;const resize=()=>{const rect=c.getBoundingClientRect(),cssPixels=Math.max(1,rect.width*rect.height),pixelCap=9_000_000,capByArea=Math.sqrt(pixelCap/cssPixels);state.dpr=Math.max(1,Math.min(window.devicePixelRatio||1,2.4,capByArea));const w=Math.max(1,Math.round(rect.width*state.dpr)),h=Math.max(1,Math.round(rect.height*state.dpr));if(c.width!==w||c.height!==h){c.width=w;c.height=h;scheduleDraw();}};if('ResizeObserver'in window){state.resizeObserver=new ResizeObserver(resize);state.resizeObserver.observe(c);}else window.addEventListener('resize',resize,{passive:true});resize();
    c.addEventListener('pointerdown',e=>{c.setPointerCapture?.(e.pointerId);state.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(state.pointers.size===1)state.drag={x:e.clientX,y:e.clientY,vx:state.view.x,vy:state.view.y};c.classList.add('dragging');});
    c.addEventListener('pointermove',e=>{if(!state.pointers.has(e.pointerId))return;state.pointers.set(e.pointerId,{x:e.clientX,y:e.clientY});if(state.pointers.size===1&&state.drag){const[dx,dy]=screenDeltaToBase(e.clientX-state.drag.x,e.clientY-state.drag.y);state.view.x=state.drag.vx-dx/state.view.scale;state.view.y=state.drag.vy-dy/state.view.scale;scheduleDraw();}else if(state.pointers.size===2){const p=[...state.pointers.values()],dist=Math.hypot(p[0].x-p[1].x,p[0].y-p[1].y),mx=(p[0].x+p[1].x)/2,my=(p[0].y+p[1].y)/2;if(state.lastPinch){zoomAt(dist/state.lastPinch.dist,mx,my);}state.lastPinch={dist,mx,my};}});
    const up=e=>{state.pointers.delete(e.pointerId);if(state.pointers.size<2)state.lastPinch=null;if(!state.pointers.size){state.drag=null;c.classList.remove('dragging');}};c.addEventListener('pointerup',up);c.addEventListener('pointercancel',up);
    c.addEventListener('wheel',e=>{e.preventDefault();zoomAt(Math.exp(-e.deltaY*.0012),e.clientX,e.clientY);},{passive:false});
  }
  function zoomAt(f,clientX,clientY){const rect=els.canvas.getBoundingClientRect(),sx=clientX-rect.left-rect.width/2,sy=clientY-rect.top-rect.height/2,[bx,by]=screenDeltaToBase(sx,sy),old=state.view.scale,nw=Math.max(3,Math.min(140,old*f)),wx=state.view.x+bx/old,wy=state.view.y+by/old;state.view.scale=nw;state.view.x=wx-bx/nw;state.view.y=wy-by/nw;updateZoomLabel();scheduleDraw();}
  function zoomStep(f){const rect=els.canvas.getBoundingClientRect();zoomAt(f,rect.left+rect.width/2,rect.top+rect.height/2);}
  function resetView(){state.view={x:0,y:0,scale:VIEW_BASE_SCALE,m:[1,0,0,1]};centerViewOnResult(currentResult(),true);updateZoomLabel();scheduleDraw();}
  function scheduleDraw(){if(state.drawQueued)return;state.drawQueued=true;requestAnimationFrame(()=>{state.drawQueued=false;draw();});}

  const paletteMemo=new Map(),edgeMemo=new WeakMap(),tileBoundsMemo=new WeakMap(),tileD4Memo=new WeakMap(),periodicTopologyMemo=new WeakMap(),finiteTopologyMemo=new WeakMap();
  function voidColor(){const q=state.colorSettings||COLOR_DEFAULTS,p=colorProfile();if(q.voidMode==='paper')return '#f4f1e9';if(q.voidMode==='ink')return '#272a27';let z;if(q.voidMode==='hue')z=ringSpec(mod((q.hue||0)/360+.5,1),p);else z=selectedStyleSpec();const s=Math.max(0,Math.min(14,z.s*(p.voidS??.07))),rawL=p.voidL??95,l=Math.max(93,rawL+(100-rawL)*.18);return `hsl(${z.h.toFixed(1)} ${s.toFixed(1)}% ${Math.min(98,l).toFixed(1)}%)`;}
  function drawHueWheel(){const c=els.hueWheel;if(!c)return;const ctx=c.getContext('2d'),W=c.width,H=c.height,q=state.colorSettings||COLOR_DEFAULTS,p=colorProfile(),cx=W*.28,cy=H*.5,R=Math.min(H*.39,W*.19),r0=R*.57,N=180;ctx.clearRect(0,0,W,H);for(let i=0;i<N;i++){const a0=i/N*Math.PI*2-Math.PI/2,a1=(i+1)/N*Math.PI*2-Math.PI/2,z=ringSpec(i/N,p);ctx.beginPath();ctx.arc(cx,cy,R,a0,a1+.004);ctx.arc(cx,cy,r0,a1+.004,a0,true);ctx.closePath();ctx.fillStyle=`hsl(${z.h.toFixed(1)} ${z.s.toFixed(1)}% ${z.l.toFixed(1)}%)`;ctx.fill();}ctx.beginPath();ctx.arc(cx,cy,r0-2,0,Math.PI*2);ctx.fillStyle=voidColor();ctx.fill();const a=(q.hue||0)*Math.PI/180-Math.PI/2,mx=cx+Math.cos(a)*(R+3),my=cy+Math.sin(a)*(R+3);ctx.beginPath();ctx.arc(mx,my,7,0,Math.PI*2);ctx.fillStyle='#faf9f5';ctx.fill();ctx.strokeStyle='#282b27';ctx.lineWidth=2;ctx.stroke();const x0=W*.54,y0=H*.19,sw=(W*.40)/7;for(let i=0;i<7;i++){ctx.fillStyle=paletteColor(i,7);ctx.fillRect(x0+i*sw,y0,Math.ceil(sw)+1,H*.28);}ctx.fillStyle=voidColor();ctx.fillRect(x0,y0+H*.40,W*.40,H*.22);ctx.strokeStyle='rgba(45,47,43,.30)';ctx.lineWidth=1;ctx.strokeRect(x0,y0+H*.40,W*.40,H*.22);ctx.fillStyle='rgba(45,47,43,.72)';ctx.font='10px ui-monospace, SFMono-Regular, Menlo, monospace';ctx.textAlign='left';ctx.fillText(p.kind||'',x0,y0+H*.73);}


  function latticeOf(r){return{a:Number(r?.lattice?.a)||r.w,b:Number(r?.lattice?.b)||0,c:Number(r?.lattice?.c)||r.h};}
  function tileDefinitions(r){const defs=[];for(const p of r.placements||[])defs.push({cells:Array.isArray(p.rawCells)?p.rawCells:p.cells,kind:'primary'});for(const comp of r.companion?.components||[])defs.push({cells:Array.isArray(comp.cells)?comp.cells:comp.quotientCells,kind:'companion'});return defs;}
  function periodicContactEdges(r,defs){
    const L=latticeOf(r),area=L.a*L.c,rec=Array(area).fill(null),edges=[],seen=new Set();
    defs.forEach((d,node)=>{for(const [x,y] of d.cells||[]){const z=reduceLattice(x,y,r),id=z.y*L.a+z.x;if(!rec[id])rec[id]={node,p:z.p,q:z.q};}});
    defs.forEach((d,node)=>{for(const [x,y] of d.cells||[])for(const [dx,dy] of [[1,0],[0,1]]){const z=reduceLattice(x+dx,y+dy,r),brec=rec[z.y*L.a+z.x];if(!brec)continue;const dp=z.p-brec.p,dq=z.q-brec.q,b=brec.node;if(node===b&&dp===0&&dq===0)continue;let a=node,bb=b,p=dp,q=dq;if(a>bb||(a===bb&&(q<0||(q===0&&p<0)))){[a,bb]=[bb,a];p=-p;q=-q;}const k=`${a}|${bb}|${p}|${q}`;if(!seen.has(k)){seen.add(k);edges.push({a,b:bb,p,q});}}});return edges;
  }
  function twoColorVoltage(edges,n){
    if(!edges.length)return{count:1,qx:1,qy:1,colors:new Int16Array(n),tileCount:n};
    for(let alpha=0;alpha<2;alpha++)for(let beta=0;beta<2;beta++){const adj=Array.from({length:n},()=>[]);for(const e of edges){const parity=1^(alpha&(Math.abs(e.p)%2))^(beta&(Math.abs(e.q)%2));adj[e.a].push([e.b,parity]);adj[e.b].push([e.a,parity]);}const base=new Int8Array(n);base.fill(-1);let ok=true;for(let s=0;s<n&&ok;s++)if(base[s]<0){base[s]=0;const q=[s];for(let h=0;h<q.length&&ok;h++){const u=q[h];for(const[v,p]of adj[u]){const want=base[u]^p;if(base[v]<0){base[v]=want;q.push(v);}else if(base[v]!==want){ok=false;break;}}}}if(ok){const qx=alpha?2:1,qy=beta?2:1,colors=new Int16Array(n*qx*qy);for(let sy=0;sy<qy;sy++)for(let sx=0;sx<qx;sx++)for(let i=0;i<n;i++)colors[(sy*qx+sx)*n+i]=base[i]^(alpha&sx)^(beta&sy);return{count:2,qx,qy,colors,tileCount:n};}}
    return null;
  }
  function expandedContactGraph(edges,n,qx,qy){const N=n*qx*qy,adj=Array.from({length:N},()=>new Set()),idx=(i,x,y)=>(mod(y,qy)*qx+mod(x,qx))*n+i;for(const e of edges)for(let sy=0;sy<qy;sy++)for(let sx=0;sx<qx;sx++){const u=idx(e.a,sx,sy),v=idx(e.b,sx+e.p,sy+e.q);if(u===v)return null;adj[u].add(v);adj[v].add(u);}return adj;}
  function greedyGraphColor(adj){const n=adj.length,colors=new Int16Array(n);colors.fill(-1);for(let done=0;done<n;done++){let best=-1,bSat=-1,bDeg=-1;for(let i=0;i<n;i++)if(colors[i]<0){const sat=new Set();for(const v of adj[i])if(colors[v]>=0)sat.add(colors[v]);if(sat.size>bSat||(sat.size===bSat&&adj[i].size>bDeg)){best=i;bSat=sat.size;bDeg=adj[i].size;}}const used=new Set();for(const v of adj[best])if(colors[v]>=0)used.add(colors[v]);let c=0;while(used.has(c))c++;colors[best]=c;}let count=0;for(const c of colors)count=Math.max(count,c+1);return{colors,count};}
  function kColorGraph(adj,k,stepLimit=160000){const n=adj.length,colors=new Int16Array(n),deg=adj.map(s=>s.size),use=new Int32Array(k);colors.fill(-1);let steps=0,colored=0;function rec(){if(++steps>stepLimit)return false;if(colored===n)return true;let u=-1,bSat=-1,bDeg=-1;for(let i=0;i<n;i++)if(colors[i]<0){const mask=new Uint8Array(k);let sat=0;for(const v of adj[i]){const c=colors[v];if(c>=0&&!mask[c]){mask[c]=1;sat++;}}if(sat>bSat||(sat===bSat&&deg[i]>bDeg)){u=i;bSat=sat;bDeg=deg[i];}}const forbidden=new Uint8Array(k);for(const v of adj[u])if(colors[v]>=0)forbidden[colors[v]]=1;const order=Array.from({length:k},(_,c)=>c).sort((a,b)=>use[a]-use[b]);for(const c of order){if(forbidden[c])continue;if(colored===0&&c!==0)continue;colors[u]=c;use[c]++;colored++;if(rec())return true;colored--;use[c]--;colors[u]=-1;}return false;}return rec()?colors:null;}
  function safeColorPeriod(edges){let best=null;for(let qx=1;qx<=14;qx++)for(let qy=1;qy<=14;qy++){let ok=true;for(const e of edges)if(e.a===e.b&&mod(e.p,qx)===0&&mod(e.q,qy)===0){ok=false;break;}if(ok&&(!best||qx*qy<best[0]*best[1]||(qx*qy===best[0]*best[1]&&Math.max(qx,qy)<Math.max(...best))))best=[qx,qy];}return best||[17,19];}
  function buildPeriodicColorMap(r){const cached=periodicTopologyMemo.get(r);if(cached)return cached;const defs=tileDefinitions(r),n=defs.length;if(!n){const z={count:1,qx:1,qy:1,colors:new Int16Array(0),tileCount:0,primaryCount:0,edges:[]};periodicTopologyMemo.set(r,z);return z;}const edges=periodicContactEdges(r,defs),two=twoColorVoltage(edges,n);if(two){two.primaryCount=r.placements.length;two.edges=edges;periodicTopologyMemo.set(r,two);return two;}const periods=[[1,1],[2,1],[1,2],[2,2],[3,1],[1,3],[3,2],[2,3],[3,3],[4,1],[1,4],[4,2],[2,4],[4,3],[3,4],[4,4],[5,1],[1,5],[5,2],[2,5],[5,3],[3,5],[5,4],[4,5],[5,5],[6,1],[1,6],[6,2],[2,6],[6,3],[3,6],[6,6]];for(const[qx,qy]of periods){const adj=expandedContactGraph(edges,n,qx,qy);if(!adj)continue;const cap=adj.length<180?700000:adj.length<500?360000:adj.length<900?180000:90000,colors=kColorGraph(adj,3,cap);if(colors){const z={count:3,qx,qy,colors,tileCount:n,primaryCount:r.placements.length,edges};periodicTopologyMemo.set(r,z);return z;}}
    let best=null;for(const[qx,qy]of periods){const adj=expandedContactGraph(edges,n,qx,qy);if(!adj)continue;const greedy=greedyGraphColor(adj);let colors=greedy.colors,count=greedy.count;for(let k=4;k<count;k++){const q=kColorGraph(adj,k,adj.length<500?300000:120000);if(q){colors=q;count=k;break;}}if(!best||count<best.count||(count===best.count&&qx*qy<best.qx*best.qy))best={count,qx,qy,colors,tileCount:n,primaryCount:r.placements.length,edges};if(best?.count===4&&qx*qy>=9)break;}if(!best){const[qx,qy]=safeColorPeriod(edges),adj=expandedContactGraph(edges,n,qx,qy),g=adj?greedyGraphColor(adj):null;if(g)best={count:g.count,qx,qy,colors:g.colors,tileCount:n,primaryCount:r.placements.length,edges};}if(!best){best={count:n,qx:1,qy:1,colors:Int16Array.from({length:n},(_,i)=>i),tileCount:n,primaryCount:r.placements.length,edges};}periodicTopologyMemo.set(r,best);return best;
  }
  function buildFiniteColorMap(r){const cached=finiteTopologyMemo.get(r);if(cached)return cached;const owner=new Map(),adj=Array.from({length:r.placements.length},()=>new Set());r.placements.forEach((p,i)=>p.cells.forEach(([x,y])=>owner.set(`${x},${y}`,i)));for(const[k,a]of owner){const[x,y]=k.split(',').map(Number);for(const[dx,dy]of[[1,0],[0,1]]){const b=owner.get(`${x+dx},${y+dy}`);if(b!==undefined&&a!==b){adj[a].add(b);adj[b].add(a);}}}let g=greedyGraphColor(adj);for(let k=1;k<g.count;k++){const q=k===1?(adj.every(x=>!x.size)?new Int16Array(adj.length):null):kColorGraph(adj,k,350000);if(q){g={colors:q,count:k};break;}}const edges=[];for(let a=0;a<adj.length;a++)for(const b of adj[a])if(a<b)edges.push({a,b,p:0,q:0});const z={...g,qx:1,qy:1,tileCount:r.placements.length,primaryCount:r.placements.length,edges};finiteTopologyMemo.set(r,z);return z;}
  function augmentColorCount(info,allowLift=false){
    const extra=Math.max(0,Math.min(12,(state.colorSettings?.extraColors||0)|0)),minCount=Math.max(1,info.count||1),desired=minCount+extra;
    if(extra<=0)return info;
    let qx=info.qx||1,qy=info.qy||1,colors=Int16Array.from(info.colors||[]),N=colors.length;
    if(allowLift&&info.tileCount>0&&N<desired){const oqx=qx,oqy=qy,base=colors;let flip=false;while(info.tileCount*qx*qy<desired&&qx*qy<64){if(flip)qy*=2;else qx*=2;flip=!flip;}colors=new Int16Array(info.tileCount*qx*qy);for(let y=0;y<qy;y++)for(let x=0;x<qx;x++)for(let i=0;i<info.tileCount;i++)colors[(y*qx+x)*info.tileCount+i]=base[((mod(y,oqy)*oqx+mod(x,oqx))*info.tileCount+i)];N=colors.length;}
    const target=Math.max(minCount,Math.min(desired,N||minCount));if(target<=minCount||N<2)return info;
    const hard=expandedContactGraph(info.edges||[],info.tileCount,qx,qy);if(!hard)return info;
    const use=new Int32Array(target),specs=Array.from({length:target},(_,i)=>paletteSpec(i,target));for(const c of colors)if(c>=0&&c<target)use[c]++;
    const order=Array.from({length:N},(_,i)=>i).sort((a,b)=>hard[b].size-hard[a].size||a-b);
    // Introduce every requested extra color first. Splitting an existing color class preserves all hard edge constraints.
    for(let c=minCount;c<target;c++){let pick=-1,best=-Infinity;for(const u of order){if(use[colors[u]]<=1)continue;let near=0;for(const v of hard[u])near+=paletteDistance(specs[c],specs[colors[v]]);const score=near+hard[u].size*.12;if(score>best){best=score;pick=u;}}if(pick<0)pick=order[(c-minCount)%order.length];use[colors[pick]]--;colors[pick]=c;use[c]++;}
    const soft=Array.from({length:N},()=>new Set());for(let u=0;u<N;u++)for(const v of hard[u])for(const w of hard[v])if(w!==u&&!hard[u].has(w))soft[u].add(w);
    // Local refinement: shared-edge conflicts are forbidden; graph-distance-2 pairs approximate corner contacts and very close tiles.
    for(let pass=0;pass<2;pass++)for(const u of order){const old=colors[u];let bestC=old,bestScore=-Infinity;for(let c=0;c<target;c++){let forbidden=false,score=0;for(const v of hard[u]){if(colors[v]===c){forbidden=true;break;}score+=1.15*paletteDistance(specs[c],specs[colors[v]]);}if(forbidden)continue;for(const v of soft[u])score+=.34*paletteDistance(specs[c],specs[colors[v]]);score-=.13*use[c];if(c!==old&&use[old]<=1)score-=1e6;if(score>bestScore){bestScore=score;bestC=c;}}if(bestC!==old){use[old]--;use[bestC]++;colors[u]=bestC;}}
    return{...info,qx,qy,count:target,colors};
  }
  function bestColorRemap(info){const k=info.count||1,identity=Array.from({length:k},(_,i)=>i);if(k<=1||k>7||!info.edges?.length)return identity;const q=state.colorSettings||COLOR_DEFAULTS,p=colorProfile(),target=Math.max(.08,Math.min(2.15,(p.neighborTarget??.95)+((q.neighborBias??0)/100)*1.05)),specs=identity.map(i=>paletteSpec(i,k)),W=Array.from({length:k},()=>new Float64Array(k)),idx=(node,x,y)=>(mod(y,info.qy)*info.qx+mod(x,info.qx))*info.tileCount+node;for(const e of info.edges)for(let sy=0;sy<info.qy;sy++)for(let sx=0;sx<info.qx;sx++){const ca=info.colors[idx(e.a,sx,sy)],cb=info.colors[idx(e.b,sx+e.p,sy+e.q)];if(ca===cb||ca<0||cb<0)continue;W[ca][cb]++;W[cb][ca]++;}let best=identity,bestScore=-Infinity,perm=new Int16Array(k),used=new Uint8Array(k);function rec(d){if(d===k){let score=0;for(let i=0;i<k;i++)for(let j=i+1;j<k;j++)if(W[i][j]){const dist=paletteDistance(specs[perm[i]],specs[perm[j]]),err=dist-target;score-=W[i][j]*err*err;}if(score>bestScore){bestScore=score;best=Array.from(perm);}return;}for(let c=0;c<k;c++)if(!used[c]){used[c]=1;perm[d]=c;rec(d+1);used[c]=0;}}rec(0);return best;}
  function ensureColorMap(r){const L=latticeOf(r),q=state.colorSettings,k=`${r.id}|${r.placements.length}|${r.companion?.components?.length||0}|${L.a},${L.b},${L.c}|${q.style}|${q.hue}|${q.saturation}|${q.brightness}|${q.extraColors}|${q.dispersion}|${q.neighborBias}|${q.orbitSpan}|${q.luma}`;if(k===state.colorCacheKey&&state.colorMap)return state.colorMap;state.colorCacheKey=k;const periodic=(r.kind==='periodic'||r.kind==='cell'||r.kind==='field'),base=periodic?buildPeriodicColorMap(r):buildFiniteColorMap(r),topo=augmentColorCount(base,periodic);state.colorMap={...topo,remap:(topo.count===base.count?bestColorRemap(topo):Array.from({length:topo.count},(_,i)=>i))};return state.colorMap;}
  function colorIndex(info,node,p=0,q=0){if(!info?.tileCount)return node;const i=(mod(q,info.qy)*info.qx+mod(p,info.qx))*info.tileCount+node,c=info.colors[i]??node;return info.remap?.[c]??c;}


  function tileEdges(cells){let e=edgeMemo.get(cells);if(e)return e;const set=new Set(cells.map(([x,y])=>`${x},${y}`));e=[];for(const[x,y]of cells){if(!set.has(`${x},${y-1}`))e.push([x,y,x+1,y]);if(!set.has(`${x+1},${y}`))e.push([x+1,y,x+1,y+1]);if(!set.has(`${x},${y+1}`))e.push([x+1,y+1,x,y+1]);if(!set.has(`${x-1},${y}`))e.push([x,y+1,x,y]);}edgeMemo.set(cells,e);return e;}
  function cellBounds(cells){let b=tileBoundsMemo.get(cells);if(b)return b;let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const[x,y]of cells){minX=Math.min(minX,x);minY=Math.min(minY,y);maxX=Math.max(maxX,x+1);maxY=Math.max(maxY,y+1);}b={minX,minY,maxX,maxY};tileBoundsMemo.set(cells,b);return b;}

  function toneCss(spec,dl=0,ds=0,a=1){const z=spec||selectedStyleSpec(),h=z.h??0,s=Math.max(0,Math.min(100,(z.s??0)+ds)),l=Math.max(0,Math.min(100,(z.l??50)+dl));return `hsla(${h.toFixed(1)},${s.toFixed(1)}%,${l.toFixed(1)}%,${Math.max(0,Math.min(1,a)).toFixed(3)})`;}
  function resolveEdgeStyle(s,fillSpec=null){
    const q=state.colorSettings||COLOR_DEFAULTS,v=(q.edge??58)/100,a=.22+v*.66,dark=(fillSpec?.l??selectedStyleSpec().l)<58,mode=q.edgeColorMode||'auto';let color;
    if(mode==='black')color=`rgba(23,25,23,${a.toFixed(3)})`;else if(mode==='white')color=`rgba(255,255,252,${a.toFixed(3)})`;else color=dark?`rgba(255,255,252,${a.toFixed(3)})`:`rgba(28,30,28,${a.toFixed(3)})`;
    return{show:(q.edgeLineStyle||'single')!=='none'&&v>.01&&s>3.2,width:Math.max(.55,Math.min(4.6,s*.022*(.72+v*2.15))),color,mode:q.edgeLineStyle||'single',gap:fillSpec?.css||toneCss(fillSpec,0,0,1)};
  }
  function boundaryPath(ctx,cells,tx,ty,v){ctx.beginPath();for(const[x1,y1,x2,y2]of tileEdges(cells)){ctx.moveTo(v.toX(x1+tx),v.toY(y1+ty));ctx.lineTo(v.toX(x2+tx),v.toY(y2+ty));}}
  function strokeTileBoundary(ctx,cells,tx,ty,v,fillSpec=null){
    const edge=resolveEdgeStyle(v.s,fillSpec);if(!edge.show)return;ctx.save();ctx.lineJoin='round';ctx.lineCap='round';ctx.setLineDash([]);
    const stroke=(width,color,dash=null,cap='round')=>{boundaryPath(ctx,cells,tx,ty,v);ctx.lineWidth=width;ctx.strokeStyle=color;ctx.lineCap=cap;ctx.setLineDash(dash||[]);ctx.stroke();};
    if(edge.mode==='double'){stroke(edge.width*2.45,edge.color);stroke(edge.width*1.45,edge.gap);stroke(edge.width*.56,edge.color);}
    else if(edge.mode==='frame'){stroke(edge.width*2.8,edge.color);stroke(edge.width*1.72,edge.gap);stroke(edge.width*.82,edge.color);}
    else if(edge.mode==='dashed')stroke(edge.width,edge.color,[edge.width*4.5,edge.width*2.8],'butt');
    else if(edge.mode==='dotted')stroke(edge.width,edge.color,[.01,edge.width*2.65],'round');
    else stroke(edge.width,edge.color);
    ctx.restore();
  }
  function strokeCellJoints(ctx,cells,tx,ty,v,fillSpec=null){
    const q=state.colorSettings||COLOR_DEFAULTS,str=(q.cellEdge??0)/100;if(str<.015||v.s<5)return;const set=new Set(cells.map(([x,y])=>`${x},${y}`)),dark=(fillSpec?.l??selectedStyleSpec().l)<58,mode=q.edgeColorMode||'auto',a=.08+str*.44;let color=mode==='black'?`rgba(24,26,24,${a})`:mode==='white'?`rgba(255,255,252,${a})`:dark?`rgba(255,255,252,${a})`:`rgba(24,26,24,${a})`;ctx.save();ctx.beginPath();for(const[x,y]of cells){if(set.has(`${x+1},${y}`)){ctx.moveTo(v.toX(x+1+tx),v.toY(y+ty));ctx.lineTo(v.toX(x+1+tx),v.toY(y+1+ty));}if(set.has(`${x},${y+1}`)){ctx.moveTo(v.toX(x+tx),v.toY(y+1+ty));ctx.lineTo(v.toX(x+1+tx),v.toY(y+1+ty));}}ctx.strokeStyle=color;ctx.lineWidth=Math.max(.45,Math.min(2.4,v.s*(.012+.035*str)));ctx.setLineDash([]);ctx.stroke();ctx.restore();
  }
  function clipTile(ctx,cells,tx,ty,v){ctx.beginPath();for(const[x,y]of cells)ctx.rect(v.toX(x+tx)-.12,v.toY(y+ty)-.12,v.s+.28,v.s+.28);ctx.clip();}
  function hashCell(x,y,k=0){let h=(Math.imul((x|0)+101,73856093)^Math.imul((y|0)+211,19349663)^Math.imul(k+17,83492791))>>>0;h^=h>>>13;h=Math.imul(h,1274126177)>>>0;return(h>>>0)/4294967295;}
  function normalizedCellsSig(cells){if(!cells?.length)return'';let minX=Infinity,minY=Infinity;for(const[x,y]of cells){minX=Math.min(minX,x);minY=Math.min(minY,y);}return cells.map(([x,y])=>[x-minX,y-minY]).sort((a,b)=>a[1]-b[1]||a[0]-b[0]).map(p=>p.join(',')).join(';');}
  function tileD4Index(cells){const cached=tileD4Memo.get(cells);if(cached!==undefined)return cached;const target=normalizedCellsSig(cells),base=normalizeShape();let out=0;for(let i=0;i<EQUIV_TRANSFORMS.length;i++){const f=EQUIV_TRANSFORMS[i],sig=normalizedCellsSig(base.map(([x,y])=>f(x,y)));if(sig===target){out=i;break;}}tileD4Memo.set(cells,out);return out;}
  function motifTransformIndex(cells,tx,ty){const mode=(state.colorSettings||COLOR_DEFAULTS).patternTransform||'d4';if(mode==='fixed')return 0;if(mode==='alternate')return mod(Math.round(tx)+Math.round(ty),4);if(mode==='mirror')return (mod(Math.round(tx)+Math.round(ty),2)?4:0);return tileD4Index(cells);}
  function withMotifTransform(ctx,cx,cy,index,fn){ctx.save();ctx.translate(cx,cy);const mats=[[1,0,0,1],[0,1,-1,0],[-1,0,0,-1],[0,-1,1,0],[-1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,-1,-1,0]],m=mats[mod(index,8)];ctx.transform(m[0],m[1],m[2],m[3],0,0);fn();ctx.restore();}
  function drawTileMaterial(ctx,cells,tx,ty,v,fillSpec=null){
    const q=state.colorSettings||COLOR_DEFAULTS,kind=q.material||'plain',strength=(q.materialStrength??34)/100;if(kind==='plain'||strength<.015||v.s<7)return;
    const opacity=(q.patternOpacity??54)/100,contrast=(q.patternContrast??48)/100,widthCtl=(q.patternWidth??44)/100,scaleCtl=(q.patternScale??54)/100,alpha=(.08+strength*.26)*(.35+opacity*.95),tone=12+contrast*22,
      light=toneCss(fillSpec,tone,-4,alpha),light2=toneCss(fillSpec,tone+12,-8,alpha*.78),dark=toneCss(fillSpec,-tone,-1,alpha*.90),dark2=toneCss(fillSpec,-tone-12,-2,alpha*.62),accent=toneCss(fillSpec,-6,6,alpha*.86);
    ctx.save();clipTile(ctx,cells,tx,ty,v);ctx.lineJoin='round';ctx.lineCap='round';ctx.setLineDash([]);
    const px=x=>v.toX(x+tx),py=y=>v.toY(y+ty),S=v.s,b=cellBounds(cells),X0=px(b.minX),Y0=py(b.minY),W=(b.maxX-b.minX)*S,H=(b.maxY-b.minY)*S,X1=X0+W,Y1=Y0+H,CX=X0+W*.5,CY=Y0+H*.5,
      side=Math.max(S*.9,Math.max(W,H)),motifSide=side*(.72+.50*scaleCtl),lw=Math.max(.45,S*(.010+.032*widthCtl)),idx=motifTransformIndex(cells,tx,ty),
      rectStroke=(x,y,pad,color,w)=>{ctx.strokeStyle=color;ctx.lineWidth=w;ctx.strokeRect(px(x)+pad,py(y)+pad,S-2*pad,S-2*pad);};
    const square=(fn)=>withMotifTransform(ctx,CX,CY,idx,()=>{ctx.save();ctx.scale?.(motifSide,motifSide);fn();ctx.restore();});
    const line=(a,b,c,d,color=dark,w=lw)=>{ctx.strokeStyle=color;ctx.lineWidth=w;ctx.beginPath();ctx.moveTo(a,b);ctx.lineTo(c,d);ctx.stroke();};
    const unitPath=(points,color=dark,w=lw,close=false)=>{ctx.strokeStyle=color;ctx.lineWidth=w;ctx.beginPath();points.forEach(([x,y],i)=>i?ctx.lineTo(CX+x*motifSide,CY+y*motifSide):ctx.moveTo(CX+x*motifSide,CY+y*motifSide));if(close)ctx.closePath();ctx.stroke();};

    // Whole-tile ornaments use a centered 1:1 motif frame regardless of polyomino aspect ratio.
    if(['textile','garden','vintage','gothic','deco','lacquer','paper','trellis','arabesque','seigaiha','asanoha','quatrefoil','guilloche','rosette','sunburst','leaf','clover','maze'].includes(kind)){
      withMotifTransform(ctx,CX,CY,idx,()=>{
        const R=motifSide*.5, x=-R,y=-R;
        if(kind==='textile'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.72;for(let k=-4;k<=4;k++){const yy=k*R*.20;ctx.beginPath();for(let j=-4;j<=4;j++){const xx=j*R*.25,dy=((j+k)&1)?R*.055:0;j===-4?ctx.moveTo(xx,yy+dy):ctx.lineTo(xx,yy+dy);}ctx.stroke();}
          ctx.strokeStyle=light2;ctx.lineWidth=lw*1.45;ctx.beginPath();ctx.moveTo(0,-R*.62);ctx.lineTo(R*.62,0);ctx.lineTo(0,R*.62);ctx.lineTo(-R*.62,0);ctx.closePath();ctx.stroke();ctx.beginPath();ctx.moveTo(0,-R*.32);ctx.lineTo(R*.32,0);ctx.lineTo(0,R*.32);ctx.lineTo(-R*.32,0);ctx.closePath();ctx.stroke();
        }else if(kind==='garden'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;ctx.beginPath();ctx.arc(0,0,R*.72,0,Math.PI*2);ctx.stroke();line(-R*.82,0,R*.82,0,dark,lw);line(0,-R*.82,0,R*.82,dark,lw);ctx.fillStyle=accent;for(let k=0;k<4;k++){const a=k*Math.PI/2;ctx.save();ctx.translate(Math.cos(a)*R*.30,Math.sin(a)*R*.30);ctx.rotate(a);ctx.beginPath();ctx.ellipse(0,0,R*.13,R*.24,0,0,Math.PI*2);ctx.fill();ctx.restore();}ctx.fillStyle=light2;ctx.beginPath();ctx.arc(0,0,R*.075,0,Math.PI*2);ctx.fill();
        }else if(kind==='vintage'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;for(const rr of [.30,.55]){ctx.beginPath();ctx.arc(0,0,R*rr,0,Math.PI*2);ctx.stroke();}ctx.strokeStyle=light;ctx.lineWidth=lw*1.3;for(let k=0;k<4;k++){const a=k*Math.PI/2;ctx.beginPath();ctx.arc(Math.cos(a)*R,Math.sin(a)*R,R*.36,a+Math.PI*.55,a+Math.PI*1.45);ctx.stroke();}
        }else if(kind==='gothic'){
          ctx.strokeStyle=dark2;ctx.lineWidth=lw*1.3;ctx.beginPath();ctx.moveTo(-R*.62,R*.72);ctx.quadraticCurveTo(-R*.62,-R*.28,0,-R*.78);ctx.quadraticCurveTo(R*.62,-R*.28,R*.62,R*.72);ctx.stroke();line(0,-R*.78,0,R*.72,dark2,lw);ctx.strokeStyle=light;for(let k=0;k<4;k++){const a=k*Math.PI/2;ctx.beginPath();ctx.arc(Math.cos(a)*R*.12,Math.sin(a)*R*.12,R*.13,0,Math.PI*2);ctx.stroke();}
        }else if(kind==='deco'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;for(let k=0;k<=8;k++){const xx=-R*.72+k*R*1.44/8;line(0,R*.72,xx,-R*.72,dark,lw*.8);}ctx.strokeStyle=light2;ctx.lineWidth=lw*1.25;for(const rr of [.48,.72]){ctx.beginPath();ctx.ellipse(0,R*.72,R*rr,R*rr*.58,0,Math.PI*1.08,Math.PI*1.92);ctx.stroke();}
        }else if(kind==='lacquer'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*1.1;ctx.strokeRect(-R*.78,-R*.78,R*1.56,R*1.56);ctx.strokeStyle=light2;ctx.lineWidth=lw*1.5;ctx.beginPath();ctx.moveTo(-R*.68,-R*.42);ctx.bezierCurveTo(-R*.24,-R*.72,R*.18,-R*.68,R*.60,-R*.38);ctx.stroke();ctx.fillStyle=light;ctx.beginPath();ctx.ellipse(R*.40,-R*.20,R*.18,R*.07,-.30,0,Math.PI*2);ctx.fill();
        }else if(kind==='paper'){
          ctx.fillStyle=light;ctx.beginPath();ctx.moveTo(-R*.72,-R*.52);ctx.lineTo(R*.40,-R*.72);ctx.lineTo(R*.74,R*.35);ctx.lineTo(-R*.36,R*.72);ctx.closePath();ctx.fill();ctx.strokeStyle=dark;ctx.lineWidth=lw;ctx.beginPath();ctx.moveTo(-R*.72,0);ctx.lineTo(0,-R*.72);ctx.lineTo(R*.72,0);ctx.lineTo(0,R*.72);ctx.closePath();ctx.stroke();line(-R*.55,R*.55,R*.55,-R*.55,light2,lw);
        }else if(kind==='trellis'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;for(let k=-3;k<=3;k++){line(-R,k*R*.32,R,k*R*.32-R*2,dark,lw);line(-R,k*R*.32,R,k*R*.32+R*2,light2,lw*.8);}
        }else if(kind==='arabesque'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;for(let k=0;k<4;k++){const a=k*Math.PI/2;ctx.save();ctx.rotate(a);ctx.beginPath();ctx.moveTo(0,0);ctx.bezierCurveTo(R*.12,-R*.58,R*.55,-R*.55,R*.62,0);ctx.bezierCurveTo(R*.42,R*.20,R*.22,R*.24,0,0);ctx.stroke();ctx.restore();}
        }else if(kind==='seigaiha'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.75;for(let row=-2;row<=2;row++)for(let col=-2;col<=2;col++){const ox=col*R*.55+(row&1)*R*.275,oy=row*R*.42;for(const rr of [.18,.28,.38]){ctx.beginPath();ctx.arc(ox,oy,R*rr,Math.PI,0);ctx.stroke();}}
        }else if(kind==='asanoha'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;for(let k=0;k<6;k++){const a=k*Math.PI/3;line(0,0,Math.cos(a)*R*.72,Math.sin(a)*R*.72,dark,lw*.8);line(Math.cos(a)*R*.72,Math.sin(a)*R*.72,Math.cos(a+Math.PI/3)*R*.72,Math.sin(a+Math.PI/3)*R*.72,light2,lw*.65);}
        }else if(kind==='quatrefoil'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;for(let k=0;k<4;k++){const a=k*Math.PI/2;ctx.beginPath();ctx.arc(Math.cos(a)*R*.25,Math.sin(a)*R*.25,R*.30,0,Math.PI*2);ctx.stroke();}ctx.strokeStyle=light2;ctx.beginPath();ctx.arc(0,0,R*.16,0,Math.PI*2);ctx.stroke();
        }else if(kind==='guilloche'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;for(let k=-2;k<=2;k++){ctx.beginPath();for(let j=0;j<=28;j++){const xx=-R+2*R*j/28,yy=k*R*.22+Math.sin(j/28*Math.PI*4)*R*.16;j?ctx.lineTo(xx,yy):ctx.moveTo(xx,yy);}ctx.stroke();}
        }else if(kind==='rosette'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;for(let k=0;k<10;k++){const a=k*Math.PI/5;ctx.save();ctx.rotate(a);ctx.beginPath();ctx.ellipse(0,-R*.32,R*.12,R*.34,0,0,Math.PI*2);ctx.stroke();ctx.restore();}ctx.fillStyle=light2;ctx.beginPath();ctx.arc(0,0,R*.10,0,Math.PI*2);ctx.fill();
        }else if(kind==='sunburst'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;for(let k=0;k<16;k++){const a=k*Math.PI/8,lineR=k%2?R*.60:R*.82;line(0,0,Math.cos(a)*lineR,Math.sin(a)*lineR,dark,lw*.8);}ctx.strokeStyle=light2;ctx.lineWidth=lw;ctx.beginPath();ctx.arc(0,0,R*.24,0,Math.PI*2);ctx.stroke();
        }else if(kind==='leaf'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;ctx.beginPath();ctx.moveTo(-R*.58,R*.56);ctx.bezierCurveTo(-R*.12,R*.18,R*.18,-R*.22,R*.54,-R*.62);ctx.stroke();for(let k=-2;k<=2;k++){const t=(k+3)/6,x=-R*.58+(R*1.12)*t,y=R*.56-(R*1.18)*t;ctx.beginPath();ctx.ellipse(x,y,R*.12,R*.25,-.72,0,Math.PI*2);ctx.stroke();}
        }else if(kind==='clover'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;for(let k=0;k<4;k++){const a=k*Math.PI/2;ctx.beginPath();ctx.ellipse(Math.cos(a)*R*.25,Math.sin(a)*R*.25,R*.22,R*.30,a,0,Math.PI*2);ctx.stroke();}line(0,R*.10,0,R*.70,light2,lw*.8);
        }else if(kind==='maze'){
          ctx.strokeStyle=dark;ctx.lineWidth=lw;const rr=[.70,.48,.26];for(const f of rr){ctx.strokeRect(-R*f,-R*f,R*f*2,R*f*2);}line(-R*.70,-R*.08,-R*.26,-R*.08,light2,lw);line(R*.26,R*.08,R*.70,R*.08,light2,lw);
        }
      });
    }else{
      // Square-module patterns: bounded, constant per-cell complexity for smooth browser rendering.
      for(const[x,y]of cells){const X=px(x),Y=py(y),cx=X+S*.5,cy=Y+S*.5;
        if(kind==='ceramic'){rectStroke(x,y,S*.075,light,lw*1.15);ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;ctx.beginPath();ctx.moveTo(X+S*.12,Y+S*.86);ctx.quadraticCurveTo(cx,Y+S*.56,X+S*.88,Y+S*.86);ctx.stroke();if(S>13){ctx.fillStyle=light2;for(const[a,b0]of[[0,-.17],[.17,0],[0,.17],[-.17,0]]){ctx.beginPath();ctx.ellipse(cx+a*S,cy+b0*S,S*.09,S*.145,Math.atan2(b0,a)+Math.PI/2,0,Math.PI*2);ctx.fill();}}
        }else if(kind==='stained'){ctx.fillStyle=light2;ctx.beginPath();ctx.moveTo(cx,Y+S*.08);ctx.lineTo(X+S*.92,cy);ctx.lineTo(cx,Y+S*.92);ctx.lineTo(X+S*.08,cy);ctx.closePath();ctx.fill();ctx.strokeStyle=dark2;ctx.lineWidth=lw*1.55;ctx.beginPath();ctx.moveTo(X+S*.08,cy);ctx.lineTo(X+S*.92,cy);ctx.moveTo(cx,Y+S*.08);ctx.lineTo(cx,Y+S*.92);ctx.moveTo(X+S*.08,Y+S*.08);ctx.lineTo(X+S*.92,Y+S*.92);ctx.moveTo(X+S*.92,Y+S*.08);ctx.lineTo(X+S*.08,Y+S*.92);ctx.stroke();
        }else if(kind==='mosaic'){const gap=Math.max(.55,lw*.85),ss=S*.5;for(let gy=0;gy<2;gy++)for(let gx=0;gx<2;gx++){ctx.fillStyle=((gx+gy)&1)?light2:dark;ctx.globalAlpha=.25+opacity*.28;ctx.fillRect(X+gx*ss+gap,Y+gy*ss+gap,ss-gap*2,ss-gap*2);}ctx.globalAlpha=1;
        }else if(kind==='terrazzo'||kind==='confetti'){const count=kind==='terrazzo'?6:4;for(let k=0;k<count;k++){const rx=.12+hashCell(x+tx,y+ty,k)*.76,ry=.12+hashCell(y+ty,x+tx,k+9)*.76,rr=S*(kind==='terrazzo'?.025+.035*hashCell(x+tx*3,y+ty*5,k+21):.018+.024*hashCell(x+tx,y+ty,k+21));ctx.fillStyle=(k%3===0)?light2:(k%3===1)?dark:accent;ctx.beginPath();ctx.arc(X+rx*S,Y+ry*S,rr,0,Math.PI*2);ctx.fill();}
        }else if(kind==='pinstripe'){ctx.strokeStyle=dark;ctx.lineWidth=lw*.6;const step=S*(.16+.20*scaleCtl);for(let xx=X-S;xx<X+S*2;xx+=step)line(xx,Y,xx+S,Y+S,dark,lw*.6);
        }else if(kind==='herringbone'){ctx.strokeStyle=dark;ctx.lineWidth=lw*.75;for(let k=-1;k<3;k++){const yy=Y+k*S*.42;line(X,yy,X+S*.5,yy+S*.32,dark,lw*.75);line(X+S*.5,yy+S*.32,X+S,yy,dark,lw*.75);}
        }else if(kind==='chevron'){ctx.strokeStyle=dark;ctx.lineWidth=lw*.8;for(let k=-1;k<4;k++){const yy=Y+k*S*.30;ctx.beginPath();ctx.moveTo(X,yy);ctx.lineTo(cx,yy+S*.18);ctx.lineTo(X+S,yy);ctx.stroke();}
        }else if(kind==='gingham'||kind==='checker'){const n=kind==='checker'?4:3,ss=S/n;for(let gy=0;gy<n;gy++)for(let gx=0;gx<n;gx++)if((gx+gy)%2===0){ctx.fillStyle=(gx%3===0)?light2:dark;ctx.globalAlpha=.16+opacity*.24;ctx.fillRect(X+gx*ss,Y+gy*ss,ss,ss);}ctx.globalAlpha=1;
        }else if(kind==='basketweave'){ctx.strokeStyle=dark;ctx.lineWidth=lw*1.25;for(let k=0;k<3;k++){const off=(k+.5)*S/3;line(X+off,Y,X+off,Y+S,dark,lw*1.1);line(X,Y+off,X+S,Y+off,light2,lw*.9);}
        }else if(kind==='linen'){ctx.strokeStyle=dark;ctx.lineWidth=lw*.45;for(const f of [.18,.42,.67,.86]){line(X+S*f,Y,X+S*f,Y+S,dark,lw*.45);line(X,Y+S*f,X+S,Y+S*f,light2,lw*.45);}
        }else if(kind==='marble'){ctx.strokeStyle=light2;ctx.lineWidth=lw*.8;ctx.beginPath();ctx.moveTo(X-S*.05,Y+S*.82);ctx.bezierCurveTo(X+S*.22,Y+S*.62,X+S*.34,Y+S*.25,X+S*.60,Y+S*.38);ctx.bezierCurveTo(X+S*.76,Y+S*.47,X+S*.80,Y+S*.18,X+S*1.05,Y+S*.08);ctx.stroke();ctx.strokeStyle=dark;ctx.lineWidth=lw*.35;ctx.stroke();
        }
      }
    }
    ctx.globalAlpha=1;ctx.restore();
  }
  function drawTranslatedTile(ctx,cells,tx,ty,color,v,fillSpec=null){const{s,toX,toY}=v,pad=0,size=s+.18;ctx.fillStyle=color;for(let k=0;k<cells.length;k++){const c=cells[k],px=toX(c[0]+tx),py=toY(c[1]+ty);ctx.fillRect(px-.08,py-.08,size,size);}drawTileMaterial(ctx,cells,tx,ty,v,fillSpec);strokeCellJoints(ctx,cells,tx,ty,v,fillSpec);strokeTileBoundary(ctx,cells,tx,ty,v,fillSpec);}
  function drawTileCells(ctx,cells,color,v,fillSpec=null){drawTranslatedTile(ctx,cells,0,0,color,v,fillSpec);}

  function draw(){
    const c=els.canvas,ctx=c.getContext('2d'),rect=c.getBoundingClientRect(),dpr=state.dpr;ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,rect.width,rect.height);ctx.fillStyle=voidColor();ctx.fillRect(0,0,rect.width,rect.height);const r=currentResult();els.emptyCanvas.classList.toggle('hidden',!!r);if(!r)return;
    const s=state.view.scale,toX=x=>rect.width/2+(x-state.view.x)*s,toY=y=>rect.height/2+(y-state.view.y)*s,m=viewMatrix(),corners=[[-rect.width/2,-rect.height/2],[rect.width/2,-rect.height/2],[rect.width/2,rect.height/2],[-rect.width/2,rect.height/2]],world=corners.map(([sx,sy])=>{const[bx,by]=screenDeltaToBase(sx,sy);return[state.view.x+bx/s,state.view.y+by/s];}),overscan=Math.max(3,10/s),worldLeft=Math.min(...world.map(p=>p[0]))-overscan,worldRight=Math.max(...world.map(p=>p[0]))+overscan,worldTop=Math.min(...world.map(p=>p[1]))-overscan,worldBottom=Math.max(...world.map(p=>p[1]))+overscan;
    ctx.save();ctx.translate(rect.width/2,rect.height/2);ctx.transform(m[0],m[2],m[1],m[3],0,0);ctx.translate(-rect.width/2,-rect.height/2);
    if(r.kind==='periodic'||r.kind==='cell'||r.kind==='field')drawPeriodic(ctx,r,{s,toX,toY,worldLeft,worldRight,worldTop,worldBottom});else drawFinite(ctx,r,{s,toX,toY,worldLeft,worldRight,worldTop,worldBottom});ctx.restore();
  }
  function drawPeriodic(ctx,r,v){
    const L=latticeOf(r),defs=tileDefinitions(r),colors=ensureColorMap(r);if(!defs.length)return;let minX=Infinity,minY=Infinity,maxX=-Infinity,maxY=-Infinity;for(const d of defs){const b=cellBounds(d.cells);minX=Math.min(minX,b.minX);minY=Math.min(minY,b.minY);maxX=Math.max(maxX,b.maxX);maxY=Math.max(maxY,b.maxY);}const q0=Math.floor((v.worldTop-maxY)/L.c)-4,q1=Math.ceil((v.worldBottom-minY)/L.c)+4;
    for(let q=q0;q<=q1;q++){const vx=q*L.b,vy=q*L.c,p0=Math.floor((v.worldLeft-vx-maxX)/L.a)-4,p1=Math.ceil((v.worldRight-vx-minX)/L.a)+4;for(let p=p0;p<=p1;p++){const tx=p*L.a+vx,ty=vy;for(let node=0;node<defs.length;node++){const d=defs[node],b=cellBounds(d.cells);if(b.maxX+tx<v.worldLeft||b.minX+tx>v.worldRight||b.maxY+ty<v.worldTop||b.minY+ty>v.worldBottom)continue;const ci=colorIndex(colors,node,p,q),spec=paletteSpec(ci,colors.count);drawTranslatedTile(ctx,d.cells,tx,ty,spec.css,v,spec);}}}
  }
  const finiteSpatialMemo=new WeakMap();
  function finiteSpatial(r){let z=finiteSpatialMemo.get(r);if(z)return z;const B=10,buckets=new Map();for(let i=0;i<r.placements.length;i++){const p=r.placements[i],b=p._bounds||(p._bounds=shapeBounds(p.cells)),x0=Math.floor(b.minX/B),x1=Math.floor(b.maxX/B),y0=Math.floor(b.minY/B),y1=Math.floor(b.maxY/B);for(let by=y0;by<=y1;by++)for(let bx=x0;bx<=x1;bx++){const k=`${bx},${by}`;let a=buckets.get(k);if(!a)buckets.set(k,a=[]);a.push(i);}}z={B,buckets};finiteSpatialMemo.set(r,z);return z;}
  function visibleFiniteIndices(r,v){if(r.placements.length<220)return null;const z=finiteSpatial(r),set=new Set(),x0=Math.floor(v.worldLeft/z.B),x1=Math.floor(v.worldRight/z.B),y0=Math.floor(v.worldTop/z.B),y1=Math.floor(v.worldBottom/z.B);for(let by=y0;by<=y1;by++)for(let bx=x0;bx<=x1;bx++){const a=z.buckets.get(`${bx},${by}`);if(a)for(const i of a)set.add(i);}return set;}
  function drawFinite(ctx,r,v){const info=ensureColorMap(r),visible=visibleFiniteIndices(r,v),indices=visible?[...visible]:r.placements.map((_,i)=>i);for(const i of indices){const p=r.placements[i],b=p._bounds||(p._bounds=shapeBounds(p.cells));if(b.maxX<v.worldLeft||b.minX>v.worldRight||b.maxY<v.worldTop||b.minY>v.worldBottom)continue;const ci=colorIndex(info,i,0,0),spec=paletteSpec(ci,info.count);drawTileCells(ctx,p.cells,spec.css,v,spec);}}
  function drawAxes(){/* Intentionally empty: a periodic base-cell boundary must never be visible. */}

  function toggleFocusMode(force){const on=force??!document.body.classList.contains('focus-tiling');document.body.classList.toggle('focus-tiling',on);if(els.focusBtn){els.focusBtn.textContent=on?'⤢':'⛶';els.focusBtn.setAttribute('aria-pressed',String(on));}if(on&&document.documentElement.requestFullscreen&&!document.fullscreenElement)document.documentElement.requestFullscreen().catch(()=>{});else if(!on&&document.fullscreenElement)document.exitFullscreen?.().catch(()=>{});setTimeout(()=>{resizeCanvas();scheduleDraw();},30);}
  function parseRatio(v){const [a,b]=String(v||'1:1').split(':').map(Number);return a>0&&b>0?a/b:1;}
  function exportLayout(r,ratio,regions){const periodic=r&&(r.kind==='periodic'||r.kind==='cell'||r.kind==='field');if(periodic){const L=latticeOf(r),cellW=Math.max(1,L.a+Math.abs(L.b)),cellH=Math.max(1,L.c),cols=Math.max(1,Math.ceil(Math.sqrt(regions*ratio*cellH/cellW))),rows=Math.max(1,Math.ceil(regions/cols)),worldW=cols*cellW,worldH=Math.max(rows*cellH,worldW/ratio);return{cx:(cols*L.a+(rows-1)*L.b)/2,cy:rows*L.c/2,worldW,worldH,regions:cols*rows};}const b=resultBounds(r);let worldW=Math.max(1,b.maxX-b.minX+2),worldH=Math.max(1,b.maxY-b.minY+2);if(worldW/worldH<ratio)worldW=worldH*ratio;else worldH=worldW/ratio;return{cx:(b.minX+b.maxX)/2,cy:(b.minY+b.maxY)/2,worldW,worldH,regions:1};}
  function renderExportCanvas(width,height,regions){const r=currentResult();if(!r)throw new Error('No tiling to export');const ratio=width/height,L=exportLayout(r,ratio,regions),canvas=document.createElement('canvas'),dpr=1;canvas.width=width;canvas.height=height;const ctx=canvas.getContext('2d',{alpha:false}),s=Math.min(width/L.worldW,height/L.worldH),v={s,toX:x=>width/2+(x-L.cx)*s,toY:y=>height/2+(y-L.cy)*s,worldLeft:L.cx-width/(2*s)-2,worldRight:L.cx+width/(2*s)+2,worldTop:L.cy-height/(2*s)-2,worldBottom:L.cy+height/(2*s)+2};ctx.fillStyle=voidColor();ctx.fillRect(0,0,width,height);if(r.kind==='periodic'||r.kind==='cell'||r.kind==='field')drawPeriodic(ctx,r,v);else drawFinite(ctx,r,v);return{canvas,layout:L};}
  function svgEsc(s){return String(s).replace(/[&<>\"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}[c]));}
  class SvgCanvasContext{
    constructor(w,h){this.w=w;this.h=h;this.items=[];this.defs=[];this.path='';this.hasPath=false;this.clipId='';this.stack=[];this.m=[1,0,0,1,0,0];this.fillStyle='#000';this.strokeStyle='#000';this.lineWidth=1;this.lineCap='butt';this.lineJoin='miter';this.globalAlpha=1;this.dash=[];}
    save(){this.stack.push({m:this.m.slice(),fillStyle:this.fillStyle,strokeStyle:this.strokeStyle,lineWidth:this.lineWidth,lineCap:this.lineCap,lineJoin:this.lineJoin,globalAlpha:this.globalAlpha,dash:this.dash.slice(),clipId:this.clipId});}
    restore(){const z=this.stack.pop();if(z)Object.assign(this,z);}
    mm(a,b){const A=this.m;this.m=[A[0]*a[0]+A[2]*a[1],A[1]*a[0]+A[3]*a[1],A[0]*a[2]+A[2]*a[3],A[1]*a[2]+A[3]*a[3],A[0]*a[4]+A[2]*a[5]+A[4],A[1]*a[4]+A[3]*a[5]+A[5]];}
    translate(x,y){this.mm([1,0,0,1,x,y]);} rotate(a){const c=Math.cos(a),d=Math.sin(a);this.mm([c,d,-d,c,0,0]);} scale(x,y){this.mm([x,0,0,y,0,0]);} transform(a,b,c,d,e,f){this.mm([a,b,c,d,e,f]);}
    p(x,y){const m=this.m;return[m[0]*x+m[2]*y+m[4],m[1]*x+m[3]*y+m[5]];}
    f(n){return Number(n.toFixed(3));}
    beginPath(){this.path='';this.hasPath=false;} closePath(){this.path+='Z ';}
    moveTo(x,y){const p=this.p(x,y);this.path+=`M${this.f(p[0])} ${this.f(p[1])} `;this.hasPath=true;}
    lineTo(x,y){const p=this.p(x,y);this.path+=`${this.hasPath?'L':'M'}${this.f(p[0])} ${this.f(p[1])} `;this.hasPath=true;}
    rect(x,y,w,h){const p=[this.p(x,y),this.p(x+w,y),this.p(x+w,y+h),this.p(x,y+h)];this.path+=`M${this.f(p[0][0])} ${this.f(p[0][1])} L${this.f(p[1][0])} ${this.f(p[1][1])} L${this.f(p[2][0])} ${this.f(p[2][1])} L${this.f(p[3][0])} ${this.f(p[3][1])} Z `;this.hasPath=true;}
    quadraticCurveTo(cx,cy,x,y){const c=this.p(cx,cy),p=this.p(x,y);this.path+=`Q${this.f(c[0])} ${this.f(c[1])} ${this.f(p[0])} ${this.f(p[1])} `;this.hasPath=true;}
    bezierCurveTo(c1x,c1y,c2x,c2y,x,y){const a=this.p(c1x,c1y),b=this.p(c2x,c2y),p=this.p(x,y);this.path+=`C${this.f(a[0])} ${this.f(a[1])} ${this.f(b[0])} ${this.f(b[1])} ${this.f(p[0])} ${this.f(p[1])} `;this.hasPath=true;}
    ellipse(cx,cy,rx,ry,rot,start,end,ccw=false){let delta=end-start;if(!ccw&&delta<0)delta+=Math.PI*2;if(ccw&&delta>0)delta-=Math.PI*2;if(Math.abs(delta)>Math.PI*2)delta=Math.sign(delta)*Math.PI*2;const steps=Math.max(6,Math.ceil(Math.abs(delta)/(Math.PI/24)));for(let i=0;i<=steps;i++){const a=start+delta*i/steps,ca=Math.cos(a),sa=Math.sin(a),cr=Math.cos(rot),sr=Math.sin(rot),x=cx+rx*ca*cr-ry*sa*sr,y=cy+rx*ca*sr+ry*sa*cr;i===0?(this.hasPath?this.lineTo(x,y):this.moveTo(x,y)):this.lineTo(x,y);}}
    arc(cx,cy,r,start,end,ccw=false){this.ellipse(cx,cy,r,r,0,start,end,ccw);}
    style(fill){const c=svgEsc(fill?this.fillStyle:this.strokeStyle),clip=this.clipId?` clip-path=\"url(#${this.clipId})\"`:'',op=this.globalAlpha<.999?` opacity=\"${this.f(this.globalAlpha)}\"`:'';if(fill)return`fill=\"${c}\" stroke=\"none\"${clip}${op}`;const dash=this.dash.length?` stroke-dasharray=\"${this.dash.map(x=>this.f(x)).join(' ')}\"`:'';return`fill=\"none\" stroke=\"${c}\" stroke-width=\"${this.f(this.lineWidth)}\" stroke-linecap=\"${this.lineCap}\" stroke-linejoin=\"${this.lineJoin}\"${dash}${clip}${op}`;}
    fill(){if(this.path)this.items.push(`<path d=\"${this.path.trim()}\" ${this.style(true)}/>`);} stroke(){if(this.path)this.items.push(`<path d=\"${this.path.trim()}\" ${this.style(false)}/>`);}
    fillRect(x,y,w,h){this.beginPath();this.rect(x,y,w,h);this.fill();} strokeRect(x,y,w,h){this.beginPath();this.rect(x,y,w,h);this.stroke();}
    clip(){if(!this.path)return;const id=`clip${this.defs.length+1}`;this.defs.push(`<clipPath id=\"${id}\"><path d=\"${this.path.trim()}\"/></clipPath>`);this.clipId=id;}
    setLineDash(a){this.dash=(a||[]).slice();}
    toSVG(title='tiling.lab β9 export'){return `<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"${this.w}\" height=\"${this.h}\" viewBox=\"0 0 ${this.w} ${this.h}\"><title>${svgEsc(title)}</title>${this.defs.length?`<defs>${this.defs.join('')}</defs>`:''}${this.items.join('')}</svg>`;}
  }
  function renderExportSVG(width,height,regions){const r=currentResult();if(!r)throw new Error('No tiling to export');const ratio=width/height,L=exportLayout(r,ratio,regions),ctx=new SvgCanvasContext(width,height),s=Math.min(width/L.worldW,height/L.worldH),v={s,toX:x=>width/2+(x-L.cx)*s,toY:y=>height/2+(y-L.cy)*s,worldLeft:L.cx-width/(2*s)-2,worldRight:L.cx+width/(2*s)+2,worldTop:L.cy-height/(2*s)-2,worldBottom:L.cy+height/(2*s)+2};ctx.fillStyle=voidColor();ctx.fillRect(0,0,width,height);if(r.kind==='periodic'||r.kind==='cell'||r.kind==='field')drawPeriodic(ctx,r,v);else drawFinite(ctx,r,v);return ctx.toSVG();}
  function exportSVG(width,height,regions){return renderExportSVG(width,height,regions);}
  function dataUrlBytes(url){const b64=url.split(',')[1],bin=atob(b64),out=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)out[i]=bin.charCodeAt(i);return out;}
  function pdfFromJpeg(dataUrl,w,h){const bytes=dataUrlBytes(dataUrl),enc=new TextEncoder(),parts=[],offsets=[0],push=s=>parts.push(typeof s==='string'?enc.encode(s):s),size=()=>parts.reduce((n,p)=>n+p.length,0),obj=(n,body)=>{offsets[n]=size();push(`${n} 0 obj\n${body}\nendobj\n`);};push('%PDF-1.4\n');obj(1,'<< /Type /Catalog /Pages 2 0 R >>');obj(2,'<< /Type /Pages /Kids [3 0 R] /Count 1 >>');const pw=720,ph=pw*h/w;obj(3,`<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${pw.toFixed(2)} ${ph.toFixed(2)}] /Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>`);const content=`q\n${pw.toFixed(2)} 0 0 ${ph.toFixed(2)} 0 0 cm\n/Im0 Do\nQ\n`;obj(4,`<< /Length ${content.length} >>\nstream\n${content}endstream`);offsets[5]=size();push(`5 0 obj\n<< /Type /XObject /Subtype /Image /Width ${w} /Height ${h} /ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${bytes.length} >>\nstream\n`);push(bytes);push('\nendstream\nendobj\n');const xref=size();push('xref\n0 6\n0000000000 65535 f \n');for(let i=1;i<=5;i++)push(String(offsets[i]).padStart(10,'0')+' 00000 n \n');push(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF`);return new Blob(parts,{type:'application/pdf'});}
  function downloadBlob(blob,name){const a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1200);}
  async function exportCurrent(){const r=currentResult();if(!r){setStatus('nothing to export','error');return;}const ratio=parseRatio(els.exportRatio?.value),longSide=Math.max(800,Math.min(6000,+els.exportLongSide?.value||2400)),regions=Math.max(1,Math.min(144,+els.exportRegions?.value||24)),width=ratio>=1?longSide:Math.round(longSide*ratio),height=ratio>=1?Math.round(longSide/ratio):longSide,format=els.exportFormat?.value||'svg',base=`tiling-lab-v9-${state.catalogId||'custom'}-${Date.now()}`;try{setStatus('exporting','busy');if(format==='svg'){const svg=exportSVG(width,height,regions);downloadBlob(new Blob([svg],{type:'image/svg+xml'}),base+'.svg');}else{const {canvas}=renderExportCanvas(width,height,regions);if(format==='png'){await new Promise(resolve=>canvas.toBlob(b=>{downloadBlob(b,base+'.png');resolve();},'image/png'));}else if(format==='jpg'){await new Promise(resolve=>canvas.toBlob(b=>{downloadBlob(b,base+'.jpg');resolve();},'image/jpeg',.96));}else{const jpeg=canvas.toDataURL('image/jpeg',.99);downloadBlob(pdfFromJpeg(jpeg,width,height),base+'.pdf');}}setStatus(t('ready'));els.downloadMenu?.classList.add('hidden');}catch(err){console.error(err);setStatus(`export error · ${err.message}`,'error');}}
  function wireEvents(){
    els.langBtn.addEventListener('click',()=>{state.lang=state.lang==='en'?'zh':'en';applyLanguage();});
    els.clearBtn.addEventListener('click',()=>{state.cells.clear();state.catalogId=null;state.catalogMeta=null;clearResultsForNewShape();renderEditor();renderCatalog();updateSolveLabel();});
    els.normalizeBtn.addEventListener('click',()=>{setShape(normalizeShape(),state.catalogId,state.catalogMeta);});
    els.sampleBtn.addEventListener('click',sampleCatalog);
    els.solveBtn.addEventListener('click',solve);els.stopBtn.addEventListener('click',stop);
    els.catalogGrid.addEventListener('click',e=>{const b=e.target.closest('[data-id]');if(!b)return;const s=state.catalogShapes.find(x=>x.id===b.dataset.id);if(s)setShape(s.cells,s.id,s);});
    els.prevPage.addEventListener('click',()=>requestCatalogPage(state.catalogPage-1));els.nextPage.addEventListener('click',()=>requestCatalogPage(state.catalogPage+1));
    els.catalogSearch.addEventListener('input',()=>{clearTimeout(state.searchDebounce);state.searchDebounce=setTimeout(()=>{state.catalogPage=0;requestCatalogPage(0);},180);});
    els.resultList.addEventListener('click',e=>{const b=e.target.closest('[data-i]');if(!b)return;state.activeResult=state.results[+b.dataset.i];state.previewResult=null;state.colorCacheKey='';renderResults();updateMetrics();centerViewOnResult(state.activeResult,true);scheduleDraw();});
    els.rotateCCWBtn?.addEventListener('click',()=>composeViewTransform([0,1,-1,0]));els.rotateCWBtn?.addEventListener('click',()=>composeViewTransform([0,-1,1,0]));els.flipVBtn?.addEventListener('click',()=>composeViewTransform([1,0,0,-1]));els.flipHBtn?.addEventListener('click',()=>composeViewTransform([-1,0,0,1]));els.zoomOutBtn?.addEventListener('click',()=>zoomStep(1/1.18));els.zoomInBtn?.addEventListener('click',()=>zoomStep(1.18));els.resetViewBtn.addEventListener('click',resetView);els.infoBtn.addEventListener('click',()=>els.infoDialog.showModal());els.focusBtn?.addEventListener('click',()=>toggleFocusMode());document.addEventListener('fullscreenchange',()=>{if(!document.fullscreenElement&&document.body.classList.contains('focus-tiling'))toggleFocusMode(false);});els.downloadBtn?.addEventListener('click',e=>{e.stopPropagation();els.downloadMenu?.classList.toggle('hidden');});els.downloadMenu?.addEventListener('click',e=>e.stopPropagation());document.addEventListener('click',()=>els.downloadMenu?.classList.add('hidden'));els.exportGo?.addEventListener('click',exportCurrent);
  }

  initEditor();initSelectors();initAdvancedSearch();initColorSettings();createSolverWorker();initCanvas();wireEvents();applyLanguage();requestCatalog();scheduleDraw();
})();
