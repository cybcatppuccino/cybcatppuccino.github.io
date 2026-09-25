(() => {
'use strict';

// -----------------------------------------------------------------------------
// Data
// -----------------------------------------------------------------------------
const V4=window.V4_DATA||{}, EXT=window.EXTENDED_DATA||{}, V9=window.V9_DATA||{}, V11=window.V11_DATA||{}, V12=window.V12_DATA||{}, V13=window.V13_DATA||{}, V15=window.V15_DATA||{}, V16=window.V16_DATA||{}, V17=window.V17_DATA||{}, V18=window.V18_DATA||{}, V20=window.V20_DATA||{}, V21=window.V21_DATA||{}, V22=window.V22_DATA||{}, V24=window.V24_DATA||{}, V25=window.V25_DATA||{}, V27=window.V27_DATA||{}, V28=window.V28_DATA||{}, V29=window.V29_DATA||{};
const CORE=V4.core||[], $=id=>document.getElementById(id);
const tiling=$('tiling'), fallback=$('fallback'), overlay=$('overlay'), ctx=overlay.getContext('2d');
let fallbackCtx=fallback.getContext('2d');
const byId=new Map(CORE.map(g=>[g.id,g]));
const bySig=new Map(CORE.map(g=>[g.signature,g]));
const list=(o,k)=>Array.isArray(o)?o:(o&&Array.isArray(o[k])?o[k]:[]);
const lambdaMap=new Map(list(V4.lambdaExact,'records').map(r=>[r.takeuchi_id,r]));
const hauptMap=new Map(list(V4.hauptmodul,'records').map(r=>[r.id,r]));
const lowById=new Map(); for(const r of list(V4.lowGenusLandscape,'records')){if(!lowById.has(r.takeuchi_id))lowById.set(r.takeuchi_id,[]);lowById.get(r.takeuchi_id).push(r)}
const genusById=new Map(); for(const r of list(V4.genus1,'rows')){if(!genusById.has(r.takeuchi_id))genusById.set(r.takeuchi_id,[]);genusById.get(r.takeuchi_id).push(r)}
const geo=EXT.geodesicElements||{}, triLow=EXT.triangleLowGenus||{}, modular=EXT.modularCongruence||[];
const modularMap=new Map(modular.map(r=>[r.label,r]));
const modularPerm=V9.modularPermutations||{};
const inclusions=(V4.inclusionEdges&&V4.inclusionEdges.edges)||[];
const hgTransforms=(V9.hypergeometricTransformations&&V9.hypergeometricTransformations.transformations)||[];
const hgCoverage=(V9.commensurabilityCoverage&&V9.commensurabilityCoverage.classes)||[];
const commIntersections=V11.commIntersections||[];
const exactQuatMap=new Map((V12.exactQuaternions||[]).map(r=>[Number(r.class),r]));
const pairRelations=V12.pairRelations||[];
const pairByGroup=new Map(); for(const r of pairRelations){for(const id of [r.a,r.b]){if(!pairByGroup.has(id))pairByGroup.set(id,[]);pairByGroup.get(id).push(r)}}
const higherGenus=V12.higherGenus||{};
const lowIndexSubgroups=V18.lowIndexExpanded||V17.lowIndexExpanded||V13.lowIndexSubgroups||{}, lowIndexMeta={...(V17.lowIndexMeta||{}),...(V18.lowIndexMeta||{})}, modularSubgroupCensus=V13.modularSubgroupCensus||[];
const explicitHGPair={...(V13.explicitHypergeometricPairs||{}),...(V15.explicitHypergeometricPairs||{}),...(V27.explicitHypergeometricPairs||{}),...(V28.explicitHypergeometricPairs||{})};
const noncompactData=V15.noncompactTriangleData||{}, noncompactRelations=V15.noncompactCoverRelations||[], quadrilateralGroups=V15.quadrilateralGroups||[], references=V15.references||[];
const exactTriangleCovers=(V16.triangleCoverTriples||[]).concat(V17.triangleCoverTriplesExpanded||[]), exactQuadrilateralCovers=V17.quadrilateralCoversExpanded||V16.quadrilateralCovers||[], cpGenus0=V17.cpGenus0Updated||V16.cpGenus0||{}, genus1J=V17.genus1JUpdated||V16.genus1J||{}, gamma0Haupt=V16.gamma0Hauptmodul||{}, gamma0Relations=V16.gamma0Relations||{}, cpQSeries20={...(V17.cpQSeries20||{}),...(V29.cpQSeries20Corrections||{})}, cpQSeries60={...(V20.cpQSeries60||{}),...(V21.cpQSeries60||{}),...(V29.cpQSeries60||{})}, v20HauptRelations={...(V20.hauptmodulRelations||{}),...(V29.hauptmodulRelations||{})}, supplementalHauptRelations=V20.supplementalHauptmodulRelations||{}, moonshine=V20.moonshine||{}, frickeDomains=V21.frickeDomains||{}, gamma0FordDomains=V21.gamma0FordDomains||{}, frickeClasses=V21.frickeClasses||{}, frickeHauptRelations=V21.frickeHauptRelations||{}, cpCongruence=V17.cpCongruenceDescriptions||{}, noncompactFourier=V16.noncompactFourier||{}, moonshineGroups=V22.moonshineGroups||{}, moonshineSymbols=V22.moonshineSymbols||{}, moonshineRelations=V22.moonshineRelations||[], moonshineAdjacency=V22.moonshineAdjacency||{}, hgComposed={...(V24.hypergeometricComposed||{}),...(V27.hypergeometricComposed||{}),...(V28.hypergeometricComposed||{})}, modularFormsGamma0=V24.modularFormsGamma0||{}, moonCongruenceRelations=(V24.moonshineCongruenceRelations||[]).concat(V25.moonshineCongruenceTransformedRelations||[]), noncompactGeneratorSeries=V27.noncompactQ60||V24.noncompactGeneratorSeries||{}, inclusionNormality=V27.inclusionNormality||{}, moonshineNormality=V27.moonshineNormality||{}, moonshineModularForms=V24.moonshineModularForms||{}, cpGenus0ModularForms=V25.cpGenus0ModularForms||{}, gamma1CharacterForms=V25.gamma1CharacterForms||{}, moonshineJRelations=V25.moonshineJRelations||{};
const relationTemplates={...(V12.relationTemplates||{}),...(V27.relationTemplates||{}),...(V28.relationTemplates||{})};
const hypergeometricV28Meta=V28.hypergeometricV28Meta||{};
const belyiDatabaseMatches=V27.belyiDatabaseMatches||{}, belyiModels=V27.belyiModels||{};
const exactTriangleCoverMap=new Map(exactTriangleCovers.map(r=>[r.parent+'|'+r.child,r]));
const exactQuadMap=new Map(exactQuadrilateralCovers.map(r=>[r.id,r]));

let current=byId.get(location.hash.slice(1))||byId.get('T2_3_I')||CORE[0];
let panel='group', model='disk', selectedCover=null, selectedHGPeer=null, selectedCommPeer=null, selectedMoonshine=null, selectedOrbifoldPoint=null, vertexHitTargets=[], commSceneActive=false, commCompare=null, frickeCompare=null;
let W=1,H=1,dpr=1,zoom=1,drag=false,lastX=0,lastY=0,keys=new Set(),keyRAF=0,lastKey=0;
let raf=0, overlayDirty=true, bgDirty=true, idleTimer=0;
const viewSettings={vertices:true,allEdgeVertices:false,ellipticVertices:true,cuspVertices:true,vertexLabel:'class',vertexSize:1,pairings:true,arrows:true,edgeLabels:true,edgeSize:1,boundarySize:1,cellSeams:true,hatching:true,triangleSeams:true,neighbors:true,neighborCount:0,neighborOpacity:1,regionTags:true,legend:true,coverMarksOnly:true};

// -----------------------------------------------------------------------------
// Language
// -----------------------------------------------------------------------------
let uiLang='en';
const UI_TEXT={
  en:{display:'Display',language:'Language',vertices:'Vertices',allEdgeVertices:'All edge vertices',ellipticVertices:'Elliptic',cusps:'Cusps',vertexLabels:'Vertex labels',vertexSize:'Vertex size',pairings:'Pairings',orientation:'Orientation',edgeLabels:'Edge labels',edgeLabelSize:'Edge label size',boundaryWidth:'Boundary width',cellSeams:'Cell seams',hatching:'Hatching',triangleSeams:'Triangle seams',neighbors:'Neighbors',neighborCount:'Neighbor layers',neighborOpacity:'Neighbor opacity',regionTags:'Region labels',legend:'Legend',coverMarksOnly:'Cover marks only',displaySettings:'Display settings',close:'Close',resizePanel:'Resize details panel',searchGroup:'Search triangle group',supergroups:'Supergroups',subgroups:'Subgroups',directInclusions:'Direct inclusions',directMaps:'Direct maps',relations:'Relations',references:'References',localParameter:'Local parameter',lowIndex:'Low-index inclusions',conjugacyClasses:'Conjugacy classes',higherGenus:'Higher genus',quadrilaterals:'Quadrilaterals',simultaneous:'simultaneous',commonSubgroups:'Common subgroups',qSeries:'series'},
  zh:{display:'显示',language:'语言',vertices:'顶点',allEdgeVertices:'所有边顶点',ellipticVertices:'椭圆点',cusps:'尖点',vertexLabels:'顶点标注',vertexSize:'顶点大小',pairings:'边配对',orientation:'方向',edgeLabels:'边记号',edgeLabelSize:'边记号大小',boundaryWidth:'边界粗细',cellSeams:'子域虚线',hatching:'填充纹理',triangleSeams:'三角分割',neighbors:'相邻域',neighborCount:'相邻层数',neighborOpacity:'相邻透明度',regionTags:'区域记号',legend:'图例',coverMarksOnly:'仅显示覆盖标记',displaySettings:'显示设置',close:'关闭',resizePanel:'调整详情栏宽度',searchGroup:'搜索三角群',supergroups:'上群',subgroups:'子群',directInclusions:'直接包含',directMaps:'直接映射',relations:'关系',references:'参考',localParameter:'局部参数',lowIndex:'低指数包含',conjugacyClasses:'共轭类',higherGenus:'高亏格',quadrilaterals:'四边形',simultaneous:'同时',commonSubgroups:'公共子群',qSeries:'级数'}
};
function tr(k){return UI_TEXT[uiLang]?.[k]??UI_TEXT.en[k]??k}
function loadLanguage(){try{const v=localStorage.getItem('atlas.language');uiLang=v==='zh'?'zh':'en'}catch{uiLang='en'}}
function saveLanguage(){try{localStorage.setItem('atlas.language',uiLang)}catch{}}
function applyLanguage({rerender=true}={}){
  document.documentElement.lang=uiLang==='zh'?'zh-CN':'en';
  document.querySelectorAll('[data-i18n]').forEach(el=>{const k=el.dataset.i18n;if(k)el.textContent=tr(k)});
  document.querySelectorAll('[data-i18n-aria]').forEach(el=>{const k=el.dataset.i18nAria;if(k)el.setAttribute('aria-label',tr(k))});
  const sel=$('set-language');if(sel)sel.value=uiLang;
  if(rerender){if($('sheet')?.classList.contains('show'))setHTML($('sheet-content'),panelHTML(panel,current));if($('group-picker')?.classList.contains('show'))picker()}
}

// -----------------------------------------------------------------------------
// Presentation / exact notation
// -----------------------------------------------------------------------------
const esc=s=>String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const ord=x=>(x==='inf'||x==null)?null:Number(x);
const sig=g=>`(${[g.a,g.b,g.c].map(x=>x==='inf'||x==null?'\\infty':x).join(',')})`;
const texErrors=[];
const TEX_COMMAND_WORDS='quad|qquad|left|right|frac|tfrac|dfrac|sqrt|mathbf|mathrm|mathfrak|operatorname|Delta|Gamma|lambda|infty|varnothing|text|binom|overline|widehat|mathbb|mathcal|zeta|tau|pi|ker|cong|subset|subseteq|supset|supseteq';
function auditTexSource(html){
  const blocks=String(html??'').match(/\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)/g)||[];
  for(const b of blocks){
    const bare=new RegExp('(^|[^\\\\A-Za-z])('+TEX_COMMAND_WORDS+')(?:[^A-Za-z]|$)');
    const doubled=new RegExp('\\\\\\\\(?:'+TEX_COMMAND_WORDS+')(?:[^A-Za-z]|$)');
    if(bare.test(b)||doubled.test(b)){const msg='TeX source audit: '+b.slice(0,260);texErrors.push(msg);console.error(msg)}
  }
}
function auditTypeset(el){const bad=[...el.querySelectorAll('[data-mml-node="merror"],mjx-merror')],raw=[],rx=new RegExp('\\\\(?:rm|'+TEX_COMMAND_WORDS+')(?:[^A-Za-z]|$)'),walk=document.createTreeWalker(el,NodeFilter.SHOW_TEXT);let n;while((n=walk.nextNode()))if(rx.test(n.nodeValue||''))raw.push((n.nodeValue||'').slice(0,180));if(bad.length||raw.length){const msg=bad.map(x=>x.textContent||'MathJax error').concat(raw.map(x=>'raw TeX: '+x)).join(' | ');texErrors.push(msg);console.error('MathJax:',msg)}return bad.length===0&&!raw.length}
function typeset(el=document.body){if(!window.MathJax?.typesetPromise)return Promise.resolve(true);return MathJax.typesetPromise([el]).then(()=>auditTypeset(el)).catch(err=>{texErrors.push(String(err));console.error(err);return false})}
function normalizeTexHtml(html){
  return String(html??'')
    .replace(/\{\}_2F_1/g,'{}_{2}F_{1}').replace(/(^|[^{}])_2F_1/g,'$1{}_{2}F_{1}')
    .replace(/\\\\([\[\]\(\)])/g,'\\$1')
    .replace(/\\\\([,;:!])/g,'\\$1')
    .replace(new RegExp('\\\\\\\\(?=(?:rm|'+TEX_COMMAND_WORDS+')(?:[^A-Za-z]|$))','g'),'\\')
    .replace(/\\rm\s+\\Gamma/g,'\\Gamma').replace(/\\rm\s+\\Delta/g,'\\Delta')
    .replace(/\\rm\b\s*([A-Za-z]+(?:[ _-][A-Za-z]+)*)/g,(_,x)=>`\\mathrm{${x.trim().replace(/\s+/g,'\\ ')}}`)
    .replace(/\\Gamma_([01])(?![A-Za-z0-9{])/g,'\\Gamma_{$1}')
    .replace(/\\mathrm\{(SL|GL)\}_([23])(?![A-Za-z0-9{])/g,'\\mathrm{$1}_{$2}')
    .replace(/([A-Za-z])_\\Gamma_\{([01])\}/g,'$1_{\\Gamma_{$2}}')
    .replace(/_\\infty(?![A-Za-z])/g,'_{\\infty}')
    .replace(/\\left\(([^\n]*?)\\right(?!\))/g,'\\left($1\\right)')
}
function compactUiHtml(html){
  return String(html??'')
    .replace(/<div class="source-note[^"]*">[\s\S]*?<\/div>/g,'')
    .replace(/<div class="mathline tiny muted">(?:\((\sigma_0,\sigma_1,\sigma_\infty)\)|no coset action)[\s\S]*?<\/div>/g,'')
    .replace(/<div class="mathline small muted">([^<]*[A-Za-z][^<]*)<\/div>/g,'')
    .replace(/supergroup → subgroup/g,uiLang==='zh'?'上群 → 子群':'supergroup → subgroup')
    .replace(/independent Ford polygons/g,'Ford')
}
function setHTML(el,html){html=compactUiHtml(normalizeTexHtml(html));auditTexSource(html);if(window.MathJax?.typesetClear)try{MathJax.typesetClear([el])}catch{}el.innerHTML=html;return typeset(el)}
function gcd(a,b){a=Math.abs(a);b=Math.abs(b);while(b){const t=a%b;a=b;b=t}return a||1}
function rationalApprox(x,maxQ=4096,tol=2e-12){if(!Number.isFinite(x))return null;if(Math.abs(x-Math.round(x))<tol)return[Math.round(x),1];let h0=0,h1=1,k0=1,k1=0,y=x;for(let i=0;i<24;i++){let a=Math.floor(y),h=a*h1+h0,k=a*k1+k0;if(k>maxQ)break;if(Math.abs(x-h/k)<tol*Math.max(1,Math.abs(x)))return[h,k];h0=h1;h1=h;k0=k1;k1=k;let f=y-a;if(Math.abs(f)<1e-15)break;y=1/f}return null}
function squarefree(n){for(let p=2;p*p<=n;p++)if(n%(p*p)===0)return false;return true}
function exactTex(x){if(typeof x==='string'){const s=x.trim();if(/^[-+]?\d+\/\d+$/.test(s))return ratTex(s);if(/^[-+]?\d+(?:\.\d+)?$/.test(s))return exactTex(Number(s));return exprTex(s)}if(!Number.isFinite(Number(x)))return String(x??'—');x=Number(x);const r=rationalApprox(x);if(r){const[p,q]=r;return q===1?String(p):`\\frac{${p}}{${q}}`}
  for(let d=2;d<=50;d++)if(squarefree(d)){const q=rationalApprox(x/Math.sqrt(d),256,8e-12);if(q&&Math.abs(q[0])<=512){const [p,n]=q,coef=n===1?(Math.abs(p)===1?(p<0?'-':''):String(p)):`\\frac{${p}}{${n}}`;return `${coef}\\sqrt{${d}}`}}
  // small a+b sqrt(d), common in trace fields
  for(let d=2;d<=30;d++)if(squarefree(d)){const sd=Math.sqrt(d);for(let den=1;den<=16;den++){const X=x*den;for(let b=-16;b<=16;b++){const a=Math.round(X-b*sd);if(Math.abs(X-(a+b*sd))<2e-10){const num=(a?String(a):'')+(b?(a&&b>0?'+':'')+(Math.abs(b)===1?(b<0?'-':''):String(b))+`\\sqrt{${d}}`:'');return den===1?num:`\\frac{${num}}{${den}}`}}}}
  return Number(x.toPrecision(10)).toString()}
function ratTex(q){q=String(q??'0').trim();let sign='';if(q.startsWith('-')){sign='-';q=q.slice(1)}const m=q.match(/^(\d+)\/(\d+)$/);return sign+(m?`\\frac{${m[1]}}{${m[2]}}`:q)}
function rqNorm(n,d=1){if(d<0){n=-n;d=-d}const g=gcd(n,d);return[n/g,d/g]}
function rqParse(x){x=String(x??'0').trim();const m=x.match(/^(-?\d+)(?:\/(\d+))?$/);return m?rqNorm(Number(m[1]),Number(m[2]||1)):null}
function rqAdd(a,b){return rqNorm(a[0]*b[1]+b[0]*a[1],a[1]*b[1])}
function rqSub(a,b){return rqNorm(a[0]*b[1]-b[0]*a[1],a[1]*b[1])}
function rqNeg(a){return[-a[0],a[1]]}
function rqMul(a,b){return rqNorm(a[0]*b[0],a[1]*b[1])}
function rqDiv(a,b){return rqNorm(a[0]*b[1],a[1]*b[0])}
function rqScale(a,k){return rqNorm(a[0]*k,a[1])}
function rqTex(a){if(!a)return'?';return a[1]===1?String(a[0]):`${a[0]<0?'-':''}\\frac{${Math.abs(a[0])}}{${a[1]}}`}
function hfTex(a,b,c,z){return `{}_{2}F_{1}\\!\\left(${rqTex(a)},${rqTex(b)};${rqTex(c)};${z}\\right)`}

function decimalExactify(s){return String(s).replace(/[-+]?(?:\d+\.\d+|\d+\.|\.\d+)(?:[eE][-+]?\d+)?/g,m=>{const x=Number(m);if(!Number.isFinite(x))return m;const t=exactTex(x),plain=Number(x.toPrecision(10)).toString();return t===plain?m:`{${t}}`})}
function exprTex(s){s=decimalExactify(String(s??'').trim());s=s.replace(/sqrt\(([^()]+)\)/g,'\\sqrt{$1}').replace(/\*/g,'\\,').replace(/\^\(?(-?\d+)\)?/g,'^{$1}');s=s.replace(/\bpi\b/g,'\\pi').replace(/\binf\b/g,'\\infty');return s}
function polyTex(s){return exprTex(String(s??'').replace(/\s+/g,' '))}
function fieldTex(s){s=String(s??'Q').replace(/^Q/,'\\mathbf Q');return exprTex(s)}
const cyclotomicCache=new Map([[1,[-1,1]]]);
function intPolyTrim(a){a=a.slice();while(a.length>1&&a[a.length-1]===0)a.pop();return a}
function intPolyDivExact(a,b){a=a.slice();b=intPolyTrim(b);const q=new Array(Math.max(1,a.length-b.length+1)).fill(0),lead=b[b.length-1];for(let k=a.length-b.length;k>=0;k--){const c=a[k+b.length-1]/lead;q[k]=c;for(let j=0;j<b.length;j++)a[j+k]-=c*b[j]}if(a.some(x=>Math.abs(x)>1e-9))return null;return intPolyTrim(q.map(x=>Math.round(x)))}
function cyclotomicCoeffs(n){n=Math.max(1,Math.round(Number(n)||1));if(cyclotomicCache.has(n))return cyclotomicCache.get(n);let p=new Array(n+1).fill(0);p[0]=-1;p[n]=1;for(let d=1;d<n;d++)if(n%d===0){const q=intPolyDivExact(p,cyclotomicCoeffs(d));if(q)p=q}p=intPolyTrim(p);cyclotomicCache.set(n,p);return p}
function integerPolyTex(coeffs,v='\\alpha'){const terms=[];for(let i=coeffs.length-1;i>=0;i--){const c=Number(coeffs[i]||0);if(!c)continue;const neg=c<0,a=Math.abs(c),mon=i===0?'':(i===1?v:`${v}^{${i}}`),coef=i===0||a!==1?String(a):'',body=coef+mon;if(!terms.length)terms.push((neg?'-':'')+body);else terms.push((neg?' - ':' + ')+body)}return terms.join('')||'0'}
function decimalTexNumber(x,d=9){if(!Number.isFinite(Number(x)))return'—';const y=Math.abs(Number(x))<5e-13?0:Number(x),z=Number(y.toPrecision(d));return String(z)}
function complexApproxTex(re,im,d=9){re=Number(re);im=Number(im);if(!Number.isFinite(re+im))return'';if(Math.abs(im)<5e-12)return decimalTexNumber(re,d);if(Math.abs(re)<5e-12)return`${im<0?'-':''}${decimalTexNumber(Math.abs(im),d)}i`;return`${decimalTexNumber(re,d)}${im<0?'-':'+'}${decimalTexNumber(Math.abs(im),d)}i`}
function quadraticRootApprox(a,b,c){a=Number(a);b=Number(b);c=Number(c);if(!Number.isFinite(a+b+c)||Math.abs(a)<1e-15)return'';const d=b*b-4*a*c;if(d>=0)return decimalTexNumber((-b+Math.sqrt(d))/(2*a),10);return complexApproxTex(-b/(2*a),Math.sqrt(-d)/(2*Math.abs(a)),10)}
function fieldGeneratorSpec(field,meta={}){
  const f=String(field||''),symbol=meta.generator_symbol_tex||meta.field_generator_symbol_tex||'\\alpha',mp=meta.generator_minpoly_tex||meta.field_minpoly_tex||meta.minpoly_tex,ap=meta.generator_approx_tex||meta.field_generator_approx_tex||meta.approx_tex;
  if(mp)return{symbol,minpoly:String(mp).replace(/\bx\b/g,symbol),approx:ap?String(ap):''};
  let m=f.match(/\\(alpha|beta|omega|xi|nu|theta)\s*=\s*\\zeta_\{?(\d+)\}?/);if(m){const sym=`\\${m[1]}`,n=Number(m[2]),th=2*Math.PI/n;return{symbol:sym,minpoly:integerPolyTex(cyclotomicCoeffs(n),sym),approx:complexApproxTex(Math.cos(th),Math.sin(th),10)}}
  m=f.match(/\\(alpha|beta|omega|xi|nu|theta)\^\{?2\}?\s*([+-])\s*\\\1\s*([+-])\s*(\d+)\s*=\s*0/);if(m){const sym=`\\${m[1]}`,bb=m[2]==='+'?1:-1,c=(m[3]==='+'?1:-1)*Number(m[4]);return{symbol:sym,minpoly:`${sym}^{2}${bb>0?'+':'-'}${sym}${c>0?'+':''}${c}`,approx:quadraticRootApprox(1,bb,c)}}
  m=f.match(/\\(alpha|beta|omega|xi|nu|theta)\^\{?2\}?\s*([+-])\s*(\d+)\s*=\s*0/);if(m){const sym=`\\${m[1]}`,c=(m[2]==='+'?1:-1)*Number(m[3]);return{symbol:sym,minpoly:`${sym}^{2}${c>0?'+':''}${c}`,approx:quadraticRootApprox(1,0,c)}}
  return null
}
function fieldGeneratorHTML(field,meta={}){const q=fieldGeneratorSpec(field,meta);if(!q||!q.minpoly)return'';const mp=String(q.minpoly).replace(/\s*=\s*0\s*$/,'');return String.raw`<div class="mathline tiny formula-scroll">\[${mp}=0${q.approx?`,\\qquad ${q.symbol}\\approx ${q.approx}`:''}.\]</div>`}
function fieldGeneratorsHTML(field,meta={}){const gs=Array.isArray(meta?.field_generators)?meta.field_generators:[];if(!gs.length)return fieldGeneratorHTML(field,meta);return gs.map(g=>{const sym='\\'+String(g.symbol||'alpha').replace(/^\\/,'');const mp=String(g.minpoly||'').replace(/\bx\b/g,sym).replace(/\s*=\s*0\s*$/,'');const ap=String(g.approx_tex||g.approx||'');return mp?String.raw`<div class="mathline tiny formula-scroll">\[${mp}=0${ap?`,\\qquad ${sym}\\approx ${ap}`:''}.\]</div>`:''}).join('')}
function areaTex(g){const vals=[ord(g.a),ord(g.b),ord(g.c)],terms=vals.map(n=>n?`\\frac1{${n}}`:'0');let x=2*(1-vals.reduce((a,n)=>a+(n?1/n:0),0));return `2\\left(1-${terms.join('-')}\\right)\\pi=${exactTex(x)}\\pi`}
function factorTex(o){if(!o||!Object.keys(o).length)return '1';return Object.entries(o).map(([p,e])=>e==1?p:`${p}^{${e}}`).join('\\,')}
function seriesTex(coeffs,maxN=14){if(!Array.isArray(coeffs))return '';const t=[];for(let n=1;n<Math.min(coeffs.length,maxN+1);n++){let q=String(coeffs[n]??'0').trim();if(q==='0')continue;const neg=q.startsWith('-');if(neg)q=q.slice(1);let c=ratTex(q),u=n===1?'u':`u^{${n}}`;if(q==='1')c='';t.push((t.length?(neg?' - ':' + '):(neg?'-':''))+c+u)}return t.join('')+`+O(u^{${Math.min(coeffs.length,maxN+1)}})`}
function formulaTex(s){let x=decimalExactify(String(s??''));x=x.replace(/2F1\(([^;]+);\s*([^\)]+)\)/g,(_,ab,c)=>`{}_{2}F_{1}\\!\\left(${ab};${c}\\right)`);x=x.replace(/2F1\(([^,]+),([^,]+);([^;]+);\s*([^\)]+)\)/g,(_,a,b,c,z)=>`{}_{2}F_{1}\\!\\left(${a},${b};${c};${z}\\right)`);x=x.replace(/\^\(([^)]+)\)/g,'^{$1}').replace(/\^([A-Za-z0-9+\/-]+)/g,'^{$1}').replace(/\*/g,'\\,').replace(/sqrt\(([^)]+)\)/g,'\\sqrt{$1}');return x}

// -----------------------------------------------------------------------------
// Complex numbers and disk isometries (holomorphic or antiholomorphic)
// -----------------------------------------------------------------------------
function C(x=0,y=0){return{x,y}} const add=(a,b)=>C(a.x+b.x,a.y+b.y),sub=(a,b)=>C(a.x-b.x,a.y-b.y),sc=(a,s)=>C(a.x*s,a.y*s);
function mul(a,b){return C(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x)} const conj=a=>C(a.x,-a.y),abs2=a=>a.x*a.x+a.y*a.y,absv=a=>Math.hypot(a.x,a.y);
function div(a,b){const d=abs2(b);return d<1e-30?C(1e14,1e14):C((a.x*b.x+a.y*b.y)/d,(a.y*b.x-a.x*b.y)/d)}
function I(){return{a:C(1),b:C(),c:C(),d:C(1)}}
function mm(A,B){return{a:add(mul(A.a,B.a),mul(A.b,B.c)),b:add(mul(A.a,B.b),mul(A.b,B.d)),c:add(mul(A.c,B.a),mul(A.d,B.c)),d:add(mul(A.c,B.b),mul(A.d,B.d))}}
function mconj(A){return{a:conj(A.a),b:conj(A.b),c:conj(A.c),d:conj(A.d)}}
function minv(A){return{a:A.d,b:sc(A.b,-1),c:sc(A.c,-1),d:A.a}}
function mob(M,z){return div(add(mul(M.a,z),M.b),add(mul(M.c,z),M.d))}
function normM(M){const m=Math.max(absv(M.a),absv(M.b),absv(M.c),absv(M.d),1e-30);return{a:sc(M.a,1/m),b:sc(M.b,1/m),c:sc(M.c,1/m),d:sc(M.d,1/m)}}
function iso(M=I(),anti=false){return{M:normM(M),anti}}
function isoApply(A,z){return mob(A.M,A.anti?conj(z):z)}
function isoCompose(A,B){return iso(mm(A.M,A.anti?mconj(B.M):B.M),A.anti!==B.anti)} // A o B
function isoInverse(A){if(!A.anti)return iso(minv(A.M),false);return iso(minv(mconj(A.M)),true)}
function trans(p){return iso({a:C(1),b:sc(p,-1),c:sc(conj(p),-1),d:C(1)},false)} // p -> 0
function cay(z){return mul(C(0,1),div(add(C(1),z),sub(C(1),z)))}
function icy(w){return div(sub(w,C(0,1)),add(w,C(0,1)))}
function hdist(a,b){const q=absv(div(sub(a,b),sub(C(1),mul(conj(b),a))));return 2*Math.atanh(Math.min(.999999999999,Math.max(0,q)))}

// -----------------------------------------------------------------------------
// Coxeter triangle, exact reflection mirrors, orientation-preserving generators
// -----------------------------------------------------------------------------
const pairs=[[1,2],[2,0],[0,1]], angle=n=>n==null?0:Math.PI/n;
function radial(op,a,b){const s=Math.sin(a)*Math.sin(b);if(Math.abs(s)<1e-14)return 1;const ch=(Math.cos(op)+Math.cos(a)*Math.cos(b))/s;if(!Number.isFinite(ch)||ch>1e15)return 1;return Math.tanh(Math.acosh(Math.max(1,ch))/2)}
function baseTriangle(g){
  const a=ord(g.a),b=ord(g.b),c=ord(g.c);
  // For cuspidal triangles use the canonical upper-half-plane realization
  // zeta_1=-exp(-pi i/m1), zeta_2=exp(pi i/m2), zeta_3=infinity and Cayley it
  // to D.  The old radial formula degenerates when two or three angles vanish.
  if(c==null){
    const v1=a?1/a:0,v2=b?1/b:0;
    const z1=C(-Math.cos(Math.PI*v1),Math.sin(Math.PI*v1));
    const z2=C( Math.cos(Math.PI*v2),Math.sin(Math.PI*v2));
    return[icy(z1),icy(z2),C(1,0)]
  }
  const A=angle(a),B=angle(b),G=angle(c),rB=radial(G,A,B),rC=radial(B,A,G);
  return[C(),C(rB,0),C(rC*Math.cos(A),rC*Math.sin(A))]
}
function diameter(u,v){return Math.abs(u.x*v.y-u.y*v.x)<1e-12}
function mirrorData(u,v,opposite){if(diameter(u,v)){const q=absv(u)>1e-10?u:v,th=Math.atan2(q.y,q.x),dir=C(Math.cos(th),Math.sin(th));return{type:0,q:dir,sign:Math.sign(dir.x*opposite.y-dir.y*opposite.x)||1,iso:iso({a:C(Math.cos(2*th),Math.sin(2*th)),b:C(),c:C(),d:C(1)},true)}}
  const ru=(1+abs2(u))/2,rv=(1+abs2(v))/2,det=u.x*v.y-u.y*v.x,cc=C((ru*v.y-u.y*rv)/det,(u.x*rv-ru*v.x)/det),R2=Math.max(0,abs2(cc)-1);const ev=abs2(sub(opposite,cc))-R2;return{type:1,c:cc,R2,sign:Math.sign(ev)||1,iso:iso({a:cc,b:C(-1),c:C(1),d:sc(conj(cc),-1)},true)}}
function mirrorSide(m,z){const e=m.type===0?(m.q.x*z.y-m.q.y*z.x):(abs2(sub(z,m.c))-m.R2);return m.sign*e}
function chooseTriangleSeed(tri,mirrors){
  let best=null,bscore=-Infinity;
  const test=z=>{if(abs2(z)>=.985*.985)return;const ss=mirrors.map(m=>mirrorSide(m,z));if(ss.some(v=>v<-1e-9))return;const score=Math.min(...ss)/(1+absv(z));if(score>bscore){best=z;bscore=score}};
  test(C());
  for(let n=6;n<=18;n+=3)for(let i=1;i<n;i++)for(let j=1;j<n-i;j++){
    const k=n-i-j;if(k<=0)continue;test(sc(add(add(sc(tri[0],i),sc(tri[1],j)),sc(tri[2],k)),1/n))
  }
  if(best)return best;
  let z=sc(add(add(tri[0],tri[1]),tri[2]),1/3);for(let i=0;i<20;i++){if(mirrors.every(m=>mirrorSide(m,z)>=-1e-8))return z;z=sc(z,.82)}return C()
}
function makeSystem(g){const tri=baseTriangle(g),mirrors=pairs.map(([j,k],i)=>mirrorData(tri[j],tri[k],tri[i])),R=mirrors.map(m=>m.iso);const X=isoCompose(R[1],R[2]),Y=isoCompose(R[2],R[0]),Z=isoCompose(R[0],R[1]);const seed=chooseTriangleSeed(tri,mirrors);return{group:g,tri,seed,mirrors,R,G:[X,Y,Z],Gi:[isoInverse(X),isoInverse(Y),isoInverse(Z)]}}
let SYS=makeSystem(current);
function insideCoxeter(z){for(const m of SYS.mirrors){let e=m.type===0?(m.q.x*z.y-m.q.y*z.x):(abs2(sub(z,m.c))-m.R2);if(m.sign*e<-1e-10)return false}return true}
function insideBaseGamma(z){if(insideCoxeter(z))return true;return insideCoxeter(isoApply(SYS.R[0],z))}
function foldToCoxeter(z,max=80){let A=iso();const reflections=[];for(let step=0;step<max;step++){let worst=0,wi=-1;for(let i=0;i<3;i++){const m=SYS.mirrors[i],e=m.type===0?(m.q.x*z.y-m.q.y*z.x):(abs2(sub(z,m.c))-m.R2),v=-m.sign*e;if(v>worst){worst=v;wi=i}}if(wi<0||worst<1e-11)return{z,A,steps:step,reflections};z=isoApply(SYS.R[wi],z);A=isoCompose(SYS.R[wi],A);reflections.push(wi)}return{z,A,steps:max,reflections}}

// Camera maps current world coordinates -> centered view disk.  Re-basing is done only
// by the currently displayed subgroup H (H=Γ when no cover is selected).  Thus the
// background and the highlighted H-fundamental-region tessellation remain the same scene.
let camera=iso();
let euclidShiftX=0,euclidAngle=0;
// q/e use the same camera semantics as WASD/dragging.  In D they rotate the
// hyperbolic view about the disk centre; in H they pan horizontally through the
// tessellation.  Keeping these as camera motions (rather than moving the chart)
// lets rebaseCamera() select the newly displayed fundamental region and avoids a
// translated overlay clip boundary in the upper-half-plane model.
function chartForward(P){
  const g=geom();
  if(model==='half')return C(P.x+euclidShiftX,P.y);
  const c=Math.cos(euclidAngle),sn=Math.sin(euclidAngle),x=P.x-g.cx,y=P.y-g.cy;
  // Screen y points down: this sign convention makes positive angle visibly CCW.
  return C(g.cx+c*x+sn*y,g.cy-sn*x+c*y)
}
function chartInverse(P){
  const g=geom();
  if(model==='half')return C(P.x-euclidShiftX,P.y);
  const c=Math.cos(euclidAngle),sn=Math.sin(euclidAngle),x=P.x-g.cx,y=P.y-g.cy;
  return C(g.cx+c*x-sn*y,g.cy+sn*x+c*y)
}
function resetEuclideanView(){euclidShiftX=0;euclidAngle=0;for(const el of [tiling,fallback,overlay])el.style.transform='';invalidate(true)}
function cameraApply(z){return isoApply(camera,z)}
function cameraInverse(u){return isoApply(isoInverse(camera),u)}
function pairAction(i,j){if(i===j)return 0;if(i===1&&j===2)return 1;if(i===2&&j===1)return-1;if(i===2&&j===0)return 2;if(i===0&&j===2)return-2;if(i===0&&j===1)return 3;if(i===1&&j===0)return-3;return null}
function reflectionWordActions(refs){const r=refs.slice();if(r.length&1)r.push(0);const out=[];for(let i=0;i<r.length;i+=2){const a=pairAction(r[i],r[i+1]);if(a==null)return null;if(a)out.push(a)}return out}
function permStep(R,s,a){if(!R.P)return 0;const j=Math.abs(a)-1;return a>0?R.P[j][s]:R.Pi[j][s]}
function rebaseCamera(){const q=cameraInverse(C()),f=foldToCoxeter(q,160);if(f.steps<=0)return;let gamma=isoInverse(f.A),refs=f.reflections.slice();if(gamma.anti){gamma=isoCompose(gamma,SYS.R[0]);refs.push(0)}const actions=reflectionWordActions(refs);if(!actions)return;const R=currentRegion();let sheet=R.root||0;for(const a of actions)sheet=permStep(R,sheet,a);const rep=R.reps?.[sheet]||iso();const h=isoCompose(gamma,isoInverse(rep));if(h.anti)return;camera=isoCompose(camera,h);camera.anti=false;camera.M=normM(camera.M)}

// -----------------------------------------------------------------------------
// Layout / projection
// -----------------------------------------------------------------------------
function rightInset(){if(!$('sheet').classList.contains('show')||W<=900)return 0;const sw=$('sheet').getBoundingClientRect().width||440;return Math.min(sw,W-260)+20}
function geom(){const availW=Math.max(260,W-rightInset()),m=Math.min(availW,H);if(model==='disk'){const r=Math.max(60,m*.43*zoom);return{cx:availW/2,cy:H/2,r,availW}}const s=Math.max(52,m*.36*zoom);return{cx:availW/2,cy:H/2,scale:s,bottom:H/2+s,availW}}
function projWorld(z){const u=cameraApply(z);return model==='disk'?u:cay(u)}
function screenQBase(q){const g=geom();return model==='disk'?C(g.cx+q.x*g.r,g.cy-q.y*g.r):C(g.cx+q.x*g.scale,g.cy-(q.y-1)*g.scale)}
function screenQ(q){return chartForward(screenQBase(q))}
function screenWorld(z){return screenQ(projWorld(z))}
function viewDiskFromScreen(x,y){const g=geom(),P=chartInverse(C(x,y));if(model==='disk')return C((P.x-g.cx)/g.r,(g.cy-P.y)/g.r);return icy(C((P.x-g.cx)/g.scale,1+(g.cy-P.y)/g.scale))}
function halfPlaneTranslationIso(t){
  // Cayley^{-1} o (w -> w+t) o Cayley: the exact parabolic Mobius motion of H.
  return iso({a:C(-t,2),b:C(t,0),c:C(-t,0),d:C(t,2)},false)
}
function halfPlaneTranslate(t,rebase=true){
  camera=isoCompose(halfPlaneTranslationIso(t),camera);camera.M=normM(camera.M);if(rebase&&!frickeCompare)rebaseCamera();
}
function panPixels(dx,dy){
  const g=geom();let u;
  if(model==='half'){
    // Restore the old plain Mobius WASD/drag camera: choose the nearby point
    // in the current H chart that should become the new centre, then send it
    // to 0 in the disk.  No screen-space translation is involved.
    u=icy(C(-dx/g.scale,1+dy/g.scale));
  }else u=C(-dx/g.r,dy/g.r);
  const r=absv(u);if(r>.42)u=sc(u,.42/r);camera=isoCompose(trans(u),camera);
  if(!frickeCompare)rebaseCamera();invalidate(true)
}
function zoomBy(f){zoom=Math.max(.45,Math.min(4.2,zoom*f));invalidate(true)}
function rotateCamera(theta){const R=iso({a:C(Math.cos(theta),Math.sin(theta)),b:C(),c:C(),d:C(1)},false);camera=isoCompose(R,camera);camera.M=normM(camera.M);invalidate(true)}
function auxiliaryMotion(sign,dt){
  // q/e are camera Mobius motions just like WASD.  In D they use the compact
  // rotation subgroup; in H they use the distinguished parabolic w -> w+t.
  if(model==='disk')rotateCamera(-sign*.95*dt);
  else{const g=geom();halfPlaneTranslate(-sign*210*dt/g.scale,!frickeCompare);invalidate(true)}
}

// -----------------------------------------------------------------------------
// GPU implicit Coxeter tessellation
// ----------------------------------------------------------------------------
let gl=null,prog=null,loc={},gpuOK=false,gpuFast=false;
const VS=`#version 300 es\nin vec2 p;void main(){gl_Position=vec4(p,0.,1.);}`;
const FS=`#version 300 es
precision highp float;
out vec4 outColor;
uniform vec2 uRes; uniform float uDpr; uniform int uModel; uniform vec4 uGeom; uniform vec2 uEuclid; uniform int uSteps; uniform vec4 uCamA; uniform vec4 uCamB;
uniform vec4 uBgA; uniform vec4 uBgB; uniform int uBgAnti;
uniform vec4 uM0; uniform vec4 uM1; uniform vec4 uM2; uniform vec3 uType; uniform vec3 uSign;
vec2 cmul(vec2 a,vec2 b){return vec2(a.x*b.x-a.y*b.y,a.x*b.y+a.y*b.x);} vec2 cconj(vec2 a){return vec2(a.x,-a.y);} float cabs2(vec2 a){return dot(a,a);} vec2 cdiv(vec2 a,vec2 b){float d=max(dot(b,b),1e-25);return vec2(dot(a,b),a.y*b.x-a.x*b.y)/d;}
vec2 mob(vec2 a,vec2 b,vec2 c,vec2 d,vec2 z){return cdiv(cmul(a,z)+b,cmul(c,z)+d);} 
vec2 camInv(vec2 z){vec2 a=uCamA.xy,b=uCamA.zw,c=uCamB.xy,d=uCamB.zw;return cdiv(cmul(d,z)-b,-cmul(c,z)+a);} 
vec2 applyBg(vec2 z){vec2 a=uBgA.xy,b=uBgA.zw,c=uBgB.xy,d=uBgB.zw;if(uBgAnti==1)z=cconj(z);return mob(a,b,c,d,z);} 
vec2 icy(vec2 w){return cdiv(w-vec2(0,1),w+vec2(0,1));}
float evalSide(int i,vec2 z){vec4 m=i==0?uM0:(i==1?uM1:uM2);if((i==0?uType.x:(i==1?uType.y:uType.z))<.5){return m.x*z.y-m.y*z.x;}vec2 c=m.xy;return cabs2(z-c)-m.z;}
vec2 refl(int i,vec2 z){vec4 m=i==0?uM0:(i==1?uM1:uM2);if((i==0?uType.x:(i==1?uType.y:uType.z))<.5){vec2 e=vec2(m.x,m.y);float t=2.*atan(e.y,e.x);vec2 q=vec2(cos(t),sin(t));return cmul(q,cconj(z));}vec2 c=m.xy,d=z-c;return c+m.z*d/max(dot(d,d),1e-25);}
void main(){vec2 fc=vec2(gl_FragCoord.x/uDpr,uRes.y-gl_FragCoord.y/uDpr);float cx=uGeom.x,cy=uGeom.y,s=uGeom.z,bottom=uGeom.w;if(uModel==0){float a=uEuclid.y,cc=cos(a),ss=sin(a);vec2 d=fc-vec2(cx,cy);fc=vec2(cx,cy)+vec2(cc*d.x-ss*d.y,ss*d.x+cc*d.y);}else{fc.x-=uEuclid.x;}vec2 u;bool inside=true;if(uModel==0){u=vec2((fc.x-cx)/s,(cy-fc.y)/s);inside=dot(u,u)<1.;}else{vec2 w=vec2((fc.x-cx)/s,1.+(cy-fc.y)/s);inside=(w.y>0.);u=icy(w);}if(!inside){outColor=vec4(0.969,0.969,0.957,1);return;}vec2 z=applyBg(camInv(u));float md=1e9,parity=0.;for(int k=0;k<192;k++){if(k>=uSteps)break;float e0=uSign.x*evalSide(0,z),e1=uSign.y*evalSide(1,z),e2=uSign.z*evalSide(2,z);float m=min(e0,min(e1,e2));if(m>=-1e-7){md=min(abs(e0),min(abs(e1),abs(e2)));break;}if(e0<=e1&&e0<=e2)z=refl(0,z);else if(e1<=e0&&e1<=e2)z=refl(1,z);else z=refl(2,z);parity=1.-parity;}float fw=max(fwidth(md),1e-7);float alpha=1.-smoothstep(.25*fw,1.25*fw,md);vec3 bg0=vec3(.979,.979,.969),bg1=vec3(.944,.945,.938),bg=mix(bg0,bg1,.46*parity),line=vec3(.07,.07,.063);outColor=vec4(mix(bg,line,.105*alpha),1.);}`;
function compile(type,src){const sh=gl.createShader(type);gl.shaderSource(sh,src);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(sh));return sh}
function initGL(){try{gl=tiling.getContext('webgl2',{antialias:false,alpha:false,preserveDrawingBuffer:false,powerPreference:'high-performance'});if(!gl)return false;prog=gl.createProgram();gl.attachShader(prog,compile(gl.VERTEX_SHADER,VS));gl.attachShader(prog,compile(gl.FRAGMENT_SHADER,FS));gl.linkProgram(prog);if(!gl.getProgramParameter(prog,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(prog));gl.useProgram(prog);const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,3,-1,-1,3]),gl.STATIC_DRAW);const ap=gl.getAttribLocation(prog,'p');gl.enableVertexAttribArray(ap);gl.vertexAttribPointer(ap,2,gl.FLOAT,false,0,0);for(const n of ['uRes','uDpr','uModel','uGeom','uEuclid','uSteps','uCamA','uCamB','uBgA','uBgB','uBgAnti','uM0','uM1','uM2','uType','uSign'])loc[n]=gl.getUniformLocation(prog,n);gpuOK=true;return true}catch(e){console.error(e);$('renderer-note').style.display='block';$('renderer-note').textContent='2D';return false}}
function setVec4(L,a,b,c,d){gl.uniform4f(L,a,b,c,d)}
function backgroundAnchor(){const q=cameraInverse(C()),f=foldToCoxeter(q,192);return f.A||iso()}
function renderGL(){if(!gpuOK)return;const scale=gpuFast?.70:1,gdpr=Math.min(gpuFast?.95:1.6,devicePixelRatio||1)*scale,ww=Math.max(1,Math.round(W*gdpr)),hh=Math.max(1,Math.round(H*gdpr));if(tiling.width!==ww||tiling.height!==hh){tiling.width=ww;tiling.height=hh}gl.viewport(0,0,ww,hh);gl.useProgram(prog);gl.uniform2f(loc.uRes,W,H);gl.uniform1f(loc.uDpr,gdpr);gl.uniform1i(loc.uModel,model==='disk'?0:1);const g=geom();gl.uniform4f(loc.uGeom,g.cx,g.cy,model==='disk'?g.r:g.scale,model==='disk'?0:g.bottom);gl.uniform2f(loc.uEuclid,euclidShiftX,euclidAngle);gl.uniform1i(loc.uSteps,gpuFast?72:144);const M=camera.M;setVec4(loc.uCamA,M.a.x,M.a.y,M.b.x,M.b.y);setVec4(loc.uCamB,M.c.x,M.c.y,M.d.x,M.d.y);const BA=backgroundAnchor(),BM=BA.M;setVec4(loc.uBgA,BM.a.x,BM.a.y,BM.b.x,BM.b.y);setVec4(loc.uBgB,BM.c.x,BM.c.y,BM.d.x,BM.d.y);gl.uniform1i(loc.uBgAnti,BA.anti?1:0);const md=SYS.mirrors.map(m=>m.type===0?[m.q.x,m.q.y,0,0]:[m.c.x,m.c.y,m.R2,0]);md.forEach((v,i)=>gl.uniform4f(loc['uM'+i],...v));gl.uniform3f(loc.uType,...SYS.mirrors.map(m=>m.type));gl.uniform3f(loc.uSign,...SYS.mirrors.map(m=>m.sign));gl.drawArrays(gl.TRIANGLES,0,3);bgDirty=false}

// -----------------------------------------------------------------------------
// CPU arc drawing for the selected exact fundamental region
// -----------------------------------------------------------------------------
function geodesicPoint(a,b,t=.5){const M=trans(a),w=isoApply(M,b),r=absv(w);if(r<1e-12)return a;const rr=Math.tanh(t*Math.atanh(Math.min(.999999999999,r)));return isoApply(isoInverse(M),sc(w,rr/r))}
function traceDisk(a,b,first=true){const A=screenQ(a),B=screenQ(b);if(diameter(a,b)){if(first)ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);return}const ru=(1+abs2(a))/2,rv=(1+abs2(b))/2,det=a.x*b.y-a.y*b.x;if(Math.abs(det)<1e-13){if(first)ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);return}const cc=C((ru*b.y-a.y*rv)/det,(a.x*rv-ru*b.x)/det),R=Math.sqrt(Math.max(0,abs2(cc)-1)),g=geom(),cs=screenQ(cc),r=R*g.r,t1=Math.atan2(A.y-cs.y,A.x-cs.x),t2=Math.atan2(B.y-cs.y,B.x-cs.x);let dd=t2-t1;while(dd<=-Math.PI)dd+=2*Math.PI;while(dd>Math.PI)dd-=2*Math.PI;if(first)ctx.moveTo(A.x,A.y);ctx.arc(cs.x,cs.y,r,t1,t1+dd,dd<0)}
function traceHalf(a,b,first=true){
  const ia=abs2(sub(C(1),a))<1e-11,ib=abs2(sub(C(1),b))<1e-11,g=geom();
  if(ia&&ib)return;
  if(ia||ib){const w=cay(ia?b:a);if(!Number.isFinite(w.x+w.y))return;const F0=screenQBase(w),T0=C(F0.x,-Math.max(2.5*H,900)),F=chartForward(F0),Top=chartForward(T0),A=ia?Top:F,B=ia?F:Top;if(first)ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);return}
  a=cay(a);b=cay(b);if(!Number.isFinite(a.x+a.y+b.x+b.y))return;const A=screenQ(a),B=screenQ(b);
  if(Math.abs(a.x-b.x)<1e-9){if(first)ctx.moveTo(A.x,A.y);ctx.lineTo(B.x,B.y);return}
  const cx=(a.x*a.x+a.y*a.y-b.x*b.x-b.y*b.y)/(2*(a.x-b.x)),R=Math.hypot(a.x-cx,a.y),cs=screenQ(C(cx,0)),r=R*g.scale,t1=Math.atan2(A.y-cs.y,A.x-cs.x),t2=Math.atan2(B.y-cs.y,B.x-cs.x);let dd=t2-t1;while(dd<=-Math.PI)dd+=2*Math.PI;while(dd>Math.PI)dd-=2*Math.PI;if(first)ctx.moveTo(A.x,A.y);ctx.arc(cs.x,cs.y,r,t1,t1+dd,dd<0)
}
function traceWorld(a,b,first=true){const u=cameraApply(a),v=cameraApply(b);if(model==='disk')traceDisk(u,v,first);else traceHalf(u,v,first)}
function pathTriangle(v){ctx.beginPath();traceWorld(v[0],v[1],true);traceWorld(v[1],v[2],false);traceWorld(v[2],v[0],false);ctx.closePath()}
function gammaCellComplex(R){
  if(R.gammaCells)return R.gammaCells;
  const B=baseRegion(),edgeMap=new Map(),cells=[];
  for(let i=0;i<R.reps.length;i++){
    const edges=[];
    for(const e of B.boundary){const a=isoApply(R.reps[i],e.a),b=isoApply(R.reps[i],e.b),k=edgeKey(a,b);edges.push({a,b,key:k});if(!edgeMap.has(k))edgeMap.set(k,{a,b,owners:[]});const rec=edgeMap.get(k);if(!rec.owners.includes(i))rec.owners.push(i)}
    cells.push(edges)
  }
  const adj=Array.from({length:R.degree},()=>new Set());
  for(const e of edgeMap.values())if(e.owners.length===2){const[a,b]=e.owners;adj[a].add(b);adj[b].add(a)}
  R.gammaCells={cells,edgeMap,adj,internal:[...edgeMap.values()].filter(e=>e.owners.length===2),all:[...edgeMap.values()]};return R.gammaCells
}
function sheetColoring(R){
  if(R.sheetColors)return R.sheetColors;const adj=gammaCellComplex(R).adj,n=R.degree,col=new Array(n).fill(-1);let left=n;
  while(left){let v=-1,bestSat=-1,bestDeg=-1;for(let i=0;i<n;i++)if(col[i]<0){const sat=new Set([...adj[i]].map(j=>col[j]).filter(c=>c>=0)).size,d=adj[i].size;if(sat>bestSat||(sat===bestSat&&d>bestDeg)){v=i;bestSat=sat;bestDeg=d}}const used=new Set([...adj[v]].map(j=>col[j]).filter(c=>c>=0));let c=0;while(used.has(c))c++;col[v]=c;left--}
  R.sheetColors=col;return col
}
function gammaCellEdges(R,internalOnly=true){const C=gammaCellComplex(R);return internalOnly?C.internal:C.all}
function fillGammaCells(R){
  const colors=sheetColoring(R),palette=['rgba(252,252,249,.62)','rgba(214,216,212,.38)','rgba(239,240,236,.48)','rgba(202,205,201,.31)','rgba(229,231,226,.40)'];
  ctx.save();for(let i=0;i<R.reps.length;i++){ctx.fillStyle=palette[colors[i]%palette.length];for(const A of [R.reps[i],isoCompose(R.reps[i],SYS.R[0])]){pathTriangle(transformTri(A));ctx.fill()}}ctx.restore()
}
function transformTri(A){return SYS.tri.map(z=>isoApply(A,z))}
function edgeKey(a,b){const f=p=>`${Math.round(p.x*1e8)},${Math.round(p.y*1e8)}`,A=f(a),B=f(b);return A<B?A+'|'+B:B+'|'+A}
function permInv(p){const q=new Array(p.length);p.forEach((v,i)=>q[v]=i);return q}
function actionIso(a){const j=Math.abs(a)-1;return a>0?SYS.G[j]:SYS.Gi[j]}
function actionStep(P,Pi,s,a){const j=Math.abs(a)-1;return a>0?P[j][s]:Pi[j][s]}
function edgeCodeForAction(a){return a===2?'C2':a===-2?'R2':a===3?'R1':a===-3?'C1':null}
function sideGeometry(rep,code){let A=rep,i=1;if(code==='C1'){i=1}else if(code==='C2'){i=2}else if(code==='R1'){A=isoCompose(rep,SYS.R[0]);i=1}else if(code==='R2'){A=isoCompose(rep,SYS.R[0]);i=2}else return null;const v=transformTri(A),[j,k]=pairs[i];return{a:v[j],b:v[k]}}
function baseGammaCenter(){const p=SYS.seed,q=isoApply(SYS.R[0],SYS.seed);return lorentzAverage([p,q])}
function regionCenterFromReps(reps){const c0=baseGammaCenter(),pts=reps.map(A=>isoApply(A,c0));return lorentzAverage(pts)}
function chamberCenter(A){return isoApply(A,SYS.seed)}
function isoProbe(A,T){const Q=isoCompose(T,A),refs=[SYS.seed,C(.071,.019),C(-.033,.084)];return refs.map(z=>isoApply(Q,z))}
function probeKey(P,anti){const q=x=>Math.round(x*2e6);return (anti?'a:':'h:')+P.map(z=>`${q(z.x)},${q(z.y)}`).join('|')}
function probesClose(A,B,eps=2e-6){if(A.anti!==B.anti)return false;for(let i=0;i<A.P.length;i++)if(absv(sub(A.P[i],B.P[i]))>eps)return false;return true}
function regionVisualCenter(boundary,fallback){if(!boundary?.length)return fallback;const mids=[];for(const e of boundary){try{const m=geodesicPoint(e.a,e.b,.5);if(Number.isFinite(m.x+m.y)&&abs2(m)<1)mids.push(m)}catch{}}return mids.length?lorentzAverage(mids):fallback}
function sideHalfMirror(code){return code==='C1'?[0,1]:code==='C2'?[0,2]:code==='R1'?[1,1]:code==='R2'?[1,2]:null}
function buildChamberTopology(chambers,center){
  // Final boundary extraction is geometric, not tied to the Schreier tree.
  // A chamber side is suppressed exactly when its reflected chamber is also
  // present in the chosen union.  This removes every internal edge, including
  // cycles created by group relations.
  const T=trans(center),probes=[],bucket=new Map();
  for(let j=0;j<chambers.length;j++){
    const P=isoProbe(chambers[j].A,T),o={P,anti:chambers[j].A.anti};
    probes.push(o);const k=probeKey(P,o.anti);if(!bucket.has(k))bucket.set(k,[]);bucket.get(k).push(j)
  }
  const lookup=A=>{
    const P=isoProbe(A,T),o={P,anti:A.anti},k=probeKey(P,o.anti);
    for(const j of bucket.get(k)||[])if(probesClose(o,probes[j],3e-6))return j;
    let best=-1,bd=Infinity;
    for(let j=0;j<probes.length;j++){
      if(probes[j].anti!==o.anti)continue;let d=0;
      for(let r=0;r<o.P.length;r++)d=Math.max(d,absv(sub(o.P[r],probes[j].P[r])));
      if(d<bd){bd=d;best=j}
    }
    return bd<3e-5?best:-1
  };
  const adjacency=Array.from({length:chambers.length},()=>[-1,-1,-1]),boundary=[];
  for(let j=0;j<chambers.length;j++){
    const A=chambers[j].A,v=transformTri(A);
    for(let i=0;i<3;i++){
      const n=lookup(isoCompose(A,SYS.R[i]));adjacency[j][i]=n;
      if(n<0){const [u,w]=pairs[i];boundary.push({a:v[u],b:v[w],chamber:j,mirror:i})}
    }
  }
  return{adjacency,boundary}
}
function isoClose(A,B,eps=4e-6){
  if(A.anti!==B.anti)return false;
  const refs=[SYS.seed,C(.071,.019),C(-.033,.084)];
  for(const z of refs)if(absv(sub(isoApply(A,z),isoApply(B,z)))>eps)return false;
  return true
}
function candidateSharedSides(t,A,reps,P,Pi){
  let shared=0;for(const a of [2,-2,3,-3]){
    const q=actionStep(P,Pi,t,a);if(!reps[q])continue;
    if(isoClose(isoCompose(A,actionIso(a)),reps[q]))shared++
  }return shared
}
function kleinPoint(p){const d=1+abs2(p);return C(2*p.x/d,2*p.y/d)}
function boundaryConvexityDefect(boundary){
  // Hyperbolic geodesics are Euclidean lines in the Klein disk.  A boundary
  // vertex strictly inside the Euclidean convex hull is therefore a genuine
  // hyperbolic convexity defect; collinear subdivision points are harmless.
  const mp=new Map(),key=p=>`${Math.round(p.x*1e8)},${Math.round(p.y*1e8)}`;
  for(const e of boundary){mp.set(key(e.a),kleinPoint(e.a));mp.set(key(e.b),kleinPoint(e.b))}
  const pts=[...mp.values()];if(pts.length<4)return 0;
  const q=pts.slice().sort((a,b)=>a.x-b.x||a.y-b.y),cross=(o,a,b)=>(a.x-o.x)*(b.y-o.y)-(a.y-o.y)*(b.x-o.x);
  const lo=[];for(const p of q){while(lo.length>=2&&cross(lo[lo.length-2],lo[lo.length-1],p)<=1e-10)lo.pop();lo.push(p)}
  const up=[];for(let i=q.length-1;i>=0;i--){const p=q[i];while(up.length>=2&&cross(up[up.length-2],up[up.length-1],p)<=1e-10)up.pop();up.push(p)}
  const hull=lo.slice(0,-1).concat(up.slice(0,-1));if(hull.length<3)return 0;
  const onHull=p=>{for(let i=0;i<hull.length;i++){const a=hull[i],b=hull[(i+1)%hull.length],cr=Math.abs(cross(a,b,p));if(cr>2e-7)continue;const dot=(p.x-a.x)*(p.x-b.x)+(p.y-a.y)*(p.y-b.y);if(dot<=2e-7)return true}return false};
  let defect=0;for(const p of pts)if(!onHull(p))defect++;return defect
}
function candidateGeometry(A,reps){
  const rr=reps.filter(Boolean).concat([A]),ch=[];for(const R of rr){ch.push({A:R});ch.push({A:isoCompose(R,SYS.R[0])})}
  const c=lorentzAverage(ch.map(z=>chamberCenter(z.A))),top=buildChamberTopology(ch,c);
  return{defect:boundaryConvexityDefect(top.boundary),boundary:top.boundary.length}
}
function geometricCornerCount(chambers,boundary){
  const pts=new Map(),ms=[ord(SYS.group.a),ord(SYS.group.b),ord(SYS.group.c)];for(const e of boundary||[]){for(const z of [e.a,e.b]){const k=pointKey(z);if(!pts.has(k))pts.set(k,{z,angle:0,ideal:abs2(z)>.999999999})}}
  for(const ch of chambers||[]){const v=transformTri(ch.A);for(let i=0;i<3;i++){const k=pointKey(v[i]),q=pts.get(k);if(q&&ms[i]!=null)q.angle+=Math.PI/ms[i]}}
  let n=0;for(const q of pts.values())if(q.ideal||Math.abs(q.angle-Math.PI)>2e-5)n++;return n
}
function repSetMetrics(reps,targetCorners=0){
  const chambers=[];for(let s0=0;s0<reps.length;s0++){chambers.push({A:reps[s0],sheet:s0,half:0});chambers.push({A:isoCompose(reps[s0],SYS.R[0]),sheet:s0,half:1})}
  const c=lorentzAverage(chambers.map(ch=>chamberCenter(ch.A))),top=buildChamberTopology(chambers,c);const seen=new Set(),q=[];let components=0;
  for(let i=0;i<chambers.length;i++)if(!seen.has(i)){components++;seen.add(i);q.length=0;q.push(i);for(let qi=0;qi<q.length;qi++){for(const j of top.adjacency[q[qi]])if(j>=0&&!seen.has(j)){seen.add(j);q.push(j)}}}
  const center=regionVisualCenter(top.boundary,c),radius=Math.max(0,...reps.map(A=>hdist(center,isoApply(A,baseGammaCenter())))),defect=boundaryConvexityDefect(top.boundary),corners=geometricCornerCount(chambers,top.boundary),cornerPenalty=targetCorners?Math.abs(corners-targetCorners):0;
  return{components,defect,boundary:top.boundary.length,radius,corners,cornerPenalty,score:components*1e6+defect*48+cornerPenalty*18+top.boundary.length*.20+radius*1.35}
}
function refineRepresentatives(reps,P,Pi,targetCorners=0){
  const n=reps.length;if(n>24)return reps;let cur=repSetMetrics(reps,targetCorners),R=reps.slice(),passes=0;
  while(passes++<10&&(cur.defect>0||cur.cornerPenalty>0)){let best=null;for(let t=1;t<n;t++)for(const a of [2,-2,3,-3]){const s0=actionStep(P,Pi,t,-a);if(s0===t||!R[s0])continue;const A=isoCompose(R[s0],actionIso(a));if(isoClose(A,R[t]))continue;let duplicate=false;for(let j=0;j<n;j++)if(j!==t&&isoClose(A,R[j])){duplicate=true;break}if(duplicate)continue;const trial=R.slice();trial[t]=A;const m=repSetMetrics(trial,targetCorners);if(m.components!==1)continue;if(m.score<cur.score-1e-7&&(!best||m.score<best.m.score))best={t,A,m}}
    if(!best)break;R[best.t]=best.A;cur=best.m
  }
  return R
}
function regionFromTriple(triple,targetCorners=0){
  const P=triple.map(a=>a.map(x=>x-1)),Pi=P.map(permInv),n=P[0].length,root=0;
  const actions=[2,-2,3,-3],reps=new Array(n),parent=new Int32Array(n),parentAction=new Int8Array(n);parent.fill(-1);reps[root]=iso();
  const c0=baseGammaCenter();let assigned=1;
  // Grow one domain at a time.  Connectivity is a hard constraint.  Among all
  // boundary lifts choose the one sharing the most already-present sides, then
  // the one closest to the Lorentz barycenter.  This strongly suppresses long
  // Schreier-tree tendrils and usually produces a near-convex connected union.
  while(assigned<n){
    const selected=[];for(let i=0;i<n;i++)if(reps[i])selected.push(reps[i]);
    const bary=regionCenterFromReps(selected),cand=[];
    for(let s0=0;s0<n;s0++)if(reps[s0])for(const a of actions){
      const t=actionStep(P,Pi,s0,a);if(reps[t])continue;
      const A=isoCompose(reps[s0],actionIso(a)),p=isoApply(A,c0),shared=candidateSharedSides(t,A,reps,P,Pi);
      const compact=hdist(bary,p),radial=hdist(c0,p);
      cand.push({t,A,p:s0,a,shared,score:6.0*shared-compact-.08*radial})
    }
    if(!cand.length)throw new Error(`coset action is not generated by Y,Z (${assigned}/${n})`);
    cand.sort((u,v)=>v.score-u.score||v.shared-u.shared);
    // Refine only the best few frontier lifts.  This keeps interaction fast
    // while explicitly preferring hyperbolically convex unions in Klein form.
    for(const q of cand.slice(0,Math.min(10,cand.length))){const z=candidateGeometry(q.A,reps);q.defect=z.defect;q.geomScore=q.score-6.5*z.defect-.035*z.boundary}
    const probe=cand.slice(0,Math.min(10,cand.length)).sort((u,v)=>v.geomScore-u.geomScore||u.defect-v.defect||v.shared-u.shared),best=probe[0]||cand[0];reps[best.t]=best.A;parent[best.t]=best.p;parentAction[best.t]=best.a;assigned++
  }
  const refined=refineRepresentatives(reps,P,Pi,targetCorners);for(let i=0;i<n;i++)reps[i]=refined[i];
  const chambers=[];for(let s0=0;s0<n;s0++){
    chambers.push({A:reps[s0],sheet:s0,half:0});
    chambers.push({A:isoCompose(reps[s0],SYS.R[0]),sheet:s0,half:1})
  }
  const center0=lorentzAverage(chambers.map(ch=>chamberCenter(ch.A))),topology=buildChamberTopology(chambers,center0);
  const center=regionVisualCenter(topology.boundary,center0);
  const cosetAdj=Array.from({length:n},(_,s0)=>({Y:P[1][s0],Yi:Pi[1][s0],Z:P[2][s0],Zi:Pi[2][s0]}));
  return{degree:n,root,P,Pi,reps,parent,parentAction,chambers,boundary:topology.boundary,chamberAdj:topology.adjacency,cosetAdj,center,neighbors:[],optimized:true,targetCorners,geometricCorners:geometricCornerCount(chambers,topology.boundary)}
}
function regionWithHalfMirror(R,h=0){
  h=Math.max(0,Math.min(2,h|0));const chambers=[];for(let s0=0;s0<R.reps.length;s0++){chambers.push({A:R.reps[s0],sheet:s0,half:0});chambers.push({A:isoCompose(R.reps[s0],SYS.R[h]),sheet:s0,half:1})}
  const center0=lorentzAverage(chambers.map(ch=>chamberCenter(ch.A))),topology=buildChamberTopology(chambers,center0),center=regionVisualCenter(topology.boundary,center0);return{...R,chambers,boundary:topology.boundary,chamberAdj:topology.adjacency,center,halfMirror:h,geometricCorners:geometricCornerCount(chambers,topology.boundary),pairingSides:null,boundaryVertexClasses:null,vertexData:null,gammaCells:null,sheetColors:null}
}
function bestHalfMirrorRegion(R,targetCorners=0){
  let best=null;for(let h=0;h<3;h++){const Q=regionWithHalfMirror(R,h),defect=boundaryConvexityDefect(Q.boundary),pen=targetCorners?Math.abs(Q.geometricCorners-targetCorners):0,score=defect*1000+pen*70+Q.geometricCorners*3+Q.boundary.length*.2;const z={Q,score,defect};if(!best||z.score<best.score)best=z}return best?.Q||R
}
function baseRegion(){
  const reps=[iso()],chambers=[{A:iso(),sheet:0,half:0},{A:SYS.R[0],sheet:0,half:1}],center0=lorentzAverage(chambers.map(ch=>chamberCenter(ch.A))),topology=buildChamberTopology(chambers,center0);
  return{degree:1,root:0,P:null,Pi:null,reps,chambers,boundary:topology.boundary,chamberAdj:topology.adjacency,cosetAdj:[{Y:0,Yi:0,Z:0,Zi:0}],center:center0,neighbors:[]}
}
function lorentzAverage(points){let X=0,Y=0,Z=0,npts=0;for(const p of points){const d=Math.max(1e-12,1-abs2(p));X+=(1+abs2(p))/d;Y+=2*p.x/d;Z+=2*p.y/d;npts++}if(!npts)return C();const n=Math.sqrt(Math.max(1e-20,X*X-Y*Y-Z*Z));X/=n;Y/=n;Z/=n;return C(Y/(X+1),Z/(X+1))}
const REGION_CACHE_MAX=40;let regionCache=new Map();
function coverTriple(r){if(!r)return null;if(r.triple)return r.triple;if(r.modular)return modularPerm[r.label]?.triple||null;const e=r.row?.exact_drv_permutation_data,rr=e?.representatives?.[r.rep||0];return rr?.triple||null}
function targetRegionCorners(r){if(!r||Number(r.genus??r.row?.upstairs_orbifold_signature?.coarse_genus??0)!==0)return 0;let e=r.elliptic_orders?.length;if(e==null)e=r.row?.upstairs_orbifold_signature?.elliptic_orders_with_multiplicity?.length;let c=r.cusp_count;if(c==null&&Array.isArray(r.cusp_widths))c=r.cusp_widths.length;if(c==null&&r.modular)c=modularCuspData(r.row).count;if(c==null)c=0;const n=Number(e||0)+Number(c||0);return n===3||n===4?n:0}
function currentRegion(){const key=selectedCover?.key||'base';if(regionCache.has(key)){const R=regionCache.get(key);regionCache.delete(key);regionCache.set(key,R);return R}const t=coverTriple(selectedCover),R=t?regionFromTriple(t,targetRegionCorners(selectedCover)):baseRegion();regionCache.set(key,R);while(regionCache.size>REGION_CACHE_MAX){const k=regionCache.keys().next().value;if(k==='base'){const b=regionCache.get(k);regionCache.delete(k);regionCache.set(k,b);continue}regionCache.delete(k)}return R}
function transformedBoundary(R,A){return R.boundary.map(e=>({a:isoApply(A,e.a),b:isoApply(A,e.b)}))}
function strokeBoundary(boundary,{width=2,stroke='rgba(38,44,58,.96)',dash=[]}={}){
  ctx.save();ctx.lineCap='round';ctx.lineJoin='round';ctx.beginPath();for(const e of boundary)traceWorld(e.a,e.b,true);ctx.strokeStyle=stroke;ctx.lineWidth=width*viewSettings.edgeSize*viewSettings.boundarySize;ctx.setLineDash(dash);ctx.stroke();ctx.restore()
}
function pairHue(n){return (12+n*137.508)%360}
function pairColor(n){const h=pairHue(n),l=34+(n%3)*5;return `hsl(${h.toFixed(1)} 72% ${l}%)`}
function pairColorLight(n){const h=pairHue(n);return `hsl(${h.toFixed(1)} 70% 72%)`}
function pairColorDark(n){const h=pairHue(n);return `hsl(${h.toFixed(1)} 78% 24%)`}
function mappedEdgeMatch(A,e,f){
  const aa=isoApply(A,e.a),bb=isoApply(A,e.b),direct=absv(sub(aa,f.a))+absv(sub(bb,f.b)),reverse=absv(sub(aa,f.b))+absv(sub(bb,f.a));
  return{score:Math.min(direct,reverse),direct:direct<=reverse,directError:direct,reverseError:reverse}
}
function boundaryPairingSides(R){
  if(R.pairingSides)return R.pairingSides;
  const out=[],pairIds=new Map(),groups=new Map(),boundaryKeys=new Set((R.boundary||[]).map(e=>edgeKey(e.a,e.b)));
  for(let s0=0;s0<R.reps.length;s0++)for(const a of [2,-2,3,-3]){
    const t=permStep(R,s0,a),next=isoCompose(R.reps[s0],actionIso(a));
    if(isoClose(next,R.reps[t]))continue;
    const e=sideGeometry(R.reps[s0],edgeCodeForAction(a));if(!e||!boundaryKeys.has(edgeKey(e.a,e.b)))continue;
    const u=`${s0}:${a}`,v=`${t}:${-a}`,key=u<v?u+'|'+v:v+'|'+u;
    if(!pairIds.has(key))pairIds.set(key,pairIds.size);
    const rec={...e,sheet:s0,action:a,target:t,pair:pairIds.get(key),pairKey:key,arrowForward:null,pairMapError:Infinity};out.push(rec);
    if(!groups.has(key))groups.set(key,[]);groups.get(key).push(rec)
  }
  // The arrow convention is defined by the actual subgroup side-pairing map,
  // not by the sign of the Schreier generator.  If r_s g_a = h r_t, then
  // h^{-1}=r_t g_a^{-1} r_s^{-1} maps the boundary side at (s,a) to the
  // paired side at (t,-a).  We orient one side a→b and transport that
  // orientation through h^{-1}; this is invariant under how the side was
  // discovered in the Schreier graph.
  let maxError=0,invalid=0;
  for(const sides of groups.values()){
    sides.sort((x,y)=>x.sheet-y.sheet||x.action-y.action);const src=sides[0];src.arrowForward=true;
    const tgt=sides.find(q=>q!==src&&q.sheet===src.target&&q.action===-src.action)||sides.find(q=>q!==src);
    if(!tgt){src.arrowForward=null;invalid++;continue}
    const phi=isoCompose(R.reps[src.target],isoCompose(actionIso(-src.action),isoInverse(R.reps[src.sheet]))),m=mappedEdgeMatch(phi,src,tgt);
    src.pairMap=phi;src.pairMapError=m.score;tgt.pairMap=isoInverse(phi);tgt.pairMapError=m.score;tgt.arrowForward=m.direct;maxError=Math.max(maxError,m.score);
    for(const q of sides)if(q!==src&&q!==tgt){const mm=mappedEdgeMatch(phi,src,q);q.arrowForward=mm.direct;q.pairMapError=mm.score;maxError=Math.max(maxError,mm.score)}
    if(m.score>2.5e-4){src.arrowForward=tgt.arrowForward=null;invalid++}
  }
  R.pairingSides=out;R.pairingCount=pairIds.size;R.pairingAudit={pairs:pairIds.size,maxEndpointError:maxError,invalid};
  if(invalid)console.warn('side-pairing orientation audit',R.pairingAudit);
  return out
}
function pairLetter(n){const A='abcdefghijklmnopqrstuvwxyz';return n<A.length?A[n]:`a_{${n+1}}`}
function drawPairingArrow(e,pair){
  if(e.arrowForward==null)return;
  let p1=screenWorld(safeGeodesicMid(e.a,e.b,.40)),p2=screenWorld(safeGeodesicMid(e.a,e.b,.60));if(e.arrowForward===false)[p1,p2]=[p2,p1];
  const dx=p2.x-p1.x,dy=p2.y-p1.y,L=Math.hypot(dx,dy);if(!Number.isFinite(L)||L<4)return;const ux=dx/L,uy=dy/L,nx=-uy,ny=ux,h=11,w=6.8;
  ctx.beginPath();ctx.moveTo(p2.x,p2.y);ctx.lineTo(p2.x-h*ux+w*nx,p2.y-h*uy+w*ny);ctx.lineTo(p2.x-h*ux-w*nx,p2.y-h*uy-w*ny);ctx.closePath();ctx.fillStyle=pairColorDark(pair);ctx.fill()
}
function drawPairings(R,{suffix='',alpha=.94}={}){
  if(!viewSettings.pairings)return;const sides=boundaryPairingSides(R);if(!sides.length||sides.length>72)return;ctx.save();ctx.lineCap='round';ctx.lineJoin='round';ctx.font=`600 ${11*viewSettings.edgeSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;ctx.textAlign='center';ctx.textBaseline='middle';
  for(const e of sides){const color=pairColor(e.pair);ctx.globalAlpha=alpha;strokeBoundary([e],{width:2.25,stroke:color});if(viewSettings.arrows)drawPairingArrow(e,e.pair);if(!viewSettings.edgeLabels)continue;const m=screenWorld(safeGeodesicMid(e.a,e.b,.5)),a=screenWorld(safeGeodesicMid(e.a,e.b,.46)),b=screenWorld(safeGeodesicMid(e.a,e.b,.54)),dx=b.x-a.x,dy=b.y-a.y,L=Math.max(1,Math.hypot(dx,dy)),nx=-dy/L,ny=dx/L,off=(10+(e.pair%2)*2)*viewSettings.edgeSize,label=pairLetter(e.pair)+suffix;ctx.globalAlpha=.97;ctx.fillStyle='rgba(249,249,246,.92)';const tw=ctx.measureText(label).width;ctx.fillRect(m.x+nx*off-tw/2-3,m.y+ny*off-6.5,tw+6,13);ctx.fillStyle=pairColorDark(e.pair);ctx.fillText(label,m.x+nx*off,m.y+ny*off)}ctx.restore()
}
function orbifoldOrderText(v){return v==null||v==='inf'?'∞':String(v)}
function permCycleLength(p,s){if(!Array.isArray(p)||s<0||s>=p.length)return 1;let j=s,n=0;do{j=p[j];n++;if(n>p.length+1)return 1}while(j!==s);return n}
function permCycleRepresentative(p,s){if(!Array.isArray(p)||s<0||s>=p.length)return s;let j=s,b=s,n=0;do{b=Math.min(b,j);j=p[j];n++;if(n>p.length+1)break}while(j!==s);return b}
function upstairsVertexOrder(R,s,i){const m=[ord(SYS.group.a),ord(SYS.group.b),ord(SYS.group.c)][i];if(m==null)return'inf';if(!R.P)return m;const L=permCycleLength(R.P[i],s),q=m/L;return Number.isInteger(q)&&q>=1?q:1}
function upstairsCuspWidth(R,s,i){const m=[ord(SYS.group.a),ord(SYS.group.b),ord(SYS.group.c)][i];if(m!=null)return null;return R.P?permCycleLength(R.P[i],s):1}
function pointKey(z){return `${Math.round(z.x*1e7)},${Math.round(z.y*1e7)}`}
function pointOnBoundaryGeodesic(R,z,eps=2.5e-6){
  const p=kleinPoint(z);if(!Number.isFinite(p.x+p.y))return false;
  for(const e of R.boundary||[]){const a=kleinPoint(e.a),b=kleinPoint(e.b),dx=b.x-a.x,dy=b.y-a.y,L2=dx*dx+dy*dy;if(L2<1e-18)continue;const t=((p.x-a.x)*dx+(p.y-a.y)*dy)/L2;if(t<-eps||t>1+eps)continue;const x=a.x+t*dx,y=a.y+t*dy;if(Math.hypot(p.x-x,p.y-y)<=eps*Math.max(1,Math.sqrt(L2)))return true}return false
}
function classLetter(n){let s='';for(n=Math.max(0,n|0);;){s=String.fromCharCode(65+n%26)+s;n=Math.floor(n/26)-1;if(n<0)return s}}
function fordCornerAngle(R,i){const n=R.boundary.length,z=R.boundary[i].a,pr=R.boundary[(i+n-1)%n],nx=R.boundary[i],a=safeGeodesicMid(pr.a,pr.b,.985),b=safeGeodesicMid(nx.a,nx.b,.015),u=sub(a,z),v=sub(b,z),du=absv(u),dv=absv(v);if(du<1e-8||dv<1e-8)return 0;return Math.acos(Math.max(-1,Math.min(1,(u.x*v.x+u.y*v.y)/(du*dv))))}
function fordVertexData(R){
  if(R.vertexData)return R.vertexData;const n=R.boundary?.length||0;if(!n)return[];const parent=Array.from({length:n},(_,i)=>i),find=x=>{while(parent[x]!==x){parent[x]=parent[parent[x]];x=parent[x]}return x},join=(a,b)=>{a=find((a+n)%n);b=find((b+n)%n);if(a!==b)parent[Math.max(a,b)]=Math.min(a,b)};
  for(let i=0;i<n;i++){const e=R.boundary[i],j=e.fordPartner;if(e.fordKind!=='arc'||!Number.isInteger(j)||!e.fordPairIso)continue;const f=R.boundary[j],aa=isoApply(e.fordPairIso,e.a),bb=isoApply(e.fordPairIso,e.b),rev=absv(sub(aa,f.b))+absv(sub(bb,f.a))<=absv(sub(aa,f.a))+absv(sub(bb,f.b));if(rev){join(i,j+1);join(i+1,j)}else{join(i,j);join(i+1,j+1)}}
  const tr=[];for(let i=0;i<n;i++)if(R.boundary[i].fordKind==='translation')tr.push(i);if(tr.length>=2){const a=tr[0],b=tr[tr.length-1];join(a,b+1);join(a+1,b)}
  const cls=new Map();for(let i=0;i<n;i++){const r=find(i),z=R.boundary[i].a,q=cls.get(r)||{sum:0,cusp:false,idx:[]};q.idx.push(i);q.cusp=q.cusp||absv(z)>.999999;q.sum+=q.cusp?0:fordCornerAngle(R,i);cls.set(r,q)}
  const sourceKey=`ford:${R.record?.symbol||R.N||'G'}`,out=[];for(let i=0;i<n;i++){const z=R.boundary[i].a,r=find(i),q=cls.get(r),ideal=q.cusp,ratio=q.sum>1e-8?2*Math.PI/q.sum:1,m=Math.max(1,Math.round(ratio)),order=ideal?'inf':(Math.abs(ratio-m)<.18?m:1);out.push({z,order,width:ideal?1:null,key:pointKey(z),classKey:`${sourceKey}:${r}`,sourceKey,sourceKind:'ford',vertexIndex:i,center:R.center})}R.vertexData=out;return out
}
function nearestVertexIndex(points,z,eps=1.2e-5){let best=-1,bd=Infinity;for(let i=0;i<points.length;i++){const d=absv(sub(points[i].z,z));if(d<bd){bd=d;best=i}}return bd<=eps?best:-1}
function boundaryVertexClasses(R){
  if(R.boundaryVertexClasses)return R.boundaryVertexClasses;if(R.ford)return fordVertexData(R);
  const points=[],byKey=new Map(),addPoint=z=>{const k=pointKey(z);if(byKey.has(k))return byKey.get(k);const i=points.length;points.push({z,k,occ:[]});byKey.set(k,i);return i};
  for(const e of R.boundary||[]){addPoint(e.a);addPoint(e.b)}
  // Every Coxeter vertex which lies on the actual polygon boundary is retained;
  // this includes subdivision vertices lying on a longer geodesic side.
  for(const ch of R.chambers||[]){const s0=Number.isInteger(ch.sheet)?ch.sheet:0,v=transformTri(ch.A);for(let i=0;i<3;i++){const z=v[i],k=pointKey(z);let j=byKey.get(k);if(j==null&&pointOnBoundaryGeodesic(R,z)){j=addPoint(z)}if(j!=null)points[j].occ.push({sheet:s0,parent:i})}}
  const parent=Array.from({length:points.length},(_,i)=>i),find=x=>{while(parent[x]!==x){parent[x]=parent[parent[x]];x=parent[x]}return x},join=(a,b)=>{a=find(a);b=find(b);if(a===b)return;if(a>b)[a,b]=[b,a];parent[b]=a};
  // Identify boundary copies by the *actual* side-pairing maps.  This avoids
  // conflating a background-triangle vertex with its order in the subgroup.
  for(const e of boundaryPairingSides(R)){if(!e.pairMap)continue;for(let i=0;i<points.length;i++){if(!pointOnBoundaryGeodesic({boundary:[e]},points[i].z,7e-6))continue;const w=isoApply(e.pairMap,points[i].z),j=nearestVertexIndex(points,w);if(j>=0)join(i,j)}}
  const ms=[ord(SYS.group.a),ord(SYS.group.b),ord(SYS.group.c)],classes=new Map();
  for(let i=0;i<points.length;i++){const root=find(i),q=classes.get(root)||{root,idx:[],angle:0,cusp:false,cuspSectors:0,parents:new Set()};q.idx.push(i);for(const o of points[i].occ){const m=ms[o.parent];q.parents.add(o.parent);if(m==null){q.cusp=true;q.cuspSectors++}else q.angle+=Math.PI/m}if(abs2(points[i].z)>.999999999)q.cusp=true;classes.set(root,q)}
  const prefix=`cover:${R.degree||1}:${R.root||0}`,out=[];
  for(const q of classes.values()){
    let order=1;if(q.cusp)order='inf';else if(q.angle>1e-10){const ratio=2*Math.PI/q.angle,m=Math.max(1,Math.round(ratio));order=Math.abs(ratio-m)<2e-4?m:1}
    const classKey=`${prefix}:bdry:${q.root}`,parentVertex=q.parents.size===1?[...q.parents][0]:null;
    const width=q.cusp?Math.max(1,Math.round(q.cuspSectors/2)):null;
    for(const i of q.idx)out.push({z:points[i].z,order,width,key:points[i].k,classKey,parentVertex,sourceKey:prefix,sourceKind:'cover',center:R.center})
  }
  // Stable cyclic order makes A,B,C,… deterministic and independent of the camera.
  out.sort((u,v)=>Math.atan2(u.z.y-(R.center?.y||0),u.z.x-(R.center?.x||0))-Math.atan2(v.z.y-(R.center?.y||0),v.z.x-(R.center?.x||0))||u.key.localeCompare(v.key));
  R.boundaryVertexClasses=out;return out
}
function regionVertexData(R){if(R.vertexData)return R.vertexData;const out=R.ford?fordVertexData(R):boundaryVertexClasses(R);R.vertexData=out;return out}
function pointInsideRegion(R,z){
  if(pointOnBoundaryGeodesic(R,z,5e-6))return true;const p=kleinPoint(z),poly=(R.boundary||[]).map(e=>kleinPoint(e.a)).filter(q=>Number.isFinite(q.x+q.y));if(poly.length<3||!Number.isFinite(p.x+p.y))return false;let inside=false;
  for(let i=0,j=poly.length-1;i<poly.length;j=i++){const a=poly[i],b=poly[j],hit=((a.y>p.y)!==(b.y>p.y))&&(p.x<(b.x-a.x)*(p.y-a.y)/(b.y-a.y+1e-30)+a.x);if(hit)inside=!inside}return inside
}
function allCellVertexData(R){
  const base=regionVertexData(R),map=new Map(base.map(q=>[q.key||pointKey(q.z),{...q}]));if(R.ford)return allFordEdgeVertexData(R,map);
  const prefix=`cover:${R.degree||1}:${R.root||0}`;
  // A tessellation vertex strictly inside a true fundamental polygon has a full
  // 2π neighbourhood in that polygon, hence is regular for the displayed group.
  // Do not inherit the order of the background Coxeter vertex here.
  for(const ch of R.chambers||[]){const v=transformTri(ch.A);for(const z of v){const k=pointKey(z);if(map.has(k))continue;if(!pointInsideRegion(R,z))continue;map.set(k,{z,order:1,width:null,key:k,classKey:`${prefix}:interior:${k}`,parentVertex:null,sourceKey:prefix,sourceKind:'cell',center:R.center,subdivision:true})}}
  return[...map.values()]
}
function allFordEdgeVertexData(R,map0=null){
  const map=map0||new Map(regionVertexData(R).map(q=>[q.key||pointKey(q.z),{...q}])),g=geom(),seen=new Set(),seed=isoInverse(backgroundAnchor()),q=[seed],sourceKey=`${R.record?.symbol||R.N||'ford'}:mesh`;
  for(let qi=0;qi<q.length&&qi<1800;qi++){const A=q[qi],v=transformTri(A),c=screenWorld(lorentzAverage(v));if(c.x<-120||c.x>g.availW+120||c.y<-120||c.y>H+120)continue;const ak=A.M.a.x.toFixed(7)+','+A.M.a.y.toFixed(7)+','+A.M.b.x.toFixed(7)+','+A.M.b.y.toFixed(7)+(A.anti?'a':'h');if(seen.has(ak))continue;seen.add(ak);for(const z of v){if(abs2(z)>=.999999999||!pointInsideRegion(R,z))continue;const k=pointKey(z);if(!map.has(k))map.set(k,{z,order:1,width:null,key:k,classKey:`${sourceKey}:${k}`,sourceKey,sourceKind:'ford-mesh',center:R.center,subdivision:true})}for(const S of SYS.R)q.push(isoCompose(A,S))}
  return[...map.values()]
}
function displayRegionVertexData(R){return viewSettings.allEdgeVertices?allCellVertexData(R):regionVertexData(R)}
function labeledRegionVertexData(R,{suffix=''}={}){
  const data=displayRegionVertexData(R).map(q=>({...q,center:q.center||R.center||C(),sourceRegion:R}));
  const classInfo=new Map();for(const q of data){const ck=q.classKey||`${q.sourceKey||''}:${q.key}`,special=q.order==='inf'||Number(q.order)>=2,old=classInfo.get(ck);if(!old)classInfo.set(ck,{ck,special,z:q.z});else if(special)old.special=true}
  const center=R.center||C(),angle=z=>Math.atan2(z.y-center.y,z.x-center.x),classes=[...classInfo.values()].sort((a,b)=>(b.special?1:0)-(a.special?1:0)||angle(a.z)-angle(b.z)||a.ck.localeCompare(b.ck));
  const letters=new Map();classes.forEach((q,i)=>letters.set(q.ck,classLetter(i)+suffix));
  for(const q of data){const ck=q.classKey||`${q.sourceKey||''}:${q.key}`,L=letters.get(ck)||suffix;q.classLetter=L;q.displayLabel=q.order==='inf'?`${q.width||1}∞${L}`:`${Number(q.order)>=1?q.order:1}${L}`}
  return data
}
function idealScreenPoint(q){
  const exact=screenWorld(q.z);if(model==='disk')return exact;
  const u=cameraApply(q.z),atInfinity=absv(sub(C(1),u))<2e-6;if(!atInfinity)return exact;
  let lo=0,hi=1,best=screenWorld(q.center||C());const g=geom();for(let i=0;i<38;i++){const t=(lo+hi)/2,z=safeGeodesicMid(q.center||C(),q.z,t),p=screenWorld(z),ok=Number.isFinite(p.x+p.y)&&p.x>=12&&p.x<=g.availW-12&&p.y>=12&&p.y<=H-12;if(ok){best=p;lo=t}else hi=t}return best
}
function pinCuspToViewport(p,anchor,margin=17){
  const g=geom(),x0=margin,x1=Math.max(x0+2,g.availW-margin),y0=margin,y1=Math.max(y0+2,H-margin);if(Number.isFinite(p?.x+p?.y)&&p.x>=x0&&p.x<=x1&&p.y>=y0&&p.y<=y1)return{p,off:false,dir:C()};const a=Number.isFinite(anchor?.x+anchor?.y)?anchor:C(g.availW/2,H/2);let tx=Number.isFinite(p?.x)?p.x-a.x:0,ty=Number.isFinite(p?.y)?p.y-a.y:-1;if(Math.abs(tx)+Math.abs(ty)<1e-9)ty=-1;let t=Infinity;if(tx>1e-9)t=Math.min(t,(x1-a.x)/tx);else if(tx<-1e-9)t=Math.min(t,(x0-a.x)/tx);if(ty>1e-9)t=Math.min(t,(y1-a.y)/ty);else if(ty<-1e-9)t=Math.min(t,(y0-a.y)/ty);if(!Number.isFinite(t)||t<=0)t=1;const q=C(Math.max(x0,Math.min(x1,a.x+tx*t)),Math.max(y0,Math.min(y1,a.y+ty*t))),r=Math.hypot(tx,ty)||1;return{p:q,off:true,dir:C(tx/r,ty/r)}
}
function drawRegionVertexLabels(R,{stroke='rgba(28,28,25,.38)',fill='rgba(249,249,245,.92)',text='rgba(22,22,20,.96)',alpha=1,suffix='',nudge=0}={}){
  if(!viewSettings.vertices)return;const data=labeledRegionVertexData(R,{suffix}),vs=viewSettings.vertexSize;ctx.save();ctx.globalAlpha=alpha;ctx.font=`600 ${12*vs}px "STIX Two Math","Cambria Math",serif`;ctx.textAlign='center';ctx.textBaseline='middle';
  for(const q of data){const ideal=q.order==='inf',regular=!ideal&&Number(q.order)<2;if(regular&&!viewSettings.allEdgeVertices)continue;if(!viewSettings.allEdgeVertices&&ideal&&!viewSettings.cuspVertices)continue;if(!viewSettings.allEdgeVertices&&!ideal&&!regular&&!viewSettings.ellipticVertices)continue;const raw=ideal?idealScreenPoint(q):screenWorld(q.z);if(!ideal&&(!Number.isFinite(raw.x+raw.y)||raw.x<-22||raw.x>W+22||raw.y<-22||raw.y>H+22))continue;const anchor=screenWorld(q.center||C()),pin=ideal?pinCuspToViewport(raw,anchor):{p:raw,off:false,dir:C()},p0=pin.p;if(!Number.isFinite(p0.x+p0.y))continue;let label=q.displayLabel||(ideal?`${q.width||1}∞${suffix}`:`${orbifoldOrderText(q.order)}${suffix}`);if(viewSettings.vertexLabel==='order')label=ideal?`${q.width||1}∞`:(regular?'1':orbifoldOrderText(q.order));else if(viewSettings.vertexLabel==='dot')label='';const rad=(viewSettings.vertexLabel==='dot'?5.2:Math.max(8.3,6.5+2.35*label.length))*vs;
    // In a two-domain comparison both copies must remain visible.  A tiny
    // deterministic normal nudge keeps coincident primed/unprimed labels apart.
    const dx=p0.x-anchor.x,dy=p0.y-anchor.y,L=Math.hypot(dx,dy)||1,nx=-dy/L,ny=dx/L,p=C(p0.x+nx*nudge*vs,p0.y+ny*nudge*vs);
    if(ideal&&pin.off){const d=pin.dir,n=C(-d.y,d.x),tip=C(p.x+d.x*(rad+7),p.y+d.y*(rad+7)),b1=C(p.x+d.x*(rad+1)+n.x*3.2,p.y+d.y*(rad+1)+n.y*3.2),b2=C(p.x+d.x*(rad+1)-n.x*3.2,p.y+d.y*(rad+1)-n.y*3.2);ctx.fillStyle=stroke;ctx.beginPath();ctx.moveTo(tip.x,tip.y);ctx.lineTo(b1.x,b1.y);ctx.lineTo(b2.x,b2.y);ctx.closePath();ctx.fill()}
    ctx.fillStyle=fill;ctx.beginPath();ctx.arc(p.x,p.y,rad,0,Math.PI*2);ctx.fill();ctx.strokeStyle=(selectedOrbifoldPoint?.classKey&&selectedOrbifoldPoint.classKey===q.classKey)?'rgba(36,73,132,.88)':stroke;ctx.lineWidth=((selectedOrbifoldPoint?.classKey&&selectedOrbifoldPoint.classKey===q.classKey)?2.2:1.25)*vs;ctx.stroke();if(label){ctx.fillStyle=text;ctx.fillText(label,p.x,p.y+.2)}vertexHitTargets.push({x:p.x,y:p.y,r:rad+6,q:{...q,label:label||q.displayLabel||'•'}})}ctx.restore()
}
function drawOrbifoldVertexLabels(){drawRegionVertexLabels(baseRegion())}

function bezoutInt(a,b){a=Math.trunc(a);b=Math.trunc(b);let oa=a,ob=b,x0=1,y0=0,x1=0,y1=1;while(b){const q=Math.trunc(a/b),t=a-q*b;a=b;b=t;[x0,x1]=[x1,x0-q*x1];[y0,y1]=[y1,y0-q*y1]}if(a<0){a=-a;x0=-x0;y0=-y0}return{g:a,x:x0,y:y0,a:oa,b:ob}}
function halfPSLIso(a,b,c,d){return iso({a:C(c-b,a+d),b:C(b+c,a-d),c:C(-(b+c),a-d),d:C(b-c,a+d)},false)}
function moonCircleIso(q,rec){
  const e=Number(q.e||1),den=Number(q.den||1),dd=Number(q.d||0),m=Number(rec.m||rec.N||1),h=Number(rec.h||1),B=den*(m/e),A=dd*e,z=bezoutInt(A,B);if(z.g!==1)return null;
  const aa=z.x,bb=-z.y,G=halfPSLIso(aa*e,bb/h,den*m*h,dd*e),sh=Number(q.shift||0);return Math.abs(sh)>1e-14?isoCompose(G,halfPlaneTranslationIso(-sh)):G
}
function endpointPairError(A,e,f){const m=mappedEdgeMatch(A,e,f);return{score:m.score,reverse:!m.direct,direct:m.direct}}
function orientFordPairings(R){
  const groups=new Map();for(let i=0;i<R.boundary.length;i++){const e=R.boundary[i],k=e.fordPairKey||`side:${i}`;if(!groups.has(k))groups.set(k,[]);groups.get(k).push({e,i})}
  let maxError=0,invalid=0;
  for(const [key,sides] of groups){for(const q of sides)q.e.fordArrowForward=null;if(sides.length<2)continue;sides.sort((a,b)=>a.i-b.i);const src=sides[0],tgt=sides[1];let A=null;
    if(src.e.fordKind==='arc'&&src.e.fordPartner===tgt.i&&src.e.fordPairIso)A=src.e.fordPairIso;
    else if(tgt.e.fordKind==='arc'&&tgt.e.fordPartner===src.i&&tgt.e.fordPairIso)A=isoInverse(tgt.e.fordPairIso);
    else if(src.e.fordKind==='translation'&&tgt.e.fordKind==='translation'){
      const p=Number(R.period||1),cands=[halfPlaneTranslationIso(p),halfPlaneTranslationIso(-p)];let best=null;for(const T of cands){const m=mappedEdgeMatch(T,src.e,tgt.e);if(!best||m.score<best.m.score)best={A:T,m}}if(best)A=best.A
    }
    if(!A){invalid++;continue}const m=mappedEdgeMatch(A,src.e,tgt.e);maxError=Math.max(maxError,m.score);if(m.score>1.5e-3){invalid++;continue}src.e.fordArrowForward=true;tgt.e.fordArrowForward=m.direct;src.e.fordPairMapError=tgt.e.fordPairMapError=m.score
  }
  R.fordOrientationAudit={pairs:groups.size,maxEndpointError:maxError,invalid};if(invalid)console.warn('Ford side-pairing orientation audit',R.fordOrientationAudit);return R
}
function attachFordPairings(R,rec){
  const arcs=R.boundary.map((e,i)=>({e,i})).filter(q=>q.e.fordKind==='arc'),h=Math.max(1,Number(rec.h||1));
  for(const q of arcs){const G=moonCircleIso(q.e.fordSegment,rec);if(!G)continue;let best=null;for(let j=-2*h;j<=2*h;j++){const Cnd=isoCompose(halfPlaneTranslationIso(j/h),G);for(const f of arcs){const er=endpointPairError(Cnd,q.e,f.e);if(!best||er.score<best.score)best={...er,A:Cnd,j:f.i}}}if(best&&best.score<1.25e-3){q.e.fordPairIso=best.A;q.e.fordPartner=best.j}}
  for(let i=0;i<R.boundary.length;i++){const e=R.boundary[i];if(e.fordKind!=='arc')continue;if(Number.isInteger(e.fordPartner)){const a=Math.min(i,e.fordPartner),b=Math.max(i,e.fordPartner);e.fordPairKey=`a:${a}:${b}`}else e.fordPairKey=e.fordPairKey||`a:${i}`}
  R.fordPairingCoverage=arcs.length?arcs.filter(q=>q.e.fordPairIso).length/arcs.length:1;return orientFordPairings(R)
}
function fordRegion(rec){
  if(!rec?.vertices?.length)return null;const vs=rec.vertices.map(v=>C(Number(v[0]),Number(v[1]))),boundary=[],seg=rec.segments||[],S=seg.length;
  for(let i=0;i<vs.length;i++){
    const e={a:vs[i],b:vs[(i+1)%vs.length]};
    if(i<S){const q=seg[i];e.fordSegment=q;const ak=`${Number(q.e||1)}:${Math.abs(Number(q.c||0)).toFixed(9)}:${Number(q.r||0).toFixed(9)}`;e.fordKind='arc';e.fordE=Number(q.e||1);e.fordC=Number(q.c||0);e.fordPairKey='a:'+ak;e.fordDir=e.fordC<0?-1:1}
    else{e.fordKind='translation';e.fordPairKey='T';e.fordDir=i===S?1:-1}
    boundary.push(e)
  }
  const ymax=Math.max(.35,...seg.map(q=>Number(q.y0||0)),...seg.map(q=>Number(q.y1||0))),cx=Number(rec.center_x||0),cy=Math.max(.62,ymax+.28),center=icy(C(cx,cy));
  const R={degree:1,root:0,P:null,Pi:null,reps:[],chambers:[],boundary,center,centerH:C(cx,cy),neighbors:[],ford:true,N:Number(rec.N||1),period:1,record:rec,fordCusp:C(1,0)};return rec.symbol?attachFordPairings(R,rec):R
}
function transformFordRegion(R,A,tag=0){if(!R)return null;const baseVD=regionVertexData(R),z=q=>isoApply(A,q),center=z(R.center),cusp=z(R.fordCusp||C(1,0));return{...R,boundary:R.boundary.map(e=>({...e,a:z(e.a),b:z(e.b)})),vertexData:baseVD.map(q=>({...q,z:z(q.z),center})),center,centerH:cay(center),fordCusp:cusp,fordShift:tag,fordTileIso:A}}
function tileIsoKey(A){
  const pts=[C(.031,.017),C(.173,-.061),C(-.119,.143)],q=[];for(const z of pts){const w=isoApply(A,z);q.push(Math.round(w.x*2e6),Math.round(w.y*2e6))}return(A.anti?'a:':'h:')+q.join(',')
}
function regionNeighborGenerators(R){
  if(!R)return[];const out=[],seen=new Set(),add=A=>{if(!A)return;const k=tileIsoKey(A);if(seen.has(k))return;seen.add(k);out.push(A)};
  if(R.ford){
    for(const e of R.boundary||[])if(e.fordKind==='arc'&&e.fordPairIso)add(isoInverse(e.fordPairIso));
    // The two vertical Ford sides are paired by the cusp translation.  Keep
    // both directions even when the record has no explicit map on those sides.
    const p=Number(R.period||1);if((R.boundary||[]).some(e=>e.fordKind==='translation')){add(halfPlaneTranslationIso(p));add(halfPlaneTranslationIso(-p))}
  }else for(const e of boundaryPairingSides(R))if(e.pairMap)add(isoInverse(e.pairMap));
  return out
}
function regionNeighborTransforms(R,depth,start=iso()){
  // v27: the layer slider is the single source of truth.  Older builds had
  // an independent Neighbors checkbox; ignoring that stale persisted bit keeps
  // depth 1/2/3 effective in every domain view.
  depth=Math.max(0,Math.min(3,Math.round(Number(depth)||0)));if(depth<1)return[];const gens=regionNeighborGenerators(R);if(!gens.length)return[];
  const out=[],seen=new Set([tileIsoKey(start)]),q=[{A:start,d:0}];for(let qi=0;qi<q.length;qi++){const cur=q[qi];if(cur.d>=depth)continue;for(const G of gens){const A=isoCompose(cur.A,G),k=tileIsoKey(A);if(seen.has(k))continue;seen.add(k);const z={A,d:cur.d+1};q.push(z);out.push(z)}}return out
}
function transformRegionCopy(R,A,layer=1){
  if(R.ford)return transformFordRegion(R,A,layer);const z=q=>isoApply(A,q),center=z(R.center||C());return{...R,boundary:(R.boundary||[]).map(e=>({...e,a:z(e.a),b:z(e.b)})),chambers:(R.chambers||[]).map(ch=>({...ch,A:isoCompose(A,ch.A)})),center,vertexData:null,boundaryVertexClasses:null,pairingSides:null,neighborLayer:layer,tileIso:A}
}
function regionNeighborCopies(R,depth=viewSettings.neighborCount,start=iso()){return regionNeighborTransforms(R,depth,start).map(q=>transformRegionCopy(R,q.A,q.d))}
function drawGenericNeighbors(R,{stroke='rgba(50,61,79,.42)',hatch='rgba(70,84,108,.065)',direction=1,dash=[7,5]}={}){
  const a=Math.max(.08,Math.min(1,Number(viewSettings.neighborOpacity)||1));for(const Q of regionNeighborCopies(R)){ctx.save();ctx.globalAlpha=a*Math.max(.30,1-.16*(Q.neighborLayer-1));if(Q.ford)drawBoundaryHatch(Q,{stroke:hatch,direction,spacing:9,width:.46});else drawRegionHatch(Q,{stroke:hatch,direction,spacing:9,width:.46});strokeBoundary(Q.boundary,{width:1.4,stroke,dash});ctx.restore()}
}
function moonshineReductionTransform(){
  if(!frickeCompare?.moonshine)return iso();const B=frickeCompare.basePlus||frickeCompare.plus;let z=cameraInverse(C()),A=iso();
  for(let step=0;step<42;step++){
    let w=cay(z);if(!Number.isFinite(w.x+w.y)||w.y<=0)break;
    const L=Number(B.record?.domain_left??-.5),n=Math.floor(w.x-L);if(n){const T=halfPlaneTranslationIso(-n);z=isoApply(T,z);A=isoCompose(T,A);continue}
    let hit=null,def=1e-9;for(const e of B.boundary){if(e.fordKind!=='arc'||!e.fordSegment)continue;const q=e.fordSegment,x=w.x;if(x<q.x0-2e-8||x>q.x1+2e-8)continue;const yy=Math.sqrt(Math.max(0,q.r*q.r-(x-q.c)*(x-q.c))),d=yy-w.y;if(d>def){def=d;hit=e}}
    if(!hit)break;const G=hit.fordPairIso||((Number(B.record?.h||1)===1)?moonCircleIso(hit.fordSegment,B.record):null);if(!G)break;z=isoApply(G,z);A=isoCompose(G,A)
  }
  return isoInverse(A)
}
function currentMoonshineFordRegion(){if(!frickeCompare?.moonshine)return frickeCompare?.plus||null;const B=frickeCompare.basePlus||frickeCompare.plus;return transformFordRegion(B,moonshineReductionTransform(),0)}
function moonshineFordNeighbors(){if(!frickeCompare?.moonshine)return[];const B=frickeCompare.basePlus||frickeCompare.plus,A=moonshineReductionTransform();return regionNeighborTransforms(B,viewSettings.neighborCount,A).map(q=>transformFordRegion(B,q.A,q.d))}
function pathBoundary(boundary){ctx.beginPath();let first=true;for(const e of boundary){traceWorld(e.a,e.b,first);first=false}ctx.closePath()}
function drawFordPairings(R,{alpha=.95,labels=true,suffix=''}={}){
  if(!R?.boundary?.length||!viewSettings.pairings)return;const keys=[...new Set(R.boundary.map(e=>e.fordPairKey||'side'))],idx=new Map(keys.map((k,i)=>[k,i]));ctx.save();ctx.globalAlpha=alpha;ctx.lineCap='round';ctx.lineJoin='round';ctx.font=`600 ${10.5*viewSettings.edgeSize}px ui-monospace, SFMono-Regular, Menlo, monospace`;ctx.textAlign='center';ctx.textBaseline='middle';
  for(const e of R.boundary){const pair=idx.get(e.fordPairKey||'side')||0,color=pairColor(pair),ee={...e,arrowForward:e.fordArrowForward};strokeBoundary([e],{width:1.95,stroke:color});if(viewSettings.arrows)drawPairingArrow(ee,pair);if(!labels||!viewSettings.edgeLabels)continue;const m=screenWorld(safeGeodesicMid(e.a,e.b,.5));if(!Number.isFinite(m.x+m.y)||m.x<-12||m.x>W+12||m.y<-12||m.y>H+12)continue;const label=(e.fordKind==='translation'?'T':pairLetter(pair))+suffix,tw=ctx.measureText(label).width;ctx.fillStyle='rgba(249,249,246,.90)';ctx.fillRect(m.x-tw/2-3,m.y-6.5,tw+6,13);ctx.fillStyle=pairColorDark(pair);ctx.fillText(label,m.x,m.y+.2)}ctx.restore()
}
function drawFordCellTag(R,label,alpha=.72){if(!viewSettings.regionTags)return;const p=screenWorld(R?.center||C());if(!Number.isFinite(p.x+p.y)||p.x<10||p.x>W-10||p.y<12||p.y>H-12)return;ctx.save();ctx.globalAlpha=alpha;ctx.font='600 11px ui-monospace, SFMono-Regular, Menlo, monospace';ctx.textAlign='center';ctx.textBaseline='middle';const tw=ctx.measureText(label).width;ctx.fillStyle='rgba(249,249,246,.88)';ctx.fillRect(p.x-tw/2-4,p.y-8,tw+8,16);ctx.strokeStyle='rgba(38,44,58,.30)';ctx.strokeRect(p.x-tw/2-4,p.y-8,tw+8,16);ctx.fillStyle='rgba(32,38,52,.88)';ctx.fillText(label,p.x,p.y+.2);ctx.restore()}
function drawMoonshineTriangleSeams(R){
  if(!viewSettings.triangleSeams||!R?.boundary?.length)return;const g=geom(),seen=new Set(),edges=new Map(),seed=isoInverse(backgroundAnchor()),q=[seed];
  for(let qi=0;qi<q.length&&qi<1800;qi++){const A=q[qi],v=transformTri(A),c=screenWorld(lorentzAverage(v));if(c.x<-120||c.x>g.availW+120||c.y<-120||c.y>H+120)continue;const k=A.M.a.x.toFixed(7)+','+A.M.a.y.toFixed(7)+','+A.M.b.x.toFixed(7)+','+A.M.b.y.toFixed(7)+(A.anti?'a':'h');if(seen.has(k))continue;seen.add(k);for(const[a,b]of[[0,1],[1,2],[2,0]]){const ek=edgeKey(v[a],v[b]);if(!edges.has(ek))edges.set(ek,{a:v[a],b:v[b]})}for(const S of SYS.R)q.push(isoCompose(A,S))}
  ctx.save();pathBoundary(R.boundary);ctx.clip();ctx.setLineDash([3.2,4.2]);ctx.strokeStyle='rgba(36,48,66,.32)';ctx.lineWidth=.72;ctx.beginPath();for(const e of edges.values())traceWorld(e.a,e.b,true);ctx.stroke();ctx.restore()
}
function drawBoundaryHatch(R,{stroke='rgba(47,91,151,.22)',direction=1,spacing=8,width=.62}={}){
  if(!viewSettings.hatching||!R?.boundary?.length)return;ctx.save();pathBoundary(R.boundary);ctx.clip();ctx.lineWidth=width;ctx.strokeStyle=stroke;ctx.lineCap='butt';ctx.beginPath();if(direction>0){for(let x=-H;x<W+H;x+=spacing){ctx.moveTo(x,0);ctx.lineTo(x+H,H)}}else{for(let x=-H;x<W+H;x+=spacing){ctx.moveTo(x,H);ctx.lineTo(x+H,0)}}ctx.stroke();ctx.restore()
}
function drawFrickeComparison(){
  if(!frickeCompare)return;
  if(frickeCompare.moonshine){
    const P=currentMoonshineFordRegion(),nb=moonshineFordNeighbors();
    for(const Q of nb){const na=Math.max(.12,Math.min(1,viewSettings.neighborOpacity))*Math.max(.30,1-.16*((Q.fordShift||1)-1));ctx.save();ctx.globalAlpha=na;drawBoundaryHatch(Q,{stroke:'rgba(96,108,132,.075)',direction:-1,spacing:9,width:.48});strokeBoundary(Q.boundary,{width:1.45,stroke:'rgba(54,67,91,.42)',dash:[7,5]});drawFordPairings(Q,{alpha:.28,labels:false});ctx.restore()}
    drawBoundaryHatch(P,{stroke:'rgba(42,91,160,.20)',direction:1,spacing:7,width:.58});drawMoonshineTriangleSeams(P);strokeBoundary(P.boundary,{width:3.05,stroke:'rgba(31,38,52,.96)'});drawFordPairings(P,{alpha:.93,labels:true});drawFordCellTag(P,'F0',.86);
    $('domain-note').innerHTML=`<span class="domain-key"><i style="display:inline-block;width:22px;border-top:3px solid #273044;vertical-align:middle"></i>Γ<sub>${esc(frickeCompare.className)}</sub></span><span>F₀</span>`;updateLegendVisibility();return
  }
  const P=frickeCompare.plus,G=frickeCompare.gamma0;
  drawGenericNeighbors(P,{stroke:'rgba(54,67,91,.40)',hatch:'rgba(42,91,160,.065)',direction:1,dash:[7,5]});
  if(G)drawGenericNeighbors(G,{stroke:'rgba(132,55,53,.34)',hatch:'rgba(176,66,63,.055)',direction:-1,dash:[8,5]});
  if(G){drawBoundaryHatch(G,{stroke:'rgba(176,66,63,.17)',direction:-1,spacing:7,width:.56});strokeBoundary(G.boundary,{width:2.0,stroke:'rgba(132,55,53,.76)',dash:[8,4]})}
  drawBoundaryHatch(P,{stroke:'rgba(42,91,160,.22)',direction:1,spacing:7,width:.58});strokeBoundary(P.boundary,{width:3.05,stroke:'rgba(31,38,52,.96)'});
  const name=frickeCompare.name,N=frickeCompare.N;$('domain-note').innerHTML=`<span class="domain-key"><i style="display:inline-block;width:22px;border-top:3px solid #273044;vertical-align:middle"></i>Γ<sub>0</sub>(${N})<sup>+</sup> · ${esc(name)}</span>${G?`<span class="domain-key"><i style="display:inline-block;width:22px;border-top:2px dashed #843735;vertical-align:middle"></i>Γ<sub>0</sub>(${N})</span>`:''}<span>Ford</span>`
}
function drawStandaloneCusp(z,width,center,style={}){drawRegionVertexLabels({boundary:[{a:z,b:z}],vertexData:[{z,order:'inf',width,center,key:'ford',classKey:`standalone:${pointKey(z)}`,sourceKind:'ford'}],center},style)}
function drawFrickeCuspLabels(){
  if(!frickeCompare)return;const P=frickeCompare.moonshine?currentMoonshineFordRegion():frickeCompare.plus;
  if(frickeCompare.moonshine){drawRegionVertexLabels(P,{stroke:'rgba(31,38,52,.52)'});for(const Q of moonshineFordNeighbors())drawRegionVertexLabels(Q,{stroke:'rgba(31,38,52,.36)',alpha:.7})}
  else{drawRegionVertexLabels(P,{stroke:'rgba(31,38,52,.52)',suffix:'',nudge:-5});if(frickeCompare.gamma0)drawRegionVertexLabels(frickeCompare.gamma0,{stroke:'rgba(31,38,52,.52)',suffix:'′',nudge:5})}
}
function drawRegionHatch(R,{stroke='rgba(47,91,151,.22)',direction=1,spacing=8,width=.62}={}){
  if(!viewSettings.hatching||!R?.chambers?.length)return;ctx.save();ctx.beginPath();
  for(const ch of R.chambers){const v=transformTri(ch.A);traceWorld(v[0],v[1],true);traceWorld(v[1],v[2],false);traceWorld(v[2],v[0],false);ctx.closePath()}
  ctx.clip();ctx.lineWidth=width;ctx.strokeStyle=stroke;ctx.lineCap='butt';ctx.beginPath();
  if(direction>0){for(let x=-H;x<W+H;x+=spacing){ctx.moveTo(x,0);ctx.lineTo(x+H,H)}}
  else{for(let x=-H;x<W+H;x+=spacing){ctx.moveTo(x,H);ctx.lineTo(x+H,0)}}
  ctx.stroke();ctx.restore()
}
function setDomainLegend(R,hasCover=true){
  updateLegendVisibility();if(!hasCover){$('domain-note').innerHTML='<span class="domain-key"><i class="line-swatch base"></i>F<sub>Γ</sub></span>';return}
  const base=viewSettings.coverMarksOnly?'':`<span class="domain-key"><i class="line-swatch base"></i>F<sub>Γ</sub></span>`;
  $('domain-note').innerHTML=`<span class="domain-key"><i class="line-swatch cover"></i>F<sub>H</sub></span><span class="domain-key"><i class="line-swatch cells"></i>g<sub>i</sub>F<sub>Γ</sub></span>${base}<span>[Γ:H]=${R.degree}</span>`
}
function updateLegendVisibility(){const el=$('domain-note');if(el)el.style.display=viewSettings.legend?'':'none'}
function drawCommComparison(){
  const Cmp=commCompare;if(!Cmp)return;const A=Cmp.a.R,B=Cmp.b.R,S=baseRegion(),hideA=viewSettings.coverMarksOnly&&A.degree===1&&B.degree>1,hideB=viewSettings.coverMarksOnly&&B.degree===1&&A.degree>1;
  const fillR=A.degree===1&&B.degree>1?B:(B.degree===1&&A.degree>1?A:null);
  if(fillR){fillGammaCells(fillR);if(viewSettings.cellSeams)strokeBoundary(gammaCellEdges(fillR,true),{width:.88,stroke:'rgba(17,17,15,.38)',dash:[4,4]})}
  if(!hideA)drawGenericNeighbors(A,{stroke:'rgba(42,91,160,.39)',hatch:'rgba(42,91,160,.055)',direction:1,dash:[7,5]});
  if(!hideB)drawGenericNeighbors(B,{stroke:'rgba(132,55,53,.35)',hatch:'rgba(176,66,63,.050)',direction:-1,dash:[8,5]});
  // v19: both simultaneously displayed fundamental regions get independent
  // dense, pale hatching.  Drawing them separately means the overlap naturally
  // carries both directions.
  drawRegionHatch(A,{stroke:'rgba(42,91,160,.23)',direction:1,spacing:7,width:.58});
  drawRegionHatch(B,{stroke:'rgba(176,66,63,.22)',direction:-1,spacing:7,width:.58});
  if(A.degree>1&&B.degree>1)strokeBoundary(S.boundary,{width:1.15,stroke:'rgba(17,17,15,.48)',dash:[2,4]});
  strokeBoundary(A.boundary,{width:3.25,stroke:'rgba(31,38,52,.97)'});
  strokeBoundary(B.boundary,{width:3.05,stroke:'rgba(17,17,15,.86)',dash:[10,5]});
  if(!hideA)drawPairings(A,{suffix:''});if(!hideB)drawPairings(B,{suffix:'′',alpha:.82});
  const baseExtra=(A.degree>1&&B.degree>1)?'<span class="domain-key"><i style="display:inline-block;width:22px;border-top:1px dotted #555;vertical-align:middle"></i>F<sub>S</sub></span>':'';
  $('domain-note').innerHTML=`<span class="domain-key"><i style="display:inline-block;width:22px;border-top:3px solid #273044;vertical-align:middle"></i>${esc(Cmp.a.short)}</span><span class="domain-key"><i style="display:inline-block;width:22px;border-top:3px dashed #222;vertical-align:middle"></i>${esc(Cmp.b.short)}</span>${baseExtra}<span>${esc(Cmp.baseShort)}</span>`
}
function drawRegion(){
  if(frickeCompare){drawFrickeComparison();return}
  if(commCompare){drawCommComparison();return}
  const R=currentRegion();
  if(selectedCover){
    // A mathematically exact coset decomposition: each g_i F_Gamma is filled
    // once; shared sides are deduplicated and drawn as a single dashed seam.
    const B=baseRegion();
    drawGenericNeighbors(R,{stroke:'rgba(54,67,91,.39)',hatch:'rgba(42,91,160,.055)',direction:1,dash:[7,5]});
    if(!viewSettings.coverMarksOnly)drawGenericNeighbors(B,{stroke:'rgba(70,70,66,.31)',hatch:'rgba(80,80,76,.04)',direction:-1,dash:[5,5]});
    fillGammaCells(R);
    if(viewSettings.cellSeams)strokeBoundary(gammaCellEdges(R,true),{width:.92,stroke:'rgba(17,17,15,.48)',dash:[4,4]});
    strokeBoundary(R.boundary,{width:3.15,stroke:'rgba(38,44,58,.97)'});
    drawPairings(R);
    strokeBoundary(B.boundary,{width:1.35,stroke:'rgba(17,17,15,.78)',dash:[6,4]});
    setDomainLegend(R,true)
  }else{
    drawGenericNeighbors(R,{stroke:'rgba(54,67,91,.39)',hatch:'rgba(42,91,160,.05)',direction:1,dash:[7,5]});
    strokeBoundary(R.boundary,{width:2.35,stroke:'rgba(38,44,58,.96)'});drawPairings(R);setDomainLegend(R,false)
  }
}
function drawCurrentVertexLabels(){if(frickeCompare){drawFrickeCuspLabels();return}if(commCompare){const A=commCompare.a,B=commCompare.b,hideA=viewSettings.coverMarksOnly&&A.degree===1&&B.degree>1,hideB=viewSettings.coverMarksOnly&&B.degree===1&&A.degree>1;if(!hideA)drawRegionVertexLabels(A.R,{stroke:'rgba(28,28,25,.42)',nudge:-5});if(!hideB)drawRegionVertexLabels(B.R,{stroke:'rgba(28,28,25,.42)',suffix:'′',nudge:5});return}const R=currentRegion();if(selectedCover){drawRegionVertexLabels(R,{stroke:'rgba(38,44,58,.52)',nudge:-5});if(!viewSettings.coverMarksOnly)drawRegionVertexLabels(baseRegion(),{stroke:'rgba(38,44,58,.52)',suffix:'′',nudge:5})}else drawRegionVertexLabels(R)}
function drawOverlay(){
  vertexHitTargets=[];ctx.clearRect(0,0,W,H);const g=geom();ctx.save();
  if(model==='disk'){const c=chartForward(C(g.cx,g.cy));ctx.beginPath();ctx.arc(c.x,c.y,g.r,0,Math.PI*2);ctx.clip()}else{const ps=[C(0,0),C(g.availW,0),C(g.availW,Math.max(0,g.bottom)),C(0,Math.max(0,g.bottom))].map(chartForward);ctx.beginPath();ctx.moveTo(ps[0].x,ps[0].y);for(let i=1;i<ps.length;i++)ctx.lineTo(ps[i].x,ps[i].y);ctx.closePath();ctx.clip()}
  if(!selectedCover||coverTriple(selectedCover))drawRegion();
  else{const B=baseRegion();strokeBoundary(B.boundary,{width:2.35,stroke:'rgba(38,44,58,.96)'});setDomainLegend(B,false);$('domain-note').innerHTML+=''}
  ctx.restore();updateLegendVisibility();drawCurrentVertexLabels();ctx.strokeStyle='rgba(17,17,15,.47)';ctx.lineWidth=1;
  if(model==='disk'){const c=chartForward(C(g.cx,g.cy));ctx.beginPath();ctx.arc(c.x,c.y,g.r,0,Math.PI*2);ctx.stroke()}else{const a=chartForward(C(0,g.bottom+.5)),b=chartForward(C(g.availW,g.bottom+.5));ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}
  overlayDirty=false
}

// -----------------------------------------------------------------------------
// Fallback background (only if WebGL2 is unavailable): small local atlas, never global BFS
// -----------------------------------------------------------------------------
function drawFallback(){
  ctx.save();const g=geom(),seen=new Set(),q=[{A:iso(),parity:0}],edges=new Map();
  for(let qi=0;qi<q.length&&qi<2800;qi++){
    const {A,parity}=q[qi],v=transformTri(A),center=screenWorld(lorentzAverage(v));
    if(center.x<-180||center.x>g.availW+180||center.y<-180||center.y>H+180)continue;
    const k=A.M.a.x.toFixed(7)+','+A.M.a.y.toFixed(7)+','+A.M.b.x.toFixed(7)+','+A.M.b.y.toFixed(7)+(A.anti?'a':'h');if(seen.has(k))continue;seen.add(k);
    ctx.fillStyle=parity?'rgba(226,227,222,.34)':'rgba(251,251,247,.30)';pathTriangle(v);ctx.fill();
    for(const[a,b]of[[0,1],[1,2],[2,0]]){const ek=edgeKey(v[a],v[b]);if(!edges.has(ek))edges.set(ek,{a:v[a],b:v[b]})}
    for(const R of SYS.R)q.push({A:isoCompose(A,R),parity:1-parity})
  }
  ctx.strokeStyle='rgba(17,17,15,.105)';ctx.lineWidth=.62;ctx.beginPath();for(const e of edges.values())traceWorld(e.a,e.b,true);ctx.stroke();ctx.restore()
}

// -----------------------------------------------------------------------------
// Render scheduling
// -----------------------------------------------------------------------------
function invalidate(interactive=false){if(interactive){gpuFast=true;clearTimeout(idleTimer);idleTimer=setTimeout(()=>{gpuFast=false;bgDirty=true;overlayDirty=true;schedule()},85)}bgDirty=true;overlayDirty=true;schedule()}
function render(){raf=0;if(bgDirty){if(gpuOK){fallback.style.display='none';renderGL()}else{fallback.style.display='block';ctx.clearRect(0,0,W,H);drawFallback();if(fallbackCtx){fallbackCtx.setTransform(1,0,0,1,0,0);fallbackCtx.clearRect(0,0,fallback.width,fallback.height);fallbackCtx.drawImage(overlay,0,0,fallback.width,fallback.height)}ctx.clearRect(0,0,W,H);bgDirty=false}}if(overlayDirty)drawOverlay()}
function schedule(){if(!raf)raf=requestAnimationFrame(render)}
function resize(){dpr=Math.min(2,devicePixelRatio||1);W=innerWidth;H=innerHeight;overlay.width=Math.round(W*dpr);overlay.height=Math.round(H*dpr);overlay.style.width=W+'px';overlay.style.height=H+'px';fallback.width=Math.round(W*dpr);fallback.height=Math.round(H*dpr);fallback.style.width=W+'px';fallback.style.height=H+'px';ctx.setTransform(dpr,0,0,dpr,0,0);if(fallbackCtx)fallbackCtx.setTransform(1,0,0,1,0,0);invalidate()}

// -----------------------------------------------------------------------------
// Covers / congr. subgroup data
// -----------------------------------------------------------------------------
function cyclePartition(p){
  if(!Array.isArray(p))return[];const seen=new Array(p.length).fill(false),out=[];
  for(let i=0;i<p.length;i++)if(!seen[i]){let j=i,n=0;while(!seen[j]){seen[j]=true;n++;j=(p[j]??1)-1}out.push(n)}
  return out.sort((a,b)=>b-a)
}
function partitionTex(p){return `(${(p||[]).join(',')})`}
function triplePartitions(t){return Array.isArray(t)?t.map(cyclePartition):null}
function coverLabel(r){if(r.modular)return r.label;return r.label||`${r.family||'H'}(${r.p??''})`}
function levelIdealData(row){
  const xs=row?.prime_ideals_with_required_residue_field_Fq||row?.level_prime_factorization_in_invariant_field||[];
  if(!xs.length)return null;const p=xs[0];return{e:p.e,f:p.f,norm:p.norm,generator:p.prime_ideal_generator}
}
const coverRecordCache=new Map();
function modularCuspData(row){const a=String(row?.cusps||'').trim().split(/\s+/).map(Number).filter(Number.isFinite),widths=[];for(let i=0;i+1<a.length;i+=2)for(let j=0;j<a[i+1];j++)widths.push(a[i]);return{count:widths.length,widths:widths.sort((x,y)=>x-y)}}
function modularEllipticOrders(row){const out=[];for(let i=0;i<Number(row?.c2||0);i++)out.push(2);for(let i=0;i<Number(row?.c3||0);i++)out.push(3);return out}
function coverRecords(g){
  if(coverRecordCache.has(g.id))return coverRecordCache.get(g.id);const out=[];
  if(g.id==='T2_3_I'){
    for(const r of modular){const cu=modularCuspData(r);out.push({key:r.key,label:r.label,family:'congruence',degree:r.index,genus:r.genus,level:r.level,modular:true,exact:true,row:r,cusp_count:cu.count,cusp_widths:cu.widths,elliptic_orders:modularEllipticOrders(r)});}
    for(const r of modularSubgroupCensus)out.push({...r,family:r.congruence===true?'congruence census':'subgroup census',exact:true,verified:true,census:true,row:r,branchPartitions:r.cycle_partitions});
    for(const r of (lowIndexSubgroups[g.id]||[]))out.push({...r,family:'rooted conjugate',exact:true,verified:true,lowIndex:true,row:r,branchPartitions:r.cycle_partitions});
  }else{
    for(const r of (genusById.get(g.id)||[])){
      const e=r.exact_drv_permutation_data,repN=e?.available?(e.representatives?.length||0):0,ideal=levelIdealData(r);
      if(repN){for(let i=0;i<repN;i++)out.push({key:`g1:${r.priority_rank_v4}:${i}`,family:r.cover_type||'X',p:r.rational_level_prime,q:r.residue_field_size_q,degree:r.belyi_degree,genus:r.genus,exact:true,row:r,rep:i,ideal,branchPartitions:e.representatives[i]?.cycle_partitions||r.branch_partitions,monodromy:r.monodromy})}
      else out.push({key:`g1:${r.priority_rank_v4}`,family:r.cover_type||'X',p:r.rational_level_prime,q:r.residue_field_size_q,degree:r.belyi_degree,genus:r.genus,exact:false,verified:true,row:r,ideal,branchPartitions:r.branch_partitions,monodromy:r.monodromy})
    }
    for(const r of (triLow[g.id]||[]))if(!out.some(x=>x.family===r.family&&x.p===r.p&&x.degree===r.degree&&x.genus===r.genus))out.push({...r,key:r.key||`tri:${r.family}:${r.p}:${r.degree}`,exact:false,verified:true});
    for(const r of (lowById.get(g.id)||[]))for(const gg of (r.possible_projective_X0_genera||[]))if(gg<=2&&!out.some(x=>x.family==='X0'&&x.p===r.p&&x.degree===r.x0_degree_q_plus_1&&x.genus===gg))out.push({key:`cand:${r.p}:${r.prime_index_above_p}:${gg}`,family:'X0',p:r.p,q:r.q_norm,degree:r.x0_degree_q_plus_1,genus:gg,exact:false,verified:false,row:r,ideal:(r.prime_ideal?{norm:r.q_norm,generator:r.prime_ideal}:null)});
    for(const r of (higherGenus[g.id]||[]))out.push({...r,higher:true,verified:true,row:r});
    for(const r of (lowIndexSubgroups[g.id]||[]))out.push({...r,family:'low-[·]=subgroup',exact:true,verified:true,lowIndex:true,row:r,branchPartitions:r.cycle_partitions})
  }
  const rows=out.sort((a,b)=>(Number(a.degree??1e12)-Number(b.degree??1e12))||(a.genus-b.genus)||String(a.label||a.family).localeCompare(String(b.label||b.family)));coverRecordCache.set(g.id,rows);return rows
}
function findCover(key){return coverRecords(current).find(r=>r.key===key)||null}
function kleinPoint(z){const d=1+abs2(z);return C(2*z.x/d,2*z.y/d)}
function poincareFromKlein(k){const r2=Math.min(.999999999,abs2(k)),d=1+Math.sqrt(Math.max(1e-12,1-r2));return C(k.x/d,k.y/d)}
function safeGeodesicMid(a,b,t=.5){const A=kleinPoint(a),B=kleinPoint(b);return poincareFromKlein(add(sc(A,1-t),sc(B,t)))}
function regionSamplePoints(R){
  const pts=[];for(const ch of R.chambers||[]){const z=chamberCenter(ch.A);if(Number.isFinite(z.x+z.y)&&abs2(z)<.999*.999)pts.push(z)}
  for(const e of R.boundary||[])for(const t of [.28,.5,.72]){const z=safeGeodesicMid(e.a,e.b,t);if(Number.isFinite(z.x+z.y)&&abs2(z)<.998*.998)pts.push(z)}return pts
}
function robustFocus(regions){
  let pts=regions.flatMap(regionSamplePoints);if(!pts.length)return regions[0]?.center||C();if(pts.length>72)pts=pts.filter((_,i)=>i%Math.ceil(pts.length/72)===0).slice(0,72);
  const cand=pts.length>28?pts.filter((_,i)=>i%Math.ceil(pts.length/28)===0):pts;let best=cand[0],score=Infinity;
  for(const c of cand){const ds=pts.map(p=>hdist(c,p)).filter(Number.isFinite).sort((a,b)=>a-b);if(!ds.length)continue;const q=ds[Math.floor(.68*(ds.length-1))]+.08*hdist(c,C());if(q<score){score=q;best=c}}return best
}
function regionScreenBBox(R){const ps=[];for(const z of regionSamplePoints(R)){const p=screenWorld(z);if(Number.isFinite(p.x+p.y))ps.push(p)}if(!ps.length)return null;const xs=ps.map(p=>p.x),ys=ps.map(p=>p.y);return{x0:Math.min(...xs),x1:Math.max(...xs),y0:Math.min(...ys),y1:Math.max(...ys),w:Math.max(...xs)-Math.min(...xs),h:Math.max(...ys)-Math.min(...ys)}}
function fitRegions(regions){
  const focus=robustFocus(regions);camera=trans(focus);camera.anti=false;camera.M=normM(camera.M);if(!Number.isFinite(camera.M.a.x+camera.M.a.y+camera.M.b.x+camera.M.b.y))camera=iso();
  const rr=[];for(const R of regions)for(const z of regionSamplePoints(R)){const r=absv(cameraApply(z));if(Number.isFinite(r)&&r<.9995)rr.push(r)}rr.sort((a,b)=>a-b);const rq=rr.length?rr[Math.min(rr.length-1,Math.floor(.72*(rr.length-1)))]:.55;
  zoom=Math.max(.92,Math.min(3.05,.70/Math.max(.19,rq)));resetEuclideanView();invalidate(true)
}
function fitFordRegion(R){
  if(!R)return;
  if(model==='half'){
    const z=R.centerH||C(0,1),t=-z.x,scl=1/Math.max(.18,z.y),T=iso({a:C(-t,2),b:C(t,0),c:C(-t,0),d:C(t,2)},false),S=iso({a:C(scl+1),b:C(scl-1),c:C(scl-1),d:C(scl+1)},false);
    camera=isoCompose(S,T);zoom=1.22;
  }else{camera=trans(R.center||C());zoom=1.38}
  camera.anti=false;camera.M=normM(camera.M);resetEuclideanView();
  const box=regionScreenBBox(R),g=geom();
  if(box&&box.w>1&&box.h>1){
    const tx=g.availW/2,ty=H/2,cx=(box.x0+box.x1)/2,cy=(box.y0+box.y1)/2;
    if(Math.abs(tx-cx)>2||Math.abs(ty-cy)>2)panPixels(cx-tx,cy-ty);
    const box2=regionScreenBBox(R);
    if(box2&&box2.w>1&&box2.h>1){
      const fx=Math.min(1.7,Math.max(.82,(g.availW*.72)/box2.w)),fy=Math.min(1.7,Math.max(.82,(H*.72)/box2.h));
      zoom=Math.max(.5,Math.min(4.2,zoom*Math.min(fx,fy)));
    }
  }
  invalidate(true)
}
function fitCurrentRegion(){if(frickeCompare)fitFordRegion(frickeCompare.plus);else fitRegions(commCompare?[commCompare.a.R,commCompare.b.R]:[currentRegion()])}
function centerOnCurrentRegion(){fitCurrentRegion()}
function selectCover(r){if(frickeCompare||commCompare||commSceneActive)restoreCurrentScene();selectedMoonshine=null;selectedOrbifoldPoint=null;selectedCover=r||null;if(!r||coverTriple(r))fitCurrentRegion();invalidate();if(panel==='covers')showPanel('covers')}
function modularDirect(labels,dir,map){const cand=labels.map(x=>map.get(x)).filter(Boolean);return cand.filter(x=>!cand.some(y=>y!==x&&((dir==='sub'?(y.row.subgroups||[]):(y.row.supergroups||[])).includes(x.label))))}
function modularNeighborhood(r){const all=coverRecords(current),map=new Map(all.filter(x=>x.modular).map(x=>[x.label,x]));if(!r){const root=map.get('1A 0');return{center:null,sup:[],sub:root?modularDirect(root.row.subgroups||[],'sub',map):[]}}return{center:r,sup:modularDirect(r.row.supergroups||[],'sup',map),sub:modularDirect(r.row.subgroups||[],'sub',map)}}
function genericNeighborhood(r,rows){
  const geom=rows.filter(x=>coverTriple(x));if(!r){const minDeg=Math.min(...geom.map(x=>x.degree),Infinity);return{center:null,sup:[],sub:geom.filter(x=>x.degree===minDeg).slice(0,8)}}
  const sup=[],sub=[];for(const q of rows){if(q.key===r.key)continue;if(q.p===r.p){if(r.family==='X1'&&q.family==='X0'&&q.degree<=r.degree)sup.push(q);if(r.family==='X0'&&q.family==='X1'&&q.degree>=r.degree)sub.push(q)}}return{center:r,sup:sup.slice(0,12),sub:sub.slice(0,18)}
}
function coverNumberMap(rows){const a=(rows||[]).slice().sort((x,y)=>Number(x.degree??Infinity)-Number(y.degree??Infinity)||Number(x.genus??Infinity)-Number(y.genus??Infinity)||String(coverLabel(x)).localeCompare(String(coverLabel(y)))||String(x.key).localeCompare(String(y.key))),m=new Map();a.forEach((r,i)=>m.set(r.key,i+1));return m}
function coverNumber(r,map=window.__coverNumbers){return r?.key&&map?.get(r.key)||null}
function coverNumberBadge(r,map=window.__coverNumbers){const n=coverNumber(r,map);return n?`<span class="object-number">${n}</span>`:''}
function coverRow(r,rel=null){const can=!!coverTriple(r),verified=r.exact||r.verified,q=rel||(!selectedCover?normalityBetween(r,null):null);return `<button class="cover-row ${can?'exact':''} ${selectedCover?.key===r.key?'selected':''}" data-cover="${esc(r.key)}">${coverNumberBadge(r)}<span class="cover-dot">${can?'●':(verified?'◦':'○')}</span><span>${esc(coverLabel(r))}</span><span class="cover-meta">${r.degree??'—'}</span><span class="cover-meta">g=${r.genus}</span>${normalityHTML(q)}</button>`}
const subgroupTreeCache=new Map();
function lowIndexClassGroups(rows){
  const census=rows.filter(r=>r.census||r.lowIndex),map=new Map();
  for(const r of census){const k=r.conjugacy_class===false?(r.rooted_from||r.key):r.key;if(!map.has(k))map.set(k,{key:k,rep:null,variants:[]});const q=map.get(k);if(r.conjugacy_class===false)q.variants.push(r);else if(!q.rep)q.rep=r}
  // A rooted row can refer to a canonical modular census row loaded earlier.
  for(const q of map.values())if(!q.rep)q.rep=census.find(r=>r.key===q.key)||q.variants[0]||null;
  return [...map.values()].filter(q=>q.rep).sort((a,b)=>(a.rep.degree-b.rep.degree)||(a.rep.genus-b.rep.genus)||String(a.rep.key).localeCompare(String(b.rep.key)))
}
function equivariantCoverMap(child,parent){
  const C=coverTriple(child),P=coverTriple(parent);if(!C||!P)return false;const n=C[0].length,m=P[0].length;if(n<=m||n%m)return false;const c=C.map(p=>p.map(x=>x-1)),p=P.map(q=>q.map(x=>x-1));
  for(let root=0;root<m;root++){const f=new Array(n).fill(-1),Q=[0];f[0]=root;let ok=true;for(let qi=0;qi<Q.length&&ok;qi++){const u=Q[qi];for(let j=0;j<3;j++){const v=c[j][u],w=p[j][f[u]];if(f[v]<0){f[v]=w;Q.push(v)}else if(f[v]!==w){ok=false;break}}}if(ok&&Q.length===n)return true}return false
}
const normalityCache=new Map();
function permInv0(p){const q=new Array(p.length);for(let i=0;i<p.length;i++)q[p[i]]=i;return q}
function permCompose0(p,q){const r=new Array(p.length);for(let i=0;i<p.length;i++)r[i]=q[p[i]];return r}
function permOrder0(p){const seen=new Uint8Array(p.length);let o=1;const lcm=(a,b)=>a/gcd(a,b)*b;for(let i=0;i<p.length;i++)if(!seen[i]){let j=i,n=0;while(!seen[j]){seen[j]=1;n++;j=p[j]}if(n)o=lcm(o,n)}return o}
function permClosure0(gens,limit=6000,stopAbove=Infinity){const n=gens[0]?.length||1,I=Array.from({length:n},(_,i)=>i),key=p=>p.join(','),all=[...gens,...gens.map(permInv0)],M=new Map([[key(I),I]]),Q=[I];for(let qi=0;qi<Q.length;qi++)for(const g of all){const z=permCompose0(Q[qi],g),k=key(z);if(!M.has(k)){M.set(k,z);Q.push(z);if(Q.length>stopAbove)return Q;if(Q.length>limit)return null}}return Q}
function quotientGroupTex(G){const n=G.length;if(n===1)return'1';const ord=G.map(permOrder0),max=Math.max(...ord),abelian=G.every(a=>G.every(b=>permCompose0(a,b).join(',')===permCompose0(b,a).join(','))),cnt=k=>ord.filter(x=>x===k).length;if([2,3,5,7,11,13,17,19,23].includes(n))return`C_{${n}}`;if(n===4)return max===4?'C_{4}':'V_{4}';if(n===6)return abelian?'C_{6}':'S_{3}';if(n===8){if(abelian)return max===8?'C_{8}':(max===2?'C_{2}^{3}':'C_{4}\\times C_{2}');return cnt(2)===1?'Q_{8}':'D_{4}'}if(n===9)return max===9?'C_{9}':'C_{3}^{2}';if(n===10)return abelian?'C_{10}':'D_{5}';if(n===12){if(abelian)return max===12?'C_{12}':'C_{6}\\times C_{2}';if(max===3)return'A_{4}';return cnt(4)?'\\mathrm{Dic}_{3}':'D_{6}'}if(n===14)return abelian?'C_{14}':'D_{7}';if(n===15)return'C_{15}';if(abelian&&max===n)return`C_{${n}}`;return`|Q|=${n}`}
function normalityBetween(child,parent=null){
  if(!child)return{status:'unknown'};if(child.modular&&parent?.modular){const z=inclusionNormality[`${parent.label}|${child.label}`];if(z)return z}
  const ck=`${child.key||child.label||'H'}|${parent?.key||parent?.label||'Γ'}`;if(normalityCache.has(ck))return normalityCache.get(ck);
  const Ct=coverTriple(child),Pt=parent?coverTriple(parent):[[1],[1],[1]];if(!Ct||!Pt){const z={status:'unknown'};normalityCache.set(ck,z);return z}const C=Ct.map(p=>p.map(x=>x-1)),P=Pt.map(p=>p.map(x=>x-1)),n=C[0].length,m=P[0].length;if(n<m||n%m){const z={status:'unknown'};normalityCache.set(ck,z);return z}
  let witness=null;for(let root=0;root<m&&!witness;root++){const f=new Array(n).fill(-1),Q=[0];f[0]=root;let ok=true;for(let qi=0;qi<Q.length&&ok;qi++){const u=Q[qi];for(let j=0;j<3;j++){const v=C[j][u],w=P[j][f[u]];if(f[v]<0){f[v]=w;Q.push(v)}else if(f[v]!==w){ok=false;break}}}if(ok&&f.every(x=>x>=0))witness={f,root}}
  if(!witness){const z={status:'unknown'};normalityCache.set(ck,z);return z}const {f,root}=witness,ratio=n/m,Pi=P.map(permInv0),Ci=C.map(permInv0),word=new Array(m).fill(null);word[root]=[];const QQ=[root];for(let qi=0;qi<QQ.length;qi++){const y=QQ[qi];for(let j=0;j<3;j++)for(const [sgn,A] of [[1,P],[-1,Pi]]){const z=A[j][y];if(word[z]==null){word[z]=word[y].concat([[j,sgn]]);QQ.push(z)}}}if(word.some(x=>x==null)){const z={status:'unknown'};normalityCache.set(ck,z);return z}
  const fiber=[];for(let i=0;i<n;i++)if(f[i]===root)fiber.push(i);const idx=new Map(fiber.map((x,i)=>[x,i])),applyWord=(x,w)=>{for(const[j,sgn]of w)x=(sgn>0?C:Ci)[j][x];return x},gens=[];
  for(let y=0;y<m;y++)for(let j=0;j<3;j++){const yp=P[j][y],w=word[y].concat([[j,1]],word[yp].slice().reverse().map(([a,b])=>[a,-b])),perm=[];let ok=true;for(const x of fiber){const z=applyWord(x,w);if(!idx.has(z)){ok=false;break}perm.push(idx.get(z))}if(ok&&perm.some((v,i)=>v!==i)&&!gens.some(g=>g.join(',')===perm.join(',')))gens.push(perm)}
  const G=permClosure0(gens,6000,ratio);if(!G){const z={status:'unknown_large',index:ratio};normalityCache.set(ck,z);return z}const normal=G.length===ratio,z={status:normal?'normal':'not_normal',index:ratio};if(normal)z.quotient_tex=quotientGroupTex(G);normalityCache.set(ck,z);return z
}
function normalityHTML(q){if(!q||q.status==='unknown'||q.status==='no_map'||q.status==='unknown_large')return'';return q.status==='normal'?String.raw`<span class="cover-meta normality">\(\triangleleft\;${q.quotient_tex||''}\)</span>`:String.raw`<span class="cover-meta normality">\(\not\triangleleft\)</span>`}
function quotientUnicode(t){return String(t||'').replace(/_\{(\d+)\}/g,(_,n)=>n.split('').map(c=>'₀₁₂₃₄₅₆₇₈₉'[+c]).join('')).replace(/\^\{(\d+)\}/g,(_,n)=>'^'+n).replace(/\\times/g,'×').replace(/\\mathrm\{([^}]+)\}/g,'$1')}

function subgroupClassTree(g,groups,numberMap=window.__coverNumbers){
  const cacheKey=g.id+'|v27n|'+groups.map(q=>`${q.rep.key}:${numberMap?.get(q.rep.key)||0}`).join(',');if(subgroupTreeCache.has(cacheKey))return subgroupTreeCache.get(cacheKey);
  const nodes=[{id:'__gamma__',label:'Γ',degree:1,genus:0,rep:null}].concat(groups.map((q,i)=>({id:q.rep.key,label:`${q.rep.degree}:${i+1}`,degree:Number(q.rep.degree),genus:Number(q.rep.genus||0),rep:q.rep,group:q}))),N=nodes.length,index=new Map(nodes.map((n,i)=>[n.id,i]));
  const contains=Array.from({length:N},()=>new Uint8Array(N));for(let j=1;j<N;j++)contains[0][j]=1;
  for(let i=1;i<N;i++)for(let j=1;j<N;j++)if(i!==j&&nodes[j].degree>nodes[i].degree&&nodes[j].degree%nodes[i].degree===0&&equivariantCoverMap(nodes[j].rep,nodes[i].rep))contains[i][j]=1;
  const edges=[];for(let i=0;i<N;i++)for(let j=1;j<N;j++)if(contains[i][j]){let direct=true;for(let k=1;k<N;k++)if(k!==i&&k!==j&&nodes[k].degree>nodes[i].degree&&nodes[k].degree<nodes[j].degree&&contains[i][k]&&contains[k][j]){direct=false;break}if(direct)edges.push({a:i,b:j,ratio:nodes[j].degree/nodes[i].degree,norm:normalityBetween(nodes[j].rep,nodes[i].rep)})}
  const parents=Array.from({length:N},()=>[]),children=Array.from({length:N},()=>[]);for(const e of edges){parents[e.b].push(e.a);children[e.a].push(e.b)}
  const levels=[...new Set(nodes.map(n=>n.degree))].sort((a,b)=>a-b),byLevel=new Map(levels.map(d=>[d,nodes.filter(n=>n.degree===d).sort((A,B)=>A.genus-B.genus||A.label.localeCompare(B.label))]));
  const rank=new Map();for(const d of levels)byLevel.get(d).forEach((n,i)=>rank.set(n.id,i));
  const score=(n,adj)=>{const a=adj[index.get(n.id)]||[];if(!a.length)return rank.get(n.id)??0;return a.reduce((z,j)=>z+(rank.get(nodes[j].id)??0),0)/a.length};
  // Alternating barycentric sweeps reduce crossings substantially while keeping
  // every [·]=level compact and deterministic.
  for(let it=0;it<7;it++){
    for(let li=1;li<levels.length;li++){const arr=byLevel.get(levels[li]);arr.sort((A,B)=>score(A,parents)-score(B,parents)||A.genus-B.genus||A.label.localeCompare(B.label));arr.forEach((n,i)=>rank.set(n.id,i))}
    for(let li=levels.length-2;li>=0;li--){const arr=byLevel.get(levels[li]);arr.sort((A,B)=>score(A,children)-score(B,children)||A.genus-B.genus||A.label.localeCompare(B.label));arr.forEach((n,i)=>rank.set(n.id,i))}
  }
  const maxN=Math.max(...levels.map(d=>byLevel.get(d).length)),w=Math.max(580,maxN*54+42),ygap=56,h=Math.max(140,levels.length*ygap+30),pos=new Map();levels.forEach((d,li)=>{const arr=byLevel.get(d),gap=w/(arr.length+1);arr.forEach((n,i)=>pos.set(n.id,{x:gap*(i+1),y:25+li*ygap}))});
  let svg=`<svg class="math-graph subgroup-lattice" viewBox="0 0 ${w} ${h}" style="width:${w}px;height:${h}px">`;
  edges.forEach((e,ei)=>{const A=nodes[e.a],B=nodes[e.b],a=pos.get(A.id),b=pos.get(B.id),ym=(a.y+b.y)/2+((ei%5)-2)*1.4;svg+=`<path class="sub-edge" d="M${a.x},${a.y+7} C${a.x},${ym} ${b.x},${ym} ${b.x},${b.y-8}"/>`});
  for(const n of nodes){const p=pos.get(n.id);if(n.id==='__gamma__')svg+=`<g><circle class="node selected" cx="${p.x}" cy="${p.y}" r="6.5"/><text class="node-number" text-anchor="middle" x="${p.x}" y="${p.y-10}">0</text><text text-anchor="middle" x="${p.x}" y="${p.y+18}">Γ</text></g>`;else{const sel=selectedCover?.key===n.rep.key,fill=n.genus===0?'rgba(40,115,76,.10)':'rgba(75,78,92,.07)',no=numberMap?.get(n.rep.key);svg+=`<g data-cover="${esc(n.rep.key)}" class="graph-hit"><circle class="node ${sel?'selected':''}" style="fill:${fill}" cx="${p.x}" cy="${p.y}" r="${sel?7:5.5}"/>${no?`<text class="node-number" text-anchor="middle" x="${p.x}" y="${p.y-9}">${no}</text>`:''}<text text-anchor="middle" x="${p.x}" y="${p.y+17}">${n.degree}:${n.genus}</text><title>${no?`#${no} · `:''}${esc(coverLabel(n.rep))} · [·]=${n.degree} · g=${n.genus}</title></g>`}}
  svg+='</svg>';const out={svg,nodes,edges};subgroupTreeCache.set(cacheKey,out);return out
}
function subgroupClassDetails(groups,numberMap=window.__coverNumbers){
  let out='<div class="subgroup-class-list">';for(const q of groups){const r=q.rep,ell=(r.elliptic_orders||[]).join(',')||'—',cus=(r.cusp_widths||[]).join(',')||'—',parts=(r.cycle_partitions||triplePartitions(coverTriple(r))||[]).map(x=>'('+x.join(',')+')').join(' / '),vars=q.variants||[],nr=normalityBetween(r,null),nt=nr.status==='normal'?` · ◁ ${quotientUnicode(nr.quotient_tex)}`:(nr.status==='not_normal'?' · ⋪':''),rn=numberMap?.get(r.key);out+=`<details class="subgroup-class"><summary><span>${rn?`#${rn} · `:''}[Γ:H]=${r.degree} · g=${r.genus}${nt}</span><small>${esc(ell)} ; ∞ ${esc(cus)}${vars.length?` · ×${vars.length+1}`:''}</small></summary><div class="class-meta">${esc(parts||'—')}</div><div class="class-actions"><button data-cover="${esc(r.key)}" class="class-cover ${selectedCover?.key===r.key?'selected':''}">${rn?`#${rn}`:'H₁'}</button>${vars.slice(0,36).map((v,i)=>{const vn=numberMap?.get(v.key);return `<button data-cover="${esc(v.key)}" class="class-cover ${selectedCover?.key===v.key?'selected':''}">${vn?`#${vn}`:`H${i+2}`}</button>`}).join('')}</div></details>`}return out+'</div>'
}

function idealTex(r){if(!r?.ideal)return'';const q=r.ideal;let sub=r.p?String(r.p):'p';return `\\mathfrak p_{${sub}}\\mid ${sub},\\quad N\\mathfrak p=${q.norm??r.q??'?'}`+(q.e&&q.f?`,\\quad(e,f)=(${q.e},${q.f})`:'')}
function cpLabelCompact(s){return String(s||'').replace(/\s+/g,'')}
function groupNameTex(s){return String(s||'').replace(/SL2\(Z\)/g,'\\mathrm{SL}_{2}(\\mathbf Z)').replace(/Γ_?0\((\d+)\)/g,'\\Gamma_{0}($1)').replace(/Γ_?1\((\d+)\)/g,'\\Gamma_{1}($1)').replace(/Γ\((\d+)\)/g,'\\Gamma($1)').replace(/Γ\^(\d+)/g,'\\Gamma^{$1}').replace(/Gamma_?0\((\d+)\)/g,'\\Gamma_{0}($1)').replace(/Gamma_?1\((\d+)\)/g,'\\Gamma_{1}($1)').replace(/Gamma\((\d+)\)/g,'\\Gamma($1)').replace(/Gamma\^(\d+)/g,'\\Gamma^{$1}').replace(/Γ|Gamma/g,'\\Gamma')}
function modularStatusKind(s){s=String(s||'').toLowerCase();return s.startsWith('exact')?'exact':(s.includes('candidate')||s.includes('likely')?'candidate':'numeric')}
function asciiJTex(raw){return String(raw||'').replace(/sqrt\((-?\d+)\)/g,'\\sqrt{$1}').replace(/\*/g,'\\,').replace(/\^(-?\d+)/g,'^{$1}')}
function matrixGroupTex(label){const p=modularPerm[label];if(!p?.matrix_generators?.length)return'';const mats=p.matrix_generators.slice(0,6).map(m=>String.raw`\begin{pmatrix}${m[0]}&${m[1]}\\${m[2]}&${m[3]}\end{pmatrix}`).join(String.raw`,\,`);return String.raw`\Gamma_H=\rho_{${p.level}}^{-1}\!\left(\left\langle ${mats}\right\rangle\right)\leq\mathrm{SL}_2(\mathbf Z)`}
function genus0JTex(label,varname){const z=cpGenus0[label];if(!z)return null;const gh=z.gamma0?gamma0Haupt[String(z.gamma0)]:null,raw=gh?.j_tex||z.j_tex;if(!raw)return null;return String(raw).replace(/\bh\b/g,varname).replace(/\bt\b/g,varname)}
function cpAsciiTex(raw){return exprTex(String(raw||'').replace(/zeta_(\d+)/g,'\\zeta_{$1}').replace(/q\^\(1\/(\d+)\)/g,'q^{1/$1}').replace(/u\^1\b/g,'u').replace(/Qbar/g,'\\overline{\\mathbf Q}'))}
function cpQSeriesHTML(label){const q=cpQSeries60[label]||cpQSeries20[label];if(!q)return'';if(/not yet stable|not asserted/i.test(q.series||''))return `<div class="source-note">local q-series: ${esc(q.series.replace(/`/g,''))}</div>`;const deg=cpQSeries60[label]?60:20;return String.raw`<details class="math-details"><summary>\(q\)-${tr('qSeries')} · ${deg}</summary><div class="mathline tiny formula-scroll">\[${cpAsciiTex(q.coordinate)}.\]</div><div class="mathline tiny formula-scroll">\[t_H=${cpAsciiTex(q.series)}.\]</div><div class="source-note">${esc(q.status)} · normalized by [u^-1]t=1 and [u^0]t=0.</div></details>`}
function exactHauptRelation(a,b){return v20HauptRelations[a+'|'+b]||v20HauptRelations[b+'|'+a]||gamma0Relations[a+'|'+b]||gamma0Relations[b+'|'+a]||null}
function adjacentHauptmodulHTML(r,z0){const lab=r.label,neigh=[...(r.row.supergroups||[]),...(r.row.subgroups||[])].filter(y=>cpGenus0[y]);if(!neigh.length&&!supplementalHauptRelations[lab])return'';let out=`<details class="math-details"><summary>Hauptmodul · ${tr('relations')}</summary>`;
  const JH=genus0JTex(lab,'t_H');for(const y of neigh.slice(0,24)){const direct=exactHauptRelation(lab,y);if(direct){const pl=direct.parent,cl=direct.child,map=String(direct.map_tex).replace(/\bu\b/g,`t_{${cl}}`),phi=String(direct.relation_tex||'').replace(/\bh\b/g,`t_{${pl}}`).replace(/\bu\b/g,`t_{${cl}}`);out+=String.raw`<div class="mathline tiny formula-scroll">\[t_{${esc(pl)}}=${map},\qquad ${esc(pl)}\supset ${esc(cl)}.\]</div>`;if(phi)out+=String.raw`<div class="mathline tiny formula-scroll">\[\Phi(t_{${esc(pl)}},t_{${esc(cl)}})=${phi}=0.\]</div>`;if(direct.field_tex){out+=String.raw`<div class="mathline tiny formula-scroll">\[${direct.field_tex}.\]</div>`;out+=fieldGeneratorHTML(direct.field_tex,direct)}if(direct.status)out+=`<div class="source-note">${esc(direct.status)}</div>`;continue}const JG=genus0JTex(y,'t_G');if(JH&&JG)out+=String.raw`<div class="mathline tiny formula-scroll">\[(${JH})-(${JG})=0,\qquad ${esc(lab)}\leftrightarrow ${esc(y)}.\]</div><div class="source-note">exact j-fibre equation; it is not labelled as the minimal covering component without a verified factor.</div>`;else out+=`<div class="source-note">${esc(y)} · ${esc(cpGenus0[y].description||cpGenus0[y].hauptmodul||'exact CP genus-0 coordinate')}</div>`}
  const sp=supplementalHauptRelations[lab];if(sp){out+=String.raw`<div class="divider"></div><div class="mathline tiny formula-scroll">\[U=${String(sp.map_tex).replace(/\bu\b/g,'t_H')},\qquad ${sp.field_tex||''}.\]</div><div class="source-note">${esc(sp.target)} · ${esc(sp.status)}</div>`;out+=fieldGeneratorHTML(sp.field_tex||'',sp)}
  return out+'<div class="source-note">Explicit rational/polynomial covering maps are preferred. Unfactored j-fibre equations are retained only as certificates and are not silently promoted to minimal relations.</div></details>'}
function moonshineSeriesTex(name,maxN=60,variable='q'){const a=moonshine.records?.[name];if(!a)return'';const q=String(variable||'q'),qn=n=>n===1?q:`${q}^{${n}}`;let t=`${q}^{-1}`;for(let n=1;n<=Math.min(maxN,a.length);n++){const raw=String(a[n-1]??'0');if(raw==='0')continue;const neg=raw[0]==='-',v=neg?raw.slice(1):raw,p=qn(n),coef=v==='1'?'':v;t+=(neg?' - ':' + ')+coef+p}return t+`+O(${q}^{${Math.min(maxN,a.length)+1}})`}
function moonshineHTML(label){
  const m=moonshine.gamma0Matches?.[label],matches=moonshine.cpSeriesMatches?.[label]||[];if(!m&&!matches.length)return'';
  let x='<div class="divider"></div><div class="relation-head">Moonshine</div>';
  if(m){const name=m.monster_class,N=m.level,k=m.phase_k||0,phase=N>1?(k?`\\zeta_{${N}}^{${k}}`:'1'):'1';x+=String.raw`<div class="mathline small formula-scroll">\[\Gamma_{${esc(name)}}=\Gamma_{0}(${N})\quad\mathrm{std},\qquad T_{${esc(name)}}(q)=${moonshineSeriesTex(name,12)}.\]</div>`;x+=String.raw`<details class="math-details"><summary>\(T_g(q)\) · \(q^{60}\)</summary><div class="mathline tiny formula-scroll">\[T_{${esc(name)}}(q)=${moonshineSeriesTex(name,60)}.\]</div></details>`;if(label==='1A 0')x+=String.raw`<div class="mathline tiny formula-scroll">\[t_{1A}=T_{1A}=j-744.\]</div>`;else{x+=String.raw`<div class="mathline tiny formula-scroll">\[t_{\mathrm{CP}}(u)=${phase}\,T_{${esc(name)}}\!\left(${phase}u\right).\]</div>`;const z=cpGenus0[label],gh=z?.gamma0?gamma0Haupt[String(z.gamma0)]:null;if(gh?.j_tex){const c=String(gh.constant||'0'),hvar=`\\left(T_{${name}}${c.startsWith('-')?c:'+'+c}\\right)`,jr=String(gh.j_tex).replace(/\bh\b/g,hvar);x+=String.raw`<div class="mathline tiny formula-scroll">\[j=${jr}.\]</div><div class="source-note">Exact algebraic relation between the matched moonshine Hauptmodul and \(j\) on the standard \(\Gamma_0(${N})\) curve.</div>`}}x+=`<div class="source-note">${esc(m.status)}${m.checked_terms?` · checked ${m.checked_terms} coefficients`:''}. The stored CP finite-matrix representative and the standard \(\Gamma_0\) representative remain distinguished.</div>`}
  const extra=matches.filter(z=>!m||z.monster_class!==m.monster_class||z.phase_k!==(m.phase_k||0));if(extra.length){x+='<details class="math-details"><summary>Moonshine · \\(q^{60}\\)</summary>';for(const z of extra.slice(0,10)){x+=String.raw`<div class="mathline tiny formula-scroll">\[${z.relation_tex}.\]</div><div class="source-note">\(\Gamma_{${esc(z.monster_class)}}\) moonshine Hauptmodul · ${esc(z.status)}</div>`}if(extra.length>10)x+=`<div class="source-note">${extra.length-10} additional phase-equivalent certificates are retained in the data bundle.</div>`;x+='</details>'}
  return x
}
function frickeRelationHTML(N){
  const r=frickeHauptRelations[String(N)];if(!r)return'';return String.raw`<div class="fricke-relation"><div class="mathline tiny formula-scroll">\[${r.map_tex},\qquad ${r.relation_tex}.\]</div><div class="source-note">\(\Gamma_0(${N})\subset\Gamma_0(${N})^+\) has [·]=2. ${esc(r.status)}</div><button class="class-cover" data-fricke="${N}">Ford</button></div>`
}
function frickeExtensionTreeHTML(Ns){
  if(!Ns?.length)return'';const cols=Math.min(8,Math.max(4,Math.ceil(Math.sqrt(Ns.length*1.6)))),cw=78,rh=58,w=cols*cw+24,rows=Math.ceil(Ns.length/cols),h=rows*rh+50;let svg=`<svg class="math-graph fricke-extension-tree" viewBox="0 0 ${w} ${h}" style="width:${w}px;height:${h}px">`;
  Ns.forEach((N,i)=>{const c=i%cols,r=Math.floor(i/cols),x=18+c*cw,y=24+r*rh,name=frickeClasses[String(N)]||`${N}A`;svg+=`<path class="fricke-ext-edge" d="M${x+25},${y+10} L${x+25},${y+31}"/><g class="graph-hit" data-fricke="${N}"><rect class="fricke-ext-node base" x="${x}" y="${y}" width="50" height="18" rx="3"/><text text-anchor="middle" x="${x+25}" y="${y+13}">Γ₀(${N})</text><rect class="fricke-ext-node plus ${frickeCompare?.N===N?'selected':''}" x="${x}" y="${y+32}" width="50" height="18" rx="3"/><text text-anchor="middle" x="${x+25}" y="${y+45}">${esc(name)}</text><title>Γ0(${N}) < Γ0(${N})+ · [·]=2 · ${esc(name)}</title></g>`});
  return `<div class="subgroup-tree-scroll fricke-extension-scroll">${svg}</div><div class="source-note">Dashed auxiliary extension lattice: each vertical edge is the index-2 extension \(\Gamma_{0}(N)\subset\Gamma_{0}(N)^{+}\). These moonshine nodes live in the commensurator and are therefore shown beside, rather than falsely inserted into, the PSL₂(ℤ) subgroup lattice.</div>`
}
function frickeCatalogueHTML(){
  const Ns=Object.keys(frickeDomains).map(Number).sort((a,b)=>a-b);if(!Ns.length)return'';const buttons=Ns.map(N=>{const name=frickeClasses[String(N)]||`${N}A`;return `<button class="fricke-domain-button ${frickeCompare?.N===N?'selected':''}" data-fricke="${N}"><span>${esc(name)}</span><small>Γ0(${N})+</small></button>`}).join('');
  const exact=Object.keys(frickeHauptRelations).map(Number).sort((a,b)=>a-b).map(N=>frickeRelationHTML(N)).join('');
  return String.raw`<details class="math-details" open><summary>Ford · ${Ns.length}</summary><div class="source-note">Independent Ford polygons are used here: their sides are not required to lie on the existing PSL₂(ℤ) chamber skeleton. This permits exact simultaneous comparison of commensurable groups with genuinely new boundaries.</div>${frickeExtensionTreeHTML(Ns)}<div class="fricke-domain-grid">${buttons}</div>${exact?`<details class="math-details"><summary>Hauptmodul · \\(\\Gamma_0(N)\\leftrightarrow\\Gamma_0(N)^+\\)</summary>${exact}</details>`:''}</details>`
}
function moonshineCatalogueHTML(){if(!moonshine.names?.length)return'';const gm=moonshine.gamma0Matches||{},exact=Object.entries(gm).sort((a,b)=>Number(a[1].level)-Number(b[1].level)||a[0].localeCompare(b[0]));let rows='';for(const [cp,m] of exact)rows+=String.raw`<div class="moonshine-row"><span>\(\Gamma_{${esc(m.monster_class)}}\)</span><span>\(T_{${esc(m.monster_class)}}=q^{-1}+O(q)\)</span><span>\(\Gamma_0(${m.level})\)</span><small>${esc(cp)}</small></div>`;return frickeCatalogueHTML()+String.raw`<details class="math-details moonshine-catalogue"><summary>Moonshine · \(T_g\) · ${moonshine.names.length}</summary><div class="source-note">Bundled PFTool v4.5 McKay–Thompson database. Every entry represents a moonshine genus-zero group \(\Gamma_g\) with normalized Hauptmodul \(T_g=q^{-1}+O(q)\), stored through \(q^{60}\) in v20. Group equality with a congr. subgroup is asserted only where separately certified; other CP links are labelled coefficient identities.</div>${rows?`<div class="relation-head">\\(\\Gamma_g\\cong H\\)</div><div class="moonshine-rel-grid">${rows}</div>`:''}<div class="moonshine-list">${moonshine.names.map(n=>`<span>${esc(n)}</span>`).join('')}</div></details>`}
// -----------------------------------------------------------------------------
// v23: complete monstrous-Moonshine geometry / inclusion lattice
// -----------------------------------------------------------------------------
const moonFordCache=new Map(), moonRelByPair=new Map(moonshineRelations.map(r=>[r.sub_symbol+'|'+r.super_symbol,r]));
function intGcd(a,b){a=Math.abs(Math.trunc(a));b=Math.abs(Math.trunc(b));while(b){const t=a%b;a=b;b=t}return a||1}
function gamma0IndexInt(n){let x=n,v=n;for(let p=2;p*p<=x;p+=(p===2?1:2))if(x%p===0){v=v*(p+1)/p;while(x%p===0)x=Math.trunc(x/p)}if(x>1)v=v*(x+1)/x;return v}
function moonFordRecord(meta){
  const key=meta.symbol;if(moonFordCache.has(key))return moonFordCache.get(key);
  const h=Number(meta.h||1),m=Number(meta.m||meta.N||1),als=(meta.al||[1]).map(Number),half=.5/h,n=1600,Cmax=56,width=1/h;
  const xs=Array.from({length:n+1},(_,i)=>-half+width*i/n),ys=new Float64Array(n+1),act=new Int32Array(n+1),cs=[];act.fill(-1);
  function addCircle(c,r,e,den,d){const id=cs.length;cs.push([c,r,e,den,d]);let i0=Math.max(0,Math.floor((c-r+half)/width*n)-1),i1=Math.min(n,Math.ceil((c+r+half)/width*n)+1);for(let i=i0;i<=i1;i++){const q=r*r-(xs[i]-c)*(xs[i]-c);if(q<=0)continue;const y=Math.sqrt(q);if(y>ys[i]){ys[i]=y;act[i]=id}}}
  for(const e of als){const se=Math.sqrt(e),me=m/e;for(let c=1;c<=Cmax;c++){const r=se/(c*m*h),D=Math.ceil(c*m/(2*e)+1/se)+1;for(let d=-D;d<=D;d++){if(intGcd(d*e,c*me)!==1)continue;const cen=-(d*e)/(c*m*h);if(cen+r<-half-1e-12||cen-r>half+1e-12)continue;addCircle(cen,r,e,c,d)}}}
  const runs=[];let st=0;for(let i=1;i<=n+1;i++)if(i===n+1||act[i]!==act[st]){if(act[st]>=0)runs.push([st,i-1,act[st]]);st=i}
  const bx=[-half];for(let k=0;k<runs.length-1;k++){const A=cs[runs[k][2]],B=cs[runs[k+1][2]];let lo=xs[runs[k][1]],hi=xs[runs[k+1][0]],fl=A[1]*A[1]-(lo-A[0])**2-(B[1]*B[1]-(lo-B[0])**2);for(let j=0;j<45;j++){const md=(lo+hi)/2,fm=A[1]*A[1]-(md-A[0])**2-(B[1]*B[1]-(md-B[0])**2);if(fl*fm<=0)hi=md;else{lo=md;fl=fm}}bx.push((lo+hi)/2)}bx.push(half);
  const base=[];for(let k=0;k<runs.length;k++){const [c,r,e,den,d]=cs[runs[k][2]],x0=bx[k],x1=bx[k+1],yy=x=>Math.sqrt(Math.max(0,r*r-(x-c)*(x-c)));base.push({c,r,e,den,d,x0,y0:yy(x0),x1,y1:yy(x1)})}
  // The kernel cell is the genuine Schreier union of the eigengroup cells
  // T^(k/h)F_E, k=0,...,h-1. Do not visually recenter these cosets by
  // altering their geometry: fitFordRegion/camera motion centers the actual
  // current tile, preserving the true side-pairings.
  const seg=[];for(let cell=0;cell<h;cell++){const sh=cell/h;for(const q of base)seg.push({...q,baseC:q.c,shift:sh,c:q.c+sh,x0:q.x0+sh,x1:q.x1+sh,cell})}
  const verts=[];for(let i=0;i<seg.length;i++){const q=seg[i];if(i===0){const z=icy(C(q.x0,q.y0));verts.push([z.x,z.y])}const z=icy(C(q.x1,q.y1));verts.push([z.x,z.y])}verts.push([1,0]);
  const center_x=(h-1)/(2*h),domain_left=-half,domain_right=1-half;
  const rec={symbol:key,N:meta.N,h,m,al:als,segments:seg,vertices:verts,center_x,domain_left,domain_right,cells:h,expected_area:h*gamma0IndexInt(m)*Math.PI/(3*Math.max(1,als.length))};moonFordCache.set(key,rec);return rec
}
function moonSymbolTex(s){const m=String(s).match(/^(\d+)(?:\|\|(\d+))?(?:\+(.*))?$/);if(!m)return esc(s);let x=m[1],h=m[2],tail=m[3];let t=h?`${x}\\!\\parallel\\!${h}`:x;if(String(s).includes('+'))t+=tail?`+${tail.replace(/,/g,',\\,')}`:'+';return t}
function moonClassesForSymbol(s){return moonshineSymbols?.[s]?.classes||[]}
function moonExprTex(s,varTex){return polyTex(String(s??'').replace(/\*\*/g,'^').replace(/\bX\b/g,varTex).replace(/\bY\b/g,varTex))}
function moonRelationTex(r){if(!r?.relation)return'';const a=r.sub_classes?.[0]||'a',b=r.super_classes?.[0]||'b';let z=String(r.relation).replace(/\*\*/g,'^').replace(/\bX\b/g,`T_{${a}}`).replace(/\bY\b/g,`T_{${b}}`);return polyTex(z)}
function moonRationalMapTex(r){if(!r?.map_num)return'';const a=r.sub_classes?.[0]||'a',b=r.super_classes?.[0]||'b',v=`T_{${a}}`,num=moonExprTex(r.map_num,v),den=moonExprTex(r.map_den||'1',v);return den==='1'?`T_{${b}}=${num}`:`T_{${b}}=\\frac{${num}}{${den}}`}
function moonClassForSymbol(s){return moonClassesForSymbol(s)?.[0]||moonshineSymbols?.[s]?.classes?.[0]||''}
function moonNormalityTag(sub,superg){const q=moonshineNormality[sub+'|'+superg];if(!q)return'';return q.status==='normal'?` · ◁${quotientUnicode(q.quotient_tex)}`:(q.status==='not_normal'?' · ⋪':'')}
function moonNeighborhoodHTML(symbol){const adj=moonshineAdjacency[symbol]||{super:[],sub:[]},up=adj.super||[],dn=adj.sub||[];if(!up.length&&!dn.length)return '<div class="source-note">No additional same-q immediate inclusion was certified by the q^60 relation pass for this group; no speculative edge is drawn.</div>';const row=(arr,kind)=>arr.map(s=>{const cls=moonClassForSymbol(s),sub=kind==='super'?symbol:s,superg=kind==='super'?s:symbol,r=moonRelByPair.get(sub+'|'+superg),nt=moonNormalityTag(sub,superg);return `<button class="moon-neighbor" data-moonshine="${esc(cls)}"><span>${esc((moonshineSymbols[s]?.classes||[cls]).join('/'))}</span><small>${esc(s)} · [·]=${r?.degree||'?'}${nt}</small></button>`}).join('');return `<div class="moon-neighborhood">${up.length?`<div class="moon-neighbor-band"><b>${tr('supergroups')}</b><div>${row(up,'super')}</div></div>`:''}<div class="moon-current-node">${esc((moonshineSymbols[symbol]?.classes||[]).join('/'))}<small>${esc(symbol)}</small></div>${dn.length?`<div class="moon-neighbor-band"><b>${tr('subgroups')}</b><div>${row(dn,'sub')}</div></div>`:''}</div>`}
function moonshineModularFormsHTML(meta,name){
  const D=moonshineModularForms?.[meta?.symbol];if(!D)return'';
  const full=D.mode==='full_group_basis', h=Number(D.h||1), src=D.source_symbol||D.symbol;
  let out=`<details class="math-details" open><summary>${full?'\\(M_k(\\Gamma_g)\\)':'\\(M_k(\\Gamma_g)^{E_g/\\Gamma_g}\\)'} · \\(k=2,4,6\\)</summary>`;
  if(full){
    const gens=(D.al_generators||[]).join(', ');
    out+=`<div class="source-note">Exact PARI/GP simultaneous +1 eigenspaces inside M_k(Γ0(${D.N})) for the recorded Atkin–Lehner involutions${gens?` W_${gens.replace(/, /g, ', W_')}`:' (none)'}. Kernel intersections are performed with the exact mfatkininit action matrices, so groups with several involutions are treated simultaneously rather than one label at a time. The displayed basis is exact through q^60; it is an invariant basis, not asserted to be Hecke-diagonal.</div>`;
  }else{
    out+=String.raw`<div class="source-note">Here h=${h}>1 and Γ_${esc(name)}=ker σ_${esc(name)} has more modular forms than the trivial-character eigenspace alone. The rigorously available subspace shown here is inherited from E_${esc(name)}≅Γ0(${D.m}) with the same Atkin–Lehner set as ${esc(src)}: if f(q) is a form on that standard eigengroup, the displayed form is f(q^${h})=f(hτ). Thus every series below lies in M_k(Γ_${esc(name)}), but this panel deliberately does not call it a complete kernel basis.</div>`;
  }
  for(const k of ['2','4','6']){
    const z=D.weights?.[k];if(!z)continue;const d=Number(z.dimension||0);
    out+=`<details class="math-details"><summary>${full?`\\(M_{${k}}(\\Gamma_{${esc(name)}})\\)`:`\\(M_{${k}}(\\Gamma_{${esc(name)}})^{E_{${esc(name)}}/\\Gamma_{${esc(name)}}}\\)`} · ${d}</summary>`;
    if(!d){out+=String.raw`<div class="mathline tiny formula-scroll">\[${full?`M_{${k}}(\Gamma_{${esc(name)}})`:`M_{${k}}(\Gamma_{${esc(name)}})^{E_${esc(name)}/\Gamma_${esc(name)}}`}=0.\]</div>`}
    else z.basis.forEach((a,i)=>{out+=String.raw`<div class="mathline tiny formula-scroll">\[F_{${k},${i+1}}^{(${esc(name)})}(q)=${mfSeriesTex(a,60)}.\]</div>`});
    out+='</details>'
  }
  return out+'</details>'
}
function moonshineDetailHTML(name){const m=moonshineGroups[name];if(!m)return'';const symbol=m.symbol,same=moonClassesForSymbol(symbol),adj=moonshineAdjacency[symbol]||{super:[],sub:[]},rels=moonshineRelations.filter(r=>r.sub_symbol===symbol||r.super_symbol===symbol),direct=rels.filter(r=>r.immediate);let x=`<div class="moon-selected"><div class="relation-head">${esc(name)} </div>`;
  x+=String.raw`<div class="mathline small formula-scroll">\[\Gamma_{${esc(name)}}=${moonSymbolTex(symbol)},\qquad T_{${esc(name)}}(q)=q^{-1}+O(q).\]</div>`;
  x+=String.raw`<div class="mathline tiny formula-scroll">\[\Gamma_0(${m.N*m.h})\subseteq\Gamma_{${esc(name)}}\subseteq E_{${esc(name)}}=\Gamma_0(${m.N}\mid ${m.h})${m.al.length>1?`+${m.al.filter(e=>e!==1).join(',')}`:''},\qquad[E_{${esc(name)}}:\Gamma_{${esc(name)}}]=${m.h}.\]</div>`;
  if(m.h>1)x+=String.raw`<div class="mathline tiny formula-scroll">\[\Gamma_{${esc(name)}}\triangleleft E_{${esc(name)}},\qquad E_{${esc(name)}}/\Gamma_{${esc(name)}}\cong C_{${m.h}}.\]</div>`;else if(m.al?.length>1){const r=Math.round(Math.log2(m.al.length));x+=String.raw`<div class="mathline tiny formula-scroll">\[\Gamma_0(${m.N})\triangleleft E_{${esc(name)}},\qquad E_{${esc(name)}}/\Gamma_0(${m.N})\cong C_2^{${r}}.\]</div>`}
  x+=`<div class="source-note">Ford domain: all recorded Atkin–Lehner cosets {${m.al.join(', ')}} are used for the eigengroup cell; the ${m.h} kernel coset${m.h===1?'':'s'} are joined as the genuine \\(T^{k/${m.h}}F_E\\) Schreier union for \\(\\Gamma_g=\\ker\\sigma_g\\). The camera, rather than the group geometry, recentres the active tile.${same.length>1?` Same invariance group: ${same.join(', ')}.`:''}</div>`;
  x+=`<button class="class-cover moon-show-domain" data-moonshine="${esc(name)}">F_Γ</button>`;
  x+=String.raw`<details class="math-details" open><summary>\(T_g(u)\) · \(u^{60}\)</summary><div class="mathline tiny formula-scroll">\[u=q=e^{2\pi i\tau},\qquad T_{${esc(name)}}(u)=${moonshineSeriesTex(name,60,'u')}.\]</div><div class="source-note">Exact integer coefficients from the bundled PFTool v4.5 database; the moonshine kernel has cusp width 1 at ∞, so this local coordinate is u=q.</div></details>`;
  x+=moonshineJRelationHTML(name);x+=moonshineSelectedPointHTML(name);x+=moonshineModularFormsHTML(m,name);
  x+=`<details class="math-details" open><summary>${tr('directInclusions')} · ${(adj.super||[]).length+(adj.sub||[]).length}</summary>${moonNeighborhoodHTML(symbol)}</details>`;
  if(direct.length){x+=`<details class="math-details" open><summary>Hauptmodul · ${tr('directMaps')}</summary>`;for(const r of direct){const other=r.sub_symbol===symbol?(r.super_classes?.[0]):(r.sub_classes?.[0]),nt=moonNormalityTag(r.sub_symbol,r.super_symbol);x+=`<div class="moon-rel"><button class="class-cover" data-moonshine="${esc(other)}">${esc(other)}</button><span>${r.degree}${nt}</span></div>`;const map=moonRationalMapTex(r);if(map)x+=String.raw`<div class="mathline tiny formula-scroll">\[${map}.\]</div>`}x+='</details>'}
  if(rels.length){x+=`<details class="math-details"><summary>\\(q^{60}\\) · ${tr('relations')} · ${rels.length}</summary>`;for(const r of rels){const sub=r.sub_classes?.[0],sup=r.super_classes?.[0],other=r.sub_symbol===symbol?sup:sub;x+=`<div class="moon-rel"><button class="class-cover" data-moonshine="${esc(other)}">${esc(other)}</button><span>${r.immediate?'':''}d=${r.degree}</span></div>`;const map=moonRationalMapTex(r);if(map)x+=String.raw`<div class="mathline tiny formula-scroll">\[${map}.\]</div>`;if(r.relation)x+=String.raw`<div class="mathline tiny formula-scroll moon-implicit">\[${moonRelationTex(r)}=0.\]</div>`}x+='</details>'}
  const gm=Object.entries(moonshine.gamma0Matches||{}).filter(([,q])=>q.monster_class===name);if(gm.length){x+=`<div class="source-note">Certified standard congr. match: ${gm.map(([cp,q])=>`${cp} ↔ Γ0(${q.level})`).join(' · ')}</div>`;const Ns=[...new Set(gm.map(([,q])=>Number(q.level)))];for(const N of Ns)x+=gamma0ModularFormsHTML(N,name)}x+=moonCongruenceForMoon(name);
  return x+'</div>'
}
function moonshineLatticeHTML(){const groups={};for(const [s,m] of Object.entries(moonshineSymbols)){const k=`${m.N}|${m.h}`;(groups[k]||(groups[k]=[])).push([s,m])}const keys=Object.keys(groups).sort((a,b)=>{const A=a.split('|').map(Number),B=b.split('|').map(Number);return A[0]-B[0]||A[1]-B[1]});let out='<div class="moon-lattice">';for(const k of keys){const [N,h]=k.split('|').map(Number),rows=groups[k].sort((a,b)=>a[1].al.length-b[1].al.length||a[0].localeCompare(b[0]));out+=`<div class="moon-level-row"><span class="moon-level-key">${N}${h>1?'‖'+h:''}</span><div class="moon-level-nodes">`;for(const [s,m] of rows){const cls=m.classes?.[0]||moonClassesForSymbol(s)[0],sup=(moonshineAdjacency[s]?.super||[]).length;out+=`<button data-moonshine="${esc(cls)}" class="moon-node ${selectedMoonshine===cls?'selected':''}" title="${esc(s)}${sup?' · ↑'+sup:''}">${esc((m.classes||[cls]).join('/'))}<small>${esc(s)}</small></button>`}out+='</div></div>'}return out+'</div>'}
function moonshineCatalogueHTML(){if(!moonshine.names?.length)return'';const selected=selectedMoonshine?moonshineDetailHTML(selectedMoonshine):'';return String.raw`<details class="math-details moonshine-catalogue" open><summary>Moonshine · ${moonshine.names.length}</summary><div class="source-note">v25 retains the full recorded Moonshine symbols and adds exact low-level j-correspondences for 39 h=1 classes (N≤18; N=18 uses the bundled exact Γ0(18) eta-Hauptmodul relation, then certified resultant quotient maps), including \(N\!\parallel\!h\) kernels and multi–Atkin–Lehner extensions. Modular-form data now covers all 171 distinct symbols: all 120 h=1 invariance groups have exact simultaneous Atkin–Lehner +1 bases in \(k=2,4,6\) through q^60; the 51 h>1 kernels show the rigorously inherited eigengroup-invariant subspace via q\mapsto q^h. The q^60-certified relation layer supplies rational Hauptmodul maps plus a transitively reduced immediate sub/super graph; uncertified edges are deliberately left absent.</div>${selected}${moonshineLatticeHTML()}</details>`}
function activateMoonshine(name){const meta=moonshineGroups[name];if(!meta)return false;const rec=moonFordRecord(meta),plus=fordRegion(rec),base=byId.get('T2_3_I')||current;if(!plus)return false;selectedMoonshine=name;selectedOrbifoldPoint=null;SYS=makeSystem(base);regionCache=new Map();selectedCover=null;commCompare=null;frickeCompare={moonshine:true,className:name,symbol:meta.symbol,N:meta.N,h:meta.h,name,basePlus:plus,plus,gamma0:null};commSceneActive=true;fitCurrentRegion();invalidate(true);return true}

function standardGamma0Level(r){
  if(!r?.modular)return null;if(r.label==='1A 0')return 1;const name=String(r.row?.name||'');if(name.includes('∩'))return null;const m=name.match(/Γ\s*0\s*\((\d+)\)/);return m?Number(m[1]):null
}
function gpCoeffTex(a){a=String(a??'0').trim();if(/^[-+]?\d+\/\d+$/.test(a))return ratTex(a);return exprTex(a.replace(/\by\b/g,'\\alpha'))}
function mfSeriesTex(a,maxN=60,v='q'){
  if(!Array.isArray(a))return'';const terms=[],last=Math.min(maxN,a.length-1);
  for(let n=0;n<=last;n++){
    let raw=String(a[n]??'0').trim();if(!raw||raw==='0')continue;
    // Keep the sign outside the coefficient exactly once.  Older rendering
    // could leave a leading minus inside gpCoeffTex and prepend another minus.
    let neg=false;raw=raw.replace(/^\+\s*/,'');while(/^\-\s*/.test(raw)){neg=!neg;raw=raw.replace(/^\-\s*/,'').trim()}if(!raw||raw==='0')continue;
    const q=n===0?'':(n===1?v:`${v}^{${n}}`),bodyTex=gpCoeffTex(raw),coef=(raw==='1'&&q)?'':bodyTex,term=q?(coef?`\\left(${coef}\\right)${q}`:q):bodyTex;
    terms.push({neg,term})
  }
  if(!terms.length)return'0';let out='';for(let i=0;i<terms.length;i++){const t=terms[i];out+=i===0?(t.neg?'-':'')+t.term:(t.neg?' - ':' + ')+t.term}return out+`+O(${v}^{${last+1}})`
}
function gamma0ModularFormsHTML(N,context=''){
  const D=modularFormsGamma0[String(N)];if(!D)return'';let out=String.raw`<details class="math-details"><summary>Γ0(${N}) · \(k=2,4,6\)</summary><div class="source-note">Exact PARI/GP basis data through q^60. Every generator is written explicitly; Hecke eigenforms below refer to the newspace.</div>`;
  for(const k of ['2','4','6']){const z=D[k];if(!z)continue;out+=`<details class="math-details"><summary>M_${k}(Γ0(${N})) · ${z.basis.length}</summary>`;z.basis.forEach((a,i)=>{out+=String.raw`<div class="mathline tiny formula-scroll">\[f_{${k},${i+1}}(q)=${mfSeriesTex(a,60)}.\]</div>`});if(z.eigen?.length){out+=`<div class="relation-head">\\(S_k^{\\mathrm{new}}\\) · Hecke · ${z.eigen.length}</div>`;if(z.fields&&z.fields!=='[]')out+=String.raw`<div class="mathline tiny formula-scroll">\[K=${gpCoeffTex(z.fields)}.\]</div>`;z.eigen.forEach((a,i)=>{out+=String.raw`<div class="mathline tiny formula-scroll">\[g_{${k},${i+1}}(q)=${mfSeriesTex(a,60)}.\]</div>`})}out+='</details>'}return out+'</details>'
}
function cpGenus0ModularFormsHTML(label){
  const D=cpGenus0ModularForms[label];if(!D)return'';let out=`<details class="math-details"><summary>\\(M_k(\\Gamma)\\) · \\(k=2,4,6\\)</summary><div class="source-note">Reconstructed on the actual CP genus-zero group from its cusp-normalized Hauptmodul and the exact Belyi branch divisor. The basis is computed in the regular local coordinate s=1/t, so the displayed u^60 coefficients do not rely on cancellation of separately truncated Laurent poles.</div>`;out+=String.raw`<div class="mathline tiny formula-scroll">\[j=\frac{${D.j_num_tex}}{${D.j_den_tex}},\qquad u=q^{1/${D.cusp_width}},\qquad ${D.field_tex||'\\mathbb Q'}.\]</div>`;out+=fieldGeneratorHTML(D.field_tex||'');
  for(const k of ['2','4','6']){const z=D.weights?.[k];if(!z)continue;out+=`<details class="math-details"><summary>M_${k}(Γ_CP(${esc(label)})) · ${z.dimension}</summary>`;if(!z.basis?.length)out+=String.raw`<div class="mathline tiny">\[M_${k}=0.\]</div>`;else z.basis.forEach((a,i)=>{out+=String.raw`<div class="mathline tiny formula-scroll">\[F_{${k},${i+1}}(u)=${mfSeriesTex(a,60,'u')}.\]</div>`});out+='</details>'}return out+'</details>'
}
function gamma1Level(r){const m=String(r?.row?.name||'').match(/Γ\s*1\s*\((\d+)\)/);return m?Number(m[1]):null}
function gamma1CharacterFormsHTML(N){const D=gamma1CharacterForms[String(N)];if(!D)return'';let out=`<details class="math-details"><summary>Γ1(${N}) \\(\\chi\\) · \\(k=2,4,6\\)</summary><div class="source-note">Exact PARI/GP decomposition into M_k(Γ0(${N}),χ) over Galois-orbit representatives of Dirichlet characters; every displayed basis is expanded through q^60.</div>`;for(const k of ['2','4','6']){const blocks=D[k]||[];out+=`<details class="math-details"><summary>k=${k} · ${blocks.length} \\(\\chi\\)${blocks.length===1?'':'s'}</summary>`;blocks.forEach((z,j)=>{out+=`<details class="math-details"><summary>χ${j+1} · |χ|=${z.order} · ${z.dimension}</summary><div class="source-note">character ${esc(z.character)} · coefficient field ${esc(z.field)}</div>`;(z.basis||[]).forEach((a,i)=>out+=String.raw`<div class="mathline tiny formula-scroll">\[f_{${k},${j+1},${i+1}}(q)=${mfSeriesTex(a,60)}.\]</div>`);out+='</details>'});out+='</details>'}return out+'</details>'}
function cexp(z){const e=Math.exp(z.x);return C(e*Math.cos(z.y),e*Math.sin(z.y))}
function cpHauptNumeric(D,z,radial=1){if(!D?.haupt_coeffs&&!D?.haupt_numeric)return null;const zz=sc(z,radial),tau=cay(zz);if(!Number.isFinite(tau.x+tau.y)||tau.y<=0)return null;const u=cexp(C(-2*Math.PI*tau.y/D.cusp_width,2*Math.PI*tau.x/D.cusp_width)),iu=div(C(1),u);let v=iu,p=C(1);for(let n=0;n<=60;n++){if(n>0)p=mul(p,u);const c=D.haupt_numeric?.[n+1];if(Array.isArray(c)){if(c[0]||c[1])v=add(v,mul(p,C(Number(c[0]||0),Number(c[1]||0))))}else{const a=Number(D.haupt_coeffs?.[n+1]||0);if(a)v=add(v,sc(p,a))}}return v}
function flattenVertexRoots(D,kind,width=null){let a=kind==='2'?D.vertices?.order2:(kind==='3'?D.vertices?.order3:D.vertices?.cusps)||[];if(kind==='inf'&&width!=null){const same=a.filter(f=>Number(f.multiplicity)===Number(width));if(same.length)a=same}const out=[];for(const f of a)for(const r of f.roots||[])out.push({...r,factor:f});return out}
function pointNumericTex(z,digits=10){if(!z||!Number.isFinite(z.x+z.y))return null;const re=Number(z.x).toPrecision(digits),im=Math.abs(Number(z.y)).toPrecision(Math.max(6,digits-2));return `${re}${z.y>=0?'+':'−'}${im}i`}
function algebraicPointFieldTex(D,f,variable='t'){
  if(D?.field_tex&&D.field_tex!=='\\mathbb Q')return D.field_tex;
  const deg=Number(f?.degree||0),poly=String(f?.poly_tex||'');if(deg>1&&poly)return `\\mathbb Q(\\alpha),\\qquad ${poly.replace(new RegExp(variable,'g'),'\\alpha')}=0`;
  return D?.field_tex||'\\mathbb Q'
}
function cpSelectedPointHTML(r){
  const q=selectedOrbifoldPoint,D=cpGenus0ModularForms[r?.label];if(!q||!D||!['cover','cell'].includes(q.sourceKind))return'';
  if(q.sourceRegion?.degree!=null&&D.signature?.mu!=null&&Number(q.sourceRegion.degree)!==Number(D.signature.mu))return'';
  const ideal=q.order==='inf',head=`<div class="divider"></div><div class="relation-head">\\(P\\) · ${esc(q.label||'')}</div>`;
  if(ideal&&absv(sub(q.z,C(1)))<2e-5)return head+String.raw`<div class="mathline small formula-scroll">\[t(P)=\infty.\]</div>`;
  const kind=ideal?'inf':String(q.order),roots=(Number(q.order)>=2||ideal)?flattenVertexRoots(D,kind,ideal?q.width:null):[];let val=cpHauptNumeric(D,q.z,ideal?.92:1),pick=null;
  if(roots.length===1)pick=roots[0];else if(roots.length&&val)pick=roots.map(z=>({...z,dist:Math.hypot(val.x-z.re,val.y-z.im)})).sort((a,b)=>a.dist-b.dist)[0];
  if(pick){
    const f=pick.factor,approx=complexApproxTex(Number(pick.re),Number(pick.im),9);let out=head;
    if(pick.exact_tex){out+=String.raw`<div class="mathline small formula-scroll">\[t(P)=${pick.exact_tex}.\]</div>`;if(approx)out+=String.raw`<div class="mathline tiny formula-scroll">\[t(P)\approx ${approx}.\]</div>`;if(/\\(?:alpha|beta|omega|xi|nu|theta)\b/.test(String(pick.exact_tex))||String(D.field_tex||'')!=='\\mathbb Q')out+=fieldGeneratorHTML(D.field_tex||'',D)}
    else{const ambient=fieldGeneratorSpec(D.field_tex||'',D),sym=ambient?'\\theta':'\\alpha',poly=String(f.poly_tex||'').replace(/\bt\b/g,sym);out+=String.raw`<div class="mathline small formula-scroll">\[t(P)=${sym}.\]</div>`;if(poly||approx)out+=String.raw`<div class="mathline tiny formula-scroll">\[${poly?`${poly}=0`:''}${poly&&approx?'\\qquad ':''}${approx?`${sym}\\approx ${approx}`:''}.\]</div>`;if(ambient)out+=fieldGeneratorHTML(D.field_tex||'',D)}
    return out
  }
  if(!val)val=cpHauptNumeric(D,q.z,.9);if(!val)return head;
  const numeric=complexApproxTex(val.x,val.y,10),real=Math.abs(val.y)<2e-8?exactTex(val.x):null;return head+(real?String.raw`<div class="mathline small formula-scroll">\[t(P)\approx ${real}.\]</div>`:String.raw`<div class="mathline small formula-scroll">\[t(P)\approx ${numeric}.\]</div>`)
}
function moonHauptNumeric(name,z,radial=1){const a=moonshine.records?.[name];if(!Array.isArray(a))return null;const zz=sc(z,radial),tau=cay(zz);if(!Number.isFinite(tau.x+tau.y)||tau.y<=0)return null;const q=cexp(C(-2*Math.PI*tau.y,2*Math.PI*tau.x)),iq=div(C(1),q);let v=iq,p=C(1);for(let n=1;n<=Math.min(60,a.length);n++){p=mul(p,q);const c=Number(a[n-1]);if(c)v=add(v,sc(p,c))}return v}
function moonJPointRoots(D,kind){
  const P=D?.points||{};let a=[];
  if(kind==='inf')a=(P.finite_cusps||[]).slice();
  else if(String(kind)==='2')a=(P.order2||[]).concat(P.quotient_branch||[]);
  else if(String(kind)==='3')a=(P.order3||[]).concat(P.quotient_branch||[]);
  else a=(P.quotient_branch||[]).slice();
  const out=[];for(const f of a)for(const r of f.roots||[])out.push({...r,factor:f});return out
}
function moonshineJRelationHTML(name){const D=moonshineJRelations[name];if(!D)return'';const pts=D.points||{},n=(pts.order2||[]).length+(pts.order3||[]).length+(pts.finite_cusps||[]).length+(pts.quotient_branch||[]).length;return String.raw`<details class="math-details"><summary>\(j\leftrightarrow T_g\) · d_j=${D.degree_j}</summary><div class="mathline tiny formula-scroll">\\[${D.relation_tex}=0.\\]</div><div class="source-note">${esc(D.method)}. The relation is reconstructed from the q^60 branch on Γ0(${D.N}) and, for Atkin–Lehner extensions, exact resultant elimination through the already coefficient-certified Hauptmodul map.</div>${n?`<div class="source-note">Exact low-level point data are retained for inherited j=0/1728 fibres, finite cusps, and quotient-map branch values; click a displayed Ford vertex to match its Hauptmodul coordinate.</div>`:''}</details>`}
function moonshineSelectedPointHTML(name){
  const q=selectedOrbifoldPoint;if(!q||!['ford','ford-mesh'].includes(q.sourceKind))return'';const D=moonshineJRelations[name],head=`<div class="divider"></div><div class="relation-head">\\(P\\) · ${esc(q.label||'')}</div>`;
  if(q.order==='inf'&&absv(sub(q.z,C(1)))<2e-5)return head+String.raw`<div class="mathline small">\[T_{${esc(name)}}(P)=\infty.\]</div>`;
  if(D){const roots=(q.order==='inf'||Number(q.order)>=2)?moonJPointRoots(D,q.order):[],znum=moonHauptNumeric(name,q.z,q.order==='inf'?.92:1);let pick=null;if(roots.length===1)pick=roots[0];else if(roots.length&&znum)pick=roots.map(r=>({...r,dist:Math.hypot(znum.x-r.re,znum.y-r.im)})).sort((a,b)=>a.dist-b.dist)[0];
    if(pick){const f=pick.factor,approx=complexApproxTex(Number(pick.re),Number(pick.im),9);let out=head;if(pick.exact_tex){out+=String.raw`<div class="mathline small formula-scroll">\[T_{${esc(name)}}(P)=${pick.exact_tex}.\]</div>`;if(approx)out+=String.raw`<div class="mathline tiny formula-scroll">\[T_{${esc(name)}}(P)\approx ${approx}.\]</div>`;if(/\\(?:alpha|beta|omega|xi|nu|theta)\b/.test(String(pick.exact_tex))&&f?.poly_tex){const poly=String(f.poly_tex).replace(/\bY\b/g,'\\alpha');out+=String.raw`<div class="mathline tiny formula-scroll">\[${poly}=0,\qquad \alpha\approx ${approx}.\]</div>`}}else{const sym='\\alpha',poly=String(f?.poly_tex||'').replace(/\bY\b/g,sym);out+=String.raw`<div class="mathline small formula-scroll">\[T_{${esc(name)}}(P)=${sym}.\]</div>`;if(poly||approx)out+=String.raw`<div class="mathline tiny formula-scroll">\[${poly?`${poly}=0`:''}${poly&&approx?'\\qquad ':''}${approx?`${sym}\\approx ${approx}`:''}.\]</div>`}return out}
  }
  const z=moonHauptNumeric(name,q.z,q.order==='inf'?.92:1);if(!z)return'';const tex=Math.abs(z.y)<2e-7?exactTex(z.x):complexApproxTex(z.x,z.y,10);return head+String.raw`<div class="mathline small formula-scroll">\[T_{${esc(name)}}(P)\approx ${tex}.\]</div>`
}
function relationMoonArgTex(r){const d=Number(r.q_scale||1),ordr=Number(r.root_order||1),a=Number(r.root_power||0);let u=d===1?'u':`u^{${d}}`;if(ordr>1)u=`\\zeta_{${ordr}}^{${a}}${u}`;return `T_{${r.moon}}\\!\\left(${u}\\right)`}
function crossRelationTex(r){return String(r.phi_tex||'').replace(/\bX\b/g,relationMoonArgTex(r)).replace(/\bY\b/g,`t_{\\mathrm{CP}}(u)`)}
function relationTransformNote(r){const d=Number(r.q_scale||1),o=Number(r.root_order||1),a=Number(r.root_power||0);return d===1&&o===1?'phase 0':`q\\mapsto ${o>1?`\\zeta_{${o}}^{${a}}`:''}q^{${d}}`}
function moonCongruenceForCP(label){const a=moonCongruenceRelations.filter(r=>r.cp===label);if(!a.length)return'';let out=String.raw`<details class="math-details"><summary>Moonshine · \(\leftrightarrow\) · \(d\le4\) · ${a.length}</summary>`;for(const r of a)out+=String.raw`<div class="mathline tiny formula-scroll">\[${crossRelationTex(r)}=0.\]</div><div class="source-note">${esc(r.moon)} · ${relationTransformNote(r)} · coefficient-certified through u^${r.checked_through||56}</div>`;return out+'<div class="source-note">v25 searches the stored local expansions with q→q²,q³ and roots of unity of exact orders 2,3,4,6 in addition to the untwisted identities; only coefficient-certified rational polynomial relations are displayed.</div></details>'}
function moonCongruenceForMoon(name){const a=moonCongruenceRelations.filter(r=>r.moon===name);if(!a.length)return'';let out=String.raw`<details class="math-details"><summary>Hauptmodul · \(d\le4\) · ${a.length}</summary>`;for(const r of a)out+=String.raw`<div class="mathline tiny formula-scroll">\[${crossRelationTex(r)}=0,\qquad t_{\\mathrm{CP}}=t_{${esc(r.cp)}}.\]</div><div class="source-note">${esc(r.cp)} · ${relationTransformNote(r)} · coefficient-certified through u^${r.checked_through||56}</div>`;return out+'</details>'}
function modularCurveTheory(r){
  if(!r?.modular)return'';const lab=r.label,z0=cpGenus0[lab],j1=genus1J[lab];let x='';
  const mg=matrixGroupTex(lab),classical=r.row?.name;
  if(z0){
    x+='<div class="divider"></div><div class="relation-head">Hauptmodul</div>';
    if(classical)x+=`<div class="mathline small formula-scroll">\\[H=${esc(classical)}.\\]</div>`;
    const desc=cpCongruence[lab]||z0.description;if(desc)x+=`<div class="source-note congruence-desc">${esc(desc.replace(/Gamma/g,'Γ').replace(/rho_/g,'ρ_'))}</div>`;
    if(mg)x+=String.raw`<details class="math-details"><summary>\(\rho_N(\Gamma)\)</summary><div class="mathline tiny formula-scroll">\\[${mg}.\\]</div><div class="source-note">The finite matrix image is the exact CP representative; classical Γ0/Γ1 names are displayed only when recorded.</div></details>`;
    const gh=z0.gamma0?gamma0Haupt[String(z0.gamma0)]:null;
    if(gh)x+=`<div class="mathline tiny formula-scroll">\\[h_${gh.N}^{\\mathrm{std}}(\\tau)=${gh.h_tex},\\qquad t^{\\mathrm{std}}=h_${gh.N}^{\\mathrm{std}}-(${gh.constant}).\\]</div><div class="mathline tiny formula-scroll">\\[j=${gh.j_tex}.\\]</div><div class="source-note">This eta formula uses the standard Γ0(${gh.N}) representative; the CP representative may be a finite conjugate, whose own cusp-normalized series is shown below.</div>`;
    else if(z0.j_tex)x+=`<div class="mathline tiny formula-scroll">\\[j=${z0.j_tex}.\\]</div>`;
    else if(z0.hauptmodul)x+=`<details class="math-details"><summary>Hauptmodul</summary><div class="source-note mono-note">${esc(z0.hauptmodul)}</div><div class="source-note mono-note">${esc(z0.certificate||'')}</div></details>`;
    if(gh&&frickeHauptRelations[String(gh.N)])x+=frickeRelationHTML(gh.N);
    x+=cpQSeriesHTML(lab);x+=cpSelectedPointHTML(r);x+=adjacentHauptmodulHTML(r,z0);x+=cpGenus0ModularFormsHTML(lab);
  }
  if(j1){const cls=modularStatusKind(j1.status);x+=String.raw`<div class="divider"></div><div class="relation-head">\(g=1\) · \(j\)</div><div class="mathline small formula-scroll">\\[j(X_H)=${asciiJTex(j1.j)}.\\]</div><div class="source-note ${cls}">${esc(j1.status)} · period residual ${esc(j1.period_error)}</div>`;if(mg)x+=String.raw`<details class="math-details"><summary>\(\rho_N(\Gamma)\)</summary><div class="mathline tiny formula-scroll">\\[${mg}.\\]</div></details>`}
  if(classical&&!z0)x+=String.raw`<div class="divider"></div><div class="relation-head">Γ</div><div class="mathline small formula-scroll">\\[H=${esc(classical)}.\\]</div>${mg?`<details class="math-details"><summary>\(\rho_N(\Gamma)\)</summary><div class="mathline tiny formula-scroll">\\[${mg}.\\]</div></details>`:''}`;
  if(!z0&&!j1&&!classical)x+=`<div class="source-note">exact congr. record: Γ_CP(${esc(cpLabelCompact(lab))})=ρ_N^{-1}(H_N); no stronger classical name is asserted here.</div>`;
  const g0=standardGamma0Level(r);if(g0)x+=gamma0ModularFormsHTML(g0,lab);const g1=gamma1Level(r);if(g1)x+=gamma1CharacterFormsHTML(g1);
  x+=moonshineHTML(lab);x+=moonCongruenceForCP(lab);
  return x
}
function strictBelyiMatch(bm){if(!bm)return false;if(bm.strict===true||bm.exact===true)return true;const z=String(bm.status||bm.match_kind||bm.proof_kind||'').toLowerCase();return z.startsWith('exact')||z.includes('strict')||z.includes('monodromy')}
function belyiLookup(r){if(!r)return null;const keys=[r.key,r.belyi_key,r.database_key,r.id,r.key&&`comm:${r.key}`,r.id&&`quad:${r.id}`].filter(Boolean);for(const k of keys){const raw=belyiDatabaseMatches[k];let bm=raw;if(typeof raw==='string'){const m=belyiModels[raw];if(m)bm={...m,lmfdb_label:raw,strict:true,status:'exact_monodromy'};}else if(raw&&raw.model){const m=belyiModels[raw.model];if(m)bm={...m,...raw,lmfdb_label:raw.model};}if(strictBelyiMatch(bm))return bm}return null}
function simpleGroupLabelTex(s){s=String(s||'').trim();if(!s)return'';if(s.includes('\\'))return s;return `\\mathrm{${s.replace(/[{}\\]/g,'')}}`}
function coverMonodromyTex(r,bm){if(bm?.group_tex)return bm.group_tex;if(bm?.monodromy_tex)return bm.monodromy_tex;if(bm?.group)return simpleGroupLabelTex(bm.group);if(r?.monodromy)return simpleGroupLabelTex(r.monodromy);const t=coverTriple(r);if(Array.isArray(t)&&t.length>=2){const gens=t.slice(0,2).map(p=>p.map(x=>Number(x)-1)),G=permClosure0(gens,20000);if(G)return quotientGroupTex(G)}return'\\text{—}'}
function belyiFieldNameTex(bm){const f=String(bm?.field_name_tex||bm?.field_tex||'\\mathbf Q');if(/^\s*\\(?:mathbf|mathbb)\s*Q\s*$/.test(f))return f;const sym=bm?.generator_symbol_tex||bm?.field_generator_symbol_tex;if(sym)return`\\mathbf Q(${sym})`;const m=f.match(/\\(?:mathbf|mathbb)\s*Q\s*\((\\(?:alpha|beta|omega|xi|nu|theta)|[αβωνξθ])\)/);if(m)return`\\mathbf Q(${m[1]})`;return f||'\\mathbf Q'}
function belyiGeneratorHTML(bm){if(!bm)return'';const meta={generator_symbol_tex:bm.generator_symbol_tex||bm.field_generator_symbol_tex,generator_minpoly_tex:bm.generator_minpoly_tex||bm.field_minpoly_tex||bm.minpoly_tex,generator_approx_tex:bm.generator_approx_tex||bm.field_generator_approx_tex||bm.approx_tex};return fieldGeneratorHTML(bm.field_tex||bm.field_name_tex||'',meta)}
function belyiMajorHTML(r,bm=belyiLookup(r)){
  if(!strictBelyiMatch(bm))return'';
  const d=bm.degree??r?.degree??'—',g=bm.genus??r?.genus??'—',G=coverMonodromyTex(r,bm),base=(r?.parent&&byId.get(r.parent))||current,abc=bm.abc_tex||[base?.a??current?.a,base?.b??current?.b,base?.c??current?.c].map(x=>x==='inf'||x==null?'\\infty':x).join(','),K=belyiFieldNameTex(bm),curve=bm.curve_tex||bm.weierstrass_tex||bm.model_tex||'',beta=bm.beta_tex||bm.map_tex||bm.belyi_map_tex||bm.rational_map_tex||'';
  let out=String.raw`<div class="belyi-major"><details class="math-details belyi-details" open><summary>\(\mathrm{Belyi}\)</summary><div class="mathline small formula-scroll">\[d=${d},\qquad g=${g},\qquad G=${G},\qquad (a,b,c)=\left(${abc}\right).\]</div><div class="mathline small formula-scroll">\[K=${K}.\]</div>`;
  if(Number(g)===0&&!curve)out+=String.raw`<div class="mathline small formula-scroll">\[X=\mathbf P^{1}.\]</div>`;
  if(curve)out+=String.raw`<div class="mathline small formula-scroll">\[${curve}.\]</div>`;
  const ec=bm.lmfdb_curve||bm.elliptic_curve_label||bm.curve_label;if(ec&&Number(g)===1)out+=String.raw`<div class="mathline tiny formula-scroll">\[E=\mathrm{${String(ec).replace(/[{}\\]/g,'')}}.\]</div>`;
  if(bm.j_tex&&Number(g)===1)out+=String.raw`<div class="mathline small formula-scroll">\[j(E)=${bm.j_tex}.\]</div>`;
  if(beta)out+=String.raw`<div class="mathline small formula-scroll">\[${beta}.\]</div>`;
  out+=belyiGeneratorHTML(bm);return out+'</details></div>'
}

function coverDetail(r){
  if(!r)return String.raw`<div class="mathline small muted">\(H\leq\Gamma\)</div>`;
  const t=coverTriple(r),no=coverNumber(r),noTex=no?`\\#${no},\\qquad `:'';let x=String.raw`<div class="mathline">\[${noTex}[\Gamma:H]=${r.degree??'—'},\qquad g(X_H)=${r.genus}.\]</div>`;
  const nq=normalityBetween(r,null);
  if(nq?.status==='normal')x+=String.raw`<div class="mathline small formula-scroll">\[H\triangleleft\Gamma,\qquad \Gamma/H\cong ${nq.quotient_tex||'1'}.\]</div>`;
  else if(nq?.status==='not_normal')x+=String.raw`<div class="mathline small formula-scroll">\[H\not\triangleleft\Gamma.\]</div>`;
  if(r.modular)x+=String.raw`<div class="mathline small">\[N=${r.level},\qquad H=${groupNameTex(r.row.name||r.label)}.\]</div>`;
  if(r.elliptic_orders||r.cusp_widths){const es=(r.elliptic_orders||[]).join(',')||'\\varnothing',cw=(r.cusp_widths||[]).join(',')||'\\varnothing',cc=r.cusp_count??(r.cusp_widths||[]).length;x+=`<div class="mathline small">\\[\\operatorname{sig}(H)=\\left(${r.genus};\\ ${es}\\right),\\qquad c(H)=${cc},\\qquad w_{\\mathrm{cusp}}=(${cw}).\\]</div>`}
  if(r.census){const c=r.congruence===true?'congr.':r.congruence===false?'noncongr.':'congr. status not precomputed';x+=`<div class="source-note">${esc(c)} · exact Kulkarni/passport record</div>`}
  if(r.lowIndex){const meta=lowIndexMeta[current.id]||{},kind=r.conjugacy_class===false?'rooted conjugate realization':(r.exhaustive===false?'verified transitive action · search not exhaustive at this index':'exact subgroup conjugacy class');x+=`<div class="source-note">${esc(kind)}${meta.complete_through_index?` · complete census through [·]=${meta.complete_through_index}`:''}</div>`;}
  if(t){const p=t.map(cyclePartition);x+=String.raw`<details class="math-details"><summary>\(\sigma_0,\sigma_1,\sigma_\infty\)</summary><div class="mathline tiny formula-scroll">\[(\sigma_0,\sigma_1,\sigma_\infty):\quad ${p.map(partitionTex).join(', ' )}.\]</div><div class="source-note">transitive permutation triple on ${t[0].length} cosets; product and triangle-ord constraints are checked in the data build.</div></details>`}
  x+=belyiMajorHTML(r);
  const ideal=idealTex(r);if(ideal)x+=`<div class="mathline small">\\[${ideal}.\\]</div>`;
  const parts=r.branchPartitions||triplePartitions(t);if(parts?.length===3)x+=`<div class="mathline small">\\[(\\lambda_0,\\lambda_1,\\lambda_\\infty)=\\left(${parts.map(partitionTex).join(',')}\\right).\\]</div>`;
  const up=r.row?.upstairs_orbifold_signature;if(up?.elliptic_orders_with_multiplicity)x+=`<div class="mathline small">\\[\\operatorname{sig}(H)=\\left(g=${up.coarse_genus};\\ ${up.elliptic_orders_with_multiplicity.join(',')}\\right).\\]</div>`;
  if(r.monodromy)x+=`<div class="source-note">${esc(r.monodromy)}</div>`;
  if(r.modular&&(r.row.c2!=null||r.row.c3!=null))x+=`<div class="mathline small">\\[e_2=${esc(r.row.c2)},\\qquad e_3=${esc(r.row.c3)}.\\]</div>`;
  const jr=r.row?.j_resolution;if(jr?.status==='orbit_resolved'){
    const hit=(jr.orbit_resolutions||[]).find(z=>z.drv_representative_index===(r.rep||0))||jr.orbit_resolutions?.[0];
    if(hit)x+=`<div class="divider"></div><div class="mathline small">\\[j=${ratTex(hit.j_invariant)},\\qquad N_E=${hit.conductor??'—'}.\\]</div><div class="source-note">${esc(hit.lmfdb_curve||'')} ${hit.newform?`· ${esc(hit.newform)}`:''}</div>`
  }
  if(jr?.status==='jacobian_resolved_curve_torsor_not_asserted'&&jr.jacobian_j_invariant){x+=String.raw`<div class="divider"></div><div class="relation-head">\(\operatorname{Jac}X_H\)</div><div class="mathline small">\\[j(\\operatorname{Jac}X_H)=${ratTex(jr.jacobian_j_invariant)},\qquad N_E=${jr.jacobian_conductor??'—'}.\]</div><div class="source-note">Exact Jacobian identification from the supplied genus-one curve record; the torsor itself is not asserted to carry the same chosen origin.</div>`}
  if(r.higher){
    if(r.model)x+=`<div class="divider"></div><div class="mathline small formula-scroll">\\[${polyTex(r.model)}.\\]</div>`;
    if(r.automorphism_group)x+=`<div class="mathline small">\\[\\operatorname{Aut}(X)=${esc(r.automorphism_group)},\\qquad |\\operatorname{Aut}(X)|=${r.automorphism_group_order}.\\]</div>`;
    if(r.belyi_identity)x+=`<div class="mathline small formula-scroll">\\[${polyTex(r.belyi_identity)}.\\]</div>`;
    if(r.jacobian)x+=`<div class="source-note">${esc(r.jacobian)}</div>`
  }
  if(r.modular)x+=modularCurveTheory(r);
  return x
}
function neighborhoodSVG(nb){const center=nb.center,w=380,top=30,mid=112,bottom=205,h=245;let s=`<svg class="math-graph cover-tree" viewBox="0 0 ${w} ${h}">`;const centerLabel=center?coverLabel(center):'Γ',centerKey=center?.key||'';s+=`<g ${center?'data-cover="'+esc(centerKey)+'"':''} class="graph-hit"><circle class="node exact selected" cx="190" cy="${mid}" r="7"/><text x="202" y="${mid+4}">${esc(centerLabel)}</text></g>`;const draw=(arr,y,up)=>{const n=arr.length;if(!n)return;arr.forEach((r,i)=>{const x=24+(i+.5)*(332/n);s+=`<line class="edge strong" x1="190" y1="${mid+(up?-7:7)}" x2="${x}" y2="${y+(up?7:-7)}"/><g data-cover="${esc(r.key)}" class="graph-hit"><circle class="node ${coverTriple(r)?'exact':''}" cx="${x}" cy="${y}" r="6"/><text text-anchor="middle" x="${x}" y="${y+(up?-11:19)}">${esc(coverLabel(r))}</text></g>`})};draw(nb.sup,top,true);draw(nb.sub,bottom,false);s+='</svg>';return s}

// -----------------------------------------------------------------------------
// Mathematical panels
// -----------------------------------------------------------------------------
function alphaPolyTex(s){return polyTex(String(s??'x').replace(/\bx\b/g,'alpha')).replace(/\balpha\b/g,'\\alpha')}
function algebraTex(s){return exprTex(String(s??'').replace(/\balpha\b/g,'α').replace(/\bbeta\b/g,'β').replace(/\by\b/g,'y').replace(/\+\/-/g,'±')).replace(/α/g,'\\alpha').replace(/β/g,'\\beta').replace(/±/g,'\\pm')}
function ramificationTex(q){
  const a=q?.finite_ramified_prime_ideals_from_prior_PARIGP||[];if(!a.length)return '\\varnothing';
  return `\\{${a.map(p=>`\\mathfrak p_{${p.rational_prime}}`).join(',')}\\}`
}
function ramificationMeta(q){
  const a=q?.finite_ramified_prime_ideals_from_prior_PARIGP||[];if(!a.length)return'';
  return a.map(p=>`\\mathfrak p_{${p.rational_prime}}:\\ (e,f,N)=(${p.e},${p.f},${p.norm})`).join('\\qquad ')
}
function restrictedNumericExpr(s,x=0){s=String(s||'').trim();if(!s||/[^0-9x+\-*/(). _a-z]/i.test(s)||/\b(?!sqrt\b|cos\b|pi\b|x\b)[a-z]+\b/i.test(s))return NaN;try{return Number(Function('x','"use strict";return ('+s.replace(/\^/g,'**').replace(/\bsqrt\b/g,'Math.sqrt').replace(/\bcos\b/g,'Math.cos').replace(/\bpi\b/g,'Math.PI')+');')(x))}catch{return NaN}}
function traceFieldGeneratorApprox(k){const roots=k?.reduced_polynomial_real_roots||[];if(!roots.length)return'';const target=restrictedNumericExpr(k.natural_generator),m=String(k.reduced_generator_in_natural||'').match(/^Mod\((.*),\s*[^,]+\)$/);if(Number.isFinite(target)&&m){let best=null;for(const r of roots){const y=restrictedNumericExpr(m[1],Number(r)),d=Math.abs(y-target);if(Number.isFinite(d)&&(!best||d<best.d))best={r:Number(r),d}}if(best)return decimalTexNumber(best.r,10)}return decimalTexNumber(Number(roots[0]),10)}
function triangleLambdaNumeric(g){const t=n=>(n==='inf'||n==null)?2:2*Math.cos(Math.PI/Number(n)),a=t(g.a),b=t(g.b),c=t(g.c);return a*a+b*b+c*c+a*b*c-4}
function arithmeticBlock(g){
  const k=g.invariant_trace_field||{},q=exactQuatMap.get(Number(g.commensurability_class)),lr=lambdaMap.get(g.id),f=k.reduced_minpoly||k.natural_minpoly||'x';
  const field=k.degree===1?'\\mathbf Q':`\\mathbf Q(\\alpha),\\quad ${alphaPolyTex(f)}=0`;
  let B='—';if(q){if(q.exact_algebra==='M_2(Q)')B='M_2(\\mathbf Q)';else if(q.hilbert_symbol){const aa=algebraTex(q.hilbert_symbol.a),bb=algebraTex(String(q.hilbert_symbol.b).replace(/\by\b/g,'alpha'));B=`\\left(\\frac{${aa},\\;${bb}}{k_\\Gamma}\\right)`}}
  let out=`<div class="divider"></div><div class="eq-grid compact"><div class="eq-key">\\(k_\\Gamma\\)</div><div class="eq-val">\\(${field}\\)</div><div class="eq-key">\\(D_k\\)</div><div class="eq-val">\\(${exactTex(k.discriminant)}\\)</div><div class="eq-key">\\(B_\\Gamma\\)</div><div class="eq-val">\\(${B}\\)</div><div class="eq-key">\\(\\operatorname{Ram}_f B\\)</div><div class="eq-val">\\(${ramificationTex(q)}\\)</div></div>`;
  if(Number(k.degree)>1){const ap=traceFieldGeneratorApprox(k);out+=String.raw`<div class="mathline tiny formula-scroll">\[${alphaPolyTex(f)}=0${ap?`,\qquad \alpha\approx ${ap}`:''}.\]</div>`}
  const rm=ramificationMeta(q);if(rm)out+=`<div class="mathline tiny muted formula-scroll">\\[${rm}.\\]</div>`;
  if(lr){const la=triangleLambdaNumeric(g);out+=`<div class="mathline small formula-scroll">\\[m_\\lambda(x)=${polyTex(lr.lambda_minpoly_exact)},\\qquad \\lambda\\approx ${decimalTexNumber(la,10)},\\qquad \\mathbf Q(\\lambda)=k_\\Gamma.\\]</div><div class="eq-grid compact"><div class="eq-key">\\([\\mathcal O_k:\\mathbf Z[\\lambda]]\\)</div><div class="eq-val">\\(${exactTex(lr.power_basis_order_index_Z_lambda_in_OK)}\\)</div><div class="eq-key">\\(\\operatorname{Tr}(\\lambda),N(\\lambda)\\)</div><div class="eq-val">\\(${exactTex(lr.lambda_field_trace_over_Q??lr.lambda_trace)},${exactTex(lr.lambda_field_norm_over_Q??lr.lambda_norm)}\\)</div></div>`}
  return out
}
function groupPanel(g){
  const rel=[ord(g.a)&&`x^{${g.a}}=1`,ord(g.b)&&`y^{${g.b}}=1`,ord(g.c)&&`z^{${g.c}}=1`].filter(Boolean).join(',\\;');
  return String.raw`<h2>\(\Gamma=\Delta${sig(g)}\)</h2><div class="mathline">\[\Gamma=\langle x,y,z\mid xyz=1,\;${rel}\rangle.\]</div><div class="mathline">\[\mu(\Gamma\backslash\mathbb H)=${areaTex(g)}.\]</div>`+arithmeticBlock(g)+noncompactSelectedPointHTML(g)
}
function standardHGTex(g,z='z'){const h=g.hypergeometric||hauptMap.get(g.id)?.hypergeometric||{},A=rqParse(h.A),B=rqParse(h.B),C0=rqParse(h.C);return hfTex(A,B,C0,z)}
function pairFor(g,peer){return (pairByGroup.get(g.id)||[]).find(r=>r.a===peer||r.b===peer)||null}
function verifiedHGPath(a,b){if(a===b)return[];const adj=new Map();for(const r of pairRelations){if(!explicitHGPair[r.key])continue;if(!adj.has(r.a))adj.set(r.a,[]);if(!adj.has(r.b))adj.set(r.b,[]);adj.get(r.a).push({to:r.b,r});adj.get(r.b).push({to:r.a,r})}const q=[a],prev=new Map([[a,null]]);for(let h=0;h<q.length;h++){const u=q[h];for(const e of adj.get(u)||[]){if(prev.has(e.to))continue;prev.set(e.to,{from:u,r:e.r});if(e.to===b){const path=[];let v=b;while(v!==a){const z=prev.get(v);path.push({from:z.from,to:v,r:z.r});v=z.from}return path.reverse()}q.push(e.to)}}return null}
function hgStepRelationTex(z,i){
  const r=z.r,q=explicitHGPair[r.key],u=z.from,v=z.to,A=`z_{${i}}`,B=`z_{${i+1}}`;
  if(q?.kind==='directed_pullback'){
    const src=q.source_is==='a'?r.a:r.b,tgt=q.source_is==='a'?r.b:r.a,phi=String(q.pullback.algebraic.phi_tex||'');
    return u===src&&v===tgt?texRenameVar(texRenameVar(phi,'x',A),'y',B):texRenameVar(texRenameVar(phi,'x',B),'y',A)
  }
  if(q?.kind==='common_triangle_cover'){
    const phi=String(q.algebraic.phi_tex||'');return u===r.a?texRenameVar(texRenameVar(phi,'x',A),'y',B):texRenameVar(texRenameVar(phi,'x',B),'y',A)
  }
  return ''
}
function hgStepIdentityHTML(z,i){
  const r=z.r,q=explicitHGPair[r.key],u=z.from,v=z.to,A=byId.get(u),B=byId.get(v),X=`z_{${i}}`,Y=`z_{${i+1}}`;
  let out=`<div class="source-note"><b>${i+1}.</b> Δ${esc(A?.signature||u)} ↔ Δ${esc(B?.signature||v)}</div>`;
  if(q?.kind==='directed_pullback'){
    const p=q.pullback,src=q.source_is==='a'?r.a:r.b,tgt=q.source_is==='a'?r.b:r.a,forward=u===src&&v===tgt;
    const Fx=forward?texRenameVar(p.source_solution_tex,'x',X):texRenameVar(p.target_solution_tex,'y',X);
    const Fy=forward?texRenameVar(p.target_solution_tex,'y',Y):texRenameVar(p.source_solution_tex,'x',Y);
    const map=forward?texRenameVar(p.map_tex,'y',Y):texRenameVar(p.map_tex,'y',X);
    const fac=forward?texRenameVar(p.prefactor_tex,'y',Y):texReciprocal(texRenameVar(p.prefactor_tex,'y',X));
    const phi=forward?texRenameVar(texRenameVar(p.algebraic?.phi_tex||'','x',X),'y',Y):texRenameVar(texRenameVar(p.algebraic?.phi_tex||'','x',Y),'y',X);
    out+=String.raw`<div class="mathline tiny formula-scroll">\[F_{${i}}(${X}):=${Fx},\qquad F_{${i+1}}(${Y}):=${Fy}.\]</div>`;
    out+=forward?String.raw`<div class="mathline tiny formula-scroll">\[${X}=${map},\qquad \Phi_{${i+1}}(${X},${Y})=${phi}=0.\]</div>`:String.raw`<div class="mathline tiny formula-scroll">\[${Y}=${map},\qquad \Phi_{${i+1}}(${X},${Y})=${phi}=0.\]</div>`;
    if(p.local_definitions_tex){const ld=texRenameVar(p.local_definitions_tex,'y',forward?Y:X);out+=String.raw`<div class="mathline tiny formula-scroll">\[${ld}.\]</div>`}
    out+=String.raw`<div class="mathline small formula-scroll pullback-ratio">\[\boxed{\frac{F_{${i}}(${X})}{F_{${i+1}}(${Y})}=${fac}}.\]</div>`;
    if(p.field_tex)out+=String.raw`<div class="mathline tiny formula-scroll">\[${p.field_tex}.\]</div>`+fieldGeneratorHTML(p.field_tex,p);
    return out
  }
  if(q?.kind==='common_triangle_cover'){
    const forward=u===r.a,Fa=forward?q.a_solution_tex:q.b_solution_tex,Fb=forward?q.b_solution_tex:q.a_solution_tex;
    const Fx=texRenameVar(Fa,'x',X),Fy=texRenameVar(Fb,'x',Y),xm=forward?q.x_of_t_tex:q.y_of_t_tex,ym=forward?q.y_of_t_tex:q.x_of_t_tex;
    const ratio=forward?(q.quotient_tex||`\\frac{${q.a_prefactor_tex}}{${q.b_prefactor_tex}}`):texReciprocal(q.quotient_tex||`\\frac{${q.a_prefactor_tex}}{${q.b_prefactor_tex}}`);
    out+=String.raw`<div class="mathline tiny formula-scroll">\[F_{${i}}(${X}):=${Fx},\qquad F_{${i+1}}(${Y}):=${Fy}.\]</div><div class="mathline tiny formula-scroll">\[${X}=${xm},\qquad ${Y}=${ym}.\]</div><div class="mathline small formula-scroll pullback-ratio">\[\boxed{\frac{F_{${i}}(${X})}{F_{${i+1}}(${Y})}=${ratio}}.\]</div>`;
    return out
  }
  if(q?.kind==='common_parameter_identity'){
    const forward=u===r.a,Fa=forward?q.a_solution_tex:q.b_solution_tex,Fb=forward?q.b_solution_tex:q.a_solution_tex;
    const Fx=texRenameVar(Fa,forward?'x':'y',X),Fy=texRenameVar(Fb,forward?'y':'x',Y),xm=forward?q.a_map_tex:q.b_map_tex,ym=forward?q.b_map_tex:q.a_map_tex,ratio=forward?q.quotient_tex:texReciprocal(q.quotient_tex);
    const t=q.parameter||'t';
    out+=String.raw`<div class="mathline tiny formula-scroll">\[F_{${i}}(${X}):=${Fx},\qquad F_{${i+1}}(${Y}):=${Fy}.\]</div><div class="mathline tiny formula-scroll">\[${X}=${xm},\qquad ${Y}=${ym},\qquad t=${t}.\]</div><div class="mathline small formula-scroll pullback-ratio">\[\boxed{\frac{F_{${i}}(${X})}{F_{${i+1}}(${Y})}=${ratio}}.\]</div>`;
    if(q.field_tex)out+=String.raw`<div class="mathline tiny formula-scroll">\[${q.field_tex}.\]</div>`+fieldGeneratorsHTML(q.field_tex,q);
    return out
  }
  const phi=hgStepRelationTex(z,i);if(phi)out+=String.raw`<div class="mathline tiny formula-scroll">\[\Phi_{${i+1}}(${X},${Y})=${phi}=0.\]</div>`;return out
}
function exactHGChainHTML(path,r,g,h){
  if(!path?.length)return'';
  const c=hgComposed[r?.key]||null,n=path.length;
  let out=String.raw`<div class="relation-head">\({}_{2}F_{1}\) · ${n}</div><div class="mathline tiny formula-scroll">\[z_{0}=x,\qquad z_{${n}}=y.\]</div>`;
  for(let i=0;i<n;i++)out+=hgStepIdentityHTML(path[i],i);
  if(c?.factor_tex){
    const fullResultant=String(c.factor_tex).length<=8000;
    out+=fullResultant?String.raw`<div class="relation-head">\(\Phi(x,y)\)</div><div class="mathline small formula-scroll">\[\boxed{\Phi_{\Gamma,\Gamma'}(x,y)=${c.factor_tex}=0}.\]</div>`:String.raw`<div class="relation-head">\(\Phi(x,y)\)</div><div class="mathline tiny formula-scroll">\[\deg_x\Phi=${c.degree_x??'?'},\qquad \deg_y\Phi=${c.degree_y??'?'}.\]</div>`;
    if(c.field_tex)out+=String.raw`<div class="mathline tiny formula-scroll">\[${c.field_tex}.\]</div>`;
    if(c.scalar_composable&&c.quotient_tex){
      out+=String.raw`<div class="relation-head">\(F_\Gamma/F_{\Gamma'}\)</div><div class="mathline tiny formula-scroll">\[F_{\Gamma}^{\mathrm{loc}}(x):=${c.start_solution_tex},\qquad F_{\Gamma'}^{\mathrm{loc}}(y):=${c.end_solution_tex}.\]</div><div class="mathline formula-scroll pullback-ratio">\[\boxed{\frac{F_{\Gamma}^{\mathrm{loc}}(x)}{F_{\Gamma'}^{\mathrm{loc}}(y)}=${c.quotient_tex}}.\]</div>`
    }
  }
  return out
}
function exactInclusionPathHTML(path){if(!path?.length)return'';let out='<div class="relation-head">Belyi · chain</div>';for(let i=0;i<path.length;i++){const e=path[i],br=e.belyi_relation,pp=e.passports?.[0]||[];out+=String.raw`<div class="source-note"><b>${i+1}.</b> \(\Delta${esc(String(e.source).replace(/inf/g,'\\infty'))}\leftarrow\Delta${esc(String(e.target).replace(/inf/g,'\\infty'))},\ d=${e.index}\)</div>`;if(br?.x_of_y_tex)out+=String.raw`<div class="mathline tiny formula-scroll">\[x=${br.x_of_y_tex}.\]</div>`;if(pp.length===3)out+=String.raw`<div class="mathline tiny formula-scroll">\[(\lambda_0,\lambda_1,\lambda_\infty)=\left(${pp.map(partitionTex).join(',')}\right).\]</div>`}out+='<div class="source-note">Every displayed rational map/passport is taken from the exact recorded inclusion data. A hypergeometric gauge factor is shown only when it has separately passed the scalar-pullback verification.</div>';return out}
function relationStatusLabel(r){const q=explicitHGPair[r?.key];if(q?.kind==='directed_pullback')return'↤';if(q?.kind==='common_triangle_cover'||q?.kind==='common_parameter_identity')return'↔';const p=r?verifiedHGPath(r.a,r.b):null;if(p?.length>1)return'∘';return r?.status==='exact_triangle_inclusion'?'⊂':r?.status==='exact_inclusion_chain'?'Φ':r?.status==='commensurable_via_recorded_intersections'?'∩':'≈'}
function peerId(r,g){return r.a===g.id?r.b:r.a}
function inclusionPassportHTML(e){
  if(!e)return'';const pp=e.passports?.[0]||[],amb=(e.passports?.length||0)>1?' (equal-ord marking ambiguity)':'';
  let out=String.raw`<div class="mathline small formula-scroll">\[X_{\Delta${esc(e.target.replace(/inf/g,'\\infty'))}}\longrightarrow X_{\Delta${esc(e.source.replace(/inf/g,'\\infty'))}},\qquad d=${e.index}.\]</div>`;
  if(e.belyi_relation?.x_of_y_tex)out+=String.raw`<div class="mathline small formula-scroll">\[x=${e.belyi_relation.x_of_y_tex}.\]</div>`;
  if(pp.length===3)out+=String.raw`<div class="mathline tiny formula-scroll">\[(\lambda_0,\lambda_1,\lambda_\infty)=\left(${pp.map(partitionTex).join(',')}\right).\]</div><div class="source-note">exact ordered passport${amb}</div>`;
  return out
}
function texRenameVar(s,from,to){return String(s??'').replace(new RegExp('\\b'+from+'\\b','g'),to)}
function texSwapXY(s){return String(s??'').replace(/\bx\b/g,'__ATLAS_X__').replace(/\by\b/g,'x').replace(/__ATLAS_X__/g,'y')}
function texReciprocal(s){s=String(s??'1');return s==='1'?'1':`\\frac{1}{\\left(${s}\\right)}`}
function branchTex(b){return b===2?'\\infty':String(b)}
function pairDetail(g,r){
  if(!r)return'';const pid=peerId(r,g),h=byId.get(pid);if(!h)return'';const q=explicitHGPair[r.key];
  let out=String.raw`<div class="pair-detail"><div class="mathline tiny formula-scroll">\[F_{\Delta${sig(g)}}^{[0]}(x):=${standardHGTex(g,'x')},\qquad F_{\Delta${sig(h)}}^{[0]}(y):=${standardHGTex(h,'y')}.\]</div>`;
  if(q?.kind==='directed_pullback'){
    const p=q.pullback,sourceId=q.source_is==='a'?r.a:r.b,currentIsSource=g.id===sourceId,br=branchTex(p.branch);
    out+=String.raw`<div class="relation-head">\(\Phi^*\)</div>`;
    if(currentIsSource){
      const inv=p.algebraic.inverse_tex,Fx=p.source_solution_tex,Fy=p.target_solution_tex;
      out+=String.raw`<div class="mathline small formula-scroll">\[F_{\Delta${sig(g)}}(x):=${Fx}.\]</div><div class="mathline small formula-scroll">\[F_{\Delta${sig(h)}}(y):=${Fy}.\]</div>`;
      out+=String.raw`<div class="mathline small formula-scroll">\[x=${p.map_tex},\qquad \Phi(x,y)=${p.algebraic.phi_tex}=0.\]</div>`;
      if(inv)out+=String.raw`<div class="mathline small formula-scroll">\[${inv}.\]</div>`;
      out+=String.raw`<div class="mathline formula-scroll pullback-ratio">\[\boxed{\frac{${Fx}}{${Fy}}=${p.prefactor_tex}},\qquad x=${p.map_tex}.\]</div>`;
    }else{
      const Fx=texRenameVar(p.target_solution_tex,'y','x'),Fy=texRenameVar(p.source_solution_tex,'x','y'),map=texRenameVar(p.map_tex,'y','x'),phi=texSwapXY(p.algebraic.phi_tex),fac=texReciprocal(texRenameVar(p.prefactor_tex,'y','x'));
      out+=String.raw`<div class="mathline small formula-scroll">\[F_{\Delta${sig(g)}}(x):=${Fx}.\]</div><div class="mathline small formula-scroll">\[F_{\Delta${sig(h)}}(y):=${Fy}.\]</div>`;
      out+=String.raw`<div class="mathline small formula-scroll">\[y=${map},\qquad \Phi(x,y)=${phi}=0.\]</div>`;
      out+=String.raw`<div class="mathline formula-scroll pullback-ratio">\[\boxed{\frac{${Fx}}{${Fy}}=${fac}},\qquad y=${map}.\]</div>`;
    }
    if(p.field_tex){out+=String.raw`<div class="mathline tiny formula-scroll">\[${p.field_tex}.\]</div>`;out+=fieldGeneratorHTML(p.field_tex,p)}
    out+=String.raw`<div class="source-note">The two functions in the boxed quotient are the explicit parameter-substituted local solutions defined immediately above. Source orbifold branch: \(${br}\). The substitution, pulled differential operator, algebraic gauge factor and local normalization are exact; no generic transformation formula is substituted silently.${q.reference?` ${esc(q.reference)}`:''}</div>`;
  }else if(q?.kind==='common_triangle_cover'){
    const currentIsA=g.id===r.a,ks=String(q.common_signature).replace(/inf/g,'\\infty');
    const Fx=currentIsA?q.a_solution_tex:q.b_solution_tex,Fy=texRenameVar(currentIsA?q.b_solution_tex:q.a_solution_tex,'x','y');
    const xMap=currentIsA?q.x_of_t_tex:q.y_of_t_tex,yMap=currentIsA?q.y_of_t_tex:q.x_of_t_tex;
    const phi=currentIsA?q.algebraic.phi_tex:texSwapXY(q.algebraic.phi_tex);
    const inv=currentIsA?q.algebraic.inverse_tex:(q.algebraic.inverse_x_tex?texSwapXY(q.algebraic.inverse_x_tex):null);
    const cPref=currentIsA?q.a_prefactor_tex:q.b_prefactor_tex,pPref=currentIsA?q.b_prefactor_tex:q.a_prefactor_tex,quotient=`\\frac{${cPref}}{${pPref}}`;
    out+=String.raw`<div class="relation-head">\(\Phi\)</div><div class="mathline small formula-scroll">\[F_{\Delta${sig(g)}}(x):=${Fx}.\]</div><div class="mathline small formula-scroll">\[F_{\Delta${sig(h)}}(y):=${Fy}.\]</div>`;
    out+=String.raw`<div class="mathline small formula-scroll">\[x=${xMap},\qquad y=${yMap},\qquad t\in X_{\Delta${ks}}.\]</div><div class="mathline small formula-scroll">\[\Phi(x,y)=${phi}=0.\]</div>`;
    if(inv)out+=String.raw`<div class="mathline small formula-scroll">\[${inv}.\]</div>`;
    out+=String.raw`<div class="mathline formula-scroll pullback-ratio">\[\boxed{\frac{${Fx}}{${Fy}}=${quotient}},\qquad x=x(t),\ y=y(t).\]</div>`;
    out+=String.raw`<details class="math-details"><summary>\(\Delta\) · \({}_{2}F_{1}\)</summary><div class="mathline tiny formula-scroll">\[F_{\Delta${ks}}(t)=${q.common_solution_tex}.\]</div><div class="mathline tiny formula-scroll">\[${Fx}=(${cPref})F_{\Delta${ks}}(t),\qquad ${Fy}=(${pPref})F_{\Delta${ks}}(t).\]</div></details><div class="source-note">Both concrete local solutions were checked after exact pullback to the same triangle equation. The quotient is displayed only when this verification exists.</div>`;
  }else if(q?.kind==='common_parameter_identity'){
    const currentIsA=g.id===r.a,Fx=currentIsA?q.a_solution_tex:texRenameVar(q.b_solution_tex,'y','x'),Fy=currentIsA?q.b_solution_tex:texRenameVar(q.a_solution_tex,'x','y');
    const xm=currentIsA?q.a_map_tex:q.b_map_tex,ym=currentIsA?q.b_map_tex:q.a_map_tex,fac=currentIsA?q.quotient_tex:texReciprocal(q.quotient_tex),t=q.parameter||'t';
    out+=String.raw`<div class="relation-head">\(\Phi^*\)</div><div class="mathline small formula-scroll">\[F_{\Delta${sig(g)}}(x):=${Fx},\qquad F_{\Delta${sig(h)}}(y):=${Fy}.\]</div><div class="mathline small formula-scroll">\[x=${xm},\qquad y=${ym},\qquad t=${t}.\]</div><div class="mathline formula-scroll pullback-ratio">\[\boxed{\frac{${Fx}}{${Fy}}=${fac}}.\]</div>`;
    if(q.field_tex)out+=String.raw`<div class="mathline tiny formula-scroll">\[${q.field_tex}.\]</div>`+fieldGeneratorsHTML(q.field_tex,q);
  }else{
    const chain=verifiedHGPath(g.id,h.id);if(chain?.length>1)out+=exactHGChainHTML(chain,r,g,h);
    if((!chain||chain.length<=1)&&r.path?.length>1)out+=exactInclusionPathHTML(r.path);
    if(r.endpoint_relation){const z=r.endpoint_relation;out+=String.raw`<div class="relation-head">\(\Phi(x,y)=0\)</div><div class="mathline small formula-scroll">\[\Phi(x,y)=${z.phi_tex}=0.\]</div>`}
    else if(r.direct)out+=inclusionPassportHTML(r.direct);
    else if(r.smallest_recorded_nontriangle_intersection){const z=r.smallest_recorded_nontriangle_intersection;out+=String.raw`<div class="mathline small">\[K:\ ${esc(String(z.signature).replace(/inf/g,'\\infty'))},\qquad g(K)=${z.genus}.\]</div>`}
    out+=chain?.length>1?String.raw`<div class="source-note">The multi-step coordinate correspondence is now eliminated to the \(\Phi(x,y)\) above; a scalar quotient is shown only when the intermediate local solutions actually cancel.</div>`:(r.path?.length>1?String.raw`<div class="source-note">The exact Belyi/inclusion path is no longer omitted. Where a step lacks a separately verified scalar gauge, v21 displays its exact map/passport but does not invent a 2F1 quotient.</div>`:String.raw`<div class="source-note">No scalar quotient is displayed here: the database currently gives the exact correspondence/intersection certificate shown above, but not a verified common hypergeometric pullback for these two local solutions.</div>`);
  }
  out+='</div>';return out
}
function kummerForms(A,B,Cc){
  const O=[1,1],T=[2,1],oneMinusC=rqSub(O,Cc),ac1=rqAdd(rqSub(A,Cc),O),bc1=rqAdd(rqSub(B,Cc),O),twoMinusC=rqSub(T,Cc),ab1c=rqSub(rqAdd(rqAdd(A,B),O),Cc),cab=rqSub(rqSub(Cc,A),B),ca=rqSub(Cc,A),cb=rqSub(Cc,B),cab1=rqAdd(cab,O),ab1=rqAdd(rqSub(A,B),O),ba1=rqAdd(rqSub(B,A),O);
  const pow=(base,e)=>e?.[0]===0?'':`${base}^{${rqTex(e)}}`,prod=(...xs)=>xs.filter(Boolean).join('\\,');
  const bases=[
    {at:'0',pre:'',a:A,b:B,c:Cc,t:'z',om:'1-z',pf:'\\frac{z}{z-1}'},
    {at:'0',pre:pow('z',oneMinusC),a:ac1,b:bc1,c:twoMinusC,t:'z',om:'1-z',pf:'\\frac{z}{z-1}'},
    {at:'1',pre:'',a:A,b:B,c:ab1c,t:'1-z',om:'z',pf:'\\frac{z-1}{z}'},
    {at:'1',pre:pow('(1-z)',cab),a:ca,b:cb,c:cab1,t:'1-z',om:'z',pf:'\\frac{z-1}{z}'},
    {at:'\\infty',pre:pow('z',rqNeg(A)),a:A,b:ac1,c:ab1,t:'z^{-1}',om:'\\frac{z-1}{z}',pf:'\\frac1{1-z}'},
    {at:'\\infty',pre:pow('z',rqNeg(B)),a:B,b:bc1,c:ba1,t:'z^{-1}',om:'\\frac{z-1}{z}',pf:'\\frac1{1-z}'}
  ];const forms=[];for(const q of bases){const e=rqSub(rqSub(q.c,q.a),q.b),ca0=rqSub(q.c,q.a),cb0=rqSub(q.c,q.b);forms.push({at:q.at,tex:prod(q.pre,hfTex(q.a,q.b,q.c,q.t))},{at:q.at,tex:prod(q.pre,pow(`\\left(${q.om}\\right)`,e),hfTex(ca0,cb0,q.c,q.t))},{at:q.at,tex:prod(q.pre,pow(`\\left(${q.om}\\right)`,rqNeg(q.a)),hfTex(q.a,cb0,q.c,q.pf))},{at:q.at,tex:prod(q.pre,pow(`\\left(${q.om}\\right)`,rqNeg(q.b)),hfTex(ca0,q.b,q.c,q.pf))})}return forms
}
function dimAutomorphicSpaces(g,w){
  const orders=[ord(g.a),ord(g.b),ord(g.c)],cusps=orders.filter(x=>x==null).length,ells=orders.filter(x=>x!=null),k=w/2;
  const elliptic=ells.reduce((s,m)=>s+Math.floor(k*(1-1/m)+1e-12),0);
  const M=Math.max(0,-(w-1)+Math.floor(w/2)*cusps+elliptic);
  const S=w===2?0:Math.max(0,-(w-1)+(Math.floor(w/2)-1)*cusps+elliptic);
  return[M,S]
}
function referenceLinks(){
  if(!references.length)return'';
  return `<details class="math-details refs"><summary>${tr('references')}</summary><div class="reference-links">${references.map(r=>`<a href="${esc(r.url)}" target="_blank" rel="noopener">${esc(r.short||r.key||r.citation||r.title||r.url)}</a>`).join('<span>·</span>')}</div></details>`
}
function noncompactSelectedPointHTML(g){
  const q=selectedOrbifoldPoint;if(!q||q.sourceKind!=='cover'||q.sourceRegion?.degree!==1||!Number.isInteger(q.parentVertex))return'';const vals=['1','0','\\infty'],v=vals[q.parentVertex];if(v==null)return'';return String.raw`<div class="divider"></div><div class="mathline small formula-scroll">\[${q.label||q.displayLabel||'P'}:\qquad J_\Gamma=${v}.\]</div>`
}
function noncompactTheory(g){
  const d=noncompactData[g.id];if(!d)return'';
  const rows=[2,4,6,8,10,12].map(w=>{const[m,s]=dimAutomorphicSpaces(g,w);return `<span class="space-row"><b>${w}</b><i>${m}</i><i>${s}</i></span>`}).join('');
  const ff=noncompactFourier[g.id]||{},forms=ff.forms||[],v27nc=noncompactGeneratorSeries[g.id]||{},expandedBases=Array.isArray(v27nc)?v27nc:(v27nc.forms||[]);
  const basisRow=f=>{const e=expandedBases.find(z=>Number(z.weight)===Number(f.weight));let z=String.raw`<div class="relation-head">\(M_{${f.weight}}(\Gamma)\) · ${Number(f.degree)+1}</div>`;if(e?.basis?.length){for(const b of e.basis)z+=String.raw`<div class="mathline tiny formula-scroll">\[f_{${f.weight},${b.ell}}(Q):=f_{${f.weight},\Gamma}(Q)J_\Gamma^{${b.ell}}=${b.series}.\]</div>`;z+=`<div class="source-note">These are the individual generators of the recorded span, obtained by exact truncated Laurent-series multiplication with the stored normalized Hauptmodul. No range notation is used in place of the basis.</div>`}else{z+=String.raw`<div class="mathline tiny formula-scroll">\[f_{${f.weight},\Gamma}(Q)=${f.series}.\]</div><div class="source-note">The stored seed begins beyond the available Fourier truncation, so additional powers of J are not fabricated here.</div>`}return z};
  let fseries='';if(forms.length){fseries=String.raw`<div class="divider"></div><div class="relation-head">\(M_k(\Gamma)\) · \(Q^{60}\)</div>`;for(const f of forms.slice(0,4))fseries+=basisRow(f);if(forms.length>4){fseries+='<details class="math-details"><summary>k</summary>';for(const f of forms.slice(4))fseries+=basisRow(f);fseries+='</details>'}}
  const ncRows=(pairByGroup.get(g.id)||[]).filter(r=>explicitHGPair[r.key]&&noncompactData[peerId(r,g)]),ncPeers=ncRows.length;
  const zser=ff.z_extended?`z(Q)=${ff.z_extended}`:d.z,mir=ff.mirror_extended?`Q(z)=${ff.mirror_extended}`:d.mirror;
  const jq=v27nc?.scaled_J_series?String.raw`<details class="math-details"><summary>\(${v27nc.scale}J_\Gamma\) · \(Q^{60}\)</summary><div class="mathline tiny formula-scroll">\[${v27nc.scale}J_\Gamma=${v27nc.scaled_J_series}.\]</div></details>`:'';
  const point=noncompactSelectedPointHTML(g);
  return String.raw`<details class="math-details nc-theory" open><summary>\(Q,\ J_\Gamma,\ M_k(\Gamma)\)</summary><div class="eq-grid compact"><div class="eq-key">\(\Gamma\)</div><div class="eq-val">\(${d.realization}\)</div><div class="eq-key">\(Q\)</div><div class="eq-val">\(Q=${d.Q},\ q=e^{2\pi i\tau}\)</div></div><div class="mathline small formula-scroll">\[${d.period}.\]</div><div class="mathline tiny formula-scroll">\[${d.J}.\]</div>${jq}${point}<div class="mathline tiny formula-scroll">\[${zser}.\]</div><div class="mathline tiny formula-scroll">\[${mir}.\]</div>${fseries}<div class="divider"></div><div class="mathline tiny formula-scroll">\[${d.ring}.\]</div><div class="space-head"><span>\(k\)</span><span>\(\dim M_k\)</span><span>\(\dim S_k\)</span></div><div class="space-table">${rows}</div>${referenceLinks()}</details>`
}
function uniformPanel(g){
  const h=hauptMap.get(g.id),hg=g.hypergeometric||h?.hypergeometric||{},As=String(hg.A??'0'),Bs=String(hg.B??'0'),Cs=String(hg.C??'1'),A=rqParse(As),B=rqParse(Bs),Cc=rqParse(Cs),delta=(hg.exponent_differences||[]).map(ratTex);
  let out=String.raw`<h2>\({}_{2}F_{1}\)</h2><div class="mathline">\[F_\Gamma^{[0]}(z):={}_{2}F_{1}\!\left(${ratTex(As)},${ratTex(Bs)};${ratTex(Cs)};z\right),\qquad(\delta_0,\delta_1,\delta_\infty)=\left(${delta.join(',')}\right).\]</div>`;
  const prs=(pairByGroup.get(g.id)||[]).slice().sort((u,v)=>{const eu=explicitHGPair[u.key]?0:1,ev=explicitHGPair[v.key]?0:1;if(eu!==ev)return eu-ev;const order={explicit_algebraic_correspondence:0,exact_triangle_inclusion:1,exact_inclusion_chain:2,commensurable_via_recorded_intersections:3,same_arithmetic_commensurability_class:4};return (order[u.status]-order[v.status])||byId.get(peerId(u,g)).signature.localeCompare(byId.get(peerId(v,g)).signature)});
  if(prs.length){if(!selectedHGPeer||!prs.some(r=>peerId(r,g)===selectedHGPeer))selectedHGPeer=peerId(prs[0],g);out+='<div class="hg-pairs">'+prs.map(r=>{const p=byId.get(peerId(r,g));return `<button class="hg-pair ${p.id===selectedHGPeer?'selected':''}" data-hg-peer="${p.id}"><span>\\(\\Delta${sig(p)}\\)</span><span class="pair-status">${esc(relationStatusLabel(r))}</span></button>`}).join('')+'</div>';out+=pairDetail(g,pairFor(g,selectedHGPeer))}
  if(A&&B&&Cc){const forms=kummerForms(A,B,Cc);out+=`<details class="math-details"><summary>Kummer · 24</summary><div class="kummer24">${forms.map((q,i)=>`<div class="kummer-form"><span class="kummer-index">${q.at} · ${i%4+1}</span><div class="formula-scroll">\\[${q.tex}\\]</div></div>`).join('')}</div></details>`}
  if(h?.z_of_u_coefficients?.length)out+=`<details class="math-details"><summary>${tr('localParameter')}</summary><div class="mathline small formula-scroll">\\[z(u)=${seriesTex(h.z_of_u_coefficients,14)}.\\]</div></details>`;
  out+=noncompactTheory(g);
  return out
}
function coversPanel(g){
  const rows=coverRecords(g),numberMap=coverNumberMap(rows),nb=g.id==='T2_3_I'?modularNeighborhood(selectedCover):genericNeighborhood(selectedCover,rows);window.__coverRows=new Map(rows.map(r=>[r.key,r]));window.__coverNumbers=numberMap;let out=String.raw`<h2>\(X_H\)</h2>`;if(g.id==='T2_3_I')out+=moonshineCatalogueHTML();
  if(selectedCover)out+=`<div class="cover-detail primary">${coverDetail(selectedCover)}</div>`;
  if(nb.sup.length){out+=`<div class="neighbor-title">${tr('supergroups')}</div>`+nb.sup.map(r=>coverRow(r,selectedCover?normalityBetween(selectedCover,r):null)).join('')}if(nb.sub.length){out+=`<div class="neighbor-title">${tr('subgroups')}</div>`+nb.sub.map(r=>coverRow(r,selectedCover?normalityBetween(r,selectedCover):null)).join('')}
  if(!selectedCover)out+=`<div class="cover-detail">${coverDetail(null)}</div>`;
  const census=rows.filter(r=>r.census||r.lowIndex);if(census.length){const groups=lowIndexClassGroups(rows),meta=lowIndexMeta[g.id]||{},tree=subgroupClassTree(g,groups,numberMap),rooted=census.filter(r=>r.conjugacy_class===false).length;out+=`<details class="math-details subgroup-tree-block" open><summary>${tr('lowIndex')} · ${groups.length}</summary><div class="source-note">${census.length} distinct rooted low-[·]=subgroups · ${meta.genus0_classes??groups.filter(q=>q.rep.genus===0).length} genus-0 conjugacy classes${meta.complete_through_index?` · exhaustive through [·]=${meta.complete_through_index}`:''}${meta.verified_nonexhaustive_classes?` · ${meta.verified_nonexhaustive_classes} additionally verified non-exhaustive classes`:''}.</div><div class="subgroup-tree-scroll">${tree.svg}</div></details><details class="math-details"><summary>${tr('conjugacyClasses')} · ${groups.length}</summary>${subgroupClassDetails(groups,numberMap)}</details>`;}
  const notable=rows.filter(r=>r.higher);if(notable.length)out+=`<details class="math-details"><summary>${tr('higherGenus')}</summary>${notable.map(coverRow).join('')}</details>`;
  return out
}
function commRelTriple(r){return r?.triple||((r?.modular_label&&modularPerm[r.modular_label]?.triple)||null)}
function directTriangleCover(parent,child){
  const x=exactTriangleCoverMap.get(parent+'|'+child);if(x)return x;
  return noncompactRelations.find(r=>r.parent===parent&&r.child===child&&commRelTriple(r))||null
}
function exactQuadRecord(id){
  const q=exactQuadMap.get(id);if(q)return q;const z=quadrilateralGroups.find(x=>x.id===id);if(!z)return null;
  return {...z,class:1,triple:z.triple||((z.modular_label&&modularPerm[z.modular_label]?.triple)||null),source:'exact modular coset action'}
}
function quadrilateralsComparableFrom(g){
  const cl=Number(g.commensurability_class),raw=exactQuadrilateralCovers.filter(q=>Number(q.class)===cl).concat(cl===1?quadrilateralGroups.map(q=>({...q,class:1,triple:q.triple||((q.modular_label&&modularPerm[q.modular_label]?.triple)||null)})):[]),seen=new Set(),out=[];
  for(const q of raw){if(seen.has(q.id)||!commRelTriple(q))continue;if(q.parent===g.id||triangleInclusionSpec(q.parent,g.id)){seen.add(q.id);out.push(q)}}
  return out.sort((a,b)=>(a.parent===g.id?0:1)-(b.parent===g.id?0:1)||a.degree-b.degree||String(a.signature).localeCompare(String(b.signature))||String(a.id).localeCompare(String(b.id)))
}
function quadrilateralsVisibleFrom(g){return quadrilateralsComparableFrom(g)}
function reduceTriangleEdges(edges){
  const uniq=new Map();for(const e of edges){if(!e.parent||!e.child||e.parent===e.child)continue;const k=e.parent+'|'+e.child;if(!uniq.has(k)||(!uniq.get(k).triple&&e.triple))uniq.set(k,e)}const es=[...uniq.values()],adj=new Map();for(const e of es){if(!adj.has(e.parent))adj.set(e.parent,[]);adj.get(e.parent).push(e.child)}
  const reaches=(a,b,skip)=>{const q=[a],seen=new Set([a]);for(const u of q)for(const v of adj.get(u)||[]){if(u===skip.parent&&v===skip.child)continue;if(v===b)return true;if(!seen.has(v)){seen.add(v);q.push(v)}}return false};return es.filter(e=>!reaches(e.parent,e.child,e))
}
function commObjects(g){
  const tri=CORE.filter(x=>x.commensurability_class===g.commensurability_class).map(x=>({id:x.id,label:`Δ${x.signature.replace(/inf/g,'∞')}`,area:Number(x.orbifold_area_over_pi),triangle:true,obj:x}));
  const qs=quadrilateralsVisibleFrom(g).map(q=>({...q,label:q.label||String(q.signature).replace(/inf/g,'∞'),area:Number(byId.get(q.parent)?.orbifold_area_over_pi||0)*Number(q.degree),triangle:false,quad:true})),all=tri.concat(qs).sort((a,b)=>Number(a.area||0)-Number(b.area||0)||(a.triangle===b.triangle?0:(a.triangle?-1:1))||String(a.label).localeCompare(String(b.label))||String(a.id).localeCompare(String(b.id)));
  all.forEach((o,i)=>o.number=i+1);return all
}
function commObjectNumber(G,id){return G?.objects?.find(o=>o.id===id)?.number||null}
function commEdges(g){
  const cl=Number(g.commensurability_class),tri=[];
  for(const e of exactTriangleCovers)if(Number(byId.get(e.parent)?.commensurability_class)===cl)tri.push({...e,index:e.degree,source:e.parent,target:e.child,exactGeometry:true});
  if(cl===1)for(const r of noncompactRelations)if(!(r.parent==='T2_3_I'&&r.child==='TIII'))tri.push({...r,index:r.degree,source:r.parent,target:r.child,exactGeometry:!!commRelTriple(r)});
  for(let i=0;i<inclusions.length;i++){const r=inclusions[i];if(Number(r.class)!==cl)continue;const p=bySig.get(r.source)?.id,c=bySig.get(r.target)?.id;if(p&&c&&!tri.some(e=>e.parent===p&&e.child===c))tri.push({...r,key:`legacy:${cl}:${i}`,parent:p,child:c,index:r.index,degree:r.index,exactGeometry:false})}
  const es=reduceTriangleEdges(tri);for(const q of quadrilateralsVisibleFrom(g))es.push({key:'quad:'+q.id,parent:q.parent,child:q.id,source:q.parent,target:q.id,index:q.degree,degree:q.degree,triple:q.triple||commRelTriple(q),quadrilateral:true,exactGeometry:true});return es
}
function commClassGraph(g){
  const objects=commObjects(g),edges=commEdges(g),ids=new Set(objects.map(o=>o.id)),depth=new Map(objects.map(o=>[o.id,0]));
  // Longest-path layering in the directed supergroup -> subgroup DAG.  This is
  // more legible than putting every equal-area object on one horizontal row.
  for(let pass=0;pass<objects.length+2;pass++){let changed=false;for(const e of edges){const a=e.parent||e.source,b=e.child||e.target;if(!ids.has(a)||!ids.has(b))continue;const d=(depth.get(a)||0)+1;if(d>(depth.get(b)||0)){depth.set(b,d);changed=true}}if(!changed)break}
  const byDepth=new Map();for(const m of objects){const d=depth.get(m.id)||0;if(!byDepth.has(d))byDepth.set(d,[]);byDepth.get(d).push(m)}
  for(const row of byDepth.values())row.sort((a,b)=>(a.triangle===b.triangle?0:(a.triangle?-1:1))||String(a.parent||'').localeCompare(String(b.parent||''))||Number(a.degree||0)-Number(b.degree||0)||a.label.localeCompare(b.label));
  const maxPer=6,w=660,rowH=82,gapDepth=26,top=34,pos=new Map();let y=top;
  for(const d of [...byDepth.keys()].sort((a,b)=>a-b)){const row=byDepth.get(d),chunks=[];for(let i=0;i<row.length;i+=maxPer)chunks.push(row.slice(i,i+maxPer));for(const chunk of chunks){chunk.forEach((m,i)=>pos.set(m.id,{x:38+(i+.5)*(w-76)/chunk.length,y,m}));y+=rowH}y+=gapDepth}
  const h=Math.max(150,y-20);let svg=`<svg class="math-graph comm-lattice" viewBox="0 0 ${w} ${h}">`;
  for(const rel of edges){const A=pos.get(rel.parent||rel.source),B=pos.get(rel.child||rel.target);if(!A||!B)continue;const ym=(A.y+B.y)/2;svg+=`<path class="edge ${rel.exactGeometry?'strong':'legacy'}" d="M${A.x},${A.y+8} C${A.x},${ym} ${B.x},${ym} ${B.x},${B.y-10}"/>`}
  for(const m of objects){const q=pos.get(m.id),active=m.id===g.id||m.id===selectedCommPeer;if(!q)continue;svg+=`<g data-comm-peer="${m.id}" class="graph-hit"><circle class="node ${m.triangle?'':'quad-node'} ${active?'selected':''}" cx="${q.x}" cy="${q.y}" r="${active?8:6.5}"/><text class="node-number" text-anchor="middle" x="${q.x}" y="${q.y-11}">${m.number}</text><text text-anchor="middle" x="${q.x}" y="${q.y+23}">${esc(m.label)}</text><title>#${m.number} · ${esc(m.label)}</title></g>`}svg+='</svg>';return{svg,objects,edges,levels:byDepth}
}
function shortestCommPath(gid,pid,edges){
  if(gid===pid)return[];const adj=new Map();for(const e of edges){const a=e.parent||e.source,b=e.child||e.target;if(!a||!b)continue;for(const [u,v] of [[a,b],[b,a]]){if(!adj.has(u))adj.set(u,[]);adj.get(u).push([v,e])}}
  const q=[gid],prev=new Map([[gid,null]]);for(const u of q){for(const [v,e] of adj.get(u)||[])if(!prev.has(v)){prev.set(v,[u,e]);q.push(v)}}if(!prev.has(pid))return[];const out=[];let v=pid;while(v!==gid){const [u,e]=prev.get(v);out.push(e);v=u}return out.reverse()
}
function triangleInclusionSpec(base,id){
  if(base===id)return{degree:1,edge:null,triple:null,targetCorners:3};
  const e=directTriangleCover(base,id),t=commRelTriple(e);return e&&t?{degree:e.degree||e.index,edge:e,triple:t,targetCorners:3}:null
}
function buildRegionFromSpec(spec){return spec.degree===1?baseRegion():regionFromTriple(spec.triple,spec.targetCorners||0)}
function commonTriangleSupergroup(a,b){
  const ga=byId.get(a),gb=byId.get(b);if(!ga||!gb||ga.commensurability_class!==gb.commensurability_class)return null;const cand=[];
  for(const S of CORE)if(S.commensurability_class===ga.commensurability_class){const A=triangleInclusionSpec(S.id,a),B=triangleInclusionSpec(S.id,b);if(A&&B)cand.push({S,A,B,score:Number(S.orbifold_area_over_pi)+1e-5*(A.degree+B.degree)})}
  return cand.sort((u,v)=>u.score-v.score)[0]||null
}
function activateFricke(N){
  N=Number(N);const rec=frickeDomains[String(N)];if(!rec)return false;const plus=fordRegion(rec),g0=fordRegion(gamma0FordDomains[String(N)]),base=byId.get('T2_3_I')||current;if(!plus)return false;
  selectedMoonshine=null;SYS=makeSystem(base);regionCache=new Map();selectedCover=null;commCompare=null;frickeCompare={N,name:frickeClasses[String(N)]||`${N}A`,plus,gamma0:g0};commSceneActive=true;fitCurrentRegion();invalidate(true);return true
}
function activateCommPeer(id){
  commCompare=null;selectedCover=null;const q=exactQuadRecord(id);
  if(q){
    const S=byId.get(q.parent),A=S?triangleInclusionSpec(S.id,current.id):null,qt=q.triple||commRelTriple(q);if(!S||!A||!qt)return false;
    SYS=makeSystem(S);regionCache=new Map();const AR=buildRegionFromSpec(A),BR=regionFromTriple(qt,4);
    commCompare={base:S,a:{id:current.id,R:AR,degree:A.degree,record:A.edge,short:`F_${current.signature.replace(/inf/g,'∞')}`},b:{id:q.id,R:BR,degree:q.degree,record:q,short:`F_${String(q.signature).replace(/inf/g,'∞')}`},baseShort:`S=Δ${S.signature.replace(/inf/g,'∞')}`};commSceneActive=true;fitCurrentRegion();return true
  }
  const h=byId.get(id),C=h?commonTriangleSupergroup(current.id,h.id):null;if(!C)return false;
  SYS=makeSystem(C.S);regionCache=new Map();const AR=buildRegionFromSpec(C.A),BR=buildRegionFromSpec(C.B);
  commCompare={base:C.S,a:{id:current.id,R:AR,degree:C.A.degree,record:C.A.edge,short:`F_${current.signature.replace(/inf/g,'∞')}`},b:{id:h.id,R:BR,degree:C.B.degree,record:C.B.edge,short:`F_${h.signature.replace(/inf/g,'∞')}`},baseShort:`S=Δ${C.S.signature.replace(/inf/g,'∞')}`};commSceneActive=true;fitCurrentRegion();return true
}
function activateCommRelation(rel){
  if(!rel)return false;const parent=byId.get(rel.parent||rel.source),triple=commRelTriple(rel);if(!parent||!triple)return false;SYS=makeSystem(parent);selectedCover={key:'comm:'+rel.key,degree:rel.degree||rel.index,triple,comm:true,baseLabel:parent.signature.replace(/inf/g,'∞'),coverLabel:(byId.get(rel.child||rel.target)?.signature||exactQuadRecord(rel.child||rel.target)?.signature||'H').replace(/inf/g,'∞')};regionCache=new Map();commSceneActive=true;fitCurrentRegion();return true
}
function restoreCurrentScene(){if(!commSceneActive&&!commCompare&&!frickeCompare)return;SYS=makeSystem(current);selectedCover=null;selectedMoonshine=null;commCompare=null;frickeCompare=null;regionCache=new Map();commSceneActive=false;fitCurrentRegion()}
function commSelectedBelyiRecord(g,G){
  if(!selectedCommPeer||selectedCommPeer===g.id)return null;
  const candidates=[];
  if(commCompare?.b?.id===selectedCommPeer&&commCompare.b.record)candidates.push(commCompare.b.record);
  const q=exactQuadRecord(selectedCommPeer);if(q)candidates.push(q);
  const peer=G?.objects?.find(o=>o.id===selectedCommPeer);if(peer?.record)candidates.push(peer.record);if(peer?.obj)candidates.push(peer.obj);
  for(const r of candidates)if(strictBelyiMatch(belyiLookup(r)))return r;
  return null
}
function commRelationDetail(g,G){
  const activeId=selectedCommPeer||g.id,peer=G.objects.find(o=>o.id===activeId)||exactQuadRecord(activeId);if(!peer)return'';const name=id=>byId.get(id)?`\\Delta${sig(byId.get(id))}`:(exactQuadRecord(id)?.label||String(exactQuadRecord(id)?.signature||id).replace(/inf/g,'\\infty')),no=commObjectNumber(G,activeId),noTex=no?`\\#${no}\\qquad `:'';
  if(activeId===g.id)return String.raw`<div class="cover-detail primary"><div class="relation-head">\(\mathcal C(\Gamma)\)</div><div class="mathline tiny formula-scroll">\[${noTex}${name(g.id)}.\]</div></div>`;
  let out=String.raw`<div class="cover-detail primary"><div class="relation-head">\(\mathcal C(\Gamma)\)</div><div class="mathline tiny formula-scroll">\[${noTex}${name(selectedCommPeer)}.\]</div>`;
  if(commCompare){const S=commCompare.base,ad=commCompare.a.degree||commCompare.a.R.degree||1,bd=commCompare.b.degree||commCompare.b.R.degree||1;out+=`<div class="mathline small formula-scroll">\\[${name(g.id)}\\longleftarrow \\Delta${sig(S)}\\longrightarrow ${name(selectedCommPeer)}.\\]</div><div class="mathline tiny formula-scroll">\\[[\\Delta${sig(S)}:${name(g.id)}]=${ad},\\qquad [\\Delta${sig(S)}:${name(selectedCommPeer)}]=${bd}.\\]</div><div class="source-note">Solid boundary = current group; dashed boundary = selected group. Both are exact unions of copies of the same triangle fundamental domain in the displayed conjugate realization.</div>`}
  else{const path=shortestCommPath(g.id,selectedCommPeer,G.edges);if(path.length){const labs=[g.id];let u=g.id;for(const r of path){u=u===(r.parent||r.source)?(r.child||r.target):(r.parent||r.source);labs.push(u)}out+=`<div class="mathline small formula-scroll">\\[${labs.map(name).join('\\longleftrightarrow ')}.\\]</div><div class="source-note">No common exact low-[·]=triangle supergroup is currently available for a simultaneous polygon comparison; only the recorded exact chain is shown.</div>`}else out+='<div class="source-note">No exact comparison is recorded for this pair.</div>'}
  const q=exactQuadRecord(selectedCommPeer);if(q){const cw=(q.cusp_widths||[]).join(',')||'\\varnothing',ee=(q.elliptic_orders||[]).join(',')||'\\varnothing';out+=`<div class="mathline tiny formula-scroll">\\[\\operatorname{sig}Q=(0;${ee}),\\qquad w_{\\mathrm{cusp}}=(${cw}),\\qquad [\\Delta${sig(byId.get(q.parent))}:Q]=${q.degree}.\\]</div>`}
  return out+'</div>'
}

function intersectionLabel(r){if(r.type==='triangle')return `Δ${String(r.signature).replace(/inf/g,'∞')}`;return String(r.signature).replace(/inf/g,'∞')}
function commPanel(g){
  const G=commClassGraph(g),extra=commIntersections.filter(r=>r.class===g.commensurability_class&&r.type!=='triangle');let out=String.raw`<h2>\(\mathcal C(\Gamma)\)</h2><div class="graph-wrap">${G.svg}</div>`+commRelationDetail(g,G);const br=commSelectedBelyiRecord(g,G);if(br)out+=belyiMajorHTML(br);
  const qs=quadrilateralsComparableFrom(g),all=exactQuadrilateralCovers.filter(q=>Number(q.class)===Number(g.commensurability_class)),comp=new Set(qs.map(q=>q.id));
  if(qs.length){out+=`<details class="math-details" open><summary>${tr('quadrilaterals')} · ${tr('simultaneous')} (${qs.length})</summary><div class="comm-quad-list">`;for(const q of qs){const no=commObjectNumber(G,q.id);out+=`<button class="comm-quad-row ${selectedCommPeer===q.id?'selected':''}" data-comm-peer="${esc(q.id)}">${no?`<span class="object-number">${no}</span>`:''}<span>\\(${esc(String(q.signature).replace(/inf/g,'\\infty'))}\\)</span><small>[·]=${q.degree} · Δ${esc(byId.get(q.parent)?.signature.replace(/inf/g,'∞')||q.parent)}</small></button>`}out+=`</div><div class="source-note">Every row above has an exact common triangle realization with the current Γ, so clicking compares both fundamental regions without changing Γ.</div></details>`}
  if(all.length){const groups=new Map();for(const q of all){if(!groups.has(q.parent))groups.set(q.parent,[]);groups.get(q.parent).push(q)}out+=`<details class="math-details"><summary>${tr('quadrilaterals')} (${all.length})</summary>`;for(const [pid,arr] of [...groups.entries()].sort((a,b)=>String(byId.get(a[0])?.signature||a[0]).localeCompare(String(byId.get(b[0])?.signature||b[0])))){out+=`<div class="neighbor-title">Δ${esc(byId.get(pid)?.signature.replace(/inf/g,'∞')||pid)} · ${arr.length}</div><div class="comm-quad-list">`;for(const q of arr.sort((a,b)=>a.degree-b.degree||String(a.signature).localeCompare(String(b.signature)))){const no=commObjectNumber(G,q.id),badge=no?`<span class="object-number">${no}</span>`:'';if(comp.has(q.id))out+=`<button class="comm-quad-row ${selectedCommPeer===q.id?'selected':''}" data-comm-peer="${esc(q.id)}">${badge}<span>\\(${esc(String(q.signature).replace(/inf/g,'\\infty'))}\\)</span><small>[·]=${q.degree}</small></button>`;else out+=`<div class="comm-quad-row passive">${badge}<span>\\(${esc(String(q.signature).replace(/inf/g,'\\infty'))}\\)</span><small>[·]=${q.degree}</small></div>`}out+='</div>'}out+='<div class="source-note">Passive rows are exact actions but no compatible simultaneous embedding with the currently selected triangle is stored; they are not made clickable rather than asserting a false polygon comparison.</div></details>'}
  if(extra.length){out+=`<details class="math-details"><summary>${tr('commonSubgroups')} (${extra.length})</summary><div class="intersection-grid">`;for(const r of extra.sort((a,b)=>Number(a.genus)-Number(b.genus)||String(a.signature).localeCompare(String(b.signature)))){const no=commObjectNumber(G,r.id),badge=no?`<span class="object-number">${no}</span>`:'';out+=`<span class="intersection-item">${badge}<span>\\(${esc(intersectionLabel(r))}\\)</span></span>`}out+='</div></details>'}
  return out
}

function panelHTML(name,g){return name==='group'?groupPanel(g):name==='uniformization'?uniformPanel(g):name==='covers'?coversPanel(g):commPanel(g)}
function showPanel(name){if(panel==='comm'&&name!=='comm')restoreCurrentScene();panel=name;document.querySelectorAll('.nav-item').forEach(b=>b.classList.toggle('active',b.dataset.panel===name));setHTML($('sheet-content'),panelHTML(name,current));$('sheet').classList.add('show');$('sheet').setAttribute('aria-hidden','false');document.body.classList.add('sheet-open');invalidate()}
function hidePanel(){$('sheet').classList.remove('show');$('sheet').setAttribute('aria-hidden','true');document.body.classList.remove('sheet-open');invalidate()}

// -----------------------------------------------------------------------------
// Group navigation
// -----------------------------------------------------------------------------
function commClassNumber(g){const n=Number(g?.commensurability_class);return Number.isInteger(n)&&n>=1&&n<=19?n:null}
function groupChoiceTex(g){const c=commClassNumber(g),prefix=c?`C_{${c}}:\\ `:'';return `\\(${prefix}\\Delta${sig(g)}\\)`}
function setGroupButtonLabel(g){const b=$('group-button');if(!b)return;b.innerHTML=groupChoiceTex(g);typeset(b)}
function picker(){const q=$('group-search').value.trim().toLowerCase().replace(/∞/g,'inf').replace(/\s/g,'');const rows=CORE.filter(g=>{const c=commClassNumber(g);return !q||g.signature.toLowerCase().includes(q)||g.id.toLowerCase().includes(q)||`delta${g.signature}`.includes(q)||(c&&(`c${c}`===q||`c_${c}`===q))});const html=rows.map(g=>{const c=commClassNumber(g),h=c?((c-1)*360/19):0;return `<button class="group-choice ${g.id===current.id?'active':''}" style="--comm-hue:${h.toFixed(4)}" data-group="${g.id}">${groupChoiceTex(g)}</button>`}).join('');setHTML($('group-grid'),html)}
function setGroup(id){const g=byId.get(id);if(!g)return;current=g;SYS=makeSystem(g);selectedCover=null;selectedMoonshine=null;selectedOrbifoldPoint=null;selectedHGPeer=null;selectedCommPeer=null;commSceneActive=false;commCompare=null;frickeCompare=null;regionCache=new Map();fitCurrentRegion();history.replaceState(null,'','#'+g.id);setGroupButtonLabel(g);picker();if($('sheet').classList.contains('show'))setHTML($('sheet-content'),panelHTML(panel,current));invalidate();$('group-picker').classList.remove('show')}

// -----------------------------------------------------------------------------
// Events
// -----------------------------------------------------------------------------
$('group-button').onclick=()=>{$('group-picker').classList.toggle('show');$('group-picker').setAttribute('aria-hidden',$('group-picker').classList.contains('show')?'false':'true');picker();if($('group-picker').classList.contains('show'))$('group-search').focus()};
$('picker-close').onclick=()=>$('group-picker').classList.remove('show');$('group-search').oninput=picker;$('group-grid').onclick=e=>{const b=e.target.closest('[data-group]');if(b)setGroup(b.dataset.group)};
document.querySelectorAll('.nav-item').forEach(b=>b.onclick=()=>showPanel(b.dataset.panel));$('sheet-close').onclick=hidePanel;
const sheetResizer=$('sheet-resizer');let sheetDrag=null;
try{const sw=Number(localStorage.getItem('atlas.sheetWidth'));if(Number.isFinite(sw)&&sw>=280)document.documentElement.style.setProperty('--sheet-w',Math.min(sw,1440)+'px')}catch{}
if(sheetResizer){sheetResizer.addEventListener('pointerdown',e=>{if(innerWidth<=900)return;e.preventDefault();const box=$('sheet').getBoundingClientRect();sheetDrag={x:e.clientX,w:box.width};sheetResizer.classList.add('dragging');sheetResizer.setPointerCapture?.(e.pointerId)});window.addEventListener('pointermove',e=>{if(!sheetDrag)return;const nw=Math.max(280,Math.min(Math.min(1440,innerWidth-180),sheetDrag.w+(sheetDrag.x-e.clientX)));document.documentElement.style.setProperty('--sheet-w',nw+'px');try{localStorage.setItem('atlas.sheetWidth',String(Math.round(nw)))}catch{}resize()});window.addEventListener('pointerup',()=>{if(!sheetDrag)return;sheetDrag=null;sheetResizer.classList.remove('dragging');resize()})}

function saveViewSettings(){try{localStorage.setItem('atlas.viewSettings',JSON.stringify(viewSettings))}catch{}}
function loadViewSettings(){try{const x=JSON.parse(localStorage.getItem('atlas.viewSettings')||'{}');for(const k of Object.keys(viewSettings))if(x[k]!=null)viewSettings[k]=x[k];if(x.neighbors===false)viewSettings.neighborCount=0;viewSettings.neighborCount=Math.max(0,Math.min(3,Math.round(Number(viewSettings.neighborCount)||0)));viewSettings.neighbors=true}catch{}}
function syncViewSettingsUI(){
  const m={vertices:'set-vertices',allEdgeVertices:'set-all-edge-vertices',ellipticVertices:'set-elliptic-vertices',cuspVertices:'set-cusp-vertices',pairings:'set-pairings',coverMarksOnly:'set-cover-marks-only',arrows:'set-arrows',edgeLabels:'set-edge-labels',cellSeams:'set-cell-seams',hatching:'set-hatching',triangleSeams:'set-triangle-seams',regionTags:'set-region-tags',legend:'set-legend'};
  for(const[k,id]of Object.entries(m)){const e=$(id);if(e)e.checked=!!viewSettings[k]}
  if($('set-vertex-label'))$('set-vertex-label').value=viewSettings.vertexLabel;if($('set-vertex-size'))$('set-vertex-size').value=viewSettings.vertexSize;if($('set-edge-size'))$('set-edge-size').value=viewSettings.edgeSize;if($('set-boundary-size'))$('set-boundary-size').value=viewSettings.boundarySize;if($('set-neighbor-count'))$('set-neighbor-count').value=viewSettings.neighborCount;if($('set-neighbor-opacity'))$('set-neighbor-opacity').value=viewSettings.neighborOpacity;if($('set-language'))$('set-language').value=uiLang
}
function bindViewSettings(){
  loadViewSettings();loadLanguage();syncViewSettingsUI();applyLanguage({rerender:false});const panel=$('display-settings'),btn=$('display-settings-button'),close=$('display-settings-close');
  if(btn)btn.onclick=()=>{panel.classList.toggle('show');panel.setAttribute('aria-hidden',panel.classList.contains('show')?'false':'true')};if(close)close.onclick=()=>{panel.classList.remove('show');panel.setAttribute('aria-hidden','true')};
  const checks={vertices:'set-vertices',allEdgeVertices:'set-all-edge-vertices',ellipticVertices:'set-elliptic-vertices',cuspVertices:'set-cusp-vertices',pairings:'set-pairings',coverMarksOnly:'set-cover-marks-only',arrows:'set-arrows',edgeLabels:'set-edge-labels',cellSeams:'set-cell-seams',hatching:'set-hatching',triangleSeams:'set-triangle-seams',regionTags:'set-region-tags',legend:'set-legend'};
  for(const[k,id]of Object.entries(checks)){const e=$(id);if(e)e.onchange=()=>{viewSettings[k]=e.checked;if(k==='allEdgeVertices'&&e.checked){viewSettings.vertices=true;if($('set-vertices'))$('set-vertices').checked=true}saveViewSettings();updateLegendVisibility();invalidate()}}
  const ranges={vertexSize:'set-vertex-size',edgeSize:'set-edge-size',boundarySize:'set-boundary-size',neighborCount:'set-neighbor-count',neighborOpacity:'set-neighbor-opacity'};for(const[k,id]of Object.entries(ranges)){const e=$(id);if(e)e.oninput=()=>{viewSettings[k]=k==='neighborCount'?Math.max(0,Math.round(Number(e.value))):Number(e.value);saveViewSettings();invalidate()}}
  if($('set-vertex-label'))$('set-vertex-label').onchange=e=>{viewSettings.vertexLabel=e.target.value;saveViewSettings();invalidate()};
  if($('set-language'))$('set-language').onchange=e=>{uiLang=e.target.value==='zh'?'zh':'en';saveLanguage();applyLanguage();invalidate()}
}
function updateMotionButtons(){const q=$('motion-q'),e=$('motion-e');if(!q||!e)return;if(model==='disk'){q.title='q / , / PageUp';e.title='e / . / PageDown'}else{q.title='q / , / PageUp';e.title='e / . / PageDown'}}
$('disk-button').onclick=()=>{model='disk';$('disk-button').classList.add('active');$('half-button').classList.remove('active');fitCurrentRegion();updateMotionButtons()};$('half-button').onclick=()=>{model='half';$('half-button').classList.add('active');$('disk-button').classList.remove('active');fitCurrentRegion();updateMotionButtons()};$('zoom-in').onclick=()=>zoomBy(1.14);$('zoom-out').onclick=()=>zoomBy(1/1.14);bindViewSettings();updateMotionButtons();
$('sheet-content').onclick=e=>{const mo=e.target.closest('[data-moonshine]');if(mo?.dataset.moonshine){activateMoonshine(mo.dataset.moonshine);if($('sheet').classList.contains('show'))setHTML($('sheet-content'),panelHTML(panel,current));return}const fr=e.target.closest('[data-fricke]');if(fr?.dataset.fricke){activateFricke(Number(fr.dataset.fricke));if($('sheet').classList.contains('show'))setHTML($('sheet-content'),panelHTML(panel,current));return}const hp=e.target.closest('[data-hg-peer]');if(hp?.dataset.hgPeer){selectedHGPeer=hp.dataset.hgPeer;if(panel==='uniformization')showPanel('uniformization');return}const cp=e.target.closest('[data-comm-peer]');if(cp?.dataset.commPeer){selectedCommPeer=cp.dataset.commPeer;restoreCurrentScene();if(selectedCommPeer!==current.id)activateCommPeer(selectedCommPeer);if(panel==='comm')setHTML($('sheet-content'),commPanel(current));return}const j=e.target.closest('[data-group-jump]');if(j?.dataset.groupJump){setGroup(j.dataset.groupJump);return}const b=e.target.closest('[data-cover]');if(b&&window.__coverRows?.has(b.dataset.cover)){const r=window.__coverRows.get(b.dataset.cover);selectCover(r)}};
function selectVertexAt(clientX,clientY){const rect=overlay.getBoundingClientRect(),x=clientX-rect.left,y=clientY-rect.top;let best=null;for(const h of vertexHitTargets){const d=Math.hypot(x-h.x,y-h.y);if(d<=h.r&&(!best||d<best.d))best={...h,d}}if(!best)return false;selectedOrbifoldPoint=best.q;if($('sheet').classList.contains('show'))setHTML($('sheet-content'),panelHTML(panel,current));invalidate();return true}
let dragTravel=0,dragStartX=0,dragStartY=0;
overlay.addEventListener('pointerdown',e=>{drag=true;dragTravel=0;dragStartX=lastX=e.clientX;dragStartY=lastY=e.clientY;gpuFast=true;overlay.classList.add('dragging');overlay.setPointerCapture?.(e.pointerId)});overlay.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-lastX,dy=e.clientY-lastY;dragTravel+=Math.hypot(dx,dy);lastX=e.clientX;lastY=e.clientY;panPixels(dx,dy)});function stopDrag(e){if(!drag)return;const click=dragTravel<5;drag=false;overlay.classList.remove('dragging');gpuFast=false;if(click&&e?.type==='pointerup')selectVertexAt(e.clientX,e.clientY);invalidate()}overlay.addEventListener('pointerup',stopDrag);overlay.addEventListener('pointercancel',stopDrag);overlay.addEventListener('wheel',e=>{e.preventDefault();zoomBy(Math.exp(-e.deltaY*.0013))},{passive:false});
function keyName(e){const k=e.key.toLowerCase();return k==='pageup'?'q':k==='pagedown'?'e':k===','?'q':k==='.'?'e':k}
window.addEventListener('keydown',e=>{if(e.target?.tagName==='INPUT')return;const k=keyName(e);if(['arrowleft','arrowright','arrowup','arrowdown','w','a','s','d','q','e','+','-','=','_'].includes(k)){e.preventDefault();if(k==='+'||k==='=')zoomBy(1.11);else if(k==='-'||k==='_')zoomBy(1/1.11);else{keys.add(k);gpuFast=true;if(!keyRAF){lastKey=performance.now();keyRAF=requestAnimationFrame(keyLoop)}}}});window.addEventListener('keyup',e=>{keys.delete(keyName(e));if(!keys.size){gpuFast=false;invalidate()}});function keyLoop(t){const dt=Math.min(.04,(t-lastKey)/1000||.016);lastKey=t;let x=0,y=0;if(keys.has('arrowleft')||keys.has('a'))x+=1;if(keys.has('arrowright')||keys.has('d'))x-=1;if(keys.has('arrowup')||keys.has('w'))y+=1;if(keys.has('arrowdown')||keys.has('s'))y-=1;if(x||y)panPixels(x*245*dt,y*245*dt);if(keys.has('q'))auxiliaryMotion(-1,dt);if(keys.has('e'))auxiliaryMotion(1,dt);keyRAF=keys.size?requestAnimationFrame(keyLoop):0}
window.addEventListener('resize',resize);

// -----------------------------------------------------------------------------
// Startup / debug hooks
// -----------------------------------------------------------------------------
function start(){SYS=makeSystem(current);loadViewSettings();loadLanguage();syncViewSettingsUI();applyLanguage({rerender:false});regionCache=new Map();centerOnCurrentRegion();initGL();setGroupButtonLabel(current);picker();setHTML($('sheet-content'),panelHTML('group',current));hidePanel();resize();typeset();window.__ATLAS_READY__=true;window.__ATLAS_DEBUG__={setGroup,showPanel,selectCover,coverRecords,currentRegion,exactTex,rebaseCamera,foldToCoxeter,buildChamberTopology,gammaCellComplex,panelHTML,setHGPeer:(id)=>{selectedHGPeer=id;showPanel('uniformization')},pairHTML:(gid,peer)=>{const gg=byId.get(gid);return gg?pairDetail(gg,pairFor(gg,peer)):''},texErrors,explicitHGPair,noncompactData,commEdges,activateCommPeer,activateMoonshine,moonFordRecord,exactTriangleCovers,exactQuadrilateralCovers,lowIndexClassGroups,subgroupClassTree,regionVertexData,regionNeighborGenerators,tileIsoKey,regionNeighborTransforms,regionNeighborCopies,geometricCornerCount,targetRegionCorners,regionFromTriple,regionWithHalfMirror,bestHalfMirrorRegion,boundaryVertexClasses,labeledRegionVertexData,coverDetail,normalizeTexHtml,mfSeriesTex,seriesTex,moonshineSeriesTex,relationTemplates,hypergeometricV28Meta,belyiDatabaseMatches,belyiModels,belyiLookup,belyiMajorHTML,fieldGeneratorSpec,fieldGeneratorHTML,traceFieldGeneratorApprox,triangleLambdaNumeric,pairingAudit:()=>{const R=currentRegion();boundaryPairingSides(R);return R.pairingAudit},stats:()=>({group:current.id,model,webgl:gpuOK,cover:selectedCover?.key||null,compare:commCompare?{base:commCompare.base.id,a:commCompare.a.id,b:commCompare.b.id}:null,degree:commCompare?null:currentRegion().degree,boundary:commCompare?null:currentRegion().boundary.length,convexityDefect:commCompare?null:boundaryConvexityDefect(currentRegion().boundary),cameraAnti:camera.anti,centerView:cameraApply(currentRegion().center),zoom,euclidShiftX,euclidAngle,bbox:commCompare?{a:regionScreenBBox(commCompare.a.R),b:regionScreenBBox(commCompare.b.R)}:regionScreenBBox(currentRegion())}),modularPermutationCount:Object.keys(modularPerm).length}}
if(window.MathJax?.startup?.promise)MathJax.startup.promise.then(start).catch(start);else start();
})();
