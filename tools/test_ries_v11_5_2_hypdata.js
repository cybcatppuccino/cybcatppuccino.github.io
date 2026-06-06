const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v11.5.2</em>')) throw new Error('ries.html visible version should be v11.5.2');
if(!html.includes('ries-script.js?v=11.5.2')) throw new Error('ries-script cache tag should be v11.5.2');
if(/<script[^>]+ries-hypdata-v11_5_2-level[456]\.js/i.test(html)) throw new Error('hypdata packages must be lazy-loaded, not part of initial HTML');
for(const lvl of [4,5,6]) if(!fs.existsSync(`assets/ries-hypdata-v11_5_2-level${lvl}.js`)) throw new Error(`hypdata level ${lvl} chunk missing`);
if(!fs.existsSync('assets/ries-hypdata-v11_5_2-stats.json')) throw new Error('hypdata v11.5.2 stats missing');
if(!fs.existsSync('changelog/RIES_v11.5.2_CHANGELOG.md')) throw new Error('v11.5.2 changelog missing');

const stats = JSON.parse(fs.readFileSync('assets/ries-hypdata-v11_5_2-stats.json','utf8'));
if(stats.version !== '11.5.2') throw new Error('stats version mismatch');
if(stats.rows !== 109738 || stats.realRows !== 36874) throw new Error('unexpected row counts');
if(stats.tierCounts['1'] !== 3159 || stats.tierCounts['2'] !== 36407 || stats.tierCounts['3'] !== 70172) throw new Error('unexpected tier counts');
if(stats.multiplierStageCounts['1'] !== 1200 || stats.multiplierStageCounts['2'] !== 5300 || stats.multiplierStageCounts['3'] !== 9500) throw new Error('unexpected multiplier stage counts');
if(!(stats.cumulativeAssetBytes.level4 < 600000 && stats.cumulativeAssetBytes.level5 < stats.oldSingleAssetBytes && stats.cumulativeAssetBytes.level6 < stats.oldSingleAssetBytes + 500000)) throw new Error('chunking did not reduce level4/5 load sizes as expected');

function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
const context = {
  window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{
    const src=String(script.src || script.getAttribute?.('src') || '');
    for(const lvl of [4,5,6]) if(src.includes(`ries-hypdata-v11_5_2-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-hypdata-v11_5_2-level${lvl}.js`,'utf8'), context);
    if(src.includes('ries-harddb-v11_4_1-filtered.js')) vm.runInContext(fs.readFileSync('assets/ries-harddb-v11_4_1-filtered.js','utf8'), context);
    if(script.text) vm.runInContext(String(script.text), context);
    if(typeof script.onload==='function') script.onload(); return script; } },
    getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
  console, performance:{ now:(()=>{let t=0; return()=>{t+=1; return t;};})() }, setTimeout, clearTimeout,
  requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
  URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
  atob:(s)=>Buffer.from(s,'base64').toString('binary')
};
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);

const T=context.__RIES_HYPDATA_TEST__;
if(!T || typeof T.ensureHypDataLoaded!=='function') throw new Error('hypdata hooks missing');
if(T.hypDataMaxStage({level:3}) !== 0) throw new Error('level 3 should not run hypdata');
if(T.hypDataMaxStage({level:4}) !== 1) throw new Error('level 4 should map to pFq stage 1');
if(T.hypDataMaxStage({level:5}) !== 2) throw new Error('level 5 should map to pFq stage 2');
if(T.hypDataMaxStage({level:6}) !== 3) throw new Error('level 6 should map to pFq stage 3');
if(T.isHypDataReady(1)) throw new Error('hypdata should start unloaded');
if(!/_2F1/.test(T.hypDataMkText('P|0|1/2,1/2|1|1/2'))) throw new Error('MK text renderer should render pFq shape');
const mkLatex = T.hypDataMkLatex('P|0|1/2,1/2|1|1/2');
if(!mkLatex.includes('F_{1}') || !mkLatex.includes('_{2}')) throw new Error('MK latex renderer should render pFq notation');

function firstRealTarget(payload){
  const values=Buffer.from(payload.realValuesB64,'base64');
  for(let i=0;i<values.length/8;i++){
    const v=values.readDoubleLE(i*8);
    if(Number.isFinite(v) && Math.abs(v)>1e-12) return v;
  }
  throw new Error('no real target in chunk');
}

(async()=>{
  let loaded=await T.ensureHypDataLoaded({stage:1,label:'hypergeometric pFq database',phase:'test'});
  if(!loaded || !T.isHypDataReady(1)) throw new Error('level4 chunk did not load');
  if(T.hypDataLoadedChunks(3).length !== 1) throw new Error('stage1 load should only load one chunk');
  const payload=context.RIES_HYPDATA_V1152_CHUNKS[0];
  if(payload.version!=='11.5.2' || payload.rows!==3159 || payload.multiplierRows!==1200) throw new Error('level4 chunk payload mismatch');
  const target=firstRealTarget(payload);
  const hits=T.hypDataSearch({target,complexTarget:false,level:4,sortMode:'confidence'});
  if(!Array.isArray(hits) || hits.length<1 || hits.length>5) throw new Error('level4 search should return 1-5 hits for known target');
  const rows=await T.hypDataRowsAsync({target,complexTarget:false,level:4,sortMode:'confidence'});
  if(!Array.isArray(rows) || rows.length<1 || rows.length>5 || !/^hypergeometric database:/i.test(rows[0].candidate)) throw new Error('formatted hypdata row mismatch');

  loaded=await T.ensureHypDataLoaded({stage:2,label:'hypergeometric pFq database',phase:'test'});
  if(!loaded || !T.isHypDataReady(2)) throw new Error('level5 cumulative chunks did not load');
  if(T.hypDataLoadedChunks(3).length !== 2) throw new Error('stage2 load should load level4+level5 chunks, not full level6');
  const hits2=T.hypDataSearch({target,complexTarget:false,level:5,sortMode:'confidence'});
  if(!Array.isArray(hits2) || hits2.length<1) throw new Error('level5 cumulative search should still include level4 H rows');
  console.log('PASS RIES v11.5.2 incremental hypdata chunks and cumulative staged search smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
