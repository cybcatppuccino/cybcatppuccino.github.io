const fs = require('fs');
const vm = require('vm');

for(const f of ['assets/ries-harddb-v11_6-level4.js','assets/ries-harddb-v11_6-level5.js','assets/ries-harddb-v11_6-stats.json']){
  if(!fs.existsSync(f)) throw new Error(`${f} missing`);
}
const stats=JSON.parse(fs.readFileSync('assets/ries-harddb-v11_6-stats.json','utf8'));
if(stats.level4Rows!==15986 || stats.level5AdditionalRows!==63946) throw new Error('unexpected harddb split counts');
if(stats.level4CategoriesCovered!==stats.totalCategories) throw new Error('level4 harddb should cover all categories');

function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
const context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{
  const src=String(script.src || script.getAttribute?.('src') || '');
  for(const lvl of [4,5]) if(src.includes(`ries-harddb-v11_6-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-harddb-v11_6-level${lvl}.js`,'utf8'), context);
  if(script.text) vm.runInContext(String(script.text), context);
  if(typeof script.onload==='function') script.onload(); return script;
} }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
console, performance:{ now:(()=>{let t=0; return()=>{t+=1; return t;};})() }, setTimeout, clearTimeout,
requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);
const T=context.__RIES_HARDDB_TEST__;
if(!T || typeof T.ensureHardDbLoaded!=='function') throw new Error('harddb test hooks missing');
const base={target:1.2345, complexTarget:false, modules:{hardDb:true}, hardDbOptions:{depth4:true, depth5:true, rational:true, power:true, exponential:true, logScale:true, rationalHeight:10, maxParamHeight:15}, moduleLimits:{hardDb:5}, stageBudgets:{hardDbMs:1000}};
if(T.hardDbMaxStage({...base, level:3})!==0) throw new Error('depth 3 should not run harddb');
if(T.hardDbMaxStage({...base, level:4})!==1) throw new Error('depth 4 should map to harddb stage 1');
if(T.hardDbMaxStage({...base, level:5})!==2) throw new Error('depth 5 should map to harddb stage 2');
if(T.isHardDbReady(1)) throw new Error('harddb should start unloaded');
(async()=>{
  let loaded=await T.ensureHardDbLoaded({settings:{...base, level:4}, stage:1, label:'harddb', phase:'test'});
  if(!loaded || !T.isHardDbReady(1)) throw new Error('harddb level4 did not load');
  if(T.hardDbLoadedChunks(2).length!==1) throw new Error('depth4 harddb should load only first chunk');
  loaded=await T.ensureHardDbLoaded({settings:{...base, level:5}, stage:2, label:'harddb', phase:'test'});
  if(!loaded || !T.isHardDbReady(2)) throw new Error('harddb level5 did not load cumulatively');
  if(T.hardDbLoadedChunks(2).length!==2) throw new Error('depth5 harddb should load both chunks');
  const rows=await T.hardDbRowsAsync({...base, level:4, target:0.5});
  if(!Array.isArray(rows) || rows.length>5) throw new Error('harddb rows should respect candidate limit');
  console.log('PASS RIES v11.6 harddb split lazy-load smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
