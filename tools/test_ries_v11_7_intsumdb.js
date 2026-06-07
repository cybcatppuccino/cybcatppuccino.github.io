const fs = require('fs');
const vm = require('vm');

function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
let context;
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){},
    querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{
  const src=String(script.src || script.getAttribute?.('src') || '');
  for(const lvl of [4,5,6]) if(src.includes(`ries-intsumdb-v11_7-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-intsumdb-v11_7-level${lvl}.js`,'utf8'), context);
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

const T=context.__RIES_INTSUMDB_TEST__;
if(!T || typeof T.ensureIntsumDbLoaded!=='function') throw new Error('intsumdb hooks missing');
function assert(cond,msg){ if(!cond) throw new Error(msg); }
function firstRealTarget(payload){ const values=Buffer.from(payload.valuesB64,'base64'); for(let i=0;i<values.length/8;i++){ const v=values.readDoubleLE(i*8); if(Number.isFinite(v) && Math.abs(v)>1e-12) return v; } throw new Error('no real target'); }
function settingsFor(target, level){ return {target, targetString:String(target), complexTarget:false, level, sortMode:'confidence', modules:{intsumDb:true}, intsumDbOptions:{depth1:true,depth2:true,depth3:true,multSimple:true,multGamma:true,multDeep:true}, moduleLimits:{intsumDb:5}, stageBudgets:{intsumDb1Ms:1000,intsumDb2Ms:5000,intsumDb3Ms:50000}}; }

(async()=>{
  assert(T.intsumDbMaxStage({level:3,intsumDbOptions:{depth1:true,depth2:true,depth3:true}})===0, 'level 3 should not run intsumdb');
  assert(T.intsumDbMaxStage({level:4,intsumDbOptions:{depth1:true,depth2:true,depth3:true}})===1, 'level 4 should map to stage 1');
  assert(T.intsumDbMaxStage({level:5,intsumDbOptions:{depth1:true,depth2:true,depth3:true}})===2, 'level 5 should map to stage 2');
  assert(T.intsumDbMaxStage({level:6,intsumDbOptions:{depth1:true,depth2:true,depth3:true}})===3, 'level 6 should map to stage 3');
  assert(!T.isIntsumDbReady(1), 'intsumdb should be lazy before stage load');

  await T.ensureIntsumDbLoaded({stage:1,label:'integral/sum database',phase:'test'});
  assert(T.isIntsumDbReady(1), 'stage 1 should be ready after load');
  assert(context.RIES_INTSUMDB_V117_CHUNKS[0].rows === 6789, 'stage 1 row count mismatch');
  assert(context.RIES_INTSUMDB_V117_CHUNKS[0].multiplierRows === 1200, 'stage 1 multiplier count mismatch');

  const target=firstRealTarget(context.RIES_INTSUMDB_V117_CHUNKS[0]);
  const rows=await T.intsumDbRowsAsync(settingsFor(target, 4));
  assert(rows.length > 0, 'expected at least one intsumdb row');
  assert(String(rows[0].candidate).startsWith('integral/sum database:'), 'candidate should identify intsumdb');
  assert(String(rows[0].latex).includes('\\approx'), 'row LaTeX should contain escaped \\approx');
  assert(String(rows[0].valueHtml).includes('\\(') && String(rows[0].valueHtml).includes('\\)'), 'valueHtml should keep LaTeX delimiters');
  assert(!String(rows[0].valueHtml).includes('xapprox'), 'valueHtml should not drop the backslash in approximation text');
  assert(rows[0].constantDbSource === 'intsumdb-v11.7.2', 'source marker mismatch');
  assert(T.resultRowCategory(rows[0]) === 'intsumdb', 'intsum category should be visible to sorter/category logic');

  const composed=T.intsumDbMulLatex('\\frac{2}{3}', '\\int_0^1 x\\,dx');
  assert(composed.includes('\\frac{2}{3}') && composed.includes('\\int_0^1 x\\,dx') && composed.includes('\\,'), 'multiplier LaTeX should preserve escapes');

  await T.ensureIntsumDbLoaded({stage:2,label:'integral/sum database',phase:'test'});
  assert(T.isIntsumDbReady(2), 'stage 2 should be ready after cumulative load');
  assert(context.RIES_INTSUMDB_V117_CHUNKS[1].rows === 29654, 'stage 2 additional row count mismatch');
  assert(context.RIES_INTSUMDB_V117_CHUNKS[1].multiplierRows === 5300, 'stage 2 multiplier count mismatch');

  await T.ensureIntsumDbLoaded({stage:3,label:'integral/sum database',phase:'test'});
  assert(T.isIntsumDbReady(3), 'stage 3 should be ready after cumulative load');
  assert(context.RIES_INTSUMDB_V117_CHUNKS[2].rows === 0, 'stage 3 should not add rows');
  assert(context.RIES_INTSUMDB_V117_CHUNKS[2].multiplierRows === 9500, 'stage 3 multiplier count mismatch');

  console.log('PASS RIES v11.7.2 integral/sum runtime smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
