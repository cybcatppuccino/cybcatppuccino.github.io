const fs = require('fs');
const vm = require('vm');
function assert(cond,msg){ if(!cond) throw new Error(msg); }
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
let context;
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){},
    querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{ if(script.text) vm.runInContext(String(script.text), context); if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
console, performance:{ now:(()=>{let t=0; return()=>{t+=1; return t;};})() }, setTimeout, clearTimeout,
requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js','assets/ries-harddb-v11_7_3-level4.js','assets/ries-hypdata-v11_5_2-level4.js','assets/ries-intsumdb-v11_7-level4.js']) vm.runInContext(fs.readFileSync(f,'utf8'), context);

const html = fs.readFileSync('ries.html','utf8');
assert(html.includes('RIES <em>v11.9</em>'), 'visible version should be v11.9');
assert(html.includes('ries-script.js?v=11.9'), 'script cache-buster should be v11.9');
for(const [id,val] of [['hardDb4BudgetMs','3000'],['hardDb5BudgetMs','15000'],['hypData1BudgetMs','3000'],['hypData2BudgetMs','15000'],['intsumDb1BudgetMs','3000'],['intsumDb2BudgetMs','15000']]){
  assert(new RegExp(`id="${id}"[^>]*value="${val}"`).test(html), `${id} default should be ${val}`);
}

const H=context.__RIES_HARDDB_TEST__, Y=context.__RIES_HYPDATA_TEST__, I=context.__RIES_INTSUMDB_TEST__;
assert(H && Y && I, 'database test hooks missing');
const base={level:4, target:2, complexTarget:false, modules:{hardDb:true,hypData:true,intsumDb:true}, hardDbOptions:{depth4:true, rational:true, power:true, exponential:true, logScale:true, rationalHeight:20, maxParamHeight:200}, hypDataOptions:{depth1:true,multSimple:true}, intsumDbOptions:{depth1:true,multSimple:true}, moduleLimits:{hardDb:20,hypData:20,intsumDb:20}, stageBudgets:{hardDb4Ms:0,hypData1Ms:0,intsumDb1Ms:0}};
const views=H.dbComparisonTargetViews(base).map(v=>v.id).join('|');
assert(views === 'x|exp|logabs', `target views mismatch: ${views}`);
assert(!H.dbComparisonTargetViews({...base,target:11}).some(v=>v.id==='exp'), 'exp(x) view must be disabled when x > 10');
assert(H.dbComparisonTargetViews({...base,target:0}).map(v=>v.id).join('|') === 'exp', 'x=0 should still permit exp(x) database matching');
assert(H.hardDbBudgetMs({...base, level:4, stageBudgets:{}}) === 3000, 'harddb level4 budget should be tripled');
assert(H.hardDbBudgetMs({...base, level:5, stageBudgets:{}}) === 15000, 'harddb level5 budget should be tripled');
assert(Y.hypDataStageBudgetMs({...base, stageBudgets:{}}, 1) === 3000, 'hypdata level4 budget should be tripled');
assert(Y.hypDataStageBudgetMs({...base, stageBudgets:{}}, 2) === 15000, 'hypdata level5 budget should be tripled');
assert(I.intsumDbStageBudgetMs({...base, stageBudgets:{}}, 1) === 3000, 'intsum level4 budget should be tripled');
assert(I.intsumDbStageBudgetMs({...base, stageBudgets:{}}, 2) === 15000, 'intsum level5 budget should be tripled');

function f64(b64){ const buf=Buffer.from(b64,'base64'); return new Float64Array(buf.buffer, buf.byteOffset, Math.floor(buf.byteLength/8)); }
function assertCleanLatex(row, lhs){
  assert(row && row.latex.includes(lhs), `missing ${lhs} in ${row && row.latex}`);
  assert(row.latex.includes('\\approx'), `closed form must use \\approx: ${row.latex}`);
  assert(!row.latex.includes('\\expx'), `exp lhs was mangled: ${row.latex}`);
  assert(!/\bapprox\b/.test(row.latex.replace(/\\approx/g,'')), `bare approx leaked: ${row.latex}`);
}
(async()=>{
  const hardVals=f64(context.RIES_HARDDB_V1173_CHUNKS[0].valuesB64);
  const hAbs=Math.abs(hardVals[0]);
  const hardExpRows=await H.hardDbRowsAsync({...base, target:Math.log(hAbs)});
  assert(hardExpRows.some(r=>r.candidate.includes('exp(x) ≈')), 'harddb did not return an exp(x) transformed match');
  assertCleanLatex(hardExpRows.find(r=>r.candidate.includes('exp(x) ≈')), '\\exp(x)');
  const hardLogRows=await H.hardDbRowsAsync({...base, target:Math.exp(hardVals[0])});
  assert(hardLogRows.some(r=>r.candidate.includes('log|x| ≈')), 'harddb did not return a log|x| transformed match');
  assertCleanLatex(hardLogRows.find(r=>r.candidate.includes('log|x| ≈')), '\\log\\left|x\\right|');

  const hypVals=f64(context.RIES_HYPDATA_V1152_CHUNKS[0].realValuesB64);
  const hypExpRows=await Y.hypDataRowsAsync({...base, target:Math.log(Math.abs(hypVals[0]))});
  assert(hypExpRows.some(r=>r.candidate.includes('exp(x) ≈')), 'hypergeom database did not return an exp(x) transformed match');
  assertCleanLatex(hypExpRows.find(r=>r.candidate.includes('exp(x) ≈')), '\\exp(x)');
  const hypLogRows=await Y.hypDataRowsAsync({...base, target:Math.exp(hypVals[0])});
  assert(hypLogRows.some(r=>r.candidate.includes('log|x| ≈')), 'hypergeom database did not return a log|x| transformed match');
  assertCleanLatex(hypLogRows.find(r=>r.candidate.includes('log|x| ≈')), '\\log\\left|x\\right|');

  const intVals=f64(context.RIES_INTSUMDB_V117_CHUNKS[0].valuesB64);
  const intExpRows=await I.intsumDbRowsAsync({...base, target:Math.log(Math.abs(intVals[0]))});
  assert(intExpRows.some(r=>r.candidate.includes('exp(x) ≈')), 'integral/sum database did not return an exp(x) transformed match');
  assertCleanLatex(intExpRows.find(r=>r.candidate.includes('exp(x) ≈')), '\\exp(x)');
  const intLogRows=await I.intsumDbRowsAsync({...base, target:Math.exp(intVals[0])});
  assert(intLogRows.some(r=>r.candidate.includes('log|x| ≈')), 'integral/sum database did not return a log|x| transformed match');
  assertCleanLatex(intLogRows.find(r=>r.candidate.includes('log|x| ≈')), '\\log\\left|x\\right|');
  console.log('PASS RIES v11.9 database transformed-target matching test');
})().catch(err=>{ console.error(err); process.exit(1); });
