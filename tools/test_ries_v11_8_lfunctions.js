const fs = require('fs');
const vm = require('vm');
function assert(cond,msg){ if(!cond) throw new Error(msg); }
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
let context;
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, matches(){ return false; },
    querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{ if(script.text) vm.runInContext(String(script.text), context); if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
console, performance:{ now:()=>0 }, setTimeout, clearTimeout,
requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js','assets/newforms.js']) vm.runInContext(fs.readFileSync(f,'utf8'), context);

assert(Array.isArray(context.RIES_LFUNCTIONS_L1) && context.RIES_LFUNCTIONS_L1.length === 358, 'weight 1 L-function rows missing');
assert(Array.isArray(context.RIES_LFUNCTIONS_L3) && context.RIES_LFUNCTIONS_L3.length === 127, 'weight 3 L-function rows missing');
assert(Array.isArray(context.NEWFORMS) && context.NEWFORMS.some(f=>f.weight===1) && context.NEWFORMS.some(f=>f.weight===3), 'homepage newform data missing new weights');
const entries=context.__RIES_LFUNC_TEST__.lfuncEntries();
assert(entries.some(e=>e.weight===1 && e.which==='1/2' && e.label==='L(f,1/2)'), 'RIES entries missing L(f,1/2)');
assert(entries.some(e=>e.weight===3 && e.which==='3/2' && e.label==='L(f,3/2)'), 'RIES entries missing L(f,3/2)');
assert(entries.some(e=>e.weight===2 && e.which==='1'), 'existing weight 2 entries missing');
assert(entries.some(e=>e.weight===4 && e.which==='2'), 'existing weight 4 entries missing');
assert(context.__RIES_PRECISION_TEST__.riesLevelDefaultModuleBudgetMs(4) === 8000, 'default stage budget should be 8s at depth 4');
assert(context.lfuncFormulaLatex('L(f,1/2)').includes('\\tfrac{1}{2}'), 'L(f,1/2) LaTeX should use a fraction');
assert(context.lfuncFormulaLatex('L(f,3/2)').includes('\\tfrac{3}{2}'), 'L(f,3/2) LaTeX should use a fraction');

function settingsForDecimal(raw){
  const P=context.__RIES_PRECISION_TEST__;
  const parsed=P.parseDecimalComplex(raw);
  return { raw, normalizedRaw:raw, parsedComplex:parsed, complexTarget:false, target:P.rationalToNumber(parsed.re), modules:{lfunc:true}, level:4, limit:10,
    lfuncOptions:{rational:true, quadratic:true, log:true, specialConstants:false}, stageBudgets:{lfuncMs:8000} };
}
async function assertDirect(value, label, keyPart){
  const rows=await context.lfuncRowsAsync(settingsForDecimal(value), 0);
  const hit=rows.find(r=>String(r.candidate).includes(`x ≈ ${label}`) && String(r.lfuncEntryKey).includes(keyPart));
  assert(hit, `missing direct RIES hit for ${label} / ${keyPart}; got\n`+rows.map(r=>r.candidate+' :: '+r.lfuncEntryKey).join('\n'));
  assert(hit.latex && hit.latex.includes('x \\approx'), 'hit should carry display LaTeX');
}
(async()=>{
  await assertDirect(context.RIES_LFUNCTIONS_L1[0][3], 'L(f,1/2)', ':L1/2');
  await assertDirect(context.RIES_LFUNCTIONS_L1[0][4], 'L(f,1)', ':L1');
  await assertDirect(context.RIES_LFUNCTIONS_L3[0][3], 'L(f,1)', ':L1');
  await assertDirect(context.RIES_LFUNCTIONS_L3[0][4], 'L(f,3/2)', ':L3/2');
  console.log('PASS RIES v11.9.1 L-function weight 1/3 integration smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
