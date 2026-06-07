const fs = require('fs');
const vm = require('vm');
function assert(cond,msg){ if(!cond) throw new Error(msg); }
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
let context;
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, defaultChecked:true, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, matches(){ return false; },
    querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
let clock=0;
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{ if(script.text) vm.runInContext(String(script.text), context); if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
console, performance:{ now:()=>clock++ }, setTimeout, clearTimeout,
requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js']) vm.runInContext(fs.readFileSync(f,'utf8'), context);

const L=context.__RIES_LFUNC_TEST__;
const H=context.__RIES_HARDDB_TEST__;
const Y=context.__RIES_HYPDATA_TEST__;
const P=context.__RIES_PRECISION_TEST__;
assert(L && H && Y && P, 'test hooks missing');
assert(L.lfuncDbStage({level:4})===1 && L.lfuncDbStage({level:5})===2 && L.lfuncDbStage({level:6})===3, 'L-function transform stage gating should follow levels 4/5/6');
assert(L.lfuncDbMultiplierCatalog(1).length < L.lfuncDbMultiplierCatalog(2).length && L.lfuncDbMultiplierCatalog(2).length < L.lfuncDbMultiplierCatalog(3).length, 'higher L-function transform stages should add multiplier products');

const hard3f2=H.hardDbFormulaLatex({category:'generalized hypergeometric value', cid:1, params:{a1:'1/2',a2:'2/3',a3:'3/4',b1:'4/5',b2:'5/6',z:'-1/2'}});
assert(hard3f2.includes('\\begin{array}{c}') && hard3f2.includes('\\\\[2pt]') && hard3f2.includes('\\frac{4}{5}, \\frac{5}{6}'), `3F2 lower parameters should be preserved in an array: ${hard3f2}`);
const hyp3f2=Y.hypDataMkLatex('P|0|1/2,2/3,3/4|4/5,5/6|-1/2');
assert(hyp3f2.includes('{}_{3}F_{2}') && hyp3f2.includes('\\begin{array}{c}') && hyp3f2.includes('\\frac{4}{5}, \\frac{5}{6}'), `hypdata 3F2 lower parameters should survive sanitization: ${hyp3f2}`);

function settingsForDecimal(raw, level=4){
  const parsed=P.parseDecimalComplex(raw);
  return { raw, normalizedRaw:raw, parsedComplex:parsed, complexTarget:false, target:P.rationalToNumber(parsed.re), modules:{lfunc:true}, level, limit:5,
    lfuncOptions:{rational:true, quadratic:true, log:true, specialConstants:false}, stageBudgets:{lfuncMs:8000} };
}
(async()=>{
  const entry=L.lfuncEntries().find(e=>e.weight===2 && e.which==='1' && Number(e.value)>0);
  assert(entry, 'positive L(f,1) entry missing');
  const raw=Math.log(Number(entry.value)).toPrecision(16);
  const rows=await context.lfuncRowsAsync(settingsForDecimal(raw,4),0);
  const hit=rows.find(r=>r.lfuncCategory==='transform' && /exp\(x\).*L\(f,1\)/.test(r.candidate) && r.lfuncEntryKey===entry.entryKey);
  assert(hit, 'L-function transformed exp(x) comparison should be included in global candidates; got\n'+rows.map(r=>r.candidate).join('\n'));
  assert(hit.latex.includes('\\exp(x) \\approx L(f,1)'), `transform hit should display transformed LHS in LaTeX: ${hit.latex}`);
  assert(rows.length<=5, 'L-function output should still be globally capped at five candidates');
  console.log('PASS RIES v11.8.2 L-function transformed database matching and hypergeometric LaTeX test');
})().catch(err=>{ console.error(err); process.exit(1); });
