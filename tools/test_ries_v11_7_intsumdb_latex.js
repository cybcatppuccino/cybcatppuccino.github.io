const fs = require('fs');
const vm = require('vm');

function assert(cond,msg){ if(!cond) throw new Error(msg); }
const assetCtx={window:{}, console}; assetCtx.window=assetCtx; vm.createContext(assetCtx);
for(const lvl of [4,5,6]) vm.runInContext(fs.readFileSync(`assets/ries-intsumdb-v11_7-level${lvl}.js`, 'utf8'), assetCtx);
const chunks=assetCtx.RIES_INTSUMDB_V117_CHUNKS;
assert(Array.isArray(chunks) && chunks.length >= 3, 'intsum asset chunks missing');

const expected={
  FINITE_EXP_POLY:['\\int','\\,dx'],
  RATIONAL_LOG_BETA:['\\int','\\frac','\\log'],
  BETA_LOG_PLUS_MINUS:['\\int','\\log'],
  LAPLACE_GAUSS_LOG:['\\int','\\infty'],
  RATIONAL_TAIL_SUM:['\\sum','\\infty','\\frac'],
  HYPERGEOM_EULER_INTEGRAL:['\\int'],
  TRIG_BETA_LOG:['\\int','\\sin','\\cos','\\log'],
  TRIG_RATIONAL_FOURIER:['\\int','\\cos','\\frac'],
  BINOMIAL_INVBINOM_SUM:['\\sum','\\binom'],
  POLYLOG_LERCH_SUM:['\\sum','\\infty']
};
const samples={};
let checkedRows=0;
for(const ch of chunks){
  if(!ch || !Number(ch.rows)) continue;
  const fams=String(ch.familyBlob||'').split('\n');
  const subs=String(ch.subBlob||'').split('\n');
  const latexes=String(ch.latexBlob||'').split('\n');
  assert(latexes.length === Number(ch.rows), `latex blob row count mismatch for stage ${ch.stage}`);
  for(let i=0;i<latexes.length;i++){
    const latex=latexes[i];
    checkedRows++;
    assert(latex && latex.includes('\\'), `row ${i} in stage ${ch.stage} has no LaTeX command/backslash`);
    assert(!/[\u0008\u0009\u000c\u000d]/.test(latex), `row ${i} in stage ${ch.stage} contains decoded control characters from bad JS escapes`);
    assert(!latex.includes('undefined') && !latex.includes('[object Object]'), `row ${i} in stage ${ch.stage} contains a formatting placeholder`);
    assert(!/[+-]{2}/.test(latex), `row ${i} in stage ${ch.stage} contains an unsimplified sign run: ${latex}`);
    const f=fams[i];
    if(f && !samples[f]) samples[f]={sub:subs[i], latex};
  }
}
assert(checkedRows === 36443, `expected 36443 latex rows, saw ${checkedRows}`);
for(const [family, patterns] of Object.entries(expected)){
  assert(samples[family], `missing latex sample for ${family}`);
  for(const pat of patterns) assert(samples[family].latex.includes(pat), `${family} sample missing ${pat}: ${samples[family].latex}`);
}

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
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js']) vm.runInContext(fs.readFileSync(f,'utf8'), context);
const T=context.__RIES_INTSUMDB_TEST__;
assert(T && typeof T.intsumDbMulLatex === 'function', 'intsumdb LaTeX hook missing');
const composed=T.intsumDbMulLatex('\\frac{3\\sqrt{2}}{5\\pi}', samples.HYPERGEOM_EULER_INTEGRAL.latex);
for(const pat of ['\\frac','\\sqrt','\\pi','\\int']) assert(composed.includes(pat), `composed latex lost ${pat}: ${composed}`);
assert(!/[\u0008\u0009\u000c\u000d]/.test(composed), 'composed LaTeX contains decoded control characters');

console.log('PASS RIES v11.7.2 integral/sum LaTeX coverage test');
