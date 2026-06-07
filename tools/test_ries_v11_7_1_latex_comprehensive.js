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
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{
  const src=String(script.src || script.getAttribute?.('src') || '');
  for(const lvl of [4,5,6]){
    if(src.includes(`ries-intsumdb-v11_7-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-intsumdb-v11_7-level${lvl}.js`,'utf8'), context);
    if(src.includes(`ries-hypdata-v11_5_2-level${lvl}.js`)) vm.runInContext(fs.readFileSync(`assets/ries-hypdata-v11_5_2-level${lvl}.js`,'utf8'), context);
  }
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

const E=context.__RIES_EQUATION_TEST__;
const H=context.__RIES_HARDDB_TEST__;
const Y=context.__RIES_HYPDATA_TEST__;
const I=context.__RIES_INTSUMDB_TEST__;
assert(E && H && Y && I, 'LaTeX test hooks missing');

const expr = E.exprToLatex('sqrt(2)+x^0+x^1+(x+1)^0+x^(1/2)+sin(pi/2)');
assert(expr.includes('\\sqrt{2}') && expr.includes('\\sqrt{x}'), `RIES sqrt/half-power display failed: ${expr}`);
assert(!expr.includes('operatorname{sqrt}') && !/\^\{0\}|\^\{1\}|\^0(?!\d)|\^1(?!\d)/.test(expr.replace(/_0\^1/g,'')), `RIES neutral powers leaked: ${expr}`);

const normalized = E.sanitizeLatexForDisplay('1+1x-1\\sin(x)+x^{1-1}+\\log(1-\\sin(x))^{0}\\log(1+\\cos(x))^{1}+e^{--\\frac{1}{2}}');
assert(!/[+\-(]1(?:x|\\sin|\\cos|\\log)/.test(normalized), `unit coefficients leaked: ${normalized}`);
assert(!/\^\{0\}|\^\{1\}/.test(normalized), `neutral brace powers leaked: ${normalized}`);
assert(!/[+-]{2}|\+\s*-|-\s*\+/.test(normalized), `sign run leaked: ${normalized}`);
assert(normalized.includes('\\log(1+\\cos(x))') && normalized.includes('e^{\\frac{1}{2}}'), `normalization changed structure unexpectedly: ${normalized}`);

const beta = H.hardDbFormulaLatex({category:'Euler beta integral', params:{a:'1',b:'1'}, cid:'beta_unit'});
assert(beta === '\\int_0^1 \\,dx', `harddb neutral powers not removed: ${beta}`);
const gamma = H.hardDbFormulaLatex({category:'gamma log-laplace integral', params:{a:'1',q:'-1',logPower:'0'}, cid:'gamma_unit'});
assert(gamma.includes('e^{x}') && !gamma.includes('e^{1x}') && !gamma.includes('--'), `harddb signed exponent/unit coefficient failed: ${gamma}`);

const hyp = Y.hypDataMkLatex('P|1/2|1/2,-1/2|1|-1/2');
assert(hyp.startsWith('\\frac{1}{2}\\,'), `hypergeom prefactor should be scalar, not P-subscript: ${hyp}`);
assert(hyp.includes('{}_{2}F_{1}') && hyp.includes('\\frac{1}{2}') && hyp.includes('-\\frac{1}{2}'), `hypergeom structure/fractions broken: ${hyp}`);
assert(!hyp.includes('P_{') && !/[\u0008\u0009\u000c\u000d]/.test(hyp), `hypergeom LaTeX escape failure: ${hyp}`);

const imul = I.intsumDbMulLatex('-1', 'x+1');
assert(imul === '-\\left(x+1\\right)', `intsum -1 multiplier grouping failed: ${imul}`);
const fracmul = I.intsumDbMulLatex('\\frac{2}{3}', '\\int_0^1 x\\,dx');
assert(fracmul === '\\frac{2}{3}\\,\\int_0^1 x\\,dx', `intsum fractional multiplier failed: ${fracmul}`);

const assetCtx={window:{}, console}; assetCtx.window=assetCtx; vm.createContext(assetCtx);
for(const lvl of [4,5,6]) vm.runInContext(fs.readFileSync(`assets/ries-intsumdb-v11_7-level${lvl}.js`, 'utf8'), assetCtx);
let checked=0;
for(const ch of assetCtx.RIES_INTSUMDB_V117_CHUNKS){
  if(!ch || !Number(ch.rows)) continue;
  const latexes=String(ch.latexBlob||'').split('\n');
  for(const latex of latexes){
    checked++;
    assert(!/[\u0008\u0009\u000c\u000d]/.test(latex), `asset control escape: ${latex}`);
    assert(!/[+-]{2}|\+\s*-|-\s*\+/.test(latex), `asset sign run: ${latex}`);
    assert(!/\^\{0\}|\^\{1\}/.test(latex), `asset neutral brace power: ${latex}`);
    assert(!/[+\-(]1(?:x|t|n|\\sin|\\cos|\\log)/.test(latex), `asset unit coefficient: ${latex}`);
    assert(!/\\sin\\cos|\\cos\\sin/.test(latex), `asset trig function lost argument: ${latex}`);
    assert(!/operatorname\{sqrt\}/.test(latex), `asset sqrt should use sqrt command: ${latex}`);
  }
}
assert(checked === 36443, `expected 36443 intsum latex rows, saw ${checked}`);

const html=fs.readFileSync('ries.html','utf8');
assert(html.includes('RIES <em>v11.7.3</em>') && html.includes('ries-script.js?v=11.7.3'), 'v11.7.3 page/cache-buster missing');
console.log('PASS RIES v11.7.3 comprehensive LaTeX normalization regression test');
