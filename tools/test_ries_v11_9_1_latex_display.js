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
context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:(script)=>context.document.head.appendChild(script) }, head:{ appendChild:(script)=>{ if(script.text) vm.runInContext(String(script.text), context); if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) },
console, performance:{ now:(()=>{let t=0; return()=>++t;})() }, setTimeout, clearTimeout,
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
assert(E && H && Y && I, 'v11.9.1 LaTeX display hooks missing');
const L=context.__RIES_LFUNC_TEST__;
const C=context.__RIES_CONSTDB_TEST__;
assert(L && C, 'v11.9.1 L-function/constant DB test hooks missing');

const hyp=Y.hypDataMkLatex('P|0|1/6,1/2,5/6|1,1|1');
assert(hyp.includes('1, 1\\end{array}'), `hypergeom lower parameter 1 should not be swallowed: ${hyp}`);
const hard=H.hardDbFormulaLatex({category:'generalized hypergeometric value', cid:'ones', params:{a1:'1/6',a2:'1/2',a3:'5/6',b1:'1',b2:'1',z:'1'}});
assert(hard.includes('1, 1\\end{array}'), `harddb 3F2 lower parameter 1 should not be swallowed: ${hard}`);

assert(E.exprToLatex('1*x')==='x', '1*x should display as x');
assert(E.exprToLatex('x*1')==='x', 'x*1 should display as x');
assert(E.exprToLatex('5*(1/5)*sqrt(2)')==='\\sqrt{2}', `scalar product should simplify: ${E.exprToLatex('5*(1/5)*sqrt(2)')}`);
const intmul=I.intsumDbMulLatex('5','\\frac{1}{5}\\,\\int_0^1 x\\,dx');
assert(intmul==='\\int_0^1 x\\,dx', `intsum scalar 5*1/5 should simplify: ${intmul}`);

const neg=E.sanitizeLatexForDisplay('\\left(-x-1\\right)^{2}+\\left(-x+1\\right)^{4}');
assert(neg.includes('x + 1') && neg.includes('x - 1') && !neg.includes('(-'), `negative even-power bases should be normalized correctly: ${neg}`);
assert(E.exprToLatex('root(2+3,4)')==='\\sqrt[4]{2+3}', `root(a,b) LaTeX is wrong: ${E.exprToLatex('root(2+3,4)')}`);
const logBase=E.exprToLatex('log_2(8)+log_pi(2)');
assert(logBase.includes('\\log_{2}') && logBase.includes('\\log_{\\pi}'), `log_base display is wrong: ${logBase}`);
const sqrtNested=E.sanitizeLatexForDisplay('√(1/(2+3))');
assert(sqrtNested==='\\sqrt{1/(2+3)}', `nested radical range should be preserved: ${sqrtNested}`);

const long=E.latexBreakLongFormulaForDisplay('x \\approx a+b+c+d+e+f+g+h+i+j+k+l+m+n+o+p+q', 20);
assert(long.startsWith('\\begin{aligned}') && long.includes('\\\\&\\quad + h') && long.endsWith('\\end{aligned}'), `long formula should be aligned and line-broken: ${long}`);
const alg=E.algebraicRowFromCoeff([1n,-2n,1n], 'algebraic relation', {re:1, im:0});
assert(alg && alg.latex && alg.latex.includes('x^{2}') && alg.latex.includes('= 0'), `algebraic relation should provide LaTeX: ${alg && alg.latex}`);


assert(E.sanitizeLatexForDisplay('1\\,\\log 2')==='\\log 2', 'neutral 1 before log should be removed');
assert(E.sanitizeLatexForDisplay('1\\,c')==='c', 'neutral 1 before generic constant should be removed');
assert(E.sanitizeLatexForDisplay('x \\approx 1\\,\\log 2 + 1\\,\\pi')==='x \\approx \\log 2 + \\pi', 'neutral 1 should simplify in sums');
assert(E.sanitizeLatexForDisplay('5\\,\\frac{1}{5}\\,S')==='S', 'scalar product 5*1/5 should disappear completely');
assert(E.sanitizeLatexForDisplay('\\frac{1}{2}\\,\\frac{2}{\\pi}')==='\\frac{1}{\\pi}', 'adjacent constant fractions should simplify over pi');
const powFrac=E.sanitizeLatexForDisplay('x^{\\frac{1}{2}-1}+y^{1/2-1}+z^{2-1}');
assert(powFrac==='x^{-\\frac{1}{2}}+y^{-\\frac{1}{2}}+z', `fractional exponent arithmetic should simplify: ${powFrac}`);
const ltex=L.lfuncFormulaLatex('1/2·2/π·L(f,1/2)','x');
assert(ltex.includes('\\frac{1}{\\pi}\\cdot L(f,\\tfrac{1}{2})'), `Lfunc constants should use frac and simplify rational*pi factors: ${ltex}`);
const alphaEq=C.constDbPolyToLatex([1,-2,1],'\\alpha');
assert(alphaEq.includes('\\alpha^{2}') && alphaEq.includes('= 0'), `constant DB alpha equation should be LaTeX with equality: ${alphaEq}`);

const html=fs.readFileSync('ries.html','utf8');
assert(html.includes('RIES <em>v11.9.1</em>') && html.includes('ries-script.js?v=11.9.1'), 'v11.9.1 page/cache-buster missing');
assert(/\.latex-render[^}]*overflow-x:auto/.test(html) && /result-meta-line[^}]*overflow:hidden[^}]*overflow:visible/.test(html), 'formula overflow containment CSS missing');
console.log('PASS RIES v11.9.1 LaTeX display and overflow regression test');
