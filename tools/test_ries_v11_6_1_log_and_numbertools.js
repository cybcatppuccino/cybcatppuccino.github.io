const fs = require('fs');
const vm = require('vm');
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
const elementStore = new Map();
function fakeElement(tag='div', id=''){
  return { id, tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; }, appendChild(child){ return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, matches(){return false;}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
function getEl(id){ if(!elementStore.has(id)) elementStore.set(id, fakeElement('div',id)); return elementStore.get(id); }
const defaults={target:'sqrt(2)', level:'4', limit:'5', maxAbs:'1e9', maxRelError:'Infinity', digits:'0123456789', restrictMode:'none', logHeight:'400', hardDbRationalHeight:'10', hardDbMaxParamHeight:'15', hardDb4BudgetMs:'1000', hardDb5BudgetMs:'1000', hypData1BudgetMs:'1000', hypData2BudgetMs:'5000', hypData3BudgetMs:'50000', riesBudgetMs:'5000', logBudgetMs:'5000', mobiusBudgetMs:'5000', lfuncBudgetMs:'5000', algBudgetMs:'3600', linearComboBudgetMs:'3000', integerFactorBudgetMs:'0'};
for(const [k,v] of Object.entries(defaults)) getEl(k).value=v;
for(const id of ['moduleRiesEq','moduleAlgebraic','moduleLog','moduleLinearCombo','moduleMobius','moduleConstantDb','moduleHardDb','moduleHypData','moduleLfunc','moduleInteger','hardDbDepth4','hardDbDepth5','hypDepth1','hypDepth2','hypDepth3','logTargetLogAbs','logTargetRaw','logTargetLogLogAbs','integerFactor','integerDb','integerShortform']) getEl(id).checked=true;
const checkedSyms=['p','e','f','n','r','s','q','l','E','S','C','T','+','-','*','/','^','v','L'];
const context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:()=>{} }, head:{ appendChild:(script)=>{ if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:getEl, querySelectorAll:(sel)=> sel==='[data-sym]:checked' ? checkedSyms.map(sym=>({dataset:{sym},checked:true})) : [], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) }, console, performance:{now:()=>0}, setTimeout, clearTimeout, requestAnimationFrame:()=>0, cancelAnimationFrame:()=>{}, URL:{createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{}}, Blob:function(parts){this.parts=parts;}, fetch:undefined, atob:(s)=>Buffer.from(s,'base64').toString('binary'), Buffer };
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);
const LT=context.__RIES_LOG_TEST__;
const loglog2=LT.logConstants.find(c=>c.id==='loglog2');
const loglog3=LT.logConstants.find(c=>c.id==='loglog3');
const rel={coeff:[1n,-1n,-1n], rhs:0, err:0, height:1n};
const tex=LT.linearCombinationLatex(rel,[loglog2,loglog3]);
if(!tex.includes('\\log\\!\\left(\\log 2\\right)') || !tex.includes('\\log\\!\\left(\\log 3\\right)')) throw new Error('log(log ·) constants should remain log(log ·) in linear-combination LaTeX: '+tex);
if(/(?:^|[^\\])\\log 2 \+ \\log 3/.test(tex)) throw new Error('linear-combination LaTeX collapsed log(log) constants: '+tex);
const PT=context.__RIES_PRECISION_TEST__;
let s1={raw:'1.2345', _hpEval:{error:false,z:{re:new context.Decimal('1.2345'), im:new context.Decimal(0), bi:null}}};
if(PT.numberToolsShouldAppear(s1)) throw new Error('plain decimal input should hide number tools');
let s2={raw:'12345'};
if(!PT.numberToolsShouldAppear(s2)) throw new Error('integer input should show number tools');
context.Decimal.set({precision:220});
let s3={raw:'sqrt(2)', _hpEval:{error:false,z:{re:new context.Decimal(2).sqrt(), im:new context.Decimal(0), bi:null}}};
if(!PT.numberToolsShouldAppear(s3)) throw new Error('computed expression should show number tools');
const dec=PT.decimalToBaseString(new context.Decimal(2).sqrt(),10,120);
const frac=(dec.split('.')[1]||'').replace(/…$/,'');
if(frac.length<100) throw new Error('base 10 expansion should keep high precision; got '+dec);
console.log('PASS RIES v11.6.1 log LaTeX and number tools smoke test');
