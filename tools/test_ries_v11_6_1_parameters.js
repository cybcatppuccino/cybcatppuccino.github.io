const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
for(const id of ['hardDbDepth4','hardDbDepth5','hardDbRationalHeight','hardDbMaxParamHeight','hardDb4BudgetMs','hardDb5BudgetMs','hypData1BudgetMs','hypData2BudgetMs','hypData3BudgetMs','logTargetLogAbs','logTargetRaw','logTargetLogLogAbs','integerAllowExternalFactorization']){
  if(!html.includes(`id="${id}"`)) throw new Error(`missing parameter control ${id}`);
}
if(html.includes('id="hardDbBudgetMs"')) throw new Error('single hardDbBudgetMs control should be replaced by depth-specific controls');
if(!/Stage time limits are active search budgets/.test(html) || !/0<\/code> to remove/.test(html)) throw new Error('stage time-limit explanation missing');
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
const elementStore = new Map();
function fakeElement(tag='div', id=''){
  return { id, tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; }, appendChild(child){ return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, matches(){return false;}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
function getEl(id){ if(!elementStore.has(id)) elementStore.set(id, fakeElement('div',id)); return elementStore.get(id); }
const defaults={target:'1.2345', level:'4', limit:'5', maxAbs:'1e9', maxRelError:'Infinity', digits:'0123456789', restrictMode:'none', logHeight:'400', hardDbRationalHeight:'10', hardDbMaxParamHeight:'15', hardDb4BudgetMs:'1000', hardDb5BudgetMs:'1000', hypData1BudgetMs:'1000', hypData2BudgetMs:'5000', hypData3BudgetMs:'50000', riesBudgetMs:'5000', logBudgetMs:'5000', mobiusBudgetMs:'5000', lfuncBudgetMs:'5000', algBudgetMs:'3600', linearComboBudgetMs:'3000', integerFactorBudgetMs:'0'};
for(const [k,v] of Object.entries(defaults)) getEl(k).value=v;
for(const id of ['moduleRiesEq','moduleAlgebraic','moduleLog','moduleLinearCombo','moduleMobius','moduleConstantDb','moduleHardDb','moduleHypData','moduleLfunc','moduleInteger','hardDbDepth4','hardDbDepth5','hardDbPassRational','hardDbPassPower','hardDbPassExponential','hardDbPassLogScale','hypDepth1','hypDepth2','hypDepth3','hypMultSimple','hypMultGamma','hypMultDeep','logTargetLogAbs','logTargetRaw','logTargetLogLogAbs','integerFactor','integerDb','integerShortform']) getEl(id).checked=true;
const checkedSyms=['p','e','f','n','r','s','q','l','E','S','C','T','+','-','*','/','^','v','L'];
const context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:()=>{} }, head:{ appendChild:(script)=>{ if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:getEl, querySelectorAll:(sel)=> sel==='[data-sym]:checked' ? checkedSyms.map(sym=>({dataset:{sym},checked:true})) : [], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) }, console, performance:{now:()=>0}, setTimeout, clearTimeout, requestAnimationFrame:()=>0, cancelAnimationFrame:()=>{}, URL:{createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{}}, Blob:function(parts){this.parts=parts;}, fetch:undefined, atob:(s)=>Buffer.from(s,'base64').toString('binary'), Buffer };
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);
const settings=context.readSettings();
if(settings.stageBudgets.riesMs!==5000 || settings.stageBudgets.algebraicMs!==3600 || settings.stageBudgets.linearComboMs!==3000) throw new Error('visible default budgets should be read literally');
if(settings.stageBudgets.hardDb4Ms!==1000 || settings.stageBudgets.hardDb5Ms!==1000) throw new Error('harddb depth budgets should be read');
getEl('logBudgetMs').value='0';
const unlimited=context.readSettings();
if(unlimited.stageBudgets.logMs!==0) throw new Error('readSettings should preserve 0 as the unlimited sentinel');
if(context.__RIES_PRECISION_TEST__.stageBudgetValueToMs(unlimited.stageBudgets.logMs,5000)!==Infinity) throw new Error('0 stage budget should become Infinity internally');
console.log('PASS RIES v11.6.1 parameter controls smoke test');
