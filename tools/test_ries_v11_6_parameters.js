const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
for(const id of ['hardDbDepth4','hardDbDepth5','hardDbRationalHeight','hardDbMaxParamHeight','hardDbBudgetMs','hypData1BudgetMs','hypData2BudgetMs','hypData3BudgetMs','logTargetLogAbs','logTargetRaw','logTargetLogLogAbs','integerAllowExternalFactorization']){
  if(!html.includes(`id="${id}"`)) throw new Error(`missing parameter control ${id}`);
}
if(html.includes('id="allowExternalFactorization"')) throw new Error('allow external should be moved out of the top/general area');
for(const id of ['riesLimit','algLimit','logLimit','linearComboLimit','mobiusLimit','constantDbLimit','hardDbLimit','hypDataLimit','lfuncLimit','integerLimit']){
  if(!html.includes(`id="${id}"`)) throw new Error(`missing module candidate limit ${id}`);
}
function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
const elementStore = new Map();
function fakeElement(tag='div', id=''){
  return { id, tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; }, appendChild(child){ return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, matches(){return false;}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
function getEl(id){ if(!elementStore.has(id)) elementStore.set(id, fakeElement('div',id)); return elementStore.get(id); }
const defaults={target:'1.2345', level:'4', limit:'5', maxAbs:'1e9', maxRelError:'Infinity', digits:'0123456789', restrictMode:'none', logHeight:'400', hardDbRationalHeight:'10', hardDbMaxParamHeight:'15', hardDbBudgetMs:'1000', hypData1BudgetMs:'1000', hypData2BudgetMs:'5000', hypData3BudgetMs:'50000', integerFactorBudgetMs:'0'};
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
if(!settings.hardDbOptions.depth4 || !settings.hardDbOptions.depth5) throw new Error('harddb depth defaults should be enabled');
if(!settings.logOptions.targetLogAbs || !settings.logOptions.targetRaw || !settings.logOptions.targetLogLogAbs) throw new Error('log target transforms should default on');
if(settings.moduleLimits.hypData !== 5 || settings.stageBudgets.hypData3Ms !== 50000) throw new Error('module limit/stage budget defaults not read');
getEl('moduleHypData').checked=false;
if(context.readSettings().modules.hypData !== false) throw new Error('module toggle should flow through readSettings');
console.log('PASS RIES v11.6 parameter controls smoke test');
