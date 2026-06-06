const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('ries.html','utf8');
for(const id of ['moduleRiesEq','moduleAlgebraic','moduleLog','moduleLinearCombo','moduleMobius','moduleConstantDb','moduleHardDb','moduleHypData','moduleLfunc','moduleInteger']){
  if(!html.includes(`id="${id}"`)) throw new Error(`missing module toggle ${id}`);
  if(!html.includes(`data-module-body="${id}"`)) throw new Error(`missing collapsible body ${id}`);
}
if(/<h3>General<\/h3>/.test(html)) throw new Error('old General block should be removed');
for(const sym of ['S','C','T','v','L']){
  const re = new RegExp(`data-sym="${sym.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')}" checked`);
  if(!re.test(html)) throw new Error(`RIES symbol ${sym} should be checked by default`);
}

function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; }, appendChild(child){ return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
const context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:()=>{} }, head:{ appendChild:(script)=>{ if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) }, console, performance:{now:()=>0}, setTimeout, clearTimeout, requestAnimationFrame:()=>0, cancelAnimationFrame:()=>{}, URL:{createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{}}, Blob:function(parts){this.parts=parts;}, fetch:undefined, atob:(s)=>Buffer.from(s,'base64').toString('binary'), Buffer };
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);

const H=context.__RIES_HARDDB_TEST__;
const T=context.__RIES_HYPDATA_TEST__;
const C=context.__RIES_CONSTDB_TEST__;
const M=context.__RIES_MOBIUS_TEST__;
if(H.hardDbPotentiallyRunnable({target:1.2,complexTarget:false,level:5,modules:{hardDb:false}})) throw new Error('harddb toggle should disable harddb');
if(T.hypDataPotentiallyRunnable({target:1.2,complexTarget:false,level:6,modules:{hypData:false},hypDataOptions:{depth1:true,depth2:true,depth3:true}})) throw new Error('hypdata module toggle should disable hypdata');
if(T.hypDataMaxStage({level:6,hypDataOptions:{depth1:true,depth2:false,depth3:true}})!==1) throw new Error('hypdata depth toggles should enforce contiguous cumulative loading');
if(C.shouldRunConstantDbRows({target:1.2,raw:'1.2',complexTarget:false,modules:{constantDb:false}})) throw new Error('constant DB toggle should disable constant DB');
if(M.shouldRunMobiusRows({target:1.2,raw:'1.2',complexTarget:false,modules:{mobius:false}})) throw new Error('Möbius toggle should disable Möbius search');
console.log('PASS RIES v11.5.2 parameter UI and module-toggle smoke test');
