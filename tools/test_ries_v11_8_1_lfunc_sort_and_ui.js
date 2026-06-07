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
console, performance:{ now:()=>0 }, setTimeout, clearTimeout,
requestAnimationFrame:(fn)=>{ if(typeof fn==='function') setTimeout(fn,0); return 1; }, cancelAnimationFrame:()=>{},
URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} }, Blob:function(parts){ this.parts=parts; }, fetch:undefined, Buffer,
atob:(s)=>Buffer.from(s,'base64').toString('binary') };
context.window=context;
vm.createContext(context);
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js']) vm.runInContext(fs.readFileSync(f,'utf8'), context);

const html=fs.readFileSync('ries.html','utf8');
assert(html.includes('RIES <em>v11.9.1</em>'), 'visible version should be v11.9.1');
assert(html.includes('ries-script.js?v=11.9.1'), 'script cache-buster should be v11.9.1');
for(const id of ['modulesAllOn','modulesAllOff','modulesDefaults']) assert(html.includes(`id="${id}"`), `missing ${id} parameter button`);

const src=fs.readFileSync('ries-script.js','utf8');
assert(src.includes('lfuncGlobalBestRows(pool, llimit, sig)'), 'L-function results should be globally capped across submodules');
assert(src.includes('data-module-toggle'), 'module master switches should target only module toggles');

const T=context.__RIES_LFUNC_TEST__;
const cmp=T.lfuncCompareCandidates(12);
const exactButTall={kind:'quadratic', i:2, j:3, height:50000, err:1e-20, formula:'complex', L:{entryKey:'a'}};
const simpleWithin100x={kind:'rational', i:1, j:0, height:2, err:8e-10, formula:'simple', L:{entryKey:'b'}};
assert([exactButTall, simpleWithin100x].sort(cmp)[0]===simpleWithin100x, 'within 100× typed tolerance, simpler/low-height L-function form should rank first');
const tooLooseSimple={kind:'rational', i:1, j:0, height:2, err:1e-6, formula:'loose', L:{entryKey:'c'}};
assert([tooLooseSimple, exactButTall].sort(cmp)[0]===exactButTall, 'outside the 100× tolerance bucket, error should dominate');

const best=T.lfuncGlobalBestRows([
  {kind:'log', formula:'g', L:{entryKey:'1'}, height:1, err:1e-20},
  {kind:'rational', i:1, j:0, formula:'r1', L:{entryKey:'2'}, height:4, err:9e-10},
  {kind:'rational', i:1, j:1, formula:'r2', L:{entryKey:'3'}, height:4, err:9e-10},
  {kind:'quadratic', i:1, j:0, formula:'q', L:{entryKey:'4'}, height:9, err:9e-10},
  {kind:'rational', i:-1, j:0, formula:'r3', L:{entryKey:'5'}, height:4, err:9e-10},
  {kind:'log', formula:'g2', L:{entryKey:'6'}, height:2, err:9e-10}
], 5, 12);
assert(best.length===5, 'L-function global result cap should keep only five rows');
assert(best[0].formula==='r1', 'global L-function cap should preserve simplicity-first order inside tolerance');

const q=T.lfuncQExpansionLatex({coeffs:[0,1,-22,333,-4444,55555,-666666,7777777,-88888888,999999999,-1111111111,2222222222]});
assert(q.includes('\\begin{aligned}') && q.includes('\\\\&\\quad'), 'long q-expansion should be split into two aligned display lines');
console.log('PASS RIES v11.9.1 L-function sorting, q-expansion split, and parameter master switches test');
