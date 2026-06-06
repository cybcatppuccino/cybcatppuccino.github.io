const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v11.5.2</em>')) throw new Error('ries.html visible version should be v11.5.2');
if(!html.includes('ries-script.js?v=11.5.2')) throw new Error('ries-script cache tag should be v11.5.2');
if(/<script[^>]+ries-harddb-v11_4_1-filtered\.js/i.test(html)) throw new Error('hard DB package must remain lazy-loaded');
if(/<script[^>]+ries-hypdata-v11_5_2-level[456]\.js/i.test(html)) throw new Error('hypdata chunks must remain lazy-loaded');

const readme = fs.readFileSync('README.md','utf8');
if(/## RIES v\d/.test(readme) || /update|changelog/i.test(readme.replace('Detailed RIES release notes are kept under `changelog/`.',''))) throw new Error('README should not display release/update notes');
if(!fs.existsSync('changelog/RIES_v11.5.2_CHANGELOG.md')) throw new Error('v11.5.2 changelog missing from changelog folder');
if(!fs.existsSync('UPDATE_GUIDELINES.md')) throw new Error('root update guidelines missing');

function canvasCtx(){ return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} }; }
function fakeElement(tag='div'){
  return { tagName:String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} }, classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); }, getAttribute(name){ return this[name]||''; }, appendChild(child){ return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); } };
}
const context={ window:{}, document:{ addEventListener:()=>{}, body:{ contains:()=>true, appendChild:()=>{} }, head:{ appendChild:(script)=>{ const src=String(script.src||script.getAttribute?.('src')||''); if(src.includes('ries-harddb-v11_4_1-filtered.js')) vm.runInContext(fs.readFileSync('assets/ries-harddb-v11_4_1-filtered.js','utf8'), context); if(typeof script.onload==='function') script.onload(); return script; } }, getElementById:()=>fakeElement('div'), querySelectorAll:()=>[], querySelector:()=>fakeElement('div'), createElement:(tag)=>fakeElement(tag) }, console, performance:{now:()=>0}, setTimeout, clearTimeout, requestAnimationFrame:()=>0, cancelAnimationFrame:()=>{}, URL:{createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{}}, Blob:function(parts){this.parts=parts;}, fetch:undefined, atob:(s)=>Buffer.from(s,'base64').toString('binary'), Buffer };
context.window=context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);

const H=context.__RIES_HARDDB_TEST__;
if(!H || typeof H.hardDbLevelEnabled!=='function') throw new Error('hard DB test hooks missing');
if(H.hardDbPotentiallyRunnable({target:1.2345,complexTarget:false,level:4})) throw new Error('hard DB must not run at level 4');
if(!H.hardDbPotentiallyRunnable({target:1.2345,complexTarget:false,level:5})) throw new Error('hard DB must run at level 5');
if(H.hardDbPotentiallyRunnable({target:1.2345,complexTarget:false,level:6})) throw new Error('hard DB must not run at level 6');
if(H.hardDbShouldRun({target:1.2345,complexTarget:false,level:5})) throw new Error('hard DB should not run before lazy package is loaded');
(async()=>{
  const loaded=await H.ensureHardDbLoaded({label:'filtered hard-constant database',phase:'test package load'});
  if(!loaded || !H.isHardDbReady()) throw new Error('hard DB package did not load lazily');
  if(!H.hardDbShouldRun({target:1.2345,complexTarget:false,level:5})) throw new Error('hard DB should run after lazy loading at level 5');
  if(H.hardDbShouldRun({target:1.2345,complexTarget:false,level:6})) throw new Error('hard DB should still be disabled at level 6');
  console.log('PASS RIES v11.5.2 packaging and level-5-only harddb smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
