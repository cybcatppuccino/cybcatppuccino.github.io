const fs = require('fs');
const vm = require('vm');
function assert(cond,msg){ if(!cond) throw new Error(msg); }

const stats = JSON.parse(fs.readFileSync('assets/ries-harddb-v11_7_3-stats.json','utf8'));
assert(stats.version === '11.7.3-harddb-pruned-full-20260607', 'harddb stats version mismatch');
assert(stats.sourceRows === 79932, 'harddb source row count changed');
assert(stats.removedRows === 56324, 'harddb removed row count mismatch');
assert(stats.remainingRows === 23608, 'harddb remaining row count mismatch');
const removedExpected = {
  'low-height hypergeometric pFq': 3048,
  'Euler beta integral fast': 2555,
  'incomplete beta integral fast': 30000,
  'beta logarithmic integral fast': 15000,
  'gamma log-laplace integral fast': 221,
  'rational Mellin integral fast': 5500,
};
for(const [k,v] of Object.entries(removedExpected)) assert(stats.removedCategories[k] === v, `removed count mismatch for ${k}`);

const html = fs.readFileSync('ries.html','utf8');
assert(html.includes('RIES <em>v11.8.1</em>'), 'visible version should be v11.8.1');
assert(html.includes('ries-script.js?v=11.8.1'), 'script cache-buster should be v11.8.1');
assert(html.includes('hardDbDepth6') && html.includes('hardDb6BudgetMs'), 'harddb depth 6 controls missing');
assert(!/depth 4 low-height 20%|depth 5 remaining rows/.test(html), 'stale harddb depth wording leaked');

const script = fs.readFileSync('ries-script.js','utf8');
assert(script.includes('ries-harddb-v11_7_3-level4.js?v=11.7.3'), 'script should keep loading the unchanged v11.7.3 harddb asset');
assert(!script.includes('ries-harddb-v11_6-level4.js?v=11.6') && !script.includes('ries-harddb-v11_6-level5.js?v=11.6'), 'old harddb split assets should not be referenced');
assert(script.includes("constantDbSource:'harddb-v11.7.3-pruned'"), 'harddb result source marker mismatch');

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
  if(src.includes('ries-harddb-v11_7_3-level4.js')) vm.runInContext(fs.readFileSync('assets/ries-harddb-v11_7_3-level4.js','utf8'), context);
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
const T=context.__RIES_HARDDB_TEST__;
assert(T && typeof T.ensureHardDbLoaded === 'function', 'harddb test hooks missing');
const base={target:1, complexTarget:false, modules:{hardDb:true}, hardDbOptions:{depth4:true, depth5:true, depth6:true, rational:true, power:true, exponential:true, logScale:true, rationalHeight:20, maxParamHeight:15}, moduleLimits:{hardDb:5}, stageBudgets:{hardDb4Ms:3000, hardDb5Ms:15000, hardDb6Ms:50000}};
assert(T.hardDbMaxStage({...base, level:3}) === 0, 'level 3 should not run harddb');
assert(T.hardDbMaxStage({...base, level:4}) === 1, 'level 4 should map to simple constants stage');
assert(T.hardDbMaxStage({...base, level:5}) === 2, 'level 5 should map to core constants stage');
assert(T.hardDbMaxStage({...base, level:6}) === 3, 'level 6 should map to extended constants stage');
assert(T.hardDbMaxStage({...base, level:6, hardDbOptions:{...base.hardDbOptions, depth6:false}}) === 2, 'depth6 toggle should constrain harddb stage');
assert(T.hardDbRationalHeightForStage(base,1) === 8, 'stage 1 rational height cap mismatch');
assert(T.hardDbRationalHeightForStage(base,2) === 12, 'stage 2 rational height cap mismatch');
assert(T.hardDbRationalHeightForStage(base,3) === 20, 'stage 3 rational height cap mismatch');
assert(T.hardDbSpecialsForStage(1).length < T.hardDbSpecialsForStage(2).length && T.hardDbSpecialsForStage(2).length < T.hardDbSpecialsForStage(3).length, 'harddb comparison constants should expand with depth');

(async()=>{
  assert(!T.isHardDbReady(1), 'harddb should begin unloaded');
  const loaded = await T.ensureHardDbLoaded({settings:{...base, level:6}, stage:3, label:'harddb', phase:'test'});
  assert(loaded && T.isHardDbReady(3), 'harddb v11.7.3 asset did not load');
  assert(T.hardDbLoadedChunks(3).length === 1, 'harddb should use one full pruned chunk at all depths');

  const asset = context.RIES_HARDDB_V1173_CHUNKS[0];
  assert(asset.rows === 23608, 'active asset rows mismatch');
  for(const k of Object.keys(removedExpected)){
    const cid = asset.categories.indexOf(k);
    assert(cid >= 0 && asset.catCounts[cid] === 0, `removed category still has active rows: ${k}`);
  }
  const removed = new Set(Object.keys(removedExpected));
  let checked=0;
  for(let i=0;i<asset.rows;i++){
    const meta=T.hardDbDecodeRowMeta(i, 3);
    assert(!removed.has(meta.category), `removed category decoded from active rows: ${meta.category}`);
    const latex=T.hardDbFormulaLatex(meta);
    assert(latex && typeof latex === 'string', `empty harddb latex at row ${i}`);
    assert(!/[\u0008\u0009\u000c\u000d]/.test(latex), `harddb control escape at row ${i}: ${latex}`);
    assert(!/[+-]{2}|\+\s*-|-\s*\+/.test(latex), `harddb sign run at row ${i}: ${latex}`);
    assert(!/\^\{0\}|\^\{1\}|\^0(?!\d)|\^1(?!\d)/.test(latex.replace(/_0\^1/g,'')), `harddb neutral power at row ${i}: ${latex}`);
    checked++;
  }
  assert(checked === 23608, `checked ${checked} harddb formulas`);
  const rows = await T.hardDbRowsAsync({...base, level:4, target:Math.abs(new Float64Array(Buffer.from(asset.valuesB64,'base64').buffer)[0])});
  assert(Array.isArray(rows) && rows.length <= 5, 'harddb rows should respect module limit');
  if(rows.length){
    assert(rows[0].constantDbSource === 'harddb-v11.7.3-pruned', 'harddb row source marker mismatch');
    assert(rows[0].latex.includes('\\approx'), 'harddb closed form should use approximate relation');
  }
  console.log('PASS RIES v11.8.1 pruned harddb database and staged constants regression test');
})().catch(err=>{ console.error(err); process.exit(1); });
