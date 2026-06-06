const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v11.5</em>')) throw new Error('ries.html visible version should be v11.5');
if(!html.includes('ries-script.js?v=11.5')) throw new Error('ries-script cache tag should be v11.5');
if(/<script[^>]+ries-hypdata-v11_5\.js/i.test(html)) throw new Error('hypdata package must be lazy-loaded, not part of the initial HTML payload');
if(!fs.existsSync('assets/ries-hypdata-v11_5.js')) throw new Error('hypdata v11.5 asset missing');
if(!fs.existsSync('assets/ries-hypdata-v11_5-stats.json')) throw new Error('hypdata v11.5 stats missing');
if(!fs.existsSync('changelog/RIES_v11.5_CHANGELOG.md')) throw new Error('v11.5 changelog missing');

const stats = JSON.parse(fs.readFileSync('assets/ries-hypdata-v11_5-stats.json','utf8'));
if(stats.version !== '11.5') throw new Error('hypdata stats version mismatch');
if(stats.rows !== 109738) throw new Error('unexpected hypdata row count: '+stats.rows);
if(stats.realRows !== 36874) throw new Error('unexpected hypdata real row count: '+stats.realRows);
if(stats.tierCounts['1'] !== 3159 || stats.tierCounts['2'] !== 36407 || stats.tierCounts['3'] !== 70172) throw new Error('unexpected hypdata tier counts');
if(stats.multiplierRows !== 16000) throw new Error('unexpected multiplier count');
if(!stats.inputStats.some(s => Number(s.badFileCount||0) > 0)) throw new Error('stats should record skipped corrupt/empty JSON blocks');

function canvasCtx(){
  return { setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){}, set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){} };
}
function fakeElement(tag='div'){
  return {
    tagName: String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'', async:false,
    style:{ setProperty(){}, removeProperty(){} },
    classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name, value){ this[name] = String(value); }, getAttribute(name){ return this[name] || ''; },
    appendChild(child){ if(child && child.text) vm.runInContext(String(child.text), context); return child; },
    prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){}, querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); }
  };
}

const context = {
  window: {},
  document: {
    addEventListener: () => {},
    body: { contains: () => true, appendChild: (script) => context.document.head.appendChild(script) },
    head: {
      appendChild: (script) => {
        const src = String(script.src || script.getAttribute?.('src') || '');
        if(src.includes('ries-hypdata-v11_5.js')) vm.runInContext(fs.readFileSync('assets/ries-hypdata-v11_5.js','utf8'), context);
        if(src.includes('ries-harddb-v11_4_1-filtered.js')) vm.runInContext(fs.readFileSync('assets/ries-harddb-v11_4_1-filtered.js','utf8'), context);
        if(script.text) vm.runInContext(String(script.text), context);
        if(typeof script.onload === 'function') script.onload();
        return script;
      }
    },
    getElementById: () => fakeElement('div'), querySelectorAll: () => [], querySelector: () => fakeElement('div'), createElement: (tag) => fakeElement(tag)
  },
  console,
  performance: { now: (() => { let t=0; return () => { t += 1; return t; }; })() },
  setTimeout, clearTimeout,
  requestAnimationFrame: (fn) => { if(typeof fn === 'function') setTimeout(fn,0); return 1; }, cancelAnimationFrame: () => {},
  URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} }, Blob: function(parts){ this.parts = parts; },
  fetch: undefined,
  Buffer,
  atob: (s) => Buffer.from(s, 'base64').toString('binary')
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);

const T = context.__RIES_HYPDATA_TEST__;
if(!T || typeof T.ensureHypDataLoaded !== 'function') throw new Error('hypdata test hooks missing');
const settings = { target: 1, complexTarget:false, level:4, sortMode:'confidence' };
if(!T.hypDataPotentiallyRunnable(settings)) throw new Error('hypdata should be potentially runnable before lazy load');
if(T.isHypDataReady()) throw new Error('hypdata should start unloaded');
if(T.hypDataMaxStage({level:4}) !== 1) throw new Error('level 4 should map to hypdata stage 1');
if(T.hypDataMaxStage({level:6}) !== 2) throw new Error('level 6 should map to hypdata stage 2');
if(T.hypDataMaxStage({level:9}) !== 3) throw new Error('level 9 should map to hypdata stage 3');
if(T.resultRowCategory({candidate:'hypergeometric database: x ≈ H', hypDataCategory:'fast/common'}) !== 'hypdata') throw new Error('hypdata must be an independent sorted module category');
if(!/_2F1/.test(T.hypDataMkText('P|0|1/2,1/2|1|1/2'))) throw new Error('MK text renderer should render pFq shape');
if(!/\{\}_\{2\}F_\{1\}/.test(T.hypDataMkLatex('P|0|1/2,1/2|1|1/2'))) throw new Error('MK latex renderer should render pFq notation');

function firstTier1RealTarget(payload){
  const values = Buffer.from(payload.realValuesB64, 'base64');
  const rows = Buffer.from(payload.realRowB64, 'base64');
  for(let i=0; i<rows.length/4; i++){
    const row = rows.readUInt32LE(i*4);
    if(row < 3159){
      const val = values.readDoubleLE(i*8);
      if(Number.isFinite(val) && Math.abs(val) > 1e-12) return val;
    }
  }
  throw new Error('no tier-1 real target found');
}

(async () => {
  const loaded = await T.ensureHypDataLoaded({label:'hypergeometric pFq database', phase:'test load'});
  if(!loaded || !T.isHypDataReady()) throw new Error('hypdata did not lazy-load');
  const payload = context.RIES_HYPDATA_V115;
  if(payload.version !== '11.5' || payload.rows !== stats.rows) throw new Error('hypdata payload shape mismatch');
  const target = firstTier1RealTarget(payload);
  const hits = T.hypDataSearch({ target, complexTarget:false, level:4, sortMode:'confidence' });
  if(!Array.isArray(hits) || hits.length < 1 || hits.length > 5) throw new Error('hypdata search should return 1–5 hits for a known tier-1 target');
  const rows = await T.hypDataRowsAsync({ target, complexTarget:false, level:4, sortMode:'confidence' });
  if(!Array.isArray(rows) || rows.length < 1 || rows.length > 5) throw new Error('hypdata row formatter should return 1–5 rows');
  if(!/^hypergeometric database:/i.test(rows[0].candidate)) throw new Error('hypdata row should use hypergeometric database prefix');
  if(!rows[0].hypDataCategory || !rows[0].copyValue || !rows[0].latex || !rows[0].valueHtml) throw new Error('hypdata rows should include category, value/copy text, and latex');
  console.log('PASS RIES v11.5 hypdata lazy asset, staged search, and result formatting smoke test');
})().catch(err => { console.error(err); process.exit(1); });
