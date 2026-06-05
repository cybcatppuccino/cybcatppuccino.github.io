const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v11.4.3</em>')) throw new Error('ries.html visible version should be v11.4.3');
if(!html.includes('ries-script.js?v=11.4.3')) throw new Error('ries-script cache tag should be v11.4.3');
if(/<script[^>]+ries-harddb-v11_4_1-filtered\.js/i.test(html)) throw new Error('hard DB package must not be part of the initial HTML payload');

const readme = fs.readFileSync('README.md','utf8');
if(/## RIES v\d/.test(readme) || /update|changelog/i.test(readme.replace('Detailed RIES release notes are kept under `changelog/`.',''))) throw new Error('README should not display release/update notes');
if(!fs.existsSync('changelog/RIES_v11.4.3_CHANGELOG.md')) throw new Error('v11.4.3 changelog missing from changelog folder');
if(!fs.existsSync('UPDATE_GUIDELINES.md')) throw new Error('root update guidelines missing');
const rootChangelogs = fs.readdirSync('.').filter(f => /^RIES_v.*_CHANGELOG\.md$/.test(f));
if(rootChangelogs.length) throw new Error('root changelog files should be moved into changelog/: '+rootChangelogs.join(','));

function canvasCtx(){
  return {
    setTransform(){}, clearRect(){}, beginPath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, save(){}, restore(){}, translate(){}, rotate(){},
    set lineWidth(v){}, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){}
  };
}
function fakeElement(tag='div'){
  const el = {
    tagName: String(tag).toUpperCase(), className:'', hidden:false, disabled:false, value:'', checked:false, dataset:{}, innerHTML:'', textContent:'', src:'',
    style:{ setProperty(){}, removeProperty(){} },
    classList:{ add(){}, remove(){}, toggle(){}, contains(){ return false; } },
    setAttribute(name, value){ this[name] = String(value); },
    getAttribute(name){ return this[name] || ''; },
    appendChild(child){ return child; }, prepend(){}, remove(){}, addEventListener(){}, removeEventListener(){},
    querySelector(){ return fakeElement('div'); }, querySelectorAll(){ return []; }, getContext(){ return canvasCtx(); }
  };
  return el;
}

const context = {
  window: {},
  document: {
    addEventListener: () => {},
    body: { contains: () => true, appendChild: () => {} },
    head: {
      appendChild: (script) => {
        const src = String(script.src || script.getAttribute?.('src') || '');
        if(src.includes('ries-harddb-v11_4_1-filtered.js')){
          vm.runInContext(fs.readFileSync('assets/ries-harddb-v11_4_1-filtered.js','utf8'), context);
        }
        if(typeof script.onload === 'function') script.onload();
        return script;
      }
    },
    getElementById: () => fakeElement('div'),
    querySelectorAll: () => [],
    querySelector: () => fakeElement('div'),
    createElement: (tag) => fakeElement(tag)
  },
  console,
  performance: { now: () => 0 },
  setTimeout, clearTimeout,
  requestAnimationFrame: () => 0,
  cancelAnimationFrame: () => {},
  URL: { createObjectURL: () => 'blob:test', revokeObjectURL: () => {} },
  Blob: function(parts){ this.parts = parts; },
  fetch: undefined
};
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), context);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);

const T = context.__RIES_CONSTDB_TEST__;
if(!T || typeof T.constantDbBudgetMs !== 'function') throw new Error('constant DB test hooks missing');
if(T.constantDbBudgetMs(4,16)!==20000) throw new Error('level 4 constant DB budget should be 20000');
if(T.constantDbBudgetMs(5,16)!==45000) throw new Error('level 5 constant DB budget should remain 45000');
if(T.constantDbBudgetMs(6,16)!==135000) throw new Error('level 6 constant DB budget should remain 135000');

const H = context.__RIES_HARDDB_TEST__;
if(!H || typeof H.ensureHardDbLoaded !== 'function') throw new Error('hard DB test hooks missing');
const settings = { target: 1.2345, complexTarget:false, maxLevel: 5 };
if(!H.hardDbPotentiallyRunnable(settings)) throw new Error('hard DB should be potentially runnable before its package is loaded');
if(H.isHardDbReady()) throw new Error('hard DB should start unloaded for startup performance');
if(H.hardDbShouldRun(settings)) throw new Error('hard DB should not run until the lazy package is ready');

(async () => {
  const loaded = await H.ensureHardDbLoaded({ label:'filtered hard-constant database', phase:'test package load' });
  if(!loaded) throw new Error('hard DB package did not load lazily');
  if(!H.isHardDbReady()) throw new Error('hard DB readiness hook should be true after lazy load');
  if(!H.hardDbShouldRun(settings)) throw new Error('hard DB should run after lazy loading');

  if(H.resultRowCategory({ candidate:'hard constant database: x ≈ |A|', hardDbCategory:'Euler beta integral' }) !== 'harddb') throw new Error('hard DB must be its own sorted module category');
  const comboLatex = H.hardDbFormulaLatex({ category:'common Log-Exp-Trig composition', params:{ template:'1', params:'{1, 1/2, 2, 3}' } });
  if(/C_\{?template/i.test(comboLatex) || !/\\log/.test(comboLatex)) throw new Error('hard DB common composition should render an explanatory formula, not a placeholder');

  const rows = [
    { candidate:'RIES equation: x ≈ 1', latex:'x=1', err:1e-12, terms:2 },
    { candidate:'RIES equation: x ≈ 2', latex:'x=2', err:1e-12, terms:2 },
    { candidate:'hard constant database: x ≈ |A|', latex:'x\\approx |A|', err:1e-12, hardDbCategory:'Euler beta integral', terms:2 },
    { candidate:'hard constant database: x ≈ sign(A)·|A|^(2)', latex:'x\\approx A^2', err:1e-12, hardDbCategory:'Euler beta integral', terms:3 },
    { candidate:'log|c| linear relation: x ≈ π', latex:'x=\\pi', err:1e-12, terms:2 },
    { candidate:'log|c| linear relation: x ≈ π^2', latex:'x=\\pi^2', err:1e-12, terms:3 }
  ];
  const sorted = H.confidenceSortedRows(rows, { sortMode:'confidence', target:1.2345, complexTarget:false });
  const hardIdx = sorted.findIndex(r => /hard constant database/.test(r.candidate));
  const hardSecondIdx = sorted.findIndex((r, i) => i > hardIdx && /hard constant database/.test(r.candidate));
  if(hardIdx < 0 || hardSecondIdx < 0) throw new Error('sorted results should contain both hard DB rows');
  if(hardSecondIdx < 3) throw new Error('hard DB second-place result should appear in the second interleaved layer, not directly after hard DB first-place result');

  console.log('PASS RIES v11.4.3 packaging, lazy hard DB, and sorting smoke test');
})().catch(err => { console.error(err); process.exit(1); });
