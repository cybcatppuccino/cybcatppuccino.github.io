const fs = require('fs');
const path = require('path');
const vm = require('vm');

function assert(cond, msg){
  if(!cond) throw new Error(msg || 'assertion failed');
}

function canvasCtx(){
  return {
    setTransform(){}, clearRect(){}, beginPath(){}, closePath(){}, moveTo(){}, lineTo(){}, stroke(){}, arc(){}, fill(){}, fillRect(){}, strokeRect(){}, save(){}, restore(){}, translate(){}, rotate(){}, scale(){}, rect(){}, clip(){}, measureText(txt){ return {width:String(txt||'').length*7}; }, fillText(){}, strokeText(){},
    set lineWidth(v){}, get lineWidth(){ return 1; }, set lineCap(v){}, set shadowBlur(v){}, set strokeStyle(v){}, set shadowColor(v){}, set fillStyle(v){}, set font(v){}, set textAlign(v){}, set textBaseline(v){}
  };
}

function createFakeElement(tag='div', id='', getContextRef){
  const el = {
    id, tagName:String(tag).toUpperCase(), nodeName:String(tag).toUpperCase(), type:'', name:'',
    className:'', hidden:false, disabled:false, value:'', checked:false, defaultChecked:false,
    dataset:{}, innerHTML:'', textContent:'', src:'', href:'', async:false, open:false,
    style:{ setProperty(){}, removeProperty(){} },
    classList:{ add(){}, remove(){}, toggle(){ return false; }, contains(){ return false; } },
    setAttribute(name,value){ this[name]=String(value); if(name==='data-sym') this.dataset.sym=String(value); },
    getAttribute(name){ return this[name] || (name==='data-sym' ? this.dataset.sym : '') || ''; },
    removeAttribute(name){ delete this[name]; },
    appendChild(child){ return child; }, prepend(){}, append(){}, remove(){}, replaceChildren(){}, insertAdjacentElement(){}, insertAdjacentHTML(){},
    addEventListener(){}, removeEventListener(){}, dispatchEvent(){ return true; },
    matches(){ return false; }, closest(){ return null; },
    querySelector(){ return createFakeElement('div','',getContextRef); }, querySelectorAll(){ return []; },
    getContext(){ return canvasCtx(); },
    click(){}, focus(){}, select(){}, scrollIntoView(){},
  };
  return el;
}

const DEFAULTS = {
  target:'sqrt(2)', onlySyms:'', neverSyms:'', digits:'0123456789', restrictMode:'none', tolerance:'', maxAbs:'1e9', maxRelError:'Infinity', level:'4', shortEffort:'3', limit:'10',
  logHeight:'400', logPrecision:'80', logSlack:'2', algHeight:'50', algDegree:'4', algPrecision:'80', algResidualPower:'2', defaultLogBasis:'', extraLogBasis:'',
  hardDbRationalHeight:'10', hardDbMaxParamHeight:'15', hardDb4BudgetMs:'1000', hardDb5BudgetMs:'1000', hypData1BudgetMs:'1000', hypData2BudgetMs:'5000', hypData3BudgetMs:'50000', intsumDb1BudgetMs:'1000', intsumDb2BudgetMs:'5000', intsumDb3BudgetMs:'50000',
  riesBudgetMs:'5000', logBudgetMs:'5000', mobiusBudgetMs:'5000', lfuncBudgetMs:'5000', algBudgetMs:'3600', linearComboBudgetMs:'3000', integerFactorBudgetMs:'0'
};

const DEFAULT_CHECKED_IDS = [
  'doEq','moduleRiesEq','doAlg','moduleAlgebraic','doLog','moduleLog','allowExternalFactorization','moduleLinearCombo','moduleMobius','moduleConstantDb','moduleHardDb','moduleHypData','moduleIntsumDb','moduleLfunc','moduleInteger',
  'cdbTransX','cdbTransExp','cdbTransLog','cdbTransInv','cdbTransSquare','cdbPassRational','cdbPassAffine','cdbPassQuadratic','cdbPassMobius','cdbPassAlgebraic','cdbPassLog',
  'hardDbDepth4','hardDbDepth5','hardDbDepth6','hardDbPassRational','hardDbPassPower','hardDbPassExponential','hardDbPassLogScale',
  'hypDepth1','hypDepth2','hypDepth3','hypMultSimple','hypMultGamma','hypMultDeep',
  'intsumDepth1','intsumDepth2','intsumDepth3','intsumMultSimple','intsumMultGamma','intsumMultDeep',
  'lfuncRational','lfuncQuadratic','lfuncLog','specialConstants','lfuncSpecialConstants',
  'logTargetLogAbs','logTargetRaw','logTargetLogLogAbs','integerFactor','integerDb','integerShortform'
];
const DEFAULT_IDS = [
  'resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn',
  'target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','maxRelError','level','shortEffort','limit','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis',
  'hardDbRationalHeight','hardDbMaxParamHeight','hardDb4BudgetMs','hardDb5BudgetMs','hypData1BudgetMs','hypData2BudgetMs','hypData3BudgetMs','intsumDb1BudgetMs','intsumDb2BudgetMs','intsumDb3BudgetMs','riesBudgetMs','logBudgetMs','mobiusBudgetMs','lfuncBudgetMs','algBudgetMs','linearComboBudgetMs','integerFactorBudgetMs',
  ...DEFAULT_CHECKED_IDS
];
const DEFAULT_SYMBOLS = ['p','e','f','n','r','s','q','l','E','S','C','T','+','-','*','/','^','v','L'];

function localScriptPath(src){
  let s = String(src || '');
  s = s.replace(/^https?:\/\/[^/]+\//, '').replace(/^file:\/\//, '');
  s = s.split('?')[0].split('#')[0].replace(/^\.\//, '');
  return s;
}

function loadTextIfExists(relPath){
  const p = path.join(process.cwd(), relPath);
  return fs.existsSync(p) ? fs.readFileSync(p, 'utf8') : null;
}

function makeDom(context, opts={}){
  const elementStore = new Map();
  function getEl(id){
    if(!elementStore.has(id)){
      const el = createFakeElement('div', id);
      if(Object.prototype.hasOwnProperty.call(DEFAULTS, id)) el.value = DEFAULTS[id];
      if(DEFAULT_CHECKED_IDS.includes(id)){ el.checked = true; el.defaultChecked = true; }
      elementStore.set(id, el);
    }
    return elementStore.get(id);
  }
  for(const id of DEFAULT_IDS) getEl(id);
  if(opts.defaults){
    for(const [id, val] of Object.entries(opts.defaults)){
      const el = getEl(id);
      if(typeof val === 'boolean') el.checked = val; else el.value = String(val);
    }
  }
  const doc = {
    addEventListener(){}, removeEventListener(){},
    body:{ contains(){ return true; }, appendChild(script){ return doc.head.appendChild(script); }, prepend(){}, append(){}, classList:{add(){},remove(){}} },
    documentElement:{ appendChild(script){ return doc.head.appendChild(script); }, style:{} },
    head:{ appendChild(script){
      const src = localScriptPath(script.src || (script.getAttribute && script.getAttribute('src')) || '');
      if(src){
        const text = loadTextIfExists(src);
        if(text !== null) vm.runInContext(text, context, {filename:src});
      }
      if(script.text || script.textContent) vm.runInContext(String(script.text || script.textContent), context);
      if(typeof script.onload === 'function') script.onload();
      return script;
    }},
    getElementById:getEl,
    querySelectorAll(sel){
      if(sel === '[data-sym]:checked') return DEFAULT_SYMBOLS.map(sym => ({dataset:{sym}, checked:true}));
      if(sel === '[data-logconst]:checked') {
        const logConsts = context.__RIES_LOG_TEST__ && Array.isArray(context.__RIES_LOG_TEST__.logConstants) ? context.__RIES_LOG_TEST__.logConstants : [];
        return logConsts.filter(c => c.default !== false).map(c => ({dataset:{logconst:c.id}, checked:true}));
      }
      return [];
    },
    querySelector(){ return createFakeElement('div'); },
    createElement(tag){ return createFakeElement(tag); },
    createTextNode(text){ return {text:String(text)}; }
  };
  return {document:doc, getEl, elementStore};
}

function loadRiesContext(opts={}){
  let t = 0;
  const context = {
    window:{}, console,
    performance:{ now:()=>{ if(opts.realTimePerformance) return Date.now(); t += opts.performanceStep === undefined ? 1 : Number(opts.performanceStep); return t; } },
    setTimeout, clearTimeout, setInterval, clearInterval,
    requestAnimationFrame:(fn)=>{ if(typeof fn === 'function') return setTimeout(()=>fn(context.performance.now()), 0); return 1; },
    cancelAnimationFrame:(id)=>clearTimeout(id),
    URL:{ createObjectURL:()=> 'blob:test', revokeObjectURL:()=>{} },
    Blob:function(parts){ this.parts = parts; },
    fetch:undefined,
    Buffer,
    atob:(s)=>Buffer.from(String(s),'base64').toString('binary'),
    btoa:(s)=>Buffer.from(String(s),'binary').toString('base64'),
    navigator:{ clipboard:null, scheduling:{ isInputPending:()=>false } },
    MathJax:null
  };
  context.window = context;
  const dom = makeDom(context, opts);
  context.document = dom.document;
  vm.createContext(context);
  const files = opts.files || ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','ries-script.js'];
  for(const f of files){
    const text = fs.readFileSync(f, 'utf8');
    vm.runInContext(text, context, {filename:f});
  }
  if(opts.loadNewforms) vm.runInContext(fs.readFileSync('assets/newforms.js','utf8'), context, {filename:'assets/newforms.js'});
  return {context, ...dom};
}

function settingsFor(ctxOrBundle, raw, level=4, overrides={}){
  const bundle = ctxOrBundle.context ? ctxOrBundle : {context:ctxOrBundle, getEl:null};
  const context = bundle.context;
  const getEl = bundle.getEl || context.document.getElementById;
  getEl('target').value = String(raw);
  getEl('level').value = String(level);
  for(const [k,v] of Object.entries(overrides.elements || {})){
    const el = getEl(k);
    if(typeof v === 'boolean') el.checked = v; else el.value = String(v);
  }
  let s = typeof context.readSettings === 'function' ? context.readSettings() : {raw:String(raw), target:Number(raw), level:Number(level)};
  s.raw = String(raw);
  s.normalizedRaw = String(raw);
  s.level = Number(level);
  if(overrides.settings) s = Object.assign(s, overrides.settings);
  return s;
}

function settingsForDecimal(context, raw, level=4, overrides={}){
  const P = context.__RIES_PRECISION_TEST__;
  const parsed = P.parseDecimalComplex(String(raw));
  return Object.assign({
    raw:String(raw), normalizedRaw:String(raw), parsedComplex:parsed, complexTarget:false,
    target:P.rationalToNumber(parsed.re), level:Number(level), limit:10,
    modules:{riesEq:true, algebraic:true, log:true, linearCombo:true, mobius:true, constantDb:true, hardDb:true, hypData:true, intsumDb:true, lfunc:true, integer:true},
    moduleLimits:{hypData:20, intsumDb:20},
    lfuncOptions:{rational:true, quadratic:true, log:true, specialConstants:true},
    stageBudgets:{lfuncMs:8000, hypData1Ms:1000, hypData2Ms:5000, hypData3Ms:50000, intsumDb1Ms:1000, intsumDb2Ms:5000, intsumDb3Ms:50000}
  }, overrides);
}

function b64Bytes(b64){ return Buffer.from(String(b64||''), 'base64'); }
function b64U8(b64){ return [...b64Bytes(b64)]; }
function b64U32(b64){ const b=b64Bytes(b64), out=[]; for(let i=0;i<b.length;i+=4) out.push(b.readUInt32LE(i)); return out; }
function b64F64(b64){ const b=b64Bytes(b64), out=[]; for(let i=0;i<b.length;i+=8) out.push(b.readDoubleLE(i)); return out; }

async function runSuite(name, tests){
  let passed = 0;
  for(const [testName, fn] of tests){
    await fn();
    passed += 1;
    console.log(`ok ${name} :: ${testName}`);
  }
  console.log(`PASS ${name} (${passed} tests)`);
}

module.exports = { assert, loadRiesContext, settingsFor, settingsForDecimal, b64Bytes, b64U8, b64U32, b64F64, runSuite };
