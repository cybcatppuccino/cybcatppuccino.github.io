
const fs = require('fs');
const vm = require('vm');
const {performance} = require('perf_hooks');
const Decimal = require('../assets/decimal.js');
global.window = global;
global.Decimal = Decimal;
require('../assets/lfunctions-l2l4.js');
function fakeEl(id){ return {id, value:'', checked:true, disabled:false, dataset:{}, className:'', textContent:'', innerHTML:'', style:{}, appendChild(){}, addEventListener(){}, querySelectorAll(){ return []; }, setAttribute(){}, getAttribute(){ return ''; }, classList:{add(){},remove(){},toggle(){}}}; }
const elems = new Map();
global.document = {
  getElementById(id){ if(!elems.has(id)) elems.set(id, fakeEl(id)); return elems.get(id); },
  createElement(tag){ return fakeEl(tag); },
  body:{appendChild(){}, innerText:''},
  querySelector(){ return fakeEl('query'); },
  querySelectorAll(){ return []; },
  addEventListener(){}
};
global.performance = performance;
global.requestAnimationFrame = cb => setTimeout(cb, 0);
let code = fs.readFileSync('ries-script.js','utf8');
code = code.replace(/\}\)\(\);\s*$/, 'window.__riesV91Test={lfuncRowsAsync,parseDecimalComplex,rationalToNumber,lfuncEntries,specialDecimalConstantRows};})();');
vm.runInThisContext(code);
function settings(raw){
  const parsed = global.__riesV91Test.parseDecimalComplex(raw);
  return {raw, normalizedRaw:raw, parsedComplex:parsed, complexTarget:false, target:global.__riesV91Test.rationalToNumber(parsed.re), limit:10};
}
function truncSig(s,d){ const [a,b=''] = s.split('.'); if(d<=a.length) return a.slice(0,d); return a+'.'+b.slice(0,d-a.length); }
(async()=>{
  const entries = global.__riesV91Test.lfuncEntries();
  const lval = entries.find(e => e.value.startsWith('2.298486058160745'));
  if(!lval) throw new Error('expected L-function fixture not found');
  for(const d of [8,9,10,12,15,18,20]){
    const raw = truncSig(lval.value, d);
    const rows = await global.__riesV91Test.lfuncRowsAsync(settings(raw), 0, ()=>{});
    if(!rows.some(r => /^L-rational/.test(r.candidate) && /x = L\(f,1\)/.test(r.candidate))){
      throw new Error(`simple L-function positive missed at ${d} sig digits (${raw})`);
    }
  }
  const gammaRows = global.__riesV91Test.specialDecimalConstantRows(settings('3.6256099082219'), 1);
  if(!gammaRows.some(r => r.candidate.includes('Γ(1/4)'))) throw new Error('Gamma(1/4) first-continue match missed');
  console.log('PASS v9.1 decimal positive L/Gamma matches');
})();
