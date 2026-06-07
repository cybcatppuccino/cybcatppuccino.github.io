const fs = require('fs');
const vm = require('vm');

function el(){
  return {
    value:'', checked:false, disabled:false, hidden:false, dataset:{}, className:'',
    textContent:'', innerHTML:'', open:false,
    addEventListener(){}, appendChild(){}, closest(){ return null; }, querySelectorAll(){ return []; }
  };
}

const ids = new Map();
const doc = {
  getElementById(id){ if(!ids.has(id)) ids.set(id, el()); return ids.get(id); },
  addEventListener(){}, querySelector(){ return el(); }, querySelectorAll(){ return []; },
  createElement(){ return el(); }, head:el(), body:el(), documentElement:el()
};
ids.set('level', { ...el(), value:'4' });
ids.set('shortEffort', { ...el(), value:'3' });
ids.set('limit', { ...el(), value:'5' });
ids.set('target', { ...el(), value:'' });
ids.set('digits', { ...el(), value:'0123456789' });
for(const id of ['doEq','doAlg','doLog']) ids.set(id, { ...el(), checked:true });
for(const id of ['maxAbs','only','never']) ids.set(id, { ...el(), value:'' });

const context = { console, performance:{ now:()=>Date.now() }, window:{}, document:doc, navigator:{}, setTimeout, clearTimeout };
context.window = context;
vm.createContext(context);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), context);

const api = context.__RIES_LINEAR_COMBO_TEST__;
if(!api) throw new Error('linear-combo test API missing');

const additions = [
  ['Re PolyGamma[0, exp(2πi/8)]', -0.36034322975434906059],
  ['Re PolyGamma[0, exp(2πi/6)]', -0.21492655872969647407],
  ['Re PolyGamma[0, exp(2πi/4)]',  0.094650320622476977272],
  ['Re PolyGamma[0, exp(2πi/3)]',  0.28507344127030352593],
  ['Im PolyGamma[0, exp(2πi/8)]',  1.2199689267137873388],
  ['Im PolyGamma[0, exp(2πi/6)]',  1.5572412247131940501],
  ['Im PolyGamma[0, exp(2πi/4)]',  2.0766740474685811741],
  ['Im PolyGamma[0, exp(2πi/3)]',  2.4232666284976326969],
  ['Σ (-1)^n/(n log n), n≥2',       0.526412246533310],
  ['Σ (-1)^n/((n−1)log n), n≥2',    1.136448654977018],
  ['Σ (-1)^n/log n, n≥2',           0.92429989722294],
  ['Σ (-1)^n/(log n)², n≥2',        1.55701835019512]
];

const previousLevel5Additions = [
  'π√2', 'π√3', 'Σ 1/(n²+1)', 'Σ 1/(n³+1)', 'Σ (-1)^n/(n²+1)',
  'Σ (-1)^n/C(2n,n)', 'Σ 1/C(3n,n)', 'Σ 1/(n²+n+1)',
  'Σ (-1)^n/(n²+n+1)', 'Σ 1/(1+2^n)'
];

const level4 = api.lowPrecisionLinearComboBasisConstants({level:4});
const level5 = api.lowPrecisionLinearComboBasisConstants({level:5});
if(level4.length !== 107) throw new Error(`expected level 4 basis size 107, got ${level4.length}`);
if(level5.length < 128) throw new Error(`expected level 5 basis size at least 128, got ${level5.length}`);
const l4 = new Set(level4.map(c=>c.label));
const l5 = new Set(level5.map(c=>c.label));
function hasLabelIncluding(list, needle){ return list.some(c => String(c.label).includes(needle)); }
for(const label of previousLevel5Additions){
  if(hasLabelIncluding(level4, label)) throw new Error('previous level-5 constant leaked into level 4 basis: '+label);
  if(!hasLabelIncluding(level5, label)) throw new Error('previous level-5 constant missing from level 5 basis: '+label);
}
for(const [label] of additions){
  if(hasLabelIncluding(level4, label)) throw new Error('v11.6.3 level-5 constant leaked into level 4 basis: '+label);
  if(!hasLabelIncluding(level5, label)) throw new Error('v11.6.3 level-5 constant missing from level 5 basis: '+label);
}

function rowsFor(value, level){
  const raw = String(Number(value).toPrecision(15));
  return api.lowPrecisionLinearComboRows({
    raw, normalizedRaw:raw, target:Number(value), complexTarget:false, level, limit:50,
    modules:{linearCombo:true}, linearComboOptions:{one:true,two:false,three:false},
    moduleLimits:{linearCombo:50}, stageBudgets:{linearComboMs:3000}
  });
}

for(const [label, value] of additions){
  const at4 = rowsFor(value, 4);
  if(at4.some(r => String(r.candidate).includes(label))) throw new Error('v11.6.3 level-5-only hit appeared at level 4: '+label);
  const at5 = rowsFor(value, 5);
  if(!at5.some(r => String(r.candidate).includes(label))){
    throw new Error('missing v11.6.3 level 5 direct hit for '+label+':\n'+at5.map(r=>r.candidate).join('\n'));
  }
}

console.log('PASS RIES v11.6.3 linear-combo level-5 additions smoke test');
