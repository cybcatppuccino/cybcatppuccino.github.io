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

const newAdditions = [
  ['Σ (-1)^n log n/n, n≥1',    0.15986890374243097176],
  ['Σ (-1)^n log n/n², n≥1',   0.10131657816350450189],
  ['Σ log n/n², n≥1',          0.93754825431584375370],
  ['Σ log n/n³, n≥1',          0.19812624288563685333]
];

const combinedLabel = 'Σ 1/(n²+1) / Im PolyGamma[0, exp(2πi/4)]';

const level4 = api.lowPrecisionLinearComboBasisConstants({level:4});
const level5 = api.lowPrecisionLinearComboBasisConstants({level:5});
if(level4.length !== 107) throw new Error(`expected level 4 basis size 107, got ${level4.length}`);
if(level5.length !== 132) throw new Error(`expected level 5 basis size 132 after merging duplicate and adding four constants, got ${level5.length}`);

function hasLabelIncluding(list, needle){ return list.some(c => String(c.label).includes(needle)); }

if(hasLabelIncluding(level4, combinedLabel)) throw new Error('combined duplicate level-5 constant leaked into level 4 basis');
if(!hasLabelIncluding(level5, combinedLabel)) throw new Error('combined duplicate level-5 constant missing from level 5 basis');
if(!hasLabelIncluding(level5, 'Σ 1/(n²+1)') || !hasLabelIncluding(level5, 'Im PolyGamma[0, exp(2πi/4)]')){
  throw new Error('combined duplicate constant should expose both labels');
}
const duplicateValueRows = level5.filter(c => Math.abs(c.value - 2.0766740474685811741) < 1e-12);
if(duplicateValueRows.length !== 1) throw new Error(`expected exactly one level-5 basis row for the duplicate value, got ${duplicateValueRows.length}`);

for(const [label] of newAdditions){
  if(hasLabelIncluding(level4, label)) throw new Error('v11.6.4 level-5 constant leaked into level 4 basis: '+label);
  if(!hasLabelIncluding(level5, label)) throw new Error('v11.6.4 level-5 constant missing from level 5 basis: '+label);
}

function rowsFor(value, level){
  const raw = String(Number(value).toPrecision(15));
  return api.lowPrecisionLinearComboRows({
    raw, normalizedRaw:raw, target:Number(value), complexTarget:false, level, limit:50,
    modules:{linearCombo:true}, linearComboOptions:{one:true,two:false,three:false},
    moduleLimits:{linearCombo:50}, stageBudgets:{linearComboMs:3000}
  });
}

for(const [label, value] of newAdditions){
  const at4 = rowsFor(value, 4);
  if(at4.some(r => String(r.candidate).includes(label))) throw new Error('v11.6.4 level-5-only hit appeared at level 4: '+label);
  const at5 = rowsFor(value, 5);
  if(!at5.some(r => String(r.candidate).includes(label))){
    throw new Error('missing v11.6.4 level 5 direct hit for '+label+':\n'+at5.map(r=>r.candidate).join('\n'));
  }
}

const duplicateRows = rowsFor(2.0766740474685811741, 5);
if(!duplicateRows.some(r => String(r.candidate).includes('Σ 1/(n²+1)') && String(r.candidate).includes('Im PolyGamma[0, exp(2πi/4)]'))){
  throw new Error('combined duplicate direct hit should show both labels');
}

console.log('PASS RIES v11.6.4 linear-combo level-5 additions smoke test');
