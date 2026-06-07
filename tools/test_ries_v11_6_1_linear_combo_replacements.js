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

const basis = api.lowPrecisionLinearComboBasisConstants();
const labels = new Set(basis.map(c=>c.label));
const newLabels = [
  'PolyGamma[1, 1/3]','PolyGamma[1, 2/3]','PolyGamma[1, 1/6]','PolyGamma[1, 5/6]',
  'log(log(2))','log(log(3))',
  'PolyGamma[2, 1/5]','PolyGamma[2, 2/5]','PolyGamma[2, 3/5]','PolyGamma[2, 4/5]',
  'PolyGamma[2, 1/8]','PolyGamma[2, 3/8]','PolyGamma[2, 5/8]','PolyGamma[2, 7/8]',
  'π log2','(log2)^2','π(log2)^2','π log3'
];
const oldLabels = [
  'Γ(1/3)','Γ(1/4)','Γ(2/3)','Γ(3/4)',
  '2cos(π/7)','2cos(2π/7)',
  'e^(π/2)','e^(π/√2)','e^(π/√3)','e^(π/√5)','e^(π√2)','e^(π√3)','e^−γ','ζ(6)',
  'Shi(1)','Chi(1)','√φ','√(2+√2)'
];
for(const label of newLabels) if(!labels.has(label)) throw new Error('new constant missing from basis: '+label);
for(const label of oldLabels) if(labels.has(label)) throw new Error('old replaced constant still present: '+label);

function expectFormula(value, raw, needle){
  const settings = {
    raw, normalizedRaw:raw, target:value, complexTarget:false, level:4, limit:5,
    modules:{linearCombo:true}, linearComboOptions:{one:true,two:true,three:true},
    moduleLimits:{linearCombo:5}, stageBudgets:{linearComboMs:3000}
  };
  const rows = api.lowPrecisionLinearComboRows(settings);
  if(!rows.some(r => String(r.candidate).includes(needle))){
    throw new Error('missing expected formula '+needle+':\n'+rows.map(r=>r.candidate).join('\n'));
  }
}

expectFormula(10.095597125427094, '10.0955971254', 'PolyGamma[1, 1/3]');
expectFormula(-0.36651292058166435, '-0.3665129206', 'log(log(2))');
expectFormula(2.177586090303602, '2.1775860903', 'π log2');

console.log('PASS RIES v11.6.1 linear-combo replacement smoke test');
