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
  ['π√2', 4.442882938158366],
  ['π√3', 5.441398092702653],
  ['Σ 1/(n²+1)', 2.0766740474685811741],
  ['Σ 1/(n³+1)', 1.6865033423386238860],
  ['Σ (-1)^n/(n²+1)', 0.63601452749106658148],
  ['Σ (-1)^n/C(2n,n)', 0.62783642361439838444],
  ['Σ 1/C(3n,n)', 1.4143220443218203919],
  ['Σ 1/(n²+n+1)', 1.7981472805626901809],
  ['Σ (-1)^n/(n²+n+1)', 0.76131020400110348639],
  ['Σ 1/(1+2^n)', 1.26449978034844420919]
];

const level4 = api.lowPrecisionLinearComboBasisConstants({level:4});
const level5 = api.lowPrecisionLinearComboBasisConstants({level:5});
if(level4.length !== 107) throw new Error(`expected level 4 basis size 107, got ${level4.length}`);
if(level5.length < 117) throw new Error(`expected level 5 basis size at least 117, got ${level5.length}`);
function hasLabelIncluding(list, needle){ return list.some(c => String(c.label).includes(needle)); }
for(const [label] of additions){
  if(hasLabelIncluding(level4, label)) throw new Error('level-5 constant leaked into level 4 basis: '+label);
  if(!hasLabelIncluding(level5, label)) throw new Error('level-5 constant missing from level 5 basis: '+label);
}

function rowsFor(value, level){
  const raw = String(Number(value).toPrecision(12));
  return api.lowPrecisionLinearComboRows({
    raw, normalizedRaw:raw, target:Number(value), complexTarget:false, level, limit:5,
    modules:{linearCombo:true}, linearComboOptions:{one:true,two:false,three:false},
    moduleLimits:{linearCombo:5}, stageBudgets:{linearComboMs:3000}
  });
}

for(const [label, value] of additions){
  const at4 = rowsFor(value, 4);
  if(at4.some(r => String(r.candidate).includes(label))) throw new Error('level-5-only hit appeared at level 4: '+label);
  const at5 = rowsFor(value, 5);
  if(!at5.some(r => String(r.candidate).includes(label))){
    throw new Error('missing level 5 direct hit for '+label+':\n'+at5.map(r=>r.candidate).join('\n'));
  }
}

console.log('PASS RIES v11.6.2 linear-combo level-5 additions smoke test');
