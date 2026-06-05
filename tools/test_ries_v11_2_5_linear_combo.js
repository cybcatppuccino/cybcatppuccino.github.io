const fs = require('fs');
const vm = require('vm');

function el(){ return { value:'', checked:false, disabled:false, hidden:false, dataset:{}, className:'', textContent:'', innerHTML:'', open:false, addEventListener(){}, appendChild(){}, closest(){ return null; }, querySelectorAll(){ return []; } }; }
const ids = new Map();
const doc = {
  getElementById(id){ if(!ids.has(id)) ids.set(id, el()); return ids.get(id); },
  addEventListener(){}, querySelector(){ return el(); }, querySelectorAll(){ return []; }, createElement(){ return el(); }, head:el(), body:el(), documentElement:el()
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

if(!fs.readFileSync('ries.html','utf8').includes('RIES <em>v11.3.1</em>')) throw new Error('ries.html version should be v11.3.1.');
const api = context.__RIES_LINEAR_COMBO_TEST__;
if(!api) throw new Error('linear-combo test API missing');
const basis = api.lowPrecisionLinearComboBasisConstants();
if(basis.some(c => c.label === '2π' || c.label === 'π/2' || c.label === 'ζ(2)')) throw new Error('basis pruning did not remove obvious rational dependences');

function run(val, needle){
  const raw = val.toPrecision(15);
  const settings = { raw, normalizedRaw:raw, target:val, complexTarget:false, level:4, limit:5 };
  const t0 = Date.now();
  const rows = api.lowPrecisionLinearComboRows(settings);
  if(rows.length > 5) throw new Error('linear-combo should return at most five candidates, got ' + rows.length);
  if(Date.now() - t0 > 3300) throw new Error('linear-combo search exceeded 3.3s');
  if(!rows.some(r => String(r.candidate).includes(needle))) throw new Error(`missing expected formula containing ${needle}:\n` + rows.map(r=>r.candidate).join('\n'));
}
run(Math.PI, 'π');
run((Math.PI + Math.log(2))/3, '(π + log 2)/3');
run((2*Math.PI + 3*Math.log(2) - 5*Math.sqrt(2))/7, '(2·π + 3·log 2 − 5·√2)/7');
console.log('PASS RIES v11.3.1 low-precision sparse linear-combination regression');
