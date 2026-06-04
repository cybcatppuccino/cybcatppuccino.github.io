// RIES v8.7 integer database simplification / cache / progress smoke test.
// Run from repository root: node tools/test_ries_v8_7_integer_simplify_cache_progress.js
const fs = require('fs');
const vm = require('vm');
let listenerCount = 0;
function fakeEl(id){
  return {
    id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(child){ return child; }, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}
  };
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='4'; els.limit.value='10';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false; els.target.value='2251875390634';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null};
sandbox.document={
  getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){ listenerCount++; },
  body:{contains:()=>true, prepend(){}}
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
function simplify(expr){ return sandbox.simplifyDExprIfBetter(sandbox.makeDExpr(0n, expr, 'test', 5, 3)).s; }
if(simplify('35^7·35+9') !== '35^8+9') throw new Error('Expected 35^7·35+9 to simplify to 35^8+9.');
if(simplify('9^49·27-5') !== '3^101-5') throw new Error('Expected 9^49·27-5 to simplify to 3^101-5.');
if(simplify('16/4·2^35+1') !== '2^37+1') throw new Error('Expected reducible fraction/power product to simplify inside an offset expression.');
const settings=sandbox.readSettings();
const c1=sandbox.getIntegerGlobalCache(settings), c2=sandbox.getIntegerGlobalCache(settings);
if(c1 !== c2) throw new Error('Integer global cache should reuse the same object for the same target/settings.');
(async()=>{
  const progress=[];
  const rows=await sandbox.integerDatabaseRowsResponsive(settings, info=>progress.push(info));
  const candidates=rows.map(r=>r.candidate).join('\n');
  if(!/database: 35\^8\+9/.test(candidates)) throw new Error('Database should display the simplified 35^8+9 form. Got:\n'+candidates);
  if(/35\^7·35\+9/.test(candidates)) throw new Error('Unsimplified 35^7·35+9 should not be displayed.');
  if(!progress.some(p=>p && p.budgetMs && p.label)) throw new Error('Database progress should include phase labels and total budget.');
  const src=fs.readFileSync('ries-script.js','utf8');
  for(const token of ['smallIntegerExhaustiveSearchAsync','integerGlobalCache','Stopping now; the current slice will finish']){
    if(!src.includes(token)) throw new Error(`Missing v8.7 integer responsiveness/cache token: ${token}`);
  }
  console.log('PASS RIES v8.7 integer simplification/cache/progress smoke test');
  process.exit(0);
})().catch(err=>{ console.error(err); process.exit(1); });
