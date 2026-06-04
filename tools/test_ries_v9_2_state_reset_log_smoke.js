// RIES v9.2 target-change state reset and log|c| smoke/stress test.
const fs = require('fs');
const vm = require('vm');
const {performance} = require('perf_hooks');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v9.2</em>')) throw new Error('ries.html was not updated to v9.2.');
for(const noisy of ['Integer inputs first show fast factorization','Decimal inputs now keep','AI minesweeper']){
  if(html.includes(noisy)) throw new Error('Visible RIES page should remain compact; found: '+noisy);
}
const Decimal = require('../assets/decimal.js');
const elems = new Map();
function fakeEl(id){ return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
  classList:{contains(){return false},add(){},remove(){},toggle(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){},
  querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}}; }
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
ids.forEach(id=>elems.set(id,fakeEl(id)));
Object.assign(elems.get('digits'), {value:'0123456789'});
Object.assign(elems.get('restrictMode'), {value:'none'});
Object.assign(elems.get('maxAbs'), {value:'1e9'});
Object.assign(elems.get('level'), {value:'4'});
Object.assign(elems.get('shortEffort'), {value:'4'});
Object.assign(elems.get('limit'), {value:'10'});
Object.assign(elems.get('logHeight'), {value:'400'});
Object.assign(elems.get('logPrecision'), {value:''});
Object.assign(elems.get('logSlack'), {value:'2'});
['doEq','doAlg','doLog'].forEach(id=>elems.get(id).checked=true);
elems.get('allowExternalFactorization').checked=false;
const sandbox={console, performance, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(performance.now()),0), cancelAnimationFrame(){}, MathJax:null, Decimal};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.__logChecked=[];
sandbox.document={
  getElementById:id=>elems.get(id)||null,
  querySelectorAll:sel=> sel==='[data-logconst]:checked' ? sandbox.__logChecked.map(id=>({dataset:{logconst:id}})) : [],
  querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}
};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
const logTest=sandbox.__RIES_LOG_TEST__;
if(!logTest || typeof logTest.resetSearchFrameworkForInputChange!=='function') throw new Error('v9.2 reset/log test exports missing.');
sandbox.__logChecked=logTest.logConstants.filter(c=>c.default).map(c=>c.id);
function decimalSettings(raw, level){
  elems.get('target').value=raw; elems.get('level').value=String(level);
  const parsed=sandbox.parseDecimalComplex(raw);
  return {raw, normalizedRaw:raw, parsedComplex:parsed, complexTarget:false, target:sandbox.rationalToNumber(parsed.re), limit:10, level};
}
let rows=logTest.logRelationRows(6*Math.PI**5, decimalSettings(String(6*Math.PI**5),4));
if(!rows.some(r=>/2 \* 3 \* π\^\(5\)/.test(r.candidate))) throw new Error('log|c| should find sparse 6*pi^5 at default level.');
if(!/2 \* 3 \* π\^\(5\)/.test(rows[0].candidate)) throw new Error('sparse 6*pi^5 should outrank noisy LLL log relations.');
const gamma13=2.6789385347077476336556;
rows=logTest.logRelationRows(gamma13/Math.log(Math.PI), decimalSettings(String(gamma13/Math.log(Math.PI)),5));
if(!rows.some(r=>/Γ\(1\/3\)/.test(r.candidate) && /log\(π\)\^\(-1\)/.test(r.candidate))) throw new Error('first Continue should find Gamma(1/3)/log(pi) in log|c|.');
logTest.solveRunCache.set('dummy',{full:new Map([['x',[1]]])});
logTest.integerGlobalCache.set('dummy',{factor:new Map([['x',[1]]]), db:new Map(), short:new Map()});
logTest.lfuncProgressCache.set('dummy',{rows:[1]});
logTest.resetSearchFrameworkForInputChange();
if(logTest.solveRunCache.size || logTest.integerGlobalCache.size || logTest.lfuncProgressCache.size) throw new Error('target input reset must clear per-input caches.');
(async()=>{
  const samples=['42','12345678','32698754'];
  for(const raw of samples){
    elems.get('target').value=raw; elems.get('shortEffort').value= raw==='32698754' ? '4' : '2';
    const settings=sandbox.readSettings();
    const progress=[]; const t0=performance.now();
    const rows=await sandbox.integerShortformRowsAsync(settings, (partial, info)=>progress.push(info.phase||''));
    const elapsed=performance.now()-t0;
    if(elapsed>7000) throw new Error(`integer smoke ${raw} took too long: ${elapsed.toFixed(0)} ms`);
    if(!Array.isArray(rows)) throw new Error('integer shortform did not return rows for '+raw);
    if(raw==='32698754' && !progress.length) throw new Error('32698754 effort 4 should report progress.');
    logTest.resetSearchFrameworkForInputChange();
  }
  console.log('PASS RIES v9.2 state reset/log/integer smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
