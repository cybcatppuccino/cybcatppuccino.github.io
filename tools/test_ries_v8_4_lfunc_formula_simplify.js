// RIES v8.4 L-function formula simplification smoke test.
// Ensures rational reciprocal matches render as 3/(2·L) instead of π/(2/3·L)-style denominator fractions.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(){},prepend(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}}}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=false; els.doAlg.checked=true; els.doLog.checked=true; els.algHeight.value='1000000000000'; els.algDegree.value='10'; els.algResidualPower.value='2'; els.logHeight.value='400'; els.logSlack.value='2';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
(async()=>{
  const D=sandbox.Decimal; D.set({precision:50});
  const L2=new D('0.25384186085591068433775892335090946104');
  els.target.value=new D(3).div(new D(2).mul(L2)).toSignificantDigits(16).toString();
  const rows=await sandbox.lfuncRowsAsync(sandbox.readSettings(),0);
  const cands=rows.map(r=>String(r.candidate));
  if(!cands.some(c=>/x = 3\/\(2·L\(f,1\)\)/.test(c))) throw new Error('Expected simplified reciprocal rational L-function formula 3/(2·L(f,1)).');
  if(cands.some(c=>/\/\([^)]*\d+\/\d+[^)]*L\(f,1\)/.test(c))) throw new Error('Found denominator containing a fractional scale factor.');
  console.log('PASS RIES v8.4 L-function rational formula simplification smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
