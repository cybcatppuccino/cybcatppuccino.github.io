const fs = require('fs');
const vm = require('vm');
function fakeEl(id){
  return {
    id, value:'', checked:true, hidden:false, disabled:false, dataset:{}, style:{setProperty(){}}, className:'', textContent:'', innerHTML:'', open:false,
    classList:{contains(){return false}, add(){}, remove(){}},
    addEventListener(){}, setAttribute(){}, appendChild(){}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}
  };
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='9.501501389884367';
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='5';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null, Buffer, atob:(s)=>Buffer.from(s,'base64').toString('binary')};
sandbox.window=sandbox;
sandbox.navigator={clipboard:null};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
for(const f of ['assets/decimal.js','assets/lfunctions-l2l4.js','assets/constantdb300.js','assets/ries-harddb-v11_4_1-filtered.js','ries-script.js']){
  vm.runInContext(fs.readFileSync(f,'utf8'), sandbox, {filename:f});
}
(async()=>{
  const settings={target:9.501501389884367, raw:'9.501501389884367', normalizedRaw:'9.501501389884367', complexTarget:false};
  const rows=await sandbox.window.__RIES_HARDDB_TEST__.hardDbRowsAsync(settings);
  if(!rows.length) throw new Error('expected filtered direct hard DB hit for a simple beta-integral self-match');
  if(!/hard constant database/.test(rows[0].candidate)) throw new Error('bad hard DB candidate text');
  if(!/Euler beta integral/.test(rows[0].valueHtml)) throw new Error('metadata/formula not decoded');
  console.log('OK filtered direct harddb runtime:', rows[0].candidate, rows[0].errText);
})().catch(e=>{ console.error(e); process.exit(1); });
