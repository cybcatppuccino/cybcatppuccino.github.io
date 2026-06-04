// RIES v9.6 integer responsiveness, substring DB, and no precision-controls smoke test.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v9.6</em>')) throw new Error('ries.html was not updated to v9.6.');
for(const forbidden of ['id="tolerance"','id="algPrecision"','id="algResidualPower"','id="logPrecision"','id="logSlack"','Working digits','Relation digits','Residual slack','Max error']){
  if(html.includes(forbidden)) throw new Error('Precision/error parameter control should not be visible: '+forbidden);
}
function fakeEl(id){
  return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
    classList:{contains(){return false},add(){},remove(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}};
}
function makeSandbox(target, effort='4'){
  const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','algHeight','algDegree','defaultLogBasis','extraLogBasis'];
  const els={}; ids.forEach(id=>els[id]=fakeEl(id));
  els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value=effort; els.limit.value='10';
  els.target.value=target; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
  const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
  sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
  sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
  vm.createContext(sandbox);
  vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
  vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
  vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
  return sandbox;
}
(async()=>{
  let sandbox=makeSandbox('9169995354','4');
  let settings=sandbox.readSettings();
  if(settings.tol!==Infinity) throw new Error('Manual max-error/tolerance should be disabled; typed precision is authoritative.');
  const progress=[];
  const t0=Date.now();
  const rows=await sandbox.integerShortformRowsAsync(settings,(partial,info)=>progress.push({n:partial.length, phase:info.phase||'', elapsed:info.elapsed||0}));
  const elapsed=Date.now()-t0;
  if(elapsed>5600) throw new Error(`9169995354 effort 4 shortform should terminate quickly; took ${elapsed} ms`);
  if(!progress.length || !progress.some(p=>/building DB|direct\/reverse|ratio|rational powers|wide/i.test(p.phase))) throw new Error('Integer shortform did not report detailed progress.');
  if(!Array.isArray(rows)) throw new Error('integerShortformRowsAsync did not return an array.');

  sandbox=makeSandbox('1208925819614629','4');
  const dbRows=await sandbox.integerDatabaseRowsResponsive(sandbox.readSettings(),()=>{});
  const sub=dbRows.filter(r=>/database substring/.test(r.candidate||''));
  if(!sub.some(r=>/2\^80|4\^40|16\^20/.test(r.candidate))) throw new Error('16+ digit substring database did not find the 2^80 family.');
  console.log('PASS RIES v9.6 integer responsiveness, substring DB, and no precision-controls smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
