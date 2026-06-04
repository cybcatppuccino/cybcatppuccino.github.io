// RIES v9.6 integer substring DB and bounded high-effort smoke test.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v9.6</em>')) throw new Error('ries.html was not updated to v9.6.');
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
  let sandbox=makeSandbox('1208925819','4');
  let t0=Date.now();
  let rows=await sandbox.integerDatabaseRowsResponsive(sandbox.readSettings(),()=>{});
  let elapsed=Date.now()-t0;
  if(elapsed>5200) throw new Error(`10-digit substring database should be bounded; took ${elapsed} ms`);
  let sub=rows.filter(r=>/database substring/.test(r.candidate||''));
  if(sub.length!==1) throw new Error('Substring database should expose at most one result for >=10 digits; got '+sub.length);
  if(!/2\^80/.test(sub[0].candidate||'')) throw new Error('10-digit substring database should find the 2^80 family, got: '+(sub[0].candidate||''));

  sandbox=makeSandbox('1208925819614629','4');
  rows=await sandbox.integerDatabaseRowsResponsive(sandbox.readSettings(),()=>{});
  sub=rows.filter(r=>/database substring/.test(r.candidate||''));
  if(sub.length>1) throw new Error('16+ digit substring database should also expose at most one result; got '+sub.length);

  sandbox=makeSandbox('9169995354','7');
  const settings=sandbox.readSettings();
  const progress=[];
  t0=Date.now();
  rows=await sandbox.integerShortformRowsAsync(settings,(partial,info)=>progress.push({phase:info.phase||'', elapsed:info.elapsed||0, n:partial.length}));
  elapsed=Date.now()-t0;
  const budget=settings._shortformBudgetMs || 7600;
  if(elapsed>budget*1.65+450) throw new Error(`High-effort integer shortform exceeded budget envelope: ${elapsed} ms for budget ${budget}`);
  if(!progress.some(p=>/final reverse pass|ratio exact pass|rational powers|building DB/.test(p.phase))) throw new Error('High-effort shortform should keep reporting phases.');
  if(!Array.isArray(rows)) throw new Error('integerShortformRowsAsync did not return an array.');
  console.log('PASS RIES v9.6 integer substring and bounded high-effort smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
