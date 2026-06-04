// RIES v10 integer substring database and high-effort bounded shortform smoke test.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v10.6.1</em>')) throw new Error('ries.html was not updated to v10.6.1.');
function fakeEl(id){
  return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
    classList:{contains(){return false},add(){},remove(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}};
}
function makeSandbox(target, effort='4'){
  const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','algHeight','algDegree','defaultLogBasis','extraLogBasis'];
  const els={}; ids.forEach(id=>els[id]=fakeEl(id));
  els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value=effort; els.limit.value='12';
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
  let sandbox=makeSandbox('120892581','4');
  let settings=sandbox.readSettings();
  const dbRows=await sandbox.integerDatabaseRowsResponsive(settings,()=>{});
  const sub=dbRows.filter(r=>/database substring/.test(r.candidate||''));
  if(sub.length!==1) throw new Error(`9+ digit substring database should display exactly one result when matching; got ${sub.length}`);
  if(!/2\^80|4\^40|16\^20/.test(sub[0].candidate)) throw new Error('Expected best substring representative from 2^80 family. Got: '+sub[0].candidate);

  sandbox=makeSandbox('9169995354','7');
  settings=sandbox.readSettings();
  const progress=[];
  const t0=Date.now();
  const rows=await sandbox.integerShortformRowsAsync(settings,(partial,info)=>progress.push({n:partial.length, phase:info.phase||'', elapsed:info.elapsed||0}));
  const elapsed=Date.now()-t0;
  const hardBudget=Number(settings._shortformHardBudgetMs || 12000);
  if(elapsed>hardBudget+850) throw new Error(`high-effort shortform exceeded hard budget: ${elapsed} ms > ${hardBudget} ms`);
  if(!progress.some(p=>/final reverse pass|ratio|rational powers|direct\/reverse|building DB/i.test(p.phase))) throw new Error('high-effort shortform did not report phase progress.');
  if(!Array.isArray(rows)) throw new Error('integerShortformRowsAsync did not return an array.');
  const src=fs.readFileSync('ries-script.js','utf8');
  if(!/targetStr\.length<9/.test(src)) throw new Error('substring database must start at 9+ digits.');
  if(!/substring-best\|/.test(src)) throw new Error('substring database must collapse to a single best representative.');
  console.log('PASS RIES v10 integer substring database and bounded high-effort smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
