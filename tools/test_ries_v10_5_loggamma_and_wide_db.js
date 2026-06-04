const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){}, insertAdjacentElement(){}, querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(name){return this.src||''}};}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value=(9n*(997n**7n)-12n).toString(); els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='20'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
function loadAsset(){ vm.runInContext(fs.readFileSync('assets/shortform100k.js','utf8'), sandbox); }
const head={appendChild(script){ if(script.src && /assets\/shortform100k\.js/.test(script.src)){ loadAsset(); if(script.onload) setTimeout(script.onload,0); } return script; }};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}, appendChild:head.appendChild}, head, documentElement:head};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
(async()=>{
  const T=sandbox.__RIES_INTEGER_TEST__;
  const logIds=sandbox.__RIES_LOG_TEST__.logConstants.map(c=>c.id);
  if(!logIds.includes('loggamma16') || logIds.includes('loglog5')) throw new Error('log basis should replace loglog5 with loggamma16: '+logIds.join(','));
  const g=sandbox.__RIES_LOG_TEST__.logConstants.find(c=>c.id==='loggamma16');
  if(!/Γ\(1\/6\)/.test(g.label) || !/Γ\(1\/6\)/.test(g.product)) throw new Error('bad Γ(1/6) label/product');
  const settings=sandbox.readSettings();
  const rows=await T.integerDatabaseRowsResponsive(settings, ()=>{});
  if(!rows.some(r=>/997\^7/.test(r.candidate) && /-12/.test(r.candidate))) throw new Error('wide C*A^B+D row missing: '+rows.map(r=>r.candidate).join(' | '));
  for(const r of rows){ if(!T.integerRowFormulaIsValid(r)) throw new Error('invalid integer row: '+JSON.stringify(r)); }
  const folded=T.simplifyIntegerExpressionDisplay('binom(997,7)+3-1');
  if(folded!=='binom(997,7)+2') throw new Error('tiny offset folding failed: '+folded);
  const powFolded=T.simplifyIntegerExpressionDisplay('8·128^7');
  if(powFolded!=='2^52') throw new Error('pure power simplification failed: '+powFolded);
  console.log('PASS RIES v10.5 log Γ(1/6) and wide integer database test');
})().catch(err=>{ console.error(err); process.exit(1); });
