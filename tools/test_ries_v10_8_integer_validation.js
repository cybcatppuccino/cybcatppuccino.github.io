const fs = require('fs');
const vm = require('vm');
function fakeEl(id){
  return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
    classList:{contains(){return false},add(){},remove(){}}, addEventListener(){}, setAttribute(){}, appendChild(child){return child}, prepend(){}, insertAdjacentElement(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(name){return this.src||''}};
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.target.value='768'; els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='20'; els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false;
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
  for (const bad of ['48^2','96^2','4^5·3','2^8·6']) {
    if (T.displayExprMatchesTarget(bad, 768n)) throw new Error('bad integer formula validated: '+bad);
  }
  for (const good of ['4^4·3','2^8·3','8·96','32·4!','floor(((5!/7)^7)/7!)']) {
    const target = good.startsWith('floor') ? 86328n : 768n;
    if (!T.displayExprMatchesTarget(good, target)) throw new Error('valid integer formula rejected: '+good);
  }
  await sandbox.ensureShortformDbLoaded();
  if (!sandbox.RIES_SHORTFORM_100K_PACKED || sandbox.RIES_SHORTFORM_100K_PACKED.version!=='10.8') throw new Error('packed DB version is not 10.8');
  const settings=sandbox.readSettings();
  const staticRows=sandbox.__RIES_INTEGER_TEST__.staticShortformRows(settings);
  const shortRows=await sandbox.__RIES_INTEGER_TEST__.integerShortformRowsAsync(settings);
  const all=[...staticRows, ...shortRows];
  if(!all.length) throw new Error('no integer rows generated for 768');
  for(const r of all){
    if(!T.integerRowFormulaIsValid(r)) throw new Error('invalid displayed row: '+JSON.stringify(r));
    if(/structured product:\s*(48\^2|96\^2|4\^5·3|2\^8·6)/.test(r.candidate)) throw new Error('known bad structured-product row survived: '+r.candidate);
    if(r.latex && /undefined|null|NaN/.test(r.latex)) throw new Error('bad LaTeX payload: '+r.latex);
  }
  console.log('PASS RIES v10.8 integer validation and 768 regression test');
})().catch(err=>{ console.error(err); process.exit(1); });
