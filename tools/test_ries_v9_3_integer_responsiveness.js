// RIES v9 integer Continue responsiveness / caching / cleanup smoke test.
const fs = require('fs');
const vm = require('vm');
const html = fs.readFileSync('ries.html','utf8');
if(!html.includes('RIES <em>v9.3</em>')) throw new Error('ries.html was not updated to v9.3.');
if(html.includes('Integer inputs first show fast factorization') || html.includes('Decimal inputs now keep')) throw new Error('Verbose release text should not be visible in ries.html results header.');
const readme = fs.readFileSync('README.md','utf8');
if(/RIES v8\.|v7\.3|changelog|CHANGELOG|Integer inputs first show/i.test(readme)) throw new Error('README should stay compact and should not display release-note details.');
let listenerCount = 0;
function fakeEl(id){
  return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,
    classList:{contains(){return false},add(){},remove(){}}, addEventListener(){ listenerCount++; }, setAttribute(){}, appendChild(child){return child}, prepend(){},
    querySelector(){return fakeEl('q')}, querySelectorAll(){return []}, getContext(){return {}}, closest(){return null}, getAttribute(){return ''}};
}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='4'; els.limit.value='10';
els.doEq.checked=true; els.doAlg.checked=true; els.doLog.checked=true; els.allowExternalFactorization.checked=false; els.target.value='32698754';
let now0 = Date.now();
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){ listenerCount++; }, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/lfunctions-l2l4.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
if(typeof sandbox.comparePowToN!=='function') throw new Error('comparePowToN should be visible for smoke testing.');
const tPow=Date.now();
const cmp=sandbox.comparePowToN(90n, 2200n, 96n, 32698754n);
if(![-1,0,1].includes(cmp)) throw new Error('comparePowToN returned invalid sign.');
if(Date.now()-tPow>250) throw new Error('Large rational-power comparison should use log guard and return quickly.');
(async()=>{
  const settings=sandbox.readSettings();
  const ic1=sandbox.getIntegerGlobalCache(settings), ic2=sandbox.getIntegerGlobalCache(settings);
  if(ic1!==ic2) throw new Error('Integer cache object was not reused.');
  const progress=[];
  const t0=Date.now();
  const rows=await sandbox.integerShortformRowsAsync(settings, (partial, info)=>progress.push({partial:partial.length, phase:info.phase||'', elapsed:info.elapsed||0}));
  const elapsed=Date.now()-t0;
  if(elapsed>6500) throw new Error(`32698754 effort 4 shortform smoke test took too long: ${elapsed} ms`);
  if(!progress.length) throw new Error('Shortform should report progress before completing.');
  if(!progress.some(p=>/building DB|direct\/reverse|ratio|rational powers|wide/i.test(p.phase))) throw new Error('Shortform progress did not include detailed phases: '+JSON.stringify(progress.slice(0,8)));
  if(!Array.isArray(rows)) throw new Error('Shortform did not return an array.');
  console.log('PASS RIES v9.3 integer responsiveness/cache/cleanup smoke test');
})().catch(err=>{ console.error(err); process.exit(1); });
