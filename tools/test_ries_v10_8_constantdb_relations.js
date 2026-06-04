// RIES v10.8 regression: constant database relation families.
const fs = require('fs');
const vm = require('vm');
function fakeEl(id){return {id,value:'',checked:true,hidden:false,disabled:false,dataset:{},style:{setProperty(){}},className:'',textContent:'',innerHTML:'',open:false,classList:{contains(){return false},add(){},remove(){}},addEventListener(){},setAttribute(){},appendChild(child){return child},prepend(){},insertAdjacentElement(){},querySelector(){return fakeEl('q')},querySelectorAll(){return []},getContext(){return {}},closest(){return null},getAttribute(){return ''}};}
const ids=['resultBody','resultTools','resultToolsMeta','sortConfidenceBtn','sortDiscoveryBtn','hpPanel','hpContent','numberTools','numberToolsContent','status','commandPreview','paramToggle','parametersPanel','stopBtn','continueBtn','runBtn','target','onlySyms','neverSyms','digits','restrictMode','tolerance','maxAbs','level','shortEffort','limit','doEq','doAlg','doLog','allowExternalFactorization','logHeight','logPrecision','logSlack','algHeight','algDegree','algPrecision','algResidualPower','defaultLogBasis','extraLogBasis'];
const els={}; ids.forEach(id=>els[id]=fakeEl(id));
els.digits.value='0123456789'; els.restrictMode.value='none'; els.maxAbs.value='1e9'; els.level.value='4'; els.shortEffort.value='3'; els.limit.value='10'; els.doEq.checked=true; els.doAlg.checked=false; els.doLog.checked=false; els.logHeight.value='400';
const sandbox={console, performance:{now:()=>Date.now()}, setTimeout, clearTimeout, requestAnimationFrame:(cb)=>setTimeout(()=>cb(Date.now()),0), cancelAnimationFrame(){}, MathJax:null};
sandbox.window=sandbox; sandbox.navigator={clipboard:null, scheduling:{isInputPending:()=>false}};
sandbox.document={getElementById:id=>els[id]||null, querySelectorAll:()=>[], querySelector:()=>fakeEl('qs'), createElement:tag=>fakeEl(tag), addEventListener(){}, body:{contains:()=>true, prepend(){}}};
vm.createContext(sandbox);
vm.runInContext(fs.readFileSync('assets/decimal.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('assets/constantdb300.js','utf8'), sandbox);
vm.runInContext(fs.readFileSync('ries-script.js','utf8'), sandbox);
if(sandbox.RIES_CONSTANT_DB_300_VERSION !== '10.8') throw new Error('constant DB version should be 10.8');
function settingsFor(v, level=4){ els.target.value=String(v); els.level.value=String(level); return sandbox.readSettings(); }
const T=sandbox.__RIES_CONSTDB_TEST__;
let cubic=T.constDbFindPolynomialRatio(Math.cbrt(2),3,16,Date.now()+100);
if(!cubic || cubic.degree!==3 || !/2/.test(String(cubic.coeff))) throw new Error('cubic ratio relation not found for cubert(2): '+JSON.stringify(cubic));
let s=settingsFor(String(1+Math.PI+Math.PI*Math.PI),4);
let rows=T.constantDbRows(s);
if(!rows.some(r=>/quadratic relation in b,1,c,c\^2/.test(r.constantDbCategory||'') && /c = π/.test(r.valueHtml||''))) throw new Error('quadratic-in-c constant DB relation missing: '+rows.map(r=>r.candidate+' '+r.constantDbCategory+' '+r.valueHtml).join(' | '));
s=settingsFor(String(Math.PI+1/Math.PI),4);
rows=T.constantDbRows(s);
if(!rows.some(r=>/reciprocal relation in b,1,c,1\/c/.test(r.constantDbCategory||'') && /c = π/.test(r.valueHtml||''))) throw new Error('reciprocal-in-c constant DB relation missing: '+rows.map(r=>r.candidate+' '+r.constantDbCategory+' '+r.valueHtml).join(' | '));
const piRec=T.constantDbRecords().find(r=>r.label==='π');
const extra=T.constDbExtraSubsetRows(settingsFor(String(Math.PI),5), {kind:'pow1', y:Math.PI, label:'x'}, piRec, Math.PI, 16, Date.now()+300, 1e-12);
if(!extra.length || !extra.some(r=>/5-term LLL subset relation/.test(r.constantDbCategory||''))) throw new Error('level-5 subset LLL relation missing');
console.log('PASS RIES v10.8 constant database relation families');
